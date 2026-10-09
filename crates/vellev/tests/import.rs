//! Copying a Weaviate into a tenant. Runs only when `VELLEV_WEAVIATE_URL`
//! names a Weaviate to load, such as the one in `docker-compose.dev.yml`.
//!
//! Where a test needs Weaviate or another writer to do something at a
//! given moment of an import, or either side to fail, the import reaches
//! Weaviate and the node each through a [`Proxy`] that does so around a
//! chosen request. One test restarts Weaviate itself, under a catch-up,
//! and runs only when `VELLEV_WEAVIATE_CONTAINER` names the Docker
//! container it runs in.

mod common;

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::response::IntoResponse;
use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Map, Value};
use vellev::import::{self, Deletes, Mode, Options, Outcome, Report, Source};
use vellev_api::{Changes, Content, Keyword, Page};
use vellev_client::{Client, Options as ClientOptions, Records, Tenant};
use vellev_core::{CollectionName, ObjectId, TenantId};

const OBJECTS: u64 = 1_200;

/// Ids from one end of their range to the other, in no order of `n`, so
/// that walking Weaviate and vellev side by side is put to the test.
fn uuid(n: u64) -> String {
    match n % 4 {
        0 => id(n),
        1 => format!("7fffffff-ffff-4fff-bfff-{n:012}"),
        2 => format!("80000000-0000-4000-8000-{n:012}"),
        _ => format!("ffffffff-ffff-ffff-ffff-{n:012}"),
    }
}

fn oid(n: u64) -> ObjectId {
    serde_json::from_value(json!(uuid(n))).unwrap()
}

/// A note. Its score has as many digits as a double holds, which a
/// reader that rounds twice gets wrong.
fn object(n: u64) -> Value {
    json!({
        "title": format!("note {n} about {}", ["foxes", "hens", "owls"][n as usize % 3]),
        "tags": [format!("t{}", n % 4)],
        "n": n,
        "score": n as f64 / 7.0,
        "seen": n.is_multiple_of(2),
        "at": format!("2026-05-{:02}T09:00:00Z", 1 + n % 28),
    })
}

fn vector(n: u64) -> Vec<f32> {
    vec![1.0, (n % 7) as f32, (n % 11) as f32 / 3.0, 0.5]
}

/// A value with every number read as vellev returns a `number`.
fn plain(value: &Value) -> Value {
    match value {
        Value::Number(number) => json!(number.as_f64()),
        Value::Array(items) => items.iter().map(plain).collect(),
        Value::Object(fields) => fields
            .iter()
            .map(|(name, value)| (name.clone(), plain(value)))
            .collect::<Map<_, _>>()
            .into(),
        other => other.clone(),
    }
}

fn map(value: Value) -> Map<String, Value> {
    value.as_object().unwrap().clone()
}

/// Objects by id: their properties and vector.
type Held = BTreeMap<ObjectId, (Value, Option<Vec<f32>>)>;

#[derive(Clone)]
struct Weaviate {
    url: String,
    http: reqwest::Client,
}

impl Weaviate {
    async fn send(&self, method: Method, path: &str, body: Option<Value>) -> Value {
        let request = self.http.request(method, format!("{}{path}", self.url));
        let request = match body {
            Some(body) => request.json(&body),
            None => request,
        };
        let answer = request.send().await.unwrap();
        let status = answer.status();
        let text = answer.text().await.unwrap();
        assert!(status.is_success(), "{path}: {status} {text}");
        serde_json::from_str(&text).unwrap_or(Value::Null)
    }

    fn schema(class: &str) -> Value {
        json!({
            "class": class,
            "vectorizer": "none",
            "vectorIndexConfig": { "distance": "cosine" },
            "invertedIndexConfig": { "bm25": { "k1": 1.1, "b": 0.6 } },
            "properties": [
                { "name": "title", "dataType": ["text"], "tokenization": "word", "indexFilterable": false },
                { "name": "tags", "dataType": ["text[]"], "tokenization": "field", "indexSearchable": false },
                { "name": "n", "dataType": ["int"] },
                { "name": "score", "dataType": ["number"] },
                { "name": "seen", "dataType": ["boolean"] },
                { "name": "at", "dataType": ["date"] },
            ],
        })
    }

    async fn drop(&self, class: &str) {
        let _ = self
            .http
            .delete(format!("{}/v1/schema/{class}", self.url))
            .send()
            .await;
    }

    /// Make the class, empty.
    async fn make(&self, class: &str) {
        let schema = Self::schema(class);
        self.send(Method::POST, "/v1/schema", Some(schema)).await;
    }

    /// A class `class` holding notes `0..objects`.
    async fn load(url: &str, class: &str, objects: u64) -> Self {
        let weaviate = Self {
            url: url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        };
        weaviate.drop(class).await;
        weaviate.make(class).await;
        weaviate.add(class, 0..objects).await;
        weaviate
    }

    async fn add(&self, class: &str, notes: std::ops::Range<u64>) {
        for chunk in notes.collect::<Vec<_>>().chunks(300) {
            let objects: Vec<Value> = chunk
                .iter()
                .map(|n| json!({ "class": class, "id": uuid(*n), "properties": object(*n), "vector": vector(*n) }))
                .collect();
            self.send(
                Method::POST,
                "/v1/batch/objects",
                Some(json!({ "objects": objects })),
            )
            .await;
        }
    }

    /// Store note `n` whole, with these properties.
    async fn put(&self, class: &str, n: u64, properties: Value) {
        let path = format!("/v1/objects/{class}/{}", uuid(n));
        let body =
            json!({ "class": class, "id": uuid(n), "properties": properties, "vector": vector(n) });
        self.send(Method::PUT, &path, Some(body)).await;
    }

    /// Store a new note `n` with these properties and this vector.
    async fn post(&self, class: &str, n: u64, properties: Value, vector: Vec<f32>) {
        let body =
            json!({ "class": class, "id": uuid(n), "properties": properties, "vector": vector });
        self.send(Method::POST, "/v1/objects", Some(body)).await;
    }

    async fn delete(&self, class: &str, n: u64) {
        let path = format!("/v1/objects/{class}/{}", uuid(n));
        self.send(Method::DELETE, &path, None).await;
    }

    /// Delete the object if it is there.
    async fn forget(&self, class: &str, id: &str) {
        let path = format!("{}/v1/objects/{class}/{id}", self.url);
        self.http.delete(path).send().await.unwrap();
    }

    /// A page of the class as Weaviate lists it to a catch-up.
    async fn listed(&self, class: &str) -> Value {
        let path = format!("/v1/objects?class={class}&limit=100");
        self.send(Method::GET, &path, None).await
    }

    /// Every object of the class.
    async fn held(&self, class: &str) -> Held {
        let mut held = Held::new();
        let mut after = String::new();
        loop {
            let path = format!("/v1/objects?class={class}&limit=100&include=vector{after}");
            let page = self.send(Method::GET, &path, None).await;
            let objects = page["objects"].as_array().unwrap();
            let Some(last) = objects.last() else {
                return held;
            };
            after = format!("&after={}", last["id"].as_str().unwrap());
            for object in objects {
                let id = serde_json::from_value(object["id"].clone()).unwrap();
                let vector = serde_json::from_value(object["vector"].clone()).unwrap();
                held.insert(id, (plain(&object["properties"]), vector));
            }
        }
    }
}

/// Every object of the collection, less the time of creation the import
/// gave each.
async fn held(records: &Records) -> Held {
    let mut held = Held::new();
    let mut page = Page::new(MAX_LIMIT as u32);
    page.vectors = true;
    loop {
        let listed = records.list(&page).await.unwrap();
        for mut stored in listed.objects {
            stored.properties.remove("createdAt");
            let properties = plain(&Value::Object(stored.properties));
            held.insert(stored.id, (properties, stored.vector));
        }
        match listed.next {
            Some(next) => page.after = Some(next),
            None => return held,
        }
    }
}

type Doing = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Something to do around a request, given the request's method and
/// path.
struct Rule {
    /// The request has every one of these in its method and path, and
    /// none of those that begin with `!`.
    has: Vec<String>,
    /// How many such requests pass untouched first.
    skip: usize,
    act: Box<dyn FnOnce(String) -> Doing + Send>,
    /// Whether to act once the request has been answered, before the
    /// answer is passed back, rather than before it is passed on.
    after: bool,
    /// What to answer in place of passing it on.
    answer: Option<(StatusCode, Value)>,
}

/// Passes requests on to a Weaviate or a node, doing what its rules say
/// around them.
#[derive(Clone)]
struct Proxy {
    url: String,
    to: String,
    http: reqwest::Client,
    rules: Arc<Mutex<Vec<Rule>>>,
    /// Every request's method and path, and when it came.
    asked: Arc<Mutex<Vec<(std::time::Instant, String)>>>,
}

impl Proxy {
    async fn to(url: &str) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy = Self {
            url: format!("http://{}", listener.local_addr().unwrap()),
            to: url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
            rules: Arc::default(),
            asked: Arc::default(),
        };
        let router = axum::Router::new()
            .fallback(Self::relay)
            .with_state(proxy.clone());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        proxy
    }

    /// Before the request after the first `skip` that have all of `has`
    /// in their method and path is passed on, do `act`.
    fn before<F: Future<Output = ()> + Send + 'static>(
        &self,
        skip: usize,
        has: &[&str],
        act: impl FnOnce(String) -> F + Send + 'static,
    ) {
        self.rule(skip, has, Box::new(|line| Box::pin(act(line))), false, None);
    }

    /// Once such a request has been answered, and before whoever made
    /// it has the answer, do `act`.
    fn after<F: Future<Output = ()> + Send + 'static>(
        &self,
        skip: usize,
        has: &[&str],
        act: impl FnOnce() -> F + Send + 'static,
    ) {
        self.rule(skip, has, Box::new(|_| Box::pin(act())), true, None);
    }

    /// Answer such a request with `status` and `body`, once.
    fn answer(&self, skip: usize, has: &[&str], status: StatusCode, body: Value) {
        let nothing = Box::new(|_| Box::pin(async {}) as Doing);
        self.rule(skip, has, nothing, false, Some((status, body)));
    }

    fn rule(
        &self,
        skip: usize,
        has: &[&str],
        act: Box<dyn FnOnce(String) -> Doing + Send>,
        after: bool,
        answer: Option<(StatusCode, Value)>,
    ) {
        self.rules.lock().unwrap().push(Rule {
            has: has.iter().map(|has| has.to_string()).collect(),
            skip,
            act,
            after,
            answer,
        });
    }

    /// Whether every rule has been used.
    fn spent(&self) -> bool {
        self.rules.lock().unwrap().is_empty()
    }

    /// Forget what was asked so far.
    fn listen(&self) {
        self.asked.lock().unwrap().clear();
    }

    /// The requests asked since, in order, that have `has` in their
    /// method and path, and when each came.
    fn asked(&self, has: &str) -> Vec<(std::time::Instant, String)> {
        let asked = self.asked.lock().unwrap();
        let fits = asked.iter().filter(|(_, line)| line.contains(has));
        fits.cloned().collect()
    }

    async fn relay(
        axum::extract::State(proxy): axum::extract::State<Proxy>,
        request: axum::extract::Request,
    ) -> axum::response::Response {
        let (parts, body) = request.into_parts();
        let path = parts.uri.path_and_query().map_or("", |path| path.as_str());
        let line = format!("{} {path}", parts.method);
        let now = std::time::Instant::now();
        proxy.asked.lock().unwrap().push((now, line.clone()));
        let due = {
            let mut rules = proxy.rules.lock().unwrap();
            let fits = |has: &String| match has.strip_prefix('!') {
                Some(lacked) => !line.contains(lacked),
                None => line.contains(has.as_str()),
            };
            let matching = rules.iter().position(|rule| rule.has.iter().all(fits));
            match matching {
                Some(at) if rules[at].skip == 0 => Some(rules.remove(at)),
                Some(at) => {
                    rules[at].skip -= 1;
                    None
                }
                None => None,
            }
        };
        let mut later = None;
        match due {
            Some(rule) if rule.after => later = Some(rule.act),
            Some(rule) => {
                (rule.act)(line.clone()).await;
                if let Some((status, body)) = rule.answer {
                    return (status, axum::Json(body)).into_response();
                }
            }
            None => {}
        }
        let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let mut passed = proxy
            .http
            .request(parts.method, format!("{}{path}", proxy.to))
            .body(body);
        for (name, value) in &parts.headers {
            if name != reqwest::header::HOST && name != reqwest::header::CONTENT_LENGTH {
                passed = passed.header(name, value);
            }
        }
        let answer = passed.send().await.unwrap();
        let status = answer.status();
        let kind = answer.headers().get(reqwest::header::CONTENT_TYPE).cloned();
        let bytes = answer.bytes().await.unwrap();
        if let Some(act) = later {
            act(line).await;
        }
        let mut answer = (status, bytes).into_response();
        if let Some(kind) = kind {
            answer
                .headers_mut()
                .insert(reqwest::header::CONTENT_TYPE, kind);
        }
        answer
    }
}

/// The id a request for a page of objects names as the one to start
/// after.
fn after(line: &str) -> &str {
    let (_, after) = line.split_once("after=").expect("a page after another");
    after.split('&').next().unwrap()
}

/// How the tests import: a page of 50, no margin for the clocks, since
/// Weaviate runs on this machine's and a margin would fetch again
/// whatever was written just before a catch-up, and no bound on what a
/// run may delete of classes this small.
fn options(state: &std::path::Path) -> Options {
    Options {
        created_as: Some("createdAt".into()),
        page: 50,
        batch_bytes: MAX_BODY / 2,
        skew: Duration::ZERO,
        deletes: Deletes::UpTo {
            count: 1_000,
            fraction: 1.0,
        },
        confirm: Duration::from_millis(20),
        patience: Duration::from_secs(10),
        ..Options::new(state.to_path_buf())
    }
}

fn source(url: &str) -> Source {
    Source {
        url: url.to_string(),
        key: None,
        tenant: None,
        name: None,
    }
}

async fn tenant(url: &str, name: &str) -> Tenant {
    let options = ClientOptions::default().batch_bytes(MAX_BODY / 2);
    let client = Client::connect(&[url], KEY, options).unwrap();
    client.tenant(&TenantId::new(name).unwrap())
}

/// What a run did and found: how many objects it stored, changed and
/// deleted, how many the ledger keeps as another writer's and how many
/// it left behind; and how many Weaviate, the ledger and the collection
/// hold.
type Counts = ([u64; 5], [u64; 3]);

fn counts(report: &Report, class: &str) -> Counts {
    let copied = &report.classes[class];
    (
        [
            copied.stored,
            copied.changed,
            copied.deleted,
            copied.kept,
            copied.left,
        ],
        [copied.weaviate, copied.imported, copied.vellev],
    )
}

/// One class of a Weaviate and the tenant it is imported into.
struct Importing {
    class: &'static str,
    weaviate: Weaviate,
    /// Between the import and Weaviate, and between it and the node.
    proxy: Proxy,
    front: Proxy,
    // Held so that the node goes on serving and the state file stays.
    _served: Served,
    dir: tempfile::TempDir,
    tenant: Tenant,
    records: Records,
    options: Options,
}

impl Importing {
    /// A class holding notes `0..objects`, and an empty tenant; nothing
    /// if no Weaviate is named.
    async fn new(class: &'static str, objects: u64) -> Option<Self> {
        let url = std::env::var("VELLEV_WEAVIATE_URL").ok()?;
        let weaviate = Weaviate::load(&url, class, objects).await;
        let served = Served::new().await;
        let front = Proxy::to(&served.url).await;
        let tenant = tenant(&front.url, &class.to_lowercase()).await;
        let dir = tempfile::tempdir().unwrap();
        let options = Options {
            classes: vec![class.into()],
            ..options(&dir.path().join("state.json"))
        };
        Some(Self {
            class,
            proxy: Proxy::to(&weaviate.url).await,
            front,
            weaviate,
            _served: served,
            records: tenant.collection(&CollectionName::new(class).unwrap()),
            tenant,
            dir,
            options,
        })
    }

    /// Import through the proxies.
    async fn run(&self) -> eyre::Result<Report> {
        self.run_as(&self.options).await
    }

    async fn run_as(&self, options: &Options) -> eyre::Result<Report> {
        import::weaviate(&source(&self.proxy.url), &self.tenant, options).await
    }

    async fn counts(&self) -> Counts {
        counts(&self.run().await.unwrap(), self.class)
    }

    fn ledger(&self) -> Records {
        let name = format!("vellev_imported_{}", self.class);
        self.tenant.collection(&CollectionName::new(name).unwrap())
    }

    async fn get(&self, n: u64) -> Option<vellev_api::Stored> {
        self.records.get(oid(n)).await.unwrap()
    }

    /// A page of objects of this class after another, as the import asks
    /// Weaviate for one: with vectors on its first walk, without to
    /// catch up.
    fn page(&self, vectors: bool) -> Vec<String> {
        let page = format!("GET /v1/objects?class={}&", self.class);
        let vectors = if vectors { "" } else { "!" };
        vec![page, "after=".into(), format!("{vectors}include=vector")]
    }

    /// The first page of a catch-up.
    fn first(&self) -> Vec<String> {
        let page = format!("GET /v1/objects?class={}&", self.class);
        vec![page, "!after=".into(), "!include=vector".into()]
    }

    /// Weaviate asked for note `n` by itself.
    fn object(&self, n: u64) -> String {
        format!("GET /v1/objects/{}/{}", self.class, uuid(n))
    }

    /// The node asked for note `n` of the collection by itself.
    fn read(&self, n: u64) -> String {
        format!("/collections/{}/objects/{}", self.class, uuid(n))
    }

    async fn assert_same(&self) {
        let weaviate = self.weaviate.held(self.class).await;
        let vellev = held(&self.records).await;
        let ids = |held: &Held| held.keys().copied().collect::<Vec<_>>();
        assert_eq!(ids(&weaviate), ids(&vellev));
        for (id, object) in &weaviate {
            assert_eq!(object, &vellev[id], "{id}");
        }
    }
}

fn strs(owned: &[String]) -> Vec<&str> {
    owned.iter().map(String::as_str).collect()
}

fn changed(n: u64) -> Value {
    json!({ "title": format!("note {n} changed"), "n": 1_000_000 + n, "score": n as f64 / 3.0 })
}

fn foreign(n: u64) -> Content {
    let mut content = Content::new(map(json!({ "title": format!("another writer's {n}") })));
    content.vector = Some(vec![0.0, 0.0, 0.0, 1.0]);
    content
}

/// Note `n` exactly as Weaviate holds it, where no time of creation is
/// kept.
fn mirrored(n: u64) -> Content {
    let mut content = Content::new(map(object(n)));
    content.vector = Some(vector(n));
    content
}

#[tokio::test]
async fn a_weaviate_class_is_copied_whole_with_its_ids_vectors_and_settings() {
    let Some(importing) = Importing::new("VellevImportWhole", OBJECTS).await else {
        return;
    };
    let report = importing.run().await.unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled);
    assert_eq!(
        counts(&report, importing.class),
        ([OBJECTS, 0, 0, 0, 0], [OBJECTS; 3])
    );

    let records = &importing.records;
    let schema = records.schema().await.unwrap().unwrap();
    assert_eq!((schema.bm25.k1, schema.bm25.b), (1.1, 0.6));
    assert_eq!(schema.embedding.dimensions, 4);
    let property = |name: &str| schema.property(name).unwrap().clone();
    assert!(property("title").searchable && !property("title").filterable);
    assert!(property("tags").filterable && !property("tags").searchable);
    for n in [0, 1, 599, OBJECTS - 1] {
        let stored = records.get(oid(n)).await.unwrap().unwrap();
        let mut expected = object(n);
        let properties = stored.properties.clone();
        expected["createdAt"] = properties["createdAt"].clone();
        assert_eq!(Value::Object(properties), expected, "{n}");
        assert_eq!(stored.vector, Some(vector(n)), "{n}");
    }
    importing.assert_same().await;
    let found = records.keyword(&Keyword::new("owls", 5)).await.unwrap();
    assert_eq!(found.results.len(), 5);

    assert_eq!(
        importing.counts().await,
        ([0; 5], [OBJECTS; 3]),
        "a finished import, caught up, has nothing more to do"
    );
}

#[tokio::test]
async fn an_import_stopped_part_way_resumes_and_catches_up_with_what_changed_meanwhile() {
    let Some(importing) = Importing::new("VellevImportResumed", OBJECTS / 2).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();

    // Weaviate goes on taking writes: to objects already copied, and both
    // sides of where the import, stopped as if part way, had reached.
    weaviate.add(class, OBJECTS / 2..OBJECTS).await;
    weaviate.put(class, 0, changed(0)).await;
    weaviate.delete(class, 1).await;
    let state = &importing.options.state;
    let mut progress: Value = serde_json::from_slice(&std::fs::read(state).unwrap()).unwrap();
    let last = (0..OBJECTS / 2).map(oid).max().unwrap();
    assert_eq!(progress["classes"][class]["after"], json!(last));
    progress["classes"][class]["done"] = json!(false);
    std::fs::write(state, serde_json::to_vec(&progress).unwrap()).unwrap();

    let report = importing.run().await.unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled);
    assert_eq!(
        counts(&report, class),
        ([OBJECTS / 2, 1, 1, 0, 0], [OBJECTS - 1; 3])
    );
    let changed = importing.get(0).await.unwrap();
    assert_eq!(changed.properties["n"], json!(1_000_000));
    assert!(importing.get(1).await.is_none());
    importing.assert_same().await;

    // The page the import stopped after is asked for again by one whose
    // state was not written: what it stored is not stored twice.
    progress["classes"][class]["after"] = Value::Null;
    std::fs::write(state, serde_json::to_vec(&progress).unwrap()).unwrap();
    assert_eq!(
        importing.counts().await,
        ([0; 5], [OBJECTS - 1; 3]),
        "every page stored before is passed over"
    );
}

#[tokio::test]
async fn a_named_vector_and_a_tenant_of_a_multi_tenant_class_are_copied_beside_a_plain_class() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let weaviate = Weaviate::load(&url, "VellevImportPlain", 10).await;
    for class in ["VellevImportNamed", "VellevImportTenanted"] {
        weaviate.drop(class).await;
    }
    let named = json!({
        "class": "VellevImportNamed",
        "vectorConfig": { "default": {
            "vectorizer": { "none": {} },
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": { "distance": "dot" },
        } },
        "properties": [{ "name": "title", "dataType": ["text"] }],
    });
    let tenanted = json!({
        "class": "VellevImportTenanted",
        "vectorizer": "none",
        "multiTenancyConfig": { "enabled": true },
        "properties": [{ "name": "title", "dataType": ["text"] }],
    });
    for schema in [named, tenanted] {
        weaviate
            .send(Method::POST, "/v1/schema", Some(schema))
            .await;
    }
    weaviate
        .send(
            Method::POST,
            "/v1/schema/VellevImportTenanted/tenants",
            Some(json!([{ "name": "t1" }, { "name": "t2" }])),
        )
        .await;
    for n in 0..7 {
        let named = json!({ "class": "VellevImportNamed", "id": uuid(n), "properties": { "title": "named" }, "vectors": { "default": vector(n) } });
        weaviate
            .send(Method::POST, "/v1/objects", Some(named))
            .await;
        for (tenant, from) in [("t1", 0), ("t2", 100)] {
            let tenanted = json!({ "class": "VellevImportTenanted", "tenant": tenant, "id": uuid(from + n), "properties": { "title": tenant }, "vector": vector(n) });
            weaviate
                .send(Method::POST, "/v1/objects", Some(tenanted))
                .await;
        }
    }

    let served = Served::new().await;
    let tenant = tenant(&served.url, "mixed").await;
    let dir = tempfile::tempdir().unwrap();
    let options = Options {
        classes: vec![
            "VellevImportPlain".into(),
            "VellevImportNamed".into(),
            "VellevImportTenanted".into(),
        ],
        ..options(&dir.path().join("state.json"))
    };
    let untenanted = source(&url);
    let refused = import::weaviate(&untenanted, &tenant, &options)
        .await
        .unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("say which Weaviate tenant to copy"),
        "{refused}"
    );

    let source = Source {
        tenant: Some("t1".into()),
        ..untenanted
    };
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled, "{report:?}");
    let counts: Vec<u64> = report
        .classes
        .values()
        .map(|copied| copied.vellev)
        .collect();
    assert_eq!(counts, [7, 10, 7]);
    let named = tenant.collection(&CollectionName::new("VellevImportNamed").unwrap());
    let schema = named.schema().await.unwrap().unwrap();
    assert_eq!(schema.distance, vellev_core::Distance::Dot);
    let stored = named.get(oid(3)).await.unwrap().unwrap();
    assert_eq!(stored.vector, Some(vector(3)));
    let tenanted = tenant.collection(&CollectionName::new("VellevImportTenanted").unwrap());
    let stored = tenanted.get(oid(3)).await.unwrap().unwrap();
    assert_eq!(stored.properties["title"], json!("t1"));
    assert!(tenanted.get(oid(103)).await.unwrap().is_none());

    // A named vector that changes, and an object of a tenant that goes,
    // are caught up with as any other.
    let renamed = json!({ "class": "VellevImportNamed", "id": uuid(3), "properties": { "title": "renamed" }, "vectors": { "default": vector(4) } });
    let path = format!("/v1/objects/VellevImportNamed/{}", uuid(3));
    weaviate.send(Method::PUT, &path, Some(renamed)).await;
    let path = format!("/v1/objects/VellevImportTenanted/{}?tenant=t1", uuid(5));
    weaviate.send(Method::DELETE, &path, None).await;
    let again = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert_eq!(again.outcome(true), Outcome::Settled, "{again:?}");
    assert_eq!(again.classes["VellevImportNamed"].changed, 1);
    assert_eq!(again.classes["VellevImportTenanted"].deleted, 1);
    let stored = named.get(oid(3)).await.unwrap().unwrap();
    assert_eq!(stored.vector, Some(vector(4)));
    assert!(tenanted.get(oid(5)).await.unwrap().is_none());
}

#[tokio::test]
async fn a_catch_up_replaces_and_deletes_only_what_the_import_left_and_nobody_has_changed() {
    let Some(importing) = Importing::new("VellevImportOwned", 40).await else {
        return;
    };
    let (class, weaviate, records) = (importing.class, &importing.weaviate, &importing.records);
    assert_eq!(importing.counts().await, ([40, 0, 0, 0, 0], [40; 3]));

    // Another writer stores a memory of its own, changes one the import
    // brought, replaces another and deletes two more.
    records.put(oid(1_000), &foreign(1_000)).await.unwrap();
    let patch = Changes::new(map(json!({ "n": -2 })));
    records.patch(oid(2), &patch).await.unwrap();
    records.put(oid(3), &foreign(3)).await.unwrap();
    records.delete(oid(4)).await.unwrap();
    records.delete(oid(6)).await.unwrap();
    // Meanwhile Weaviate deletes, changes and gains objects.
    weaviate.delete(class, 1).await;
    weaviate.delete(class, 2).await;
    weaviate.put(class, 3, changed(3)).await;
    weaviate.put(class, 4, changed(4)).await;
    weaviate.put(class, 5, changed(5)).await;
    weaviate.add(class, 50..51).await;

    let report = importing.run().await.unwrap();
    assert_eq!(
        counts(&report, class),
        ([1, 1, 1, 2, 1], [39; 3]),
        "note 50 stored, 5 changed, 1 deleted; 3 and 4 kept as found; 2 left behind"
    );
    assert_eq!(report.outcome(false), Outcome::Settled);
    assert_eq!(report.outcome(true), Outcome::Unsettled);

    let untouched = |stored: Option<vellev_api::Stored>, content: &Content| {
        let stored = stored.unwrap();
        assert_eq!(stored.properties, content.properties);
        assert_eq!(stored.vector, content.vector);
    };
    untouched(importing.get(1_000).await, &foreign(1_000));
    untouched(importing.get(3).await, &foreign(3));
    assert!(importing.get(1).await.is_none());
    assert_eq!(importing.get(2).await.unwrap().properties["n"], json!(-2));
    assert!(
        importing.get(4).await.is_none(),
        "deleted, not brought back"
    );
    assert!(
        importing.get(6).await.is_none(),
        "deleted, not brought back"
    );
    assert_eq!(
        importing.get(5).await.unwrap().properties["n"],
        json!(1_000_005)
    );
    assert!(importing.get(50).await.is_some());

    // Run again, the import finds nothing of its own to do, leaves the
    // other writer's objects where they are, and says the same of them.
    assert_eq!(importing.counts().await, ([0, 0, 0, 2, 0], [39; 3]));
    untouched(importing.get(1_000).await, &foreign(1_000));
    untouched(importing.get(3).await, &foreign(3));

    // Weaviate changes an object the other writer deleted: it stays
    // deleted.
    weaviate.put(class, 6, changed(6)).await;
    assert_eq!(importing.counts().await, ([0, 0, 0, 3, 0], [39; 3]));
    assert!(importing.get(6).await.is_none());
}

#[tokio::test]
async fn objects_a_mirror_wrote_under_weaviates_ids_are_taken_as_they_are() {
    let Some(mut importing) = Importing::new("VellevImportMirrored", 10).await else {
        return;
    };
    // No time of creation is kept, so that a mirror can store an object
    // exactly as the import would.
    importing.options.created_as = None;
    let (class, weaviate, records) = (importing.class, &importing.weaviate, &importing.records);
    assert_eq!(importing.counts().await, ([10, 0, 0, 0, 0], [10; 3]));

    // Weaviate gains notes 10..30. A mirror has already stored 10..20 as
    // Weaviate has them and 20..25 otherwise, as it would a change the
    // import has yet to see.
    weaviate.add(class, 10..30).await;
    for n in 10..20 {
        records.put(oid(n), &mirrored(n)).await.unwrap();
    }
    for n in 20..25 {
        records.put(oid(n), &foreign(n)).await.unwrap();
    }
    // The import begins again from nothing, as one whose state was lost:
    // every page holds objects that are already there.
    std::fs::remove_file(&importing.options.state).unwrap();
    assert_eq!(
        importing.counts().await,
        ([5, 0, 0, 5, 0], [30; 3]),
        "25..30 stored; 20..25 kept as the mirror has them"
    );
    for n in 20..25 {
        assert_eq!(
            importing.get(n).await.unwrap().properties,
            foreign(n).properties
        );
    }
    assert_eq!(importing.counts().await, ([0, 0, 0, 5, 0], [30; 3]));

    // Weaviate changes a note and the mirror stores the change: the
    // object is as the import would have it, and the import's.
    weaviate.put(class, 20, changed(20)).await;
    let mut caught_up = Content::new(map(changed(20)));
    caught_up.vector = Some(vector(20));
    records.put(oid(20), &caught_up).await.unwrap();
    assert_eq!(importing.counts().await, ([0, 0, 0, 4, 0], [30; 3]));

    // What the mirror stored as Weaviate has it goes when Weaviate's
    // does; what it stored otherwise is left for it to delete.
    weaviate.delete(class, 12).await;
    weaviate.delete(class, 22).await;
    assert_eq!(importing.counts().await, ([0, 0, 1, 3, 1], [28, 28, 29]));
    assert!(importing.get(12).await.is_none());
    assert!(importing.get(22).await.is_some());
}

#[tokio::test]
async fn an_object_another_writer_changes_or_deletes_while_the_import_deletes_it_is_left() {
    let Some(importing) = Importing::new("VellevImportRaced", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    weaviate.delete(class, 3).await;
    weaviate.delete(class, 6).await;

    // The other writer's delete of note 6, and its change to note 3,
    // land between the import's reading each and its deleting it.
    let records = importing.records.clone();
    let read = importing.read(6);
    importing
        .front
        .after(0, &["GET", &read], move || async move {
            records.delete(oid(6)).await.unwrap();
        });
    let records = importing.records.clone();
    let read = importing.read(3);
    importing
        .front
        .after(0, &["GET", &read], move || async move {
            let patch = Changes::new(map(json!({ "n": 99 })));
            records.patch(oid(3), &patch).await.unwrap();
        });
    assert_eq!(importing.counts().await, ([0, 0, 0, 0, 1], [8, 8, 9]));
    assert!(importing.front.spent());
    assert!(importing.get(6).await.is_none());
    assert_eq!(importing.get(3).await.unwrap().properties["n"], json!(99));
}

#[tokio::test]
async fn an_object_another_writer_changes_while_the_import_replaces_it_is_kept() {
    let Some(importing) = Importing::new("VellevImportRacedReplace", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    // Note 20 has no tags.
    let mut untagged = object(20);
    untagged.as_object_mut().unwrap().remove("tags");
    weaviate.post(class, 20, untagged, vector(20)).await;
    importing.run().await.unwrap();
    weaviate.put(class, 4, changed(4)).await;
    weaviate.put(class, 20, changed(20)).await;

    // The other writer changes a value of note 4, and gives note 20 one
    // where it had none, between the import's reading the two and its
    // storing Weaviate's in their place.
    let records = importing.records.clone();
    let read = format!("/collections/{class}/list");
    importing
        .front
        .after(0, &["POST", &read], move || async move {
            let patch = Changes::new(map(json!({ "n": 99 })));
            records.patch(oid(4), &patch).await.unwrap();
            let patch = Changes::new(map(json!({ "tags": ["theirs"] })));
            records.patch(oid(20), &patch).await.unwrap();
        });
    assert_eq!(importing.counts().await, ([0, 0, 0, 2, 0], [11; 3]));
    assert!(importing.front.spent());
    let kept = importing.get(4).await.unwrap();
    assert_eq!(kept.properties["n"], json!(99));
    assert_eq!(kept.properties["title"], json!("note 4 about hens"));
    let kept = importing.get(20).await.unwrap();
    assert_eq!(kept.properties["tags"], json!(["theirs"]));
    assert_eq!(kept.properties["title"], json!("note 20 about owls"));
}

#[tokio::test]
async fn an_object_another_writer_stores_while_the_import_stores_its_page_is_kept() {
    let Some(importing) = Importing::new("VellevImportRacedStore", 10).await else {
        return;
    };
    // The collection is made by the import's first request to change
    // the tenant; its second stores the page, none of whose ids the
    // ledger has. The other writer stores note 3 just before it.
    let records = importing.records.clone();
    importing
        .front
        .before(1, &["POST", "/changes"], move |_| async move {
            records.put(oid(3), &foreign(3)).await.unwrap();
        });
    assert_eq!(importing.counts().await, ([9, 0, 0, 1, 0], [10; 3]));
    assert!(importing.front.spent());
    assert_eq!(
        importing.get(3).await.unwrap().properties,
        foreign(3).properties
    );
}

#[tokio::test]
async fn an_entry_written_while_an_object_is_settled_is_read_before_it_is_settled_again() {
    let Some(mut importing) = Importing::new("VellevImportRacedEntry", 10).await else {
        return;
    };
    importing.options.created_as = None;
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    weaviate.add(class, 30..31).await;

    // Another import, as it were, stores note 30 while this one is about
    // to, and then its entry while this one settles the note by itself.
    let read = importing.read(30);
    let records = importing.records.clone();
    importing
        .front
        .after(0, &["GET", &read], move || async move {
            records.put(oid(30), &mirrored(30)).await.unwrap();
        });
    let ledger = importing.ledger();
    importing
        .front
        .after(0, &["GET", &read], move || async move {
            let entry = Content::new(map(json!({ "hash": "another import's" })));
            ledger.put(oid(30), &entry).await.unwrap();
        });
    assert_eq!(importing.counts().await, ([0; 5], [11; 3]));
    assert!(importing.front.spent());
    importing.assert_same().await;
}

#[tokio::test]
async fn a_refusal_that_no_other_writer_explains_stops_the_run() {
    let Some(importing) = Importing::new("VellevImportBusy", 10).await else {
        return;
    };
    importing.run().await.unwrap();
    importing.weaviate.put(importing.class, 4, changed(4)).await;
    // The node answers as it does of a tenant it is purging: the same
    // code as for an object another writer changed, with nothing changed.
    let busy = json!({ "error": { "code": "conflict", "message": "tenant is being purged" } });
    // The first request to change the tenant, which sees to its
    // collections, is let through.
    for skip in [1, 0, 0] {
        let changes = ["POST", "/changes"];
        importing
            .front
            .answer(skip, &changes, StatusCode::CONFLICT, busy.clone());
    }
    let refused = importing.run().await.unwrap_err();
    assert!(
        format!("{refused:#}").contains("being purged"),
        "{refused:#}"
    );
}

#[tokio::test]
async fn a_class_that_gained_a_property_goes_on_being_imported_with_it() {
    let Some(importing) = Importing::new("VellevImportGrown", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();

    let mood = json!({ "name": "mood", "dataType": ["text"], "tokenization": "field" });
    let path = format!("/v1/schema/{class}/properties");
    weaviate.send(Method::POST, &path, Some(mood)).await;
    let mut moody = object(1);
    moody["mood"] = json!("glad");
    weaviate.put(class, 1, moody.clone()).await;
    let mut newcomer = object(20);
    newcomer["mood"] = json!("wary");
    weaviate.post(class, 20, newcomer, vector(20)).await;

    let report = importing.run().await.unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled, "{report:?}");
    assert_eq!(counts(&report, class), ([1, 1, 0, 0, 0], [11; 3]));
    let schema = importing.records.schema().await.unwrap().unwrap();
    let mood = schema.property("mood").unwrap();
    assert_eq!(mood.data_type, vellev_core::DataType::Text);
    assert!(
        mood.filterable && !mood.searchable,
        "mapped as on a first import"
    );
    assert_eq!(
        importing.get(1).await.unwrap().properties["mood"],
        json!("glad")
    );
    assert_eq!(importing.get(0).await.unwrap().properties.get("mood"), None);
    importing.assert_same().await;

    // A class made again with a property of another type, and vectors of
    // another size, is refused, and says both, even to one told to go
    // on from a class made again.
    weaviate.drop(class).await;
    let mut schema = Weaviate::schema(class);
    schema["properties"][2] = json!({ "name": "n", "dataType": ["text"] });
    weaviate
        .send(Method::POST, "/v1/schema", Some(schema))
        .await;
    let other = json!({ "class": class, "id": uuid(0), "properties": { "n": "nought" }, "vector": [1.0, 2.0, 3.0] });
    weaviate
        .send(Method::POST, "/v1/objects", Some(other))
        .await;
    let remade = Options {
        accept_remade: true,
        ..importing.options.clone()
    };
    let refused = importing.run_as(&remade).await.unwrap_err();
    let refused = format!("{refused:#}");
    for reason in [
        "no longer fits its collection",
        "its vectors have 3 dimensions and the collection's 4",
        "property n is text and the collection's int",
    ] {
        assert!(refused.contains(reason), "{reason} in {refused}");
    }
    assert_eq!(importing.records.count(None).await.unwrap(), 11);
}

#[tokio::test]
async fn an_object_gone_from_weaviate_between_being_listed_and_fetched_is_gone_from_the_collection()
{
    let Some(importing) = Importing::new("VellevImportVanished", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, importing.weaviate.clone());
    importing.run().await.unwrap();

    // Note 4 changes, so the catch-up fetches it, and is deleted before
    // it does. Note 30 is new, and deleted before it is fetched.
    weaviate.put(class, 4, changed(4)).await;
    weaviate.add(class, 30..31).await;
    for n in [4, 30] {
        let weaviate = weaviate.clone();
        let fetch = importing.object(n);
        importing.proxy.before(0, &[&fetch], move |_| async move {
            weaviate.delete(class, n).await;
        });
    }
    let report = importing.run().await.unwrap();
    assert!(importing.proxy.spent());
    assert_eq!(counts(&report, class), ([0, 0, 1, 0, 0], [19; 3]));
    assert_eq!(report.outcome(true), Outcome::Settled);
    assert!(importing.get(4).await.is_none());
    importing.assert_same().await;
}

#[tokio::test]
async fn an_import_of_a_weaviate_written_to_all_the_while_is_exact_after_the_next_run() {
    let Some(importing) = Importing::new("VellevImportLive", 300).await else {
        return;
    };
    let (class, weaviate) = (importing.class, importing.weaviate.clone());

    // Before a page is passed on, Weaviate gains objects, has others
    // changed and loses others, on both sides of where the walk has
    // reached, the one the page starts after among them: on the first
    // walk's third page, and on the catch-up's second and fourth.
    let disturb = |vectors: bool, skip: usize, nth: u64| {
        let weaviate = weaviate.clone();
        let page = importing.page(vectors);
        importing
            .proxy
            .before(skip, &strs(&page), move |line| async move {
                weaviate
                    .add(class, 1_000 + 30 * nth..1_008 + 30 * nth)
                    .await;
                for n in 10 + 30 * nth..18 + 30 * nth {
                    weaviate.put(class, n, changed(n)).await;
                }
                for n in 150 + 30 * nth..158 + 30 * nth {
                    weaviate.delete(class, n).await;
                }
                weaviate.forget(class, after(&line)).await;
            });
    };
    disturb(true, 1, 0);
    disturb(false, 1, 1);
    disturb(false, 2, 2);

    let first = importing.run().await.unwrap();
    assert!(importing.proxy.spent());
    let copied = &first.classes[class];
    assert_eq!(
        first.outcome(false),
        Outcome::Behind,
        "written to behind the catch-up, the run says it has not caught up: {copied:?}"
    );
    assert_eq!((copied.kept, copied.left), (0, 0));

    let second = importing.run().await.unwrap();
    assert_eq!(second.outcome(true), Outcome::Settled, "{second:?}");
    importing.assert_same().await;
    let held = weaviate.held(class).await;
    assert!(
        (1_000..1_008).all(|n| held.contains_key(&oid(n))) && !held.contains_key(&oid(150)),
        "Weaviate was written to"
    );
    assert_eq!(held[&oid(70)].0["n"], json!(1_000_070.0));
    assert_eq!(
        importing.counts().await,
        ([0; 5], [held.len() as u64; 3]),
        "and then there is nothing more to do"
    );
}

#[tokio::test]
async fn a_run_deletes_nothing_when_more_is_gone_than_it_may_delete() {
    let Some(importing) = Importing::new("VellevImportBudget", 40).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    for n in [1, 2, 3] {
        weaviate.delete(class, n).await;
    }
    weaviate.put(class, 5, changed(5)).await;

    let caught_up = || {
        let state = std::fs::read(&importing.options.state).unwrap();
        let progress: Value = serde_json::from_slice(&state).unwrap();
        progress["classes"][class]["caught_up_to"].clone()
    };
    let before = caught_up();

    // By default a run deletes at most one object in twenty of a class.
    let usual = Options {
        deletes: Options::new(Default::default()).deletes,
        ..importing.options.clone()
    };
    let report = importing.run_as(&usual).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(counts(&report, class), ([0, 1, 0, 0, 0], [37, 40, 40]));
    assert_eq!((copied.pending, copied.budget), (3, Some(2)));
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    assert_eq!(
        caught_up(),
        before,
        "a catch-up that deleted nothing it should have is not one to go on from"
    );
    for n in [1, 2, 3] {
        assert!(importing.get(n).await.is_some());
    }

    // Told to delete nothing, it counts what is gone and is done.
    let none = Options {
        deletes: Deletes::None,
        ..importing.options.clone()
    };
    let report = importing.run_as(&none).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(counts(&report, class), ([0; 5], [37, 40, 40]));
    assert_eq!((copied.pending, copied.budget), (3, None));
    assert_eq!(
        report.outcome(false),
        Outcome::Unsettled,
        "what is gone and was not deleted is still to be settled"
    );
    assert_ne!(caught_up(), before);

    // Within what it may delete, it deletes: three of the sixty the
    // ledger then holds.
    weaviate.add(class, 100..120).await;
    let report = importing.run_as(&none).await.unwrap();
    assert_eq!(counts(&report, class), ([20, 0, 0, 0, 0], [57, 60, 60]));
    let report = importing.run_as(&usual).await.unwrap();
    assert_eq!(counts(&report, class), ([0, 0, 3, 0, 0], [57; 3]));
    assert_eq!(report.outcome(true), Outcome::Settled);
}

#[tokio::test]
async fn nothing_is_deleted_for_a_class_made_again_or_dropped() {
    let Some(importing) = Importing::new("VellevImportRemade", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, importing.weaviate.clone());
    importing.run().await.unwrap();
    let all = async || {
        let objects = importing.records.count(None).await.unwrap();
        (objects, importing.ledger().count(None).await.unwrap())
    };

    // Made again, as a Weaviate begun afresh at the same address would
    // have it, with one note it had, changed, and one it had not: the
    // class is not the one imported, and nothing of it is stored either.
    weaviate.drop(class).await;
    weaviate.make(class).await;
    weaviate.post(class, 0, changed(0), vector(0)).await;
    let refused = importing.run().await.unwrap_err();
    assert!(
        refused.to_string().contains("has been made again"),
        "{refused}"
    );
    assert_eq!(all().await, (20, 20));
    assert_eq!(importing.get(0).await.unwrap().properties["n"], json!(0));

    // Told to go on from it, a run still deletes no more than it may:
    // here nothing, where nineteen of twenty are gone.
    let usual = Options {
        accept_remade: true,
        deletes: Options::new(Default::default()).deletes,
        ..importing.options.clone()
    };
    let report = importing.run_as(&usual).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(
        (copied.deleted, copied.pending, copied.budget),
        (0, 19, Some(1))
    );
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    assert_eq!(all().await, (20, 20));

    // Made again while a catch-up walks it, with no bound on what a run
    // may delete: what the walk no longer lists is asked for, and the
    // class is found to be another before anything is deleted.
    let weaviate = Weaviate::load(&weaviate.url, class, 20).await;
    let remade = Options {
        accept_remade: true,
        page: 5,
        ..importing.options.clone()
    };
    importing.run_as(&remade).await.unwrap();
    assert_eq!(all().await, (20, 20));
    let again = weaviate.clone();
    let page = importing.page(false);
    importing
        .proxy
        .before(1, &strs(&page), move |_| async move {
            again.drop(class).await;
            again.make(class).await;
        });
    let options = Options {
        page: 5,
        ..importing.options.clone()
    };
    let refused = importing.run_as(&options).await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("has been made again"),
        "{refused}"
    );
    assert_eq!(all().await, (20, 20));

    // Dropped as the deleting begins: Weaviate answers of an object in a
    // class it no longer has as of one it deleted.
    let weaviate = Weaviate::load(&weaviate.url, class, 20).await;
    importing.run_as(&remade).await.unwrap();
    weaviate.delete(class, 5).await;
    let dropping = weaviate.clone();
    let shards = format!("GET /v1/schema/{class}/shards");
    importing.proxy.before(1, &[&shards], move |_| async move {
        dropping.drop(class).await;
    });
    let refused = importing.run().await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("no longer has class"),
        "{refused}"
    );
    assert_eq!(all().await, (20, 20));
}

#[tokio::test]
async fn a_weaviate_that_is_starting_is_not_believed_about_what_it_does_not_hold() {
    let Some(importing) = Importing::new("VellevImportStarting", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();

    // A Weaviate that is starting lists less than it holds, and answers
    // that it holds no such object for objects it holds. Here it leaves
    // notes 5 and 6 out of its listing, says it is not ready, then that
    // a shard is still loading, and answers the first asking for note 5
    // as if it held none.
    let mut listed = weaviate.listed(class).await;
    let objects = listed["objects"].as_array_mut().unwrap();
    objects.retain(|object| ![json!(uuid(5)), json!(uuid(6))].contains(&object["id"]));
    let first = importing.first();
    let proxy = &importing.proxy;
    proxy.answer(0, &strs(&first), StatusCode::OK, listed.clone());
    let unready = StatusCode::SERVICE_UNAVAILABLE;
    proxy.answer(0, &["/v1/.well-known/ready"], unready, json!({}));
    let shards = format!("GET /v1/schema/{class}/shards");
    let loading = json!([{ "name": "s", "status": "LOADING" }]);
    proxy.answer(1, &[&shards], StatusCode::OK, loading.clone());
    let missing = StatusCode::NOT_FOUND;
    proxy.answer(0, &[&importing.object(5)], missing, json!({}));
    assert_eq!(importing.counts().await, ([0; 5], [20; 3]));
    assert!(proxy.spent());

    // One whose shards never finish loading is waited on, and then the
    // run stops, having deleted nothing, though here the note is gone.
    weaviate.delete(class, 6).await;
    let impatient = Options {
        patience: Duration::from_secs(1),
        ..importing.options.clone()
    };
    for skip in [1, 0, 0, 0, 0, 0, 0, 0] {
        proxy.answer(skip, &[&shards], StatusCode::OK, loading.clone());
    }
    let refused = importing.run_as(&impatient).await.unwrap_err();
    assert!(
        refused.to_string().contains("was not ready to say"),
        "{refused}"
    );
    assert!(importing.get(6).await.is_some());
    proxy.rules.lock().unwrap().clear();

    // Nor is one believed that answers the question of whether it is
    // ready with anything but that it is.
    let ready = ["/v1/.well-known/ready"];
    proxy.answer(0, &ready, StatusCode::NOT_FOUND, json!({}));
    let refused = importing.run().await.unwrap_err();
    assert!(
        refused.to_string().contains("whether it is ready"),
        "{refused}"
    );
    assert!(importing.get(6).await.is_some());

    // Ready, and asked twice, it is believed.
    assert_eq!(importing.counts().await, ([0, 0, 1, 0, 0], [19; 3]));
}

#[tokio::test]
async fn a_class_emptied_in_weaviate_is_emptied_by_a_run_allowed_to() {
    let Some(importing) = Importing::new("VellevImportEmptied", 10).await else {
        return;
    };
    importing.run().await.unwrap();
    for n in 0..10 {
        importing.weaviate.delete(importing.class, n).await;
    }
    // No object is left to tell the length of the class's vectors by:
    // the collection says it. And the tests let a run delete all of a
    // class.
    assert_eq!(importing.counts().await, ([0, 0, 10, 0, 0], [0; 3]));
}

#[tokio::test]
async fn a_weaviate_that_turns_requests_away_or_fails_for_a_while_is_asked_again() {
    let Some(importing) = Importing::new("VellevImportRetried", 120).await else {
        return;
    };
    let page = importing.page(true);
    for status in [
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::REQUEST_TIMEOUT,
        StatusCode::SERVICE_UNAVAILABLE,
    ] {
        importing.proxy.answer(0, &strs(&page), status, json!({}));
    }
    assert_eq!(importing.counts().await, ([120, 0, 0, 0, 0], [120; 3]));
    assert!(importing.proxy.spent());

    // For no longer than it is told to be patient.
    for _ in 0..4 {
        let unavailable = StatusCode::SERVICE_UNAVAILABLE;
        importing
            .proxy
            .answer(0, &["GET /v1/schema"], unavailable, json!({}));
    }
    let impatient = Options {
        patience: Duration::from_secs(1),
        ..importing.options.clone()
    };
    let refused = importing.run_as(&impatient).await.unwrap_err();
    assert!(refused.to_string().contains("Weaviate failed"), "{refused}");
    assert!(!importing.proxy.spent(), "it gave up before the fourth");
}

#[tokio::test]
async fn a_weaviate_that_lists_its_ids_out_of_order_is_not_caught_up_with() {
    let Some(importing) = Importing::new("VellevImportUnordered", 120).await else {
        return;
    };
    importing.run().await.unwrap();

    // Walked beside the ledger, a listing out of order would make
    // objects that are there look gone.
    let page = importing.page(false);
    let unordered = json!({ "objects": [{ "id": uuid(3) }, { "id": uuid(2) }] });
    importing
        .proxy
        .answer(0, &strs(&page), StatusCode::OK, unordered);
    let refused = importing.run().await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("its ids are not in order"),
        "{refused}"
    );
    assert_eq!(importing.records.count(None).await.unwrap(), 120);

    // Nor is a ledger the node lists out of order walked.
    let list = format!("/collections/vellev_imported_{}/list", importing.class);
    let entry = |n: u64| json!({ "id": uuid(n), "properties": { "hash": "h" } });
    let unordered = json!({ "objects": [entry(3), entry(2)] });
    importing
        .front
        .answer(0, &["POST", &list], StatusCode::OK, unordered);
    let refused = importing.run().await.unwrap_err();
    assert!(importing.front.spent());
    assert!(refused.to_string().contains("out of order"), "{refused}");
    assert_eq!(importing.counts().await, ([0; 5], [120; 3]));
}

#[tokio::test]
async fn a_collection_is_caught_up_only_from_the_weaviate_it_was_imported_from_and_as_it_was() {
    let Some(importing) = Importing::new("VellevImportBound", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    weaviate.delete(class, 1).await;

    // The same Weaviate at another address, with no state to go by, is
    // taken for another Weaviate, and nothing is caught up from it.
    let elsewhere = Options {
        state: importing.dir.path().join("another.json"),
        ..importing.options.clone()
    };
    let direct = source(&weaviate.url);
    let refused = import::weaviate(&direct, &importing.tenant, &elsewhere)
        .await
        .unwrap_err();
    let was = format!("was imported from {}", importing.proxy.url);
    assert!(refused.to_string().contains(&was), "{refused}");
    assert_eq!(importing.records.count(None).await.unwrap(), 10);

    // Named as it was known then, it is caught up from.
    let named = Source {
        name: Some(importing.proxy.url.clone()),
        ..direct
    };
    let report = import::weaviate(&named, &importing.tenant, &elsewhere)
        .await
        .unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled, "{report:?}");
    assert_eq!(report.classes[class].deleted, 1);

    let otherwise = Options {
        created_as: None,
        ..importing.options.clone()
    };
    let refused = importing.run_as(&otherwise).await.unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("an import goes on as it began"),
        "{refused}"
    );
}

#[tokio::test]
async fn two_runs_do_not_share_a_state_file() {
    let Some(importing) = Importing::new("VellevImportLocked", 120).await else {
        return;
    };
    let (one, other) = tokio::join!(importing.run(), importing.run());
    let refused = match (one, other) {
        (Ok(_), Err(refused)) | (Err(refused), Ok(_)) => refused,
        (one, other) => panic!("one of two runs is refused: {one:?} {other:?}"),
    };
    assert!(
        refused.to_string().contains("another import is running"),
        "{refused}"
    );
    assert_eq!(importing.counts().await, ([0; 5], [120; 3]));
}

#[tokio::test]
async fn a_collection_dropped_is_imported_again_from_the_start() {
    let Some(importing) = Importing::new("VellevImportAgain", 20).await else {
        return;
    };
    importing.run().await.unwrap();
    importing.records.drop().await.unwrap();
    assert_eq!(importing.counts().await, ([20, 0, 0, 0, 0], [20; 3]));
    importing.assert_same().await;
}

#[tokio::test]
async fn a_catch_up_fetches_and_stores_many_changes_by_the_page() {
    let Some(importing) = Importing::new("VellevImportBatched", 60).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    for n in 0..30 {
        weaviate.put(class, n, changed(n)).await;
    }
    weaviate.add(class, 100..120).await;

    // Half of what the catch-up lists has changed or is new: neither
    // Weaviate nor the node is asked for an object by itself.
    let singly = format!("GET /v1/objects/{class}/");
    importing.proxy.before(0, &[&singly], |_| async {});
    let read = format!("/collections/{class}/objects/");
    importing.front.before(0, &["GET", &read], |_| async {});
    assert_eq!(importing.counts().await, ([20, 30, 0, 0, 0], [80; 3]));
    assert!(!importing.proxy.spent() && !importing.front.spent());
    importing.assert_same().await;
}

/// What a run withheld.
fn withheld(report: &Report, class: &str) -> u64 {
    report.classes[class].withheld
}

#[tokio::test]
async fn a_collection_without_a_ledger_is_taken_up_and_what_it_lacks_is_withheld_and_said_so() {
    let Some(importing) = Importing::new("VellevImportLedgerless", 120).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();

    // As a collection imported before there were ledgers: the import
    // cannot know which of its objects it brought, nor whether one that
    // is missing was deleted or is new to Weaviate.
    importing.ledger().drop().await.unwrap();
    weaviate.put(class, 2, changed(2)).await;
    weaviate.delete(class, 3).await;
    importing.records.delete(oid(9)).await.unwrap();
    weaviate.add(class, 300..301).await;

    // The taking up stops part way, and goes on from there.
    let page = importing.page(true);
    let fault = StatusCode::BAD_REQUEST;
    importing.proxy.answer(0, &strs(&page), fault, json!({}));
    let stopped = importing.run().await.unwrap_err();
    assert!(stopped.to_string().contains("400"), "{stopped}");
    let state = std::fs::read(&importing.options.state).unwrap();
    let progress: Value = serde_json::from_slice(&state).unwrap();
    assert_eq!(progress["classes"][class]["done"], json!(false));
    assert_ne!(progress["classes"][class]["after"], Value::Null);
    assert_eq!(importing.ledger().count(None).await.unwrap(), 50);
    // Weaviate makes a note meanwhile, ahead of where the walk stopped:
    // made since the taking up began, it is new to Weaviate.
    weaviate.add(class, 310..311).await;

    importing.front.listen();
    let report = importing.run().await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([1, 0, 0, 0, 0], [121, 118, 120]), 3),
        "note 2 is not as Weaviate has it, notes 9 and 300 are not there: withheld"
    );
    assert!(importing.get(310).await.is_some());
    assert_eq!(
        report.outcome(false),
        Outcome::Unsettled,
        "a run that did not copy what Weaviate holds has not settled"
    );
    assert_eq!(importing.get(2).await.unwrap().properties["n"], json!(2));
    assert!(importing.get(3).await.is_some(), "nobody's to delete");
    assert!(importing.get(9).await.is_none());
    assert!(importing.get(300).await.is_none());
    // Each page of the collection was read from where the one before
    // ended, not from the start: the two pages left, and at most once
    // more for the end of one the collection holds less of.
    let read = format!("/collections/{class}/list");
    assert!(importing.front.asked(&read).len() <= 3);

    // It says the same of them on every run, whatever Weaviate does to
    // them, and what it took up is the import's from then on, as is
    // what is new to Weaviate from then on.
    weaviate.put(class, 300, changed(300)).await;
    weaviate.put(class, 7, changed(7)).await;
    weaviate.delete(class, 8).await;
    weaviate.add(class, 301..302).await;
    let report = importing.run().await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([1, 1, 1, 0, 0], [121, 118, 120]), 3)
    );
    assert!(importing.get(301).await.is_some());
    assert!(importing.get(300).await.is_none());
    // And by one that has lost its state and walks from the start: the
    // taking up is over, and is not begun again.
    weaviate.add(class, 302..303).await;
    std::fs::remove_file(&importing.options.state).unwrap();
    let report = importing.run().await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([1, 0, 0, 0, 0], [122, 119, 121]), 3)
    );
    assert!(importing.get(302).await.is_some());

    // Told that nothing else wrote to the collection, any later run
    // copies what was withheld, and deletes what Weaviate deleted before
    // there was a ledger, which no entry covers.
    let sole = Options {
        sole_writer: true,
        ..importing.options.clone()
    };
    let report = importing.run_as(&sole).await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([2, 1, 1, 0, 0], [122; 3]), 0),
        "notes 9 and 300 stored, note 2 replaced, note 3 deleted"
    );
    assert_eq!(report.outcome(true), Outcome::Settled);
    importing.assert_same().await;
}

#[tokio::test]
async fn a_collection_without_a_ledger_and_no_other_writer_is_brought_to_what_weaviate_holds() {
    let Some(importing) = Importing::new("VellevImportSole", 60).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    weaviate.put(class, 2, changed(2)).await;
    for n in [3, 4, 5, 6] {
        weaviate.delete(class, n).await;
    }
    importing.records.delete(oid(9)).await.unwrap();
    weaviate.add(class, 300..301).await;

    // What Weaviate deleted before there was a ledger is deleted like
    // anything else: no more of it than a run may delete.
    let usual = Options {
        sole_writer: true,
        deletes: Options::new(Default::default()).deletes,
        ..importing.options.clone()
    };
    let report = importing.run_as(&usual).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(
        counts(&report, class),
        ([2, 1, 0, 0, 0], [57, 57, 61]),
        "notes 9 and 300 stored, note 2 replaced"
    );
    assert_eq!((copied.pending, copied.budget), (4, Some(3)));
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    assert_eq!(
        importing.get(2).await.unwrap().properties["n"],
        json!(1_000_002)
    );

    // And only once Weaviate has twice said, of every copy, that it does
    // not hold it.
    importing.proxy.listen();
    let sole = Options {
        sole_writer: true,
        ..importing.options.clone()
    };
    let report = importing.run_as(&sole).await.unwrap();
    assert_eq!(counts(&report, class), ([0, 0, 4, 0, 0], [57; 3]));
    assert_eq!(report.outcome(true), Outcome::Settled);
    let asked = importing.proxy.asked("consistency_level=ALL");
    assert_eq!(asked.len(), 8);
    importing.assert_same().await;
}

#[tokio::test]
async fn an_import_ended_drops_its_ledger_and_is_not_taken_up_again() {
    let Some(importing) = Importing::new("VellevImportEnded", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    let unknown = import::finalise(&importing.tenant, &["Nothing".to_string()]).await;
    assert!(unknown
        .unwrap_err()
        .to_string()
        .contains("was not imported"));
    assert_eq!(
        import::finalise(&importing.tenant, &[]).await.unwrap(),
        [class]
    );
    assert!(importing.ledger().schema().await.unwrap().is_none());

    weaviate.delete(class, 1).await;
    weaviate.put(class, 2, changed(2)).await;
    let refused = importing.run().await.unwrap_err();
    assert!(refused.to_string().contains("was ended"), "{refused}");
    assert_eq!(importing.records.count(None).await.unwrap(), 20);
    assert_eq!(importing.get(2).await.unwrap().properties["n"], json!(2));
    assert!(importing.ledger().schema().await.unwrap().is_none());
}

#[tokio::test]
async fn a_comparison_finds_what_the_counts_do_not() {
    let Some(importing) = Importing::new("VellevImportVerified", 120).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    let verify = async |pages: Option<usize>| {
        let options = Options {
            mode: Mode::Verify { pages },
            ..importing.options.clone()
        };
        let report = importing.run_as(&options).await.unwrap();
        let verified = report.classes[class].verified.clone().unwrap();
        (verified, report.outcome(false))
    };
    let (all, outcome) = verify(None).await;
    assert_eq!(
        (all.looked, all.same, outcome),
        (120, 120, Outcome::Settled)
    );
    let (some, _) = verify(Some(3)).await;
    assert!(some.looked > 0 && some.looked == some.same, "{some:?}");

    // Another writer changes one of the import's objects and deletes
    // another, a third is the other writer's by the ledger, and Weaviate
    // gains one: as many objects as before, and a run that settles.
    let patch = Changes::new(map(json!({ "n": -2 })));
    importing.records.patch(oid(2), &patch).await.unwrap();
    importing.records.delete(oid(3)).await.unwrap();
    importing.records.put(oid(4), &foreign(4)).await.unwrap();
    weaviate.put(class, 4, changed(4)).await;
    assert_eq!(importing.counts().await, ([0, 0, 0, 1, 0], [120, 120, 119]));
    weaviate.add(class, 300..301).await;
    let before = held(&importing.records).await;
    let (all, outcome) = verify(None).await;
    assert_eq!(
        (all.looked, all.same, all.different, all.missing),
        (121, 117, 1, 1)
    );
    assert_eq!((all.kept, all.unaccounted), (1, 1));
    let mut examples = all.examples.clone();
    examples.sort();
    assert_eq!(examples, [uuid(2), uuid(3)]);
    assert_eq!(outcome, Outcome::Unsettled);
    assert_eq!(held(&importing.records).await, before, "it changes nothing");
    assert_eq!(importing.ledger().count(None).await.unwrap(), 120);
}

#[tokio::test]
async fn an_object_vellev_cannot_hold_stops_the_run_or_is_passed_over_and_reported() {
    let Some(importing) = Importing::new("VellevImportRefused", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    // A vector longer than vellev measures.
    weaviate
        .post(class, 20, object(20), vec![3.0e30, 0.0, 0.0, 0.0])
        .await;
    let refused = importing.run().await.unwrap_err();
    assert!(
        refused
            .to_string()
            .contains(&format!("object {}", uuid(20))),
        "{refused}"
    );

    let skipping = Options {
        skip_refused: true,
        ..importing.options.clone()
    };
    let report = importing.run_as(&skipping).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(counts(&report, class), ([10, 0, 0, 0, 0], [11, 10, 10]));
    let refused: Vec<&String> = copied.refused.keys().collect();
    assert_eq!(refused, [&uuid(20)]);
    assert!(copied.refused[&uuid(20)].contains("too long"));
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    // It is said again by every run, and by the catch-up that finds it.
    let report = importing.run_as(&skipping).await.unwrap();
    assert_eq!(report.classes[class].refused.len(), 1);
}

#[tokio::test]
async fn an_object_too_large_to_replace_safely_is_left_and_reported() {
    let Some(importing) = Importing::new("VellevImportLarge", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    // It fits a request once, and not twice: once to say what must not
    // have changed and once more to replace it.
    let large = |mark: &str| json!({ "title": format!("{mark} {}", "x".repeat(MAX_BODY * 2 / 3)) });
    weaviate.post(class, 20, large("first"), vector(20)).await;
    assert_eq!(importing.counts().await, ([11, 0, 0, 0, 0], [11; 3]));
    weaviate.put(class, 20, large("second")).await;
    weaviate.put(class, 4, changed(4)).await;
    let report = importing.run().await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(counts(&report, class), ([0, 1, 0, 0, 0], [11; 3]));
    assert_eq!(copied.too_large, [uuid(20)]);
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    let title = importing.get(20).await.unwrap().properties["title"].clone();
    assert!(title.as_str().unwrap().starts_with("first"));
}

#[tokio::test]
async fn a_change_is_caught_up_with_however_far_behind_weaviates_clock_is() {
    let Some(importing) = Importing::new("VellevImportBehind", 10).await else {
        return;
    };
    let class = importing.class;
    importing.run().await.unwrap();

    // As if Weaviate's clock were an hour behind this one, and no margin
    // allowed for it: by its time alone the change is too old to look at.
    let state = &importing.options.state;
    let mut progress: Value = serde_json::from_slice(&std::fs::read(state).unwrap()).unwrap();
    let caught_up = &mut progress["classes"][class]["caught_up_to"];
    *caught_up = json!(caught_up.as_i64().unwrap() + 3_600_000);
    std::fs::write(state, serde_json::to_vec(&progress).unwrap()).unwrap();
    importing.weaviate.put(class, 5, changed(5)).await;
    assert_eq!(importing.counts().await, ([0, 1, 0, 0, 0], [10; 3]));
    importing.assert_same().await;
}

#[tokio::test]
async fn a_change_that_left_weaviates_time_as_it_was_is_found_within_the_margin() {
    let Some(importing) = Importing::new("VellevImportSameTime", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    let before = weaviate.listed(class).await;
    weaviate.put(class, 5, changed(5)).await;

    // Weaviate tells time by the millisecond: an object changed twice in
    // one, the import reading it between, is listed with the time the
    // ledger has. Here the listing says of note 5 what it said before.
    let first = importing.first();
    let listing = || {
        importing
            .proxy
            .answer(0, &strs(&first), StatusCode::OK, before.clone())
    };
    listing();
    assert_eq!(
        importing.counts().await,
        ([0; 5], [10; 3]),
        "with no margin it is taken for unchanged"
    );
    listing();
    let margin = Options {
        skew: Duration::from_secs(3_600),
        ..importing.options.clone()
    };
    let report = importing.run_as(&margin).await.unwrap();
    assert_eq!(counts(&report, class), ([0, 1, 0, 0, 0], [10; 3]));
    importing.assert_same().await;
}

#[tokio::test]
async fn an_object_written_again_unchanged_has_only_its_time_noted() {
    let Some(importing) = Importing::new("VellevImportRewritten", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    let noted = async || {
        let entry = importing.ledger().get(oid(5)).await.unwrap().unwrap();
        entry.properties["updated"].clone()
    };
    let listed = async || {
        let listed = weaviate.listed(class).await;
        let objects = listed["objects"].as_array().unwrap();
        let note = objects.iter().find(|object| object["id"] == json!(uuid(5)));
        note.unwrap()["lastUpdateTimeUnix"].clone()
    };
    let before = noted().await;
    assert_eq!(before, listed().await);
    weaviate.put(class, 5, changed(5)).await;
    weaviate.put(class, 5, object(5)).await;
    assert_ne!(before, listed().await);
    assert_eq!(importing.counts().await, ([0; 5], [10; 3]));
    assert_eq!(noted().await, listed().await);
    // So that the next catch-up has no reason to fetch it.
    let singly = importing.object(5);
    importing.proxy.before(0, &[&singly], |_| async {});
    assert_eq!(importing.counts().await, ([0; 5], [10; 3]));
    assert!(!importing.proxy.spent());
}

#[tokio::test]
async fn the_object_with_the_last_id_of_all_is_deleted_when_weaviate_deletes_it() {
    let Some(importing) = Importing::new("VellevImportLast", 10).await else {
        return;
    };
    importing.run().await.unwrap();
    let last = (0..10).max_by_key(|n| oid(*n)).unwrap();
    importing.weaviate.delete(importing.class, last).await;
    assert_eq!(importing.counts().await, ([0, 0, 1, 0, 0], [9; 3]));
    assert!(importing.get(last).await.is_none());
}

#[tokio::test]
async fn an_import_begun_again_does_not_bring_back_what_another_writer_deleted() {
    let Some(importing) = Importing::new("VellevImportBegunAgain", 10).await else {
        return;
    };
    importing.run().await.unwrap();
    for n in 0..10 {
        importing.records.delete(oid(n)).await.unwrap();
    }
    // With its state lost the import walks every page again, and finds
    // nothing in the collection under any id of the page: only the
    // ledger says the objects were there.
    std::fs::remove_file(&importing.options.state).unwrap();
    assert_eq!(importing.counts().await, ([0; 5], [10, 10, 0]));
}

#[tokio::test]
async fn a_class_vellev_embeds_is_caught_up_by_its_properties_alone() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let class = "VellevImportEmbedded";
    let weaviate = Weaviate::load(&url, class, 10).await;
    let model = keyed_embedding_server("sk-right").await;
    let served = Cluster::new()
        .serve_embedding_keyed(&model, "sk-right")
        .await;
    let tenant = tenant(&served.url, "embedded").await;
    let dir = tempfile::tempdir().unwrap();
    let embedding = json!({
        "provider": "http",
        "api": "openai",
        "url": model,
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    let options = Options {
        classes: vec![class.into()],
        embeddings: [(
            class.to_string(),
            serde_json::from_value(embedding).unwrap(),
        )]
        .into(),
        ..options(&dir.path().join("state.json"))
    };
    let source = source(&url);
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled, "{report:?}");
    let records = tenant.collection(&CollectionName::new(class).unwrap());
    let embedded = |title: &str| {
        Some(vec![
            title.len() as f32,
            title.split_whitespace().count() as f32,
        ])
    };
    let stored = records.get(oid(2)).await.unwrap().unwrap();
    assert_eq!(stored.vector, embedded("note 2 about owls"));

    // The vector vellev made is no part of what the import stored: the
    // object is still the import's, to change and to delete.
    weaviate.put(class, 2, changed(2)).await;
    weaviate.delete(class, 3).await;
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled, "{report:?}");
    assert_eq!(counts(&report, class), ([0, 1, 1, 0, 0], [9; 3]));
    let stored = records.get(oid(2)).await.unwrap().unwrap();
    assert_eq!(stored.properties["title"], json!("note 2 changed"));
    assert_eq!(stored.vector, embedded("note 2 changed"));
}

#[tokio::test]
async fn a_rate_that_is_no_number_or_none_at_all_is_refused() {
    let tenant = tenant("http://127.0.0.1:9", "unreached").await;
    for rate in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let options = Options {
            per_second: Some(rate),
            ..options(std::path::Path::new("unwritten.json"))
        };
        let refused = import::weaviate(&source("http://127.0.0.1:9"), &tenant, &options)
            .await
            .unwrap_err();
        assert!(
            refused.to_string().contains("the rate of copying"),
            "{rate}: {refused}"
        );
    }
}

/// An address that cuts short its answer to the first request made of
/// it, a 404 said to be longer than what is sent, and passes every other
/// connection on to `to`.
async fn cutting_short(to: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let to = to.trim_start_matches("http://").to_string();
    tokio::spawn(async move {
        let mut cut = false;
        loop {
            let (mut asking, _) = listener.accept().await.unwrap();
            if !std::mem::replace(&mut cut, true) {
                let mut asked = [0; 4096];
                assert!(asking.read(&mut asked).await.unwrap() > 0);
                let short = b"HTTP/1.1 404 Not Found\r\ncontent-length: 100\r\n\r\ncut";
                asking.write_all(short).await.unwrap();
                continue;
            }
            let to = to.clone();
            tokio::spawn(async move {
                let mut onward = tokio::net::TcpStream::connect(&to).await.unwrap();
                let _ = tokio::io::copy_bidirectional(&mut asking, &mut onward).await;
            });
        }
    });
    url
}

#[tokio::test]
async fn an_answer_cut_short_is_asked_for_again_though_it_began_as_a_404() {
    let Some(importing) = Importing::new("VellevImportCut", 10).await else {
        return;
    };
    // Taken at its word, a 404 would say Weaviate holds nothing there.
    let cutting = cutting_short(&importing.weaviate.url).await;
    let report = import::weaviate(&source(&cutting), &importing.tenant, &importing.options)
        .await
        .unwrap();
    assert_eq!(report.outcome(true), Outcome::Settled, "{report:?}");
    assert_eq!(report.classes[importing.class].stored, 10);
}

#[tokio::test]
async fn a_class_whose_name_leaves_no_room_for_its_ledgers_is_refused_before_anything_is_made() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let class = format!("VellevImportLong{}", "g".repeat(105));
    Weaviate::load(&url, &class, 1).await;
    let served = Served::new().await;
    let tenant = tenant(&served.url, "long").await;
    let dir = tempfile::tempdir().unwrap();
    let options = Options {
        classes: vec![class.clone()],
        ..options(&dir.path().join("state.json"))
    };
    let refused = import::weaviate(&source(&url), &tenant, &options)
        .await
        .unwrap_err();
    assert!(
        refused.to_string().contains("too long for a ledger"),
        "{refused}"
    );
    assert_eq!(tenant.collections().await.unwrap(), []);
}

// What a second review asked of the import, kept as it asked it.

fn awkward(count: usize) -> Vec<f64> {
    let mut state = 7_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    std::iter::repeat_with(move || f64::from_bits(next()))
        .filter(|n| n.is_finite())
        .take(count)
        .collect()
}

fn usual(options: &Options) -> Options {
    Options {
        deletes: Options::new(Default::default()).deletes,
        ..options.clone()
    }
}

#[tokio::test]
async fn objects_holding_doubles_of_every_size_are_replaced_when_they_change() {
    let Some(importing) = Importing::new("VellevImportDoubles", 0).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    let doubles = awkward(800);
    let store = async |from: usize| {
        let note = |n: u64| {
            let score = doubles[from + n as usize];
            let properties = json!({ "title": format!("d {n}"), "n": n, "score": score });
            json!({ "class": class, "id": uuid(n), "properties": properties, "vector": vector(n) })
        };
        let objects: Vec<Value> = (0..400).map(note).collect();
        let stored = json!({ "objects": objects });
        weaviate
            .send(Method::POST, "/v1/batch/objects", Some(stored))
            .await;
    };
    store(0).await;
    let report = importing.run().await.unwrap();
    assert_eq!(counts(&report, class), ([400, 0, 0, 0, 0], [400; 3]));
    assert_eq!(report.outcome(true), Outcome::Settled);
    importing.assert_same().await;
    store(400).await;
    let report = importing.run().await.unwrap();
    assert_eq!(counts(&report, class), ([0, 400, 0, 0, 0], [400; 3]));
    assert_eq!(report.outcome(true), Outcome::Settled);
    importing.assert_same().await;
    let weaviate_held = weaviate.held(class).await;
    let vellev_held = held(&importing.records).await;
    for (id, (properties, _)) in &weaviate_held {
        let theirs = properties["score"].as_f64().unwrap().to_bits();
        let ours = vellev_held[id].0["score"].as_f64().unwrap().to_bits();
        assert_eq!(theirs, ours, "{id}");
    }
    let verify = Options {
        mode: Mode::Verify { pages: None },
        ..importing.options.clone()
    };
    let report = importing.run_as(&verify).await.unwrap();
    let verified = report.classes[class].verified.clone().unwrap();
    assert_eq!((verified.looked, verified.same), (400, 400), "{verified:?}");
    let report = importing.run().await.unwrap();
    assert_eq!(counts(&report, class), ([0; 5], [400; 3]));
}

#[tokio::test]
async fn a_class_made_again_with_one_new_object_has_nothing_deleted_for_it_however_often_it_is_run()
{
    let Some(importing) = Importing::new("VellevImportRemadeNew", 200).await else {
        return;
    };
    let (class, weaviate) = (importing.class, importing.weaviate.clone());
    importing.run().await.unwrap();
    weaviate.drop(class).await;
    weaviate.make(class).await;
    weaviate.post(class, 500, object(500), vector(500)).await;
    let usual = usual(&importing.options);
    let refused = importing.run_as(&usual).await.unwrap_err();
    assert!(refused.to_string().contains("made again"), "{refused}");
    assert_eq!(importing.records.count(None).await.unwrap(), 200);
    assert!(importing.get(500).await.is_none());
    let accept = Options {
        accept_remade: true,
        ..usual.clone()
    };
    let report = importing.run_as(&accept).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(
        (copied.deleted, copied.pending, copied.budget),
        (0, 200, Some(10))
    );
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    assert_eq!(importing.records.count(None).await.unwrap(), 201);
    for _ in 0..3 {
        let report = importing.run_as(&usual).await.unwrap();
        let copied = &report.classes[class];
        assert_eq!((copied.deleted, copied.pending), (0, 200), "{copied:?}");
        assert_eq!(report.outcome(false), Outcome::Unsettled);
    }
    assert_eq!(importing.records.count(None).await.unwrap(), 201);
}

#[tokio::test]
async fn a_class_made_again_under_a_catch_up_with_the_same_ids_is_not_stored_over_the_imports() {
    let Some(importing) = Importing::new("VellevImportRemadeMid", 20).await else {
        return;
    };
    let class = importing.class;
    let options = Options {
        page: 5,
        ..usual(&importing.options)
    };
    importing.run_as(&options).await.unwrap();
    let again = importing.weaviate.clone();
    let page = importing.page(false);
    importing
        .proxy
        .before(1, &strs(&page), move |_| async move {
            again.drop(class).await;
            again.make(class).await;
            for n in 0..20 {
                again.post(class, n, changed(n), vector(n)).await;
            }
        });
    let refused = importing.run_as(&options).await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("has been made again"),
        "{refused}"
    );
    for n in 0..20 {
        let stored = importing.get(n).await.unwrap();
        assert_eq!(stored.properties["n"], json!(n), "stored over");
    }
}

#[tokio::test]
#[ignore = "takes a condition on an object's version, which vellev does not have yet: DW-677"]
async fn an_object_whose_vector_alone_another_writer_changes_while_the_import_deletes_it_is_kept() {
    let Some(importing) = Importing::new("VellevImportVectorDelete", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    weaviate.delete(class, 3).await;
    let records = importing.records.clone();
    let read = importing.read(3);
    importing
        .front
        .after(0, &["GET", &read], move || async move {
            let mut patch = Changes::new(Map::new());
            patch.vector = Some(vec![0.0, 0.0, 0.0, 1.0]);
            records.patch(oid(3), &patch).await.unwrap();
        });
    importing.run().await.unwrap();
    assert!(importing.front.spent());
    assert!(importing.get(3).await.is_some());
}

#[tokio::test]
#[ignore = "takes a condition on an object's version, which vellev does not have yet: DW-677"]
async fn an_object_whose_vector_alone_another_writer_changes_while_the_import_replaces_it_is_kept()
{
    let Some(importing) = Importing::new("VellevImportVectorReplace", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    weaviate.put(class, 4, changed(4)).await;
    let records = importing.records.clone();
    let read = importing.read(4);
    importing
        .front
        .after(0, &["GET", &read], move || async move {
            let mut patch = Changes::new(Map::new());
            patch.vector = Some(vec![0.0, 0.0, 0.0, 1.0]);
            records.patch(oid(4), &patch).await.unwrap();
        });
    importing.run().await.unwrap();
    assert!(importing.front.spent());
    assert_eq!(importing.get(4).await.unwrap().properties["n"], json!(4));
}

/// As a collection an earlier release imported: no ledger. Weaviate has
/// gone on being written to since.
async fn ledgerless(class: &'static str) -> Option<Importing> {
    let importing = Importing::new(class, 100).await?;
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    importing.weaviate.add(class, 300..320).await;
    importing.weaviate.put(class, 2, changed(2)).await;
    Some(importing)
}

#[tokio::test]
async fn a_taking_up_does_not_settle_over_what_it_did_not_copy() {
    let Some(importing) = ledgerless("VellevImportAdoptExit").await else {
        return;
    };
    let class = importing.class;
    let report = importing.run().await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([0; 5], [120, 99, 100]), 21),
        "the twenty notes Weaviate gained and the one it changed are not copied, and counted so"
    );
    assert_eq!(report.outcome(false), Outcome::Unsettled);
}

#[tokio::test]
async fn a_run_told_of_no_other_writer_after_a_taking_up_brings_the_collection_level() {
    let Some(importing) = ledgerless("VellevImportAdoptLate").await else {
        return;
    };
    let class = importing.class;
    importing.run().await.unwrap();
    let sole = Options {
        sole_writer: true,
        ..importing.options.clone()
    };
    let report = importing.run_as(&sole).await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([20, 1, 0, 0, 0], [120; 3]), 0)
    );
    assert_eq!(report.outcome(true), Outcome::Settled);
    importing.assert_same().await;
}

#[tokio::test]
async fn an_object_weaviate_makes_ahead_of_a_taking_up_is_stored() {
    let Some(importing) = Importing::new("VellevImportAdoptNew", 100).await else {
        return;
    };
    let class = importing.class;
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    let adding = importing.weaviate.clone();
    let page = importing.page(true);
    // Note 403 has the last id of all: the walk has yet to come to it.
    importing
        .proxy
        .before(0, &strs(&page), move |_| async move {
            adding.add(class, 403..404).await;
        });
    let report = importing.run().await.unwrap();
    assert!(importing.proxy.spent());
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([1, 0, 0, 0, 0], [101; 3]), 0),
        "made since the taking up began, it is new to Weaviate"
    );
    assert_eq!(report.outcome(true), Outcome::Settled);
    assert!(importing.get(403).await.is_some());
}

#[tokio::test]
async fn what_was_deleted_from_a_collection_without_a_ledger_does_not_come_back() {
    let Some(importing) = Importing::new("VellevImportAdoptDeleted", 100).await else {
        return;
    };
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    for n in [1, 9, 50, 99] {
        importing.records.delete(oid(n)).await.unwrap();
    }
    for round in 0..3 {
        if round == 2 {
            std::fs::remove_file(&importing.options.state).unwrap();
        }
        importing.run().await.unwrap();
        for n in [1, 9, 50, 99] {
            assert!(
                importing.get(n).await.is_none(),
                "round {round} note {n} came back"
            );
        }
    }
    // Weaviate writes the object again: still not brought back.
    importing.weaviate.put(importing.class, 9, changed(9)).await;
    importing.run().await.unwrap();
    assert!(importing.get(9).await.is_none());
}

#[tokio::test]
async fn the_last_object_of_a_class_is_not_deleted_by_a_run_as_it_usually_is() {
    let Some(importing) = Importing::new("VellevImportTiny", 1).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    let usual = usual(&importing.options);
    importing.run_as(&usual).await.unwrap();
    let gone = async || {
        let report = importing.run_as(&usual).await.unwrap();
        let copied = &report.classes[class];
        (copied.deleted, copied.pending, copied.budget)
    };
    // One of one: all the class holds.
    weaviate.delete(class, 0).await;
    assert_eq!(gone().await, (0, 1, Some(0)));
    assert!(importing.get(0).await.is_some());
    // One of the four the ledger then has: note 0 or note 10, and no
    // more than one of them in a run.
    weaviate.add(class, 10..13).await;
    assert_eq!(gone().await, (0, 1, Some(0)));
    weaviate.delete(class, 10).await;
    assert_eq!(gone().await, (0, 2, Some(1)));
    let all = Options {
        deletes: importing.options.deletes,
        ..usual.clone()
    };
    let report = importing.run_as(&all).await.unwrap();
    assert_eq!(counts(&report, class), ([0, 0, 2, 0, 0], [2; 3]));
    // Two of two.
    weaviate.delete(class, 11).await;
    weaviate.delete(class, 12).await;
    assert_eq!(gone().await, (0, 2, Some(1)));
    assert_eq!(importing.records.count(None).await.unwrap(), 2);
}

fn restart() {
    let name = std::env::var("VELLEV_WEAVIATE_CONTAINER").expect("VELLEV_WEAVIATE_CONTAINER");
    let done = std::process::Command::new("docker")
        .args(["restart", &name])
        .output()
        .unwrap();
    assert!(done.status.success());
}

async fn ready(weaviate: &Weaviate, class: &str) {
    loop {
        let url = format!("{}/v1/objects?class={class}&limit=1", weaviate.url);
        let ready = format!("{}/v1/.well-known/ready", weaviate.url);
        let up = weaviate
            .http
            .get(ready)
            .send()
            .await
            .is_ok_and(|a| a.status().is_success());
        let lists = weaviate
            .http
            .get(url)
            .send()
            .await
            .is_ok_and(|a| a.status().is_success());
        if up && lists {
            return;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_weaviate_restarted_under_a_catch_up_has_nothing_it_holds_deleted() {
    if std::env::var("VELLEV_WEAVIATE_CONTAINER").is_err() {
        return;
    }
    let Some(importing) = Importing::new("VellevImportRestart", 600).await else {
        return;
    };
    let (class, weaviate) = (importing.class, importing.weaviate.clone());
    importing.run().await.unwrap();
    let paced = Options {
        per_second: Some(300.0),
        confirm: Duration::from_secs(2),
        patience: Duration::from_secs(120),
        ..importing.options.clone()
    };
    let all = "consistency_level=ALL";
    for round in 0..9u64 {
        for n in 0..5 {
            weaviate.delete(class, round * 10 + n).await;
        }
        for n in 300..320 {
            weaviate.put(class, n, changed(n + round)).await;
        }
        let mut timed = None;
        match round {
            // Before the first asking, of all copies, for what is gone.
            0 => importing.proxy.before(0, &[all], |_| async {
                tokio::task::spawn_blocking(restart).await.unwrap()
            }),
            // Before the second round of asking.
            1 => importing.proxy.before(5, &[all], |_| async {
                tokio::task::spawn_blocking(restart).await.unwrap()
            }),
            // In the middle of the second round, not waited for.
            2 => importing.proxy.before(7, &[all], |_| async {
                tokio::task::spawn_blocking(restart);
            }),
            // In the middle of the first round, not waited for.
            3 => importing.proxy.before(2, &[all], |_| async {
                tokio::task::spawn_blocking(restart);
            }),
            // Before the check that follows the second round.
            4 => importing
                .proxy
                .before(2, &["/v1/.well-known/ready"], |_| async {
                    tokio::task::spawn_blocking(restart).await.unwrap()
                }),
            // While the walk lists.
            5 => importing
                .proxy
                .before(3, &strs(&importing.page(false)), |_| async {
                    tokio::task::spawn_blocking(restart);
                }),
            n => {
                let wait = Duration::from_millis(400 * (n - 5));
                timed = Some(tokio::spawn(async move {
                    tokio::time::sleep(wait).await;
                    tokio::task::spawn_blocking(restart).await.unwrap();
                }));
            }
        }
        let result = importing.run_as(&paced).await;
        if let Some(timed) = timed {
            timed.await.unwrap();
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
        ready(&weaviate, class).await;
        // A run a restart stops has failed, and that is all it has done.
        drop(result);
        importing.proxy.rules.lock().unwrap().clear();
        let theirs = weaviate.held(class).await;
        let ours = held(&importing.records).await;
        let lost: Vec<_> = theirs.keys().filter(|id| !ours.contains_key(id)).collect();
        assert!(
            lost.is_empty(),
            "round {round}: deleted though Weaviate holds them: {lost:?}"
        );
        // A quiet run then settles everything.
        let report = importing.run().await.unwrap();
        assert_eq!(report.outcome(true), Outcome::Settled, "round {round}");
        importing.assert_same().await;
    }
}

#[tokio::test]
async fn a_tenant_of_a_class_emptied_and_made_again_has_nothing_deleted_for_it() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let class = "VellevImportWiped";
    let weaviate = Weaviate::load(&url, "VellevImportWipedPlain", 1).await;
    weaviate.drop(class).await;
    let schema = json!({
        "class": class,
        "vectorizer": "none",
        "multiTenancyConfig": { "enabled": true },
        "properties": [{ "name": "title", "dataType": ["text"] }],
    });
    weaviate
        .send(Method::POST, "/v1/schema", Some(schema))
        .await;
    let tenants = format!("/v1/schema/{class}/tenants");
    weaviate
        .send(Method::POST, &tenants, Some(json!([{ "name": "t1" }])))
        .await;
    let add = async |notes: std::ops::Range<u64>| {
        for chunk in notes.collect::<Vec<_>>().chunks(300) {
            let objects: Vec<Value> = chunk
                .iter()
                .map(|n| json!({ "class": class, "tenant": "t1", "id": uuid(*n), "properties": { "title": format!("t {n}") }, "vector": vector(*n) }))
                .collect();
            weaviate
                .send(
                    Method::POST,
                    "/v1/batch/objects",
                    Some(json!({ "objects": objects })),
                )
                .await;
        }
    };
    add(0..100).await;
    let served = Served::new().await;
    let tenant = tenant(&served.url, "vellevimportwiped").await;
    let dir = tempfile::tempdir().unwrap();
    let options = Options {
        classes: vec![class.into()],
        deletes: Options::new(Default::default()).deletes,
        ..options(&dir.path().join("state.json"))
    };
    let source = Source {
        tenant: Some("t1".into()),
        ..source(&url)
    };
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert_eq!(report.classes[class].vellev, 100);
    // The tenant is deleted and made again, as an emptied Weaviate would have it, with one new object.
    weaviate
        .send(Method::DELETE, &tenants, Some(json!(["t1"])))
        .await;
    weaviate
        .send(Method::POST, &tenants, Some(json!([{ "name": "t1" }])))
        .await;
    add(500..501).await;
    for _ in 0..3 {
        let report = import::weaviate(&source, &tenant, &options).await.unwrap();
        let copied = &report.classes[class];
        assert_eq!((copied.deleted, copied.pending), (0, 100));
        assert_eq!(report.outcome(false), Outcome::Unsettled);
    }
    // The tenant's new objects are imported, and the ledger grows with
    // them: so does what a run may delete, which is why a class with
    // tenants, which Weaviate gives nothing to know it by, is guarded
    // by that alone.
    add(1000..2900).await;
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!(
        (copied.stored, copied.deleted, copied.pending),
        (1900, 0, 100)
    );
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    let copied = &report.classes[class];
    assert_eq!((copied.deleted, copied.pending), (100, 0));
}

#[tokio::test]
async fn a_collection_with_neither_ledger_nor_note_is_given_nothing_it_lacks() {
    let Some(importing) = Importing::new("VellevImportUnbound", 100).await else {
        return;
    };
    let class = importing.class;
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    let imports = CollectionName::new("vellev_imports").unwrap();
    importing.tenant.collection(&imports).drop().await.unwrap();
    std::fs::remove_file(&importing.options.state).unwrap();
    for n in [1, 9, 50, 99] {
        importing.records.delete(oid(n)).await.unwrap();
    }
    importing.weaviate.put(class, 2, changed(2)).await;
    for _ in 0..2 {
        let report = importing.run().await.unwrap();
        assert_eq!(withheld(&report, class), 5);
        for n in [1, 9, 50, 99] {
            assert!(importing.get(n).await.is_none(), "note {n} came back");
        }
        assert_eq!(importing.get(2).await.unwrap().properties["n"], json!(2));
    }
}

#[tokio::test]
async fn a_comparison_of_some_of_a_class_whose_ids_lie_together_looks_at_them() {
    let Some(importing) = Importing::new("VellevImportClustered", 0).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    // Ids that all begin 00000000-: as time-ordered or counted ids lie.
    for n in (0..800u64).step_by(8) {
        weaviate.post(class, n, object(n), vector(n)).await;
    }
    importing.run().await.unwrap();
    // Another writer spoils every object.
    for n in (0..800u64).step_by(8) {
        let patch = Changes::new(map(json!({ "n": -1 })));
        importing.records.patch(oid(n), &patch).await.unwrap();
    }
    let verify = |pages: usize| Options {
        mode: Mode::Verify { pages: Some(pages) },
        page: 10,
        ..importing.options.clone()
    };
    let report = importing.run_as(&verify(3)).await.unwrap();
    let verified = report.classes[class].verified.clone().unwrap();
    assert_eq!(
        (verified.looked, verified.different),
        (30, 30),
        "{verified:?}"
    );
    assert_eq!(report.outcome(false), Outcome::Unsettled);
    // Asked for more pages than there are, it looks at all of them.
    let report = importing.run_as(&verify(50)).await.unwrap();
    let verified = report.classes[class].verified.clone().unwrap();
    assert_eq!((verified.looked, verified.different), (100, 100));
    // What Weaviate has gained since, among the ids the ledger has, is
    // looked at once, with the page of the ledger it falls on.
    for n in (4..164u64).step_by(8) {
        weaviate.post(class, n, object(n), vector(n)).await;
    }
    let report = importing.run_as(&verify(50)).await.unwrap();
    let verified = report.classes[class].verified.clone().unwrap();
    assert_eq!(
        (verified.looked, verified.different, verified.unaccounted),
        (120, 100, 20)
    );

    // With nothing in the ledger to look at, of a class that holds
    // objects, it has compared nothing, and says so.
    importing.ledger().drop().await.unwrap();
    let report = importing.run_as(&verify(3)).await;
    assert!(report.is_err() || report.unwrap().outcome(false) == Outcome::Unsettled);
}

#[tokio::test]
async fn a_taking_up_resumed_gives_the_rest_nothing_it_lacks_either() {
    let Some(importing) = Importing::new("VellevImportAdoptResume", 120).await else {
        return;
    };
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    let deleted = [9, 55, 98, 99, 115];
    for n in deleted {
        importing.records.delete(oid(n)).await.unwrap();
    }
    let page = importing.page(true);
    importing
        .proxy
        .answer(0, &strs(&page), StatusCode::BAD_REQUEST, json!({}));
    let stopped = importing.run().await.unwrap_err();
    assert!(stopped.to_string().contains("400"), "{stopped}");
    assert_eq!(importing.ledger().count(None).await.unwrap(), 50);
    let report = importing.run().await.unwrap();
    assert_eq!(withheld(&report, importing.class), 5);
    for n in deleted {
        assert!(importing.get(n).await.is_none(), "note {n} came back");
    }
}

#[tokio::test]
async fn what_is_gone_is_asked_for_twice_of_every_copy_a_pause_apart_with_weaviate_ready_each_time()
{
    let Some(importing) = Importing::new("VellevImportAsked", 10).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    weaviate.delete(class, 3).await;
    weaviate.delete(class, 6).await;
    let pause = Duration::from_millis(400);
    let options = Options {
        confirm: pause,
        ..importing.options.clone()
    };
    importing.proxy.listen();
    importing.front.listen();
    let report = importing.run_as(&options).await.unwrap();
    assert_eq!(counts(&report, class), ([0, 0, 2, 0, 0], [8; 3]));

    let when = |asked: Vec<(std::time::Instant, String)>| {
        asked.into_iter().map(|(when, _)| when).collect::<Vec<_>>()
    };
    let singly = importing.proxy.asked(&format!("GET /v1/objects/{class}/"));
    assert!(
        singly
            .iter()
            .all(|(_, line)| line.contains("consistency_level=ALL")),
        "of every copy: {singly:?}"
    );
    let asks = when(singly);
    assert_eq!(asks.len(), 4, "each of the two, twice");
    let listed = when(
        importing
            .proxy
            .asked(&format!("GET /v1/objects?class={class}")),
    );
    let shards = when(importing.proxy.asked(&format!("/v1/schema/{class}/shards")));
    let ready = when(importing.proxy.asked("/v1/.well-known/ready"));
    let deletes = when(
        importing
            .front
            .asked(&format!("/collections/{class}/objects/")),
    );
    let between = |from: std::time::Instant, to: std::time::Instant| {
        let within = |at: &&std::time::Instant| from < **at && **at < to;
        shards
            .iter()
            .filter(within)
            .count()
            .min(ready.iter().filter(within).count())
    };
    // Weaviate says it is ready, and the class's shards loaded, after
    // the walk and before the first asking; before the second; and
    // after the second, before the collection is read.
    assert_eq!(between(*listed.last().unwrap(), asks[0]), 1);
    assert_eq!(between(asks[1], asks[2]), 1);
    assert_eq!(between(asks[3], deletes[0]), 1);
    assert!(asks[2] - asks[1] >= pause, "a pause apart");
    // And nothing is asked of Weaviate between reading an object and
    // deleting it.
    let last = *deletes.last().unwrap();
    let asked = importing.proxy.asked("");
    let meanwhile = asked
        .iter()
        .filter(|(at, _)| deletes[0] < *at && *at < last);
    assert_eq!(meanwhile.count(), 0);
}

#[tokio::test]
async fn a_weaviate_that_is_not_ready_is_not_believed_about_a_listing_or_an_object_it_lacks() {
    let Some(importing) = Importing::new("VellevImportUnready", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();
    let proxy = &importing.proxy;
    let (missing, unready) = (StatusCode::NOT_FOUND, StatusCode::SERVICE_UNAVAILABLE);
    let ready = ["/v1/.well-known/ready"];

    // It answers the listing as it would of a class it does not have,
    // and then that it is not ready: the listing is asked for again.
    let first = importing.first();
    proxy.answer(0, &strs(&first), missing, json!({}));
    proxy.answer(0, &ready, unready, json!({}));
    assert_eq!(importing.counts().await, ([0; 5], [20; 3]));
    assert!(proxy.spent());
    // Ready, it is believed: the class is gone.
    proxy.answer(0, &strs(&first), missing, json!({}));
    let refused = importing.run().await.unwrap_err();
    assert!(refused.to_string().contains("has no class"), "{refused}");

    // It leaves note 5 out of its listing, and answers each asking for
    // it as if it held none, but then says it is not ready: the note is
    // asked for again, and found.
    let mut listed = weaviate.listed(class).await;
    let objects = listed["objects"].as_array_mut().unwrap();
    objects.retain(|object| object["id"] != json!(uuid(5)));
    proxy.answer(0, &strs(&first), StatusCode::OK, listed);
    for _ in 0..2 {
        proxy.answer(0, &[&importing.object(5)], missing, json!({}));
    }
    // The first time it is asked whether it is ready is before the note
    // is asked for.
    proxy.answer(1, &ready, unready, json!({}));
    assert_eq!(importing.counts().await, ([0; 5], [20; 3]));
    assert!(importing.get(5).await.is_some());
    proxy.rules.lock().unwrap().clear();

    // A class with no shard to be seen is not one whose shards are all
    // loaded.
    weaviate.delete(class, 6).await;
    let impatient = Options {
        patience: Duration::from_secs(1),
        ..importing.options.clone()
    };
    let shards = format!("GET /v1/schema/{class}/shards");
    for skip in [1, 0, 0, 0, 0, 0, 0, 0] {
        proxy.answer(skip, &[&shards], StatusCode::OK, json!([]));
    }
    let refused = importing.run_as(&impatient).await.unwrap_err();
    assert!(
        refused.to_string().contains("was not ready to say"),
        "{refused}"
    );
    assert!(importing.get(6).await.is_some());
}

#[tokio::test]
async fn a_page_taken_up_one_object_at_a_time_is_still_given_nothing_it_lacks() {
    let Some(importing) = Importing::new("VellevImportAdoptSingly", 60).await else {
        return;
    };
    let class = importing.class;
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    importing.records.delete(oid(4)).await.unwrap();
    // An entry appears for note 8 once the collection has been read: the
    // request that takes up the page is refused, and its objects are
    // settled one at a time.
    let ledger = importing.ledger();
    let read = format!("/collections/{class}/list");
    importing
        .front
        .after(0, &["POST", &read], move || async move {
            let entry = Content::new(map(json!({ "hash": "another import's" })));
            ledger.put(oid(8), &entry).await.unwrap();
        });
    let report = importing.run().await.unwrap();
    assert!(importing.front.spent());
    assert_eq!(withheld(&report, class), 1);
    assert!(importing.get(4).await.is_none(), "note 4 came back");
}

#[tokio::test]
async fn a_run_of_every_class_passes_over_one_whose_import_was_ended() {
    let Some(importing) = Importing::new("VellevImportEndedOne", 5).await else {
        return;
    };
    let (class, other) = (importing.class, "VellevImportEndedOther");
    let weaviate = Weaviate::load(&importing.weaviate.url, other, 5).await;
    let both = Options {
        classes: vec![class.into(), other.into()],
        ..importing.options.clone()
    };
    importing.run_as(&both).await.unwrap();
    import::finalise(&importing.tenant, &[class.to_string()])
        .await
        .unwrap();
    weaviate.add(other, 10..11).await;
    weaviate.add(class, 10..11).await;

    // Weaviate holds other tests' classes too: here it says it has
    // these two.
    let schema = async |class: &str| {
        let path = format!("/v1/schema/{class}");
        weaviate.send(Method::GET, &path, None).await
    };
    let classes = json!({ "classes": [schema(class).await, schema(other).await] });
    let listing = ["GET /v1/schema", "!/v1/schema/"];
    let every = Options {
        classes: vec![],
        ..importing.options.clone()
    };
    importing.proxy.answer(0, &listing, StatusCode::OK, classes);
    let report = importing.run_as(&every).await.unwrap();
    assert!(importing.proxy.spent());
    assert_eq!(report.ended, [class]);
    assert_eq!(counts(&report, other), ([1, 0, 0, 0, 0], [6; 3]));
    assert_eq!(report.outcome(true), Outcome::Settled);
    assert_eq!(importing.records.count(None).await.unwrap(), 5);
    // Named, it is refused.
    let refused = importing.run().await.unwrap_err();
    assert!(refused.to_string().contains("was ended"), "{refused}");
}

#[tokio::test]
async fn a_class_made_again_under_its_first_walk_is_not_gone_on_storing() {
    let Some(mut importing) = Importing::new("VellevImportRemadeWalk", 20).await else {
        return;
    };
    importing.options.page = 5;
    let class = importing.class;
    let again = importing.weaviate.clone();
    let page = importing.page(true);
    importing
        .proxy
        .before(1, &strs(&page), move |_| async move {
            again.drop(class).await;
            again.make(class).await;
            for n in 0..20 {
                again.post(class, n, changed(n), vector(n)).await;
            }
        });
    let refused = importing.run().await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("has been made again"),
        "{refused}"
    );
    let stored = held(&importing.records).await;
    assert_eq!(
        stored.len(),
        10,
        "the two pages of the class that was begun"
    );
    for n in 0..20 {
        if let Some((properties, _)) = stored.get(&oid(n)) {
            assert_eq!(properties["n"], json!(n as f64));
        }
    }
}

#[tokio::test]
async fn a_page_taken_up_is_read_whole_however_much_else_the_collection_holds_there() {
    let Some(importing) = Importing::new("VellevImportAdoptCrowded", 60).await else {
        return;
    };
    let class = importing.class;
    importing.run().await.unwrap();
    importing.ledger().drop().await.unwrap();
    // Another writer's objects lie among the ids of Weaviate's first
    // page, so that the collection holds more there than one reading of
    // it brings.
    for n in (1_000..1_040).step_by(4) {
        importing.records.put(oid(n), &foreign(n)).await.unwrap();
    }
    let report = importing.run().await.unwrap();
    assert_eq!(
        (counts(&report, class), withheld(&report, class)),
        (([0; 5], [60, 60, 70]), 0)
    );
}
