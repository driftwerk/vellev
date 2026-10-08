//! Copying a Weaviate into a tenant. Runs only when `VELLEV_WEAVIATE_URL`
//! names a Weaviate to load, such as the one in `docker-compose.dev.yml`.
//!
//! Where a test needs Weaviate to change at a given moment of an import,
//! or to fail, the import reaches it through a [`Proxy`] that does so
//! before passing a chosen request on.

mod common;

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::response::IntoResponse;
use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Map, Value};
use vellev::import::{self, Options, Report, Source};
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

fn object(n: u64) -> Value {
    json!({
        "title": format!("note {n} about {}", ["foxes", "hens", "owls"][n as usize % 3]),
        "tags": [format!("t{}", n % 4)],
        "n": n,
        "score": n as f64 / 8.0,
        "seen": n.is_multiple_of(2),
        "at": format!("2026-05-{:02}T09:00:00Z", 1 + n % 28),
    })
}

fn vector(n: u64) -> Vec<f32> {
    vec![1.0, (n % 7) as f32, (n % 11) as f32, 0.5]
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

    /// A class `class` holding notes `0..objects`.
    async fn load(url: &str, class: &str, objects: u64) -> Self {
        let weaviate = Self {
            url: url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        };
        weaviate.drop(class).await;
        let schema = Self::schema(class);
        weaviate
            .send(Method::POST, "/v1/schema", Some(schema))
            .await;
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

    /// Store a new note `n` with these properties.
    async fn post(&self, class: &str, n: u64, properties: Value) {
        let body =
            json!({ "class": class, "id": uuid(n), "properties": properties, "vector": vector(n) });
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

/// Something to do before a request is passed on, given the request's
/// method and path.
struct Rule {
    /// The request has every one of these in its method and path, and
    /// none of those that begin with `!`.
    has: Vec<String>,
    /// How many such requests pass untouched first.
    skip: usize,
    act: Box<dyn FnOnce(String) -> Doing + Send>,
    /// What to answer in place of passing it on.
    answer: Option<(StatusCode, Value)>,
}

/// Passes requests on to a Weaviate, doing what its rules say first.
#[derive(Clone)]
struct Proxy {
    url: String,
    to: String,
    http: reqwest::Client,
    rules: Arc<Mutex<Vec<Rule>>>,
}

impl Proxy {
    async fn to(weaviate: &Weaviate) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy = Self {
            url: format!("http://{}", listener.local_addr().unwrap()),
            to: weaviate.url.clone(),
            http: weaviate.http.clone(),
            rules: Arc::default(),
        };
        let router = axum::Router::new()
            .fallback(Self::relay)
            .with_state(proxy.clone());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        proxy
    }

    /// Before the request after the first `skip` that have all of `has`
    /// in their method and path, do `act`.
    fn before<F: Future<Output = ()> + Send + 'static>(
        &self,
        skip: usize,
        has: &[&str],
        act: impl FnOnce(String) -> F + Send + 'static,
    ) {
        self.rule(skip, has, Box::new(|line| Box::pin(act(line))), None);
    }

    /// Answer such a request with `status` and `body`, once.
    fn answer(&self, skip: usize, has: &[&str], status: StatusCode, body: Value) {
        let nothing = Box::new(|_| Box::pin(async {}) as Doing);
        self.rule(skip, has, nothing, Some((status, body)));
    }

    fn rule(
        &self,
        skip: usize,
        has: &[&str],
        act: Box<dyn FnOnce(String) -> Doing + Send>,
        answer: Option<(StatusCode, Value)>,
    ) {
        self.rules.lock().unwrap().push(Rule {
            has: has.iter().map(|has| has.to_string()).collect(),
            skip,
            act,
            answer,
        });
    }

    /// Whether every rule has been used.
    fn spent(&self) -> bool {
        self.rules.lock().unwrap().is_empty()
    }

    async fn relay(
        axum::extract::State(proxy): axum::extract::State<Proxy>,
        request: axum::extract::Request,
    ) -> axum::response::Response {
        let (parts, body) = request.into_parts();
        let path = parts.uri.path_and_query().map_or("", |path| path.as_str());
        let line = format!("{} {path}", parts.method);
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
        if let Some(rule) = due {
            (rule.act)(line.clone()).await;
            if let Some((status, body)) = rule.answer {
                return (status, axum::Json(body)).into_response();
            }
        }
        let body = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        let mut passed = proxy
            .http
            .request(parts.method, format!("{}{path}", proxy.to))
            .body(body);
        if let Some(kind) = parts.headers.get(reqwest::header::CONTENT_TYPE) {
            passed = passed.header(reqwest::header::CONTENT_TYPE, kind);
        }
        let answer = passed.send().await.unwrap();
        (answer.status(), answer.bytes().await.unwrap()).into_response()
    }
}

/// The id a request for a page of objects names as the one to start
/// after.
fn after(line: &str) -> &str {
    let (_, after) = line.split_once("after=").expect("a page after another");
    after.split('&').next().unwrap()
}

fn options(state: &std::path::Path) -> Options {
    Options {
        classes: vec![],
        embeddings: HashMap::new(),
        vectors: HashMap::new(),
        dimensions: HashMap::new(),
        created_as: Some("createdAt".into()),
        updated_as: None,
        page: 50,
        batch_bytes: MAX_BODY / 2,
        per_second: None,
        // Weaviate runs on this machine's clock; a margin would fetch
        // again whatever was written just before a catch-up.
        skew: Duration::ZERO,
        state: state.to_path_buf(),
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

async fn tenant(served: &Served, name: &str) -> Tenant {
    let options = ClientOptions::default().batch_bytes(MAX_BODY / 2);
    let client = Client::connect(&[&served.url], KEY, options).unwrap();
    client.tenant(&TenantId::new(name).unwrap())
}

/// One class of a Weaviate and the tenant it is imported into.
struct Importing {
    class: &'static str,
    weaviate: Weaviate,
    proxy: Proxy,
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
        let tenant = tenant(&served, &class.to_lowercase()).await;
        let dir = tempfile::tempdir().unwrap();
        let options = Options {
            classes: vec![class.into()],
            ..options(&dir.path().join("state.json"))
        };
        Some(Self {
            class,
            proxy: Proxy::to(&weaviate).await,
            weaviate,
            _served: served,
            records: tenant.collection(&CollectionName::new(class).unwrap()),
            tenant,
            dir,
            options,
        })
    }

    /// Import through the proxy.
    async fn run(&self) -> eyre::Result<Report> {
        import::weaviate(&source(&self.proxy.url), &self.tenant, &self.options).await
    }

    /// Import, and answer with what was stored, changed, deleted and
    /// kept, and how many objects Weaviate, the ledger and the collection
    /// hold.
    async fn counts(&self) -> ([u64; 4], [u64; 3]) {
        let report = self.run().await.unwrap();
        let copied = &report.classes[self.class];
        (
            [copied.stored, copied.changed, copied.deleted, copied.kept],
            [copied.weaviate, copied.imported, copied.vellev],
        )
    }

    fn ledger(&self) -> Records {
        let name = format!("vellev_imported_{}", self.class);
        self.tenant.collection(&CollectionName::new(name).unwrap())
    }

    async fn get(&self, n: u64) -> Option<vellev_api::Stored> {
        self.records.get(oid(n)).await.unwrap()
    }

    /// A page of objects of this class, as the import asks Weaviate for
    /// one: with vectors on its first pass, without to catch up.
    fn page(&self, vectors: bool) -> Vec<String> {
        let page = format!("GET /v1/objects?class={}&", self.class);
        let vectors = if vectors { "" } else { "!" };
        vec![page, "after=".into(), format!("{vectors}include=vector")]
    }

    fn object(&self, n: u64) -> String {
        format!("GET /v1/objects/{}/{}", self.class, uuid(n))
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

fn changed(n: u64) -> Value {
    json!({ "title": format!("note {n} changed"), "n": 1_000_000 + n })
}

fn foreign(n: u64) -> Content {
    let properties = json!({ "title": format!("another writer's {n}") });
    let mut content = Content::new(properties.as_object().unwrap().clone());
    content.vector = Some(vec![0.0, 0.0, 0.0, 1.0]);
    content
}

#[tokio::test]
async fn a_weaviate_class_is_copied_whole_with_its_ids_vectors_and_settings() {
    let Some(importing) = Importing::new("VellevImportWhole", OBJECTS).await else {
        return;
    };
    let report = importing.run().await.unwrap();
    assert!(report.matched());
    let copied = &report.classes[importing.class];
    assert_eq!(
        (
            copied.stored,
            copied.weaviate,
            copied.imported,
            copied.vellev
        ),
        (OBJECTS, OBJECTS, OBJECTS, OBJECTS)
    );
    assert_eq!((copied.changed, copied.deleted, copied.kept), (0, 0, 0));

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
        ([0, 0, 0, 0], [OBJECTS; 3]),
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
    assert!(report.matched());
    let copied = &report.classes[class];
    assert_eq!(copied.stored, OBJECTS / 2);
    assert_eq!((copied.changed, copied.deleted, copied.kept), (1, 1, 0));
    assert_eq!(
        [copied.weaviate, copied.imported, copied.vellev],
        [OBJECTS - 1; 3]
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
        ([0, 0, 0, 0], [OBJECTS - 1; 3]),
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
    let tenant = tenant(&served, "mixed").await;
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
    assert!(report.matched(), "{report:?}");
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

    // A named vector that changes is caught up with as any other.
    let renamed = json!({ "class": "VellevImportNamed", "id": uuid(3), "properties": { "title": "renamed" }, "vectors": { "default": vector(4) } });
    let path = format!("/v1/objects/VellevImportNamed/{}", uuid(3));
    weaviate.send(Method::PUT, &path, Some(renamed)).await;
    let again = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert!(again.matched(), "{again:?}");
    assert_eq!(again.classes["VellevImportNamed"].changed, 1);
    let stored = named.get(oid(3)).await.unwrap().unwrap();
    assert_eq!(stored.vector, Some(vector(4)));
}

#[tokio::test]
async fn a_catch_up_replaces_and_deletes_only_what_the_import_left_and_nobody_has_changed() {
    let Some(importing) = Importing::new("VellevImportOwned", 40).await else {
        return;
    };
    let (class, weaviate, records) = (importing.class, &importing.weaviate, &importing.records);
    assert_eq!(importing.counts().await, ([40, 0, 0, 0], [40; 3]));

    // Another writer stores a memory of its own, changes one the import
    // brought, replaces another and deletes two more.
    records.put(oid(1_000), &foreign(1_000)).await.unwrap();
    let patch = Changes::new(json!({ "n": -2 }).as_object().unwrap().clone());
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
    let copied = &report.classes[class];
    assert_eq!(
        [copied.stored, copied.changed, copied.deleted, copied.kept],
        [1, 1, 1, 3],
        "note 50 stored, 5 changed, 1 deleted; 2, 3 and 4 kept as found"
    );
    assert_eq!([copied.weaviate, copied.imported, copied.vellev], [39; 3]);
    assert!(copied.caught_up() && !copied.untouched() && !report.matched());

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

    // Run again, the import finds nothing of its own to do, and leaves
    // the other writer's objects where they are.
    assert_eq!(importing.counts().await, ([0, 0, 0, 0], [39; 3]));
    untouched(importing.get(1_000).await, &foreign(1_000));
    untouched(importing.get(3).await, &foreign(3));

    // Weaviate changes an object the other writer deleted: it stays
    // deleted.
    weaviate.put(class, 6, changed(6)).await;
    assert_eq!(importing.counts().await, ([0, 0, 0, 1], [39; 3]));
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
    assert_eq!(importing.counts().await, ([10, 0, 0, 0], [10; 3]));

    // Weaviate gains notes 10..30. A mirror has already stored 10..20 as
    // Weaviate has them and 20..25 otherwise, as it would a change the
    // import has yet to see.
    weaviate.add(class, 10..30).await;
    for n in 10..20 {
        let mut content = Content::new(object(n).as_object().unwrap().clone());
        content.vector = Some(vector(n));
        records.put(oid(n), &content).await.unwrap();
    }
    for n in 20..25 {
        records.put(oid(n), &foreign(n)).await.unwrap();
    }
    // The import begins again from nothing, as one whose state was lost:
    // every page holds objects that are already there.
    std::fs::remove_file(&importing.options.state).unwrap();
    assert_eq!(
        importing.counts().await,
        ([5, 0, 0, 5], [30; 3]),
        "25..30 stored; 20..25 kept as the mirror has them"
    );
    for n in 20..25 {
        assert_eq!(
            importing.get(n).await.unwrap().properties,
            foreign(n).properties
        );
    }
    assert_eq!(importing.counts().await, ([0, 0, 0, 0], [30; 3]));

    // What the mirror stored as Weaviate has it goes when Weaviate's
    // does; what it stored otherwise is left for it to delete.
    weaviate.delete(class, 12).await;
    weaviate.delete(class, 22).await;
    assert_eq!(importing.counts().await, ([0, 0, 1, 1], [28, 28, 29]));
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
    weaviate.put(class, 4, changed(4)).await;

    // The import asks Weaviate whether it still has the class once it
    // has read the object it is about to delete: the other writer's
    // delete of note 6, and then its change to note 3, land between
    // that read and the import's delete.
    assert!(oid(6) < oid(3), "note 6 is come to first");
    let schema = format!("GET /v1/schema/{class}");
    let records = importing.records.clone();
    importing.proxy.before(0, &[&schema], move |_| async move {
        records.delete(oid(6)).await.unwrap();
    });
    let records = importing.records.clone();
    importing.proxy.before(0, &[&schema], move |_| async move {
        let patch = Changes::new(json!({ "n": 99 }).as_object().unwrap().clone());
        records.patch(oid(3), &patch).await.unwrap();
    });
    assert_eq!(importing.counts().await, ([0, 1, 0, 1], [8, 8, 9]));
    assert!(importing.proxy.spent());
    assert!(importing.get(6).await.is_none());
    assert_eq!(importing.get(3).await.unwrap().properties["n"], json!(99));
    assert_eq!(
        importing.get(4).await.unwrap().properties["n"],
        json!(1_000_004)
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
    weaviate.post(class, 20, newcomer).await;

    let report = importing.run().await.unwrap();
    assert!(report.matched(), "{report:?}");
    let copied = &report.classes[class];
    assert_eq!((copied.stored, copied.changed), (1, 1));
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
    // another size, is refused, and says both.
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
    let refused = importing.run().await.unwrap_err();
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
    let Some(importing) = Importing::new("VellevImportVanished", 10).await else {
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
    let copied = &report.classes[class];
    assert_eq!(
        [copied.stored, copied.changed, copied.deleted, copied.kept],
        [0, 0, 1, 0]
    );
    assert_eq!([copied.weaviate, copied.imported, copied.vellev], [9; 3]);
    assert!(report.matched());
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
    // pass's third page, and on the catch-up's second and fourth.
    let disturb = |vectors: bool, skip: usize, nth: u64| {
        let weaviate = weaviate.clone();
        let page = importing.page(vectors);
        let page: Vec<&str> = page.iter().map(String::as_str).collect();
        importing.proxy.before(skip, &page, move |line| async move {
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
        copied.caught_up(),
        copied.weaviate == copied.imported,
        "a run says whether it caught up"
    );
    assert_eq!(copied.kept, 0);

    let second = importing.run().await.unwrap();
    assert!(second.matched(), "{second:?}");
    importing.assert_same().await;
    let held = weaviate.held(class).await;
    assert!(
        (1_000..1_008).all(|n| held.contains_key(&oid(n))) && !held.contains_key(&oid(150)),
        "Weaviate was written to"
    );
    assert_eq!(held[&oid(70)].0["n"], json!(1_000_070.0));
    assert_eq!(
        importing.counts().await,
        ([0, 0, 0, 0], [held.len() as u64; 3]),
        "and then there is nothing more to do"
    );
}

#[tokio::test]
async fn nothing_is_deleted_for_a_class_weaviate_no_longer_has_or_has_nothing_of() {
    let Some(importing) = Importing::new("VellevImportDropped", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, importing.weaviate.clone());
    importing.run().await.unwrap();

    // The class is dropped just as the catch-up asks for an object it
    // did not list: Weaviate answers as it would of a deleted object.
    weaviate.delete(class, 5).await;
    let fetch = importing.object(5);
    let dropping = weaviate.clone();
    importing.proxy.before(0, &[&fetch], move |_| async move {
        dropping.drop(class).await;
    });
    let refused = importing.run().await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("no longer has class"),
        "{refused}"
    );
    assert_eq!(importing.records.count(None).await.unwrap(), 20);

    // Made again and empty, as a Weaviate begun afresh at the same
    // address would have it, it is not taken to have deleted everything.
    let schema = Weaviate::schema(class);
    weaviate
        .send(Method::POST, "/v1/schema", Some(schema))
        .await;
    let refused = importing.run().await.unwrap_err();
    assert!(
        refused.to_string().contains("holds nothing of class"),
        "{refused}"
    );
    assert_eq!(importing.records.count(None).await.unwrap(), 20);
    assert_eq!(importing.ledger().count(None).await.unwrap(), 20);
}

#[tokio::test]
async fn a_weaviate_that_turns_requests_away_or_fails_for_a_while_is_asked_again() {
    let Some(importing) = Importing::new("VellevImportRetried", 120).await else {
        return;
    };
    let page = importing.page(true);
    let page: Vec<&str> = page.iter().map(String::as_str).collect();
    for status in [
        StatusCode::TOO_MANY_REQUESTS,
        StatusCode::SERVICE_UNAVAILABLE,
    ] {
        importing.proxy.answer(0, &page, status, json!({}));
    }
    assert_eq!(importing.counts().await, ([120, 0, 0, 0], [120; 3]));
    assert!(importing.proxy.spent());
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
    let page: Vec<&str> = page.iter().map(String::as_str).collect();
    let unordered = json!({ "objects": [{ "id": uuid(3) }, { "id": uuid(2) }] });
    importing.proxy.answer(0, &page, StatusCode::OK, unordered);
    let refused = importing.run().await.unwrap_err();
    assert!(importing.proxy.spent());
    assert!(
        refused.to_string().contains("its ids are not in order"),
        "{refused}"
    );
    assert_eq!(importing.records.count(None).await.unwrap(), 120);
    assert_eq!(importing.counts().await, ([0, 0, 0, 0], [120; 3]));
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
    assert!(report.matched(), "{report:?}");
    assert_eq!(report.classes[class].deleted, 1);

    let otherwise = Options {
        created_as: None,
        ..importing.options.clone()
    };
    let refused = import::weaviate(&source(&importing.proxy.url), &importing.tenant, &otherwise)
        .await
        .unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("an import goes on as it began"),
        "{refused}"
    );
}

#[tokio::test]
async fn a_collection_dropped_is_imported_again_from_the_start() {
    let Some(importing) = Importing::new("VellevImportAgain", 20).await else {
        return;
    };
    importing.run().await.unwrap();
    importing.records.drop().await.unwrap();
    assert_eq!(importing.counts().await, ([20, 0, 0, 0], [20; 3]));
    importing.assert_same().await;
}

#[tokio::test]
async fn without_its_ledger_an_import_takes_up_what_is_as_weaviate_has_it_and_deletes_nothing() {
    let Some(importing) = Importing::new("VellevImportLedgerless", 20).await else {
        return;
    };
    let (class, weaviate) = (importing.class, &importing.weaviate);
    importing.run().await.unwrap();

    // As a collection imported before there were ledgers: the import
    // cannot know which of its objects it brought.
    importing.ledger().drop().await.unwrap();
    weaviate.put(class, 2, changed(2)).await;
    weaviate.delete(class, 3).await;
    assert_eq!(
        importing.counts().await,
        ([0, 0, 0, 1], [19, 19, 20]),
        "note 2 is not as Weaviate has it, and kept; note 3 is nobody's to delete"
    );
    assert_eq!(importing.get(2).await.unwrap().properties["n"], json!(2));
    assert!(importing.get(3).await.is_some());

    // What it took up is the import's from then on.
    weaviate.put(class, 7, changed(7)).await;
    weaviate.delete(class, 8).await;
    assert_eq!(importing.counts().await, ([0, 1, 1, 0], [18, 18, 19]));
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
    let tenant = tenant(&served, "embedded").await;
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
    assert!(report.matched(), "{report:?}");
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
    assert!(report.matched(), "{report:?}");
    let copied = &report.classes[class];
    assert_eq!((copied.changed, copied.deleted, copied.kept), (1, 1, 0));
    let stored = records.get(oid(2)).await.unwrap().unwrap();
    assert_eq!(stored.properties["title"], json!("note 2 changed"));
    assert_eq!(stored.vector, embedded("note 2 changed"));
}

#[tokio::test]
async fn a_rate_that_is_no_number_or_none_at_all_is_refused() {
    let client = Client::connect(&["http://127.0.0.1:9"], KEY, ClientOptions::default()).unwrap();
    let tenant = client.tenant(&TenantId::new("unreached").unwrap());
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

#[tokio::test]
async fn a_collection_made_anew_is_copied_by_the_page_whatever_the_state_file_says() {
    let Some(importing) = Importing::new("VellevImportAnew", 20).await else {
        return;
    };
    importing.run().await.unwrap();
    importing.records.drop().await.unwrap();

    // The state file says the class was copied whole. Believed, it would
    // leave every object to the catch-up, to fetch one at a time.
    let singly = format!("GET /v1/objects/{}/", importing.class);
    importing.proxy.before(0, &[&singly], |_| async {});
    assert_eq!(importing.counts().await, ([20, 0, 0, 0], [20; 3]));
    assert!(!importing.proxy.spent(), "no object was fetched by itself");
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
    assert_eq!(importing.counts().await, ([0, 1, 0, 0], [10; 3]));
    importing.assert_same().await;
}

#[tokio::test]
async fn the_object_with_the_last_id_of_all_is_deleted_when_weaviate_deletes_it() {
    let Some(importing) = Importing::new("VellevImportLast", 10).await else {
        return;
    };
    importing.run().await.unwrap();
    let last = (0..10).max_by_key(|n| oid(*n)).unwrap();
    importing.weaviate.delete(importing.class, last).await;
    assert_eq!(importing.counts().await, ([0, 0, 1, 0], [9; 3]));
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
    assert_eq!(importing.counts().await, ([0, 0, 0, 0], [10, 10, 0]));
}
