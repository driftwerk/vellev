//! Copying a Weaviate into a tenant. Runs only when `VELLEV_WEAVIATE_URL`
//! names a Weaviate to load, such as the one in `docker-compose.dev.yml`.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use common::*;
use serde_json::{json, Value};
use vellev::import::{self, Options, Source};
use vellev_api::Keyword;
use vellev_client::{Client, Options as ClientOptions};
use vellev_core::{CollectionName, ObjectId, TenantId};

const OBJECTS: u64 = 1_200;

fn uuid(n: u64) -> String {
    id(n)
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

struct Weaviate {
    url: String,
    http: reqwest::Client,
}

impl Weaviate {
    async fn send(&self, method: reqwest::Method, path: &str, body: Option<Value>) -> Value {
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

    /// A class `class` holding notes `0..objects`.
    async fn load(url: &str, class: &str, objects: u64) -> Self {
        let weaviate = Self {
            url: url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        };
        let _ = weaviate
            .http
            .delete(format!("{}/v1/schema/{class}", weaviate.url))
            .send()
            .await;
        let schema = json!({
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
        });
        weaviate
            .send(reqwest::Method::POST, "/v1/schema", Some(schema))
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
                reqwest::Method::POST,
                "/v1/batch/objects",
                Some(json!({ "objects": objects })),
            )
            .await;
        }
    }
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
        per_second: None,
        // Weaviate runs on this machine's clock; a margin would store
        // again whatever was written just before a catch-up.
        skew: Duration::ZERO,
        state: state.to_path_buf(),
    }
}

async fn tenant(served: &Served, name: &str) -> vellev_client::Tenant {
    let options = ClientOptions::default().batch_bytes(MAX_BODY / 2);
    let client = Client::connect(&[&served.url], KEY, options).unwrap();
    client.tenant(&TenantId::new(name).unwrap())
}

fn oid(n: u64) -> ObjectId {
    serde_json::from_value(json!(uuid(n))).unwrap()
}

#[tokio::test]
async fn a_weaviate_class_is_copied_whole_with_its_ids_vectors_and_settings() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let class = "VellevImportWhole";
    Weaviate::load(&url, class, OBJECTS).await;
    let served = Served::new().await;
    let tenant = tenant(&served, "imported").await;
    let dir = tempfile::tempdir().unwrap();
    let options = Options {
        classes: vec![class.into()],
        ..options(&dir.path().join("state.json"))
    };
    let source = Source {
        url: url.clone(),
        key: None,
        tenant: None,
    };
    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert!(report.matched());
    let copied = &report.classes[class];
    assert_eq!(
        (copied.stored, copied.weaviate, copied.vellev),
        (OBJECTS, OBJECTS, OBJECTS)
    );
    assert_eq!(copied.changed, 0);
    assert_eq!(copied.deleted, 0);

    let records = tenant.collection(&CollectionName::new(class).unwrap());
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
    let found = records.keyword(&Keyword::new("owls", 5)).await.unwrap();
    assert_eq!(found.results.len(), 5);

    let again = import::weaviate(&source, &tenant, &options).await.unwrap();
    let copied = &again.classes[class];
    assert_eq!(
        (copied.stored, copied.changed, copied.deleted),
        (0, 0, 0),
        "a finished import, caught up, has nothing more to do"
    );
}

#[tokio::test]
async fn an_import_stopped_part_way_resumes_and_catches_up_with_what_changed_meanwhile() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let class = "VellevImportResumed";
    let weaviate = Weaviate::load(&url, class, OBJECTS / 2).await;
    let served = Served::new().await;
    let tenant = tenant(&served, "resumed").await;
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state.json");
    let options = Options {
        classes: vec![class.into()],
        ..options(&state)
    };
    let source = Source {
        url,
        key: None,
        tenant: None,
    };
    import::weaviate(&source, &tenant, &options).await.unwrap();

    // Weaviate goes on taking writes: to objects already copied, and past
    // where the import, stopped as if part way, had reached.
    weaviate.add(class, OBJECTS / 2..OBJECTS).await;
    let path = |n: u64| format!("/v1/objects/{class}/{}", uuid(n));
    let changed = json!({ "class": class, "id": uuid(0), "properties": { "title": "changed", "n": 1_000_000 }, "vector": vector(0) });
    weaviate
        .send(reqwest::Method::PUT, &path(0), Some(changed))
        .await;
    weaviate.send(reqwest::Method::DELETE, &path(1), None).await;
    let mut progress: Value = serde_json::from_slice(&std::fs::read(&state).unwrap()).unwrap();
    assert_eq!(
        progress["classes"][class]["after"],
        json!(uuid(OBJECTS / 2 - 1))
    );
    progress["classes"][class]["done"] = json!(false);
    std::fs::write(&state, serde_json::to_vec(&progress).unwrap()).unwrap();

    let report = import::weaviate(&source, &tenant, &options).await.unwrap();
    assert!(report.matched());
    let copied = &report.classes[class];
    assert_eq!(copied.stored, OBJECTS / 2, "resumed past what was stored");
    assert_eq!((copied.changed, copied.deleted), (1, 1));
    assert_eq!((copied.weaviate, copied.vellev), (OBJECTS - 1, OBJECTS - 1));
    let records = tenant.collection(&CollectionName::new(class).unwrap());
    let changed = records.get(oid(0)).await.unwrap().unwrap();
    assert_eq!(changed.properties["n"], json!(1_000_000));
    assert!(records.get(oid(1)).await.unwrap().is_none());

    let other = Source {
        url: "http://elsewhere:8080".into(),
        ..source.clone()
    };
    let refused = import::weaviate(&other, &tenant, &options)
        .await
        .unwrap_err();
    assert!(
        refused
            .to_string()
            .contains("is the state of an import from"),
        "{refused}"
    );
}

#[tokio::test]
async fn a_named_vector_and_a_tenant_of_a_multi_tenant_class_are_copied_beside_a_plain_class() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        return;
    };
    let weaviate = Weaviate::load(&url, "VellevImportPlain", 10).await;
    for class in ["VellevImportNamed", "VellevImportTenanted"] {
        let _ = weaviate
            .http
            .delete(format!("{}/v1/schema/{class}", weaviate.url))
            .send()
            .await;
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
            .send(reqwest::Method::POST, "/v1/schema", Some(schema))
            .await;
    }
    weaviate
        .send(
            reqwest::Method::POST,
            "/v1/schema/VellevImportTenanted/tenants",
            Some(json!([{ "name": "t1" }, { "name": "t2" }])),
        )
        .await;
    for n in 0..7 {
        let named = json!({ "class": "VellevImportNamed", "id": uuid(n), "properties": { "title": "named" }, "vectors": { "default": vector(n) } });
        weaviate
            .send(reqwest::Method::POST, "/v1/objects", Some(named))
            .await;
        for (tenant, from) in [("t1", 0), ("t2", 100)] {
            let tenanted = json!({ "class": "VellevImportTenanted", "tenant": tenant, "id": uuid(from + n), "properties": { "title": tenant }, "vector": vector(n) });
            weaviate
                .send(reqwest::Method::POST, "/v1/objects", Some(tenanted))
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
    let untenanted = Source {
        url: url.clone(),
        key: None,
        tenant: None,
    };
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
}
