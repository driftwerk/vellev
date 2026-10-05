mod common;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use reqwest::StatusCode;
use serde_json::{json, Value};
use vellev_api::{
    Change, Changes, Code, Content, Hybrid, Keyword, Nearest, OnlyIf, Outcome, Page, MIN_POSITION,
};
use vellev_client::{Client, Error, Options};
use vellev_core::{Collection, CollectionName, ObjectId, Property, TenantId};
use vellev_index::filter::Filter;

fn quick() -> Options {
    Options {
        timeout: Duration::from_secs(5),
        attempts: 3,
        backoff: Duration::from_millis(150),
        ..Options::default()
    }
}

fn acme() -> TenantId {
    TenantId::new("acme").unwrap()
}

fn notes_named() -> CollectionName {
    CollectionName::new("notes").unwrap()
}

fn schema() -> Collection {
    serde_json::from_value(notes(supplied())).unwrap()
}

fn oid(n: u64) -> ObjectId {
    serde_json::from_value(json!(id(n))).unwrap()
}

fn content(title: &str, n: u64, vector: Option<[f32; 2]>) -> Content {
    Content {
        properties: json!({ "title": title, "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: vector.map(Vec::from),
    }
}

fn failed_with(error: &Error, expected: Code) -> bool {
    error.code() == Some(expected)
}

#[tokio::test]
async fn the_whole_api_is_reached_through_the_client() {
    let served = Served::new().await;
    let client = Client::connect(&[&served.url], KEY, quick()).unwrap();
    let tenant = client.tenant(&acme());
    let notes = tenant.collection(&notes_named());

    assert!(failed_with(
        &tenant.collections().await.unwrap_err(),
        Code::NotFound
    ));
    tenant.create().await.unwrap();
    tenant.create().await.unwrap();
    assert_eq!(client.tenants(None, 10).await.unwrap().tenants, [acme()]);
    assert_eq!(notes.schema().await.unwrap(), None);

    let created = notes.create(&schema()).await.unwrap();
    assert_eq!(created.outcomes, [Outcome::Applied]);
    assert_eq!(
        notes.create(&schema()).await.unwrap().outcomes,
        [Outcome::Unchanged]
    );
    assert_eq!(notes.schema().await.unwrap(), Some(schema()));
    assert_eq!(tenant.collections().await.unwrap(), [schema()]);
    let reviewed: Property =
        serde_json::from_value(json!({ "name": "reviewed", "data_type": "bool" })).unwrap();
    notes.add_property(&reviewed).await.unwrap();

    assert_eq!(notes.get(oid(1)).await.unwrap(), None);
    notes
        .put(oid(1), &content("a red fox", 1, Some([1.0, 0.0])))
        .await
        .unwrap();
    notes
        .put(oid(2), &content("a hen", 2, Some([0.0, 1.0])))
        .await
        .unwrap();
    let stored = notes.get(oid(1)).await.unwrap().unwrap();
    assert_eq!((stored.id, stored.vector), (oid(1), Some(vec![1.0, 0.0])));
    let patched = Changes {
        properties: json!({ "n": 5 }).as_object().unwrap().clone(),
        vector: None,
        only_if: Some(OnlyIf {
            property: "n".into(),
            equals: json!(1),
        }),
    };
    notes.patch(oid(1), &patched).await.unwrap();
    assert!(failed_with(
        &notes.patch(oid(1), &patched).await.unwrap_err(),
        Code::Conflict
    ));

    let changed = tenant
        .change(vec![
            Change::Increment {
                collection: notes_named(),
                id: oid(1),
                property: "n".into(),
                by: 2.0,
            },
            Change::Delete {
                collection: notes_named(),
                id: oid(9),
            },
        ])
        .await
        .unwrap();
    assert_eq!(
        changed.outcomes,
        [Outcome::Value { value: json!(7) }, Outcome::Unchanged]
    );
    assert_eq!(client.position(&acme()), Some(changed.position));

    let nearest = Nearest {
        vector: Some(vec![1.0, 0.0]),
        text: None,
        limit: 1,
        filter: None,
        oversample: None,
        vectors: true,
    };
    let found = notes.nearest(&nearest).await.unwrap();
    assert_eq!(
        (found.results[0].object.id, found.results[0].distance),
        (oid(1), Some(0.0))
    );
    assert_eq!(found.results[0].object.vector, Some(vec![1.0, 0.0]));
    let keyword = Keyword {
        text: "hen".into(),
        properties: vec![],
        limit: 5,
        filter: None,
        vectors: false,
    };
    let found = notes.keyword(&keyword).await.unwrap();
    assert_eq!(found.results.len(), 1);
    assert!(found.results[0].keyword.unwrap() > 0.0);
    let hybrid = Hybrid {
        text: "fox".into(),
        vector: Some(vec![1.0, 0.0]),
        alpha: None,
        fusion: Default::default(),
        properties: vec![],
        limit: 5,
        candidates: None,
        filter: None,
        oversample: None,
        vectors: false,
    };
    let fused = notes.hybrid(&hybrid).await.unwrap();
    assert_eq!(fused.results[0].object.id, oid(1));
    assert!(!fused.degraded);

    let page = Page {
        filter: None,
        after: None,
        limit: 1,
        vectors: false,
    };
    let first = notes.list(&page).await.unwrap();
    assert_eq!((first.objects.len(), first.next), (1, Some(oid(1))));
    let big = Filter::Gte {
        property: "n".into(),
        value: json!(5),
    };
    assert_eq!(notes.count(None).await.unwrap(), 2);
    assert_eq!(notes.count(Some(big)).await.unwrap(), 1);
    assert!(failed_with(
        &notes.list(&Page { limit: 0, ..page }).await.unwrap_err(),
        Code::Invalid
    ));

    notes.delete(oid(2)).await.unwrap();
    assert_eq!(notes.count(None).await.unwrap(), 1);
    notes.drop().await.unwrap();
    assert!(failed_with(
        &notes.count(None).await.unwrap_err(),
        Code::NotFound
    ));
    assert!(tenant.drop().await.unwrap());
    assert!(!tenant.drop().await.unwrap());
    assert_eq!(client.position(&acme()), None);

    let stranger = Client::connect(&[&served.url], "not-a-key", quick()).unwrap();
    let refused = stranger.tenants(None, 5).await.unwrap_err();
    assert!(matches!(
        refused,
        Error::Api {
            status: StatusCode::UNAUTHORIZED,
            code: Code::Unauthorized,
            ..
        }
    ));
    assert!(Client::connect(&[], KEY, quick()).is_err());
}

/// An address nothing listens on.
async fn nowhere() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    format!("http://{}", listener.local_addr().unwrap())
}

#[tokio::test]
async fn a_node_that_cannot_be_reached_is_passed_over_and_then_left_alone() {
    let served = Served::new().await;
    let dead = nowhere().await;
    let client = Client::connect(&[&dead, &served.url], KEY, quick()).unwrap();
    let tenant = client.tenant(&acme());
    tenant.create().await.unwrap();
    tenant
        .collection(&notes_named())
        .create(&schema())
        .await
        .unwrap();

    let started = Instant::now();
    for n in 0..5 {
        tenant
            .change(vec![Change::Increment {
                collection: notes_named(),
                id: oid(1),
                property: "n".into(),
                by: 1.0,
            }])
            .await
            .map(|_| ())
            .or_else(|error| {
                failed_with(&error, Code::NotFound)
                    .then_some(())
                    .ok_or(error)
            })
            .unwrap_or_else(|error| panic!("write {n}: {error}"));
    }
    assert!(
        started.elapsed() < Duration::from_millis(140),
        "the dead node was asked again"
    );

    let nobody = Client::connect(&[&dead, &nowhere().await], KEY, quick()).unwrap();
    let started = Instant::now();
    assert!(matches!(nobody.tenants(None, 5).await, Err(Error::Http(_))));
    assert!(
        started.elapsed() >= Duration::from_millis(450),
        "three tries, two waits"
    );
}

#[tokio::test]
async fn what_may_pass_is_tried_again_unless_that_could_change_things_twice() {
    let served = Cluster::new().unstarted(&[]).await;
    let client = Client::connect(&[&served.url], KEY, quick()).unwrap();
    let tenant = client.tenant(&acme());
    let notes = tenant.collection(&notes_named());
    let timed = |started: Instant| started.elapsed();

    let started = Instant::now();
    let unavailable = notes
        .put(oid(1), &content("fox", 1, None))
        .await
        .unwrap_err();
    assert!(failed_with(&unavailable, Code::Unavailable));
    assert!(
        timed(started) >= Duration::from_millis(450),
        "a whole object is sent again"
    );

    for change in [
        Change::Increment {
            collection: notes_named(),
            id: oid(1),
            property: "n".into(),
            by: 1.0,
        },
        Change::Insert {
            collection: notes_named(),
            id: oid(1),
            properties: Default::default(),
            vector: None,
        },
        Change::Append {
            collection: notes_named(),
            id: oid(1),
            property: "tags".into(),
            values: vec!["a".into()],
            unique: false,
        },
    ] {
        let started = Instant::now();
        assert!(failed_with(
            &tenant.change(vec![change]).await.unwrap_err(),
            Code::Unavailable
        ));
        assert!(timed(started) < Duration::from_millis(140), "sent once");
    }
    let guarded = Changes {
        properties: Default::default(),
        vector: None,
        only_if: Some(OnlyIf {
            property: "n".into(),
            equals: json!(1),
        }),
    };
    let started = Instant::now();
    assert!(notes.patch(oid(1), &guarded).await.is_err());
    assert!(timed(started) < Duration::from_millis(140));

    let (app, node) = (served.app.clone(), served.node.clone());
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        app.start(node);
    });
    tenant.create().await.unwrap();
    assert_eq!(client.tenants(None, 5).await.unwrap().tenants, [acme()]);
}

#[tokio::test]
async fn a_client_asks_to_read_at_least_what_it_wrote() {
    use axum::extract::State;
    use axum::http::HeaderMap;
    use axum::routing::{get, put};
    use axum::Json;
    type Seen = Arc<Mutex<Vec<Option<String>>>>;
    let record = |seen: &Seen, headers: &HeaderMap| {
        let asked = headers
            .get(MIN_POSITION)
            .map(|value| value.to_str().unwrap().to_string());
        seen.lock().unwrap().push(asked);
    };
    let wrote = move |State(seen): State<Seen>, headers: HeaderMap| async move {
        record(&seen, &headers);
        Json(
            json!({ "position": 7, "outcomes": [{ "outcome": "applied" }, { "outcome": "from the future" }] }),
        )
    };
    let listed = move |State(seen): State<Seen>, headers: HeaderMap| async move {
        record(&seen, &headers);
        Json(json!({ "collections": [] }))
    };
    let seen = Seen::default();
    let router = axum::Router::new()
        .route(
            "/v1/tenants/acme/collections/notes/objects/{id}",
            put(wrote),
        )
        .route("/v1/tenants/acme/collections", get(listed))
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

    let client = Client::connect(&[&url], KEY, quick()).unwrap();
    let tenant = client.tenant(&acme());
    tenant.collections().await.unwrap();
    let written = tenant
        .collection(&notes_named())
        .put(oid(1), &content("fox", 1, None))
        .await
        .unwrap();
    assert_eq!(written.outcomes, [Outcome::Applied, Outcome::Unknown]);
    tenant.collections().await.unwrap();
    assert_eq!(*seen.lock().unwrap(), [None, None, Some("7".to_string())]);
    assert_eq!(client.position(&acme()), Some(7));

    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let at = |position: &str| {
        let request = served
            .http
            .get(format!("{}{NOTES}", served.url))
            .bearer_auth(KEY)
            .header(MIN_POSITION, position);
        async move {
            let response = request.send().await.unwrap();
            (response.status(), response.json::<Value>().await.unwrap())
        }
    };
    assert_eq!(at("1").await.0, StatusCode::OK);
    let (status, body) = at("2").await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::SERVICE_UNAVAILABLE, "unavailable")
    );
    let (status, body) = at("soon").await;
    assert_eq!((status, code(&body)), (StatusCode::BAD_REQUEST, "invalid"));
}

#[tokio::test]
async fn many_objects_are_stored_a_few_requests_at_a_time() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let position = |options: Options| {
        let client = Client::connect(&[&served.url], KEY, options).unwrap();
        let notes = client.tenant(&acme()).collection(&notes_named());
        async move {
            let objects = (0..25).map(|n| {
                (
                    oid(n),
                    content(&format!("note {n}"), n, Some([1.0, n as f32])),
                )
            });
            let last = notes.store_all(objects).await.unwrap().unwrap();
            assert_eq!(notes.count(None).await.unwrap(), 25);
            assert_eq!(notes.store_all([]).await.unwrap(), None);
            last
        }
    };
    let start = 1;
    let by_count = position(Options {
        batch_objects: 10,
        ..quick()
    })
    .await;
    assert_eq!(by_count, start + 3);
    let whole = position(quick()).await;
    assert_eq!(whole, by_count, "storing the same again changes nothing");
    let by_size = position(Options {
        batch_bytes: 1,
        ..quick()
    })
    .await;
    assert_eq!(by_size, whole);

    served.ok(reqwest::Method::DELETE, NOTES, None).await;
    served
        .ok(reqwest::Method::PUT, NOTES, Some(notes(supplied())))
        .await;
    let before = start + 3 + 2;
    let one_each = position(Options {
        batch_bytes: 1,
        ..quick()
    })
    .await;
    assert_eq!(one_each, before + 25);
}

#[tokio::test]
async fn a_node_that_does_not_answer_in_time_is_given_up_on() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            held.push(socket);
        }
    });
    let impatient = Options {
        timeout: Duration::from_millis(100),
        attempts: 2,
        backoff: Duration::from_millis(10),
        ..Options::default()
    };
    let client = Client::connect(&[&url], KEY, impatient).unwrap();
    let tenant = client.tenant(&acme());
    let started = Instant::now();
    assert!(matches!(tenant.collections().await, Err(Error::Http(error)) if error.is_timeout()));
    let read = started.elapsed();
    assert!(
        read >= Duration::from_millis(200) && read < Duration::from_secs(2),
        "{read:?}"
    );

    let started = Instant::now();
    let counted = tenant.change(vec![Change::Increment {
        collection: notes_named(),
        id: oid(1),
        property: "n".into(),
        by: 1.0,
    }]);
    assert!(counted.await.is_err());
    assert!(
        started.elapsed() < Duration::from_millis(190),
        "it may have arrived, so it is not sent again"
    );
}
