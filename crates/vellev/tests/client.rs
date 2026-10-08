mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::{Method as Asked, StatusCode as Answered, Uri};
use common::*;
use reqwest::StatusCode;
use serde_json::{json, Value};
use vellev_api::{
    Bounded, Change, Changes, Code, Content, Hybrid, Keyword, Limit, Missing, Nearest, OnlyIf,
    Outcome, Page, AFTER, REQUEST,
};
use vellev_client::api::After;
use vellev_client::{Client, Error, Options};
use vellev_core::{Collection, CollectionName, ObjectId, Property, TenantId};
use vellev_index::filter::Filter;

/// A wait so long that a test which waits it has failed.
const NEVER: Duration = Duration::from_secs(600);

fn tries(attempts: u32) -> Options {
    Options {
        timeout: Duration::from_secs(10),
        attempts,
        backoff: NEVER,
        ..Options::default()
    }
}

/// As [`tries`], for a client of one node, which waits between tries.
fn quick(attempts: u32) -> Options {
    Options {
        backoff: Duration::from_millis(1),
        ..tries(attempts)
    }
}

fn tenant(id: &str) -> TenantId {
    TenantId::new(id).unwrap()
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

fn increment() -> Vec<Change> {
    vec![Change::Increment {
        collection: notes_named(),
        id: oid(1),
        property: "n".into(),
        by: 1.0,
    }]
}

/// A stand-in node that answers each request as `answer` says and
/// remembers what it was asked, with the freshness each request asked for
/// and the name each write was sent under.
#[derive(Clone)]
struct Stub {
    url: String,
    asked: Arc<Mutex<Vec<Request>>>,
}

/// A request's method and path, the freshness it asked for, and its name.
type Request = (String, Option<String>, Option<String>);

impl Stub {
    async fn answering(
        answer: impl Fn(&Asked, &str) -> (u16, Value) + Send + Sync + 'static,
    ) -> Self {
        let answer = Arc::new(answer);
        let asked = Arc::new(Mutex::new(Vec::new()));
        let seen = asked.clone();
        let router = axum::Router::new().fallback(
            move |method: Asked, uri: Uri, headers: axum::http::HeaderMap| {
                let answer = answer.clone();
                let seen = seen.clone();
                async move {
                    let header = |name| {
                        headers.get(name).map(|value: &axum::http::HeaderValue| {
                            value.to_str().unwrap().to_string()
                        })
                    };
                    seen.lock().unwrap().push((
                        format!("{method} {}", uri.path()),
                        header(AFTER),
                        header(REQUEST),
                    ));
                    let (status, body) = answer(&method, uri.path());
                    let status = Answered::from_u16(status).unwrap();
                    match body {
                        Value::String(text) => (status, text).into_response(),
                        body => (status, axum::Json(body)).into_response(),
                    }
                }
            },
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self { url, asked }
    }

    /// How many requests it has had whose path contains `part`.
    fn asked(&self, part: &str) -> usize {
        let asked = self.asked.lock().unwrap();
        asked
            .iter()
            .filter(|(request, ..)| request.contains(part))
            .count()
    }
}

use axum::response::IntoResponse;

fn failure(code: &str) -> Value {
    json!({ "error": { "code": code, "message": format!("stand-in {code}") } })
}

/// Every tenant is named for how the stub answers for it.
fn by_tenant(_: &Asked, path: &str) -> (u16, Value) {
    let named = |tenant: &str| path.contains(&format!("/tenants/{tenant}"));
    match () {
        () if named("away") => (503, failure("unavailable")),
        () if named("torn") => (503, failure("uncertain")),
        () if named("proxied") => (502, json!("<html>Bad Gateway</html>")),
        () if named("missing") => (404, failure("not_found")),
        () if named("clash") => (409, failure("conflict")),
        () if named("wrong") => (400, failure("invalid")),
        () if named("locked") => (401, failure("unauthorized")),
        () if named("broken") => (500, failure("internal")),
        () if named("modelless") => (501, failure("model_unavailable")),
        () if named("newer") => (503, failure("something_new")),
        () if named("garbled") => (200, json!("not what was asked for")),
        () => (
            200,
            json!({ "incarnation": 3, "position": 7, "outcomes": [{ "outcome": "from the future" }] }),
        ),
    }
}

#[tokio::test]
async fn the_whole_api_is_reached_through_the_client() {
    let served = Served::new().await;
    let client = Client::connect(&[&served.url], KEY, tries(1)).unwrap();
    client.live().await.unwrap();
    client.ready().await.unwrap();
    let acme = client.tenant(&tenant("acme"));
    let notes = acme.collection(&notes_named());

    assert!(acme.collections().await.unwrap_err().is_not_found());
    acme.create().await.unwrap();
    acme.create().await.unwrap();
    assert_eq!(
        client.tenants(None, 10).await.unwrap().tenants,
        [tenant("acme")]
    );
    assert_eq!(notes.schema().await.unwrap(), None);

    let created = notes.create(&schema()).await.unwrap();
    assert_eq!(created.outcomes, [Outcome::Applied]);
    assert_eq!(
        notes.create(&schema()).await.unwrap().outcomes,
        [Outcome::Unchanged]
    );
    assert_eq!(notes.schema().await.unwrap(), Some(schema()));
    assert_eq!(acme.collections().await.unwrap(), [schema()]);
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
    let clash = notes.patch(oid(1), &patched).await.unwrap_err();
    assert_eq!(clash.code(), Some(Code::Conflict));
    assert!(!clash.is_unknown_outcome());

    let changed = acme
        .apply(
            [
                increment(),
                vec![Change::Delete {
                    collection: notes_named(),
                    id: oid(9),
                }],
            ]
            .concat(),
        )
        .await
        .unwrap();
    assert_eq!(
        changed.outcomes,
        [Outcome::Value { value: json!(6) }, Outcome::Unchanged]
    );
    assert_eq!(client.after(&tenant("acme")), Some(changed.after()));

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
    assert_eq!(notes.count(Some(&big)).await.unwrap(), 1);
    let none = notes.list(&Page { limit: 0, ..page }).await.unwrap_err();
    assert_eq!(none.code(), Some(Code::Invalid));

    notes.delete(oid(2)).await.unwrap();
    assert_eq!(notes.count(None).await.unwrap(), 1);
    notes.drop().await.unwrap();
    assert!(notes.count(None).await.unwrap_err().is_not_found());
    assert!(acme.drop().await.unwrap());
    assert!(!acme.drop().await.unwrap());
    assert_eq!(client.after(&tenant("acme")), None);

    let stranger = Client::connect(&[&served.url], "not-a-key", quick(3)).unwrap();
    let refused = stranger.tenants(None, 5).await.unwrap_err();
    assert!(
        matches!(
            refused,
            Error::Api {
                status: StatusCode::UNAUTHORIZED,
                code: Code::Unauthorized,
                ..
            }
        ),
        "{refused}"
    );
    for unusable in [
        Client::connect(&[], KEY, tries(1)),
        Client::connect(&[&served.url], KEY, tries(0)),
        Client::connect(&["no scheme"], KEY, tries(1)),
    ] {
        assert!(matches!(unusable, Err(Error::Unusable(_))));
    }
}

#[tokio::test]
async fn a_request_is_sent_again_unless_it_was_refused_for_good() {
    let stub = Stub::answering(by_tenant).await;
    let client = Client::connect(&[&stub.url], KEY, quick(3)).unwrap();
    let write = |name: &'static str| {
        let tenant = client.tenant(&tenant(name));
        async move { tenant.apply(increment()).await }
    };
    let read = |name: &'static str| {
        let tenant = client.tenant(&tenant(name));
        async move { tenant.collections().await.map(|_| ()) }
    };
    let asked = |name: &str, method: &str| {
        let asked = stub.asked.lock().unwrap();
        let of = |(request, ..): &&Request| {
            request.starts_with(method) && request.contains(&format!("/tenants/{name}/"))
        };
        asked.iter().filter(of).count()
    };
    // The names the writes to a tenant were sent under.
    let named = |name: &str| {
        let asked = stub.asked.lock().unwrap();
        let names: std::collections::HashSet<String> = asked
            .iter()
            .filter(|(request, ..)| {
                request.starts_with("POST") && request.contains(&format!("/tenants/{name}/"))
            })
            .map(|(.., named)| named.clone().expect("every write is named"))
            .collect();
        names
    };

    for (name, sends, unknown) in [
        ("away", 3, false),
        ("torn", 3, true),
        ("proxied", 3, true),
        ("missing", 1, false),
        ("clash", 1, false),
        ("wrong", 1, false),
        ("locked", 1, false),
        ("broken", 1, false),
        ("modelless", 1, false),
        ("newer", 1, false),
        ("garbled", 3, true),
    ] {
        let failed = write(name).await.unwrap_err();
        assert_eq!(asked(name, "POST"), sends, "a write to {name}: {failed}");
        assert_eq!(
            named(name).len(),
            1,
            "every try of a write to {name} has one name"
        );
        assert_eq!(
            failed.is_unknown_outcome(),
            unknown,
            "a write to {name}: {failed}"
        );
    }
    for (name, sends) in [
        ("away", 3),
        ("proxied", 3),
        ("garbled", 3),
        ("missing", 1),
        ("wrong", 1),
        ("locked", 1),
        ("broken", 1),
        ("modelless", 1),
        ("newer", 1),
    ] {
        let failed = read(name).await.unwrap_err();
        assert_eq!(asked(name, "GET"), sends, "a read of {name}: {failed}");
    }

    let put = |name: &'static str| {
        let notes = client.tenant(&tenant(name)).collection(&notes_named());
        async move { notes.put(oid(1), &content("fox", 1, None)).await }
    };
    let patch = |name: &'static str| {
        let notes = client.tenant(&tenant(name)).collection(&notes_named());
        let changes = Changes {
            properties: json!({ "n": 2 }).as_object().unwrap().clone(),
            vector: None,
            only_if: None,
        };
        async move { notes.patch(oid(1), &changes).await }
    };
    for (name, sends) in [("torn", 3), ("proxied", 3), ("garbled", 3), ("wrong", 1)] {
        let failed = put(name).await.unwrap_err();
        assert_eq!(
            asked(name, "PUT"),
            sends,
            "a whole object put to {name}, made twice as once: {failed}"
        );
    }
    patch("torn").await.unwrap_err();
    assert_eq!(asked("torn", "PATCH"), 3, "and a patch, which is named");
    let posted = asked("torn", "POST");
    let stored = client
        .tenant(&tenant("torn"))
        .collection(&notes_named())
        .store_all([(oid(1), content("fox", 1, None))])
        .await;
    assert!(matches!(stored, Err(Error::Partial { stored: 0, .. })));
    assert_eq!(
        asked("torn", "POST") - posted,
        3,
        "whole objects stored together are sent again"
    );

    let proxied = write("proxied").await.unwrap_err();
    assert!(
        matches!(&proxied, Error::Status { status: StatusCode::BAD_GATEWAY, body } if body.contains("Bad Gateway")),
        "{proxied}"
    );
    assert!(matches!(write("garbled").await, Err(Error::Decode(_))));
    assert_eq!(write("fine").await.unwrap().outcomes, [Outcome::Unknown]);
}

#[tokio::test]
async fn each_tenant_is_asked_of_the_node_that_last_answered_for_it() {
    let away = Stub::answering(|_, _| (503, failure("unavailable"))).await;
    let served = Served::new().await;
    let client = Client::connect(&[&away.url, &served.url], KEY, tries(2)).unwrap();
    let acme = client.tenant(&tenant("acme"));
    acme.create().await.unwrap();
    acme.collection(&notes_named())
        .create(&schema())
        .await
        .unwrap();
    for n in 0..5 {
        let notes = acme.collection(&notes_named());
        notes.put(oid(n), &content("fox", n, None)).await.unwrap();
        assert_eq!(notes.count(None).await.unwrap(), n + 1);
    }
    assert_eq!(
        away.asked("/tenants/acme"),
        1,
        "the node with nothing to offer was asked once"
    );
    client.tenants(None, 5).await.unwrap();
    client.tenants(None, 5).await.unwrap();
    assert_eq!(away.asked("/v1/tenants"), 2);

    let cluster = Cluster::new();
    let (first, second) = (cluster.serve().await, cluster.serve().await);
    let both = Client::connect(&[&first.url, &second.url], KEY, tries(2)).unwrap();
    for (served, name) in [(&first, "aaa"), (&second, "bbb")] {
        let direct = Client::connect(&[&served.url], KEY, tries(1)).unwrap();
        direct.tenant(&tenant(name)).create().await.unwrap();
        direct
            .tenant(&tenant(name))
            .collection(&notes_named())
            .create(&schema())
            .await
            .unwrap();
    }
    for _ in 0..10 {
        for name in ["aaa", "bbb"] {
            let notes = both.tenant(&tenant(name)).collection(&notes_named());
            assert_eq!(notes.count(None).await.unwrap(), 0, "{name}");
        }
    }
}

#[tokio::test]
async fn a_node_still_starting_is_waited_for() {
    let served = Cluster::new().unstarted(&[], None).await;
    let patient = Options {
        backoff: Duration::from_millis(50),
        ..tries(8)
    };
    let client = Client::connect(&[&served.url], KEY, patient).unwrap();
    let (app, node) = (served.app.clone(), served.node.clone());
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        app.start(node);
    });
    client.tenant(&tenant("acme")).create().await.unwrap();
    assert_eq!(
        client.tenants(None, 5).await.unwrap().tenants,
        [tenant("acme")]
    );
}

/// A node that takes a request and then drops the connection, or never
/// answers it, counting the connections it was sent.
async fn faulty(answers: bool) -> (String, Arc<AtomicUsize>) {
    use tokio::io::AsyncReadExt;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let connections = Arc::new(AtomicUsize::new(0));
    let counted = connections.clone();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((mut socket, _)) = listener.accept().await {
            counted.fetch_add(1, Ordering::SeqCst);
            let mut request = [0; 1024];
            let _ = socket.read(&mut request).await;
            if !answers {
                held.push(socket);
            }
        }
    });
    (url, connections)
}

#[tokio::test]
async fn a_request_cut_off_or_left_unanswered_is_sent_again() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let (dropping, dropped) = faulty(true).await;
    let client = Client::connect(&[&dropping, &served.url], KEY, tries(2)).unwrap();
    let acme = client.tenant(&tenant("acme"));
    acme.collection(&notes_named())
        .put(oid(1), &content("fox", 1, None))
        .await
        .unwrap();
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        1,
        "a write cut off is sent again"
    );
    let reader = Client::connect(&[&dropping, &served.url], KEY, tries(2)).unwrap();
    let collections = reader.tenant(&tenant("acme")).collections().await.unwrap();
    assert_eq!(collections.len(), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 2, "and so is a read");

    let (silent, kept) = faulty(false).await;
    let patient = Client::connect(&[&silent], KEY, quick(2)).unwrap();
    let impatient = patient.within(Duration::from_millis(100));
    let started = Instant::now();
    let unanswered = impatient
        .tenant(&tenant("acme"))
        .collections()
        .await
        .unwrap_err();
    assert!(
        matches!(&unanswered, Error::Http(error) if error.is_timeout()),
        "{unanswered}"
    );
    assert_eq!(kept.load(Ordering::SeqCst), 2, "a read is asked again");
    let unwritten = impatient
        .tenant(&tenant("acme"))
        .apply(increment())
        .await
        .unwrap_err();
    assert!(unwritten.is_unknown_outcome());
    assert_eq!(kept.load(Ordering::SeqCst), 4, "and a write");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the client's own timeout was not waited"
    );

    let nobody = {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    };
    let client = Client::connect(&[&nobody, &served.url], KEY, tries(2)).unwrap();
    let acme = client.tenant(&tenant("acme")).collection(&notes_named());
    acme.put(oid(1), &content("fox", 1, None)).await.unwrap();
    assert_eq!(
        acme.count(None).await.unwrap(),
        1,
        "a write that never arrived is sent on"
    );
}

#[tokio::test]
async fn a_read_asks_to_be_as_fresh_as_the_clients_own_last_write() {
    let written = Arc::new(Mutex::new(vec![(3, 9), (3, 7), (5, 1)].into_iter()));
    let stub = {
        let written = written.clone();
        Stub::answering(move |method, _| match *method {
            Asked::GET => (200, json!({ "collections": [] })),
            _ => {
                let (incarnation, position) = written.lock().unwrap().next().unwrap();
                let answer = json!({
                    "incarnation": incarnation,
                    "position": position,
                    "outcomes": [{ "outcome": "applied" }],
                });
                (200, answer)
            }
        })
        .await
    };
    let client = Client::connect(&[&stub.url], KEY, tries(1)).unwrap();
    let acme = client.tenant(&tenant("acme"));
    let notes = acme.collection(&notes_named());
    acme.collections().await.unwrap();
    notes.put(oid(1), &content("fox", 1, None)).await.unwrap();
    acme.collections().await.unwrap();
    notes.put(oid(2), &content("hen", 2, None)).await.unwrap();
    acme.collections().await.unwrap();
    notes.put(oid(3), &content("owl", 3, None)).await.unwrap();
    acme.collections().await.unwrap();
    client.tenant(&tenant("other")).collections().await.unwrap();
    let fresh: Vec<Option<String>> = stub
        .asked
        .lock()
        .unwrap()
        .iter()
        .filter(|(asked, ..)| asked.starts_with("GET "))
        .map(|(_, fresh, _)| fresh.clone())
        .collect();
    let at = |text: &str| Some(text.to_string());
    assert_eq!(
        fresh,
        [None, at("3.9"), at("3.9"), at("5.1"), None],
        "an older position is kept; a new incarnation replaces it"
    );
    assert_eq!(
        client.after(&tenant("acme")),
        Some(After {
            incarnation: 5,
            position: 1
        })
    );
    assert_eq!(client.after(&tenant("other")), None);
}

#[tokio::test]
async fn many_objects_are_stored_a_few_requests_at_a_time() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let notes_of = |options: Options| {
        let client = Client::connect(&[&served.url], KEY, options).unwrap();
        client.tenant(&tenant("acme")).collection(&notes_named())
    };
    let objects = || {
        (0..25).map(|n| {
            (
                oid(n),
                content(&format!("note {n}"), n, Some([1.0, n as f32])),
            )
        })
    };
    let start = 1;

    let by_count = notes_of(Options {
        batch_objects: 10,
        ..tries(1)
    });
    assert_eq!(
        by_count.store_all(objects()).await.unwrap(),
        Some(start + 3)
    );
    assert_eq!(by_count.count(None).await.unwrap(), 25);
    assert_eq!(by_count.store_all([]).await.unwrap(), None);
    let whole = notes_of(tries(1));
    assert_eq!(
        whole.store_all(objects()).await.unwrap(),
        Some(start + 4),
        "storing the same again changes nothing, but is remembered as asked"
    );
    assert_eq!(whole.count(None).await.unwrap(), 25);

    served.ok(reqwest::Method::DELETE, NOTES, None).await;
    served
        .ok(reqwest::Method::PUT, NOTES, Some(notes(supplied())))
        .await;
    let one_each = notes_of(Options {
        batch_bytes: 1,
        ..tries(1)
    });
    assert_eq!(
        one_each.store_all(objects()).await.unwrap(),
        Some(start + 4 + 2 + 25)
    );

    served.ok(reqwest::Method::DELETE, NOTES, None).await;
    served
        .ok(reqwest::Method::PUT, NOTES, Some(notes(supplied())))
        .await;
    let spoilt = objects().map(|(id, content)| match id == oid(14) {
        true => (
            id,
            Content {
                vector: Some(vec![1.0; 3]),
                ..content
            },
        ),
        false => (id, content),
    });
    let failed = by_count.store_all(spoilt).await.unwrap_err();
    assert!(
        matches!(
            &failed,
            Error::Partial {
                stored: 10,
                position: Some(_),
                ..
            }
        ),
        "{failed}"
    );
    assert_eq!(failed.code(), Some(Code::Invalid));
    assert_eq!(by_count.count(None).await.unwrap(), 10);
}

#[tokio::test]
async fn a_node_stopped_and_replaced_under_load_fails_no_request() {
    let cluster = Cluster::new();
    let (leaving, staying) = (cluster.serve().await, cluster.serve().await);
    let replacement = cluster.unstarted(&[], None).await;
    let nodes = [&leaving.url, &staying.url, &replacement.url];
    let client = Client::connect(&nodes.map(String::as_str), KEY, Options::default()).unwrap();
    let acme = client.tenant(&tenant("acme"));
    acme.create().await.unwrap();
    acme.collection(&notes_named())
        .create(&schema())
        .await
        .unwrap();

    const WORKERS: u64 = 8;
    let (done, after_stop) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let load: Vec<_> = (0..WORKERS)
        .map(|worker| {
            let notes = acme.collection(&notes_named());
            let (done, after_stop, stopped) = (done.clone(), after_stop.clone(), stopped.clone());
            tokio::spawn(async move {
                let mut failures = Vec::new();
                let mut n = worker;
                while after_stop.load(Ordering::SeqCst) < 200 {
                    n += WORKERS;
                    let put = notes
                        .put(oid(n), &content("fox", n, Some([1.0, 0.0])))
                        .await;
                    let read = notes.get(oid(n)).await;
                    match (put, read) {
                        (Ok(_), Ok(Some(_))) => {}
                        other => failures.push((n, format!("{other:?}"))),
                    }
                    done.fetch_add(1, Ordering::SeqCst);
                    if stopped.load(Ordering::SeqCst) {
                        after_stop.fetch_add(1, Ordering::SeqCst);
                    }
                }
                failures
            })
        })
        .collect();

    while done.load(Ordering::SeqCst) < 100 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    leaving.app.stopping();
    stopped.store(true, Ordering::SeqCst);
    let (status, _) = leaving.get("/health/ready").await;
    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "not ready once stopping"
    );
    let live = leaving
        .http
        .get(format!("{}/health/live", leaving.url))
        .send()
        .await
        .unwrap();
    assert_eq!(live.status(), StatusCode::OK);
    assert_eq!(
        live.headers()
            .get("connection")
            .and_then(|value| value.to_str().ok()),
        Some("close"),
        "callers are asked to close the connection"
    );
    let node = leaving.node.clone();
    leaving.stop().await;
    node.shutdown().await;
    replacement.app.start(replacement.node.clone());

    let mut failures = Vec::new();
    for worker in load {
        failures.extend(worker.await.unwrap());
    }
    assert!(failures.is_empty(), "{failures:?}");
    let written = done.load(Ordering::SeqCst);
    let count = acme.collection(&notes_named()).count(None).await.unwrap();
    assert_eq!(count, written as u64);
    assert_eq!(replacement.get("/health/ready").await.0, StatusCode::OK);
}

/// A node in front of `served` that passes every request on and answers
/// as it answered, except that it loses the answer to the first write,
/// after `meanwhile` has run, saying instead that the write's fate is
/// unknown.
async fn losing_first_answer(
    served: &str,
    meanwhile: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
        + Send
        + Sync
        + 'static,
) -> String {
    let served = served.to_string();
    let lost = Arc::new(AtomicUsize::new(0));
    let meanwhile = Arc::new(meanwhile);
    let router = axum::Router::new().fallback(
        move |method: Asked, uri: Uri, headers: axum::http::HeaderMap, body: axum::body::Bytes| {
            let (served, lost, meanwhile) = (served.clone(), lost.clone(), meanwhile.clone());
            async move {
                let passed = headers
                    .iter()
                    .filter(|(name, _)| *name != axum::http::header::HOST)
                    .fold(
                        reqwest::header::HeaderMap::new(),
                        |mut passed, (name, value)| {
                            passed.append(name.clone(), value.clone());
                            passed
                        },
                    );
                let method = reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap();
                let writes = method != reqwest::Method::GET;
                let answer = reqwest::Client::new()
                    .request(method, format!("{served}{uri}"))
                    .headers(passed)
                    .body(body)
                    .send()
                    .await
                    .unwrap();
                let status = Answered::from_u16(answer.status().as_u16()).unwrap();
                let body = answer.bytes().await.unwrap();
                if writes && lost.fetch_add(1, Ordering::SeqCst) == 0 {
                    meanwhile().await;
                    return (
                        Answered::SERVICE_UNAVAILABLE,
                        axum::Json(failure("uncertain")),
                    )
                        .into_response();
                }
                (status, body).into_response()
            }
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

#[tokio::test]
async fn a_write_whose_answer_was_lost_is_answered_as_it_was_and_a_later_write_survives() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let direct = Client::connect(&[&served.url], KEY, tries(1)).unwrap();
    let notes = direct.tenant(&tenant("acme")).collection(&notes_named());
    notes.put(oid(1), &content("fox", 1, None)).await.unwrap();

    let other = notes.clone();
    let lossy = losing_first_answer(&served.url, move || {
        let other = other.clone();
        Box::pin(async move {
            other.put(oid(1), &content("fox", 100, None)).await.unwrap();
        })
    })
    .await;
    let client = Client::connect(&[&lossy], KEY, quick(3)).unwrap();
    let written = client
        .tenant(&tenant("acme"))
        .apply(increment())
        .await
        .unwrap();
    assert_eq!(written.outcomes, [Outcome::Value { value: json!(2) }]);
    let stored = notes.get(oid(1)).await.unwrap().unwrap();
    assert_eq!(
        stored.properties["n"],
        json!(100),
        "the later write survives"
    );
}

#[tokio::test]
async fn a_request_turned_away_for_asking_too_much_is_asked_again_once_the_wait_is_over() {
    let served = Cluster::new().serve_limited(2.0).await;
    served.with_notes(supplied()).await;
    let client = Client::connect(&[&served.url], KEY, quick(4)).unwrap();
    let acme = client.tenant(&tenant("acme"));
    let started = Instant::now();
    for _ in 0..4 {
        acme.collections().await.unwrap();
    }
    assert!(
        started.elapsed() >= Duration::from_millis(900),
        "waited as told: {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_failure_says_what_was_asked_too_much_of_or_is_missing_and_any_key_learns_the_limits() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let options = Options {
        batch_bytes: usize::MAX,
        ..quick(3)
    };
    let client = Client::connect(&[&served.url], KEY, options).unwrap();
    let limits = client
        .limits()
        .await
        .unwrap()
        .expect("this node tells them");
    assert_eq!(
        serde_json::to_value(limits).unwrap(),
        json!({
            "results": MAX_LIMIT,
            "candidates": 100,
            "oversample": 1_024,
            "body_bytes": MAX_BODY,
            "restore_window_secs": 10 * 366 * 24 * 60 * 60,
        })
    );
    let issued = client.tenant(&tenant("acme")).issue_key().await.unwrap();
    let tenants_own = Client::connect(&[&served.url], &issued.key, quick(1)).unwrap();
    assert_eq!(tenants_own.limits().await.unwrap(), Some(limits));

    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let page = |limit: u64| Page {
        filter: None,
        after: None,
        limit: limit as u32,
        vectors: false,
    };
    let too_many = notes.list(&page(limits.results + 1)).await.unwrap_err();
    assert_eq!(
        too_many.limit(),
        Some(Limit {
            name: Bounded::Results,
            max: limits.results
        })
    );
    assert_eq!(too_many.limit_max(Bounded::Results), Some(limits.results));
    assert_eq!(too_many.limit_max(Bounded::Oversample), None);
    assert_eq!(too_many.missing(), None);
    let none = notes.list(&page(0)).await.unwrap_err();
    assert_eq!(
        (none.code(), none.limit()),
        (Some(Code::Invalid), None),
        "asking for none is not asking too much"
    );
    let large = content(&"x".repeat(MAX_BODY), 1, Some([1.0, 0.0]));
    let unsent = notes.store_all([(oid(1), large)]).await.unwrap_err();
    assert!(
        matches!(unsent, Error::Partial { stored: 0, .. }),
        "{unsent}"
    );
    assert_eq!(
        unsent.limit_max(Bounded::BodyBytes),
        Some(limits.body_bytes)
    );

    let ghost = client.tenant(&tenant("ghost")).collections().await;
    assert_eq!(ghost.unwrap_err().missing(), Some(Missing::Tenant));
    let absent = client
        .tenant(&tenant("acme"))
        .collection(&CollectionName::new("absent").unwrap());
    let unlisted = absent.list(&page(1)).await.unwrap_err();
    assert_eq!(
        (unlisted.missing(), unlisted.limit()),
        (Some(Missing::Collection), None)
    );
    let unstored = absent
        .store_all([(oid(1), content("fox", 1, Some([1.0, 0.0])))])
        .await
        .unwrap_err();
    assert!(
        matches!(unstored, Error::Partial { stored: 0, .. }),
        "{unstored}"
    );
    assert_eq!(unstored.missing(), Some(Missing::Collection));
    assert_eq!(
        notes.get(oid(9)).await.unwrap(),
        None,
        "a missing object is still none"
    );
}

#[tokio::test]
async fn a_node_from_before_the_limits_were_told_has_none_to_tell() {
    // As a release before the route answers an admin key, and a tenant's.
    let no_route = json!({ "error": { "code": "not_found", "message": "no such route" } });
    let not_for_it =
        json!({ "error": { "code": "forbidden", "message": "only an admin key may do this" } });
    for (status, answer) in [(404, no_route), (403, not_for_it)] {
        let older = Stub::answering(move |_, _| (status, answer.clone())).await;
        let client = Client::connect(&[&older.url], KEY, tries(3)).unwrap();
        assert_eq!(client.limits().await.unwrap(), None, "{status}");
        assert_eq!(older.asked("/v1/limits"), 1);
    }
    let away = Stub::answering(|_, _| (503, failure("unavailable"))).await;
    let client = Client::connect(&[&away.url], KEY, tries(1)).unwrap();
    let failed = client.limits().await.unwrap_err();
    assert_eq!(
        failed.code(),
        Some(Code::Unavailable),
        "not every failure is none"
    );
}

/// A stand-in node that cannot use the model, or answers as `otherwise`
/// says once it is told it can.
async fn modelless(otherwise: (u16, Value)) -> (Stub, Arc<std::sync::atomic::AtomicBool>) {
    let able = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let asked = able.clone();
    let stub = Stub::answering(move |_, _| match asked.load(Ordering::SeqCst) {
        true => otherwise.clone(),
        false => (501, failure("model_unavailable")),
    })
    .await;
    (stub, able)
}

#[tokio::test]
async fn a_request_one_node_cannot_do_as_it_is_set_up_is_asked_of_each_other_node_once() {
    let found = || (200, json!({ "collections": [] }));
    let (unable, _) = modelless(found()).await;
    let (other, _) = modelless(found()).await;
    let able = Stub::answering(move |_, _| found()).await;
    let ask = |client: &Client| {
        let acme = client.tenant(&tenant("acme"));
        async move { acme.collections().await.map(|_| ()) }
    };

    // One try, and a wait that would never end: the other nodes are asked
    // at once, for nothing but the first's being unable.
    let nodes = [unable.url.as_str(), other.url.as_str(), able.url.as_str()];
    let client = Client::connect(&nodes, KEY, tries(1)).unwrap();
    ask(&client).await.unwrap();
    ask(&client).await.unwrap();
    assert_eq!(
        (
            unable.asked("acme"),
            other.asked("acme"),
            able.asked("acme")
        ),
        (1, 1, 2),
        "and the node that could is asked first from then on"
    );

    let client = Client::connect(&[&unable.url, &other.url], KEY, tries(4)).unwrap();
    let before = (unable.asked("acme"), other.asked("acme"));
    let failed = ask(&client).await.unwrap_err();
    assert!(failed.is_model_unavailable(), "{failed}");
    assert!(!failed.is_unknown_outcome());
    assert_eq!(
        (unable.asked("acme"), other.asked("acme")),
        (before.0 + 1, before.1 + 1),
        "final once every node has said so, each asked once"
    );

    let alone = Client::connect(&[&unable.url], KEY, tries(4)).unwrap();
    assert!(ask(&alone).await.unwrap_err().is_model_unavailable());
    assert_eq!(unable.asked("acme"), before.0 + 2, "one node is asked once");
}

#[tokio::test]
async fn a_node_that_cannot_is_left_out_while_the_others_are_tried_again() {
    let (unable, _) = modelless((200, json!(null))).await;
    let away = Stub::answering(|_, _| (503, failure("unavailable"))).await;
    let client = Client::connect(&[&unable.url, &away.url], KEY, quick(3)).unwrap();
    let failed = client.tenant(&tenant("acme")).collections().await;
    assert_eq!(failed.unwrap_err().code(), Some(Code::Unavailable));
    assert_eq!(
        (unable.asked("acme"), away.asked("acme")),
        (1, 3),
        "every try is spent on the node that may yet answer"
    );

    // The other way about: away first, then unable, then away again.
    let (unable, _) = modelless((200, json!(null))).await;
    let away = Stub::answering(|_, _| (503, failure("unavailable"))).await;
    let client = Client::connect(&[&away.url, &unable.url], KEY, quick(3)).unwrap();
    let failed = client.tenant(&tenant("acme")).collections().await;
    assert_eq!(failed.unwrap_err().code(), Some(Code::Unavailable));
    assert_eq!((unable.asked("acme"), away.asked("acme")), (1, 3));
}

#[tokio::test]
async fn a_search_a_node_cannot_embed_for_is_answered_by_one_that_can() {
    let url = keyed_embedding_server("sk-right").await;
    let cluster = Cluster::new();
    let right = cluster.serve_embedding_keyed(&url, "sk-right").await;
    let wrong = cluster.serve_embedding_keyed(&url, "sk-wrong").await;
    let embedding = json!({
        "provider": "http",
        "api": "openai",
        "url": url,
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    right.with_notes(embedding).await;
    // Asked of the node set up wrong first, with one try and no waiting.
    let client = Client::connect(&[&wrong.url, &right.url], KEY, tries(1)).unwrap();
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let fox = Content {
        properties: json!({ "title": "a red fox" }).as_object().unwrap().clone(),
        vector: None,
    };
    notes.put(oid(1), &fox).await.unwrap();
    let nearest = Nearest {
        vector: None,
        text: Some("a red fox".into()),
        limit: 1,
        filter: None,
        oversample: None,
        vectors: true,
    };
    let found = notes.nearest(&nearest).await.unwrap();
    assert_eq!(found.results[0].object.vector, Some(vec![9.0, 3.0]));

    // With no node set up right, the search fails as final, and says so.
    let only_wrong = Client::connect(&[&wrong.url], KEY, tries(3)).unwrap();
    let notes = only_wrong
        .tenant(&tenant("acme"))
        .collection(&notes_named());
    let failed = notes.nearest(&nearest).await.unwrap_err();
    assert!(failed.is_model_unavailable(), "{failed}");
}
