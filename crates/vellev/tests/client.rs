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
    Outcome, Page, Reason, RestoreTo, AFTER, REQUEST,
};
use vellev_client::api::After;
use vellev_client::{Client, Error, Options};
use vellev_core::{Collection, CollectionName, ObjectId, Property, TenantId};
use vellev_index::filter::Filter;

/// A wait so long that a test which waits it has failed.
const NEVER: Duration = Duration::from_secs(600);

fn tries(attempts: u32) -> Options {
    Options::default()
        .timeout(Duration::from_secs(10))
        .attempts(attempts)
        .backoff(NEVER)
}

/// As [`tries`], for a client of one node, which waits between tries.
fn quick(attempts: u32) -> Options {
    tries(attempts).backoff(Duration::from_millis(1))
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
    let properties = json!({ "title": title, "n": n });
    let mut content = Content::new(properties.as_object().unwrap().clone());
    content.vector = vector.map(Vec::from);
    content
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
    let patched = Changes::new(json!({ "n": 5 }).as_object().unwrap().clone()).only_if(OnlyIf {
        property: "n".into(),
        equals: json!(1),
    });
    notes.patch(oid(1), &patched).await.unwrap();
    let clash = notes.patch(oid(1), &patched).await.unwrap_err();
    assert_eq!(clash.code(), Some(Code::Conflict));
    assert!(!clash.is_unknown_outcome());

    // Every object read says its version, a write can be made only at
    // it, and a refusal says the one the object is at.
    let at = notes.get(oid(1)).await.unwrap().unwrap().version.unwrap();
    assert_eq!(clash.version(), Some(at));
    let listed = notes.list(&Page::new(10)).await.unwrap();
    assert_eq!(listed.objects[0].version, Some(at));
    let found = notes
        .nearest(&Nearest::to_vector(vec![1.0, 0.0], 1))
        .await
        .unwrap();
    assert_eq!(found.results[0].object.version, Some(at));
    let stale = "00000000000000000000000000000000".parse().unwrap();
    let refused = notes
        .put(
            oid(1),
            &content("a red fox", 6, Some([1.0, 0.0])).only_if_version(stale),
        )
        .await
        .unwrap_err();
    assert_eq!(
        (refused.code(), refused.version()),
        (Some(Code::Conflict), Some(at))
    );
    let refused = notes.delete_if_version(oid(1), stale).await.unwrap_err();
    assert_eq!(
        (refused.code(), refused.version()),
        (Some(Code::Conflict), Some(at))
    );
    let moved = Changes::new(json!({ "n": 6 }).as_object().unwrap().clone()).only_if_version(at);
    notes.patch(oid(1), &moved).await.unwrap();
    let now = notes.get(oid(1)).await.unwrap().unwrap().version.unwrap();
    assert_ne!(now, at);
    // A store refused part of the way at a stale version says the
    // version too, through the partial failure.
    let partial = notes
        .store_all([
            (oid(3), content("owl", 3, None)),
            (oid(1), content("a red fox", 7, None).only_if_version(stale)),
        ])
        .await
        .unwrap_err();
    assert!(
        matches!(partial, Error::Partial { stored: 0, .. }),
        "{partial:?}"
    );
    assert_eq!(
        (partial.code(), partial.version()),
        (Some(Code::Conflict), Some(now))
    );
    assert_eq!(notes.get(oid(3)).await.unwrap(), None);
    let deleted = notes.delete_if_version(oid(1), now).await.unwrap();
    assert_eq!(deleted.outcomes, vec![Outcome::Applied]);
    let gone = notes.delete_if_version(oid(1), now).await.unwrap_err();
    assert_eq!(
        (gone.code(), gone.missing()),
        (Some(Code::NotFound), Some(Missing::Object))
    );
    notes
        .put(oid(1), &content("a red fox", 5, Some([1.0, 0.0])))
        .await
        .unwrap();

    let changed = acme
        .apply(
            [
                increment(),
                vec![Change::Delete {
                    collection: notes_named(),
                    id: oid(9),
                    only_if_version: None,
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

    let nearest = Nearest::to_vector(vec![1.0, 0.0], 1).vectors(true);
    let found = notes.nearest(&nearest).await.unwrap();
    assert_eq!(
        (found.results[0].object.id, found.results[0].distance),
        (oid(1), Some(0.0))
    );
    assert_eq!(found.results[0].object.vector, Some(vec![1.0, 0.0]));
    let keyword = Keyword::new("hen", 5);
    let found = notes.keyword(&keyword).await.unwrap();
    assert_eq!(found.results.len(), 1);
    assert!(found.results[0].keyword.unwrap() > 0.0);
    let hybrid = Hybrid::new("fox", 5).vector(vec![1.0, 0.0]);
    let fused = notes.hybrid(&hybrid).await.unwrap();
    assert_eq!(fused.results[0].object.id, oid(1));
    assert!(!fused.degraded);

    let first = notes.list(&Page::new(1)).await.unwrap();
    assert_eq!((first.objects.len(), first.next), (1, Some(oid(1))));
    let big = Filter::Gte {
        property: "n".into(),
        value: json!(5),
    };
    assert_eq!(notes.count(None).await.unwrap(), 2);
    assert_eq!(notes.count(Some(&big)).await.unwrap(), 1);
    let none = notes.list(&Page::new(0)).await.unwrap_err();
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
        ("modelless", 3),
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
        let changes = Changes::new(json!({ "n": 2 }).as_object().unwrap().clone());
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
    let patient = tries(8).backoff(Duration::from_millis(50));
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

    let by_count = notes_of(tries(1).batch_objects(10));
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
    let one_each = notes_of(tries(1).batch_bytes(1));
    assert_eq!(
        one_each.store_all(objects()).await.unwrap(),
        Some(start + 4 + 2 + 25)
    );

    served.ok(reqwest::Method::DELETE, NOTES, None).await;
    served
        .ok(reqwest::Method::PUT, NOTES, Some(notes(supplied())))
        .await;
    let spoilt = objects().map(|(id, content)| match id == oid(14) {
        true => (id, content.vector(vec![1.0; 3])),
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
async fn a_restore_whose_answer_was_lost_is_answered_as_it_was_and_a_later_write_survives() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let direct = Client::connect(&[&served.url], KEY, tries(1)).unwrap();
    let notes = direct.tenant(&tenant("acme")).collection(&notes_named());
    let kept = notes.put(oid(1), &content("fox", 1, None)).await.unwrap();
    notes.put(oid(2), &content("hen", 2, None)).await.unwrap();

    let other = notes.clone();
    let lossy = losing_first_answer(&served.url, move || {
        let other = other.clone();
        Box::pin(async move {
            other.put(oid(3), &content("owl", 3, None)).await.unwrap();
        })
    })
    .await;
    let client = Client::connect(&[&lossy], KEY, quick(3)).unwrap();
    let restored = client
        .tenant(&tenant("acme"))
        .restore(RestoreTo::Position(kept.position))
        .await
        .unwrap();
    assert_eq!(restored.from.position, kept.position);
    assert!(notes.get(oid(2)).await.unwrap().is_none(), "it went back");
    assert!(
        notes.get(oid(3)).await.unwrap().is_some(),
        "the later write survives"
    );
    assert_eq!(restored.incarnation, kept.incarnation);
    assert_eq!(client.after(&tenant("acme")), Some(restored.after()));
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
async fn a_read_a_busy_node_turns_away_is_asked_again_once_the_wait_it_gave_is_over() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    // Every turn the node has for a read is taken, for a while that is
    // longer than the client's own backoff and shorter than the second
    // the node says to wait.
    let other = tenant("beta");
    let mut taken = Vec::new();
    for _ in 0..READS {
        taken.push(served.app.reading(&other).await.unwrap());
    }
    let freeing = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(400)).await;
        drop(taken);
    });
    let client = Client::connect(&[&served.url], KEY, quick(2)).unwrap();
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let started = Instant::now();
    notes
        .list(&Page::new(1))
        .await
        .expect("answered on the second try");
    assert!(
        started.elapsed() >= Duration::from_millis(900),
        "waited as told: {:?}",
        started.elapsed()
    );
    freeing.await.unwrap();

    // With no try left, the failure says how long the node said to wait.
    let mut taken = Vec::new();
    for _ in 0..READS {
        taken.push(served.app.reading(&other).await.unwrap());
    }
    let once = Client::connect(&[&served.url], KEY, quick(1)).unwrap();
    let notes = once.tenant(&tenant("acme")).collection(&notes_named());
    let refused = notes.list(&Page::new(1)).await.unwrap_err();
    assert!(
        matches!(
            refused,
            Error::Api {
                code: Code::Unavailable,
                wait: Some(wait),
                ..
            } if wait == Duration::from_secs(1)
        ),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_failure_says_what_was_asked_too_much_of_or_is_missing_and_any_key_learns_the_limits() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    let options = quick(3).batch_bytes(usize::MAX);
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
            "deleted": 8_192,
            "reads_at_once": READS,
        })
    );
    assert_eq!(limits.max(Bounded::Deleted), Some(8_192));
    assert_eq!(limits.max(Bounded::ReadsAtOnce), Some(READS as u64));
    let issued = client.tenant(&tenant("acme")).issue_key().await.unwrap();
    let tenants_own = Client::connect(&[&served.url], &issued.key, quick(1)).unwrap();
    assert_eq!(tenants_own.limits().await.unwrap(), Some(limits));

    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let page = |limit: u64| Page::new(limit as u32);
    let too_many = notes.list(&page(limits.results + 1)).await.unwrap_err();
    assert_eq!(
        too_many.limit(),
        Some(Limit::new(Bounded::Results, limits.results))
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
    let answered = |code: &str, message: &str, more: Value| {
        let mut problem = json!({ "code": code, "message": message });
        let more = more.as_object().unwrap().clone();
        problem.as_object_mut().unwrap().extend(more);
        json!({ "error": problem })
    };
    let limits = |status: u16, answer: Value| async move {
        let node = Stub::answering(move |_, _| (status, answer.clone())).await;
        let client = Client::connect(&[&node.url], KEY, tries(1)).unwrap();
        let told = client.limits().await;
        assert_eq!(node.asked("/v1/limits"), 1);
        told
    };
    let admins_only = "only an admin key may do this";
    // As a release before the route answers an admin key, and a tenant's;
    // and as one since would, were the route taken out.
    for (status, answer) in [
        (404, answered("not_found", "no such route", json!({}))),
        (403, answered("forbidden", admins_only, json!({}))),
        (
            404,
            answered("not_found", "no such route", json!({ "missing": "route" })),
        ),
    ] {
        let told = limits(status, answer.clone()).await;
        assert_eq!(told.unwrap(), None, "{answer}");
    }
    // Nothing else is a node with none to tell: least of all anything
    // else that is forbidden.
    for (status, answer, code) in [
        (
            401,
            answered("unauthorized", "a valid API key is required", json!({})),
            Some(Code::Unauthorized),
        ),
        (
            403,
            answered("forbidden", "this key may not read limits", json!({})),
            Some(Code::Forbidden),
        ),
        (
            403,
            answered("forbidden", admins_only, json!({ "missing": "tenant" })),
            Some(Code::Forbidden),
        ),
        (403, json!("<html>Forbidden</html>"), None),
        (
            404,
            answered(
                "not_found",
                "tenant acme does not exist",
                json!({ "missing": "tenant" }),
            ),
            Some(Code::NotFound),
        ),
        (
            503,
            answered("unavailable", "not now", json!({})),
            Some(Code::Unavailable),
        ),
        (
            500,
            answered("internal", "the node failed", json!({})),
            Some(Code::Internal),
        ),
    ] {
        let failed = limits(status, answer.clone()).await.unwrap_err();
        assert_eq!(failed.code(), code, "{answer}");
    }
}

#[tokio::test]
async fn a_write_no_node_can_embed_fails_at_the_first_to_say_so_and_says_where_the_fault_lies() {
    // Every address says what the node holding the tenant said: there is
    // nothing to be had from asking another, or the same again.
    let cannot = || {
        let refusal = json!({ "error": {
            "code": "model_unavailable", "message": "stand-in", "reason": "setup",
        } });
        Stub::answering(move |_, _| (501, refusal.clone()))
    };
    let (first, second) = (cannot().await, cannot().await);
    let client = Client::connect(&[&first.url, &second.url], KEY, tries(4)).unwrap();
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let fox = content("fox", 1, None);
    let failed = tokio::time::timeout(Duration::from_secs(30), notes.put(oid(1), &fox))
        .await
        .expect("failed at once, with no wait")
        .unwrap_err();
    assert!(failed.is_model_unavailable(), "{failed}");
    assert!(!failed.is_unknown_outcome());
    assert_eq!(failed.reason(), Some(Reason::Setup));
    assert_eq!((first.asked("acme"), second.asked("acme")), (1, 0));

    let stored = notes
        .store_all([(oid(1), content("fox", 1, None))])
        .await
        .unwrap_err();
    assert!(
        matches!(stored, Error::Partial { stored: 0, .. }),
        "{stored}"
    );
    assert!(stored.is_model_unavailable());
    assert_eq!(
        stored.reason(),
        Some(Reason::Setup),
        "told of the objects not stored as of any request"
    );
    assert_eq!((first.asked("acme"), second.asked("acme")), (2, 0));

    // A read of the same nodes is another matter: each embeds for itself.
    let client = Client::connect(&[&first.url, &second.url], KEY, tries(1)).unwrap();
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let failed = notes.count(None).await.unwrap_err();
    assert!(failed.is_model_unavailable());
    assert_eq!((first.asked("count"), second.asked("count")), (1, 1));
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
    tokio::time::timeout(Duration::from_secs(30), ask(&client))
        .await
        .expect("no wait before another node is asked")
        .unwrap();
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

    // Once every address has said so, the tries are spent as on any
    // failure that may pass: behind an address there may be another node.
    let client = Client::connect(&[&unable.url, &other.url], KEY, quick(3)).unwrap();
    let before = (unable.asked("acme"), other.asked("acme"));
    let failed = ask(&client).await.unwrap_err();
    assert!(failed.is_model_unavailable(), "{failed}");
    assert!(!failed.is_unknown_outcome());
    assert_eq!(
        (unable.asked("acme"), other.asked("acme")),
        (before.0 + 2, before.1 + 2),
        "one send more than there are tries, for two addresses"
    );

    let alone = Client::connect(&[&unable.url], KEY, quick(4)).unwrap();
    assert!(ask(&alone).await.unwrap_err().is_model_unavailable());
    assert_eq!(
        unable.asked("acme"),
        before.0 + 2 + 4,
        "one address is asked as often as there are tries"
    );
}

#[tokio::test]
async fn one_address_with_a_node_behind_it_that_can_answers_on_a_later_try() {
    // A load balancer in front of two nodes, one of which cannot use the
    // model: every other request reaches it.
    let sent = Arc::new(AtomicUsize::new(0));
    let counted = sent.clone();
    let balancer = Stub::answering(
        move |_, _| match counted.fetch_add(1, Ordering::SeqCst) % 2 {
            0 => (501, failure("model_unavailable")),
            _ => (200, json!({ "collections": [] })),
        },
    )
    .await;
    let options = tries(4).backoff(Duration::from_millis(300));
    let client = Client::connect(&[&balancer.url], KEY, options).unwrap();
    let acme = client.tenant(&tenant("acme"));
    let started = Instant::now();
    acme.collections().await.unwrap();
    assert_eq!(balancer.asked("acme"), 2);
    assert!(
        started.elapsed() >= Duration::from_millis(300),
        "after the wait any other try has: {:?}",
        started.elapsed()
    );

    // With only the node that cannot behind it, the request fails once
    // its tries are spent, and says how.
    let (unable, _) = modelless((200, json!(null))).await;
    let client = Client::connect(&[&unable.url], KEY, quick(4)).unwrap();
    let failed = client
        .tenant(&tenant("acme"))
        .collections()
        .await
        .unwrap_err();
    assert!(failed.is_model_unavailable(), "{failed}");
    assert_eq!(unable.asked("acme"), 4);
}

/// A hybrid search's answer by the words alone, for the reason given.
fn by_words(degraded_by: &str) -> (u16, Value) {
    (
        200,
        json!({ "results": [], "degraded": true, "degraded_by": degraded_by }),
    )
}

/// A hybrid search for foxes through `client`.
async fn foxes(client: &Client) -> vellev_client::Result<vellev_api::Results> {
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let asked = Hybrid::new("fox", 5);
    // Nothing here is to wait: a search that does has gone wrong.
    tokio::time::timeout(Duration::from_secs(30), notes.hybrid(&asked))
        .await
        .expect("a search answered without the waits a failure has")
}

#[tokio::test]
async fn a_search_answered_by_the_words_alone_is_given_at_once_unless_an_address_not_yet_asked_does_better(
) {
    let cannot = || Stub::answering(|_, _| by_words("model_unavailable"));
    let full = Stub::answering(|_, _| (200, json!({ "results": [] }))).await;
    let later = Stub::answering(|_, _| by_words("unavailable")).await;
    let by_the_words = |found: &vellev_api::Results| {
        assert_eq!(
            (found.degraded, found.degraded_by),
            (true, Some(Code::ModelUnavailable))
        );
    };

    // One address: its answer is the answer, there and then, whatever
    // tries and waits a failure would have had.
    let alone = cannot().await;
    let client = Client::connect(&[&alone.url], KEY, tries(4)).unwrap();
    by_the_words(&foxes(&client).await.unwrap());
    assert_eq!(alone.asked("hybrid"), 1);

    // Several: each other address is asked once, at once, and the first
    // whole answer is the answer, which the client then asks first.
    let (first, second) = (cannot().await, cannot().await);
    let nodes = [first.url.as_str(), second.url.as_str(), full.url.as_str()];
    let client = Client::connect(&nodes, KEY, tries(1)).unwrap();
    let found = foxes(&client).await.unwrap();
    assert_eq!((found.degraded, found.degraded_by), (false, None));
    foxes(&client).await.unwrap();
    assert_eq!(
        (
            first.asked("hybrid"),
            second.asked("hybrid"),
            full.asked("hybrid")
        ),
        (1, 1, 2)
    );

    // Where none does better, the answer by the words is the answer,
    // not a failure, whatever the others said; and none is asked twice.
    for (status, said) in [
        (503, failure("unavailable")),
        (404, failure("not_found")),
        (501, failure("model_unavailable")),
        (429, failure("limited")),
        (200, by_words("model_unavailable").1),
    ] {
        let first = cannot().await;
        let answer = said.clone();
        let other = Stub::answering(move |_, _| (status, answer.clone())).await;
        let third = cannot().await;
        let nodes = [first.url.as_str(), other.url.as_str(), third.url.as_str()];
        let client = Client::connect(&nodes, KEY, tries(4)).unwrap();
        by_the_words(&foxes(&client).await.unwrap());
        assert_eq!(
            (
                first.asked("hybrid"),
                other.asked("hybrid"),
                third.asked("hybrid")
            ),
            (1, 1, 1),
            "{said}"
        );
    }

    // An address that failed before the answer came is not asked again.
    let away = Stub::answering(|_, _| (503, failure("unavailable"))).await;
    let first = cannot().await;
    let before = full.asked("hybrid");
    let nodes = [away.url.as_str(), first.url.as_str(), full.url.as_str()];
    let client = Client::connect(&nodes, KEY, tries(4)).unwrap();
    let found = foxes(&client).await.unwrap();
    assert_eq!((found.degraded, found.degraded_by), (false, None));
    assert_eq!(
        (
            away.asked("hybrid"),
            first.asked("hybrid"),
            full.asked("hybrid") - before
        ),
        (1, 1, 1)
    );

    // A text that may be embedded later is not worth another address now.
    let before = full.asked("hybrid");
    let client = Client::connect(&[&later.url, &full.url], KEY, tries(3)).unwrap();
    let found = foxes(&client).await.unwrap();
    assert_eq!(found.degraded_by, Some(Code::Unavailable));
    assert_eq!(
        (later.asked("hybrid"), full.asked("hybrid") - before),
        (1, 0)
    );
}

#[tokio::test]
async fn an_address_that_never_answers_does_not_keep_back_the_answer_in_hand() {
    let cannot = Stub::answering(|_, _| by_words("model_unavailable")).await;
    let silent = axum::Router::new().fallback(std::future::pending::<()>);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hung = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, silent).await.unwrap() });

    // Each try may take a minute; the answer in hand waits two seconds,
    // unless the client says otherwise.
    let options = tries(4).timeout(Duration::from_secs(60));
    let client = Client::connect(&[&cannot.url, &hung], KEY, options).unwrap();
    let started = Instant::now();
    let found = foxes(&client).await.unwrap();
    assert_eq!(found.degraded_by, Some(Code::ModelUnavailable));
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(1_900) && took < Duration::from_secs(10),
        "the other address was given two seconds, not its minute: {took:?}"
    );
    assert_eq!(cannot.asked("hybrid"), 1);

    let brief = tries(4)
        .timeout(Duration::from_secs(60))
        .better_within(Duration::from_millis(200));
    let client = Client::connect(&[&cannot.url, &hung], KEY, brief).unwrap();
    let started = Instant::now();
    foxes(&client).await.unwrap();
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(190) && took < Duration::from_millis(1_500),
        "as long as the client gives, and no longer: {took:?}"
    );

    // Given no time at all, no other address has time to better it.
    let at_once = tries(4)
        .timeout(Duration::from_secs(60))
        .better_within(Duration::ZERO);
    let client = Client::connect(&[&cannot.url, &hung], KEY, at_once).unwrap();
    let started = Instant::now();
    let found = foxes(&client).await.unwrap();
    assert_eq!(found.degraded_by, Some(Code::ModelUnavailable));
    assert!(started.elapsed() < Duration::from_millis(150));
}

#[tokio::test]
async fn an_answer_by_the_words_alone_does_not_teach_the_client_which_node_to_ask_first() {
    // One node lists collections and cannot embed; the other is the
    // other way about.
    let lists = Stub::answering(|_, path| match path.ends_with("/hybrid") {
        true => (503, failure("unavailable")),
        false => (200, json!({ "collections": [] })),
    })
    .await;
    let ranks = Stub::answering(|_, path| match path.ends_with("/hybrid") {
        true => by_words("model_unavailable"),
        false => (503, failure("unavailable")),
    })
    .await;
    let client = Client::connect(&[&ranks.url, &lists.url], KEY, quick(3)).unwrap();
    let acme = client.tenant(&tenant("acme"));
    acme.collections().await.unwrap();
    assert_eq!((ranks.asked("GET "), lists.asked("GET ")), (1, 1));
    let found = foxes(&client).await.unwrap();
    assert_eq!(found.degraded_by, Some(Code::ModelUnavailable));
    acme.collections().await.unwrap();
    assert_eq!(
        (ranks.asked("GET "), lists.asked("GET ")),
        (1, 2),
        "the node that listed is still the one asked first"
    );
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
    let url = keyed_embedding_server("sk-right-0123456789abcdef").await;
    let cluster = Cluster::new();
    let right = cluster
        .serve_embedding_keyed(&url, "sk-right-0123456789abcdef")
        .await;
    let wrong = cluster
        .serve_embedding_keyed(&url, "sk-wrong-0123456789abcdef")
        .await;
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
    let fox = Content::new(json!({ "title": "a red fox" }).as_object().unwrap().clone());
    notes.put(oid(1), &fox).await.unwrap();
    let nearest = Nearest::to_text("a red fox", 1).vectors(true);
    let found = notes.nearest(&nearest).await.unwrap();
    assert_eq!(found.results[0].object.vector, Some(vec![9.0, 3.0]));

    let hybrid = Hybrid::new("a red fox", 1);
    // A client that has yet to learn which node can: it asks the node set
    // up wrong first, and goes on past its answer by the words alone.
    let fresh = Client::connect(&[&wrong.url, &right.url], KEY, tries(1)).unwrap();
    let fresh = fresh.tenant(&tenant("acme")).collection(&notes_named());
    let fused = fresh.hybrid(&hybrid).await.unwrap();
    assert_eq!((fused.degraded, fused.degraded_by), (false, None));
    assert!(fused.results[0].distance.is_some(), "ranked by meaning too");

    // With no node set up right, the search fails once its tries are
    // spent, and says how and where the fault lies; a hybrid search is
    // answered by the words alone.
    let only_wrong = Client::connect(&[&wrong.url], KEY, quick(3)).unwrap();
    let notes = only_wrong
        .tenant(&tenant("acme"))
        .collection(&notes_named());
    let failed = notes.nearest(&nearest).await.unwrap_err();
    assert!(failed.is_model_unavailable(), "{failed}");
    assert_eq!(failed.reason(), Some(Reason::Setup));
    let by_words = notes.hybrid(&hybrid).await.unwrap();
    assert_eq!(
        (by_words.degraded, by_words.degraded_by),
        (true, Some(Code::ModelUnavailable))
    );
    assert!(by_words.results[0].distance.is_none());
}

/// A node of 0.1 ignores a query it does not know on a `DELETE`, and
/// would delete whatever the condition; it refuses a change it does not
/// know, 400 `invalid`, its `Change` being `deny_unknown_fields` and its
/// delete having `collection` and `id` alone. So the client sends a
/// conditional delete as a batch, and against such a node is refused
/// rather than obeyed: here against a stand-in that answers as 0.1 does.
#[tokio::test]
async fn a_conditional_delete_goes_as_a_change_a_node_of_0_1_refuses_rather_than_ignores() {
    use axum::body::to_bytes;
    use axum::extract::Request;
    let deletes = Arc::new(Mutex::new(0usize));
    let posts = Arc::new(Mutex::new(Vec::<String>::new()));
    let (deleted, posted) = (deletes.clone(), posts.clone());
    let applied =
        || json!({ "incarnation": 1, "position": 2, "outcomes": [{ "outcome": "applied" }] });
    let router = axum::Router::new().fallback(move |request: Request| {
        let (deleted, posted) = (deleted.clone(), posted.clone());
        async move {
            let method = request.method().clone();
            let path = request.uri().path().to_string();
            let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
            if method == Asked::DELETE {
                *deleted.lock().unwrap() += 1;
                return (Answered::OK, axum::Json(applied())).into_response();
            }
            if method == Asked::POST && path.ends_with("/changes") {
                posted
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&body).into_owned());
                #[derive(serde::Deserialize)]
                #[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
                #[allow(dead_code)]
                enum OldChange {
                    Delete { collection: String, id: uuid::Uuid },
                }
                #[derive(serde::Deserialize)]
                #[serde(deny_unknown_fields)]
                #[allow(dead_code)]
                struct OldBatch {
                    changes: Vec<OldChange>,
                }
                return match serde_json::from_slice::<OldBatch>(&body) {
                    Ok(_) => (Answered::OK, axum::Json(applied())).into_response(),
                    Err(error) => {
                        let message = format!("Failed to deserialize the JSON body: {error}");
                        let refused = json!({ "error": { "code": "invalid", "message": message } });
                        (Answered::BAD_REQUEST, axum::Json(refused)).into_response()
                    }
                };
            }
            (Answered::NOT_FOUND, axum::Json(failure("not_found"))).into_response()
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let client = Client::connect(&[&url], KEY, quick(3)).unwrap();
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let at = "0123456789abcdef0123456789abcdef".parse().unwrap();
    let refused = notes.delete_if_version(oid(1), at).await.unwrap_err();
    assert!(
        matches!(&refused, Error::Api { status, code: Code::Invalid, .. } if status.as_u16() == 400),
        "{refused:?}"
    );
    assert_eq!(refused.code(), Some(Code::Invalid));
    assert!(!refused.is_unknown_outcome(), "final, not uncertain");
    assert_eq!(*deletes.lock().unwrap(), 0, "nothing deleted");
    let sent = posts.lock().unwrap().clone();
    assert_eq!(sent.len(), 1, "one try: a 400 is final");
    assert!(
        sent[0].contains("\"only_if_version\":\"0123456789abcdef0123456789abcdef\""),
        "{sent:?}"
    );
    assert!(sent[0].contains("\"op\":\"delete\""), "{sent:?}");
    // A plain delete still goes as a DELETE and is made.
    let made = notes.delete(oid(1)).await.unwrap();
    assert_eq!(made.outcomes, vec![Outcome::Applied]);
    assert_eq!(*deletes.lock().unwrap(), 1);
    assert_eq!(posts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_node_that_does_not_serve_the_tenant_is_not_asked_again_and_a_read_goes_elsewhere() {
    let misdirected = || (421, failure("not_served_here"));
    let elsewhere = Stub::answering(move |_, _| misdirected()).await;
    let serving = Stub::answering(|_, _| (200, json!({ "collections": [] }))).await;

    // A write is refused at the first node, once, and nothing was done.
    let client = Client::connect(&[&elsewhere.url, &serving.url], KEY, tries(4)).unwrap();
    let notes = client.tenant(&tenant("acme")).collection(&notes_named());
    let failed = tokio::time::timeout(
        Duration::from_secs(30),
        notes.put(oid(1), &content("fox", 1, None)),
    )
    .await
    .expect("failed at once, with no wait")
    .unwrap_err();
    assert_eq!(failed.code(), Some(Code::NotServedHere), "{failed}");
    assert!(!failed.is_unknown_outcome());
    assert_eq!((elsewhere.asked("acme"), serving.asked("acme")), (1, 0));

    // A read is asked of the next node at once, which serves it.
    let acme = client.tenant(&tenant("acme"));
    tokio::time::timeout(Duration::from_secs(30), acme.collections())
        .await
        .expect("no wait before another node is asked")
        .unwrap();
    assert_eq!((elsewhere.asked("acme"), serving.asked("acme")), (2, 1));
}
