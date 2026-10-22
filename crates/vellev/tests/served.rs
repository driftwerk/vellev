//! A node serves the tenants it is told to and no other: a request for any
//! other is refused, read or write, and sent on to no node.

mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

const MISDIRECTED: StatusCode = StatusCode::MISDIRECTED_REQUEST;

fn fox(n: u64) -> Value {
    json!({ "properties": { "title": "a red fox", "n": n }, "vector": [1.0, 0.0] })
}

/// Whether the answer is the refusal of a tenant the node does not serve.
fn not_served((status, body): &(StatusCode, Value)) -> bool {
    *status == MISDIRECTED && code(body) == "not_served_here"
}

#[tokio::test]
async fn a_node_refuses_reads_and_writes_of_a_tenant_another_holds_and_sends_none_on() {
    let cluster = Cluster::new();
    let acme = cluster.serve_serving(&["acme"], None).await;
    let memories = cluster.serve_serving(&["memories"], None).await;
    acme.with_notes(supplied()).await;
    acme.ok(Method::PUT, &object(1), Some(fox(1))).await;

    // A write, which would have gone on to the holder.
    let written = memories.put(&object(2), fox(2)).await;
    assert!(not_served(&written), "{written:?}");
    assert_eq!(
        written.1["error"]["message"],
        "tenant acme is not served by this node; ask a node that serves it"
    );
    let changed = memories
        .post("/v1/tenants/acme/changes", json!({ "changes": [] }))
        .await;
    assert!(not_served(&changed), "{changed:?}");
    assert!(not_served(&memories.delete(&object(1)).await));
    assert_eq!(
        acme.get(&object(2)).await.0,
        StatusCode::NOT_FOUND,
        "nothing reached the holder"
    );
    // The holder still holds it, and writes on.
    let written = acme.ok(Method::PUT, &object(2), Some(fox(2))).await;
    assert_eq!(written["position"], 3);

    // Reads: a search, an object, the schema and the collections.
    let nearest = json!({ "vector": [1.0, 0.0], "limit": 1 });
    for query in ["nearest", "list", "count"] {
        let body = match query {
            "nearest" => nearest.clone(),
            _ => json!({}),
        };
        let asked = memories.post(&format!("{NOTES}/{query}"), body).await;
        assert!(not_served(&asked), "{query}: {asked:?}");
    }
    for path in [
        object(1),
        NOTES.to_string(),
        "/v1/tenants/acme/collections".into(),
    ] {
        let read = memories.get(&path).await;
        assert!(not_served(&read), "{path}: {read:?}");
    }
    assert_eq!(memories.node.copies(), 0, "no copy of acme followed");
    assert_eq!(memories.node.resident(), 0, "nor anything of it held");

    // A caller without a key learns nothing of where tenants are served.
    let (status, _) = memories.call_as(None, Method::GET, &object(1), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn every_route_of_a_tenant_the_node_does_not_serve_is_refused() {
    let cluster = Cluster::new();
    let backups = memory();
    let acme = cluster
        .serve_serving(&["acme"], Some(backups.clone()))
        .await;
    let memories = cluster
        .serve_serving(&["memories"], Some(backups.clone()))
        .await;
    acme.with_notes(supplied()).await;
    acme.ok(Method::PUT, &object(1), Some(fox(1))).await;
    acme.ok(Method::POST, "/v1/tenants/acme/keys", None).await;

    let asked = [
        (Method::PUT, "/v1/tenants/acme", None),
        (Method::DELETE, "/v1/tenants/acme", None),
        (Method::PUT, "/v1/tenants/beta", None),
        (Method::GET, "/v1/tenants/acme/keys", None),
        (Method::POST, "/v1/tenants/acme/keys", None),
        (
            Method::DELETE,
            "/v1/tenants/acme/keys/00000000-0000-0000-0000-000000000000",
            None,
        ),
        (Method::GET, "/v1/tenants/acme/restore-points", None),
        (
            Method::PUT,
            "/v1/tenants/acme/restore-window",
            Some(json!({ "secs": 60 })),
        ),
        (Method::GET, "/v1/tenants/acme/backups", None),
        (Method::POST, "/v1/tenants/acme/backups", None),
        (
            Method::POST,
            "/v1/tenants/acme/restore",
            Some(json!({ "position": 1 })),
        ),
    ];
    for (method, path, body) in asked {
        let answer = memories.call(method.clone(), path, body).await;
        assert!(not_served(&answer), "{method} {path}: {answer:?}");
    }
    // None of it was done.
    assert_eq!(acme.get(&object(1)).await.0, StatusCode::OK);
    let keys = acme.ok(Method::GET, "/v1/tenants/acme/keys", None).await;
    assert_eq!(keys["keys"].as_array().unwrap().len(), 1);
    let kept = acme.ok(Method::GET, "/v1/tenants/acme/backups", None).await;
    assert_eq!(kept["backups"], json!([]));
    assert_eq!(
        acme.get("/v1/tenants/beta/collections").await.0,
        MISDIRECTED
    );
    let listed = memories.ok(Method::GET, "/v1/tenants?limit=10", None).await;
    assert_eq!(
        listed["tenants"],
        json!(["acme"]),
        "what is not a tenant's is answered"
    );

    // Where it is served, the same routes are answered.
    acme.ok(Method::POST, "/v1/tenants/acme/backups", None)
        .await;
    acme.ok(Method::GET, "/v1/tenants/acme/restore-points", None)
        .await;
}

#[tokio::test]
async fn a_tenant_every_node_serves_is_sent_on_to_its_holder_as_before() {
    let cluster = Cluster::new();
    let every = cluster.serve().await;
    let listing = cluster.serve_serving(&["acme"], None).await;
    every.with_notes(supplied()).await;
    let written = listing.ok(Method::PUT, &object(1), Some(fox(1))).await;
    assert_eq!(written["position"], 2, "sent on to the holder");
    assert_eq!(every.get(&object(1)).await.1["properties"]["n"], 1);
    let searched = listing
        .ok(
            Method::POST,
            &format!("{NOTES}/nearest"),
            Some(json!({ "vector": [1.0, 0.0], "limit": 1 })),
        )
        .await;
    assert_eq!(searched["results"][0]["id"], id(1));
}
