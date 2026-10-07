//! Putting a tenant back as it stood, through the API.

mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

const POINTS: &str = "/v1/tenants/acme/restore-points";
const RESTORE: &str = "/v1/tenants/acme/restore";

fn titled(n: u64) -> Value {
    json!({ "properties": { "title": format!("note {n}"), "n": n }, "vector": [1.0, 0.0] })
}

async fn titles(served: &Served) -> Vec<String> {
    let listed = served
        .ok(
            Method::POST,
            &format!("{NOTES}/list"),
            Some(json!({ "limit": 10 })),
        )
        .await;
    listed["objects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|found| found["properties"]["title"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn a_tenant_is_put_back_by_position_or_moment_from_any_node() {
    let cluster = Cluster::new();
    let (holding, other) = (cluster.serve().await, cluster.serve().await);
    holding.with_notes(supplied()).await;
    holding.put(&object(1), titled(1)).await;
    let kept = holding.ok(Method::GET, POINTS, None).await["latest"].clone();
    holding.put(&object(2), titled(2)).await;
    holding.delete(&object(1)).await;
    assert_eq!(titles(&holding).await, ["note 2"]);

    let restored = other
        .ok(
            Method::POST,
            RESTORE,
            Some(json!({ "position": kept["position"] })),
        )
        .await;
    assert_eq!(restored["from"], kept);
    assert_eq!(titles(&holding).await, ["note 1"]);

    let points = holding.ok(Method::GET, POINTS, None).await;
    assert_eq!(points["latest"]["position"], restored["position"]);
    assert_eq!(points["window_secs"], json!(14 * 24 * 60 * 60));
    let undone = holding
        .ok(
            Method::POST,
            RESTORE,
            Some(json!({ "at": points["latest"]["at"] })),
        )
        .await;
    assert_eq!(undone["from"]["position"], restored["position"]);
    assert_eq!(titles(&holding).await, ["note 1"]);
}

#[tokio::test]
async fn a_point_out_of_reach_is_not_found_and_the_window_is_set_per_tenant() {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    served.put(&object(1), titled(1)).await;
    for (to, why) in [
        (json!({ "position": 999 }), "never committed"),
        (json!({ "at": "2999-01-01T00:00:00Z" }), "in the future"),
        (json!({ "at": "2000-01-01T00:00:00Z" }), "before the window"),
    ] {
        let (status, body) = served.post(RESTORE, to).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{why}: {body}");
    }
    let (status, _) = served.post(RESTORE, json!({ "everything": true })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = served
        .post("/v1/tenants/gamma/restore", json!({ "position": 1 }))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = served.get("/v1/tenants/gamma/restore-points").await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let window = "/v1/tenants/acme/restore-window";
    let (status, _) = served.put(window, json!({ "secs": 3_600 })).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let points = served.ok(Method::GET, POINTS, None).await;
    assert_eq!(points["window_secs"], json!(3_600));
    let (status, _) = served.put(window, json!({ "secs": 0 })).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = served.put(window, json!({ "secs": null })).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let points = served.ok(Method::GET, POINTS, None).await;
    assert_eq!(points["window_secs"], json!(14 * 24 * 60 * 60));
    let (status, _) = served
        .put("/v1/tenants/gamma/restore-window", json!({ "secs": 60 }))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
