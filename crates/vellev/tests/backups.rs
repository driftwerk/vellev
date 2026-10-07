//! Backing tenants up to a second store and restoring them from it,
//! through the API.

mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};

const BACKUPS: &str = "/v1/tenants/acme/backups";
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
async fn a_tenant_is_backed_up_and_put_back_from_the_backup() {
    let served = Cluster::new().serve_backing_up_to(memory()).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let backup = served.ok(Method::POST, BACKUPS, None).await;
    served.delete(&object(1)).await;
    served.ok(Method::PUT, &object(2), Some(titled(2))).await;

    let listed = served.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(listed["backups"], json!([backup.clone()]));
    let restored = served
        .ok(
            Method::POST,
            RESTORE,
            Some(json!({ "backup": backup["id"] })),
        )
        .await;
    assert_eq!(restored["from"]["position"], backup["position"]);
    assert_eq!(titles(&served).await, ["note 1"]);
}

#[tokio::test]
async fn a_cluster_that_lost_a_tenant_gets_it_back_with_its_keys_and_window() {
    let store = memory();
    let lost = Cluster::new().serve_backing_up_to(store.clone()).await;
    lost.with_notes(supplied()).await;
    lost.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let issued = lost.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let window = json!({ "secs": 3_600 });
    lost.ok(Method::PUT, "/v1/tenants/acme/restore-window", Some(window))
        .await;
    let backup = lost.ok(Method::POST, BACKUPS, None).await;

    let fresh = Cluster::new().serve_backing_up_to(store).await;
    let (status, _) = fresh.get(NOTES).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    fresh
        .ok(
            Method::POST,
            RESTORE,
            Some(json!({ "backup": backup["id"] })),
        )
        .await;
    assert_eq!(titles(&fresh).await, ["note 1"]);
    let key = issued["key"].as_str().unwrap();
    let (status, _) = fresh.call_as(Some(key), Method::GET, NOTES, None).await;
    assert_eq!(status, StatusCode::OK, "the tenant's key reaches it again");
    let points = fresh
        .ok(Method::GET, "/v1/tenants/acme/restore-points", None)
        .await;
    assert_eq!(points["window_secs"], json!(3_600));
}

#[tokio::test]
async fn a_key_revoked_since_a_backup_stays_revoked_after_restoring_it() {
    let served = Cluster::new().serve_backing_up_to(memory()).await;
    served.with_notes(supplied()).await;
    let issued = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let backup = served.ok(Method::POST, BACKUPS, None).await;
    let revoke = format!("/v1/tenants/acme/keys/{}", issued["id"].as_str().unwrap());
    served.ok(Method::DELETE, &revoke, None).await;

    served
        .ok(
            Method::POST,
            RESTORE,
            Some(json!({ "backup": backup["id"] })),
        )
        .await;
    let (status, _) = served
        .call_as(issued["key"].as_str(), Method::GET, NOTES, None)
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn backups_need_a_store_a_tenant_and_a_backup_that_exists() {
    let without = Served::new().await;
    without.with_notes(supplied()).await;
    let (status, _) = without.post(BACKUPS, json!(null)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let served = Cluster::new().serve_backing_up_to(memory()).await;
    served.with_notes(supplied()).await;
    let (status, _) = served
        .post(RESTORE, json!({ "backup": uuid::Uuid::new_v4() }))
        .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = served.post("/v1/tenants/gamma/backups", json!(null)).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let issued = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let (status, _) = served
        .call_as(issued["key"].as_str(), Method::POST, BACKUPS, None)
        .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}
