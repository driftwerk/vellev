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

#[tokio::test]
async fn a_lost_tenant_whose_restore_failed_is_restored_when_asked_again() {
    let store = memory();
    let lost = Cluster::new().serve_backing_up_to(store.clone()).await;
    lost.with_notes(supplied()).await;
    lost.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let issued = lost.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let backup = lost.ok(Method::POST, BACKUPS, None).await;
    let files = store.list("_backups/acme/files/").await.unwrap();
    let hidden = files.first().unwrap();
    let kept = store.get(&hidden.key).await.unwrap();
    store.delete(&hidden.key).await.unwrap();

    let fresh = Cluster::new().serve_backing_up_to(store.clone()).await;
    let (status, _) = fresh.post(RESTORE, json!({ "backup": backup["id"] })).await;
    assert!(status.is_server_error(), "{status}");
    fresh.node.back_up_due().await.unwrap();
    let listed = fresh.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(
        listed["backups"],
        json!([backup.clone()]),
        "no backup of the empty tenant is taken in the meantime"
    );

    store.put(&hidden.key, kept).await.unwrap();
    fresh
        .ok(
            Method::POST,
            RESTORE,
            Some(json!({ "backup": backup["id"] })),
        )
        .await;
    assert_eq!(titles(&fresh).await, ["note 1"]);
    let (status, _) = fresh
        .call_as(issued["key"].as_str(), Method::GET, NOTES, None)
        .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn the_client_backs_up_lists_and_restores() {
    let served = Cluster::new().serve_backing_up_to(memory()).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let client = vellev_client::Client::connect(&[&served.url], KEY, Default::default()).unwrap();
    let acme = client.tenant(&vellev_core::TenantId::new("acme").unwrap());

    let backup = acme.back_up().await.unwrap();
    assert_eq!(acme.backups().await.unwrap(), std::slice::from_ref(&backup));
    let points = acme.restore_points().await.unwrap();
    let kept = points.latest.unwrap();
    served.delete(&object(1)).await;
    let restored = acme
        .restore(vellev_api::RestoreTo::Position(kept.position))
        .await
        .unwrap();
    assert_eq!(restored.from, kept);
    assert_eq!(titles(&served).await, ["note 1"]);
    acme.set_restore_window(Some(60)).await.unwrap();
    assert_eq!(acme.restore_points().await.unwrap().window_secs, 60);
}

/// What the command line prints, or why it failed.
async fn command(served: &Served, args: &[&str]) -> Result<Value, String> {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_vellev"));
    command
        .args([
            "backup",
            "--node",
            &served.url,
            "--key",
            KEY,
            "--tenant",
            "acme",
        ])
        .args(args);
    let output = tokio::task::spawn_blocking(move || command.output())
        .await
        .unwrap()
        .unwrap();
    match output.status.success() {
        true => Ok(serde_json::from_slice(&output.stdout).unwrap()),
        false => Err(String::from_utf8_lossy(&output.stderr).into_owned()),
    }
}

#[tokio::test]
async fn the_command_line_backs_up_and_restores() {
    let served = Cluster::new().serve_backing_up_to(memory()).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(titled(1))).await;

    let backup = command(&served, &["take"]).await.unwrap();
    assert_eq!(command(&served, &["list"]).await.unwrap(), json!([backup]));
    served.delete(&object(1)).await;
    let id = backup["id"].as_str().unwrap();
    let restored = command(&served, &["restore", "--backup", id])
        .await
        .unwrap();
    assert_eq!(restored["from"]["position"], backup["position"]);
    assert_eq!(titles(&served).await, ["note 1"]);
    let window = command(&served, &["window", "120"]).await.unwrap();
    assert_eq!(window["window_secs"], json!(120));

    let refused = command(&served, &["restore"]).await.unwrap_err();
    assert!(refused.contains("required arguments"), "{refused}");
    let refused = command(&served, &["restore", "--position", "1", "--backup", id])
        .await
        .unwrap_err();
    assert!(refused.contains("cannot be used with"), "{refused}");
}

#[tokio::test]
async fn removing_a_tenant_purges_its_files_backups_and_restore_points() {
    let store = memory();
    let cluster = Cluster::new();
    let served = cluster.serve_backing_up_to(store.clone()).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(titled(1))).await;
    served.ok(Method::POST, BACKUPS, None).await;
    served.ok(Method::DELETE, "/v1/tenants/acme", None).await;

    let (status, body) = served.put("/v1/tenants/acme", json!(null)).await;
    assert_eq!(
        (status, code(&body)),
        (StatusCode::CONFLICT, "conflict"),
        "not made again before it is purged"
    );
    served.node.purge_due().await.unwrap();
    assert!(store.list("_backups/acme/").await.unwrap().is_empty());
    assert!(cluster.blobs().list("acme/").await.unwrap().is_empty());

    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let points = served
        .ok(Method::GET, "/v1/tenants/acme/restore-points", None)
        .await;
    assert_eq!(points["earliest"], json!(null), "no restore point is left");
    let listed = served.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(listed["backups"], json!([]), "nor any backup");
}
