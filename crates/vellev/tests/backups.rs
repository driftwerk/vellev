//! Backing tenants up to a second store and restoring them from it,
//! through the API.

mod common;

use std::time::Duration;

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
    let listed = fresh.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(
        listed["backups"],
        json!([backup.clone()]),
        "what is kept under the name of a tenant it does not have"
    );
    let own = json!({ "backup": backup["id"] });
    let (status, _) = fresh.post(RESTORE, own.clone()).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "a tenant is restored from its own backups, and there is none"
    );
    let (status, _) = fresh.get(NOTES).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "none is made unasked");

    let adopt = json!({ "adopt": { "backup": backup["id"], "with_keys": true } });
    fresh.ok(Method::POST, RESTORE, Some(adopt.clone())).await;
    assert_eq!(titles(&fresh).await, ["note 1"]);
    let key = issued["key"].as_str();
    let (status, _) = fresh.call_as(key, Method::GET, NOTES, None).await;
    assert_eq!(status, StatusCode::OK, "the tenant's key reaches it again");
    let (status, _) = fresh.call_as(key, Method::POST, RESTORE, Some(adopt)).await;
    assert_eq!(
        status,
        StatusCode::FORBIDDEN,
        "to adopt a backup is for whoever runs the cluster"
    );
    fresh.delete(&object(1)).await;
    fresh.ok(Method::POST, RESTORE, Some(own)).await;
    assert_eq!(titles(&fresh).await, ["note 1"], "its own backup now");
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
async fn a_node_served_for_these_tests_backs_nothing_up_of_its_own_accord() {
    let cluster = Cluster::new();
    // Written through a node since stopped, so that none holds the tenant
    // while the clock is moved on.
    let writing = cluster.serve().await;
    writing.with_notes(supplied()).await;
    writing.ok(Method::PUT, &object(1), Some(titled(1))).await;
    writing.node.shutdown().await;
    writing.stop().await;
    let served = cluster.serve_backing_up_to(memory()).await;

    // Longer than a node waits before it first looks for tenants to back
    // up, and long enough after for it to have backed one up.
    tokio::time::pause();
    tokio::time::sleep(Duration::from_secs(11 * 60)).await;
    tokio::time::resume();
    let acme = vellev_core::TenantId::new("acme").unwrap();
    assert_eq!(served.node.backups(&acme).await.unwrap(), []);
}

#[tokio::test]
async fn a_tenant_with_nothing_committed_says_so_rather_than_that_it_is_busy() {
    let served = Cluster::new().serve_backing_up_to(memory()).await;
    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    let (status, body) = served.post(BACKUPS, json!(null)).await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["error"]["code"], "empty", "{body}");
    let listed = served.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(listed["backups"], json!([]));
}

#[tokio::test]
async fn a_lost_tenant_whose_restore_failed_is_restored_when_asked_again() {
    let store = memory();
    let lost = Cluster::new().serve_backing_up_to(store.clone()).await;
    lost.with_notes(supplied()).await;
    lost.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let issued = lost.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let backup = lost.ok(Method::POST, BACKUPS, None).await;
    let kept_of = store.list("_backups/acme/").await.unwrap();
    let mut files = kept_of.iter().filter(|kept| kept.key.contains("/data/"));
    let hidden = files.next().unwrap();
    let kept = store.get(&hidden.key).await.unwrap();
    store.delete(&hidden.key).await.unwrap();

    let fresh = Cluster::new().serve_backing_up_to(store.clone()).await;
    let adopt = json!({ "adopt": { "backup": backup["id"], "with_keys": true } });
    // Refused as a fact about the backup, for whoever restores in a hurry
    // to act on: which backup lacks which file, over the API and on the
    // command line alike.
    let (status, body) = fresh.post(RESTORE, adopt.clone()).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["code"], "conflict", "{body}");
    let id = backup["id"].as_str().unwrap();
    let file = hidden.key.rsplit_once("/acme/").unwrap().1;
    for told in [
        body["error"]["message"].as_str().unwrap().to_string(),
        command(&fresh, &["restore", "--adopt", id])
            .await
            .unwrap_err(),
    ] {
        assert!(told.contains(id), "{told}");
        assert!(told.contains(file), "{told}");
        assert!(told.contains("not in the backup store"), "{told}");
    }
    let keys = fresh.ok(Method::GET, "/v1/tenants/acme/keys", None).await;
    assert_eq!(
        keys["keys"],
        json!([]),
        "no key of the backup's is a tenant's that has not taken it up"
    );
    fresh.node.back_up_due().await.unwrap();
    let listed = fresh.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(
        listed["backups"],
        json!([backup.clone()]),
        "no backup of the empty tenant is taken in the meantime"
    );

    store.put(&hidden.key, kept).await.unwrap();
    fresh.ok(Method::POST, RESTORE, Some(adopt)).await;
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
    let refused = command(&served, &["restore", "--adopt", id, "--backup", id])
        .await
        .unwrap_err();
    assert!(refused.contains("cannot be used with"), "{refused}");
}

#[tokio::test]
async fn the_command_line_adopts_a_backup_into_a_cluster_that_lost_the_tenant() {
    let store = memory();
    let lost = Cluster::new().serve_backing_up_to(store.clone()).await;
    lost.with_notes(supplied()).await;
    lost.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let backup = command(&lost, &["take"]).await.unwrap();
    let id = backup["id"].as_str().unwrap();

    let fresh = Cluster::new().serve_backing_up_to(store).await;
    assert_eq!(command(&fresh, &["list"]).await.unwrap(), json!([backup]));
    let refused = command(&fresh, &["restore", "--backup", id]).await;
    assert!(refused.unwrap_err().contains("does not exist"));
    let refused = command(&fresh, &["restore", "--backup", id, "--with-keys"]).await;
    assert!(refused.is_err(), "keys come only with adopting");
    let restored = command(&fresh, &["restore", "--adopt", id]).await.unwrap();
    assert_eq!(restored["from"]["position"], backup["position"]);
    assert_eq!(titles(&fresh).await, ["note 1"]);
}

#[tokio::test]
async fn adopting_a_backup_gives_its_keys_only_when_asked_through_the_api_and_the_command_line() {
    let store = memory();
    let lost = Cluster::new().serve_backing_up_to(store.clone()).await;
    lost.with_notes(supplied()).await;
    lost.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let issued = lost.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let backup = lost.ok(Method::POST, BACKUPS, None).await;
    let id = backup["id"].as_str().unwrap();
    let reaches = |served: Served| async move {
        let key = issued["key"].as_str();
        let (status, _) = served.call_as(key, Method::GET, NOTES, None).await;
        assert_eq!(titles(&served).await, ["note 1"]);
        status
    };

    let plain = Cluster::new().serve_backing_up_to(store.clone()).await;
    let adopt = json!({ "adopt": { "backup": id } });
    plain.ok(Method::POST, RESTORE, Some(adopt)).await;
    let listed = plain.ok(Method::GET, "/v1/tenants/acme/keys", None).await;
    assert_eq!(listed["keys"], json!([]), "no key unless asked for");

    let by_line = Cluster::new().serve_backing_up_to(store.clone()).await;
    command(&by_line, &["restore", "--adopt", id])
        .await
        .unwrap();
    let asked = Cluster::new().serve_backing_up_to(store.clone()).await;
    command(&asked, &["restore", "--adopt", id, "--with-keys"])
        .await
        .unwrap();
    let statuses = [
        reaches.clone()(plain).await,
        reaches.clone()(by_line).await,
        reaches(asked).await,
    ];
    let (no, yes) = (StatusCode::UNAUTHORIZED, StatusCode::OK);
    assert_eq!(statuses, [no, no, yes]);
}

/// Everything kept under acme's name in `store`.
async fn kept_of_acme(store: &std::sync::Arc<dyn vellev_storage::blobs::Blobs>) -> Vec<String> {
    let kept = store.list("_backups/acme/").await.unwrap();
    kept.into_iter().map(|kept| kept.key).collect()
}

#[tokio::test]
async fn a_tenant_made_again_is_given_neither_the_backups_nor_the_keys_of_the_one_before() {
    let store = memory();
    let cluster = Cluster::new();
    let served = cluster.serve_backing_up_to(store.clone()).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(titled(1))).await;
    let old_key = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    let backup = served.ok(Method::POST, BACKUPS, None).await;
    // All of the backup lands again later, as writes given up on may.
    let mut late = Vec::new();
    for key in kept_of_acme(&store).await {
        late.push((key.clone(), store.get(&key).await.unwrap()));
    }
    served.ok(Method::DELETE, "/v1/tenants/acme", None).await;
    cluster.pass(backup_claim());
    served.node.purge_due().await.unwrap();
    assert!(kept_of_acme(&store).await.is_empty());

    served.ok(Method::PUT, "/v1/tenants/acme", None).await;
    for (key, bytes) in late {
        store.put(&key, bytes).await.unwrap();
    }
    let listed = served.ok(Method::GET, BACKUPS, None).await;
    assert_eq!(
        listed["backups"],
        json!([backup.clone()]),
        "never written to, it is shown what an admin may have it take up"
    );
    let own = json!({ "backup": backup["id"] });
    let (status, _) = served.post(RESTORE, own).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = served
        .call_as(old_key["key"].as_str(), Method::GET, NOTES, None)
        .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Once it is written to, not even whoever runs the cluster can have it
    // take the backup up.
    let adopt = json!({ "adopt": { "backup": backup["id"], "with_keys": true } });
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(2), Some(titled(2))).await;
    let (status, body) = served.post(RESTORE, adopt).await;
    assert_eq!((status, code(&body)), (StatusCode::CONFLICT, "conflict"));
    assert_eq!(titles(&served).await, ["note 2"]);
}

#[tokio::test]
async fn a_node_started_backs_tenants_up_and_purges_those_removed_of_its_own_accord() {
    let store = memory();
    let cluster = Cluster::new();
    let served = cluster.serve_started_backing_up_to(store).await;
    served.with_notes(supplied()).await;
    served.ok(Method::PUT, &object(1), Some(titled(1))).await;
    served.ok(Method::PUT, "/v1/tenants/gone", None).await;
    served.ok(Method::DELETE, "/v1/tenants/gone", None).await;
    cluster.pass(backup_claim());

    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        let listed = served.ok(Method::GET, BACKUPS, None).await;
        let backed_up = listed["backups"].as_array().unwrap().len();
        let (purged, _) = served.put("/v1/tenants/gone", json!(null)).await;
        if backed_up == 1 && purged.is_success() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "backups: {backed_up}; making the tenant removed again: {purged}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
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

    let again = served.http.put(format!("{}/v1/tenants/acme", served.url));
    let refused = again.bearer_auth(KEY).send().await.unwrap();
    let wait = refused.headers().get("retry-after").cloned();
    let status = refused.status();
    let body: Value = refused.json().await.unwrap();
    assert_eq!(
        (status, code(&body)),
        (StatusCode::CONFLICT, "purging"),
        "not made again before it is purged"
    );
    let until_due = backup_claim() + vellev_node::PURGES_LOOKED_FOR_EVERY;
    assert_eq!(
        wait.unwrap().to_str().unwrap(),
        until_due.as_secs().to_string(),
        "and told when to ask again: once its purge is due and a node has looked"
    );
    served.node.purge_due().await.unwrap();
    assert!(
        !store.list("_backups/acme/").await.unwrap().is_empty(),
        "nor purged while a backup of it could still hold its claim"
    );
    cluster.pass(backup_claim());
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
