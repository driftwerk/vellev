//! DW-668 PR 6: the clean floor. While a partition's history is shipped,
//! a clean deletes nothing at or above the cap the run holds it to, so
//! that every checkpoint yet to be shipped, and every file it lists, is
//! there to ship; and the cap follows what the run ships, so that what is
//! shipped is cleaned.

mod common;

use common::*;
use vellev_storage::coordinator::{ClusterId, Lineage, ShipBase};
use vellev_storage::format::{decode_head, Head as Stored};
use vellev_storage::shipper::{Batch, Shipper};
use vellev_storage::Merging;

fn merging(writer: Writer) -> Writer {
    writer.merging_with(Merging {
        fanout: 2,
        ..Merging::default()
    })
}

/// The checkpoints committed, oldest first, by position and key.
async fn checkpoints(cluster: &Cluster) -> Vec<(LogPosition, String)> {
    let now = cluster.committed().await.incarnation;
    let rows = (cluster.coordinator)
        .commits_after(&cluster.key, now, LogPosition(0), 1_000)
        .await
        .unwrap();
    rows.into_iter()
        .filter(|row| row.head.0.contains("/manifest/"))
        .map(|row| (row.position, row.head.0))
        .collect()
}

/// Every file the checkpoint under `key` lists: its segments' files and
/// the logs it replays.
async fn listed(cluster: &Cluster, key: &str) -> Vec<String> {
    let frame = cluster.blobs.get(key).await.unwrap();
    let Ok(Stored::Manifest(manifest)) = decode_head(key, &frame) else {
        panic!("a manifest")
    };
    let mut files: Vec<String> = manifest
        .segments
        .iter()
        .flat_map(|segment| {
            [
                Some(segment.key.clone()),
                segment.index.clone(),
                segment.vectors.clone(),
            ]
        })
        .flatten()
        .collect();
    files.extend(manifest.replay);
    files
}

async fn there(cluster: &Cluster, key: &str) -> bool {
    match cluster.blobs.get(key).await {
        Ok(_) => true,
        Err(StorageError::Missing { .. }) => false,
        Err(error) => panic!("{error}"),
    }
}

/// Held for shipping, based at the checkpoint at `at` (its own replay
/// start: these checkpoints replay nothing), the cap there.
async fn based_at(cluster: &Cluster, at: LogPosition, checkpoint: &str) {
    let now = cluster.committed().await.incarnation;
    let c = &cluster.coordinator;
    c.hold_for_shipping(&cluster.key, now, at).await.unwrap();
    let base = ShipBase {
        position: at,
        checkpoint: ManifestRef(checkpoint.into()),
        checkpoint_at: at,
    };
    c.record_base(&cluster.key, now, &base).await.unwrap();
}

/// Three checkpoints, the last two a flush and a merge of the first two's
/// segments, and one more after.
async fn checkpointed(cluster: &Cluster, writer: &Writer) {
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    assert!(writer.merge().await.unwrap().is_some());
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    writer.flush().await.unwrap();
    assert!(checkpoints(cluster).await.len() >= 4);
}

#[tokio::test]
async fn a_clean_held_for_shipping_deletes_nothing_at_or_above_the_cap() {
    // Unheld, a clean takes the first checkpoint and what only it lists.
    let cluster = Cluster::new().await;
    let writer = merging(cluster.writer().await);
    checkpointed(&cluster, &writer).await;
    let (_, first) = checkpoints(&cluster).await[0].clone();
    assert!(writer.clean(Duration::ZERO).await.unwrap() > 0);
    assert!(!there(&cluster, &first).await);

    // Held at it: it stays, and every file it and every checkpoint after
    // it lists.
    let cluster = Cluster::new().await;
    let writer = merging(cluster.writer().await);
    checkpointed(&cluster, &writer).await;
    let all = checkpoints(&cluster).await;
    let (at, first) = all[0].clone();
    based_at(&cluster, at, &first).await;
    let mut kept = Vec::new();
    for (_, key) in &all {
        kept.push(key.clone());
        kept.extend(listed(&cluster, key).await);
    }
    writer.clean(Duration::ZERO).await.unwrap();
    for key in &kept {
        assert!(there(&cluster, key).await, "{key}");
    }
    let cleaned = cluster.coordinator.cleaned(&cluster.key).await.unwrap();
    let below = cleaned.as_ref().map(|cleaned| cleaned.position);
    assert!(below.is_none_or(|below| below <= at), "{cleaned:?}");
}

fn shipper(cluster: &Cluster, backup: &Arc<dyn Blobs>) -> Shipper {
    Shipper::new(
        cluster.blobs.clone(),
        backup.clone(),
        cluster.coordinator.clone(),
        cluster.key.clone(),
        Lineage(uuid::Uuid::from_u128(1)),
        ClusterId(uuid::Uuid::from_u128(2)),
    )
}

async fn ship_all(cluster: &Cluster, shipper: &Shipper) {
    let now = cluster.committed().await.incarnation;
    let committed = cluster.committed().await.log_position;
    let shipped = shipper.ship_batch(now, 1_000).await.unwrap();
    assert_eq!(
        shipped,
        Batch::Shipped {
            through: Some(committed),
            stalled: None
        }
    );
}

#[tokio::test]
async fn a_put_back_above_the_watermark_keeps_its_old_keys_and_once_shipped_they_go() {
    let cluster = Cluster::new().await;
    let writer = merging(cluster.writer().await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let (first_at, first) = checkpoints(&cluster).await[0].clone();
    based_at(&cluster, first_at, &first).await;
    let old = listed(&cluster, &first).await;
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    assert!(writer.merge().await.unwrap().is_some());
    let (merged_at, merged) = checkpoints(&cluster).await.last().cloned().unwrap();
    // Shipped through the merge: the cap follows it there.
    let backup: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let shipper = shipper(&cluster, &backup);
    ship_all(&cluster, &shipper).await;
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!(shipping.cap, Some(merged_at));
    // Put back to the first checkpoint: a checkpoint above the watermark
    // listing the first's files again, under their old keys.
    writer
        .restore(Some(&ManifestRef(first.clone())), first_at)
        .await
        .unwrap();
    let (put_back_at, put_back) = checkpoints(&cluster).await.last().cloned().unwrap();
    assert!(put_back_at > merged_at);
    let relisted = listed(&cluster, &put_back).await;
    assert!(
        old.iter().any(|key| relisted.contains(key)),
        "{old:?} {relisted:?}"
    );
    // A clean held to the merge keeps them: the put-back checkpoint is on
    // the chain, above the cap, and lists them.
    writer.clean(Duration::ZERO).await.unwrap();
    for key in &relisted {
        assert!(there(&cluster, key).await, "{key}");
    }
    let merged_files = listed(&cluster, &merged).await;
    assert!(there(&cluster, &merged).await);
    // Once the put-back is shipped, the cap follows it, and the next clean
    // takes what only the merge listed.
    ship_all(&cluster, &shipper).await;
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!(shipping.cap, Some(put_back_at));
    writer.clean(Duration::ZERO).await.unwrap();
    let cleaned = cluster
        .coordinator
        .cleaned(&cluster.key)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cleaned.position, put_back_at);
    assert!(!there(&cluster, &merged).await);
    for key in merged_files.iter().filter(|key| !relisted.contains(key)) {
        assert!(!there(&cluster, key).await, "{key}");
    }
    for key in &relisted {
        assert!(there(&cluster, key).await, "{key}");
    }
}

#[tokio::test]
async fn a_clean_whose_record_a_hold_has_capped_meanwhile_is_refused_and_deletes_nothing() {
    let cluster = Cluster::new().await;
    let coordinator = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let writer = merging(cluster.writer_on(coordinator.clone()).await);
    checkpointed(&cluster, &writer).await;
    let (first_at, _) = checkpoints(&cluster).await[0];
    let before: Vec<_> = (cluster.blobs.list("").await.unwrap().into_iter())
        .map(|file| file.key)
        .collect();
    // The hold is taken after the clean read the cap, none, and before it
    // records what it deletes below.
    *coordinator.hold_on_clean.lock().unwrap() = Some(first_at);
    let refused = writer.clean(Duration::ZERO).await;
    assert!(
        matches!(
            refused,
            Err(StorageError::Coordinator(
                CoordinatorError::CleanAboveCap { .. }
            ))
        ),
        "{refused:?}"
    );
    let after: Vec<_> = (cluster.blobs.list("").await.unwrap().into_iter())
        .map(|file| file.key)
        .collect();
    assert_eq!(after, before);
}

#[tokio::test]
async fn a_clean_that_cannot_read_the_cap_fails_and_deletes_nothing() {
    let cluster = Cluster::new().await;
    let coordinator = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let writer = merging(cluster.writer_on(coordinator.clone()).await);
    checkpointed(&cluster, &writer).await;
    let before = cluster.blobs.list("").await.unwrap().len();
    coordinator.fail_shipping.store(true, Ordering::SeqCst);
    assert!(writer.clean(Duration::ZERO).await.is_err());
    assert_eq!(cluster.blobs.list("").await.unwrap().len(), before);
    coordinator.fail_shipping.store(false, Ordering::SeqCst);
    assert!(writer.clean(Duration::ZERO).await.unwrap() > 0);
}
