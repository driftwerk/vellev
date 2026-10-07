//! A partition put back as it stood at an earlier head: a checkpoint past
//! where it is, which writers, followers and cleaning up all take as any
//! other.

mod common;

use common::*;
use vellev_core::RequestId;
use vellev_storage::{Batch, Embedded, Merging};

/// A partition's head and position, as a restore names it.
async fn now(cluster: &Cluster) -> (ManifestRef, LogPosition) {
    let committed = cluster.committed().await;
    (committed.manifest.unwrap(), committed.log_position)
}

fn ns(state: &State, objects: &[u8]) -> Vec<Option<i64>> {
    objects
        .iter()
        .map(|object| n_of(state, "notes", *object))
        .collect()
}

/// Writes each merge folding two segments into one.
async fn writer(cluster: &Cluster) -> Writer {
    cluster.writer().await.merging_with(Merging {
        fanout: 2,
        ..Merging::default()
    })
}

/// Notes 1 and 2 flushed into a segment, then note 2 changed and note 3
/// added, in memory: the head to go back to. Then all of them changed,
/// flushed and merged, and note 1 deleted.
async fn history(cluster: &Cluster, writer: &Writer) -> (ManifestRef, LogPosition) {
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 1),
            upsert("notes", 2, 2),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .commit(vec![upsert("notes", 2, 20), upsert("notes", 3, 3)])
        .await
        .unwrap();
    let target = now(cluster).await;
    writer
        .commit(vec![
            upsert("notes", 1, 100),
            upsert("notes", 2, 200),
            upsert("notes", 3, 300),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    assert!(writer.merge().await.unwrap().is_some());
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    target
}

#[tokio::test]
async fn a_partition_is_put_back_as_it_stood_and_goes_on_from_there() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    let before = writer.partition().read(State::position);

    let restored = writer.restore(&head, position).await.unwrap();
    assert_eq!(restored, before.next().unwrap(), "past where it was");
    let read = |state: &State| ns(state, &[1, 2, 3]);
    assert_eq!(writer.partition().read(read), [Some(1), Some(20), Some(3)]);
    assert_eq!(cluster.committed().await.log_position, restored);

    writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
    let follower = cluster.open().await;
    assert_eq!(
        follower.partition().read(|state| ns(state, &[1, 2, 3, 4])),
        [Some(1), Some(20), Some(3), Some(4)],
        "read afresh from storage"
    );
}

#[tokio::test]
async fn a_follower_catches_up_through_a_restore() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    let follower = cluster.open().await;
    assert_eq!(follower.partition().read(|state| ns(state, &[1])), [None]);

    writer.restore(&head, position).await.unwrap();
    writer.commit(vec![upsert("notes", 3, 30)]).await.unwrap();
    let committed = cluster.committed().await;
    assert_eq!(
        follower.catch_up(&committed).await.unwrap(),
        committed.log_position
    );
    assert_eq!(
        follower.partition().read(|state| ns(state, &[1, 2, 3])),
        [Some(1), Some(20), Some(30)]
    );
}

#[tokio::test]
async fn a_restore_can_be_undone_by_restoring_to_before_it() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    let (latest, latest_at) = now(&cluster).await;

    writer.restore(&head, position).await.unwrap();
    writer.restore(&latest, latest_at).await.unwrap();
    assert_eq!(
        writer.partition().read(|state| ns(state, &[1, 2, 3])),
        [None, Some(200), Some(300)]
    );
}

#[tokio::test]
async fn requests_answered_before_a_restore_stay_answered() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    let request = RequestId::generate();
    let batch = || Batch {
        request: Some(request),
        mutations: vec![upsert("notes", 5, 5)],
    };
    writer
        .commit_batches(vec![batch()], Embedded::new())
        .await
        .unwrap();

    writer.restore(&head, position).await.unwrap();
    assert!(writer.partition().read(|state| state.answered(request)));
    writer
        .commit_batches(vec![batch()], Embedded::new())
        .await
        .unwrap();
    assert_eq!(
        writer.partition().read(|state| ns(state, &[5])),
        [None],
        "asked again, the request is answered as before and not made again"
    );
}

#[tokio::test]
async fn cleaning_up_keeps_what_a_restore_needs_however_old() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    writer.restore(&head, position).await.unwrap();

    let deleted = writer.clean(Duration::ZERO).await.unwrap();
    assert!(deleted > 0, "what nothing needs goes");
    assert_eq!(
        cluster
            .open()
            .await
            .partition()
            .read(|state| ns(state, &[1, 2, 3])),
        [Some(1), Some(20), Some(3)]
    );

    writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
    writer.flush().await.unwrap();
    while writer.merge().await.unwrap().is_some() {}
    writer.clean(Duration::ZERO).await.unwrap();
    assert_eq!(
        cluster
            .open()
            .await
            .partition()
            .read(|state| ns(state, &[1, 2, 3, 4])),
        [Some(1), Some(20), Some(3), Some(4)]
    );
}

/// A store whose files are all old except those written at or after a
/// position, as if that much time had gone by since the rest were.
#[derive(Debug)]
struct Aged {
    inner: Arc<dyn Blobs>,
    young_from: AtomicUsize,
}

#[async_trait]
impl Blobs for Aged {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        let young_from = self.young_from.load(Ordering::SeqCst);
        let position = |key: &str| {
            let name = key.rsplit('/').next()?;
            name.get(..20)?.parse::<usize>().ok()
        };
        Ok(self
            .inner
            .list(prefix)
            .await?
            .into_iter()
            .map(|listed| Listed {
                modified: match position(&listed.key).is_some_and(|at| at >= young_from) {
                    true => listed.modified,
                    false => Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap(),
                },
                ..listed
            })
            .collect())
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_restore_keeps_its_old_segments_once_its_target_has_aged_out() {
    let aged = Arc::new(Aged {
        inner: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
        young_from: AtomicUsize::new(0),
    });
    let cluster = Cluster {
        blobs: aged.clone(),
        ..Cluster::new().await
    };
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    let (latest, latest_at) = now(&cluster).await;
    let restored = writer.restore(&head, position).await.unwrap();
    writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();

    // Only the restore and what came after is recent: the latest
    // checkpoint before it is the merge, which no longer lists the
    // segment the restore brought back.
    aged.young_from.store(restored.0 as usize, Ordering::SeqCst);
    writer.clean(Duration::from_secs(60 * 60)).await.unwrap();
    assert_eq!(
        cluster
            .open()
            .await
            .partition()
            .read(|state| ns(state, &[1, 2, 3, 4])),
        [Some(1), Some(20), Some(3), Some(4)]
    );
    // The merge, which the restore follows, and the restore: the
    // checkpoints before the merge only states gone out of the window need.
    let checkpoints = cluster
        .blobs
        .list(&Layout(&cluster.key).prefix())
        .await
        .unwrap()
        .into_iter()
        .filter(|listed| listed.key.ends_with(".manifest"))
        .count();
    assert_eq!(checkpoints, 2);
    writer.restore(&latest, latest_at).await.unwrap();
    assert_eq!(
        writer.partition().read(|state| ns(state, &[1, 2, 3])),
        [None, Some(200), Some(300)],
        "how it stood just before the restore is restorable still"
    );
}

#[tokio::test]
async fn how_a_partition_stood_when_its_window_began_stays_restorable() {
    let aged = Arc::new(Aged {
        inner: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
        young_from: AtomicUsize::new(0),
    });
    let cluster = Cluster {
        blobs: aged.clone(),
        ..Cluster::new().await
    };
    let writer = writer(&cluster).await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 1, 10)]).await.unwrap();
    writer.flush().await.unwrap();
    let (head, position) = now(&cluster).await;
    assert!(writer.merge().await.unwrap().is_some());

    // Everything from the merge on is recent; the head before it is how
    // the partition stood when the window began.
    aged.young_from
        .store(position.next().unwrap().0 as usize, Ordering::SeqCst);
    writer.clean(Duration::from_secs(60 * 60)).await.unwrap();
    writer.restore(&head, position).await.unwrap();
    assert_eq!(
        cluster
            .open()
            .await
            .partition()
            .read(|state| ns(state, &[1])),
        [Some(10)]
    );
}
