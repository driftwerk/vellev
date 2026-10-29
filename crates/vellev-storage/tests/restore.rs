//! A partition put back as it stood at an earlier head: a checkpoint past
//! where it is, which writers, followers and cleaning up all take as any
//! other.

mod common;

use common::*;
use vellev_core::RequestId;
use vellev_storage::{Batch, Embedded, Keywords, Merging, Query};

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

    let restored = writer.restore(Some(&head), position).await.unwrap();
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

    writer.restore(Some(&head), position).await.unwrap();
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

    writer.restore(Some(&head), position).await.unwrap();
    writer.restore(Some(&latest), latest_at).await.unwrap();
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

    writer.restore(Some(&head), position).await.unwrap();
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
    writer.restore(Some(&head), position).await.unwrap();

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
    let restored = writer.restore(Some(&head), position).await.unwrap();
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
    writer.restore(Some(&latest), latest_at).await.unwrap();
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
    writer.restore(Some(&head), position).await.unwrap();
    assert_eq!(
        cluster
            .open()
            .await
            .partition()
            .read(|state| ns(state, &[1])),
        [Some(10)]
    );
}

#[tokio::test]
async fn writes_are_refused_and_cleaning_up_waits_while_a_restore_runs() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment");
    let writer = Arc::new(writer(&cluster).await);
    let (head, position) = history(&cluster, &writer).await;

    paused.armed.store(true, Ordering::SeqCst);
    let restoring = tokio::spawn({
        let writer = writer.clone();
        async move { writer.restore(Some(&head), position).await }
    });
    assert_eq!(seen.recv().await, Some("stored"));
    paused.armed.store(false, Ordering::SeqCst);
    assert!(matches!(
        writer.commit(vec![upsert("notes", 4, 4)]).await,
        Err(StorageError::Restoring)
    ));
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!cleaning.is_finished(), "cleaning up waits for the restore");

    paused.release.notify_one();
    restoring.await.unwrap().unwrap();
    cleaning.await.unwrap().unwrap();
    writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
    assert_eq!(
        cluster
            .open()
            .await
            .partition()
            .read(|state| ns(state, &[1, 2, 3, 4])),
        [Some(1), Some(20), Some(3), Some(4)]
    );
}

#[tokio::test]
async fn a_restore_that_fails_before_it_commits_leaves_the_writer_writing() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (_, position) = history(&cluster, &writer).await;
    let nowhere = ManifestRef(format!(
        "{}manifest/missing.manifest",
        Layout(&cluster.key).prefix()
    ));
    assert!(writer.restore(Some(&nowhere), position).await.is_err());
    writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
}

#[tokio::test]
async fn a_restore_whose_commit_fails_closes_the_writer() {
    let mut cluster = Cluster::new().await;
    let observed = cluster.observed();
    cluster.blobs = observed.clone();
    let writer = writer(&cluster).await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let (head, position) = now(&cluster).await;
    writer.commit(vec![upsert("notes", 1, 10)]).await.unwrap();

    observed.fail_puts.store(true, Ordering::SeqCst);
    assert!(writer.restore(Some(&head), position).await.is_err());
    observed.fail_puts.store(false, Ordering::SeqCst);
    assert!(matches!(
        writer.commit(vec![upsert("notes", 2, 2)]).await,
        Err(StorageError::Closed)
    ));
    let reopened = cluster.writer().await;
    assert_eq!(
        reopened.partition().read(|state| ns(state, &[1])),
        [Some(10)]
    );
}

#[tokio::test]
async fn restoring_to_where_a_partition_is_changes_nothing_in_it() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    history(&cluster, &writer).await;
    let (head, position) = now(&cluster).await;
    let before = writer.partition().read(|state| ns(state, &[1, 2, 3]));
    assert_eq!(
        writer.restore(Some(&head), position).await.unwrap(),
        position.next().unwrap()
    );
    assert_eq!(
        writer.partition().read(|state| ns(state, &[1, 2, 3])),
        before
    );
}

#[tokio::test(start_paused = true)]
async fn a_restore_given_up_while_it_waits_for_a_commit_refuses_writes_no_longer() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "/log/");
    let writer = Arc::new(cluster.writer().await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let (head, position) = now(&cluster).await;

    // A commit holds the writer, and a restore waits behind it.
    paused.armed.store(true, Ordering::SeqCst);
    let committing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(vec![upsert("notes", 2, 2)]).await }
    });
    while seen.recv().await != Some("stored") {}
    let restoring = tokio::spawn({
        let writer = writer.clone();
        async move { writer.restore(Some(&head), position).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!restoring.is_finished(), "it waits for the commit");
    restoring.abort();
    assert!(restoring.await.unwrap_err().is_cancelled());

    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_one();
    committing.await.unwrap().unwrap();
    assert!(!writer.is_closed());
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    assert_eq!(
        writer.partition().read(|state| ns(state, &[1, 2, 3])),
        [Some(1), Some(2), Some(3)]
    );
}

/// The notes as every way of asking finds them: how many there are, one
/// by its id, those a filter passes, those nearest a vector and those a
/// word matches.
type Found = (
    usize,
    Option<i64>,
    Vec<ObjectId>,
    Vec<ObjectId>,
    Vec<ObjectId>,
);

fn found(state: &State) -> Found {
    let sorted = |mut ids: Vec<ObjectId>| {
        ids.sort_unstable();
        ids
    };
    let notes = name("notes");
    let second = Filter::Eq {
        property: "n".into(),
        value: json!(2),
    };
    let passing = state.select(&notes, &second).unwrap();
    let nearest = state.search(&notes, &Query::nearest(&[2.0, 0.5], 3));
    let matching = state.keyword(&notes, &Keywords::matching("row", 10));
    (
        state.count(&notes),
        n_of(state, "notes", 2),
        sorted(passing.iter().map(|live| live.id()).collect()),
        sorted(nearest.unwrap().iter().map(|hit| hit.object.id).collect()),
        sorted(
            matching
                .unwrap()
                .iter()
                .map(|scored| scored.object.id)
                .collect(),
        ),
    )
}

/// Three notes flushed into a segment, the head to go back to, and then
/// their collection dropped and that checkpointed.
async fn dropped(cluster: &Cluster, writer: &Writer) -> (ManifestRef, LogPosition) {
    let rows = (1..=3).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let kept = now(cluster).await;
    let drop = Mutation::DropCollection(name("notes"));
    writer.commit(vec![drop]).await.unwrap();
    // A checkpoint without the collection: whoever opens the partition
    // from here on indexes the segment with nothing of its rows.
    writer.flush().await.unwrap();
    kept
}

/// Every note found every way, as a copy opened afresh finds them.
async fn all_found(cluster: &Cluster, state: Found, whose: &str) {
    let all = vec![id(1), id(2), id(3)];
    let whole = (3, Some(2), vec![id(2)], all.clone(), all);
    assert_eq!(state, whole, "{whose}");
    assert_eq!(cluster.open().await.partition().read(found), whole);
}

#[tokio::test]
async fn a_writer_opened_since_a_collection_was_dropped_finds_its_rows_once_put_back() {
    let cluster = Cluster::new().await;
    let (head, position) = dropped(&cluster, &cluster.writer().await).await;

    // Opened with the collection gone, its segment indexed without it.
    let writer = cluster.writer().await;
    writer.restore(Some(&head), position).await.unwrap();
    all_found(&cluster, writer.partition().read(found), "the writer").await;
}

#[tokio::test]
async fn a_copy_opened_since_a_collection_was_dropped_finds_its_rows_once_it_follows_the_restore() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let (head, position) = dropped(&cluster, &writer).await;
    let follower = cluster.open().await;

    writer.restore(Some(&head), position).await.unwrap();
    follower.catch_up(&cluster.committed().await).await.unwrap();
    all_found(&cluster, follower.partition().read(found), "the copy").await;
    all_found(&cluster, writer.partition().read(found), "the writer").await;
}

/// The manifest that is the partition's head.
async fn head_manifest(cluster: &Cluster) -> Manifest {
    let head = cluster.committed().await.manifest.unwrap().0;
    let bytes = cluster.blobs.get(&head).await.unwrap();
    format::decode(&head, Kind::Manifest, &bytes).unwrap()
}

#[tokio::test]
async fn a_partition_put_back_as_one_further_on_than_itself_goes_on_past_that() {
    // A partition with some history, and another of the same name made
    // anew over the same files, as a tenant made again and restored from a
    // backup of what it was.
    let was = Cluster::new().await;
    let writer = was.writer().await;
    history(&was, &writer).await;
    writer.flush().await.unwrap();
    let (head, position) = now(&was).await;
    assert!(position > LogPosition(3));
    let mut anew = Cluster::new().await;
    anew.blobs = was.blobs.clone();
    let writer = anew.writer().await;
    assert_eq!(writer.partition().read(State::position), LogPosition(0));

    let restored = writer.restore(Some(&head), position).await.unwrap();
    assert_eq!(
        restored,
        position.next().unwrap(),
        "past what it went back to"
    );
    let read = |state: &State| ns(state, &[1, 2, 3]);
    let all = [None, Some(200), Some(300)];
    assert_eq!(writer.partition().read(read), all);
    let written = writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
    assert!(written.position > restored);
    writer.flush().await.unwrap();
    writer.clean(Duration::ZERO).await.unwrap();
    assert_eq!(anew.open().await.partition().read(read), all);
}

#[tokio::test]
async fn a_collection_made_after_a_restore_never_takes_the_id_of_one_the_restore_undid() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let (head, position) = now(&cluster).await;
    writer
        .commit(vec![create("tmp"), upsert("tmp", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let id_of = |manifest: &Manifest, collection: &str| {
        let mut collections = manifest.collections.iter();
        let found = collections.find(|(_, schema)| schema.name == name(collection));
        found.map(|(id, _)| *id)
    };
    let undone = id_of(&head_manifest(&cluster).await, "tmp").unwrap();

    // Back to before there was a tmp; the segments that hold its rows are
    // still there to go back to, under its id.
    writer.restore(Some(&head), position).await.unwrap();
    writer
        .commit(vec![create("other"), upsert("other", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let after = head_manifest(&cluster).await;
    assert_eq!(id_of(&after, "tmp"), None);
    assert_ne!(id_of(&after, "other"), Some(undone));
}

/// A state further on than a partition made anew over the same files, as
/// a backup of what a tenant was is to the tenant made again: a restore to
/// it commits past it, however little is committed to the partition, so
/// that no such commit fails merely for not advancing. Returns the
/// partition made anew, one lease on it, and the state to go back to,
/// which has rows in memory that a restore writes a segment of.
async fn further_on() -> (Cluster, Lease, Committed, (ManifestRef, LogPosition)) {
    let was = Cluster::new().await;
    let writer = was.writer().await;
    history(&was, &writer).await;
    writer.commit(vec![upsert("notes", 4, 400)]).await.unwrap();
    let target = now(&was).await;
    let mut anew = Cluster::new().await;
    anew.blobs = was.blobs.clone();
    let (lease, committed) = anew.lease().await;
    (anew, lease, committed, target)
}

/// What is committed, as a copy opened afresh reads it.
async fn committed_notes(cluster: &Cluster) -> Vec<Option<i64>> {
    let read = |state: &State| ns(state, &[1, 2, 3, 4]);
    cluster.open().await.partition().read(read)
}

#[tokio::test]
async fn a_writer_left_behind_on_a_lease_still_current_puts_nothing_back() {
    let (cluster, lease, committed, (head, position)) = further_on().await;
    let open = || {
        let (blobs, coordinator) = (cluster.blobs.clone(), cluster.coordinator.clone());
        Writer::open(blobs, coordinator, lease.clone(), &committed)
    };
    let (behind, ahead) = (open().await.unwrap(), open().await.unwrap());
    ahead
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let before = cluster.committed().await;
    assert!(position > before.log_position);

    // Its commit is under the lease and past all that is committed, and
    // is refused for not following it: by the writer if it asks first, and
    // by the coordinator whether or not.
    let refused = behind.restore(Some(&head), position).await.unwrap_err();
    assert!(
        matches!(
            refused,
            StorageError::Diverged { .. }
                | StorageError::Coordinator(CoordinatorError::Regressed { .. })
        ),
        "{refused}"
    );
    assert_eq!(cluster.committed().await, before);
    ahead.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    ahead.flush().await.unwrap();
    assert_eq!(
        committed_notes(&cluster).await,
        [Some(1), Some(2), None, None]
    );
}

#[tokio::test]
async fn a_restore_overtaken_by_another_writer_of_its_lease_is_refused_by_the_coordinator() {
    let (mut cluster, lease, committed, (head, position)) = further_on().await;
    let store = cluster.blobs.clone();
    let (paused, mut seen) = Paused::new(&mut cluster, "segment/");
    let coordinator = cluster.coordinator.clone();
    let overtaken = Writer::open(
        cluster.blobs.clone(),
        coordinator,
        lease.clone(),
        &committed,
    );
    let overtaken = Arc::new(overtaken.await.unwrap());
    let other = Writer::open(store, cluster.coordinator.clone(), lease, &committed);
    let other = other.await.unwrap();

    // The first is where the partition is as it begins its restore, so
    // asking the coordinator then tells it nothing, and is overtaken while
    // the restore's segment goes up.
    paused.armed.store(true, Ordering::SeqCst);
    let restoring = tokio::spawn({
        let overtaken = overtaken.clone();
        async move { overtaken.restore(Some(&head), position).await }
    });
    while seen.recv().await != Some("stored") {}
    other
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let before = cluster.committed().await;
    assert!(position > before.log_position);
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_one();

    // Its commit is under the lease and past all that is committed: only
    // for not following what is committed is it refused.
    let refused = restoring.await.unwrap().unwrap_err();
    assert!(
        matches!(
            refused,
            StorageError::Coordinator(CoordinatorError::Regressed { .. })
        ),
        "{refused}"
    );
    assert_eq!(cluster.committed().await, before);
    other.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    assert_eq!(
        committed_notes(&cluster).await,
        [Some(1), Some(2), None, None]
    );
}

/// A restore in place moves the writer's epoch on, so that no merge
/// planned before it is committed after it, though the same segments may
/// be back with other rows live in them.
#[tokio::test]
async fn a_restore_in_place_moves_the_epoch_on() {
    let cluster = Cluster::new().await;
    let writer = writer(&cluster).await;
    let (head, position) = history(&cluster, &writer).await;
    let at = writer.epoch();
    writer.restore(Some(&head), position).await.unwrap();
    assert_eq!(writer.epoch(), at + 1);
}
