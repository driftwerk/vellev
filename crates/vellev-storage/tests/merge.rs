mod common;

use common::*;
use vellev_storage::Live;
use vellev_storage::Merging;

fn ids(found: Vec<Live<'_>>) -> Vec<ObjectId> {
    let mut ids: Vec<ObjectId> = found.into_iter().map(|found| found.id()).collect();
    ids.sort_unstable();
    ids
}

/// Every file of the partition, by kind.
async fn files(cluster: &Cluster) -> (Vec<String>, Vec<String>, Vec<String>) {
    let listed = cluster
        .blobs
        .list(&Layout(&cluster.key).prefix())
        .await
        .unwrap();
    let of = |kind: &str| -> Vec<String> {
        let mut keys: Vec<String> = listed
            .iter()
            .map(|listed| listed.key.clone())
            .filter(|key| key.ends_with(&format!(".{kind}")))
            .collect();
        keys.sort();
        keys
    };
    (of("log"), of("segment"), of("manifest"))
}

async fn segment(cluster: &Cluster, key: &str) -> SegmentFile {
    let bytes = cluster.blobs.get(key).await.unwrap();
    format::decode(key, Kind::Segment, &bytes).unwrap()
}

/// What a fresh copy of the partition lists as its segments.
async fn segments_of(cluster: &Cluster) -> Vec<SegmentMeta> {
    let committed = cluster.committed().await;
    let head = committed.manifest.unwrap().0;
    let bytes = cluster.blobs.get(&head).await.unwrap();
    let manifest: Manifest = format::decode(&head, Kind::Manifest, &bytes).unwrap();
    manifest.segments
}

#[tokio::test]
async fn small_segments_are_merged_into_one_and_reads_stay_the_same() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    writer.flush().await.unwrap();
    for round in 0..4u8 {
        let writes = (0..10).map(|n| upsert("notes", round * 10 + n, i64::from(round)));
        writer.commit(writes.collect()).await.unwrap();
        writer
            .commit(vec![delete("notes", round * 10)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    assert_eq!(segments_of(&cluster).await.len(), 4);
    let before = writer
        .partition()
        .read(|state| ids(state.scan(&name("notes")).collect()));
    assert_eq!(before.len(), 36);

    let position = writer
        .merge()
        .await
        .unwrap()
        .expect("four small segments to merge");
    assert_eq!(writer.partition().read(State::position), position);
    let merged = segments_of(&cluster).await;
    assert_eq!(merged.len(), 1);
    assert!(merged[0].key.ends_with("-merged.segment"));
    assert_eq!(
        (merged[0].rows, merged[0].tombstones),
        (36, 0),
        "dead rows and tombstones gone"
    );
    assert_eq!(writer.merge().await.unwrap(), None, "nothing left to merge");

    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            assert_eq!(ids(state.scan(&name("notes")).collect()), before);
            assert_eq!(n_of(state, "notes", 11), Some(1));
            assert_eq!(n_of(state, "notes", 10), None);
        });
    }

    writer
        .commit(vec![
            upsert("notes", 11, 99),
            delete("notes", 12),
            upsert("notes", 50, 5),
        ])
        .await
        .unwrap();
    let after = writer.partition().read(|state| {
        (
            n_of(state, "notes", 11),
            n_of(state, "notes", 12),
            n_of(state, "notes", 50),
            state.count(&name("notes")),
        )
    });
    assert_eq!(after, (Some(99), None, Some(5), 36));
    let hits = writer.partition().read(|state| {
        let query = vellev_storage::Query::nearest(&[99.0, 0.5], 1);
        state.search(&name("notes"), &query).unwrap()[0].object.id
    });
    assert_eq!(hits, id(11));
    writer.flush().await.unwrap();
    cluster.open().await.partition().read(|state| {
        assert_eq!(n_of(state, "notes", 11), Some(99));
        assert_eq!(state.count(&name("notes")), 36);
    });
}

#[tokio::test]
async fn a_segment_half_dead_is_rewritten_alone_keeping_only_the_tombstones_older_rows_need() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let first = (0..10).map(|n| upsert("notes", n, 0));
    writer
        .commit([create("notes")].into_iter().chain(first).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let second = (10..20)
        .map(|n| upsert("notes", n, 1))
        .chain((0..4).map(|n| delete("notes", n)));
    writer.commit(second.collect()).await.unwrap();
    writer.flush().await.unwrap();
    assert_eq!(
        writer.merge().await.unwrap(),
        None,
        "four of ten is not half"
    );

    writer.commit(vec![delete("notes", 4)]).await.unwrap();
    writer.flush().await.unwrap();
    writer
        .merge()
        .await
        .unwrap()
        .expect("half the first segment is dead");
    let segments = segments_of(&cluster).await;
    let shape: Vec<(u64, u64)> = segments
        .iter()
        .map(|segment| (segment.rows, segment.tombstones))
        .collect();
    assert_eq!(
        shape,
        [(5, 0), (10, 4), (0, 1)],
        "nothing older for tombstones to act on"
    );
    assert!(segments[0].key.ends_with("-merged.segment"));

    writer
        .commit((10..15).map(|n| delete("notes", n)).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .merge()
        .await
        .unwrap()
        .expect("half the second segment is dead");
    let segments = segments_of(&cluster).await;
    assert_eq!((segments[1].rows, segments[1].tombstones), (5, 4));
    let file = segment(&cluster, &segments[1].key).await;
    assert_eq!(
        file.tombstones.len(),
        4,
        "still shadow the first segment's rows"
    );

    let expected: Vec<ObjectId> = (5..10).chain(15..20).map(id).collect();
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| assert_eq!(ids(state.scan(&name("notes")).collect()), expected));
    }
}

#[tokio::test]
async fn a_merge_that_leaves_later_segments_keeps_every_row_where_it_is_found() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        fanout: 2,
        rows: 100,
        ..Merging::default()
    });
    writer.commit(vec![create("notes")]).await.unwrap();
    for (round, count) in [10u8, 10, 60, 10].into_iter().enumerate() {
        let first = (round as u8) * 60;
        let writes = (0..count).map(|n| upsert("notes", first + n, i64::from(round as u8)));
        writer.commit(writes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    writer
        .merge()
        .await
        .unwrap()
        .expect("the two small segments in front merge");
    let shape: Vec<u64> = segments_of(&cluster)
        .await
        .iter()
        .map(|segment| segment.rows)
        .collect();
    assert_eq!(
        shape,
        [20, 60, 10],
        "the big segment and the one after stay"
    );
    assert_eq!(writer.merge().await.unwrap(), None);

    writer
        .commit(vec![
            upsert("notes", 180, 8),
            delete("notes", 181),
            upsert("notes", 120, 7),
        ])
        .await
        .unwrap();
    let check = |state: &State| {
        assert_eq!(state.count(&name("notes")), 89);
        assert_eq!(n_of(state, "notes", 180), Some(8));
        assert_eq!(n_of(state, "notes", 181), None);
        assert_eq!(n_of(state, "notes", 182), Some(3));
        assert_eq!(n_of(state, "notes", 120), Some(7));
        assert_eq!(n_of(state, "notes", 121), Some(2));
        assert_eq!(n_of(state, "notes", 61), Some(1));
        assert_eq!(ids(state.scan(&name("notes")).collect()).len(), 89);
    };
    writer.partition().read(check);
    writer.flush().await.unwrap();
    cluster.open().await.partition().read(check);

    writer
        .commit((120..150).map(|n| delete("notes", n)).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .merge()
        .await
        .unwrap()
        .expect("half the big segment is dead");
    let shape: Vec<(u64, u64)> = segments_of(&cluster)
        .await
        .iter()
        .map(|segment| (segment.rows, segment.tombstones))
        .collect();
    assert_eq!(shape[1], (30, 0));
    let check = |state: &State| {
        assert_eq!(state.count(&name("notes")), 59);
        assert_eq!(n_of(state, "notes", 180), Some(8));
        assert_eq!(n_of(state, "notes", 150), Some(2));
        assert_eq!(n_of(state, "notes", 149), None);
    };
    writer.partition().read(check);
    cluster.open().await.partition().read(check);
}

#[tokio::test]
async fn files_nothing_reaches_are_deleted_but_never_those_ahead_of_the_writer() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..4u8 {
        writer
            .commit(vec![upsert("notes", round, 1)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    writer.merge().await.unwrap().unwrap();
    writer.commit(vec![upsert("notes", 9, 9)]).await.unwrap();
    let prefix = Layout(&cluster.key).prefix();
    let stale = format!("{prefix}log/00000000000000000002-00000000000000000000-gone.log");
    let ahead = format!("{prefix}log/00000000000000000099-00000000000000000009-next.log");
    for stray in [&stale, &ahead] {
        cluster
            .blobs
            .put(stray, Bytes::from_static(b"not committed"))
            .await
            .unwrap();
    }
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!((segments.len(), manifests.len()), (5, 5));
    let before = logs.len() + segments.len() + manifests.len();

    assert_eq!(
        writer.clean(Duration::from_secs(3_600)).await.unwrap(),
        0,
        "every head is young"
    );
    let deleted = writer.clean(Duration::ZERO).await.unwrap();
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!((segments.len(), manifests.len()), (1, 1));
    assert!(!logs.contains(&stale));
    assert_eq!(
        logs.iter().filter(|log| **log != ahead).count(),
        1,
        "the commit since the checkpoint is reached from the head"
    );
    assert!(
        logs.contains(&ahead),
        "a newer writer's upload is not touched"
    );
    assert_eq!(
        deleted,
        before - 4 + 4,
        "four files stay: the merged segment, the checkpoint, the commit since and the stray ahead; four vectors files go with their segments"
    );

    let follower = cluster.open().await;
    follower
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 5));
    writer.commit(vec![upsert("notes", 10, 1)]).await.unwrap();
    assert_eq!(writer.clean(Duration::ZERO).await.unwrap(), 0);
}

#[tokio::test]
async fn how_old_a_file_is_is_told_by_the_clock_the_writer_is_given() {
    let cluster = Cluster::new().await;
    // Two hours behind the machine's clock, by which the store says when
    // each file was written.
    let hour = Duration::from_secs(3_600);
    let clock = Arc::new(ManualClock::starting_at(Utc::now() - 2 * hour));
    let writer = cluster.writer().await.timed_by(clock.clone());
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..4u8 {
        writer
            .commit(vec![upsert("notes", round, 1)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    writer.merge().await.unwrap().unwrap();
    let (_, segments, manifests) = files(&cluster).await;
    assert_eq!((segments.len(), manifests.len()), (5, 5));

    assert_eq!(
        writer.clean(Duration::ZERO).await.unwrap(),
        0,
        "by the writer's clock no file is written yet, though by the machine's each is old"
    );
    // Three hours ahead of the machine's.
    clock.advance(5 * hour);
    assert_eq!(
        writer.clean(4 * hour).await.unwrap(),
        0,
        "three hours old by the writer's clock, and kept four"
    );
    let deleted = writer.clean(2 * hour).await.unwrap();
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!(
        (logs.len(), segments.len(), manifests.len()),
        (0, 1, 1),
        "kept two hours they are old, though by the machine's clock each was written just now"
    );
    assert_eq!(
        deleted,
        5 + 4 + 4 + 4,
        "five logs, four segments and their vectors, four checkpoints"
    );
    let follower = cluster.open().await;
    follower
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 4));
}

/// Blobs that list the files named in `old` as written long ago.
#[derive(Debug)]
struct Aged {
    inner: Arc<dyn Blobs>,
    old: Mutex<BTreeSet<String>>,
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
        let listed = self.inner.list(prefix).await?;
        let old = self.old.lock().unwrap();
        Ok(listed
            .into_iter()
            .map(|listed| match old.contains(&listed.key) {
                true => Listed {
                    modified: Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap(),
                    ..listed
                },
                false => listed,
            })
            .collect())
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn files_a_recent_head_still_reaches_stay_until_that_head_is_old() {
    let mut cluster = Cluster::new().await;
    let aged = Arc::new(Aged {
        inner: cluster.blobs.clone(),
        old: Mutex::default(),
    });
    cluster.blobs = aged.clone();
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..4u8 {
        writer
            .commit(vec![upsert("notes", round, 1)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    let age = || async {
        let listed = aged
            .inner
            .list(&Layout(&cluster.key).prefix())
            .await
            .unwrap();
        aged.old
            .lock()
            .unwrap()
            .extend(listed.into_iter().map(|listed| listed.key));
    };
    age().await;
    writer.commit(vec![upsert("notes", 9, 9)]).await.unwrap();
    writer.merge().await.unwrap().unwrap();
    let position = writer.partition().read(State::position);
    let bogus = Manifest {
        position: LogPosition(position.0 - 1),
        prev: None,
        next_collection_id: 1,
        collections: Vec::new(),
        segments: Vec::new(),
        replay: Vec::new(),
        replay_after: LogPosition(0),
        answered: Vec::new(),
    };
    let stale = format!(
        "{}manifest/{:020}-00000000000000000000-lost.manifest",
        Layout(&cluster.key).prefix(),
        position.0 - 1
    );
    let frame = format::encode(Kind::Manifest, &bogus).unwrap();
    cluster.blobs.put(&stale, frame).await.unwrap();
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!(
        (logs.len(), segments.len(), manifests.len()),
        (6, 6, 6),
        "the merge also flushed the write before it"
    );

    let grace = Duration::from_secs(3_600);
    assert_eq!(writer.clean(grace).await.unwrap(), 5 + 3);
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!(
        (logs.len(), segments.len(), manifests.len()),
        (1, 6, 3),
        "the write just before the merge is recent, so what it saw stays"
    );
    let before = cluster.committed().await;
    let was = Committed {
        log_position: LogPosition(before.log_position.0 - 1),
        manifest: Some(ManifestRef(logs[0].clone())),
        ..before
    };
    let follower = Follower::open(cluster.key.clone(), cluster.blobs.clone(), &was)
        .await
        .unwrap();
    follower
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 5));

    age().await;
    assert_eq!(
        writer.clean(grace).await.unwrap(),
        1 + 4 + 4 + 2,
        "the log, the segments and their vectors, the checkpoint and a stale writer's manifest"
    );
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!((logs.len(), segments.len(), manifests.len()), (0, 2, 1));
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 5));
}

#[tokio::test]
async fn a_commit_in_flight_while_cleaning_is_not_mistaken_for_an_orphan() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "/log/");
    let writer = Arc::new(cluster.writer().await);
    writer.commit(vec![create("notes")]).await.unwrap();
    paused.armed.store(true, Ordering::SeqCst);

    let committing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(vec![upsert("notes", 1, 1)]).await }
    });
    assert_eq!(seen.recv().await, Some("stored"));
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    assert_eq!(seen.recv().await, Some("listed"));
    paused.release.notify_one();
    committing.await.unwrap().unwrap();
    assert_eq!(cleaning.await.unwrap().unwrap(), 0);

    let (logs, _, _) = files(&cluster).await;
    assert_eq!(logs.len(), 2);
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(n_of(state, "notes", 1), Some(1)));
}

/// Two segments of ten notes each to merge, with the uploads of logs and
/// of merged segments each held when armed.
async fn two_to_merge() -> (Cluster, Arc<Writer>, [Held; 2]) {
    let mut cluster = Cluster::new().await;
    let logs = Paused::new(&mut cluster, "/log/");
    let merged = Paused::new(&mut cluster, "-merged.segment");
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = Arc::new(cluster.writer().await.merging_with(merging));
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..2u8 {
        let writes = (0..10).map(|n| upsert("notes", round * 10 + n, 1));
        writer.commit(writes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    (cluster, writer, [logs, merged])
}

/// Uploads of one kind that can be held, and word of each that is.
type Held = (
    Arc<Paused>,
    tokio::sync::mpsc::UnboundedReceiver<&'static str>,
);

/// Wait for an upload to be held, skipping the listings reported.
async fn held(seen: &mut tokio::sync::mpsc::UnboundedReceiver<&'static str>) {
    let stored =
        async { while seen.recv().await.expect("the store is still there") != "stored" {} };
    soon(stored).await;
}

/// How many rows and tombstones each segment committed has.
async fn shape(cluster: &Cluster) -> Vec<(u64, u64)> {
    let segments = segments_of(cluster).await;
    let shape = |segment: &SegmentMeta| (segment.rows, segment.tombstones);
    segments.iter().map(shape).collect()
}

/// Note 3 is dead to the writer and to a copy opened afresh.
async fn third_is_dead(cluster: &Cluster, writer: &Writer) {
    let check = |state: &State| {
        assert_eq!(n_of(state, "notes", 3), None);
        assert_eq!(state.count(&name("notes")), 19);
        assert_eq!(ids(state.scan(&name("notes")).collect()).len(), 19);
    };
    writer.partition().read(check);
    cluster.open().await.partition().read(check);
}

#[tokio::test]
async fn a_row_deleted_once_its_merged_segment_is_built_is_dead_in_it_and_flushed_with_it() {
    let (cluster, writer, [_logs, (merged, mut seen)]) = two_to_merge().await;
    // The merged segment, with the row in it, is built and stored, and
    // what is unflushed not yet looked at, when the row is deleted.
    merged.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    held(&mut seen).await;
    writer.commit(vec![delete("notes", 3)]).await.unwrap();
    merged.armed.store(false, Ordering::SeqCst);
    merged.release.notify_one();
    let merged = soon(merging).await.unwrap().unwrap();
    merged.expect("the two segments merge");

    // The delete is flushed with the merge: a segment of its tombstone.
    assert_eq!(shape(&cluster).await, [(20, 0), (0, 1)]);
    third_is_dead(&cluster, &writer).await;
    let again = writer.merge().await.unwrap();
    again.expect("the merged segment and the flush merge again");
    assert_eq!(
        shape(&cluster).await,
        [(19, 0)],
        "left out, not brought back"
    );
    third_is_dead(&cluster, &writer).await;
}

#[tokio::test]
async fn a_row_deleted_once_its_merge_has_looked_at_what_is_unflushed_is_dead_by_replay() {
    let (cluster, writer, [(logs, mut logged), (merged, mut seen)]) = two_to_merge().await;
    // The delete holds the writer, its log stored and not yet committed.
    logs.armed.store(true, Ordering::SeqCst);
    let deleting = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(vec![delete("notes", 3)]).await }
    });
    held(&mut logged).await;
    logs.armed.store(false, Ordering::SeqCst);
    // The merge builds its segment and looks at what is unflushed, which
    // is nothing yet, and waits for the writer: nothing it waits on from
    // here to there is anything but ready.
    merged.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    held(&mut seen).await;
    merged.armed.store(false, Ordering::SeqCst);
    merged.release.notify_one();
    for _ in 0..100 {
        tokio::task::yield_now().await;
    }
    assert!(!merging.is_finished(), "it waits for the writer");
    logs.release.notify_one();
    soon(deleting).await.unwrap().unwrap();
    let merged = soon(merging).await.unwrap().unwrap();
    merged.expect("the two segments merge");

    // The delete was committed while the merge was built: the merge's
    // manifest replays it, and the row is dead with nothing flushed.
    assert_eq!(shape(&cluster).await, [(20, 0)]);
    third_is_dead(&cluster, &writer).await;
    let nothing = writer.merge().await.unwrap();
    assert_eq!(nothing, None, "one segment, a twentieth of it dead");
    writer.flush().await.unwrap();
    assert_eq!(shape(&cluster).await, [(20, 0), (0, 1)]);
    let again = writer.merge().await.unwrap();
    again.expect("the merged segment and the flush merge");
    assert_eq!(
        shape(&cluster).await,
        [(19, 0)],
        "left out, not brought back"
    );
    third_is_dead(&cluster, &writer).await;
}

/// What `work` came to, failing the test rather than letting it hang if
/// that took a minute.
async fn soon<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(60), work)
        .await
        .expect("done within a minute")
}

#[tokio::test]
async fn a_merge_planned_over_segments_another_replaced_meanwhile_does_nothing() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "/segment/");
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = Arc::new(cluster.writer().await.merging_with(merging));
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..2u8 {
        let writes = (0..10).map(|n| upsert("notes", round * 10 + n, 1));
        writer.commit(writes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    paused.armed.store(true, Ordering::SeqCst);

    let first = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    assert_eq!(soon(seen.recv()).await, Some("stored"));
    let second = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    tokio::task::yield_now().await;
    paused.release.notify_one();
    soon(first)
        .await
        .unwrap()
        .unwrap()
        .expect("the first merge goes ahead");
    assert_eq!(soon(second).await.unwrap().unwrap(), None);

    let shape: Vec<u64> = segments_of(&cluster)
        .await
        .iter()
        .map(|segment| segment.rows)
        .collect();
    assert_eq!(shape, [20]);
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| assert_eq!(state.count(&name("notes")), 20));
    }
}

#[tokio::test]
async fn a_run_stops_where_the_rows_a_merge_may_hold_run_out() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        fanout: 2,
        rows: 100,
        ..Merging::default()
    });
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..3u8 {
        let writes = (0..40).map(|n| upsert("notes", round * 40 + n, 1));
        writer.commit(writes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    writer.merge().await.unwrap().expect("two segments fit");
    let shape = |segments: Vec<SegmentMeta>| -> Vec<u64> {
        segments.iter().map(|segment| segment.rows).collect()
    };
    assert_eq!(shape(segments_of(&cluster).await), [80, 40]);
    assert_eq!(writer.merge().await.unwrap(), None, "a third would not fit");
    assert_eq!(shape(segments_of(&cluster).await), [80, 40]);
}

#[tokio::test]
async fn flushes_a_little_over_a_fanouts_share_are_merged_as_far_as_they_fit() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        fanout: 4,
        rows: 100,
        ..Merging::default()
    });
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..8u8 {
        let writes = (0..27).map(|n| upsert("notes", round * 27 + n, 1));
        writer.commit(writes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    while writer.merge().await.unwrap().is_some() {}
    let shape: Vec<u64> = segments_of(&cluster)
        .await
        .iter()
        .map(|segment| segment.rows)
        .collect();
    assert_eq!(
        shape,
        [81, 81, 54],
        "three of 27 fill a merge, and the last two together reach half of one"
    );
    writer
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 8 * 27));
}

#[tokio::test]
async fn a_segment_with_nothing_dead_is_never_stale_however_low_the_bar() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        stale: 0.0,
        ..Merging::default()
    });
    let rows = (0..10).map(|n| upsert("notes", n, 1));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    assert_eq!(writer.merge().await.unwrap(), None);
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    writer.flush().await.unwrap();
    writer
        .merge()
        .await
        .unwrap()
        .expect("one dead row is enough");
    assert_eq!(writer.merge().await.unwrap(), None);
}

#[tokio::test]
async fn a_writer_opened_from_a_checkpoint_carries_the_chain_on() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..2u8 {
        writer
            .commit(vec![upsert("notes", round, 1)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    let writer = cluster.writer().await;
    writer.commit(vec![upsert("notes", 2, 1)]).await.unwrap();
    writer.flush().await.unwrap();
    let head = cluster.committed().await.manifest.unwrap().0;
    let bytes = cluster.blobs.get(&head).await.unwrap();
    let manifest: Manifest = format::decode(&head, Kind::Manifest, &bytes).unwrap();
    let (_, _, manifests) = files(&cluster).await;
    assert_eq!(manifest.prev.as_deref(), Some(manifests[1].as_str()));
    assert_eq!(writer.clean(Duration::ZERO).await.unwrap(), 4 + 2);
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!((logs.len(), segments.len(), manifests.len()), (0, 3, 1));
}

/// Blobs over a cache that say the copy on local disk of each vectors file
/// that `picks` picks is where that of `instead` is, once there is one.
#[derive(Debug)]
struct Miscopied {
    inner: CachedBlobs,
    picks: fn(&str) -> bool,
    instead: Mutex<Option<String>>,
}

#[async_trait]
impl Blobs for Miscopied {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        let instead = self.instead.lock().unwrap().clone();
        match instead.filter(|_| key.ends_with(".vectors") && (self.picks)(key)) {
            Some(other) => self.inner.local(&other).await,
            None => self.inner.local(key).await,
        }
    }
}

/// A writer over such blobs, with three notes flushed into a segment, the
/// key of whose vectors is returned, and two more written since.
async fn miscopying(
    dir: &std::path::Path,
    picks: fn(&str) -> bool,
) -> (Cluster, Arc<Miscopied>, Writer, String) {
    let mut cluster = Cluster::new().await;
    let miscopied = Arc::new(Miscopied {
        inner: CachedBlobs::new(cluster.blobs.clone(), dir),
        picks,
        instead: Mutex::default(),
    });
    cluster.blobs = miscopied.clone();
    let writer = cluster.writer().await.merging_with(Merging {
        fanout: 2,
        ..Merging::default()
    });
    let rows = (1..=3).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let first = segments_of(&cluster).await.remove(0).vectors.unwrap();
    writer
        .commit(vec![upsert("notes", 4, 4), upsert("notes", 5, 5)])
        .await
        .unwrap();
    (cluster, miscopied, writer, first)
}

fn notes(state: &State) -> Vec<Option<i64>> {
    (1..=6).map(|n| n_of(state, "notes", n)).collect()
}

#[tokio::test]
async fn a_flush_whose_segment_is_not_fit_fails_before_it_commits_anything() {
    let dir = tempfile::tempdir().unwrap();
    let (cluster, miscopied, writer, first) = miscopying(dir.path(), |_| true).await;
    let before = cluster.committed().await;

    // The two rows' vectors are read back as the first segment's three.
    *miscopied.instead.lock().unwrap() = Some(first);
    let unfit = writer.flush().await.unwrap_err();
    assert!(matches!(unfit, StorageError::Corrupt { .. }), "{unfit}");
    assert_eq!(cluster.committed().await, before, "nothing was committed");
    assert!(!writer.is_closed());

    *miscopied.instead.lock().unwrap() = None;
    writer.commit(vec![upsert("notes", 6, 6)]).await.unwrap();
    writer.flush().await.unwrap();
    let all = [1, 2, 3, 4, 5, 6].map(Some);
    assert_eq!(writer.partition().read(notes), all);
    assert_eq!(cluster.open().await.partition().read(notes), all);
}

#[tokio::test]
async fn a_merge_one_of_whose_segments_is_not_fit_changes_nothing_in_memory_or_out() {
    let dir = tempfile::tempdir().unwrap();
    let flushed: fn(&str) -> bool = |key| !key.contains("-merged");
    let (cluster, miscopied, writer, first) = miscopying(dir.path(), flushed).await;
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 6, 6)]).await.unwrap();
    let before = cluster.committed().await;

    // The merged segment is fit; the one of the row flushed with it has
    // its vectors read back as the first segment's three.
    *miscopied.instead.lock().unwrap() = Some(first);
    let unfit = writer.merge().await.unwrap_err();
    assert!(matches!(unfit, StorageError::Corrupt { .. }), "{unfit}");
    assert_eq!(cluster.committed().await, before, "nothing was committed");
    assert!(!writer.is_closed());
    let all = [1, 2, 3, 4, 5, 6].map(Some);
    assert_eq!(writer.partition().read(notes), all);
    assert_eq!(writer.partition().read(|state| state.segments().count()), 2);

    *miscopied.instead.lock().unwrap() = None;
    writer
        .merge()
        .await
        .unwrap()
        .expect("two segments to merge");
    writer.commit(vec![upsert("notes", 1, 10)]).await.unwrap();
    let all = [10, 2, 3, 4, 5, 6].map(Some);
    assert_eq!(writer.partition().read(notes), all);
    assert_eq!(cluster.open().await.partition().read(notes), all);
}

#[tokio::test]
async fn a_writer_whose_lease_was_taken_since_cleans_nothing_up() {
    let cluster = Cluster::new().await;
    let behind = cluster.writer().await.merging_with(Merging {
        fanout: 2,
        ..Merging::default()
    });
    behind.commit(vec![create("notes")]).await.unwrap();
    for round in 0..2u8 {
        behind
            .commit(vec![upsert("notes", round, 1)])
            .await
            .unwrap();
        behind.flush().await.unwrap();
    }
    behind
        .merge()
        .await
        .unwrap()
        .expect("two segments to merge");
    behind.flush().await.unwrap();
    let before = files(&cluster).await;

    // Another has the partition now, and may have put back into use what
    // this one would take for litter: it is not this one's to clean.
    let _current = cluster.writer().await;
    let refused = behind.clean(Duration::ZERO).await.unwrap_err();
    assert!(
        matches!(refused, StorageError::Diverged { .. }),
        "{refused}"
    );
    assert_eq!(files(&cluster).await, before);
}

/// A writer whose merges fold two segments, with two segments of notes
/// flushed: a merge to be had.
async fn two_segments(cluster: &Cluster) -> Writer {
    let writer = cluster.writer().await.merging_with(Merging {
        fanout: 2,
        ..Merging::default()
    });
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    writer
}

/// A merge's commit moves the writer's epoch on; a flush, a commit of
/// curves and writes do not, as none of them can leave a merge planned
/// before it describing rows that are not there.
#[tokio::test]
async fn a_merge_moves_the_epoch_on_and_flushes_curves_and_writes_do_not() {
    let cluster = Cluster::new().await;
    let writer = two_segments(&cluster).await;
    let at = writer.epoch();
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    writer.flush().await.unwrap();
    let key = segments_of(&cluster).await[0].key.clone();
    assert!(
        writer
            .commit_curves(vec![(key, curve_for(0))])
            .await
            .is_err(),
        "a curve for no index is refused, and moves nothing"
    );
    assert_eq!(writer.epoch(), at, "writes, flushes and curves");
    assert!(writer.merge().await.unwrap().is_some());
    assert_eq!(writer.epoch(), at + 1, "a merge");
}

/// A curve for a collection with no clustered index in the segment, for
/// the commit above to refuse.
fn curve_for(collection: u64) -> vellev_storage::format::SegmentCurve {
    use vellev_storage::curve::{Curve, Source};
    vellev_storage::format::SegmentCurve {
        collection: vellev_storage::format::CollectionId(collection),
        band: 0,
        curve: Curve::of(Source::Rows, 1, 10, vec![3]).unwrap(),
    }
}

/// A merge is committed only at the epoch it was planned at: one planned
/// at another commits nothing, and leaves the partition as it was for a
/// merge planned now.
#[tokio::test]
async fn a_merge_planned_at_another_epoch_commits_nothing() {
    let cluster = Cluster::new().await;
    let writer = two_segments(&cluster).await;
    let head = cluster.committed().await;
    let segments = segments_of(&cluster).await;
    let stale = writer.epoch() + 1;
    assert_eq!(writer.merge_planned_at(Some(stale)).await.unwrap(), None);
    assert_eq!(cluster.committed().await.manifest, head.manifest);
    assert_eq!(segments_of(&cluster).await, segments);
    assert_eq!(writer.epoch(), stale - 1);
    assert!(
        writer.merge().await.unwrap().is_some(),
        "planned now, it is"
    );
    let ids = writer
        .partition()
        .read(|state| ids(state.list(&name("notes"), None, None, 10).unwrap()));
    assert_eq!(
        ids,
        [
            ObjectId(uuid::Uuid::from_u128(1)),
            ObjectId(uuid::Uuid::from_u128(2))
        ]
    );
}

/// A merge's build runs with no lock a flush needs: a flush made while a
/// merge waits between its build and its commit goes through at once, and
/// the merge then commits too, every row where it should be.
#[tokio::test]
async fn a_flush_is_never_held_for_a_merge_builds_length() {
    let cluster = Cluster::new().await;
    let writer = Arc::new(two_segments(&cluster).await);
    let gate = writer.hold_merges_before_commit();
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    gate.reached.notified().await;
    assert!(
        writer.kept_while_merging() > 0,
        "its uploads are kept meanwhile"
    );
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    tokio::time::timeout(Duration::from_secs(10), writer.flush())
        .await
        .expect("the flush is not held for the merge")
        .unwrap();
    gate.release.notify_one();
    assert!(merging.await.unwrap().unwrap().is_some());
    assert_eq!(writer.kept_while_merging(), 0);
    let follower = cluster.open().await;
    for note in 1..=3u8 {
        assert_eq!(
            n_of(&follower.partition().read(|s| s.clone()), "notes", note),
            Some(i64::from(note))
        );
    }
}

/// A restore between a merge's plan and its commit drops the merge, though
/// it brings back the very segments the merge was planned over: a row dead
/// when it was planned, and so not in what it built, is live again, and is
/// found.
#[tokio::test]
async fn a_restore_between_a_merges_plan_and_its_commit_drops_it_even_with_the_same_keys() {
    let cluster = Cluster::new().await;
    let writer = Arc::new(two_segments(&cluster).await);
    let committed = cluster.committed().await;
    let (head, position) = (committed.manifest.unwrap(), committed.log_position);
    writer.commit(vec![delete("notes", 1)]).await.unwrap();
    let gate = writer.hold_merges_before_commit();
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    gate.reached.notified().await;
    writer.restore(Some(&head), position).await.unwrap();
    let restored = cluster.committed().await;
    gate.release.notify_one();
    assert_eq!(
        merging.await.unwrap().unwrap(),
        None,
        "the merge is dropped"
    );
    assert_eq!(writer.kept_while_merging(), 0, "and keeps nothing");
    assert_eq!(cluster.committed().await.manifest, restored.manifest);
    let both = |s: &State| (n_of(s, "notes", 1), n_of(s, "notes", 2));
    assert_eq!(writer.partition().read(both), (Some(1), Some(2)));
    let follower = cluster.open().await;
    assert_eq!(follower.partition().read(both), (Some(1), Some(2)));
}

/// Cleaning up while a merge waits between its build and its commit
/// deletes nothing the merge has uploaded, however short the grace: the
/// merge then commits, and a fresh copy reads it.
#[tokio::test]
async fn cleaning_up_during_a_merge_deletes_nothing_it_has_uploaded() {
    let cluster = Cluster::new().await;
    let writer = Arc::new(two_segments(&cluster).await);
    let gate = writer.hold_merges_before_commit();
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    gate.reached.notified().await;
    // Flushes moving the base past the merge's position, so that its files
    // lie below what cleaning up may reach.
    for note in 3..6u8 {
        writer
            .commit(vec![upsert("notes", note, i64::from(note))])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    writer.clean(Duration::ZERO).await.unwrap();
    gate.release.notify_one();
    assert!(merging.await.unwrap().unwrap().is_some());
    let follower = cluster.open().await;
    let read = follower
        .partition()
        .read(|s| (1..6u8).map(|n| n_of(s, "notes", n)).collect::<Vec<_>>());
    assert_eq!(read, (1..6).map(Some).collect::<Vec<_>>());
}
