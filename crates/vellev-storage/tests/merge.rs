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

#[tokio::test]
async fn a_row_deleted_while_its_merge_is_built_stays_dead() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "/log/");
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

    let deleting = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(vec![delete("notes", 3)]).await }
    });
    assert_eq!(seen.recv().await, Some("stored"));
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    tokio::task::yield_now().await;
    paused.release.notify_one();
    deleting.await.unwrap().unwrap();
    merging
        .await
        .unwrap()
        .unwrap()
        .expect("the two segments merge");

    let check = |state: &State| {
        assert_eq!(n_of(state, "notes", 3), None);
        assert_eq!(state.count(&name("notes")), 19);
        assert_eq!(ids(state.scan(&name("notes")).collect()).len(), 19);
    };
    writer.partition().read(check);
    cluster.open().await.partition().read(check);

    writer
        .merge()
        .await
        .unwrap()
        .expect("the merged segment and the flush merge again");
    let shape: Vec<(u64, u64)> = segments_of(&cluster)
        .await
        .iter()
        .map(|segment| (segment.rows, segment.tombstones))
        .collect();
    assert_eq!(
        shape,
        [(19, 0)],
        "the row is left out rather than brought back"
    );
    writer.partition().read(check);
    cluster.open().await.partition().read(check);
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
    assert_eq!(seen.recv().await, Some("stored"));
    let second = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    tokio::task::yield_now().await;
    paused.release.notify_one();
    first
        .await
        .unwrap()
        .unwrap()
        .expect("the first merge goes ahead");
    assert_eq!(second.await.unwrap().unwrap(), None);

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
