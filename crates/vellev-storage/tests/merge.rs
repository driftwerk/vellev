mod common;

use common::*;
use vellev_core::Object;

fn ids(objects: Vec<&Object>) -> Vec<ObjectId> {
    let mut ids: Vec<ObjectId> = objects.into_iter().map(|object| object.id).collect();
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
async fn a_segment_mostly_dead_is_rewritten_alone_and_keeps_the_tombstones_older_rows_need() {
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
    assert_eq!(writer.merge().await.unwrap(), None, "nothing is stale yet");

    writer
        .commit((10..16).map(|n| delete("notes", n)).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .merge()
        .await
        .unwrap()
        .expect("the second segment is mostly dead");
    let segments = segments_of(&cluster).await;
    assert_eq!(segments.len(), 3);
    let rewritten = &segments[1];
    assert!(rewritten.key.ends_with("-merged.segment"));
    assert_eq!((rewritten.rows, rewritten.tombstones), (4, 4));
    let file = segment(&cluster, &rewritten.key).await;
    assert_eq!(
        file.tombstones.len(),
        4,
        "still shadow the first segment's rows"
    );

    cluster.open().await.partition().read(|state| {
        let live = ids(state.scan(&name("notes")).collect());
        let expected: Vec<ObjectId> = (4..10).chain(16..20).map(id).collect();
        assert_eq!(live, expected);
    });
}

#[tokio::test]
async fn files_nothing_reaches_are_deleted_once_old_and_never_before() {
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
    let stray = format!(
        "{}log/00000000000000000099-stale-writer.log",
        Layout(&cluster.key).prefix()
    );
    cluster
        .blobs
        .put(&stray, Bytes::from_static(b"not committed"))
        .await
        .unwrap();
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!((segments.len(), manifests.len()), (5, 5));
    assert!(logs.contains(&stray));
    let before = logs.len() + segments.len() + manifests.len();

    assert_eq!(
        writer.clean(Duration::from_secs(3_600)).await.unwrap(),
        0,
        "all too young"
    );
    let deleted = writer.clean(Duration::ZERO).await.unwrap();
    let (logs, segments, manifests) = files(&cluster).await;
    assert_eq!((segments.len(), manifests.len()), (1, 1));
    assert!(!logs.contains(&stray));
    assert_eq!(
        logs.len(),
        1,
        "the commit since the checkpoint is reached from the head"
    );
    assert_eq!(deleted, before - 3);

    let follower = cluster.open().await;
    follower
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), 5));
    writer.commit(vec![upsert("notes", 10, 1)]).await.unwrap();
    assert_eq!(writer.clean(Duration::ZERO).await.unwrap(), 0);
}
