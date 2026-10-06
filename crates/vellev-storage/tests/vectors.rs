mod common;

use common::*;
use vellev_storage::Query;

async fn segments_of(cluster: &Cluster) -> Vec<SegmentMeta> {
    let head = cluster.committed().await.manifest.unwrap().0;
    let bytes = cluster.blobs.get(&head).await.unwrap();
    let manifest: Manifest = format::decode(&head, Kind::Manifest, &bytes).unwrap();
    manifest.segments
}

fn vector_of(state: &State, object: u8) -> Option<Vec<f32>> {
    state
        .get(&name("notes"), id(object))
        .unwrap()
        .vector
        .map(<[f32]>::to_vec)
}

#[tokio::test]
async fn a_segment_file_carries_no_vectors_and_its_column_does() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();

    let segment = segments_of(&cluster).await.remove(0);
    let vectors = segment.vectors.clone().expect("rows with vectors");
    assert!(vectors.contains("/vectors/") && vectors.ends_with(".vectors"));
    let bytes = cluster.blobs.get(&segment.key).await.unwrap();
    let file: SegmentFile = format::decode(&segment.key, Kind::Segment, &bytes).unwrap();
    assert_eq!(file.rows.len(), 10);
    assert!(file.rows.iter().all(|row| row.object.vector.is_none()));

    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            assert_eq!(vector_of(state, 3), Some(vec![3.0, 0.5]));
            let hits = state
                .search(&name("notes"), &Query::nearest(&[7.0, 0.5], 1))
                .unwrap();
            assert_eq!(hits[0].object.id, id(7));
            assert_eq!(hits[0].vector, &[7.0, 0.5]);
            let listed = state.list(&name("notes"), None, None, 3).unwrap();
            assert!(listed.iter().all(|found| found.vector.is_some()));
        });
    }

    writer.commit(vec![upsert("notes", 3, 30)]).await.unwrap();
    writer
        .partition()
        .read(|state| assert_eq!(vector_of(state, 3), Some(vec![30.0, 0.5])));
    writer.flush().await.unwrap();
    writer.merge().await.unwrap();
    cluster.open().await.partition().read(|state| {
        assert_eq!(vector_of(state, 3), Some(vec![30.0, 0.5]));
        assert_eq!(vector_of(state, 4), Some(vec![4.0, 0.5]));
    });
}

#[tokio::test]
async fn a_segment_without_vectors_has_no_column() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let bare = |object: u8| Mutation::Upsert {
        collection: name("notes"),
        id: id(object),
        properties: json!({ "title": "no vector", "n": 1 })
            .as_object()
            .unwrap()
            .clone(),
        vector: None,
    };
    writer
        .commit(vec![create("notes"), bare(1), bare(2)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let segment = segments_of(&cluster).await.remove(0);
    assert_eq!(segment.vectors, None);
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(vector_of(state, 1), None));
}

#[tokio::test]
async fn a_node_with_a_local_copy_reads_vectors_in_place_and_survives_a_damaged_one() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let vectors = segments_of(&cluster).await.remove(0).vectors.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let store = cluster.observed();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store.clone(), dir.path()));
    let follower = cluster.open_on(cached.clone()).await.unwrap();
    let copy = dir.path().join(&vectors);
    assert!(copy.is_file(), "the column was copied to local disk");
    assert!(store.fetched().contains(&vectors));
    follower
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));

    std::fs::remove_file(&copy).unwrap();
    follower
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));

    std::fs::write(&copy, b"not a vectors file").unwrap();
    let again = cluster.open_on(cached.clone()).await.unwrap();
    assert!(store.fetched().contains(&vectors), "fetched again");
    again
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));
    assert!(format::verify(&vectors, &std::fs::read(&copy).unwrap()).is_ok());
}
