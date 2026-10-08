mod common;

use common::*;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::vector::VectorIndex;
use vellev_storage::{Merging, Query};

const DIMENSIONS: usize = 8;
const ROWS: u32 = 4_500;

fn wide(collection: &str) -> Mutation {
    let mut schema = schema(collection);
    schema.embedding.dimensions = DIMENSIONS as u32;
    Mutation::CreateCollection(Box::new(schema))
}

fn row(collection: &str, n: u32, vector: Vec<f32>) -> Mutation {
    Mutation::Upsert {
        collection: name(collection),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "title": format!("row {n}"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vector),
    }
}

/// Vectors in twenty loose groups, one per row.
fn vectors() -> Vec<Vec<f32>> {
    let mut random = Random(17);
    let centres: Vec<Vec<f32>> = (0..20).map(|_| random.vector(DIMENSIONS)).collect();
    (0..ROWS)
        .map(|n| {
            let jitter = random.vector(DIMENSIONS);
            centres[n as usize % 20]
                .iter()
                .zip(jitter)
                .map(|(c, e)| c + 0.3 * e)
                .collect()
        })
        .collect()
}

/// The ids of the true nearest ten of `query` by cosine distance.
fn exact(vectors: &[Vec<f32>], query: &[f32]) -> Vec<ObjectId> {
    let mut all: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(n, v)| {
            (
                vellev_index::vector::distance(Distance::Cosine, query, v),
                n as u32,
            )
        })
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all[..10]
        .iter()
        .map(|(_, n)| ObjectId(uuid::Uuid::from_u128(u128::from(*n))))
        .collect()
}

/// How many times the limit is scored: few enough of these rows that an
/// index which misleads shows in what is found.
const OVERSAMPLE: usize = 64;

fn found(state: &State, vectors: &[Vec<f32>]) -> (usize, usize) {
    let (mut hits, mut total) = (0, 0);
    for query in vectors.iter().step_by(vectors.len() / 25) {
        let nearest = exact(vectors, query);
        let asked = Query {
            oversample: OVERSAMPLE,
            ..Query::nearest(query, 10)
        };
        let got = state.search(&name("notes"), &asked).unwrap();
        assert_eq!(got.len(), 10);
        hits += got
            .iter()
            .filter(|hit| nearest.contains(&hit.object.id))
            .count();
        total += 10;
    }
    (hits, total)
}

/// A writer with every row committed, in `segments` flushed segments.
async fn written(cluster: &Cluster, vectors: &[Vec<f32>], segments: usize) -> Writer {
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer.commit(vec![wide("notes")]).await.unwrap();
    for (segment, chunk) in vectors.chunks(vectors.len() / segments).enumerate() {
        for (batch, rows) in chunk.chunks(250).enumerate() {
            let first = segment * (vectors.len() / segments) + batch * 250;
            let writes = rows
                .iter()
                .enumerate()
                .map(|(i, vector)| row("notes", (first + i) as u32, vector.clone()));
            writer.commit(writes.collect()).await.unwrap();
        }
        writer.flush().await.unwrap();
    }
    writer
}

/// Four segments' worth of rows, flushed and merged into one.
async fn merged(cluster: &Cluster, vectors: &[Vec<f32>]) -> Writer {
    let writer = written(cluster, vectors, 4).await;
    writer.merge().await.unwrap().expect("four segments merge");
    writer
}

async fn segments_of(cluster: &Cluster) -> Vec<SegmentMeta> {
    let head = cluster.committed().await.manifest.unwrap().0;
    let bytes = cluster.blobs.get(&head).await.unwrap();
    let manifest: Manifest = format::decode(&head, Kind::Manifest, &bytes).unwrap();
    manifest.segments
}

#[tokio::test]
async fn a_large_merged_segment_is_clustered_kept_in_a_file_and_loaded_from_it() {
    let mut cluster = Cluster::new().await;
    let observed = cluster.observed();
    cluster.blobs = observed.clone();
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;

    let segments = segments_of(&cluster).await;
    assert_eq!(segments.len(), 1);
    let segment = &segments[0];
    let index = segment
        .index
        .clone()
        .expect("a clustered segment has an index file");
    assert!(index.contains("/index/") && index.ends_with("-merged.index"));
    let frame = cluster.blobs.get(&index).await.unwrap();
    let stored: SegmentIndex = format::decode(&index, Kind::Index, &frame).unwrap();
    assert_eq!(stored.segment, segment.key);
    assert_eq!(stored.vectors.len(), 1);
    assert_eq!(stored.vectors[0].1.len(), ROWS as usize);

    let (hits, total) = writer.partition().read(|state| found(state, &vectors));
    assert!(hits * 100 >= total * 97, "{hits} of {total} on the writer");

    observed.fetched();
    let follower = cluster.open().await;
    let fetched = observed.fetched();
    assert!(
        fetched.contains(&index),
        "the follower reads the index file"
    );
    let on_follower = follower.partition().read(|state| found(state, &vectors));
    assert_eq!(
        on_follower,
        (hits, total),
        "the same index gives the same hits"
    );

    cluster.blobs.delete(&index).await.unwrap();
    let rebuilt = cluster.open().await;
    let (hits, total) = rebuilt.partition().read(|state| found(state, &vectors));
    assert!(
        hits * 100 >= total * 97,
        "{hits} of {total} after rebuilding"
    );
}

#[tokio::test]
async fn cleanup_keeps_an_index_file_with_its_segment() {
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    let deleted = writer.clean(Duration::ZERO).await.unwrap();
    assert!(deleted > 0);
    let listed = cluster
        .blobs
        .list(&Layout(&cluster.key).prefix())
        .await
        .unwrap();
    let mut kinds: Vec<&str> = listed
        .iter()
        .map(|listed| listed.key.rsplit('.').next().unwrap())
        .collect();
    kinds.sort_unstable();
    assert_eq!(kinds, ["index", "manifest", "segment", "vectors"]);
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), ROWS as usize));
}

#[tokio::test]
async fn a_large_flush_is_clustered_too_and_its_index_outlives_later_checkpoints() {
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = written(&cluster, &vectors, 1).await;
    let segments = segments_of(&cluster).await;
    let index = segments[0]
        .index
        .clone()
        .expect("a large flush is clustered");
    assert!(index.ends_with(".index") && !index.contains("merged"));

    writer
        .commit(vec![row("notes", ROWS, vectors[0].clone())])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let segments = segments_of(&cluster).await;
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[1].index, None, "a small flush is not");
    writer.clean(Duration::ZERO).await.unwrap();
    cluster
        .blobs
        .get(&index)
        .await
        .expect("the index of a segment still listed stays");
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(state.count(&name("notes")), ROWS as usize + 1));
}

/// Replace the index file under `key` with one holding `index` for
/// `segment`.
async fn replace_index(cluster: &Cluster, key: &str, segment: &str, index: ClusteredIndex) {
    let frame = cluster.blobs.get(key).await.unwrap();
    let stored: SegmentIndex = format::decode(key, Kind::Index, &frame).unwrap();
    let tampered = SegmentIndex {
        segment: segment.to_string(),
        vectors: vec![(stored.vectors[0].0, Arc::new(index))],
    };
    let frame = format::encode(Kind::Index, &tampered).unwrap();
    cluster.blobs.delete(key).await.unwrap();
    cluster.blobs.put(key, frame).await.unwrap();
}

#[tokio::test]
async fn an_index_file_of_another_segment_or_of_other_rows_is_not_used() {
    let mut cluster = Cluster::new().await;
    let observed = cluster.observed();
    cluster.blobs = observed.clone();
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    let segment = segments_of(&cluster).await.remove(0);
    let index = segment.index.unwrap();
    let on_writer = writer.partition().read(|state| found(state, &vectors));

    let backwards: Vec<(u32, &[f32])> = vectors
        .iter()
        .rev()
        .enumerate()
        .map(|(row, vector)| (row as u32, vector.as_slice()))
        .collect();
    let wrong = ClusteredIndex::build(Distance::Cosine, &backwards).unwrap();
    replace_index(&cluster, &index, "another", wrong.clone()).await;
    observed.fetched();
    let follower = cluster.open().await;
    assert!(observed.fetched().contains(&index));
    let (hits, total) = follower.partition().read(|state| found(state, &vectors));
    assert_eq!(total, on_writer.1);
    assert!(
        hits * 100 >= total * 97,
        "{hits} of {total} by a scan instead"
    );

    let fewer = ClusteredIndex::build(Distance::Cosine, &backwards[..100]).unwrap();
    replace_index(&cluster, &index, &segment.key, fewer).await;
    let follower = cluster.open().await;
    let (hits, total) = follower.partition().read(|state| found(state, &vectors));
    assert!(
        hits * 100 >= total * 97,
        "{hits} of {total} by a scan instead"
    );

    replace_index(&cluster, &index, &segment.key, wrong).await;
    let misled = cluster.open().await;
    let (hits, total) = misled.partition().read(|state| found(state, &vectors));
    assert!(
        hits * 100 < total * 50,
        "{hits} of {total}: the check is what tells"
    );
}

/// An index file as it is stored, to write one that no build would give.
#[derive(Serialize, Deserialize)]
struct StoredIndexes {
    segment: String,
    vectors: Vec<(CollectionId, StoredIndex)>,
}

#[derive(Serialize, Deserialize)]
struct StoredIndex {
    metric: Distance,
    dimensions: usize,
    rotation: StoredRotation,
    centroids: Vec<f32>,
    clusters: Vec<StoredCluster>,
    probes: usize,
}

#[derive(Serialize, Deserialize)]
struct StoredRotation {
    width: usize,
    seed: u64,
}

#[derive(Serialize, Deserialize)]
struct StoredCluster {
    rows: ByteBuf,
    codes: ByteBuf,
    norms: ByteBuf,
    aligns: ByteBuf,
    alongs: ByteBuf,
}

/// An index file whose checksum holds and which reads, but which is no
/// index of the segment's vectors as they are, or says to scan a number of
/// clusters it has not got, is not used: the segment is searched by a scan
/// of its codes instead, and finds what it should.
#[tokio::test]
async fn an_index_file_that_does_not_fit_the_segments_vectors_is_not_used() {
    let cluster = Cluster::new().await;
    let vectors = vectors();
    merged(&cluster, &vectors).await;
    let segment = segments_of(&cluster).await.remove(0);
    let index = segment.index.unwrap();
    let original = cluster.blobs.get(&index).await.unwrap();
    let found_by_a_scan = |what: String| {
        let (cluster, vectors) = (&cluster, &vectors);
        async move {
            let follower = cluster.open().await;
            let (hits, total) = follower.partition().read(|state| found(state, vectors));
            assert!(
                hits * 100 >= total * 97,
                "{what}: {hits} of {total} by a scan instead"
            );
        }
    };

    let built = |metric: Distance, shift: u32, vectors: &[Vec<f32>]| {
        let rows: Vec<(u32, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(row, vector)| (row as u32 + shift, vector.as_slice()))
            .collect();
        ClusteredIndex::build(metric, &rows).unwrap()
    };
    // Of the vectors in reverse, so that an index of them misleads if it
    // is used, as the test of another segment's shows.
    let backwards: Vec<Vec<f32>> = vectors.iter().rev().cloned().collect();
    let wider: Vec<Vec<f32>> = vectors
        .iter()
        .map(|vector| vector.iter().copied().chain([0.0; 8]).collect())
        .collect();
    let unfitting = [
        ("another metric", built(Distance::Dot, 0, &backwards)),
        ("other dimensions", built(Distance::Cosine, 0, &wider)),
        ("other rows", built(Distance::Cosine, 1, &vectors)),
    ];
    for (what, unfitting) in unfitting {
        replace_index(&cluster, &index, &segment.key, unfitting).await;
        found_by_a_scan(format!("an index of {what}")).await;
    }

    for probes in [0, usize::MAX] {
        let mut stored: StoredIndexes = format::decode(&index, Kind::Index, &original).unwrap();
        stored.vectors[0].1.probes = probes;
        let frame = format::encode(Kind::Index, &stored).unwrap();
        cluster.blobs.delete(&index).await.unwrap();
        cluster.blobs.put(&index, frame).await.unwrap();
        found_by_a_scan(format!("an index to scan {probes} clusters")).await;
    }
}

#[tokio::test]
async fn a_segment_too_small_for_a_shortlist_gets_no_index_file() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 100,
        ..Merging::default()
    });
    writer.commit(vec![wide("notes")]).await.unwrap();
    let vectors = vectors();
    for rows in vectors[..3_000].chunks(250) {
        let writes = rows
            .iter()
            .enumerate()
            .map(|(i, vector)| row("notes", i as u32, vector.clone()));
        writer.commit(writes.collect()).await.unwrap();
    }
    writer.flush().await.unwrap();
    assert_eq!(segments_of(&cluster).await[0].index, None);
}

#[tokio::test]
async fn the_nearest_all_in_one_of_several_clustered_segments_are_all_found() {
    let cluster = Cluster::new().await;
    let mut random = Random(5);
    let query = random.vector(DIMENSIONS);
    let mut vectors: Vec<Vec<f32>> = (0..3 * ROWS).map(|_| random.vector(DIMENSIONS)).collect();
    for n in 0..10 {
        let jitter = random.vector(DIMENSIONS);
        vectors[(2 * ROWS + 100 + n) as usize] = query
            .iter()
            .zip(jitter)
            .map(|(q, j)| q + j * 0.01)
            .collect();
    }
    let writer = written(&cluster, &vectors, 3).await;
    let segments = segments_of(&cluster).await;
    assert_eq!(segments.len(), 3);
    assert!(segments.iter().all(|segment| segment.index.is_some()));

    let mut nearest = exact(&vectors, &query);
    nearest.sort();
    // Budgets one segment alone resolves at these few dimensions; what is
    // tested is that the other segments do not crowd them out.
    for oversample in [8, 32] {
        let asked = Query {
            oversample,
            ..Query::nearest(&query, 10)
        };
        let mut found: Vec<ObjectId> = writer.partition().read(|state| {
            state
                .search(&name("notes"), &asked)
                .unwrap()
                .iter()
                .map(|hit| hit.object.id)
                .collect()
        });
        found.sort();
        assert_eq!(found, nearest, "oversample {oversample}");
    }
}
