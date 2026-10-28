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
        only_if_version: None,
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

    // More clusters than any index has: used, it overflows where a query
    // works out what a scan visits. An index to scan none is refused too,
    // which the index's own tests tell; here it would find the same.
    let mut stored: StoredIndexes = format::decode(&index, Kind::Index, &original).unwrap();
    stored.vectors[0].1.probes = usize::MAX;
    let frame = format::encode(Kind::Index, &stored).unwrap();
    cluster.blobs.delete(&index).await.unwrap();
    cluster.blobs.put(&index, frame).await.unwrap();
    found_by_a_scan("an index to scan more clusters than it has".to_string()).await;
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

/// A clustered segment's index is counted in what the state holds.
#[tokio::test]
async fn a_clustered_index_is_counted_in_what_the_state_holds() {
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    let segment = segments_of(&cluster).await[0].clone();
    let key = segment.index.clone().unwrap();
    let frame = cluster.blobs.get(&key).await.unwrap();
    let stored: SegmentIndex = format::decode(&key, Kind::Index, &frame).unwrap();
    let index = stored.vectors[0].1.resident();
    writer.partition().read(|state| {
        let counted = state.indexes_resident();
        assert!(
            counted * 10 >= index * 9 && counted <= index * 2,
            "{counted} against {index}"
        );
        assert!(state.resident() >= counted + vectors.len() * DIMENSIONS * 4);
    });
}

/// A segment whose clustered index is built here keeps, in its manifest
/// entry, the curve of the rows the index was tuned on, which scans as the
/// index's own count does; and an index read back from its file scans by
/// that curve too.
#[tokio::test]
async fn a_built_clustered_segment_keeps_the_curve_of_its_rows() {
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    let segments = segments_of(&cluster).await;
    let segment = &segments[0];
    let index = segment.index.clone().expect("a clustered segment");
    let frame = cluster.blobs.get(&index).await.unwrap();
    let stored: SegmentIndex = format::decode(&index, Kind::Index, &frame).unwrap();
    let (collection, clustered) = &stored.vectors[0];
    let curves = &segment.probes;
    assert_eq!(curves.len(), 1, "{curves:?}");
    let (of, curve) = (&curves[0].collection, &curves[0].curve);
    assert_eq!(curves[0].band, 0);
    assert_eq!(of, collection);
    assert_eq!(curve.source, vellev_index::curve::Source::Rows);
    assert!(curve.fits(clustered.clusters()));
    assert_eq!(curve.probes(), clustered.probes());
    drop(writer);
    let follower = cluster.open().await;
    let opened: Vec<SegmentMeta> = follower
        .partition()
        .read(|state| state.segments().cloned().collect());
    assert_eq!(&opened[0].probes, curves, "every node reads the same curve");
}

/// Curves committed for a segment are kept in its entry and read by every
/// node, the memtable flushed along; a curve for a segment a merge has
/// replaced is not committed; and a later merge's segment keeps none of
/// its run's.
#[tokio::test]
async fn curves_committed_for_a_segment_are_kept_and_a_replaced_one_gets_none() {
    use vellev_index::curve::{Curve, Source};
    use vellev_storage::format::SegmentCurve;
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    let segment = segments_of(&cluster).await[0].clone();
    let built = segment.probes[0].clone();
    let clusters = built.curve.clusters;
    let asked = SegmentCurve {
        collection: built.collection,
        band: 0,
        curve: Curve::of(
            Source::Queries,
            50,
            clusters,
            (0..500).map(|n| n % clusters).collect(),
        )
        .unwrap(),
    };
    // One row more in the memtable, flushed along with the commit.
    writer
        .commit(vec![row("notes", ROWS, vectors[0].clone())])
        .await
        .unwrap();
    let committed = writer
        .commit_curves(vec![(segment.key.clone(), asked.clone())])
        .await
        .unwrap();
    assert!(committed.is_some(), "a curve for a segment that is there");
    let kept = segments_of(&cluster).await;
    assert_eq!(kept.len(), 2, "the memtable was flushed along: {kept:?}");
    assert_eq!(kept[0].key, segment.key);
    assert_eq!(
        kept[0].probes,
        vec![asked.clone()],
        "the queries curve replaces the rows one"
    );
    let follower = cluster.open().await;
    let read: Vec<SegmentMeta> = follower
        .partition()
        .read(|state| state.segments().cloned().collect());
    assert_eq!(read[0].probes, vec![asked.clone()], "every node reads it");

    // A curve for a segment no longer there: nothing committed.
    let gone = writer
        .commit_curves(vec![("tenant-a/0/segment/not-there.segment".into(), asked)])
        .await
        .unwrap();
    assert_eq!(gone, None);
}

/// Blobs as a node of a version before curves would leave them: every
/// manifest read with its segments' curves left out.
#[derive(Debug)]
struct Older {
    inner: Arc<dyn Blobs>,
}

#[async_trait::async_trait]
impl Blobs for Older {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        let bytes = self.inner.get(key).await?;
        if !key.ends_with(".manifest") {
            return Ok(bytes);
        }
        let mut manifest: Manifest = format::decode(key, Kind::Manifest, &bytes)?;
        manifest
            .segments
            .iter_mut()
            .for_each(|segment| segment.probes.clear());
        format::encode(Kind::Manifest, &manifest)
    }
    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// A segment from before curves, its index read back from its file, gets
/// the very curve its build would have given, by a pass over its rows; a
/// segment that has its curve gets nothing.
#[tokio::test]
async fn a_segment_from_before_curves_gets_the_curve_its_build_would_have_given() {
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    let built = segments_of(&cluster).await[0].probes.clone();
    assert_eq!(built.len(), 1);
    assert_eq!(writer.rows_pass().await.unwrap(), None, "nothing missing");
    drop(writer);

    let (lease, committed) = cluster.lease().await;
    let older: Arc<dyn Blobs> = Arc::new(Older {
        inner: cluster.blobs.clone(),
    });
    let successor = Writer::open(older, cluster.coordinator.clone(), lease, &committed)
        .await
        .unwrap();
    let seen: Vec<SegmentMeta> = successor
        .partition()
        .read(|state| state.segments().cloned().collect());
    assert!(seen[0].probes.is_empty(), "read as before curves");
    assert!(successor.rows_pass().await.unwrap().is_some());
    let after = segments_of(&cluster).await;
    assert_eq!(after[0].probes, built, "the curve its build gave");
    let kept: Vec<SegmentMeta> = successor
        .partition()
        .read(|state| state.segments().cloned().collect());
    assert_eq!(kept[0].probes, built);
}

/// The clustered indexes a partition holds are counted one for each, by
/// where their probes come from (recall note, R7): the curve of band 0 for
/// the index's collection by its source, a curve of a source this version
/// does not know as unknown, one of a known source that does not fit the
/// index as unfit, and none at all as the index's own count. A curve of
/// another band, or for a collection with no clustered index in the
/// segment, counts for nothing, and neither does a segment with no index.
#[tokio::test]
async fn clustered_indexes_are_counted_one_each_by_where_their_probes_come_from() {
    use vellev_index::curve::{Curve, Source};
    use vellev_storage::format::{CollectionId, SegmentCurve};
    use vellev_storage::{Probes, State};
    let cluster = Cluster::new().await;
    let vectors = vectors();
    let writer = merged(&cluster, &vectors).await;
    writer
        .commit(vec![row("notes", ROWS, vectors[0].clone())])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let segments = segments_of(&cluster).await;
    assert_eq!(segments.len(), 2, "{segments:?}");
    let segment = segments.iter().find(|s| s.index.is_some()).unwrap().clone();
    assert!(segments.iter().any(|s| s.index.is_none()));
    let built = segment.probes[0].clone();
    let probes = |writer: &Writer| writer.partition().read(State::probes);
    assert_eq!(
        probes(&writer),
        Probes {
            rows: 1,
            ..Probes::default()
        }
    );

    let clusters = built.curve.clusters;
    let curve = |source, clusters| SegmentCurve {
        collection: built.collection,
        band: 0,
        curve: Curve::of(
            source,
            50,
            clusters,
            (0..500).map(|n| n % clusters).collect(),
        )
        .unwrap(),
    };
    let one = |field: fn(&mut Probes) -> &mut u64| {
        let mut probes = Probes::default();
        *field(&mut probes) = 1;
        probes
    };
    for (what, given, counted) in [
        (
            "questions",
            curve(Source::Queries, clusters),
            one(|p| &mut p.queries),
        ),
        (
            "unknown",
            curve(Source::Unknown, clusters),
            one(|p| &mut p.unknown),
        ),
        (
            "unfit",
            curve(Source::Rows, clusters + 1),
            one(|p| &mut p.unfit),
        ),
    ] {
        writer
            .commit_curves(vec![(segment.key.clone(), given)])
            .await
            .unwrap()
            .unwrap();
        assert_eq!(probes(&writer), counted, "a curve of {what}");
    }

    // The rows curve back, with one of another band and one for a
    // collection the segment has no index of: still one index, of rows.
    let elsewhere = SegmentCurve {
        collection: CollectionId(built.collection.0 + 1),
        ..built.clone()
    };
    let banded = SegmentCurve {
        band: 2,
        ..curve(Source::Queries, clusters)
    };
    writer
        .commit_curves(vec![
            (segment.key.clone(), built.clone()),
            (segment.key.clone(), elsewhere),
            (segment.key.clone(), banded),
        ])
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        segments_of(&cluster)
            .await
            .iter()
            .map(|s| s.probes.len())
            .sum::<usize>(),
        3
    );
    assert_eq!(probes(&writer), one(|p| &mut p.rows));
    drop(writer);

    // Read as before curves: the index scans by its own count.
    let (lease, committed) = cluster.lease().await;
    let older: Arc<dyn Blobs> = Arc::new(Older {
        inner: cluster.blobs.clone(),
    });
    let successor = Writer::open(older, cluster.coordinator.clone(), lease, &committed)
        .await
        .unwrap();
    assert_eq!(probes(&successor), one(|p| &mut p.index));
}
