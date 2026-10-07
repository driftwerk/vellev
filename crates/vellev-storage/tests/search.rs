mod common;

use common::*;
use vellev_core::{Object, ObjectError};
use vellev_storage::{Hit, Merging, Query};

const VECTORS: &str = "vectors";

fn collection(dimensions: u32, metric: Distance) -> Mutation {
    let mut schema = schema(VECTORS);
    schema.embedding.dimensions = dimensions;
    schema.distance = metric;
    Mutation::CreateCollection(Box::new(schema))
}

fn row(n: u64) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n)))
}

fn put(n: u64, group: i64, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Upsert {
        collection: name(VECTORS),
        id: row(n),
        properties: json!({ "n": group }).as_object().unwrap().clone(),
        vector,
    }
}

fn remove(n: u64) -> Mutation {
    Mutation::Delete {
        collection: name(VECTORS),
        id: row(n),
    }
}

fn ids(hits: &[Hit<'_>]) -> Vec<ObjectId> {
    hits.iter().map(|hit| hit.object.id).collect()
}

/// The distance worked out the long way, so that the search is not checked
/// against its own arithmetic.
fn distance(metric: Distance, a: &[f32], b: &[f32]) -> f32 {
    let sum = |term: fn(f32, f32) -> f32| a.iter().zip(b).map(|(x, y)| term(*x, *y)).sum::<f32>();
    match metric {
        Distance::Cosine => {
            let lengths = (sum(|x, _| x * x) * sum(|_, y| y * y)).sqrt();
            if lengths > 0.0 {
                1.0 - sum(|x, y| x * y) / lengths
            } else {
                1.0
            }
        }
        Distance::Dot => -sum(|x, y| x * y),
        Distance::L2 => sum(|x, y| (x - y) * (x - y)),
    }
}

/// What a search must return when every row is scored: the live objects
/// with a vector that pass `keep`, nearest first, ties by id.
fn brute_force(
    state: &State,
    metric: Distance,
    query: &[f32],
    limit: usize,
    keep: impl Fn(&Object) -> bool,
) -> Vec<ObjectId> {
    let mut scored: Vec<(f32, ObjectId)> = state
        .scan(&name(VECTORS))
        .filter(|found| keep(found.object))
        .filter_map(|found| Some((distance(metric, query, found.vector?), found.id())))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(limit).map(|(_, id)| id).collect()
}

fn group_is(group: i64) -> Filter {
    Filter::Eq {
        property: "n".into(),
        value: json!(group),
    }
}

#[tokio::test]
async fn search_matches_scoring_every_row_under_each_metric_and_across_layers() {
    for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
        let cluster = Cluster::new().await;
        let writer = cluster.writer().await;
        let mut random = Random(11);
        let first = (0..60).map(|n| put(n, (n % 3) as i64, Some(random.vector(8))));
        writer
            .commit([collection(8, metric)].into_iter().chain(first).collect())
            .await
            .unwrap();
        writer.flush().await.unwrap();
        let second = (40..90).map(|n| put(n, (n % 3) as i64, Some(random.vector(8))));
        writer
            .commit(second.chain((0..10).map(remove)).collect())
            .await
            .unwrap();
        writer.flush().await.unwrap();
        let third = (85..100).map(|n| put(n, (n % 3) as i64, Some(random.vector(8))));
        writer
            .commit(third.chain([remove(50), put(200, 0, None)]).collect())
            .await
            .unwrap();

        for copy in [
            writer.partition().clone(),
            cluster.open().await.partition().clone(),
        ] {
            copy.read(|state| {
                for _ in 0..20 {
                    let query = random.vector(8);
                    for limit in [1, 7, 200] {
                        let hits = state
                            .search(&name(VECTORS), &Query::nearest(&query, limit))
                            .unwrap();
                        assert_eq!(
                            ids(&hits),
                            brute_force(state, metric, &query, limit, |_| true),
                            "{metric:?}"
                        );
                        assert!(hits
                            .windows(2)
                            .all(|pair| pair[0].distance <= pair[1].distance));
                    }
                    let only = group_is(1);
                    let hits = state
                        .search(&name(VECTORS), &Query::nearest(&query, 5).filtered(&only))
                        .unwrap();
                    let in_group =
                        |object: &Object| object.properties.get("n") == Some(&Value::Int(1));
                    assert_eq!(
                        ids(&hits),
                        brute_force(state, metric, &query, 5, in_group),
                        "{metric:?}"
                    );
                }
                assert_eq!(
                    state
                        .search(&name(VECTORS), &Query::nearest(&[0.1; 8], 500))
                        .unwrap()
                        .len(),
                    89
                );
            });
        }
    }
}

#[tokio::test]
async fn cosine_results_carry_certainty_and_ties_break_by_id() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            collection(2, Distance::Cosine),
            put(3, 0, Some(vec![2.0, 0.0])),
            put(1, 0, Some(vec![1.0, 0.0])),
            put(2, 0, Some(vec![0.0, 3.0])),
            put(4, 0, Some(vec![-1.0, 0.0])),
            put(5, 0, Some(vec![0.0, 0.0])),
            put(6, 0, None),
        ])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let hits = state
            .search(&name(VECTORS), &Query::nearest(&[1.0, 0.0], 10))
            .unwrap();
        assert_eq!(ids(&hits), [row(1), row(3), row(2), row(5), row(4)]);
        let certainties: Vec<f32> = hits.iter().map(|hit| hit.certainty.unwrap()).collect();
        assert_eq!(certainties, [1.0, 1.0, 0.5, 0.5, 0.0]);
        assert_eq!(
            state
                .search(&name(VECTORS), &Query::nearest(&[1.0, 0.0], 0))
                .unwrap(),
            vec![]
        );
    });

    let other = cluster.writer().await;
    other
        .commit(vec![
            Mutation::DropCollection(name(VECTORS)),
            collection(2, Distance::L2),
            put(1, 0, Some(vec![3.0, 4.0])),
        ])
        .await
        .unwrap();
    other.partition().read(|state| {
        let hits = state
            .search(&name(VECTORS), &Query::nearest(&[0.0, 0.0], 10))
            .unwrap();
        assert_eq!(
            (hits.len(), hits[0].distance, hits[0].certainty),
            (1, 25.0, None)
        );
    });
}

#[tokio::test]
async fn a_query_the_collection_cannot_answer_is_refused() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            collection(2, Distance::Cosine),
            put(1, 0, Some(vec![1.0, 0.0])),
        ])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let search = |query: Query<'_>| state.search(&name(VECTORS), &query).unwrap_err();
        assert!(matches!(
            search(Query::nearest(&[1.0, 0.0, 0.0], 1)),
            StorageError::Object(ObjectError::VectorLength {
                expected: 2,
                got: 3
            })
        ));
        assert!(matches!(
            search(Query::nearest(&[f32::NAN, 0.0], 1)),
            StorageError::Object(ObjectError::VectorNotFinite)
        ));
        let unknown = Filter::IsNull {
            property: "nope".into(),
        };
        assert!(matches!(
            search(Query::nearest(&[1.0, 0.0], 1).filtered(&unknown)),
            StorageError::Filter(_)
        ));
        assert!(matches!(
            state.search(&name("missing"), &Query::nearest(&[1.0, 0.0], 1)),
            Err(StorageError::UnknownCollection(_))
        ));
    });
}

/// A collection of clustered vectors in one flushed segment, with the
/// queries to ask of it. Groups cut across the clusters, so a filter on the
/// group leaves every query with near neighbours to find.
async fn clustered(rows: u64) -> (Cluster, Writer, Vec<Vec<f32>>) {
    const DIMENSIONS: usize = 64;
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut random = Random(5);
    let centres: Vec<Vec<f32>> = (0..40).map(|_| random.vector(DIMENSIONS)).collect();
    let mut scattered = |n: u64| -> Vec<f32> {
        let noise = random.vector(DIMENSIONS);
        centres[(n % 40) as usize]
            .iter()
            .zip(noise)
            .map(|(c, e)| c + 0.3 * e)
            .collect()
    };
    let puts: Vec<Mutation> = (0..rows)
        .map(|n| put(n, (n / 40 % 4) as i64, Some(scattered(n))))
        .collect();
    let create = collection(DIMENSIONS as u32, Distance::Cosine);
    writer
        .commit([create].into_iter().chain(puts).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let queries = (0..20).map(|n| scattered(n * 13)).collect();
    (cluster, writer, queries)
}

/// How many of the true ten nearest each query finds, summed over `queries`.
fn found(
    state: &State,
    queries: &[Vec<f32>],
    oversample: usize,
    filter: Option<&Filter>,
    keep: &dyn Fn(&Object) -> bool,
) -> usize {
    queries
        .iter()
        .map(|query| {
            let asked = Query {
                filter,
                oversample,
                ..Query::nearest(query, 10)
            };
            let hits = state.search(&name(VECTORS), &asked).unwrap();
            assert!(
                hits.iter().all(|hit| keep(hit.object)),
                "a hit the filter excludes"
            );
            let truth = brute_force(state, Distance::Cosine, query, 10, keep);
            ids(&hits).iter().filter(|id| truth.contains(id)).count()
        })
        .sum()
}

fn in_group_two(object: &Object) -> bool {
    object.properties.get("n") == Some(&Value::Int(2))
}

#[tokio::test]
async fn a_large_segment_is_shortlisted_only_where_many_rows_are_allowed() {
    let (_cluster, writer, queries) = clustered(12_000).await;
    let not_two = Filter::Ne {
        property: "n".into(),
        value: json!(2),
    };
    writer.partition().read(|state| {
        let everything = found(state, &queries, 64, None, &|_| true);
        assert!(
            everything >= 190,
            "{everything} of 200 true neighbours found"
        );
        let many = found(state, &queries, 64, Some(&not_two), &|object| {
            !in_group_two(object)
        });
        assert!(
            many >= 190,
            "{many} of 200 found among the 9,000 rows that pass"
        );

        // With a shortlist no longer than the answer, only scoring every
        // row finds all of them: 3,000 rows pass, which is few enough.
        assert_eq!(
            found(state, &queries, 1, Some(&group_is(2)), &in_group_two),
            200
        );
        assert!(found(state, &queries, 1, None, &|_| true) < 200);
        assert_eq!(
            found(state, &queries, 0, None, &|_| true),
            found(state, &queries, 1, None, &|_| true)
        );
        let narrow = Query {
            oversample: 0,
            ..Query::nearest(&queries[0], 10)
        };
        assert_eq!(state.search(&name(VECTORS), &narrow).unwrap().len(), 10);
    });
}

#[tokio::test]
async fn a_large_segment_with_few_rows_left_alive_is_scored_exactly() {
    let (cluster, writer, queries) = clustered(12_000).await;
    writer
        .commit((0..12_000).filter(|n| n % 10 != 0).map(remove).collect())
        .await
        .unwrap();
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            assert_eq!(state.count(&name(VECTORS)), 1_200);
            assert_eq!(found(state, &queries, 1, None, &|_| true), 200);
        });
    }
}

#[tokio::test]
async fn a_shortlisted_segment_gives_way_to_newer_layers() {
    let (cluster, writer, queries) = clustered(12_000).await;
    let query = &queries[0];
    let nearest = writer.partition().read(|state| {
        ids(&state
            .search(&name(VECTORS), &Query::nearest(query, 3))
            .unwrap())
    });
    let as_number = |id: ObjectId| id.0.as_u128() as u64;
    let far: Vec<f32> = query.iter().map(|v| -v).collect();
    writer
        .commit(vec![
            remove(as_number(nearest[0])),
            put(as_number(nearest[1]), 0, Some(far)),
            put(99_999, 0, Some(query.clone())),
        ])
        .await
        .unwrap();
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            let hits = state
                .search(&name(VECTORS), &Query::nearest(query, 3))
                .unwrap();
            assert_eq!(hits[0].object.id, row(99_999));
            assert_eq!(hits[0].certainty, Some(1.0));
            assert_eq!(hits[1].object.id, nearest[2]);
            assert!(!ids(&hits).contains(&nearest[0]) && !ids(&hits).contains(&nearest[1]));
        });
    }
}

/// More rows than are scored one by one, so that the segment they are
/// flushed to is searched through its vector index.
const SHORTLISTED: u64 = 5_000;

/// Write `vectors`, by id, to a collection of their length measured by
/// `metric`, and flush them to one segment.
async fn flushed(writer: &Writer, metric: Distance, vectors: Vec<(u64, Vec<f32>)>) {
    let dimensions = vectors[0].1.len() as u32;
    let puts = vectors
        .into_iter()
        .map(|(n, vector)| put(n, 0, Some(vector)));
    writer
        .commit(
            [collection(dimensions, metric)]
                .into_iter()
                .chain(puts)
                .collect(),
        )
        .await
        .unwrap();
    writer.flush().await.unwrap();
}

/// Vectors leaning one way, far from the origin, and one fifty times as
/// long as the query in its very direction: neither the lean they share
/// nor its length may hide it from a cosine search.
#[tokio::test]
async fn a_shortlisted_segment_finds_a_vector_in_the_querys_direction_whatever_its_length() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut random = Random(23);
    let query = random.vector(64);
    let leaning = (0..SHORTLISTED).map(|n| {
        let vector = random.vector(64).into_iter().map(|v| v + 1.0).collect();
        (n, vector)
    });
    let along = (SHORTLISTED, query.iter().map(|v| 50.0 * v).collect());
    flushed(&writer, Distance::Cosine, leaning.chain([along]).collect()).await;
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            let hits = state
                .search(&name(VECTORS), &Query::nearest(&query, 1))
                .unwrap();
            assert_eq!(ids(&hits), [row(SHORTLISTED)]);
            assert!(hits[0].distance < 1e-6, "{}", hits[0].distance);
        });
    }
}

/// One vector a hundred times as long as the rest, at nearly a right angle
/// to the query: its inner product with the query, 40, is still far the
/// largest, the rest's being under 3.
#[tokio::test]
async fn a_shortlisted_segment_finds_a_long_vector_with_the_largest_inner_product() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut random = Random(29);
    let query = random.vector(64);
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let others: Vec<(u64, Vec<f32>)> = (0..SHORTLISTED).map(|n| (n, random.vector(64))).collect();
    assert!(others.iter().all(|(_, vector)| dot(vector, &query) < 3.0));
    let sideways = random.vector(64);
    let off = (dot(&sideways, &query) - 0.4) / dot(&query, &query);
    let long: Vec<f32> = sideways
        .iter()
        .zip(&query)
        .map(|(s, q)| 100.0 * (s - off * q))
        .collect();
    assert!((dot(&long, &query) - 40.0).abs() < 0.01);
    flushed(
        &writer,
        Distance::Dot,
        others.into_iter().chain([(SHORTLISTED, long)]).collect(),
    )
    .await;
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            for limit in [1, 5] {
                let hits = state
                    .search(&name(VECTORS), &Query::nearest(&query, limit))
                    .unwrap();
                assert_eq!(hits[0].object.id, row(SHORTLISTED), "at limit {limit}");
            }
        });
    }
}

/// Rows of one vector, written in descending id order so that their rows
/// run the other way, come back in id order from a segment searched
/// through either kind of index.
#[tokio::test]
async fn rows_of_one_vector_come_back_in_id_order_from_a_shortlisted_segment() {
    let clustered = Merging {
        clustered_above: 4_000,
        ..Merging::default()
    };
    for merging in [Merging::default(), clustered] {
        let cluster = Cluster::new().await;
        let writer = cluster.writer().await.merging_with(merging);
        let same = vec![0.5; 8];
        let backwards = (1..=SHORTLISTED).rev().map(|n| (n, same.clone()));
        flushed(&writer, Distance::L2, backwards.collect()).await;
        for copy in [
            writer.partition().clone(),
            cluster.open().await.partition().clone(),
        ] {
            copy.read(|state| {
                for query in [same.clone(), vec![-1.0; 8]] {
                    for limit in [1, 3] {
                        let hits = state
                            .search(&name(VECTORS), &Query::nearest(&query, limit))
                            .unwrap();
                        let first: Vec<ObjectId> = (1..=limit as u64).map(row).collect();
                        assert_eq!(ids(&hits), first, "{merging:?} at limit {limit}");
                    }
                }
            });
        }
    }
}

/// Copies of one vector in two clustered segments, each written in
/// descending id order, searched scoring no more than each segment's own
/// nearest few: those few are its lowest ids, so the lowest of all are
/// found.
#[tokio::test]
async fn each_clustered_segments_own_nearest_few_are_its_lowest_ids() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer
        .commit(vec![collection(8, Distance::L2)])
        .await
        .unwrap();
    let same = vec![0.5; 8];
    for first in [1, SHORTLISTED + 1] {
        let backwards = (first..first + SHORTLISTED).rev();
        writer
            .commit(backwards.map(|n| put(n, 0, Some(same.clone()))).collect())
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            // Six rows scored, three from each segment.
            let asked = Query {
                oversample: 2,
                ..Query::nearest(&same, 3)
            };
            let hits = state.search(&name(VECTORS), &asked).unwrap();
            assert_eq!(ids(&hits), [row(1), row(2), row(3)]);
        });
    }
}

/// Values each finite but large enough that a distance overflows: the
/// inner product of the first two with the third is not a number, the
/// cosine of the first against [1, 0] comes out as a right angle, and the
/// last two are infinitely far from their opposites.
#[tokio::test]
async fn vectors_too_long_to_measure_are_refused_on_write_and_on_query() {
    let too_long = [
        vec![1e30, -1e30],
        vec![1e30, 1e30],
        vec![1e20, 0.0],
        vec![1.3e19, 0.0],
        vec![-1.3e19, 0.0],
    ];
    for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
        let cluster = Cluster::new().await;
        let writer = cluster.writer().await;
        writer
            .commit(vec![collection(2, metric), put(1, 0, Some(vec![1.0, 1.0]))])
            .await
            .unwrap();
        for vector in &too_long {
            let error = writer
                .commit(vec![put(2, 0, Some(vector.clone()))])
                .await
                .unwrap_err();
            assert!(
                matches!(error, StorageError::Object(ObjectError::VectorTooLong)),
                "{metric:?} {vector:?}: {error:?}"
            );
            writer.partition().read(|state| {
                let error = state
                    .search(&name(VECTORS), &Query::nearest(vector, 3))
                    .unwrap_err();
                assert!(
                    matches!(error, StorageError::Object(ObjectError::VectorTooLong)),
                    "{metric:?} {vector:?}: {error:?}"
                );
            });
        }
        writer.partition().read(|state| {
            assert_eq!(
                ids(&state
                    .search(&name(VECTORS), &Query::nearest(&[4e18, 0.0], 3))
                    .unwrap()),
                [row(1)]
            );
        });
    }
}
