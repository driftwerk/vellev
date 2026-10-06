mod common;

use common::*;
use vellev_core::{Object, ObjectError};
use vellev_storage::{Hit, Query};

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
