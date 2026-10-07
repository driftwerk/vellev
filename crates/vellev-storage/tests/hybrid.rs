mod common;

use common::*;
use vellev_core::ObjectError;
use vellev_storage::{Fused, Fusion, Hybrid, Keywords, Query, DEFAULT_ALPHA};

const NOTES: &str = "notes";
const EAST: [f32; 2] = [1.0, 0.0];

fn note(object: u8, title: &str, n: i64, vector: Option<[f32; 2]>) -> Mutation {
    Mutation::Upsert {
        collection: name(NOTES),
        id: id(object),
        properties: json!({ "title": title, "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: vector.map(Vec::from),
    }
}

/// Four notes: 1 nearest `EAST`, then 2, then 3, and 4 with no vector;
/// "fox" in 1, 2 and 4.
async fn notes() -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create(NOTES),
            note(1, "fox", 0, Some([1.0, 0.0])),
            note(2, "fox fox den", 1, Some([0.0, 1.0])),
            note(3, "hen", 1, Some([-1.0, 0.0])),
            note(4, "a fox and a hen", 0, None),
        ])
        .await
        .unwrap();
    (cluster, writer)
}

fn fox(vector: Option<&[f32]>) -> Hybrid<'_> {
    Hybrid::matching("fox", vector, 10)
}

fn hybrid<'a>(state: &'a State, query: &Hybrid<'_>) -> Vec<Fused<'a>> {
    state.hybrid(&name(NOTES), query).unwrap()
}

fn ids(fused: &[Fused<'_>]) -> Vec<ObjectId> {
    fused.iter().map(|fused| fused.object.id).collect()
}

/// The keyword score of each note holding "fox".
fn wording(state: &State) -> BTreeMap<ObjectId, f32> {
    state
        .keyword(&name(NOTES), &Keywords::matching("fox", 10))
        .unwrap()
        .into_iter()
        .map(|scored| (scored.object.id, scored.score))
        .collect()
}

fn close(found: f32, expected: f32) {
    assert!((found - expected).abs() < 1e-6, "{found} is not {expected}");
}

#[tokio::test]
async fn each_search_adds_its_weighted_share_between_its_worst_and_best() {
    let (cluster, writer) = notes().await;
    writer.flush().await.unwrap();
    writer.commit(vec![note(5, "owl", 0, None)]).await.unwrap();
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            let wording = wording(state);
            assert_eq!(
                wording.keys().copied().collect::<Vec<_>>(),
                [id(1), id(2), id(4)]
            );
            let (worst, best) = wording
                .values()
                .fold((f32::MAX, f32::MIN), |(worst, best), score| {
                    (worst.min(*score), best.max(*score))
                });
            let worded = |object: u8| 0.25 * (wording[&id(object)] - worst) / (best - worst);

            let fused = hybrid(state, &fox(Some(&EAST)));
            let by_id: BTreeMap<ObjectId, &Fused<'_>> =
                fused.iter().map(|fused| (fused.object.id, fused)).collect();
            assert_eq!(by_id.len(), 4);
            close(by_id[&id(1)].score, 0.75 + worded(1));
            close(by_id[&id(2)].score, 0.375 + worded(2));
            close(by_id[&id(3)].score, 0.0);
            close(by_id[&id(4)].score, worded(4));
            assert!(fused.windows(2).all(|pair| pair[0].score >= pair[1].score));
            assert_eq!(fused[0].object.id, id(1));

            let sides = |object: u8| {
                let fused = by_id[&id(object)];
                (fused.distance, fused.certainty, fused.keyword)
            };
            assert_eq!(sides(1), (Some(0.0), Some(1.0), Some(wording[&id(1)])));
            assert_eq!(sides(3), (Some(2.0), Some(0.0), None));
            assert_eq!(sides(4), (None, None, Some(wording[&id(4)])));
        });
    }
}

#[tokio::test]
async fn a_search_without_weight_is_not_run_and_no_vector_leaves_the_words_alone() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let by_words: Vec<ObjectId> = state
            .keyword(&name(NOTES), &Keywords::matching("fox", 10))
            .unwrap()
            .iter()
            .map(|scored| scored.object.id)
            .collect();
        let words_only = hybrid(
            state,
            &Hybrid {
                alpha: 0.0,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(ids(&words_only), by_words);
        close(words_only[0].score, 1.0);
        close(words_only[2].score, 0.0);
        assert!(words_only
            .iter()
            .all(|fused| fused.distance.is_none() && fused.keyword.is_some()));
        for alpha in [0.0, DEFAULT_ALPHA, 1.0] {
            assert_eq!(hybrid(state, &Hybrid { alpha, ..fox(None) }), words_only);
        }

        let by_vector: Vec<ObjectId> = state
            .search(&name(NOTES), &Query::nearest(&EAST, 10))
            .unwrap()
            .iter()
            .map(|hit| hit.object.id)
            .collect();
        let vector_only = hybrid(
            state,
            &Hybrid {
                alpha: 1.0,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(ids(&vector_only), by_vector);
        assert_eq!(ids(&vector_only), [id(1), id(2), id(3)]);
        let scores: Vec<f32> = vector_only.iter().map(|fused| fused.score).collect();
        assert_eq!(scores, [1.0, 0.5, 0.0]);
        assert!(vector_only.iter().all(|fused| fused.keyword.is_none()));

        assert_eq!(hybrid(state, &Hybrid::matching("", None, 10)), vec![]);
        assert_eq!(
            hybrid(
                state,
                &Hybrid {
                    limit: 0,
                    ..fox(Some(&EAST))
                }
            ),
            vec![]
        );
    });
}

#[tokio::test]
async fn ranked_fusion_counts_places_and_ties_break_by_id() {
    let (_cluster, writer) = notes().await;
    writer
        .commit(vec![
            note(7, "fox", 0, Some([1.0, 0.0])),
            note(6, "fox", 0, Some([2.0, 0.0])),
        ])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let relative = hybrid(state, &fox(Some(&EAST)));
        assert_eq!(ids(&relative)[..3], [id(1), id(6), id(7)]);
        assert_eq!(relative[0].score, relative[2].score);

        let by_words: Vec<ObjectId> = state
            .keyword(&name(NOTES), &Keywords::matching("fox", 10))
            .unwrap()
            .iter()
            .map(|scored| scored.object.id)
            .collect();
        assert_eq!(by_words[..3], [id(1), id(6), id(7)]);
        let ranked = hybrid(
            state,
            &Hybrid {
                fusion: Fusion::Ranked,
                alpha: 0.5,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(ids(&ranked)[..3], [id(1), id(6), id(7)]);
        close(ranked[0].score, 0.5 / 60.0 + 0.5 / 60.0);
        close(ranked[1].score, 0.5 / 61.0 + 0.5 / 61.0);
        close(ranked[2].score, 0.5 / 62.0 + 0.5 / 62.0);
        let hen = ranked
            .iter()
            .find(|fused| fused.object.id == id(3))
            .unwrap();
        close(hen.score, 0.5 / 64.0);
    });
}

#[tokio::test]
async fn each_search_offers_its_best_candidates_and_the_filter_binds_both() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let one_each = hybrid(
            state,
            &Hybrid {
                candidates: 1,
                limit: 1,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(ids(&one_each), [id(1)]);
        let best_worded = state
            .keyword(&name(NOTES), &Keywords::matching("fox", 1))
            .unwrap()[0]
            .object
            .id;
        assert_eq!(best_worded, id(1));
        close(one_each[0].score, 1.0);

        let few = hybrid(
            state,
            &Hybrid {
                candidates: 1,
                limit: 3,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(few.len(), 3, "a limit above the candidates raises them");

        let ones = Filter::Eq {
            property: "n".into(),
            value: json!(1),
        };
        let filtered = hybrid(
            state,
            &Hybrid {
                filter: Some(&ones),
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(ids(&filtered), [id(2), id(3)]);
        close(filtered[0].score, 1.0);
        close(filtered[1].score, 0.0);
    });
}

#[tokio::test]
async fn a_limit_beyond_the_usual_candidates_is_met() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut random = Random(9);
    let many = (0..150).map(|n| note(n, "fox", 0, Some([random.unit(), random.unit()])));
    writer
        .commit([create(NOTES)].into_iter().chain(many).collect())
        .await
        .unwrap();
    writer.partition().read(|state| {
        assert_eq!(
            hybrid(
                state,
                &Hybrid {
                    limit: 120,
                    ..fox(Some(&EAST))
                }
            )
            .len(),
            120
        );
        assert_eq!(
            hybrid(
                state,
                &Hybrid {
                    limit: 500,
                    ..fox(Some(&EAST))
                }
            )
            .len(),
            150
        );
        let usual = hybrid(
            state,
            &Hybrid {
                alpha: 1.0,
                limit: 100,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(usual.len(), 100);
        close(usual[99].score, 0.0);
    });
}

#[tokio::test]
async fn a_query_the_collection_cannot_answer_is_refused() {
    let (_cluster, writer) = notes().await;
    writer.partition().read(|state| {
        let refused = |query: Hybrid<'_>| state.hybrid(&name(NOTES), &query).unwrap_err();
        for alpha in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    refused(Hybrid {
                        alpha,
                        ..fox(Some(&EAST))
                    }),
                    StorageError::InvalidQuery(_)
                ),
                "{alpha}"
            );
            assert!(matches!(
                refused(Hybrid { alpha, ..fox(None) }),
                StorageError::InvalidQuery(_)
            ));
        }
        assert!(matches!(
            refused(fox(Some(&[1.0, 0.0, 0.0]))),
            StorageError::Object(ObjectError::VectorLength {
                expected: 2,
                got: 3
            })
        ));
        assert!(matches!(
            refused(fox(Some(&[1e20, 0.0]))),
            StorageError::Object(ObjectError::VectorTooLong)
        ));
        assert!(matches!(
            refused(Hybrid {
                properties: &[("n", 1.0)],
                ..fox(Some(&EAST))
            }),
            StorageError::InvalidQuery(_)
        ));
        for alpha in [0.0, 1.0] {
            let unusable = [1.0, 0.0, 0.0];
            assert!(matches!(
                refused(Hybrid {
                    alpha,
                    ..fox(Some(&unusable))
                }),
                StorageError::Object(ObjectError::VectorLength { .. })
            ));
            assert!(matches!(
                refused(Hybrid {
                    alpha,
                    properties: &[("n", 1.0)],
                    ..fox(Some(&EAST))
                }),
                StorageError::InvalidQuery(_)
            ));
        }
        let unknown = Filter::IsNull {
            property: "nope".into(),
        };
        assert!(matches!(
            refused(Hybrid {
                filter: Some(&unknown),
                ..fox(None)
            }),
            StorageError::Filter(_)
        ));
        assert!(matches!(
            state.hybrid(&name("missing"), &fox(None)),
            Err(StorageError::UnknownCollection(_))
        ));
    });
}

#[tokio::test]
async fn equal_scores_are_ordered_by_wording_then_nearness_then_id() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create(NOTES),
            note(1, "hen", 0, Some([1.0, 0.0])),
            note(2, "fox", 0, None),
            note(3, "owl", 0, Some([0.0, 1.0])),
            note(4, "fox fox fox and a hen with an owl", 0, None),
            note(5, "owl", 0, Some([0.0, 2.0])),
        ])
        .await
        .unwrap();
    writer.partition().read(|state| {
        for fusion in [Fusion::RelativeScore, Fusion::Ranked] {
            let fused = hybrid(
                state,
                &Hybrid {
                    alpha: 0.5,
                    fusion,
                    ..fox(Some(&EAST))
                },
            );
            assert_eq!(
                ids(&fused),
                [id(1), id(2), id(4), id(3), id(5)],
                "{fusion:?}"
            );
            assert_eq!(fused[0].score, fused[1].score);
        }
        let relative = hybrid(
            state,
            &Hybrid {
                alpha: 0.5,
                ..fox(Some(&EAST))
            },
        );
        assert_eq!(relative[2].score, relative[4].score);
    });
}

#[tokio::test]
async fn distances_of_any_size_fuse_to_scores_that_are_numbers() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut schema = schema(NOTES);
    schema.distance = Distance::L2;
    writer
        .commit(vec![
            Mutation::CreateCollection(Box::new(schema)),
            note(1, "fox", 0, Some([1.0, 0.0])),
            note(2, "fox den", 0, Some([0.0, 2.0])),
            // As long as a vector may be.
            note(3, "hen", 0, Some([-4.6e18, 0.0])),
        ])
        .await
        .unwrap();
    writer
        .commit(vec![delete(NOTES, 2), note(4, "den", 0, Some([0.0, 2.0]))])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let fused = hybrid(state, &fox(Some(&EAST)));
        assert_eq!(ids(&fused), [id(1), id(4), id(3)]);
        assert!(fused[2]
            .distance
            .is_some_and(|distance| distance > 2e37 && distance.is_finite()));
        let scores: Vec<f32> = fused.iter().map(|fused| fused.score).collect();
        assert_eq!((scores[0], scores[2]), (1.0, 0.0));
        assert!(scores[1] > 0.74 && scores[1] <= 0.75);
    });
}

#[tokio::test]
async fn objects_both_searches_offer_at_swapped_places_are_ordered_by_wording() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create(NOTES),
            note(1, "fox den den", 0, Some([1.0, 0.0])),
            note(2, "fox", 0, Some([1.0, 0.5])),
        ])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let query = Hybrid {
            alpha: 0.5,
            fusion: Fusion::Ranked,
            ..fox(Some(&EAST))
        };
        let fused = hybrid(state, &query);
        assert_eq!(fused[0].score, fused[1].score);
        assert!(fused[0].distance > fused[1].distance);
        assert_eq!(ids(&fused), [id(2), id(1)]);
    });
}
