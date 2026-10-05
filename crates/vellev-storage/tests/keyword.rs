mod common;

use common::*;
use vellev_core::Object;
use vellev_index::text::{terms, words};
use vellev_storage::{Keywords, Scored, MAX_WEIGHT};

const NOTES: &str = "notes";
const EVERY: [(&str, f32); 3] = [("title", 1.0), ("body", 1.0), ("labels", 1.0)];

fn notes(bm25: Bm25) -> Mutation {
    let mut schema = schema(NOTES);
    schema.properties.extend([
        Property::new("body", DataType::Text).searchable(),
        Property::new("labels", DataType::TextArray).searchable(),
    ]);
    schema.bm25 = bm25;
    Mutation::CreateCollection(Box::new(schema))
}

fn note(object: u8, properties: serde_json::Value) -> Mutation {
    Mutation::Upsert {
        collection: name(NOTES),
        id: id(object),
        properties: properties.as_object().unwrap().clone(),
        vector: None,
    }
}

fn titled(object: u8, title: &str) -> Mutation {
    note(object, json!({ "title": title }))
}

fn search<'a>(state: &'a State, query: &Keywords<'_>) -> Vec<Scored<'a>> {
    state.keyword(&name(NOTES), query).unwrap()
}

fn ranked(matches: &[Scored<'_>]) -> Vec<(ObjectId, f32)> {
    matches.iter().map(|m| (m.object.id, m.score)).collect()
}

fn texts<'a>(object: &'a Object, property: &str) -> Option<Vec<&'a str>> {
    match object.properties.get(property)? {
        Value::Text(text) => Some(vec![text]),
        Value::TextArray(texts) => Some(texts.iter().map(String::as_str).collect()),
        _ => None,
    }
}

/// BM25 worked out the long way from a scan of the live objects, so that
/// the search is checked against neither its indexes nor its running totals.
fn brute_force(
    state: &State,
    text: &str,
    properties: &[(&str, f32)],
    bm25: Bm25,
    keep: impl Fn(&Object) -> bool,
) -> BTreeMap<ObjectId, f64> {
    let objects: Vec<&Object> = state.scan(&name(NOTES)).collect();
    let (k1, b) = (f64::from(bm25.k1), f64::from(bm25.b));
    let asked: Vec<String> = terms(text).collect();
    let distinct: BTreeSet<&String> = asked.iter().collect();
    let mut scores = BTreeMap::new();
    for (property, weight) in properties {
        let held: Vec<(&Object, Vec<String>)> = objects
            .iter()
            .filter_map(|object| {
                let texts = texts(object, property)?;
                Some((*object, texts.into_iter().flat_map(words).collect()))
            })
            .collect();
        let total: usize = held.iter().map(|(_, tokens)| tokens.len()).sum();
        let average = total as f64 / held.len() as f64;
        for token in &distinct {
            let repeats = asked.iter().filter(|other| other == token).count() as f64;
            let holding = held
                .iter()
                .filter(|(_, tokens)| tokens.contains(token))
                .count() as f64;
            let rarity = (1.0 + (objects.len() as f64 - holding + 0.5) / (holding + 0.5)).ln();
            for (object, tokens) in &held {
                let frequency = tokens.iter().filter(|other| other == token).count() as f64;
                if frequency > 0.0 && keep(object) {
                    let length = tokens.len() as f64 / average;
                    *scores.entry(object.id).or_insert(0.0) +=
                        f64::from(*weight) * repeats * rarity * frequency
                            / (frequency + k1 * (1.0 - b + b * length));
                }
            }
        }
    }
    scores
}

/// Check a search of `properties` for `text` against the long way: the same
/// objects, the same scores, best first, and any limit a prefix of the whole.
fn agrees(state: &State, text: &str, properties: &[(&str, f32)], bm25: Bm25) {
    let query = Keywords::matching(text, usize::MAX).within(properties);
    let found = search(state, &query);
    let expected = brute_force(state, text, properties, bm25, |_| true);
    let scored: BTreeMap<ObjectId, f32> = ranked(&found).into_iter().collect();
    assert_eq!(
        scored.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "{text:?} in {properties:?}"
    );
    for (id, score) in &scored {
        let expected = expected[id];
        assert!(
            (f64::from(*score) - expected).abs() <= 1e-4 * expected.max(1.0),
            "{text:?} scores {id:?} {score}, not {expected}"
        );
    }
    assert!(found.windows(2).all(|pair| {
        pair[0].score > pair[1].score
            || (pair[0].score == pair[1].score && pair[0].object.id < pair[1].object.id)
    }));
    for limit in [0, 1, 3] {
        let limited = search(
            state,
            &Keywords {
                limit,
                ..query.clone()
            },
        );
        assert_eq!(ranked(&limited), ranked(&found[..limit.min(found.len())]));
    }
}

#[tokio::test]
async fn scores_are_bm25_over_the_live_objects() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            notes(Bm25::default()),
            titled(1, "Red fox"),
            titled(2, "red, RED hen"),
            titled(3, "the blue hen"),
            note(4, json!({ "n": 4 })),
        ])
        .await
        .unwrap();
    let saturation = |frequency: f32, length: f32, average: f32| {
        frequency / (frequency + 1.2 * (0.25 + 0.75 * length / average))
    };
    writer.partition().read(|state| {
        let rarity = (1.0 + 2.5f32 / 2.5).ln();
        let expected = [
            (id(2), rarity * saturation(2.0, 3.0, 8.0 / 3.0)),
            (id(1), rarity * saturation(1.0, 2.0, 8.0 / 3.0)),
        ];
        let found = ranked(&search(state, &Keywords::matching("red", 10)));
        assert_eq!(found.len(), 2);
        for ((id, score), (expected_id, expected)) in found.iter().zip(expected) {
            assert_eq!(*id, expected_id);
            assert!((score - expected).abs() < 1e-6, "{score} is not {expected}");
        }
    });

    writer.commit(vec![delete(NOTES, 2)]).await.unwrap();
    writer.partition().read(|state| {
        let rarity = (1.0 + 2.5f32 / 1.5).ln();
        let found = ranked(&search(state, &Keywords::matching("red", 10)));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, id(1));
        let expected = rarity * saturation(1.0, 2.0, 2.5);
        assert!(
            (found[0].1 - expected).abs() < 1e-6,
            "{} is not {expected}",
            found[0].1
        );
    });
}

#[tokio::test]
async fn search_matches_the_long_way_across_layers_updates_and_deletes() {
    for bm25 in [
        Bm25::default(),
        Bm25 { k1: 2.0, b: 0.0 },
        Bm25 { k1: 0.5, b: 1.0 },
    ] {
        let cluster = Cluster::new().await;
        let writer = cluster.writer().await;
        let words = [
            "deploy", "rollback", "database", "the", "cache", "queue", "Deploy!",
        ];
        let mut random = Random(3);
        let mut pick = |count: usize| -> String {
            (0..count)
                .map(|_| words[((random.unit() + 0.5) * words.len() as f32) as usize % words.len()])
                .collect::<Vec<_>>()
                .join(" ")
        };
        let mut written = |object: u8| {
            let labels = [pick(1), pick(2)];
            match object % 4 {
                0 => note(object, json!({ "title": pick(3), "n": object })),
                1 => note(
                    object,
                    json!({ "title": pick(1), "body": pick(9), "n": object }),
                ),
                2 => note(
                    object,
                    json!({ "body": pick(4), "labels": labels, "n": object }),
                ),
                _ => note(object, json!({ "labels": [], "n": object })),
            }
        };
        let first: Vec<Mutation> = (0..60).map(&mut written).collect();
        writer
            .commit([notes(bm25)].into_iter().chain(first).collect())
            .await
            .unwrap();
        writer.flush().await.unwrap();
        let second: Vec<Mutation> = (40..90).map(&mut written).collect();
        writer
            .commit(
                second
                    .into_iter()
                    .chain((0..10).map(|n| delete(NOTES, n)))
                    .collect(),
            )
            .await
            .unwrap();
        writer.flush().await.unwrap();
        let third: Vec<Mutation> = (85..100).map(&mut written).collect();
        writer
            .commit(third.into_iter().chain([delete(NOTES, 50)]).collect())
            .await
            .unwrap();

        let queries = [
            "deploy",
            "rollback the database",
            "cache cache queue",
            "DEPLOY-queue",
        ];
        let fresh = cluster.open().await;
        for text in queries {
            for properties in [
                &EVERY[..],
                &[("title", 1.0)],
                &[("body", 0.5), ("labels", 3.0)],
            ] {
                let query = Keywords::matching(text, usize::MAX).within(properties);
                let kept = writer.partition().read(|state| {
                    agrees(state, text, properties, bm25);
                    ranked(&search(state, &query))
                });
                let loaded = fresh
                    .partition()
                    .read(|state| ranked(&search(state, &query)));
                assert_eq!(
                    kept, loaded,
                    "totals kept while writing differ from a fresh load"
                );
            }
        }
        writer.partition().read(|state| {
            let all = search(state, &Keywords::matching("deploy queue", usize::MAX));
            assert_eq!(
                ranked(&all),
                ranked(&search(
                    state,
                    &Keywords::matching("deploy queue", usize::MAX).within(&EVERY)
                ))
            );

            let odd = Filter::Eq {
                property: "n".into(),
                value: json!(41),
            };
            let even = Filter::Not(Box::new(odd.clone()));
            let without: Vec<_> = ranked(&all)
                .into_iter()
                .filter(|(found, _)| *found != id(41))
                .collect();
            assert!(without.len() < all.len());
            assert_eq!(
                ranked(&search(
                    state,
                    &Keywords::matching("deploy queue", usize::MAX).filtered(&even)
                )),
                without
            );
            let only = search(
                state,
                &Keywords::matching("deploy queue", usize::MAX).filtered(&odd),
            );
            assert_eq!(only.len(), 1);
            assert!(ranked(&all).contains(&ranked(&only)[0]));
        });
    }
}

#[tokio::test]
async fn a_query_with_nothing_to_find_finds_nothing() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![notes(Bm25::default())]).await.unwrap();
    let nothing = |state: &State, text: &str| {
        assert_eq!(search(state, &Keywords::matching(text, 10)), vec![])
    };
    writer.partition().read(|state| nothing(state, "fox"));
    writer
        .commit(vec![
            titled(1, "The quick fox"),
            note(2, json!({ "tags": ["fox"], "n": 1 })),
        ])
        .await
        .unwrap();
    writer.partition().read(|state| {
        for text in ["", "  ...  ", "the and of", "wolf", "foxes", "fo"] {
            nothing(state, text);
        }
        assert_eq!(search(state, &Keywords::matching("fox", 0)), vec![]);
        assert_eq!(
            ranked(&search(state, &Keywords::matching("the FOX?", 10))).len(),
            1
        );
    });
    writer.commit(vec![delete(NOTES, 1)]).await.unwrap();
    writer.partition().read(|state| nothing(state, "fox"));
}

#[tokio::test]
async fn a_dropped_collection_takes_its_text_and_totals_with_it() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let many = (0..30).map(|n| titled(n, "fox fox fox den"));
    writer
        .commit([notes(Bm25::default())].into_iter().chain(many).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![titled(40, "fox")]).await.unwrap();
    writer
        .commit(vec![
            titled(41, "fox den"),
            Mutation::DropCollection(name(NOTES)),
            notes(Bm25::default()),
            titled(1, "a fox"),
            titled(2, "hen"),
        ])
        .await
        .unwrap();
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            agrees(state, "fox hen den", &EVERY, Bm25::default());
            assert_eq!(search(state, &Keywords::matching("fox", 10)).len(), 1);
        });
    }
}

#[tokio::test]
async fn a_property_added_later_is_searched_from_then_on() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            notes(Bm25::default()),
            titled(1, "fox"),
            titled(2, "hen"),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let follower = cluster.open().await;
    let summary = Mutation::AddProperty {
        collection: name(NOTES),
        property: Property::new("summary", DataType::Text).searchable(),
    };
    writer
        .commit(vec![
            summary,
            note(3, json!({ "summary": "a fox in the den" })),
            note(1, json!({ "title": "fox", "summary": "den" })),
        ])
        .await
        .unwrap();
    follower.catch_up(&cluster.committed().await).await.unwrap();
    let every = [
        ("title", 1.0),
        ("body", 1.0),
        ("labels", 1.0),
        ("summary", 1.0),
    ];
    for copy in [
        writer.partition(),
        follower.partition(),
        cluster.open().await.partition(),
    ] {
        copy.read(|state| {
            agrees(state, "fox den hen", &every, Bm25::default());
            agrees(state, "fox den", &[("summary", 1.0)], Bm25::default());
            let found = search(state, &Keywords::matching("den", 10));
            assert_eq!(found.len(), 2);
        });
    }
}

#[tokio::test]
async fn a_query_the_collection_cannot_answer_is_refused() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![notes(Bm25::default()), titled(1, "fox")])
        .await
        .unwrap();
    writer.partition().read(|state| {
        let refused = |query: Keywords<'_>| state.keyword(&name(NOTES), &query).unwrap_err();
        let fox = Keywords::matching("fox", 10);
        for properties in [
            &[("tags", 1.0)][..],
            &[("n", 1.0)],
            &[("nope", 1.0)],
            &[("title", 0.0)],
            &[("title", -1.0)],
            &[("title", f32::NAN)],
            &[("title", f32::INFINITY)],
            &[("title", MAX_WEIGHT * 2.0)],
            &[("title", 1.0), ("body", 1.0), ("title", 2.0)],
        ] {
            assert!(
                matches!(
                    refused(fox.clone().within(properties)),
                    StorageError::InvalidQuery(_)
                ),
                "{properties:?}"
            );
        }
        let unknown = Filter::IsNull {
            property: "nope".into(),
        };
        assert!(matches!(
            refused(fox.clone().filtered(&unknown)),
            StorageError::Filter(_)
        ));
        assert!(matches!(
            state.keyword(&name("missing"), &fox),
            Err(StorageError::UnknownCollection(_))
        ));
    });
}

#[derive(Debug, Clone)]
enum Step {
    Put(
        u8,
        Vec<&'static str>,
        Option<Vec<&'static str>>,
        Vec<&'static str>,
    ),
    Delete(u8),
    Flush,
    Recreate,
}

fn step() -> impl Strategy<Value = Step> {
    let word = || prop::sample::select(vec!["fox", "hen", "den", "the", "Fox", "owl", "hen-fox"]);
    let words = move |most: usize| prop::collection::vec(word(), 0..=most);
    prop_oneof![
        6 => (0..12u8, words(5), prop::option::of(words(8)), words(3))
            .prop_map(|(object, title, body, labels)| Step::Put(object, title, body, labels)),
        3 => (0..12u8).prop_map(Step::Delete),
        2 => Just(Step::Flush),
        1 => Just(Step::Recreate),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn scores_never_drift_from_the_long_way(steps in prop::collection::vec(step(), 1..40)) {
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        runtime.block_on(async {
            let cluster = Cluster::new().await;
            let writer = cluster.writer().await;
            let bm25 = Bm25::default();
            writer.commit(vec![notes(bm25)]).await.unwrap();
            let trailing = cluster.open().await;
            let weighted = [("title", 2.0), ("body", 0.5), ("labels", 1.0)];
            for step in steps {
                match step {
                    Step::Put(object, title, body, labels) => {
                        let mut properties = json!({ "title": title.join(" "), "labels": labels });
                        if let Some(body) = body {
                            properties["body"] = json!(body.join(" "));
                        }
                        writer.commit(vec![note(object, properties)]).await.unwrap();
                    }
                    Step::Delete(object) => {
                        writer.commit(vec![delete(NOTES, object)]).await.unwrap();
                    }
                    Step::Flush => {
                        writer.flush().await.unwrap();
                    }
                    Step::Recreate => {
                        let recreate = vec![Mutation::DropCollection(name(NOTES)), notes(bm25)];
                        writer.commit(recreate).await.unwrap();
                    }
                }
                let fresh = cluster.open().await;
                trailing.catch_up(&cluster.committed().await).await.unwrap();
                for text in ["fox", "hen owl", "den den fox"] {
                    let query = Keywords::matching(text, usize::MAX).within(&weighted);
                    let kept = writer.partition().read(|state| {
                        agrees(state, text, &EVERY, bm25);
                        agrees(state, text, &weighted, bm25);
                        ranked(&search(state, &query))
                    });
                    for follower in [&fresh, &trailing] {
                        let followed = follower.partition().read(|state| ranked(&search(state, &query)));
                        assert_eq!(kept, followed);
                    }
                }
            }
        });
    }
}
