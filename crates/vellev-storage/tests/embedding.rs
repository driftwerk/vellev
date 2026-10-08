mod common;

use common::*;
use vellev_storage::{Outcome, Unembedded, Vectorize};

const NOTES: &str = "notes";

/// Embeds a text as its length and its number of words, and remembers what
/// it was asked for.
#[derive(Debug, Default)]
struct Lengths {
    asked: Mutex<Vec<Vec<String>>>,
    calls: AtomicUsize,
    fault: AtomicUsize,
}

const AWAY: usize = 1;
const TOO_LONG: usize = 2;
const NOT_A_NUMBER: usize = 3;
const TOO_FEW: usize = 4;
const SILENT: usize = 5;

fn vector_of(text: &str) -> Vec<f32> {
    vec![text.len() as f32, text.split_whitespace().count() as f32]
}

#[async_trait]
impl Vectorize for Lengths {
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, Unembedded> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let fault = self.fault.load(Ordering::SeqCst);
        if fault == AWAY || embedding.model != "lengths" {
            return Err(Unembedded::for_now("the model is away"));
        }
        if fault == SILENT {
            std::future::pending::<()>().await;
        }
        let vectors: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| match fault {
                TOO_LONG => [vector_of(text), vec![0.0]].concat(),
                NOT_A_NUMBER => vec![f32::NAN, 1.0],
                _ => vector_of(text),
            })
            .skip(usize::from(fault == TOO_FEW))
            .collect();
        self.asked.lock().unwrap().push(texts);
        Ok(vectors)
    }
}

impl Lengths {
    fn asked(&self) -> Vec<Vec<String>> {
        std::mem::take(&mut self.asked.lock().unwrap())
    }
}

fn embedding_notes() -> Mutation {
    let mut schema = schema(NOTES);
    schema.embedding = Embedding {
        provider: EmbeddingProvider::Local,
        model: "lengths".into(),
        properties: vec!["title".into(), "tags".into()],
        ..schema.embedding
    };
    Mutation::CreateCollection(Box::new(schema))
}

fn note(object: u8, properties: serde_json::Value, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Upsert {
        collection: name(NOTES),
        id: id(object),
        properties: properties.as_object().unwrap().clone(),
        vector,
    }
}

fn patch(object: u8, properties: serde_json::Value, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Patch {
        collection: name(NOTES),
        id: id(object),
        properties: properties.as_object().unwrap().clone(),
        vector,
        only_if: None,
    }
}

async fn embedding() -> (Cluster, Writer, Arc<Lengths>) {
    let cluster = Cluster::new().await;
    let lengths = Arc::new(Lengths::default());
    let writer = cluster.writer().await.embedding_with(lengths.clone());
    writer.commit(vec![embedding_notes()]).await.unwrap();
    (cluster, writer, lengths)
}

fn vector(writer: &Writer, object: u8) -> Option<Vec<f32>> {
    writer.partition().read(|state| {
        state
            .get(&name(NOTES), id(object))
            .unwrap()
            .vector
            .map(<[f32]>::to_vec)
    })
}

fn asked(texts: &[&str]) -> Vec<Vec<String>> {
    vec![texts.iter().map(|text| text.to_string()).collect()]
}

#[tokio::test]
async fn an_object_written_without_a_vector_is_given_the_one_its_text_embeds_to() {
    let (cluster, writer, lengths) = embedding().await;
    writer
        .commit(vec![
            note(
                1,
                json!({ "title": "a red fox", "tags": ["den", "night"] }),
                None,
            ),
            note(2, json!({ "title": "supplied" }), Some(vec![7.0, 7.0])),
            note(3, json!({ "n": 3 }), None),
            note(4, json!({ "title": "  ", "tags": [] }), None),
        ])
        .await
        .unwrap();
    assert_eq!(lengths.asked(), asked(&["a red fox\nden\nnight"]));
    assert_eq!(vector(&writer, 1), Some(vector_of("a red fox\nden\nnight")));
    assert_eq!(vector(&writer, 2), Some(vec![7.0, 7.0]));
    assert_eq!(vector(&writer, 3), None);
    assert_eq!(vector(&writer, 4), None);

    let follower = cluster.open().await;
    let followed = follower.partition().read(|state| {
        state
            .get(&name(NOTES), id(1))
            .unwrap()
            .vector
            .map(<[f32]>::to_vec)
    });
    assert_eq!(followed, vector(&writer, 1));
    assert!(lengths.asked().is_empty());
}

#[tokio::test]
async fn a_vector_is_kept_while_the_text_stands_and_remade_when_it_changes() {
    let (_cluster, writer, lengths) = embedding().await;
    let fox = json!({ "title": "fox", "n": 1 });
    writer
        .commit(vec![note(1, fox.clone(), None)])
        .await
        .unwrap();
    lengths.asked();

    let again = writer.commit(vec![note(1, fox, None)]).await.unwrap();
    assert_eq!(again.outcomes, [Outcome::Unchanged]);
    writer
        .commit(vec![
            patch(1, json!({ "n": 2 }), None),
            Mutation::Increment {
                collection: name(NOTES),
                id: id(1),
                property: "n".into(),
                by: 1.0,
            },
            note(1, json!({ "title": "fox", "score": 0.5 }), None),
        ])
        .await
        .unwrap();
    assert!(lengths.asked().is_empty());
    assert_eq!(vector(&writer, 1), Some(vector_of("fox")));

    writer
        .commit(vec![patch(1, json!({ "title": "a hen" }), None)])
        .await
        .unwrap();
    assert_eq!(lengths.asked(), asked(&["a hen"]));
    assert_eq!(vector(&writer, 1), Some(vector_of("a hen")));

    let append = Mutation::Append {
        collection: name(NOTES),
        id: id(1),
        property: "tags".into(),
        values: vec!["coop".into()],
        unique: true,
    };
    writer.commit(vec![append]).await.unwrap();
    assert_eq!(lengths.asked(), asked(&["a hen\ncoop"]));

    writer
        .commit(vec![patch(
            1,
            json!({ "title": "an owl" }),
            Some(vec![9.0, 9.0]),
        )])
        .await
        .unwrap();
    writer
        .commit(vec![patch(1, json!({ "n": 9 }), None)])
        .await
        .unwrap();
    assert!(lengths.asked().is_empty());
    assert_eq!(vector(&writer, 1), Some(vec![9.0, 9.0]));

    writer
        .commit(vec![patch(1, json!({ "title": null, "tags": null }), None)])
        .await
        .unwrap();
    assert!(lengths.asked().is_empty());
    assert_eq!(vector(&writer, 1), None);
}

#[tokio::test]
async fn one_commit_embeds_each_object_once_as_it_ends_up() {
    let (cluster, writer, lengths) = embedding().await;
    let stale = writer.commit_group(vec![
        vec![
            note(1, json!({ "title": "first" }), None),
            patch(1, json!({ "title": "second" }), None),
            patch(1, json!({ "n": 4 }), None),
            note(2, json!({ "title": "written then deleted" }), None),
            delete(NOTES, 2),
        ],
        vec![
            note(3, json!({ "title": "rejected with its batch" }), None),
            patch(9, json!({ "n": 1 }), None),
        ],
        vec![
            note(4, json!({ "title": "other caller" }), None),
            note(5, json!({ "title": "has one" }), None),
            patch(5, json!({ "title": "has two" }), Some(vec![1.0, 1.0])),
            note(6, json!({ "title": "had one" }), Some(vec![1.0, 1.0])),
            patch(6, json!({ "title": "has none now" }), None),
        ],
    ]);
    let results = stale.await.unwrap();
    assert!(results[0].is_ok() && results[2].is_ok());
    assert!(matches!(
        results[1],
        Err(StorageError::UnknownObject { .. })
    ));
    assert_eq!(
        lengths.asked(),
        asked(&["second", "other caller", "has none now"])
    );

    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            let vector = |object: u8| {
                state
                    .get(&name(NOTES), id(object))
                    .unwrap()
                    .vector
                    .map(<[f32]>::to_vec)
            };
            assert_eq!(vector(1), Some(vector_of("second")));
            assert_eq!(vector(4), Some(vector_of("other caller")));
            assert_eq!(vector(5), Some(vec![1.0, 1.0]));
            assert_eq!(vector(6), Some(vector_of("has none now")));
            assert_eq!(state.count(&name(NOTES)), 4);
        });
    }

    writer
        .commit(vec![
            note(7, json!({ "title": "gone with its collection" }), None),
            Mutation::DropCollection(name(NOTES)),
        ])
        .await
        .unwrap();
    assert!(lengths.asked().is_empty());
}

#[tokio::test]
async fn a_commit_that_cannot_be_embedded_changes_nothing_and_the_writer_carries_on() {
    let (_cluster, writer, lengths) = embedding().await;
    let before = writer.partition().read(State::position);
    let fox = || vec![note(1, json!({ "title": "fox" }), None)];

    lengths.fault.store(AWAY, Ordering::SeqCst);
    let failed = writer.commit(fox()).await.unwrap_err();
    assert!(
        matches!(&failed, StorageError::Embedding { model, reason, lasting: None } if model == "lengths" && reason == "the model is away"),
        "{failed}"
    );
    // Vectors of another size than the collection's will not change by
    // asking again; too few, or one not finite, might.
    for (fault, lasts) in [
        (TOO_LONG, Some(vellev_core::ModelFault::Provider)),
        (NOT_A_NUMBER, None),
        (TOO_FEW, None),
    ] {
        lengths.fault.store(fault, Ordering::SeqCst);
        let failed = writer.commit(fox()).await.unwrap_err();
        assert!(
            matches!(failed, StorageError::Embedding { lasting, .. } if lasting == lasts),
            "{fault}: {failed:?}"
        );
    }
    lengths.fault.store(0, Ordering::SeqCst);
    lengths.asked();

    writer.partition().read(|state| {
        assert_eq!(state.position(), before);
        assert_eq!(state.count(&name(NOTES)), 0);
    });
    writer
        .commit(vec![note(
            2,
            json!({ "title": "needs none" }),
            Some(vec![1.0, 2.0]),
        )])
        .await
        .unwrap();
    writer.commit(fox()).await.unwrap();
    assert_eq!(vector(&writer, 1), Some(vector_of("fox")));
}

#[tokio::test]
async fn a_writer_with_nothing_to_embed_with_refuses_only_what_needs_it() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let mut plain = schema("plain");
    plain.embedding.properties = vec!["title".into()];
    writer
        .commit(vec![
            embedding_notes(),
            Mutation::CreateCollection(Box::new(plain)),
            upsert("plain", 1, 1),
            note(1, json!({ "title": "supplied" }), Some(vec![1.0, 2.0])),
            note(2, json!({ "n": 2 }), None),
        ])
        .await
        .unwrap();
    writer
        .commit(vec![patch(1, json!({ "n": 5 }), None)])
        .await
        .unwrap();
    let refused = writer
        .commit(vec![note(3, json!({ "title": "fox" }), None)])
        .await
        .unwrap_err();
    assert!(
        matches!(
            refused,
            StorageError::Embedding {
                lasting: Some(vellev_core::ModelFault::Setup),
                ..
            }
        ),
        "{refused}"
    );

    let plain = Mutation::Patch {
        collection: name("plain"),
        id: id(1),
        properties: json!({ "title": "changed" }).as_object().unwrap().clone(),
        vector: None,
        only_if: None,
    };
    writer.commit(vec![plain]).await.unwrap();
    let kept = writer.partition().read(|state| {
        state
            .get(&name("plain"), id(1))
            .unwrap()
            .vector
            .map(<[f32]>::to_vec)
    });
    assert_eq!(kept, Some(vec![1.0, 0.5]));
}

fn other_notes() -> Mutation {
    let mut schema = schema("other");
    schema.embedding = Embedding {
        provider: EmbeddingProvider::Local,
        model: "absent".into(),
        properties: vec!["title".into()],
        ..schema.embedding
    };
    Mutation::CreateCollection(Box::new(schema))
}

fn other(object: u8, title: &str, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Upsert {
        collection: name("other"),
        id: id(object),
        properties: json!({ "title": title }).as_object().unwrap().clone(),
        vector,
    }
}

#[tokio::test]
async fn a_batch_whose_vectors_cannot_be_made_is_rejected_alone() {
    let (cluster, writer, lengths) = embedding().await;
    writer
        .commit(vec![other_notes(), create("plain")])
        .await
        .unwrap();
    let results = writer
        .commit_group(vec![
            vec![upsert("plain", 1, 1)],
            vec![
                other(1, "cannot be embedded", None),
                note(8, json!({ "title": "lost with it" }), None),
            ],
            vec![note(1, json!({ "title": "fox" }), None)],
            vec![other(2, "supplied", Some(vec![1.0, 2.0]))],
            vec![Mutation::Patch {
                collection: name("other"),
                id: id(1),
                properties: json!({ "n": 1 }).as_object().unwrap().clone(),
                vector: None,
                only_if: None,
            }],
            vec![
                note(2, json!({ "title": "fox" }), None),
                note(3, json!({ "title": "hen" }), None),
            ],
        ])
        .await
        .unwrap();
    assert!(
        matches!(&results[1], Err(StorageError::Embedding { model, .. }) if model == "absent"),
        "{:?}",
        results[1]
    );
    assert!(matches!(
        results[4],
        Err(StorageError::UnknownObject { .. })
    ));
    let accepted: Vec<_> = [0, 2, 3, 5]
        .iter()
        .map(|at| results[*at].as_ref().unwrap())
        .collect();
    assert!(accepted
        .windows(2)
        .all(|pair| pair[0].position == pair[1].position));
    let texts: Vec<String> = lengths.asked().into_iter().flatten().collect();
    for text in ["fox", "hen"] {
        let times = texts.iter().filter(|asked| *asked == text).count();
        assert_eq!(times, 1, "{text} once, however many objects or rounds");
    }

    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            assert_eq!(state.count(&name("plain")), 1);
            assert_eq!(state.count(&name("other")), 1);
            assert_eq!(state.count(&name(NOTES)), 3);
            let vector = |object: u8| {
                state
                    .get(&name(NOTES), id(object))
                    .unwrap()
                    .vector
                    .map(<[f32]>::to_vec)
            };
            assert_eq!(vector(1), Some(vector_of("fox")));
            assert_eq!(vector(2), Some(vector_of("fox")));
            assert_eq!(vector(3), Some(vector_of("hen")));
        });
    }
}

#[tokio::test]
async fn callers_sharing_a_commit_do_not_share_an_embedding_failure() {
    let (_cluster, writer, _lengths) = embedding().await;
    writer
        .commit(vec![other_notes(), create("plain")])
        .await
        .unwrap();
    let commits = vellev_storage::GroupCommitter::spawn(Arc::new(writer), 64, 64);
    let submitted = (0..20u8).map(|n| {
        let commits = commits.clone();
        async move {
            let mutation = match n % 2 {
                0 => upsert("plain", n, 1),
                _ => other(n, "cannot be embedded", None),
            };
            commits.submit(vec![mutation]).await
        }
    });
    for (n, outcome) in join_all(submitted).await.into_iter().enumerate() {
        match n % 2 {
            0 => assert!(outcome.is_ok(), "{outcome:?}"),
            _ => assert!(
                matches!(outcome, Err(StorageError::Embedding { .. })),
                "{outcome:?}"
            ),
        }
    }
}

#[tokio::test]
async fn text_that_already_has_a_vector_is_never_embedded_again() {
    let (_cluster, writer, lengths) = embedding().await;
    writer
        .commit(vec![
            note(1, json!({ "title": "fox" }), None),
            note(2, json!({ "title": "kept" }), Some(vec![4.0, 4.0])),
        ])
        .await
        .unwrap();
    lengths.asked();

    writer
        .commit(vec![
            patch(1, json!({ "title": "hen" }), None),
            patch(1, json!({ "title": "fox" }), None),
            patch(2, json!({ "title": "changed" }), None),
            patch(2, json!({ "title": "kept" }), None),
            delete(NOTES, 1),
            note(1, json!({ "title": "fox", "n": 2 }), None),
        ])
        .await
        .unwrap();
    assert!(lengths.asked().is_empty());
    assert_eq!(vector(&writer, 1), Some(vector_of("fox")));
    assert_eq!(vector(&writer, 2), Some(vec![4.0, 4.0]));

    writer.commit(vec![delete(NOTES, 1)]).await.unwrap();
    let insert = Mutation::Insert {
        collection: name(NOTES),
        id: id(1),
        properties: json!({ "title": "fox" }).as_object().unwrap().clone(),
        vector: None,
    };
    writer.commit(vec![insert]).await.unwrap();
    assert_eq!(lengths.asked(), asked(&["fox"]));
}

#[tokio::test]
async fn vectors_made_at_commit_survive_a_flush_and_a_new_writer_with_no_embedder() {
    let (cluster, writer, _lengths) = embedding().await;
    writer
        .commit(vec![note(1, json!({ "title": "a red fox" }), None)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer
        .commit(vec![note(2, json!({ "title": "hen" }), None)])
        .await
        .unwrap();
    let expected = [Some(vector_of("a red fox")), Some(vector_of("hen"))];
    let vectors = |state: &State| {
        [1, 2].map(|object| {
            state
                .get(&name(NOTES), id(object))
                .unwrap()
                .vector
                .map(<[f32]>::to_vec)
        })
    };
    assert_eq!(cluster.open().await.partition().read(vectors), expected);

    cluster.clock.advance(TTL);
    let plain = cluster.writer().await;
    assert_eq!(plain.partition().read(vectors), expected);
    plain
        .commit(vec![patch(1, json!({ "n": 3 }), None)])
        .await
        .unwrap();
    assert_eq!(plain.partition().read(vectors), expected);
}

#[tokio::test(start_paused = true)]
async fn an_embedder_that_never_answers_fails_the_commit_and_frees_the_writer() {
    let (_cluster, writer, lengths) = embedding().await;
    lengths.fault.store(SILENT, Ordering::SeqCst);
    let waited = tokio::time::timeout(
        Duration::from_secs(3_600),
        writer.commit(vec![note(1, json!({ "title": "fox" }), None)]),
    );
    assert!(matches!(
        waited.await,
        Ok(Err(StorageError::Embedding { .. }))
    ));
    lengths.fault.store(0, Ordering::SeqCst);
    writer.flush().await.unwrap();
    writer
        .commit(vec![note(1, json!({ "title": "fox" }), None)])
        .await
        .unwrap();
    assert_eq!(vector(&writer, 1), Some(vector_of("fox")));
}

#[tokio::test]
async fn a_failing_embedder_is_asked_once_a_commit_and_only_the_batch_that_needs_it_is_refused() {
    let (_cluster, writer, lengths) = embedding().await;
    writer
        .commit(vec![note(1, json!({ "title": "fox", "n": 1 }), None)])
        .await
        .unwrap();
    lengths.fault.store(AWAY, Ordering::SeqCst);
    lengths.calls.store(0, Ordering::SeqCst);
    let retitled = |object: u8| {
        vec![note(
            object,
            json!({ "title": format!("new {object}") }),
            None,
        )]
    };
    let counted = vec![Mutation::Increment {
        collection: name(NOTES),
        id: id(1),
        property: "n".into(),
        by: 1.0,
    }];
    let batches = [vec![retitled(1), counted], (2..10).map(retitled).collect()].concat();
    let results = writer.commit_group(batches).await.unwrap();
    assert!(matches!(results[0], Err(StorageError::Embedding { .. })));
    assert_eq!(
        results[1].as_ref().unwrap().outcomes,
        [Outcome::Value(Value::Int(2))]
    );
    assert!(results[2..]
        .iter()
        .all(|result| matches!(result, Err(StorageError::Embedding { .. }))));
    assert_eq!(lengths.calls.load(Ordering::SeqCst), 1);
    assert_eq!(vector(&writer, 1), Some(vector_of("fox")));
    assert_eq!(n_of_note(&writer, 1), 2);

    lengths.calls.store(0, Ordering::SeqCst);
    let rewritten = |title: &str| vec![patch(1, json!({ "title": title }), None)];
    let chained = vec![rewritten("first"), rewritten("second"), rewritten("third")];
    let results = writer.commit_group(chained).await.unwrap();
    assert!(results
        .iter()
        .all(|result| matches!(result, Err(StorageError::Embedding { .. }))));
    assert_eq!(
        lengths.calls.load(Ordering::SeqCst),
        1,
        "one call, though three rounds"
    );
}

fn n_of_note(writer: &Writer, object: u8) -> i64 {
    writer
        .partition()
        .read(|state| n_of(state, NOTES, object).unwrap())
}

#[tokio::test]
async fn vectors_made_ahead_are_used_only_for_the_text_and_embedding_the_commit_arrives_at() {
    let (_cluster, writer, lengths) = embedding().await;
    let mutations = vec![
        note(1, json!({ "title": "fox" }), None),
        note(2, json!({ "title": "hen" }), None),
        note(3, json!({ "title": "owl" }), None),
        note(4, json!({ "title": "supplied" }), Some(vec![0.0, 1.0])),
    ];
    let wanted = writer.wanted(&mutations);
    let texts: Vec<&str> = wanted.iter().map(|(_, text)| text.as_str()).collect();
    assert_eq!(texts, ["fox", "hen", "owl"]);
    let embedding = wanted[0].0.clone();
    let other = Embedding {
        model: "another".into(),
        ..embedding.clone()
    };
    let made = [
        ((embedding.clone(), "fox".to_string()), vec![5.0, 5.0]),
        ((embedding.clone(), "hen".to_string()), vec![5.0, 5.0, 5.0]),
        ((other, "owl".to_string()), vec![6.0, 6.0]),
        ((embedding, "elk".to_string()), vec![7.0, 7.0]),
    ]
    .into();
    writer
        .commit_group_with(vec![mutations], made)
        .await
        .unwrap()[0]
        .as_ref()
        .unwrap();
    assert_eq!(lengths.asked(), asked(&["hen", "owl"]));
    assert_eq!(vector(&writer, 1), Some(vec![5.0, 5.0]));
    assert_eq!(vector(&writer, 2), Some(vector_of("hen")));
    assert_eq!(vector(&writer, 3), Some(vector_of("owl")));
    assert_eq!(vector(&writer, 4), Some(vec![0.0, 1.0]));
    assert!(writer
        .wanted(&[note(1, json!({ "title": "fox" }), None)])
        .is_empty());
}
