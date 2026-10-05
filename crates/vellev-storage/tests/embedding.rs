mod common;

use common::*;
use vellev_storage::{Outcome, Vectorize};

const NOTES: &str = "notes";

/// Embeds a text as its length and its number of words, and remembers what
/// it was asked for.
#[derive(Debug, Default)]
struct Lengths {
    asked: Mutex<Vec<Vec<String>>>,
    failing: AtomicBool,
    dimensions: AtomicUsize,
}

fn vector_of(text: &str) -> Vec<f32> {
    vec![text.len() as f32, text.split_whitespace().count() as f32]
}

#[async_trait]
impl Vectorize for Lengths {
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> Result<Vec<Option<Vec<f32>>>, String> {
        assert_eq!(embedding.model, "lengths");
        if self.failing.load(Ordering::SeqCst) {
            return Err("the model is away".into());
        }
        let extra = self.dimensions.load(Ordering::SeqCst);
        let vectors = texts
            .iter()
            .map(|text| Some([vector_of(text), vec![0.0; extra]].concat()))
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
    writer
        .partition()
        .read(|state| state.get(&name(NOTES), id(object)).unwrap().vector.clone())
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
    let followed = follower
        .partition()
        .read(|state| state.get(&name(NOTES), id(1)).unwrap().vector.clone());
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
        ],
        vec![
            note(2, json!({ "title": "written then deleted" }), None),
            delete(NOTES, 2),
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
            let vector = |object: u8| state.get(&name(NOTES), id(object)).unwrap().vector.clone();
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

    lengths.failing.store(true, Ordering::SeqCst);
    let failed = writer.commit(fox()).await.unwrap_err();
    assert!(
        matches!(&failed, StorageError::Embedding { model, reason } if model == "lengths" && reason == "the model is away"),
        "{failed}"
    );
    lengths.failing.store(false, Ordering::SeqCst);

    lengths.dimensions.store(1, Ordering::SeqCst);
    assert!(matches!(
        writer.commit(fox()).await,
        Err(StorageError::Embedding { .. })
    ));
    lengths.dimensions.store(0, Ordering::SeqCst);

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
    writer
        .commit(vec![
            embedding_notes(),
            create("plain"),
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
        matches!(refused, StorageError::Embedding { .. }),
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
    let kept = writer
        .partition()
        .read(|state| state.get(&name("plain"), id(1)).unwrap().vector.clone());
    assert_eq!(kept, Some(vec![1.0, 0.5]));
}
