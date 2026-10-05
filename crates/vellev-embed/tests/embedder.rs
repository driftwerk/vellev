use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use vellev_core::{Embedding, EmbeddingProvider, HttpApi, Object, ObjectId, Value};
use vellev_embed::{document, EmbedError, Embedder, Embedders, Provider, Result, Settings};

/// Embeds each text as its length and its count of spaces, or as told.
#[derive(Debug, Default)]
struct Fake {
    asked: Mutex<Vec<Vec<String>>>,
    answer: Option<Vec<Vec<f32>>>,
}

#[async_trait]
impl Provider for Fake {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let vectors = texts
            .iter()
            .map(|text| vec![text.len() as f32, text.matches(' ').count() as f32])
            .collect();
        self.asked.lock().unwrap().push(texts);
        Ok(self.answer.clone().unwrap_or(vectors))
    }
}

fn embedding(provider: EmbeddingProvider) -> Embedding {
    Embedding {
        provider,
        model: "fake".into(),
        dimensions: 2,
        query_prefix: Some("find: ".into()),
        document_prefix: None,
        properties: vec!["title".into(), "labels".into(), "body".into()],
    }
}

fn embedder(fake: Fake) -> (Arc<Fake>, Embedder) {
    let fake = Arc::new(fake);
    (
        fake.clone(),
        Embedder::new(fake, embedding(EmbeddingProvider::Local)),
    )
}

fn texts(texts: &[&str]) -> Vec<String> {
    texts.iter().map(|text| text.to_string()).collect()
}

#[tokio::test]
async fn blank_text_has_no_vector_and_is_never_sent() {
    let (fake, embedder) = embedder(Fake::default());
    let vectors = embedder
        .documents(&texts(&["fox", "", "a hen", " \n\t"]))
        .await
        .unwrap();
    assert_eq!(
        vectors,
        [Some(vec![3.0, 0.0]), None, Some(vec![5.0, 1.0]), None]
    );
    assert_eq!(*fake.asked.lock().unwrap(), [texts(&["fox", "a hen"])]);

    assert_eq!(
        embedder.documents(&texts(&["", "  "])).await.unwrap(),
        [None, None]
    );
    assert_eq!(embedder.documents(&[]).await.unwrap(), []);
    assert_eq!(embedder.query(" ").await.unwrap(), None);
    assert_eq!(fake.asked.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn each_kind_of_text_carries_its_prefix() {
    let (fake, embedder) = embedder(Fake::default());
    embedder.documents(&texts(&["fox"])).await.unwrap();
    assert_eq!(embedder.query("fox").await.unwrap(), Some(vec![9.0, 1.0]));
    assert_eq!(
        *fake.asked.lock().unwrap(),
        [texts(&["fox"]), texts(&["find: fox"])]
    );

    let prefixed = Embedding {
        query_prefix: None,
        document_prefix: Some("store: ".into()),
        ..embedding(EmbeddingProvider::Local)
    };
    let fake = Arc::new(Fake::default());
    let embedder = Embedder::new(fake.clone(), prefixed);
    embedder.documents(&texts(&["fox"])).await.unwrap();
    embedder.query("fox").await.unwrap();
    assert_eq!(
        *fake.asked.lock().unwrap(),
        [texts(&["store: fox"]), texts(&["fox"])]
    );
}

#[tokio::test]
async fn a_repeated_query_is_embedded_once_and_the_oldest_are_forgotten() {
    let (fake, embedder) = embedder(Fake::default());
    let first = embedder.query("fox").await.unwrap();
    assert_eq!(embedder.query("fox").await.unwrap(), first);
    embedder.query("hen").await.unwrap();
    embedder.documents(&texts(&["fox"])).await.unwrap();
    assert_eq!(fake.asked.lock().unwrap().len(), 3);

    for n in 0..1_024 {
        embedder.query(&format!("query {n}")).await.unwrap();
    }
    let before = fake.asked.lock().unwrap().len();
    embedder.query("query 1023").await.unwrap();
    embedder.query("query 0").await.unwrap();
    assert_eq!(fake.asked.lock().unwrap().len(), before);
    embedder.query("fox").await.unwrap();
    assert_eq!(fake.asked.lock().unwrap().len(), before + 1);
}

#[tokio::test]
async fn an_answer_that_does_not_fit_is_refused() {
    for (answer, reason) in [
        (vec![], "0 vectors for 1 texts"),
        (
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            "2 vectors for 1 texts",
        ),
        (vec![vec![1.0, 2.0, 3.0]], "a vector of 3 dimensions, not 2"),
        (vec![vec![]], "a vector of 0 dimensions, not 2"),
        (vec![vec![1.0, f32::NAN]], "a vector that is not finite"),
        (
            vec![vec![f32::INFINITY, 0.0]],
            "a vector that is not finite",
        ),
    ] {
        let fake = Fake {
            answer: Some(answer),
            ..Fake::default()
        };
        let (_, embedder) = embedder(fake);
        let expected = EmbedError::Malformed {
            model: "fake".into(),
            reason: reason.into(),
        };
        assert_eq!(
            embedder.documents(&texts(&["fox"])).await,
            Err(expected.clone())
        );
        assert_eq!(embedder.query("fox").await, Err(expected));
    }
}

#[test]
fn an_objects_text_is_its_embedded_properties_in_the_schemas_order() {
    let object = |properties: Vec<(&str, Value)>| Object {
        id: ObjectId(uuid::Uuid::from_u128(1)),
        properties: properties
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect::<BTreeMap<_, _>>(),
        vector: None,
    };
    let embedding = embedding(EmbeddingProvider::None);
    let full = object(vec![
        ("body", Value::Text("the body".into())),
        ("labels", Value::TextArray(vec!["one".into(), "two".into()])),
        ("n", Value::Int(4)),
        ("other", Value::Text("not embedded".into())),
        ("title", Value::Text("A title".into())),
    ]);
    assert_eq!(document(&embedding, &full), "A title\none\ntwo\nthe body");
    assert_eq!(
        document(&embedding, &object(vec![("n", Value::Int(4))])),
        ""
    );
    assert_eq!(
        document(
            &embedding,
            &object(vec![("labels", Value::TextArray(vec![]))])
        ),
        ""
    );
}

#[tokio::test]
async fn collections_that_embed_alike_share_an_embedder() {
    let embedders = Embedders::new(Settings::default());
    assert!(embedders
        .embedder(&embedding(EmbeddingProvider::None))
        .unwrap()
        .is_none());

    let http = |url: &str| {
        embedding(EmbeddingProvider::Http {
            url: url.into(),
            api: HttpApi::Tei,
        })
    };
    let first = embedders
        .embedder(&http("http://one.test"))
        .unwrap()
        .unwrap();
    let again = embedders
        .embedder(&http("http://one.test"))
        .unwrap()
        .unwrap();
    let elsewhere = embedders
        .embedder(&http("http://two.test"))
        .unwrap()
        .unwrap();
    let reworded = Embedding {
        query_prefix: None,
        ..http("http://one.test")
    };
    let reworded = embedders.embedder(&reworded).unwrap().unwrap();
    assert!(Arc::ptr_eq(&first, &again));
    assert!(!Arc::ptr_eq(&first, &elsewhere));
    assert!(!Arc::ptr_eq(&first, &reworded));
    assert_eq!(first.embedding(), &http("http://one.test"));

    let bedrock = embedding(EmbeddingProvider::Bedrock {
        region: "eu-west-2".into(),
    });
    assert!(embedders.embedder(&bedrock).unwrap().is_some());
}

#[cfg(feature = "local")]
#[test]
fn a_local_model_must_be_one_that_can_be_run_at_its_own_size() {
    let embedders = Embedders::new(Settings::default());
    let local = |model: &str, dimensions| Embedding {
        model: model.into(),
        dimensions,
        ..embedding(EmbeddingProvider::Local)
    };
    for model in [
        "BAAI/bge-base-en-v1.5",
        "Xenova/bge-base-en-v1.5",
        "bge-base-en-v1.5",
    ] {
        assert!(
            embedders.embedder(&local(model, 768)).unwrap().is_some(),
            "{model}"
        );
    }
    assert_eq!(
        embedders
            .embedder(&local("BAAI/bge-base-en-v1.5", 1024))
            .unwrap_err(),
        EmbedError::Unsupported("BAAI/bge-base-en-v1.5 has 768 dimensions, not 1024".into())
    );
    assert_eq!(
        embedders
            .embedder(&local("acme/unheard-of", 768))
            .unwrap_err(),
        EmbedError::Unsupported("acme/unheard-of cannot be run locally".into())
    );
}
