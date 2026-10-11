//! A model that cannot be used is found so once, and said to be so once,
//! however many collections embed with it and however each words its
//! texts: and it takes no other model with it. The test is alone in its
//! file, and so in its process, for the reason the one in
//! `fault_logged_once.rs` is: what is logged is listened for on one
//! thread.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};
use tracing_subscriber::layer::SubscriberExt;
use vellev_core::{Embedding, EmbeddingProvider, HttpApi, ModelFault};
use vellev_embed::{EmbedError, Embedders, Settings};

/// What is logged of a model that cannot be used, while it is listened
/// for.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<String>>>);

impl Logged {
    fn faults(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Logged {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        struct Message(String);
        impl tracing::field::Visit for Message {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = format!("{value:?}");
                }
            }
        }
        let mut message = Message(String::new());
        event.record(&mut message);
        let error = *event.metadata().level() == tracing::Level::ERROR;
        if error && message.0.contains("cannot be used") {
            self.0.lock().unwrap().push(message.0);
        }
    }
}

/// The texts an endpoint was sent, each with the model it was asked of.
type Sent = Arc<Mutex<Vec<(String, String)>>>;

/// An endpoint that speaks as OpenAI does. It has no model called
/// "gone", and its one key is `sk-right-0123456789abcdef`. Every vector it makes has two
/// numbers: how long the text is, and how many spaces are in it.
async fn endpoint(sent: Sent) -> String {
    let embed = |State(sent): State<Sent>, headers: HeaderMap, Json(body): Json<Value>| async move {
        let model = body["model"].as_str().unwrap().to_string();
        let texts: Vec<&str> = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| text.as_str().unwrap())
            .collect();
        let mut sent = sent.lock().unwrap();
        sent.extend(texts.iter().map(|text| (model.clone(), text.to_string())));
        let refusal = |status, code: &str| {
            let error = json!({ "error": { "message": "no", "type": "t", "code": code } });
            Err((status, Json(error)))
        };
        let key = headers
            .get("authorization")
            .map(|key| key.to_str().unwrap());
        if key != Some("Bearer sk-right-0123456789abcdef") {
            return refusal(StatusCode::UNAUTHORIZED, "invalid_api_key");
        }
        if model == "gone" {
            return refusal(StatusCode::NOT_FOUND, "model_not_found");
        }
        let data: Vec<Value> = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                let vector = [text.len(), text.matches(' ').count()];
                json!({ "index": index, "embedding": vector })
            })
            .collect();
        Ok(Json(json!({ "data": data })))
    };
    let router = Router::new()
        .route("/v1/embeddings", post(embed))
        .with_state(sent);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

/// The ways collections word their texts for one model, which change
/// nothing of whether the model can be used.
fn worded(embedding: &Embedding) -> Vec<Embedding> {
    let with = |query: Option<&str>, document: Option<&str>, properties: &[&str]| Embedding {
        query_prefix: query.map(String::from),
        document_prefix: document.map(String::from),
        properties: properties.iter().map(|name| name.to_string()).collect(),
        ..embedding.clone()
    };
    vec![
        with(None, None, &[]),
        with(Some("find: "), None, &["title"]),
        with(Some("find: "), Some("store: "), &["title", "body"]),
        with(None, Some("passage: "), &["body"]),
    ]
}

#[tokio::test]
async fn a_model_that_cannot_be_used_is_found_so_once_and_takes_no_other_with_it() {
    let logged = Logged::default();
    let _listening =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(logged.clone()));
    let sent = Sent::default();
    let url = endpoint(sent.clone()).await;
    let asked = || sent.lock().unwrap().len();
    let through = |model: &str, dimensions: u32| Embedding {
        provider: EmbeddingProvider::Http {
            url: url.clone(),
            api: HttpApi::Openai,
        },
        model: model.into(),
        dimensions,
        query_prefix: None,
        document_prefix: None,
        properties: Vec::new(),
    };
    // The fault stands for as long as one can, which is longer than any
    // run of this could take.
    let node = |key: &str| {
        Embedders::new(Settings {
            keys: [(url.clone(), key.to_string())].into(),
            fault_stands: Duration::from_secs(60 * 60),
            ..Settings::default()
        })
    };
    let fault = |refused: &EmbedError| refused.lasting();

    // A key the endpoint has revoked, and four collections that embed
    // with one model through it, each wording its texts its own way.
    // The endpoint is asked once, and the fault is said once.
    let embedders = node("sk-revoked-0123456789");
    for (nth, embedding) in worded(&through("m", 2)).iter().enumerate() {
        let embedder = embedders.embedder(embedding).unwrap().unwrap();
        let query = embedder.query(&format!("query {nth}")).await;
        assert_eq!(fault(&query.unwrap_err()), Some(ModelFault::Setup));
        let documents = embedder.documents(&[format!("document {nth}")]).await;
        assert_eq!(fault(&documents.unwrap_err()), Some(ModelFault::Setup));
    }
    assert_eq!((asked(), logged.faults()), (1, 1));
    assert_eq!(embedders.held(), 1);
    // Another model through it is another model, found out for itself.
    for embedding in worded(&through("n", 2)) {
        let embedder = embedders.embedder(&embedding).unwrap().unwrap();
        assert!(embedder.query("fox").await.is_err());
    }
    assert_eq!((asked(), logged.faults()), (2, 2));

    // A node with the right key. The endpoint has no such model as one
    // collection names: that is found once, however many embed with it.
    let embedders = node("sk-right-0123456789abcdef");
    for embedding in worded(&through("gone", 2)) {
        let embedder = embedders.embedder(&embedding).unwrap().unwrap();
        let refused = embedder.query("fox").await.unwrap_err();
        assert_eq!(fault(&refused), Some(ModelFault::Provider));
    }
    assert_eq!((asked(), logged.faults()), (3, 3));

    // It does not keep a model the endpoint does have from being used,
    // by all who embed with that one: nor does a collection that names
    // that model at a size it does not have, which is a fault of its
    // own, found once.
    let used = worded(&through("m", 2));
    let wrong_size = worded(&through("m", 3));
    for round in 0..2 {
        for embedding in &wrong_size {
            let embedder = embedders.embedder(embedding).unwrap().unwrap();
            let refused = embedder.query("fox").await.unwrap_err();
            assert_eq!(fault(&refused), Some(ModelFault::Provider), "{refused}");
        }
        for (nth, embedding) in used.iter().enumerate() {
            let embedder = embedders.embedder(embedding).unwrap().unwrap();
            let text = format!("round {round} of {nth}");
            let vector = embedder.query(&text).await.unwrap().unwrap();
            let prefix = embedding.query_prefix.as_deref().unwrap_or_default();
            assert_eq!(vector[0] as usize, prefix.len() + text.len());
        }
    }
    // Once for the wrong size, and once for each query answered.
    assert_eq!((asked(), logged.faults()), (3 + 1 + 2 * 4, 4));
    assert_eq!(embedders.held(), 3);

    // Those who embed with a model share the vectors it has made for
    // queries, by what it was sent to make them: the text with the same
    // words before it is made once, whichever properties are embedded,
    // and with other words before it is another text.
    let before = asked();
    for embedding in &used {
        let embedder = embedders.embedder(embedding).unwrap().unwrap();
        let vector = embedder.query("a fox").await.unwrap().unwrap();
        let prefix = embedding.query_prefix.as_deref().unwrap_or_default();
        let spaces = prefix.matches(' ').count() + 1;
        assert_eq!(vector, [(prefix.len() + 5) as f32, spaces as f32]);
    }
    let made: Vec<(String, String)> = sent.lock().unwrap()[before..].to_vec();
    let made: Vec<&str> = made.iter().map(|(_, text)| text.as_str()).collect();
    assert_eq!(made, ["a fox", "find: a fox"]);
}
