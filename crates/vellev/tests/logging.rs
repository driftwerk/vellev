//! What a node logs. The test is alone in its file, and so in its
//! process: what is logged is listened for on one thread, and tests beside
//! it that log from the same places would decide between them whether
//! those places are listened to at all.

mod common;

use common::*;
use reqwest::{Method, StatusCode};
use serde_json::json;

/// What is logged while it is listened for, with the level of each.
#[derive(Clone, Default)]
struct Logged(std::sync::Arc<std::sync::Mutex<Vec<(tracing::Level, String)>>>);

impl Logged {
    /// How many times something holding `words` was logged at `level`.
    fn times(&self, level: tracing::Level, words: &str) -> usize {
        let logged = self.0.lock().unwrap();
        logged
            .iter()
            .filter(|(at, said)| *at == level && said.contains(words))
            .count()
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
        let level = *event.metadata().level();
        self.0.lock().unwrap().push((level, message.0));
    }
}

#[tokio::test]
async fn a_fault_that_stands_is_logged_once_however_many_searches_fall_back_for_it() {
    // Should anything here wait for ever, it fails rather than hangs.
    tokio::time::timeout(std::time::Duration::from_secs(120), logged_once())
        .await
        .expect("nothing waits on anything");
}

async fn logged_once() {
    use tracing_subscriber::layer::SubscriberExt;
    let logged = Logged::default();
    let _listening =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(logged.clone()));

    let url = keyed_embedding_server("sk-right").await;
    // The fault stands for as long as one can, which is longer than any
    // run of this could take: how often it is said does not turn on how
    // fast this machine is.
    let embedding = vellev_embed::Settings {
        keys: [(url.clone(), "sk-wrong".to_string())].into(),
        fault_stands: std::time::Duration::from_secs(60 * 60),
        ..Default::default()
    };
    let served = Cluster::new().serve_embedding_with(&url, embedding).await;
    let embedding = json!({
        "provider": "http",
        "api": "openai",
        "url": url,
        "model": "stand-in",
        "dimensions": 2,
        "properties": ["title"],
    });
    served.with_notes(embedding).await;
    for _ in 0..3 {
        let by_words = served
            .ok(
                Method::POST,
                &format!("{NOTES}/hybrid"),
                Some(json!({ "text": "fox", "limit": 1 })),
            )
            .await;
        assert_eq!(by_words["degraded_by"], "model_unavailable");
        let nearest = json!({ "text": "fox", "limit": 1 });
        let refused = served.post(&format!("{NOTES}/nearest"), nearest).await;
        assert_eq!(
            (
                refused.0,
                &refused.1["error"]["code"],
                &refused.1["error"]["reason"]
            ),
            (
                StatusCode::NOT_IMPLEMENTED,
                &json!("model_unavailable"),
                &json!("setup")
            )
        );
    }
    assert_eq!(
        logged.times(tracing::Level::ERROR, "the embedding model cannot be used"),
        1,
        "said once while it stands"
    );
    let per_request = logged.times(tracing::Level::WARN, "searching by words alone")
        + logged.times(tracing::Level::WARN, "embedding failed")
        + logged.times(tracing::Level::ERROR, "searching by words alone");
    assert_eq!(per_request, 0, "and not again for each request");
    assert_eq!(
        logged.times(tracing::Level::DEBUG, "searching by words alone"),
        3,
        "but for whoever follows one request"
    );
}
