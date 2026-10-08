//! How often a fault that stands is logged. The test is alone in its
//! file, and so in its process: what is logged is listened for on one
//! thread, and tests beside it that log from the same place would decide
//! between them whether that place is listened to at all.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tracing_subscriber::layer::SubscriberExt;
use vellev_core::{Embedding, EmbeddingProvider, ModelFault};
use vellev_embed::{EmbedError, Embedder, Provider, Purpose, Result, FAULT_STANDS};

/// A model that refuses every call for good, after a moment.
#[derive(Debug, Default)]
struct Refusing {
    calls: AtomicUsize,
}

#[async_trait]
impl Provider for Refusing {
    async fn embed(&self, _: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(50)).await;
        Err(EmbedError::Unusable {
            model: "fake".into(),
            reason: "401 Unauthorized: no such key".into(),
            fault: ModelFault::Setup,
        })
    }
}

/// What is logged while it is listened for, with the level of each.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<(tracing::Level, String)>>>);

impl Logged {
    fn errors(&self) -> usize {
        let logged = self.0.lock().unwrap();
        logged
            .iter()
            .filter(|(level, said)| {
                *level == tracing::Level::ERROR && said.contains("cannot be used")
            })
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

#[tokio::test(start_paused = true)]
async fn a_fault_met_by_many_at_once_is_logged_once_in_each_time_it_stands() {
    let logged = Logged::default();
    let _listening =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(logged.clone()));
    let model = Arc::new(Refusing::default());
    let embedding = Embedding {
        provider: EmbeddingProvider::Local,
        model: "fake".into(),
        dimensions: 2,
        query_prefix: None,
        document_prefix: None,
        properties: vec!["title".into()],
    };
    let embedder = Embedder::new(model.clone(), embedding);
    let eight = |word: &str| {
        let texts: Vec<String> = (0..8).map(|n| format!("{word} {n}")).collect();
        let embedder = &embedder;
        async move {
            let asked = texts.iter().map(|text| embedder.query(text));
            let answers = futures::future::join_all(asked).await;
            assert!(answers.iter().all(Result::is_err));
        }
    };
    let so_far = || (model.calls.load(Ordering::SeqCst), logged.errors());
    eight("first").await;
    assert_eq!(so_far(), (1, 1), "eight at once: asked once, said once");
    for _ in 0..3 {
        tokio::time::advance(FAULT_STANDS / 4).await;
        eight("meanwhile").await;
        assert_eq!(so_far(), (1, 1), "run into while it stands");
    }
    // It stands from when it was met, not from when it was last run into.
    tokio::time::advance(FAULT_STANDS / 4).await;
    eight("again").await;
    assert_eq!(so_far(), (2, 2));
    tokio::time::advance(FAULT_STANDS).await;
    eight("and again").await;
    assert_eq!(so_far(), (3, 3));
}
