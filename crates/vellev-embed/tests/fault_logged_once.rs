//! How often a fault that stands is logged. The test is alone in its
//! file, and so in its process: what is logged is listened for on one
//! thread, and tests beside it that log from the same place would decide
//! between them whether that place is listened to at all.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tracing_subscriber::layer::SubscriberExt;
use vellev_core::{Embedding, EmbeddingProvider, ModelFault};
use vellev_embed::{EmbedError, Embedder, Provider, Purpose, Result, FAULT_STANDS};

/// A model that refuses every call for good, after a moment: a longer
/// one for each call that is out at the same time. Until it is told to
/// refuse, it answers; and it always answers a text that begins "ok".
#[derive(Debug, Default)]
struct Refusing {
    calls: AtomicUsize,
    out: AtomicUsize,
    answering: AtomicBool,
}

#[async_trait]
impl Provider for Refusing {
    async fn embed(&self, texts: Vec<String>, _: Purpose) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let out = self.out.fetch_add(1, Ordering::SeqCst) as u32 + 1;
        tokio::time::sleep(Duration::from_millis(50) * out).await;
        self.out.fetch_sub(1, Ordering::SeqCst);
        let ok = texts.iter().all(|text| text.starts_with("ok"));
        if ok || self.answering.load(Ordering::SeqCst) {
            return Ok(vec![vec![1.0, 2.0]; texts.len()]);
        }
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
    // Should anything here wait for ever, it fails rather than hangs.
    tokio::time::timeout(Duration::from_secs(24 * 60 * 60), logged_once())
        .await
        .expect("nothing waits on anything");
}

async fn logged_once() {
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
    let ms = Duration::from_millis;

    // Eight at once, with nothing known of the model: each asks, the
    // first hearing of the fault 50 ms in and the last 400, and it is said
    // once. It stands from when the first heard, not the last.
    let started = tokio::time::Instant::now();
    eight("first").await;
    assert_eq!(started.elapsed(), ms(400));
    assert_eq!(so_far(), (8, 1));
    tokio::time::advance(FAULT_STANDS / 2).await;
    eight("meanwhile").await;
    assert_eq!(so_far(), (8, 1), "run into while it stands: not asked");
    tokio::time::advance(FAULT_STANDS / 2 - ms(350) - ms(1)).await;
    eight("to the last moment").await;
    assert_eq!(so_far(), (8, 1));
    tokio::time::advance(ms(1)).await;
    eight("no longer").await;
    assert_eq!(so_far(), (16, 2), "said again, once, in the time after");

    // A model that has been answering and then refuses: likewise.
    model.answering.store(true, Ordering::SeqCst);
    tokio::time::advance(FAULT_STANDS).await;
    assert!(embedder.query("it answers").await.is_ok());
    model.answering.store(false, Ordering::SeqCst);
    eight("all at once").await;
    assert_eq!(so_far(), (16 + 1 + 8, 3));
    eight("meanwhile").await;
    assert_eq!(so_far(), (16 + 1 + 8, 3));

    // A model that answers one call and refuses the next, asked by many
    // at once and over and over: each answer lifts the fault, so it goes
    // on being asked, by every call. And the fault is still said once in
    // each time one stands, and no more.
    tokio::time::advance(FAULT_STANDS * 2).await;
    let (calls, said) = so_far();
    let started = tokio::time::Instant::now();
    let mut asked = 0;
    for time in 0..4 {
        tokio::time::sleep_until(started + FAULT_STANDS * time).await;
        for _ in 0..3 {
            let texts: Vec<String> = (asked..asked + 64)
                .map(|n| format!("{} {n}", ["no", "ok"][n % 2]))
                .collect();
            asked += texts.len();
            let answers =
                futures::future::join_all(texts.iter().map(|text| embedder.query(text))).await;
            let refused = answers.iter().filter(|answer| answer.is_err()).count();
            assert_eq!(refused, 32, "every other call");
            assert_eq!(so_far(), (calls + asked, said + time as usize + 1));
        }
        assert!(started.elapsed() < FAULT_STANDS * (time + 1));
    }
}
