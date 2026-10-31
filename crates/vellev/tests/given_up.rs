//! A tenant given up as its maintenance panicked. Alone in its file, and so
//! in its process, as what is logged is listened for on one thread (see
//! `logging.rs`).

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::http::Method;
use common::*;
use serde_json::json;
use vellev_node::Settings;
use vellev_storage::blobs::{Blobs, Listed};

/// A store whose next write of a segment, once armed, panics.
#[derive(Debug)]
struct Panicking {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
}

#[async_trait::async_trait]
impl Blobs for Panicking {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        if key.ends_with(".segment") && self.armed.swap(false, Ordering::SeqCst) {
            panic!("a write panics, as a test asks");
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// What is logged at error while it is listened for.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<String>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Logged {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        struct Fields(String);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0.push_str(&format!(" {}={value:?}", field.name()));
            }
        }
        if *event.metadata().level() == tracing::Level::ERROR {
            let mut fields = Fields(String::new());
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }
}

/// A merge lost to a panic is in neither count of merges stopped, and with
/// no panic hook all a panic leaves is a line on stderr that names no
/// tenant. So what gives a tenant up for one says so at error, naming it,
/// and counts it, on /metrics from the start.
#[tokio::test]
async fn a_tenant_given_up_for_a_panic_is_logged_by_name_and_counted() {
    use tracing_subscriber::layer::SubscriberExt;
    let logged = Logged::default();
    let _listening =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(logged.clone()));
    let store = Arc::new(Panicking {
        inner: memory(),
        armed: AtomicBool::new(false),
    });
    let served = Cluster::new()
        .through(store.clone())
        .serve_unstarted_with(|settings| Settings {
            flush_rows: 3,
            ..settings
        })
        .await;
    let given_up = || {
        served.app.metrics().encoded().lines().find_map(|line| {
            line.strip_prefix("vellev_tenants_given_up_total ")?
                .parse()
                .ok()
        })
    };
    assert_eq!(given_up(), Some(0.0), "there from the start");
    served.with_notes(supplied()).await;
    store.armed.store(true, Ordering::SeqCst);
    for n in 1..=3 {
        let note = json!({ "properties": { "title": "note" }, "vector": [1.0, n as f64] });
        served.ok(Method::PUT, &object(n), Some(note)).await;
    }
    tokio::time::timeout(Duration::from_secs(20), async {
        while given_up() != Some(1.0) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the tenant given up was counted");
    let said = logged.0.lock().unwrap().clone();
    assert!(
        said.iter()
            .any(|line| line.contains("acme") && line.contains("given up")),
        "{said:?}"
    );
    served.stop().await;
}
