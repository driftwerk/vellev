//! A node's sweeps (resident accounting PR 1, delta reviews of #165): one
//! at a time, so that no sweep holds what another must find held by
//! nothing else; and those asked for as tenants are used coalesced, so
//! that a sweep held up, as by a flush that hangs, never has a sweep a
//! request queued behind it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, RequestId, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Node, Settings};
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator, Point};
use vellev_storage::{Mutation, State};

fn tenant(name: &str) -> TenantId {
    TenantId::new(name).unwrap()
}

fn notes() -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: CollectionName::new("notes").unwrap(),
        properties: vec![Property::new("title", DataType::Text).searchable()],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "stand-in".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

fn note(n: u128) -> Mutation {
    Mutation::Upsert {
        collection: CollectionName::new("notes").unwrap(),
        id: ObjectId(uuid::Uuid::from_u128(n)),
        properties: serde_json::json!({ "title": format!("note {n}") })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![1.0, n as f32]),
        only_if_version: None,
    }
}

/// A node, as `set` sets it up, on a coordinator of its own and `blobs`.
fn node_on(blobs: Arc<dyn Blobs>, set: impl FnOnce(&mut Settings)) -> Node {
    let coordinator: Arc<dyn Coordinator> = Arc::new(MemoryCoordinator::default());
    let mut settings = Settings {
        warm: None,
        ..Settings::new(NodeId::generate("node"))
    };
    set(&mut settings);
    Node::new(
        coordinator,
        blobs,
        Arc::new(Embedders::new(EmbedSettings::default())),
        settings,
    )
}

fn memory() -> Arc<dyn Blobs> {
    Arc::new(ObjectBlobs::new(Arc::new(
        object_store::memory::InMemory::new(),
    )))
}

/// Wait until `holds`, at most `limit` seconds.
async fn until(limit: u64, what: &str, holds: impl Fn() -> bool) {
    tokio::time::timeout(Duration::from_secs(limit), async {
        while !holds() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what} within {limit} s"));
}

/// A store whose puts, once it is armed, wait until it is disarmed: a
/// flush that hangs on the store.
#[derive(Debug)]
struct HeldPuts {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    held: AtomicBool,
}

#[async_trait::async_trait]
impl Blobs for HeldPuts {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        while self.armed.load(Ordering::SeqCst) {
            self.held.store(true, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(5)).await;
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

/// Delta review of #165, M1: every use of a tenant over the budget asks for
/// a sweep, and a sweep can hang in the flush of a tenant it lets go. Those
/// asked for meanwhile are coalesced: while one sweep is held in a flush,
/// a thousand uses of a tenant leave at most one more sweep to run once it
/// ends, not a thousand queued behind it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeps_asked_for_while_one_runs_are_coalesced_into_one() {
    let store = Arc::new(HeldPuts {
        inner: memory(),
        armed: AtomicBool::new(false),
        held: AtomicBool::new(false),
    });
    let node = node_on(store.clone(), |settings| {
        settings.memory = Some(1);
        // Flushed only as it is let go; never let go for being idle.
        settings.flush_rows = 1_000_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    for name in ["slow", "used"] {
        node.create_tenant(&tenant(name)).await.unwrap();
        let held = node.tenant(&tenant(name)).await.unwrap();
        held.write(vec![notes(), note(1)]).await.unwrap();
    }
    // `slow` unused for longer than any tenant is kept regardless; `used`
    // used again just before each sweep, and so kept.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    store.armed.store(true, Ordering::SeqCst);
    let before = node.sweeps();
    // A sweep that lets `slow` go, and hangs in its flush.
    drop(node.tenant(&tenant("used")).await.unwrap());
    until(
        5,
        "the sweep never came to flush the tenant it lets go",
        || store.held.load(Ordering::SeqCst),
    )
    .await;
    assert_eq!(node.sweeps(), before + 1, "one sweep, held in the flush");
    for _ in 0..1_000 {
        drop(node.tenant(&tenant("used")).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(node.sweeps(), before + 1, "none began while one ran");
    store.armed.store(false, Ordering::SeqCst);
    // Time enough for the held sweep to end, and for a thousand queued
    // behind it to have run.
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(
        node.sweeps(),
        before + 2,
        "the sweep held, and one more for all those asked for meanwhile"
    );
    let slow = node.tenant(&tenant("slow")).await.unwrap();
    assert_eq!(slow.read(State::unflushed), 0, "flushed as it was let go");
}

/// Delta review of #165, item 3: sweeps never overlap. One sweep is held
/// as it is about to take tenants out; another asked for meanwhile does not
/// begin until it is let go.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sweep_does_not_begin_while_another_runs() {
    let node = node_on(memory(), |_| {});
    let hold = node.holding_sweeps();
    let first = tokio::spawn({
        let node = node.clone();
        async move { node.sweep_now().await }
    });
    until(5, "the first sweep never reached the hold", || {
        hold.reached() == 1
    })
    .await;
    let begun = node.sweeps();
    let second = tokio::spawn({
        let node = node.clone();
        async move { node.sweep_now().await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(node.sweeps(), begun, "a second sweep began while one ran");
    assert_eq!(hold.reached(), 1);
    hold.release();
    tokio::time::timeout(Duration::from_secs(5), async {
        first.await.unwrap();
        second.await.unwrap();
    })
    .await
    .expect("both sweeps end once let go");
    assert_eq!(node.sweeps(), begun + 1, "the second, once the first ended");
}

/// Final review of #165, L1: the sweeps a tenant's use asks for are spaced.
/// Each use over the budget asks for one, and coalesced, with nothing to
/// let go, they would run back to back for as long as uses come, each
/// taking the node's lock twice. Asked for without let-up for two seconds,
/// they run no more often than `SWEEPS_SPACED` allows; the first, asked for
/// after a quiet spell, begins at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeps_asked_for_by_use_are_spaced() {
    use vellev_node::SWEEPS_SPACED;
    let node = node_on(memory(), |settings| {
        // A budget, so that each use asks for a sweep; nothing over it.
        settings.memory = Some(1 << 40);
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    node.create_tenant(&tenant("used")).await.unwrap();
    let used = node.tenant(&tenant("used")).await.unwrap();
    used.write(vec![notes(), note(1)]).await.unwrap();
    drop(used);
    // A quiet spell longer than the spacing.
    tokio::time::sleep(SWEEPS_SPACED * 2).await;
    let before = node.sweeps();
    let asked = std::time::Instant::now();
    drop(node.tenant(&tenant("used")).await.unwrap());
    until(1, "the sweep asked for never began", || {
        node.sweeps() > before
    })
    .await;
    assert!(
        asked.elapsed() < SWEEPS_SPACED / 2,
        "the first sweep after a quiet spell waited {:?}",
        asked.elapsed()
    );
    let streaming = Duration::from_secs(2);
    let began = std::time::Instant::now();
    let mut uses = 0;
    while began.elapsed() < streaming {
        drop(node.tenant(&tenant("used")).await.unwrap());
        uses += 1;
    }
    let ran = node.sweeps() - before;
    let most = (streaming.as_millis() / SWEEPS_SPACED.as_millis()) as u64 + 2;
    assert!(uses > 1_000, "only {uses} uses");
    assert!(
        ran <= most,
        "{ran} sweeps for {uses} uses in {streaming:?}, at most {most} spaced {SWEEPS_SPACED:?}"
    );
    assert!(ran >= 2, "the sweeps asked for ran: {ran}");
}

/// Review of #167, L1: a sweep waiting out the spacing does so before it
/// takes the lock, so a sweep the lock is wanted for meanwhile, as a test's
/// `sweep_now`, never waits on the spacing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sweep_asked_for_now_never_waits_on_the_spacing() {
    use vellev_node::SWEEPS_SPACED;
    let node = node_on(memory(), |settings| {
        settings.memory = Some(1 << 40);
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    node.create_tenant(&tenant("used")).await.unwrap();
    let used = node.tenant(&tenant("used")).await.unwrap();
    used.write(vec![notes(), note(1)]).await.unwrap();
    drop(used);
    tokio::time::sleep(SWEEPS_SPACED * 2).await;
    let before = node.sweeps();
    // A sweep a use asks for, at once after a quiet spell, and ended.
    drop(node.tenant(&tenant("used")).await.unwrap());
    until(1, "the sweep asked for never began", || {
        node.sweeps() > before
    })
    .await;
    tokio::time::sleep(SWEEPS_SPACED / 5).await;
    // Another asked for, which waits out the rest of the spacing.
    drop(node.tenant(&tenant("used")).await.unwrap());
    tokio::time::sleep(Duration::from_millis(10)).await;
    let asked = std::time::Instant::now();
    node.sweep_now().await;
    assert!(
        asked.elapsed() < SWEEPS_SPACED / 2,
        "a sweep asked for now waited {:?} on the spacing",
        asked.elapsed()
    );
}

/// M7-R7: a restore holds its tenant, as a request does, with no request
/// waiting for it and no merge, however far over the budget the node is.
/// The sweeps that find it so leave it be until the restore ends, and the
/// restore lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_a_restore_holds_is_not_let_go_until_the_restore_ends() {
    let store = Arc::new(HeldPuts {
        inner: memory(),
        armed: AtomicBool::new(false),
        held: AtomicBool::new(false),
    });
    let node = node_on(store.clone(), |settings| {
        settings.memory = Some(1);
        // Flushed only as it is let go; never let go for being idle.
        settings.flush_rows = 1_000_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let acme = tenant("restored");
    node.create_tenant(&acme).await.unwrap();
    let window = Some(Duration::from_secs(60 * 60));
    assert!(node.set_restore_window(&acme, window).await.unwrap());
    let held = node.tenant(&acme).await.unwrap();
    let kept = held.write(vec![notes(), note(1)]).await.unwrap();
    held.write(vec![note(2)]).await.unwrap();
    drop(held);

    store.armed.store(true, Ordering::SeqCst);
    let point = Point::Position(kept.position);
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move { node.restore(&acme, point, RequestId::generate()).await }
    });
    until(5, "the restore never came to write", || {
        store.held.load(Ordering::SeqCst)
    })
    .await;
    // Whoever asked stops waiting: the restore goes on, and only it holds
    // the tenant. Unused for longer than any tenant is kept regardless.
    restoring.abort();
    let _ = restoring.await;
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    let before = node.tenants_kept_by_sweeps();
    // The census is taken as a sweep begins: a second one finds what the
    // first let go.
    node.sweep_now().await;
    node.sweep_now().await;
    assert_eq!(node.census().holding.writers, 1, "let go mid-restore");
    assert!(node.tenants_kept_by_sweeps() > before, "no sweep looked");
    assert_eq!(node.opens().writers, 1, "opened afresh mid-restore");
    store.armed.store(false, Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(10), async {
        while node.resident() > 0 {
            node.sweep_now().await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("the tenant was never let go once the restore ended");
    let opened = node.tenant(&acme).await.unwrap();
    let collection = CollectionName::new("notes").unwrap();
    assert_eq!(
        opened.read(|state| state.count(&collection)),
        1,
        "the restore did not land"
    );
}
