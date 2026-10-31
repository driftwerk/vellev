//! A node's sweeps (resident accounting PR 1, delta reviews of #165): one
//! at a time, so that no sweep holds what another must find held by
//! nothing else; and those asked for as tenants are used coalesced, so
//! that a sweep held up, as by a flush that hangs, never has a sweep a
//! request queued behind it.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, RequestId, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Node, Settings, Tenant, LET_GO_FLUSHES_AT_ONCE, LET_GO_FLUSH_WITHIN};
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator, Point};
use vellev_storage::timings::{Holder, Timed};
use vellev_storage::{Mutation, State, StorageError, Writer};

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

/// A store that holds flushes up tenant by tenant: the next put of a tenant
/// named by [`Self::hold`] waits as long as that says, and the vectors of
/// one named by [`Self::fail_vectors`] fail to go up.
#[derive(Debug)]
struct TenantPuts {
    inner: Arc<dyn Blobs>,
    held: Mutex<HashMap<String, Duration>>,
    failing: Mutex<HashSet<String>>,
    /// The tenants whose flushes have come to a put held.
    reached: Mutex<HashSet<String>>,
    /// How many puts held wait now, and the most that ever did at once.
    waiting: AtomicUsize,
    most_waiting: AtomicUsize,
}

impl TenantPuts {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: memory(),
            held: Mutex::default(),
            failing: Mutex::default(),
            reached: Mutex::default(),
            waiting: AtomicUsize::new(0),
            most_waiting: AtomicUsize::new(0),
        })
    }

    fn hold(&self, tenant: &str, held: Duration) {
        self.held.lock().unwrap().insert(tenant.into(), held);
    }

    fn fail_vectors(&self, tenant: &str, failing: bool) {
        let mut tenants = self.failing.lock().unwrap();
        match failing {
            true => tenants.insert(tenant.into()),
            false => tenants.remove(tenant),
        };
    }

    fn reached(&self) -> usize {
        self.reached.lock().unwrap().len()
    }
}

#[async_trait::async_trait]
impl Blobs for TenantPuts {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        let tenant = key.split('/').next().unwrap_or_default();
        if key.ends_with(".vectors") && self.failing.lock().unwrap().contains(tenant) {
            return Err(StorageError::Background("no room for vectors".into()));
        }
        let held = self.held.lock().unwrap().remove(tenant);
        if let Some(held) = held {
            self.reached.lock().unwrap().insert(tenant.into());
            let waiting = self.waiting.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_waiting.fetch_max(waiting, Ordering::SeqCst);
            tokio::time::sleep(held).await;
            self.waiting.fetch_sub(1, Ordering::SeqCst);
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

/// A node over its memory budget from its first row, whose tenants are
/// flushed only as they are let go and let go only for the budget.
fn over_budget(blobs: Arc<dyn Blobs>) -> Node {
    node_on(blobs, |settings| {
        settings.memory = Some(1);
        settings.flush_rows = 1_000_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    })
}

/// `name` made on `node` with `rows` notes in memory, and no longer used:
/// its writer, which holds no tenant, is what is left of it here.
async fn written(node: &Node, name: &str, rows: u128) -> Arc<Writer> {
    node.create_tenant(&tenant(name)).await.unwrap();
    let held = node.tenant(&tenant(name)).await.unwrap();
    held.write(vec![notes()]).await.unwrap();
    let rows: Vec<Mutation> = (1..=rows).map(note).collect();
    held.write(rows).await.unwrap();
    // Told apart from the next by when each was last used.
    tokio::time::sleep(Duration::from_millis(10)).await;
    held.writer().clone()
}

/// Past when a tenant last used is kept whatever the budget says.
async fn unused_long_enough() {
    tokio::time::sleep(Duration::from_millis(1_100)).await;
}

/// How many notes `tenant` holds.
fn noted(tenant: &Tenant) -> usize {
    let notes = CollectionName::new("notes").unwrap();
    tenant.read(|state| state.count(&notes))
}

/// The tenants `node` writes for, by name.
fn written_for(node: &Node) -> Vec<String> {
    let figures = node.tenant_figures();
    figures.iter().map(|(id, _)| id.to_string()).collect()
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
    // first let go. Bounded: a sweep that took the tenant out would wait in
    // its flush, on the puts held here, for good.
    let sweeps = async {
        node.sweep_now().await;
        node.sweep_now().await;
    };
    tokio::time::timeout(Duration::from_secs(5), sweeps)
        .await
        .expect("a sweep took out the tenant mid-restore, and is held in its flush");
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

/// Merge note, §9.7 rule 8: the flush a sweep makes before it lets a
/// tenant go over the memory budget is timed under its own name, its wait
/// for one of the sweep's flushes apart from its hold, as the bench holds
/// that wait to a mark of its own; and the tenant is counted as let go for
/// the budget.
#[tokio::test(flavor = "multi_thread")]
async fn the_flush_before_letting_a_tenant_go_is_timed_under_its_own_name() {
    let node = node_on(memory(), |settings| settings.memory = Some(1));
    node.timings().keep_events(true);
    node.create_tenant(&tenant("over")).await.unwrap();
    let over = node.tenant(&tenant("over")).await.unwrap();
    over.write(vec![notes(), note(1), note(2)]).await.unwrap();
    drop(over);
    // Used longer ago than a tenant is kept whatever the budget says.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    node.sweep_now().await;
    assert_eq!(node.let_go().for_budget, 1, "{:?}", node.let_go());
    let events = node.timings().drain_events();
    let named = |timed: Timed| events.iter().filter(|event| event.timed == timed).count();
    assert_eq!(named(Timed::LetGoFlushWait), 1, "{events:?}");
    assert_eq!(named(Timed::LetGoFlushHeld), 1, "{events:?}");
    assert_eq!(named(Timed::BuildingHeld(Holder::Flush)), 1, "{events:?}");
}

/// M7-R5: a sweep waits on no lock of a tenant's. A tenant chosen with its
/// building lock held, by what holds nothing that would keep it otherwise,
/// is kept, however long that lasts; the sweep ends in moments, and a second
/// tenant goes in it. The first goes at the first sweep once the lock is free.
#[tokio::test(start_paused = true)]
async fn a_tenant_whose_building_lock_is_held_is_kept_and_holds_up_no_sweep() {
    let node = over_budget(memory());
    let writer = written(&node, "held", 2).await;
    written(&node, "idle", 2).await;
    unused_long_enough().await;
    let building = writer.building_when_free().await;
    let kept = node.tenants_kept_by_sweeps();
    tokio::time::timeout(Duration::from_secs(1), node.sweep_now())
        .await
        .expect("the sweep waited for the lock held");
    assert_eq!(node.let_go().for_budget, 1, "the second tenant stays");
    assert_eq!(written_for(&node), ["held"], "the wrong tenant went");
    assert_eq!(node.tenants_kept_by_sweeps(), kept + 1, "not kept");
    // Held for the 30 s of a slow clean or restore: no sweep waits for it.
    tokio::time::sleep(Duration::from_secs(30)).await;
    tokio::time::timeout(Duration::from_secs(1), node.sweep_now())
        .await
        .expect("a later sweep waited for the lock held");
    assert_eq!(written_for(&node), ["held"], "let go with its lock held");
    drop(building);
    tokio::time::timeout(Duration::from_secs(1), node.sweep_now())
        .await
        .expect("the sweep once the lock was free");
    assert_eq!(node.let_go().for_budget, 2, "{:?}", node.let_go());
    assert_eq!(written_for(&node), Vec::<String>::new());
    let opened = node.tenant(&tenant("held")).await.unwrap();
    assert_eq!(noted(&opened), 2);
    assert_eq!(opened.read(State::unflushed), 0, "not flushed as it went");
}

/// M7-R5b: a store that never answers holds the let-go's flush for as long
/// as a sweep gives its flushes and no longer. The flush is cut there, and
/// counted as skipped and as no flush that failed; the tenant is let go; and
/// the next opener finds every row, by replaying them.
#[tokio::test(start_paused = true)]
async fn a_let_go_s_flush_that_hangs_is_cut_at_its_bound_and_its_rows_replay() {
    let store = TenantPuts::new();
    let node = over_budget(store.clone());
    written(&node, "hung", 2).await;
    unused_long_enough().await;
    store.hold("hung", Duration::from_secs(60 * 60));
    let asked = tokio::time::Instant::now();
    let bound = LET_GO_FLUSH_WITHIN + Duration::from_secs(1);
    tokio::time::timeout(bound, node.sweep_now())
        .await
        .expect("the sweep was held past the bound by a flush that hangs");
    assert!(
        asked.elapsed() >= LET_GO_FLUSH_WITHIN,
        "{:?}",
        asked.elapsed()
    );
    assert_eq!(store.reached(), 1, "the flush never came to the store");
    assert_eq!(node.let_go().flushes_skipped, 1, "{:?}", node.let_go());
    assert_eq!(node.let_go().for_budget, 1, "{:?}", node.let_go());
    assert_eq!(node.flushing().failed, 0, "a flush cut counted as failed");
    assert_eq!(written_for(&node), Vec::<String>::new(), "not let go");
    let opened = node.tenant(&tenant("hung")).await.unwrap();
    assert_eq!(noted(&opened), 2, "rows lost");
    assert_eq!(opened.read(State::unflushed), 2, "not by replay");
}

/// M7-R5b, nothing else in the gap: the tenant's own flush task comes to try
/// again as the tenant waits among those a sweep flushes, its building lock
/// taken as it was taken out. It queues behind that lock, and never has it:
/// the flush that goes through is the let-go's, under the lock it was taken
/// with, and losing the hold ends what queued.
#[tokio::test(start_paused = true)]
async fn a_flush_queued_behind_a_let_go_never_runs() {
    let store = TenantPuts::new();
    let backoff = Duration::from_secs(2);
    let node = node_on(store.clone(), |settings| {
        settings.memory = Some(1);
        settings.flush_rows = 3;
        settings.flush_backoff = backoff;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    node.timings().keep_events(true);
    written(&node, "first", 1).await;
    written(&node, "second", 1).await;
    store.fail_vectors("third", true);
    written(&node, "third", 3).await;
    until(5, "its flush task's flush failed", || {
        node.flushing().failed == 1
    })
    .await;
    store.fail_vectors("third", false);
    node.timings().drain_events();
    unused_long_enough().await;
    // The two before it hold the sweep's flushes past the try of its flush
    // task, with `backoff` left of 2 s from the failure.
    store.hold("first", Duration::from_secs(3));
    store.hold("second", Duration::from_secs(3));
    tokio::time::timeout(Duration::from_secs(10), node.sweep_now())
        .await
        .expect("the sweep ended");
    let events = node.timings().drain_events();
    let named = |timed: Timed| events.iter().filter(|event| event.timed == timed).count();
    assert_eq!(
        named(Timed::BuildingWait(Holder::Flush)),
        0,
        "a flush had the lock between a tenant's taking and its let-go: {events:?}"
    );
    assert_eq!(named(Timed::LetGoFlushHeld), 3, "{events:?}");
    assert_eq!(node.let_go().for_budget, 3, "{:?}", node.let_go());
    let opened = node.tenant(&tenant("third")).await.unwrap();
    assert_eq!(noted(&opened), 3, "rows lost");
    assert_eq!(opened.read(State::unflushed), 0, "not flushed as it went");
}

/// M7-R5c: a sweep's let-gos run together, so that none waits for the whole
/// of another's, and flush at most two at once, so that a sweep of many does
/// not build them all together. Three flushes of 4 s each take 8 s: 12 s
/// one after another, and 4 s with no limit, three at once.
#[tokio::test(start_paused = true)]
async fn a_sweep_flushes_two_tenants_at_once_and_no_more() {
    assert_eq!(LET_GO_FLUSHES_AT_ONCE, 2);
    let store = TenantPuts::new();
    let node = over_budget(store.clone());
    node.timings().keep_events(true);
    let names = ["one", "two", "three"];
    for name in names {
        written(&node, name, 1).await;
    }
    unused_long_enough().await;
    let flush = Duration::from_secs(4);
    for name in names {
        store.hold(name, flush);
    }
    let asked = tokio::time::Instant::now();
    tokio::time::timeout(2 * flush + Duration::from_secs(1), node.sweep_now())
        .await
        .expect("let go one after another");
    assert_eq!(
        store.most_waiting.load(Ordering::SeqCst),
        LET_GO_FLUSHES_AT_ONCE,
        "flushes at once, after {:?}",
        asked.elapsed()
    );
    assert_eq!(node.let_go().flushes_skipped, 0, "{:?}", node.let_go());
    assert_eq!(written_for(&node), Vec::<String>::new());
    // What a let-go's flush waited for: the others of its sweep.
    let events = node.timings().drain_events();
    let waited: Vec<Duration> = events
        .iter()
        .filter(|event| event.timed == Timed::LetGoFlushWait)
        .map(|event| event.took)
        .collect();
    assert_eq!(waited.len(), 3, "{events:?}");
    let least = |at_least: Duration| waited.iter().filter(|took| **took >= at_least).count();
    assert_eq!(least(flush), 1, "one waited behind the others: {waited:?}");
}

/// M7-R5c: one deadline for the whole sweep. With every flush hanging, the
/// two let-gos at the head are cut at it, those behind are not reached and
/// never begin, each is counted as skipped, and all are let go by then, not
/// one pair after another. Those not reached replay their rows; one with
/// nothing to flush waits for nothing and is no flush skipped.
#[tokio::test(start_paused = true)]
async fn one_deadline_ends_a_sweep_s_flushes_and_those_not_reached_replay() {
    let store = TenantPuts::new();
    let node = over_budget(store.clone());
    node.timings().keep_events(true);
    let names = ["one", "two", "three", "four"];
    for name in names {
        written(&node, name, 1).await;
    }
    let writer = written(&node, "flushed", 1).await;
    writer.flush().await.unwrap();
    node.timings().drain_events();
    unused_long_enough().await;
    for name in names {
        store.hold(name, Duration::from_secs(60 * 60));
    }
    let bound = LET_GO_FLUSH_WITHIN + Duration::from_secs(1);
    tokio::time::timeout(bound, node.sweep_now())
        .await
        .expect("the sweep's flushes outlasted its one deadline");
    assert_eq!(
        store.reached(),
        LET_GO_FLUSHES_AT_ONCE,
        "begun past the bound"
    );
    assert_eq!(node.let_go().flushes_skipped, 4, "{:?}", node.let_go());
    assert_eq!(node.let_go().for_budget, 5, "{:?}", node.let_go());
    assert_eq!(written_for(&node), Vec::<String>::new(), "not let go");
    let events = node.timings().drain_events();
    let waited: Vec<Duration> = events
        .iter()
        .filter(|event| event.timed == Timed::LetGoFlushWait)
        .map(|event| event.took)
        .collect();
    assert_eq!(waited.len(), 4, "{events:?}");
    let whole = waited.iter().filter(|took| **took >= LET_GO_FLUSH_WITHIN);
    assert_eq!(whole.count(), 2, "not reached, so not told: {waited:?}");
    let begun = events
        .iter()
        .filter(|event| event.timed == Timed::BuildingHeld(Holder::Flush));
    assert_eq!(
        begun.count(),
        LET_GO_FLUSHES_AT_ONCE,
        "one begun at the deadline"
    );
    store.held.lock().unwrap().clear();
    for name in names {
        let opened = node.tenant(&tenant(name)).await.unwrap();
        assert_eq!(noted(&opened), 1, "{name}'s rows lost");
        assert_eq!(opened.read(State::unflushed), 1, "{name}, not by replay");
    }
}

/// M7-R9: a tail of one row, far under a flush's worth, is flushed as its
/// tenant is let go: the next opener replays no log.
#[tokio::test(start_paused = true)]
async fn a_tail_under_a_flush_s_rows_is_flushed_as_its_tenant_is_let_go() {
    let node = over_budget(memory());
    node.timings().keep_events(true);
    written(&node, "tail", 1).await;
    unused_long_enough().await;
    tokio::time::timeout(Duration::from_secs(5), node.sweep_now())
        .await
        .expect("the sweep ended");
    assert_eq!(node.let_go().for_budget, 1, "{:?}", node.let_go());
    let events = node.timings().drain_events();
    let held = events
        .iter()
        .filter(|event| event.timed == Timed::LetGoFlushHeld);
    assert_eq!(held.count(), 1, "no flush made: {events:?}");
    let opened = node.tenant(&tenant("tail")).await.unwrap();
    assert_eq!(noted(&opened), 1);
    assert_eq!(opened.read(State::unflushed), 0, "its log was replayed");
}

/// The eviction flush's mark is held to flushes: a let-go with nothing to
/// flush adds no sample to it, which a run of them would make read better
/// than it is, and takes up no flush of its sweep's, so that one behind it
/// waits for nothing.
#[tokio::test(start_paused = true)]
async fn a_let_go_with_nothing_to_flush_adds_no_sample() {
    let store = TenantPuts::new();
    let node = over_budget(store.clone());
    node.timings().keep_events(true);
    for name in ["flushed", "empty"] {
        written(&node, name, 1).await.flush().await.unwrap();
    }
    written(&node, "tail", 1).await;
    node.timings().drain_events();
    unused_long_enough().await;
    tokio::time::timeout(Duration::from_secs(5), node.sweep_now())
        .await
        .expect("the sweep ended");
    assert_eq!(node.let_go().for_budget, 3, "{:?}", node.let_go());
    assert_eq!(node.let_go().flushes_skipped, 0, "{:?}", node.let_go());
    let events = node.timings().drain_events();
    let named = |timed: Timed| events.iter().filter(|event| event.timed == timed).count();
    assert_eq!(named(Timed::LetGoFlushWait), 1, "{events:?}");
    assert_eq!(named(Timed::LetGoFlushHeld), 1, "{events:?}");
}
