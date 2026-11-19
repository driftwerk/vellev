//! A node's sweeps (resident accounting PR 1, delta reviews of #165): one
//! at a time, so that no sweep holds what another must find held by
//! nothing else; and those asked for as tenants are used coalesced, so
//! that a sweep held up, as by a slow flush, never has a sweep a request
//! queued behind it.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, RequestId, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{
    KeptBySweeps, Node, Settings, Tenant, LET_GO_FLUSHES_AT_ONCE, LET_GO_FLUSH_WITHIN,
};
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator, Point};
use vellev_storage::timings::{Event, Holder, KeptFor, Timed};
use vellev_storage::{Holding, MergeGate, Mutation, State, StorageError, Writer};

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

/// A store that holds flushes up. While it is armed every put waits until
/// it is disarmed: a flush that hangs on the store. Tenant by tenant, the
/// next put of one named by [`Self::hold`] waits as long as that says, and
/// the vectors of one named by [`Self::fail_vectors`] fail to go up.
#[derive(Debug)]
struct HeldPuts {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    /// Whether a put has waited while the store was armed.
    held_armed: AtomicBool,
    holds: Mutex<HashMap<String, Duration>>,
    failing: Mutex<HashSet<String>>,
    /// The tenants whose flushes have come to a put held.
    reached: Mutex<HashSet<String>>,
    /// How many puts held wait now, and the most that ever did at once.
    waiting: AtomicUsize,
    most_waiting: AtomicUsize,
}

impl HeldPuts {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: memory(),
            armed: AtomicBool::new(false),
            held_armed: AtomicBool::new(false),
            holds: Mutex::default(),
            failing: Mutex::default(),
            reached: Mutex::default(),
            waiting: AtomicUsize::new(0),
            most_waiting: AtomicUsize::new(0),
        })
    }

    /// Hold every put from now on, until disarmed.
    fn arm(&self, armed: bool) {
        self.armed.store(armed, Ordering::SeqCst);
    }

    /// Whether a put has waited while the store was armed.
    fn held_armed(&self) -> bool {
        self.held_armed.load(Ordering::SeqCst)
    }

    fn hold(&self, tenant: &str, held: Duration) {
        self.holds.lock().unwrap().insert(tenant.into(), held);
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
impl Blobs for HeldPuts {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        let tenant = key.split('/').next().unwrap_or_default();
        if key.ends_with(".vectors") && self.failing.lock().unwrap().contains(tenant) {
            return Err(StorageError::Background("no room for vectors".into()));
        }
        while self.armed.load(Ordering::SeqCst) {
            self.held_armed.store(true, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let held = self.holds.lock().unwrap().remove(tenant);
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

/// The durations of the events of `timed` among `events`.
fn took(events: &[Event], timed: Timed) -> Vec<Duration> {
    let of = events.iter().filter(|event| event.timed == timed);
    of.map(|event| event.took).collect()
}

/// How many of `events` are of `timed`.
fn count(events: &[Event], timed: Timed) -> usize {
    events.iter().filter(|event| event.timed == timed).count()
}

/// The tenants `node` writes for, by name.
fn written_for(node: &Node) -> Vec<String> {
    let figures = node.tenant_figures();
    figures.iter().map(|(id, _)| id.to_string()).collect()
}

/// Delta review of #165, M1: every use of a tenant over the budget asks for
/// a sweep, and a sweep can be held, for up to its deadline, in the flush
/// of a tenant it lets go. Those asked for meanwhile are coalesced: while
/// one sweep is held in a flush,
/// a thousand uses of a tenant leave at most one more sweep to run once it
/// ends, not a thousand queued behind it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweeps_asked_for_while_one_runs_are_coalesced_into_one() {
    let store = HeldPuts::new();
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
    store.arm(true);
    let before = node.sweeps();
    // A sweep that lets `slow` go, and hangs in its flush.
    drop(node.tenant(&tenant("used")).await.unwrap());
    until(
        5,
        "the sweep never came to flush the tenant it lets go",
        || store.held_armed(),
    )
    .await;
    assert_eq!(node.sweeps(), before + 1, "one sweep, held in the flush");
    for _ in 0..1_000 {
        drop(node.tenant(&tenant("used")).await.unwrap());
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(node.sweeps(), before + 1, "none began while one ran");
    store.arm(false);
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
    let store = HeldPuts::new();
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

    store.arm(true);
    let point = Point::Position(kept.position);
    let restoring = tokio::spawn({
        let (node, acme) = (node.clone(), acme.clone());
        async move { node.restore(&acme, point, RequestId::generate()).await }
    });
    until(5, "the restore never came to write", || store.held_armed()).await;
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
    store.arm(false);
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
    assert_eq!(count(&events, Timed::LetGoFlushWait), 1, "{events:?}");
    assert_eq!(count(&events, Timed::LetGoFlushHeld), 1, "{events:?}");
    assert_eq!(
        count(&events, Timed::BuildingHeld(Holder::Flush)),
        1,
        "{events:?}"
    );
}

/// M7-R5: a sweep waits on no lock of a tenant's. A tenant whose building
/// lock is held is kept, while the sweep ends in moments with a second
/// tenant gone; the first goes once the lock is free.
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
    let for_building = KeptBySweeps {
        building: 1,
        held: 0,
    };
    assert_eq!(
        node.kept_by_sweeps(),
        for_building,
        "kept, but not for its lock"
    );
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

/// The busy-lock keep (§9.6): a tenant kept over the budget for its busy
/// building lock, sweep after sweep, is timed as one spell, from the first
/// sweep that kept it so to the last, at five minutes. A merge holds the
/// lock from a round, which holds the tenant, so that it is kept as held:
/// only what holds the lock and not the tenant, as a sweep's race with a
/// task or this test's seam, keeps it for its lock.
#[tokio::test(start_paused = true)]
async fn a_tenant_kept_for_its_building_lock_is_timed_as_one_spell() {
    let node = over_budget(memory());
    node.timings().keep_events(true);
    let writer = written(&node, "busy", 2).await;
    unused_long_enough().await;
    let building = writer.building_when_free().await;
    node.timings().drain_events();
    // Five minutes held, as by a merge's long commit, swept every 30 s,
    // the last sweep as the five minutes end.
    let held_for = Duration::from_secs(5 * 60);
    let every = Duration::from_secs(30);
    for _ in 0..held_for.as_secs() / every.as_secs() {
        node.sweep_now().await;
        tokio::time::sleep(every).await;
    }
    node.sweep_now().await;
    let kept = KeptBySweeps {
        building: 11,
        held: 0,
    };
    assert_eq!(node.kept_by_sweeps(), kept);
    drop(building);
    node.sweep_now().await;
    assert_eq!(written_for(&node), Vec::<String>::new(), "not let go");
    let events = node.timings().drain_events();
    let spells = took(&events, Timed::Kept(KeptFor::Building));
    assert_eq!(spells.len(), 1, "{events:?}");
    assert!(
        spells[0] >= held_for && spells[0] < held_for + Duration::from_secs(1),
        "{spells:?}"
    );
}

/// A node over the budget, flushing every ten rows, with tenant "merging"
/// made on it and its maintenance merging it, that merge held between its
/// build and its commit, past where a stop ends it; and the tenant, still
/// held by the caller.
async fn merging_held_before_commit() -> (Node, Arc<Tenant>, Arc<MergeGate>) {
    const ROWS: u128 = 10;
    let node = node_on(memory(), |settings| {
        settings.memory = Some(1);
        settings.flush_rows = ROWS as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    node.timings().keep_events(true);
    node.create_tenant(&tenant("merging")).await.unwrap();
    let held = node.tenant(&tenant("merging")).await.unwrap();
    let gate = held.hold_merges_before_commit();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let rows: Vec<Mutation> = (0..ROWS).map(|n| note(flush * ROWS + n)).collect();
        held.write(rows).await.unwrap();
        until(30, "the node never flushed the tenant", || {
            held.read(State::unflushed) == 0
        })
        .await;
    }
    tokio::time::timeout(Duration::from_secs(30), gate.reached.notified())
        .await
        .expect("the merge never came to its commit");
    (node, held, gate)
}

/// Sweep `node` every 30 s for `held_for`, then, once `release` lets its
/// tenant's merge go on, until it is let go. Says what sweeps had kept by
/// the release.
async fn swept_while_merging(
    node: &Node,
    held_for: Duration,
    release: impl FnOnce(),
) -> KeptBySweeps {
    let every = Duration::from_secs(30);
    for _ in 0..held_for.as_secs() / every.as_secs() {
        node.sweep_now().await;
        tokio::time::sleep(every).await;
    }
    let kept = node.kept_by_sweeps();
    release();
    let let_go = async {
        while !written_for(node).is_empty() {
            node.sweep_now().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(30), let_go)
        .await
        .expect("never let go once its merge ended");
    kept
}

/// A keep as held (§9.6): a tenant kept over the budget while its
/// own maintenance merges it, sweep after sweep, is timed as one spell kept
/// as held, from the first sweep that kept it to the sweep that takes it
/// out. The merge is held between its build and its commit, past where a
/// stop ends it: its round holds the tenant, but not yet the building lock.
#[tokio::test(start_paused = true)]
async fn a_tenant_its_merge_holds_is_timed_as_one_held_spell() {
    let (node, held, gate) = merging_held_before_commit().await;
    drop(held);
    unused_long_enough().await;
    node.timings().drain_events();
    let swept_from = tokio::time::Instant::now();
    let held_for = Duration::from_secs(5 * 60);
    let kept = swept_while_merging(&node, held_for, || gate.release.notify_one()).await;
    assert!(kept.held >= 10 && kept.building == 0, "{kept:?}");
    let spanned = swept_from.elapsed();
    let events = node.timings().drain_events();
    let spells = took(&events, Timed::Kept(KeptFor::Held));
    assert_eq!(spells.len(), 1, "{events:?}");
    assert!(
        spells[0] >= held_for && spells[0] <= spanned,
        "{spells:?}, swept over {spanned:?}"
    );
}

/// As above, with the merge held in its commit, under the building lock,
/// for 150 s. That is under what the writer allows the commit's turn
/// before the node gives the tenant up as stuck ([`Writer::stalled`]):
/// about 235 s on a 30 s lease, its 5 s in memory, one upload request's
/// 187 s, 20 s on local disk and three tries to commit of 7.5 s.
#[tokio::test(start_paused = true)]
async fn a_tenant_its_merge_commit_holds_under_the_lock_is_timed_as_one_held_spell() {
    let (node, held, gate) = merging_held_before_commit().await;
    let writer = held.writer().clone();
    let committing = writer.hold_checkpoints_once_adopted();
    writer.stop_holding_merges();
    node.timings().drain_events();
    let swept_from = tokio::time::Instant::now();
    gate.release.notify_one();
    tokio::time::timeout(Duration::from_secs(30), committing.reached.notified())
        .await
        .expect("the merge never came into its commit");
    drop(held);
    unused_long_enough().await;
    let held_for = Duration::from_secs(150);
    let release = || {
        writer.stop_holding_checkpoints();
        committing.release.notify_one();
    };
    let kept = swept_while_merging(&node, held_for, release).await;
    assert!(kept.held >= 5 && kept.building == 0, "{kept:?}");
    let spanned = swept_from.elapsed();
    let events = node.timings().drain_events();
    let commits = took(&events, Timed::BuildingHeld(Holder::MergeCommit));
    assert_eq!(commits.len(), 1, "{events:?}");
    assert!(
        commits[0] >= held_for && commits[0] <= spanned,
        "the lock was not held through the keep: {commits:?}, swept over {spanned:?}"
    );
    let spells = took(&events, Timed::Kept(KeptFor::Held));
    assert_eq!(spells.len(), 1, "{events:?}");
    assert!(
        spells[0] >= held_for && spells[0] <= spanned,
        "{spells:?}, swept over {spanned:?}"
    );
}

/// Review of f800a51, M1: a tenant still kept when the events are read has
/// its spell so far reported, as no event of an ended spell tells it: so
/// that a tenant kept to the end of a run is held to the mark.
#[tokio::test(start_paused = true)]
async fn a_spell_still_open_when_read_is_reported_as_it_stands() {
    let node = over_budget(memory());
    node.timings().keep_events(true);
    let writer = written(&node, "busy", 2).await;
    unused_long_enough().await;
    let _building = writer.building_when_free().await;
    node.timings().drain_events();
    node.sweep_now().await;
    let kept_for = Duration::from_secs(90);
    tokio::time::sleep(kept_for).await;
    node.sweep_now().await;
    let ended = node.timings().drain_events();
    assert_eq!(
        took(&ended, Timed::Kept(KeptFor::Building)),
        Vec::<Duration>::new(),
        "{ended:?}"
    );
    let open = node.kept_spells_open();
    let spells = took(&open, Timed::Kept(KeptFor::Building));
    assert_eq!(spells.len(), 1, "{open:?}");
    assert!(
        spells[0] >= kept_for && spells[0] < kept_for + Duration::from_secs(1),
        "{spells:?}"
    );
    assert_eq!(open[0].partition, "busy/0");
}

/// A tenant a request holds, chosen to be let go, is kept as held, not as
/// one whose building lock is busy: what keeps it is looked at first.
#[tokio::test(start_paused = true)]
async fn a_tenant_a_request_holds_is_kept_as_held() {
    let node = over_budget(memory());
    node.create_tenant(&tenant("asked")).await.unwrap();
    let asked = node.tenant(&tenant("asked")).await.unwrap();
    asked.write(vec![notes(), note(1)]).await.unwrap();
    unused_long_enough().await;
    tokio::time::timeout(Duration::from_secs(1), node.sweep_now())
        .await
        .expect("the sweep ended");
    let held = KeptBySweeps {
        building: 0,
        held: 1,
    };
    assert_eq!(node.kept_by_sweeps(), held);
    assert_eq!(node.tenants_kept_by_sweeps(), 1);
    drop(asked);
}

/// M7-R5b: a store that never answers holds the let-go's flush for as long
/// as a sweep gives its flushes and no longer. The flush is cut there, and
/// counted as skipped and as no flush that failed; the tenant is let go; and
/// the next opener finds every row, by replaying them.
#[tokio::test(start_paused = true)]
async fn a_let_go_s_flush_that_hangs_is_cut_at_its_bound_and_its_rows_replay() {
    let store = HeldPuts::new();
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

/// Review A of #177, M1: a let-go's flush cut at its deadline mid-build
/// stops that build. The build runs off the async threads, where dropping
/// the flush does not reach it, and would otherwise run on for nobody,
/// holding the rows it copied and a turn among the work off the async
/// threads. A build held until its flush is stopped stands for one that runs
/// long: the cut lets it go, and so its turn, within moments, not at the
/// minute a test hold lasts at most. Real time, as a build held off the async
/// threads holds a paused clock where it is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_let_go_s_flush_cut_mid_build_gives_its_build_up() {
    let within = Duration::from_secs(1);
    let node = node_on(memory(), |settings| {
        settings.memory = Some(1);
        settings.flush_rows = 1_000_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        settings.let_go_flush_within = within;
    });
    let builds = node.merge_builds().clone();
    written(&node, "building", 2).await;
    unused_long_enough().await;
    builds.hold(Some(Holding::FlushUntilStopped));
    let sweep = tokio::spawn({
        let node = node.clone();
        async move { node.sweep_now().await }
    });
    until(5, "the let-go's flush came to build", || builds.held() == 1).await;
    tokio::time::timeout(within + Duration::from_secs(2), sweep)
        .await
        .expect("the sweep outlasted what it gives its flushes")
        .unwrap();
    assert_eq!(node.let_go().flushes_skipped, 1, "{:?}", node.let_go());
    until(3, "the cut flush's build ran on, holding its turn", || {
        builds.held() == 0
    })
    .await;
    builds.hold(None);
    let opened = node.tenant(&tenant("building")).await.unwrap();
    assert_eq!(noted(&opened), 2, "rows lost");
}

/// Review of 79b8103, M2: the hold a test puts on a flush's build holds no
/// merge commit's. A merge's commit flushes the rows still in memory under
/// the building lock, and nothing stops it: held as a flush's, it would wait
/// out the hold's bound. Real time, as the merge builds off the runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_commit_s_flush_is_not_held_as_a_flush_s() {
    let (node, held, gate) = merging_held_before_commit().await;
    let builds = node.merge_builds().clone();
    held.write(vec![note(1_000)]).await.unwrap();
    builds.hold(Some(Holding::FlushUntilStopped));
    gate.release.notify_one();
    let committed = tokio::time::timeout(Duration::from_secs(20), async {
        while held.read(State::unflushed) > 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    builds.hold(None);
    committed.expect("the merge commit's flush was held as a flush's");
}

/// Review A of #177, M2: a let-go's flush cut by the sweep's deadline after
/// its checkpoint was committed and adopted, as it frees what it took out,
/// went through. It is no flush skipped, nor one that failed, and the next
/// opener replays nothing: the head holds the rows.
#[tokio::test(start_paused = true)]
async fn a_let_go_s_flush_cut_once_adopted_is_no_flush_skipped() {
    let node = over_budget(memory());
    let writer = written(&node, "adopted", 2).await;
    unused_long_enough().await;
    // Never released: the checkpoint waits there until the deadline cuts it.
    let gate = writer.hold_checkpoints_once_adopted();
    let bound = LET_GO_FLUSH_WITHIN + Duration::from_secs(1);
    tokio::time::timeout(bound, node.sweep_now())
        .await
        .expect("the sweep outlasted what it gives its flushes");
    tokio::time::timeout(Duration::from_secs(1), gate.reached.notified())
        .await
        .expect("the flush was cut before its checkpoint was adopted");
    drop(gate);
    writer.stop_holding_checkpoints();
    assert_eq!(node.let_go().flushes_skipped, 0, "{:?}", node.let_go());
    assert_eq!(node.let_go().for_budget, 1, "{:?}", node.let_go());
    assert_eq!(
        node.flushing().failed,
        0,
        "a flush that went through counted as failed"
    );
    let opened = node.tenant(&tenant("adopted")).await.unwrap();
    assert_eq!(noted(&opened), 2, "rows lost");
    assert_eq!(opened.read(State::unflushed), 0, "its rows replayed");
}

/// Review A of #177, L4: a tenant reopened on this node while its let-go
/// waits for one of the sweep's flushes fences the let-go's lease, and the
/// hold on it is lost at its next renewal. The let-go, given its turn, finds
/// it superseded: it flushes nothing, as its commit would be refused, and
/// counts no flush failed and none skipped. The reopened tenant has every row.
#[tokio::test(start_paused = true)]
async fn a_let_go_superseded_while_it_waits_for_a_flush_does_not_flush() {
    let store = HeldPuts::new();
    let node = node_on(store.clone(), |settings| {
        settings.memory = Some(1);
        settings.flush_rows = 1_000_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        // Renewed every second: the fenced hold is lost long before the
        // two flushes ahead of it end.
        settings.lease = Duration::from_secs(3);
    });
    node.timings().keep_events(true);
    for name in ["first", "second"] {
        written(&node, name, 1).await;
    }
    written(&node, "third", 2).await;
    node.timings().drain_events();
    unused_long_enough().await;
    let flush = Duration::from_secs(10);
    store.hold("first", flush);
    store.hold("second", flush);
    let sweep = tokio::spawn({
        let node = node.clone();
        async move { node.sweep_now().await }
    });
    until(5, "the two flushes ahead came to the store", || {
        store.reached() == 2
    })
    .await;
    let reopened = node.tenant(&tenant("third")).await.unwrap();
    tokio::time::timeout(LET_GO_FLUSH_WITHIN, sweep)
        .await
        .expect("the sweep ended")
        .unwrap();
    assert_eq!(
        node.flushing().failed,
        0,
        "a superseded let-go counted as failed"
    );
    assert_eq!(node.let_go().flushes_skipped, 0, "{:?}", node.let_go());
    let events = node.timings().drain_events();
    let held = count(&events, Timed::LetGoFlushHeld);
    assert_eq!(held, 2, "the superseded let-go flushed: {events:?}");
    assert_eq!(noted(&reopened), 2, "rows lost");
}

/// M7-R5b: the tenant's flush task, trying again while the tenant waits
/// among a sweep's let-gos, queues behind the building lock taken with it
/// and never has it: losing the hold ends what queued. That it queued is
/// asserted (review A of #177, L5). The order of `lose` and the lock's drop
/// is not shown: on this runtime nothing runs between them, so only a
/// multi-thread runtime could see it, and only by a race.
#[tokio::test(start_paused = true)]
async fn a_flush_queued_behind_a_let_go_never_runs() {
    let store = HeldPuts::new();
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
    let third = written(&node, "third", 3).await;
    until(5, "its flush task's flush failed", || {
        node.flushing().failed == 1
    })
    .await;
    store.fail_vectors("third", false);
    node.timings().drain_events();
    unused_long_enough().await;
    let asked = third.flushes_asked_for_building();
    // The two before it hold the sweep's flushes past the try of its flush
    // task, with `backoff` left of 2 s from the failure.
    store.hold("first", Duration::from_secs(3));
    store.hold("second", Duration::from_secs(3));
    tokio::time::timeout(Duration::from_secs(10), node.sweep_now())
        .await
        .expect("the sweep ended");
    let events = node.timings().drain_events();
    assert_eq!(
        count(&events, Timed::LetGoFlushHeld),
        3,
        "a tenant's rows went with a flush of another's making: {events:?}"
    );
    assert_eq!(
        count(&events, Timed::BuildingWait(Holder::Flush)),
        0,
        "a flush had the lock between a tenant's taking and its hold lost: {events:?}"
    );
    assert_eq!(
        third.flushes_asked_for_building(),
        asked + 1,
        "its flush task never queued for the lock as it waited to be flushed"
    );
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
    let store = HeldPuts::new();
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
    let waited = took(&events, Timed::LetGoFlushWait);
    assert_eq!(waited.len(), 3, "{events:?}");
    let least = |at_least: Duration| waited.iter().filter(|took| **took >= at_least).count();
    assert_eq!(least(flush), 1, "one waited behind the others: {waited:?}");
    // The wait split in two: each that had a slot is queued for as long as
    // its wait, and none was not reached.
    assert_eq!(took(&events, Timed::LetGoFlushQueued), waited, "{events:?}");
    assert_eq!(
        took(&events, Timed::LetGoFlushNotReached),
        Vec::<Duration>::new(),
        "{events:?}"
    );
}

/// Review B of #177, L7: how many a sweep flushes at once is a setting. A
/// node set to one flushes one at a time: three flushes of 4 s each take
/// 12 s, and never two wait on the store together.
#[tokio::test(start_paused = true)]
async fn a_node_set_to_one_flush_at_a_time_flushes_one_at_a_time() {
    let store = HeldPuts::new();
    let node = node_on(store.clone(), |settings| {
        settings.memory = Some(1);
        settings.flush_rows = 1_000_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        settings.let_go_flushes_at_once = 1;
    });
    let names = ["one", "two", "three"];
    for name in names {
        written(&node, name, 1).await;
    }
    unused_long_enough().await;
    let flush = Duration::from_secs(4);
    for name in names {
        store.hold(name, flush);
    }
    tokio::time::timeout(3 * flush + Duration::from_secs(1), node.sweep_now())
        .await
        .expect("the sweep ended");
    assert_eq!(
        store.most_waiting.load(Ordering::SeqCst),
        1,
        "flushes at once"
    );
    assert_eq!(node.let_go().flushes_skipped, 0, "{:?}", node.let_go());
    assert_eq!(written_for(&node), Vec::<String>::new());
}

/// M7-R5c: one deadline for the whole sweep. With every flush hanging, the
/// two let-gos at the head are cut at it, those behind are not reached and
/// never begin, each is counted as skipped, and all are let go by then, not
/// one pair after another. Neither of those cut is timed as a flush's
/// hold. Those not reached replay their rows; one with nothing to flush
/// waits for nothing and is no flush skipped.
#[tokio::test(start_paused = true)]
async fn one_deadline_ends_a_sweep_s_flushes_and_those_not_reached_replay() {
    let store = HeldPuts::new();
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
    let waited = took(&events, Timed::LetGoFlushWait);
    assert_eq!(waited.len(), 4, "{events:?}");
    let whole = waited.iter().filter(|took| **took >= LET_GO_FLUSH_WITHIN);
    assert_eq!(whole.count(), 2, "not reached, so not told: {waited:?}");
    // The wait split in two: the two that had a slot are queued, the two
    // the deadline reached first are not reached, each for the whole of
    // what the flushes are given.
    assert_eq!(count(&events, Timed::LetGoFlushQueued), 2, "{events:?}");
    let not_reached = took(&events, Timed::LetGoFlushNotReached);
    assert_eq!(not_reached.len(), 2, "{events:?}");
    assert!(
        not_reached.iter().all(|took| *took >= LET_GO_FLUSH_WITHIN),
        "{not_reached:?}"
    );
    // Review A of #177, L1: the two begun, as `reached` says, were cut, and
    // a cut hold is no flush's hold, nor a let-go flush's.
    assert_eq!(
        count(&events, Timed::BuildingHeld(Holder::Flush)),
        0,
        "a cut flush timed as held"
    );
    assert_eq!(
        count(&events, Timed::LetGoFlushHeld),
        0,
        "a cut flush timed as held"
    );
    store.holds.lock().unwrap().clear();
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
    let held = count(&events, Timed::LetGoFlushHeld);
    assert_eq!(held, 1, "no flush made: {events:?}");
    let opened = node.tenant(&tenant("tail")).await.unwrap();
    assert_eq!(noted(&opened), 1);
    assert_eq!(opened.read(State::unflushed), 0, "its log was replayed");
}

/// Review B of #177, L6: a tenant whose last writes were deletes, with no
/// row written since its last flush, still has what a flush takes: its
/// tombstones. It is flushed as it is let go, and the next opener replays no
/// log of them.
#[tokio::test(start_paused = true)]
async fn a_tenant_whose_last_writes_were_deletes_is_flushed_as_it_is_let_go() {
    let node = over_budget(memory());
    node.timings().keep_events(true);
    let writer = written(&node, "deleted", 2).await;
    writer.flush().await.unwrap();
    let held = node.tenant(&tenant("deleted")).await.unwrap();
    held.write(vec![Mutation::Delete {
        collection: CollectionName::new("notes").unwrap(),
        id: ObjectId(uuid::Uuid::from_u128(1)),
        only_if_version: None,
    }])
    .await
    .unwrap();
    assert_eq!(
        held.read(State::unflushed),
        0,
        "a row written since the flush"
    );
    drop(held);
    node.timings().drain_events();
    unused_long_enough().await;
    tokio::time::timeout(Duration::from_secs(5), node.sweep_now())
        .await
        .expect("the sweep ended");
    assert_eq!(node.let_go().for_budget, 1, "{:?}", node.let_go());
    let events = node.timings().drain_events();
    let held = count(&events, Timed::LetGoFlushHeld);
    assert_eq!(held, 1, "its deletes were not flushed: {events:?}");
    let opened = node.tenant(&tenant("deleted")).await.unwrap();
    assert_eq!(noted(&opened), 1, "the delete was lost");
}

/// The eviction flush's mark is held to flushes: a let-go with nothing to
/// flush adds no sample to it, which a run of them would make read better
/// than it is, and takes up no flush of its sweep's, so that one behind it
/// waits for nothing.
#[tokio::test(start_paused = true)]
async fn a_let_go_with_nothing_to_flush_adds_no_sample() {
    let store = HeldPuts::new();
    let node = over_budget(store.clone());
    node.timings().keep_events(true);
    written(&node, "flushed", 1).await.flush().await.unwrap();
    written(&node, "tail", 1).await;
    node.timings().drain_events();
    unused_long_enough().await;
    tokio::time::timeout(Duration::from_secs(5), node.sweep_now())
        .await
        .expect("the sweep ended");
    assert_eq!(node.let_go().for_budget, 2, "{:?}", node.let_go());
    assert_eq!(node.let_go().flushes_skipped, 0, "{:?}", node.let_go());
    let events = node.timings().drain_events();
    assert_eq!(count(&events, Timed::LetGoFlushWait), 1, "{events:?}");
    assert_eq!(count(&events, Timed::LetGoFlushHeld), 1, "{events:?}");
}
