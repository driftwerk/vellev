//! A node's merges (merge note, step 3, §7 and §7.7). Over its memory
//! budget it stops the merge of a tenant it would let go and cannot yet,
//! and lets the tenant go at a later round, once the build has ended; never
//! while a request holds the tenant. It builds at most `MERGE_BUILDS`
//! merges at once, a stopped one that runs on still among them until it
//! ends, and what that one was reckoned to hold counts as draining until
//! then.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Node, Settings};
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator};
use vellev_storage::{Holding, Mutation, State, MERGE_BUILDS};

/// Notes written between flushes: as many as a flush is made at.
const NOTES: u128 = 10;

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
        properties: serde_json::json!({ "title": format!("note {n} of a few") })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![1.0, n as f32]),
        only_if_version: None,
    }
}

/// A node, as `set` sets it up, on a coordinator and store of its own.
fn node(set: impl FnOnce(&mut Settings)) -> Node {
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(
        object_store::memory::InMemory::new(),
    )));
    node_on(blobs, set)
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

/// What `work` comes to within `limit` seconds: a merge that is never
/// stopped, or a turn never given back, fails its rule in seconds rather
/// than hanging it.
async fn within<T>(limit: u64, what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(limit), work)
        .await
        .unwrap_or_else(|_| panic!("{what} within {limit} s"))
}

/// Wait until `holds`, at most `limit` seconds.
async fn until(limit: u64, what: &str, holds: impl Fn() -> bool) {
    within(limit, what, async {
        while !holds() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
}

/// Make tenant `name` through `node`, and write it four flushes' worth of
/// notes, each flushed by the node as it looks after the tenant: the round
/// after the fourth merges the four segments. The tenant is not held after.
async fn four_flushes(node: &Node, name: &str) {
    node.create_tenant(&tenant(name)).await.unwrap();
    let held = node.tenant(&tenant(name)).await.unwrap();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        let flushed = || held.read(State::unflushed) == 0;
        until(30, "the node never flushed the tenant", flushed).await;
    }
}

/// The tenants a sweep during a free finds: "freeing", held, with what its
/// merge replaced waiting to be freed; "merging", with a merge of its own
/// under way, held where it builds; and "idle", which nothing holds.
struct Freeing {
    held: Arc<vellev_node::Tenant>,
    frees: Arc<vellev_storage::FreesHeld>,
}

impl Drop for Freeing {
    fn drop(&mut self) {
        self.frees.let_go();
    }
}

/// `node`'s three tenants, as [`Freeing`] says.
async fn freeing_beside_a_merge(node: &Node) -> Freeing {
    let builds = node.merge_builds();
    node.create_tenant(&tenant("idle")).await.unwrap();
    let idle = node.tenant(&tenant("idle")).await.unwrap();
    idle.write(vec![notes(), note(1)]).await.unwrap();
    drop(idle);
    node.create_tenant(&tenant("freeing")).await.unwrap();
    let held = node.tenant(&tenant("freeing")).await.unwrap();
    let frees = held.partition().hold_frees();
    let freeing = Freeing { held, frees };
    freeing.held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        freeing.held.write(batch).await.unwrap();
        let flushed = || freeing.held.read(State::unflushed) == 0;
        until(30, "the node never flushed the tenant", flushed).await;
    }
    until(30, "the merge's free never began", || {
        freeing.frees.reached() > 0
    })
    .await;
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(node, "merging").await;
    until(30, "the other merge never began", || builds.held() == 1).await;
    freeing
}

/// What a tenant has yet to free leaves by itself: a sweep during a free
/// neither lets a tenant go nor stops a merge for it, though it would put
/// the node over its budget were it counted in what the tenants hold. The
/// budget here lies between the two, worked out on a node of its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sweep_during_a_free_stops_nothing_and_lets_nothing_go() {
    let set = |settings: &mut Settings| settings.flush_rows = NOTES as usize;
    let measured = node(set);
    let freeing = freeing_beside_a_merge(&measured).await;
    let to_free = measured.freeing();
    assert!(to_free > 0, "nothing counted as being freed");
    assert_eq!(to_free, freeing.held.partition().freeing());
    let mut held = 0;
    for name in ["idle", "freeing", "merging"] {
        let tenant = measured.tenant(&tenant(name)).await.unwrap();
        held += tenant.partition().figures().resident;
    }
    measured.merge_builds().hold(None);
    drop(freeing);

    let node = node(|settings| {
        set(settings);
        settings.memory = Some(held + to_free / 2);
    });
    let freeing = freeing_beside_a_merge(&node).await;
    let builds = node.merge_builds().clone();
    // Past the least a tenant is kept for, so that it could be let go.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    node.sweep_now().await;
    let merges = node.merges();
    assert_eq!(merges.stopped_for_budget, 0, "a merge was stopped");
    assert_eq!(merges.draining, 0, "a merge was stopped");
    assert_eq!(builds.held(), 1, "the other merge did not go on");
    assert_eq!(node.census().holding.writers, 3, "a tenant was let go");
    assert!(node.freeing() > 0, "the free is over");
    // Nor does warming see it: what the tenants hold is what their states
    // come to.
    let mut figures = 0;
    for name in ["idle", "freeing", "merging"] {
        let tenant = node.tenant(&tenant(name)).await.unwrap();
        figures += tenant.partition().figures().resident;
    }
    assert_eq!(node.resident(), figures);
    builds.hold(None);
    drop(freeing);
}

/// Rule 4b: over its memory budget, a node stops the merge of a tenant it
/// would let go but cannot yet, as its maintenance holds it; the build
/// ends at once, not when it would have, and the node lets the tenant go
/// at a later round. What it counts, held and draining, falls.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn over_its_budget_a_node_stops_a_tenant_s_merge_and_lets_it_go_once_the_build_ends() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.memory = Some(1);
        // Swept every second; let go for being idle only after the budget
        // has let it go.
        settings.tenant_idle = Duration::from_secs(4);
    });
    let builds = node.merge_builds().clone();
    // A build that runs on until its merge is stopped.
    builds.hold(Some(Holding::UntilStopped));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    // Unused for longer than any tenant is kept regardless.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    assert!(node.resident() > 0);
    node.sweeping();
    // The first sweep, a second from now, finds the node over its budget.
    until(
        3,
        "the merge's build ran on for over 2 s after the sweep that would let its tenant go",
        || builds.under_way() == 0,
    )
    .await;
    assert_eq!(
        builds.draining(),
        0,
        "a stopped merge that has ended is counted as draining"
    );
    assert!(
        node.merges().stopped_for_budget >= 1,
        "the merge stopped for the budget was not counted"
    );
    until(10, "the tenant was never let go", || node.resident() == 0).await;
}

/// A sweep cannot tell a request's hold on a tenant from its maintenance's:
/// over its budget it stops a merge the request makes, but never takes the
/// tenant out from under the request, however many rounds go by. Once the
/// request ends, the tenant is let go.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_a_request_holds_is_not_let_go_while_its_merge_is_stopped() {
    let node = node(|settings| {
        // Flushed only as the request asks.
        settings.flush_rows = 1_000;
        settings.memory = Some(1);
        settings.tenant_idle = Duration::from_secs(4);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilStopped));
    node.create_tenant(&tenant("asked")).await.unwrap();
    let held = node.tenant(&tenant("asked")).await.unwrap();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        held.flush().await.unwrap();
    }
    // A request that merges the tenant, and goes on holding it after.
    let (merged, compacted) = tokio::sync::oneshot::channel();
    let go_on = Arc::new(tokio::sync::Notify::new());
    let request = tokio::spawn({
        let go_on = go_on.clone();
        async move {
            let compacted = held.compact(Duration::from_secs(60 * 60)).await;
            merged.send(compacted.map(|_| ())).unwrap();
            go_on.notified().await;
            held.write(vec![note(1_000)]).await.map(|_| ())
        }
    });
    until(30, "the request's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    node.sweeping();
    let compacted = within(5, "the request's merge was never stopped", compacted).await;
    compacted.unwrap().expect("a stopped merge is no failure");
    // Round after round, a sweep a second.
    tokio::time::sleep(Duration::from_secs(3)).await;
    assert!(
        node.resident() > 0,
        "a tenant was let go while a request held it"
    );
    // Re-review of #165, L2: one merge was stopped, though every round
    // since found the tenant held and asked again.
    assert_eq!(
        node.merges().stopped_for_budget,
        1,
        "stops counted that stopped no merge"
    );
    assert!(node.tenants_kept_by_sweeps() >= 3, "kept round after round");
    go_on.notify_one();
    let written = within(10, "the request never ended", request)
        .await
        .unwrap();
    written.expect("the request lost its tenant part way");
    until(
        10,
        "the tenant was never let go once the request ended",
        || node.resident() == 0,
    )
    .await;
}

/// Rule 7: a node builds at most `MERGE_BUILDS` merges at once, and a
/// stopped one whose build runs on is still among them: no further merge
/// gathers beside it until it ends. What it was reckoned to hold counts as
/// draining until then, and nothing once every build has ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_node_builds_at_most_merge_builds_at_once_a_stopped_one_among_them_until_it_ends() {
    assert_eq!(MERGE_BUILDS, 2, "the rule is written for two");
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    // Builds slow to come to where they ask their stop.
    builds.hold(Some(Holding::UntilLetGo));
    let most = Arc::new(AtomicUsize::new(0));
    let counting = Arc::new(AtomicBool::new(true));
    let watching = tokio::spawn({
        let (builds, most, counting) = (builds.clone(), most.clone(), counting.clone());
        async move {
            while counting.load(Ordering::SeqCst) {
                // Builds running in their closures, held or not, and those
                // counted under way.
                most.fetch_max(builds.held(), Ordering::SeqCst);
                most.fetch_max(builds.under_way(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    });
    four_flushes(&node, "first").await;
    four_flushes(&node, "second").await;
    until(30, "two merges never came to build", || builds.held() == 2).await;
    four_flushes(&node, "third").await;
    // The third waits for a turn: neither built nor counted.
    for _ in 0..50 {
        assert_eq!(builds.held(), 2, "a third merge built beside two");
        assert_eq!(builds.under_way(), 2, "a third merge took a turn");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Letting the first go stops its merge and drops it; its build, held,
    // runs on.
    node.drop_tenant(&tenant("first")).await.unwrap();
    until(5, "the stopped merge was never counted as draining", || {
        builds.draining() > 0
    })
    .await;
    for _ in 0..100 {
        assert!(
            builds.held() <= 2 && builds.under_way() <= 2,
            "a third merge built while a stopped one still ran"
        );
        assert!(
            builds.draining() > 0,
            "a stopped build that runs on stopped draining"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    builds.hold(None);
    let third = node.tenant(&tenant("third")).await.unwrap();
    let merged = || third.read(|state| state.segments().count()) == 1;
    until(60, "the third merge never committed", merged).await;
    until(30, "a build was left counted under way", || {
        builds.under_way() == 0
    })
    .await;
    assert_eq!(builds.draining(), 0, "a stopped build was left draining");
    counting.store(false, Ordering::SeqCst);
    watching.await.unwrap();
    assert!(
        most.load(Ordering::SeqCst) <= MERGE_BUILDS,
        "{} merges were built at once",
        most.load(Ordering::SeqCst)
    );
}

/// What a stopped merge still drains counts against the budget: a node
/// within it by what its tenants hold lets another go while a tenant it
/// gave up has a build that runs on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn what_a_stopped_merge_drains_counts_against_the_budget() {
    let few = || -> Vec<Mutation> {
        std::iter::once(notes())
            .chain((0..NOTES / 2).map(|n| note(10_000 + n)))
            .collect()
    };
    // What a tenant of a few notes, unflushed, holds.
    let sizing = node(|settings| settings.flush_rows = NOTES as usize);
    sizing.create_tenant(&tenant("kept")).await.unwrap();
    sizing
        .tenant(&tenant("kept"))
        .await
        .unwrap()
        .write(few())
        .await
        .unwrap();
    let one = sizing.resident();
    assert!(one > 0);

    // Just within the budget with that tenant alone. Swept every five
    // seconds; let go for being idle only after twenty, well after the
    // rule is done.
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.memory = Some(one);
        settings.tenant_idle = Duration::from_secs(20);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    node.create_tenant(&tenant("kept")).await.unwrap();
    node.tenant(&tenant("kept"))
        .await
        .unwrap()
        .write(few())
        .await
        .unwrap();
    four_flushes(&node, "dropped").await;
    until(30, "the merge never came to build", || builds.held() == 1).await;
    // Given up: no longer held, its build runs on, draining.
    node.drop_tenant(&tenant("dropped")).await.unwrap();
    until(5, "the stopped merge was never counted as draining", || {
        builds.draining() > 0
    })
    .await;
    assert_eq!(node.resident(), one, "the tenant kept holds what it held");
    // Used now, however long the rule took to come here: let go for being
    // idle only well after the rule is done.
    node.tenant(&tenant("kept")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    node.sweeping();
    until(
        8,
        "a tenant was not let go for what a stopped merge drains",
        || node.resident() == 0,
    )
    .await;
    builds.hold(None);
    until(30, "the stopped build was never let go", || {
        builds.under_way() == 0
    })
    .await;
    assert_eq!(builds.draining(), 0);
}

/// A tenant's flushes never wait behind other tenants' merges: with every
/// turn held by other tenants' builds, a tenant written to steadily, whose
/// own merge waits for a turn, goes on flushing, and none of its writes is
/// refused for its backlog. Its merge is made once a turn is free.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_s_flushes_never_wait_behind_other_tenants_merges() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "first").await;
    four_flushes(&node, "second").await;
    until(30, "two merges never came to build", || builds.held() == 2).await;
    // Four flushes: its merge is due, and waits for a turn.
    four_flushes(&node, "written").await;
    let written = node.tenant(&tenant("written")).await.unwrap();
    let segments = || written.read(|state| state.segments().count());
    let before = segments();
    // Fifteen flushes' worth over three seconds, steadily: the backlog,
    // four flushes' worth, is reached in under a second if none is made.
    for round in 0..30 {
        let batch: Vec<Mutation> = (0..NOTES / 2)
            .map(|n| note(1_000 + round * NOTES + n))
            .collect();
        let wrote = written.write(batch).await;
        assert!(
            wrote.is_ok(),
            "a write was refused while other tenants' merges held every turn: {wrote:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(
        segments() >= before + 10,
        "the tenant flushed {} times while other tenants' merges held every turn",
        segments() - before
    );
    assert_eq!(builds.held(), 2, "the tenant's merge built without a turn");
    builds.hold(None);
    until(
        60,
        "the tenant's merge was never made once a turn was free",
        || segments() == 1,
    )
    .await;
    until(30, "a build was left counted under way", || {
        builds.under_way() == 0
    })
    .await;
}

/// Over its budget, a node lets go of what it can take out before it stops
/// any merge: a tenant mid-merge that cannot be taken out, the least
/// recently used, keeps its merge when letting another go is enough.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn over_its_budget_a_node_stops_no_merge_when_letting_others_go_is_enough() {
    let few = || -> Vec<Mutation> {
        std::iter::once(notes())
            .chain((0..NOTES / 2).map(|n| note(10_000 + n)))
            .collect()
    };
    // What the two hold together: one mid-merge, one with a few notes.
    let sizing = node(|settings| settings.flush_rows = NOTES as usize);
    sizing.merge_builds().hold(Some(Holding::UntilLetGo));
    four_flushes(&sizing, "merging").await;
    sizing.create_tenant(&tenant("idle")).await.unwrap();
    let idle = sizing.tenant(&tenant("idle")).await.unwrap();
    idle.write(few()).await.unwrap();
    drop(idle);
    let both = sizing.resident();
    sizing.merge_builds().hold(None);

    // Over the budget with both, within it with either.
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.memory = Some(both - 1);
        // Swept every two seconds; let go for being idle only well after
        // the rule is done.
        settings.tenant_idle = Duration::from_secs(8);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilStopped));
    four_flushes(&node, "merging").await;
    until(30, "the merge never came to build", || builds.held() == 1).await;
    node.create_tenant(&tenant("idle")).await.unwrap();
    node.tenant(&tenant("idle"))
        .await
        .unwrap()
        .write(few())
        .await
        .unwrap();
    assert_eq!(node.resident(), both);
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    node.sweeping();
    until(4, "the tenant that could go was never let go", || {
        node.resident() < both
    })
    .await;
    // A round more: the merge goes on.
    for _ in 0..250 {
        assert_eq!(
            builds.held(),
            1,
            "a merge was stopped though letting another tenant go was enough"
        );
        assert_eq!(builds.draining(), 0);
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    builds.hold(None);
    until(60, "the merge was never made", || builds.under_way() == 0).await;
}

/// Losing its hold on a tenant stops a merge that a request makes, which
/// none of the hold's own tasks runs and losing the hold aborts nothing of:
/// the request has its answer at once, not once the build would have ended,
/// and the build's turn is free.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn losing_a_tenant_stops_a_merge_a_request_makes() {
    let node = node(|settings| {
        // Flushed only as the request asks.
        settings.flush_rows = 1_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    // A build that runs on until its merge is stopped.
    builds.hold(Some(Holding::UntilStopped));
    node.create_tenant(&tenant("asked")).await.unwrap();
    let held = node.tenant(&tenant("asked")).await.unwrap();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        held.flush().await.unwrap();
    }
    let request = tokio::spawn(async move {
        // Whatever it answers once the tenant is gone: it answers.
        let _ = held.compact(Duration::from_secs(60 * 60)).await;
    });
    until(30, "the request's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    node.drop_tenant(&tenant("asked")).await.unwrap();
    within(
        5,
        "a merge a request makes ran on once its tenant was lost",
        request,
    )
    .await
    .unwrap();
    until(5, "the stopped build's turn was never free", || {
        builds.under_way() == 0 && builds.turns_free() == MERGE_BUILDS
    })
    .await;
}

/// Write `held` steadily, half a flush every tenth of a second, until
/// `writing` is cleared: each write must go through.
fn steadily(
    held: Arc<vellev_node::Tenant>,
    writing: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut round = 0;
        while writing.load(Ordering::SeqCst) {
            let batch: Vec<Mutation> = (0..NOTES / 2)
                .map(|n| note(100_000 + round * NOTES + n))
                .collect();
            let wrote = held.write(batch).await;
            assert!(wrote.is_ok(), "a write was refused: {wrote:?}");
            round += 1;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
}

/// A tenant written steadily, whose merge gives way to each flush that
/// falls due while it waits for a turn, keeps its place: the next turn to
/// come free is its, not that of a merge that queued behind it, however
/// often it gives way. Its writes go on, and through, all the while.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_written_steadily_keeps_its_place_for_a_merge_turn() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "first").await;
    four_flushes(&node, "second").await;
    until(30, "two merges never came to build", || builds.held() == 2).await;
    // Its merge queues first, then gives way to flush after flush.
    four_flushes(&node, "written").await;
    let written = node.tenant(&tenant("written")).await.unwrap();
    let key = written.partition().key().clone();
    let writing = Arc::new(AtomicBool::new(true));
    let writes = steadily(written.clone(), writing.clone());
    tokio::time::sleep(Duration::from_millis(500)).await;
    // Two more merges queue behind it.
    four_flushes(&node, "behind").await;
    four_flushes(&node, "further").await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    // One turn comes free: the written tenant's.
    builds.let_one_go();
    until(
        10,
        "the next turn went to a merge that queued behind one that gave way",
        || builds.under_way_for(&key),
    )
    .await;
    // The rest of the merges go on, and the writes with them.
    builds.hold(None);
    let merged = || written.read(|state| state.segments().count()) <= 2;
    until(60, "the written tenant's merge was never made", merged).await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    writing.store(false, Ordering::SeqCst);
    within(10, "the writes never ended", writes).await.unwrap();
    until(60, "a build was left counted under way", || {
        builds.under_way() == 0
    })
    .await;
}

/// A store whose listings, once armed, wait until it is disarmed: as
/// cleaning up waits on a slow store.
#[derive(Debug)]
struct SlowListing {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
}

#[async_trait::async_trait]
impl Blobs for SlowListing {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        while self.armed.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// A maintenance round whose merge gives way to a flush does nothing more:
/// it doesn't clean up, which waits on the store, before the flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_round_whose_merge_gives_way_to_a_flush_does_nothing_more() {
    let slow = Arc::new(SlowListing {
        inner: Arc::new(ObjectBlobs::new(Arc::new(
            object_store::memory::InMemory::new(),
        ))),
        armed: AtomicBool::new(false),
    });
    let node = node_on(slow.clone(), |settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        // Cleaning up is not due again for an hour.
        settings.clean_at_least_every = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "first").await;
    four_flushes(&node, "second").await;
    until(30, "two merges never came to build", || builds.held() == 2).await;
    four_flushes(&node, "written").await;
    let written = node.tenant(&tenant("written")).await.unwrap();
    let before = written.read(|state| state.segments().count());
    slow.armed.store(true, Ordering::SeqCst);
    let writing = Arc::new(AtomicBool::new(true));
    let writes = steadily(written.clone(), writing.clone());
    tokio::time::sleep(Duration::from_secs(3)).await;
    writing.store(false, Ordering::SeqCst);
    within(10, "the writes never ended", writes).await.unwrap();
    let flushed = written.read(|state| state.segments().count()) - before;
    assert!(
        flushed >= 10,
        "the tenant flushed {flushed} times while its rounds' cleaning up waited on the store"
    );
    slow.armed.store(false, Ordering::SeqCst);
    builds.hold(None);
    until(60, "a build was left counted under way", || {
        builds.under_way() == 0
    })
    .await;
}

/// A store that counts its listings: cleaning up lists the partition.
#[derive(Debug)]
struct CountedListing {
    inner: Arc<dyn Blobs>,
    listed: AtomicUsize,
}

#[async_trait::async_trait]
impl Blobs for CountedListing {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.listed.fetch_add(1, Ordering::SeqCst);
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// Under writes that never let up, whose flushes a tenant's merge gives
/// way to round after round, its maintenance still cleans up, at least as
/// often as the node is set to.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn under_writes_that_never_let_up_a_tenant_is_still_cleaned_up() {
    let counted = Arc::new(CountedListing {
        inner: Arc::new(ObjectBlobs::new(Arc::new(
            object_store::memory::InMemory::new(),
        ))),
        listed: AtomicUsize::new(0),
    });
    let node = node_on(counted.clone(), |settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        settings.clean_at_least_every = Duration::from_secs(1);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "first").await;
    four_flushes(&node, "second").await;
    until(30, "two merges never came to build", || builds.held() == 2).await;
    four_flushes(&node, "written").await;
    let written = node.tenant(&tenant("written")).await.unwrap();
    let writing = Arc::new(AtomicBool::new(true));
    let writes = steadily(written.clone(), writing.clone());
    tokio::time::sleep(Duration::from_millis(500)).await;
    let before = counted.listed.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs(4)).await;
    let cleaned = counted.listed.load(Ordering::SeqCst) - before;
    writing.store(false, Ordering::SeqCst);
    within(10, "the writes never ended", writes).await.unwrap();
    assert!(
        cleaned >= 1,
        "a tenant written without let-up was never cleaned up in 4 s"
    );
    builds.hold(None);
    until(60, "a build was left counted under way", || {
        builds.under_way() == 0
    })
    .await;
}
