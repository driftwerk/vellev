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
use vellev_node::{MergesStopped, Node, NodeError, Settings};
use vellev_storage::blobs::{Blobs, Listed, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator};
use vellev_storage::{Holding, MergeGate, Mutation, State, MERGE_BUILDS};

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
    // Past the least a tenant is kept for, so that it could be let go: all
    // but "freeing", used now, which were it among those chosen would cover
    // what is over, and leave the other merge be.
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    node.tenant(&tenant("freeing")).await.unwrap();
    node.sweep_now().await;
    // The census is taken as a sweep begins: a second one finds what the
    // first let go.
    node.sweep_now().await;
    let merges = node.merges();
    assert_eq!(
        (node.census().holding.writers, merges.stopped_for_budget),
        (3, 0),
        "tenants held and merges stopped"
    );
    assert_eq!(merges.draining, 0, "a merge was stopped");
    assert_eq!(builds.held(), 1, "the other merge did not go on");
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

/// A sweep takes a request's hold on a tenant as it does its maintenance's:
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

/// Tenant `name`, with a request merging it, held where its build holds it
/// (see [`MergeBuilds::hold`]), and that request.
async fn merged_by_a_request(
    node: &Node,
    name: &str,
) -> (Arc<vellev_node::Tenant>, tokio::task::JoinHandle<()>) {
    let builds = node.merge_builds().clone();
    node.create_tenant(&tenant(name)).await.unwrap();
    let held = node.tenant(&tenant(name)).await.unwrap();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        held.flush().await.unwrap();
    }
    let request = tokio::spawn({
        let held = held.clone();
        async move {
            // Whatever it answers once the merge is stopped: it answers.
            let _ = held.compact(Duration::from_secs(60 * 60)).await;
        }
    });
    until(30, "the request's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    (held, request)
}

/// M7-R1, as far as the split shows it: a tenant's maintenance works on what
/// its tasks need, not on the tenant requests hold. So in the middle of a
/// merge, nothing holds the tenant but the node and whoever asks here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_s_maintenance_holds_no_tenant_while_it_merges() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    let held = node.tenant(&tenant("merging")).await.unwrap();
    assert_eq!(
        Arc::strong_count(&held),
        2,
        "held by more than the node and this rule while its merge builds"
    );
    builds.hold(None);
    until(60, "the merge was never made", || builds.under_way() == 0).await;
}

/// What the split gives on its own: nothing of a tenant's maintenance keeps
/// the tenant, so that once nothing else holds it, its merge is stopped at
/// once, where it would have kept the tenant until it ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_nothing_holds_any_more_has_its_merge_stopped() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    // A build that runs on until its merge is stopped.
    builds.hold(Some(Holding::UntilStopped));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    drop(node);
    until(5, "a merge outlived all that held its tenant", || {
        builds.under_way() == 0
    })
    .await;
}

/// M7-R6, on the idle path, and so for as long as a round keeps its tenant:
/// a tenant unused for longer than the node keeps one, with its merge
/// building, is kept until the merge ends, and let go at a sweep after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_idle_while_it_merges_is_kept_until_its_merge_ends() {
    let idle = Duration::from_secs(2);
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = idle;
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    tokio::time::sleep(idle + Duration::from_millis(200)).await;
    // The census is taken as a sweep begins: a second one finds what the
    // first let go.
    node.sweep_now().await;
    node.sweep_now().await;
    assert_eq!(node.census().holding.writers, 1, "let go mid-merge");
    assert_eq!(builds.held(), 1, "its merge was stopped");
    assert_eq!(builds.draining(), 0, "its merge was stopped");
    builds.hold(None);
    node.sweeping();
    until(
        20,
        "the tenant was never let go once its merge ended",
        || node.resident() == 0,
    )
    .await;
    assert_eq!(builds.under_way(), 0);
}

/// M7-R4, at the node: a merge is counted as stopped once, by what first
/// stopped it. A tenant dropped mid-merge, with the build held, and a
/// request that goes on writing finds the hold lost, and loses it again,
/// 20 times over.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_stopped_again_and_again_is_counted_once_as_what_first_stopped_it() {
    let node = node(|settings| {
        // Flushed only as the request asks.
        settings.flush_rows = 1_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    let (held, request) = merged_by_a_request(&node, "asked").await;
    node.drop_tenant(&tenant("asked")).await.unwrap();
    for n in 0..20 {
        let wrote = held.write(vec![note(1_000 + n)]).await;
        assert!(
            matches!(wrote, Err(NodeError::Storage(_))),
            "a write after the hold was lost: {wrote:?}"
        );
    }
    assert_eq!(builds.held(), 1, "the build held is held still");
    assert_eq!(
        node.merges().stopped,
        MergesStopped {
            drop: 1,
            ..MergesStopped::default()
        }
    );
    builds.hold(None);
    within(30, "the request never ended", request)
        .await
        .unwrap();
}

/// M7-R4: a merge stopped by a lease lost is counted as that, and the same
/// on a node given up, each once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_stopped_by_a_lease_lost_or_a_shutdown_is_counted_as_that() {
    let lease = Duration::from_secs(1);
    let coordinator: Arc<dyn Coordinator> = Arc::new(MemoryCoordinator::default());
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(
        object_store::memory::InMemory::new(),
    )));
    let on_both = || {
        let settings = Settings {
            warm: None,
            lease,
            // Flushed only as the request asks.
            flush_rows: 1_000,
            tenant_idle: Duration::from_secs(60 * 60),
            ..Settings::new(NodeId::generate("node"))
        };
        let embedders = Arc::new(Embedders::new(EmbedSettings::default()));
        Node::new(coordinator.clone(), blobs.clone(), embedders, settings)
    };
    let (node, other) = (on_both(), on_both());
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    let (_held, leased) = merged_by_a_request(&node, "leased").await;
    // Removed elsewhere: this node finds out as it next renews its lease.
    other.drop_tenant(&tenant("leased")).await.unwrap();
    until(5, "the lost lease stopped no merge", || {
        node.merges().stopped.lease == 1
    })
    .await;
    let lost = MergesStopped {
        lease: 1,
        ..MergesStopped::default()
    };
    assert_eq!(node.merges().stopped, lost);
    builds.hold(None);
    within(30, "the request never ended", leased).await.unwrap();
    until(5, "the stopped build's turn was never free", || {
        builds.under_way() == 0
    })
    .await;

    builds.hold(Some(Holding::UntilLetGo));
    let (_held, shutting) = merged_by_a_request(&node, "shutting").await;
    node.shutdown().await;
    assert_eq!(
        node.merges().stopped,
        MergesStopped {
            shutdown: 1,
            ..lost
        }
    );
    builds.hold(None);
    within(30, "the request never ended", shutting)
        .await
        .unwrap();
}

/// M7-R4, for the sweep's own stop: every sweep that finds the tenant held
/// asks again, with the build held throughout, and the merge is counted
/// once, there and nowhere else, whatever goes on to stop it again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_stopped_for_the_budget_is_counted_once_and_nowhere_else() {
    let node = node(|settings| {
        // Flushed only as the request asks.
        settings.flush_rows = 1_000;
        settings.memory = Some(1);
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    let (held, request) = merged_by_a_request(&node, "asked").await;
    // Unused for longer than any tenant is kept regardless.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    for _ in 0..3 {
        node.sweep_now().await;
    }
    assert!(builds.draining() > 0, "the merge was not stopped");
    assert_eq!(builds.held(), 1, "the build held is held still");
    assert_eq!(
        node.merges().stopped_for_budget,
        1,
        "one merge, three sweeps"
    );
    node.drop_tenant(&tenant("asked")).await.unwrap();
    assert_eq!(node.merges().stopped, MergesStopped::default());
    builds.hold(None);
    within(30, "the request never ended", request)
        .await
        .unwrap();
    drop(held);
}

/// Tenant `name`, its maintenance merging it, that merge held past loading
/// back what it built, where a stop no longer ends it. The tenant is not
/// held after.
async fn past_loading(node: &Node, name: &str) -> Arc<MergeGate> {
    node.create_tenant(&tenant(name)).await.unwrap();
    let held = node.tenant(&tenant(name)).await.unwrap();
    let gate = held.hold_merges_before_commit();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        let flushed = || held.read(State::unflushed) == 0;
        until(30, "the node never flushed the tenant", flushed).await;
    }
    within(
        30,
        "the merge never came past loading",
        gate.reached.notified(),
    )
    .await;
    gate
}

/// Two nodes on one coordinator and store, flushing every `NOTES` rows, with
/// a lease of a second, so that either learns of a tenant removed by the
/// other within a third of that.
fn two_nodes() -> (Node, Node) {
    let coordinator: Arc<dyn Coordinator> = Arc::new(MemoryCoordinator::default());
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(
        object_store::memory::InMemory::new(),
    )));
    let on_both = || {
        let settings = Settings {
            warm: None,
            lease: Duration::from_secs(1),
            flush_rows: NOTES as usize,
            tenant_idle: Duration::from_secs(60 * 60),
            ..Settings::new(NodeId::generate("node"))
        };
        let embedders = Arc::new(Embedders::new(EmbedSettings::default()));
        Node::new(coordinator.clone(), blobs.clone(), embedders, settings)
    };
    (on_both(), on_both())
}

/// A stop does not end a merge past loading, but aborting its task does: it
/// is lost all the same, the dearest there is to lose, and is counted, once,
/// as what lost it. For a tenant dropped, a renewal refused and a shutdown.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_past_loading_cut_off_with_its_task_is_counted_as_what_cut_it() {
    let (node, other) = two_nodes();
    let builds = node.merge_builds().clone();
    let _gate = past_loading(&node, "dropped").await;
    assert!(node.drop_tenant(&tenant("dropped")).await.unwrap());
    let mut counted = MergesStopped {
        drop: 1,
        ..MergesStopped::default()
    };
    assert_eq!(node.merges().stopped, counted);
    until(5, "the merge cut off left its build under way", || {
        builds.under_way() == 0
    })
    .await;

    let _gate = past_loading(&node, "leased").await;
    other.drop_tenant(&tenant("leased")).await.unwrap();
    until(5, "the refused renewal counted nothing", || {
        node.merges().stopped.lease == 1
    })
    .await;
    counted.lease = 1;
    assert_eq!(node.merges().stopped, counted);
    until(5, "the merge cut off left its build under way", || {
        builds.under_way() == 0
    })
    .await;

    let _gate = past_loading(&node, "shutting").await;
    node.shutdown().await;
    counted.shutdown = 1;
    assert_eq!(node.merges().stopped, counted);
}

/// Losing a hold aborts nothing of a request's: its merge past loading goes
/// on to its commit, so that no stop ended it, and none is counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_past_loading_on_a_request_is_not_counted_when_the_hold_is_lost() {
    let node = node(|settings| {
        // Flushed only as the request asks.
        settings.flush_rows = 1_000;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    node.create_tenant(&tenant("asked")).await.unwrap();
    let held = node.tenant(&tenant("asked")).await.unwrap();
    let gate = held.hold_merges_before_commit();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        held.flush().await.unwrap();
    }
    let request = tokio::spawn({
        let held = held.clone();
        async move { held.compact(Duration::from_secs(60 * 60)).await }
    });
    within(
        30,
        "the merge never came past loading",
        gate.reached.notified(),
    )
    .await;
    node.drop_tenant(&tenant("asked")).await.unwrap();
    assert_eq!(node.merges().stopped, MergesStopped::default());
    gate.release.notify_one();
    let answered = within(30, "the request never ended", request).await;
    // Refused or not, it ran to its end, with nothing to stop it.
    let _ = answered.unwrap();
    assert_eq!(node.merges().stopped, MergesStopped::default());
}

/// The sweep's stop aborts nothing either. One too late for a merge stops
/// nothing, and counts nothing, and the merge commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_for_the_budget_too_late_counts_nothing_and_the_merge_commits() {
    let node = node(|settings| {
        // Flushed only as the request asks.
        settings.flush_rows = 1_000;
        settings.memory = Some(1);
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    node.create_tenant(&tenant("asked")).await.unwrap();
    let held = node.tenant(&tenant("asked")).await.unwrap();
    let gate = held.hold_merges_before_commit();
    held.write(vec![notes()]).await.unwrap();
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        held.write(batch).await.unwrap();
        held.flush().await.unwrap();
    }
    let request = tokio::spawn({
        let held = held.clone();
        async move { held.compact(Duration::from_secs(60 * 60)).await }
    });
    within(
        30,
        "the merge never came past loading",
        gate.reached.notified(),
    )
    .await;
    // Unused for longer than any tenant is kept regardless.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    node.sweep_now().await;
    assert_eq!(node.merges().stopped_for_budget, 0, "a stop too late");
    gate.release.notify_one();
    let answered = within(30, "the request never ended", request).await;
    answered.unwrap().expect("the merge commits");
    assert_eq!(held.read(|state| state.segments().count()), 1, "merged");
    assert_eq!(node.merges().stopped_for_budget, 0);
}

/// A merge the sweep's stop came too late for, cut off with its task after,
/// is one merge, counted once, as what cut it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_too_late_for_the_budget_and_then_cut_off_is_counted_once() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.memory = Some(1);
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let _gate = past_loading(&node, "merging").await;
    // Unused for longer than any tenant is kept regardless.
    tokio::time::sleep(Duration::from_millis(1_100)).await;
    node.sweep_now().await;
    assert_eq!(node.merges().stopped_for_budget, 0, "a stop too late");
    node.drop_tenant(&tenant("merging")).await.unwrap();
    let merges = node.merges();
    assert_eq!(
        (merges.stopped_for_budget, merges.stopped),
        (
            0,
            MergesStopped {
                drop: 1,
                ..MergesStopped::default()
            }
        )
    );
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

/// A tenant written steadily, whose flushes go on as its merge waits for a
/// turn, keeps its place: the next turn to come free is its, not that of a
/// merge that queued behind it, however often it flushes. Its writes go on,
/// and through, all the while.
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

/// A tenant's flushes go on whatever its round waits on. Written when a
/// round's merge gave way to a flush, and did nothing more: since M5 no merge
/// of a node gives way, and the round waits for its turn, cleaning up not
/// being overdue. Giving way's own rules are in the storage crate.
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

/// Under writes that never let up, with a tenant's merge queued for a turn,
/// its maintenance still cleans up, at least as often as the node is set to.
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

/// M5-R1 and M5-R2: with a merge of a tenant's held where `holding` says, rows
/// falling due are flushed within a flush's own time, not the build's, and
/// no write is refused up to its backlog.
async fn a_flush_goes_on_beside(holding: Holding) {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(holding));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to its hold", || {
        builds.held() == 1
    })
    .await;
    let held = node.tenant(&tenant("merging")).await.unwrap();
    let before = held.read(|state| state.segments().count());
    for round in 0..8 {
        let batch: Vec<Mutation> = (0..NOTES)
            .map(|n| note(1_000 + round * NOTES + n))
            .collect();
        held.write(batch).await.unwrap();
        until(5, "a flush due was held behind its tenant's merge", || {
            held.read(State::unflushed) == 0
        })
        .await;
    }
    assert_eq!(
        held.read(|state| state.segments().count()),
        before + 8,
        "flushed through the build"
    );
    assert_eq!(builds.held(), 1, "the merge ended for the flushes");
    assert_eq!(node.flushing().backlogged, 0, "a write was refused");
    builds.hold(None);
    until(60, "a build was left under way", || builds.under_way() == 0).await;
}

/// M5-R1: rule 1 of the build priority note's §2.5, end to end: a flush due
/// while its tenant's merge builds commits within the flush's own time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_due_as_its_tenant_s_merge_builds_commits_in_its_own_time() {
    a_flush_goes_on_beside(Holding::UntilLetGo).await;
}

/// M5-R2: and as its gather copies rows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_due_as_its_tenant_s_merge_gathers_commits_in_its_own_time() {
    a_flush_goes_on_beside(Holding::GatheringUntilLetGo).await;
}

/// M5-R3: a flush that lands during a round is not what that round read:
/// it is followed by a round of its own, for what it may have made due, and
/// the tenant stays unsettled until that has gone through.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_landing_during_a_round_is_followed_by_a_round() {
    let node = node(|settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilLetGo));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    let held = node.tenant(&tenant("merging")).await.unwrap();
    for round in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES)
            .map(|n| note(1_000 + round * NOTES + n))
            .collect();
        held.write(batch).await.unwrap();
        until(10, "the flush never landed", || {
            held.read(State::unflushed) == 0
        })
        .await;
    }
    assert_eq!(
        node.unsettled(),
        1,
        "a round is under way, and four flushes"
    );
    builds.hold(None);
    until(30, "no round followed the flushes", || {
        node.unsettled() == 0
    })
    .await;
    assert_eq!(
        held.read(|state| state.segments().count()),
        1,
        "settled with a merge due"
    );
}

/// A store whose deletes, once armed, wait until it is disarmed, noting one
/// dropped where it waits, as a clean cut part way drops it.
#[derive(Debug, Default)]
struct DeletesHeld {
    armed: AtomicBool,
    deleting: AtomicUsize,
    cut: AtomicUsize,
}

#[derive(Debug)]
struct HeldDeletes {
    inner: Arc<dyn Blobs>,
    held: Arc<DeletesHeld>,
}

/// Notes a delete dropped where it waits.
struct UnlessLanded<'a>(&'a AtomicUsize, bool);

impl Drop for UnlessLanded<'_> {
    fn drop(&mut self) {
        if !self.1 {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[async_trait::async_trait]
impl Blobs for HeldDeletes {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<bytes::Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        if self.held.armed.load(Ordering::SeqCst) {
            let mut unless = UnlessLanded(&self.held.cut, false);
            self.held.deleting.fetch_add(1, Ordering::SeqCst);
            while self.held.armed.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            unless.1 = true;
        }
        self.inner.delete(key).await
    }
}

/// A node cleaning its tenants at least every second, none of what it
/// cleans being kept for a restore, on a store that holds deletes; and a
/// tenant with a merge held building until it is stopped, and a clean beside
/// it that has come to delete what the flush after it left behind.
async fn cleaning_beside_a_merge() -> (Node, Arc<vellev_node::Tenant>, Arc<DeletesHeld>) {
    let held = Arc::new(DeletesHeld::default());
    let store = Arc::new(HeldDeletes {
        inner: Arc::new(ObjectBlobs::new(Arc::new(
            object_store::memory::InMemory::new(),
        ))),
        held: held.clone(),
    });
    let node = node_on(store, |settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        settings.clean_at_least_every = Duration::from_secs(1);
        settings.restore_window = Duration::ZERO;
        settings.grace = Duration::ZERO;
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::UntilStopped));
    four_flushes(&node, "merging").await;
    until(30, "the tenant's merge never came to build", || {
        builds.held() == 1
    })
    .await;
    held.armed.store(true, Ordering::SeqCst);
    let tenant_held = node.tenant(&tenant("merging")).await.unwrap();
    let batch: Vec<Mutation> = (0..NOTES).map(|n| note(1_000 + n)).collect();
    tenant_held.write(batch).await.unwrap();
    until(
        30,
        "the clean beside the merge never came to delete",
        || held.deleting.load(Ordering::SeqCst) > 0,
    )
    .await;
    (node, tenant_held, held)
}

/// A clean beside a tenant's merges is never cut, whatever ends them
/// meanwhile: it stops where it is safe to, its lot of deletes having
/// landed, and a round ends only once it has.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clean_under_way_as_its_tenant_s_merges_end_is_not_cut() {
    let (node, held, deletes) = cleaning_beside_a_merge().await;
    let builds = node.merge_builds().clone();
    held.stop_merges();
    until(10, "the stopped merge never ended", || {
        builds.under_way() == 0
    })
    .await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        deletes.cut.load(Ordering::SeqCst),
        0,
        "a clean was cut as the merges beside it ended"
    );
    deletes.armed.store(false, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(deletes.cut.load(Ordering::SeqCst), 0, "cut once let go");
}

/// What ends a tenant's tasks ends a clean beside its merges with them,
/// wherever it is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clean_under_way_ends_with_the_hold() {
    let (node, _held, deletes) = cleaning_beside_a_merge().await;
    node.shutdown().await;
    until(5, "a clean outlived the hold on its tenant", || {
        deletes.cut.load(Ordering::SeqCst) > 0
    })
    .await;
    deletes.armed.store(false, Ordering::SeqCst);
}

/// A tenant's merge queued for a turn other tenants' builds hold, however
/// long, holds neither its cleaning up nor its pass over its rows behind it,
/// as [`Settings::clean_at_least_every`] promises: quiet or written to.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_queued_for_a_turn_holds_nothing_of_its_round_behind_it() {
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
    four_flushes(&node, "quiet").await;
    let quiet = node.tenant(&tenant("quiet")).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let (listed, passes) = (counted.listed.load(Ordering::SeqCst), quiet.rows_passes());
    tokio::time::sleep(Duration::from_secs(4)).await;
    let cleaned = counted.listed.load(Ordering::SeqCst) - listed;
    assert!(cleaned >= 2, "cleaned {cleaned} times in 4 s, at 1 s apart");
    assert!(quiet.rows_passes() - passes >= 2, "no pass over its rows");
    assert_eq!(node.unsettled(), 3, "read as settled with its merge due");
    builds.hold(None);
    until(60, "a build was left counted under way", || {
        builds.under_way() == 0
    })
    .await;
}

/// A node whose flush task, once a flush has failed, tries no more for the
/// length of the rule, on a store that fails as many flushes as are armed:
/// and a tenant whose merge is held past loading, with a flush's worth of
/// rows left to it by a flush that failed.
async fn rows_left_to_a_merge() -> (
    Node,
    Arc<FailingFlushes>,
    Arc<vellev_node::Tenant>,
    Arc<MergeGate>,
) {
    let store = Arc::new(FailingFlushes {
        inner: Arc::new(ObjectBlobs::new(Arc::new(
            object_store::memory::InMemory::new(),
        ))),
        failing: AtomicUsize::new(0),
        holding: AtomicBool::new(false),
        held: AtomicUsize::new(0),
    });
    let node = node_on(store.clone(), |settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = Duration::from_secs(60 * 60);
        settings.flush_backoff = Duration::from_secs(60 * 60);
    });
    let gate = past_loading(&node, "merging").await;
    let held = node.tenant(&tenant("merging")).await.unwrap();
    store.failing.store(1, Ordering::SeqCst);
    let batch: Vec<Mutation> = (0..NOTES).map(|n| note(1_000 + n)).collect();
    held.write(batch).await.unwrap();
    until(10, "the flush never failed", || node.flushing().failed == 1).await;
    (node, store, held, gate)
}

/// A flush first that fails counts as any flush that fails, in the tenant's
/// backoff and the node's count, and leaves its rows to the commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_first_that_fails_counts_as_a_flush_that_failed() {
    let (node, store, held, gate) = rows_left_to_a_merge().await;
    store.failing.store(1, Ordering::SeqCst);
    gate.release.notify_one();
    until(30, "the flush first's failure was not counted", || {
        node.flushing().failed == 2
    })
    .await;
    until(30, "the merge never flushed its rows along", || {
        held.read(State::unflushed) == 0
    })
    .await;
    assert_eq!(node.flushing().failed, 2);
}

/// A flush first that goes through ends the tenant's backoff, as any flush
/// that goes through. The round it was made in follows it, as it follows a
/// flush along: it leaves the tenant settled in moments, with no round of
/// its own to wait for, and nothing owed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_first_that_goes_through_leaves_its_tenant_settled() {
    let (node, _store, held, gate) = rows_left_to_a_merge().await;
    assert_eq!(node.unsettled(), 1);
    gate.release.notify_one();
    until(10, "a flush first left its tenant unsettled", || {
        node.unsettled() == 0
    })
    .await;
    assert_eq!(held.read(State::unflushed), 0);
    assert_eq!(node.flushing().failed, 1);
    let batch: Vec<Mutation> = (0..NOTES).map(|n| note(2_000 + n)).collect();
    held.write(batch).await.unwrap();
    until(10, "the backoff outlived the flush first", || {
        held.read(State::unflushed) == 0
    })
    .await;
}

/// The merge of [`rows_left_to_a_merge`] let go to its commit, with its flush
/// first held part way.
async fn a_flush_first_held_part_way(store: &FailingFlushes, gate: &MergeGate) {
    store.holding.store(true, Ordering::SeqCst);
    gate.release.notify_one();
    until(30, "the flush first never began to go up", || {
        store.held.load(Ordering::SeqCst) == 1
    })
    .await;
}

/// A stop comes too late for a merge flushing first, past where one ends it:
/// the flush goes on, and the merge commits, as it would have at its commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_during_a_flush_first_ends_neither_it_nor_its_merge() {
    let (node, store, held, gate) = rows_left_to_a_merge().await;
    a_flush_first_held_part_way(&store, &gate).await;
    held.stop_merges();
    tokio::time::sleep(Duration::from_millis(200)).await;
    store.holding.store(false, Ordering::SeqCst);
    until(
        30,
        "the merge stopped during its flush first never ended",
        || node.unsettled() == 0,
    )
    .await;
    assert_eq!(held.read(State::unflushed), 0);
    assert_eq!(
        held.read(|state| state.segments().count()),
        2,
        "the merge was lost: its segment and the flush's are all there should be"
    );
    assert_eq!(node.merges().stopped, MergesStopped::default());
}

/// Losing the hold cuts a merge flushing first with its task, as it cuts one
/// past loading anywhere: counted once, its build let go, and the rows of its
/// flush, which went nowhere, read again by the next to open the tenant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hold_lost_during_a_flush_first_ends_it_with_its_merge_counted_once() {
    let (node, store, _held, gate) = rows_left_to_a_merge().await;
    a_flush_first_held_part_way(&store, &gate).await;
    let builds = node.merge_builds().clone();
    node.shutdown().await;
    let counted = MergesStopped {
        shutdown: 1,
        ..MergesStopped::default()
    };
    assert_eq!(node.merges().stopped, counted);
    until(5, "the merge cut off left its build under way", || {
        builds.under_way() == 0
    })
    .await;
    assert_eq!(node.merges().stopped, counted);
    store.holding.store(false, Ordering::SeqCst);
}

/// A store whose next writes of flushed vectors, as many as are armed, fail.
#[derive(Debug)]
struct FailingFlushes {
    inner: Arc<dyn Blobs>,
    failing: AtomicUsize,
    /// Whether writes of flushed vectors wait until it is cleared.
    holding: AtomicBool,
    /// How many writes of flushed vectors have come to wait.
    held: AtomicUsize,
}

#[async_trait::async_trait]
impl Blobs for FailingFlushes {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        if key.ends_with(".vectors")
            && !key.contains("-merged")
            && self
                .failing
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                    left.checked_sub(1)
                })
                .is_ok()
        {
            return Err(vellev_storage::StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        if key.ends_with(".vectors")
            && !key.contains("-merged")
            && self.holding.load(Ordering::SeqCst)
        {
            self.held.fetch_add(1, Ordering::SeqCst);
            while self.holding.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
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

/// A store that counts flushes going up at once, each taking a while.
#[derive(Debug)]
struct OneAtATime {
    inner: Arc<dyn Blobs>,
    going_up: AtomicUsize,
    most: AtomicUsize,
}

#[async_trait::async_trait]
impl Blobs for OneAtATime {
    async fn put(&self, key: &str, bytes: bytes::Bytes) -> vellev_storage::Result<()> {
        if !key.ends_with(".vectors") || key.contains("-merged") {
            return self.inner.put(key, bytes).await;
        }
        let going_up = self.going_up.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(going_up, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(30)).await;
        let put = self.inner.put(key, bytes).await;
        self.going_up.fetch_sub(1, Ordering::SeqCst);
        put
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

/// With its flush task, its merges and a request all flushing it under
/// writes that never let up, and a flush before letting it go at the end, a
/// tenant's flushes go up one at a time: a node reckons one to a tenant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_written_tenant_s_flushes_go_up_one_at_a_time() {
    let store = Arc::new(OneAtATime {
        inner: Arc::new(ObjectBlobs::new(Arc::new(
            object_store::memory::InMemory::new(),
        ))),
        going_up: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
    });
    let idle = Duration::from_millis(300);
    let node = node_on(store.clone(), |settings| {
        settings.flush_rows = NOTES as usize;
        settings.tenant_idle = idle;
    });
    four_flushes(&node, "written").await;
    let written = node.tenant(&tenant("written")).await.unwrap();
    let writing = Arc::new(AtomicBool::new(true));
    let writes = steadily(written.clone(), writing.clone());
    for _ in 0..30 {
        written.flush().await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    writing.store(false, Ordering::SeqCst);
    within(10, "the writes never ended", writes).await.unwrap();
    let batch: Vec<Mutation> = (0..NOTES / 2).map(|n| note(1_000_000 + n)).collect();
    written.write(batch).await.unwrap();
    drop(written);
    tokio::time::sleep(idle + Duration::from_millis(200)).await;
    node.sweeping();
    until(30, "the tenant was never let go", || node.resident() == 0).await;
    assert_eq!(store.most.load(Ordering::SeqCst), 1, "two flushes at once");
}
