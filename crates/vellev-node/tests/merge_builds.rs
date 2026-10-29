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
use vellev_storage::blobs::{Blobs, ObjectBlobs};
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
    let coordinator: Arc<dyn Coordinator> = Arc::new(MemoryCoordinator::default());
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(
        object_store::memory::InMemory::new(),
    )));
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
    go_on.notify_one();
    let written = within(10, "the request never ended", request).await.unwrap();
    written.expect("the request lost its tenant part way");
    until(10, "the tenant was never let go once the request ended", || {
        node.resident() == 0
    })
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
        assert!(builds.draining() > 0, "a stopped build that runs on stopped draining");
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
