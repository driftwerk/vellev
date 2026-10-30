//! M4-G5: a merge's gather holds up nothing else on a node, however few the
//! async threads and whatever waits on the tenant meanwhile: the copy is on
//! the blocking pool, and holds no lock (node memory note, M4). A test
//! binary of its own, its marks being in milliseconds.
//!
//! Each rule is a plain test on a runtime of its own, watched from outside
//! it: what the rules guard against is that runtime stalled, when neither
//! its timers nor a timeout in it would ever fire.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_embed::{Embedders, Settings as EmbedSettings};
use vellev_node::{Node, Settings};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator};
use vellev_storage::{Holding, Mutation, State};

/// Notes written between flushes: as many as a flush is made at.
const NOTES: u128 = 10;
/// The node's lease, and how often it is renewed: a third of it.
const LEASE: Duration = Duration::from_millis(900);
const RENEWED_EVERY: Duration = Duration::from_millis(300);
/// How long a gather is held, and what goes on is watched for.
const WATCHED: Duration = Duration::from_secs(2);
/// The timer that goes on, and how long it may take to fire at most.
const TIMER: Duration = Duration::from_millis(50);
const FIRES_WITHIN: Duration = Duration::from_millis(250);

/// One rule at a time: each has its marks in milliseconds.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

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

/// Wait until `holds`, at most `limit`.
fn until(limit: Duration, what: &str, holds: impl Fn() -> bool) {
    let by = Instant::now() + limit;
    while !holds() {
        assert!(Instant::now() < by, "{what} within {limit:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The longest gap among `marks` and the ends of the time they were taken
/// over, from `from` to `to`.
fn longest_gap(from: Instant, marks: &[Instant], to: Instant) -> Duration {
    let mut longest = Duration::ZERO;
    let mut last = from;
    for mark in marks.iter().chain([&to]) {
        longest = longest.max(mark.duration_since(last));
        last = *mark;
    }
    longest
}

/// With a tenant's merge held part way through its gather, and a commit and
/// reads of that tenant under way, a timer on another task and another
/// tenant's renewals go on as if nothing else were: on a runtime of
/// `workers` threads, which the copy, were it on one of them, would take
/// one of, and reads on another once the commit waits for the lock.
fn nothing_waits_on_a_gather_on(workers: usize) {
    let _alone = ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|held| held.into_inner());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .unwrap();
    let node = runtime.block_on(async {
        let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(
            object_store::memory::InMemory::new(),
        )));
        let coordinator: Arc<dyn Coordinator> = Arc::new(MemoryCoordinator::default());
        let settings = Settings {
            warm: None,
            flush_rows: NOTES as usize,
            lease: LEASE,
            ..Settings::new(NodeId::generate("node"))
        };
        let embedders = Arc::new(Embedders::new(EmbedSettings::default()));
        Arc::new(Node::new(coordinator, blobs, embedders, settings))
    });
    let builds = node.merge_builds().clone();
    builds.hold(Some(Holding::GatheringUntilLetGo));
    let (merging, other) = runtime.block_on(async {
        node.create_tenant(&tenant("merging")).await.unwrap();
        node.create_tenant(&tenant("other")).await.unwrap();
        let merging = node.tenant(&tenant("merging")).await.unwrap();
        let other = node.tenant(&tenant("other")).await.unwrap();
        merging.write(vec![notes()]).await.unwrap();
        (merging, other)
    });
    for flush in 0..4 {
        let batch: Vec<Mutation> = (0..NOTES).map(|n| note(flush * NOTES + n)).collect();
        runtime.block_on(merging.write(batch)).unwrap();
        until(
            Duration::from_secs(30),
            "the node flushed the tenant",
            || flush == 3 || merging.read(State::unflushed) == 0,
        );
    }
    until(Duration::from_secs(30), "a gather came to be held", || {
        builds.held() == 1
    });

    let fired = Arc::new(Mutex::new(Vec::new()));
    let firing = runtime.spawn({
        let fired = fired.clone();
        async move {
            loop {
                tokio::time::sleep(TIMER).await;
                fired.lock().unwrap().push(Instant::now());
            }
        }
    });
    let written = Arc::new(AtomicBool::new(false));
    let writing = runtime.spawn({
        let (merging, written) = (merging.clone(), written.clone());
        async move {
            merging.write(vec![note(1_000)]).await.unwrap();
            written.store(true, Ordering::SeqCst);
        }
    });
    // As a read of a tenant's schema does, on an async thread.
    let reading = runtime.spawn({
        let merging = merging.clone();
        async move {
            loop {
                merging.read(State::unflushed);
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
    });

    let from = Instant::now();
    let mut renewed = Vec::new();
    let mut until_now = other.held_until();
    while from.elapsed() < WATCHED {
        let held_until = other.held_until();
        if held_until != until_now {
            until_now = held_until;
            renewed.push(Instant::now());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let to = Instant::now();
    let committed = written.load(Ordering::SeqCst);
    let held = builds.held();
    // Let go before anything is said of it: nothing else ends it, and the
    // runtime is not dropped before it is.
    builds.hold(None);

    let fired = fired.lock().unwrap().clone();
    let timer = longest_gap(from, &fired, to);
    let renewal = longest_gap(from, &renewed, to);
    firing.abort();
    reading.abort();
    assert_eq!(held, 1, "the gather was held throughout");
    assert!(
        timer <= FIRES_WITHIN && renewal <= RENEWED_EVERY + FIRES_WITHIN,
        "on {workers} workers, a {TIMER:?} timer took {timer:?} to fire, and another \
         tenant's lease went {renewal:?} unrenewed"
    );
    assert!(committed, "the commit went through, on {workers} workers");
    runtime.block_on(async {
        tokio::time::timeout(Duration::from_secs(30), writing)
            .await
            .unwrap()
            .unwrap();
    });
    drop((merging, other, node));
    runtime.shutdown_timeout(Duration::from_secs(30));
}

/// The rule on `workers`, on a thread of its own, within two minutes: what
/// bounds all it waits for, the runtime's own timers among what it may stop.
fn within_two_minutes_on(workers: usize) {
    let (ended, end) = std::sync::mpsc::channel();
    let rule = std::thread::spawn(move || {
        nothing_waits_on_a_gather_on(workers);
        ended.send(()).unwrap();
    });
    if let Err(std::sync::mpsc::RecvTimeoutError::Timeout) =
        end.recv_timeout(Duration::from_secs(120))
    {
        panic!("the rule ended within two minutes on {workers} workers");
    }
    rule.join().unwrap();
}

/// M4-G5: on two workers, with a commit and reads of the tenant queued.
#[test]
fn a_gather_held_on_two_workers_holds_up_no_timer_and_no_lease() {
    within_two_minutes_on(2);
}

/// M4-G5: on one worker, which a copy on an async thread would take, with
/// no lock held and nothing else waiting.
#[test]
fn a_gather_held_on_one_worker_holds_up_no_timer_and_no_lease() {
    within_two_minutes_on(1);
}
