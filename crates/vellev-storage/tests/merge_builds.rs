//! A merge's turn among a node's merge builds (merge note, step 3, §7 and
//! §7.7): taken before it gathers a row, given up at its stop while it
//! waits, never waited for by a merge with nothing to do, and held, with
//! its count, until the last work that holds what it built has ended, an
//! upload left to settle among it.

mod common;

use common::*;
use vellev_storage::{Holding, MergeBuilds, Merging};

/// Notes in each segment merged.
const NOTES: u32 = 300;

fn note(n: u32, random: &mut Random) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1_000)),
        properties: json!({ "title": format!("note {n} of a few"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(random.vector(2)),
        only_if_version: None,
    }
}

/// A writer among `builds` of `segments` flushed segments, which merge
/// into one once there are two.
async fn flushed(cluster: &Cluster, builds: &Arc<MergeBuilds>, segments: u32) -> Arc<Writer> {
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = cluster
        .writer()
        .await
        .merging_with(merging)
        .merging_among(builds.clone());
    let writer = Arc::new(writer);
    writer.commit(vec![create("notes")]).await.unwrap();
    let mut random = Random(7);
    for round in 0..segments {
        let notes = (0..NOTES).map(|n| note(round * NOTES + n, &mut random));
        writer.commit(notes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    writer
}

/// What `work` comes to within `limit` seconds: a merge that never gives
/// up, or a turn never given back, fails its rule in seconds rather than
/// hanging it.
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

/// A merge waiting for a turn among the node's merge builds, all taken,
/// gives the wait up at its stop: it never builds, and the next, once a
/// turn is free, commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_waiting_for_a_turn_gives_up_at_its_stop() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilLetGo));
    let (one, other) = (Cluster::new().await, Cluster::new().await);
    let first = flushed(&one, &builds, 2).await;
    let second = flushed(&other, &builds, 2).await;
    let building = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    until(60, "the first merge never came to build", || {
        builds.held() == 1
    })
    .await;
    let waiting = tokio::spawn({
        let second = second.clone();
        async move { second.merge().await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!waiting.is_finished(), "a merge gave up unstopped");
    assert_eq!(
        builds.under_way(),
        1,
        "a merge took a turn none was free for"
    );
    second.stop_merges();
    let given_up = within(10, "a merge waiting for a turn never gave up", waiting).await;
    assert_eq!(given_up.unwrap().unwrap(), None);
    assert_eq!(second.kept_while_merging(), 0);
    builds.hold(None);
    let built = within(60, "the first merge never ended", building).await;
    assert!(built.unwrap().unwrap().is_some());
    let next = within(60, "the next merge never ended", second.merge()).await;
    assert!(
        next.unwrap().is_some(),
        "after one stopped waiting, the next commits"
    );
    assert_eq!(builds.under_way(), 0);
    assert_eq!(builds.turns_free(), 1);
}

/// A merge with nothing to do waits for no turn, however long the node's
/// merge builds run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_with_nothing_to_do_waits_for_no_turn() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilLetGo));
    let (one, other) = (Cluster::new().await, Cluster::new().await);
    let first = flushed(&one, &builds, 2).await;
    let alone = flushed(&other, &builds, 1).await;
    let building = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    until(60, "the first merge never came to build", || {
        builds.held() == 1
    })
    .await;
    let merged = within(
        5,
        "a merge with nothing to do waited for a turn",
        alone.merge(),
    )
    .await;
    assert_eq!(merged.unwrap(), None);
    builds.hold(None);
    let built = within(60, "the first merge never ended", building).await;
    assert!(built.unwrap().unwrap().is_some());
}

/// A store whose uploads of merged vectors, once armed, wait, holding what
/// they were given, until let go or dropped: an upload a stopped merge
/// leaves to settle, or aborts. What the merge builds stood at as the one
/// held ended is noted.
#[derive(Debug)]
struct HeldUpload {
    inner: Arc<dyn Blobs>,
    builds: Arc<MergeBuilds>,
    armed: AtomicBool,
    putting: tokio::sync::mpsc::UnboundedSender<()>,
    let_go: tokio::sync::Notify,
    ended: Arc<Mutex<Option<Seen>>>,
}

/// The builds under way and the turns free, as an upload ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Seen {
    under_way: usize,
    turns_free: usize,
}

/// Notes, as it is dropped, how the builds stood as an upload ended,
/// gone up or dropped where it waited.
struct Ends {
    builds: Arc<MergeBuilds>,
    ended: Arc<Mutex<Option<Seen>>>,
}

impl Drop for Ends {
    fn drop(&mut self) {
        *self.ended.lock().unwrap() = Some(Seen {
            under_way: self.builds.under_way(),
            turns_free: self.builds.turns_free(),
        });
    }
}

#[async_trait]
impl Blobs for HeldUpload {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if !(self.armed.load(Ordering::SeqCst) && key.contains("-merged.vectors")) {
            return self.inner.put(key, bytes).await;
        }
        let _ends = Ends {
            builds: self.builds.clone(),
            ended: self.ended.clone(),
        };
        self.putting.send(()).unwrap();
        self.let_go.notified().await;
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// A merge stopped while its vectors go up gives up at once. The upload
/// holds the merged vectors until it ends, aborted or let land: until
/// then the build is counted under way, holds its turn, and drains what
/// it was reckoned to hold, even once the merge itself has returned. Then
/// all three are let go.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_while_it_uploads_holds_its_turn_and_count_until_the_upload_ends() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldUpload {
        inner: cluster.blobs.clone(),
        builds: builds.clone(),
        armed: AtomicBool::new(false),
        putting,
        let_go: tokio::sync::Notify::new(),
        ended: Arc::default(),
    });
    cluster.blobs = held.clone();
    let writer = flushed(&cluster, &builds, 2).await;
    held.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    within(60, "the merge's vectors never began to go up", put.recv()).await;
    writer.stop_merges();
    let merged = within(10, "the merge never gave up", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    if held.ended.lock().unwrap().is_none() {
        assert_eq!(
            builds.under_way(),
            1,
            "a build was let go while its upload held what it built"
        );
        assert_eq!(
            builds.turns_free(),
            0,
            "a build's turn was given back early"
        );
        assert!(
            builds.draining() > 0,
            "a stopped build stopped draining early"
        );
    }
    held.let_go.notify_one();
    until(30, "the build was never let go", || builds.under_way() == 0).await;
    let seen = held.ended.lock().unwrap().expect("the upload ended");
    assert_eq!(
        seen,
        Seen {
            under_way: 1,
            turns_free: 0
        },
        "as its upload ended, its build was no longer counted, or its turn was free"
    );
    assert_eq!(builds.draining(), 0);
    assert_eq!(builds.turns_free(), 1);
}

/// A merge stopped once it has loaded back what it built commits, as it
/// would have: what it holds is committed, not drained, so none of it
/// counts as draining while it does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_past_loading_back_drains_nothing() {
    let builds = MergeBuilds::new(1);
    let cluster = Cluster::new().await;
    let writer = flushed(&cluster, &builds, 2).await;
    let gate = writer.hold_merges_before_commit();
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    within(60, "the merge was never built", gate.reached.notified()).await;
    assert_eq!(builds.under_way(), 1);
    writer.stop_merges();
    assert_eq!(
        builds.draining(),
        0,
        "a merge past where a stop ends it counted as draining"
    );
    gate.release.notify_one();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(
        merged.unwrap().unwrap().is_some(),
        "past loading back, it commits"
    );
    assert_eq!(builds.under_way(), 0);
    assert_eq!(builds.draining(), 0);
}
