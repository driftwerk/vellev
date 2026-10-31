//! A merge's turn among a node's merge builds (merge note, step 3, §7 and
//! §7.7): taken before it gathers a row, given up at its stop while it
//! waits, never waited for by a merge with nothing to do, and held, with
//! its count, until the last work that holds what it built has ended, an
//! upload left to settle among it.

mod common;

use common::*;
use vellev_storage::backups::Held;
use vellev_storage::coordinator::{Clock, Restore};
use vellev_storage::timings::{Holder, MergeEnd, MergePhase, Timed};
use vellev_storage::{Holding, MergeBuilds, MergeGate, MergeStage, Merging, TURN_IDLE_AT_MOST};

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
    flushed_due_at(cluster, builds, segments, 0).await
}

/// As [`flushed`], its flushes falling due at `due` rows, a tenant's backlog
/// four times that, as a node has them; none, at no rows.
async fn flushed_due_at(
    cluster: &Cluster,
    builds: &Arc<MergeBuilds>,
    segments: u32,
    due: usize,
) -> Arc<Writer> {
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = cluster
        .writer()
        .await
        .merging_with(merging)
        .merging_among(builds.clone())
        .flushing_due_at(due, 4 * due);
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

/// A store whose uploads of the files `kind` says, once armed, wait,
/// holding what they were given, until let go or dropped: an upload a
/// stopped merge leaves to settle, or aborts, or a flush held part way.
/// What the merge builds stood at as the one held ended is noted.
#[derive(Debug)]
struct HeldUpload {
    inner: Arc<dyn Blobs>,
    kind: fn(&str) -> bool,
    builds: Arc<MergeBuilds>,
    armed: AtomicBool,
    putting: tokio::sync::mpsc::UnboundedSender<()>,
    let_go: tokio::sync::Notify,
    ended: Arc<Mutex<Option<Seen>>>,
}

/// The builds under way, the turns free and whether anything drained, as
/// an upload ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Seen {
    under_way: usize,
    turns_free: usize,
    draining: bool,
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
            draining: self.builds.draining() > 0,
        });
    }
}

#[async_trait]
impl Blobs for HeldUpload {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if !(self.armed.load(Ordering::SeqCst) && (self.kind)(key)) {
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
        kind: |key| key.contains("-merged.vectors"),
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
    // Its turn goes just after its count.
    until(5, "the build's turn was never given back", || {
        builds.turns_free() == 1
    })
    .await;
    let seen = held.ended.lock().unwrap().expect("the upload ended");
    assert_eq!(
        seen,
        Seen {
            under_way: 1,
            turns_free: 0,
            draining: true,
        },
        "as its upload ended, its build was no longer counted, its turn was free, \
         or it no longer drained"
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

/// A merge dropped where it waits for its build, its build past the last
/// place it asks its stop, leaves the build to end with what it built,
/// unawaited: the build holds its turn, and its count, until what it built
/// is dropped, never only until its work returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_build_nobody_awaits_holds_its_turn_until_what_it_built_is_dropped() {
    let builds = MergeBuilds::new(1);
    builds.probe(true);
    builds.hold(Some(Holding::BuiltUntilLetGo));
    let cluster = Cluster::new().await;
    let writer = flushed(&cluster, &builds, 2).await;
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    until(60, "the merge was never built", || builds.held() == 1).await;
    merging.abort();
    assert!(merging.await.unwrap_err().is_cancelled());
    builds.hold(None);
    until(30, "the build was never let go", || builds.under_way() == 0).await;
    let probed = builds
        .probed()
        .expect("what the build made was never dropped");
    assert_eq!(
        probed.under_way, 1,
        "the build's count was let go before what it built"
    );
    assert_eq!(
        probed.turns_free, 0,
        "the build's turn was free before what it built was dropped"
    );
    assert_eq!(builds.turns_free(), 1);
}

/// A merge dropped without being stopped, as when the task that runs it
/// is aborted, stops its build: the build gives up, and its turn is free,
/// at once rather than once the build would have ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_dropped_unstopped_stops_its_build() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilStopped));
    let cluster = Cluster::new().await;
    let writer = flushed(&cluster, &builds, 2).await;
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    until(60, "the merge never came to build", || builds.held() == 1).await;
    merging.abort();
    assert!(merging.await.unwrap_err().is_cancelled());
    until(
        5,
        "the build of a merge dropped unstopped ran on, holding its turn",
        || builds.under_way() == 0,
    )
    .await;
    assert_eq!(builds.turns_free(), 1);
    assert!(!writer.merge_under_way());
    builds.hold(None);
    let next = within(60, "the next merge never ended", writer.merge()).await;
    assert!(next.unwrap().is_some());
}

/// M5-R6, and #164's rule with its bound moved: a merge holds its turn while
/// it waits for its partition's building lock for no longer than
/// [`TURN_IDLE_AT_MOST`], as a flush or a restore may hold the lock for long:
/// it then gives the turn back, and another tenant's merge takes it. Once the
/// lock is free, it takes a turn again and commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_is_held_idle_for_its_partition_s_building_lock_no_longer_than_its_bound() {
    let builds = MergeBuilds::new(1);
    let (mut one, other) = (Cluster::new().await, Cluster::new().await);
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldUpload {
        inner: one.blobs.clone(),
        kind: |key| key.ends_with(".vectors") && !key.contains("-merged"),
        builds: builds.clone(),
        armed: AtomicBool::new(false),
        putting,
        let_go: tokio::sync::Notify::new(),
        ended: Arc::default(),
    });
    one.blobs = held.clone();
    let first = flushed(&one, &builds, 2).await;
    first.timings().keep_events(true);
    let mut random = Random(11);
    let more = (0..NOTES).map(|n| note(10 * NOTES + n, &mut random));
    first.commit(more.collect()).await.unwrap();
    // A flush held part way, holding the partition's building lock.
    held.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let first = first.clone();
        async move { first.flush().await }
    });
    within(60, "the flush's vectors never began to go up", put.recv()).await;
    let second = flushed(&other, &builds, 2).await;
    let waiting = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    let key = first.lease().key.clone();
    until(10, "the merge never took its turn", || {
        builds.under_way_for(&key)
    })
    .await;
    let merged = within(
        TURN_IDLE_AT_MOST.as_secs() + 1,
        "a merge waited for a turn another held while waiting for its own building lock",
        second.merge(),
    )
    .await;
    assert!(merged.unwrap().is_some());
    assert!(!waiting.is_finished());
    held.armed.store(false, Ordering::SeqCst);
    held.let_go.notify_one();
    within(60, "the held flush never ended", flushing)
        .await
        .unwrap()
        .unwrap();
    let merged = within(60, "the waiting merge never ended", waiting).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(builds.under_way(), 0);
    assert_eq!(builds.turns_free(), 1);
    // A8: each wait with a turn, however it ended, timed, within its bound.
    let idle: Vec<Duration> = first
        .timings()
        .drain_events()
        .iter()
        .filter(|event| event.timed == Timed::Merge(MergePhase::TurnIdle))
        .map(|event| event.took)
        .collect();
    assert!(
        idle.iter().any(|idle| *idle >= TURN_IDLE_AT_MOST / 2),
        "the wait that ran out was not timed: {idle:?}"
    );
    let bound = TURN_IDLE_AT_MOST + Duration::from_secs(1);
    assert!(idle.iter().all(|idle| *idle <= bound), "{idle:?}");
}

/// A store whose listings, once armed, wait until it is disarmed: as a
/// clean, which holds the building lock, waits on a slow store.
#[derive(Debug)]
struct HeldListing {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    listing: AtomicBool,
}

impl HeldListing {
    fn over(inner: Arc<dyn Blobs>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            armed: AtomicBool::new(false),
            listing: AtomicBool::new(false),
        })
    }
}

#[async_trait]
impl Blobs for HeldListing {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        while self.armed.load(Ordering::SeqCst) {
            self.listing.store(true, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// M5-R5: a tenant's flush holds its building lock, slowly, as its merge is
/// granted a turn, with a second tenant's merge queued behind. The first
/// keeps its turn, takes the lock as the flush ends, and builds first: it
/// does not lose its place to the second on each flush of its own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_keeps_its_turn_through_its_own_flush_and_builds_before_the_next() {
    let builds = MergeBuilds::new(1);
    let (mut one, other) = (Cluster::new().await, Cluster::new().await);
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldUpload {
        inner: one.blobs.clone(),
        kind: |key| key.ends_with(".vectors") && !key.contains("-merged"),
        builds: builds.clone(),
        armed: AtomicBool::new(false),
        putting,
        let_go: tokio::sync::Notify::new(),
        ended: Arc::default(),
    });
    one.blobs = held.clone();
    let first = flushed(&one, &builds, 2).await;
    let second = flushed(&other, &builds, 2).await;
    let mut random = Random(11);
    let more = (0..NOTES).map(|n| note(10 * NOTES + n, &mut random));
    first.commit(more.collect()).await.unwrap();
    builds.hold(Some(Holding::UntilLetGo));
    held.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let first = first.clone();
        async move { first.flush().await }
    });
    within(60, "the flush's vectors never began to go up", put.recv()).await;
    let merging = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    let key = first.lease().key.clone();
    until(10, "the merge never took its turn", || {
        builds.under_way_for(&key)
    })
    .await;
    let behind = tokio::spawn({
        let second = second.clone();
        async move { second.merge().await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    held.armed.store(false, Ordering::SeqCst);
    held.let_go.notify_one();
    within(60, "the held flush never ended", flushing)
        .await
        .unwrap()
        .unwrap();
    until(10, "no merge came to build", || builds.held() == 1).await;
    // The build held is the first's: let go, it ends, and the second's comes.
    builds.let_one_go();
    let merged = within(
        30,
        "a merge lost its turn to one queued behind it, for a flush of its own",
        merging,
    )
    .await;
    assert!(merged.unwrap().unwrap().is_some());
    until(10, "the merge behind never came to build", || {
        builds.held() == 1
    })
    .await;
    assert!(!behind.is_finished());
    builds.hold(None);
    let merged = within(60, "the second merge never ended", behind).await;
    assert!(merged.unwrap().unwrap().is_some());
}

/// M5-R5 with a clean as the holder: cleaning up beside a tenant's merges
/// holds its building lock, and the merge comes through that as it comes
/// through a flush, its turn kept, its place ahead of the merge behind it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_keeps_its_turn_through_its_own_clean_and_builds_before_the_next() {
    let builds = MergeBuilds::new(1);
    let (mut one, other) = (Cluster::new().await, Cluster::new().await);
    let held = HeldListing::over(one.blobs.clone());
    one.blobs = held.clone();
    let first = flushed(&one, &builds, 2).await;
    let second = flushed(&other, &builds, 2).await;
    builds.hold(Some(Holding::UntilLetGo));
    held.armed.store(true, Ordering::SeqCst);
    let cleaning = tokio::spawn({
        let first = first.clone();
        async move { first.clean(Duration::ZERO).await }
    });
    until(10, "the clean never began", || {
        held.listing.load(Ordering::SeqCst)
    })
    .await;
    let merging = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    let key = first.lease().key.clone();
    until(10, "the merge never took its turn", || {
        builds.under_way_for(&key)
    })
    .await;
    let behind = tokio::spawn({
        let second = second.clone();
        async move { second.merge().await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    held.armed.store(false, Ordering::SeqCst);
    within(60, "the held clean never ended", cleaning)
        .await
        .unwrap()
        .unwrap();
    until(10, "no merge came to build", || builds.held() == 1).await;
    // The build held is the first's: let go, it ends, and the second's comes.
    builds.let_one_go();
    let merged = within(
        30,
        "a merge lost its turn to one queued behind it, for a clean of its own",
        merging,
    )
    .await;
    assert!(merged.unwrap().unwrap().is_some());
    until(10, "the merge behind never came to build", || {
        builds.held() == 1
    })
    .await;
    assert!(!behind.is_finished());
    builds.hold(None);
    let merged = within(60, "the second merge never ended", behind).await;
    assert!(merged.unwrap().unwrap().is_some());
}

/// A merge stopped while it waits for the lock with its turn gives the turn
/// back at once, and is counted as one that had a turn and was stopped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_as_it_waits_with_its_turn_gives_the_turn_back_at_once() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let held = HeldListing::over(cluster.blobs.clone());
    cluster.blobs = held.clone();
    let writer = flushed(&cluster, &builds, 2).await;
    held.armed.store(true, Ordering::SeqCst);
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    until(10, "the clean never began", || {
        held.listing.load(Ordering::SeqCst)
    })
    .await;
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let key = writer.lease().key.clone();
    until(10, "the merge never took its turn", || {
        builds.under_way_for(&key)
    })
    .await;
    assert!(writer.stop_merges());
    let merged = within(2, "a merge stopped with its turn went on waiting", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    assert_eq!(builds.turns_free(), 1);
    assert_eq!(builds.under_way(), 0);
    assert_eq!(writer.timings().merges_ended(MergeEnd::Stopped), 1);
    held.armed.store(false, Ordering::SeqCst);
    within(60, "the clean never ended", cleaning)
        .await
        .unwrap()
        .unwrap();
}

/// A restore that has begun ends a merge's wait with its turn at once: it
/// holds the lock for far longer than the wait, and only drops what a merge
/// planned before it. The merge asked for its turn as the restore began.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_under_way_sends_a_turn_waiting_for_its_lock_back_at_once() {
    let builds = MergeBuilds::new(1);
    let (cluster, other) = (Cluster::new().await, Cluster::new().await);
    builds.hold(Some(Holding::UntilLetGo));
    let writer = flushed(&cluster, &builds, 2).await;
    let holder = flushed(&other, &builds, 2).await;
    let holding = tokio::spawn({
        let holder = holder.clone();
        async move { holder.merge().await }
    });
    until(60, "the other merge never came to build", || {
        builds.held() == 1
    })
    .await;
    // Past its look for a restore, in the queue for the turn the other holds.
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    let at = cluster.committed().await;
    let restore = Restore {
        request: RequestId::generate(),
        asked: [0; 32],
        from: at.log_position,
        from_at: cluster.clock.now(),
    };
    let restoring = writer
        .restore_from(
            cluster.blobs.clone(),
            at.manifest.as_ref(),
            restore,
            None,
            &Claimed,
        )
        .await
        .unwrap();
    builds.hold(None);
    within(60, "the other merge never ended", holding)
        .await
        .unwrap()
        .unwrap();
    until(3, "a turn stayed idle behind a restore", || {
        builds.turns_free() == 1 && builds.under_way() == 0
    })
    .await;
    assert!(!merging.is_finished());
    drop(restoring);
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(builds.turns_free(), 1);
}

/// A store whose writes of flushed vectors, once armed, each take
/// `slowly`: flushes that take longer than a merge waits with its turn.
#[derive(Debug)]
struct SlowFlushes {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    slowly: Duration,
}

#[async_trait]
impl Blobs for SlowFlushes {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.armed.load(Ordering::SeqCst)
            && key.ends_with(".vectors")
            && !key.contains("-merged")
        {
            tokio::time::sleep(self.slowly).await;
        }
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

/// Flushes each longer than a merge waits with its turn, back to back, one
/// always queued behind the one under way, never starve a merge. It waits
/// with its turn and gives it back, and keeps its place for the lock without
/// one; holding the lock it takes a turn that is free, there and then: let
/// go to ask for one, the next flush would have the lock again, and the merge
/// would see the start of a flush each time, with a turn idling each time.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flushes_longer_than_a_turn_waits_back_to_back_never_starve_a_merge() {
    let builds = MergeBuilds::new(1);
    builds.idle_for(Duration::from_millis(100));
    let mut cluster = Cluster::new().await;
    let slow = Arc::new(SlowFlushes {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
        slowly: Duration::from_millis(300),
    });
    cluster.blobs = slow.clone();
    let writer = flushed(&cluster, &builds, 2).await;
    slow.armed.store(true, Ordering::SeqCst);
    let flushing = Arc::new(AtomicBool::new(true));
    let flushers: Vec<_> = (0..2)
        .map(|flusher| {
            let (writer, flushing) = (writer.clone(), flushing.clone());
            tokio::spawn(async move {
                let mut random = Random(100 + flusher);
                let mut n = 100 * NOTES + flusher as u32 * 100_000;
                while flushing.load(Ordering::SeqCst) {
                    writer.commit(vec![note(n, &mut random)]).await.unwrap();
                    n += 1;
                    writer.flush().await.unwrap();
                }
            })
        })
        .collect();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let merged = within(
        15,
        "flushes back to back starved a merge",
        writer.merge_planned_at(None),
    )
    .await;
    flushing.store(false, Ordering::SeqCst);
    for flusher in flushers {
        within(30, "a flush never ended", flusher).await.unwrap();
    }
    assert!(merged.unwrap().is_some());
    assert_eq!(builds.turns_free(), 1);
}

/// A merge held past loading back what it built, on a writer of `builds`:
/// its gate, and the merge.
async fn at_its_gate(writer: &Arc<Writer>) -> (Arc<MergeGate>, tokio::task::JoinHandle<Merged>) {
    let gate = writer.hold_merges_before_commit();
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    within(
        60,
        "the merge never came to its commit",
        gate.reached.notified(),
    )
    .await;
    (gate, merging)
}

type Merged = vellev_storage::Result<Option<LogPosition>>;

/// `rows` more notes, numbered from `from`, committed a thousand at a time.
async fn more_notes(writer: &Writer, from: u32, rows: u32) {
    let mut random = Random(13);
    for chunk in (from..from + rows).collect::<Vec<u32>>().chunks(1_000) {
        let notes = chunk.iter().map(|n| note(*n, &mut random)).collect();
        writer.commit(notes).await.unwrap();
    }
}

/// How long each hold of the lock by `holder` kept since last asked.
fn holds(events: &[vellev_storage::timings::Event], holder: Holder) -> Vec<Duration> {
    events
        .iter()
        .filter(|event| event.timed == Timed::BuildingHeld(holder))
        .map(|event| event.took)
        .collect()
}

/// How many rows each segment of `writer` holds, with the merged one's among
/// them.
fn rows_by_segment(writer: &Writer) -> Vec<u64> {
    writer
        .partition()
        .read(|state| state.segments().map(|segment| segment.rows).collect())
}

/// M5-R7: with 40k rows unflushed as a merge comes to commit, its commit's
/// hold of the building lock is under a flush of 4k rows. Those rows go
/// first, as a plain flush of their own.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_flushes_first_and_its_commit_holds_the_lock_for_no_flush_of_them() {
    let builds = MergeBuilds::new(1);
    let cluster = Cluster::new().await;
    let writer = flushed_due_at(&cluster, &builds, 2, 10_000).await;
    writer.timings().keep_events(true);
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 4_000).await;
    writer.flush().await.unwrap();
    let four_thousand = holds(&writer.timings().drain_events(), Holder::Flush)[0];
    more_notes(&writer, 2_000_000, 40_000).await;
    let before = rows_by_segment(&writer).len();
    gate.release.notify_one();
    let merged = within(120, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    let events = writer.timings().drain_events();
    let commit = holds(&events, Holder::MergeCommit);
    assert_eq!(commit.len(), 1, "{commit:?}");
    assert!(
        commit[0] < four_thousand,
        "a commit held the lock {:?}, a flush of 4k rows {four_thousand:?}",
        commit[0]
    );
    assert_eq!(
        holds(&events, Holder::Flush).len(),
        1,
        "the 40k were not flushed first, on their own"
    );
    let rows = rows_by_segment(&writer);
    assert!(rows.contains(&40_000), "{rows:?}");
    // The two merged into one, and the 40k: nothing else went up.
    assert_eq!(rows.len(), before, "{rows:?}");
    assert_eq!(writer.partition().read(State::unflushed), 0);
}

/// Fewer rows than a flush are left to the commit, in one checkpoint, as
/// they were: flushed first they would make two small segments of each
/// commit made under writes. So what a commit flushes under its hold stays
/// under a flush, unless its flush first could not run.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_leaves_fewer_rows_than_a_flush_to_its_commit() {
    let builds = MergeBuilds::new(1);
    let cluster = Cluster::new().await;
    let writer = flushed_due_at(&cluster, &builds, 2, 1_000).await;
    writer.timings().keep_events(true);
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 999).await;
    gate.release.notify_one();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    let events = writer.timings().drain_events();
    assert_eq!(holds(&events, Holder::Flush), [], "flushed first, undue");
    let rows = rows_by_segment(&writer);
    assert!(rows.contains(&999), "{rows:?}");
    assert_eq!(rows.len(), 2, "{rows:?}");
}

/// A store whose next write of flushed vectors, once armed, fails.
#[derive(Debug)]
struct FlushFails {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
}

#[async_trait]
impl Blobs for FlushFails {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if key.ends_with(".vectors")
            && !key.contains("-merged")
            && self.armed.swap(false, Ordering::SeqCst)
        {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
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

/// A flush first that fails costs nothing: the merge commits with the whole
/// flush along, as it would have, and every row is in what it committed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_first_that_fails_leaves_its_rows_to_the_commit() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let fails = Arc::new(FlushFails {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
    });
    cluster.blobs = fails.clone();
    let writer = flushed_due_at(&cluster, &builds, 2, 1_000).await;
    writer.timings().keep_events(true);
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 2_000).await;
    fails.armed.store(true, Ordering::SeqCst);
    gate.release.notify_one();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert!(
        !fails.armed.load(Ordering::SeqCst),
        "no flush first was made"
    );
    let rows = rows_by_segment(&writer);
    assert!(rows.contains(&2_000), "{rows:?}");
    assert_eq!(writer.partition().read(State::unflushed), 0);
}

/// A flush first's wait for the lock running out costs nothing but the
/// commit's hold: the merge gives its turn back, as it would at its commit,
/// and flushes along. A flush first under way is never cut: only the wait is
/// bounded, or what it had built would run on, with a second build of the
/// same rows beside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_first_is_bounded_in_its_wait_and_never_in_its_flush() {
    let builds = MergeBuilds::new(1);
    builds.idle_for(Duration::from_millis(100));
    let mut cluster = Cluster::new().await;
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldUpload {
        inner: cluster.blobs.clone(),
        kind: |key| key.ends_with(".vectors") && !key.contains("-merged"),
        builds: builds.clone(),
        armed: AtomicBool::new(false),
        putting,
        let_go: tokio::sync::Notify::new(),
        ended: Arc::default(),
    });
    cluster.blobs = held.clone();
    let writer = flushed_due_at(&cluster, &builds, 2, 1_000).await;
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 2_000).await;
    held.armed.store(true, Ordering::SeqCst);
    gate.release.notify_one();
    within(60, "the flush first never began to go up", put.recv()).await;
    // Well past the bound, and no second flush begins.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(put.try_recv().is_err(), "a second flush beside the first");
    held.armed.store(false, Ordering::SeqCst);
    held.let_go.notify_one();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    let rows = rows_by_segment(&writer);
    assert!(rows.contains(&2_000), "{rows:?}");
    assert_eq!(rows.len(), 2, "{rows:?}");
}

/// A flush first's wait running out costs nothing but the commit's hold: the
/// turn, having waited as long as it may, goes back, stays counted, and the
/// rows go with the commit, behind what held the lock.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_first_whose_wait_runs_out_leaves_its_rows_to_the_commit() {
    let builds = MergeBuilds::new(1);
    builds.idle_for(Duration::from_millis(200));
    let mut cluster = Cluster::new().await;
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldUpload {
        inner: cluster.blobs.clone(),
        kind: |key| key.ends_with(".vectors") && !key.contains("-merged"),
        builds: builds.clone(),
        armed: AtomicBool::new(false),
        putting,
        let_go: tokio::sync::Notify::new(),
        ended: Arc::default(),
    });
    cluster.blobs = held.clone();
    let writer = flushed_due_at(&cluster, &builds, 2, 1_000).await;
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 1_000).await;
    held.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    within(60, "the flush never began to go up", put.recv()).await;
    more_notes(&writer, 2_000_000, 1_000).await;
    gate.release.notify_one();
    until(5, "a flush first's wait held its turn", || {
        builds.turns_free() == 1
    })
    .await;
    assert!(builds.draining() > 0, "what the merge holds is not counted");
    held.armed.store(false, Ordering::SeqCst);
    held.let_go.notify_one();
    within(60, "the flush never ended", flushing)
        .await
        .unwrap()
        .unwrap();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(builds.draining(), 0);
    let rows = rows_by_segment(&writer);
    assert!(rows.contains(&1_000), "{rows:?}");
    assert_eq!(rows.len(), 3, "{rows:?}");
    assert_eq!(writer.partition().read(State::unflushed), 0);
}

/// A flush under way as a merge comes to commit takes what was due, and the
/// flush first, once it has the lock, finds too few rows to flush: those go
/// with the commit, as fewer than a flush do, and make no segment of their
/// own, as a flush behind a flush would on every commit under writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_first_behind_a_flush_that_took_its_rows_flushes_nothing() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldUpload {
        inner: cluster.blobs.clone(),
        kind: |key| key.ends_with(".vectors") && !key.contains("-merged"),
        builds: builds.clone(),
        armed: AtomicBool::new(false),
        putting,
        let_go: tokio::sync::Notify::new(),
        ended: Arc::default(),
    });
    cluster.blobs = held.clone();
    let writer = flushed_due_at(&cluster, &builds, 2, 1_000).await;
    writer.timings().keep_events(true);
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 2_000).await;
    held.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    within(60, "the flush never began to go up", put.recv()).await;
    more_notes(&writer, 2_000_000, 10).await;
    gate.release.notify_one();
    tokio::time::sleep(Duration::from_millis(200)).await;
    held.armed.store(false, Ordering::SeqCst);
    held.let_go.notify_one();
    within(60, "the flush never ended", flushing)
        .await
        .unwrap()
        .unwrap();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    let events = writer.timings().drain_events();
    assert_eq!(
        holds(&events, Holder::Flush).len(),
        1,
        "flushed behind a flush"
    );
    assert_eq!(writer.partition().read(State::unflushed), 0);
}

/// What flushes goes up one flush at a time, whoever asks: a flush asked for
/// by a request or a tenant's flush task, a merge's flush first, a flush
/// before letting go, and a merge's flush along. Each holds a flush's bytes,
/// and a node reckons one flush to a tenant.
#[derive(Debug)]
struct OneAtATime {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    going_up: AtomicUsize,
    most: AtomicUsize,
    uploads: AtomicUsize,
}

#[async_trait]
impl Blobs for OneAtATime {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if !key.ends_with(".vectors") || key.contains("-merged") {
            return self.inner.put(key, bytes).await;
        }
        let going_up = self.going_up.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(going_up, Ordering::SeqCst);
        self.uploads.fetch_add(1, Ordering::SeqCst);
        while self.armed.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let put = self.inner.put(key, bytes).await;
        self.going_up.fetch_sub(1, Ordering::SeqCst);
        put
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

/// Two flushes' worth of a tenant's bytes never overlap: with a flush held
/// part way, a merge's flush first and a flush before letting go wait behind
/// it, and the merge's flush along behind those, each with rows to flush.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tenant_flushes_one_at_a_time_whoever_asks() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let store = Arc::new(OneAtATime {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
        going_up: AtomicUsize::new(0),
        most: AtomicUsize::new(0),
        uploads: AtomicUsize::new(0),
    });
    cluster.blobs = store.clone();
    let writer = flushed_due_at(&cluster, &builds, 2, 100).await;
    let before = store.uploads.load(Ordering::SeqCst);
    let (gate, merging) = at_its_gate(&writer).await;
    more_notes(&writer, 1_000_000, 200).await;
    store.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    let uploads = || store.uploads.load(Ordering::SeqCst) - before;
    until(10, "the flush never began to go up", || uploads() == 1).await;
    more_notes(&writer, 2_000_000, 200).await;
    gate.release.notify_one();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let letting_go = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush_timed().await }
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(uploads(), 1, "a flush went up beside another");
    store.armed.store(false, Ordering::SeqCst);
    within(60, "the flush never ended", flushing)
        .await
        .unwrap()
        .unwrap();
    within(60, "the flush before letting go never ended", letting_go)
        .await
        .unwrap()
        .unwrap();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(store.most.load(Ordering::SeqCst), 1, "two flushes at once");
    assert!(uploads() >= 2, "only one flushed anything");
}

/// A commit waits for the building lock with what it built held. Behind a
/// long hold, as a clean's of up to two minutes on a slow store, it holds
/// its turn no longer than a merge waiting to plan does: the turn goes back
/// and another tenant's merge takes it. It keeps its place for the lock, and
/// what it holds stays counted, as draining, against the node's budget, as
/// no turn stands for it now, until it commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_commit_behind_a_long_hold_gives_its_turn_back_and_stays_counted() {
    let builds = MergeBuilds::new(1);
    let (mut one, other) = (Cluster::new().await, Cluster::new().await);
    let held = HeldListing::over(one.blobs.clone());
    one.blobs = held.clone();
    let first = flushed(&one, &builds, 2).await;
    first.timings().keep_events(true);
    let (gate, committing) = at_its_gate(&first).await;
    assert_eq!(builds.turns_free(), 0);
    assert_eq!(builds.draining(), 0, "a merge under way is no draining");
    held.armed.store(true, Ordering::SeqCst);
    let cleaning = tokio::spawn({
        let first = first.clone();
        async move { first.clean(Duration::ZERO).await }
    });
    until(10, "the clean never began", || {
        held.listing.load(Ordering::SeqCst)
    })
    .await;
    gate.release.notify_one();
    let waited = std::time::Instant::now();
    let bound = TURN_IDLE_AT_MOST.as_secs() + 1;
    until(bound, "a commit held its turn behind a clean", || {
        builds.turns_free() == 1
    })
    .await;
    assert!(
        waited.elapsed() >= TURN_IDLE_AT_MOST / 2,
        "the turn went back at once, after {:?}",
        waited.elapsed()
    );
    let reckoned = builds.largest_reckoned();
    assert!(reckoned > 0);
    assert_eq!(builds.draining(), reckoned, "what it holds is not counted");
    assert_eq!(builds.under_way(), 0, "counted among the turns held");
    let second = flushed(&other, &builds, 2).await;
    let merged = within(60, "no other merge took the turn", second.merge()).await;
    assert!(merged.unwrap().is_some());
    assert_eq!(builds.draining(), reckoned, "let go before it committed");
    held.armed.store(false, Ordering::SeqCst);
    within(60, "the clean never ended", cleaning)
        .await
        .unwrap()
        .unwrap();
    let merged = within(60, "the commit never ended", committing).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(builds.draining(), 0, "still counted after it committed");
    assert_eq!(builds.turns_free(), 1);
    let idle: Vec<Duration> = first
        .timings()
        .drain_events()
        .iter()
        .filter(|event| event.timed == Timed::Merge(MergePhase::TurnIdle))
        .map(|event| event.took)
        .collect();
    let bound = TURN_IDLE_AT_MOST + Duration::from_secs(1);
    assert!(
        idle.iter().any(|idle| *idle >= TURN_IDLE_AT_MOST / 2),
        "{idle:?}"
    );
    assert!(idle.iter().all(|idle| *idle <= bound), "{idle:?}");
}

/// A store that hands a copy of each merged vectors file it is given, once
/// armed, to a thread of its own, which holds it until let go: as a cache
/// writes its local copy on a thread of its own, or a local store writes on
/// a blocking thread, whoever stops waiting.
#[derive(Debug)]
struct Copying {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    let_go: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

#[async_trait]
impl Blobs for Copying {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.armed.load(Ordering::SeqCst) && key.contains("-merged.vectors") {
            let (copy, let_go) = (bytes.clone(), self.let_go.clone());
            std::thread::spawn(move || {
                let (gone, told) = &*let_go;
                let mut gone = gone.lock().unwrap();
                while !*gone {
                    gone = told.wait(gone).unwrap();
                }
                drop(copy);
            });
        }
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

/// What a stopped merge uploaded may be held on by a store's own thread
/// after the merge, its uploads and their deletes have all ended: the build
/// holds its turn, its count and what it drains until the last copy is let
/// go, wherever it is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_copy_of_a_stopped_merge_s_upload_holds_its_turn_until_it_is_let_go() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let copying = Arc::new(Copying {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
        let_go: Arc::default(),
    });
    cluster.blobs = copying.clone();
    let writer = flushed(&cluster, &builds, 2).await;
    copying.armed.store(true, Ordering::SeqCst);
    // Stopped once uploaded, as it loads back what it built.
    writer.stop_merges_at(Some(MergeStage::Loading));
    let merged = within(60, "the merge never gave up", writer.merge()).await;
    assert_eq!(merged.unwrap(), None);
    for _ in 0..30 {
        assert_eq!(
            builds.under_way(),
            1,
            "a build was let go while a store's thread held what it built"
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
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let (gone, told) = &*copying.let_go;
    *gone.lock().unwrap() = true;
    told.notify_all();
    until(5, "the build was never let go once the copy was", || {
        builds.under_way() == 0 && builds.turns_free() == 1
    })
    .await;
    assert_eq!(builds.draining(), 0);
}

/// A merge that commits gives its turn back, though a store keeps what it
/// uploaded, as one in memory does: what it built is the partition's now,
/// counted as what the partition holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_committed_merge_gives_its_turn_back_though_its_store_keeps_its_files() {
    let builds = MergeBuilds::new(1);
    let cluster = Cluster::new().await;
    let writer = flushed(&cluster, &builds, 2).await;
    let mut random = Random(13);
    for round in 0..3 {
        let merged = within(
            60,
            "a merge waited for a turn a committed one kept",
            writer.merge(),
        );
        assert!(merged.await.unwrap().is_some(), "merge {round}");
        assert_eq!(builds.under_way(), 0, "merge {round}");
        assert_eq!(builds.turns_free(), 1, "merge {round}");
        let notes = (0..NOTES).map(|n| note((20 + round) * NOTES + n, &mut random));
        writer.commit(notes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
}

/// Holds the claim a restore is made under.
struct Claimed;

#[async_trait]
impl Held for Claimed {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(true)
    }
    fn still(&self) -> bool {
        true
    }
}

/// A merge on a writer that is restoring takes no turn: it could only be
/// dropped, the partition put back, and the turn is another's meanwhile.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_while_its_writer_restores_takes_no_turn() {
    let builds = MergeBuilds::new(1);
    let cluster = Cluster::new().await;
    let writer = flushed(&cluster, &builds, 2).await;
    let at = cluster.committed().await;
    let restore = Restore {
        request: RequestId::generate(),
        asked: [0; 32],
        from: at.log_position,
        from_at: cluster.clock.now(),
    };
    let restoring = writer
        .restore_from(
            cluster.blobs.clone(),
            at.manifest.as_ref(),
            restore,
            None,
            &Claimed,
        )
        .await
        .unwrap();
    let merged = within(
        5,
        "a merge on a writer that is restoring waited",
        writer.merge(),
    )
    .await;
    assert_eq!(merged.unwrap(), None);
    assert_eq!(builds.under_way(), 0);
    assert_eq!(builds.turns_free(), 1);
    drop(restoring);
    let next = within(60, "the next merge never ended", writer.merge()).await;
    assert!(next.unwrap().is_some());
}

/// A store that, once armed, fails every write of a manifest.
#[derive(Debug)]
struct NoManifests {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
}

#[async_trait]
impl Blobs for NoManifests {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.armed.load(Ordering::SeqCst) && key.contains("/manifest/") {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
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

/// A merge whose commit fails once it has kept its files gives its turn
/// back, though its store keeps them, as one in memory does: from there
/// they are the partition's to clean up, not the merge's to delete.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_whose_commit_fails_once_its_files_are_kept_gives_its_turn_back() {
    let builds = MergeBuilds::new(1);
    let mut cluster = Cluster::new().await;
    let failing = Arc::new(NoManifests {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
    });
    cluster.blobs = failing.clone();
    let writer = flushed(&cluster, &builds, 2).await;
    failing.armed.store(true, Ordering::SeqCst);
    let merged = within(60, "the merge never ended", writer.merge()).await;
    assert!(
        merged.is_err(),
        "a merge whose manifest is refused committed"
    );
    until(5, "a merge whose commit failed kept its turn", || {
        builds.under_way() == 0 && builds.turns_free() == 1
    })
    .await;
}

/// Two writers among `builds` of one turn: the first's merge built and
/// held there, the second's waiting for the turn until `due` says its
/// flush is due, when it gives way. Returns once it has.
async fn one_gave_way(
    builds: &Arc<MergeBuilds>,
    due: &Arc<AtomicBool>,
) -> (
    Arc<Writer>,
    tokio::task::JoinHandle<vellev_storage::Result<Option<LogPosition>>>,
    Arc<Writer>,
    Cluster,
) {
    let (one, other) = (Cluster::new().await, Cluster::new().await);
    let first = flushed(&one, builds, 2).await;
    let second = flushed(&other, builds, 2).await;
    let building = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    until(60, "the first merge never came to build", || {
        builds.held() == 1
    })
    .await;
    let waiting = tokio::spawn({
        let (second, due) = (second.clone(), due.clone());
        async move {
            second
                .merge_giving_way(&|| due.load(Ordering::SeqCst), &|_| {})
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!waiting.is_finished(), "a merge gave way with no flush due");
    due.store(true, Ordering::SeqCst);
    let gave_way = within(5, "a merge never gave way to a due flush", waiting).await;
    assert_eq!(gave_way.unwrap().unwrap(), None);
    (first, building, second, other)
}

/// A merge that gives way to a flush while it waits for its turn keeps its
/// place: the turn that comes free next is kept for its next attempt, not
/// taken by a merge that queued behind it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_that_gives_way_to_a_flush_keeps_its_place_for_a_turn() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilLetGo));
    let (one, other, third) = (
        Cluster::new().await,
        Cluster::new().await,
        Cluster::new().await,
    );
    let first = flushed(&one, &builds, 2).await;
    let second = flushed(&other, &builds, 2).await;
    let behind = flushed(&third, &builds, 2).await;
    let building = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    until(60, "the first merge never came to build", || {
        builds.held() == 1
    })
    .await;
    let due = Arc::new(AtomicBool::new(false));
    let waiting = tokio::spawn({
        let (second, due) = (second.clone(), due.clone());
        async move {
            second
                .merge_giving_way(&|| due.load(Ordering::SeqCst), &|_| {})
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    // Queued behind the second.
    let queued = tokio::spawn({
        let behind = behind.clone();
        async move { behind.merge().await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    due.store(true, Ordering::SeqCst);
    let gave_way = within(5, "a merge never gave way to a due flush", waiting).await;
    assert_eq!(gave_way.unwrap().unwrap(), None);
    // The turn comes free.
    builds.let_one_go();
    let built = within(60, "the first merge never ended", building).await;
    assert!(built.unwrap().unwrap().is_some());
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !builds.under_way_for(&behind.lease().key),
        "a merge queued behind one that gave way to a flush took its turn"
    );
    assert_eq!(builds.held(), 0);
    // Its next attempt takes the turn kept for it.
    due.store(false, Ordering::SeqCst);
    let next = tokio::spawn({
        let (second, due) = (second.clone(), due.clone());
        async move {
            second
                .merge_giving_way(&|| due.load(Ordering::SeqCst), &|_| {})
                .await
        }
    });
    until(10, "the merge that gave way never took its turn", || {
        builds.under_way_for(&second.lease().key)
    })
    .await;
    builds.hold(None);
    let merged = within(60, "the merge that gave way never ended", next).await;
    assert!(merged.unwrap().unwrap().is_some());
    let merged = within(60, "the merge behind it never ended", queued).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(builds.under_way(), 0);
    assert_eq!(builds.turns_free(), 1);
}

/// A turn kept for a merge that gave way to a flush is let go once its
/// writer's merges are stopped, as letting the tenant go stops them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_kept_for_a_merge_is_let_go_once_its_merges_are_stopped() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilLetGo));
    let due = Arc::new(AtomicBool::new(false));
    let (_first, building, second, _other) = one_gave_way(&builds, &due).await;
    builds.let_one_go();
    within(60, "the first merge never ended", building)
        .await
        .unwrap()
        .unwrap();
    until(
        5,
        "the turn was not kept for the merge that gave way",
        || builds.turns_free() == 0,
    )
    .await;
    second.stop_merges();
    until(
        5,
        "a turn kept for a stopped writer was never let go",
        || builds.turns_free() == 1,
    )
    .await;
}

/// A turn kept for a merge that gave way to a flush is let go once its
/// writer is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_kept_for_a_merge_is_let_go_once_its_writer_is_dropped() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilLetGo));
    let due = Arc::new(AtomicBool::new(false));
    let (_first, building, second, _other) = one_gave_way(&builds, &due).await;
    builds.let_one_go();
    within(60, "the first merge never ended", building)
        .await
        .unwrap()
        .unwrap();
    until(
        5,
        "the turn was not kept for the merge that gave way",
        || builds.turns_free() == 0,
    )
    .await;
    drop(second);
    until(
        5,
        "a turn kept for a dropped writer was never let go",
        || builds.turns_free() == 1,
    )
    .await;
}

/// A turn kept for a merge that gave way to a flush, never taken up, is let
/// go once it has been kept as long as it is kept at most.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_kept_for_a_merge_never_taken_up_is_let_go() {
    let builds = MergeBuilds::new(1);
    builds.park_for(Duration::from_millis(300));
    builds.hold(Some(Holding::UntilLetGo));
    let due = Arc::new(AtomicBool::new(false));
    let (_first, building, _second, _other) = one_gave_way(&builds, &due).await;
    builds.let_one_go();
    within(60, "the first merge never ended", building)
        .await
        .unwrap()
        .unwrap();
    until(
        5,
        "the turn was not kept for the merge that gave way",
        || builds.turns_free() == 0,
    )
    .await;
    until(5, "a turn kept and never taken up was never let go", || {
        builds.turns_free() == 1
    })
    .await;
}

/// A merge dropped with its runtime, where nothing is left to delete what
/// it uploaded from a store that keeps what it is given, as one in memory
/// does, gives its turn back all the same.
#[test]
fn a_merge_dropped_with_its_runtime_gives_its_turn_back() {
    let builds = MergeBuilds::new(1);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    // The store outlives the runtime, as a node's does its tasks.
    let store = runtime.block_on(async {
        let mut cluster = Cluster::new().await;
        let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
        let held = Arc::new(HeldUpload {
            inner: cluster.blobs.clone(),
            // Its vectors are kept by then.
            kind: |key| key.contains("-merged.segment"),
            builds: builds.clone(),
            armed: AtomicBool::new(false),
            putting,
            let_go: tokio::sync::Notify::new(),
            ended: Arc::default(),
        });
        cluster.blobs = held.clone();
        let writer = flushed(&cluster, &builds, 2).await;
        held.armed.store(true, Ordering::SeqCst);
        tokio::spawn(async move { writer.merge().await });
        within(60, "the merge's segment never began to go up", put.recv()).await;
        assert_eq!(builds.turns_free(), 0);
        held.inner.clone()
    });
    drop(runtime);
    assert_eq!(
        builds.turns_free(),
        1,
        "a merge dropped with its runtime kept its turn"
    );
    drop(store);
}

/// A store that, once armed, keeps the merged vectors it is given and
/// answers that the write failed: as a write that will land late, or a
/// thread that never ends.
#[derive(Debug)]
struct Keeping {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    kept: Mutex<Vec<Bytes>>,
}

#[async_trait]
impl Blobs for Keeping {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if self.armed.load(Ordering::SeqCst) && key.contains("-merged.vectors") {
            self.kept.lock().unwrap().push(bytes);
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
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

/// What a store keeps of a failed merge's upload holds the merge's turn,
/// but not for good: once held longer than any store's write of it may
/// take, the turn is let go, so that a writer's merges never wait forever
/// behind a write that will not end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_held_by_what_a_store_keeps_of_a_failed_merge_is_let_go_in_time() {
    let builds = MergeBuilds::new(1);
    builds.let_go_held_after(Duration::from_millis(300));
    let mut cluster = Cluster::new().await;
    let keeping = Arc::new(Keeping {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
        kept: Mutex::default(),
    });
    cluster.blobs = keeping.clone();
    let writer = flushed(&cluster, &builds, 2).await;
    keeping.armed.store(true, Ordering::SeqCst);
    let failed = within(60, "the merge never ended", writer.merge()).await;
    assert!(failed.is_err(), "a merge whose upload failed committed");
    assert_eq!(keeping.kept.lock().unwrap().len(), 1);
    keeping.armed.store(false, Ordering::SeqCst);
    let next = within(
        10,
        "a merge waited for good behind what a store keeps of a failed one",
        writer.merge(),
    )
    .await;
    assert!(next.unwrap().is_some());
    // Its turn gone, what the store keeps is counted still, as draining,
    // until the store lets it go.
    assert_eq!(builds.held_let_go(), 1);
    assert!(
        builds.draining() > 0,
        "what a store keeps past the bound was no longer counted"
    );
    keeping.kept.lock().unwrap().clear();
    until(5, "what a store let go was counted still", || {
        builds.draining() == 0
    })
    .await;
}

/// A turn kept for a merge that gave way to a flush is let go as soon as its
/// writer's next merge finds it may not merge, as with its lease taken by
/// another writer, not only once kept as long as it is kept at most.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_kept_for_a_merge_is_let_go_once_its_next_attempt_may_not_merge() {
    let builds = MergeBuilds::new(1);
    builds.hold(Some(Holding::UntilLetGo));
    let due = Arc::new(AtomicBool::new(false));
    let (_first, building, second, other) = one_gave_way(&builds, &due).await;
    builds.let_one_go();
    within(60, "the first merge never ended", building)
        .await
        .unwrap()
        .unwrap();
    until(
        5,
        "the turn was not kept for the merge that gave way",
        || builds.turns_free() == 0,
    )
    .await;
    // Another writer takes the lease.
    let _owner = other.writer().await;
    let merged = within(10, "a merge on a lease taken never ended", second.merge()).await;
    assert_eq!(merged.unwrap(), None);
    until(
        5,
        "a turn kept for a merge whose writer may no longer merge was never let go",
        || builds.turns_free() == 1,
    )
    .await;
}
