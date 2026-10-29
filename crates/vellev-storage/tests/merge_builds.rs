//! A merge's turn among a node's merge builds (merge note, step 3, §7 and
//! §7.7): taken before it gathers a row, given up at its stop while it
//! waits, never waited for by a merge with nothing to do, and held, with
//! its count, until the last work that holds what it built has ended, an
//! upload left to settle among it.

mod common;

use common::*;
use vellev_storage::backups::Held;
use vellev_storage::coordinator::{Clock, Restore};
use vellev_storage::{Holding, MergeBuilds, MergeStage, Merging};

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

/// A merge never holds a turn while it waits for its partition's building
/// lock, as a flush or a restore holds it for long: it gives the turn back
/// meanwhile, and another tenant's merge takes it. Once the lock is free,
/// it takes a turn again and commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_turn_is_not_held_while_its_partition_s_building_lock_is() {
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
    let waiting = tokio::spawn({
        let first = first.clone();
        async move { first.merge().await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!waiting.is_finished());
    let second = flushed(&other, &builds, 2).await;
    let merged = within(
        10,
        "a merge waited for a turn another held while waiting for its own building lock",
        second.merge(),
    )
    .await;
    assert!(merged.unwrap().is_some());
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
