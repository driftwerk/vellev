//! A merge asked to stop gives up wherever it is between its plan and its
//! commit, and leaves nothing behind: nothing committed, nothing kept from
//! cleaning up, none of its files in the store (merge note, step 3, §7).
//! A test binary of its own: its merges build clustered indexes, which
//! take turns on the process's one tuning thread.

mod common;

use common::*;
use serde_json::json;
use vellev_index::stop::Stage;
use vellev_storage::{MergeStage, Merging};

/// Notes in each of the two segments merged: the merge holds more vectors
/// than a clustered index is built above here, and more than a binary one
/// would be.
const NOTES: u32 = 2_600;

fn note(n: u32, random: &mut Random) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1_000)),
        properties: json!({ "title": format!("note {n} of many"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(random.vector(2)),
        only_if_version: None,
    }
}

/// A writer of two flushed segments of [`NOTES`] notes each, which merge
/// into one with a clustered index.
async fn two_large(cluster: &Cluster) -> Arc<Writer> {
    let merging = Merging {
        fanout: 2,
        clustered_above: 4_000,
        ..Merging::default()
    };
    let writer = Arc::new(cluster.writer().await.merging_with(merging));
    writer.commit(vec![create("notes")]).await.unwrap();
    let mut random = Random(7);
    for round in 0..2 {
        let notes = (0..NOTES).map(|n| note(round * NOTES + n, &mut random));
        writer.commit(notes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    writer
}

/// The keys under the partition that hold `part`.
async fn stored(cluster: &Cluster, part: &str) -> Vec<String> {
    let listed = cluster.blobs.list(&Layout(&cluster.key).prefix()).await;
    let keys = listed.unwrap().into_iter().map(|file| file.key);
    keys.filter(|key| key.contains(part)).collect()
}

/// Wait, up to a minute, until `holds`.
async fn settled<F: std::future::Future<Output = bool>>(what: &str, holds: impl Fn() -> F) {
    let waited = tokio::time::timeout(Duration::from_secs(60), async {
        while !holds().await {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    waited.await.unwrap_or_else(|_| panic!("{what}"));
}

/// What `work` comes to within `limit`: a merge that never gives up, or a
/// flush held behind one, fails the rule in seconds rather than hanging it.
async fn within<T>(limit: u64, what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(limit), work)
        .await
        .unwrap_or_else(|_| panic!("{what} within {limit} s"))
}

/// Every note, read by a copy loaded afresh.
async fn all_there(cluster: &Cluster) {
    let count = |state: &State| state.count(&name("notes"));
    assert_eq!(
        cluster.open().await.partition().read(count),
        2 * NOTES as usize
    );
}

/// The stages of a merge, each stopped in on its own.
const STAGES: [MergeStage; 12] = [
    MergeStage::Planning,
    MergeStage::Gathering,
    MergeStage::Turn,
    MergeStage::Indexing,
    MergeStage::Vectors(Stage::Sampling),
    MergeStage::Vectors(Stage::Clustering),
    MergeStage::Vectors(Stage::Coding),
    MergeStage::Vectors(Stage::Tuning),
    MergeStage::Vectors(Stage::Ranking),
    MergeStage::Encoding,
    MergeStage::Uploading,
    MergeStage::Loading,
];

/// A merge stopped in any of its stages gives up there: it commits
/// nothing, keeps nothing from cleaning up, and leaves none of its files in
/// the store. The stop is the merge's own: the next merge commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_in_any_stage_gives_up_there_and_leaves_nothing() {
    for stage in STAGES {
        let cluster = Cluster::new().await;
        let writer = two_large(&cluster).await;
        let before = cluster.committed().await;
        writer.stop_merges_at(Some(stage));
        let merged = within(120, "the merge never gave up", writer.merge()).await;
        assert_eq!(
            merged.unwrap(),
            None,
            "a merge stopped in {stage:?} committed"
        );
        assert_eq!(
            cluster.committed().await.manifest,
            before.manifest,
            "{stage:?}"
        );
        assert_eq!(writer.kept_while_merging(), 0, "{stage:?}");
        settled(
            &format!("files of a merge stopped in {stage:?} stay"),
            || async { stored(&cluster, "-merged").await.is_empty() },
        )
        .await;
        all_there(&cluster).await;

        writer.stop_merges_at(None);
        let next = within(120, "the next merge never ended", writer.merge()).await;
        assert!(
            next.unwrap().is_some(),
            "after a merge stopped in {stage:?}, the next one commits"
        );
        all_there(&cluster).await;
    }
}

/// A merge stopped from outside while one of its files is going up gives
/// up without waiting for the upload, and deletes what it put.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_while_it_uploads_gives_up_and_deletes_what_it_put() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged.index");
    let writer = two_large(&cluster).await;
    let before = cluster.committed().await;
    paused.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let stored_one = async { while seen.recv().await.expect("the store is there") != "stored" {} };
    tokio::time::timeout(Duration::from_secs(120), stored_one)
        .await
        .expect("the merge's index goes up");
    assert!(!stored(&cluster, "-merged").await.is_empty());
    writer.stop_merges();
    let merged = tokio::time::timeout(Duration::from_secs(10), merging)
        .await
        .expect("the stopped merge waits for no upload")
        .unwrap()
        .unwrap();
    assert_eq!(merged, None);
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    assert_eq!(writer.kept_while_merging(), 0);
    // The upload given up on settles, and only then is what it put
    // deleted.
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_waiters();
    settled("the stopped merge's files stay", || async {
        stored(&cluster, "-merged").await.is_empty()
    })
    .await;
    all_there(&cluster).await;
}

/// Stopping a merge stops no flush: a flush made while a stopped merge
/// gives up commits what it was given.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stopping_a_merge_stops_no_flush() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged.index");
    let writer = two_large(&cluster).await;
    paused.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let stored_one = async { while seen.recv().await.expect("the store is there") != "stored" {} };
    tokio::time::timeout(Duration::from_secs(120), stored_one)
        .await
        .expect("the merge's index goes up");
    writer.stop_merges();
    let mut random = Random(11);
    let more: Vec<Mutation> = (0..NOTES)
        .map(|n| note(2 * NOTES + n, &mut random))
        .collect();
    within(30, "the commit was held", writer.commit(more))
        .await
        .unwrap();
    within(60, "the flush was held", writer.flush())
        .await
        .expect("the flush commits");
    let merged = within(10, "the merge never gave up", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    let count = |state: &State| state.count(&name("notes"));
    assert_eq!(
        cluster.open().await.partition().read(count),
        3 * NOTES as usize
    );
}

/// A merge whose lease is taken by another writer while it builds never
/// commits: the head stays the new owner's (rule 3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_whose_lease_is_taken_mid_build_never_commits() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged.index");
    let writer = two_large(&cluster).await;
    paused.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let stored_one = async { while seen.recv().await.expect("the store is there") != "stored" {} };
    tokio::time::timeout(Duration::from_secs(120), stored_one)
        .await
        .expect("the merge's index goes up");
    let owner = cluster.writer().await;
    let taken = cluster.committed().await;
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_waiters();
    let merged = within(120, "the merge never ended", merging).await.unwrap();
    assert!(!matches!(merged, Ok(Some(_))), "{merged:?}");
    let now = cluster.committed().await;
    assert_eq!(now.manifest, taken.manifest, "the head is the new owner's");
    assert_eq!(now.generation, owner.lease().generation);
    all_there(&cluster).await;
}

/// A merge planned on a lease already taken builds and uploads nothing: it
/// finds out at once, not at its commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_on_a_lease_already_taken_uploads_nothing() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged");
    let writer = two_large(&cluster).await;
    let _owner = cluster.writer().await;
    paused.armed.store(true, Ordering::SeqCst);
    let merged = within(
        30,
        "a merge on a lost lease went on to upload",
        writer.merge(),
    )
    .await;
    assert!(!matches!(merged, Ok(Some(_))), "{merged:?}");
    while let Ok(event) = seen.try_recv() {
        assert_ne!(event, "stored", "a merge on a lost lease uploaded a file");
    }
    assert!(stored(&cluster, "-merged").await.is_empty());
}

/// Rule 4a, timed: a merge of a size where each of its stages takes
/// seconds, stopped at moments spread from its start to its end, gives up
/// within 2 s of each stop. Run in release, on CI, as
/// `cargo test --release -p vellev-storage --test merge_stop -- --ignored`;
/// `VELLEV_STOP_ROWS` sets the rows in each of the two segments merged.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "timed, in release, at a size where each stage takes seconds"]
async fn a_merge_stopped_at_any_moment_gives_up_within_two_seconds() {
    let rows: u32 = std::env::var("VELLEV_STOP_ROWS")
        .ok()
        .and_then(|rows| rows.parse().ok())
        .unwrap_or(250_000);
    let cluster = Cluster::new().await;
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = Arc::new(cluster.writer().await.merging_with(merging));
    let mut big = schema("notes");
    big.embedding.dimensions = 384;
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(big))])
        .await
        .unwrap();
    let mut random = Random(13);
    for round in 0..2 {
        for start in (0..rows).step_by(10_000) {
            let notes = (start..rows.min(start + 10_000)).map(|n| {
                let mut note = note(round * rows + n, &mut random);
                if let Mutation::Upsert { vector, .. } = &mut note {
                    *vector = Some(random.vector(384));
                }
                note
            });
            writer.commit(notes.collect()).await.unwrap();
        }
        writer.flush().await.unwrap();
    }
    let before = cluster.committed().await;

    let mut worst = Duration::ZERO;
    let mut stopped = 0;
    let mut at = Duration::from_millis(500);
    loop {
        let begun = std::time::Instant::now();
        let merging = tokio::spawn({
            let writer = writer.clone();
            async move { writer.merge().await }
        });
        tokio::time::sleep(at).await;
        if merging.is_finished() {
            let merged = merging.await.unwrap().unwrap();
            assert!(merged.is_some(), "the merge unstopped commits");
            println!("merged unstopped in {:.1} s", begun.elapsed().as_secs_f64());
            break;
        }
        writer.stop_merges();
        let asked = std::time::Instant::now();
        let merged = within(60, "the merge never gave up", merging).await;
        let took = asked.elapsed();
        // A stop that lands once what was built is loaded back is past the
        // point of no return: the merge commits, by design, and the sweep
        // ends there. Only then: a commit after a stop it was told of in
        // time would be a stop not heeded.
        if merged.as_ref().unwrap().as_ref().unwrap().is_some() {
            assert_eq!(
                writer.stops_past_loading(),
                1,
                "a merge stopped at {at:?} committed before it had loaded back"
            );
            println!(
                "stopped at {:.1} s, past loading back: committed in {:.3} s",
                at.as_secs_f64(),
                took.as_secs_f64()
            );
            break;
        }
        assert_eq!(cluster.committed().await.manifest, before.manifest);
        assert_eq!(writer.kept_while_merging(), 0);
        println!(
            "stopped at {:.1} s: gave up in {:.3} s",
            at.as_secs_f64(),
            took.as_secs_f64()
        );
        worst = worst.max(took);
        stopped += 1;
        at = at.mul_f64(1.6);
    }
    println!(
        "{stopped} stops, the slowest given up in {:.3} s",
        worst.as_secs_f64()
    );
    assert!(stopped >= 5, "too few stops to say: {stopped}");
    assert!(worst <= Duration::from_secs(2), "a stop took {worst:?}");
}

/// A merge waiting its turn behind another when merges are stopped is
/// stopped too: once its turn comes it gives up before it gathers a row,
/// and uploads nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_waiting_its_turn_when_merges_are_stopped_is_stopped_too() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged.index");
    let writer = two_large(&cluster).await;
    let before = cluster.committed().await;
    paused.armed.store(true, Ordering::SeqCst);
    let first = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let stored_one = async { while seen.recv().await.expect("the store is there") != "stored" {} };
    within(120, "the first merge's index went up", stored_one).await;
    let queued = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    // Waiting its turn, behind the first.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!queued.is_finished());
    writer.stop_merges();
    let first = within(10, "the first merge never gave up", first).await;
    assert_eq!(first.unwrap().unwrap(), None);
    // Were it to build, its index's upload would be held, and it would
    // never end.
    let queued = within(30, "the merge waiting its turn never gave up", queued).await;
    assert_eq!(queued.unwrap().unwrap(), None);
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    assert_eq!(writer.kept_while_merging(), 0);
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_waiters();
    settled("the stopped merges' files stay", || async {
        stored(&cluster, "-merged").await.is_empty()
    })
    .await;
    assert!(within(120, "the next merge", writer.merge())
        .await
        .unwrap()
        .is_some());
}

/// A store whose uploads of one kind, once armed, are written only when
/// let land, on a task of their own that goes on if whoever uploads stops
/// waiting: as a local store writes on a blocking thread, or a store
/// answers the last request of an upload in parts after it was sent.
#[derive(Debug)]
struct Late {
    inner: Arc<dyn Blobs>,
    kind: &'static str,
    armed: AtomicBool,
    putting: tokio::sync::mpsc::UnboundedSender<()>,
    land: Arc<tokio::sync::Notify>,
}

#[async_trait]
impl Blobs for Late {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if !(self.armed.load(Ordering::SeqCst) && key.contains(self.kind)) {
            return self.inner.put(key, bytes).await;
        }
        let (inner, land, key) = (self.inner.clone(), self.land.clone(), key.to_string());
        let landing = tokio::spawn(async move {
            land.notified().await;
            inner.put(&key, bytes).await
        });
        self.putting.send(()).unwrap();
        landing.await.expect("the write lands")
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

/// A file a stopped merge was uploading that lands after the merge gave up,
/// as a local store's write on a blocking thread may, is referenced by
/// nothing and kept by nothing: cleaning up deletes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopped_upload_that_lands_late_is_left_to_cleaning_up() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    let (putting, mut put) = tokio::sync::mpsc::unbounded_channel();
    let late = Arc::new(Late {
        inner: Arc::new(ObjectBlobs::new(Arc::new(local))),
        kind: "-merged.index",
        armed: AtomicBool::new(false),
        putting,
        land: Arc::default(),
    });
    cluster.blobs = late.clone();
    let writer = two_large(&cluster).await;
    let before = cluster.committed().await;
    late.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    within(120, "the merge's index began to go up", put.recv()).await;
    writer.stop_merges();
    let merged = within(10, "the merge never gave up", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    assert_eq!(writer.kept_while_merging(), 0);
    // The index lands only now, after the merge has given up and deleted
    // what it had put.
    tokio::time::sleep(Duration::from_millis(200)).await;
    late.land.notify_one();
    settled("the late file lands", || async {
        !stored(&cluster, "-merged.index").await.is_empty()
    })
    .await;
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    // Once the partition has moved past where the merge named its files,
    // cleaning up deletes what landed late.
    let mut random = Random(17);
    for round in 0..3 {
        writer
            .commit(vec![note(9_000 + round, &mut random)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    writer.clean(Duration::ZERO).await.unwrap();
    assert!(stored(&cluster, "-merged").await.is_empty());
    let count = |state: &State| state.count(&name("notes"));
    assert_eq!(
        cluster.open().await.partition().read(count),
        2 * NOTES as usize + 3
    );
}

/// A stop that comes once what was built is loaded back is too late: the
/// merge commits, as it would have, and keeps nothing after.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stop_once_loaded_back_is_too_late_and_the_merge_commits() {
    let cluster = Cluster::new().await;
    let writer = two_large(&cluster).await;
    let gate = writer.hold_merges_before_commit();
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    within(120, "the merge was built", gate.reached.notified()).await;
    assert!(!writer.stop_merges(), "a stop too late stops no merge");
    gate.release.notify_one();
    let merged = within(60, "the merge never ended", merging).await;
    assert!(
        merged.unwrap().unwrap().is_some(),
        "past loading back, it commits"
    );
    assert_eq!(
        writer.stops_past_loading(),
        1,
        "the stop was seen, too late"
    );
    assert_eq!(writer.kept_while_merging(), 0);
    assert!(!writer.merge_under_way());
    all_there(&cluster).await;
}

/// A store whose local copies of merged vectors, once armed, are not said
/// to be anywhere until let go: a merge's load-back held where it waits
/// on the disk.
#[derive(Debug)]
struct HeldLoad {
    inner: CachedBlobs,
    armed: AtomicBool,
    loading: tokio::sync::mpsc::UnboundedSender<()>,
    release: tokio::sync::Notify,
}

#[async_trait]
impl Blobs for HeldLoad {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
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
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        if self.armed.load(Ordering::SeqCst) && key.contains("-merged.vectors") {
            self.loading.send(()).unwrap();
            self.release.notified().await;
        }
        self.inner.local(key).await
    }
}

/// A merge stopped while it loads what it built back from local disk gives
/// up without waiting for the disk, and leaves nothing.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_while_it_loads_back_gives_up_and_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    let disk = vellev_storage::Disk::new(Duration::from_secs(10), 64 * 1024 * 1024, 4);
    let (loading, mut loads) = tokio::sync::mpsc::unbounded_channel();
    let held = Arc::new(HeldLoad {
        inner: CachedBlobs::new(cluster.blobs.clone(), dir.path()).on(disk),
        armed: AtomicBool::new(false),
        loading,
        release: tokio::sync::Notify::new(),
    });
    cluster.blobs = held.clone();
    let writer = two_large(&cluster).await;
    let before = cluster.committed().await;
    held.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    within(120, "the merge came to load back", loads.recv()).await;
    writer.stop_merges();
    let merged = within(10, "the merge never gave up", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    assert_eq!(writer.kept_while_merging(), 0);
    held.armed.store(false, Ordering::SeqCst);
    held.release.notify_waiters();
    settled("the stopped merge's files stay", || async {
        stored(&cluster, "-merged").await.is_empty()
    })
    .await;
    all_there(&cluster).await;
}

/// A merge whose build panics leaves no stop behind: nothing says a merge
/// is under way, and the next merge commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_whose_build_panics_leaves_no_stop_behind() {
    let cluster = Cluster::new().await;
    let writer = two_large(&cluster).await;
    writer.panic_merges_in_build(true);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let ended = within(120, "the merge that panics never ended", merging).await;
    assert!(
        matches!(ended, Err(_) | Ok(Err(_))),
        "a build that panics is no merge: {ended:?}"
    );
    assert!(!writer.merge_under_way(), "its stop is left behind");
    assert_eq!(writer.kept_while_merging(), 0);
    writer.panic_merges_in_build(false);
    assert!(within(120, "the next merge", writer.merge())
        .await
        .unwrap()
        .is_some());
}

/// A merge stopped while it sums what it built, read back in place from
/// local disk, gives up at the next piece, and leaves nothing.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_while_it_sums_what_it_built_gives_up_and_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    let disk = vellev_storage::Disk::new(Duration::from_secs(10), 64 * 1024 * 1024, 4);
    cluster.blobs = Arc::new(CachedBlobs::new(cluster.blobs.clone(), dir.path()).on(disk));
    let writer = two_large(&cluster).await;
    let before = cluster.committed().await;
    writer.stop_merges_at(Some(MergeStage::Summing));
    let merged = within(120, "the merge never gave up", writer.merge()).await;
    assert_eq!(
        merged.unwrap(),
        None,
        "a merge stopped as it sums committed"
    );
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    assert_eq!(writer.kept_while_merging(), 0);
    settled("the stopped merge's files stay", || async {
        stored(&cluster, "-merged").await.is_empty()
    })
    .await;
    writer.stop_merges_at(None);
    assert!(within(120, "the next merge", writer.merge())
        .await
        .unwrap()
        .is_some());
    all_there(&cluster).await;
}

/// A store that counts the reads of merged vectors made of it.
#[derive(Debug)]
struct ReadsCounted {
    inner: Arc<dyn Blobs>,
    merged_reads: AtomicUsize,
}

#[async_trait]
impl Blobs for ReadsCounted {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if key.contains("-merged.vectors") {
            self.merged_reads.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// What is logged, kept to be looked at.
#[derive(Clone, Default)]
struct Logged(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Logged {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A merge stopped as it sums what it built is not taken for a copy found
/// wanting: nothing says the copy is not read in place, and nothing is
/// fetched from the store afresh in its stead.
#[cfg(unix)]
#[tokio::test]
async fn a_merge_stopped_while_it_sums_fetches_nothing_and_warns_of_nothing() {
    let logged = Logged::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer({
            let logged = logged.clone();
            move || logged.clone()
        })
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .finish();
    let _logging = tracing::subscriber::set_default(subscriber);
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    let counted = Arc::new(ReadsCounted {
        inner: cluster.blobs.clone(),
        merged_reads: AtomicUsize::new(0),
    });
    let disk = vellev_storage::Disk::new(Duration::from_secs(10), 64 * 1024 * 1024, 4);
    cluster.blobs = Arc::new(CachedBlobs::new(counted.clone(), dir.path()).on(disk));
    let writer = two_large(&cluster).await;
    writer.stop_merges_at(Some(MergeStage::Summing));
    let merged = within(120, "the merge never gave up", writer.merge()).await;
    assert_eq!(merged.unwrap(), None);
    assert_eq!(
        counted.merged_reads.load(Ordering::SeqCst),
        0,
        "the merged vectors were fetched afresh"
    );
    let logged = String::from_utf8(logged.0.lock().unwrap().clone()).unwrap();
    assert!(!logged.contains("not read in place"), "{logged}");
    assert!(!logged.contains("is not kept"), "{logged}");
}
