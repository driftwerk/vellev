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
    assert_eq!(cluster.open().await.partition().read(count), 2 * NOTES as usize);
}

/// The stages of a merge, each stopped in on its own.
const STAGES: [MergeStage; 11] = [
    MergeStage::Planning,
    MergeStage::Gathering,
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
        assert_eq!(merged.unwrap(), None, "a merge stopped in {stage:?} committed");
        assert_eq!(cluster.committed().await.manifest, before.manifest, "{stage:?}");
        assert_eq!(writer.kept_while_merging(), 0, "{stage:?}");
        settled(&format!("files of a merge stopped in {stage:?} stay"), || async {
            stored(&cluster, "-merged").await.is_empty()
        })
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
    settled("the stopped merge's files stay", || async {
        stored(&cluster, "-merged").await.is_empty()
    })
    .await;
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_waiters();
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
    let more: Vec<Mutation> = (0..NOTES).map(|n| note(2 * NOTES + n, &mut random)).collect();
    within(30, "the commit was held", writer.commit(more)).await.unwrap();
    within(60, "the flush was held", writer.flush()).await.expect("the flush commits");
    let merged = within(10, "the merge never gave up", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    let count = |state: &State| state.count(&name("notes"));
    assert_eq!(cluster.open().await.partition().read(count), 3 * NOTES as usize);
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
    let merged = tokio::time::timeout(Duration::from_secs(120), writer.merge())
        .await
        .expect("the merge ends without anything going up");
    assert!(!matches!(merged, Ok(Some(_))), "{merged:?}");
    while let Ok(event) = seen.try_recv() {
        assert_ne!(event, "stored", "a merge on a lost lease uploaded a file");
    }
    assert!(stored(&cluster, "-merged").await.is_empty());
}
