//! A merge's gather (node memory note, M4): planned in moments under the
//! building lock and the state's, then copied with neither held, off the
//! async threads, a chunk of rows at a time. So a flush, a write and a
//! search go on through it, and a stop or a restore ends it within a chunk.
//! The rules of the same name that need a node are in the node's tests.

mod common;

use common::*;
use vellev_storage::timings::{MergeEnd, MergePhase, Timed};
use vellev_storage::{Holding, MergeBuilds, Merging, Query, GATHERED_AT_ONCE};

/// Rows in each of the two segments merged: several chunks, so that a
/// gather held at its second has most of a segment left to copy.
const ROWS: u32 = 3 * GATHERED_AT_ONCE as u32 + 1_000;

/// How many rows a write carries.
const WRITTEN_AT_ONCE: u32 = 2_000;

fn row(n: u32) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1_000))
}

fn vector(n: u32) -> Vec<f32> {
    let n = n as f32;
    vec![n.sin(), n.cos()]
}

/// Row `n` as `value`.
fn upsert_as(n: u32, value: i64) -> Mutation {
    upsert_in("notes", n, value)
}

/// Row `n` of `collection` as `value`.
fn upsert_in(collection: &str, n: u32, value: i64) -> Mutation {
    Mutation::Upsert {
        collection: name(collection),
        id: row(n),
        properties: json!({ "title": format!("note {n} of many"), "n": value })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vector(n)),
        only_if_version: None,
    }
}

fn gone(n: u32) -> Mutation {
    Mutation::Delete {
        collection: name("notes"),
        id: row(n),
        only_if_version: None,
    }
}

/// A writer among `builds` of two flushed segments of [`ROWS`] rows each,
/// row `n` holding `n`, which merge into one.
async fn two_large(cluster: &Cluster, builds: &Arc<MergeBuilds>) -> Arc<Writer> {
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
    within(LONG, "the writer was written", async {
        writer.commit(vec![create("notes")]).await.unwrap();
        for segment in 0..2 {
            for from in (0..ROWS).step_by(WRITTEN_AT_ONCE as usize) {
                let to = (from + WRITTEN_AT_ONCE).min(ROWS);
                let rows = (from..to).map(|n| segment * ROWS + n);
                let rows = rows.map(|n| upsert_as(n, i64::from(n)));
                writer.commit(rows.collect()).await.unwrap();
            }
            writer.flush().await.unwrap();
        }
    })
    .await;
    writer
}

/// What `work` comes to within `limit`: a gather that holds up what it
/// should not, or never gives up, fails its rule in seconds.
async fn within<T>(limit: Duration, what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(limit, work)
        .await
        .unwrap_or_else(|_| panic!("{what} within {limit:?}"))
}

const SECOND: Duration = Duration::from_secs(1);
const LONG: Duration = Duration::from_secs(60);

/// Wait until `holds`, at most `limit`.
async fn until(limit: Duration, what: &str, holds: impl Fn() -> bool) {
    within(limit, what, async {
        while !holds() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
}

/// Gathers held, until this is dropped, however the rule ends: so that one
/// that fails with a merge held fails in seconds, not at the hold's end.
struct Held(Arc<MergeBuilds>);

impl Held {
    fn let_go(&self) {
        self.0.hold(None);
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.let_go();
    }
}

/// A merge of `writer` begun, on a task of its own, and held part way
/// through its gather, past its first chunk.
async fn gather_held(
    writer: &Arc<Writer>,
    builds: &Arc<MergeBuilds>,
) -> (
    Held,
    tokio::task::JoinHandle<vellev_storage::Result<Option<LogPosition>>>,
) {
    builds.hold(Some(Holding::GatheringUntilLetGo));
    let held = Held(builds.clone());
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    until(LONG, "no gather came to be held", || builds.held() == 1).await;
    let copied = builds.last_gathered();
    assert!((1..=GATHERED_AT_ONCE).contains(&copied), "{copied} rows");
    (held, merging)
}

/// The rows of `collection` as `state` holds them.
fn rows_of(state: &State, collection: &str) -> BTreeMap<ObjectId, i64> {
    let (of, mut rows, mut after) = (name(collection), BTreeMap::new(), None);
    loop {
        let page = state.list(&of, None, after, 1_000).unwrap();
        let Some(last) = page.last() else {
            return rows;
        };
        after = Some(last.id());
        for found in &page {
            let Value::Int(n) = found.object.properties["n"] else {
                panic!("{:?}", found.object.properties);
            };
            assert!(
                rows.insert(found.id(), n).is_none(),
                "row {:?} found twice",
                found.id()
            );
        }
    }
}

/// `state` holds just the rows `truth` does of `collection`, each once, a
/// scan finding as many as a walk.
fn holds_in(state: &State, collection: &str, truth: &BTreeMap<u32, i64>) {
    let want: BTreeMap<ObjectId, i64> = truth.iter().map(|(n, value)| (row(*n), *value)).collect();
    let rows = rows_of(state, collection);
    let wrong = want
        .iter()
        .filter(|(id, value)| rows.get(id) != Some(*value));
    assert_eq!(wrong.count(), 0, "a row is lost, or as it was");
    assert_eq!(rows.len(), want.len(), "a row is brought back");
    assert_eq!(state.count(&name(collection)), want.len());
}

/// What a merge's writer is to hold: the rows of each collection, and the
/// rows of the notes deleted.
struct Truth {
    notes: BTreeMap<u32, i64>,
    extra: BTreeMap<u32, i64>,
    deleted: Vec<u32>,
}

impl Truth {
    /// `state` holds just these, the marks of each segment being what the
    /// rows that are live leave, and a search finds no row deleted.
    fn held_by(&self, state: &State) {
        holds_in(state, "notes", &self.notes);
        holds_in(state, "extra", &self.extra);
        assert!(state.dead_rows_agree(), "the marks are not the live rows'");
        for n in &self.deleted {
            let near = vector(*n);
            let hits = state
                .search(&name("notes"), &Query::nearest(&near, 3))
                .unwrap();
            assert!(hits.iter().all(|hit| hit.object.id != row(*n)), "{n} found");
        }
    }
}

/// `mutations` committed, in moments whatever a gather is doing.
async fn written(writer: &Writer, mutations: Vec<Mutation>) {
    let committed = within(5 * SECOND, "the write committed", writer.commit(mutations));
    committed.await.unwrap();
}

/// M4-G1: with the gather held mid-copy, a flush of the same tenant takes
/// the building lock and commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flush_commits_while_a_gather_is_held() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let (held, merging) = gather_held(&writer, &builds).await;
    written(&writer, vec![upsert_as(0, -1)]).await;
    let before = cluster.committed().await;
    within(5 * SECOND, "the flush took the lock", writer.flush())
        .await
        .unwrap();
    let flushed = cluster.committed().await;
    assert!(flushed.log_position > before.log_position);
    assert_ne!(flushed.manifest, before.manifest, "the flush committed");
    assert_eq!(builds.held(), 1, "and the gather is held still");
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some(), "the merge commits");
    let got = writer
        .partition()
        .read(|state| rows_of(state, "notes")[&row(0)]);
    assert_eq!(got, -1);
}

/// M4-G2: with the gather held mid-copy, a write to the same tenant
/// commits, and a search answers, within a second: the search asked once
/// the write has had the time to wait for the state's lock, were the lock
/// held, as a lock that lets no reader in past a waiting writer would show.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_commits_and_a_search_answers_while_a_gather_is_held() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let (held, merging) = gather_held(&writer, &builds).await;
    let writing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(vec![upsert_as(2 * ROWS, 7)]).await }
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let reading = tokio::task::spawn_blocking({
        let partition = writer.partition().clone();
        move || {
            partition.read(|state| {
                let near = vector(1);
                let asked = Query::nearest(&near, 1);
                state.search(&name("notes"), &asked).unwrap().len()
            })
        }
    });
    let found = within(SECOND, "the search answered", reading).await;
    assert_eq!(found.unwrap(), 1);
    let written = within(SECOND, "the write committed", writing).await;
    written.unwrap().unwrap();
    assert_eq!(builds.held(), 1, "and the gather is held still");
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some(), "the merge commits");
}

/// M4-G3: a restore made during the gather ends it within a chunk, by the
/// epoch it moved, its refusal of writes being over by then; and nothing is
/// committed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_made_during_a_gather_ends_it_within_a_chunk() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let committed = cluster.committed().await;
    let (head, position) = (committed.manifest.unwrap(), committed.log_position);
    written(&writer, vec![upsert_as(0, -1)]).await;
    let (held, merging) = gather_held(&writer, &builds).await;
    let restoring = writer.restore(Some(&head), position);
    within(LONG, "the restore went through", restoring)
        .await
        .unwrap();
    let restored = cluster.committed().await;
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None, "the merge is dropped");
    assert!(
        builds.last_gathered() <= 2 * GATHERED_AT_ONCE,
        "the gather went on past the restore, to {} rows",
        builds.last_gathered()
    );
    assert_eq!(cluster.committed().await.manifest, restored.manifest);
    assert_eq!(writer.kept_while_merging(), 0);
}

/// How the writer's merges ended, those that did.
fn ended(writer: &Writer) -> Vec<(MergeEnd, u64)> {
    let timings = writer.timings();
    MergeEnd::ALL
        .iter()
        .map(|end| (*end, timings.merges_ended(*end)))
        .filter(|(_, count)| *count > 0)
        .collect()
}

/// §9.7 rule 6: a gather a restore ended is counted stale, whether the
/// restore was made or is under way, never as a run found changed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gather_ended_by_a_restore_made_is_counted_stale() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let committed = cluster.committed().await;
    let (head, position) = (committed.manifest.unwrap(), committed.log_position);
    written(&writer, vec![upsert_as(0, -1)]).await;
    let (held, merging) = gather_held(&writer, &builds).await;
    let restoring = writer.restore(Some(&head), position);
    within(LONG, "the restore went through", restoring)
        .await
        .unwrap();
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None, "the merge is dropped");
    assert_eq!(ended(&writer), vec![(MergeEnd::Stale, 1)]);
}

/// A merge's gather is timed for the copy, however long it runs, and its
/// build begins after it ends: neither counts the other (§9.7 rule 5).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_gather_is_timed_as_the_copy_and_apart_from_the_build() {
    const HELD: Duration = Duration::from_millis(500);
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let timings = writer.timings().clone();
    timings.keep_events(true);
    let (held, merging) = gather_held(&writer, &builds).await;
    tokio::time::sleep(HELD).await;
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some(), "the merge committed");
    let events = timings.drain_events();
    let one = |phase: MergePhase| {
        let mut of = events
            .iter()
            .filter(|event| event.timed == Timed::Merge(phase));
        let event = of.next().expect("a phase timed");
        assert!(of.next().is_none(), "{phase:?} timed twice");
        event.clone()
    };
    let (gather, build) = (one(MergePhase::Gather), one(MergePhase::Build));
    assert!(gather.took >= HELD, "the copy timed {:?}", gather.took);
    assert!(
        build.at - build.took >= gather.at,
        "the build began before the gather ended: it took {:?}",
        build.took
    );
}

/// A store whose gets, once armed, wait until let go.
#[derive(Debug)]
struct GetsHeld {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    reached: tokio::sync::Notify,
    let_go: tokio::sync::Semaphore,
}

#[async_trait]
impl Blobs for GetsHeld {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if self.armed.load(Ordering::SeqCst) {
            self.reached.notify_one();
            drop(self.let_go.acquire().await);
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

/// M4-G3, while it runs: a restore under way, which has moved no epoch yet,
/// ends the gather within a chunk too, by the refusal of writes it began
/// with: so that a long restore does not leave a gather running beside it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_under_way_ends_a_gather_within_a_chunk() {
    let (builds, mut cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let store = Arc::new(GetsHeld {
        inner: cluster.blobs.clone(),
        armed: AtomicBool::new(false),
        reached: tokio::sync::Notify::new(),
        let_go: tokio::sync::Semaphore::new(0),
    });
    cluster.blobs = store.clone();
    let writer = two_large(&cluster, &builds).await;
    let committed = cluster.committed().await;
    let (head, position) = (committed.manifest.unwrap(), committed.log_position);
    written(&writer, vec![upsert_as(0, -1)]).await;
    let (held, merging) = gather_held(&writer, &builds).await;
    let epoch = writer.epoch();
    store.armed.store(true, Ordering::SeqCst);
    let restoring = tokio::spawn({
        let writer = writer.clone();
        async move { writer.restore(Some(&head), position).await }
    });
    within(
        LONG,
        "the restore came to the store",
        store.reached.notified(),
    )
    .await;
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None, "the merge is dropped");
    assert_eq!(writer.epoch(), epoch, "with the restore under way still");
    assert_eq!(ended(&writer), vec![(MergeEnd::Stale, 1)]);
    assert!(
        builds.last_gathered() <= 2 * GATHERED_AT_ONCE,
        "the gather went on through the restore, to {} rows",
        builds.last_gathered()
    );
    store.armed.store(false, Ordering::SeqCst);
    store
        .let_go
        .add_permits(tokio::sync::Semaphore::MAX_PERMITS / 2);
    let restored = within(LONG, "the restore went through", restoring).await;
    restored.unwrap().unwrap();
}

/// M4-G4: a stop mid-segment ends the gather within a chunk: nothing is
/// committed, and the next merge is the stop's no more.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_mid_segment_ends_a_gather_within_a_chunk() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let before = cluster.committed().await;
    let (held, merging) = gather_held(&writer, &builds).await;
    assert!(writer.stop_merges(), "a merge was under way");
    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None, "the merge is dropped");
    assert!(
        builds.last_gathered() <= 2 * GATHERED_AT_ONCE,
        "the gather went on past its stop, to {} rows",
        builds.last_gathered()
    );
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    assert_eq!(builds.under_way(), 0);
    let next = within(LONG, "the next merge ended", writer.merge()).await;
    assert!(next.unwrap().is_some(), "the next one commits");
}

/// M4-G4, at the end of the copy: a stop that comes once the last chunk is
/// copied ends the gather before its frame is sealed, a sum over every
/// vector: no build begins, held where it would begin were it to.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stop_after_the_last_chunk_ends_a_gather_before_any_build() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let before = cluster.committed().await;
    let (held, merging) = gather_held(&writer, &builds).await;
    let whole = 2 * ROWS as usize;
    while builds.last_gathered() < whole {
        let copied = builds.last_gathered();
        builds.let_one_go();
        let next = || builds.last_gathered() > copied && builds.held() == 1;
        until(10 * SECOND, "the gather asked at its next chunk", next).await;
    }
    builds.hold(Some(Holding::UntilLetGo));
    assert!(writer.stop_merges(), "a merge was under way");
    builds.let_one_go();
    let merged = within(10 * SECOND, "the merge ended, unbuilt", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None, "the merge is dropped");
    assert_eq!(builds.held(), 0, "and no build began");
    assert_eq!(cluster.committed().await.manifest, before.manifest);
    drop(held);
}

/// A merge dropped mid-gather ends the gather within a chunk, and the
/// gather holds the merge's turn to its end: what it has copied is held,
/// though nobody waits for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merge_dropped_mid_gather_holds_its_turn_until_the_gather_ends() {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let writer = two_large(&cluster, &builds).await;
    let (held, merging) = gather_held(&writer, &builds).await;
    merging.abort();
    let dropped = within(LONG, "the merge was dropped", merging).await;
    assert!(dropped.unwrap_err().is_cancelled());
    assert_eq!(builds.held(), 1, "the gather runs on");
    assert_eq!(builds.under_way(), 1, "counted");
    assert_eq!(builds.turns_free(), 0, "and holds its turn");
    held.let_go();
    until(LONG, "the gather gave its turn back", || {
        builds.turns_free() == 1
    })
    .await;
    assert_eq!(builds.under_way(), 0);
    assert!(
        builds.last_gathered() <= 2 * GATHERED_AT_ONCE,
        "the gather ran on for nobody, to {} rows",
        builds.last_gathered()
    );
}

/// Rows of the second collection in each segment of the run.
const EXTRA: u32 = 300;
/// Where the rows of a segment older than the run lie, and how many: too
/// many for it to be merged in, under [`RUN_AT_MOST`].
const OLDER_FROM: u32 = 3 * ROWS;
const OLDER: u32 = 20_000;
const RUN_AT_MOST: usize = 30_000;

/// M4-G7: what is written while a gather copies goes where it would had the
/// merge not been under way, whichever side of the copy it falls, flushed
/// or not, a collection dropped and made again among it: the merge commits,
/// and no row is lost, or comes back, or is found as it was, in the writer
/// or in a copy loaded afresh, then or after a further merge. With a segment
/// `older` than the run, the run's tombstones for its rows are kept.
async fn writes_during_a_gather_stay_as_written(older: bool) {
    let (builds, cluster) = (MergeBuilds::new(1), Cluster::new().await);
    let merging = Merging {
        fanout: 2,
        rows: RUN_AT_MOST,
        ..Merging::default()
    };
    let writer = cluster
        .writer()
        .await
        .merging_with(merging)
        .merging_among(builds.clone());
    let writer = Arc::new(writer);
    let mut truth = Truth {
        notes: BTreeMap::new(),
        extra: BTreeMap::new(),
        deleted: Vec::new(),
    };
    within(LONG, "the writer was written", async {
        writer
            .commit(vec![create("notes"), create("extra")])
            .await
            .unwrap();
        if older {
            for from in (OLDER_FROM..OLDER_FROM + OLDER).step_by(WRITTEN_AT_ONCE as usize) {
                let to = (from + WRITTEN_AT_ONCE).min(OLDER_FROM + OLDER);
                let rows = (from..to).map(|n| upsert_as(n, i64::from(n)));
                writer.commit(rows.collect()).await.unwrap();
                truth.notes.extend((from..to).map(|n| (n, i64::from(n))));
            }
            writer.flush().await.unwrap();
        }
        for segment in 0..2 {
            for from in (0..ROWS).step_by(WRITTEN_AT_ONCE as usize) {
                let to = (from + WRITTEN_AT_ONCE).min(ROWS);
                let rows = (from..to).map(|n| segment * ROWS + n);
                let rows = rows.map(|n| upsert_as(n, i64::from(n)));
                writer.commit(rows.collect()).await.unwrap();
            }
            let notes = segment * ROWS..(segment + 1) * ROWS;
            truth.notes.extend(notes.map(|n| (n, i64::from(n))));
            let extra = segment * EXTRA..(segment + 1) * EXTRA;
            let rows = extra.clone().map(|n| upsert_in("extra", n, i64::from(n)));
            writer.commit(rows.collect()).await.unwrap();
            truth.extra.extend(extra.map(|n| (n, i64::from(n))));
            if older {
                // Rows of the older segment the run's tombstones kill.
                let (over, gone_too) = (OLDER_FROM + segment * 2, OLDER_FROM + segment * 2 + 1);
                let mutations = vec![upsert_as(over, -7), gone(gone_too)];
                writer.commit(mutations).await.unwrap();
                truth.notes.insert(over, -7);
                truth.notes.remove(&gone_too);
                truth.deleted.push(gone_too);
            }
            writer.flush().await.unwrap();
        }
    })
    .await;
    // Rows dead as it is planned, one of which is written again under it.
    let (dead, again) = (20, 21);
    written(&writer, vec![gone(dead), gone(again)]).await;
    truth.notes.remove(&dead);
    truth.notes.remove(&again);
    truth.deleted.push(dead);
    let (held, merging) = gather_held(&writer, &builds).await;

    // In each of the rows it has copied (under 4,096), those it is yet to
    // in the same segment, and those of the next, and those of the segment
    // older than it.
    let copied = 0..10;
    let to_copy = 9_000..9_010;
    let next = ROWS..ROWS + 10;
    let outside = if older {
        OLDER_FROM + 100..OLDER_FROM + 110
    } else {
        0..0
    };
    let rows = || {
        let all = copied.clone().chain(to_copy.clone()).chain(next.clone());
        all.chain(outside.clone())
    };
    for n in rows() {
        let mutation = match n % 3 {
            0 => {
                truth.notes.remove(&n);
                truth.deleted.push(n);
                gone(n)
            }
            1 => {
                truth.notes.insert(n, -i64::from(n) - 1);
                upsert_as(n, -i64::from(n) - 1)
            }
            _ => continue,
        };
        written(&writer, vec![mutation]).await;
    }
    let (new, reborn) = (2 * ROWS, again);
    written(&writer, vec![upsert_as(new, 1), upsert_as(reborn, 2)]).await;
    truth.notes.insert(new, 1);
    truth.notes.insert(reborn, 2);
    // A collection the run holds rows of goes.
    written(&writer, vec![Mutation::DropCollection(name("extra"))]).await;
    let flushed = within(5 * SECOND, "the flush committed", writer.flush());
    flushed.await.unwrap();
    // And over what that flush took, the rest left in memory; the
    // collection made again, under the name and with some of the ids of
    // the one that went.
    for n in rows() {
        let mutation = match n % 3 {
            1 => {
                truth.notes.insert(n, 1_000 + i64::from(n));
                upsert_as(n, 1_000 + i64::from(n))
            }
            2 => {
                truth.notes.remove(&n);
                truth.deleted.push(n);
                gone(n)
            }
            _ => continue,
        };
        written(&writer, vec![mutation]).await;
    }
    written(&writer, vec![create("extra")]).await;
    let again_in_extra = (0..5).map(|n| upsert_in("extra", n, 50 + i64::from(n)));
    written(&writer, again_in_extra.collect()).await;
    truth.extra = (0..5).map(|n| (n, 50 + i64::from(n))).collect();
    assert_eq!(builds.held(), 1, "the gather is held still");
    truth.held_by(&writer.partition().read(State::clone));

    held.let_go();
    let merged = within(LONG, "the merge ended", merging).await;
    assert!(merged.unwrap().unwrap().is_some(), "the merge commits");
    writer.partition().read(|state| truth.held_by(state));
    let loaded = within(LONG, "a copy was loaded", cluster.open()).await;
    loaded.partition().read(|state| truth.held_by(state));

    within(LONG, "the flush committed", writer.flush())
        .await
        .unwrap();
    let further = within(LONG, "the further merge ended", writer.merge()).await;
    assert!(further.unwrap().is_some(), "a further merge commits");
    writer.partition().read(|state| truth.held_by(state));
    let loaded = within(LONG, "a copy was loaded", cluster.open()).await;
    loaded.partition().read(|state| truth.held_by(state));
}

/// M4-G7, of a run from the first segment.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_during_a_gather_are_neither_lost_nor_brought_back() {
    writes_during_a_gather_stay_as_written(false).await;
}

/// M4-G7, of a run past the first segment.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_during_a_gather_of_a_later_run_are_neither_lost_nor_brought_back() {
    writes_during_a_gather_stay_as_written(true).await;
}
