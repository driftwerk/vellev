//! Merges made while writes go on, and merges cut short: whatever is
//! committed in the windows of a merge, between its segments going up and
//! its commit, is there afterwards, in the writer, in a copy opened afresh
//! and in one that follows, at every moment gone back to and after a
//! takeover and a clean; and a merge that ends at any point leaves all
//! that was committed, and nothing it uploaded and never tried to commit.

mod common;

use common::*;
use vellev_core::RequestId;
use vellev_storage::{Batch, Embedded, Merging, Partition};

/// Every live row: its collection and object, and its `n` and vector.
type View = BTreeMap<(String, u8), (i64, Vec<f32>)>;

const COLLECTIONS: [&str; 2] = ["notes", "tmp"];

/// What `state` holds, found alike by a scan, by key and by count.
fn view(state: &State) -> View {
    let mut seen = View::new();
    for collection in COLLECTIONS {
        if state.collection(&name(collection)).is_none() {
            continue;
        }
        for live in state.scan(&name(collection)) {
            let n = match live.object.properties["n"] {
                Value::Int(n) => n,
                ref other => panic!("{other:?}"),
            };
            let key = (collection.to_string(), live.id().0.as_u128() as u8);
            let vector = live.vector.map(<[f32]>::to_vec).unwrap_or_default();
            assert!(seen.insert(key, (n, vector)).is_none(), "a key live twice");
        }
    }
    // What is found by key is what a scan finds.
    for ((collection, object), (n, vector)) in &seen {
        let found = state.get(&name(collection), id(*object)).expect("by key");
        assert_eq!(n_of(state, collection, *object), Some(*n));
        assert_eq!(
            found.vector.map(<[f32]>::to_vec).unwrap_or_default(),
            *vector
        );
    }
    assert_eq!(
        state.rows(),
        seen.len(),
        "live rows and scanned rows differ"
    );
    seen
}

/// One mutation, as the model knows it.
#[derive(Debug, Clone)]
enum Op {
    Up(&'static str, u8, i64),
    Del(&'static str, u8),
    Inc(u8, i64),
    Drop(&'static str),
    Create(&'static str),
}

/// What a partition should hold: its rows and collections, and the
/// requests of one mutation each that it has answered.
#[derive(Debug, Clone, Default)]
struct Model {
    rows: View,
    collections: BTreeSet<&'static str>,
    requests: Vec<(RequestId, Op)>,
}

impl Model {
    fn mutation(op: &Op) -> Mutation {
        match op {
            Op::Up(collection, object, n) => upsert(collection, *object, *n),
            Op::Del(collection, object) => delete(collection, *object),
            Op::Inc(object, by) => Mutation::Increment {
                collection: name("notes"),
                id: id(*object),
                property: "n".into(),
                by: *by as f64,
            },
            Op::Drop(collection) => Mutation::DropCollection(name(collection)),
            Op::Create(collection) => create(collection),
        }
    }

    /// Whether `op` would be made, not refused, as things are.
    fn applies(&self, op: &Op) -> bool {
        match op {
            Op::Up(collection, ..) | Op::Del(collection, _) => {
                self.collections.contains(collection)
            }
            Op::Inc(object, _) => self.rows.contains_key(&("notes".to_string(), *object)),
            Op::Drop(collection) => self.collections.contains(collection),
            Op::Create(collection) => !self.collections.contains(collection),
        }
    }

    fn note(&mut self, op: &Op) {
        match op {
            Op::Up(collection, object, n) => {
                self.rows.insert(
                    (collection.to_string(), *object),
                    (*n, vec![*n as f32, 0.5]),
                );
            }
            Op::Del(collection, object) => {
                self.rows.remove(&(collection.to_string(), *object));
            }
            Op::Inc(object, by) => {
                self.rows
                    .get_mut(&("notes".to_string(), *object))
                    .unwrap()
                    .0 += by;
            }
            Op::Drop(collection) => {
                self.rows.retain(|(c, _), _| c != collection);
                self.collections.remove(collection);
            }
            Op::Create(collection) => {
                self.collections.insert(collection);
            }
        }
    }

    /// Commit `ops` as one batch of a request of its own, and note them.
    async fn step(&mut self, writer: &Writer, ops: &[Op]) {
        let request = RequestId::generate();
        let batch = Batch {
            request: Some(request),
            mutations: ops.iter().map(Self::mutation).collect(),
        };
        let made = writer.commit_batches(vec![batch], Embedded::new()).await;
        made.unwrap().pop().unwrap().unwrap();
        for op in ops {
            self.note(op);
        }
        if let [op] = ops {
            self.requests.push((request, op.clone()));
        }
    }
}

/// Let whatever was spawned to delete files run to its end.
async fn settle() {
    for _ in 0..200 {
        tokio::task::yield_now().await;
    }
}

/// Wait for an upload to be held, skipping the listings reported.
async fn held(seen: &mut tokio::sync::mpsc::UnboundedReceiver<&'static str>) {
    let wait = async { while seen.recv().await != Some("stored") {} };
    tokio::time::timeout(Duration::from_secs(600), wait)
        .await
        .expect("an upload to be held");
}

/// A partition with a run to merge. With `later_run` the run starts after
/// a segment too large to merge, which the run's tombstones act on.
async fn mergeable(cluster: &Cluster, later_run: bool) -> (Writer, Model) {
    let merging = Merging {
        rows: 60,
        fanout: 3,
        stale: 0.9,
        ..Merging::default()
    };
    let writer = cluster.writer().await.merging_with(merging);
    let mut model = Model::default();
    model
        .step(&writer, &[Op::Create("notes"), Op::Create("tmp")])
        .await;
    if later_run {
        let ops: Vec<Op> = (50..90).map(|n| Op::Up("notes", n, 1)).collect();
        model.step(&writer, &ops).await;
        writer.flush().await.unwrap();
    }
    for round in 0..3u8 {
        let mut ops: Vec<Op> = (0..10)
            .map(|n| Op::Up("notes", round * 10 + n, 1))
            .collect();
        ops.push(Op::Up("tmp", 200 + round, 2));
        if later_run {
            // Tombstones in the run for rows of the segment before it.
            ops.push(Op::Del("notes", 50 + round));
            ops.push(Op::Up("notes", 60 + round, 5));
        }
        model.step(&writer, &ops).await;
        writer.flush().await.unwrap();
    }
    model
        .step(&writer, &[Op::Up("notes", 100, 1), Op::Del("notes", 3)])
        .await;
    (writer, model)
}

/// The writer and a copy opened afresh both hold what `model` does, and
/// have answered every request it has.
async fn same_everywhere(cluster: &Cluster, writer: &Writer, model: &Model, what: &str) {
    assert_eq!(writer.partition().read(view), model.rows, "writer: {what}");
    let fresh = cluster.open().await;
    assert_eq!(fresh.partition().read(view), model.rows, "opened: {what}");
    for (request, _) in &model.requests {
        assert!(
            writer.partition().read(|state| state.answered(*request)),
            "writer forgot a request: {what}"
        );
        assert!(
            fresh.partition().read(|state| state.answered(*request)),
            "opened copy forgot a request: {what}"
        );
    }
}

/// Write in both windows of a merge, then go back to every moment kept,
/// take the partition over, clean it and ask every request again.
async fn in_the_windows(later_run: bool) {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment/");
    let (writer, mut model) = mergeable(&cluster, later_run).await;
    let writer = Arc::new(writer);
    let before_follower = cluster.open().await;
    let before = (cluster.committed().await, model.clone());

    paused.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    // Window one: the merged segment is built and uploaded; the snapshot of
    // what is unflushed is not yet taken.
    held(&mut seen).await;
    model.step(&writer, &[Op::Up("notes", 1, 7)]).await;
    model.step(&writer, &[Op::Del("notes", 2)]).await;
    model
        .step(&writer, &[Op::Del("notes", 4), Op::Up("notes", 4, 9)])
        .await;
    model
        .step(&writer, &[Op::Up("notes", 150, 1), Op::Del("notes", 150)])
        .await;
    model.step(&writer, &[Op::Inc(5, 3)]).await;
    model.step(&writer, &[Op::Drop("tmp")]).await;
    model.step(&writer, &[Op::Up("notes", 100, 5)]).await;
    model.step(&writer, &[Op::Del("notes", 8)]).await;
    if later_run {
        model.step(&writer, &[Op::Del("notes", 70)]).await;
        model.step(&writer, &[Op::Up("notes", 52, 4)]).await;
    }
    let first = (cluster.committed().await, model.clone());
    paused.release.notify_one();
    // Window two: the snapshot is taken and its segment uploaded; what is
    // committed now is replayed.
    held(&mut seen).await;
    assert!(!merging.is_finished());
    model.step(&writer, &[Op::Del("notes", 1)]).await;
    model.step(&writer, &[Op::Up("notes", 2, 11)]).await;
    model.step(&writer, &[Op::Del("notes", 6)]).await;
    model.step(&writer, &[Op::Up("notes", 7, 13)]).await;
    model.step(&writer, &[Op::Create("tmp")]).await;
    model.step(&writer, &[Op::Up("tmp", 200, 42)]).await;
    model.step(&writer, &[Op::Inc(5, 4)]).await;
    let middle_follower = cluster.open().await;
    let middle = (cluster.committed().await, model.clone());
    model.step(&writer, &[Op::Del("notes", 100)]).await;
    // Deleted before the snapshot and again after it, having been written
    // between: the tombstone is in the flushed segment and in the replay.
    model.step(&writer, &[Op::Up("notes", 8, 21)]).await;
    model.step(&writer, &[Op::Del("notes", 8)]).await;
    model.step(&writer, &[Op::Inc(9, 2)]).await;
    if later_run {
        model.step(&writer, &[Op::Del("notes", 71)]).await;
    }
    paused.release.notify_one();
    let merged_at = tokio::time::timeout(Duration::from_secs(600), merging)
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .expect("a merge");
    paused.armed.store(false, Ordering::SeqCst);
    let after = (cluster.committed().await, model.clone());
    assert_eq!(after.0.log_position, merged_at);
    let listed = writer.partition().read(|state| state.segments().count());
    assert_eq!(listed, if later_run { 3 } else { 2 }, "the run was merged");

    same_everywhere(&cluster, &writer, &model, "after the merge").await;
    for (follower, which) in [(&before_follower, "before"), (&middle_follower, "middle")] {
        follower.catch_up(&after.0).await.unwrap();
        assert_eq!(follower.partition().read(view), model.rows, "{which}");
    }

    // A second flush and merge on top, then more writes.
    model.step(&writer, &[Op::Up("notes", 8, 3)]).await;
    model.step(&writer, &[Op::Del("notes", 9)]).await;
    writer.flush().await.unwrap();
    same_everywhere(&cluster, &writer, &model, "after a flush on the merge").await;
    let latest = (cluster.committed().await, model.clone());

    // Back to every moment kept, and forward again.
    for (which, (at, then)) in [
        ("middle", &middle),
        ("first", &first),
        ("before", &before),
        ("after", &after),
        ("latest", &latest),
    ] {
        writer
            .restore(at.manifest.as_ref(), at.log_position)
            .await
            .unwrap();
        let mut then = then.clone();
        then.requests = model.requests.clone();
        same_everywhere(&cluster, &writer, &then, which).await;
    }
    model.step(&writer, &[Op::Up("notes", 120, 6)]).await;
    drop(writer);

    // Taken over, cleaned of everything not needed, and opened again.
    let successor = cluster.writer().await;
    same_everywhere(&cluster, &successor, &model, "taken over").await;
    successor.flush().await.unwrap();
    successor.clean(Duration::ZERO).await.unwrap();
    same_everywhere(&cluster, &successor, &model, "cleaned").await;
    // Every request is still answered as it was, and makes nothing.
    for (request, op) in model.requests.clone() {
        let batch = Batch {
            request: Some(request),
            mutations: vec![Model::mutation(&op)],
        };
        let again = successor.commit_batches(vec![batch], Embedded::new()).await;
        again.unwrap().pop().unwrap().unwrap();
    }
    same_everywhere(&cluster, &successor, &model, "asked again").await;
}

#[tokio::test(start_paused = true)]
async fn writes_in_both_windows_of_a_merge_of_the_oldest_run() {
    in_the_windows(false).await;
}

#[tokio::test(start_paused = true)]
async fn writes_in_both_windows_of_a_merge_of_a_later_run() {
    in_the_windows(true).await;
}

/// Where a merge ends short of its end.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Crash {
    /// The merge is dropped once its merged segment is uploaded.
    AfterMerged,
    /// And once the flushed segment is.
    AfterFlushed,
    /// Both segments are uploaded, and the writer is closed by a commit
    /// that failed before the merge takes its turn.
    ClosedBeforeTurn,
    /// The head is uploaded and the commit is refused.
    AfterHead,
    /// The commit lands and is never answered.
    AfterCommit,
    /// The commit lands, its answer is lost once, and the next try is told.
    AnswerLostOnce,
}

/// A merge that ends at `crash`, with writes made while it ran.
async fn crashed(crash: Crash, later_run: bool) {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment/");
    let unreliable = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let (writer, mut model) = mergeable(&cluster, later_run).await;
    drop(writer);
    let merging = Merging {
        rows: 60,
        fanout: 3,
        stale: 0.9,
        ..Merging::default()
    };
    let writer = Arc::new(
        cluster
            .writer_on(unreliable.clone())
            .await
            .merging_with(merging),
    );
    let segments_before = writer.partition().read(|state| state.segments().count());

    let segment_files = |cluster: &Cluster| {
        let (blobs, prefix) = (cluster.blobs.clone(), Layout(&cluster.key).prefix());
        async move {
            let listed = blobs.list(&prefix).await.unwrap();
            let keys = listed.into_iter().map(|file| file.key);
            keys.filter(|key| !key.contains("/log/") && !key.contains("/manifest/"))
                .collect::<BTreeSet<String>>()
        }
    };
    let stored_before = segment_files(&cluster).await;
    paused.armed.store(true, Ordering::SeqCst);
    let merge = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    held(&mut seen).await;
    model.step(&writer, &[Op::Up("notes", 1, 7)]).await;
    model.step(&writer, &[Op::Del("notes", 2)]).await;
    model.step(&writer, &[Op::Drop("tmp")]).await;
    if crash == Crash::AfterMerged {
        merge.abort();
        let _ = merge.await;
        // The upload held when the merge was dropped settles, and only
        // then is what the merge put deleted: never before it lands.
        paused.armed.store(false, Ordering::SeqCst);
        paused.release.notify_one();
        settle().await;
        assert!(!writer.is_closed());
        let left = cluster.blobs.list(&Layout(&cluster.key).prefix()).await;
        let left: Vec<_> = left.unwrap().into_iter().map(|file| file.key).collect();
        assert!(!left.iter().any(|key| key.contains("-merged")), "{left:?}");
        model.step(&writer, &[Op::Up("notes", 130, 1)]).await;
        same_everywhere(&cluster, &writer, &model, "dropped after merged").await;
        writer.merge().await.unwrap().expect("a merge");
        same_everywhere(&cluster, &writer, &model, "merged after all").await;
        return;
    }
    paused.release.notify_one();
    held(&mut seen).await;
    model.step(&writer, &[Op::Del("notes", 1)]).await;
    model.step(&writer, &[Op::Up("notes", 2, 11)]).await;
    model.step(&writer, &[Op::Del("notes", 6)]).await;
    let written = cluster.committed().await;
    if crash == Crash::AfterFlushed {
        merge.abort();
        let _ = merge.await;
        // As above: the held upload settles before what was put is deleted.
        paused.armed.store(false, Ordering::SeqCst);
        paused.release.notify_one();
        settle().await;
        assert!(!writer.is_closed());
        let left = cluster.blobs.list(&Layout(&cluster.key).prefix()).await;
        let left: BTreeSet<String> = left.unwrap().into_iter().map(|file| file.key).collect();
        assert!(!left.iter().any(|key| key.contains("-merged")), "{left:?}");
        same_everywhere(&cluster, &writer, &model, "dropped after flushed").await;
        writer.merge().await.unwrap().expect("a merge");
        same_everywhere(&cluster, &writer, &model, "merged after all").await;
        return;
    }
    if crash == Crash::ClosedBeforeTurn {
        unreliable
            .refuse_commits
            .store(usize::MAX, Ordering::SeqCst);
        assert!(writer.commit(vec![upsert("notes", 140, 1)]).await.is_err());
        unreliable.refuse_commits.store(0, Ordering::SeqCst);
        assert!(writer.is_closed());
        assert!(segment_files(&cluster).await.len() > stored_before.len());
        paused.armed.store(false, Ordering::SeqCst);
        paused.release.notify_one();
        let outcome = tokio::time::timeout(Duration::from_secs(600), merge)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(outcome, Err(StorageError::Closed)), "{outcome:?}");
        settle().await;
        assert_eq!(
            segment_files(&cluster).await,
            stored_before,
            "what the merge uploaded and never tried to commit is deleted"
        );
        let opened = cluster.open().await;
        assert_eq!(opened.partition().read(view), model.rows);
        return;
    }
    match crash {
        Crash::AfterHead => unreliable
            .refuse_commits
            .store(usize::MAX, Ordering::SeqCst),
        Crash::AfterCommit => unreliable.hang_after_commit.store(true, Ordering::SeqCst),
        Crash::AnswerLostOnce => unreliable.lose_responses.store(1, Ordering::SeqCst),
        _ => unreachable!(),
    }
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_one();
    let outcome = tokio::time::timeout(Duration::from_secs(600), merge)
        .await
        .unwrap()
        .unwrap();
    unreliable.refuse_commits.store(0, Ordering::SeqCst);
    unreliable.hang_after_commit.store(false, Ordering::SeqCst);
    settle().await;
    let committed = cluster.committed().await;
    match crash {
        Crash::AfterHead => {
            assert!(outcome.is_err() && writer.is_closed());
            assert_eq!(committed, written, "nothing was committed");
        }
        Crash::AfterCommit => {
            assert!(outcome.is_err() && writer.is_closed(), "{outcome:?}");
            assert!(committed.log_position > written.log_position, "it landed");
        }
        Crash::AnswerLostOnce => {
            assert_eq!(outcome.unwrap(), Some(committed.log_position));
            assert!(!writer.is_closed());
            assert_eq!(writer.partition().read(view), model.rows);
        }
        _ => unreachable!(),
    }
    // What is committed is whole in the store, and is what was acknowledged.
    let files = Partition::files(cluster.key.clone(), cluster.blobs.clone(), &committed);
    for key in files.await.unwrap() {
        cluster.blobs.get(&key).await.expect("a committed file");
    }
    let opened = cluster.open().await;
    assert_eq!(opened.partition().read(view), model.rows, "{crash:?}");
    let merged = opened.partition().read(|state| state.segments().count());
    match crash {
        Crash::AfterHead => assert_eq!(merged, segments_before),
        _ => assert!(merged < segments_before, "{merged} of {segments_before}"),
    }
    drop(writer);
    let successor = cluster.writer().await.merging_with(merging);
    same_everywhere(&cluster, &successor, &model, "taken over").await;
    model.step(&successor, &[Op::Up("notes", 131, 1)]).await;
    successor.flush().await.unwrap();
    while successor.merge().await.unwrap().is_some() {}
    successor.clean(Duration::ZERO).await.unwrap();
    same_everywhere(&cluster, &successor, &model, "cleaned").await;
}

#[tokio::test(start_paused = true)]
async fn a_merge_that_ends_at_any_point_leaves_what_was_committed() {
    for later_run in [false, true] {
        for crash in [
            Crash::AfterMerged,
            Crash::AfterFlushed,
            Crash::ClosedBeforeTurn,
            Crash::AfterHead,
            Crash::AfterCommit,
            Crash::AnswerLostOnce,
        ] {
            crashed(crash, later_run).await;
        }
    }
}

/// Blobs whose uploads of segments take a little real time, so that
/// commits fall in the windows of a flush or a merge.
#[derive(Debug)]
struct Lagging(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for Lagging {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if key.contains("segment/") {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        self.0.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.0.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.0.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.0.delete(key).await
    }
}

/// Numbers that look random, the same for the same seed.
struct Dice(u64);

impl Dice {
    fn roll(&mut self, sides: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) % sides
    }
}

/// Five hundred writes checked against a model while two tasks flush,
/// merge and clean as they please. Returns how many merges and flushes
/// were made.
async fn history(seed: u64) -> (usize, usize) {
    let mut cluster = Cluster::new().await;
    cluster.blobs = Arc::new(Lagging(cluster.blobs.clone()));
    let merging = Merging {
        rows: 40,
        fanout: 2,
        stale: 0.4,
        ..Merging::default()
    };
    let writer = Arc::new(cluster.writer().await.merging_with(merging));
    let mut model = Model::default();
    model.step(&writer, &[Op::Create("notes")]).await;
    let follower = cluster.open().await;
    let stop = Arc::new(AtomicBool::new(false));
    let merges = Arc::new(AtomicUsize::new(0));
    let flushes = Arc::new(AtomicUsize::new(0));
    let maintaining: Vec<_> = (0..2u64)
        .map(|worker| {
            let (writer, stop) = (writer.clone(), stop.clone());
            let (merges, flushes) = (merges.clone(), flushes.clone());
            tokio::spawn(async move {
                let mut dice = Dice(seed * 31 + worker + 1);
                while !stop.load(Ordering::SeqCst) {
                    match dice.roll(5) {
                        0 | 1 => {
                            let before = writer.partition().read(State::position);
                            if writer.flush().await.unwrap() > before {
                                flushes.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        2 | 3 => {
                            if writer.merge().await.unwrap().is_some() {
                                merges.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                        _ => {
                            writer.clean(Duration::ZERO).await.unwrap();
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(dice.roll(3))).await;
                }
            })
        })
        .collect();
    let mut dice = Dice(seed + 1);
    for step in 0..500 {
        let object = dice.roll(48) as u8;
        let n = dice.roll(1_000) as i64;
        let op = match dice.roll(20) {
            0..=8 => Op::Up("notes", object, n),
            9..=13 => Op::Del("notes", object),
            14..=16 => Op::Inc(object, n),
            17 => Op::Up("tmp", object, n),
            18 => Op::Create("tmp"),
            _ => Op::Drop("tmp"),
        };
        if !model.applies(&op) {
            continue;
        }
        model.step(&writer, &[op]).await;
        if dice.roll(4) == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        if step % 20 == 0 {
            assert_eq!(writer.partition().read(view), model.rows, "step {step}");
            // A copy may ask for files a clean of no grace has taken; one
            // of what is committed now, read again, finds them.
            let mut tries = 0;
            loop {
                let committed = cluster.committed().await;
                let caught_up = follower.catch_up(&committed).await;
                let opened = cluster.open_on(cluster.blobs.clone()).await;
                match (caught_up, opened) {
                    (Ok(_), Ok(opened)) => {
                        let at = follower.partition().read(State::position);
                        if at == writer.partition().read(State::position) {
                            assert_eq!(follower.partition().read(view), model.rows);
                        }
                        if opened.partition().read(State::position)
                            == writer.partition().read(State::position)
                        {
                            assert_eq!(opened.partition().read(view), model.rows);
                        }
                        break;
                    }
                    (a, b) => {
                        tries += 1;
                        assert!(tries < 50, "step {step}: {:?} {:?}", a.err(), b.err());
                    }
                }
            }
        }
    }
    stop.store(true, Ordering::SeqCst);
    for task in maintaining {
        task.await.unwrap();
    }
    model.requests = model
        .requests
        .split_off(model.requests.len().saturating_sub(40));
    same_everywhere(&cluster, &writer, &model, "at the end").await;
    let committed = cluster.committed().await;
    follower.catch_up(&committed).await.unwrap();
    assert_eq!(follower.partition().read(view), model.rows);
    drop(writer);
    let successor = cluster.writer().await.merging_with(merging);
    same_everywhere(&cluster, &successor, &model, "taken over").await;
    successor.flush().await.unwrap();
    while successor.merge().await.unwrap().is_some() {}
    successor.clean(Duration::ZERO).await.unwrap();
    same_everywhere(&cluster, &successor, &model, "cleaned").await;
    (
        merges.load(Ordering::SeqCst),
        flushes.load(Ordering::SeqCst),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn histories_of_writes_under_flushes_merges_and_cleaning_match_a_model() {
    let (mut merges, mut flushes) = (0, 0);
    for seed in 0..12 {
        let (merged, flushed) = history(seed).await;
        merges += merged;
        flushes += flushed;
    }
    assert!(
        merges > 50 && flushes > 50,
        "{merges} merges, {flushes} flushes"
    );
}
