//! How long a merge's gather keeps a flush and a commit of its tenant
//! waiting: each is timed from just after a merge's first chunk of rows is
//! copied, with the merge stopped once both are done, and with no merge
//! under way, to tell what it costs to wait from what each takes anyway.
//! The building lock and the state's are all a gather can keep them behind.
//!
//! Then a merge is let commit, with reads and writes going on beside it, to
//! show what is left: the commit's walk of the run's rows under the read
//! lock, and of the live map under the write lock, which no gather's change
//! moves (node memory note, §2.9). The log's `adopting` is how long the
//! write lock was held; the longest read and write are what that and the
//! walk under the read lock cost them. They go on for AFTER_MERGE seconds
//! after the merge returns, so that they see what it replaced being freed,
//! which it no longer waits for. A plain flush follows, for its own
//! `adopting`, as what is left walks the tenant's live rows as well.
//!
//! For a small merge in a large tenant, LARGE_SEGMENTS of LARGE_ROWS rows
//! each go first, too large to merge: the hold then follows the tenant, not
//! the merge. WORDS keeps those rows' text small enough to hold.
//!
//!   cargo run --release -p vellev-storage --example gather_hold
//!
//! Environment: ROWS (100000, a segment), SEGMENTS (4), DIMS (768),
//! RUNS (5), AFTER_MERGE (5), WORDS (80), LARGE_SEGMENTS (0),
//! LARGE_ROWS (1000000).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use serde_json::json;
use tracing::subscriber::Interest;
use tracing_subscriber::filter::DynFilterFn;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{MergeBuilds, Merging, Mutation, State, Writer};

const PER_COMMIT: usize = 500;

/// Whether what a checkpoint held the write lock for is logged: only for
/// the merge let commit.
static LOGGING: AtomicBool = AtomicBool::new(false);

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn notes(dimensions: u32) -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: CollectionName::new("notes").unwrap(),
        properties: vec![Property::new("text", DataType::Text).searchable()],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "stand-in".into(),
            dimensions,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

/// A note of `words` words of text: about 700 bytes of it at 80, as the
/// bench's passages are.
fn note(n: u64, dims: usize, words: usize, random: &mut Random) -> Mutation {
    let text: Vec<String> = (0..words)
        .map(|_| format!("w{:06}", random.next() % 200_000))
        .collect();
    Mutation::Upsert {
        collection: CollectionName::new("notes").unwrap(),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1)),
        properties: json!({ "text": text.join(" ") })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some((0..dims).map(|_| random.unit()).collect()),
        only_if_version: None,
    }
}

/// A flush and a commit, side by side, each timed.
async fn flush_and_commit(
    writer: &Writer,
    n: u64,
    dims: usize,
    words: usize,
) -> (Duration, Duration) {
    let asked = Instant::now();
    let flushing = async {
        writer.flush().await.unwrap();
        asked.elapsed()
    };
    let committing = async {
        let mut random = Random(n | 1);
        let written = note(n, dims, words, &mut random);
        writer.commit(vec![written]).await.unwrap();
        asked.elapsed()
    };
    tokio::join!(flushing, committing)
}

#[tokio::main]
async fn main() {
    let adopted =
        DynFilterFn::new(|_, _| LOGGING.load(Ordering::SeqCst)).with_callsite_filter(|meta| {
            match meta.fields().field("adopting") {
                Some(_) => Interest::sometimes(),
                None => Interest::never(),
            }
        });
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(adopted))
        .init();
    let (rows, segments, dims, runs) = (
        setting("ROWS", 100_000),
        setting("SEGMENTS", 4),
        setting("DIMS", 768),
        setting("RUNS", 5),
    );
    let (after_merge, words) = (setting("AFTER_MERGE", 5), setting("WORDS", 80));
    let (large_segments, large_rows) = (
        setting("LARGE_SEGMENTS", 0),
        setting("LARGE_ROWS", 1_000_000),
    );
    let large = large_segments * large_rows;
    let coordinator = Arc::new(MemoryCoordinator::default());
    let key = PartitionKey {
        tenant: TenantId::new("t").unwrap(),
        partition: vellev_core::Partition::DEFAULT,
    };
    coordinator.register(&key).await.unwrap();
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let hour = Duration::from_secs(3_600);
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("n"), hour)
        .await
        .unwrap()
    else {
        panic!("a fresh partition is free to take")
    };
    // Not clustered: the merge let commit is timed at its commit, not its
    // tuning.
    let merging = Merging {
        fanout: segments,
        clustered_above: usize::MAX,
        ..Merging::default()
    };
    let builds = MergeBuilds::new(1);
    let writer = Writer::open(blobs, coordinator.clone(), lease, &committed)
        .await
        .unwrap()
        .merging_with(merging)
        .merging_among(builds.clone());
    let writer = Arc::new(writer);
    writer.commit(vec![notes(dims as u32)]).await.unwrap();
    let mut random = Random(0x9e37_79b9_7f4a_7c15);
    let began = Instant::now();
    let sizes = std::iter::repeat_n(large_rows, large_segments);
    let mut first = 0;
    for (segment, size) in sizes.chain(std::iter::repeat_n(rows, segments)).enumerate() {
        for from in (0..size).step_by(PER_COMMIT) {
            let batch = (from..(from + PER_COMMIT).min(size))
                .map(|n| note((first + n) as u64, dims, words, &mut random))
                .collect();
            writer.commit(batch).await.unwrap();
        }
        first += size;
        writer.flush().await.unwrap();
        println!("segment {segment} flushed at {:?}", began.elapsed());
    }

    let mut n = (large + segments * rows) as u64;
    let mut timed = |writer: &Arc<Writer>| {
        n += 1;
        let writer = writer.clone();
        let at = n;
        async move { flush_and_commit(&writer, at, dims, words).await }
    };
    for run in 0..runs {
        let (flush_alone, commit_alone) = timed(&writer).await;
        // What the last gather came to, until this one is seen to begin.
        let last = builds.last_gathered();
        let merge = tokio::spawn({
            let writer = writer.clone();
            async move { writer.merge().await }
        });
        while builds.last_gathered() == last {
            tokio::time::sleep(Duration::from_micros(50)).await;
        }
        while builds.last_gathered() == 0 {
            tokio::time::sleep(Duration::from_micros(50)).await;
        }
        let (flush, commit) = timed(&writer).await;
        writer.stop_merges();
        let merged = merge.await.unwrap().unwrap();
        assert!(merged.is_none(), "a stopped merge commits nothing");
        println!(
            "run {run}: with a gather under way, flush {flush:?}, commit {commit:?}; \
             with none, flush {flush_alone:?}, commit {commit_alone:?}"
        );
    }

    // The commit, with a write at a time and reads going on beside it.
    let gate = writer.hold_merges_before_commit();
    let merge = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    gate.reached.notified().await;
    println!("the merge is built, at {:?}", began.elapsed());
    let going = Arc::new(AtomicBool::new(true));
    let reading = std::thread::spawn({
        let (going, partition) = (going.clone(), writer.partition().clone());
        move || {
            let mut longest = Duration::ZERO;
            while going.load(Ordering::SeqCst) {
                let asked = Instant::now();
                partition.read(State::rows);
                longest = longest.max(asked.elapsed());
                std::thread::sleep(Duration::from_micros(100));
            }
            longest
        }
    });
    let writing = tokio::spawn({
        let (going, writer) = (going.clone(), writer.clone());
        let mut n = n + 1;
        async move {
            let mut longest = Duration::ZERO;
            let mut random = Random(n | 1);
            while going.load(Ordering::SeqCst) {
                n += 1;
                let written = note(n, dims, words, &mut random);
                let asked = Instant::now();
                writer.commit(vec![written]).await.unwrap();
                longest = longest.max(asked.elapsed());
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
            longest
        }
    });
    LOGGING.store(true, Ordering::SeqCst);
    let released = Instant::now();
    gate.release.notify_one();
    let committed = merge.await.unwrap().unwrap();
    let commit_took = released.elapsed();
    LOGGING.store(false, Ordering::SeqCst);
    assert!(committed.is_some(), "the merge commits");
    tokio::time::sleep(Duration::from_secs(after_merge as u64)).await;
    going.store(false, Ordering::SeqCst);
    let (longest_read, longest_write) = (reading.join().unwrap(), writing.await.unwrap());
    println!(
        "the commit took {commit_took:?}; beside it and for {after_merge} s after, the longest \
         read took {longest_read:?}, the longest write {longest_write:?}"
    );
    LOGGING.store(true, Ordering::SeqCst);
    writer.flush().await.unwrap();
    LOGGING.store(false, Ordering::SeqCst);
}
