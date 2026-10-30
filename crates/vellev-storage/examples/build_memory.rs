//! What a flush, a merge and an open hold in memory, tokenizing in chunks
//! and every row at once: the node memory note's tables (§1.1 and §2.1).
//!
//!   cargo run --release -p vellev-storage --example build_memory
//!
//! Environment: ROWS (200000), DIMS (768), TEXT (passage, or wide: 200 words
//! over a large vocabulary), CLUSTERED (0; 1 clusters what is merged).
//!
//! A counting allocator measures each peak, above what was held before it.
//! A thread reads the count every millisecond, and each crest it saw is
//! printed, in MB, in the order they came: for a merge of passages, the rows
//! gathered, each chunk's tokens beside the index so far, the vectors, and
//! encoding. Stages are told from one another by that order alone, so that
//! nothing is marked in the code measured. Printed to stdout, for a job's
//! log to keep.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use object_store::local::LocalFileSystem;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{tokenize_at_once, Follower, Merging, Mutation, Writer, TOKENIZED_AT_ONCE};

/// The system allocator, counting what is held now and at most since the
/// peak was last reset.
struct Counting {
    now: AtomicUsize,
    peak: AtomicUsize,
}

// SAFETY: every call is the system allocator's, counted on the way.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            self.added(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        self.now.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if !moved.is_null() {
            // Both are held while the old is copied into the new.
            self.added(new_size);
            self.now.fetch_sub(layout.size(), Ordering::SeqCst);
        }
        moved
    }
}

impl Counting {
    fn added(&self, size: usize) {
        let now = self.now.fetch_add(size, Ordering::SeqCst) + size;
        self.peak.fetch_max(now, Ordering::SeqCst);
    }
}

#[global_allocator]
static COUNTING: Counting = Counting {
    now: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};

const MB: f64 = 1_000_000.0;

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Random(u64);

impl Random {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn ranked(&mut self) -> usize {
        50_000f32.powf(self.unit()) as usize
    }

    /// About 120 words in about 690 bytes, as `flush_memory`'s are.
    fn passage(&mut self) -> String {
        const STOPWORDS: [&str; 8] = ["the", "of", "and", "in", "to", "a", "that", "it"];
        let word = |at: usize| match at % 3 {
            0 => STOPWORDS[(self.unit() * 8.0) as usize].to_string(),
            _ => format!("pas{}", self.ranked()),
        };
        (0..120).map(word).collect::<Vec<_>>().join(" ")
    }

    fn wide(&mut self) -> String {
        let word = |_| format!("w{}", self.ranked());
        (0..200).map(word).collect::<Vec<_>>().join(" ")
    }
}

/// What to write.
#[derive(Clone, Copy)]
struct Shape {
    rows: usize,
    dimensions: usize,
    wide: bool,
    clustered: bool,
}

/// A writer and a store on local disk, as a node's cache is.
struct Store {
    _dir: tempfile::TempDir,
    key: PartitionKey,
    blobs: Arc<dyn Blobs>,
    coordinator: Arc<MemoryCoordinator>,
    writer: Writer,
}

async fn store(shape: Shape) -> Store {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::on_disk(Arc::new(local), dir.path()));
    let coordinator = Arc::new(MemoryCoordinator::default());
    let key = PartitionKey {
        tenant: TenantId::new("t").unwrap(),
        partition: vellev_core::Partition::DEFAULT,
    };
    coordinator.register(&key).await.unwrap();
    let hold = Duration::from_secs(3600);
    let holder = NodeId::generate("n");
    let Acquired::Lease { lease, committed } =
        coordinator.acquire(&key, &holder, hold).await.unwrap()
    else {
        panic!("lease")
    };
    let writer = Writer::open(blobs.clone(), coordinator.clone(), lease, &committed)
        .await
        .unwrap()
        .merging_with(Merging {
            clustered_above: match shape.clustered {
                true => 1_000,
                false => Merging::default().clustered_above,
            },
            ..Merging::default()
        });
    let properties = vec![
        Property::new("title", DataType::Text).searchable(),
        Property::new("n", DataType::Int),
    ];
    let collection = Collection {
        name: CollectionName::new("docs").unwrap(),
        properties,
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "m".into(),
            dimensions: shape.dimensions as u32,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    };
    let create = Mutation::CreateCollection(Box::new(collection));
    writer.commit(vec![create]).await.unwrap();
    Store {
        _dir: dir,
        key,
        blobs,
        coordinator,
        writer,
    }
}

async fn written(writer: &Writer, random: &mut Random, shape: Shape, first: usize, rows: usize) {
    for batch in (first..first + rows).step_by(500) {
        let commit = (batch..(batch + 500).min(first + rows))
            .map(|n| Mutation::Upsert {
                collection: CollectionName::new("docs").unwrap(),
                id: ObjectId(uuid::Uuid::from_u128(n as u128)),
                properties: json!({
                    "title": if shape.wide { random.wide() } else { random.passage() },
                    "n": n,
                })
                .as_object()
                .unwrap()
                .clone(),
                vector: Some((0..shape.dimensions).map(|_| random.unit()).collect()),
                only_if_version: None,
            })
            .collect();
        writer.commit(commit).await.unwrap();
    }
}

/// `work`, with what it held: its peak above what was held as it began,
/// and each crest it came to.
async fn measured<T>(what: &str, work: impl std::future::Future<Output = T>) -> T {
    static SAMPLING: AtomicBool = AtomicBool::new(false);
    SAMPLING.store(true, Ordering::SeqCst);
    // Made first, so that what it holds is held before as well as after.
    let mut samples = Vec::with_capacity(4_000_000);
    let before = COUNTING.now.load(Ordering::SeqCst);
    COUNTING.peak.store(before, Ordering::SeqCst);
    let sampler = std::thread::spawn(move || {
        while SAMPLING.load(Ordering::SeqCst) && samples.len() < samples.capacity() {
            samples.push(COUNTING.now.load(Ordering::SeqCst));
            std::thread::sleep(Duration::from_millis(1));
        }
        samples
    });
    let done = work.await;
    let peak = COUNTING.peak.load(Ordering::SeqCst);
    SAMPLING.store(false, Ordering::SeqCst);
    let samples = sampler.join().unwrap();
    println!(
        "{what}: peak {:.1} MB, crests {:?}",
        peak.saturating_sub(before) as f64 / MB,
        crests(&samples, before, peak)
    );
    done
}

/// The crests of `samples`, in MB above `before`: each high that falls by at
/// least a fiftieth of what the peak came to, before it rises by as much.
fn crests(samples: &[usize], before: usize, peak: usize) -> Vec<f64> {
    let fall = peak.saturating_sub(before) / 50;
    let in_mb = |held: usize| (held as f64 / MB * 10.0).round() / 10.0;
    let mut crests = Vec::new();
    let (mut high, mut low, mut falling) = (0, 0, false);
    for sample in samples.iter().map(|sample| sample.saturating_sub(before)) {
        if falling {
            low = low.min(sample);
            if sample > low + fall {
                (high, falling) = (sample, false);
            }
        } else if sample > high {
            high = sample;
        } else if sample + fall < high {
            crests.push(in_mb(high));
            (low, falling) = (sample, true);
        }
    }
    if !falling {
        crests.push(in_mb(high));
    }
    crests
}

/// A flush of all the rows, and an open of the one segment it wrote.
async fn flushed_and_opened(shape: Shape, arm: &str, at_once: usize) {
    tokenize_at_once(at_once);
    let store = store(shape).await;
    let mut random = Random(11);
    written(&store.writer, &mut random, shape, 0, shape.rows).await;
    let flushed = measured(&format!("flush {arm}"), store.writer.flush()).await;
    flushed.unwrap();
    drop(store.writer);
    let committed = store
        .coordinator
        .committed(&store.key)
        .await
        .unwrap()
        .unwrap();
    let open = Follower::open(store.key.clone(), store.blobs.clone(), &committed);
    let opened = measured(&format!("open {arm}"), open).await.unwrap();
    println!(
        "open {arm}: {} rows' tokens held at once",
        opened.partition().replayed().tokenized_at_once
    );
}

/// A merge of four flushed segments of a quarter of the rows each.
async fn merged(shape: Shape, arm: &str, at_once: usize) {
    tokenize_at_once(at_once);
    let store = store(shape).await;
    let mut random = Random(11);
    for segment in 0..4 {
        let first = segment * shape.rows / 4;
        written(&store.writer, &mut random, shape, first, shape.rows / 4).await;
        store.writer.flush().await.unwrap();
    }
    let merge = store.writer.merge();
    let done = measured(&format!("merge {arm}"), merge).await;
    done.unwrap().expect("the segments merge");
}

#[tokio::main]
async fn main() {
    let shape = Shape {
        rows: setting("ROWS", 200_000),
        dimensions: setting("DIMS", 768),
        wide: std::env::var("TEXT").is_ok_and(|text| text == "wide"),
        clustered: setting("CLUSTERED", 0) == 1,
    };
    for (arm, at_once) in [
        ("all at once", usize::MAX),
        ("in chunks", TOKENIZED_AT_ONCE),
    ] {
        flushed_and_opened(shape, arm, at_once).await;
        merged(shape, arm, at_once).await;
    }
}
