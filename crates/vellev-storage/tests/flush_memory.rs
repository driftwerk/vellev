//! What a flush or a merge holds in memory while it runs (vellev memory
//! note, fixes 2 and 4).
//!
//! A flush of a memtable of N rows of d-dimensional vectors writes them to
//! a segment. It used to copy every memtable row, vector and properties,
//! into a snapshot, gather the vectors again into a column grown by
//! doubling, and encode them a third time into the column's frame: four to
//! five copies of the vectors at its peak, which a node with one large
//! tenant could not afford. Here a counting allocator measures the peak.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::AtomicUsize;

use common::*;

/// The system allocator, counting what is held now and at most since the
/// peak was last reset.
struct Counting {
    now: AtomicUsize,
    peak: AtomicUsize,
}

impl Counting {
    const fn new() -> Self {
        Self {
            now: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn now(&self) -> usize {
        self.now.load(Ordering::SeqCst)
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }

    fn reset_peak(&self) {
        self.peak.store(self.now(), Ordering::SeqCst);
    }

    fn added(&self, size: usize) {
        let now = self.now.fetch_add(size, Ordering::SeqCst) + size;
        self.peak.fetch_max(now, Ordering::SeqCst);
    }
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

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
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

#[global_allocator]
static COUNTING: Counting = Counting::new();

/// One measurement at a time: the count is the whole process's.
static ALONE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const ROWS: usize = 20_000;
/// The width measured, and one so narrow that its vectors cost next to
/// nothing: the difference between their peaks is what the vectors cost,
/// whatever each row costs besides.
const WIDE: usize = 768;
const NARROW: usize = 8;

fn collection(dimensions: usize) -> Mutation {
    let mut schema = schema("docs");
    schema.embedding.dimensions = dimensions as u32;
    Mutation::CreateCollection(Box::new(schema))
}

/// A writer on a store that is a local directory, as a node's cache is:
/// nothing the store keeps is held in memory.
async fn on_disk(dir: &std::path::Path) -> (Cluster, Writer) {
    let mut cluster = Cluster::new().await;
    let local = object_store::local::LocalFileSystem::new_with_prefix(dir).unwrap();
    cluster.blobs = Arc::new(ObjectBlobs::on_disk(Arc::new(local), dir));
    let writer = cluster.writer().await;
    (cluster, writer)
}

/// Commit `rows` rows of `dimensions`, numbered from `first`, with little
/// besides their vectors.
async fn committed(
    writer: &Writer,
    random: &mut Random,
    first: usize,
    rows: usize,
    dimensions: usize,
) {
    for batch in (0..rows).step_by(500) {
        let writes = (batch..(batch + 500).min(rows))
            .map(|n| Mutation::Upsert {
                collection: name("docs"),
                id: ObjectId::generate(),
                properties: json!({ "n": first + n }).as_object().unwrap().clone(),
                vector: Some(random.vector(dimensions)),
                only_if_version: None,
            })
            .collect();
        writer.commit(writes).await.unwrap();
    }
}

/// What a flush of `ROWS` rows of `dimensions` holds at its peak beyond
/// what was held before it.
async fn flush_peak(dimensions: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    writer.commit(vec![collection(dimensions)]).await.unwrap();
    committed(&writer, &mut Random(7), 0, ROWS, dimensions).await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.flush().await.unwrap();
    COUNTING.peak() - before
}

/// What a merge of four flushed segments of `ROWS` rows in all, of
/// `dimensions`, holds at its peak beyond what was held before it.
async fn merge_peak(dimensions: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    let writer = writer.merging_with(vellev_storage::Merging {
        fanout: 4,
        ..vellev_storage::Merging::default()
    });
    writer.commit(vec![collection(dimensions)]).await.unwrap();
    let mut random = Random(11);
    for segment in 0..4 {
        committed(&writer, &mut random, segment * ROWS, ROWS / 4, dimensions).await;
        writer.flush().await.unwrap();
    }
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.merge().await.unwrap().expect("the segments merge");
    COUNTING.peak() - before
}

/// What coding a layer's vectors holds whatever the rows: the sample of
/// rotated vectors its centre is taken from, and the rotation itself, at
/// their widest about 17 MiB at 768 dimensions.
const CODING: usize = 32 * 1024 * 1024;

/// Hold the vectors' share of `wide` against `copies` copies of the
/// vectors, with a tenth of one for what else grows with the width, and
/// what coding holds whatever the rows.
fn held_to(what: &str, wide: usize, narrow: usize, copies: usize) {
    let vectors = ROWS * (WIDE - NARROW) * 4;
    let theirs = wide.saturating_sub(narrow).saturating_sub(CODING);
    eprintln!(
        "{what} of {ROWS} rows held {:.2} times its vectors' worth for them ({wide} bytes at {WIDE} dimensions, {narrow} at {NARROW})",
        theirs as f64 / vectors as f64
    );
    assert!(
        theirs * 10 <= vectors * (10 * copies + 1),
        "{what} held {theirs} bytes for its vectors at its peak, {:.2} times their {vectors}",
        theirs as f64 / vectors as f64
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_flush_holds_its_vectors_once() {
    let _alone = ALONE.lock().await;
    let (wide, narrow) = (flush_peak(WIDE).await, flush_peak(NARROW).await);
    // The column the segment keeps, which is its encoded frame too.
    held_to("a flush", wide, narrow, 1);
}

#[tokio::test(flavor = "current_thread")]
async fn a_merge_holds_its_vectors_once() {
    let _alone = ALONE.lock().await;
    let (wide, narrow) = (merge_peak(WIDE).await, merge_peak(NARROW).await);
    // The merged segment's column, which is its encoded frame too.
    held_to("a merge", wide, narrow, 1);
}
