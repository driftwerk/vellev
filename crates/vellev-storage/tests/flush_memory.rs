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
const DIMENSIONS: usize = 768;

fn wide(collection: &str) -> Mutation {
    let mut schema = schema(collection);
    schema.embedding.dimensions = DIMENSIONS as u32;
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

/// What a flush may hold for each row besides vectors: the row itself
/// without its vector, its codes and index entries, and its part of the
/// segment's encoded file, all of which the segment keeps or writes. A
/// row of a few properties takes well under this.
const PER_ROW: usize = 2 * 1024;

#[tokio::test(flavor = "current_thread")]
async fn a_flush_holds_its_vectors_no_more_than_twice_over() {
    let _alone = ALONE.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    writer.commit(vec![wide("docs")]).await.unwrap();
    let mut random = Random(7);
    for batch in 0..ROWS / 500 {
        let rows = (0..500)
            .map(|n| Mutation::Upsert {
                collection: name("docs"),
                id: ObjectId::generate(),
                // Little besides the vector, so that what is measured is
                // what becomes of the vectors.
                properties: json!({ "n": batch * 500 + n }).as_object().unwrap().clone(),
                vector: Some(random.vector(DIMENSIONS)),
                only_if_version: None,
            })
            .collect();
        writer.commit(rows).await.unwrap();
    }
    let vectors = ROWS * DIMENSIONS * 4;
    // The column the segment keeps, and its encoded frame.
    let allowed = 2 * vectors + ROWS * PER_ROW;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.flush().await.unwrap();
    let held = COUNTING.peak() - before;
    eprintln!(
        "a flush of {ROWS} rows held {:.2} times its vectors at its peak",
        held as f64 / vectors as f64
    );
    assert!(
        held <= allowed,
        "a flush held {held} bytes at its peak, {:.2} times its vectors' {vectors}; it may hold {allowed}",
        held as f64 / vectors as f64
    );
}

/// A merge of the segments of `segments` flushes of `ROWS / segments` rows.
async fn merged_peak(segments: usize) -> (usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    let writer = writer.merging_with(vellev_storage::Merging {
        fanout: segments,
        ..vellev_storage::Merging::default()
    });
    writer.commit(vec![wide("docs")]).await.unwrap();
    let mut random = Random(11);
    for segment in 0..segments {
        for batch in 0..ROWS / segments / 500 {
            let rows = (0..500)
                .map(|n| Mutation::Upsert {
                    collection: name("docs"),
                    id: ObjectId::generate(),
                    properties: json!({ "n": segment * ROWS + batch * 500 + n })
                        .as_object()
                        .unwrap()
                        .clone(),
                    vector: Some(random.vector(DIMENSIONS)),
                    only_if_version: None,
                })
                .collect();
            writer.commit(rows).await.unwrap();
        }
        writer.flush().await.unwrap();
    }
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.merge().await.unwrap().expect("the segments merge");
    (COUNTING.peak() - before, ROWS * DIMENSIONS * 4)
}

#[tokio::test(flavor = "current_thread")]
async fn a_merge_holds_its_vectors_no_more_than_twice_over() {
    let _alone = ALONE.lock().await;
    let (held, vectors) = merged_peak(4).await;
    // The merged segment's column and its encoded frame.
    let allowed = 2 * vectors + ROWS * PER_ROW;
    eprintln!(
        "a merge of {ROWS} rows held {:.2} times its vectors at its peak",
        held as f64 / vectors as f64
    );
    assert!(
        held <= allowed,
        "a merge held {held} bytes at its peak, {:.2} times its vectors' {vectors}; it may hold {allowed}",
        held as f64 / vectors as f64
    );
}
