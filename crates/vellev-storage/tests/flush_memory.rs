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

/// Rows flushed or merged: two counts, so that what is held whatever the
/// rows (coding's sample of rotated vectors and its rotation, say) falls
/// out of the difference.
const FEWER: usize = 10_000;
const MORE: usize = 20_000;
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

/// Commit `rows` rows of `dimensions`, numbered from `first`.
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
                properties: json!({ "title": format!("row {}", first + n), "n": first + n })
                    .as_object()
                    .unwrap()
                    .clone(),
                vector: Some(random.vector(dimensions)),
                only_if_version: None,
            })
            .collect();
        writer.commit(writes).await.unwrap();
    }
}

/// What a flush of `rows` rows of `dimensions` holds at its peak beyond
/// what was held before it.
async fn flush_peak(rows: usize, dimensions: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    writer.commit(vec![collection(dimensions)]).await.unwrap();
    committed(&writer, &mut Random(7), 0, rows, dimensions).await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.flush().await.unwrap();
    COUNTING.peak() - before
}

/// What a merge of four flushed segments of `rows` rows in all, of
/// `dimensions`, holds at its peak beyond what was held before it.
async fn merge_peak(rows: usize, dimensions: usize) -> usize {
    merge_measured(rows, dimensions).await.0
}

/// What a merge as [`merge_peak`] makes holds at its peak, and what it
/// reckoned it would once its rows were gathered, and as it was planned.
async fn merge_measured(rows: usize, dimensions: usize) -> (usize, usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    let builds = vellev_storage::MergeBuilds::new(1);
    let writer = writer
        .merging_with(vellev_storage::Merging {
            fanout: 4,
            ..vellev_storage::Merging::default()
        })
        .merging_among(builds.clone());
    writer.commit(vec![collection(dimensions)]).await.unwrap();
    let mut random = Random(11);
    for segment in 0..4 {
        committed(&writer, &mut random, segment * rows, rows / 4, dimensions).await;
        writer.flush().await.unwrap();
    }
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.merge().await.unwrap().expect("the segments merge");
    (
        COUNTING.peak() - before,
        builds.last_reckoned(),
        builds.last_planned(),
    )
}

/// Hold what a further row's vector costs at the peak against `copies`
/// copies of it, with a tenth of one for what else grows with the width:
/// the slope between `FEWER` and `MORE` rows of what the wide peak holds
/// over the narrow, in which whatever is held whatever the rows cancels.
async fn held_to<F, P>(what: &str, peak: F, copies: usize)
where
    F: Fn(usize, usize) -> P,
    P: std::future::Future<Output = usize>,
{
    let theirs = |wide: usize, narrow: usize| wide as f64 - narrow as f64;
    let fewer = theirs(peak(FEWER, WIDE).await, peak(FEWER, NARROW).await);
    let more = theirs(peak(MORE, WIDE).await, peak(MORE, NARROW).await);
    let per_row = (more - fewer) / (MORE - FEWER) as f64;
    let vector = ((WIDE - NARROW) * 4) as f64;
    eprintln!(
        "{what} held {:.2} times each further row's vector at its peak ({per_row:.0} bytes a row)",
        per_row / vector
    );
    assert!(
        per_row <= vector * (copies as f64 + 0.1),
        "{what} held {per_row:.0} bytes for each further row's vector at its peak, {:.2} times its {vector}",
        per_row / vector
    );
}

#[tokio::test(flavor = "current_thread")]
async fn a_flush_holds_its_vectors_once() {
    let _alone = ALONE.lock().await;
    // The column the segment keeps, which is its encoded frame too.
    held_to("a flush", flush_peak, 1).await;
}

#[tokio::test(flavor = "current_thread")]
async fn a_merge_holds_its_vectors_once() {
    let _alone = ALONE.lock().await;
    // The merged segment's column, which is its encoded frame too.
    held_to("a merge", merge_peak, 1).await;
}

/// A merge reckons what it holds at its peak to within 30% of what it is
/// measured to hold once its rows are gathered, and to within half as it
/// is planned, from what its run's segments hold (merge note, §7.6): the
/// node counts what a stopped merge was reckoned to hold as draining,
/// until it has ended.
#[tokio::test(flavor = "current_thread")]
async fn a_merge_reckons_what_it_holds_at_its_peak_within_thirty_percent() {
    let _alone = ALONE.lock().await;
    let mut measured = Vec::new();
    for (rows, dimensions) in [(FEWER, WIDE), (MORE, WIDE), (FEWER, NARROW), (MORE, NARROW)] {
        let (peak, reckoned, planned) = merge_measured(rows, dimensions).await;
        let off = |by: usize| (by as f64 - peak as f64) / peak as f64;
        eprintln!(
            "a merge of {rows} rows of {dimensions} held {peak} bytes at its peak, \
             reckoned {reckoned} ({:+.0}%), planned {planned} ({:+.0}%)",
            off(reckoned) * 100.0,
            off(planned) * 100.0
        );
        measured.push((rows, dimensions, peak, reckoned, planned));
    }
    for (rows, dimensions, peak, reckoned, planned) in measured {
        let off = |by: usize| (by as f64 - peak as f64) / peak as f64;
        assert!(
            off(reckoned).abs() <= 0.3,
            "a merge of {rows} rows of {dimensions} reckoned {reckoned} bytes, \
             held {peak} at its peak: {:+.0}%",
            off(reckoned) * 100.0
        );
        assert!(
            off(planned).abs() <= 0.5,
            "a merge of {rows} rows of {dimensions} was planned at {planned} bytes, \
             held {peak} at its peak: {:+.0}%",
            off(planned) * 100.0
        );
    }
}
