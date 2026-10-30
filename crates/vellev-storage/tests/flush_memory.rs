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
use std::time::Duration;

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
/// Rows merged where each holds long text: fewer, as each holds far more.
const TEXTS: usize = 4_000;
/// Rows merged just past where their vectors are given an index, so that
/// a clustered index's training is much of what the merge holds.
const TRAINED: usize = 6_000;
/// Rows merged whose tokens are held a chunk at a time, not all at once:
/// well past a chunk, and few enough to be a measure of seconds.
const CHUNKED: usize = 3 * vellev_storage::TOKENIZED_AT_ONCE;
/// The width measured, and one so narrow that its vectors cost next to
/// nothing: the difference between their peaks is what the vectors cost,
/// whatever each row costs besides.
const WIDE: usize = 768;
const NARROW: usize = 8;
/// How long a merge's rows may take to write and flush, and how long it
/// may take to merge them, in seconds, on a runner far slower than asked.
const WRITTEN_WITHIN: u64 = 600;
const MERGED_WITHIN: u64 = 600;

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

/// The text each row's title holds.
#[derive(Debug, Clone, Copy)]
enum Text {
    /// "row" and its number.
    Short,
    /// 200 words, each found in this row alone: as code, paths and ids are.
    Unique,
    /// 200 words drawn from 50,000, the commoner far more often.
    Wide,
    /// 200 words drawn from 20.
    Few,
    /// 100 words of 40 characters, each found in this row alone.
    LongUnique,
    /// About 120 words in about 690 bytes, about 60 of them different and
    /// no stopwords, the rest of them stopwords: as the bench's passages
    /// are.
    Passage,
}

impl Text {
    /// Row `n`'s title.
    fn of(self, n: usize, random: &mut Random) -> String {
        let words = |word: &mut dyn FnMut(usize) -> String| {
            (0..200).map(word).collect::<Vec<_>>().join(" ")
        };
        match self {
            Self::Short => format!("row {n}"),
            Self::Unique => words(&mut |at| format!("u{n}x{at}")),
            Self::Wide => words(&mut |_| format!("w{}", Self::ranked(random))),
            Self::Few => words(&mut |_| format!("f{}", (Self::drawn(random) * 20.0) as usize)),
            Self::LongUnique => (0..100)
                .map(|at| format!("{:x>40}", format!("l{n}y{at}")))
                .collect::<Vec<_>>()
                .join(" "),
            Self::Passage => {
                const STOPWORDS: [&str; 8] = ["the", "of", "and", "in", "to", "a", "that", "it"];
                (0..120)
                    .map(|at| match at % 3 {
                        0 => STOPWORDS[(Self::drawn(random) * 8.0) as usize].to_string(),
                        _ => format!("pas{}", Self::ranked(random)),
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    /// A draw from 0 up to 1.
    fn drawn(random: &mut Random) -> f32 {
        random.unit() + 0.5
    }

    /// A rank of 50,000: that to the power of a uniform draw, each rank
    /// about as often as its inverse, as words are.
    fn ranked(random: &mut Random) -> usize {
        50_000f32.powf(Self::drawn(random)) as usize
    }
}

/// Commit `rows` rows of `dimensions`, numbered from `first`, titled as
/// `text` says. Where `rewriting`, every fourth row rewrites the one
/// `rewriting` rows before it, which dies.
async fn committed(
    writer: &Writer,
    random: &mut Random,
    first: usize,
    rows: usize,
    dimensions: usize,
    text: Text,
    rewriting: Option<usize>,
) {
    let id = |n: usize| match rewriting {
        Some(back) if n % 4 == 3 && n >= back => {
            ObjectId(uuid::Uuid::from_u128((n - back) as u128))
        }
        _ => ObjectId(uuid::Uuid::from_u128(n as u128)),
    };
    for batch in (0..rows).step_by(500) {
        let writes = (batch..(batch + 500).min(rows))
            .map(|n| Mutation::Upsert {
                collection: name("docs"),
                id: id(first + n),
                properties: json!({ "title": text.of(first + n, random), "n": first + n })
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
    let mut random = Random(7);
    committed(&writer, &mut random, 0, rows, dimensions, Text::Short, None).await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    writer.flush().await.unwrap();
    COUNTING.peak() - before
}

/// What a merge of four flushed segments of `rows` rows in all, of
/// `dimensions`, holds at its peak beyond what was held before it.
async fn merge_peak(rows: usize, dimensions: usize) -> usize {
    let at_once = vellev_storage::TOKENIZED_AT_ONCE;
    merge_measured(rows, dimensions, Text::Short, false, false, at_once)
        .await
        .peak
}

/// What a merge is measured to hold, and reckoned to.
#[derive(Debug, Clone, Copy)]
struct Measured {
    /// What it held at its peak.
    peak: usize,
    /// What it reckoned it would once its rows were gathered.
    reckoned: usize,
    /// What it reckoned it would as it was planned.
    planned: usize,
    /// How many rows' tokens it held at once, at the most.
    tokenized_at_once: usize,
}

/// What `work` comes to within `limit`: a merge that never ends fails the
/// rule rather than hanging it.
async fn within<T>(limit: u64, what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(limit), work)
        .await
        .unwrap_or_else(|_| panic!("{what} within {limit} s"))
}

/// What a merge as [`merge_peak`] makes, its rows titled as `text` says,
/// their vectors `clustered` or not, a quarter of each segment's rows
/// `rewriting` the segment's before, and `at_once` rows' text tokenized at
/// a time, is [`Measured`] to hold.
async fn merge_measured(
    rows: usize,
    dimensions: usize,
    text: Text,
    clustered: bool,
    rewriting: bool,
    at_once: usize,
) -> Measured {
    let dir = tempfile::tempdir().unwrap();
    let (_cluster, writer) = on_disk(dir.path()).await;
    let builds = vellev_storage::MergeBuilds::new(1);
    let writer = writer
        .merging_with(vellev_storage::Merging {
            fanout: 4,
            clustered_above: match clustered {
                true => 1_000,
                false => vellev_storage::Merging::default().clustered_above,
            },
            ..vellev_storage::Merging::default()
        })
        .merging_among(builds.clone());
    writer.tokenize_merges_at_once(at_once);
    writer.commit(vec![collection(dimensions)]).await.unwrap();
    let mut random = Random(11);
    let written = async {
        for segment in 0..4 {
            let back = rewriting.then_some(rows);
            let first = segment * rows;
            committed(
                &writer,
                &mut random,
                first,
                rows / 4,
                dimensions,
                text,
                back,
            )
            .await;
            writer.flush().await.unwrap();
        }
    };
    within(WRITTEN_WITHIN, "the rows written and flushed", written).await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    let merged = within(MERGED_WITHIN, "the merge ended", writer.merge()).await;
    merged.unwrap().expect("the segments merge");
    Measured {
        peak: COUNTING.peak() - before,
        reckoned: builds.last_reckoned(),
        planned: builds.last_planned(),
        tokenized_at_once: builds.last_tokenized_at_once(),
    }
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
/// is planned, from what its run's segments hold (merge note, §7.6 and
/// §7.7): the node counts what a stopped merge was reckoned to hold as
/// draining, until it has ended. Over short text and long, text of words
/// each in one row, over a large vocabulary and a small one, and vectors
/// clustered or not.
///
/// A dozen merges and more, too slow built for debugging: run optimised,
/// in CI's Test job, as
/// `cargo test --release -p vellev-storage --test flush_memory -- --ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_merge_reckons_what_it_holds_at_its_peak_within_thirty_percent() {
    let _alone = ALONE.lock().await;
    let shapes = [
        (FEWER, WIDE, Text::Short, false, false),
        (MORE, WIDE, Text::Short, false, false),
        (FEWER, NARROW, Text::Short, false, false),
        (MORE, NARROW, Text::Short, false, false),
        (TEXTS, NARROW, Text::Unique, false, false),
        (TEXTS, WIDE, Text::Unique, false, false),
        (TEXTS, NARROW, Text::LongUnique, false, false),
        (TEXTS, NARROW, Text::Wide, false, false),
        (TEXTS, NARROW, Text::Few, false, false),
        // A quarter of each segment's rows dead by the merge.
        (FEWER, NARROW, Text::Short, false, true),
        // Few enough rows that training is much of the peak.
        (TRAINED, WIDE, Text::Short, true, false),
        (FEWER, WIDE, Text::Short, true, false),
        (MORE, WIDE, Text::Short, true, false),
        (FEWER, NARROW, Text::Short, true, false),
        // Past a chunk, so that most of the tokens are never held together.
        (CHUNKED, NARROW, Text::Short, false, false),
        (CHUNKED, NARROW, Text::Few, false, false),
        (CHUNKED, NARROW, Text::Wide, false, false),
        (CHUNKED, NARROW, Text::Passage, false, false),
        (CHUNKED, WIDE, Text::Passage, false, false),
        (CHUNKED, NARROW, Text::Unique, false, false),
    ];
    let mut measured = Vec::new();
    for (rows, dimensions, text, clustered, rewriting) in shapes {
        let at_once = vellev_storage::TOKENIZED_AT_ONCE;
        let Measured {
            peak,
            reckoned,
            planned,
            ..
        } = merge_measured(rows, dimensions, text, clustered, rewriting, at_once).await;
        let off = |by: usize| (by as f64 - peak as f64) / peak as f64;
        let shape = format!(
            "{rows} rows of {dimensions}, {text:?} text, clustered {clustered}, \
             rewriting {rewriting}"
        );
        eprintln!(
            "a merge of {shape} held {peak} bytes at its peak, \
             reckoned {reckoned} ({:+.0}%), planned {planned} ({:+.0}%)",
            off(reckoned) * 100.0,
            off(planned) * 100.0
        );
        measured.push((shape, peak, reckoned, planned));
    }
    for (shape, peak, reckoned, planned) in measured {
        let off = |by: usize| (by as f64 - peak as f64) / peak as f64;
        assert!(
            off(reckoned).abs() <= 0.3,
            "a merge of {shape} reckoned {reckoned} bytes, held {peak} at its peak: {:+.0}%",
            off(reckoned) * 100.0
        );
        assert!(
            off(planned).abs() <= 0.5,
            "a merge of {shape} was planned at {planned} bytes, held {peak} at its peak: {:+.0}%",
            off(planned) * 100.0
        );
    }
}

/// The rows a merge's whole build is measured at, of passages.
const PASSAGES: [usize; 3] = [20_000, 100_000, 200_000];

/// M1-R1. A merge's build holds at most `TOKENIZED_AT_ONCE` rows' tokens at
/// once, however many it merges, and over the whole of it, stage by stage,
/// holds no more at its peak for that than tokenizing every row at once
/// held (node memory note, §2.1). What it saves is what the build's later
/// stages leave of the tokens' share, printed here for the PR to say.
///
/// As the reckoning's rule: run optimised, in CI's Test job, as
/// `cargo test --release -p vellev-storage --test flush_memory -- --ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_merge_holds_a_chunk_of_tokens_at_once_and_its_whole_build_no_more_for_it() {
    let _alone = ALONE.lock().await;
    let at_once = vellev_storage::TOKENIZED_AT_ONCE;
    let mut measured = Vec::new();
    for rows in PASSAGES {
        let all = merge_measured(rows, WIDE, Text::Passage, false, false, usize::MAX).await;
        let chunked = merge_measured(rows, WIDE, Text::Passage, false, false, at_once).await;
        eprintln!(
            "a merge of {rows} passages held {} bytes at its peak all at once ({} rows' tokens \
             at a time), {} in chunks ({} rows' tokens at a time): {:+.1}%",
            all.peak,
            all.tokenized_at_once,
            chunked.peak,
            chunked.tokenized_at_once,
            (chunked.peak as f64 - all.peak as f64) / all.peak as f64 * 100.0,
        );
        measured.push((rows, all, chunked));
    }
    for (rows, all, chunked) in measured {
        assert!(
            chunked.tokenized_at_once <= at_once,
            "a merge of {rows} rows held {} rows' tokens at once",
            chunked.tokenized_at_once
        );
        assert!(
            chunked.peak <= all.peak,
            "a merge of {rows} rows held {} bytes in chunks, {} tokenizing every row at once",
            chunked.peak,
            all.peak
        );
    }
}
