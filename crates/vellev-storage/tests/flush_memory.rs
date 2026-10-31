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
use serde_json::{Map, Value as JsonValue};

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
/// Rows merged of which a chunk is a small share: where every row's tokens
/// reckoned at once would be far more than the merge ever held.
const CHUNKED_MORE: usize = 6 * vellev_storage::TOKENIZED_AT_ONCE;
/// The width measured, and one so narrow that its vectors cost next to
/// nothing: the difference between their peaks is what the vectors cost,
/// whatever each row costs besides.
const WIDE: usize = 768;
const NARROW: usize = 8;
/// How long a merge's rows may take to write and flush, and how long it
/// may take to merge them, in seconds, on a runner far slower than asked.
const WRITTEN_WITHIN: u64 = 600;
const MERGED_WITHIN: u64 = 600;

/// The most properties a row here holds besides those of `schema`.
const WIDEST_SCHEMA: usize = 30;

/// The name of the `of`th collection written to.
fn docs(of: usize) -> CollectionName {
    name(&format!("docs{of}"))
}

fn collection(of: usize, dimensions: usize) -> Mutation {
    let mut schema = schema(&format!("docs{of}"));
    schema.embedding.dimensions = dimensions as u32;
    schema.properties.push(Property {
        filterable: false,
        ..Property::new("body", DataType::Text)
    });
    schema.properties.push(Property {
        filterable: false,
        ..Property::new("notes", DataType::Text).searchable()
    });
    for property in 0..WIDEST_SCHEMA {
        let name = format!("p{property}");
        schema.properties.push(Property::new(&name, DataType::Int));
    }
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

/// As [`on_disk`], with a cache in `cache` over the store, as a node has:
/// so that a merge's vectors are gathered there, and nothing the store or
/// its cache keeps is held in memory.
async fn cached_on_disk(dir: &std::path::Path, cache: &std::path::Path) -> (Cluster, Writer) {
    let mut cluster = Cluster::new().await;
    let local = object_store::local::LocalFileSystem::new_with_prefix(dir).unwrap();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::on_disk(Arc::new(local), dir));
    cluster.blobs = Arc::new(CachedBlobs::new(store, cache));
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
    /// 200 words drawn evenly from 50,000, each of a row's nearly always
    /// different: as many tokens as text of its length can hold.
    Even,
    /// 100 words of 40 characters, each found in this row alone.
    LongUnique,
    /// About 120 words in about 690 bytes, about 60 of them different and
    /// no stopwords, the rest of them stopwords: as the bench's passages
    /// are.
    Passage,
    /// The rows from `from` up to `to` of the run, however its segments
    /// fall, hold `words` words drawn evenly from 50,000, as a bulk import of
    /// long documents does, or if `short`, of four letters drawn from 65,536,
    /// whose tokens outweigh their text the more, and if `unfiltered`, in a
    /// property searched on and not filtered on, so that tokens are most of
    /// all such a row comes to; of the rest, one in two holds "row" and its
    /// number and the other no text at all.
    Block {
        from: usize,
        to: usize,
        words: usize,
        short: bool,
        unfiltered: bool,
    },
    /// "row" and its number, and a body of 800 words, each found in this row
    /// alone, in a property neither searched nor filtered on.
    Bodied,
    /// "row" and its number, and this many properties of a number found in
    /// this row alone, each filtered on.
    Properties(usize),
    /// "row" and its number, and a text array of this many tags, each found
    /// in this row alone and filtered on.
    Tags(usize),
}

impl Text {
    /// Row `n`'s title.
    fn of(self, n: usize, random: &mut Random) -> String {
        let words = |word: &mut dyn FnMut(usize) -> String| {
            (0..200).map(word).collect::<Vec<_>>().join(" ")
        };
        match self {
            Self::Short | Self::Bodied | Self::Properties(_) | Self::Tags(_) => {
                format!("row {n}")
            }
            Self::Unique => words(&mut |at| format!("u{n}x{at}")),
            Self::Wide => words(&mut |_| format!("w{}", Self::ranked(random))),
            Self::Few => words(&mut |_| format!("f{}", (Self::drawn(random) * 20.0) as usize)),
            Self::Even => words(&mut |_| format!("e{}", (Self::drawn(random) * 50_000.0) as usize)),
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
            Self::Block {
                from,
                to,
                words,
                short,
                ..
            } if (from..to).contains(&n) => (0..words)
                .map(|_| match short {
                    true => format!("{:04x}", (Self::drawn(random) * 65_536.0) as usize),
                    false => format!("e{}", (Self::drawn(random) * 50_000.0) as usize),
                })
                .collect::<Vec<_>>()
                .join(" "),
            Self::Block { .. } if n.is_multiple_of(2) => String::new(),
            Self::Block { .. } => format!("row {n}"),
        }
    }

    /// The properties of the row at `place` of the run, numbered `n`.
    fn properties(self, place: usize, n: usize, random: &mut Random) -> Map<String, JsonValue> {
        let mut properties = json!({ "n": n }).as_object().unwrap().clone();
        let title = self.of(place, random);
        let in_notes = matches!(self, Self::Block { from, to, unfiltered: true, .. }
            if (from..to).contains(&place));
        if !title.is_empty() {
            let name = if in_notes { "notes" } else { "title" };
            properties.insert(name.to_string(), json!(title));
        }
        match self {
            Self::Bodied => {
                let words: Vec<String> = (0..800).map(|at| format!("b{n}x{at}")).collect();
                properties.insert("body".to_string(), json!(words.join(" ")));
            }
            Self::Properties(many) => {
                for property in 0..many {
                    properties.insert(format!("p{property}"), json!(n * WIDEST_SCHEMA + property));
                }
            }
            Self::Tags(many) => {
                let tags: Vec<String> = (0..many).map(|at| format!("t{n}x{at}")).collect();
                properties.insert("tags".to_string(), json!(tags));
            }
            _ => {}
        }
        properties
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

/// Rows to commit.
#[derive(Debug, Clone, Copy)]
struct Writes {
    /// How many rows, of how many dimensions.
    rows: usize,
    dimensions: usize,
    /// The first row's number, which its id is made from.
    first: usize,
    /// The first row's place among all the rows the run holds.
    place: usize,
    text: Text,
    /// If given, every fourth row rewrites the one this many rows before
    /// it, which dies.
    rewriting: Option<usize>,
    /// How many collections the rows go to, each taking every so many rows.
    collections: usize,
}

impl Writes {
    /// The id of the row numbered `n`.
    fn id(&self, n: usize) -> ObjectId {
        let n = match self.rewriting {
            Some(back) if n % 4 == 3 && n >= back => n - back,
            _ => n,
        };
        ObjectId(uuid::Uuid::from_u128(n as u128))
    }
}

/// Commit `writes`.
async fn committed(writer: &Writer, random: &mut Random, writes: Writes) {
    for batch in (0..writes.rows).step_by(500) {
        let commit = (batch..(batch + 500).min(writes.rows))
            .map(|at| {
                let n = writes.first + at;
                Mutation::Upsert {
                    collection: docs(n % writes.collections),
                    id: writes.id(n),
                    properties: writes.text.properties(writes.place + at, n, random),
                    vector: Some(random.vector(writes.dimensions)),
                    only_if_version: None,
                }
            })
            .collect();
        writer.commit(commit).await.unwrap();
    }
}

/// What a flush of `rows` rows of `dimensions` holds at its peak beyond
/// what was held before it.
async fn flush_peak(rows: usize, dimensions: usize) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let opened = within(WRITTEN_WITHIN, "the writer opened", on_disk(dir.path()));
    let (_cluster, writer) = opened.await;
    let mut random = Random(7);
    let writes = Writes {
        rows,
        dimensions,
        first: 0,
        place: 0,
        text: Text::Short,
        rewriting: None,
        collections: 1,
    };
    let written = async {
        writer
            .commit(vec![collection(0, dimensions)])
            .await
            .unwrap();
        committed(&writer, &mut random, writes).await;
    };
    within(WRITTEN_WITHIN, "the rows written", written).await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    let flushed = within(MERGED_WITHIN, "the flush ended", writer.flush()).await;
    flushed.unwrap();
    COUNTING.peak() - before
}

/// A merge to measure: of how many rows, what they hold, and how they fall
/// into the segments of its run.
#[derive(Debug, Clone, Copy)]
struct Shape {
    rows: usize,
    dimensions: usize,
    text: Text,
    clustered: bool,
    /// A quarter of each segment's rows rewrite the segment's before.
    rewriting: bool,
    /// Each segment's share of the rows.
    shares: &'static [usize],
    /// The rows, by their place in the run, deleted before it is merged:
    /// from this one up to that one.
    dying: (usize, usize),
    /// How many collections the rows go to.
    collections: usize,
    /// A collection whose rows are spared if a row dying is of another.
    sparing: Option<usize>,
}

impl Shape {
    /// Of `rows` rows, in four equal segments.
    const fn of(rows: usize, dimensions: usize, text: Text) -> Self {
        Self {
            rows,
            dimensions,
            text,
            clustered: false,
            rewriting: false,
            shares: &[1, 1, 1, 1],
            dying: (0, 0),
            collections: 1,
            sparing: None,
        }
    }

    const fn clustered(self) -> Self {
        Self {
            clustered: true,
            ..self
        }
    }

    const fn rewriting(self) -> Self {
        Self {
            rewriting: true,
            ..self
        }
    }

    const fn shared(self, shares: &'static [usize]) -> Self {
        Self { shares, ..self }
    }

    const fn dying(self, from: usize, to: usize) -> Self {
        Self {
            dying: (from, to),
            ..self
        }
    }

    /// Every collection's rows dying, but for the first's.
    const fn sparing_one(self) -> Self {
        Self {
            dying: (0, self.rows),
            sparing: Some(0),
            ..self
        }
    }

    const fn over(self, collections: usize) -> Self {
        Self {
            collections,
            ..self
        }
    }
}

/// What a merge of four flushed segments of `rows` rows in all, of
/// `dimensions`, holds at its peak beyond what was held before it.
async fn merge_peak(rows: usize, dimensions: usize) -> usize {
    let at_once = vellev_storage::TOKENIZED_AT_ONCE;
    merge_measured(
        Shape::of(rows, dimensions, Text::Short),
        at_once,
        Gathered::InMemory,
    )
    .await
    .peak
}

/// Where a merge's vectors are gathered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gathered {
    /// In memory, as on a store that keeps no copies of its own.
    InMemory,
    /// In a file of the store's cache, read in place.
    OnDisk,
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
    /// How many bytes of vectors it gathered on local disk.
    spilled: usize,
}

/// What `work` comes to within `limit`: a merge that never ends fails the
/// rule rather than hanging it.
async fn within<T>(limit: u64, what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(limit), work)
        .await
        .unwrap_or_else(|_| panic!("{what} within {limit} s"))
}

/// Every build tokenizes `rows` rows at a time while this is held.
struct Tokenizing;

impl Tokenizing {
    fn at_once(rows: usize) -> Self {
        vellev_storage::tokenize_at_once(rows);
        Self
    }
}

impl Drop for Tokenizing {
    fn drop(&mut self) {
        vellev_storage::tokenize_at_once(vellev_storage::TOKENIZED_AT_ONCE);
    }
}

/// Wait until the count stands still: what the merges before this one let
/// go is freed off the async threads, and nothing waits for it. Measured
/// while that falls, a merge reads as holding less than it does, by whatever
/// fell before its peak: 3 collections' merge, the same to the byte alone,
/// read 13 MB of its 47 MB short in the rule's order, and looked like a
/// spread as threads fell.
async fn count_settled() {
    within(WRITTEN_WITHIN, "the count settled", async {
        let (mut last, mut still) = (COUNTING.now(), 0);
        while still < COUNT_STILL_FOR {
            tokio::time::sleep(Duration::from_millis(20)).await;
            let now = COUNTING.now();
            still = match now.abs_diff(last) < COUNT_STILL_WITHIN {
                true => still + 1,
                false => 0,
            };
            last = now;
        }
    })
    .await;
}

/// How many readings 20 ms apart show the count standing still, and how
/// little it may move between them: a task's few allocations go on.
const COUNT_STILL_FOR: usize = 5;
const COUNT_STILL_WITHIN: usize = 16 * 1024;

/// What a merge of `shape`, `at_once` rows' text tokenized at a time, is
/// [`Measured`] to hold.
async fn merge_measured(shape: Shape, at_once: usize, gathered: Gathered) -> Measured {
    let _tokenizing = Tokenizing::at_once(at_once);
    let (dir, cache) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let (_cluster, writer) = within(WRITTEN_WITHIN, "the writer opened", async {
        match gathered {
            Gathered::InMemory => on_disk(dir.path()).await,
            Gathered::OnDisk => cached_on_disk(dir.path(), cache.path()).await,
        }
    })
    .await;
    let builds = vellev_storage::MergeBuilds::new(1);
    let writer = writer
        .merging_with(vellev_storage::Merging {
            fanout: shape.shares.len(),
            clustered_above: match shape.clustered {
                true => 1_000,
                false => vellev_storage::Merging::default().clustered_above,
            },
            ..vellev_storage::Merging::default()
        })
        .merging_among(builds.clone());
    let mut random = Random(11);
    let written = async {
        let collections = (0..shape.collections).map(|of| collection(of, shape.dimensions));
        writer.commit(collections.collect()).await.unwrap();
        let (mut place, all) = (0, shape.shares.iter().sum::<usize>());
        let mut dying = Vec::new();
        for (segment, share) in shape.shares.iter().enumerate() {
            let rows = shape.rows * share / all;
            let writes = Writes {
                rows,
                dimensions: shape.dimensions,
                first: segment * shape.rows,
                place,
                text: shape.text,
                rewriting: shape.rewriting.then_some(shape.rows),
                collections: shape.collections,
            };
            committed(&writer, &mut random, writes).await;
            writer.flush().await.unwrap();
            let (from, to) = shape.dying;
            let these = from.max(place)..to.min(place + rows);
            dying.extend(these.map(|at| (at - place + writes.first, writes)));
            place += rows;
        }
        dying.retain(|(n, writes)| shape.sparing != Some(n % writes.collections));
        for batch in dying.chunks(500) {
            let deletes = batch.iter().map(|(n, writes)| Mutation::Delete {
                collection: docs(n % writes.collections),
                id: writes.id(*n),
                only_if_version: None,
            });
            writer.commit(deletes.collect()).await.unwrap();
        }
    };
    within(WRITTEN_WITHIN, "the rows written and flushed", written).await;
    count_settled().await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    let merged = within(MERGED_WITHIN, "the merge ended", writer.merge()).await;
    merged.unwrap().expect("the segments merge");
    Measured {
        peak: COUNTING.peak() - before,
        reckoned: builds.last_reckoned(),
        planned: builds.last_planned(),
        tokenized_at_once: writer.last_tokenized_at_once(),
        spilled: writer.last_spilled(),
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

/// How much a reckoning may be off what was held: within 30% once the rows
/// are gathered, and within half as it is planned.
fn within_bounds(shape: &str, peak: usize, reckoned: usize, planned: usize) {
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

/// The merges the reckoning is held to.
const RECKONED: [Shape; 28] = [
    Shape::of(FEWER, WIDE, Text::Short),
    Shape::of(MORE, WIDE, Text::Short),
    Shape::of(FEWER, NARROW, Text::Short),
    Shape::of(MORE, NARROW, Text::Short),
    Shape::of(TEXTS, NARROW, Text::Unique),
    Shape::of(TEXTS, WIDE, Text::Unique),
    Shape::of(TEXTS, NARROW, Text::LongUnique),
    Shape::of(TEXTS, NARROW, Text::Wide),
    Shape::of(TEXTS, NARROW, Text::Few),
    // A quarter of each segment's rows dead by the merge.
    Shape::of(FEWER, NARROW, Text::Short).rewriting(),
    // Few enough rows that training is much of the peak.
    Shape::of(TRAINED, WIDE, Text::Short).clustered(),
    Shape::of(FEWER, WIDE, Text::Short).clustered(),
    Shape::of(MORE, WIDE, Text::Short).clustered(),
    Shape::of(FEWER, NARROW, Text::Short).clustered(),
    // Past a chunk, so that most of the tokens are never held together.
    Shape::of(CHUNKED, NARROW, Text::Short),
    Shape::of(CHUNKED, NARROW, Text::Few),
    Shape::of(CHUNKED_MORE, NARROW, Text::Wide),
    Shape::of(CHUNKED, NARROW, Text::Even),
    Shape::of(CHUNKED, NARROW, Text::Passage),
    Shape::of(CHUNKED, WIDE, Text::Passage),
    Shape::of(CHUNKED, NARROW, Text::Unique),
    // A chunk's rows of long documents together, not spread over the run:
    // in its last segments, across two, and a small share of many rows.
    Shape::of(
        CHUNKED,
        NARROW,
        Text::Block {
            from: CHUNKED - AT_ONCE,
            to: CHUNKED,
            words: 200,
            short: false,
            unfiltered: false,
        },
    ),
    Shape::of(
        CHUNKED,
        NARROW,
        Text::Block {
            from: 16_000,
            to: 16_000 + AT_ONCE,
            words: 200,
            short: false,
            unfiltered: false,
        },
    ),
    Shape::of(
        CHUNKED_MORE,
        NARROW,
        Text::Block {
            from: CHUNKED_MORE - AT_ONCE,
            to: CHUNKED_MORE,
            words: 800,
            short: false,
            unfiltered: false,
        },
    ),
    Shape::of(
        CHUNKED_MORE,
        NARROW,
        Text::Block {
            from: CHUNKED_MORE - AT_ONCE,
            to: CHUNKED_MORE,
            words: 800,
            short: true,
            unfiltered: false,
        },
    ),
    // Tokens past all else a row comes to, in a property no filter holds,
    // and enough of them that encoding them does not stand in for them: a
    // chunk's are what the peak is, reckoned spread evenly over the run or
    // not at all, past the bound.
    Shape::of(
        CHUNKED_MORE,
        NARROW,
        Text::Block {
            from: CHUNKED_MORE - AT_ONCE,
            to: CHUNKED_MORE,
            words: 1_600,
            short: true,
            unfiltered: true,
        },
    ),
    // Rows mostly of what nothing indexes, and encoded all the same.
    Shape::of(TEXTS, NARROW, Text::Bodied),
    Shape::of(MORE, WIDE, Text::Bodied),
];

/// Long rows, a chunk of them, at the head of what follows.
const LONG_FIRST: Text = Text::Block {
    from: 0,
    to: AT_ONCE,
    words: 800,
    short: false,
    unfiltered: false,
};

/// Runs of other than four equal segments, whose indexes add up otherwise.
const RECKONED_RUNS: [Shape; 4] = [
    Shape::of(3_000, NARROW, Text::Unique).shared(&[1, 1, 1]),
    Shape::of(TEXTS, NARROW, Text::Unique).shared(&[1, 1]),
    Shape::of(MORE, NARROW, Text::Wide).shared(&[8, 1, 1, 1, 1]),
    Shape::of(CHUNKED, NARROW, Text::Passage).shared(&[6, 1, 1]),
];

/// Rows that die together, not evenly, and a stale segment's rewrite.
const RECKONED_DEATHS: [Shape; 7] = [
    // All the deaths one collection's but for the first: what is left is
    // coded and trained as a whole collection, not a share of each of eight.
    Shape::of(40_000, WIDE, Text::Short)
        .over(8)
        .shared(&[1])
        .sparing_one(),
    // Long rows stay as short rows die, whose run is a third long at plan.
    Shape::of(CHUNKED, NARROW, LONG_FIRST)
        .shared(&[1])
        .dying(AT_ONCE, CHUNKED),
    Shape::of(2 * AT_ONCE, NARROW, LONG_FIRST)
        .shared(&[1])
        .dying(AT_ONCE, 2 * AT_ONCE),
    // Long rows die as short rows stay.
    Shape::of(CHUNKED, NARROW, LONG_FIRST)
        .shared(&[1, 1])
        .dying(0, AT_ONCE),
    Shape::of(2 * AT_ONCE, NARROW, LONG_FIRST)
        .shared(&[1])
        .dying(0, AT_ONCE),
    // Half and nine tenths of a segment of its own words gone: its map is
    // what its live rows hold, not what the segment did.
    Shape::of(8_192, NARROW, Text::Unique)
        .shared(&[1])
        .dying(0, 4_096),
    Shape::of(8_192, NARROW, Text::Unique)
        .shared(&[1])
        .dying(0, 7_373),
];

/// Collections, of which a build trains one at a time, and rows of many
/// properties and many values filtered on, which hold what their maps do.
const RECKONED_WIDE: [Shape; 6] = [
    Shape::of(15_000, WIDE, Text::Short).over(3),
    Shape::of(40_000, WIDE, Text::Short).over(8),
    Shape::of(FEWER, NARROW, Text::Properties(WIDEST_SCHEMA)),
    Shape::of(FEWER, NARROW, Text::Properties(12)),
    Shape::of(2_000, NARROW, Text::Tags(100)),
    Shape::of(FEWER, NARROW, Text::Tags(12)),
];

/// A merge reckons what it holds at its peak to within 30% of what it is
/// measured to hold once its rows are gathered, and to within half as it
/// is planned, from what its run's segments hold (merge note, §7.6 and
/// §7.7): the node counts what a stopped merge was reckoned to hold as
/// draining, until it has ended. Over short text and long, passages, text
/// of words each in one row, over a large vocabulary, drawn by rank or
/// evenly, and a small one, long documents all together, rows mostly of
/// what nothing indexes, vectors clustered or not, runs of other than four
/// equal segments, rows dying together, several collections, many properties
/// and values filtered on, up to and past what a build tokenizes at once.
///
/// What the bounds cover is these shapes: the note's §2.1 says what lies
/// outside them.
///
/// A score of merges and more, too slow built for debugging: run optimised,
/// in CI's Test job, as
/// `cargo test --release -p vellev-storage --test flush_memory -- --ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_merge_reckons_what_it_holds_at_its_peak_within_thirty_percent() {
    let _alone = ALONE.lock().await;
    let mut measured = Vec::new();
    let shapes = RECKONED
        .iter()
        .chain(&RECKONED_RUNS)
        .chain(&RECKONED_DEATHS)
        .chain(&RECKONED_WIDE);
    for shape in shapes {
        let merge = merge_measured(*shape, AT_ONCE, Gathered::InMemory).await;
        let off = |by: usize| (by as f64 - merge.peak as f64) / merge.peak as f64 * 100.0;
        eprintln!(
            "a merge of {shape:?} held {} bytes at its peak, reckoned {} ({:+.0}%), planned {} \
             ({:+.0}%)",
            merge.peak,
            merge.reckoned,
            off(merge.reckoned),
            merge.planned,
            off(merge.planned)
        );
        measured.push((shape, merge));
    }
    for (shape, merge) in measured {
        within_bounds(
            &format!("{shape:?}"),
            merge.peak,
            merge.reckoned,
            merge.planned,
        );
    }
}

/// M8-R5. The same bounds, over the same merges, of a merge whose vectors
/// are gathered in the store's cache: reckoned, planned and gathered, at
/// what it holds on the heap, which leaves its vectors out.
///
/// As the rule above: run optimised, in CI's Test job.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_merge_gathered_on_disk_reckons_what_it_holds_within_thirty_percent() {
    let _alone = ALONE.lock().await;
    let mut measured = Vec::new();
    let shapes = RECKONED
        .iter()
        .chain(&RECKONED_RUNS)
        .chain(&RECKONED_DEATHS)
        .chain(&RECKONED_WIDE);
    for shape in shapes {
        let merge = merge_measured(*shape, AT_ONCE, Gathered::OnDisk).await;
        let off = |by: usize| (by as f64 - merge.peak as f64) / merge.peak as f64 * 100.0;
        eprintln!(
            "a merge gathered on disk of {shape:?} held {} bytes at its peak, reckoned {} \
             ({:+.0}%), planned {} ({:+.0}%), {} bytes spilled",
            merge.peak,
            merge.reckoned,
            off(merge.reckoned),
            merge.planned,
            off(merge.planned),
            merge.spilled,
        );
        measured.push((shape, merge));
    }
    for (shape, merge) in measured {
        assert!(merge.spilled > 0, "{shape:?} gathered in memory");
        within_bounds(
            &format!("{shape:?}"),
            merge.peak,
            merge.reckoned,
            merge.planned,
        );
    }
}

/// The merges whose vectors' heap is weighed: passages at three sizes, and
/// a clustered one, whose training reads what is gathered.
const SPILLED_BUILDS: [Shape; 4] = [
    Shape::of(20_000, WIDE, Text::Passage),
    Shape::of(100_000, WIDE, Text::Passage),
    Shape::of(200_000, WIDE, Text::Passage),
    Shape::of(20_000, WIDE, Text::Short).clustered(),
];

/// How much of its vectors' bytes a merge's heap peak must fall by, gathered
/// on disk: they are all gone from it, but for what is held as the frame's
/// last piece, where each row starts.
const SPILL_SAVES_AT_LEAST: f64 = 0.9;

/// M8-R2. A merge gathered on disk holds, at its peak on the heap, less than
/// the same merge gathered in memory by at least 0.9 of its vectors' bytes,
/// by the counting allocator: whatever stage sets the peak, its vectors
/// are held at none of them.
///
/// As the reckoning's rule: run optimised, in CI's Test job.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_merge_gathered_on_disk_holds_none_of_its_vectors_on_the_heap() {
    let _alone = ALONE.lock().await;
    let mut measured = Vec::new();
    for shape in SPILLED_BUILDS {
        let in_memory = merge_measured(shape, AT_ONCE, Gathered::InMemory).await;
        let on_disk = merge_measured(shape, AT_ONCE, Gathered::OnDisk).await;
        let vectors = shape.rows * shape.dimensions * 4;
        let saved = in_memory.peak as f64 - on_disk.peak as f64;
        eprintln!(
            "a merge of {shape:?} held {} bytes in memory and {} on disk, {:.3} of its {vectors} \
             bytes of vectors less, {} bytes spilled",
            in_memory.peak,
            on_disk.peak,
            saved / vectors as f64,
            on_disk.spilled,
        );
        measured.push((shape, in_memory, on_disk));
    }
    for (shape, in_memory, on_disk) in measured {
        let vectors = shape.rows * shape.dimensions * 4;
        assert_eq!(in_memory.spilled, 0, "{shape:?}");
        assert_eq!(on_disk.spilled, vectors, "{shape:?} gathered in memory");
        let saved = in_memory.peak as f64 - on_disk.peak as f64;
        assert!(
            saved >= SPILL_SAVES_AT_LEAST * vectors as f64,
            "{shape:?} saved {saved} of {vectors} bytes of vectors: {} in memory, {} on disk",
            in_memory.peak,
            on_disk.peak
        );
    }
}

/// What a build tokenizes at once.
const AT_ONCE: usize = vellev_storage::TOKENIZED_AT_ONCE;

/// The merges whose whole build is measured: passages at three sizes, and
/// text of 200 words over a large vocabulary, whose tokens hold more than
/// what later stages come to, past a chunk, so that it is measured to save
/// them.
const WHOLE_BUILDS: [Shape; 4] = [
    Shape::of(20_000, WIDE, Text::Passage),
    Shape::of(100_000, WIDE, Text::Passage),
    Shape::of(200_000, WIDE, Text::Passage),
    Shape::of(CHUNKED, NARROW, Text::Wide),
];

/// How much of a merge whose tokens set its peak the chunks must save, of
/// the 17.5–21.3% measured: a quarter of a chunk's tokens kept on would take
/// it under.
const MERGE_SAVES_AT_LEAST: f64 = 0.15;

/// How much two builds' peaks may differ with nothing to tell them by: see
/// the rule below.
const PEAK_TOLD_TO: f64 = 0.001;

/// M1-R1. A merge's build holds at most `TOKENIZED_AT_ONCE` rows' tokens at
/// once, however many it merges, and over the whole of it, stage by stage,
/// holds no more at its peak for that than the same build tokenizing every
/// row at once, as it did before it took chunks (node memory note, §2.1).
///
/// What this holds of the tokens themselves differs by shape, and only Wide
/// text's peak tells of them. At 768 dimensions, passages lose a quarter from
/// the stage that indexes them, but encoding the segment, with its body and
/// its frame both held, comes to as much as that stage held before, and sets
/// the peak now, so that a build keeping a chunk's tokens on for longer
/// stays under it unseen here: the peak of an open, which encodes nothing,
/// is what bites on tokens at passages (see below). Wide text's tokens hold
/// more than encoding does, and chunks must save 15% of it. The count of
/// rows held is what the build says: it cannot see tokens kept on after they
/// are indexed. Each merge holds the reckoning's bounds too.
///
/// "No more" is to within [`PEAK_TOLD_TO`], a thousandth. Where a stage
/// after the tokens sets the peak, the two builds make the same allocations
/// there, and differ only by which thread frees what first: 64 bytes of
/// 181 MB has been seen. Compared to the byte the rule would fail at random.
/// The all-at-once build's own peak has run 2% apart from itself as its
/// indexing stage and encoding tie, which helps nothing here: a thousandth
/// of 1.7 GB is 1.7 MB, against the 42 MB of a chunk of passages' tokens.
///
/// As the reckoning's rule: run optimised, in CI's Test job, as
/// `cargo test --release -p vellev-storage --test flush_memory -- --ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_merge_holds_a_chunk_of_tokens_at_once_and_its_whole_build_no_more_for_it() {
    let _alone = ALONE.lock().await;
    let mut measured = Vec::new();
    for shape in WHOLE_BUILDS {
        let all = merge_measured(shape, usize::MAX, Gathered::InMemory).await;
        let chunked = merge_measured(shape, AT_ONCE, Gathered::InMemory).await;
        let saved = (all.peak as f64 - chunked.peak as f64) / all.peak as f64;
        let off = |by: usize| (by as f64 - chunked.peak as f64) / chunked.peak as f64 * 100.0;
        eprintln!(
            "a merge of {} rows of {}, {:?} text, held {} bytes at its peak with every row \
             tokenized at once ({} rows' tokens held), {} in chunks ({} rows' tokens held), \
             {:.1}% less, reckoned {} ({:+.0}%), planned {} ({:+.0}%)",
            shape.rows,
            shape.dimensions,
            shape.text,
            all.peak,
            all.tokenized_at_once,
            chunked.peak,
            chunked.tokenized_at_once,
            saved * 100.0,
            chunked.reckoned,
            off(chunked.reckoned),
            chunked.planned,
            off(chunked.planned),
        );
        measured.push((shape, all, chunked, saved));
    }
    for (shape, all, chunked, saved) in measured {
        let rows = shape.rows;
        assert!(
            (AT_ONCE.min(rows / 2)..=AT_ONCE).contains(&chunked.tokenized_at_once),
            "a merge of {rows} rows held {} rows' tokens at once",
            chunked.tokenized_at_once
        );
        assert!(
            chunked.peak as f64 <= all.peak as f64 * (1.0 + PEAK_TOLD_TO),
            "a merge of {rows} rows held {} bytes in chunks, {} tokenizing every row at once",
            chunked.peak,
            all.peak
        );
        if matches!(shape.text, Text::Wide) {
            assert!(
                saved >= MERGE_SAVES_AT_LEAST,
                "a merge of {rows} rows saved {:.1}% of its peak, its tokens setting it",
                saved * 100.0
            );
        }
        within_bounds(
            &format!("{shape:?}"),
            chunked.peak,
            chunked.reckoned,
            chunked.planned,
        );
    }
}

/// A flush and then an open of the rows it wrote.
#[derive(Debug, Clone, Copy)]
struct Built {
    /// How many rows' tokens the flush held at once, and what it held at
    /// its peak.
    flush_tokens: usize,
    flush_peak: usize,
    /// How many of its one segment's rows' tokens the open held at once,
    /// and what it held at its peak.
    open_tokens: usize,
    open_peak: usize,
}

/// A flush of `rows` rows of `dimensions` titled as `text` says, and an open
/// of what it wrote, `at_once` rows' text tokenized at a time.
async fn flush_and_open(rows: usize, dimensions: usize, text: Text, at_once: usize) -> Built {
    let _tokenizing = Tokenizing::at_once(at_once);
    let dir = tempfile::tempdir().unwrap();
    let opened = within(WRITTEN_WITHIN, "the writer opened", on_disk(dir.path()));
    let (cluster, writer) = opened.await;
    let mut random = Random(11);
    let writes = Writes {
        rows,
        dimensions,
        first: 0,
        place: 0,
        text,
        rewriting: None,
        collections: 1,
    };
    let written = async {
        writer
            .commit(vec![collection(0, dimensions)])
            .await
            .unwrap();
        committed(&writer, &mut random, writes).await;
    };
    within(WRITTEN_WITHIN, "the rows written", written).await;
    COUNTING.reset_peak();
    let before = COUNTING.now();
    let flushed = within(MERGED_WITHIN, "the flush ended", writer.flush()).await;
    flushed.unwrap();
    let flush_peak = COUNTING.peak() - before;
    let flush_tokens = writer.last_tokenized_at_once();
    drop(writer);
    COUNTING.reset_peak();
    let before = COUNTING.now();
    let copy = within(MERGED_WITHIN, "the open ended", cluster.open()).await;
    Built {
        flush_tokens,
        flush_peak,
        open_tokens: copy.partition().replayed().tokenized_at_once,
        open_peak: COUNTING.peak() - before,
    }
}

const MB: usize = 1_000_000;

/// The flushes and opens measured, and what each may hold at its peak in
/// chunks: a ceiling, 2% over what was measured, where a chunk's peak
/// repeats to a few hundred kilobytes. A ratio to the all-at-once arm would
/// carry that arm's spread of a tenth, and leave room for tokens kept on:
/// a chunk of passages' tokens is 42 MB, past each of these.
const FLUSHED_AND_OPENED: [(usize, usize, Text, usize, usize); 3] = [
    (100_000, WIDE, Text::Passage, 868 * MB, 471 * MB),
    (200_000, WIDE, Text::Passage, 1_727 * MB, 917 * MB),
    (CHUNKED, NARROW, Text::Wide, 385 * MB, 386 * MB),
];

/// M1-R5. A flush and an open tokenize a chunk at a time, as a merge does,
/// each of an open's segments its own chunk, and hold no more than their
/// ceilings. An open encodes nothing, so its peak is the stage that indexes
/// its rows: the allocator's count of what the count of rows above takes on
/// trust, at passages as at Wide text. What an open saves is each segment's,
/// and each of those it loads at once holds its own chunk.
///
/// As the reckoning's rule: run optimised, in CI's Test job, as
/// `cargo test --release -p vellev-storage --test flush_memory -- --ignored`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "measured optimised, in CI's Test job"]
async fn a_flush_and_an_open_hold_a_chunk_of_tokens_at_once_and_an_open_saves_them() {
    let _alone = ALONE.lock().await;
    let mut measured = Vec::new();
    for (rows, dimensions, text, flush_at_most, open_at_most) in FLUSHED_AND_OPENED {
        let all = flush_and_open(rows, dimensions, text, usize::MAX).await;
        let chunked = flush_and_open(rows, dimensions, text, AT_ONCE).await;
        let saved = |all: usize, chunked: usize| (all as f64 - chunked as f64) / all as f64 * 100.0;
        eprintln!(
            "{rows} rows of {dimensions}, {text:?} text: a flush held {} bytes at its peak with \
             every row tokenized at once, {} in chunks, {:.1}% less; an open {}, {}, {:.1}% less",
            all.flush_peak,
            chunked.flush_peak,
            saved(all.flush_peak, chunked.flush_peak),
            all.open_peak,
            chunked.open_peak,
            saved(all.open_peak, chunked.open_peak),
        );
        measured.push((rows, all, chunked, flush_at_most, open_at_most));
    }
    for (rows, all, chunked, flush_at_most, open_at_most) in measured {
        assert_eq!(all.flush_tokens, rows, "the flush tokenized all at once");
        assert_eq!(all.open_tokens, rows, "the open tokenized all at once");
        assert!(
            (1..=AT_ONCE).contains(&chunked.flush_tokens),
            "a flush of {rows} rows held {} rows' tokens at once",
            chunked.flush_tokens
        );
        assert!(
            (1..=AT_ONCE).contains(&chunked.open_tokens),
            "an open of {rows} rows held {} rows' tokens at once",
            chunked.open_tokens
        );
        assert!(
            chunked.flush_peak <= flush_at_most,
            "a flush of {rows} rows held {}, past {flush_at_most}",
            chunked.flush_peak
        );
        assert!(
            chunked.open_peak <= open_at_most,
            "an open of {rows} rows held {}, past {open_at_most}",
            chunked.open_peak
        );
    }
}
