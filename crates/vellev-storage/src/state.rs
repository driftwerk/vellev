use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::Bound;
use std::sync::Arc;

use roaring::RoaringBitmap;
use vellev_core::{
    Collection, CollectionName, Distance, Object, ObjectId, RequestId, Value, Version,
};
use vellev_index::clustered::{ClusteredIndex, Unfit};
use vellev_index::filter::{Filter, Predicate, RowIndex};
use vellev_index::text::{TextIndex, Tokenized};
use vellev_index::vector::{BinaryIndex, Coded, Ties, VectorIndex};

use crate::answers::Answers;
use crate::catalog::Catalog;
use crate::column::{Column, Gathering};
use crate::coordinator::LogPosition;
use crate::format::{
    Answer, CollectionId, LogSegment, Manifest, Op, Row, SegmentCurve, SegmentFile, SegmentIndex,
    SegmentMeta,
};
use crate::{Result, StorageError};

type RowKey = (CollectionId, ObjectId);

/// Up to this many of a collection's rows a layer scores every one exactly
/// on a vector search; only beyond it is a vector index built, and with it
/// any approximation used. Scoring this many takes well under a millisecond.
pub(crate) const SHORTLIST_ABOVE: usize = 4_000;
/// A collection with more vectors than this in one segment gets a clustered
/// index, which takes a while to build and is kept in a file of its own, in
/// place of a scan of every code.
pub const CLUSTERED_ABOVE: usize = 200_000;

/// When segments are merged, and when one is large enough for a clustered
/// vector index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Merging {
    /// A segment with at least this share of its rows dead is rewritten on
    /// its own, to give the space back.
    pub stale: f64,
    /// How many adjacent small segments are merged into one. Fewer than two
    /// is taken as two.
    pub fanout: usize,
    /// The most live rows a merge produces; a segment holding under half
    /// of it counts as small.
    pub rows: usize,
    /// A collection with more vectors than this in one segment gets a
    /// clustered index.
    pub clustered_above: usize,
}

impl Default for Merging {
    fn default() -> Self {
        Self {
            stale: 0.5,
            fanout: 4,
            rows: 1_000_000,
            clustered_above: CLUSTERED_ABOVE,
        }
    }
}

/// A run of adjacent segments to be replaced by one holding their live rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Merge {
    /// The segments' places in the list, oldest first.
    pub run: std::ops::Range<usize>,
    /// Their files' keys, to tell at commit time that the list is as it was.
    pub keys: Vec<String>,
}

/// How one collection's rows written since the last checkpoint are coded
/// for search (burst plan F2b).
#[derive(Debug, Clone, PartialEq)]
pub struct MemtableCoding {
    /// How many versions of the collection's rows with a vector have been
    /// applied since the checkpoint, dead or not: what decides coding.
    pub applied: usize,
    /// How many of the memtable's rows have a code.
    pub coded: usize,
    /// The centre the codes are taken about, in the rotated space.
    pub centre: Vec<f32>,
    /// A digest of the codes, bit for bit: see [`vellev_index`]'s
    /// `BinaryIndex::digest`.
    pub digest: u64,
}

/// What is indexed about one collection's rows in a layer: their filterable
/// values, their searchable text and, once there are too many to score one
/// by one, their vectors.
#[derive(Debug, Clone, Default)]
pub(crate) struct LayerIndex {
    pub rows: RowIndex,
    pub text: TextIndex,
    pub vectors: Option<Vectors>,
}

/// A layer's vector index: a scan of every code while the layer is small
/// enough, clusters of them above [`CLUSTERED_ABOVE`].
#[derive(Debug, Clone)]
pub(crate) enum Vectors {
    Binary(BinaryIndex),
    Clustered(Arc<ClusteredIndex>),
}

impl VectorIndex for Vectors {
    fn len(&self) -> usize {
        match self {
            Vectors::Binary(index) => index.len(),
            Vectors::Clustered(index) => index.len(),
        }
    }

    fn unmeasured(&self, only: Option<&RoaringBitmap>, skipped: &[bool]) -> Vec<u32> {
        match self {
            Vectors::Binary(index) => index.unmeasured(only, skipped),
            Vectors::Clustered(index) => index.unmeasured(only, skipped),
        }
    }

    fn estimates(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
        ties: Ties<'_>,
    ) -> Vec<(f32, u32)> {
        match self {
            Vectors::Binary(index) => index.estimates(query, count, only, skipped, ties),
            Vectors::Clustered(index) => index.estimates(query, count, only, skipped, ties),
        }
    }
}

/// Clustered indexes read from a segment's index file, by collection.
pub(crate) type Stored = HashMap<CollectionId, Arc<ClusteredIndex>>;

/// The keys a segment and its companion files are written under.
#[derive(Debug, Clone)]
pub(crate) struct Names {
    pub segment: String,
    pub index: String,
    pub vectors: String,
}

/// What a segment to be written is made of, before it is named.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    pub rows: usize,
    pub tombstones: usize,
    pub clustered: bool,
    pub vectors: bool,
}

impl Shape {
    /// Of a segment of `rows`, which have their vectors with them still.
    #[cfg(test)]
    pub fn of(catalog: &Catalog, rows: &[Row], tombstones: usize, clustered_above: usize) -> Self {
        let has = |row: usize| rows[row].object.vector.is_some();
        Self::by(catalog, rows, has, tombstones, clustered_above)
    }

    /// Of a segment of `rows`, whose vectors are gathered in `vectors`.
    pub fn of_gathered(
        catalog: &Catalog,
        rows: &[Row],
        vectors: &Gathering,
        tombstones: usize,
        clustered_above: usize,
    ) -> Self {
        let has = |row: usize| vectors.has(row);
        Self::by(catalog, rows, has, tombstones, clustered_above)
    }

    fn by(
        catalog: &Catalog,
        rows: &[Row],
        has: impl Fn(usize) -> bool,
        tombstones: usize,
        clustered_above: usize,
    ) -> Self {
        Self {
            rows: rows.len(),
            tombstones,
            clustered: clustered(catalog, rows, &has, clustered_above),
            vectors: (0..rows.len()).any(&has),
        }
    }
}

impl Names {
    /// How the manifest lists a segment of `shape` written under these
    /// names.
    pub fn meta(&self, shape: Shape) -> SegmentMeta {
        SegmentMeta {
            key: self.segment.clone(),
            rows: shape.rows as u64,
            tombstones: shape.tombstones as u64,
            index: shape.clustered.then(|| self.index.clone()),
            vectors: shape.vectors.then(|| self.vectors.clone()),
            // A new segment scans as its index was built: its curve comes
            // with the pass that makes one (recall note, R7).
            probes: vec![],
        }
    }
}

impl From<SegmentIndex> for Stored {
    fn from(index: SegmentIndex) -> Self {
        index.vectors.into_iter().collect()
    }
}

impl LayerIndex {
    fn insert(&mut self, schema: &Collection, row: u32, object: &Object) {
        self.rows.insert(schema, row, object);
        self.text.insert(schema, row, object);
    }
}

/// How many live rows of a collection have text in a property, and how many
/// words between them.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FieldTotals {
    pub rows: u64,
    pub words: u64,
}

/// What keyword scoring needs to know about a collection's live rows as a
/// whole. It is moved with every row that becomes live or dead, so it is
/// exact at every position and costs a query nothing to read.
#[derive(Debug, Clone, Default)]
pub(crate) struct Corpus {
    pub objects: u64,
    pub fields: HashMap<String, FieldTotals>,
}

impl Corpus {
    fn count(&mut self, index: Option<&LayerIndex>, row: u32, live: bool) {
        let step = |total: &mut u64, by: u64| {
            *total = if live {
                *total + by
            } else {
                total.checked_sub(by).expect("a row dies once")
            };
        };
        step(&mut self.objects, 1);
        for (property, length) in index.into_iter().flat_map(|index| index.text.lengths(row)) {
            let field = match self.fields.get_mut(property) {
                Some(field) => field,
                None => self.fields.entry(property.to_string()).or_default(),
            };
            step(&mut field.rows, 1);
            step(&mut field.words, u64::from(length));
        }
    }
}

/// A layer's indexes, one per collection with rows in it. Like liveness
/// they are derived from the rows and never stored.
pub(crate) type Indexes = HashMap<CollectionId, LayerIndex>;

/// A live object, and its vector where the layer keeps the two apart.
#[derive(Debug, Clone, Copy)]
pub struct Live<'a> {
    pub object: &'a Object,
    pub vector: Option<&'a [f32]>,
}

impl Live<'_> {
    pub fn id(&self) -> ObjectId {
        self.object.id
    }

    /// The object whole, with its vector.
    pub fn to_object(self) -> Object {
        Object {
            vector: self.vector.map(<[f32]>::to_vec),
            ..self.object.clone()
        }
    }

    /// The object's version: of its properties and vector, as they are
    /// stored.
    pub fn version(&self) -> Version {
        Version::of(&self.object.properties, self.vector)
    }

    /// Whether this is `object`, vector and all.
    pub fn is(&self, object: &Object) -> bool {
        self.object.id == object.id
            && self.object.properties == object.properties
            && self.vector == object.vector.as_deref()
    }
}

/// The rows of one layer, a segment or the memtable, with what is known
/// about them. Later layers shadow earlier ones. A segment's rows carry no
/// vectors; its column does.
pub(crate) struct Layer<'a> {
    pub rows: &'a [Row],
    pub dead: &'a [bool],
    /// How many of the rows are dead.
    pub deaths: usize,
    pub indexes: &'a Indexes,
    pub vectors: Option<&'a Column>,
    /// How many clusters each collection's clustered index scans, where
    /// the segment's curve says (recall note, R7); none for the memtable.
    pub curves: &'a [SegmentCurve],
}

impl<'a> Layer<'a> {
    pub fn vector(&self, row: usize) -> Option<&'a [f32]> {
        match self.vectors {
            Some(column) => column.get(row),
            None => self.rows[row].object.vector.as_deref(),
        }
    }

    pub fn id(&self, row: u32) -> ObjectId {
        self.rows[row as usize].object.id
    }

    pub fn live(&self, row: usize) -> Live<'a> {
        Live {
            object: &self.rows[row].object,
            vector: self.vector(row),
        }
    }
}

/// A segment as loaded: its file, without the rows' vectors, which are in
/// its column, and its indexes, built already or, failing that, read from
/// its index file as far as they were kept.
#[derive(Debug, Clone)]
pub(crate) struct Loaded {
    pub file: Arc<SegmentFile>,
    pub vectors: Arc<Column>,
    pub indexes: Option<Arc<Indexes>>,
    pub stored: Stored,
    /// How many threads split its rows' text at once, at the most, as its
    /// indexes were built here: 0 if they were not.
    pub tokenized_on: usize,
}

impl Loaded {
    pub fn fresh(file: SegmentFile, vectors: Column, stored: Stored) -> Self {
        Self {
            file: Arc::new(file),
            vectors: Arc::new(vectors),
            indexes: None,
            stored,
            tokenized_on: 0,
        }
    }

    /// The segment with its indexes built, under the collections of
    /// `catalog`, so that building them need not wait for other segments.
    pub fn indexed_now(self, key: &str, catalog: &Catalog) -> Result<Self> {
        let (indexes, tokenized_on) = self.indexed_on(key, catalog)?;
        Ok(Self {
            indexes: Some(indexes),
            tokenized_on,
            ..self
        })
    }

    /// The segment's indexes under the collections of `catalog`, once the
    /// column is seen to fit the rows: those built already if they index
    /// every collection there that the segment has rows of, and otherwise
    /// built from its column. Indexes built while a collection was dropped
    /// have nothing of its rows, and a restore to before the drop brings
    /// the collection back: they are then built again, keeping whatever
    /// clustered indexes they had.
    fn indexed(&self, key: &str, catalog: &Catalog) -> Result<Arc<Indexes>> {
        self.indexed_on(key, catalog).map(|(indexes, _)| indexes)
    }

    /// As [`Self::indexed`], with how many threads split the rows' text at
    /// once, at the most, if they were built here.
    fn indexed_on(&self, key: &str, catalog: &Catalog) -> Result<(Arc<Indexes>, usize)> {
        if self.vectors.len() != self.file.rows.len() {
            return Err(StorageError::corrupt(
                key,
                "the vectors are not one per row",
            ));
        }
        let built = |stored: &Stored| {
            let column = |row| self.vectors.get(row);
            // Said once, as the segment is loaded: it is searched by a scan
            // of its codes from now until it is next merged.
            let unused = |collection: CollectionId, unfit: Unfit| {
                tracing::warn!(
                    key,
                    collection = collection.0,
                    %unfit,
                    "a segment's stored index is not used"
                );
            };
            let (indexes, threads) =
                indexed_on(catalog, &self.file.rows, column, stored, unused, None);
            (Arc::new(indexes), threads)
        };
        Ok(match &self.indexes {
            Some(indexes) if self.lacking(indexes, catalog) => built(&stored(key, indexes).into()),
            Some(indexes) => (indexes.clone(), 0),
            None => built(&self.stored),
        })
    }

    /// Whether `indexes` have nothing of a collection of `catalog` that the
    /// segment has rows of.
    fn lacking(&self, indexes: &Indexes, catalog: &Catalog) -> bool {
        let unindexed: Vec<CollectionId> = catalog
            .collections()
            .map(|(collection, _)| collection)
            .filter(|collection| !indexes.contains_key(collection))
            .collect();
        let rows = &self.file.rows;
        !unindexed.is_empty() && rows.iter().any(|row| unindexed.contains(&row.collection))
    }

    /// The segment as `meta` names it, seen to be fit to adopt, with its
    /// indexes built under the collections of `catalog` if they were not
    /// built already.
    pub fn fit(self, meta: SegmentMeta, catalog: &Catalog) -> Result<Fit> {
        let indexes = self.indexed(&meta.key, catalog)?;
        let Loaded { file, vectors, .. } = self;
        Ok(Fit {
            meta,
            file,
            vectors,
            indexes,
        })
    }
}

/// A segment the writer built and loaded back, checked while that could
/// still fail and change nothing: its vectors are one per row and its
/// indexes built, so the state adopts it, once the commit that names it is
/// made, with nothing left that could fail part way.
#[derive(Debug)]
pub(crate) struct Fit {
    pub meta: SegmentMeta,
    file: Arc<SegmentFile>,
    vectors: Arc<Column>,
    indexes: Arc<Indexes>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    /// A row index in the memtable.
    Memtable(usize),
    /// A segment's index in the state, and a row index in its file.
    Segment(usize, usize),
}

#[derive(Debug, Clone)]
struct Segment {
    meta: SegmentMeta,
    file: Arc<SegmentFile>,
    vectors: Arc<Column>,
    indexes: Arc<Indexes>,
    dead: Vec<bool>,
    deaths: usize,
    /// About how many bytes the segment takes in memory.
    resident: usize,
}

/// About how many bytes a segment of `rows` with `vectors` takes in memory:
/// its rows and what indexes them, and its vectors if they are held in
/// memory rather than read in place.
fn segment_bytes(rows: &[Row], vectors: &Column) -> usize {
    rows.iter().map(row_bytes).sum::<usize>() + vectors.resident()
}

/// About how many bytes a segment's vector indexes take in memory: its
/// clustered indexes' codes, factors, rows and centroids, or its one-bit
/// codes.
fn indexes_bytes(indexes: &Indexes) -> usize {
    indexes
        .values()
        .map(|index| match &index.vectors {
            Some(Vectors::Clustered(clustered)) => clustered.resident(),
            Some(Vectors::Binary(codes)) => codes.resident(),
            None => 0,
        })
        .sum()
}

/// About how many bytes a row takes in memory, with its place among the
/// live rows and what the filter and keyword indexes hold of it: about as
/// much again as its text, and a little for each value.
fn row_bytes(Row { object, .. }: &Row) -> usize {
    const ROW: usize = 160 + 64;
    let properties: usize = object
        .properties
        .iter()
        .map(|(name, value)| {
            name.len()
                + 80
                + match value {
                    Value::Text(text) => text.len() * 2,
                    Value::TextArray(texts) => texts.iter().map(|text| text.len() * 2 + 24).sum(),
                    _ => 0,
                }
        })
        .sum();
    ROW + properties + object.vector.as_ref().map_or(0, |vector| vector.len() * 4)
}

#[derive(Debug, Clone, Default)]
struct Memtable {
    rows: Vec<Row>,
    dead: Vec<bool>,
    deaths: usize,
    tombstones: Vec<RowKey>,
    indexes: Indexes,
    /// About how many bytes `rows` take, with what indexes them.
    bytes: usize,
    /// For each collection, how many versions of its rows with a vector
    /// have been applied since the checkpoint, dead or not: what decides
    /// when its rows are coded (see [`CODE_FROM`]). It only rises.
    applied: HashMap<CollectionId, usize>,
    /// Moved on whenever a collection's memtable is coded afresh or let
    /// go, so that codes prepared before are known for another centre's.
    /// Unique in the process, not just in this state, so that codes
    /// prepared from a state another has since replaced are known too.
    coding: Coding,
}

/// A coding generation: drawn from one count for the whole process, so
/// that no two states, and no two codings of one, share one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Coding(u64);

impl Coding {
    fn next() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

impl Default for Coding {
    fn default() -> Self {
        Self::next()
    }
}

/// Codes worked out for a log's rows before it is applied: see
/// [`State::prepared`]. Keyed by the row's place among the log's ops.
#[derive(Debug, Default)]
pub(crate) struct Prepared {
    coding: Option<Coding>,
    codes: HashMap<usize, Option<Coded>>,
    /// Each upsert's text split into tokens, with the searchable properties
    /// of its collection it was split by: used only if they are still the
    /// searchable properties as the upsert is applied.
    texts: HashMap<usize, (std::sync::Arc<Vec<String>>, Tokenized)>,
}

/// The searchable properties of `schema`, in its order: what splitting an
/// object's text into tokens depends on.
fn searchable(schema: &Collection) -> Vec<String> {
    schema
        .properties
        .iter()
        .filter(|property| property.searchable)
        .map(|property| property.name.clone())
        .collect()
}

/// The memtable a flush keeps, made before the state is locked to adopt
/// the flush: see [`State::kept`].
#[derive(Debug)]
pub(crate) struct Kept {
    held: usize,
    rows: usize,
    indexes: Indexes,
    applied: HashMap<CollectionId, usize>,
}

/// From how many versions of a collection's rows with a vector applied
/// since the checkpoint the memtable's rows of it are coded one bit a
/// dimension, for a search to shortlist as it does a segment's (burst plan
/// F2b). The codes are taken about the mean of those first versions, dead
/// or not, in log order: what any copy replaying the log from the same
/// checkpoint arrives at too. Once coded, the memtable stays coded until
/// it is flushed, however many of its rows die.
///
/// Those first versions are never coded, and are scored exactly beside the
/// shortlist: reaching the threshold sums them and codes nothing, so that
/// the apply that does it holds the lock readers wait on for well under a
/// millisecond.
pub const CODE_FROM: usize = 1_024;

/// The memtable's rows of `collection` with a vector, centred on the mean
/// of the first [`CODE_FROM`] of them in row order, which is log order
/// since the checkpoint: those first rows taken to be scored exactly and
/// never coded, every row after them coded. At the threshold that is a sum
/// and no code; at a flush, a code for each row kept past the first
/// [`CODE_FROM`], as applying them would have made.
fn coded(rows: &[Row], collection: CollectionId, metric: Distance) -> Option<Vectors> {
    let vectored: Vec<(u32, &[f32])> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.collection == collection)
        .filter_map(|(at, row)| Some((at as u32, row.object.vector.as_deref()?)))
        .collect();
    let first: Vec<&[f32]> = vectored
        .iter()
        .take(CODE_FROM)
        .map(|(_, vector)| *vector)
        .collect();
    let mut codes = BinaryIndex::about_mean(metric, &first)?;
    for (at, (row, vector)) in vectored.iter().enumerate() {
        match at < CODE_FROM {
            true => codes.push_exact(*row),
            false => codes.push(*row, vector),
        }
    }
    Some(Vectors::Binary(codes))
}

/// A partition in memory as of one position: flushed segments, the rows
/// written since the last checkpoint, and which row is the live one for each
/// key. Liveness is derived on load and never stored; it is a function of the
/// committed files.
#[derive(Debug, Clone, Default)]
pub struct State {
    position: LogPosition,
    head: Option<String>,
    /// The latest checkpoint's key.
    checkpoint: Option<String>,
    catalog: Catalog,
    segments: Vec<Segment>,
    memtable: Memtable,
    /// Where the live row of each key lies, in key order: a collection's
    /// live rows are a range of it, by id, which is how they are paged.
    live: BTreeMap<RowKey, Location>,
    corpora: HashMap<CollectionId, Corpus>,
    /// The logs applied since the last checkpoint, oldest first: what is
    /// in the memtable and in no segment.
    since: Vec<String>,
    answers: Answers,
    texts_applied: TextsApplied,
}

/// How the text of the upserts a state applied was split into tokens: by
/// [`State::prepared`], before the state was locked, or under the lock,
/// because it was not prepared or the searchable properties it was split
/// by had changed. Counted so a test can hold that prepared tokens are
/// used, and not split again under the lock.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TextsApplied {
    /// Upserts indexed with the tokens prepared for them.
    pub prepared: usize,
    /// Upserts whose text was split into tokens under the lock.
    pub under_lock: usize,
}

/// What a flush takes out of the memtable: freeing it takes long enough
/// that it is better done without holding the state.
pub(crate) struct Discarded {
    _rows: Vec<Row>,
    _indexes: Indexes,
}

/// What a flush takes from the state to build its segment while commits go
/// on: the memtable's live rows and tombstones at `position`, and where
/// each row lies in the memtable.
#[derive(Debug, Clone)]
pub(crate) struct Snapshot {
    pub position: LogPosition,
    pub checkpoint: Option<String>,
    pub catalog: Catalog,
    /// The rows taken, without their vectors, which are in `vectors`.
    pub rows: Vec<Row>,
    /// The rows' vectors, gathered once into the frame of the column the
    /// flushed segment keeps, rather than copied with each row and gathered
    /// again.
    vectors: Gathering,
    pub tombstones: Vec<RowKey>,
    /// Where each of `rows` lies in the memtable.
    at: Vec<usize>,
    /// How many rows the memtable held: every one of them is in the
    /// flushed segment or dead by the time the flush is adopted.
    held: usize,
    /// How many of the logs since the checkpoint it covers.
    logs: usize,
}

#[cfg(test)]
impl State {
    /// Move the state on to `position` with nothing committed there, as
    /// nothing a writer lets happen does.
    pub(crate) fn moved_to(&mut self, position: LogPosition) {
        self.position = position;
    }
}

impl Snapshot {
    /// The taken rows' vectors as gathered, taken out of the snapshot, to
    /// be sealed into their column where nothing waits on it.
    pub(crate) fn gathered(&mut self) -> Gathering {
        std::mem::replace(&mut self.vectors, Gathering::with_capacity(0, 0))
    }
}

#[cfg(test)]
impl Snapshot {
    /// The taken rows with their vectors put back, as a segment file's rows
    /// carry them before they are built.
    pub(crate) fn rows_with_vectors(&self) -> Vec<Row> {
        self.rows
            .iter()
            .enumerate()
            .map(|(n, row)| {
                let mut row = row.clone();
                row.object.vector = self.vectors.get(n).map(<[f32]>::to_vec);
                row
            })
            .collect()
    }

    /// As this, having taken none of the rows there were.
    pub(crate) fn taking_none(self) -> Self {
        Self {
            at: Vec::new(),
            ..self
        }
    }

    /// As this, taken of a memtable of `held` rows.
    pub(crate) fn of(self, held: usize) -> Self {
        Self { held, ..self }
    }
}

/// Index `rows` for filtering, keyword search and vector search, for the
/// collections `catalog` knows, numbered as the rows lie. A clustered index
/// in `stored` for a collection is used rather than built again if it is
/// one of just the collection's rows with vectors here, of their
/// dimensions and by its metric, and `unused` is told of one that is not;
/// otherwise one is built for a collection with
/// more vectors than `clustered_above`, when that is given, which only the
/// writer does. The filter indexes are summarised, the rows being all
/// there will be. An index built under an older catalog stays right: a
/// property never changes once added, and neither a property nor a
/// collection added since has anything in an older layer.
pub(crate) fn indexed<'a>(
    catalog: &Catalog,
    rows: &'a [Row],
    vector_of: impl Fn(usize) -> Option<&'a [f32]> + Sync,
    stored: &Stored,
    unused: impl Fn(CollectionId, Unfit),
    clustered_above: Option<usize>,
) -> Indexes {
    indexed_on(catalog, rows, vector_of, stored, unused, clustered_above).0
}

/// As [`indexed`], with how many threads split the rows' text at once, at
/// the most. The text of every row is split on several threads first, and
/// then indexed in row order, which is what indexing each row in turn
/// makes.
pub(crate) fn indexed_on<'a>(
    catalog: &Catalog,
    rows: &'a [Row],
    vector_of: impl Fn(usize) -> Option<&'a [f32]> + Sync,
    stored: &Stored,
    unused: impl Fn(CollectionId, Unfit),
    clustered_above: Option<usize>,
) -> (Indexes, usize) {
    use rayon::prelude::*;
    let threads = Threads::default();
    let tokenized: Vec<Option<Tokenized>> = rows
        .par_iter()
        .map(|Row { collection, object }| {
            let schema = catalog.schema(*collection)?;
            Some(threads.on(|| TextIndex::tokenize(schema, object)))
        })
        .collect();
    // The filterable values and vectors on one thread, the text on
    // another: each in row order, as indexing each row in turn makes them.
    let ((mut indexes, vectors), texts) = rayon::join(
        || {
            let mut indexes = Indexes::new();
            let mut vectors: HashMap<CollectionId, Vec<(u32, &[f32])>> = HashMap::new();
            for (row, Row { collection, object }) in rows.iter().enumerate() {
                if let Some(schema) = catalog.schema(*collection) {
                    let index = indexes.entry(*collection).or_default();
                    index.rows.insert(schema, row as u32, object);
                    if let Some(vector) = vector_of(row) {
                        vectors
                            .entry(*collection)
                            .or_default()
                            .push((row as u32, vector));
                    }
                }
            }
            (indexes, vectors)
        },
        || {
            let mut texts: HashMap<CollectionId, TextIndex> = HashMap::new();
            for ((row, Row { collection, .. }), tokenized) in rows.iter().enumerate().zip(tokenized)
            {
                if let Some(tokenized) = tokenized {
                    texts
                        .entry(*collection)
                        .or_default()
                        .insert_tokenized(row as u32, tokenized);
                }
            }
            texts
        },
    );
    for (collection, text) in texts {
        indexes.entry(collection).or_default().text = text;
    }
    for (collection, vectors) in vectors.iter().filter(|(_, v)| v.len() > SHORTLIST_ABOVE) {
        let Some((index, schema)) = indexes.get_mut(collection).zip(catalog.schema(*collection))
        else {
            continue;
        };
        let kept = stored.get(collection).filter(|clustered| {
            clustered
                .fit(schema.distance, vectors)
                .map_err(|unfit| unused(*collection, unfit))
                .is_ok()
        });
        index.vectors = match kept {
            Some(clustered) => Some(Vectors::Clustered(clustered.clone())),
            None if clustered_above
                .is_some_and(|above| vectors.len() > above.max(SHORTLIST_ABOVE)) =>
            {
                ClusteredIndex::build(schema.distance, vectors)
                    .map(|clustered| Vectors::Clustered(Arc::new(clustered)))
            }
            None => BinaryIndex::build(schema.distance, vectors).map(Vectors::Binary),
        };
    }
    for index in indexes.values_mut() {
        index.rows.summarise();
    }
    (indexes, threads.most())
}

/// How many threads were at some work at once, at the most.
#[derive(Debug, Default)]
pub(crate) struct Threads {
    now: std::sync::atomic::AtomicUsize,
    most: std::sync::atomic::AtomicUsize,
}

impl Threads {
    /// Run `work`, counted as one thread at it while it runs.
    pub(crate) fn on<T>(&self, work: impl FnOnce() -> T) -> T {
        use std::sync::atomic::Ordering::Relaxed;
        let now = self.now.fetch_add(1, Relaxed) + 1;
        self.most.fetch_max(now, Relaxed);
        let done = work();
        self.now.fetch_sub(1, Relaxed);
        done
    }

    pub(crate) fn most(&self) -> usize {
        self.most.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Whether indexing `rows` with `clustered_above` builds any clustered
/// index, which is to say whether the segment has an index file.
pub(crate) fn clustered(
    catalog: &Catalog,
    rows: &[Row],
    has_vector: impl Fn(usize) -> bool,
    clustered_above: usize,
) -> bool {
    let mut counts: HashMap<CollectionId, usize> = HashMap::new();
    for (row, Row { collection, .. }) in rows.iter().enumerate() {
        if has_vector(row) && catalog.schema(*collection).is_some() {
            *counts.entry(*collection).or_default() += 1;
        }
    }
    counts
        .values()
        .any(|count| *count > clustered_above.max(SHORTLIST_ABOVE))
}

/// The clustered indexes among `indexes`, as the index file of the segment
/// under `segment` keeps them.
pub(crate) fn stored(segment: &str, indexes: &Indexes) -> SegmentIndex {
    let mut vectors: Vec<(CollectionId, Arc<ClusteredIndex>)> = indexes
        .iter()
        .filter_map(|(collection, index)| match &index.vectors {
            Some(Vectors::Clustered(clustered)) => Some((*collection, clustered.clone())),
            _ => None,
        })
        .collect();
    vectors.sort_by_key(|(collection, _)| *collection);
    SegmentIndex {
        segment: segment.to_string(),
        vectors,
    }
}

/// The curves of the rows each clustered index among `indexes` was tuned
/// on when built, as the segment's manifest entry keeps them (recall note,
/// R7): none for an index read back from a file.
pub(crate) fn curves(indexes: &Indexes) -> Vec<SegmentCurve> {
    let mut curves: Vec<SegmentCurve> = indexes
        .iter()
        .filter_map(|(collection, index)| match &index.vectors {
            Some(Vectors::Clustered(clustered)) => Some(SegmentCurve {
                collection: *collection,
                band: 0,
                curve: clustered.rows_curve()?,
            }),
            _ => None,
        })
        .collect();
    curves.sort_by_key(|entry| entry.collection);
    curves
}

impl State {
    /// How the text of the upserts this state applied was split into
    /// tokens: see [`TextsApplied`].
    pub fn texts_applied(&self) -> TextsApplied {
        self.texts_applied
    }

    pub fn position(&self) -> LogPosition {
        self.position
    }

    /// The checkpoint the state rests on, if any.
    pub(crate) fn checkpoint(&self) -> Option<&str> {
        self.checkpoint.as_deref()
    }

    /// About how many bytes the state takes in memory: its rows, what
    /// indexes them, and whatever vectors are held in memory rather than
    /// read in place from local files. Rough, and cheap to ask: each part
    /// is counted as it is made.
    /// About how many bytes the segments' vector indexes take: counted in
    /// [`Self::resident`].
    pub fn indexes_resident(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| indexes_bytes(&segment.indexes))
            .sum()
    }

    pub fn resident(&self) -> usize {
        let segments: usize = self.segments.iter().map(|segment| segment.resident).sum();
        let codes: usize = self
            .memtable
            .indexes
            .values()
            .map(|index| match &index.vectors {
                Some(Vectors::Binary(codes)) => codes.resident(),
                _ => 0,
            })
            .sum();
        segments + self.memtable.bytes + codes
    }

    /// How many rows are live. What a writer does in memory as it commits
    /// a checkpoint takes about as long as there are of them.
    pub fn rows(&self) -> usize {
        self.live.len()
    }

    /// How the memtable's rows of `collection` are coded for search, if
    /// they are: see [`MemtableCoding`].
    pub fn memtable_coding(&self, collection: &CollectionName) -> Option<MemtableCoding> {
        let (id, _) = self.catalog.known(collection).ok()?;
        match &self.memtable.indexes.get(&id)?.vectors {
            Some(Vectors::Binary(codes)) => Some(MemtableCoding {
                applied: self.memtable.applied.get(&id).copied().unwrap_or(0),
                coded: codes.coded(),
                centre: codes.centre().to_vec(),
                digest: codes.digest(),
            }),
            _ => None,
        }
    }

    /// Code the memtable's row `row` of `collection`, just applied, if it
    /// has a vector: on its own once the collection's rows are coded, or
    /// with every row before it as this version is the
    /// [`CODE_FROM`]th applied since the checkpoint.
    fn code(&mut self, collection: CollectionId, row: usize, prepared: Option<Option<Coded>>) {
        let Some(schema) = self.catalog.schema(collection) else {
            return;
        };
        let metric = schema.distance;
        let Memtable {
            rows,
            indexes,
            applied,
            ..
        } = &mut self.memtable;
        let Some(vector) = rows[row].object.vector.as_deref() else {
            return;
        };
        let applied = applied.entry(collection).or_default();
        *applied += 1;
        let Some(index) = indexes.get_mut(&collection) else {
            return;
        };
        match (&mut index.vectors, prepared) {
            (Some(Vectors::Binary(codes)), Some(coded)) => codes.push_coded(row as u32, coded),
            (Some(Vectors::Binary(codes)), None) => codes.push(row as u32, vector),
            _ if *applied == CODE_FROM => {
                index.vectors = coded(rows, collection, metric);
                self.memtable.coding = Coding::next();
            }
            _ => {}
        };
    }

    /// How many rows have been written since the last checkpoint, counting
    /// those since replaced: what a flush would move out of memory.
    pub fn unflushed(&self) -> usize {
        self.memtable.rows.len()
    }

    /// About how many bytes what is held in memory until the next flush
    /// takes: the rows written since the last checkpoint, with whatever
    /// codes they are searched by. Counted in [`Self::resident`] too.
    pub fn unflushed_bytes(&self) -> usize {
        let codes: usize = self
            .memtable
            .indexes
            .values()
            .map(|index| match &index.vectors {
                Some(Vectors::Binary(codes)) => codes.resident(),
                _ => 0,
            })
            .sum();
        self.memtable.bytes + codes
    }

    pub fn head(&self) -> Option<&str> {
        self.head.as_deref()
    }

    /// The segments the flushed data is in, oldest first.
    pub fn segments(&self) -> impl Iterator<Item = &SegmentMeta> {
        self.segments.iter().map(|segment| &segment.meta)
    }

    pub(crate) fn at(&self) -> (LogPosition, Option<String>) {
        (self.position, self.head.clone())
    }

    pub fn collections(&self) -> impl Iterator<Item = &Collection> {
        self.catalog.collections().map(|(_, schema)| schema)
    }

    pub fn collection(&self, name: &CollectionName) -> Option<&Collection> {
        self.catalog.get(name).map(|(_, schema)| schema)
    }

    pub fn get(&self, collection: &CollectionName, id: ObjectId) -> Option<Live<'_>> {
        self.row(self.catalog.get(collection)?.0, id)
    }

    pub(crate) fn row(&self, collection: CollectionId, id: ObjectId) -> Option<Live<'_>> {
        self.live
            .get(&(collection, id))
            .map(|location| self.located(*location))
    }

    /// The row at `location`, which must be a live one.
    fn located(&self, location: Location) -> Live<'_> {
        match location {
            Location::Memtable(row) => Live {
                object: &self.memtable.rows[row].object,
                vector: self.memtable.rows[row].object.vector.as_deref(),
            },
            Location::Segment(segment, row) => Live {
                object: &self.segments[segment].file.rows[row].object,
                vector: self.segments[segment].vectors.get(row),
            },
        }
    }

    /// The collection's live rows in id order, from just after `after`.
    fn in_order(
        &self,
        collection: CollectionId,
        after: Option<ObjectId>,
    ) -> impl Iterator<Item = (&RowKey, &Location)> {
        let from = match after {
            Some(after) => Bound::Excluded((collection, after)),
            None => Bound::Included((collection, ObjectId(uuid::Uuid::nil()))),
        };
        self.live
            .range((from, Bound::Unbounded))
            .take_while(move |((c, _), _)| *c == collection)
    }

    /// Every live object in the collection, in no particular order.
    pub fn scan<'a>(&'a self, collection: &CollectionName) -> impl Iterator<Item = Live<'a>> {
        let wanted = self.catalog.get(collection).map(|(id, _)| id);
        self.layers().flat_map(move |layer| {
            (0..layer.rows.len())
                .filter(move |row| !layer.dead[*row] && Some(layer.rows[*row].collection) == wanted)
                .map(move |row| layer.live(row))
        })
    }

    pub fn count(&self, collection: &CollectionName) -> usize {
        self.scan(collection).count()
    }

    /// Every live object in the collection that passes `filter`, in no
    /// particular order.
    pub fn select(&self, collection: &CollectionName, filter: &Filter) -> Result<Vec<Live<'_>>> {
        let (id, schema) = self.catalog.known(collection)?;
        let predicate = filter.compile(schema)?;
        Ok(self.matching(id, &predicate).collect())
    }

    /// One page of the collection in object id order: at most `limit`
    /// objects after `after` that pass `filter`. Paging by the last id seen
    /// stays correct while rows are written and deleted in between.
    ///
    /// The live rows are kept in key order, so a page with no filter costs
    /// a seek and its own size, and a walk of the whole collection costs
    /// its size once. With a filter, the rows that pass are found in each
    /// layer's index, as a search finds them, and the page is taken from
    /// them in whichever of two ways is cheaper, both giving the same
    /// rows: when few pass, the ones after `after` are partly sorted; when
    /// many do, the live rows are met in order from `after` and kept if
    /// they pass, which costs about the page's size times the share that
    /// do not. Either way a walk with a filter costs about the collection's
    /// size in all, where it used to cost that for every page.
    pub fn list(
        &self,
        collection: &CollectionName,
        filter: Option<&Filter>,
        after: Option<ObjectId>,
        limit: usize,
    ) -> Result<Vec<Live<'_>>> {
        let (id, schema) = self.catalog.known(collection)?;
        let Some(filter) = filter else {
            let page = self.in_order(id, after).take(limit);
            return Ok(page.map(|(_, location)| self.located(*location)).collect());
        };
        let predicate = filter.compile(schema)?;
        let passing: Vec<RoaringBitmap> = self
            .layers()
            .map(|layer| {
                layer
                    .indexes
                    .get(&id)
                    .map(|index| predicate.select(&index.rows))
                    .unwrap_or_default()
            })
            .collect();
        let matches: u64 = passing.iter().map(RoaringBitmap::len).sum();
        let rows = self.corpora.get(&id).map_or(0, |corpus| corpus.objects);
        // Sorting the matches costs about as many steps as there are of
        // them; meeting them in order costs about `limit` times the rows
        // there are for each that passes.
        if matches.saturating_mul(matches) <= (limit as u64).saturating_mul(rows) {
            let mut page: Vec<Live<'_>> = self
                .layers()
                .zip(&passing)
                .flat_map(|(layer, passing)| {
                    passing
                        .iter()
                        .map(|row| row as usize)
                        .filter(move |row| !layer.dead[*row])
                        .map(move |row| layer.live(row))
                })
                .filter(|found| after.is_none_or(|after| found.id() > after))
                .collect();
            if page.len() > limit {
                page.select_nth_unstable_by_key(limit, Live::id);
                page.truncate(limit);
            }
            page.sort_unstable_by_key(Live::id);
            return Ok(page);
        }
        let memtable = self.segments.len();
        let page = self
            .in_order(id, after)
            .filter(|(_, location)| match **location {
                Location::Memtable(row) => passing[memtable].contains(row as u32),
                Location::Segment(segment, row) => passing[segment].contains(row as u32),
            })
            .take(limit);
        Ok(page.map(|(_, location)| self.located(*location)).collect())
    }

    /// Every layer, oldest first.
    pub(crate) fn layers(&self) -> impl Iterator<Item = Layer<'_>> {
        let memtable = Layer {
            rows: &self.memtable.rows,
            dead: &self.memtable.dead,
            deaths: self.memtable.deaths,
            indexes: &self.memtable.indexes,
            vectors: None,
            curves: &[],
        };
        self.segments
            .iter()
            .map(|segment| Layer {
                rows: &segment.file.rows,
                dead: &segment.dead,
                deaths: segment.deaths,
                indexes: &segment.indexes,
                vectors: Some(&segment.vectors),
                curves: &segment.meta.probes,
            })
            .chain([memtable])
    }

    pub(crate) fn matching<'a, 'p>(
        &'a self,
        collection: CollectionId,
        predicate: &'p Predicate,
    ) -> impl Iterator<Item = Live<'a>> + 'p
    where
        'a: 'p,
    {
        self.layers().flat_map(move |layer| {
            layer
                .indexes
                .get(&collection)
                .map(|index| predicate.select(&index.rows))
                .unwrap_or_default()
                .into_iter()
                .map(|row| row as usize)
                .filter(move |row| !layer.dead[*row])
                .map(move |row| layer.live(row))
        })
    }

    pub(crate) fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub(crate) fn corpus(&self, collection: CollectionId) -> Option<&Corpus> {
        self.corpora.get(&collection)
    }

    pub(crate) fn loaded_segments(&self) -> impl Iterator<Item = (&str, Loaded)> {
        self.segments.iter().map(|segment| {
            let loaded = Loaded {
                file: segment.file.clone(),
                vectors: segment.vectors.clone(),
                indexes: Some(segment.indexes.clone()),
                stored: Stored::default(),
                tokenized_on: 0,
            };
            (segment.meta.key.as_str(), loaded)
        })
    }

    /// As [`Self::loaded_segments`], each with the key of its vectors
    /// file, if it has one.
    pub(crate) fn loaded_with_vectors(&self) -> impl Iterator<Item = (&str, Option<&str>, Loaded)> {
        let vectors = self
            .segments
            .iter()
            .map(|segment| segment.meta.vectors.as_deref());
        let loaded = self.loaded_segments().zip(vectors);
        loaded.map(|((key, loaded), vectors)| (key, vectors, loaded))
    }

    fn mark_dead(&mut self, location: Location) {
        let (rows, dead, deaths, indexes, row) = match location {
            Location::Memtable(row) => {
                let Memtable {
                    rows,
                    dead,
                    deaths,
                    indexes,
                    ..
                } = &mut self.memtable;
                (rows.as_slice(), dead, deaths, &*indexes, row)
            }
            Location::Segment(segment, row) => {
                let segment = &mut self.segments[segment];
                (
                    segment.file.rows.as_slice(),
                    &mut segment.dead,
                    &mut segment.deaths,
                    &*segment.indexes,
                    row,
                )
            }
        };
        if !dead[row] {
            *deaths += 1;
        }
        dead[row] = true;
        let collection = rows[row].collection;
        if let Some(corpus) = self.corpora.get_mut(&collection) {
            corpus.count(indexes.get(&collection), row as u32, false);
        }
    }

    fn forget(&mut self, key: RowKey) -> bool {
        match self.live.remove(&key) {
            Some(location) => {
                self.mark_dead(location);
                true
            }
            None => false,
        }
    }

    /// Rebuild the partition as of a checkpoint. `segments` are the
    /// manifest's in its order, oldest first; a newer segment's rows and
    /// tombstones shadow an older one's.
    pub(crate) fn from_manifest(
        key: &str,
        manifest: Manifest,
        segments: Vec<Loaded>,
    ) -> Result<Self> {
        let catalog = Catalog::restore(manifest.next_collection_id, manifest.collections)
            .map_err(|reason| StorageError::corrupt(key, reason))?;
        let state = State {
            position: manifest.position,
            head: Some(key.to_string()),
            checkpoint: Some(key.to_string()),
            catalog,
            answers: Answers::of(manifest.answered),
            texts_applied: TextsApplied::default(),
            ..State::default()
        };
        manifest
            .segments
            .into_iter()
            .zip(segments)
            .try_fold(state, |mut state, (meta, loaded)| {
                state.attach(meta, loaded)?;
                Ok(state)
            })
    }

    /// Add a segment as the newest. The file is checked before anything is
    /// changed, so on error the state is as it was.
    fn attach(&mut self, meta: SegmentMeta, loaded: Loaded) -> Result<()> {
        let indexes = loaded.indexed(&meta.key, &self.catalog)?;
        let Loaded { file, vectors, .. } = loaded;
        let corrupt = |reason: &str| StorageError::corrupt(&meta.key, reason);
        if u32::try_from(file.rows.len()).is_err() {
            return Err(corrupt("more rows than a segment can number"));
        }
        if (file.rows.len() as u64, file.tombstones.len() as u64) != (meta.rows, meta.tombstones) {
            return Err(corrupt("row or tombstone count differs from the manifest"));
        }
        let mut keys = HashSet::with_capacity(file.rows.len());
        if !file
            .rows
            .iter()
            .all(|row| keys.insert((row.collection, row.object.id)))
        {
            return Err(corrupt("a key appears twice"));
        }
        let index = self.segments.len();
        for key in &file.tombstones {
            self.forget(*key);
        }
        let dead: Vec<bool> = file
            .rows
            .iter()
            .enumerate()
            .map(|(row, Row { collection, object })| {
                let live = self.catalog.contains(*collection);
                if live {
                    let location = Location::Segment(index, row);
                    if let Some(shadowed) = self.live.insert((*collection, object.id), location) {
                        self.mark_dead(shadowed);
                    }
                    self.corpora.entry(*collection).or_default().count(
                        indexes.get(collection),
                        row as u32,
                        true,
                    );
                }
                !live
            })
            .collect();
        self.segments.push(Segment {
            resident: segment_bytes(&file.rows, &vectors) + indexes_bytes(&indexes),
            meta,
            file,
            vectors,
            indexes,
            deaths: dead.iter().filter(|dead| **dead).count(),
            dead,
        });
        Ok(())
    }

    /// Apply one committed log segment, which must be the commit made
    /// directly on top of this state. Either the whole commit is applied or,
    /// on error, none of it. Its text is split into tokens as it is
    /// applied: a test's way of applying one log as it is.
    #[cfg(test)]
    pub(crate) fn apply(&mut self, key: &str, log: &LogSegment) -> Result<()> {
        self.apply_prepared(key, log, Prepared::default())
    }

    /// Whether any collection's rows in the memtable are coded.
    pub(crate) fn coding_anything(&self) -> bool {
        self.memtable
            .indexes
            .values()
            .any(|index| matches!(index.vectors, Some(Vectors::Binary(_))))
    }

    /// The codes of `log`'s rows with a vector whose collection's memtable
    /// is coded now, worked out without changing the state: so that they
    /// can be made before the state is locked to apply the log, and the
    /// apply only stores them. A row whose collection is coded only during
    /// the log, at its threshold, is coded as it is applied.
    pub(crate) fn prepared(&self, log: &LogSegment) -> Prepared {
        let codes = log
            .ops
            .iter()
            .enumerate()
            .filter_map(|(at, op)| match op {
                Op::Upsert { collection, object } => {
                    let vector = object.vector.as_deref()?;
                    match &self.memtable.indexes.get(collection)?.vectors {
                        Some(Vectors::Binary(codes)) => Some((at, codes.code(vector))),
                        _ => None,
                    }
                }
                _ => None,
            })
            .collect();
        let mut by: HashMap<CollectionId, std::sync::Arc<Vec<String>>> = HashMap::new();
        let texts = log
            .ops
            .iter()
            .enumerate()
            .filter_map(|(at, op)| match op {
                Op::Upsert { collection, object } => {
                    let schema = self.catalog.schema(*collection)?;
                    let by = by
                        .entry(*collection)
                        .or_insert_with(|| std::sync::Arc::new(searchable(schema)))
                        .clone();
                    Some((at, (by, TextIndex::tokenize(schema, object))))
                }
                _ => None,
            })
            .collect();
        Prepared {
            coding: Some(self.memtable.coding),
            codes,
            texts,
        }
    }

    /// As [`Self::apply`], storing the codes `prepared` holds for the rows
    /// it holds them for, if it was prepared from the state as it is.
    pub(crate) fn apply_prepared(
        &mut self,
        key: &str,
        log: &LogSegment,
        mut prepared: Prepared,
    ) -> Result<()> {
        if prepared.coding != Some(self.memtable.coding) {
            prepared.codes.clear();
        }
        let corrupt = |reason: String| StorageError::corrupt(key, reason);
        if self.position.next() != Some(log.position) || log.prev.as_deref() != self.head() {
            return Err(corrupt(format!(
                "commit {:?} on {:?} does not follow {:?} at {:?}",
                log.position, log.prev, self.head, self.position
            )));
        }
        self.catalog = log
            .ops
            .iter()
            .try_fold(self.catalog.clone(), |mut catalog, op| {
                catalog.apply(op).map_err(corrupt)?;
                Ok::<_, StorageError>(catalog)
            })?;
        // The catalog is the commit's whole catalog by now: each
        // collection's searchable properties are worked out once for it.
        let mut searchable_now: HashMap<CollectionId, Vec<String>> = HashMap::new();
        for (op_at, op) in log.ops.iter().enumerate() {
            match op {
                Op::CreateCollection { .. } | Op::AddProperty { .. } => {}
                Op::DropCollection { collection } => {
                    let dropped: Vec<RowKey> = self
                        .in_order(*collection, None)
                        .map(|(key, _)| *key)
                        .collect();
                    dropped.into_iter().for_each(|key| {
                        self.forget(key);
                    });
                    self.memtable.tombstones.retain(|(c, _)| c != collection);
                    // Nothing can ask for a dropped collection's rows;
                    // this only frees the memory.
                    self.memtable.indexes.remove(collection);
                    self.memtable.applied.remove(collection);
                    self.memtable.coding = Coding::next();
                    self.corpora.remove(collection);
                }
                Op::Upsert { collection, object } => {
                    let key = (*collection, object.id);
                    self.forget(key);
                    self.memtable
                        .tombstones
                        .retain(|tombstone| *tombstone != key);
                    let row = self.memtable.rows.len();
                    self.live.insert(key, Location::Memtable(row));
                    // A memtable is flushed long before it holds as many
                    // rows as a segment may number. Its rows with a vector
                    // are coded once a collection has CODE_FROM of them
                    // (see `code`), the first of them scored one by one.
                    //
                    // A collection dropped later in the same commit is
                    // already gone from the catalog; its rows go unindexed.
                    let text = prepared.texts.remove(&op_at);
                    let texts_applied = &mut self.texts_applied;
                    let index = self.catalog.schema(*collection).map(|schema| {
                        let now = searchable_now
                            .entry(*collection)
                            .or_insert_with(|| searchable(schema));
                        let index = self.memtable.indexes.entry(*collection).or_default();
                        match text.filter(|(by, _)| **by == *now) {
                            Some((_, tokenized)) => {
                                texts_applied.prepared += 1;
                                index.rows.insert(schema, row as u32, object);
                                index.text.insert_tokenized(row as u32, tokenized);
                            }
                            None => {
                                texts_applied.under_lock += 1;
                                index.insert(schema, row as u32, object);
                            }
                        }
                        &*index
                    });
                    self.corpora
                        .entry(*collection)
                        .or_default()
                        .count(index, row as u32, true);
                    let at = row;
                    let row = Row {
                        collection: *collection,
                        object: object.clone(),
                    };
                    self.memtable.bytes += row_bytes(&row);
                    self.memtable.rows.push(row);
                    self.memtable.dead.push(false);
                    self.code(*collection, at, prepared.codes.remove(&op_at));
                }
                Op::Delete { collection, id } => {
                    let key = (*collection, *id);
                    if self.forget(key) {
                        self.memtable.tombstones.push(key);
                    }
                }
            }
        }
        for answered in &log.answered {
            self.answers.remember(log.position, answered.clone());
        }
        self.position = log.position;
        self.head = Some(key.to_string());
        self.since.push(key.to_string());
        Ok(())
    }

    /// Whether this partition answered `request` lately, so that a try of
    /// it again will be answered as it was.
    pub fn answered(&self, request: RequestId) -> bool {
        self.answers.get(request).is_some()
    }

    /// Where `request` became visible and how it was answered, if this
    /// partition answered it lately.
    pub(crate) fn answer(&self, request: RequestId) -> Option<(LogPosition, &Answer)> {
        self.answers.get(request)
    }

    /// When a commit made at `now`, in milliseconds since the epoch, is to
    /// say it was made: see [`Answers::stamp`].
    pub(crate) fn stamp(&self, now: u64) -> u64 {
        self.answers.stamp(now)
    }

    /// The merge most worth doing, if any: a stale segment on its own, else
    /// the oldest run of enough adjacent small segments.
    pub(crate) fn merge_plan(&self, merging: &Merging) -> Option<Merge> {
        let fanout = merging.fanout.max(2);
        let live: Vec<usize> = self
            .segments
            .iter()
            .map(|segment| segment.dead.iter().filter(|dead| !**dead).count())
            .collect();
        let plan = |run: std::ops::Range<usize>| Merge {
            keys: self.segments[run.clone()]
                .iter()
                .map(|segment| segment.meta.key.clone())
                .collect(),
            run,
        };
        let stale = self.segments.iter().zip(&live).position(|(segment, live)| {
            let dead = segment.file.rows.len() - live;
            dead > 0 && dead as f64 >= segment.file.rows.len() as f64 * merging.stale
        });
        if let Some(at) = stale {
            return Some(plan(at..at + 1));
        }
        // A segment is small while it holds under half a merge's worth of
        // rows. A run of small ones is merged once it is `fanout` long, or
        // once it is as large as it can grow, the next being too many to
        // add: flushes take in what is written while they build, so they
        // do not come out a neat fraction of a merge.
        let small = |live: usize| live < merging.rows / 2;
        let mut start = 0;
        while start < live.len() {
            let mut end = start;
            let mut rows = 0;
            while end < live.len() && small(live[end]) && rows + live[end] <= merging.rows {
                rows += live[end];
                end += 1;
            }
            let full = end < live.len() && small(live[end]) || rows >= merging.rows / 2;
            if end - start >= fanout || end - start >= 2 && full {
                return Some(plan(start..end));
            }
            start = end.max(start + 1);
        }
        None
    }

    /// The segment that replaces `merge`'s run: their live rows, oldest
    /// first, and their tombstones if there is anything older left for the
    /// tombstones to act on. `position` is the one the rows are taken at.
    /// The live rows of the run `merge` was planned over, without their
    /// vectors, and those vectors gathered once into the merged segment's
    /// column, sized exactly beforehand: not copied into each row and then
    /// gathered again.
    pub(crate) fn merged(
        &self,
        merge: &Merge,
        position: LogPosition,
    ) -> Option<(SegmentFile, Gathering)> {
        if !self.unchanged(merge) {
            return None;
        }
        let run = &self.segments[merge.run.clone()];
        fn live(segment: &Segment) -> impl Iterator<Item = usize> + '_ {
            (0..segment.file.rows.len()).filter(|row| !segment.dead[*row])
        }
        let (count, values) = run.iter().fold((0, 0), |(count, values), segment| {
            live(segment).fold((count, values), |(count, values), row| {
                (
                    count + 1,
                    values + segment.vectors.get(row).map_or(0, <[f32]>::len),
                )
            })
        });
        let mut rows: Vec<Row> = Vec::with_capacity(count);
        let mut vectors = Gathering::with_capacity(values, count);
        for segment in run {
            for row in live(segment) {
                let Row { collection, object } = &segment.file.rows[row];
                vectors.push(segment.vectors.get(row));
                rows.push(Row {
                    collection: *collection,
                    object: Object {
                        id: object.id,
                        properties: object.properties.clone(),
                        vector: None,
                    },
                });
            }
        }
        let tombstones: Vec<RowKey> = match merge.run.start {
            0 => Vec::new(),
            _ => {
                let mut seen = HashSet::new();
                run.iter()
                    .flat_map(|segment| segment.file.tombstones.iter().copied())
                    .filter(|key| seen.insert(*key))
                    .collect()
            }
        };
        let file = SegmentFile {
            position,
            rows,
            tombstones,
        };
        Some((file, vectors))
    }

    /// Whether the run `merge` was planned over is still the same
    /// segments, so that a file built from them still describes it.
    pub(crate) fn unchanged(&self, merge: &Merge) -> bool {
        self.segments.get(merge.run.clone()).is_some_and(|run| {
            run.iter()
                .map(|segment| segment.meta.key.as_str())
                .eq(merge.keys.iter().map(String::as_str))
        })
    }

    /// The manifest a merge commits at `position`: as a flush built from
    /// `snapshot` commits (see [`Self::flushed_manifest`]), with the run
    /// replaced by `merged`. None if the run is no longer the segments the
    /// merge was planned over, when `merged` describes nothing there is.
    pub(crate) fn merged_manifest(
        &self,
        position: LogPosition,
        snapshot: &Snapshot,
        flushed: Option<&Fit>,
        merge: &Merge,
        merged: SegmentMeta,
    ) -> Result<Option<Manifest>> {
        if !self.unchanged(merge) {
            return Ok(None);
        }
        let mut manifest = self.flushed_manifest(position, snapshot, flushed)?;
        manifest.segments.splice(merge.run.clone(), [merged]);
        Ok(Some(manifest))
    }

    /// Carry out a committed merge: the run's segments go, and the merged
    /// one takes their place. A row of the run that died since the merged
    /// file was made is dead in it too. The flush committed with it is
    /// adopted after, by [`Self::flushed`].
    pub(crate) fn compacted(&mut self, merge: &Merge, merged: Fit) {
        let Fit {
            meta,
            file,
            vectors,
            indexes,
        } = merged;
        let run = merge.run.clone();
        let gone = run.len() - 1;
        let at_row: HashMap<RowKey, usize> = file
            .rows
            .iter()
            .enumerate()
            .map(|(row, Row { collection, object })| ((*collection, object.id), row))
            .collect();
        let dead: Vec<bool> = file
            .rows
            .iter()
            .map(|Row { collection, object }| {
                !matches!(
                    self.live.get(&(*collection, object.id)),
                    Some(Location::Segment(segment, _)) if run.contains(segment)
                )
            })
            .collect();
        for location in self.live.values_mut() {
            if let Location::Segment(segment, row) = location {
                if run.contains(segment) {
                    let key = &self.segments[*segment].file.rows[*row];
                    *location =
                        Location::Segment(run.start, at_row[&(key.collection, key.object.id)]);
                } else if *segment >= run.end {
                    *location = Location::Segment(*segment - gone, *row);
                }
            }
        }
        self.segments.splice(
            run.clone(),
            [Segment {
                resident: segment_bytes(&file.rows, &vectors) + indexes_bytes(&indexes),
                meta,
                file,
                vectors,
                indexes,
                deaths: dead.iter().filter(|dead| **dead).count(),
                dead,
            }],
        );
    }

    /// What a flush needs to build its segment without the writer, or
    /// `None` when nothing was committed since the last checkpoint. Its
    /// rows are left to be copied by [`Self::copy_rows`], a part at a time,
    /// so that commits are not held up while all of them are.
    pub(crate) fn snapshot(&self) -> Option<Snapshot> {
        (!self.since.is_empty()).then(|| self.taken())
    }

    /// As [`Self::snapshot`], of whatever was committed since the last
    /// checkpoint, be it nothing: what a merge flushes along with it.
    pub(crate) fn taken(&self) -> Snapshot {
        let at: Vec<usize> = (0..self.memtable.rows.len())
            .filter(|row| !self.memtable.dead[*row])
            .collect();
        let values: usize = at
            .iter()
            .map(|row| {
                self.memtable.rows[*row]
                    .object
                    .vector
                    .as_ref()
                    .map_or(0, Vec::len)
            })
            .sum();
        let vectors = Gathering::with_capacity(values, at.len());
        Snapshot {
            position: self.position,
            checkpoint: self.checkpoint.clone(),
            catalog: self.catalog.clone(),
            rows: Vec::with_capacity(at.len()),
            vectors,
            tombstones: self.memtable.tombstones.clone(),
            at,
            held: self.memtable.rows.len(),
            logs: self.since.len(),
        }
    }

    /// Copy into `snapshot` up to `most` more of its rows. A memtable row
    /// never changes once written, and none goes until the flush that took
    /// the snapshot is adopted, so they may be copied in parts while
    /// commits go on. Returns whether any are left to copy.
    pub(crate) fn copy_rows(&self, snapshot: &mut Snapshot, most: usize) -> bool {
        let from = snapshot.rows.len();
        let to = (from + most).min(snapshot.at.len());
        for &row in &snapshot.at[from..to] {
            let Row { collection, object } = &self.memtable.rows[row];
            snapshot.vectors.push(object.vector.as_deref());
            snapshot.rows.push(Row {
                collection: *collection,
                object: Object {
                    id: object.id,
                    properties: object.properties.clone(),
                    vector: None,
                },
            });
        }
        to < snapshot.at.len()
    }

    /// The manifest a flush built from `snapshot` commits at `position`:
    /// the segments there are, then the flushed one if any, the collections
    /// as at the snapshot, and the logs committed since to apply on top.
    ///
    /// It is made only for a flush this state can adopt (see
    /// [`Self::unadoptable`]), and whoever makes it holds the writer from
    /// here until [`Self::flushed`] adopts the flush, so that what held
    /// here holds there, where nothing may fail any more.
    pub(crate) fn flushed_manifest(
        &self,
        position: LogPosition,
        snapshot: &Snapshot,
        flushed: Option<&Fit>,
    ) -> Result<Manifest> {
        if let Some(reason) = self.unadoptable(snapshot, flushed) {
            let key = self.head.as_deref().unwrap_or_default();
            return Err(StorageError::corrupt(key, reason));
        }
        let flushed = flushed.map(|fit| fit.meta.clone());
        Ok(Manifest {
            replay: self.since[snapshot.logs..].to_vec(),
            replay_after: snapshot.position,
            ..self.manifest(position, &snapshot.catalog, flushed)
        })
    }

    /// Why this state cannot adopt a flush built from `snapshot` with
    /// `flushed` for its segment, if it cannot: none of which the writer
    /// lets happen, and any of which would leave the state wrong once the
    /// flush was committed. A memtable row only ever dies and the logs
    /// since the checkpoint only grow until a checkpoint is adopted, so
    /// what holds now holds until then.
    fn unadoptable(&self, snapshot: &Snapshot, flushed: Option<&Fit>) -> Option<&'static str> {
        let Memtable { rows, dead, .. } = &self.memtable;
        let taken = |row: &usize| snapshot.at.binary_search(row).is_ok();
        if self.checkpoint != snapshot.checkpoint {
            Some("a checkpoint was committed since the flush was begun")
        } else if snapshot.held > rows.len() || snapshot.logs > self.since.len() {
            Some("the flush was begun from more than there is")
        } else if flushed.map_or(0, |fit| fit.file.rows.len()) != snapshot.at.len() {
            Some("the flushed segment does not hold the rows the flush took")
        } else if !(0..snapshot.held).all(|row| dead[row] || taken(&row)) {
            Some("a row the flush did not take is still live")
        } else {
            None
        }
    }

    /// A manifest at `position` that puts the partition back as `target`,
    /// with its rows in memory written as `flushed`: the next checkpoint
    /// after this state's, keeping the requests this state answered lately,
    /// which were answered whatever is restored.
    pub(crate) fn restored_manifest(
        &self,
        position: LogPosition,
        target: &State,
        flushed: Option<SegmentMeta>,
    ) -> Manifest {
        Manifest {
            prev: self.checkpoint.clone(),
            next_collection_id: self.catalog.next_id().max(target.catalog.next_id()),
            answered: self.answers.kept(),
            ..target.manifest(position, &target.catalog, flushed)
        }
    }

    /// A manifest at `position` of the segments there are, then `flushed`,
    /// with the collections of `catalog` and nothing to replay.
    fn manifest(
        &self,
        position: LogPosition,
        catalog: &Catalog,
        flushed: Option<SegmentMeta>,
    ) -> Manifest {
        Manifest {
            position,
            prev: self.checkpoint.clone(),
            next_collection_id: catalog.next_id(),
            collections: catalog
                .collections()
                .map(|(id, schema)| (id, schema.clone()))
                .collect(),
            segments: self
                .segments
                .iter()
                .map(|segment| segment.meta.clone())
                .chain(flushed)
                .collect(),
            replay: Vec::new(),
            replay_after: LogPosition::default(),
            answered: self.answers.kept(),
        }
    }

    /// Adopt a committed flush built from `snapshot`, returning the rows
    /// and indexes it takes out of memory. Each of its rows
    /// still live moves from the memtable to the new segment, where it is
    /// live; one replaced or deleted since is dead there too. The memtable
    /// then keeps only what was committed after the snapshot, indexed
    /// afresh: as much as was committed while the flush was built.
    ///
    /// The flush must be one [`Self::flushed_manifest`] made a manifest
    /// for, with the writer held since: its segment then has a row for
    /// each the snapshot took, and every row the memtable held by then is
    /// one of those or dead.
    pub(crate) fn flushed(
        &mut self,
        manifest_key: &str,
        position: LogPosition,
        snapshot: Snapshot,
        segment: Option<Fit>,
        kept: Option<Kept>,
    ) -> Discarded {
        if let Some(Fit {
            meta,
            file,
            vectors,
            indexes,
        }) = segment
        {
            let index = self.segments.len();
            let dead: Vec<bool> = file
                .rows
                .iter()
                .zip(&snapshot.at)
                .enumerate()
                .map(|(at, (Row { collection, object }, row))| {
                    let key = (*collection, object.id);
                    let still = self.live.get(&key) == Some(&Location::Memtable(*row));
                    if still {
                        self.live.insert(key, Location::Segment(index, at));
                        self.memtable.dead[*row] = true;
                        self.memtable.deaths += 1;
                    }
                    !still
                })
                .collect();
            self.segments.push(Segment {
                resident: segment_bytes(&file.rows, &vectors) + indexes_bytes(&indexes),
                meta,
                file,
                vectors,
                indexes,
                deaths: dead.iter().filter(|dead| **dead).count(),
                dead,
            });
        }
        let flushed: HashSet<RowKey> = snapshot.tombstones.into_iter().collect();
        self.memtable
            .tombstones
            .retain(|tombstone| !flushed.contains(tombstone));
        let discarded = self.drop_flushed_rows(snapshot.held, kept);
        self.since.drain(..snapshot.logs);
        self.position = position;
        self.head = Some(manifest_key.to_string());
        self.checkpoint = Some(manifest_key.to_string());
        discarded
    }

    /// The memtable as a flush that holds its first `held` rows leaves it:
    /// the rows after them indexed and, past [`CODE_FROM`] versions of a
    /// collection, coded, numbered as they will be. Made under the state's
    /// read lock, which readers share, so that adopting the flush under
    /// the write lock only swaps it in, however many rows were written
    /// beside the flush.
    pub(crate) fn kept(&self, snapshot: &Snapshot) -> Kept {
        let held = snapshot.held;
        let rows = &self.memtable.rows;
        let kept = rows.get(held..).unwrap_or_default();
        let mut indexes = Indexes::default();
        let mut applied: HashMap<CollectionId, usize> = HashMap::new();
        for (row, Row { collection, object }) in kept.iter().enumerate() {
            if let Some(schema) = self.catalog.schema(*collection) {
                indexes
                    .entry(*collection)
                    .or_default()
                    .insert(schema, row as u32, object);
            }
            if object.vector.is_some() {
                *applied.entry(*collection).or_default() += 1;
            }
        }
        for (collection, versions) in &applied {
            if *versions < CODE_FROM {
                continue;
            }
            let (Some(schema), Some(index)) = (
                self.catalog.schema(*collection),
                indexes.get_mut(collection),
            ) else {
                continue;
            };
            index.vectors = coded(kept, *collection, schema.distance);
        }
        Kept {
            held,
            rows: rows.len(),
            indexes,
            applied,
        }
    }

    /// Take out the memtable's first `held` rows, every one of them dead,
    /// renumbering and indexing afresh those after them. Returns what was
    /// taken out, to be freed without holding the state.
    fn drop_flushed_rows(&mut self, held: usize, kept: Option<Kept>) -> Discarded {
        // Made beforehand of just these rows, it is swapped in; else the
        // rows kept are indexed and coded here.
        let kept = kept.filter(|kept| kept.held == held && kept.rows == self.memtable.rows.len());
        self.memtable.coding = Coding::next();
        let Memtable {
            rows,
            dead,
            deaths,
            indexes,
            bytes,
            applied,
            ..
        } = &mut self.memtable;
        let discarded = Discarded {
            _rows: rows.drain(..held).collect(),
            _indexes: std::mem::take(indexes),
        };
        *bytes = rows.iter().map(row_bytes).sum();
        dead.drain(..held);
        *deaths -= held;
        if let Some(Kept {
            indexes: made,
            applied: counted,
            ..
        }) = kept
        {
            *indexes = made;
            *applied = counted;
            for location in self.live.values_mut() {
                if let Location::Memtable(row) = location {
                    *row -= held;
                }
            }
            return discarded;
        }
        for (row, Row { collection, object }) in rows.iter().enumerate() {
            if let Some(schema) = self.catalog.schema(*collection) {
                indexes
                    .entry(*collection)
                    .or_default()
                    .insert(schema, row as u32, object);
            }
        }
        // The rows kept are the versions applied since the new checkpoint,
        // dead ones among them: coded again as a copy replaying the log
        // from that checkpoint codes them.
        applied.clear();
        for row in rows.iter().filter(|row| row.object.vector.is_some()) {
            *applied.entry(row.collection).or_default() += 1;
        }
        for (collection, versions) in applied.iter() {
            if *versions < CODE_FROM {
                continue;
            }
            let (Some(schema), Some(index)) = (
                self.catalog.schema(*collection),
                indexes.get_mut(collection),
            ) else {
                continue;
            };
            index.vectors = coded(rows, *collection, schema.distance);
        }
        for location in self.live.values_mut() {
            if let Location::Memtable(row) = location {
                *row -= held;
            }
        }
        discarded
    }

    /// The state rebuilt from a manifest's segments, brought through the
    /// manifest's replayed logs to the manifest itself, under `key` at
    /// `position`. The logs must start just after `after`, the position
    /// the segments hold, as applying them checks, and end just before
    /// the manifest.
    pub(crate) fn replayed(
        self,
        key: &str,
        position: LogPosition,
        after: LogPosition,
        logs: &[(String, LogSegment)],
    ) -> Result<(Self, Together)> {
        let Some((_, first)) = logs.first() else {
            return Ok((self, Together::default()));
        };
        let mut state = State {
            position: after,
            head: first.prev.clone(),
            ..self
        };
        let together = state.apply_together(logs)?;
        if state.position.next() != Some(position) {
            return Err(StorageError::corrupt(
                key,
                "its replayed logs do not end just before it",
            ));
        }
        Ok((
            State {
                position,
                head: Some(key.to_string()),
                checkpoint: Some(key.to_string()),
                ..state
            },
            together,
        ))
    }

    /// Applies `logs` in turn, [`REPLAYED_TOGETHER`] at a time: the text
    /// and codes of each run of them made together on several threads
    /// from the state as the run begins, and only stored as each log is
    /// applied. A collection made, coded or made searchable during a run
    /// has its rows split or coded as they are applied.
    ///
    /// How many threads made them at once, at the most.
    pub(crate) fn apply_together(&mut self, logs: &[(String, LogSegment)]) -> Result<Together> {
        let mut together = Together::default();
        for run in logs.chunks(REPLAYED_TOGETHER) {
            let (prepared, most) = self.prepared_together(run);
            together.tokenized_on = together.tokenized_on.max(most);
            if self.coding_anything() {
                together.coded_on = together.coded_on.max(most);
            }
            for ((key, log), prepared) in run.iter().zip(prepared) {
                self.apply_prepared(key, log, prepared)?;
            }
        }
        Ok(together)
    }

    /// What [`Self::prepared`] makes of each of `run`, made on several
    /// threads, and how many made them at once, at the most.
    pub(crate) fn prepared_together(&self, run: &[(String, LogSegment)]) -> (Vec<Prepared>, usize) {
        use rayon::prelude::*;
        let threads = Threads::default();
        let prepared = run
            .par_iter()
            .map(|(_, log)| threads.on(|| self.prepared(log)))
            .collect();
        (prepared, threads.most())
    }
}

/// How many logs a replay prepares together before applying them.
pub(crate) const REPLAYED_TOGETHER: usize = 16;

/// How many threads at once, at the most, split rows' text into tokens
/// and coded their vectors for logs applied together.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Together {
    pub tokenized_on: usize,
    pub coded_on: usize,
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vellev_core::{Bm25, DataType, Distance, Embedding, EmbeddingProvider, Property};

    use super::*;
    use crate::column::Column;

    fn schema(name: &str) -> Collection {
        Collection {
            name: CollectionName::new(name).unwrap(),
            properties: vec![
                Property::new("n", DataType::Int),
                Property::new("k", DataType::Text),
            ],
            embedding: Embedding {
                provider: EmbeddingProvider::None,
                model: "test".into(),
                dimensions: 2,
                query_prefix: None,
                document_prefix: None,
                properties: vec![],
            },
            distance: Distance::Cosine,
            bm25: Bm25::default(),
            access_rule: None,
        }
    }

    fn oid(id: u128) -> ObjectId {
        ObjectId(uuid::Uuid::from_u128(id))
    }

    /// One row in 199 has `k == "x"`; the rest `"y"`.
    fn k(n: i64) -> &'static str {
        if n % 199 == 0 {
            "x"
        } else {
            "y"
        }
    }

    fn upsert(collection: u64, id: u128, n: i64) -> Op {
        Op::Upsert {
            collection: CollectionId(collection),
            object: Object {
                id: oid(id),
                properties: BTreeMap::from([
                    ("n".to_string(), Value::Int(n)),
                    ("k".to_string(), Value::Text(k(n).to_string())),
                ]),
                vector: Some(vec![n as f32, 0.5]),
            },
        }
    }

    fn delete(collection: u64, id: u128) -> Op {
        Op::Delete {
            collection: CollectionId(collection),
            id: oid(id),
        }
    }

    /// A partition in memory, with the writer's part of flushing and
    /// merging done here: segments are built from what the state gives
    /// and adopted as a committed flush or merge would be.
    struct Built {
        state: State,
        files: u64,
    }

    impl Built {
        fn new(collections: &[&str]) -> Self {
            let mut built = Self {
                state: State::default(),
                files: 0,
            };
            let created = collections
                .iter()
                .enumerate()
                .map(|(id, name)| Op::CreateCollection {
                    id: CollectionId(id as u64),
                    schema: Box::new(schema(name)),
                })
                .collect();
            built.commit(created);
            built
        }

        fn commit(&mut self, ops: Vec<Op>) {
            let position = self.state.position().next().unwrap();
            let log = LogSegment {
                position,
                prev: self.state.head().map(str::to_string),
                ops,
                answered: Vec::new(),
            };
            self.state
                .apply(&format!("log-{}", position.0), &log)
                .unwrap();
        }

        /// A segment of `file`, whose rows have their vectors with them,
        /// as the writer builds and loads one back.
        fn fit(&mut self, mut file: SegmentFile, catalog: &Catalog) -> Fit {
            self.files += 1;
            let shape = Shape::of(catalog, &file.rows, file.tombstones.len(), CLUSTERED_ABOVE);
            let column = Column::taken_from(&mut file.rows);
            let names = Names {
                segment: format!("segment-{}", self.files),
                index: format!("index-{}", self.files),
                vectors: format!("vectors-{}", self.files),
            };
            Loaded::fresh(file, column, Stored::default())
                .fit(names.meta(shape), catalog)
                .unwrap()
        }

        fn flush(&mut self) {
            let Some(mut snapshot) = self.state.snapshot() else {
                return;
            };
            while self.state.copy_rows(&mut snapshot, 7) {}
            let fit = (!snapshot.rows.is_empty()).then(|| {
                let file = SegmentFile {
                    position: snapshot.position,
                    rows: snapshot.rows_with_vectors(),
                    tombstones: snapshot.tombstones.clone(),
                };
                let catalog = snapshot.catalog.clone();
                self.fit(file, &catalog)
            });
            let position = self.state.position().next().unwrap();
            self.state.flushed(
                &format!("manifest-{}", position.0),
                position,
                snapshot,
                fit,
                None,
            );
        }

        /// Merge a run of segments if there is one to merge.
        fn merge(&mut self) -> bool {
            let merging = Merging {
                fanout: 2,
                ..Merging::default()
            };
            let Some(plan) = self.state.merge_plan(&merging) else {
                return false;
            };
            let (mut file, gathered) = self.state.merged(&plan, self.state.position()).unwrap();
            for (row, at) in file.rows.iter_mut().zip(0..) {
                row.object.vector = gathered.get(at).map(<[f32]>::to_vec);
            }
            let catalog = self.state.catalog().clone();
            let fit = self.fit(file, &catalog);
            self.state.compacted(&plan, fit);
            true
        }

        fn walk(&self, collection: &str, filter: Option<&Filter>, page: usize) -> Vec<ObjectId> {
            let name = CollectionName::new(collection).unwrap();
            let mut walked = Vec::new();
            let mut after = None;
            loop {
                let found = self.state.list(&name, filter, after, page).unwrap();
                let ids: Vec<ObjectId> = found.iter().map(Live::id).collect();
                assert!(ids.len() <= page, "a page of {page} held {}", ids.len());
                let ascending = ids.windows(2).all(|pair| pair[0] < pair[1]);
                assert!(
                    ascending && after.is_none_or(|after| ids.first().is_none_or(|id| *id > after)),
                    "a page after {after:?} held {ids:?}"
                );
                walked.extend(ids);
                if found.len() < page {
                    return walked;
                }
                after = walked.last().copied();
            }
        }

        /// The page as the release before listed it: every match, sorted
        /// and cut.
        fn as_before(
            &self,
            collection: &str,
            filter: Option<&Filter>,
            after: Option<ObjectId>,
            limit: usize,
        ) -> Vec<ObjectId> {
            let name = CollectionName::new(collection).unwrap();
            let mut page: Vec<ObjectId> = match filter {
                Some(filter) => self.state.select(&name, filter).unwrap(),
                None => self.state.scan(&name).collect(),
            }
            .iter()
            .map(Live::id)
            .filter(|id| after.is_none_or(|after| *id > after))
            .collect();
            page.sort_unstable();
            page.truncate(limit);
            page
        }
    }

    /// What is live, as the test knows it: `n` by collection and id.
    type Oracle = BTreeMap<(u64, u128), i64>;

    fn expected(oracle: &Oracle, collection: u64, passes: impl Fn(i64) -> bool) -> Vec<ObjectId> {
        oracle
            .iter()
            .filter(|((c, _), n)| *c == collection && passes(**n))
            .map(|((_, id), _)| oid(*id))
            .collect()
    }

    fn dense(half: i64) -> Filter {
        Filter::Gte {
            property: "n".into(),
            value: json!(half),
        }
    }

    fn sparse() -> Filter {
        Filter::Eq {
            property: "k".into(),
            value: json!("x"),
        }
    }

    /// A small deterministic generator.
    struct Random(u64);

    impl Random {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    #[test]
    fn pages_walk_every_live_row_once_in_id_order_however_the_rows_lie() {
        let mut built = Built::new(&["notes", "other"]);
        let mut oracle = Oracle::new();
        let mut random = Random(0x5eed);
        let (ids, rows) = (3_000u64, 2_000);
        let mut n = 0i64;
        for step in 0..200 {
            let mut ops = Vec::new();
            for _ in 0..rows / 100 {
                let id = u128::from(random.below(ids));
                n += 1;
                ops.push(upsert(0, id, n));
                oracle.insert((0, id), n);
                // The other collection shares ids with this one.
                if random.below(3) == 0 {
                    ops.push(upsert(1, id, -n));
                    oracle.insert((1, id), -n);
                }
            }
            for _ in 0..3 {
                let id = u128::from(random.below(ids));
                ops.push(delete(0, id));
                oracle.remove(&(0, id));
            }
            built.commit(ops);
            if step % 5 == 4 {
                built.flush();
            }
            if step % 11 == 10 {
                assert!(built.merge(), "two segments lie side by side");
            }
        }
        // And some rows written since the last flush.
        let since: Vec<Op> = (0..20)
            .map(|_| {
                let id = u128::from(random.below(ids));
                n += 1;
                oracle.insert((0, id), n);
                upsert(0, id, n)
            })
            .collect();
        built.commit(since);
        let segments = built.state.segments().count();
        assert!(
            segments >= 2 && built.state.unflushed() > 0,
            "{segments} segments and {} unflushed rows: rows lie in several places",
            built.state.unflushed()
        );
        let live = expected(&oracle, 0, |_| true).len() as u64;
        assert!(live > 1_000, "{live}");

        let half = n / 2;
        type Passes = Box<dyn Fn(i64) -> bool>;
        let cases: [(Option<Filter>, Passes); 3] = [
            (None, Box::new(|_| true)),
            (Some(dense(half)), Box::new(move |n| n >= half)),
            (Some(sparse()), Box::new(|n| k(n) == "x")),
        ];
        for (filter, passes) in &cases {
            let want = expected(&oracle, 0, passes);
            for page in [1, 2, 3, 7, 50, 1_000, 10_000] {
                assert_eq!(
                    built.walk("notes", filter.as_ref(), page),
                    want,
                    "{filter:?} at page {page}"
                );
            }
        }
        // A dense filter at a small page is met along the live rows, a
        // sparse one is sorted: both ways of taking a filtered page are
        // walked above.
        let matching = |filter: &Filter| {
            built
                .state
                .select(&CollectionName::new("notes").unwrap(), filter)
                .unwrap()
                .len() as u64
        };
        let (many, few) = (matching(&dense(half)), matching(&sparse()));
        assert!(
            many * many > live,
            "{many} of {live} are met in order at page 1"
        );
        assert!(few * few <= live, "{few} of {live} are sorted at page 1");

        // Every boundary: a page after any id, held or not, is the ids
        // after it, and a page past the end is empty.
        let all = expected(&oracle, 0, |_| true);
        let notes = CollectionName::new("notes").unwrap();
        for (at, id) in all.iter().enumerate() {
            let page = built.state.list(&notes, None, Some(*id), 3).unwrap();
            let ids: Vec<ObjectId> = page.iter().map(Live::id).collect();
            assert_eq!(ids, all[at + 1..(at + 4).min(all.len())]);
            let gone = oid(id.0.as_u128() + 1);
            if !oracle.contains_key(&(0, gone.0.as_u128())) {
                let page = built.state.list(&notes, None, Some(gone), 3).unwrap();
                let ids: Vec<ObjectId> = page.iter().map(Live::id).collect();
                assert_eq!(ids, all[at + 1..(at + 4).min(all.len())]);
            }
        }
        assert!(built
            .state
            .list(&notes, None, Some(oid(u128::MAX)), 3)
            .unwrap()
            .is_empty());
        assert!(built.state.list(&notes, None, None, 0).unwrap().is_empty());

        // And at random: as the release before listed it, which found
        // every match and sorted them.
        for _ in 0..300 {
            let filter = match random.below(3) {
                0 => None,
                1 => Some(dense(random.below(n as u64) as i64)),
                _ => Some(sparse()),
            };
            let after = match random.below(3) {
                0 => None,
                1 => Some(oid(u128::from(random.below(ids)))),
                _ => all.get(random.below(all.len() as u64) as usize).copied(),
            };
            let limit = 1 + random.below(60) as usize;
            let page = built
                .state
                .list(&notes, filter.as_ref(), after, limit)
                .unwrap();
            let ids: Vec<ObjectId> = page.iter().map(Live::id).collect();
            let before = built.as_before("notes", filter.as_ref(), after, limit);
            assert_eq!(ids, before, "{filter:?} after {after:?} limit {limit}");
            for (found, id) in page.iter().zip(&ids) {
                assert_eq!(
                    found.object.properties["n"],
                    Value::Int(oracle[&(0, id.0.as_u128())]),
                    "the live row of {id}"
                );
            }
        }
        assert_eq!(built.walk("other", None, 7), expected(&oracle, 1, |_| true));
    }

    #[test]
    fn a_row_written_again_or_deleted_is_listed_once_or_not_at_all_wherever_its_copies_lie() {
        let mut built = Built::new(&["notes"]);
        let notes = CollectionName::new("notes").unwrap();
        let listed = |built: &Built| -> Vec<(ObjectId, i64)> {
            built
                .state
                .list(&notes, None, None, 100)
                .unwrap()
                .iter()
                .map(|found| match found.object.properties["n"] {
                    Value::Int(n) => (found.id(), n),
                    ref other => panic!("{other:?}"),
                })
                .collect()
        };
        built.commit((1..=6).map(|id| upsert(0, id, id as i64)).collect());
        built.flush();
        // Written again in a second segment, and again in the memtable;
        // deleted from a segment, and deleted from the memtable.
        built.commit(vec![upsert(0, 2, 20), upsert(0, 3, 30), delete(0, 4)]);
        built.flush();
        built.commit(vec![upsert(0, 3, 300), upsert(0, 7, 7), delete(0, 5)]);
        built.commit(vec![delete(0, 7), upsert(0, 8, 8)]);
        let want = vec![
            (oid(1), 1),
            (oid(2), 20),
            (oid(3), 300),
            (oid(6), 6),
            (oid(8), 8),
        ];
        assert_eq!(listed(&built), want);
        // The first segment is mostly dead and is rewritten alone.
        assert!(built.merge());
        assert_eq!(built.state.segments().count(), 2);
        assert_eq!(listed(&built), want);
        built.flush();
        assert_eq!(listed(&built), want);
        while built.merge() {
            assert_eq!(listed(&built), want);
        }
        assert_eq!(built.state.segments().count(), 1);
        assert_eq!(listed(&built), want);
        let paged = |after| {
            built
                .state
                .list(&notes, None, Some(oid(after)), 2)
                .unwrap()
                .iter()
                .map(Live::id)
                .collect::<Vec<_>>()
        };
        assert_eq!(paged(2), [oid(3), oid(6)]);
        assert_eq!(paged(4), [oid(6), oid(8)]);
        assert_eq!(paged(8), []);
    }

    #[test]
    fn a_dropped_collection_takes_only_its_own_rows_and_a_page_of_one_never_holds_anothers() {
        let mut built = Built::new(&["a", "b", "c"]);
        let ops = (1..=9u128)
            .flat_map(|id| [upsert(0, id, 1), upsert(1, id, 2), upsert(2, id, 3)])
            .collect();
        built.commit(ops);
        built.flush();
        built.commit(vec![upsert(1, 10, 2), upsert(0, 10, 1)]);
        let ids: Vec<ObjectId> = (1..=10).map(oid).collect();
        for name in ["a", "b", "c"] {
            let want = if name == "c" { &ids[..9] } else { &ids[..] };
            assert_eq!(built.walk(name, None, 4), want, "{name}");
        }
        built.commit(vec![Op::DropCollection {
            collection: CollectionId(1),
        }]);
        assert_eq!(built.state.rows(), 19);
        assert_eq!(built.walk("a", None, 4), ids);
        assert_eq!(built.walk("c", None, 4), &ids[..9]);
        let b = CollectionName::new("b").unwrap();
        assert!(matches!(
            built.state.list(&b, None, None, 4),
            Err(StorageError::UnknownCollection(_))
        ));
        built.commit(vec![Op::CreateCollection {
            id: CollectionId(3),
            schema: Box::new(schema("b")),
        }]);
        assert!(built.state.list(&b, None, None, 4).unwrap().is_empty());
        built.commit(vec![upsert(3, 5, 5)]);
        assert_eq!(built.walk("b", None, 4), [oid(5)]);
        assert_eq!(built.state.count(&b), 1);
    }

    /// Codes prepared for a log before a flush that codes the memtable
    /// afresh, about another centre, are not stored when the log is applied
    /// after it: the row is coded about the centre it is stored beside.
    #[test]
    fn codes_prepared_before_the_memtable_is_coded_afresh_are_not_stored() {
        let index = |built: &Built| -> BinaryIndex {
            match &built.state.memtable.indexes[&CollectionId(0)].vectors {
                Some(Vectors::Binary(codes)) => codes.clone(),
                _ => panic!("coded"),
            }
        };
        let mut built = Built::new(&["notes"]);
        built.commit((0..1_100).map(|n| upsert(0, n, n as i64)).collect());
        let mut snapshot = built.state.snapshot().unwrap();
        while built.state.copy_rows(&mut snapshot, 7) {}
        // Beside the flush, more than CODE_FROM rows of other directions.
        let beside: Vec<Op> = (5_000..6_100).map(|n| upsert(0, n, -(n as i64))).collect();
        let late = vec![upsert(0, 9_999, 77)];
        built.commit(beside.clone());
        let position = built.state.position().next().unwrap();
        let log = LogSegment {
            position,
            prev: built.state.head().map(str::to_string),
            ops: late.clone(),
            answered: Vec::new(),
        };
        let prepared = built.state.prepared(&log);
        let before = index(&built).centre().to_vec();
        let file = SegmentFile {
            position: snapshot.position,
            rows: snapshot.rows_with_vectors(),
            tombstones: snapshot.tombstones.clone(),
        };
        let catalog = snapshot.catalog.clone();
        let fit = built.fit(file, &catalog);
        let at = built.state.position().next().unwrap();
        built
            .state
            .flushed(&format!("manifest-{}", at.0), at, snapshot, Some(fit), None);
        assert_ne!(index(&built).centre(), before.as_slice(), "coded afresh");
        let position = built.state.position().next().unwrap();
        let log = LogSegment {
            prev: built.state.head().map(str::to_string),
            position,
            ..log
        };
        built
            .state
            .apply_prepared(&format!("log-{}", position.0), &log, prepared)
            .unwrap();
        // The rows the flush kept, and the late one, coded with nothing
        // prepared.
        let mut plain = Built::new(&["notes"]);
        plain.commit(beside.into_iter().chain(late).collect());
        let query = [0.3f32, -0.9];
        let ranked = |codes: &BinaryIndex| {
            codes
                .estimates(
                    &query,
                    10_000,
                    None,
                    &[],
                    &vellev_index::vector::in_row_order,
                )
                .into_iter()
                .map(|(estimate, _)| estimate.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(ranked(&index(&built)), ranked(&index(&plain)));
    }

    /// Codes prepared from one state are not stored by another, however
    /// alike the two have been changed: a state replaced, as a writer's is
    /// when it reopens or restores, is coded about its own centre.
    #[test]
    fn codes_prepared_from_one_state_are_not_stored_by_another() {
        let index = |built: &Built| -> BinaryIndex {
            match &built.state.memtable.indexes[&CollectionId(0)].vectors {
                Some(Vectors::Binary(codes)) => codes.clone(),
                _ => panic!("coded"),
            }
        };
        // Alike in every count, coded about different centres.
        let mut one = Built::new(&["notes"]);
        one.commit((0..1_100).map(|n| upsert(0, n, n as i64)).collect());
        let mut other = Built::new(&["notes"]);
        other.commit((0..1_100).map(|n| upsert(0, n, -(n as i64) - 7)).collect());
        assert_ne!(index(&one).centre(), index(&other).centre());
        let late = vec![upsert(0, 9_999, 77)];
        let position = other.state.position().next().unwrap();
        let log = LogSegment {
            position,
            prev: other.state.head().map(str::to_string),
            ops: late.clone(),
            answered: Vec::new(),
        };
        let prepared = one.state.prepared(&log);
        other
            .state
            .apply_prepared(&format!("log-{}", position.0), &log, prepared)
            .unwrap();
        let mut plain = Built::new(&["notes"]);
        plain.commit((0..1_100).map(|n| upsert(0, n, -(n as i64) - 7)).collect());
        plain.commit(late);
        let query = [0.3f32, -0.9];
        let ranked = |codes: &BinaryIndex| {
            codes
                .estimates(
                    &query,
                    10_000,
                    None,
                    &[],
                    &vellev_index::vector::in_row_order,
                )
                .into_iter()
                .map(|(estimate, _)| estimate.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(ranked(&index(&other)), ranked(&index(&plain)));
    }
}
