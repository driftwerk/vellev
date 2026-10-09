use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use roaring::RoaringBitmap;
use vellev_core::{Collection, CollectionName, Object, ObjectId, RequestId, Value};
use vellev_index::clustered::{ClusteredIndex, Unfit};
use vellev_index::filter::{Filter, Predicate, RowIndex};
use vellev_index::text::TextIndex;
use vellev_index::vector::{BinaryIndex, Ties, VectorIndex};

use crate::answers::Answers;
use crate::catalog::Catalog;
use crate::column::Column;
use crate::coordinator::LogPosition;
use crate::format::{
    Answer, CollectionId, LogSegment, Manifest, Op, Row, SegmentFile, SegmentIndex, SegmentMeta,
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
    pub fn of(catalog: &Catalog, rows: &[Row], tombstones: usize, clustered_above: usize) -> Self {
        Self {
            rows: rows.len(),
            tombstones,
            clustered: clustered(catalog, rows, clustered_above),
            vectors: rows.iter().any(|row| row.object.vector.is_some()),
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
}

impl Loaded {
    pub fn fresh(file: SegmentFile, vectors: Column, stored: Stored) -> Self {
        Self {
            file: Arc::new(file),
            vectors: Arc::new(vectors),
            indexes: None,
            stored,
        }
    }

    /// The segment with its indexes built, under the collections of
    /// `catalog`, so that building them need not wait for other segments.
    pub fn indexed_now(self, key: &str, catalog: &Catalog) -> Result<Self> {
        let indexes = self.indexed(key, catalog)?;
        Ok(Self {
            indexes: Some(indexes),
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
            Arc::new(indexed(
                catalog,
                &self.file.rows,
                column,
                stored,
                unused,
                None,
            ))
        };
        Ok(match &self.indexes {
            Some(indexes) if self.lacking(indexes, catalog) => built(&stored(key, indexes).into()),
            Some(indexes) => indexes.clone(),
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
    live: HashMap<RowKey, Location>,
    corpora: HashMap<CollectionId, Corpus>,
    /// The logs applied since the last checkpoint, oldest first: what is
    /// in the memtable and in no segment.
    since: Vec<String>,
    answers: Answers,
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
    pub rows: Vec<Row>,
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

#[cfg(test)]
impl Snapshot {
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
    vector_of: impl Fn(usize) -> Option<&'a [f32]>,
    stored: &Stored,
    unused: impl Fn(CollectionId, Unfit),
    clustered_above: Option<usize>,
) -> Indexes {
    let mut indexes = Indexes::new();
    let mut vectors: HashMap<CollectionId, Vec<(u32, &[f32])>> = HashMap::new();
    for (row, Row { collection, object }) in rows.iter().enumerate() {
        if let Some(schema) = catalog.schema(*collection) {
            let index = indexes.entry(*collection).or_default();
            index.insert(schema, row as u32, object);
            if let Some(vector) = vector_of(row) {
                vectors
                    .entry(*collection)
                    .or_default()
                    .push((row as u32, vector));
            }
        }
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
    indexes
}

/// Whether indexing `rows` with `clustered_above` builds any clustered
/// index, which is to say whether the segment has an index file.
pub(crate) fn clustered(catalog: &Catalog, rows: &[Row], clustered_above: usize) -> bool {
    let mut counts: HashMap<CollectionId, usize> = HashMap::new();
    for Row { collection, object } in rows {
        if object.vector.is_some() && catalog.schema(*collection).is_some() {
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

impl State {
    pub fn position(&self) -> LogPosition {
        self.position
    }

    /// About how many bytes the state takes in memory: its rows, what
    /// indexes them, and whatever vectors are held in memory rather than
    /// read in place from local files. Rough, and cheap to ask: each part
    /// is counted as it is made.
    pub fn resident(&self) -> usize {
        let segments: usize = self.segments.iter().map(|segment| segment.resident).sum();
        segments + self.memtable.bytes
    }

    /// How many rows are live. What a writer does in memory as it commits
    /// a checkpoint takes about as long as there are of them.
    pub fn rows(&self) -> usize {
        self.live.len()
    }

    /// How many rows have been written since the last checkpoint, counting
    /// those since replaced: what a flush would move out of memory.
    pub fn unflushed(&self) -> usize {
        self.memtable.rows.len()
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
            .map(|location| match *location {
                Location::Memtable(row) => Live {
                    object: &self.memtable.rows[row].object,
                    vector: self.memtable.rows[row].object.vector.as_deref(),
                },
                Location::Segment(segment, row) => Live {
                    object: &self.segments[segment].file.rows[row].object,
                    vector: self.segments[segment].vectors.get(row),
                },
            })
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
    /// Each page selects every matching row and sorts them, so walking a
    /// whole collection costs its size once per page.
    pub fn list(
        &self,
        collection: &CollectionName,
        filter: Option<&Filter>,
        after: Option<ObjectId>,
        limit: usize,
    ) -> Result<Vec<Live<'_>>> {
        let mut page = match filter {
            Some(filter) => self.select(collection, filter)?,
            None => {
                self.catalog.known(collection)?;
                self.scan(collection).collect()
            }
        };
        page.retain(|found| after.is_none_or(|after| found.id() > after));
        page.sort_unstable_by_key(Live::id);
        page.truncate(limit);
        Ok(page)
    }

    /// Every layer, oldest first.
    pub(crate) fn layers(&self) -> impl Iterator<Item = Layer<'_>> {
        let memtable = Layer {
            rows: &self.memtable.rows,
            dead: &self.memtable.dead,
            deaths: self.memtable.deaths,
            indexes: &self.memtable.indexes,
            vectors: None,
        };
        self.segments
            .iter()
            .map(|segment| Layer {
                rows: &segment.file.rows,
                dead: &segment.dead,
                deaths: segment.deaths,
                indexes: &segment.indexes,
                vectors: Some(&segment.vectors),
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
            };
            (segment.meta.key.as_str(), loaded)
        })
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
            resident: segment_bytes(&file.rows, &vectors),
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
    /// on error, none of it.
    pub(crate) fn apply(&mut self, key: &str, log: &LogSegment) -> Result<()> {
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
        for op in &log.ops {
            match op {
                Op::CreateCollection { .. } | Op::AddProperty { .. } => {}
                Op::DropCollection { collection } => {
                    let dropped: Vec<RowKey> = self
                        .live
                        .keys()
                        .filter(|(c, _)| c == collection)
                        .copied()
                        .collect();
                    dropped.into_iter().for_each(|key| {
                        self.forget(key);
                    });
                    self.memtable.tombstones.retain(|(c, _)| c != collection);
                    // Nothing can ask for a dropped collection's rows;
                    // this only frees the memory.
                    self.memtable.indexes.remove(collection);
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
                    // rows as a segment may number. It has no vector index:
                    // its rows change, and are few enough to score one by
                    // one.
                    //
                    // A collection dropped later in the same commit is
                    // already gone from the catalog; its rows go unindexed.
                    let index = self.catalog.schema(*collection).map(|schema| {
                        let index = self.memtable.indexes.entry(*collection).or_default();
                        index.insert(schema, row as u32, object);
                        &*index
                    });
                    self.corpora
                        .entry(*collection)
                        .or_default()
                        .count(index, row as u32, true);
                    let row = Row {
                        collection: *collection,
                        object: object.clone(),
                    };
                    self.memtable.bytes += row_bytes(&row);
                    self.memtable.rows.push(row);
                    self.memtable.dead.push(false);
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
    pub(crate) fn merged(&self, merge: &Merge, position: LogPosition) -> Option<SegmentFile> {
        if !self.unchanged(merge) {
            return None;
        }
        let run = &self.segments[merge.run.clone()];
        let rows: Vec<Row> = run
            .iter()
            .flat_map(|segment| {
                (0..segment.file.rows.len())
                    .filter(|row| !segment.dead[*row])
                    .map(|row| {
                        let mut copy = segment.file.rows[row].clone();
                        copy.object.vector = segment.vectors.get(row).map(<[f32]>::to_vec);
                        copy
                    })
            })
            .collect();
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
        Some(SegmentFile {
            position,
            rows,
            tombstones,
        })
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
                resident: segment_bytes(&file.rows, &vectors),
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
        Snapshot {
            position: self.position,
            checkpoint: self.checkpoint.clone(),
            catalog: self.catalog.clone(),
            rows: Vec::with_capacity(at.len()),
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
        let rows = &self.memtable.rows;
        snapshot
            .rows
            .extend(snapshot.at[from..to].iter().map(|row| rows[*row].clone()));
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
                resident: segment_bytes(&file.rows, &vectors),
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
        let discarded = self.drop_flushed_rows(snapshot.held);
        self.since.drain(..snapshot.logs);
        self.position = position;
        self.head = Some(manifest_key.to_string());
        self.checkpoint = Some(manifest_key.to_string());
        discarded
    }

    /// Take out the memtable's first `held` rows, every one of them dead,
    /// renumbering and indexing afresh those after them. Returns what was
    /// taken out, to be freed without holding the state.
    fn drop_flushed_rows(&mut self, held: usize) -> Discarded {
        let Memtable {
            rows,
            dead,
            deaths,
            indexes,
            bytes,
            ..
        } = &mut self.memtable;
        let discarded = Discarded {
            _rows: rows.drain(..held).collect(),
            _indexes: std::mem::take(indexes),
        };
        *bytes = rows.iter().map(row_bytes).sum();
        dead.drain(..held);
        *deaths -= held;
        for (row, Row { collection, object }) in rows.iter().enumerate() {
            if let Some(schema) = self.catalog.schema(*collection) {
                indexes
                    .entry(*collection)
                    .or_default()
                    .insert(schema, row as u32, object);
            }
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
    ) -> Result<Self> {
        let Some((_, first)) = logs.first() else {
            return Ok(self);
        };
        let mut state = State {
            position: after,
            head: first.prev.clone(),
            ..self
        };
        for (log_key, log) in logs {
            state.apply(log_key, log)?;
        }
        if state.position.next() != Some(position) {
            return Err(StorageError::corrupt(
                key,
                "its replayed logs do not end just before it",
            ));
        }
        Ok(State {
            position,
            head: Some(key.to_string()),
            checkpoint: Some(key.to_string()),
            ..state
        })
    }
}
