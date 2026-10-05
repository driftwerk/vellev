use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use vellev_core::{Collection, CollectionName, Object, ObjectId};
use vellev_index::filter::{Filter, Predicate, RowIndex};

use crate::catalog::Catalog;
use crate::coordinator::LogPosition;
use crate::format::{CollectionId, LogSegment, Manifest, Op, Row, SegmentFile, SegmentMeta};
use crate::{Result, StorageError};

type RowKey = (CollectionId, ObjectId);

/// A layer's filter indexes, one per collection with rows in it. Like
/// liveness they are derived from the rows and never stored.
pub(crate) type Indexes = HashMap<CollectionId, RowIndex>;

/// A segment as loaded: its file and the indexes built over it, shared by
/// every state that holds the segment.
pub(crate) type Loaded = (Arc<SegmentFile>, Option<Arc<Indexes>>);

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
    indexes: Arc<Indexes>,
    dead: Vec<bool>,
}

#[derive(Debug, Clone, Default)]
struct Memtable {
    rows: Vec<Row>,
    dead: Vec<bool>,
    tombstones: Vec<RowKey>,
    indexes: Indexes,
}

/// A partition in memory as of one position: flushed segments, the rows
/// written since the last checkpoint, and which row is the live one for each
/// key. Liveness is derived on load and never stored; it is a function of the
/// committed files.
#[derive(Debug, Clone, Default)]
pub struct State {
    position: LogPosition,
    head: Option<String>,
    catalog: Catalog,
    segments: Vec<Segment>,
    memtable: Memtable,
    live: HashMap<RowKey, Location>,
    has_commits_since_checkpoint: bool,
}

fn live_rows<'a>(rows: &'a [Row], dead: &'a [bool]) -> impl Iterator<Item = &'a Row> {
    rows.iter()
        .zip(dead)
        .filter(|(_, dead)| !**dead)
        .map(|(row, _)| row)
}

impl State {
    pub fn position(&self) -> LogPosition {
        self.position
    }

    pub fn head(&self) -> Option<&str> {
        self.head.as_deref()
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

    pub fn get(&self, collection: &CollectionName, id: ObjectId) -> Option<&Object> {
        self.row(self.catalog.get(collection)?.0, id)
    }

    pub(crate) fn row(&self, collection: CollectionId, id: ObjectId) -> Option<&Object> {
        self.live
            .get(&(collection, id))
            .map(|location| match *location {
                Location::Memtable(row) => &self.memtable.rows[row].object,
                Location::Segment(segment, row) => &self.segments[segment].file.rows[row].object,
            })
    }

    /// Every live object in the collection, in no particular order.
    pub fn scan<'a>(&'a self, collection: &CollectionName) -> impl Iterator<Item = &'a Object> {
        let wanted = self.catalog.get(collection).map(|(id, _)| id);
        self.segments
            .iter()
            .flat_map(|segment| live_rows(&segment.file.rows, &segment.dead))
            .chain(live_rows(&self.memtable.rows, &self.memtable.dead))
            .filter(move |row| Some(row.collection) == wanted)
            .map(|row| &row.object)
    }

    pub fn count(&self, collection: &CollectionName) -> usize {
        self.scan(collection).count()
    }

    /// Every live object in the collection that passes `filter`, in no
    /// particular order.
    pub fn select(&self, collection: &CollectionName, filter: &Filter) -> Result<Vec<&Object>> {
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
    ) -> Result<Vec<&Object>> {
        let mut page = match filter {
            Some(filter) => self.select(collection, filter)?,
            None => {
                self.catalog.known(collection)?;
                self.scan(collection).collect()
            }
        };
        page.retain(|object| after.is_none_or(|after| object.id > after));
        page.sort_unstable_by_key(|object| object.id);
        page.truncate(limit);
        Ok(page)
    }

    pub(crate) fn matching<'a, 'p>(
        &'a self,
        collection: CollectionId,
        predicate: &'p Predicate,
    ) -> impl Iterator<Item = &'a Object> + 'p
    where
        'a: 'p,
    {
        let layer = move |indexes: &'a Indexes, rows: &'a [Row], dead: &'a [bool]| {
            indexes
                .get(&collection)
                .map(|index| predicate.select(index))
                .unwrap_or_default()
                .into_iter()
                .map(|row| row as usize)
                .filter(move |row| !dead[*row])
                .map(move |row| &rows[row].object)
        };
        self.segments
            .iter()
            .flat_map(move |segment| layer(&segment.indexes, &segment.file.rows, &segment.dead))
            .chain(layer(
                &self.memtable.indexes,
                &self.memtable.rows,
                &self.memtable.dead,
            ))
    }

    pub(crate) fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub(crate) fn loaded_segments(&self) -> impl Iterator<Item = (&str, Loaded)> {
        self.segments.iter().map(|segment| {
            let loaded = (segment.file.clone(), Some(segment.indexes.clone()));
            (segment.meta.key.as_str(), loaded)
        })
    }

    /// Whether recovery would have to replay log on top of the last checkpoint.
    pub(crate) fn has_commits_since_checkpoint(&self) -> bool {
        self.has_commits_since_checkpoint
    }

    fn mark_dead(&mut self, location: Location) {
        match location {
            Location::Memtable(row) => self.memtable.dead[row] = true,
            Location::Segment(segment, row) => self.segments[segment].dead[row] = true,
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
            catalog,
            ..State::default()
        };
        manifest.segments.into_iter().zip(segments).try_fold(
            state,
            |mut state, (meta, (file, indexes))| {
                state.attach(meta, file, indexes)?;
                Ok(state)
            },
        )
    }

    /// Add a segment as the newest. The file is checked before anything is
    /// changed, so on error the state is as it was.
    fn attach(
        &mut self,
        meta: SegmentMeta,
        file: Arc<SegmentFile>,
        indexes: Option<Arc<Indexes>>,
    ) -> Result<()> {
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
        let dead = file
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
                }
                !live
            })
            .collect();
        let indexes = indexes.unwrap_or_else(|| Arc::new(self.indexed(&file.rows)));
        self.segments.push(Segment {
            meta,
            file,
            indexes,
            dead,
        });
        Ok(())
    }

    /// Index `rows` for filtering. An index built under an older catalog
    /// stays right: a collection's filterable properties never change, and a
    /// collection created since has no rows in an older layer.
    pub(crate) fn indexed(&self, rows: &[Row]) -> Indexes {
        let mut indexes = Indexes::new();
        for (row, Row { collection, object }) in rows.iter().enumerate() {
            if let Some(schema) = self.catalog.schema(*collection) {
                indexes
                    .entry(*collection)
                    .or_default()
                    .insert(schema, row as u32, object);
            }
        }
        indexes
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
                }
                Op::Upsert { collection, object } => {
                    let key = (*collection, object.id);
                    self.forget(key);
                    self.memtable
                        .tombstones
                        .retain(|tombstone| *tombstone != key);
                    let row = self.memtable.rows.len();
                    self.live.insert(key, Location::Memtable(row));
                    if let Some(schema) = self.catalog.schema(*collection) {
                        let index = self.memtable.indexes.entry(*collection).or_default();
                        // A memtable is flushed long before it holds as
                        // many rows as a segment may number.
                        index.insert(schema, row as u32, object);
                    }
                    self.memtable.rows.push(Row {
                        collection: *collection,
                        object: object.clone(),
                    });
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
        self.position = log.position;
        self.head = Some(key.to_string());
        self.has_commits_since_checkpoint = true;
        Ok(())
    }

    /// What a checkpoint at `position` writes: a segment under `segment_key`
    /// holding the memtable's live rows and tombstones, if it has any, and a
    /// manifest listing it after the existing segments.
    pub(crate) fn checkpoint(
        &self,
        position: LogPosition,
        segment_key: &str,
    ) -> (Option<(SegmentMeta, SegmentFile)>, Manifest) {
        let rows: Vec<Row> = live_rows(&self.memtable.rows, &self.memtable.dead)
            .cloned()
            .collect();
        let tombstones = self.memtable.tombstones.clone();
        let segment = (!rows.is_empty() || !tombstones.is_empty()).then(|| {
            let meta = SegmentMeta {
                key: segment_key.to_string(),
                rows: rows.len() as u64,
                tombstones: tombstones.len() as u64,
            };
            (
                meta,
                SegmentFile {
                    position,
                    rows,
                    tombstones,
                },
            )
        });
        let flushed = segment.iter().map(|(meta, _)| meta.clone());
        let manifest = Manifest {
            position,
            next_collection_id: self.catalog.next_id(),
            collections: self
                .catalog
                .collections()
                .map(|(id, schema)| (id, schema.clone()))
                .collect(),
            segments: self
                .segments
                .iter()
                .map(|s| s.meta.clone())
                .chain(flushed)
                .collect(),
        };
        (segment, manifest)
    }

    /// Adopt a committed checkpoint made from this state by [`Self::checkpoint`].
    /// Attaching the segment repoints every live memtable row at it, which
    /// leaves the memtable with nothing live to lose.
    pub(crate) fn checkpointed(
        &mut self,
        manifest_key: &str,
        position: LogPosition,
        segment: Option<(SegmentMeta, Loaded)>,
    ) -> Result<()> {
        if let Some((meta, (file, indexes))) = segment {
            self.attach(meta, file, indexes)?;
        }
        self.memtable = Memtable::default();
        self.position = position;
        self.head = Some(manifest_key.to_string());
        self.has_commits_since_checkpoint = false;
        Ok(())
    }
}
