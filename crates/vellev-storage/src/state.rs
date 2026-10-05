use std::collections::HashMap;
use std::sync::Arc;

use vellev_core::{Collection, CollectionName, Object, ObjectId};

use crate::catalog::Catalog;
use crate::coordinator::LogPosition;
use crate::format::{CollectionId, LogSegment, Manifest, Op, Row, SegmentFile, SegmentMeta};
use crate::{Result, StorageError};

type RowKey = (CollectionId, ObjectId);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Location {
    Memtable(usize),
    Segment(usize, usize),
}

#[derive(Debug, Clone)]
struct Segment {
    meta: SegmentMeta,
    file: Arc<SegmentFile>,
    dead: Vec<bool>,
}

#[derive(Debug, Clone, Default)]
struct Memtable {
    rows: Vec<Row>,
    dead: Vec<bool>,
    tombstones: Vec<RowKey>,
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

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub fn collection(&self, name: &CollectionName) -> Option<&Collection> {
        self.catalog.get(name).map(|(_, schema)| schema)
    }

    pub fn get(&self, collection: &CollectionName, id: ObjectId) -> Option<&Object> {
        let (collection, _) = self.catalog.get(collection)?;
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

    pub(crate) fn segment_files(&self) -> impl Iterator<Item = (&str, &Arc<SegmentFile>)> {
        self.segments
            .iter()
            .map(|segment| (segment.meta.key.as_str(), &segment.file))
    }

    pub(crate) fn has_unflushed_writes(&self) -> bool {
        !self.memtable.rows.is_empty() || !self.memtable.tombstones.is_empty()
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

    /// Rebuild the partition as of a checkpoint. `files` are the manifest's
    /// segments in its order, oldest first; a newer segment's rows and
    /// tombstones shadow an older one's.
    pub(crate) fn from_manifest(
        key: &str,
        manifest: Manifest,
        files: Vec<Arc<SegmentFile>>,
    ) -> Self {
        let mut state = State {
            position: manifest.position,
            head: Some(key.to_string()),
            catalog: Catalog::restore(manifest.next_collection_id, manifest.collections),
            ..State::default()
        };
        for (meta, file) in manifest.segments.into_iter().zip(files) {
            state.attach(meta, file);
        }
        state
    }

    fn attach(&mut self, meta: SegmentMeta, file: Arc<SegmentFile>) {
        let index = self.segments.len();
        for key in &file.tombstones {
            self.forget(*key);
        }
        let dead = file
            .rows
            .iter()
            .enumerate()
            .map(|(row, Row { collection, object })| {
                if !self.catalog.contains(*collection) {
                    return true;
                }
                let key = (*collection, object.id);
                if let Some(shadowed) = self.live.insert(key, Location::Segment(index, row)) {
                    self.mark_dead(shadowed);
                }
                false
            })
            .collect();
        self.segments.push(Segment { meta, file, dead });
    }

    /// Apply one committed log segment. It must be the commit made directly
    /// on top of this state.
    pub(crate) fn apply(&mut self, key: &str, log: &LogSegment) -> Result<()> {
        let expected = LogPosition(self.position.0 + 1);
        if log.position != expected || log.prev.as_deref() != self.head() {
            return Err(StorageError::corrupt(
                key,
                format!(
                    "commit {:?} on {:?} does not follow {:?} at {:?}",
                    log.position, log.prev, self.head, self.position
                ),
            ));
        }
        for op in &log.ops {
            self.catalog
                .apply(op)
                .map_err(|reason| StorageError::corrupt(key, reason))?;
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
                }
                Op::Upsert { collection, object } => {
                    let key = (*collection, object.id);
                    self.forget(key);
                    self.memtable
                        .tombstones
                        .retain(|tombstone| *tombstone != key);
                    self.live
                        .insert(key, Location::Memtable(self.memtable.rows.len()));
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
        Ok(())
    }

    /// What a checkpoint at `position` writes: the segment holding the
    /// memtable's live rows and tombstones, if there are any, and the
    /// manifest listing `segment_key` after the existing segments.
    pub(crate) fn checkpoint(
        &self,
        position: LogPosition,
        segment_key: &str,
    ) -> (Option<SegmentFile>, Manifest) {
        let segment = self.has_unflushed_writes().then(|| SegmentFile {
            position,
            rows: live_rows(&self.memtable.rows, &self.memtable.dead)
                .cloned()
                .collect(),
            tombstones: self.memtable.tombstones.clone(),
        });
        let flushed = segment.as_ref().map(|file| SegmentMeta {
            key: segment_key.to_string(),
            rows: file.rows.len() as u64,
            tombstones: file.tombstones.len() as u64,
        });
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

    /// Adopt a committed checkpoint made from this state: the memtable's rows
    /// now live in `segment`.
    pub(crate) fn checkpointed(
        &mut self,
        manifest_key: &str,
        manifest: &Manifest,
        segment: Option<SegmentFile>,
    ) {
        self.live
            .retain(|_, location| !matches!(location, Location::Memtable(_)));
        self.memtable = Memtable::default();
        if let (Some(file), Some(meta)) = (segment, manifest.segments.last()) {
            self.attach(meta.clone(), Arc::new(file));
        }
        self.position = manifest.position;
        self.head = Some(manifest_key.to_string());
    }
}
