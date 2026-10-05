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
    commits_since_checkpoint: usize,
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

    pub fn collections(&self) -> impl Iterator<Item = &Collection> {
        self.catalog.collections().map(|(_, schema)| schema)
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

    pub(crate) fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    pub(crate) fn segment_files(&self) -> impl Iterator<Item = (&str, &Arc<SegmentFile>)> {
        self.segments
            .iter()
            .map(|segment| (segment.meta.key.as_str(), &segment.file))
    }

    /// Whether recovery would have to replay log on top of the last checkpoint.
    pub(crate) fn has_commits_since_checkpoint(&self) -> bool {
        self.commits_since_checkpoint > 0
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
    ) -> Result<Self> {
        let catalog = Catalog::restore(manifest.next_collection_id, manifest.collections)
            .map_err(|reason| StorageError::corrupt(key, reason))?;
        let state = State {
            position: manifest.position,
            head: Some(key.to_string()),
            catalog,
            ..State::default()
        };
        manifest
            .segments
            .into_iter()
            .zip(files)
            .try_fold(state, |mut state, (meta, file)| {
                state.attach(meta, file)?;
                Ok(state)
            })
    }

    fn attach(&mut self, meta: SegmentMeta, file: Arc<SegmentFile>) -> Result<()> {
        let corrupt = |reason: &str| StorageError::corrupt(&meta.key, reason);
        if (file.rows.len() as u64, file.tombstones.len() as u64) != (meta.rows, meta.tombstones) {
            return Err(corrupt("row or tombstone count differs from the manifest"));
        }
        let index = self.segments.len();
        for key in &file.tombstones {
            self.forget(*key);
        }
        let mut dead = Vec::with_capacity(file.rows.len());
        for (row, Row { collection, object }) in file.rows.iter().enumerate() {
            let live = self.catalog.contains(*collection);
            if live {
                match self
                    .live
                    .insert((*collection, object.id), Location::Segment(index, row))
                {
                    Some(Location::Segment(segment, _)) if segment == index => {
                        return Err(corrupt("a key appears twice"));
                    }
                    Some(shadowed) => self.mark_dead(shadowed),
                    None => {}
                }
            }
            dead.push(!live);
        }
        self.segments.push(Segment { meta, file, dead });
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
        self.commits_since_checkpoint += 1;
        Ok(())
    }

    /// What a checkpoint at `position` writes: a segment holding the
    /// memtable's live rows and tombstones, if it has any, and the manifest
    /// listing `segment_key` after the existing segments.
    pub(crate) fn checkpoint(
        &self,
        position: LogPosition,
        segment_key: &str,
    ) -> (Option<SegmentFile>, Manifest) {
        let rows: Vec<Row> = live_rows(&self.memtable.rows, &self.memtable.dead)
            .cloned()
            .collect();
        let tombstones = self.memtable.tombstones.clone();
        let segment = (!rows.is_empty() || !tombstones.is_empty()).then_some(SegmentFile {
            position,
            rows,
            tombstones,
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

    /// Adopt a committed checkpoint made from this state by [`Self::checkpoint`]:
    /// the memtable's rows now live in `segment`, the manifest's last segment.
    pub(crate) fn checkpointed(
        &mut self,
        manifest_key: &str,
        manifest: &Manifest,
        segment: Option<SegmentFile>,
    ) -> Result<()> {
        self.live
            .retain(|_, location| !matches!(location, Location::Memtable(_)));
        self.memtable = Memtable::default();
        if let (Some(file), Some(meta)) = (segment, manifest.segments.last()) {
            self.attach(meta.clone(), Arc::new(file))?;
        }
        self.position = manifest.position;
        self.head = Some(manifest_key.to_string());
        self.commits_since_checkpoint = 0;
        Ok(())
    }
}
