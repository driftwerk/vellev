use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::{join_all, try_join_all};
use tokio::sync::Mutex;
use vellev_core::Embedding;

use crate::blobs::{is_plain, Author, Blobs, Layout, Listed};
use crate::coordinator::{
    Commit, Committed, Coordinator, Generation, Lease, LogPosition, ManifestRef, PartitionKey,
};
use crate::draft::{Draft, Wanted};
use crate::format::{
    decode, decode_head, encode, Head, Kind, LogSegment, Manifest, Op, SegmentFile, SegmentIndex,
    SegmentMeta,
};
use crate::mutation::{Mutation, Receipt, Vectorize};
use crate::state::{self, Loaded, Merging, State, Stored};
use crate::{Result, StorageError};

/// How many times a commit whose outcome is unknown is tried. Repeating is
/// safe because the coordinator accepts an identical commit twice.
const COMMIT_ATTEMPTS: u32 = 3;

/// A segment's frame and its index file's.
type Frames = (Bytes, Option<Bytes>);
/// A segment ready to be written and attached.
type Indexed = (SegmentMeta, Loaded, Frames);
const COMMIT_RETRY_DELAY: Duration = Duration::from_millis(50);

/// How long a commit waits for the vectors it needs made.
const EMBEDDING_TIMEOUT: Duration = Duration::from_secs(300);

/// Vectors by the text they were made from and the way it was embedded.
pub type Embedded = HashMap<(Embedding, String), Vec<f32>>;

/// A partition's committed data in memory. It only ever moves forward, and
/// only through its [`Follower`] or [`Writer`].
#[derive(Debug)]
pub struct Partition {
    key: PartitionKey,
    blobs: Arc<dyn Blobs>,
    state: RwLock<State>,
    /// Serialises catch-up, and remembers the newest coordinator generation
    /// seen so an older reading of the committed state is recognised as stale.
    seen: Mutex<Generation>,
}

/// The committed commits between a state and a newer head, oldest first, and
/// the checkpoint they start from when the chain reaches one before it
/// reaches the state.
struct Tail {
    checkpoint: Option<(String, Manifest)>,
    logs: Vec<(String, LogSegment)>,
}

impl Partition {
    pub fn key(&self) -> &PartitionKey {
        &self.key
    }

    /// Run `f` against a consistent view of the partition. The lock is a
    /// blocking one and is never held across an await.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.state.read().expect("partition lock"))
    }

    async fn load(key: PartitionKey, blobs: Arc<dyn Blobs>, committed: &Committed) -> Result<Self> {
        let partition = Self {
            key,
            blobs,
            state: RwLock::new(State::default()),
            seen: Mutex::new(Generation(0)),
        };
        partition.advance(committed).await?;
        Ok(partition)
    }

    /// Fetch a file named by data rather than built by [`Layout`], which must
    /// be a plain key of this partition's.
    async fn fetch(&self, key: &str) -> Result<Bytes> {
        if !(is_plain(key) && key.starts_with(&Layout(&self.key).prefix())) {
            return Err(StorageError::corrupt(
                key,
                "the key is not one of this partition's",
            ));
        }
        self.blobs.get(key).await
    }

    /// Generations only grow, so a reading from an older generation than one
    /// already applied is out of date whatever its position, and so is a
    /// lower position from the same generation. A reading that is not ahead
    /// and not stale means the history this copy holds was replaced.
    async fn advance(&self, committed: &Committed) -> Result<LogPosition> {
        let mut seen = self.seen.lock().await;
        let (position, head) = self.read(State::at);
        let target = committed.manifest.as_ref().map(|head| head.0.as_str());
        let stale = committed.generation < *seen
            || (committed.generation == *seen && committed.log_position < position);
        if stale {
            return Ok(position);
        }
        let lost = || match (&head, target) {
            (Some(_), _) => StorageError::Diverged { head: head.clone() },
            (None, Some(target)) => {
                StorageError::corrupt(target, "its history does not reach the start of the log")
            }
            (None, None) => StorageError::corrupt(
                &Layout(&self.key).prefix(),
                format!("{:?} is committed with no head", committed.log_position),
            ),
        };
        if committed.log_position == position && target == head.as_deref() {
            *seen = committed.generation;
            return Ok(position);
        }
        if committed.log_position <= position {
            return Err(lost());
        }
        let tail = self
            .tail(committed, position, head.as_deref())
            .await?
            .ok_or_else(lost)?;
        *seen = committed.generation;
        match tail.checkpoint {
            Some((key, manifest)) => {
                let mut state = self.restore(&key, manifest).await?;
                tail.logs
                    .iter()
                    .try_for_each(|(key, log)| state.apply(key, log))?;
                *self.state.write().expect("partition lock") = state;
            }
            None => {
                let mut state = self.state.write().expect("partition lock");
                tail.logs
                    .iter()
                    .try_for_each(|(key, log)| state.apply(key, log))?;
            }
        }
        Ok(committed.log_position)
    }

    /// Walk back from the committed head until reaching the head this copy
    /// is at, or a checkpoint; `None` when the chain leads to neither. Each
    /// file must sit exactly one position below the one that named it, so the
    /// walk is bounded and cannot loop.
    async fn tail(
        &self,
        committed: &Committed,
        position: LogPosition,
        known: Option<&str>,
    ) -> Result<Option<Tail>> {
        let mut logs = Vec::new();
        let mut cursor = committed.manifest.as_ref().map(|head| head.0.clone());
        let mut expected = committed.log_position;
        let checkpoint = loop {
            let key = match cursor {
                reached if reached.as_deref() == known && expected == position => break None,
                Some(key) if expected > position => key,
                _ => return Ok(None),
            };
            let head = decode_head(&key, &self.fetch(&key).await?)?;
            let found = match &head {
                Head::Log(log) => log.position,
                Head::Manifest(manifest) => manifest.position,
            };
            if found != expected {
                let reason = format!("it is at {found:?} where {expected:?} was committed");
                return Err(StorageError::corrupt(&key, reason));
            }
            match head {
                Head::Manifest(manifest) => break Some((key, manifest)),
                Head::Log(log) => {
                    expected = LogPosition(expected.0 - 1);
                    cursor = log.prev.clone();
                    logs.push((key, log));
                }
            }
        };
        logs.reverse();
        Ok(Some(Tail { checkpoint, logs }))
    }

    async fn restore(&self, key: &str, manifest: Manifest) -> Result<State> {
        let held: HashMap<String, Loaded> = self.read(|state| {
            state
                .loaded_segments()
                .map(|(key, loaded)| (key.to_string(), loaded))
                .collect()
        });
        let segments = try_join_all(manifest.segments.iter().map(|meta| {
            let held = held.get(&meta.key).cloned();
            async move {
                if let Some(loaded) = held {
                    return Ok::<_, StorageError>(loaded);
                }
                let file = decode(&meta.key, Kind::Segment, &self.fetch(&meta.key).await?)?;
                let stored = match &meta.index {
                    Some(index) => self.stored(index).await,
                    None => Stored::default(),
                };
                Ok(Loaded::fresh(file, stored))
            }
        }))
        .await?;
        let key = key.to_string();
        blocking(move || State::from_manifest(&key, manifest, segments)).await?
    }

    /// The clustered indexes kept under `key`, or none if they cannot be
    /// read, in which case they are built again.
    async fn stored(&self, key: &str) -> Stored {
        let read = async { decode::<SegmentIndex>(key, Kind::Index, &self.fetch(key).await?) };
        match read.await {
            Ok(index) => index.into(),
            Err(error) => {
                tracing::warn!(key, %error, "building a segment's indexes afresh");
                Stored::default()
            }
        }
    }
}

/// Run `work`, which takes long enough to hold up the async threads, on
/// one of its own.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| StorageError::Indexing(error.to_string()))
}

/// A read-only copy of a partition that follows what the writer commits.
#[derive(Debug, Clone)]
pub struct Follower(Arc<Partition>);

impl Follower {
    /// Load the partition exactly as committed.
    pub async fn open(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        committed: &Committed,
    ) -> Result<Self> {
        Partition::load(key, blobs, committed)
            .await
            .map(Arc::new)
            .map(Self)
    }

    pub fn partition(&self) -> &Arc<Partition> {
        &self.0
    }

    /// Bring this copy up to `committed`, reading only files reachable from
    /// its head, and return the position it is now at. A reading of the
    /// committed state older than one already applied is ignored.
    pub async fn catch_up(&self, committed: &Committed) -> Result<LogPosition> {
        self.0.advance(committed).await
    }
}

/// The one node allowed to change a partition, for as long as its lease's
/// generation is current. Nothing it writes is visible, here or anywhere,
/// until the coordinator has accepted the commit.
///
/// A file is named by its position and its author, and is never rewritten.
/// The writer therefore never attempts a position twice: from the first
/// upload of an attempt until it is applied in memory the writer is not
/// ready, and it becomes ready again only on success. An error, or a caller
/// that stops waiting, leaves it closed for good.
#[derive(Debug)]
pub struct Writer {
    partition: Arc<Partition>,
    coordinator: Arc<dyn Coordinator>,
    lease: Lease,
    author: Author,
    vectorize: Option<Arc<dyn Vectorize>>,
    merging: Merging,
    ready: Mutex<bool>,
}

impl Writer {
    /// Start writing from the state the lease was taken at: `committed` must
    /// be what the coordinator returned together with `lease`.
    pub async fn open(
        blobs: Arc<dyn Blobs>,
        coordinator: Arc<dyn Coordinator>,
        lease: Lease,
        committed: &Committed,
    ) -> Result<Self> {
        let partition = Arc::new(Partition::load(lease.key.clone(), blobs, committed).await?);
        let author = Author {
            generation: lease.generation,
            writer: uuid::Uuid::new_v4(),
        };
        Ok(Self {
            partition,
            coordinator,
            lease,
            author,
            vectorize: None,
            merging: Merging::default(),
            ready: Mutex::new(true),
        })
    }

    /// Embed with `vectorize` for the collections that embed their own
    /// objects. Without it, a write that needs a vector made is refused.
    pub fn embedding_with(self, vectorize: Arc<dyn Vectorize>) -> Self {
        Self {
            vectorize: Some(vectorize),
            ..self
        }
    }

    /// Merge and index segments as `merging` says, instead of as by default.
    pub fn merging_with(self, merging: Merging) -> Self {
        self.partition
            .state
            .write()
            .expect("partition lock")
            .cluster_above(merging.clustered_above);
        Self { merging, ..self }
    }

    pub fn partition(&self) -> &Arc<Partition> {
        &self.partition
    }

    pub fn lease(&self) -> &Lease {
        &self.lease
    }

    pub fn author(&self) -> &Author {
        &self.author
    }

    /// Commit `mutations` as one atomic step. A rejected mutation fails the
    /// whole batch and changes nothing.
    pub async fn commit(&self, mutations: Vec<Mutation>) -> Result<Receipt> {
        self.commit_group(vec![mutations])
            .await?
            .pop()
            .expect("one result per batch")
    }

    /// Commit several callers' batches as one log segment. Each batch is
    /// atomic on its own and sees the batches before it; a rejected batch is
    /// left out with its own error and does not affect the others. Every
    /// accepted batch becomes visible at the same position.
    ///
    /// A batch that needs a vector which cannot be made is rejected like
    /// any other, and the rest are resolved again without it.
    pub async fn commit_group(&self, batches: Vec<Vec<Mutation>>) -> Result<Vec<Result<Receipt>>> {
        self.commit_group_with(batches, Embedded::new()).await
    }

    /// The texts `mutations` would leave to be embedded if they were
    /// committed now, with how each is embedded. A caller can make their
    /// vectors ahead of the commit and hand them to
    /// [`Self::commit_group_with`], rather than have the commit make them
    /// while every other write waits.
    pub fn wanted(&self, mutations: &[Mutation]) -> Vec<(Embedding, String)> {
        self.partition.read(|state| {
            let mut draft = Draft::new(state);
            let _ = draft.batch(mutations.to_vec());
            let (_, wanted) = draft.finish();
            wanted
                .into_iter()
                .map(|wanted| (wanted.embedding, wanted.text))
                .collect()
        })
    }

    /// As [`Self::commit_group`], using `made` for any vector the commit
    /// needs that it holds: one is used only for the very text and way of
    /// embedding the commit itself arrives at, and only if the collection
    /// can hold it.
    pub async fn commit_group_with(
        &self,
        batches: Vec<Vec<Mutation>>,
        made: Embedded,
    ) -> Result<Vec<Result<Receipt>>> {
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let mut unembedded: Vec<Option<(String, String)>> = vec![None; batches.len()];
        let mut made: Embedded = made
            .into_iter()
            .filter_map(|((embedding, text), vector)| {
                let vector = embedding.checked(vector).ok()?;
                Some(((embedding, text), vector))
            })
            .collect();
        let mut failed: HashMap<Embedding, String> = HashMap::new();
        let ((position, prev), outcomes, ops) = loop {
            let (at, outcomes, (mut ops, wanted)) = self.partition.read(|state| {
                let mut draft = Draft::new(state);
                let outcomes: Vec<_> = batches
                    .iter()
                    .zip(&unembedded)
                    .map(|(batch, unembedded)| match unembedded {
                        Some(_) => draft.batch(Vec::new()),
                        None => draft.batch(batch.clone()),
                    })
                    .collect();
                (state.at(), outcomes, draft.finish())
            });
            self.make(&wanted, &mut made, &mut failed).await;
            let unmade = |wanted: &&Wanted| failed.contains_key(&wanted.embedding);
            if !wanted.iter().any(|wanted| unmade(&wanted)) {
                for Wanted {
                    op,
                    embedding,
                    text,
                    ..
                } in wanted
                {
                    if let Op::Upsert { object, .. } = &mut ops[op] {
                        object.vector = made.get(&(embedding, text)).cloned();
                    }
                }
                break (at, outcomes, ops);
            }
            let rejected = |unembedded: &[Option<_>]| unembedded.iter().flatten().count();
            let before = rejected(&unembedded);
            for wanted in wanted.iter().filter(unmade) {
                let reason = failed[&wanted.embedding].clone();
                unembedded[wanted.batch] = Some((wanted.embedding.model.clone(), reason));
            }
            if rejected(&unembedded) == before {
                let (model, reason) = failed
                    .into_iter()
                    .next()
                    .map(|(embedding, reason)| (embedding.model, reason))
                    .unwrap_or_default();
                return Err(StorageError::Embedding { model, reason });
            }
        };
        let position = if ops.is_empty() {
            position
        } else {
            let log = LogSegment {
                position: following(position)?,
                prev,
                ops,
            };
            let key = Layout(&self.lease.key).log(log.position, &self.author);
            let frame = encode(Kind::Log, &log)?;
            attempt(&mut ready, async {
                self.publish(&key, frame, log.position).await?;
                self.partition
                    .state
                    .write()
                    .expect("partition lock")
                    .apply(&key, &log)
            })
            .await?;
            log.position
        };
        Ok(outcomes
            .into_iter()
            .zip(unembedded)
            .map(|(outcomes, unembedded)| match unembedded {
                Some((model, reason)) => Err(StorageError::Embedding { model, reason }),
                None => outcomes.map(|outcomes| Receipt { position, outcomes }),
            })
            .collect())
    }

    /// Make the vectors `wanted` that are not `made` yet, each way of
    /// embedding at the same time as the others, and note in `failed` why
    /// for each way that could not make all of its own. A way that has
    /// failed is not asked again.
    ///
    /// The writer is held meanwhile, so that no other write can change a
    /// text a vector is being made for.
    async fn make(
        &self,
        wanted: &[Wanted],
        made: &mut Embedded,
        failed: &mut HashMap<Embedding, String>,
    ) {
        let mut missing: HashMap<&Embedding, Vec<&String>> = HashMap::new();
        for Wanted {
            embedding, text, ..
        } in wanted
        {
            let known = failed.contains_key(embedding)
                || made.contains_key(&(embedding.clone(), text.clone()));
            let texts = missing.entry(embedding).or_default();
            if !known && !texts.contains(&text) {
                texts.push(text);
            }
        }
        let asked = missing
            .into_iter()
            .filter(|(_, texts)| !texts.is_empty())
            .map(|(embedding, texts)| async move {
                let texts: Vec<String> = texts.into_iter().cloned().collect();
                let vectors = self.vectors(embedding, texts.clone()).await;
                (embedding, vectors.map(|vectors| (texts, vectors)))
            });
        for (embedding, outcome) in join_all(asked).await {
            match outcome {
                Ok((texts, vectors)) => made.extend(
                    texts
                        .into_iter()
                        .zip(vectors)
                        .map(|(text, vector)| ((embedding.clone(), text), vector)),
                ),
                Err(reason) => {
                    failed.insert(embedding.clone(), reason);
                }
            }
        }
    }

    /// A vector the collection can hold for each of `texts`, or why not.
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let vectorize = self
            .vectorize
            .as_ref()
            .ok_or("this writer has nothing to embed with")?;
        let asked = texts.len();
        let vectors = tokio::time::timeout(EMBEDDING_TIMEOUT, vectorize.vectors(embedding, texts))
            .await
            .map_err(|_| format!("no vectors within {EMBEDDING_TIMEOUT:?}"))??;
        if vectors.len() != asked {
            return Err(format!("{} vectors for {asked} texts", vectors.len()));
        }
        vectors
            .into_iter()
            .map(|vector| embedding.checked(vector).map_err(|error| error.to_string()))
            .collect()
    }

    /// Write a checkpoint: move the rows held in memory into a segment and
    /// commit a manifest, so recovery no longer needs the log before it.
    pub async fn flush(&self) -> Result<LogPosition> {
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let (position, pending) = self
            .partition
            .read(|state| (state.position(), state.has_commits_since_checkpoint()));
        if !pending {
            return Ok(position);
        }
        let position = following(position)?;
        let layout = Layout(&self.lease.key);
        let segment_key = layout.segment(position, &self.author);
        let key = layout.manifest(position, &self.author);
        let (segment, manifest) = self
            .partition
            .read(|state| state.checkpoint(position, &segment_key, &layout.index(&segment_key)));
        let segment = match segment {
            Some((meta, file)) => Some(self.indexed(meta, file).await?),
            None => None,
        };
        let frame = encode(Kind::Manifest, &manifest)?;
        attempt(&mut ready, async {
            if let Some((meta, _, files)) = &segment {
                self.upload(meta, files).await?;
            }
            self.publish(&key, frame, position).await?;
            let mut state = self.partition.state.write().expect("partition lock");
            state.checkpointed(
                &key,
                position,
                segment.map(|(meta, loaded, _)| (meta, loaded)),
            )
        })
        .await?;
        Ok(position)
    }

    /// Encode a segment and index its rows, off the async threads and before
    /// anything is written, so that readers are not held behind the write
    /// lock while a large segment is indexed. Returns the segment ready to
    /// attach and the frames to upload: the segment's, and its index file's
    /// if it has one.
    async fn indexed(&self, meta: SegmentMeta, file: SegmentFile) -> Result<Indexed> {
        let (catalog, above) = self
            .partition
            .read(|state| (state.catalog().clone(), state.clustered_above()));
        let indexed = meta.index.is_some();
        let (loaded, frames) = blocking(move || {
            let segment = encode(Kind::Segment, &file)?;
            let indexes = state::indexed(&catalog, &file.rows, &Stored::default(), above);
            let index = indexed
                .then(|| encode(Kind::Index, &state::stored(&indexes)))
                .transpose()?;
            Ok::<_, StorageError>((Loaded::built(file, indexes), (segment, index)))
        })
        .await??;
        Ok((meta, loaded, frames))
    }

    /// Upload a segment's files, the index first so that the segment never
    /// sits in the store without it.
    async fn upload(&self, meta: &SegmentMeta, (segment, index): &Frames) -> Result<()> {
        if let Some((key, frame)) = meta.index.as_deref().zip(index.clone()) {
            self.partition.blobs.put(key, frame).await?;
        }
        self.partition.blobs.put(&meta.key, segment.clone()).await
    }

    /// Carry out the merge the partition most needs, if any: replace a run
    /// of segments with one holding their live rows, in a checkpoint that
    /// also flushes whatever is unflushed. The merged file is built from a
    /// snapshot, without holding the writer; a row that dies meanwhile is
    /// dead in the new segment too. Returns the checkpoint's position, or
    /// `None` when there was nothing to merge or the segments changed under
    /// the snapshot, in which case asking again is fine.
    pub async fn merge(&self) -> Result<Option<LogPosition>> {
        let Some(merge) = self.partition.read(|state| state.merge_plan(&self.merging)) else {
            return Ok(None);
        };
        let Some((file, wants_index)) = self.partition.read(|state| {
            let file = state.merged(&merge, state.position())?;
            let wants_index =
                state::clustered(state.catalog(), &file.rows, state.clustered_above());
            Some((file, wants_index))
        }) else {
            return Ok(None);
        };
        let layout = Layout(&self.lease.key);
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let position = following(self.partition.read(State::position))?;
        let merged_key = layout.merged(position, &self.author);
        let segment_key = layout.segment(position, &self.author);
        let key = layout.manifest(position, &self.author);
        let meta = SegmentMeta {
            key: merged_key.clone(),
            rows: file.rows.len() as u64,
            tombstones: file.tombstones.len() as u64,
            index: wants_index.then(|| layout.index(&merged_key)),
        };
        let Some((flushed, manifest)) = self.partition.read(|state| {
            state.unchanged(&merge).then(|| {
                let index_key = layout.index(&segment_key);
                state.compaction(position, &segment_key, &index_key, &merge, meta.clone())
            })
        }) else {
            return Ok(None);
        };
        let merged = self.indexed(meta, file).await?;
        let flushed = match flushed {
            Some((meta, file)) => Some(self.indexed(meta, file).await?),
            None => None,
        };
        if !self.partition.read(|state| state.unchanged(&merge)) {
            return Ok(None);
        }
        let frame = encode(Kind::Manifest, &manifest)?;
        attempt(&mut ready, async {
            self.upload(&merged.0, &merged.2).await?;
            if let Some((meta, _, files)) = &flushed {
                self.upload(meta, files).await?;
            }
            self.publish(&key, frame, position).await?;
            let mut state = self.partition.state.write().expect("partition lock");
            state.compacted(
                &key,
                position,
                flushed.map(|(meta, loaded, _)| (meta, loaded)),
                &merge,
                (merged.0, merged.1),
            )
        })
        .await?;
        Ok(Some(position))
    }

    /// Delete the partition's files that nothing needs any more: a stale
    /// writer's uploads, and the segments and manifests that merges and
    /// checkpoints have replaced. Everything reached from any head written
    /// in the last `grace` stays, so that the partition can be put back as
    /// it was at any moment in that window and a reader that took the
    /// committed position within it still finds its files. Nothing at or
    /// ahead of this writer's position is touched, which keeps a newer
    /// writer's uploads safe whatever the clocks say. Returns how many
    /// files were deleted.
    ///
    /// What is needed is decided by the committed chain alone: it is walked
    /// back to the latest checkpoint at or before the oldest recent file,
    /// and everything before that checkpoint which it does not list goes.
    /// The files are listed before the head is read, with the writer held
    /// so that no commit is in flight: a file this writer uploaded and had
    /// not yet published when the listing was taken is reached by then or
    /// belongs to a commit that failed.
    pub async fn clean(&self, grace: Duration) -> Result<usize> {
        let layout = Layout(&self.lease.key);
        let listed = self.partition.blobs.list(&layout.prefix()).await?;
        let (position, head) = {
            let ready = self.ready.lock().await;
            if !*ready {
                return Err(StorageError::Closed);
            }
            self.partition.read(State::at)
        };
        let cutoff = chrono::Duration::from_std(grace)
            .ok()
            .and_then(|grace| chrono::Utc::now().checked_sub_signed(grace));
        let Some(cutoff) = cutoff else {
            return Ok(0);
        };
        let files: Vec<(&Listed, LogPosition)> = listed
            .iter()
            .filter_map(|file| Some((file, layout.position(&file.key)?)))
            .collect();
        let young = files
            .iter()
            .filter(|(file, _)| file.modified >= cutoff)
            .map(|(_, at)| *at);
        let floor = young.fold(position, std::cmp::min);
        let mut cursor = head;
        let base = loop {
            let Some(key) = cursor else { break None };
            cursor = match decode_head(&key, &self.partition.fetch(&key).await?)? {
                Head::Log(log) => log.prev,
                Head::Manifest(manifest) if manifest.position <= floor => break Some(manifest),
                Head::Manifest(manifest) => manifest.prev,
            };
        };
        let Some(base) = base else {
            return Ok(0);
        };
        let kept: HashSet<&str> = base
            .segments
            .iter()
            .flat_map(|meta| [Some(meta.key.as_str()), meta.index.as_deref()])
            .flatten()
            .collect();
        let orphans = files
            .iter()
            .filter(|(file, at)| *at < base.position && !kept.contains(file.key.as_str()));
        let mut deleted = 0;
        for (file, _) in orphans {
            self.partition.blobs.delete(&file.key).await?;
            deleted += 1;
        }
        tracing::debug!(partition = %self.lease.key, ?position, deleted, "cleaned");
        Ok(deleted)
    }

    async fn publish(&self, key: &str, frame: Bytes, position: LogPosition) -> Result<()> {
        self.partition.blobs.put(key, frame).await?;
        let commit = Commit {
            log_position: position,
            manifest: Some(ManifestRef(key.to_string())),
        };
        let mut tries = 1;
        loop {
            match self.coordinator.commit(&self.lease, commit.clone()).await {
                Err(error) if error.is_retryable() && tries < COMMIT_ATTEMPTS => {
                    tokio::time::sleep(COMMIT_RETRY_DELAY * tries).await;
                    tries += 1;
                }
                outcome => return outcome.map(|_| ()).map_err(Into::into),
            }
        }
    }
}

fn following(position: LogPosition) -> Result<LogPosition> {
    position
        .next()
        .ok_or(StorageError::Exhausted("log positions"))
}

/// Run one write attempt with the writer not ready, making it ready again
/// only if the attempt ran to completion and succeeded.
async fn attempt<T>(ready: &mut bool, work: impl Future<Output = Result<T>>) -> Result<T> {
    *ready = false;
    let done = work.await?;
    *ready = true;
    Ok(done)
}
