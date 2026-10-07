use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::{join_all, try_join_all};
use tokio::sync::Mutex;
use vellev_core::Embedding;

use crate::blobs::{is_plain, Author, Blobs, Layout, Listed};
use crate::catalog::Catalog;
use crate::column::Column;
use crate::coordinator::{
    Commit, Committed, Coordinator, Generation, Incarnation, Lease, LogPosition, ManifestRef,
    PartitionKey,
};
use crate::draft::{Draft, Wanted};
use crate::format::{
    decode, decode_head, encode, Answer, Answered, Head, Kind, LogSegment, Manifest, Op,
    SegmentFile, SegmentIndex, SegmentMeta,
};
use crate::mutation::{Batch, Mutation, Receipt, Vectorize};
use crate::state::{self, Indexes, Loaded, Merging, Names, Shape, Snapshot, State, Stored};
use crate::{Refusal, Result, StorageError};

/// How many times a commit whose outcome is unknown is tried. Repeating is
/// safe because the coordinator accepts an identical commit twice.
const COMMIT_ATTEMPTS: u32 = 3;
/// How many rows a flush copies from memory while commits wait.
const COPIED_AT_ONCE: usize = 4_096;

/// A segment built and ready to write: its file without the rows'
/// vectors, which the column holds, its indexes, and the frames of the
/// file and the column.
struct Built {
    file: SegmentFile,
    vectors: Column,
    indexes: Indexes,
    segment: Bytes,
    column: Bytes,
}

/// A built segment as the manifest names it, with its index file's frame
/// if it has one.
struct Prepared {
    meta: SegmentMeta,
    built: Built,
    index: Option<Bytes>,
}
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
    /// The partition as registered when this copy was loaded; a copy never
    /// follows it into another.
    incarnation: Incarnation,
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

    pub fn incarnation(&self) -> Incarnation {
        self.incarnation
    }

    /// Whether this copy has everything up to `position` of `incarnation`.
    /// A later incarnation has every point of an earlier one behind it.
    pub fn reached(&self, incarnation: Incarnation, position: LogPosition) -> bool {
        match self.incarnation.cmp(&incarnation) {
            std::cmp::Ordering::Greater => true,
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => self.read(State::position) >= position,
        }
    }

    /// Run `f` against a consistent view of the partition. The lock is a
    /// blocking one and is never held across an await.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.state.read().expect("partition lock"))
    }

    async fn load(key: PartitionKey, blobs: Arc<dyn Blobs>, committed: &Committed) -> Result<Self> {
        Self::load_reusing(key, blobs, committed, None).await
    }

    /// Load the partition as committed, taking the segments `donor` has
    /// loaded already rather than reading them again.
    async fn load_reusing(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        committed: &Committed,
        donor: Option<&Partition>,
    ) -> Result<Self> {
        let partition = Self::empty(key, blobs, committed.incarnation);
        partition.advance_reusing(committed, donor).await?;
        Ok(partition)
    }

    fn empty(key: PartitionKey, blobs: Arc<dyn Blobs>, incarnation: Incarnation) -> Self {
        Self {
            key,
            incarnation,
            blobs,
            state: RwLock::new(State::default()),
            seen: Mutex::new(Generation(0)),
        }
    }

    /// The files the partition as `committed` is loaded from: the logs back
    /// to its latest checkpoint, the checkpoint, and what that lists.
    pub async fn files(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        committed: &Committed,
    ) -> Result<Vec<String>> {
        let partition = Self::empty(key, blobs, committed.incarnation);
        let Tail { checkpoint, logs } = partition
            .tail(committed, LogPosition::default(), None)
            .await?
            .ok_or_else(|| {
                StorageError::corrupt(
                    &Layout(&partition.key).prefix(),
                    "its history reaches neither a checkpoint nor the start",
                )
            })?;
        let checkpoint = checkpoint
            .iter()
            .flat_map(|(key, manifest)| std::iter::once(key.as_str()).chain(listed_by(manifest)));
        Ok(checkpoint
            .map(str::to_string)
            .chain(logs.into_iter().map(|(key, _)| key))
            .collect())
    }

    /// The segments loaded, by key.
    fn held(&self) -> HashMap<String, Loaded> {
        self.read(|state| {
            state
                .loaded_segments()
                .map(|(key, loaded)| (key.to_string(), loaded))
                .collect()
        })
    }

    /// Fetch a file named by data rather than built by [`Layout`], which must
    /// be a plain key of this partition's.
    async fn fetch(&self, key: &str) -> Result<Bytes> {
        self.own(key)?;
        self.blobs.get(key).await
    }

    fn own(&self, key: &str) -> Result<()> {
        match is_plain(key) && key.starts_with(&Layout(&self.key).prefix()) {
            true => Ok(()),
            false => Err(StorageError::corrupt(
                key,
                "the key is not one of this partition's",
            )),
        }
    }

    /// The vectors under `key`, read in place from the store's copy on
    /// local disk when it keeps one; otherwise `held`, the same vectors
    /// still in memory, or when there is no such thing the file fetched
    /// and decoded into memory.
    async fn column(&self, key: &str, held: Option<Column>) -> Result<Column> {
        match self.blobs.local(key).await {
            Ok(Some(path)) => {
                let name = key.to_string();
                match blocking(move || Column::map(&name, &path)).await? {
                    Ok(column) => return Ok(column),
                    Err(error) => {
                        tracing::warn!(key, %error, "a vectors file on local disk could not be read in place")
                    }
                }
            }
            Ok(None) => {}
            Err(error) => {
                tracing::warn!(key, %error, "a vectors file could not be copied to local disk")
            }
        }
        match held {
            Some(column) => Ok(column),
            None => {
                self.own(key)?;
                let (name, bytes) = (key.to_string(), self.blobs.get(key).await?);
                blocking(move || Column::decode(&name, &bytes)).await?
            }
        }
    }

    /// Generations only grow, so a reading from an older generation than one
    /// already applied is out of date whatever its position, and so is a
    /// lower position from the same generation. A reading that is not ahead
    /// and not stale means the history this copy holds was replaced.
    async fn advance(&self, committed: &Committed) -> Result<LogPosition> {
        self.advance_reusing(committed, None).await
    }

    /// As [`Self::advance`], taking the segments a checkpoint lists from
    /// `donor` where it holds them, rather than only from this copy.
    async fn advance_reusing(
        &self,
        committed: &Committed,
        donor: Option<&Partition>,
    ) -> Result<LogPosition> {
        let mut seen = self.seen.lock().await;
        let (position, head) = self.read(State::at);
        let target = committed.manifest.as_ref().map(|head| head.0.as_str());
        let stale = committed.generation < *seen
            || (committed.generation == *seen && committed.log_position < position);
        if stale {
            return Ok(position);
        }
        if committed.incarnation != self.incarnation {
            return Err(StorageError::Diverged { head });
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
                let held = donor.unwrap_or(self).held();
                let mut state = self.restore(&key, manifest, held).await?;
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

    async fn restore(
        &self,
        key: &str,
        manifest: Manifest,
        held: HashMap<String, Loaded>,
    ) -> Result<State> {
        // Each segment is decoded and indexed on a thread of its own as it
        // arrives, rather than all of them in turn once every one has.
        let catalog = Arc::new(
            Catalog::restore(manifest.next_collection_id, manifest.collections.clone())
                .map_err(|reason| StorageError::corrupt(key, reason))?,
        );
        let segments = try_join_all(manifest.segments.iter().map(|meta| {
            let held = held.get(&meta.key).cloned();
            let catalog = catalog.clone();
            async move {
                if let Some(loaded) = held {
                    return Ok::<_, StorageError>(loaded);
                }
                let bytes = self.fetch(&meta.key).await?;
                let vectors = match &meta.vectors {
                    Some(key) => Some(self.column(key, None).await?),
                    None => None,
                };
                let stored = match &meta.index {
                    Some(index) => self.stored(index, &meta.key).await,
                    None => Stored::default(),
                };
                let key = meta.key.clone();
                blocking(move || {
                    let mut file: SegmentFile = decode(&key, Kind::Segment, &bytes)?;
                    let vectors = vectors.unwrap_or_else(|| Column::taken_from(&mut file.rows));
                    Loaded::fresh(file, vectors, stored).indexed_now(&key, &catalog)
                })
                .await?
            }
        }))
        .await?;
        let replay = try_join_all(manifest.replay.iter().map(|key| async move {
            let log: LogSegment = decode(key, Kind::Log, &self.fetch(key).await?)?;
            Ok::<_, StorageError>((key.clone(), log))
        }))
        .await?;
        let key = key.to_string();
        blocking(move || {
            let (position, after) = (manifest.position, manifest.replay_after);
            State::from_manifest(&key, manifest, segments)?.replayed(&key, position, after, &replay)
        })
        .await?
    }

    /// The clustered indexes kept under `key` for the segment under
    /// `segment`, or none if they cannot be read or are another segment's,
    /// in which case the segment is searched by a scan of its codes until
    /// it is next loaded.
    async fn stored(&self, key: &str, segment: &str) -> Stored {
        let read = async {
            let index = decode::<SegmentIndex>(key, Kind::Index, &self.fetch(key).await?)?;
            match index.segment == segment {
                true => Ok(index),
                false => Err(StorageError::corrupt(key, "it indexes another segment")),
            }
        };
        match read.await {
            Ok(index) => index.into(),
            Err(error) => {
                tracing::warn!(key, %error, "a segment's index file is not used");
                Stored::default()
            }
        }
    }
}

/// What a batch's request was answered by before this commit, if
/// anything.
enum Earlier {
    None,
    /// A commit before this one, which made it or refused it.
    Answered(std::result::Result<Receipt, Refusal>),
    /// The batch at this place in the same group.
    InGroup(usize),
}

/// Run `work`, which takes long enough to hold up the async threads, on
/// one of its own.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| StorageError::Background(error.to_string()))
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
    /// Held while a flush or a merge builds its segment without `ready`,
    /// so that one never builds on a state the other is about to replace.
    building: Mutex<()>,
    /// Set while a restore runs, when writes are refused.
    restoring: AtomicBool,
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
            building: Mutex::new(()),
            restoring: AtomicBool::new(false),
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
        let batches = batches.into_iter().map(Batch::from).collect();
        self.commit_batches(batches, made).await
    }

    /// As [`Self::commit_group_with`], for batches that may name the
    /// request they came in. A request this partition answered lately is
    /// answered again as it was and nothing of it is made, and so is a
    /// request that comes twice in one group, the second time. A request
    /// is answered by the commit that makes it, refuses it or finds it
    /// changes nothing, and is recorded with that commit, so that a writer
    /// which takes the partition over answers it alike. One that fails in
    /// a way that may pass, an embedding that could not be made, is not
    /// answered and may be tried again.
    pub async fn commit_batches(
        &self,
        batches: Vec<Batch>,
        made: Embedded,
    ) -> Result<Vec<Result<Receipt>>> {
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        if self.restoring.load(Ordering::SeqCst) {
            return Err(StorageError::Restoring);
        }
        let receipt = |position, outcomes| Receipt {
            incarnation: self.partition.incarnation,
            position,
            outcomes,
        };
        // Each batch either answered already, or the first of its request
        // in the group, if it is not that first itself.
        let earlier: Vec<Earlier> = self.partition.read(|state| {
            batches
                .iter()
                .enumerate()
                .map(|(at, batch)| {
                    let Some(request) = batch.request else {
                        return Earlier::None;
                    };
                    if let Some((position, answer)) = state.answer(request) {
                        return Earlier::Answered(match answer {
                            Answer::Made(outcomes) => Ok(receipt(position, outcomes.clone())),
                            Answer::Refused(refusal) => Err(refusal.clone()),
                        });
                    }
                    match batches[..at]
                        .iter()
                        .position(|before| before.request == Some(request))
                    {
                        Some(first) => Earlier::InGroup(first),
                        None => Earlier::None,
                    }
                })
                .collect()
        });
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
                    .zip(&earlier)
                    .map(
                        |((batch, unembedded), earlier)| match (unembedded, earlier) {
                            (None, Earlier::None) => draft.batch(batch.mutations.clone()),
                            _ => draft.batch(Vec::new()),
                        },
                    )
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
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis() as u64);
        let at = self.partition.read(|state| state.stamp(now));
        // Every request whose fate this commit decides, made or refused,
        // even if it changed nothing, so that no later try makes it.
        let answered: Vec<Answered> = batches
            .iter()
            .zip(&outcomes)
            .zip(unembedded.iter().zip(&earlier))
            .filter_map(|((batch, outcomes), decided)| {
                let answer = match (outcomes, decided) {
                    (Ok(outcomes), (None, Earlier::None)) => Answer::Made(outcomes.clone()),
                    (Err(error), (None, Earlier::None)) => Answer::Refused(error.refusal()?),
                    _ => return None,
                };
                Some(Answered {
                    request: batch.request?,
                    at,
                    answer,
                })
            })
            .collect();
        let position = if ops.is_empty() && answered.is_empty() {
            position
        } else {
            let log = LogSegment {
                position: following(position)?,
                prev,
                ops,
                answered,
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
        // A batch whose request comes again later in the group shares its
        // failure with that later one.
        let repeated: HashSet<usize> = earlier
            .iter()
            .filter_map(|earlier| match earlier {
                Earlier::InGroup(first) => Some(*first),
                _ => None,
            })
            .collect();
        let mut results: Vec<Result<Receipt>> = outcomes
            .into_iter()
            .zip(unembedded)
            .enumerate()
            .map(|(at, (outcomes, unembedded))| {
                let result = match unembedded {
                    Some((model, reason)) => Err(StorageError::Embedding { model, reason }),
                    None => outcomes.map(|outcomes| receipt(position, outcomes)),
                };
                result.map_err(|error| match repeated.contains(&at) {
                    true => StorageError::Shared(Arc::new(error)),
                    false => error,
                })
            })
            .collect();
        for (at, earlier) in earlier.into_iter().enumerate() {
            results[at] = match earlier {
                Earlier::None => continue,
                Earlier::Answered(answer) => answer.map_err(StorageError::Refused),
                Earlier::InGroup(first) => match &results[first] {
                    Ok(receipt) => Ok(receipt.clone()),
                    Err(StorageError::Shared(shared)) => Err(StorageError::Shared(shared.clone())),
                    Err(_) => unreachable!("a repeated request's failure is shared"),
                },
            };
        }
        Ok(results)
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
    /// The segment is built from the partition as it is now and written
    /// while commits go on; the manifest then names the logs committed
    /// meanwhile, which are replayed over it. Returns the position the
    /// checkpoint was committed at, or the current one when nothing needed
    /// flushing or a checkpoint made meanwhile left nothing to do.
    pub async fn flush(&self) -> Result<LogPosition> {
        let _building = self.building.lock().await;
        let Some(mut snapshot) = self.partition.read(State::snapshot) else {
            return Ok(self.partition.read(State::position));
        };
        while self
            .partition
            .read(|state| state.copy_rows(&mut snapshot, COPIED_AT_ONCE))
        {
            tokio::task::yield_now().await;
        }
        let named_at = snapshot.position;
        let segment = match self.uploaded(&mut snapshot, named_at).await? {
            Some(prepared) => Some(self.loaded(prepared).await?),
            None => None,
        };
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let position = following(self.partition.read(State::position))?;
        let key = Layout(&self.lease.key).manifest(position, &self.author);
        let flushed = segment.as_ref().map(|(meta, _)| meta.clone());
        let manifest = self
            .partition
            .read(|state| state.flushed_manifest(position, &snapshot, flushed));
        let frame = encode(Kind::Manifest, &manifest)?;
        let discarded = attempt(&mut ready, async {
            self.publish(&key, frame, position).await?;
            let mut state = self.partition.state.write().expect("partition lock");
            state.flushed(&key, position, snapshot, segment)
        })
        .await?;
        drop(ready);
        blocking(move || drop(discarded)).await?;
        Ok(position)
    }

    /// Put the partition back as it stood when `head`, committed at
    /// `position`, was its head; with none, as it was before anything was
    /// committed.
    /// Nothing is rewound: the state then becomes a checkpoint after the
    /// partition's position and the target's both, of the segments it had
    /// and the rows it held in memory, written as one segment. Every state
    /// before stays as restorable as it was, and the requests answered
    /// lately stay answered. Writes are refused while it runs, so that none
    /// is answered only to be undone. Returns the checkpoint's position.
    ///
    /// It holds the building lock throughout, as cleaning up does, so the
    /// target's files are not deleted while it is read, and the segment,
    /// named at the position the partition is at, stays until committed.
    pub async fn restore(
        &self,
        head: Option<&ManifestRef>,
        position: LogPosition,
    ) -> Result<LogPosition> {
        let _building = self.building.lock().await;
        let _refusing = Refusing::writes(self).await;
        let at = Committed {
            generation: self.lease.generation,
            incarnation: self.partition.incarnation,
            log_position: position,
            manifest: head.cloned(),
        };
        let target = Partition::load_reusing(
            self.lease.key.clone(),
            self.partition.blobs.clone(),
            &at,
            Some(&self.partition),
        )
        .await?;
        let mut held = target.held();
        let segment = match target.read(State::snapshot) {
            Some(mut snapshot) => {
                target.read(|state| state.copy_rows(&mut snapshot, usize::MAX));
                let named_at = self.partition.read(State::position);
                match self.uploaded(&mut snapshot, named_at).await? {
                    Some(prepared) => {
                        let (meta, loaded) = self.loaded(prepared).await?;
                        held.insert(meta.key.clone(), loaded);
                        Some(meta)
                    }
                    None => None,
                }
            }
            None => None,
        };
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let restored = following(self.partition.read(State::position).max(position))?;
        let key = Layout(&self.lease.key).manifest(restored, &self.author);
        let manifest = self.partition.read(|current| {
            target.read(|target| current.restored_manifest(restored, target, segment))
        });
        let frame = encode(Kind::Manifest, &manifest)?;
        let state = target.restore(&key, manifest, held).await?;
        let replaced = attempt(&mut ready, async {
            self.publish(&key, frame, restored).await?;
            Ok(std::mem::replace(
                &mut *self.partition.state.write().expect("partition lock"),
                state,
            ))
        })
        .await?;
        drop(ready);
        blocking(move || drop((replaced, target))).await?;
        Ok(restored)
    }

    /// Build and upload a segment of the rows taken out of `snapshot` and
    /// its tombstones, its files named at `named_at`; none if it has
    /// neither.
    async fn uploaded(
        &self,
        snapshot: &mut Snapshot,
        named_at: LogPosition,
    ) -> Result<Option<Prepared>> {
        if snapshot.rows.is_empty() && snapshot.tombstones.is_empty() {
            return Ok(None);
        }
        let names = self.staged(named_at);
        let rows = std::mem::take(&mut snapshot.rows);
        let above = self.merging.clustered_above;
        let shape = Shape::of(&snapshot.catalog, &rows, snapshot.tombstones.len(), above);
        let file = SegmentFile {
            position: snapshot.position,
            rows,
            tombstones: snapshot.tombstones.clone(),
        };
        let built = self.built(file, snapshot.catalog.clone()).await?;
        let prepared = Self::prepared(names.meta(shape), built)?;
        self.upload(&prepared).await?;
        Ok(Some(prepared))
    }

    /// Names for the files of a segment built from the partition as it was
    /// at `position`, apart from those of any other build from there.
    fn staged(&self, position: LogPosition) -> Names {
        let nonce = uuid::Uuid::new_v4().as_u64_pair().0;
        self.names(position, &format!("-{nonce:016x}"))
    }

    /// Names for a segment's files written at `position`, told apart from
    /// others written there by `suffix`.
    fn names(&self, position: LogPosition, suffix: &str) -> Names {
        let layout = Layout(&self.lease.key);
        let named = |kind| layout.named(kind, position, &self.author, suffix);
        Names {
            segment: named("segment"),
            index: named("index"),
            vectors: named("vectors"),
        }
    }

    /// Index a segment's rows, which have their vectors with them, then
    /// take the vectors into their column and encode both, off the async
    /// threads and before anything is written, so that readers are not
    /// held behind the write lock while a large segment is indexed.
    async fn built(&self, mut file: SegmentFile, catalog: Catalog) -> Result<Built> {
        let above = Some(self.merging.clustered_above);
        blocking(move || {
            let inline = |row: usize| file.rows[row].object.vector.as_deref();
            let indexes = state::indexed(&catalog, &file.rows, inline, &Stored::default(), above);
            let vectors = Column::taken_from(&mut file.rows);
            let segment = encode(Kind::Segment, &file)?;
            let column = vectors.encode();
            Ok(Built {
                file,
                vectors,
                indexes,
                segment,
                column,
            })
        })
        .await?
    }

    /// A built segment as `meta` names it, with its index file's frame if
    /// it has one.
    fn prepared(meta: SegmentMeta, built: Built) -> Result<Prepared> {
        let index = meta
            .index
            .is_some()
            .then(|| encode(Kind::Index, &state::stored(&meta.key, &built.indexes)))
            .transpose()?;
        Ok(Prepared { meta, built, index })
    }

    /// Upload a segment's files, the segment's own last so that it never
    /// sits in the store without its companions.
    async fn upload(&self, prepared: &Prepared) -> Result<()> {
        let Prepared { meta, built, index } = prepared;
        if let Some(key) = &meta.vectors {
            self.partition.blobs.put(key, built.column.clone()).await?;
        }
        if let Some((key, frame)) = meta.index.as_deref().zip(index.clone()) {
            self.partition.blobs.put(key, frame).await?;
        }
        self.partition
            .blobs
            .put(&meta.key, built.segment.clone())
            .await
    }

    /// An uploaded segment as the state holds it, with its vectors read
    /// from local disk where the store keeps a copy.
    async fn loaded(&self, prepared: Prepared) -> Result<(SegmentMeta, Loaded)> {
        let Prepared { meta, built, .. } = prepared;
        let vectors = match &meta.vectors {
            Some(key) => self.partition.column(key, Some(built.vectors)).await?,
            None => built.vectors,
        };
        let loaded = Loaded {
            file: Arc::new(built.file),
            vectors: Arc::new(vectors),
            indexes: Some(Arc::new(built.indexes)),
            stored: Stored::default(),
        };
        Ok((meta, loaded))
    }

    /// Carry out the merge the partition most needs, if any: replace a run
    /// of segments with one holding their live rows, in a checkpoint that
    /// also flushes whatever is unflushed. The merged file is built from a
    /// snapshot, without holding the writer; a row that dies meanwhile is
    /// dead in the new segment too. Returns the checkpoint's position, or
    /// `None` when there was nothing to merge or the segments changed under
    /// the snapshot, in which case asking again is fine.
    pub async fn merge(&self) -> Result<Option<LogPosition>> {
        let _building = self.building.lock().await;
        let Some(merge) = self.partition.read(|state| state.merge_plan(&self.merging)) else {
            return Ok(None);
        };
        let above = self.merging.clustered_above;
        let Some((file, shape)) = self.partition.read(|state| {
            let file = state.merged(&merge, state.position())?;
            let shape = Shape::of(state.catalog(), &file.rows, file.tombstones.len(), above);
            Some((file, shape))
        }) else {
            return Ok(None);
        };
        let catalog = self.partition.read(|state| state.catalog().clone());
        let built = self.built(file, catalog).await?;
        let layout = Layout(&self.lease.key);
        let mut ready = self.ready.lock().await;
        if !*ready {
            return Err(StorageError::Closed);
        }
        let position = following(self.partition.read(State::position))?;
        let key = layout.manifest(position, &self.author);
        let meta = self.names(position, "-merged").meta(shape);
        let flushed_names = self.names(position, "");
        let Some((flushed, manifest)) = self.partition.read(|state| {
            state
                .unchanged(&merge)
                .then(|| state.compaction(position, &flushed_names, above, &merge, meta.clone()))
        }) else {
            return Ok(None);
        };
        let merged = Self::prepared(meta, built)?;
        let flushed = match flushed {
            Some((meta, file)) => {
                let catalog = self.partition.read(|state| state.catalog().clone());
                Some(Self::prepared(meta, self.built(file, catalog).await?)?)
            }
            None => None,
        };
        if !self.partition.read(|state| state.unchanged(&merge)) {
            return Ok(None);
        }
        let frame = encode(Kind::Manifest, &manifest)?;
        attempt(&mut ready, async {
            self.upload(&merged).await?;
            if let Some(flushed) = &flushed {
                self.upload(flushed).await?;
            }
            self.publish(&key, frame, position).await?;
            let merged = self.loaded(merged).await?;
            let flushed = match flushed {
                Some(flushed) => Some(self.loaded(flushed).await?),
                None => None,
            };
            let mut state = self.partition.state.write().expect("partition lock");
            state.compacted(&key, position, flushed, &merge, merged)
        })
        .await?;
        Ok(Some(position))
    }

    /// Delete the partition's files that nothing needs any more: a stale
    /// writer's uploads, and the segments and manifests that merges and
    /// checkpoints have replaced. Everything reached from any head written
    /// in the last `grace` stays, and from the head just before the oldest
    /// of them, so that the partition can be put back as it was at any
    /// moment in that window and as it stood when the window began, and a
    /// reader that took the committed position within it still finds its
    /// files. Nothing at or ahead of this writer's position is touched,
    /// which keeps a newer writer's uploads safe whatever the clocks say.
    /// Returns how many files were deleted.
    ///
    /// What is needed is decided by the committed chain alone: it is walked
    /// back to the latest checkpoint before the oldest recent file, and
    /// everything before that checkpoint which no checkpoint on the way
    /// lists goes. A restore's checkpoint lists segments far older than
    /// itself, which stay for as long as it is on the way.
    /// The files are listed before the head is read, with the writer held
    /// so that no commit is in flight: a file this writer uploaded and had
    /// not yet published when the listing was taken is reached by then or
    /// belongs to a commit that failed. A flush uploads its segment's files
    /// without the writer, under the position of the partition it was built
    /// from, which is past the latest checkpoint and so past anything this
    /// deletes until a checkpoint after it has been committed.
    pub async fn clean(&self, grace: Duration) -> Result<usize> {
        let _building = self.building.lock().await;
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
        let floor = young
            .map(|at| LogPosition(at.0.saturating_sub(1)))
            .fold(position, std::cmp::min);
        let mut cursor = head;
        let mut checkpoints = Vec::new();
        let base = loop {
            let Some(key) = cursor else { break None };
            cursor = match decode_head(&key, &self.partition.fetch(&key).await?)? {
                Head::Log(log) => log.prev,
                Head::Manifest(manifest) if manifest.position <= floor => break Some(manifest),
                Head::Manifest(manifest) => {
                    let prev = manifest.prev.clone();
                    checkpoints.push(manifest);
                    prev
                }
            };
        };
        let Some(base) = base else {
            return Ok(0);
        };
        let kept: HashSet<&str> = checkpoints
            .iter()
            .chain([&base])
            .flat_map(listed_by)
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

/// The files a checkpoint lists: its segments' and the logs it replays.
fn listed_by(checkpoint: &Manifest) -> impl Iterator<Item = &str> {
    checkpoint
        .segments
        .iter()
        .flat_map(|meta| {
            [
                Some(meta.key.as_str()),
                meta.index.as_deref(),
                meta.vectors.as_deref(),
            ]
        })
        .flatten()
        .chain(checkpoint.replay.iter().map(String::as_str))
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

/// Writes refused for as long as this lives.
struct Refusing<'a>(&'a AtomicBool);

impl<'a> Refusing<'a> {
    /// Refuse `writer`'s writes from now on, once any commit under way has
    /// been made.
    async fn writes(writer: &'a Writer) -> Refusing<'a> {
        writer.restoring.store(true, Ordering::SeqCst);
        drop(writer.ready.lock().await);
        Refusing(&writer.restoring)
    }
}

impl Drop for Refusing<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}
