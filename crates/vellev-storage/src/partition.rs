use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::{join_all, try_join_all};
use tokio::sync::{Mutex, MutexGuard};
use tokio::time::Instant;
use vellev_core::{Embedding, ModelFault, ObjectError};

use crate::backups::Held;
use crate::blobs::{is_plain, Author, Blobs, Disk, Layout, Listed, Local, Transfers};
use crate::catalog::Catalog;
use crate::column::{self, Column};
use crate::coordinator::{
    Cleaned, Clock, Commit, Committed, Coordinator, CoordinatorError, Generation, Incarnation,
    Lease, Lineage, LogPosition, ManifestRef, PartitionKey, Prior, Restore, SystemClock,
};
use crate::draft::{Draft, Wanted};
use crate::format::{
    decode, decode_head, encode, rekeyed_index, Answer, Answered, Head, Kind, LogSegment, Manifest,
    Op, SegmentFile, SegmentIndex, SegmentMeta, SUMMED_AT_ONCE,
};
use crate::mutation::{Batch, Mutation, Receipt, Unembedded, Vectorize, Weight};
use crate::state::{
    self, Discarded, Fit, Indexes, Loaded, Merging, Names, Shape, Snapshot, State, Stored,
};
use crate::{Refusal, Result, StorageError};

/// How many times a commit whose outcome is unknown is tried. Repeating is
/// safe because the coordinator accepts an identical commit twice.
const COMMIT_ATTEMPTS: u32 = 3;
/// How many rows a flush copies from memory while commits wait.
const COPIED_AT_ONCE: usize = 4_096;
/// How many segments a restore from another store is copying at once.
const SEGMENTS_COPIED_AT_ONCE: usize = 4;
/// How many of the files a clean deletes are being deleted at any moment.
const CLEANING_AT_ONCE: usize = 16;
/// How long a clean goes on finding what to delete, and how long again
/// asking for it to be deleted: long enough for all it deletes at once
/// from a store that answers as stores do, and no longer than anything
/// that waits for it can well wait.
const CLEANING_WITHIN: Duration = Duration::from_secs(60);

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
    /// local disk, or `held`, the same vectors still in memory, when that
    /// copy is not there, is not whole or is not reached in time.
    ///
    /// With neither, a store that keeps copies fetches the file into its
    /// copy, a part at a time, and the copy is read in place. Vectors a
    /// store keeps copies of are never read whole into memory: they may be
    /// more than it holds. A copy whose disk does not answer, or does not
    /// take the file, fails the load, to be tried again, and one that
    /// answers slowly is waited for as long as it gets on. Only from a
    /// store that keeps no copies at all, which a node is not run on, are
    /// they fetched and held in memory; and anywhere but on Unix, where no
    /// file is read in place.
    async fn column(&self, key: &str, held: Option<Column>) -> Result<Column> {
        self.own(key)?;
        let in_place = cfg!(unix);
        let copy = match self.blobs.local(key).await? {
            Some(local) if in_place => self.mapped(key, local).await,
            _ => Ok(None),
        };
        match (copy, held) {
            (Ok(Some(column)), _) => return Ok(column),
            (Ok(None), Some(held)) => return Ok(held),
            (Err(error), Some(held)) => {
                tracing::warn!(key, %error, "vectors are held in memory, their copy on local disk not reached");
                return Ok(held);
            }
            (Err(unreached), None) => return Err(unreached),
            (Ok(None), None) => {}
        }
        let fetched = match in_place {
            true => self.blobs.fetch_local(key).await?,
            false => None,
        };
        match fetched {
            Some(local) => self.mapped(key, local).await?.ok_or_else(|| {
                StorageError::corrupt(key, "as fetched to local disk it is no vectors file")
            }),
            None => {
                let (name, bytes) = (key.to_string(), self.blobs.get(key).await?);
                blocking(move || Column::decode(&name, &bytes)).await?
            }
        }
    }

    /// The vectors under `key` read in place from the store's copy `local`,
    /// on the disk the copy lies on, once the whole file is seen to sum to
    /// what it says: in its turn among the large files read there, for as
    /// long as that takes while it gets on, unless this process has summed
    /// the file before as it lies now. None if the copy is not a whole
    /// vectors file, and a failure if its disk gets nowhere with it.
    async fn mapped(&self, key: &str, local: Local) -> Result<Option<Column>> {
        let Local { path, disk } = local;
        let unreached = |error: std::io::Error| StorageError::Blob {
            key: key.to_string(),
            source: error.into(),
        };
        // A file this process has summed as it lies now is only laid out
        // in place again, which reads its ends alone and needs no turn
        // among the large files being read.
        let looked_at = disk.run(0, {
            let path = path.clone();
            move || column::seen(&path)
        });
        let file = match looked_at.await {
            Ok(file) => file,
            Err(gone) if gone.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(unreached(error)),
        };
        let name = key.to_string();
        let summed = match column::summed_before(&file) {
            true => {
                let laid_out = move || Ok(Column::map(&name, &path));
                let column = disk.run(0, laid_out).await;
                column.map(|column| column.map(column::Unsummed::trusted_as_summed))
            }
            false => {
                // The turn goes with the step, to be kept until the thread
                // that reads the file is done with it, whoever stops
                // waiting first.
                let loading = disk.loading().await.map_err(unreached)?;
                let read = disk.run_while_it_gets_on(SUMMED_AT_ONCE, move |progress| {
                    let _loading = loading;
                    let column = Column::map(&name, &path);
                    let told = |bytes| progress.made(bytes);
                    Ok(column.and_then(|column| column.summed_as(&name, file, told)))
                });
                read.await
            }
        };
        match summed {
            Ok(Ok(column)) => Ok(Some(column)),
            Ok(Err(error)) => {
                tracing::warn!(key, %error, "a vectors file on local disk is not read in place");
                Ok(None)
            }
            Err(error) => Err(unreached(error)),
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

/// How many pieces of work run at once off the async threads, all the
/// partitions of the process together, apart from what commits do there.
/// The runtime has so many threads for such work and no more (512 unless
/// it is built otherwise), and work that finds none free waits for one:
/// were loading and building segments to take them all, as many tenants
/// opened at once would, a commit would wait behind them with its writer
/// held. Kept to this many they leave the rest free for a commit's own
/// steps (see [`at_once`]).
///
/// A pool of threads apart for commits would do as much, but would be a
/// second thing to size, and time under test would no longer stand still
/// while a commit works on it, as the runtime's clock does for its own.
/// What else uses the runtime's threads is not counted here: a read of a
/// tenant, a request to a store on local disk. Against those nothing is
/// kept free, so a commit may still wait for a thread, which is then not
/// held against its writer (see [`Turn::at_once`]).
const WORKING_AT_ONCE: usize = 128;

fn working() -> Arc<tokio::sync::Semaphore> {
    static WORKING: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
    let working = || Arc::new(tokio::sync::Semaphore::new(WORKING_AT_ONCE));
    WORKING.get_or_init(working).clone()
}

/// Run `work`, which takes long enough to hold up the async threads, on
/// one of its own, in its turn among at most [`WORKING_AT_ONCE`]: kept
/// until the work ends, whoever stops waiting for it.
async fn blocking<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T> {
    let turn = working().acquire_owned().await;
    let turn = turn.map_err(|error| StorageError::Background(error.to_string()))?;
    at_once(move || {
        let _turn = turn;
        work()
    })
    .await
}

/// Run `work` off the async threads without waiting for a turn: for what
/// a commit does with its writer held, which must not wait behind
/// whatever else is being loaded or built.
async fn at_once<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> Result<T> {
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
/// The writer therefore never attempts a position twice: an attempt runs
/// from its first upload until it is applied in memory, and only one that
/// runs to the end and succeeds leaves the writer open. An error, or a
/// caller that stops waiting, leaves it closed for good.
///
/// Every try to commit is given up after a share of the lease, so a writer
/// needs a Tokio runtime with its time driver enabled. Its requests to the
/// store are given up as the store says, as
/// [`ObjectBlobs`](crate::blobs::ObjectBlobs) does by [`Transfers`], and its
/// steps on local disk as its [`Disk`] says.
#[derive(Debug)]
pub struct Writer {
    partition: Arc<Partition>,
    coordinator: Arc<dyn Coordinator>,
    lease: Lease,
    author: Author,
    vectorize: Option<Arc<dyn Vectorize>>,
    merging: Merging,
    /// How the store gives requests up, by which an attempt that outlives
    /// every bound on its steps is known to be stuck.
    transfers: Transfers,
    /// What tells the time of day when [`Self::clean`] asks how old a file
    /// is, which the store says the other half of: when it was written.
    clock: Arc<dyn Clock>,
    /// Held by whatever is deciding or making a commit, one at a time.
    writing: Mutex<()>,
    /// Set for good once an attempt fails or is given up part way.
    closed: AtomicBool,
    /// Held while a flush or a merge builds its segment without `writing`,
    /// so that one never builds on a state the other is about to replace.
    building: Mutex<()>,
    /// Set while a restore runs, when writes are refused.
    restoring: AtomicBool,
    /// When the turn now held, if any, was taken, and how long it may be
    /// held in all.
    held: Arc<std::sync::Mutex<Option<Holding>>>,
}

/// A turn being held: since when, how long it may be in all, since when
/// it has been waiting for a thread to work on, if it is, and how long
/// such a wait may be before it counts against the turn like any other.
#[derive(Debug, Clone, Copy)]
struct Holding {
    since: Instant,
    allowed: Duration,
    queued: Option<Instant>,
    unheld: Duration,
}

impl Holding {
    /// How long the turn may be held by now: what it was allowed, and the
    /// wait it is in for a thread, as far as that is not held against it.
    fn allowed_now(&self) -> Duration {
        let waiting = self
            .queued
            .map_or(Duration::ZERO, |queued| queued.elapsed());
        self.allowed.saturating_add(waiting.min(self.unheld))
    }
}

/// How long what a turn does in memory may take whatever the partition
/// holds, and how many live rows a second it gets through at the least.
/// The most a turn does in memory is to write a checkpoint's manifest and
/// adopt it, which walks every live row once. On a tenant of 4.3 million
/// live rows (`examples/apply_timing.rs`, twice, the slower with its
/// vectors in memory) the writer was held 0.07s and 0.22s by a flush of
/// 50,000 rows and 0.21s and 0.79s by a merge: at the slowest over five
/// million rows a second, of which this allows a twenty-fifth.
const IN_MEMORY_FLOOR: Duration = Duration::from_secs(5);
const IN_MEMORY_ROWS_PER_SECOND: u64 = 200_000;

/// How long a turn on a partition of `rows` live rows may spend in memory.
fn in_memory_within(rows: usize) -> Duration {
    IN_MEMORY_FLOOR + Duration::from_millis(rows as u64 * 1_000 / IN_MEMORY_ROWS_PER_SECOND)
}

/// How many mutations a second, and how many of their bytes, a commit gets
/// through at the least while it resolves them, encodes their log and
/// applies it. On the tenant above, with rows like its own, a group of
/// 2,310 mutations and one of 5,310, each of 12 MiB, were committed in
/// 0.15s and 0.16s, some 0.1s of it applying them, and one of 8,192
/// deletes in 0.02s: at the slowest fifteen thousand mutations and 76 MiB
/// a second, of which this allows less than a seventh and a nineteenth,
/// and both at once.
const GROUP_ROWS_PER_SECOND: u64 = 2_000;
const GROUP_BYTES_PER_SECOND: u64 = 4 * 1024 * 1024;

/// How long a commit of `weight` may spend on it in memory, over what any
/// turn may.
fn group_within(weight: Weight) -> Duration {
    let (rows, bytes) = (weight.rows as u64, weight.bytes as u64);
    Duration::from_millis(rows.saturating_mul(1_000) / GROUP_ROWS_PER_SECOND)
        + Duration::from_millis(bytes.saturating_mul(1_000) / GROUP_BYTES_PER_SECOND)
}

impl Writer {
    /// The most files one call to [`Self::clean`] deletes. A call that
    /// deleted as many may have left some, which another finds.
    pub const CLEANED_AT_ONCE: usize = 1_000;

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
            transfers: Transfers::default(),
            clock: Arc::new(SystemClock),
            writing: Mutex::new(()),
            closed: AtomicBool::new(false),
            building: Mutex::new(()),
            restoring: AtomicBool::new(false),
            held: Arc::default(),
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

    /// Take requests to the store to be given up as `transfers` says, the
    /// store's own, instead of as by default.
    pub fn transferring(self, transfers: Transfers) -> Self {
        Self { transfers, ..self }
    }

    /// Tell how old a file is by `clock` when cleaning up
    /// ([`Self::clean`]), instead of by the machine's.
    pub fn timed_by(self, clock: Arc<dyn Clock>) -> Self {
        Self { clock, ..self }
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

    /// Whether the writer can write no more: an attempt to commit failed,
    /// or was given up part way, so its outcome is unknown or the lease is
    /// lost. Anything else that fails leaves the writer as it was.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Wait until nothing else is committing, unless the writer is closed.
    /// From then until the turn is given back the writer is known to be
    /// held, and for how long it may be: what is done in memory while it
    /// is, which takes about as long as the partition has rows, and every
    /// step the turn then waits on, each of which allows itself its bound
    /// first (see [`Turn::allow`]).
    async fn turn(&self) -> Result<Turn<'_>> {
        let guard = self.writing.lock().await;
        if self.is_closed() {
            return Err(StorageError::Closed);
        }
        let (taken, in_memory) = (
            Instant::now(),
            in_memory_within(self.partition.read(State::rows)),
        );
        *self.held.lock().expect("held lock") = Some(Holding {
            since: taken,
            allowed: in_memory,
            queued: None,
            unheld: self.commit_within(),
        });
        Ok(Turn {
            _guard: guard,
            taken,
            held: self.held.clone(),
        })
    }

    /// Run one write attempt, closing the writer unless it runs to
    /// completion and succeeds.
    async fn attempt<T>(&self, work: impl Future<Output = Result<T>>) -> Result<T> {
        let mut attempt = Attempt {
            closed: &self.closed,
            succeeded: false,
        };
        let done = work.await?;
        attempt.succeeded = true;
        Ok(done)
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

    /// The collections `mutations` would make, were they committed now:
    /// none for one that is there already exactly so, and none at all if
    /// the batch would be refused, since none of a batch that is refused
    /// joins the draft. A commit made meanwhile may change that, and the
    /// commit of these still decides what is made.
    pub fn creates(&self, mutations: &[Mutation]) -> Vec<vellev_core::Collection> {
        self.partition.read(|state| {
            let mut draft = Draft::new(state);
            let _ = draft.batch(mutations.to_vec());
            let (ops, _) = draft.finish();
            ops.into_iter()
                .filter_map(|op| match op {
                    Op::CreateCollection { schema, .. } => Some(*schema),
                    _ => None,
                })
                .collect()
        })
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
    ///
    /// Whatever fails a batch without the writer trying to commit, a
    /// refusal, a vector not made or a restore under way, is that batch's
    /// own result. An error for the whole group means the writer is closed:
    /// it was already, or the commit failed or was given up part way; or,
    /// with nothing tried, that the runtime is shutting down.
    ///
    /// The batches are resolved, their log encoded and the commit applied
    /// in memory with the writer held, each on a thread for work that takes
    /// a while, never one of the async ones, and the writer is allowed as
    /// long for it as so many mutations and bytes take. How many share one
    /// commit is for the caller to bound, as [`GroupCommitter`] does.
    ///
    /// [`GroupCommitter`]: crate::GroupCommitter
    pub async fn commit_batches(
        &self,
        batches: Vec<Batch>,
        made: Embedded,
    ) -> Result<Vec<Result<Receipt>>> {
        let turn = self.turn().await?;
        if self.restoring.load(Ordering::SeqCst) {
            return Ok(batches
                .iter()
                .map(|_| Err(StorageError::Restoring))
                .collect());
        }
        let weight: Weight = batches.iter().map(Batch::weight).sum();
        turn.allow(group_within(weight));
        let batches = Arc::new(batches);
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
        let mut unembedded: Vec<Option<(String, Unembedded)>> = vec![None; batches.len()];
        let mut made: Embedded = made
            .into_iter()
            .filter_map(|((embedding, text), vector)| {
                let vector = embedding.checked(vector).ok()?;
                Some(((embedding, text), vector))
            })
            .collect();
        let mut failed: HashMap<Embedding, Unembedded> = HashMap::new();
        let ((position, prev), outcomes, ops) = loop {
            let decided: Vec<bool> = unembedded
                .iter()
                .zip(&earlier)
                .map(|(unembedded, earlier)| {
                    unembedded.is_some() || !matches!(earlier, Earlier::None)
                })
                .collect();
            let (partition, batches) = (self.partition.clone(), batches.clone());
            let resolved = turn.at_once(move || {
                partition.read(|state| {
                    let mut draft = Draft::new(state);
                    let outcomes: Vec<_> = batches
                        .iter()
                        .zip(decided)
                        .map(|(batch, decided)| match decided {
                            false => draft.batch(batch.mutations.clone()),
                            true => draft.batch(Vec::new()),
                        })
                        .collect();
                    (state.at(), outcomes, draft.finish())
                })
            });
            let (at, outcomes, (mut ops, wanted)) = resolved.await?;
            self.make(&turn, &wanted, &mut made, &mut failed).await;
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
                let why = failed[&wanted.embedding].clone();
                unembedded[wanted.batch] = Some((wanted.embedding.model.clone(), why));
            }
            // A vector that cannot be made for no batch of its own is the
            // failure of every batch still undecided, and the next round
            // commits what the rest decided.
            if rejected(&unembedded) == before {
                let (model, reason) = failed
                    .iter()
                    .next()
                    .map(|(embedding, reason)| (embedding.model.clone(), reason.clone()))
                    .unwrap_or_default();
                unembedded
                    .iter_mut()
                    .zip(&earlier)
                    .filter(|(unembedded, earlier)| {
                        unembedded.is_none() && matches!(earlier, Earlier::None)
                    })
                    .for_each(|(unembedded, _)| {
                        *unembedded = Some((model.clone(), reason.clone()));
                    });
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
            self.attempt(async {
                let position = following(position)?;
                let key = Layout(&self.lease.key).log(position, &self.author);
                let resolved = turn.taken.elapsed();
                let (log, frame) = turn
                    .at_once(move || {
                        let log = LogSegment {
                            position,
                            prev,
                            ops,
                            answered,
                        };
                        encode(Kind::Log, &log).map(|frame| (log, frame))
                    })
                    .await??;
                let (encoded, bytes) = (turn.taken.elapsed(), frame.len());
                self.publish(&turn, &key, frame, position, None, None)
                    .await?;
                let published = turn.taken.elapsed();
                // Applied to its end even if whoever waits here stops
                // waiting, which closes the writer: the commit is made.
                let (partition, applied) = (self.partition.clone(), key.clone());
                turn.at_once(move || {
                    let mut state = partition.state.write().expect("partition lock");
                    state.apply(&applied, &log)
                })
                .await??;
                tracing::debug!(
                    partition = %self.lease.key,
                    ?position,
                    mutations = weight.rows,
                    bytes,
                    ?resolved,
                    encoding = ?encoded.saturating_sub(resolved),
                    publishing = ?published.saturating_sub(encoded),
                    applying = ?turn.taken.elapsed().saturating_sub(published),
                    "committed a group"
                );
                Ok(position)
            })
            .await?
        };
        let mut results: Vec<Result<Receipt>> = outcomes
            .into_iter()
            .zip(unembedded)
            .map(|(outcomes, unembedded)| match unembedded {
                Some((model, reason)) => Err(StorageError::unembedded(model, reason)),
                None => outcomes.map(|outcomes| receipt(position, outcomes)),
            })
            .collect();
        for (at, earlier) in earlier.into_iter().enumerate() {
            results[at] = match earlier {
                Earlier::None => continue,
                Earlier::Answered(answer) => answer.map_err(StorageError::Refused),
                Earlier::InGroup(first) => told_again(&mut results[first]),
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
        turn: &Turn<'_>,
        wanted: &[Wanted],
        made: &mut Embedded,
        failed: &mut HashMap<Embedding, Unembedded>,
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
        if missing.values().any(|texts| !texts.is_empty()) {
            turn.allow(EMBEDDING_TIMEOUT);
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
                Err(why) => {
                    failed.insert(embedding.clone(), why);
                }
            }
        }
    }

    /// A vector the collection can hold for each of `texts`, or why not.
    /// A vector of another size than the collection's means the model is
    /// not the one it was made for; one that is not finite may be one
    /// text's misfortune.
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, Unembedded> {
        let vectorize = self.vectorize.as_ref().ok_or_else(|| {
            Unembedded::for_good(ModelFault::Setup, "this writer has nothing to embed with")
        })?;
        let asked = texts.len();
        let vectors = tokio::time::timeout(EMBEDDING_TIMEOUT, vectorize.vectors(embedding, texts))
            .await
            .map_err(|_| {
                Unembedded::for_now(format!("no vectors within {EMBEDDING_TIMEOUT:?}"))
            })??;
        if vectors.len() != asked {
            return Err(Unembedded::for_now(format!(
                "{} vectors for {asked} texts",
                vectors.len()
            )));
        }
        vectors
            .into_iter()
            .map(|vector| {
                embedding.checked(vector).map_err(|error| match error {
                    ObjectError::VectorLength { .. } => {
                        Unembedded::for_good(ModelFault::Provider, error.to_string())
                    }
                    _ => Unembedded::for_now(error.to_string()),
                })
            })
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
        let Some(snapshot) = self.partition.read(State::snapshot) else {
            return Ok(self.partition.read(State::position));
        };
        let mut staged = self.staging();
        let (snapshot, segment) = self.flushing(snapshot, &mut staged).await?;
        let turn = self.turn().await?;
        let position = following(self.partition.read(State::position))?;
        let key = Layout(&self.lease.key).manifest(position, &self.author);
        let manifest = self
            .partition
            .read(|state| state.flushed_manifest(position, &snapshot, segment.as_ref()))?;
        let frame = encode(Kind::Manifest, &manifest)?;
        staged.keep();
        self.checkpoint(turn, &key, frame, position, |state| {
            state.flushed(&key, position, snapshot, segment)
        })
        .await?;
        Ok(position)
    }

    /// Commit the checkpoint `frame` under `key` at `position` and adopt it
    /// in memory as `adopt` does, which is all a checkpoint does while the
    /// writer is held; then give the turn back and free what the checkpoint
    /// took out of memory. Once the commit is made nothing is left to fail:
    /// what `adopt` adopts was seen to be fit before the writer was held.
    async fn checkpoint(
        &self,
        turn: Turn<'_>,
        key: &str,
        frame: Bytes,
        position: LogPosition,
        adopt: impl FnOnce(&mut State) -> Discarded,
    ) -> Result<()> {
        let discarded = self
            .attempt(async {
                self.publish(&turn, key, frame, position, None, None)
                    .await?;
                let adopting = Instant::now();
                let adopted = adopt(&mut self.partition.state.write().expect("partition lock"));
                tracing::debug!(
                    partition = %self.lease.key,
                    ?position,
                    rows = self.partition.read(State::rows),
                    adopting = ?adopting.elapsed(),
                    held = ?turn.taken.elapsed(),
                    "adopted a checkpoint"
                );
                Ok(adopted)
            })
            .await?;
        drop(turn);
        blocking(move || drop(discarded)).await
    }

    /// What a checkpoint does before the writer is held: copy the rows of
    /// `snapshot` out of memory, a part at a time so that commits are not
    /// held up while all of them are, and build, upload and load back the
    /// segment of them, if there are any, which is then seen to be fit to
    /// adopt.
    async fn flushing(
        &self,
        mut snapshot: Snapshot,
        staged: &mut Staged,
    ) -> Result<(Snapshot, Option<Fit>)> {
        while self
            .partition
            .read(|state| state.copy_rows(&mut snapshot, COPIED_AT_ONCE))
        {
            tokio::task::yield_now().await;
        }
        let named_at = snapshot.position;
        let segment = match self.uploaded(&mut snapshot, named_at, staged).await? {
            Some(prepared) => Some(self.fit(prepared, &snapshot.catalog).await?),
            None => None,
        };
        Ok((snapshot, segment))
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
    /// named at the position the checkpoint is committed at, stays until
    /// committed.
    pub async fn restore(
        &self,
        head: Option<&ManifestRef>,
        position: LogPosition,
    ) -> Result<LogPosition> {
        self.put_back(head, position, None, None).await
    }

    /// As [`Self::restore`] to `head` at `restore.from`, recording `restore`
    /// with the commit that makes it, so that the coordinator answers its
    /// request as it was from then on; and, `adopting`, making the tenant
    /// of that lineage with it, as the coordinator was told it would, or
    /// not committing at all.
    pub async fn restore_recorded(
        &self,
        head: Option<&ManifestRef>,
        restore: Restore,
        adopting: Option<Lineage>,
    ) -> Result<LogPosition> {
        self.put_back(head, restore.from, Some(restore), adopting)
            .await
    }

    async fn put_back(
        &self,
        head: Option<&ManifestRef>,
        position: LogPosition,
        restore: Option<Restore>,
        adopting: Option<Lineage>,
    ) -> Result<LogPosition> {
        let _building = self.building.lock().await;
        let _refusing = Refusing::writes(self).await;
        self.abreast().await?;
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
        let mut staged = self.staging();
        // With writes refused and the building lock held nothing commits,
        // so the position the partition is at is the one it will be
        // restored from, and all but the commit is done before the writer
        // is held. The segment is named at the position the restore
        // commits, past the latest checkpoint: at that checkpoint's own, it
        // would be a young file there, and a clean would walk past it to
        // the checkpoint before, which an earlier clean may have deleted.
        let current = self.partition.read(State::position);
        let restored = following(current.max(position))?;
        let segment = match target.read(State::snapshot) {
            Some(mut snapshot) => {
                target.read(|state| state.copy_rows(&mut snapshot, usize::MAX));
                match self.uploaded(&mut snapshot, restored, &mut staged).await? {
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
        let key = Layout(&self.lease.key).manifest(restored, &self.author);
        let manifest = self.partition.read(|current| {
            target.read(|target| current.restored_manifest(restored, target, segment))
        });
        let frame = encode(Kind::Manifest, &manifest)?;
        let state = target.restore(&key, manifest, held).await?;
        let turn = self.turn().await?;
        // Were the partition anywhere else, what was made above would put
        // back over commits it never saw; nothing is committed, and the
        // segment uploaded for it is deleted.
        if self.partition.read(State::position) != current {
            return Err(StorageError::Diverged {
                head: self.partition.read(State::at).1,
            });
        }
        staged.keep();
        let replaced = self
            .attempt(async {
                self.publish(&turn, &key, frame, restored, restore, adopting)
                    .await?;
                Ok(std::mem::replace(
                    &mut *self.partition.state.write().expect("partition lock"),
                    state,
                ))
            })
            .await?;
        drop(turn);
        blocking(move || drop((replaced, target))).await?;
        Ok(restored)
    }

    /// Make ready to put the partition back as `source` holds it at `head`,
    /// at `restore.from`, from files that were in another store: a backup
    /// of this tenant or of the one it is to become, `adopting`. `source`
    /// answers by the keys the files had there, and nothing is written to
    /// it. The restore is made by [`RestoreFrom::commit`].
    ///
    /// Nothing is brought in under the key it had (DW-680): every segment,
    /// index and vectors file is copied into this partition's store under
    /// a key drawn for this restore, and the logs and checkpoint of the
    /// source's head are read into memory and never put. The state is
    /// loaded from those copies alone, read back from the store, as a node
    /// with nothing would load it.
    ///
    /// The copies are named at the position of the checkpoint the restore
    /// is to commit, past where the partition and the source are: never at
    /// the source's, which would put young files below what a clean has
    /// let go, and never at the partition's, which may be its latest
    /// checkpoint's own. A file young at a position has a clean keep the
    /// checkpoints from one before it on, and from one before the latest
    /// the walk would reach a checkpoint an earlier clean deleted.
    ///
    /// Writes are refused, and the building lock held, from the start
    /// until the restore is committed or fails. Dropped, the value made
    /// gives both up and deletes the copies.
    pub async fn restore_from<'a>(
        &'a self,
        source: Arc<dyn Blobs>,
        head: Option<&ManifestRef>,
        restore: Restore,
        adopting: Option<Lineage>,
        held: &dyn Held,
    ) -> Result<RestoreFrom<'a>> {
        let building = self.building.lock().await;
        let refusing = Refusing::writes(self).await;
        self.abreast().await?;
        // With writes refused and the building lock held nothing commits
        // and no checkpoint is made, so the copies, named at the position
        // of the checkpoint to be committed, lie past the latest one until
        // the restore is committed, as every upload not yet published does.
        let (current, rested_on) = self.partition.read(|state| {
            let rested_on = state.checkpoint().map(str::to_string);
            (state.position(), rested_on)
        });
        let restored = following(current.max(restore.from))?;
        let key = self.lease.key.clone();
        let at = Committed {
            generation: self.lease.generation,
            incarnation: self.partition.incarnation,
            log_position: restore.from,
            manifest: head.cloned(),
        };
        self.still(held)?;
        let chain = Partition::empty(key.clone(), source.clone(), self.partition.incarnation);
        let Tail { checkpoint, logs } = chain
            .tail(&at, LogPosition::default(), None)
            .await?
            .ok_or_else(|| {
                let reason = format!(
                    "the source's history at {:?} reaches neither a checkpoint nor the start",
                    restore.from
                );
                StorageError::corrupt(&Layout(&key).prefix(), reason)
            })?;
        let mut memory = HashMap::new();
        let mut sourced = HashSet::new();
        for (key, log) in &logs {
            memory.insert(key.clone(), encode(Kind::Log, log)?);
            sourced.insert(key.clone());
        }
        let mut staged = self.staging();
        let mut put = HashSet::new();
        if let Some((checkpoint_key, manifest)) = checkpoint {
            sourced.insert(checkpoint_key.clone());
            sourced.extend(listed_by(&manifest).map(str::to_string));
            // The logs it replays are read with it, by the keys they had.
            for log in &manifest.replay {
                self.still(held)?;
                memory.insert(log.clone(), chain.fetch(log).await?);
            }
            let copies: Vec<(SegmentMeta, Names)> = manifest
                .segments
                .iter()
                .map(|meta| (meta.clone(), self.named(restored, "-restored")))
                .collect();
            for (meta, names) in &copies {
                let named = [
                    meta.vectors.as_ref().map(|_| &names.vectors),
                    meta.index.as_ref().map(|_| &names.index),
                    Some(&names.segment),
                ];
                staged.keys.extend(named.into_iter().flatten().cloned());
            }
            use futures::{StreamExt, TryStreamExt};
            let indexing = Mutex::new(());
            let copied: Vec<Copied> = futures::stream::iter(copies)
                .map(|(meta, names)| self.copied(&*source, meta, names, &indexing, held))
                .buffered(SEGMENTS_COPIED_AT_ONCE)
                .try_collect()
                .await?;
            let mut fresh = HashMap::new();
            let mut dropped = HashSet::new();
            for Copied { files, unused } in copied {
                put.extend(files.iter().map(|(_, to)| to.clone()));
                fresh.extend(files);
                dropped.extend(unused);
            }
            let loading = loading(&checkpoint_key, manifest, &fresh, &dropped)?;
            memory.insert(checkpoint_key, encode(Kind::Manifest, &loading)?);
        }
        // The state is loaded from the copies alone, read back from the
        // store and summed, as a node with nothing would read them: the
        // one check that each came out whole.
        let overlay = Overlay {
            memory,
            put: put.clone(),
            primary: self.partition.blobs.clone(),
        };
        let target = Partition::load_reusing(key, Arc::new(overlay), &at, None).await?;
        Ok(RestoreFrom {
            writer: self,
            _building: building,
            _refusing: refusing,
            staged,
            target,
            put,
            source: sourced,
            rested_on,
            current,
            restored,
            restore,
            adopting,
        })
    }

    /// `Unheld` unless `held` still is.
    fn still(&self, held: &dyn Held) -> Result<()> {
        match held.still() {
            true => Ok(()),
            false => Err(StorageError::Unheld(self.lease.key.tenant.clone())),
        }
    }

    /// Copy the segment `meta` names in `source`, with its vectors and its
    /// index, under `names`: the vectors and the segment a part at a time,
    /// the segment last, as an upload puts it, and the index, read whole
    /// and given the copy's key, one at a time of all a restore copies, by
    /// `indexing`. An index that cannot be read as one, or that indexes
    /// another segment, is passed over, as every load of the source passed
    /// it over; one the source cannot give fails the copy, as any file it
    /// cannot give does.
    async fn copied(
        &self,
        source: &dyn Blobs,
        meta: SegmentMeta,
        names: Names,
        indexing: &Mutex<()>,
        held: &dyn Held,
    ) -> Result<Copied> {
        let blobs = &self.partition.blobs;
        let mut files = Vec::new();
        let mut unused = None;
        if let Some(vectors) = meta.vectors {
            self.still(held)?;
            blobs
                .put_parts(&names.vectors, source.parts(&vectors))
                .await?;
            files.push((vectors, names.vectors));
        }
        if let Some(index) = meta.index {
            self.still(held)?;
            let _one = indexing.lock().await;
            let frame = source.get(&index).await?;
            match rekeyed_index(&index, &frame, &meta.key, &names.segment) {
                Ok(rekeyed) => {
                    drop(frame);
                    blobs.put(&names.index, rekeyed).await?;
                    files.push((index, names.index));
                }
                Err(error) => {
                    tracing::warn!(key = index, %error, "a segment's index file is not copied");
                    unused = Some(index);
                }
            }
        }
        self.still(held)?;
        blobs
            .put_parts(&names.segment, source.parts(&meta.key))
            .await?;
        files.push((meta.key, names.segment));
        Ok(Copied { files, unused })
    }

    /// See that the partition is committed just where this writer has it,
    /// and under its lease, as it is for the one writer of a lease still
    /// current.
    ///
    /// A writer is left behind in two ways. Its lease is taken by another
    /// node, which the coordinator fences: every commit of the old
    /// generation is refused. Or a second writer is opened on the very
    /// lease it holds, which no generation tells apart; a node opens one
    /// writer for each lease it takes, so only something that opens two on
    /// one lease, as a test or a tool might, makes that happen, and then
    /// each commit is refused unless it follows what is committed (see
    /// [`Commit`]). Either way the coordinator refuses the commit, and
    /// this is not what keeps a writer left behind from committing. It
    /// tells one so before it has built and uploaded a restore's segment
    /// for nothing, with its writer left open; and it is what cleaning up,
    /// which commits nothing, goes by (see [`Self::current`]).
    async fn abreast(&self) -> Result<()> {
        let (position, head) = self.partition.read(State::at);
        let within = self.commit_within();
        let asked = self.coordinator.committed(&self.lease.key);
        let committed = tokio::time::timeout(within, asked).await.map_err(|_| {
            let reason = format!("the coordinator did not say what is committed within {within:?}");
            CoordinatorError::backend(std::io::Error::new(std::io::ErrorKind::TimedOut, reason))
        })??;
        let here = committed.is_some_and(|committed| {
            let at = committed.manifest.as_ref().map(|head| head.0.as_str());
            committed.generation == self.lease.generation
                && committed.log_position == position
                && at == head.as_deref()
        });
        match here {
            true => Ok(()),
            false => Err(StorageError::Diverged { head }),
        }
    }

    /// Build and upload a segment of the rows taken out of `snapshot` and
    /// its tombstones, its files named at `named_at`; none if it has
    /// neither.
    async fn uploaded(
        &self,
        snapshot: &mut Snapshot,
        named_at: LogPosition,
        staged: &mut Staged,
    ) -> Result<Option<Prepared>> {
        if snapshot.rows.is_empty() && snapshot.tombstones.is_empty() {
            return Ok(None);
        }
        let names = self.named(named_at, "");
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
        self.upload(&prepared, staged).await?;
        Ok(Some(prepared))
    }

    /// The files this writer uploads for a commit it has yet to attempt,
    /// none so far.
    fn staging(&self) -> Staged {
        Staged {
            blobs: self.partition.blobs.clone(),
            keys: Vec::new(),
        }
    }

    /// Names for the files of a segment at `position`, apart from those of
    /// any other build there, each ending in `marker`. That is at or past
    /// the latest checkpoint: a flush's or a merge's at the position of the
    /// partition it was built from, a restore's at the position it commits.
    fn named(&self, position: LogPosition, marker: &str) -> Names {
        let nonce = uuid::Uuid::new_v4().as_u64_pair().0;
        let suffix = format!("-{nonce:016x}{marker}");
        let layout = Layout(&self.lease.key);
        let named = |kind| layout.named(kind, position, &self.author, &suffix);
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
            let indexes = state::indexed(
                &catalog,
                &file.rows,
                inline,
                &Stored::default(),
                |_, _| {},
                above,
            );
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
    /// sits in the store without its companions, noting each in `staged`
    /// before it goes up.
    async fn upload(&self, prepared: &Prepared, staged: &mut Staged) -> Result<()> {
        let Prepared { meta, built, index } = prepared;
        let files = [
            meta.vectors.as_deref().map(|key| (key, &built.column)),
            meta.index.as_deref().zip(index.as_ref()),
            Some((meta.key.as_str(), &built.segment)),
        ];
        for (key, bytes) in files.into_iter().flatten() {
            staged.keys.push(key.to_string());
            self.partition.blobs.put(key, bytes.clone()).await?;
        }
        Ok(())
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

    /// As [`Self::loaded`], and seen to be fit to adopt under the
    /// collections of `catalog`: here, before the writer is held, so that
    /// a segment that is not fails with nothing committed or changed.
    async fn fit(&self, prepared: Prepared, catalog: &Catalog) -> Result<Fit> {
        let (meta, loaded) = self.loaded(prepared).await?;
        loaded.fit(meta, catalog)
    }

    /// Carry out the merge the partition most needs, if any: replace a run
    /// of segments with one holding their live rows, in a checkpoint that
    /// also flushes what is unflushed, as [`Self::flush`] does. Both
    /// segments are built from the partition as it is now and written while
    /// commits go on; a row that dies meanwhile is dead in the new segment
    /// too. Returns the checkpoint's position, or `None` when there was
    /// nothing to merge.
    pub async fn merge(&self) -> Result<Option<LogPosition>> {
        let _building = self.building.lock().await;
        let Some(merge) = self.partition.read(|state| state.merge_plan(&self.merging)) else {
            return Ok(None);
        };
        let above = self.merging.clustered_above;
        let Some((file, shape, named_at)) = self.partition.read(|state| {
            let named_at = state.position();
            let file = state.merged(&merge, named_at)?;
            let shape = Shape::of(state.catalog(), &file.rows, file.tombstones.len(), above);
            Some((file, shape, named_at))
        }) else {
            return Ok(None);
        };
        let catalog = self.partition.read(|state| state.catalog().clone());
        let built = self.built(file, catalog.clone()).await?;
        // Both segments, however large, are uploaded without the writer,
        // under names of their own at the position each was built from;
        // see `clean` for why nothing deletes them first.
        let mut staged = self.staging();
        let merged = Self::prepared(self.named(named_at, "-merged").meta(shape), built)?;
        self.upload(&merged, &mut staged).await?;
        let merged = self.fit(merged, &catalog).await?;
        let snapshot = self.partition.read(State::taken);
        let (snapshot, segment) = self.flushing(snapshot, &mut staged).await?;
        let turn = self.turn().await?;
        let position = following(self.partition.read(State::position))?;
        let key = Layout(&self.lease.key).manifest(position, &self.author);
        // Only what holds the building lock replaces segments, so the run
        // is as it was planned over. Were it not, the merged segment would
        // describe nothing there is: nothing is committed, and what was
        // uploaded is deleted.
        let Some(manifest) = self.partition.read(|state| {
            let flushed = segment.as_ref();
            state.merged_manifest(position, &snapshot, flushed, &merge, merged.meta.clone())
        })?
        else {
            return Ok(None);
        };
        let frame = encode(Kind::Manifest, &manifest)?;
        staged.keep();
        self.checkpoint(turn, &key, frame, position, |state| {
            state.compacted(&merge, merged);
            state.flushed(&key, position, snapshot, segment)
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
    /// `grace` is taken as it is given, and nothing here checks it: it is
    /// the whole margin between this and a restore to a point of the
    /// partition's restore window, which names files this takes for litter
    /// once they are older. Whoever calls this with less than that window
    /// and, on top, the time a file may be in the store before its commit,
    /// a restore may wait behind a call of this, and the clocks may differ
    /// by, may delete files such a restore needs, and that is the caller's
    /// breach. The first of those is as long as a publish may take: the
    /// head file's upload, one request to the store for a file of one part
    /// and as many as its parts and three more for a larger one, plus its
    /// copy to local disk, which is made once the store has it, plus three
    /// tries to commit, each a quarter of the lease or as long as the
    /// coordinator is given if that is less, and the waits between them.
    /// The second is the minute a call of this is given to find what to
    /// delete, in which it reads the clock: it may hold the building lock
    /// for longer, but deletes only what it found then. A reader of a
    /// committed position has that window and the margin to fetch its
    /// files in, each given two requests a part
    /// ([`Transfers::read_within`]), which a file of several parts may
    /// outlast where the window is short. A node refuses a margin under
    /// a quarter of an hour, or under three requests to the store, which
    /// covers a head file of one part under a lease and a coordinator's
    /// bound near their defaults and no more; nor does any margin cover a
    /// clock wrong by more than it, or a flush or a merge that holds the
    /// building lock for longer. Deciding this by the coordinator's rows
    /// instead of by time is the clean floor's, a change of its own.
    ///
    /// What is needed is decided by the committed chain alone: it is walked
    /// back to the latest checkpoint before the oldest recent file, and
    /// everything before that checkpoint which no checkpoint on the way
    /// lists goes. A restore's checkpoint lists segments far older than
    /// itself, which stay for as long as it is on the way.
    ///
    /// What a clean has deleted stays deleted, and nothing else is taken
    /// for it (DW-681). Each clean deletes the checkpoint before the one it
    /// deletes below, and a later one that keeps more, because a file under
    /// an old position is young to it or because it is given more grace or
    /// reckons by a clock behind, walks back to that checkpoint. Before its
    /// first delete a clean therefore records, in the coordinator and as
    /// the holder of its lease, the checkpoint it deletes below, and the
    /// record is only ever raised; so every checkpoint a clean has deleted
    /// lies below the one recorded. A walk that has passed the recorded
    /// checkpoint and finds a checkpoint's predecessor missing ends there,
    /// as a walk whose floor were that checkpoint would. Anywhere else a
    /// predecessor missing is a file lost by something other than a clean,
    /// and the clean fails for it and deletes nothing, as does one whose
    /// predecessor cannot be read, or whose record cannot be. A record that
    /// names no checkpoint of this chain is never passed, and is said so.
    ///
    /// Nothing younger than `grace` by this node's clock against the
    /// store's date is deleted, whatever the chain says: a walk that ends
    /// at what an earlier clean deleted has young files below it, which a
    /// writer left behind or a put that landed late put there, or a restore
    /// from a backup that fetches files back before it commits them.
    /// The files are listed first, and the head read after, with the
    /// writer held for just that reading so that no commit is in flight: a
    /// file this writer uploaded and had not yet published when the
    /// listing was taken is reached by then or belongs to a commit that
    /// failed.
    ///
    /// A flush, a merge and a restore upload their segments' files without
    /// the writer, under a name of their own: a flush's and a merge's under
    /// the position of the partition they were built from, a restore's
    /// under the position it commits, which is past that. All of them, like
    /// this, hold the building lock from before they read the partition
    /// until they commit or fail. Only they commit checkpoints, so none is
    /// committed past the partition's position while such files wait to be
    /// published, and log commits do not move the checkpoint this deletes
    /// below. Those files therefore sit at or past the latest checkpoint,
    /// never below it, and are not deleted until a checkpoint after them
    /// has been committed and, if they were never published, until they
    /// are older than `grace`.
    ///
    /// The building lock is held until the last file is deleted, not only
    /// while what to delete is worked out. Let go before, a restore could
    /// commit meanwhile a checkpoint that lists segments older than the
    /// one this deletes below, which nothing on the way listed when the
    /// chain was walked, and they would be deleted from under it. So that
    /// a flush, a merge or a restore never waits long for it, one call
    /// deletes at most [`Self::CLEANED_AT_ONCE`] files, several at a time,
    /// and stops asking for more after a minute; what is left goes with
    /// the next call, which finds it again. Finding what to delete is
    /// given a minute too, and a call that takes longer fails.
    pub async fn clean(&self, grace: Duration) -> Result<usize> {
        let _building = self.building.lock().await;
        let unneeded = tokio::time::timeout(CLEANING_WITHIN, self.unneeded(grace));
        let Unneeded { orphans, below } = unneeded.await.map_err(|_| StorageError::Blob {
            key: Layout(&self.lease.key).prefix(),
            source: format!("what to clean up was not found within {CLEANING_WITHIN:?}").into(),
        })??;
        if let (Some(below), false) = (below, orphans.is_empty()) {
            self.recorded(below).await?;
        }
        // A few at a time, each lot only once the coordinator has said
        // again that the partition is this writer's still and where it has
        // it: see `current`.
        let blobs = &self.partition.blobs;
        let given_up = tokio::time::sleep(CLEANING_WITHIN);
        tokio::pin!(given_up);
        let mut deleted = 0;
        for lot in orphans.chunks(CLEANING_AT_ONCE) {
            self.current().await?;
            let deleting = try_join_all(lot.iter().map(|key| blobs.delete(key)));
            tokio::select! {
                biased;
                () = &mut given_up => break,
                done = deleting => done.map(|_| deleted += lot.len())?,
            }
        }
        tracing::debug!(partition = %self.lease.key, deleted, "cleaned");
        Ok(deleted)
    }

    /// Where the partition is, once the coordinator has said that it is
    /// this writer's still and committed just there: asked with the writer
    /// held, so that no commit of its own is under way.
    ///
    /// Cleaning up commits nothing, so no coordinator refuses it, and what
    /// it deletes is decided by the chain as its writer has it. A writer
    /// whose lease another has taken has not seen what that one committed:
    /// a restore since may have put back into use segments this one takes
    /// for litter. So it asks before it works out what to delete, and
    /// again before each lot it deletes. What is left is a lot already
    /// asked for when the lease is lost, and only if, before those
    /// requests are answered, another writer has taken the lease, opened
    /// the partition and committed a restore to a state that lists one of
    /// those very files: a lease is taken from a node only once it has run
    /// out, by when the node has long stopped whatever was working on the
    /// tenant, this included.
    async fn current(&self) -> Result<(LogPosition, Option<String>)> {
        let turn = self.turn().await?;
        turn.allow(self.commit_within());
        self.abreast().await?;
        Ok(self.partition.read(State::at))
    }

    /// See that the coordinator holds a record at or above the checkpoint
    /// a clean is about to delete below, raising it if it is lower, before
    /// anything is deleted: asked as the lots are, with the partition seen
    /// to be this writer's still. A record it answers with that names
    /// another checkpoint at or above this one is said so: the partition
    /// then keeps to what a clean with no record does.
    async fn recorded(&self, below: Below) -> Result<()> {
        let Below {
            base,
            position,
            recorded,
            passed,
        } = below;
        let raised = match recorded {
            Some(recorded) if recorded.position >= position => recorded,
            _ => {
                self.current().await?;
                let within = self.commit_within();
                let asked = self.coordinator.clean_below(&self.lease, &base, position);
                tokio::time::timeout(within, asked).await.map_err(|_| {
                    let reason =
                        format!("the coordinator did not record a clean within {within:?}");
                    CoordinatorError::backend(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        reason,
                    ))
                })??
            }
        };
        // A record on this chain at or above the base is the base itself,
        // or one the walk passed on its way down to it.
        if raised.base != base && !passed {
            tracing::warn!(
                partition = %self.lease.key,
                recorded = %raised.base.0,
                base = %base.0,
                "what a clean recorded it deletes below is no checkpoint on this chain; \
                 a clean here takes no missing checkpoint for a clean's work until it is \
                 passed"
            );
        }
        Ok(())
    }

    /// What the coordinator records a clean of this partition deleted
    /// below, if any: a clean that cannot be told fails.
    async fn cleaned(&self) -> Result<Option<Cleaned>> {
        let within = self.commit_within();
        let asked = self.coordinator.cleaned(&self.lease.key);
        Ok(tokio::time::timeout(within, asked).await.map_err(|_| {
            let reason = format!("the coordinator did not say what was cleaned within {within:?}");
            CoordinatorError::backend(std::io::Error::new(std::io::ErrorKind::TimedOut, reason))
        })??)
    }

    /// The files [`Self::clean`] with `grace` deletes, as many as it does
    /// at once: every file listed, set against the committed chain. Each
    /// request it makes is given up by the store, but a partition of a
    /// great many files, or of a long chain, takes many, so whoever asks
    /// gives the whole of it up after a while.
    async fn unneeded(&self, grace: Duration) -> Result<Unneeded> {
        let layout = Layout(&self.lease.key);
        let listed = self.partition.blobs.list(&layout.prefix()).await?;
        let (position, head) = self.current().await?;
        let recorded = self.cleaned().await?;
        let cutoff = chrono::Duration::from_std(grace)
            .ok()
            .and_then(|grace| self.clock.now().checked_sub_signed(grace));
        let Some(cutoff) = cutoff else {
            return Ok(Unneeded::default());
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
        let mut checkpoints: Vec<(String, Manifest)> = Vec::new();
        // Whether the walk has passed the checkpoint a clean recorded it
        // deleted below: past it, and only there, a checkpoint missing is a
        // clean's work.
        let mut passed = false;
        let base = loop {
            let Some(key) = cursor else { break None };
            let frame = match self.partition.fetch(&key).await {
                Err(StorageError::Missing { .. }) if passed => {
                    tracing::debug!(
                        partition = %self.lease.key,
                        missing = key,
                        "a clean's walk ends where an earlier clean deleted"
                    );
                    break checkpoints.pop();
                }
                fetched => fetched?,
            };
            cursor = match decode_head(&key, &frame)? {
                Head::Log(log) => log.prev,
                Head::Manifest(manifest) if manifest.position <= floor => {
                    break Some((key, manifest))
                }
                Head::Manifest(manifest) => {
                    passed |= recorded
                        .as_ref()
                        .is_some_and(|recorded| recorded.base.0 == key);
                    let prev = manifest.prev.clone();
                    checkpoints.push((key, manifest));
                    prev
                }
            };
        };
        let Some((base_key, base)) = base else {
            return Ok(Unneeded::default());
        };
        let kept: HashSet<&str> = checkpoints
            .iter()
            .map(|(_, manifest)| manifest)
            .chain([&base])
            .flat_map(listed_by)
            .collect();
        let orphans = files
            .iter()
            .filter(|(file, at)| {
                *at < base.position && file.modified < cutoff && !kept.contains(file.key.as_str())
            })
            .map(|(file, _)| file.key.clone())
            .take(Self::CLEANED_AT_ONCE)
            .collect();
        Ok(Unneeded {
            orphans,
            below: Some(Below {
                base: ManifestRef(base_key),
                position: base.position,
                recorded,
                passed,
            }),
        })
    }

    /// Whether something holds the writer now, as whatever commits does
    /// while it does. For tests that watch what is done meanwhile.
    #[doc(hidden)]
    pub fn is_held(&self) -> bool {
        self.held.lock().expect("held lock").is_some()
    }

    /// Whether the writer has been held for longer than everything done
    /// while it was may take, which only something that ignores its bound,
    /// such as a store that never gives a request up, could make it be.
    /// A commit waiting a while for a thread to work on is not that: see
    /// [`Turn::at_once`].
    pub fn stalled(&self) -> bool {
        let held = *self.held.lock().expect("held lock");
        held.is_some_and(|held| held.since.elapsed() > held.allowed_now())
    }

    /// How long one try to commit may take: a share of the lease, so that
    /// the tries and the waits between them end within it.
    fn commit_within(&self) -> Duration {
        self.lease.ttl / (COMMIT_ATTEMPTS + 1)
    }

    /// The most a publish of a head file of `bytes` takes: its upload and
    /// its copy to local disk, then every try to commit and the waits
    /// between them.
    fn publish_within(&self, bytes: usize) -> Duration {
        let waits = (1..COMMIT_ATTEMPTS)
            .map(|tries| COMMIT_RETRY_DELAY * tries)
            .sum();
        let head = self.transfers.upload_within(bytes) + Disk::shared().answers_within(bytes);
        head + self.commit_within() * COMMIT_ATTEMPTS + waits
    }

    /// Upload the head file under `key` and commit it at `position`, with
    /// the restore it makes if it makes one, and the lineage it is
    /// `adopting` if it takes a backup up. The upload is given up as the
    /// store says, and each try to commit after [`Self::commit_within`];
    /// one given up is as unknown as one whose answer was lost.
    async fn publish(
        &self,
        turn: &Turn<'_>,
        key: &str,
        frame: Bytes,
        position: LogPosition,
        restore: Option<Restore>,
        adopting: Option<Lineage>,
    ) -> Result<()> {
        turn.allow(self.publish_within(frame.len()));
        self.partition.blobs.put(key, frame).await?;
        // The state this was built on, which nothing has changed since: the
        // writer is held from before it was read until this is applied.
        let (was_at, head) = self.partition.read(State::at);
        let commit = Commit {
            after: Some(Prior {
                position: was_at,
                head: head.map(ManifestRef),
            }),
            log_position: position,
            manifest: Some(ManifestRef(key.to_string())),
            restore,
            adopting,
        };
        let within = self.commit_within();
        let unanswered = || {
            let reason = format!("the coordinator did not answer a commit within {within:?}");
            CoordinatorError::backend(std::io::Error::new(std::io::ErrorKind::TimedOut, reason))
        };
        let mut tries = 1;
        loop {
            let tried =
                tokio::time::timeout(within, self.coordinator.commit(&self.lease, commit.clone()))
                    .await
                    .unwrap_or_else(|_| Err(unanswered()));
            match tried {
                Err(error) if error.is_retryable() && tries < COMMIT_ATTEMPTS => {
                    tokio::time::sleep(COMMIT_RETRY_DELAY * tries).await;
                    tries += 1;
                }
                outcome => return outcome.map(|_| ()).map_err(Into::into),
            }
        }
    }
}

/// What a clean is to delete, and the checkpoint it deletes below with
/// what the coordinator recorded when it began.
#[derive(Default)]
struct Unneeded {
    orphans: Vec<String>,
    below: Option<Below>,
}

/// The checkpoint a clean deletes below, at its position; what the
/// coordinator recorded a clean deleted below when this one began; and
/// whether this one's walk passed that checkpoint.
struct Below {
    base: ManifestRef,
    position: LogPosition,
    recorded: Option<Cleaned>,
    passed: bool,
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

/// A write attempt under way, which closes the writer as it ends unless it
/// succeeded: a failure and a future dropped part way alike.
struct Attempt<'a> {
    closed: &'a AtomicBool,
    succeeded: bool,
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        if !self.succeeded {
            self.closed.store(true, Ordering::SeqCst);
        }
    }
}

/// The writer held, from when it was taken until this is dropped.
struct Turn<'a> {
    _guard: MutexGuard<'a, ()>,
    taken: Instant,
    held: Arc<std::sync::Mutex<Option<Holding>>>,
}

impl Turn<'_> {
    /// Allow the turn `more` for a step it is about to wait on.
    fn allow(&self, more: Duration) {
        if let Some(held) = self.held.lock().expect("held lock").as_mut() {
            held.allowed = held.allowed.saturating_add(more);
        }
    }

    /// Run `work`, a step of the commit this turn is for, off the async
    /// threads and ahead of whatever waits its turn there (see
    /// [`at_once`]). Those threads are the runtime's, which reads and a
    /// store on local disk use too, so there may be none free all the
    /// same. The wait for one is not counted against the turn, for up to
    /// as long as one try to commit may take: nothing of the writer's is
    /// stuck meanwhile, and giving its lease up for it would only add an
    /// opening of the tenant to a node already busy. A wait longer than
    /// that is no longer a busy node's: whatever keeps every thread, as
    /// reads stuck on a disk that stopped answering would, keeps this
    /// writer from writing, and the rest of the wait is held against it
    /// like anything else it waits on, so that the lease is given up and
    /// another node takes the tenant.
    async fn at_once<T: Send + 'static>(
        &self,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<T> {
        let (held, taken) = (self.held.clone(), self.taken);
        if let Some(held) = self.held.lock().expect("held lock").as_mut() {
            held.queued = Some(Instant::now());
        }
        at_once(move || {
            let mut holding = held.lock().expect("held lock");
            if let Some(held) = holding.as_mut().filter(|held| held.since == taken) {
                held.allowed = held.allowed_now();
                held.queued = None;
            }
            drop(holding);
            work()
        })
        .await
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        *self.held.lock().expect("held lock") = None;
    }
}

/// Files uploaded for a commit not yet attempted. Dropped before it is, by
/// a failure or by a caller that stops waiting, they are deleted on a task
/// of its own, which holds up nothing: nothing names them or ever will.
/// Once the commit is attempted they are kept whatever comes of it, for it
/// may have been made.
struct Staged {
    blobs: Arc<dyn Blobs>,
    keys: Vec<String>,
}

impl Staged {
    /// Keep the files: the commit that names them is about to be tried.
    fn keep(&mut self) {
        self.keys.clear();
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        let (false, Ok(runtime)) = (self.keys.is_empty(), tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let (blobs, keys) = (self.blobs.clone(), std::mem::take(&mut self.keys));
        runtime.spawn(async move {
            for key in keys {
                if let Err(error) = blobs.delete(&key).await {
                    tracing::warn!(key, %error, "a file never published could not be deleted");
                }
            }
        });
    }
}

/// A batch's result, as a later batch of the same request in its group
/// is answered: alike. A refusal or a vector not made is told again as it
/// is, as neither is a failure of the commit; any other failure, which the
/// batch's own result then shares, is told as shared.
fn told_again(first: &mut Result<Receipt>) -> Result<Receipt> {
    let failure = match first {
        Ok(receipt) => return Ok(receipt.clone()),
        Err(failure) => failure,
    };
    let refusal = failure.refusal();
    Err(match (failure, refusal) {
        (
            StorageError::Embedding {
                model,
                reason,
                lasting,
            },
            _,
        ) => StorageError::Embedding {
            model: model.clone(),
            reason: reason.clone(),
            lasting: *lasting,
        },
        (StorageError::Shared(shared), _) => StorageError::Shared(shared.clone()),
        (_, Some(refusal)) => StorageError::Refused(refusal),
        (other, None) => {
            let shared = Arc::new(std::mem::replace(other, StorageError::Closed));
            *other = StorageError::Shared(shared.clone());
            StorageError::Shared(shared)
        }
    })
}

/// Writes refused for as long as this lives.
struct Refusing<'a>(&'a AtomicBool);

impl<'a> Refusing<'a> {
    /// Refuse `writer`'s writes from now on, once any commit under way has
    /// been made. A caller that stops waiting for that commit refuses them
    /// no longer.
    async fn writes(writer: &'a Writer) -> Refusing<'a> {
        writer.restoring.store(true, Ordering::SeqCst);
        let refusing = Refusing(&writer.restoring);
        drop(writer.writing.lock().await);
        refusing
    }
}

impl Drop for Refusing<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// A restore from another store made ready by [`Writer::restore_from`]:
/// its copies made, and the state it puts back loaded from them, with
/// writes refused and the building lock held. Committed, or dropped, which
/// deletes the copies.
pub struct RestoreFrom<'a> {
    writer: &'a Writer,
    _building: MutexGuard<'a, ()>,
    _refusing: Refusing<'a>,
    staged: Staged,
    /// The state put back, loaded from the copies.
    target: Partition,
    /// Every file the restore put, and the keys the source's files had.
    put: HashSet<String>,
    source: HashSet<String>,
    /// The checkpoint the partition's head rested on when it began, and
    /// the position it was at.
    rested_on: Option<String>,
    current: LogPosition,
    /// The position of the checkpoint it is to commit, its files' too.
    restored: LogPosition,
    restore: Restore,
    adopting: Option<Lineage>,
}

/// A segment a restore copied: each of its files by the key it had and
/// the one it was given, and its index file if that was passed over.
struct Copied {
    files: Vec<(String, String)>,
    unused: Option<String>,
}

impl RestoreFrom<'_> {
    /// Commit the restore: the state loaded, with whatever the source's
    /// logs left in memory written as one more segment, as a checkpoint
    /// past where the partition is and where the source was, resting on
    /// the partition's own checkpoint and replaying nothing. Its position.
    ///
    /// The checkpoint is held to DW-680's rule before it is published, and
    /// one that breaks it is refused with nothing committed (see
    /// [`kept_out`]). Taken by value, so that the copies are not deleted
    /// while the commit is tried: from then on they are kept whatever comes
    /// of it, for it may have been made.
    pub async fn commit(self) -> Result<LogPosition> {
        let Self {
            writer,
            _building,
            _refusing,
            mut staged,
            target,
            mut put,
            source,
            rested_on,
            current,
            restored,
            restore,
            adopting,
        } = self;
        let mut held = target.held();
        let segment = match target.read(State::snapshot) {
            Some(mut snapshot) => {
                target.read(|state| state.copy_rows(&mut snapshot, usize::MAX));
                match writer
                    .uploaded(&mut snapshot, restored, &mut staged)
                    .await?
                {
                    Some(prepared) => {
                        let meta = &prepared.meta;
                        let named = [meta.index.as_ref(), meta.vectors.as_ref()];
                        put.extend(named.into_iter().flatten().cloned());
                        put.insert(meta.key.clone());
                        let (meta, loaded) = writer.loaded(prepared).await?;
                        held.insert(meta.key.clone(), loaded);
                        Some(meta)
                    }
                    None => None,
                }
            }
            None => None,
        };
        let key = Layout(&writer.lease.key).manifest(restored, &writer.author);
        let manifest = writer
            .partition
            .read(|now| target.read(|target| now.restored_manifest(restored, target, segment)));
        kept_out(&manifest, &put, &source, rested_on.as_deref())?;
        let frame = encode(Kind::Manifest, &manifest)?;
        let state = target.restore(&key, manifest, held).await?;
        let turn = writer.turn().await?;
        // Nothing moves the partition while the building lock is held and
        // writes are refused, and a commit that follows anything else the
        // coordinator refuses: this only says so before the copies are
        // kept, as `put_back` does, and no test can tell it from that.
        if writer.partition.read(State::position) != current {
            return Err(StorageError::Diverged {
                head: writer.partition.read(State::at).1,
            });
        }
        staged.keep();
        let replaced = writer
            .attempt(async {
                writer
                    .publish(&turn, &key, frame, restored, Some(restore), adopting)
                    .await?;
                Ok(std::mem::replace(
                    &mut *writer.partition.state.write().expect("partition lock"),
                    state,
                ))
            })
            .await?;
        drop(turn);
        blocking(move || drop((replaced, target))).await?;
        Ok(restored)
    }
}

/// The checkpoint a restore from another store loads what it copied
/// through, made of the source's `checkpoint`, read from under `key`: each
/// segment, index and vectors file by the key `fresh` gives it, the index
/// of each segment whose index is among `dropped` none, no checkpoint
/// before it, and the logs to replay as the source has them.
///
/// The map is what says which files were copied, and is held to that: one
/// that gives a file the key it had, or that names a log or a checkpoint,
/// which are never copied, is refused; and so is one that lacks a key for
/// any file the checkpoint names (see [`Manifest::remapped`]), but for the
/// logs to replay, which are read from memory by the keys they had: they
/// cannot be named anew, for each says by key the head it follows.
fn loading(
    key: &str,
    checkpoint: Manifest,
    fresh: &HashMap<String, String>,
    dropped: &HashSet<String>,
) -> Result<Manifest> {
    for (old, new) in fresh {
        if old == new {
            let reason = format!("{old} is given the key it had");
            return Err(StorageError::corrupt(key, reason));
        }
        for named in [old, new] {
            if matches!(named.rsplit('/').nth(1), Some("log" | "manifest")) {
                let reason = format!("{named} is a log or a checkpoint, which is never copied");
                return Err(StorageError::corrupt(key, reason));
            }
        }
    }
    let mut checkpoint = checkpoint;
    checkpoint.prev = None;
    for segment in &mut checkpoint.segments {
        if segment
            .index
            .as_ref()
            .is_some_and(|index| dropped.contains(index))
        {
            segment.index = None;
        }
    }
    let replay = std::mem::take(&mut checkpoint.replay);
    let mut loading = checkpoint.remapped(key, fresh)?;
    loading.replay = replay;
    Ok(loading)
}

/// What a restore from another store loads the state it puts back
/// through: the files of the source's head chain, held in memory under the
/// keys they had, its checkpoint as [`loading`] made it; and the files the
/// restore put, under the keys it gave them, from the partition's store.
/// Nothing else is read, and nothing written: a file of the source's that
/// the partition's store holds still under the key it had is not one of
/// the restore's, and is refused.
#[derive(Debug)]
struct Overlay {
    memory: HashMap<String, Bytes>,
    put: HashSet<String>,
    primary: Arc<dyn Blobs>,
}

impl Overlay {
    /// Refused unless the file under `key` is one the restore put.
    fn put_by_the_restore(&self, key: &str) -> Result<()> {
        match self.put.contains(key) {
            true => Ok(()),
            false => Err(StorageError::corrupt(
                key,
                "a restore from another store reads only the source's head and its own copies",
            )),
        }
    }
}

/// The refusal of a write to what a restore loads through.
fn loads_only(key: &str) -> StorageError {
    StorageError::corrupt(
        key,
        "a restore from another store writes nothing as it loads",
    )
}

#[async_trait::async_trait]
impl Blobs for Overlay {
    async fn put(&self, key: &str, _bytes: Bytes) -> Result<()> {
        Err(loads_only(key))
    }
    async fn put_parts(
        &self,
        key: &str,
        _parts: futures::stream::BoxStream<'_, Result<Bytes>>,
    ) -> Result<()> {
        Err(loads_only(key))
    }
    async fn get(&self, key: &str) -> Result<Bytes> {
        if let Some(bytes) = self.memory.get(key) {
            return Ok(bytes.clone());
        }
        self.put_by_the_restore(key)?;
        self.primary.get(key).await
    }
    async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
        Err(loads_only(prefix))
    }
    async fn delete(&self, key: &str) -> Result<()> {
        Err(loads_only(key))
    }
    async fn local(&self, key: &str) -> Result<Option<Local>> {
        if self.memory.contains_key(key) {
            return Ok(None);
        }
        self.put_by_the_restore(key)?;
        self.primary.local(key).await
    }
    fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
        use futures::StreamExt;
        if let Some(bytes) = self.memory.get(key) {
            return futures::stream::once(std::future::ready(Ok(bytes.clone()))).boxed();
        }
        match self.put_by_the_restore(key) {
            Ok(()) => self.primary.parts(key),
            Err(refused) => futures::stream::once(std::future::ready(Err(refused))).boxed(),
        }
    }
    async fn fetch_local(&self, key: &str) -> Result<Option<Local>> {
        if self.memory.contains_key(key) {
            return Ok(None);
        }
        self.put_by_the_restore(key)?;
        self.primary.fetch_local(key).await
    }
}

/// Whether the checkpoint a restore from another store is to commit, as
/// `manifest`, keeps DW-680's rule (K4).
///
/// It holds when the restore, which put `put`, put no key the `source`'s
/// files had; every segment, index and vectors file the checkpoint names
/// is one the restore put; the checkpoint replays no log; and it rests on
/// `rested_on`, the checkpoint the partition rested on when the restore
/// began, which may be one the source has too.
fn kept_out(
    manifest: &Manifest,
    put: &HashSet<String>,
    source: &HashSet<String>,
    rested_on: Option<&str>,
) -> Result<()> {
    let key = format!("the checkpoint at {:?}", manifest.position);
    let broken = |reason: String| {
        let reason = format!("DW-680 K4: {reason}");
        Err(StorageError::corrupt(&key, reason))
    };
    if let Some(again) = put.iter().find(|key| source.contains(*key)) {
        return broken(format!(
            "the restore put {again}, a key the source's files had"
        ));
    }
    let named = manifest.segments.iter().flat_map(|segment| {
        let files = [segment.index.as_ref(), segment.vectors.as_ref()];
        files.into_iter().flatten().chain([&segment.key])
    });
    if let Some(unput) = named.into_iter().find(|key| !put.contains(*key)) {
        return broken(format!("it names {unput}, which the restore did not put"));
    }
    if !manifest.replay.is_empty() {
        return broken(format!("it replays {:?}", manifest.replay));
    }
    if manifest.prev.as_deref() != rested_on {
        return broken(format!(
            "it rests on {:?}, and the partition rested on {rested_on:?}",
            manifest.prev
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blobs::ObjectBlobs;
    use crate::coordinator::{Acquired, MemoryCoordinator};
    use object_store::memory::InMemory;
    use vellev_core::{
        Bm25, Collection, CollectionName, DataType, Distance, EmbeddingProvider, NodeId, ObjectId,
        Property, TenantId,
    };

    fn notes() -> CollectionName {
        CollectionName::new("notes").unwrap()
    }

    fn create() -> Mutation {
        Mutation::CreateCollection(Box::new(Collection {
            name: notes(),
            properties: vec![Property::new("n", DataType::Int)],
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
        }))
    }

    fn upsert(n: u8) -> Mutation {
        Mutation::Upsert {
            collection: notes(),
            id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
            properties: serde_json::Map::from_iter([("n".to_string(), n.into())]),
            vector: Some(vec![f32::from(n), 0.5]),
            only_if_version: None,
        }
    }

    const MERGING: Merging = Merging {
        stale: 0.5,
        fanout: 2,
        rows: 1_000_000,
        clustered_above: state::CLUSTERED_ABOVE,
    };

    /// Blobs that tell of each segment they are given and keep it waiting
    /// until let go.
    #[derive(Debug)]
    struct Gated {
        inner: ObjectBlobs,
        kind: &'static str,
        waiting: tokio::sync::Notify,
        go_on: tokio::sync::Notify,
    }

    impl Gated {
        /// Over a store in memory, keeping files whose keys hold `kind`.
        fn keeping(kind: &'static str) -> Arc<Self> {
            Arc::new(Self {
                inner: ObjectBlobs::new(Arc::new(InMemory::new())),
                kind,
                waiting: tokio::sync::Notify::new(),
                go_on: tokio::sync::Notify::new(),
            })
        }
    }

    #[async_trait::async_trait]
    impl Blobs for Gated {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
            if key.contains(self.kind) {
                self.waiting.notify_one();
                self.go_on.notified().await;
            }
            self.inner.put(key, bytes).await
        }
        async fn get(&self, key: &str) -> Result<Bytes> {
            self.inner.get(key).await
        }
        async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
            self.inner.list(prefix).await
        }
        async fn delete(&self, key: &str) -> Result<()> {
            self.inner.delete(key).await
        }
    }

    #[tokio::test]
    async fn a_restore_made_ready_for_a_partition_that_then_moved_commits_nothing() {
        let gated = Gated::keeping("/segment/");
        let (writer, coordinator) = writer_on(gated.clone()).await;
        let writer = Arc::new(writer);
        writer.commit(vec![create(), upsert(1)]).await.unwrap();
        let committed = coordinator.committed(&writer.lease.key).await;
        let committed = committed.unwrap().unwrap();

        // While the restore's segment goes up the partition moves, which
        // nothing but a fault in the writer could make it do.
        let restoring = tokio::spawn({
            let (writer, head) = (writer.clone(), committed.manifest.clone());
            async move { writer.restore(head.as_ref(), committed.log_position).await }
        });
        gated.waiting.notified().await;
        let moved = following(committed.log_position).unwrap();
        let state = &writer.partition.state;
        state.write().unwrap().moved_to(moved);
        gated.go_on.notify_one();

        let refused = restoring.await.unwrap().unwrap_err();
        assert!(
            matches!(refused, StorageError::Diverged { .. }),
            "{refused}"
        );
        assert!(!writer.is_closed(), "nothing was tried");
        let after = coordinator.committed(&writer.lease.key).await;
        assert_eq!(after.unwrap().unwrap(), committed);
    }

    /// A writer of a partition of its own, in memory, which merges two
    /// segments as soon as there are two.
    async fn writer() -> Writer {
        let blobs = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        writer_on(blobs).await.0
    }

    /// As [`writer`], over `blobs`, with its coordinator.
    async fn writer_on(blobs: Arc<dyn Blobs>) -> (Writer, Arc<MemoryCoordinator>) {
        let key = PartitionKey {
            tenant: TenantId::new("tenant-a").unwrap(),
            partition: vellev_core::Partition::DEFAULT,
        };
        let coordinator = Arc::new(MemoryCoordinator::default());
        coordinator.register(&key).await.unwrap();
        let (holder, ttl) = (NodeId::generate("node"), Duration::from_secs(30));
        let Acquired::Lease { lease, committed } =
            coordinator.acquire(&key, &holder, ttl).await.unwrap()
        else {
            panic!("a fresh partition is free to take");
        };
        let writer = Writer::open(blobs, coordinator.clone(), lease, &committed).await;
        (writer.unwrap().merging_with(MERGING), coordinator)
    }

    #[test]
    fn a_commit_waiting_for_a_thread_to_work_on_is_not_taken_for_stuck() {
        // One thread for work off the async ones, and a clock that moves
        // only when told.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let writer = Arc::new(writer().await);
            writer.commit(vec![create()]).await.unwrap();

            // The thread is taken by something else, and an hour passes.
            let (let_go, kept) = std::sync::mpsc::channel::<()>();
            let kept = tokio::task::spawn_blocking(move || {
                let _ = kept.recv();
            });
            let committing = tokio::spawn({
                let writer = writer.clone();
                async move { writer.commit(vec![upsert(1)]).await }
            });
            while !writer.is_held() {
                tokio::task::yield_now().await;
            }
            // Past the five seconds a turn has in memory, and within the
            // quarter of its lease of thirty that a wait for a thread is
            // not held against it.
            tokio::time::advance(Duration::from_secs(10)).await;
            assert!(writer.is_held() && !committing.is_finished());
            assert!(!writer.stalled(), "it waits for a thread, and is not stuck");
            // Past that too: whatever keeps the thread keeps the writer
            // from writing, and it is stuck like any other.
            tokio::time::advance(Duration::from_secs(5)).await;
            assert!(writer.stalled(), "no thread in all that time");

            let_go.send(()).unwrap();
            kept.await.unwrap();
            committing.await.unwrap().unwrap();
            assert!(!writer.is_closed() && !writer.stalled());
        });
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_turn_at_a_large_file_is_kept_until_its_thread_is_done_whoever_stops_waiting() {
        let dir = tempfile::tempdir().unwrap();
        let disk = Disk::new(Duration::from_secs(60), 20 * 1024 * 1024, 4)
            .loading_at_once(1)
            .unwrap();
        let store = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let cached = crate::blobs::CachedBlobs::new(store, dir.path()).on(disk.clone());
        let blobs: Arc<dyn Blobs> = Arc::new(cached);
        let (writer, coordinator) = writer_on(blobs.clone()).await;
        writer.commit(vec![create(), upsert(1)]).await.unwrap();
        writer.flush().await.unwrap();
        let key = writer.lease.key.clone();
        let committed = coordinator.committed(&key).await.unwrap().unwrap();
        let vectors = writer
            .partition
            .read(|state| state.segments().next().cloned());
        let copy = dir.path().join(vectors.unwrap().vectors.unwrap());

        // The copy is a pipe nobody writes to: whoever opens it waits.
        std::fs::remove_file(&copy).unwrap();
        let made = std::process::Command::new("mkfifo").arg(&copy).status();
        assert!(made.unwrap().success());
        let opening = tokio::spawn(async move { Partition::load(key, blobs, &committed).await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        opening.abort();
        assert!(opening.await.unwrap_err().is_cancelled());

        // Nobody waits for it any more, and its thread reads on: the one
        // turn on the disk is its own still.
        let another = tokio::time::timeout(Duration::from_millis(300), disk.loading());
        assert!(another.await.is_err(), "the turn is kept");
        // Until the thread is done.
        drop(std::fs::File::options().read(true).write(true).open(&copy));
        let freed = tokio::time::timeout(Duration::from_secs(30), disk.loading());
        drop(freed.await.expect("the turn is let go").unwrap());
    }

    #[tokio::test]
    async fn vectors_are_read_in_their_turn_among_the_large_files_of_their_disk() {
        let dir = tempfile::tempdir().unwrap();
        let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 4)
            .loading_at_once(1)
            .unwrap();
        let store = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let cached = crate::blobs::CachedBlobs::new(store, dir.path()).on(disk.clone());
        let blobs: Arc<dyn Blobs> = Arc::new(cached);
        let (writer, coordinator) = writer_on(blobs.clone()).await;
        writer.commit(vec![create(), upsert(1)]).await.unwrap();
        writer.flush().await.unwrap();
        let key = writer.lease.key.clone();
        let committed = coordinator.committed(&key).await.unwrap().unwrap();

        // The one turn there is on the disk is taken: the vectors of a
        // partition opened meanwhile wait for it, their file being one
        // nobody has summed as it lies now.
        let vectors = writer
            .partition
            .read(|state| state.segments().next().cloned());
        let copy = dir.path().join(vectors.unwrap().vectors.unwrap());
        let touched = std::fs::File::options().write(true).open(&copy).unwrap();
        let earlier = std::time::SystemTime::now() - Duration::from_secs(60);
        touched.set_modified(earlier).unwrap();
        let turn = disk.loading().await.unwrap();
        let (again, committed_again) = ((key.clone(), blobs.clone()), committed.clone());
        let opening = tokio::spawn(async move { Partition::load(key, blobs, &committed).await });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!opening.is_finished(), "it waits its turn");
        drop(turn);
        opening.await.unwrap().unwrap();

        // Summed once, it is not summed again while it lies as it did:
        // opened again, it waits for no turn, though none is to be had.
        let _taken = disk.loading().await.unwrap();
        let (key, blobs) = again;
        let opening = Partition::load(key, blobs, &committed_again);
        let opened = tokio::time::timeout(Duration::from_secs(30), opening).await;
        opened.expect("it needs no turn").unwrap();
    }

    #[test]
    fn a_wait_for_a_thread_to_work_on_is_allowed_for_once_the_thread_is_had() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let gated = Gated::keeping("/log/");
            let (writer, _) = writer_on(gated.clone()).await;
            let writer = Arc::new(writer);
            let allowed = |writer: &Writer| {
                let held = writer.held.lock().unwrap();
                held.map(|held| (held.allowed, held.queued.is_some()))
            };

            // Five seconds with the one thread taken by something else,
            // within what a wait for one may be.
            let (let_go, kept) = std::sync::mpsc::channel::<()>();
            let kept = tokio::task::spawn_blocking(move || {
                let _ = kept.recv();
            });
            let committing = tokio::spawn({
                let writer = writer.clone();
                async move { writer.commit(vec![create()]).await }
            });
            while !writer.is_held() {
                tokio::task::yield_now().await;
            }
            let (before, waiting) = allowed(&writer).unwrap();
            assert!(waiting);
            tokio::time::advance(Duration::from_secs(5)).await;
            let_go.send(()).unwrap();
            kept.await.unwrap();

            // The thread is had, and the commit has got as far as storing
            // its log: the five seconds it waited are no longer a wait it
            // is in, but time it was allowed, over what the steps since
            // allowed themselves.
            gated.waiting.notified().await;
            let (after, waiting) = allowed(&writer).unwrap();
            assert!(!waiting);
            let since = writer.publish_within(0);
            assert!(
                after >= before + Duration::from_secs(5) + since,
                "{after:?}"
            );
            assert!(!writer.stalled());
            gated.go_on.notify_one();
            committing.await.unwrap().unwrap();
        });
    }

    #[test]
    fn a_commit_never_waits_behind_work_that_waits_its_turn() {
        // Two threads more than what loads and builds may take between
        // them, and far more of that asked for than may run.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(WORKING_AT_ONCE + 2)
            .build()
            .unwrap();
        runtime.block_on(async {
            let writer = writer().await;
            let (let_go, kept) = std::sync::mpsc::channel::<()>();
            let kept = Arc::new(std::sync::Mutex::new(kept));
            let waiting: Vec<_> = (0..WORKING_AT_ONCE * 2)
                .map(|_| {
                    let kept = kept.clone();
                    tokio::spawn(blocking(move || {
                        let _ = kept.lock().unwrap().recv();
                    }))
                })
                .collect();
            tokio::time::sleep(Duration::from_millis(200)).await;

            let committing = writer.commit(vec![create(), upsert(1)]);
            let committed = tokio::time::timeout(Duration::from_secs(30), committing).await;
            committed.expect("a thread was free for it").unwrap();
            drop(let_go);
            for waited in waiting {
                waited.await.unwrap().unwrap();
            }
        });
    }

    #[tokio::test]
    async fn a_merge_planned_over_segments_since_replaced_makes_no_manifest() {
        let (writer, merging) = (writer().await, MERGING);
        writer.commit(vec![create()]).await.unwrap();
        for n in 0..2 {
            writer.commit(vec![upsert(n)]).await.unwrap();
            writer.flush().await.unwrap();
        }

        let planned = writer.partition.read(|state| state.merge_plan(&merging));
        let planned = planned.expect("two segments to merge");
        // The manifest a merge over that run would commit, of whatever
        // segment stands in for the merged one.
        let manifest = |state: &State| {
            let merged = state.segments().next().unwrap().clone();
            let position = following(state.position()).unwrap();
            let merged = state.merged_manifest(position, &state.taken(), None, &planned, merged);
            merged.unwrap()
        };
        assert!(writer.partition.read(manifest).is_some());
        writer.merge().await.unwrap().expect("a merge");
        assert!(
            writer.partition.read(manifest).is_none(),
            "the run is no longer the segments it was planned over"
        );
    }

    #[tokio::test]
    async fn a_flush_the_state_could_not_adopt_makes_no_manifest() {
        let writer = writer().await;
        writer.commit(vec![create(), upsert(1)]).await.unwrap();
        let position = following(writer.partition.read(State::position)).unwrap();
        let taken = writer.partition.read(State::taken);
        let refused = |snapshot: &Snapshot| {
            let made = |state: &State| state.flushed_manifest(position, snapshot, None);
            writer.partition.read(made).unwrap_err().to_string()
        };
        // A snapshot that took a row, and no segment of it.
        assert!(refused(&taken).contains("does not hold the rows"));

        // The row dies; one written since is not the flush's to drop.
        writer.commit(vec![upsert(1), upsert(2)]).await.unwrap();
        let none_taken = writer.partition.read(State::taken).taking_none();
        assert!(refused(&none_taken).contains("still live"));
        let of_more = none_taken.of(usize::MAX);
        assert!(refused(&of_more).contains("more than there is"));

        // A flush is committed while another was building.
        writer.flush().await.unwrap();
        assert!(refused(&taken).contains("a checkpoint was committed"));
        let now = writer.partition.read(State::taken);
        let manifest = |state: &State| state.flushed_manifest(position, &now, None);
        assert!(writer.partition.read(manifest).is_ok());
    }

    /// Keys of the partition's files, of each kind, at `position`.
    fn file(kind: &str, name: &str) -> String {
        format!("tenant-a/0/{kind}/{name}.{kind}")
    }

    fn segment(name: &str, index: bool, vectors: bool) -> SegmentMeta {
        SegmentMeta {
            key: file("segment", name),
            rows: 3,
            tombstones: 1,
            index: index.then(|| file("index", name)),
            vectors: vectors.then(|| file("vectors", name)),
        }
    }

    /// A checkpoint of two segments, the first with its index, both with
    /// their vectors, with a checkpoint before it and logs to replay.
    fn checkpoint(first: &str) -> Manifest {
        Manifest {
            position: LogPosition(7),
            prev: Some(file("manifest", "before")),
            next_collection_id: 3,
            collections: Vec::new(),
            segments: vec![segment(first, true, true), segment("second", false, true)],
            replay: vec![file("log", "six"), file("log", "seven")],
            replay_after: LogPosition(5),
            answered: Vec::new(),
        }
    }

    /// Each of `manifest`'s segment, index and vectors files given a key
    /// of its own, ending in "-new".
    fn fresh_for(manifest: &Manifest) -> HashMap<String, String> {
        let named = manifest.segments.iter().flat_map(|segment| {
            [
                Some(&segment.key),
                segment.index.as_ref(),
                segment.vectors.as_ref(),
            ]
            .into_iter()
            .flatten()
        });
        named
            .map(|key| (key.clone(), key.replace('.', "-new.")))
            .collect()
    }

    /// The checkpoint a restore from a source loads through, of each kind
    /// of checkpoint a source may be at: every segment, index and vectors
    /// file by its new key, a dropped index none, nothing before it, and
    /// the logs to replay exactly as the source has them.
    #[test]
    fn a_restore_from_a_source_loads_through_its_checkpoint_named_anew() {
        let restored = Manifest {
            replay: Vec::new(),
            replay_after: LogPosition::default(),
            ..checkpoint("first")
        };
        let cases: [(&str, Manifest, bool); 4] = [
            ("a flush", checkpoint("first"), false),
            ("a merge", checkpoint("first-merged"), false),
            ("a restore", restored, false),
            (
                "a flush, its first index dropped",
                checkpoint("first"),
                true,
            ),
        ];
        for (what, source, drop_index) in cases {
            let mut fresh = fresh_for(&source);
            let mut dropped = HashSet::new();
            let index = source.segments[0].index.clone().unwrap();
            if drop_index {
                fresh.remove(&index);
                dropped.insert(index.clone());
            }
            let key = file("manifest", "source");
            let loading = loading(&key, source.clone(), &fresh, &dropped).unwrap();
            assert_eq!(loading.prev, None, "{what}");
            assert_eq!(loading.replay, source.replay, "{what}");
            assert_eq!(loading.replay_after, source.replay_after, "{what}");
            assert_eq!(loading.position, source.position, "{what}");
            assert_eq!(
                loading.next_collection_id, source.next_collection_id,
                "{what}"
            );
            assert_eq!(loading.segments.len(), source.segments.len(), "{what}");
            for (new, old) in loading.segments.iter().zip(&source.segments) {
                assert_eq!(new.key, fresh[&old.key], "{what}");
                assert_eq!(
                    new.vectors,
                    old.vectors.as_ref().map(|key| fresh[key].clone())
                );
                assert_eq!((new.rows, new.tombstones), (old.rows, old.tombstones));
            }
            let first = &loading.segments[0];
            match drop_index {
                true => assert_eq!(first.index, None, "{what}"),
                false => assert_eq!(first.index.as_ref(), Some(&fresh[&index]), "{what}"),
            }
        }
    }

    /// A change made to the map a restore loads through.
    type Change<'a> = Box<dyn Fn(&mut HashMap<String, String>) + 'a>;

    /// The map a restore from a source loads through is what says which
    /// files it copied: one that lacks a segment, an index or vectors
    /// file, or that gives a file the key it had, or that names a log or
    /// a checkpoint, which are never copied, is refused.
    #[test]
    fn a_restore_from_a_source_refuses_a_map_of_anything_but_its_copies() {
        let source = checkpoint("first");
        let key = file("manifest", "source");
        let none = HashSet::new();
        let first = source.segments[0].clone();
        let cases: [(&str, Change<'_>); 7] = [
            (
                "lacking a segment",
                Box::new(|fresh| drop(fresh.remove(&first.key))),
            ),
            (
                "lacking an index",
                Box::new(|fresh| drop(fresh.remove(first.index.as_ref().unwrap()))),
            ),
            (
                "lacking vectors",
                Box::new(|fresh| drop(fresh.remove(first.vectors.as_ref().unwrap()))),
            ),
            (
                "a segment kept under its key",
                Box::new(|fresh| drop(fresh.insert(first.key.clone(), first.key.clone()))),
            ),
            (
                "a log",
                Box::new(|fresh| {
                    let log = file("log", "six");
                    fresh.insert(log.clone(), file("log", "six-new"));
                }),
            ),
            (
                "a checkpoint",
                Box::new(|fresh| {
                    let before = file("manifest", "before");
                    fresh.insert(before, file("manifest", "before-new"));
                }),
            ),
            (
                "a log kept under its key",
                Box::new(|fresh| {
                    let log = file("log", "six");
                    fresh.insert(log.clone(), log);
                }),
            ),
        ];
        for (what, change) in cases {
            let mut fresh = fresh_for(&source);
            change(&mut fresh);
            let refused = loading(&key, source.clone(), &fresh, &none);
            assert!(
                matches!(refused, Err(StorageError::Corrupt { .. })),
                "{what}: {refused:?}"
            );
        }
    }

    /// DW-680's rule as the commit of a restore from a source holds it to:
    /// it put no key the source's files had; every segment, index and
    /// vectors file the checkpoint names is one it put; the checkpoint
    /// replays nothing; and it rests on the checkpoint the partition did,
    /// which may be one the source has too.
    #[test]
    fn a_restore_from_a_source_commits_nothing_that_breaks_dw_680s_rule() {
        let committed = Manifest {
            prev: Some(file("manifest", "rested-on")),
            replay: Vec::new(),
            replay_after: LogPosition::default(),
            ..checkpoint("first-new")
        };
        let put: HashSet<String> = fresh_for(&committed).into_keys().collect();
        let source: HashSet<String> = [file("segment", "first"), file("manifest", "rested-on")]
            .into_iter()
            .collect();
        let rested_on = Some(file("manifest", "rested-on"));
        let held = |manifest: &Manifest, put: &HashSet<String>, rested: Option<&String>| {
            kept_out(manifest, put, &source, rested.map(String::as_str))
        };
        assert!(held(&committed, &put, rested_on.as_ref()).is_ok());
        let mut unput = put.clone();
        unput.remove(&committed.segments[1].key);
        let mut put_again = put.clone();
        put_again.insert(file("segment", "first"));
        let replaying = Manifest {
            replay: vec![file("log", "six")],
            ..committed.clone()
        };
        let elsewhere = Some(file("manifest", "other"));
        let broken = [
            (
                "a key it did not put",
                held(&committed, &unput, rested_on.as_ref()),
            ),
            (
                "a key the source had",
                held(&committed, &put_again, rested_on.as_ref()),
            ),
            (
                "a log to replay",
                held(&replaying, &put, rested_on.as_ref()),
            ),
            (
                "another checkpoint before",
                held(&committed, &put, elsewhere.as_ref()),
            ),
            ("none before", held(&committed, &put, None)),
        ];
        for (what, broken) in broken {
            assert!(
                matches!(&broken, Err(StorageError::Corrupt { reason, .. }) if reason.contains("DW-680")),
                "{what}: {broken:?}"
            );
        }
    }

    /// A store a restore loads through, over `primary`, holding `memory`
    /// and the files `put`.
    fn overlay(primary: Arc<dyn Blobs>, memory: &[(&str, &str)], put: &[&str]) -> Overlay {
        Overlay {
            memory: memory
                .iter()
                .map(|(key, bytes)| (key.to_string(), Bytes::from(bytes.to_string())))
                .collect(),
            put: put.iter().map(|key| key.to_string()).collect(),
            primary,
        }
    }

    /// What a restore from a source loads through: the source's head chain
    /// from memory, even where the store holds a file under the same key;
    /// the files the restore put from the store; and nothing else, even
    /// what the store holds under a key the source had. Nothing is written.
    #[tokio::test]
    async fn a_restore_from_a_source_loads_only_its_chain_and_its_copies() {
        let primary: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let (chain, copy, old) = (file("log", "a"), file("segment", "b"), file("segment", "c"));
        for key in [&chain, &copy, &old] {
            primary.put(key, Bytes::from("in the store")).await.unwrap();
        }
        let overlay = overlay(primary.clone(), &[(&chain, "in memory")], &[&copy]);

        assert_eq!(overlay.get(&chain).await.unwrap(), "in memory");
        assert_eq!(overlay.get(&copy).await.unwrap(), "in the store");
        use futures::{StreamExt, TryStreamExt};
        let parts: Vec<Bytes> = overlay.parts(&copy).try_collect().await.unwrap();
        assert_eq!(parts.concat(), b"in the store");
        let parts: Vec<Bytes> = overlay.parts(&chain).try_collect().await.unwrap();
        assert_eq!(parts.concat(), b"in memory");
        for key in [old.as_str(), "tenant-a/0/segment/nowhere.segment"] {
            assert!(overlay.get(key).await.is_err(), "{key}");
            let parts: Result<Vec<Bytes>> = overlay.parts(key).try_collect().await;
            assert!(parts.is_err(), "{key}");
            assert!(overlay.local(key).await.is_err(), "{key}");
            assert!(overlay.fetch_local(key).await.is_err(), "{key}");
        }
        assert!(overlay.put(&copy, Bytes::from("x")).await.is_err());
        let parts = futures::stream::iter([Ok(Bytes::from("x"))]).boxed();
        assert!(overlay.put_parts(&copy, parts).await.is_err());
        assert!(overlay.delete(&copy).await.is_err());
        assert!(overlay.list("").await.is_err());
        assert_eq!(primary.get(&copy).await.unwrap(), "in the store");
    }

    /// A file the restore put is read in place from the store's copy on
    /// local disk, fetched there a part at a time, as any vectors are.
    #[tokio::test]
    async fn a_restore_from_a_source_reads_its_copies_through_the_stores_copy_on_local_disk() {
        let dir = tempfile::tempdir().unwrap();
        let inner: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let primary: Arc<dyn Blobs> = Arc::new(crate::blobs::CachedBlobs::new(inner, dir.path()));
        let copy = file("vectors", "b");
        primary.put(&copy, Bytes::from("vectors")).await.unwrap();
        primary.forget_local("tenant-a/0/").await.unwrap();
        let overlay = overlay(primary, &[], &[&copy]);

        assert!(overlay.local(&copy).await.unwrap().is_none());
        let fetched = overlay.fetch_local(&copy).await.unwrap().unwrap();
        assert_eq!(std::fs::read(&fetched.path).unwrap(), b"vectors");
        assert!(overlay.local(&copy).await.unwrap().is_some());
    }

    /// A restore from a source whose checkpoint would name a file it did
    /// not put, or rest on another checkpoint than the partition did, is
    /// refused at its commit, which commits nothing: whatever made it so.
    #[tokio::test]
    async fn a_restore_from_a_source_that_would_break_dw_680s_rule_commits_nothing() {
        let (source, _) = writer_on(Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))).await;
        source.commit(vec![create(), upsert(1)]).await.unwrap();
        source.flush().await.unwrap();
        source.commit(vec![upsert(2)]).await.unwrap();
        let (at, head) = source.partition.read(State::at);
        let restore = || Restore {
            request: vellev_core::RequestId::generate(),
            asked: [0; 32],
            from: at,
            from_at: chrono::Utc::now(),
        };
        struct Held;
        #[async_trait::async_trait]
        impl crate::backups::Held for Held {
            async fn held(&self) -> Result<bool> {
                Ok(true)
            }
            fn still(&self) -> bool {
                true
            }
        }
        let head = head.map(ManifestRef);
        let (writer, coordinator) =
            writer_on(Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))).await;
        writer.commit(vec![create(), upsert(9)]).await.unwrap();
        let before = coordinator.committed(&writer.lease.key).await.unwrap();
        type Tamper = fn(&mut RestoreFrom<'_>);
        let tampered: [(&str, Tamper); 2] = [
            ("names a file it did not put", |ready| ready.put.clear()),
            ("rests elsewhere", |ready| {
                ready.rested_on = Some(file("manifest", "x"))
            }),
        ];
        let stored = || async {
            let listed = writer.partition.blobs.list("").await.unwrap();
            listed
                .into_iter()
                .map(|file| file.key)
                .collect::<HashSet<_>>()
        };
        for (what, tamper) in tampered {
            let had = stored().await;
            let blobs = source.partition.blobs.clone();
            let ready = writer.restore_from(blobs, head.as_ref(), restore(), None, &Held);
            let mut ready = ready.await.unwrap();
            tamper(&mut ready);
            let refused = ready.commit().await;
            assert!(
                matches!(&refused, Err(StorageError::Corrupt { reason, .. }) if reason.contains("DW-680")),
                "{what}: {refused:?}"
            );
            let after = coordinator.committed(&writer.lease.key).await.unwrap();
            assert_eq!(after, before, "{what}");
            assert!(!writer.is_closed(), "{what}: nothing was tried");
            // Refused before they were kept, its copies are deleted.
            let started = std::time::Instant::now();
            while stored().await != had {
                assert!(
                    started.elapsed() < Duration::from_secs(30),
                    "{what}: copies kept"
                );
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}
