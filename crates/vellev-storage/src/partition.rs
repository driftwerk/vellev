use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use bytes::Bytes;
use futures::future::{join_all, try_join_all};
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::{Mutex, MutexGuard};
use tokio::time::Instant;
use vellev_core::{Embedding, ModelFault, ObjectError};

use crate::backups::Held;
use crate::blobs::{is_plain, Author, Blobs, Disk, Layout, Linked, Listed, Local, Transfers};
use crate::catalog::Catalog;
use crate::column::{self, Column, Gathering};
use crate::coordinator::{
    Cleaned, Clock, Commit, Committed, Coordinator, CoordinatorError, Generation, Incarnation,
    Lease, Lineage, LogPosition, ManifestRef, PartitionKey, Prior, Restore, SystemClock,
};
use crate::draft::{Draft, Wanted};
use crate::format::{
    decode, decode_head, encode, encode_as, rekeyed_index, Answer, Answered, Came, Coming, Framed,
    Head, Kind, LogSegment, Manifest, Op, SegmentCurve, SegmentFile, SegmentIndex, SegmentMeta,
    DEFAULT_MAX_FORMAT, FORMAT_VERSION, SUMMED_AT_ONCE,
};
use crate::mutation::{Batch, Mutation, Receipt, Unembedded, Vectorize, Weight};
use crate::state::{
    self, Discarded, Fit, Indexes, Loaded, Merging, Names, Shape, Snapshot, State, Stored,
};
use crate::stopping::{MergeStage, Stopping};
use crate::{Refusal, Result, StorageError};
use vellev_index::stop::Stop;

/// How many times a commit whose outcome is unknown is tried. Repeating is
/// safe because the coordinator accepts an identical commit twice.
const COMMIT_ATTEMPTS: u32 = 3;
/// How many rows a flush copies from memory while commits wait.
const COPIED_AT_ONCE: usize = 4_096;
/// How many segments a restore from another store is copying at once.
const SEGMENTS_COPIED_AT_ONCE: usize = 4;
/// How many segments a restore from another store decodes at once, as it
/// loads the state it puts back from its copies: few, for it holds them
/// beside the live state, which it keeps until it commits.
const RESTORED_DECODED_AT_ONCE: usize = 2;
/// How many times in all a restore from another store tries a step on
/// local disk that did not answer in time, as it loads the state.
const RELOAD_TRIES: usize = 3;
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
    /// Where the partition's commits are recorded, if known: so that the
    /// logs a load replays can be fetched together, named by the commits'
    /// rows, rather than one after another, each named by the next.
    commits: Option<Arc<dyn Coordinator>>,
    /// What replaying commits into this copy has done.
    replayed: std::sync::Mutex<Replayed>,
}

/// What loading a copy and replaying commits into it has done, since it
/// was made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Replayed {
    /// How many threads coded rows at once, at the most, for the commits
    /// replayed together.
    pub coded_on_threads: usize,
    /// How many threads split rows' text into tokens at once, at the most,
    /// to index them: as a segment was loaded, or commits replayed.
    pub tokenized_on_threads: usize,
}

use crate::state::REPLAYED_TOGETHER;
/// How many fetches a load has under way at once, at the most.
const FETCHED_AT_ONCE: usize = 16;
/// How many commits back from the head a load first fetches together,
/// doubled each time the walk back reaches past what it fetched.
const FETCHED_FIRST: u64 = 64;

/// The heads of commits a walk back from a head fetched before it reached
/// them: the commits the coordinator records between a position and the
/// head, fetched and decoded together, a window at a time from the head
/// back, the window doubling each time the walk reaches past it.
struct Ahead {
    /// The heads fetched, by key: each decoded, or why not.
    fetched: HashMap<String, Result<Head>>,
    /// Down to which position commits have been asked for.
    from: u64,
    window: u64,
    incarnation: Incarnation,
    /// The newest checkpoint listed, by its name: nothing below it is asked
    /// for or fetched.
    floor: Option<u64>,
}

impl Ahead {
    fn new(committed: &Committed) -> Self {
        Self {
            fetched: HashMap::new(),
            from: committed.log_position.0 + 1,
            window: FETCHED_FIRST,
            incarnation: committed.incarnation,
            floor: None,
        }
    }

    /// The head of `key`, at `expected`, if it was or can now be fetched
    /// together with others; `None` to fetch it alone: the partition knows
    /// no coordinator to ask, or it names none of these commits.
    async fn take(
        &mut self,
        partition: &Partition,
        key: &str,
        expected: LogPosition,
        position: LogPosition,
    ) -> Option<Result<Head>> {
        if let Some(head) = self.fetched.remove(key) {
            return Some(head);
        }
        let commits = partition.commits.as_ref()?;
        // Asked for already as far down as this: the coordinator named
        // none of them by this key.
        if expected.0 >= self.from || expected <= position {
            return None;
        }
        // Below the checkpoint listed, the walk has ended or the name
        // misled: fetch alone.
        if self.floor.is_some_and(|at| expected.0 < at) {
            return None;
        }
        let (below, window) = next_window(self.from, self.window, position);
        let asked = commits
            .commits_after(
                &partition.key,
                self.incarnation,
                LogPosition(below),
                (self.from - below) as usize,
            )
            .await;
        let points = match asked {
            Ok(points) => points,
            Err(error) => {
                tracing::warn!(partition = %partition.key, %error, "the commits to load could not be listed; fetching them one by one");
                self.from = 0;
                return None;
            }
        };
        self.from = below + 1;
        self.window = window;
        let listed: Vec<_> = points
            .into_iter()
            .filter(|point| point.position.0 < expected.0 + 1)
            .collect();
        // The walk back ends at the newest checkpoint: nothing from before
        // it is fetched, and no window is asked for below it. Should the
        // name mislead, the walk checks each head it reaches as ever, and
        // fetches what it was not given alone.
        let checkpoint = listed
            .iter()
            .filter(|point| crate::blobs::named_as_checkpoint(&point.head.0))
            .map(|point| point.position.0)
            .max();
        if let Some(at) = checkpoint {
            self.from = self.from.max(at);
            self.floor = Some(at);
        }
        let keys: Vec<String> = listed
            .into_iter()
            .filter(|point| checkpoint.is_none_or(|at| point.position.0 >= at))
            .map(|point| point.head.0)
            .filter(|key| !self.fetched.contains_key(key))
            .collect();
        let fetched = futures::stream::iter(keys)
            .map(|key| async move {
                let head = match partition.fetch(&key).await {
                    Ok(bytes) => {
                        let decoding = key.clone();
                        tokio::task::spawn_blocking(move || decode_head(&decoding, &bytes))
                            .await
                            .unwrap_or_else(|panic| std::panic::resume_unwind(panic.into_panic()))
                    }
                    Err(error) => Err(error),
                };
                (key, head)
            })
            .buffer_unordered(FETCHED_AT_ONCE)
            .collect::<Vec<_>>()
            .await;
        self.fetched.extend(fetched);
        self.fetched.remove(key)
    }
}

/// The next window of commits a walk back fetches together: from above
/// `position` up to below `from`, `window` of them, and the window after
/// it twice as long, so that a tail of `n` commits takes about
/// `log2(n / FETCHED_FIRST)` asks of the coordinator.
fn next_window(from: u64, window: u64, position: LogPosition) -> (u64, u64) {
    let below = from.saturating_sub(window).max(position.0);
    (below, window.saturating_mul(2))
}

/// The committed commits between a state and a newer head, oldest first, and
/// the checkpoint they start from when the chain reaches one before it
/// reaches the state.
struct Tail {
    checkpoint: Option<(String, Manifest)>,
    logs: Vec<(String, LogSegment)>,
}

/// What a load of a checkpoint takes as loaded already, and how it reads
/// the rest.
#[derive(Default)]
struct Loading<'a> {
    /// The segments loaded already, by key, each taken as it is: those of
    /// the copy being loaded into if none are given.
    held: Option<HashMap<String, Loaded>>,
    /// The right a restore from another store loads under, if this is
    /// one's load (see [`Partition::restore`]).
    reloading: Option<&'a dyn Held>,
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
        Self::load_reusing(key, blobs, committed, Loading::default()).await
    }

    /// As [`Self::load`], the logs it replays found through `commits` and
    /// fetched together; and so is every catch-up after.
    async fn load_through(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        commits: Arc<dyn Coordinator>,
        committed: &Committed,
    ) -> Result<Self> {
        let mut partition = Self::empty(key, blobs, committed.incarnation);
        partition.commits = Some(commits);
        partition
            .advance_reusing(committed, Loading::default())
            .await?;
        Ok(partition)
    }

    /// Load the partition as committed, taking the segments `loading`
    /// holds as loaded already rather than reading them again.
    async fn load_reusing(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        committed: &Committed,
        loading: Loading<'_>,
    ) -> Result<Self> {
        let partition = Self::empty(key, blobs, committed.incarnation);
        partition.advance_reusing(committed, loading).await?;
        Ok(partition)
    }

    fn empty(key: PartitionKey, blobs: Arc<dyn Blobs>, incarnation: Incarnation) -> Self {
        Self {
            key,
            incarnation,
            blobs,
            state: RwLock::new(State::default()),
            seen: Mutex::new(Generation(0)),
            commits: None,
            replayed: std::sync::Mutex::default(),
        }
    }

    /// What replaying commits into this copy has done.
    pub fn replayed(&self) -> Replayed {
        *self.replayed.lock().expect("replayed lock")
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
            Some(local) if in_place => self.mapped(key, local, None).await.map(Mapped::read),
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
            Some(local) => self.mapped(key, local, None).await?.read().ok_or_else(|| {
                StorageError::corrupt(key, "as fetched to local disk it is no vectors file")
            }),
            None => {
                let (name, bytes) = (key.to_string(), self.blobs.get(key).await?);
                blocking(move || Column::decode(&name, &bytes)).await?
            }
        }
    }

    /// As [`Self::column`] with nothing held; for a restore from another
    /// store loading under the right `reloading`, a step on local disk
    /// that did not answer in time is tried again, while the right is
    /// held, up to [`RELOAD_TRIES`] times in all. Nothing else is tried
    /// again: not a store that did not answer, nor a file that is not
    /// whole. Each try takes as long again as the disk is given, and
    /// writes are refused all the while.
    async fn column_reloaded(&self, key: &str, reloading: Option<&dyn Held>) -> Result<Column> {
        let Some(right) = reloading else {
            return self.column(key, None).await;
        };
        let mut tried = 1;
        loop {
            match self.column(key, None).await {
                Err(slow) if tried < RELOAD_TRIES && disk_timed_out(&slow) => {
                    tracing::warn!(key, error = %slow, tried, "vectors whose local disk did not answer in time are read again");
                    if !right.still() {
                        return Err(StorageError::Unheld(self.key.tenant.clone()));
                    }
                    tried += 1;
                }
                read => return read,
            }
        }
    }

    /// The column of a segment this writer has just built and put under
    /// `key`, read in place from the store's copy on local disk: the copy
    /// the put made, or, if that is not there or not whole, one fetched
    /// afresh, as a load would. `held`, the column as built, is kept only
    /// where the store keeps no copies at all, or anywhere but on Unix.
    ///
    /// A store meant to keep copies whose disk does not take one, or does
    /// not answer in time, fails the build at once: nothing is committed,
    /// the rows stay in memory as they were, and the next build tries
    /// again. It is not tried again here: a step that did not answer may
    /// still hold the disk's turn at large files, which a second try would
    /// wait on for as long as that step hangs. Held instead, the vectors
    /// would be in memory until the partition is next loaded, unseen under
    /// a cap on memory that counts on their being read in place.
    async fn column_built(
        &self,
        key: &str,
        held: Column,
        stopping: Option<&Stopping>,
    ) -> Result<Column> {
        self.own(key)?;
        if !cfg!(unix) {
            return Ok(held);
        }
        // A merge stopped as it sums the copy gives up at once: no warning,
        // and no copy fetched afresh, as the copy is whole for all it knows.
        let stopped = || StorageError::Background(format!("loading {key} back was stopped"));
        let read = async {
            if let Some(local) = self.blobs.local(key).await? {
                match self.mapped(key, local, stopping).await? {
                    Mapped::Read(column) => return Ok(Some(column)),
                    Mapped::Stopped => return Err(stopped()),
                    Mapped::NotWhole => {}
                }
            }
            match self.blobs.fetch_local(key).await? {
                None => Ok(None),
                Some(local) => match self.mapped(key, local, stopping).await? {
                    Mapped::Read(column) => Ok(Some(column)),
                    Mapped::Stopped => Err(stopped()),
                    Mapped::NotWhole => Err(StorageError::corrupt(
                        key,
                        "as fetched to local disk it is no vectors file",
                    )),
                },
            }
        };
        match read.await {
            Ok(Some(column)) => Ok(column),
            Ok(None) => Ok(held),
            Err(error) if stopping.is_some_and(Stopping::is_stopped) => Err(error),
            Err(error) => {
                tracing::error!(key, %error, "a segment just built is not kept: its vectors could not be read from local disk");
                Err(error)
            }
        }
    }

    /// The vectors under `key` read in place from the store's copy `local`,
    /// on the disk the copy lies on, once the whole file is seen to sum to
    /// what it says: in its turn among the large files read there, for as
    /// long as that takes while it gets on, unless this process has summed
    /// the file before as it lies now. None if the copy is not a whole
    /// vectors file, and a failure if its disk gets nowhere with it.
    /// A merge's own `stopping`, if given, is asked before each piece of
    /// the file is summed: a stopped merge gives the disk's thread and turn
    /// back at once. Nothing else stops the summing: a load whose wait is
    /// given up at its bound, or dropped beside another that failed, sums
    /// on to the end, and the file is remembered as summed for the next.
    async fn mapped(&self, key: &str, local: Local, stopping: Option<&Stopping>) -> Result<Mapped> {
        let asked = stopping.cloned();
        let stopping = stopping.cloned();
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
            Err(gone) if gone.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Mapped::NotWhole)
            }
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
                    let given_up = || {
                        stopping
                            .as_ref()
                            .is_some_and(|stopping| stopping.stopped(MergeStage::Summing))
                    };
                    Ok(column
                        .and_then(|column| column.summed_as_unless(&name, file, told, given_up)))
                });
                read.await
            }
        };
        match summed {
            Ok(Ok(column)) => Ok(Mapped::Read(column)),
            // Given up as the merge it was summed for was stopped: the copy
            // is not found wanting.
            Ok(Err(_)) if asked.as_ref().is_some_and(Stopping::is_stopped) => Ok(Mapped::Stopped),
            Ok(Err(error)) => {
                tracing::warn!(key, %error, "a vectors file on local disk is not read in place");
                Ok(Mapped::NotWhole)
            }
            Err(error) => Err(unreached(error)),
        }
    }

    /// Generations only grow, so a reading from an older generation than one
    /// already applied is out of date whatever its position, and so is a
    /// lower position from the same generation. A reading that is not ahead
    /// and not stale means the history this copy holds was replaced.
    async fn advance(&self, committed: &Committed) -> Result<LogPosition> {
        self.advance_reusing(committed, Loading::default()).await
    }

    /// As [`Self::advance`], taking the segments a checkpoint lists from
    /// what `loading` holds, rather than from this copy, and reading the
    /// rest as it says.
    async fn advance_reusing(
        &self,
        committed: &Committed,
        loading: Loading<'_>,
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
                let Loading { held, reloading } = loading;
                let held = held.unwrap_or_else(|| self.held());
                let mut state = self.restore(&key, manifest, held, reloading).await?;
                let together = state.apply_together(&tail.logs)?;
                self.replayed_together(together);
                *self.state.write().expect("partition lock") = state;
            }
            None => {
                // A run of logs at a time, the codes of each made together
                // on several threads while readers may still read, and
                // only stored under the lock, a log at a time. A run is
                // coded as the state is when it begins: a collection coded
                // during it codes its rows as they are applied.
                for run in tail.logs.chunks(REPLAYED_TOGETHER) {
                    let prepared: Vec<state::Prepared> = self.read(|state| {
                        let (prepared, most) = state.prepared_together(run);
                        self.replayed_together(state::Together {
                            tokenized_on: most,
                            coded_on: if state.coding_anything() { most } else { 0 },
                        });
                        prepared
                    });
                    for ((key, log), prepared) in run.iter().zip(prepared) {
                        let mut state = self.state.write().expect("partition lock");
                        state.apply_prepared(key, log, prepared)?;
                    }
                }
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
        let mut ahead = Ahead::new(committed);
        let checkpoint = loop {
            let key = match cursor {
                reached if reached.as_deref() == known && expected == position => break None,
                Some(key) if expected > position => key,
                _ => return Ok(None),
            };
            // Fetched together with the commits before it when the
            // coordinator names them; the walk back checks each as it
            // would one fetched alone.
            let head = match ahead.take(self, &key, expected, position).await {
                Some(head) => head?,
                None => decode_head(&key, &self.fetch(&key).await?)?,
            };
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

    /// The state as `manifest`, under `key`, says, taking each segment
    /// `held` holds as it is. Loading for a restore from another store,
    /// under the right `reloading`, it decodes few segments at once, and
    /// tries a step on local disk again that did not answer in time (see
    /// [`Self::column_reloaded`]).
    async fn restore(
        &self,
        key: &str,
        manifest: Manifest,
        held: HashMap<String, Loaded>,
        reloading: Option<&dyn Held>,
    ) -> Result<State> {
        use futures::{StreamExt, TryStreamExt};
        // Each segment is decoded and indexed on a thread of its own as it
        // arrives, rather than all of them in turn once every one has.
        let catalog = Arc::new(
            Catalog::restore(manifest.next_collection_id, manifest.collections.clone())
                .map_err(|reason| StorageError::corrupt(key, reason))?,
        );
        let segment = |meta: &SegmentMeta| {
            let held = held.get(&meta.key).cloned();
            let catalog = catalog.clone();
            let meta = meta.clone();
            async move {
                if let Some(loaded) = held {
                    return Ok::<_, StorageError>(loaded);
                }
                let bytes = self.fetch(&meta.key).await?;
                let vectors = match &meta.vectors {
                    Some(key) => Some(self.column_reloaded(key, reloading).await?),
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
        };
        let segments = match reloading {
            None => try_join_all(manifest.segments.iter().map(segment)).await?,
            Some(_) => {
                futures::stream::iter(manifest.segments.iter().map(segment))
                    .buffered(RESTORED_DECODED_AT_ONCE)
                    .try_collect()
                    .await?
            }
        };
        let tokenized_on = segments
            .iter()
            .map(|loaded: &Loaded| loaded.tokenized_on)
            .max()
            .unwrap_or(0);
        {
            let mut replayed = self.replayed.lock().expect("replayed lock");
            replayed.tokenized_on_threads = replayed.tokenized_on_threads.max(tokenized_on);
        }
        let replay = try_join_all(manifest.replay.iter().map(|key| async move {
            let log: LogSegment = decode(key, Kind::Log, &self.fetch(key).await?)?;
            Ok::<_, StorageError>((key.clone(), log))
        }))
        .await?;
        let key = key.to_string();
        let (state, together) = blocking(move || {
            let (position, after) = (manifest.position, manifest.replay_after);
            State::from_manifest(&key, manifest, segments)?.replayed(&key, position, after, &replay)
        })
        .await??;
        self.replayed_together(together);
        Ok(state)
    }

    /// Counts in [`Replayed`] how many threads made what logs applied
    /// together were applied with.
    fn replayed_together(&self, together: state::Together) {
        let mut replayed = self.replayed.lock().expect("replayed lock");
        replayed.tokenized_on_threads = replayed.tokenized_on_threads.max(together.tokenized_on);
        replayed.coded_on_threads = replayed.coded_on_threads.max(together.coded_on);
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

/// What reading a vectors file's local copy in place came to.
enum Mapped {
    Read(Column),
    /// Not a whole vectors file, or not there: to be fetched afresh.
    NotWhole,
    /// Given up, its summing cut short by the stop of the merge it was read
    /// back for.
    Stopped,
}

impl Mapped {
    /// The column, if it was read: for a load no merge stops.
    fn read(self) -> Option<Column> {
        match self {
            Mapped::Read(column) => Some(column),
            Mapped::NotWhole | Mapped::Stopped => None,
        }
    }
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

    /// As [`Self::open`], finding the logs it replays through `commits`,
    /// the coordinator that recorded them, so that they can be fetched
    /// together rather than one after another.
    pub async fn open_through(
        key: PartitionKey,
        blobs: Arc<dyn Blobs>,
        commits: Arc<dyn Coordinator>,
        committed: &Committed,
    ) -> Result<Self> {
        Partition::load_through(key, blobs, commits, committed)
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
    /// The file format the cluster writes, as far as this release writes
    /// it: what every file this writer puts is stamped as.
    format: Arc<AtomicU16>,
    /// How many times this writer's state has been put back wholesale or
    /// merged: see [`Self::epoch`].
    epoch: AtomicU64,
    /// Held while a merge plans, builds and commits, so that a partition
    /// builds one merge at a time, as it did when merges held `building`
    /// throughout: what bounds the memory merges take in it.
    merge_build: Mutex<()>,
    /// The files a merge under way has uploaded and not yet committed or
    /// dropped: [`Self::clean`] leaves them, however old.
    keeping: Arc<std::sync::Mutex<HashSet<String>>>,
    /// A gate a merge waits at between its build and its commit, if a test
    /// has set one: see [`Self::hold_merges_before_commit`].
    merge_gate: std::sync::Mutex<Option<Arc<MergeGate>>>,
    /// The stop of the merge under way, if one is: see
    /// [`Self::stop_merges`].
    merge_stop: Arc<std::sync::Mutex<Option<Stop>>>,
    /// The stage a test has each merge stop itself in, if any: see
    /// [`Self::stop_merges_at`].
    merge_stop_at: std::sync::Mutex<Option<MergeStage>>,
    /// How many times merges were stopped: a merge waiting its turn when
    /// they were is stopped too, as it finds this moved once its turn
    /// comes.
    merges_stopped: AtomicU64,
    /// Whether each merge's build panics, as a test asks: see
    /// [`Self::panic_merges_in_build`].
    merge_panics: AtomicBool,
    /// How many merges were stopped only once they had loaded back what
    /// they built, past where a stop ends them.
    stops_past_loading: AtomicU64,
    /// Whether the merge under way is past loading back what it built,
    /// from where a stop no longer ends it.
    merge_past_loading: AtomicBool,
}

/// The stop of the merge under way, there until the merge ends, however it
/// ends.
struct MergeStopped {
    under_way: Arc<std::sync::Mutex<Option<Stop>>>,
}

impl Drop for MergeStopped {
    fn drop(&mut self) {
        *lock(&self.under_way) = None;
    }
}

/// `mutex` locked, whether or not a holder panicked: what it guards, a
/// merge's stop, is whole either way.
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Where a test holds a merge between its build and its commit: `reached`
/// is told when a merge arrives, and the merge waits for `release`.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct MergeGate {
    pub reached: tokio::sync::Notify,
    pub release: tokio::sync::Notify,
}

/// A merge's uploads kept from [`Writer::clean`] while it is under way,
/// and let go of however it ends, a panic included.
struct Keeping {
    keeping: Arc<std::sync::Mutex<HashSet<String>>>,
    keys: Vec<String>,
}

impl Drop for Keeping {
    fn drop(&mut self) {
        let mut keeping = self.keeping.lock().unwrap_or_else(|held| held.into_inner());
        for key in &self.keys {
            keeping.remove(key);
        }
    }
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
        let partition = Arc::new(
            Partition::load_through(lease.key.clone(), blobs, coordinator.clone(), committed)
                .await?,
        );
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
            format: Arc::new(AtomicU16::new(DEFAULT_MAX_FORMAT)),
            epoch: AtomicU64::new(0),
            merge_build: Mutex::new(()),
            keeping: Arc::default(),
            merge_gate: std::sync::Mutex::new(None),
            merge_stop: Arc::default(),
            merge_stop_at: std::sync::Mutex::new(None),
            merges_stopped: AtomicU64::new(0),
            merge_panics: AtomicBool::new(false),
            stops_past_loading: AtomicU64::new(0),
            merge_past_loading: AtomicBool::new(false),
        })
    }

    /// Write the format `format` holds, as it is when each file is
    /// written, instead of [`DEFAULT_MAX_FORMAT`]: a node shares the
    /// format the cluster writes with every writer it opens.
    pub fn writing_format(self, format: Arc<AtomicU16>) -> Self {
        Self { format, ..self }
    }

    /// The format this writer writes now: no older than 1, and no newer
    /// than this release writes.
    pub fn writes(&self) -> u16 {
        self.format.load(Ordering::SeqCst).clamp(1, FORMAT_VERSION)
    }

    /// `value`, which format `least` can hold, framed as `kind` in the
    /// format this writer writes.
    fn stamped<T: Serialize>(&self, kind: Kind, value: &T, least: u16) -> Result<Bytes> {
        stamped(kind, value, least, self.writes())
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
            let mut draft = Draft::new(state, self.writes());
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
            let mut draft = Draft::new(state, self.writes());
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
        // One format for the whole commit, drafted and logged alike, though
        // the cluster's rise meanwhile.
        let format = self.writes();
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
                    let mut draft = Draft::new(state, format);
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
                        let least = log.least_format();
                        stamped(Kind::Log, &log, least, format).map(|frame| (log, frame))
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
                    // Coded first, while readers may still read; the apply
                    // under the lock then only stores the codes.
                    let prepared = partition.read(|state| state.prepared(&log));
                    let mut state = partition.state.write().expect("partition lock");
                    state.apply_prepared(&applied, &log, prepared)
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
        let frame = self.stamped(Kind::Manifest, &manifest, manifest.least_format())?;
        staged.keep();
        // The memtable it keeps, made while readers may still read: with
        // the turn held, nothing is committed before it is adopted.
        let kept = self.partition.read(|state| state.kept(&snapshot));
        self.checkpoint(turn, &key, frame, position, |state| {
            state.flushed(&key, position, snapshot, segment, Some(kept))
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
            Some(prepared) => Some(self.fit(prepared, &snapshot.catalog, None).await?),
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
        let loading = Loading {
            held: Some(self.partition.held()),
            reloading: None,
        };
        let target = Partition::load_reusing(
            self.lease.key.clone(),
            self.partition.blobs.clone(),
            &at,
            loading,
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
                        let (meta, loaded) = self.loaded(prepared, None).await?;
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
        let frame = self.stamped(Kind::Manifest, &manifest, manifest.least_format())?;
        let state = target.restore(&key, manifest, held, None).await?;
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
                let replaced = std::mem::replace(
                    &mut *self.partition.state.write().expect("partition lock"),
                    state,
                );
                self.epoch.fetch_add(1, Ordering::SeqCst);
                Ok(replaced)
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
    /// with nothing would load it, two segments at a time; but for a
    /// segment the live state holds, which is taken as it is for the copy
    /// of a source's segment once that copy is proved to be the very file
    /// the live state was loaded from (see [`Self::reused`]). A copy is
    /// never trusted unread: each is read back whole, to be decoded or to
    /// be proved.
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
        let mut reused = HashMap::new();
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
            // Writes are refused and the building lock held, so the live
            // state holds just these until the restore is done.
            let donors: HashMap<String, Donor> = self.partition.read(|state| {
                let held = state.loaded_with_vectors();
                held.map(|(key, vectors, loaded)| {
                    let vectors = vectors.map(str::to_string);
                    (key.to_string(), Donor { vectors, loaded })
                })
                .collect()
            });
            let copied: Vec<Copied> = futures::stream::iter(copies)
                .map(|(meta, names)| {
                    let donor = donors.get(&meta.key).cloned();
                    self.copied(&*source, meta, names, donor, &indexing, held)
                })
                .buffered(SEGMENTS_COPIED_AT_ONCE)
                .try_collect()
                .await?;
            drop(donors);
            let segments = copied.len();
            let mut fresh = HashMap::new();
            let mut dropped = HashSet::new();
            for copied in copied {
                put.extend(copied.files.iter().map(|(_, to)| to.clone()));
                fresh.extend(copied.files);
                dropped.extend(copied.unused);
                reused.extend(copied.reused);
            }
            tracing::info!(
                partition = ?self.lease.key,
                segments,
                taken_as_held = reused.len(),
                decoded = segments - reused.len(),
                "a restore from another store copied the source's segments"
            );
            let loading = loading(&checkpoint_key, manifest, &fresh, &dropped)?;
            memory.insert(checkpoint_key, encode(Kind::Manifest, &loading)?);
        }
        // The state is loaded from the copies alone, read back from the
        // store and summed, as a node with nothing would read them: the
        // one check that each came out whole. A segment taken as the live
        // state holds it had its copy read back and proved already.
        let overlay = Overlay {
            memory,
            put: put.clone(),
            primary: self.partition.blobs.clone(),
        };
        let loading = Loading {
            held: Some(reused),
            reloading: Some(held),
        };
        let target = Partition::load_reusing(key, Arc::new(overlay), &at, loading).await?;
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
    ///
    /// The segment the live state holds under the key the source's had,
    /// `donor`, is taken for the copy if the copy is proved to be that
    /// very file (see [`Self::reused`]).
    async fn copied(
        &self,
        source: &dyn Blobs,
        meta: SegmentMeta,
        names: Names,
        donor: Option<Donor>,
        indexing: &Mutex<()>,
        held: &dyn Held,
    ) -> Result<Copied> {
        let blobs = &self.partition.blobs;
        let mut files = Vec::new();
        let mut unused = None;
        let mut sent = Vec::new();
        if let Some(vectors) = meta.vectors.clone() {
            self.still(held)?;
            let (came, stamp) = self.put_copy(source, &vectors, &names.vectors).await?;
            sent.push((Kind::Vectors, names.vectors.clone(), came));
            self.copy_stamped(&vectors, stamp)?;
            files.push((vectors, names.vectors.clone()));
        }
        if let Some(index) = meta.index {
            self.still(held)?;
            let _one = indexing.lock().await;
            let frame = source.get(&index).await?;
            self.copy_stamped(&index, crate::format::stamp(&frame))?;
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
        let (came, stamp) = self.put_copy(source, &meta.key, &names.segment).await?;
        sent.push((Kind::Segment, names.segment.clone(), came));
        self.copy_stamped(&meta.key, stamp)?;
        let linked = meta.vectors.clone().zip(Some(names.vectors));
        let reused = match donor {
            Some(donor) if donor.vectors == meta.vectors => {
                let reused = self.reused(donor, &sent, linked, held).await?;
                reused.map(|loaded| (names.segment.clone(), loaded))
            }
            _ => None,
        };
        files.push((meta.key, names.segment));
        Ok(Copied {
            files,
            unused,
            reused,
        })
    }

    /// Refuse a copy of the source's file `key`, stamped as format
    /// `version`, that is newer than this writer writes: a copy keeps the
    /// stamp it came with, and one newer than the cluster writes is one the
    /// release before, which the cluster's format still lets roll back to,
    /// cannot read. The copy already put is staged, so it is cleaned up.
    /// A stamp that cannot be read is left to the load, which reads every
    /// copy and fails the restore on one that is not whole.
    fn copy_stamped(&self, key: &str, version: Option<u16>) -> Result<()> {
        let writes = self.writes();
        match version {
            Some(version) if version > writes => Err(StorageError::InvalidMutation(format!(
                "{key} is stamped as file format {version}, and this cluster writes format \
                 {writes}: a restore from it would leave files the release before cannot \
                 read. Let every node write format {version} (VELLEV_MAX_FORMAT={version}) \
                 first"
            ))),
            _ => Ok(()),
        }
    }

    /// Put the file under `from` in `source` into the partition's store
    /// under `to`, a part at a time; what it came to as the source sent it,
    /// and the format its frame is stamped as, if it is long enough to say.
    async fn put_copy(
        &self,
        source: &dyn Blobs,
        from: &str,
        to: &str,
    ) -> Result<(Came, Option<u16>)> {
        use futures::{StreamExt, TryStreamExt};
        let coming = std::sync::Mutex::new((Coming::default(), Vec::new()));
        let parts = source.parts(from).map_ok(|part| {
            let (came, head) = &mut *coming.lock().expect("coming");
            came.update(&part);
            let wanted = crate::format::STAMPED_WITHIN.saturating_sub(head.len());
            head.extend_from_slice(&part[..wanted.min(part.len())]);
            part
        });
        self.partition.blobs.put_parts(to, parts.boxed()).await?;
        let (came, head) = coming.into_inner().expect("coming");
        Ok((came.came(), crate::format::stamp(&head)))
    }

    /// The segment the live state holds, `donor`, if the copies a restore
    /// made of the source's segment of the same key, and of its vectors of
    /// the same key, are proved to be the very files the live state was
    /// loaded from; none, for the copy to be decoded, if they are not.
    ///
    /// Each copy, `sent` with its kind and what the source sent of it, is
    /// read back from the store a part at a time, kept nowhere, and must be
    /// one whole frame of its kind and come to just what the source sent.
    /// That proves the copy is what the source sent: its frame alone
    /// proves only that it is whole, and what the source sent may itself
    /// be damaged. The source's file is the live state's because the key
    /// is the same and a key is written once (K1), and the live state's
    /// was summed when it was loaded. A copy that cannot be read back is
    /// not proved; it is decoded, which reads it again, and fails the
    /// restore if it is not whole.
    ///
    /// The vectors the donor holds may be read in place from its copy on
    /// local disk, under the key it had. Its column is taken only once the
    /// store gives that copy to the new key too, `linked` (see
    /// [`Blobs::link_local`]), or has no copy read in place at all: never
    /// is a file read in place under a key the state does not name.
    async fn reused(
        &self,
        donor: Donor,
        sent: &[(Kind, String, Came)],
        linked: Option<(String, String)>,
        held: &dyn Held,
    ) -> Result<Option<Loaded>> {
        for (kind, key, came) in sent {
            self.still(held)?;
            match self.streamed_back(key, *kind).await {
                Ok(back) if back == *came => {}
                Ok(_) => {
                    tracing::warn!(key, "a copy read back is not what the source sent");
                    return Ok(None);
                }
                Err(error) => {
                    tracing::warn!(key, %error, "a copy read back is not proved whole");
                    return Ok(None);
                }
            }
        }
        if let Some((from, to)) = linked {
            self.still(held)?;
            match self.partition.blobs.link_local(&from, &to).await? {
                Linked::Linked(_) | Linked::Unmapped => {}
                Linked::Unlinked => return Ok(None),
            }
        }
        Ok(Some(donor.loaded))
    }

    /// What the file under `key` in the partition's store comes to, read
    /// back a part at a time and kept nowhere, if it is one whole frame of
    /// `kind`.
    async fn streamed_back(&self, key: &str, kind: Kind) -> Result<Came> {
        use futures::TryStreamExt;
        let mut framed = Framed::default();
        let mut parts = self.partition.blobs.parts(key);
        while let Some(part) = parts.try_next().await? {
            framed.update(&part);
        }
        framed.came(key, kind)
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
        let vectors = snapshot.gathered();
        let above = self.merging.clustered_above;
        let tombstones = snapshot.tombstones.len();
        let shape = Shape::of_gathered(&snapshot.catalog, &rows, &vectors, tombstones, above);
        let file = SegmentFile {
            position: snapshot.position,
            rows,
            tombstones: snapshot.tombstones.clone(),
        };
        let built = self
            .built(file, Some(vectors), snapshot.catalog.clone())
            .await?;
        let prepared = Self::prepared(names.meta(shape), built, self.writes())?;
        self.upload(&prepared, staged).await?;
        Ok(Some(prepared))
    }

    /// The files this writer uploads for a commit it has yet to attempt,
    /// none so far.
    fn staging(&self) -> Staged {
        Staged {
            blobs: self.partition.blobs.clone(),
            keys: Vec::new(),
            putting: Vec::new(),
        }
    }

    /// Names for the files of a segment at `position`, apart from those of
    /// any other build there, each ending in `marker`. That is at or past
    /// the latest checkpoint: a flush's or a merge's at the position of the
    /// partition it was built from, a restore's at the position it commits.
    fn named(&self, position: LogPosition, marker: &str) -> Names {
        self.named_drawing(position, marker, uuid::Uuid::new_v4().as_u64_pair().0)
    }

    /// As [`Self::named`], with `nonce` as drawn: so that what a writer
    /// names its files is pinned to [`Layout::nonced`], whatever it draws.
    fn named_drawing(&self, position: LogPosition, marker: &str, nonce: u64) -> Names {
        let layout = Layout(&self.lease.key);
        let named = |kind| layout.nonced(kind, position, &self.author, nonce, marker);
        Names {
            segment: named("segment"),
            index: named("index"),
            vectors: named("vectors"),
        }
    }

    /// Index the segment of `file`, its vectors gathered in `vectors` or,
    /// without, still in its rows, to be taken from them into their column;
    /// then encode both, sealing the gathered frame, off the async threads
    /// and out of the state's lock, before anything is written, so that
    /// neither readers nor commits wait while a large segment is indexed.
    async fn built(
        &self,
        file: SegmentFile,
        vectors: Option<Gathering>,
        catalog: Catalog,
    ) -> Result<Built> {
        match self
            .built_stopping(file, vectors, catalog, &Stopping::never())
            .await?
        {
            Some(built) => Ok(built),
            None => unreachable!("a build nothing stops is never stopped"),
        }
    }

    /// As [`Self::built`], giving up, `None`, once `stopping` is stopped:
    /// asked as the rows are indexed, and before and after they are
    /// encoded.
    async fn built_stopping(
        &self,
        mut file: SegmentFile,
        vectors: Option<Gathering>,
        catalog: Catalog,
        stopping: &Stopping,
    ) -> Result<Option<Built>> {
        let above = Some(self.merging.clustered_above);
        let format = self.writes();
        // Its turn among the work off the async threads is waited for as
        // the merge's stop is: given up before it begins, the work never
        // runs. Once begun it is waited for to its end, which comes at the
        // stop as it asks it: so that the merge gives up only once what the
        // work holds is let go, not while it runs on unseen.
        let turn = working().acquire_owned();
        let turn = match stopping.unless_stopped(MergeStage::Turn, turn).await {
            Some(turn) => turn.map_err(|error| StorageError::Background(error.to_string()))?,
            None => return Ok(None),
        };
        let stopping = stopping.clone();
        at_once(move || {
            let _turn = turn;
            assert!(!stopping.panics(), "a merge's build panics, as a test asks");
            let vectors = vectors.map(|gathered| gathered.finish(format));
            let indexes = match &vectors {
                Some(column) => state::indexed(
                    &catalog,
                    &file.rows,
                    |row: usize| column.get(row),
                    &Stored::default(),
                    |_, _| {},
                    above,
                    &stopping,
                ),
                None => state::indexed(
                    &catalog,
                    &file.rows,
                    |row: usize| file.rows[row].object.vector.as_deref(),
                    &Stored::default(),
                    |_, _| {},
                    above,
                    &stopping,
                ),
            };
            let Some(indexes) = indexes else {
                return Ok(None);
            };
            if stopping.stopped(MergeStage::Encoding) {
                return Ok(None);
            }
            let vectors = vectors.unwrap_or_else(|| Column::taken_from(&mut file.rows));
            let segment = encode_as(Kind::Segment, format, &file)?;
            if stopping.stopped(MergeStage::Encoding) {
                return Ok(None);
            }
            let column = vectors.encode_as(format);
            Ok(Some(Built {
                file,
                vectors,
                indexes,
                segment,
                column,
            }))
        })
        .await?
    }

    /// A built segment as `meta` names it, with its index file's frame if
    /// it has one, and its indexes' curves.
    fn prepared(mut meta: SegmentMeta, built: Built, format: u16) -> Result<Prepared> {
        // A clustered index built here scans as the curve of the rows it
        // was tuned on says, and the segment keeps that curve.
        if meta.index.is_some() {
            meta.probes = state::curves(&built.indexes);
        }
        let index = meta
            .index
            .is_some()
            .then(|| {
                encode_as(
                    Kind::Index,
                    format,
                    &state::stored(&meta.key, &built.indexes),
                )
            })
            .transpose()?;
        Ok(Prepared { meta, built, index })
    }

    /// Upload a segment's files, the segment's own last so that it never
    /// sits in the store without its companions, noting each in `staged`
    /// before it goes up.
    async fn upload(&self, prepared: &Prepared, staged: &mut Staged) -> Result<()> {
        match self
            .upload_stopping(prepared, staged, &Stopping::never())
            .await?
        {
            true => Ok(()),
            false => unreachable!("an upload nothing stops is never stopped"),
        }
    }

    /// As [`Self::upload`], giving up, `false`, once `stopping` is
    /// stopped: asked before each file, and while each goes up. A file
    /// given up part way is aborted as its upload is dropped, its parts
    /// discarded ([`crate::Transfers`]); those that went up are deleted
    /// as `staged` is dropped.
    async fn upload_stopping(
        &self,
        prepared: &Prepared,
        staged: &mut Staged,
        stopping: &Stopping,
    ) -> Result<bool> {
        let Prepared { meta, built, index } = prepared;
        let files = [
            meta.vectors.as_deref().map(|key| (key, &built.column)),
            meta.index.as_deref().zip(index.as_ref()),
            Some((meta.key.as_str(), &built.segment)),
        ];
        for (key, bytes) in files.into_iter().flatten() {
            if stopping.stopped(MergeStage::Uploading) {
                return Ok(false);
            }
            // Noted before it goes up, with the task it goes up on: however
            // this ends, given up, stopped or dropped, the file is deleted
            // once the upload has settled, never before.
            staged.keys.push(key.to_string());
            let (blobs, owned, bytes) =
                (self.partition.blobs.clone(), key.to_string(), bytes.clone());
            staged
                .putting
                .push(tokio::spawn(async move { blobs.put(&owned, bytes).await }));
            let put = staged.putting.last_mut().expect("just pushed");
            match stopping.unless_stopped(MergeStage::Uploading, put).await {
                Some(Ok(put)) => {
                    staged.putting.pop();
                    put?;
                }
                Some(Err(failed)) => {
                    staged.putting.pop();
                    match failed.try_into_panic() {
                        Ok(panic) => std::panic::resume_unwind(panic),
                        // Cancelled, as the runtime shuts down.
                        Err(failed) => return Err(StorageError::Background(failed.to_string())),
                    }
                }
                None => return Ok(false),
            }
        }
        Ok(true)
    }

    /// An uploaded segment as the state holds it, with its vectors read
    /// from local disk where the store keeps a copy.
    async fn loaded(
        &self,
        prepared: Prepared,
        stopping: Option<&Stopping>,
    ) -> Result<(SegmentMeta, Loaded)> {
        let Prepared { meta, built, .. } = prepared;
        let vectors = match &meta.vectors {
            Some(key) => {
                self.partition
                    .column_built(key, built.vectors, stopping)
                    .await?
            }
            None => built.vectors,
        };
        let loaded = Loaded {
            file: Arc::new(built.file),
            vectors: Arc::new(vectors),
            indexes: Some(Arc::new(built.indexes)),
            stored: Stored::default(),
            tokenized_on: 0,
        };
        Ok((meta, loaded))
    }

    /// As [`Self::loaded`], and seen to be fit to adopt under the
    /// collections of `catalog`: here, before the writer is held, so that
    /// a segment that is not fails with nothing committed or changed.
    async fn fit(
        &self,
        prepared: Prepared,
        catalog: &Catalog,
        stopping: Option<&Stopping>,
    ) -> Result<Fit> {
        let (meta, loaded) = self.loaded(prepared, stopping).await?;
        loaded.fit(meta, catalog)
    }

    /// Carry out the merge the partition most needs, if any: replace a run
    /// of segments with one holding their live rows, in a checkpoint that
    /// also flushes what is unflushed, as [`Self::flush`] does. Both
    /// segments are built from the partition as it is now and written while
    /// commits go on; a row that dies meanwhile is dead in the new segment
    /// too. Returns the checkpoint's position, or `None` when there was
    /// nothing to merge.
    /// Make and commit the curve of its rows for every clustered index of a
    /// segment that has none, as a segment written before curves (recall
    /// note, R9.5): the curve its build would have given, made off the
    /// async threads, then committed as [`Self::commit_curves`] commits.
    /// What is missing is worked out afresh at each call, from the segments
    /// there are, so a pass cut short, by the tenant being let go or the
    /// lease lost, is made again by the next. Returns the position
    /// committed, or None if nothing was missing.
    pub async fn rows_pass(&self) -> Result<Option<LogPosition>> {
        let missing = self.partition.read(State::curves_missing);
        if missing.is_empty() {
            return Ok(None);
        }
        let curves = blocking(move || {
            missing
                .into_iter()
                .filter_map(|missing| Some((missing.key.clone(), missing.curve()?)))
                .collect::<Vec<_>>()
        })
        .await?;
        self.commit_curves(curves).await
    }

    /// Commit `curves`, each for the segment its key names (recall note,
    /// R7 and R9.2): a checkpoint at the next position, as a merge commits
    /// one, the memtable flushed along, every segment as it is but those
    /// named, which hold their new curves. Only what holds the building
    /// lock replaces segments, and this holds it too, so a curve made for a
    /// segment that a merge has replaced since is not committed, and a
    /// merge after keeps the curves of the segments it does not replace.
    /// Returns the position committed, or None if no segment named is still
    /// there.
    pub async fn commit_curves(
        &self,
        curves: Vec<(String, SegmentCurve)>,
    ) -> Result<Option<LogPosition>> {
        let _building = self.building.lock().await;
        let there: HashSet<String> = self
            .partition
            .read(|state| state.segments().map(|meta| meta.key.clone()).collect());
        let curves: Vec<(String, SegmentCurve)> = curves
            .into_iter()
            .filter(|(key, _)| there.contains(key))
            .collect();
        if curves.is_empty() {
            return Ok(None);
        }
        // A curve is of a clustered index: one for a collection its segment
        // has none of is a caller's mistake, and none of what was given is
        // committed.
        if let Some((key, curve)) = self.partition.read(|state| {
            curves
                .iter()
                .find(|(key, curve)| !state.clustered_in(key, curve.collection))
                .cloned()
        }) {
            return Err(StorageError::InvalidMutation(format!(
                "a curve for collection {} of segment {key}, which has no clustered index of it",
                curve.collection.0
            )));
        }
        let mut staged = self.staging();
        let snapshot = self.partition.read(State::taken);
        let (snapshot, segment) = self.flushing(snapshot, &mut staged).await?;
        let turn = self.turn().await?;
        let position = following(self.partition.read(State::position))?;
        let key = Layout(&self.lease.key).manifest(position, &self.author);
        let Some(manifest) = self
            .partition
            .read(|state| state.tuned_manifest(position, &snapshot, segment.as_ref(), &curves))?
        else {
            return Ok(None);
        };
        let frame = self.stamped(Kind::Manifest, &manifest, manifest.least_format())?;
        staged.keep();
        let kept = self.partition.read(|state| state.kept(&snapshot));
        self.checkpoint(turn, &key, frame, position, |state| {
            state.tuned(&curves);
            state.flushed(&key, position, snapshot, segment, Some(kept))
        })
        .await?;
        Ok(Some(position))
    }

    pub async fn merge(&self) -> Result<Option<LogPosition>> {
        self.merge_planned_at(None).await
    }

    /// Whether a merge planned at epoch `planned` is stale now, and so is
    /// to be dropped; said if so.
    fn stale(&self, planned: u64) -> bool {
        let now = self.epoch();
        if now != planned {
            tracing::warn!(
                planned,
                now,
                "a merge planned before the partition was put back is dropped"
            );
        }
        now != planned
    }

    /// How many times this writer has put its state back wholesale (a
    /// restore in place, or from another store) or committed a merge. A
    /// merge is committed only at the epoch it was planned at: one
    /// planned before a restore that brought back the same segments, with
    /// other rows live in them, would otherwise describe rows that are
    /// not there. Flushes, curves and writes leave it as it is: they add
    /// segments after a run, or change entries, or kill rows, which a
    /// merge's commit already takes as it finds them.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    /// As [`Self::merge`], committed only if the epoch is still `planned`
    /// (the one read at planning if `None`): how a test shows that a
    /// merge planned at another epoch commits nothing. Not for use.
    #[doc(hidden)]
    pub async fn merge_planned_at(&self, planned: Option<u64>) -> Result<Option<LogPosition>> {
        let asked = self.merges_stopped.load(Ordering::SeqCst);
        let _merging = self.merge_build.lock().await;
        // Stopped by whoever lets the tenant go, or finds the lease lost,
        // from here to the end of loading back what it built: it then gives
        // up where it is, and leaves nothing (merge note, step 3).
        let panics = self.merge_panics.load(Ordering::SeqCst);
        let stopping = Stopping::merge(*lock(&self.merge_stop_at), panics);
        {
            let mut under_way = lock(&self.merge_stop);
            self.merge_past_loading.store(false, Ordering::SeqCst);
            *under_way = Some(stopping.stop().clone());
        }
        let _stopped = MergeStopped {
            under_way: self.merge_stop.clone(),
        };
        // Stopped while it waited its turn: the stop is looked for once
        // this merge's own is where a stop finds it, so none is missed.
        if self.merges_stopped.load(Ordering::SeqCst) != asked {
            return Ok(None);
        }
        if stopping.stopped(MergeStage::Planning) {
            return Ok(None);
        }
        // A lease another writer holds now: the merge could never commit,
        // and is not built for nothing.
        if !self.still_leased().await? {
            tracing::warn!("a merge is not made on a lease another writer has taken");
            return Ok(None);
        }
        // Planned, and its rows gathered, under the building lock; built,
        // uploaded and loaded without it, so that flushes go on meanwhile;
        // committed under it again, only if nothing put the partition back
        // since and the run is as it was (merge note, step 2).
        let planning = self.building.lock().await;
        let planned = planned.unwrap_or_else(|| self.epoch());
        let Some(merge) = self.partition.read(|state| state.merge_plan(&self.merging)) else {
            return Ok(None);
        };
        let above = self.merging.clustered_above;
        let Some((file, vectors, shape, named_at, alive)) = self.partition.read(|state| {
            let named_at = state.position();
            let (file, vectors) = state.merged(&merge, named_at, &stopping)?;
            let tombstones = file.tombstones.len();
            let shape =
                Shape::of_gathered(state.catalog(), &file.rows, &vectors, tombstones, above);
            Some((file, vectors, shape, named_at, state.alive(&merge)))
        }) else {
            return Ok(None);
        };
        let catalog = self.partition.read(|state| state.catalog().clone());
        drop(planning);
        // A merge planned at another epoch than this is dropped before any
        // work is spent on it, and again at its commit below.
        if self.stale(planned) {
            return Ok(None);
        }
        let built = self.built_stopping(file, Some(vectors), catalog.clone(), &stopping);
        let Some(built) = built.await? else {
            return Ok(None);
        };
        // Both segments, however large, are uploaded without the writer,
        // under names of their own at the position each was built from;
        // `clean` leaves the merge's while it is under way.
        let mut staged = self.staging();
        let merged = Self::prepared(
            self.named(named_at, "-merged").meta(shape),
            built,
            self.writes(),
        )?;
        if stopping.stopped(MergeStage::Encoding) {
            return Ok(None);
        }
        let _keeping = self.keep(&merged.meta);
        if !self
            .upload_stopping(&merged, &mut staged, &stopping)
            .await?
        {
            return Ok(None);
        }
        let fit = self.fit(merged, &catalog, Some(&stopping));
        let Some(merged) = stopping.unless_stopped(MergeStage::Loading, fit).await else {
            return Ok(None);
        };
        // From here the merge commits, or is dropped by what it finds under
        // the building lock: a stop no longer ends it. Marked so under the
        // lock a stop is asked under, so that every stop is either heeded
        // here or found too late by whoever asked it.
        let heeded = {
            let _under_way = lock(&self.merge_stop);
            let heeded = stopping.is_stopped();
            if !heeded {
                self.merge_past_loading.store(true, Ordering::SeqCst);
            }
            heeded
        };
        // A stop heeded here wins over what loading back came to: a load
        // given up as it was stopped fails, and is no failure of the merge.
        if heeded {
            return Ok(None);
        }
        let merged = merged?;
        let gate = self.merge_gate.lock().expect("gate lock").clone();
        if let Some(gate) = gate {
            gate.reached.notify_one();
            gate.release.notified().await;
        }
        let _building = self.building.lock().await;
        // Not at another epoch than it was planned at: a restore since may
        // have brought the same segments back with other rows live.
        if self.stale(planned) {
            return Ok(None);
        }
        // Nor with a row the run holds live missing from the merged file,
        // which carrying it out could not place: checked before the turn
        // is taken, as no row comes back to life without a restore, which
        // moves the epoch, and the building lock is held.
        if !self
            .partition
            .read(|state| state.covers(&merge, &alive, &merged))
        {
            tracing::error!(
                "a merged segment lacks a row its run holds live; the merge is dropped"
            );
            return Ok(None);
        }
        let snapshot = self.partition.read(State::taken);
        let (snapshot, segment) = self.flushing(snapshot, &mut staged).await?;
        let turn = self.turn().await?;
        let position = following(self.partition.read(State::position))?;
        let key = Layout(&self.lease.key).manifest(position, &self.author);
        // Only what holds the building lock replaces segments: a flush
        // since the plan adds segments after the run, and the run is as it
        // was. Were it not, the merged segment would describe nothing there
        // is: nothing is committed, and what was uploaded is deleted.
        let Some(manifest) = self.partition.read(|state| {
            let flushed = segment.as_ref();
            state.merged_manifest(position, &snapshot, flushed, &merge, merged.meta.clone())
        })?
        else {
            return Ok(None);
        };
        let frame = self.stamped(Kind::Manifest, &manifest, manifest.least_format())?;
        staged.keep();
        let kept = self.partition.read(|state| state.kept(&snapshot));
        self.checkpoint(turn, &key, frame, position, |state| {
            state.compacted(&merge, merged);
            state.flushed(&key, position, snapshot, segment, Some(kept))
        })
        .await?;
        self.epoch.fetch_add(1, Ordering::SeqCst);
        Ok(Some(position))
    }

    /// Stop the merge under way, if one is: it gives up at the next row or
    /// block of rows it comes to, or at once if it waits on the store or the
    /// disk, and leaves nothing behind. One past loading back what it built
    /// commits, or is dropped, as it would have been. A merge begun after
    /// this is not stopped.
    pub fn stop_merges(&self) {
        self.merges_stopped.fetch_add(1, Ordering::SeqCst);
        if let Some(stop) = &*lock(&self.merge_stop) {
            // Too late for one past loading back: counted, for tests that
            // tell a stop too late from one not heeded.
            if self.merge_past_loading.load(Ordering::SeqCst) {
                self.stops_past_loading.fetch_add(1, Ordering::SeqCst);
            }
            stop.stop();
        }
    }

    /// Have each merge's build panic from now on, or not. For tests that
    /// show a merge that panics leaves no stop behind.
    #[doc(hidden)]
    pub fn panic_merges_in_build(&self, panics: bool) {
        self.merge_panics.store(panics, Ordering::SeqCst);
    }

    /// How many stops came too late: to a merge already past loading back
    /// what it built, from where it commits whatever comes. For tests.
    #[doc(hidden)]
    pub fn stops_past_loading(&self) -> u64 {
        self.stops_past_loading.load(Ordering::SeqCst)
    }

    /// Whether a merge is under way, its stop where [`Self::stop_merges`]
    /// finds it. For tests.
    #[doc(hidden)]
    pub fn merge_under_way(&self) -> bool {
        lock(&self.merge_stop).is_some()
    }

    /// Have each merge from now on stop itself the first time it asks in
    /// `stage`, or none if `None`. For tests that show each stage asks.
    #[doc(hidden)]
    pub fn stop_merges_at(&self, stage: Option<MergeStage>) {
        *lock(&self.merge_stop_at) = stage;
    }

    /// Whether the coordinator says this writer's lease is the one
    /// committed under still. Only the generation is compared: commits
    /// under way may have moved the position meanwhile.
    async fn still_leased(&self) -> Result<bool> {
        let within = self.commit_within();
        let asked = self.coordinator.committed(&self.lease.key);
        let committed = tokio::time::timeout(within, asked).await.map_err(|_| {
            let reason = format!("the coordinator did not say what is committed within {within:?}");
            CoordinatorError::backend(std::io::Error::new(std::io::ErrorKind::TimedOut, reason))
        })??;
        Ok(committed.is_some_and(|committed| committed.generation == self.lease.generation))
    }

    /// Keep `meta`'s files from [`Self::clean`] until what is given back is
    /// dropped.
    fn keep(&self, meta: &SegmentMeta) -> Keeping {
        let keys: Vec<String> = [Some(&meta.key), meta.index.as_ref(), meta.vectors.as_ref()]
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        self.keeping
            .lock()
            .expect("keeping lock")
            .extend(keys.iter().cloned());
        Keeping {
            keeping: self.keeping.clone(),
            keys,
        }
    }

    /// How many files merges under way keep from [`Self::clean`] now. For
    /// tests.
    #[doc(hidden)]
    pub fn kept_while_merging(&self) -> usize {
        self.keeping.lock().expect("keeping lock").len()
    }

    /// Hold every merge from now on between its build and its commit, at
    /// the gate given back, until it is released. For tests that make
    /// things happen meanwhile.
    #[doc(hidden)]
    pub fn hold_merges_before_commit(&self) -> Arc<MergeGate> {
        let gate = Arc::new(MergeGate::default());
        *self.merge_gate.lock().expect("gate lock") = Some(gate.clone());
        gate
    }

    /// Hold merges no more: a merge from now on goes straight from its
    /// build to its commit. For tests.
    #[doc(hidden)]
    pub fn stop_holding_merges(&self) {
        *self.merge_gate.lock().expect("gate lock") = None;
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
    /// While the partition's history is shipped to a backup store, nothing
    /// at or above the cap the run holds cleans to is deleted (DW-668 PR
    /// 6): the clean's base is the newest checkpoint at or below it, and
    /// every checkpoint above it, each on the chain, is walked and keeps
    /// what it lists, old keys a restore put back included. The cap trails
    /// what the run has shipped, so a run that lags, or has stalled, pins
    /// every file from the cap up and the partition's storage grows until
    /// it ships on: the node's lag and stall alarm is for that, and must
    /// be in place before shipping is switched on.
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
    /// under the position it commits, which is past that.
    /// - A flush and a restore hold the building lock from before they read
    ///   the partition until they commit or fail. Only what holds it commits
    ///   checkpoints, so none is committed past the partition's position
    ///   while their files wait to be published, and log commits do not move
    ///   the checkpoint this deletes below: their files sit at or past the
    ///   latest checkpoint, never below it.
    /// - A merge does not: it plans under the building lock, builds,
    ///   uploads and loads its segment without it, and takes it again only to
    ///   commit (merge note, step 2). Meanwhile flushes commit checkpoints
    ///   past the position its files are named at, so they can sit below the
    ///   latest checkpoint, and this could reach them. Two things keep them:
    ///   while the merge is under way its files are in the writer's
    ///   keep-set, which this never deletes from, however old; and the
    ///   young-file floor, a file younger than `grace` holding what this
    ///   deletes below to under its position (the merge's planned position,
    ///   less one). A writer that takes over from another has its own,
    ///   empty, keep-set and may delete the other's merge files once they
    ///   are older than `grace`: safe only because the old writer's commit
    ///   is fenced (`publish` refuses a commit of the old lease), so no head
    ///   ever names them.
    ///
    /// None of these files is deleted until a checkpoint after them has
    /// been committed and, if they were never published, until they are
    /// older than `grace` and kept by no merge under way.
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
    /// asked for when the lease is lost, which may land after another
    /// writer has taken the lease. A restore from a backup by that writer
    /// names none of those files, whatever it restores: its files are
    /// copies under keys of its own ([`Self::restore_from`]). A restore to
    /// a point in the window may name one, and is kept clear of it only by
    /// the grace: a lease is taken from a node only once it has run out,
    /// by when the node has long stopped whatever was working on the
    /// tenant, this included, and what such a restore may go back to only
    /// shrinks with time.
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

    /// The cap a run of shipping holds this partition's cleans to, if it is
    /// held for shipping (DW-668 PR 6): a clean that cannot be told fails,
    /// and deletes nothing, as one that cannot read what was cleaned does.
    async fn ship_cap(&self) -> Result<Option<LogPosition>> {
        let within = self.commit_within();
        let asked = self.coordinator.shipping(&self.lease.key);
        let shipping = tokio::time::timeout(within, asked).await.map_err(|_| {
            let reason = format!("the coordinator did not say how it is shipped within {within:?}");
            CoordinatorError::backend(std::io::Error::new(std::io::ErrorKind::TimedOut, reason))
        })??;
        Ok(shipping.cap)
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
        let cap = self.ship_cap().await?;
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
        // Held below what a run of shipping has yet to ship (DW-668 PR 6):
        // every checkpoint above the cap, and so every file one lists, is
        // walked and kept.
        let floor = young
            .map(|at| LogPosition(at.0.saturating_sub(1)))
            .chain(cap)
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
        let merging = self.keeping.lock().expect("keeping lock").clone();
        let orphans = files
            .iter()
            .filter(|(file, at)| {
                *at < base.position
                    && file.modified < cutoff
                    && !kept.contains(file.key.as_str())
                    && !merging.contains(&file.key)
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
    /// Uploads under way, each on a task of its own: one given up on is
    /// aborted, and its files deleted once it has ended.
    putting: Vec<tokio::task::JoinHandle<Result<()>>>,
}

impl Staged {
    /// Keep the files: the commit that names them is about to be tried.
    fn keep(&mut self) {
        self.keys.clear();
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        // Uploads under way are aborted, whatever else: an upload in parts
        // is then aborted at the store, as its guards are dropped with it,
        // and the bytes it holds are let go, rather than the whole file
        // going up for nothing.
        let putting = std::mem::take(&mut self.putting);
        putting.iter().for_each(tokio::task::JoinHandle::abort);
        let (false, Ok(runtime)) = (self.keys.is_empty(), tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let (blobs, keys) = (self.blobs.clone(), std::mem::take(&mut self.keys));
        // Counted as on its way, as an upload's abort is, so that a
        // process that stops waits for the deletes for as long as it gives
        // them.
        let deleting = crate::blobs::Aborts::process().begun();
        runtime.spawn(async move {
            let _deleting = deleting;
            let delete = |keys: Vec<String>| {
                let blobs = blobs.clone();
                async move {
                    for key in keys {
                        if let Err(error) = blobs.delete(&key).await {
                            tracing::warn!(key, %error, "a file never published could not be deleted");
                        }
                    }
                }
            };
            // Once each upload aborted has ended. A file a store was
            // already writing may land after this all the same: a local
            // store writes and renames on a blocking thread that an abort
            // does not stop, and a store may answer the last request of an
            // upload in parts after it was sent. Such a file is referenced
            // by nothing and kept by nothing, and cleaning up deletes it
            // once it is older than the grace.
            for put in putting {
                let _ = put.await;
            }
            delete(keys).await;
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
/// the one it was given, its index file if that was passed over, and the
/// segment the live state holds taken for it, under its copy's key, if
/// one is.
struct Copied {
    files: Vec<(String, String)>,
    unused: Option<String>,
    reused: Option<(String, Loaded)>,
}

/// A segment the live state holds, with the key of its vectors file: one
/// a restore from another store may take for a segment it copies.
#[derive(Clone)]
struct Donor {
    vectors: Option<String>,
    loaded: Loaded,
}

/// Whether `error` is a step on local disk that did not answer in time,
/// as the disk itself says, and nothing else: not a store that did not
/// answer, nor a time-out passed on in another failure.
/// `value`, which format `least` can hold, framed as `kind` and stamped as
/// format `format`. Refused if `format` is older than `least`: a file the
/// release before would read without what it cannot see. Nothing a writer
/// commits is, as its drafts refuse what its format cannot hold; a
/// checkpoint of what a newer format restored might be.
fn stamped<T: Serialize>(kind: Kind, value: &T, least: u16, format: u16) -> Result<Bytes> {
    if least > format {
        return Err(StorageError::Encode(format!(
            "this {kind:?} file needs format {least}, and the cluster writes format {format}"
        )));
    }
    encode_as(kind, format, value)
}

fn disk_timed_out(error: &StorageError) -> bool {
    let StorageError::Blob { source, .. } = error else {
        return false;
    };
    let disk = source.downcast_ref::<std::io::Error>();
    disk.is_some_and(|error| error.kind() == std::io::ErrorKind::TimedOut)
}

/// The file under `key` in `blobs`, its parts gathered into one.
async fn collected(blobs: &dyn Blobs, key: &str) -> Result<Bytes> {
    use futures::TryStreamExt;
    let mut parts: Vec<Bytes> = blobs.parts(key).try_collect().await?;
    if parts.len() == 1 {
        return Ok(parts.remove(0));
    }
    let mut whole = bytes::BytesMut::with_capacity(parts.iter().map(Bytes::len).sum());
    for part in parts {
        whole.extend_from_slice(&part);
    }
    Ok(whole.freeze())
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
                        let (meta, loaded) = writer.loaded(prepared, None).await?;
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
        let frame = writer.stamped(Kind::Manifest, &manifest, manifest.least_format())?;
        let state = target.restore(&key, manifest, held, None).await?;
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
                let replaced = std::mem::replace(
                    &mut *writer.partition.state.write().expect("partition lock"),
                    state,
                );
                writer.epoch.fetch_add(1, Ordering::SeqCst);
                Ok(replaced)
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
    /// A file the restore put is read through [`Blobs::parts`], of which
    /// a store that keeps copies on local disk keeps none: each is read
    /// once, to be loaded, and copies of them all would take as much disk
    /// as the tenant's files.
    async fn get(&self, key: &str) -> Result<Bytes> {
        if let Some(bytes) = self.memory.get(key) {
            return Ok(bytes.clone());
        }
        self.put_by_the_restore(key)?;
        collected(&*self.primary, key).await
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

    #[test]
    fn a_walk_back_asks_for_twice_as_many_commits_each_time() {
        let (mut from, mut window, mut asks) = (100_001u64, FETCHED_FIRST, 0);
        while from > 1 {
            let (below, next) = next_window(from, window, LogPosition(0));
            assert!(below < from);
            (from, window) = (below + 1, next);
            asks += 1;
            if below == 0 {
                break;
            }
        }
        assert!(asks <= 12, "{asks} asks for a tail of 100,000 commits");
        assert_eq!(
            next_window(1_000, 64, LogPosition(990)),
            (990, 128),
            "never below the state's own position"
        );
    }
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

    #[tokio::test]
    async fn a_writer_names_its_checkpoint_files_as_the_layout_does_its_nonce_in_sixteen_digits() {
        let writer = writer().await;
        let layout = Layout(&writer.lease.key);
        let at = LogPosition(7);
        // A nonce whose leading digits are zeros, kept as drawn.
        let nonce = 0xab;
        for marker in ["", "-merged", "-restored"] {
            let names = writer.named_drawing(at, marker, nonce);
            for (kind, name) in [
                ("segment", &names.segment),
                ("index", &names.index),
                ("vectors", &names.vectors),
            ] {
                let laid_out = layout.nonced(kind, at, writer.author(), nonce, marker);
                assert_eq!(*name, laid_out);
                let ending = format!("-00000000000000ab{marker}.{kind}");
                assert!(name.ends_with(&ending), "{name}");
                let named = layout.named_by(name).unwrap();
                assert_eq!(named.nonce, Some(nonce), "{name}");
            }
        }
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
            probes: vec![],
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

    /// The right a restore from a source is made under, never lost.
    struct Always;

    #[async_trait::async_trait]
    impl crate::backups::Held for Always {
        async fn held(&self) -> Result<bool> {
            Ok(true)
        }
        fn still(&self) -> bool {
            true
        }
    }

    /// The files `writer`'s partition is loaded from, as committed, copied
    /// into a store of their own by the keys they have: a backup of it.
    async fn own_backup(
        writer: &Writer,
        coordinator: &MemoryCoordinator,
    ) -> (Arc<dyn Blobs>, Committed) {
        own_backup_in(writer, coordinator, memory()).await
    }

    /// As [`own_backup`], into `source`.
    async fn own_backup_in(
        writer: &Writer,
        coordinator: &MemoryCoordinator,
        source: Arc<dyn Blobs>,
    ) -> (Arc<dyn Blobs>, Committed) {
        let committed = coordinator
            .committed(&writer.lease.key)
            .await
            .unwrap()
            .unwrap();
        let blobs = writer.partition.blobs.clone();
        let files = Partition::files(writer.lease.key.clone(), blobs.clone(), &committed);
        for key in files.await.unwrap() {
            source
                .put(&key, blobs.get(&key).await.unwrap())
                .await
                .unwrap();
        }
        (source, committed)
    }

    /// Restore `writer`'s partition from `source` at `committed`.
    async fn restored_from(
        writer: &Writer,
        source: Arc<dyn Blobs>,
        committed: &Committed,
    ) -> Result<LogPosition> {
        let restore = Restore {
            request: vellev_core::RequestId::generate(),
            asked: [0; 32],
            from: committed.log_position,
            from_at: chrono::Utc::now(),
        };
        let head = committed.manifest.as_ref();
        let ready = writer
            .restore_from(source, head, restore, None, &Always)
            .await?;
        ready.commit().await
    }

    /// The segments the partition holds, each file by where it lies in
    /// memory.
    fn segment_files(partition: &Partition) -> Vec<*const SegmentFile> {
        partition.read(|state| {
            let held = state.loaded_segments();
            held.map(|(_, loaded)| Arc::as_ptr(&loaded.file)).collect()
        })
    }

    /// A writer with two segments flushed, each with its vectors, and a
    /// row written after them.
    async fn flushed_twice(blobs: Arc<dyn Blobs>) -> (Writer, Arc<MemoryCoordinator>) {
        let (writer, coordinator) = writer_on(blobs).await;
        writer.commit(vec![create(), upsert(1)]).await.unwrap();
        writer.flush().await.unwrap();
        writer.commit(vec![upsert(2)]).await.unwrap();
        writer.flush().await.unwrap();
        writer.commit(vec![upsert(3)]).await.unwrap();
        (writer, coordinator)
    }

    fn ns(partition: &Partition) -> Vec<bool> {
        let id = |n: u8| ObjectId(uuid::Uuid::from_u128(u128::from(n)));
        partition.read(|state| {
            (1..=3)
                .map(|n| state.get(&notes(), id(n)).is_some())
                .collect()
        })
    }

    /// A tenant put back from its own backup taken a moment ago takes each
    /// segment the live state holds as it is, decoded once already: its
    /// copy is streamed back and seen to be whole and to be what the
    /// backup sent.
    #[tokio::test]
    async fn a_restore_from_its_own_backup_takes_the_segments_it_holds_as_they_are() {
        let (writer, coordinator) = flushed_twice(memory()).await;
        let (source, committed) = own_backup(&writer, &coordinator).await;
        let before = segment_files(&writer.partition);
        assert_eq!(before.len(), 2);

        restored_from(&writer, source, &committed).await.unwrap();

        let after = segment_files(&writer.partition);
        for held in &before {
            assert!(after.contains(held), "a segment was decoded again");
        }
        assert_eq!(ns(&writer.partition), [true, true, true]);
    }

    /// A store whose `parts` of a file of `kind` fail, the first time each
    /// such file is asked for, or always.
    #[derive(Debug)]
    struct Unstreamed {
        inner: Arc<dyn Blobs>,
        kind: &'static str,
        once: bool,
        failed: std::sync::Mutex<HashSet<String>>,
    }

    #[async_trait::async_trait]
    impl Blobs for Unstreamed {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
            self.inner.put(key, bytes).await
        }
        async fn put_parts(
            &self,
            key: &str,
            parts: futures::stream::BoxStream<'_, Result<Bytes>>,
        ) -> Result<()> {
            self.inner.put_parts(key, parts).await
        }
        async fn get(&self, key: &str) -> Result<Bytes> {
            self.inner.get(key).await
        }
        fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
            use futures::StreamExt;
            let first = self.failed.lock().unwrap().insert(key.to_string());
            if key.contains(self.kind) && (first || !self.once) {
                let lost = StorageError::Missing {
                    key: key.to_string(),
                };
                return futures::stream::once(async { Err(lost) }).boxed();
            }
            self.inner.parts(key)
        }
        async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
            self.inner.list(prefix).await
        }
        async fn delete(&self, key: &str) -> Result<()> {
            self.inner.delete(key).await
        }
    }

    fn memory() -> Arc<dyn Blobs> {
        Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))
    }

    /// A copy that cannot be streamed back is not taken for the segment the
    /// live state holds: the segment is decoded from it, read as any file.
    #[tokio::test]
    async fn a_copy_that_cannot_be_streamed_back_is_decoded_and_not_taken_as_held() {
        let inner = memory();
        let (writer, coordinator) = flushed_twice(inner.clone()).await;
        let (source, committed) = own_backup(&writer, &coordinator).await;
        let before = segment_files(&writer.partition);
        // The writer's store, its restores' copies of segments failing to
        // stream the first time each is asked for.
        let unstreamed = Arc::new(Unstreamed {
            inner,
            kind: "-restored.segment",
            once: true,
            failed: Default::default(),
        });
        let (lease, at) = (writer.lease.clone(), committed.clone());
        drop(writer);
        let writer = Writer::open(unstreamed, coordinator.clone(), lease, &at)
            .await
            .unwrap();
        let before_reopened = segment_files(&writer.partition);
        assert_ne!(before, before_reopened);

        restored_from(&writer, source, &committed).await.unwrap();

        let after = segment_files(&writer.partition);
        for held in &before_reopened {
            assert!(!after.contains(held), "a segment was taken unproved");
        }
        assert_eq!(ns(&writer.partition), [true, true, true]);
    }

    /// A store whose copies a restore puts of segments come out whole
    /// frames, of no rows: not what was sent.
    #[derive(Debug)]
    struct Rewriting(Arc<dyn Blobs>);

    #[async_trait::async_trait]
    impl Blobs for Rewriting {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
            self.0.put(key, bytes).await
        }
        async fn put_parts(
            &self,
            key: &str,
            parts: futures::stream::BoxStream<'_, Result<Bytes>>,
        ) -> Result<()> {
            use futures::TryStreamExt;
            if !key.ends_with("-restored.segment") {
                return self.0.put_parts(key, parts).await;
            }
            let _sent: Vec<Bytes> = parts.try_collect().await?;
            let empty = SegmentFile {
                position: LogPosition(1),
                rows: vec![],
                tombstones: vec![],
            };
            self.0
                .put(key, encode(Kind::Segment, &empty).unwrap())
                .await
        }
        async fn get(&self, key: &str) -> Result<Bytes> {
            self.0.get(key).await
        }
        fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
            self.0.parts(key)
        }
        async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
            self.0.list(prefix).await
        }
        async fn delete(&self, key: &str) -> Result<()> {
            self.0.delete(key).await
        }
    }

    /// A copy that comes out a whole frame, but not the one the source
    /// sent, is not taken for the segment the live state holds: it is
    /// decoded, is not the segment the checkpoint lists, and nothing is
    /// committed.
    #[tokio::test]
    async fn a_copy_whole_but_not_what_was_sent_is_not_taken_as_the_segment_held() {
        let inner = memory();
        let (writer, coordinator) = flushed_twice(inner.clone()).await;
        let (source, committed) = own_backup(&writer, &coordinator).await;
        let (lease, at) = (writer.lease.clone(), committed.clone());
        drop(writer);
        let rewriting = Arc::new(Rewriting(inner));
        let writer = Writer::open(rewriting, coordinator.clone(), lease, &at)
            .await
            .unwrap();

        let refused = restored_from(&writer, source, &committed).await;

        assert!(refused.is_err(), "{refused:?}");
        let now = coordinator
            .committed(&writer.lease.key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(now, committed);
    }

    /// A backup whose file is damaged, copied whole, makes a copy that
    /// streams back just as the backup sent it; it is not taken for the
    /// segment the live state holds, whose good copy would hide it. It is
    /// decoded, which fails, and nothing is committed.
    #[tokio::test]
    async fn a_copy_of_a_damaged_backup_file_is_not_taken_as_the_segment_held() {
        let (writer, coordinator) = flushed_twice(memory()).await;
        let (source, committed) = own_backup(&writer, &coordinator).await;
        let listed = source.list("").await.unwrap();
        let segment = listed
            .iter()
            .find(|file| file.key.contains("/segment/"))
            .unwrap();
        let mut frame = source.get(&segment.key).await.unwrap().to_vec();
        let last = frame.len() - 1;
        frame[last] ^= 1;
        source.put(&segment.key, Bytes::from(frame)).await.unwrap();

        let refused = restored_from(&writer, source, &committed).await;

        assert!(refused.is_err(), "{refused:?}");
        let now = coordinator
            .committed(&writer.lease.key)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(now, committed);
    }

    /// A store that links nothing, over another.
    #[derive(Debug)]
    struct NoLink(Arc<dyn Blobs>);

    #[async_trait::async_trait]
    impl Blobs for NoLink {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
            self.0.put(key, bytes).await
        }
        async fn put_parts(
            &self,
            key: &str,
            parts: futures::stream::BoxStream<'_, Result<Bytes>>,
        ) -> Result<()> {
            self.0.put_parts(key, parts).await
        }
        async fn get(&self, key: &str) -> Result<Bytes> {
            self.0.get(key).await
        }
        async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
            self.0.list(prefix).await
        }
        async fn delete(&self, key: &str) -> Result<()> {
            self.0.delete(key).await
        }
        async fn local(&self, key: &str) -> Result<Option<Local>> {
            self.0.local(key).await
        }
        async fn fetch_local(&self, key: &str) -> Result<Option<Local>> {
            self.0.fetch_local(key).await
        }
        fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
            self.0.parts(key)
        }
        async fn forget_local(&self, prefix: &str) -> Result<()> {
            self.0.forget_local(prefix).await
        }
        async fn link_local(&self, _from: &str, _to: &str) -> Result<Linked> {
            Ok(Linked::Unlinked)
        }
    }

    /// The vectors files the partition's segments name.
    fn vectors_named(partition: &Partition) -> Vec<String> {
        partition.read(|state| {
            let held = state.loaded_with_vectors();
            held.filter_map(|(_, vectors, _)| vectors.map(str::to_string))
                .collect()
        })
    }

    /// From the review of #73. A segment whose vectors the store does not
    /// link to the new key is not taken as held, though its copies are
    /// proved: it is decoded, and its vectors read anew under the new key,
    /// which then has a copy of its own on local disk.
    #[tokio::test]
    async fn a_restore_whose_vectors_are_not_linked_decodes_and_reads_them_anew() {
        let dir = tempfile::tempdir().unwrap();
        let cached: Arc<dyn Blobs> = Arc::new(crate::blobs::CachedBlobs::new(memory(), dir.path()));
        let blobs: Arc<dyn Blobs> = Arc::new(NoLink(cached));
        let (writer, coordinator) = flushed_twice(blobs.clone()).await;
        let (source, committed) = own_backup(&writer, &coordinator).await;
        let before = segment_files(&writer.partition);
        let vectors_before = vectors_named(&writer.partition);
        assert!(!vectors_before.is_empty(), "the segments have vectors");

        restored_from(&writer, source, &committed).await.unwrap();

        let after = segment_files(&writer.partition);
        for held in &before {
            assert!(
                !after.contains(held),
                "a segment was taken though its vectors were not linked"
            );
        }
        assert_eq!(ns(&writer.partition), [true, true, true]);
        let vectors_after = vectors_named(&writer.partition);
        let new = vectors_after
            .iter()
            .filter(|key| !vectors_before.contains(key));
        for key in new {
            let local = blobs.local(key).await.unwrap();
            assert!(local.is_some(), "{key} has no copy of its own");
        }
    }

    /// A backup that gives each file in many small parts: what the source
    /// sent is summed over all of them, and so each segment the live state
    /// holds is still taken as it is.
    #[tokio::test]
    async fn a_restore_from_its_own_backup_sums_every_part_the_source_sends() {
        let (writer, coordinator) = flushed_twice(memory()).await;
        let small = Transfers {
            part_bytes: 16,
            ..Transfers::default()
        };
        let parted = ObjectBlobs::new(Arc::new(InMemory::new())).transferring(small);
        let (source, committed) = own_backup_in(&writer, &coordinator, Arc::new(parted)).await;
        let segments = source.list("").await.unwrap();
        let segment = segments.iter().find(|file| file.key.contains("/segment/"));
        let parts = source.parts(&segment.unwrap().key);
        use futures::StreamExt;
        assert!(parts.count().await > 2, "the backup sends a file in parts");
        let before = segment_files(&writer.partition);

        restored_from(&writer, source, &committed).await.unwrap();

        let after = segment_files(&writer.partition);
        for held in &before {
            assert!(after.contains(held), "a segment was decoded again");
        }
        assert_eq!(ns(&writer.partition), [true, true, true]);
    }

    /// A backup whose checkpoint names a segment the live state holds,
    /// with its vectors under another key than the live state's: the
    /// segment is not taken as held, though both files come out what the
    /// backup sent. It is decoded from the copies.
    #[tokio::test]
    async fn a_segment_held_with_vectors_under_another_key_is_not_taken_as_held() {
        let (writer, coordinator) = flushed_twice(memory()).await;
        let (source, committed) = own_backup(&writer, &coordinator).await;
        let listed = source.list("").await.unwrap();
        let checkpoint = listed
            .iter()
            .find(|file| file.key.contains("/manifest/"))
            .unwrap();
        let frame = source.get(&checkpoint.key).await.unwrap();
        let mut manifest: Manifest = decode(&checkpoint.key, Kind::Manifest, &frame).unwrap();
        let first = &mut manifest.segments[0];
        let vectors = first.vectors.clone().unwrap();
        let elsewhere = vectors.replace(".vectors", "-elsewhere.vectors");
        let bytes = source.get(&vectors).await.unwrap();
        source.put(&elsewhere, bytes).await.unwrap();
        first.vectors = Some(elsewhere);
        let held = first.key.clone();
        let frame = encode(Kind::Manifest, &manifest).unwrap();
        source.put(&checkpoint.key, frame).await.unwrap();
        let before = writer.partition.read(|state| {
            let held_now = state.loaded_segments();
            let mut held_now = held_now.filter(|(key, _)| *key == held);
            Arc::as_ptr(&held_now.next().unwrap().1.file)
        });

        restored_from(&writer, source, &committed).await.unwrap();

        let after = segment_files(&writer.partition);
        assert!(
            !after.contains(&before),
            "a segment was taken though its vectors' key is not the live state's"
        );
        assert_eq!(ns(&writer.partition), [true, true, true]);
    }
}
