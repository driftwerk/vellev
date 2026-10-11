//! Immutable files in object storage. A key is written once and never
//! changes, which is what makes caching them on local disk safe.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::{
    GetOptions, GetRange, GetResultPayload, MultipartUpload, ObjectStore, PutPayload,
};

use crate::coordinator::{Generation, LogPosition, PartitionKey};
use crate::format::verify;
use crate::{Result, StorageError};

/// Every call must end: whatever a store waits on, another machine or a
/// local disk, it gives up after a while of its own, as [`ObjectBlobs`] and
/// [`CachedBlobs`] do. A writer waits on these calls while it holds what
/// other work waits for, and bounds none of them itself.
#[async_trait]
pub trait Blobs: fmt::Debug + Send + Sync {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()>;
    async fn get(&self, key: &str) -> Result<Bytes>;
    /// Every file whose key starts with `prefix`.
    async fn list(&self, prefix: &str) -> Result<Vec<Listed>>;
    /// As [`Self::list`], given up as failed once the store has taken
    /// longer than `each` over any one answer: so that how long a listing
    /// may take grows with how much there is to list, and a long one is not
    /// given up for being long. A store that answers a listing all at once
    /// is given `each` for the whole of it.
    async fn list_within(&self, prefix: &str, each: Duration) -> Result<Vec<Listed>> {
        tokio::time::timeout(each, self.list(prefix))
            .await
            .map_err(|_| unanswered(prefix, each))?
    }
    /// Remove a file; one that is not there is no error.
    async fn delete(&self, key: &str) -> Result<()>;
    /// The copy of the file on local disk, if the store has one there now.
    /// Nothing is fetched for it: a store that keeps copies makes one of
    /// each file as it is read or written. The copy is never written again
    /// once it is there.
    async fn local(&self, _key: &str) -> Result<Option<Local>> {
        Ok(None)
    }
    /// The file a part at a time, in order, each no more than the store
    /// reads in one request, so that a large file can be put somewhere
    /// without ever being held whole. By default all of it, as one part.
    fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
        use futures::StreamExt;
        futures::stream::once(self.get(key)).boxed()
    }
    /// Put a file that comes a part at a time, in order, as [`Self::parts`]
    /// of another store gives one, without ever holding it whole. Nothing
    /// is under `key` for it until all of it has come: a failure of
    /// `parts` fails the put and leaves nothing. By default refused, with
    /// nothing taken from `parts`: a store that gathered them to put the
    /// file as one would hold it whole, unseen.
    async fn put_parts(
        &self,
        key: &str,
        _parts: futures::stream::BoxStream<'_, Result<Bytes>>,
    ) -> Result<()> {
        Err(StorageError::Blob {
            key: key.to_string(),
            source: "this store takes no file a part at a time".into(),
        })
    }
    /// Fetch the file afresh into the store's copy of it on local disk, in
    /// place of whatever copy is there, a part at a time and never held
    /// whole in memory, and say where it lies. `None` from a store that
    /// keeps no copies.
    async fn fetch_local(&self, _key: &str) -> Result<Option<Local>> {
        Ok(None)
    }
    /// Drop whatever copies of files under `prefix` the store keeps on
    /// local disk, leaving the files themselves be.
    async fn forget_local(&self, _prefix: &str) -> Result<()> {
        Ok(())
    }
}

/// A copy of a file on local disk: where it lies, and the disk it lies on,
/// by which whatever reads it there is run and given up.
#[derive(Debug, Clone)]
pub struct Local {
    pub path: PathBuf,
    pub disk: Disk,
}

/// The failure of a request to the store about `key` that went unanswered
/// for `within`.
pub fn unanswered(key: &str, within: Duration) -> StorageError {
    StorageError::Blob {
        key: key.to_string(),
        source: format!("no answer within {within:?}").into(),
    }
}

/// A file as a listing shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    pub key: String,
    pub modified: chrono::DateTime<chrono::Utc>,
}

/// Who wrote a file: the lease's generation and the writer instance. Two
/// writers never share both, so no two writers can produce the same key, even
/// when one is reopened on a lease that is still current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Author {
    pub generation: Generation,
    pub writer: uuid::Uuid,
}

/// Where a partition's files live.
#[derive(Debug, Clone, Copy)]
pub struct Layout<'a>(pub &'a PartitionKey);

impl Layout<'_> {
    pub fn prefix(&self) -> String {
        format!("{}/{}/", self.0.tenant, self.0.partition.0)
    }

    pub fn log(&self, position: LogPosition, author: &Author) -> String {
        self.key("log", position, author)
    }

    pub fn segment(&self, position: LogPosition, author: &Author) -> String {
        self.key("segment", position, author)
    }

    pub fn manifest(&self, position: LogPosition, author: &Author) -> String {
        self.key("manifest", position, author)
    }

    /// A segment made by merging others at `position`, named apart from the
    /// one a flush at the same position writes.
    pub fn merged(&self, position: LogPosition, author: &Author) -> String {
        self.named("segment", position, author, "-merged")
    }

    /// The index file of the segment a flush at `position` writes.
    pub fn index(&self, position: LogPosition, author: &Author) -> String {
        self.key("index", position, author)
    }

    /// The index file of the segment a merge at `position` writes.
    pub fn merged_index(&self, position: LogPosition, author: &Author) -> String {
        self.named("index", position, author, "-merged")
    }

    /// The vectors file of the segment a flush at `position` writes.
    pub fn vectors(&self, position: LogPosition, author: &Author) -> String {
        self.key("vectors", position, author)
    }

    /// The vectors file of the segment a merge at `position` writes.
    pub fn merged_vectors(&self, position: LogPosition, author: &Author) -> String {
        self.named("vectors", position, author, "-merged")
    }

    fn key(&self, kind: &str, position: LogPosition, author: &Author) -> String {
        self.named(kind, position, author, "")
    }

    pub(crate) fn named(
        &self,
        kind: &str,
        position: LogPosition,
        author: &Author,
        suffix: &str,
    ) -> String {
        format!(
            "{}{kind}/{:020}-{:020}-{}{suffix}.{kind}",
            self.prefix(),
            position.0,
            author.generation.0,
            author.writer.simple()
        )
    }

    /// The position a file under this partition was written at, or `None`
    /// for a key not laid out this way.
    pub(crate) fn position(&self, key: &str) -> Option<LogPosition> {
        let (kind, name) = key.strip_prefix(&self.prefix())?.split_once('/')?;
        if !matches!(kind, "log" | "segment" | "manifest" | "index" | "vectors") {
            return None;
        }
        name.get(..20)?.parse().ok().map(LogPosition)
    }
}

/// Whether `key` is a plain path: no empty, `.` or `..` parts and no
/// backslash. Keys read out of files are data, and only a plain one names
/// the same object to every store and to the cache.
pub(crate) fn is_plain(key: &str) -> bool {
    !key.contains('\\') && key.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

/// How files go to and come from object storage, and when a request is
/// given up.
///
/// A file of more than `part_bytes` goes up in parts of that size, each a
/// request of its own, so that none is ever larger than a store takes in
/// one (5 GiB on S3) and a slow upload of a large file is never mistaken
/// for a stopped one; a file comes down likewise, a range of `part_bytes`
/// at a time. Each request, a whole small file, the start or end of an
/// upload in parts, one part, or one range, is given up after
/// `request_within`, which must allow for a part at the slowest the store
/// is expected to move it, and for the store client's own tries of it. An
/// upload given up, or dropped unfinished, is aborted, its parts discarded;
/// one cut off too abruptly for that, as by a crash, leaves parts that a
/// bucket's lifecycle rule must discard (see the README).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transfers {
    pub part_bytes: usize,
    pub request_within: Duration,
}

/// The most parts an upload may have, as S3 allows.
pub const MOST_PARTS: usize = 10_000;

/// How many requests may be made for one range of a file coming down
/// before the read is given up: each after the first goes on from the
/// bytes already received.
const RANGE_TRIES: usize = 2;

impl Transfers {
    /// How many parts or ranges a file of `bytes` takes.
    pub fn parts(&self, bytes: usize) -> usize {
        bytes.div_ceil(self.part_bytes.max(1)).max(1)
    }

    /// How many requests an upload of `bytes` takes at most: one for a file
    /// of a part or less; else one to start it, one a part, one to complete
    /// it and, if it fails, one to abort it.
    pub fn upload_requests(&self, bytes: usize) -> usize {
        match bytes > self.part_bytes {
            true => self.parts(bytes) + 3,
            false => 1,
        }
    }

    /// The most an upload of `bytes` takes before it is given up.
    pub fn upload_within(&self, bytes: usize) -> Duration {
        self.request_within * requests(self.upload_requests(bytes))
    }

    /// The most a read of `bytes` takes before it is given up: each range
    /// asked for as often as it may be.
    pub fn read_within(&self, bytes: usize) -> Duration {
        self.request_within * requests(self.parts(bytes) * RANGE_TRIES)
    }

    /// The most one part of a file takes to come down before the read is
    /// given up: asked for as often as it may be.
    pub fn part_within(&self) -> Duration {
        self.request_within * requests(RANGE_TRIES)
    }
}

/// `count` requests, as a factor of a duration.
fn requests(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

impl Default for Transfers {
    /// Parts of 16 MiB, each request given up after three minutes and seven
    /// seconds: four tries of forty-six seconds, thirty and what 16 MiB
    /// take at a mebibyte a second, as the store client is set up to make
    /// (see `vellev::config`), and the waits between them.
    fn default() -> Self {
        Self {
            part_bytes: 16 * 1024 * 1024,
            request_within: Duration::from_secs(4 * 46 + 3),
        }
    }
}

/// Where steps on local disk run, and how long each is waited for: on
/// threads of its own, at most `threads` at once, each step waited for
/// `floor` and as long again as its bytes take at `bytes_per_second`,
/// slower than which a disk is failing.
///
/// With every thread taken, a step waits its turn for one, first come
/// first served, for as long as it would itself be waited for, so that a
/// disk that is busy and answering is not taken for one that is not. A
/// step given up runs on to its end on its thread, which it keeps until
/// then, and whoever waited carries on without it. With every thread kept
/// by a step already given up, a step is refused at once, so that a disk
/// that has stopped answering costs each caller nothing more, and never
/// takes a thread from the runtime's own, on which segments are built.
///
/// A step that reads through a whole large file says how far it has got
/// as it goes (see [`Progress`]), and is given up only once it has said
/// nothing for as long as one piece of the file is waited for: however
/// slow the disk, such a step ends, and only one on a disk that has
/// stopped is given up; nor one that only crawls, slower all told than its
/// share of a quarter of `bytes_per_second`, which would keep its turn
/// from every other for hours. No more than [`LOADING_AT_ONCE`] of them
/// run at once on a disk unless it is set otherwise, the rest waiting
/// their turn, so that they do not slow one another down, for as long as
/// those with a turn get on (see [`Disk::loading_still_for`]).
#[derive(Debug, Clone)]
pub struct Disk {
    floor: Duration,
    bytes_per_second: u64,
    threads: Arc<Threads>,
    /// Held by each step that reads or writes a whole large file, as many
    /// as there are turns.
    loading: Arc<tokio::sync::Semaphore>,
    turns: usize,
    /// How long those waiting for a turn go on waiting while nothing
    /// moves: see [`Self::loading_still_for`].
    still_for: Duration,
    /// Counts every turn given and every piece done by whoever has one.
    moved: Arc<AtomicU64>,
    /// How many turns are had now, and how often that has changed.
    loads: Arc<Loads>,
}

/// How many turns at a large file are had on a disk, and how many times
/// one has been given or given back, kept in one word so that the two are
/// only ever read as they stood together: whoever sees fewer turns had
/// than it saw before sees, in the same look, that a turn was given back.
#[derive(Debug, Default)]
struct Loads(AtomicU64);

impl Loads {
    /// The low bits hold how many turns are had, far more than a disk
    /// ever has; the rest count the changes.
    const HAD: u64 = (1 << 16) - 1;
    const CHANGE: u64 = Self::HAD + 1;

    /// Count a turn given.
    fn given(&self) {
        self.0.fetch_add(Self::CHANGE + 1, Ordering::Relaxed);
    }

    /// Count a turn given back: one fewer had, and one more change.
    fn given_back(&self) {
        self.0.fetch_add(Self::CHANGE - 1, Ordering::Relaxed);
    }

    /// How many turns are had, and how many times that has changed.
    fn now(&self) -> (u64, u64) {
        let both = self.0.load(Ordering::Relaxed);
        (both & Self::HAD, both / Self::CHANGE)
    }
}

/// A turn to read or write a whole large file on a [`Disk`], given back
/// when this goes.
#[derive(Debug)]
pub(crate) struct Loading {
    _turn: tokio::sync::OwnedSemaphorePermit,
    loads: Arc<Loads>,
}

impl Drop for Loading {
    fn drop(&mut self) {
        self.loads.given_back();
    }
}

/// What a long step owes by now, going no slower than it may: its share
/// of the least a disk gives, for as long as it has been going, the share
/// being what it was at each time and not what it is now. A file read
/// beside another owes half as much for that time as one read alone, and
/// goes on owing no more for it once the other is done.
#[derive(Debug, Default)]
struct Pace {
    /// Owed so far, in bytes for each thousandth of a second at a byte a
    /// second.
    owed: u128,
    /// How long the step had been owing when this was last brought up to
    /// date, and how many turns had been given and given back by then.
    at: Duration,
    shifts: u64,
}

impl Pace {
    /// Bring what is owed up to `at`, at `least` bytes a second shared
    /// between `beside` files: as many as are being read now, if no turn
    /// was given or given back since the last time, which `shifts` tells.
    /// If one was, how many were read meanwhile is not known, and the
    /// time since is owed for as if they were `most`, as many as ever
    /// are: a step is never held to more than its share was.
    fn owe(&mut self, at: Duration, shifts: u64, beside: u64, most: u64, least: u64) {
        let shared = match shifts == self.shifts {
            true => beside,
            false => most,
        };
        let since = at.saturating_sub(self.at);
        self.owed += u128::from(least / shared.max(1)) * since.as_millis();
        self.at = self.at.max(at);
        self.shifts = shifts;
    }

    /// Whether a step that has done `done` bytes has done less than it
    /// owes.
    fn behind(&self, done: u64) -> bool {
        u128::from(done) * 1_000 < self.owed
    }
}

/// The threads a [`Disk`] runs its steps on.
#[derive(Debug)]
struct Threads {
    free: Arc<tokio::sync::Semaphore>,
    most: usize,
    /// When each step now on a thread is given up, by a number of its own.
    due: std::sync::Mutex<HashMap<u64, tokio::time::Instant>>,
    steps: AtomicU64,
}

/// How many whole large files are read or written at once on one disk,
/// unless it is set otherwise.
pub const LOADING_AT_ONCE: usize = 2;

/// The most whole large files a disk may be set to read or write at once.
pub const MOST_LOADING_AT_ONCE: usize = Loads::HAD as usize;

/// How much longer those waiting for a turn at a large file go on waiting
/// than anyone with a turn may go without getting on.
const LOADING_STILL_BEYOND: Duration = Duration::from_secs(30);

/// How far a long step on local disk has got, as the step itself tells,
/// and with it how far everything reading or writing large files on its
/// disk has.
#[derive(Debug, Clone, Default)]
pub struct Progress(Arc<AtomicU64>, Arc<AtomicU64>);

impl Progress {
    /// Tell of `bytes` more done.
    pub fn made(&self, bytes: usize) {
        self.0.fetch_add(bytes as u64, Ordering::Relaxed);
        self.1.fetch_add(1, Ordering::Relaxed);
    }

    fn so_far(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

impl Threads {
    /// Whether every thread is kept by a step that was given up.
    fn stuck(&self) -> bool {
        let now = tokio::time::Instant::now();
        let due = self.due.lock().expect("disk steps lock");
        due.len() >= self.most && due.values().all(|due| *due <= now)
    }
}

/// A thread of a [`Disk`] taken for one step, free again once this goes.
struct Stepping {
    threads: Arc<Threads>,
    step: u64,
    _thread: tokio::sync::OwnedSemaphorePermit,
}

impl Drop for Stepping {
    fn drop(&mut self) {
        let mut due = self.threads.due.lock().expect("disk steps lock");
        due.remove(&self.step);
    }
}

impl Disk {
    pub fn new(floor: Duration, bytes_per_second: u64, threads: usize) -> Self {
        Self {
            floor,
            bytes_per_second,
            threads: Arc::new(Threads {
                free: Arc::new(tokio::sync::Semaphore::new(threads)),
                most: threads,
                due: std::sync::Mutex::default(),
                steps: AtomicU64::new(0),
            }),
            loading: Arc::new(tokio::sync::Semaphore::new(LOADING_AT_ONCE)),
            turns: LOADING_AT_ONCE,
            still_for: Duration::ZERO,
            moved: Arc::default(),
            loads: Arc::default(),
        }
        .fetching(&Transfers::default())
    }

    /// As this disk, with files fetched to it from a store as `transfers`
    /// says, which decides how long a turn at a large file is waited for
    /// while nothing moves: see [`Self::loading_still_for`].
    pub fn fetching(self, transfers: &Transfers) -> Self {
        // Its file made, its part fetched, then either that part written
        // or, if it could not be fetched, the file looked at to see
        // whether it is empty.
        let written = self.answers_within(transfers.part_bytes);
        let part = self
            .answers_within(0)
            .saturating_add(transfers.part_within())
            .saturating_add(written.max(transfers.request_within));
        Self {
            still_for: part.saturating_add(LOADING_STILL_BEYOND),
            ..self
        }
    }

    /// How long those waiting for a turn at a large file go on waiting
    /// while nothing moves: no turn is given and nobody who has one gets
    /// any further. While anything moves, a turn is waited for however
    /// long it takes: a tenant of many large files, or many tenants
    /// opened at once, all get theirs.
    ///
    /// It is longer than anyone with a turn may go without getting on,
    /// whatever the store is given to answer in: the longest is a file
    /// being fetched, from its turn to its first part written. That is
    /// the file made on disk, a part come from the store, asked for as
    /// often as it may be ([`Transfers::part_within`]), and the part
    /// written, each step on disk having waited for a thread
    /// ([`Self::answers_within`]); or, in place of the writing, one more
    /// request to the store, which a part that could not be fetched is
    /// followed by.
    pub fn loading_still_for(&self) -> Duration {
        self.still_for
    }

    /// The disk every store and writer of this process uses unless given
    /// another: ten seconds and twenty MiB a second, on at most 64 threads.
    pub fn shared() -> &'static Disk {
        static SHARED: std::sync::OnceLock<Disk> = std::sync::OnceLock::new();
        SHARED.get_or_init(|| Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 64))
    }

    /// How long a step on `bytes` is waited for once it has a thread, and
    /// as long at most for the thread before that.
    pub fn within(&self, bytes: usize) -> Duration {
        let carrying = (bytes as u64).saturating_mul(1_000) / self.bytes_per_second.max(1);
        self.floor.saturating_add(Duration::from_millis(carrying))
    }

    /// The most a step on `bytes` takes before whoever asked for it has an
    /// answer: its wait for a thread, and its wait on that thread.
    pub fn answers_within(&self, bytes: usize) -> Duration {
        self.within(bytes).saturating_mul(2)
    }

    /// A thread for a step that is waited for `within`, waited for as long
    /// at most, and not at all when every thread is kept by a step that
    /// was given up.
    async fn thread(&self, within: Duration) -> std::io::Result<Stepping> {
        let busy = |reason: String| std::io::Error::new(std::io::ErrorKind::ResourceBusy, reason);
        let free = self.threads.free.clone();
        let thread = match free.clone().try_acquire_owned() {
            Ok(thread) => thread,
            Err(_) if self.threads.stuck() => {
                return Err(busy(
                    "every thread for local disk is kept by a step it never answered".into(),
                ))
            }
            Err(_) => tokio::time::timeout(within, free.acquire_owned())
                .await
                .map_err(|_| {
                    busy(format!(
                        "no thread for local disk came free within {within:?}"
                    ))
                })?
                .map_err(std::io::Error::other)?,
        };
        let step = self.threads.steps.fetch_add(1, Ordering::Relaxed);
        let due = tokio::time::Instant::now() + within;
        self.threads
            .due
            .lock()
            .expect("disk steps lock")
            .insert(step, due);
        Ok(Stepping {
            threads: self.threads.clone(),
            step,
            _thread: thread,
        })
    }

    /// As this disk, reading or writing `at_once` whole large files at
    /// once, instead of [`LOADING_AT_ONCE`]; at least one. More than
    /// [`MOST_LOADING_AT_ONCE`] is refused: no disk reads so many at
    /// once to any purpose, and they are counted in as much room as
    /// holds that many.
    pub fn loading_at_once(self, at_once: usize) -> std::io::Result<Self> {
        if at_once > MOST_LOADING_AT_ONCE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "a disk reads at most {MOST_LOADING_AT_ONCE} large files at once, \
                     not {at_once}"
                ),
            ));
        }
        Ok(Self {
            loading: Arc::new(tokio::sync::Semaphore::new(at_once.max(1))),
            turns: at_once.max(1),
            moved: Arc::default(),
            loads: Arc::default(),
            ..self
        })
    }

    /// A turn to read or write a whole large file on this disk, waited
    /// for behind those having theirs for as long as anything moves: a
    /// turn is given, or someone with one gets further. Only when nothing
    /// has for [`Self::loading_still_for`] is whoever waits told to ask
    /// again. Taken before any thread, so that none is kept by a step
    /// waiting, and kept by whoever has it until the thread it works on
    /// is done, not only until it stops waiting for that.
    pub(crate) async fn loading(&self) -> std::io::Result<Loading> {
        let loading = self.loading.clone().acquire_owned();
        tokio::pin!(loading);
        let still_for = self.still_for;
        let mut seen = self.moved.load(Ordering::Relaxed);
        loop {
            match tokio::time::timeout(still_for, &mut loading).await {
                Ok(turn) => {
                    let turn = turn.map_err(std::io::Error::other)?;
                    self.moved.fetch_add(1, Ordering::Relaxed);
                    self.loads.given();
                    return Ok(Loading {
                        _turn: turn,
                        loads: self.loads.clone(),
                    });
                }
                Err(_) if self.moved.load(Ordering::Relaxed) == seen => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::ResourceBusy,
                        format!(
                            "nothing reading or writing a large file on local disk \
                             got any further in {still_for:?}"
                        ),
                    ))
                }
                Err(_) => seen = self.moved.load(Ordering::Relaxed),
            }
        }
    }

    /// Word of how far a step with a turn at a large file has got.
    pub(crate) fn progress(&self) -> Progress {
        Progress(Arc::default(), self.moved.clone())
    }

    /// How many whole large files this disk reads or writes at once.
    pub fn loads_at_once(&self) -> usize {
        self.turns
    }

    /// Run `step`, which works through something large on local disk a
    /// piece of at most `piece` bytes at a time, telling of each as it is
    /// done, and wait for it for as long as it gets on, and gets on at
    /// its share of a quarter of the disk's `bytes_per_second` or better
    /// all told (see [`Pace`]): it is given up once it falls behind that,
    /// or once it has told of nothing for as long as a step on one piece
    /// is waited for.
    pub(crate) async fn run_while_it_gets_on<T: Send + 'static>(
        &self,
        piece: usize,
        step: impl FnOnce(&Progress) -> std::io::Result<T> + Send + 'static,
    ) -> std::io::Result<T> {
        let within = self.within(piece);
        let thread = self.thread(within).await?;
        let (threads, number) = (self.threads.clone(), thread.step);
        let progress = self.progress();
        let (done, mut waited) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("vellev-disk".into())
            .spawn({
                let progress = progress.clone();
                move || {
                    let _thread = thread;
                    let _ = done.send(step(&progress));
                }
            })?;
        // What it must have done by each time it is looked at, to be no
        // slower than a quarter of what the disk gives, shared between as
        // many large files as were read or written on it at once: owed
        // from once it has had as long as a piece is waited for to begin.
        let begun = tokio::time::Instant::now();
        let least = self.bytes_per_second / 4;
        // Nothing is owed for the time up to the first look, which is
        // where the count of changes is first taken from.
        let mut pace = Pace::default();
        let mut seen = progress.so_far();
        loop {
            let stepped = tokio::time::timeout(within, &mut waited).await;
            let owing = begun.elapsed().saturating_sub(within);
            // How many are read, and whether that has changed, in one
            // look (see [`Loads`]): a neighbour gone since the last look
            // is seen as a change, and the time it was there is never
            // owed for as if it was not. The step's own turn is one of
            // them, where it has one.
            let (had, shifts) = self.loads.now();
            let (beside, most) = (had.max(1), self.turns as u64);
            pace.owe(owing, shifts, beside, most, least);
            match stepped {
                Ok(stepped) => return stepped.map_err(std::io::Error::other)?,
                Err(_) if progress.so_far() == seen => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("local disk got no further within {within:?}"),
                    ))
                }
                Err(_) if pace.behind(progress.so_far()) => {
                    let slowest = least / beside;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("local disk is slower than {slowest} bytes a second"),
                    ));
                }
                Err(_) => {
                    seen = progress.so_far();
                    let mut due = threads.due.lock().expect("disk steps lock");
                    if let Some(due) = due.get_mut(&number) {
                        *due = tokio::time::Instant::now() + within;
                    }
                }
            }
        }
    }

    /// Run `step`, which works on `bytes` on local disk, waiting for it at
    /// most [`Self::within`] of them once it has a thread.
    pub(crate) async fn run<T: Send + 'static>(
        &self,
        bytes: usize,
        step: impl FnOnce() -> std::io::Result<T> + Send + 'static,
    ) -> std::io::Result<T> {
        let within = self.within(bytes);
        let thread = self.thread(within).await?;
        let (done, waited) = tokio::sync::oneshot::channel();
        std::thread::Builder::new()
            .name("vellev-disk".into())
            .spawn(move || {
                let _thread = thread;
                let _ = done.send(step());
            })?;
        match tokio::time::timeout(within, waited).await {
            Ok(stepped) => stepped.map_err(std::io::Error::other)?,
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("local disk did not answer within {within:?}"),
            )),
        }
    }

    /// How many bytes the file at `path` has.
    pub(crate) async fn sized(&self, path: &Path) -> std::io::Result<usize> {
        let path = path.to_path_buf();
        let size = self.run(0, move || std::fs::metadata(path).map(|meta| meta.len()));
        Ok(usize::try_from(size.await?).unwrap_or(usize::MAX))
    }
}

impl Local {
    /// The file at `path` on `disk`, if one is there as far as the disk
    /// says in time.
    async fn on(disk: &Disk, path: PathBuf) -> Option<Self> {
        let disk = disk.clone();
        disk.sized(&path).await.ok().map(|_| Self { path, disk })
    }
}

#[derive(Debug, Clone)]
pub struct ObjectBlobs {
    store: Arc<dyn ObjectStore>,
    /// Where the store keeps its files when it is a local directory, whose
    /// files are then read in place.
    root: Option<PathBuf>,
    transfers: Transfers,
}

impl ObjectBlobs {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            root: None,
            transfers: Transfers::default(),
        }
    }

    /// A store that is the local directory `root`, which writes a file
    /// whole under another name and renames it into place, as
    /// `object_store`'s local file system does.
    pub fn on_disk(store: Arc<dyn ObjectStore>, root: impl Into<PathBuf>) -> Self {
        Self {
            root: Some(root.into()),
            ..Self::new(store)
        }
    }

    /// Transfer as `transfers` says, instead of as by default.
    pub fn transferring(self, transfers: Transfers) -> Self {
        Self { transfers, ..self }
    }

    /// Upload `bytes` under `key` in parts, each its own request, and abort
    /// the upload if any of it fails or it is dropped unfinished.
    async fn put_in_parts(&self, key: &str, path: &ObjectPath, bytes: Bytes) -> Result<()> {
        let within = self.transfers.request_within;
        let mut upload = Unfinished {
            upload: Some(bounded(key, within, self.store.put_multipart(path)).await?),
            within,
        };
        let part = self.transfers.part_bytes.max(1);
        let uploaded = async {
            for start in (0..bytes.len()).step_by(part) {
                let piece = bytes.slice(start..(start + part).min(bytes.len()));
                let sent = upload.parts().put_part(PutPayload::from_bytes(piece));
                bounded(key, within, sent).await?;
            }
            bounded(key, within, upload.parts().complete()).await
        }
        .await;
        match uploaded {
            Ok(_) => {
                upload.upload = None;
                Ok(())
            }
            Err(error) => {
                upload.abort(key).await;
                Err(error)
            }
        }
    }

    /// Bring `reading` of the file under `key` as far as `upto`, or its end
    /// if that comes first, with at most [`RANGE_TRIES`] requests, each
    /// going on from the bytes already received. A file not there is told
    /// at once.
    async fn range(
        &self,
        key: &str,
        path: &ObjectPath,
        reading: &mut Reading,
        upto: u64,
    ) -> Result<()> {
        let within = self.transfers.request_within;
        let mut tries = 1;
        loop {
            let asked = bounded(key, within, reading.more(self.store.as_ref(), path, upto));
            match asked.await.and_then(|()| reading.reached(key, upto)) {
                Err(StorageError::Missing { key }) => return Err(StorageError::Missing { key }),
                Err(error) if tries < RANGE_TRIES => {
                    tracing::warn!(key, %error, at = reading.at(), "a range of a file is asked for again");
                    tries += 1;
                }
                read => return read,
            }
        }
    }

    /// Read the next range of the file under `key` into `reading`, of at
    /// most a part, and say whether there was one: not once the file's end
    /// is reached, nor of a file with nothing in it.
    async fn next(&self, key: &str, path: &ObjectPath, reading: &mut Reading) -> Result<bool> {
        let part = self.transfers.part_bytes.max(1) as u64;
        match reading.size() {
            Some(size) if reading.at() >= size => Ok(false),
            Some(size) => {
                let upto = (reading.at() + part).min(size);
                self.range(key, path, reading, upto).await.map(|()| true)
            }
            None => match self.range(key, path, reading, part).await {
                Ok(()) => Ok(true),
                Err(error) => match self.empty(key, path).await {
                    true => Ok(false),
                    false => Err(error),
                },
            },
        }
    }

    /// Whether the file under `key` is there and empty, which no range of
    /// it can be asked for to tell.
    async fn empty(&self, key: &str, path: &ObjectPath) -> bool {
        let within = self.transfers.request_within;
        matches!(bounded(key, within, self.store.head(path)).await, Ok(meta) if meta.size == 0)
    }
}

type ObjectPath = object_store::path::Path;

/// A file coming down: what has been received of it and not yet taken,
/// how much was taken before that, and the version and size the first
/// answer gave, which every later range must be of. One read whole has
/// room made for all of it once its size is known; one read in parts
/// never holds more than a part.
#[derive(Debug, Default)]
struct Reading {
    received: bytes::BytesMut,
    taken: u64,
    whole: bool,
    of: Option<(Option<String>, u64)>,
}

impl Reading {
    /// Of a file that is wanted whole, in one piece.
    fn whole() -> Self {
        Self {
            whole: true,
            ..Self::default()
        }
    }

    /// How far into the file what has come reaches.
    fn at(&self) -> u64 {
        self.taken + self.received.len() as u64
    }

    /// Take what has been received, to go on from after it.
    fn take(&mut self) -> Bytes {
        let part = self.received.split().freeze();
        self.taken += part.len() as u64;
        part
    }

    /// Ask for the bytes from those received up to `upto` and take them in
    /// as they come, so that what came before a failure is kept.
    async fn more(
        &mut self,
        store: &dyn ObjectStore,
        path: &ObjectPath,
        upto: u64,
    ) -> object_store::Result<()> {
        use futures::{StreamExt, TryStreamExt};
        let options = GetOptions {
            range: Some(GetRange::Bounded(self.at()..upto)),
            if_match: self.of.as_ref().and_then(|(version, _)| version.clone()),
            ..GetOptions::default()
        };
        let found = store.get_opts(path, options).await?;
        let (version, size) = (found.meta.e_tag.clone(), found.meta.size);
        match &self.of {
            Some((_, pinned)) if *pinned != size => {
                return Err(object_store::Error::Generic {
                    store: "vellev",
                    source: format!("{path} was {pinned} bytes and is now {size}").into(),
                })
            }
            Some(_) => {}
            None => {
                let left = size.saturating_sub(self.at());
                if self.whole {
                    self.received
                        .reserve(usize::try_from(left).unwrap_or(usize::MAX));
                }
                self.of = Some((version, size));
            }
        }
        // A file on local disk is read in one go; any other as it arrives.
        let mut arriving = match &found.payload {
            GetResultPayload::Stream(_) => found.into_stream(),
            _ => futures::stream::once(found.bytes()).boxed(),
        };
        while let Some(piece) = arriving.try_next().await? {
            self.received.extend_from_slice(&piece);
        }
        Ok(())
    }

    /// The file's size, once an answer has given it.
    fn size(&self) -> Option<u64> {
        self.of.as_ref().map(|(_, size)| *size)
    }

    /// Whether what was received reaches exactly `upto` or the file's end.
    fn reached(&self, key: &str, upto: u64) -> Result<()> {
        let wanted = self.size().map_or(upto, |size| size.min(upto));
        match self.at() == wanted {
            true => Ok(()),
            false => Err(StorageError::corrupt(
                key,
                format!("{} bytes came where {wanted} were asked for", self.at()),
            )),
        }
    }
}

/// An upload in parts not yet complete, aborted if it is dropped so.
#[derive(Debug)]
struct Unfinished {
    upload: Option<Box<dyn MultipartUpload>>,
    within: Duration,
}

impl Unfinished {
    fn parts(&mut self) -> &mut Box<dyn MultipartUpload> {
        self.upload.as_mut().expect("an unfinished upload")
    }

    /// Abort the upload now, waiting for it. Until that is done the upload
    /// stays unfinished, so that dropping this part way aborts it still.
    async fn abort(&mut self, key: &str) {
        if let Some(upload) = self.upload.as_mut() {
            if let Err(error) = bounded(key, self.within, upload.abort()).await {
                tracing::warn!(key, %error, "an upload given up could not be aborted");
            }
        }
        self.upload = None;
    }
}

impl Drop for Unfinished {
    /// An upload dropped unfinished, by a caller that stopped waiting, is
    /// aborted on a task of its own.
    fn drop(&mut self) {
        let (Some(mut upload), Ok(runtime)) =
            (self.upload.take(), tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let within = self.within;
        runtime.spawn(async move {
            if let Err(error) = bounded("an upload", within, upload.abort()).await {
                tracing::warn!(%error, "an upload dropped unfinished could not be aborted");
            }
        });
    }
}

/// `request` to the store for `key`, given up after `within`.
async fn bounded<T>(
    key: &str,
    within: Duration,
    request: impl std::future::Future<Output = object_store::Result<T>>,
) -> Result<T> {
    match tokio::time::timeout(within, request).await {
        Ok(answered) => answered.map_err(|error| blob_error(key, error)),
        Err(_) => Err(StorageError::Blob {
            key: key.to_string(),
            source: format!("the store did not answer within {within:?}").into(),
        }),
    }
}

fn blob_error(key: &str, error: object_store::Error) -> StorageError {
    match error {
        object_store::Error::NotFound { .. } => StorageError::Missing {
            key: key.to_string(),
        },
        other => StorageError::Blob {
            key: key.to_string(),
            source: Box::new(other),
        },
    }
}

#[async_trait]
impl Blobs for ObjectBlobs {
    /// A file larger than [`Transfers::part_bytes`] goes up in parts. One
    /// too large for [`MOST_PARTS`] of them is refused before any of it
    /// goes up.
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
        let path = ObjectPath::from(key);
        if self.transfers.parts(bytes.len()) > MOST_PARTS {
            return Err(StorageError::Blob {
                key: key.to_string(),
                source: format!(
                    "{} bytes need more than {MOST_PARTS} parts of {}",
                    bytes.len(),
                    self.transfers.part_bytes
                )
                .into(),
            });
        }
        if bytes.len() > self.transfers.part_bytes {
            return self.put_in_parts(key, &path, bytes).await;
        }
        let within = self.transfers.request_within;
        let payload = PutPayload::from_bytes(bytes);
        bounded(key, within, self.store.put(&path, payload))
            .await
            .map(|_| ())
    }

    /// As [`Self::put`] sends the same file, whatever the sizes its parts
    /// come in: parts of [`Transfers::part_bytes`] and a last one of what
    /// is left, each sent once it is known that more follows it, or whole
    /// if a part or less is all that comes. No more than a part of what
    /// came is held beside what last came; but a piece cut from a part
    /// that came keeps all of that part alive, so what is held at once is
    /// as many of the parts that came as one part is cut from, the first
    /// perhaps only for its end, and the one that came last. An empty one
    /// is let go at once. How large the file is cannot be known before it
    /// has come, so one too large for [`MOST_PARTS`] is refused only at the
    /// first part too many, and what went up of it aborted.
    async fn put_parts(
        &self,
        key: &str,
        mut parts: futures::stream::BoxStream<'_, Result<Bytes>>,
    ) -> Result<()> {
        use futures::TryStreamExt;
        let path = ObjectPath::from(key);
        let within = self.transfers.request_within;
        let part = self.transfers.part_bytes.max(1);
        let mut upload = Unfinished {
            upload: None,
            within,
        };
        // What has come and not gone up, no more than a part of it, and
        // how many parts have.
        let (mut held, mut holding, mut sent) = (Vec::new(), 0, 0);
        let too_many = || StorageError::Blob {
            key: key.to_string(),
            source: format!("more than {MOST_PARTS} parts of {part} bytes came").into(),
        };
        let uploaded = async {
            while let Some(mut came) = parts.try_next().await? {
                if came.is_empty() {
                    continue;
                }
                while holding + came.len() > part {
                    held.push(came.split_to(part - holding));
                    if upload.upload.is_none() {
                        let started = bounded(key, within, self.store.put_multipart(&path));
                        upload.upload = Some(started.await?);
                    }
                    if sent == MOST_PARTS {
                        return Err(too_many());
                    }
                    let piece = PutPayload::from_iter(std::mem::take(&mut held));
                    bounded(key, within, upload.parts().put_part(piece)).await?;
                    (holding, sent) = (0, sent + 1);
                }
                holding += came.len();
                held.push(came);
            }
            let last = PutPayload::from_iter(held);
            if upload.upload.is_none() {
                let put = bounded(key, within, self.store.put(&path, last));
                return put.await.map(|_| ());
            }
            if sent == MOST_PARTS {
                return Err(too_many());
            }
            bounded(key, within, upload.parts().put_part(last)).await?;
            bounded(key, within, upload.parts().complete()).await?;
            Ok(())
        }
        .await;
        match uploaded {
            Ok(()) => {
                upload.upload = None;
                Ok(())
            }
            Err(error) => {
                upload.abort(key).await;
                Err(error)
            }
        }
    }

    /// A file comes down a range of [`Transfers::part_bytes`] at a time,
    /// each its own request, every range after the first pinned to the
    /// version and size the first was of.
    async fn get(&self, key: &str) -> Result<Bytes> {
        let path = ObjectPath::from(key);
        let mut reading = Reading::whole();
        while self.next(key, &path, &mut reading).await? {}
        Ok(reading.take())
    }

    /// As [`Self::get`] reads it, each range handed on as it has come.
    fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
        use futures::StreamExt;
        let reading = (Reading::default(), ObjectPath::from(key));
        let parts = futures::stream::try_unfold(reading, move |(mut reading, path)| async move {
            let more = self.next(key, &path, &mut reading).await?;
            Ok(more.then(|| (reading.take(), (reading, path))))
        });
        parts.boxed()
    }

    /// Each page of the listing is a request of its own.
    async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
        use futures::TryStreamExt;
        let within = self.transfers.request_within;
        let under = ObjectPath::from(prefix);
        let mut listing = self.store.list(Some(&under));
        let mut listed = Vec::new();
        while let Some(meta) = bounded(prefix, within, listing.try_next()).await? {
            listed.push(Listed {
                key: meta.location.to_string(),
                modified: meta.last_modified,
            });
        }
        Ok(listed)
    }

    async fn list_within(&self, prefix: &str, each: Duration) -> Result<Vec<Listed>> {
        use futures::TryStreamExt;
        let under = object_store::path::Path::from(prefix);
        let mut listing = self.store.list(Some(&under));
        let mut listed = Vec::new();
        // The store answers a page at a time, and each file of a page at
        // once: the wait for the next file is the wait for its page.
        loop {
            let next = tokio::time::timeout(each, listing.try_next())
                .await
                .map_err(|_| unanswered(prefix, each))?
                .map_err(|e| blob_error(prefix, e))?;
            match next {
                Some(meta) => listed.push(Listed {
                    key: meta.location.to_string(),
                    modified: meta.last_modified,
                }),
                None => return Ok(listed),
            }
        }
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let within = self.transfers.request_within;
        match bounded(key, within, self.store.delete(&ObjectPath::from(key))).await {
            Ok(()) | Err(StorageError::Missing { .. }) => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn local(&self, key: &str) -> Result<Option<Local>> {
        let Some(path) = self.root.as_ref().filter(|_| is_plain(key)) else {
            return Ok(None);
        };
        Ok(Local::on(Disk::shared(), path.join(key)).await)
    }
}

/// Keeps a copy of every file read or written under `dir`. The cache is an
/// optimisation and never the authority: an entry that fails its checksum is
/// dropped and read from the store again, and a failure to read or write
/// one, or to do so as soon as its [`Disk`] waits, is logged and otherwise
/// ignored.
///
/// Several processes on one host may share a cache directory: an entry is
/// written under a name of its writer's own and renamed into place, and
/// only files left part written an hour ago or more are swept away.
#[derive(Debug, Clone)]
pub struct CachedBlobs {
    inner: Arc<dyn Blobs>,
    dir: PathBuf,
    disk: Disk,
}

/// What a file being written into the cache is named while it is written.
const PARTIAL_MARK: &str = "partial-";

/// The name the file at `path` is written into the cache under before it
/// is renamed there: one drawn at random, which no other writer of it
/// shares, be it in this process or in another on the same directory,
/// whose process id and whatever it counts may well be the same.
fn partial(path: &Path) -> PathBuf {
    path.with_extension(format!("{PARTIAL_MARK}{}", uuid::Uuid::new_v4().simple()))
}

/// How long a file left part written in the cache is left alone: far
/// longer than any write of one is waited for, so that none another process
/// is still writing is taken from under it.
const PARTIAL_STALE: Duration = Duration::from_secs(60 * 60);

impl CachedBlobs {
    pub fn new(inner: Arc<dyn Blobs>, dir: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            dir: dir.into(),
            disk: Disk::shared().clone(),
        }
    }

    /// Work on local disk as `disk` says, instead of on the shared one.
    pub fn on(self, disk: Disk) -> Self {
        Self { disk, ..self }
    }

    /// As [`Self::new`], first deleting the files left part written in
    /// `dir` long ago, as far as the disk lets it in time: what is not
    /// swept now is swept on, and does no harm meanwhile.
    pub async fn open(inner: Arc<dyn Blobs>, dir: impl Into<PathBuf>) -> Self {
        let cached = Self::new(inner, dir);
        let dir = cached.dir.clone();
        match cached.disk.run(0, move || Ok(sweep(&dir))).await {
            Ok(0) => {}
            Ok(swept) => tracing::info!(swept, "deleted files left part written in the cache"),
            Err(error) => {
                tracing::warn!(%error, "the cache was not swept of files left part written")
            }
        }
        cached
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        is_plain(key).then(|| self.dir.join(key))
    }

    async fn cached(&self, key: &str, path: &Path) -> Option<Bytes> {
        let size = self.disk.sized(path).await.ok()?;
        let read = {
            let path = path.to_path_buf();
            move || std::fs::read(&path)
        };
        let bytes = match self.disk.run(size, read).await {
            Ok(bytes) => Bytes::from(bytes),
            Err(error) => {
                tracing::warn!(key, %error, "a cache entry could not be read");
                return None;
            }
        };
        match verify(key, &bytes) {
            Ok(()) => Some(bytes),
            Err(error) => {
                tracing::warn!(key, %error, "dropping a damaged cache entry");
                let path = path.to_path_buf();
                crate::column::forget_summed(&path);
                let _ = self.disk.run(0, move || std::fs::remove_file(path)).await;
                None
            }
        }
    }

    /// Remove the copy on local disk of the file under `key`, if there is
    /// one, and what was known of it.
    async fn forget(&self, key: &str) -> Result<()> {
        let Some(path) = self.path(key) else {
            return Ok(());
        };
        crate::column::forget_summed(&path);
        let removed = self.disk.run(0, move || match std::fs::remove_file(path) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        });
        removed.await.map_err(|error| StorageError::Blob {
            key: key.to_string(),
            source: error.into(),
        })
    }

    /// Written under a name of its own and then renamed, so a reader never
    /// sees a partly written file. Nothing is synced: an entry torn by a
    /// crash fails its checksum and is fetched again. The whole of it runs
    /// on one thread, which runs to its end whoever stops waiting, so that
    /// a write given up never leaves the partly written file behind.
    async fn remember(&self, key: &str, bytes: &Bytes) {
        let Some(path) = self.path(key) else { return };
        let written = self.disk.run(bytes.len(), {
            let bytes = bytes.clone();
            move || {
                let parent = path
                    .parent()
                    .ok_or_else(|| std::io::Error::other("cache path has no parent"))?;
                std::fs::create_dir_all(parent)?;
                let partial = partial(&path);
                // Whatever copy there was is replaced, and with it what
                // was known of it.
                crate::column::forget_summed(&path);
                let written = std::fs::write(&partial, &bytes)
                    .and_then(|()| std::fs::rename(&partial, &path));
                if written.is_err() {
                    let _ = std::fs::remove_file(&partial);
                }
                written
            }
        });
        if let Err(error) = written.await {
            tracing::warn!(key, %error, "could not cache a file on local disk");
        }
    }
}

/// A file being fetched into the cache under a name of its own, removed
/// if this is dropped before it is renamed into place: by a failure, or
/// by a caller that stops waiting, as one whose client went away does. It
/// is removed on a task of its own, as a step on the disk like any other.
///
/// A step of the fetch may be on its thread still when this goes, and
/// would write the file after it was removed: each looks, once it has
/// written, whether the fetch was given up meanwhile, and removes the file
/// itself if so. Whichever of the two comes last leaves nothing.
struct Partial {
    disk: Disk,
    path: Option<PathBuf>,
    given_up: Arc<std::sync::atomic::AtomicBool>,
}

impl Partial {
    /// A step that writes to the file as `write` does, and takes it away
    /// again if the fetch was given up by the time it has.
    fn step(
        &self,
        write: impl FnOnce(&Path) -> std::io::Result<()> + Send + 'static,
    ) -> impl FnOnce() -> std::io::Result<()> + Send + 'static {
        let given_up = self.given_up.clone();
        let path = self.path.clone();
        move || {
            let path = path.ok_or_else(|| std::io::Error::other("the fetch is done"))?;
            let written = write(&path);
            if given_up.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(&path);
            }
            written
        }
    }
}

impl Drop for Partial {
    fn drop(&mut self) {
        let (Some(path), Ok(runtime)) = (self.path.take(), tokio::runtime::Handle::try_current())
        else {
            return;
        };
        self.given_up.store(true, Ordering::SeqCst);
        let disk = self.disk.clone();
        runtime.spawn(async move {
            let _ = disk.run(0, move || std::fs::remove_file(path)).await;
        });
    }
}

/// The copy of the file under `key` to forget once the file is put, and
/// forgotten on a task of its own if this is dropped before.
struct Forgetting {
    blobs: Option<CachedBlobs>,
    key: String,
}

impl Drop for Forgetting {
    fn drop(&mut self) {
        let (Some(blobs), Ok(runtime)) = (self.blobs.take(), tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let key = std::mem::take(&mut self.key);
        runtime.spawn(async move {
            if let Err(error) = blobs.forget(&key).await {
                tracing::warn!(key, %error, "a copy of a file put in parts was not forgotten");
            }
        });
    }
}

/// Delete every file under `dir` left part written [`PARTIAL_STALE`] ago
/// or more, returning how many. A file or directory that cannot be read,
/// or goes meanwhile, is passed over.
fn sweep(dir: &Path) -> usize {
    let stale = |entry: &std::fs::DirEntry| {
        let partial = entry
            .path()
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.starts_with(PARTIAL_MARK));
        let age = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|written| written.elapsed().ok());
        partial && age.is_some_and(|age| age >= PARTIAL_STALE)
    };
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(kind) if kind.is_dir() => sweep(&entry.path()),
            Ok(_) if stale(&entry) => usize::from(std::fs::remove_file(entry.path()).is_ok()),
            _ => 0,
        })
        .sum()
}

#[async_trait]
impl Blobs for CachedBlobs {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
        self.inner.put(key, bytes.clone()).await?;
        self.remember(key, &bytes).await;
        Ok(())
    }

    /// Handed to the store as it comes, and no copy of it made: one is
    /// made when it is first read. Any copy of a file that was under `key`
    /// goes first, as for [`Self::delete`], and nothing goes up if it
    /// cannot. Whatever copy there is goes again once the store has
    /// answered, whether the put succeeded or not, and on a task of its
    /// own if the put is dropped before: a read while the file went up
    /// kept a copy of the one that was there. A copy that cannot be
    /// forgotten then fails the put, though the file may be there. What
    /// remains is as beside a put: a read that fetched the file that was
    /// there before this one landed, and keeps its copy after that last
    /// forgetting, leaves the old copy.
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, Result<Bytes>>,
    ) -> Result<()> {
        self.forget(key).await?;
        let mut forgetting = Forgetting {
            blobs: Some(self.clone()),
            key: key.to_string(),
        };
        let put = self.inner.put_parts(key, parts).await;
        let forgot = self.forget(key).await;
        forgetting.blobs = None;
        put.and(forgot)
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        if let Some(path) = self.path(key) {
            if let Some(cached) = self.cached(key, &path).await {
                return Ok(cached);
            }
        }
        let bytes = self.inner.get(key).await?;
        self.remember(key, &bytes).await;
        Ok(bytes)
    }

    async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }

    async fn list_within(&self, prefix: &str, each: Duration) -> Result<Vec<Listed>> {
        self.inner.list_within(prefix, each).await
    }

    /// The copy on local disk goes first, and the file only once it has:
    /// were the file deleted and its copy left, a read would be answered
    /// with a file the store no longer has.
    async fn delete(&self, key: &str) -> Result<()> {
        self.forget(key).await?;
        self.inner.delete(key).await
    }

    async fn local(&self, key: &str) -> Result<Option<Local>> {
        match self.path(key) {
            Some(path) => Ok(Local::on(&self.disk, path).await),
            None => Ok(None),
        }
    }

    fn parts<'a>(&'a self, key: &'a str) -> futures::stream::BoxStream<'a, Result<Bytes>> {
        self.inner.parts(key)
    }

    /// Each part is written to the copy as it comes, under a name of the
    /// copy's own until all of it is there. A disk that does not take a
    /// part in time fails the fetch: nothing is kept in memory instead.
    async fn fetch_local(&self, key: &str) -> Result<Option<Local>> {
        use futures::TryStreamExt;
        let Some(path) = self.path(key) else {
            return Ok(None);
        };
        let unwritten = |error: std::io::Error| StorageError::Blob {
            key: key.to_string(),
            source: error.into(),
        };
        let _loading = self.disk.loading().await.map_err(unwritten)?;
        let partial = partial(&path);
        let mut unfinished = Partial {
            disk: self.disk.clone(),
            path: Some(partial.clone()),
            given_up: Arc::default(),
        };
        let begun = unfinished.step(|partial| {
            let parent = partial.parent();
            let parent = parent.ok_or_else(|| std::io::Error::other("no parent"))?;
            std::fs::create_dir_all(parent)?;
            std::fs::File::create(partial).map(|_| ())
        });
        let begun = self.disk.run(0, begun);
        let fetched = async {
            begun.await.map_err(unwritten)?;
            let mut parts = self.inner.parts(key);
            while let Some(part) = parts.try_next().await? {
                let bytes = part.len();
                let written = unfinished.step(move |partial| {
                    use std::io::Write;
                    let mut copy = std::fs::File::options().append(true).open(partial)?;
                    copy.write_all(&part)
                });
                self.disk.run(bytes, written).await.map_err(unwritten)?;
                self.disk.progress().made(0);
            }
            let (from, to) = (partial.clone(), path.clone());
            crate::column::forget_summed(&path);
            let renamed = self.disk.run(0, move || std::fs::rename(from, to));
            renamed.await.map_err(unwritten)
        };
        // A failure, like a caller that stops waiting, leaves the guard to
        // remove what was written.
        fetched.await?;
        unfinished.path = None;
        Ok(Some(Local {
            path,
            disk: self.disk.clone(),
        }))
    }

    async fn forget_local(&self, prefix: &str) -> Result<()> {
        let Some(path) = self.path(prefix.trim_end_matches('/')) else {
            return Ok(());
        };
        crate::column::forget_summed_under(&path);
        let removed = self
            .disk
            .run(0, move || match std::fs::remove_dir_all(&path) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
                _ => Ok(()),
            });
        removed.await.map_err(|error| StorageError::Blob {
            key: prefix.to_string(),
            source: error.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    #[test]
    fn a_file_written_into_the_cache_goes_under_a_name_drawn_at_random() {
        let path = Path::new("cache/tenant-a/0/vectors/7.vectors");
        let random = |partial: &Path| {
            let extension = partial.extension()?.to_str()?;
            uuid::Uuid::try_parse(extension.strip_prefix(PARTIAL_MARK)?).ok()
        };
        let (one, other) = (partial(path), partial(path));
        assert_eq!(one.parent(), path.parent());
        assert_eq!(one.file_stem(), path.file_stem());
        // Nothing of it is the process's own, which another's may equal.
        let (one, other) = (random(&one).unwrap(), random(&other).unwrap());
        assert_eq!(one.get_version_num(), 4);
        assert_ne!(one, other);
    }

    #[tokio::test]
    async fn room_is_made_for_all_of_a_file_once_its_size_is_known() {
        let store = InMemory::new();
        let path = ObjectPath::from("large");
        let large = Bytes::from(vec![7u8; 100_000]);
        store.put(&path, large.clone().into()).await.unwrap();

        let mut reading = Reading::whole();
        reading.more(&store, &path, 1_000).await.unwrap();
        assert_eq!(reading.received.len(), 1_000);
        assert!(reading.received.capacity() >= large.len());
        let whole = reading.received.as_ptr();
        reading
            .more(&store, &path, large.len() as u64)
            .await
            .unwrap();
        assert_eq!(reading.received.as_ptr(), whole, "never moved to grow");
        assert_eq!(reading.received.freeze(), large);
    }

    #[tokio::test]
    async fn a_file_read_in_parts_is_never_held_whole() {
        let parts = Transfers {
            part_bytes: 1_000,
            ..Transfers::default()
        };
        let blobs = ObjectBlobs::new(Arc::new(InMemory::new())).transferring(parts);
        let large = Bytes::from((0..10_500u32).map(|n| n as u8).collect::<Vec<_>>());
        blobs.put("large", large.clone()).await.unwrap();
        blobs.put("empty", Bytes::new()).await.unwrap();

        use futures::TryStreamExt;
        let read: Vec<Bytes> = blobs.parts("large").try_collect().await.unwrap();
        let sizes: Vec<usize> = read.iter().map(Bytes::len).collect();
        assert_eq!(sizes, [[1_000; 10].as_slice(), &[500]].concat());
        assert_eq!(read.concat(), large);
        let none: Vec<Bytes> = blobs.parts("empty").try_collect().await.unwrap();
        assert!(none.is_empty());
        let missing: Result<Vec<Bytes>> = blobs.parts("missing").try_collect().await;
        assert!(matches!(missing, Err(StorageError::Missing { .. })));

        // No room is made for more than is asked for at a time.
        let (store, path) = (InMemory::new(), ObjectPath::from("large"));
        store.put(&path, large.clone().into()).await.unwrap();
        let mut reading = Reading::default();
        reading.more(&store, &path, 1_000).await.unwrap();
        assert!(reading.received.capacity() < large.len());
        assert_eq!(reading.take().len(), 1_000);
        assert_eq!(reading.at(), 1_000);
    }

    /// A long step on a disk that does what the test tells it to when
    /// the test tells it, so that how fast it goes is the test's to say
    /// and no clock's but the one the test sets.
    struct Told {
        say: Option<std::sync::mpsc::Sender<usize>>,
        said: std::sync::mpsc::Receiver<()>,
        disk: Disk,
        waited: tokio::task::JoinHandle<std::io::Result<&'static str>>,
    }

    impl Told {
        /// Begin a step on `disk`, waited for from now, a little before
        /// the first of what it is told to do: so that nothing it tells
        /// of falls on the moment it is looked at.
        async fn begun(disk: &Disk) -> Self {
            let (say, told) = std::sync::mpsc::channel::<usize>();
            let (tell, said) = std::sync::mpsc::channel();
            let step = move |progress: &Progress| {
                for bytes in told {
                    progress.made(bytes);
                    tell.send(()).unwrap();
                }
                Ok("done")
            };
            let waited = tokio::spawn({
                let disk = disk.clone();
                async move { disk.run_while_it_gets_on(1, step).await }
            });
            tokio::time::sleep(Duration::from_millis(5)).await;
            Self {
                say: Some(say),
                said,
                disk: disk.clone(),
                waited,
            }
        }

        /// Go at `rate` bytes a second for `millis`, telling of what is
        /// done every ten.
        async fn goes(&self, rate: usize, millis: u64) {
            for _ in 0..millis / 10 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                self.say.as_ref().unwrap().send(rate / 100).unwrap();
                self.said.recv().unwrap();
            }
        }

        /// Whether whoever waited for the step has given it up by now.
        fn given_up(&self) -> bool {
            self.waited.is_finished()
        }

        /// End the step, and answer as whoever waited for it was answered.
        async fn ended(mut self) -> std::io::Result<&'static str> {
            drop(self.say.take());
            // Its thread is done before the clock is let go on, so that
            // an answer on its way is not taken for none.
            while self.disk.threads.free.available_permits() < self.disk.threads.most {
                std::thread::yield_now();
            }
            self.waited.await.unwrap()
        }
    }

    /// A disk that waits fifty milliseconds for a step on a piece, and is
    /// failing if it gives less than a thousand bytes a second.
    fn slow_disk() -> Disk {
        Disk::new(Duration::from_millis(50), 4_000, 4)
    }

    #[tokio::test(start_paused = true)]
    async fn a_step_that_gets_on_is_waited_for_and_one_that_stops_is_given_up() {
        let disk = slow_disk();
        // Ten times as long as a piece is waited for, and telling of it
        // as it goes, ten thousand bytes a second.
        let quick = Told::begun(&disk).await;
        quick.goes(10_000, 500).await;
        assert_eq!(quick.ended().await.unwrap(), "done");

        // Exactly as fast as a disk must be is fast enough, for the first
        // fifty milliseconds are owed nothing.
        let barely = Told::begun(&disk).await;
        barely.goes(1_000, 2_000).await;
        assert_eq!(barely.ended().await.unwrap(), "done");

        // Getting on all the while, at a hundred bytes a second: slower
        // than a disk may be, however long it is given.
        let crawling = Told::begun(&disk).await;
        crawling.goes(100, 140).await;
        assert!(crawling.given_up(), "by the second look at it");
        let too_slow = crawling.ended().await.unwrap_err();
        assert!(
            too_slow.to_string().contains("slower than 1000 bytes"),
            "{too_slow}"
        );

        // After telling of some, nothing more.
        let stopped = Told::begun(&disk).await;
        stopped.goes(10_000, 10).await;
        tokio::time::sleep(Duration::from_millis(90)).await;
        assert!(stopped.given_up(), "by the second look at it");
        let given_up = stopped.ended().await.unwrap_err();
        assert_eq!(given_up.kind(), std::io::ErrorKind::TimedOut);
        assert!(given_up.to_string().contains("no further"), "{given_up}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_large_file_may_be_as_much_slower_as_there_are_others_read_beside_it() {
        // Read at six hundred bytes a second, of the thousand the disk
        // must give.
        let disk = slow_disk();
        let alone = disk.loading().await.unwrap();
        let reading = Told::begun(&disk).await;
        reading.goes(600, 1_000).await;
        let too_slow = reading.ended().await.unwrap_err();
        assert!(too_slow.to_string().contains("slower than"), "{too_slow}");

        // With another read beside it, each has half the disk to itself.
        let _beside = disk.loading().await.unwrap();
        let reading = Told::begun(&disk).await;
        reading.goes(600, 1_000).await;
        assert_eq!(reading.ended().await.unwrap(), "done");
        // But no more than half: four hundred is too slow for either.
        let reading = Told::begun(&disk).await;
        reading.goes(400, 1_000).await;
        let too_slow = reading.ended().await.unwrap_err();
        assert!(
            too_slow.to_string().contains("slower than 500 bytes"),
            "{too_slow}"
        );
        drop(alone);
    }

    #[tokio::test(start_paused = true)]
    async fn a_file_within_its_share_is_not_given_up_when_the_one_read_beside_it_ends() {
        let disk = slow_disk();
        let beside = disk.loading().await.unwrap();
        let _own = disk.loading().await.unwrap();
        // Six hundred bytes a second while it shares the disk, which is
        // within its share; twelve hundred once it has the disk alone,
        // which is within that. What it did while it shared is not held
        // to what a file read alone must do.
        let reading = Told::begun(&disk).await;
        reading.goes(600, 1_000).await;
        drop(beside);
        reading.goes(1_200, 3_000).await;
        assert!(!reading.given_up());
        assert_eq!(reading.ended().await.unwrap(), "done");

        // Alone from then on, it is held to the whole of it: going on as
        // it did beside the other is too slow now.
        let reading = Told::begun(&disk).await;
        reading.goes(600, 1_000).await;
        let too_slow = reading.ended().await.unwrap_err();
        assert!(too_slow.to_string().contains("slower than"), "{too_slow}");
    }

    #[test]
    fn turns_had_and_how_often_that_changed_are_counted_together() {
        let loads = Loads::default();
        assert_eq!(loads.now(), (0, 0));
        loads.given();
        loads.given();
        assert_eq!(loads.now(), (2, 2));
        loads.given_back();
        assert_eq!(loads.now(), (1, 3));
        loads.given_back();
        loads.given();
        assert_eq!(loads.now(), (1, 5), "as many had, and not as it was");
        // As many turns as a disk may ever have are counted whole, and
        // none of them as a change.
        let full = Loads::default();
        for given in 1..=MOST_LOADING_AT_ONCE as u64 {
            full.given();
            if given == 40_000 {
                assert_eq!(full.now(), (40_000, 40_000));
            }
        }
        assert_eq!(full.now(), (65_535, 65_535));
        full.given_back();
        assert_eq!(full.now(), (65_534, 65_536));
        let disk = Disk::new(Duration::from_secs(10), u64::MAX, 8);
        let most = disk.clone().loading_at_once(MOST_LOADING_AT_ONCE).unwrap();
        assert_eq!(most.loads_at_once(), 65_535);
        let refused = disk.clone().loading_at_once(MOST_LOADING_AT_ONCE + 1);
        assert_eq!(
            refused.unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(disk.loading_at_once(0).unwrap().loads_at_once(), 1);

        // However many times it changes, the count of turns is its own.
        for _ in 0..200_000 {
            loads.given();
            loads.given_back();
        }
        assert_eq!(loads.now(), (1, 400_005));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fewer_large_files_read_is_never_seen_without_the_change_being_seen() {
        let disk = Disk::new(Duration::from_secs(10), u64::MAX, 8);
        assert_eq!(disk.loads_at_once(), 2);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // One looks, over and over, while others take turns and give
        // them back as fast as they can.
        let looking = std::thread::spawn({
            let (loads, done) = (disk.loads.clone(), done.clone());
            move || {
                let (mut looks, mut falls) = (0u64, 0u64);
                let mut last = loads.now();
                while !done.load(Ordering::Relaxed) {
                    let now = loads.now();
                    assert!(now.0 <= 2, "{now:?}: more than the disk has");
                    assert!(now.1 >= last.1, "{last:?} then {now:?}");
                    match now.1 == last.1 {
                        true => assert_eq!(now.0, last.0, "changed, and not seen to"),
                        false => falls += u64::from(now.0 < last.0),
                    }
                    (last, looks) = (now, looks + 1);
                }
                (looks, falls)
            }
        });
        let taking = |_| {
            let disk = disk.clone();
            tokio::spawn(async move {
                for _ in 0..20_000 {
                    let turn = disk.loading().await.unwrap();
                    tokio::task::yield_now().await;
                    drop(turn);
                }
            })
        };
        for taken in (0..6).map(taking).collect::<Vec<_>>() {
            taken.await.unwrap();
        }
        done.store(true, Ordering::Relaxed);
        let (looks, falls) = looking.join().unwrap();
        assert!(falls > 0, "no fall was looked at in {looks} looks");
        assert_eq!(disk.loads.now(), (0, 240_000));
    }

    #[test]
    fn what_a_step_owes_is_its_share_at_each_time_and_not_its_share_now() {
        const MS: fn(u64) -> Duration = Duration::from_millis;
        // A thousand bytes a second at the least, on a disk that reads
        // two large files at once.
        let owe = |pace: &mut Pace, at, shifts, beside| pace.owe(MS(at), shifts, beside, 2, 1_000);
        // Beside another for a second, it owes five hundred bytes.
        // The first look at a step comes as its time to begin ends, with
        // nothing owed yet, and whatever turns were given before it.
        let mut pace = Pace::default();
        owe(&mut pace, 0, 7, 2);
        assert!(!pace.behind(0));
        for at in (100..=1_000).step_by(100) {
            owe(&mut pace, at, 7, 2);
        }
        assert!(pace.behind(499) && !pace.behind(500));
        // The other ends. For the time it was last looked at in, it owes
        // as if the other was there throughout, which is not known; from
        // then on, the whole. It never owes the whole for the second it
        // shared.
        owe(&mut pace, 1_100, 8, 1);
        assert!(pace.behind(549) && !pace.behind(550));
        owe(&mut pace, 1_200, 8, 1);
        assert!(pace.behind(649) && !pace.behind(650));
        // A time it has been brought up to is not owed for twice.
        owe(&mut pace, 1_200, 8, 1);
        owe(&mut pace, 900, 8, 1);
        assert!(pace.behind(649) && !pace.behind(650));

        // One that crawls is behind however often the others beside it
        // come and go: a hundred bytes a second is less than any share.
        let mut pace = Pace::default();
        for look in 1..=10 {
            owe(&mut pace, look * 100, look, 1 + look % 2);
            assert!(pace.behind(look * 10), "{pace:?}");
        }
        // Alone and at the least it may go, it is never behind.
        let mut pace = Pace::default();
        for look in 1..=10 {
            owe(&mut pace, look * 100, 0, 1);
            assert!(!pace.behind(look * 100) && pace.behind(look * 100 - 1));
        }
    }

    #[test]
    fn a_turn_is_waited_for_longer_than_a_part_may_take_whatever_the_store_is_given() {
        let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 4);
        let longest = |disk: &Disk, transfers: &Transfers| {
            // The file made; every request for the part; and its
            // writing, having waited for a thread, or one request more.
            let written = disk.answers_within(transfers.part_bytes);
            disk.answers_within(0)
                + transfers.request_within * RANGE_TRIES as u32
                + written.max(transfers.request_within)
        };
        let default = Transfers::default();
        assert_eq!(default.part_within(), Duration::from_secs(374));
        assert!(disk.loading_still_for() > longest(&disk, &default));
        for (secs, part_bytes) in [(1, 1), (187, 16 << 20), (900, 64 << 20), (36_000, 1 << 30)] {
            let transfers = Transfers {
                part_bytes,
                request_within: Duration::from_secs(secs),
            };
            let disk = disk.clone().fetching(&transfers);
            let still = disk.loading_still_for();
            let longest = longest(&disk, &transfers);
            assert_eq!(still, longest + LOADING_STILL_BEYOND, "{secs}");
            // Reading more files at once changes nothing of it.
            assert_eq!(disk.loading_at_once(5).unwrap().loading_still_for(), still);
        }
    }

    #[tokio::test]
    async fn no_more_whole_files_are_read_at_once_than_a_disk_takes() {
        let disk = Disk::new(Duration::from_secs(10), u64::MAX, 8);
        let reading = Arc::new(AtomicU64::new(0));
        let most = Arc::new(AtomicU64::new(0));
        let read = |_| {
            let (disk, reading, most) = (disk.clone(), reading.clone(), most.clone());
            async move {
                let _turn = disk.loading().await;
                let now = reading.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(5)).await;
                reading.fetch_sub(1, Ordering::SeqCst);
            }
        };
        futures::future::join_all((0..10).map(read)).await;
        assert_eq!(most.load(Ordering::SeqCst), LOADING_AT_ONCE as u64);

        // Or as many as it is set to take.
        let disk = disk.loading_at_once(3).unwrap();
        let turns = futures::future::join_all((0..3).map(|_| disk.loading())).await;
        assert!(turns.iter().all(|turn| turn.is_ok()));
        let fourth = tokio::time::timeout(Duration::from_millis(50), disk.loading());
        assert!(fourth.await.is_err(), "it waits for one of the three");
    }

    #[tokio::test(start_paused = true)]
    async fn a_turn_at_a_large_file_is_waited_for_while_anything_moves_and_no_longer() {
        let disk = Disk::new(Duration::from_secs(10), u64::MAX, 8)
            .loading_at_once(1)
            .unwrap();
        let still = disk.loading_still_for();
        // Twelve large files wait their turn, as those of one tenant or of
        // several opened together do. Each takes four minutes, getting on
        // all the while: the last has waited three quarters of an hour.
        let started = tokio::time::Instant::now();
        let load = |_| {
            let disk = disk.clone();
            async move {
                let _turn = disk.loading().await?;
                for _ in 0..8 {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    disk.progress().made(1);
                }
                Ok::<_, std::io::Error>(())
            }
        };
        let loaded = futures::future::join_all((0..12).map(load)).await;
        assert!(loaded.iter().all(|loaded| loaded.is_ok()), "{loaded:?}");
        assert_eq!(started.elapsed(), Duration::from_secs(12 * 4 * 60));

        // One with a turn gets nowhere: whoever waits behind it is told so
        // once nothing has moved for as long as nothing may.
        let _stuck = disk.loading().await.unwrap();
        let started = tokio::time::Instant::now();
        let refused = disk.loading().await.unwrap_err();
        assert_eq!(refused.kind(), std::io::ErrorKind::ResourceBusy);
        assert_eq!(started.elapsed(), still);
    }

    #[tokio::test(start_paused = true)]
    async fn one_large_file_that_takes_long_and_gets_on_keeps_those_behind_it_waiting() {
        let disk = Disk::new(Duration::from_secs(10), u64::MAX, 8)
            .loading_at_once(1)
            .unwrap();
        let still = disk.loading_still_for();
        // One file has the only turn for four times as long as nothing
        // may move, telling of each piece as it goes. No turn is given
        // meanwhile: what it tells of is all that moves.
        let turn = disk.loading().await.unwrap();
        let behind: Vec<_> = (0..3)
            .map(|_| {
                let disk = disk.clone();
                tokio::spawn(async move { disk.loading().await.map(drop) })
            })
            .collect();
        let progress = disk.progress();
        let started = tokio::time::Instant::now();
        while started.elapsed() < still * 4 {
            tokio::time::sleep(still / 3).await;
            progress.made(1);
        }
        assert!(
            behind.iter().all(|waiting| !waiting.is_finished()),
            "none has given up, and none has the turn"
        );
        drop(turn);
        for waited in behind {
            waited.await.unwrap().expect("its turn, in the end");
        }
    }

    #[tokio::test]
    async fn a_step_of_a_fetch_given_up_takes_away_what_it_wrote_after() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.partial-late");
        let unfinished = Partial {
            disk: Disk::shared().clone(),
            path: Some(path.clone()),
            given_up: Arc::default(),
        };
        // The step is on its way to a thread when the fetch is given up,
        // and writes only after what was there has been removed.
        let late = unfinished.step(|path| std::fs::File::create(path).map(|_| ()));
        drop(unfinished);
        tokio::time::sleep(Duration::from_millis(50)).await;
        late().unwrap();
        assert!(!path.exists(), "not even an empty file is left");
    }

    #[tokio::test]
    async fn a_file_is_fetched_into_the_cache_in_its_turn_among_the_large_files_of_its_disk() {
        let dir = tempfile::tempdir().unwrap();
        let disk = Disk::new(Duration::from_secs(10), u64::MAX, 4)
            .loading_at_once(1)
            .unwrap();
        let store = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        store
            .put("a/large", Bytes::from_static(b"file"))
            .await
            .unwrap();
        let cached = Arc::new(CachedBlobs::new(store, dir.path()).on(disk.clone()));

        let turn = disk.loading().await.unwrap();
        let fetching = tokio::spawn({
            let cached = cached.clone();
            async move { cached.fetch_local("a/large").await }
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!fetching.is_finished(), "it waits its turn");
        assert!(
            !dir.path().join("a").exists(),
            "and writes nothing meanwhile"
        );
        drop(turn);
        let fetched = fetching
            .await
            .unwrap()
            .unwrap()
            .expect("a store that keeps copies");
        assert_eq!(std::fs::read(fetched.path).unwrap(), b"file");
    }

    #[tokio::test]
    async fn a_copy_removed_or_replaced_is_no_longer_known_as_summed() {
        use crate::column::{seen, summed_before, Column};
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let cached = CachedBlobs::new(store, dir.path());
        let vectors = Column::taken_from(&mut []).encode();
        // A copy of the file under `key`, summed as vectors are when a
        // partition is opened: how it lay then.
        let summed = |key: &'static str| {
            let (cached, vectors) = (&cached, vectors.clone());
            async move {
                cached.put(key, vectors).await.unwrap();
                let path = cached.local(key).await.unwrap().unwrap().path;
                let lay = seen(&path).unwrap();
                let column = Column::map(key, &path).unwrap();
                column.summed_as(key, lay.clone(), |_| {}).unwrap();
                assert!(summed_before(&lay));
                lay
            }
        };

        let deleted = summed("t/0/deleted.vectors").await;
        cached.delete("t/0/deleted.vectors").await.unwrap();
        assert!(!summed_before(&deleted));

        let (one, other) = (summed("t/1/a.vectors").await, summed("t/1/b.vectors").await);
        let beside = summed("t/10/a.vectors").await;
        cached.forget_local("t/1/").await.unwrap();
        assert!(!summed_before(&one) && !summed_before(&other));
        assert!(summed_before(&beside), "another directory's is kept");

        let replaced = summed("t/2/replaced.vectors").await;
        cached
            .put("t/2/replaced.vectors", vectors.clone())
            .await
            .unwrap();
        assert!(!summed_before(&replaced));
        let fetched = summed("t/3/fetched.vectors").await;
        cached.fetch_local("t/3/fetched.vectors").await.unwrap();
        assert!(!summed_before(&fetched));
    }
}
