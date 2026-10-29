//! Immutable files in object storage. A key is written once and never
//! changes, which is what makes caching them on local disk safe.
//!
//! Nothing brought in from outside a tenant's store, from a backup or from
//! shipped history, is ever written or named there under the key it had:
//! it is copied under keys of its own, which nobody could have known
//! before, and the head that takes it up names only those
//! ([`Writer::restore_from`]). A delete asked for of a key, however late
//! it lands, so never takes a file a head brought in that way names. That
//! is the whole of the rule, and the code enforces it (DW-680).
//!
//! A head made inside the store does name older keys: every checkpoint
//! carries forward the files of the one before, and a restore to a point
//! in the window names the files of that point. The first is safe by the
//! cleaner's own rule ([`Writer::clean`]). The second is safe only by a
//! margin of time, the grace, which nothing enforces: the clocks the
//! cleaner and the coordinator go by must not disagree by more than it.
//! Capping what a clean deletes by the commits the coordinator still has
//! is DW-681.
//!
//! [`Writer::restore_from`]: crate::Writer::restore_from
//! [`Writer::clean`]: crate::Writer::clean

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::{
    GetOptions, GetRange, GetResultPayload, MultipartUpload, ObjectStore, PutMode, PutOptions,
    PutPayload,
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
    /// Put a file under `key` only if nothing is there, in one request,
    /// and say which: [`Created::Written`] if this put it there,
    /// [`Created::AlreadyThere`] if the store refused it. A refusal says a
    /// file is, or may be, under `key`, and nothing of whose it is: it may
    /// be this call's own, landed before a failed answer that the store's
    /// client tried again; or no file may be there at all, another create
    /// of it having been under way and failed since. So whoever is told
    /// `AlreadyThere` reads the key to know what it holds, and one that
    /// finds nothing there creates it again.
    ///
    /// Any error leaves it unknown whether the file went up: a create the
    /// store had but did not answer, or one given up waiting for, may have
    /// landed, and asking again is answered `AlreadyThere` if it did. The
    /// one exception is [`StorageError::NotCreated`], a create refused
    /// before anything was sent, which asking again is refused alike: by a
    /// store that cannot create only once, or for a file larger than
    /// [`Self::creates_at_most`].
    ///
    /// By default refused so, with nothing put: a store that cannot refuse
    /// a second create of a key must never be taken for one that can, and
    /// a put in its place would replace what is there.
    async fn put_new(&self, key: &str, _bytes: Bytes) -> Result<Created> {
        Err(StorageError::NotCreated {
            key: key.to_string(),
            reason: "this store does not create a file only if none is there".into(),
        })
    }
    /// The most bytes [`Self::put_new`] creates a file of; `None` from a
    /// store that creates none.
    fn creates_at_most(&self) -> Option<usize> {
        None
    }
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
    /// Up to `limit` files whose keys start with `prefix` and come after
    /// `after`, byte by byte, never `after` itself, in one request: so that
    /// whether anything is past a key is known without listing what is
    /// before it. A store that lists in key order, as S3 and one in memory
    /// do, gives the first of them; one on local disk gives files after
    /// `after` in no order, walking the directory to find them. S3 asks
    /// for them from `after` in one request; a store that cannot, as S3
    /// Express, one in memory and one on local disk, lists under `prefix`
    /// and leaves out what is not after `after`: what it gives is the
    /// same, in more than one request, or longer, for a large prefix.
    ///
    /// By default refused, with [`StorageError::NotListed`]: a store that
    /// listed everything to find what is past a key would take as long as
    /// the listing, unseen.
    async fn list_from(&self, prefix: &str, _after: &str, _limit: usize) -> Result<Vec<Listed>> {
        Err(StorageError::NotListed {
            prefix: prefix.to_string(),
            reason: "this store does not list from a key in one request".into(),
        })
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
    /// Give the file under `to`, which the caller has seen to hold just
    /// what the file under `from` holds, the copy on local disk `from` has,
    /// without copying it. Never fails: a store that cannot says so, and
    /// the caller reads `to` as any file.
    ///
    /// By default nothing is linked: a store with a copy of `from` on
    /// local disk, which may have been read in place, has not given it to
    /// `to`; one without, whose files are read whole into memory, has
    /// nothing to give.
    ///
    /// The default goes by [`Self::local`]. A store over another that
    /// keeps copies read in place must forward `local`, or forward this
    /// too: one that forwards neither answers [`Linked::Unmapped`] for a
    /// file its inner store reads in place, and a caller may then take
    /// what was read of `from` for `to`'s, whose copy on local disk it is
    /// not. Such a store answers `local` with none, so nothing read
    /// through it is read in place: what is held of `from` is in memory,
    /// and is the same file's. Forwarding `local` alone is safe, and
    /// links nothing.
    async fn link_local(&self, from: &str, _to: &str) -> Result<Linked> {
        Ok(match self.local(from).await {
            Ok(None) => Linked::Unmapped,
            Ok(Some(_)) | Err(_) => Linked::Unlinked,
        })
    }
}

/// What came of [`Blobs::link_local`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Linked {
    /// The copy on local disk is the new key's too, lying here.
    Linked(PathBuf),
    /// The store keeps no copy it reads in place: nothing to link, and
    /// nothing of a file read from it is held by a name.
    Unmapped,
    /// Not linked: whatever was read of `from` in place is not to be
    /// taken for `to`'s.
    Unlinked,
}

/// What came of [`Blobs::put_new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Created {
    /// The file is there, put by this request.
    Written,
    /// The store refused: a file is under the key, or another create of it
    /// was under way. Not a failure, and not word of what the file holds.
    AlreadyThere,
}

/// Whether `blobs` refuses a second create of a key, by creating `key`: a
/// store that refuses it, having the file from an earlier check, passes;
/// one that puts it is asked to create it again, and passes only if it
/// refuses that. A store that puts it again ignores the condition and
/// fails; so does one whose create fails, which tells nothing either way.
/// Whatever is under `key` is left there.
///
/// This is a tripwire, not a qualification of the store: one store at a
/// time can only tell a store that ignores the condition. One that checks
/// it and writes in two steps, as Adobe's S3Mock does, lets two creates
/// racing for a key both through, and passes this.
pub async fn creates_only_once(blobs: &dyn Blobs, key: &str) -> Result<()> {
    let probe = Bytes::from_static(b"vellev: a file is created here only once\n");
    let ignored = || StorageError::Blob {
        key: key.to_string(),
        source: "the store put a file again where one was already, \
                 so it does not create a file only if none is there"
            .into(),
    };
    match blobs.put_new(key, probe.clone()).await? {
        Created::AlreadyThere => Ok(()),
        Created::Written => match blobs.put_new(key, probe).await? {
            Created::AlreadyThere => Ok(()),
            Created::Written => Err(ignored()),
        },
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

/// Whether `key` is named as a checkpoint is, in every layout this
/// release writes (see [`Layout::manifest`], and a link's from another
/// store). Only a hint: what a head is, its frame says; a walk back that
/// trusts this to fetch ahead checks every head it reaches all the same.
pub(crate) fn named_as_checkpoint(key: &str) -> bool {
    key.ends_with(".manifest")
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

    /// The key a writer names a checkpoint file by: `kind` at `position`, of
    /// `author`, apart from any other by `nonce`, which it draws, and then
    /// `marker`: none for a flush, `-merged` for a merge, `-restored` for a
    /// restore from another store's files (DW-680). Every checkpoint file a
    /// writer has written since DW-658 is named so; [`Self::named_by`] reads
    /// these names back, and those of files written before, with no nonce.
    pub(crate) fn nonced(
        &self,
        kind: &str,
        position: LogPosition,
        author: &Author,
        nonce: u64,
        marker: &str,
    ) -> String {
        self.named(kind, position, author, &format!("-{nonce:016x}{marker}"))
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

    /// What a key of this partition says of its file, if it is laid out
    /// exactly as [`Self::named`] and [`Self::nonced`] lay one out: the
    /// kind, the position, the generation, the writer; on a segment, index
    /// or vectors file only, the nonce a writer drew since DW-658, sixteen
    /// lowercase hex digits, and a marker, `-merged` or `-restored`; and the
    /// kind again as its extension. `-restored` comes only with a nonce,
    /// as no writer named a restored file before DW-658; `-merged` comes
    /// with or without one. A log or a manifest is named with neither. Nothing else passes, so every key read is plain: each part
    /// of it is one of those.
    pub(crate) fn named_by<'k>(&self, key: &'k str) -> Option<Named<'k>> {
        let (kind, name) = key.strip_prefix(&self.prefix())?.split_once('/')?;
        let mergeable = matches!(kind, "segment" | "index" | "vectors");
        if !mergeable && !matches!(kind, "log" | "manifest") {
            return None;
        }
        let stem = name.strip_suffix(kind)?.strip_suffix('.')?;
        let (stem, merged, restored) =
            match (stem.strip_suffix("-merged"), stem.strip_suffix("-restored")) {
                (Some(stem), _) if mergeable => (stem, true, false),
                (_, Some(stem)) if mergeable => (stem, false, true),
                (None, None) => (stem, false, false),
                _ => return None,
            };
        let mut parts = stem.split('-');
        let number = |part: &str| {
            let digits = part.len() == 20 && part.bytes().all(|byte| byte.is_ascii_digit());
            digits.then(|| part.parse::<u64>().ok()).flatten()
        };
        let hex = |part: &str, len: usize| {
            let lower = |byte: u8| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte);
            part.len() == len && part.bytes().all(lower)
        };
        let position = LogPosition(number(parts.next()?)?);
        let generation = Generation(number(parts.next()?)?);
        if !hex(parts.next()?, 32) {
            return None;
        }
        let nonce = match parts.next() {
            None => None,
            Some(drawn) if mergeable && hex(drawn, 16) => {
                Some(u64::from_str_radix(drawn, 16).ok()?)
            }
            Some(_) => return None,
        };
        // `restore_from` (DW-680) came after the nonce (DW-658): a file
        // marked restored without one was never written.
        if parts.next().is_some() || (restored && nonce.is_none()) {
            return None;
        }
        Some(Named {
            kind,
            position,
            generation,
            merged,
            restored,
            nonce,
        })
    }
}

/// What a key laid out by [`Layout`] says of its file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Named<'k> {
    pub kind: &'k str,
    pub position: LogPosition,
    pub generation: Generation,
    pub merged: bool,
    pub restored: bool,
    /// What the writer drew to name it apart, if it is a checkpoint file
    /// named since DW-658.
    pub nonce: Option<u64>,
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
pub struct Progress(Arc<AtomicU64>, Arc<AtomicU64>, Arc<AtomicBool>);

impl Progress {
    /// Tell of `bytes` more done.
    pub fn made(&self, bytes: usize) {
        self.0.fetch_add(bytes as u64, Ordering::Relaxed);
        self.1.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether nobody waits for the step any more: its caller stopped
    /// waiting, as a stopped merge does. A step asks between pieces, and
    /// gives up too, so that the disk's turn and thread go to work someone
    /// waits for. A step whose wait ended at its bound is not told so: it
    /// runs on, as what it does may be kept for the next attempt.
    pub fn given_up(&self) -> bool {
        self.2.load(Ordering::Relaxed)
    }

    fn so_far(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Tells a step nobody waits for it any more, if it is dropped while its
/// caller still waits: not once the wait has ended, answered or given up
/// at its bound.
struct GivenUp {
    progress: Progress,
    dropped: bool,
}

impl Drop for GivenUp {
    fn drop(&mut self) {
        if self.dropped {
            self.progress.2.store(true, Ordering::Relaxed);
        }
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
        Progress(Arc::default(), self.moved.clone(), Arc::default())
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
        // Dropped unfinished, by a caller that stops waiting, as a stopped
        // merge does, the step is told so, and gives up at its next piece.
        // Not when the wait ends at its bound: a check of a file on a slow
        // disk runs on to its end, and is remembered, so that the next
        // load need not begin it again from nothing (see `column::Summed`).
        let mut waiting = GivenUp {
            progress: progress.clone(),
            dropped: true,
        };
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
            let ended = match stepped {
                Ok(stepped) => stepped.map_err(std::io::Error::other).and_then(|done| done),
                Err(_) if progress.so_far() == seen => Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("local disk got no further within {within:?}"),
                )),
                Err(_) if pace.behind(progress.so_far()) => {
                    let slowest = least / beside;
                    Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("local disk is slower than {slowest} bytes a second"),
                    ))
                }
                Err(_) => {
                    seen = progress.so_far();
                    let mut due = threads.due.lock().expect("disk steps lock");
                    if let Some(due) = due.get_mut(&number) {
                        *due = tokio::time::Instant::now() + within;
                    }
                    continue;
                }
            };
            waiting.dropped = false;
            return ended;
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
    aborts: Aborts,
}

impl ObjectBlobs {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            root: None,
            transfers: Transfers::default(),
            aborts: Aborts::process(),
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

    /// Count the aborts of uploads dropped unfinished in `aborts`,
    /// instead of in the process's own.
    pub fn aborting(self, aborts: Aborts) -> Self {
        Self { aborts, ..self }
    }

    /// Start an upload in parts under `path`, on a task of its own: one
    /// whose caller stops waiting before the store answers is aborted once
    /// the store does, never left begun with no parts and nothing to abort
    /// it.
    async fn started(&self, key: &str, path: &ObjectPath) -> Result<Box<dyn MultipartUpload>> {
        let within = self.transfers.request_within;
        let (store, path, named) = (self.store.clone(), path.clone(), key.to_string());
        let mut starting = Starting {
            start: Some(tokio::spawn(async move {
                bounded(&named, within, store.put_multipart(&path)).await
            })),
            within,
            aborts: self.aborts.clone(),
        };
        let started = starting.start.as_mut().expect("just begun").await;
        starting.start = None;
        match started {
            Ok(started) => started,
            Err(failed) => match failed.try_into_panic() {
                Ok(panic) => std::panic::resume_unwind(panic),
                // Cancelled, as the runtime shuts down.
                Err(failed) => Err(StorageError::Background(failed.to_string())),
            },
        }
    }

    /// Upload `bytes` under `key` in parts, each its own request, and abort
    /// the upload if any of it fails or it is dropped unfinished.
    async fn put_in_parts(&self, key: &str, path: &ObjectPath, bytes: Bytes) -> Result<()> {
        let within = self.transfers.request_within;
        let mut upload = Unfinished {
            upload: Some(self.started(key, path).await?),
            within,
            aborts: self.aborts.clone(),
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

/// The aborts of uploads dropped unfinished that are on their way, and the
/// deletes of files a build given up uploaded: each is
/// sent on a task of its own, which nothing waits for, and a process that
/// exits at once may never send it, leaving the upload's parts in the
/// store, where no listing shows them. A process that stops waits for
/// them, for as long as it gives them (see [`Aborts::finished`]).
///
/// Each is on its way until it is answered or given up, at the bound of a
/// request to the store, so waiting for longer than that bound waits for
/// all there are.
#[derive(Debug, Clone, Default)]
pub struct Aborts(Arc<OnTheirWay>);

#[derive(Debug, Default)]
struct OnTheirWay {
    pending: std::sync::atomic::AtomicUsize,
    done: tokio::sync::Notify,
}

impl Aborts {
    /// The aborts every store of the process counts in, unless it is given
    /// others.
    pub fn process() -> Aborts {
        static PROCESS: std::sync::OnceLock<Aborts> = std::sync::OnceLock::new();
        PROCESS.get_or_init(Aborts::default).clone()
    }

    /// How many are on their way.
    pub fn pending(&self) -> usize {
        self.0.pending.load(Ordering::SeqCst)
    }

    /// Wait until none is on its way, or `within` has passed: how many
    /// still are.
    pub async fn finished(&self, within: Duration) -> usize {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            // Asked for before the count is read, so that the last one to
            // end between the two is not missed.
            let done = self.0.done.notified();
            tokio::pin!(done);
            done.as_mut().enable();
            let pending = self.pending();
            if pending == 0 {
                return 0;
            }
            if tokio::time::timeout_at(deadline, done).await.is_err() {
                return self.pending();
            }
        }
    }

    /// One more on its way, until what this gives is dropped.
    pub(crate) fn begun(&self) -> Aborting {
        self.0.pending.fetch_add(1, Ordering::SeqCst);
        Aborting(self.clone())
    }
}

/// An abort on its way, counted in its [`Aborts`] until this is dropped:
/// once it is answered or given up, or the task that sends it goes.
pub(crate) struct Aborting(Aborts);

impl Drop for Aborting {
    fn drop(&mut self) {
        let OnTheirWay { pending, done } = &*self.0 .0;
        pending.fetch_sub(1, Ordering::SeqCst);
        done.notify_waiters();
    }
}

/// An upload in parts not yet complete, aborted if it is dropped so.
#[derive(Debug)]
struct Unfinished {
    upload: Option<Box<dyn MultipartUpload>>,
    within: Duration,
    aborts: Aborts,
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

/// An upload in parts being started, aborted once the store answers if
/// whoever started it stopped waiting first.
struct Starting {
    start: Option<tokio::task::JoinHandle<Result<Box<dyn MultipartUpload>>>>,
    within: Duration,
    aborts: Aborts,
}

impl Drop for Starting {
    fn drop(&mut self) {
        let (Some(start), Ok(runtime)) = (self.start.take(), tokio::runtime::Handle::try_current())
        else {
            return;
        };
        let (within, aborting) = (self.within, self.aborts.begun());
        runtime.spawn(async move {
            let _aborting = aborting;
            if let Ok(Ok(mut upload)) = start.await {
                if let Err(error) = bounded("an upload", within, upload.abort()).await {
                    tracing::warn!(%error, "an upload started for nobody could not be aborted");
                }
            }
        });
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
        let aborting = self.aborts.begun();
        runtime.spawn(async move {
            let _aborting = aborting;
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

    /// One request, which `object_store` makes create only: on S3 with
    /// `If-None-Match: *`, refused with 412 or 304, or with 409 while
    /// another create of the key is under way. An upload in parts cannot
    /// be made so, so a file larger than [`Transfers::part_bytes`] is
    /// refused before anything goes up. The S3 client tries a create again
    /// after an answer of 5xx, 429 or 408, and after a request it could not
    /// send, as it does a put; not after one that timed out or was cut off
    /// on the way, which it does not know to be safe to repeat. A create
    /// tried again after its own landed is refused, and so answered
    /// `AlreadyThere`.
    async fn put_new(&self, key: &str, bytes: Bytes) -> Result<Created> {
        if bytes.len() > self.transfers.part_bytes {
            return Err(StorageError::NotCreated {
                key: key.to_string(),
                reason: format!(
                    "{} bytes are more than the {} one request takes, \
                     and a file is created only once in one request",
                    bytes.len(),
                    self.transfers.part_bytes
                ),
            });
        }
        let options = PutOptions {
            mode: PutMode::Create,
            ..PutOptions::default()
        };
        let path = ObjectPath::from(key);
        let put = self
            .store
            .put_opts(&path, PutPayload::from_bytes(bytes), options);
        let created = async {
            match put.await {
                Ok(_) => Ok(Created::Written),
                Err(object_store::Error::AlreadyExists { .. }) => Ok(Created::AlreadyThere),
                Err(error) => Err(error),
            }
        };
        bounded(key, self.transfers.request_within, created).await
    }

    fn creates_at_most(&self) -> Option<usize> {
        Some(self.transfers.part_bytes)
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
            aborts: self.aborts.clone(),
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
                        upload.upload = Some(self.started(key, &path).await?);
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

    /// One request of the store's listing, from `after`: S3's
    /// `start-after`, given up after `request_within` as any request is.
    async fn list_from(&self, prefix: &str, after: &str, limit: usize) -> Result<Vec<Listed>> {
        use futures::TryStreamExt;
        let within = self.transfers.request_within;
        let under = ObjectPath::from(prefix);
        let offset = ObjectPath::from(after);
        let mut listing = self.store.list_with_offset(Some(&under), &offset);
        let mut listed = Vec::new();
        while listed.len() < limit {
            let Some(meta) = bounded(prefix, within, listing.try_next()).await? else {
                break;
            };
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

/// A link into the cache (see [`CachedBlobs::link_local`]), given up
/// once this is dropped unanswered: by a failure, a time-out or a caller
/// that stops waiting.
struct Linking {
    given_up: Arc<std::sync::atomic::AtomicBool>,
    answered: bool,
}

impl Drop for Linking {
    fn drop(&mut self) {
        if !self.answered {
            self.given_up.store(true, Ordering::SeqCst);
        }
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

    // No `put_new`, which is refused by default: nothing creates through a
    // cache, and a copy kept of a create could be bytes the store does not
    // hold, beside a put or a delete of the same key.

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

    /// As the store lists from a key. The cache creates no file only if
    /// none is there (`put_new` is refused, by design: #66's review, N3),
    /// so a backup store, which a shipper creates links in, is never
    /// wrapped in a cache: this is for a listing alone.
    async fn list_from(&self, prefix: &str, after: &str, limit: usize) -> Result<Vec<Listed>> {
        self.inner.list_from(prefix, after, limit).await
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

    /// From the store, as it gives the file, and never from the copy on
    /// local disk, which [`Self::get`] reads first: a store over this one
    /// that reads a file whole as its parts put together, as a node's
    /// bounded stores do, passes the copy by. None reads through a cache
    /// today: a node's backup store keeps no copies, and the tenants'
    /// store is read through a bounded one only to list and delete.
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

    /// A hard link of the copy of `from` under the name of `to`'s, made
    /// under a name drawn at random and renamed into place, so that it is
    /// there whole or not at all, on one thread of the disk's. Both names
    /// are one file from then on: a delete of either removes that name
    /// alone. The new name is taken as summed only if the old one was, as
    /// it lay just before, and the new is the very file it was.
    ///
    /// Not linked if either key is not a plain one, or the copy of `from`
    /// is not there, or the disk will not make the link, as across disks
    /// or in time: the caller then reads `to` as any file. Nothing is
    /// linked off Unix, where no file is read in place (see
    /// `Partition::column`), and nothing is to be.
    ///
    /// A link the caller stops waiting for, by a failure or by going away,
    /// may yet be made by the thread still at it, after a delete of `to`
    /// has removed whatever copy there was: the copy would then lie under
    /// a key the store no longer has. So the thread looks, before it
    /// renames and once it has, whether the link was given up, and if so
    /// removes what it made. What remains is a link given up after that
    /// last look: it is in place before the caller hears, so a delete of
    /// `to` that follows removes it, and a fetch of `to` replaces it with
    /// the same bytes. Nothing but the thread removes the new name, and
    /// only while it is still the very file linked: a copy of `to` another
    /// fetch has put in its place, which may be read in place by now, is
    /// left alone, but for a fetch that renames into place between the
    /// thread's look and its removal, whose copy goes and is fetched again
    /// the next time it is read.
    async fn link_local(&self, from: &str, to: &str) -> Result<Linked> {
        let (Some(old), Some(new)) = (self.path(from), self.path(to)) else {
            return Ok(Linked::Unlinked);
        };
        if cfg!(not(unix)) {
            return Ok(Linked::Unmapped);
        }
        let mut linking = Linking {
            given_up: Arc::default(),
            answered: false,
        };
        let given_up = linking.given_up.clone();
        let linked = move || {
            let given_up = || given_up.load(Ordering::SeqCst);
            let gone = || std::io::Error::other("the link was given up");
            let before = crate::column::seen(&old)?;
            let summed = crate::column::summed_before(&before);
            let parent = new
                .parent()
                .ok_or_else(|| std::io::Error::other("cache path has no parent"))?;
            std::fs::create_dir_all(parent)?;
            let partial = partial(&new);
            std::fs::hard_link(&old, &partial)?;
            if given_up() {
                let _ = std::fs::remove_file(&partial);
                return Err(gone());
            }
            crate::column::forget_summed(&new);
            if let Err(error) = std::fs::rename(&partial, &new) {
                let _ = std::fs::remove_file(&partial);
                return Err(error);
            }
            if given_up() {
                let linked = crate::column::seen(&new);
                if linked.is_ok_and(|linked| linked.same_file(&before)) {
                    crate::column::forget_summed(&new);
                    let _ = std::fs::remove_file(&new);
                }
                return Err(gone());
            }
            let after = crate::column::seen(&new)?;
            if summed && after.same_file(&before) {
                crate::column::remember_summed(after);
            }
            Ok(new)
        };
        match self.disk.run(0, linked).await {
            Ok(new) => {
                linking.answered = true;
                Ok(Linked::Linked(new))
            }
            Err(error) => {
                tracing::warn!(from, to, %error, "a copy on local disk is not linked to a new key");
                Ok(Linked::Unlinked)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step whose wait is given up at its bound, as one too slow for
    /// the disk is, is not told nobody waits: it runs on to its end, so that
    /// a check of a large file on a slow disk is finished and remembered,
    /// and the next load does not begin it again from nothing.
    #[tokio::test(start_paused = true)]
    async fn a_step_given_up_at_its_bound_is_not_told_to_give_up() {
        let disk = slow_disk();
        let (say, told) = std::sync::mpsc::channel::<usize>();
        let (tell, said) = std::sync::mpsc::channel();
        let (seen, heard) = std::sync::mpsc::channel();
        let step = move |progress: &Progress| {
            for bytes in told {
                progress.made(bytes);
                tell.send(()).unwrap();
            }
            seen.send(progress.given_up()).unwrap();
            Ok("done")
        };
        let waited = tokio::spawn({
            let disk = disk.clone();
            async move { disk.run_while_it_gets_on(1, step).await }
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        // A hundred bytes a second: slower than the disk may be.
        for _ in 0..14 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            say.send(1).unwrap();
            said.recv().unwrap();
        }
        assert!(waited.is_finished(), "given up for its pace");
        let too_slow = waited.await.unwrap().unwrap_err();
        assert!(too_slow.to_string().contains("slower than"), "{too_slow}");
        drop(say);
        let told_so = heard.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(!told_so, "a step given up for its pace was told to give up");
    }

    /// A step on local disk whose caller stops waiting is told so, and
    /// can give up, rather than holding the disk's thread and turn to the
    /// end of work nobody wants.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_step_nobody_waits_for_is_told_so() {
        let disk = Disk::new(Duration::from_secs(10), 1024 * 1024, 2);
        let (told, heard) = std::sync::mpsc::channel();
        let step = disk.run_while_it_gets_on(1, move |progress| {
            let begun = std::time::Instant::now();
            while !progress.given_up() && begun.elapsed() < Duration::from_secs(30) {
                progress.made(1);
                std::thread::sleep(Duration::from_millis(5));
            }
            let _ = told.send(progress.given_up());
            Ok(())
        });
        assert!(tokio::time::timeout(Duration::from_millis(100), step)
            .await
            .is_err());
        let heard = tokio::task::spawn_blocking(move || heard.recv_timeout(Duration::from_secs(5)));
        assert_eq!(heard.await.unwrap(), Ok(true), "the step was never told");
    }
    use object_store::memory::InMemory;

    #[test]
    fn every_key_a_partition_lays_out_is_read_back_exactly_and_nothing_else_is() {
        let partition = PartitionKey {
            tenant: vellev_core::TenantId::new("tenant-a").unwrap(),
            partition: vellev_core::Partition(0),
        };
        let layout = Layout(&partition);
        let author = Author {
            generation: Generation(2),
            writer: uuid::Uuid::from_u128(0xabc),
        };
        let at = LogPosition(7);
        let named = |kind, merged| Named {
            kind,
            position: at,
            generation: Generation(2),
            merged,
            restored: false,
            nonce: None,
        };
        let laid_out = [
            (layout.log(at, &author), named("log", false)),
            (layout.manifest(at, &author), named("manifest", false)),
            (layout.segment(at, &author), named("segment", false)),
            (layout.merged(at, &author), named("segment", true)),
            (layout.index(at, &author), named("index", false)),
            (layout.merged_index(at, &author), named("index", true)),
            (layout.vectors(at, &author), named("vectors", false)),
            (layout.merged_vectors(at, &author), named("vectors", true)),
        ];
        for (key, named) in &laid_out {
            assert_eq!(layout.named_by(key), Some(*named), "{key}");
        }

        let log = layout.log(at, &author);
        let writer = author.writer.simple().to_string();
        let refused = [
            "tenant-a/0/log/00000000000000000007-x/y".to_string(),
            format!("{log}x"),
            log.replace(".log", ".manifest"),
            log.replace(".log", "-merged.log"),
            layout
                .manifest(at, &author)
                .replace(".manifest", "-merged.manifest"),
            log.replace("/00000000000000000007-", "/0000000000000000007-"),
            log.replace("-00000000000000000002-", "-0000000000000000002-"),
            log.replace(&writer, &writer.to_uppercase()),
            log.replace(&writer, &writer[1..]),
            log.replace(&writer, &format!("{writer}-x")),
            log.replace(&writer, &format!("{}g", &writer[..31])),
            log.replace("tenant-a/0/", "tenant-a/1/"),
            log.replace("/log/", "/other/").replace(".log", ".other"),
            log.replace("/00000000000000000007-", "/+0000000000000000007-"),
        ];
        for key in &refused {
            assert_eq!(layout.named_by(key), None, "{key}");
        }
    }

    #[test]
    fn every_name_a_writer_gives_a_checkpoint_file_is_read_back_and_nothing_else_is() {
        let partition = PartitionKey {
            tenant: vellev_core::TenantId::new("tenant-a").unwrap(),
            partition: vellev_core::Partition(0),
        };
        let layout = Layout(&partition);
        let author = Author {
            generation: Generation(2),
            writer: uuid::Uuid::from_u128(0xabc),
        };
        let at = LogPosition(7);
        let nonce = 0x0123_4567_89ab_cdef;
        // As a flush, a merge and a restore from another store's files name
        // them (Writer::named), each kind of checkpoint file.
        for kind in ["segment", "index", "vectors"] {
            for marker in ["", "-merged", "-restored"] {
                let key = layout.nonced(kind, at, &author, nonce, marker);
                let named = Named {
                    kind,
                    position: at,
                    generation: Generation(2),
                    merged: marker == "-merged",
                    restored: marker == "-restored",
                    nonce: Some(nonce),
                };
                assert_eq!(layout.named_by(&key), Some(named), "{key}");
            }
            // A nonce of any value, its sixteen digits as drawn.
            let key = layout.nonced(kind, at, &author, 0, "");
            assert_eq!(
                layout.named_by(&key).and_then(|named| named.nonce),
                Some(0),
                "{key}"
            );
            // Merged with no nonce, as before DW-658, read as ever.
            let key = layout.named(kind, at, &author, "-merged");
            let named = layout.named_by(&key).unwrap();
            assert!(named.merged && named.nonce.is_none(), "{key}");
            // Restored with no nonce: no writer ever named one so, as
            // `restore_from` (DW-680) came after the nonce (DW-658).
            let key = layout.named(kind, at, &author, "-restored");
            assert_eq!(layout.named_by(&key), None, "{key}");
        }
        let segment = layout.nonced("segment", at, &author, nonce, "");
        let drawn = format!("-{nonce:016x}");
        let refused = [
            // A nonce of other than sixteen lowercase hex digits.
            segment.replace(&drawn, &drawn[..16]),
            segment.replace(&drawn, &format!("{drawn}0")),
            segment.replace(&drawn, &drawn.to_uppercase()),
            segment.replace(&drawn, "-0123456789abcdeg"),
            // Two of them, or one after the marker.
            segment.replace(&drawn, &format!("{drawn}{drawn}")),
            layout.named("segment", at, &author, &format!("-merged{drawn}")),
            // Two markers.
            layout.nonced("segment", at, &author, nonce, "-merged-restored"),
            // A head, never named with either.
            layout.nonced("log", at, &author, nonce, ""),
            layout.nonced("manifest", at, &author, nonce, ""),
            layout.named("log", at, &author, "-restored"),
            layout.named("manifest", at, &author, "-restored"),
        ];
        for key in &refused {
            assert_eq!(layout.named_by(key), None, "{key}");
        }
    }

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

    /// The stores a file is created in by `object_store` itself: in
    /// memory, and a directory on local disk, kept for as long as the
    /// store is.
    fn creating_stores() -> Vec<(&'static str, ObjectBlobs, Option<tempfile::TempDir>)> {
        let dir = tempfile::tempdir().unwrap();
        let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let on_disk = ObjectBlobs::on_disk(Arc::new(local), dir.path());
        vec![
            ("memory", ObjectBlobs::new(Arc::new(InMemory::new())), None),
            ("file", on_disk, Some(dir)),
        ]
    }

    #[tokio::test]
    async fn a_file_created_only_once_is_refused_a_second_create_and_keeps_its_first_bytes() {
        for (store, blobs, _dir) in creating_stores() {
            let first = blobs.put_new("t/0/link", Bytes::from("A")).await;
            assert_eq!(first.unwrap(), Created::Written, "{store}");
            let second = blobs.put_new("t/0/link", Bytes::from("B")).await;
            assert_eq!(second.unwrap(), Created::AlreadyThere, "{store}");
            assert_eq!(blobs.get("t/0/link").await.unwrap(), "A", "{store}");
        }
    }

    #[tokio::test]
    async fn of_creates_racing_for_one_key_exactly_one_is_written_and_its_bytes_are_kept() {
        for (store, blobs, _dir) in creating_stores() {
            let blobs = Arc::new(blobs);
            for round in 0..20 {
                let key = format!("t/0/race-{round}");
                let racing = (0..20).map(|racer| {
                    let (blobs, key) = (blobs.clone(), key.clone());
                    tokio::spawn(async move {
                        let bytes = Bytes::from(format!("racer {racer}"));
                        (racer, blobs.put_new(&key, bytes).await.unwrap())
                    })
                });
                let came = futures::future::join_all(racing).await;
                let written: Vec<_> = came
                    .into_iter()
                    .map(|joined| joined.unwrap())
                    .filter(|(_, created)| *created == Created::Written)
                    .map(|(racer, _)| racer)
                    .collect();
                assert_eq!(written.len(), 1, "{store} round {round}: {written:?}");
                let kept = blobs.get(&key).await.unwrap();
                assert_eq!(kept, format!("racer {}", written[0]), "{store} {round}");
            }
        }
    }

    #[tokio::test]
    async fn a_file_larger_than_one_request_takes_is_refused_a_create_and_nothing_goes_up() {
        let parts = Transfers {
            part_bytes: 8,
            ..Transfers::default()
        };
        for (store, blobs, _dir) in creating_stores() {
            let blobs = blobs.transferring(parts);
            assert_eq!(blobs.creates_at_most(), Some(8), "{store}");
            let refused = blobs.put_new("t/0/big", Bytes::from("123456789")).await;
            // Known to have put nothing, and so not to be asked again.
            assert!(
                matches!(refused, Err(StorageError::NotCreated { .. })),
                "{store}: {refused:?}"
            );
            let missing = blobs.get("t/0/big").await;
            assert!(
                matches!(missing, Err(StorageError::Missing { .. })),
                "{store}"
            );
            // One request's worth is created.
            let fits = blobs.put_new("t/0/big", Bytes::from("12345678")).await;
            assert_eq!(fits.unwrap(), Created::Written, "{store}");
        }
    }

    /// A store that says nothing of creating only once, and notes every
    /// put it is asked for.
    #[derive(Debug, Default)]
    struct Plain(std::sync::Mutex<Vec<String>>);

    #[async_trait]
    impl Blobs for Plain {
        async fn put(&self, key: &str, _bytes: Bytes) -> Result<()> {
            self.0.lock().unwrap().push(key.to_string());
            Ok(())
        }
        async fn get(&self, key: &str) -> Result<Bytes> {
            Err(StorageError::Missing {
                key: key.to_string(),
            })
        }
        async fn list(&self, _prefix: &str) -> Result<Vec<Listed>> {
            Ok(Vec::new())
        }
        async fn delete(&self, _key: &str) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_store_that_does_not_say_how_to_create_only_once_refuses_and_puts_nothing() {
        let plain = Plain::default();
        assert_eq!(plain.creates_at_most(), None);
        let refused = plain.put_new("t/0/link", Bytes::from("A")).await;
        assert!(
            matches!(refused, Err(StorageError::NotCreated { .. })),
            "{refused:?}"
        );
        assert!(plain.0.lock().unwrap().is_empty());
    }

    /// A store that puts every file it is asked to create, there or not.
    #[derive(Debug)]
    struct Careless(ObjectBlobs);

    #[async_trait]
    impl Blobs for Careless {
        async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
            self.0.put(key, bytes).await
        }
        async fn put_new(&self, key: &str, bytes: Bytes) -> Result<Created> {
            self.0.put(key, bytes).await.map(|()| Created::Written)
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
    }

    #[tokio::test]
    async fn a_store_is_taken_to_create_only_once_only_if_it_refuses_a_second_create() {
        let key = "_backups/_probe/create-only";
        for (store, blobs, _dir) in creating_stores() {
            // Its first check, which creates the file, and every one after.
            creates_only_once(&blobs, key).await.unwrap();
            creates_only_once(&blobs, key).await.unwrap();
            assert!(blobs.get(key).await.is_ok(), "{store}");
        }
        let careless = Careless(ObjectBlobs::new(Arc::new(InMemory::new())));
        let ignored = creates_only_once(&careless, key).await;
        assert!(matches!(ignored, Err(StorageError::Blob { .. })));
        let unable = creates_only_once(&Plain::default(), key).await;
        assert!(
            matches!(unable, Err(StorageError::NotCreated { .. })),
            "{unable:?}"
        );
    }

    #[tokio::test]
    async fn a_create_through_the_cache_is_refused_and_nothing_is_put_or_copied() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let cached = CachedBlobs::new(store.clone(), dir.path());
        assert_eq!(cached.creates_at_most(), None);
        let key = "t/0/log/1.log";
        let file = crate::format::encode(crate::format::Kind::Log, &"A").unwrap();
        let refused = cached.put_new(key, file).await;
        assert!(
            matches!(refused, Err(StorageError::NotCreated { .. })),
            "{refused:?}"
        );
        assert!(matches!(
            store.get(key).await,
            Err(StorageError::Missing { .. })
        ));
        assert!(cached.local(key).await.unwrap().is_none());
    }

    /// Files under `links/` and beside it, in `store`.
    async fn listed_from(store: &dyn Blobs) {
        for key in [
            "t/links/1",
            "t/links/2",
            "t/links/3",
            "t/linksx/9",
            "u/links/4",
        ] {
            store.put(key, Bytes::from("x")).await.unwrap();
        }
    }

    /// The keys `store` lists under `t/links/` after `after`, `limit` at most.
    async fn keys_from(store: &dyn Blobs, after: &str, limit: usize) -> Vec<String> {
        let listed = store.list_from("t/links/", after, limit).await.unwrap();
        let mut keys: Vec<_> = listed.into_iter().map(|file| file.key).collect();
        keys.sort();
        keys
    }

    /// A listing from a key gives files under the prefix after the key,
    /// never the key itself, and no more than asked for: in key order in
    /// memory, as in S3; on local disk, after the key in some order.
    #[tokio::test]
    async fn a_listing_from_a_key_gives_only_files_under_the_prefix_after_it() {
        let dir = tempfile::tempdir().unwrap();
        let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let stores: [(&str, Arc<dyn ObjectStore>); 2] = [
            ("memory", Arc::new(InMemory::new())),
            ("local", Arc::new(local)),
        ];
        for (name, store) in stores {
            let store = ObjectBlobs::new(store);
            listed_from(&store).await;
            let all = keys_from(&store, "t/links/1", 10).await;
            assert_eq!(all, ["t/links/2", "t/links/3"], "{name}");
            assert!(
                keys_from(&store, "t/links/3", 10).await.is_empty(),
                "{name}"
            );
            let one = keys_from(&store, "t/links/0", 1).await;
            assert_eq!(one.len(), 1, "{name}");
            assert!(one[0].as_str() > "t/links/0", "{name}");
            if name == "memory" {
                assert_eq!(one, ["t/links/1"]);
            }
            assert!(keys_from(&store, "t/links/1", 0).await.is_empty(), "{name}");
        }
    }

    /// A store that says nothing of listing from a key refuses to, and the
    /// cache lists from a key as its store does.
    #[tokio::test]
    async fn a_listing_from_a_key_is_refused_by_a_store_that_does_not_say_it_can() {
        let refused = Plain::default().list_from("t/links/", "t/links/1", 1).await;
        assert!(
            matches!(refused, Err(StorageError::NotListed { .. })),
            "{refused:?}"
        );
        let (_dir, cached) = cache();
        listed_from(&cached).await;
        assert_eq!(
            keys_from(&cached, "t/links/1", 10).await,
            ["t/links/2", "t/links/3"]
        );
    }

    /// A cache over a store in memory, in a directory of its own.
    fn cache() -> (tempfile::TempDir, CachedBlobs) {
        let dir = tempfile::tempdir().unwrap();
        let inner = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
        let cached = CachedBlobs::new(inner, dir.path());
        (dir, cached)
    }

    fn vectors(n: u8) -> String {
        format!("tenant-a/0/vectors/0000000000000000000{n}-x.vectors")
    }

    fn frame(n: u8) -> Bytes {
        crate::format::encode(crate::format::Kind::Index, &n).unwrap()
    }

    /// A file's copy on local disk is given to another key that holds the
    /// same, as the one file under both names: either name deleted leaves
    /// the other, and nothing is left part written.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_copy_on_local_disk_is_linked_to_a_key_holding_the_same() {
        use std::os::unix::fs::MetadataExt;
        let (dir, cached) = cache();
        let (from, to) = (vectors(1), vectors(2));
        cached.put(&from, frame(1)).await.unwrap();
        cached.inner.put(&to, frame(1)).await.unwrap();

        let linked = cached.link_local(&from, &to).await.unwrap();

        let (from_path, to_path) = (dir.path().join(&from), dir.path().join(&to));
        assert_eq!(linked, Linked::Linked(to_path.clone()));
        let inode = |path: &Path| std::fs::metadata(path).unwrap().ino();
        assert_eq!(inode(&from_path), inode(&to_path));
        let near: Vec<_> = std::fs::read_dir(to_path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(near.len(), 2, "{near:?}");

        cached.delete(&from).await.unwrap();
        assert_eq!(std::fs::read(&to_path).unwrap(), frame(1));
        assert!(cached.local(&to).await.unwrap().is_some());
        cached.put(&from, frame(1)).await.unwrap();
        cached.delete(&to).await.unwrap();
        assert!(cached.local(&from).await.unwrap().is_some());
    }

    /// No copy of the file on local disk, or no plain key, is no link: the
    /// caller reads the new key as any file. (Off Unix nothing is read in
    /// place, and nothing is linked.)
    #[cfg(unix)]
    #[tokio::test]
    async fn a_copy_not_on_local_disk_is_not_linked() {
        let (_dir, cached) = cache();
        let (from, to) = (vectors(1), vectors(2));
        cached.inner.put(&from, frame(1)).await.unwrap();
        cached.inner.put(&to, frame(1)).await.unwrap();
        assert_eq!(
            cached.link_local(&from, &to).await.unwrap(),
            Linked::Unlinked
        );
        assert!(cached.local(&to).await.unwrap().is_none());
        let stray = "tenant-a/0/../vectors/x.vectors";
        assert_eq!(
            cached.link_local(stray, &to).await.unwrap(),
            Linked::Unlinked
        );
    }

    /// A store that reads nothing in place has nothing to link; one that
    /// reads its own files in place cannot give one key another's.
    #[tokio::test]
    async fn a_store_without_a_cache_links_nothing() {
        let (from, to) = (vectors(1), vectors(2));
        let memory = ObjectBlobs::new(Arc::new(InMemory::new()));
        assert_eq!(
            memory.link_local(&from, &to).await.unwrap(),
            Linked::Unmapped
        );
        let dir = tempfile::tempdir().unwrap();
        let files = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
        let on_disk = ObjectBlobs::on_disk(Arc::new(files), dir.path());
        on_disk.put(&from, frame(1)).await.unwrap();
        assert_eq!(
            on_disk.link_local(&from, &to).await.unwrap(),
            Linked::Unlinked
        );
    }

    /// The copy linked is taken as summed under its new name only if it
    /// was summed under the old, as it lay just before.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_linked_copy_is_taken_as_summed_only_if_the_old_one_was() {
        let (dir, cached) = cache();
        let (from, to) = (vectors(3), vectors(4));
        for (summed, from, to) in [(false, vectors(5), vectors(6)), (true, from, to)] {
            cached.put(&from, frame(1)).await.unwrap();
            cached.inner.put(&to, frame(1)).await.unwrap();
            let from_path = dir.path().join(&from);
            if summed {
                crate::column::remember_summed(crate::column::seen(&from_path).unwrap());
            }
            cached.link_local(&from, &to).await.unwrap();
            let to_seen = crate::column::seen(&dir.path().join(&to)).unwrap();
            assert_eq!(crate::column::summed_before(&to_seen), summed, "{summed}");
        }
    }
}
