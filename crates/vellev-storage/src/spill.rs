//! A merge's vectors gathered on local disk (node memory note, M8): in a
//! file of the store's cache, under a name of its own, written a piece at a
//! time and, once whole, read in place. So the frame is on the heap at no
//! stage of the build, and, taken up by the cache as the file's copy once
//! the store has it, is written once.
//!
//! Its name lasts as long as anything may read it: the last to let a
//! [`Spill`] go takes it away, on whichever thread and however it ends. A
//! process that dies never does, so a spill is held locked from its first
//! moment, as the system lets go of a dead process's locks: whoever opens
//! the cache next finds one nobody holds, and removes it at once, however
//! new, leaving alone those of another process sharing the directory.
//!
//! A file is never read part written. It is written through the file, all of
//! it, and only then mapped: a disk that fills is told of by the write, as
//! an error the merge fails on, never by a fault in a mapping. Nothing
//! writes to it once it is mapped. It is only ever renamed or unlinked,
//! either of which leaves a mapping on it as it was, which is why spills are
//! made on Unix alone, as [`Column::map`] maps on Unix alone.
//!
//! A page of a mapping gone from memory is a read from disk on whichever
//! thread touches it, which no bound can give up and which holds the thread
//! for as long as the disk does. The build reads it on threads of its own,
//! but a store sends what it is given on the async threads, a whole part at
//! a time as it signs it. So what goes up is read from the mapping by the
//! disk's own steps, a piece at a time (see [`Spill::pieces`]), and held in
//! memory until it has gone: a store's part, no more.
//!
//! [`Column::map`]: crate::column::Column::map

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use tokio::runtime::Handle;

use crate::blobs::Disk;
use crate::format::{self, Kind, BODY};
use crate::freed::off_the_async_threads;
use crate::stopping::{MergeStage, Stopping};
use crate::{Result, StorageError};

/// How a spill's name ends, by which [`left_behind`] is asked of it.
pub(crate) const SPILL_EXTENSION: &str = "spill";

/// How many bytes of a frame are held in memory as it is written, at the
/// most: a step on the disk for each, so that a slow disk is told of as soon
/// as any other write to it would be, and a stop waits for no more than one.
pub(crate) const SPILLED_AT_ONCE: usize = 4 * 1024 * 1024;

/// Where a store's frames may be gathered on local disk: a directory of the
/// same file system as the store's copies, so that a copy is taken up by
/// renaming it, and the disk every step there is run on.
#[derive(Debug, Clone)]
pub struct Spills {
    dir: PathBuf,
    disk: Disk,
}

impl Spills {
    pub(crate) fn in_dir(dir: PathBuf, disk: Disk) -> Self {
        Self { dir, disk }
    }

    /// What has been spilled here, so far.
    pub fn figures(&self) -> SpillFigures {
        SpillFigures::default()
    }

    /// How many bytes of a frame are in memory as it goes up, at the most: a
    /// part of the store's, the piece that fills it, and no more.
    pub(crate) fn held_going_up(&self) -> usize {
        self.disk.part_bytes() + SPILLED_AT_ONCE
    }

    /// As asked of by a merge whose gather blocks on `runtime` for each
    /// step, for as long as `stopping` lets it, of the file to be put under
    /// `key`, by which any failure is told.
    pub(crate) fn asked(self, runtime: Handle, stopping: Stopping, key: String) -> SpillAsked {
        SpillAsked {
            spills: self,
            runtime,
            stopping,
            key,
        }
    }
}

/// What is needed to begin a spill, for a gather that does not know yet
/// whether it has any vectors to spill.
#[derive(Debug)]
pub(crate) struct SpillAsked {
    spills: Spills,
    runtime: Handle,
    stopping: Stopping,
    key: String,
}

/// A name in the cache, taken away as this goes. A file that nothing refers
/// to by name any more is the system's to free, once what has it open or
/// mapped has let it go.
#[derive(Debug, Default)]
struct Name(Option<PathBuf>, Option<Arc<SpillsTold>>);

impl Drop for Name {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            if let Some(told) = self.1.take() {
                told.unlinks();
            }
            if let Err(error) = std::fs::remove_file(&path) {
                // Already gone, as by a cache emptied under it.
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(?path, %error, "a spill could not be removed");
                }
            }
        }
    }
}

fn unwritten(key: &str, error: std::io::Error) -> StorageError {
    StorageError::Blob {
        key: key.to_string(),
        source: error.into(),
    }
}

/// A frame being written to its spill, in order, from its first byte: its
/// header's room, then its body.
#[derive(Debug)]
pub(crate) struct Spilling {
    file: Arc<File>,
    name: Name,
    disk: Disk,
    runtime: Handle,
    stopping: Stopping,
    /// The file it is to be put as, by which any failure is told.
    key: String,
    /// How much of the frame is written.
    written: u64,
    /// The sum of what of its body is written.
    body: crc32fast::Hasher,
}

impl SpillAsked {
    /// A spill begun under a name drawn at random, locked before anyone can
    /// have reason to look at it: `None` if the merge is stopped first.
    pub(crate) fn begin(self) -> Result<Option<Spilling>> {
        let Self {
            spills: Spills { dir, disk },
            runtime,
            stopping,
            key,
        } = self;
        let name = format!("{}.{SPILL_EXTENSION}", uuid::Uuid::new_v4().simple());
        let path = dir.join(name);
        // On one step, which goes on to its end whoever stops waiting: all
        // it made goes with what it returns, should nobody take that.
        let told = disk.spills_are_told();
        let begun = {
            move || {
                std::fs::create_dir_all(&dir)?;
                let file = File::options()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                let name = Name(Some(path), told);
                match file.try_lock() {
                    Ok(()) => Ok((file, name)),
                    // A file system with no locks: spills left behind there
                    // go as anything left part written does, once stale.
                    Err(std::fs::TryLockError::Error(_)) => Ok((file, name)),
                    // One opening the cache got there first: it takes the
                    // name away, not knowing this is not left behind.
                    Err(std::fs::TryLockError::WouldBlock) => {
                        Err(std::io::ErrorKind::WouldBlock.into())
                    }
                }
            }
        };
        let waited = stopping.unless_stopped(MergeStage::Gathering, disk.run(0, begun));
        let Some(begun) = runtime.block_on(waited) else {
            return Ok(None);
        };
        let (file, name) = begun.map_err(|error| unwritten(&key, error))?;
        Ok(Some(Spilling {
            file: Arc::new(file),
            name,
            disk,
            runtime,
            stopping,
            key,
            written: 0,
            body: crc32fast::Hasher::new(),
        }))
    }
}

impl Spilling {
    /// The frame's next `words`, written and taken out of it, its room kept
    /// for the next: `false` if the merge is stopped first. The step the
    /// wait for it gives up runs on with what it was given, and may write
    /// on to a file that has lost its name, which is nothing to anyone.
    pub(crate) fn written(&mut self, words: &mut Vec<u32>) -> Result<bool> {
        let bytes = words.len() * 4;
        let piece = std::mem::take(words);
        let frame: &[u8] = bytemuck::cast_slice(&piece);
        // The header goes in last, over the room left for it.
        let header = BODY.saturating_sub(self.written as usize).min(bytes);
        self.body.update(&frame[header..]);
        let (file, told) = (self.file.clone(), self.disk.spills_are_told());
        let end = self.written + bytes as u64;
        let step = move || {
            if let Some(told) = told {
                told.writes(end)?;
            }
            (&*file).write_all(bytemuck::cast_slice(&piece))?;
            Ok(piece)
        };
        let run = self.disk.run(bytes, step);
        let waited = self.stopping.unless_stopped(MergeStage::Gathering, run);
        let Some(stepped) = self.runtime.block_on(waited) else {
            return Ok(false);
        };
        let mut piece = stepped.map_err(|error| unwritten(&self.key, error))?;
        piece.clear();
        *words = piece;
        self.written = end;
        Ok(true)
    }

    /// The frame, all of it written, sealed as format `version` and read in
    /// place from now on: `None` if the merge is stopped first.
    pub(crate) fn sealed(mut self, version: u16) -> Result<Option<Spill>> {
        let length = self.written as usize - BODY;
        let header = format::header(Kind::Vectors, version, length, &self.body);
        let file = self.file.clone();
        let step = move || {
            let mut to = &*file;
            to.seek(SeekFrom::Start(0))?;
            to.write_all(&header)?;
            // SAFETY: nothing writes to the file from here on: the only
            // write given up on would have failed the gather before this,
            // and whatever else comes to it by name only renames or unlinks
            // it, either of which leaves the mapping as it is.
            unsafe { memmap2::Mmap::map(&*file) }
        };
        let run = self.disk.run(0, step);
        let waited = self.stopping.unless_stopped(MergeStage::Gathering, run);
        let Some(mapped) = self.runtime.block_on(waited) else {
            return Ok(None);
        };
        let map = mapped.map_err(|error| unwritten(&self.key, error))?;
        if let Some(told) = self.disk.spills_are_told() {
            let from = map.as_ptr() as usize;
            told.mapped_from.store(from, Ordering::SeqCst);
            told.mapped_to.store(from + map.len(), Ordering::SeqCst);
        }
        Ok(Some(Spill {
            map: Some(map),
            version,
            file: Some(self.file),
            name: Mutex::new(std::mem::take(&mut self.name)),
            disk: self.disk,
        }))
    }
}

/// A frame gathered in its spill: written whole, and read in place by all
/// that holds it. The mapping and the name go with the last of those, on no
/// async thread, however it goes: the column and the frame its build holds,
/// the bytes a store is sent, whoever keeps those on.
#[derive(Debug)]
pub struct Spill {
    map: Option<memmap2::Mmap>,
    version: u16,
    /// Open for as long as it is mapped, with the lock that says so.
    file: Option<Arc<File>>,
    /// Where it lies, until a cache takes it up, when it is the cache's.
    name: Mutex<Name>,
    /// What it is read on as it goes up.
    disk: Disk,
}

/// The name held, whoever panicked holding it: a rename or a removal, of
/// which either leaves it as it says.
fn named(name: &Mutex<Name>) -> std::sync::MutexGuard<'_, Name> {
    name.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Spill {
    /// The frame.
    pub(crate) fn bytes(&self) -> &[u8] {
        self.map.as_deref().expect("mapped until dropped")
    }

    /// The format it is sealed as.
    pub(crate) fn version(&self) -> u16 {
        self.version
    }

    /// The frame, to go up as `key`, a piece of at most [`SPILLED_AT_ONCE`]
    /// at a time, each read into memory by a step of the disk's and waited
    /// for as any is: so that nothing that takes them reads the mapping,
    /// and a disk that stops answering fails the upload at its bound.
    pub(crate) fn pieces(
        self: Arc<Self>,
        key: &str,
    ) -> futures::stream::BoxStream<'static, Result<Bytes>> {
        use futures::StreamExt;
        let key = key.to_string();
        let pieces = futures::stream::try_unfold(0, move |at| {
            let (spill, key) = (self.clone(), key.clone());
            async move {
                let end = (at + SPILLED_AT_ONCE).min(spill.bytes().len());
                if at == end {
                    return Ok(None);
                }
                let (disk, told) = (spill.disk.clone(), spill.disk.spills_are_told());
                let read = disk.run(end - at, move || {
                    if let Some(told) = told {
                        told.read();
                    }
                    Ok(Bytes::copy_from_slice(&spill.bytes()[at..end]))
                });
                let piece = read.await.map_err(|error| unwritten(&key, error))?;
                Ok(Some((piece, end)))
            }
        });
        pieces.boxed()
    }

    /// The file's name from now on is `path`, whatever was there: it is
    /// no spill any more, and is left there as this goes.
    pub(crate) fn renamed_to(&self, path: &Path) -> std::io::Result<()> {
        let mut name = named(&self.name);
        let from = name.0.as_ref().ok_or(std::io::ErrorKind::NotFound)?;
        std::fs::rename(from, path)?;
        name.0 = None;
        Ok(())
    }
}

impl Drop for Spill {
    /// Off the async threads: the name's removal is a step on a disk that
    /// may not answer, and a mapping of gigabytes takes as long to let go
    /// as what was freed before it. Should that thread never come, as the
    /// runtime goes, these go with the work unrun, and the name with them.
    fn drop(&mut self) {
        let map = self.map.take();
        let file = self.file.take();
        let name = std::mem::take(&mut *named(&self.name));
        off_the_async_threads(move || {
            drop(map);
            drop(file);
            drop(name);
        });
    }
}

/// Whether the spill at `path` was left by a process that is gone: nobody
/// holds its lock. On a file system with no locks, nothing tells: it is
/// left to `stale`, as a file left part written is.
pub(crate) fn left_behind(path: &Path, stale: impl FnOnce() -> bool) -> bool {
    let Ok(file) = File::options().read(true).write(true).open(path) else {
        return false;
    };
    nobody_holds(file.try_lock(), stale)
}

/// What the lock of a spill came to says of it: see [`left_behind`].
fn nobody_holds(
    locked: std::result::Result<(), std::fs::TryLockError>,
    stale: impl FnOnce() -> bool,
) -> bool {
    match locked {
        Ok(()) => true,
        Err(std::fs::TryLockError::WouldBlock) => false,
        Err(std::fs::TryLockError::Error(_)) => stale(),
    }
}

/// What is spilled to a disk, as a node reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpillFigures {
    /// How many bytes of spills are on the disk now, each as much as its
    /// frame comes to whole, from the moment it begins.
    pub bytes: u64,
    /// How many merges have not spilled for want of room, and were skipped.
    pub skipped: u64,
    /// How many spills the disk has failed.
    pub failed: u64,
}

/// What a test has a disk do to what is spilled to it, as a disk that fills
/// or stops answering would. Not for use.
#[doc(hidden)]
#[derive(Debug)]
pub struct SpillsTold {
    /// How many bytes of a spill the disk takes before it is full.
    pub at_most: AtomicU64,
    /// How long, in milliseconds, each write takes at the least.
    pub taking: AtomicU64,
    /// How long, in milliseconds, a cache takes to take a spill up.
    pub taken_up_in: AtomicU64,
    /// How long, in milliseconds, each piece of a spill takes to read as it
    /// goes up.
    pub reading: AtomicU64,
    /// How many bytes have been written to spills.
    pub written: AtomicU64,
    /// How long, in milliseconds, a spill's name takes to remove.
    pub unlinking: AtomicU64,
    /// How long, in milliseconds, a cache goes before it looks again, once
    /// it has taken a spill up.
    pub looks_again_in: AtomicU64,
    /// How many bytes the disk says are free, if not [`Self::ASK_THE_DISK`].
    pub free: AtomicU64,
    /// Where the last spill made is mapped, and where it ends: for a store
    /// a test has read as a mapping is read.
    pub mapped_from: AtomicUsize,
    pub mapped_to: AtomicUsize,
}

impl Default for SpillsTold {
    fn default() -> Self {
        Self {
            at_most: AtomicU64::new(u64::MAX),
            taking: AtomicU64::new(0),
            taken_up_in: AtomicU64::new(0),
            reading: AtomicU64::new(0),
            written: AtomicU64::new(0),
            unlinking: AtomicU64::new(0),
            looks_again_in: AtomicU64::new(0),
            free: AtomicU64::new(Self::ASK_THE_DISK),
            mapped_from: AtomicUsize::new(0),
            mapped_to: AtomicUsize::new(0),
        }
    }
}

impl SpillsTold {
    /// That the disk is asked how much is free, as if told nothing.
    pub const ASK_THE_DISK: u64 = u64::MAX;

    /// Whether `bytes` lies in the last spill's mapping.
    pub fn maps(&self, bytes: *const u8) -> bool {
        let from = self.mapped_from.load(Ordering::SeqCst);
        (from..self.mapped_to.load(Ordering::SeqCst)).contains(&(bytes as usize))
    }

    /// A spill's name about to be removed, as told.
    fn unlinks(&self) {
        let taking = self.unlinking.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(taking));
    }

    /// A cache about to look again at a spill it took up, as told.
    pub(crate) fn looks_again(&self) {
        let taking = self.looks_again_in.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(taking));
    }

    /// A piece about to be read as it goes up, as told.
    fn read(&self) {
        let taking = self.reading.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(taking));
    }

    /// A spill about to be taken up, as told.
    pub(crate) fn taken_up(&self) {
        let taking = self.taken_up_in.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(taking));
    }

    /// A write that comes to `end` bytes of its spill, as told.
    fn writes(&self, end: u64) -> std::io::Result<()> {
        let taking = self.taking.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(taking));
        self.written.store(end, Ordering::SeqCst);
        match end > self.at_most.load(Ordering::SeqCst) {
            true => Err(std::io::ErrorKind::StorageFull.into()),
            false => Ok(()),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A spill of a few words in `dir`, sealed, made on a thread that may
    /// wait on the disk.
    pub(crate) async fn spill_in(dir: &Path) -> Arc<Spill> {
        let spills = Spills::in_dir(dir.to_path_buf(), Disk::shared().clone());
        let asked = spills.asked(Handle::current(), Stopping::never(), "k".into());
        let made = tokio::task::spawn_blocking(move || {
            let mut spilling = asked.begin().unwrap().unwrap();
            let mut words = vec![0; 16];
            assert!(spilling.written(&mut words).unwrap());
            spilling.sealed(1).unwrap().unwrap()
        });
        Arc::new(made.await.unwrap())
    }

    fn names(dir: &Path) -> usize {
        std::fs::read_dir(dir).unwrap().count()
    }

    /// The name goes with the last that holds the spill, on whichever thread
    /// that is, and not before.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_spill_s_name_goes_with_the_last_that_holds_it() {
        let dir = tempfile::tempdir().unwrap();
        let spill = spill_in(dir.path()).await;
        let other = spill.clone();
        drop(spill);
        assert_eq!(names(dir.path()), 1);
        drop(other);
        let wait = async {
            while names(dir.path()) > 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        };
        tokio::time::timeout(Duration::from_secs(30), wait)
            .await
            .expect("the name stays");
    }

    /// One let go as the runtime shuts down, when nothing is left to do
    /// work off the async threads, leaves its name no more than any other.
    #[test]
    fn a_spill_let_go_as_its_runtime_goes_leaves_no_name() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let spill = runtime.block_on(spill_in(dir.path()));
        let held = runtime.spawn(async move {
            let _spill = spill;
            std::future::pending::<()>().await;
        });
        assert_eq!(names(dir.path()), 1);
        drop(held);
        drop(runtime);
        assert_eq!(names(dir.path()), 0, "left as the runtime went");
    }

    /// A step given up on, and a stop, leave nothing, and whatever is
    /// written to a file that has lost its name is nothing to anyone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_spill_nobody_goes_on_with_leaves_no_name() {
        let dir = tempfile::tempdir().unwrap();
        let spills = Spills::in_dir(dir.path().to_path_buf(), Disk::shared().clone());
        let asked = spills.asked(Handle::current(), Stopping::never(), "k".into());
        let begun = tokio::task::spawn_blocking(move || asked.begin().unwrap().unwrap());
        let spilling = begun.await.unwrap();
        assert_eq!(names(dir.path()), 1);
        drop(spilling);
        assert_eq!(names(dir.path()), 0);
    }

    /// A spill whose file a lock cannot be had of is one nobody holds only
    /// as what is left part written is: once stale. A file system with locks
    /// says at once. A file not there is left behind by nobody.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn only_a_spill_nobody_holds_is_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let spill = spill_in(dir.path()).await;
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert!(!left_behind(&path, || true), "held");
        let held = std::fs::File::open(&path).unwrap();
        drop(spill);
        let wait = async {
            while path.exists() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        };
        tokio::time::timeout(Duration::from_secs(30), wait)
            .await
            .expect("the name stays");
        drop(held);
        assert!(!left_behind(&path, || true), "not there");
        std::fs::write(&path, b"left").unwrap();
        assert!(left_behind(&path, || false), "nobody holds it");
    }

    /// What goes up is in memory of its own, the frame a piece at a time, so
    /// that a store's thread stuck on one keeps that piece, not the spill:
    /// its name goes with the last that holds the spill, whoever keeps the
    /// pieces on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn what_goes_up_of_a_spill_holds_nothing_of_it() {
        use futures::TryStreamExt;
        let dir = tempfile::tempdir().unwrap();
        let spill = spill_in(dir.path()).await;
        let frame = spill.bytes().to_vec();
        let mapped = spill.bytes().as_ptr_range();
        let pieces: Vec<Bytes> = spill.pieces("k").try_collect().await.unwrap();
        assert_eq!(pieces.concat(), frame);
        for piece in &pieces {
            assert!(!mapped.contains(&piece.as_ptr()), "read from the mapping");
        }
        let wait = async {
            while names(dir.path()) > 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        };
        tokio::time::timeout(Duration::from_secs(30), wait)
            .await
            .expect("the pieces keep the name");
        assert_eq!(pieces.concat(), frame);
    }

    /// Where a file system has no locks, nothing says who holds a spill: it
    /// goes as what is left part written goes, once stale, and not before.
    /// Where it has, what the lock says is all that is asked.
    #[test]
    fn a_file_system_with_no_locks_leaves_a_spill_until_it_is_stale() {
        use std::fs::TryLockError::{Error, WouldBlock};
        let unsupported = || Err(Error(std::io::ErrorKind::Unsupported.into()));
        assert!(!nobody_holds(unsupported(), || false));
        assert!(nobody_holds(unsupported(), || true));
        assert!(!nobody_holds(Err(WouldBlock), || true), "held");
        assert!(nobody_holds(Ok(()), || panic!("the lock says")));
    }
}
