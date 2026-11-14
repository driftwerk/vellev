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
//! Nothing waits for a name to go. Removing one is a step on a disk that may
//! have stopped, and a build blocked on it would hold its turn among the
//! node's merges for as long: so it is sent to a thread of the disk's and
//! left there, on any path. A process stopping waits for it as long as it
//! waits for an upload's abort; what is left then, a lock no process holds
//! now says of it, and the next open of the cache removes it. Only an open
//! does: nothing sweeps a cache as a node runs.
//!
//! How much room there is is asked before a spill begins, as its size is
//! known exactly and it is held for the whole of a build. A merge with too
//! little is skipped, not written until the disk is full, which would fail a
//! tenant's open or a flush's copy beside it.
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

use crate::blobs::{disk_room, Aborts, Disk};
use crate::column::{seen, Seen};
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

/// How much must be free on a spill's disk past the spill itself, at the
/// least: what a node's writers may flush beside it, each at its backlog, a
/// copy of a few hundred MB, about a gibibyte for three of them.
const ROOM_SPARED_AT_LEAST: u64 = 1024 * 1024 * 1024;

/// How much a spill of `frame` bytes leaves free, past itself: as much again,
/// as nothing reserves what is checked and another merge's spill beside this
/// comes to about as much, or what flushes take, if that is more. It goes by
/// what competes for the room, not by the size of the disk: what ext4 keeps
/// for root is left out of what it says is free already, 5% by default.
fn room_spared(frame: u64) -> u64 {
    ROOM_SPARED_AT_LEAST.max(frame)
}

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
        self.disk.spills_counted().figures()
    }

    /// How many bytes of a frame are in memory as it goes up, at the most:
    /// pieces enough to hold a part of the store's, and the piece that came
    /// after it, as a part ends in a piece that lives on for the part after.
    pub(crate) fn held_going_up(&self) -> usize {
        held_of_parts(self.disk.part_bytes())
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

/// See [`Spills::held_going_up`].
fn held_of_parts(part: usize) -> usize {
    part.next_multiple_of(SPILLED_AT_ONCE) + SPILLED_AT_ONCE
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

/// A name in the cache, and the disk it lies on, taken away as this goes:
/// on a thread of the disk's that nobody waits for, however this goes.
#[derive(Debug, Default)]
struct Name {
    path: Option<PathBuf>,
    disk: Option<Disk>,
    /// What the frame comes to whole, counted as on the disk until the name
    /// is gone or the file is a cache's.
    frame: u64,
}

impl Name {
    /// The name is the cache's now.
    fn given_up(&mut self) {
        self.path = None;
        if let Some(disk) = &self.disk {
            disk.spills_counted().gone(self.frame);
        }
    }
}

impl Drop for Name {
    fn drop(&mut self) {
        let (Some(path), Some(disk)) = (self.path.take(), self.disk.take()) else {
            return;
        };
        let (told, frame) = (disk.spills_are_told(), self.frame);
        let counted = disk.spills_counted();
        // Counted as on its way, as an upload's abort is, so that a process
        // that stops waits for it for as long as it gives those.
        let leaving = Aborts::process().begun();
        disk.run_unwaited(move || {
            let _leaving = leaving;
            if let Some(told) = told {
                told.unlinks();
            }
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                // Already gone, as by a cache emptied under it.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => tracing::warn!(?path, %error, "a spill could not be removed"),
            }
            counted.gone(frame);
        });
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
    /// A spill of a frame of `frame` bytes begun under a name drawn at
    /// random, if there is room for it, and locked in the step that makes
    /// it: `None` if the merge is stopped first, and
    /// [`StorageError::NoRoom`] if there is too little room.
    ///
    /// Between the file made and the lock taken a cache opening beside this
    /// can find it unheld and take it. Its lock comes first, or this one's
    /// does: the one that finds the other's fails, or finds the name gone
    /// by the time it renames, which costs the merge or its copy and nothing
    /// else, and a name drawn at random is no name an open is looking for.
    pub(crate) fn begin(self, frame: u64) -> Result<Option<Spilling>> {
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
        let of_disk = disk.clone();
        let begun = {
            move || {
                std::fs::create_dir_all(&dir)?;
                let free = room_on(&dir, told.as_deref());
                let wanted = frame.saturating_add(room_spared(frame));
                if free < wanted {
                    return Ok(Err((free, wanted)));
                }
                let file = File::options()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                of_disk.spills_counted().began(frame);
                let name = Name {
                    path: Some(path),
                    disk: Some(of_disk),
                    frame,
                };
                match file.try_lock() {
                    Ok(()) => Ok(Ok((file, name))),
                    // A file system with no locks: spills left behind there
                    // go as anything left part written does, once stale.
                    Err(std::fs::TryLockError::Error(_)) => Ok(Ok((file, name))),
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
        let (file, name) = match begun {
            Ok(Ok(begun)) => begun,
            Ok(Err((free, wanted))) => {
                disk.spills_counted().skipped();
                return Err(StorageError::NoRoom { key, free, wanted });
            }
            Err(error) => {
                disk.spills_counted().failed();
                return Err(unwritten(&key, error));
            }
        };
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
    /// `error`, the disk's, counted.
    fn failed(&self, error: std::io::Error) -> StorageError {
        self.disk.spills_counted().failed();
        unwritten(&self.key, error)
    }

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
        let mut piece = stepped.map_err(|error| self.failed(error))?;
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
        let map = mapped.map_err(|error| self.failed(error))?;
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
    pub fn pieces(
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
                let counted = disk.spills_counted();
                let read = disk.run(end - at, move || {
                    if let Some(told) = told {
                        told.read();
                    }
                    Ok(Bytes::copy_from_slice(&spill.bytes()[at..end]))
                });
                let piece = read.await.map_err(|error| {
                    counted.failed();
                    unwritten(&key, error)
                })?;
                Ok(Some((piece, end)))
            }
        });
        pieces.boxed()
    }

    /// The file's name from now on is `path`, whatever was there: it is
    /// no spill any more, and is left there as this goes. Returns the file
    /// as it lay, to tell it from any other put there since.
    pub(crate) fn renamed_to(&self, path: &Path) -> std::io::Result<Seen> {
        let mut name = named(&self.name);
        let from = name.path.as_ref().ok_or(std::io::ErrorKind::NotFound)?;
        let was = seen(from)?;
        std::fs::rename(from, path)?;
        name.given_up();
        Ok(was)
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

/// How much room there is where `dir` lies: as told, if a test says so, and
/// as much as anything could want where a disk cannot say, so that it says
/// so itself as it is written to.
fn room_on(dir: &Path, told: Option<&SpillsTold>) -> u64 {
    match told.map(|told| told.free.load(Ordering::SeqCst)) {
        None | Some(SpillsTold::ASK_THE_DISK) => disk_room(dir).map_or(u64::MAX, |(free, _)| free),
        Some(free) => free,
    }
}

/// What is spilled to a disk as it goes, shared by all that lie on it.
#[derive(Debug, Default)]
pub(crate) struct SpillsCounted {
    bytes: AtomicU64,
    skipped: AtomicU64,
    failed: AtomicU64,
}

impl SpillsCounted {
    fn began(&self, frame: u64) {
        self.bytes.fetch_add(frame, Ordering::Relaxed);
    }

    fn gone(&self, frame: u64) {
        self.bytes.fetch_sub(frame, Ordering::Relaxed);
    }

    fn skipped(&self) {
        self.skipped.fetch_add(1, Ordering::Relaxed);
    }

    fn failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    fn figures(&self) -> SpillFigures {
        SpillFigures {
            bytes: self.bytes.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
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
            let mut spilling = asked.begin(64).unwrap().unwrap();
            let mut words = vec![0; 16];
            assert!(spilling.written(&mut words).unwrap());
            spilling.sealed(1).unwrap().unwrap()
        });
        Arc::new(made.await.unwrap())
    }

    fn names(dir: &Path) -> usize {
        std::fs::read_dir(dir).unwrap().count()
    }

    /// Wait until a name let go is gone: nothing waits for that but this.
    fn no_names_left(dir: &Path) {
        let begun = std::time::Instant::now();
        while names(dir) > 0 {
            assert!(begun.elapsed() < Duration::from_secs(30), "a name stays");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// As much as a file's parts and pieces hold at once, by walking them
    /// as `ObjectBlobs::put_parts` cuts them: each part sent holds every
    /// piece it is cut from, and the one that came and showed it was full.
    fn most_held_by_parts(part: usize, pieces: usize) -> usize {
        let (mut first, mut holding, mut most) = (0, 0, 0);
        for piece in 0..pieces {
            let mut came = SPILLED_AT_ONCE;
            while holding + came > part {
                came -= part - holding;
                most = usize::max(most, (piece + 1 - first) * SPILLED_AT_ONCE);
                first = piece;
                holding = 0;
            }
            holding += came;
        }
        most
    }

    /// What is held as a frame goes up is reckoned at no less than it comes
    /// to, whether or not a part is a whole number of pieces.
    #[test]
    fn what_goes_up_is_reckoned_at_no_less_than_its_parts_hold() {
        const MIB: usize = 1024 * 1024;
        for part in [4, 5, 6, 7, 8, 9, 15, 16, 17, 64] {
            let part = part * MIB;
            let most = most_held_by_parts(part, 400);
            assert!(most <= held_of_parts(part), "{part}: {most}");
        }
    }

    /// A spill leaves as much again as it takes, as another merge's beside
    /// it takes, and no less than a backlog of flushes.
    #[test]
    fn a_spill_spares_as_much_as_it_takes_and_a_gibibyte_at_the_least() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(room_spared(0), GIB);
        assert_eq!(room_spared(GIB / 2), GIB);
        assert_eq!(room_spared(3 * GIB), 3 * GIB);
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
        no_names_left(dir.path());
    }

    /// A step given up on, and a stop, leave nothing, and whatever is
    /// written to a file that has lost its name is nothing to anyone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_spill_nobody_goes_on_with_leaves_no_name() {
        let dir = tempfile::tempdir().unwrap();
        let spills = Spills::in_dir(dir.path().to_path_buf(), Disk::shared().clone());
        let asked = spills.asked(Handle::current(), Stopping::never(), "k".into());
        let begun = tokio::task::spawn_blocking(move || asked.begin(64).unwrap().unwrap());
        let spilling = begun.await.unwrap();
        assert_eq!(names(dir.path()), 1);
        drop(spilling);
        no_names_left(dir.path());
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
