//! A deterministic simulation of one partition under failure.
//!
//! Random histories interleave writes from a current writer and from stalled
//! ones, each a named request that inserts rows with vectors and may delete
//! one, or changes or deletes a row only if it is at a version a reader
//! gave or at one nothing is at, the same requests made again,
//! checkpoints, merges, restores alone and beside a flush, a merge or a
//! clean, cleaning up with and without grace, takeovers and follower
//! catch-ups, with faults in the store and the coordinator. After every
//! step the committed state is read through fault-free handles, as rows
//! with their versions and as a walk of pages, and checked against what
//! the coordinator really recorded, whatever the writers were told.
//!
//! A step takes an hour, and a wait as many more as it says, by three
//! clocks that go on together. The store dates each file by its own as the
//! file lands, and a writer cleaning up sets the node's against those
//! dates, so that cleaning up with a grace of some hours deletes what is
//! older and keeps what is not. The coordinator dates each commit by its
//! own, and by it commits pass out of the window a restore may reach back
//! and are forgotten. A history begins with the coordinator's some steps
//! either side of the store's, which nothing is to notice, and the node's
//! up to as many steps behind it as it may later run ahead. What a step
//! leaves to be done on a task of its own, as a writer deletes files it
//! never published, is done before the next begins.
//!
//! A delete may be given up by whoever asked for it and land later, as a
//! put may. Every head the coordinator recorded is kept with the rows
//! committed at it. After every step each of them a restore may still go
//! to by the coordinator's clock is seen to be listed by the coordinator,
//! asked by its position and by its time, and no other is; and each that
//! every clean there has been was to keep is seen to load as those rows. A
//! clean with nothing in its way must succeed, whatever cleans came before
//! it, and records the checkpoint it deletes below before it deletes
//! anything, never above what it keeps: see `World::cleaned`. The
//! store notes every delete as it is asked for, and every file as it first
//! landed, and no head is to name a file whose delete was asked for before
//! it was committed: see `World::never_named_again`. A clean with grace
//! asks for the delete of no file a head a restore may go to needs: see
//! `World::cleaned`.
//!
//! One pairing is not made: a restore to a state that a clean was not to
//! keep, while a delete of one of that state's files, given up, is yet to
//! land or may come to be. Only a clean with no grace, which no node makes,
//! keeps less than a restore may go back to; the restore would find the
//! file there still and put it back into use, and the delete would then
//! take it, by what the clean was asked and by no fault of the writer.
//!
//! The committed head's files may be copied out, as a backup copies them,
//! and the partition put back from the copy, as from a backup: copied in
//! under keys of its own, as DW-680 has it, and held to that rule. The way
//! a node made such a restore before DW-680, which breaks the rule every
//! time, is made only in the histories a test sets out. See
//! `World::restore_from`.

mod common;

use chrono::{DateTime, SubsecRound};
use std::collections::VecDeque;
use vellev_core::{RequestId, Version};
use vellev_storage::coordinator::{
    Cleaned, Clock, KeyHash, Point, Restore, RestorePoint, Restored, TenantKey,
};
use vellev_storage::format::{decode_head, Head as Stored};

use common::*;
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestRng, TestRunner};
use vellev_storage::{Batch, Embedded, Live, Merging, Outcome};

const COLLECTION: &str = "sim";

/// The id of the `n`th row written: spread over the id space, so that
/// id order is not write order.
fn row(n: u64) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(
        n.wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )))
}

fn counter() -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::MAX))
}

fn lost() -> StorageError {
    StorageError::Blob {
        key: "simulated".into(),
        source: "connection reset".into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PutFault {
    Refuse,
    LandThenFail,
    /// The writer sees an error and the file arrives later.
    Delay,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CommitFault {
    Refuse,
    LoseResponse,
}

/// A fault in what a clean asks the coordinator of what cleans recorded.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RecordFault {
    /// The next record is refused, and not made.
    Refuse,
    /// The next record is made, and its answer lost.
    LoseResponse,
    /// The next reading of the record fails.
    FailRead,
}

/// A dependency broken on purpose, to prove the checker notices.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Canary {
    /// The coordinator says yes to a commit it did not record.
    AcknowledgesUnrecordedCommits,
    /// The store accepts a segment and keeps nothing.
    DropsSegments,
    /// Cleaning up keeps nothing for its age, whatever grace it is asked
    /// to give.
    CleansWithNoGrace,
    /// The coordinator forgets every commit but the newest, whatever
    /// window it is asked to keep.
    ForgetsWithinTheWindow,
    /// A file whose delete was given up is written again once the delete
    /// has landed, as by a restore that fetches it back from a backup.
    WritesAgainAfterALateDelete,
    /// A restore is made to a state no clean was to keep while a delete
    /// given up is yet to land: the file is there still, the restore's
    /// checkpoint lists it, and the delete then takes it.
    RestoresUnderALateDelete,
    /// The store refuses every delete, and makes none of them.
    RefusesDeletes,
    /// The coordinator lets a restore go to a commit by its position for
    /// twice as long as the window it is asked to keep.
    ListsOutOfReach,
    /// The coordinator forgets what it should and says it forgot one more.
    MiscountsTheForgotten,
    /// The store notes a delete only once it makes it, not when it is
    /// asked for it: the checker is then blind to a head that names a file
    /// whose delete was given up and is yet to land, and must be shown to
    /// be, so that what it notes cannot be made less unseen.
    NotesDeletesOnlyMade,
}

#[derive(Debug, Default)]
struct Faults {
    puts: VecDeque<PutFault>,
    commits: VecDeque<CommitFault>,
    failing_gets: usize,
    /// A file every read of fails as a store fails a request, whether it
    /// is there or not: a fault on one file, not a missing one.
    failing_key: Option<String>,
    /// How many deletes are answered before one is held, if one is to be.
    held_delete: Option<usize>,
    /// What is to go wrong with the record of what cleans deleted below.
    records: VecDeque<RecordFault>,
}

/// A store that fails on command and records every file it was ever given,
/// deleted since or not, so a second write to one key is noticed. Whoever
/// asks anything of it waits a moment for the answer, as of any store, so
/// that what is done at once takes turns.
#[derive(Debug)]
struct SimBlobs {
    clean: Arc<dyn Blobs>,
    faults: Arc<Mutex<Faults>>,
    canary: Option<Canary>,
    /// The clock every file is dated by as it lands.
    clock: Arc<ManualClock>,
    written: Mutex<BTreeSet<String>>,
    /// How many files had ever landed when the store was last asked to
    /// list them, which only a clean asks.
    listed: AtomicUsize,
    delayed: Mutex<Vec<(String, Bytes)>>,
    rewritten: Mutex<Vec<String>>,
    /// The deletes given up by whoever asked for them, which the store has
    /// yet to make.
    held: Mutex<Vec<String>>,
    /// How many of those were held for want of an answer alone: the
    /// request was dropped while the store had it; and how many deletes
    /// the store answered as lost.
    unanswered: AtomicUsize,
    refused: AtomicUsize,
    /// How many deletes the store has been asked for, and how many it has
    /// made.
    asked: AtomicUsize,
    deleted: AtomicUsize,
    /// How many deletes the store has been asked for and has neither
    /// answered nor seen dropped.
    deleting: AtomicUsize,
    /// When each file landed, and how many had before it.
    landed: Mutex<BTreeMap<String, (DateTime<Utc>, usize)>>,
    /// Every file that landed, in the order it did, again if it did again.
    landings: Mutex<Vec<String>>,
    /// Every file that ever landed, as it first did, so that a head can be
    /// read once its files are gone.
    shadow: Mutex<BTreeMap<String, Bytes>>,
    /// Every file that was read, in the order it was.
    read: Mutex<Vec<String>>,
    /// Every key a delete has been asked for, in the order it was: sent,
    /// whatever became of it. The coordinator notes how many there were
    /// with each head it records.
    sent: Arc<Mutex<Vec<String>>>,
    /// The position the coordinator recorded a clean deleted below, as the
    /// coordinator keeps it; and for each key a delete was asked for, that
    /// position as it was when it was last asked.
    record: Arc<Mutex<Option<u64>>>,
    asked_under: Mutex<BTreeMap<String, Option<u64>>>,
    /// Whether a checkpoint's next uploads are to wait, and where:
    /// `waiting` is told when they do, and they go on once `resumed` is.
    pausing: Mutex<Option<Pause>>,
    waiting: tokio::sync::Notify,
    resumed: tokio::sync::Notify,
}

/// Where the store keeps a checkpoint's uploads waiting. A checkpoint
/// uploads a segment's vectors, then its index if it has one, then the
/// segment itself.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Pause {
    /// Before the first of them.
    Before,
    /// Once the segment, the last of them, has landed or failed. Any of
    /// them that fails ends the checkpoint there, and nothing waits.
    After,
}

/// A delete the store has been asked for. Dropped unanswered, it is held:
/// a request given up may be made all the same.
struct Asked<'a> {
    blobs: &'a SimBlobs,
    key: Option<String>,
    refused: bool,
}

impl Drop for Asked<'_> {
    fn drop(&mut self) {
        self.blobs.deleting.fetch_sub(1, Ordering::SeqCst);
        if let Some(key) = self.key.take() {
            self.blobs.held.lock().unwrap().push(key);
            let unanswered = usize::from(!self.refused);
            self.blobs
                .unanswered
                .fetch_add(unanswered, Ordering::SeqCst);
        }
    }
}

/// The position in the name of a partition's file.
fn position(key: &str) -> Option<u64> {
    key.rsplit('/').next()?.get(..20)?.parse().ok()
}

/// How long a step is, in the age of a file.
const STEP: Duration = Duration::from_secs(60 * 60);
/// How many steps back a restore may go.
const WINDOW: usize = 6;
/// How many steps old a file is before a clean with grace deletes it: the
/// window and some more, as a node cleans.
const GRACE: usize = WINDOW + 2;

impl SimBlobs {
    async fn land(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if !self.written.lock().unwrap().insert(key.to_string()) {
            self.rewritten.lock().unwrap().push(key.to_string());
        }
        {
            let mut landed = self.landed.lock().unwrap();
            let before = landed.len();
            let now = self.clock.now();
            // A file put again is dated anew, as a store dates it, and
            // keeps its place among those of its name.
            (landed.entry(key.to_string()))
                .and_modify(|(at, _)| *at = now)
                .or_insert((now, before));
        }
        self.landings.lock().unwrap().push(key.to_string());
        (self.shadow.lock().unwrap())
            .entry(key.to_string())
            .or_insert_with(|| bytes.clone());
        match self.canary {
            Some(Canary::DropsSegments) if key.contains("/segment/") => Ok(()),
            _ => self.clean.put(key, bytes).await,
        }
    }

    /// How many deletes the store has been asked for, and how many made.
    fn deletes(&self) -> (usize, usize) {
        let asked = self.asked.load(Ordering::SeqCst);
        (asked, self.deleted.load(Ordering::SeqCst))
    }

    async fn deliver_delayed(&self) {
        let delayed = std::mem::take(&mut *self.delayed.lock().unwrap());
        for (key, bytes) in delayed {
            self.land(&key, bytes).await.unwrap();
        }
    }

    async fn remove(&self, key: &str) -> vellev_storage::Result<()> {
        self.deleted.fetch_add(1, Ordering::SeqCst);
        if self.canary == Some(Canary::NotesDeletesOnlyMade) {
            self.sent.lock().unwrap().push(key.to_string());
        }
        self.clean.delete(key).await
    }

    /// The files there are, and the lowest position of any that landed at
    /// or after `from`.
    async fn files(&self, from: DateTime<Utc>) -> (BTreeSet<String>, Option<u64>) {
        let files = self.clean.list("").await.unwrap();
        let landed = self.landed.lock().unwrap();
        let young = files
            .iter()
            .filter(|file| landed.get(&file.key).is_some_and(|(at, _)| *at >= from))
            .filter_map(|file| position(&file.key))
            .min();
        (files.into_iter().map(|file| file.key).collect(), young)
    }

    /// Wait until whatever was asked of the store on a task of its own has
    /// been answered: a writer deletes the files it uploaded for a commit
    /// it never tried on one, and the store answers a delete only as the
    /// clock moves.
    async fn settle(&self) {
        loop {
            tokio::task::yield_now().await;
            if self.deleting.load(Ordering::SeqCst) == 0 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Make the deletes that were given up, and say how many there were
    /// and how many of them took a file that was there still.
    async fn deliver_held_deletes(&self) -> (usize, usize) {
        let held = std::mem::take(&mut *self.held.lock().unwrap());
        let mut took = 0;
        for key in &held {
            let there = self.clean.get(key).await.ok();
            took += usize::from(there.is_some());
            self.remove(key).await.unwrap();
            if let (Some(Canary::WritesAgainAfterALateDelete), Some(bytes)) = (self.canary, there) {
                self.land(key, bytes).await.unwrap();
            }
        }
        (held.len(), took)
    }
}

#[async_trait]
impl Blobs for SimBlobs {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        tokio::task::yield_now().await;
        let head = key.ends_with(".log") || key.ends_with(".manifest");
        let pause = {
            let mut pausing = self.pausing.lock().unwrap();
            match *pausing {
                Some(Pause::Before) if !head => pausing.take(),
                Some(Pause::After) if key.contains("/segment/") => pausing.take(),
                _ => None,
            }
        };
        if pause == Some(Pause::Before) {
            self.waiting.notify_one();
            self.resumed.notified().await;
        }
        let fault = self.faults.lock().unwrap().puts.pop_front();
        let put = match fault {
            None => self.land(key, bytes).await,
            Some(PutFault::Refuse) => Err(lost()),
            Some(PutFault::LandThenFail) => self.land(key, bytes).await.and(Err(lost())),
            Some(PutFault::Delay) => {
                self.delayed.lock().unwrap().push((key.to_string(), bytes));
                Err(lost())
            }
        };
        if pause == Some(Pause::After) {
            self.waiting.notify_one();
            self.resumed.notified().await;
        }
        put
    }

    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        tokio::task::yield_now().await;
        let failing = {
            let mut faults = self.faults.lock().unwrap();
            let failing = faults.failing_gets > 0;
            faults.failing_gets = faults.failing_gets.saturating_sub(1);
            failing
        };
        let failing_key = self.faults.lock().unwrap().failing_key.as_deref() == Some(key);
        if failing || failing_key {
            return Err(lost());
        }
        self.read.lock().unwrap().push(key.to_string());
        self.clean.get(key).await
    }

    /// Put as [`Self::put`] puts, with its faults, once every part has
    /// come: what lands is noted as any file that lands is.
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        use futures::TryStreamExt;
        let parts: Vec<Bytes> = parts.try_collect().await?;
        self.put(key, Bytes::from(parts.concat())).await
    }

    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        tokio::task::yield_now().await;
        let mut listed = self.clean.list(prefix).await?;
        let landed = self.landed.lock().unwrap();
        self.listed.store(landed.len(), Ordering::SeqCst);
        for file in &mut listed {
            file.modified = landed[&file.key].0;
        }
        // By name, as a store lists, but for the files of one kind at one
        // position and of one generation, whose names differ only in what
        // their writer drew at random: those as they landed, so that which
        // of two a delete to be held falls on is not left to the draw. A
        // name is the position, a dash and the generation, twenty digits
        // each, and then the writer and what it drew.
        listed.sort_by_key(|file| {
            let named = file.key.rfind('/').map_or(0, |at| at + 1) + 41;
            let stem = file.key[..named.min(file.key.len())].to_string();
            (stem, landed[&file.key].1)
        });
        Ok(listed)
    }
    /// A delete is answered only once nothing else has anything to do, on
    /// the simulation's clock, which moves only then: whatever may run
    /// while files are being deleted does, to its end. One the store is to
    /// hold is answered as lost, and made when the held deletes are.
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.asked.fetch_add(1, Ordering::SeqCst);
        let record = *self.record.lock().unwrap();
        (self.asked_under.lock().unwrap()).insert(key.to_string(), record);
        if self.canary != Some(Canary::NotesDeletesOnlyMade) {
            self.sent.lock().unwrap().push(key.to_string());
        }
        self.deleting.fetch_add(1, Ordering::SeqCst);
        let mut asked = Asked {
            blobs: self,
            key: Some(key.to_string()),
            refused: false,
        };
        tokio::time::sleep(Duration::from_millis(1)).await;
        let held = {
            let mut faults = self.faults.lock().unwrap();
            let held = faults.held_delete == Some(0);
            faults.held_delete = faults.held_delete.and_then(|after| after.checked_sub(1));
            held
        };
        if held {
            asked.refused = true;
            self.refused.fetch_add(1, Ordering::SeqCst);
            return Err(lost());
        }
        asked.key = None;
        if self.canary == Some(Canary::RefusesDeletes) {
            self.refused.fetch_add(1, Ordering::SeqCst);
            return Err(lost());
        }
        self.remove(key).await
    }
}

/// A head the coordinator recorded, and when by its clock.
#[derive(Debug, Clone, PartialEq)]
struct Head {
    position: LogPosition,
    key: ManifestRef,
    at: DateTime<Utc>,
    /// Whether the commit that made it was a restore.
    restore: bool,
    /// How many deletes the store had been asked for when it was.
    sent: usize,
}

/// A coordinator that fails on command, counts the commits it recorded,
/// keeps every head they made, and notes any commit it recorded for a
/// lease that was no longer the current one.
#[derive(Debug)]
struct SimCoordinator {
    clean: Arc<MemoryCoordinator>,
    clock: Arc<ManualClock>,
    /// Every head recorded, oldest first. A commit made again, which the
    /// coordinator answers as it did, makes none.
    heads: Mutex<Vec<Head>>,
    faults: Arc<Mutex<Faults>>,
    canary: Option<Canary>,
    recorded: AtomicUsize,
    /// How many of the commits recorded made a restore.
    restores: AtomicUsize,
    unfenced: Mutex<Vec<String>>,
    /// The deletes the store has been asked for.
    sent: Arc<Mutex<Vec<String>>>,
    /// The record of what cleans deleted below, as the store sees it when
    /// each delete is asked for.
    record: Arc<Mutex<Option<u64>>>,
    /// Every record a clean asked for, and whether it was made.
    records: Mutex<Vec<(Cleaned, bool)>>,
    /// What the coordinator answers a reading of the record with instead
    /// of what it holds, if anything: one put back from earlier, or one of
    /// no checkpoint of this chain.
    answering: Mutex<Option<Option<Cleaned>>>,
}

impl SimCoordinator {
    /// How many of the heads recorded are logs, as every write's is and no
    /// checkpoint's.
    fn logs(&self) -> usize {
        let heads = self.heads.lock().unwrap();
        let logs = heads.iter().filter(|head| head.key.0.ends_with(".log"));
        logs.count()
    }
}

#[async_trait]
impl Coordinator for SimCoordinator {
    async fn register(&self, key: &PartitionKey) -> Result<(), CoordinatorError> {
        self.clean.register(key).await
    }
    async fn remove(&self, key: &PartitionKey) -> Result<bool, CoordinatorError> {
        self.clean.remove(key).await
    }
    async fn acquire(
        &self,
        key: &PartitionKey,
        holder: &NodeId,
        ttl: Duration,
    ) -> Result<Acquired, CoordinatorError> {
        self.clean.acquire(key, holder, ttl).await
    }
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease, CoordinatorError> {
        self.clean.renew(lease, ttl).await
    }
    async fn release(&self, lease: &Lease) -> Result<(), CoordinatorError> {
        self.clean.release(lease).await
    }
    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed, CoordinatorError> {
        let reset = || CoordinatorError::backend(std::io::Error::other("connection reset"));
        let fault = self.faults.lock().unwrap().commits.pop_front();
        if fault == Some(CommitFault::Refuse) {
            return Err(reset());
        }
        let outcome = self.clean.commit(lease, commit.clone()).await;
        match &outcome {
            Ok(committed) => {
                self.recorded.fetch_add(1, Ordering::SeqCst);
                let restores = usize::from(commit.restore.is_some());
                self.restores.fetch_add(restores, Ordering::SeqCst);
                let mut heads = self.heads.lock().unwrap();
                let newest = heads.last().map(|head: &Head| head.position);
                if let Some(key) = commit
                    .manifest
                    .clone()
                    .filter(|_| newest.is_none_or(|newest| commit.log_position > newest))
                {
                    heads.push(Head {
                        position: commit.log_position,
                        key,
                        at: self.clock.now().trunc_subsecs(6),
                        restore: commit.restore.is_some(),
                        sent: self.sent.lock().unwrap().len(),
                    });
                }
                if committed.generation != lease.generation {
                    let stale = format!("{:?} at {:?}", lease.generation, commit.log_position);
                    self.unfenced.lock().unwrap().push(stale);
                }
            }
            Err(_) if self.canary == Some(Canary::AcknowledgesUnrecordedCommits) => {
                return Ok(Committed {
                    generation: lease.generation,
                    incarnation: self
                        .clean
                        .committed(&lease.key)
                        .await?
                        .map_or(Incarnation(0), |committed| committed.incarnation),
                    log_position: commit.log_position,
                    manifest: commit.manifest,
                });
            }
            Err(_) => {}
        }
        match fault {
            Some(CommitFault::LoseResponse) => Err(reset()),
            _ => outcome,
        }
    }
    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>, CoordinatorError> {
        self.clean.committed(key).await
    }
    async fn cleaned(
        &self,
        key: &PartitionKey,
    ) -> Result<Option<vellev_storage::coordinator::Cleaned>, CoordinatorError> {
        let reset = || CoordinatorError::backend(std::io::Error::other("connection reset"));
        let failing = {
            let mut faults = self.faults.lock().unwrap();
            let failing = faults.records.front() == Some(&RecordFault::FailRead);
            if failing {
                faults.records.pop_front();
            }
            failing
        };
        if failing {
            return Err(reset());
        }
        let answering = self.answering.lock().unwrap().clone();
        if let Some(answering) = answering {
            return Ok(answering);
        }
        self.clean.cleaned(key).await
    }
    async fn clean_below(
        &self,
        lease: &Lease,
        base: &vellev_storage::coordinator::ManifestRef,
        position: vellev_storage::coordinator::LogPosition,
    ) -> Result<vellev_storage::coordinator::Cleaned, CoordinatorError> {
        let reset = || CoordinatorError::backend(std::io::Error::other("connection reset"));
        let fault = {
            let mut faults = self.faults.lock().unwrap();
            match faults.records.front() {
                Some(RecordFault::Refuse | RecordFault::LoseResponse) => faults.records.pop_front(),
                _ => None,
            }
        };
        let asked = Cleaned {
            base: base.clone(),
            position,
        };
        if fault == Some(RecordFault::Refuse) {
            self.records.lock().unwrap().push((asked, false));
            return Err(reset());
        }
        let made = self.clean.clean_below(lease, base, position).await;
        self.records.lock().unwrap().push((asked, made.is_ok()));
        if let Ok(made) = &made {
            *self.record.lock().unwrap() = Some(made.position.0);
        }
        match fault {
            Some(_) => Err(reset()),
            None => made,
        }
    }
    async fn partitions(
        &self,
        after: Option<&PartitionKey>,
        limit: usize,
    ) -> Result<Vec<PartitionKey>, CoordinatorError> {
        self.clean.partitions(after, limit).await
    }
    async fn ping(&self) -> Result<(), CoordinatorError> {
        self.clean.ping().await
    }
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<u16, CoordinatorError> {
        self.clean.announce(node, versions).await
    }
    async fn depart(&self, node: &NodeId) -> Result<(), CoordinatorError> {
        self.clean.depart(node).await
    }
    async fn oldest(&self) -> Result<Option<Versions>, CoordinatorError> {
        self.clean.oldest().await
    }
    async fn format(&self) -> Result<u16, CoordinatorError> {
        self.clean.format().await
    }
    async fn raise_format(&self, format: u16) -> Result<u16, CoordinatorError> {
        self.clean.raise_format(format).await
    }

    async fn issue_key(
        &self,
        tenant: &TenantId,
        id: uuid::Uuid,
        hash: &KeyHash,
    ) -> Result<Option<TenantKey>, CoordinatorError> {
        self.clean.issue_key(tenant, id, hash).await
    }

    async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>, CoordinatorError> {
        self.clean.keys(tenant).await
    }

    async fn revoke_key(
        &self,
        tenant: &TenantId,
        id: uuid::Uuid,
    ) -> Result<bool, CoordinatorError> {
        self.clean.revoke_key(tenant, id).await
    }

    async fn key_tenant(&self, hash: &KeyHash) -> Result<Option<TenantId>, CoordinatorError> {
        self.clean.key_tenant(hash).await
    }

    async fn restore_point(
        &self,
        key: &PartitionKey,
        point: Point,
        window: Duration,
    ) -> Result<Option<RestorePoint>, CoordinatorError> {
        let window = match (self.canary, point) {
            (Some(Canary::ListsOutOfReach), Point::Position(_)) => window * 2,
            _ => window,
        };
        self.clean.restore_point(key, point, window).await
    }

    async fn restore_points(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<Option<(RestorePoint, RestorePoint)>, CoordinatorError> {
        self.clean.restore_points(key, window).await
    }

    async fn forget_history(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<u64, CoordinatorError> {
        let window = match self.canary {
            Some(Canary::ForgetsWithinTheWindow) => Duration::ZERO,
            _ => window,
        };
        let forgotten = self.clean.forget_history(key, window).await?;
        Ok(match self.canary {
            Some(Canary::MiscountsTheForgotten) => forgotten + 1,
            _ => forgotten,
        })
    }

    async fn restored(
        &self,
        key: &PartitionKey,
        request: RequestId,
    ) -> Result<Option<Restored>, CoordinatorError> {
        self.clean.restored(key, request).await
    }

    async fn forget_removed(
        &self,
        key: &PartitionKey,
        after: Option<Forgotten>,
        limit: usize,
    ) -> Result<Option<Forgotten>, CoordinatorError> {
        self.clean.forget_removed(key, after, limit).await
    }

    async fn forget_restores(&self) -> Result<u64, CoordinatorError> {
        self.clean.forget_restores().await
    }

    async fn restore_window(
        &self,
        tenant: &TenantId,
    ) -> Result<Option<Duration>, CoordinatorError> {
        self.clean.restore_window(tenant).await
    }

    async fn purges(
        &self,
        due: Duration,
        lately: Duration,
    ) -> Result<vellev_storage::coordinator::Purges, CoordinatorError> {
        self.clean.purges(due, lately).await
    }

    async fn identity(
        &self,
        tenant: &TenantId,
    ) -> Result<Option<vellev_storage::coordinator::Identity>, CoordinatorError> {
        self.clean.identity(tenant).await
    }

    async fn cluster(&self) -> Result<vellev_storage::coordinator::ClusterId, CoordinatorError> {
        self.clean.cluster().await
    }

    async fn adopt_on_commit(
        &self,
        key: &PartitionKey,
        generation: vellev_storage::coordinator::Generation,
        adoption: &vellev_storage::coordinator::Adoption,
    ) -> Result<bool, CoordinatorError> {
        self.clean.adopt_on_commit(key, generation, adoption).await
    }

    async fn adopt_no_more(
        &self,
        key: &PartitionKey,
        generation: vellev_storage::coordinator::Generation,
    ) -> Result<(), CoordinatorError> {
        self.clean.adopt_no_more(key, generation).await
    }

    async fn claim_purge(
        &self,
        tenant: &TenantId,
        due: Duration,
        hold: Duration,
    ) -> Result<Option<vellev_storage::coordinator::PurgeClaim>, CoordinatorError> {
        self.clean.claim_purge(tenant, due, hold).await
    }

    async fn renew_purge(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.clean.renew_purge(tenant, claim, hold).await
    }

    async fn release_purge(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
    ) -> Result<(), CoordinatorError> {
        self.clean.release_purge(tenant, claim).await
    }

    async fn look_overs(
        &self,
        lately: Duration,
        every: Duration,
        limit: usize,
    ) -> Result<Vec<vellev_storage::coordinator::Removed>, CoordinatorError> {
        self.clean.look_overs(lately, every, limit).await
    }

    async fn purged(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        forget_after: Duration,
    ) -> Result<(), CoordinatorError> {
        self.clean.purged(tenant, claim, forget_after).await
    }

    async fn reopen_purge(&self, tenant: &TenantId) -> Result<bool, CoordinatorError> {
        self.clean.reopen_purge(tenant).await
    }

    async fn claim_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.clean.claim_backup(tenant, claim, due, hold).await
    }

    async fn renew_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        hold: Duration,
    ) -> Result<bool, CoordinatorError> {
        self.clean.renew_backup(tenant, claim, hold).await
    }

    async fn release_backup(
        &self,
        tenant: &TenantId,
        claim: uuid::Uuid,
        made: bool,
    ) -> Result<(), CoordinatorError> {
        self.clean.release_backup(tenant, claim, made).await
    }

    async fn set_restore_window(
        &self,
        tenant: &TenantId,
        window: Option<Duration>,
    ) -> Result<bool, CoordinatorError> {
        self.clean.set_restore_window(tenant, window).await
    }
}

/// Which writer acts: the one the cluster would use, or any writer there
/// has ever been, which may be stalled, fenced or closed and not know it.
#[derive(Debug, Clone, Copy)]
enum Who {
    Current,
    Any(usize),
}

/// What goes on beside a restore.
#[derive(Debug, Clone, Copy)]
enum Beside {
    Flush,
    Merge,
    Clean,
}

#[derive(Debug, Clone)]
enum Action {
    /// A request of its own: insert so many rows and count them, and with
    /// `delete` delete one of the rows there are.
    Write {
        who: Who,
        rows: u64,
        delete: Option<usize>,
    },
    /// A write as [`Action::Write`] makes, made while a checkpoint is
    /// being written: the checkpoint is built of the rows there were when
    /// it began, and replays the write over them. With `first`, the
    /// checkpoint's uploads are made, or one fails, before the write: an
    /// upload to fail fails the checkpoint's, and the write's log lands
    /// after its segment. Without, the write is made before any of them.
    WriteBesideFlush {
        who: Who,
        rows: u64,
        delete: Option<usize>,
        first: bool,
    },
    /// A request of its own that changes one of the rows there are, or
    /// deletes it, only if it is at the version a reader last gave for
    /// it or, when not `fresh`, at a version nothing is at.
    Conditional {
        who: Who,
        pick: usize,
        kind: u8,
        fresh: bool,
    },
    /// Make one of the requests made before again, as a caller that never
    /// heard its answer would.
    Again {
        who: Who,
        pick: usize,
    },
    Flush {
        who: Who,
    },
    Merge {
        who: Who,
    },
    /// Delete what nothing needs. With `aged`, what was needed until some
    /// hours ago is kept, as a node cleans. Without, the current writer
    /// deletes whatever is not needed now; any other, which may be stale,
    /// never cleans without that grace, here as in a real cluster.
    Clean {
        who: Who,
        aged: bool,
    },
    /// Put the partition back as it stood so many commits ago, if that is
    /// within the window, with `beside` going on at the same time, begun
    /// just before it or just after.
    Restore {
        who: Who,
        back: usize,
        beside: Option<(Beside, bool)>,
    },
    /// The lease expires and another node takes it.
    Takeover,
    /// The node holding the lease restarts its writer on it.
    Reopen,
    FailUpload(PutFault),
    FailCommits(CommitFault, usize),
    FailReads(usize),
    /// The next record of what a clean deleted below, or the next reading
    /// of it, goes wrong as the fault says.
    FailRecord(RecordFault),
    DeliverDelayed,
    /// The store answers so many deletes and then holds one, which whoever
    /// asked for it is told was lost.
    DelayDelete(usize),
    /// The store makes the deletes that were given up.
    DeliverDelayedDeletes,
    /// So many hours pass with nothing done.
    Wait(usize),
    /// The node's clock runs so many steps ahead, as far as [`AHEAD`]
    /// steps ahead of the store's in all.
    NodeAhead(usize),
    /// The coordinator forgets the commits no restore may go back to, as a
    /// node has it do before it cleans.
    ForgetHistory,
    Follow,
    FollowStale(usize),
    /// The committed head's files are copied out of the store, with the
    /// rows committed at it, as a backup of the partition is.
    Snapshot,
    /// The partition is put back as one of the snapshots holds it, as a
    /// node puts a tenant back from a backup: see `World::restore_from`.
    RestoreFrom(usize),
    /// As [`Action::RestoreFrom`], the way a node did before DW-680. Made
    /// only in the histories a test sets out, never in random ones.
    RestoreFromAsToday(usize),
}

fn action() -> impl Strategy<Value = Action> {
    let who = || prop_oneof![4 => Just(Who::Current), 1 => any::<usize>().prop_map(Who::Any)];
    let put_fault = prop_oneof![
        Just(PutFault::Refuse),
        Just(PutFault::LandThenFail),
        Just(PutFault::Delay)
    ];
    let commit_fault = prop_oneof![Just(CommitFault::Refuse), Just(CommitFault::LoseResponse)];
    // Three failed tries close a writer, so most runs of faults stop short of that.
    let times = prop_oneof![3 => 1..3usize, 1 => 3..5usize];
    let delete = || prop_oneof![2 => Just(None), 1 => any::<usize>().prop_map(Some)];
    let beside = prop_oneof![
        3 => Just(None),
        1 => any::<bool>().prop_map(|first| Some((Beside::Flush, first))),
        1 => any::<bool>().prop_map(|first| Some((Beside::Merge, first))),
        2 => any::<bool>().prop_map(|first| Some((Beside::Clean, first))),
    ];
    let write = |(who, rows, delete)| Action::Write { who, rows, delete };
    let beside_flush = |(who, rows, delete, first)| Action::WriteBesideFlush {
        who,
        rows,
        delete,
        first,
    };
    let conditional = |(who, pick, kind, fresh)| Action::Conditional {
        who,
        pick,
        kind,
        fresh,
    };
    let restore = |(who, back, beside)| Action::Restore { who, back, beside };
    prop_oneof![
        14 => (who(), 1..4u64, delete()).prop_map(write),
        4 => (who(), 1..4u64, delete(), any::<bool>()).prop_map(beside_flush),
        6 => (who(), any::<usize>(), 0..4u8, prop_oneof![3 => Just(true), 1 => Just(false)])
            .prop_map(conditional),
        2 => (who(), any::<usize>()).prop_map(|(who, pick)| Action::Again { who, pick }),
        4 => who().prop_map(|who| Action::Flush { who }),
        3 => who().prop_map(|who| Action::Merge { who }),
        3 => (who(), any::<bool>()).prop_map(|(who, aged)| Action::Clean { who, aged }),
        3 => (who(), 0..8usize, beside).prop_map(restore),
        2 => Just(Action::Takeover),
        3 => Just(Action::Reopen),
        2 => put_fault.prop_map(Action::FailUpload),
        2 => (commit_fault, times).prop_map(|(fault, times)| Action::FailCommits(fault, times)),
        1 => (1..4usize).prop_map(Action::FailReads),
        1 => prop_oneof![
            Just(RecordFault::Refuse),
            Just(RecordFault::LoseResponse),
            Just(RecordFault::FailRead)
        ]
        .prop_map(Action::FailRecord),
        1 => Just(Action::DeliverDelayed),
        1 => (0..6usize).prop_map(Action::DelayDelete),
        1 => Just(Action::DeliverDelayedDeletes),
        1 => (1..GRACE).prop_map(Action::Wait),
        1 => (1..=AHEAD as usize).prop_map(Action::NodeAhead),
        2 => Just(Action::ForgetHistory),
        3 => Just(Action::Follow),
        1 => any::<usize>().prop_map(Action::FollowStale),
        1 => Just(Action::Snapshot),
        2 => any::<usize>().prop_map(Action::RestoreFrom),
    ]
}

/// What a request does to the rows if it is made.
#[derive(Debug, Clone)]
enum Effect {
    /// Inserts `rows` and deletes `deleted`.
    Write {
        rows: Vec<u64>,
        deleted: Option<ObjectId>,
    },
    /// Changes the row `id`, only if it is at `version`.
    Conditional {
        id: ObjectId,
        version: Version,
        change: Change,
    },
}

/// What a conditional request does to its row: the whole of what the row
/// becomes is known when the request is made, since it is made only on
/// the content the version names.
#[derive(Debug, Clone, PartialEq)]
enum Change {
    /// A patch that sets `n`, which may be what it is already.
    Patch(i64),
    /// A replace that keeps `n` and gives this vector.
    Vector(Vec<f32>),
    Delete,
}

/// One request: what it does, as it was asked, what its writer was told,
/// and whether the coordinator recorded its answer or it was refused,
/// either of which is its answer for good.
#[derive(Debug)]
struct Attempt {
    batch: Batch,
    effect: Effect,
    acknowledged: bool,
    recorded: bool,
    refused: bool,
}

/// A row as a checker compares it: `n`, the vector, and the version.
type Row = (i64, Vec<f32>, Version);

/// Rows as a checker compares them, by id.
type Rows = BTreeMap<ObjectId, Row>;

/// The row of `n` and `vector`, with the version such content has; a
/// row with no vector has none.
fn value(n: i64, vector: Vec<f32>) -> Row {
    let properties = BTreeMap::from([("n".to_string(), Value::Int(n))]);
    let version = Version::of(&properties, (!vector.is_empty()).then_some(&vector[..]));
    (n, vector, version)
}

/// A partition's rows and the position they are at.
type Snapshot = (LogPosition, Rows);

fn n_of(found: &Live<'_>) -> i64 {
    match found.object.properties.get("n") {
        Some(Value::Int(n)) => *n,
        other => panic!("{other:?}"),
    }
}

fn snapshot(state: &State) -> Snapshot {
    let rows = state
        .scan(&name(COLLECTION))
        .map(|found| {
            let vector = found.vector.map(<[f32]>::to_vec).unwrap_or_default();
            (found.id(), (n_of(&found), vector, found.version()))
        })
        .collect();
    (state.position(), rows)
}

/// The rows as a walk of pages lists them, `page` at a time from the
/// start, with `filter`, which must be every row that passes in id order:
/// what `scan` finds, which the snapshot is checked by.
fn walked(state: &State, page: usize, filter: Option<&Filter>, step: usize) -> Violation {
    // A copy from before the collection was made has nothing to page.
    if state.collection(&name(COLLECTION)).is_none() {
        return Ok(());
    }
    let passes = |n: i64| match filter {
        None => true,
        Some(Filter::Gte { value, .. }) => n >= value.as_i64().unwrap(),
        Some(Filter::Eq { value, .. }) => n == value.as_i64().unwrap(),
        other => panic!("{other:?}"),
    };
    let expected: Vec<(ObjectId, i64)> = state
        .scan(&name(COLLECTION))
        .map(|found| (found.id(), n_of(&found)))
        .filter(|(_, n)| passes(*n))
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect();
    let mut listed = Vec::new();
    let mut after = None;
    loop {
        let found = state
            .list(&name(COLLECTION), filter, after, page)
            .map_err(|error| format!("a page could not be listed: {error}"))?;
        check(found.len() <= page, || {
            format!("a page of {page} held {} rows", found.len())
        })?;
        let ids: Vec<ObjectId> = found.iter().map(Live::id).collect();
        let ascending = ids.windows(2).all(|pair| pair[0] < pair[1]);
        let onward = after.is_none_or(|after| ids.first().is_none_or(|id| *id > after));
        check(ascending && onward, || {
            format!("a page after {after:?} held {ids:?}")
        })?;
        listed.extend(found.iter().map(|found| (found.id(), n_of(found))));
        if found.len() < page {
            break;
        }
        after = listed.last().map(|(id, _)| *id);
    }
    check(listed == expected, || {
        format!(
            "at step {step} a walk of pages of {page} with {filter:?} listed {listed:?}, \
             and the rows are {expected:?}"
        )
    })
}

/// The pages of a walk made at `step`: a size and a filter that vary
/// with it, so that over a history every size from one up, no filter, a
/// filter most rows pass and one few pass are all walked, and the way a
/// filtered page is taken, by sorting the few or by meeting the many in
/// order, is both.
fn walk_at(step: usize) -> (usize, Option<Filter>) {
    let page = 1 + step % 5;
    let filter = match step % 3 {
        0 => None,
        1 => Some(Filter::Gte {
            property: "n".into(),
            value: json!((step % 7) as i64),
        }),
        _ => Some(Filter::Eq {
            property: "n".into(),
            value: json!((step / 3) as i64),
        }),
    };
    (page, filter)
}

/// A snapshot of `state`, once its pages are seen to list the same rows.
fn snapshot_walked(state: &State, step: usize) -> std::result::Result<Snapshot, String> {
    let (page, filter) = walk_at(step);
    walked(state, page, filter.as_ref(), step)?;
    Ok(snapshot(state))
}

/// A head the coordinator recorded, the rows committed at it, and the
/// step it was recorded in.
#[derive(Debug)]
struct Recorded {
    head: Head,
    rows: Rows,
    step: usize,
}

/// Where, in `heads`, those to be had still begin, when those `within` are
/// and the one before them is, which is how the partition stood when they
/// began: what the coordinator lets a restore go back to, of commits and a
/// window.
fn reach(heads: &[Recorded], within: impl Fn(&Recorded) -> bool) -> usize {
    heads
        .partition_point(|head| !within(head))
        .saturating_sub(1)
}

/// How far back a restore may go, as the coordinator is asked.
const REACH: Duration = Duration::from_secs(STEP.as_secs() * WINDOW as u64);

/// What is in the way of a clean.
#[derive(Debug, Clone, PartialEq)]
enum Way {
    /// A fault to come, or a writer not the partition's own, not open or
    /// not abreast of what is committed: the clean may fail or not.
    Hindered,
    /// Nothing: the clean must succeed.
    Clear,
}

/// A writer, and whether it can still commit. Any failed write closes one,
/// and so does anything else that leaves it saying it is closed.
struct Seat {
    writer: Writer,
    open: bool,
}

struct World {
    cluster: Cluster,
    faults: Arc<Mutex<Faults>>,
    blobs: Arc<SimBlobs>,
    coordinator: Arc<SimCoordinator>,
    /// The lease most recently taken, whether or not its writer opened.
    held: Lease,
    seats: Vec<Seat>,
    follower: Follower,
    attempts: Vec<Attempt>,
    /// The rows the last restore recorded went back to, and the attempts
    /// recorded since, in the order they were: what is committed is those
    /// rows with each of these made on them in turn.
    base: Rows,
    since: Vec<usize>,
    readings: Vec<Committed>,
    next_row: u64,
    /// What was committed at each position, as read when it was the newest.
    history: BTreeMap<LogPosition, Rows>,
    /// Every head the coordinator recorded, oldest first, each a state to
    /// go back to: those a step ended at and whatever was committed and
    /// then committed over within one step.
    heads: Vec<Recorded>,
    /// How many of them the coordinator was last told to forget.
    forgotten: usize,
    /// The clocks the store and the node go by; the coordinator's is the
    /// cluster's. The node's is `ahead` steps ahead of the store's.
    store_clock: Arc<ManualClock>,
    node_clock: Arc<ManualClock>,
    ahead: i64,
    /// The floors of the cleans so far that asked the store to delete
    /// anything, every one of them and those with grace, each the highest
    /// there has been: a clean keeps every head from the newest at or below
    /// its floor on (see `World::floor`). The second is only counted.
    kept_at: Option<u64>,
    /// How many records of what cleans deleted below the coordinator had
    /// been asked for, and where the record was, when the last clean was
    /// seen to.
    records_seen: usize,
    record_seen: Option<u64>,
    /// What cleans with no grace condemned by the checkpoints they recorded
    /// they deleted below: see [`World::condemned_below`].
    condemned: BTreeSet<String>,
    graced_at: Option<u64>,
    /// Which of the deletes the store has been asked for, by their place
    /// among them, cleans with no grace asked for.
    unkept: BTreeSet<usize>,
    /// How many deletes had been given up unanswered, and how many
    /// answered as lost, when that was last looked at.
    given_up: (usize, usize),
    /// How many heads had been loaded when those kept last were, and how
    /// many deletes the store had made.
    loaded: (usize, usize),
    /// The step at which the store last deleted anything.
    cleaned_at: Option<usize>,
    /// How many of the heads the coordinator recorded have been seen to
    /// name no file whose delete was asked for before them.
    named: usize,
    /// The snapshots taken, oldest first, and the positions of the heads
    /// that restores from them committed.
    snapshots: Vec<Frozen>,
    sourced: BTreeSet<LogPosition>,
    /// What the last restore from a snapshot did, to be held to DW-680's
    /// rule once the step it was made in is checked.
    from_source: Option<FromSource>,
    step: usize,
    made: Made,
}

/// The files of a committed head, by key, as a snapshot took them, with
/// the rows committed at it and when it was taken.
#[derive(Debug, Clone)]
struct Frozen {
    at: Committed,
    files: BTreeMap<String, Bytes>,
    rows: Rows,
    taken_at: DateTime<Utc>,
}

/// A restore from a snapshot that was recorded: the files the snapshot
/// holds, those the restore put into the store, the checkpoint the
/// partition's head rested on when it began, and the head it committed.
#[derive(Debug)]
struct FromSource {
    source: BTreeSet<String>,
    put: Vec<String>,
    rested_on: Option<String>,
    head: ManifestRef,
}

/// What a snapshot holds, as a backup's source gives it: by the keys the
/// files had, and never written.
#[derive(Debug)]
struct Snapshotted(BTreeMap<String, Bytes>);

#[async_trait]
impl Blobs for Snapshotted {
    async fn put(&self, key: &str, _bytes: Bytes) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        tokio::task::yield_now().await;
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| StorageError::Missing {
                key: key.to_string(),
            })
    }
    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        Err(only_read(prefix))
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
}

fn only_read(key: &str) -> StorageError {
    StorageError::Corrupt {
        key: key.to_string(),
        reason: "a snapshot is only read".to_string(),
    }
}

/// The right to read a snapshot, which is never lost.
struct Held;

#[async_trait]
impl vellev_storage::backups::Held for Held {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(true)
    }
    fn still(&self) -> bool {
        true
    }
}

/// How much a history did of what could as well have found nothing to do:
/// merges made, files cleaned away with no grace and for their age,
/// restores recorded, and those of them made beside something else.
#[derive(Debug, Clone, Copy, Default)]
struct Made {
    merges: usize,
    cleaned: usize,
    aged: usize,
    restores: usize,
    raced: usize,
    /// Times the store was told to hold a delete, and the deletes it held:
    /// the one it was told to, and every other given up with it.
    delays: usize,
    held: usize,
    /// Times held deletes were made, and how many of them took a file that
    /// was there still.
    deliveries: usize,
    late: usize,
    /// Waits, and the heads that had passed out of a restore's reach by
    /// the end.
    waits: usize,
    passed: usize,
    /// Times the coordinator was told to forget, and the commits it forgot.
    forgets: usize,
    forgotten: usize,
    /// Heads other than the newest loaded and compared with what was
    /// committed at them.
    reloaded: usize,
    /// Restores recorded while a delete given up was yet to land, and
    /// those not made because it would have taken what they put back.
    under: usize,
    avoided: usize,
    /// Cleans that had nothing in their way and so had to succeed.
    unhindered: usize,
    /// Records of what cleans deleted below asked for, and of those the
    /// ones not made.
    records: usize,
    unrecorded: usize,
    /// Checkpoints recorded, and those of them that replay logs committed
    /// while they were written.
    checkpoints: usize,
    replaying: usize,
    /// Steps that ended with the oldest head a restore may go to just the
    /// one a clean with grace kept from, and those of them with the node's
    /// clock as far ahead as it may be.
    edge: usize,
    edge_ahead: usize,
    /// Restores from a snapshot recorded, and those of them recorded while
    /// a delete given up was yet to land.
    sourced: usize,
    sourced_under: usize,
}

impl Made {
    fn and(self, other: Self) -> Self {
        Self {
            merges: self.merges + other.merges,
            cleaned: self.cleaned + other.cleaned,
            aged: self.aged + other.aged,
            restores: self.restores + other.restores,
            raced: self.raced + other.raced,
            delays: self.delays + other.delays,
            held: self.held + other.held,
            deliveries: self.deliveries + other.deliveries,
            late: self.late + other.late,
            waits: self.waits + other.waits,
            passed: self.passed + other.passed,
            forgets: self.forgets + other.forgets,
            forgotten: self.forgotten + other.forgotten,
            reloaded: self.reloaded + other.reloaded,
            under: self.under + other.under,
            avoided: self.avoided + other.avoided,
            unhindered: self.unhindered + other.unhindered,
            records: self.records + other.records,
            unrecorded: self.unrecorded + other.unrecorded,
            checkpoints: self.checkpoints + other.checkpoints,
            replaying: self.replaying + other.replaying,
            edge: self.edge + other.edge,
            edge_ahead: self.edge_ahead + other.edge_ahead,
            sourced: self.sourced + other.sourced,
            sourced_under: self.sourced_under + other.sourced_under,
        }
    }
}

/// Merges as soon as two segments lie side by side.
const MERGING: Merging = Merging {
    stale: 0.5,
    fanout: 2,
    rows: 1_000_000,
    clustered_above: 200_000,
};

/// How many steps the coordinator's clock and the node's are ahead of
/// the store's, which is the true one, as a history begins: the
/// coordinator's by as many as it likes, which nothing set against a time
/// it gives is to notice, and the node's by up to [`AHEAD`].
#[derive(Debug, Clone, Copy, Default)]
struct Clocks {
    coordinator: i64,
    node: i64,
}

/// How far the node's clock may run ahead of the store's, in steps: as
/// far as a clean with grace keeps more than a restore may go back to.
/// Further, and such a clean deletes files of a head a restore may still
/// go to, which is DW-681's to prevent.
const AHEAD: i64 = (GRACE - WINDOW) as i64;

/// `steps` steps, as a span of time.
fn steps(steps: i64) -> chrono::Duration {
    chrono::Duration::from_std(STEP).unwrap() * steps as i32
}

type Violation = std::result::Result<(), String>;

fn check(holds: bool, violation: impl FnOnce() -> String) -> Violation {
    holds.then_some(()).ok_or_else(violation)
}

impl World {
    async fn new(canary: Option<Canary>) -> Self {
        Self::with_clocks(canary, Clocks::default()).await
    }

    /// A world whose clocks are as `clocks` has them. The coordinator is a
    /// memory one of its own, on a clock of its own, which is the
    /// cluster's: what waits out a lease moves it.
    async fn with_clocks(canary: Option<Canary>, clocks: Clocks) -> Self {
        let cluster = Cluster::new().await;
        let store_clock = cluster.clock.clone();
        let ahead = |by: i64| Arc::new(ManualClock::starting_at(store_clock.now() + steps(by)));
        let (coordinator_clock, node_clock) = (ahead(clocks.coordinator), ahead(clocks.node));
        let memory = Arc::new(MemoryCoordinator::new(coordinator_clock.clone()));
        memory.register(&cluster.key).await.unwrap();
        let cluster = Cluster {
            coordinator: memory,
            clock: coordinator_clock,
            ..cluster
        };
        let faults = Arc::new(Mutex::new(Faults::default()));
        let sent = Arc::new(Mutex::new(Vec::new()));
        let record = Arc::new(Mutex::new(None));
        let coordinator = Arc::new(SimCoordinator {
            clean: cluster.coordinator.clone(),
            clock: cluster.clock.clone(),
            heads: Mutex::default(),
            faults: faults.clone(),
            canary,
            recorded: AtomicUsize::new(0),
            restores: AtomicUsize::new(0),
            unfenced: Mutex::default(),
            sent: sent.clone(),
            record: record.clone(),
            records: Mutex::default(),
            answering: Mutex::default(),
        });
        let blobs = Arc::new(SimBlobs {
            clean: cluster.blobs.clone(),
            faults: faults.clone(),
            canary,
            clock: store_clock.clone(),
            written: Mutex::default(),
            listed: AtomicUsize::new(0),
            delayed: Mutex::default(),
            rewritten: Mutex::default(),
            held: Mutex::default(),
            unanswered: AtomicUsize::new(0),
            refused: AtomicUsize::new(0),
            asked: AtomicUsize::new(0),
            deleted: AtomicUsize::new(0),
            deleting: AtomicUsize::new(0),
            landed: Mutex::default(),
            landings: Mutex::default(),
            shadow: Mutex::default(),
            read: Mutex::default(),
            sent,
            record,
            asked_under: Mutex::default(),
            pausing: Mutex::default(),
            waiting: tokio::sync::Notify::new(),
            resumed: tokio::sync::Notify::new(),
        });
        let follower = Follower::open(
            cluster.key.clone(),
            blobs.clone(),
            &cluster.committed().await,
        )
        .await
        .unwrap();
        let (held, committed) = cluster.lease().await;
        let writer = Writer::open(blobs.clone(), coordinator.clone(), held.clone(), &committed)
            .await
            .unwrap()
            .merging_with(MERGING)
            .timed_by(node_clock.clone());
        let setup = vec![
            create(COLLECTION),
            Mutation::Insert {
                collection: name(COLLECTION),
                id: counter(),
                properties: json!({ "n": 0 }).as_object().unwrap().clone(),
                vector: None,
            },
        ];
        writer.commit(setup).await.unwrap();
        let mut world = Self {
            cluster,
            faults,
            blobs,
            coordinator,
            held,
            seats: vec![Seat { writer, open: true }],
            follower,
            attempts: Vec::new(),
            base: BTreeMap::from([(counter(), value(0, Vec::new()))]),
            since: Vec::new(),
            readings: Vec::new(),
            next_row: 0,
            history: BTreeMap::from([(LogPosition(0), BTreeMap::new())]),
            heads: Vec::new(),
            forgotten: 0,
            store_clock,
            node_clock,
            ahead: clocks.node,
            kept_at: None,
            records_seen: 0,
            record_seen: None,
            condemned: BTreeSet::new(),
            graced_at: None,
            unkept: BTreeSet::new(),
            given_up: (0, 0),
            loaded: (0, 0),
            cleaned_at: None,
            named: 0,
            snapshots: Vec::new(),
            sourced: BTreeSet::new(),
            from_source: None,
            step: 0,
            made: Made::default(),
        };
        world.verify().await.unwrap();
        world
    }

    /// The lease expires and a new node takes it; every earlier writer is
    /// now a zombie, whether or not the new one manages to open.
    async fn take_over(&mut self) {
        // Waiting out the lease moves the coordinator's clock, and the
        // others go with it, each as far from it as before.
        let before = self.cluster.clock.now();
        let (lease, committed) = self.cluster.lease().await;
        let waited = (self.cluster.clock.now() - before).to_std().unwrap();
        self.store_clock.advance(waited);
        self.node_clock.advance(waited);
        self.held = lease;
        self.open_writer(&committed).await;
    }

    /// Open a writer on the held lease through the faulty store. A node
    /// whose first try fails while reads are failing tries once more.
    async fn open_writer(&mut self, committed: &Committed) {
        for _ in 0..2 {
            let opened = Writer::open(
                self.blobs.clone(),
                self.coordinator.clone(),
                self.held.clone(),
                committed,
            );
            if let Ok(writer) = opened.await {
                let writer = writer
                    .merging_with(MERGING)
                    .timed_by(self.node_clock.clone());
                self.seats.push(Seat { writer, open: true });
                return;
            }
            self.faults.lock().unwrap().failing_gets = 0;
        }
    }

    /// The seat `who` writes from. The cluster always has a current writer:
    /// when the newest is closed a node takes over before the write.
    async fn seat(&mut self, who: Who) -> usize {
        match who {
            Who::Any(pick) => pick % self.seats.len(),
            Who::Current => {
                if !self.seats[self.seats.len() - 1].open {
                    self.take_over().await;
                }
                self.seats.len() - 1
            }
        }
    }

    async fn write(
        &mut self,
        who: Who,
        rows: u64,
        delete: Option<usize>,
        beside_a_flush: Option<Pause>,
    ) -> Violation {
        let rows: Vec<u64> = (self.next_row..self.next_row + rows).collect();
        self.next_row += rows.len() as u64;
        let attempt = self.attempts.len() as i64;
        let insert = |n: &u64| Mutation::Insert {
            collection: name(COLLECTION),
            id: row(*n),
            properties: json!({ "n": attempt }).as_object().unwrap().clone(),
            vector: Some(vec![attempt as f32, 0.5]),
        };
        let count = Mutation::Increment {
            collection: name(COLLECTION),
            id: counter(),
            property: "n".into(),
            by: 1.0,
        };
        // One of the rows there are, which a writer left behind may no
        // longer have, or not yet: it then refuses the request.
        let live = self.expected();
        let deleted = delete.and_then(|pick| {
            let rows = live.keys().filter(|id| **id != counter());
            rows.clone().nth(pick % rows.count().max(1)).copied()
        });
        let delete = deleted.map(|id| Mutation::Delete {
            collection: name(COLLECTION),
            id,
            only_if_version: None,
        });
        let mutations = rows.iter().map(insert).chain(delete).chain([count]);
        self.attempts.push(Attempt {
            batch: Batch {
                request: Some(RequestId::generate()),
                mutations: mutations.collect(),
            },
            effect: Effect::Write { rows, deleted },
            acknowledged: false,
            recorded: false,
            refused: false,
        });
        let made = self.attempts.len() - 1;
        self.attempt(who, made, beside_a_flush).await
    }

    /// A write to one of the rows a reader last saw, conditional on the
    /// version the reader gave for it, or on one nothing is at: `kind`
    /// picks a patch that changes `n`, a patch that leaves it as it is, a
    /// replace that changes the vector alone, or a delete. The request
    /// counts like any other.
    async fn conditional(&mut self, who: Who, pick: usize, kind: u8, fresh: bool) -> Violation {
        let read = self.history.values().next_back().expect("a reading");
        let rows: Vec<(&ObjectId, &Row)> =
            read.iter().filter(|(id, _)| **id != counter()).collect();
        if rows.is_empty() {
            return Ok(());
        }
        let (id, (n, _, version)) = rows[pick % rows.len()];
        let attempt = self.attempts.len() as i64;
        let version = match fresh {
            true => *version,
            // Content nothing has: a row's `n` is never negative.
            false => value(-1 - attempt, Vec::new()).2,
        };
        let change = match kind % 4 {
            0 => Change::Patch(attempt),
            1 => Change::Patch(*n),
            2 => Change::Vector(vec![attempt as f32, 0.25]),
            _ => Change::Delete,
        };
        let properties = |n: i64| json!({ "n": n }).as_object().unwrap().clone();
        let mutation = match &change {
            Change::Patch(n) => Mutation::Patch {
                collection: name(COLLECTION),
                id: *id,
                properties: properties(*n),
                vector: None,
                only_if: None,
                only_if_version: Some(version),
            },
            Change::Vector(vector) => Mutation::Upsert {
                collection: name(COLLECTION),
                id: *id,
                properties: properties(*n),
                vector: Some(vector.clone()),
                only_if_version: Some(version),
            },
            Change::Delete => Mutation::Delete {
                collection: name(COLLECTION),
                id: *id,
                only_if_version: Some(version),
            },
        };
        let count = Mutation::Increment {
            collection: name(COLLECTION),
            id: counter(),
            property: "n".into(),
            by: 1.0,
        };
        self.attempts.push(Attempt {
            batch: Batch {
                request: Some(RequestId::generate()),
                mutations: vec![mutation, count],
            },
            effect: Effect::Conditional {
                id: *id,
                version,
                change,
            },
            acknowledged: false,
            recorded: false,
            refused: false,
        });
        self.attempt(who, self.attempts.len() - 1, None).await
    }

    /// Make the request of attempt `made`, for the first time or again. One
    /// recorded already is answered and makes nothing; one refused stays
    /// refused; any other is made now if its commit is recorded, or
    /// refused for good by that commit, which is what a conditional
    /// request whose condition does not hold comes to.
    ///
    /// `beside_a_flush`, the request is made while a checkpoint is being
    /// written: after the checkpoint has taken the rows it is of, and
    /// before it uploads the segment of them, if it has any to write, or
    /// once it has uploaded it or failed to. The checkpoint's
    /// commit is recorded like any other, so there the request's was if a
    /// log was: a write's head is a log and a checkpoint's is not.
    async fn attempt(&mut self, who: Who, made: usize, beside_a_flush: Option<Pause>) -> Violation {
        let seat = self.seat(who).await;
        let before = self.coordinator.recorded.load(Ordering::SeqCst);
        let logged = self.coordinator.logs();
        let batch = self.attempts[made].batch.clone();
        let writer = &self.seats[seat].writer;
        let writing = writer.commit_batches(vec![batch], Embedded::new());
        let (outcome, committed) = match beside_a_flush {
            Some(pause) => {
                // The checkpoint is written as far as its first upload,
                // where the store keeps it waiting; the request is made,
                // and the checkpoint then goes on. Were the two left to take
                // turns as they came, which came first would lie with the
                // threads that build a segment and encode a commit, and no
                // history would be made the same way twice.
                let flushing = writer.flush();
                tokio::pin!(flushing);
                *self.blobs.pausing.lock().unwrap() = Some(pause);
                let paused = tokio::select! {
                    biased;
                    _ = &mut flushing => false,
                    () = self.blobs.waiting.notified() => true,
                };
                *self.blobs.pausing.lock().unwrap() = None;
                let outcome = writing.await;
                if paused {
                    self.blobs.resumed.notify_one();
                    let _ = flushing.await;
                }
                (outcome, self.coordinator.logs() > logged)
            }
            None => {
                let outcome = writing.await;
                let recorded = self.coordinator.recorded.load(Ordering::SeqCst);
                (outcome, recorded > before)
            }
        };
        let outcome = outcome.and_then(|mut answers| answers.pop().expect("one batch"));
        self.seats[seat].open = !self.seats[seat].writer.is_closed();

        // Whether a conditional request's object is at its version in
        // what is committed, by which the commit that answers it must
        // have made it or refused it: the writer that committed had seen
        // everything committed before.
        let holds = match &self.attempts[made].effect {
            Effect::Write { .. } => true,
            Effect::Conditional { id, version, .. } => {
                self.expected().get(id).map(|row| row.2) == Some(*version)
            }
        };
        let refused = matches!(&outcome, Err(error) if error.refusal().is_some());
        let attempt = &mut self.attempts[made];
        if attempt.recorded {
            return check(!(committed && outcome.is_ok()), || {
                format!("request {made}, answered before, was made again")
            });
        }
        if committed && !attempt.refused {
            check(holds || outcome.is_err(), || {
                format!("request {made} was made while its object was not at the version given")
            })?;
            check(!holds || !refused, || {
                format!("request {made} was refused while its object was at the version given")
            })?;
        }
        attempt.refused |= refused || (committed && !holds);
        attempt.acknowledged |= outcome.is_ok();
        if attempt.refused {
            return check(outcome.is_err(), || {
                format!("request {made}, refused before, was made after all")
            });
        }
        if committed {
            attempt.recorded = true;
            self.since.push(made);
        }
        let Ok(receipt) = outcome else { return Ok(()) };
        let counted = self.expected()[&counter()].0;
        check(
            receipt.outcomes.last() == Some(&Outcome::Value(Value::Int(counted))),
            || {
                format!(
                    "a write was told the counter is {:?} when it is {counted}",
                    receipt.outcomes.last()
                )
            },
        )
    }

    /// What is committed, by what was recorded: the rows the last restore
    /// went back to, and each attempt recorded since made on them in turn.
    fn expected(&self) -> Rows {
        let mut rows = self.base.clone();
        for made in &self.since {
            match &self.attempts[*made].effect {
                Effect::Write {
                    rows: written,
                    deleted,
                } => {
                    let value = value(*made as i64, vec![*made as f32, 0.5]);
                    rows.extend(written.iter().map(|n| (row(*n), value.clone())));
                    if let Some(deleted) = deleted {
                        rows.remove(deleted);
                    }
                }
                Effect::Conditional { id, change, .. } => {
                    let held = rows.get(id).cloned().expect("made at its version");
                    match change {
                        Change::Patch(n) => {
                            rows.insert(*id, value(*n, held.1));
                        }
                        Change::Vector(vector) => {
                            rows.insert(*id, value(held.0, vector.clone()));
                        }
                        Change::Delete => {
                            rows.remove(id);
                        }
                    }
                }
            }
            let counted = rows.get(&counter()).expect("the counter").0 + 1;
            rows.insert(counter(), value(counted, Vec::new()));
        }
        rows
    }

    /// Flush, merge or clean from `writer`, and say how many merges were
    /// made and what came of a clean.
    async fn beside(
        writer: &Writer,
        beside: Beside,
        grace: Duration,
    ) -> (usize, vellev_storage::Result<usize>) {
        match beside {
            Beside::Flush => writer.flush().await.map_or((0, Ok(0)), |_| (0, Ok(0))),
            Beside::Merge => (
                usize::from(matches!(writer.merge().await, Ok(Some(_)))),
                Ok(0),
            ),
            Beside::Clean => (0, writer.clean(grace).await),
        }
    }

    /// Put the partition back as it stood at one of its last few heads,
    /// with `beside` going on at once if anything. A restore recorded,
    /// whatever its writer was told, is where everything committed goes on
    /// from. One that fails must have a reason to: a fault in the store or
    /// the coordinator, a writer no longer the partition's, or a state to
    /// go back to that a clean has since deleted.
    async fn restore(
        &mut self,
        who: Who,
        back: usize,
        beside: Option<(Beside, bool)>,
    ) -> Violation {
        let seat = self.seat(who).await;
        let Some(target) = self.heads[self.in_reach()..].iter().rev().nth(back) else {
            return Ok(());
        };
        let (head, seen_at, rows) = (target.head.clone(), target.step, target.rows.clone());
        let point = self.resolved(&head).await?;
        // Only the partition's own writer cleans with no grace.
        let grace = match who {
            Who::Current => Duration::ZERO,
            Who::Any(_) => STEP * GRACE as u32,
        };
        // What a clean beside it is to keep, worked out before either is
        // made, as by the files it will find.
        let cleaning = match beside {
            Some((Beside::Clean, _)) => Some((grace, self.floor(grace).await)),
            _ => None,
        };
        // What begins first takes the building lock first, and holds it
        // until it is done.
        let clean_first = matches!(beside, Some((Beside::Clean, true)));
        let made_anyway = self.blobs.canary == Some(Canary::RestoresUnderALateDelete);
        if !made_anyway
            && self
                .taken_by_a_late_delete(&point, cleaning, clean_first)
                .await
        {
            self.made.avoided += 1;
            return Ok(());
        }
        let under = !self.blobs.held.lock().unwrap().is_empty();
        let way = match cleaning {
            Some(_) => Some(self.way(seat, true).await),
            None => None,
        };
        let landed = self.blobs.landed.lock().unwrap().len();
        let faulty = {
            let faults = self.faults.lock().unwrap();
            !faults.puts.is_empty() || !faults.commits.is_empty() || faults.failing_gets > 0
        };
        // The partition's own writer, and abreast of it: another on the
        // same lease, as a node that started its writer again leaves, may
        // have committed since.
        let at = self.seats[seat].writer.partition().read(State::position);
        let abreast = self.seats[seat].writer.lease().generation == self.held.generation
            && self.cluster.committed().await.log_position == at;
        let current = abreast && seat + 1 == self.seats.len() && self.seats[seat].open;
        let before = self.coordinator.restores.load(Ordering::SeqCst);
        let (asked, deleted) = self.blobs.deletes();
        let (reach, sent) = (self.in_reach(), self.blobs.sent.lock().unwrap().len());
        let given = self.given(grace);
        let writer = &self.seats[seat].writer;
        let restore = Restore {
            request: RequestId::generate(),
            asked: [0; 32],
            from: point.log_position,
            from_at: self.cluster.clock.now(),
        };
        let restoring = writer.restore_recorded(point.manifest.as_ref(), restore, None);
        let (outcome, (merged, cleaned)) = match beside {
            None => (restoring.await, (0, Ok(0))),
            Some((beside, false)) => tokio::join!(restoring, Self::beside(writer, beside, given)),
            Some((beside, true)) => {
                let (made, outcome) = tokio::join!(Self::beside(writer, beside, given), restoring);
                (outcome, made)
            }
        };
        let recorded = self.coordinator.restores.load(Ordering::SeqCst) > before;
        self.seats[seat].open = !self.seats[seat].writer.is_closed();
        if let (Some((grace, floor)), Some(way)) = (cleaning, way) {
            // Whether the clean listed the files after the restore had
            // uploaded its own, and so was made after it.
            let after = self.blobs.listed.load(Ordering::SeqCst) > landed;
            check(after || clean_first || !recorded, || {
                format!("a clean begun after a restore to {point:?} was made before it")
            })?;
            // And so is its floor, by the files it listed, the restore's
            // among them, and the restore's head, which it is abreast of.
            let floor = match after {
                true => self.floor(grace).await,
                false => floor,
            };
            self.cleaned(grace, asked, floor, reach, sent)?;
            self.came_to(way, &cleaned)?;
        }
        let cleaned = cleaned.unwrap_or(0);
        self.made.merges += merged;
        self.made.cleaned += cleaned;
        self.made.raced += usize::from(beside.is_some() && recorded);
        self.made.under += usize::from(under && recorded);
        if self.blobs.deletes().1 > deleted {
            self.cleaned_at = Some(self.step);
        }
        check(recorded || outcome.is_err(), || {
            format!("a restore to {point:?} was acknowledged and never recorded")
        })?;
        // Whoever had not seen all that was committed must not put any of
        // it back: its lease may be the partition's still, where a node
        // started its writer again, but its commit follows nothing.
        check(abreast || !recorded, || {
            format!("a writer left behind at {at:?} had its restore to {point:?} recorded")
        })?;
        if let Err(error) = &outcome {
            let deleted = self.cleaned_at.is_some_and(|cleaned| cleaned >= seen_at);
            check(faulty || !current || deleted, || {
                format!("a restore to {point:?} failed with nothing in its way: {error}")
            })?;
        }
        if recorded {
            self.made.restores += 1;
            self.base = rows;
            self.since.clear();
        }
        Ok(())
    }

    /// The committed head a restore to `head` goes back to, as the
    /// coordinator resolves it: by its position, which it must answer with
    /// that head; and by its time as well, which it must answer with that
    /// head too, where that is the newest head recorded by then and is
    /// within the window. One committed over within a step is not, and the
    /// oldest a restore may reach is before the window.
    async fn resolved(&self, head: &Head) -> std::result::Result<Committed, String> {
        let key = &self.cluster.key;
        let point = RestorePoint {
            position: head.position,
            head: head.key.clone(),
            at: head.at,
        };
        let asked = |point| async move {
            let resolved = self.coordinator.restore_point(key, point, REACH).await;
            resolved.map_err(|error| format!("the coordinator resolves no restore: {error}"))
        };
        let by_position = asked(Point::Position(head.position)).await?;
        check(by_position.as_ref() == Some(&point), || {
            format!("a restore to {point:?} by its position goes to {by_position:?}")
        })?;
        let start = self.cluster.clock.now() - REACH;
        let newest = self
            .heads
            .iter()
            .rev()
            .find(|other| other.head.at <= head.at);
        if head.at >= start && newest.is_some_and(|newest| newest.head == *head) {
            let by_time = asked(Point::At(head.at)).await?;
            check(by_time.as_ref() == Some(&point), || {
                format!("a restore to {point:?} by its time goes to {by_time:?}")
            })?;
        }
        Ok(Committed {
            log_position: head.position,
            manifest: Some(head.key.clone()),
            ..self.cluster.committed().await
        })
    }

    /// What a clean asked to give `grace` is given.
    fn given(&self, grace: Duration) -> Duration {
        match self.blobs.canary {
            Some(Canary::CleansWithNoGrace) => Duration::ZERO,
            _ => grace,
        }
    }

    /// Note what a clean with `grace` and `floor` was to keep, if it asked
    /// the store to delete anything, as the store had been `asked` so many
    /// times before it: every head from the newest at or below its floor.
    /// Note too which of the deletes asked for, from the `sent`th on, a
    /// clean with no grace asked for.
    ///
    /// And see that a clean given grace let go nothing a restore may go
    /// to needs: of the deletes it asked for, none that had not been asked
    /// for before is of a file one of the heads from the `reach`th on, as
    /// the coordinator lets a restore reach them when it began, needs.
    /// What a restore may reach only moves on, so this, at each such
    /// clean, is that at every step. It holds while the node's clock runs
    /// no more ahead of the store's than such a clean keeps more than a
    /// restore may go back to, [`AHEAD`] steps, whatever the coordinator's
    /// clock says, which it sets against its own times alone. DW-680's K2
    /// for a restore to a point in the window rests on it. A node whose
    /// clock runs further ahead breaks it, and DW-681, which is to cap what
    /// a clean deletes by the commits the coordinator still has, is to
    /// mend that. What a clean with no grace let go before is not this
    /// one's to answer for.
    fn cleaned(
        &mut self,
        grace: Duration,
        asked: usize,
        floor: u64,
        reach: usize,
        sent: usize,
    ) -> Violation {
        if self.blobs.deletes().0 > asked {
            self.kept_at = self.kept_at.max(Some(floor));
            if grace >= STEP * GRACE as u32 {
                self.graced_at = self.graced_at.max(Some(floor));
            }
        }
        let now = self.blobs.sent.lock().unwrap().clone();
        self.kept_young(grace, &now[sent..])?;
        let recorded = self.recorded_first(floor, &now[sent..])?;
        let given = self.given(grace);
        if given.is_zero() {
            self.unkept.extend(sent..now.len());
            if let Some(recorded) = recorded {
                self.condemned_below(&recorded)?;
            }
            return Ok(());
        }
        let (before, after) = now.split_at(sent);
        let newly: Vec<&String> = after
            .iter()
            .filter(|key| !before.contains(key) && !self.condemned.contains(*key))
            .collect();
        for recorded in self.heads.get(reach..).unwrap_or_default() {
            let needed = self.named_at(&recorded.head.key)?;
            let let_go = newly.iter().find(|key| needed.contains(**key));
            check(let_go.is_none(), || {
                format!(
                    "DW-681: a clean with grace asked for the delete of {let_go:?}, which {:?}, \
                     a head a restore may go to, needs",
                    recorded.head
                )
            })?;
        }
        Ok(())
    }

    /// DW-681's R2: a clean with grace asks for the delete of no file
    /// younger than its grace, by the node's clock against the store's
    /// date. Today every young file lies above the base a clean deletes
    /// below, so this holds of itself; once a clean may end its walk at
    /// what an earlier one deleted, young files lie below it too, and it
    /// is a rule. Of the deletes `asked`, only those below the newest head
    /// are a clean's: a flush, a merge or a restore beside it drops its
    /// own uploads, which lie at or past it. A writer left behind would
    /// drop its own below that head, but none does in a clean's step: it
    /// drops them on a task of its own once its own commit fails, which
    /// the step that made them settles before the next begins; and its
    /// restore beside a clean fails at `abreast` before it uploads
    /// anything. [`World::recorded_first`] rests on the same.
    fn kept_young(&self, grace: Duration, asked: &[String]) -> Violation {
        if grace.is_zero() {
            return Ok(());
        }
        let young_from = self.young_from(grace);
        let newest = (self.coordinator.heads.lock().unwrap())
            .last()
            .map_or(0, |head| head.position.0);
        let landed = self.blobs.landed.lock().unwrap();
        let young = asked
            .iter()
            .filter(|key| position(key).is_some_and(|at| at < newest))
            .find(|key| landed.get(*key).is_some_and(|(at, _)| *at >= young_from));
        check(young.is_none(), || {
            format!("DW-681 R2: a clean with grace {grace:?} asked for the delete of {young:?}, younger than it")
        })
    }

    /// DW-681's K8, of a clean whose floor was `floor` and which asked for
    /// the deletes `asked`, both ways. Each of its deletes was asked for
    /// with a record in the coordinator above it, so recorded first; every
    /// record it asked for is of a checkpoint at or below its floor, which
    /// is where its walk ends unless it ends at what a clean deleted below
    /// the record, never above what it keeps; and the record is the highest
    /// any clean had made, and never went down. Only deletes below the
    /// newest head are a clean's: a flush, a merge or a restore beside it
    /// drops its own uploads, which lie at or past it.
    fn recorded_first(
        &mut self,
        floor: u64,
        asked: &[String],
    ) -> std::result::Result<Option<Cleaned>, String> {
        let newest = (self.coordinator.heads.lock().unwrap())
            .last()
            .map_or(0, |head| head.position.0);
        let under = self.blobs.asked_under.lock().unwrap().clone();
        let unrecorded = asked
            .iter()
            .filter(|key| position(key).is_some_and(|at| at < newest))
            .find(|key| {
                let at = position(key).unwrap_or(0);
                !under
                    .get(*key)
                    .is_some_and(|record| record.is_some_and(|record| at < record))
            });
        check(unrecorded.is_none(), || {
            let record = unrecorded.and_then(|key| under.get(key));
            format!(
                "DW-681 K8: a clean asked for the delete of {unrecorded:?} with the record of what \
                 cleans deleted below at {record:?}, not above it"
            )
        })?;
        let records = self.coordinator.records.lock().unwrap().clone();
        let above = records[self.records_seen..]
            .iter()
            .find(|(asked, _)| asked.position.0 > floor);
        check(above.is_none(), || {
            format!("DW-681 K8: a clean with its floor at {floor} recorded {above:?}, above it")
        })?;
        let new = &records[self.records_seen..];
        self.made.records += new.len();
        self.made.unrecorded += new.iter().filter(|(_, made)| !made).count();
        let made_now = (new.iter().filter(|(_, made)| *made))
            .map(|(asked, _)| asked.clone())
            .max_by_key(|asked| asked.position);
        self.records_seen = records.len();
        let record = *self.blobs.record.lock().unwrap();
        let highest = records
            .iter()
            .filter(|(_, made)| *made)
            .map(|(asked, _)| asked.position.0)
            .max();
        check(record == highest && record >= self.record_seen, || {
            format!(
                "DW-681 K8: the record is at {record:?}, after it was at {:?}, and the highest \
                 a clean recorded is {highest:?}",
                self.record_seen
            )
        })?;
        self.record_seen = record;
        Ok(made_now)
    }

    /// Note what a clean with no grace that recorded `recorded` condemned:
    /// every file below that checkpoint that no head at or above it needs.
    /// A later clean whose walk ends at what that clean deleted deletes
    /// the rest of it, which that clean was to delete and may not have got
    /// to; and a clean with no grace keeps less than a restore may go back
    /// to, which no node does. So a clean with grace is not charged with
    /// those deletes, as one is not with the deletes such a clean asked
    /// for itself: the files are of heads that clean left unloadable.
    fn condemned_below(&mut self, recorded: &Cleaned) -> Violation {
        let above: Vec<ManifestRef> = (self.heads.iter())
            .filter(|recorded_head| recorded_head.head.position >= recorded.position)
            .map(|recorded_head| recorded_head.head.key.clone())
            .collect();
        let mut kept = BTreeSet::new();
        for head in &above {
            kept.extend(self.named_at(head)?);
        }
        let landed: Vec<String> = self.blobs.landed.lock().unwrap().keys().cloned().collect();
        let below = landed.into_iter().filter(|key| {
            position(key).is_some_and(|at| at < recorded.position.0) && !kept.contains(key)
        });
        self.condemned.extend(below);
        Ok(())
    }

    /// The time from which a file is young to a clean made now with
    /// `grace`, by the node's clock, which a clean reckons by, set against
    /// the store's dates: a file just as old as the grace is young still.
    fn young_from(&self, grace: Duration) -> DateTime<Utc> {
        self.node_clock.now() - grace
    }

    /// The floor of a clean made now with `grace`, as the files there are
    /// give it: the position before the lowest of a young file, or that of
    /// the newest head if none is lower. The clean keeps every file the
    /// checkpoints from the newest at or below it on name, and so every
    /// head from that checkpoint on.
    async fn floor(&self, grace: Duration) -> u64 {
        let (_, young) = self.blobs.files(self.young_from(grace)).await;
        let heads = self.coordinator.heads.lock().unwrap();
        let newest = heads.last().map_or(0, |head| head.position.0);
        young.map_or(newest, |young| newest.min(young.saturating_sub(1)))
    }

    /// Where in `heads` the newest at or below `floor` is, from which on a
    /// clean with that floor kept every head.
    fn index_of(&self, floor: Option<u64>) -> usize {
        let floor = floor.unwrap_or(0);
        let below = self
            .heads
            .partition_point(|head| head.head.position.0 <= floor);
        below.saturating_sub(1)
    }

    /// Where in `heads` those begin that a restore may go to by the
    /// coordinator's clock.
    fn in_reach(&self) -> usize {
        let start = self.cluster.clock.now() - REACH;
        reach(&self.heads, |recorded| recorded.head.at >= start)
    }

    /// Whether a restore to `point`, beside a clean with the grace and the
    /// floor of `cleaning` if beside one, which begins before it if `clean_first`,
    /// would have what it puts back into use taken by a delete that lands
    /// late.
    ///
    /// A clean holds the building lock until the store has answered every
    /// delete, but one answered as lost, or dropped with the rest of its
    /// lot, may be made later all the same. A restore meanwhile to a state
    /// that lists such a file finds it there still and commits a
    /// checkpoint that lists it, and the delete then takes it from under
    /// what is committed. Only a state some clean was not to keep has
    /// files a clean asked to be deleted, and only a clean with no grace
    /// keeps less than a restore may go back to: a node cleans with the
    /// tenant's window and more. So this is what such a clean was asked
    /// for, and no fault of the writer.
    ///
    /// The restore is not made to a state no clean was to keep in just two
    /// cases. One is that a delete given up and yet to land is of a file
    /// that state needs: its head, the logs and the checkpoint behind it,
    /// or a file the checkpoint lists. The other is that such a delete may
    /// come to be beside it: a clean with no grace begins before the
    /// restore with a delete to be held. A clean that begins after the
    /// restore is made after it, from the restore's own head, which lists
    /// what the state needs, and so asks for none of it; and a clean with
    /// grace keeps every state a restore here goes to, which is one whose
    /// head is younger than that grace. Every other restore is made,
    /// whatever is on its way.
    async fn taken_by_a_late_delete(
        &self,
        point: &Committed,
        cleaning: Option<(Duration, u64)>,
        clean_first: bool,
    ) -> bool {
        let kept_at = cleaning.map_or(self.kept_at, |(_, floor)| self.kept_at.max(Some(floor)));
        let kept = self.index_of(kept_at);
        let at = |recorded: &Recorded| recorded.head.position == point.log_position;
        if self.heads.iter().position(at).is_some_and(|at| at >= kept) {
            return false;
        }
        let armed = self.faults.lock().unwrap().held_delete.is_some();
        if armed && clean_first && cleaning.is_some_and(|(grace, _)| grace.is_zero()) {
            return true;
        }
        let held = self.blobs.held.lock().unwrap().clone();
        if held.is_empty() {
            return false;
        }
        let needed = self.needed_at(point.manifest.as_ref()).await;
        needed.is_some_and(|needed| held.iter().any(|key| needed.contains(key)))
    }

    /// Every file the state at `head` needs: the head, the logs behind it
    /// back to its checkpoint, and what that lists. None if any of the
    /// heads cannot be read.
    async fn needed_at(&self, head: Option<&ManifestRef>) -> Option<BTreeSet<String>> {
        let mut needed = BTreeSet::new();
        let mut cursor = head.map(|head| head.0.clone());
        while let Some(key) = cursor {
            let frame = self.blobs.clean.get(&key).await.ok()?;
            cursor = match decode_head(&key, &frame).ok()? {
                Stored::Log(log) => log.prev,
                Stored::Manifest(checkpoint) => {
                    for segment in checkpoint.segments {
                        needed.insert(segment.key);
                        needed.extend(segment.index);
                        needed.extend(segment.vectors);
                    }
                    needed.extend(checkpoint.replay);
                    None
                }
            };
            needed.insert(key);
        }
        Some(needed)
    }

    /// What is in the way of a clean from `seat`, beside a restore or by
    /// itself. A clean reads and deletes; it uploads nothing
    /// and commits nothing, so an upload or a commit that is to fail is in
    /// the way only of a restore beside it, which may then leave anything.
    async fn way(&self, seat: usize, beside_a_restore: bool) -> Way {
        let faulty = {
            let faults = self.faults.lock().unwrap();
            let failing = !faults.puts.is_empty() || !faults.commits.is_empty();
            let failing = beside_a_restore && failing;
            let recording = !faults.records.is_empty();
            failing || recording || faults.failing_gets > 0 || faults.held_delete.is_some()
        };
        let writer = &self.seats[seat].writer;
        let at = writer.partition().read(State::position);
        let abreast = writer.lease().generation == self.held.generation
            && self.cluster.committed().await.log_position == at;
        if faulty || !abreast || seat + 1 != self.seats.len() || writer.is_closed() {
            return Way::Hindered;
        }
        Way::Clear
    }

    /// See that a clean with `way` before it came to what it had to: with
    /// nothing in its way it succeeded, whatever cleans came before it.
    fn came_to(&mut self, way: Way, cleaned: &vellev_storage::Result<usize>) -> Violation {
        self.made.unhindered += usize::from(way == Way::Clear);
        match (way, cleaned) {
            (Way::Hindered, _) | (Way::Clear, Ok(_)) => Ok(()),
            (Way::Clear, Err(error)) => {
                Err(format!("a clean failed with nothing in its way: {error}"))
            }
        }
    }

    /// Let `hours` pass, for the store and for the coordinator alike.
    fn pass(&mut self, hours: usize) {
        self.step += hours;
        for clock in [&self.cluster.clock, &self.store_clock, &self.node_clock] {
            clock.advance(STEP * hours as u32);
        }
    }

    /// An hour passes and `action` is made in it, with whatever it leaves
    /// to be done on a task of its own.
    async fn step(&mut self, action: Action) -> Violation {
        self.pass(1);
        let made = self.act(action).await;
        self.blobs.settle().await;
        made
    }

    async fn act(&mut self, action: Action) -> Violation {
        match action {
            Action::Write { who, rows, delete } => {
                return self.write(who, rows, delete, None).await
            }
            Action::WriteBesideFlush {
                who,
                rows,
                delete,
                first,
            } => {
                let pause = match first {
                    true => Pause::After,
                    false => Pause::Before,
                };
                return self.write(who, rows, delete, Some(pause)).await;
            }
            Action::Conditional {
                who,
                pick,
                kind,
                fresh,
            } => return self.conditional(who, pick, kind, fresh).await,
            Action::Again { who, pick } => {
                if self.attempts.is_empty() {
                    return Ok(());
                }
                return self.attempt(who, pick % self.attempts.len(), None).await;
            }
            Action::Flush { who } => {
                let seat = self.seat(who).await;
                let _ = self.seats[seat].writer.flush().await;
                self.seats[seat].open = !self.seats[seat].writer.is_closed();
            }
            Action::Merge { who } => {
                let seat = self.seat(who).await;
                let merged = self.seats[seat].writer.merge().await;
                self.made.merges += usize::from(matches!(merged, Ok(Some(_))));
                self.seats[seat].open = !self.seats[seat].writer.is_closed();
            }
            Action::Clean { who, aged } => {
                let seat = self.seat(who).await;
                let grace = match (who, aged) {
                    (Who::Current, false) => Duration::ZERO,
                    _ => STEP * GRACE as u32,
                };
                let (asked, deleted) = self.blobs.deletes();
                let floor = self.floor(grace).await;
                let (reach, sent) = (self.in_reach(), self.blobs.sent.lock().unwrap().len());
                let way = self.way(seat, false).await;
                let cleaning = self.seats[seat].writer.clean(self.given(grace));
                let cleaned = cleaning.await;
                self.cleaned(grace, asked, floor, reach, sent)?;
                self.came_to(way, &cleaned)?;
                let cleaned = cleaned.unwrap_or(0);
                match grace.is_zero() {
                    true => self.made.cleaned += cleaned,
                    false => self.made.aged += cleaned,
                }
                // Whatever the writer was told: a clean that fails may have
                // deleted some of what it meant to.
                if self.blobs.deletes().1 > deleted {
                    self.cleaned_at = Some(self.step);
                }
                self.seats[seat].open = !self.seats[seat].writer.is_closed();
            }
            Action::Restore { who, back, beside } => return self.restore(who, back, beside).await,
            Action::Takeover => self.take_over().await,
            Action::Reopen => {
                let committed = self.cluster.committed().await;
                self.open_writer(&committed).await;
            }
            Action::FailUpload(fault) => self.faults.lock().unwrap().puts.push_back(fault),
            Action::FailCommits(fault, times) => self
                .faults
                .lock()
                .unwrap()
                .commits
                .extend(std::iter::repeat_n(fault, times)),
            Action::FailReads(times) => self.faults.lock().unwrap().failing_gets += times,
            Action::FailRecord(fault) => self.faults.lock().unwrap().records.push_back(fault),
            Action::DeliverDelayed => self.blobs.deliver_delayed().await,
            Action::DelayDelete(after) => {
                self.faults.lock().unwrap().held_delete = Some(after);
                self.made.delays += 1;
            }
            Action::DeliverDelayedDeletes => {
                let (held, took) = self.blobs.deliver_held_deletes().await;
                self.made.deliveries += 1;
                self.made.held += held;
                self.made.late += took;
                if took > 0 {
                    self.cleaned_at = Some(self.step);
                }
            }
            Action::Wait(hours) => {
                self.pass(hours);
                self.made.waits += 1;
            }
            Action::NodeAhead(by) => {
                let by = (by as i64).min(AHEAD - self.ahead).max(0);
                self.node_clock.advance(STEP * by as u32);
                self.ahead += by;
            }
            Action::ForgetHistory => {
                let key = &self.cluster.key;
                let forgot = self.coordinator.forget_history(key, REACH).await;
                let forgot = forgot.map_err(|error| format!("nothing was forgotten: {error}"))?;
                self.listed().await?;
                let passed = self.in_reach() - self.forgotten;
                self.forgotten += passed;
                self.made.forgets += 1;
                self.made.forgotten += forgot as usize;
                return check(forgot as usize == passed, || {
                    format!(
                        "the coordinator forgot {forgot} commits when {passed} had passed out of reach"
                    )
                });
            }
            Action::Follow => return self.follow().await,
            Action::FollowStale(pick) => return self.follow_stale(pick).await,
            Action::Snapshot => self.snapshot().await,
            Action::RestoreFrom(pick) => return self.restore_from(pick, false).await,
            Action::RestoreFromAsToday(pick) => return self.restore_from(pick, true).await,
        }
        Ok(())
    }

    /// A follower catches up through the faulty store. It may fail while
    /// reads are failing; once they recover it must reach the committed state.
    async fn follow(&mut self) -> Violation {
        let committed = self.cluster.committed().await;
        let reads_were_failing = self.faults.lock().unwrap().failing_gets > 0;
        if let Err(error) = self.follower.catch_up(&committed).await {
            check(reads_were_failing, || {
                format!("a follower failed with a healthy store: {error}")
            })?;
            self.faults.lock().unwrap().failing_gets = 0;
            self.follower
                .catch_up(&committed)
                .await
                .map_err(|error| format!("a follower did not recover: {error}"))?;
        }
        let truth = self.truth(&committed).await?;
        let followed = self.follower.partition().read(snapshot);
        check(followed == truth, || {
            format!("a follower holds {followed:?}, committed is {truth:?}")
        })?;
        self.readings.push(committed);
        Ok(())
    }

    async fn follow_stale(&mut self, pick: usize) -> Violation {
        let Some(stale) = self.readings.get(pick % self.readings.len().max(1)) else {
            return Ok(());
        };
        let before = self.follower.partition().read(snapshot);
        let outcome = self.follower.catch_up(stale).await;
        let after = self.follower.partition().read(snapshot);
        check(outcome.is_ok() && after == before, || {
            format!("a stale reading moved a follower from {before:?} to {after:?} ({outcome:?})")
        })
    }

    /// Copy the committed head's files out of the store, with the rows
    /// committed at it: every file the state there needs, read through a
    /// healthy handle, as a backup reads them. None is taken if any cannot
    /// be read.
    async fn snapshot(&mut self) {
        let committed = self.cluster.committed().await;
        let Some(needed) = self.needed_at(committed.manifest.as_ref()).await else {
            return;
        };
        let mut files = BTreeMap::new();
        for key in needed {
            let Ok(bytes) = self.blobs.clean.get(&key).await else {
                return;
            };
            files.insert(key, bytes);
        }
        self.snapshots.push(Frozen {
            at: committed,
            files,
            rows: self.expected(),
            taken_at: self.cluster.clock.now(),
        });
    }

    /// Put the partition back as snapshot `pick` holds it, from its current
    /// writer, as a node puts a tenant back from a backup: the snapshot's
    /// files read by the keys they had and copied in under keys of the
    /// restore's own (`Writer::restore_from`). A restore recorded is where
    /// everything committed goes on from, with the rows the snapshot holds;
    /// one that fails changes nothing, which every step is checked for, and
    /// must have a reason to: a fault in the store or the coordinator, or a
    /// writer no longer the partition's. Either way it is held to DW-680's
    /// rule (`World::kept_out_of_old_keys`).
    ///
    /// `as_today`, it is made as a node made it before DW-680: each file of
    /// the snapshot the store lacks is put back under the key it had, and
    /// the partition restored to the snapshot's head, whose checkpoint then
    /// names by their old keys the files the store had still. That breaks
    /// the rule every time, so it is made only in the histories a test sets
    /// out, to show the checker catches each way it goes wrong.
    async fn restore_from(&mut self, pick: usize, as_today: bool) -> Violation {
        let Some(frozen) = self
            .snapshots
            .get(pick % self.snapshots.len().max(1))
            .cloned()
        else {
            return Ok(());
        };
        let seat = self.seat(Who::Current).await;
        let committed = self.cluster.committed().await;
        let rested_on = self.checkpoint_of(committed.manifest.as_ref());
        let landed = self.blobs.landings.lock().unwrap().len();
        let before = self.coordinator.restores.load(Ordering::SeqCst);
        let faulty = {
            let faults = self.faults.lock().unwrap();
            !faults.puts.is_empty() || !faults.commits.is_empty() || faults.failing_gets > 0
        };
        let at = self.seats[seat].writer.partition().read(State::position);
        let current = self.seats[seat].writer.lease().generation == self.held.generation
            && committed.log_position == at
            && seat + 1 == self.seats.len()
            && self.seats[seat].open;
        let restore = Restore {
            request: RequestId::generate(),
            asked: [0; 32],
            from: frozen.at.log_position,
            from_at: frozen.taken_at,
        };
        let head = frozen.at.manifest.as_ref();
        let outcome = match as_today {
            true => {
                let mut fetched = Ok(());
                for (key, bytes) in &frozen.files {
                    if self.blobs.clean.get(key).await.is_err() {
                        fetched = fetched.and(self.blobs.put(key, bytes.clone()).await);
                    }
                }
                match fetched {
                    Ok(()) => {
                        let writer = &self.seats[seat].writer;
                        writer.restore_recorded(head, restore, None).await
                    }
                    Err(error) => Err(error),
                }
            }
            false => {
                let source: Arc<dyn Blobs> = Arc::new(Snapshotted(frozen.files.clone()));
                let writer = &self.seats[seat].writer;
                match writer
                    .restore_from(source, head, restore, None, &Held)
                    .await
                {
                    Ok(ready) => ready.commit().await,
                    Err(error) => Err(error),
                }
            }
        };
        let recorded = self.coordinator.restores.load(Ordering::SeqCst) > before;
        self.seats[seat].open = !self.seats[seat].writer.is_closed();
        check(recorded || outcome.is_err(), || {
            format!(
                "a restore from the snapshot of {:?} was acknowledged and never recorded",
                frozen.at
            )
        })?;
        if let (Err(error), false) = (&outcome, as_today) {
            check(faulty || !current, || {
                format!(
                    "a restore from the snapshot of {:?} failed with nothing in its way: {error}",
                    frozen.at
                )
            })?;
        }
        if recorded {
            let head = self.coordinator.heads.lock().unwrap().last().cloned();
            let head = head.expect("a restore recorded makes a head");
            self.sourced.insert(head.position);
            self.from_source = Some(FromSource {
                source: frozen.files.keys().cloned().collect(),
                put: self.blobs.landings.lock().unwrap()[landed..].to_vec(),
                rested_on,
                head: head.key,
            });
            self.made.restores += 1;
            self.made.sourced += 1;
            self.made.sourced_under += usize::from(!self.blobs.held.lock().unwrap().is_empty());
            self.base = frozen.rows;
            self.since.clear();
        }
        Ok(())
    }

    /// The head named `key`, read from what landed under it, whether it is
    /// there still or not.
    fn landed_head(&self, key: &str) -> Option<Stored> {
        let shadow = self.blobs.shadow.lock().unwrap();
        decode_head(key, shadow.get(key)?).ok()
    }

    /// The checkpoint the state at `head` rests on: the head itself, if it
    /// is one, or the one its logs follow. None if there is none, or a
    /// head on the way never landed.
    fn checkpoint_of(&self, head: Option<&ManifestRef>) -> Option<String> {
        let mut cursor = head.map(|head| head.0.clone());
        while let Some(key) = cursor {
            cursor = match self.landed_head(&key)? {
                Stored::Log(log) => log.prev,
                Stored::Manifest(_) => return Some(key),
            };
        }
        None
    }

    /// Every file the state at `head` names, as `World::needed_at` finds
    /// them, read from what landed rather than from what is there.
    fn named_at(&self, head: &ManifestRef) -> std::result::Result<BTreeSet<String>, String> {
        let mut named = BTreeSet::new();
        let mut cursor = Some(head.0.clone());
        while let Some(key) = cursor {
            let read = self.landed_head(&key);
            cursor = match read.ok_or_else(|| format!("the head {key} never landed whole"))? {
                Stored::Log(log) => log.prev,
                Stored::Manifest(checkpoint) => {
                    for segment in checkpoint.segments {
                        named.insert(segment.key);
                        named.extend(segment.index);
                        named.extend(segment.vectors);
                    }
                    named.extend(checkpoint.replay);
                    None
                }
            };
            named.insert(key);
        }
        Ok(named)
    }

    /// DW-680's K2: no head the coordinator recorded names a file whose
    /// delete the store had been asked for before it was, whatever became
    /// of the delete. A delete may be given up and land later, so a head
    /// that names such a file may lose it at any time.
    ///
    /// For a head a restore to a point in the window made, this holds only
    /// while every clean keeps what such a restore may go back to: one
    /// with no grace keeps less, and may have asked for the deletes of a
    /// state the restore then names (see `World::taken_by_a_late_delete`,
    /// by which a history is spared that pairing). Such a head is excused
    /// for a file it names just when every delete of it asked for came
    /// from a clean with no grace. Every other head is held to it for
    /// every file, and so is such a head for one a clean with grace, or
    /// anything else, asked to be deleted.
    ///
    /// While `World::cleaned` holds every clean with grace to asking for
    /// no file a head a restore may go to needs, such a head never names a
    /// file a clean with grace asked to be deleted, and the excuse could
    /// as well take every clean's deletes. It is kept to those of cleans
    /// with no grace as the rule for when DW-681 lets that check go.
    fn never_named_again(&mut self) -> Violation {
        let heads = self.coordinator.heads.lock().unwrap().clone();
        let sent = self.blobs.sent.lock().unwrap().clone();
        for head in &heads[self.named..] {
            let named = self.named_at(&head.key)?;
            let restored = head.restore && !self.sourced.contains(&head.position);
            let unkept = |key: &String| {
                let asked = sent[..head.sent].iter().enumerate();
                let mut asked = asked.filter(|(_, other)| *other == key);
                asked.all(|(at, _)| self.unkept.contains(&at))
            };
            let again = sent[..head.sent]
                .iter()
                .find(|key| named.contains(*key) && !(restored && unkept(key)));
            check(again.is_none(), || {
                format!(
                    "DW-680 K2: the head {} at {:?} names {again:?}, whose delete had been \
                     asked for before it was committed",
                    head.key.0, head.position
                )
            })?;
        }
        self.named = heads.len();
        Ok(())
    }

    /// DW-680's K4, the rule, for the restore from a snapshot made in the
    /// step just checked, if one was recorded: it put no file under a key
    /// the snapshot had; every segment, index and vectors file the head it
    /// committed names is one it put; that head replays no log; and the
    /// checkpoint before it is the one the partition's head rested on when
    /// the restore began, which may be one the snapshot had too.
    fn kept_out_of_old_keys(&mut self) -> Violation {
        let Some(restored) = self.from_source.take() else {
            return Ok(());
        };
        let FromSource {
            source,
            put,
            rested_on,
            head,
        } = restored;
        let again: Vec<&String> = put.iter().filter(|key| source.contains(*key)).collect();
        check(again.is_empty(), || {
            format!("DW-680 K4: a restore from a snapshot put {again:?}, the keys they had")
        })?;
        let Some(Stored::Manifest(checkpoint)) = self.landed_head(&head.0) else {
            return Err(format!(
                "a restore from a snapshot committed {head:?}, no checkpoint"
            ));
        };
        let named = checkpoint.segments.iter().flat_map(|segment| {
            let files = [&segment.index, &segment.vectors];
            files.into_iter().flatten().chain([&segment.key])
        });
        let unput: Vec<&String> = named.filter(|key| !put.contains(*key)).collect();
        check(unput.is_empty(), || {
            format!(
                "DW-680 K4: a restore from a snapshot committed {} naming {unput:?}, which it \
                 did not put",
                head.0
            )
        })?;
        check(checkpoint.replay.is_empty(), || {
            format!(
                "DW-680 K4: a restore from a snapshot committed {} replaying {:?}",
                head.0, checkpoint.replay
            )
        })?;
        check(checkpoint.prev == rested_on, || {
            format!(
                "DW-680 K4: a restore from a snapshot committed {} after {:?}, and the \
                 partition rested on {rested_on:?}",
                head.0, checkpoint.prev
            )
        })
    }

    /// The committed state as a node starting with nothing reads it from a
    /// healthy store.
    async fn truth(&self, committed: &Committed) -> std::result::Result<Snapshot, String> {
        let fresh = Follower::open(
            self.cluster.key.clone(),
            self.blobs.clean.clone(),
            committed,
        )
        .await
        .map_err(|error| format!("the committed state cannot be read: {error}"))?;
        fresh
            .partition()
            .read(|state| snapshot_walked(state, self.step))
    }

    /// Keep the heads the coordinator recorded since the last step, each
    /// with the rows committed at it: those the step ended with for the
    /// head of a write or of a restore, and for a checkpoint's, which
    /// changes none, those of the head before it.
    fn record(&mut self, expected: &Rows, committed: &Committed) -> Violation {
        let heads = self.coordinator.heads.lock().unwrap();
        let mut rows = match self.heads.last() {
            Some(newest) => newest.rows.clone(),
            None => expected.clone(),
        };
        for head in &heads[self.heads.len()..] {
            if head.restore || head.key.0.ends_with(".log") {
                rows.clone_from(expected);
            }
            self.heads.push(Recorded {
                head: head.clone(),
                rows: rows.clone(),
                step: self.step,
            });
        }
        let newest = heads.last().map(|head| (head.position, &head.key));
        let head = committed.manifest.as_ref();
        check(
            newest == head.map(|head| (committed.log_position, head)),
            || format!("{committed:?} is committed, and the newest head recorded is {newest:?}"),
        )?;
        check(rows == *expected, || {
            format!("the rows committed changed with no write or restore recorded: {heads:?}")
        })
    }

    /// Count the checkpoints among the heads recorded from the `had`th on,
    /// and those of them that replay logs.
    async fn count_checkpoints(&mut self, had: usize) {
        for recorded in &self.heads[had..] {
            let key = &recorded.head.key.0;
            let Ok(frame) = self.blobs.clean.get(key).await else {
                continue;
            };
            if let Ok(Stored::Manifest(checkpoint)) = decode_head(key, &frame) {
                self.made.checkpoints += 1;
                self.made.replaying += usize::from(!checkpoint.replay.is_empty());
            }
        }
    }

    /// See that the coordinator lists for a restore just the heads in
    /// reach: every head recorded within the window, as its clock has it,
    /// and the one before them, whatever it has been told to forget. Asked
    /// by position it lists each of those and no other; asked by a time
    /// within the window it lists the newest recorded by then, and asked
    /// by one before the window, none.
    async fn listed(&self) -> Violation {
        let reach = self.in_reach();
        let start = self.cluster.clock.now() - REACH;
        let coordinator = &self.coordinator;
        for (index, recorded) in self.heads.iter().enumerate() {
            let head = &recorded.head;
            let listed = coordinator
                .restore_point(&self.cluster.key, Point::Position(head.position), REACH)
                .await
                .map_err(|error| format!("the coordinator lists no restore points: {error}"))?;
            let point = RestorePoint {
                position: head.position,
                head: head.key.clone(),
                at: head.at,
            };
            match (index >= reach, listed) {
                (true, None) => {
                    return Err(format!(
                        "a restore point within reach is not listed by the coordinator: {point:?}"
                    ))
                }
                (false, Some(listed)) => {
                    return Err(format!(
                        "the coordinator lists a restore point out of reach: {listed:?}"
                    ))
                }
                (true, Some(listed)) => check(listed == point, || {
                    format!("the coordinator lists {listed:?} where {point:?} was recorded")
                })?,
                (false, None) => {}
            }
            let listed = coordinator
                .restore_point(&self.cluster.key, Point::At(head.at), REACH)
                .await
                .map_err(|error| format!("the coordinator lists no restore points: {error}"))?;
            let by_then = self.heads.partition_point(|other| other.head.at <= head.at);
            let newest = &self.heads[by_then - 1].head;
            let point = (head.at >= start).then(|| RestorePoint {
                position: newest.position,
                head: newest.key.clone(),
                at: newest.at,
            });
            match (point, listed) {
                (Some(point), None) => {
                    return Err(format!(
                        "a restore point within reach is not listed by the coordinator \
                         for its time: {point:?}"
                    ))
                }
                (None, Some(listed)) => {
                    return Err(format!(
                        "the coordinator lists a restore point out of reach, for {:?}: {listed:?}",
                        head.at
                    ))
                }
                (Some(point), Some(listed)) => check(listed == point, || {
                    format!(
                        "the coordinator lists {listed:?} for {:?}, and the newest recorded \
                         by then is {point:?}",
                        head.at
                    )
                })?,
                (None, None) => {}
            }
        }
        let ends = coordinator.restore_points(&self.cluster.key, REACH).await;
        let ends =
            ends.map_err(|error| format!("the coordinator lists no restore points: {error}"))?;
        let ends = ends.map(|(oldest, newest)| (oldest.position, newest.position));
        let reached = &self.heads[reach..];
        let reached = reached.first().zip(reached.last());
        let reached = reached.map(|(oldest, newest)| (oldest.head.position, newest.head.position));
        check(ends == reached, || {
            format!("the coordinator lists restore points from and to {ends:?}, and in reach are {reached:?}")
        })
    }

    /// See that every head that every clean so far was to keep loads, from
    /// a healthy store, as the rows committed at it: each clean is held to
    /// its own promise, whatever the coordinator would still let a restore
    /// go to. A node forgets what is out of the tenant's window and then
    /// cleans with that window and more for grace, so by its cleans these
    /// are every head a restore may go to and some more. A clean with no
    /// grace, which no node makes, was to keep only the newest.
    ///
    /// A file once written is never written again, so a head that loaded
    /// loads until the store deletes something: each is loaded when it is
    /// first seen and again whenever a delete has been made since. The
    /// newest is `committed`, read already as the truth.
    /// Count a step that ends with the oldest head a restore may go to the
    /// very one a clean with grace kept from, by its floor, and those of
    /// them with the node's clock as far ahead as it may be: the edge at
    /// which `World::cleaned` is to hold. Only counted: the floor is of
    /// the files the clean found, which a clean before it may have thinned.
    fn at_the_edge(&mut self) {
        let (reach, graced) = (self.in_reach(), self.index_of(self.graced_at));
        let edge = self.graced_at.is_some() && graced == reach;
        self.made.edge += usize::from(edge);
        self.made.edge_ahead += usize::from(edge && self.ahead == AHEAD);
    }

    async fn reload(&mut self, committed: &Committed) -> Violation {
        let kept = self.index_of(self.kept_at);
        let deleted = self.blobs.deletes().1;
        let newest = self.heads.len() - 1;
        let from = match deleted > self.loaded.1 {
            true => kept,
            false => kept.max(self.loaded.0),
        };
        for recorded in &self.heads[from.min(newest)..newest] {
            let at = Committed {
                log_position: recorded.head.position,
                manifest: Some(recorded.head.key.clone()),
                ..committed.clone()
            };
            let key = self.cluster.key.clone();
            let loaded = Follower::open(key, self.blobs.clean.clone(), &at)
                .await
                .map_err(|error| {
                    format!(
                        "a head every clean was to keep, {:?}, cannot be loaded: {error}",
                        recorded.head
                    )
                })?;
            let (position, rows) = loaded.partition().read(snapshot);
            check(
                position == recorded.head.position && rows == recorded.rows,
                || {
                    format!(
                        "a head every clean was to keep, {:?}, loads at {position:?} as \
                         {rows:?}, and {:?} was committed at it",
                        recorded.head, recorded.rows
                    )
                },
            )?;
            self.made.reloaded += 1;
        }
        self.loaded = (self.heads.len(), deleted);
        Ok(())
    }

    async fn verify(&mut self) -> Violation {
        let rewritten = self.blobs.rewritten.lock().unwrap().clone();
        check(rewritten.is_empty(), || {
            format!("files were written twice: {rewritten:?}")
        })?;
        let unfenced = self.coordinator.unfenced.lock().unwrap().clone();
        check(unfenced.is_empty(), || {
            format!("a superseded lease committed: {unfenced:?}")
        })?;

        let committed = self.cluster.committed().await;
        let newest = self.history.keys().next_back().copied().unwrap_or_default();
        check(committed.log_position >= newest, || {
            format!(
                "committed went back from {newest:?} to {:?}",
                committed.log_position
            )
        })?;

        let unrecorded: Vec<_> = self
            .attempts
            .iter()
            .filter(|a| a.acknowledged && !a.recorded)
            .collect();
        check(unrecorded.is_empty(), || {
            format!("acknowledged but never recorded: {unrecorded:?}")
        })?;

        let (_, rows) = self.truth(&committed).await?;
        let expected = self.expected();
        check(rows == expected, || {
            let missing: Vec<_> = expected
                .iter()
                .filter(|(k, v)| rows.get(k) != Some(v))
                .collect();
            let extra: Vec<_> = rows
                .iter()
                .filter(|(k, v)| expected.get(k) != Some(v))
                .collect();
            format!("committed rows differ from recorded attempts: missing {missing:?}, unexpected {extra:?}")
        })?;
        self.history.insert(committed.log_position, rows);
        let had = self.heads.len();
        self.record(&expected, &committed)?;
        self.count_checkpoints(had).await;
        self.listed().await?;
        self.reload(&committed).await?;
        self.at_the_edge();
        // A delete is given up unanswered only with one the store answered
        // as lost: whoever asked for them together stops at the first to
        // fail. So none is since this was last seen unless one was lost
        // since, which only a delete that was to be held is. Asked of the
        // step, not of the history: a delete held once does not excuse a
        // clean that gives deletes up later for want of patience, as one
        // given a bound too short to wait out the store would.
        let unanswered = self.blobs.unanswered.load(Ordering::SeqCst);
        let refused = self.blobs.refused.load(Ordering::SeqCst);
        let (had_unanswered, had_refused) = self.given_up;
        check(
            unanswered == had_unanswered || refused > had_refused,
            || {
                format!(
                    "{} deletes were given up with none of them held",
                    unanswered - had_unanswered
                )
            },
        )?;
        self.given_up = (unanswered, refused);

        let mut copies = self
            .seats
            .iter()
            .map(|seat| seat.writer.partition())
            .chain([self.follower.partition()]);
        copies.try_for_each(|copy| {
            let (position, held) = copy.read(|state| snapshot_walked(state, self.step))?;
            check(self.history.get(&position) == Some(&held), || {
                format!(
                    "a copy at {position:?} holds rows that were never committed there: {held:?}"
                )
            })
        })?;
        drop(copies);
        self.never_named_again()?;
        self.kept_out_of_old_keys()
    }
}

async fn simulate(
    clocks: Clocks,
    actions: Vec<Action>,
    canary: Option<Canary>,
) -> Result<Made, String> {
    let mut world = World::with_clocks(canary, clocks).await;
    for (step, action) in actions.into_iter().enumerate() {
        let described = format!("step {step} ({action:?})");
        world
            .step(action)
            .await
            .map_err(|violation| format!("{described}: {violation}"))?;
        world
            .verify()
            .await
            .map_err(|violation| format!("{described}: {violation}"))?;
    }
    world.faults.lock().unwrap().failing_gets = 0;
    world.blobs.deliver_delayed().await;
    let (held, took) = world.blobs.deliver_held_deletes().await;
    world.made.held += held;
    world.made.late += took;
    world.made.passed = world.in_reach();
    world
        .verify()
        .await
        .map_err(|violation| format!("after late uploads and deletes arrived: {violation}"))?;
    world
        .follow()
        .await
        .map_err(|violation| format!("at the end: {violation}"))?;
    Ok(world.made)
}

fn run(actions: Vec<Action>, canary: Option<Canary>) -> Result<Made, String> {
    run_on((Clocks::default(), actions), canary)
}

/// `run`, with the clocks as a history has them.
fn run_on(
    (clocks, actions): (Clocks, Vec<Action>),
    canary: Option<Canary>,
) -> Result<Made, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap()
        .block_on(simulate(clocks, actions, canary))
}

/// Two writes each with a checkpoint after it, a merge of the two
/// segments, a clean with no grace that has a delete held, and a restore
/// back past what it deleted: the restore is then to a state whose
/// segments are yet to be taken by the deletes held, the one a history is
/// spared (see `World::taken_by_a_late_delete`). A restore lists the
/// segments of the state it goes to again, and nothing else of it, so
/// only these are what a held delete takes from under it. Left to come
/// about by chance, this comes about in a few histories in a thousand.
fn restore_past_a_held_delete() -> impl Strategy<Value = Vec<Action>> {
    let who = Who::Current;
    (1..4u64, 0..3usize, 1..4usize).prop_map(move |(rows, after, back)| {
        let write = Action::Write {
            who,
            rows,
            delete: None,
        };
        vec![
            write.clone(),
            Action::Flush { who },
            write,
            Action::Flush { who },
            Action::Merge { who },
            Action::DelayDelete(after),
            Action::Clean { who, aged: false },
            Action::Restore {
                who,
                back,
                beside: None,
            },
        ]
    })
}

/// A history, and the clocks it begins with: the coordinator's some
/// steps either side of the store's, and the node's as far behind it as
/// it may be ahead.
fn histories() -> impl Strategy<Value = (Clocks, Vec<Action>)> {
    let part = prop_oneof![
        150 => action().prop_map(|action| vec![action]),
        1 => restore_past_a_held_delete(),
    ];
    let actions = prop::collection::vec(part, 1..80).prop_map(|parts| parts.concat());
    let clocks =
        (-3..=3i64, -AHEAD..=0).prop_map(|(coordinator, node)| Clocks { coordinator, node });
    (clocks, actions)
}

proptest! {
    // Four times as many histories as by default: some faults want a
    // writer left behind, a restore and a clean in just the right order.
    #![proptest_config(ProptestConfig::with_cases(1_024))]

    #[test]
    fn no_history_of_faults_loses_an_acknowledged_write_or_shows_an_unrecorded_one(
        history in histories()
    ) {
        run_on(history, None).map_err(TestCaseError::fail)?;
    }
}

/// The first violation the simulation reports over a fixed set of histories
/// when one of its dependencies is broken in the given way, and how much
/// the histories without one made between them.
fn first_violation(canary: Option<Canary>) -> (Option<String>, Made) {
    let config = Config {
        cases: 300,
        failure_persistence: None,
        max_shrink_iters: 0,
        ..Config::default()
    };
    let rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
    let found = Mutex::new((None, Made::default()));
    let _ = TestRunner::new_with_rng(config, rng).run(&histories(), |history| {
        let outcome = run_on(history, canary);
        let mut found = found.lock().unwrap();
        match outcome {
            Ok(made) => {
                found.1 = found.1.and(made);
                Ok(())
            }
            Err(violation) => {
                found.0.get_or_insert(violation.clone());
                Err(TestCaseError::fail(violation))
            }
        }
    });
    found.into_inner().unwrap()
}

/// The first violation reported with `canary`, which some history of the
/// fixed set must meet.
fn caught(canary: Canary) -> String {
    let (violation, _) = first_violation(Some(canary));
    violation.unwrap_or_else(|| panic!("no history caught the canary {canary:?}"))
}

#[test]
fn the_checker_passes_a_healthy_system_and_catches_a_broken_one() {
    let (violation, made) = first_violation(None);
    assert_eq!(violation, None);
    // Merges, cleaning up and restores are tried in earnest, not only asked
    // for where there is nothing for them to do.
    assert!(
        made.merges > 100 && made.cleaned > 100 && made.restores > 100,
        "{made:?}"
    );
    assert!(made.aged > 20 && made.raced > 20, "{made:?}");
    // So are deletes that land late, commits passing out of reach and being
    // forgotten, and the loading of heads a restore may still go to.
    assert!(made.held > 100 && made.late > 100, "{made:?}");
    assert!(made.passed > 1_000 && made.forgotten > 500, "{made:?}");
    assert!(made.reloaded > 100 && made.under > 0, "{made:?}");
    // And checkpoints written while commits go on, which replay them.
    assert!(made.replaying > 100, "{made:?}");
    // And restores from a snapshot, some of them while a delete given up
    // is yet to land, which DW-680's rule needs no margin of time for.
    assert!(made.sourced > 50 && made.sourced_under > 10, "{made:?}");
    // Cleans alone and beside a restore, and no restore that had no clean
    // beside it: many have nothing in their way.
    assert!(made.unhindered > 300, "{made:?}");
    // And cleans record what they delete below, some of them refused.
    assert!(made.records > 200 && made.unrecorded > 15, "{made:?}");
    // And the oldest head a restore may go to is just the one a clean with
    // grace kept from, with the node's clock as far ahead as it may be too.
    assert!(made.edge > 50 && made.edge_ahead > 10, "{made:?}");

    // A commit acknowledged without being recorded shows up as whichever of
    // its symptoms the history reaches first.
    let symptoms = ["never recorded", "never committed", "was told the counter"];
    let unrecorded = caught(Canary::AcknowledgesUnrecordedCommits);
    assert!(
        symptoms.iter().any(|symptom| unrecorded.contains(symptom)),
        "{unrecorded}"
    );

    let dropped = caught(Canary::DropsSegments);
    assert!(dropped.contains("cannot be read"), "{dropped}");

    let cleaned = caught(Canary::CleansWithNoGrace);
    // A clean that keeps too little is caught by a file it asked for the
    // delete of that is younger than the grace it was given (DW-681's R2),
    // or by the head it left a restore unable to go to, not by the newest,
    // which it keeps.
    assert!(
        cleaned.contains("DW-681 R2: a clean with grace")
            || cleaned.contains("every clean was to keep") && cleaned.contains("cannot be loaded"),
        "{cleaned}"
    );

    let forgotten = caught(Canary::ForgetsWithinTheWindow);
    assert!(
        forgotten.contains("is not listed by the coordinator"),
        "{forgotten}"
    );

    // The store keeps every key it was ever given, deleted since or not.
    let rewritten = caught(Canary::WritesAgainAfterALateDelete);
    assert!(rewritten.contains("written twice"), "{rewritten}");

    let refused = caught(Canary::RefusesDeletes);
    assert!(
        refused.contains("a clean failed with nothing in its way"),
        "{refused}"
    );

    let listed = caught(Canary::ListsOutOfReach);
    assert!(
        listed.contains("the coordinator lists a restore point out of reach"),
        "{listed}"
    );

    let miscounted = caught(Canary::MiscountsTheForgotten);
    assert!(
        miscounted.contains("commits when") && miscounted.contains("had passed out of reach"),
        "{miscounted}"
    );
}

/// The store of the simulation answers a delete it is to hold as lost and
/// keeps the file, holds with it whatever delete is dropped unanswered, and
/// makes them all when told to.
#[test]
fn the_store_holds_a_delete_it_gave_up_and_makes_it_later() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();
    runtime.block_on(async {
        let world = World::new(None).await;
        let blobs = &world.blobs;
        for key in ["answered", "held", "dropped"] {
            blobs.put(key, Bytes::from_static(b"file")).await.unwrap();
        }
        world.faults.lock().unwrap().held_delete = Some(1);
        blobs.delete("answered").await.unwrap();
        let lot = ["held", "dropped"].map(|key| blobs.delete(key));
        futures::future::try_join_all(lot).await.unwrap_err();
        assert!(blobs.clean.get("answered").await.is_err());
        assert!(blobs.clean.get("held").await.is_ok() && blobs.clean.get("dropped").await.is_ok());
        assert_eq!(blobs.unanswered.load(Ordering::SeqCst), 1);

        assert_eq!(blobs.deliver_held_deletes().await, (2, 2));
        assert!(
            blobs.clean.get("held").await.is_err() && blobs.clean.get("dropped").await.is_err()
        );
        assert_eq!(blobs.deliver_held_deletes().await, (0, 0));
    });
}

/// The one pairing no history makes, and what it comes to when made. Two
/// segments are merged into one; a clean with no grace asks the store to
/// delete them and is told the delete was lost; a restore goes back, within
/// its window, to the checkpoint that lists them, finds them there still
/// and commits a checkpoint that lists them too; and the delete lands.
/// Nothing is written twice: the file was there, a head came to name it,
/// and the delete took it.
///
/// The clean was asked to keep nothing of that window, so this is what it
/// was asked for: a node cleans with the tenant's window and more for
/// grace, and never with less. A history is spared the restore and nothing
/// else of it: the delete is held, and lands, all the same.
#[test]
fn a_late_delete_takes_what_a_restore_put_back_only_of_a_clean_with_no_grace() {
    let who = Who::Current;
    let write = || Action::Write {
        who,
        rows: 1,
        delete: None,
    };
    let history = vec![
        write(),
        Action::Flush { who },
        write(),
        Action::Flush { who },
        Action::Merge { who },
        Action::DelayDelete(0),
        Action::Clean { who, aged: false },
        Action::Restore {
            who,
            back: 1,
            beside: None,
        },
        Action::DeliverDelayedDeletes,
    ];
    let made = run(history.clone(), None).unwrap();
    assert_eq!((made.merges, made.restores, made.avoided), (1, 0, 1));
    assert!(made.held > 0 && made.late == made.held, "{made:?}");

    let lost = run(history, Some(Canary::RestoresUnderALateDelete)).unwrap_err();
    assert!(
        lost.contains("DeliverDelayedDeletes") && lost.contains("cannot be read"),
        "{lost}"
    );
    assert!(!lost.contains("written twice"), "{lost}");
}

/// A world in which the partition's own writer has made `actions`, an
/// hour each, and all was well after each.
async fn after(actions: Vec<Action>) -> World {
    let mut world = World::new(None).await;
    for action in actions {
        world.step(action).await.unwrap();
        world.verify().await.unwrap();
    }
    world
}

fn paused() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap()
}

/// DW-681 trigger D: a clean that keeps more than the one before it. The
/// first, with no grace, records the newest checkpoint and deletes every
/// one before it; the second, with some hours', finds the first segment
/// young and walks back past the newest checkpoint, which it passes as
/// the one recorded, to the one deleted. Its walk ends there, and it
/// succeeds. A node meets this when a tenant's window is made longer.
#[test]
fn a_clean_that_keeps_more_than_the_last_succeeds_past_what_the_last_deleted() {
    paused().block_on(async {
        let who = Who::Current;
        let write = Action::Write {
            who,
            rows: 1,
            delete: None,
        };
        let mut world = after(vec![Action::Flush { who }, write, Action::Flush { who }]).await;
        world.pass(1);
        let writer = &world.seats[0].writer;
        assert!(writer.clean(Duration::ZERO).await.unwrap() > 0);
        if let Err(error) = writer.clean(STEP * GRACE as u32).await {
            panic!("DW-681: a clean failed with nothing in its way: {error}");
        }
    });
}

/// DW-681 trigger C, with one grace throughout, no fault, no file left
/// under an old position and no writer left behind: what a node does by
/// itself. Two checkpoints are written and nine hours pass; a clean with
/// eight hours' grace records the second and deletes the first, as it
/// should. A merge, made while the head is the second checkpoint, names
/// its segment at that checkpoint's position: a young file at the
/// checkpoint the clean deleted up to. The next clean's floor is below
/// it, so it walks past the second checkpoint, the one recorded, to the
/// first, which is deleted: its walk ends there, and it succeeds.
#[test]
fn a_clean_succeeds_after_a_merge_past_what_the_clean_before_it_deleted() {
    paused().block_on(async {
        let who = Who::Current;
        let write = || Action::Write {
            who,
            rows: 1,
            delete: None,
        };
        let flush = || Action::Flush { who };
        let world = after(vec![
            write(),
            flush(),
            write(),
            flush(),
            Action::Wait(7),
            Action::Clean { who, aged: true },
            Action::Merge { who },
        ])
        .await;
        assert!(
            world.made.aged > 0 && world.made.merges == 1,
            "{:?}",
            world.made
        );
        assert_eq!(world.made.unhindered, 1);
        let grace = STEP * GRACE as u32;
        if let Err(error) = world.seats[0].writer.clean(grace).await {
            panic!("DW-681: a clean failed with nothing in its way: {error}");
        }
    });
}

/// A write of one row from the partition's own writer.
fn written() -> Action {
    Action::Write {
        who: Who::Current,
        rows: 1,
        delete: None,
    }
}

/// A snapshot of a checkpoint, two checkpoints after it merged into one,
/// and every file the snapshot holds cleaned away by a clean with no
/// grace, with its deletes held from the one `held` on if any; then the
/// partition is put back from the snapshot by `restoring`. The store has
/// the snapshot's files still only if their deletes are held.
fn restored_from_a_snapshot_cleaned_away(
    held: Option<usize>,
    restoring: fn(usize) -> Action,
) -> Vec<Action> {
    let who = Who::Current;
    [
        vec![
            written(),
            Action::Flush { who },
            Action::Snapshot,
            written(),
            Action::Flush { who },
            Action::Merge { who },
        ],
        held.map(Action::DelayDelete).into_iter().collect(),
        vec![
            Action::Clean { who, aged: false },
            restoring(0),
            Action::DeliverDelayedDeletes,
        ],
    ]
    .concat()
}

/// The two ways a restore from a backup goes wrong as a node makes it
/// before DW-680, each caught by what is checked of every history whatever
/// the rule is. Gone from the store, a file the snapshot holds is put back
/// under its key, which is to write it twice. There still, with its delete
/// given up and yet to land, it is put back into use by the checkpoint the
/// restore commits, which names a file whose delete was asked for: written
/// once, and the delete then takes it.
#[test]
fn a_restore_from_a_snapshot_as_a_node_makes_one_today_is_caught_either_way() {
    let today = Action::RestoreFromAsToday;
    let rewritten = run(restored_from_a_snapshot_cleaned_away(None, today), None).unwrap_err();
    assert!(
        rewritten.contains("RestoreFrom") && rewritten.contains("written twice"),
        "{rewritten}"
    );

    let present = run(restored_from_a_snapshot_cleaned_away(Some(0), today), None).unwrap_err();
    assert!(
        present.contains("RestoreFrom") && present.contains("DW-680 K2"),
        "{present}"
    );
    assert!(!present.contains("written twice"), "{present}");
}

/// The same two histories, the partition put back from the snapshot as
/// DW-680 has it: each file copied in under a key of the restore's own,
/// whether the store has the file still under the key it had, its delete
/// given up by a clean with no grace and yet to land, or not. Nothing is
/// written twice, no head names a file whose delete was asked for, the
/// rule is kept, and what is read is what the snapshot held: with no
/// margin of time to stand on, as the late delete of a clean with no grace
/// leaves none.
#[test]
fn a_restore_from_a_snapshot_of_files_cleaned_away_is_made_whole() {
    for held in [None, Some(0)] {
        let history = restored_from_a_snapshot_cleaned_away(held, Action::RestoreFrom);
        let made = run(history, None).unwrap();
        assert_eq!(made.sourced, 1, "{held:?}");
        assert_eq!(made.sourced_under, usize::from(held.is_some()), "{held:?}");
    }
}

/// A snapshot of a checkpoint, two checkpoints after it merged into one,
/// and, once they are old, every file the snapshot holds cleaned away by a
/// clean with grace, as a node cleans; then another such clean, which,
/// with nothing put back, succeeds. Every clean keeps the same grace, so
/// that each takes its base from where the last left it.
async fn cleaned_long_ago() -> World {
    let who = Who::Current;
    let mut world = after(vec![
        written(),
        Action::Flush { who },
        Action::Snapshot,
        written(),
        Action::Flush { who },
        Action::Merge { who },
        Action::Wait(GRACE + 1),
        Action::Clean { who, aged: true },
    ])
    .await;
    assert!(world.made.aged > 0, "{:?}", world.made);
    let mut gone = 0;
    for key in world.snapshots[0].files.keys() {
        gone += usize::from(world.blobs.clean.get(key).await.is_err());
    }
    assert!(gone > 0, "the clean deleted what the snapshot holds");
    world.pass(1);
    let writer = &world.seats[world.seats.len() - 1].writer;
    writer.clean(STEP * GRACE as u32).await.unwrap();
    world
}

/// A restore as DW-680 makes it puts nothing at an old position, nor at
/// the position of the partition's latest checkpoint: its copies are named
/// at the checkpoint it commits. So the clean after it, which succeeds
/// before it, succeeds after it too.
#[test]
fn a_clean_after_a_restore_from_a_snapshot_cleaned_away_long_ago_succeeds() {
    paused().block_on(async {
        let mut world = cleaned_long_ago().await;
        world.step(Action::RestoreFrom(0)).await.unwrap();
        world.verify().await.unwrap();
        assert_eq!(world.made.restores, 1);
        world.pass(1);
        let writer = &world.seats[world.seats.len() - 1].writer;
        writer.clean(STEP * GRACE as u32).await.unwrap();
        world.verify().await.unwrap();
    });
}

/// A copy a restore from a snapshot gave up on, which lands only after the
/// restore failed and its copies were deleted, lies under a key nothing
/// names: no head the coordinator records names it, and nothing reads it.
#[test]
fn a_copy_a_restore_from_a_snapshot_gave_up_on_that_lands_late_is_never_read() {
    let who = Who::Current;
    paused().block_on(async {
        let mut world = after(vec![
            written(),
            Action::Flush { who },
            Action::Snapshot,
            written(),
            Action::FailUpload(PutFault::Delay),
            Action::RestoreFrom(0),
        ])
        .await;
        assert_eq!(world.made.restores, 0, "given up");
        let landed = world.blobs.landings.lock().unwrap().len();
        world.step(Action::DeliverDelayed).await.unwrap();
        let late: Vec<String> = world.blobs.landings.lock().unwrap()[landed..].to_vec();
        assert_eq!(late.len(), 1, "{late:?}");
        assert!(late[0].contains("-restored"), "{late:?}");
        let read = world.blobs.read.lock().unwrap().len();
        for action in [
            written(),
            Action::Flush { who },
            Action::Merge { who },
            Action::RestoreFrom(0),
            Action::Follow,
            Action::Clean { who, aged: true },
        ] {
            world.step(action).await.unwrap();
            world.verify().await.unwrap();
        }
        assert_eq!(world.made.restores, 1);
        let reads = world.blobs.read.lock().unwrap()[read..].to_vec();
        assert!(!reads.contains(&late[0]), "read {}", late[0]);
        let heads = world.coordinator.heads.lock().unwrap().clone();
        for head in heads {
            let named = world.named_at(&head.key).unwrap();
            assert!(
                !named.contains(&late[0]),
                "{} names {}",
                head.key.0,
                late[0]
            );
        }
    });
}

/// What is checked of a history is blind, as it should not be, to a head
/// that names a file whose delete is yet to land, where the store notes a
/// delete only once it makes it: the restore of a snapshot whose files are
/// there still, their deletes held, is not then caught for what it names,
/// but only for breaking DW-680's rule, which it does in any case.
#[test]
fn a_late_delete_noted_only_once_made_goes_unseen_by_what_a_head_names() {
    let history = restored_from_a_snapshot_cleaned_away(Some(0), Action::RestoreFromAsToday);
    let unseen = run(history, Some(Canary::NotesDeletesOnlyMade)).unwrap_err();
    assert!(
        unseen.contains("RestoreFrom") && !unseen.contains("DW-680 K2"),
        "{unseen}"
    );
    assert!(unseen.contains("DW-680 K4"), "{unseen}");
}

/// Whether a world has made the head a history is for.
type MadeHead = fn(&World) -> bool;

/// The head the first snapshot holds.
fn snapshot_head(world: &World) -> String {
    world.snapshots[0].at.manifest.clone().unwrap().0
}

/// The checkpoint the state the first snapshot holds rests on.
fn snapshot_checkpoint(world: &World) -> Manifest {
    let head = world.snapshots[0].at.manifest.as_ref();
    let checkpoint = world.checkpoint_of(head).expect("a checkpoint");
    match world.landed_head(&checkpoint) {
        Some(Stored::Manifest(checkpoint)) => checkpoint,
        other => panic!("{checkpoint} is no checkpoint: {other:?}"),
    }
}

/// DW-680's rule, K4, held to by every restore from a snapshot of each
/// kind of head there is: a checkpoint a flush made; a log after one; a
/// checkpoint that replays a write made while it was written; one a merge
/// made; one a restore made; and the very checkpoint the partition rests
/// on when it is restored, which its committed head is then to follow.
/// Each history is seen to have made the head it is for before the
/// restore is. A node breaks the rule today, each time, by naming in the
/// checkpoint it commits the files it finds in the store by the keys they
/// had.
#[test]
fn a_restore_from_a_snapshot_brings_nothing_in_under_the_key_it_had() {
    let who = Who::Current;
    let flush = || Action::Flush { who };
    let beside = Action::WriteBesideFlush {
        who,
        rows: 1,
        delete: None,
        first: false,
    };
    let restore = Action::Restore {
        who,
        back: 1,
        beside: None,
    };
    let histories: [(&str, Vec<Action>, MadeHead); 6] = [
        (
            "a checkpoint a flush made",
            vec![written(), flush(), Action::Snapshot, written()],
            |world| snapshot_head(world).ends_with(".manifest"),
        ),
        (
            "a log after a checkpoint",
            vec![written(), flush(), written(), Action::Snapshot, written()],
            |world| snapshot_head(world).ends_with(".log"),
        ),
        (
            "a checkpoint that replays a write",
            vec![written(), beside, Action::Snapshot, written()],
            |world| !snapshot_checkpoint(world).replay.is_empty(),
        ),
        (
            "a checkpoint a merge made",
            vec![
                written(),
                flush(),
                written(),
                flush(),
                Action::Merge { who },
                Action::Snapshot,
                written(),
            ],
            |world| {
                let checkpoint = snapshot_checkpoint(world);
                let segments = checkpoint.segments.iter();
                segments
                    .into_iter()
                    .any(|segment| segment.key.contains("-merged"))
            },
        ),
        (
            "a checkpoint a restore made",
            vec![written(), flush(), written(), restore, Action::Snapshot],
            |world| {
                let heads = world.coordinator.heads.lock().unwrap();
                let at = world.snapshots[0].at.log_position;
                heads.iter().any(|head| head.position == at && head.restore)
            },
        ),
        (
            "the checkpoint the partition rests on",
            vec![written(), flush(), Action::Snapshot],
            |world| {
                let newest = world.coordinator.heads.lock().unwrap().last().cloned();
                Some(snapshot_head(world)) == newest.map(|head| head.key.0)
            },
        ),
    ];
    let mut broken = Vec::new();
    for (what, history, made) in histories {
        let violation = paused().block_on(async {
            let mut world = after(history).await;
            assert!(made(&world), "the snapshot is not of {what}");
            let restores = world.made.restores;
            world.step(Action::RestoreFrom(0)).await.unwrap();
            assert_eq!(world.made.restores, restores + 1, "{what}");
            world.verify().await.err()
        });
        if let Some(violation) = violation {
            assert!(violation.contains("DW-680 K4"), "{what}: {violation}");
            broken.push(format!("{what}: {violation}"));
        }
    }
    if let Some(first) = broken.first() {
        panic!(
            "DW-680 K4: a restore from a snapshot broke the rule in {} histories: {first}",
            broken.len()
        );
    }
}

/// A restore from a snapshot that fails changes nothing: the partition
/// stays as it was committed, whatever the snapshot holds.
#[test]
fn a_restore_from_a_snapshot_that_fails_changes_nothing() {
    let who = Who::Current;
    let history = [
        written(),
        Action::Flush { who },
        Action::Snapshot,
        written(),
        Action::FailCommits(CommitFault::Refuse, 3),
        Action::RestoreFrom(0),
        written(),
    ];
    paused().block_on(async {
        let mut world = after(history[..5].to_vec()).await;
        let before = world.cluster.committed().await;
        world.step(history[5].clone()).await.unwrap();
        world.verify().await.unwrap();
        assert_eq!(world.made.restores, 0);
        assert_eq!(world.cluster.committed().await, before);
        world.step(history[6].clone()).await.unwrap();
        world.verify().await.unwrap();
    });
}

/// DW-681 trigger E, by the way a restore from a backup made before
/// DW-680: of the history above, the restore puts back, under their keys
/// and young, files a clean deleted from below the checkpoint it recorded
/// and deleted up to. The next clean finds them young and walks past that
/// checkpoint to the one before it, which the first deleted: its walk
/// ends there, it succeeds, and the files put back, being young, stay.
/// DW-680's restore puts nothing at an old position; this test keeps the
/// record of the way a node made it.
#[test]
fn a_clean_succeeds_after_a_restore_puts_back_what_the_clean_before_it_deleted() {
    paused().block_on(async {
        let mut world = cleaned_long_ago().await;
        world.step(Action::RestoreFromAsToday(0)).await.unwrap();
        assert_eq!(world.made.restores, 1);
        assert!(!world.blobs.rewritten.lock().unwrap().is_empty());
        world.pass(1);
        let writer = &world.seats[world.seats.len() - 1].writer;
        if let Err(error) = writer.clean(STEP * GRACE as u32).await {
            panic!("DW-681: a clean failed with nothing in its way: {error}");
        }
    });
}

/// Known to fail, and DW-681 by a node's clock: O1 of DW-680's design. The
/// node's clock runs three steps ahead of the store's, one more than a
/// clean with grace keeps beyond what a restore may go back to. A write
/// and a checkpoint, then the head to be the oldest a restore may go to
/// and a checkpoint after it, then a write; hours pass, and a clean with
/// grace, one of whose deletes is held, finds that write the oldest young
/// file. It keeps from the checkpoint after the head a restore may still
/// go to, and asks for the deletes of what that head needs. Once they
/// land, it cannot be loaded.
///
/// What a clean deletes is reckoned by the node's clock against the
/// store's dates, and what a restore may go to by the coordinator's
/// against its own: no margin of grace holds whatever the clocks do.
/// DW-681 is to cap what a clean deletes by the commits the coordinator
/// still has.
#[test]
#[should_panic(expected = "DW-681: a clean with grace asked for the delete of")]
fn a_clean_by_a_node_whose_clock_runs_ahead_deletes_what_a_restore_may_go_to() {
    paused().block_on(async {
        let who = Who::Current;
        let ahead = Clocks {
            coordinator: 0,
            node: AHEAD + 1,
        };
        let mut world = World::with_clocks(None, ahead).await;
        let history = [
            written(),
            Action::Flush { who },
            written(),
            Action::Flush { who },
            written(),
            Action::Wait(2),
            Action::DelayDelete(0),
        ];
        for action in history {
            world.step(action).await.unwrap();
            world.verify().await.unwrap();
        }
        let violation = world.step(Action::Clean { who, aged: true }).await;
        let violation = violation.unwrap_err();
        let oldest = world.in_reach();
        let head = world.heads[oldest].head.clone();
        assert!(head.key.0.ends_with(".log"), "{head:?}: {violation}");
        let checkpoint = &world.heads[oldest + 1].head;
        assert!(checkpoint.key.0.ends_with(".manifest"), "{checkpoint:?}");
        assert!(
            !world.blobs.held.lock().unwrap().is_empty(),
            "a delete is held"
        );
        world.blobs.deliver_held_deletes().await;
        let at = Committed {
            log_position: head.position,
            manifest: Some(head.key.clone()),
            ..world.cluster.committed().await
        };
        let key = world.cluster.key.clone();
        let loaded = Follower::open(key, world.blobs.clean.clone(), &at).await;
        assert!(loaded.is_err(), "{head:?} a restore may go to loads still");
        panic!("{violation}");
    });
}

/// A clean with grace made an hour after an idle partition was cleaned
/// with that grace, its head C a checkpoint that replays a write and its
/// files all old, with the partition put back as C held it in that hour
/// if `restored`. The first clean ends its walk at C and deletes the
/// checkpoint before it. Returns C, the checkpoint before it, and what the
/// second clean came to.
async fn cleaned_after_an_idle_head(
    restored: bool,
) -> (String, String, vellev_storage::Result<usize>) {
    let who = Who::Current;
    let beside = Action::WriteBesideFlush {
        who,
        rows: 1,
        delete: None,
        first: false,
    };
    let mut world = after(vec![
        written(),
        Action::Flush { who },
        written(),
        beside,
        Action::Wait(GRACE),
        Action::Clean { who, aged: true },
    ])
    .await;
    let committed = world.cluster.committed().await;
    let head = committed.manifest.clone().expect("a head").0;
    let prev = match world.landed_head(&head) {
        Some(Stored::Manifest(checkpoint)) if !checkpoint.replay.is_empty() => {
            checkpoint.prev.expect("a checkpoint before it")
        }
        other => panic!("the head {head} is no checkpoint that replays a write: {other:?}"),
    };
    assert!(world.made.aged > 0, "{:?}", world.made);
    let (files, _) = world.blobs.files(world.young_from(Duration::ZERO)).await;
    assert!(!files.contains(&prev), "the first clean left {prev}");
    world.pass(1);
    let writer = &world.seats[0].writer;
    if restored {
        let at = committed.manifest.as_ref();
        writer.restore(at, committed.log_position).await.unwrap();
    }
    (head, prev, writer.clean(STEP * GRACE as u32).await)
}

/// DW-685: a point-in-time restore to an idle partition's head names what
/// it uploads at the checkpoint it commits, not at that head. A segment
/// of the rows the head replays, named at the head's own position, is a
/// young file there: the next clean then walks past the head to the
/// checkpoint before it, which the clean before deleted, and every clean fails until
/// that segment is old.
#[test]
fn a_clean_after_a_restore_to_an_idle_head_succeeds() {
    let (head, prev, cleaned) = paused().block_on(cleaned_after_an_idle_head(true));
    if let Err(error) = cleaned {
        panic!("DW-685: a clean after a restore to {head} walked to {prev}: {error}");
    }
}

/// The control for the test above: with no restore, the second clean
/// succeeds.
#[test]
fn a_clean_after_a_clean_of_an_idle_head_succeeds() {
    let (head, prev, cleaned) = paused().block_on(cleaned_after_an_idle_head(false));
    if let Err(error) = cleaned {
        panic!("a clean after a clean of {head} walked to {prev}: {error}");
    }
}

/// A clean with no grace by the partition's own writer deletes the logs
/// of two heads a restore may still go to, as such a clean may; a clean
/// with grace after it, by a node one step ahead, finds the oldest young
/// files gone and so a floor above them. It deletes nothing the first had
/// not asked for already, and nothing the oldest head a restore may go to
/// needs that was there still: what was let go below its floor the first
/// let go. A random history came to this (DW-680, PR 6's runs).
///
/// Told by its floor, taken from the files it lists, the second clean
/// would be charged with what the first let go: hence what a clean with
/// grace is held to is what it asks to be deleted itself (`World::cleaned`).
#[test]
fn a_clean_with_grace_answers_only_for_what_it_lets_go_itself() {
    paused().block_on(async {
        let who = Who::Current;
        let written = || Action::Write {
            who,
            rows: 1,
            delete: None,
        };
        let restored = || Action::Restore {
            who,
            back: 0,
            beside: None,
        };
        let mut world = after(vec![
            Action::Takeover,
            Action::Takeover,
            Action::Takeover,
            Action::Snapshot,
            Action::Takeover,
            Action::Takeover,
            Action::Flush { who },
            restored(),
            restored(),
            written(),
            Action::FailCommits(CommitFault::Refuse, 1),
            written(),
            Action::DelayDelete(3),
            restored(),
            written(),
            Action::Clean { who, aged: false },
            Action::NodeAhead(1),
        ])
        .await;
        let sent = world.blobs.sent.lock().unwrap().clone();
        world.step(Action::Clean { who, aged: true }).await.unwrap();
        world.verify().await.unwrap();
        let asked = world.blobs.sent.lock().unwrap()[sent.len()..].to_vec();
        assert!(!asked.is_empty(), "the clean with grace asked for deletes");
        assert!(
            asked.iter().all(|key| sent.contains(key)),
            "the clean with grace asked for {asked:?}, not all of them asked for before"
        );
        let oldest = world.heads[world.in_reach()].head.clone();
        let needed = world.named_at(&oldest.key).unwrap();
        let files = world.blobs.clean.list("").await.unwrap();
        let there: BTreeSet<String> = files.into_iter().map(|file| file.key).collect();
        let gone: Vec<&String> = needed.iter().filter(|key| !there.contains(*key)).collect();
        assert!(!gone.is_empty(), "{oldest:?} has all it needs");
        assert!(
            gone.iter().all(|key| sent.contains(*key)),
            "{oldest:?} lost {gone:?}, not all of them asked for before the clean with grace"
        );
        let graced = world.index_of(world.graced_at);
        assert!(graced > world.in_reach(), "and its floor is past it");
    });
}

/// Known to fail, and DW-681 by a node's clock, as O1 is, where what a
/// clean with grace lets go is to be told by what it asks to be deleted
/// and not by its floor over every file that ever landed. With the node's
/// clock three steps ahead, a write and a checkpoint, then a write the
/// partition's writer is taken over from, and a write beside a checkpoint:
/// the oldest step a restore may go to. The writer left behind uploads
/// files for a checkpoint of its own, named at its old position; the
/// first lands, it is told otherwise and deletes it. A write, hours pass,
/// and a clean with grace finds that write the oldest young file. It
/// keeps from the checkpoint beside the write, and asks for the deletes
/// of the logs and the checkpoint the head before them needs, and the
/// write beside that checkpoint too.
#[test]
#[should_panic(expected = "DW-681: a clean with grace asked for the delete of")]
fn a_clean_by_a_node_whose_clock_runs_ahead_is_not_excused_by_a_file_it_never_saw() {
    let who = Who::Current;
    let history = vec![
        written(),
        Action::Flush { who },
        written(),
        Action::Takeover,
        Action::WriteBesideFlush {
            who,
            rows: 1,
            delete: None,
            first: false,
        },
        Action::FailUpload(PutFault::LandThenFail),
        Action::Flush { who: Who::Any(0) },
        written(),
        Action::Wait(1),
        Action::Clean { who, aged: true },
    ];
    let ahead = Clocks {
        coordinator: 0,
        node: AHEAD + 1,
    };
    let violation = run_on((ahead, history), None).unwrap_err();
    assert!(violation.starts_with("step 9 "), "{violation}");
    panic!("{violation}");
}

/// Known to fail, and DW-681 by a node's clock, as O1 is, for a clean
/// with grace beside a restore to a point in the window: one made by a
/// node whose writer is the partition's own, which cleans with the window
/// and more, begun just after the restore or just before it. With the
/// node's clock three steps ahead, either way the clean asks for the
/// deletes of what the oldest head a restore may go to needs.
#[test]
#[should_panic(expected = "DW-681: a clean with grace asked for the delete of")]
fn a_clean_beside_a_restore_by_a_node_whose_clock_runs_ahead_deletes_what_a_restore_may_go_to() {
    let who = Who::Current;
    let ahead = Clocks {
        coordinator: 0,
        node: AHEAD + 1,
    };
    let caught = [false, true].map(|clean_first| {
        let history = vec![
            written(),
            Action::Flush { who },
            written(),
            Action::Flush { who },
            written(),
            Action::Wait(3),
            Action::Restore {
                who: Who::Any(0),
                back: 0,
                beside: Some((Beside::Clean, clean_first)),
            },
        ];
        let violation = run_on((ahead, history), None).unwrap_err();
        let restore = "step 6 (Restore { who: Any(0), back: 0, beside: Some((Clean,";
        assert!(violation.starts_with(restore), "{violation}");
        violation
    });
    assert!(
        caught[1].contains("DW-681: a clean with grace"),
        "{}",
        caught[1]
    );
    panic!("{}", caught[0]);
}

/// A clean with no grace asks for the deletes of the logs below its
/// checkpoint and of the two checkpoints before it, the first of which a
/// head a restore may still go to needs, and its deletes from that one on
/// are held. Two steps on, a clean with grace, by a node three steps
/// ahead, finds those checkpoints there still and asks for the first one's
/// delete again. It asks for nothing that had not been asked for, and is
/// not to answer for what a clean with none let go.
#[test]
fn a_clean_with_grace_is_not_charged_with_a_delete_it_asks_for_again() {
    paused().block_on(async {
        let who = Who::Current;
        let ahead = Clocks {
            coordinator: 0,
            node: AHEAD + 1,
        };
        let mut world = World::with_clocks(None, ahead).await;
        let history = [
            written(),
            Action::Flush { who },
            written(),
            written(),
            Action::Flush { who },
            written(),
            Action::Merge { who },
            Action::DelayDelete(5),
            Action::Clean { who, aged: false },
            Action::Follow,
        ];
        for action in history {
            world.step(action).await.unwrap();
            world.verify().await.unwrap();
        }
        assert!(
            !world.blobs.held.lock().unwrap().is_empty(),
            "deletes are held"
        );
        let sent = world.blobs.sent.lock().unwrap().len();
        world.step(Action::Clean { who, aged: true }).await.unwrap();
        world.verify().await.unwrap();
        let all = world.blobs.sent.lock().unwrap().clone();
        let (before, again) = all.split_at(sent);
        assert!(!again.is_empty(), "the clean with grace asked for deletes");
        assert!(again.iter().all(|key| before.contains(key)), "{again:?}");
        let oldest = world.heads[world.in_reach()].head.clone();
        let needed = world.named_at(&oldest.key).unwrap();
        assert!(
            again.iter().any(|key| needed.contains(key)),
            "{oldest:?} needs none of {again:?}"
        );
    });
}

// DW-681: what a clean may take for an earlier clean's work. Each test
// below is written to its design's rule (`dw-681-design.md`, revision 3):
// those that fail today are `should_panic` on the message of the rule
// they hold to, and go plain with the change that makes them pass; those
// that pass today pin what that change must not break.

/// A flush by the partition's own writer.
fn flushed() -> Action {
    Action::Flush { who: Who::Current }
}

/// A world in which the partition's writers have made `actions`, an hour
/// each, checked only by what each test asserts: several leave a writer
/// behind, or a file under an old position, on purpose.
async fn made(actions: Vec<Action>) -> World {
    let mut world = World::new(None).await;
    for action in actions {
        world.step(action).await.unwrap();
    }
    world
}

impl World {
    /// The newest writer.
    fn current(&self) -> &Writer {
        &self.seats[self.seats.len() - 1].writer
    }

    /// The files there are now.
    async fn keys(&self) -> BTreeSet<String> {
        let listed = self.blobs.list("").await.unwrap();
        listed.into_iter().map(|file| file.key).collect()
    }

    /// A clean by the newest writer with `grace`, which is to succeed:
    /// nothing is in its way. How many files it deleted.
    async fn cleans(&self, grace: Duration) -> usize {
        match self.current().clean(grace).await {
            Ok(deleted) => deleted,
            Err(error) => panic!("DW-681: a clean failed with nothing in its way: {error}"),
        }
    }

    /// What the coordinator holds of what cleans deleted below.
    async fn cleaned_record(&self) -> Option<Cleaned> {
        self.coordinator
            .clean
            .cleaned(&self.cluster.key)
            .await
            .unwrap()
    }

    /// A clean by the newest writer with `grace`, which is to fail and
    /// delete nothing: what it failed with.
    async fn fails_and_deletes_nothing(&self, grace: Duration) -> StorageError {
        let asked = self.blobs.deletes().0;
        let before = self.keys().await;
        let error = self.current().clean(grace).await.unwrap_err();
        assert_eq!(self.blobs.deletes().0, asked, "{error}");
        assert_eq!(self.keys().await, before);
        error
    }

    /// The manifest the head's checkpoint follows: the checkpoint before
    /// it, which a clean with the head's checkpoint as its base deletes.
    async fn before_the_head(&self) -> String {
        let head = self.cluster.committed().await.manifest.unwrap().0;
        let frame = self.blobs.shadow.lock().unwrap()[&head].clone();
        match decode_head(&head, &frame).unwrap() {
            Stored::Manifest(checkpoint) => checkpoint.prev.unwrap(),
            Stored::Log(_) => panic!("the head {head} is not a checkpoint"),
        }
    }
}

/// The grace a node gives: the window and more.
fn graced() -> Duration {
    STEP * GRACE as u32
}

/// DW-681's R1′: a chain manifest lost above the checkpoint a clean
/// recorded it deleted below is a loss, not a clean's work. This is the
/// test that tells a walk that has passed the record from one that only
/// knows there is a record: it passes on a cleaner with no record at all,
/// which fails at any checkpoint missing, and fails on one that ends its
/// walk at any checkpoint missing (R1, rejected). A clean records
/// the checkpoint it deletes below, which is below two young ones; another
/// is written, and the older of those two is lost. The next clean walks
/// past the newest two and meets the loss before it has passed the
/// checkpoint recorded: it fails and deletes nothing.
#[test]
fn a_chain_manifest_lost_above_what_was_recorded_fails_the_clean_and_deletes_nothing() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), written(), flushed()]).await;
        world.pass(GRACE + 1);
        for action in [written(), flushed(), written(), flushed()] {
            world.step(action).await.unwrap();
        }
        let lost = world.before_the_head().await;
        assert!(world.cleans(graced()).await > 0);
        for action in [written(), flushed()] {
            world.step(action).await.unwrap();
        }
        world.blobs.delete(&lost).await.unwrap();
        let error = world.fails_and_deletes_nothing(graced()).await;
        assert!(
            matches!(&error, StorageError::Missing { key } if *key == lost),
            "{error}"
        );
    });
}

/// DW-681 trigger A, a put that lands late, with one grace throughout. A
/// writer's put of a log is given up and the writer closed; the next
/// writes, flushes, writes and flushes; fourteen hours on a clean deletes
/// below the newest checkpoint, the one before it among them. Then the
/// put lands, young and under an old position, and the next clean walks
/// past the newest checkpoint to the one deleted. On a node the put would
/// have to land more than a grace late, past what `LANDED_AFTER` allows;
/// a caller of `Writer::clean` with a short grace meets it.
#[test]
fn a_clean_succeeds_after_a_put_lands_late_under_an_old_position() {
    paused().block_on(async {
        let world = made(vec![
            written(),
            Action::FailUpload(PutFault::Delay),
            written(),
            written(),
            flushed(),
            written(),
            flushed(),
            Action::Wait(7),
            Action::Wait(7),
        ])
        .await;
        assert!(world.cleans(graced()).await > 0);
        assert!(!world.blobs.delayed.lock().unwrap().is_empty());
        world.blobs.deliver_delayed().await;
        world.cleans(graced()).await;
    });
}

/// DW-681 trigger B, a writer left behind, with one grace throughout. A
/// writer is deposed; the next writes and flushes twice, and fourteen
/// hours on a clean deletes below its newest checkpoint. The deposed one
/// then flushes: its segment and its manifest land, young, under its old
/// positions, before the coordinator refuses the commit. A node meets it
/// when a writer is paused for longer than the grace and resumes.
#[test]
fn a_clean_succeeds_after_a_writer_left_behind_uploads_under_an_old_position() {
    paused().block_on(async {
        let mut world = made(vec![
            written(),
            Action::Takeover,
            written(),
            flushed(),
            written(),
            flushed(),
            Action::Wait(7),
            Action::Wait(7),
        ])
        .await;
        assert!(world.cleans(graced()).await > 0);
        world
            .step(Action::Flush { who: Who::Any(0) })
            .await
            .unwrap();
        world.cleans(graced()).await;
    });
}

/// DW-681 trigger C′, a plain flush at a checkpoint, with no fault. A write
/// made beside a flush is replayed by the checkpoint the flush commits, so
/// its rows stay in memory and the checkpoint is the head. A clean deletes
/// below it; a flush with nothing written since then names its segment at
/// that checkpoint's own position, young, and the next clean walks past
/// the checkpoint to the one deleted.
#[test]
fn a_clean_succeeds_after_a_flush_at_a_checkpoint_that_replays_a_write() {
    paused().block_on(async {
        let beside = Action::WriteBesideFlush {
            who: Who::Current,
            rows: 1,
            delete: None,
            first: false,
        };
        let mut world = made(vec![
            written(),
            flushed(),
            written(),
            beside,
            Action::Wait(9),
        ])
        .await;
        assert!(world.cleans(graced()).await > 0);
        let before = world.cluster.committed().await.log_position;
        world.step(flushed()).await.unwrap();
        assert!(world.cluster.committed().await.log_position > before);
        world.cleans(graced()).await;
    });
}

/// DW-681 trigger G, a clock that reckons the cutoff further back. A clean
/// deletes below a checkpoint just older than its grace. The lease then
/// goes to a node whose clock is two hours behind: to it that checkpoint
/// is young, so its clean walks past it to the one deleted. The same
/// happens when one node's clock is stepped back.
#[test]
fn a_clean_succeeds_by_a_clock_behind_the_last_cleaners() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), written(), flushed()]).await;
        world.pass(GRACE + 1);
        assert!(world.cleans(graced()).await > 0);
        world.take_over().await;
        let behind = Arc::new(ManualClock::starting_at(world.node_clock.now() - steps(2)));
        let Seat { writer, open } = world.seats.pop().unwrap();
        let writer = writer.timed_by(behind);
        world.seats.push(Seat { writer, open });
        world.cleans(graced()).await;
    });
}

/// DW-681 trigger H, the store's dates made anew, as an operator's copy in
/// place or a move of the bucket does. Every file is young, the floor
/// falls to nothing, and a clean walks to what an earlier one deleted. It
/// is to succeed and delete nothing until the files are old again.
#[test]
fn a_clean_succeeds_and_deletes_nothing_once_the_store_dates_every_file_anew() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), written(), flushed()]).await;
        world.pass(GRACE + 1);
        assert!(world.cleans(graced()).await > 0);
        let now = world.blobs.clock.now();
        for (at, _) in world.blobs.landed.lock().unwrap().values_mut() {
            *at = now;
        }
        world.pass(1);
        assert_eq!(world.cleans(graced()).await, 0);
    });
}

/// DW-681's loss: a chain manifest gone by anything but a clean (an
/// operator, a store's own expiry, a bucket put back in part) is not taken
/// for a clean's work. A clean whose walk meets it fails, as it does today,
/// and deletes nothing: not the head that stood when its window began, nor
/// anything else. The cleaner's failing is the only routine sign that such
/// a file was lost.
#[test]
fn a_chain_manifest_lost_otherwise_than_by_a_clean_fails_the_clean_and_deletes_nothing() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), Action::Wait(9)]).await;
        for action in [written(), flushed(), written(), flushed()] {
            world.step(action).await.unwrap();
        }
        let lost = world.before_the_head().await;
        world.blobs.delete(&lost).await.unwrap();
        let before = world.keys().await;
        let cleaned = world.current().clean(graced()).await;
        let gone: Vec<_> = before.difference(&world.keys().await).cloned().collect();
        assert!(
            gone.is_empty(),
            "the clean took {lost} for a clean's and deleted {gone:?}"
        );
        assert!(
            matches!(&cleaned, Err(StorageError::Missing { key }) if *key == lost),
            "{cleaned:?}"
        );
    });
}

/// DW-681: a read of a checkpoint's predecessor that fails otherwise than
/// by the file's being missing fails the clean, which deletes nothing. It
/// may say nothing of what an earlier clean deleted.
#[test]
fn a_clean_fails_and_deletes_nothing_when_a_predecessor_cannot_be_read() {
    paused().block_on(async {
        let world = made(vec![written(), flushed(), written(), flushed()]).await;
        let before = world.before_the_head().await;
        world.faults.lock().unwrap().failing_key = Some(before.clone());
        let asked = world.blobs.deletes().0;
        let cleaned = world.current().clean(graced()).await;
        assert!(
            matches!(cleaned, Err(StorageError::Blob { .. })),
            "{cleaned:?}"
        );
        assert_eq!(world.blobs.deletes().0, asked);
    });
}

/// DW-681: as above, below the checkpoint an earlier clean deleted up to.
/// A clean with two hours' grace deletes the checkpoint before the newest;
/// one with eight walks back past the newest, and the read of the one
/// deleted fails otherwise than by its being missing. That fails the
/// clean, which deletes nothing: a walk may end only where a file is known
/// to be missing, never where it could not be read.
#[test]
fn a_clean_fails_and_deletes_nothing_when_a_predecessor_below_what_was_cleaned_cannot_be_read() {
    paused().block_on(async {
        let mut world = made(vec![flushed(), written(), flushed()]).await;
        world.pass(3);
        let deleted = world.before_the_head().await;
        assert!(world.cleans(STEP * 2).await > 0);
        assert!(!world.keys().await.contains(&deleted));
        world.faults.lock().unwrap().failing_key = Some(deleted);
        let asked = world.blobs.deletes().0;
        let cleaned = world.current().clean(graced()).await;
        assert!(
            matches!(cleaned, Err(StorageError::Blob { .. })),
            "{cleaned:?}"
        );
        assert_eq!(world.blobs.deletes().0, asked);
    });
}

/// DW-681: the head's own chain missing a file is a partition that cannot
/// be opened, whatever an earlier clean did, and a clean fails for it.
#[test]
fn a_clean_fails_when_the_heads_own_checkpoint_is_missing() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed()]).await;
        let head = world.cluster.committed().await.manifest.unwrap().0;
        assert!(head.contains("/manifest/"), "{head}");
        world.blobs.delete(&head).await.unwrap();
        world.pass(1);
        let cleaned = world.current().clean(graced()).await;
        assert!(
            matches!(&cleaned, Err(StorageError::Missing { key }) if *key == head),
            "{cleaned:?}"
        );
    });
}

/// DW-681's R2: what a writer left behind uploads under an old position,
/// below what a clean deleted up to, stays while it is younger than the
/// grace, and goes once it is older.
#[test]
fn files_left_young_below_what_was_cleaned_stay_until_they_are_old() {
    paused().block_on(async {
        let mut world = made(vec![
            written(),
            Action::Takeover,
            written(),
            flushed(),
            written(),
            flushed(),
            Action::Wait(7),
            Action::Wait(7),
        ])
        .await;
        world.cleans(graced()).await;
        let before = world.keys().await;
        world
            .step(Action::Flush { who: Who::Any(0) })
            .await
            .unwrap();
        let left: BTreeSet<String> = world.keys().await.difference(&before).cloned().collect();
        assert!(!left.is_empty());
        assert_eq!(world.cleans(graced()).await, 0);
        world.pass(GRACE + 1);
        assert_eq!(world.cleans(graced()).await, left.len());
        assert!(world.keys().await.is_disjoint(&left));
    });
}

/// DW-681: a clean whose walk ends at what an earlier clean deleted deletes
/// just what a clean whose floor were that checkpoint deletes: here the
/// files a writer left behind uploaded, which are now old, and nothing
/// else, while those another left behind uploaded since, young, stay, and
/// so does the checkpoint the walk ended at, below a newer one.
#[test]
fn a_clean_ended_by_what_was_cleaned_deletes_just_what_is_old_below_it() {
    paused().block_on(async {
        let mut world = made(vec![
            written(),
            Action::Takeover,
            written(),
            Action::Takeover,
            written(),
            flushed(),
            written(),
            flushed(),
            Action::Wait(7),
            Action::Wait(7),
        ])
        .await;
        assert!(world.cleans(graced()).await > 0);
        let before = world.keys().await;
        world
            .step(Action::Flush { who: Who::Any(0) })
            .await
            .unwrap();
        let old: BTreeSet<String> = world.keys().await.difference(&before).cloned().collect();
        world.pass(GRACE + 1);
        let before = world.keys().await;
        world
            .step(Action::Flush { who: Who::Any(1) })
            .await
            .unwrap();
        let young: BTreeSet<String> = world.keys().await.difference(&before).cloned().collect();
        assert!(!old.is_empty() && !young.is_empty());
        // A checkpoint above the one the walk is to end at, so that the
        // walk takes two before it meets what was deleted.
        world.step(written()).await.unwrap();
        world.step(flushed()).await.unwrap();
        let before = world.keys().await;
        world.cleans(graced()).await;
        let gone: BTreeSet<String> = before.difference(&world.keys().await).cloned().collect();
        assert_eq!(gone, old);
    });
}

/// DW-681, a clean cut short: one with grace is stopped part way through
/// its deletes by a delete the store gives up, having recorded its base,
/// deleted the checkpoint below it and left files it was to delete. A
/// later clean that keeps more, with three hours' more grace, walks past
/// that base: it is to succeed, and to delete nothing the first was not
/// to; and with the record answered as none, it is to fail and delete
/// nothing. Before the record, the later clean failed by trigger D, its
/// grace being the longer, so its `second ⊆ first` check, like the record
/// part, bites only on a cleaner that can get past what was deleted.
#[test]
fn a_clean_after_one_cut_short_deletes_only_what_that_one_was_to() {
    paused().block_on(async {
        let mut world = made(vec![
            written(),
            flushed(),
            written(),
            flushed(),
            Action::Merge { who: Who::Current },
            written(),
            flushed(),
        ])
        .await;
        world.pass(GRACE + 1);
        let head = world.cluster.committed().await;
        let base = head.log_position.0;
        let deleted = world.before_the_head().await;
        // Logs and manifests are listed, and so deleted, before segments
        // and vectors: let those go, and give up the rest.
        let heads = (world.keys().await.into_iter())
            .filter(|key| key.contains("/log/") || key.contains("/manifest/"))
            .filter(|key| position(key).is_some_and(|at| at < base))
            .count();
        world.faults.lock().unwrap().held_delete = Some(heads);
        let sent = world.blobs.sent.lock().unwrap().len();
        assert!(world.current().clean(graced()).await.is_err());
        world.blobs.settle().await;
        world.faults.lock().unwrap().held_delete = None;
        let first: BTreeSet<String> = world.blobs.sent.lock().unwrap()[sent..]
            .iter()
            .cloned()
            .collect();
        let keys = world.keys().await;
        assert!(
            !keys.contains(&deleted),
            "the clean cut short left {deleted}"
        );
        assert!(
            keys.iter().any(|key| first.contains(key)),
            "the clean cut short deleted all it was to"
        );
        // With no record, the later clean takes the checkpoint deleted for
        // one lost: it fails and deletes nothing. The record written before
        // the first clean was cut short is what lets its walk end.
        assert!(world.cleaned_record().await.is_some());
        let asked = world.blobs.deletes().0;
        *world.coordinator.answering.lock().unwrap() = Some(None);
        let unrecorded = world.current().clean(graced() + STEP * 3).await;
        assert!(
            matches!(unrecorded, Err(StorageError::Missing { .. })),
            "{unrecorded:?}"
        );
        assert_eq!(world.blobs.deletes().0, asked);
        *world.coordinator.answering.lock().unwrap() = None;
        let sent = world.blobs.sent.lock().unwrap().len();
        world.cleans(graced() + STEP * 3).await;
        let second: BTreeSet<String> = world.blobs.sent.lock().unwrap()[sent..]
            .iter()
            .cloned()
            .collect();
        assert!(second.is_subset(&first), "{:?}", second.difference(&first));
    });
}

/// DW-681's R1′: a record put back from earlier, as a coordinator restored
/// from a backup has it, or none, takes nothing missing for a clean's work
/// that it does not reach: the clean fails and deletes nothing, as before
/// there was a record, until the record is as it was.
#[test]
fn a_clean_takes_nothing_for_a_cleans_work_on_a_record_from_earlier() {
    paused().block_on(async {
        let mut world = made(vec![flushed(), written(), flushed()]).await;
        world.pass(3);
        assert!(world.cleans(STEP * 2).await > 0);
        assert!(world.cleaned_record().await.is_some());
        *world.coordinator.answering.lock().unwrap() = Some(None);
        let error = world.fails_and_deletes_nothing(graced()).await;
        assert!(matches!(error, StorageError::Missing { .. }), "{error}");
        *world.coordinator.answering.lock().unwrap() = None;
        world.cleans(graced()).await;
    });
}

/// DW-681's R1′: a record of no checkpoint of this chain, however high, is
/// never passed, so nothing missing is taken for a clean's work by it.
#[test]
fn a_record_of_no_checkpoint_of_this_chain_is_never_passed() {
    paused().block_on(async {
        let mut world = made(vec![flushed(), written(), flushed()]).await;
        world.pass(3);
        assert!(world.cleans(STEP * 2).await > 0);
        *world.coordinator.answering.lock().unwrap() = Some(Some(foreign()));
        let error = world.fails_and_deletes_nothing(graced()).await;
        assert!(matches!(error, StorageError::Missing { .. }), "{error}");
    });
}

/// DW-681's R1′: with a record of no checkpoint of this chain above what it
/// deletes below, and nothing missing in its way, a clean deletes as one
/// with no record would, says so, and leaves the record as it is.
#[test]
fn a_clean_deletes_beside_a_record_of_no_checkpoint_of_this_chain() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), written(), flushed()]).await;
        world.pass(GRACE + 1);
        *world.coordinator.answering.lock().unwrap() = Some(Some(foreign()));
        assert!(world.cleans(graced()).await > 0);
        assert!(world.coordinator.records.lock().unwrap().is_empty());
        assert_eq!(world.cleaned_record().await, None);
    });
}

/// A record of what cleans deleted below that names no checkpoint here.
fn foreign() -> Cleaned {
    Cleaned {
        base: ManifestRef("tenant-a/0/manifest/elsewhere.manifest".into()),
        position: LogPosition(999),
    }
}

/// DW-681's R1′: a clean records what it deletes below before it deletes
/// anything. A record refused, or made with its answer lost, fails the
/// clean, which deletes nothing; the record lost on its way back is the
/// coordinator's all the same, and the next clean deletes on it.
#[test]
fn a_clean_records_what_it_deletes_below_before_it_deletes_anything() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), written(), flushed()]).await;
        world.pass(GRACE + 1);
        let base = world.cluster.committed().await.manifest.unwrap();
        world
            .faults
            .lock()
            .unwrap()
            .records
            .push_back(RecordFault::Refuse);
        world.fails_and_deletes_nothing(graced()).await;
        assert_eq!(world.cleaned_record().await, None);
        (world.faults.lock().unwrap().records).push_back(RecordFault::LoseResponse);
        world.fails_and_deletes_nothing(graced()).await;
        assert_eq!(
            world.cleaned_record().await.map(|record| record.base),
            Some(base)
        );
        assert!(world.cleans(graced()).await > 0);
    });
}

/// DW-681's R1′: a clean that cannot read what cleans recorded fails, and
/// deletes nothing: it does not take there to be no record.
#[test]
fn a_clean_that_cannot_read_what_cleans_recorded_fails_and_deletes_nothing() {
    paused().block_on(async {
        let mut world = made(vec![written(), flushed(), written(), flushed()]).await;
        world.pass(GRACE + 1);
        world
            .faults
            .lock()
            .unwrap()
            .records
            .push_back(RecordFault::FailRead);
        world.fails_and_deletes_nothing(graced()).await;
        assert!(world.cleans(graced()).await > 0);
    });
}

/// DW-681: a clean with grace is not charged with deleting what a clean
/// with no grace condemned by the checkpoint it recorded and did not get to
/// delete. The clean with no grace records the newest checkpoint and is cut
/// short by a delete given up, leaving files of heads a restore may still
/// go to, which it left unloadable; its deletes land. A clean with grace
/// then walks past the record to what it deleted, ends there, and deletes
/// the rest. Found by the random histories (minimal history as found).
#[test]
fn a_clean_with_grace_is_not_charged_with_what_a_clean_with_no_grace_condemned() {
    let w = |rows| Action::Write {
        who: Who::Current,
        rows,
        delete: None,
    };
    let history = vec![
        w(1),
        w(3),
        flushed(),
        w(1),
        w(1),
        flushed(),
        w(2),
        flushed(),
        w(2),
        flushed(),
        Action::Merge { who: Who::Current },
        Action::DelayDelete(0),
        Action::Clean {
            who: Who::Current,
            aged: false,
        },
        Action::Restore {
            who: Who::Current,
            back: 1,
            beside: None,
        },
        Action::DeliverDelayedDeletes,
        Action::Clean {
            who: Who::Current,
            aged: true,
        },
    ];
    let clocks = Clocks {
        coordinator: 0,
        node: -1,
    };
    run_on((clocks, history), None).unwrap();
}
