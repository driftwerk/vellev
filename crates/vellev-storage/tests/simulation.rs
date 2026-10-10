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
//! A step takes an hour, and a wait as many more as it says. The store says
//! each file was written as many hours ago as have passed since, and half
//! an hour more, so that
//! cleaning up with a grace of some hours deletes what is older and keeps
//! what is not, as it does where files age by the clock; and the
//! coordinator's clock moves with it, so that commits pass out of the
//! window a restore may reach back and are forgotten.
//!
//! A delete may be given up by whoever asked for it and land later, as a
//! put may. Every head the coordinator recorded is kept with the rows
//! committed at it, and after every step each of them a restore may still
//! go to, by the coordinator and by every clean there has been, is seen to
//! be listed by the coordinator and to load as those rows. A clean with
//! nothing in its way must succeed, but for one known to fail, which must
//! fail just so: see `World::walks_into_what_was_cleaned`.
//!
//! One pairing is not made: a restore to a state that a clean was not to
//! keep, while a delete given up is yet to land. Only a clean with no
//! grace, which no node makes, keeps less than a restore may go back to;
//! the restore would find the file there still and put it back into use,
//! and the delete would then take it, by what the clean was asked and by
//! no fault of the writer.

mod common;

use chrono::{DateTime, SubsecRound};
use std::collections::VecDeque;
use vellev_core::{RequestId, Version};
use vellev_storage::coordinator::{
    Clock, KeyHash, Point, Restore, RestorePoint, Restored, TenantKey,
};

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
}

#[derive(Debug, Default)]
struct Faults {
    puts: VecDeque<PutFault>,
    commits: VecDeque<CommitFault>,
    failing_gets: usize,
    /// How many deletes are answered before one is held, if one is to be.
    held_delete: Option<usize>,
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
    written: Mutex<BTreeSet<String>>,
    delayed: Mutex<Vec<(String, Bytes)>>,
    rewritten: Mutex<Vec<String>>,
    /// The deletes given up by whoever asked for them, which the store has
    /// yet to make.
    held: Mutex<Vec<String>>,
    /// How many of those were held for want of an answer alone: the
    /// request was dropped while the store had it.
    unanswered: AtomicUsize,
    /// How many deletes the store has been asked for, and how many it has
    /// made.
    asked: AtomicUsize,
    deleted: AtomicUsize,
    /// How many hours the history has taken, and at which each file landed.
    step: AtomicUsize,
    landed: Mutex<BTreeMap<String, usize>>,
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
        let step = self.step.load(Ordering::SeqCst);
        self.landed.lock().unwrap().insert(key.to_string(), step);
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
        self.clean.delete(key).await
    }

    /// The files there are, and the lowest position of any that landed in
    /// or after the hour `from`.
    async fn files(&self, from: usize) -> (BTreeSet<String>, Option<u64>) {
        let files = self.clean.list("").await.unwrap();
        let landed = self.landed.lock().unwrap();
        let young = files
            .iter()
            .filter(|file| landed.get(&file.key).is_some_and(|hour| *hour >= from))
            .filter_map(|file| position(&file.key))
            .min();
        (files.into_iter().map(|file| file.key).collect(), young)
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
        let fault = self.faults.lock().unwrap().puts.pop_front();
        match fault {
            None => self.land(key, bytes).await,
            Some(PutFault::Refuse) => Err(lost()),
            Some(PutFault::LandThenFail) => self.land(key, bytes).await.and(Err(lost())),
            Some(PutFault::Delay) => {
                self.delayed.lock().unwrap().push((key.to_string(), bytes));
                Err(lost())
            }
        }
    }

    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        tokio::task::yield_now().await;
        let failing = {
            let mut faults = self.faults.lock().unwrap();
            let failing = faults.failing_gets > 0;
            faults.failing_gets = faults.failing_gets.saturating_sub(1);
            failing
        };
        if failing {
            return Err(lost());
        }
        self.clean.get(key).await
    }

    async fn list(
        &self,
        prefix: &str,
    ) -> vellev_storage::Result<Vec<vellev_storage::blobs::Listed>> {
        tokio::task::yield_now().await;
        let mut listed = self.clean.list(prefix).await?;
        let (now, step) = (Utc::now(), self.step.load(Ordering::SeqCst));
        let landed = self.landed.lock().unwrap();
        for file in &mut listed {
            // Half an hour more than the hours passed, so that a file as
            // many hours old as a clean's grace is older than it whenever
            // the cleaner reads the time, and not younger when it reads
            // it within the moment this was read.
            let steps = step - landed.get(&file.key).copied().unwrap_or(step);
            file.modified = now - STEP * steps as u32 - STEP / 2;
        }
        Ok(listed)
    }
    /// A delete is answered only once nothing else has anything to do, on
    /// the simulation's clock, which moves only then: whatever may run
    /// while files are being deleted does, to its end. One the store is to
    /// hold is answered as lost, and made when the held deletes are.
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.asked.fetch_add(1, Ordering::SeqCst);
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
            return Err(lost());
        }
        asked.key = None;
        if self.canary == Some(Canary::RefusesDeletes) {
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
        self.clean.forget_history(key, window).await
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
    DeliverDelayed,
    /// The store answers so many deletes and then holds one, which whoever
    /// asked for it is told was lost.
    DelayDelete(usize),
    /// The store makes the deletes that were given up.
    DeliverDelayedDeletes,
    /// So many hours pass with nothing done.
    Wait(usize),
    /// The coordinator forgets the commits no restore may go back to, as a
    /// node has it do before it cleans.
    ForgetHistory,
    Follow,
    FollowStale(usize),
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
    let delete = prop_oneof![2 => Just(None), 1 => any::<usize>().prop_map(Some)];
    let beside = prop_oneof![
        3 => Just(None),
        1 => any::<bool>().prop_map(|first| Some((Beside::Flush, first))),
        1 => any::<bool>().prop_map(|first| Some((Beside::Merge, first))),
        2 => any::<bool>().prop_map(|first| Some((Beside::Clean, first))),
    ];
    let write = |(who, rows, delete)| Action::Write { who, rows, delete };
    let conditional = |(who, pick, kind, fresh)| Action::Conditional {
        who,
        pick,
        kind,
        fresh,
    };
    let restore = |(who, back, beside)| Action::Restore { who, back, beside };
    prop_oneof![
        14 => (who(), 1..4u64, delete).prop_map(write),
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
        1 => Just(Action::DeliverDelayed),
        1 => (0..6usize).prop_map(Action::DelayDelete),
        1 => Just(Action::DeliverDelayedDeletes),
        1 => (1..GRACE).prop_map(Action::Wait),
        2 => Just(Action::ForgetHistory),
        3 => Just(Action::Follow),
        1 => any::<usize>().prop_map(Action::FollowStale),
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

/// A head the coordinator recorded, the hour it was recorded in, and the
/// rows committed at it.
#[derive(Debug)]
struct Recorded {
    head: Head,
    hour: usize,
    rows: Rows,
}

/// Where, in `heads`, those to be had still begin, when those `within` are
/// and the one before them is, which is how the partition stood when they
/// began: what the coordinator lets a restore go back to, of commits and a
/// window, and what a clean keeps, of files and a grace.
fn reach(heads: &[Recorded], within: impl Fn(&Recorded) -> bool) -> usize {
    heads
        .partition_point(|head| !within(head))
        .saturating_sub(1)
}

/// How far back a restore may go, as the coordinator is asked.
const REACH: Duration = Duration::from_secs(STEP.as_secs() * WINDOW as u64);

/// What is in the way of a clean.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Way {
    /// A fault to come, or a writer not the partition's own, not open or
    /// not abreast of what is committed: the clean may fail or not.
    Hindered,
    /// Nothing: the clean must succeed.
    Clear,
    /// Nothing but DW-680's finding F2, by which the clean fails for a
    /// head missing: see `World::walks_into_what_was_cleaned`.
    KnownToFail,
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
    /// Every head there has been, each a state to go back to, with the
    /// step it was first seen at.
    points: Vec<(usize, Committed)>,
    /// Every head the coordinator recorded, oldest first: `points` and
    /// whatever was committed and then committed over within one step.
    heads: Vec<Recorded>,
    /// How many of them the coordinator was last told to forget.
    forgotten: usize,
    /// The hour from which every clean so far was to keep what was
    /// committed, with the head before it: the latest any of them began
    /// its grace, as a clean with none begins it after everything.
    kept_from: usize,
    /// How many heads had been loaded when those in reach last were, and
    /// how many deletes the store had made.
    loaded: (usize, usize),
    /// The step at which the store last deleted anything.
    cleaned_at: Option<usize>,
    step: usize,
    made: Made,
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
    /// Cleans that had nothing in their way and so had to succeed, and
    /// those known to fail all the same, which did not have to.
    unhindered: usize,
    excused: usize,
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
            excused: self.excused + other.excused,
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

type Violation = std::result::Result<(), String>;

fn check(holds: bool, violation: impl FnOnce() -> String) -> Violation {
    holds.then_some(()).ok_or_else(violation)
}

impl World {
    async fn new(canary: Option<Canary>) -> Self {
        let cluster = Cluster::new().await;
        let faults = Arc::new(Mutex::new(Faults::default()));
        let coordinator = Arc::new(SimCoordinator {
            clean: cluster.coordinator.clone(),
            clock: cluster.clock.clone(),
            heads: Mutex::default(),
            faults: faults.clone(),
            canary,
            recorded: AtomicUsize::new(0),
            restores: AtomicUsize::new(0),
            unfenced: Mutex::default(),
        });
        let blobs = Arc::new(SimBlobs {
            clean: cluster.blobs.clone(),
            faults: faults.clone(),
            canary,
            written: Mutex::default(),
            delayed: Mutex::default(),
            rewritten: Mutex::default(),
            held: Mutex::default(),
            unanswered: AtomicUsize::new(0),
            asked: AtomicUsize::new(0),
            deleted: AtomicUsize::new(0),
            step: AtomicUsize::new(0),
            landed: Mutex::default(),
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
            .merging_with(MERGING);
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
            points: Vec::new(),
            heads: Vec::new(),
            forgotten: 0,
            kept_from: 0,
            loaded: (0, 0),
            cleaned_at: None,
            step: 0,
            made: Made::default(),
        };
        world.verify().await.unwrap();
        world
    }

    /// The lease expires and a new node takes it; every earlier writer is
    /// now a zombie, whether or not the new one manages to open.
    async fn take_over(&mut self) {
        let (lease, committed) = self.cluster.lease().await;
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
                let writer = writer.merging_with(MERGING);
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

    async fn write(&mut self, who: Who, rows: u64, delete: Option<usize>) -> Violation {
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
        self.attempt(who, self.attempts.len() - 1).await
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
        self.attempt(who, self.attempts.len() - 1).await
    }

    /// Make the request of attempt `made`, for the first time or again. One
    /// recorded already is answered and makes nothing; one refused stays
    /// refused; any other is made now if its commit is recorded, or
    /// refused for good by that commit, which is what a conditional
    /// request whose condition does not hold comes to.
    async fn attempt(&mut self, who: Who, made: usize) -> Violation {
        let seat = self.seat(who).await;
        let before = self.coordinator.recorded.load(Ordering::SeqCst);
        let batch = self.attempts[made].batch.clone();
        let writer = &self.seats[seat].writer;
        let outcome = writer.commit_batches(vec![batch], Embedded::new()).await;
        let outcome = outcome.and_then(|mut answers| answers.pop().expect("one batch"));
        let committed = self.coordinator.recorded.load(Ordering::SeqCst) > before;
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
        let within = |(step, _): &&(usize, Committed)| step + WINDOW >= self.step;
        let Some((seen_at, point)) = self.points.iter().rev().filter(within).nth(back).cloned()
        else {
            return Ok(());
        };
        // Only the partition's own writer cleans with no grace.
        let grace = match who {
            Who::Current => Duration::ZERO,
            Who::Any(_) => STEP * GRACE as u32,
        };
        let cleaning = matches!(beside, Some((Beside::Clean, _))).then_some(grace);
        let made_anyway = self.blobs.canary == Some(Canary::RestoresUnderALateDelete);
        if self.taken_by_a_late_delete(&point, cleaning) && !made_anyway {
            self.made.avoided += 1;
            return Ok(());
        }
        let under = !self.blobs.held.lock().unwrap().is_empty();
        let way = self.way(seat, grace).await;
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
        let given = self.given(grace);
        let writer = &self.seats[seat].writer;
        let restore = Restore {
            request: RequestId::generate(),
            asked: [0; 32],
            from: point.log_position,
            from_at: Utc::now(),
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
        if let Some(grace) = cleaning {
            self.cleaned(grace, asked);
        }
        let recorded = self.coordinator.restores.load(Ordering::SeqCst) > before;
        self.seats[seat].open = !self.seats[seat].writer.is_closed();
        if cleaning.is_some() {
            Self::came_to(way, &cleaned)?;
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
            self.base = self.history[&point.log_position].clone();
            self.since.clear();
        }
        Ok(())
    }

    /// What a clean asked to give `grace` is given.
    fn given(&self, grace: Duration) -> Duration {
        match self.blobs.canary {
            Some(Canary::CleansWithNoGrace) => Duration::ZERO,
            _ => grace,
        }
    }

    /// Note what a clean with `grace` was to keep, if it asked the store
    /// to delete anything, as the store had been `asked` so many times
    /// before it: what was committed in the hours of its grace, of which
    /// this is one, and the head before them.
    fn cleaned(&mut self, grace: Duration, asked: usize) {
        if self.blobs.deletes().0 > asked {
            self.kept_from = self.kept_from.max(Self::keeping_from(self.step, grace));
        }
    }

    fn keeping_from(hour: usize, grace: Duration) -> usize {
        (hour + 1).saturating_sub((grace.as_secs() / STEP.as_secs()) as usize)
    }

    /// Where in `heads` those begin that a restore may go to by the
    /// coordinator's clock.
    fn in_reach(&self) -> usize {
        let start = self.cluster.clock.now() - REACH;
        reach(&self.heads, |recorded| recorded.head.at >= start)
    }

    /// Whether a restore to `point`, beside a clean with the grace of
    /// `cleaning` if beside one, would have what it puts back into use
    /// taken by a delete that lands late.
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
    /// for, and no fault of the writer. The restore to such a state is not
    /// made while a delete is on its way, or may come to be beside it;
    /// every other is, whatever is on its way.
    fn taken_by_a_late_delete(&self, point: &Committed, cleaning: Option<Duration>) -> bool {
        let pending = !self.blobs.held.lock().unwrap().is_empty();
        let may_be = cleaning.is_some() && self.faults.lock().unwrap().held_delete.is_some();
        let kept_from = cleaning.map_or(self.kept_from, |grace| {
            self.kept_from.max(Self::keeping_from(self.step, grace))
        });
        let kept = reach(&self.heads, |recorded| recorded.hour >= kept_from);
        let at = |recorded: &Recorded| recorded.head.position == point.log_position;
        let unkept = self.heads.iter().position(at).is_none_or(|at| at < kept);
        (pending || may_be) && unkept
    }

    /// What is in the way of a clean from `seat` with `grace`.
    async fn way(&mut self, seat: usize, grace: Duration) -> Way {
        let faulty = {
            let faults = self.faults.lock().unwrap();
            let failing = !faults.puts.is_empty() || !faults.commits.is_empty();
            failing || faults.failing_gets > 0 || faults.held_delete.is_some()
        };
        let writer = &self.seats[seat].writer;
        let at = writer.partition().read(State::position);
        let abreast = writer.lease().generation == self.held.generation
            && self.cluster.committed().await.log_position == at;
        if faulty || !abreast || seat + 1 != self.seats.len() || writer.is_closed() {
            return Way::Hindered;
        }
        let known = self.walks_into_what_was_cleaned(grace).await;
        self.made.unhindered += usize::from(!known);
        self.made.excused += usize::from(known);
        match known {
            true => Way::KnownToFail,
            false => Way::Clear,
        }
    }

    /// See that a clean with `way` before it came to what it had to: with
    /// nothing in its way it succeeded, and where it is known to fail it
    /// failed just so, for a head missing, and for nothing else.
    fn came_to(way: Way, cleaned: &vellev_storage::Result<usize>) -> Violation {
        match (way, cleaned) {
            (Way::Hindered, _) | (Way::Clear, Ok(_)) => Ok(()),
            (Way::KnownToFail, Err(StorageError::Missing { .. })) => Ok(()),
            (Way::KnownToFail, Ok(_)) => {
                Err("a clean known to fail for a head missing succeeded".to_string())
            }
            (_, Err(error)) => Err(format!("a clean failed with nothing in its way: {error}")),
        }
    }

    /// Whether a clean with `grace` walks back into a head that has been
    /// deleted, and so fails, which is DW-680's finding F2. A clean walks
    /// from the newest head back to the last checkpoint before the oldest
    /// young file, log by log to the first checkpoint and checkpoint by
    /// checkpoint from there, and takes it that every one on the way is
    /// there. One an earlier clean deleted is not, and the walk reaches it
    /// whenever a file young by this clean's grace lies below the
    /// checkpoint the earlier one deleted up to: every clean then fails
    /// until that file is old. The walk is made here over the heads the
    /// coordinator recorded.
    ///
    /// The shortest such history has a clean keep more than the one before
    /// it, as an aged one does after one with no grace, and as a node's
    /// does once a tenant's window has been made longer.
    async fn walks_into_what_was_cleaned(&self, grace: Duration) -> bool {
        let young_from = Self::keeping_from(self.step, grace);
        let (files, young) = self.blobs.files(young_from).await;
        let heads = self.coordinator.heads.lock().unwrap();
        let newest = heads.last().map_or(0, |head| head.position.0);
        let floor = young.map_or(newest, |young| newest.min(young.saturating_sub(1)));
        let mut past_logs = false;
        for head in heads.iter().rev() {
            let checkpoint = head.key.0.ends_with(".manifest");
            if past_logs && !checkpoint {
                continue;
            }
            if !files.contains(&head.key.0) {
                return true;
            }
            if checkpoint && head.position.0 <= floor {
                break;
            }
            past_logs |= checkpoint;
        }
        false
    }

    /// Let `hours` pass, for the store and for the coordinator alike.
    fn pass(&mut self, hours: usize) {
        self.step += hours;
        self.blobs.step.store(self.step, Ordering::SeqCst);
        self.cluster.clock.advance(STEP * hours as u32);
    }

    async fn step(&mut self, action: Action) -> Violation {
        self.pass(1);
        match action {
            Action::Write { who, rows, delete } => return self.write(who, rows, delete).await,
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
                return self.attempt(who, pick % self.attempts.len()).await;
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
                let way = self.way(seat, grace).await;
                let cleaning = self.seats[seat].writer.clean(self.given(grace));
                let cleaned = cleaning.await;
                Self::came_to(way, &cleaned)?;
                let cleaned = cleaned.unwrap_or(0);
                self.cleaned(grace, asked);
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
                hour: self.step,
                rows: rows.clone(),
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

    /// See that the coordinator lists for a restore just the heads in
    /// reach: every head recorded within the window, as its clock has it,
    /// and the one before them, whatever it has been told to forget.
    async fn listed(&self) -> Violation {
        let reach = self.in_reach();
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

    /// See that every head a restore may go to loads, from a healthy
    /// store, as the rows committed at it: those in the coordinator's
    /// reach that every clean so far was to keep. A clean with no grace,
    /// which no node makes, keeps only the newest, and takes the others
    /// out of this until they have passed out of reach; a node forgets
    /// what is out of the tenant's window and then cleans with that window
    /// and more for grace, and by that every head in reach is kept.
    ///
    /// A file once written is never written again, so a head that loaded
    /// loads until the store deletes something: each is loaded when it is
    /// first seen and again whenever a delete has been made since. The
    /// newest is `committed`, read already as the truth.
    async fn reload(&mut self, committed: &Committed) -> Violation {
        let kept = reach(&self.heads, |recorded| recorded.hour >= self.kept_from);
        let kept = kept.max(self.in_reach());
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
                        "a restore point within reach, {:?}, cannot be loaded: {error}",
                        recorded.head
                    )
                })?;
            let (position, rows) = loaded.partition().read(snapshot);
            check(
                position == recorded.head.position && rows == recorded.rows,
                || {
                    format!(
                        "a restore point within reach, {:?}, loads at {position:?} as {rows:?}, \
                         and {:?} was committed at it",
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
        if self.points.last().map(|(_, point)| point) != Some(&committed) {
            self.points.push((self.step, committed.clone()));
        }
        self.record(&expected, &committed)?;
        self.listed().await?;
        self.reload(&committed).await?;
        let unanswered = self.blobs.unanswered.load(Ordering::SeqCst);
        check(unanswered == 0 || self.made.delays > 0, || {
            format!("{unanswered} deletes were given up with none of them held")
        })?;

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
        })
    }
}

async fn simulate(actions: Vec<Action>, canary: Option<Canary>) -> Result<Made, String> {
    let mut world = World::new(canary).await;
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
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap()
        .block_on(simulate(actions, canary))
}

fn histories() -> impl Strategy<Value = Vec<Action>> {
    prop::collection::vec(action(), 1..80)
}

proptest! {
    // Four times as many histories as by default: some faults want a
    // writer left behind, a restore and a clean in just the right order.
    #![proptest_config(ProptestConfig::with_cases(1_024))]

    #[test]
    fn no_history_of_faults_loses_an_acknowledged_write_or_shows_an_unrecorded_one(
        actions in histories()
    ) {
        run(actions, None).map_err(TestCaseError::fail)?;
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
    let _ = TestRunner::new_with_rng(config, rng).run(&histories(), |actions| {
        let outcome = run(actions, canary);
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
    assert!(
        made.unhindered > 300 && made.unhindered > 4 * made.excused,
        "{made:?}"
    );

    // A commit acknowledged without being recorded shows up as whichever of
    // its symptoms the history reaches first.
    let symptoms = ["never recorded", "never committed", "was told the counter"];
    let (unrecorded, _) = first_violation(Some(Canary::AcknowledgesUnrecordedCommits));
    let unrecorded = unrecorded.unwrap();
    assert!(
        symptoms.iter().any(|symptom| unrecorded.contains(symptom)),
        "{unrecorded}"
    );

    let dropped = first_violation(Some(Canary::DropsSegments)).0.unwrap();
    assert!(dropped.contains("cannot be read"), "{dropped}");

    let cleaned = first_violation(Some(Canary::CleansWithNoGrace)).0.unwrap();
    // A clean that keeps too little is caught by the head it left a restore
    // unable to go to, not by the newest, which it keeps.
    assert!(
        cleaned.contains("within reach") && cleaned.contains("cannot be loaded"),
        "{cleaned}"
    );

    let forgotten = first_violation(Some(Canary::ForgetsWithinTheWindow))
        .0
        .unwrap();
    assert!(
        forgotten.contains("is not listed by the coordinator"),
        "{forgotten}"
    );

    // The store keeps every key it was ever given, deleted since or not.
    let rewritten = first_violation(Some(Canary::WritesAgainAfterALateDelete))
        .0
        .unwrap();
    assert!(rewritten.contains("written twice"), "{rewritten}");

    let refused = first_violation(Some(Canary::RefusesDeletes)).0.unwrap();
    assert!(
        refused.contains("a clean failed with nothing in its way"),
        "{refused}"
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

/// Known to fail, and DW-680's finding F2: a clean that is to keep more
/// than an earlier one fails for a checkpoint the earlier one deleted. The
/// first clean, with no grace, deletes every checkpoint before the newest;
/// the second, with an hour's, finds the first segment young, which lies
/// below the newest checkpoint, walks back past that checkpoint to the one
/// deleted, and fails. So does every clean after it until that segment is
/// an hour old. A node meets it when a tenant's window is made longer, and
/// when a restore fetches files back under their old positions.
///
/// Once it passes, every clean with nothing in its way is to be held to
/// succeeding, and `Way::KnownToFail` is to go.
#[test]
#[should_panic(expected = "DW-680 F2: a clean failed with nothing in its way")]
fn a_clean_that_keeps_more_than_the_last_fails_for_what_the_last_deleted() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();
    runtime.block_on(async {
        let who = Who::Current;
        let mut world = World::new(None).await;
        let write = Action::Write {
            who,
            rows: 1,
            delete: None,
        };
        for action in [Action::Flush { who }, write, Action::Flush { who }] {
            world.step(action).await.unwrap();
            world.verify().await.unwrap();
        }
        let writer = &world.seats[0].writer;
        assert!(writer.clean(Duration::ZERO).await.unwrap() > 0);
        if let Err(error) = writer.clean(STEP * GRACE as u32).await {
            panic!("DW-680 F2: a clean failed with nothing in its way: {error}");
        }
    });
}
