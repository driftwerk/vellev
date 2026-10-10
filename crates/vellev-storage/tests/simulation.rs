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
//! A step stands for an hour in the age of a file: the store says each was
//! written as many hours ago as there have been steps since, so that
//! cleaning up with a grace of some hours deletes what is older and keeps
//! what is not, as it does where files age by the clock.

mod common;

use std::collections::VecDeque;
use vellev_core::{RequestId, Version};
use vellev_storage::coordinator::{KeyHash, Point, Restore, RestorePoint, Restored, TenantKey};

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
}

#[derive(Debug, Default)]
struct Faults {
    puts: VecDeque<PutFault>,
    commits: VecDeque<CommitFault>,
    failing_gets: usize,
}

/// A store that fails on command and records every file it was given, so a
/// second write to one key is noticed. Whoever asks anything of it waits a
/// moment for the answer, as of any store, so that what is done at once
/// takes turns.
#[derive(Debug)]
struct SimBlobs {
    clean: Arc<dyn Blobs>,
    faults: Arc<Mutex<Faults>>,
    canary: Option<Canary>,
    written: Mutex<BTreeSet<String>>,
    delayed: Mutex<Vec<(String, Bytes)>>,
    rewritten: Mutex<Vec<String>>,
    /// How many steps the history has taken, and at which each file landed.
    step: AtomicUsize,
    landed: Mutex<BTreeMap<String, usize>>,
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

    async fn deliver_delayed(&self) {
        let delayed = std::mem::take(&mut *self.delayed.lock().unwrap());
        for (key, bytes) in delayed {
            self.land(&key, bytes).await.unwrap();
        }
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
            let steps = step - landed.get(&file.key).copied().unwrap_or(step);
            file.modified = now - STEP * steps as u32;
        }
        Ok(listed)
    }
    /// A delete is answered only once nothing else has anything to do, on
    /// the simulation's clock, which moves only then: whatever may run
    /// while files are being deleted does, to its end.
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        tokio::time::sleep(Duration::from_millis(1)).await;
        self.clean.delete(key).await
    }
}

/// A coordinator that fails on command, counts the commits it recorded, and
/// notes any it recorded for a lease that was no longer the current one.
#[derive(Debug)]
struct SimCoordinator {
    clean: Arc<MemoryCoordinator>,
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
    /// The step at which a clean last deleted anything.
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

    /// Flush, merge or clean from `writer`, and say how many files a clean
    /// deleted.
    async fn beside(writer: &Writer, beside: Beside, grace: Duration) -> (usize, usize) {
        match beside {
            Beside::Flush => writer.flush().await.map_or((0, 0), |_| (0, 0)),
            Beside::Merge => (usize::from(matches!(writer.merge().await, Ok(Some(_)))), 0),
            Beside::Clean => (0, writer.clean(grace).await.unwrap_or(0)),
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
        let writer = &self.seats[seat].writer;
        let restore = Restore {
            request: RequestId::generate(),
            asked: [0; 32],
            from: point.log_position,
            from_at: Utc::now(),
        };
        let restoring = writer.restore_recorded(point.manifest.as_ref(), restore, None);
        // Only the partition's own writer cleans with no grace.
        let grace = match who {
            Who::Current => Duration::ZERO,
            Who::Any(_) => STEP * GRACE as u32,
        };
        let (outcome, (merged, cleaned)) = match beside {
            None => (restoring.await, (0, 0)),
            Some((beside, false)) => tokio::join!(restoring, Self::beside(writer, beside, grace)),
            Some((beside, true)) => {
                let (made, outcome) = tokio::join!(Self::beside(writer, beside, grace), restoring);
                (outcome, made)
            }
        };
        let recorded = self.coordinator.restores.load(Ordering::SeqCst) > before;
        self.seats[seat].open = !self.seats[seat].writer.is_closed();
        self.made.merges += merged;
        self.made.cleaned += cleaned;
        self.made.raced += usize::from(beside.is_some() && recorded);
        if cleaned > 0 {
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

    async fn step(&mut self, action: Action) -> Violation {
        self.step += 1;
        self.blobs.step.store(self.step, Ordering::SeqCst);
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
                let cleaned = self.seats[seat].writer.clean(grace).await.unwrap_or(0);
                match grace.is_zero() {
                    true => self.made.cleaned += cleaned,
                    false => self.made.aged += cleaned,
                }
                if cleaned > 0 {
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
            self.points.push((self.step, committed));
        }

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
    world
        .verify()
        .await
        .map_err(|violation| format!("after late uploads arrived: {violation}"))?;
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
        cases: 200,
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
                found.1.merges += made.merges;
                found.1.cleaned += made.cleaned;
                found.1.aged += made.aged;
                found.1.restores += made.restores;
                found.1.raced += made.raced;
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
}
