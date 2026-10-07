//! A deterministic simulation of one partition under failure.
//!
//! Random histories interleave writes from a current writer and from stalled
//! ones, checkpoints, takeovers and follower catch-ups with faults in the
//! store and the coordinator. After every step the committed state is read
//! through fault-free handles and checked against what the coordinator
//! really recorded, whatever the writers were told.

mod common;

use std::collections::VecDeque;
use vellev_storage::coordinator::{KeyHash, Point, RestorePoint, TenantKey};

use common::*;
use proptest::test_runner::{Config, RngAlgorithm, TestCaseError, TestRng, TestRunner};
use vellev_storage::Outcome;

const COLLECTION: &str = "sim";

fn row(n: u64) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n)))
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
/// second write to one key is noticed.
#[derive(Debug)]
struct SimBlobs {
    clean: Arc<dyn Blobs>,
    faults: Arc<Mutex<Faults>>,
    canary: Option<Canary>,
    written: Mutex<BTreeSet<String>>,
    delayed: Mutex<Vec<(String, Bytes)>>,
    rewritten: Mutex<Vec<String>>,
}

impl SimBlobs {
    async fn land(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if !self.written.lock().unwrap().insert(key.to_string()) {
            self.rewritten.lock().unwrap().push(key.to_string());
        }
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
        self.clean.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
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
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<(), CoordinatorError> {
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

    async fn restore_window(
        &self,
        tenant: &TenantId,
    ) -> Result<Option<Duration>, CoordinatorError> {
        self.clean.restore_window(tenant).await
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

#[derive(Debug, Clone)]
enum Action {
    Write {
        who: Who,
        rows: u64,
    },
    Flush {
        who: Who,
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
    prop_oneof![
        14 => (who(), 1..4u64).prop_map(|(who, rows)| Action::Write { who, rows }),
        4 => who().prop_map(|who| Action::Flush { who }),
        2 => Just(Action::Takeover),
        1 => Just(Action::Reopen),
        2 => put_fault.prop_map(Action::FailUpload),
        2 => (commit_fault, times).prop_map(|(fault, times)| Action::FailCommits(fault, times)),
        1 => (1..4usize).prop_map(Action::FailReads),
        1 => Just(Action::DeliverDelayed),
        3 => Just(Action::Follow),
        1 => any::<usize>().prop_map(Action::FollowStale),
    ]
}

/// One attempt to commit: the rows it tried to insert, what the writer was
/// told, and whether the coordinator recorded it.
#[derive(Debug)]
struct Attempt {
    rows: Vec<u64>,
    acknowledged: bool,
    recorded: bool,
}

/// A partition's rows as a checker compares them: position, and `n` by id.
type Snapshot = (LogPosition, BTreeMap<ObjectId, i64>);

fn snapshot(state: &State) -> Snapshot {
    let rows = state
        .scan(&name(COLLECTION))
        .map(|found| match found.object.properties.get("n") {
            Some(Value::Int(n)) => (found.id(), *n),
            other => panic!("{other:?}"),
        })
        .collect();
    (state.position(), rows)
}

/// A writer, and whether it can still commit. Any failed write closes one.
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
    readings: Vec<Committed>,
    next_row: u64,
    /// What was committed at each position, as read when it was the newest.
    history: BTreeMap<LogPosition, BTreeMap<ObjectId, i64>>,
}

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
            unfenced: Mutex::default(),
        });
        let blobs = Arc::new(SimBlobs {
            clean: cluster.blobs.clone(),
            faults: faults.clone(),
            canary,
            written: Mutex::default(),
            delayed: Mutex::default(),
            rewritten: Mutex::default(),
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
            .unwrap();
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
            readings: Vec::new(),
            next_row: 0,
            history: BTreeMap::from([(LogPosition(0), BTreeMap::new())]),
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

    async fn write(&mut self, who: Who, rows: u64) -> Violation {
        let seat = self.seat(who).await;
        let rows: Vec<u64> = (self.next_row..self.next_row + rows).collect();
        self.next_row += rows.len() as u64;
        let attempt = self.attempts.len() as i64;
        let insert = |n: &u64| Mutation::Insert {
            collection: name(COLLECTION),
            id: row(*n),
            properties: json!({ "n": attempt }).as_object().unwrap().clone(),
            vector: None,
        };
        let count = Mutation::Increment {
            collection: name(COLLECTION),
            id: counter(),
            property: "n".into(),
            by: 1.0,
        };
        let mutations = rows.iter().map(insert).chain([count]).collect();
        let before = self.coordinator.recorded.load(Ordering::SeqCst);
        let outcome = self.seats[seat].writer.commit(mutations).await;
        let recorded = self.coordinator.recorded.load(Ordering::SeqCst) > before;
        self.seats[seat].open = outcome.is_ok();
        self.attempts.push(Attempt {
            rows,
            acknowledged: outcome.is_ok(),
            recorded,
        });

        let Ok(receipt) = outcome else { return Ok(()) };
        let counted = self
            .attempts
            .iter()
            .filter(|attempt| attempt.recorded)
            .count() as i64;
        check(
            receipt.outcomes.last() == Some(&Outcome::Value(Value::Int(counted))),
            || {
                format!(
                    "a write was told the counter is {:?} after {counted} recorded attempts",
                    receipt.outcomes.last()
                )
            },
        )
    }

    async fn step(&mut self, action: Action) -> Violation {
        match action {
            Action::Write { who, rows } => return self.write(who, rows).await,
            Action::Flush { who } => {
                let seat = self.seat(who).await;
                self.seats[seat].open = self.seats[seat].writer.flush().await.is_ok();
            }
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
        Follower::open(
            self.cluster.key.clone(),
            self.blobs.clean.clone(),
            committed,
        )
        .await
        .map(|fresh| fresh.partition().read(snapshot))
        .map_err(|error| format!("the committed state cannot be read: {error}"))
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
        let recorded = || {
            self.attempts
                .iter()
                .enumerate()
                .filter(|(_, attempt)| attempt.recorded)
        };
        let expected: BTreeMap<ObjectId, i64> = recorded()
            .flat_map(|(n, attempt)| attempt.rows.iter().map(move |r| (row(*r), n as i64)))
            .chain([(counter(), recorded().count() as i64)])
            .collect();
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

        let copies = self
            .seats
            .iter()
            .map(|seat| seat.writer.partition())
            .chain([self.follower.partition()]);
        copies.map(|copy| copy.read(snapshot)).try_for_each(|(position, held)| {
            check(self.history.get(&position) == Some(&held), || {
                format!("a copy at {position:?} holds rows that were never committed there: {held:?}")
            })
        })
    }
}

async fn simulate(actions: Vec<Action>, canary: Option<Canary>) -> Violation {
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
        .map_err(|violation| format!("at the end: {violation}"))
}

fn run(actions: Vec<Action>, canary: Option<Canary>) -> Violation {
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
    #[test]
    fn no_history_of_faults_loses_an_acknowledged_write_or_shows_an_unrecorded_one(
        actions in histories()
    ) {
        run(actions, None).map_err(TestCaseError::fail)?;
    }
}

/// The first violation the simulation reports over a fixed set of histories
/// when one of its dependencies is broken in the given way.
fn first_violation(canary: Option<Canary>) -> Option<String> {
    let config = Config {
        cases: 200,
        failure_persistence: None,
        max_shrink_iters: 0,
        ..Config::default()
    };
    let rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
    let found = Mutex::new(None);
    let _ = TestRunner::new_with_rng(config, rng).run(&histories(), |actions| {
        run(actions, canary).map_err(|violation| {
            found.lock().unwrap().get_or_insert(violation.clone());
            TestCaseError::fail(violation)
        })
    });
    found.into_inner().unwrap()
}

#[test]
fn the_checker_passes_a_healthy_system_and_catches_a_broken_one() {
    assert_eq!(first_violation(None), None);

    // A commit acknowledged without being recorded shows up as whichever of
    // its symptoms the history reaches first.
    let symptoms = ["never recorded", "never committed", "was told the counter"];
    let unrecorded = first_violation(Some(Canary::AcknowledgesUnrecordedCommits)).unwrap();
    assert!(
        symptoms.iter().any(|symptom| unrecorded.contains(symptom)),
        "{unrecorded}"
    );

    let dropped = first_violation(Some(Canary::DropsSegments)).unwrap();
    assert!(dropped.contains("cannot be read"), "{dropped}");
}
