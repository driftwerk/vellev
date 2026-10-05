//! A deterministic simulation of one partition under failure.
//!
//! Random histories interleave writes from a current writer and from stalled
//! ones, checkpoints, takeovers and follower catch-ups with faults in the
//! store and the coordinator. After every step the committed state is read
//! through fault-free handles and checked against what the coordinator
//! really recorded, whatever the writers were told.

mod common;

use std::collections::VecDeque;

use common::*;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
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
    None,
    /// The coordinator says yes to a commit it did not record.
    AcknowledgesUnrecordedCommits,
    /// The store accepts an upload and keeps nothing.
    DropsUploads,
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
    canary: Canary,
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
            Canary::DropsUploads if key.contains("/segment/") => Ok(()),
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
}

/// A coordinator that fails on command and counts the commits it recorded.
#[derive(Debug)]
struct SimCoordinator {
    clean: Arc<MemoryCoordinator>,
    faults: Arc<Mutex<Faults>>,
    canary: Canary,
    recorded: AtomicUsize,
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
        let unreachable = || CoordinatorError::backend(std::io::Error::other("connection reset"));
        let fault = self.faults.lock().unwrap().commits.pop_front();
        if fault == Some(CommitFault::Refuse) {
            return Err(unreachable());
        }
        let outcome = self.clean.commit(lease, commit.clone()).await;
        match outcome {
            Ok(_) => {
                self.recorded.fetch_add(1, Ordering::SeqCst);
            }
            Err(_) if self.canary == Canary::AcknowledgesUnrecordedCommits => {
                return Ok(Committed {
                    generation: lease.generation,
                    log_position: commit.log_position,
                    manifest: commit.manifest,
                });
            }
            Err(_) => {}
        }
        match fault {
            Some(CommitFault::LoseResponse) => Err(unreachable()),
            _ => outcome,
        }
    }
    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>, CoordinatorError> {
        self.clean.committed(key).await
    }
    async fn ping(&self) -> Result<(), CoordinatorError> {
        self.clean.ping().await
    }
}

#[derive(Debug, Clone)]
enum Action {
    /// A writer, counted back from the newest, inserts this many rows.
    Write {
        writer: usize,
        rows: u64,
    },
    Flush {
        writer: usize,
    },
    Takeover,
    /// The newest writer's node restarts it on the lease it still holds.
    Reopen,
    FailUpload(PutFault),
    FailCommits(CommitFault, usize),
    FailReads(usize),
    DeliverDelayed,
    Follow,
    FollowStale(usize),
}

fn action() -> impl Strategy<Value = Action> {
    let writer = || prop_oneof![4 => Just(0usize), 1 => 1..4usize];
    let put_fault = prop_oneof![
        Just(PutFault::Refuse),
        Just(PutFault::LandThenFail),
        Just(PutFault::Delay)
    ];
    let commit_fault = prop_oneof![Just(CommitFault::Refuse), Just(CommitFault::LoseResponse)];
    prop_oneof![
        12 => (writer(), 1..4u64).prop_map(|(writer, rows)| Action::Write { writer, rows }),
        3 => writer().prop_map(|writer| Action::Flush { writer }),
        3 => Just(Action::Takeover),
        2 => Just(Action::Reopen),
        2 => put_fault.prop_map(Action::FailUpload),
        3 => (commit_fault, 1..5usize).prop_map(|(fault, times)| Action::FailCommits(fault, times)),
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
        .map(|object| match object.properties.get("n") {
            Some(Value::Int(n)) => (object.id, *n),
            other => panic!("{other:?}"),
        })
        .collect();
    (state.position(), rows)
}

struct World {
    key: PartitionKey,
    clock: Arc<ManualClock>,
    faults: Arc<Mutex<Faults>>,
    blobs: Arc<SimBlobs>,
    coordinator: Arc<SimCoordinator>,
    writers: Vec<Writer>,
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
    async fn new(canary: Canary) -> Self {
        let clock = Arc::new(ManualClock::starting_at(
            Utc.with_ymd_and_hms(2026, 10, 5, 9, 0, 0).unwrap(),
        ));
        let faults = Arc::new(Mutex::new(Faults::default()));
        let key = PartitionKey {
            tenant: TenantId::new("sim").unwrap(),
            partition: PartitionNumber::DEFAULT,
        };
        let clean = Arc::new(MemoryCoordinator::new(clock.clone()));
        clean.register(&key).await.unwrap();
        let coordinator = Arc::new(SimCoordinator {
            clean,
            faults: faults.clone(),
            canary,
            recorded: AtomicUsize::new(0),
        });
        let blobs = Arc::new(SimBlobs {
            clean: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
            faults: faults.clone(),
            canary,
            written: Mutex::default(),
            delayed: Mutex::default(),
            rewritten: Mutex::default(),
        });
        let empty = coordinator.clean.committed(&key).await.unwrap().unwrap();
        let follower = Follower::open(key.clone(), blobs.clone(), &empty)
            .await
            .unwrap();
        let mut world = Self {
            key,
            clock,
            faults,
            blobs,
            coordinator,
            writers: Vec::new(),
            follower,
            attempts: Vec::new(),
            readings: Vec::new(),
            next_row: 0,
            history: BTreeMap::from([(LogPosition(0), BTreeMap::new())]),
        };
        world.take_over().await;
        let setup = vec![
            create(COLLECTION),
            Mutation::Insert {
                collection: name(COLLECTION),
                id: counter(),
                properties: json!({ "n": 0 }).as_object().unwrap().clone(),
                vector: None,
            },
        ];
        world.writers[0].commit(setup).await.unwrap();
        world.verify().await.unwrap();
        world
    }

    async fn committed(&self) -> Committed {
        self.coordinator
            .clean
            .committed(&self.key)
            .await
            .unwrap()
            .unwrap()
    }

    /// The lease expires and a new node takes it. Opening its writer reads
    /// through the faulty store, so it may fail; the lease has moved either
    /// way, and every earlier writer is now a zombie.
    async fn take_over(&mut self) {
        self.clock.advance(TTL);
        let holder = NodeId::generate("node");
        let Acquired::Lease { lease, committed } = self
            .coordinator
            .acquire(&self.key, &holder, TTL)
            .await
            .unwrap()
        else {
            panic!("the lease should have expired");
        };
        self.open_writer(lease, &committed).await;
    }

    async fn open_writer(&mut self, lease: Lease, committed: &Committed) {
        let opened = Writer::open(
            self.blobs.clone(),
            self.coordinator.clone(),
            lease,
            committed,
        );
        if let Ok(writer) = opened.await {
            self.writers.push(writer);
        }
    }

    fn writer(&self, back: usize) -> &Writer {
        &self.writers[self.writers.len() - 1 - back % self.writers.len()]
    }

    async fn write(&mut self, back: usize, rows: u64) {
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
        let outcome = self.writer(back).commit(mutations).await;
        let recorded = self.coordinator.recorded.load(Ordering::SeqCst) > before;
        let acknowledged = matches!(
            &outcome,
            Ok(receipt) if matches!(receipt.outcomes.last(), Some(Outcome::Value(_)))
        );
        self.attempts.push(Attempt {
            rows,
            acknowledged,
            recorded,
        });
    }

    async fn step(&mut self, action: Action) -> Violation {
        match action {
            Action::Write { writer, rows } => self.write(writer, rows).await,
            Action::Flush { writer } => {
                let _ = self.writer(writer).flush().await;
            }
            Action::Takeover => self.take_over().await,
            Action::Reopen => {
                let (lease, committed) = (self.writer(0).lease().clone(), self.committed().await);
                self.open_writer(lease, &committed).await;
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
        let committed = self.committed().await;
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
        Follower::open(self.key.clone(), self.blobs.clean.clone(), committed)
            .await
            .map(|fresh| fresh.partition().read(snapshot))
            .map_err(|error| format!("the committed state cannot be read: {error}"))
    }

    async fn verify(&mut self) -> Violation {
        let rewritten = self.blobs.rewritten.lock().unwrap().clone();
        check(rewritten.is_empty(), || {
            format!("files were written twice: {rewritten:?}")
        })?;

        let committed = self.committed().await;
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
        let recorded: Vec<&Attempt> = self.attempts.iter().filter(|a| a.recorded).collect();
        let expected: BTreeMap<ObjectId, i64> = self
            .attempts
            .iter()
            .enumerate()
            .filter(|(_, attempt)| attempt.recorded)
            .flat_map(|(n, attempt)| attempt.rows.iter().map(move |r| (row(*r), n as i64)))
            .chain([(counter(), recorded.len() as i64)])
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
            .writers
            .iter()
            .map(Writer::partition)
            .chain([self.follower.partition()]);
        copies.map(|copy| copy.read(snapshot)).try_for_each(|(position, held)| {
            check(self.history.get(&position) == Some(&held), || {
                format!("a copy at {position:?} holds rows that were never committed there: {held:?}")
            })
        })
    }
}

async fn simulate(actions: Vec<Action>, canary: Canary) -> Violation {
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

fn run(actions: Vec<Action>, canary: Canary) -> Violation {
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
        run(actions, Canary::None).map_err(TestCaseError::fail)?;
    }
}

/// Whether the simulation finds a violation when one of its dependencies is
/// broken in the given way.
fn caught(canary: Canary) -> bool {
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 2_000,
            failure_persistence: None,
            ..Config::default()
        },
        proptest::test_runner::TestRng::deterministic_rng(
            proptest::test_runner::RngAlgorithm::ChaCha,
        ),
    );
    runner
        .run(&histories(), |actions| {
            run(actions, canary).map_err(TestCaseError::fail)
        })
        .is_err()
}

#[test]
fn the_checker_passes_a_healthy_system_and_catches_a_broken_one() {
    assert!(!caught(Canary::None));
    assert!(caught(Canary::AcknowledgesUnrecordedCommits));
    assert!(caught(Canary::DropsUploads));
}
