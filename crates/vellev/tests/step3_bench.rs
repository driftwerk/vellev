//! The step 3 bench (merge note, §3 and §9): whether merges built on a
//! pool of their own, smaller or lower in priority, are worth it once the
//! building lock no longer holds flushes behind them.
//!
//! Each arm is two processes, as `scripts/step3-bench/run.sh` runs them,
//! each the same in every arm:
//! - `step3_arm`, the node: a whole node as the image runs one (its
//!   allocator, its HTTP API, an object store behind a cache directory),
//!   six tenants preloaded the same way from the bench fixture, then held
//!   to a memory budget it is always over through a soak of half an hour
//!   or more. It samples its own memory and CPU, alone: no load is made
//!   in it.
//! - `step3_load`, the load generator, given the node's address through a
//!   file the node writes: readers at twice the cores back to back, and a
//!   thousand rows a second written over a set of tenants that moves on
//!   by the batch, never by the clock.
//!
//! Every mark is worked out from exact durations, never from a
//! histogram's buckets. `step3_compare` then sets the arms side by side,
//! against the marks fixed before the run, comparing only arms run alike.
//!
//! Ignored: it takes the fixture and the better part of an hour. Run as
//! `scripts/step3-bench/run.sh` says.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use object_store::local::LocalFileSystem;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use vellev::server::{App, Limits};
use vellev_bench::fixture::{Fixture, Row};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_embed::Embedders;
use vellev_node::{MergePool, Node, Settings};
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::{Coordinator, MemoryCoordinator};
use vellev_storage::timings::{Event, Holder, MergePhase, OpenWaits, Timed};
use vellev_storage::{Disk, Mutation, Transfers, MERGE_BUILDS};

/// The node's allocator, as the binary runs on it.
#[global_allocator]
static GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

/// And its options, as the binary sets them.
#[cfg(target_os = "linux")]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static OPTIONS: vellev::allocator::Options =
    vellev::allocator::Options(vellev::allocator::OPTIONS.as_ptr());

const KEY: &str = "step3-bench-key";
const COLLECTION: &str = "passages";
const DIMENSIONS: usize = 768;
/// The uid the image runs as.
const IMAGE_UID: u32 = 10001;
/// CAP_SYS_NICE's bit in a capability set.
const CAP_SYS_NICE: u32 = 23;
/// How far below the process the rows passes' thread runs.
const PASSES_NICE: i64 = 10;

/// An arm of §3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// Merges on the global pool, at the process's priority: the control.
    A,
    /// A pool of half the threads, at the process's priority: the baseline.
    B,
    /// A pool of every thread, 5 lower.
    C,
    /// A pool of every thread, 10 lower.
    D,
    /// (b) run again, its spread from (b) the noise floor of the runs.
    B2,
}

impl Arm {
    fn named(name: &str) -> Arm {
        match name {
            "a" => Arm::A,
            "b" => Arm::B,
            "c" => Arm::C,
            "d" => Arm::D,
            "b2" => Arm::B2,
            other => panic!("STEP3_ARM is a, b, c, d or b2, not {other:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Arm::A => "a",
            Arm::B => "b",
            Arm::C => "c",
            Arm::D => "d",
            Arm::B2 => "b2",
        }
    }

    fn pool(self, threads: usize) -> MergePool {
        match self {
            Arm::A => MergePool::Global,
            Arm::B | Arm::B2 => MergePool::Own {
                threads: (threads / 2).max(1),
                nice: 0,
            },
            Arm::C => MergePool::Own { threads, nice: 5 },
            Arm::D => MergePool::Own { threads, nice: 10 },
        }
    }
}

/// Every arm is a pool `Settings::check` would let through on the machines
/// §9 runs them on, as the bench itself makes no use of that check (its
/// grace is below the floor, §9.5) and asks this before it preloads. One
/// thread is too few to share among the builds, for any arm of its own.
#[test]
fn every_arm_is_a_pool_a_node_could_run() {
    for arm in [Arm::A, Arm::B, Arm::C, Arm::D, Arm::B2] {
        for threads in [4, 16] {
            assert_eq!(arm.pool(threads).problem(MERGE_BUILDS), None, "{arm:?}");
        }
    }
    assert!(Arm::B.pool(1).problem(MERGE_BUILDS).is_some());
    assert_eq!(Arm::A.pool(1).problem(MERGE_BUILDS), None);
}

/// How the bench is run, from the environment.
#[derive(Clone, Debug)]
struct Bench {
    arm: Arm,
    fixture: PathBuf,
    out: PathBuf,
    data: PathBuf,
    soak: Duration,
    settle: Duration,
    rows_a_second: usize,
    batch: usize,
    readers: usize,
    threads: usize,
    flush_rows: usize,
    budget_share: f64,
    scale: f64,
    active: usize,
    rotate: Duration,
    window: Duration,
    as_image: bool,
    seed: u64,
    /// The commit the bench was built from, as the script found it.
    sha: String,
    /// Whether the script ran this process in a systemd scope of its own:
    /// `None` if it did not say.
    scope: Option<bool>,
    /// How long the load generator waits for the node to be ready.
    ready_within: Duration,
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} cannot be read: {value:?}")),
        Err(_) => default,
    }
}

impl Bench {
    fn from_env() -> Bench {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let path = |name: &str| {
            PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
        };
        Bench {
            arm: Arm::named(&std::env::var("STEP3_ARM").expect("STEP3_ARM must be set")),
            fixture: path("STEP3_FIXTURE"),
            out: path("STEP3_OUT"),
            data: path("STEP3_DATA"),
            soak: Duration::from_secs(env("STEP3_SOAK_SECS", 1_800)),
            settle: Duration::from_secs(env("STEP3_SETTLE_SECS", 300)),
            rows_a_second: env("STEP3_ROWS_A_SECOND", 1_000),
            batch: env("STEP3_BATCH", 100),
            readers: env("STEP3_READERS", 2 * threads),
            threads,
            flush_rows: env("STEP3_FLUSH_ROWS", 10_000),
            budget_share: env("STEP3_BUDGET_SHARE", 0.5),
            scale: env("STEP3_SCALE", 1.0),
            active: env("STEP3_ACTIVE", 3),
            rotate: Duration::from_secs(env("STEP3_ROTATE_SECS", 60)),
            window: Duration::from_secs(env("STEP3_WINDOW_SECS", 300)),
            as_image: env("STEP3_AS_IMAGE", 0u8) == 1,
            seed: env("STEP3_SEED", 0x5eed_0003),
            sha: std::env::var("STEP3_SHA").unwrap_or_default(),
            scope: std::env::var("STEP3_SCOPE").ok().map(|scope| scope == "1"),
            ready_within: Duration::from_secs(env("STEP3_READY_WITHIN_SECS", 2 * 60 * 60)),
        }
    }

    /// Whether this is the bench as §3 and §9.5 run it, not a smoke of it:
    /// every value the workload is fixed at, pinned.
    fn is_measurement(&self) -> bool {
        self.soak >= Duration::from_secs(1_800)
            && self.scale == 1.0
            && self.rows_a_second == 1_000
            && self.readers == 2 * self.threads
            && self.flush_rows == 10_000
            && self.budget_share == 0.5
            && self.active == 3
            && self.rotate == Duration::from_secs(60)
            && self.batch == 100
    }

    /// How the bench was set to run, `arm` among it: what two arms must
    /// share, `arm` apart, for their marks to be compared. What a run
    /// measured, as how long its preload took, is not among it.
    fn config(&self) -> Value {
        json!({
            "arm": self.arm.name(),
            "soak_s": self.soak.as_secs(), "settle_s": self.settle.as_secs(),
            "rows_a_second": self.rows_a_second, "batch": self.batch, "readers": self.readers,
            "threads": self.threads, "flush_rows": self.flush_rows, "scale": self.scale,
            "budget_share": self.budget_share, "active": self.active,
            "rotate_s": self.rotate.as_secs(), "window_s": self.window.as_secs(),
            "as_image": self.as_image, "seed": self.seed,
        })
    }

    /// The bench as §9.5 measures it, for the rules.
    #[cfg(test)]
    fn measured() -> Bench {
        Bench {
            arm: Arm::B,
            fixture: PathBuf::new(),
            out: PathBuf::new(),
            data: PathBuf::new(),
            soak: Duration::from_secs(1_800),
            settle: Duration::from_secs(300),
            rows_a_second: 1_000,
            batch: 100,
            readers: 8,
            threads: 4,
            flush_rows: 10_000,
            budget_share: 0.5,
            scale: 1.0,
            active: 3,
            rotate: Duration::from_secs(60),
            window: Duration::from_secs(300),
            as_image: true,
            seed: 1,
            sha: "abc".into(),
            scope: Some(true),
            ready_within: Duration::from_secs(60),
        }
    }
}

/// Every value §9.5 fixes the workload at is pinned by `is_measurement`:
/// a run that moves any one of them is a smoke, never compared as the
/// measurement (review of #169).
#[test]
fn a_run_off_any_workload_value_is_a_smoke() {
    assert!(Bench::measured().is_measurement());
    type Moving = fn(&mut Bench);
    let moved: [(&str, Moving); 9] = [
        ("soak", |b| b.soak = Duration::from_secs(1_799)),
        ("scale", |b| b.scale = 0.5),
        ("rate", |b| b.rows_a_second = 999),
        ("readers", |b| b.readers = 7),
        ("flush_rows", |b| b.flush_rows = 50_000),
        ("budget share", |b| b.budget_share = 0.6),
        ("active", |b| b.active = 2),
        ("rotation", |b| b.rotate = Duration::from_secs(30)),
        ("batch", |b| b.batch = 50),
    ];
    for (what, moving) in moved {
        let mut bench = Bench::measured();
        moving(&mut bench);
        assert!(
            !bench.is_measurement(),
            "{what} moved, and still the measurement"
        );
    }
}

/// When each place in the active set writes, and to which tenant: fixed by
/// the batch's index, never by the clock, so that every arm writes each
/// tenant the same rows whatever its timing (review of #169).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Schedule {
    /// From one of a place's batches to its next.
    every: Duration,
    /// A place's batches in one turn of the active set.
    a_turn: u64,
    /// A place's batches over the whole soak.
    batches: u64,
    rotate: Duration,
    active: usize,
    batch: u64,
}

impl Schedule {
    /// The bench's schedule, worked out in whole numbers.
    ///
    /// # Panics
    ///
    /// Unless a place's batches come a whole number of nanoseconds apart,
    /// and a turn and the soak are each a whole number of them.
    fn of(bench: &Bench) -> Schedule {
        let (rate, batch, active) = (
            bench.rows_a_second as u64,
            bench.batch as u64,
            bench.active as u64,
        );
        let whole = |of: u64, by: u64, what: &str| {
            assert!(
                by > 0 && of.is_multiple_of(by),
                "{what} is not whole in batches: STEP3_ROWS_A_SECOND, STEP3_BATCH,                  STEP3_ACTIVE, STEP3_ROTATE_SECS and STEP3_SOAK_SECS must divide"
            );
            of / by
        };
        let every = whole(
            batch * active * 1_000_000_000,
            rate,
            "a place's batch interval",
        );
        Schedule {
            every: Duration::from_nanos(every),
            a_turn: whole(bench.rotate.as_secs() * rate, batch * active, "a turn"),
            batches: whole(bench.soak.as_secs() * rate, batch * active, "the soak"),
            rotate: bench.rotate,
            active: bench.active,
            batch,
        }
    }

    /// The tenant place `place` writes its `k`-th batch to: the active set
    /// begins at t0, and moves on one tenant each turn.
    fn tenant(&self, place: usize, k: u64) -> usize {
        ((k / self.a_turn) as usize + place) % PRELOADS.len()
    }

    /// When a place's `k`-th batch is due, after the soak's start.
    fn due(&self, k: u64) -> Duration {
        Duration::from_nanos(self.every.as_nanos() as u64 * k)
    }

    /// The tenants active `at` into the soak, for readers: who count no
    /// rows, and so go by the clock.
    fn active_at(&self, at: Duration) -> Vec<usize> {
        let turn = (at.as_nanos() / self.rotate.as_nanos().max(1)) as usize;
        (0..self.active)
            .map(|place| (turn + place) % PRELOADS.len())
            .collect()
    }

    /// The rows each tenant is written over the soak, every batch written.
    fn rows_by_tenant(&self) -> BTreeMap<String, u64> {
        let mut rows: BTreeMap<String, u64> = (0..PRELOADS.len())
            .map(|at| (tenant_id(at).to_string(), 0))
            .collect();
        for place in 0..self.active {
            for k in 0..self.batches {
                *rows
                    .get_mut(tenant_id(self.tenant(place, k)).as_str())
                    .unwrap() += self.batch;
            }
        }
        rows
    }
}

/// The active set moves on by the batch: as §9.5 sets it, each place
/// writes 200 batches, a minute's, to a tenant before the next; the three
/// places never write one tenant at the same batch; and over the soak
/// every tenant is written the same 300,000 rows (review of #169).
#[test]
fn the_active_set_moves_on_by_the_batch_not_by_the_clock() {
    let schedule = Schedule::of(&Bench::measured());
    assert_eq!(schedule.every, Duration::from_millis(300));
    assert_eq!((schedule.a_turn, schedule.batches), (200, 6_000));
    assert_eq!(schedule.tenant(0, 0), 0);
    assert_eq!(schedule.tenant(0, 199), 0);
    assert_eq!(schedule.tenant(0, 200), 1);
    assert_eq!(schedule.tenant(2, 0), 2);
    assert_eq!(schedule.tenant(2, 1_000), 1);
    assert_eq!(schedule.due(200), Duration::from_secs(60));
    for k in 0..schedule.batches {
        let mut places: Vec<usize> = (0..3).map(|place| schedule.tenant(place, k)).collect();
        places.sort_unstable();
        places.dedup();
        assert_eq!(places.len(), 3, "two places on one tenant at batch {k}");
    }
    let rows = schedule.rows_by_tenant();
    assert_eq!(rows.len(), 6);
    assert!(rows.values().all(|rows| *rows == 300_000), "{rows:?}");
    // Readers, by the clock, find the set the writers are in.
    assert_eq!(schedule.active_at(Duration::from_secs(59)), vec![0, 1, 2]);
    assert_eq!(schedule.active_at(Duration::from_secs(60)), vec![1, 2, 3]);
    // A smoke's two minutes: two turns, the same for every arm.
    let smoke = Schedule::of(&Bench {
        soak: Duration::from_secs(120),
        ..Bench::measured()
    });
    assert_eq!(smoke.rows_by_tenant().values().sum::<u64>(), 120_000);
}

/// A tenant's preload: one merged segment of so many rows, if any, made of
/// four flushes; then flushed segments of so many rows each (§9.5).
struct Preload {
    merged: Option<u64>,
    flushed: [u64; 3],
}

const PRELOADS: [Preload; 6] = [
    Preload {
        merged: Some(400_000),
        flushed: [60_000; 3],
    },
    Preload {
        merged: Some(300_000),
        flushed: [50_000; 3],
    },
    Preload {
        merged: None,
        flushed: [80_000; 3],
    },
    Preload {
        merged: None,
        flushed: [60_000; 3],
    },
    Preload {
        merged: None,
        flushed: [40_000; 3],
    },
    Preload {
        merged: None,
        flushed: [30_000; 3],
    },
];

fn tenant_id(at: usize) -> TenantId {
    TenantId::new(format!("t{at}")).unwrap()
}

/// A row's id in tenant `at`: the tenant in the high bits, its n-th row
/// in the low.
fn object_id(at: usize, n: u64) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(((at as u128) << 64) | u128::from(n)))
}

fn schema() -> Collection {
    Collection {
        name: CollectionName::new(COLLECTION).unwrap(),
        properties: vec![
            Property::new("text", DataType::Text).searchable(),
            Property::new("n", DataType::Int),
            Property::new("bucket", DataType::Int),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "bge-base-en-v1.5".into(),
            dimensions: DIMENSIONS as u32,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

/// The fixture's passages, over and over, a shard at a time: what the
/// load generator holds is one shard's vectors and the batch it makes.
struct Passages {
    fixture: Fixture,
    rows: u64,
    current: vellev_bench::fixture::Passages,
}

impl Passages {
    fn of(fixture: &Path) -> Passages {
        let rows = std::fs::read_dir(fixture.join("vectors/passages"))
            .expect("the fixture's passage vectors")
            .filter(|entry| {
                let path = entry.as_ref().unwrap().path();
                path.extension().is_some_and(|ext| ext == "f32")
            })
            .map(|entry| entry.unwrap().metadata().unwrap().len() / (DIMENSIONS as u64 * 4))
            .sum();
        let fixture = Fixture::at(fixture);
        let current = fixture.passages(rows, 1).expect("the fixture's passages");
        Passages {
            fixture,
            rows,
            current,
        }
    }

    fn next_row(&mut self) -> Row {
        loop {
            match self.current.next() {
                Some(rows) => return rows.expect("a fixture row").pop().unwrap(),
                None => {
                    self.current = self.fixture.passages(self.rows, 1).unwrap();
                }
            }
        }
    }
}

fn upsert(at: usize, n: u64, row: &Row) -> Mutation {
    Mutation::Upsert {
        collection: CollectionName::new(COLLECTION).unwrap(),
        id: object_id(at, n),
        properties: json!({ "text": row.text, "n": n, "bucket": row.bucket })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(row.vector.clone()),
        only_if_version: None,
    }
}

/// A small, seeded generator: the same writes and reads in every arm.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// The store as production has it, standing in for S3 on local disk: a
/// bucket reached only through requests, never read in place, behind a
/// cache directory on the shared disk.
async fn store(data: &Path) -> Arc<dyn Blobs> {
    let bucket = data.join("bucket");
    let cache = data.join("cache");
    std::fs::create_dir_all(&bucket).unwrap();
    std::fs::create_dir_all(&cache).unwrap();
    let transfers = Transfers::default();
    let local = LocalFileSystem::new_with_prefix(&bucket).unwrap();
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(local)).transferring(transfers));
    let disk: Disk = Disk::shared()
        .clone()
        .loading_at_once(2)
        .unwrap()
        .fetching(&transfers);
    Arc::new(CachedBlobs::open(blobs, cache).await.on(disk))
}

fn node(coordinator: &Arc<MemoryCoordinator>, blobs: &Arc<dyn Blobs>, settings: Settings) -> Node {
    let coordinator: Arc<dyn Coordinator> = coordinator.clone();
    Node::new(
        coordinator,
        blobs.clone(),
        Arc::new(Embedders::new(vellev_embed::Settings::default())),
        settings,
    )
}

/// A segment as a tenant holds it, as far as the preload makes it the
/// same in every arm: not its key, which is drawn afresh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct Segment {
    rows: u64,
    tombstones: u64,
    clustered: bool,
}

/// What a tenant held once preloaded: the same in every arm.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Preloaded {
    tenant: String,
    rows: u64,
    clustered: usize,
    clustered_rows: u64,
    plain: usize,
    plain_rows: u64,
    /// Its segments, oldest first.
    segments: Vec<Segment>,
}

/// The rows a preload of `rows` comes to at `scale`.
fn scaled(rows: u64, scale: f64) -> u64 {
    ((rows as f64 * scale) as u64).max(1)
}

/// What §9.5's preload leaves each tenant holding at `scale`, oldest
/// segment first: the reference every arm's preload is held to, so that
/// none is held to another arm's.
fn preload_expected(scale: f64) -> BTreeMap<String, Vec<Segment>> {
    let above = vellev_storage::Merging::default().clustered_above as u64;
    let segment = |rows: u64| Segment {
        rows,
        tombstones: 0,
        clustered: rows > above,
    };
    PRELOADS
        .iter()
        .enumerate()
        .map(|(at, preload)| {
            let merged = preload
                .merged
                .map(|merged| segment(scaled(merged, scale) / 4 * 4));
            let flushed = preload.flushed.map(|rows| segment(scaled(rows, scale)));
            (
                tenant_id(at).to_string(),
                merged.into_iter().chain(flushed).collect(),
            )
        })
        .collect()
}

/// Preload every tenant as §9.5 says, through a node that does nothing of
/// its own accord in the time it takes: every flush and merge asked for
/// here, none by timing. Returns what each tenant came to, what they are
/// reckoned to hold together, and how many rows each was written.
async fn preload(
    bench: &Bench,
    coordinator: &Arc<MemoryCoordinator>,
    blobs: &Arc<dyn Blobs>,
    passages: &mut Passages,
) -> (Vec<Preloaded>, usize, Vec<u64>) {
    let settings = Settings {
        // No flush asked for by writes, and no maintenance round within
        // the hour: a day's restore window has rounds hourly.
        flush_rows: usize::MAX / 8,
        restore_window: Duration::from_secs(24 * 60 * 60),
        warm: None,
        ..Settings::new(NodeId::generate("preload"))
    };
    let preloading = node(coordinator, blobs, settings);
    let mut counts = Vec::new();
    let mut held_segments: BTreeMap<String, Vec<Segment>> = BTreeMap::new();
    for (at, preload) in PRELOADS.iter().enumerate() {
        let id = tenant_id(at);
        preloading.create_tenant(&id).await.unwrap();
        let tenant = preloading.tenant(&id).await.unwrap();
        tenant
            .write(vec![Mutation::CreateCollection(Box::new(schema()))])
            .await
            .unwrap();
        let scaled = |rows: u64| scaled(rows, bench.scale);
        let mut segments: Vec<u64> = Vec::new();
        if let Some(merged) = preload.merged {
            segments.extend([scaled(merged) / 4; 4]);
        }
        let merged_first = preload.merged.is_some();
        let mut n = 0;
        for (segment, rows) in segments
            .iter()
            .copied()
            .chain(preload.flushed.map(scaled))
            .enumerate()
        {
            let mut left = rows;
            while left > 0 {
                let take = left.min(2_000);
                let batch: Vec<Mutation> = (0..take)
                    .map(|i| upsert(at, n + i, &passages.next_row()))
                    .collect();
                tenant.write(batch).await.unwrap();
                n += take;
                left -= take;
            }
            tenant.flush().await.unwrap();
            if merged_first && segment == 3 {
                tenant
                    .compact(Duration::from_secs(24 * 60 * 60))
                    .await
                    .unwrap();
            }
        }
        counts.push(n);
        let held: Vec<Segment> = tenant.read(|state| {
            state
                .segments()
                .map(|meta| Segment {
                    rows: meta.rows,
                    tombstones: meta.tombstones,
                    clustered: meta.index.is_some(),
                })
                .collect()
        });
        held_segments.insert(id.to_string(), held);
        eprintln!("preloaded {id}: {n} rows");
    }
    let figures = preloading.tenant_figures();
    let resident = figures.iter().map(|(_, figures)| figures.resident).sum();
    let mut preloaded: Vec<Preloaded> = figures
        .into_iter()
        .map(|(tenant, figures)| Preloaded {
            tenant: tenant.to_string(),
            rows: figures.clustered_rows + figures.plain_rows,
            clustered: figures.clustered,
            clustered_rows: figures.clustered_rows,
            plain: figures.plain,
            plain_rows: figures.plain_rows,
            segments: held_segments.remove(tenant.as_str()).unwrap_or_default(),
        })
        .collect();
    preloaded.sort_by(|a, b| a.tenant.cmp(&b.tenant));
    preloading.shutdown().await;
    (preloaded, resident, counts)
}

/// What the process says of itself on Linux: `None` elsewhere.
fn proc_status(field: &str) -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{field}:")))
        .map(|value| value.trim().to_string())
}

fn rss_anon() -> Option<u64> {
    let kb = proc_status("RssAnon")?;
    kb.trim_end_matches(" kB")
        .trim()
        .parse::<u64>()
        .ok()
        .map(|kb| kb * 1024)
}

/// The process's cgroup, as cgroup v2 names it.
fn cgroup() -> Option<String> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(path.to_string())
}

/// The cgroup's peak memory, where the process has a cgroup of its own
/// that keeps it.
fn cgroup_peak() -> Option<u64> {
    let path = cgroup()?;
    let peak = std::fs::read_to_string(format!("/sys/fs/cgroup{path}/memory.peak")).ok()?;
    peak.trim().parse().ok()
}

/// Where the process ran: in a systemd scope of its own, as the script
/// says, and the cgroup it found itself in.
fn process(bench: &Bench) -> Value {
    json!({ "scope_made": bench.scope, "cgroup": cgroup() })
}

/// What the machine is, as far as two arms must share it to be compared.
fn machine() -> Value {
    let line_of = |path: &str, field: &str| {
        let text = std::fs::read_to_string(path).ok()?;
        let line = text.lines().find(|line| line.starts_with(field))?;
        Some(line.split_once(':')?.1.trim().to_string())
    };
    json!({
        "cpu_model": line_of("/proc/cpuinfo", "model name"),
        "memory": line_of("/proc/meminfo", "MemTotal"),
        "threads": std::thread::available_parallelism().map_or(1, usize::from),
        "os": std::env::consts::OS,
    })
}

/// How long `cpu.stat` says a cgroup was held back by its CPU limit.
fn throttled_usec(cpu_stat: &str) -> Option<u64> {
    cpu_stat
        .lines()
        .find_map(|line| line.strip_prefix("throttled_usec "))?
        .trim()
        .parse()
        .ok()
}

/// How long the process's cgroup has been held back by its CPU limit.
fn cgroup_throttled() -> Option<u64> {
    let path = cgroup()?;
    throttled_usec(&std::fs::read_to_string(format!("/sys/fs/cgroup{path}/cpu.stat")).ok()?)
}

#[test]
fn a_cgroups_throttling_is_read_from_its_cpu_stat() {
    let stat = "usage_usec 10\nnr_throttled 2\nthrottled_usec 1234\n";
    assert_eq!(throttled_usec(stat), Some(1_234));
    assert_eq!(throttled_usec("usage_usec 10\n"), None);
}

/// The share of the CPU stolen in the worst interval of `samples` within
/// each window of `window_s`, and of the whole soak: a short spell a
/// whole soak's share hides. `samples` are (seconds in, stolen ticks, all
/// ticks), as sampled.
fn steal_peaks(samples: &[(f64, u64, u64)], window_s: f64) -> Value {
    let mut windows: BTreeMap<u64, f64> = BTreeMap::new();
    for pair in samples.windows(2) {
        let (ticks, stolen) = (pair[1].2 - pair[0].2, pair[1].1 - pair[0].1);
        if ticks > 0 {
            let share = stolen as f64 / ticks as f64;
            let peak = windows.entry((pair[1].0 / window_s) as u64).or_default();
            *peak = peak.max(share);
        }
    }
    let peak = windows.values().copied().fold(0.0, f64::max);
    let windows: Vec<Value> = windows
        .into_iter()
        .map(|(window, peak)| json!({ "from_s": window as f64 * window_s, "peak": peak }))
        .collect();
    json!({ "peak": peak, "windows": windows })
}

/// A spell of steal shows as its window's peak though the soak's share
/// is small (review of #169).
#[test]
fn a_spell_of_steal_shows_in_its_windows_peak() {
    let samples = [
        (0.0, 0, 0),
        (10.0, 0, 1_000),
        (20.0, 300, 2_000),
        (30.0, 300, 3_000),
        (310.0, 300, 31_000),
    ];
    let peaks = steal_peaks(&samples, 300.0);
    assert_eq!(peaks["peak"], json!(0.3), "{peaks}");
    assert_eq!(peaks["windows"][0]["peak"], json!(0.3), "{peaks}");
    assert_eq!(peaks["windows"][1]["peak"], json!(0.0), "{peaks}");
}

/// The share of the time `samples` cover that the host stole over the mark,
/// `None` with too few to tell: a soak's whole average would hide the spell
/// a p99 is made of. `samples` are as for [`steal_peaks`].
fn stolen_for(samples: &[(f64, u64, u64)]) -> Option<f64> {
    let (mut over, mut all) = (0.0, 0.0);
    for pair in samples.windows(2) {
        let (ticks, stolen) = (pair[1].2 - pair[0].2, pair[1].1 - pair[0].1);
        let seconds = pair[1].0 - pair[0].0;
        all += seconds;
        if ticks > 0 && stolen as f64 / ticks as f64 > STEAL_AT_MOST {
            over += seconds;
        }
    }
    (all > 0.0).then(|| over / all)
}

/// `samples` with the soak's two ends, as the sampler reads every 10 s and
/// would leave out the time before its first and after its last.
fn with_ends(
    samples: &[(f64, u64, u64)],
    first: Option<(u64, u64)>,
    last: Option<(u64, u64)>,
    soak_s: f64,
) -> Vec<(f64, u64, u64)> {
    let first = first.map(|(stolen, all)| (0.0, stolen, all));
    let last = last.map(|(stolen, all)| (soak_s, stolen, all));
    first
        .into_iter()
        .chain(samples.iter().copied())
        .chain(last)
        .collect()
}

/// What `samples`, ends and all, say of steal: its share of the CPU over
/// all they cover, and the share of the time over the mark.
fn steal_read(samples: &[(f64, u64, u64)]) -> (Option<f64>, Option<f64>) {
    let share = samples
        .first()
        .zip(samples.last())
        .and_then(|(first, last)| {
            let all = last.2.checked_sub(first.2).filter(|all| *all > 0)?;
            Some(last.1.saturating_sub(first.1) as f64 / all as f64)
        });
    (share, stolen_for(samples))
}

/// Whether the kernel accounts steal here at all, from `/proc/stat`: a
/// machine that shows none on any CPU since boot cannot be told from one
/// that does not count it, so that a run on it has steal unmeasured.
fn steal_accounted(stat: &str) -> Option<bool> {
    let cpus = stat.lines().filter(|line| {
        line.strip_prefix("cpu")
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
    });
    let stolen: Vec<u64> = cpus
        .map(|line| line.split_whitespace().nth(8)?.parse().ok())
        .collect::<Option<_>>()?;
    (!stolen.is_empty()).then(|| stolen.iter().any(|ticks| *ticks > 0))
}

#[test]
fn a_kernel_that_shows_no_steal_on_any_cpu_has_it_unmeasured() {
    let stat = |one: u64, two: u64| {
        format!(
            "cpu  9 0 9 9 0 0 0 {}\ncpu0 4 0 4 4 0 0 0 {one}\ncpu1 5 0 5 5 0 0 0 {two}\n",
            one + two
        )
    };
    assert_eq!(steal_accounted(&stat(0, 0)), Some(false));
    assert_eq!(steal_accounted(&stat(0, 3)), Some(true));
    assert_eq!(steal_accounted("intr 1\n"), None);
}

/// A spell over the mark counts for the time it took, however small a
/// share of the soak's CPU it comes to (review of #169).
#[test]
fn a_spell_of_steal_counts_for_how_long_it_lasts() {
    let samples = [
        (0.0, 0, 0),
        (10.0, 0, 1_000),
        (20.0, 100, 2_000),
        (100.0, 100, 10_000),
    ];
    assert_eq!(stolen_for(&samples), Some(0.1));
    assert_eq!(stolen_for(&samples[..1]), None);
}

/// The machine's CPU ticks so far, from `/proc/stat`: those its host
/// took for others (steal), and all of them.
fn cpu_ticks() -> Option<(u64, u64)> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let line = stat.lines().find(|line| line.starts_with("cpu "))?;
    // user, nice, system, idle, iowait, irq, softirq, steal.
    let ticks: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .take(8)
        .map(|n| n.parse().ok())
        .collect::<Option<_>>()?;
    (ticks.len() == 8).then(|| (ticks[7], ticks.iter().sum()))
}

/// How the process runs, against how the image runs it (§9.3).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Running {
    uid: Option<u32>,
    cap_sys_nice: Option<bool>,
    rlimit_nice: Option<String>,
    jemalloc_global: Option<bool>,
    threads: usize,
    os: String,
}

impl Running {
    fn now() -> Running {
        let uid = proc_status("Uid")
            .and_then(|uids| uids.split_whitespace().nth(1).map(str::to_string))
            .and_then(|uid| uid.parse().ok());
        let cap_sys_nice = proc_status("CapEff")
            .and_then(|caps| u64::from_str_radix(&caps, 16).ok())
            .map(|caps| caps & (1 << CAP_SYS_NICE) != 0);
        let rlimit_nice = std::fs::read_to_string("/proc/self/limits")
            .ok()
            .and_then(|limits| {
                limits
                    .lines()
                    .find(|line| line.starts_with("Max nice priority"))
                    .map(|line| {
                        line.split_whitespace()
                            .skip(3)
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
            });
        Running {
            uid,
            cap_sys_nice,
            rlimit_nice,
            jemalloc_global: GLOBAL.is_global(),
            threads: std::thread::available_parallelism().map_or(1, usize::from),
            os: std::env::consts::OS.to_string(),
        }
    }

    /// Whether it runs as the image does: uid 10001, without CAP_SYS_NICE,
    /// with the default nice limit of 0, on jemalloc.
    fn as_the_image(&self) -> bool {
        self.uid == Some(IMAGE_UID)
            && self.cap_sys_nice == Some(false)
            && self.rlimit_nice.as_deref() == Some("0 0")
            && self.jemalloc_global == Some(true)
    }
}

/// One thread of the process, as Linux says.
struct Task {
    name: String,
    nice: i64,
    /// User and system time, in clock ticks.
    ticks: u64,
}

fn tasks() -> Vec<Task> {
    let Ok(dir) = std::fs::read_dir("/proc/self/task") else {
        return Vec::new();
    };
    let mut tasks = Vec::new();
    for entry in dir.flatten() {
        let path = entry.path();
        let (Ok(name), Ok(stat)) = (
            std::fs::read_to_string(path.join("comm")),
            std::fs::read_to_string(path.join("stat")),
        ) else {
            continue;
        };
        // After the command in parentheses: state is the 3rd field of the
        // line, utime the 14th, stime the 15th, nice the 19th.
        let Some(close) = stat.rfind(')') else {
            continue;
        };
        let fields: Vec<&str> = stat[close + 2..].split(' ').collect();
        let field = |n: usize| fields.get(n - 3).and_then(|f| f.parse::<i64>().ok());
        let (Some(utime), Some(stime), Some(nice)) = (field(14), field(15), field(19)) else {
            continue;
        };
        tasks.push(Task {
            name: name.trim_end().to_string(),
            nice,
            ticks: (utime + stime) as u64,
        });
    }
    tasks
}

/// The process's own CPU, threads gone included, in clock ticks.
fn process_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let close = stat.rfind(')')?;
    let fields: Vec<&str> = stat[close + 2..].split(' ').collect();
    let utime: u64 = fields.get(14 - 3)?.parse().ok()?;
    let stime: u64 = fields.get(15 - 3)?.parse().ok()?;
    Some(utime + stime)
}

/// The pool a thread is one of, by its name.
fn pool_of(name: &str) -> &'static str {
    for (prefix, pool) in [
        ("vellev-read", "reads"),
        ("vellev-merge", "merges"),
        ("vellev-tune", "tuning"),
        ("vellev-pass", "passes"),
        ("vellev-build", "global"),
        (RUNTIME_THREADS, "tokio"),
        ("jemalloc_bg_thd", "jemalloc"),
    ] {
        if name.starts_with(prefix) {
            return pool;
        }
    }
    "other"
}

/// What the bench's runtime names its threads, workers and blocking ones
/// alike: named here, as tokio's own name for them changes between
/// releases (review of #169).
const RUNTIME_THREADS: &str = "vellev-rt";

/// Each thread's pool, told from its name, so that CPU lands under the
/// pool that spent it and not under "other" (review of #169).
#[test]
fn a_threads_pool_is_told_from_its_name() {
    for (name, pool) in [
        ("vellev-rt", "tokio"),
        ("vellev-merge-1", "merges"),
        ("vellev-build-3", "global"),
        ("vellev-read-0", "reads"),
        ("vellev-tune", "tuning"),
        ("vellev-pass", "passes"),
        ("jemalloc_bg_thd", "jemalloc"),
        ("something", "other"),
    ] {
        assert_eq!(pool_of(name), pool, "{name}");
    }
}

/// CPU seconds by pool: the threads there are now, and what the process
/// spent in threads gone, as `other_gone`.
fn cpu_by_pool() -> BTreeMap<String, f64> {
    let ticks_a_second = 100.0;
    let mut by: BTreeMap<String, f64> = BTreeMap::new();
    let mut live = 0;
    for task in tasks() {
        live += task.ticks;
        *by.entry(pool_of(&task.name).to_string()).or_default() += task.ticks as f64;
    }
    if let Some(total) = process_ticks() {
        by.insert("gone".into(), total.saturating_sub(live) as f64);
    }
    by.values_mut().for_each(|ticks| *ticks /= ticks_a_second);
    by
}

/// Each thread whose priority is not as the arm says it must be (§9.2):
/// the merge pool's at the arm's nice and as many as it has, the passes'
/// at its own, and every other at the process's.
fn nice_wrong(pool: MergePool) -> Vec<String> {
    let (threads, nice) = match pool {
        MergePool::Global => (0, 0),
        MergePool::Own { threads, nice } => {
            (MERGE_BUILDS * (threads / MERGE_BUILDS), i64::from(nice))
        }
    };
    let mut wrong = Vec::new();
    let mut merging = 0;
    for task in tasks() {
        let expected = match pool_of(&task.name) {
            "merges" => {
                merging += 1;
                nice
            }
            "passes" => PASSES_NICE,
            _ => 0,
        };
        if task.nice != expected {
            wrong.push(format!("{} at {}, not {expected}", task.name, task.nice));
        }
    }
    if merging != threads {
        wrong.push(format!("{merging} merge pool threads, not {threads}"));
    }
    wrong
}

/// What the node samples as the soak goes.
#[derive(Default)]
struct Sampled {
    /// Each tenant's segments and rows awaiting a merge, every 10 s while
    /// it is held: (seconds into the soak, segments, unmerged rows).
    debt: BTreeMap<String, Vec<(f64, f64, f64)>>,
    /// Each tenant's clustered rows as last seen: a change is a clustered
    /// merge committed.
    clustered_rows: HashMap<String, u64>,
    clustered_merges_seen: u64,
    allocated_peak: u64,
    rss_anon_peak: u64,
    unflushed_rows_peak: usize,
    unflushed_bytes_peak: usize,
    tenant_unflushed_peak: usize,
    draining_peak: usize,
    freeing_peak: usize,
    nice_checks: u64,
    nice_wrong: Vec<String>,
    /// The machine's CPU ticks every 10 s: (seconds in, stolen, all).
    steal: Vec<(f64, u64, u64)>,
    /// What the node holds, as it reckons it and as it really is, every
    /// 10 s: for telling where memory goes past the budget.
    memory: Vec<Value>,
}

/// How a read was answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Answered {
    Yes,
    /// Turned away, to be asked again: a 429 or a 503.
    Refused,
    /// Anything else: an error, a timeout, or an answer not read whole.
    Failed,
}

/// One read as a client timed it: when it began, in seconds into the
/// soak; how long it took to be answered, refused or to fail, in
/// milliseconds, at the client's timeout if it timed out; and how.
#[derive(Clone, Copy, Debug)]
struct Read {
    at: f32,
    ms: f32,
    answered: Answered,
}

/// The reads of a soak of `soak_s`, whole and by windows of `window_s`.
/// Every read is in the latencies, refused and failed ones at what they
/// took: a client waits for those as for any other, and leaving them out
/// would make an arm that turns reads away look faster (review of #169).
fn reads_summary(reads: &[Read], soak_s: f64, window_s: f64) -> Value {
    let latencies = |within: &dyn Fn(&Read) -> bool| {
        sorted(
            reads
                .iter()
                .filter(|read| within(read))
                .map(|read| f64::from(read.ms))
                .collect(),
        )
    };
    let all = latencies(&|_| true);
    let count = |answered: Answered| reads.iter().filter(|r| r.answered == answered).count();
    let windows: Vec<Value> = (0..)
        .map(|w| f64::from(w) * window_s)
        .take_while(|from| *from < soak_s)
        .map(|from| {
            let to = from + window_s;
            let within = latencies(&|read| f64::from(read.at) >= from && f64::from(read.at) < to);
            json!({ "from_s": from, "reads": within.len(), "p50_ms": percentile(&within, 50.0),
                    "p99_ms": percentile(&within, 99.0) })
        })
        .collect();
    json!({
        "count": reads.len(), "answered": count(Answered::Yes),
        "refused": count(Answered::Refused), "failed": count(Answered::Failed),
        "p50_ms": percentile(&all, 50.0), "p99_ms": percentile(&all, 99.0),
        "windows": windows,
    })
}

/// A read refused or failed is in the p99 at what it took: 98 reads of 10
/// ms and two refused after half a second have a p99 of half a second, not
/// of 10 ms; and each window counts them (review of #169).
#[test]
fn reads_refused_or_failed_are_in_the_p99_at_what_they_took() {
    let mut reads: Vec<Read> = (0..98)
        .map(|n| Read {
            at: n as f32,
            ms: 10.0,
            answered: Answered::Yes,
        })
        .collect();
    reads.push(Read {
        at: 5.0,
        ms: 500.0,
        answered: Answered::Refused,
    });
    reads.push(Read {
        at: 6.0,
        ms: 120_000.0,
        answered: Answered::Failed,
    });
    let summary = reads_summary(&reads, 100.0, 50.0);
    assert_eq!(summary["count"], 100);
    assert_eq!(
        (
            &summary["answered"],
            &summary["refused"],
            &summary["failed"]
        ),
        (&json!(98), &json!(1), &json!(1))
    );
    assert_eq!(summary["p99_ms"], json!(500.0), "{summary}");
    assert_eq!(summary["windows"][0]["reads"], 52);
    assert_eq!(summary["windows"][0]["p99_ms"], json!(120_000.0));
}

/// Why requests failed, for the report: a count by kind, and the first
/// few as they were said, so that a run with failures says what they were.
#[derive(Default)]
struct Failures {
    by_kind: BTreeMap<String, u64>,
    first: Vec<String>,
}

impl Failures {
    /// The most kept as they were said.
    const KEPT: usize = 5;

    fn note(&mut self, kind: String, said: String) {
        *self.by_kind.entry(kind).or_default() += 1;
        if self.first.len() < Self::KEPT {
            self.first.push(said);
        }
    }

    fn report(&self) -> Value {
        json!({ "by_kind": self.by_kind, "first": self.first })
    }

    /// The kind of failure an answer or an error is.
    fn kind_of(answer: &Result<reqwest::StatusCode, reqwest::Error>) -> String {
        match answer {
            Ok(status) => format!("status {}", status.as_u16()),
            Err(error) if error.is_timeout() => "timeout".into(),
            Err(error) if error.is_body() || error.is_decode() => "body".into(),
            Err(_) => "transport".into(),
        }
    }
}

/// What the writers wrote, all told and by tenant.
#[derive(Default)]
struct WritesDone {
    rows: u64,
    /// Those written by the soak's end: the rate it was held to.
    rows_in_soak: u64,
    batches: u64,
    refused: u64,
    failed: u64,
    behind_most: f64,
    /// Which batch fell furthest behind, and when: for finding the cause.
    worst_lag: Option<Value>,
    /// The batch that took longest to be taken, retries among it.
    slowest_batch: Option<(f64, Value)>,
    /// Batches written after the soak's end, and how long after at most.
    late: u64,
    late_most: f64,
    /// Rows written to each tenant: every batch the schedule gives it,
    /// unless the arm could not take them all.
    by_tenant: BTreeMap<String, u64>,
    failures: Failures,
}

/// How long past the soak's end the writers may go on to write the batches
/// the schedule gives: one batch late at the end does not void an arm, and
/// the marks, windowed to the soak's end, never see them.
const LATE_AT_MOST: Duration = Duration::from_secs(60);

/// Whether a writer gives up on a batch at `now`: only once it is past
/// the soak's end by more than may be.
fn past_late(now: Instant, soak_end: Instant) -> bool {
    now >= soak_end + LATE_AT_MOST
}

/// A batch at the soak's end, or within a minute of it, is still written
/// (review of #169).
#[test]
fn a_batch_late_at_the_end_is_still_written_for_a_while() {
    let end = Instant::now();
    assert!(!past_late(end, end));
    assert!(!past_late(end + Duration::from_secs(59), end));
    assert!(past_late(end + Duration::from_secs(60), end));
}

/// The share of the machine's CPU its host taking for others (steal) that
/// marks a sampled interval as stolen from: past it, the arm ran on less
/// than the machine it is compared on (§9.6).
const STEAL_AT_MOST: f64 = 0.05;

/// The most of a soak such intervals may cover for its marks to count: the
/// 1% of reads a p99 leaves out. Stolen from for longer, the p99 itself
/// could be the host's doing, whatever the soak's average says (§9.6).
const STOLEN_FOR_AT_MOST: f64 = 0.01;

/// What the node's run must have done for its marks to count (§9.6).
struct NodeValidity {
    let_go_flushes: usize,
    clustered_merges: u64,
    nice_checks: u64,
    nice_wrong: usize,
    as_the_image: bool,
    measurement: bool,
    /// The share of the machine's CPU stolen through the soak, if read.
    steal: Option<f64>,
    /// The share of the soak's time over the mark for steal, if read.
    stolen_for: Option<f64>,
    load_reported: bool,
    /// Whether what the marks rest on was read as the soak ended.
    read_as_it_ended: bool,
}

impl NodeValidity {
    /// Why the run's marks don't count, if they don't.
    fn invalid(&self) -> Vec<String> {
        // Over the soak, and for how long: a spell the first would hide, a
        // steady theft spread under the second's mark.
        let steal = match (self.steal, self.stolen_for) {
            (Some(steal), Some(_)) if steal > STEAL_AT_MOST => Some(format!(
                "steal took {:.1}% of the CPU, over {:.0}%",
                steal * 100.0,
                STEAL_AT_MOST * 100.0
            )),
            (Some(_), Some(stolen_for)) if stolen_for > STOLEN_FOR_AT_MOST => Some(format!(
                "steal over {:.0}% for {:.1}% of the soak, over {:.0}%",
                STEAL_AT_MOST * 100.0,
                stolen_for * 100.0,
                STOLEN_FOR_AT_MOST * 100.0
            )),
            (Some(_), Some(_)) => None,
            _ => Some("steal not read".to_string()),
        };
        [
            (self.let_go_flushes == 0).then(|| "no flush before letting a tenant go".to_string()),
            (self.clustered_merges == 0)
                .then(|| "no clustered merge (over 200k rows) committed".to_string()),
            (self.nice_wrong > 0 || self.nice_checks == 0).then(|| {
                "the priorities were not as the arm says, or were never checked".to_string()
            }),
            (!self.as_the_image).then(|| "not run as the image runs".to_string()),
            (!self.measurement).then(|| "a smoke, not the measurement".to_string()),
            steal,
            (!self.load_reported).then(|| "the load generator never reported".to_string()),
            (!self.read_as_it_ended).then(|| "the soak's end not read as it ended".to_string()),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// An arm is void if its host stole over the mark for more of the soak
/// than a p99 leaves out, however low the soak's average, or over it on
/// average, however brief each spell; within both, it is not. Steal not
/// read at all voids it too (review of #169).
#[test]
fn an_arm_its_host_stole_from_for_too_long_is_invalid() {
    let run = |steal: Option<f64>, stolen_for: Option<f64>| NodeValidity {
        let_go_flushes: 7,
        clustered_merges: 2,
        nice_checks: 4,
        nice_wrong: 0,
        as_the_image: true,
        measurement: true,
        steal,
        stolen_for,
        load_reported: true,
        read_as_it_ended: true,
    };
    assert_eq!(run(Some(0.0), Some(0.0)).invalid(), Vec::<String>::new());
    assert_eq!(run(Some(0.05), Some(0.01)).invalid(), Vec::<String>::new());
    let stolen = run(Some(0.01), Some(0.02)).invalid();
    assert_eq!(stolen, vec!["steal over 5% for 2.0% of the soak, over 1%"]);
    let steady = run(Some(0.06), Some(0.0)).invalid();
    assert_eq!(steady, vec!["steal took 6.0% of the CPU, over 5%"]);
    assert_eq!(run(None, Some(0.0)).invalid(), vec!["steal not read"]);
    assert_eq!(run(Some(0.0), None).invalid(), vec!["steal not read"]);
    let late = NodeValidity {
        read_as_it_ended: false,
        ..run(Some(0.0), Some(0.0))
    };
    assert_eq!(late.invalid(), vec!["the soak's end not read as it ended"]);
}

/// A steady 4.9%, under the mark in every interval, with one of 100% in a
/// hundred, is 1% of the time and voids by the average, as the time alone
/// would let it through (review of #169).
#[test]
fn a_steady_theft_with_one_bad_interval_voids_by_the_average() {
    let mut samples = vec![(0.0, 0, 0)];
    let (mut stolen, mut all) = (0, 0);
    for n in 1..=100 {
        all += 1_000;
        stolen += if n == 50 { 1_000 } else { 49 };
        samples.push((f64::from(n) * 10.0, stolen, all));
    }
    let (steal, stolen_for) = steal_read(&samples);
    assert_eq!(stolen_for, Some(0.01));
    assert!(steal.unwrap() > STEAL_AT_MOST, "{steal:?}");
    let run = NodeValidity {
        let_go_flushes: 7,
        clustered_merges: 2,
        nice_checks: 4,
        nice_wrong: 0,
        as_the_image: true,
        measurement: true,
        steal,
        stolen_for,
        load_reported: true,
        read_as_it_ended: true,
    };
    assert_eq!(run.invalid().len(), 1, "{:?}", run.invalid());
}

/// What the load generator's run must have done for the arm's marks to
/// count.
struct LoadValidity {
    as_the_image: bool,
    rows_written_a_second: f64,
    rows_a_second: usize,
    reads: u64,
    failed_reads: u64,
    refused_reads: u64,
    /// Readers that died, or never came back: each spares the node its
    /// share of the reads.
    readers_lost: usize,
    /// How long after the soak's end the last batch was written.
    late_most: f64,
}

/// The most of its reads an arm may have refused for its marks to count:
/// each is a read a client did not get, and over quickly, so that a p99
/// alone would flatter it. A thousandth sits an order of magnitude below
/// the 1% a p99 leaves out. Likely inert as things stand: readers at twice
/// the cores never come near the node's 256 turns to read (§9.6).
const REFUSED_AT_MOST: f64 = 0.001;

/// The least share of its rate an arm must write for its marks to count:
/// one that wrote less was spared the load §3 sets.
const WRITTEN_AT_LEAST: f64 = 0.95;

impl LoadValidity {
    /// Why the arm's marks don't count, if they don't, for its load.
    fn invalid(&self) -> Vec<String> {
        let wrote_too_little =
            self.rows_written_a_second < WRITTEN_AT_LEAST * self.rows_a_second as f64;
        [
            wrote_too_little.then(|| {
                format!(
                    "wrote {:.0} rows a second, under {:.0}% of {}",
                    self.rows_written_a_second,
                    WRITTEN_AT_LEAST * 100.0,
                    self.rows_a_second
                )
            }),
            (self.failed_reads > 0).then(|| format!("{} reads failed", self.failed_reads)),
            (self.refused_reads as f64 > REFUSED_AT_MOST * self.reads as f64).then(|| {
                format!(
                    "{} of {} reads refused, over {}%",
                    self.refused_reads,
                    self.reads,
                    REFUSED_AT_MOST * 100.0
                )
            }),
            (self.reads == 0).then(|| "no reads".into()),
            (self.readers_lost > 0).then(|| format!("{} readers lost", self.readers_lost)),
            (self.late_most > LATE_AT_MOST.as_secs_f64())
                .then(|| format!("the last batch {:.0} s late", self.late_most)),
            (!self.as_the_image).then(|| "the load generator not run as the image runs".into()),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// An arm that wrote under 95% of its rate is invalid, its marks not
/// counted: the CI runner's first full-size run wrote 416 rows a second of
/// 1,000 (scratch run 41054724534), a load §3 does not set. One that wrote
/// its rate, or just under, is not invalid for it.
#[test]
fn an_arm_that_wrote_under_its_rate_is_invalid() {
    let run = |written: f64| LoadValidity {
        as_the_image: true,
        rows_written_a_second: written,
        rows_a_second: 1_000,
        reads: 1_000,
        failed_reads: 0,
        refused_reads: 0,
        readers_lost: 0,
        late_most: 0.0,
    };
    assert_eq!(run(1_000.0).invalid(), Vec::<String>::new());
    assert_eq!(run(951.0).invalid(), Vec::<String>::new());
    let short = run(416.0).invalid();
    assert_eq!(short.len(), 1, "{short:?}");
    assert!(short[0].starts_with("wrote 416 rows a second"), "{short:?}");
    assert_eq!(run(949.0).invalid().len(), 1);
}

/// An arm with a read that failed is invalid: an error or a timeout is no
/// latency to compare (review of #169).
#[test]
fn an_arm_with_a_failed_read_is_invalid() {
    let run = |failed_reads| LoadValidity {
        as_the_image: true,
        rows_written_a_second: 1_000.0,
        rows_a_second: 1_000,
        reads: 1_000,
        failed_reads,
        refused_reads: 0,
        readers_lost: 0,
        late_most: 0.0,
    };
    assert_eq!(run(0).invalid(), Vec::<String>::new());
    assert_eq!(run(1).invalid(), vec!["1 reads failed"]);
}

/// An arm no reader read, or that lost one, or that wrote its last batch
/// past the window, was not given the load §3 sets, however clean what
/// was read (review of #169).
#[test]
fn a_run_short_of_its_load_is_invalid() {
    let valid = || LoadValidity {
        as_the_image: true,
        rows_written_a_second: 1_000.0,
        rows_a_second: 1_000,
        reads: 1_000,
        failed_reads: 0,
        refused_reads: 0,
        readers_lost: 0,
        late_most: 59.0,
    };
    assert_eq!(valid().invalid(), Vec::<String>::new());
    let unread = LoadValidity {
        reads: 0,
        ..valid()
    };
    assert_eq!(unread.invalid(), vec!["no reads"]);
    let lost = LoadValidity {
        readers_lost: 2,
        ..valid()
    };
    assert_eq!(lost.invalid(), vec!["2 readers lost"]);
    let late = LoadValidity {
        late_most: 61.0,
        ..valid()
    };
    assert_eq!(late.invalid(), vec!["the last batch 61 s late"]);
}

/// An arm that turned away over a thousandth of its reads is invalid: a
/// refused read is over quickly, so the p99 alone would flatter it
/// (review of #169).
#[test]
fn an_arm_that_refused_more_than_a_small_share_of_reads_is_invalid() {
    let run = |refused_reads| LoadValidity {
        as_the_image: true,
        rows_written_a_second: 1_000.0,
        rows_a_second: 1_000,
        reads: 10_000,
        failed_reads: 0,
        refused_reads,
        readers_lost: 0,
        late_most: 0.0,
    };
    assert_eq!(run(10).invalid(), Vec::<String>::new());
    assert_eq!(
        run(11).invalid(),
        vec!["11 of 10000 reads refused, over 0.1%"]
    );
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

/// p50, p99, the most and the count and sum of `values`, in seconds.
fn summary(values: Vec<f64>) -> Value {
    let values = sorted(values);
    json!({
        "count": values.len(),
        "p50": percentile(&values, 50.0),
        "p99": percentile(&values, 99.0),
        "max": values.last(),
        "sum": values.iter().sum::<f64>(),
    })
}

/// The least-squares slope of `points`, a unit of y for each unit of x.
fn slope(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let (mx, my) = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    (sxx > 0.0).then(|| sxy / sxx)
}

/// A tenant's segments every 10 s from the soak's start to `until`, each
/// the last sampled at or before it, `start` before any: carried forward
/// while the tenant is let go, as nothing else writes it. Never a sample
/// from after the point (review of #169).
fn carried(points: &[(f64, f64)], start: f64, until: f64) -> Vec<(f64, f64)> {
    let mut series = Vec::new();
    let mut last = start;
    let mut points = points.iter().peekable();
    let mut t = 0.0;
    while t <= until {
        while let Some(point) = points.next_if(|point| point.0 <= t) {
            last = point.1;
        }
        series.push((t, last));
        t += 10.0;
    }
    series
}

/// Merge debt is carried forward from the samples at or before each
/// point, never one taken after it: a sample 3 s past a point is the next
/// point's (review of #169).
#[test]
fn merge_debt_is_carried_forward_from_samples_at_or_before_each_point() {
    let series = carried(&[(10.0, 5.0), (13.0, 7.0), (25.0, 9.0)], 4.0, 40.0);
    assert_eq!(
        series,
        vec![
            (0.0, 4.0),
            (10.0, 5.0),
            (20.0, 7.0),
            (30.0, 9.0),
            (40.0, 9.0)
        ]
    );
}

/// The share of an arm's merge work on idle tenants above which a
/// flush-wait or flush-due pass is not to be taken at face value (§9.5).
const IDLE_SHARE_LARGE: f64 = 0.10;

/// How much of an arm's merge work was on tenants outside the active set
/// when it began, from `spans`, gathers and builds, stopped and failed ones
/// among them: each one's partition, the seconds into the soak it ended,
/// and how long it ran. Wall-clock seconds, not CPU. A 5-minute restore
/// window has a tenant looked after every minute where production waits an
/// hour (§9.5), so that idle tenants do merge work production would not
/// yet, which takes CPU from others' flushes, whose holds are the
/// flush-wait mark's measure. It decides nothing: it says how far to trust
/// that mark.
fn idle_merge_builds(spans: &[(String, f64, f64)], schedule: &Schedule) -> Value {
    let (mut idle, mut idle_s, mut all_s) = (0, 0.0, 0.0);
    for (partition, ended, took) in spans {
        all_s += took;
        let tenant = partition.split('/').next().unwrap_or(partition);
        let began = Duration::from_secs_f64((ended - took).max(0.0));
        let active = schedule.active_at(began);
        if !active.iter().any(|at| tenant_id(*at).as_str() == tenant) {
            idle += 1;
            idle_s += took;
        }
    }
    json!({
        "spans": spans.len(), "idle": idle, "wall_s": all_s, "idle_wall_s": idle_s,
        "idle_share_of_wall_s": (all_s > 0.0).then(|| idle_s / all_s),
        "large": all_s > 0.0 && idle_s / all_s > IDLE_SHARE_LARGE,
    })
}

/// Whether `timed` is a span of a merge's work that holds a thread of its
/// pool: gathering or building, stopped and failed ones with those that
/// finished, as a merge stopped and started again spends its work each
/// time and idle tenants' share must count them.
fn merge_work(timed: Timed) -> bool {
    matches!(
        timed,
        Timed::Merge(MergePhase::Gather | MergePhase::Build)
            | Timed::MergeGivenUp(MergePhase::Gather | MergePhase::Build)
    )
}

#[test]
fn work_given_up_counts_among_a_merges_work() {
    for phase in [MergePhase::Gather, MergePhase::Build] {
        assert!(merge_work(Timed::Merge(phase)));
        assert!(merge_work(Timed::MergeGivenUp(phase)));
    }
    assert!(!merge_work(Timed::Merge(MergePhase::Upload)));
    assert!(!merge_work(Timed::MergeGivenUp(MergePhase::Upload)));
    assert!(!merge_work(Timed::FlushDue));
}

/// Whether `event` began in the soak, from `started` to `ended`: a wait
/// that began in it counts, however long after it ended. Kept by when they
/// ended, the longest would be the ones lost (review of #169).
fn began_within(event: &Event, started: Instant, ended: Instant) -> bool {
    let began = event.at.checked_sub(event.took);
    began.is_some_and(|began| began >= started && began <= ended)
}

/// What ended after the soak, of `events`, which began in it, by what it
/// was: a wait and a build are no one thing.
fn open_past(events: &[Event], soak_end: Instant) -> Value {
    let mut by: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for event in events.iter().filter(|event| event.at > soak_end) {
        let kind = format!("{:?}", event.timed);
        by.entry(kind).or_default().push(event.took.as_secs_f64());
    }
    let by = by.into_iter().map(|(kind, took)| {
        let summary = json!({ "count": took.len(), "longest_s": sorted(took).last() });
        (kind, summary)
    });
    Value::Object(by.collect())
}

/// How long the waits `open` have run by `at`, flush due and backlog apart,
/// of those that began from `started` to `ended`.
fn open_within(
    open: &[OpenWaits],
    started: Instant,
    ended: Instant,
    at: Instant,
) -> (Vec<f64>, Vec<f64>) {
    let within = |began: &&Instant| **began >= started && **began <= ended;
    let by = |began: &Instant| at.saturating_duration_since(*began).as_secs_f64();
    let due = open.iter().flat_map(|waits| waits.flush_due.iter());
    let capped = open.iter().flat_map(|waits| waits.backlog.iter());
    (
        due.filter(within).map(by).collect(),
        capped.filter(within).map(by).collect(),
    )
}

/// What [`open_within`] came to, for the report.
fn open_summary(open: &(Vec<f64>, Vec<f64>)) -> Value {
    let one = |waits: &[f64]| {
        let longest = waits.iter().copied().fold(0.0, f64::max);
        json!({ "count": waits.len(), "longest_s": longest })
    };
    json!({ "flush_due": one(&open.0), "backlog": one(&open.1) })
}

/// A wait begun in the soak and ended after it counts, and is said to have
/// been open at the end; one begun before, or after, does not.
#[test]
fn a_wait_counts_by_when_it_began_not_when_it_ended() {
    let started = Instant::now() + Duration::from_secs(1_000);
    let soak_end = started + Duration::from_secs(100);
    let event = |began: Duration, took: u64| Event {
        timed: Timed::FlushDue,
        took: Duration::from_secs(took),
        partition: "t0".into(),
        at: started + began + Duration::from_secs(took),
    };
    let seconds = Duration::from_secs;
    let long = event(seconds(90), 150);
    assert!(began_within(&long, started, soak_end), "a long wait lost");
    assert!(began_within(&event(seconds(10), 5), started, soak_end));
    assert!(!began_within(&event(seconds(120), 5), started, soak_end));
    let before = Event {
        at: started + seconds(5),
        took: seconds(20),
        ..long.clone()
    };
    assert!(!began_within(&before, started, soak_end));
    let spell = Event {
        timed: Timed::Backlog,
        ..event(seconds(95), 30)
    };
    let open = open_past(&[long, spell, event(seconds(10), 5)], soak_end);
    assert_eq!(
        open,
        json!({ "FlushDue": { "count": 1, "longest_s": 150.0 },
                "Backlog": { "count": 1, "longest_s": 30.0 } })
    );
}

/// How merges have ended so far, by outcome.
fn merges_ended(node: &Node) -> BTreeMap<&'static str, u64> {
    vellev_storage::timings::MergeEnd::ALL
        .iter()
        .map(|end| (end.name(), node.timings().merges_ended(*end)))
        .collect()
}

/// `after` less `before`, key by key: what happened between them.
fn since(
    before: &BTreeMap<&'static str, u64>,
    after: &BTreeMap<&'static str, u64>,
) -> BTreeMap<&'static str, u64> {
    after
        .iter()
        .map(|(key, n)| (*key, n - before.get(key).copied().unwrap_or(0)))
        .collect()
}

/// Figures set against marks windowed to the soak are the soak's: those
/// before it, and those after, are not among them (review of #169).
#[test]
fn figures_are_of_the_soak_alone() {
    let before = BTreeMap::from([("committed", 2), ("stopped", 1)]);
    let after = BTreeMap::from([("committed", 5), ("stopped", 1)]);
    let over = since(&before, &after);
    assert_eq!(over, BTreeMap::from([("committed", 3), ("stopped", 0)]));
}

/// Where the last whole turn of the active set round every tenant begins,
/// in seconds into a soak: a rotation of six tenants, 360 s at §9.5's
/// values, over which each tenant has had its turn, and no longer, as a
/// longer tail would halve a debt that is rising (§9.5).
fn last_cycle_from(soaked: Duration, rotate: Duration) -> f64 {
    let cycle = rotate.as_secs_f64() * PRELOADS.len() as f64;
    soaked.as_secs_f64() - cycle
}

/// What a tenant's segments came to on average from `from` on: merges make
/// a sawtooth of them, so that one instant at the end says more of where
/// the tooth stood than of the arm (§9.6).
fn tail_mean(series: &[(f64, f64)], from: f64) -> Option<f64> {
    let tail: Vec<f64> = series
        .iter()
        .filter(|(t, _)| *t >= from)
        .map(|(_, segments)| *segments)
        .collect();
    (!tail.is_empty()).then(|| tail.iter().sum::<f64>() / tail.len() as f64)
}

#[test]
fn debt_is_the_tails_average_not_its_last_instant() {
    let tooth = [
        (0.0, 9.0),
        (10.0, 3.0),
        (20.0, 5.0),
        (30.0, 7.0),
        (40.0, 3.0),
    ];
    assert_eq!(tail_mean(&tooth, 10.0), Some(4.5));
    assert_eq!(tail_mean(&tooth, 50.0), None);
}

/// A debt rising to 19.6% worse by the end of a soak is held to the 10% of
/// §3: over the last 20 minutes its average would come to under half of
/// that, and pass (review of #169).
#[test]
fn a_rising_debt_is_not_halved_by_the_tail_it_is_averaged_over() {
    let soak = Duration::from_secs(1_800);
    let rising: Vec<(f64, f64)> = (0..=180)
        .map(|n| {
            let t = f64::from(n) * 10.0;
            (t, 100.0 + (t - 600.0).max(0.0) * 19.6 / 1_200.0)
        })
        .collect();
    let from = last_cycle_from(soak, Duration::from_secs(60));
    assert_eq!(from, 1_440.0);
    let held = tail_mean(&rising, from).unwrap();
    assert!(held > 110.0, "a debt ending 19.6% worse came to {held}");
    assert!(tail_mean(&rising, 600.0).unwrap() < 110.0);
}

/// How an arm's merge work used its slots (§9.5): the seconds any of it
/// ran and those it ran alone, the share of the pools idle meanwhile, and
/// how long the builds that overlapped nothing took. `spans` are whether
/// each is a build, when it ended, and how long it took; a slot is held a
/// while longer, through a merge's upload and load, which use no thread of
/// its pool. `slots` is `None` for an arm with no pools of its own, where
/// nothing of them can stand idle. Clipped to the soak of `soak_s`, as what
/// runs on past it runs beside nothing that was timed, and would read as
/// time alone; a build cut off so is no lone build. It decides nothing.
fn slot_use(spans: &[(bool, f64, f64)], slots: Option<usize>, soak_s: f64) -> Value {
    let whole = spans.iter().filter(|span| span.1 <= soak_s).count();
    let spans: Vec<(bool, f64, f64)> = spans
        .iter()
        .map(|(build, ended, took)| {
            let clipped = ended.min(soak_s);
            (*build, clipped, took - (ended - clipped))
        })
        .collect();
    debug_assert!(whole <= spans.len());
    let mut edges: Vec<(f64, i64)> = spans
        .iter()
        .flat_map(|(_, ended, took)| [(ended - took, 1), (*ended, -1)])
        .collect();
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mut running, mut last, mut busy_s, mut lone_s, mut slot_s) = (0, 0.0, 0.0, 0.0, 0.0);
    for (at, change) in edges {
        if running > 0 {
            busy_s += at - last;
            slot_s += (at - last) * running as f64;
            if running == 1 {
                lone_s += at - last;
            }
        }
        running += change;
        last = at;
    }
    let overlaps = |one: &(bool, f64, f64), other: &(bool, f64, f64)| {
        one.1 - one.2 < other.1 && other.1 - other.2 < one.1
    };
    let lone_builds: Vec<f64> = spans
        .iter()
        .filter(|one| one.0 && one.1 < soak_s)
        .filter(|one| spans.iter().filter(|other| overlaps(one, other)).count() == 1)
        .map(|one| one.2)
        .collect();
    json!({
        "busy_s": busy_s, "lone_s": lone_s,
        "idle_share": slots
            .filter(|_| busy_s > 0.0)
            .map(|slots| 1.0 - slot_s / (busy_s * slots as f64)),
        "lone_builds": summary(lone_builds),
        "builds": spans.iter().filter(|span| span.0).count(),
    })
}

/// Alone for 10 s, then alongside another for 10 s, then alone for 10 s:
/// a third of the time two ran, so the pools stood a third idle; and only
/// a build that overlapped nothing counts as lone.
#[test]
fn slot_use_says_how_long_a_build_ran_alone_and_the_pools_stood_idle() {
    let spans = [(true, 20.0, 20.0), (true, 30.0, 20.0), (true, 50.0, 5.0)];
    let used = slot_use(&spans, Some(2), 100.0);
    assert_eq!(used["busy_s"], json!(35.0), "{used}");
    assert_eq!(used["lone_s"], json!(25.0), "{used}");
    assert_eq!(used["idle_share"], json!(1.0 - 45.0 / 70.0), "{used}");
    assert_eq!(used["lone_builds"]["count"], json!(1), "{used}");
    assert_eq!(used["lone_builds"]["max"], json!(5.0), "{used}");
    assert_eq!(slot_use(&[], Some(2), 100.0)["idle_share"], json!(null));
    // Run on past the soak of 25 s, a span is cut off there: nothing of it
    // reads as time alone, and it is no lone build.
    let cut = slot_use(&spans, Some(2), 25.0);
    assert_eq!(cut["busy_s"], json!(25.0), "{cut}");
    assert_eq!(cut["lone_s"], json!(15.0), "{cut}");
    assert_eq!(cut["lone_builds"]["count"], json!(0), "{cut}");
    // Arm (a) has no slots, so nothing of them stands idle: its figure
    // would mean nothing, though its builds are still timed.
    let global = slot_use(&spans, None, 100.0);
    assert_eq!(global["idle_share"], json!(null), "{global}");
    assert_eq!(global["lone_s"], json!(25.0), "{global}");
}

/// A build counts as idle by when it began: one begun on an active tenant
/// that ends after the set has moved on was asked for by a flush, not the
/// timer; one begun off the set was not (review of #169).
#[test]
fn a_merge_build_is_idle_by_whether_its_tenant_was_active_as_it_began() {
    let schedule = Schedule::of(&Bench::measured());
    let builds = [
        // Begun at 50 s, with t0 active, and ended at 130 s, with it not.
        ("t0/0".to_string(), 130.0, 80.0),
        // Begun at 10 s, with t3 idle, and ended at 70 s, with it active.
        ("t3/0".to_string(), 70.0, 60.0),
        // Active throughout.
        ("t1/0".to_string(), 40.0, 20.0),
    ];
    let found = idle_merge_builds(&builds, &schedule);
    assert_eq!(found["spans"], 3, "{found}");
    assert_eq!(found["idle"], 1, "{found}");
    assert_eq!(found["idle_wall_s"], json!(60.0), "{found}");
    assert_eq!(
        found["idle_share_of_wall_s"],
        json!(60.0 / 160.0),
        "{found}"
    );

    assert_eq!(found["large"], json!(true), "{found}");
    let small = [
        ("t3/0".to_string(), 70.0, 10.0),
        ("t1/0".to_string(), 150.0, 90.0),
    ];
    assert_eq!(idle_merge_builds(&small, &schedule)["large"], json!(false));
    let none = idle_merge_builds(&[], &schedule);
    assert_eq!(none["idle_share_of_wall_s"], json!(null));
}

/// The flush-due mark (§9.6): how long rows wait from falling due to the
/// commit that takes them, which sees what the wait for the lock does not.
/// Its p99 is below twice a flush's own p99; and what it comes to beyond a
/// flush's own p99, the wait and not the flush, is below the arm's median
/// merge build: §3's "never a merge build's length". A flush that is slow,
/// but never waits, does not fail it. All three in seconds, sorted.
fn flush_due_mark(due: &[f64], held: &[f64], builds: &[f64]) -> Value {
    let (due_p99, held_p99, build_p50) = (
        percentile(due, 99.0),
        percentile(held, 99.0),
        percentile(builds, 50.0),
    );
    json!({
        "passed": match (due_p99, held_p99, build_p50) {
            (Some(due), Some(held), Some(build)) => due < 2.0 * held && due - held < build,
            _ => false,
        },
        "due_p99_s": due_p99, "held_p99_s": held_p99, "merge_build_p50_s": build_p50,
        "mark": "flush_due p99 < 2 x held p99, and its excess over held p99 < the median merge build",
    })
}

/// Waits behind a merge build, as the 110-156 s of §9.9, fail though the
/// lock was never waited for; one a flush's own time does not (review of
/// #169).
#[test]
fn rows_waiting_a_merge_builds_length_fail_the_flush_due_mark() {
    let passed = |due: f64, held: f64, build: f64| {
        flush_due_mark(&[1.0, due], &[held], &[build])["passed"] == json!(true)
    };
    assert!(passed(3.0, 2.0, 60.0));
    // Slow, but never held: as long as it takes itself, whatever a merge
    // build takes.
    assert!(passed(64.0, 63.3, 14.8), "a slow flush that never waited");
    // Run 41057905038: 156 s, of a flush of at most 63 s, builds of 15 s.
    assert!(!passed(156.0, 63.3, 14.8), "held up 93 s");
    assert!(!passed(135.0, 2.0, 75.0), "a merge build's length");
    assert!(!passed(30.0, 2.0, 75.0), "fifteen flushes' length");
    // Within twice its own, but held a merge build's length beyond it.
    assert!(
        !passed(70.0, 40.0, 20.0),
        "held 30 s, over the median build"
    );
    assert_eq!(flush_due_mark(&[], &[1.0], &[1.0])["passed"], json!(false));
    assert_eq!(flush_due_mark(&[1.0], &[1.0], &[])["passed"], json!(false));
}

/// A stage of the node's searches as its `/metrics` histogram has it:
/// reported beside the client's p99, deciding nothing. Worked out from
/// buckets, its p99 is the bound of the bucket it falls in, or past the
/// last bound, said so, never clamped to it.
fn stage_summary(metrics: &str, stage: &str) -> Value {
    let wanted = |line: &str, series: &str| {
        line.starts_with(&format!("vellev_stage_seconds_{series}{{"))
            && line.contains(&format!("stage=\"{stage}\""))
            && line.contains("nearest")
    };
    let value = |line: &str| -> f64 {
        line.rsplit(' ')
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    };
    let mut buckets: Vec<(String, f64)> = Vec::new();
    let (mut count, mut sum) = (0.0, 0.0);
    for line in metrics.lines() {
        if wanted(line, "bucket") {
            let le = line
                .split("le=\"")
                .nth(1)
                .and_then(|rest| rest.split('"').next())
                .unwrap_or("");
            buckets.push((le.to_string(), value(line)));
        } else if wanted(line, "count") {
            count += value(line);
        } else if wanted(line, "sum") {
            sum += value(line);
        }
    }
    let target = (0.99 * count).ceil();
    let last_bound = buckets
        .iter()
        .rfind(|(le, _)| le != "+Inf")
        .map(|(le, _)| le.clone());
    let p99 = buckets
        .iter()
        .find(|(_, cumulative)| count > 0.0 && *cumulative >= target)
        .map(|(le, _)| match le.as_str() {
            "+Inf" => json!(format!("over {} s", last_bound.clone().unwrap_or_default())),
            le => json!(le.parse::<f64>().ok()),
        });
    json!({
        "count": count, "mean_s": (count > 0.0).then(|| sum / count),
        "p99_at_most_s": p99, "from": "the /metrics histogram's buckets",
    })
}

#[test]
fn a_stages_p99_from_buckets_says_when_it_is_past_the_last() {
    let text = |queue: [u32; 3]| {
        format!(
            "vellev_stage_seconds_sum{{route=\"/v1/tenants/{{tenant}}/collections/{{collection}}/nearest\",stage=\"queue\"}} 9.0\n\
             vellev_stage_seconds_count{{route=\"/v1/tenants/{{tenant}}/collections/{{collection}}/nearest\",stage=\"queue\"}} {2}\n\
             vellev_stage_seconds_bucket{{le=\"0.1\",route=\"/v1/tenants/{{tenant}}/collections/{{collection}}/nearest\",stage=\"queue\"}} {0}\n\
             vellev_stage_seconds_bucket{{le=\"3.2768\",route=\"/v1/tenants/{{tenant}}/collections/{{collection}}/nearest\",stage=\"queue\"}} {1}\n\
             vellev_stage_seconds_bucket{{le=\"+Inf\",route=\"/v1/tenants/{{tenant}}/collections/{{collection}}/nearest\",stage=\"queue\"}} {2}\n\
             vellev_stage_seconds_bucket{{le=\"+Inf\",route=\"/v1/tenants/{{tenant}}/collections/{{collection}}/nearest\",stage=\"read\"}} 50\n",
            queue[0], queue[1], queue[2]
        )
    };
    let within = stage_summary(&text([99, 100, 100]), "queue");
    assert_eq!(within["p99_at_most_s"], json!(0.1), "{within}");
    assert_eq!(within["count"], json!(100.0));
    let past = stage_summary(&text([50, 90, 100]), "queue");
    assert_eq!(past["p99_at_most_s"], json!("over 3.2768 s"), "{past}");
}

async fn http_json(
    http: &reqwest::Client,
    url: String,
    body: &Value,
) -> Result<reqwest::Response, reqwest::Error> {
    http.post(url).bearer_auth(KEY).json(body).send().await
}

/// What the node tells the load generator, through a file, once it
/// serves: where, when the soak starts by the wall clock, which both
/// processes read alike, and the rows each tenant was preloaded with.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Ready {
    url: String,
    start_unix_ms: u64,
    soak_s: u64,
    rows_of: Vec<u64>,
    sha: String,
}

fn ready_path(bench: &Bench) -> PathBuf {
    bench.out.join(format!("ready-{}.json", bench.arm.name()))
}

fn load_path(bench: &Bench) -> PathBuf {
    bench.out.join(format!("load-{}.json", bench.arm.name()))
}

/// Write `text` to `path` whole: whoever waits for the file never reads
/// part of it.
fn written_whole(path: &Path, text: &str) {
    let partial = path.with_extension("partial");
    std::fs::write(&partial, text).unwrap();
    std::fs::rename(&partial, path).unwrap();
}

/// What `path` holds once it is there, if it comes `within`.
async fn appeared(path: &Path, within: Duration) -> Option<String> {
    let until = Instant::now() + within;
    loop {
        if let Ok(text) = std::fs::read_to_string(path) {
            return Some(text);
        }
        if Instant::now() >= until {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// The instant the wall clock comes to `unix_ms` at, by this process's
/// clock.
fn instant_at(at_unix_ms: u64) -> Instant {
    let (now, now_ms) = (Instant::now(), unix_ms());
    if at_unix_ms >= now_ms {
        now + Duration::from_millis(at_unix_ms - now_ms)
    } else {
        now - Duration::from_millis(now_ms - at_unix_ms)
    }
}

/// One read of what the node's run is held to: the same figures at the
/// soak's start, at its end and once settled, so that which is which is
/// plain in what uses them.
#[derive(Clone, Debug, Default)]
struct Reading {
    /// When it was read: `None` only in a rule.
    taken_at: Option<Instant>,
    ended: BTreeMap<&'static str, u64>,
    let_go_idle: u64,
    let_go_budget: u64,
    backlogged: u64,
    top_level_cpu: Duration,
    ticks: Option<(u64, u64)>,
    throttled: Option<u64>,
    cpu: BTreeMap<String, f64>,
    largest: usize,
    waits: Vec<OpenWaits>,
    metrics: String,
}

impl Reading {
    fn of(node: &Node, app: &App) -> Reading {
        let let_go = node.let_go();
        Reading {
            taken_at: Some(Instant::now()),
            ended: merges_ended(node),
            let_go_idle: let_go.idle,
            let_go_budget: let_go.for_budget,
            backlogged: node.flushing().backlogged,
            top_level_cpu: node.merge_builds().top_level_cpu(),
            ticks: cpu_ticks(),
            throttled: cgroup_throttled(),
            cpu: cpu_by_pool(),
            largest: node.merges().largest,
            waits: node.waits_open(),
            metrics: app.metrics().encoded(),
        }
    }
}

/// All the node's run came to.
struct Soak {
    started: Instant,
    soak_end: Instant,
    soaked: Duration,
    before: Reading,
    at_end: Reading,
    settled: Reading,
    shutdown_at: Instant,
    waits_at_shutdown: Vec<OpenWaits>,
    /// Every event kept, whenever it began.
    events: Vec<Event>,
    sampled: Sampled,
    preloaded: Vec<Preloaded>,
    preloaded_resident: usize,
    preload_secs: f64,
    settled_in: f64,
    budget: usize,
    baseline_allocated: u64,
    baseline_rss_anon: Option<u64>,
    load_reported: bool,
    steal_accounted: Option<bool>,
}

/// The report of a node's run, and every event in it as timed: from what
/// was read, so that what it takes from when can be held to a rule. When
/// `arm` reads each [`Reading`] is not: [`read_as_it_ended`] voids a run
/// whose end was read late.
fn reported(
    bench: &Bench,
    pool: MergePool,
    running: &Running,
    schedule: &Schedule,
    soak: Soak,
) -> (Value, Vec<Value>) {
    let Soak {
        started,
        soak_end,
        soaked,
        before,
        at_end,
        settled,
        shutdown_at,
        waits_at_shutdown,
        events,
        sampled,
        preloaded,
        preloaded_resident,
        preload_secs,
        settled_in,
        budget,
        baseline_allocated,
        baseline_rss_anon,
        load_reported,
        steal_accounted,
    } = soak;
    let events: Vec<Event> = events
        .into_iter()
        .filter(|event| began_within(event, started, soak_end))
        .collect();
    // Every event as timed, for working out what happened when.
    let raw: Vec<Value> = events
        .iter()
        .map(|event| {
            json!({
                "timed": format!("{:?}", event.timed),
                "partition": event.partition,
                "took": event.took.as_secs_f64(),
                "at": (event.at - started).as_secs_f64(),
            })
        })
        .collect();
    let of = |timed: Timed| -> Vec<f64> {
        events
            .iter()
            .filter(|event| event.timed == timed)
            .map(|event| event.took.as_secs_f64())
            .collect()
    };
    let merge_work_timed: Vec<(String, f64, f64)> = events
        .iter()
        .filter(|event| merge_work(event.timed))
        .map(|event| {
            (
                event.partition.clone(),
                (event.at - started).as_secs_f64(),
                event.took.as_secs_f64(),
            )
        })
        .collect();
    let mut timings = serde_json::Map::new();
    for holder in Holder::ALL {
        timings.insert(
            format!("building_wait.{}", holder.name()),
            summary(of(Timed::BuildingWait(holder))),
        );
        timings.insert(
            format!("building_held.{}", holder.name()),
            summary(of(Timed::BuildingHeld(holder))),
        );
    }
    for phase in MergePhase::ALL {
        timings.insert(
            format!("merge.{}", phase.name()),
            summary(of(Timed::Merge(phase))),
        );
        timings.insert(
            format!("merge_given_up.{}", phase.name()),
            summary(of(Timed::MergeGivenUp(phase))),
        );
    }
    timings.insert("flush_due".into(), summary(of(Timed::FlushDue)));
    timings.insert("backlog".into(), summary(of(Timed::Backlog)));
    timings.insert(
        "let_go_flush.wait".into(),
        summary(of(Timed::LetGoFlushWait)),
    );
    timings.insert(
        "let_go_flush.held".into(),
        summary(of(Timed::LetGoFlushHeld)),
    );

    // Merge debt: each tenant carried forward while let go, from its
    // preload until it is first seen held.
    let last_from = soaked.as_secs_f64() - 20.0 * 60.0;
    let last_cycle = last_cycle_from(soaked, bench.rotate);
    let mut debt = serde_json::Map::new();
    let mut slopes = Vec::new();
    let mut tails = Vec::new();
    for preload in &preloaded {
        let seen = sampled
            .debt
            .get(&preload.tenant)
            .cloned()
            .unwrap_or_default();
        let start = (preload.clustered + preload.plain) as f64;
        let points: Vec<(f64, f64)> = seen.iter().map(|point| (point.0, point.1)).collect();
        let series = carried(&points, start, soaked.as_secs_f64());
        let last = series.last().map_or(start, |point| point.1);
        tails.push(tail_mean(&series, last_cycle).unwrap_or(last));
        let tail: Vec<(f64, f64)> = series
            .iter()
            .filter(|(t, _)| *t >= last_from)
            .map(|(t, segments)| (t / 60.0, *segments))
            .collect();
        let tenant_slope = slope(&tail);
        slopes.push(tenant_slope.unwrap_or(f64::NAN));
        debt.insert(
            preload.tenant.clone(),
            json!({
                "start_segments": start,
                "end_segments": last,
                "slope_last_20_min_per_min": tenant_slope,
                "samples_held": seen.len(),
                "unmerged_rows_last": seen.last().map(|point| point.2),
            }),
        );
    }
    let cpu: BTreeMap<String, f64> = at_end
        .cpu
        .iter()
        .map(|(pool, after)| {
            (
                pool.clone(),
                after - before.cpu.get(pool).copied().unwrap_or(0.0),
            )
        })
        .collect();
    let steal_samples = with_ends(
        &sampled.steal,
        before.ticks,
        at_end.ticks,
        soaked.as_secs_f64(),
    );
    let (steal, stolen_for) = steal_read(&steal_samples);

    // The marks this arm can be held to alone (§9.6); backlog refusals are
    // held to arm (a)'s by the comparison.
    let flush_wait = sorted(of(Timed::BuildingWait(Holder::Flush)));
    let flush_held = sorted(of(Timed::BuildingHeld(Holder::Flush)));
    let open_at_end = open_within(&at_end.waits, started, soak_end, soak_end);
    let open_at_shutdown = open_within(&waits_at_shutdown, started, soak_end, shutdown_at);
    // Those open at shutdown ended in no event, and are as long at least.
    let mut flush_due = of(Timed::FlushDue);
    flush_due.extend(&open_at_shutdown.0);
    let flush_due = sorted(flush_due);
    let merge_builds = sorted(of(Timed::Merge(MergePhase::Build)));
    let let_go_wait = sorted(of(Timed::LetGoFlushWait));
    let flush_wait_p99 = percentile(&flush_wait, 99.0);
    let flush_held_p99 = percentile(&flush_held, 99.0);
    let merge_build_p50 = percentile(&merge_builds, 50.0);
    let slot_spans: Vec<(bool, f64, f64)> = events
        .iter()
        .filter(|event| merge_work(event.timed))
        .map(|event| {
            let build = matches!(
                event.timed,
                Timed::Merge(MergePhase::Build) | Timed::MergeGivenUp(MergePhase::Build)
            );
            let ended = (event.at - started).as_secs_f64();
            (build, ended, event.took.as_secs_f64())
        })
        .collect();
    let peak = sampled.allocated_peak.saturating_sub(baseline_allocated);
    // The largest merge build as the soak ended: what the soak's peak can
    // have come from, not what settling built after.
    let largest_at_soak_end = at_end.largest;
    let memory_bound = budget as u64 + (MERGE_BUILDS * largest_at_soak_end) as u64;
    let max_slope = slopes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let marks = json!({
        "merge_debt": {
            "passed": slopes.iter().all(|s| s.is_finite() && *s < 0.1),
            "max_slope_per_min": max_slope, "mark": "< 0.1 segments a minute, every tenant",
        },
        "flush_wait": {
            "passed": match (flush_wait_p99, flush_held_p99, merge_build_p50) {
                (Some(wait), Some(held), Some(build)) =>
                    wait < 2.0 * held && flush_wait.last().copied().unwrap_or(0.0) < build,
                _ => false,
            },
            "wait_p99_s": flush_wait_p99, "held_p99_s": flush_held_p99,
            "wait_max_s": flush_wait.last(), "merge_build_p50_s": merge_build_p50,
            "mark": "wait p99 < 2 x held p99, and the longest wait < the median merge build",
        },
        "flush_due": flush_due_mark(&flush_due, &flush_held, &merge_builds),
        "node_memory": {
            "passed": peak <= memory_bound,
            "peak_over_baseline_bytes": peak, "bound_bytes": memory_bound,
            "budget_bytes": budget, "largest_merge_build_bytes": largest_at_soak_end,
            "mark": "jemalloc allocated peak over baseline <= budget + MERGE_BUILDS x largest merge build",
        },
        "eviction_flush": {
            "passed": percentile(&let_go_wait, 99.0).is_some_and(|p99| p99 < 2.0),
            "wait_p99_s": percentile(&let_go_wait, 99.0), "flushes": let_go_wait.len(),
            "mark": "p99 wait < 2 s",
        },
        "backlog_refusals": { "refused": at_end.backlogged - before.backlogged, "mark": "no more than arm (a)'s" },
    });
    let invalid = NodeValidity {
        let_go_flushes: let_go_wait.len(),
        clustered_merges: sampled.clustered_merges_seen,
        nice_checks: sampled.nice_checks,
        nice_wrong: sampled.nice_wrong.len(),
        as_the_image: running.as_the_image(),
        measurement: bench.is_measurement(),
        steal,
        stolen_for,
        load_reported,
        read_as_it_ended: read_as_it_ended(at_end.taken_at, soak_end),
    }
    .invalid();

    let report = json!({
        "arm": bench.arm.name(),
        "merge_pool": format!("{pool:?}"),
        "running": running,
        "process": process(bench),
        "sha": bench.sha,
        "bench": bench.config(),
        "run": {
            "soak_s": soaked.as_secs_f64(), "settled_in_s": settled_in, "preload_s": preload_secs,
        },
        "preloaded": preloaded,
        "invalid": invalid,
        "marks": marks,
        "server_stages": {
            "queue": stage_summary(&at_end.metrics, "queue"),
            "read": stage_summary(&at_end.metrics, "read"),
        },
        "merge_debt": debt,
        "tail_segments_mean": tails.iter().sum::<f64>() / tails.len() as f64,
        "timings": timings,
        "merges_ended": since(&before.ended, &at_end.ended),
        "waits_open": {
            "at_soak_end": open_summary(&open_at_end),
            "at_shutdown": open_summary(&open_at_shutdown),
            "ended_after_soak": open_past(&events, soak_end),
        },
        "idle_merge_builds": idle_merge_builds(&merge_work_timed, schedule),
        "slot_use": slot_use(
            &slot_spans,
            (pool != MergePool::Global).then_some(MERGE_BUILDS),
            soaked.as_secs_f64(),
        ),
        "clustered_merges_seen": sampled.clustered_merges_seen,
        "let_go": { "idle": at_end.let_go_idle - before.let_go_idle,
                    "budget": at_end.let_go_budget - before.let_go_budget },
        "memory": {
            "baseline_allocated": baseline_allocated, "allocated_peak": sampled.allocated_peak,
            "baseline_rss_anon": baseline_rss_anon, "rss_anon_peak": sampled.rss_anon_peak,
            "cgroup_peak_of_the_whole_process": cgroup_peak(),
            "draining_peak": sampled.draining_peak,
            "freeing_peak": sampled.freeing_peak,
            "preloaded_resident": preloaded_resident,
            "largest_merge_build_at_soak_end": largest_at_soak_end,
            "largest_merge_build_settled": settled.largest,
        },
        "memtable": {
            "unflushed_rows_peak": sampled.unflushed_rows_peak,
            "unflushed_bytes_peak": sampled.unflushed_bytes_peak,
            "tenant_unflushed_peak": sampled.tenant_unflushed_peak,
        },
        "cpu_s_by_pool": cpu,
        "steal": { "share": steal, "at_most": STEAL_AT_MOST,
                   "stolen_for": stolen_for, "for_at_most": STOLEN_FOR_AT_MOST,
                   "accounted": steal_accounted,
                   "peaks": steal_peaks(&steal_samples, bench.window.as_secs_f64()) },
        "throttled_usec": before.throttled.zip(at_end.throttled)
            .map(|(before, after)| after.saturating_sub(before)),
        "machine": machine(),
        "merge_builds": { "top_level_cpu_s": at_end.top_level_cpu.saturating_sub(before.top_level_cpu).as_secs_f64() },
        "memory_series": sampled.memory,
        "nice": { "checks": sampled.nice_checks, "wrong": sampled.nice_wrong },
    });
    (report, raw)
}

/// Whether what was read at `taken_at` was read as the soak ended, at
/// `soak_end`. Reading a node with many tenants takes moments; a reading
/// made after settling is out by minutes, and voids the run: no rule can
/// see where `arm` reads, so that the run itself does.
fn read_as_it_ended(taken_at: Option<Instant>, soak_end: Instant) -> bool {
    taken_at.is_some_and(|at| {
        at.saturating_duration_since(soak_end) < READ_WITHIN
            && soak_end.saturating_duration_since(at) < READ_WITHIN
    })
}

/// How close to the soak's end its end must be read.
const READ_WITHIN: Duration = Duration::from_secs(5);

#[cfg(test)]
mod reported_from_what_was_read {
    use super::*;

    fn a_soak() -> Soak {
        let started = Instant::now();
        let soak_end = started + Duration::from_secs(1_800);
        let at_end = Reading {
            taken_at: Some(soak_end),
            ..Reading::default()
        };
        Soak {
            started,
            soak_end,
            soaked: Duration::from_secs(1_800),
            before: Reading::default(),
            at_end,
            settled: Reading::default(),
            shutdown_at: soak_end + Duration::from_secs(300),
            waits_at_shutdown: Vec::new(),
            events: Vec::new(),
            sampled: Sampled::default(),
            preloaded: Vec::new(),
            preloaded_resident: 0,
            preload_secs: 0.0,
            settled_in: 0.0,
            budget: 0,
            baseline_allocated: 0,
            baseline_rss_anon: None,
            load_reported: true,
            steal_accounted: Some(true),
        }
    }

    fn report_of(soak: Soak) -> Value {
        let (bench, running) = (Bench::measured(), Running::now());
        let schedule = Schedule::of(&bench);
        reported(&bench, MergePool::Global, &running, &schedule, soak).0
    }

    fn event(soak: &Soak, timed: Timed, began: u64, took: u64) -> Event {
        Event {
            timed,
            took: Duration::from_secs(took),
            partition: "t0/0".into(),
            at: soak.started + Duration::from_secs(began + took),
        }
    }

    /// The figures set against the marks are those read at the soak's start
    /// and end: what settling adds is none of it.
    #[test]
    fn figures_are_those_read_at_the_soaks_two_ends() {
        let mut soak = a_soak();
        soak.before.ended = BTreeMap::from([("committed", 1)]);
        soak.at_end.ended = BTreeMap::from([("committed", 4)]);
        soak.settled.ended = BTreeMap::from([("committed", 9)]);
        (soak.before.let_go_budget, soak.at_end.let_go_budget) = (2, 5);
        soak.settled.let_go_budget = 9;
        (soak.before.backlogged, soak.at_end.backlogged) = (1, 3);
        soak.settled.backlogged = 9;
        soak.at_end.top_level_cpu = Duration::from_secs(7);
        soak.settled.top_level_cpu = Duration::from_secs(9);
        (soak.at_end.largest, soak.settled.largest) = (5, 9);
        let report = report_of(soak);
        assert_eq!(report["merges_ended"]["committed"], 3, "{report}");
        assert_eq!(report["let_go"]["budget"], 3, "{report}");
        assert_eq!(
            report["marks"]["backlog_refusals"]["refused"], 2,
            "{report}"
        );
        assert_eq!(report["merge_builds"]["top_level_cpu_s"], json!(7.0));
        let largest = &report["marks"]["node_memory"]["largest_merge_build_bytes"];
        assert_eq!(largest, &json!(5), "{report}");
    }

    /// A wait counts by when it began, in the mark as in the report, though
    /// it ended after the soak; and what began after it does not.
    #[test]
    fn a_wait_begun_in_the_soak_is_in_the_mark_though_it_ended_after() {
        let mut soak = a_soak();
        let long = event(&soak, Timed::FlushDue, 1_750, 150);
        let after = event(&soak, Timed::FlushDue, 1_810, 999);
        soak.events = vec![long, after];
        let report = report_of(soak);
        assert_eq!(report["marks"]["flush_due"]["due_p99_s"], json!(150.0));
        let ended = &report["waits_open"]["ended_after_soak"]["FlushDue"];
        assert_eq!(ended["count"], 1, "{report}");
    }

    /// A wait open at shutdown has no event, and is in the mark and the
    /// report at what it had come to: one begun after the soak is not.
    #[test]
    fn a_wait_open_at_shutdown_is_in_the_mark_and_the_report() {
        let mut soak = a_soak();
        let in_soak = soak.started + Duration::from_secs(1_700);
        soak.at_end.waits = vec![OpenWaits {
            flush_due: vec![in_soak],
            backlog: Some(in_soak),
        }];
        soak.waits_at_shutdown = vec![OpenWaits {
            flush_due: vec![in_soak, soak.soak_end + Duration::from_secs(1)],
            backlog: Some(in_soak),
        }];
        let report = report_of(soak);
        assert_eq!(report["marks"]["flush_due"]["due_p99_s"], json!(400.0));
        let open = &report["waits_open"];
        assert_eq!(open["at_soak_end"]["flush_due"]["longest_s"], json!(100.0));
        assert_eq!(open["at_shutdown"]["flush_due"]["count"], 1, "{open}");
        assert_eq!(open["at_shutdown"]["backlog"]["longest_s"], json!(400.0));
    }

    /// Steal is read end to end: what comes before the sampler's first read
    /// and after its last counts, by the soak's own two readings.
    #[test]
    fn steal_is_read_from_the_soaks_two_ends() {
        let mut soak = a_soak();
        soak.before.ticks = Some((0, 0));
        soak.at_end.ticks = Some((2_000, 180_000));
        soak.sampled.steal = vec![(10.0, 0, 1_000), (1_790.0, 0, 179_000)];
        let report = report_of(soak);
        assert_eq!(report["steal"]["stolen_for"], json!(10.0 / 1_800.0));
    }

    /// The soak's end read late voids the run, as `arm` reading it after
    /// settling would.
    #[test]
    fn a_soak_end_read_late_voids_the_run() {
        let mut soak = a_soak();
        soak.before.ticks = Some((0, 0));
        soak.at_end.ticks = Some((0, 1_000));
        soak.sampled.steal = vec![(10.0, 0, 500)];
        soak.at_end.taken_at = Some(soak.soak_end + Duration::from_secs(60));
        let report = report_of(soak);
        let invalid = report["invalid"].to_string();
        assert!(invalid.contains("not read as it ended"), "{invalid}");
    }

    /// The 95% gate is held to rows within the soak: all rows written, late
    /// ones with them, would make it empty.
    #[test]
    fn the_rate_is_of_the_rows_written_within_the_soak() {
        let (bench, running) = (Bench::measured(), Running::now());
        let schedule = Schedule::of(&bench);
        let writes = WritesDone {
            rows: 1_800_000,
            rows_in_soak: 1_600_000,
            ..WritesDone::default()
        };
        let load = Load {
            reads: Vec::new(),
            writes,
            read_failures: json!({}),
            readers_lost: 0,
            cpu_s: None,
        };
        let report = load_report(&bench, &running, &schedule, &load);
        let invalid = report["invalid"].to_string();
        assert!(invalid.contains("wrote 889 rows a second"), "{invalid}");
    }
}

/// One arm's node (§9): preload, serve through the soak while sampling
/// itself, report. Run by `scripts/step3-bench/run.sh`, beside
/// `step3_load`.
#[test]
#[ignore = "the step 3 bench: the fixture and half an hour or more, run by scripts/step3-bench/run.sh"]
fn step3_arm() {
    let bench = Bench::from_env();
    let running = Running::now();
    if bench.as_image {
        assert!(
            running.as_the_image(),
            "STEP3_AS_IMAGE: not run as the image runs: {running:?}"
        );
    }
    // The global pool as rayon would make it, named, so that its threads'
    // CPU is told from the others'.
    rayon::ThreadPoolBuilder::new()
        .thread_name(|n| format!("vellev-build-{n}"))
        .build_global()
        .expect("the global pool, made first");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name(RUNTIME_THREADS)
        .build()
        .unwrap();
    runtime.block_on(arm(bench, running));
}

async fn arm(bench: Bench, running: Running) {
    // Refused before anything is preloaded: the bench makes its node without
    // `Settings::check`, whose floor on the grace it departs from (§9.5).
    if let Some(problem) = bench.arm.pool(bench.threads).problem(MERGE_BUILDS) {
        panic!("arm {:?} cannot run here: {problem}", bench.arm);
    }
    std::fs::create_dir_all(&bench.out).unwrap();
    let _ = std::fs::remove_file(ready_path(&bench));
    let _ = std::fs::remove_file(load_path(&bench));
    // Refused before anything is preloaded, if the schedule is not whole.
    let schedule = Schedule::of(&bench);
    let data = bench.data.join(format!("arm-{}", bench.arm.name()));
    let _ = std::fs::remove_dir_all(&data);
    std::fs::create_dir_all(&data).unwrap();
    let coordinator = Arc::new(MemoryCoordinator::default());
    let blobs = store(&data).await;
    let mut passages = Passages::of(&bench.fixture);
    let preloading = Instant::now();
    let (preloaded, preloaded_resident, rows_of) =
        preload(&bench, &coordinator, &blobs, &mut passages).await;
    drop(passages);
    let preload_secs = preloading.elapsed().as_secs_f64();
    eprintln!("preloaded in {preload_secs:.0} s: {preloaded:?}");
    let budget = (preloaded_resident as f64 * bench.budget_share) as usize;
    // Let jemalloc give back what the preload freed before the baseline.
    tokio::time::sleep(Duration::from_secs(30)).await;
    let baseline_allocated = GLOBAL.held().map_or(0, |held| held.allocated as u64);
    let baseline_rss_anon = rss_anon();

    let pool = bench.arm.pool(bench.threads);
    let settings = Settings {
        flush_rows: bench.flush_rows,
        restore_window: Duration::from_secs(5 * 60),
        grace: Duration::from_secs(5 * 60),
        memory: Some(budget),
        merge_pool: pool,
        warm: None,
        ..Settings::new(NodeId::generate("node"))
    };
    let node = Arc::new(node(&coordinator, &blobs, settings));
    node.timings().keep_events(true);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let limits = Limits {
        results: 1_000,
        body: 32 * 1024 * 1024,
        embedding_urls: Arc::new([]),
        reads: 256,
        reads_wait: Duration::from_millis(500),
    };
    let app = App::new([KEY.to_string()].into(), limits);
    let (stop_serving, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(vellev::server::serve(listener, app.clone(), async {
        stopped.await.ok();
    }));
    app.start(node.clone());

    // The load generator is told where, and when the soak starts.
    let ready = Ready {
        url,
        start_unix_ms: unix_ms() + 5_000,
        soak_s: bench.soak.as_secs(),
        rows_of,
        sha: bench.sha.clone(),
    };
    written_whole(&ready_path(&bench), &serde_json::to_string(&ready).unwrap());
    let started = instant_at(ready.start_unix_ms);
    tokio::time::sleep_until(started.into()).await;
    let soak_end = started + bench.soak;
    let before = Reading::of(&node, &app);
    let soaking = Arc::new(AtomicBool::new(true));
    let events: Arc<Mutex<Vec<Event>>> = Arc::default();
    let sampled = Arc::new(Mutex::new(Sampled::default()));

    // The samplers: the node's memory and CPU, measured alone.
    let sampler = tokio::spawn({
        let (node, soaking, sampled, events) = (
            node.clone(),
            soaking.clone(),
            sampled.clone(),
            events.clone(),
        );
        async move {
            let mut tick: u64 = 0;
            while soaking.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(250)).await;
                tick += 1;
                let mut s = sampled.lock().unwrap();
                if let Some(held) = GLOBAL.held() {
                    s.allocated_peak = s.allocated_peak.max(held.allocated as u64);
                }
                if let Some(anon) = rss_anon() {
                    s.rss_anon_peak = s.rss_anon_peak.max(anon);
                }
                if !tick.is_multiple_of(4) {
                    continue;
                }
                let flushing = node.flushing();
                s.unflushed_rows_peak = s.unflushed_rows_peak.max(flushing.unflushed_rows);
                s.unflushed_bytes_peak = s.unflushed_bytes_peak.max(flushing.unflushed_bytes);
                s.draining_peak = s.draining_peak.max(node.merges().draining);
                let freeing = node.freeing();
                s.freeing_peak = s.freeing_peak.max(freeing);
                let at = started.elapsed().as_secs_f64();
                let above = vellev_storage::Merging::default().clustered_above as u64;
                let (mut writers_resident, mut writers) = (0, 0);
                for (tenant, figures) in node.tenant_figures() {
                    let tenant = tenant.to_string();
                    s.tenant_unflushed_peak = s.tenant_unflushed_peak.max(figures.unflushed);
                    let before = s
                        .clustered_rows
                        .insert(tenant.clone(), figures.clustered_rows);
                    if before.is_some_and(|rows| rows != figures.clustered_rows)
                        && figures.clustered_rows > above
                    {
                        s.clustered_merges_seen += 1;
                    }
                    writers_resident += figures.resident;
                    writers += 1;
                    if tick.is_multiple_of(40) {
                        let segments = (figures.clustered + figures.plain) as f64;
                        let point = (at, segments, figures.unmerged_rows as f64);
                        s.debt.entry(tenant).or_default().push(point);
                    }
                }
                if tick.is_multiple_of(40) {
                    if let Some((stolen, all)) = cpu_ticks() {
                        s.steal.push((at, stolen, all));
                    }
                    let held = GLOBAL.held();
                    let census = node.census().holding;
                    let merges = node.merges();
                    let let_go = node.let_go();
                    let freeing_peak = s.freeing_peak;
                    s.memory.push(json!({
                        "t": at,
                        "allocated": held.map(|held| held.allocated),
                        "allocator_resident": held.map(|held| held.resident),
                        "rss_anon": rss_anon(),
                        "resident_now": node.resident(),
                        "writers_now": writers,
                        "writers_resident_now": writers_resident,
                        "census_writers": census.writers,
                        "census_copies": census.copies,
                        "census_resident": census.resident,
                        "census_largest": census.largest,
                        "merge_builds_under_way": node.merge_builds().under_way(),
                        "draining": merges.draining,
                        "freeing": freeing,
                        "freeing_peak": freeing_peak,
                        "largest_merge_build": merges.largest,
                        "unflushed_bytes": flushing.unflushed_bytes,
                        "let_go_budget": let_go.for_budget,
                        "let_go_idle": let_go.idle,
                        "sweeps": node.sweeps(),
                        "kept_by_sweeps": node.tenants_kept_by_sweeps(),
                        "stopped_for_budget": merges.stopped_for_budget,
                    }));
                }
                drop(s);
                events.lock().unwrap().extend(node.timings().drain_events());
                if tick.is_multiple_of(240) {
                    let wrong = nice_wrong(pool);
                    let mut s = sampled.lock().unwrap();
                    s.nice_checks += 1;
                    s.nice_wrong.extend(wrong);
                }
            }
        }
    });

    // The soak runs its length; what it is held to is read as it ends, not
    // after settling, as the marks are windowed to the soak.
    tokio::time::sleep_until(soak_end.into()).await;
    soaking.store(false, Ordering::SeqCst);
    let soaked = started.elapsed();
    let at_end = Reading::of(&node, &app);
    let _ = tokio::time::timeout(Duration::from_secs(30), sampler).await;
    events.lock().unwrap().extend(node.timings().drain_events());
    // Served until the load generator has its last answers and has
    // reported, or for ten minutes at most.
    let load_reported = appeared(&load_path(&bench), Duration::from_secs(10 * 60))
        .await
        .is_some();
    // The node settles, outside every mark.
    let settling = Instant::now();
    while node.unsettled() > 0 && settling.elapsed() < bench.settle {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let settled_in = settling.elapsed().as_secs_f64();
    let settled = Reading::of(&node, &app);
    std::fs::write(
        bench.out.join(format!("metrics-{}.txt", bench.arm.name())),
        &settled.metrics,
    )
    .unwrap();
    // Shutdown flushes nothing: a wait open here ends in no event.
    let (shutdown_at, waits_at_shutdown) = (Instant::now(), node.waits_open());
    node.shutdown().await;
    events.lock().unwrap().extend(node.timings().drain_events());
    let _ = stop_serving.send(());
    let _ = serving.await;

    let steal_accounted = std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|stat| steal_accounted(&stat));
    let sampled = std::mem::take(&mut *sampled.lock().unwrap());
    let events = std::mem::take(&mut *events.lock().unwrap());
    let (report, raw) = reported(
        &bench,
        pool,
        &running,
        &schedule,
        Soak {
            started,
            soak_end,
            soaked,
            before,
            at_end,
            settled,
            shutdown_at,
            waits_at_shutdown,
            events,
            sampled,
            preloaded,
            preloaded_resident,
            preload_secs,
            settled_in,
            budget,
            baseline_allocated,
            baseline_rss_anon,
            load_reported,
            steal_accounted,
        },
    );
    std::fs::write(
        bench.out.join(format!("events-{}.json", bench.arm.name())),
        serde_json::to_string(&raw).unwrap(),
    )
    .unwrap();
    let path = bench.out.join(format!("arm-{}.json", bench.arm.name()));
    written_whole(&path, &serde_json::to_string_pretty(&report).unwrap());
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report["marks"]).unwrap()
    );
    eprintln!(
        "invalid: {}; written to {}",
        report["invalid"],
        path.display()
    );
    let _ = std::fs::remove_dir_all(&data);
}

/// What the load generator's run came to.
struct Load {
    reads: Vec<Read>,
    writes: WritesDone,
    read_failures: Value,
    readers_lost: usize,
    cpu_s: Option<f64>,
}

/// The load generator's report of what it came to.
fn load_report(bench: &Bench, running: &Running, schedule: &Schedule, load: &Load) -> Value {
    let Load {
        reads,
        writes,
        read_failures,
        readers_lost,
        cpu_s,
    } = load;
    let soaked = bench.soak.as_secs_f64();
    let answered = |answered: Answered| {
        reads
            .iter()
            .filter(|read| read.answered == answered)
            .count() as u64
    };
    let invalid = LoadValidity {
        as_the_image: running.as_the_image(),
        rows_written_a_second: writes.rows_in_soak as f64 / soaked,
        rows_a_second: bench.rows_a_second,
        reads: reads.len() as u64,
        failed_reads: answered(Answered::Failed),
        refused_reads: answered(Answered::Refused),
        readers_lost: *readers_lost,
        late_most: writes.late_most,
    }
    .invalid();
    let mut by_tenant = schedule.rows_by_tenant();
    by_tenant.values_mut().for_each(|rows| *rows = 0);
    by_tenant.extend(writes.by_tenant.clone());
    json!({
        "arm": bench.arm.name(),
        "running": running,
        "process": process(bench),
        "sha": bench.sha,
        "bench": bench.config(),
        "cpu_s": cpu_s,
        "reads": reads_summary(reads, soaked, bench.window.as_secs_f64()),
        "read_failures": read_failures,
        "writes": {
            "rows": writes.rows, "rows_in_soak": writes.rows_in_soak,
            "rows_a_second": writes.rows_in_soak as f64 / soaked,
            "batches": writes.batches, "refused": writes.refused, "failed": writes.failed,
            "behind_most_s": writes.behind_most, "worst_lag": writes.worst_lag,
            "slowest_batch": writes.slowest_batch.as_ref().map(|(_, batch)| batch),
            "batches_late": writes.late, "late_most_s": writes.late_most,
            "by_tenant": by_tenant, "scheduled_by_tenant": schedule.rows_by_tenant(),
            "failures": writes.failures.report(),
        },
        "invalid": invalid,
    })
}

/// One arm's load generator: told where the node serves through the file
/// its `step3_arm` writes, it reads and writes through the soak, in a
/// process of its own so that the node's memory and CPU are the node's
/// alone. Run by `scripts/step3-bench/run.sh`.
#[test]
#[ignore = "the step 3 bench's load generator, run by scripts/step3-bench/run.sh"]
fn step3_load() {
    let bench = Bench::from_env();
    let running = Running::now();
    if bench.as_image {
        assert!(
            running.as_the_image(),
            "STEP3_AS_IMAGE: not run as the image runs: {running:?}"
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(load(bench, running));
}

async fn load(bench: Bench, running: Running) {
    let schedule = Schedule::of(&bench);
    let cpu_before = process_ticks();
    // The fixture first, so that the soak starts with nothing left to
    // load: the node gives only seconds between ready and the start.
    let queries: Arc<Vec<Vec<f32>>> = Arc::new(
        Fixture::at(&bench.fixture)
            .queries("queries")
            .expect("the fixture's queries")
            .into_iter()
            .map(|query| query.vector)
            .collect(),
    );
    let passages_of: Vec<Passages> = (0..bench.active)
        .map(|_| Passages::of(&bench.fixture))
        .collect();
    let ready = appeared(&ready_path(&bench), bench.ready_within)
        .await
        .unwrap_or_else(|| panic!("the node was not ready within {:?}", bench.ready_within));
    let ready: Ready = serde_json::from_str(&ready).unwrap();
    assert_eq!(
        (ready.soak_s, ready.sha.as_str()),
        (bench.soak.as_secs(), bench.sha.as_str()),
        "the node and the load generator were not set alike"
    );
    let rows_of = Arc::new(ready.rows_of);
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .pool_max_idle_per_host(bench.readers + bench.active + 4)
        .build()
        .unwrap();
    let started = instant_at(ready.start_unix_ms);
    let soak_end = started + bench.soak;
    let url = ready.url;

    // The writers, one for each place in the active set, each its share of
    // the rate on a fixed schedule: its k-th batch to the tenant the
    // schedule gives, the same rows in every arm. One a place, so that one
    // tenant slow to take a write holds up only its own share, as one
    // client among many would. A batch refused is tried again after the
    // wait it was told, up to a minute past the soak's end; a batch not
    // written by then leaves its tenant short, and the arm is not compared.
    let writes = Arc::new(Mutex::new(WritesDone::default()));
    let mut writers = Vec::new();
    for (place, passages) in passages_of.into_iter().enumerate() {
        let passages = Arc::new(Mutex::new(passages));
        writers.push(tokio::spawn({
            let (bench, http, url, writes) = (bench.clone(), http.clone(), url.clone(), writes.clone());
            let rows_of = rows_of.clone();
            async move {
                let mut random = Random(bench.seed ^ (0xa11c_e000 + place as u64));
                let batch = bench.batch as u64;
                for k in 0..schedule.batches {
                    let due = started + schedule.due(k);
                    let now = Instant::now();
                    if due > now {
                        tokio::time::sleep(due - now).await;
                    } else {
                        let behind = (now - due).as_secs_f64();
                        let mut done = writes.lock().unwrap();
                        if behind > done.behind_most {
                            done.behind_most = behind;
                            done.worst_lag = Some(json!({
                                "behind_s": behind, "place": place, "batch": k,
                                "tenant": tenant_id(schedule.tenant(place, k)).to_string(),
                                "at_s": (now - started).as_secs_f64(),
                            }));
                        }
                    }
                    if past_late(Instant::now(), soak_end) {
                        break;
                    }
                    let at = schedule.tenant(place, k);
                    let picks: Vec<(bool, u64)> = (0..batch)
                        .map(|_| (random.below(5) == 0, random.next()))
                        .collect();
                    // One in five replaces a preloaded row; the rest are
                    // new, each id this place's and this batch's.
                    let preloaded = rows_of[at];
                    let first_new = preloaded + ((place as u64) << 40) + k * batch;
                    let passages = passages.clone();
                    let changes = tokio::task::spawn_blocking(move || {
                        let mut passages = passages.lock().unwrap();
                        picks
                            .into_iter()
                            .enumerate()
                            .map(|(i, (replaces, pick))| {
                                let row = passages.next_row();
                                let n = match replaces {
                                    true => pick % preloaded.max(1),
                                    false => first_new + i as u64,
                                };
                                json!({
                                    "op": "upsert",
                                    "collection": COLLECTION,
                                    "id": object_id(at, n).0,
                                    "properties": { "text": row.text, "n": n, "bucket": row.bucket },
                                    "vector": row.vector,
                                })
                            })
                            .collect::<Vec<Value>>()
                    });
                    let body = json!({ "changes": changes.await.unwrap() });
                    let route = format!("{url}/v1/tenants/{}/changes", tenant_id(at));
                    let sent = Instant::now();
                    loop {
                        match http_json(&http, route.clone(), &body).await {
                            Ok(answer) if answer.status().is_success() => {
                                let now = Instant::now();
                                let took = (now - sent).as_secs_f64();
                                let mut done = writes.lock().unwrap();
                                if done.slowest_batch.as_ref().is_none_or(|(most, _)| took > *most) {
                                    done.slowest_batch = Some((took, json!({
                                        "took_s": took, "place": place, "batch": k,
                                        "tenant": tenant_id(at).to_string(),
                                        "at_s": (sent - started).as_secs_f64(),
                                    })));
                                }
                                if now > soak_end {
                                    let late = (now - soak_end).as_secs_f64();
                                    done.late += 1;
                                    done.late_most = done.late_most.max(late);
                                }
                                if now <= soak_end {
                                    done.rows_in_soak += batch;
                                }
                                done.rows += batch;
                                done.batches += 1;
                                *done.by_tenant.entry(tenant_id(at).to_string()).or_default() +=
                                    batch;
                                break;
                            }
                            Ok(answer) if matches!(answer.status().as_u16(), 429 | 503) => {
                                writes.lock().unwrap().refused += 1;
                                let wait = answer
                                    .headers()
                                    .get("retry-after")
                                    .and_then(|wait| wait.to_str().ok()?.parse::<u64>().ok())
                                    .map_or(Duration::from_secs(1), Duration::from_secs);
                                if past_late(Instant::now() + wait, soak_end) {
                                    break;
                                }
                                tokio::time::sleep(wait).await;
                            }
                            answer => {
                                let answer = answer.map(|answer| answer.status());
                                let said = format!("batch {k} of place {place}: {answer:?}");
                                eprintln!("a write failed: {said}");
                                let mut done = writes.lock().unwrap();
                                done.failed += 1;
                                done.failures.note(Failures::kind_of(&answer), said);
                                break;
                            }
                        }
                    }
                }
            }
        }));
    }

    // The readers, back to back, each to a seeded tenant of those active
    // by the clock: reads count no rows.
    let reads: Vec<Arc<Mutex<Vec<Read>>>> = (0..bench.readers).map(|_| Arc::default()).collect();
    let read_failures: Arc<Mutex<Failures>> = Arc::default();
    let mut readers = Vec::new();
    for (reader, done) in reads.iter().enumerate() {
        readers.push(tokio::spawn({
            let (http, url, done, queries) =
                (http.clone(), url.clone(), done.clone(), queries.clone());
            let failures = read_failures.clone();
            let mut random = Random(bench.seed ^ (reader as u64 + 1).wrapping_mul(0x9e37_79b9));
            async move {
                tokio::time::sleep_until(started.into()).await;
                while Instant::now() < soak_end {
                    let active = schedule.active_at(started.elapsed());
                    let at = active[random.below(active.len() as u64) as usize];
                    let query = &queries[random.below(queries.len() as u64) as usize];
                    let body = json!({ "vector": query, "limit": 10 });
                    let route = format!(
                        "{url}/v1/tenants/{}/collections/{COLLECTION}/nearest",
                        tenant_id(at)
                    );
                    let began = Instant::now();
                    let answer = match http_json(&http, route, &body).await {
                        Ok(answer) => {
                            let status = answer.status();
                            answer.bytes().await.map(|_| status)
                        }
                        Err(error) => Err(error),
                    };
                    let answered = match &answer {
                        Ok(status) if status.is_success() => Answered::Yes,
                        Ok(status) if matches!(status.as_u16(), 429 | 503) => Answered::Refused,
                        _ => Answered::Failed,
                    };
                    if answered == Answered::Failed {
                        let said = format!(
                            "a read of {} at {:?}: {answer:?}",
                            tenant_id(at),
                            started.elapsed()
                        );
                        failures
                            .lock()
                            .unwrap()
                            .note(Failures::kind_of(&answer), said);
                    }
                    done.lock().unwrap().push(Read {
                        at: (began - started).as_secs_f32(),
                        ms: began.elapsed().as_secs_f32() * 1_000.0,
                        answered,
                    });
                }
            }
        }));
    }
    for writer in writers {
        writer.await.unwrap();
    }
    let mut readers_lost = 0;
    for reader in readers {
        let ended = tokio::time::timeout(Duration::from_secs(150), reader).await;
        if !matches!(ended, Ok(Ok(()))) {
            readers_lost += 1;
        }
    }
    let reads: Vec<Read> = reads
        .iter()
        .flat_map(|done| std::mem::take(&mut *done.lock().unwrap()))
        .collect();
    let writes = std::mem::take(&mut *writes.lock().unwrap());
    let cpu_s = cpu_before
        .zip(process_ticks())
        .map(|(before, after)| after.saturating_sub(before) as f64 / 100.0);
    let read_failures = read_failures.lock().unwrap().report();
    let report = load_report(
        &bench,
        &running,
        &schedule,
        &Load {
            reads,
            writes,
            read_failures,
            readers_lost,
            cpu_s,
        },
    );
    written_whole(
        &load_path(&bench),
        &serde_json::to_string_pretty(&report).unwrap(),
    );
    eprintln!("load invalid: {}", report["invalid"]);
}

/// An arm's two reports, as compared: the node's, and the load
/// generator's if it wrote one.
struct ArmRun {
    name: String,
    node: Value,
    load: Option<Value>,
}

/// How an arm was set to run, its name apart.
fn alike(report: &Value) -> Value {
    let mut config = report["bench"].clone();
    if let Some(config) = config.as_object_mut() {
        config.remove("arm");
    }
    config
}

/// Why each arm is not compared: its own reasons, its load generator's,
/// and any way it was not run as every other arm was. Nothing is held to
/// one arm's run: settings and commits must be the same across all the
/// arms, the rows written to each tenant those the schedule gives, and
/// the preload §9.5's (review of #169).
fn not_compared(runs: &[ArmRun]) -> BTreeMap<String, Vec<String>> {
    let strings = |value: &Value| -> Vec<String> {
        value
            .as_array()
            .map(|reasons| {
                reasons
                    .iter()
                    .filter_map(|why| why.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut reasons = BTreeMap::new();
    for run in runs {
        let mut why = strings(&run.node["invalid"]);
        let config = alike(&run.node);
        let sha = run.node["sha"].as_str().unwrap_or("");
        if sha.is_empty() {
            why.push("no commit stamped".into());
        }
        if sha.ends_with("-dirty") {
            why.push("built from a tree with uncommitted changes".into());
        }
        match &run.load {
            None => why.push("no load generator report".into()),
            Some(load) => {
                why.extend(strings(&load["invalid"]));
                if alike(load) != config || load["sha"] != run.node["sha"] {
                    why.push("its node and load generator were not set alike".into());
                }
                let written = &load["writes"]["by_tenant"];
                if written != &load["writes"]["scheduled_by_tenant"] || written.is_null() {
                    why.push(format!(
                        "wrote {written} to its tenants, not the schedule's {}",
                        load["writes"]["scheduled_by_tenant"]
                    ));
                }
            }
        }
        for other in runs.iter().filter(|other| other.name != run.name) {
            let theirs = alike(&other.node);
            if theirs != config {
                let differ: Vec<&String> = config
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter(|(key, value)| theirs.get(key.as_str()) != Some(value))
                    .map(|(key, _)| key)
                    .collect();
                why.push(format!("set unlike arm {} in {differ:?}", other.name));
            }
            if other.node["sha"] != run.node["sha"] {
                why.push(format!("built from another commit than arm {}", other.name));
            }
            if other.node["machine"] != run.node["machine"] {
                why.push(format!("run on another machine than arm {}", other.name));
            }
        }
        let scale = config["scale"].as_f64().unwrap_or(f64::NAN);
        let expected = preload_expected(scale);
        let preloaded: Vec<Preloaded> =
            serde_json::from_value(run.node["preloaded"].clone()).unwrap_or_default();
        let held: BTreeMap<String, Vec<Segment>> = preloaded
            .into_iter()
            .map(|tenant| (tenant.tenant, tenant.segments))
            .collect();
        if held != expected {
            why.push("its preload is not §9.5's".into());
        }
        reasons.insert(run.name.clone(), why);
    }
    reasons
}

/// A run of each arm as the rules need it: valid, every tenant written
/// as scheduled and preloaded as §9.5 says.
#[cfg(test)]
fn a_run(name: &str) -> ArmRun {
    let bench = Bench {
        arm: Arm::named(name),
        ..Bench::measured()
    };
    let schedule = Schedule::of(&bench);
    let preloaded: Vec<Value> = preload_expected(1.0)
        .into_iter()
        .map(|(tenant, segments)| {
            json!({ "tenant": tenant, "rows": 0, "clustered": 0, "clustered_rows": 0,
                    "plain": 0, "plain_rows": 0, "segments": segments })
        })
        .collect();
    ArmRun {
        name: name.into(),
        node: json!({ "bench": bench.config(), "sha": "abc", "invalid": [],
                      "preloaded": preloaded, "machine": { "cpu_model": "same" } }),
        load: Some(
            json!({ "bench": bench.config(), "sha": "abc", "invalid": [],
                           "writes": { "by_tenant": schedule.rows_by_tenant(),
                                       "scheduled_by_tenant": schedule.rows_by_tenant() } }),
        ),
    }
}

/// Arms are compared only if run alike: the same settings, `arm` apart,
/// the same threads, one commit, each tenant written the rows the
/// schedule gives, and the preload §9.5's. An arm off in any of these is
/// not compared, and none is the reference: arm (a)'s own preload going
/// wrong leaves the others' as they were (review of #169).
#[test]
fn arms_are_compared_only_if_run_alike() {
    let names = ["a", "b", "c", "d", "b2"];
    let runs = || names.map(a_run).into_iter().collect::<Vec<_>>();
    let none = not_compared(&runs());
    assert!(none.values().all(Vec::is_empty), "{none:?}");

    let mut flush_rows = runs();
    flush_rows[2].node["bench"]["flush_rows"] = json!(50_000);
    flush_rows[2].load.as_mut().unwrap()["bench"]["flush_rows"] = json!(50_000);
    let found = not_compared(&flush_rows);
    assert!(
        found["c"].iter().any(|why| why.contains("flush_rows")),
        "{found:?}"
    );
    assert!(
        found["b"].iter().any(|why| why.contains("unlike arm c")),
        "{found:?}"
    );

    let mut threads = runs();
    threads[1].node["bench"]["threads"] = json!(16);
    assert!(not_compared(&threads)["b"]
        .iter()
        .any(|why| why.contains("threads")));

    let mut commit = runs();
    commit[3].node["sha"] = json!("def");
    commit[3].load.as_mut().unwrap()["sha"] = json!("def");
    let found = not_compared(&commit);
    assert!(
        found["d"].iter().any(|why| why.contains("another commit")),
        "{found:?}"
    );
    let mut unstamped = runs();
    for run in &mut unstamped {
        run.node["sha"] = json!("");
        run.load.as_mut().unwrap()["sha"] = json!("");
    }
    assert!(not_compared(&unstamped)["a"].contains(&"no commit stamped".to_string()));

    let mut dirty = runs();
    for run in &mut dirty {
        run.node["sha"] = json!("abc-dirty");
        run.load.as_mut().unwrap()["sha"] = json!("abc-dirty");
    }
    let found = not_compared(&dirty);
    assert_eq!(
        found["a"],
        vec!["built from a tree with uncommitted changes"]
    );

    let mut machines = runs();
    machines[2].node["machine"]["cpu_model"] = json!("another");
    let found = not_compared(&machines);
    assert!(
        found["c"].iter().any(|why| why.contains("another machine")),
        "{found:?}"
    );

    let mut short = runs();
    short[1].load.as_mut().unwrap()["writes"]["by_tenant"]["t3"] = json!(299_900);
    let found = not_compared(&short);
    assert!(
        found["b"]
            .iter()
            .any(|why| why.contains("not the schedule's")),
        "{found:?}"
    );
    assert!(found["c"].is_empty(), "{found:?}");

    let mut preload = runs();
    preload[0].node["preloaded"][2]["segments"][0]["rows"] = json!(79_999);
    let found = not_compared(&preload);
    assert_eq!(found["a"], vec!["its preload is not §9.5's"]);
    assert!(
        found["b"].is_empty(),
        "held to arm (a)'s preload: {found:?}"
    );

    let mut unreported = runs();
    unreported[4].load = None;
    assert_eq!(
        not_compared(&unreported)["b2"],
        vec!["no load generator report"]
    );
}

/// The backlog refusals every arm is held to: arm (a)'s, only if (a) is
/// compared. An invalid (a)'s refusals bound nothing (review of #169).
fn backlog_bound(runs: &[ArmRun], invalid: &BTreeMap<String, Vec<String>>) -> Option<u64> {
    let valid = invalid.get("a").is_some_and(Vec::is_empty);
    runs.iter()
        .find(|run| run.name == "a")
        .filter(|_| valid)
        .and_then(|a| a.node["marks"]["backlog_refusals"]["refused"].as_u64())
}

#[test]
fn an_invalid_arm_as_refusals_bound_nothing() {
    let mut a = a_run("a");
    a.node["marks"] = json!({ "backlog_refusals": { "refused": 7 } });
    let runs = vec![a, a_run("b")];
    let mut invalid = BTreeMap::from([("a".to_string(), vec![]), ("b".to_string(), vec![])]);
    assert_eq!(backlog_bound(&runs, &invalid), Some(7));
    invalid.insert(
        "a".to_string(),
        vec!["a smoke, not the measurement".to_string()],
    );
    assert_eq!(backlog_bound(&runs, &invalid), None);
    assert_eq!(backlog_bound(&runs[1..], &BTreeMap::new()), None);
}

/// The noise floor: how far (b) lies from itself run again, as a share of
/// (b). Null unless both runs are valid: a spread from a run not compared
/// is no floor. It decides nothing (review of #169).
fn noise_floor(b: Option<Compared>, b2: Option<Compared>) -> Value {
    match (b, b2) {
        (Some(b), Some(b2)) if b.valid && b2.valid => json!({
            "read_p99_spread": (b.read_p99_ms - b2.read_p99_ms).abs() / b.read_p99_ms,
            "end_segments_spread": (b.end_segments - b2.end_segments).abs() / b.end_segments,
        }),
        _ => json!(null),
    }
}

#[test]
fn the_noise_floor_is_bs_spread_from_itself_when_both_runs_count() {
    let run = |valid, read_p99_ms, end_segments| Compared {
        valid,
        all_marks: Some(true),
        read_p99_ms,
        end_segments,
    };
    let floor = noise_floor(Some(run(true, 100.0, 10.0)), Some(run(true, 90.0, 12.0)));
    assert_eq!(floor["read_p99_spread"], json!(0.1), "{floor}");
    assert_eq!(floor["end_segments_spread"], json!(0.2), "{floor}");
    let invalid = Some(run(false, 90.0, 12.0));
    assert_eq!(
        noise_floor(Some(run(true, 100.0, 10.0)), invalid),
        json!(null)
    );
    assert_eq!(
        noise_floor(invalid, Some(run(true, 100.0, 10.0))),
        json!(null)
    );
    assert_eq!(noise_floor(Some(run(true, 100.0, 10.0)), None), json!(null));
}

/// Whether an arm met every mark: `None` if that rests on the backlog
/// bound and there is none, as when arm (a) is not compared.
fn all_marks(marks: &Value, backlog_within_a: Option<bool>) -> Option<bool> {
    let passed = |mark: &str| marks[mark]["passed"].as_bool().unwrap_or(false);
    let alone = passed("merge_debt")
        && passed("flush_wait")
        && passed("flush_due")
        && passed("node_memory")
        && passed("eviction_flush");
    match (alone, backlog_within_a) {
        (false, _) | (_, Some(false)) => Some(false),
        (true, None) => None,
        (true, Some(true)) => Some(true),
    }
}

/// What "beats (b)" needs of an arm as compared (§3, §9.6).
#[derive(Clone, Copy, Debug)]
struct Compared {
    valid: bool,
    all_marks: Option<bool>,
    read_p99_ms: f64,
    end_segments: f64,
}

/// How an arm stands against (b).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beats {
    Yes,
    No,
    /// By §3's margin, but by no more than (b) lies from itself.
    WithinNoise,
    Undecided,
}

impl Beats {
    fn said(self) -> &'static str {
        match self {
            Beats::Yes => "yes",
            Beats::No => "no",
            Beats::WithinNoise => "within noise",
            Beats::Undecided => "undecided",
        }
    }
}

/// Whether `arm` beats (b): read p99 at least 15% better and end-of-soak
/// segments a tenant within 10% of (b)'s, with every mark met. (b) need
/// only be valid, not meet every mark: §3 holds the arm, not (b), to them.
/// Undecided if either is missing or not compared, or the arm's marks rest
/// on a backlog bound there is none of. A margin no greater than `spread`,
/// how far (b) lies from itself run again, is within noise; with no spread
/// to go by, a margin says nothing, and it is undecided (review of #169).
fn beats_b(arm: Option<Compared>, b: Option<Compared>, spread: Option<f64>) -> Beats {
    let (Some(arm), Some(b)) = (arm, b) else {
        return Beats::Undecided;
    };
    if !arm.valid || !b.valid {
        return Beats::Undecided;
    }
    let better =
        arm.read_p99_ms <= 0.85 * b.read_p99_ms && arm.end_segments <= 1.10 * b.end_segments;
    match (arm.all_marks, better) {
        (None, _) => Beats::Undecided,
        (Some(false), _) | (Some(true), false) => Beats::No,
        (Some(true), true) => match spread {
            None => Beats::Undecided,
            Some(spread) if (b.read_p99_ms - arm.read_p99_ms) / b.read_p99_ms <= spread => {
                Beats::WithinNoise
            }
            Some(_) => Beats::Yes,
        },
    }
}

/// The winner by §3: an arm that beats (b), the better of two that both
/// do; (b) if neither does and (b) met every mark itself, as no arm that
/// failed its marks is named; undecided if an arm or (b)'s own marks are.
fn winner(c: Beats, d: Beats, p99_c: f64, p99_d: f64, b_marks: Option<bool>) -> &'static str {
    match (c, d) {
        (Beats::Undecided, _) | (_, Beats::Undecided) => "undecided",
        (Beats::Yes, Beats::Yes) if p99_c <= p99_d => "c",
        (Beats::Yes, Beats::Yes) => "d",
        (Beats::Yes, _) => "c",
        (_, Beats::Yes) => "d",
        _ => match b_marks {
            Some(true) => "b",
            Some(false) => "none met every mark",
            None => "undecided",
        },
    }
}

/// "Beats (b)" asks (b) to be valid, not to meet every mark; a margin is
/// held to (b)'s spread from itself; and what cannot be decided is said to
/// be undecided, never (b) by default, as (b) is never named having failed
/// its marks (reviews of #169).
#[test]
fn beats_b_asks_b_be_valid_not_that_it_pass_and_says_undecided() {
    let run = |valid, all_marks, read_p99_ms| Compared {
        valid,
        all_marks,
        read_p99_ms,
        end_segments: 10.0,
    };
    let quiet = Some(0.05);
    // (b) valid but failing a mark: (c), 20% better and meeting every
    // mark, beats it.
    let b = run(true, Some(false), 100.0);
    let beats = |arm| beats_b(Some(arm), Some(b), quiet);
    assert_eq!(beats(run(true, Some(true), 80.0)), Beats::Yes);
    assert_eq!(beats(run(true, Some(true), 90.0)), Beats::No);
    assert_eq!(beats(run(true, Some(false), 50.0)), Beats::No);
    // A margin no greater than (b)'s spread from itself is noise, and with
    // no spread at all it says nothing.
    let fast = run(true, Some(true), 80.0);
    assert_eq!(beats_b(Some(fast), Some(b), Some(0.25)), Beats::WithinNoise);
    assert_eq!(beats_b(Some(fast), Some(b), None), Beats::Undecided);
    // Undecided: (b) or the arm not compared, missing, or resting on no
    // backlog bound.
    let fast = run(true, Some(true), 50.0);
    let invalid_b = run(false, Some(true), 100.0);
    assert_eq!(
        beats_b(Some(fast), Some(invalid_b), quiet),
        Beats::Undecided
    );
    assert_eq!(
        beats_b(Some(run(false, Some(true), 50.0)), Some(b), quiet),
        Beats::Undecided
    );
    assert_eq!(beats_b(None, Some(b), quiet), Beats::Undecided);
    assert_eq!(beats_b(Some(fast), None, quiet), Beats::Undecided);
    assert_eq!(
        beats_b(Some(run(true, None, 50.0)), Some(b), quiet),
        Beats::Undecided
    );
    // The winner: never a (b) that failed its marks.
    let met = Some(true);
    assert_eq!(winner(Beats::Yes, Beats::Yes, 70.0, 80.0, met), "c");
    assert_eq!(winner(Beats::Yes, Beats::Yes, 90.0, 80.0, met), "d");
    assert_eq!(winner(Beats::No, Beats::Yes, 70.0, 80.0, met), "d");
    assert_eq!(winner(Beats::No, Beats::No, 70.0, 80.0, met), "b");
    assert_eq!(winner(Beats::WithinNoise, Beats::No, 70.0, 80.0, met), "b");
    assert_eq!(
        winner(Beats::No, Beats::No, 70.0, 80.0, Some(false)),
        "none met every mark"
    );
    assert_eq!(winner(Beats::No, Beats::No, 70.0, 80.0, None), "undecided");
    assert_eq!(
        winner(Beats::Yes, Beats::Undecided, 70.0, 80.0, met),
        "undecided"
    );
    assert_eq!(
        winner(Beats::Undecided, Beats::Undecided, 70.0, 80.0, met),
        "undecided"
    );
    // No backlog bound, as when (a) is not compared: an arm meeting every
    // other mark is undecided, one failing another is not.
    let marks = json!({ "merge_debt": { "passed": true }, "flush_wait": { "passed": true },
                        "flush_due": { "passed": true }, "node_memory": { "passed": true },
                        "eviction_flush": { "passed": true } });
    assert_eq!(all_marks(&marks, None), None);
    assert_eq!(all_marks(&marks, Some(true)), Some(true));
    let mut failing = marks.clone();
    failing["flush_wait"]["passed"] = json!(false);
    assert_eq!(all_marks(&failing, None), Some(false));
    let mut late = marks.clone();
    late["flush_due"]["passed"] = json!(false);
    assert_eq!(all_marks(&late, Some(true)), Some(false));
}

/// What is said of an arm's run. One on a machine that does not account
/// steal stays valid, as §3 names that runner, and says so: nothing has
/// shown its host took nothing.
fn run_said(why: &[String], steal_accounted: Option<bool>) -> String {
    let said = match why.is_empty() {
        true => "valid".to_string(),
        false => why.join("; "),
    };
    match steal_accounted {
        Some(true) => said,
        _ => format!("{said} (steal unmeasured)"),
    }
}

#[test]
fn a_run_where_steal_is_unaccounted_says_so_beside_valid() {
    assert_eq!(run_said(&[], Some(true)), "valid");
    assert_eq!(run_said(&[], Some(false)), "valid (steal unmeasured)");
    assert_eq!(run_said(&[], None), "valid (steal unmeasured)");
    let why = ["a smoke, not the measurement".to_string()];
    assert_eq!(
        run_said(&why, Some(false)),
        "a smoke, not the measurement (steal unmeasured)"
    );
}

/// The arms side by side (§9.6): each mark, arm (a) as the backlog's
/// bound if it is compared, the noise floor from (b) run twice, and which
/// arm wins by "beats (b)". Reads `STEP3_OUT/arm-*.json` and
/// `load-*.json`, and writes `summary.md` and `verdict.json` beside them.
#[test]
#[ignore = "the step 3 bench's comparison, run by scripts/step3-bench/run.sh"]
fn step3_compare() {
    let out = PathBuf::from(std::env::var("STEP3_OUT").expect("STEP3_OUT must be set"));
    let read = |name: String| -> Option<Value> {
        let text = std::fs::read_to_string(out.join(name)).ok()?;
        Some(serde_json::from_str(&text).unwrap())
    };
    let runs: Vec<ArmRun> = ["a", "b", "c", "d", "b2"]
        .into_iter()
        .filter_map(|name| {
            Some(ArmRun {
                name: name.to_string(),
                node: read(format!("arm-{name}.json"))?,
                load: read(format!("load-{name}.json")),
            })
        })
        .collect();
    assert!(!runs.is_empty(), "no arm-*.json in {}", out.display());
    let invalid = not_compared(&runs);
    let p99 = |run: &ArmRun| {
        run.load
            .as_ref()
            .and_then(|load| load["reads"]["p99_ms"].as_f64())
            .unwrap_or(f64::NAN)
    };
    // Backlog refusals are held to arm (a)'s only if (a) is compared.
    let control_refused = backlog_bound(&runs, &invalid);
    let mut rows = Vec::new();
    let mut idle_builds = Vec::new();
    let mut slots = Vec::new();
    let mut verdicts = serde_json::Map::new();
    let mut compared: BTreeMap<String, Compared> = BTreeMap::new();
    for run in &runs {
        let marks = &run.node["marks"];
        let refused = marks["backlog_refusals"]["refused"]
            .as_u64()
            .unwrap_or(u64::MAX);
        let backlog = control_refused.map(|control| refused <= control);
        let all = all_marks(marks, backlog);
        let why = &invalid[&run.name];
        let idle = &run.node["idle_merge_builds"];
        idle_builds.push(format!(
            "{}: {} of {} gathers and builds, {} of their wall-clock seconds{}",
            run.name,
            idle["idle"],
            idle["spans"],
            idle["idle_share_of_wall_s"],
            if idle["large"] == json!(true) {
                " (large: read its flush marks with care)"
            } else {
                ""
            },
        ));
        let used = &run.node["slot_use"];
        slots.push(format!(
            "{}: {} s of merge work, {} s of it alone, the pools {} idle; lone builds took at most {} s; the index's encode at most {} s on an async thread",
            run.name,
            used["busy_s"],
            used["lone_s"],
            used["idle_share"],
            used["lone_builds"]["max"],
            run.node["timings"]["merge.encode"]["max"],
        ));
        compared.insert(
            run.name.clone(),
            Compared {
                valid: why.is_empty(),
                all_marks: all,
                read_p99_ms: p99(run),
                end_segments: run.node["tail_segments_mean"].as_f64().unwrap_or(f64::NAN),
            },
        );
        let passed = |mark: &str| marks[mark]["passed"].as_bool().unwrap_or(false);
        let mark = |ok: bool| if ok { "pass" } else { "FAIL" };
        let stage = |stage: &str| {
            run.node["server_stages"][stage]["p99_at_most_s"]
                .to_string()
                .trim_matches('"')
                .to_string()
        };
        rows.push(format!(
            "| {} | {} | {:.1} | {} | {} | {:.2} | {} | {} | {} | {} | {} | {} | {} | {} |",
            run.name,
            run.node["merge_pool"].as_str().unwrap_or("?"),
            p99(run),
            stage("queue"),
            stage("read"),
            run.node["tail_segments_mean"].as_f64().unwrap_or(f64::NAN),
            mark(passed("merge_debt")),
            mark(passed("flush_wait")),
            mark(passed("flush_due")),
            backlog.map_or("undecided", mark),
            mark(passed("node_memory")),
            mark(passed("eviction_flush")),
            all.map_or("undecided", mark),
            run_said(why, run.node["steal"]["accounted"].as_bool()),
        ));
        verdicts.insert(
            run.name.clone(),
            json!({ "all_marks": all, "backlog_within_a": backlog, "invalid": why,
                    "steal_accounted": run.node["steal"]["accounted"],
                    "idle_merge_work_large": idle["large"],
                    "slot_use": run.node["slot_use"] }),
        );
    }
    // The noise floor: (b) against itself, deciding only what a margin
    // inside it is worth.
    let noise = noise_floor(compared.get("b").copied(), compared.get("b2").copied());
    let spread = noise["read_p99_spread"].as_f64();
    let (b, against_b) = (compared.get("b").copied(), |arm: &str| {
        beats_b(
            compared.get(arm).copied(),
            compared.get("b").copied(),
            spread,
        )
    });
    let (c, d) = (against_b("c"), against_b("d"));
    let p99_of = |name: &str| compared.get(name).map_or(f64::NAN, |arm| arm.read_p99_ms);
    let b_marks = b.filter(|b| b.valid).and_then(|b| b.all_marks);
    let winner = winner(c, d, p99_of("c"), p99_of("d"), b_marks);
    let mut summary = String::from(
        "| arm | merge pool | read p99 ms | server queue p99 s | server read p99 s | tail segments/tenant | merge debt | flush wait | flush due | backlog <= (a) | node memory | eviction flush | all marks | run |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    summary.push_str(&rows.join("\n"));
    summary.push_str(&format!(
        "\n\n(c) beats (b): {}. (d) beats (b): {}. Winner by §3: **{winner}**.\n\nNoise floor, (b) against (b) run again: {noise}. The server's stages are from its histogram's buckets, and decide nothing.\n\nMerge builds on idle tenants, which the flush-wait mark is to be read against (§9.5), deciding nothing: {}.\n\nSlots, deciding nothing: {}.\n",
        c.said(),
        d.said(),
        idle_builds.join("; "),
        slots.join("; "),
    ));
    std::fs::write(out.join("summary.md"), &summary).unwrap();
    let verdict = json!({ "arms": verdicts, "c_beats_b": c.said(), "d_beats_b": d.said(),
                          "winner": winner, "noise_floor": noise });
    std::fs::write(
        out.join("verdict.json"),
        serde_json::to_string_pretty(&verdict).unwrap(),
    )
    .unwrap();
    println!("{summary}");
}
