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
use vellev_storage::timings::{Event, Holder, MergePhase, Timed};
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
        ("vellev-tune-wa", "tuning_wait"),
        ("vellev-tune", "tuning"),
        ("vellev-pass", "passes"),
        ("vellev-build", "global"),
        ("tokio-runtime-w", "tokio"),
        ("jemalloc_bg_thd", "jemalloc"),
    ] {
        if name.starts_with(prefix) {
            return pool;
        }
    }
    "other"
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
/// at its own, a thread waiting for tuning as whatever made it, and every
/// other at the process's.
fn nice_wrong(pool: MergePool) -> Vec<String> {
    let (threads, nice) = match pool {
        MergePool::Global => (0, 0),
        MergePool::Own { threads, nice } => (threads, i64::from(nice)),
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
            "tuning_wait" => continue,
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
    nice_checks: u64,
    nice_wrong: Vec<String>,
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

/// What the writers wrote, all told and by tenant.
#[derive(Default)]
struct WritesDone {
    rows: u64,
    batches: u64,
    refused: u64,
    failed: u64,
    behind_most: f64,
    /// Rows written to each tenant: every batch the schedule gives it,
    /// unless the arm could not take them all.
    by_tenant: BTreeMap<String, u64>,
}

/// The most of the machine's CPU its host may take for others (steal)
/// during an arm's soak for its marks to count: past it, the arm ran on
/// less than the machine it is compared on (review of #169).
const STEAL_AT_MOST: f64 = 0.05;

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
    load_reported: bool,
}

impl NodeValidity {
    /// Why the run's marks don't count, if they don't.
    fn invalid(&self) -> Vec<String> {
        let steal = match self.steal {
            None => Some("steal not read".to_string()),
            Some(steal) if steal > STEAL_AT_MOST => Some(format!(
                "steal took {:.1}% of the CPU, over {:.0}%",
                steal * 100.0,
                STEAL_AT_MOST * 100.0
            )),
            Some(_) => None,
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
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// An arm whose host stole over 5% of the machine's CPU through its soak
/// is invalid; one under it, or at it, is not (review of #169).
#[test]
fn an_arm_its_host_stole_from_is_invalid() {
    let run = |steal: Option<f64>| NodeValidity {
        let_go_flushes: 7,
        clustered_merges: 2,
        nice_checks: 4,
        nice_wrong: 0,
        as_the_image: true,
        measurement: true,
        steal,
        load_reported: true,
    };
    assert_eq!(run(Some(0.0)).invalid(), Vec::<String>::new());
    assert_eq!(run(Some(0.05)).invalid(), Vec::<String>::new());
    let stolen = run(Some(0.08)).invalid();
    assert_eq!(stolen, vec!["steal took 8.0% of the CPU, over 5%"]);
    assert_eq!(run(None).invalid(), vec!["steal not read"]);
}

/// What the load generator's run must have done for the arm's marks to
/// count.
struct LoadValidity {
    as_the_image: bool,
    rows_written_a_second: f64,
    rows_a_second: usize,
    failed_reads: u64,
}

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
        failed_reads: 0,
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
        failed_reads,
    };
    assert_eq!(run(0).invalid(), Vec::<String>::new());
    assert_eq!(run(1).invalid(), vec!["1 reads failed"]);
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
        .build()
        .unwrap();
    runtime.block_on(arm(bench, running));
}

async fn arm(bench: Bench, running: Running) {
    std::fs::create_dir_all(&bench.out).unwrap();
    let _ = std::fs::remove_file(ready_path(&bench));
    let _ = std::fs::remove_file(load_path(&bench));
    // Refused before anything is preloaded, if the schedule is not whole.
    Schedule::of(&bench);
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
    let cpu_before = cpu_by_pool();
    let ticks_before = cpu_ticks();
    let backlogged_before = node.flushing().backlogged;
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
                    let held = GLOBAL.held();
                    let census = node.census().holding;
                    let merges = node.merges();
                    let let_go = node.let_go();
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

    // The soak runs its length; what it is held to is taken as it ends.
    tokio::time::sleep_until(soak_end.into()).await;
    soaking.store(false, Ordering::SeqCst);
    let soaked = started.elapsed();
    let cpu_after = cpu_by_pool();
    let ticks_after = cpu_ticks();
    let largest_at_soak_end = node.merges().largest;
    let backlogged = node.flushing().backlogged - backlogged_before;
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
    let metrics_text = app.metrics().encoded();
    std::fs::write(
        bench.out.join(format!("metrics-{}.txt", bench.arm.name())),
        &metrics_text,
    )
    .unwrap();
    let merges = node.merges();
    let let_go = node.let_go();
    let ended: BTreeMap<&str, u64> = vellev_storage::timings::MergeEnd::ALL
        .iter()
        .map(|end| (end.name(), node.timings().merges_ended(*end)))
        .collect();
    node.shutdown().await;
    let _ = stop_serving.send(());
    let _ = serving.await;

    // The report.
    let events: Vec<Event> = std::mem::take(&mut *events.lock().unwrap())
        .into_iter()
        .filter(|event| event.at >= started && event.at <= soak_end)
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
    std::fs::write(
        bench.out.join(format!("events-{}.json", bench.arm.name())),
        serde_json::to_string(&raw).unwrap(),
    )
    .unwrap();
    let of = |timed: Timed| -> Vec<f64> {
        events
            .iter()
            .filter(|event| event.timed == timed)
            .map(|event| event.took.as_secs_f64())
            .collect()
    };
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
    let sampled = std::mem::take(&mut *sampled.lock().unwrap());
    let last_from = soaked.as_secs_f64() - 20.0 * 60.0;
    let mut debt = serde_json::Map::new();
    let mut slopes = Vec::new();
    let mut ends = Vec::new();
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
        let tail: Vec<(f64, f64)> = series
            .iter()
            .filter(|(t, _)| *t >= last_from)
            .map(|(t, segments)| (t / 60.0, *segments))
            .collect();
        let tenant_slope = slope(&tail);
        slopes.push(tenant_slope.unwrap_or(f64::NAN));
        ends.push(last);
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
    let cpu: BTreeMap<String, f64> = cpu_after
        .iter()
        .map(|(pool, after)| {
            (
                pool.clone(),
                after - cpu_before.get(pool).copied().unwrap_or(0.0),
            )
        })
        .collect();
    let steal = match (ticks_before, ticks_after) {
        (Some((stolen_before, all_before)), Some((stolen_after, all_after)))
            if all_after > all_before =>
        {
            Some((stolen_after - stolen_before) as f64 / (all_after - all_before) as f64)
        }
        _ => None,
    };

    // The marks this arm can be held to alone (§9.6); backlog refusals are
    // held to arm (a)'s by the comparison.
    let flush_wait = sorted(of(Timed::BuildingWait(Holder::Flush)));
    let flush_held = sorted(of(Timed::BuildingHeld(Holder::Flush)));
    let merge_builds = sorted(of(Timed::Merge(MergePhase::Build)));
    let let_go_wait = sorted(of(Timed::LetGoFlushWait));
    let flush_wait_p99 = percentile(&flush_wait, 99.0);
    let flush_held_p99 = percentile(&flush_held, 99.0);
    let merge_build_p50 = percentile(&merge_builds, 50.0);
    let peak = sampled.allocated_peak.saturating_sub(baseline_allocated);
    // The largest merge build as the soak ended: what the soak's peak can
    // have come from, not what settling built after.
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
        "backlog_refusals": { "refused": backlogged, "mark": "no more than arm (a)'s" },
    });
    let invalid = NodeValidity {
        let_go_flushes: let_go_wait.len(),
        clustered_merges: sampled.clustered_merges_seen,
        nice_checks: sampled.nice_checks,
        nice_wrong: sampled.nice_wrong.len(),
        as_the_image: running.as_the_image(),
        measurement: bench.is_measurement(),
        steal,
        load_reported,
    }
    .invalid();

    let report = json!({
        "arm": bench.arm.name(),
        "merge_pool": format!("{pool:?}"),
        "running": running,
        "process": process(&bench),
        "sha": bench.sha,
        "bench": bench.config(),
        "run": {
            "soak_s": soaked.as_secs_f64(), "settled_in_s": settled_in, "preload_s": preload_secs,
        },
        "preloaded": preloaded,
        "invalid": invalid,
        "marks": marks,
        "server_stages": {
            "queue": stage_summary(&metrics_text, "queue"),
            "read": stage_summary(&metrics_text, "read"),
        },
        "merge_debt": debt,
        "end_segments_mean": ends.iter().sum::<f64>() / ends.len() as f64,
        "timings": timings,
        "merges_ended": ended,
        "clustered_merges_seen": sampled.clustered_merges_seen,
        "let_go": { "idle": let_go.idle, "budget": let_go.for_budget },
        "memory": {
            "baseline_allocated": baseline_allocated, "allocated_peak": sampled.allocated_peak,
            "baseline_rss_anon": baseline_rss_anon, "rss_anon_peak": sampled.rss_anon_peak,
            "cgroup_peak": cgroup_peak(), "draining_peak": sampled.draining_peak,
            "preloaded_resident": preloaded_resident,
            "largest_merge_build_at_soak_end": largest_at_soak_end,
            "largest_merge_build_settled": merges.largest,
        },
        "memtable": {
            "unflushed_rows_peak": sampled.unflushed_rows_peak,
            "unflushed_bytes_peak": sampled.unflushed_bytes_peak,
            "tenant_unflushed_peak": sampled.tenant_unflushed_peak,
        },
        "cpu_s_by_pool": cpu,
        "steal": { "share": steal, "at_most": STEAL_AT_MOST },
        "memory_series": sampled.memory,
        "nice": { "checks": sampled.nice_checks, "wrong": sampled.nice_wrong },
    });
    let path = bench.out.join(format!("arm-{}.json", bench.arm.name()));
    written_whole(&path, &serde_json::to_string_pretty(&report).unwrap());
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report["marks"]).unwrap()
    );
    eprintln!("invalid: {invalid:?}; written to {}", path.display());
    let _ = std::fs::remove_dir_all(&data);
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
    let ready = appeared(&ready_path(&bench), bench.ready_within)
        .await
        .unwrap_or_else(|| panic!("the node was not ready within {:?}", bench.ready_within));
    let ready: Ready = serde_json::from_str(&ready).unwrap();
    assert_eq!(
        (ready.soak_s, ready.sha.as_str()),
        (bench.soak.as_secs(), bench.sha.as_str()),
        "the node and the load generator were not set alike"
    );
    let queries: Arc<Vec<Vec<f32>>> = Arc::new(
        Fixture::at(&bench.fixture)
            .queries("queries")
            .expect("the fixture's queries")
            .into_iter()
            .map(|query| query.vector)
            .collect(),
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
    // wait it was told, while the soak lasts; a batch not written by its
    // end leaves its tenant short, and the arm is not compared.
    let writes = Arc::new(Mutex::new(WritesDone::default()));
    let mut writers = Vec::new();
    for place in 0..bench.active {
        let passages = Arc::new(Mutex::new(Passages::of(&bench.fixture)));
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
                        let behind_most = &mut writes.lock().unwrap().behind_most;
                        *behind_most = behind_most.max(behind);
                    }
                    if Instant::now() >= soak_end {
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
                    loop {
                        match http_json(&http, route.clone(), &body).await {
                            Ok(answer) if answer.status().is_success() => {
                                let mut done = writes.lock().unwrap();
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
                                if Instant::now() + wait >= soak_end {
                                    break;
                                }
                                tokio::time::sleep(wait).await;
                            }
                            answer => {
                                writes.lock().unwrap().failed += 1;
                                eprintln!("a write failed: {:?}", answer.map(|a| a.status()));
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
    let mut readers = Vec::new();
    for (reader, done) in reads.iter().enumerate() {
        readers.push(tokio::spawn({
            let (http, url, done, queries) =
                (http.clone(), url.clone(), done.clone(), queries.clone());
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
                    let answered = match http_json(&http, route, &body).await {
                        Ok(answer) => {
                            let status = answer.status().as_u16();
                            let read = answer.bytes().await.is_ok();
                            match status {
                                200..300 if read => Answered::Yes,
                                429 | 503 => Answered::Refused,
                                _ => Answered::Failed,
                            }
                        }
                        Err(_) => Answered::Failed,
                    };
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
    for reader in readers {
        let _ = tokio::time::timeout(Duration::from_secs(150), reader).await;
    }
    let soaked = bench.soak.as_secs_f64();
    let reads: Vec<Read> = reads
        .iter()
        .flat_map(|done| std::mem::take(&mut *done.lock().unwrap()))
        .collect();
    let failed_reads = reads
        .iter()
        .filter(|read| read.answered == Answered::Failed)
        .count() as u64;
    let writes = std::mem::take(&mut *writes.lock().unwrap());
    let invalid = LoadValidity {
        as_the_image: running.as_the_image(),
        rows_written_a_second: writes.rows as f64 / soaked,
        rows_a_second: bench.rows_a_second,
        failed_reads,
    }
    .invalid();
    let mut by_tenant = schedule.rows_by_tenant();
    by_tenant.values_mut().for_each(|rows| *rows = 0);
    by_tenant.extend(writes.by_tenant.clone());
    let report = json!({
        "arm": bench.arm.name(),
        "running": running,
        "process": process(&bench),
        "sha": bench.sha,
        "bench": bench.config(),
        "reads": reads_summary(&reads, soaked, bench.window.as_secs_f64()),
        "writes": {
            "rows": writes.rows, "rows_a_second": writes.rows as f64 / soaked,
            "batches": writes.batches, "refused": writes.refused, "failed": writes.failed,
            "behind_most_s": writes.behind_most,
            "by_tenant": by_tenant, "scheduled_by_tenant": schedule.rows_by_tenant(),
        },
        "invalid": invalid,
    });
    written_whole(
        &load_path(&bench),
        &serde_json::to_string_pretty(&report).unwrap(),
    );
    eprintln!("load invalid: {invalid:?}");
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
                      "preloaded": preloaded }),
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

/// Whether an arm met every mark: `None` if that rests on the backlog
/// bound and there is none, as when arm (a) is not compared.
fn all_marks(marks: &Value, backlog_within_a: Option<bool>) -> Option<bool> {
    let passed = |mark: &str| marks[mark]["passed"].as_bool().unwrap_or(false);
    let alone = passed("merge_debt")
        && passed("flush_wait")
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

/// Whether `arm` beats (b): read p99 at least 15% better and end-of-soak
/// segments a tenant within 10% of (b)'s, with every mark met. (b) need
/// only be valid, not meet every mark: §3 holds the arm, not (b), to them.
/// `None`, undecided, if either is missing or not compared, or the arm's
/// marks rest on a backlog bound there is none of (review of #169).
fn beats_b(arm: Option<Compared>, b: Option<Compared>) -> Option<bool> {
    let (arm, b) = (arm?, b?);
    if !arm.valid || !b.valid {
        return None;
    }
    let better =
        arm.read_p99_ms <= 0.85 * b.read_p99_ms && arm.end_segments <= 1.10 * b.end_segments;
    match arm.all_marks? {
        false => Some(false),
        true => Some(better),
    }
}

/// The winner by §3: an arm that beats (b), the better of two that both
/// do; (b) if neither does; undecided if either is undecided.
fn winner(c: Option<bool>, d: Option<bool>, p99_c: f64, p99_d: f64) -> &'static str {
    match (c, d) {
        (Some(true), Some(true)) => {
            if p99_c <= p99_d {
                "c"
            } else {
                "d"
            }
        }
        (Some(true), Some(false)) => "c",
        (Some(false), Some(true)) => "d",
        (Some(false), Some(false)) => "b",
        _ => "undecided",
    }
}

/// "Beats (b)" asks (b) to be valid, not to meet every mark; and what
/// cannot be decided is said to be undecided, never (b) by default
/// (review of #169).
#[test]
fn beats_b_asks_b_be_valid_not_that_it_pass_and_says_undecided() {
    let run = |valid, all_marks, read_p99_ms| Compared {
        valid,
        all_marks,
        read_p99_ms,
        end_segments: 10.0,
    };
    // (b) valid but failing a mark: (c), 20% better and meeting every
    // mark, beats it.
    let b = run(true, Some(false), 100.0);
    assert_eq!(
        beats_b(Some(run(true, Some(true), 80.0)), Some(b)),
        Some(true)
    );
    assert_eq!(
        beats_b(Some(run(true, Some(true), 90.0)), Some(b)),
        Some(false)
    );
    assert_eq!(
        beats_b(Some(run(true, Some(false), 50.0)), Some(b)),
        Some(false)
    );
    // Undecided: (b) or the arm not compared, missing, or resting on no
    // backlog bound.
    let fast = run(true, Some(true), 50.0);
    assert_eq!(
        beats_b(Some(fast), Some(run(false, Some(true), 100.0))),
        None
    );
    assert_eq!(beats_b(Some(run(false, Some(true), 50.0)), Some(b)), None);
    assert_eq!(beats_b(None, Some(b)), None);
    assert_eq!(beats_b(Some(fast), None), None);
    assert_eq!(beats_b(Some(run(true, None, 50.0)), Some(b)), None);
    // The winner.
    assert_eq!(winner(Some(true), Some(true), 70.0, 80.0), "c");
    assert_eq!(winner(Some(false), Some(true), 70.0, 80.0), "d");
    assert_eq!(winner(Some(false), Some(false), 70.0, 80.0), "b");
    assert_eq!(winner(Some(true), None, 70.0, 80.0), "undecided");
    assert_eq!(winner(None, None, 70.0, 80.0), "undecided");
    // No backlog bound, as when (a) is not compared: an arm meeting every
    // other mark is undecided, one failing another is not.
    let marks = json!({ "merge_debt": { "passed": true }, "flush_wait": { "passed": true },
                        "node_memory": { "passed": true }, "eviction_flush": { "passed": true } });
    assert_eq!(all_marks(&marks, None), None);
    assert_eq!(all_marks(&marks, Some(true)), Some(true));
    let mut failing = marks.clone();
    failing["flush_wait"]["passed"] = json!(false);
    assert_eq!(all_marks(&failing, None), Some(false));
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
    let valid = |name: &str| invalid.get(name).is_some_and(Vec::is_empty);
    let run_of = |name: &str| runs.iter().find(|run| run.name == name);
    let p99 = |run: &ArmRun| {
        run.load
            .as_ref()
            .and_then(|load| load["reads"]["p99_ms"].as_f64())
            .unwrap_or(f64::NAN)
    };
    // Backlog refusals are held to arm (a)'s only if (a) is compared.
    let control_refused = run_of("a")
        .filter(|_| valid("a"))
        .and_then(|a| a.node["marks"]["backlog_refusals"]["refused"].as_u64());
    let mut rows = Vec::new();
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
        compared.insert(
            run.name.clone(),
            Compared {
                valid: why.is_empty(),
                all_marks: all,
                read_p99_ms: p99(run),
                end_segments: run.node["end_segments_mean"].as_f64().unwrap_or(f64::NAN),
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
            "| {} | {} | {:.1} | {} | {} | {:.2} | {} | {} | {} | {} | {} | {} | {} |",
            run.name,
            run.node["merge_pool"].as_str().unwrap_or("?"),
            p99(run),
            stage("queue"),
            stage("read"),
            run.node["end_segments_mean"].as_f64().unwrap_or(f64::NAN),
            mark(passed("merge_debt")),
            mark(passed("flush_wait")),
            backlog.map_or("undecided", mark),
            mark(passed("node_memory")),
            mark(passed("eviction_flush")),
            all.map_or("undecided", mark),
            if why.is_empty() {
                "valid".to_string()
            } else {
                why.join("; ")
            },
        ));
        verdicts.insert(
            run.name.clone(),
            json!({ "all_marks": all, "backlog_within_a": backlog, "invalid": why }),
        );
    }
    let (c, d) = (
        beats_b(compared.get("c").copied(), compared.get("b").copied()),
        beats_b(compared.get("d").copied(), compared.get("b").copied()),
    );
    let p99_of = |name: &str| compared.get(name).map_or(f64::NAN, |arm| arm.read_p99_ms);
    let winner = winner(c, d, p99_of("c"), p99_of("d"));
    // The noise floor: (b) against itself, deciding nothing.
    let noise = match (
        valid("b") && valid("b2"),
        compared.get("b"),
        compared.get("b2"),
    ) {
        (true, Some(b), Some(b2)) => json!({
            "read_p99_spread": (b.read_p99_ms - b2.read_p99_ms).abs() / b.read_p99_ms,
            "end_segments_spread": (b.end_segments - b2.end_segments).abs() / b.end_segments,
        }),
        _ => json!(null),
    };
    let said = |beats: Option<bool>| match beats {
        Some(true) => "yes",
        Some(false) => "no",
        None => "undecided",
    };
    let mut summary = String::from(
        "| arm | merge pool | read p99 ms | server queue p99 s | server read p99 s | end segments/tenant | merge debt | flush wait | backlog <= (a) | node memory | eviction flush | all marks | run |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    summary.push_str(&rows.join("\n"));
    summary.push_str(&format!(
        "\n\n(c) beats (b): {}. (d) beats (b): {}. Winner by §3: **{winner}**.\n\nNoise floor, (b) against (b) run again: {noise}. The server's stages are from its histogram's buckets, and decide nothing.\n",
        said(c),
        said(d),
    ));
    std::fs::write(out.join("summary.md"), &summary).unwrap();
    let verdict = json!({ "arms": verdicts, "c_beats_b": c, "d_beats_b": d, "winner": winner,
                          "noise_floor": noise });
    std::fs::write(
        out.join("verdict.json"),
        serde_json::to_string_pretty(&verdict).unwrap(),
    )
    .unwrap();
    println!("{summary}");
}
