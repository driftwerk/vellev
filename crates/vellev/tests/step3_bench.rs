//! The step 3 bench (merge note, §3 and §9): whether merges built on a
//! pool of their own, smaller or lower in priority, are worth it once the
//! building lock no longer holds flushes behind them.
//!
//! One arm a process, as `scripts/step3-bench/run.sh` runs them, each the
//! same: a whole node in this process as the image runs one (its
//! allocator, its HTTP API, an object store behind a cache directory), six
//! tenants preloaded the same way from the bench fixture, then a soak of
//! half an hour or more with readers at twice the cores back to back and
//! a thousand rows a second written, the node held to a memory budget it
//! is always over. Every mark is worked out from exact durations, never
//! from a histogram's buckets. `step3_compare` then sets the arms side by
//! side, against the marks fixed before the run.
//!
//! Ignored: it takes the fixture and the better part of an hour. Run as
//! `scripts/step3-bench/run.sh` says.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
/// What a clustered merge's output is over (`CLUSTERED_ABOVE`).
const CLUSTERED_ABOVE: u64 = 200_000;
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
}

impl Arm {
    fn named(name: &str) -> Arm {
        match name {
            "a" => Arm::A,
            "b" => Arm::B,
            "c" => Arm::C,
            "d" => Arm::D,
            other => panic!("STEP3_ARM is a, b, c or d, not {other:?}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Arm::A => "a",
            Arm::B => "b",
            Arm::C => "c",
            Arm::D => "d",
        }
    }

    fn pool(self, threads: usize) -> MergePool {
        match self {
            Arm::A => MergePool::Global,
            Arm::B => MergePool::Own {
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
        }
    }

    /// Whether this is the bench as §3 runs it, not a smoke of it.
    fn is_measurement(&self) -> bool {
        self.soak >= Duration::from_secs(1_800)
            && self.scale == 1.0
            && self.rows_a_second == 1_000
            && self.readers == 2 * self.threads
    }
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

/// What a tenant held once preloaded: the same in every arm.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Preloaded {
    tenant: String,
    rows: u64,
    clustered: usize,
    clustered_rows: u64,
    plain: usize,
    plain_rows: u64,
}

/// Preload every tenant as §9.5 says, through a node that does nothing of
/// its own accord in the time it takes: every flush and merge asked for
/// here, none by timing. Returns what each tenant came to, what they are
/// reckoned to hold together, and where the fixture's rows go on from.
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
    for (at, preload) in PRELOADS.iter().enumerate() {
        let id = tenant_id(at);
        preloading.create_tenant(&id).await.unwrap();
        let tenant = preloading.tenant(&id).await.unwrap();
        tenant
            .write(vec![Mutation::CreateCollection(Box::new(schema()))])
            .await
            .unwrap();
        let scaled = |rows: u64| ((rows as f64 * bench.scale) as u64).max(1);
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

/// The cgroup's peak memory, where the process has a cgroup of its own
/// that keeps it.
fn cgroup_peak() -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    let peak = std::fs::read_to_string(format!("/sys/fs/cgroup{path}/memory.peak")).ok()?;
    peak.trim().parse().ok()
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

/// What the soak samples as it goes.
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

/// One read as a client timed it: when it began, in seconds into the
/// soak, and how long it took, in milliseconds.
type Timing = (f32, f32);

#[derive(Default)]
struct ReadsDone {
    timings: Vec<Timing>,
    refused: u64,
    failed: u64,
}

#[derive(Default)]
struct WritesDone {
    rows: u64,
    batches: u64,
    refused: u64,
    failed: u64,
    behind_most: f64,
}

/// What an arm's run must have done for its marks to count (§9.6).
struct Validity {
    let_go_flushes: usize,
    clustered_merges: u64,
    nice_checks: u64,
    nice_wrong: usize,
    as_the_image: bool,
    measurement: bool,
    rows_written_a_second: f64,
    rows_a_second: usize,
}

/// The least share of its rate an arm must write for its marks to count:
/// one that wrote less was spared the load §3 sets.
const WRITTEN_AT_LEAST: f64 = 0.95;

impl Validity {
    /// Why the run's marks don't count, if they don't.
    fn invalid(&self) -> Vec<String> {
        let wrote_too_little =
            self.rows_written_a_second < WRITTEN_AT_LEAST * self.rows_a_second as f64;
        [
            (
                self.let_go_flushes == 0,
                "no flush before letting a tenant go".to_string(),
            ),
            (
                self.clustered_merges == 0,
                "no clustered merge (over 200k rows) committed".to_string(),
            ),
            (
                self.nice_wrong > 0 || self.nice_checks == 0,
                "the priorities were not as the arm says, or were never checked".to_string(),
            ),
            (!self.as_the_image, "not run as the image runs".to_string()),
            (
                !self.measurement,
                "a smoke, not the measurement".to_string(),
            ),
            (
                wrote_too_little,
                format!(
                    "wrote {:.0} rows a second, under {:.0}% of {}",
                    self.rows_written_a_second,
                    WRITTEN_AT_LEAST * 100.0,
                    self.rows_a_second
                ),
            ),
        ]
        .into_iter()
        .filter(|(wrong, _)| *wrong)
        .map(|(_, why)| why)
        .collect()
    }
}

/// An arm that wrote under 95% of its rate is invalid, its marks not
/// counted: the CI runner's first full-size run wrote 416 rows a second of
/// 1,000 (scratch run 41054724534), a load §3 does not set. One that wrote
/// its rate, or just under, is not invalid for it.
#[test]
fn an_arm_that_wrote_under_its_rate_is_invalid() {
    let run = |written: f64| Validity {
        let_go_flushes: 7,
        clustered_merges: 2,
        nice_checks: 4,
        nice_wrong: 0,
        as_the_image: true,
        measurement: true,
        rows_written_a_second: written,
        rows_a_second: 1_000,
    };
    assert_eq!(run(1_000.0).invalid(), Vec::<String>::new());
    assert_eq!(run(951.0).invalid(), Vec::<String>::new());
    let short = run(416.0).invalid();
    assert_eq!(short.len(), 1, "{short:?}");
    assert!(short[0].starts_with("wrote 416 rows a second"), "{short:?}");
    assert_eq!(run(949.0).invalid().len(), 1);
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

async fn http_json(
    http: &reqwest::Client,
    url: String,
    body: &Value,
) -> Result<reqwest::Response, reqwest::Error> {
    http.post(url).bearer_auth(KEY).json(body).send().await
}

/// One arm (§9): preload, soak, report. Run by `scripts/step3-bench/run.sh`.
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
    let data = bench.data.join(format!("arm-{}", bench.arm.name()));
    let _ = std::fs::remove_dir_all(&data);
    std::fs::create_dir_all(&data).unwrap();
    let coordinator = Arc::new(MemoryCoordinator::default());
    let blobs = store(&data).await;
    let mut passages = Passages::of(&bench.fixture);
    let preloading = Instant::now();
    let (preloaded, preloaded_resident, rows_of) =
        preload(&bench, &coordinator, &blobs, &mut passages).await;
    let preload_secs = preloading.elapsed().as_secs_f64();
    eprintln!("preloaded in {preload_secs:.0} s: {preloaded:?}");
    let budget = (preloaded_resident as f64 * bench.budget_share) as usize;
    let queries: Vec<Vec<f32>> = Fixture::at(&bench.fixture)
        .queries("queries")
        .expect("the fixture's queries")
        .into_iter()
        .map(|query| query.vector)
        .collect();
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

    // The soak.
    let started = Instant::now();
    let soaking = Arc::new(AtomicBool::new(true));
    let active = Arc::new(Mutex::new((0..bench.active).collect::<Vec<usize>>()));
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .pool_max_idle_per_host(bench.readers + 4)
        .build()
        .unwrap();
    let cpu_before = cpu_by_pool();
    let backlogged_before = node.flushing().backlogged;
    let events: Arc<Mutex<Vec<Event>>> = Arc::default();
    let sampled = Arc::new(Mutex::new(Sampled::default()));
    let mut tasks = Vec::new();

    // The active set, moved on one tenant each turn.
    tasks.push(tokio::spawn({
        let (active, soaking, bench) = (active.clone(), soaking.clone(), bench.clone());
        async move {
            let mut first = 0;
            while soaking.load(Ordering::SeqCst) {
                tokio::time::sleep(bench.rotate).await;
                first = (first + 1) % PRELOADS.len();
                let now = (0..bench.active)
                    .map(|k| (first + k) % PRELOADS.len())
                    .collect();
                *active.lock().unwrap() = now;
            }
        }
    }));

    // The writers, one for each place in the active set, each writing its
    // share of the rate to whichever tenant holds that place, on a fixed
    // schedule, caught up on once it falls behind; a batch refused is
    // tried again after the wait it was told. One a place, so that one
    // tenant slow to take a write (reopened, or at its backlog) holds up
    // only its own share, as one client among many would.
    let writes = Arc::new(Mutex::new(WritesDone::default()));
    let passages = Arc::new(Mutex::new(passages));
    let rows_of = Arc::new(Mutex::new(rows_of));
    let mut writers = Vec::new();
    for place in 0..bench.active {
        writers.push(tokio::spawn({
            let (active, soaking, bench, http) =
                (active.clone(), soaking.clone(), bench.clone(), http.clone());
            let (url, writes) = (url.clone(), writes.clone());
            let (passages, rows_of) = (passages.clone(), rows_of.clone());
            async move {
                let mut random = Random(bench.seed ^ (0xa11c_e000 + place as u64));
                let rate = bench.rows_a_second as f64 / bench.active as f64;
                let every = Duration::from_secs_f64(bench.batch as f64 / rate);
                let mut k: u32 = 0;
                while soaking.load(Ordering::SeqCst) {
                    let due = started + every * k;
                    let now = Instant::now();
                    if due > now {
                        tokio::time::sleep(due - now).await;
                    } else {
                        let behind = (now - due).as_secs_f64();
                        let behind_most = &mut writes.lock().unwrap().behind_most;
                        *behind_most = behind_most.max(behind);
                    }
                    let at = active.lock().unwrap()[place];
                    let (batch, size) = (bench.batch, bench.batch as u64);
                    let picks: Vec<(bool, u64)> = (0..batch)
                        .map(|_| (random.below(5) == 0, random.next()))
                        .collect();
                    // Read from the fixture off the async threads: a shard
                    // is read whole as it is first wanted.
                    let (passages, rows_of) = (passages.clone(), rows_of.clone());
                    let changes = tokio::task::spawn_blocking(move || {
                        let mut passages = passages.lock().unwrap();
                        let mut rows_of = rows_of.lock().unwrap();
                        picks
                            .into_iter()
                            .map(|(replaces, pick)| {
                                let row = passages.next_row();
                                // One in five replaces a row the tenant holds.
                                let n = match replaces {
                                    true => pick % rows_of[at].max(1),
                                    false => {
                                        rows_of[at] += 1;
                                        rows_of[at] - 1
                                    }
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
                        // None once written or given up; else the wait it
                        // was told before it is tried again.
                        let refused = match http_json(&http, route.clone(), &body).await {
                            Ok(answer) if answer.status().is_success() => {
                                let mut done = writes.lock().unwrap();
                                done.rows += size;
                                done.batches += 1;
                                None
                            }
                            Ok(answer) if answer.status().as_u16() == 429 => {
                                writes.lock().unwrap().refused += 1;
                                let wait = answer
                                    .headers()
                                    .get("retry-after")
                                    .and_then(|wait| wait.to_str().ok()?.parse::<u64>().ok())
                                    .unwrap_or(1);
                                Some(Duration::from_secs(wait))
                            }
                            Ok(answer) => {
                                writes.lock().unwrap().failed += 1;
                                eprintln!("a write failed: {}", answer.status());
                                None
                            }
                            Err(error) => {
                                writes.lock().unwrap().failed += 1;
                                eprintln!("a write failed: {error}");
                                None
                            }
                        };
                        let Some(wait) = refused else { break };
                        tokio::time::sleep(wait).await;
                        if !soaking.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                    k += 1;
                }
            }
        }));
    }

    // The readers, back to back.
    let reads: Vec<Arc<Mutex<ReadsDone>>> = (0..bench.readers).map(|_| Arc::default()).collect();
    let queries = Arc::new(queries);
    for (reader, done) in reads.iter().enumerate() {
        tasks.push(tokio::spawn({
            let (active, soaking, http, url) =
                (active.clone(), soaking.clone(), http.clone(), url.clone());
            let (done, queries) = (done.clone(), queries.clone());
            let mut random = Random(bench.seed ^ (reader as u64 + 1).wrapping_mul(0x9e37_79b9));
            async move {
                while soaking.load(Ordering::SeqCst) {
                    let at = {
                        let active = active.lock().unwrap();
                        active[random.below(active.len() as u64) as usize]
                    };
                    let query = &queries[random.below(queries.len() as u64) as usize];
                    let body = json!({ "vector": query, "limit": 10 });
                    let route = format!(
                        "{url}/v1/tenants/{}/collections/{COLLECTION}/nearest",
                        tenant_id(at)
                    );
                    let began = Instant::now();
                    let answer = http_json(&http, route, &body).await;
                    let answered = match answer {
                        Ok(answer) => {
                            let status = answer.status();
                            let read = answer.bytes().await.is_ok();
                            (status.is_success() && read)
                                .then_some(())
                                .ok_or(status.as_u16())
                        }
                        Err(_) => Err(0),
                    };
                    let took = began.elapsed();
                    let mut done = done.lock().unwrap();
                    match answered {
                        Ok(()) => done.timings.push((
                            (began - started).as_secs_f32(),
                            took.as_secs_f32() * 1_000.0,
                        )),
                        Err(429) => done.refused += 1,
                        Err(_) => done.failed += 1,
                    }
                }
            }
        }));
    }

    // The samplers.
    tasks.push(tokio::spawn({
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
                let (mut writers_resident, mut writers) = (0, 0);
                for (tenant, figures) in node.tenant_figures() {
                    let tenant = tenant.to_string();
                    s.tenant_unflushed_peak = s.tenant_unflushed_peak.max(figures.unflushed);
                    let before = s
                        .clustered_rows
                        .insert(tenant.clone(), figures.clustered_rows);
                    if before.is_some_and(|rows| rows != figures.clustered_rows)
                        && figures.clustered_rows > CLUSTERED_ABOVE
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
    }));

    // The soak runs its length, then everything stops asking.
    tokio::time::sleep(bench.soak).await;
    soaking.store(false, Ordering::SeqCst);
    let soaked = started.elapsed();
    let cpu_after = cpu_by_pool();
    for writer in writers {
        writer.await.unwrap();
    }
    for task in tasks {
        let _ = tokio::time::timeout(Duration::from_secs(150), task).await;
    }
    events.lock().unwrap().extend(node.timings().drain_events());
    let backlogged = node.flushing().backlogged - backlogged_before;
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
    let soak_end = started + soaked;
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

    // Reads: whole soak, and each window.
    let mut read_timings: Vec<Timing> = Vec::new();
    let (mut refused, mut failed) = (0, 0);
    for done in &reads {
        let mut done = done.lock().unwrap();
        read_timings.append(&mut done.timings);
        refused += done.refused;
        failed += done.failed;
    }
    let all_reads = sorted(read_timings.iter().map(|(_, ms)| f64::from(*ms)).collect());
    let windows: Vec<Value> = (0..)
        .map(|w| w as f64 * bench.window.as_secs_f64())
        .take_while(|from| *from < soaked.as_secs_f64())
        .map(|from| {
            let to = from + bench.window.as_secs_f64();
            let within = sorted(
                read_timings
                    .iter()
                    .filter(|(at, _)| f64::from(*at) >= from && f64::from(*at) < to)
                    .map(|(_, ms)| f64::from(*ms))
                    .collect(),
            );
            json!({ "from_s": from, "reads": within.len(), "p50_ms": percentile(&within, 50.0),
                    "p99_ms": percentile(&within, 99.0) })
        })
        .collect();

    // Merge debt: each tenant carried forward while let go, from its
    // preload until it is first seen held.
    let sampled = std::mem::take(&mut *sampled.lock().unwrap());
    let last_from = soaked.as_secs_f64() - 20.0 * 60.0;
    let mut debt = serde_json::Map::new();
    let mut slopes = Vec::new();
    let mut ends = Vec::new();
    for (at, preload) in preloaded.iter().enumerate() {
        let seen = sampled
            .debt
            .get(&preload.tenant)
            .cloned()
            .unwrap_or_default();
        let start = (preload.clustered + preload.plain) as f64;
        // Every 10 s, the last value seen at or before it.
        let mut series = Vec::new();
        let mut last = start;
        let mut points = seen.iter().peekable();
        let mut t = 0.0;
        while t <= soaked.as_secs_f64() {
            while let Some(point) = points.next_if(|point| point.0 <= t + 5.0) {
                last = point.1;
            }
            series.push((t, last));
            t += 10.0;
        }
        let tail: Vec<(f64, f64)> = series
            .iter()
            .filter(|(t, _)| *t >= last_from)
            .map(|(t, segments)| (t / 60.0, *segments))
            .collect();
        let tenant_slope = slope(&tail);
        slopes.push(tenant_slope.unwrap_or(f64::NAN));
        ends.push(last);
        debt.insert(
            format!("t{at}"),
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
    let writes = std::mem::take(&mut *writes.lock().unwrap());

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
    let memory_bound = budget as u64 + (MERGE_BUILDS * merges.largest) as u64;
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
            "budget_bytes": budget, "largest_merge_build_bytes": merges.largest,
            "mark": "jemalloc allocated peak over baseline <= budget + MERGE_BUILDS x largest merge build",
        },
        "eviction_flush": {
            "passed": percentile(&let_go_wait, 99.0).is_some_and(|p99| p99 < 2.0),
            "wait_p99_s": percentile(&let_go_wait, 99.0), "flushes": let_go_wait.len(),
            "mark": "p99 wait < 2 s",
        },
        "backlog_refusals": { "refused": backlogged, "mark": "no more than arm (a)'s" },
    });
    let invalid = Validity {
        let_go_flushes: let_go_wait.len(),
        clustered_merges: sampled.clustered_merges_seen,
        nice_checks: sampled.nice_checks,
        nice_wrong: sampled.nice_wrong.len(),
        as_the_image: running.as_the_image(),
        measurement: bench.is_measurement(),
        rows_written_a_second: writes.rows as f64 / soaked.as_secs_f64(),
        rows_a_second: bench.rows_a_second,
    }
    .invalid();

    let report = json!({
        "arm": bench.arm.name(),
        "merge_pool": format!("{pool:?}"),
        "running": running,
        "bench": {
            "soak_s": soaked.as_secs_f64(), "settled_in_s": settled_in, "preload_s": preload_secs,
            "rows_a_second": bench.rows_a_second, "batch": bench.batch, "readers": bench.readers,
            "threads": bench.threads, "flush_rows": bench.flush_rows, "scale": bench.scale,
            "budget_share": bench.budget_share, "active": bench.active,
            "rotate_s": bench.rotate.as_secs(), "seed": bench.seed,
        },
        "preloaded": preloaded,
        "invalid": invalid,
        "marks": marks,
        "reads": {
            "count": all_reads.len(), "refused": refused, "failed": failed,
            "p50_ms": percentile(&all_reads, 50.0), "p99_ms": percentile(&all_reads, 99.0),
            "windows": windows,
        },
        "writes": {
            "rows": writes.rows, "rows_a_second": writes.rows as f64 / soaked.as_secs_f64(),
            "batches": writes.batches, "refused": writes.refused, "failed": writes.failed,
            "behind_most_s": writes.behind_most,
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
            "load_generator_shard_bytes": (10_000 * DIMENSIONS * 4),
        },
        "memtable": {
            "unflushed_rows_peak": sampled.unflushed_rows_peak,
            "unflushed_bytes_peak": sampled.unflushed_bytes_peak,
            "tenant_unflushed_peak": sampled.tenant_unflushed_peak,
        },
        "cpu_s_by_pool": cpu,
        "memory_series": sampled.memory,
        "nice": { "checks": sampled.nice_checks, "wrong": sampled.nice_wrong },
    });
    let path = bench.out.join(format!("arm-{}.json", bench.arm.name()));
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    eprintln!(
        "{}",
        serde_json::to_string_pretty(&report["marks"]).unwrap()
    );
    eprintln!("invalid: {invalid:?}; written to {}", path.display());
    let _ = std::fs::remove_dir_all(&data);
}

/// The arms side by side (§9.6): each mark, arm (a) as the backlog's
/// bound, and which arm wins by "beats (b)". Reads `STEP3_OUT/arm-*.json`
/// and writes `summary.md` and `verdict.json` beside them.
#[test]
#[ignore = "the step 3 bench's comparison, run by scripts/step3-bench/run.sh"]
fn step3_compare() {
    let out = PathBuf::from(std::env::var("STEP3_OUT").expect("STEP3_OUT must be set"));
    let mut arms: BTreeMap<String, Value> = BTreeMap::new();
    for arm in ["a", "b", "c", "d"] {
        let path = out.join(format!("arm-{arm}.json"));
        if let Ok(text) = std::fs::read_to_string(&path) {
            arms.insert(arm.into(), serde_json::from_str(&text).unwrap());
        }
    }
    assert!(!arms.is_empty(), "no arm-*.json in {}", out.display());
    let control_refused = arms
        .get("a")
        .and_then(|a| a["marks"]["backlog_refusals"]["refused"].as_u64());
    let preload_of = |arm: &Value| arm["preloaded"].clone();
    let first_preload = preload_of(arms.values().next().unwrap());
    let mut rows = Vec::new();
    let mut verdicts = serde_json::Map::new();
    let mut passed_all: BTreeMap<String, bool> = BTreeMap::new();
    for (name, arm) in &arms {
        let mut invalid: Vec<String> = arm["invalid"]
            .as_array()
            .unwrap()
            .iter()
            .map(|why| why.as_str().unwrap().to_string())
            .collect();
        if preload_of(arm) != first_preload {
            invalid.push("its preload differs from the other arms'".into());
        }
        let marks = &arm["marks"];
        let refused = marks["backlog_refusals"]["refused"].as_u64().unwrap();
        let backlog = control_refused.map(|control| refused <= control);
        let passed = |mark: &str| marks[mark]["passed"].as_bool().unwrap_or(false);
        let all = passed("merge_debt")
            && passed("flush_wait")
            && passed("node_memory")
            && passed("eviction_flush")
            && backlog.unwrap_or(false);
        passed_all.insert(name.clone(), all && invalid.is_empty());
        let mark = |ok: bool| if ok { "pass" } else { "FAIL" };
        rows.push(format!(
            "| {name} | {} | {:.1} | {:.2} | {} | {} | {} | {} | {} | {} | {} |",
            arm["merge_pool"].as_str().unwrap(),
            arm["reads"]["p99_ms"].as_f64().unwrap_or(f64::NAN),
            arm["end_segments_mean"].as_f64().unwrap_or(f64::NAN),
            mark(passed("merge_debt")),
            mark(passed("flush_wait")),
            backlog.map_or("no (a)", mark),
            mark(passed("node_memory")),
            mark(passed("eviction_flush")),
            mark(all),
            if invalid.is_empty() {
                "valid".to_string()
            } else {
                invalid.join("; ")
            },
        ));
        verdicts.insert(
            name.clone(),
            json!({ "all_marks": all, "backlog_within_a": backlog, "invalid": invalid }),
        );
    }
    // "Beats (b)": read p99 at least 15% better than (b)'s, end-of-soak
    // segments a tenant within 10% of (b)'s, and every mark met.
    let p99 = |arm: &str| arms.get(arm).and_then(|a| a["reads"]["p99_ms"].as_f64());
    let end = |arm: &str| arms.get(arm).and_then(|a| a["end_segments_mean"].as_f64());
    let beats_b = |arm: &str| -> Option<bool> {
        let (p99_arm, p99_b, end_arm, end_b) = (p99(arm)?, p99("b")?, end(arm)?, end("b")?);
        Some(
            p99_arm <= 0.85 * p99_b
                && end_arm <= 1.10 * end_b
                && *passed_all.get(arm)?
                && *passed_all.get("b")?,
        )
    };
    let (c, d) = (beats_b("c"), beats_b("d"));
    let winner = match (c, d) {
        (Some(true), Some(true)) => {
            if p99("c") <= p99("d") {
                "c"
            } else {
                "d"
            }
        }
        (Some(true), _) => "c",
        (_, Some(true)) => "d",
        _ => "b",
    };
    let decided = arms.contains_key("b")
        && passed_all.get("b").copied().unwrap_or(false)
        && c.is_some()
        && d.is_some();
    let mut summary = String::from(
        "| arm | merge pool | read p99 ms | end segments/tenant | merge debt | flush wait | backlog <= (a) | node memory | eviction flush | all marks | run |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    summary.push_str(&rows.join("\n"));
    summary.push_str(&format!(
        "\n\n(c) beats (b): {c:?}. (d) beats (b): {d:?}. Winner by §3: **{winner}**{}.\n",
        if decided {
            ""
        } else {
            " (not decided: (b) failed a mark or was invalid, or an arm is missing)"
        }
    ));
    std::fs::write(out.join("summary.md"), &summary).unwrap();
    let verdict = json!({ "arms": verdicts, "c_beats_b": c, "d_beats_b": d, "winner": winner,
                          "decided": decided });
    std::fs::write(
        out.join("verdict.json"),
        serde_json::to_string_pretty(&verdict).unwrap(),
    )
    .unwrap();
    println!("{summary}");
}
