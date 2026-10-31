//! Scratch: a merge under a cgroup's memory limit, on production's jemalloc
//! and options, with a reader keeping a mapped file hot beside it. Not for
//! merging. Run inside `systemd-run --scope` as root, for `memory.reclaim`.
//!
//! Environment: ROWS, DIMS, ROOT, READER_MB, SEGMENTS.
//!
//! The store keeps logs and manifests on disk and throws away what goes up of
//! a segment, at the `ObjectStore`: `ObjectBlobs` runs whole, as it does over
//! S3, but nothing lands on this disk that a node's would not. Nothing here
//! goes over a network, which is out of reach of this measurement.

use std::alloc::{GlobalAlloc, Layout};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path as StorePath;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, UploadPart,
};
use serde_json::json;
use vellev::allocator::{Allocator, Options, OPTIONS};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{MergeBuilds, Merging, Mutation, Writer};

#[global_allocator]
static ALLOCATOR: Allocator = Allocator::new();

#[used]
#[unsafe(export_name = "_rjem_malloc_conf")]
static MALLOC_CONF: Options = Options(OPTIONS.as_ptr());

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[derive(Debug)]
struct Discarding(LocalFileSystem);

impl std::fmt::Display for Discarding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "discarding")
    }
}

fn gone_up(location: &StorePath) -> bool {
    let name = location.as_ref();
    name.ends_with(".vectors") || name.ends_with(".segment") || name.ends_with(".index")
}

fn done() -> PutResult {
    PutResult { e_tag: None, version: None }
}

#[derive(Debug)]
struct Nowhere;

#[async_trait]
impl MultipartUpload for Nowhere {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        // Held until sent, and read, as a client's request reads it.
        Box::pin(async move {
            let sum: u64 = data.iter().map(|chunk| chunk.iter().map(|b| u64::from(*b)).sum::<u64>()).sum();
            std::hint::black_box(sum);
            Ok(())
        })
    }
    async fn complete(&mut self) -> object_store::Result<PutResult> {
        Ok(done())
    }
    async fn abort(&mut self) -> object_store::Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ObjectStore for Discarding {
    async fn put_opts(&self, location: &StorePath, payload: PutPayload, opts: PutOptions) -> object_store::Result<PutResult> {
        match gone_up(location) {
            true => Ok(done()),
            false => self.0.put_opts(location, payload, opts).await,
        }
    }
    async fn put_multipart_opts(&self, location: &StorePath, opts: PutMultipartOptions) -> object_store::Result<Box<dyn MultipartUpload>> {
        match gone_up(location) {
            true => Ok(Box::new(Nowhere)),
            false => self.0.put_multipart_opts(location, opts).await,
        }
    }
    async fn get_opts(&self, location: &StorePath, options: GetOptions) -> object_store::Result<GetResult> {
        self.0.get_opts(location, options).await
    }
    async fn delete(&self, location: &StorePath) -> object_store::Result<()> {
        self.0.delete(location).await
    }
    fn list(&self, prefix: Option<&StorePath>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.0.list(prefix)
    }
    async fn list_with_delimiter(&self, prefix: Option<&StorePath>) -> object_store::Result<ListResult> {
        self.0.list_with_delimiter(prefix).await
    }
    async fn copy(&self, from: &StorePath, to: &StorePath) -> object_store::Result<()> {
        self.0.copy(from, to).await
    }
    async fn copy_if_not_exists(&self, from: &StorePath, to: &StorePath) -> object_store::Result<()> {
        self.0.copy_if_not_exists(from, to).await
    }
}

struct Random(u64);

impl Random {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn ranked(&mut self) -> usize {
        50_000f32.powf(self.unit()) as usize
    }

    fn passage(&mut self) -> String {
        const STOPWORDS: [&str; 8] = ["the", "of", "and", "in", "to", "a", "that", "it"];
        let word = |at: usize| match at % 3 {
            0 => STOPWORDS[(self.unit() * 8.0) as usize].to_string(),
            _ => format!("pas{}", self.ranked()),
        };
        (0..120).map(word).collect::<Vec<_>>().join(" ")
    }
}

fn cgroup() -> String {
    let own = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let path = own.lines().find_map(|line| line.strip_prefix("0::")).unwrap_or("/");
    format!("/sys/fs/cgroup{path}")
}

fn pairs(text: &str) -> BTreeMap<String, u64> {
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(' ')?;
            Some((key.to_string(), value.trim().parse().ok()?))
        })
        .collect()
}

/// `some` and `full` total stall, in microseconds, from memory.pressure.
fn pressure(group: &str) -> (u64, u64) {
    let text = std::fs::read_to_string(format!("{group}/memory.pressure")).unwrap_or_default();
    let total = |kind: &str| {
        text.lines()
            .find(|line| line.starts_with(kind))
            .and_then(|line| line.rsplit("total=").next()?.trim().parse().ok())
            .unwrap_or(0)
    };
    (total("some"), total("full"))
}

#[derive(Clone, Default)]
struct Sample {
    stat: BTreeMap<String, u64>,
    events: BTreeMap<String, u64>,
    some: u64,
    full: u64,
}

fn sampled(group: &str) -> Sample {
    let read = |name: &str| std::fs::read_to_string(format!("{group}/{name}")).unwrap_or_default();
    let (some, full) = pressure(group);
    Sample { stat: pairs(&read("memory.stat")), events: pairs(&read("memory.events")), some, full }
}

fn cpu() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = stat.rsplit(')').next().unwrap_or("");
    let fields: Vec<&str> = after.split_whitespace().collect();
    let ticks = |at: usize| fields.get(at).and_then(|f| f.parse::<f64>().ok()).unwrap_or(0.0);
    (ticks(11) + ticks(12)) / 100.0
}

/// This thread's major faults.
fn own_majors() -> u64 {
    let stat = std::fs::read_to_string("/proc/thread-self/stat").unwrap_or_default();
    let after = stat.rsplit(')').next().unwrap_or("");
    after.split_whitespace().nth(9).and_then(|f| f.parse().ok()).unwrap_or(0)
}

/// What the reader did in a span: how many pages, how long the worst took,
/// how many took over a millisecond, and its own major faults.
#[derive(Default, Clone, Copy, Debug)]
struct Read {
    pages: u64,
    slow: u64,
    worst_us: u64,
    majors: u64,
    seconds: f64,
}

/// A reader touching the pages of a mapped file at random, as searches read
/// a hot tenant's vectors, a span at a time, each span ended by `spans`.
fn reader(path: std::path::PathBuf, stop: Arc<AtomicBool>, span: Arc<AtomicU64>, spans: Arc<std::sync::Mutex<Vec<Read>>>) {
    let file = std::fs::File::open(path).unwrap();
    let map = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let pages = map.len() / 4096;
    let mut random = 88172645463325252u64;
    let mut now = Read::default();
    let (mut seen, mut majors, mut began) = (span.load(Ordering::SeqCst), own_majors(), Instant::now());
    while !stop.load(Ordering::SeqCst) {
        for _ in 0..64 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let at = Instant::now();
            std::hint::black_box(map[(random as usize % pages) * 4096]);
            let took = at.elapsed().as_micros() as u64;
            now.pages += 1;
            now.worst_us = now.worst_us.max(took);
            now.slow += u64::from(took >= 1_000);
        }
        std::thread::sleep(Duration::from_micros(100));
        let span_now = span.load(Ordering::SeqCst);
        if span_now != seen {
            now.majors = own_majors() - majors;
            now.seconds = began.elapsed().as_secs_f64();
            spans.lock().unwrap().push(now);
            now = Read::default();
            seen = span_now;
            majors = own_majors();
            began = Instant::now();
        }
    }
}

#[tokio::main]
async fn main() {
    assert_eq!(ALLOCATOR.is_global(), Some(true));
    let rows = setting("ROWS", 200_000);
    let dimensions = setting("DIMS", 768);
    let segments = setting("SEGMENTS", 4);
    let reader_mb = setting("READER_MB", 200);
    let root = std::env::var("ROOT").unwrap_or_else(|_| ".".into());
    let root = tempfile::tempdir_in(root).unwrap();
    let (dir, cache) = (root.path().join("store"), root.path().join("cache"));
    std::fs::create_dir_all(&dir).unwrap();
    let group = cgroup();
    let local = LocalFileSystem::new_with_prefix(&dir).unwrap();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(Discarding(local))));
    let blobs: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store, &cache));
    let coordinator = Arc::new(MemoryCoordinator::default());
    let key = PartitionKey { tenant: TenantId::new("t").unwrap(), partition: vellev_core::Partition::DEFAULT };
    coordinator.register(&key).await.unwrap();
    let hold = Duration::from_secs(3_600);
    let Acquired::Lease { lease, committed } = coordinator.acquire(&key, &NodeId::generate("n"), hold).await.unwrap() else {
        panic!("lease")
    };
    let builds = MergeBuilds::new(1);
    let writer: Writer = Writer::open(blobs.clone(), coordinator.clone(), lease, &committed)
        .await
        .unwrap()
        .merging_among(builds.clone());
    let collection = Collection {
        name: CollectionName::new("docs").unwrap(),
        properties: vec![Property::new("title", DataType::Text).searchable(), Property::new("n", DataType::Int)],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "m".into(),
            dimensions: dimensions as u32,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    };
    writer.commit(vec![Mutation::CreateCollection(Box::new(collection))]).await.unwrap();
    let mut random = Random(11);
    for segment in 0..segments {
        let (first, last) = (segment * rows / segments, (segment + 1) * rows / segments);
        for batch in (first..last).step_by(500) {
            let commit = (batch..(batch + 500).min(last))
                .map(|n| Mutation::Upsert {
                    collection: CollectionName::new("docs").unwrap(),
                    id: ObjectId(uuid::Uuid::from_u128(n as u128)),
                    properties: json!({ "title": random.passage(), "n": n }).as_object().unwrap().clone(),
                    vector: Some((0..dimensions).map(|_| random.unit()).collect()),
                    only_if_version: None,
                })
                .collect();
            writer.commit(commit).await.unwrap();
        }
        writer.flush().await.unwrap();
    }
    eprintln!("loaded");
    // The hot file, written and read through by the reader before anything
    // is evicted, as a hot tenant's vectors are.
    let hot = root.path().join("hot");
    let mut file = std::fs::File::create(&hot).unwrap();
    let block: Vec<u8> = (0..1 << 20).map(|at: u32| (at % 251) as u8).collect();
    for _ in 0..reader_mb {
        file.write_all(&block).unwrap();
    }
    file.sync_all().unwrap();
    drop(file);
    std::process::Command::new("sync").status().unwrap();
    // Sources gone from memory, by what reaches mapped pages, as a node's
    // would be under pressure: asked for all the file pages there are.
    tokio::time::sleep(Duration::from_secs(12)).await;
    let before_reclaim = sampled(&group);
    let asked = before_reclaim.stat.get("file").copied().unwrap_or(0);
    let reclaimed = std::fs::write(format!("{group}/memory.reclaim"), asked.to_string());
    let after_reclaim = sampled(&group);
    eprintln!(
        "reclaim of {asked}: {reclaimed:?}, file {} to {}",
        before_reclaim.stat.get("file").copied().unwrap_or(0),
        after_reclaim.stat.get("file").copied().unwrap_or(0)
    );
    let (stop, span, spans) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), Arc::new(std::sync::Mutex::new(Vec::new())));
    let reading = std::thread::spawn({
        let (stop, span, spans) = (stop.clone(), span.clone(), spans.clone());
        move || reader(hot, stop, span, spans)
    });
    // Warmed, then a quiet span to weigh the merge's against.
    tokio::time::sleep(Duration::from_secs(6)).await;
    span.fetch_add(1, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_secs(4)).await;
    span.fetch_add(1, Ordering::SeqCst);
    let base = sampled(&group);
    let base_allocated = ALLOCATOR.held().unwrap().allocated as u64;
    static SAMPLING: AtomicBool = AtomicBool::new(true);
    let sampler = std::thread::spawn({
        let (group, base) = (group.clone(), base.clone());
        move || {
            let of = |sample: &Sample, name: &str| sample.stat.get(name).copied().unwrap_or(0);
            let (mut anon, mut file, mut dirty, mut writeback, mut total) = (0, 0, 0, 0, 0u64);
            let (mut allocated, mut integral, mut last) = (0, 0.0f64, Instant::now());
            while SAMPLING.load(Ordering::SeqCst) {
                let now = sampled(&group);
                let held = ALLOCATOR.held().unwrap().allocated as u64;
                anon = anon.max(of(&now, "anon"));
                file = file.max(of(&now, "file"));
                dirty = dirty.max(of(&now, "file_dirty"));
                writeback = writeback.max(of(&now, "file_writeback"));
                allocated = allocated.max(held);
                let both = held.saturating_sub(base_allocated) + of(&now, "file").saturating_sub(of(&base, "file"));
                total = total.max(both);
                integral += both as f64 * last.elapsed().as_secs_f64();
                last = Instant::now();
                // What a kill leaves to say how far it got.
                println!("sample anon={} file={} dirty={} oom_kill={}", of(&now, "anon"), of(&now, "file"), of(&now, "file_dirty"), now.events.get("oom_kill").copied().unwrap_or(0));
                let _ = std::io::stdout().flush();
                std::thread::sleep(Duration::from_millis(50));
            }
            (anon, file, dirty, writeback, allocated, total, integral)
        }
    });
    let (cpu_before, wall) = (cpu(), Instant::now());
    let merged = writer.merge().await;
    let wall = wall.elapsed().as_secs_f64();
    let cpu_after = cpu();
    span.fetch_add(1, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(50)).await;
    SAMPLING.store(false, Ordering::SeqCst);
    let (anon, file, dirty, writeback, allocated, total, integral) = sampler.join().unwrap();
    let end = sampled(&group);
    stop.store(true, Ordering::SeqCst);
    reading.join().unwrap();
    merged.unwrap().expect("the segments merge");
    let spilled = "not asked".to_string(); // BEFORE-ONLY
    let spilled = writer.last_spilled().to_string(); // AFTER-ONLY
    let clustered = writer.partition().read(|state| state.segments().any(|segment| segment.index.is_some()));
    const MB: f64 = 1_000_000.0;
    let less = |name: &str| end.stat.get(name).copied().unwrap_or(0).saturating_sub(base.stat.get(name).copied().unwrap_or(0));
    let events = |name: &str| end.events.get(name).copied().unwrap_or(0).saturating_sub(base.events.get(name).copied().unwrap_or(0));
    let at = |sample: &Sample, name: &str| sample.stat.get(name).copied().unwrap_or(0) as f64 / MB;
    let spans = spans.lock().unwrap().clone();
    let (quiet, busy) = (spans[1], spans[2]);
    let rate = |read: Read| read.pages as f64 / read.seconds.max(0.001);
    println!(
        "merge rows={rows} dims={dimensions} wall={wall:.2} cpu={:.2} clustered={clustered} spilled={spilled} \
         base_anon={:.1} anon_peak={:.1} base_file={:.1} file_peak={:.1} file_dirty_peak={:.1} file_writeback_peak={:.1} \
         allocated_peak={:.1} base_allocated={:.1} total_above_base={:.1} gb_s={:.2} \
         oom_kill={} oom={} max={} high={} pgmajfault={} refault_file={} refault_anon={} pgscan={} pgsteal={} \
         pressure_some_ms={:.1} pressure_full_ms={:.1} reclaimed_to={:.1} \
         reader_quiet_pages_s={:.0} reader_busy_pages_s={:.0} reader_quiet_slow={} reader_busy_slow={} reader_quiet_worst_us={} reader_busy_worst_us={} reader_quiet_majors={} reader_busy_majors={}",
        cpu_after - cpu_before,
        at(&base, "anon"),
        anon as f64 / MB,
        at(&base, "file"),
        file as f64 / MB,
        dirty as f64 / MB,
        writeback as f64 / MB,
        allocated as f64 / MB,
        base_allocated as f64 / MB,
        total as f64 / MB,
        integral / 1e9,
        events("oom_kill"),
        events("oom"),
        events("max"),
        events("high"),
        less("pgmajfault"),
        less("workingset_refault_file"),
        less("workingset_refault_anon"),
        less("pgscan"),
        less("pgsteal"),
        end.some.saturating_sub(base.some) as f64 / 1000.0,
        end.full.saturating_sub(base.full) as f64 / 1000.0,
        at(&after_reclaim, "file"),
        rate(quiet),
        rate(busy),
        quiet.slow,
        busy.slow,
        quiet.worst_us,
        busy.worst_us,
        quiet.majors,
        busy.majors,
    );
}
