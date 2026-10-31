//! What a node measures of the requests it answers, served for Prometheus
//! on a listener of its own (`VELLEV_METRICS_LISTEN`), never on the API's.
//!
//! Every request to a keyed route is timed whole, and in the stages that
//! make it up (see [`Stage`]), so that a slow answer says where its time
//! went. A search also reports the work it did: the rows it scored, the
//! postings it read and how much of what it searched its filter allowed.
//! Time depends on what else the node is doing; the work does not, so the
//! two together tell a costly search from a starved one.
//!
//! A stage is recorded from wherever it happens on the request's task,
//! through a task-local the timing middleware sets for the request, so no
//! handler passes anything along. Outside a request, recording does
//! nothing.

use std::future::Future;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::{MatchedPath, Request, State as Shared};
use axum::http::header::CONTENT_TYPE;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use prometheus_client::collector::Collector;
use prometheus_client::encoding::{DescriptorEncoder, EncodeLabelSet, EncodeMetric};
use prometheus_client::metrics::counter::ConstCounter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::ConstGauge;
use prometheus_client::metrics::histogram::{exponential_buckets, Histogram};
use prometheus_client::registry::Registry;
use vellev_node::Node;
use vellev_storage::Work;

use crate::allocator::Allocator;
use crate::memory::{Memory, MemoryFiles};
use crate::server::App;
use crate::turns::Turns;

/// A part of answering a request, timed on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Finding the tenant a tenant key reaches: from what the node
    /// remembers, or by asking the coordinator. An admin key has none.
    Auth,
    /// Receiving and parsing a JSON body.
    Body,
    /// Finding what a read is answered from: the tenant's writer, or a
    /// copy, opened or brought up to date if it must be. The tenant's
    /// rate is checked here too.
    Reader,
    /// Embedding a query's text, for a search the node embeds for.
    Embed,
    /// Waiting for one of the node's turns to read.
    Turn,
    /// Waiting, with a turn, for a thread to read on.
    Queue,
    /// The read itself: a search, a page or a count, on its thread.
    Read,
    /// From the read's end to the response being made: what the handler
    /// does with the results, and encoding them.
    Serialise,
}

impl Stage {
    /// Every stage, in the order a read passes through them.
    pub const ALL: [Stage; 8] = [
        Stage::Auth,
        Stage::Body,
        Stage::Reader,
        Stage::Embed,
        Stage::Turn,
        Stage::Queue,
        Stage::Read,
        Stage::Serialise,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Stage::Auth => "auth",
            Stage::Body => "body",
            Stage::Reader => "reader",
            Stage::Embed => "embed",
            Stage::Turn => "turn",
            Stage::Queue => "queue",
            Stage::Read => "read",
            Stage::Serialise => "serialise",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct RequestLabels {
    route: String,
    method: String,
    status: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct StageLabels {
    route: String,
    stage: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct WorkLabels {
    route: String,
    kind: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, EncodeLabelSet)]
struct RouteLabels {
    route: String,
}

type Histograms<L> = Family<L, Histogram, fn() -> Histogram>;

/// Seconds, from 100 µs to about 3.3 s.
fn seconds() -> Histogram {
    Histogram::new(exponential_buckets(0.0001, 2.0, 16))
}

/// Counts of work, from 1 to about 4 million.
fn counts() -> Histogram {
    Histogram::new(exponential_buckets(1.0, 4.0, 12))
}

/// Shares, from a thousandth to the whole.
fn shares() -> Histogram {
    Histogram::new([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
}

/// Where a node reads what memory it really holds: the allocator it runs
/// on, if it was told, and the process's and its cgroup's files.
#[derive(Debug, Clone)]
pub(crate) struct Sources {
    pub allocator: Option<&'static Allocator>,
    pub files: MemoryFiles,
}

impl Default for Sources {
    fn default() -> Self {
        Self {
            allocator: None,
            files: MemoryFiles::of_this_process(),
        }
    }
}

/// What the process and its cgroup held as the node started, before it
/// opened any tenant: what its tenants' memory comes on top of.
#[derive(Debug, Clone, Default)]
struct Baseline {
    memory: Memory,
    allocated: Option<usize>,
}

/// The node's measures.
#[derive(Debug)]
pub struct Metrics {
    registry: Registry,
    sources: Sources,
    baseline: Arc<OnceLock<Baseline>>,
    requests: Histograms<RequestLabels>,
    stages: Histograms<StageLabels>,
    work: Histograms<WorkLabels>,
    selectivity: Histograms<RouteLabels>,
}

impl Metrics {
    /// The measures of a node to be served from `node` once there is
    /// one, its reads taking `turns`.
    pub(crate) fn new(node: Arc<OnceLock<Arc<Node>>>, turns: Arc<Turns>, sources: Sources) -> Self {
        let requests = Histograms::new_with_constructor(seconds as fn() -> Histogram);
        let stages = Histograms::new_with_constructor(seconds as fn() -> Histogram);
        let work = Histograms::new_with_constructor(counts as fn() -> Histogram);
        let selectivity = Histograms::new_with_constructor(shares as fn() -> Histogram);
        let mut registry = Registry::with_prefix("vellev");
        registry.register(
            "request_seconds",
            "How long a request to a keyed route took, from its arrival to its response being made",
            requests.clone(),
        );
        registry.register(
            "stage_seconds",
            "How long each stage of a request took",
            stages.clone(),
        );
        registry.register(
            "search_work",
            "What a search did: rows scored exactly (exact_memtable, exact_segments), rows a clustered index estimated (estimated), postings read and rows scored by the keyword leg (postings, keyword_rows)",
            work.clone(),
        );
        registry.register(
            "filter_selectivity",
            "The share of the rows a filtered search searched that its filter allowed",
            selectivity.clone(),
        );
        let baseline = Arc::<OnceLock<Baseline>>::default();
        registry.register_collector(Box::new(Counted {
            node,
            turns,
            sources: sources.clone(),
            baseline: baseline.clone(),
        }));
        Self {
            registry,
            sources,
            baseline,
            requests,
            stages,
            work,
            selectivity,
        }
    }

    /// Keep what the process and its cgroup hold now as the node's
    /// baseline: called as the node starts, before it opens any tenant.
    /// Only the first call keeps anything.
    pub(crate) fn started(&self) {
        let _ = self.baseline.set(Baseline {
            memory: self.sources.files.read(),
            allocated: self
                .sources
                .allocator
                .and_then(Allocator::held)
                .map(|held| held.allocated),
        });
    }

    /// Everything measured, in Prometheus' text format.
    pub fn encoded(&self) -> String {
        let mut text = String::new();
        prometheus_client::encoding::text::encode(&mut text, &self.registry)
            .expect("writing to a string");
        text
    }

    fn stage(&self, route: &str, stage: Stage, took: Duration) {
        self.stages
            .get_or_create(&StageLabels {
                route: route.to_string(),
                stage: stage.name(),
            })
            .observe(took.as_secs_f64());
    }

    fn work(&self, route: &str, work: &Work) {
        let kinds = [
            ("exact_memtable", work.exact_memtable),
            ("exact_segments", work.exact_segments),
            ("estimated", work.estimated),
            ("postings", work.postings),
            ("keyword_rows", work.keyword_rows),
            // A vector search's parts: more than 1 where it was split.
            ("split_parts", work.split_parts),
        ];
        for (kind, count) in kinds {
            self.work
                .get_or_create(&WorkLabels {
                    route: route.to_string(),
                    kind,
                })
                .observe(count as f64);
        }
        if let Some(share) = work.selectivity() {
            self.selectivity
                .get_or_create(&RouteLabels {
                    route: route.to_string(),
                })
                .observe(share);
        }
    }
}

/// What the node and its turns count, read when the metrics are.
#[derive(Debug)]
struct Counted {
    node: Arc<OnceLock<Arc<Node>>>,
    turns: Arc<Turns>,
    sources: Sources,
    baseline: Arc<OnceLock<Baseline>>,
}

/// A gauge of `value`, as large as a gauge goes if it is larger.
fn gauge_of(value: impl TryInto<i64>) -> ConstGauge<i64> {
    ConstGauge::new(value.try_into().unwrap_or(i64::MAX))
}

impl Counted {
    /// What the process, its cgroup and its allocator hold, and held as the
    /// node started, as far as each can be read: what cannot is left out.
    fn memory(&self, encoder: &mut DescriptorEncoder) -> Result<(), std::fmt::Error> {
        if let Some(held) = self.sources.allocator.and_then(Allocator::held) {
            for (name, help, value) in [
                (
                    "allocator_allocated_bytes",
                    "Bytes the node's Rust code has allocated through jemalloc and not freed; not what libraries allocate through the C library's malloc, as the local embedding model does",
                    held.allocated,
                ),
                (
                    "allocator_resident_bytes",
                    "Bytes in the pages jemalloc holds resident: allocated, its metadata, and freed pages not yet given back",
                    held.resident,
                ),
            ] {
                let gauge = gauge_of(value);
                gauge.encode(encoder.encode_descriptor(name, help, None, gauge.metric_type())?)?;
            }
        }
        let memory = self.sources.files.read();
        labelled(
            encoder,
            "process_resident_bytes",
            "The process's resident memory, as Linux says: anonymous (RssAnon), and backed by files (RssFile), which mapped vectors are",
            "kind",
            [("anon", memory.anon), ("file", memory.file)],
        )?;
        for (name, help, value) in [
            (
                "cgroup_memory_bytes",
                "The memory the node's cgroup holds now (memory.current): what its limit is held to",
                memory.current,
            ),
            (
                "cgroup_memory_peak_bytes",
                "The most memory the node's cgroup has held (memory.peak), where the kernel keeps it",
                memory.peak,
            ),
            (
                "cgroup_memory_max_bytes",
                "The node's cgroup's own memory limit (memory.max), if it has one: the leaf's, not a lower one an ancestor may set, which binds it too",
                memory.max,
            ),
            (
                "cgroup_memory_high_bytes",
                "The node's cgroup's own high mark (memory.high), past which it is throttled and reclaimed from, if it has one",
                memory.high,
            ),
        ] {
            if let Some(value) = value {
                let gauge = gauge_of(value);
                gauge.encode(encoder.encode_descriptor(name, help, None, gauge.metric_type())?)?;
            }
        }
        if !memory.events.is_empty() {
            let counter = ConstCounter::new(0u64);
            let mut family = encoder.encode_descriptor(
                "cgroup_memory_events",
                "Times the node's cgroup was throttled over its high mark (high), met its limit (max), ran out (oom), and had a process killed for it (oom_kill), as memory.events says",
                None,
                counter.metric_type(),
            )?;
            for (event, count) in &memory.events {
                ConstCounter::new(*count).encode(family.encode_family(&[("event", *event)])?)?;
            }
        }
        labelled(
            encoder,
            "cgroup_memory_stat_bytes",
            "The node's cgroup's anonymous and file-backed memory, as memory.stat says",
            "kind",
            memory
                .stat
                .iter()
                .map(|(kind, bytes)| (*kind, Some(*bytes))),
        )?;
        if let Some(baseline) = self.baseline.get() {
            labelled(
                encoder,
                "memory_baseline_bytes",
                "What the process and its cgroup held as the node started, before it opened any tenant: its cgroup's memory, the process's anonymous memory, and what it had allocated through jemalloc. The local embedding model is loaded when first used, so it is not in these: see vellev_local_models_loaded",
                "kind",
                [
                    ("cgroup", baseline.memory.current),
                    ("process_anon", baseline.memory.anon),
                    (
                        "allocator_allocated",
                        baseline.allocated.map(|bytes| bytes as u64),
                    ),
                ],
            )?;
        }
        Ok(())
    }
}

/// A family of gauges `name`, one for each value there is, labelled
/// `label`; none at all if there is no value.
fn labelled(
    encoder: &mut DescriptorEncoder,
    name: &str,
    help: &str,
    label: &str,
    values: impl IntoIterator<Item = (&'static str, Option<u64>)>,
) -> Result<(), std::fmt::Error> {
    let present: Vec<(&str, u64)> = values
        .into_iter()
        .filter_map(|(key, value)| Some((key, value?)))
        .collect();
    if present.is_empty() {
        return Ok(());
    }
    let mut family = encoder.encode_descriptor(name, help, None, gauge_of(0).metric_type())?;
    for (key, value) in present {
        gauge_of(value).encode(family.encode_family(&[(label, key)])?)?;
    }
    Ok(())
}

impl Collector for Counted {
    fn encode(&self, mut encoder: DescriptorEncoder) -> Result<(), std::fmt::Error> {
        let (reading, waiting) = self.turns.counts();
        for (name, help, value) in [
            ("reads_in_flight", "Reads that have a turn now", reading),
            ("reads_waiting", "Reads waiting for a turn now", waiting),
            (
                "reads_at_once",
                "How many reads may have a turn at once",
                self.turns.most(),
            ),
        ] {
            let gauge = ConstGauge::new(value as i64);
            gauge.encode(encoder.encode_descriptor(name, help, None, gauge.metric_type())?)?;
        }
        let gauge = ConstGauge::new(0i64);
        let mut family = encoder.encode_descriptor(
            "threads",
            "The threads of each pool the node runs on: the async runtime's workers, and Rayon's, which builds indexes",
            None,
            gauge.metric_type(),
        )?;
        let workers = tokio::runtime::Handle::try_current()
            .map_or(0, |runtime| runtime.metrics().num_workers());
        for (pool, count) in [
            ("tokio_workers", workers),
            ("rayon", rayon::current_num_threads()),
        ] {
            ConstGauge::new(i64::try_from(count).unwrap_or(i64::MAX))
                .encode(family.encode_family(&[("pool", pool)])?)?;
        }
        self.memory(&mut encoder)?;
        let Some(node) = self.node.get() else {
            return Ok(());
        };
        // As the last sweep found the node, which reading holds no tenant
        // for: a sweep lets go only what nothing else holds.
        let census = node.census();
        let holding = census.holding;
        let age = ConstGauge::new(node.census_age().as_secs_f64());
        age.encode(encoder.encode_descriptor(
            "census_age_seconds",
            "How long ago the last sweep took the census the tenant gauges come from, or since the node started if none has: a sweep that has stopped, and with it the memory budget, shows as this grows",
            None,
            age.metric_type(),
        )?)?;
        let gauge0 = ConstGauge::new(0i64);
        let mut family = encoder.encode_descriptor(
            "tenants_open",
            "Tenants this node holds open, as their writer or as a copy to read from, as the last sweep found them",
            None,
            gauge0.metric_type(),
        )?;
        for (kind, count) in [("writer", holding.writers), ("copy", holding.copies)] {
            gauge_of(count).encode(family.encode_family(&[("kind", kind)])?)?;
        }
        for (name, help, value) in [
            (
                "tenants_resident_bytes",
                "About how many bytes the tenants this node holds take in memory together, as the node reckons them for its memory budget, by how it reckons them (rows: from each row's size), as the last sweep found them: the figure it held to the budget",
                holding.resident,
            ),
            (
                "tenant_resident_largest_bytes",
                "About how many bytes the one tenant this node holds that takes the most takes, reckoned as for the memory budget, as the last sweep found it",
                holding.largest,
            ),
        ] {
            let mut family = encoder.encode_descriptor(name, help, None, gauge0.metric_type())?;
            gauge_of(value).encode(family.encode_family(&[("by", "rows")])?)?;
        }
        if let Some(budget) = node.memory_budget() {
            let gauge = gauge_of(budget);
            gauge.encode(encoder.encode_descriptor(
                "memory_budget_bytes",
                "About how many bytes the tenants this node holds may take in memory together, as it reckons them, before it lets the least recently used go",
                None,
                gauge.metric_type(),
            )?)?;
        }
        let gauge = ConstGauge::new(i64::try_from(node.unsettled()).unwrap_or(i64::MAX));
        gauge.encode(encoder.encode_descriptor(
            "tenants_unsettled",
            "Tenants this node writes for with a flush or merge due or under way",
            None,
            gauge.metric_type(),
        )?)?;
        let held = node.segments();
        for (name, help, values) in [
            (
                "segments",
                "Segments the tenants this node writes hold, by whether they have an index file: clustered counts those with one (a clustered vector index for at least one of their collections), plain the rest",
                [("clustered", held.clustered as u64), ("plain", held.plain as u64)],
            ),
            (
                "segment_rows",
                "Rows in the segments the tenants this node writes hold, by whether the segment has an index file, as the segments were written: rows since replaced or deleted are still counted, and rows of every collection in the segment, indexed or not",
                [("clustered", held.clustered_rows), ("plain", held.plain_rows)],
            ),
        ] {
            let gauge = ConstGauge::new(0i64);
            let mut family = encoder.encode_descriptor(name, help, None, gauge.metric_type())?;
            for (index, value) in values {
                ConstGauge::new(i64::try_from(value).unwrap_or(i64::MAX))
                    .encode(family.encode_family(&[("index", index)])?)?;
            }
        }
        let gauge = ConstGauge::new(0i64);
        let mut family = encoder.encode_descriptor(
            "segment_probes",
            "Clustered indexes of the segments the tenants this node writes hold, one for each, by where their probes come from: a curve of the index's rows, one of questions asked, a curve of a source this version does not know (scanning by the index's own count), a curve of a known source that does not fit its index (corruption or a bug, scanning by the index's own count), or no curve at all (a segment from before curves, scanning by its index's own count until its rows pass)",
            None,
            gauge.metric_type(),
        )?;
        for (source, value) in [
            ("rows", held.probes.rows),
            ("queries", held.probes.queries),
            ("unknown", held.probes.unknown),
            ("unfit", held.probes.unfit),
            ("index", held.probes.index),
        ] {
            ConstGauge::new(i64::try_from(value).unwrap_or(i64::MAX))
                .encode(family.encode_family(&[("source", source)])?)?;
        }
        let keys = node.key_lookups();
        let counter = ConstCounter::new(0u64);
        let mut family = encoder.encode_descriptor(
            "key_lookups",
            "Tenant keys presented, by how they were answered: from what the node remembered, by asking the coordinator, by joining a lookup under way, or turned away as busy",
            None,
            counter.metric_type(),
        )?;
        for (outcome, count) in [
            ("remembered", keys.remembered),
            ("looked_up", keys.looked_up),
            ("shared", keys.shared),
            ("busy", keys.busy),
        ] {
            ConstCounter::new(count).encode(family.encode_family(&[("outcome", outcome)])?)?;
        }
        let opens = node.opens();
        let mut family = encoder.encode_descriptor(
            "tenant_opens",
            "Tenants this node opened, as their writer or as a copy to read from",
            None,
            counter.metric_type(),
        )?;
        for (kind, count) in [("writer", opens.writers), ("copy", opens.copies)] {
            ConstCounter::new(count).encode(family.encode_family(&[("kind", kind)])?)?;
        }
        let merges = node.merges();
        let counter = ConstCounter::new(node.sweeps());
        counter.encode(encoder.encode_descriptor(
            "sweeps",
            "Sweeps begun: those asked for while one ran are coalesced into one more",
            None,
            counter.metric_type(),
        )?)?;
        let counter = ConstCounter::new(node.tenants_kept_by_sweeps());
        counter.encode(encoder.encode_descriptor(
            "sweep_tenants_kept",
            "Tenants a sweep chose to let go, for being idle or for the memory budget, but found something else holding, as a request or a merge does: let go at a later round",
            None,
            counter.metric_type(),
        )?)?;
        let counter = ConstCounter::new(merges.stopped_for_budget);
        counter.encode(encoder.encode_descriptor(
            "merges_stopped_for_budget",
            "Merges sweeps stopped, each counted once, so that a tenant over the memory budget, held by something else as it merged, could be let go",
            None,
            counter.metric_type(),
        )?)?;
        let mut family = encoder.encode_descriptor(
            "merges_stopped",
            "Merges stopped, each counted once, by what first stopped it: a sweep letting its tenant go, its lease lost or its writer unable to write, its tenant dropped, or the node shutting down. Those stopped for the memory budget are counted apart, in merges_stopped_for_budget",
            None,
            counter.metric_type(),
        )?;
        for (reason, count) in [
            ("let_go", merges.stopped.let_go),
            ("lease", merges.stopped.lease),
            ("drop", merges.stopped.drop),
            ("shutdown", merges.stopped.shutdown),
        ] {
            ConstCounter::new(count).encode(family.encode_family(&[("reason", reason)])?)?;
        }
        let gauge = gauge_of(node.freeing());
        gauge.encode(encoder.encode_descriptor(
            "tenants_freeing_bytes",
            "About how many bytes the tenants this node writes have taken out of their state, as a merge's commit does what it replaced, and have yet to free: in the process's memory, not in what they are reckoned to hold, and held to no budget, as it goes by itself",
            None,
            gauge.metric_type(),
        )?)?;
        let gauge = gauge_of(node.local_models_loaded());
        gauge.encode(encoder.encode_descriptor(
            "local_models_loaded",
            "Local embedding models loaded or loading: each takes memory outside jemalloc, which shows as a step in the process's anonymous memory as it loads",
            None,
            gauge.metric_type(),
        )?)?;
        let counter = ConstCounter::new(merges.held_let_go);
        counter.encode(encoder.encode_descriptor(
            "merge_turns_let_go",
            "Merge turns let go as a store held what a stopped or failed merge uploaded \
             longer than any write of it may take",
            None,
            counter.metric_type(),
        )?)?;
        for (name, help, value) in [
            (
                "merge_draining_bytes",
                "About how many bytes merges stopped and not yet ended hold, counted against \
                 the memory budget",
                merges.draining,
            ),
            (
                "merge_abandoned_bytes",
                "About how many bytes stores keep of merges whose turns were let go for it",
                merges.abandoned,
            ),
        ] {
            let gauge = ConstGauge::new(i64::try_from(value).unwrap_or(i64::MAX));
            gauge.encode(encoder.encode_descriptor(name, help, None, gauge.metric_type())?)?;
        }
        let flushing = node.flushing();
        for (name, help, count) in [
            (
                "flush_failures",
                "Flushes that failed, their rows left in memory for the next try",
                flushing.failed,
            ),
            (
                "writes_backlogged",
                "Writes refused as their tenant held as many rows unflushed as it may",
                flushing.backlogged,
            ),
        ] {
            let counter = ConstCounter::new(count);
            counter.encode(encoder.encode_descriptor(
                name,
                help,
                None,
                counter.metric_type(),
            )?)?;
        }
        for (name, help, value) in [
            (
                "unflushed_rows",
                "Rows written to the tenants this node writes and not flushed yet",
                flushing.unflushed_rows,
            ),
            (
                "unflushed_bytes",
                "About how many bytes the rows not flushed yet take in memory",
                flushing.unflushed_bytes,
            ),
        ] {
            let gauge = ConstGauge::new(i64::try_from(value).unwrap_or(i64::MAX));
            gauge.encode(encoder.encode_descriptor(name, help, None, gauge.metric_type())?)?;
        }
        Ok(())
    }
}

/// The request being answered on this task.
#[derive(Debug)]
struct Timing {
    metrics: Arc<Metrics>,
    route: String,
    /// When a read of it ended, from which [`Stage::Serialise`] is timed.
    read_ended: Mutex<Option<Instant>>,
}

tokio::task_local! {
    static TIMING: Arc<Timing>;
}

fn with(f: impl FnOnce(&Timing)) {
    let _ = TIMING.try_with(|timing| f(timing));
}

/// Record that `stage` of the request on this task took `took`.
pub(crate) fn record(stage: Stage, took: Duration) {
    with(|timing| timing.metrics.stage(&timing.route, stage, took));
}

/// `future`, recorded as `stage` of the request on this task, however it
/// ends; not if it is dropped unfinished.
pub(crate) async fn timed<F: Future>(stage: Stage, future: F) -> F::Output {
    let began = Instant::now();
    let output = future.await;
    record(stage, began.elapsed());
    output
}

/// Record what a read of the request on this task did, and that the read
/// is over.
pub(crate) fn read_ended(work: &Work) {
    with(|timing| {
        timing.metrics.work(&timing.route, work);
        *timing.read_ended.lock().expect("timing lock") = Some(Instant::now());
    });
}

fn status_class(response: &Response) -> &'static str {
    match response.status().as_u16() / 100 {
        1 => "1xx",
        2 => "2xx",
        3 => "3xx",
        4 => "4xx",
        _ => "5xx",
    }
}

/// Time a request to a keyed route, whole and in its stages. Layered
/// outside every other route layer, so that authorising it is timed too.
pub(crate) async fn timing(Shared(app): Shared<App>, request: Request, next: Next) -> Response {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map_or_else(|| "unmatched".to_string(), |path| path.as_str().to_string());
    let method = request.method().to_string();
    let timing = Arc::new(Timing {
        metrics: app.metrics().clone(),
        route,
        read_ended: Mutex::default(),
    });
    let began = Instant::now();
    let response = TIMING.scope(timing.clone(), next.run(request)).await;
    let now = Instant::now();
    if let Some(ended) = *timing.read_ended.lock().expect("timing lock") {
        let serialised = now.saturating_duration_since(ended);
        timing
            .metrics
            .stage(&timing.route, Stage::Serialise, serialised);
    }
    timing
        .metrics
        .requests
        .get_or_create(&RequestLabels {
            route: timing.route.clone(),
            method,
            status: status_class(&response),
        })
        .observe(now.duration_since(began).as_secs_f64());
    response
}

/// What the metrics listener serves: `GET /metrics` and nothing else.
pub fn router(app: App) -> Router {
    Router::new().route(
        "/metrics",
        get(move || async move {
            (
                [(CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
                app.metrics().encoded(),
            )
                .into_response()
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stage_has_its_own_name() {
        let names: std::collections::HashSet<_> = Stage::ALL.iter().map(|s| s.name()).collect();
        assert_eq!(names.len(), Stage::ALL.len());
    }

    #[tokio::test]
    async fn outside_a_request_nothing_is_recorded_and_nothing_fails() {
        record(Stage::Read, Duration::from_millis(1));
        read_ended(&Work::default());
        assert_eq!(timed(Stage::Auth, async { 7 }).await, 7);
    }
}
