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

/// The node's measures.
#[derive(Debug)]
pub struct Metrics {
    registry: Registry,
    requests: Histograms<RequestLabels>,
    stages: Histograms<StageLabels>,
    work: Histograms<WorkLabels>,
    selectivity: Histograms<RouteLabels>,
}

impl Metrics {
    /// The measures of a node to be served from `node` once there is
    /// one, its reads taking `turns`.
    pub(crate) fn new(node: Arc<OnceLock<Arc<Node>>>, turns: Arc<Turns>) -> Self {
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
        registry.register_collector(Box::new(Counted { node, turns }));
        Self {
            registry,
            requests,
            stages,
            work,
            selectivity,
        }
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
        let Some(node) = self.node.get() else {
            return Ok(());
        };
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
