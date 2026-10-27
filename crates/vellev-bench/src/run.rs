//! One engine's run (design r2, §6): fetch is the workflow's; then up,
//! load, settle, warm up, measure and down, each timed.

use std::sync::Arc;
use std::time::{Duration, Instant};

use eyre::{ensure, Result};
use futures::stream::{self, StreamExt, TryStreamExt};
use serde::Serialize;

use crate::engine::{Engine, Knob};
use crate::fixture::{Case, Filter, Fixture, Query, Truth};
use crate::measure::{found_at, median, percentile, Recall};

/// What one run measures.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Passages loaded, the first of the fixture's.
    pub rows: u64,
    pub limits: Vec<usize>,
    pub cases: Vec<Case>,
    /// Repetitions of each latency figure (§2): recall is deterministic
    /// and taken from the first.
    pub reps: usize,
    /// Rows a load request carries, and requests in flight at once.
    pub batch: usize,
    pub in_flight: usize,
    /// The longest a load may take to settle.
    pub settle_within: Duration,
}

/// A figure repeated: its median, and the spread of the repetitions.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Repeated {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Repeated {
    pub fn of(values: &[f64]) -> Repeated {
        Repeated {
            median: median(values),
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// One knob point under one case at one limit.
#[derive(Debug, Clone, Serialize)]
pub struct Measured {
    pub limit: usize,
    pub case: Case,
    pub knob: Knob,
    pub label: String,
    pub recall: Recall,
    pub recall_share: f64,
    pub same_top_1: Recall,
    /// Milliseconds, each the median of the repetitions.
    pub p50_ms: Repeated,
    pub p99_ms: Repeated,
}

#[derive(Debug, Clone, Serialize)]
pub struct Phases {
    pub load_seconds: f64,
    pub settle_seconds: f64,
    pub warmup_seconds: f64,
    pub measure_seconds: f64,
    pub rows_per_second_loading: f64,
}

/// Load the plan's rows, as many requests in flight as it says.
pub async fn load(engine: Arc<dyn Engine>, fixture: &Fixture, plan: &Plan) -> Result<f64> {
    let started = Instant::now();
    let batches = fixture.passages(plan.rows, plan.batch)?;
    let loaded = std::sync::atomic::AtomicU64::new(0);
    let total = plan.rows;
    stream::iter(batches)
        .map(|batch| {
            let engine = engine.clone();
            let loaded = &loaded;
            async move {
                let batch = batch?;
                engine.load(&batch).await?;
                let done = loaded
                    .fetch_add(batch.len() as u64, std::sync::atomic::Ordering::SeqCst)
                    + batch.len() as u64;
                if done % 50_000 < batch.len() as u64 {
                    eprintln!(
                        "{}: {done} of {total} rows loaded, {:.0} s",
                        engine.name(),
                        started.elapsed().as_secs_f64()
                    );
                }
                eyre::Ok(())
            }
        })
        .buffer_unordered(plan.in_flight)
        .try_collect::<Vec<()>>()
        .await?;
    Ok(started.elapsed().as_secs_f64())
}

/// Wait for the engine to say the writes are in place.
pub async fn settle(engine: &dyn Engine, within: Duration) -> Result<f64> {
    let started = Instant::now();
    loop {
        if engine.settled().await? {
            return Ok(started.elapsed().as_secs_f64());
        }
        ensure!(
            started.elapsed() < within,
            "{} not settled within {within:?}",
            engine.name()
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// Ask each warm-up question once, at the engine's default, unfiltered but
/// for deleted rows. Never measured.
pub async fn warm(engine: &dyn Engine, warmup: &[Query]) -> Result<f64> {
    let started = Instant::now();
    let default = engine.points(10)[0];
    engine.set(default).await?;
    for query in warmup {
        engine
            .nearest(
                &query.vector,
                Filter::of(Case::Broad, query.owner),
                10,
                default,
            )
            .await?;
    }
    Ok(started.elapsed().as_secs_f64())
}

/// One pass over the queries, one at a time: the ids of each answer, and
/// how long each took in milliseconds.
async fn pass(
    engine: &dyn Engine,
    queries: &[Query],
    case: Case,
    limit: usize,
    knob: Knob,
) -> Result<(Vec<Vec<u64>>, Vec<f64>)> {
    let mut answers = Vec::with_capacity(queries.len());
    let mut took = Vec::with_capacity(queries.len());
    for query in queries {
        let filter = Filter::of(case, query.owner);
        let started = Instant::now();
        let ids = engine.nearest(&query.vector, filter, limit, knob).await?;
        took.push(started.elapsed().as_secs_f64() * 1e3);
        answers.push(ids);
    }
    Ok((answers, took))
}

/// Every knob point of the plan, under every case, at every limit.
pub async fn measure(
    engine: &dyn Engine,
    queries: &[Query],
    truth: &Truth,
    plan: &Plan,
) -> Result<Vec<Measured>> {
    let mut measured = Vec::new();
    for &limit in &plan.limits {
        for knob in engine.points(limit) {
            engine.set(knob).await?;
            for &case in &plan.cases {
                let started = Instant::now();
                let (mut p50s, mut p99s) = (Vec::new(), Vec::new());
                let (mut recall, mut same) = (Recall::default(), Recall::default());
                for rep in 0..plan.reps {
                    let (answers, took) = pass(engine, queries, case, limit, knob).await?;
                    if rep == 0 {
                        for (query, ids) in queries.iter().zip(&answers) {
                            let truth = truth.of(query.id, case)?;
                            recall = recall + found_at(limit, ids, truth);
                            same = same
                                + Recall {
                                    found: u64::from(crate::measure::same_top_1(ids, truth)),
                                    of: 1,
                                };
                        }
                    }
                    p50s.push(percentile(&took, 50.0));
                    p99s.push(percentile(&took, 99.0));
                }
                let point = Measured {
                    limit,
                    case,
                    knob,
                    label: knob.label(),
                    recall,
                    recall_share: recall.share(),
                    same_top_1: same,
                    p50_ms: Repeated::of(&p50s),
                    p99_ms: Repeated::of(&p99s),
                };
                eprintln!(
                    "{} limit {limit} {} {}: recall {:.4} top-1 {:.3} p50 {:.2} ms p99 {:.2} ms ({:.0} s)",
                    engine.name(),
                    case.name(),
                    point.label,
                    point.recall_share,
                    same.share(),
                    point.p50_ms.median,
                    point.p99_ms.median,
                    started.elapsed().as_secs_f64(),
                );
                measured.push(point);
            }
        }
    }
    Ok(measured)
}
