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
    /// Only each engine's default point, for the pull-request gate (§8).
    pub defaults_only: bool,
    /// Concurrent queries in a burst (§4), and bursts of each size.
    pub burst_sizes: Vec<usize>,
    pub bursts: usize,
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
    /// At limit 10: queries asked `size` at once.
    pub bursts: Vec<Burst>,
    /// Queries the engine turned away as busy, over every repetition: no
    /// answer for recall, no time for the latencies.
    pub refused: u64,
}

/// Queries asked `size` at once, `bursts` times (§4): each query's time
/// from the burst's start to its answer, over every query of every burst,
/// each figure the median of the repetitions.
#[derive(Debug, Clone, Serialize)]
pub struct Burst {
    pub size: usize,
    pub bursts: usize,
    /// Queries the engine turned away as busy (429 or 503), over the
    /// repetitions: left out of the times, and published beside them.
    pub refused: u64,
    pub asked: u64,
    pub p50_ms: Repeated,
    pub p99_ms: Repeated,
    pub max_ms: Repeated,
}

#[derive(Debug, Clone, Serialize)]
pub struct Phases {
    pub load_seconds: f64,
    pub settle_seconds: f64,
    pub warmup_seconds: f64,
    /// Warm-up questions the engine turned away as busy.
    pub warmup_refused: u64,
    pub measure_seconds: f64,
    pub rows_per_second_loading: f64,
}

/// One query asked and timed: its answer and milliseconds, or `None` if
/// the engine turned it away as busy. Refusals are counted and published
/// beside the times, never taken as answers or as the run failing.
async fn timed<T>(ask: impl std::future::Future<Output = Result<T>>) -> Result<Option<(T, f64)>> {
    let started = Instant::now();
    match ask.await {
        Ok(answer) => Ok(Some((answer, started.elapsed().as_secs_f64() * 1e3))),
        Err(error) if crate::engine::is_refused(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

/// The p50 and p99 of `took`, which a pass all of whose queries were
/// refused has not.
fn quantiles(took: &[f64], what: &str) -> Result<(f64, f64)> {
    ensure!(!took.is_empty(), "every query of {what} was refused");
    Ok((percentile(took, 50.0), percentile(took, 99.0)))
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
pub async fn warm(engine: &dyn Engine, warmup: &[Query]) -> Result<(f64, u64)> {
    let started = Instant::now();
    let default = engine.points(10)[0];
    engine.set(default).await?;
    let mut refused = 0;
    for query in warmup {
        let filter = Filter::of(Case::Broad, query.owner);
        if timed(engine.nearest(&query.vector, filter, 10, default))
            .await?
            .is_none()
        {
            refused += 1;
        }
    }
    Ok((started.elapsed().as_secs_f64(), refused))
}

/// One pass over the queries, one at a time: the ids of each answer, and
/// how long each took in milliseconds.
async fn pass(
    engine: &dyn Engine,
    queries: &[Query],
    case: Case,
    limit: usize,
    knob: Knob,
) -> Result<(Vec<Vec<u64>>, Vec<f64>, u64)> {
    let mut answers = Vec::with_capacity(queries.len());
    let mut took = Vec::with_capacity(queries.len());
    let mut refused = 0;
    for query in queries {
        let filter = Filter::of(case, query.owner);
        match timed(engine.nearest(&query.vector, filter, limit, knob)).await? {
            Some((ids, ms)) => {
                took.push(ms);
                answers.push(ids);
            }
            None => {
                refused += 1;
                answers.push(Vec::new());
            }
        }
    }
    Ok((answers, took, refused))
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
        for knob in points(engine, limit, plan.defaults_only) {
            engine.set(knob).await?;
            for &case in &plan.cases {
                let started = Instant::now();
                let (mut p50s, mut p99s) = (Vec::new(), Vec::new());
                let (mut recall, mut same) = (Recall::default(), Recall::default());
                let mut refused = 0;
                for rep in 0..plan.reps {
                    let (answers, took, turned) = pass(engine, queries, case, limit, knob).await?;
                    refused += turned;
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
                    let (p50, p99) = quantiles(&took, &knob.label())?;
                    p50s.push(p50);
                    p99s.push(p99);
                }
                let mut bursts = Vec::new();
                if limit == 10 {
                    for &size in &plan.burst_sizes {
                        bursts.push(
                            burst(engine, queries, case, knob, size, plan.bursts, plan.reps)
                                .await?,
                        );
                    }
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
                    bursts,
                    refused,
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

/// The fixed cost of a query (§4 beside the latencies): each measured
/// question asked of the empty collection, as the searches are asked, the
/// figures each the median of the repetitions. Reported beside the
/// latencies, never taken off them.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Floor {
    pub p50_ms: Repeated,
    pub p99_ms: Repeated,
    pub refused: u64,
}

pub async fn floor(engine: &dyn Engine, queries: &[Query], reps: usize) -> Result<Floor> {
    let (mut p50s, mut p99s) = (Vec::new(), Vec::new());
    let mut refused = 0;
    for _ in 0..reps {
        let mut took = Vec::with_capacity(queries.len());
        for query in queries {
            match timed(engine.floor(&query.vector)).await? {
                Some((_, ms)) => took.push(ms),
                None => refused += 1,
            }
        }
        let (p50, p99) = quantiles(&took, "the empty collection")?;
        p50s.push(p50);
        p99s.push(p99);
    }
    Ok(Floor {
        p50_ms: Repeated::of(&p50s),
        p99_ms: Repeated::of(&p99s),
        refused,
    })
}

/// How many results each leg of a hybrid search offers for fusing: both
/// engines' default, and the truth's depth, so that the truth's top is
/// exactly what an exact vector leg offers.
pub const CANDIDATES: usize = 100;

/// A hybrid knob point (§4): its recall@10 against the engine's own
/// exact-leg answer, and its latency. Never compared across engines.
#[derive(Debug, Clone, Serialize)]
pub struct HybridPoint {
    pub knob: Knob,
    pub label: String,
    pub recall: Recall,
    pub recall_share: f64,
    pub p50_ms: Repeated,
    pub p99_ms: Repeated,
    pub refused: u64,
}

/// vellev's own answer with an exact vector leg, at limit 10 under broad:
/// the truth's exact top 100 as the vector leg, vellev's keyword search as
/// the keyword leg (exact, as keyword search always is), fused by vellev's
/// own code at its defaults.
pub async fn vellev_reference(
    engine: &crate::engine::Vellev,
    queries: &[Query],
    truth: &Truth,
) -> Result<Vec<Vec<u64>>> {
    let mut reference = Vec::with_capacity(queries.len());
    for query in queries {
        let filter = Filter::of(Case::Broad, query.owner);
        let wording = engine.keyword(&query.text, filter, CANDIDATES).await?;
        let (ids, scores) = truth.with_scores(query.id, Case::Broad)?;
        let meaning: Vec<(u64, f32)> = ids
            .iter()
            .zip(scores)
            .take(CANDIDATES)
            .map(|(&id, &cosine)| (id, cosine as f32))
            .collect();
        let fused = vellev_index::fusion::fuse(
            vellev_index::fusion::Fusion::RelativeScore,
            0.75,
            &meaning,
            &wording,
            10,
        );
        reference.push(fused.iter().map(|f| f.key).collect());
    }
    Ok(reference)
}

/// Weaviate's own answer with a near-exhaustive vector leg (§4: `ef` =
/// rows), at limit 10 under broad.
pub async fn weaviate_reference(
    engine: &dyn Engine,
    queries: &[Query],
    rows: u64,
) -> Result<Vec<Vec<u64>>> {
    let all = Knob::Ef(u32::try_from(rows)?);
    engine.set(all).await?;
    let mut reference = Vec::with_capacity(queries.len());
    for query in queries {
        let filter = Filter::of(Case::Broad, query.owner);
        // The reference must be had: a refusal is asked again, a second
        // later, up to a minute.
        let mut tries = 0;
        let answer = loop {
            match timed(engine.hybrid(&query.text, &query.vector, filter, 10, all)).await? {
                Some((answer, _)) => break answer,
                None if tries < 60 => {
                    tries += 1;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                None => eyre::bail!("query {}: refused a minute long", query.id),
            }
        };
        reference.push(answer);
    }
    Ok(reference)
}

/// Every knob point's hybrid answers at limit 10 under broad, against the
/// engine's own `reference`.
pub async fn hybrid(
    engine: &dyn Engine,
    queries: &[Query],
    reference: &[Vec<u64>],
    reps: usize,
    defaults_only: bool,
) -> Result<Vec<HybridPoint>> {
    let mut measured = Vec::new();
    for knob in points(engine, 10, defaults_only) {
        engine.set(knob).await?;
        let (mut p50s, mut p99s) = (Vec::new(), Vec::new());
        let mut recall = Recall::default();
        let mut refused = 0;
        for rep in 0..reps {
            let mut took = Vec::with_capacity(queries.len());
            for (query, wanted) in queries.iter().zip(reference) {
                let filter = Filter::of(Case::Broad, query.owner);
                let got = match timed(engine.hybrid(&query.text, &query.vector, filter, 10, knob))
                    .await?
                {
                    Some((got, ms)) => {
                        took.push(ms);
                        got
                    }
                    None => {
                        refused += 1;
                        Vec::new()
                    }
                };
                if rep == 0 {
                    recall = recall + found_at(10, &got, wanted);
                }
            }
            let (p50, p99) = quantiles(&took, &knob.label())?;
            p50s.push(p50);
            p99s.push(p99);
        }
        let point = HybridPoint {
            knob,
            label: knob.label(),
            recall,
            recall_share: recall.share(),
            p50_ms: Repeated::of(&p50s),
            p99_ms: Repeated::of(&p99s),
            refused,
        };
        eprintln!(
            "{} hybrid {}: recall against its own exact-leg answer {:.4}, p50 {:.2} ms p99 {:.2} ms",
            engine.name(),
            point.label,
            point.recall_share,
            point.p50_ms.median,
            point.p99_ms.median
        );
        measured.push(point);
    }
    Ok(measured)
}

/// The knob points to measure at `limit`: all of them, or the default
/// alone.
fn points(engine: &dyn Engine, limit: usize, defaults_only: bool) -> Vec<Knob> {
    let mut points = engine.points(limit);
    if defaults_only {
        points.truncate(1);
    }
    points
}

/// `count` bursts of `size` queries asked at once, the queries taken in
/// turn from `queries`, at limit 10.
pub async fn burst(
    engine: &dyn Engine,
    queries: &[Query],
    case: Case,
    knob: Knob,
    size: usize,
    count: usize,
    reps: usize,
) -> Result<Burst> {
    let (mut p50s, mut p99s, mut maxes) = (Vec::new(), Vec::new(), Vec::new());
    let (mut refused, mut asked_all) = (0u64, 0u64);
    let mut next = 0;
    for _ in 0..reps {
        let mut took = Vec::with_capacity(size * count);
        for _ in 0..count {
            let asked: Vec<&Query> = (0..size)
                .map(|n| &queries[(next + n) % queries.len()])
                .collect();
            next = (next + size) % queries.len();
            let started = Instant::now();
            let answered =
                futures::future::try_join_all(asked.into_iter().map(|query| async move {
                    let filter = Filter::of(case, query.owner);
                    match engine.nearest(&query.vector, filter, 10, knob).await {
                        Ok(_) => eyre::Ok(Some(started.elapsed().as_secs_f64() * 1e3)),
                        Err(error) if error.downcast_ref::<crate::engine::Refused>().is_some() => {
                            Ok(None)
                        }
                        Err(error) => Err(error),
                    }
                }))
                .await?;
            asked_all += answered.len() as u64;
            refused += answered.iter().filter(|took| took.is_none()).count() as u64;
            took.extend(answered.into_iter().flatten());
        }
        ensure!(
            !took.is_empty(),
            "every query of {count} bursts of {size} was refused"
        );
        p50s.push(percentile(&took, 50.0));
        p99s.push(percentile(&took, 99.0));
        maxes.push(percentile(&took, 100.0));
    }
    Ok(Burst {
        size,
        bursts: count,
        refused,
        asked: asked_all,
        p50_ms: Repeated::of(&p50s),
        p99_ms: Repeated::of(&p99s),
        max_ms: Repeated::of(&maxes),
    })
}

/// The first query after a restart (§4), in one stated state: how long
/// the engine took to say it was ready, and the first query's own time.
#[derive(Debug, Clone, Serialize)]
pub struct Opened {
    pub state: String,
    pub ready_seconds: Repeated,
    pub first_query_ms: Repeated,
}
