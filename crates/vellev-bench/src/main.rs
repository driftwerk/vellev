//! `vellev-bench`: runs both engines, one after the other, on one runner,
//! and writes `results.json`, the Markdown summary and vellev's
//! `/metrics` snapshot (design r2, §6).

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use eyre::{ensure, Result, WrapErr};
use serde_json::{json, Value};
use vellev_bench::docker::{self, Pinning};
use vellev_bench::engine::{Engine, Vellev, Weaviate};
use vellev_bench::fixture::{Case, Fixture, Truth};
use vellev_bench::report;
use vellev_bench::run::{self, Measured, Phases, Plan};

#[derive(Parser, Debug)]
struct Args {
    /// The unpacked fixture.
    #[arg(long)]
    fixture: PathBuf,
    /// The truth for the rows loaded.
    #[arg(long)]
    truth: PathBuf,
    /// How many of the fixture's passages to load, from the first.
    #[arg(long)]
    rows: u64,
    /// Where results.json, summary.md and the metrics go.
    #[arg(long)]
    out: PathBuf,
    /// Where the engines keep their data on the runner's disk.
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    vellev_image: String,
    #[arg(long)]
    weaviate_image: String,
    #[arg(long)]
    postgres_image: String,
    /// `vellev,weaviate`, `weaviate,vellev`, or `alternate`: vellev first
    /// on even days of the year (§2).
    #[arg(long, default_value = "alternate")]
    order: String,
    #[arg(long, value_delimiter = ',', default_value = "10,100")]
    limits: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "broad,mid,narrow,owner")]
    cases: Vec<String>,
    #[arg(long, default_value_t = 3)]
    reps: usize,
    #[arg(long, default_value = "0-2")]
    cpuset: String,
    #[arg(long, default_value_t = 3)]
    cpus: u32,
    #[arg(long, default_value_t = 10_240)]
    memory_mb: u64,
    #[arg(long, default_value_t = 500)]
    batch: usize,
    #[arg(long, default_value_t = 4)]
    in_flight: usize,
    /// Weaviate's tuned build (§1): maxConnections 64, efConstruction 256.
    #[arg(long)]
    tuned: bool,
    /// Measure only the first N queries (a smoke run); all by default.
    #[arg(long)]
    queries: Option<usize>,
}

fn shell(command: &str) -> String {
    Command::new("sh")
        .args(["-c", command])
        .output()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_default()
}

/// What the run ran on (§2).
fn environment() -> Value {
    json!({
        "lscpu": shell("lscpu"),
        "kernel": shell("uname -a"),
        "cgroup": shell("stat -fc %T /sys/fs/cgroup"),
        "docker": shell("docker version --format '{{json .Server}}'"),
        "runner_image": std::env::var("ImageVersion").ok(),
        "runner_os": std::env::var("ImageOS").ok(),
        "commit": std::env::var("GITHUB_SHA").ok(),
        "run": std::env::var("GITHUB_RUN_ID").ok(),
        "client_cpus": shell("taskset -pc $$ 2>/dev/null"),
    })
}

/// vellev first on even days of the year.
fn vellev_first(order: &str) -> Result<bool> {
    Ok(match order {
        "vellev,weaviate" => true,
        "weaviate,vellev" => false,
        "alternate" => {
            let day: u32 = shell("date -u +%j")
                .trim_start_matches('0')
                .parse()
                .unwrap_or(0);
            day.is_multiple_of(2)
        }
        other => eyre::bail!("no order {other:?}"),
    })
}

struct EngineRun {
    settings: Value,
    phases: Phases,
    measured: Vec<Measured>,
    floor: run::Floor,
    hybrid: Vec<run::HybridPoint>,
    hybrid_reference: &'static str,
    image: Value,
    extra: Value,
}

async fn measured(
    engine: Arc<dyn Engine>,
    fixture: &Fixture,
    truth: &Truth,
    plan: &Plan,
    queries: &[vellev_bench::fixture::Query],
    warmup: &[vellev_bench::fixture::Query],
) -> Result<(Phases, Vec<Measured>, run::Floor)> {
    engine.create().await?;
    let load_seconds = run::load(engine.clone(), fixture, plan).await?;
    let settle_seconds = run::settle(engine.as_ref(), plan.settle_within).await?;
    eprintln!(
        "{}: loaded in {load_seconds:.0} s, settled {settle_seconds:.0} s later",
        engine.name()
    );
    let warmup_seconds = run::warm(engine.as_ref(), warmup).await?;
    let started = std::time::Instant::now();
    let measured = run::measure(engine.as_ref(), queries, truth, plan).await?;
    let floor = run::floor(engine.as_ref(), queries, plan.reps).await?;
    eprintln!(
        "{}: a query of the empty collection, p50 {:.2} ms p99 {:.2} ms",
        engine.name(),
        floor.p50_ms.median,
        floor.p99_ms.median
    );
    Ok((
        Phases {
            load_seconds,
            settle_seconds,
            warmup_seconds,
            measure_seconds: started.elapsed().as_secs_f64(),
            rows_per_second_loading: plan.rows as f64 / load_seconds,
        },
        measured,
        floor,
    ))
}

async fn vellev(
    args: &Args,
    fixture: &Fixture,
    truth: &Truth,
    plan: &Plan,
    pinning: &Pinning,
    queries: &[vellev_bench::fixture::Query],
    warmup: &[vellev_bench::fixture::Query],
) -> Result<EngineRun> {
    docker::pull(&args.postgres_image).ok();
    let data = args.data.join("vellev");
    let up = docker::vellev_up(
        &args.vellev_image,
        &args.postgres_image,
        &data,
        pinning,
        &[],
    )?;
    let outcome = async {
        let url = format!("http://127.0.0.1:{}", docker::VELLEV_PORT);
        docker::ready(&format!("{url}/health/ready"), Duration::from_secs(120), &up.vellev).await?;
        let engine = Arc::new(Vellev::new(
            &url,
            &format!("http://127.0.0.1:{}", docker::VELLEV_METRICS_PORT),
            docker::VELLEV_KEY,
            "bench",
        )?);
        // The pools were set by environment (§2): read back before anything
        // is loaded, or the run fails.
        Vellev::threads_as_set(&engine.metrics().await?, pinning.cpus)?;
        let (phases, measured, floor) =
            measured(engine.clone(), fixture, truth, plan, queries, warmup).await?;
        let reference = run::vellev_reference(&engine, queries, truth).await?;
        let hybrid = run::hybrid(engine.as_ref(), queries, &reference, plan.reps).await?;
        let settings = engine.settings().await?;
        let metrics = engine.metrics().await?;
        std::fs::write(args.out.join("vellev-metrics.txt"), &metrics)?;
        eyre::Ok(EngineRun {
            settings,
            phases,
            measured,
            floor,
            hybrid,
            hybrid_reference: "the truth's exact top 100 as the vector leg, vellev's keyword search as the keyword leg, fused by vellev's own fusion (vellev_index::fusion::fuse) at alpha 0.75 with relative scores; fused in f32 from f64 cosines, so fused scores within about 1e-6 may swap at the 10th place",
            image: json!({ "image": args.vellev_image, "digest": docker::digest(&args.vellev_image)?, "postgres": args.postgres_image, "postgres_digest": docker::digest(&args.postgres_image)? }),
            extra: json!({ "postgres": "runs on the same CPUs as vellev, within the same memory: 512 MiB of it" }),
        })
    }
    .await;
    if outcome.is_err() {
        eprintln!("vellev logs:\n{}", up.vellev.logs());
    }
    docker::down(&up.containers(), &data)?;
    outcome
}

async fn weaviate(
    args: &Args,
    fixture: &Fixture,
    truth: &Truth,
    plan: &Plan,
    pinning: &Pinning,
    queries: &[vellev_bench::fixture::Query],
    warmup: &[vellev_bench::fixture::Query],
) -> Result<EngineRun> {
    docker::pull(&args.weaviate_image).ok();
    let data = args.data.join("weaviate");
    let container = docker::weaviate_up(&args.weaviate_image, &data, pinning)?;
    let outcome = async {
        let url = format!("http://127.0.0.1:{}", docker::WEAVIATE_PORT);
        docker::ready(&format!("{url}/v1/.well-known/ready"), Duration::from_secs(120), &container).await?;
        let index = match args.tuned {
            true => json!({ "maxConnections": 64, "efConstruction": 256 }),
            false => json!({}),
        };
        let engine = Arc::new(Weaviate::new(&url, index)?);
        let (phases, measured, floor) =
            measured(engine.clone(), fixture, truth, plan, queries, warmup).await?;
        let reference = run::weaviate_reference(engine.as_ref(), queries, plan.rows).await?;
        let hybrid = run::hybrid(engine.as_ref(), queries, &reference, plan.reps).await?;
        let settings = engine.settings().await?;
        eyre::Ok(EngineRun {
            settings,
            phases,
            measured,
            floor,
            hybrid,
            hybrid_reference: "Weaviate's own hybrid (alpha 0.75, relativeScoreFusion) with ef set to the number of rows: near-exhaustive, not proven exact",
            image: json!({ "image": args.weaviate_image, "digest": docker::digest(&args.weaviate_image)?, "tuned": args.tuned }),
            extra: json!({
                "env": { "GOMAXPROCS": pinning.cpus, "LIMIT_RESOURCES": true },
                "settled": "the vector queue of every shard empty and its indexing READY; Weaviate exposes no signal for compaction, so compaction is not waited for",
            }),
        })
    }
    .await;
    if outcome.is_err() {
        eprintln!("weaviate logs:\n{}", container.logs());
    }
    docker::down(&[&container], &data)?;
    outcome
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out)?;
    let fixture = Fixture::at(&args.fixture);
    let truth = Truth::read(&args.truth).wrap_err("the truth")?;
    let cases = args
        .cases
        .iter()
        .map(|name| Case::from_name(name))
        .collect::<Result<Vec<_>>>()?;
    let mut queries = fixture.queries("queries")?;
    if let Some(n) = args.queries {
        queries.truncate(n);
    }
    let warmup = fixture.queries("warmup")?;
    let plan = Plan {
        rows: args.rows,
        limits: args.limits.clone(),
        cases: cases.clone(),
        reps: args.reps,
        batch: args.batch,
        in_flight: args.in_flight,
        settle_within: Duration::from_secs(3 * 3600),
    };
    let pinning = Pinning {
        cpuset: args.cpuset.clone(),
        cpus: args.cpus,
        memory_mb: args.memory_mb,
    };
    let first_vellev = vellev_first(&args.order)?;
    let environment = environment();
    let (mut vellev_run, mut weaviate_run) = (None, None);
    for turn in 0..2 {
        if (turn == 0) == first_vellev {
            vellev_run =
                Some(vellev(&args, &fixture, &truth, &plan, &pinning, &queries, &warmup).await?);
        } else {
            weaviate_run =
                Some(weaviate(&args, &fixture, &truth, &plan, &pinning, &queries, &warmup).await?);
        }
    }
    let (vellev_run, weaviate_run) = (vellev_run.expect("ran"), weaviate_run.expect("ran"));
    let mut matches = report::matches("vellev", &vellev_run.measured);
    matches.extend(report::matches("weaviate", &weaviate_run.measured));
    matches.sort_by(|a, b| {
        a.case
            .cmp(&b.case)
            .then(a.target.total_cmp(&b.target))
            .then(a.engine.cmp(&b.engine))
    });
    let gate = report::gate(
        &vellev_run.measured,
        &cases,
        report::hybrid_check(&vellev_run.hybrid)
            .into_iter()
            .collect(),
    );
    let engine_json = |run: &EngineRun| {
        json!({
            "image": run.image,
            "settings": run.settings,
            "phases": run.phases,
            "notes": run.extra,
            "query_floor": run.floor,
            "hybrid": { "reference": run.hybrid_reference, "points": run.hybrid, "matched_0_99": report::hybrid_matched(&run.hybrid) },
            "points": run.measured,
        })
    };
    let results = json!({
        "schema": 1,
        "rows": args.rows,
        "queries": queries.len(),
        "order": if first_vellev { "vellev,weaviate" } else { "weaviate,vellev" },
        "pinning": { "engine_cpuset": args.cpuset, "engine_cpus": args.cpus, "engine_memory_mb": args.memory_mb },
        "reps": args.reps,
        "targets": report::TARGETS.map(|t| t as f64 / 100.0),
        "environment": environment,
        "engines": { "vellev": engine_json(&vellev_run), "weaviate": engine_json(&weaviate_run) },
        "matched": matches,
        "gate": gate,
    });
    std::fs::write(
        args.out.join("results.json"),
        serde_json::to_string_pretty(&results)?,
    )?;
    let header = format!(
        "Order: {}. Engines pinned to CPUs {} ({} CPUs, {} MiB); the client on the rest. Each latency is the median of {} runs.",
        results["order"].as_str().unwrap_or_default(),
        args.cpuset,
        args.cpus,
        args.memory_mb,
        args.reps
    );
    let floors = [
        ("vellev", vellev_run.floor),
        ("weaviate", weaviate_run.floor),
    ];
    let hybrids = [
        ("vellev", report::hybrid_matched(&vellev_run.hybrid)),
        ("weaviate", report::hybrid_matched(&weaviate_run.hybrid)),
    ];
    let summary = report::summary(&matches, &hybrids, &floors, &gate, args.rows, &header);
    std::fs::write(args.out.join("summary.md"), &summary)?;
    println!("{summary}");
    ensure!(gate.passed, "the hard gate failed: see summary.md");
    Ok(())
}
