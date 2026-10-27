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
use vellev_bench::fixture::{Case, Filter, Fixture, Query, Truth};
use vellev_bench::report;
use vellev_bench::run::{self, Measured, Phases, Plan};
use vellev_bench::sample::{self, Sampler};
use vellev_bench::tenants;

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
    /// The SQuAD recall fixture (memories.jsonl, queries.vectors.jsonl), for
    /// the many-tenants measure (§3e); left out, it is not run.
    #[arg(long)]
    tenants: Option<PathBuf>,
    /// The pull-request gate (§8): vellev alone, at its defaults, limit 10;
    /// no bursts, restarts or many tenants. Its exit says whether it held.
    #[arg(long)]
    gate_only: bool,
    /// Concurrent queries in a burst (§4).
    #[arg(long, value_delimiter = ',', default_value = "16,64")]
    burst_sizes: Vec<usize>,
    /// Bursts of each size.
    #[arg(long, default_value_t = 50)]
    bursts: usize,
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
    memory: Value,
    opened: Vec<run::Opened>,
    image: Value,
    extra: Value,
}

/// What every engine goes through once it is up: load, settle, warm up,
/// measure, the fixed cost of a query; each phase sampled for memory and
/// disk.
async fn measured(
    engine: Arc<dyn Engine>,
    sampler: &Sampler,
    fixture: &Fixture,
    truth: &Truth,
    plan: &Plan,
    queries: &[Query],
    warmup: &[Query],
) -> Result<(Phases, Vec<Measured>, run::Floor)> {
    engine.create().await?;
    sampler.phase("load");
    let load_seconds = run::load(engine.clone(), fixture, plan).await?;
    sampler.phase("settle");
    let settle_seconds = run::settle(engine.as_ref(), plan.settle_within).await?;
    eprintln!(
        "{}: loaded in {load_seconds:.0} s, settled {settle_seconds:.0} s later",
        engine.name()
    );
    sampler.phase("warmup");
    let warmup_seconds = run::warm(engine.as_ref(), warmup).await?;
    sampler.phase("measure");
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

/// The engine's anonymous memory at rest: read once settled, warmed and
/// measured, after a few quiet seconds.
async fn steady(cgroups: &[std::path::PathBuf], disk: &std::path::Path) -> Result<sample::Reading> {
    tokio::time::sleep(Duration::from_secs(5)).await;
    sample::read(cgroups, disk)
}

/// Restart the engine `reps` times in `state` and time it until ready and
/// its first query answered (§4).
async fn reopened(
    state: &str,
    reps: usize,
    restart: &(dyn Fn() -> Result<()> + Sync),
    ready_url: &str,
    container: &docker::Container,
    engine: &dyn Engine,
    query: &Query,
) -> Result<run::Opened> {
    let (mut ready, mut first) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        restart()?;
        let ready_in = docker::ready(ready_url, Duration::from_secs(600), container).await?;
        let started = std::time::Instant::now();
        let default = engine.points(10)[0];
        engine
            .nearest(
                &query.vector,
                Filter::of(Case::Broad, query.owner),
                10,
                default,
            )
            .await?;
        ready.push(ready_in.as_secs_f64());
        first.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let opened = run::Opened {
        state: state.into(),
        ready_seconds: run::Repeated::of(&ready),
        first_query_ms: run::Repeated::of(&first),
    };
    eprintln!(
        "{}: reopened ({state}): ready in {:.1} s, first query {:.1} ms",
        engine.name(),
        opened.ready_seconds.median,
        opened.first_query_ms.median
    );
    Ok(opened)
}

fn memory(
    steady: sample::Reading,
    phases: std::collections::BTreeMap<String, sample::PhaseMemory>,
) -> Value {
    json!({
        "steady": {
            "anon_bytes": steady.anon,
            "file_cache_bytes": steady.file,
            "memory_peak_bytes_including_cache": steady.peak_including_cache,
        },
        "phases": phases,
        "note": "anon is memory.stat anon, the engine's own; the file cache is reported apart; memory.peak includes the cache",
    })
}

async fn vellev(
    args: &Args,
    fixture: &Fixture,
    truth: &Truth,
    plan: &Plan,
    pinning: &Pinning,
    queries: &[Query],
    warmup: &[Query],
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
        let ready_url = format!("{url}/health/ready");
        docker::ready(&ready_url, Duration::from_secs(120), &up.vellev).await?;
        let engine = Arc::new(Vellev::new(
            &url,
            &format!("http://127.0.0.1:{}", docker::VELLEV_METRICS_PORT),
            docker::VELLEV_KEY,
            "bench",
        )?);
        // The pools were set by environment (§2): read back before anything
        // is loaded, or the run fails.
        Vellev::threads_as_set(&engine.metrics().await?, pinning.cpus)?;
        let cgroups = vec![up.vellev.cgroup()?, up.postgres.cgroup()?];
        let sampler = Sampler::start(cgroups.clone(), args.data.clone(), "create");
        let (phases, measured, floor) =
            measured(engine.clone(), &sampler, fixture, truth, plan, queries, warmup).await?;
        let at_rest = steady(&cgroups, &args.data).await?;
        sampler.phase("hybrid");
        let reference = run::vellev_reference(&engine, queries, truth).await?;
        let hybrid = run::hybrid(engine.as_ref(), queries, &reference, plan.reps, plan.defaults_only).await?;
        let settings = engine.settings().await?;
        std::fs::write(args.out.join("vellev-metrics.txt"), engine.metrics().await?)?;
        sampler.phase("open");
        let cold = || {
            up.vellev.stop()?;
            docker::emptied(&up.cache)?;
            docker::drop_caches()?;
            up.vellev.start()
        };
        let warm = || {
            up.vellev.stop()?;
            up.vellev.start()
        };
        let opened = match plan.defaults_only {
            true => vec![],
            false => vec![
                reopened("empty local disk (its cache directory emptied, the store kept) and the OS page cache dropped", plan.reps, &cold, &ready_url, &up.vellev, engine.as_ref(), &queries[0]).await?,
                reopened("warm OS page cache", plan.reps, &warm, &ready_url, &up.vellev, engine.as_ref(), &queries[0]).await?,
            ],
        };
        let memory = memory(at_rest, sampler.stop());
        eyre::Ok(EngineRun {
            settings,
            phases,
            measured,
            floor,
            hybrid,
            hybrid_reference: "the truth's exact top 100 as the vector leg, vellev's keyword search as the keyword leg, fused by vellev's own fusion (vellev_index::fusion::fuse) at alpha 0.75 with relative scores; fused in f32 from f64 cosines, so fused scores within about 1e-6 may swap at the 10th place",
            memory,
            opened,
            image: json!({ "image": args.vellev_image, "digest": docker::digest(&args.vellev_image)?, "postgres": args.postgres_image, "postgres_digest": docker::digest(&args.postgres_image)? }),
            extra: json!({ "postgres": "runs on the same CPUs as vellev, within the same memory (512 MiB of it), and its memory is counted with vellev's" }),
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
    queries: &[Query],
    warmup: &[Query],
) -> Result<EngineRun> {
    docker::pull(&args.weaviate_image).ok();
    let data = args.data.join("weaviate");
    let container = docker::weaviate_up(&args.weaviate_image, &data, pinning, &[])?;
    let outcome = async {
        let url = format!("http://127.0.0.1:{}", docker::WEAVIATE_PORT);
        let ready_url = format!("{url}/v1/.well-known/ready");
        docker::ready(&ready_url, Duration::from_secs(120), &container).await?;
        let index = match args.tuned {
            true => json!({ "maxConnections": 64, "efConstruction": 256 }),
            false => json!({}),
        };
        let engine = Arc::new(Weaviate::new(&url, index)?);
        let cgroups = vec![container.cgroup()?];
        let sampler = Sampler::start(cgroups.clone(), args.data.clone(), "create");
        let (phases, measured, floor) =
            measured(engine.clone(), &sampler, fixture, truth, plan, queries, warmup).await?;
        let at_rest = steady(&cgroups, &args.data).await?;
        sampler.phase("hybrid");
        let reference = run::weaviate_reference(engine.as_ref(), queries, plan.rows).await?;
        let hybrid = run::hybrid(engine.as_ref(), queries, &reference, plan.reps, plan.defaults_only).await?;
        let settings = engine.settings().await?;
        sampler.phase("open");
        engine.set(engine.points(10)[0]).await?;
        let cold = || {
            container.stop()?;
            docker::drop_caches()?;
            container.start()
        };
        let warm = || {
            container.stop()?;
            container.start()
        };
        let opened = vec![
            reopened("the OS page cache dropped (Weaviate keeps its data on the local disk alone, so there is no emptier state)", plan.reps, &cold, &ready_url, &container, engine.as_ref(), &queries[0]).await?,
            reopened("warm OS page cache", plan.reps, &warm, &ready_url, &container, engine.as_ref(), &queries[0]).await?,
        ];
        let memory = memory(at_rest, sampler.stop());
        eyre::Ok(EngineRun {
            settings,
            phases,
            measured,
            floor,
            hybrid,
            hybrid_reference: "Weaviate's own hybrid (alpha 0.75, relativeScoreFusion) with ef set to the number of rows: near-exhaustive, not proven exact",
            memory,
            opened,
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

/// The many-tenants measure (§3e) on one engine, sampled for anonymous
/// memory once loaded, settled and searched.
#[allow(clippy::too_many_arguments)]
async fn many_tenants(
    engine: &dyn tenants::Tenanted,
    cgroups: &[std::path::PathBuf],
    disk: &std::path::Path,
    memories: &[tenants::Memory],
    queries: &[Vec<f32>],
    reps: usize,
    index: &str,
    headline: bool,
) -> Result<tenants::Measured> {
    let (load_seconds, settle_seconds, p50_ms, p99_ms) =
        tenants::measure(engine, memories, queries, reps, Duration::from_secs(3600)).await?;
    let anon = steady(cgroups, disk).await?.anon;
    eprintln!(
        "many tenants ({index}): loaded in {load_seconds:.0} s, settled {settle_seconds:.0} s later; p50 {:.2} ms p99 {:.2} ms; anon {:.0} MiB",
        p50_ms.median,
        p99_ms.median,
        anon as f64 / (1024.0 * 1024.0)
    );
    Ok(tenants::Measured {
        index: index.into(),
        headline,
        load_seconds,
        settle_seconds,
        p50_ms,
        p99_ms,
        anon_bytes: anon,
    })
}

async fn vellev_tenants(
    args: &Args,
    dir: &std::path::Path,
    pinning: &Pinning,
    reps: usize,
) -> Result<Vec<tenants::Measured>> {
    let memories = tenants::read_memories(&dir.join("memories.jsonl"))?;
    let queries = tenants::read_queries(&dir.join("queries.vectors.jsonl"))?;
    let data = args.data.join("vellev-tenants");
    // A tenant not searched for 10 minutes is let go (§3e).
    let env = [("VELLEV_TENANT_IDLE_SECS".to_string(), "600".to_string())];
    let up = docker::vellev_up(
        &args.vellev_image,
        &args.postgres_image,
        &data,
        pinning,
        &env,
    )?;
    let outcome = async {
        let url = format!("http://127.0.0.1:{}", docker::VELLEV_PORT);
        docker::ready(
            &format!("{url}/health/ready"),
            Duration::from_secs(120),
            &up.vellev,
        )
        .await?;
        let engine = tenants::VellevTenants {
            url,
            key: docker::VELLEV_KEY.into(),
            metrics: format!("http://127.0.0.1:{}", docker::VELLEV_METRICS_PORT),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(600))
                .build()?,
        };
        let cgroups = vec![up.vellev.cgroup()?, up.postgres.cgroup()?];
        let measured = many_tenants(
            &engine,
            &cgroups,
            &args.data,
            &memories,
            &queries,
            reps,
            "vellev, a tenant each",
            true,
        )
        .await?;
        eyre::Ok(vec![measured])
    }
    .await;
    if outcome.is_err() {
        eprintln!("vellev logs:\n{}", up.vellev.logs());
    }
    docker::down(&up.containers(), &data)?;
    outcome
}

async fn weaviate_tenants(
    args: &Args,
    dir: &std::path::Path,
    pinning: &Pinning,
    reps: usize,
) -> Result<Vec<tenants::Measured>> {
    let memories = tenants::read_memories(&dir.join("memories.jsonl"))?;
    let queries = tenants::read_queries(&dir.join("queries.vectors.jsonl"))?;
    let mut measured = Vec::new();
    // Weaviate's guidance for small tenants: dynamic as the headline, flat
    // beside it; default HNSW reported but not as the headline (§3e). Each
    // in a Weaviate of its own, so that each one's memory is its own.
    for (index, headline) in [("dynamic", true), ("flat", false), ("hnsw", false)] {
        let data = args.data.join(format!("weaviate-tenants-{index}"));
        let env = [("ASYNC_INDEXING".to_string(), "true".to_string())];
        let container = docker::weaviate_up(&args.weaviate_image, &data, pinning, &env)?;
        let outcome = async {
            let url = format!("http://127.0.0.1:{}", docker::WEAVIATE_PORT);
            docker::ready(
                &format!("{url}/v1/.well-known/ready"),
                Duration::from_secs(120),
                &container,
            )
            .await?;
            let engine = tenants::WeaviateTenants {
                url,
                http: reqwest::Client::builder()
                    .timeout(Duration::from_secs(600))
                    .build()?,
                index: index.into(),
            };
            let cgroups = vec![container.cgroup()?];
            let label = format!("weaviate, {index}");
            many_tenants(
                &engine, &cgroups, &args.data, &memories, &queries, reps, &label, headline,
            )
            .await
        }
        .await;
        if outcome.is_err() {
            eprintln!("weaviate logs:\n{}", container.logs());
        }
        docker::down(&[&container], &data)?;
        measured.push(outcome?);
    }
    Ok(measured)
}

/// The resources section of the summary: memory, load and settle, open
/// times, and bursts at each engine's matched broad point.
fn resources(runs: &[(&str, &EngineRun)], matches: &[report::Match]) -> String {
    let mib = |bytes: &Value| bytes.as_f64().unwrap_or(0.0) / (1024.0 * 1024.0);
    let mut out = String::from(
        "\n## Memory, load and open time\n\nAnonymous memory is the engine's own (`memory.stat` anon); vellev's includes its Postgres. Load to settled is not a headline measure: the engines put the work in different places.\n\n| Engine | anon at rest MiB | anon peak MiB (phase) | disk peak GiB | load s | settle s | rows/s |\n|---|---|---|---|---|---|---|\n",
    );
    for (name, run) in runs {
        let phases = run.memory["phases"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        let (peak_phase, peak) = phases
            .iter()
            .map(|(phase, m)| (phase.clone(), mib(&m["anon_peak"])))
            .fold((String::new(), 0.0), |a, b| if b.1 > a.1 { b } else { a });
        let disk = phases
            .values()
            .map(|m| m["disk_used_peak"].as_f64().unwrap_or(0.0))
            .fold(0.0, f64::max)
            / (1024.0 * 1024.0 * 1024.0);
        out.push_str(&format!(
            "| {name} | {:.0} | {peak:.0} ({peak_phase}) | {disk:.1} | {:.0} | {:.0} | {:.0} |\n",
            mib(&run.memory["steady"]["anon_bytes"]),
            run.phases.load_seconds,
            run.phases.settle_seconds,
            run.phases.rows_per_second_loading,
        ));
    }
    out.push_str("\n| Engine | Restart state | ready s | first query ms |\n|---|---|---|---|\n");
    for (name, run) in runs {
        for opened in &run.opened {
            out.push_str(&format!(
                "| {name} | {} | {:.1} | {:.1} |\n",
                opened.state, opened.ready_seconds.median, opened.first_query_ms.median
            ));
        }
    }
    out.push_str("\n### Bursts at each engine's matched point (broad, limit 10, 0.99)\n\nQueries an engine turned away as busy (429 or 503) are left out of the times and counted beside them.\n\n| Engine | Point | Size | p50 ms | p99 ms | max ms | refused |\n|---|---|---|---|---|---|---|\n");
    for m in matches
        .iter()
        .filter(|m| m.case == Case::Broad && m.target == 0.99)
    {
        if let Some(point) = &m.point {
            for burst in &point.bursts {
                out.push_str(&format!(
                    "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {} of {} |\n",
                    m.engine,
                    point.label,
                    burst.size,
                    burst.p50_ms.median,
                    burst.p99_ms.median,
                    burst.max_ms.median,
                    burst.refused,
                    burst.asked
                ));
            }
        }
    }
    out
}

/// The pull-request gate (§8): vellev alone at its defaults on the slice
/// loaded, held to the same thresholds as the nightly.
#[allow(clippy::too_many_arguments)]
async fn gate_only(
    args: &Args,
    fixture: &Fixture,
    truth: &Truth,
    plan: &Plan,
    pinning: &Pinning,
    queries: &[Query],
    warmup: &[Query],
    cases: &[Case],
    environment: Value,
) -> Result<()> {
    let run = vellev(args, fixture, truth, plan, pinning, queries, warmup).await?;
    let gate = report::gate(
        &run.measured,
        cases,
        report::hybrid_check(&run.hybrid).into_iter().collect(),
    );
    let results = json!({
        "schema": 1,
        "kind": "pull-request gate",
        "rows": args.rows,
        "queries": queries.len(),
        "environment": environment,
        "vellev": { "image": run.image, "settings": run.settings, "phases": run.phases, "points": run.measured, "hybrid": run.hybrid },
        "gate": gate,
    });
    std::fs::write(
        args.out.join("results.json"),
        serde_json::to_string_pretty(&results)?,
    )?;
    let summary = report::summary(
        &[],
        &[],
        &[],
        &gate,
        args.rows,
        "The pull-request gate: vellev at its defaults, limit 10.",
    );
    std::fs::write(args.out.join("summary.md"), &summary)?;
    println!("{summary}");
    ensure!(gate.passed, "the hard gate failed: see summary.md");
    Ok(())
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
        defaults_only: args.gate_only,
        burst_sizes: match args.gate_only {
            true => vec![],
            false => args.burst_sizes.clone(),
        },
        bursts: args.bursts,
    };
    let pinning = Pinning {
        cpuset: args.cpuset.clone(),
        cpus: args.cpus,
        memory_mb: args.memory_mb,
    };
    let environment = environment();
    if args.gate_only {
        return gate_only(
            &args,
            &fixture,
            &truth,
            &plan,
            &pinning,
            &queries,
            &warmup,
            &cases,
            environment,
        )
        .await;
    }
    let first_vellev = vellev_first(&args.order)?;
    let (mut vellev_run, mut weaviate_run) = (None, None);
    let (mut vellev_many, mut weaviate_many) = (Vec::new(), Vec::new());
    for turn in 0..2 {
        if (turn == 0) == first_vellev {
            vellev_run =
                Some(vellev(&args, &fixture, &truth, &plan, &pinning, &queries, &warmup).await?);
            if let Some(dir) = &args.tenants {
                vellev_many = vellev_tenants(&args, dir, &pinning, args.reps).await?;
            }
        } else {
            weaviate_run =
                Some(weaviate(&args, &fixture, &truth, &plan, &pinning, &queries, &warmup).await?);
            if let Some(dir) = &args.tenants {
                weaviate_many = weaviate_tenants(&args, dir, &pinning, args.reps).await?;
            }
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
            "memory": run.memory,
            "opened": run.opened,
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
        "many_tenants": {
            "vellev": vellev_many,
            "weaviate": weaviate_many,
            "tenants": "200 of 593 rows and one of 50,000, cut independently from the SQuAD recall fixture's 118,609 rows; each query asks a tenant drawn with a fixed seed",
            "settings": "vellev lets a tenant go after 10 minutes idle (VELLEV_TENANT_IDLE_SECS=600); Weaviate runs with ASYNC_INDEXING=true, which its dynamic index needs",
            "idle_tenant_open": "not published: Weaviate has no equivalent of letting a tenant go once idle (its tenants go cold only when told), so the same event cannot be measured on both sides (§3e)",
        },
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
    let mut summary = report::summary(&matches, &hybrids, &floors, &gate, args.rows, &header);
    if !vellev_many.is_empty() || !weaviate_many.is_empty() {
        summary.push_str("\n## Many tenants\n\n200 tenants of 593 rows and one of 50,000, each query asking a tenant drawn with a fixed seed, limit 10. The cost of opening an idle tenant is not published: Weaviate has no equivalent of letting a tenant go once idle.\n\n| Engine | Headline | p50 ms | p99 ms | anon MiB | load s |\n|---|---|---|---|---|---|\n");
        for m in vellev_many.iter().chain(&weaviate_many) {
            summary.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.0} | {:.0} |\n",
                m.index,
                if m.headline { "yes" } else { "no" },
                m.p50_ms.median,
                m.p99_ms.median,
                m.anon_bytes as f64 / (1024.0 * 1024.0),
                m.load_seconds
            ));
        }
    }
    summary.push_str(&resources(
        &[("vellev", &vellev_run), ("weaviate", &weaviate_run)],
        &matches,
    ));
    std::fs::write(args.out.join("summary.md"), &summary)?;
    println!("{summary}");
    ensure!(gate.passed, "the hard gate failed: see summary.md");
    Ok(())
}
