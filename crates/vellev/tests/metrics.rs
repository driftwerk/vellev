//! What a node measures of the requests it answers, and where it serves
//! the measures: on a listener of their own, never on the API's.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::http::{Method, StatusCode};
use common::*;
use serde_json::{json, Value};
use vellev::metrics::{self, Stage};
use vellev_node::Settings;

const HYBRID: &str = "/v1/tenants/{tenant}/collections/{collection}/hybrid";
const LIST: &str = "/v1/tenants/{tenant}/collections/{collection}/list";

/// The value of the sample `name{labels}`, the labels given in the order
/// the node writes them; `None` if it has no such sample.
fn sample(text: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    let labels: Vec<String> = labels
        .iter()
        .map(|(label, value)| format!("{label}=\"{value}\""))
        .collect();
    let wanted = match labels.is_empty() {
        true => format!("{name} "),
        false => format!("{name}{{{}}} ", labels.join(",")),
    };
    text.lines()
        .find_map(|line| line.strip_prefix(&wanted))
        .map(|value| value.trim().parse().unwrap())
}

fn stage_count(text: &str, route: &str, stage: Stage) -> f64 {
    sample(
        text,
        "vellev_stage_seconds_count",
        &[("route", route), ("stage", stage.name())],
    )
    .unwrap_or(0.0)
}

fn work_sum(text: &str, route: &str, kind: &str) -> Option<f64> {
    sample(
        text,
        "vellev_search_work_sum",
        &[("route", route), ("kind", kind)],
    )
}

/// A node with `acme`'s notes embedded through a stand-in endpoint, three
/// of them, and a key for `acme`.
async fn embedded() -> (Served, String) {
    let url = keyed_embedding_server("sk-0123456789abcdef").await;
    let served = Cluster::new()
        .serve_embedding_keyed(&url, "sk-0123456789abcdef")
        .await;
    served
        .with_notes(json!({
            "provider": "http",
            "api": "openai",
            "url": url,
            "model": "stand-in",
            "dimensions": 2,
            "properties": ["title"],
        }))
        .await;
    for (n, title, kept) in [(1, "a red fox", 1), (2, "a fox den", 1), (3, "hens", 0)] {
        let note = json!({ "properties": { "title": title, "n": kept } });
        served.ok(Method::PUT, &object(n), Some(note)).await;
    }
    let issued = served.ok(Method::POST, "/v1/tenants/acme/keys", None).await;
    (served, issued["key"].as_str().unwrap().to_string())
}

async fn hybrid(served: &Served, key: &str, body: Value) -> Value {
    let path = format!("{NOTES}/hybrid");
    let (status, answer) = served
        .call_as(Some(key), Method::POST, &path, Some(body))
        .await;
    assert_eq!(status, StatusCode::OK, "{answer}");
    answer
}

#[tokio::test]
async fn a_search_by_text_is_timed_whole_and_in_every_stage() {
    let (served, key) = embedded().await;
    let filter = json!({ "eq": { "property": "n", "value": 1 } });
    let body = json!({ "text": "fox", "limit": 10, "filter": filter });
    hybrid(&served, &key, body).await;
    let text = served.app.metrics().encoded();
    for stage in Stage::ALL {
        assert_eq!(stage_count(&text, HYBRID, stage), 1.0, "{stage:?}\n{text}");
    }
    let whole = [("route", HYBRID), ("method", "POST"), ("status", "2xx")];
    assert_eq!(
        sample(&text, "vellev_request_seconds_count", &whole),
        Some(1.0)
    );
    served.stop().await;
}

#[tokio::test]
async fn a_search_reports_the_work_it_did_and_its_filters_selectivity() {
    let (served, key) = embedded().await;
    let filter = json!({ "eq": { "property": "n", "value": 1 } });
    let body = json!({ "text": "fox", "limit": 10, "filter": filter });
    hybrid(&served, &key, body).await;
    let text = served.app.metrics().encoded();
    assert_eq!(
        work_sum(&text, HYBRID, "exact_memtable"),
        Some(2.0),
        "the two notes the filter allows, each scored once\n{text}"
    );
    assert_eq!(work_sum(&text, HYBRID, "exact_segments"), Some(0.0));
    assert_eq!(
        work_sum(&text, HYBRID, "keyword_rows"),
        Some(2.0),
        "both allowed notes hold fox"
    );
    assert!(work_sum(&text, HYBRID, "postings").unwrap() >= 2.0);
    let share = sample(&text, "vellev_filter_selectivity_sum", &[("route", HYBRID)]);
    assert_eq!(share, Some(2.0 / 3.0), "two of three notes");
    served.stop().await;
}

#[tokio::test]
async fn a_read_that_is_no_search_does_no_search_work() {
    let (served, key) = embedded().await;
    let path = format!("{NOTES}/list");
    let (status, _) = served
        .call_as(
            Some(&key),
            Method::POST,
            &path,
            Some(json!({ "limit": 10 })),
        )
        .await;
    assert_eq!(status, StatusCode::OK);
    let text = served.app.metrics().encoded();
    assert_eq!(stage_count(&text, LIST, Stage::Read), 1.0);
    assert_eq!(stage_count(&text, LIST, Stage::Embed), 0.0);
    assert_eq!(work_sum(&text, LIST, "exact_memtable"), Some(0.0));
    assert_eq!(work_sum(&text, LIST, "postings"), Some(0.0));
    served.stop().await;
}

#[tokio::test]
async fn a_key_is_counted_as_looked_up_once_and_then_remembered() {
    let (served, key) = embedded().await;
    let lookups = |text: &str, outcome: &str| {
        sample(text, "vellev_key_lookups_total", &[("outcome", outcome)])
    };
    let before = served.app.metrics().encoded();
    assert_eq!(lookups(&before, "looked_up"), Some(0.0));
    for _ in 0..3 {
        hybrid(&served, &key, json!({ "text": "fox", "limit": 1 })).await;
    }
    let text = served.app.metrics().encoded();
    assert_eq!(lookups(&text, "looked_up"), Some(1.0), "{text}");
    assert_eq!(lookups(&text, "remembered"), Some(2.0));
    assert_eq!(lookups(&text, "busy"), Some(0.0));
    assert_eq!(
        stage_count(&text, HYBRID, Stage::Auth),
        3.0,
        "each request's key is found, however"
    );
    served.stop().await;
}

#[tokio::test]
async fn an_admin_key_is_no_lookup() {
    let (served, _) = embedded().await;
    served
        .ok(
            Method::POST,
            &format!("{NOTES}/hybrid"),
            Some(json!({ "text": "fox", "limit": 1 })),
        )
        .await;
    let text = served.app.metrics().encoded();
    assert_eq!(stage_count(&text, HYBRID, Stage::Auth), 0.0);
    assert_eq!(stage_count(&text, HYBRID, Stage::Read), 1.0);
    served.stop().await;
}

#[tokio::test]
async fn the_tenants_a_node_opens_and_its_reads_in_flight_are_counted() {
    let (served, _) = embedded().await;
    let text = served.app.metrics().encoded();
    let opens = |kind: &str| sample(&text, "vellev_tenant_opens_total", &[("kind", kind)]);
    assert_eq!(opens("writer"), Some(1.0), "{text}");
    assert_eq!(opens("copy"), Some(0.0));
    assert_eq!(sample(&text, "vellev_reads_in_flight", &[]), Some(0.0));
    assert_eq!(sample(&text, "vellev_reads_waiting", &[]), Some(0.0));
    served.stop().await;
}

#[tokio::test]
async fn metrics_are_served_on_their_own_listener_and_never_on_the_apis() {
    let (served, _) = embedded().await;
    for key in [Some(KEY), None] {
        let (status, _) = served.call_as(key, Method::GET, "/metrics", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "the API serves no metrics");
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/metrics", listener.local_addr().unwrap());
    let router = metrics::router(served.app.clone());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let response = reqwest::get(&url).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let kind = response.headers()["content-type"]
        .to_str()
        .unwrap()
        .to_string();
    assert!(kind.starts_with("text/plain; version=0.0.4"), "{kind}");
    let text = response.text().await.unwrap();
    assert!(text.contains("vellev_tenant_opens_total"), "{text}");
    assert!(text.ends_with("# EOF\n"), "{text}");
    served.stop().await;
}

#[tokio::test]
async fn a_refused_request_is_counted_by_its_status_class() {
    let (served, key) = embedded().await;
    let path = format!("{NOTES}/hybrid");
    let body = json!({ "text": "fox", "limit": 1, "alpha": 5.0 });
    let (status, answer) = served
        .call_as(Some(&key), Method::POST, &path, Some(body))
        .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{answer}");
    let text = served.app.metrics().encoded();
    let class = |status: &str| {
        let labels = [("route", HYBRID), ("method", "POST"), ("status", status)];
        sample(&text, "vellev_request_seconds_count", &labels)
    };
    assert_eq!(class("4xx"), Some(1.0), "{text}");
    assert_eq!(class("2xx"), None);
    served.stop().await;
}

#[tokio::test]
async fn the_stages_of_a_request_do_not_overlap() {
    let (served, key) = embedded().await;
    hybrid(&served, &key, json!({ "text": "fox", "limit": 10 })).await;
    let text = served.app.metrics().encoded();
    let stages: f64 = Stage::ALL
        .iter()
        .map(|stage| {
            sample(
                &text,
                "vellev_stage_seconds_sum",
                &[("route", HYBRID), ("stage", stage.name())],
            )
            .unwrap_or(0.0)
        })
        .sum();
    let whole = [("route", HYBRID), ("method", "POST"), ("status", "2xx")];
    let whole = sample(&text, "vellev_request_seconds_sum", &whole).unwrap();
    assert!(
        stages <= whole,
        "the stages, {stages}s, are parts of the request, {whole}s\n{text}"
    );
    served.stop().await;
}

/// Track 2, V1: what waits in memory for a flush, and flushes that failed,
/// are on /metrics, so that rows piling up behind failing flushes are seen
/// before writes are refused for them.
#[tokio::test]
async fn what_waits_for_a_flush_and_flushes_that_failed_are_reported() {
    let (served, _key) = embedded().await;
    // What waits for a flush is as the last sweep found it.
    served.node.sweep_now().await;
    let text = served.app.metrics().encoded();
    let value = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .and_then(|value| value.parse::<f64>().ok())
    };
    assert_eq!(value("vellev_flush_failures_total"), Some(0.0), "{text}");
    assert_eq!(value("vellev_writes_backlogged_total"), Some(0.0), "{text}");
    assert_eq!(
        value("vellev_merge_turns_let_go_total"),
        Some(0.0),
        "{text}"
    );
    assert_eq!(value("vellev_merge_draining_bytes"), Some(0.0), "{text}");
    assert_eq!(value("vellev_merge_abandoned_bytes"), Some(0.0), "{text}");
    assert!(
        value("vellev_unflushed_rows").is_some_and(|rows| rows >= 1.0),
        "{text}"
    );
    assert!(
        value("vellev_unflushed_bytes").is_some_and(|bytes| bytes > 0.0),
        "{text}"
    );
    served.stop().await;
}

/// Bench signals: the thread pools a node runs on, and how many reads may
/// run at once, are reported as they are, so that a run that sets them
/// can check they took.
#[tokio::test]
async fn the_thread_pools_and_reads_at_once_are_reported_as_they_are() {
    let (served, _) = embedded().await;
    let text = served.app.metrics().encoded();
    let threads = |pool: &str| sample(&text, "vellev_threads", &[("pool", pool)]);
    let workers = tokio::runtime::Handle::current().metrics().num_workers();
    assert_eq!(threads("tokio_workers"), Some(workers as f64), "{text}");
    assert_eq!(threads("rayon"), Some(rayon::current_num_threads() as f64));
    let limits = served.ok(Method::GET, "/v1/limits", None).await;
    assert_eq!(
        sample(&text, "vellev_reads_at_once", &[]),
        limits["reads_at_once"].as_f64(),
        "{limits}"
    );
    served.stop().await;
}

/// Bench signals: tenants with a flush or merge due or under way are
/// counted; three notes are fewer than a flush.
#[tokio::test]
async fn tenants_with_a_flush_or_merge_due_are_counted() {
    let (served, _) = embedded().await;
    served.node.sweep_now().await;
    let text = served.app.metrics().encoded();
    assert_eq!(
        sample(&text, "vellev_tenants_unsettled", &[]),
        Some(0.0),
        "{text}"
    );
    served.stop().await;
}

/// The segments a node's tenants hold are reported by whether they are
/// clustered, with their rows; three notes are in no segment yet.
#[tokio::test]
async fn segments_are_reported_by_whether_they_are_clustered() {
    let (served, _) = embedded().await;
    served.node.sweep_now().await;
    let text = served.app.metrics().encoded();
    for index in ["clustered", "plain"] {
        assert_eq!(
            sample(&text, "vellev_segments", &[("index", index)]),
            Some(0.0),
            "{text}"
        );
        assert_eq!(
            sample(&text, "vellev_segment_rows", &[("index", index)]),
            Some(0.0)
        );
    }
    // And the clustered indexes by where their probes come from.
    for source in ["rows", "queries", "unknown", "unfit", "index"] {
        assert_eq!(
            sample(&text, "vellev_segment_probes", &[("source", source)]),
            Some(0.0),
            "{text}"
        );
    }
    served.stop().await;
}

/// Write `acme` three notes and `beta` one, with vectors supplied.
async fn two_tenants(served: &Served) {
    served.with_notes(supplied()).await;
    for n in 1..=3 {
        let note = json!({ "properties": { "title": format!("note {n}"), "n": n }, "vector": [1.0, n as f64] });
        served.ok(Method::PUT, &object(n), Some(note)).await;
    }
    note_in(served, "beta", 1).await;
}

/// Write `tenant` one note, numbered `n`, creating it and its notes
/// first if it has none.
async fn note_in(served: &Served, tenant: &str, n: u64) {
    let notes = format!("/v1/tenants/{tenant}/collections/notes");
    if n == 1 {
        served
            .ok(Method::PUT, &format!("/v1/tenants/{tenant}"), None)
            .await;
        served
            .ok(Method::PUT, &notes, Some(common::notes(supplied())))
            .await;
    }
    let note = json!({ "properties": { "title": format!("note {n}"), "n": n }, "vector": [1.0, n as f64] });
    let path = format!("{notes}/objects/{}", id(n));
    served.ok(Method::PUT, &path, Some(note)).await;
}

/// A node with a memory budget of `budget` bytes, if any, and two
/// tenants: `acme` with three notes and `beta` with one.
async fn holding(budget: Option<usize>) -> Served {
    let cluster = Cluster::new();
    let served = match budget {
        Some(bytes) => cluster.serve_within_memory(bytes).await,
        None => cluster.serve().await,
    };
    two_tenants(&served).await;
    served
}

/// Resident accounting PR 1: what the node reckons its tenants hold, the
/// figure its memory budget is held to, is on /metrics with that budget,
/// with how many tenants it holds open and the most any one holds, so
/// that they can be seen beside what the process really holds.
#[tokio::test]
async fn what_the_tenants_hold_and_the_memory_budget_are_reported() {
    const BUDGET: usize = 64 * 1024 * 1024;
    let served = holding(Some(BUDGET)).await;
    served.node.sweep_now().await;
    let text = served.app.metrics().encoded();
    let resident = served.node.resident();
    assert!(resident > 0, "two tenants with notes hold something");
    assert_eq!(
        sample(&text, "vellev_tenants_resident_bytes", &[("by", "rows")]),
        Some(resident as f64),
        "{text}"
    );
    assert_eq!(
        sample(&text, "vellev_memory_budget_bytes", &[]),
        Some(BUDGET as f64),
        "{text}"
    );
    assert_eq!(
        sample(&text, "vellev_tenants_open", &[("kind", "writer")]),
        Some(2.0),
        "{text}"
    );
    assert_eq!(
        sample(&text, "vellev_tenants_open", &[("kind", "copy")]),
        Some(0.0),
        "{text}"
    );
    // `acme` holds three notes to `beta`'s one: the most one holds is
    // more than half of what both hold, and less than all of it.
    let largest = sample(
        &text,
        "vellev_tenant_resident_largest_bytes",
        &[("by", "rows")],
    )
    .unwrap_or_else(|| panic!("no largest: {text}"));
    assert!(
        largest > resident as f64 / 2.0 && largest < resident as f64,
        "the most one of two tenants holds is {largest} of {resident}: {text}"
    );
    served.stop().await;
}

/// What the tenants hold, how many are open, and what waits for a flush
/// are reported as the last sweep found them: the figures the memory
/// budget was last held to, read without holding any tenant. Rows written
/// since show once the node next sweeps.
#[tokio::test]
async fn what_the_tenants_hold_is_reported_as_the_last_sweep_found_it() {
    let served = Cluster::new()
        .serve_unstarted_with(|settings| Settings {
            memory: Some(64 * 1024 * 1024),
            ..settings
        })
        .await;
    note_in(&served, "acme", 1).await;
    served.node.sweep_now().await;
    let swept = served.app.metrics().encoded();
    let resident = |text: &str| sample(text, "vellev_tenants_resident_bytes", &[("by", "rows")]);
    let unflushed = |text: &str| sample(text, "vellev_unflushed_rows", &[]);
    assert_eq!(unflushed(&swept), Some(1.0), "{swept}");
    let first = resident(&swept).unwrap_or_else(|| panic!("{swept}"));
    note_in(&served, "acme", 2).await;
    note_in(&served, "beta", 1).await;
    let unswept = served.app.metrics().encoded();
    assert_eq!(resident(&unswept), Some(first), "as last swept: {unswept}");
    assert_eq!(unflushed(&unswept), Some(1.0), "as last swept: {unswept}");
    assert_eq!(
        sample(&unswept, "vellev_tenants_open", &[("kind", "writer")]),
        Some(1.0),
        "as last swept: {unswept}"
    );
    served.node.sweep_now().await;
    let again = served.app.metrics().encoded();
    assert!(
        resident(&again).is_some_and(|bytes| bytes > first),
        "{again}"
    );
    assert_eq!(unflushed(&again), Some(3.0), "{again}");
    assert_eq!(
        sample(&again, "vellev_tenants_open", &[("kind", "writer")]),
        Some(2.0),
        "{again}"
    );
    served.stop().await;
}

/// Review of #165, M1: reading the metrics never changes what the node
/// does. A sweep lets a tenant go only if nothing but the node holds it,
/// and over the budget stops the merges of those it must keep; a scrape
/// that held the tenants as it read them would keep them, and have their
/// merges stopped. Scraped without let-up while eight idle tenants put
/// the node over its budget, every one is let go and no merge is stopped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scraping_never_keeps_a_tenant_from_being_let_go() {
    let served = Cluster::new()
        .serve_made(
            |settings| Settings {
                memory: Some(1),
                // Swept every second.
                tenant_idle: Duration::from_secs(4),
                ..settings
            },
            |app| app,
        )
        .await;
    let tenants: Vec<String> = (0..8).map(|n| format!("t{n}")).collect();
    for tenant in &tenants {
        note_in(&served, tenant, 1).await;
    }
    let scraping = Arc::new(AtomicBool::new(true));
    let scrapers: Vec<_> = (0..2)
        .map(|_| {
            let (app, scraping) = (served.app.clone(), scraping.clone());
            std::thread::spawn(move || {
                let mut scrapes = 0u64;
                while scraping.load(Ordering::Relaxed) {
                    std::hint::black_box(app.metrics().encoded());
                    scrapes += 1;
                }
                scrapes
            })
        })
        .collect();
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    // Let go, and a sweep since that found none.
    while served.node.census().holding.writers > 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "tenants still held: {:?}",
            served.node.census().holding
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    scraping.store(false, Ordering::Relaxed);
    let scrapes: u64 = scrapers.into_iter().map(|s| s.join().unwrap()).sum();
    assert!(scrapes > 100, "scraped only {scrapes} times");
    assert_eq!(
        served.node.merges().stopped_for_budget,
        0,
        "a sweep stopped merges of tenants it found held: by the scrapes"
    );
    let text = served.app.metrics().encoded();
    assert_eq!(
        sample(&text, "vellev_merges_stopped_for_budget_total", &[]),
        Some(0.0),
        "{text}"
    );
    served.stop().await;
}

/// A node with no memory budget says so by having no budget to report,
/// not a budget of nothing; what its tenants hold is reported all the
/// same.
#[tokio::test]
async fn a_node_without_a_memory_budget_reports_none() {
    let served = holding(None).await;
    served.node.sweep_now().await;
    let text = served.app.metrics().encoded();
    assert!(
        !text.contains("vellev_memory_budget_bytes"),
        "no budget, so none reported: {text}"
    );
    assert!(
        sample(&text, "vellev_tenants_resident_bytes", &[("by", "rows")])
            .is_some_and(|bytes| bytes > 0.0),
        "{text}"
    );
    served.stop().await;
}

const ALLOCATOR_COUNTS: [&str; 2] = [
    "vellev_allocator_allocated_bytes",
    "vellev_allocator_resident_bytes",
];

/// The allocator's own counts are reported only by a node told which
/// allocator it runs on: this test binary runs on the system's, where
/// jemalloc's counts would say nothing of the process.
#[tokio::test]
async fn a_node_not_told_its_allocator_reports_no_allocator_counts() {
    let served = holding(None).await;
    let text = served.app.metrics().encoded();
    for name in ALLOCATOR_COUNTS {
        assert!(!text.contains(name), "{name} reported: {text}");
    }
    served.stop().await;
}

/// jemalloc, linked but not the allocator this test binary runs on.
static NOT_GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

/// Review of #165, M5: a node told it runs on jemalloc when it does not
/// finds jemalloc has allocated nothing, and reports none of its counts
/// rather than counts that say nothing of the process.
#[tokio::test]
async fn a_node_told_an_allocator_it_does_not_run_on_reports_no_allocator_counts() {
    let served = Cluster::new()
        .serve_made(|settings| settings, |app| app.running_on(&NOT_GLOBAL))
        .await;
    let text = served.app.metrics().encoded();
    for name in ALLOCATOR_COUNTS {
        assert!(!text.contains(name), "{name} reported: {text}");
    }
    served.stop().await;
}

/// The local embedding models loaded are counted, so that the step in the
/// process's memory as one loads, outside jemalloc, can be told from what
/// tenants hold. This node has none.
#[tokio::test]
async fn the_local_models_loaded_are_counted() {
    let served = holding(None).await;
    let text = served.app.metrics().encoded();
    assert_eq!(
        sample(&text, "vellev_local_models_loaded", &[]),
        Some(0.0),
        "{text}"
    );
    served.stop().await;
}

/// Linux's memory files as a container has them: the process's status,
/// and its cgroup's (v2) memory, its limit and its high mark (`max` for
/// none), its peak, its events and its kinds of memory, with `current` as
/// the cgroup's memory now.
fn linux_files(current: u64, max: &str, high: &str) -> (tempfile::TempDir, MemoryFiles) {
    let dir = tempfile::tempdir().unwrap();
    let status = dir.path().join("status");
    std::fs::write(
        &status,
        "Name:\tvellev\nVmRSS:\t    3000 kB\nRssAnon:\t    2000 kB\nRssFile:\t     900 kB\nRssShmem:\t     100 kB\n",
    )
    .unwrap();
    let cgroup = dir.path().join("cgroup");
    std::fs::create_dir(&cgroup).unwrap();
    let write = |name: &str, text: &str| std::fs::write(cgroup.join(name), text).unwrap();
    write("memory.current", &format!("{current}\n"));
    write("memory.max", &format!("{max}\n"));
    write("memory.high", &format!("{high}\n"));
    write("memory.peak", "6000000\n");
    write("memory.events", "low 0\nhigh 3\nmax 4\noom 1\noom_kill 2\n");
    write(
        "memory.stat",
        "anon 1048576\nfile 2097152\nkernel 4096\nshmem 0\nfile_mapped 8192\n",
    );
    let files = MemoryFiles::at(status, Some(cgroup));
    (dir, files)
}

/// Resident accounting PR 1: the process's resident memory, anonymous and
/// file-backed apart, and its cgroup's memory, peak, limit, high mark,
/// events and kinds of memory are on /metrics as Linux's files say: what
/// the node is really held to, beside what it reckons its tenants hold.
#[tokio::test]
async fn the_process_and_its_cgroups_memory_are_reported_as_linux_says() {
    let (_dir, files) = linux_files(5_000_000, "2147483648", "1500000000");
    let served = Cluster::new().serve_reading_memory_from(files).await;
    let text = served.app.metrics().encoded();
    let value = |name: &str, labels: &[(&str, &str)]| sample(&text, name, labels);
    let resident = "vellev_process_resident_bytes";
    assert_eq!(
        value(resident, &[("kind", "anon")]),
        Some(2_048_000.0),
        "{text}"
    );
    assert_eq!(
        value(resident, &[("kind", "file")]),
        Some(921_600.0),
        "{text}"
    );
    assert_eq!(
        value("vellev_cgroup_memory_bytes", &[]),
        Some(5_000_000.0),
        "{text}"
    );
    assert_eq!(
        value("vellev_cgroup_memory_peak_bytes", &[]),
        Some(6_000_000.0),
        "{text}"
    );
    assert_eq!(
        value("vellev_cgroup_memory_max_bytes", &[]),
        Some(2_147_483_648.0),
        "{text}"
    );
    assert_eq!(
        value("vellev_cgroup_memory_high_bytes", &[]),
        Some(1_500_000_000.0),
        "{text}"
    );
    let events = "vellev_cgroup_memory_events_total";
    assert_eq!(value(events, &[("event", "high")]), Some(3.0), "{text}");
    assert_eq!(value(events, &[("event", "max")]), Some(4.0), "{text}");
    assert_eq!(value(events, &[("event", "oom")]), Some(1.0), "{text}");
    assert_eq!(value(events, &[("event", "oom_kill")]), Some(2.0), "{text}");
    let stat = "vellev_cgroup_memory_stat_bytes";
    assert_eq!(
        value(stat, &[("kind", "anon")]),
        Some(1_048_576.0),
        "{text}"
    );
    assert_eq!(
        value(stat, &[("kind", "file")]),
        Some(2_097_152.0),
        "{text}"
    );
    served.stop().await;
}

/// Where there are no such files, as on macOS, or no cgroup v2, nothing
/// of them is reported: no figure at all, never a figure of nothing. A
/// cgroup with no limit or high mark reports neither; one too old to keep
/// its peak reports none; and a figure too large to be one is left out.
#[tokio::test]
async fn memory_linux_does_not_say_is_not_reported() {
    let dir = tempfile::tempdir().unwrap();
    let nowhere = MemoryFiles::at(dir.path().join("status"), Some(dir.path().join("cgroup")));
    let served = Cluster::new().serve_reading_memory_from(nowhere).await;
    let text = served.app.metrics().encoded();
    for name in [
        "vellev_process_resident_bytes",
        "vellev_cgroup_memory",
        "vellev_memory_baseline_bytes",
    ] {
        assert!(!text.contains(name), "{name} reported: {text}");
    }
    served.stop().await;
    let none = MemoryFiles::at(dir.path().join("status"), None);
    let served = Cluster::new().serve_reading_memory_from(none).await;
    let text = served.app.metrics().encoded();
    assert!(!text.contains("vellev_cgroup_memory"), "{text}");
    served.stop().await;
    let (unlimited_dir, unlimited) = linux_files(5_000_000, "max", "max");
    std::fs::remove_file(unlimited_dir.path().join("cgroup/memory.peak")).unwrap();
    std::fs::write(
        unlimited_dir.path().join("status"),
        format!("RssAnon:\t{} kB\nRssFile:\t900 kB\n", u64::MAX / 1000),
    )
    .unwrap();
    let served = Cluster::new().serve_reading_memory_from(unlimited).await;
    let text = served.app.metrics().encoded();
    assert_eq!(
        sample(&text, "vellev_cgroup_memory_bytes", &[]),
        Some(5_000_000.0),
        "{text}"
    );
    for name in [
        "vellev_cgroup_memory_max_bytes",
        "vellev_cgroup_memory_high_bytes",
        "vellev_cgroup_memory_peak_bytes",
    ] {
        assert!(!text.contains(name), "{name} reported: {text}");
    }
    let resident = "vellev_process_resident_bytes";
    assert_eq!(sample(&text, resident, &[("kind", "anon")]), None, "{text}");
    assert_eq!(
        sample(&text, resident, &[("kind", "file")]),
        Some(921_600.0),
        "{text}"
    );
    served.stop().await;
}

/// What the process and its cgroup held as the node started, before any
/// tenant was opened, is kept as it was then: the node's own baseline,
/// which what its tenants hold comes on top of.
#[tokio::test]
async fn the_memory_held_as_the_node_started_is_kept_as_its_baseline() {
    let (dir, files) = linux_files(5_000_000, "max", "max");
    let served = Cluster::new().serve_reading_memory_from(files).await;
    std::fs::write(dir.path().join("cgroup/memory.current"), "7000000\n").unwrap();
    std::fs::write(
        dir.path().join("status"),
        "RssAnon:\t    9000 kB\nRssFile:\t     900 kB\n",
    )
    .unwrap();
    let text = served.app.metrics().encoded();
    let value = |name: &str, labels: &[(&str, &str)]| sample(&text, name, labels);
    assert_eq!(
        value("vellev_cgroup_memory_bytes", &[]),
        Some(7_000_000.0),
        "{text}"
    );
    let baseline = "vellev_memory_baseline_bytes";
    assert_eq!(
        value(baseline, &[("kind", "cgroup")]),
        Some(5_000_000.0),
        "{text}"
    );
    assert_eq!(
        value(baseline, &[("kind", "process_anon")]),
        Some(2_048_000.0),
        "{text}"
    );
    assert_eq!(
        value("vellev_process_resident_bytes", &[("kind", "anon")]),
        Some(9_216_000.0),
        "{text}"
    );
    served.stop().await;
}
