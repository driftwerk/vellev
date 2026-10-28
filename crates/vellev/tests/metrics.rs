//! What a node measures of the requests it answers, and where it serves
//! the measures: on a listener of their own, never on the API's.

mod common;

use axum::http::{Method, StatusCode};
use common::*;
use serde_json::{json, Value};
use vellev::metrics::{self, Stage};

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
    let text = served.app.metrics().encoded();
    let value = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name} ")))
            .and_then(|value| value.parse::<f64>().ok())
    };
    assert_eq!(value("vellev_flush_failures_total"), Some(0.0), "{text}");
    assert_eq!(value("vellev_writes_backlogged_total"), Some(0.0), "{text}");
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
    for source in ["rows", "queries", "unknown", "index"] {
        assert_eq!(
            sample(&text, "vellev_segment_probes", &[("source", source)]),
            Some(0.0),
            "{text}"
        );
    }
    served.stop().await;
}
