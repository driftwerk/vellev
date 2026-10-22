//! How many rows a hybrid search asked over the API scores exactly: with
//! no `oversample`, no more than the node's default allows whatever the
//! limit; with one, as many as asked (burst plan F1).

mod common;

use axum::http::{Method, StatusCode};
use common::*;
use serde_json::{json, Value};
use vellev_core::TenantId;

const HYBRID: &str = "/v1/tenants/{tenant}/collections/{collection}/hybrid";
/// The most rows a hybrid search with no `oversample` scores exactly.
const DEFAULT_RESCORE: f64 = 2_048.0;
const ROWS: u64 = 6_000;

/// `acme`'s notes: `ROWS` of them, flushed to a segment, which shortlists.
async fn flushed() -> Served {
    let served = Served::new().await;
    served.with_notes(supplied()).await;
    for start in (0..ROWS).step_by(200) {
        let changes: Vec<Value> = (start..start + 200)
            .map(|n| {
                let angle = n as f64 * 0.001;
                json!({
                    "op": "upsert",
                    "collection": "notes",
                    "id": id(n + 1),
                    "properties": { "title": "fox", "n": n % 10 },
                    "vector": [angle.cos(), angle.sin()],
                })
            })
            .collect();
        served
            .ok(
                Method::POST,
                "/v1/tenants/acme/changes",
                Some(json!({ "changes": changes })),
            )
            .await;
    }
    let acme = TenantId::new("acme").unwrap();
    served
        .node
        .tenant(&acme)
        .await
        .unwrap()
        .flush()
        .await
        .unwrap();
    served
}

/// The rows the hybrid searches so far scored exactly in segments.
fn scored(served: &Served) -> f64 {
    let text = served.app.metrics().encoded();
    let wanted = format!("vellev_search_work_sum{{route=\"{HYBRID}\",kind=\"exact_segments\"}} ");
    text.lines()
        .find_map(|line| line.strip_prefix(&wanted))
        .map_or(0.0, |value| value.trim().parse().unwrap())
}

async fn search(served: &Served, body: Value) {
    let (status, answer) = served.post(&format!("{NOTES}/hybrid"), body).await;
    assert_eq!(status, StatusCode::OK, "{answer}");
}

#[tokio::test]
async fn a_hybrid_search_with_no_oversample_scores_at_most_the_default_whatever_its_limit() {
    let served = flushed().await;
    let mut before = 0.0;
    for limit in [10, 100, 1_000] {
        let body = json!({ "text": "fox", "vector": [1.0, 0.0], "limit": limit });
        search(&served, body).await;
        let now = scored(&served);
        assert!(
            now - before <= DEFAULT_RESCORE,
            "limit {limit}: scored {} rows exactly",
            now - before
        );
        before = now;
    }
    served.stop().await;
}

#[tokio::test]
async fn an_explicit_oversample_is_honoured_over_the_api() {
    let served = flushed().await;
    let body = json!({ "text": "fox", "vector": [1.0, 0.0], "limit": 10, "oversample": 1_000 });
    search(&served, body).await;
    assert_eq!(
        scored(&served),
        ROWS as f64,
        "10 x 1,000 asked for, every row there is"
    );
    served.stop().await;
}
