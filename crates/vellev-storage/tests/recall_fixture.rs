//! The hybrid rescore budget's recall on real embeddings (burst plan F1).
//!
//! Ignored unless asked for, and reads its data from outside the
//! repository: `VELLEV_RECALL_FIXTURE` names a JSON Lines file, one object
//! a line, `{"id": "<uuid>", "content": "<text>", "vector": [f32; N]}`. No
//! fixture is committed. Run it built for speed:
//!
//! ```sh
//! VELLEV_RECALL_FIXTURE=/path/memories.jsonl \
//!   cargo test --release -p vellev-storage --test recall_fixture -- --ignored --nocapture
//! ```
//!
//! Every 69th object, up to 200, is held out as a query: its vector, and
//! the first six words of its content as the text. The rest are written
//! in three layouts: one segment; segments of uneven size, one too small
//! to shortlist, and a memtable; and one segment with a memtable. Each
//! object is given a bucket from 0 to 99 by its id, so that a filter can
//! allow all (no filter), a tenth (mid) or a hundredth (narrow).
//!
//! For each layout, filter and limit (10, 50, 100) it compares the fused
//! top `limit` of a hybrid search with the oversample a default search
//! gets under each cap swept, with that of the same search whose vector
//! leg is exact. It prints the mean overlap and the share with the same
//! top result for each cap, and holds [`HYBRID_DEFAULT_RESCORE`] to a mean
//! overlap of at least 0.99 and the same top result for at least 99% of
//! queries, everywhere.

mod common;

use common::*;
use vellev_storage::{Hybrid, Work, DEFAULT_OVERSAMPLE, HYBRID_DEFAULT_RESCORE};

const MEMORIES: &str = "memories";
const CAPS: [usize; 7] = [128, 256, 512, 1_024, 2_048, 4_096, 8_192];
const LIMITS: [usize; 3] = [10, 50, 100];

struct Memory {
    id: uuid::Uuid,
    content: String,
    vector: Vec<f32>,
}

fn fixture() -> Option<Vec<Memory>> {
    let path = std::env::var("VELLEV_RECALL_FIXTURE").ok()?;
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    Some(
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                Memory {
                    id: value["id"].as_str().unwrap().parse().unwrap(),
                    content: value["content"].as_str().unwrap_or_default().to_string(),
                    vector: serde_json::from_value(value["vector"].clone()).unwrap(),
                }
            })
            .collect(),
    )
}

fn bucket(id: &uuid::Uuid) -> i64 {
    (id.as_u128() % 100) as i64
}

fn create(dimensions: usize) -> Mutation {
    Mutation::CreateCollection(Box::new(Collection {
        name: name(MEMORIES),
        properties: vec![
            Property::new("content", DataType::Text).searchable(),
            Property::new("bucket", DataType::Int),
        ],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "fixture".into(),
            dimensions: dimensions as u32,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }))
}

fn row(memory: &Memory) -> Mutation {
    Mutation::Upsert {
        collection: name(MEMORIES),
        id: ObjectId(memory.id),
        properties: json!({ "content": memory.content, "bucket": bucket(&memory.id) })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(memory.vector.clone()),
        only_if_version: None,
    }
}

/// `rows` written in parts of the given shares, each flushed to a segment
/// of its own but the last when `memtable` is set.
async fn laid_out(rows: &[&Memory], shares: &[f64], memtable: bool) -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create(rows[0].vector.len())])
        .await
        .unwrap();
    let mut at = 0;
    for (i, share) in shares.iter().enumerate() {
        let end = match i == shares.len() - 1 {
            true => rows.len(),
            false => at + (rows.len() as f64 * share) as usize,
        };
        for chunk in rows[at..end].chunks(1_000) {
            writer
                .commit(chunk.iter().map(|memory| row(memory)).collect())
                .await
                .unwrap();
        }
        at = end;
        if !(memtable && i == shares.len() - 1) {
            writer.flush().await.unwrap();
        }
    }
    (cluster, writer)
}

/// The oversample a default search for `limit` gets under `cap`.
fn oversample_under(cap: usize, limit: usize) -> usize {
    DEFAULT_OVERSAMPLE.min((cap / limit.max(1)).max(1))
}

fn ids(state: &State, query: &Hybrid<'_>, work: &mut Work) -> Vec<ObjectId> {
    state
        .hybrid_counted(&name(MEMORIES), query, work)
        .unwrap()
        .iter()
        .map(|fused| fused.object.id)
        .collect()
}

#[tokio::test]
#[ignore = "reads VELLEV_RECALL_FIXTURE; run with --release --ignored"]
async fn the_default_rescore_cap_keeps_recall_on_real_embeddings() {
    let Some(all) = fixture() else {
        panic!("set VELLEV_RECALL_FIXTURE to a JSON Lines file of embeddings");
    };
    let (mut queries, mut rows) = (Vec::new(), Vec::new());
    for (n, memory) in all.iter().enumerate() {
        match n % 69 == 0 && queries.len() < 200 {
            true => queries.push(memory),
            false => rows.push(memory),
        }
    }
    let layouts: [(&str, &[f64], bool); 3] = [
        ("one segment", &[1.0], false),
        (
            "uneven segments and a memtable",
            &[0.5, 0.3, 0.06, 0.14],
            true,
        ),
        ("a segment and a memtable", &[0.85, 0.15], true),
    ];
    let filters = [
        ("all", None),
        (
            "a tenth",
            Some(Filter::Lt {
                property: "bucket".into(),
                value: json!(10),
            }),
        ),
        (
            "a hundredth",
            Some(Filter::Lt {
                property: "bucket".into(),
                value: json!(1),
            }),
        ),
    ];
    let mut failures = Vec::new();
    println!(
        "{} rows, {} queries, {} dimensions",
        rows.len(),
        queries.len(),
        rows[0].vector.len()
    );
    for (layout, shares, memtable) in layouts {
        let (_cluster, writer) = laid_out(&rows, shares, memtable).await;
        writer.partition().read(|state| {
            for (allowed, filter) in &filters {
                for limit in LIMITS {
                    // Per cap: summed overlap, same top results, rows scored.
                    let mut tally = vec![(0.0, 0usize, 0u64); CAPS.len()];
                    for query in &queries {
                        let text: String = query
                            .content
                            .split_whitespace()
                            .take(6)
                            .collect::<Vec<_>>()
                            .join(" ");
                        let base = Hybrid {
                            filter: filter.as_ref(),
                            ..Hybrid::matching(&text, Some(&query.vector), limit)
                        };
                        let exact = ids(
                            state,
                            &Hybrid {
                                oversample: usize::MAX,
                                ..base.clone()
                            },
                            &mut Work::default(),
                        );
                        for (at, cap) in CAPS.iter().enumerate() {
                            let mut work = Work::default();
                            let found = ids(
                                state,
                                &Hybrid {
                                    oversample: oversample_under(*cap, limit),
                                    ..base.clone()
                                },
                                &mut work,
                            );
                            let shared = exact.iter().filter(|id| found.contains(id)).count();
                            tally[at].0 += match exact.len() {
                                0 => 1.0,
                                len => shared as f64 / len as f64,
                            };
                            tally[at].1 += usize::from(found.first() == exact.first());
                            tally[at].2 += work.exact_segments + work.exact_memtable;
                        }
                    }
                    let asked = queries.len() as f64;
                    for (at, cap) in CAPS.iter().enumerate() {
                        let (overlap, same, scored) = tally[at];
                        let (mean, first) = (overlap / asked, same as f64 / asked);
                        println!(
                            "{layout:32} {allowed:12} limit {limit:3} cap {cap:5}: \
                             overlap {mean:.4}  same top {first:.3}  rows scored {:.0}",
                            scored as f64 / asked
                        );
                        if *cap == HYBRID_DEFAULT_RESCORE && (mean < 0.99 || first < 0.99) {
                            failures.push(format!(
                                "{layout}, {allowed}, limit {limit}: overlap {mean:.4}, same top {first:.3}"
                            ));
                        }
                    }
                }
            }
        });
    }
    assert!(
        failures.is_empty(),
        "the default cap of {HYBRID_DEFAULT_RESCORE} loses recall:\n{}",
        failures.join("\n")
    );
}
