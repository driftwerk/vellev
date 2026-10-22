//! A keyword search under a filter (burst plan K1).
//!
//! - T6b: a search reads postings in proportion to the rows its filter
//!   allows, not to every row holding its terms. Under a narrow filter
//!   the heap of best rows seldom fills, so no bound prunes, and walking
//!   every posting to find the few allowed rows is the whole cost.
//! - T8: what it finds is what scoring every row and then keeping the
//!   allowed ones finds, scores and order alike.

mod common;

use common::*;
use vellev_storage::{Keywords, Work};

const NOTES: &str = "notes";
const ROWS: u32 = 10_000;

fn row(n: u32) -> Mutation {
    // Every row holds "fox"; one in three "hen"; one in seven "den", twice.
    let mut title = String::from("fox");
    if n % 3 == 0 {
        title.push_str(" hen");
    }
    if n % 7 == 0 {
        title.push_str(" den den");
    }
    Mutation::Upsert {
        collection: name(NOTES),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "title": title, "n": n % 100 })
            .as_object()
            .unwrap()
            .clone(),
        vector: None,
        only_if_version: None,
    }
}

/// `ROWS` notes, some flushed to a segment and the rest in the memtable,
/// with a few deleted from each if `deleting`.
async fn notes(deleting: bool) -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![create(NOTES)]).await.unwrap();
    let rows: Vec<Mutation> = (0..ROWS).map(row).collect();
    for (i, chunk) in rows.chunks(2_000).enumerate() {
        writer.commit(chunk.to_vec()).await.unwrap();
        if i == 2 {
            writer.flush().await.unwrap();
        }
    }
    if !deleting {
        return (cluster, writer);
    }
    let gone: Vec<Mutation> = (0..ROWS)
        .step_by(997)
        .map(|n| Mutation::Delete {
            collection: name(NOTES),
            id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
            only_if_version: None,
        })
        .collect();
    writer.commit(gone).await.unwrap();
    (cluster, writer)
}

fn n_below(n: i64) -> Filter {
    Filter::Lt {
        property: "n".into(),
        value: json!(n),
    }
}

fn searched(
    state: &State,
    text: &str,
    filter: Option<&Filter>,
    limit: usize,
) -> (Vec<(ObjectId, f32)>, Work) {
    let mut work = Work::default();
    let found = state
        .keyword_counted(
            &name(NOTES),
            &Keywords {
                filter,
                ..Keywords::matching(text, limit)
            },
            &mut work,
        )
        .unwrap()
        .into_iter()
        .map(|scored| (scored.object.id, scored.score))
        .collect();
    (found, work)
}

#[tokio::test]
async fn t6b_a_narrow_filter_reads_postings_for_the_rows_it_allows() {
    let (_cluster, writer) = notes(false).await;
    writer.partition().read(|state| {
        for text in ["fox", "fox hen", "den hen"] {
            for (allowed, filter) in [(100, n_below(1)), (1_000, n_below(10))] {
                let (found, work) = searched(state, text, Some(&filter), 10);
                assert!(!found.is_empty(), "{text}: something is found");
                let terms = text.split(' ').count() as u64;
                // A posting per allowed row and term, and as many again
                // for the rows the walk lands on next. With no deletes,
                // no posting is read to weigh the terms.
                let most = 2 * terms * allowed;
                assert!(
                    work.postings <= most,
                    "{text}, {allowed} rows allowed: read {} postings, at most {most}",
                    work.postings
                );
            }
        }
    });
}

#[tokio::test]
async fn t8_a_filtered_search_finds_what_scoring_every_row_and_keeping_the_allowed_finds() {
    let (_cluster, writer) = notes(true).await;
    writer.partition().read(|state| {
        for text in ["fox", "fox hen", "den hen", "den"] {
            for cut in [1, 10, 50, 100] {
                let filter = n_below(cut);
                let allowed = |id: &ObjectId| (id.0.as_u128() % 100) < cut as u128;
                let (everything, _) = searched(state, text, None, ROWS as usize);
                for limit in [1, 10, 200] {
                    let (found, _) = searched(state, text, Some(&filter), limit);
                    let wanted: Vec<(ObjectId, f32)> = everything
                        .iter()
                        .filter(|(id, _)| allowed(id))
                        .take(limit)
                        .cloned()
                        .collect();
                    assert_eq!(found, wanted, "{text}, n < {cut}, limit {limit}");
                }
            }
        }
    });
}
