//! How much of a collection a search scores exactly, and what it must
//! still find for it: the rules the rescore budget (burst plan F1) and
//! top-k selection (F3) are held to, written before either.
//!
//! - T1: a hybrid search with the default oversample scores no more rows
//!   exactly than its own result limit times that oversample, and the
//!   fusion pool, call for; never the fusion pool times the oversample.
//! - T2: whatever the budget, the fused results are those an exact vector
//!   leg gives, across filters of every selectivity and across layouts of
//!   one segment, several of uneven size, and segments with a memtable.
//! - T5: a search's results are exactly the nearest rows by distance, ties
//!   in object id order, whatever its limit, across layers.

mod common;

use common::*;
use vellev_storage::{
    default_oversample, Hybrid, Query, Work, DEFAULT_CANDIDATES, DEFAULT_OVERSAMPLE,
};

const NOTES: &str = "notes";
const DIMENSIONS: usize = 128;

/// The most rows a default hybrid search for `limit` results may score
/// exactly: its limit times the default oversample, or the fusion pool if
/// that is larger.
fn default_budget(limit: usize) -> u64 {
    (limit * DEFAULT_OVERSAMPLE).max(DEFAULT_CANDIDATES) as u64
}

fn wide() -> Mutation {
    let mut schema = schema(NOTES);
    schema.embedding.dimensions = DIMENSIONS as u32;
    Mutation::CreateCollection(Box::new(schema))
}

fn row(n: u32, vector: Vec<f32>) -> Mutation {
    let words = ["fox", "hen", "den", "owl", "elk"];
    let title = format!("{} {}", words[n as usize % 5], words[n as usize / 5 % 5]);
    Mutation::Upsert {
        collection: name(NOTES),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        // `n % 100` lets a filter allow a hundredth, a tenth, or nearly all.
        properties: json!({ "title": title, "n": n % 100 })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vector),
        only_if_version: None,
    }
}

/// `rows` vectors in groups of uneven size around thirty centres, and
/// some near-duplicates, as real embeddings have.
fn vectors(rows: u32, seed: u64) -> Vec<Vec<f32>> {
    let mut random = Random(seed);
    let centres: Vec<Vec<f32>> = (0..30).map(|_| random.vector(DIMENSIONS)).collect();
    (0..rows)
        .map(|n| {
            // Centre i takes a share proportional to 1/(i+1).
            let pick = (random.unit() + 0.5).powi(3);
            let centre = &centres[(pick * 30.0) as usize % 30];
            let spread = if n % 50 == 0 { 0.01 } else { 0.35 };
            let jitter = random.vector(DIMENSIONS);
            centre
                .iter()
                .zip(jitter)
                .map(|(c, e)| c + spread * e)
                .collect()
        })
        .collect()
}

/// Written in batches of `sizes`, each flushed to a segment of its own
/// but the last when `memtable` is set.
async fn laid_out(vectors: &[Vec<f32>], sizes: &[usize], memtable: bool) -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer.commit(vec![wide()]).await.unwrap();
    let mut at = 0;
    for (i, size) in sizes.iter().enumerate() {
        let batch: Vec<Mutation> = (at..at + size)
            .map(|n| row(n as u32, vectors[n].clone()))
            .collect();
        for chunk in batch.chunks(2_000) {
            writer.commit(chunk.to_vec()).await.unwrap();
        }
        at += size;
        if !(memtable && i == sizes.len() - 1) {
            writer.flush().await.unwrap();
        }
    }
    (cluster, writer)
}

fn n_below(n: i64) -> Filter {
    Filter::Lt {
        property: "n".into(),
        value: json!(n),
    }
}

fn hybrid_work(state: &State, query: &Hybrid<'_>) -> Work {
    let mut work = Work::default();
    state
        .hybrid_counted(&name(NOTES), query, &mut work)
        .unwrap();
    work
}

#[tokio::test]
async fn t1_a_default_hybrid_search_scores_at_most_its_default_budget() {
    let all = vectors(12_000, 3);
    let (_cluster, writer) = laid_out(&all, &[12_000], false).await;
    writer.partition().read(|state| {
        for limit in [1, 10, 50] {
            let query = Hybrid::matching("fox", Some(&all[7]), limit);
            let work = hybrid_work(state, &query);
            let scored = work.exact_memtable + work.exact_segments;
            assert!(
                scored <= default_budget(limit),
                "limit {limit}: scored {scored} rows exactly, the budget is {}",
                default_budget(limit)
            );
        }
    });
}

#[tokio::test]
async fn an_explicit_oversample_is_scored_as_asked() {
    let all = vectors(12_000, 5);
    let (_cluster, writer) = laid_out(&all, &[12_000], false).await;
    writer.partition().read(|state| {
        let query = Hybrid {
            oversample: 1_000,
            ..Hybrid::matching("fox", Some(&all[3]), 10)
        };
        let work = hybrid_work(state, &query);
        assert!(
            work.exact_segments >= 10_000,
            "an explicit oversample of 1,000 at limit 10 scores 10,000 rows: {work:?}"
        );
    });
}

#[tokio::test]
async fn t1_the_budget_holds_per_shortlisting_segment_and_small_layers_are_scored_whole() {
    let all = vectors(12_000, 13);
    // Two segments that shortlist, one scored whole, and a memtable.
    let (_cluster, writer) = laid_out(&all, &[6_500, 4_200, 800, 500], true).await;
    writer.partition().read(|state| {
        for limit in [10, 50] {
            let query = Hybrid {
                oversample: default_oversample(limit),
                ..Hybrid::matching("fox", Some(&all[7]), limit)
            };
            let budget = (limit * default_oversample(limit)).max(DEFAULT_CANDIDATES) as u64;
            let work = hybrid_work(state, &query);
            assert!(
                work.exact_segments <= 2 * budget + 800,
                "limit {limit}: {} segment rows scored, the bound is {}",
                work.exact_segments,
                2 * budget + 800
            );
            assert!(work.exact_memtable <= 500, "{work:?}");
        }
    });
}

#[tokio::test]
async fn a_small_explicit_oversample_still_scores_the_whole_fusion_pool() {
    let all = vectors(12_000, 5);
    let (_cluster, writer) = laid_out(&all, &[12_000], false).await;
    writer.partition().read(|state| {
        let query = Hybrid {
            oversample: 1,
            ..Hybrid::matching("fox", Some(&all[3]), 10)
        };
        let work = hybrid_work(state, &query);
        assert!(
            work.exact_segments >= DEFAULT_CANDIDATES as u64,
            "10 x 1 asked for, but the vector leg offers {DEFAULT_CANDIDATES} for fusing: {work:?}"
        );
    });
}

/// How alike two orderings of results are: the share of `exact`'s ids
/// that `found` holds, and whether both start with the same one.
fn overlap(found: &[ObjectId], exact: &[ObjectId]) -> (f64, bool) {
    let shared = exact.iter().filter(|id| found.contains(id)).count();
    let share = match exact.len() {
        0 => 1.0,
        len => shared as f64 / len as f64,
    };
    (share, found.first() == exact.first())
}

/// T2 over one layout: for queries near stored rows, under no filter, a
/// tenth and a hundredth, the default budget's fused top ten, and top
/// fifty where the node's 1,024-row cap lowers the oversample, against an
/// exact vector leg's.
fn holds_recall(state: &State, all: &[Vec<f32>], layout: &str) {
    for limit in [10, 50] {
        holds_recall_at(state, all, layout, limit);
    }
}

/// As [`holds_recall`], for `limit` results and the oversample the node
/// gives a search that names none.
fn holds_recall_at(state: &State, all: &[Vec<f32>], layout: &str, limit: usize) {
    let filters = [None, Some(n_below(10)), Some(n_below(1))];
    for filter in &filters {
        let (mut shares, mut firsts, mut asked) = (0.0, 0, 0);
        for (q, near) in all.iter().step_by(all.len() / 60).enumerate() {
            let mut random = Random(1_000 + q as u64);
            let query: Vec<f32> = near
                .iter()
                .zip(random.vector(DIMENSIONS))
                .map(|(v, e)| v + 0.05 * e)
                .collect();
            let text = ["fox", "hen den", "owl"][q % 3];
            let default = Hybrid {
                filter: filter.as_ref(),
                oversample: default_oversample(limit),
                ..Hybrid::matching(text, Some(&query), limit)
            };
            let exact = Hybrid {
                oversample: usize::MAX,
                ..default.clone()
            };
            let ids = |query: &Hybrid<'_>| -> Vec<ObjectId> {
                state
                    .hybrid(&name(NOTES), query)
                    .unwrap()
                    .iter()
                    .map(|fused| fused.object.id)
                    .collect()
            };
            let (share, first) = overlap(&ids(&default), &ids(&exact));
            shares += share;
            firsts += usize::from(first);
            asked += 1;
        }
        let mean = shares / asked as f64;
        let same_first = firsts as f64 / asked as f64;
        assert!(
            mean >= 0.99 && same_first >= 0.99,
            "{layout}, limit {limit}, filter {filter:?}: mean overlap {mean:.3}, same top result {same_first:.3}"
        );
    }
}

#[tokio::test]
async fn t2_the_default_budget_finds_what_an_exact_vector_leg_finds_in_one_segment() {
    let all = vectors(12_000, 11);
    let (_cluster, writer) = laid_out(&all, &[12_000], false).await;
    writer
        .partition()
        .read(|state| holds_recall(state, &all, "one segment"));
}

#[tokio::test]
async fn t2_the_default_budget_finds_what_an_exact_vector_leg_finds_across_uneven_segments() {
    let all = vectors(12_000, 13);
    // A large segment, one just above the shortlisting size, one small
    // enough to be scored whole, and a memtable.
    let (_cluster, writer) = laid_out(&all, &[6_500, 4_200, 800, 500], true).await;
    writer
        .partition()
        .read(|state| holds_recall(state, &all, "uneven segments and a memtable"));
}

/// The exact nearest `limit` of every live row of the collection, by
/// distance then id, worked out here apart from the search.
fn nearest_by_hand(state: &State, query: &[f32], limit: usize) -> Vec<(ObjectId, f32)> {
    let mut all: Vec<(ObjectId, f32)> = state
        .list(&name(NOTES), None, None, usize::MAX)
        .unwrap()
        .into_iter()
        .filter_map(|live| {
            let vector = live.vector?;
            let distance = vellev_index::vector::distance(Distance::Cosine, query, vector);
            Some((live.id(), distance))
        })
        .collect();
    all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    all.truncate(limit);
    all
}

#[tokio::test]
async fn t5_a_search_returns_exactly_the_nearest_rows_ties_in_id_order() {
    let mut all = vectors(1_500, 17);
    // Rows that tie: the same vector under several ids, some flushed and
    // some still in the memtable.
    for n in (0..1_500).step_by(7) {
        all[n] = all[0].clone();
    }
    let (_cluster, writer) = laid_out(&all, &[900, 600], true).await;
    writer.partition().read(|state| {
        for query in [&all[0], &all[1], &all[700]] {
            for limit in [0, 1, 2, 10, 150, 1_499, 1_500, 5_000] {
                let found: Vec<(ObjectId, f32)> = state
                    .search(&name(NOTES), &Query::nearest(query, limit))
                    .unwrap()
                    .into_iter()
                    .map(|hit| (hit.object.id, hit.distance))
                    .collect();
                assert_eq!(found, nearest_by_hand(state, query, limit), "limit {limit}");
            }
        }
    });
}
