//! A search split across threads finds what one search finds.

mod common;

use common::*;
use vellev_storage::{Hybrid, Merging, Query};

const DIMENSIONS: usize = 8;

fn wide(collection: &str) -> Mutation {
    let mut schema = schema(collection);
    schema.embedding.dimensions = DIMENSIONS as u32;
    Mutation::CreateCollection(Box::new(schema))
}

fn row(n: u32, vector: Vec<f32>) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n))),
        properties: json!({ "title": format!("row {n} of group {}", n % 20), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vector),
        only_if_version: None,
    }
}

/// Every hit as a search reads it: object, distance to the bit, rank.
fn hits(found: &[vellev_storage::Hit<'_>]) -> Vec<(ObjectId, u32)> {
    found
        .iter()
        .map(|hit| (hit.object.id, hit.distance.to_bits()))
        .collect()
}

/// A clustered segment, a plain one and the memtable, searched whole and
/// split into 2, 3 and 8 parts: the same hits, nearest first, for vector
/// and hybrid searches, filtered or not.
#[tokio::test]
async fn a_search_split_across_threads_finds_what_one_search_finds() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer.commit(vec![wide("notes")]).await.unwrap();
    let mut random = Random(23);
    let centres: Vec<Vec<f32>> = (0..20).map(|_| random.vector(DIMENSIONS)).collect();
    let vector = |random: &mut Random, n: u32| -> Vec<f32> {
        let jitter = random.vector(DIMENSIONS);
        centres[n as usize % 20]
            .iter()
            .zip(jitter)
            .map(|(c, e)| c + 0.3 * e)
            .collect()
    };
    // 4,800 rows in four flushes merged into one clustered segment, 600
    // more flushed alone, and 300 left in the memtable.
    let mut n = 0u32;
    for (flushes, rows) in [(4, 1_200), (1, 600), (0, 300)] {
        for _ in 0..flushes.max(1) {
            for _ in 0..rows / 300 {
                let writes = (0..300).map(|_| {
                    n += 1;
                    row(n, vector(&mut random, n))
                });
                writer.commit(writes.collect()).await.unwrap();
            }
            if flushes > 0 {
                writer.flush().await.unwrap();
            }
        }
        if flushes == 4 {
            writer.merge().await.unwrap().expect("four segments merge");
        }
    }
    let filter = vellev_storage::Filter::Lt {
        property: "n".into(),
        value: json!(1_000),
    };
    writer.partition().read(|state| {
        let segments: Vec<bool> = state.segments().map(|meta| meta.index.is_some()).collect();
        assert!(
            segments.contains(&true) && segments.contains(&false),
            "{segments:?}"
        );
        for asked in 0..20u32 {
            let query = vector(&mut Random(u64::from(asked) + 99), asked);
            for filtered in [None, Some(&filter)] {
                let one = Query {
                    filter: filtered,
                    oversample: 16,
                    ..Query::nearest(&query, 10)
                };
                let whole = hits(&state.search(&name("notes"), &one).unwrap());
                let text = format!("group {}", asked % 20);
                let hybrid_one = Hybrid {
                    filter: filtered,
                    ..Hybrid::matching(&text, Some(&query), 10)
                };
                let hybrid_whole: Vec<_> = state
                    .hybrid(&name("notes"), &hybrid_one)
                    .unwrap()
                    .iter()
                    .map(|fused| (fused.object.id, fused.score.to_bits()))
                    .collect();
                for split in [2, 3, 8] {
                    let asked_split = Query {
                        split,
                        ..one.clone()
                    };
                    let found = hits(&state.search(&name("notes"), &asked_split).unwrap());
                    assert_eq!(
                        found,
                        whole,
                        "query {asked}, split {split}, filtered {}",
                        filtered.is_some()
                    );
                    let hybrid_split = Hybrid {
                        split,
                        ..hybrid_one.clone()
                    };
                    let fused: Vec<_> = state
                        .hybrid(&name("notes"), &hybrid_split)
                        .unwrap()
                        .iter()
                        .map(|fused| (fused.object.id, fused.score.to_bits()))
                        .collect();
                    assert_eq!(fused, hybrid_whole, "hybrid {asked}, split {split}");
                }
            }
        }
    });
}
