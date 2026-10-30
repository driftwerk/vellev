//! Rule 7 (merge note, §9.7, as the review of #169 asked): a merge's
//! build runs its parallel sections on the pool its node gives merge
//! builds, and a flush's never do; and no build is ever begun on a thread
//! already in one. A test binary of its own: which thread each section
//! runs on is kept for the whole process.

mod common;

use common::*;
use vellev_storage::{sections_ran_on, MergeBuilds, Merging};

fn note(n: u32) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1_000)),
        properties: json!({ "title": format!("note {n}"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![n as f32, 1.0]),
        only_if_version: None,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_merges_sections_run_on_its_pool_and_a_flushs_never_do() {
    let cluster = Cluster::new().await;
    let pool = Arc::new(vellev_index::pool::merge_builds(2, 0));
    let builds = MergeBuilds::building_on(1, pool);
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = cluster
        .writer()
        .await
        .merging_with(merging)
        .merging_among(builds.clone());
    writer.commit(vec![create("notes")]).await.unwrap();
    sections_ran_on(true);
    for round in 0..2 {
        writer
            .commit((round * 10..round * 10 + 10).map(note).collect())
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    assert!(writer.merge().await.unwrap().is_some());
    let ran_on = sections_ran_on(false);
    let on_pool = |thread: &String| thread.starts_with("vellev-merge-");
    let (merges, flushes): (Vec<_>, Vec<_>) = ran_on.iter().partition(|(merge, _)| *merge);
    assert!(!merges.is_empty() && !flushes.is_empty(), "{ran_on:?}");
    assert!(
        merges.iter().all(|(_, thread)| on_pool(thread)),
        "{ran_on:?}"
    );
    assert!(
        !flushes.iter().any(|(_, thread)| on_pool(thread)),
        "{ran_on:?}"
    );
    assert_eq!(builds.nested(), 0, "a build begun inside another");
}
