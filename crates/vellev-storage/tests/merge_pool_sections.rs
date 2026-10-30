//! Rule 7 (merge note, §9.7, as the reviews of #169 asked): a merge's
//! build, a clustered one's every pass among it, runs its sections on the
//! pool its node gives merge builds, and a flush's never do; and no
//! section is ever begun inside another. A test binary of its own: which
//! thread each section runs on is kept for the whole process.

mod common;

use common::*;
use vellev_storage::{sections_ran_on, MergeBuilds, Merging, SectionRan};

fn note(n: u32) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1_000)),
        properties: json!({ "title": format!("note {n}"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![n as f32, f32::from((n % 97) as u16) - 48.0]),
        only_if_version: None,
    }
}

/// Rows in each of the two flushes merged: together past what a clustered
/// index is built above here, and past the rows below which none ever is.
const FLUSHED: u32 = 2_500;

#[tokio::test(flavor = "multi_thread")]
async fn a_clustered_merges_sections_run_on_its_pool_and_a_flushs_never_do() {
    let cluster = Cluster::new().await;
    let pool = Arc::new(vellev_index::pool::merge_builds(2, 0));
    let builds = MergeBuilds::building_on(1, pool);
    let merging = Merging {
        fanout: 2,
        clustered_above: 4_500,
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
        for at in (0..FLUSHED).step_by(500) {
            let from = round * FLUSHED + at;
            writer
                .commit((from..from + 500).map(note).collect())
                .await
                .unwrap();
        }
        writer.flush().await.unwrap();
    }
    assert!(writer.merge().await.unwrap().is_some());
    let ran_on = sections_ran_on(false);
    let figures = writer.partition().read(|state| state.figures());
    assert_eq!(figures.clustered, 1, "no clustered merge: {figures:?}");

    let on_pool = |ran: &&SectionRan| ran.thread.starts_with("vellev-merge-");
    let (merges, flushes): (Vec<&SectionRan>, Vec<&SectionRan>) =
        ran_on.iter().partition(|ran| ran.merge);
    assert!(!flushes.is_empty(), "{ran_on:?}");
    assert!(!flushes.iter().any(on_pool), "{ran_on:?}");
    assert!(merges.iter().all(on_pool), "{ran_on:?}");
    for section in [
        "tokenizing",
        "indexing",
        "sampling",
        "clustering",
        "coding",
        "summarising",
        "encoding",
        "encoding_vectors",
        "parts",
    ] {
        assert!(
            merges.iter().any(|ran| ran.name == section),
            "{section} never ran among {merges:?}"
        );
    }
    assert_eq!(
        builds.nested_sections(),
        0,
        "a section begun inside another"
    );
    assert!(
        builds.top_level_cpu() > Duration::ZERO,
        "what a build does on its own thread not counted"
    );
}
