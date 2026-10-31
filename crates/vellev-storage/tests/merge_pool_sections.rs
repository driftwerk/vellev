//! Rule 7 (merge note, §9.7, as the reviews of #169 asked): a merge's
//! build, a clustered one's every pass among it, runs its sections on the
//! pool of the slot it holds, and a flush's never do; and two merges at
//! once, through the same code a node runs, each keep to a pool of their
//! own and both finish. A test binary of its own: which thread each
//! section runs on is kept for the whole process, so its tests take turns.

mod common;

use common::*;
use vellev_storage::{merge_slots, sections_ran_on, Holding, MergeBuilds, Merging, SectionRan};

/// Held by each test: what `sections_ran_on` keeps is the whole process's.
static ONE_AT_A_TIME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// `slots` pools of `threads` threads each, as a node makes them.
fn slots(slots: usize, threads: usize) -> Arc<MergeBuilds> {
    let pools = merge_slots(slots, slots * threads, 0);
    MergeBuilds::building_on(pools.into_iter().map(Arc::new).collect())
}

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

/// A writer of `cluster` among `builds`, of two flushes that merge into a
/// clustered segment.
async fn two_flushed(cluster: &Cluster, builds: &Arc<MergeBuilds>) -> Arc<Writer> {
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
    Arc::new(writer)
}

/// Two real merges at once, on pools of `threads` threads each, through
/// the code a node runs: both are held until both are building, so that
/// they overlap; each section of each runs on one pool alone, not the
/// other's; and both commit.
async fn two_merges_at_once(threads: usize) {
    let _alone = ONE_AT_A_TIME.lock().await;
    let (one, other) = (Cluster::new().await, Cluster::new().await);
    let builds = slots(2, threads);
    let (one, other) = (
        two_flushed(&one, &builds).await,
        two_flushed(&other, &builds).await,
    );
    sections_ran_on(true);
    builds.hold(Some(Holding::UntilLetGo));
    let merging = [one.clone(), other.clone()]
        .map(|writer| tokio::spawn(async move { writer.merge().await }));
    let both = tokio::time::timeout(Duration::from_secs(60), async {
        while builds.held() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    both.await.expect("the two merges never built together");
    builds.hold(None);
    for merge in merging {
        let merged = tokio::time::timeout(Duration::from_secs(120), merge);
        let merged = merged.await.expect("a merge waits on the other for good");
        assert!(merged.unwrap().unwrap().is_some(), "a merge did not commit");
    }
    let ran_on = sections_ran_on(false);
    let mut pools_of: std::collections::BTreeMap<u64, std::collections::BTreeSet<String>> =
        Default::default();
    for ran in &ran_on {
        let merge = ran.merge.expect("only merges build here");
        let pool: Vec<&str> = ran.thread.rsplitn(2, '-').collect();
        pools_of
            .entry(merge)
            .or_default()
            .insert(pool[1].to_string());
    }
    let pools: Vec<_> = pools_of.values().collect();
    assert_eq!(pools.len(), 2, "{ran_on:?}");
    assert!(pools.iter().all(|pool| pool.len() == 1), "{pools:?}");
    assert_ne!(pools[0], pools[1], "two builds on one pool");
    for pool in pools.iter().flat_map(|pool| pool.iter()) {
        assert!(pool.starts_with("vellev-merge-"), "{pool}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn two_merges_at_once_on_pools_of_one_thread_each_keep_to_their_own() {
    two_merges_at_once(1).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn two_merges_at_once_on_pools_of_two_threads_each_keep_to_their_own() {
    two_merges_at_once(2).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_clustered_merges_sections_run_on_its_pool_and_a_flushs_never_do() {
    let _alone = ONE_AT_A_TIME.lock().await;
    let cluster = Cluster::new().await;
    let builds = slots(1, 2);
    sections_ran_on(true);
    let writer = two_flushed(&cluster, &builds).await;
    assert!(writer.merge().await.unwrap().is_some());
    let ran_on = sections_ran_on(false);
    let figures = writer.partition().read(|state| state.figures());
    assert_eq!(figures.clustered, 1, "no clustered merge: {figures:?}");

    let on_pool = |ran: &&SectionRan| ran.thread.starts_with("vellev-merge-");
    let (merges, flushes): (Vec<&SectionRan>, Vec<&SectionRan>) =
        ran_on.iter().partition(|ran| ran.merge.is_some());
    assert!(!flushes.is_empty(), "{ran_on:?}");
    assert!(!flushes.iter().any(on_pool), "{ran_on:?}");
    assert!(merges.iter().all(on_pool), "{ran_on:?}");
    for section in [
        "gathering",
        "finishing",
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
    assert!(
        builds.top_level_cpu() > Duration::ZERO,
        "what a build does on its own thread not counted"
    );
}
