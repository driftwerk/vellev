//! What a writer times of its background work (merge note, §9.4 and
//! §9.7): how long rows wait once their flush is due, however the
//! checkpoint that takes them comes; spells at the backlog; the building
//! lock, waited for and held, by whoever holds it; a merge's phases, its
//! gather apart from its build; and how each merge ended. Which pool a
//! build's sections run on is told in `merge_pool_sections`.

mod common;

use std::time::Instant;

use common::*;
use vellev_storage::timings::{Event, Holder, MergeEnd, MergePhase, Timed, Timings};
use vellev_storage::{Holding, MergeBuilds, Merging};

/// Rows a flush is due at here.
const DUE: usize = 50;

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

/// `count` notes from `from` on.
fn notes(from: u32, count: usize) -> Vec<Mutation> {
    (from..from + count as u32).map(note).collect()
}

/// A writer of the cluster's partition timed into `timings`, which keeps
/// every duration, its flushes due at [`DUE`] rows and its backlog at
/// `backlog`, merging two segments at a time among `builds`, with a
/// collection to write to.
async fn timed_writer(
    cluster: &Cluster,
    timings: &Arc<Timings>,
    backlog: usize,
    builds: Arc<MergeBuilds>,
) -> Arc<Writer> {
    timings.keep_events(true);
    let merging = Merging {
        fanout: 2,
        ..Merging::default()
    };
    let writer = cluster
        .writer()
        .await
        .merging_with(merging)
        .merging_among(builds)
        .measured_by(timings.clone())
        .flushing_due_at(DUE, backlog);
    writer.commit(vec![create("notes")]).await.unwrap();
    Arc::new(writer)
}

/// The durations kept of `timed`, oldest first.
fn kept(events: &[Event], timed: Timed) -> Vec<Duration> {
    events
        .iter()
        .filter(|event| event.timed == timed)
        .map(|event| event.took)
        .collect()
}

/// Wait, up to ten seconds, until `holds`.
async fn until(what: &str, holds: impl Fn() -> bool) {
    let waited = tokio::time::timeout(Duration::from_secs(10), async {
        while !holds() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    waited.await.unwrap_or_else(|_| panic!("{what}"));
}

/// How long a test holds work where it makes others wait.
const HELD: Duration = Duration::from_millis(1_000);

/// Rule 1. Rows that fall due while their tenant's merge builds wait for
/// the merge, which flushes them along with its commit, as a tenant's
/// maintenance does: their wait is timed from when they fell due to that
/// commit, though no flush of their own ever took them.
#[tokio::test(flavor = "multi_thread")]
async fn rows_due_while_a_merge_builds_are_timed_to_the_merge_that_takes_them() {
    let cluster = Cluster::new().await;
    let timings = Arc::new(Timings::default());
    let builds = MergeBuilds::new(1);
    let writer = timed_writer(&cluster, &timings, 0, builds.clone()).await;
    for round in 0..2 {
        writer.commit(notes(round * 10, 10)).await.unwrap();
        writer.flush().await.unwrap();
    }
    timings.drain_events();
    builds.hold(Some(Holding::UntilLetGo));
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    until("the merge's build is held", || builds.held() == 1).await;
    let due = Instant::now();
    writer.commit(notes(100, DUE)).await.unwrap();
    tokio::time::sleep(HELD).await;
    builds.hold(None);
    assert!(
        merging.await.unwrap().unwrap().is_some(),
        "the merge commits"
    );
    let waited = due.elapsed();
    assert_eq!(writer.partition().read(State::unflushed), 0);
    let flushed = kept(&timings.drain_events(), Timed::FlushDue);
    assert_eq!(flushed.len(), 1, "one wait timed, by the merge's commit");
    assert!(
        flushed[0] >= HELD && flushed[0] <= waited,
        "timed {:?}, held {HELD:?}, waited at most {waited:?}",
        flushed[0]
    );
}

/// Rule 2. Rows written while a flush builds fall due once those it did
/// not take come to what a flush is due at, and are timed from then, not
/// from when the flush before began or was committed.
#[tokio::test(flavor = "multi_thread")]
async fn rows_due_while_a_flush_builds_are_timed_from_when_they_fell_due() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment");
    let timings = Arc::new(Timings::default());
    let writer = timed_writer(&cluster, &timings, 0, MergeBuilds::new(1)).await;
    writer.commit(notes(0, DUE)).await.unwrap();
    paused.armed.store(true, Ordering::SeqCst);
    let first = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(seen.recv().await, Some("stored"));
    let due = Instant::now();
    writer.commit(notes(1_000, DUE)).await.unwrap();
    tokio::time::sleep(HELD).await;
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_one();
    first.await.unwrap().unwrap();
    let after_first = Instant::now();
    tokio::time::sleep(HELD / 2).await;
    writer.flush().await.unwrap();
    let waited = due.elapsed();
    let flushed = kept(&timings.drain_events(), Timed::FlushDue);
    assert_eq!(flushed.len(), 2, "both flushes timed: {flushed:?}");
    let second = flushed[1];
    assert!(
        second >= HELD + HELD / 2 && second <= waited,
        "the rows due mid-flush timed {second:?}: due {waited:?} ago, the flush before \
         committed {:?} ago",
        after_first.elapsed()
    );
}

/// A flush that finds nothing to flush builds nothing, so its hold of the
/// building lock is no flush's build and is not timed as one: the flush
/// wait mark sets the wait against a flush's own build, which a run of
/// empty holds would make look shorter than it is (review of #169). Its
/// wait for the lock is real, and is timed as any flush's.
#[tokio::test(flavor = "multi_thread")]
async fn a_flush_with_nothing_to_flush_is_no_hold_of_a_flush() {
    let cluster = Cluster::new().await;
    let timings = Arc::new(Timings::default());
    let writer = timed_writer(&cluster, &timings, 0, MergeBuilds::new(1)).await;
    writer.commit(notes(0, 3)).await.unwrap();
    writer.flush().await.unwrap();
    timings.drain_events();
    // Nothing written since: nothing to flush.
    let (_, times) = writer.flush_timed().await.unwrap();
    assert_eq!(times.held, None, "{times:?}");
    let events = timings.drain_events();
    assert_eq!(kept(&events, Timed::BuildingWait(Holder::Flush)).len(), 1);
    assert_eq!(
        kept(&events, Timed::BuildingHeld(Holder::Flush)),
        Vec::<Duration>::new(),
        "an empty flush timed as a flush's hold"
    );
    // One that flushes is held, and timed so.
    writer.commit(notes(10, 3)).await.unwrap();
    let (_, times) = writer.flush_timed().await.unwrap();
    assert!(times.held.is_some(), "{times:?}");
    let events = timings.drain_events();
    assert_eq!(kept(&events, Timed::BuildingWait(Holder::Flush)).len(), 1);
    assert_eq!(kept(&events, Timed::BuildingHeld(Holder::Flush)).len(), 1);
}

/// Rule 3. A spell at the backlog runs from the rows unflushed coming to
/// it until the commit that takes the tenant below it: not from the flush
/// that does so beginning, which comes a while later, nor to it.
#[tokio::test(flavor = "multi_thread")]
async fn a_spell_at_the_backlog_runs_to_the_commit_that_ends_it() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment");
    let timings = Arc::new(Timings::default());
    let writer = timed_writer(&cluster, &timings, 2 * DUE, MergeBuilds::new(1)).await;
    let capped = Instant::now();
    writer.commit(notes(0, 2 * DUE)).await.unwrap();
    tokio::time::sleep(HELD).await;
    paused.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(seen.recv().await, Some("stored"));
    tokio::time::sleep(HELD).await;
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_one();
    flushing.await.unwrap().unwrap();
    let spell = capped.elapsed();
    let spells = kept(&timings.drain_events(), Timed::Backlog);
    assert_eq!(spells.len(), 1, "one spell: {spells:?}");
    assert!(
        spells[0] >= 2 * HELD && spells[0] <= spell,
        "the spell timed {:?}: at the backlog {HELD:?} before the flush began, which was \
         held {HELD:?}, and at most {spell:?} in all",
        spells[0]
    );
}

/// Rule 4. Each holder of the building lock is timed under its own name:
/// a clean that waits for a flush held mid-upload shows its wait as a
/// clean's, and the flush's hold as a flush's.
#[tokio::test(flavor = "multi_thread")]
async fn each_holder_of_the_building_lock_is_timed_under_its_own_name() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment");
    let timings = Arc::new(Timings::default());
    let writer = timed_writer(&cluster, &timings, 0, MergeBuilds::new(1)).await;
    writer.commit(notes(0, 10)).await.unwrap();
    timings.drain_events();
    paused.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    assert_eq!(seen.recv().await, Some("stored"));
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::from_secs(3_600)).await }
    });
    tokio::time::sleep(HELD).await;
    paused.armed.store(false, Ordering::SeqCst);
    paused.release.notify_one();
    flushing.await.unwrap().unwrap();
    cleaning.await.unwrap().unwrap();
    let events = timings.drain_events();
    let held = kept(&events, Timed::BuildingHeld(Holder::Flush));
    let waited = kept(&events, Timed::BuildingWait(Holder::Clean));
    assert_eq!(held.len(), 1, "{events:?}");
    assert_eq!(waited.len(), 1, "{events:?}");
    assert!(held[0] >= HELD, "the flush held {:?}", held[0]);
    assert!(
        waited[0] >= HELD * 9 / 10,
        "the clean waited {:?} for the flush",
        waited[0]
    );
    assert!(
        kept(&events, Timed::BuildingWait(Holder::Flush))
            .iter()
            .all(|wait| *wait < HELD / 2),
        "the clean's wait was put down to a flush: {events:?}"
    );
    assert_eq!(kept(&events, Timed::BuildingHeld(Holder::Clean)).len(), 1);
}

/// Rule 5. A merge's gather, under the building lock as it plans, is
/// timed apart from its build, which holds no lock: with the build held
/// for a while, the plan's hold and the gather are each short, and the
/// build takes the while.
#[tokio::test(flavor = "multi_thread")]
async fn a_merges_gather_is_timed_apart_from_its_build() {
    let cluster = Cluster::new().await;
    let timings = Arc::new(Timings::default());
    let builds = MergeBuilds::new(1);
    let writer = timed_writer(&cluster, &timings, 0, builds.clone()).await;
    for round in 0..2 {
        writer.commit(notes(round * 10, 10)).await.unwrap();
        writer.flush().await.unwrap();
    }
    timings.drain_events();
    builds.hold(Some(Holding::UntilLetGo));
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    until("the merge's build is held", || builds.held() == 1).await;
    tokio::time::sleep(HELD).await;
    builds.hold(None);
    assert!(merging.await.unwrap().unwrap().is_some());
    let events = timings.drain_events();
    let plan = kept(&events, Timed::BuildingHeld(Holder::MergePlan));
    let gather = kept(&events, Timed::Merge(MergePhase::Gather));
    let build = kept(&events, Timed::Merge(MergePhase::Build));
    let commit = kept(&events, Timed::BuildingHeld(Holder::MergeCommit));
    assert_eq!(
        (plan.len(), gather.len(), build.len()),
        (1, 1, 1),
        "{events:?}"
    );
    assert_eq!(commit.len(), 1, "{events:?}");
    assert!(plan[0] < HELD / 2, "the plan held the lock {:?}", plan[0]);
    assert!(
        gather[0] <= plan[0],
        "the gather {:?} is within the plan",
        gather[0]
    );
    assert!(build[0] >= HELD, "the build took {:?}", build[0]);
    for phase in [MergePhase::Turn, MergePhase::Upload, MergePhase::Load] {
        assert_eq!(
            kept(&events, Timed::Merge(phase)).len(),
            1,
            "{phase:?}: {events:?}"
        );
    }
}

/// Rule 6. Each merge that takes a turn is counted by how it ended: one
/// planned at another epoch as stale, one stopped as stopped, and one
/// that went through as committed.
#[tokio::test(flavor = "multi_thread")]
async fn each_merge_is_counted_by_how_it_ended() {
    let cluster = Cluster::new().await;
    let timings = Arc::new(Timings::default());
    let writer = timed_writer(&cluster, &timings, 0, MergeBuilds::new(1)).await;
    for round in 0..2 {
        writer.commit(notes(round * 10, 10)).await.unwrap();
        writer.flush().await.unwrap();
    }
    let now = writer.epoch();
    assert_eq!(writer.merge_planned_at(Some(now + 1)).await.unwrap(), None);
    writer.stop_merges_at(Some(vellev_storage::MergeStage::Gathering));
    assert_eq!(writer.merge().await.unwrap(), None);
    writer.stop_merges_at(None);
    assert!(writer.merge().await.unwrap().is_some());
    let counted: Vec<(MergeEnd, u64)> = MergeEnd::ALL
        .iter()
        .map(|end| (*end, timings.merges_ended(*end)))
        .filter(|(_, count)| *count > 0)
        .collect();
    assert_eq!(
        counted,
        vec![
            (MergeEnd::Committed, 1),
            (MergeEnd::Stale, 1),
            (MergeEnd::Stopped, 1)
        ]
    );
}
