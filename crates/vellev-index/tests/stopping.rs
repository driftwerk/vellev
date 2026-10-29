//! A build asked to stop gives up in whichever long pass it is in (merge
//! note, step 3): a merge given up when its tenant is let go, or its lease
//! lost, frees what it holds within seconds, not at the end of a build of
//! a million rows. A test binary of its own: the tuning pool and its turns
//! are the process's, and other tests' builds would take turns with these.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::stop::{Stage, Stop, Stopped};
use vellev_index::vector::{in_row_order, BinaryIndex, VectorIndex};

/// Vectors in loose groups, one a row.
fn grouped(rows: usize, dimensions: usize, groups: usize) -> Vec<Vec<f32>> {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let centres: Vec<Vec<f32>> = (0..groups)
        .map(|_| (0..dimensions).map(|_| next()).collect())
        .collect();
    (0..rows)
        .map(|row| {
            centres[row % groups]
                .iter()
                .map(|value| value + 0.3 * next())
                .collect()
        })
        .collect()
}

fn numbered(vectors: &[Vec<f32>]) -> Vec<(u32, &[f32])> {
    vectors
        .iter()
        .enumerate()
        .map(|(row, vector)| (row as u32 * 2, vector.as_slice()))
        .collect()
}

/// The passes of a clustered build, in the order it makes them.
const CLUSTERED: [Stage; 5] = [
    Stage::Sampling,
    Stage::Clustering,
    Stage::Coding,
    Stage::Tuning,
    Stage::Ranking,
];

/// The passes of a binary build, in the order it makes them.
const BINARY: [Stage; 2] = [Stage::Sampling, Stage::Coding];

/// Rows enough for several tuning blocks of 4,096.
const ROWS: usize = 9_000;

/// A build asks its stop in every long pass it makes, and as often as the
/// pass goes through rows, or blocks of them for tuning: a pass that never
/// asked would run to its end, however long, once stopped.
#[test]
fn a_build_asks_its_stop_in_each_long_pass() {
    let vectors = grouped(ROWS, 16, 12);
    let rows = numbered(&vectors);

    let stop = Stop::counting();
    let index = ClusteredIndex::build_stopping(Distance::Cosine, &rows, &stop)
        .expect("never stopped")
        .expect("rows to index");
    assert!(index.rows_curve().is_some());
    for stage in CLUSTERED {
        assert!(
            stop.checks(stage) > 0,
            "a clustered build never asked in {stage:?}"
        );
    }
    assert!(stop.checks(Stage::Coding) >= ROWS, "coding asks once a row");
    assert!(
        stop.checks(Stage::Tuning) > ROWS.div_ceil(4_096),
        "tuning asks once its turn is taken, and once a block of rows"
    );

    let stop = Stop::counting();
    BinaryIndex::build_stopping(Distance::Cosine, &rows, &stop)
        .expect("never stopped")
        .expect("rows to index");
    for stage in BINARY {
        assert!(
            stop.checks(stage) > 0,
            "a binary build never asked in {stage:?}"
        );
    }
    assert!(stop.checks(Stage::Coding) >= ROWS, "coding asks once a row");
}

/// A build stopped in any of its passes gives up there, and says so: it
/// makes none of the passes after.
#[test]
fn a_build_stopped_in_any_pass_gives_up_there() {
    let vectors = grouped(ROWS, 16, 12);
    let rows = numbered(&vectors);
    for (at, stage) in CLUSTERED.iter().enumerate() {
        let stop = Stop::at_first(*stage);
        let built = ClusteredIndex::build_stopping(Distance::Cosine, &rows, &stop);
        assert_eq!(
            built.err(),
            Some(Stopped),
            "a clustered build stopped in {stage:?}"
        );
        // Waiting for the turn to tune, which only a build that has to wait
        // asks in, comes before tuning: a build stopped before it never
        // comes to wait either.
        if CLUSTERED[at] != Stage::Tuning && CLUSTERED[at] != Stage::Ranking {
            assert_eq!(
                stop.checks(Stage::Waiting),
                0,
                "stopped in {stage:?}, it went on to wait to tune"
            );
        }
        for later in &CLUSTERED[at + 1..] {
            assert_eq!(
                stop.checks(*later),
                0,
                "stopped in {stage:?}, it went on to {later:?}"
            );
        }
    }
    for (at, stage) in BINARY.iter().enumerate() {
        let stop = Stop::at_first(*stage);
        let built = BinaryIndex::build_stopping(Distance::Cosine, &rows, &stop);
        assert_eq!(
            built.err(),
            Some(Stopped),
            "a binary build stopped in {stage:?}"
        );
        for later in &BINARY[at + 1..] {
            assert_eq!(
                stop.checks(*later),
                0,
                "stopped in {stage:?}, it went on to {later:?}"
            );
        }
    }
}

/// A stop asked for through a clone, as the node asks it of a merge it
/// lets go, is the build's stop: the build gives up.
#[test]
fn a_build_stopped_through_a_clone_gives_up() {
    let vectors = grouped(ROWS, 16, 12);
    let rows = numbered(&vectors);
    let stop = Stop::new();
    assert!(!stop.is_stopped());
    stop.clone().stop();
    assert!(stop.is_stopped());
    let built = ClusteredIndex::build_stopping(Distance::Cosine, &rows, &stop);
    assert_eq!(built.err(), Some(Stopped));
    let built = BinaryIndex::build_stopping(Distance::Cosine, &rows, &stop);
    assert_eq!(built.err(), Some(Stopped));
}

/// A build never stopped is the same as one built without a stop: the
/// same probes, curve and nearest rows.
#[test]
fn a_build_never_stopped_is_as_one_without_a_stop() {
    let vectors = grouped(ROWS, 16, 12);
    let rows = numbered(&vectors);
    let plain = ClusteredIndex::build(Distance::Cosine, &rows).unwrap();
    let stopping = ClusteredIndex::build_stopping(Distance::Cosine, &rows, &Stop::new())
        .unwrap()
        .unwrap();
    assert_eq!(plain.probes(), stopping.probes());
    assert_eq!(plain.clusters(), stopping.clusters());
    assert_eq!(plain.rows_curve(), stopping.rows_curve());
    let binary = BinaryIndex::build(Distance::Cosine, &rows).unwrap();
    let binary_stopping = BinaryIndex::build_stopping(Distance::Cosine, &rows, &Stop::new())
        .unwrap()
        .unwrap();
    for query in vectors.iter().step_by(997) {
        assert_eq!(
            plain.estimates(query, 10, None, &[], &in_row_order),
            stopping.estimates(query, 10, None, &[], &in_row_order)
        );
        assert_eq!(
            binary.estimates(query, 10, None, &[], &in_row_order),
            binary_stopping.estimates(query, 10, None, &[], &in_row_order)
        );
    }
}

/// A build stopped while it waits for its turn on the tuning pool, which
/// one tuning holds at a time, gives up without waiting the turn out: a
/// merge let go behind another tenant's tuning of a million rows would
/// otherwise hold its memory for as long.
#[test]
fn a_stopped_build_waiting_its_turn_to_tune_gives_up() {
    let vectors = grouped(ROWS, 16, 12);
    let rows = numbered(&vectors);
    let (holding, held) = mpsc::channel::<()>();
    let (release, released) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        let holder = scope.spawn(move || {
            vellev_index::pool::tune(&Stop::new(), move || {
                holding.send(()).unwrap();
                let _ = released.recv_timeout(Duration::from_secs(300));
            })
        });
        held.recv().unwrap();
        let stop = Stop::counting();
        let stopper = stop.clone();
        let building =
            scope.spawn(move || ClusteredIndex::build_stopping(Distance::Cosine, &rows, &stop));
        // Until the build waits for its turn, having coded every row.
        let began = Instant::now();
        while stopper.checks(Stage::Waiting) == 0 && began.elapsed() < Duration::from_secs(300) {
            std::thread::park_timeout(Duration::from_millis(10));
        }
        assert!(
            stopper.checks(Stage::Waiting) > 0,
            "the build never waited to tune"
        );
        stopper.stop();
        let asked = Instant::now();
        let (done, finished) = mpsc::channel();
        scope.spawn(move || {
            let _ = done.send(building.join().unwrap());
        });
        let answer = finished.recv_timeout(Duration::from_secs(60));
        let _ = release.send(());
        holder.join().unwrap().expect("the holder never stopped");
        let built = answer.expect("a stopped build waited out another's tuning");
        assert_eq!(built.err(), Some(Stopped));
        assert!(asked.elapsed() < Duration::from_secs(60));
    });
}

/// A turn given up by a stopped waiter is passed over: whoever waits after
/// it tunes once the turn before is done, never waiting on the one that
/// left.
#[test]
fn a_turn_given_up_is_passed_over() {
    let (holding, held) = mpsc::channel::<()>();
    let (release, released) = mpsc::channel::<()>();
    std::thread::scope(|scope| {
        let holder = scope.spawn(move || {
            vellev_index::pool::tune(&Stop::new(), move || {
                holding.send(()).unwrap();
                let _ = released.recv_timeout(Duration::from_secs(300));
            })
        });
        held.recv().unwrap();
        let leaving = Stop::counting();
        let leaver = leaving.clone();
        let left = scope.spawn(move || vellev_index::pool::tune(&leaving, || ()));
        let began = Instant::now();
        while leaver.checks(Stage::Waiting) == 0 && began.elapsed() < Duration::from_secs(60) {
            std::thread::park_timeout(Duration::from_millis(5));
        }
        let waiting = Stop::counting();
        let waiter = waiting.clone();
        let after = scope.spawn(move || vellev_index::pool::tune(&waiting, || "tuned"));
        while waiter.checks(Stage::Waiting) == 0 && began.elapsed() < Duration::from_secs(60) {
            std::thread::park_timeout(Duration::from_millis(5));
        }
        leaver.stop();
        assert_eq!(left.join().unwrap(), Err(Stopped));
        release.send(()).unwrap();
        holder.join().unwrap().expect("the holder never stopped");
        let (done, finished) = mpsc::channel();
        scope.spawn(move || {
            let _ = done.send(after.join().unwrap());
        });
        let answer = finished.recv_timeout(Duration::from_secs(60));
        // Let a waiter stuck behind the turn that left go, for the scope.
        waiter.stop();
        assert_eq!(
            answer.expect("the turn after one given up never came"),
            Ok("tuned")
        );
    });
}
