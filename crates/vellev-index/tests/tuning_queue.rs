//! A merge's own tuning and a rows pass each have a queue of their own. A
//! test binary of its own: the tuning and passes pools are the process's,
//! and other tests building indexes on them would wait in the same queues.

use std::sync::mpsc;
use std::time::Duration;

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;

/// A pool's threads, held until this is dropped.
struct Held(Vec<mpsc::Sender<()>>);

impl Drop for Held {
    fn drop(&mut self) {
        for release in &self.0 {
            let _ = release.send(());
        }
    }
}

/// Hold `pool`'s every thread until what is given back is dropped.
fn held(pool: &'static rayon::ThreadPool) -> Held {
    let (started, busy) = mpsc::channel::<()>();
    let mut releases = Vec::new();
    for _ in 0..pool.current_num_threads() {
        let (release, wait) = mpsc::channel::<()>();
        let started = started.clone();
        pool.spawn(move || {
            started.send(()).unwrap();
            let _ = wait.recv();
        });
        releases.push(release);
    }
    for _ in 0..pool.current_num_threads() {
        busy.recv().unwrap();
    }
    Held(releases)
}

/// `work`'s answer, if it finishes while `held` is held, within minutes:
/// were it queued behind what holds the pool it would never finish, so
/// the limit tells a deadlock, not a slow machine. Else, once `held` is
/// let go and the work done, a failure.
fn finishes<T: Send>(what: &str, held: Held, work: impl FnOnce() -> T + Send) -> T {
    let (done, finished) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let _ = done.send(work());
        });
        let answer = finished.recv_timeout(Duration::from_secs(300));
        drop(held);
        answer.unwrap_or_else(|_| {
            let _ = finished.recv();
            panic!("{what} waited")
        })
    })
}

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

/// A merge's own tuning and a rows pass, which after an upgrade may run
/// segment after segment for minutes, each have a queue of their own: an
/// index builds, tuned, while a pass holds its pool, and a pass makes its
/// curve while tuning holds the tuning pool (review of #147).
#[test]
fn a_merges_tuning_does_not_wait_behind_a_rows_pass() {
    let vectors = grouped(3_000, 16, 12);
    let numbered: Vec<(u32, &[f32])> = vectors
        .iter()
        .enumerate()
        .map(|(row, v)| (row as u32 * 2, v.as_slice()))
        .collect();
    let index = finishes(
        "a build behind a rows pass",
        held(vellev_index::pool::passes()),
        || ClusteredIndex::build(Distance::Cosine, &numbered).unwrap(),
    );
    assert!(index.rows_curve().is_some());
    let curve = finishes(
        "a rows pass behind tuning",
        held(vellev_index::pool::tuning()),
        || index.curve_of_rows(&numbered),
    );
    assert_eq!(curve, index.rows_curve());
}
