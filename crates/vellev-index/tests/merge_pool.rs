//! A pool for merges to build on (merge note, §9.2): a build given a pool
//! runs its parallel passes there and nothing else, its serial work and
//! its turn to tune on the thread that called it, so that a pool's thread
//! never waits for anything but a pass. A test binary of its own: the
//! tuning pool is the process's. Its threads' priority is told in
//! `merge_pool_nice`, alone, as other pools here would be counted.

use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use rayon::ThreadPool;
use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::pool::{self, Passes};
use vellev_index::stop::{Stop, Stopped};
use vellev_index::vector::BinaryIndex;

/// Passes run on a pool, each kept: its name, and the thread it ran on.
#[derive(Clone, Copy)]
struct OnPool<'a> {
    pool: &'a ThreadPool,
    ran: &'a Mutex<Vec<(&'static str, String)>>,
}

impl Passes for OnPool<'_> {
    fn pass<R: Send>(
        self,
        name: &'static str,
        pass: impl FnOnce() -> R + Send,
    ) -> Result<R, Stopped> {
        assert_eq!(
            rayon::current_thread_index(),
            None,
            "{name} asked for from a pool's thread"
        );
        Ok(self.pool.install(|| {
            let thread = std::thread::current().name().unwrap_or("").to_string();
            self.ran.lock().unwrap().push((name, thread));
            pass()
        }))
    }
}

/// Rows of `dimensions` spread over enough of the space to cluster.
fn rows(count: u32, dimensions: usize) -> Vec<(u32, Vec<f32>)> {
    (0..count)
        .map(|row| {
            let vector = (0..dimensions)
                .map(|at| (((row as usize + 1) * (at + 3) % 97) as f32) - 48.0)
                .collect();
            (row, vector)
        })
        .collect()
}

fn borrowed(rows: &[(u32, Vec<f32>)]) -> Vec<(u32, &[f32])> {
    rows.iter().map(|(row, v)| (*row, v.as_slice())).collect()
}

/// A clustered build given a pool runs every parallel pass there, asked
/// for from the thread that called it, and comes out as the build without
/// one does. So does a binary one.
#[test]
fn a_build_given_a_pool_runs_every_pass_there_and_comes_out_the_same() {
    let merges = pool::merge_builds(2, 0);
    let rows = rows(6_000, 32);
    let vectors = borrowed(&rows);
    let stop = Stop::new();
    let ran = Mutex::new(Vec::new());
    let on = OnPool {
        pool: &merges,
        ran: &ran,
    };

    let built = ClusteredIndex::build_stopping_on(Distance::Cosine, &vectors, &stop, on);
    let plain = ClusteredIndex::build_stopping(Distance::Cosine, &vectors, &stop);
    let bytes = |index: &ClusteredIndex| rmp_serde::to_vec(index).unwrap();
    assert_eq!(
        bytes(&built.unwrap().unwrap()),
        bytes(&plain.unwrap().unwrap())
    );
    let clustered = std::mem::take(&mut *ran.lock().unwrap());
    let names: Vec<&str> = clustered.iter().map(|(name, _)| *name).collect();
    for name in ["sampling", "clustering", "coding"] {
        assert!(names.contains(&name), "{name} not among {names:?}");
    }
    assert!(
        clustered
            .iter()
            .all(|(_, t)| t.starts_with("vellev-merge-")),
        "{clustered:?}"
    );

    let built = BinaryIndex::build_stopping_on(Distance::Cosine, &vectors, &stop, on);
    let plain = BinaryIndex::build_stopping(Distance::Cosine, &vectors, &stop);
    assert_eq!(
        format!("{:?}", built.unwrap().unwrap()),
        format!("{:?}", plain.unwrap().unwrap())
    );
    let binary = std::mem::take(&mut *ran.lock().unwrap());
    let names: Vec<&str> = binary.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, vec!["sampling", "coding"]);
    assert!(
        binary.iter().all(|(_, t)| t.starts_with("vellev-merge-")),
        "{binary:?}"
    );
}

/// Two clustered builds on a merge pool of one thread, asked for together,
/// both finish: each waits for its turn to tune on its own thread, so
/// neither parks the pool's one thread where the other needs it.
#[test]
fn two_builds_on_a_merge_pool_of_one_thread_both_finish() {
    // Threads of their own, not scoped: were the builds to wait on each
    // other for good, the test fails at its limit rather than hanging on
    // them.
    let merges = Arc::new(pool::merge_builds(1, 0));
    let (done, finished) = mpsc::channel();
    for build in 0..2 {
        let (merges, done) = (merges.clone(), done.clone());
        std::thread::spawn(move || {
            let rows = rows(5_000 + build, 32);
            let ran = Mutex::new(Vec::new());
            let on = OnPool {
                pool: &merges,
                ran: &ran,
            };
            let vectors = borrowed(&rows);
            let built =
                ClusteredIndex::build_stopping_on(Distance::Cosine, &vectors, &Stop::new(), on);
            done.send(built.unwrap().is_some()).unwrap();
        });
    }
    for _ in 0..2 {
        let one = finished.recv_timeout(Duration::from_secs(60));
        assert!(one.expect("both built within 60 s: they wait on each other"));
    }
}

/// A merge pool's threads are named for it, as many as it was made with.
#[test]
fn a_merge_pools_threads_are_named_for_it() {
    let merges = pool::merge_builds(2, 0);
    assert_eq!(merges.current_num_threads(), 2);
    let named = merges.install(|| std::thread::current().name().map(str::to_string));
    assert!(named.unwrap().starts_with("vellev-merge-"));
}
