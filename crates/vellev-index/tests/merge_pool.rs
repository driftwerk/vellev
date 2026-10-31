//! Pools for merges to build on (merge note, §9.2): a build given a pool
//! runs its parallel passes there and nothing else, its serial work and
//! its turn to tune on the thread that called it, so that a pool's thread
//! never waits for anything but a pass. Two real merges through the
//! storage crate's pools are told in its `merge_pool_sections`. A test
//! binary of its own: the tuning pool is the process's. Its threads'
//! priority is told in `merge_pool_nice`, alone, as other pools here
//! would be counted.

use std::sync::Mutex;

use rayon::ThreadPool;
use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::pool::{self, Passes};
use vellev_index::stop::Stop;
use vellev_index::vector::BinaryIndex;

/// Passes run on a pool, each kept: its name, and the thread it ran on.
#[derive(Clone, Copy)]
struct OnPool<'a> {
    pool: &'a ThreadPool,
    ran: &'a Mutex<Vec<(&'static str, String)>>,
}

impl Passes for OnPool<'_> {
    fn pass<R: Send>(self, name: &'static str, pass: impl FnOnce() -> R + Send) -> R {
        assert_eq!(
            rayon::current_thread_index(),
            None,
            "{name} asked for from a pool's thread"
        );
        self.pool.install(|| {
            let thread = std::thread::current().name().unwrap_or("").to_string();
            self.ran.lock().unwrap().push((name, thread));
            pass()
        })
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
    let merges = pool::merge_slots(1, 2, 0).pop().unwrap();
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

/// A merge pool's threads are named for its slot, each pool as many as
/// its share of the threads, and any left over from whole shares unused.
#[test]
fn merge_slots_are_equal_shares_each_named_for_its_slot() {
    let slots = pool::merge_slots(2, 5, 0);
    assert_eq!(slots.len(), 2);
    for (slot, merges) in slots.iter().enumerate() {
        assert_eq!(merges.current_num_threads(), 2);
        let named = merges.install(|| std::thread::current().name().map(str::to_string));
        let named = named.unwrap();
        assert!(
            named.starts_with(&format!("vellev-merge-{slot}-")),
            "{named}"
        );
    }
}
