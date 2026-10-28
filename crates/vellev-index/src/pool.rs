//! The threads a search split into parts runs its parts on.
//!
//! Reads have a pool of their own, apart from rayon's global pool, which
//! builds indexes for flushes and merges: a split read's parts queued
//! behind a build's would wait out the build, and the read holds its
//! partition's read lock meanwhile, which holds back a commit waiting for
//! the lock and every read of the tenant after it (review of #141). With
//! both pools busy at once the threads outnumber the cores for a while;
//! a read splits only over the threads other reads leave idle, so that
//! lasts no longer than a build overlaps a lull in reads.

use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// The read pool: as many threads as rayon's global pool has, made when
/// first asked for.
pub fn reads() -> &'static ThreadPool {
    static READS: OnceLock<ThreadPool> = OnceLock::new();
    READS.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(global_threads())
            .thread_name(|n| format!("vellev-read-{n}"))
            .build()
            .expect("the read pool's threads")
    })
}

/// How many threads rayon gives its global pool, worked out as rayon does
/// (RAYON_NUM_THREADS where it is set and not 0, else as many as the
/// machine runs at once), not read from whichever pool the caller is in:
/// a first split asked from inside a pool of one thread would otherwise
/// make a read pool of one.
fn global_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|threads| threads.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}
