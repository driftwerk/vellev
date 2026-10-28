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

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use rayon::{ThreadPool, ThreadPoolBuilder};

/// The read pool: as many threads as rayon's global pool has, sized as
/// rayon sizes that (RAYON_NUM_THREADS, and so on), whichever pool first
/// asks for it, made when first asked for.
pub fn reads() -> &'static ThreadPool {
    static READS: OnceLock<ThreadPool> = OnceLock::new();
    READS.get_or_init(|| {
        ThreadPoolBuilder::new()
            .thread_name(|n| format!("vellev-read-{n}"))
            .build()
            .expect("the read pool's threads")
    })
}

/// How many index builds are under way in the process.
static BUILDS: AtomicUsize = AtomicUsize::new(0);
/// How many index builds have begun in the process, ever.
static BEGUN: AtomicU64 = AtomicU64::new(0);

/// How many index builds are under way: a read does not split while any
/// is, as split it is no faster then and slows the build (review of #141).
pub fn builds() -> usize {
    BUILDS.load(Ordering::Relaxed)
}

/// How many builds have begun in the process, ever: what a test counts by,
/// as others may build meanwhile.
pub fn builds_begun() -> u64 {
    BEGUN.load(Ordering::Relaxed)
}

/// A build counted as under way until this is dropped.
#[must_use = "a build is counted only while this is held"]
pub struct Building(());

impl Building {
    pub fn begun() -> Self {
        BUILDS.fetch_add(1, Ordering::Relaxed);
        BEGUN.fetch_add(1, Ordering::Relaxed);
        Self(())
    }
}

impl Drop for Building {
    fn drop(&mut self) {
        BUILDS.fetch_sub(1, Ordering::Relaxed);
    }
}
