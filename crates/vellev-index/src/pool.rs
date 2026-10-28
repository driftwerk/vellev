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

/// The pool a build's tuning runs on: one thread of its own (recall note,
/// R2.6). Tuning is a pass over a segment's vectors for every sample, tens
/// of seconds a million rows; on the global pool it took every core for
/// that long at each merge, and reads waited. On one thread it takes
/// longer and leaves the rest to reads, flushes and merges. A rows pass,
/// the same work for a segment from before curves, has [`passes`].
pub fn tuning() -> &'static ThreadPool {
    static TUNING: OnceLock<ThreadPool> = OnceLock::new();
    TUNING.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "vellev-tune".to_string())
            .build()
            .expect("the tuning pool's thread")
    })
}

/// The pool a rows pass makes its curve on (recall note, R9.5): one
/// thread of its own, apart from the tuning pool, so that a merge's own
/// tuning never waits behind the passes an upgrade makes, segment after
/// segment, each tens of seconds a million rows. At most one pass runs on
/// a node at a time, so tuning and passes take two threads at the most.
pub fn passes() -> &'static ThreadPool {
    static PASSES: OnceLock<ThreadPool> = OnceLock::new();
    PASSES.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "vellev-pass".to_string())
            .build()
            .expect("the rows passes' pool's thread")
    })
}
