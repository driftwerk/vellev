//! The read pool is sized as rayon sizes its global pool, whichever pool
//! first asks for it. A test binary of its own, as the pool is made once
//! a process.

/// First asked for from inside a pool of one thread, the read pool still
/// has as many threads as rayon's global pool, however rayon sized that.
#[test]
fn the_read_pool_is_sized_as_the_global_pool_whoever_asks_first() {
    let global = rayon::current_num_threads();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let sized = one.install(|| vellev_index::pool::reads().current_num_threads());
    assert_eq!(sized, global);
}

/// Tuning's and the rows passes' threads run below the rest of the
/// process, as far as Linux lets a thread lower itself, so that on a small
/// box they yield the CPU to reads.
#[cfg(target_os = "linux")]
#[test]
fn the_background_pools_run_at_a_lower_priority() {
    let of_this_thread = || unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::getpriority(libc::PRIO_PROCESS, tid)
    };
    let here = of_this_thread();
    let lower = (here + vellev_index::pool::BACKGROUND_NICE).min(19);
    for (what, pool) in [
        ("tuning", vellev_index::pool::tuning()),
        ("passes", vellev_index::pool::passes()),
    ] {
        assert_eq!(pool.install(of_this_thread), lower, "{what}");
    }
    assert_eq!(
        vellev_index::pool::reads().install(of_this_thread),
        here,
        "reads"
    );
}
