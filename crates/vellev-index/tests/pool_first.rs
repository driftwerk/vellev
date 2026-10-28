//! A pool's threads take the priority of the thread that makes them, on
//! Linux. A test binary of its own, as each pool is made once a process.

/// The read pool and the tuning pool, first asked for from a rows pass's
/// thread, which runs at a lower priority, still run at the process's own:
/// both are made before the passes' pool.
#[cfg(target_os = "linux")]
#[test]
fn the_pools_at_the_process_priority_stay_there_whoever_asks_first() {
    let of_this_thread = || unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::getpriority(libc::PRIO_PROCESS, tid)
    };
    let here = of_this_thread();
    let (tuning, reads) = vellev_index::pool::passes().install(|| {
        (
            vellev_index::pool::tuning().install(of_this_thread),
            vellev_index::pool::reads().install(of_this_thread),
        )
    });
    assert_eq!(tuning, here, "tuning");
    assert_eq!(reads, here, "reads");
}
