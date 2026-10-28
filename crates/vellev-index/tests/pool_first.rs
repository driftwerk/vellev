//! A pool's threads take the priority of the thread that makes them, on
//! Linux. A test binary of its own, as each pool is made once a process.

/// The read pool, first asked for from a thread of a background pool,
/// which runs at a lower priority, still runs at the process's own: the
/// background pools are made after it.
#[cfg(target_os = "linux")]
#[test]
fn the_read_pool_runs_at_the_process_priority_whoever_asks_first() {
    let of_this_thread = || unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::getpriority(libc::PRIO_PROCESS, tid)
    };
    let here = of_this_thread();
    let reads = vellev_index::pool::tuning()
        .install(|| vellev_index::pool::reads().install(of_this_thread));
    assert_eq!(reads, here);
}
