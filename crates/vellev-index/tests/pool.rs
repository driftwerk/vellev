//! The read pool is sized as rayon sizes its global pool, whichever pool
//! first asks for it. A test binary of its own, as the pool is made once
//! a process.

/// First asked for from inside a pool of one thread, the read pool still
/// has as many threads as rayon's global pool: RAYON_NUM_THREADS where it
/// is set, else as many as the machine runs at once.
#[test]
fn the_read_pool_is_sized_as_the_global_pool_whoever_asks_first() {
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let sized = one.install(|| vellev_index::pool::reads().current_num_threads());
    let global = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|threads| threads.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    assert_eq!(sized, global);
    assert_eq!(rayon::current_num_threads(), global, "as rayon's own global pool");
}
