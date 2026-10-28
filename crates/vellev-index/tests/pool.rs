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
