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

/// Builds are counted while they run, across threads, and no longer once
/// each is done; and each one begun is counted for good. Counted by how
/// they change, as other tests may build meanwhile.
#[test]
fn builds_are_counted_while_they_run() {
    let (running, begun) = (
        vellev_index::pool::builds(),
        vellev_index::pool::builds_begun(),
    );
    let first = vellev_index::pool::Building::begun();
    let (tx, rx) = std::sync::mpsc::channel();
    let (go, wait) = std::sync::mpsc::channel::<()>();
    let second = std::thread::spawn(move || {
        let _building = vellev_index::pool::Building::begun();
        tx.send(()).unwrap();
        wait.recv().unwrap();
    });
    rx.recv().unwrap();
    assert!(vellev_index::pool::builds() >= running + 2);
    go.send(()).unwrap();
    second.join().unwrap();
    drop(first);
    assert!(vellev_index::pool::builds_begun() >= begun + 2);
}
