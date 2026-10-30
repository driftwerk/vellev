//! A pool for merges to build on (merge note, §9.2): its builds take their
//! turns to tune on the one tuning thread as any build does, and two of
//! them on its threads never wait on each other for good. A test binary
//! of its own: the tuning pool is the process's. Its threads' priority is
//! told in `merge_pool_nice`, alone, as other pools here would be counted.

use std::sync::mpsc;
use std::time::Duration;

use vellev_index::pool;
use vellev_index::stop::Stop;

/// Two builds on a merge pool of one thread, the second asked for while
/// the first runs, each tuning once it has begun, both finish: the first,
/// waiting for its tuning, never takes the second on in its stead, which
/// would then wait for its own turn behind the first's for good.
#[test]
fn two_builds_on_a_merge_pool_never_wait_on_each_other_for_good() {
    // Threads of their own, not scoped: were the builds to wait on each
    // other for good, the test fails at its limit rather than hanging on
    // them.
    let merges = std::sync::Arc::new(pool::merge_builds(1, 0));
    let (done, finished) = mpsc::channel();
    let (began, first_began) = mpsc::channel();
    let first = (merges.clone(), done.clone());
    std::thread::spawn(move || {
        let (merges, done) = first;
        let tuned = merges.install(|| {
            began.send(()).unwrap();
            // The second build is asked for meanwhile, and waits for this
            // pool's one thread.
            std::thread::sleep(Duration::from_millis(300));
            pool::tune(&Stop::new(), || {
                std::thread::sleep(Duration::from_millis(300));
                1
            })
        });
        done.send(tuned.unwrap()).unwrap();
    });
    first_began.recv().unwrap();
    std::thread::spawn(move || {
        let tuned = merges.install(|| pool::tune(&Stop::new(), || 2));
        done.send(tuned.unwrap()).unwrap();
    });
    let mut tuned = Vec::new();
    for _ in 0..2 {
        let one = finished.recv_timeout(Duration::from_secs(30));
        tuned.push(one.expect("both builds tuned within 30 s: they wait on each other"));
    }
    tuned.sort();
    assert_eq!(tuned, vec![1, 2]);
}

/// A merge pool's threads are named for it, as many as it was made with.
#[test]
fn a_merge_pools_threads_are_named_for_it() {
    let merges = pool::merge_builds(2, 0);
    assert_eq!(merges.current_num_threads(), 2);
    let named = merges.install(|| std::thread::current().name().map(str::to_string));
    assert!(named.unwrap().starts_with("vellev-merge-"));
}
