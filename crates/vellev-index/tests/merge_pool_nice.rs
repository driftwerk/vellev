//! On Linux a merge pool's threads run at the priority it was made with,
//! and no other pool's do (merge note, §9.2). A test binary of its own,
//! with the one merge pool: every thread of the process is looked at, and
//! another test's pool would be counted with it.

#![cfg(target_os = "linux")]

use vellev_index::pool;

/// The nice value of each thread of the process whose name begins with
/// `prefix`, as Linux reports it.
fn nice_of_threads(prefix: &str) -> Vec<i64> {
    let mut nices = Vec::new();
    for task in std::fs::read_dir("/proc/self/task").unwrap() {
        let dir = task.unwrap().path();
        let Ok(comm) = std::fs::read_to_string(dir.join("comm")) else {
            continue;
        };
        if !comm.trim_end().starts_with(prefix) {
            continue;
        }
        let stat = std::fs::read_to_string(dir.join("stat")).unwrap();
        // After the command, in parentheses, which may hold spaces: the
        // nice value is the 19th field of the line, the 17th after it.
        let after = &stat[stat.rfind(')').unwrap() + 2..];
        let nice = after.split(' ').nth(16).unwrap().parse().unwrap();
        nices.push(nice);
    }
    nices
}

/// On Linux a merge pool made with a nice of 5 runs each of its threads 5
/// lower, and the read and tuning pools, made first, as the process runs.
#[test]
fn a_merge_pools_threads_run_at_its_nice_and_no_other_pools_do() {
    let merges = pool::merge_slots(1, 3, 5).pop().unwrap();
    // Every thread started and running.
    merges.broadcast(|_| ());
    pool::reads().broadcast(|_| ());
    pool::tuning().broadcast(|_| ());
    let niced = nice_of_threads("vellev-merge");
    assert_eq!(niced, vec![5; 3], "the merge pool's threads");
    let reads = nice_of_threads("vellev-read");
    assert_eq!(reads.len(), pool::reads().current_num_threads());
    assert!(
        reads.iter().all(|nice| *nice == 0),
        "the read pool's: {reads:?}"
    );
    assert_eq!(
        nice_of_threads("vellev-tune"),
        vec![0],
        "the tuning thread's"
    );
}
