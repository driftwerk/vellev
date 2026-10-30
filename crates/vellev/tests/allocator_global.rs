//! Whether jemalloc is the process's global allocator is told rightly
//! however much other threads allocate and free meanwhile (re-review of
//! #165, L1): a binary of its own, on the node's allocator, since what it
//! measures is the whole process's.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[global_allocator]
static GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

/// Another thread allocating and freeing 64 MiB without let-up never makes
/// the process's own allocator look like another's.
#[test]
fn the_global_allocator_is_told_however_other_threads_allocate_and_free() {
    let churning = Arc::new(AtomicBool::new(true));
    let churner = std::thread::spawn({
        let churning = churning.clone();
        move || {
            while churning.load(Ordering::Relaxed) {
                let block = std::hint::black_box(vec![1u8; 64 * 1024 * 1024]);
                drop(block);
            }
        }
    });
    let told: Vec<Option<bool>> = (0..500).map(|_| GLOBAL.is_global()).collect();
    churning.store(false, Ordering::Relaxed);
    churner.join().unwrap();
    let wrong = told.iter().filter(|told| **told != Some(true)).count();
    assert_eq!(wrong, 0, "told otherwise {wrong} times of 500");
}
