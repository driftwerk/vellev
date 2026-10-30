//! What the allocator holds is on /metrics, as it is when the metrics are
//! read (resident accounting, PR 1): beside what the node reckons its
//! tenants hold, it shows how far that reckoning is from what the process
//! really allocates.
//!
//! The test reads the whole process's allocator, so it is the only test in
//! this binary, which runs on the node's allocator as the node does.

use std::sync::Arc;
use std::time::Duration;

use vellev::server::{App, Limits};

#[global_allocator]
static GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

const MIB: usize = 1024 * 1024;

/// The value of the unlabelled sample `name`, if there is one.
fn sample(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .find_map(|line| line.strip_prefix(&format!("{name} ")))
        .map(|value| value.trim().parse().unwrap())
}

fn app() -> App {
    let limits = Limits {
        results: 100,
        body: MIB,
        embedding_urls: Arc::from([]),
        reads: 4,
        reads_wait: Duration::from_millis(50),
    };
    App::new(Arc::from(["key".to_string()]), limits).running_on(&GLOBAL)
}

/// 64 MiB allocated and written between two readings shows in the
/// second, and goes once freed: each reading is jemalloc's count as of
/// then, not as of whenever it last refreshed its statistics. What is
/// resident is at least what is allocated.
#[tokio::test]
async fn what_the_allocator_holds_is_reported_as_it_is_when_read() {
    let app = app();
    let read = |name: &str| {
        let text = app.metrics().encoded();
        sample(&text, name).unwrap_or_else(|| panic!("no {name}: {text}"))
    };
    let before = read("vellev_allocator_allocated_bytes");
    let held = vec![1u8; 64 * MIB];
    let allocated = read("vellev_allocator_allocated_bytes");
    let resident = read("vellev_allocator_resident_bytes");
    assert!(
        allocated - before >= (63 * MIB) as f64,
        "64 MiB more allocated, but reported {before} then {allocated}"
    );
    assert!(
        resident >= allocated,
        "resident {resident} below allocated {allocated}"
    );
    assert_eq!(
        held.iter().map(|byte| *byte as usize).sum::<usize>(),
        64 * MIB
    );
    drop(held);
    let after = read("vellev_allocator_allocated_bytes");
    assert!(
        after <= allocated - (63 * MIB) as f64,
        "64 MiB freed, but reported {allocated} then {after}"
    );
}
