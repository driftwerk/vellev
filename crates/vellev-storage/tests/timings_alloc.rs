//! A duration timed while no bench keeps events allocates nothing: the
//! partition it is timed for is named, as a string of its own, only for an
//! event kept (review of #169). Every flush, merge phase and hold of the
//! building lock is timed, so a name made for each would be an allocation
//! on every one of them for nothing. In a binary of its own, as it counts
//! every allocation this thread makes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt;
use std::time::Duration;

use vellev_storage::timings::{Timed, Timings};

/// The system's allocator, counting what the thread that asked makes
/// while it asks.
struct Counting;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static COUNTED: Cell<u64> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.with(Cell::get) {
            COUNTED.with(|counted| counted.set(counted.get() + 1));
        }
        // SAFETY: as the caller asked of this allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as the caller asked of this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// How many allocations `f` made on this thread.
fn allocations(f: impl FnOnce()) -> u64 {
    COUNTED.with(|counted| counted.set(0));
    COUNTING.with(|counting| counting.set(true));
    f();
    COUNTING.with(|counting| counting.set(false));
    COUNTED.with(Cell::get)
}

/// A partition's name, as a writer's lease names it: made into a string
/// only if asked.
struct Partition(u32);

impl fmt::Display for Partition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tenant-{}/partition", self.0)
    }
}

#[test]
fn a_duration_timed_while_no_events_are_kept_allocates_nothing() {
    let timings = Timings::default();
    let partition = Partition(7);
    let unkept = allocations(|| {
        for n in 0..1_000 {
            timings.record(Timed::FlushDue, &partition, Duration::from_millis(n));
        }
    });
    assert_eq!(unkept, 0, "allocations for durations no bench keeps");

    // Kept, the event names its partition: the count sees that.
    timings.keep_events(true);
    let kept = allocations(|| {
        timings.record(Timed::FlushDue, &partition, Duration::from_millis(1));
    });
    assert!(kept > 0, "a kept event names its partition");
    let events = timings.drain_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].partition, "tenant-7/partition");
}
