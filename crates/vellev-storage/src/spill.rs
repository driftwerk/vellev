//! A merge's vectors gathered on local disk (node memory note, M8).

use std::sync::atomic::AtomicU64;

/// Where a store's frames are gathered on local disk.
#[derive(Debug, Clone)]
pub struct Spills;

/// What a test has a disk do to what is spilled to it. Not for use.
#[doc(hidden)]
#[derive(Debug)]
pub struct SpillsTold {
    /// How many bytes of each spill the disk takes before it is full.
    pub at_most: AtomicU64,
    /// How long, in milliseconds, each write takes at the least.
    pub taking: AtomicU64,
    /// How long, in milliseconds, a cache takes to take a spill up.
    pub taken_up_in: AtomicU64,
}

impl Default for SpillsTold {
    fn default() -> Self {
        Self {
            at_most: AtomicU64::new(u64::MAX),
            taking: AtomicU64::new(0),
            taken_up_in: AtomicU64::new(0),
        }
    }
}
