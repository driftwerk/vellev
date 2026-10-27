//! The node's memory allocator. For now the system's, as before: its rule,
//! that memory freed among what is kept is given back, is written first.

use std::alloc::{GlobalAlloc, Layout, System};

pub struct Allocator(System);

impl Allocator {
    pub const fn new() -> Self {
        Self(System)
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every call is the system allocator's.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { self.0.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(ptr, layout) }
    }
}
