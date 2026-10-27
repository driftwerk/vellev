//! The node's memory allocator: jemalloc, which gives back what is freed.
//!
//! A flush or a merge copies a run's vectors several times over and frees
//! the copies, leaving a little behind among them. glibc's malloc did not
//! give such memory back: the node held its high-water mark, 4.3 GiB of
//! anonymous memory at rest against 0.9 GiB in use at 300,000 rows, and
//! was killed for memory loading 1M rows of 768 dimensions in 10 GiB.
//! With jemalloc the same load settled at 3.1-3.3 GiB.
//!
//! On Linux a background thread purges what was freed, so memory goes back
//! while the node is idle too, not only as it next allocates: dirty pages
//! after 10 s, as jemalloc decays them by default, then given back to the
//! system. jemalloc runs the same on macOS, purging as it allocates.

use std::alloc::{GlobalAlloc, Layout};

/// The allocator the `vellev` binary runs on, and its tests that hold it to
/// giving memory back.
pub struct Allocator(tikv_jemallocator::Jemalloc);

impl Allocator {
    pub const fn new() -> Self {
        Self(tikv_jemallocator::Jemalloc)
    }
}

impl Default for Allocator {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: every call is jemalloc's, as `tikv_jemallocator::Jemalloc` makes
// it, with the layout it is given.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { self.0.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { self.0.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { self.0.realloc(ptr, layout, new_size) }
    }
}

/// The options the `vellev` binary gives jemalloc, which reads them once
/// as it starts from the symbol `_rjem_malloc_conf` (tikv-jemalloc-sys
/// prefixes jemalloc's names): a background thread that purges freed pages
/// on Linux, so that memory goes back while the node is idle too. Only a
/// binary exports that symbol, so that a crate depending on this library
/// is not given the node's options, nor kept from setting its own.
pub const OPTIONS: &std::ffi::CStr = c"background_thread:true";

/// What exports [`OPTIONS`] under the name jemalloc reads them from: a
/// pointer to a static C string, which nothing writes.
#[repr(transparent)]
pub struct Options(pub *const std::ffi::c_char);

// SAFETY: it points at a static string that nothing writes.
unsafe impl Sync for Options {}
