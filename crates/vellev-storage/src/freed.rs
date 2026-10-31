//! Freeing off the async threads, as what a merge built or replaced must
//! be: a segment of a million rows takes seconds to free, all of it small
//! allocations, and the thread that does it does nothing else meanwhile.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use crate::merge_builds::MergeBuild;

/// Run `free` off the async threads, if this is a runtime's thread at all.
/// Sent off without a turn among the work there, and not waited for: what
/// lets it go may have given up waiting for that very turn, may hold a
/// lock that other work waits for, or may be past waiting for anything.
pub(crate) fn off_the_async_threads(free: impl FnOnce() + Send + 'static) {
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => drop(runtime.spawn_blocking(free)),
        Err(_) => free(),
    }
}

/// `T`, sent [`off_the_async_threads`] should it be let go, as by a merge
/// stopped or dropped: for what is destructured to be used, which so cannot
/// be `Drop` itself.
pub(crate) struct FreedOff<T: Send + 'static> {
    value: Option<T>,
    /// The build it is what's left of, if it is: the turn and the reckoning
    /// go with the last of what the build made, wherever that is freed.
    build: Option<Arc<MergeBuild>>,
}

impl<T: Send + 'static> FreedOff<T> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            value: Some(value),
            build: None,
        }
    }

    /// As what's left of `build`, if it is.
    pub(crate) fn of(mut self, build: Option<Arc<MergeBuild>>) -> Self {
        self.build = build;
        self
    }

    /// The build it is what's left of, if it is.
    pub(crate) fn build(&self) -> Option<Arc<MergeBuild>> {
        self.build.clone()
    }

    /// `T` itself, which is then for the caller to free.
    pub(crate) fn into_inner(mut self) -> T {
        self.value.take().expect("held until let go")
    }
}

impl<T: Send + 'static> Deref for FreedOff<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value.as_ref().expect("held until let go")
    }
}

impl<T: Send + 'static> DerefMut for FreedOff<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("held until let go")
    }
}

impl<T: Send + 'static> Drop for FreedOff<T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            let build = self.build.take();
            off_the_async_threads(move || {
                drop(value);
                drop(build);
            });
        }
    }
}
