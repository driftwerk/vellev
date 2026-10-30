//! A partition's state, which can only be changed so that what it comes to
//! is published with the change: read without the state's lock, by a node
//! for its metrics and its memory budget, it is never behind a change made.

use std::sync::{Mutex, RwLock};

use crate::state::{Figures, State};

/// The state, and what it came to as it was last changed. The fields are
/// private, so that nothing changes the state but [`Self::changing`] and
/// [`Self::applying`], which publish what it comes to under the same lock.
#[derive(Debug, Default)]
pub(crate) struct Published {
    state: RwLock<State>,
    figures: Mutex<Figures>,
}

impl Published {
    /// Run `f` against the state as it is. The lock is a blocking one and
    /// is never held across an await.
    pub(crate) fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        f(&self.state.read().expect("partition lock"))
    }

    /// What the state came to as it was last changed, read without its
    /// lock, so never waiting on a change under way.
    pub(crate) fn figures(&self) -> Figures {
        *self.figures.lock().expect("figures lock")
    }

    /// Change the state as `f` does, under its lock, and publish all it
    /// then comes to, under the same lock, so that what is published
    /// follows the changes in their order: for any change of its segments
    /// (a flush, a merge, curves, a restore, a state loaded in its place).
    pub(crate) fn changing<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        let mut state = self.state.write().expect("partition lock");
        let changed = f(&mut state);
        *self.figures.lock().expect("figures lock") = state.figures();
        changed
    }

    /// As [`Self::changing`], for applying a commit, which changes the
    /// memtable and which rows are live, never the segments: what it holds
    /// and what waits for a flush are worked out afresh, and the segments'
    /// counts and probes kept as they were.
    pub(crate) fn applying<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        let mut state = self.state.write().expect("partition lock");
        let changed = f(&mut state);
        let mut figures = self.figures.lock().expect("figures lock");
        *figures = Figures {
            resident: state.resident(),
            unflushed: state.unflushed(),
            unflushed_bytes: state.unflushed_bytes(),
            ..*figures
        };
        changed
    }

    /// Change the state without publishing it: only for a test that sets
    /// up what no change makes.
    #[cfg(test)]
    pub(crate) fn unpublished<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        f(&mut self.state.write().expect("partition lock"))
    }
}
