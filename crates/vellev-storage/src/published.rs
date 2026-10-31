//! A partition's state, which can only be changed so that what it comes to
//! is published with the change: read without the state's lock, by a node
//! for its metrics and its memory budget, it is never behind a change made.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;

use crate::state::{Discarded, Figures, State};

/// The longest a free waits where a test holds it, were the test never to
/// let it go.
const HELD_AT_MOST: Duration = Duration::from_secs(60);

/// Where a test holds what a merge's commit took out of a partition, as its
/// free is about to begin. Not for use.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct FreesHeld {
    let_go: Mutex<bool>,
    told: Condvar,
    reached: AtomicUsize,
}

impl FreesHeld {
    /// How many frees have come to where they are held.
    pub fn reached(&self) -> usize {
        self.reached.load(Ordering::SeqCst)
    }

    /// Let every free held go, and those to come pass.
    pub fn let_go(&self) {
        *self.let_go.lock().expect("held lock") = true;
        self.told.notify_all();
    }

    /// Wait here until let go, at most [`HELD_AT_MOST`].
    pub(crate) fn wait(&self) {
        self.reached.fetch_add(1, Ordering::SeqCst);
        let let_go = self.let_go.lock().expect("held lock");
        let held = self
            .told
            .wait_timeout_while(let_go, HELD_AT_MOST, |let_go| !*let_go);
        drop(held.expect("held lock"));
    }
}

/// The state, and what it came to as it was last changed. The fields are
/// private, so that nothing changes the state but [`Self::changing`] and
/// [`Self::applying`], which publish what it comes to under the same lock.
#[derive(Debug, Default)]
pub(crate) struct Published {
    state: RwLock<State>,
    figures: Mutex<Figures>,
    /// What segments taken out of the state still hold, until they are
    /// freed: see [`Self::freeing`].
    freeing: Arc<AtomicUsize>,
    /// Where a test holds frees, if it does: see [`Self::hold_frees`].
    frees_held: Mutex<Option<Arc<FreesHeld>>>,
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

    /// What segments taken out of the state are counted to hold until they
    /// are freed, which what the state comes to does not count. Read
    /// without its lock.
    pub(crate) fn freeing(&self) -> usize {
        self.freeing.load(Ordering::SeqCst)
    }

    /// What the state came to as it was last changed, with what it took out
    /// and has yet to free: all it holds in memory. Read in this order, as
    /// [`Self::adopting`] counts in the other, so that it never falls short.
    pub(crate) fn resident(&self) -> usize {
        self.figures().resident + self.freeing()
    }

    /// Hold each merge's commit's free from now on: see [`FreesHeld`].
    pub(crate) fn hold_frees(&self) -> Arc<FreesHeld> {
        let held = Arc::new(FreesHeld::default());
        *self.frees_held.lock().expect("frees held lock") = Some(held.clone());
        held
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

    /// As [`Self::changing`], for adopting a checkpoint: the segments `f`
    /// takes out of the state go on being counted, in [`Self::freeing`],
    /// until they are freed. Counted before what the state comes to without
    /// them is published, so that they are never gone from both.
    pub(crate) fn adopting(&self, f: impl FnOnce(&mut State) -> Discarded) -> Discarded {
        self.changing(|state| {
            let mut discarded = f(state);
            discarded.count_in(&self.freeing);
            if discarded.replaced() {
                discarded.held = self.frees_held.lock().expect("frees held lock").clone();
            }
            discarded
        })
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
        debug_assert_eq!(
            *figures,
            state.figures(),
            "applying a commit changed what only a change of segments may"
        );
        changed
    }

    /// Whether a change holds the state's lock now, as a test asks of a
    /// state nothing else is writing to.
    #[cfg(test)]
    pub(crate) fn being_changed(&self) -> bool {
        use std::sync::TryLockError;
        matches!(self.state.try_read(), Err(TryLockError::WouldBlock))
    }

    /// Change the state without publishing it: only for a test that sets
    /// up what no change makes.
    #[cfg(test)]
    pub(crate) fn unpublished<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        f(&mut self.state.write().expect("partition lock"))
    }
}
