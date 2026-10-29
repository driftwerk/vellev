//! Asking a long build to give up (merge note, step 3).
//!
//! A merge builds its indexes for seconds or minutes, a million rows at a
//! time, and holds gigabytes meanwhile. When its tenant is let go, or its
//! lease lost, the merge can never commit, and what it holds is wanted
//! back: so a build is handed a [`Stop`], and asks it in each long pass it
//! makes, a row or a block of rows at a time, giving up with [`Stopped`]
//! once it is stopped. Asking costs a load of one flag.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Whether a build is to give up: shared by clones, so that whoever holds
/// one may stop a build another runs.
#[derive(Clone, Default)]
pub struct Stop(Arc<Inner>);

#[derive(Default)]
struct Inner {
    stopped: AtomicBool,
    /// How often each pass asked, for tests: counted only when asked for,
    /// so that a build's asking stays a load of one flag.
    counted: Option<Counts>,
}

struct Counts {
    checks: [AtomicUsize; Stage::ALL.len()],
    /// The pass at whose first asking the stop stops itself.
    at_first: Option<Stage>,
}

/// The long passes of a build, in which it asks its stop. Named for tests
/// that show each one asks; not for use.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Drawing and rotating the sample that clusters or a centre are
    /// taken from.
    Sampling,
    /// Clustering the sample: each iteration, and each row of it assigned.
    Clustering,
    /// Coding each row.
    Coding,
    /// Tuning how many clusters a query scans: waiting for the turn on the
    /// tuning pool, each block of rows of the pass over them, and each
    /// sample ranked.
    Tuning,
}

impl Stage {
    const ALL: [Stage; 4] = [
        Stage::Sampling,
        Stage::Clustering,
        Stage::Coding,
        Stage::Tuning,
    ];

    fn at(self) -> usize {
        self as usize
    }
}

/// What a build gives back when it gave up, stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopped;

impl fmt::Display for Stopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the build was stopped")
    }
}

impl std::error::Error for Stopped {}

impl fmt::Debug for Stop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stop")
            .field("stopped", &self.is_stopped())
            .finish()
    }
}

impl Stop {
    /// A stop not yet stopped.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stop every build this stop, or a clone of it, was given: each gives
    /// up at the next row or block of rows it comes to.
    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::Relaxed);
    }

    /// Whether this stop has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.0.stopped.load(Ordering::Relaxed)
    }

    /// A stop that counts how often each pass asks it. For tests.
    #[doc(hidden)]
    pub fn counting() -> Self {
        Self::counted(None)
    }

    /// A counting stop that stops itself the first time `stage` asks it.
    /// For tests.
    #[doc(hidden)]
    pub fn at_first(stage: Stage) -> Self {
        Self::counted(Some(stage))
    }

    fn counted(at_first: Option<Stage>) -> Self {
        Self(Arc::new(Inner {
            stopped: AtomicBool::new(false),
            counted: Some(Counts {
                checks: Default::default(),
                at_first,
            }),
        }))
    }

    /// How often `stage` asked this stop, if it counts. For tests.
    #[doc(hidden)]
    pub fn checks(&self, stage: Stage) -> usize {
        self.0
            .counted
            .as_ref()
            .map_or(0, |counts| counts.checks[stage.at()].load(Ordering::Relaxed))
    }

    /// Asked by `stage`: [`Stopped`] if the build is to give up.
    pub(crate) fn check(&self, stage: Stage) -> Result<(), Stopped> {
        if let Some(counts) = &self.0.counted {
            counts.checks[stage.at()].fetch_add(1, Ordering::Relaxed);
            if counts.at_first == Some(stage) {
                self.stop();
            }
        }
        match self.is_stopped() {
            true => Err(Stopped),
            false => Ok(()),
        }
    }
}
