//! The merge builds a node runs at once (merge note, step 3, §7 and §7.7).
//!
//! A merge takes a turn among at most [`MERGE_BUILDS`] before it gathers a
//! row, and reckons what its build holds at its peak. Both are held by
//! every piece of work that holds what the merge gathered or built: the
//! merge itself, its build off the async threads, and each of its uploads.
//! They are let go with the last of those to end, not when the merge is
//! dropped: a stopped build that runs on, or an upload that has yet to
//! settle, still holds its turn, so that no further merge gathers beside
//! it. What a stopped merge is reckoned to hold counts as draining until
//! then.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use vellev_index::stop::Stop;

use crate::stopping::{MergeStage, Stopping};
use crate::{Result, StorageError};

/// How many merges a node builds at once. What a merge gathers, a copy of
/// its run's live rows and their vectors, is the bulk of what it holds, so
/// the node's merges hold at most this many times the largest.
pub const MERGE_BUILDS: usize = 2;

/// The longest a build waits where a test holds it, were the test never to
/// let it go.
const HELD_AT_MOST: Duration = Duration::from_secs(60);

/// How often a build a test holds asks whether to go on, and a merge
/// waiting for its turn whether a flush is due.
const ASKS_EVERY: Duration = Duration::from_millis(20);

/// A node's merge builds: their turns, and what each under way is reckoned
/// to hold. Shared by every writer the node opens; a writer on its own has
/// one of its own.
#[derive(Debug)]
pub struct MergeBuilds {
    turns: Arc<Semaphore>,
    /// Each build under way, from its turn to the end of the last work
    /// that holds what it gathered or built.
    under_way: Mutex<Vec<Arc<Reckoned>>>,
    /// Where a test has builds wait, if anywhere: see [`Self::hold`].
    holding: Mutex<Option<Holding>>,
    let_go: Condvar,
    /// How many builds wait where a test holds them now.
    held: AtomicUsize,
    /// What the last build to end was reckoned to hold, as it was planned
    /// and once its rows were gathered.
    last_planned: AtomicUsize,
    last_reckoned: AtomicUsize,
    /// Whether each build's output carries a [`Probe`], as a test asks.
    probing: AtomicBool,
    /// How the builds stood as the last probe was dropped.
    probed: Mutex<Option<Probed>>,
}

/// What one build under way is reckoned to hold, and the stop of its merge.
#[derive(Debug)]
struct Reckoned {
    stop: Stop,
    bytes: AtomicUsize,
    /// What it was reckoned to hold as it was planned.
    planned: AtomicUsize,
    /// Whether its merge is past where a stop ends it: one stopped after
    /// commits what it holds, and drains nothing.
    committing: AtomicBool,
}

/// How a test holds each merge's build off the async threads, with its
/// turn taken and its rows gathered. Not for use.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holding {
    /// At its start, until [`MergeBuilds::hold`] lets it go, stopped or
    /// not: a build slow to come to where it asks its stop.
    UntilLetGo,
    /// At its start, until its merge is stopped: a build that would
    /// otherwise run on.
    UntilStopped,
    /// Once built, past the last place it asks its stop, until let go: a
    /// build that ends with what it built, whether or not it is awaited.
    BuiltUntilLetGo,
}

/// How the builds stood as a [`Probe`] was dropped. Not for use.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Probed {
    pub under_way: usize,
    pub turns_free: usize,
    pub draining: usize,
}

/// Carried by what a build makes, as a test asks: notes how the builds
/// stood as it is dropped, with the last of what the build made.
#[derive(Debug)]
pub(crate) struct Probe {
    builds: Arc<MergeBuilds>,
}

impl Drop for Probe {
    fn drop(&mut self) {
        let builds = &self.builds;
        let probed = Probed {
            under_way: builds.under_way(),
            turns_free: builds.turns_free(),
            draining: builds.draining(),
        };
        *lock(&builds.probed) = Some(probed);
    }
}

impl MergeBuilds {
    /// Builds of merges `at_once` at a time.
    pub fn new(at_once: usize) -> Arc<Self> {
        Arc::new(Self {
            turns: Arc::new(Semaphore::new(at_once)),
            under_way: Mutex::default(),
            holding: Mutex::default(),
            let_go: Condvar::new(),
            held: AtomicUsize::new(0),
            last_planned: AtomicUsize::new(0),
            last_reckoned: AtomicUsize::new(0),
            probing: AtomicBool::new(false),
            probed: Mutex::default(),
        })
    }

    /// About how many bytes the merges stopped and not yet ended hold:
    /// what each was reckoned to hold at its peak, until the last of its
    /// work has ended. Merges running unstopped are not counted, nor any
    /// stopped once past where a stop ends it, which commits instead.
    pub fn draining(&self) -> usize {
        lock(&self.under_way)
            .iter()
            .filter(|build| build.stop.is_stopped() && !build.committing.load(Ordering::SeqCst))
            .map(|build| build.bytes.load(Ordering::SeqCst))
            .sum()
    }

    /// How many builds are under way: each from its turn to the end of the
    /// last work that holds what it gathered or built. For tests.
    #[doc(hidden)]
    pub fn under_way(&self) -> usize {
        lock(&self.under_way).len()
    }

    /// How many turns are free. For tests.
    #[doc(hidden)]
    pub fn turns_free(&self) -> usize {
        self.turns.available_permits()
    }

    /// Hold each merge's build from now on as `holding` says, or none, and
    /// let go of those held if `None`. For tests that make things happen
    /// while builds run.
    #[doc(hidden)]
    pub fn hold(&self, holding: Option<Holding>) {
        *lock(&self.holding) = holding;
        self.let_go.notify_all();
    }

    /// How many builds wait where a test holds them now. For tests.
    #[doc(hidden)]
    pub fn held(&self) -> usize {
        self.held.load(Ordering::SeqCst)
    }

    /// What the last build to end was reckoned to hold at its peak. For
    /// tests that hold the reckoning to what a build is measured to hold.
    #[doc(hidden)]
    pub fn last_reckoned(&self) -> usize {
        self.last_reckoned.load(Ordering::SeqCst)
    }

    /// What the last build to end was reckoned to hold at its peak as it
    /// was planned, before its rows were gathered. For tests.
    #[doc(hidden)]
    pub fn last_planned(&self) -> usize {
        self.last_planned.load(Ordering::SeqCst)
    }

    /// Have what each build makes from now on carry a probe, or not. For
    /// tests that see how the builds stand as the last of it is dropped.
    #[doc(hidden)]
    pub fn probe(&self, probing: bool) {
        self.probing.store(probing, Ordering::SeqCst);
    }

    /// How the builds stood as the last probe was dropped, if one was. For
    /// tests.
    #[doc(hidden)]
    pub fn probed(&self) -> Option<Probed> {
        *lock(&self.probed)
    }

    /// A turn to build the merge `stopping` stops, once one is free; `None`
    /// if the merge is stopped first, or `flush_due` says its writer's flush
    /// is due first, when it never builds. A turn free at once is taken,
    /// whatever is due: what then holds the flush is the merge's own build,
    /// not other merges'.
    pub(crate) async fn turn(
        self: &Arc<Self>,
        stopping: &Stopping,
        flush_due: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<MergeBuild>> {
        let flush_first = async {
            while !flush_due() {
                tokio::time::sleep(ASKS_EVERY).await;
            }
        };
        let turn = async {
            tokio::select! {
                biased;
                turn = self.turns.clone().acquire_owned() => Some(turn),
                () = flush_first => None,
            }
        };
        let Some(Some(turn)) = stopping.unless_stopped(MergeStage::Queued, turn).await else {
            return Ok(None);
        };
        let turn = turn.map_err(|error| StorageError::Background(error.to_string()))?;
        let reckoned = Arc::new(Reckoned {
            stop: stopping.stop().clone(),
            bytes: AtomicUsize::new(0),
            planned: AtomicUsize::new(0),
            committing: AtomicBool::new(false),
        });
        lock(&self.under_way).push(reckoned.clone());
        Ok(Some(MergeBuild {
            _turn: turn,
            counted: Counted {
                builds: self.clone(),
                reckoned,
            },
        }))
    }
}

/// One merge's build: its turn, and its count among the builds under way.
/// Shared, as an `Arc`, by every piece of work that holds what the merge
/// gathered or built, and let go with the last.
#[derive(Debug)]
pub(crate) struct MergeBuild {
    _turn: OwnedSemaphorePermit,
    counted: Counted,
}

impl MergeBuild {
    /// Reckon the build to hold `bytes` at its peak, from now on.
    pub(crate) fn reckon(&self, bytes: usize) {
        self.counted.reckoned.bytes.store(bytes, Ordering::SeqCst);
    }

    /// Note the merge past where a stop ends it: what it holds is
    /// committed, not drained, however it is stopped from now on.
    pub(crate) fn past_stopping(&self) {
        self.counted
            .reckoned
            .committing
            .store(true, Ordering::SeqCst);
    }

    /// Wait here, off the async threads, while a test holds builds: see
    /// [`MergeBuilds::hold`]. At most [`HELD_AT_MOST`].
    pub(crate) fn held_here(&self, stopping: &Stopping) {
        let builds = &self.counted.builds;
        let mut holding = lock(&builds.holding);
        if holding.is_none() {
            return;
        }
        builds.held.fetch_add(1, Ordering::SeqCst);
        let until = Instant::now() + HELD_AT_MOST;
        loop {
            let goes_on = match *holding {
                None => true,
                Some(Holding::UntilLetGo) => false,
                Some(Holding::UntilStopped) => stopping.is_stopped(),
                Some(Holding::BuiltUntilLetGo) => true,
            };
            if goes_on || Instant::now() >= until {
                break;
            }
            holding = builds
                .let_go
                .wait_timeout(holding, ASKS_EVERY)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        builds.held.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A build's place among those under way, given up as it is dropped.
#[derive(Debug)]
struct Counted {
    builds: Arc<MergeBuilds>,
    reckoned: Arc<Reckoned>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        let planned = self.reckoned.planned.load(Ordering::SeqCst);
        let bytes = self.reckoned.bytes.load(Ordering::SeqCst);
        self.builds.last_planned.store(planned, Ordering::SeqCst);
        self.builds.last_reckoned.store(bytes, Ordering::SeqCst);
        lock(&self.builds.under_way).retain(|build| !Arc::ptr_eq(build, &self.reckoned));
    }
}

/// `mutex` locked, whether or not a holder panicked: what each guards is
/// whole either way.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
