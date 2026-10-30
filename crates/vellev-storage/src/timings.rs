//! How long a node's background work waits and takes, timed where it
//! happens (merge note, §9): the building lock, by whoever holds it; a
//! merge, phase by phase, and how each ended; and how long rows wait in
//! memory once a flush of them is due, and how long a tenant sits at its
//! backlog. Every writer a node opens records into the node's one
//! [`Timings`]; a writer on its own has one of its own.
//!
//! Each measure is a histogram of seconds, from a millisecond to about 35
//! minutes in buckets twice as wide each time: a hold of the building lock
//! by a large merge's commit, or a spell at the backlog, runs for minutes.
//! A bench that needs each duration exactly, not its bucket, keeps them as
//! they come ([`Timings::keep_events`]).

use std::fmt::Display;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Who holds a partition's building lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Holder {
    /// A flush: from before it copies the memtable's rows to its commit.
    Flush,
    /// A merge planning: from its plan to the end of gathering its rows.
    MergePlan,
    /// A merge committing: its flush along, and its checkpoint.
    MergeCommit,
    /// The curves of segments' rows committed.
    Curves,
    /// Cleaning up the files nothing needs any more.
    Clean,
    /// A restore, in place or from another store's files.
    Restore,
}

impl Holder {
    pub const ALL: [Holder; 6] = [
        Holder::Flush,
        Holder::MergePlan,
        Holder::MergeCommit,
        Holder::Curves,
        Holder::Clean,
        Holder::Restore,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Holder::Flush => "flush",
            Holder::MergePlan => "merge_plan",
            Holder::MergeCommit => "merge_commit",
            Holder::Curves => "curves",
            Holder::Clean => "clean",
            Holder::Restore => "restore",
        }
    }
}

/// A phase of a merge, timed on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergePhase {
    /// Waiting for a turn among the node's merge builds.
    Turn,
    /// Gathering the run's live rows (`merged()`), under the building lock.
    Gather,
    /// Building the merged segment: indexing and encoding it, without the
    /// lock.
    Build,
    /// Uploading its files.
    Upload,
    /// Loading it back, and seeing it fit to adopt.
    Load,
}

impl MergePhase {
    pub const ALL: [MergePhase; 5] = [
        MergePhase::Turn,
        MergePhase::Gather,
        MergePhase::Build,
        MergePhase::Upload,
        MergePhase::Load,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MergePhase::Turn => "turn",
            MergePhase::Gather => "gather",
            MergePhase::Build => "build",
            MergePhase::Upload => "upload",
            MergePhase::Load => "load",
        }
    }
}

/// How a merge that took a turn among the node's merge builds ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MergeEnd {
    /// Committed.
    Committed,
    /// Found nothing to merge once planned again under the building lock.
    Nothing,
    /// Planned at another epoch than it would commit at: a restore came
    /// between.
    Stale,
    /// Its run was not as planned once it came to commit.
    Changed,
    /// A row its run holds live was missing from what it built.
    Uncovered,
    /// Stopped: its tenant let go, its lease lost, or the node stopping.
    Stopped,
    /// Failed with an error.
    Failed,
}

impl MergeEnd {
    pub const ALL: [MergeEnd; 7] = [
        MergeEnd::Committed,
        MergeEnd::Nothing,
        MergeEnd::Stale,
        MergeEnd::Changed,
        MergeEnd::Uncovered,
        MergeEnd::Stopped,
        MergeEnd::Failed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MergeEnd::Committed => "committed",
            MergeEnd::Nothing => "nothing",
            MergeEnd::Stale => "stale",
            MergeEnd::Changed => "changed",
            MergeEnd::Uncovered => "uncovered",
            MergeEnd::Stopped => "stopped",
            MergeEnd::Failed => "failed",
        }
    }
}

/// What is timed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Timed {
    /// Waiting for a partition's building lock.
    BuildingWait(Holder),
    /// Holding it.
    BuildingHeld(Holder),
    /// A phase of a merge.
    Merge(MergePhase),
    /// From a tenant's rows unflushed reaching what a flush is due at to
    /// the commit of the checkpoint that took them out of memory.
    FlushDue,
    /// A spell at the backlog: from a tenant's rows unflushed reaching the
    /// most it may hold to the commit that took it below.
    Backlog,
    /// The flush a node makes before it lets a tenant go: its wait for
    /// the building lock.
    LetGoFlushWait,
    /// The same flush's hold of the lock: its whole build and commit.
    LetGoFlushHeld,
}

impl Timed {
    /// Everything timed, in a fixed order.
    pub fn all() -> Vec<Timed> {
        let mut all = Vec::new();
        all.extend(Holder::ALL.map(Timed::BuildingWait));
        all.extend(Holder::ALL.map(Timed::BuildingHeld));
        all.extend(MergePhase::ALL.map(Timed::Merge));
        all.extend([
            Timed::FlushDue,
            Timed::Backlog,
            Timed::LetGoFlushWait,
            Timed::LetGoFlushHeld,
        ]);
        all
    }

    fn slot(self) -> usize {
        let holders = Holder::ALL.len();
        let phases = MergePhase::ALL.len();
        let index = |holder: Holder| Holder::ALL.iter().position(|h| *h == holder).unwrap();
        match self {
            Timed::BuildingWait(holder) => index(holder),
            Timed::BuildingHeld(holder) => holders + index(holder),
            Timed::Merge(phase) => {
                2 * holders + MergePhase::ALL.iter().position(|p| *p == phase).unwrap()
            }
            Timed::FlushDue => 2 * holders + phases,
            Timed::Backlog => 2 * holders + phases + 1,
            Timed::LetGoFlushWait => 2 * holders + phases + 2,
            Timed::LetGoFlushHeld => 2 * holders + phases + 3,
        }
    }
}

const SLOTS: usize = 2 * Holder::ALL.len() + MergePhase::ALL.len() + 4;

/// The bucket bounds, in seconds: a millisecond, and twice as long each
/// time, to 2,097 seconds, about 35 minutes. Past the last is counted only
/// in the whole.
pub const BOUNDS: [f64; 22] = {
    let mut bounds = [0.0; 22];
    let mut i = 0;
    let mut bound = 0.001;
    while i < bounds.len() {
        bounds[i] = bound;
        bound *= 2.0;
        i += 1;
    }
    bounds
};

/// One histogram of seconds.
#[derive(Debug, Default)]
struct Histogram {
    /// How many fell in each bucket: at most its bound and over the one
    /// before; the last, past every bound. The count is their sum, worked
    /// out as they are read, so that a read made while one is observed
    /// never counts it in the whole and not in a bucket (review of #169).
    buckets: [AtomicU64; BOUNDS.len() + 1],
    micros: AtomicU64,
    /// Where an observe waits, its bucket counted and nothing else yet,
    /// while a test reads it half observed.
    #[cfg(test)]
    paused: Mutex<Option<std::sync::Arc<std::sync::Barrier>>>,
}

impl Histogram {
    fn observe(&self, took: Duration) {
        let seconds = took.as_secs_f64();
        let bucket = BOUNDS
            .iter()
            .position(|bound| seconds <= *bound)
            .unwrap_or(BOUNDS.len());
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        self.paused_here();
        let micros = u64::try_from(took.as_micros()).unwrap_or(u64::MAX);
        self.micros.fetch_add(micros, Ordering::Relaxed);
    }

    /// Wait while a test reads this half observed, if one asked to.
    #[cfg(test)]
    fn paused_here(&self) {
        let paused = self.paused.lock().unwrap().clone();
        if let Some(paused) = paused {
            paused.wait();
            paused.wait();
        }
    }

    fn read(&self) -> Seconds {
        let each: [u64; BOUNDS.len() + 1] =
            std::array::from_fn(|bucket| self.buckets[bucket].load(Ordering::Relaxed));
        let mut cumulative = 0;
        let buckets = std::array::from_fn(|bucket| {
            cumulative += each[bucket];
            (BOUNDS[bucket], cumulative)
        });
        Seconds {
            count: each.iter().sum(),
            sum: self.micros.load(Ordering::Relaxed) as f64 / 1e6,
            buckets,
        }
    }
}

/// A histogram as read: as Prometheus takes one, each bucket counting
/// everything at most its bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seconds {
    pub count: u64,
    pub sum: f64,
    pub buckets: [(f64, u64); BOUNDS.len()],
}

/// One duration as it was timed, kept for a bench (see
/// [`Timings::keep_events`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub timed: Timed,
    pub took: Duration,
    /// The tenant's partition it was timed for.
    pub partition: String,
    /// When it ended.
    pub at: Instant,
}

/// The node's timings: see the module's documentation.
#[derive(Debug)]
pub struct Timings {
    histograms: [Histogram; SLOTS],
    ended: [AtomicU64; MergeEnd::ALL.len()],
    /// Whether a bench keeps events: looked at before anything is made
    /// for one, so that a duration timed while none does takes no lock
    /// and allocates nothing.
    keeping: AtomicBool,
    /// Every duration as it is timed, while a bench keeps them.
    events: Mutex<Option<Vec<Event>>>,
}

impl Default for Timings {
    fn default() -> Self {
        Self {
            histograms: std::array::from_fn(|_| Histogram::default()),
            ended: std::array::from_fn(|_| AtomicU64::new(0)),
            keeping: AtomicBool::new(false),
            events: Mutex::new(None),
        }
    }
}

impl Timings {
    /// Count `took` as `timed`, for `partition`, which is named only for
    /// an event kept: while no bench keeps them, this allocates nothing.
    pub fn record(&self, timed: Timed, partition: &(impl Display + ?Sized), took: Duration) {
        self.histograms[timed.slot()].observe(took);
        if !self.keeping.load(Ordering::Acquire) {
            return;
        }
        let event = Event {
            timed,
            took,
            partition: partition.to_string(),
            at: Instant::now(),
        };
        let mut events = self.events.lock().unwrap_or_else(|held| held.into_inner());
        // Kept until asked otherwise: one timed as keeping stopped is not.
        if let Some(events) = events.as_mut() {
            events.push(event);
        }
    }

    /// Count a merge that ended as `end`.
    pub fn ended(&self, end: MergeEnd) {
        let at = MergeEnd::ALL.iter().position(|e| *e == end).unwrap();
        self.ended[at].fetch_add(1, Ordering::Relaxed);
    }

    /// The histogram of `timed` as it stands.
    pub fn seconds(&self, timed: Timed) -> Seconds {
        self.histograms[timed.slot()].read()
    }

    /// How many merges ended as `end`.
    pub fn merges_ended(&self, end: MergeEnd) -> u64 {
        let at = MergeEnd::ALL.iter().position(|e| *e == end).unwrap();
        self.ended[at].load(Ordering::Relaxed)
    }

    /// Keep every duration as it is timed from now on, until drained, or
    /// stop keeping them. For a bench, which holds its marks to exact
    /// durations rather than to buckets; not for use.
    #[doc(hidden)]
    pub fn keep_events(&self, keep: bool) {
        let mut events = self.events.lock().unwrap_or_else(|held| held.into_inner());
        *events = keep.then(Vec::new);
        self.keeping.store(keep, Ordering::Release);
    }

    /// The durations kept since the last drain, oldest first. For a bench.
    #[doc(hidden)]
    pub fn drain_events(&self) -> Vec<Event> {
        let mut events = self.events.lock().unwrap_or_else(|held| held.into_inner());
        events.as_mut().map(std::mem::take).unwrap_or_default()
    }
}

/// When a tenant's flush fell due and its backlog was reached, worked out
/// as rows are applied and checkpoints committed (merge note, §7.5 and
/// §9.4). Off until a node says what a flush is due at.
#[derive(Debug, Default)]
pub(crate) struct Lag {
    state: Mutex<LagState>,
    /// Where a note of rows applied waits, before it is made, while a test
    /// asks for a checkpoint beside it.
    #[cfg(test)]
    pub(crate) paused: Mutex<Option<std::sync::Arc<std::sync::Barrier>>>,
}

#[derive(Debug, Default)]
struct LagState {
    /// The rows unflushed a flush is due at, and the most a tenant may
    /// hold: zero for none.
    due_at: usize,
    backlog: usize,
    /// When the rows not taken by a checkpoint under way came to `due_at`.
    due: Option<Instant>,
    /// When the rows unflushed came to `backlog`.
    capped: Option<Instant>,
    /// The checkpoint under way, if one is: how many rows it took, and
    /// when the rows it took had fallen due.
    taking: Option<(usize, Option<Instant>)>,
}

impl LagState {
    /// Note the rows unflushed as `unflushed`, at `now`.
    fn saw(&mut self, unflushed: usize, now: Instant) {
        let taken = self.taking.map_or(0, |(taken, _)| taken);
        if self.due_at > 0 && self.due.is_none() && unflushed.saturating_sub(taken) >= self.due_at {
            self.due = Some(now);
        }
        if self.backlog > 0 && self.capped.is_none() && unflushed >= self.backlog {
            self.capped = Some(now);
        }
    }
}

impl Lag {
    fn state(&self) -> std::sync::MutexGuard<'_, LagState> {
        self.state.lock().unwrap_or_else(|held| held.into_inner())
    }

    /// Time flushes from when they fall due at `due_at` rows unflushed,
    /// and spells at a backlog of `backlog`, from now on; the partition
    /// holds `unflushed` now.
    pub(crate) fn due_at(&self, due_at: usize, backlog: usize, unflushed: usize) {
        let mut state = self.state();
        state.due_at = due_at;
        state.backlog = backlog;
        state.saw(unflushed, Instant::now());
    }

    /// Rows were applied, and the partition holds `unflushed`.
    pub(crate) fn applied(&self, unflushed: usize) {
        #[cfg(test)]
        {
            let paused = self.paused.lock().unwrap().take();
            if let Some(paused) = paused {
                paused.wait();
                paused.wait();
            }
        }
        self.state().saw(unflushed, Instant::now());
    }

    /// A checkpoint took the `taken` rows the memtable held: those due are
    /// timed once it is committed. Given up unless committed.
    pub(crate) fn taking(&self, taken: usize) -> Taking<'_> {
        let mut state = self.state();
        let due = state.due.take();
        state.taking = Some((taken, due));
        Taking {
            lag: self,
            committed: false,
        }
    }

    /// The partition was put back wholesale, holding `unflushed`: the rows
    /// that were due are not flushed but undone, and nothing is timed.
    pub(crate) fn put_back(&self, unflushed: usize) {
        let mut state = self.state();
        state.due = None;
        state.capped = None;
        state.taking = None;
        state.saw(unflushed, Instant::now());
    }
}

/// A checkpoint under way that takes the memtable's rows, as
/// [`Lag::taking`] noted it.
pub(crate) struct Taking<'a> {
    lag: &'a Lag,
    committed: bool,
}

impl Taking<'_> {
    /// The checkpoint is committed, and the partition holds `unflushed`:
    /// how long the rows it took waited since they fell due, if they had,
    /// and how long the spell at the backlog it ended lasted, if it ended
    /// one.
    pub(crate) fn committed(mut self, unflushed: usize) -> (Option<Duration>, Option<Duration>) {
        self.committed = true;
        let now = Instant::now();
        let mut state = self.lag.state();
        let due = state.taking.take().and_then(|(_, due)| due);
        let waited = due.map(|due| now.saturating_duration_since(due));
        let capped = match state.capped {
            Some(capped) if unflushed < state.backlog => {
                state.capped = None;
                Some(now.saturating_duration_since(capped))
            }
            _ => None,
        };
        state.saw(unflushed, now);
        (waited, capped)
    }
}

impl Drop for Taking<'_> {
    /// Given up: the rows it took are still unflushed, and due from when
    /// they were.
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut state = self.lag.state();
        if let Some((_, due)) = state.taking.take() {
            state.due = match (due, state.due) {
                (Some(earlier), _) => Some(earlier),
                (None, since) => since,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_measure_has_its_own_slot() {
        let all = Timed::all();
        let slots: std::collections::HashSet<usize> = all.iter().map(|t| t.slot()).collect();
        assert_eq!(slots.len(), all.len());
        assert_eq!(all.len(), SLOTS);
        assert!(slots.iter().all(|slot| *slot < SLOTS));
    }

    #[test]
    fn a_histogram_counts_each_duration_in_the_least_bucket_that_holds_it() {
        let timings = Timings::default();
        let timed = Timed::FlushDue;
        timings.record(timed, "p", Duration::from_micros(500));
        timings.record(timed, "p", Duration::from_millis(3));
        timings.record(timed, "p", Duration::from_secs(4_000));
        let read = timings.seconds(timed);
        assert_eq!(read.count, 3);
        assert_eq!(read.buckets[0], (0.001, 1));
        // 3 ms is over 2 ms and at most 4 ms.
        assert_eq!(read.buckets[1].1, 1);
        assert_eq!(read.buckets[2].1, 2);
        // Over the last bound: in the whole alone.
        assert_eq!(read.buckets[BOUNDS.len() - 1].1, 2);
        assert!((read.sum - 4_000.003_5).abs() < 1e-6);
    }

    /// Review of #169 (MEDIUM 1): a histogram read while a duration is
    /// half observed, its bucket counted and nothing else yet, is whole:
    /// its count is never less than its buckets hold. The observe is held
    /// there while it is read, so that the race is shown every time, not
    /// hoped for among millions of reads.
    #[test]
    fn a_histogram_read_half_observed_is_whole() {
        let timings = std::sync::Arc::new(Timings::default());
        let timed = Timed::Backlog;
        let paused = std::sync::Arc::new(std::sync::Barrier::new(2));
        *timings.histograms[timed.slot()].paused.lock().unwrap() = Some(paused.clone());
        let observing = {
            let timings = timings.clone();
            std::thread::spawn(move || timings.record(timed, "p", Duration::from_millis(3)))
        };
        // Its bucket is counted, and it waits there until read.
        paused.wait();
        let half = timings.seconds(timed);
        paused.wait();
        observing.join().unwrap();
        let in_buckets = half.buckets[BOUNDS.len() - 1].1;
        assert_eq!(in_buckets, 1, "{half:?}");
        assert!(half.count >= in_buckets, "{half:?}");
        let whole = timings.seconds(timed);
        assert_eq!((whole.count, whole.buckets[BOUNDS.len() - 1].1), (1, 1));
    }

    #[test]
    fn events_are_kept_only_while_asked_for() {
        let timings = Timings::default();
        timings.record(Timed::Backlog, "p", Duration::from_millis(1));
        timings.keep_events(true);
        timings.record(Timed::Backlog, "p", Duration::from_millis(2));
        let kept = timings.drain_events();
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].took, Duration::from_millis(2));
        assert!(timings.drain_events().is_empty());
    }
}
