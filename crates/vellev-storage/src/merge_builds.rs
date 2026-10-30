//! The merge builds a node runs at once (merge note, step 3, §7 and §7.7).
//!
//! A merge takes a turn among at most [`MERGE_BUILDS`] before it gathers a
//! row, and reckons what its build holds at its peak. Both are held by
//! everything that holds what the merge gathered or built: the merge
//! itself, its build off the async threads, and every copy of the bytes it
//! uploads, wherever it is. They are let go with the last of those, not
//! when the merge is dropped: a stopped build that runs on, or a store's
//! thread still writing what it uploaded, holds its turn, so that no
//! further merge gathers beside it. What a stopped merge is reckoned to
//! hold counts as draining until then.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use vellev_index::pool::Passes;
use vellev_index::stop::{Stop, Stopped};

use crate::coordinator::PartitionKey;
use crate::stopping::{MergeStage, Stopping};
use crate::Result;

/// How many merges a node builds at once. What a merge gathers, a copy of
/// its run's live rows and their vectors, is the bulk of what it holds, so
/// the node's merges hold at most this many times the largest.
pub const MERGE_BUILDS: usize = 2;

/// Where a build's sections run: on its merge builds' pool, or, without
/// one, on rayon's global pool. A build's top level, its serial glue, its
/// turn to tune and the waits between its sections, stays on the thread
/// that runs the build, never on the pool.
///
/// A pool runs one section at a time, taking turns in the order asked for,
/// and a section waits for its turn on the build's own thread. Were two on
/// the pool at once, a thread of it waiting in one would take the other
/// on, whatever the size of either, and the first could neither end nor
/// heed its stop until the other's had: rayon gives a waiting worker
/// whatever else is queued (review of #169). A build stopped while it
/// waits for a turn gives up at once, leaving its place in the order: one
/// under way runs out, heeding its own stop as it goes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OnPool<'a> {
    /// The merge builds of the build, if it is a merge's, and its stop.
    builds: Option<(&'a MergeBuilds, &'a Stop)>,
}

/// Whether each section's thread is kept, as a test asks: see
/// [`sections_ran_on`].
static RECORDING: AtomicBool = AtomicBool::new(false);
static RAN_ON: Mutex<Vec<SectionRan>> = Mutex::new(Vec::new());

/// A section as [`sections_ran_on`] kept it.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionRan {
    /// Whether the build was a merge's.
    pub merge: bool,
    pub thread: String,
    pub name: &'static str,
}

/// Keep, from now on, the thread each build's section runs on and whether
/// the build was a merge's, or stop keeping them; and give back those kept
/// so far. For tests that show which pool each build's sections run on.
#[doc(hidden)]
pub fn sections_ran_on(keep: bool) -> Vec<SectionRan> {
    RECORDING.store(keep, Ordering::SeqCst);
    std::mem::take(&mut *lock(&RAN_ON))
}

impl<'a> OnPool<'a> {
    /// For a build that is no merge's, a flush's or a load's: its sections
    /// on rayon's global pool.
    pub(crate) fn global() -> Self {
        Self { builds: None }
    }

    /// For a merge's build among `builds`, its sections on their pool if
    /// they have one; giving up waiting for a turn once `stop` is stopped.
    pub(crate) fn merge(builds: &'a MergeBuilds, stop: &'a Stop) -> Self {
        Self {
            builds: Some((builds, stop)),
        }
    }

    /// Run the section `section`, named `name`, where this says; or give
    /// up, not running it, should the build be stopped while it waits.
    pub(crate) fn section<R: Send>(
        self,
        name: &'static str,
        section: impl FnOnce() -> R + Send,
    ) -> std::result::Result<R, Stopped> {
        let builds = self.builds.map(|(builds, _)| builds);
        let noted = move || {
            let _inside = InSection::entered(builds);
            if RECORDING.load(Ordering::SeqCst) {
                lock(&RAN_ON).push(SectionRan {
                    merge: builds.is_some(),
                    thread: std::thread::current().name().unwrap_or("").to_string(),
                    name,
                });
            }
            section()
        };
        // Inside a section already, none begins: it is counted, and runs
        // here, as it could never have its turn behind its own.
        let inside = IN_SECTIONS.with(|inside| inside.get() > 0);
        let on_pool = self
            .builds
            .and_then(|(builds, stop)| Some((builds, stop, builds.pool.as_ref()?)));
        match on_pool.filter(|_| !inside) {
            Some((builds, stop, pool)) => {
                let _turn = builds.section_turn(stop)?;
                Ok(pool.install(noted))
            }
            None => Ok(noted()),
        }
    }
}

impl Passes for OnPool<'_> {
    fn pass<R: Send>(
        self,
        name: &'static str,
        pass: impl FnOnce() -> R + Send,
    ) -> std::result::Result<R, Stopped> {
        self.section(name, pass)
    }
}

std::thread_local! {
    /// How many sections this thread is inside.
    static IN_SECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The tickets of the turns sections take on a merge pool: handed out in
/// the order asked for, the one being served, and those whose builds were
/// stopped waiting, to be passed over.
#[derive(Debug, Default)]
struct Tickets {
    next: u64,
    serving: u64,
    left: BTreeSet<u64>,
}

/// The turns sections take on a merge pool.
#[derive(Debug, Default)]
struct SectionTurns {
    tickets: Mutex<Tickets>,
    turned: Condvar,
}

/// How often a section waiting for its turn asks its build's stop.
const WAITING_ASKS_EVERY: Duration = Duration::from_millis(20);

/// A section's turn on the pool, handed on as it is dropped, the section
/// done or panicked.
struct SectionTurn<'a>(&'a SectionTurns);

impl Drop for SectionTurn<'_> {
    fn drop(&mut self) {
        let mut tickets = lock(&self.0.tickets);
        tickets.serving += 1;
        loop {
            let serving = tickets.serving;
            if !tickets.left.remove(&serving) {
                break;
            }
            tickets.serving += 1;
        }
        drop(tickets);
        self.0.turned.notify_all();
    }
}

/// A section entered on this thread, left as this is dropped: one entered
/// inside another is counted among its merge builds' nested sections.
struct InSection;

impl InSection {
    fn entered(builds: Option<&MergeBuilds>) -> Self {
        let depth = IN_SECTIONS.with(|inside| inside.replace(inside.get() + 1));
        if let Some(builds) = builds.filter(|_| depth > 0) {
            builds.nested_sections.fetch_add(1, Ordering::SeqCst);
        }
        Self
    }
}

impl Drop for InSection {
    fn drop(&mut self) {
        IN_SECTIONS.with(|inside| inside.set(inside.get() - 1));
    }
}

/// The longest a build waits where a test holds it, were the test never to
/// let it go.
const HELD_AT_MOST: Duration = Duration::from_secs(60);

/// How long a turn granted to a merge that gave way to a flush is kept for
/// its next attempt, at the most: past a flush, which takes seconds, and
/// far short of a build, which takes minutes. One not taken up by then, as
/// by a writer whose rounds have stopped, is let go.
const PARKED_AT_MOST: Duration = Duration::from_secs(60);

/// How long what a store keeps of a stopped or failed merge's bytes holds
/// its turn at most: longer than any store's own write of them may take, a
/// cache's local copy of three gigabytes on a disk at 20 MiB a second and
/// its turn for a thread included, so that only a write that will not end
/// is let go of early. Then the turn goes, and what the write holds is no
/// longer counted, said in the log: a writer's merges would otherwise wait
/// behind it for good.
const STORE_HELD_AT_MOST: Duration = Duration::from_secs(10 * 60);

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
    /// See [`Self::held_let_go`].
    held_let_go: AtomicU64,
    /// See [`Self::abandoned`].
    abandoned: AtomicUsize,
    /// How many builds a test has let go one at a time, not yet gone on.
    released: AtomicUsize,
    /// How many rows the gather last to come to a chunk had copied by then.
    last_gathered: AtomicUsize,
    /// How long a turn granted to a merge that gave way to a flush is kept
    /// for its next attempt: see [`Self::park_for`].
    parked_at_most: Mutex<Duration>,
    /// How long what a store keeps of a stopped or failed merge's bytes
    /// holds its turn at most: see [`Self::let_go_held_after`].
    held_at_most: Mutex<Duration>,
    /// What the last build to end was reckoned to hold, as it was planned
    /// and once its rows were gathered.
    last_planned: AtomicUsize,
    last_reckoned: AtomicUsize,
    /// Whether each build's output carries a [`Probe`], as a test asks.
    probing: AtomicBool,
    /// How the builds stood as the last probe was dropped.
    probed: Mutex<Option<Probed>>,
    /// The pool merges build on, if not rayon's global one: see
    /// [`Self::building_on`].
    pool: Option<Arc<rayon::ThreadPool>>,
    /// The most any build was reckoned to hold at its peak, since the node
    /// started.
    largest: AtomicUsize,
    /// How many sections began on a thread already inside one: see
    /// [`Self::nested_sections`].
    nested_sections: AtomicU64,
    section_turns: SectionTurns,
    /// How long sections have waited for their turns, in microseconds.
    sections_waited: AtomicU64,
    /// The CPU builds spent on their own threads, in microseconds.
    top_level_cpu: AtomicU64,
}

/// What one build under way is reckoned to hold, and the stop of its merge.
#[derive(Debug)]
struct Reckoned {
    /// The partition merged.
    key: PartitionKey,
    stop: Stop,
    bytes: AtomicUsize,
    /// What it was reckoned to hold as it was planned.
    planned: AtomicUsize,
    /// Whether its merge is past where a stop ends it: one stopped after
    /// commits what it holds, and drains nothing.
    committing: AtomicBool,
}

/// How a test holds each merge's build off the async threads, with its
/// turn taken. Not for use.
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
    /// Part way through gathering its rows, at each chunk of them it comes
    /// to with some copied, until let go, stopped or not: a long gather.
    GatheringUntilLetGo,
}

/// Where a merge's build asks whether a test holds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeldAt {
    /// With its rows gathered, before any is indexed.
    Start,
    /// Part way through gathering them.
    Gathering,
    /// Once built.
    Built,
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
    ///
    /// # Panics
    ///
    /// If `at_once` is 0: no merge would ever be built.
    pub fn new(at_once: usize) -> Arc<Self> {
        Self::made(at_once, None)
    }

    /// As [`Self::new`], each merge's build run on `pool` in sections, one
    /// at a time across all the builds: every parallel pass, and the long
    /// serial steps, finishing the gathered vectors, encoding and working
    /// out the parts. The rest stays on the build's own thread, off the
    /// pool and at the process's priority: the serial glue between
    /// sections, a clustered index's assembly, and its wait for a turn to
    /// tune, on the one tuning thread. Flushes, loads and what a merge does
    /// around its build stay where they are: on rayon's global pool, and on
    /// the async runtime's threads. A pool smaller, or lower in priority,
    /// than the global one leaves the rest of the box to reads and flushes
    /// while merges build (merge note, §3).
    pub fn building_on(at_once: usize, pool: Arc<rayon::ThreadPool>) -> Arc<Self> {
        Self::made(at_once, Some(pool))
    }

    fn made(at_once: usize, pool: Option<Arc<rayon::ThreadPool>>) -> Arc<Self> {
        assert!(at_once >= 1, "merges are built at least one at a time");
        Arc::new(Self {
            turns: Arc::new(Semaphore::new(at_once)),
            under_way: Mutex::default(),
            holding: Mutex::default(),
            let_go: Condvar::new(),
            held: AtomicUsize::new(0),
            held_let_go: AtomicU64::new(0),
            abandoned: AtomicUsize::new(0),
            released: AtomicUsize::new(0),
            last_gathered: AtomicUsize::new(0),
            parked_at_most: Mutex::new(PARKED_AT_MOST),
            held_at_most: Mutex::new(STORE_HELD_AT_MOST),
            last_planned: AtomicUsize::new(0),
            last_reckoned: AtomicUsize::new(0),
            probing: AtomicBool::new(false),
            probed: Mutex::default(),
            pool,
            largest: AtomicUsize::new(0),
            nested_sections: AtomicU64::new(0),
            section_turns: SectionTurns::default(),
            sections_waited: AtomicU64::new(0),
            top_level_cpu: AtomicU64::new(0),
        })
    }

    /// About how many bytes the largest build since the node started was
    /// reckoned to hold at its peak: what the node's memory, past its
    /// budget, may hold [`MERGE_BUILDS`] times over (merge note, §2.4).
    pub fn largest_reckoned(&self) -> usize {
        self.largest.load(Ordering::SeqCst)
    }

    /// The CPU merge builds have spent on their own threads, off any pool,
    /// since the node started: at the process's priority whatever pool
    /// they build on. Without a pool of their own, that is their sections
    /// too. Nothing where a thread's CPU cannot be read.
    pub fn top_level_cpu(&self) -> Duration {
        Duration::from_micros(self.top_level_cpu.load(Ordering::SeqCst))
    }

    /// Note `spent`, CPU on a build's own thread.
    pub(crate) fn spent_on_top(&self, spent: Duration) {
        let spent = u64::try_from(spent.as_micros()).unwrap_or(u64::MAX);
        self.top_level_cpu.fetch_add(spent, Ordering::SeqCst);
    }

    /// Wait for a section's turn on the pool.
    fn section_turn(&self, stop: &Stop) -> std::result::Result<SectionTurn<'_>, Stopped> {
        let asked = Instant::now();
        let turns = &self.section_turns;
        let mut tickets = lock(&turns.tickets);
        let ticket = tickets.next;
        tickets.next += 1;
        while tickets.serving != ticket {
            if stop.is_stopped() {
                tickets.left.insert(ticket);
                return Err(Stopped);
            }
            tickets = turns
                .turned
                .wait_timeout(tickets, WAITING_ASKS_EVERY)
                .unwrap_or_else(|held| held.into_inner())
                .0;
        }
        drop(tickets);
        let waited = u64::try_from(asked.elapsed().as_micros()).unwrap_or(u64::MAX);
        self.sections_waited.fetch_add(waited, Ordering::SeqCst);
        Ok(SectionTurn(turns))
    }

    /// How long sections of merge builds have waited for their turns on
    /// the pool since the node started. One waits while another build's
    /// section runs, though that may leave threads of the pool idle.
    pub fn sections_waited(&self) -> Duration {
        Duration::from_micros(self.sections_waited.load(Ordering::SeqCst))
    }

    /// How many sections of merge builds began on a thread already inside
    /// a section, since the node started: a thread of the pool taken up
    /// with one build's section while another's waits on it below. None.
    pub fn nested_sections(&self) -> u64 {
        self.nested_sections.load(Ordering::SeqCst)
    }

    /// About how many bytes the merges stopped and not yet ended hold:
    /// what each was reckoned to hold at its peak, until the last of its
    /// work has ended. Merges running unstopped are not counted, nor any
    /// stopped once past where a stop ends it, which commits instead. With
    /// what stores keep of merges whose turns were let go for it
    /// ([`Self::abandoned`]).
    pub fn draining(&self) -> usize {
        let stopped: usize = lock(&self.under_way)
            .iter()
            .filter(|build| build.stop.is_stopped() && !build.committing.load(Ordering::SeqCst))
            .map(|build| build.bytes.load(Ordering::SeqCst))
            .sum();
        stopped + self.abandoned()
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

    /// Let one build held go on, the first to ask, whichever it is. For
    /// tests that hand turns over one at a time.
    #[doc(hidden)]
    pub fn let_one_go(&self) {
        self.released.fetch_add(1, Ordering::SeqCst);
        self.let_go.notify_all();
    }

    /// Whether a build of `key`'s partition is under way. For tests.
    #[doc(hidden)]
    pub fn under_way_for(&self, key: &PartitionKey) -> bool {
        lock(&self.under_way).iter().any(|build| &build.key == key)
    }

    /// Let a turn held by what a store keeps of a stopped or failed merge's
    /// bytes go after `held`, instead of [`STORE_HELD_AT_MOST`]. For tests.
    #[doc(hidden)]
    pub fn let_go_held_after(&self, held: Duration) {
        *lock(&self.held_at_most) = held;
    }

    /// Keep a turn granted to a merge that gave way to a flush for its next
    /// attempt for at most `parked`, instead of [`PARKED_AT_MOST`]. For
    /// tests.
    #[doc(hidden)]
    pub fn park_for(&self, parked: Duration) {
        *lock(&self.parked_at_most) = parked;
    }

    /// How many times what a store kept of a stopped or failed merge's
    /// bytes held its turn so long that the turn was let go.
    pub fn held_let_go(&self) -> u64 {
        self.held_let_go.load(Ordering::SeqCst)
    }

    /// About how many bytes stores still keep of stopped or failed merges
    /// whose turns were let go for it: counted in [`Self::draining`].
    pub fn abandoned(&self) -> usize {
        self.abandoned.load(Ordering::SeqCst)
    }

    /// How many rows the gather last to come to a chunk had copied by
    /// then. For tests that show how soon a gather gives up.
    #[doc(hidden)]
    pub fn last_gathered(&self) -> usize {
        self.last_gathered.load(Ordering::SeqCst)
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

    /// A turn to build the merge of `key`'s partition that `stopping`
    /// stops, once one is free; `None` if the merge is stopped first, or
    /// `flush_due` says its writer's flush is due first, when it never
    /// builds. Its place among those waiting is kept in `kept` should it
    /// give way to a flush: the turn that comes to it meanwhile waits there
    /// for the writer's next attempt, as [`Reservation`] says, and a merge
    /// that queued behind it doesn't take it.
    pub(crate) async fn turn(
        self: &Arc<Self>,
        key: &PartitionKey,
        stopping: &Stopping,
        flush_due: &(dyn Fn() -> bool + Sync),
        kept: &Mutex<Option<Reservation>>,
    ) -> Result<Option<MergeBuild>> {
        let turn = loop {
            let reservation = lock(kept).take().unwrap_or_else(|| Reservation::new(self));
            let flush_first = async {
                while !flush_due() {
                    tokio::time::sleep(ASKS_EVERY).await;
                }
            };
            let waited = async {
                tokio::select! {
                    biased;
                    turn = reservation.granted() => Some(turn),
                    () = flush_first => None,
                }
            };
            match stopping.unless_stopped(MergeStage::Queued, waited).await {
                // Stopped: its place, or the turn granted it, is given up.
                None => return Ok(None),
                // A flush first: its place is kept for its next attempt,
                // unless it was stopped meanwhile, seen under the lock a
                // stop gives the place up under.
                Some(None) => {
                    let mut place = lock(kept);
                    if !stopping.is_stopped() {
                        *place = Some(reservation);
                    }
                    return Ok(None);
                }
                Some(Some(Some(turn))) => break turn,
                // Kept past the park time and let go: asked for afresh.
                Some(Some(None)) => continue,
            }
        };
        let reckoned = Arc::new(Reckoned {
            key: key.clone(),
            stop: stopping.stop().clone(),
            bytes: AtomicUsize::new(0),
            planned: AtomicUsize::new(0),
            committing: AtomicBool::new(false),
        });
        lock(&self.under_way).push(reckoned.clone());
        Ok(Some(MergeBuild {
            counted: Counted {
                builds: self.clone(),
                reckoned,
            },
            _turn: turn,
        }))
    }
}

/// A writer's place among the merges waiting for a turn, asked for on a
/// task of its own, so that a merge that gives way to a flush keeps it:
/// the turn granted to it while the writer flushes waits for the writer's
/// next attempt, idling at most for a flush, which takes seconds where a
/// build takes minutes. A turn granted and not taken up within the park
/// time ([`MergeBuilds::park_for`]) is let go; and the place, or the turn
/// granted it, is given up as this is dropped, as when the writer's
/// merges are stopped, it restores, or it goes. Nothing is left behind.
#[derive(Debug)]
pub(crate) struct Reservation {
    kept: Arc<Kept>,
    asking: tokio::task::JoinHandle<()>,
}

/// Where a [`Reservation`]'s turn waits once granted.
#[derive(Debug, Default)]
struct Kept {
    turn: Mutex<Granted>,
    told: tokio::sync::Notify,
}

#[derive(Debug, Default)]
enum Granted {
    #[default]
    Asking,
    Turn(OwnedSemaphorePermit),
    Gone,
}

impl Reservation {
    /// A place at the back of `builds`' queue.
    fn new(builds: &Arc<MergeBuilds>) -> Self {
        let kept = Arc::new(Kept::default());
        let parked = *lock(&builds.parked_at_most);
        let (turns, granted) = (builds.turns.clone(), kept.clone());
        let asking = tokio::spawn(async move {
            match turns.acquire_owned().await {
                Ok(turn) => *lock(&granted.turn) = Granted::Turn(turn),
                Err(_) => *lock(&granted.turn) = Granted::Gone,
            }
            granted.told.notify_one();
            tokio::time::sleep(parked).await;
            let mut turn = lock(&granted.turn);
            if matches!(*turn, Granted::Turn(_)) {
                // Never taken up: let go.
                *turn = Granted::Gone;
            }
            drop(turn);
            granted.told.notify_one();
        });
        Self { kept, asking }
    }

    /// The turn, once granted; `None` if it was let go first.
    async fn granted(&self) -> Option<OwnedSemaphorePermit> {
        loop {
            {
                let mut turn = lock(&self.kept.turn);
                match std::mem::take(&mut *turn) {
                    Granted::Turn(granted) => {
                        *turn = Granted::Gone;
                        return Some(granted);
                    }
                    Granted::Gone => {
                        *turn = Granted::Gone;
                        return None;
                    }
                    Granted::Asking => *turn = Granted::Asking,
                }
            }
            self.kept.told.notified().await;
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.asking.abort();
        *lock(&self.kept.turn) = Granted::Gone;
    }
}

/// One merge's build: its count among the builds under way, and its turn.
/// Shared, as an `Arc`, by every piece of work that holds what the merge
/// gathered or built, and let go with the last: its count first, then its
/// turn, as fields drop in order, so that the builds under way never
/// count one more than there are turns.
#[derive(Debug)]
pub(crate) struct MergeBuild {
    counted: Counted,
    _turn: OwnedSemaphorePermit,
}

impl MergeBuild {
    /// Reckon the build, as it is planned, to hold `bytes` at its peak.
    pub(crate) fn reckon_planned(&self, bytes: usize) {
        self.counted.reckoned.planned.store(bytes, Ordering::SeqCst);
        self.reckon(bytes);
    }

    /// Reckon the build to hold `bytes` at its peak, from now on.
    pub(crate) fn reckon(&self, bytes: usize) {
        self.counted.reckoned.bytes.store(bytes, Ordering::SeqCst);
        let builds = &self.counted.builds;
        builds.largest.fetch_max(bytes, Ordering::SeqCst);
    }

    /// The node's merge builds this is one of.
    pub(crate) fn builds(&self) -> &Arc<MergeBuilds> {
        &self.counted.builds
    }

    /// A probe for what the build makes to carry, if a test asks: see
    /// [`MergeBuilds::probe`].
    pub(crate) fn probe(&self) -> Option<Probe> {
        let builds = &self.counted.builds;
        builds.probing.load(Ordering::SeqCst).then(|| Probe {
            builds: builds.clone(),
        })
    }

    /// Note the merge past where a stop ends it: what it holds is
    /// committed, not drained, however it is stopped from now on.
    pub(crate) fn past_stopping(&self) {
        self.counted
            .reckoned
            .committing
            .store(true, Ordering::SeqCst);
    }

    /// Note the gather come to a chunk, `rows` copied so far, and wait
    /// here while a test holds gathers part way through them.
    pub(crate) fn gathering(&self, rows: usize, stopping: &Stopping) {
        let builds = &self.counted.builds;
        builds.last_gathered.store(rows, Ordering::SeqCst);
        if rows > 0 {
            self.held_here(stopping, HeldAt::Gathering);
        }
    }

    /// Wait here while a test holds builds `at` this point: see
    /// [`MergeBuilds::hold`]. At most [`HELD_AT_MOST`].
    pub(crate) fn held_here(&self, stopping: &Stopping, at: HeldAt) {
        let builds = &self.counted.builds;
        let mut holding = lock(&builds.holding);
        let here = match *holding {
            None => false,
            Some(Holding::BuiltUntilLetGo) => at == HeldAt::Built,
            Some(Holding::UntilLetGo | Holding::UntilStopped) => at == HeldAt::Start,
            Some(Holding::GatheringUntilLetGo) => at == HeldAt::Gathering,
        };
        if !here {
            return;
        }
        builds.held.fetch_add(1, Ordering::SeqCst);
        let until = Instant::now() + HELD_AT_MOST;
        loop {
            let goes_on = match *holding {
                None => true,
                Some(
                    Holding::UntilLetGo | Holding::BuiltUntilLetGo | Holding::GatheringUntilLetGo,
                ) => false,
                Some(Holding::UntilStopped) => stopping.is_stopped(),
            };
            let one_let_go = !goes_on
                && builds
                    .released
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                        left.checked_sub(1)
                    })
                    .is_ok();
            if goes_on || one_let_go || Instant::now() >= until {
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

/// A merge's share of its build carried by the bytes it uploads: held by
/// every copy of them, wherever it goes, a store's own threads included,
/// and let go with the last, so that what a store still holds of a stopped
/// merge's files holds its turn. Once the merge keeps its files to commit
/// them, what it built is the partition's, committed or, should the commit
/// fail, left for cleaning up, and the share is let go: a store that keeps
/// what it is given, as one in memory does, would otherwise hold the turn
/// for as long as it keeps the file.
///
/// A copy a store's thread keeps for as long as that thread is stuck, as on
/// a local disk that has stopped answering, keeps the turn as long: what it
/// holds is held, and letting the turn go would let another merge gather
/// beside it. Said in the log, a minute on and each minute after, once the
/// merge has given up and only such copies are left.
#[derive(Debug)]
pub(crate) struct Uploading {
    share: Mutex<Option<Arc<MergeBuild>>>,
    /// How many bytes were handed to the store carrying the share.
    bytes: AtomicUsize,
    /// What its build was reckoned to hold, counted as abandoned once the
    /// share was let go as a store held the bytes too long, until the last
    /// copy goes: see [`MergeBuilds::abandoned`].
    abandoned: Mutex<Option<(Arc<MergeBuilds>, usize)>>,
}

impl Drop for Uploading {
    /// The last copy of the bytes is gone: what was abandoned is no more.
    fn drop(&mut self) {
        if let Some((builds, bytes)) = lock(&self.abandoned).take() {
            builds.abandoned.fetch_sub(bytes, Ordering::SeqCst);
        }
    }
}

impl Uploading {
    pub(crate) fn new(build: Arc<MergeBuild>) -> Arc<Self> {
        Arc::new(Self {
            share: Mutex::new(Some(build)),
            bytes: AtomicUsize::new(0),
            abandoned: Mutex::new(None),
        })
    }

    /// `bytes`, carrying the share: the same bytes, not a copy.
    pub(crate) fn carried_by(self: &Arc<Self>, bytes: &Bytes) -> Bytes {
        self.bytes.fetch_add(bytes.len(), Ordering::SeqCst);
        Bytes::from_owner(Carried {
            bytes: bytes.clone(),
            _uploading: self.clone(),
        })
    }

    /// Let the share go, as a store held the bytes too long, counting what
    /// its build was reckoned to hold as abandoned until the last copy of
    /// the bytes goes, so that it is counted against the budget still.
    fn abandon(&self) {
        let Some(build) = lock(&self.share).take() else {
            return;
        };
        let builds = build.counted.builds.clone();
        let bytes = build.counted.reckoned.bytes.load(Ordering::SeqCst);
        builds.abandoned.fetch_add(bytes, Ordering::SeqCst);
        builds.held_let_go.fetch_add(1, Ordering::SeqCst);
        *lock(&self.abandoned) = Some((builds, bytes));
    }

    /// Let the share go, whatever still holds the bytes: the merge keeps
    /// them to commit, when they are the partition's from there, committed
    /// or left for cleaning up; or nothing is left to delete them from a
    /// store that keeps what it is given.
    pub(crate) fn let_go(&self) {
        lock(&self.share).take();
    }

    /// Watch, from now on, as the merge that uploaded the bytes has given
    /// up, what a store still keeps of them, and with them the turn: said in
    /// the log a minute on and each minute after, and let go once kept
    /// longer than [`STORE_HELD_AT_MOST`], when only a write that will not
    /// end can be keeping them.
    pub(crate) fn watched(self: Arc<Self>) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let held_at_most = match &*lock(&self.share) {
            Some(build) => *lock(&build.counted.builds.held_at_most),
            None => return,
        };
        let watching = Arc::downgrade(&self);
        drop(self);
        runtime.spawn(async move {
            let since = tokio::time::Instant::now();
            loop {
                let left = held_at_most.saturating_sub(since.elapsed());
                tokio::time::sleep(STILL_HELD_EVERY.min(left)).await;
                let Some(uploading) = watching.upgrade() else {
                    return;
                };
                if lock(&uploading.share).is_none() {
                    return;
                }
                let bytes = uploading.bytes.load(Ordering::SeqCst);
                if since.elapsed() >= held_at_most {
                    tracing::error!(
                        bytes,
                        "a store has held what a stopped merge uploaded for longer than any \
                         write of it may take: its merge turn is let go, and what it holds is \
                         counted as abandoned until the store lets it go"
                    );
                    uploading.abandon();
                    return;
                }
                tracing::warn!(
                    bytes,
                    "a store still holds what a stopped merge uploaded, and with it one of \
                     the node's merge turns: a write stuck, as on a local disk that has \
                     stopped answering"
                );
            }
        });
    }
}

/// How often a store still holding what a stopped merge uploaded is said
/// in the log.
const STILL_HELD_EVERY: Duration = Duration::from_secs(60);

/// A stopped merge's share in what it uploaded, let go as this is dropped
/// unless the deletes it waits for were done: should nothing be left to do
/// them, as when the runtime goes, a store that keeps what it is given
/// would otherwise hold the turn for good.
#[derive(Debug)]
pub(crate) struct UnlessDeleted(Option<Arc<Uploading>>);

impl UnlessDeleted {
    pub(crate) fn new(uploading: Arc<Uploading>) -> Self {
        Self(Some(uploading))
    }

    /// The deletes are done: what is left of the bytes is a store's own.
    pub(crate) fn deleted(mut self) {
        self.0.take();
    }
}

impl Drop for UnlessDeleted {
    fn drop(&mut self) {
        if let Some(uploading) = self.0.take() {
            uploading.let_go();
        }
    }
}

/// Bytes uploaded, and the share they carry.
struct Carried {
    bytes: Bytes,
    _uploading: Arc<Uploading>,
}

impl AsRef<[u8]> for Carried {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
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

#[cfg(test)]
mod nesting {
    use super::*;

    /// Review of #169 (HIGH): on a merge pool of two threads, a build whose
    /// own work is half a second, in a section of two halves, ends in about
    /// that though a section of three seconds of a second build is asked
    /// for while it waits in its own. Were that queued on the pool, the
    /// thread waiting in the first would take it on, and the first build
    /// would end, or heed its stop, only once the second's section had.
    #[test]
    fn a_build_waiting_in_its_section_never_takes_on_anothers_section() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .thread_name(|n| format!("vellev-merge-{n}"))
                .build()
                .unwrap(),
        );
        let builds = MergeBuilds::building_on(2, pool);
        let stop = Arc::new(Stop::new());
        let first = std::thread::spawn({
            let (builds, stop) = (builds.clone(), stop.clone());
            move || {
                let began = Instant::now();
                OnPool::merge(&builds, &stop)
                    .section("first", || {
                        rayon::join(
                            || std::thread::sleep(Duration::from_millis(50)),
                            || std::thread::sleep(Duration::from_millis(500)),
                        )
                    })
                    .unwrap();
                began.elapsed()
            }
        });
        std::thread::sleep(Duration::from_millis(20));
        let second = std::thread::spawn({
            let (builds, stop) = (builds.clone(), stop.clone());
            move || {
                OnPool::merge(&builds, &stop)
                    .section("second", || std::thread::sleep(Duration::from_secs(3)))
                    .unwrap();
            }
        });
        let took = first.join().unwrap();
        second.join().unwrap();
        assert!(
            took < Duration::from_millis(1_500),
            "the first build took {took:?}, its own work half a second"
        );
        assert_eq!(builds.nested_sections(), 0);
        assert!(builds.sections_waited() > Duration::from_millis(300));
    }

    /// Sections on a pool take turns in the order asked for, each waiting
    /// on the build's own thread, so that one build never starves another
    /// by asking again as its section ends.
    #[test]
    fn sections_on_a_pool_take_turns_in_the_order_asked_for() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        let builds = MergeBuilds::building_on(2, pool);
        let order = Arc::new(Mutex::new(Vec::new()));
        let threads: Vec<_> = (0..3u64)
            .map(|n| {
                let (builds, order) = (builds.clone(), order.clone());
                let thread = std::thread::spawn(move || {
                    OnPool::merge(&builds, &Stop::new())
                        .section("turn", || {
                            std::thread::sleep(Duration::from_millis(100));
                            order.lock().unwrap().push(n);
                        })
                        .unwrap();
                });
                std::thread::sleep(Duration::from_millis(30));
                thread
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2]);
    }

    /// The count can fire: a section begun inside another, on the pool or
    /// off it, is counted, and runs rather than wait for its own turn
    /// behind the one it is in.
    #[test]
    fn a_section_begun_inside_another_is_counted() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        for builds in [MergeBuilds::building_on(2, pool), MergeBuilds::new(2)] {
            let stop = Stop::new();
            let on = OnPool::merge(&builds, &stop);
            let inner = on.section("outer", || on.section("inner", || 7));
            assert_eq!(inner.unwrap().unwrap(), 7);
            assert_eq!(builds.nested_sections(), 1);
            on.section("after", || ()).unwrap();
            assert_eq!(builds.nested_sections(), 1, "the count left behind");
        }
    }

    /// A build stopped while it waits for a turn gives up at once, not
    /// holding what it gathered and its turn among the merge builds until
    /// the section under way is done, and the build behind it is not left
    /// waiting on the ticket it gave up (review of #169).
    #[test]
    fn a_build_stopped_waiting_for_a_turn_gives_up_and_holds_nobody_up() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        );
        let builds = MergeBuilds::building_on(2, pool);
        let first = std::thread::spawn({
            let builds = builds.clone();
            move || {
                OnPool::merge(&builds, &Stop::new())
                    .section("first", || std::thread::sleep(Duration::from_millis(900)))
                    .unwrap();
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        let stop = Arc::new(Stop::new());
        let second = std::thread::spawn({
            let (builds, stop) = (builds.clone(), stop.clone());
            move || {
                let asked = Instant::now();
                let ran = OnPool::merge(&builds, &stop).section("second", || ());
                (ran, asked.elapsed())
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn({
            let builds = builds.clone();
            move || {
                let ran = OnPool::merge(&builds, &Stop::new()).section("third", || ());
                done.send(ran).unwrap();
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        stop.stop();
        let (ran, took) = second.join().unwrap();
        assert_eq!(ran, Err(Stopped), "it ran, stopped");
        assert!(
            took < Duration::from_millis(500),
            "it held on for {took:?}, with a section of 900 ms ahead of it"
        );
        first.join().unwrap();
        let third = finished.recv_timeout(Duration::from_secs(5));
        assert_eq!(
            third.expect("the build behind it waits on the ticket it gave up"),
            Ok(())
        );
    }
}
