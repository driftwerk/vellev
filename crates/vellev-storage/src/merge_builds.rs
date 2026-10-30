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

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use vellev_index::stop::Stop;

use crate::coordinator::PartitionKey;
use crate::stopping::{MergeStage, Stopping};
use crate::Result;

/// How many merges a node builds at once. What a merge gathers, a copy of
/// its run's live rows and their vectors, is the bulk of what it holds, so
/// the node's merges hold at most this many times the largest.
pub const MERGE_BUILDS: usize = 2;

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
    ///
    /// # Panics
    ///
    /// If `at_once` is 0: no merge would ever be built.
    pub fn new(at_once: usize) -> Arc<Self> {
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
            parked_at_most: Mutex::new(PARKED_AT_MOST),
            held_at_most: Mutex::new(STORE_HELD_AT_MOST),
            last_planned: AtomicUsize::new(0),
            last_reckoned: AtomicUsize::new(0),
            probing: AtomicBool::new(false),
            probed: Mutex::default(),
        })
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

    /// Wait here, off the async threads, while a test holds builds at
    /// their start, or once `built`: see [`MergeBuilds::hold`]. At most
    /// [`HELD_AT_MOST`].
    pub(crate) fn held_here(&self, stopping: &Stopping, built: bool) {
        let builds = &self.counted.builds;
        let mut holding = lock(&builds.holding);
        let here = match *holding {
            None => false,
            Some(Holding::BuiltUntilLetGo) => built,
            Some(Holding::UntilLetGo | Holding::UntilStopped) => !built,
        };
        if !here {
            return;
        }
        builds.held.fetch_add(1, Ordering::SeqCst);
        let until = Instant::now() + HELD_AT_MOST;
        loop {
            let goes_on = match *holding {
                None => true,
                Some(Holding::UntilLetGo | Holding::BuiltUntilLetGo) => false,
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
