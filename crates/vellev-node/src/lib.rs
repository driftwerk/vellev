//! One node's part in the database: the tenants it writes for, each opened
//! when first asked for and written to through one shared queue.

use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use tokio::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, SubsecRound, Utc};
use futures::future::join_all;
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, watch, OnceCell};
use tokio::task::AbortHandle;
use vellev_core::{Embedding, ModelFault, NodeId, Partition, RequestId, TenantId};
use vellev_embed::{EmbedError, Embedders};
use vellev_storage::blobs::Blobs;
use vellev_storage::coordinator::{
    Acquired, Clock, Coordinator, CoordinatorError, Generation, Lease, Lineage, LogPosition,
    ManifestRef, PartitionKey, Point, Restore, RestorePoint, Restored, SystemClock, TenantKey,
    Versions, MAX_RESTORE_WINDOW,
};
use vellev_storage::format::{DEFAULT_MAX_FORMAT, FORMAT_VERSION};
use vellev_storage::timings::{Timed, Timings};
use vellev_storage::{
    Batch, Embedded, GroupCommitter, MergeBuilds, Mutation, Receipt, State, StorageError,
    Transfers, Unembedded, Vectorize, Writer, MERGE_BUILDS,
};

mod backing;
pub use backing::LANDED_AFTER;
mod claim_clock;
mod keys;
mod limits;
mod purging;
mod reading;
mod warm;

use keys::Keys;
use limits::Asked;
use limits::Limits;

pub use claim_clock::{BootClock, ClaimClock};
pub use keys::{Issued, KeyLookups, REMEMBERED_FOR as KEYS_REMEMBERED_FOR};
pub use purging::{LOOKED_FOR_EVERY as PURGES_LOOKED_FOR_EVERY, LOOKED_OVER_EVERY, REMEMBERED};
pub use reading::{Freshness, Reader};

/// The node-to-node protocol this release speaks: what a request one node
/// sends on to another may hold. A release that lets requests hold
/// something new raises it, and a node only sends that on once every node
/// announces the new one.
pub const PROTOCOL: u16 = 1;

/// A key nothing is kept under, asked for to tell that object storage
/// answers: that it is missing is the answer.
const READINESS: &str = "vellev-readiness";

/// How many commits and restores of a removed tenant are forgotten at once.
const FORGOTTEN_AT_ONCE: usize = 10_000;

/// The least [`Settings::grace`] a node runs with, however quick its store:
/// [`Settings::check`] refuses less, and less than three requests to object
/// storage ([`RequestBounds::least_grace`]).
///
/// The grace is all that stands between cleaning up and a restore to a
/// point of the window: nothing else keeps a file such a restore names
/// from being deleted. This floor is a stop-gap under that margin, not a
/// proof of it. As a node is set up by default, and with every clock
/// right, the grace must cover
///
/// - the time a head file is in the store before its commit: its upload,
///   one request to the store while the file is of one part, plus its
///   copy to local disk, which is made once the store has it, plus three
///   tries to commit, each a quarter of [`Settings::lease`] or the
///   coordinator's bound if that is shorter, and the waits between them.
///   Under four minutes by default, 3 minutes 51 seconds for a file of a
///   whole part: 3 minutes 7 seconds for the request, 21.6 seconds for
///   the disk, 22.5 for the tries and 150 ms between them;
/// - the time a restore that has found its target waits for a clean-up
///   under way to settle what it deletes: a minute at most, in which the
///   clean-up reads the clock and makes its list. The clean-up may keep
///   the restore waiting for a good deal longer than that, as it deletes
///   for a minute more and waits before each lot behind any publish of
///   the log under way, but it deletes nothing that was not on the list;
/// - what the node's clock, the store's and the database's differ by,
///   seconds on machines that keep time.
///
/// That is about five minutes by default, and this about three times it.
/// A reader that took a committed position and has yet to fetch its files
/// has the tenant's window to do so in as well as the grace, and each
/// file is given two requests a part before it is given up
/// ([`vellev_storage::Transfers::read_within`]): 6 minutes 14 seconds for
/// a file of one part by default, and as much again for each part more,
/// so that under a short window of the tenant's own a segment of several
/// parts may take longer to fetch than this floor keeps it.
///
/// The floor follows the store's bound and nothing else, so it does
/// **not** cover:
///
/// - a long lease with a long bound on the coordinator: an hour's lease
///   and fifteen minutes for the database let a head file wait through
///   forty-five minutes of tries for its commit, after its upload, and a
///   grace of fifteen is still taken;
/// - a head file of more than one part (16 MiB by default), whose upload
///   may take as many requests as it has parts and three more: one write
///   of that size is past fifteen minutes on its own, by default;
/// - a clock wrong by more than the grace;
/// - a flush or a merge that keeps a restore waiting for longer than the
///   grace.
///
/// Until cleaning up is fenced by the coordinator's rows instead of by
/// time (DW-681, with the clean floor of DW-668), raise the grace with the
/// lease and the coordinator's bound, and with both, as raising one alone
/// may change nothing where the other is the lesser: it should exceed one
/// request to object storage plus three times the lesser of a quarter of
/// the lease and the coordinator's bound, plus some twenty seconds for
/// the file's copy to local disk, with room to spare.
pub const LEAST_GRACE: Duration = Duration::from_secs(15 * 60);

/// How long object storage having answered counts for readiness, so that
/// health checks do not ask it every few seconds.
const STORE_ANSWERED: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct Settings {
    pub node: NodeId,
    /// How long the node's hold on a tenant lasts without being renewed.
    /// It is renewed every third of this while the tenant is open.
    pub lease: Duration,
    /// The most callers' writes that share one commit.
    pub group: usize,
    /// The most writes that wait for a commit before callers are held back.
    pub queue: usize,
    /// A tenant with this many rows written since its last checkpoint is
    /// flushed.
    pub flush_rows: usize,
    /// The most rows a tenant holds in memory unflushed: once it holds
    /// them, a write is refused, [`NodeError::Backlogged`], until a flush
    /// goes through. They pile up while flushes fail, as one does whose
    /// segment's vectors cannot be kept on local disk, or while writes come
    /// faster than flushes keep up with. The bound is checked as a write
    /// arrives, so the writes let through at once, each up to a whole
    /// batch, may each add to what is held: it is passed by at most what
    /// those carry. `None` is four times `flush_rows`; one below
    /// `flush_rows` is refused by [`Settings::check`], as no flush would
    /// ever be asked for and every write would be refused.
    pub backlog_rows: Option<usize>,
    /// How long after a failed flush the tenant's next is tried, at the
    /// least: twice as long after each that fails in a row, up to 64 times
    /// this, and from this again once one goes through.
    pub flush_backoff: Duration,
    /// How far back a tenant may be restored, unless it says otherwise; at
    /// most [`MAX_RESTORE_WINDOW`]. What a restore that far back needs is
    /// kept, every log among it. A tenant open here is looked after every
    /// 24th of this, at least hourly and at most once a minute.
    pub restore_window: Duration,
    /// How much longer than its restore window a file nothing reaches any
    /// more is kept: files are written a little before they are committed,
    /// and a reader that took a committed position still finds its files.
    /// At least [`LEAST_GRACE`] and three requests to object storage, as
    /// [`Settings::check`] makes sure: it is the whole margin between
    /// cleaning up and a restore to a point of the window. That floor does
    /// not follow the lease or the coordinator's bound: see
    /// [`LEAST_GRACE`] for what it leaves to whoever sets this.
    pub grace: Duration,
    /// How often a copy of a tenant kept for reads looks for new commits.
    pub follow_every: Duration,
    /// How long a copy of a tenant kept for reads is kept unused.
    pub follow_idle: Duration,
    /// The most copies of tenants kept for reads at once; the least
    /// recently used go first.
    pub followers: usize,
    /// Where the node keeps the tenants it has lately served, to open
    /// copies of them as it starts, before anyone reads them (burst plan
    /// F7); none, to keep no such list.
    pub warm: Option<std::path::PathBuf>,
    /// The most tenants the node keeps in that list, and opens as it
    /// starts.
    pub warm_most: usize,
    /// The newest file format this node raises the cluster's to, whatever
    /// every node reads. A release that adds a format keeps this at the one
    /// before, so that rolling back stays possible until an operator raises
    /// it. Once the cluster writes a format, raised by any node, this node
    /// writes it too, as far as it reads it.
    pub max_format: u16,
    /// How long a tenant this node writes for is kept unused before it is
    /// let go, its memory given back and its lease left for any node.
    pub tenant_idle: Duration,
    /// How long a tenant's maintenance goes, at most, without cleaning up
    /// and making the curves of its segments' rows, though each round
    /// gives way to a flush: under writes that never let up, rounds would
    /// otherwise never get that far.
    pub clean_at_least_every: Duration,
    /// The most queries a second one tenant may ask of this node, if any.
    pub queries_per_second: Option<f64>,
    /// The most objects a second one tenant may write through this node,
    /// if any.
    pub writes_per_second: Option<f64>,
    /// About how many bytes the tenants this node holds, written for or
    /// copied for reads, may take in memory together, if there is a bound.
    /// Past it the least recently used that nobody is using go first.
    pub memory: Option<usize>,
    /// Where tenants are backed up to, if anywhere.
    pub backups: Option<Arc<dyn Blobs>>,
    /// How often each tenant is backed up.
    pub backup_every: Duration,
    /// Whether the node looks for tenants due a backup of its own accord
    /// (see [`Node::backing_up`]). Without it a tenant is backed up when
    /// that is asked for and not otherwise, by [`Node::back_up`] or
    /// [`Node::back_up_due`]: for whoever times backups itself, as a test
    /// does that counts them.
    pub backups_scheduled: bool,
    /// How long a backup is kept; the newest of a tenant always is.
    pub backups_kept: Duration,
    /// How long a claim to back a tenant up or restore it lasts unless it
    /// is renewed, as it is every third of that while the work runs: a node
    /// gone, or a claim it never heard it took, blocks the tenant for no
    /// longer. The node takes it for lost a third of it before it runs out,
    /// a margin that must cover a request to object storage already sent,
    /// as [`Settings::check`] makes sure.
    pub backup_claim: Duration,
    /// How long one request to object storage, retries and all, and one
    /// request to the coordinator, may take to land or fail.
    pub request_bounds: RequestBounds,
    /// The clock backup claims are measured on, which counts the time the
    /// machine was suspended, as the coordinator's does.
    pub claim_clock: Arc<dyn ClaimClock>,
    /// What tells the time of day: when a backup is taken, how old one is
    /// when backups are pruned, and how old a tenant's file is when what
    /// nothing needs any more is cleaned up. The machine's, unless told
    /// otherwise.
    pub clock: Arc<dyn Clock>,
    /// How often the node looks for tenants due a backup, the first time
    /// at a moment of its own within as long, so that nodes started
    /// together look at different times.
    pub backups_looked_for_every: Duration,
    /// How often the node looks for removed tenants to purge, and for
    /// those purged lately to look over again.
    pub purges_looked_for_every: Duration,
    /// How long after a tenant is removed everything kept of it is
    /// deleted: long enough for any node that held it to notice and stop
    /// committing. Where tenants are backed up it is no sooner than
    /// [`Settings::backup_claim`], so that no backup of the tenant still
    /// holds its claim. A file written for the tenant later still is
    /// deleted when a node next looks the tenant over, for a day after.
    pub purge_after: Duration,
    /// How files go to and come from object storage, and when a request is
    /// given up.
    pub transfers: Transfers,
    /// The tenants this node serves. A request for any other is refused
    /// as [`NodeError::NotServedHere`], read or write, and is sent on to
    /// no other node: the node never holds, follows or backs up a tenant
    /// it does not serve. Every tenant, unless told otherwise.
    pub served: Served,
    /// Where merges are built: on rayon's global pool, with flushes and
    /// loads, unless told otherwise (merge note, §3).
    pub merge_pool: MergePool,
}

/// The threads a node builds its merges on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MergePool {
    /// Rayon's global pool, which builds flushes and loads too, at the
    /// process's priority.
    #[default]
    Global,
    /// A pool of their own of `threads` threads, at least one, each run
    /// `nice` lower than the rest of the process, on Linux, from 0, which
    /// runs them as it does, to 19, as [`Settings::check`] makes sure. A
    /// build's clustered indexes tune on the one tuning thread whatever
    /// this says.
    Own { threads: usize, nice: i32 },
}

impl MergePool {
    /// Why a node cannot build its merges on this pool, if it cannot: one
    /// of no threads builds nothing, and a nice outside 0 to 19 is either
    /// a priority above the process's, which a merge never needs and the
    /// image's user may not take, or below the lowest there is.
    pub fn problem(&self) -> Option<String> {
        match *self {
            MergePool::Global => None,
            MergePool::Own { threads: 0, .. } => {
                Some("a merge pool of no threads would build no merge".into())
            }
            MergePool::Own { nice, .. } if !(0..=19).contains(&nice) => Some(format!(
                "a nice of {nice} for the merge pool is outside 0 to 19: a merge pool runs \
                 at the process's priority or lower, never higher"
            )),
            MergePool::Own { .. } => None,
        }
    }
}

/// The tenants a node serves.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Served {
    /// Every tenant, as a node did before it could be told otherwise.
    #[default]
    Every,
    /// These tenants and no other.
    Only(BTreeSet<TenantId>),
}

impl Served {
    /// Whether `tenant` is one of these.
    pub fn serves(&self, tenant: &TenantId) -> bool {
        match self {
            Served::Every => true,
            Served::Only(listed) => listed.contains(tenant),
        }
    }
}

impl Settings {
    pub fn new(node: NodeId) -> Self {
        Self {
            node,
            lease: Duration::from_secs(30),
            group: 64,
            queue: 1_024,
            flush_rows: 50_000,
            backlog_rows: None,
            flush_backoff: Duration::from_secs(1),
            restore_window: Duration::from_secs(14 * 24 * 60 * 60),
            grace: Duration::from_secs(60 * 60),
            follow_every: Duration::from_secs(1),
            follow_idle: Duration::from_secs(5 * 60),
            followers: 64,
            warm: None,
            warm_most: 16,
            max_format: DEFAULT_MAX_FORMAT,
            tenant_idle: Duration::from_secs(15 * 60),
            clean_at_least_every: Duration::from_secs(10 * 60),
            queries_per_second: None,
            writes_per_second: None,
            memory: None,
            backups: None,
            backup_every: Duration::from_secs(24 * 60 * 60),
            backups_scheduled: true,
            backups_kept: Duration::from_secs(35 * 24 * 60 * 60),
            backup_claim: Duration::from_secs(15 * 60),
            request_bounds: RequestBounds::default(),
            claim_clock: Arc::new(BootClock),
            clock: Arc::new(SystemClock),
            backups_looked_for_every: Duration::from_secs(10 * 60),
            purges_looked_for_every: PURGES_LOOKED_FOR_EVERY,
            purge_after: Duration::from_secs(10 * 60),
            transfers: Transfers::default(),
            served: Served::Every,
            merge_pool: MergePool::Global,
        }
    }

    /// Refuse settings a node cannot run by, saying why: the grace first,
    /// if more than one thing is wrong. [`Node::new`] does not ask this:
    /// whoever makes a node from settings of their own calls it, or runs
    /// without the floor under the grace and the check of the backup
    /// claim.
    pub fn check(&self) -> Result<()> {
        self.request_bounds
            .grace_problem(self.grace)
            .or_else(|| self.request_bounds.claim_problem(self.backup_claim))
            .or_else(|| self.backlog_problem())
            .or_else(|| self.merge_pool.problem())
            .map_or(Ok(()), |problem| Err(NodeError::Settings(problem)))
    }

    /// The most rows a tenant holds unflushed, as these settings say.
    pub fn backlog(&self) -> usize {
        self.backlog_rows
            .unwrap_or(self.flush_rows.saturating_mul(4))
    }

    /// Why a tenant could not be written to with these settings' backlog,
    /// if it could not: one below `flush_rows` refuses every write before
    /// a flush is ever asked for, and one of no time at all between
    /// flushes tried after a failure tries them back to back.
    pub fn backlog_problem(&self) -> Option<String> {
        if self.backlog() < self.flush_rows.max(1) {
            return Some(format!(
                "a backlog of {} rows is below the {} a tenant is flushed at, so no flush \
                 would be asked for and every write would be refused",
                self.backlog(),
                self.flush_rows
            ));
        }
        (self.flush_backoff.is_zero())
            .then(|| "a flush backoff of no time would try failed flushes back to back".into())
    }
}

/// How long one request may take to land or fail: to object storage,
/// retries and all, so that a delete already sent lands within it; and to
/// the coordinator, each attempt of which the node gives up on after it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestBounds {
    pub store: Duration,
    pub coordinator: Duration,
}

impl RequestBounds {
    /// The least [`Settings::grace`] a node runs with as these bounds say:
    /// [`LEAST_GRACE`], or three requests to object storage if those take
    /// longer.
    pub fn least_grace(&self) -> Duration {
        LEAST_GRACE.max(self.store * 3)
    }

    /// Why a node cannot run with a grace of `grace` and requests that
    /// take as long as these bounds say, if it cannot: it is below
    /// [`Self::least_grace`].
    pub fn grace_problem(&self, grace: Duration) -> Option<String> {
        let least = self.least_grace();
        (grace < least).then(|| {
            format!(
                "a grace of {grace:?} is too short to keep the files a restore within a \
                 tenant's window needs from being cleaned up: it must be at least {least:?}, \
                 the longer of {LEAST_GRACE:?} and three requests to object storage, {:?} each",
                self.store
            )
        })
    }
}

impl Default for RequestBounds {
    /// What one request to object storage may take as the store client is
    /// set up unless told otherwise ([`vellev_storage::Transfers`]), and a
    /// coordinator request of ten seconds.
    fn default() -> Self {
        Self {
            store: vellev_storage::Transfers::default().request_within,
            coordinator: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("the node's settings cannot be run by: {0}")]
    Settings(String),
    #[error("tenant {0} does not exist")]
    UnknownTenant(TenantId),
    #[error("tenant {tenant} is written by {holder}")]
    Elsewhere { tenant: TenantId, holder: NodeId },
    #[error("the node is stopping")]
    Stopped,
    #[error("the node is looking up too many keys at once")]
    Busy,
    /// Looking a key up failed; each ask waiting on the lookup is told.
    #[error("looking the key up failed: {0}")]
    KeyLookup(Arc<StorageError>),
    /// Looking a key up panicked; each ask waiting on the lookup is told,
    /// and the key is looked up again on the next ask.
    #[error("looking the key up failed in the node")]
    KeyLookupPanicked,
    #[error("tenant {tenant} is asking too much of this node; try again in {wait:?}")]
    Limited { tenant: TenantId, wait: Duration },
    /// The tenant holds as many rows unflushed as it may
    /// ([`Settings::backlog_rows`]), as its flushes fail or fall behind,
    /// and takes no more until one goes through.
    #[error(
        "tenant {tenant} holds as many rows unflushed as it may, and takes no more \
         until a flush goes through; try again in {wait:?}"
    )]
    Backlogged { tenant: TenantId, wait: Duration },
    #[error("this node's hold on tenant {0} was lost; ask again")]
    Lost(TenantId),
    #[error("this node has not caught up with tenant {0} as far as asked")]
    Behind(TenantId),
    #[error("the point asked for is past anything committed to tenant {0}")]
    Ahead(TenantId),
    #[error("tenant {0} has no state to go back to at that point")]
    NoRestorePoint(TenantId),
    #[error("no backup store is configured")]
    NoBackups,
    #[error("tenant {0} is being backed up or restored from a backup already")]
    BackingUp(TenantId),
    #[error("tenant {0} is being restored already, as another request asked")]
    Restoring(TenantId),
    #[error("request {request} asked for another restore of tenant {tenant} before")]
    Reused {
        tenant: TenantId,
        request: RequestId,
    },
    #[error("tenant {0} has nothing committed to back up")]
    Empty(TenantId),
    #[error(
        "tenant {tenant} was removed and what was kept of it is being deleted; it can be \
         made again, empty, once that is done: ask again in {wait:?}"
    )]
    Purging { tenant: TenantId, wait: Duration },
    #[error(
        "tenant {0} has keys, issued for whatever was to be put in it; revoke them before \
         it takes up a backup"
    )]
    Keyed(TenantId),
    #[error(
        "tenant {0} has been written to here, and is not the tenant that backup is of; \
         only a tenant this cluster has lost is taken up again from a backup"
    )]
    NotLost(TenantId),
    #[error("the claim to back tenant {0} up or restore it was lost before the work was done")]
    ClaimLost(TenantId),
    #[error("tenant {0} has no backup {1}")]
    UnknownBackup(TenantId, uuid::Uuid),
    /// The node is not one that serves the tenant: see
    /// [`Settings::served`]. Nothing was done, and nothing is sent on.
    #[error("tenant {tenant} is not served by this node; ask a node that serves it")]
    NotServedHere { tenant: TenantId },
    #[error("the cluster writes file format {cluster}, and this release reads only up to {reads}")]
    Outdated { cluster: u16, reads: u16 },
    #[error(transparent)]
    Storage(#[from] StorageError),
    #[error(transparent)]
    Embed(#[from] EmbedError),
    /// A collection was to be made with a model no build of this release
    /// runs.
    #[error("{0}")]
    NoSuchModel(String),
}

impl From<CoordinatorError> for NodeError {
    fn from(error: CoordinatorError) -> Self {
        Self::Storage(error.into())
    }
}

pub type Result<T> = std::result::Result<T, NodeError>;

fn key(tenant: &TenantId) -> PartitionKey {
    PartitionKey {
        tenant: tenant.clone(),
        partition: Partition::DEFAULT,
    }
}

/// How far back `tenant` may be restored: `default` unless it says
/// otherwise.
async fn restore_window(
    coordinator: &dyn Coordinator,
    tenant: &TenantId,
    default: Duration,
) -> Result<Duration> {
    Ok(coordinator
        .restore_window(tenant)
        .await?
        .unwrap_or(default)
        .min(MAX_RESTORE_WINDOW))
}

/// The node's embedders, as a writer asks for vectors.
#[derive(Debug)]
struct Documents(Arc<Embedders>);

#[async_trait]
impl Vectorize for Documents {
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, Unembedded> {
        let embedder = self
            .0
            .embedder(embedding)
            .map_err(unembedded)?
            .ok_or_else(|| {
                Unembedded::for_good(
                    ModelFault::Setup,
                    "its collection's clients supply the vectors",
                )
            })?;
        let vectors = embedder.documents(&texts).await.map_err(unembedded)?;
        vectors
            .into_iter()
            .collect::<Option<_>>()
            .ok_or_else(|| Unembedded::for_now("no vector for a text that is not blank"))
    }
}

/// An embedding failure as a writer is told it.
fn unembedded(error: EmbedError) -> Unembedded {
    Unembedded {
        reason: error.to_string(),
        lasting: error.lasting(),
    }
}

/// A tenant this node writes for.
#[derive(Debug)]
pub struct Tenant {
    id: TenantId,
    writer: Arc<Writer>,
    commits: GroupCommitter,
    embedders: Arc<Embedders>,
    flush_rows: usize,
    /// See [`Settings::backlog_rows`].
    backlog_rows: usize,
    /// See [`Settings::flush_backoff`].
    flush_backoff: Duration,
    /// See [`Settings::clean_at_least_every`].
    clean_at_least_every: Duration,
    /// When it was last cleaned up all through, or opened.
    cleaned_at: Mutex<Instant>,
    /// Failed flushes in a row, and when the next may be tried.
    backoff: Mutex<(u32, Option<Instant>)>,
    /// The node's counts of flushes failed and writes refused for them.
    flushing: Arc<FlushCounts>,
    flushes: mpsc::Sender<()>,
    /// Whether a flush or merge is due or under way: see [`Node::unsettled`].
    settling: Settling,
    /// Lost once the writer can no longer write or the node lets the tenant
    /// go, so that the node opens the tenant afresh for the next caller.
    hold: Arc<Hold>,
    /// When the hold on the tenant runs out unless renewed, by this node's
    /// own clock.
    deadline: Arc<Mutex<Instant>>,
    /// When the tenant was last asked for.
    used: Mutex<Instant>,
    limits: Arc<Limits>,
    /// The restore running here, if any.
    running: Mutex<Option<Running>>,
}

/// What a tenant's maintenance has done, to tell whether a flush or merge
/// is due or under way.
#[derive(Debug, Default)]
struct Settling {
    /// Flushes that went through.
    flushed: AtomicU64,
    /// [`Self::flushed`] as it stood after the flush, if any, of the last
    /// round that went through. A round merges until a merge finds nothing
    /// to do, so once it has followed every flush, no merge is due.
    maintained: AtomicU64,
}

/// A node's hold on a tenant, as everything that works on the tenant sees
/// it. Once it is lost, for whatever reason, the tasks working on the
/// tenant are stopped, so that none stuck on something that never answers
/// goes on holding the tenant's data in memory.
#[derive(Debug, Default)]
struct Hold {
    lost: AtomicBool,
    /// The tenant's queue of writes, once it is open.
    commits: OnceLock<GroupCommitter>,
    /// The tenant's writer, once it is open.
    writer: OnceLock<Weak<Writer>>,
    tasks: Mutex<Vec<AbortHandle>>,
}

impl Hold {
    fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// Lose the hold, stopping every task working on the tenant: its
    /// writes, a commit under way being answered as unknown, and whatever
    /// maintains or restores it. A merge under way is stopped first: its
    /// build, which an aborted task would leave running, gives up and lets
    /// go of what it holds (merge note, step 3). All of it under the tasks'
    /// lock, which [`Self::opened`] takes too, so that writes and a writer
    /// set as the tenant opens are stopped whichever comes first.
    fn lose(&self) {
        let mut tasks = self.tasks.lock().expect("tasks lock");
        self.lost.store(true, Ordering::SeqCst);
        if let Some(commits) = self.commits.get() {
            commits.stop();
        }
        if let Some(writer) = self.writer.get().and_then(Weak::upgrade) {
            writer.stop_merges();
        }
        std::mem::take(&mut *tasks)
            .iter()
            .for_each(AbortHandle::abort);
    }

    /// Note the tenant's `writer` and queue of `commits` as it opens:
    /// stopped at once if the hold was lost meanwhile, as losing it would
    /// have stopped them.
    fn opened(&self, writer: &Arc<Writer>, commits: &GroupCommitter) {
        let _tasks = self.tasks.lock().expect("tasks lock");
        self.writer
            .set(Arc::downgrade(writer))
            .expect("a tenant's writer is opened once");
        self.commits
            .set(commits.clone())
            .expect("a tenant's writes are queued once");
        if self.lost.load(Ordering::SeqCst) {
            commits.stop();
            writer.stop_merges();
        }
    }

    /// Have `task` stopped when the hold is lost, at once if it is already.
    fn stops(&self, task: AbortHandle) {
        let mut tasks = self.tasks.lock().expect("tasks lock");
        match self.is_lost() {
            true => task.abort(),
            false => {
                tasks.retain(|task| !task.is_finished());
                tasks.push(task);
            }
        }
    }
}

impl Drop for Tenant {
    fn drop(&mut self) {
        self.hold.lose();
    }
}

/// A restore running on a tenant: as it was asked for, and what ends as
/// the restore ends.
#[derive(Debug)]
struct Running {
    asking: Asking,
    ended: watch::Receiver<()>,
}

/// A restore as one request asks for it: the request, and a digest of the
/// state it asks to go back to, the same on every try of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Asking {
    request: RequestId,
    digest: [u8; 32],
}

/// Which state a restore goes back to.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Target {
    Point(Point),
    Backup(uuid::Uuid),
    /// A backup kept under the tenant's name, taken up by a tenant this
    /// cluster had lost, with the keys the backup kept or without.
    Adopt {
        backup: uuid::Uuid,
        with_keys: bool,
    },
}

impl Asking {
    fn of(request: RequestId, target: Target) -> Self {
        let mut digest = Sha256::new();
        match target {
            Target::Point(Point::At(at)) => {
                digest.update(b"at");
                digest.update(at.timestamp().to_be_bytes());
                digest.update(at.timestamp_subsec_nanos().to_be_bytes());
            }
            Target::Point(Point::Position(position)) => {
                digest.update(b"position");
                digest.update(position.0.to_be_bytes());
            }
            Target::Backup(id) => {
                digest.update(b"backup");
                digest.update(id.as_bytes());
            }
            Target::Adopt { backup, with_keys } => {
                digest.update(b"adopt");
                digest.update(backup.as_bytes());
                digest.update([u8::from(with_keys)]);
            }
        }
        Self {
            request,
            digest: digest.finalize().into(),
        }
    }

    /// The restore this asks for, going back to `from`, which was
    /// committed or backed up at `from_at`.
    pub(crate) fn to(self, from: LogPosition, from_at: DateTime<Utc>) -> Restore {
        Restore {
            request: self.request,
            asked: self.digest,
            from,
            // To the microsecond, as the coordinator records it: the
            // answer given now is the one given from the record later.
            from_at: from_at.trunc_subsecs(6),
        }
    }

    /// The restore of `tenant` this request made lately, here or on any
    /// node, if it made one; a refusal if what it made was another.
    async fn recorded(
        &self,
        coordinator: &dyn Coordinator,
        tenant: &TenantId,
    ) -> Result<Option<Restored>> {
        match coordinator.restored(&key(tenant), self.request).await? {
            Some(done) if done.restore.asked == self.digest => Ok(Some(done)),
            Some(_) => Err(self.reused(tenant)),
            None => Ok(None),
        }
    }

    /// The refusal of this request for asking another restore of `tenant`
    /// than it asked before.
    fn reused(&self, tenant: &TenantId) -> NodeError {
        NodeError::Reused {
            tenant: tenant.clone(),
            request: self.request,
        }
    }
}

/// The end of a restore, however it ends, even before it began: the tenant
/// is free to restore again, and then whatever waited for it learns that
/// it ended.
struct Ending {
    tenant: Arc<Tenant>,
    _ends: watch::Sender<()>,
}

impl Drop for Ending {
    fn drop(&mut self) {
        *self.tenant.running.lock().expect("running lock") = None;
    }
}

impl Tenant {
    pub fn id(&self) -> &TenantId {
        &self.id
    }

    /// Whether this node still holds the tenant, as far as it can tell
    /// without asking: its hold has not run out and nothing has failed.
    fn held(&self) -> bool {
        !self.hold.is_lost() && Instant::now() < *self.deadline()
    }

    fn deadline(&self) -> std::sync::MutexGuard<'_, Instant> {
        self.deadline.lock().expect("deadline lock")
    }

    fn touch(&self) {
        *self.used.lock().expect("used lock") = Instant::now();
    }

    fn idle(&self) -> Duration {
        self.used.lock().expect("used lock").elapsed()
    }

    /// When the tenant was last used.
    fn used_at(&self) -> Instant {
        *self.used.lock().expect("used lock")
    }

    /// The tenant's data as this node holds it in memory.
    pub fn partition(&self) -> &Arc<vellev_storage::Partition> {
        self.writer.partition()
    }

    /// Run `f` against the tenant's data as of its last commit.
    pub fn read<T>(&self, f: impl FnOnce(&State) -> T) -> T {
        self.writer.partition().read(f)
    }

    /// The vectors `mutations` will need made, for as many as can be made
    /// now. Callers embed side by side here; whatever is not made here,
    /// for any reason, is made or refused by the commit, one commit at a
    /// time.
    async fn embedded(&self, mutations: &[Mutation]) -> Embedded {
        let mut alike: HashMap<Embedding, Vec<String>> = HashMap::new();
        for (embedding, text) in self.writer.wanted(mutations) {
            alike.entry(embedding).or_default().push(text);
        }
        let made = alike.into_iter().map(|(embedding, texts)| async move {
            let embedder = self.embedders.embedder(&embedding).ok()??;
            let vectors = embedder.documents(&texts).await.ok()?;
            Some((embedding, texts, vectors))
        });
        join_all(made)
            .await
            .into_iter()
            .flatten()
            .flat_map(|(embedding, texts, vectors)| {
                texts
                    .into_iter()
                    .zip(vectors)
                    .filter_map(move |(text, vector)| Some(((embedding.clone(), text), vector?)))
            })
            .collect()
    }

    /// Refuse to make a collection that could not be embedded for, rather
    /// than make one every write to which would be refused. Only the
    /// collections the batch would in fact make are looked at, once the
    /// batch is known to be one that would be made: a batch that is
    /// invalid or clashes with what is stored is refused as that, and one
    /// that makes again a collection there exactly so changes nothing, as
    /// ever.
    ///
    /// A model this release does not run is the request's fault. That is
    /// this node's answer, the node that writes the tenant now: one of a
    /// release that knows other models may do so later and answer
    /// otherwise. Whatever only using the model would show, such as an
    /// endpoint that refuses the node, is not looked for.
    fn can_embed(&self, mutations: &[Mutation]) -> Result<()> {
        let makes = |mutation: &Mutation| matches!(mutation, Mutation::CreateCollection(_));
        if !mutations.iter().any(makes) {
            return Ok(());
        }
        self.writer
            .creates(mutations)
            .iter()
            .try_for_each(|schema| {
                self.embedders
                    .admits(&schema.embedding)
                    .map_err(|refused| match refused {
                        EmbedError::NoSuchModel(reason) => NodeError::NoSuchModel(reason),
                        other => NodeError::Embed(other),
                    })
            })
    }

    /// Commit a batch as one atomic step, sharing the commit with whoever
    /// else is writing. A batch naming a request answered lately is
    /// answered as it was, and not made again.
    pub async fn write(&self, batch: impl Into<Batch>) -> Result<Receipt> {
        let batch = batch.into();
        // A request answered already is answered as it was: it costs
        // nothing, and needs no vectors made.
        let answered = batch
            .request
            .is_some_and(|request| self.read(|state| state.answered(request)));
        if !answered {
            // Refused for its backlog before it is counted against the
            // tenant's limits: a write turned away costs it nothing.
            self.not_backlogged()?;
            self.limits
                .admit(&self.id, Asked::Write, batch.mutations.len())?;
            self.can_embed(&batch.mutations)?;
        }
        let made = match answered {
            true => Embedded::new(),
            false => self.embedded(&batch.mutations).await,
        };
        let receipt = self
            .commits
            .submit_batch(batch, made)
            .await
            .map_err(|error| self.failed(error))?;
        // A delete by filter was let through as one write: every object
        // it deleted is one more, owed from now on.
        if !answered {
            let deleted = receipt.outcomes.iter().map(|outcome| match outcome {
                vellev_storage::Outcome::Deleted(deleted) => {
                    usize::try_from(*deleted).unwrap_or(usize::MAX)
                }
                _ => 0,
            });
            self.limits.owe(&self.id, Asked::Write, deleted.sum());
        }
        if self.read(State::unflushed) >= self.flush_rows {
            let _ = self.flushes.try_send(());
        }
        Ok(receipt)
    }

    /// Whether no flush or merge is due or under way: fewer rows unflushed
    /// than a flush, which they stay until the flush of them is done, and
    /// every flush followed by a maintenance round that went through. Rows
    /// fewer than a flush stay in memory, searched exactly, and leave the
    /// tenant settled.
    fn settled(&self) -> bool {
        let settling = &self.settling;
        self.figures().unflushed < self.flush_rows
            && settling.maintained.load(Ordering::SeqCst) >= settling.flushed.load(Ordering::SeqCst)
    }

    /// What the tenant comes to, as it was last changed: read without the
    /// lock on its state, as a scrape and a sweep read it.
    fn figures(&self) -> vellev_storage::Figures {
        self.writer.partition().figures()
    }

    /// Refuse a write while the tenant holds [`Settings::backlog_rows`]
    /// rows unflushed, asking for a flush, and saying when one may next
    /// be tried.
    fn not_backlogged(&self) -> Result<()> {
        if self.read(State::unflushed) < self.backlog_rows {
            return Ok(());
        }
        let _ = self.flushes.try_send(());
        self.flushing.backlogged.fetch_add(1, Ordering::Relaxed);
        let next = self.backoff.lock().expect("backoff lock").1;
        let wait = next.map_or(Duration::ZERO, |next| {
            next.saturating_duration_since(Instant::now())
        });
        Err(NodeError::Backlogged {
            tenant: self.id.clone(),
            wait: wait.max(self.flush_backoff),
        })
    }

    /// Whether the tenant's maintenance would flush it now: it holds a
    /// flush's worth of rows, and no failed flush puts the next off.
    fn flush_due(&self) -> bool {
        self.read(State::unflushed) >= self.flush_rows && self.flush_may_be_tried()
    }

    /// Whether a flush may be tried now, none having failed lately.
    fn flush_may_be_tried(&self) -> bool {
        let next = self.backoff.lock().expect("backoff lock").1;
        next.is_none_or(|next| Instant::now() >= next)
    }

    /// When the next flush may be tried, if the last one tried failed.
    fn flush_retried_at(&self) -> Option<Instant> {
        self.backoff.lock().expect("backoff lock").1
    }

    /// Note how the flush tried went: one that failed puts the next off,
    /// twice as long as the one before, and one that went through ends that.
    fn flush_tried(&self, went_through: bool) {
        let mut backoff = self.backoff.lock().expect("backoff lock");
        if went_through {
            *backoff = (0, None);
            return;
        }
        self.flushing.failed.fetch_add(1, Ordering::Relaxed);
        let failed = backoff.0.saturating_add(1);
        let wait = self.flush_backoff * 2u32.pow((failed - 1).min(6));
        *backoff = (failed, Some(Instant::now() + wait));
    }

    /// Move what has been written since the last checkpoint out of memory.
    /// However it is asked for, one that fails puts the next the node
    /// tries off, as [`Settings::flush_backoff`] says, and is counted in
    /// [`Node::flushing`]; one that goes through ends that.
    pub async fn flush(&self) -> Result<LogPosition> {
        let flushed = self.writer.flush().await;
        self.flush_tried(flushed.is_ok());
        if flushed.is_ok() {
            self.settling.flushed.fetch_add(1, Ordering::SeqCst);
        }
        flushed.map_err(|error| self.failed(error))
    }

    /// As [`Self::flush`], made before the tenant is let go: its wait for
    /// the building lock and its hold are timed under their own name, as
    /// the merge bench holds the wait to a mark of its own (merge note,
    /// §3).
    async fn flush_to_let_go(&self) -> Result<LogPosition> {
        let flushed = self.writer.flush_timed().await;
        self.flush_tried(flushed.is_ok());
        let (position, times) = flushed.map_err(|error| self.failed(error))?;
        self.settling.flushed.fetch_add(1, Ordering::SeqCst);
        let timings = self.writer.timings();
        let partition = &self.writer.lease().key;
        timings.record(Timed::LetGoFlushWait, partition, times.waited);
        if let Some(held) = times.held {
            timings.record(Timed::LetGoFlushHeld, partition, held);
        }
        Ok(position)
    }

    /// Merge segments until a merge finds nothing to do, and then delete
    /// the files nothing needs any more, as [`Writer::clean`] with `grace`,
    /// which deletes so many at once, until one finds fewer than that.
    ///
    /// `grace` is taken as it is given, and nothing here checks it. A node
    /// looking after its tenants gives the tenant's restore window and
    /// [`Settings::grace`] on top, which is at least
    /// [`RequestBounds::least_grace`] where [`Settings::check`] was asked;
    /// whoever calls this with less may delete files a restore to a point
    /// of that window names, and that is the caller's breach. What that
    /// floor does not cover is said at [`LEAST_GRACE`].
    pub async fn compact(&self, grace: Duration) -> Result<usize> {
        let compacted = self.compacted(grace, &|| false).await?;
        Ok(compacted.unwrap_or(0))
    }

    /// As [`Self::compact`], each merge giving way to a flush that falls due
    /// while it waits for a turn among the node's merge builds
    /// ([`Writer::merge_giving_way`]): so that the tenant's maintenance,
    /// which flushes it, is never held behind other tenants' merges, and
    /// its writes are never refused for a backlog those let pile up.
    /// Returns how many files were deleted, or `None` if cleaning up was
    /// left for a later round.
    async fn compacted(
        &self,
        grace: Duration,
        flush_due: &(dyn Fn() -> bool + Sync),
    ) -> Result<Option<usize>> {
        while self
            .writer
            .merge_giving_way(flush_due)
            .await
            .map_err(|error| self.failed(error))?
            .is_some()
        {}
        // Given way to a flush, or one is due: cleaning up, which waits on
        // the store, waits for a later round, and the flush goes first;
        // but not for longer than [`Settings::clean_at_least_every`], or
        // writes that never let up would leave it undone for good.
        let cleaned_lately = self.cleaned_at().elapsed() < self.clean_at_least_every;
        if flush_due() && cleaned_lately {
            return Ok(None);
        }
        let mut deleted = 0;
        loop {
            let cleaned = self.writer.clean(grace).await;
            let more = cleaned.map_err(|error| self.failed(error))?;
            deleted += more;
            if more < Writer::CLEANED_AT_ONCE {
                *self.cleaned_at() = Instant::now();
                return Ok(Some(deleted));
            }
        }
    }

    /// When the tenant was last cleaned up all through, or opened.
    fn cleaned_at(&self) -> std::sync::MutexGuard<'_, Instant> {
        self.cleaned_at.lock().expect("cleaned lock")
    }

    /// Forget the commits the tenant can no longer go back to, then merge
    /// its segments and delete the files no restore within its window
    /// needs, kept `grace` longer; then give any segment from before
    /// curves the curve of its rows (recall note, R9.5). Returns how many
    /// files were deleted.
    async fn maintained(
        &self,
        coordinator: &dyn Coordinator,
        default_window: Duration,
        grace: Duration,
    ) -> Result<usize> {
        let window = restore_window(coordinator, &self.id, default_window).await?;
        coordinator.forget_history(&key(&self.id), window).await?;
        let flush_due = || self.flush_due();
        let compacted = self
            .compacted(window.saturating_add(grace), &flush_due)
            .await?;
        // The round ends here if cleaning up was left for a later round:
        // the flush due goes first, and the rest is done then.
        let Some(deleted) = compacted else {
            return Ok(0);
        };
        // One pass at a time on the node: another tenant's next round
        // takes its turn. What is missing is worked out afresh each time,
        // so a pass skipped or cut short is made at a later round.
        if let Ok(_one) = ROWS_PASSES.try_acquire() {
            if let Err(error) = self.writer.rows_pass().await {
                tracing::warn!(tenant = %self.id, %error, "the curves of segments' rows were not made");
            }
        }
        Ok(deleted)
    }

    /// The generation of the lease this node writes the tenant under.
    pub(crate) fn generation(&self) -> Generation {
        self.writer.lease().generation
    }

    /// Put the tenant back as it stood at `head`, as `restore` asks, and
    /// as [`Writer::restore_recorded`] does, `adopting` the lineage of the
    /// backup it takes up if it takes one up. A failure that closes the
    /// writer is only told here: the hold is lost by whoever runs this on
    /// a task (see [`Self::restoring`]), once the task has ended, since
    /// losing it stops that task wherever it is, with whatever it had
    /// still to say and to tidy up.
    async fn restored(
        &self,
        head: Option<ManifestRef>,
        restore: Restore,
        adopting: Option<Lineage>,
    ) -> Result<Restored> {
        let position = self
            .writer
            .restore_recorded(head.as_ref(), restore.clone(), adopting)
            .await
            .map_err(|error| self.told(error))?;
        Ok(Restored {
            restore,
            incarnation: self.writer.partition().incarnation(),
            position,
        })
    }

    /// Restore the tenant as `work` does, as `asking` asks, on a task of
    /// its own that goes on whoever stops waiting, until it ends or this
    /// node's hold on the tenant is lost. A request whose restore was
    /// recorded lately, here or on any node, is answered as it was and not
    /// made again, or refused if it asked for another. While one restore
    /// runs here, another of the same request waits for it to end and
    /// returns `None`, to be asked again of the tenant as it is then; one
    /// of any other request is refused.
    ///
    /// A restore that fails or is stopped is not thereby one that was not
    /// made: see [`Self::unfinished`].
    async fn restoring<F>(
        self: &Arc<Self>,
        coordinator: &Arc<dyn Coordinator>,
        asking: Asking,
        work: F,
    ) -> Result<Option<Restored>>
    where
        F: Future<Output = Result<Restored>> + Send + 'static,
    {
        let started = {
            let mut running = self.running.lock().expect("running lock");
            match &*running {
                Some(same) if same.asking == asking => Err(same.ended.clone()),
                Some(other) if other.asking.request == asking.request => {
                    return Err(asking.reused(&self.id))
                }
                Some(_) => return Err(NodeError::Restoring(self.id.clone())),
                None => {
                    let (ends, ended) = watch::channel(());
                    *running = Some(Running { asking, ended });
                    Ok(ends)
                }
            }
        };
        let ending = match started {
            Ok(ends) => Ending {
                tenant: self.clone(),
                _ends: ends,
            },
            Err(mut ended) => {
                // Only its end is waited for: nothing is ever sent.
                let _ = ended.changed().await;
                return Ok(None);
            }
        };
        let asked = coordinator.clone();
        // The task hands the restore's end back with what it came to, so
        // that whoever waits for that end learns of it only once a writer
        // the restore closed has cost this node its hold: they then find
        // the tenant opened afresh, not this one again.
        let restoring = tokio::spawn(async move {
            let recorded = asking.recorded(asked.as_ref(), &ending.tenant.id).await;
            let restored = match recorded {
                Ok(Some(done)) => Ok(done),
                Ok(None) => work.await,
                Err(failure) => Err(failure),
            };
            (restored, ending)
        });
        self.hold.stops(restoring.abort_handle());
        // Whoever asked may stop waiting, and losing the hold stops the
        // task above wherever it is: so a task of its own, which neither
        // reaches, sees it end, and only then loses the hold if the
        // restore closed the writer, which is gone for good.
        let tenant = self.clone();
        let ended = tokio::spawn(async move {
            let ended = restoring.await;
            if tenant.writer.is_closed() {
                tenant.hold.lose();
            }
            ended.map(|(restored, _ending)| restored)
        });
        let failure = match ended.await {
            Ok(Ok(Ok(restored))) => return Ok(Some(restored)),
            Ok(Ok(Err(failure))) => failure,
            Ok(Err(stopped)) if stopped.is_cancelled() => NodeError::Lost(self.id.clone()),
            Ok(Err(stopped)) | Err(stopped) => StorageError::Background(stopped.to_string()).into(),
        };
        self.unfinished(coordinator.as_ref(), asking, failure).await
    }

    /// What a restore asked for as `asking` that ended in `failure`, or
    /// was stopped, came to. Its commit may have been made all the same:
    /// stopped while its answer was awaited, or failed in what follows it.
    /// So the coordinator is asked, and a restore it has recorded is
    /// answered as made. One it has not is told as it failed if no commit
    /// of it can yet be made: the writer is still open, or said it was
    /// closed before anything was tried, so none was begun; or the
    /// coordinator refused the commit outright, as it does one by a lease
    /// since taken. Otherwise a commit was begun whose answer never came,
    /// which may yet be made, and the caller is told that whether it was
    /// is unknown, to ask again by the same name.
    async fn unfinished(
        &self,
        coordinator: &dyn Coordinator,
        asking: Asking,
        failure: NodeError,
    ) -> Result<Option<Restored>> {
        let unmade = match &failure {
            NodeError::Storage(StorageError::Closed) => true,
            NodeError::UnknownTenant(_) => true,
            NodeError::Storage(failure) => matches!(
                failure.root(),
                StorageError::Coordinator(refused) if !refused.is_retryable()
            ),
            _ => false,
        };
        match asking.recorded(coordinator, &self.id).await {
            Ok(Some(done)) => Ok(Some(done)),
            Ok(None) if unmade || !self.writer.is_closed() => Err(failure),
            Err(reused @ NodeError::Reused { .. }) => Err(reused),
            Ok(None) | Err(_) => Err(NodeError::Storage(match failure {
                NodeError::Storage(shared @ StorageError::Shared(_)) => shared,
                NodeError::Storage(failure) => StorageError::Shared(Arc::new(failure)),
                _ => StorageError::Shared(Arc::new(StorageError::Closed)),
            })),
        }
    }

    /// A failure of the writer's, as its caller should see it, noting
    /// that the tenant must be opened afresh if the writer is closed now:
    /// whether it is, not how the failure reads, says so, as a refusal or
    /// a restore under way leave it open, and a commit that failed for
    /// any reason closes it. A writer, or the committer in front of it,
    /// that says it is closed is gone for good too.
    fn failed(&self, error: StorageError) -> NodeError {
        if self.writer.is_closed() || matches!(error.root(), StorageError::Closed) {
            self.hold.lose();
        }
        self.told(error)
    }

    /// A failure of the writer's as its caller should see it.
    fn told(&self, error: StorageError) -> NodeError {
        match error.root() {
            StorageError::Coordinator(CoordinatorError::UnknownPartition(_)) => {
                NodeError::UnknownTenant(self.id.clone())
            }
            _ => error.into(),
        }
    }
}

type Opening = Arc<OnceCell<Arc<Tenant>>>;

/// How many tries of a restore by others one caller waits for before it is
/// told a restore is running, rather than making a try of its own.
const RESTORE_WAITS: usize = 3;

/// How long a tenant or copy is kept after it was last used, whatever the
/// memory budget.
const RESIDENT_AT_LEAST: Duration = Duration::from_secs(1);

/// The least time between sweeps that uses of tenants ask for (see
/// `Inner::sweep_soon`). Over the budget every use asks for one, and with
/// nothing to let go, coalesced, they would run back to back for as long
/// as uses come, each taking the node's lock twice. A quarter of
/// [`RESIDENT_AT_LEAST`]: a tenant used within that is kept whatever a
/// sweep finds, so sweeping more often lets nothing more go, and four
/// looks a second still answer an open that takes the node over its
/// budget long before what it holds grows much further. The node's own
/// sweeps, a second apart at the least, and [`Node::sweep_now`] are not
/// held to it.
pub const SWEEPS_SPACED: Duration = Duration::from_millis(250);

/// A tenant written for or copied, as a sweep finds it.
struct Held {
    id: TenantId,
    kind: Kind,
    unused: Duration,
    size: usize,
    /// A copy warming opened that nobody has read.
    warmed: bool,
}

enum Kind {
    Writer(Opening, Arc<Tenant>),
    Copy(reading::Following, Arc<reading::Followed>),
}

/// What a sweep took out of what the node holds, and what it chose and
/// could not.
#[derive(Default)]
struct Taken {
    /// The tenants written for, taken out and still to be let go.
    going: Vec<Arc<Tenant>>,
    /// The tenants written for that something else held, left as they are,
    /// with what each holds, in the order they were chosen.
    kept: Vec<(Arc<Tenant>, usize)>,
    /// How many chosen, written for or copied, something else held.
    held_elsewhere: u64,
    /// About how many bytes what was taken out held.
    freed: usize,
    /// How many were taken out for being idle, and for the budget.
    idle: u64,
    for_budget: u64,
}

impl Taken {
    /// Take `held` out of `open` if it is still the one held and nobody but
    /// the node and this sweep has it, nor is opening it; else keep it.
    /// Counted as taken out for the budget, or for being idle.
    fn take(&mut self, open: &mut Open, held: &Held, for_budget: bool) {
        if !self.taken(open, held) {
            return;
        }
        match for_budget {
            true => self.for_budget += 1,
            false => self.idle += 1,
        }
    }

    /// As [`Self::take`], saying whether it was taken out.
    fn taken(&mut self, open: &mut Open, held: &Held) -> bool {
        match &held.kind {
            Kind::Writer(opening, tenant) => {
                let same = open
                    .tenants
                    .get(&held.id)
                    .is_some_and(|now| Arc::ptr_eq(now, opening));
                if same && Arc::strong_count(opening) == 2 && Arc::strong_count(tenant) == 2 {
                    open.tenants.remove(&held.id);
                    self.going.push(tenant.clone());
                    self.freed += held.size;
                    return true;
                } else if same {
                    self.kept.push((tenant.clone(), held.size));
                    self.held_elsewhere += 1;
                }
                false
            }
            Kind::Copy(following, copy) => {
                let same = open
                    .followers
                    .get(&held.id)
                    .is_some_and(|now| Arc::ptr_eq(now, following));
                if same && Arc::strong_count(following) == 2 && Arc::strong_count(copy) == 2 {
                    open.followers.remove(&held.id);
                    self.freed += held.size;
                    return true;
                } else if same {
                    self.held_elsewhere += 1;
                }
                false
            }
        }
    }
}

#[derive(Debug, Default)]
struct Open {
    tenants: HashMap<TenantId, Opening>,
    followers: reading::Followers,
    stopped: bool,
}

#[derive(Debug)]
struct Inner {
    coordinator: Arc<dyn Coordinator>,
    blobs: Arc<dyn Blobs>,
    embedders: Arc<Embedders>,
    settings: Settings,
    open: Mutex<Open>,
    /// When object storage last answered a readiness check.
    store_answered: Mutex<Option<Instant>>,
    keys: Keys,
    /// How many writers and copies for reads this node has opened.
    writers_opened: AtomicU64,
    copies_opened: AtomicU64,
    limits: Arc<Limits>,
    /// The tenants purged lately whose cached files this node has dropped.
    dropped: Mutex<std::collections::HashSet<TenantId>>,
    /// The newest file format the cluster was seen to write; it never
    /// falls.
    cluster_format: AtomicU16,
    /// The format every writer this node opens writes: the cluster's, as
    /// far as this release reads it. 0 until the cluster's is first seen,
    /// which a writer takes as format 1. It never falls.
    ///
    /// `max_format` bounds how far this node raises the cluster's format,
    /// not what it writes once the cluster writes a format: that rose only
    /// as every node registered read it, so each can read what any writes,
    /// and a node that wrote older would fail to checkpoint a tenant
    /// another had given what only the newer format holds.
    writes_format: Arc<AtomicU16>,
    /// Flushes failed and writes refused for them, on every tenant.
    flushing: Arc<FlushCounts>,
    /// The warm list, and the warming that reads it.
    warm: warm::Warm,
    /// The merge builds of every tenant this node writes for, taken in
    /// turns, at most [`MERGE_BUILDS`] at once.
    merge_builds: Arc<MergeBuilds>,
    /// What the last sweep found the node holding: see [`Node::census`].
    census: Mutex<Census>,
    /// Held across a sweep, so that no two run at once: each would hold
    /// what the other must find held by nothing else, and keep it.
    sweeping: Arc<tokio::sync::Mutex<()>>,
    /// A sweep was asked for while one ran: the one running sweeps once
    /// more as it ends (see [`Inner::sweep_soon`]).
    sweep_again: AtomicBool,
    /// How many sweeps have begun.
    sweeps_begun: AtomicU64,
    /// When the last sweep began: those uses ask for are spaced from it.
    /// Tokio's time, so that spacing follows `tokio::time::pause()`.
    last_swept: Mutex<Option<Instant>>,
    /// One waits to sweep, spaced, not yet holding [`Inner::sweeping`].
    sweep_queued: AtomicBool,
    /// The node's own sweep was asked for, not held to [`SWEEPS_SPACED`].
    sweep_on_time: AtomicBool,
    /// Where a test holds each sweep as it is about to take tenants out.
    sweep_hold: Mutex<Option<Arc<SweepHold>>>,
    /// When the node was made: how old its census is until its first sweep.
    started: Instant,
    /// How many tenants sweeps chose to let go but found something else
    /// holding.
    held_elsewhere: AtomicU64,
    /// How many merges sweeps have stopped so that a tenant over the
    /// budget could be let go.
    stopped_for_budget: AtomicU64,
    /// How many tenants, written for or copied, sweeps have let go: those
    /// idle, and those let go for the memory budget.
    let_go_idle: AtomicU64,
    let_go_for_budget: AtomicU64,
    /// Where every writer this node opens times its background work.
    timings: Arc<Timings>,
}

/// Flushes that failed, and writes refused while they did.
#[derive(Debug, Default)]
struct FlushCounts {
    failed: AtomicU64,
    backlogged: AtomicU64,
}

/// One pass making the curves of segments' rows at a time on this node:
/// each is a pass over a segment's vectors, and after an upgrade every
/// clustered segment of every tenant needs one.
static ROWS_PASSES: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
/// The segments the tenants this node writes hold: those with an index file
/// (a clustered vector index for at least one of their collections) and the
/// rest, each with their rows as written: rows since replaced or deleted
/// are still counted, as are the rows of every collection a segment holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Segments {
    pub clustered: usize,
    pub clustered_rows: u64,
    pub plain: usize,
    pub plain_rows: u64,
    /// Segments as flushes and restores wrote them, which no merge has
    /// taken in yet, and their rows as written: the rows awaiting a merge.
    pub unmerged: usize,
    pub unmerged_rows: u64,
    /// Where the clustered indexes' probes come from.
    pub probes: Probes,
}

pub use vellev_storage::Probes;

impl Segments {
    fn add(&mut self, figures: &vellev_storage::Figures) {
        self.clustered += figures.clustered;
        self.clustered_rows += figures.clustered_rows;
        self.plain += figures.plain;
        self.plain_rows += figures.plain_rows;
        self.unmerged += figures.unmerged;
        self.unmerged_rows += figures.unmerged_rows;
        self.probes += figures.probes;
    }
}

/// The merges this node builds, as memory goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merges {
    /// About how many bytes merges stopped and not yet ended hold, and
    /// what stores keep of those whose turns were let go: counted against
    /// the memory budget.
    pub draining: usize,
    /// About how many bytes stores keep of stopped or failed merges whose
    /// turns were let go for it, as the store held them too long.
    pub abandoned: usize,
    /// How many merge turns were let go so, since the node started.
    pub held_let_go: u64,
    /// How many merges sweeps have stopped since the node started, so that
    /// a tenant over the memory budget could be let go.
    pub stopped_for_budget: u64,
    /// About how many bytes the largest merge build since the node started
    /// was reckoned to hold at its peak.
    pub largest: usize,
}

/// How many tenants sweeps have let go: those idle for
/// [`Settings::tenant_idle`], and those let go for the memory budget.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LetGo {
    pub idle: u64,
    pub for_budget: u64,
}

/// What this node holds open, and what it reckons that holds in memory:
/// the figures its memory budget is held to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Holding {
    /// Tenants it holds open as their writer.
    pub writers: usize,
    /// Tenants it holds open as a copy to read from.
    pub copies: usize,
    /// About how many bytes they take in memory together, as
    /// [`Node::resident`] reckons them.
    pub resident: usize,
    /// About how many bytes the one that takes the most takes.
    pub largest: usize,
}

/// Where a test holds each sweep as it is about to take tenants out (see
/// [`Node::holding_sweeps`]).
#[derive(Debug, Default)]
#[doc(hidden)]
pub struct SweepHold {
    reached: AtomicU64,
    released: AtomicBool,
    release: tokio::sync::Notify,
}

impl SweepHold {
    /// How many sweeps have reached the hold.
    pub fn reached(&self) -> u64 {
        self.reached.load(Ordering::SeqCst)
    }

    /// Let every sweep held, and every one to come, go on.
    pub fn release(&self) {
        self.released.store(true, Ordering::SeqCst);
        self.release.notify_waiters();
    }

    async fn reached_here(&self) {
        self.reached.fetch_add(1, Ordering::SeqCst);
        loop {
            let released = self.release.notified();
            if self.released.load(Ordering::SeqCst) {
                return;
            }
            released.await;
        }
    }
}

/// What a sweep found the node holding as it began, which is what it
/// held to the memory budget: published by every sweep, so that reading
/// it holds no tenant. A sweep lets a tenant go only if nothing else holds
/// it, so whatever held the tenants to read them would keep them. What is
/// not the sweep's own decision (what waits for a flush, the segments) is
/// read as it is instead, from each tenant's published figures.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Census {
    /// The tenants held, and what they are reckoned to take.
    pub holding: Holding,
    /// When the sweep that took it began; none before the first.
    pub swept_at: Option<Instant>,
}

impl Census {
    /// Of the tenants and copies in `held`, read one at a time.
    fn of(held: &[Held]) -> Self {
        let mut census = Self::default();
        for held in held {
            census.holding.resident += held.size;
            census.holding.largest = census.holding.largest.max(held.size);
            match &held.kind {
                Kind::Writer(..) => census.holding.writers += 1,
                Kind::Copy(..) => census.holding.copies += 1,
            }
        }
        census
    }
}

/// What this node's tenants hold unflushed, and how their flushes went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Flushing {
    /// Flushes that failed, since the node started.
    pub failed: u64,
    /// Writes refused as their tenant held its backlog, since it started.
    pub backlogged: u64,
    /// Rows written to the tenants this node writes and not flushed yet.
    pub unflushed_rows: usize,
    /// About how many bytes those rows take in memory.
    pub unflushed_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct Node(Arc<Inner>);

/// How many tenants a node has opened since it started. Each is a load of
/// the tenant's committed state, which reads that wait on it wait for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Opens {
    pub writers: u64,
    pub copies: u64,
}

impl Node {
    /// A node that runs as `settings` say, taken as they are: nothing here
    /// refuses them. Call [`Settings::check`] first, as the `vellev`
    /// binary does, or the node runs with whatever grace and backup claim
    /// it was given.
    pub fn new(
        coordinator: Arc<dyn Coordinator>,
        blobs: Arc<dyn Blobs>,
        embedders: Arc<Embedders>,
        settings: Settings,
    ) -> Self {
        Self(Arc::new(Inner {
            coordinator,
            blobs,
            embedders,
            open: Mutex::default(),
            store_answered: Mutex::default(),
            keys: Keys::default(),
            writers_opened: AtomicU64::new(0),
            copies_opened: AtomicU64::new(0),
            dropped: Mutex::default(),
            cluster_format: AtomicU16::new(0),
            writes_format: Arc::new(AtomicU16::new(0)),
            flushing: Arc::default(),
            warm: warm::Warm::default(),
            merge_builds: match settings.merge_pool {
                MergePool::Global => MergeBuilds::new(MERGE_BUILDS),
                MergePool::Own { threads, nice } => MergeBuilds::building_on(
                    MERGE_BUILDS,
                    Arc::new(vellev_storage::merge_pool(threads, nice)),
                ),
            },
            census: Mutex::default(),
            sweeping: Arc::new(tokio::sync::Mutex::new(())),
            sweep_again: AtomicBool::new(false),
            sweeps_begun: AtomicU64::new(0),
            last_swept: Mutex::new(None),
            sweep_queued: AtomicBool::new(false),
            sweep_on_time: AtomicBool::new(false),
            sweep_hold: Mutex::default(),
            started: Instant::now(),
            held_elsewhere: AtomicU64::new(0),
            stopped_for_budget: AtomicU64::new(0),
            let_go_idle: AtomicU64::new(0),
            let_go_for_budget: AtomicU64::new(0),
            timings: Arc::default(),
            limits: Arc::new(Limits::new(
                settings.queries_per_second,
                settings.writes_per_second,
            )),
            settings,
        }))
    }

    /// Whether the node can reach Postgres and object storage and can read
    /// what the cluster writes.
    pub async fn ready(&self) -> Result<()> {
        self.0.readable()?;
        self.store_answers().await?;
        self.0.saw_format(self.0.coordinator.format().await?);
        self.0.readable()
    }

    /// Whether object storage answered within [`STORE_ANSWERED`], asking
    /// it again if not.
    async fn store_answers(&self) -> Result<()> {
        let fresh = |at: &Option<Instant>| at.is_some_and(|at| at.elapsed() < STORE_ANSWERED);
        if fresh(&self.0.store_answered.lock().expect("readiness lock")) {
            return Ok(());
        }
        match self.0.blobs.get(READINESS).await {
            Ok(_) | Err(StorageError::Missing { .. }) => {
                *self.0.store_answered.lock().expect("readiness lock") = Some(Instant::now());
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    /// What this node reads.
    pub fn versions(&self) -> Versions {
        Versions {
            format: FORMAT_VERSION,
            protocol: PROTOCOL,
        }
    }

    /// Register this node as running and reading what it reads, now and
    /// every third of a lease until it stops, and each time raise the file
    /// format the cluster writes as far as this node may write, which only
    /// happens once every node registered reads it. A node that finds the
    /// cluster writing a format newer than it reads says so, and is no
    /// longer ready.
    pub fn announcing(&self) {
        let every = (self.0.settings.lease / 3).max(Duration::from_secs(1));
        let (inner, versions) = (Arc::downgrade(&self.0), self.versions());
        let writes = versions.format.min(self.0.settings.max_format);
        tokio::spawn(async move {
            loop {
                let Some(inner) = inner.upgrade() else { return };
                if inner.opened().stopped {
                    return;
                }
                let node = &inner.settings.node;
                let announced = async {
                    inner.saw_format(inner.coordinator.announce(node, versions).await?);
                    inner.saw_format(inner.coordinator.raise_format(writes).await?);
                    Ok::<_, CoordinatorError>(())
                };
                if let Err(error) = announced.await {
                    tracing::warn!(%error, "announcing the node failed");
                }
                drop(inner);
                tokio::time::sleep(every).await;
            }
        });
    }

    /// Let go, now and then, of each tenant unused for longer than
    /// [`Settings::tenant_idle`] that nobody is using now, and of the least
    /// recently used beyond [`Settings::memory`]: flush what it holds in
    /// memory, so that whoever opens it next need not read its log, and
    /// release its lease.
    /// Keep within the memory budget, if there is one, without waiting for
    /// the next sweep: called as each tenant or copy is opened.
    pub(crate) fn bound(&self) {
        if self.0.settings.memory.is_some() {
            self.0.sweep_soon();
        }
    }

    /// About how many bytes the tenants and copies this node holds take
    /// in memory together, as of now. Read under the node's own lock and
    /// holding no tenant, as warming asks it before each copy it opens: a
    /// sweep lets go only what nothing else holds, so asking must never
    /// hold one. Each is a read of what its state counts as it changes.
    pub fn resident(&self) -> usize {
        self.0.resident()
    }

    /// How long ago the last sweep took its census, or since the node was
    /// made if none has: a sweep that has stopped, and with it the memory
    /// budget, shows as this grows.
    pub fn census_age(&self) -> Duration {
        let swept = self.census().swept_at.unwrap_or(self.0.started);
        swept.elapsed()
    }

    /// How many sweeps have begun since the node started.
    pub fn sweeps(&self) -> u64 {
        self.0.sweeps_begun.load(Ordering::Relaxed)
    }

    /// Hold each sweep from now on, as it is about to take tenants out,
    /// until what is given back is released: for tests that a sweep does
    /// not begin while another runs.
    #[doc(hidden)]
    pub fn holding_sweeps(&self) -> Arc<SweepHold> {
        let hold = Arc::new(SweepHold::default());
        *self.0.sweep_hold.lock().expect("sweep hold lock") = Some(hold.clone());
        hold
    }

    /// How many tenants sweeps chose to let go, for being idle or for the
    /// memory budget, but found something else holding, as a request or a
    /// merge does: let go at a later round.
    pub fn tenants_kept_by_sweeps(&self) -> u64 {
        self.0.held_elsewhere.load(Ordering::Relaxed)
    }

    /// What the last sweep found this node holding, as it began: what it
    /// held to the memory budget. Read without holding any tenant, so that
    /// reading it, as /metrics does, never keeps a tenant from being let
    /// go. All zero until the first sweep.
    pub fn census(&self) -> Census {
        *self.0.census.lock().expect("census lock")
    }

    /// Sweep now, as the node does every so often: for a test that reads
    /// what a sweep found.
    #[doc(hidden)]
    pub async fn sweep_now(&self) {
        let sweeping = self.0.sweeping.clone().lock_owned().await;
        self.0.sweep().await;
        drop(sweeping);
        // Any asked for meanwhile, as one that found it running would.
        if self.0.sweep_again.load(Ordering::SeqCst) {
            self.0.sweep_soon();
        }
    }

    /// How many local embedding models are loaded, or loading.
    pub fn local_models_loaded(&self) -> usize {
        self.0.embedders.local_models_loaded()
    }

    /// The memory budget this node holds its tenants to, if it has one:
    /// see [`Settings::memory`].
    pub fn memory_budget(&self) -> Option<usize> {
        self.0.settings.memory
    }

    /// The merges this node builds, as memory goes.
    pub fn merges(&self) -> Merges {
        let builds = &self.0.merge_builds;
        Merges {
            draining: builds.draining(),
            abandoned: builds.abandoned(),
            held_let_go: builds.held_let_go(),
            stopped_for_budget: self.0.stopped_for_budget.load(Ordering::Relaxed),
            largest: builds.largest_reckoned(),
        }
    }

    /// Where the writers this node opens time their background work: the
    /// building lock by holder, merges by phase and by how they ended,
    /// flushes from when they fell due, spells at the backlog, and the
    /// flush made before a tenant is let go.
    pub fn timings(&self) -> &Arc<Timings> {
        &self.0.timings
    }

    /// How many tenants, written for or copied, sweeps have let go since
    /// the node started.
    pub fn let_go(&self) -> LetGo {
        LetGo {
            idle: self.0.let_go_idle.load(Ordering::Relaxed),
            for_budget: self.0.let_go_for_budget.load(Ordering::Relaxed),
        }
    }

    /// Each tenant this node writes for, and what it comes to as it was
    /// last changed: read as [`Self::segments`] reads them, holding none.
    pub fn tenant_figures(&self) -> Vec<(TenantId, vellev_storage::Figures)> {
        let mut figures = Vec::new();
        self.0
            .each_writer(|tenant| figures.push((tenant.id.clone(), tenant.figures())));
        figures
    }

    /// The merge builds of the tenants this node writes for. For tests.
    #[doc(hidden)]
    pub fn merge_builds(&self) -> &Arc<MergeBuilds> {
        &self.0.merge_builds
    }

    /// Whether this node serves `tenant`: see [`Settings::served`].
    pub fn serves(&self, tenant: &TenantId) -> bool {
        self.0.settings.served.serves(tenant)
    }

    pub fn sweeping(&self) {
        let every = self.0.sweep_every();
        let inner = Arc::downgrade(&self.0);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let Some(inner) = inner.upgrade() else { return };
                if inner.opened().stopped {
                    return;
                }
                inner.limits.forget_idle();
                // Not held to the spacing of those uses ask for.
                inner.sweep_on_time.store(true, Ordering::SeqCst);
                inner.sweep_soon();
            }
        });
    }

    /// What the cluster reads and writes: the file format it writes, and
    /// the oldest protocol any node registered speaks.
    pub async fn cluster(&self) -> Result<Versions> {
        let format = self.0.coordinator.format().await?;
        let protocol = self
            .0
            .coordinator
            .oldest()
            .await?
            .map_or(PROTOCOL, |oldest| oldest.protocol.min(PROTOCOL));
        Ok(Versions { format, protocol })
    }

    /// Issue a key that reaches `tenant`; `None` if there is no such tenant.
    pub async fn issue_key(&self, tenant: &TenantId) -> Result<Option<Issued>> {
        self.0.keys.issue(self.0.coordinator.as_ref(), tenant).await
    }

    /// The keys that reach `tenant`, oldest first, without their secrets.
    pub async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>> {
        Ok(self.0.coordinator.keys(tenant).await?)
    }

    /// Revoke one of `tenant`'s keys. Returns whether it was one.
    pub async fn revoke_key(&self, tenant: &TenantId, id: uuid::Uuid) -> Result<bool> {
        let revoked = self.0.coordinator.revoke_key(tenant, id).await?;
        self.0.keys.forget(tenant);
        Ok(revoked)
    }

    /// How the keys presented to this node have been answered since it
    /// started.
    pub fn key_lookups(&self) -> KeyLookups {
        self.0.keys.lookups()
    }

    /// What the tenants this node writes hold unflushed, and how many of
    /// their flushes failed: as they are now, from each tenant's published
    /// figures, holding none of them (see `Inner::each_writer`).
    pub fn flushing(&self) -> Flushing {
        let (mut rows, mut bytes) = (0, 0);
        self.0.each_writer(|tenant| {
            let figures = tenant.figures();
            rows += figures.unflushed;
            bytes += figures.unflushed_bytes;
        });
        Flushing {
            failed: self.0.flushing.failed.load(Ordering::Relaxed),
            backlogged: self.0.flushing.backlogged.load(Ordering::Relaxed),
            unflushed_rows: rows,
            unflushed_bytes: bytes,
        }
    }

    /// How many tenants this node writes for have a flush or merge due or
    /// under way, as they are now. A node with none has nothing left to do
    /// in the background but clean up files past their grace.
    pub fn unsettled(&self) -> usize {
        let mut unsettled = 0;
        self.0
            .each_writer(|tenant| unsettled += usize::from(!tenant.settled()));
        unsettled
    }

    /// The segments the tenants this node writes hold, by whether they
    /// have a clustered index, as they are now.
    pub fn segments(&self) -> Segments {
        let mut segments = Segments::default();
        self.0.each_writer(|tenant| segments.add(&tenant.figures()));
        segments
    }

    /// How many tenants this node has opened since it started: as their
    /// writer, and as copies to read from.
    pub fn opens(&self) -> Opens {
        Opens {
            writers: self.0.writers_opened.load(Ordering::Relaxed),
            copies: self.0.copies_opened.load(Ordering::Relaxed),
        }
    }

    /// The tenant the key `secret` reaches, if it is one.
    pub async fn key_tenant(&self, secret: &str) -> Result<Option<TenantId>> {
        self.0.keys.tenant(&self.0.coordinator, secret).await
    }

    /// Creating a tenant that exists changes nothing.
    pub async fn create_tenant(&self, tenant: &TenantId) -> Result<()> {
        self.0.register(tenant).await
    }

    /// Forget the tenant, and then its history, a batch at a time. Returns
    /// whether it existed. Its files stay in object storage, where nothing
    /// refers to them any more, until it is purged, which forgets whatever
    /// of its history is left. A node that holds the tenant notices at its
    /// next renewal, within a third of the lease.
    pub async fn drop_tenant(&self, tenant: &TenantId) -> Result<bool> {
        let existed = self.0.coordinator.remove(&key(tenant)).await?;
        self.0.keys.forget(tenant);
        self.0.opened().followers.remove(tenant);
        let opening = self.0.opened().tenants.remove(tenant);
        if let Some(opened) = opening.as_ref().and_then(|opening| opening.get()) {
            opened.hold.lose();
        }
        // What is left of its history is forgotten by its purge, under the
        // claim that purges it, a batch at a time.
        Ok(existed)
    }

    /// Put `tenant` back as it stood at `point`, which must lie within its
    /// restore window, as a new commit, for `request`. Returns what it went
    /// back to and the commit that put it back. The restore goes on even if
    /// the caller stops waiting for it, until it ends or this node's hold
    /// on the tenant is lost, and the request made again is answered as it
    /// was, here or on any node, not made twice; another restore of the
    /// tenant is refused while it runs. One that fails or is stopped once
    /// its commit was begun is answered as made if the coordinator has it,
    /// and otherwise as a failure shared, its outcome unknown.
    pub async fn restore(
        &self,
        tenant: &TenantId,
        point: Point,
        request: RequestId,
    ) -> Result<Restored> {
        let asking = Asking::of(request, Target::Point(point));
        let restored = self
            .restoring(tenant, asking, move |node, opened| async move {
                let id = opened.id();
                let window = node.0.restore_window(id).await?;
                let target = node
                    .0
                    .coordinator
                    .restore_point(&key(id), point, window)
                    .await?
                    .ok_or_else(|| NodeError::NoRestorePoint(id.clone()))?;
                let restore = asking.to(target.position, target.at);
                opened.restored(Some(target.head), restore, None).await
            })
            .await?;
        tracing::info!(%tenant, ?restored, "restored");
        Ok(restored)
    }

    /// Restore `tenant` as `work` does with the tenant as it is opened, as
    /// [`Tenant::restoring`] does, asking again of the tenant opened afresh
    /// whenever a try of the same request that ran meanwhile ended without
    /// being recorded, as one whose writer closed under it. Whoever asks
    /// makes the restore's work once at most: only a caller that waited for
    /// another's try asks again, and one that has waited for
    /// [`RESTORE_WAITS`] of them is told a restore is running instead, to
    /// ask again later.
    async fn restoring<F>(
        &self,
        tenant: &TenantId,
        asking: Asking,
        work: impl Fn(Node, Arc<Tenant>) -> F,
    ) -> Result<Restored>
    where
        F: Future<Output = Result<Restored>> + Send + 'static,
    {
        for _ in 0..=RESTORE_WAITS {
            let opened = self.tenant(tenant).await?;
            let made = work(self.clone(), opened.clone());
            if let Some(restored) = opened.restoring(&self.0.coordinator, asking, made).await? {
                return Ok(restored);
            }
        }
        Err(NodeError::Restoring(tenant.clone()))
    }

    /// The earliest and the latest state `tenant` may be restored to, with
    /// how far back it may go; `None` if it does not exist.
    pub async fn restore_points(
        &self,
        tenant: &TenantId,
    ) -> Result<Option<(Duration, Option<(RestorePoint, RestorePoint)>)>> {
        if self.0.coordinator.committed(&key(tenant)).await?.is_none() {
            return Ok(None);
        }
        let window = self.0.restore_window(tenant).await?;
        let points = self
            .0
            .coordinator
            .restore_points(&key(tenant), window)
            .await?;
        Ok(Some((window, points)))
    }

    /// Set how far back `tenant` may be restored, or with `None` go back to
    /// the node's default. Returns whether there is such a tenant.
    pub async fn set_restore_window(
        &self,
        tenant: &TenantId,
        window: Option<Duration>,
    ) -> Result<bool> {
        Ok(self
            .0
            .coordinator
            .set_restore_window(tenant, window)
            .await?)
    }

    /// Up to `limit` tenants after `after`, in id order; fewer where a
    /// tenant has several partitions.
    pub async fn tenants(&self, after: Option<&TenantId>, limit: usize) -> Result<Vec<TenantId>> {
        let after = after.map(|tenant| PartitionKey {
            tenant: tenant.clone(),
            partition: Partition(u32::MAX),
        });
        let keys = self.0.coordinator.partitions(after.as_ref(), limit).await?;
        let mut tenants: Vec<TenantId> = keys.into_iter().map(|key| key.tenant).collect();
        tenants.dedup();
        Ok(tenants)
    }

    /// The tenant, opened if this is the first call for it or its writer
    /// was lost. Callers arriving together share one opening, which runs
    /// to its end even if they all stop waiting for it. A tenant the node
    /// does not serve is refused before anything is kept for it or its
    /// lease is asked for.
    pub async fn tenant(&self, tenant: &TenantId) -> Result<Arc<Tenant>> {
        self.0.served(tenant)?;
        let opening = {
            let mut open = self.0.opened();
            if open.stopped {
                return Err(NodeError::Stopped);
            }
            let lost = open
                .tenants
                .get(tenant)
                .and_then(|opening| opening.get())
                .filter(|tenant| !tenant.held())
                .cloned();
            if let Some(lost) = lost {
                lost.hold.lose();
                open.tenants.remove(tenant);
            }
            open.tenants.entry(tenant.clone()).or_default().clone()
        };
        let (inner, id, shared) = (self.0.clone(), tenant.clone(), opening.clone());
        let opened = tokio::spawn(async move {
            let opened = shared.get_or_try_init(|| inner.opening(&id)).await;
            if opened.is_ok() {
                inner.opened().followers.remove(&id);
            }
            opened.cloned()
        })
        .await
        .unwrap_or_else(|stopped| match stopped.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            Err(_) => Err(NodeError::Stopped),
        });
        if let Ok(tenant) = &opened {
            tenant.touch();
            self.bound();
        }
        if opened.is_err() {
            let mut open = self.0.opened();
            let unopened = open
                .tenants
                .get(tenant)
                .is_some_and(|held| Arc::ptr_eq(held, &opening) && held.get().is_none());
            if unopened {
                open.tenants.remove(tenant);
            }
        }
        opened
    }

    /// Give up every tenant, so that another node can write for them at
    /// once rather than when the leases run out, and open no more.
    pub async fn shutdown(&self) {
        // Warming stopped first, so that what was served until now, and
        // what it had yet to warm, is the last the list is written with.
        self.remember_warm_last().await;
        let openings: Vec<Opening> = {
            let mut open = self.0.opened();
            open.stopped = true;
            open.followers.clear();
            open.tenants.drain().map(|(_, opening)| opening).collect()
        };
        for opening in openings {
            let unopened = || async { Err(NodeError::Stopped) };
            if let Ok(tenant) = opening.get_or_try_init(unopened).await {
                tenant.hold.lose();
                self.0.release(tenant.writer.lease()).await;
            }
        }
        if let Err(error) = self.0.coordinator.depart(&self.0.settings.node).await {
            tracing::warn!(%error, "the node could not take itself off the register");
        }
    }
}

impl Inner {
    fn opened(&self) -> std::sync::MutexGuard<'_, Open> {
        self.open.lock().expect("tenants lock")
    }

    /// Note that the cluster writes `format`, saying so plainly the first
    /// time it is newer than this release reads.
    fn saw_format(&self, format: u16) {
        let before = self.cluster_format.fetch_max(format, Ordering::SeqCst);
        self.writes_format
            .fetch_max(format.min(FORMAT_VERSION), Ordering::SeqCst);
        if format > FORMAT_VERSION && before <= FORMAT_VERSION {
            tracing::error!(
                cluster = format,
                reads = FORMAT_VERSION,
                "the cluster writes a file format newer than this release reads; \
                 this node is not ready, and is to be replaced by a newer release"
            );
        }
    }

    /// Whether this release reads the newest format the cluster was seen
    /// to write.
    fn readable(&self) -> Result<()> {
        let cluster = self.cluster_format.load(Ordering::SeqCst);
        if cluster > FORMAT_VERSION {
            return Err(NodeError::Outdated {
                cluster,
                reads: FORMAT_VERSION,
            });
        }
        Ok(())
    }

    /// How often tenants are looked over for letting go.
    fn sweep_every(&self) -> Duration {
        (self.settings.tenant_idle / 4).clamp(Duration::from_secs(1), Duration::from_secs(10))
    }

    /// Let go of each tenant unused for longer than
    /// [`Settings::tenant_idle`], and of the least recently used beyond
    /// [`Settings::memory`]; for a tenant written for, flush what it holds
    /// in memory, so that whoever opens it next need not read its log, and
    /// release its lease. None used within [`RESIDENT_AT_LEAST`] is let go,
    /// so that tenants in use are not opened over and over, and none a
    /// request holds or is opening.
    ///
    /// Over the budget, a tenant chosen that something else holds, as its
    /// maintenance does while it merges, has its merge stopped, and is let
    /// go at a later round, once its build has ended and nothing else
    /// holds it: the sweep cannot tell maintenance from a request, so it
    /// waits on neither. What a stopped merge was reckoned to hold counts
    /// against the budget until its work has ended (merge note, §7.7).
    async fn sweep(&self) {
        self.sweeps_begun.fetch_add(1, Ordering::Relaxed);
        *self.last_swept.lock().expect("last swept lock") = Some(Instant::now());
        let held = self.held_now();
        *self.census.lock().expect("census lock") = Census {
            swept_at: Some(Instant::now()),
            ..Census::of(&held)
        };
        let (idle, recent) = (self.settings.tenant_idle, RESIDENT_AT_LEAST);
        let draining = self.merge_builds.draining();
        let total = held.iter().map(|held| held.size).sum::<usize>() + draining;
        let mut idle: Vec<&Held> = held
            .iter()
            .filter(|held| matches!(held.kind, Kind::Writer(..)) && held.unused >= idle)
            .collect();
        // The least recently used first, as the merges of those kept are
        // stopped in this order.
        idle.sort_by_key(|held| std::cmp::Reverse(held.unused));
        let mut older: Vec<&Held> = match self.settings.memory {
            Some(_) => held
                .iter()
                .filter(|held| {
                    held.unused >= recent && !idle.iter().any(|c| std::ptr::eq(*c, *held))
                })
                .collect(),
            None => Vec::new(),
        };
        // Copies warming opened that nobody has read go first.
        older.sort_by_key(|held| (!held.warmed, std::cmp::Reverse(held.unused)));
        let hold = self.sweep_hold.lock().expect("sweep hold lock").clone();
        if let Some(hold) = hold {
            hold.reached_here().await;
        }
        let taken = self.taken_out(&idle, &older, total);
        self.held_elsewhere
            .fetch_add(taken.held_elsewhere, Ordering::Relaxed);
        self.let_go_idle.fetch_add(taken.idle, Ordering::Relaxed);
        self.let_go_for_budget
            .fetch_add(taken.for_budget, Ordering::Relaxed);
        if let Some(budget) = self.settings.memory {
            // Only what was taken out comes off: what something else held
            // is held still.
            let left = total - taken.freed;
            if left > budget {
                // Still over with all that could go taken out: the merges of
                // tenants chosen that something else holds are stopped, the
                // least recently used first, as many as cover what is over,
                // so that each can be let go at a later round. None is
                // stopped while letting others go is enough.
                let mut over = left - budget;
                for (tenant, size) in &taken.kept {
                    if over == 0 {
                        break;
                    }
                    tracing::debug!(tenant = %tenant.id, "over the memory budget: a merge is stopped so that the tenant can be let go");
                    if tenant.writer.stop_merges() {
                        self.stopped_for_budget.fetch_add(1, Ordering::Relaxed);
                    }
                    over = over.saturating_sub(*size);
                }
                tracing::warn!(
                    budget,
                    held = left,
                    draining,
                    "over the memory budget with nothing more that can be let go now"
                );
            }
        }
        for tenant in taken.going {
            if tenant.read(State::unflushed) > 0 {
                if let Err(error) = tenant.flush_to_let_go().await {
                    tracing::warn!(tenant = %tenant.id, %error, "flush before letting go failed");
                }
            }
            tenant.hold.lose();
            self.release(tenant.writer.lease()).await;
            tracing::info!(tenant = %tenant.id, "let go of a tenant");
        }
    }

    /// Sweep soon, one at a time, so that each finds held only what
    /// something else holds and publishes its census in turn; and those
    /// asked for while one runs coalesced into one more, as it ends. Every
    /// use of a tenant over the budget asks for one, and a sweep can wait
    /// long on a flush or on releasing a lease: queued, they would be one
    /// for each request, each holding the node. At most one runs and one
    /// waits to, spaced from the last (see [`Self::spaced`]) before it takes
    /// the lock, so that a sweep the lock is wanted for meanwhile, as
    /// [`Node::sweep_now`]'s, never waits on the spacing.
    ///
    /// The ask is marked before the lock is tried, and the sweep running
    /// looks for it again once it has let the lock go, so that none is
    /// lost between its last look and its letting go.
    fn sweep_soon(self: &Arc<Self>) {
        self.sweep_again.store(true, Ordering::SeqCst);
        // A sweep running looks for the ask once it has let the lock go.
        if self.sweeping.try_lock().is_err() {
            return;
        }
        // One waiting to sweep already, which will find the ask.
        if self.sweep_queued.swap(true, Ordering::SeqCst) {
            return;
        }
        let inner = self.clone();
        tokio::spawn(async move {
            inner.spaced().await;
            inner.sweep_queued.store(false, Ordering::SeqCst);
            // Taken by a sweep meanwhile, which looks for the ask as it
            // lets the lock go.
            if let Ok(sweeping) = inner.sweeping.clone().try_lock_owned() {
                inner.sweeping_while_asked(sweeping).await;
            }
        });
    }

    /// Sweep if a sweep is asked for, holding `sweeping`, and then let it
    /// go; and ask once more, spaced, if one was asked for as it ran or as
    /// it was let go.
    async fn sweeping_while_asked(self: Arc<Self>, sweeping: tokio::sync::OwnedMutexGuard<()>) {
        if self.sweep_again.swap(false, Ordering::SeqCst) {
            self.sweep().await;
        }
        drop(sweeping);
        if self.sweep_again.load(Ordering::SeqCst) {
            self.sweep_soon();
        }
    }

    /// Wait until [`SWEEPS_SPACED`] has passed since the last sweep began,
    /// unless the node's own sweep is asked for, and give the worker up
    /// either way, so that sweeps asked for without let-up never hold it.
    async fn spaced(&self) {
        let on_time = self.sweep_on_time.swap(false, Ordering::SeqCst);
        let last = *self.last_swept.lock().expect("last swept lock");
        if let Some(last) = last.filter(|_| !on_time) {
            tokio::time::sleep_until(last + SWEEPS_SPACED).await;
        }
        tokio::task::yield_now().await;
    }

    /// What the tenants and copies held take, read under the node's lock
    /// from what each published as it last changed, holding none of them
    /// and waiting on no lock of theirs: see [`Node::resident`].
    fn resident(&self) -> usize {
        let open = self.opened();
        let writers: usize = (open.tenants.values())
            .filter_map(|opening| opening.get())
            .map(|tenant| tenant.figures().resident)
            .sum();
        let copies: usize = (open.followers.values())
            .filter_map(|following| following.get())
            .map(|copy| copy.resident())
            .sum();
        writers + copies
    }

    /// Call `f` with each tenant written for, under the node's lock and
    /// holding none of them: a sweep lets a tenant go only if nothing but
    /// the node holds it, so reading one must never hold it. `f` reads
    /// what each published as it last changed, never waiting on its lock.
    fn each_writer(&self, mut f: impl FnMut(&Tenant)) {
        let open = self.opened();
        for tenant in open.tenants.values().filter_map(|opening| opening.get()) {
            f(tenant);
        }
    }

    /// What the node holds now, sized and aged without holding the lock.
    fn held_now(&self) -> Vec<Held> {
        let held: Vec<(TenantId, Kind)> = {
            let open = self.opened();
            let writers = open.tenants.iter().filter_map(|(id, opening)| {
                let tenant = opening.get()?.clone();
                Some((id.clone(), Kind::Writer(opening.clone(), tenant)))
            });
            let copies = open.followers.iter().filter_map(|(id, following)| {
                let copy = following.get()?.clone();
                Some((id.clone(), Kind::Copy(following.clone(), copy)))
            });
            writers.chain(copies).collect()
        };
        held.into_iter()
            .map(|(id, kind)| {
                let (unused, size, warmed) = match &kind {
                    Kind::Writer(_, tenant) => (tenant.idle(), tenant.figures().resident, false),
                    Kind::Copy(_, copy) => (copy.idle(), copy.resident(), copy.warmed().is_some()),
                };
                Held {
                    id,
                    kind,
                    unused,
                    size,
                    warmed,
                }
            })
            .collect()
    }

    /// Take out of what the node holds each of `idle`, and then, while what
    /// is left of `total` is over the budget, each of `older` in turn: each
    /// only if it is still the one held and nobody but the node and this
    /// sweep has it, nor is opening it. Those something else holds are kept,
    /// and the next is tried, so that what can go goes first.
    fn taken_out(&self, idle: &[&Held], older: &[&Held], total: usize) -> Taken {
        let mut open = self.opened();
        let mut taken = Taken::default();
        for held in idle {
            taken.take(&mut open, held, false);
        }
        if let Some(budget) = self.settings.memory {
            for held in older {
                if total - taken.freed <= budget {
                    break;
                }
                taken.take(&mut open, held, true);
            }
        }
        taken
    }

    /// Refuse `tenant` if this node does not serve it, before anything is
    /// done or counted for it.
    fn served(&self, tenant: &TenantId) -> Result<()> {
        match self.settings.served.serves(tenant) {
            true => Ok(()),
            false => Err(NodeError::NotServedHere {
                tenant: tenant.clone(),
            }),
        }
    }

    /// Take `cost` of what `tenant` may ask for now.
    fn admit(&self, tenant: &TenantId, asked: Asked, cost: usize) -> Result<()> {
        self.limits.admit(tenant, asked, cost)
    }

    async fn release(&self, lease: &Lease) {
        if let Err(error) = self.coordinator.release(lease).await {
            tracing::warn!(partition = %lease.key, %error, "lease not released");
        }
    }

    async fn opening(&self, tenant: &TenantId) -> Result<Arc<Tenant>> {
        let Settings {
            node,
            lease,
            group,
            queue,
            flush_rows,
            flush_backoff,
            ..
        } = &self.settings;
        let (lease, committed) = match self.coordinator.acquire(&key(tenant), node, *lease).await {
            Ok(Acquired::Lease { lease, committed }) => (lease, committed),
            Ok(Acquired::Held { holder, .. }) => {
                return Err(NodeError::Elsewhere {
                    tenant: tenant.clone(),
                    holder,
                })
            }
            Err(CoordinatorError::UnknownPartition(_)) => {
                return Err(NodeError::UnknownTenant(tenant.clone()))
            }
            Err(error) => return Err(error.into()),
        };
        let hold = Arc::new(Hold::default());
        let deadline = Arc::new(Mutex::new(lease.deadline()));
        self.renewing(lease.clone(), hold.clone(), &deadline);
        let opened = Writer::open(
            self.blobs.clone(),
            self.coordinator.clone(),
            lease.clone(),
            &committed,
        )
        .await;
        let still_held =
            !hold.is_lost() && Instant::now() < *deadline.lock().expect("deadline lock");
        let writer = match opened {
            Ok(writer) if !self.opened().stopped && still_held => {
                self.writers_opened.fetch_add(1, Ordering::Relaxed);
                let documents = Arc::new(Documents(self.embedders.clone()));
                Arc::new(
                    writer
                        .writing_format(self.writes_format.clone())
                        .merging_among(self.merge_builds.clone())
                        .measured_by(self.timings.clone())
                        .flushing_due_at(*flush_rows, self.settings.backlog())
                        .embedding_with(documents)
                        .transferring(self.settings.transfers)
                        .timed_by(self.settings.clock.clone()),
                )
            }
            Ok(_) if self.opened().stopped => {
                self.release(&lease).await;
                return Err(NodeError::Stopped);
            }
            Ok(_) => {
                self.release(&lease).await;
                return Err(NodeError::Lost(tenant.clone()));
            }
            Err(error) => {
                self.release(&lease).await;
                return Err(error.into());
            }
        };
        let (flushes, wanted) = mpsc::channel(1);
        let commits = GroupCommitter::spawn(writer.clone(), *group, *queue);
        hold.opened(&writer, &commits);
        let opened = Arc::new(Tenant {
            id: tenant.clone(),
            commits,
            writer,
            embedders: self.embedders.clone(),
            flush_rows: *flush_rows,
            backlog_rows: self.settings.backlog(),
            flush_backoff: *flush_backoff,
            clean_at_least_every: self.settings.clean_at_least_every,
            cleaned_at: Mutex::new(Instant::now()),
            backoff: Mutex::new((0, None)),
            flushing: self.flushing.clone(),
            flushes,
            settling: Settling::default(),
            hold,
            deadline,
            used: Mutex::new(Instant::now()),
            limits: self.limits.clone(),
            running: Mutex::default(),
        });
        self.maintaining(&opened, wanted);
        Ok(opened)
    }

    /// Flush the tenant when asked to and it is still due, and merge and
    /// clean up its files after each flush and now and then regardless,
    /// for as long as the tenant is open and held. The task holds no
    /// reference of its own between rounds, so dropping the tenant ends
    /// it, and losing the hold stops it wherever it is.
    fn maintaining(&self, tenant: &Arc<Tenant>, mut wanted: mpsc::Receiver<()>) {
        let Settings {
            restore_window,
            grace,
            ..
        } = self.settings;
        let every =
            (restore_window / 24).clamp(Duration::from_secs(60), Duration::from_secs(60 * 60));
        let coordinator = self.coordinator.clone();
        let hold = tenant.hold.clone();
        let tenant = Arc::downgrade(tenant);
        let maintaining = tokio::spawn(async move {
            loop {
                // While flushes fail, the next is tried as soon as its
                // backoff lets it, whether or not a write asks for it.
                let retried_at = tenant
                    .upgrade()
                    .and_then(|tenant| tenant.flush_retried_at());
                let wait = retried_at.map_or(every, |at| {
                    every.min(at.saturating_duration_since(Instant::now()))
                });
                let flush = match tokio::time::timeout(wait, wanted.recv()).await {
                    Ok(Some(())) => true,
                    Ok(None) => return,
                    Err(_) => false,
                };
                let Some(tenant) = tenant.upgrade() else {
                    return;
                };
                if !tenant.held() {
                    return;
                }
                // One that failed is tried again on its own, if no write
                // asks, but never sooner than its backoff lets it. One put
                // off for that still lets the tenant be looked after.
                let due = flush || tenant.flush_retried_at().is_some();
                if due
                    && tenant.read(State::unflushed) >= tenant.flush_rows
                    && tenant.flush_may_be_tried()
                {
                    if let Err(error) = tenant.flush().await {
                        tracing::error!(tenant = %tenant.id, %error, "flush failed; rows wait in memory");
                        continue;
                    }
                }
                let through = tenant.settling.flushed.load(Ordering::SeqCst);
                let maintained = tenant.maintained(coordinator.as_ref(), restore_window, grace);
                match maintained.await {
                    Ok(deleted) => {
                        tenant
                            .settling
                            .maintained
                            .fetch_max(through, Ordering::SeqCst);
                        if deleted > 0 {
                            tracing::info!(tenant = %tenant.id, deleted, "cleaned up");
                        }
                    }
                    Err(error) => tracing::warn!(tenant = %tenant.id, %error, "compaction failed"),
                }
            }
        });
        hold.stops(maintaining.abort_handle());
    }

    /// How far back `tenant` may be restored.
    async fn restore_window(&self, tenant: &TenantId) -> Result<Duration> {
        restore_window(
            self.coordinator.as_ref(),
            tenant,
            self.settings.restore_window,
        )
        .await
    }

    /// Keep the hold the tenant was taken with for as long as `deadline`
    /// is held, from the opening on, and note when the hold is lost. Each
    /// renewal is timed from when the one before was asked for, so a slow
    /// answer does not let the hold run out.
    ///
    /// The hold is given up once its writer, when it is open, is closed, as
    /// by an attempt that failed or a caller that stopped waiting part way,
    /// or has been publishing a commit for longer than its bounds allow:
    /// a writer that can write no more would otherwise keep the tenant
    /// from every other node.
    fn renewing(&self, lease: Lease, hold: Arc<Hold>, deadline: &Arc<Mutex<Instant>>) {
        let coordinator = self.coordinator.clone();
        let ttl = self.settings.lease;
        let every = ttl / 3;
        let deadline = Arc::downgrade(deadline);
        tokio::spawn(async move {
            let mut next = lease.requested_at + every;
            loop {
                tokio::time::sleep_until(next).await;
                let Some(deadline) = deadline.upgrade() else {
                    return;
                };
                if hold.is_lost() {
                    return;
                }
                let unusable = hold
                    .writer
                    .get()
                    .and_then(Weak::upgrade)
                    .is_some_and(|writer| writer.is_closed() || writer.stalled());
                if unusable {
                    tracing::warn!(partition = %lease.key, "the writer is closed or stuck; the hold on the tenant is given up");
                    hold.lose();
                    return;
                }
                let asked = Instant::now();
                match tokio::time::timeout(every, coordinator.renew(&lease, ttl)).await {
                    Ok(Ok(renewed)) => {
                        *deadline.lock().expect("deadline lock") = renewed.deadline();
                        next = renewed.requested_at + every;
                    }
                    Ok(Err(error)) if !error.is_retryable() => {
                        tracing::warn!(partition = %lease.key, %error, "hold on the tenant lost");
                        hold.lose();
                        return;
                    }
                    Ok(Err(error)) => {
                        tracing::warn!(partition = %lease.key, %error, "renewing the hold failed");
                        next = asked + (every / 4);
                    }
                    Err(_) => {
                        tracing::warn!(partition = %lease.key, "renewing the hold took too long");
                        next = asked + (every / 4);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use vellev_core::EmbeddingProvider;

    use super::*;

    #[test]
    fn a_restore_is_asked_for_to_the_microsecond_as_it_is_recorded() {
        let taken_at = DateTime::<Utc>::from_timestamp(1_760_000_000, 1_500).unwrap();
        let asking = Asking::of(RequestId::generate(), Target::Backup(uuid::Uuid::nil()));
        let restore = asking.to(LogPosition(1), taken_at);
        assert_eq!(restore.from_at.timestamp_subsec_nanos(), 1_000);
        assert_eq!(restore.from_at.timestamp(), 1_760_000_000);
    }

    #[tokio::test]
    async fn a_collection_whose_clients_supply_the_vectors_is_never_embedded_for() {
        let supplied = Embedding {
            provider: EmbeddingProvider::None,
            model: "client".into(),
            dimensions: 2,
            query_prefix: None,
            document_prefix: None,
            properties: vec!["title".into()],
        };
        let documents = Documents(Arc::new(Embedders::new(Default::default())));
        assert_eq!(
            documents.vectors(&supplied, vec!["fox".into()]).await,
            Err(Unembedded::for_good(
                ModelFault::Setup,
                "its collection's clients supply the vectors"
            ))
        );
    }
}

#[cfg(test)]
mod segment_tests {
    use super::*;
    use vellev_storage::format::SegmentMeta;

    fn meta(rows: u64, clustered: bool, merged: bool) -> SegmentMeta {
        let marker = if merged { "-merged" } else { "" };
        SegmentMeta {
            key: format!("segment/{rows}{marker}.segment"),
            rows,
            tombstones: 0,
            index: clustered.then(|| format!("index-{rows}")),
            vectors: Some(format!("vectors-{rows}")),
            probes: vec![],
        }
    }

    /// A segment with an index file is counted as clustered, and one no
    /// merge wrote as awaiting a merge, whichever it is.
    #[test]
    fn a_segment_with_an_index_file_is_counted_as_clustered() {
        // Two tenants' figures, the first of two segments, added up.
        let (mut first, mut second) = (vellev_storage::Figures::default(), Default::default());
        for (rows, clustered, merged) in [(800_000, true, true), (200_000, false, false)] {
            first.count(&meta(rows, clustered, merged));
        }
        vellev_storage::Figures::count(&mut second, &meta(50_000, false, false));
        let mut segments = Segments::default();
        segments.add(&first);
        segments.add(&second);
        assert_eq!(
            segments,
            Segments {
                clustered: 1,
                clustered_rows: 800_000,
                plain: 2,
                plain_rows: 250_000,
                unmerged: 2,
                unmerged_rows: 250_000,
                probes: Probes::default(),
            }
        );
    }
}
