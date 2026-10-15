//! Who may write to a partition, and what is committed.
//!
//! Correctness rests on the generation, not on clocks. Registering a
//! partition, and taking or releasing its lease, each move it to a fresh
//! generation, and renew and commit
//! succeed only while the lease's generation is still current, so a writer
//! that stalls past its lease and carries on is fenced the moment it tries
//! to commit. Generations come from one sequence per coordinator, so a
//! generation is never seen twice, even after a partition is removed and
//! registered again.

mod bounded;
mod memory;
mod postgres;

use std::fmt;
use std::time::Duration;

use tokio::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use vellev_core::{NodeId, Partition, RequestId, TenantId};

pub use bounded::Bounded;
pub use memory::{Clock, ManualClock, MemoryCoordinator, SystemClock};
pub use postgres::PostgresCoordinator;

/// Longer than any sensible lease, and short enough that expiry arithmetic
/// cannot overflow in either implementation.
pub const MAX_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PartitionKey {
    pub tenant: TenantId,
    pub partition: Partition,
}

impl fmt::Display for PartitionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.tenant, self.partition.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Generation(pub u64);

/// What a partition is from its registration to its removal: a partition
/// removed and registered again has a greater one, and its positions start
/// again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Incarnation(pub u64);

/// Which cluster a cluster is: an id it is given once, when its database is
/// first made, and keeps. It is written into the backups the cluster takes
/// and into where they lie, so that no cluster deletes what another took.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClusterId(pub Uuid);

impl fmt::Display for ClusterId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.simple())
    }
}

/// Which tenant a tenant is, wherever it is: given it when it is first
/// made, kept for as long as it lives, and written into its backups, so
/// that a tenant made later under the same name is never taken for it. It
/// is not the incarnation, which is one cluster's own count: a cluster that
/// lost a tenant and adopts a backup of it takes its lineage up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Lineage(pub Uuid);

impl fmt::Display for Lineage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.simple())
    }
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LogPosition(pub u64);

impl LogPosition {
    pub fn next(self) -> Option<LogPosition> {
        self.0.checked_add(1).map(LogPosition)
    }
}

/// The object key of the newest committed file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManifestRef(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Committed {
    /// The partition's generation when this was read. It moves with every
    /// lease taken or released, not only with commits.
    pub generation: Generation,
    pub incarnation: Incarnation,
    pub log_position: LogPosition,
    pub manifest: Option<ManifestRef>,
}

/// Where a shipped timeline begins (DW-668): the position `B` its first
/// link is at, and the checkpoint `C` whose restore starts there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShipBase {
    pub position: LogPosition,
    pub checkpoint: ManifestRef,
    pub checkpoint_at: LogPosition,
}

/// How a partition's history is shipped to the backup store (DW-668), in
/// its current incarnation.
///
/// - `hold_from` and `cap`: shipping's hold, taken before a base is
///   chosen. Commit rows at and above `hold_from` are kept from then on,
///   and a clean records nothing above `cap`, the newest checkpoint when
///   the hold was taken, or 0 if there was none.
/// - `base`: the base of the run of shipping now under way, raised only.
/// - `shipped` and `stalled`: that run's own, fenced by its base: the
///   newest position up to which every commit from the base on has been
///   shipped, only ever raised, and why it stopped, if it has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shipping {
    pub incarnation: Incarnation,
    pub hold_from: Option<LogPosition>,
    pub cap: Option<LogPosition>,
    pub base: Option<ShipBase>,
    pub shipped: Option<LogPosition>,
    pub stalled: Option<String>,
}

/// A partition with commits its run of shipping has yet to ship: the
/// run's incarnation and base, what it has shipped, and the position
/// committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unshipped {
    pub key: PartitionKey,
    pub incarnation: Incarnation,
    pub base: LogPosition,
    pub shipped: Option<LogPosition>,
    pub committed: LogPosition,
}

/// Why a shipping hold or base was refused as stale: what it named no
/// longer stands, and the caller reads again and asks anew.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stale {
    /// A clean has recorded it deletes above the checkpoint named.
    CleanedAbove,
    /// The checkpoint named is above what is committed.
    AboveCommitted,
    /// No hold is taken, which a base needs.
    NotHeld,
    /// The hold is above the base named.
    HoldAbove,
    /// The cap is above the base's checkpoint, so it would be lowered.
    CapAbove,
    /// No commit row is at the base named.
    NoRowAtBase,
    /// No commit row at the checkpoint's position names its key.
    CheckpointNotThere,
}

/// What a clean of a partition last recorded it deletes below: the key of
/// the checkpoint that was its base, and that checkpoint's position. Every
/// checkpoint a clean has deleted lies below it, since a clean records it
/// before its first delete and it is only ever raised (DW-681).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cleaned {
    pub base: ManifestRef,
    pub position: LogPosition,
}

/// What a commit was built on: the position and the head committed just
/// before it, as its writer had them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prior {
    pub position: LogPosition,
    pub head: Option<ManifestRef>,
}

/// A commit, and what it follows. A coordinator makes it only if `after`
/// is what is committed still, in the one step in which it checks the
/// lease, so that of two writers under one lease, or of a writer and
/// whatever else moved the partition on, only the one that saw all there
/// was commits: the other is told its commit does not follow. A lease is a
/// fence between generations and not within one, and a commit that
/// advances the position is no proof its writer saw the position before
/// it, since a restore commits past whatever it goes back to.
///
/// The very commit made already is taken again as made, whatever it
/// followed, so that a writer whose answer was lost may ask again.
///
/// With no `after` the commit is made on whatever is committed, as every
/// commit was by releases before this one; a writer always says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    pub after: Option<Prior>,
    pub log_position: LogPosition,
    pub manifest: Option<ManifestRef>,
    /// The restore the commit makes, if it makes one, to be recorded with
    /// it.
    pub restore: Option<Restore>,
    /// The lineage the commit makes its tenant, if it takes a backup up:
    /// the one noted by [`Coordinator::adopt_on_commit`] for the lease
    /// that commits, or the commit is refused, so that a restore meant to
    /// adopt never lands as a plain one.
    pub adopting: Option<Lineage>,
}

/// The last commit of a removed partition forgotten, from where to go on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Forgotten {
    pub incarnation: Incarnation,
    pub position: LogPosition,
}

/// A restore as the request that asked for it names it, recorded with the
/// commit that makes it, so that the request made again, on any node, is
/// answered as it was rather than made twice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restore {
    pub request: RequestId,
    /// A digest of what the request asked to go back to, so that the same
    /// request asking for another state is told so rather than answered.
    pub asked: [u8; 32],
    /// The position gone back to.
    pub from: LogPosition,
    /// When the state gone back to was committed, or backed up.
    pub from_at: DateTime<Utc>,
}

/// A restore made: what it went back to, and the commit that put it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Restored {
    pub restore: Restore,
    pub incarnation: Incarnation,
    pub position: LogPosition,
}

/// The right to write a partition. `expires_at` is coordinator time; a holder
/// measures its own deadline with [`Lease::deadline`], from an instant taken
/// before the request was sent, so clock skew never extends a lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub key: PartitionKey,
    pub holder: NodeId,
    pub generation: Generation,
    pub ttl: Duration,
    pub expires_at: DateTime<Utc>,
    pub requested_at: Instant,
}

impl Lease {
    pub fn deadline(&self) -> Instant {
        self.requested_at + self.ttl
    }

    pub fn fenced_by(&self, current: Generation) -> CoordinatorError {
        CoordinatorError::Fenced {
            key: self.key.clone(),
            held: self.generation,
            current,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Acquired {
    Lease {
        lease: Lease,
        committed: Committed,
    },
    Held {
        holder: NodeId,
        expires_at: DateTime<Utc>,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("partition {0} is not registered")]
    UnknownPartition(PartitionKey),
    #[error(
        "tenant {tenant} was removed and what was kept of it is being deleted; \
         it can be made again, empty, once that is done"
    )]
    Purging {
        tenant: TenantId,
        /// How long ago its purge was asked for.
        asked: Duration,
    },
    #[error(
        "tenant {0} is taking up a backup, and until that is done or given up \
         commits nothing else; nor does it take one up once it has keys or has \
         been written to"
    )]
    Adopting(TenantId),
    #[error(
        "the commit says it takes a backup up for tenant {0}, and no adoption is noted \
         for the lease that makes it"
    )]
    Unadopted(TenantId),
    #[error(
        "lease on {key} at generation {held:?} is fenced; the current generation is {current:?}"
    )]
    Fenced {
        key: PartitionKey,
        held: Generation,
        current: Generation,
    },
    #[error(
        "commit at {attempted:?} does not follow what is committed to {key}, at {committed:?}"
    )]
    Regressed {
        key: PartitionKey,
        committed: LogPosition,
        attempted: LogPosition,
    },
    /// Shipping was asked about another incarnation than the partition's
    /// current one: by a shipper of one removed, or not yet seen made
    /// again.
    #[error("partition {key} is at incarnation {current:?}, not {asked:?}")]
    OtherIncarnation {
        key: PartitionKey,
        asked: Incarnation,
        current: Incarnation,
    },
    /// A shipper's write names another base than the run under way's, or
    /// none is under way.
    #[error("shipping of {key} is from {recorded:?}, not from {asked:?}")]
    OtherBase {
        key: PartitionKey,
        asked: LogPosition,
        recorded: Option<LogPosition>,
    },
    /// A shipping hold or base that no longer stands.
    #[error("shipping of {key} refused as stale: {why:?}")]
    StaleShipping { key: PartitionKey, why: Stale },
    #[error("{0}")]
    InvalidArgument(String),
    #[error("coordinator backend failed: {0}")]
    Backend(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl CoordinatorError {
    pub fn backend(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Backend(Box::new(error))
    }

    /// A backend failure or transient contention leaves the outcome unknown
    /// and is worth retrying; every other error is a definite answer.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Backend(_))
    }
}

pub type Result<T> = std::result::Result<T, CoordinatorError>;

/// What a partition's rows say, for a shipping hold or base to be
/// decided on: by the memory coordinator, which applies what it decides,
/// and by Postgres, to say why a conditional write changed nothing.
#[derive(Debug, Clone)]
pub(crate) struct ShipFacts {
    pub shipping: Shipping,
    pub committed: LogPosition,
    pub cleaned: Option<LogPosition>,
    /// Whether a commit row of the incarnation is at the base asked for.
    pub row_at_base: bool,
    /// The head of the commit row at the checkpoint's position asked for.
    pub head_at_checkpoint: Option<ManifestRef>,
}

/// What a shipping write is to do.
#[derive(Debug)]
pub(crate) enum Ship {
    /// Make it.
    Write,
    /// Leave what stands, and return it.
    Stand,
}

fn stale(key: &PartitionKey, why: Stale) -> CoordinatorError {
    CoordinatorError::StaleShipping {
        key: key.clone(),
        why,
    }
}

/// Refused unless `asked` is `current`, the partition's incarnation.
pub(crate) fn same_incarnation(
    key: &PartitionKey,
    asked: Incarnation,
    current: Incarnation,
) -> Result<()> {
    match asked == current {
        true => Ok(()),
        false => Err(CoordinatorError::OtherIncarnation {
            key: key.clone(),
            asked,
            current,
        }),
    }
}

/// Whether a hold with `cap` is taken, given `facts`.
pub(crate) fn hold_decision(
    key: &PartitionKey,
    incarnation: Incarnation,
    cap: LogPosition,
    facts: &ShipFacts,
) -> Result<Ship> {
    same_incarnation(key, incarnation, facts.shipping.incarnation)?;
    if facts.shipping.hold_from.is_some() {
        return Ok(Ship::Stand);
    }
    if facts.cleaned.is_some_and(|cleaned| cleaned > cap) {
        return Err(stale(key, Stale::CleanedAbove));
    }
    if cap > facts.committed {
        return Err(stale(key, Stale::AboveCommitted));
    }
    Ok(Ship::Write)
}

/// Whether `base` is recorded, given `facts`.
pub(crate) fn base_decision(
    key: &PartitionKey,
    incarnation: Incarnation,
    base: &ShipBase,
    facts: &ShipFacts,
) -> Result<Ship> {
    same_incarnation(key, incarnation, facts.shipping.incarnation)?;
    let shipping = &facts.shipping;
    if shipping
        .base
        .as_ref()
        .is_some_and(|recorded| recorded.position >= base.position)
    {
        return Ok(Ship::Stand);
    }
    let (Some(hold), Some(cap)) = (shipping.hold_from, shipping.cap) else {
        return Err(stale(key, Stale::NotHeld));
    };
    let why = if hold > base.position {
        Stale::HoldAbove
    } else if cap > base.checkpoint_at {
        Stale::CapAbove
    } else if facts
        .cleaned
        .is_some_and(|cleaned| cleaned > base.checkpoint_at)
    {
        Stale::CleanedAbove
    } else if !facts.row_at_base {
        Stale::NoRowAtBase
    } else if facts.head_at_checkpoint.as_ref() != Some(&base.checkpoint) {
        Stale::CheckpointNotThere
    } else {
        return Ok(Ship::Write);
    };
    Err(stale(key, why))
}

/// `position` as one noted shipped of the run from `base`: at or above it.
pub(crate) fn checked_note(base: LogPosition, position: LogPosition) -> Result<()> {
    if position < base {
        return Err(CoordinatorError::InvalidArgument(format!(
            "position {} is below the base {} it is noted shipped from",
            position.0, base.0
        )));
    }
    Ok(())
}

/// `base` as one to record: at a commit, and at or below its checkpoint.
pub(crate) fn checked_ship_base(base: &ShipBase) -> Result<()> {
    if base.position == LogPosition(0) || base.position > base.checkpoint_at {
        return Err(CoordinatorError::InvalidArgument(format!(
            "a shipping base at {} with its checkpoint at {}: a base is at a commit, \
             and at or below its checkpoint",
            base.position.0, base.checkpoint_at.0
        )));
    }
    Ok(())
}

fn checked_ttl(ttl: Duration) -> Result<Duration> {
    (ttl <= MAX_TTL).then_some(ttl).ok_or_else(|| {
        CoordinatorError::InvalidArgument(format!("ttl {ttl:?} exceeds {MAX_TTL:?}"))
    })
}

#[async_trait]
pub trait Coordinator: fmt::Debug + Send + Sync {
    /// Which cluster this is.
    async fn cluster(&self) -> Result<ClusterId>;

    /// Create the partition's row at a fresh generation, of its tenant's
    /// lineage, or of a new one if it is the tenant's first. Registering an
    /// existing partition changes nothing; one of a tenant whose files are
    /// still to be purged is refused.
    async fn register(&self, key: &PartitionKey) -> Result<()>;

    /// Which tenant `tenant` is now, if there is one.
    async fn identity(&self, tenant: &TenantId) -> Result<Option<Identity>>;

    /// Have the partition's tenant become the tenant `adoption` says with
    /// the commit that restores it, and not before: as a cluster that lost
    /// a tenant takes it up again from a backup. Until then the tenant is
    /// as it was. The commit at the position `adoption` names, by the
    /// holder of the lease at `generation`, saying it adopts that lineage
    /// ([`Commit::adopting`]), makes the tenant of that lineage, with that
    /// restore window and those keys, all at once with the commit or not
    /// at all; any other commit of the partition is refused meanwhile, and
    /// so is that one if the tenant has since been given a key or written
    /// to elsewhere. An adoption asked for again takes the place of the
    /// one before; one whose lease has moved on is no more. Returns
    /// whether it was noted: not for no partition, another generation, or
    /// a tenant that has committed or has keys.
    async fn adopt_on_commit(
        &self,
        key: &PartitionKey,
        generation: Generation,
        adoption: &Adoption,
    ) -> Result<bool>;

    /// Give up the adoption the partition was to make with a commit by the
    /// lease at `generation`, if that is the one noted: one noted since,
    /// for the lease that took the partition over, is left for its holder
    /// to make or give up.
    async fn adopt_no_more(&self, key: &PartitionKey, generation: Generation) -> Result<()>;

    /// Delete the partition's row. Returns whether it existed. With the
    /// tenant's last partition go its keys, so that none reaches a tenant
    /// made again under the same id, and its files are marked to be purged.
    /// Its history and restores, however many, no longer reach it and are
    /// deleted by [`Self::forget_removed`], a few at a time, so that this
    /// takes as long for a busy partition as for an idle one.
    async fn remove(&self, key: &PartitionKey) -> Result<bool>;

    /// Forget the restores kept of the partition from before it was last
    /// removed, and up to `limit` of its commits from then, the earliest
    /// first after `after`, where the call before got to. Returns where
    /// this one got to, or `None` once none were left. Going on from where
    /// the last call stopped, each takes as long as the first.
    async fn forget_removed(
        &self,
        key: &PartitionKey,
        after: Option<Forgotten>,
        limit: usize,
    ) -> Result<Option<Forgotten>>;

    /// Take the lease if nobody holds it, the holder's lease has expired, or
    /// the holder is `holder` itself (whose earlier lease is then fenced).
    async fn acquire(&self, key: &PartitionKey, holder: &NodeId, ttl: Duration)
        -> Result<Acquired>;

    /// Extend the lease. Succeeds while the generation is still current, even
    /// after the lease expired, because the generation is the fence: an
    /// expired lease nobody has taken over cannot have been written past.
    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease>;

    /// Give the lease up early. A lease that is no longer current is left
    /// alone, so a slow release never ends a newer holder's lease.
    async fn release(&self, lease: &Lease) -> Result<()>;

    /// Record a new committed position and manifest. Fenced when the lease's
    /// generation is no longer current; refused when the position does not
    /// advance. Repeating a commit that already landed succeeds, so a writer
    /// whose response was lost can retry.
    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed>;

    /// The committed state, as a follower reads it.
    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>>;

    /// What a clean of the partition last recorded it deletes below, if
    /// any clean of this incarnation has: see [`Self::clean_below`]. Fails
    /// for a partition not registered.
    async fn cleaned(&self, key: &PartitionKey) -> Result<Option<Cleaned>>;

    /// Record, as the holder of `lease`, that a clean is about to delete
    /// below the checkpoint `base` at `position`, and return the record as
    /// it then stands. Fenced, and nothing recorded, when the lease's
    /// generation is no longer current, as a commit is. The record is only
    /// ever raised: one already at `position` or above is left as it is,
    /// and returned, which is not an error, so that a clean whose base is
    /// below the record deletes on what the record already says. A record
    /// returned that names another key at or above `position` is not of
    /// this partition's chain, and the clean that asked keeps to what it
    /// would do with none. The record goes with the partition's row when
    /// it is removed. Every coordinator has its own: there is no default,
    /// so that one that does not record can never seem to.
    async fn clean_below(
        &self,
        lease: &Lease,
        base: &ManifestRef,
        position: LogPosition,
    ) -> Result<Cleaned>;

    /// How the partition's history is shipped, in its current incarnation.
    /// Fails for a partition not registered.
    async fn shipping(&self, key: &PartitionKey) -> Result<Shipping>;

    /// Take shipping's hold on the partition's incarnation `incarnation`:
    /// from the lowest commit row of it still there (or the position after
    /// the committed one, if none is), rows are kept, and a clean records
    /// nothing above `cap`, the newest checkpoint the caller knows (0 for
    /// none). Taken once per incarnation, in a transaction of its own that
    /// first takes the lock `forget_history` takes, and then reads the
    /// lowest row afresh. A hold already taken is left and returned, which
    /// is not an error. Refused unless `cap` is at or above what a clean
    /// has recorded and at or below what is committed, as stale; and for
    /// another incarnation. Nothing holds a lease to ship, so the
    /// incarnation fences, as for every shipping write. No default on any
    /// coordinator, so that one that does not record can never seem to;
    /// and nothing but tests calls it until `forget_history` honours it.
    async fn hold_for_shipping(
        &self,
        key: &PartitionKey,
        incarnation: Incarnation,
        cap: LogPosition,
    ) -> Result<Shipping>;

    /// Record `base` as the base of a run of shipping of the partition's
    /// incarnation `incarnation`, raise the cap to its checkpoint, and
    /// begin the run's own record, nothing yet shipped or stalled, in the
    /// same transaction, the partition's row first. Raised only: a base at
    /// or above it is left and returned, which is not an error, and the
    /// caller ships from the base it is given back. Refused, as stale,
    /// unless the hold is taken and at or below the base, the cap and
    /// what a clean has recorded are at or below the checkpoint, a commit
    /// row is at the base, and the row at the checkpoint's position names
    /// it; and for another incarnation. That the base is where the
    /// checkpoint's restore starts is for the caller, which has decoded
    /// the checkpoint. No default, as for [`Self::hold_for_shipping`].
    async fn record_base(
        &self,
        key: &PartitionKey,
        incarnation: Incarnation,
        base: &ShipBase,
    ) -> Result<Shipping>;

    /// Note that every commit of the run from `base` up to `position` has
    /// been shipped. Only ever raised: a note at or below what stands is
    /// left, and returned. Fenced by the incarnation and the base on the
    /// run's own record, never through the partition's row: refused for
    /// another incarnation, and for another base or none. No default.
    async fn note_shipped(
        &self,
        key: &PartitionKey,
        incarnation: Incarnation,
        base: LogPosition,
        position: LogPosition,
    ) -> Result<Shipping>;

    /// Note that the run from `base` has stopped, and why. The first
    /// reason stands; a later one is left, and the record returned.
    /// Fenced as [`Self::note_shipped`] is, so that a stale run never
    /// stops the one under way. No default.
    async fn stall_shipping(
        &self,
        key: &PartitionKey,
        incarnation: Incarnation,
        base: LogPosition,
        reason: &str,
    ) -> Result<Shipping>;

    /// Up to `limit` commits of the partition's incarnation `incarnation`
    /// at positions above `after`, oldest first, each with its head and the
    /// time it was committed, by the database's clock, to the microsecond:
    /// what a shipper ships, read from the rows the commit made, never
    /// from a listing. Positions may skip, as a restore commits after the
    /// greater of the current position and the one it restores; none is
    /// missing that is still kept. Refused for another incarnation than
    /// the partition's current one, and for a partition not registered.
    async fn commits_after(
        &self,
        key: &PartitionKey,
        incarnation: Incarnation,
        after: LogPosition,
        limit: usize,
    ) -> Result<Vec<RestorePoint>>;

    /// Up to `limit` partitions after `after`, in key order, whose run of
    /// shipping has commits it has yet to ship: a base is recorded, the run
    /// has not stalled, and the committed position is above what it has
    /// shipped (at or above its base, if it has shipped nothing). A stalled
    /// run waits for an operator, not a shipper. Paged by key, as
    /// [`Self::partitions`] is, so that a sweep takes partitions in turn.
    async fn unshipped(&self, after: Option<&PartitionKey>, limit: usize)
        -> Result<Vec<Unshipped>>;

    /// Up to `limit` registered partitions after `after`, in key order,
    /// tenants comparing byte by byte.
    async fn partitions(
        &self,
        after: Option<&PartitionKey>,
        limit: usize,
    ) -> Result<Vec<PartitionKey>>;

    /// Whether the coordinator can be reached.
    async fn ping(&self) -> Result<()>;

    /// Record that `node` is running now and reads up to `versions`.
    /// Returns the file format the cluster writes once the node counts: a
    /// format no newer than the node reads never rises past it while it
    /// stays registered, and a newer one was raised before it counted.
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<u16>;

    /// Forget `node`, which has stopped.
    async fn depart(&self, node: &NodeId) -> Result<()>;

    /// The oldest file format and protocol among the nodes registered, each
    /// the least of any node's; `None` if none is.
    async fn oldest(&self) -> Result<Option<Versions>>;

    /// The file format the cluster writes.
    async fn format(&self) -> Result<u16>;

    /// Raise the file format the cluster writes to `format`, if every node
    /// registered reads it, counting any whose [`Self::announce`] is under
    /// way; it never falls. Returns the format after.
    async fn raise_format(&self, format: u16) -> Result<u16>;

    /// Keep a key that reaches `tenant`, by the hash of its secret. `None`
    /// if there is no such tenant.
    async fn issue_key(
        &self,
        tenant: &TenantId,
        id: Uuid,
        hash: &KeyHash,
    ) -> Result<Option<TenantKey>>;

    /// The keys that reach `tenant`, oldest first.
    async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>>;

    /// Forget one of `tenant`'s keys. Returns whether it was one.
    async fn revoke_key(&self, tenant: &TenantId, id: Uuid) -> Result<bool>;

    /// The tenant the key with this hash reaches, if any.
    async fn key_tenant(&self, hash: &KeyHash) -> Result<Option<TenantId>>;

    /// The state of the partition as it is now, since it was last
    /// registered, that `point` names, if a restore reaching back `window`
    /// may go to it: one committed within the window, or the last committed
    /// before it, which is how the partition stood when the window began.
    /// A moment must itself lie in the window and not after now.
    async fn restore_point(
        &self,
        key: &PartitionKey,
        point: Point,
        window: Duration,
    ) -> Result<Option<RestorePoint>>;

    /// The earliest and the latest state a restore reaching back `window`
    /// may go to, if any.
    async fn restore_points(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<Option<(RestorePoint, RestorePoint)>>;

    /// Forget the commits a restore reaching back `window` can no longer go
    /// to, and the restores recorded over [`RESTORES_KEPT_FOR`] ago.
    /// Returns how many commits.
    async fn forget_history(&self, key: &PartitionKey, window: Duration) -> Result<u64>;

    /// The restore of the partition, since it was last registered, that
    /// `request` asked for, if it was recorded within [`RESTORES_KEPT_FOR`].
    async fn restored(&self, key: &PartitionKey, request: RequestId) -> Result<Option<Restored>>;

    /// Forget every restore recorded over [`RESTORES_KEPT_FOR`] ago, of any
    /// partition, kept up or not. Returns how many.
    async fn forget_restores(&self) -> Result<u64>;

    /// Take the right to back `tenant` up, or restore it from a backup, as
    /// `claim`, an id the caller picks, for `hold` from now: if nobody
    /// holds it, and with `due` only if its last backup is at least that
    /// old. A claim already held is held for `hold` from now, whether or
    /// not it is due, so that a caller who never heard whether it was
    /// taken can ask again. Returns whether `claim` is held.
    async fn claim_backup(
        &self,
        tenant: &TenantId,
        claim: Uuid,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<bool>;

    /// Hold `claim` for `hold` from now, if it is still held. One let go,
    /// or taken over once it ran out, is not taken again. Returns whether
    /// `claim` is held.
    async fn renew_backup(&self, tenant: &TenantId, claim: Uuid, hold: Duration) -> Result<bool>;

    /// Let go of `claim`, noting that the tenant was backed up now if
    /// `made`. A claim that ran out and was taken by another is left be.
    async fn release_backup(&self, tenant: &TenantId, claim: Uuid, made: bool) -> Result<()>;

    /// The tenants removed at least `due` ago whose files are still to be
    /// purged, and those purged within `lately`.
    async fn purges(&self, due: Duration, lately: Duration) -> Result<Purges>;

    /// Take the right to purge `tenant` for `hold`, if its purge is still
    /// to be done, it was removed at least `due` ago, and nobody holds the
    /// right. Returns the claim, and which tenant it was, if it was taken.
    async fn claim_purge(
        &self,
        tenant: &TenantId,
        due: Duration,
        hold: Duration,
    ) -> Result<Option<PurgeClaim>>;

    /// Hold `claim` to purge `tenant` for `hold` from now, if it is still
    /// held, the purge is still to be done, and the tenant has not been
    /// made again. Returns whether it is: once it is not, nothing more of
    /// the tenant may be deleted under it.
    async fn renew_purge(&self, tenant: &TenantId, claim: Uuid, hold: Duration) -> Result<bool>;

    /// Let go of `claim` to purge `tenant`, the purge not done: it may be
    /// claimed again at once. A claim that ran out and was taken by
    /// another, or a purge noted done, is left be.
    async fn release_purge(&self, tenant: &TenantId, claim: Uuid) -> Result<()>;

    /// Note that `tenant`'s files are purged under `claim`, and so looked
    /// over now, so that it may be made again, and forget purges done over
    /// `forget_after` ago. A claim that ran out, or a purge done already,
    /// is left be.
    async fn purged(&self, tenant: &TenantId, claim: Uuid, forget_after: Duration) -> Result<()>;

    /// Take up again the purge of `tenant`, done already, if the tenant has
    /// not been made again since: it is to be purged once more, at once,
    /// and until that is done cannot be made again, so that what was found
    /// left of it is deleted with nothing new. Returns whether it was taken
    /// up.
    async fn reopen_purge(&self, tenant: &TenantId) -> Result<bool>;

    /// Take up to `limit` of the tenants purged within `lately` that no
    /// node has looked over within `every`, to look for anything written
    /// for them since: each is noted as looked over now, so that one node
    /// at a time looks at a tenant, now and then.
    async fn look_overs(
        &self,
        lately: Duration,
        every: Duration,
        limit: usize,
    ) -> Result<Vec<Removed>>;

    /// How far back `tenant` may be restored, if it is not the default.
    async fn restore_window(&self, tenant: &TenantId) -> Result<Option<Duration>>;

    /// Set how far back `tenant` may be restored, or with `None` go back to
    /// the default; at most [`MAX_RESTORE_WINDOW`]. Returns whether there is
    /// such a tenant.
    async fn set_restore_window(&self, tenant: &TenantId, window: Option<Duration>)
        -> Result<bool>;
}

/// How long a restore is recorded, and the request that asked for it
/// answered as it was: far longer than one try of a restore is waited for,
/// an hour through the command line or when sent on to another node, so
/// that a try sent again after one given up finds the first. A request made
/// again later than this is a restore of its own and goes back again; its
/// record then replaces the old one.
pub const RESTORES_KEPT_FOR: Duration = Duration::from_secs(24 * 60 * 60);

/// The longest a backup may be claimed for at once.
pub const MAX_BACKUP_HOLD: Duration = Duration::from_secs(24 * 60 * 60);

fn checked_hold(hold: Duration) -> Result<Duration> {
    (hold <= MAX_BACKUP_HOLD).then_some(hold).ok_or_else(|| {
        CoordinatorError::InvalidArgument(format!(
            "a backup claim of {hold:?} exceeds {MAX_BACKUP_HOLD:?}"
        ))
    })
}

/// The longest a restore window may be.
pub const MAX_RESTORE_WINDOW: Duration = Duration::from_secs(10 * 366 * 24 * 60 * 60);

/// `window`, unless it is longer than any restore window may be.
fn checked_window(window: Duration) -> Result<Duration> {
    (window <= MAX_RESTORE_WINDOW)
        .then_some(window)
        .ok_or_else(|| {
            CoordinatorError::InvalidArgument(format!(
                "a restore window of {window:?} exceeds {MAX_RESTORE_WINDOW:?}"
            ))
        })
}

/// The moment a window reaching back `window` from `now` begins.
fn window_start(now: DateTime<Utc>, window: Duration) -> Result<DateTime<Utc>> {
    chrono::Duration::from_std(window)
        .ok()
        .and_then(|window| now.checked_sub_signed(window))
        .ok_or_else(|| {
            CoordinatorError::InvalidArgument(format!("a window of {window:?} is too long"))
        })
}

/// A commit as the history keeps it: a state the partition can be put
/// back to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestorePoint {
    pub position: LogPosition,
    pub head: ManifestRef,
    pub at: DateTime<Utc>,
}

/// Which state of a partition to go back to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Point {
    /// The one committed last at or before a moment in the window.
    At(DateTime<Utc>),
    /// The one committed at a position.
    Position(LogPosition),
}

/// Which tenant a tenant is: what tells it from any other of its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    pub lineage: Lineage,
    /// Whether any partition of it has ever committed anything.
    pub committed: bool,
}

/// What a tenant becomes with the commit that restores it from a backup
/// it adopts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Adoption {
    /// The tenant the backup is of.
    pub lineage: Lineage,
    /// The position of the commit that restores it.
    pub at: LogPosition,
    /// The keys it is given: those the backup kept, if they were asked
    /// for.
    pub keys: Vec<TenantKey>,
    /// How far back it may be restored, if not the default.
    pub restore_window: Option<Duration>,
}

/// A tenant that was removed, and which it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Removed {
    pub tenant: TenantId,
    pub lineage: Lineage,
}

/// The right to purge a removed tenant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgeClaim {
    pub id: Uuid,
    pub removed: Removed,
}

/// Tenants removed whose files are, or lately were, to be purged.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Purges {
    pub due: Vec<TenantId>,
    pub lately: Vec<TenantId>,
}

/// The SHA-256 of a key's secret: all the coordinator keeps of it.
pub type KeyHash = [u8; 32];

/// A key that reaches one tenant, as listed: when it was issued, the id it
/// is revoked by, and the hash of its secret, by which a backup keeps it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TenantKey {
    pub id: Uuid,
    pub created: DateTime<Utc>,
    pub hash: KeyHash,
}

/// The newest file format and node-to-node protocol a node can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Versions {
    pub format: u16,
    pub protocol: u16,
}

impl Versions {
    /// Each version the lesser of the two.
    pub fn least(self, other: Versions) -> Versions {
        Versions {
            format: self.format.min(other.format),
            protocol: self.protocol.min(other.protocol),
        }
    }
}
