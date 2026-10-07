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

mod memory;
mod postgres;

use std::fmt;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use vellev_core::{NodeId, Partition, TenantId};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    pub log_position: LogPosition,
    pub manifest: Option<ManifestRef>,
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
        "lease on {key} at generation {held:?} is fenced; the current generation is {current:?}"
    )]
    Fenced {
        key: PartitionKey,
        held: Generation,
        current: Generation,
    },
    #[error("commit at {attempted:?} does not advance {key} past {committed:?}")]
    Regressed {
        key: PartitionKey,
        committed: LogPosition,
        attempted: LogPosition,
    },
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

fn checked_ttl(ttl: Duration) -> Result<Duration> {
    (ttl <= MAX_TTL).then_some(ttl).ok_or_else(|| {
        CoordinatorError::InvalidArgument(format!("ttl {ttl:?} exceeds {MAX_TTL:?}"))
    })
}

#[async_trait]
pub trait Coordinator: fmt::Debug + Send + Sync {
    /// Create the partition's row at a fresh generation. Registering an
    /// existing partition changes nothing.
    async fn register(&self, key: &PartitionKey) -> Result<()>;

    /// Delete the partition's row. Returns whether it existed. With the
    /// tenant's last partition go its keys, so that none reaches a tenant
    /// made again under the same id.
    async fn remove(&self, key: &PartitionKey) -> Result<bool>;

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
    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<()>;

    /// Forget `node`, which has stopped.
    async fn depart(&self, node: &NodeId) -> Result<()>;

    /// The oldest file format and protocol among the nodes registered, each
    /// the least of any node's; `None` if none is.
    async fn oldest(&self) -> Result<Option<Versions>>;

    /// The file format the cluster writes.
    async fn format(&self) -> Result<u16>;

    /// Raise the file format the cluster writes to `format`, if every node
    /// registered reads it; it never falls. Returns the format after.
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
    /// to. Returns how many.
    async fn forget_history(&self, key: &PartitionKey, window: Duration) -> Result<u64>;

    /// How far back `tenant` may be restored, if it is not the default.
    async fn restore_window(&self, tenant: &TenantId) -> Result<Option<Duration>>;

    /// Set how far back `tenant` may be restored, or with `None` go back to
    /// the default; at most [`MAX_RESTORE_WINDOW`]. Returns whether there is
    /// such a tenant.
    async fn set_restore_window(&self, tenant: &TenantId, window: Option<Duration>)
        -> Result<bool>;
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

/// The SHA-256 of a key's secret: all the coordinator keeps of it.
pub type KeyHash = [u8; 32];

/// A key that reaches one tenant, as listed: when it was issued, and the
/// id it is revoked by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantKey {
    pub id: Uuid,
    pub created: DateTime<Utc>,
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
