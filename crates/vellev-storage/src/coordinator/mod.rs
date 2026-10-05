//! Who may write to a partition, and what is committed.
//!
//! Correctness rests on the generation, not on clocks. Taking or releasing a
//! lease moves the partition to a fresh generation, and renew and commit
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
use vellev_core::{NodeId, Partition, TenantId};

pub use memory::{Clock, ManualClock, MemoryCoordinator, SystemClock};
pub use postgres::PostgresCoordinator;

/// Longer than any sensible lease, and short enough that expiry arithmetic
/// cannot overflow in either implementation.
pub const MAX_TTL: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LogPosition(pub u64);

/// The object key of a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManifestRef(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Committed {
    /// The partition's generation when this was read. It moves with every
    /// lease taken or released, not only with commits.
    pub generation: Generation,
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

    /// Only a backend failure leaves the outcome unknown; every other error
    /// is a definite answer.
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

fn bigint(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| CoordinatorError::InvalidArgument(format!("{value} exceeds the bigint range")))
}

#[async_trait]
pub trait Coordinator: Send + Sync {
    /// Create the partition's row. Registering an existing partition changes nothing.
    async fn register(&self, key: &PartitionKey) -> Result<()>;

    /// Delete the partition's row. Returns whether it existed.
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

    /// Whether the coordinator can be reached.
    async fn ping(&self) -> Result<()>;
}
