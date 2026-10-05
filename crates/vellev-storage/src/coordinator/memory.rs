use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use vellev_core::NodeId;

use super::{
    checked_ttl, Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease,
    LogPosition, ManifestRef, PartitionKey, Result,
};

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A clock that moves only when told to, for tests and simulation.
#[derive(Debug)]
pub struct ManualClock(Mutex<DateTime<Utc>>);

impl ManualClock {
    pub fn starting_at(now: DateTime<Utc>) -> Self {
        Self(Mutex::new(now))
    }

    pub fn advance(&self, by: Duration) {
        let mut now = self.0.lock().expect("clock lock");
        *now += by;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> DateTime<Utc> {
        *self.0.lock().expect("clock lock")
    }
}

#[derive(Debug, Clone)]
struct Row {
    generation: Generation,
    lease: Option<(NodeId, DateTime<Utc>)>,
    log_position: LogPosition,
    manifest: Option<ManifestRef>,
}

impl Row {
    fn committed(&self) -> Committed {
        Committed {
            generation: self.generation,
            log_position: self.log_position,
            manifest: self.manifest.clone(),
        }
    }
}

/// The coordinator's rules without a database: the same fencing, driven by an
/// injectable clock.
pub struct MemoryCoordinator {
    clock: Arc<dyn Clock>,
    generations: AtomicU64,
    rows: Mutex<HashMap<PartitionKey, Row>>,
}

impl std::fmt::Debug for MemoryCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryCoordinator")
            .field("generations", &self.generations)
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

impl Default for MemoryCoordinator {
    fn default() -> Self {
        Self::new(Arc::new(SystemClock))
    }
}

impl MemoryCoordinator {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            generations: AtomicU64::new(0),
            rows: Mutex::new(HashMap::new()),
        }
    }

    fn next_generation(&self) -> Generation {
        Generation(self.generations.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn with_row<T>(
        &self,
        key: &PartitionKey,
        f: impl FnOnce(&mut Row, DateTime<Utc>) -> Result<T>,
    ) -> Result<T> {
        let now = self.clock.now();
        let mut rows = self.rows.lock().expect("coordinator lock");
        let row = rows
            .get_mut(key)
            .ok_or_else(|| CoordinatorError::UnknownPartition(key.clone()))?;
        f(row, now)
    }

    fn with_current<T>(
        &self,
        lease: &Lease,
        f: impl FnOnce(&mut Row, DateTime<Utc>) -> Result<T>,
    ) -> Result<T> {
        self.with_row(&lease.key, |row, now| {
            if row.generation == lease.generation {
                f(row, now)
            } else {
                Err(lease.fenced_by(row.generation))
            }
        })
    }
}

fn expiry(now: DateTime<Utc>, ttl: Duration) -> Result<DateTime<Utc>> {
    chrono::Duration::from_std(checked_ttl(ttl)?)
        .ok()
        .and_then(|ttl| now.checked_add_signed(ttl))
        .ok_or_else(|| {
            CoordinatorError::InvalidArgument(format!("ttl {ttl:?} overflows the clock"))
        })
}

#[async_trait]
impl Coordinator for MemoryCoordinator {
    async fn register(&self, key: &PartitionKey) -> Result<()> {
        self.rows
            .lock()
            .expect("coordinator lock")
            .entry(key.clone())
            .or_insert(Row {
                generation: Generation(0),
                lease: None,
                log_position: LogPosition(0),
                manifest: None,
            });
        Ok(())
    }

    async fn remove(&self, key: &PartitionKey) -> Result<bool> {
        Ok(self
            .rows
            .lock()
            .expect("coordinator lock")
            .remove(key)
            .is_some())
    }

    async fn acquire(
        &self,
        key: &PartitionKey,
        holder: &NodeId,
        ttl: Duration,
    ) -> Result<Acquired> {
        let requested_at = Instant::now();
        self.with_row(key, |row, now| {
            match &row.lease {
                Some((current, expires_at)) if *expires_at > now && current != holder => {
                    return Ok(Acquired::Held {
                        holder: current.clone(),
                        expires_at: *expires_at,
                    })
                }
                _ => {}
            }
            let expires_at = expiry(now, ttl)?;
            row.generation = self.next_generation();
            row.lease = Some((holder.clone(), expires_at));
            Ok(Acquired::Lease {
                lease: Lease {
                    key: key.clone(),
                    holder: holder.clone(),
                    generation: row.generation,
                    ttl,
                    expires_at,
                    requested_at,
                },
                committed: row.committed(),
            })
        })
    }

    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease> {
        let requested_at = Instant::now();
        self.with_current(lease, |row, now| {
            let expires_at = expiry(now, ttl)?;
            row.lease = Some((lease.holder.clone(), expires_at));
            Ok(Lease {
                ttl,
                expires_at,
                requested_at,
                ..lease.clone()
            })
        })
    }

    async fn release(&self, lease: &Lease) -> Result<()> {
        let mut rows = self.rows.lock().expect("coordinator lock");
        if let Some(row) = rows
            .get_mut(&lease.key)
            .filter(|row| row.generation == lease.generation)
        {
            row.generation = self.next_generation();
            row.lease = None;
        }
        Ok(())
    }

    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed> {
        self.with_current(lease, |row, _| {
            let advances = commit.log_position > row.log_position;
            let repeats =
                commit.log_position == row.log_position && commit.manifest == row.manifest;
            if !(advances || repeats) {
                return Err(CoordinatorError::Regressed {
                    key: lease.key.clone(),
                    committed: row.log_position,
                    attempted: commit.log_position,
                });
            }
            row.log_position = commit.log_position;
            row.manifest = commit.manifest;
            Ok(row.committed())
        })
    }

    async fn committed(&self, key: &PartitionKey) -> Result<Option<Committed>> {
        Ok(self
            .rows
            .lock()
            .expect("coordinator lock")
            .get(key)
            .map(Row::committed))
    }

    async fn ping(&self) -> Result<()> {
        Ok(())
    }
}
