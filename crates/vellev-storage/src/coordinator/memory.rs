use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use vellev_core::NodeId;

use super::{
    Acquired, Commit, Committed, Coordinator, CoordinatorError, Generation, Lease, LogPosition,
    ManifestRef, PartitionKey, Result,
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
    rows: Mutex<HashMap<PartitionKey, Row>>,
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
            rows: Mutex::new(HashMap::new()),
        }
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

    fn current<T>(
        &self,
        lease: &Lease,
        f: impl FnOnce(&mut Row, DateTime<Utc>) -> Result<T>,
    ) -> Result<T> {
        self.with_row(&lease.key, |row, now| {
            if row.generation != lease.generation {
                return Err(CoordinatorError::Fenced {
                    key: lease.key.clone(),
                    held: lease.generation,
                    current: row.generation,
                });
            }
            f(row, now)
        })
    }
}

fn expiry(now: DateTime<Utc>, ttl: Duration) -> DateTime<Utc> {
    now + chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::MAX)
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
        self.with_row(key, |row, now| {
            if let Some((holder, expires_at)) = &row.lease {
                if *expires_at > now {
                    return Ok(Acquired::Held {
                        holder: holder.clone(),
                        expires_at: *expires_at,
                    });
                }
            }
            let expires_at = expiry(now, ttl);
            row.generation = Generation(row.generation.0 + 1);
            row.lease = Some((holder.clone(), expires_at));
            Ok(Acquired::Lease {
                lease: Lease {
                    key: key.clone(),
                    holder: holder.clone(),
                    generation: row.generation,
                    ttl,
                    expires_at,
                },
                committed: row.committed(),
            })
        })
    }

    async fn renew(&self, lease: &Lease, ttl: Duration) -> Result<Lease> {
        self.current(lease, |row, now| {
            let expires_at = expiry(now, ttl);
            row.lease = Some((lease.holder.clone(), expires_at));
            Ok(Lease {
                ttl,
                expires_at,
                ..lease.clone()
            })
        })
    }

    async fn release(&self, lease: &Lease) -> Result<()> {
        let mut rows = self.rows.lock().expect("coordinator lock");
        if let Some(row) = rows.get_mut(&lease.key) {
            if row.generation == lease.generation {
                row.generation = Generation(row.generation.0 + 1);
                row.lease = None;
            }
        }
        Ok(())
    }

    async fn commit(&self, lease: &Lease, commit: Commit) -> Result<Committed> {
        self.current(lease, |row, _| {
            if commit.log_position <= row.log_position {
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
