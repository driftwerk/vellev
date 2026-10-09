use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use vellev_core::{NodeId, RequestId, TenantId};

use super::{
    checked_hold, checked_ttl, checked_window, window_start, Acquired, Commit, Committed,
    Coordinator, CoordinatorError, Forgotten, Generation, Incarnation, KeyHash, Lease, LogPosition,
    ManifestRef, PartitionKey, Point, Purges, RestorePoint, Restored, Result, TenantKey, Versions,
    RESTORES_KEPT_FOR,
};

pub trait Clock: fmt::Debug + Send + Sync {
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
    incarnation: Incarnation,
    lease: Option<(NodeId, DateTime<Utc>)>,
    log_position: LogPosition,
    manifest: Option<ManifestRef>,
    /// Every commit since the partition was registered, oldest first.
    history: Vec<RestorePoint>,
    /// The restores recorded lately, with when each was.
    restores: Vec<(DateTime<Utc>, Restored)>,
    restore_window: Option<Duration>,
    backed_up_at: Option<DateTime<Utc>>,
    /// Who holds the right to back the tenant up, and until when.
    backup_claim: Option<(Uuid, DateTime<Utc>)>,
}

impl Row {
    fn committed(&self) -> Committed {
        Committed {
            generation: self.generation,
            incarnation: self.incarnation,
            log_position: self.log_position,
            manifest: self.manifest.clone(),
        }
    }

    /// The commits a restore reaching back from `start` may go to.
    fn reachable(&self, start: DateTime<Utc>) -> &[RestorePoint] {
        let within = self.history.partition_point(|commit| commit.at < start);
        &self.history[within.saturating_sub(1)..]
    }
}

/// A removed tenant's purge.
#[derive(Debug, Clone)]
struct Purge {
    requested: DateTime<Utc>,
    purged: Option<DateTime<Utc>>,
    claim: Option<(Uuid, DateTime<Utc>)>,
}

/// The coordinator's rules without a database: the same fencing, driven by an
/// injectable clock.
#[derive(Debug)]
pub struct MemoryCoordinator {
    clock: Arc<dyn Clock>,
    generations: AtomicU64,
    rows: Mutex<HashMap<PartitionKey, Row>>,
    nodes: Mutex<HashMap<NodeId, Versions>>,
    format: Mutex<u16>,
    keys: Mutex<Vec<(TenantId, KeyHash, TenantKey)>>,
    purges: Mutex<HashMap<TenantId, Purge>>,
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
            nodes: Mutex::new(HashMap::new()),
            format: Mutex::new(1),
            keys: Mutex::new(Vec::new()),
            purges: Mutex::default(),
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

fn unknown_is_none<T>(error: CoordinatorError) -> Result<Option<T>> {
    match error {
        CoordinatorError::UnknownPartition(_) => Ok(None),
        other => Err(other),
    }
}

fn expiry(now: DateTime<Utc>, ttl: Duration) -> Result<DateTime<Utc>> {
    chrono::Duration::from_std(ttl)
        .ok()
        .and_then(|ttl| now.checked_add_signed(ttl))
        .ok_or_else(|| {
            CoordinatorError::InvalidArgument(format!("ttl {ttl:?} overflows the clock"))
        })
}

#[async_trait]
impl Coordinator for MemoryCoordinator {
    async fn register(&self, key: &PartitionKey) -> Result<()> {
        let mut rows = self.rows.lock().expect("coordinator lock");
        let purging = self.purges.lock().expect("coordinator lock");
        if purging
            .get(&key.tenant)
            .is_some_and(|purge| purge.purged.is_none())
        {
            return Err(CoordinatorError::Purging(key.tenant.clone()));
        }
        if !rows.contains_key(key) {
            let generation = self.next_generation();
            let row = Row {
                generation,
                incarnation: Incarnation(generation.0),
                lease: None,
                log_position: LogPosition(0),
                manifest: None,
                history: Vec::new(),
                restores: Vec::new(),
                restore_window: None,
                backed_up_at: None,
                backup_claim: None,
            };
            rows.insert(key.clone(), row);
        }
        Ok(())
    }

    async fn remove(&self, key: &PartitionKey) -> Result<bool> {
        let mut rows = self.rows.lock().expect("coordinator lock");
        let existed = rows.remove(key).is_some();
        if !rows.keys().any(|other| other.tenant == key.tenant) {
            self.keys
                .lock()
                .expect("coordinator lock")
                .retain(|(tenant, ..)| tenant != &key.tenant);
            if existed {
                let purge = Purge {
                    requested: self.clock.now(),
                    purged: None,
                    claim: None,
                };
                self.purges
                    .lock()
                    .expect("coordinator lock")
                    .insert(key.tenant.clone(), purge);
            }
        }
        Ok(existed)
    }

    async fn acquire(
        &self,
        key: &PartitionKey,
        holder: &NodeId,
        ttl: Duration,
    ) -> Result<Acquired> {
        let ttl = checked_ttl(ttl)?;
        let requested_at = Instant::now();
        self.with_row(key, |row, now| {
            let live = row
                .lease
                .as_ref()
                .filter(|(current, expires_at)| *expires_at > now && current != holder);
            if let Some((current, expires_at)) = live {
                return Ok(Acquired::Held {
                    holder: current.clone(),
                    expires_at: *expires_at,
                });
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
        let ttl = checked_ttl(ttl)?;
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
        self.with_current(lease, |row, now| {
            let follows = commit.after.as_ref().is_none_or(|after| {
                after.position == row.log_position && after.head == row.manifest
            });
            let advances = follows && commit.log_position > row.log_position;
            let repeats =
                commit.log_position == row.log_position && commit.manifest == row.manifest;
            if !(advances || repeats) {
                return Err(CoordinatorError::Regressed {
                    key: lease.key.clone(),
                    committed: row.log_position,
                    attempted: commit.log_position,
                });
            }
            if let (true, Some(head)) = (advances, &commit.manifest) {
                row.history.push(RestorePoint {
                    position: commit.log_position,
                    head: head.clone(),
                    at: now,
                });
            }
            if let (true, Some(restore)) = (advances, commit.restore) {
                let restored = Restored {
                    restore,
                    incarnation: row.incarnation,
                    position: commit.log_position,
                };
                row.restores.push((now, restored));
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

    async fn partitions(
        &self,
        after: Option<&PartitionKey>,
        limit: usize,
    ) -> Result<Vec<PartitionKey>> {
        let rows = self.rows.lock().expect("coordinator lock");
        let mut keys: Vec<PartitionKey> = rows
            .keys()
            .filter(|key| after.is_none_or(|after| *key > after))
            .cloned()
            .collect();
        keys.sort_unstable();
        keys.truncate(limit);
        Ok(keys)
    }

    async fn ping(&self) -> Result<()> {
        Ok(())
    }

    async fn announce(&self, node: &NodeId, versions: Versions) -> Result<()> {
        self.nodes
            .lock()
            .expect("coordinator lock")
            .insert(node.clone(), versions);
        Ok(())
    }

    async fn depart(&self, node: &NodeId) -> Result<()> {
        self.nodes.lock().expect("coordinator lock").remove(node);
        Ok(())
    }

    async fn oldest(&self) -> Result<Option<Versions>> {
        Ok(self
            .nodes
            .lock()
            .expect("coordinator lock")
            .values()
            .copied()
            .reduce(Versions::least))
    }

    async fn format(&self) -> Result<u16> {
        Ok(*self.format.lock().expect("coordinator lock"))
    }

    async fn raise_format(&self, format: u16) -> Result<u16> {
        let nodes = self.nodes.lock().expect("coordinator lock");
        let mut current = self.format.lock().expect("coordinator lock");
        if *current < format && nodes.values().all(|node| node.format >= format) {
            *current = format;
        }
        Ok(*current)
    }

    async fn issue_key(
        &self,
        tenant: &TenantId,
        id: Uuid,
        hash: &KeyHash,
    ) -> Result<Option<TenantKey>> {
        let rows = self.rows.lock().expect("coordinator lock");
        if !rows.keys().any(|key| &key.tenant == tenant) {
            return Ok(None);
        }
        let issued = TenantKey {
            id,
            created: self.clock.now(),
            hash: *hash,
        };
        let mut keys = self.keys.lock().expect("coordinator lock");
        if keys
            .iter()
            .any(|(_, kept, key)| kept == hash || key.id == id)
        {
            return Err(CoordinatorError::InvalidArgument(
                "a key with this id or secret is already kept".into(),
            ));
        }
        keys.push((tenant.clone(), *hash, issued.clone()));
        Ok(Some(issued))
    }

    async fn keys(&self, tenant: &TenantId) -> Result<Vec<TenantKey>> {
        Ok(self
            .keys
            .lock()
            .expect("coordinator lock")
            .iter()
            .filter(|(of, ..)| of == tenant)
            .map(|(.., key)| key.clone())
            .collect())
    }

    async fn revoke_key(&self, tenant: &TenantId, id: Uuid) -> Result<bool> {
        let mut keys = self.keys.lock().expect("coordinator lock");
        let before = keys.len();
        keys.retain(|(of, _, key)| !(of == tenant && key.id == id));
        Ok(keys.len() < before)
    }

    async fn key_tenant(&self, hash: &KeyHash) -> Result<Option<TenantId>> {
        Ok(self
            .keys
            .lock()
            .expect("coordinator lock")
            .iter()
            .find(|(_, kept, _)| kept == hash)
            .map(|(tenant, ..)| tenant.clone()))
    }

    async fn restore_point(
        &self,
        key: &PartitionKey,
        point: Point,
        window: Duration,
    ) -> Result<Option<RestorePoint>> {
        self.with_row(key, |row, now| {
            let start = window_start(now, window)?;
            let reachable = row.reachable(start);
            Ok(match point {
                Point::At(at) if at > now || at < start => None,
                Point::At(at) => reachable.iter().rev().find(|commit| commit.at <= at),
                Point::Position(position) => {
                    reachable.iter().find(|commit| commit.position == position)
                }
            }
            .cloned())
        })
        .or_else(unknown_is_none)
    }

    async fn restore_points(
        &self,
        key: &PartitionKey,
        window: Duration,
    ) -> Result<Option<(RestorePoint, RestorePoint)>> {
        self.with_row(key, |row, now| {
            let reachable = row.reachable(window_start(now, window)?);
            Ok(reachable.first().cloned().zip(reachable.last().cloned()))
        })
        .or_else(unknown_is_none)
    }

    async fn forget_history(&self, key: &PartitionKey, window: Duration) -> Result<u64> {
        self.with_row(key, |row, now| {
            let start = window_start(now, window)?;
            let forgotten = row.history.len() - row.reachable(start).len();
            row.history.drain(..forgotten);
            let kept_since = window_start(now, RESTORES_KEPT_FOR)?;
            row.restores.retain(|(at, _)| *at >= kept_since);
            Ok(forgotten as u64)
        })
        .or_else(|error| unknown_is_none::<()>(error).map(|_| 0))
    }

    async fn restored(&self, key: &PartitionKey, request: RequestId) -> Result<Option<Restored>> {
        self.with_row(key, |row, now| {
            let kept_since = window_start(now, RESTORES_KEPT_FOR)?;
            Ok(row
                .restores
                .iter()
                .filter(|(at, _)| *at >= kept_since)
                .map(|(_, restored)| restored)
                .find(|restored| restored.restore.request == request)
                .cloned())
        })
        .or_else(unknown_is_none)
    }

    /// Nothing is kept of a partition removed here: its row, history and
    /// restores and all, go together.
    async fn forget_removed(
        &self,
        _key: &PartitionKey,
        _after: Option<Forgotten>,
        _limit: usize,
    ) -> Result<Option<Forgotten>> {
        Ok(None)
    }

    async fn forget_restores(&self) -> Result<u64> {
        let kept_since = window_start(self.clock.now(), RESTORES_KEPT_FOR)?;
        let mut rows = self.rows.lock().expect("coordinator lock");
        Ok(rows
            .values_mut()
            .map(|row| {
                let before = row.restores.len();
                row.restores.retain(|(at, _)| *at >= kept_since);
                (before - row.restores.len()) as u64
            })
            .sum())
    }

    async fn purges(&self, due: Duration, lately: Duration) -> Result<Purges> {
        let now = self.clock.now();
        let (due, lately) = (window_start(now, due)?, window_start(now, lately)?);
        let purges = self.purges.lock().expect("coordinator lock");
        let mut found = Purges::default();
        for (tenant, purge) in purges.iter() {
            match purge.purged {
                None if purge.requested <= due => found.due.push(tenant.clone()),
                Some(purged) if purged >= lately => found.lately.push(tenant.clone()),
                _ => {}
            }
        }
        found.due.sort();
        found.lately.sort();
        Ok(found)
    }

    async fn claim_purge(&self, tenant: &TenantId, hold: Duration) -> Result<Option<Uuid>> {
        let now = self.clock.now();
        let until = expiry(now, checked_hold(hold)?)?;
        let mut purges = self.purges.lock().expect("coordinator lock");
        let Some(purge) = purges.get_mut(tenant).filter(|purge| {
            purge.purged.is_none() && purge.claim.is_none_or(|(_, until)| until < now)
        }) else {
            return Ok(None);
        };
        let claim = Uuid::new_v4();
        purge.claim = Some((claim, until));
        Ok(Some(claim))
    }

    async fn purged(&self, tenant: &TenantId, claim: Uuid, forget_after: Duration) -> Result<()> {
        let now = self.clock.now();
        let forgotten = window_start(now, forget_after)?;
        let mut purges = self.purges.lock().expect("coordinator lock");
        let held = purges.get_mut(tenant).filter(|purge| {
            purge.purged.is_none() && purge.claim.is_some_and(|(held, _)| held == claim)
        });
        if let Some(purge) = held {
            purge.purged = Some(now);
            purge.claim = None;
        }
        purges.retain(|_, purge| purge.purged.is_none_or(|purged| purged >= forgotten));
        Ok(())
    }

    async fn claim_backup(
        &self,
        tenant: &TenantId,
        due: Option<Duration>,
        hold: Duration,
    ) -> Result<Option<Uuid>> {
        let now = self.clock.now();
        let until = expiry(now, checked_hold(hold)?)?;
        let since = due.map(|due| window_start(now, due)).transpose()?;
        let mut rows = self.rows.lock().expect("coordinator lock");
        let mut rows: Vec<&mut Row> = rows
            .iter_mut()
            .filter(|(key, _)| &key.tenant == tenant)
            .map(|(_, row)| row)
            .collect();
        let free = |row: &&mut Row| {
            row.backup_claim.is_none_or(|(_, until)| until < now)
                && since.is_none_or(|since| row.backed_up_at.is_none_or(|at| at <= since))
        };
        if rows.is_empty() || !rows.iter().all(free) {
            return Ok(None);
        }
        let claim = Uuid::new_v4();
        for row in &mut rows {
            row.backup_claim = Some((claim, until));
        }
        Ok(Some(claim))
    }

    async fn release_backup(&self, tenant: &TenantId, claim: Uuid, made: bool) -> Result<()> {
        let now = self.clock.now();
        let mut rows = self.rows.lock().expect("coordinator lock");
        let held = rows.iter_mut().filter(|(key, row)| {
            &key.tenant == tenant && row.backup_claim.is_some_and(|(held, _)| held == claim)
        });
        for (_, row) in held {
            row.backup_claim = None;
            if made {
                row.backed_up_at = Some(now);
            }
        }
        Ok(())
    }

    async fn restore_window(&self, tenant: &TenantId) -> Result<Option<Duration>> {
        Ok(self
            .rows
            .lock()
            .expect("coordinator lock")
            .iter()
            .find(|(key, _)| &key.tenant == tenant)
            .and_then(|(_, row)| row.restore_window))
    }

    async fn set_restore_window(
        &self,
        tenant: &TenantId,
        window: Option<Duration>,
    ) -> Result<bool> {
        let window = window.map(checked_window).transpose()?;
        let mut rows = self.rows.lock().expect("coordinator lock");
        let mut found = false;
        for (_, row) in rows.iter_mut().filter(|(key, _)| &key.tenant == tenant) {
            row.restore_window = window;
            found = true;
        }
        Ok(found)
    }
}
