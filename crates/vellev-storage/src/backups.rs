//! Copies of tenants in a second store, from which a tenant is put back as
//! it was when copied: in its own cluster, or in another that has lost it.
//!
//! Every file of a tenant is copied once, under `_backups/{tenant}/files/`,
//! and each backup lists the ones it needs. Files never change, so a backup
//! costs what changed since the one before. A backup is a record under
//! `_backups/{tenant}/backups/`, written as pending before any file is
//! copied, so that pruning, which keeps every file any record lists, leaves
//! them be, and then as complete once every file it lists is there. All of
//! a tenant lies under its own prefix, in one bucket however many tenants
//! there are; no tenant's own files could lie under `_backups/`, so that
//! even a store given as both deletes none of them.
//!
//! A record is trusted only so far: one naming another tenant, or a file
//! outside the partition it names, is not restored from.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use vellev_core::{Partition as PartitionNumber, TenantId};

use crate::blobs::{is_plain, Blobs, Layout};
use crate::coordinator::{Coordinator, Incarnation, LogPosition, PartitionKey, TenantKey};
use crate::partition::Partition;
use crate::{Result, StorageError};

/// How many times a backup starts again when a file it needs is deleted
/// while it copies, as cleaning up does to what a newer head no longer
/// needs; the files of the head it then reads are never deleted.
const ATTEMPTS: usize = 3;
/// How many files are copied or read at once.
const AT_ONCE: usize = 8;
/// How long a pending record is kept: longer than any backup takes.
const PENDING_KEPT: Duration = Duration::from_secs(24 * 60 * 60);
const COMPLETE: &str = ".backup.json";
const PENDING: &str = ".pending.json";

/// A tenant as it was when copied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Backup {
    pub id: Uuid,
    pub tenant: TenantId,
    pub taken_at: DateTime<Utc>,
    pub partitions: Vec<Taken>,
    /// The tenant's keys, by the hashes of their secrets.
    pub keys: Vec<TenantKey>,
    /// How far back the tenant may be restored, if not the default.
    pub restore_window: Option<Duration>,
}

impl Backup {
    /// The files every partition it took is loaded from.
    pub fn files(&self) -> impl Iterator<Item = &String> {
        self.partitions.iter().flat_map(|taken| &taken.files)
    }

    /// Where the tenant's first partition was when it was taken.
    pub fn position(&self) -> LogPosition {
        self.partitions
            .first()
            .map_or(LogPosition::default(), |taken| taken.position)
    }
}

/// One partition as a backup took it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Taken {
    pub partition: PartitionNumber,
    pub incarnation: Incarnation,
    pub position: LogPosition,
    pub head: Option<String>,
    /// The files it is loaded from, by their keys in the tenant's store.
    pub files: Vec<String>,
}

impl Taken {
    pub fn key(&self, tenant: &TenantId) -> PartitionKey {
        PartitionKey {
            tenant: tenant.clone(),
            partition: self.partition,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Backups {
    store: Arc<dyn Blobs>,
}

impl Backups {
    pub fn new(store: Arc<dyn Blobs>) -> Self {
        Self { store }
    }

    /// Copy `partitions` of `tenant`, as committed now, out of `primary`.
    /// Nothing else may back the tenant up or prune its backups meanwhile.
    pub async fn take(
        &self,
        primary: &Arc<dyn Blobs>,
        coordinator: &dyn Coordinator,
        tenant: &TenantId,
        partitions: &[PartitionKey],
    ) -> Result<Backup> {
        let mut attempt = 1;
        loop {
            match self.attempt(primary, coordinator, tenant, partitions).await {
                Err(StorageError::Missing { key }) if attempt < ATTEMPTS => {
                    tracing::info!(%tenant, key, "a file went while it was copied; starting again");
                    attempt += 1;
                }
                taken => return taken,
            }
        }
    }

    async fn attempt(
        &self,
        primary: &Arc<dyn Blobs>,
        coordinator: &dyn Coordinator,
        tenant: &TenantId,
        partitions: &[PartitionKey],
    ) -> Result<Backup> {
        let backup = taken(primary, coordinator, tenant, partitions).await?;
        let pending = record(&backup, PENDING);
        self.store.put(&pending, encoded(&backup)?).await?;
        let completed = async {
            self.copied(primary, &backup).await?;
            self.store
                .put(&record(&backup, COMPLETE), encoded(&backup)?)
                .await
        }
        .await;
        if let Err(error) = self.store.delete(&pending).await {
            tracing::warn!(%tenant, key = pending, %error, "a pending backup record was left");
        }
        completed.map(|()| backup)
    }

    /// Copy the files of `backup` not yet in the store, and make sure that
    /// every one it lists is there.
    async fn copied(&self, primary: &Arc<dyn Blobs>, backup: &Backup) -> Result<()> {
        let prefix = &files(&backup.tenant);
        let held = self.keys(prefix).await?;
        let wanted: Vec<(String, String)> = backup
            .files()
            .map(|key| (key.to_string(), format!("{prefix}{key}")))
            .filter(|(_, copied)| !held.contains(copied))
            .collect();
        let copies = wanted
            .into_iter()
            .map(|(key, copied)| copy(primary.as_ref(), key, self.store.as_ref(), copied));
        stream::iter(copies)
            .buffer_unordered(AT_ONCE)
            .try_collect::<()>()
            .await?;
        let held = self.keys(prefix).await?;
        match backup
            .files()
            .find(|key| !held.contains(&format!("{prefix}{key}")))
        {
            Some(key) => Err(StorageError::Missing {
                key: format!("{prefix}{key}"),
            }),
            None => Ok(()),
        }
    }

    /// The complete backups of `tenant`, oldest first.
    pub async fn list(&self, tenant: &TenantId) -> Result<Vec<Backup>> {
        Ok(self
            .records(tenant, COMPLETE)
            .await?
            .into_iter()
            .map(|(_, backup)| backup)
            .collect())
    }

    /// The complete backup of `tenant` with this id, if there is one.
    pub async fn find(&self, tenant: &TenantId, id: Uuid) -> Result<Option<Backup>> {
        Ok(self
            .list(tenant)
            .await?
            .into_iter()
            .find(|backup| backup.id == id))
    }

    /// The records of `tenant`'s backups whose keys end in `ending`, by
    /// key, oldest first. One that went meanwhile, cannot be read, or names
    /// another tenant is passed over.
    async fn records(&self, tenant: &TenantId, ending: &str) -> Result<Vec<(String, Backup)>> {
        let listed = self.store.list(&records(tenant)).await?;
        let reads = listed
            .into_iter()
            .filter(|listed| listed.key.ends_with(ending))
            .map(|listed| read(self.store.as_ref(), listed.key, tenant));
        let read: Vec<Option<(String, Backup)>> = stream::iter(reads)
            .buffer_unordered(AT_ONCE)
            .try_collect()
            .await?;
        let mut records: Vec<(String, Backup)> = read.into_iter().flatten().collect();
        records.sort_by(|(key, backup), (other, than)| {
            (backup.taken_at, key).cmp(&(than.taken_at, other))
        });
        Ok(records)
    }

    /// Forget `tenant`'s complete backups taken over `keep` ago, save the
    /// newest, and pending ones left by a backup that never finished; then
    /// delete the files no record left lists. Returns how many backups were
    /// forgotten. Nothing may back the tenant up meanwhile.
    pub async fn prune(&self, tenant: &TenantId, keep: Duration) -> Result<usize> {
        let now = Utc::now();
        let before = |kept: Duration| {
            chrono::Duration::from_std(kept)
                .ok()
                .and_then(|kept| now.checked_sub_signed(kept))
        };
        let (complete, pending): (Vec<_>, Vec<_>) = self
            .records(tenant, ".json")
            .await?
            .into_iter()
            .partition(|(key, _)| key.ends_with(COMPLETE));
        let newest = complete.last().map(|(key, _)| key.clone());
        let old = |(key, backup): &(String, Backup), kept| {
            Some(key) != newest.as_ref() && before(kept).is_some_and(|at| backup.taken_at < at)
        };
        let (gone, kept): (Vec<_>, Vec<_>) = complete
            .into_iter()
            .map(|record| (old(&record, keep), record))
            .chain(
                pending
                    .into_iter()
                    .map(|record| (old(&record, PENDING_KEPT), record)),
            )
            .partition(|(gone, _)| *gone);
        for (_, (key, _)) in &gone {
            self.store.delete(key).await?;
        }
        let prefix = files(tenant);
        let listed: HashSet<String> = kept
            .iter()
            .flat_map(|(_, (_, backup))| backup.files())
            .map(|key| format!("{prefix}{key}"))
            .collect();
        for key in self.keys(&prefix).await?.difference(&listed) {
            self.store.delete(key).await?;
        }
        Ok(gone.len())
    }

    /// Copy the files of `backup` that `primary` lacks back into it, under
    /// the keys they had. Returns how many were copied.
    pub async fn fetch_back(&self, primary: &Arc<dyn Blobs>, backup: &Backup) -> Result<usize> {
        let prefix = &files(&backup.tenant);
        let mut held = HashSet::new();
        for taken in &backup.partitions {
            let layout = Layout(&taken.key(&backup.tenant)).prefix();
            if let Some(stray) = taken
                .files
                .iter()
                .find(|key| !(is_plain(key) && key.starts_with(&layout)))
            {
                return Err(StorageError::corrupt(
                    stray,
                    format!("backup {} lists it, outside {layout}", backup.id),
                ));
            }
            held.extend(
                primary
                    .list(&layout)
                    .await?
                    .into_iter()
                    .map(|listed| listed.key),
            );
        }
        let wanted: Vec<(String, String)> = backup
            .files()
            .filter(|key| !held.contains(*key))
            .map(|key| (format!("{prefix}{key}"), key.to_string()))
            .collect();
        let copied = wanted.len();
        let copies = wanted
            .into_iter()
            .map(|(kept, key)| copy(self.store.as_ref(), kept, primary.as_ref(), key));
        stream::iter(copies)
            .buffer_unordered(AT_ONCE)
            .try_collect::<()>()
            .await?;
        Ok(copied)
    }

    /// Forget `backup`; the files it alone listed go at the next prune.
    pub async fn forget(&self, backup: &Backup) -> Result<()> {
        self.store.delete(&record(backup, COMPLETE)).await
    }

    /// Delete everything kept of `tenant`. Returns how many files went.
    pub async fn purge(&self, tenant: &TenantId) -> Result<usize> {
        let listed = self.store.list(&kept(tenant)).await?;
        for file in &listed {
            self.store.delete(&file.key).await?;
        }
        Ok(listed.len())
    }

    async fn keys(&self, prefix: &str) -> Result<HashSet<String>> {
        Ok(self
            .store
            .list(prefix)
            .await?
            .into_iter()
            .map(|listed| listed.key)
            .collect())
    }
}

/// What a backup of `partitions` as committed now is to hold.
async fn taken(
    primary: &Arc<dyn Blobs>,
    coordinator: &dyn Coordinator,
    tenant: &TenantId,
    partitions: &[PartitionKey],
) -> Result<Backup> {
    let mut taken = Vec::with_capacity(partitions.len());
    for key in partitions {
        let committed = coordinator
            .committed(key)
            .await?
            .ok_or_else(|| crate::coordinator::CoordinatorError::UnknownPartition(key.clone()))?;
        taken.push(Taken {
            partition: key.partition,
            incarnation: committed.incarnation,
            position: committed.log_position,
            head: committed.manifest.as_ref().map(|head| head.0.clone()),
            files: Partition::files(key.clone(), primary.clone(), &committed).await?,
        });
    }
    Ok(Backup {
        id: Uuid::new_v4(),
        tenant: tenant.clone(),
        taken_at: Utc::now(),
        partitions: taken,
        keys: coordinator.keys(tenant).await?,
        restore_window: coordinator.restore_window(tenant).await?,
    })
}

async fn copy(from: &dyn Blobs, key: String, to: &dyn Blobs, copied: String) -> Result<()> {
    to.put(&copied, from.get(&key).await?).await
}

/// The record under `key`, if it is one of `tenant`'s.
async fn read(
    store: &dyn Blobs,
    key: String,
    tenant: &TenantId,
) -> Result<Option<(String, Backup)>> {
    let bytes = match store.get(&key).await {
        Ok(bytes) => bytes,
        Err(StorageError::Missing { .. }) => return Ok(None),
        Err(error) => return Err(error),
    };
    match serde_json::from_slice::<Backup>(&bytes) {
        Ok(backup) if &backup.tenant == tenant => Ok(Some((key, backup))),
        Ok(backup) => {
            tracing::warn!(key, named = %backup.tenant, "a backup record names another tenant");
            Ok(None)
        }
        Err(error) => {
            tracing::warn!(key, %error, "a backup record cannot be read");
            Ok(None)
        }
    }
}

/// Where everything of `tenant` is kept.
fn kept(tenant: &TenantId) -> String {
    format!("_backups/{tenant}/")
}

fn files(tenant: &TenantId) -> String {
    format!("{}files/", kept(tenant))
}

fn records(tenant: &TenantId) -> String {
    format!("{}backups/", kept(tenant))
}

fn record(backup: &Backup, suffix: &str) -> String {
    format!(
        "{}{:020}-{}{suffix}",
        records(&backup.tenant),
        backup.taken_at.timestamp_millis(),
        backup.id.simple()
    )
}

fn encoded(backup: &Backup) -> Result<Bytes> {
    serde_json::to_vec_pretty(backup)
        .map(Bytes::from)
        .map_err(|error| StorageError::Encode(error.to_string()))
}
