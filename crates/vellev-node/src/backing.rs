//! Tenants backed up to a second store, daily and when asked, and put back
//! from it: into their own cluster, or into one that has lost them.
//!
//! One node at a time backs a tenant up or restores it from a backup, by a
//! claim in the coordinator, so that pruning never deletes a file another
//! backup is copying. The daily backups need no node to hold the tenant,
//! so tenants nobody has used for a while are backed up as well.

use std::time::Duration;

use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::backups::{Backup, Backups};
use vellev_storage::coordinator::{LogPosition, PartitionKey};
use vellev_storage::StorageError;

use crate::{key, Inner, Node, NodeError, Result};

/// How long a claim to back a tenant up or restore it lasts: longer than
/// any backup takes.
const HOLD: Duration = Duration::from_secs(6 * 60 * 60);
/// How often a node looks for tenants due a backup.
const LOOKED_FOR_EVERY: Duration = Duration::from_secs(10 * 60);
/// How many tenants are looked through at a time.
const PAGE: usize = 256;

impl Node {
    /// Back `tenant` up now, whenever it was last backed up.
    pub async fn back_up(&self, tenant: &TenantId) -> Result<Backup> {
        let backups = self.0.backups()?;
        if self.0.coordinator.committed(&key(tenant)).await?.is_none() {
            return Err(NodeError::UnknownTenant(tenant.clone()));
        }
        let claim = self.0.claim(tenant, None).await?;
        self.0.backed_up(&backups, tenant, claim).await
    }

    /// The backups of `tenant` there are, oldest first.
    pub async fn backups(&self, tenant: &TenantId) -> Result<Vec<Backup>> {
        Ok(self.0.backups()?.list(tenant).await?)
    }

    /// Put `tenant` back as backup `id` took it, as a new commit. A tenant
    /// this cluster does not have is made, with the keys and the restore
    /// window it had; one it has keeps its own, so that a key revoked since
    /// stays revoked. Returns the backup and the position of the commit
    /// that restored it.
    pub async fn restore_backup(
        &self,
        tenant: &TenantId,
        id: Uuid,
    ) -> Result<(Backup, LogPosition)> {
        let backups = self.0.backups()?;
        let backup = backups
            .find(tenant, id)
            .await?
            .ok_or_else(|| NodeError::UnknownBackup(tenant.clone(), id))?;
        let coordinator = &self.0.coordinator;
        if coordinator.committed(&key(tenant)).await?.is_none() {
            coordinator.register(&key(tenant)).await?;
            coordinator
                .set_restore_window(tenant, backup.restore_window)
                .await?;
            for kept in &backup.keys {
                coordinator.issue_key(tenant, kept.id, &kept.hash).await?;
            }
        }
        let claim = self.0.claim(tenant, None).await?;
        let restored = self.restored_from(&backups, tenant, &backup).await;
        coordinator.release_backup(tenant, claim, false).await?;
        restored.map(|restored| (backup, restored))
    }

    async fn restored_from(
        &self,
        backups: &Backups,
        tenant: &TenantId,
        backup: &Backup,
    ) -> Result<LogPosition> {
        let taken = backup
            .partitions
            .iter()
            .find(|taken| taken.key(tenant) == key(tenant))
            .ok_or_else(|| NodeError::UnknownBackup(tenant.clone(), backup.id))?;
        let copied = backups.fetch_back(&self.0.blobs, backup).await?;
        let opened = self.tenant(tenant).await?;
        let restoring = {
            let (opened, head) = (opened.clone(), taken.head.clone());
            let position = taken.position;
            tokio::spawn(async move {
                let head = head.map(vellev_storage::coordinator::ManifestRef);
                opened
                    .writer
                    .restore(head.as_ref(), position)
                    .await
                    .map_err(|error| opened.closed(error))
            })
        };
        let restored = restoring
            .await
            .map_err(|error| StorageError::Background(error.to_string()))??;
        tracing::info!(%tenant, backup = %backup.id, copied, ?restored, "restored from a backup");
        Ok(restored)
    }

    /// Back up each tenant whose last backup is as old as the node backs
    /// tenants up, that no node is backing up already.
    pub async fn back_up_due(&self) -> Result<()> {
        self.0.back_up_due().await
    }

    /// Back up every tenant due a backup, now and then, for as long as the
    /// node runs. Does nothing without a backup store.
    pub fn backing_up(&self) {
        if self.0.settings.backups.is_none() {
            return;
        }
        let inner = std::sync::Arc::downgrade(&self.0);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(LOOKED_FOR_EVERY).await;
                let Some(inner) = inner.upgrade() else { return };
                if inner.opened().stopped {
                    return;
                }
                if let Err(error) = inner.back_up_due().await {
                    tracing::warn!(%error, "could not look for tenants to back up");
                }
            }
        });
    }
}

impl Inner {
    fn backups(&self) -> Result<Backups> {
        self.settings
            .backups
            .clone()
            .map(Backups::new)
            .ok_or(NodeError::NoBackups)
    }

    async fn back_up_due(&self) -> Result<()> {
        let backups = self.backups()?;
        let mut after: Option<PartitionKey> = None;
        loop {
            let page = self.coordinator.partitions(after.as_ref(), PAGE).await?;
            for key in &page {
                let due = Some(self.settings.backup_every);
                let Some(claim) = self
                    .coordinator
                    .claim_backup(&key.tenant, due, HOLD)
                    .await?
                else {
                    continue;
                };
                if let Err(error) = self.backed_up(&backups, &key.tenant, claim).await {
                    tracing::warn!(tenant = %key.tenant, %error, "backup failed");
                }
            }
            match page.last() {
                Some(last) if page.len() == PAGE => after = Some(last.clone()),
                _ => return Ok(()),
            }
        }
    }

    async fn claim(&self, tenant: &TenantId, due: Option<Duration>) -> Result<Uuid> {
        self.coordinator
            .claim_backup(tenant, due, HOLD)
            .await?
            .ok_or_else(|| NodeError::BackingUp(tenant.clone()))
    }

    /// Back `tenant` up under `claim`, prune its backups, and let the claim
    /// go. A tenant removed while it was copied has its backups purged,
    /// so that nothing of it outlives it.
    async fn backed_up(&self, backups: &Backups, tenant: &TenantId, claim: Uuid) -> Result<Backup> {
        let made = async {
            let partitions = [key(tenant)];
            let backup = backups
                .take(&self.blobs, self.coordinator.as_ref(), tenant, &partitions)
                .await?;
            let now = self.coordinator.committed(&key(tenant)).await?;
            let same = now.is_some_and(|now| {
                backup
                    .partitions
                    .iter()
                    .all(|taken| taken.incarnation == now.incarnation)
            });
            if !same {
                backups.purge(tenant).await?;
                return Err(NodeError::UnknownTenant(tenant.clone()));
            }
            let pruned = backups.prune(tenant, self.settings.backups_kept).await?;
            tracing::info!(%tenant, backup = %backup.id, pruned, "backed up");
            Ok(backup)
        }
        .await;
        self.coordinator
            .release_backup(tenant, claim, made.is_ok())
            .await?;
        made
    }
}
