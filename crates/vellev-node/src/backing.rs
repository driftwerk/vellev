//! Tenants backed up to a second store, daily and when asked, and put back
//! from it: into their own cluster, or into one that has lost them.
//!
//! One node at a time backs a tenant up or restores it from a backup, by a
//! claim in the coordinator, so that pruning never deletes a file another
//! backup is copying. Each runs to its end once begun, whoever stops
//! waiting, and lets its claim go. The daily backups need no node to hold
//! the tenant, so tenants nobody has used for a while are backed up too.

use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::backups::{Backup, Backups};
use vellev_storage::coordinator::{LogPosition, ManifestRef, PartitionKey};
use vellev_storage::StorageError;

use crate::{key, Inner, Node, NodeError, Result};

/// How long a claim to back a tenant up or restore it lasts: longer than
/// any backup takes.
const HOLD: Duration = Duration::from_secs(6 * 60 * 60);
/// How often a node looks for tenants due a backup.
const LOOKED_FOR_EVERY: Duration = Duration::from_secs(10 * 60);
/// How many tenants are looked through at a time.
const PAGE: usize = 256;
/// How many times a restore starts again when cleaning up deletes a file
/// it needs from the store before it is read.
const ATTEMPTS: usize = 3;

impl Node {
    /// Back `tenant` up now, whenever it was last backed up.
    pub async fn back_up(&self, tenant: &TenantId) -> Result<Backup> {
        let backups = self.0.backups()?;
        if self.0.coordinator.committed(&key(tenant)).await?.is_none() {
            return Err(NodeError::UnknownTenant(tenant.clone()));
        }
        let claim = self.0.claim(tenant).await?;
        let (inner, tenant) = (self.0.clone(), tenant.clone());
        finished(async move { inner.backed_up(&backups, &tenant, claim).await }).await
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
        backups
            .find(tenant, id)
            .await?
            .ok_or_else(|| NodeError::UnknownBackup(tenant.clone(), id))?;
        self.0.coordinator.register(&key(tenant)).await?;
        let claim = self.0.claim(tenant).await?;
        let (node, tenant) = (self.clone(), tenant.clone());
        finished(async move {
            let restored = node.restored_from(&backups, &tenant, id).await;
            node.0.let_go(&tenant, claim, true).await;
            restored
        })
        .await
    }

    /// The restore of [`Self::restore_backup`], under its claim. A tenant
    /// that never had a commit, as one just made has not, takes the keys
    /// and window the backup kept, any it lacks, so that a restore that
    /// failed part way can be asked for again.
    async fn restored_from(
        &self,
        backups: &Backups,
        tenant: &TenantId,
        id: Uuid,
    ) -> Result<(Backup, LogPosition)> {
        let backup = backups
            .find(tenant, id)
            .await?
            .ok_or_else(|| NodeError::UnknownBackup(tenant.clone(), id))?;
        let taken = backup
            .partitions
            .iter()
            .find(|taken| taken.key(tenant) == key(tenant))
            .ok_or_else(|| StorageError::Corrupt {
                key: id.to_string(),
                reason: format!("backup {id} holds nothing of tenant {tenant}"),
            })?;
        let coordinator = &self.0.coordinator;
        let fresh = coordinator
            .committed(&key(tenant))
            .await?
            .is_some_and(|committed| committed.manifest.is_none());
        if fresh {
            coordinator
                .set_restore_window(tenant, backup.restore_window)
                .await?;
            let held = coordinator.keys(tenant).await?;
            let missing = backup
                .keys
                .iter()
                .filter(|kept| held.iter().all(|key| key.id != kept.id));
            for kept in missing {
                coordinator.issue_key(tenant, kept.id, &kept.hash).await?;
            }
        }
        let opened = self.tenant(tenant).await?;
        let head = taken.head.clone().map(ManifestRef);
        let mut attempt = 1;
        loop {
            let copied = backups.fetch_back(&self.0.blobs, &backup).await?;
            match opened.restored(head.clone(), taken.position).await {
                Err(NodeError::Storage(StorageError::Missing { key })) if attempt < ATTEMPTS => {
                    tracing::info!(%tenant, key, "a file went before it was restored; again");
                    attempt += 1;
                }
                restored => {
                    let restored = restored?;
                    tracing::info!(%tenant, backup = %id, copied, ?restored, "restored from a backup");
                    return Ok((backup, restored));
                }
            }
        }
    }

    /// Back up each tenant whose last backup is as old as the node backs
    /// tenants up, that no node is backing up already.
    pub async fn back_up_due(&self) -> Result<()> {
        self.0.back_up_due().await
    }

    /// Back up every tenant due a backup, now and then, for as long as the
    /// node runs, starting a while after it does so that nodes started
    /// together look at different times. Does nothing without a backup
    /// store.
    pub fn backing_up(&self) {
        if self.0.settings.backups.is_none() {
            return;
        }
        let inner = Arc::downgrade(&self.0);
        let first = LOOKED_FOR_EVERY.mul_f64(f64::from(Uuid::new_v4().as_u128() as u16) / 65_536.0);
        tokio::spawn(async move {
            tokio::time::sleep(first).await;
            loop {
                let Some(node) = inner.upgrade() else { return };
                if let Err(error) = node.back_up_due().await {
                    tracing::warn!(%error, "could not look for tenants to back up");
                }
                drop(node);
                tokio::time::sleep(LOOKED_FOR_EVERY).await;
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
                if self.opened().stopped {
                    return Ok(());
                }
                let due = Some(self.settings.backup_every);
                match self.coordinator.claim_backup(&key.tenant, due, HOLD).await {
                    Ok(Some(claim)) => {
                        if let Err(error) = self.backed_up(&backups, &key.tenant, claim).await {
                            tracing::warn!(tenant = %key.tenant, %error, "backup failed");
                        }
                    }
                    Ok(None) => {}
                    Err(error) => tracing::warn!(tenant = %key.tenant, %error, "could not claim"),
                }
            }
            match page.last() {
                Some(last) if page.len() == PAGE => after = Some(last.clone()),
                _ => return Ok(()),
            }
        }
    }

    async fn claim(&self, tenant: &TenantId) -> Result<Uuid> {
        self.coordinator
            .claim_backup(tenant, None, HOLD)
            .await?
            .ok_or_else(|| NodeError::BackingUp(tenant.clone()))
    }

    /// Let `claim` go, noting a backup made now if `made`. A claim that
    /// cannot be let go runs out by itself.
    async fn let_go(&self, tenant: &TenantId, claim: Uuid, made: bool) {
        if let Err(error) = self.coordinator.release_backup(tenant, claim, made).await {
            tracing::warn!(%tenant, %error, "a backup claim was left to run out");
        }
    }

    /// Back `tenant` up under `claim`, prune its backups, and let the claim
    /// go.
    async fn backed_up(&self, backups: &Backups, tenant: &TenantId, claim: Uuid) -> Result<Backup> {
        let partitions = [key(tenant)];
        let taken = backups
            .take(&self.blobs, self.coordinator.as_ref(), tenant, &partitions)
            .await;
        let made = match taken {
            Ok(backup) => self.kept(backups, tenant, backup).await,
            Err(error) => Err(error.into()),
        };
        self.let_go(tenant, claim, made.is_ok()).await;
        made
    }

    /// `backup`, if its tenant is still the one it took, with the tenant's
    /// backups pruned. One of a tenant removed while it was copied is
    /// discarded, files and all, as the tenant's purge may have run.
    async fn kept(&self, backups: &Backups, tenant: &TenantId, backup: Backup) -> Result<Backup> {
        let now = self.coordinator.committed(&key(tenant)).await?;
        let same = now.is_some_and(|now| {
            backup
                .partitions
                .iter()
                .all(|taken| taken.incarnation == now.incarnation)
        });
        if !same {
            backups.discard(&backup).await?;
            return Err(NodeError::UnknownTenant(tenant.clone()));
        }
        match backups.prune(tenant, self.settings.backups_kept).await {
            Ok(pruned) => tracing::info!(%tenant, backup = %backup.id, pruned, "backed up"),
            Err(error) => {
                tracing::warn!(%tenant, backup = %backup.id, %error, "backed up; pruning failed")
            }
        }
        Ok(backup)
    }
}

/// Run `work` to its end on a task of its own, whoever stops waiting.
async fn finished<T: Send + 'static>(
    work: impl std::future::Future<Output = Result<T>> + Send + 'static,
) -> Result<T> {
    tokio::spawn(work)
        .await
        .map_err(|error| StorageError::Background(error.to_string()))?
}
