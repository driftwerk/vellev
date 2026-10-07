//! What is kept of a removed tenant, deleted once no node can still be
//! writing it: its files in the store, its backups, and every node's copies
//! of its files. Until then the tenant cannot be made again, so that
//! nothing of a new tenant under the same id is deleted with the old.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use vellev_core::TenantId;
use vellev_storage::backups::Backups;

use crate::{Inner, Node, Result};

/// How often a node looks for tenants to purge.
const LOOKED_FOR_EVERY: Duration = Duration::from_secs(60);
/// How long a purge is remembered after it is done, for every node to drop
/// what it cached of the tenant.
const REMEMBERED: Duration = Duration::from_secs(24 * 60 * 60);
/// How long after it is purged a node still drops what it cached of a
/// tenant, if it has not yet.
const LATELY: Duration = Duration::from_secs(60 * 60);

impl Node {
    /// Purge the tenants removed at least as long ago as the node waits,
    /// and drop what this node cached of those purged lately.
    pub async fn purge_due(&self) -> Result<()> {
        self.0.purge_due().await
    }

    /// Purge, now and then, for as long as the node runs.
    pub fn purging(&self) {
        let inner = Arc::downgrade(&self.0);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(LOOKED_FOR_EVERY).await;
                let Some(node) = inner.upgrade() else { return };
                if node.opened().stopped {
                    return;
                }
                if let Err(error) = node.purge_due().await {
                    tracing::warn!(%error, "could not purge removed tenants");
                }
            }
        });
    }
}

impl Inner {
    async fn purge_due(&self) -> Result<()> {
        let purges = self
            .coordinator
            .purges(self.settings.purge_after, LATELY)
            .await?;
        for tenant in &purges.due {
            match self.purge(tenant).await {
                Ok(deleted) => tracing::info!(%tenant, deleted, "purged"),
                Err(error) => tracing::warn!(%tenant, %error, "purge failed"),
            }
        }
        let lately: HashSet<TenantId> = purges.lately.into_iter().chain(purges.due).collect();
        let fresh: Vec<TenantId> = {
            let mut dropped = self.dropped.lock().expect("dropped lock");
            let fresh = lately.difference(&dropped).cloned().collect();
            *dropped = lately;
            fresh
        };
        for tenant in fresh {
            self.blobs.forget_local(&format!("{tenant}/")).await?;
        }
        Ok(())
    }

    /// Delete everything kept of `tenant`, then note it purged. Returns
    /// how many files went.
    async fn purge(&self, tenant: &TenantId) -> Result<usize> {
        let listed = self.blobs.list(&format!("{tenant}/")).await?;
        for file in &listed {
            self.blobs.delete(&file.key).await?;
        }
        let backed_up = match self.settings.backups.clone() {
            Some(store) => Backups::new(store).purge(tenant).await?,
            None => 0,
        };
        self.coordinator.purged(tenant, REMEMBERED).await?;
        Ok(listed.len() + backed_up)
    }
}
