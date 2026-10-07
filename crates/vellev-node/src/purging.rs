//! What is kept of a removed tenant, deleted once any node that held it
//! has noticed and stopped committing: its files in the store, its backups,
//! and every node's copies of its files. Until then the tenant cannot be
//! made again, so that nothing of a new tenant under the same id is deleted
//! with the old. One node at a time purges a tenant, by a claim in the
//! coordinator, and only the node holding it may say it is done.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, StreamExt, TryStreamExt};
use vellev_core::TenantId;
use vellev_storage::backups::Backups;
use vellev_storage::blobs::Blobs;

use crate::{key, Inner, Node, Result};

/// How often a node looks for tenants to purge.
const LOOKED_FOR_EVERY: Duration = Duration::from_secs(60);
/// How long a purge is remembered after it is done, for every node to drop
/// what it cached of the tenant.
const REMEMBERED: Duration = Duration::from_secs(24 * 60 * 60);
/// How long a claim to purge a tenant lasts.
const HOLD: Duration = Duration::from_secs(60 * 60);
/// How many files are deleted at once.
const AT_ONCE: usize = 16;

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
            .purges(self.settings.purge_after, REMEMBERED)
            .await?;
        for tenant in &purges.due {
            match self.purge(tenant).await {
                Ok(Some(deleted)) => tracing::info!(%tenant, deleted, "purged"),
                Ok(None) => {}
                Err(error) => tracing::warn!(%tenant, %error, "purge failed"),
            }
        }
        let lately: HashSet<TenantId> = purges.lately.into_iter().collect();
        let fresh: Vec<TenantId> = {
            let mut dropped = self.dropped.lock().expect("dropped lock");
            dropped.retain(|tenant| lately.contains(tenant));
            lately.difference(&dropped).cloned().collect()
        };
        for tenant in fresh {
            let gone = self.coordinator.committed(&key(&tenant)).await?.is_none();
            if gone {
                self.blobs.forget_local(&format!("{tenant}/")).await?;
            }
            self.dropped.lock().expect("dropped lock").insert(tenant);
        }
        Ok(())
    }

    /// Delete everything kept of `tenant` under a claim, then note it
    /// purged. Returns how many files went, or nothing if another node is
    /// purging it.
    async fn purge(&self, tenant: &TenantId) -> Result<Option<usize>> {
        let Some(claim) = self.coordinator.claim_purge(tenant, HOLD).await? else {
            return Ok(None);
        };
        if self.coordinator.committed(&key(tenant)).await?.is_some() {
            tracing::error!(%tenant, "a tenant to be purged is registered; left be");
            return Ok(None);
        }
        let listed: Vec<String> = self
            .blobs
            .list(&format!("{tenant}/"))
            .await?
            .into_iter()
            .map(|file| file.key)
            .collect();
        let deleted = listed.len();
        let deletes = listed
            .into_iter()
            .map(|key| delete(self.blobs.as_ref(), key));
        stream::iter(deletes)
            .buffer_unordered(AT_ONCE)
            .try_collect::<()>()
            .await?;
        let backed_up = match self.settings.backups.clone() {
            Some(store) => Backups::new(store).purge(tenant).await?,
            None => 0,
        };
        self.coordinator.purged(tenant, claim, REMEMBERED).await?;
        Ok(Some(deleted + backed_up))
    }
}

async fn delete(blobs: &dyn Blobs, key: String) -> vellev_storage::Result<()> {
    blobs.delete(&key).await
}
