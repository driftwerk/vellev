//! What is kept of a removed tenant, deleted once any node that held it
//! has noticed and stopped committing: its files in the store, its backups,
//! and every node's copies of its files. Until then the tenant cannot be
//! made again, so that nothing of a new tenant under the same name is
//! deleted with the old.
//!
//! One node at a time purges a tenant, by a claim in the coordinator, held
//! and fenced as a claim on a tenant's backups is: renewed while the purge
//! runs, taken for lost a third of its hold before another could take it,
//! and renewed with the coordinator before every listing and every batch
//! of deletes. The coordinator renews it only while the purge is still to
//! be done, the claim is still this node's, and the tenant has not been
//! made again: a node that stalled past its claim, while another finished
//! the purge and the tenant was made again, deletes nothing of the new
//! tenant. What it may still delete is what it was told it might before it
//! stalled: the files of the batch it had listed then. In the store those
//! are the old tenant's, whose names no later tenant's files take. In the
//! backups they are of the old tenant's lineage and this cluster's taking,
//! which is only ever some other tenant's again if that very lineage is
//! taken up again here, from a backup another cluster took or one put
//! back, and backed up, all before the stalled node goes on: a file its new
//! backup reuses may then be among them, and is put back as any file
//! deleted late from under a backup is, by the backup looking again once it
//! is written and by the next that lists the file.
//!
//! Removing a tenant removes everything kept of it, which a node still
//! writing for it when it is removed could undo: a writer uploading a file,
//! or a backup copying one, perhaps from its cache long after the store
//! lost the file. So:
//!
//! 1. A purge waits [`Settings::purge_after`], and where tenants are backed
//!    up at least as long as a backup claim lasts: by then no node takes a
//!    claim on the tenant's backups, taken before it was removed, for held.
//! 2. A purge is done only once a pass over the tenant's files and backups
//!    found nothing left to delete.
//! 3. For as long as a purge is remembered, [`REMEMBERED`], the tenant is
//!    looked over again every [`LOOKED_OVER_EVERY`], by one node at a time.
//!    The backups this cluster took of it lie under its own lineage and
//!    the cluster's id, so whatever is found there is deleted, whether or
//!    not a tenant of the name has been made since; unless that tenant is
//!    the very one, taken up again from a backup. Its
//!    files in the store lie under its name alone, so they are looked for
//!    only while no tenant of the name is there: if any is found, the purge
//!    is taken up again in the coordinator, which refuses to make the
//!    tenant again meanwhile, and done as before.
//!
//! What is left for good is only what is written later than that: a file
//! that lands more than [`REMEMBERED`] after its tenant's purge; or one
//! that lands in the store, not the backups, after a tenant of the name was
//! made again, which is left to that tenant's own cleaning. Neither can be
//! reached: a backup left so is of a lineage no tenant has.
//!
//! [`Settings::purge_after`]: crate::Settings::purge_after

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use uuid::Uuid;
use vellev_core::TenantId;
use vellev_storage::backups::{self, Held, Owner};
use vellev_storage::blobs::Blobs;
use vellev_storage::coordinator::{Coordinator, Removed};
use vellev_storage::StorageError;

use crate::backing::{claimed, Claim, Claimed, Right, Within};
use crate::{key, Inner, Node, NodeError, Result};

/// How long a purge is remembered after it is done: for every node to drop
/// what it cached of the tenant, and for the tenant to be looked over
/// again for anything written for it since.
pub const REMEMBERED: Duration = Duration::from_secs(24 * 60 * 60);
/// How often a tenant purged lately is looked over again: each look lists
/// what is kept of it, in the store and in its backups.
pub const LOOKED_OVER_EVERY: Duration = Duration::from_secs(15 * 60);
/// How often a node looks for removed tenants to purge, and for those
/// purged lately to look over again, unless it is told otherwise.
pub const LOOKED_FOR_EVERY: Duration = Duration::from_secs(60);
/// How many tenants a node takes to look over at a time, and looks over
/// together.
const LOOKED_OVER_AT_ONCE: usize = 32;
/// How many times in one pass a purge deletes what it finds, looking for
/// nothing left, before it leaves the tenant to a later pass.
const ROUNDS: usize = 3;

impl Node {
    /// Purge the tenants removed at least as long ago as the node waits,
    /// drop what this node cached of those purged lately, look over those
    /// of them due it, and forget the restores recorded too long ago to be
    /// asked for again, whatever became of their tenants.
    pub async fn purge_due(&self) -> Result<()> {
        self.0.purge_due().await
    }

    /// Purge, now and then, for as long as the node runs.
    pub fn purging(&self) {
        let inner = Arc::downgrade(&self.0);
        let every = self.0.settings.purges_looked_for_every;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
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
    /// How long after a tenant is removed it is purged: as long as the node
    /// is set to wait and, where tenants are backed up, no sooner than a
    /// claim on its backups taken before it was removed has run out.
    fn purged_after(&self) -> Duration {
        match self.settings.backups {
            Some(_) => self.settings.purge_after.max(self.settings.backup_claim),
            None => self.settings.purge_after,
        }
    }

    /// How long it may be until a tenant whose purge was asked for `asked`
    /// ago is purged, and may be made again: until its purge is due, and a
    /// node next looks for purges to do.
    pub(crate) fn purged_within(&self, asked: Duration) -> Duration {
        self.purged_after().saturating_sub(asked) + self.settings.purges_looked_for_every
    }

    async fn purge_due(&self) -> Result<()> {
        let forgotten = self.coordinator.forget_restores().await?;
        if forgotten > 0 {
            tracing::info!(forgotten, "forgot old restores");
        }
        let purges = self
            .coordinator
            .purges(self.purged_after(), REMEMBERED)
            .await?;
        for tenant in &purges.due {
            self.purge_logged(tenant).await;
        }
        self.drop_cached(purges.lately.into_iter().collect())
            .await?;
        self.look_over_due().await
    }

    /// Drop what this node cached of each tenant purged `lately` that it
    /// has not dropped yet, if the tenant has not been made again.
    async fn drop_cached(&self, lately: HashSet<TenantId>) -> Result<()> {
        let fresh: Vec<TenantId> = {
            let mut dropped = self.dropped.lock().expect("dropped lock");
            dropped.retain(|tenant| lately.contains(tenant));
            lately.difference(&dropped).cloned().collect()
        };
        for together in fresh.chunks(LOOKED_OVER_AT_ONCE) {
            let drops = together.iter().map(|tenant| self.dropped_cached(tenant));
            futures::future::try_join_all(drops).await?;
        }
        Ok(())
    }

    /// Drop what this node cached of `tenant`, purged, unless it has been
    /// made again, and note it done.
    async fn dropped_cached(&self, tenant: &TenantId) -> Result<()> {
        let gone = self.coordinator.committed(&key(tenant)).await?.is_none();
        if gone {
            self.blobs.forget_local(&format!("{tenant}/")).await?;
        }
        let mut dropped = self.dropped.lock().expect("dropped lock");
        dropped.insert(tenant.clone());
        Ok(())
    }

    /// Look over the tenants purged lately that are due it, a few at a
    /// time, for as long as the coordinator has any to give: each is given
    /// to one node, and noted as looked over as it is.
    async fn look_over_due(&self) -> Result<()> {
        loop {
            let due = self
                .coordinator
                .look_overs(REMEMBERED, LOOKED_OVER_EVERY, LOOKED_OVER_AT_ONCE)
                .await?;
            let looks = due.iter().map(|removed| self.looked_over_logged(removed));
            futures::future::join_all(looks).await;
            if due.len() < LOOKED_OVER_AT_ONCE {
                return Ok(());
            }
        }
    }

    async fn looked_over_logged(&self, removed: &Removed) {
        if let Err(error) = self.looked_over(removed).await {
            tracing::warn!(tenant = %removed.tenant, %error, "could not look for what is left of a purged tenant");
        }
    }

    /// Delete what was written since its purge for the tenant `removed`
    /// was. The backups this cluster took of it are its own lineage's, and
    /// go whoever is of its name now, unless that is the tenant itself,
    /// taken up again. Its files in the store go only while no tenant of
    /// the name is there, by its purge taken up again.
    async fn looked_over(&self, removed: &Removed) -> Result<()> {
        let tenant = &removed.tenant;
        match self.coordinator.identity(tenant).await? {
            Some(now) if now.lineage == removed.lineage => Ok(()),
            Some(_) => self.swept(removed).await,
            None => {
                if self.left(removed).await? && self.coordinator.reopen_purge(tenant).await? {
                    tracing::warn!(%tenant, "files were written for a tenant after its purge; purging it again");
                    self.purge_logged(tenant).await;
                }
                Ok(())
            }
        }
    }

    /// Delete what this cluster keeps in the backups of the tenant
    /// `removed` was, though a tenant of its name has been made since: for
    /// as long as that tenant is not of its lineage, asked before each
    /// batch.
    async fn swept(&self, removed: &Removed) -> Result<()> {
        if self.settings.backups.is_none() {
            return Ok(());
        }
        let (backups, kept) = (self.backups()?, self.kept(removed).await?);
        if !backups.any_kept(&kept).await? {
            return Ok(());
        }
        let dead = Dead {
            coordinator: self.coordinator.as_ref(),
            removed,
        };
        let deleted = backups.purge(&kept, &dead).await?;
        tracing::warn!(tenant = %removed.tenant, lineage = %removed.lineage, deleted, "backup files were written for a tenant after its purge; deleted");
        Ok(())
    }

    /// The tenant `removed` was, as the backups this cluster took of it
    /// know it.
    async fn kept(&self, removed: &Removed) -> Result<Owner> {
        Ok(Owner {
            tenant: removed.tenant.clone(),
            lineage: removed.lineage,
            cluster: self.coordinator.cluster().await?,
        })
    }

    /// Whether anything is kept of the tenant `removed` was, in the store
    /// or by this cluster in its backups.
    async fn left(&self, removed: &Removed) -> Result<bool> {
        if !self.files(&removed.tenant).await?.is_empty() {
            return Ok(true);
        }
        match self.settings.backups {
            Some(_) => {
                let kept = self.kept(removed).await?;
                Ok(self.backups()?.any_kept(&kept).await?)
            }
            None => Ok(false),
        }
    }

    /// The store tenants are kept in, each request to it given up after as
    /// long as one may take, as those to the backup store are: so that a
    /// purge waits for none past its claim.
    fn store(&self) -> Within {
        Within {
            inner: self.blobs.clone(),
            within: self.settings.request_bounds.store,
        }
    }

    /// The keys of every file kept under `tenant`'s name in the store.
    async fn files(&self, tenant: &TenantId) -> Result<Vec<String>> {
        let listed = self.store().list(&format!("{tenant}/")).await?;
        Ok(listed.into_iter().map(|file| file.key).collect())
    }

    async fn purge_logged(&self, tenant: &TenantId) {
        match self.purge(tenant).await {
            Ok(Some(deleted)) => tracing::info!(%tenant, deleted, "purged"),
            Ok(None) => {}
            Err(NodeError::ClaimLost(_)) => {
                tracing::warn!(%tenant, "the claim to purge a tenant was lost; its purge is left to whoever holds it")
            }
            Err(error) => tracing::warn!(%tenant, %error, "purge failed"),
        }
    }

    /// Delete everything kept of `tenant` under a claim, and note it purged
    /// once nothing is found left. Returns how many files went, or nothing
    /// if it is not due, another node is purging it, or it is still being
    /// written to.
    async fn purge(&self, tenant: &TenantId) -> Result<Option<usize>> {
        let hold = self.settings.backup_claim;
        let asked = self.settings.claim_clock.now();
        let claimed = self
            .coordinator
            .claim_purge(tenant, self.purged_after(), hold)
            .await?;
        let Some(purge) = claimed else {
            return Ok(None);
        };
        let claim = self.purge_claim(tenant, purge.id, hold).renewed(asked);
        let mut deleted = 0;
        for _ in 0..ROUNDS {
            let found = claim.during(self.deleted(&purge.removed, &claim)).await?;
            if found == 0 {
                claim.during(self.forgotten(tenant, &claim)).await?;
                self.coordinator
                    .purged(tenant, purge.id, REMEMBERED)
                    .await?;
                return Ok(Some(deleted));
            }
            deleted += found;
        }
        tracing::warn!(
            %tenant,
            deleted,
            "a tenant being purged is still written to; its purge is left to a later pass"
        );
        Ok(None)
    }

    /// Forget what is kept of the removed `tenant`'s history, a batch at a
    /// time, each well within any bound on one call to the coordinator and
    /// each once `claim` is renewed: a node that stalled past its claim
    /// forgets nothing of a tenant made again meanwhile.
    async fn forgotten(&self, tenant: &TenantId, claim: &Claim) -> Result<()> {
        let mut after = None;
        loop {
            held(claim, tenant).await?;
            after = self
                .coordinator
                .forget_removed(&key(tenant), after, crate::FORGOTTEN_AT_ONCE)
                .await?;
            if after.is_none() {
                return Ok(());
            }
        }
    }

    /// The claim `id` to purge `tenant`, held for `hold` at a time.
    fn purge_claim(&self, tenant: &TenantId, id: Uuid, hold: Duration) -> Claimed {
        Claimed {
            right: Right::Purge,
            coordinator: self.coordinator.clone(),
            clock: self.settings.claim_clock.clone(),
            tenant: tenant.clone(),
            id,
            hold,
            within: self.settings.request_bounds.coordinator,
        }
    }

    /// Delete everything found kept of the tenant `removed` was, now, under
    /// `claim`: renewed with the coordinator before each listing and each
    /// batch of deletes, and nothing more is deleted once it is not.
    /// Returns how many files were found.
    async fn deleted(&self, removed: &Removed, claim: &Claim) -> Result<usize> {
        let tenant = &removed.tenant;
        held(claim, tenant).await?;
        let files = self.files(tenant).await?;
        backups::delete(&self.store(), tenant, &files, claim)
            .await
            .map_err(claimed)?;
        let backed_up = match self.settings.backups {
            Some(_) => {
                held(claim, tenant).await?;
                let kept = self.kept(removed).await?;
                let purged = self.backups()?.purge(&kept, claim).await;
                purged.map_err(claimed)?
            }
            None => 0,
        };
        Ok(files.len() + backed_up)
    }
}

/// Renew `claim` on `tenant` with the coordinator, and refuse to go on if
/// it is no longer held.
async fn held(claim: &Claim, tenant: &TenantId) -> Result<()> {
    if claim.held().await? {
        return Ok(());
    }
    Err(NodeError::ClaimLost(tenant.clone()))
}

/// The right to delete what a lineage no tenant has any more left in its
/// backups: held for as long as the tenant of its name is not of it, as
/// one taken up again from a backup would be.
struct Dead<'a> {
    coordinator: &'a dyn Coordinator,
    removed: &'a Removed,
}

#[async_trait::async_trait]
impl Held for Dead<'_> {
    async fn held(&self) -> vellev_storage::Result<bool> {
        let now = self
            .coordinator
            .identity(&self.removed.tenant)
            .await
            .map_err(StorageError::from)?;
        Ok(now.is_none_or(|now| now.lineage != self.removed.lineage))
    }

    /// Nothing of this right runs out by the clock.
    fn still(&self) -> bool {
        true
    }
}
