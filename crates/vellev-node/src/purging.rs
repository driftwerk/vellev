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
//! tenant.
//!
//! # A delete whose fate is not known
//!
//! A delete sent to a store and not answered may yet land: given up after
//! as long as a request may take, cut short with the work it was part of,
//! or dropped with the rest of its batch when one of them failed. Where it
//! lands matters, because a file's name is not the old tenant's alone. A
//! tenant made afresh under the name writes files under other names, by
//! its own incarnation and author; but one that takes the old tenant's
//! lineage up again from a backup has the backup's files written back
//! under the very keys they had, in the store, and backs them up again
//! under the very keys in the backups. A delete of the old tenant's that
//! lands after that takes a file from under the new one, which has
//! committed to it. So no tenant may be made under the name while such a
//! delete may still land, and the rule is kept by the claim:
//!
//! - A purge that is done has had every delete it sent answered. A pass
//!   counts as having deleted only if every delete of every batch was
//!   answered as made: one that fails, or is given up, fails its batch,
//!   which drops the others of it, and the pass, and the purge with it; no
//!   error from a delete is taken for anything else on the way. And the
//!   purge is noted done only by a later pass that found nothing and sent
//!   nothing. The same holds of a purge that gives up while the tenant is
//!   still being written to: every delete was answered, and its claim is
//!   let go at once.
//! - A purge that fails, or is cut short, or is dropped, once any delete
//!   was sent under its claim, leaves the claim to run out rather than let
//!   it go. A delete is sent only while the claim is taken for held, a
//!   third of its hold before it could run out; the node gives it up
//!   within what a request to the store may take; and the store is taken
//!   to apply it or let it drop within [`LANDED_AFTER`] of that. A node is
//!   not run with a claim a third of which is less than the two together.
//!   So by when another node may claim the purge, and so before the purge
//!   can be done and the tenant made again, every delete sent under the
//!   claim has landed or never will. The last step is an assumption about
//!   the store and not something the node enforces: giving a request up
//!   stops the node waiting, not the store working, and the margin given
//!   it is what [`LANDED_AFTER`] says and why. A purge that failed before
//!   any delete was sent lets its claim go at once.
//! - A node that stalls past its claim, unseen by its own timers, is held
//!   to the same: it asks its claim's clock, which counts time the machine
//!   was suspended, before every delete, and sends none once the claim is
//!   no longer taken for held; and whoever purges next does so only once
//!   the claim has run out.
//!
//! - The looking over of a tenant purged (below) deletes under no claim,
//!   and only in the backups, never in the tenants' store: what this
//!   cluster took of a lineage no tenant has. It asks before each batch
//!   that no tenant of the name is of that lineage, so a batch may be on
//!   its way as the lineage is taken up again here. What it then deletes
//!   late is a file of this cluster's backups of that lineage, which a
//!   backup the tenant takes afterwards may list: that backup looks again
//!   once it is written that every file it lists is there, and copies
//!   back any that went, as does the next that lists the file.
//!
//! A machine suspended while a delete waits, as between one try and the
//! next inside the store's client, which tries again by timers that stop
//! with the machine, sends nothing once woken: the work under a claim asks
//! the claim's clock every time it is taken up again, whatever woke it,
//! and is dropped where it stands once the claim is no longer taken for
//! held.
//!
//! What stays open, and nothing short of the store refusing a stale delete
//! would close:
//!
//! - A machine suspended inside one step of the work, after the clock was
//!   asked and before the bytes left. Woken past its claim, it sends what
//!   that step was sending: at most [`DELETED_AT_ONCE`] deletes, of files
//!   it listed before it stalled.
//! - A pause that stops every clock the machine has, as a hypervisor
//!   freezing a guest does: no clock it can ask counts it, and the node
//!   goes on as if no time had passed. It sends the batch it was
//!   sending, at most [`DELETED_AT_ONCE`] deletes, and no other: the
//!   claim is renewed with the coordinator before each batch, and is not
//!   once another node holds it, the purge is done or the tenant is there
//!   again.
//!
//! A file in the backups deleted so is put back as any deleted late from
//! under a backup is, by the backup looking again once it is written and
//! by the next that lists the file. A file in the tenants' store is not
//! put back: the tenant fails loudly when it is next opened, naming the
//! file, and the file is still in the backup the lineage was taken up
//! from. A restore from a backup looks, just before it commits, that every
//! file the backup lists is there, by name alone, not by size or sum, and
//! fetches again any that went; a delete that lands after the commit is
//! not seen until the file is read.
//!
//! [`DELETED_AT_ONCE`]: vellev_storage::backups::DELETED_AT_ONCE
//! [`LANDED_AFTER`]: crate::LANDED_AFTER
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
use vellev_storage::coordinator::{Coordinator, PurgeClaim, Removed};
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
        // What becomes of the claim is what keeps a delete of unknown fate
        // from landing on a tenant made since: see the module's account.
        let claim = self.purge_claim(tenant, purge.id, hold).renewed(asked);
        match self.purged_under(&purge, &claim).await {
            // Done: the coordinator has ended the claim.
            Ok(Some(deleted)) => {
                claim.ended();
                Ok(Some(deleted))
            }
            // Still written to: every delete sent was answered, and the
            // next pass may go on at once.
            Ok(None) => {
                claim.let_go(false).await;
                Ok(None)
            }
            // Failed or cut short: dropped, the claim is let go if no
            // delete was sent under it, and otherwise left to run out, as
            // it is if this future is dropped before it gets here.
            Err(error) => {
                drop(claim);
                Err(error)
            }
        }
    }

    /// The purge `claimed`, under `claim`: everything found kept of the
    /// tenant deleted, its history forgotten and the purge noted done, in
    /// which case the coordinator has ended the claim. Returns how many
    /// files went, or nothing if the tenant is still being written to.
    async fn purged_under(&self, claimed: &PurgeClaim, claim: &Claim) -> Result<Option<usize>> {
        let tenant = &claimed.removed.tenant;
        let mut deleted = 0;
        for _ in 0..ROUNDS {
            let found = claim.during(self.deleted(&claimed.removed, claim)).await?;
            if found == 0 {
                claim.during(self.forgotten(tenant, claim)).await?;
                self.coordinator
                    .purged(tenant, claimed.id, REMEMBERED)
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
            deleted: Arc::default(),
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
        let store = claim.deleting_through(Arc::new(self.store()));
        backups::delete(store.as_ref(), tenant, &files, claim)
            .await
            .map_err(claimed)?;
        let backed_up = match self.settings.backups {
            Some(_) => {
                held(claim, tenant).await?;
                let kept = self.kept(removed).await?;
                let backups = self.backups_through(|store| claim.deleting_through(store))?;
                let purged = backups.purge(&kept, claim).await;
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
