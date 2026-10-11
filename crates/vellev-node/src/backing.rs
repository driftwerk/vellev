//! Tenants backed up to a second store, daily and when asked, and put back
//! from it: into their own cluster, or into one that has lost them.
//!
//! One node at a time backs a tenant up or restores it from a backup, by a
//! claim in the coordinator, so that pruning never deletes a file another
//! backup is copying or a restore is reading. A claim is taken by an id the
//! node picks, so that one asked for again after its answer was lost is the
//! node's still. It is held for [`Settings::backup_claim`] at a time and
//! renewed every third of that while the work runs, so that a node gone, or
//! a claim it never heard it took, blocks the tenant for no longer.
//!
//! A claim is fenced in three layers, as no one clock can fence it whole:
//!
//! 1. The node takes its claim for lost a third of a hold before the
//!    coordinator could let another take it, measured from when it last
//!    asked to renew it, on a clock that counts the time the system was
//!    suspended ([`ClaimClock`]). Every check reads that deadline against
//!    the clock: before each file a backup copies, once it is read; its
//!    complete record; each file a restore fetches, once it is read, and
//!    its commit; and each file deleted. A node
//!    that stalled or slept past the deadline does nothing more. The margin
//!    is at least as long as the node waits for a request to the backup
//!    store ([`Settings::check`]), each of which it gives up after that
//!    long, so that while the machine runs it waits for no write once
//!    another may take the claim. That wait is timed by the runtime, whose
//!    clock stops with the machine: a request in flight when the machine
//!    slept is still waited for when it wakes, and may have landed since,
//!    as may one the node gave up on, the store having had it. Those are
//!    the third layer's to put right.
//! 2. Deleting goes [`DELETED_AT_ONCE`] files at a time, each batch once
//!    the coordinator has renewed the claim, so a node whose claim another
//!    took deletes nothing more once it asks. This is the fence that needs
//!    no clock: a guest paused by its hypervisor resumes with its clocks
//!    where they were, and the first layer sees nothing.
//! 3. What can still happen is a delete that lands after another took the
//!    claim, of a file that one's backup reuses: sent before a stall, or
//!    one of the at most [`DELETED_AT_ONCE`] of a batch a paused guest was
//!    told it might delete before it was paused. So a backup, once its
//!    complete record is written, looks again, under its claim, that every
//!    file it lists is there, and copies any that is not from the tenant's
//!    own store. A backup that cannot be made whole so is forgotten and the
//!    run fails. A delete that lands later still is put right by the next
//!    backup that lists the file, as every backup copies whatever its files
//!    lack; until then, restoring the backup it broke fails for the missing
//!    file rather than restore anything else.
//!
//! Asking the coordinator before every delete, not every sixteenth, would
//! shrink what a paused guest can delete late from sixteen files to one, at
//! sixteen times the writes to the coordinator, and the one would still
//! need the third layer; asking again once some seconds have gone by since
//! it last answered would not help at all, as a paused guest's clock shows
//! none gone. So the batch stays.
//!
//! The steps of a restore before its commit stop once the claim is lost;
//! the commit, once begun, is not the claim's to stop: it runs to its end
//! and its outcome is the restore's. A restore runs as any restore does
//! ([`Tenant::restoring`]): on a task of its own, stopped by the node losing
//! its hold on the tenant, recorded with its commit under the request that
//! asked for it, so that the request made again is answered as it was. A
//! claim whose work is stopped so is let go all the same, by a task of its
//! own. A restore that adopts a backup takes
//! nothing of it before that commit: it has the coordinator note what the
//! tenant is to become, and the commit, of the lease the node then holds
//! and no other, makes it so in the one transaction, or the tenant stays
//! as it was. A backup whose complete record was
//! written is the run's outcome whatever the claim does after. Each piece
//! of work runs to its end, or to where it stops, whoever stops waiting for
//! it, and its claim is let go when it does. A claim is left to run out
//! within its hold by a panic, by the runtime stopping, by a coordinator
//! that does not answer the letting go, and, on purpose, where it is a
//! claim to purge under which a delete was sent and the purge did not end
//! well ([`Claim::left_to_run_out`]). The daily backups
//! need no node to hold the tenant, so tenants nobody has used for a while
//! are backed up too; a tenant that never committed is not, holding
//! nothing. A backup of a tenant removed meanwhile forgets itself, and what
//! it wrote is the tenant's purge to delete, which looks again for a day
//! after it is done.
//!
//! [`Settings::backup_claim`]: crate::Settings::backup_claim
//! [`Settings::check`]: crate::Settings::check
//! [`DELETED_AT_ONCE`]: vellev_storage::backups::DELETED_AT_ONCE

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::stream::{self, BoxStream, StreamExt};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use uuid::Uuid;
use vellev_core::{RequestId, TenantId};
use vellev_storage::backups::{Backup, Backups, Held, Owner};
use vellev_storage::blobs::{unanswered, Blobs, Created, Listed};
use vellev_storage::coordinator::{
    Adoption, Coordinator, CoordinatorError, Generation, Identity, LogPosition, ManifestRef,
    PartitionKey, Restored, MAX_BACKUP_HOLD,
};
use vellev_storage::StorageError;

use crate::{
    key, Asking, ClaimClock, Inner, Node, NodeError, RequestBounds, Result, Target, Tenant,
};

/// How many times the coordinator is asked to take, renew or let go of a
/// claim when it fails to answer; asking again is harmless.
const CLAIM_ATTEMPTS: u32 = 3;
/// How long to wait before asking again, times the attempts so far.
const CLAIM_RETRY_DELAY: Duration = Duration::from_millis(200);
/// How long after the node gave a request to object storage up the store
/// is taken to have applied it or let it drop. This is an assumption about
/// the store, and the one thing here the node cannot see to itself: giving
/// a request up stops the node waiting for it, not the store working on it.
/// A store answers a delete in milliseconds and gives up a request it
/// holds within seconds, so half a minute is several times what any does;
/// a store that applied one later still would be beyond any margin, and
/// what it could then undo is what `purging` says it could.
pub const LANDED_AFTER: Duration = Duration::from_secs(30);
/// How many tenants are looked through at a time.
const PAGE: usize = 256;
/// How many times a restore starts again when cleaning up deletes a file
/// it needs from the store before it is read.
const ATTEMPTS: usize = 3;

impl RequestBounds {
    /// Why a backup claim of `hold` cannot be held with requests that take
    /// as long as these bounds say, if it cannot: its margin, a third of
    /// it, must outlast a request to object storage already sent and the
    /// [`LANDED_AFTER`] the store is given to apply it once given up; and
    /// the rest must leave time for every attempt to take or renew it.
    pub fn claim_problem(&self, hold: Duration) -> Option<String> {
        claim_problem(hold, *self)
    }
}

fn claim_problem(hold: Duration, bounds: RequestBounds) -> Option<String> {
    let margin = hold / 3;
    let asking = bounds.coordinator * CLAIM_ATTEMPTS
        + (1..CLAIM_ATTEMPTS)
            .map(|attempt| CLAIM_RETRY_DELAY * attempt)
            .sum::<Duration>();
    if hold > MAX_BACKUP_HOLD {
        Some(format!(
            "a backup claim of {hold:?} is longer than the {MAX_BACKUP_HOLD:?} one may last"
        ))
    } else if margin < bounds.store + LANDED_AFTER {
        Some(format!(
            "a backup claim of {hold:?} is taken for lost a third of it before it runs out, \
             which must outlast a request to object storage, {:?}, and the {LANDED_AFTER:?} \
             the store is given to apply one given up",
            bounds.store
        ))
    } else if hold - margin < asking {
        Some(format!(
            "a backup claim of {hold:?} leaves {:?} before it is taken for lost, less than \
             the {asking:?} its {CLAIM_ATTEMPTS} attempts to take or renew it may take",
            hold - margin
        ))
    } else {
        None
    }
}

impl Node {
    /// Back `tenant` up now, whenever it was last backed up.
    pub async fn back_up(&self, tenant: &TenantId) -> Result<Backup> {
        let backups = self.0.backups()?;
        if self.0.coordinator.identity(tenant).await?.is_none() {
            return Err(NodeError::UnknownTenant(tenant.clone()));
        }
        let (inner, tenant) = (self.0.clone(), tenant.clone());
        finished(async move {
            let claim = inner.claim(&tenant, None).await?;
            let claim = claim.ok_or_else(|| NodeError::BackingUp(tenant.clone()))?;
            inner.backed_up(&backups, claim).await
        })
        .await
    }

    /// The backups `tenant` may be put back from, oldest first. Of a tenant
    /// that has been written to they are its own, whichever cluster took
    /// them, and none of any other tenant of its name. Of a tenant this
    /// cluster does not have, or has and has never written to, they are
    /// every backup kept under the name, each saying the lineage it is of
    /// and the cluster that took it: what the tenant may [take up]. Only
    /// its own are restored from otherwise.
    ///
    /// [take up]: Self::adopt_backup
    pub async fn backups(&self, tenant: &TenantId) -> Result<Vec<Backup>> {
        let backups = self.0.backups()?;
        match self.0.known(tenant).await? {
            Some((now, owner)) if now.committed => Ok(backups.list(&owner).await?),
            _ => Ok(backups.of_any(tenant).await?),
        }
    }

    /// Put `tenant` back as its own backup `id` took it, as a new commit,
    /// for `request`. It keeps the keys it has now, so that a key revoked
    /// since stays revoked. A backup of another tenant of its name is not
    /// found. Returns what it went back to and the commit that put it back.
    /// The request made again is answered as it was, as for
    /// [`Self::restore`].
    pub async fn restore_backup(
        &self,
        tenant: &TenantId,
        id: Uuid,
        request: RequestId,
    ) -> Result<Restored> {
        let backups = self.0.backups()?;
        let asking = Asking::of(request, Target::Backup(id));
        let known = self.0.known(tenant).await?;
        let (_, owner) = known.ok_or_else(|| NodeError::UnknownTenant(tenant.clone()))?;
        let Some(backup) = backups.find(&owner, id).await? else {
            // A restore made from a backup pruned since is answered still.
            let coordinator = self.0.coordinator.as_ref();
            let recorded = asking.recorded(coordinator, tenant).await?;
            return recorded.ok_or_else(|| NodeError::UnknownBackup(tenant.clone(), id));
        };
        self.restored(backups, backup, Asked::ItsOwn, asking).await
    }

    /// Take up again a tenant this cluster has lost, as backup `id` kept
    /// under its name took it: the tenant is made, if there is none, and
    /// with the commit that restores it becomes the tenant the backup is
    /// of, with the restore window it had and, `with_keys`, the keys the
    /// backup kept, which are those it had when the backup was taken: one
    /// revoked since comes back. Until that commit nothing of the backup is
    /// the tenant's, and a restore that fails leaves it as it was, to be
    /// asked again or to take up another.
    ///
    /// Only a tenant that has never been written to and has no key takes a
    /// backup up: one written to is some other tenant of the name, and a
    /// key issued to an empty tenant was issued for whatever was to be put
    /// in it, which may not be this. Both are refused, and the keys are to
    /// be revoked first. This is what brings a tenant into a cluster from
    /// another's backups, so it is for whoever runs the cluster to ask for.
    /// The request made again is answered as it was, as for
    /// [`Self::restore`]; one that asked for another backup, or for the
    /// keys the other way, is refused.
    pub async fn adopt_backup(
        &self,
        tenant: &TenantId,
        id: Uuid,
        with_keys: bool,
        request: RequestId,
    ) -> Result<Restored> {
        let backups = self.0.backups()?;
        let asking = Asking::of(
            request,
            Target::Adopt {
                backup: id,
                with_keys,
            },
        );
        let kept = backups.of_any(tenant).await?;
        let backup = kept.into_iter().find(|backup| backup.id == id);
        let Some(backup) = backup else {
            let coordinator = self.0.coordinator.as_ref();
            let recorded = asking.recorded(coordinator, tenant).await?;
            return recorded.ok_or_else(|| NodeError::UnknownBackup(tenant.clone(), id));
        };
        // Made already, or made and written to since: the request that
        // made it is answered as it was before anything is refused.
        let coordinator = self.0.coordinator.as_ref();
        if let Some(recorded) = asking.recorded(coordinator, tenant).await? {
            return Ok(recorded);
        }
        self.0.adoptable(tenant).await?;
        self.0.register(tenant).await?;
        let asked = Asked::ToAdopt { with_keys };
        self.restored(backups, backup, asked, asking).await
    }

    /// Restore its tenant from `backup` as `asking` asks, under a claim,
    /// as [`Tenant::restoring`] runs a restore: on a task of its own, to
    /// the end whoever stops waiting, unless the node's hold on the tenant
    /// is lost. The claim is let go whatever ends the restore.
    async fn restored(
        &self,
        backups: Backups,
        backup: Backup,
        asked: Asked,
        asking: Asking,
    ) -> Result<Restored> {
        let tenant = backup.tenant.clone();
        self.restoring(&tenant, asking, move |node, opened| {
            let (backups, backup) = (backups.clone(), backup.clone());
            async move {
                let tenant = opened.id();
                let claim = node.0.claim(tenant, None).await?;
                let claim = claim.ok_or_else(|| NodeError::BackingUp(tenant.clone()))?;
                let restored = node
                    .restored_from(&backups, &opened, &backup, asked, asking, &claim)
                    .await;
                if restored.is_err() {
                    let generation = opened.generation();
                    node.0.adopting_no_more(tenant, asked, generation).await;
                }
                // A restore is no backup: a tenant put back as a backup
                // took it, its own or one it has taken up, is as due one
                // of its own as before, and one whose restore failed is
                // too. If it was left empty, none is taken.
                claim.let_go(false).await;
                restored
            }
        })
        .await
    }

    /// The restore of [`Self::restored`] into `opened`, under `claim`,
    /// which may stop any step before the commit; the commit, once begun,
    /// runs to its end.
    async fn restored_from(
        &self,
        backups: &Backups,
        opened: &Tenant,
        backup: &Backup,
        asked: Asked,
        asking: Asking,
        claim: &Claim,
    ) -> Result<Restored> {
        let (tenant, id) = (&backup.tenant, backup.id);
        claim.during(self.0.still_asked(backup, asked)).await?;
        let taken = backup
            .partitions
            .iter()
            .find(|taken| taken.key(tenant) == key(tenant))
            .ok_or_else(|| StorageError::Corrupt {
                key: id.to_string(),
                reason: format!("backup {id} holds nothing of tenant {tenant}"),
            })?;
        let head = taken.head.clone().map(ManifestRef);
        let restore = asking.to(taken.position, backup.taken_at);
        // Each write into the tenant's store is given up as one to the
        // backup store is, so that none is waited for past the claim.
        let primary: Arc<dyn Blobs> = Arc::new(Within {
            inner: self.0.blobs.clone(),
            within: self.0.settings.request_bounds.store,
        });
        let mut attempt = 1;
        loop {
            let fetching = backups.fetch_back(&primary, backup, claim);
            let copied = claim
                .during(async { fetching.await.map_err(claimed) })
                .await?;
            let adopting = matches!(asked, Asked::ToAdopt { .. }).then_some(backup.lineage);
            if let Asked::ToAdopt { with_keys } = asked {
                let generation = opened.generation();
                let noting = self
                    .0
                    .adopting(backup, taken.position, with_keys, generation);
                claim.during(noting).await?;
            }
            // Looked for again just before the commit, under the claim:
            // loading the state to restore reads the files it does not
            // hold already, and no others, so one that went since it was
            // fetched, as by a delete sent long before and landing late,
            // could be committed to unseen. Any that went is fetched again
            // with the next try, and the restore refused if it keeps on.
            let looking = backups.lacking(&primary, backup);
            let lacking = claim
                .during(async { looking.await.map_err(claimed) })
                .await?;
            claim.still()?;
            let restored = match lacking.into_iter().next() {
                Some(key) => Err(NodeError::Storage(StorageError::Missing { key })),
                None => {
                    opened
                        .restored(head.clone(), restore.clone(), adopting)
                        .await
                }
            };
            match restored {
                Err(NodeError::Storage(StorageError::Missing { key })) if attempt < ATTEMPTS => {
                    tracing::info!(%tenant, key, "a file went before it was restored; again");
                    attempt += 1;
                }
                // Gone again each time it was fetched back: said of the
                // backup, with the file, rather than as a file missing
                // from somewhere.
                Err(NodeError::Storage(StorageError::Missing { key })) => {
                    let reason = format!(
                        "backup {id} of tenant {tenant} lists it, and it went from the \
                         tenant's store each of the {ATTEMPTS} times it was fetched back"
                    );
                    return Err(StorageError::Corrupt { key, reason }.into());
                }
                restored => {
                    let restored = restored?;
                    tracing::info!(%tenant, backup = %id, copied, ?restored, "restored from a backup");
                    return Ok(restored);
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
    /// store, or on a node set not to back up of its own accord.
    pub fn backing_up(&self) {
        let settings = &self.0.settings;
        if settings.backups.is_none() || !settings.backups_scheduled {
            return;
        }
        let inner = Arc::downgrade(&self.0);
        let every = self.0.settings.backups_looked_for_every;
        let first = every.mul_f64(f64::from(Uuid::new_v4().as_u128() as u16) / 65_536.0);
        tokio::spawn(async move {
            tokio::time::sleep(first).await;
            loop {
                let Some(node) = inner.upgrade() else { return };
                if node.opened().stopped {
                    return;
                }
                if let Err(error) = node.back_up_due().await {
                    tracing::warn!(%error, "could not look for tenants to back up");
                }
                drop(node);
                tokio::time::sleep(every).await;
            }
        });
    }
}

impl Inner {
    /// The backup store, each request to it given up after as long as a
    /// request to object storage may take: the bound a claim's margin is
    /// checked against is the one the node keeps to.
    pub(crate) fn backups(&self) -> Result<Backups> {
        self.backups_through(|store| store)
    }

    /// [`Self::backups`], asked through whatever `through` makes of the
    /// store so bounded.
    pub(crate) fn backups_through(
        &self,
        through: impl FnOnce(Arc<dyn Blobs>) -> Arc<dyn Blobs>,
    ) -> Result<Backups> {
        let within = self.settings.request_bounds.store;
        let inner = self.settings.backups.clone().ok_or(NodeError::NoBackups)?;
        let store = through(Arc::new(Within { inner, within }));
        Ok(Backups::new(store).timed_by(self.settings.clock.clone()))
    }

    /// Which tenant `tenant` is now, and as the backups this cluster takes
    /// of it know it, if there is one.
    async fn known(&self, tenant: &TenantId) -> Result<Option<(Identity, Owner)>> {
        let Some(now) = self.coordinator.identity(tenant).await? else {
            return Ok(None);
        };
        let cluster = self.coordinator.cluster().await?;
        Ok(Some((now, Owner::new(tenant, now, cluster))))
    }

    /// Make `tenant`, if it is not there. One whose name is still being
    /// purged is refused, saying how long it may be until that is done.
    pub(crate) async fn register(&self, tenant: &TenantId) -> Result<()> {
        match self.coordinator.register(&key(tenant)).await {
            Err(CoordinatorError::Purging { tenant, asked }) => Err(NodeError::Purging {
                tenant,
                wait: self.purged_within(asked),
            }),
            registered => Ok(registered?),
        }
    }

    /// Refuse unless `tenant` may take a backup up: it is not there, or has
    /// never been written to and has no key.
    async fn adoptable(&self, tenant: &TenantId) -> Result<()> {
        let now = self.coordinator.identity(tenant).await?;
        if now.is_some_and(|now| now.committed) {
            return Err(NodeError::NotLost(tenant.clone()));
        }
        if !self.coordinator.keys(tenant).await?.is_empty() {
            return Err(NodeError::Keyed(tenant.clone()));
        }
        Ok(())
    }

    /// Refuse unless `backup` may still be restored into its tenant as
    /// `asked`, now that the claim is held: its own while the tenant is
    /// still the one it is of, and to adopt while the tenant may take one
    /// up.
    async fn still_asked(&self, backup: &Backup, asked: Asked) -> Result<()> {
        let tenant = &backup.tenant;
        match asked {
            Asked::ToAdopt { .. } => self.adoptable(tenant).await,
            Asked::ItsOwn => match self.known(tenant).await? {
                Some((_, owner)) if owner.owns(backup) => Ok(()),
                Some(_) => Err(NodeError::UnknownBackup(tenant.clone(), backup.id)),
                None => Err(NodeError::UnknownTenant(tenant.clone())),
            },
        }
    }

    /// Have the tenant `backup` is of become that tenant with the commit
    /// that restores its partition from `position`, by the lease this node
    /// holds on it now, at `generation`, and with no other: its lineage,
    /// its restore window and, `with_keys`, its keys.
    async fn adopting(
        &self,
        backup: &Backup,
        position: LogPosition,
        with_keys: bool,
        generation: Generation,
    ) -> Result<()> {
        let tenant = &backup.tenant;
        let key = key(tenant);
        let at = position.next().ok_or_else(|| StorageError::Corrupt {
            key: backup.id.to_string(),
            reason: format!("backup {} is at the last position there is", backup.id),
        })?;
        let adoption = Adoption {
            lineage: backup.lineage,
            at,
            keys: backup.keys.iter().filter(|_| with_keys).cloned().collect(),
            restore_window: backup.restore_window,
        };
        let noted = self
            .coordinator
            .adopt_on_commit(&key, generation, &adoption);
        if noted.await? {
            return Ok(());
        }
        self.adoptable(tenant).await?;
        Err(NodeError::Lost(tenant.clone()))
    }

    /// Give up the adoption `asked` for, if one was, noted or not, by the
    /// lease at `generation`: its restore failed. One noted since by
    /// another lease, which took the tenant over meanwhile, is that
    /// lease's to make.
    async fn adopting_no_more(&self, tenant: &TenantId, asked: Asked, generation: Generation) {
        if asked == Asked::ItsOwn {
            return;
        }
        let key = key(tenant);
        let given_up = self.coordinator.adopt_no_more(&key, generation);
        if let Err(error) = given_up.await {
            tracing::warn!(%tenant, %error, "an adoption given up was left noted");
        }
    }

    async fn back_up_due(&self) -> Result<()> {
        let backups = self.backups()?;
        let due = Some(self.settings.backup_every);
        let mut after: Option<PartitionKey> = None;
        loop {
            let page = self.coordinator.partitions(after.as_ref(), PAGE).await?;
            for key in &page {
                if self.opened().stopped {
                    return Ok(());
                }
                // A partition that never committed holds nothing to back
                // up, and is passed over without taking its tenant's lock.
                match self.coordinator.committed(key).await {
                    Ok(Some(committed)) if committed.manifest.is_some() => {}
                    Ok(_) => continue,
                    Err(error) => {
                        tracing::warn!(tenant = %key.tenant, %error, "could not tell if it is due");
                        continue;
                    }
                }
                match self.claim(&key.tenant, due).await {
                    Ok(Some(claim)) => match self.backed_up(&backups, claim).await {
                        Ok(_) | Err(NodeError::Empty(_)) => {}
                        Err(error) => tracing::warn!(tenant = %key.tenant, %error, "backup failed"),
                    },
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

    /// A new claim on `tenant`, taken as [`Coordinator::claim_backup`]
    /// takes it with `due`; `None` if it was not taken. One whose every
    /// answer was lost is let go, as it may have been taken.
    async fn claim(&self, tenant: &TenantId, due: Option<Duration>) -> Result<Option<Claim>> {
        let claimed = Claimed {
            right: Right::BackUp,
            coordinator: self.coordinator.clone(),
            clock: self.settings.claim_clock.clone(),
            tenant: tenant.clone(),
            id: Uuid::new_v4(),
            hold: self.settings.backup_claim,
            within: self.settings.request_bounds.coordinator,
            deleted: Arc::default(),
        };
        let asked = self.settings.claim_clock.now();
        let (id, hold, within) = (claimed.id, claimed.hold, claimed.within);
        match retried(within, || {
            self.coordinator.claim_backup(tenant, id, due, hold)
        })
        .await
        {
            Ok(taken) => Ok(taken.then(|| claimed.renewed(asked))),
            Err(error) => {
                claimed.released(false).await;
                Err(error.into())
            }
        }
    }

    /// Back up the tenant `claim` is on, under it, prune its backups, and
    /// let the claim go.
    async fn backed_up(&self, backups: &Backups, mut claim: Claim) -> Result<Backup> {
        let made = self.backed_up_under(backups, &mut claim).await;
        claim.let_go(made.is_ok()).await;
        made
    }

    /// The backup of [`Self::backed_up`]: of the tenant as it is once the
    /// claim is held, if it is still that tenant once the backup is made,
    /// made sure of and with the tenant's backups pruned. One of a tenant
    /// removed while it was copied is forgotten: its files lie under that
    /// tenant's lineage, which its purge deletes, and looks over again for
    /// a day after.
    async fn backed_up_under(&self, backups: &Backups, claim: &mut Claim) -> Result<Backup> {
        let tenant = claim.of.tenant.clone();
        let known = self.known(&tenant).await?;
        let (_, owner) = known.ok_or_else(|| NodeError::UnknownTenant(tenant.clone()))?;
        let partitions = [key(&tenant)];
        let copying = backups.copy(
            &self.blobs,
            self.coordinator.as_ref(),
            &owner,
            &partitions,
            &*claim,
        );
        let copied = claim
            .during(async { copying.await.map_err(claimed) })
            .await?
            .ok_or_else(|| NodeError::Empty(tenant.clone()))?;
        // Not the claim's to stop once begun: a complete record written is
        // a backup made, whatever happens to the claim after.
        let backup = backups.complete(copied, &*claim).await.map_err(claimed)?;
        if !self.unchanged(&owner, &backup).await? {
            backups.forget(&backup).await?;
            tracing::info!(%tenant, backup = %backup.id, "a backup of a tenant removed meanwhile was forgotten");
            return Err(NodeError::UnknownTenant(tenant));
        }
        self.made_sure_of(backups, claim, &backup).await?;
        let pruning = backups.prune(&owner, self.settings.backups_kept, &*claim);
        match claim.during(async { Ok(pruning.await?) }).await {
            Ok(pruned) => tracing::info!(%tenant, backup = %backup.id, pruned, "backed up"),
            Err(NodeError::ClaimLost(_) | NodeError::Storage(StorageError::Unheld(_))) => {
                tracing::warn!(
                    %tenant,
                    backup = %backup.id,
                    "backed up, but the claim was lost while pruning, so pruning stopped; the \
                     backup is noted as made only if the claim was still this node's when let go"
                )
            }
            Err(error @ NodeError::Storage(StorageError::UnusableRecords { .. })) => {
                tracing::error!(%tenant, backup = %backup.id, %error, "backed up; pruning is refused")
            }
            Err(error) => {
                tracing::warn!(%tenant, backup = %backup.id, %error, "backed up; pruning failed")
            }
        }
        Ok(backup)
    }

    /// Look again, under `claim`, that every file `backup` lists is there,
    /// copying any that is not from the tenant's store: a node that stalled
    /// past its claim may have deleted one this backup reuses. A backup
    /// that cannot be made whole is forgotten, and the run fails. One that
    /// cannot be looked at again, the claim lost or the store failing, is
    /// kept: it was whole when its record was written.
    async fn made_sure_of(&self, backups: &Backups, claim: &Claim, backup: &Backup) -> Result<()> {
        match backups.mended(&self.blobs, backup, claim).await {
            Ok(lacking) if lacking.is_empty() => Ok(()),
            Ok(lacking) => {
                tracing::error!(
                    tenant = %backup.tenant,
                    backup = %backup.id,
                    ?lacking,
                    "a backup lacks files the tenant no longer has; it is forgotten"
                );
                backups.forget(backup).await?;
                Err(StorageError::Missing {
                    key: lacking.join(", "),
                }
                .into())
            }
            Err(error) => {
                tracing::warn!(tenant = %backup.tenant, backup = %backup.id, %error, "a backup could not be looked at again");
                Ok(())
            }
        }
    }

    /// Whether the tenant is still `owner`, and every partition `backup`
    /// took still the one it took.
    async fn unchanged(&self, owner: &Owner, backup: &Backup) -> Result<bool> {
        let known = self.known(&owner.tenant).await?;
        if known.map(|(_, now)| now).as_ref() != Some(owner) {
            return Ok(false);
        }
        for taken in &backup.partitions {
            let now = self
                .coordinator
                .committed(&taken.key(&owner.tenant))
                .await?;
            if now.is_none_or(|now| now.incarnation != taken.incarnation) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Which backup a restore was asked to be from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    /// One of the tenant's own.
    ItsOwn,
    /// One kept under the tenant's name, whose tenant it is to become, with
    /// the keys the backup kept or without.
    ToAdopt { with_keys: bool },
}

/// A store each request to which is given up, as failed, after `within`;
/// a listing, once any one answer of it has taken that long, however many
/// it takes; and a file read or put a part at a time, once the store has
/// taken that long over any one part handed on, however many there are. A
/// request given up may still land: the store had it already.
#[derive(Debug)]
pub(crate) struct Within {
    pub(crate) inner: Arc<dyn Blobs>,
    pub(crate) within: Duration,
}

impl Within {
    async fn bounded<T>(
        &self,
        key: &str,
        request: impl Future<Output = vellev_storage::Result<T>>,
    ) -> vellev_storage::Result<T> {
        tokio::time::timeout(self.within, request)
            .await
            .map_err(|_| unanswered(key, self.within))?
    }
}

/// Word that a store putting a file a part at a time waits for the next
/// part to come, and, once this goes, of when it stopped waiting: with the
/// part, or because it no longer asked for one.
struct Awaited<'a>(&'a watch::Sender<Option<tokio::time::Instant>>);

impl<'a> Awaited<'a> {
    fn begun(left: &'a watch::Sender<Option<tokio::time::Instant>>) -> Self {
        left.send_replace(None);
        Self(left)
    }
}

impl Drop for Awaited<'_> {
    fn drop(&mut self) {
        self.0.send_replace(Some(tokio::time::Instant::now()));
    }
}

#[async_trait::async_trait]
impl Blobs for Within {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.bounded(key, self.inner.put(key, bytes)).await
    }
    /// A create given up may have landed, as a put given up may.
    async fn put_new(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<Created> {
        self.bounded(key, self.inner.put_new(key, bytes)).await
    }
    /// The store is given `within` each time it is left to itself: from
    /// when it is asked until it asks for the first part, from each part
    /// it is handed until it asks for the next, and from the end of them
    /// until it answers. So a file is given as long as its parts take,
    /// however many they are, and a store that stops is given up on. The
    /// bound is on each such time, not on each request the store makes
    /// in it: a store that starts an upload, sends a part and completes
    /// it has two of them in the first and in the last. How long a part
    /// takes to come is not counted against it: whatever the parts are
    /// read from gives that up itself.
    async fn put_parts(
        &self,
        key: &str,
        parts: BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        // Since when the store has been left to itself; `None` while it
        // waits for a part to come.
        let (left, mut leaving) = watch::channel(Some(tokio::time::Instant::now()));
        let left = &left;
        let handed = stream::unfold(parts, move |mut parts| async move {
            let _awaited = Awaited::begun(left);
            let part = parts.next().await;
            part.map(|part| (part, parts))
        });
        let put = self.inner.put_parts(key, handed.boxed());
        tokio::pin!(put);
        loop {
            let since = *leaving.borrow_and_update();
            let overdue = async {
                match since {
                    Some(since) => tokio::time::sleep_until(since + self.within).await,
                    None => std::future::pending().await,
                }
            };
            // A put that has answered is taken as answered, though its
            // bound ran out at the same moment: given up, a file that
            // went up whole would be taken for one that did not.
            tokio::select! {
                biased;
                put = &mut put => return put,
                _ = leaving.changed() => {}
                () = overdue => return Err(unanswered(key, self.within)),
            }
        }
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.bounded(key, self.inner.get(key)).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list_within(prefix, self.within).await
    }
    async fn list_within(
        &self,
        prefix: &str,
        each: Duration,
    ) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list_within(prefix, each.min(self.within)).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.bounded(key, self.inner.delete(key)).await
    }
    /// As the store gives the file, each part given up once it has taken
    /// `within` to come, after which no more of it is asked for.
    fn parts<'a>(&'a self, key: &'a str) -> BoxStream<'a, vellev_storage::Result<Bytes>> {
        let parts = Some(self.inner.parts(key));
        stream::unfold(parts, move |parts| async move {
            let mut parts = parts?;
            match tokio::time::timeout(self.within, parts.next()).await {
                Ok(part) => part.map(|part| (part, Some(parts))),
                Err(_) => Some((Err(unanswered(key, self.within)), None)),
            }
        })
        .boxed()
    }
}

/// A store that notes, before it is asked for each, that a delete was sent
/// through it under a claim: sent, whatever becomes of it, answered or
/// given up or its future dropped, for the store may have it.
#[derive(Debug)]
struct Deleting {
    inner: Arc<dyn Blobs>,
    sent: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl Blobs for Deleting {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn put_new(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<Created> {
        self.inner.put_new(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        self.inner.put_parts(key, parts).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn list_within(
        &self,
        prefix: &str,
        each: Duration,
    ) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list_within(prefix, each).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.sent.store(true, Ordering::SeqCst);
        self.inner.delete(key).await
    }
}

/// What a claim is the right to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Right {
    /// Back a tenant up, or restore it from a backup.
    BackUp,
    /// Purge a tenant removed.
    Purge,
}

/// What a claim is, whether held or not: which claim, to do what, on which
/// tenant, through which coordinator, how long it is held at a time, how
/// long each request about it may take, and the clock it is measured on.
#[derive(Debug, Clone)]
pub(crate) struct Claimed {
    pub(crate) right: Right,
    pub(crate) coordinator: Arc<dyn Coordinator>,
    pub(crate) clock: Arc<dyn ClaimClock>,
    pub(crate) tenant: TenantId,
    pub(crate) id: Uuid,
    pub(crate) hold: Duration,
    pub(crate) within: Duration,
    /// Whether a delete was sent under the claim, through a store that
    /// says so ([`Claim::deleting_through`]).
    pub(crate) deleted: Arc<AtomicBool>,
}

impl Claimed {
    /// Until when the claim, asked for or renewed at `asked` on its clock,
    /// is taken for held: a third of a hold before it could run out.
    fn held_until(&self, asked: Duration) -> Duration {
        asked + self.hold - self.hold / 3
    }

    /// Let the claim go, each attempt given up after `within`, noting a
    /// backup made now if `made`, which is nothing to a claim to purge.
    /// Harmless if it was never taken; one that cannot be let go runs out
    /// by itself.
    async fn released(&self, made: bool) {
        let (tenant, id) = (&self.tenant, self.id);
        let release = || async move {
            match self.right {
                Right::BackUp => self.coordinator.release_backup(tenant, id, made).await,
                Right::Purge => self.coordinator.release_purge(tenant, id).await,
            }
        };
        if let Err(error) = retried(self.within, release).await {
            tracing::warn!(%tenant, %error, "a claim was left to run out");
        }
    }

    /// The claim, asked for at `asked` on its clock and taken, renewed from
    /// now on.
    pub(crate) fn renewed(self, asked: Duration) -> Claim {
        let deadline = Arc::new(watch::Sender::new(self.held_until(asked)));
        let renewing = tokio::spawn(renewing(self.clone(), asked, deadline.clone()));
        Claim {
            of: self,
            deadline,
            renewing,
            let_go: false,
        }
    }

    /// Renew the claim, an attempt given up after `within`. Returns whether
    /// it is still held.
    async fn renewal(&self) -> std::result::Result<bool, CoordinatorError> {
        let (coordinator, tenant) = (&self.coordinator, &self.tenant);
        match self.right {
            Right::BackUp => {
                let renewing = || coordinator.renew_backup(tenant, self.id, self.hold);
                retried(self.within, renewing).await
            }
            Right::Purge => {
                let renewing = || coordinator.renew_purge(tenant, self.id, self.hold);
                retried(self.within, renewing).await
            }
        }
    }
}

/// A claim held, on backing a tenant up or restoring it or on purging it,
/// renewed on a task of its own until it is let go or dropped.
///
/// What says whether it is held is a deadline on its clock, which every
/// check reads against the clock: a node that stalls or sleeps, its renewal
/// with it, and wakes past the deadline finds the claim lost before it does
/// anything more, whichever of its tasks runs first.
pub(crate) struct Claim {
    pub(crate) of: Claimed,
    /// Until when, on its clock, the claim is taken for held. Moved on by
    /// each renewal, and back to now once the claim is known to be lost.
    deadline: Arc<watch::Sender<Duration>>,
    renewing: JoinHandle<()>,
    /// Whether it has been let go.
    let_go: bool,
}

/// Move `deadline` on to `until`, if that is later.
fn extended(deadline: &watch::Sender<Duration>, until: Duration) {
    deadline.send_if_modified(|deadline| {
        let later = until > *deadline;
        if later {
            *deadline = until;
        }
        later
    });
}

/// Move `deadline` back to `now`, if it is later: the claim is lost.
fn lost(deadline: &watch::Sender<Duration>, now: Duration) {
    deadline.send_if_modified(|deadline| {
        let later = *deadline > now;
        if later {
            *deadline = now;
        }
        later
    });
}

impl Claim {
    /// Whether the claim is still taken for held, by its clock now.
    fn still(&self) -> Result<()> {
        if self.of.clock.now() >= *self.deadline.borrow() {
            return Err(NodeError::ClaimLost(self.of.tenant.clone()));
        }
        Ok(())
    }

    /// Once the claim is no longer taken for held, as far as the runtime's
    /// timers tell: they say when to look at the claim's clock again, and
    /// stop with the machine, so this may come late to a machine woken
    /// from sleep. What keeps such a machine from going on is not this but
    /// [`Self::still`], which reads the claim's clock, and which every step
    /// of the work asks before it writes or deletes anything.
    async fn lost(&self) {
        let mut deadline = self.deadline.subscribe();
        loop {
            let until = *deadline.borrow_and_update();
            let now = self.of.clock.now();
            if now >= until {
                return;
            }
            // The sender lives as long as this claim, so `changed` only
            // ever says the deadline moved.
            tokio::select! {
                _ = tokio::time::sleep(until - now) => {}
                _ = deadline.changed() => {}
            }
        }
    }

    /// `work`, stopped where it is once the claim is lost, and not taken
    /// up again, at any point it waited, once the claim's clock says so;
    /// one lost just as the work is done counts as lost.
    pub(crate) async fn during<T>(&self, work: impl Future<Output = Result<T>>) -> Result<T> {
        self.still()?;
        // The claim's clock is asked each time the work is taken up again,
        // whatever woke it: `lost` below looks at the clock only when its
        // own timer fires or the deadline moves, and work woken by a timer
        // of its own, as a request waiting out the store client's backoff
        // between two tries is on a machine just woken from sleep, would
        // go on and send with no clock asked.
        let mut work = std::pin::pin!(work);
        let asked = std::future::poll_fn(|context| {
            if self.still().is_err() {
                return std::task::Poll::Ready(Err(NodeError::ClaimLost(self.of.tenant.clone())));
            }
            work.as_mut().poll(context)
        });
        tokio::select! {
            biased;
            () = self.lost() => Err(NodeError::ClaimLost(self.of.tenant.clone())),
            done = asked => done,
        }
    }

    /// Stop renewing the claim and let it go, noting a backup made now if
    /// `made`. The release runs on a task of its own, as a dropped claim's
    /// does, so that whoever stops waiting here does not leave the claim
    /// to run out.
    pub(crate) async fn let_go(mut self, made: bool) {
        self.renewing.abort();
        self.let_go = true;
        let of = self.of.clone();
        let releasing = tokio::spawn(async move { of.released(made).await });
        if releasing.await.is_err() {
            tracing::warn!(tenant = %self.of.tenant, "a claim's release did not run to its end");
        }
    }

    /// Stop renewing a claim the coordinator has ended already, as it has
    /// a claim to purge once the purge is noted done: there is nothing to
    /// let go.
    pub(crate) fn ended(mut self) {
        self.renewing.abort();
        self.let_go = true;
    }

    /// `store`, noting on the claim each delete sent through it: so that
    /// the claim, dropped, knows whether one may still be on its way.
    pub(crate) fn deleting_through(&self, store: Arc<dyn Blobs>) -> Arc<dyn Blobs> {
        Arc::new(Deleting {
            inner: store,
            sent: self.of.deleted.clone(),
        })
    }

    /// Whether the claim, dropped now, is to be left to run out rather
    /// than let go: a claim to purge under which a delete was sent. The
    /// purge it was dropped from did not end well, or it would have been
    /// let go or ended, so the fate of its last deletes is not known: one
    /// given up unanswered, or cut short with the work, may yet land. It
    /// lands within what a request to the store may take, and was sent
    /// while the claim was taken for held, a third of a hold before it
    /// could run out; a third of a hold is at least what a request may
    /// take ([`RequestBounds::claim_problem`]). So by when the claim has
    /// run out, and no sooner, nobody purges the tenant or makes it again
    /// under a delete still to land. Let go at once, the next pass would
    /// purge it, the name be taken, and the delete land on the new tenant:
    /// on the very keys, if it took the old one's lineage up from a
    /// backup.
    fn left_to_run_out(&self) -> bool {
        self.of.right == Right::Purge && self.of.deleted.load(Ordering::SeqCst)
    }
}

/// A claim dropped without being let go, as one is when the task holding
/// it is stopped or the work under it fails, is let go on a task of its
/// own, which runs to its end whatever becomes of the one that held the
/// claim: so a claim is not left to run out because the work under it was
/// cut short. One that cannot be let go, the coordinator not answering,
/// runs out by itself; and one to purge under which a delete was sent is
/// left to ([`Claim::left_to_run_out`]).
impl Drop for Claim {
    fn drop(&mut self) {
        self.renewing.abort();
        if self.let_go || self.left_to_run_out() {
            return;
        }
        let of = self.of.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move { of.released(false).await });
        }
    }
}

#[async_trait::async_trait]
impl Held for Claim {
    /// Renew the claim now, waiting for the answer no longer than it is
    /// held, then say whether it still is, by its clock after the answer.
    async fn held(&self) -> vellev_storage::Result<bool> {
        let asked = self.of.clock.now();
        let until = *self.deadline.borrow();
        let renewal = tokio::time::timeout(until.saturating_sub(asked), self.of.renewal());
        match renewal.await {
            Ok(Ok(true)) => extended(&self.deadline, self.of.held_until(asked)),
            Ok(Ok(false)) => lost(&self.deadline, self.of.clock.now()),
            Ok(Err(error)) => return Err(error.into()),
            Err(_) => {}
        }
        Ok(self.still().is_ok())
    }

    fn still(&self) -> bool {
        Claim::still(self).is_ok()
    }
}

/// Renew `claim`, first asked for at `asked` on its clock, every third of
/// its hold, moving `deadline` on with each renewal, until it may no longer
/// be taken for held: another holds it, or the tenant went, or `deadline`
/// passed. Then move `deadline` back to now. A renewal is waited for no
/// longer than `deadline`.
async fn renewing(claim: Claimed, asked: Duration, deadline: Arc<watch::Sender<Duration>>) {
    let every = claim.hold / 3;
    let mut next = asked + every;
    loop {
        let (now, until) = (claim.clock.now(), *deadline.borrow());
        tokio::time::sleep(next.min(until).saturating_sub(now)).await;
        let (now, until) = (claim.clock.now(), *deadline.borrow());
        if now >= until {
            tracing::warn!(tenant = %claim.tenant, "a backup claim could not be renewed in time");
            break;
        }
        if now < next {
            continue;
        }
        match tokio::time::timeout((until - now).min(every), claim.renewal()).await {
            Ok(Ok(true)) => {
                extended(&deadline, claim.held_until(now));
                next = now + every;
            }
            Ok(Ok(false)) => {
                tracing::warn!(tenant = %claim.tenant, "a backup claim is no longer held");
                break;
            }
            Ok(Err(error)) if !error.is_retryable() => {
                tracing::warn!(tenant = %claim.tenant, %error, "renewing a backup claim was refused");
                break;
            }
            Ok(Err(error)) => {
                tracing::warn!(tenant = %claim.tenant, %error, "renewing a backup claim failed");
                next = now + every / 4;
            }
            Err(_) => {
                tracing::warn!(tenant = %claim.tenant, "renewing a backup claim took too long");
                next = now + every / 4;
            }
        }
    }
    lost(&deadline, claim.clock.now());
}

/// `call` made, each attempt given up after `within`, and made again while
/// it fails in a way worth retrying, up to [`CLAIM_ATTEMPTS`] times.
async fn retried<T, F>(
    within: Duration,
    call: impl Fn() -> F,
) -> std::result::Result<T, CoordinatorError>
where
    F: Future<Output = std::result::Result<T, CoordinatorError>>,
{
    let mut attempt = 1;
    loop {
        let answered = tokio::time::timeout(within, call())
            .await
            .unwrap_or_else(|_| {
                Err(CoordinatorError::Backend(
                    format!("no answer within {within:?}").into(),
                ))
            });
        match answered {
            Err(error) if error.is_retryable() && attempt < CLAIM_ATTEMPTS => {
                tracing::info!(%error, "the coordinator failed to answer; asking again");
                tokio::time::sleep(CLAIM_RETRY_DELAY * attempt).await;
                attempt += 1;
            }
            answered => return answered,
        }
    }
}

/// `error`, from work under a claim, as the node says it: the store's
/// refusal to go on once the claim is lost is the claim lost.
pub(crate) fn claimed(error: StorageError) -> NodeError {
    match error {
        StorageError::Unheld(tenant) => NodeError::ClaimLost(tenant),
        other => other.into(),
    }
}

/// Run `work` to its end on a task of its own, whoever stops waiting.
async fn finished<T: Send + 'static>(
    work: impl Future<Output = Result<T>> + Send + 'static,
) -> Result<T> {
    tokio::spawn(work)
        .await
        .map_err(|error| StorageError::Background(error.to_string()))?
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures::TryStreamExt;
    use tokio::time::Instant;

    use super::*;

    const WITHIN: Duration = Duration::from_secs(10);
    /// Less than the bound, and more than it by the second time.
    const EACH: Duration = Duration::from_secs(6);

    /// A store of one file, read and put a part at a time and no other
    /// way, that takes `each` before it asks for a part put or hands on
    /// one read, the end of them included, and stops for good in place of
    /// the `silent`th time.
    #[derive(Debug, Default)]
    struct Paced {
        each: Duration,
        silent: Option<usize>,
        kept: Mutex<Vec<Bytes>>,
    }

    impl Paced {
        async fn paced(&self, time: usize) {
            if self.silent == Some(time) {
                std::future::pending::<()>().await;
            }
            tokio::time::sleep(self.each).await;
        }
    }

    /// The failure of anything asked of it but a file a part at a time.
    fn whole(key: &str) -> StorageError {
        StorageError::Missing {
            key: key.to_string(),
        }
    }

    #[async_trait::async_trait]
    impl Blobs for Paced {
        async fn put(&self, key: &str, _bytes: Bytes) -> vellev_storage::Result<()> {
            Err(whole(key))
        }
        /// Answered as created once `each` has passed, or never if silent
        /// the first time.
        async fn put_new(&self, _key: &str, bytes: Bytes) -> vellev_storage::Result<Created> {
            self.paced(0).await;
            *self.kept.lock().unwrap() = vec![bytes];
            Ok(Created::Written)
        }
        async fn put_parts(
            &self,
            _key: &str,
            mut parts: BoxStream<'_, vellev_storage::Result<Bytes>>,
        ) -> vellev_storage::Result<()> {
            let mut taken = Vec::new();
            loop {
                self.paced(taken.len()).await;
                match parts.try_next().await? {
                    Some(part) => taken.push(part),
                    None => break,
                }
            }
            // The end of them was the time after the last.
            self.paced(taken.len() + 1).await;
            *self.kept.lock().unwrap() = taken;
            Ok(())
        }
        async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
            Err(whole(key))
        }
        async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
            Err(whole(prefix))
        }
        async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
            Err(whole(key))
        }
        fn parts<'a>(&'a self, _key: &'a str) -> BoxStream<'a, vellev_storage::Result<Bytes>> {
            let kept = self.kept.lock().unwrap().clone();
            stream::unfold(0, move |time| {
                let part = kept.get(time).cloned();
                async move {
                    self.paced(time).await;
                    part.map(|part| (Ok(part), time + 1))
                }
            })
            .boxed()
        }
    }

    /// What `work` came to, failing the test rather than letting it hang
    /// if that took an hour, which on a paused clock means it never ends.
    async fn ended<T>(work: impl Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(60 * 60), work)
            .await
            .expect("done within its bound")
    }

    fn file() -> Vec<Bytes> {
        ["one", "two", "three"].map(Bytes::from).to_vec()
    }

    fn coming(file: Vec<Bytes>) -> BoxStream<'static, vellev_storage::Result<Bytes>> {
        stream::iter(file.into_iter().map(Ok)).boxed()
    }

    fn bounded(inner: Arc<Paced>) -> Within {
        let within = WITHIN;
        Within { inner, within }
    }

    fn unanswered_within(result: vellev_storage::Result<()>) {
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            unanswered("file", WITHIN).to_string(),
            "{error}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_file_put_a_part_at_a_time_through_a_bounded_store_is_given_its_bound_for_each_part()
    {
        let each = EACH;
        let inner = Arc::new(Paced {
            each,
            ..Paced::default()
        });
        let started = Instant::now();
        let store = bounded(inner.clone());
        ended(store.put_parts("file", coming(file())))
            .await
            .unwrap();
        // Longer than the bound in all, and no time it was left to itself.
        assert_eq!(started.elapsed(), EACH * 5);
        assert!(started.elapsed() > WITHIN);
        assert_eq!(*inner.kept.lock().unwrap(), file());
    }

    #[tokio::test(start_paused = true)]
    async fn a_store_that_stops_while_a_file_is_put_a_part_at_a_time_is_given_up_within_the_bound()
    {
        // Before it asks for the first part, between two, once it has
        // been told there are no more, and with them all, before it
        // answers.
        for silent in [0, 2, 3, 4] {
            let inner = Arc::new(Paced {
                each: EACH,
                silent: Some(silent),
                ..Paced::default()
            });
            let started = Instant::now();
            let store = bounded(inner.clone());
            unanswered_within(ended(store.put_parts("file", coming(file()))).await);
            let before = u32::try_from(silent).unwrap();
            assert_eq!(started.elapsed(), EACH * before + WITHIN, "{silent}");
            assert_eq!(*inner.kept.lock().unwrap(), Vec::<Bytes>::new());
        }
    }

    /// A store that asks for a part, stops waiting for it once `each` has
    /// passed, and never answers.
    #[derive(Debug)]
    struct Abandoning(Duration);

    #[async_trait::async_trait]
    impl Blobs for Abandoning {
        async fn put(&self, key: &str, _bytes: Bytes) -> vellev_storage::Result<()> {
            Err(whole(key))
        }
        async fn put_parts(
            &self,
            _key: &str,
            mut parts: BoxStream<'_, vellev_storage::Result<Bytes>>,
        ) -> vellev_storage::Result<()> {
            let _ = tokio::time::timeout(self.0, parts.next()).await;
            drop(parts);
            std::future::pending().await
        }
        async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
            Err(whole(key))
        }
        async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
            Err(whole(prefix))
        }
        async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
            Err(whole(key))
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_store_that_stops_waiting_for_a_part_and_never_answers_is_given_up_within_the_bound()
    {
        let store = Within {
            inner: Arc::new(Abandoning(EACH)),
            within: WITHIN,
        };
        let started = Instant::now();
        let never = stream::pending().boxed();
        unanswered_within(ended(store.put_parts("file", never)).await);
        assert_eq!(started.elapsed(), EACH + WITHIN);
    }

    fn in_memory() -> Arc<dyn Blobs> {
        let store = object_store::memory::InMemory::new();
        Arc::new(vellev_storage::blobs::ObjectBlobs::new(Arc::new(store)))
    }

    #[tokio::test]
    async fn a_file_is_created_only_once_through_a_bounded_store_as_through_the_store_itself() {
        let inner = in_memory();
        let store = Within {
            inner: inner.clone(),
            within: WITHIN,
        };
        let first = store.put_new("file", Bytes::from("A")).await.unwrap();
        assert_eq!(first, Created::Written);
        let second = store.put_new("file", Bytes::from("B")).await.unwrap();
        assert_eq!(second, Created::AlreadyThere);
        assert_eq!(inner.get("file").await.unwrap(), "A");
    }

    #[tokio::test(start_paused = true)]
    async fn a_create_through_a_bounded_store_is_given_up_within_the_bound() {
        let answering = Arc::new(Paced {
            each: EACH,
            ..Paced::default()
        });
        let started = Instant::now();
        let created = ended(bounded(answering.clone()).put_new("file", "A".into())).await;
        assert_eq!(created.unwrap(), Created::Written);
        assert_eq!(started.elapsed(), EACH);
        assert_eq!(*answering.kept.lock().unwrap(), vec![Bytes::from("A")]);

        let silent = Arc::new(Paced {
            each: EACH,
            silent: Some(0),
            ..Paced::default()
        });
        let started = Instant::now();
        let created = ended(bounded(silent).put_new("file", "A".into())).await;
        unanswered_within(created.map(|_| ()));
        assert_eq!(started.elapsed(), WITHIN);
    }

    #[tokio::test]
    async fn a_create_through_a_store_that_notes_deletes_is_made_and_is_no_delete() {
        let inner = in_memory();
        let sent = Arc::new(AtomicBool::new(false));
        let store = Deleting {
            inner: inner.clone(),
            sent: sent.clone(),
        };
        let first = store.put_new("file", Bytes::from("A")).await.unwrap();
        assert_eq!(first, Created::Written);
        let second = store.put_new("file", Bytes::from("B")).await.unwrap();
        assert_eq!(second, Created::AlreadyThere);
        assert_eq!(inner.get("file").await.unwrap(), "A");
        assert!(!sent.load(Ordering::SeqCst));
    }

    #[tokio::test(start_paused = true)]
    async fn how_long_a_part_takes_to_come_is_not_counted_against_the_store_it_is_put_to() {
        let inner = Arc::new(Paced::default());
        let slow = coming(file()).then(|part| async {
            tokio::time::sleep(WITHIN * 3).await;
            part
        });
        let started = Instant::now();
        let store = bounded(inner.clone());
        ended(store.put_parts("file", slow.boxed())).await.unwrap();
        assert_eq!(started.elapsed(), WITHIN * 9);
        assert_eq!(*inner.kept.lock().unwrap(), file());

        // Nor is a failure of theirs taken for the store's.
        let lost = stream::iter([Ok(Bytes::from("one")), Err(whole("from"))]);
        let failed = store.put_parts("file", lost.boxed()).await.unwrap_err();
        assert!(matches!(failed, StorageError::Missing { .. }), "{failed}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_file_is_read_a_part_at_a_time_through_a_bounded_store_each_part_within_the_bound() {
        let paced = |silent| {
            Arc::new(Paced {
                each: EACH,
                silent,
                kept: Mutex::new(file()),
            })
        };
        let started = Instant::now();
        let store = bounded(paced(None));
        let read: Vec<Bytes> = ended(store.parts("file").try_collect()).await.unwrap();
        assert_eq!(read, file(), "as the store parts it, never asked for whole");
        assert_eq!(started.elapsed(), EACH * 4);

        let (started, store) = (Instant::now(), bounded(paced(Some(1))));
        let mut parts = store.parts("file");
        assert_eq!(parts.try_next().await.unwrap(), Some(Bytes::from("one")));
        unanswered_within(ended(parts.try_next()).await.map(|_| ()));
        assert_eq!(started.elapsed(), EACH + WITHIN);
        assert!(
            ended(parts.next()).await.is_none(),
            "and no more is asked for"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_file_put_a_part_at_a_time_under_a_claim_is_put_and_no_delete_noted() {
        let inner = Arc::new(Paced::default());
        let deleting = Deleting {
            inner: inner.clone(),
            sent: Arc::default(),
        };
        deleting.put_parts("file", coming(file())).await.unwrap();
        assert_eq!(*inner.kept.lock().unwrap(), file());
        assert!(!deleting.sent.load(Ordering::SeqCst));
    }
}
