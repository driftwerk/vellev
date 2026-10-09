//! Copies of tenants in a second store, from which a tenant is put back as
//! it was when copied: in its own cluster, or in another that has lost it.
//!
//! A backup is of one tenant, taken by one cluster, and says both: the
//! tenant's lineage, which tells it from any other tenant of its name, and
//! the cluster's id. Its record lies under
//! `_backups/{tenant}/records/{lineage}/{cluster}/`, and the files it lists
//! under `_backups/{tenant}/data/{lineage}/{cluster}/`. So:
//!
//! - A tenant lists and restores from the backups of its own lineage,
//!   whichever cluster took them, and none of any other tenant of its name.
//! - A cluster writes, prunes and purges only under its own id. What
//!   another cluster took of the same tenant, as the one a cluster adopted
//!   a tenant from did, is never deleted by this one, nor are its files
//!   shared with it: each cluster copies every file it needs once, under
//!   its own id, and each backup lists the ones it needs. Files never
//!   change, so a backup costs what changed since the cluster's one before.
//! - Every record kept under a name lies under `records/`, apart from the
//!   files, so that finding them lists no file.
//! - Neither lies under `_backups/{tenant}/backups/` or
//!   `_backups/{tenant}/files/`, where the release before lineages kept
//!   what it took, so that a node of that release, pruning what lies there
//!   as its own, never lists, deletes or restores from what this one
//!   keeps, nor this one from what it kept.
//!
//! A record is written as pending before any file is copied, so that
//! pruning, which keeps every file any record beside it lists, leaves them
//! be, and then as complete once every file it lists is there. No tenant's
//! own files could lie under `_backups/`, so that even a store given as
//! both deletes none of them.
//!
//! A record is trusted only so far: one naming another tenant, lineage or
//! cluster than where it lies, or a file outside the partition it names,
//! is not restored from. Nor is one that cannot be read, or that says no
//! lineage or cluster, or that a newer release wrote in a form this one
//! does not know. While a complete record that cannot be used is there, or
//! a pending one young enough to be of a backup under way, no file kept
//! beside it is deleted, as it may list any of them; an older pending one
//! is forgotten as any other is.

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
use crate::coordinator::{
    ClusterId, Coordinator, Identity, Incarnation, Lineage, LogPosition, PartitionKey, TenantKey,
};
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
/// How many files are deleted, together, for each time the right to delete
/// them is renewed with the coordinator: each batch costs one renewal, and
/// a node that stalls just after one deletes at most this many more.
pub const DELETED_AT_ONCE: usize = 16;

/// The right to write and delete a tenant's backups: a file another backup
/// or restore needs may go only while nothing else may back the tenant up
/// or restore it. It is asked by the clock before each file is written or
/// deleted, and renewed with the coordinator before each batch of
/// [`DELETED_AT_ONCE`] deletes.
#[async_trait::async_trait]
pub trait Held: Send + Sync {
    /// Renew the right now, and say whether it is still held.
    async fn held(&self) -> Result<bool>;
    /// Whether the right is still held as of now, without asking anyone:
    /// by the clock, so that one that ran out while its holder stalled is
    /// not taken for held.
    fn still(&self) -> bool;
}

/// `Unheld` for `tenant` unless `held` still is.
fn still(held: &dyn Held, tenant: &TenantId) -> Result<()> {
    if held.still() {
        return Ok(());
    }
    Err(StorageError::Unheld(tenant.clone()))
}

/// A tenant as the backups a cluster takes of it know it: its name, which
/// tenant of that name it is, and which cluster this is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Owner {
    pub tenant: TenantId,
    pub lineage: Lineage,
    pub cluster: ClusterId,
}

impl Owner {
    pub fn new(tenant: &TenantId, identity: Identity, cluster: ClusterId) -> Self {
        Self {
            tenant: tenant.clone(),
            lineage: identity.lineage,
            cluster,
        }
    }

    /// Whether `backup` is of this tenant, whichever cluster took it.
    pub fn owns(&self, backup: &Backup) -> bool {
        backup.tenant == self.tenant && backup.lineage == self.lineage
    }

    /// Where this cluster keeps its backups of the tenant.
    fn home(&self) -> Home {
        Home {
            tenant: self.tenant.clone(),
            lineage: self.lineage,
            cluster: self.cluster,
        }
    }
}

/// The version of the records this release writes, and the newest it
/// reads. A field added to a record has a default, so that records
/// written before it still read, and releases before it read the record
/// as they did; a change those releases would misread, such as one to
/// where files are kept or what a list of them means, raises it.
pub const RECORD_VERSION: u32 = 1;

/// A tenant as it was when copied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Backup {
    /// The version of the record, the first for one written before
    /// records had one.
    #[serde(default = "first_version")]
    pub version: u32,
    pub id: Uuid,
    pub tenant: TenantId,
    /// Which tenant of that name it is of.
    pub lineage: Lineage,
    /// Which cluster took it.
    pub cluster: ClusterId,
    pub taken_at: DateTime<Utc>,
    pub partitions: Vec<Taken>,
    /// The tenant's keys, by the hashes of their secrets.
    #[serde(default)]
    pub keys: Vec<TenantKey>,
    /// How far back the tenant may be restored, if not the default.
    #[serde(default)]
    pub restore_window: Option<Duration>,
}

fn first_version() -> u32 {
    1
}

/// The version a record says it is, read before the rest of it, which a
/// newer release may have laid out otherwise.
#[derive(Deserialize)]
struct Versioned {
    #[serde(default = "first_version")]
    version: u32,
}

impl Backup {
    /// The files every partition it took is loaded from.
    pub fn files(&self) -> impl Iterator<Item = &String> {
        self.partitions.iter().flat_map(|taken| &taken.files)
    }

    /// Whether any partition it took had committed anything, which is what
    /// holding data means here: a tenant whose every object was deleted
    /// since has committed, and its backups hold data.
    pub fn holds_data(&self) -> bool {
        self.partitions.iter().any(|taken| taken.head.is_some())
    }

    /// Where the tenant's default partition was when it was taken.
    pub fn position(&self) -> LogPosition {
        self.partitions
            .iter()
            .find(|taken| taken.partition == PartitionNumber::DEFAULT)
            .map_or(LogPosition::default(), |taken| taken.position)
    }

    /// Where it and what is kept beside it lie.
    fn home(&self) -> Home {
        Home {
            tenant: self.tenant.clone(),
            lineage: self.lineage,
            cluster: self.cluster,
        }
    }
}

/// The files of a backup copied, under a pending record, that is a
/// backup once it is completed.
#[derive(Debug)]
pub struct Copied {
    backup: Backup,
}

impl Copied {
    /// The backup it is to be.
    pub fn backup(&self) -> &Backup {
        &self.backup
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

/// Where the backups one cluster took of one tenant lie.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Home {
    tenant: TenantId,
    lineage: Lineage,
    cluster: ClusterId,
}

impl Home {
    fn files(&self) -> String {
        let Self {
            tenant,
            lineage,
            cluster,
        } = self;
        format!("{}data/{lineage}/{cluster}/", named(tenant))
    }

    fn records(&self) -> String {
        format!(
            "{}{}/",
            of_lineage(&self.tenant, self.lineage),
            self.cluster
        )
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

    /// Copy `partitions` of `owner`, as committed now, out of `primary`:
    /// one that never committed as empty, so that a restore empties it
    /// too. `None`, copying nothing, if none has committed: a backup of
    /// nothing would be the newest, and pruning would go by it. What is
    /// copied becomes a backup once it is [completed]; until then a pending
    /// record keeps pruning off its files. Nothing else may back the tenant
    /// up or prune its backups meanwhile: `held` is asked before each file
    /// is written, and nothing more is if it is not.
    ///
    /// [completed]: Self::complete
    pub async fn copy(
        &self,
        primary: &Arc<dyn Blobs>,
        coordinator: &dyn Coordinator,
        owner: &Owner,
        partitions: &[PartitionKey],
        held: &dyn Held,
    ) -> Result<Option<Copied>> {
        let mut attempt = 1;
        loop {
            match self
                .attempt(primary, coordinator, owner, partitions, held)
                .await
            {
                Err(StorageError::Missing { key }) if attempt < ATTEMPTS => {
                    tracing::info!(tenant = %owner.tenant, key, "a file went while it was copied; starting again");
                    attempt += 1;
                }
                copied => return copied,
            }
        }
    }

    async fn attempt(
        &self,
        primary: &Arc<dyn Blobs>,
        coordinator: &dyn Coordinator,
        owner: &Owner,
        partitions: &[PartitionKey],
        held: &dyn Held,
    ) -> Result<Option<Copied>> {
        let Some(backup) = taken(primary, coordinator, owner, partitions).await? else {
            return Ok(None);
        };
        let pending = record(&backup, PENDING);
        still(held, &owner.tenant)?;
        self.store.put(&pending, encoded(&backup)?).await?;
        match self.copied(primary, &backup, held).await {
            Ok(()) => Ok(Some(Copied { backup })),
            Err(error) => {
                self.forget_record(&backup, PENDING).await;
                Err(error)
            }
        }
    }

    /// Make what was `copied` a backup, if `held` still is: write its
    /// complete record, and forget the pending one. Once the complete
    /// record is written the backup is made, whatever happens to `held`
    /// after. If it is not, both records go, the complete one too, as a
    /// write that failed or was given up on may have landed all the same,
    /// and pruning may take the files.
    pub async fn complete(&self, copied: Copied, held: &dyn Held) -> Result<Backup> {
        let Copied { backup } = copied;
        let completed = match still(held, &backup.tenant) {
            Ok(()) => {
                let record = encoded(&backup)?;
                self.store
                    .put(&self::record(&backup, COMPLETE), record)
                    .await
            }
            Err(unheld) => Err(unheld),
        };
        if completed.is_err() {
            self.forget_record(&backup, COMPLETE).await;
        }
        self.forget_record(&backup, PENDING).await;
        completed.map(|()| backup)
    }

    /// Forget the record of `backup` ending in `suffix`, saying so if it
    /// is left.
    async fn forget_record(&self, backup: &Backup, suffix: &str) {
        let key = record(backup, suffix);
        if let Err(error) = self.store.delete(&key).await {
            tracing::warn!(tenant = %backup.tenant, key, %error, "a backup record was left");
        }
    }

    /// Copy the files of `backup` not yet in the store, each while `held`
    /// still is, and make sure that every one it lists is there.
    async fn copied(
        &self,
        primary: &Arc<dyn Blobs>,
        backup: &Backup,
        held: &dyn Held,
    ) -> Result<()> {
        self.copy_missing(primary, backup, held).await?;
        match self.missing(backup).await?.into_iter().next() {
            Some(key) => Err(StorageError::Missing { key }),
            None => Ok(()),
        }
    }

    /// The keys in the store of the files `backup` lists that are not
    /// there.
    async fn missing(&self, backup: &Backup) -> Result<Vec<String>> {
        let prefix = &backup.home().files();
        let present = self.keys(prefix).await?;
        Ok(backup
            .files()
            .map(|key| format!("{prefix}{key}"))
            .filter(|copied| !present.contains(copied))
            .collect())
    }

    /// Copy from `primary` the files of `backup` not in the store, each
    /// while `held` still is. A file `primary` no longer has is left out.
    async fn copy_missing(
        &self,
        primary: &Arc<dyn Blobs>,
        backup: &Backup,
        held: &dyn Held,
    ) -> Result<()> {
        let prefix = &backup.home().files();
        let copies = self.missing(backup).await?.into_iter().map(|copied| {
            let key = copied[prefix.len()..].to_string();
            async move {
                let bytes = match primary.get(&key).await {
                    Err(StorageError::Missing { .. }) => return Ok(()),
                    bytes => bytes?,
                };
                // Asked once the file is read, which may take long, and
                // just before it is written.
                still(held, &backup.tenant)?;
                self.store.put(&copied, bytes).await
            }
        });
        stream::iter(copies)
            .buffer_unordered(AT_ONCE)
            .try_collect::<()>()
            .await
    }

    /// Put back, from `primary`, any file `backup` lists that the store
    /// lacks, as one a node that stalled past its claim deleted from under
    /// it would be, each while `held` still is. Returns the keys of those
    /// still lacking, which `primary` no longer has either.
    pub async fn mended(
        &self,
        primary: &Arc<dyn Blobs>,
        backup: &Backup,
        held: &dyn Held,
    ) -> Result<Vec<String>> {
        let lacking = self.missing(backup).await?;
        if lacking.is_empty() {
            return Ok(lacking);
        }
        tracing::error!(tenant = %backup.tenant, backup = %backup.id, ?lacking, "a backup lacks files; copying them again");
        self.copy_missing(primary, backup, held).await?;
        self.missing(backup).await
    }

    /// The complete backups of `owner`'s tenant, oldest first: those of its
    /// lineage, whichever cluster took them. Any whose record cannot be
    /// used is left out, and so is every backup of another tenant of its
    /// name.
    pub async fn list(&self, owner: &Owner) -> Result<Vec<Backup>> {
        let under = of_lineage(&owner.tenant, owner.lineage);
        self.complete_under(&under, &owner.tenant).await
    }

    /// The complete backup of `owner`'s tenant with this id, if there is
    /// one.
    pub async fn find(&self, owner: &Owner, id: Uuid) -> Result<Option<Backup>> {
        Ok(self
            .list(owner)
            .await?
            .into_iter()
            .find(|backup| backup.id == id))
    }

    /// Every complete backup kept under the name `tenant`, of whichever
    /// tenant of that name and taken by whichever cluster, oldest first:
    /// what a cluster that has lost a tenant may take it up again from.
    pub async fn of_any(&self, tenant: &TenantId) -> Result<Vec<Backup>> {
        let under = format!("{}records/", named(tenant));
        self.complete_under(&under, tenant).await
    }

    /// The complete backups of the name `tenant` whose records lie under
    /// `prefix`, oldest first. One is taken for what it says it is only
    /// where it lies where a backup of that tenant, lineage and cluster is
    /// written; any other, and any that cannot be read, is said and left
    /// out.
    async fn complete_under(&self, prefix: &str, tenant: &TenantId) -> Result<Vec<Backup>> {
        let listed = self.store.list(prefix).await?;
        let reads = listed
            .into_iter()
            .filter(|listed| listed.key.ends_with(COMPLETE))
            .map(|listed| read(self.store.as_ref(), listed.key));
        let read: Vec<Option<(String, std::result::Result<Backup, Unusable>)>> =
            stream::iter(reads)
                .buffer_unordered(AT_ONCE)
                .try_collect()
                .await?;
        let mut found = Vec::new();
        for (key, record) in read.into_iter().flatten() {
            match record.and_then(|backup| placed(&key, tenant, backup)) {
                Ok(backup) => found.push((key, backup)),
                Err(unusable) => {
                    tracing::error!(key, %unusable, "a backup record cannot be used")
                }
            }
        }
        found.sort_by(by_age);
        Ok(found.into_iter().map(|(_, backup)| backup).collect())
    }

    /// The records kept in `home` whose keys end in `ending`, but those
    /// that went meanwhile.
    async fn records(&self, home: &Home, ending: &str) -> Result<Records> {
        let listed = self.store.list(&home.records()).await?;
        let reads = listed
            .into_iter()
            .filter(|listed| listed.key.ends_with(ending))
            .map(|listed| read(self.store.as_ref(), listed.key));
        let read: Vec<Option<(String, std::result::Result<Backup, Unusable>)>> =
            stream::iter(reads)
                .buffer_unordered(AT_ONCE)
                .try_collect()
                .await?;
        let mut records = Records::default();
        for (key, record) in read.into_iter().flatten() {
            match record.and_then(|backup| belonging(home, backup)) {
                Ok(backup) => records.usable.push((key, backup)),
                Err(unusable) => {
                    tracing::error!(key, %unusable, "a backup record cannot be used");
                    records.unusable.push((key, unusable));
                }
            }
        }
        records.usable.sort_by(by_age);
        Ok(records)
    }

    /// Forget the complete backups this cluster took of `owner`'s tenant
    /// over `keep` ago, save the newest, and the newest that holds
    /// anything, which one of nothing, as a release before this one could
    /// take, never stands in for; and pending ones left by a backup that
    /// never finished, of those that can be used or not; then delete the
    /// files no record left lists. What another cluster took of the tenant
    /// is left alone, records and files. Returns how many records were
    /// forgotten. Refused, deleting nothing beside it, while a complete
    /// record, or a pending one that may be of a backup under way, cannot
    /// be used. Nothing may back the tenant up meanwhile: `held` is asked
    /// so just before anything is deleted, and nothing more is if it is
    /// not.
    pub async fn prune(&self, owner: &Owner, keep: Duration, held: &dyn Held) -> Result<usize> {
        let (home, now) = (owner.home(), Utc::now());
        let Records { usable, unusable } = self
            .records(&home, ".json")
            .await?
            .deletable(&owner.tenant, now)?;
        let complete = || usable.iter().filter(|(key, _)| key.ends_with(COMPLETE));
        let newest = |holding: fn(&Backup) -> bool| {
            let newest = complete().rev().find(|(_, backup)| holding(backup));
            newest.map(|(key, _)| key.clone())
        };
        let aged = Aged {
            keep,
            now,
            kept: [newest(|_| true), newest(Backup::holds_data)],
        };
        let (gone, kept): (Vec<_>, Vec<_>) = usable.iter().partition(|record| aged.out(record));
        let gone: Vec<String> = gone
            .into_iter()
            .map(|(key, _)| key.clone())
            .chain(unusable.into_iter().map(|(key, _)| key))
            .collect();
        self.delete(&owner.tenant, &gone, held).await?;
        self.sweep(&home, kept.iter().map(|(_, backup)| backup), held)
            .await?;
        Ok(gone.len())
    }

    /// Delete the files kept in `home` that none of `kept` lists, if
    /// `held` is.
    async fn sweep<'a>(
        &self,
        home: &Home,
        kept: impl Iterator<Item = &'a Backup>,
        held: &dyn Held,
    ) -> Result<()> {
        let prefix = home.files();
        let listed: HashSet<String> = kept
            .flat_map(Backup::files)
            .map(|key| format!("{prefix}{key}"))
            .collect();
        let unlisted: Vec<String> = self
            .keys(&prefix)
            .await?
            .difference(&listed)
            .cloned()
            .collect();
        self.delete(&home.tenant, &unlisted, held).await
    }

    /// Delete `keys` of `tenant`'s backups, [`DELETED_AT_ONCE`] at a time
    /// and together, each batch once `held` is renewed, and each key while
    /// it still is.
    async fn delete(&self, tenant: &TenantId, keys: &[String], held: &dyn Held) -> Result<()> {
        delete(self.store.as_ref(), tenant, keys, held).await
    }

    /// Copy the files of `backup` that `primary` lacks back into it, under
    /// the keys they had, each while `held` still is. Returns how many
    /// were copied.
    pub async fn fetch_back(
        &self,
        primary: &Arc<dyn Blobs>,
        backup: &Backup,
        held: &dyn Held,
    ) -> Result<usize> {
        let prefix = &backup.home().files();
        let mut present = HashSet::new();
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
            present.extend(
                primary
                    .list(&layout)
                    .await?
                    .into_iter()
                    .map(|listed| listed.key),
            );
        }
        let wanted: Vec<(String, String)> = backup
            .files()
            .filter(|key| !present.contains(*key))
            .map(|key| (format!("{prefix}{key}"), key.to_string()))
            .collect();
        let copied = wanted.len();
        let copies = wanted.into_iter().map(|(kept, key)| async move {
            still(held, &backup.tenant)?;
            let bytes = self.store.get(&kept).await?;
            // Asked again once the file is read, just before it is written.
            still(held, &backup.tenant)?;
            primary.put(&key, bytes).await
        });
        stream::iter(copies)
            .buffer_unordered(AT_ONCE)
            .try_collect::<()>()
            .await?;
        Ok(copied)
    }

    /// Forget `backup`, complete or pending, and leave its files, for
    /// pruning to delete once no backup lists them.
    pub async fn forget(&self, backup: &Backup) -> Result<()> {
        for key in [record(backup, COMPLETE), record(backup, PENDING)] {
            self.store.delete(&key).await?;
        }
        Ok(())
    }

    /// Whether this cluster keeps anything of `owner`'s tenant.
    pub async fn any_kept(&self, owner: &Owner) -> Result<bool> {
        Ok(!self.kept_of(owner).await?.is_empty())
    }

    /// Delete everything this cluster keeps of `owner`'s tenant, and
    /// nothing of any other tenant of its name, nor anything another
    /// cluster took of it: `held` is asked before each batch, and nothing
    /// more is deleted if it is not. Returns how many files went.
    pub async fn purge(&self, owner: &Owner, held: &dyn Held) -> Result<usize> {
        let kept = self.kept_of(owner).await?;
        self.delete(&owner.tenant, &kept, held).await?;
        Ok(kept.len())
    }

    /// The keys of everything this cluster keeps of `owner`'s tenant:
    /// records, usable or not, and files.
    async fn kept_of(&self, owner: &Owner) -> Result<Vec<String>> {
        let home = owner.home();
        let (records, files) = (home.records(), home.files());
        let (records, files) = futures::try_join!(self.keys(&records), self.keys(&files))?;
        Ok(records.into_iter().chain(files).collect())
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

/// Delete `keys` of `tenant` from `store`, [`DELETED_AT_ONCE`] at a time and
/// together, each batch once `held` is renewed, and each key while it still
/// is.
pub async fn delete(
    store: &dyn Blobs,
    tenant: &TenantId,
    keys: &[String],
    held: &dyn Held,
) -> Result<()> {
    for batch in keys.chunks(DELETED_AT_ONCE) {
        if !held.held().await? {
            return Err(StorageError::Unheld(tenant.clone()));
        }
        let deletes = batch.iter().map(|key| deleted(store, tenant, key, held));
        futures::future::try_join_all(deletes).await?;
    }
    Ok(())
}

/// Delete `key` of `tenant` from `store`, if `held` still is.
async fn deleted(store: &dyn Blobs, tenant: &TenantId, key: &str, held: &dyn Held) -> Result<()> {
    still(held, tenant)?;
    store.delete(key).await
}

/// Which of the records a cluster keeps of a tenant have aged out.
struct Aged {
    /// How long a complete backup is kept.
    keep: Duration,
    now: DateTime<Utc>,
    /// The keys of the records kept whatever their age.
    kept: [Option<String>; 2],
}

impl Aged {
    /// Whether the record is to be forgotten: a complete one taken longer
    /// ago than backups are kept, unless it is kept whatever its age; or a
    /// pending one left longer than any backup takes.
    fn out(&self, (key, backup): &(String, Backup)) -> bool {
        let before = |kept: Duration| {
            chrono::Duration::from_std(kept)
                .ok()
                .and_then(|kept| self.now.checked_sub_signed(kept))
                .is_some_and(|at| backup.taken_at < at)
        };
        if key.ends_with(PENDING) {
            before(PENDING_KEPT)
        } else {
            !self.kept.iter().flatten().any(|kept| kept == key) && before(self.keep)
        }
    }
}

/// What a backup of `partitions` as committed now is to hold, or nothing if
/// none of them has committed.
async fn taken(
    primary: &Arc<dyn Blobs>,
    coordinator: &dyn Coordinator,
    owner: &Owner,
    partitions: &[PartitionKey],
) -> Result<Option<Backup>> {
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
    if !taken.iter().any(|taken| taken.head.is_some()) {
        return Ok(None);
    }
    Ok(Some(Backup {
        version: RECORD_VERSION,
        id: Uuid::new_v4(),
        tenant: owner.tenant.clone(),
        lineage: owner.lineage,
        cluster: owner.cluster,
        taken_at: Utc::now(),
        partitions: taken,
        keys: coordinator.keys(&owner.tenant).await?,
        restore_window: coordinator.restore_window(&owner.tenant).await?,
    }))
}

/// The records kept in one place.
#[derive(Debug, Default)]
struct Records {
    /// Those this release reads that belong there, by key, oldest first.
    usable: Vec<(String, Backup)>,
    /// The others, by key, with why each cannot be used.
    unusable: Vec<(String, Unusable)>,
}

impl Records {
    /// The records, if files may be deleted around them as of `now`: when
    /// those that cannot be used are all pending ones older than
    /// [`PENDING_KEPT`], which no backup still under way left, and which
    /// may be forgotten as any other. Refused while any other record
    /// cannot be used: what it lists is not known, so no file may be
    /// deleted around it.
    fn deletable(self, tenant: &TenantId, now: DateTime<Utc>) -> Result<Records> {
        let (stale, blocking): (Vec<_>, Vec<_>) = self
            .unusable
            .into_iter()
            .partition(|(key, _)| key.ends_with(PENDING) && written_before(key, now, PENDING_KEPT));
        if blocking.is_empty() {
            return Ok(Records {
                usable: self.usable,
                unusable: stale,
            });
        }
        Err(StorageError::UnusableRecords {
            tenant: tenant.clone(),
            records: blocking
                .into_iter()
                .map(|(key, unusable)| format!("{key}: {unusable}"))
                .collect(),
        })
    }
}

/// Why a backup record cannot be used.
#[derive(Debug, thiserror::Error)]
enum Unusable {
    #[error("it cannot be read: {0}")]
    Unreadable(serde_json::Error),
    #[error("it is of version {0}, and this release reads up to {RECORD_VERSION}")]
    Version(u32),
    #[error("it names tenant {0}")]
    Tenant(TenantId),
    #[error("it is of another tenant of the name, or by another cluster, than where it lies says")]
    Misplaced,
}

/// `backup`, read from a record in `home`, if it belongs there.
fn belonging(home: &Home, backup: Backup) -> std::result::Result<Backup, Unusable> {
    if backup.tenant != home.tenant {
        Err(Unusable::Tenant(backup.tenant))
    } else if backup.home() != *home {
        Err(Unusable::Misplaced)
    } else {
        Ok(backup)
    }
}

/// `backup`, read from the complete record under `key` among those of the
/// name `tenant`, if that is where a backup saying what it says is written.
fn placed(key: &str, tenant: &TenantId, backup: Backup) -> std::result::Result<Backup, Unusable> {
    if &backup.tenant != tenant {
        Err(Unusable::Tenant(backup.tenant))
    } else if key != record(&backup, COMPLETE) {
        Err(Unusable::Misplaced)
    } else {
        Ok(backup)
    }
}

/// Oldest first, and by key among those taken at once.
fn by_age(
    (key, backup): &(String, Backup),
    (other, than): &(String, Backup),
) -> std::cmp::Ordering {
    (backup.taken_at, key).cmp(&(than.taken_at, other))
}

/// Whether the record under `key` was written longer than `kept` before
/// `now`, by the time its key starts with. One whose key says no time is
/// taken for written now.
fn written_before(key: &str, now: DateTime<Utc>, kept: Duration) -> bool {
    let written = key
        .rsplit('/')
        .next()
        .and_then(|name| name.split('-').next())
        .and_then(|millis| millis.parse::<i64>().ok())
        .and_then(DateTime::from_timestamp_millis);
    let before = chrono::Duration::from_std(kept)
        .ok()
        .and_then(|kept| now.checked_sub_signed(kept));
    written
        .zip(before)
        .is_some_and(|(written, before)| written < before)
}

/// The record under `key`, unless it went meanwhile: a backup this release
/// reads, or why it is not.
async fn read(
    store: &dyn Blobs,
    key: String,
) -> Result<Option<(String, std::result::Result<Backup, Unusable>)>> {
    match store.get(&key).await {
        Ok(bytes) => Ok(Some((key, usable(&bytes)))),
        Err(StorageError::Missing { .. }) => Ok(None),
        Err(error) => Err(error),
    }
}

/// The backup `record` is, or why it is not one this release can read.
fn usable(record: &[u8]) -> std::result::Result<Backup, Unusable> {
    let Versioned { version } = serde_json::from_slice(record).map_err(Unusable::Unreadable)?;
    if !(first_version()..=RECORD_VERSION).contains(&version) {
        return Err(Unusable::Version(version));
    }
    serde_json::from_slice(record).map_err(Unusable::Unreadable)
}

/// Where everything kept under the name `tenant` lies, of whichever tenant
/// of that name.
fn named(tenant: &TenantId) -> String {
    format!("_backups/{tenant}/")
}

/// Where the records of the backups of the tenant of `lineage` named
/// `tenant` lie, whichever cluster took them.
fn of_lineage(tenant: &TenantId, lineage: Lineage) -> String {
    format!("{}records/{lineage}/", named(tenant))
}

fn record(backup: &Backup, suffix: &str) -> String {
    format!(
        "{}{:020}-{}{suffix}",
        backup.home().records(),
        backup.taken_at.timestamp_millis(),
        backup.id.simple()
    )
}

fn encoded(backup: &Backup) -> Result<Bytes> {
    serde_json::to_vec_pretty(backup)
        .map(Bytes::from)
        .map_err(|error| StorageError::Encode(error.to_string()))
}
