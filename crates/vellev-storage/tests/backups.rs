//! A tenant copied to a second store, and put back from it.

mod common;

use common::*;
use vellev_storage::backups::{Backup, Backups, Held, Owner, DELETED_AT_ONCE, RECORD_VERSION};
use vellev_storage::blobs::unanswered;
use vellev_storage::coordinator::{ClusterId, Lineage};

/// The right to delete backup files, held or not, as a claim would say.
struct Claimed(bool);

#[async_trait]
impl Held for Claimed {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.0)
    }
    fn still(&self) -> bool {
        self.0
    }
}

fn memory() -> Arc<dyn Blobs> {
    Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))
}

fn ns(state: &State, objects: &[u8]) -> Vec<Option<i64>> {
    objects
        .iter()
        .map(|object| n_of(state, "notes", *object))
        .collect()
}

/// Which tenant `cluster`'s is now, as the backups the cluster takes of it
/// know it.
async fn owner(cluster: &Cluster) -> Owner {
    let identity = cluster.coordinator.identity(&cluster.key.tenant).await;
    let id = cluster.coordinator.cluster().await.unwrap();
    Owner::new(&cluster.key.tenant, identity.unwrap().unwrap(), id)
}

/// Where the records of the backups `owner`'s cluster took of it lie.
fn records(owner: &Owner) -> String {
    let Owner {
        tenant,
        lineage,
        cluster,
    } = owner;
    format!("_backups/{tenant}/records/{lineage}/{cluster}/")
}

/// Where the files of the backups `owner`'s cluster took of it lie.
fn files(owner: &Owner) -> String {
    let Owner {
        tenant,
        lineage,
        cluster,
    } = owner;
    format!("_backups/{tenant}/data/{lineage}/{cluster}/")
}

/// Whether `key` is of what `owner`'s cluster keeps of it.
fn kept_by(owner: &Owner, key: &str) -> bool {
    key.starts_with(&records(owner)) || key.starts_with(&files(owner))
}

/// A backup of `cluster`'s partition, copied and completed while `held`,
/// or why not.
async fn taken_while(
    backups: &Backups,
    cluster: &Cluster,
    held: &dyn Held,
) -> vellev_storage::Result<Option<Backup>> {
    let copied = backups
        .copy(
            &cluster.blobs,
            cluster.coordinator.as_ref(),
            &owner(cluster).await,
            std::slice::from_ref(&cluster.key),
            held,
        )
        .await?;
    match copied {
        Some(copied) => backups.complete(copied, held).await.map(Some),
        None => Ok(None),
    }
}

async fn taken(backups: &Backups, cluster: &Cluster) -> Option<Backup> {
    taken_while(backups, cluster, &Claimed(true)).await.unwrap()
}

async fn take(backups: &Backups, cluster: &Cluster) -> Backup {
    taken(backups, cluster)
        .await
        .expect("a partition that committed is backed up")
}

#[tokio::test]
async fn a_partition_that_never_committed_is_not_backed_up() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    assert_eq!(taken(&backups, &cluster).await, None);
    assert!(keys(&store, "").await.is_empty(), "nothing is written");
}

#[tokio::test]
async fn pruning_keeps_the_newest_backup_holding_data_whatever_is_newer() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let holding = take(&backups, &cluster).await;
    let owner = owner(&cluster).await;
    let (records, files) = (records(&owner), files(&owner));
    // A backup of nothing, newer, as a release before this one could take.
    let empty = Backup {
        id: uuid::Uuid::new_v4(),
        taken_at: holding.taken_at + chrono::Duration::hours(25),
        partitions: vec![vellev_storage::backups::Taken {
            head: None,
            files: Vec::new(),
            ..holding.partitions[0].clone()
        }],
        ..holding.clone()
    };
    let record = format!(
        "{records}{:020}-{}.backup.json",
        empty.taken_at.timestamp_millis(),
        empty.id.simple()
    );
    store
        .put(&record, Bytes::from(serde_json::to_vec(&empty).unwrap()))
        .await
        .unwrap();
    let copied = keys(&store, &files).await;

    assert_eq!(
        backups
            .prune(&owner, Duration::ZERO, &Claimed(true))
            .await
            .unwrap(),
        0
    );
    assert_eq!(backups.list(&owner).await.unwrap(), [holding, empty]);
    assert_eq!(
        keys(&store, &files).await,
        copied,
        "the files of the newest backup holding data are kept"
    );
}

async fn keys(store: &Arc<dyn Blobs>, prefix: &str) -> BTreeSet<String> {
    store
        .list(prefix)
        .await
        .unwrap()
        .into_iter()
        .map(|listed| listed.key)
        .collect()
}

/// Notes 1 and 2 in a segment, note 1 changed in memory since.
async fn written(cluster: &Cluster) -> Writer {
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 1, 1),
            upsert("notes", 2, 2),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 1, 10)]).await.unwrap();
    writer
}

#[tokio::test]
async fn a_backup_puts_a_tenant_back_in_a_store_that_lost_it() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let backup = take(&backups, &cluster).await;
    assert_eq!(
        backups.list(&owner(&cluster).await).await.unwrap(),
        std::slice::from_ref(&backup)
    );

    let lost = Cluster {
        blobs: memory(),
        ..Cluster::new().await
    };
    let copied = backups
        .fetch_back(&lost.blobs, &backup, &Claimed(true))
        .await
        .unwrap();
    assert_eq!(copied, backup.partitions[0].files.len());
    let taken = &backup.partitions[0];
    let committed = Committed {
        log_position: taken.position,
        manifest: taken.head.clone().map(ManifestRef),
        ..lost.committed().await
    };
    let follower = Follower::open(lost.key.clone(), lost.blobs.clone(), &committed)
        .await
        .unwrap();
    assert_eq!(
        follower.partition().read(|state| ns(state, &[1, 2])),
        [Some(10), Some(2)]
    );
    assert_eq!(
        backups
            .fetch_back(&lost.blobs, &backup, &Claimed(true))
            .await
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn each_file_is_copied_once_whatever_the_backups_that_need_it() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    let writer = written(&cluster).await;
    let first = take(&backups, &cluster).await;
    let files = files(&owner(&cluster).await);
    let after_first = keys(&store, &files).await;
    assert_eq!(after_first.len(), first.partitions[0].files.len());

    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    let second = take(&backups, &cluster).await;
    let after_second = keys(&store, &files).await;
    assert_eq!(
        after_second.len() - after_first.len(),
        1,
        "only the new log is copied"
    );
    assert!(second.partitions[0]
        .files
        .iter()
        .all(|key| after_second.contains(&format!("{files}{key}"))));
}

#[tokio::test]
async fn pruning_keeps_the_newest_and_every_file_a_kept_backup_lists() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    let writer = written(&cluster).await;
    let first = take(&backups, &cluster).await;
    writer.flush().await.unwrap();
    writer.merge().await.unwrap();
    writer.clean(Duration::ZERO).await.unwrap();
    let second = take(&backups, &cluster).await;
    let owner = owner(&cluster).await;
    let files = files(&owner);

    assert_eq!(
        backups
            .prune(&owner, Duration::from_secs(3_600), &Claimed(true))
            .await
            .unwrap(),
        0
    );
    assert_eq!(backups.list(&owner).await.unwrap().len(), 2);
    assert_eq!(
        backups
            .prune(&owner, Duration::ZERO, &Claimed(true))
            .await
            .unwrap(),
        1
    );
    assert_eq!(
        backups.list(&owner).await.unwrap(),
        std::slice::from_ref(&second)
    );
    let left = keys(&store, &files).await;
    let wanted: BTreeSet<String> = second.partitions[0]
        .files
        .iter()
        .map(|key| format!("{files}{key}"))
        .collect();
    assert_eq!(left, wanted, "only what the newest needs is left");
    assert!(first.partitions[0]
        .files
        .iter()
        .any(|key| !left.contains(&format!("{files}{key}"))));
    assert_eq!(
        backups
            .prune(&owner, Duration::ZERO, &Claimed(true))
            .await
            .unwrap(),
        0,
        "the newest is kept however old"
    );
}

/// A store in which the first read of a file reports it missing, as if
/// cleaning up had just taken it.
#[derive(Debug)]
struct Vanishing {
    inner: Arc<dyn Blobs>,
    vanished: AtomicBool,
}

#[async_trait]
impl Blobs for Vanishing {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if key.contains("/segment/") && !self.vanished.swap(true, Ordering::SeqCst) {
            return Err(StorageError::Missing { key: key.into() });
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_backup_starts_again_when_a_file_goes_while_it_copies() {
    let cluster = Cluster::new().await;
    written(&cluster).await;
    let vanishing = Cluster {
        blobs: Arc::new(Vanishing {
            inner: cluster.blobs.clone(),
            vanished: AtomicBool::new(false),
        }),
        ..cluster
    };
    let store = memory();
    let backups = Backups::new(store.clone());
    let backup = take(&backups, &vanishing).await;
    assert_eq!(
        backups.list(&owner(&vanishing).await).await.unwrap(),
        [backup]
    );
    let records = keys(&store, &records(&owner(&vanishing).await)).await;
    assert_eq!(records.len(), 1, "no pending record is left: {records:?}");
}

/// A store that counts what is written to it.
#[derive(Debug)]
struct Counted {
    inner: Arc<dyn Blobs>,
    puts: AtomicUsize,
}

#[async_trait]
impl Blobs for Counted {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_backup_writes_only_the_files_no_backup_has_copied_yet() {
    let cluster = Cluster::new().await;
    let counted = Arc::new(Counted {
        inner: memory(),
        puts: AtomicUsize::new(0),
    });
    let backups = Backups::new(counted.clone());
    let writer = written(&cluster).await;
    take(&backups, &cluster).await;
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    counted.puts.store(0, Ordering::SeqCst);
    take(&backups, &cluster).await;
    assert_eq!(
        counted.puts.load(Ordering::SeqCst),
        3,
        "the new log, and the record as pending and as complete"
    );
}

#[tokio::test]
async fn pruning_leaves_a_backup_under_way_and_its_files_alone() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    let writer = written(&cluster).await;
    let under_way = take(&backups, &cluster).await;
    let owner = owner(&cluster).await;
    let (records, files) = (records(&owner), files(&owner));
    let complete = keys(&store, &records).await.pop_first().unwrap();
    let pending = complete.replace(".backup.json", ".pending.json");
    store
        .put(&pending, store.get(&complete).await.unwrap())
        .await
        .unwrap();
    store.delete(&complete).await.unwrap();

    writer.flush().await.unwrap();
    writer.merge().await.unwrap();
    writer.clean(Duration::ZERO).await.unwrap();
    take(&backups, &cluster).await;
    backups
        .prune(&owner, Duration::ZERO, &Claimed(true))
        .await
        .unwrap();
    assert!(keys(&store, &records).await.contains(&pending));
    let copied = keys(&store, &files).await;
    assert!(under_way
        .files()
        .all(|key| copied.contains(&format!("{files}{key}"))));
}

#[tokio::test]
async fn records_unreadable_or_of_another_tenant_are_passed_over() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let owner = owner(&cluster).await;
    let records = records(&owner);
    store
        .put(&format!("{records}0-broken.backup.json"), Bytes::from("{"))
        .await
        .unwrap();
    let mut another = serde_json::to_value(&backup).unwrap();
    another["tenant"] = json!("beta");
    another["id"] = json!(uuid::Uuid::new_v4());
    store
        .put(
            &format!("{records}1-another.backup.json"),
            Bytes::from(serde_json::to_vec(&another).unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(
        backups.list(&owner).await.unwrap(),
        std::slice::from_ref(&backup)
    );
}

/// Two backups of a tenant, the first listing files the second does not,
/// and their store.
async fn two_backups() -> (Cluster, Arc<dyn Blobs>, Backups, Backup, Backup) {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    let writer = written(&cluster).await;
    let first = take(&backups, &cluster).await;
    writer.flush().await.unwrap();
    writer.merge().await.unwrap();
    writer.clean(Duration::ZERO).await.unwrap();
    let second = take(&backups, &cluster).await;
    assert!(first
        .files()
        .any(|key| !second.files().any(|kept| kept == key)));
    (cluster, store, backups, first, second)
}

/// `backup`'s record, as JSON to be changed.
fn record_of(backup: &Backup) -> serde_json::Value {
    serde_json::to_value(backup).unwrap()
}

/// A way to make a backup's record one that cannot be used.
type Breaking = fn(serde_json::Value) -> Bytes;

#[tokio::test]
async fn pruning_deletes_nothing_while_a_record_cannot_be_used() {
    let unusable: [(&str, Breaking); 8] = [
        ("unreadable", |_| Bytes::from("{")),
        ("listing no files", |mut record| {
            record["partitions"][0]
                .as_object_mut()
                .unwrap()
                .remove("files");
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
        ("of another tenant", |mut record| {
            record["tenant"] = json!("beta");
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
        ("of another lineage", |mut record| {
            record["lineage"] = json!(uuid::Uuid::new_v4());
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
        ("of no lineage", |mut record| {
            record.as_object_mut().unwrap().remove("lineage");
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
        ("by another cluster", |mut record| {
            record["cluster"] = json!(uuid::Uuid::new_v4());
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
        ("by no cluster", |mut record| {
            record.as_object_mut().unwrap().remove("cluster");
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
        ("of a newer version", |mut record| {
            record["version"] = json!(2);
            record["partitions"][0]["files"] = json!([]);
            record["partitions"][0]["objects"] = json!(["alpha/0/segment/elsewhere"]);
            Bytes::from(serde_json::to_vec(&record).unwrap())
        }),
    ];
    for (how, broken) in unusable {
        let (cluster, store, backups, first, second) = two_backups().await;
        let owner = owner(&cluster).await;
        let records = records(&owner);
        let unusable = format!(
            "{records}{:020}-{}.backup.json",
            Utc::now().timestamp_millis(),
            uuid::Uuid::new_v4().simple()
        );
        store
            .put(&unusable, broken(record_of(&second)))
            .await
            .unwrap();
        let before = keys(&store, "").await;

        let refused = backups
            .prune(&owner, Duration::ZERO, &Claimed(true))
            .await
            .expect_err(how);
        assert!(
            matches!(&refused, StorageError::UnusableRecords { records, .. } if records.len() == 1 && records[0].starts_with(&unusable)),
            "{how}: {refused}"
        );
        assert_eq!(keys(&store, "").await, before, "{how}: nothing is deleted");
        assert_eq!(
            backups.list(&owner).await.unwrap(),
            [first, second],
            "{how}: the usable backups are still listed"
        );
    }
}

#[tokio::test]
async fn every_record_that_cannot_be_used_is_named() {
    let (cluster, store, backups, ..) = two_backups().await;
    let owner = owner(&cluster).await;
    let records = records(&owner);
    let unusable = ["0-broken", "1-broken"].map(|name| format!("{records}{name}.backup.json"));
    for key in &unusable {
        store.put(key, Bytes::from("{")).await.unwrap();
    }
    let refused = backups
        .prune(&owner, Duration::ZERO, &Claimed(true))
        .await
        .unwrap_err();
    let StorageError::UnusableRecords { records, .. } = &refused else {
        panic!("{refused}");
    };
    assert_eq!(records.len(), 2, "{refused}");
    assert!(unusable
        .iter()
        .all(|key| records.iter().any(|record| record.starts_with(key))));
}

#[tokio::test]
async fn a_pending_record_that_cannot_be_used_blocks_pruning_only_while_it_may_be_under_way() {
    let (cluster, store, backups, first, second) = two_backups().await;
    let owner = owner(&cluster).await;
    let (records, files) = (records(&owner), files(&owner));
    let pending = |written: chrono::DateTime<Utc>| {
        format!(
            "{records}{:020}-{}.pending.json",
            written.timestamp_millis(),
            uuid::Uuid::new_v4().simple()
        )
    };
    let young = pending(Utc::now());
    store.put(&young, Bytes::from("{")).await.unwrap();
    let refused = backups
        .prune(&owner, Duration::ZERO, &Claimed(true))
        .await
        .unwrap_err();
    assert!(
        matches!(&refused, StorageError::UnusableRecords { records, .. } if records[0].starts_with(&young)),
        "{refused}"
    );

    store.delete(&young).await.unwrap();
    let stale = pending(Utc::now() - chrono::Duration::hours(25));
    store.put(&stale, Bytes::from("{")).await.unwrap();
    assert_eq!(
        backups
            .prune(&owner, Duration::ZERO, &Claimed(true))
            .await
            .unwrap(),
        2,
        "the first backup and the stale record are forgotten"
    );
    assert!(!keys(&store, &stale).await.contains(&stale));
    assert_eq!(backups.list(&owner).await.unwrap(), [second]);
    let copied = keys(&store, &files).await;
    assert!(
        first
            .files()
            .any(|key| !copied.contains(&format!("{files}{key}"))),
        "what only the first listed is deleted"
    );
}

#[tokio::test]
async fn a_pending_record_that_cannot_be_used_and_says_no_time_blocks_pruning() {
    let (cluster, store, backups, ..) = two_backups().await;
    let owner = owner(&cluster).await;
    let records = records(&owner);
    let timeless = format!(
        "{records}sometime-{}.pending.json",
        uuid::Uuid::new_v4().simple()
    );
    store.put(&timeless, Bytes::from("{")).await.unwrap();
    let before = keys(&store, "").await;
    let refused = backups
        .prune(&owner, Duration::ZERO, &Claimed(true))
        .await
        .unwrap_err();
    assert!(
        matches!(&refused, StorageError::UnusableRecords { records, .. } if records[0].starts_with(&timeless)),
        "{refused}"
    );
    assert_eq!(
        keys(&store, "").await,
        before,
        "it may be of a backup under way"
    );
}

#[tokio::test]
async fn a_record_written_before_records_had_a_version_reads_as_the_first() {
    let (cluster, store, backups, first, _) = two_backups().await;
    let owner = owner(&cluster).await;
    let records = records(&owner);
    let complete = keys(&store, &records)
        .await
        .into_iter()
        .find(|key| key.contains(&first.id.simple().to_string()))
        .unwrap();
    let mut older = record_of(&first);
    let fields = older.as_object_mut().unwrap();
    for added in ["version", "keys", "restore_window"] {
        fields.remove(added);
    }
    store
        .put(&complete, Bytes::from(serde_json::to_vec(&older).unwrap()))
        .await
        .unwrap();
    let listed = backups.list(&owner).await.unwrap();
    assert_eq!(listed[0].id, first.id);
    assert_eq!(listed[0].version, RECORD_VERSION);
    assert!(listed[0].keys.is_empty());
    assert_eq!(
        backups
            .prune(&owner, Duration::ZERO, &Claimed(true))
            .await
            .unwrap(),
        1
    );
}

#[tokio::test]
async fn nothing_is_deleted_once_the_claim_is_lost() {
    let (cluster, store, backups, ..) = two_backups().await;
    let owner = owner(&cluster).await;
    let before = keys(&store, "").await;

    let refused = backups
        .prune(&owner, Duration::ZERO, &Claimed(false))
        .await
        .unwrap_err();
    assert!(matches!(refused, StorageError::Unheld(_)), "{refused}");
    assert_eq!(keys(&store, "").await, before, "pruning deletes nothing");

    let refused = backups.purge(&owner, &Claimed(false)).await.unwrap_err();
    assert!(matches!(refused, StorageError::Unheld(_)), "{refused}");
    assert_eq!(keys(&store, "").await, before, "nor does purging");
}
/// The right to write and delete, held for so many more checks.
struct Expiring(AtomicUsize);

impl Expiring {
    fn after(checks: usize) -> Self {
        Self(AtomicUsize::new(checks))
    }
}

#[async_trait]
impl Held for Expiring {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.still())
    }
    fn still(&self) -> bool {
        self.0
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                left.checked_sub(1)
            })
            .is_ok()
    }
}

#[tokio::test]
async fn nothing_is_written_once_the_claim_is_lost() {
    let cluster = Cluster::new().await;
    written(&cluster).await;
    let files = take(&Backups::new(memory()), &cluster)
        .await
        .files()
        .count();
    assert!(files > 1);
    let counted = Arc::new(Counted {
        inner: memory(),
        puts: AtomicUsize::new(0),
    });
    let backups = Backups::new(counted.clone());
    let unheld = |taken: vellev_storage::Result<Option<Backup>>| {
        matches!(taken, Err(StorageError::Unheld(_)))
    };

    // Lost once the pending record is written: no file is copied.
    assert!(unheld(
        taken_while(&backups, &cluster, &Expiring::after(1)).await
    ));
    assert_eq!(counted.puts.load(Ordering::SeqCst), 1, "the pending record");
    // Lost once every file is copied: the backup is not completed.
    counted.puts.store(0, Ordering::SeqCst);
    let lasting = Expiring::after(1 + files);
    assert!(unheld(taken_while(&backups, &cluster, &lasting).await));
    assert_eq!(
        counted.puts.load(Ordering::SeqCst),
        1 + files,
        "no complete record"
    );
    assert_eq!(backups.list(&owner(&cluster).await).await.unwrap(), []);
    let records = records(&owner(&cluster).await);
    assert!(
        keys(&counted.inner, &records).await.is_empty(),
        "nor a pending one"
    );
}

/// A tenant's store that takes so long to read a file from, once a backup
/// has begun to copy, that the claim is lost meanwhile.
#[derive(Debug)]
struct LosingAsRead {
    inner: Arc<dyn Blobs>,
    backups: Arc<dyn Blobs>,
    lost: Arc<AtomicBool>,
}

#[async_trait]
impl Blobs for LosingAsRead {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if !self.backups.list("_backups/").await?.is_empty() {
            self.lost.store(true, Ordering::SeqCst);
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// The right to write and delete, until told it is lost.
struct Until(Arc<AtomicBool>);

#[async_trait]
impl Held for Until {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.still())
    }
    fn still(&self) -> bool {
        !self.0.load(Ordering::SeqCst)
    }
}

#[tokio::test]
async fn a_file_read_as_the_claim_was_lost_is_not_written() {
    let cluster = Cluster::new().await;
    written(&cluster).await;
    let store = memory();
    let lost = Arc::new(AtomicBool::new(false));
    let slow = Cluster {
        blobs: Arc::new(LosingAsRead {
            inner: cluster.blobs.clone(),
            backups: store.clone(),
            lost: lost.clone(),
        }),
        ..cluster
    };
    let backups = Backups::new(store.clone());
    let taken = taken_while(&backups, &slow, &Until(lost)).await;
    assert!(matches!(taken, Err(StorageError::Unheld(_))), "{taken:?}");
    assert!(
        keys(&store, "").await.is_empty(),
        "no file is written, and the pending record is forgotten"
    );
}

#[tokio::test]
async fn nothing_is_fetched_back_once_the_claim_is_lost() {
    let cluster = Cluster::new().await;
    let backups = Backups::new(memory());
    written(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let lost = memory();
    let refused = backups
        .fetch_back(&lost, &backup, &Expiring::after(1))
        .await
        .unwrap_err();
    assert!(matches!(refused, StorageError::Unheld(_)), "{refused}");
    assert!(
        keys(&lost, "").await.is_empty(),
        "the file asked for while the claim was held is read, and not written once it is not"
    );
}

#[tokio::test]
async fn a_backup_lacking_a_file_is_mended_from_the_tenants_store_while_it_has_it() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let file = backup.files().next().unwrap().clone();
    let kept = format!("{}{file}", files(&owner(&cluster).await));

    store.delete(&kept).await.unwrap();
    let lacking = backups
        .mended(&cluster.blobs, &backup, &Claimed(true))
        .await
        .unwrap();
    assert_eq!(lacking, [] as [String; 0]);
    store.get(&kept).await.unwrap();

    store.delete(&kept).await.unwrap();
    cluster.blobs.delete(&file).await.unwrap();
    let lacking = backups
        .mended(&cluster.blobs, &backup, &Claimed(true))
        .await
        .unwrap();
    assert_eq!(lacking, [kept], "what the tenant's store lacks too");
}

/// The right to delete, renewed with the coordinator once and found lost
/// the next time.
struct RenewedOnce(AtomicUsize);

#[async_trait]
impl Held for RenewedOnce {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.0.fetch_add(1, Ordering::SeqCst) == 0)
    }
    fn still(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn deleting_asks_the_coordinator_again_before_each_batch() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    take(&backups, &cluster).await;
    let files = files(&owner(&cluster).await);
    let before = keys(&store, &files).await.len();
    for n in 0..DELETED_AT_ONCE + 4 {
        store
            .put(&format!("{files}left/{n}"), Bytes::from("left"))
            .await
            .unwrap();
    }
    let held = RenewedOnce(AtomicUsize::new(0));
    let refused = backups
        .prune(&owner(&cluster).await, Duration::ZERO, &held)
        .await
        .unwrap_err();
    assert!(matches!(refused, StorageError::Unheld(_)), "{refused}");
    assert_eq!(
        keys(&store, &files).await.len(),
        before + 4,
        "one batch went, and no more once the coordinator was asked again"
    );
}

#[tokio::test]
async fn a_partition_that_never_committed_is_backed_up_as_empty_beside_one_that_did() {
    let cluster = Cluster::new().await;
    written(&cluster).await;
    let other = PartitionKey {
        partition: PartitionNumber(1),
        ..cluster.key.clone()
    };
    cluster.coordinator.register(&other).await.unwrap();
    let backups = Backups::new(memory());
    let copied = backups
        .copy(
            &cluster.blobs,
            cluster.coordinator.as_ref(),
            &owner(&cluster).await,
            &[other.clone(), cluster.key.clone()],
            &Claimed(true),
        )
        .await
        .unwrap()
        .unwrap();
    let backup = backups.complete(copied, &Claimed(true)).await.unwrap();
    let empty = &backup.partitions[0];
    assert_eq!((empty.partition, &empty.head), (other.partition, &None));
    assert!(empty.files.is_empty());
    assert_eq!(backup.position(), backup.partitions[1].position);
}

#[test]
fn a_backup_says_where_its_default_partition_was() {
    let taken = |partition, position| vellev_storage::backups::Taken {
        partition,
        incarnation: Incarnation(1),
        position: LogPosition(position),
        head: None,
        files: Vec::new(),
    };
    let backup = Backup {
        version: RECORD_VERSION,
        id: uuid::Uuid::new_v4(),
        tenant: TenantId::new("acme").unwrap(),
        lineage: Lineage(uuid::Uuid::new_v4()),
        cluster: ClusterId(uuid::Uuid::new_v4()),
        taken_at: Utc::now(),
        partitions: vec![
            taken(PartitionNumber(1), 7),
            taken(PartitionNumber::DEFAULT, 3),
        ],
        keys: Vec::new(),
        restore_window: None,
    };
    assert_eq!(backup.position(), LogPosition(3));
}

#[tokio::test]
async fn a_backup_listing_a_file_outside_its_partition_is_not_restored() {
    let cluster = Cluster::new().await;
    let backups = Backups::new(memory());
    written(&cluster).await;
    let mut backup = take(&backups, &cluster).await;
    backup.partitions[0]
        .files
        .push("beta/0/segment/stray.segment".into());
    let refused = backups
        .fetch_back(&memory(), &backup, &Claimed(true))
        .await
        .unwrap_err();
    assert!(matches!(refused, StorageError::Corrupt { .. }), "{refused}");
}

/// A store that loses the first file written to it, as if a prune had
/// deleted it under a backup whose claim ran out.
#[derive(Debug)]
struct Losing {
    inner: Arc<dyn Blobs>,
    lost: AtomicBool,
}

#[async_trait]
impl Blobs for Losing {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if key.contains("/data/") && !self.lost.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_backup_is_complete_only_once_every_file_it_lists_is_there() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(Arc::new(Losing {
        inner: store.clone(),
        lost: AtomicBool::new(false),
    }));
    written(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let files = keys(&store, &files(&owner(&cluster).await)).await;
    assert_eq!(files.len(), backup.partitions[0].files.len());
}

/// The tenant of `cluster` removed and purged, and one of its name made.
async fn made_again(cluster: &Cluster) -> Owner {
    assert!(cluster.coordinator.remove(&cluster.key).await.unwrap());
    purge(cluster.coordinator.as_ref(), &cluster.key.tenant).await;
    cluster.coordinator.register(&cluster.key).await.unwrap();
    owner(cluster).await
}

/// The key of the complete record of `backup`, kept among `records`.
fn complete_key(records: &str, backup: &Backup) -> String {
    format!(
        "{records}{:020}-{}.backup.json",
        backup.taken_at.timestamp_millis(),
        backup.id.simple()
    )
}

const NONE: [String; 0] = [];

#[tokio::test]
async fn a_tenant_made_again_under_a_name_has_nothing_of_the_backups_of_the_one_before() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let before = owner(&cluster).await;
    let old = take(&backups, &cluster).await;
    assert_eq!((old.lineage, old.cluster), (before.lineage, before.cluster));
    let kept = keys(&store, "").await;
    assert!(
        kept.iter().all(|key| kept_by(&before, key)),
        "all of it lies under its lineage and its cluster: {kept:?}"
    );

    let again = made_again(&cluster).await;
    assert_ne!(again.lineage, before.lineage);
    assert_eq!(backups.list(&again).await.unwrap(), []);
    assert_eq!(backups.find(&again, old.id).await.unwrap(), None);
    let pruned = backups.prune(&again, Duration::ZERO, &Claimed(true)).await;
    assert_eq!(pruned.unwrap(), 0);
    assert!(!backups.any_kept(&again).await.unwrap());
    let purged = backups.purge(&again, &Claimed(true)).await;
    assert_eq!(purged.unwrap(), 0);
    assert_eq!(
        keys(&store, "").await,
        kept,
        "nothing of the one before is touched"
    );

    written(&cluster).await;
    let new = take(&backups, &cluster).await;
    assert_eq!(
        backups.list(&again).await.unwrap(),
        std::slice::from_ref(&new)
    );
    assert_eq!(
        backups.list(&before).await.unwrap(),
        std::slice::from_ref(&old)
    );
    assert_eq!(
        backups.of_any(&cluster.key.tenant).await.unwrap(),
        [old.clone(), new.clone()],
        "whoever has lost a tenant sees every backup of its name"
    );

    // The one before, purged, takes nothing of the one made since with it.
    assert!(backups.any_kept(&before).await.unwrap());
    let purged = backups.purge(&before, &Claimed(true)).await;
    assert_eq!(purged.unwrap(), kept.len());
    let left = keys(&store, "").await;
    assert!(!left.is_empty() && left.iter().all(|key| kept_by(&again, key)));
    assert_eq!(
        backups.list(&again).await.unwrap(),
        std::slice::from_ref(&new)
    );
    let lacking = backups.mended(&memory(), &new, &Claimed(true)).await;
    assert_eq!(lacking.unwrap(), NONE, "every file it lists is there");
}

/// `owner`'s tenant as another cluster knows it.
fn elsewhere(owner: &Owner) -> Owner {
    Owner {
        cluster: ClusterId(uuid::Uuid::new_v4()),
        ..owner.clone()
    }
}

#[tokio::test]
async fn a_cluster_restores_from_what_another_took_of_its_tenant_and_deletes_none_of_it() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    let writer = written(&cluster).await;
    // The cluster the tenant was adopted from, and this one, which took it
    // up: the same tenant, each backing it up under its own id.
    let live = owner(&cluster).await;
    let taken_live = take(&backups, &cluster).await;
    let theirs = keys(&store, "").await;
    assert!(theirs.iter().all(|key| kept_by(&live, key)));

    let drill = elsewhere(&live);
    assert_eq!(
        backups.list(&drill).await.unwrap(),
        std::slice::from_ref(&taken_live),
        "it lists what the other took of its tenant"
    );
    let lost = memory();
    let fetched = backups.fetch_back(&lost, &taken_live, &Claimed(true));
    assert_eq!(fetched.await.unwrap(), taken_live.files().count());
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    let copied = backups.copy(
        &cluster.blobs,
        cluster.coordinator.as_ref(),
        &drill,
        std::slice::from_ref(&cluster.key),
        &Claimed(true),
    );
    let copied = copied.await.unwrap().unwrap();
    let taken_drill = backups.complete(copied, &Claimed(true)).await.unwrap();
    assert_eq!(taken_drill.cluster, drill.cluster);
    let ours: BTreeSet<String> = keys(&store, "")
        .await
        .difference(&theirs)
        .cloned()
        .collect();
    assert!(
        ours.iter().all(|key| kept_by(&drill, key)),
        "every file it needs is copied under its own id, sharing none: {ours:?}"
    );
    assert_eq!(
        ours.iter().filter(|key| key.contains("/data/")).count(),
        taken_drill.files().count()
    );
    assert_eq!(
        backups.list(&live).await.unwrap(),
        [taken_live.clone(), taken_drill.clone()]
    );

    // It prunes, and purges, only what it took.
    let pruned = backups.prune(&drill, Duration::ZERO, &Claimed(true)).await;
    assert_eq!(pruned.unwrap(), 0, "its newest is its one");
    assert_eq!(keys(&store, "").await.len(), theirs.len() + ours.len());
    assert!(backups.any_kept(&drill).await.unwrap());
    let purged = backups.purge(&drill, &Claimed(true)).await;
    assert_eq!(purged.unwrap(), ours.len());
    assert_eq!(keys(&store, "").await, theirs, "the other's is as it was");
    assert!(!backups.any_kept(&drill).await.unwrap());
    assert_eq!(backups.list(&live).await.unwrap(), [taken_live]);
}

#[tokio::test]
async fn a_record_that_says_no_lineage_or_cluster_is_nobodys() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let own = owner(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let tenant = &own.tenant;
    // As a release before tenants had lineages kept them: under the name
    // alone, saying neither.
    let mut before = record_of(&backup);
    for unsaid in ["lineage", "cluster"] {
        before.as_object_mut().unwrap().remove(unsaid);
    }
    let before = Bytes::from(serde_json::to_vec(&before).unwrap());
    let old = [
        complete_key(&format!("_backups/{tenant}/backups/"), &backup),
        format!("_backups/{tenant}/files/{}", backup.files().next().unwrap()),
    ];
    for key in &old {
        store.put(key, before.clone()).await.unwrap();
    }
    let kept = keys(&store, "").await;

    assert_eq!(
        backups.list(&own).await.unwrap(),
        std::slice::from_ref(&backup)
    );
    assert_eq!(
        backups.of_any(tenant).await.unwrap(),
        std::slice::from_ref(&backup),
        "it is not there to be taken up"
    );
    assert_eq!(backups.find(&own, backup.id).await.unwrap(), Some(backup));
    let pruned = backups.prune(&own, Duration::ZERO, &Claimed(true)).await;
    assert_eq!(pruned.unwrap(), 0);
    assert_eq!(keys(&store, "").await, kept);
    backups.purge(&own, &Claimed(true)).await.unwrap();
    assert_eq!(
        keys(&store, "").await,
        BTreeSet::from(old),
        "nor purged with any tenant of the name"
    );
}

/// The release before lineages lists `_backups/{tenant}/backups/` and
/// `_backups/{tenant}/files/` to their ends as its own, and deletes under
/// them whatever no record it reads lists: so nothing of this release may
/// lie under either, or a node of that release, run beside this one or
/// after a rollback, prunes the files of every backup this one takes.
#[tokio::test]
async fn nothing_this_release_keeps_lies_where_the_release_before_looks() {
    let cluster = Cluster::new().await;
    let listings = Arc::new(Listings {
        inner: memory(),
        listed: std::sync::Mutex::default(),
    });
    let store: Arc<dyn Blobs> = listings.clone();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let own = owner(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let Owner {
        tenant,
        lineage,
        cluster: by,
    } = &own;
    let (before_records, before_files) = (
        format!("_backups/{tenant}/backups/"),
        format!("_backups/{tenant}/files/"),
    );
    let kept = keys(&store, &format!("_backups/{tenant}/")).await;
    assert!(!kept.is_empty());
    let record = complete_key(&records(&own), &backup);
    assert!(kept.contains(&record), "{kept:?}");
    assert!(record.starts_with(&format!("_backups/{tenant}/records/{lineage}/{by}/")));
    let file = format!("{}{}", files(&own), backup.files().next().unwrap());
    assert!(kept.contains(&file), "{kept:?}");
    assert!(file.starts_with(&format!("_backups/{tenant}/data/{lineage}/{by}/")));
    for key in &kept {
        assert!(
            !key.starts_with(&before_records) && !key.starts_with(&before_files),
            "{key} lies where the release before looks"
        );
    }
    assert!(keys(&store, &before_records).await.is_empty());
    assert!(keys(&store, &before_files).await.is_empty());

    // Nor does anything of this release look there: finding backups lists
    // the records' place alone, and pruning and purging list records and
    // files under the lineage and cluster alone.
    let (listed_records, listed_files) = (
        format!("_backups/{tenant}/records/"),
        format!("_backups/{tenant}/data/"),
    );
    listings.listed();
    backups.of_any(tenant).await.unwrap();
    assert_eq!(listings.listed(), std::slice::from_ref(&listed_records));
    backups.list(&own).await.unwrap();
    assert_eq!(listings.listed(), [format!("{listed_records}{lineage}/")]);
    backups.find(&own, backup.id).await.unwrap();
    assert_eq!(listings.listed(), [format!("{listed_records}{lineage}/")]);
    backups
        .prune(&own, Duration::ZERO, &Claimed(true))
        .await
        .unwrap();
    backups.any_kept(&own).await.unwrap();
    backups.purge(&own, &Claimed(true)).await.unwrap();
    let listed = listings.listed();
    assert!(!listed.is_empty());
    for prefix in listed {
        assert!(
            prefix.starts_with(&format!("{listed_records}{lineage}/{by}/"))
                || prefix.starts_with(&format!("{listed_files}{lineage}/{by}/")),
            "{prefix} was listed"
        );
    }
}

/// Object storage that notes each prefix listed of it.
#[derive(Debug)]
struct Listings {
    inner: Arc<dyn Blobs>,
    listed: std::sync::Mutex<Vec<String>>,
}

impl Listings {
    /// The prefixes listed since last asked.
    fn listed(&self) -> Vec<String> {
        std::mem::take(&mut self.listed.lock().unwrap())
    }
}

#[async_trait]
impl Blobs for Listings {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.listed.lock().unwrap().push(prefix.to_string());
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_record_lying_where_it_does_not_say_is_no_backup_to_take_a_tenant_up_from() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    let own = owner(&cluster).await;
    let backup = take(&backups, &cluster).await;
    let record = Bytes::from(serde_json::to_vec(&backup).unwrap());
    let another = Owner {
        lineage: Lineage(uuid::Uuid::new_v4()),
        ..own.clone()
    };
    for misplaced in [&another, &elsewhere(&own)] {
        let misplaced = complete_key(&records(misplaced), &backup);
        store.put(&misplaced, record.clone()).await.unwrap();
    }
    let mut named = record_of(&backup);
    named["tenant"] = json!("beta");
    named["id"] = json!(uuid::Uuid::new_v4());
    let named = Bytes::from(serde_json::to_vec(&named).unwrap());
    let beside = format!("{}1-another.backup.json", records(&own));
    store.put(&beside, named).await.unwrap();

    assert_eq!(
        backups.of_any(&own.tenant).await.unwrap(),
        std::slice::from_ref(&backup)
    );
    assert_eq!(backups.list(&another).await.unwrap(), []);
    assert_eq!(backups.list(&own).await.unwrap(), [backup]);
}

/// A store in which a complete record lands, and is then said to have
/// failed: as a write the store had when the node gave up on it does.
#[derive(Debug)]
struct LandingUnsaid(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for LandingUnsaid {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.0.put(key, bytes).await?;
        match key.ends_with(".backup.json") {
            true => Err(unanswered(key, Duration::from_secs(1))),
            false => Ok(()),
        }
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.0.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.0.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.0.delete(key).await
    }
}

#[tokio::test]
async fn a_backup_whose_complete_record_is_not_known_to_be_written_leaves_none() {
    let cluster = Cluster::new().await;
    written(&cluster).await;
    let store = memory();
    let backups = Backups::new(Arc::new(LandingUnsaid(store.clone())));
    let taken = taken_while(&backups, &cluster, &Claimed(true)).await;
    assert!(taken.is_err(), "{taken:?}");
    let owner = owner(&cluster).await;
    let records = keys(&store, &records(&owner)).await;
    assert!(
        records.is_empty(),
        "neither the complete record, which landed, nor the pending one: {records:?}"
    );
    assert_eq!(Backups::new(store).list(&owner).await.unwrap(), []);
}

/// A backup store that takes so long to read a file from that the claim is
/// lost meanwhile.
#[derive(Debug)]
struct LosingAsFetched {
    inner: Arc<dyn Blobs>,
    lost: Arc<AtomicBool>,
}

#[async_trait]
impl Blobs for LosingAsFetched {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if key.contains("/data/") {
            self.lost.store(true, Ordering::SeqCst);
        }
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

#[tokio::test]
async fn a_file_fetched_as_the_claim_was_lost_is_not_put_back() {
    let cluster = Cluster::new().await;
    let store = memory();
    written(&cluster).await;
    let backup = take(&Backups::new(store.clone()), &cluster).await;
    let lost = Arc::new(AtomicBool::new(false));
    let slow = Backups::new(Arc::new(LosingAsFetched {
        inner: store,
        lost: lost.clone(),
    }));
    let primary = memory();
    let refused = slow.fetch_back(&primary, &backup, &Until(lost)).await;
    let refused = refused.unwrap_err();
    assert!(matches!(refused, StorageError::Unheld(_)), "{refused}");
    assert!(
        keys(&primary, "").await.is_empty(),
        "nothing read once the claim was lost is written"
    );
}

#[tokio::test]
async fn purging_asks_the_coordinator_again_before_each_batch() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    take(&backups, &cluster).await;
    let owner = owner(&cluster).await;
    for n in 0..DELETED_AT_ONCE + 4 {
        let left = format!("{}left/{n}", files(&owner));
        store.put(&left, Bytes::from("left")).await.unwrap();
    }
    let before = keys(&store, "").await.len();
    let held = RenewedOnce(AtomicUsize::new(0));
    let refused = backups.purge(&owner, &held).await.unwrap_err();
    assert!(matches!(refused, StorageError::Unheld(_)), "{refused}");
    assert_eq!(
        keys(&store, "").await.len(),
        before - DELETED_AT_ONCE,
        "one batch went, and no more once the coordinator was asked again"
    );
}

#[tokio::test(start_paused = true)]
async fn a_listing_is_given_up_once_one_answer_of_it_takes_too_long_however_long_the_whole() {
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    let second = Duration::from_secs(1);
    let slowly = ThrottleConfig {
        wait_list_per_entry: second,
        ..ThrottleConfig::default()
    };
    let store = ObjectBlobs::new(Arc::new(ThrottledStore::new(InMemory::new(), slowly)));
    for n in 0..5 {
        let key = format!("listed/{n}");
        store.put(&key, Bytes::from("listed")).await.unwrap();
    }
    let started = tokio::time::Instant::now();
    let listed = store.list_within("listed/", second * 2).await.unwrap();
    assert_eq!(listed.len(), 5);
    assert!(
        started.elapsed() >= second * 5,
        "the whole took longer than any one answer may"
    );
    let refused = store.list_within("listed/", second / 2).await.unwrap_err();
    assert!(matches!(refused, StorageError::Blob { .. }), "{refused}");

    // A store that says nothing of how it lists is given that long for the
    // whole.
    let whole = LandingUnsaid(Arc::new(store));
    let refused = whole.list_within("listed/", second * 2).await.unwrap_err();
    assert!(matches!(refused, StorageError::Blob { .. }), "{refused}");
    assert_eq!(
        whole
            .list_within("listed/", second * 6)
            .await
            .unwrap()
            .len(),
        5
    );
}

/// A store each delete from which takes a second.
#[derive(Debug)]
struct DeletingSlowly(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for DeletingSlowly {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.0.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.0.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.0.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        tokio::time::sleep(Duration::from_secs(1)).await;
        self.0.delete(key).await
    }
}

#[tokio::test(start_paused = true)]
async fn the_files_of_a_batch_are_deleted_together() {
    let store = memory();
    let gone = Owner {
        tenant: TenantId::new("acme").unwrap(),
        lineage: Lineage(uuid::Uuid::new_v4()),
        cluster: ClusterId(uuid::Uuid::new_v4()),
    };
    let batches = 3;
    for n in 0..batches * DELETED_AT_ONCE {
        let kept = format!("{}{n}", files(&gone));
        store.put(&kept, Bytes::from("kept")).await.unwrap();
    }
    let backups = Backups::new(Arc::new(DeletingSlowly(store.clone())));
    let started = tokio::time::Instant::now();
    let purged = backups.purge(&gone, &Claimed(true)).await.unwrap();
    assert_eq!(purged, batches * DELETED_AT_ONCE);
    assert!(keys(&store, "").await.is_empty());
    assert_eq!(
        started.elapsed(),
        Duration::from_secs(batches as u64),
        "as long as one delete for each batch"
    );
}

#[tokio::test]
async fn a_backup_says_when_it_was_taken_by_its_clock_to_the_microsecond() {
    let cluster = Cluster::new().await;
    written(&cluster).await;
    // A clock to the nanosecond, as a Linux machine's is: a coordinator
    // keeps the time a restore went back to to the microsecond, and a
    // backup that said a finer one would be another time read back.
    let fine = chrono::DateTime::<Utc>::from_timestamp(1_760_000_000, 123_456_789).unwrap();
    let clock = Arc::new(ManualClock::starting_at(fine));
    let backups = Backups::new(memory()).timed_by(clock);
    let backup = take(&backups, &cluster).await;
    assert_eq!(backup.taken_at.timestamp(), 1_760_000_000);
    assert_eq!(backup.taken_at.timestamp_subsec_nanos(), 123_456_000);
    let own = owner(&cluster).await;
    assert_eq!(backups.list(&own).await.unwrap(), [backup]);
}
