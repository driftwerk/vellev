//! A tenant copied to a second store, and put back from it.

mod common;

use common::*;
use vellev_storage::backups::{Backup, Backups};

fn memory() -> Arc<dyn Blobs> {
    Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())))
}

fn ns(state: &State, objects: &[u8]) -> Vec<Option<i64>> {
    objects
        .iter()
        .map(|object| n_of(state, "notes", *object))
        .collect()
}

async fn take(backups: &Backups, cluster: &Cluster) -> Backup {
    backups
        .take(
            &cluster.blobs,
            cluster.coordinator.as_ref(),
            &cluster.key.tenant,
            std::slice::from_ref(&cluster.key),
        )
        .await
        .unwrap()
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
        backups.list(&cluster.key.tenant).await.unwrap(),
        std::slice::from_ref(&backup)
    );

    let lost = Cluster {
        blobs: memory(),
        ..Cluster::new().await
    };
    let copied = backups.fetch_back(&lost.blobs, &backup).await.unwrap();
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
    assert_eq!(backups.fetch_back(&lost.blobs, &backup).await.unwrap(), 0);
}

#[tokio::test]
async fn each_file_is_copied_once_whatever_the_backups_that_need_it() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    let writer = written(&cluster).await;
    let first = take(&backups, &cluster).await;
    let files = format!("{}/files/", cluster.key.tenant);
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
    let tenant = &cluster.key.tenant;

    assert_eq!(
        backups
            .prune(tenant, Duration::from_secs(3_600))
            .await
            .unwrap(),
        0
    );
    assert_eq!(backups.list(tenant).await.unwrap().len(), 2);
    assert_eq!(backups.prune(tenant, Duration::ZERO).await.unwrap(), 1);
    assert_eq!(
        backups.list(tenant).await.unwrap(),
        std::slice::from_ref(&second)
    );
    let files = format!("{tenant}/files/");
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
        backups.prune(tenant, Duration::ZERO).await.unwrap(),
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
    assert_eq!(backups.list(&vanishing.key.tenant).await.unwrap(), [backup]);
    let records = keys(&store, &format!("{}/backups/", vanishing.key.tenant)).await;
    assert_eq!(records.len(), 1, "no pending record is left: {records:?}");
}

#[tokio::test]
async fn purging_leaves_nothing_of_the_tenant() {
    let cluster = Cluster::new().await;
    let store = memory();
    let backups = Backups::new(store.clone());
    written(&cluster).await;
    take(&backups, &cluster).await;
    assert!(backups.purge(&cluster.key.tenant).await.unwrap() > 0);
    assert!(keys(&store, "").await.is_empty());
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
    let tenant = &cluster.key.tenant;
    let records = format!("{tenant}/backups/");
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
    backups.prune(tenant, Duration::ZERO).await.unwrap();
    assert!(keys(&store, &records).await.contains(&pending));
    let files = keys(&store, &format!("{tenant}/files/")).await;
    assert!(under_way
        .files()
        .all(|key| files.contains(&format!("{tenant}/files/{key}"))));
}
