//! Continuous backup's shipper (DW-668, S2): a partition's commits shipped
//! to a backup store as links, from the rows of its commits, with each
//! checkpoint's files before its link.

mod common;

use common::*;
use vellev_storage::blobs::Created;
use vellev_storage::chain;
use vellev_storage::coordinator::{ClusterId, Lineage, ShipBase};
use vellev_storage::format::{decode_head, Head as Stored};
use vellev_storage::link::{self, Body, Header, Link, Timeline};
use vellev_storage::shipper::{Batch, Idle, Shipper, Stall};
use vellev_storage::Transfers;

fn lineage() -> Lineage {
    Lineage(uuid::Uuid::from_u128(1))
}

fn cluster_id() -> ClusterId {
    ClusterId(uuid::Uuid::from_u128(2))
}

/// A backup store that creates files of `part_bytes` at most.
fn backup_of(part_bytes: usize) -> Arc<dyn Blobs> {
    let transfers = Transfers {
        part_bytes,
        ..Transfers::default()
    };
    Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())).transferring(transfers))
}

fn a_backup() -> Arc<dyn Blobs> {
    backup_of(Transfers::default().part_bytes)
}

/// A partition with a checkpoint at 3 (notes 1 and 2, then a flush), and
/// two writes after it, at 4 and 5.
async fn written() -> (Cluster, Writer) {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
    writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
    (cluster, writer)
}

/// The newest checkpoint committed, and its position.
async fn newest_checkpoint(cluster: &Cluster) -> (LogPosition, ManifestRef) {
    let now = cluster.committed().await.incarnation;
    let rows = cluster
        .coordinator
        .commits_after(&cluster.key, now, LogPosition(0), 1_000)
        .await
        .unwrap();
    let row = rows
        .into_iter()
        .rfind(|row| row.head.0.contains("/manifest/"))
        .expect("a checkpoint");
    (row.position, row.head)
}

/// Shipping begun as bootstrap begins it: held, then based where the
/// newest checkpoint's restore starts (none of these replay anything).
async fn begun(cluster: &Cluster) -> ShipBase {
    let now = cluster.committed().await.incarnation;
    let (at, checkpoint) = newest_checkpoint(cluster).await;
    let c = &cluster.coordinator;
    c.hold_for_shipping(&cluster.key, now, at).await.unwrap();
    let base = ShipBase {
        position: at,
        checkpoint,
        checkpoint_at: at,
    };
    c.record_base(&cluster.key, now, &base).await.unwrap();
    base
}

fn shipper_of(cluster: &Cluster, backup: &Arc<dyn Blobs>) -> Shipper {
    Shipper::new(
        cluster.blobs.clone(),
        backup.clone(),
        cluster.coordinator.clone(),
        cluster.key.clone(),
        lineage(),
        cluster_id(),
    )
}

async fn ship(shipper: &Shipper, cluster: &Cluster, limit: usize) -> Batch {
    let now = cluster.committed().await.incarnation;
    shipper.ship_batch(now, limit).await.unwrap()
}

/// Every link in `backup` of `timeline`, by position.
async fn links(backup: &Arc<dyn Blobs>, timeline: &Timeline) -> BTreeMap<LogPosition, Link> {
    let listed = backup
        .list(&format!("{}links/", timeline.prefix()))
        .await
        .unwrap();
    let mut links = BTreeMap::new();
    for file in listed {
        let link = link::decode(&file.key, &backup.get(&file.key).await.unwrap()).unwrap();
        links.insert(link.header.position, link);
    }
    links
}

#[tokio::test]
async fn a_run_is_shipped_from_its_base_each_commit_a_link_of_its_row_and_frame() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    let committed = cluster.committed().await;
    let shipped = ship(&shipper, &cluster, 100).await;
    assert_eq!(
        shipped,
        Batch::Shipped {
            through: Some(committed.log_position),
            stalled: None
        }
    );
    let now = committed.incarnation;
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!(shipping.shipped, Some(committed.log_position));

    let timeline = shipper.timeline(now);
    let links = links(&backup, &timeline).await;
    let rows = cluster
        .coordinator
        .commits_after(&cluster.key, now, LogPosition(base.position.0 - 1), 100)
        .await
        .unwrap();
    assert_eq!(
        links.keys().copied().collect::<Vec<_>>(),
        rows.iter().map(|row| row.position).collect::<Vec<_>>()
    );
    let mut before: Option<(LogPosition, link::Sum)> = None;
    for row in &rows {
        let link = &links[&row.position];
        let header = &link.header;
        // The head the row names, byte for byte, at the time its row was made.
        let frame = cluster.blobs.get(&row.head.0).await.unwrap();
        assert_eq!(link.frame.as_deref(), Some(&frame[..]));
        assert_eq!(header.key, row.head.0);
        assert_eq!(header.committed_at, row.at);
        assert_eq!(header.body, Body::Inline);
        assert_eq!(
            header.before.as_ref().map(|b| (b.position, b.sum.clone())),
            before
        );
        before = Some((header.position, header.sum.clone()));
        // A checkpoint's files there before it, as summed.
        for needed in &header.files {
            let copy = backup.get(&timeline.file(&needed.key)).await.unwrap();
            assert_eq!(copy.len() as u64, needed.size);
            assert_eq!(link::Sum::of(&copy), needed.sum);
            assert_eq!(copy, cluster.blobs.get(&needed.key).await.unwrap());
        }
    }
    assert!(
        links[&base.position].header.files.len() >= 2,
        "a checkpoint's files"
    );
    // What restoring the newest commit needs is all there.
    let headers: BTreeMap<_, _> = links
        .iter()
        .map(|(at, link)| (*at, link.header.clone()))
        .collect();
    let checkpoint = chain::checkpoint_for(&headers, committed.log_position)
        .unwrap()
        .unwrap();
    let Ok(Stored::Manifest(manifest)) = decode_head(
        &checkpoint.key,
        links[&checkpoint.position].frame.as_ref().unwrap(),
    ) else {
        panic!("a manifest")
    };
    let needed = chain::needed(&headers, committed.log_position, Some(&manifest)).unwrap();
    assert_eq!(needed.checkpoint, Some(base.position));
}

#[tokio::test]
async fn a_run_is_shipped_a_batch_at_a_time_each_going_on_from_what_the_last_shipped() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    let first = ship(&shipper, &cluster, 1).await;
    assert_eq!(
        first,
        Batch::Shipped {
            through: Some(base.position),
            stalled: None
        }
    );
    let second = ship(&shipper, &cluster, 1).await;
    let next = LogPosition(base.position.0 + 1);
    assert_eq!(
        second,
        Batch::Shipped {
            through: Some(next),
            stalled: None
        }
    );
    let now = cluster.committed().await.incarnation;
    let links = links(&backup, &shipper.timeline(now)).await;
    let before = links[&next].header.before.clone().unwrap();
    assert_eq!(before.position, base.position);
    assert_eq!(before.sum, links[&base.position].header.sum);
    // Nothing more past what is committed: shipped through nothing.
    ship(&shipper, &cluster, 100).await;
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: None,
            stalled: None
        }
    );
}

#[tokio::test]
async fn nothing_is_shipped_without_a_base_of_this_incarnation_or_once_stalled() {
    let (cluster, _writer) = written().await;
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    let now = cluster.committed().await.incarnation;
    assert_eq!(
        shipper.ship_batch(now, 10).await.unwrap(),
        Batch::Idle(Idle::NotShipping)
    );
    let base = begun(&cluster).await;
    let other = Incarnation(now.0 + 1);
    assert_eq!(
        shipper.ship_batch(other, 10).await.unwrap(),
        Batch::Idle(Idle::OtherIncarnation)
    );
    cluster
        .coordinator
        .stall_shipping(&cluster.key, now, base.position, "for a test")
        .await
        .unwrap();
    assert_eq!(
        shipper.ship_batch(now, 10).await.unwrap(),
        Batch::Idle(Idle::Stalled("for a test".into()))
    );
    assert!(backup.list("").await.unwrap().is_empty());
}

/// The link the shipper would make at `position`, made by a first
/// shipper into a store of its own, noting nothing, and read back.
async fn link_at(cluster: &Cluster, position: LogPosition) -> Link {
    let elsewhere = a_backup();
    let (first, unnoting) = shipper_over_rows(cluster, &elsewhere);
    unnoting.unnoted.store(true, Ordering::SeqCst);
    ship(&first, cluster, 100).await;
    let now = cluster.committed().await.incarnation;
    links(&elsewhere, &first.timeline(now))
        .await
        .remove(&position)
        .unwrap()
}

/// `header` with `frame`, put at its position's link in `backup`.
async fn put_link(backup: &Arc<dyn Blobs>, header: &Header, frame: Option<&[u8]>) {
    let at = Timeline::of(header).link(header.position);
    let made = backup
        .put_new(&at, link::encode(header, frame).unwrap())
        .await
        .unwrap();
    assert_eq!(made, Created::Written);
}

#[tokio::test]
async fn a_link_already_there_of_the_same_commit_is_shipped_and_one_of_another_stalls() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let next = LogPosition(base.position.0 + 1);
    let same = link_at(&cluster, next).await;
    // The same commit, shipped already, as by a shipper that crashed
    // before it noted it: shipped.
    let backup = a_backup();
    put_link(&backup, &same.header, same.frame.as_deref()).await;
    let shipper = shipper_of(&cluster, &backup);
    let committed = cluster.committed().await.log_position;
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(committed),
            stalled: None
        }
    );
    // Another commit at that position: a fork, the run stopped before it.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let mut other = link_at(&cluster, next).await.header;
    other.committed_at += chrono::Duration::microseconds(1);
    let backup = a_backup();
    put_link(
        &backup,
        &other,
        link_at(&cluster, next).await.frame.as_deref(),
    )
    .await;
    let shipper = shipper_of(&cluster, &backup);
    let stalled = ship(&shipper, &cluster, 100).await;
    let fork = Stall::Fork {
        position: next,
        fields: vec!["committed_at"],
    };
    assert_eq!(
        stalled,
        Batch::Shipped {
            through: Some(base.position),
            stalled: Some(fork.clone())
        }
    );
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!(shipping.shipped, Some(base.position));
    assert_eq!(shipping.stalled, Some(fork.to_string()));
}

#[tokio::test]
async fn at_its_base_a_run_takes_a_link_that_differs_only_in_the_commit_before_it_and_no_more() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let mine = link_at(&cluster, base.position).await;
    // An older run's link of the same commit, which knew the commit
    // before it: tolerated at the base.
    let mut older = mine.header.clone();
    older.before = Some(link::Before {
        position: LogPosition(base.position.0 - 1),
        sum: link::Sum::of(b"the commit before"),
    });
    let backup = a_backup();
    put_link(&backup, &older, mine.frame.as_deref()).await;
    let shipper = shipper_of(&cluster, &backup);
    let committed = cluster.committed().await.log_position;
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(committed),
            stalled: None
        }
    );

    // One that differs in more, at the base: another commit, a fork, its
    // fields named.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let mine = link_at(&cluster, base.position).await;
    let mut another = mine.header.clone();
    another.committed_at += chrono::Duration::microseconds(1);
    let backup = a_backup();
    put_link(&backup, &another, mine.frame.as_deref()).await;
    let shipper = shipper_of(&cluster, &backup);
    let stalled = Stall::Fork {
        position: base.position,
        fields: vec!["committed_at"],
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: None,
            stalled: Some(stalled)
        }
    );
}

/// A store that creates nothing: every create refused before it is sent.
#[derive(Debug)]
struct Uncreating(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for Uncreating {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.0.put(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        self.0.put_parts(key, parts).await
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
async fn a_store_that_creates_nothing_stalls_the_run() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let backup: Arc<dyn Blobs> = Arc::new(Uncreating(a_backup()));
    let shipper = shipper_of(&cluster, &backup);
    match ship(&shipper, &cluster, 100).await {
        Batch::Shipped {
            through: None,
            stalled: Some(Stall::NotCreated { position, .. }),
        } => {
            assert_eq!(position, base.position)
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_head_missing_or_damaged_in_the_primary_stalls_the_run_where_it_is() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let now = cluster.committed().await.incarnation;
    let rows = cluster
        .coordinator
        .commits_after(&cluster.key, now, base.position, 10)
        .await
        .unwrap();
    let after = &rows[0];
    cluster.blobs.delete(&after.head.0).await.unwrap();
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    let missing = Stall::PrimaryMissing {
        key: after.head.0.clone(),
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(base.position),
            stalled: Some(missing)
        }
    );

    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let rows = cluster
        .coordinator
        .commits_after(&cluster.key, now, base.position, 10)
        .await
        .unwrap();
    let after = &rows[0];
    cluster
        .blobs
        .put(&after.head.0, Bytes::from_static(b"not a frame"))
        .await
        .unwrap();
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    match ship(&shipper, &cluster, 100).await {
        Batch::Shipped {
            through: Some(through),
            stalled: Some(Stall::Corrupt { key, .. }),
        } => {
            assert_eq!(through, base.position);
            assert_eq!(key, after.head.0);
        }
        other => panic!("{other:?}"),
    }
}

/// A store that fails every put of a key holding `failing`, as a store
/// that cannot be reached for it does.
#[derive(Debug)]
struct Failing {
    inner: Arc<dyn Blobs>,
    failing: &'static str,
}

impl Failing {
    fn failed(&self, key: &str) -> Option<StorageError> {
        key.contains(self.failing).then(|| StorageError::Blob {
            key: key.into(),
            source: "the store failed".into(),
        })
    }
}

#[async_trait]
impl Blobs for Failing {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        match self.failed(key) {
            Some(error) => Err(error),
            None => self.inner.put(key, bytes).await,
        }
    }
    async fn put_new(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<Created> {
        match self.failed(key) {
            Some(error) => Err(error),
            None => self.inner.put_new(key, bytes).await,
        }
    }
    fn creates_at_most(&self) -> Option<usize> {
        self.inner.creates_at_most()
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        match self.failed(key) {
            Some(error) => Err(error),
            None => self.inner.put_parts(key, parts).await,
        }
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
async fn a_checkpoints_link_is_made_only_once_its_files_are_there() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let inner = a_backup();
    let failing: Arc<dyn Blobs> = Arc::new(Failing {
        inner: inner.clone(),
        failing: "/files/",
    });
    let shipper = shipper_of(&cluster, &failing);
    let error = cluster.committed().await;
    assert!(shipper.ship_batch(error.incarnation, 100).await.is_err());
    // Nothing of the checkpoint linked, and nothing noted shipped.
    assert!(inner
        .list(&format!(
            "{}links/",
            shipper.timeline(error.incarnation).prefix()
        ))
        .await
        .unwrap()
        .is_empty());
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!((shipping.shipped, shipping.stalled), (None, None));
    // Once it can be reached, the run goes on from its base.
    let shipper_again = Shipper::new(
        cluster.blobs.clone(),
        inner.clone(),
        cluster.coordinator.clone(),
        cluster.key.clone(),
        lineage(),
        cluster_id(),
    );
    let shipped = ship(&shipper_again, &cluster, 100).await;
    assert_eq!(
        shipped,
        Batch::Shipped {
            through: Some(error.log_position),
            stalled: None
        }
    );
    assert!(base.position <= error.log_position);
}

#[tokio::test]
async fn a_store_failing_part_way_ends_the_batch_with_what_it_shipped_noted() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let now = cluster.committed().await.incarnation;
    let inner = a_backup();
    let next = format!("links/{:020}.link", base.position.0 + 1);
    let failing: Arc<dyn Blobs> = Arc::new(Failing {
        inner: inner.clone(),
        failing: Box::leak(next.into_boxed_str()),
    });
    let shipper = shipper_of(&cluster, &failing);
    assert!(shipper.ship_batch(now, 100).await.is_err());
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!(
        (shipping.shipped, shipping.stalled),
        (Some(base.position), None)
    );
}

#[tokio::test]
async fn the_link_of_what_was_shipped_gone_from_the_backup_store_stalls_the_run() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    ship(&shipper, &cluster, 1).await;
    let now = cluster.committed().await.incarnation;
    backup
        .delete(&shipper.timeline(now).link(base.position))
        .await
        .unwrap();
    let missing = Stall::LinkMissing {
        position: base.position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 10).await,
        Batch::Shipped {
            through: None,
            stalled: Some(missing)
        }
    );
}

#[tokio::test]
async fn a_stale_shipper_of_a_run_since_re_based_ships_nothing_more_and_stalls_nothing() {
    let (cluster, writer) = written().await;
    let base = begun(&cluster).await;
    let backup = a_backup();
    let stale = shipper_of(&cluster, &backup);
    ship(&stale, &cluster, 1).await;
    // A newer checkpoint, and the run re-based on it.
    writer.flush().await.unwrap();
    let now = cluster.committed().await.incarnation;
    let (at, checkpoint) = newest_checkpoint(&cluster).await;
    let raised = ShipBase {
        position: at,
        checkpoint,
        checkpoint_at: at,
    };
    cluster
        .coordinator
        .record_base(&cluster.key, now, &raised)
        .await
        .unwrap();
    assert!(at > base.position);
    // The stale shipper reads the run afresh: it ships the new run, from
    // its base, not the old one's.
    let shipped = ship(&stale, &cluster, 100).await;
    assert_eq!(
        shipped,
        Batch::Shipped {
            through: Some(at),
            stalled: None
        }
    );
}

/// The shipper of `cluster` over a coordinator whose commit rows the test
/// gives, and that coordinator.
fn shipper_over_rows(cluster: &Cluster, backup: &Arc<dyn Blobs>) -> (Shipper, Arc<Unreliable>) {
    let rows = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let shipper = Shipper::new(
        cluster.blobs.clone(),
        backup.clone(),
        rows.clone(),
        cluster.key.clone(),
        lineage(),
        cluster_id(),
    );
    (shipper, rows)
}

/// The commit rows of `cluster` after its base.
async fn rows_after(cluster: &Cluster, base: &ShipBase) -> Vec<RestorePoint> {
    let now = cluster.committed().await.incarnation;
    cluster
        .coordinator
        .commits_after(&cluster.key, now, base.position, 10)
        .await
        .unwrap()
}

#[tokio::test]
async fn a_run_whose_base_has_no_row_stalls_before_shipping_anything() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let backup = a_backup();
    let (shipper, rows) = shipper_over_rows(&cluster, &backup);
    rows.rows.lock().unwrap().insert(base.position, None);
    let stalled = Stall::BaseRowMissing {
        position: base.position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: None,
            stalled: Some(stalled)
        }
    );
    assert!(backup.list("").await.unwrap().is_empty());
}

#[tokio::test]
async fn a_row_naming_another_positions_head_or_none_of_the_partitions_stalls_the_run() {
    for named in ["another position", "another partition"] {
        let (cluster, _writer) = written().await;
        let base = begun(&cluster).await;
        let after = rows_after(&cluster, &base).await;
        let backup = a_backup();
        let (shipper, rows) = shipper_over_rows(&cluster, &backup);
        let mut row = after[1].clone();
        row.head = match named {
            "another position" => after[0].head.clone(),
            _ => ManifestRef(row.head.0.replace("tenant-a", "tenant-b")),
        };
        rows.rows
            .lock()
            .unwrap()
            .insert(row.position, Some(row.clone()));
        let stalled = Stall::NotTheRowsHead {
            position: row.position,
        };
        assert_eq!(
            ship(&shipper, &cluster, 100).await,
            Batch::Shipped {
                through: Some(after[0].position),
                stalled: Some(stalled)
            },
            "{named}"
        );
    }
}

#[tokio::test]
async fn a_log_skipping_a_position_or_made_on_another_head_stalls_the_run() {
    // A row gone from between two logs: the later one skips it.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let after = rows_after(&cluster, &base).await;
    let backup = a_backup();
    let (shipper, rows) = shipper_over_rows(&cluster, &backup);
    rows.rows.lock().unwrap().insert(after[0].position, None);
    let stalled = Stall::NotTheRowsHead {
        position: after[1].position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(base.position),
            stalled: Some(stalled)
        }
    );

    // A log of the right position, made on another head than the row's
    // before it.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let after = rows_after(&cluster, &base).await;
    let log = LogSegment {
        position: after[1].position,
        prev: Some(after[0].head.0.replace(".log", "-other.log")),
        ops: Vec::new(),
        answered: Vec::new(),
    };
    cluster
        .blobs
        .put(&after[1].head.0, format::encode(Kind::Log, &log).unwrap())
        .await
        .unwrap();
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    let stalled = Stall::NotTheRowsHead {
        position: after[1].position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(after[0].position),
            stalled: Some(stalled)
        }
    );

    // Another position's frame under the row's head.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let after = rows_after(&cluster, &base).await;
    let frame = cluster.blobs.get(&after[0].head.0).await.unwrap();
    cluster.blobs.put(&after[1].head.0, frame).await.unwrap();
    let backup = a_backup();
    let shipper = shipper_of(&cluster, &backup);
    let stalled = Stall::NotTheRowsHead {
        position: after[1].position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(after[0].position),
            stalled: Some(stalled)
        }
    );
}

#[tokio::test]
async fn a_commit_made_before_the_one_before_it_stalls_the_run() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let after = rows_after(&cluster, &base).await;
    let backup = a_backup();
    let (shipper, rows) = shipper_over_rows(&cluster, &backup);
    let mut row = after[1].clone();
    row.at = after[0].at - chrono::Duration::microseconds(1);
    rows.rows
        .lock()
        .unwrap()
        .insert(row.position, Some(row.clone()));
    let stalled = Stall::TimeWentBack {
        position: row.position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(after[0].position),
            stalled: Some(stalled)
        }
    );

    // Against the commit a batch before shipped, read from its link.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let after = rows_after(&cluster, &base).await;
    let backup = a_backup();
    let (shipper, rows) = shipper_over_rows(&cluster, &backup);
    ship(&shipper, &cluster, 1).await;
    let mut row = after[0].clone();
    row.at = after[0].at - chrono::Duration::days(1);
    rows.rows
        .lock()
        .unwrap()
        .insert(row.position, Some(row.clone()));
    let stalled = Stall::TimeWentBack {
        position: row.position,
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: None,
            stalled: Some(stalled)
        }
    );
}

/// A store that answers so many creates `AlreadyThere` with nothing there,
/// as a 409 for another create of the key, under way and failed since.
#[derive(Debug)]
struct FalselyThere {
    inner: Arc<dyn Blobs>,
    creates: AtomicUsize,
}

#[async_trait]
impl Blobs for FalselyThere {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn put_new(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<Created> {
        let falsely = self
            .creates
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |more| {
                more.checked_sub(1)
            });
        match falsely {
            Ok(_) => Ok(Created::AlreadyThere),
            Err(_) => self.inner.put_new(key, bytes).await,
        }
    }
    fn creates_at_most(&self) -> Option<usize> {
        self.inner.creates_at_most()
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        self.inner.put_parts(key, parts).await
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
async fn a_link_said_to_be_there_and_not_there_when_read_is_created_again_a_few_times() {
    // Twice said there, and then made.
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let inner = a_backup();
    let falsely: Arc<dyn Blobs> = Arc::new(FalselyThere {
        inner: inner.clone(),
        creates: AtomicUsize::new(2),
    });
    let shipper = shipper_of(&cluster, &falsely);
    let committed = cluster.committed().await;
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(committed.log_position),
            stalled: None
        }
    );
    let timeline = shipper.timeline(committed.incarnation);
    assert!(inner.get(&timeline.link(base.position)).await.is_ok());

    // Said there every time it is asked: the batch ends, the outcome
    // unknown, nothing noted and nothing stalled.
    let (cluster, _writer) = written().await;
    begun(&cluster).await;
    let falsely: Arc<dyn Blobs> = Arc::new(FalselyThere {
        inner: a_backup(),
        creates: AtomicUsize::new(usize::MAX),
    });
    let shipper = shipper_of(&cluster, &falsely);
    let now = cluster.committed().await.incarnation;
    assert!(shipper.ship_batch(now, 100).await.is_err());
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!((shipping.shipped, shipping.stalled), (None, None));
}

#[tokio::test]
async fn a_checkpoint_after_a_skip_is_made_on_the_row_before_it_not_the_position_before() {
    // A checkpoint whose position skips past the row before it, as a
    // restore's does: here the log before it, gone from the rows.
    let (cluster, writer) = written().await;
    let base = begun(&cluster).await;
    writer.flush().await.unwrap();
    let after = rows_after(&cluster, &base).await;
    let (skipped, checkpoint) = (&after[1], &after[2]);
    assert!(checkpoint.head.0.contains("/manifest/"), "{checkpoint:?}");
    let backup = a_backup();
    let (shipper, rows) = shipper_over_rows(&cluster, &backup);
    rows.rows.lock().unwrap().insert(skipped.position, None);
    let shipped = ship(&shipper, &cluster, 100).await;
    let through = Some(checkpoint.position);
    assert_eq!(
        shipped,
        Batch::Shipped {
            through,
            stalled: None
        }
    );
    let now = cluster.committed().await.incarnation;
    let links = links(&backup, &shipper.timeline(now)).await;
    assert!(!links.contains_key(&skipped.position));
    let before = links[&checkpoint.position].header.before.clone().unwrap();
    assert_eq!(before.position, after[0].position);
    assert_eq!(before.sum, links[&after[0].position].header.sum);
}

/// A partition as [`written`] makes it, in a primary store that reads and
/// writes files `part_bytes` at a time, with a log at 4 of 200 writes,
/// too large to go inline in [`a_small_backup`].
async fn written_large(part_bytes: usize) -> (Cluster, Writer) {
    let transfers = Transfers {
        part_bytes,
        ..Transfers::default()
    };
    let cluster = Cluster {
        blobs: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())).transferring(transfers)),
        ..Cluster::new().await
    };
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    let many = (0..200u8).map(|n| upsert("notes", n, n.into())).collect();
    writer.commit(many).await.unwrap();
    writer.commit(vec![upsert("notes", 201, 1)]).await.unwrap();
    (cluster, writer)
}

/// A backup store that creates files of 1 KiB at most: a checkpoint's
/// link, or a large log's, with its frame, is larger; a small log's is not.
fn a_small_backup() -> Arc<dyn Blobs> {
    backup_of(1024)
}

/// The log of 200 writes [`written_large`] makes, its row.
async fn the_large_log(cluster: &Cluster, base: &ShipBase) -> RestorePoint {
    let after = rows_after(cluster, base).await;
    after[0].clone()
}

#[tokio::test]
async fn a_frame_too_large_to_go_inline_is_spilled_first_and_then_linked() {
    let (cluster, _writer) = written_large(64).await;
    let base = begun(&cluster).await;
    let backup = a_small_backup();
    let shipper = shipper_of(&cluster, &backup);
    let committed = cluster.committed().await;
    let shipped = ship(&shipper, &cluster, 100).await;
    let through = Some(committed.log_position);
    assert_eq!(
        shipped,
        Batch::Shipped {
            through,
            stalled: None
        }
    );
    let timeline = shipper.timeline(committed.incarnation);
    let links = links(&backup, &timeline).await;
    let large = the_large_log(&cluster, &base).await;
    let small = LogPosition(large.position.0 + 1);
    // The checkpoint, from memory, and the large log, a part at a time,
    // spilled; the small log inline.
    let bodies: Vec<_> = [base.position, large.position, small]
        .iter()
        .map(|at| links.get(at).map(|link| link.header.body))
        .collect();
    let bodies: Vec<_> = bodies.into_iter().flatten().collect();
    assert_eq!(bodies, [Body::Spilled, Body::Spilled, Body::Inline]);
    for at in [base.position, large.position] {
        let link = &links[&at];
        assert_eq!(link.frame, None);
        let spilled = backup.get(&timeline.frame(&link.header.key)).await.unwrap();
        assert_eq!(spilled, cluster.blobs.get(&link.header.key).await.unwrap());
        assert_eq!(link::Sum::of(&spilled), link.header.sum);
        assert_eq!(spilled.len() as u64, link.header.size);
    }
    assert!(links[&base.position].header.files.len() >= 2);
    assert_eq!(
        links[&large.position]
            .header
            .before
            .as_ref()
            .unwrap()
            .position,
        base.position
    );
    // Restoring the newest commit needs what is there, the checkpoint
    // read from its spilled frame.
    let headers: BTreeMap<_, _> = links
        .iter()
        .map(|(at, link)| (*at, link.header.clone()))
        .collect();
    let checkpoint = &links[&base.position].header;
    let frame = backup.get(&timeline.frame(&checkpoint.key)).await.unwrap();
    let Ok(Stored::Manifest(manifest)) = decode_head(&checkpoint.key, &frame) else {
        panic!("a manifest")
    };
    let needed = chain::needed(&headers, committed.log_position, Some(&manifest)).unwrap();
    assert_eq!(needed.checkpoint, Some(base.position));
}

#[tokio::test]
async fn a_link_too_large_for_the_store_even_spilled_stalls_the_run() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let backup = backup_of(256);
    let shipper = shipper_of(&cluster, &backup);
    match ship(&shipper, &cluster, 100).await {
        Batch::Shipped {
            through: None,
            stalled: Some(Stall::NotCreated { position, why }),
        } => {
            assert_eq!(position, base.position);
            assert!(why.contains("header"), "{why}");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn a_spill_that_fails_makes_no_link_and_ends_the_batch() {
    let (cluster, _writer) = written_large(64).await;
    let base = begun(&cluster).await;
    let inner = a_small_backup();
    let failing: Arc<dyn Blobs> = Arc::new(Failing {
        inner: inner.clone(),
        failing: "/frames/",
    });
    let shipper = shipper_of(&cluster, &failing);
    let now = cluster.committed().await.incarnation;
    assert!(shipper.ship_batch(now, 100).await.is_err());
    let timeline = shipper.timeline(now);
    assert!(inner
        .list(&format!("{}links/", timeline.prefix()))
        .await
        .unwrap()
        .is_empty());
    let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
    assert_eq!((shipping.shipped, shipping.stalled), (None, None));
    assert!(base.position > LogPosition(0));
}

/// `frame` put under the large log's key in the primary of `cluster`.
async fn the_large_log_made(cluster: &Cluster, base: &ShipBase, frame: Bytes) -> RestorePoint {
    let large = the_large_log(cluster, base).await;
    cluster.blobs.put(&large.head.0, frame).await.unwrap();
    large
}

/// The large log of `cluster`, decoded.
async fn the_large_log_decoded(cluster: &Cluster, base: &ShipBase) -> LogSegment {
    let large = the_large_log(cluster, base).await;
    let frame = cluster.blobs.get(&large.head.0).await.unwrap();
    match decode_head(&large.head.0, &frame).unwrap() {
        Stored::Log(log) => log,
        Stored::Manifest(_) => panic!("a log"),
    }
}

#[tokio::test]
async fn a_spilled_log_whose_checksum_fails_stalls_the_run_and_is_neither_linked_nor_left() {
    let (cluster, _writer) = written_large(64).await;
    let base = begun(&cluster).await;
    let large = the_large_log(&cluster, &base).await;
    let mut frame = cluster.blobs.get(&large.head.0).await.unwrap().to_vec();
    let last = frame.len() - 1;
    frame[last] ^= 0xff;
    the_large_log_made(&cluster, &base, frame.into()).await;
    let backup = a_small_backup();
    let shipper = shipper_of(&cluster, &backup);
    match ship(&shipper, &cluster, 100).await {
        Batch::Shipped {
            through,
            stalled: Some(Stall::Corrupt { key, .. }),
        } => {
            assert_eq!(through, Some(base.position));
            assert_eq!(key, large.head.0);
        }
        other => panic!("{other:?}"),
    }
    let timeline = shipper.timeline(cluster.committed().await.incarnation);
    assert!(backup.get(&timeline.link(large.position)).await.is_err());
    assert!(backup.get(&timeline.frame(&large.head.0)).await.is_err());
}

#[tokio::test]
async fn a_spilled_log_of_another_position_or_made_on_another_head_stalls_the_run() {
    for broken in ["position", "prev"] {
        let (cluster, _writer) = written_large(64).await;
        let base = begun(&cluster).await;
        let mut log = the_large_log_decoded(&cluster, &base).await;
        match broken {
            "position" => log.position = LogPosition(log.position.0 + 5),
            _ => log.prev = Some(format!("{}-other", log.prev.unwrap())),
        }
        let frame = format::encode(Kind::Log, &log).unwrap();
        let large = the_large_log_made(&cluster, &base, frame).await;
        let backup = a_small_backup();
        let shipper = shipper_of(&cluster, &backup);
        let stalled = Stall::NotTheRowsHead {
            position: large.position,
        };
        assert_eq!(
            ship(&shipper, &cluster, 100).await,
            Batch::Shipped {
                through: Some(base.position),
                stalled: Some(stalled)
            },
            "{broken}"
        );
        let timeline = shipper.timeline(cluster.committed().await.incarnation);
        assert!(
            backup.get(&timeline.link(large.position)).await.is_err(),
            "{broken}"
        );
    }
}

#[tokio::test]
async fn a_commit_spilled_by_one_shipper_and_inline_by_another_is_one_commit() {
    let (cluster, _writer) = written_large(64).await;
    let base = begun(&cluster).await;
    let large = the_large_log(&cluster, &base).await;
    // Shipped spilled, into a small store, noting nothing.
    let small = a_small_backup();
    let (spilling, unnoting) = shipper_over_rows(&cluster, &small);
    unnoting.unnoted.store(true, Ordering::SeqCst);
    ship(&spilling, &cluster, 100).await;
    // And again, as by a shipper that crashed before noting it: the same
    // frames put again, and the same links found there.
    let committed = cluster.committed().await.log_position;
    assert_eq!(
        ship(&spilling, &cluster, 100).await,
        Batch::Shipped {
            through: Some(committed),
            stalled: None
        }
    );
    let now = cluster.committed().await.incarnation;
    let spilled = links(&small, &spilling.timeline(now))
        .await
        .remove(&large.position)
        .unwrap();
    assert_eq!(spilled.header.body, Body::Spilled);
    // That link in a store a shipper would ship it inline to: the same
    // commit, shipped.
    let backup = a_backup();
    put_link(&backup, &spilled.header, None).await;
    let shipper = shipper_of(&cluster, &backup);
    let committed = cluster.committed().await.log_position;
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(committed),
            stalled: None
        }
    );
}

#[tokio::test]
async fn above_its_base_a_link_that_differs_only_in_the_commit_before_it_is_a_fork() {
    let (cluster, _writer) = written().await;
    let base = begun(&cluster).await;
    let next = LogPosition(base.position.0 + 1);
    let mine = link_at(&cluster, next).await;
    let mut other = mine.header.clone();
    other.before = None;
    let backup = a_backup();
    put_link(&backup, &other, mine.frame.as_deref()).await;
    let shipper = shipper_of(&cluster, &backup);
    let fork = Stall::Fork {
        position: next,
        fields: vec!["before"],
    };
    assert_eq!(
        ship(&shipper, &cluster, 100).await,
        Batch::Shipped {
            through: Some(base.position),
            stalled: Some(fork)
        }
    );
}

/// A primary store that fails every read of a key holding `failing` as a
/// store that cannot be reached for it does: not as one without the file.
#[derive(Debug)]
struct Unreadable {
    inner: Arc<dyn Blobs>,
    failing: &'static str,
}

#[async_trait]
impl Blobs for Unreadable {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        if key.contains(self.failing) {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "the store could not be reached".into(),
            });
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
async fn a_primary_that_cannot_be_read_ends_the_batch_and_stalls_nothing() {
    for failing in ["/log/", "/manifest/", "/segment/"] {
        let (cluster, _writer) = written().await;
        begun(&cluster).await;
        let backup = a_backup();
        let primary: Arc<dyn Blobs> = Arc::new(Unreadable {
            inner: cluster.blobs.clone(),
            failing,
        });
        let shipper = Shipper::new(
            primary,
            backup.clone(),
            cluster.coordinator.clone(),
            cluster.key.clone(),
            lineage(),
            cluster_id(),
        );
        let now = cluster.committed().await.incarnation;
        let failed = shipper.ship_batch(now, 100).await;
        assert!(
            matches!(failed, Err(StorageError::Blob { .. })),
            "{failing}: {failed:?}"
        );
        let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
        assert_eq!(shipping.stalled, None, "{failing}");
    }
}

#[tokio::test]
async fn a_run_re_based_during_a_batch_is_neither_noted_nor_stalled_by_it() {
    for stalling in [false, true] {
        let (cluster, writer) = written().await;
        let base = begun(&cluster).await;
        writer.flush().await.unwrap();
        let now = cluster.committed().await.incarnation;
        let (at, checkpoint) = newest_checkpoint(&cluster).await;
        let raised = ShipBase {
            position: at,
            checkpoint,
            checkpoint_at: at,
        };
        let backup = a_backup();
        let (batch, rows) = shipper_over_rows(&cluster, &backup);
        // The run is re-based as the batch reads its rows.
        *rows.re_base.lock().unwrap() = Some(raised.clone());
        if stalling {
            // And the batch would stall: a row it reads names no head.
            let after = rows_after(&cluster, &base).await;
            let mut row = after[1].clone();
            row.head = after[0].head.clone();
            rows.rows.lock().unwrap().insert(row.position, Some(row));
        }
        let refused = batch.ship_batch(now, 100).await;
        assert!(
            matches!(
                refused,
                Err(StorageError::Coordinator(
                    CoordinatorError::OtherBase { .. }
                ))
            ),
            "{stalling}: {refused:?}"
        );
        let shipping = cluster.coordinator.shipping(&cluster.key).await.unwrap();
        assert_eq!(shipping.base, Some(raised.clone()), "{stalling}");
        assert_eq!(
            (shipping.shipped, shipping.stalled),
            (None, None),
            "{stalling}"
        );
        // What the batch left in the store, the new run takes as its own.
        let shipper = shipper_of(&cluster, &backup);
        let committed = cluster.committed().await.log_position;
        assert_eq!(
            ship(&shipper, &cluster, 100).await,
            Batch::Shipped {
                through: Some(committed),
                stalled: None
            },
            "{stalling}"
        );
    }
}
