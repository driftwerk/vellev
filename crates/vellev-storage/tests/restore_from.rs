//! A partition put back from files another store holds, as a backup does,
//! by the keys they had there: copied in under keys of its own, and loaded
//! from those copies alone (DW-680).

mod common;

use common::*;
use futures::FutureExt;
use vellev_storage::backups::Held;
use vellev_storage::coordinator::{Clock, Restore};
use vellev_storage::{Merging, Partition, RestoreFrom};

/// The objects every history here writes some of.
const ALL: [u8; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

fn ns(state: &State) -> Vec<Option<i64>> {
    ALL.iter()
        .map(|object| n_of(state, "notes", *object))
        .collect()
}

/// A right that is held, or one that has run out.
struct Holding(bool);

#[async_trait]
impl Held for Holding {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.0)
    }
    fn still(&self) -> bool {
        self.0
    }
}

/// What another store holds of a partition, as a backup's source gives
/// it: read by the keys the files had, and never written. What is read
/// whole and what a part at a time is noted, and so is how many index
/// files are being read at once, each of which takes a moment.
#[derive(Debug)]
struct Source {
    inner: Arc<dyn Blobs>,
    keys: BTreeSet<String>,
    whole: Mutex<Vec<String>>,
    in_parts: Mutex<Vec<String>>,
    indexes: AtomicUsize,
    most_indexes: AtomicUsize,
}

impl Source {
    fn read_whole(&self, kind: &str) -> Vec<String> {
        let whole = self.whole.lock().unwrap();
        whole
            .iter()
            .filter(|key| key.contains(kind))
            .cloned()
            .collect()
    }
}

fn only_read(key: &str) -> StorageError {
    StorageError::Blob {
        key: key.to_string(),
        source: "a source is only read from".into(),
    }
}

#[async_trait]
impl Blobs for Source {
    async fn put(&self, key: &str, _bytes: Bytes) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
    async fn put_parts(
        &self,
        key: &str,
        _parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.whole.lock().unwrap().push(key.to_string());
        if key.contains("/index/") {
            let now = self.indexes.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_indexes.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(5)).await;
            self.indexes.fetch_sub(1, Ordering::SeqCst);
        }
        self.inner.get(key).await
    }
    fn parts<'a>(
        &'a self,
        key: &'a str,
    ) -> futures::stream::BoxStream<'a, vellev_storage::Result<Bytes>> {
        self.in_parts.lock().unwrap().push(key.to_string());
        self.inner.parts(key)
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        Err(only_read(prefix))
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
}

/// A partition's head as a backup of it holds it: the files it is loaded
/// from, copied into a store of their own, the rows it held and where.
struct Frozen {
    source: Arc<Source>,
    head: Option<ManifestRef>,
    at: LogPosition,
    rows: Vec<Option<i64>>,
    /// The checkpoint the head rests on, as it was.
    checkpoint: Option<Manifest>,
}

impl Frozen {
    fn as_source(&self) -> Arc<dyn Blobs> {
        self.source.clone()
    }
}

/// What `cluster` holds at its head now, frozen.
async fn frozen(cluster: &Cluster) -> Frozen {
    let committed = cluster.committed().await;
    let files = Partition::files(cluster.key.clone(), cluster.blobs.clone(), &committed)
        .await
        .unwrap();
    let inner: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    for key in &files {
        let bytes = cluster.blobs.get(key).await.unwrap();
        inner.put(key, bytes).await.unwrap();
    }
    let checkpoint = checkpoint_of(&*inner, committed.manifest.as_ref()).await;
    let checkpoint = match checkpoint {
        Some(key) => Some(read_manifest(&*inner, &key).await),
        None => None,
    };
    let rows = cluster.open().await.partition().read(ns);
    Frozen {
        source: Arc::new(Source {
            inner,
            keys: files.into_iter().collect(),
            whole: Mutex::default(),
            in_parts: Mutex::default(),
            indexes: AtomicUsize::new(0),
            most_indexes: AtomicUsize::new(0),
        }),
        head: committed.manifest,
        at: committed.log_position,
        rows,
        checkpoint,
    }
}

async fn read_manifest(blobs: &dyn Blobs, key: &str) -> Manifest {
    format::decode(key, Kind::Manifest, &blobs.get(key).await.unwrap()).unwrap()
}

/// The checkpoint the state at `head` rests on, as `blobs` holds it.
async fn checkpoint_of(blobs: &dyn Blobs, head: Option<&ManifestRef>) -> Option<String> {
    let mut cursor = head.map(|head| head.0.clone());
    while let Some(key) = cursor {
        match format::decode_head(&key, &blobs.get(&key).await.unwrap()).unwrap() {
            format::Head::Log(log) => cursor = log.prev,
            format::Head::Manifest(_) => return Some(key),
        }
    }
    None
}

fn restore(cluster: &Cluster, from: LogPosition) -> Restore {
    Restore {
        request: RequestId::generate(),
        asked: [0; 32],
        from,
        from_at: cluster.clock.now(),
    }
}

/// Put `cluster`'s partition back as `frozen` holds it, through `writer`.
async fn restored_from(
    cluster: &Cluster,
    writer: &Writer,
    frozen: &Frozen,
) -> vellev_storage::Result<LogPosition> {
    let restore = restore(cluster, frozen.at);
    let ready: RestoreFrom<'_> = writer
        .restore_from(
            frozen.as_source(),
            frozen.head.as_ref(),
            restore,
            None,
            &Holding(true),
        )
        .await?;
    ready.commit().await
}

/// Every key the store holds.
async fn keys(cluster: &Cluster) -> BTreeSet<String> {
    let listed = cluster.blobs.list("").await.unwrap();
    listed.into_iter().map(|file| file.key).collect()
}

/// The position in the name of a partition's file.
fn position_of(key: &str) -> u64 {
    key.rsplit('/').next().unwrap()[..20].parse().unwrap()
}

/// Notes 9 and 8 written, flushed into a checkpoint, and then note 8
/// written again: the partition restored into, which has a checkpoint
/// of its own and a log after it.
async fn lived_in(cluster: &Cluster) -> Writer {
    let writer = cluster.writer().await;
    writer
        .commit(vec![
            create("notes"),
            upsert("notes", 9, 9),
            upsert("notes", 8, 8),
        ])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 8, 80)]).await.unwrap();
    writer
}

/// How a history leaves the partition a source is taken of.
type History = fn(Cluster) -> futures::future::BoxFuture<'static, Cluster>;

/// Notes 1 and 2 flushed into a checkpoint; notes 1 to 3 written and
/// flushed again, note 4 written while that flush was being built, which
/// its checkpoint then replays; and note 5 written after it.
fn a_flush_that_replays_a_write(cluster: Cluster) -> futures::future::BoxFuture<'static, Cluster> {
    async move {
        let mut cluster = cluster;
        let (paused, mut stored) = Paused::new(&mut cluster, "/segment/");
        let writer = Arc::new(cluster.writer().await);
        writer
            .commit(vec![
                create("notes"),
                upsert("notes", 1, 1),
                upsert("notes", 2, 2),
            ])
            .await
            .unwrap();
        writer.flush().await.unwrap();
        writer
            .commit(vec![upsert("notes", 1, 10), upsert("notes", 3, 3)])
            .await
            .unwrap();
        paused.armed.store(true, Ordering::SeqCst);
        let flushing = tokio::spawn({
            let writer = writer.clone();
            async move { writer.flush().await }
        });
        assert_eq!(stored.recv().await, Some("stored"));
        paused.armed.store(false, Ordering::SeqCst);
        writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
        paused.release.notify_one();
        flushing.await.unwrap().unwrap();
        writer.commit(vec![upsert("notes", 5, 5)]).await.unwrap();
        cluster
    }
    .boxed()
}

/// Notes written and flushed twice, the two segments merged, and one more
/// note written after.
fn a_merge(cluster: Cluster) -> futures::future::BoxFuture<'static, Cluster> {
    async move {
        let writer = cluster.writer().await.merging_with(Merging {
            fanout: 2,
            ..Merging::default()
        });
        writer
            .commit(vec![create("notes"), upsert("notes", 1, 1)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
        writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
        writer.flush().await.unwrap();
        assert!(writer.merge().await.unwrap().is_some());
        writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
        cluster
    }
    .boxed()
}

/// Notes written, flushed and written again, the partition put back to
/// before the last write, and one more written after.
fn a_restore(cluster: Cluster) -> futures::future::BoxFuture<'static, Cluster> {
    async move {
        let writer = cluster.writer().await;
        writer
            .commit(vec![create("notes"), upsert("notes", 1, 1)])
            .await
            .unwrap();
        writer.flush().await.unwrap();
        writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
        let back = cluster.committed().await;
        writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
        writer
            .restore(back.manifest.as_ref(), back.log_position)
            .await
            .unwrap();
        writer.commit(vec![upsert("notes", 4, 4)]).await.unwrap();
        cluster
    }
    .boxed()
}

/// Nothing ever committed.
fn nothing(cluster: Cluster) -> futures::future::BoxFuture<'static, Cluster> {
    async move { cluster }.boxed()
}

/// DW-680's rule, and the checkpoint a restore from a source commits, for
/// each kind of head a source may be at: what the partition reads after
/// is what the source held; the checkpoint is past where the partition and
/// the source were; it rests on the partition's own checkpoint; it names
/// the source's segments by keys drawn for the restore, at its own
/// position, in the source's order, and after them one more
/// of what the source's logs left in memory; it replays nothing; and the
/// restore put no key the source's files had.
#[tokio::test]
async fn a_restore_from_a_source_commits_a_checkpoint_of_its_own_copies_alone() {
    // What the source is, and whether its logs leave anything in memory.
    let sources: [(&str, History, bool); 4] = [
        (
            "a flush that replays a write",
            a_flush_that_replays_a_write,
            true,
        ),
        ("a merge", a_merge, true),
        ("a restore", a_restore, true),
        ("no head", nothing, false),
    ];
    for (what, history, flushed) in sources {
        let source = history(Cluster::new().await).await;
        let frozen = frozen(&source).await;
        match what {
            "a flush that replays a write" => {
                let checkpoint = frozen.checkpoint.as_ref().unwrap();
                assert!(!checkpoint.replay.is_empty(), "{what}: replays a log");
                assert!(checkpoint.prev.is_some(), "{what}");
                let prev = checkpoint.prev.as_ref().unwrap();
                assert!(
                    !frozen.source.keys.contains(prev),
                    "{what}: before the source"
                );
            }
            "a merge" => {
                let checkpoint = frozen.checkpoint.as_ref().unwrap();
                let merged = checkpoint.segments.iter();
                assert!(merged
                    .into_iter()
                    .any(|segment| segment.key.contains("-merged")));
            }
            "a restore" => {
                let checkpoint = frozen.checkpoint.as_ref().unwrap();
                assert!(checkpoint.replay.is_empty(), "{what}");
                let prev = checkpoint.prev.as_ref().unwrap();
                assert!(
                    !frozen.source.keys.contains(prev),
                    "{what}: before the source"
                );
            }
            _ => assert!(frozen.head.is_none(), "{what}"),
        }

        let target = Cluster::new().await;
        let writer = lived_in(&target).await;
        let before = target.committed().await;
        let rested_on = checkpoint_of(&*target.blobs, before.manifest.as_ref()).await;
        let had = keys(&target).await;

        let restored = restored_from(&target, &writer, &frozen).await.unwrap();

        let r = before.log_position.max(frozen.at).next().unwrap();
        assert_eq!(restored, r, "{what}");
        let committed = target.committed().await;
        assert_eq!(committed.log_position, r, "{what}");
        let head = committed.manifest.unwrap().0;
        let manifest = read_manifest(&*target.blobs, &head).await;
        assert_eq!(manifest.position, r, "{what}");
        assert_eq!(manifest.prev, rested_on, "{what}: the partition's own");
        assert!(manifest.replay.is_empty(), "{what}");
        assert_eq!(manifest.replay_after, LogPosition::default(), "{what}");
        let copied = frozen
            .checkpoint
            .as_ref()
            .map_or(Vec::new(), |checkpoint| checkpoint.segments.clone());
        let more = usize::from(flushed);
        assert_eq!(manifest.segments.len(), copied.len() + more, "{what}");
        for (copy, of) in manifest.segments.iter().zip(&copied) {
            assert_ne!(copy.key, of.key, "{what}");
            assert!(
                copy.key.ends_with("-restored.segment"),
                "{what}: {}",
                copy.key
            );
            assert_eq!(position_of(&copy.key), r.0, "{what}: at the checkpoint's");
            assert_eq!(
                (copy.rows, copy.tombstones),
                (of.rows, of.tombstones),
                "{what}"
            );
            assert_eq!(copy.index.is_some(), of.index.is_some(), "{what}");
            assert_eq!(copy.vectors.is_some(), of.vectors.is_some(), "{what}");
        }
        if flushed {
            let last = manifest.segments.last().unwrap();
            assert!(!last.key.ends_with("-restored.segment"), "{what}");
            assert_eq!(position_of(&last.key), r.0, "{what}");
        }
        let now = keys(&target).await;
        let put: BTreeSet<&String> = now.difference(&had).collect();
        let named = manifest.segments.iter().flat_map(|segment| {
            [
                Some(&segment.key),
                segment.index.as_ref(),
                segment.vectors.as_ref(),
            ]
            .into_iter()
            .flatten()
        });
        for key in named {
            assert!(
                put.contains(key),
                "{what}: names {key}, which it did not put"
            );
        }
        let again: Vec<_> = put
            .iter()
            .filter(|key| frozen.source.keys.contains(**key))
            .collect();
        assert!(again.is_empty(), "{what}: put {again:?}, the keys they had");
        assert_eq!(
            target.open().await.partition().read(ns),
            frozen.rows,
            "{what}"
        );
        assert_eq!(writer.partition().read(ns), frozen.rows, "{what}");
    }
}

/// A tenant put back from its own backup taken a moment ago: the store has
/// every file the backup lists, under the keys they had, and the backup's
/// checkpoint is the very one the partition rests on. The restore copies
/// them all the same, and the checkpoint it commits rests on that one.
#[tokio::test]
async fn a_restore_from_a_partitions_own_backup_copies_what_the_store_has_still() {
    let cluster = Cluster::new().await;
    let writer = lived_in(&cluster).await;
    let frozen = frozen(&cluster).await;
    let had = keys(&cluster).await;
    assert!(frozen.source.keys.is_subset(&had));
    let rested_on = checkpoint_of(&*cluster.blobs, frozen.head.as_ref()).await;
    assert!(frozen.source.keys.contains(rested_on.as_ref().unwrap()));

    restored_from(&cluster, &writer, &frozen).await.unwrap();

    let head = cluster.committed().await.manifest.unwrap().0;
    let manifest = read_manifest(&*cluster.blobs, &head).await;
    assert_eq!(manifest.prev, rested_on);
    for segment in &manifest.segments {
        assert!(
            !frozen.source.keys.contains(&segment.key),
            "{}",
            segment.key
        );
        assert!(!had.contains(&segment.key), "{}", segment.key);
    }
    assert_eq!(cluster.open().await.partition().read(ns), frozen.rows);
}

/// A partition that never committed, put back from a source: its
/// checkpoint rests on nothing.
#[tokio::test]
async fn a_restore_into_a_partition_that_never_committed_rests_on_nothing() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let target = Cluster::new().await;
    let writer = target.writer().await;
    assert!(target.committed().await.manifest.is_none());

    let restored = restored_from(&target, &writer, &frozen).await.unwrap();

    assert_eq!(restored, frozen.at.next().unwrap());
    let head = target.committed().await.manifest.unwrap().0;
    let manifest = read_manifest(&*target.blobs, &head).await;
    assert_eq!(manifest.position, restored);
    assert_eq!(manifest.prev, None);
    assert!(manifest.replay.is_empty());
    let copied = frozen.checkpoint.as_ref().unwrap().segments.len();
    assert_eq!(
        manifest.segments.len(),
        copied + 1,
        "and what its log wrote"
    );
    for segment in &manifest.segments {
        assert!(!frozen.source.keys.contains(&segment.key));
    }
    assert_eq!(target.open().await.partition().read(ns), frozen.rows);
}

/// A store that notes every file read from it whole.
#[derive(Debug)]
struct Counted {
    inner: Arc<dyn Blobs>,
    gets: Mutex<Vec<String>>,
}

impl Counted {
    fn fetched(&self) -> Vec<String> {
        std::mem::take(&mut self.gets.lock().unwrap())
    }
}

#[async_trait]
impl Blobs for Counted {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        self.inner.put_parts(key, parts).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.gets.lock().unwrap().push(key.to_string());
        self.inner.get(key).await
    }
    fn parts<'a>(
        &'a self,
        key: &'a str,
    ) -> futures::stream::BoxStream<'a, vellev_storage::Result<Bytes>> {
        self.inner.parts(key)
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// Segments, and their vectors, come from the source a part at a time and
/// go into the store so; and the vectors are loaded from the store's copy
/// on local disk, which it fetches a part at a time, never read whole.
#[tokio::test]
async fn a_restore_from_a_source_never_reads_a_segment_or_its_vectors_whole_from_it() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let dir = tempfile::tempdir().unwrap();
    let counted = Arc::new(Counted {
        inner: Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
        gets: Mutex::default(),
    });
    let target = Cluster {
        blobs: Arc::new(CachedBlobs::new(counted.clone(), dir.path())),
        ..Cluster::new().await
    };
    let writer = lived_in(&target).await;
    counted.fetched();

    restored_from(&target, &writer, &frozen).await.unwrap();

    assert_eq!(frozen.source.read_whole("/segment/"), Vec::<String>::new());
    assert_eq!(frozen.source.read_whole("/vectors/"), Vec::<String>::new());
    let in_parts = frozen.source.in_parts.lock().unwrap().clone();
    assert!(
        in_parts.iter().any(|key| key.contains("/vectors/")),
        "{in_parts:?}"
    );
    let whole: Vec<String> = counted.fetched();
    let vectors: Vec<&String> = whole
        .iter()
        .filter(|key| key.contains("/vectors/"))
        .collect();
    assert!(vectors.is_empty(), "read whole from the store: {vectors:?}");
    assert_eq!(target.open().await.partition().read(ns), frozen.rows);
}

/// Over 4,000 rows with vectors in each of `segments` segments, each of
/// which then has an index file, and one more row written after.
async fn indexed(cluster: &Cluster, segments: u32) {
    let writer = cluster.writer().await.merging_with(Merging {
        clustered_above: 4_000,
        ..Merging::default()
    });
    writer.commit(vec![create("notes")]).await.unwrap();
    for segment in 0..segments {
        let rows: Vec<u32> = (segment * 10_000..segment * 10_000 + 4_100).collect();
        for batch in rows.chunks(500) {
            let row = |n: &u32| {
                let angle = (n % 8) as f32 * std::f32::consts::FRAC_PI_4 + (n % 97) as f32 * 1e-3;
                Mutation::Upsert {
                    collection: name("notes"),
                    id: ObjectId(uuid::Uuid::from_u128(u128::from(*n) + 1_000)),
                    properties: json!({ "n": n }).as_object().unwrap().clone(),
                    vector: Some(vec![angle.cos(), angle.sin()]),
                    only_if_version: None,
                }
            };
            writer
                .commit(batch.iter().map(row).collect())
                .await
                .unwrap();
        }
        writer.flush().await.unwrap();
    }
    writer.commit(vec![upsert("notes", 1, 1)]).await.unwrap();
}

/// What the index file under `key` says it indexes, if it is one.
async fn indexes(blobs: &dyn Blobs, key: &str) -> Option<String> {
    let frame = blobs.get(key).await.ok()?;
    let index: SegmentIndex = format::decode(key, Kind::Index, &frame).ok()?;
    Some(index.segment)
}

/// A segment's index file is copied as the index of the segment's copy,
/// and is used: it says so. One the source holds that indexes another
/// segment, or that cannot be read as an index, is passed over, as every
/// load of the source passed it over, and the copy has none. One the
/// source lists and cannot give fails the restore, which commits nothing.
#[tokio::test]
async fn a_segments_index_file_is_copied_as_its_copys_or_passed_over_as_unusable() {
    let source = Cluster::new().await;
    indexed(&source, 1).await;
    let index = |frozen: &Frozen| {
        let checkpoint = frozen.checkpoint.as_ref().unwrap();
        let indexed = checkpoint
            .segments
            .iter()
            .find(|segment| segment.index.is_some());
        let segment = indexed.expect("a segment with an index file");
        (segment.key.clone(), segment.index.clone().unwrap())
    };
    // What becomes of the index file in the source, and whether the copy
    // then has one, or the restore fails.
    let cases: [(&str, Option<bool>); 4] = [
        ("as it is", Some(true)),
        ("of another segment", Some(false)),
        ("unreadable", Some(false)),
        ("gone", None),
    ];
    for (what, indexed) in cases {
        let frozen = frozen(&source).await;
        let (segment, key) = index(&frozen);
        let inner = &frozen.source.inner;
        match what {
            "of another segment" => {
                let frame = inner.get(&key).await.unwrap();
                let other = "tenant-a/0/segment/another.segment";
                let rekeyed = format::rekeyed_index(&key, &frame, &segment, other).unwrap();
                inner.put(&key, rekeyed).await.unwrap();
            }
            "unreadable" => inner
                .put(&key, Bytes::from_static(b"no index"))
                .await
                .unwrap(),
            "gone" => inner.delete(&key).await.unwrap(),
            _ => {}
        }
        let target = Cluster::new().await;
        let writer = lived_in(&target).await;
        let before = target.committed().await;

        let outcome = restored_from(&target, &writer, &frozen).await;

        let Some(indexed) = indexed else {
            assert!(
                matches!(outcome, Err(StorageError::Missing { .. })),
                "{what}: {outcome:?}"
            );
            assert_eq!(target.committed().await, before, "{what}");
            writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
            continue;
        };
        outcome.unwrap();
        let head = target.committed().await.manifest.unwrap().0;
        let manifest = read_manifest(&*target.blobs, &head).await;
        let copy = &manifest.segments[0];
        assert_eq!(copy.index.is_some(), indexed, "{what}");
        if let Some(copied) = &copy.index {
            assert_ne!(*copied, key);
            let said = indexes(&*target.blobs, copied).await;
            assert_eq!(said.as_ref(), Some(&copy.key), "{what}: it says its copy");
        }
        assert_eq!(
            target.open().await.partition().read(ns),
            frozen.rows,
            "{what}"
        );
    }
}

/// Index files, which are read whole and held twice over while they are
/// given their copy's key, are copied one at a time.
#[tokio::test]
async fn a_restore_from_a_source_copies_one_index_file_at_a_time() {
    let source = Cluster::new().await;
    indexed(&source, 2).await;
    let frozen = frozen(&source).await;
    let checkpoint = frozen.checkpoint.as_ref().unwrap();
    let with_index = checkpoint
        .segments
        .iter()
        .filter(|segment| segment.index.is_some());
    assert_eq!(with_index.count(), 2);
    let target = Cluster::new().await;
    let writer = lived_in(&target).await;

    restored_from(&target, &writer, &frozen).await.unwrap();

    assert_eq!(frozen.source.read_whole("/index/").len(), 2);
    assert_eq!(frozen.source.most_indexes.load(Ordering::SeqCst), 1);
}

/// A store whose copies of segments, put a part at a time, are each one
/// byte off.
#[derive(Debug)]
struct Garbling(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for Garbling {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.0.put(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        use futures::TryStreamExt;
        if !key.contains("/segment/") {
            return self.0.put_parts(key, parts).await;
        }
        let parts: Vec<Bytes> = parts.try_collect().await?;
        let mut whole = parts.concat();
        let last = whole.len() - 1;
        whole[last] ^= 1;
        self.0.put(key, Bytes::from(whole)).await
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

/// A copy that does not come out whole fails the restore, which commits
/// nothing: the state is loaded from what was copied, not from what was
/// read from the source.
#[tokio::test]
async fn a_restore_whose_copy_comes_out_wrong_fails_and_commits_nothing() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let plain = Cluster::new().await;
    let target = Cluster {
        blobs: Arc::new(Garbling(plain.blobs.clone())),
        ..plain
    };
    let writer = lived_in(&target).await;
    let before = target.committed().await;

    let outcome = restored_from(&target, &writer, &frozen).await;

    assert!(outcome.is_err(), "{outcome:?}");
    assert_eq!(target.committed().await, before);
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
}

/// Writes are refused from the moment a restore from a source is made
/// ready until it is committed, and taken again once it is dropped; and
/// what it copied, dropped, is deleted.
#[tokio::test]
async fn writes_are_refused_while_a_restore_from_a_source_is_ready_and_its_copies_go_if_it_is_dropped(
) {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let target = Cluster::new().await;
    let writer = lived_in(&target).await;
    let before = target.committed().await;
    let had = keys(&target).await;

    let ready = writer
        .restore_from(
            frozen.as_source(),
            frozen.head.as_ref(),
            restore(&target, frozen.at),
            None,
            &Holding(true),
        )
        .await
        .unwrap();
    assert!(keys(&target).await.len() > had.len(), "copied");
    assert!(matches!(
        writer.commit(vec![upsert("notes", 2, 2)]).await,
        Err(StorageError::Restoring)
    ));
    drop(ready);

    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    let started = std::time::Instant::now();
    loop {
        let now = keys(&target).await;
        let put: Vec<&String> = now
            .difference(&had)
            .filter(|key| !key.contains("/log/"))
            .collect();
        if put.is_empty() {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(30), "kept {put:?}");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        target.committed().await.log_position,
        before.log_position.next().unwrap()
    );
}

/// A restore from a source while the right to read it is not held stops
/// before it copies anything.
#[tokio::test]
async fn a_restore_from_a_source_no_longer_held_copies_nothing() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let target = Cluster::new().await;
    let writer = lived_in(&target).await;
    let had = keys(&target).await;

    let outcome = writer
        .restore_from(
            frozen.as_source(),
            frozen.head.as_ref(),
            restore(&target, frozen.at),
            None,
            &Holding(false),
        )
        .await;

    assert!(
        matches!(outcome, Err(StorageError::Unheld(_))),
        "{:?}",
        outcome.err()
    );
    assert_eq!(frozen.source.in_parts.lock().unwrap().len(), 0);
    assert_eq!(keys(&target).await, had);
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
}

/// A restore from a source whose commit fails commits nothing, and closes
/// the writer, as any commit that fails does.
#[tokio::test]
async fn a_restore_from_a_source_whose_commit_fails_commits_nothing() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let target = Cluster::new().await;
    let unreliable = Arc::new(Unreliable::over(target.coordinator.clone()));
    let writer = target.writer_on(unreliable.clone()).await;
    writer
        .commit(vec![create("notes"), upsert("notes", 9, 9)])
        .await
        .unwrap();
    let before = target.committed().await;
    unreliable.refuse_commits.store(10, Ordering::SeqCst);

    let outcome = restored_from(&target, &writer, &frozen).await;

    assert!(outcome.is_err(), "{outcome:?}");
    assert_eq!(target.committed().await, before);
    assert!(writer.is_closed());
}

/// A store whose first file put a part at a time is kept waiting until
/// let go, once it is told so.
#[derive(Debug)]
struct Waiting {
    inner: Arc<dyn Blobs>,
    waiting: tokio::sync::Notify,
    go_on: tokio::sync::Notify,
    armed: AtomicBool,
}

#[async_trait]
impl Blobs for Waiting {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        if self.armed.swap(false, Ordering::SeqCst) {
            self.waiting.notify_one();
            self.go_on.notified().await;
        }
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

/// A flush, and a clean that keeps nothing for its age, made while a
/// restore from a source copies its files, wait for it: the copies are
/// named at the position the partition is at, which no checkpoint has
/// passed while they are made, and none of them is taken.
#[tokio::test]
async fn a_flush_and_a_clean_wait_for_a_restore_from_a_source_and_take_none_of_its_copies() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = Arc::new(frozen(&source).await);
    let plain = Cluster::new().await;
    let waiting = Arc::new(Waiting {
        inner: plain.blobs.clone(),
        waiting: tokio::sync::Notify::new(),
        go_on: tokio::sync::Notify::new(),
        armed: AtomicBool::new(true),
    });
    let target = Cluster {
        blobs: waiting.clone(),
        ..plain
    };
    let writer = Arc::new(lived_in(&target).await);
    let restore = restore(&target, frozen.at);

    let restoring = tokio::spawn({
        let (writer, frozen) = (writer.clone(), frozen.clone());
        async move {
            let ready = writer
                .restore_from(
                    frozen.as_source(),
                    frozen.head.as_ref(),
                    restore,
                    None,
                    &Holding(true),
                )
                .await?;
            ready.commit().await
        }
    });
    let copying = tokio::time::timeout(Duration::from_secs(30), waiting.waiting.notified());
    copying.await.expect("a copy is made");
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!flushing.is_finished(), "a flush waits for the restore");
    assert!(!cleaning.is_finished(), "a clean waits for the restore");

    waiting.go_on.notify_one();
    restoring.await.unwrap().unwrap();
    flushing.await.unwrap().unwrap();
    cleaning.await.unwrap().unwrap();
    assert_eq!(target.open().await.partition().read(ns), frozen.rows);
}

/// A restore from a source made ready on a partition another writer then
/// takes and commits to, at the very position the restore's copies are
/// named at: the restore commits nothing, its copies lie under keys
/// nothing names, and the other writer's partition, cleaned up with no
/// grace, reads as it committed it.
#[tokio::test]
async fn a_restore_from_a_source_overtaken_at_its_own_position_leaves_only_litter() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let target = Cluster::new().await;
    let writer = lived_in(&target).await;
    let before = target.committed().await;
    let r = before.log_position.max(frozen.at).next().unwrap();
    let had = keys(&target).await;
    let ready = writer
        .restore_from(
            frozen.as_source(),
            frozen.head.as_ref(),
            restore(&target, frozen.at),
            None,
            &Holding(true),
        )
        .await
        .unwrap();
    let copies: Vec<String> = keys(&target).await.difference(&had).cloned().collect();
    assert!(
        copies.iter().all(|key| position_of(key) == r.0),
        "{copies:?}"
    );

    let other = target.writer().await;
    let mut position = before.log_position;
    while position < r {
        let n = 70 + position.0 as i64;
        other.commit(vec![upsert("notes", 7, n)]).await.unwrap();
        position = target.committed().await.log_position;
    }
    assert_eq!(position, r, "the other writer is where the copies are");
    assert!(ready.commit().await.is_err());
    assert_eq!(target.committed().await.log_position, r);

    other.clean(Duration::ZERO).await.unwrap();
    let read = target.open().await.partition().read(ns);
    assert_eq!(read, other.partition().read(ns));
    assert_eq!(read[6], Some(70 + r.0 as i64 - 1));
    let head = target.committed().await.manifest.unwrap().0;
    assert!(!copies.contains(&head));
}

/// A restore from a source whose commit is made and never answered, and
/// which its caller then stops waiting for, keeps its copies: the head it
/// committed names them.
#[tokio::test]
async fn a_restore_from_a_source_whose_commit_is_made_unanswered_keeps_its_copies() {
    let source = a_merge(Cluster::new().await).await;
    let frozen = frozen(&source).await;
    let target = Cluster::new().await;
    let unreliable = Arc::new(Unreliable::over(target.coordinator.clone()));
    let writer = target.writer_on(unreliable.clone()).await;
    writer
        .commit(vec![create("notes"), upsert("notes", 9, 9)])
        .await
        .unwrap();
    let before = target.committed().await;
    let ready = writer
        .restore_from(
            frozen.as_source(),
            frozen.head.as_ref(),
            restore(&target, frozen.at),
            None,
            &Holding(true),
        )
        .await
        .unwrap();
    unreliable.hang_after_commit.store(true, Ordering::SeqCst);

    let waited = tokio::time::timeout(Duration::from_secs(2), ready.commit()).await;

    assert!(waited.is_err(), "never answered");
    assert!(target.committed().await.log_position > before.log_position);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(target.open().await.partition().read(ns), frozen.rows);
}
