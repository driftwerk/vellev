//! A partition put back from files another store holds, as a backup does,
//! by the keys they had there: copied in under keys of its own, and loaded
//! from those copies alone (DW-680).

mod common;

use common::*;
use futures::FutureExt;
use vellev_storage::backups::Held;
use vellev_storage::blobs::Linked;
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
/// files are in hand at once: from when one is asked for, which takes a
/// moment, until its copy has gone up (see [`IndexesPut`]).
#[derive(Debug)]
struct Source {
    inner: Arc<dyn Blobs>,
    keys: BTreeSet<String>,
    whole: Mutex<Vec<String>>,
    in_parts: Mutex<Vec<String>>,
    indexes: Arc<AtomicUsize>,
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
            indexes: Arc::default(),
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
    restored_under(cluster, writer, frozen, &Holding(true)).await
}

/// As [`restored_from`], under the right `held`.
async fn restored_under(
    cluster: &Cluster,
    writer: &Writer,
    frozen: &Frozen,
    held: &dyn Held,
) -> vellev_storage::Result<LogPosition> {
    let restore = restore(cluster, frozen.at);
    let ready: RestoreFrom<'_> = writer
        .restore_from(
            frozen.as_source(),
            frozen.head.as_ref(),
            restore,
            None,
            held,
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

/// A store that, as each index file's copy goes up, which takes a moment,
/// tells the source it came from that it is no longer in hand.
#[derive(Debug)]
struct IndexesPut {
    inner: Arc<dyn Blobs>,
    in_hand: Arc<AtomicUsize>,
}

#[async_trait]
impl Blobs for IndexesPut {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        let put = self.inner.put(key, bytes).await;
        if key.contains("/index/") {
            tokio::time::sleep(Duration::from_millis(5)).await;
            self.in_hand.fetch_sub(1, Ordering::SeqCst);
        }
        put
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

/// Index files, which are read whole and held twice over while they are
/// given their copy's key, are copied one at a time: no second is asked
/// for until the first has gone up.
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
    let plain = Cluster::new().await;
    let target = Cluster {
        blobs: Arc::new(IndexesPut {
            inner: plain.blobs.clone(),
            in_hand: frozen.source.indexes.clone(),
        }),
        ..plain
    };
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

/// Notes written and flushed five times, the first three segments merged
/// into one: a source of several segments, each with its vectors.
async fn of_several_segments(cluster: &Cluster) {
    let writer = cluster.writer().await.merging_with(Merging {
        fanout: 3,
        ..Merging::default()
    });
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    for n in 2..6 {
        writer
            .commit(vec![upsert("notes", n, i64::from(n))])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    writer.commit(vec![upsert("notes", 6, 6)]).await.unwrap();
}

/// A source whose files stop coming from the `after`th asked for a part
/// at a time on.
#[derive(Debug)]
struct Losing {
    inner: Arc<dyn Blobs>,
    after: usize,
    asked: AtomicUsize,
}

#[async_trait]
impl Blobs for Losing {
    async fn put(&self, key: &str, _bytes: Bytes) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    fn parts<'a>(
        &'a self,
        key: &'a str,
    ) -> futures::stream::BoxStream<'a, vellev_storage::Result<Bytes>> {
        use futures::StreamExt;
        if self.asked.fetch_add(1, Ordering::SeqCst) >= self.after {
            let gone = StorageError::Missing {
                key: key.to_string(),
            };
            return futures::stream::once(async move { Err(gone) }).boxed();
        }
        self.inner.parts(key)
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        Err(only_read(prefix))
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        Err(only_read(key))
    }
}

/// A restore from a source that stops giving its files part way through
/// the copies, at each file in turn, commits nothing, leaves none of the
/// copies it made, and takes writes again.
#[tokio::test]
async fn a_restore_from_a_source_that_fails_part_way_leaves_nothing() {
    let source = Cluster::new().await;
    of_several_segments(&source).await;
    let frozen = frozen(&source).await;
    let streamed = 2 * frozen.checkpoint.as_ref().unwrap().segments.len();
    for after in 0..streamed {
        let target = Cluster::new().await;
        let writer = lived_in(&target).await;
        let before = target.committed().await;
        let had = keys(&target).await;
        let losing = Arc::new(Losing {
            inner: frozen.source.clone(),
            after,
            asked: AtomicUsize::new(0),
        });
        let restore = restore(&target, frozen.at);
        let head = frozen.head.as_ref();
        let outcome = match writer
            .restore_from(losing, head, restore, None, &Holding(true))
            .await
        {
            Ok(ready) => ready.commit().await.map(|_| ()),
            Err(error) => Err(error),
        };

        assert!(
            matches!(outcome, Err(StorageError::Missing { .. })),
            "{after}: {outcome:?}"
        );
        assert_eq!(target.committed().await, before, "{after}");
        let started = std::time::Instant::now();
        loop {
            let now = keys(&target).await;
            let left: Vec<&String> = now.difference(&had).collect();
            if left.is_empty() {
                break;
            }
            let waited = started.elapsed();
            assert!(waited < Duration::from_secs(30), "{after}: left {left:?}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        writer.commit(vec![upsert("notes", 7, 7)]).await.unwrap();
    }
}

/// A store that, once told, never answers the `at`th request made of it
/// from then on, and says so when it comes to it; and that notes each
/// request made of it.
#[derive(Debug)]
struct Stopping {
    inner: Arc<dyn Blobs>,
    armed: AtomicBool,
    made: AtomicUsize,
    at: usize,
    stopped: tokio::sync::Notify,
    asked: Mutex<Vec<String>>,
}

impl Stopping {
    async fn asked(&self, what: &str, key: &str) {
        if !self.armed.load(Ordering::SeqCst) {
            return;
        }
        self.asked.lock().unwrap().push(format!("{what} {key}"));
        if self.made.fetch_add(1, Ordering::SeqCst) == self.at {
            self.armed.store(false, Ordering::SeqCst);
            self.stopped.notify_one();
            std::future::pending::<()>().await;
        }
    }
}

#[async_trait]
impl Blobs for Stopping {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.asked("put", key).await;
        self.inner.put(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        self.asked("put_parts", key).await;
        self.inner.put_parts(key, parts).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.asked("get", key).await;
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.asked("list", prefix).await;
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
    fn parts<'a>(
        &'a self,
        key: &'a str,
    ) -> futures::stream::BoxStream<'a, vellev_storage::Result<Bytes>> {
        self.inner.parts(key)
    }
    async fn fetch_local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        self.asked("fetch_local", key).await;
        self.inner.fetch_local(key).await
    }
}

/// A restore from a source given up at each request it makes of the
/// partition's store in turn, by a caller that stops waiting there: the
/// partition is as it was, or restored, and opens; nothing of what it put
/// is left that the head does not need, but where the checkpoint itself
/// was put, whose commit may then have been made; and writes are taken.
#[tokio::test]
async fn a_restore_from_a_source_given_up_at_any_point_leaves_no_copies_unneeded() {
    let source = Cluster::new().await;
    of_several_segments(&source).await;
    let frozen = Arc::new(frozen(&source).await);
    let mut at = 0;
    loop {
        let plain = Cluster::new().await;
        let stopping = Arc::new(Stopping {
            inner: plain.blobs.clone(),
            armed: AtomicBool::new(false),
            made: AtomicUsize::new(0),
            at,
            stopped: tokio::sync::Notify::new(),
            asked: Mutex::default(),
        });
        let target = Cluster {
            blobs: stopping.clone(),
            ..plain
        };
        let writer = Arc::new(lived_in(&target).await);
        let before = target.committed().await;
        let rows_before = target.open().await.partition().read(ns);
        let had: BTreeSet<String> = (stopping.inner.list("").await.unwrap())
            .into_iter()
            .map(|file| file.key)
            .collect();
        let restore = restore(&target, frozen.at);
        stopping.armed.store(true, Ordering::SeqCst);
        let restoring = tokio::spawn({
            let (writer, frozen) = (writer.clone(), frozen.clone());
            async move {
                let head = frozen.head.as_ref();
                let ready = writer
                    .restore_from(frozen.as_source(), head, restore, None, &Holding(true))
                    .await?;
                ready.commit().await
            }
        });
        let stopped = tokio::time::timeout(Duration::from_secs(30), async {
            while !restoring.is_finished() && stopping.armed.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        stopped.await.expect("the restore stops or ends");
        if stopping.armed.load(Ordering::SeqCst) {
            restoring.await.unwrap().unwrap();
            assert!(at > 10, "made {at} requests in all");
            break;
        }
        let what = stopping.asked.lock().unwrap().last().cloned().unwrap();
        restoring.abort();
        let _ = restoring.await;
        tokio::time::sleep(Duration::from_millis(100)).await;

        let after = target.committed().await;
        let opened = target.open_on(stopping.inner.clone()).await;
        let opened = opened.unwrap_or_else(|error| panic!("{at} ({what}): {error}"));
        let read = opened.partition().read(ns);
        let needed = Partition::files(target.key.clone(), stopping.inner.clone(), &after);
        let needed: BTreeSet<String> = needed.await.unwrap().into_iter().collect();
        let now: BTreeSet<String> = (stopping.inner.list("").await.unwrap())
            .into_iter()
            .map(|file| file.key)
            .collect();
        let litter: Vec<&String> = now
            .difference(&had)
            .filter(|key| !needed.contains(*key))
            .collect();
        match after == before {
            true => {
                assert_eq!(read, rows_before, "{at} ({what})");
                let asked = stopping.asked.lock().unwrap();
                let tried = asked.iter().any(|asked| asked.contains("/manifest/"));
                assert!(tried || litter.is_empty(), "{at} ({what}): {litter:?}");
            }
            false => assert_eq!(read, frozen.rows, "{at} ({what})"),
        }
        if writer.commit(vec![upsert("notes", 7, 7)]).await.is_err() {
            let again = target.writer().await;
            again.commit(vec![upsert("notes", 7, 7)]).await.unwrap();
        }
        at += 1;
        assert!(at < 400, "too many requests");
    }
}

/// A right that runs out once it has been asked about `left` times.
struct RunningOut(AtomicUsize);

#[async_trait]
impl Held for RunningOut {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.still())
    }
    fn still(&self) -> bool {
        let left = self.0.load(Ordering::SeqCst);
        self.0.store(left.saturating_sub(1), Ordering::SeqCst);
        left > 0
    }
}

/// A restore from a source asks whether its right still holds before it
/// reads the source's head and before each file it copies: once the right
/// runs out, it puts no other file and commits nothing.
#[tokio::test]
async fn a_restore_from_a_source_whose_right_runs_out_copies_no_file_after() {
    let source = Cluster::new().await;
    of_several_segments(&source).await;
    let frozen = frozen(&source).await;
    let files = 2 * frozen.checkpoint.as_ref().unwrap().segments.len();
    for left in 0..=files {
        let plain = Cluster::new().await;
        let counted = Arc::new(Counted {
            inner: plain.blobs.clone(),
            gets: Mutex::default(),
        });
        let puts = Arc::new(AtomicUsize::new(0));
        let target = Cluster {
            blobs: Arc::new(PutsCounted {
                inner: counted,
                puts: puts.clone(),
            }),
            ..plain
        };
        let writer = lived_in(&target).await;
        let before = target.committed().await;
        puts.store(0, Ordering::SeqCst);
        let held = RunningOut(AtomicUsize::new(left));
        let outcome = writer
            .restore_from(
                frozen.as_source(),
                frozen.head.as_ref(),
                restore(&target, frozen.at),
                None,
                &held,
            )
            .await;

        assert!(matches!(outcome, Err(StorageError::Unheld(_))), "{left}");
        // The head is read once the right is asked about the first time,
        // and each file is put once it is asked about again.
        let put = puts.load(Ordering::SeqCst);
        assert!(put < left.max(1), "{left}: {put} files put");
        assert_eq!(target.committed().await, before, "{left}");
    }
}

/// A store that counts the files put into it, whole or a part at a time.
#[derive(Debug)]
struct PutsCounted {
    inner: Arc<dyn Blobs>,
    puts: Arc<AtomicUsize>,
}

#[async_trait]
impl Blobs for PutsCounted {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.puts.fetch_add(1, Ordering::SeqCst);
        self.inner.put(key, bytes).await
    }
    async fn put_parts(
        &self,
        key: &str,
        parts: futures::stream::BoxStream<'_, vellev_storage::Result<Bytes>>,
    ) -> vellev_storage::Result<()> {
        self.puts.fetch_add(1, Ordering::SeqCst);
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

/// DW-680, planted bug 13, on the path a restore of a tenant from its own
/// backup takes: every segment the backup lists is one the live state
/// holds. A copy that comes out wrong still fails the restore, which
/// commits nothing: a segment the live state holds is taken in its place
/// only once its copy is seen, streamed back, to be what the backup sent.
#[tokio::test]
async fn a_restore_from_its_own_backup_whose_copy_comes_out_wrong_fails_and_commits_nothing() {
    let plain = Cluster::new().await;
    let target = Cluster {
        blobs: Arc::new(Garbling(plain.blobs.clone())),
        ..plain
    };
    let writer = lived_in(&target).await;
    let frozen = frozen(&target).await;
    let before = target.committed().await;

    let outcome = restored_from(&target, &writer, &frozen).await;

    assert!(outcome.is_err(), "{outcome:?}");
    assert_eq!(target.committed().await, before);
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
}

/// Five segments flushed apart, none merged, and a note written after.
async fn of_five_segments(cluster: &Cluster) {
    let writer = cluster.writer().await;
    writer.commit(vec![create("notes")]).await.unwrap();
    for n in 1..=5 {
        writer
            .commit(vec![upsert("notes", n, i64::from(n))])
            .await
            .unwrap();
        writer.flush().await.unwrap();
    }
    writer.commit(vec![upsert("notes", 6, 6)]).await.unwrap();
}

/// A store that counts the segment files being read at once, each taking
/// a moment, whole or a part at a time, and the most at any one time.
#[derive(Debug)]
struct Crowded {
    inner: Arc<dyn Blobs>,
    reading: Arc<AtomicUsize>,
    most: AtomicUsize,
}

impl Crowded {
    fn over(inner: Arc<dyn Blobs>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            reading: Arc::default(),
            most: AtomicUsize::new(0),
        })
    }

    /// One more segment file being read, until what this gives is dropped.
    fn begun(&self) -> Reading {
        let now = self.reading.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(now, Ordering::SeqCst);
        Reading(self.reading.clone())
    }
}

struct Reading(Arc<AtomicUsize>);

impl Drop for Reading {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl Blobs for Crowded {
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
        if !key.contains("/segment/") {
            return self.inner.get(key).await;
        }
        let _reading = self.begun();
        tokio::time::sleep(Duration::from_millis(20)).await;
        self.inner.get(key).await
    }
    fn parts<'a>(
        &'a self,
        key: &'a str,
    ) -> futures::stream::BoxStream<'a, vellev_storage::Result<Bytes>> {
        use futures::StreamExt;
        if !key.contains("/segment/") {
            return self.inner.parts(key);
        }
        let reading = self.begun();
        let parts = self.inner.parts(key);
        let slow = futures::stream::once(tokio::time::sleep(Duration::from_millis(20)));
        let parts = slow
            .flat_map(move |()| futures::stream::empty())
            .chain(parts);
        parts
            .map(move |part| {
                let _held = &reading;
                part
            })
            .boxed()
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.inner.delete(key).await
    }
}

/// A restore decodes the segments it loads from its copies two at a time,
/// so that what it holds beside the live state stays small; an open of
/// the same partition reads them all at once, as it always has.
#[tokio::test]
async fn a_restore_decodes_its_copies_two_at_a_time_and_an_open_all_at_once() {
    let source = Cluster::new().await;
    of_five_segments(&source).await;
    let frozen = frozen(&source).await;
    assert_eq!(frozen.checkpoint.as_ref().unwrap().segments.len(), 5);
    let plain = Cluster::new().await;
    let crowded = Crowded::over(plain.blobs.clone());
    let target = Cluster {
        blobs: crowded.clone(),
        ..plain
    };
    let writer = lived_in(&target).await;

    restored_from(&target, &writer, &frozen).await.unwrap();
    assert_eq!(crowded.most.load(Ordering::SeqCst), 2, "a restore");

    crowded.most.store(0, Ordering::SeqCst);
    assert_eq!(target.open().await.partition().read(ns), frozen.rows);
    assert!(crowded.most.load(Ordering::SeqCst) > 2, "an open");
}

/// A target over a cache in a directory of its own.
async fn cached(dir: &std::path::Path) -> (Cluster, Arc<CachedBlobs>) {
    let plain = Cluster::new().await;
    let cached = Arc::new(CachedBlobs::new(plain.blobs.clone(), dir));
    let target = Cluster {
        blobs: cached.clone(),
        ..plain
    };
    (target, cached)
}

/// The files of `kind` the committed head names.
async fn named(cluster: &Cluster, kind: &str) -> Vec<String> {
    let committed = cluster.committed().await;
    let files = Partition::files(cluster.key.clone(), cluster.blobs.clone(), &committed);
    let files = files.await.unwrap().into_iter();
    files.filter(|key| key.contains(kind)).collect()
}

/// A restore reads the segments it copied without keeping a copy of them
/// on local disk, which loading them needs none of; the vectors, read in
/// place, it does keep.
#[tokio::test]
async fn a_restore_keeps_no_copy_of_a_segment_on_local_disk() {
    let source = Cluster::new().await;
    of_five_segments(&source).await;
    let frozen = frozen(&source).await;
    let dir = tempfile::tempdir().unwrap();
    let (target, cache) = cached(dir.path()).await;
    let writer = lived_in(&target).await;

    restored_from(&target, &writer, &frozen).await.unwrap();

    let restored = |key: &String| key.contains("-restored.");
    let segments: Vec<String> = named(&target, "/segment/")
        .await
        .into_iter()
        .filter(restored)
        .collect();
    assert_eq!(segments.len(), 5);
    for segment in &segments {
        assert!(cache.local(segment).await.unwrap().is_none(), "{segment}");
    }
    let vectors: Vec<String> = named(&target, "/vectors/")
        .await
        .into_iter()
        .filter(restored)
        .collect();
    assert_eq!(vectors.len(), 5);
    for vectors in &vectors {
        assert!(cache.local(vectors).await.unwrap().is_some(), "{vectors}");
    }
}

/// A tenant restored from its own backup gives each vectors file it takes
/// as held the copy on local disk it has, under the new key: no copy is
/// fetched again, and the old key's going leaves the new key's.
#[cfg(unix)]
#[tokio::test]
async fn a_restore_from_its_own_backup_links_the_vectors_it_holds_on_local_disk() {
    use std::os::unix::fs::MetadataExt;
    let dir = tempfile::tempdir().unwrap();
    let (target, cache) = cached(dir.path()).await;
    let writer = lived_in(&target).await;
    let old = named(&target, "/vectors/").await;
    assert!(!old.is_empty());
    let frozen = frozen(&target).await;

    restored_from(&target, &writer, &frozen).await.unwrap();

    let restored = |key: &String| key.contains("-restored.");
    let new: Vec<String> = named(&target, "/vectors/")
        .await
        .into_iter()
        .filter(restored)
        .collect();
    assert_eq!(new.len(), old.len());
    let inode = |key: &String| std::fs::metadata(dir.path().join(key)).unwrap().ino();
    let old_inodes: BTreeSet<u64> = old.iter().map(inode).collect();
    for key in &new {
        assert!(
            old_inodes.contains(&inode(key)),
            "{key} is a copy of its own"
        );
    }
    for key in &old {
        cache.delete(key).await.unwrap();
    }
    for key in &new {
        assert!(cache.local(key).await.unwrap().is_some(), "{key}");
    }
    assert_eq!(target.open().await.partition().read(ns), frozen.rows);
}

/// A store over a cache whose fetch of each vectors file to local disk
/// fails the first `times` times as `failure` makes it fail.
#[derive(Debug)]
struct SlowDisk {
    inner: Arc<CachedBlobs>,
    failure: Failure,
    times: usize,
    /// How many times each file was fetched.
    fetched: Mutex<BTreeMap<String, usize>>,
    /// Whether any fetch has failed.
    failed: Arc<AtomicBool>,
}

impl SlowDisk {
    fn over(plain: &Cluster, dir: &std::path::Path, failure: Failure, times: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(CachedBlobs::new(plain.blobs.clone(), dir)),
            failure,
            times,
            fetched: Mutex::default(),
            failed: Arc::default(),
        })
    }

    /// The most times any one file was fetched.
    fn most_fetched(&self) -> usize {
        let fetched = self.fetched.lock().unwrap();
        fetched.values().copied().max().unwrap_or(0)
    }
}

#[async_trait]
impl Blobs for SlowDisk {
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
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        self.inner.local(key).await
    }
    async fn fetch_local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        let fetched = {
            let mut fetched = self.fetched.lock().unwrap();
            let fetched = fetched.entry(key.to_string()).or_default();
            *fetched += 1;
            *fetched
        };
        if fetched <= self.times {
            self.failed.store(true, Ordering::SeqCst);
            return Err((self.failure)(key));
        }
        self.inner.fetch_local(key).await
    }
    async fn link_local(&self, from: &str, to: &str) -> vellev_storage::Result<Linked> {
        self.inner.link_local(from, to).await
    }
}

/// How a fetch fails, of the file under the key it is given.
type Failure = fn(&str) -> StorageError;

/// A local disk that was slow once: what failed to be fetched to it is
/// fetched again and the restore goes on. Any other failure, of the
/// store or of what was fetched, is not tried again: it fails the restore.
#[tokio::test]
async fn a_restore_tries_a_fetch_to_a_slow_local_disk_again_and_nothing_else() {
    let slow: Failure = |key| StorageError::Blob {
        key: key.to_string(),
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "local disk did not answer within 10.8s",
        )),
    };
    let unanswered: Failure =
        |key| vellev_storage::blobs::unanswered(key, Duration::from_secs(187));
    let other_disk: Failure = |key| StorageError::Blob {
        key: key.to_string(),
        source: Box::new(std::io::Error::other("no space left")),
    };
    let wrapped: Failure = |key| {
        let inner = StorageError::Blob {
            key: key.to_string(),
            source: Box::new(std::io::Error::new(std::io::ErrorKind::TimedOut, "slow")),
        };
        StorageError::Shared(Arc::new(inner))
    };
    let cases: [(&str, Failure, bool); 4] = [
        ("a local disk that timed out", slow, true),
        ("a store that did not answer", unanswered, false),
        ("a local disk otherwise failing", other_disk, false),
        ("a time-out not the failure's own", wrapped, false),
    ];
    let source = Cluster::new().await;
    of_five_segments(&source).await;
    let frozen = frozen(&source).await;
    for (what, failure, tried_again) in cases {
        let dir = tempfile::tempdir().unwrap();
        let plain = Cluster::new().await;
        let slow = SlowDisk::over(&plain, dir.path(), failure, 1);
        let target = Cluster {
            blobs: slow.clone(),
            ..plain
        };
        let writer = lived_in(&target).await;
        let before = target.committed().await;

        let outcome = restored_from(&target, &writer, &frozen).await;

        assert_eq!(outcome.is_ok(), tried_again, "{what}: {outcome:?}");
        let fetched = if tried_again { 2 } else { 1 };
        assert_eq!(slow.most_fetched(), fetched, "{what}");
        if !tried_again {
            assert_eq!(target.committed().await, before, "{what}");
        }
    }
}

/// A right held until a fetch has failed, if `failed` is a store's.
struct HeldUntil(Arc<AtomicBool>);

#[async_trait]
impl Held for HeldUntil {
    async fn held(&self) -> vellev_storage::Result<bool> {
        Ok(self.still())
    }
    fn still(&self) -> bool {
        !self.0.load(Ordering::SeqCst)
    }
}

/// A fetch to a local disk that does not answer in time is tried three
/// times in all, and each time again only while the restore's right
/// holds; a disk slow every time, or a right lost, fails the restore,
/// which commits nothing.
#[tokio::test]
async fn a_restore_tries_a_slow_local_disk_three_times_in_all_while_its_right_holds() {
    let slow: Failure = |key| StorageError::Blob {
        key: key.to_string(),
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "local disk did not answer within 10.8s",
        )),
    };
    let source = Cluster::new().await;
    of_five_segments(&source).await;
    let frozen = frozen(&source).await;
    let cases = [
        ("slow twice", 2, false, true, 3),
        ("slow three times", 3, false, false, 3),
        ("its right lost once slow", 1, true, false, 1),
    ];
    for (what, times, lost, restored, fetched) in cases {
        let dir = tempfile::tempdir().unwrap();
        let plain = Cluster::new().await;
        let disk = SlowDisk::over(&plain, dir.path(), slow, times);
        let target = Cluster {
            blobs: disk.clone(),
            ..plain
        };
        let writer = lived_in(&target).await;
        let before = target.committed().await;
        let failed = match lost {
            true => disk.failed.clone(),
            false => Arc::default(),
        };

        let outcome = restored_under(&target, &writer, &frozen, &HeldUntil(failed)).await;

        assert_eq!(outcome.is_ok(), restored, "{what}: {outcome:?}");
        if lost {
            assert!(
                matches!(outcome, Err(StorageError::Unheld(_))),
                "{what}: {outcome:?}"
            );
        }
        assert_eq!(disk.most_fetched(), fetched, "{what}");
        if !restored {
            assert_eq!(target.committed().await, before, "{what}");
        }
    }
}
