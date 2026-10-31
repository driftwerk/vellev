//! A merge's vectors gathered on local disk (node memory note, M8): into a
//! file of the store's cache, under a name of its own, read in place by the
//! build, uploaded from where they lie and taken up by the cache as the
//! file's copy once the store has it. So the frame is written once, and is
//! never on the heap. What a stop, a failure or a crash leaves of it is
//! nothing, and a merge on a store with no cache goes on as it did.
//!
//! Every rule says first that the merge did gather on disk: without that
//! each would hold of a merge that never touched it.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use futures::stream::BoxStream;
use object_store::path::Path as StorePath;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult, UploadPart,
};
use vellev_index::stop::Stage;
use vellev_storage::blobs::{disk_room, SpillsTold};
use vellev_storage::{Disk, Holding, MergeBuilds, MergeStage, Merging, Transfers};

/// Rows in each of the two segments merged, of [`WIDTH`] dimensions: their
/// vectors are several of the pieces a spill is written in.
const ROWS: u32 = 12_000;
const WIDTH: u32 = 64;

/// Rows written at once.
const WRITTEN_AT_ONCE: u32 = 2_000;

/// How long anything here is waited for, far more than it takes.
const LONG: Duration = Duration::from_secs(120);

/// How long a merge stopped may take to give up: the rule of merge_stop's
/// timed test.
const GIVEN_UP_WITHIN: Duration = Duration::from_secs(2);

/// What `work` comes to within `limit`.
async fn within<T>(limit: Duration, what: &str, work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(limit, work)
        .await
        .unwrap_or_else(|_| panic!("{what} within {limit:?}"))
}

/// Wait, as long as anything here is waited for, until `holds`.
async fn settled(what: &str, holds: impl Fn() -> bool) {
    within(LONG, what, async {
        while !holds() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
}

/// Every file under `dir`, by its path from there.
fn files(dir: &Path) -> BTreeSet<String> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => walk(&entry.path(), found),
                _ => found.push(entry.path()),
            }
        }
    }
    let mut found = Vec::new();
    walk(dir, &mut found);
    let from = |path: PathBuf| {
        let path = path.strip_prefix(dir).expect("under the directory");
        path.to_string_lossy().into_owned()
    };
    found.into_iter().map(from).collect()
}

/// The files under `dir` that are spills.
fn spills(dir: &Path) -> Vec<String> {
    let of = files(dir).into_iter();
    of.filter(|file| file.ends_with(".spill")).collect()
}

fn row(n: u32) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(u128::from(n) + 1_000))
}

/// The vector of row `n`, of `width` dimensions, whose values tell it from
/// any other row's.
fn vector(n: u32, width: u32) -> Vec<f32> {
    (0..width)
        .map(|at| ((n * 31 + at) as f32).sin() * 0.5)
        .collect()
}

fn upsert_of(n: u32, vector: Option<Vec<f32>>) -> Mutation {
    Mutation::Upsert {
        collection: name("notes"),
        id: row(n),
        properties: json!({ "title": format!("note {n} of many"), "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector,
        only_if_version: None,
    }
}

fn created(width: u32) -> Mutation {
    let mut notes = schema("notes");
    notes.embedding.dimensions = width;
    Mutation::CreateCollection(Box::new(notes))
}

/// What a test writes: rows in each of two segments, of `width`, the
/// rows that have vectors as `has` says, then those `dies` says deleted.
#[derive(Clone, Copy)]
struct Written {
    rows: u32,
    width: u32,
    has: fn(u32) -> bool,
    dies: fn(u32) -> bool,
    clustered_above: Option<usize>,
}

impl Written {
    const fn of(rows: u32, width: u32) -> Self {
        Self {
            rows,
            width,
            has: |_| true,
            dies: |_| false,
            clustered_above: None,
        }
    }
}

/// `writer`, among `builds`, of `written`'s two flushed segments.
async fn written_on(writer: Writer, builds: &Arc<MergeBuilds>, written: Written) -> Arc<Writer> {
    let merging = Merging {
        fanout: 2,
        clustered_above: written
            .clustered_above
            .unwrap_or(Merging::default().clustered_above),
        ..Merging::default()
    };
    let writer = Arc::new(writer.merging_with(merging).merging_among(builds.clone()));
    within(LONG, "the rows were written", async {
        let Written { rows, width, .. } = written;
        writer.commit(vec![created(width)]).await.unwrap();
        for segment in 0..2 {
            for from in (0..rows).step_by(WRITTEN_AT_ONCE as usize) {
                let to = (from + WRITTEN_AT_ONCE).min(rows);
                let batch = (from..to).map(|n| segment * rows + n).map(|n| {
                    let vector = (written.has)(n).then(|| vector(n, width));
                    upsert_of(n, vector)
                });
                writer.commit(batch.collect()).await.unwrap();
            }
            writer.flush().await.unwrap();
        }
        let deaths = (0..2 * rows).filter(|n| (written.dies)(*n));
        let deaths: Vec<u32> = deaths.collect();
        for batch in deaths.chunks(WRITTEN_AT_ONCE as usize) {
            let deletes = batch.iter().map(|n| Mutation::Delete {
                collection: name("notes"),
                id: row(*n),
                only_if_version: None,
            });
            writer.commit(deletes.collect()).await.unwrap();
        }
    })
    .await;
    writer
}

/// A cluster whose store keeps copies in a directory of its own, on `disk`
/// if given, with a merge's two segments in it.
struct Cached {
    cluster: Cluster,
    /// The store under the cache.
    store: Arc<dyn Blobs>,
    cache: tempfile::TempDir,
    builds: Arc<MergeBuilds>,
    writer: Arc<Writer>,
}

impl Cached {
    async fn of(written: Written) -> Self {
        Self::on(written, None).await
    }

    async fn on(written: Written, disk: Option<Disk>) -> Self {
        let cluster = Cluster::new().await;
        let store = cluster.blobs.clone();
        Self::over(cluster, store, written, disk).await
    }

    /// As [`Self::on`], the cache over `store`.
    async fn over(
        mut cluster: Cluster,
        store: Arc<dyn Blobs>,
        written: Written,
        disk: Option<Disk>,
    ) -> Self {
        let cache = tempfile::tempdir().unwrap();
        let cached = CachedBlobs::new(store.clone(), cache.path());
        cluster.blobs = Arc::new(match disk {
            Some(disk) => cached.on(disk),
            None => cached,
        });
        let builds = MergeBuilds::new(1);
        let writer = written_on(cluster.writer().await, &builds, written).await;
        Self {
            cluster,
            store,
            cache,
            builds,
            writer,
        }
    }

    fn dir(&self) -> &Path {
        self.cache.path()
    }

    /// The merge, on a task of its own.
    fn merging(&self) -> tokio::task::JoinHandle<vellev_storage::Result<Option<LogPosition>>> {
        let writer = self.writer.clone();
        tokio::spawn(async move { writer.merge().await })
    }

    /// Wait until a merge is held, as `self.builds` was told.
    async fn held(&self) {
        settled("the merge was never held", || self.builds.held() == 1).await;
    }

    /// What is committed, and every row of it as it should be.
    async fn all_there(&self, written: Written) {
        let opened = self.cluster.open().await;
        let (rows, width) = (written.rows, written.width);
        opened.partition().read(|state| {
            for n in 0..2 * rows {
                let has = (written.has)(n) && !(written.dies)(n);
                assert_eq!(
                    vector_in(state, n),
                    has.then(|| vector(n, width)),
                    "row {n}"
                );
            }
        });
    }
}

/// Row `n`'s vector as `state` holds it, if it has one and is live.
fn vector_in(state: &State, n: u32) -> Option<Vec<f32>> {
    let live = state.get(&name("notes"), row(n))?;
    live.vector.map(<[f32]>::to_vec)
}

/// A store that reads what it is given where a store's client reads it, on
/// the async thread that sends it, and counts the parts of a merge's vectors
/// file. A page of a spill's mapping gone from memory would hold that thread
/// for as long as the disk takes: so what lies in the mapping is taken to
/// be such a page, for as long as `told` says reading it takes.
#[derive(Debug)]
struct Client {
    inner: InMemory,
    told: Arc<SpillsTold>,
    parts: Arc<AtomicUsize>,
}

impl Client {
    fn over(told: Arc<SpillsTold>) -> Arc<Self> {
        Arc::new(Self {
            inner: InMemory::new(),
            told,
            parts: Arc::default(),
        })
    }
}

impl std::fmt::Display for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a client")
    }
}

/// Read what `payload` holds as `told` says the disk takes to.
fn read_as_told(told: &SpillsTold, payload: &PutPayload) {
    if payload.iter().any(|chunk| told.maps(chunk.as_ptr())) {
        let taking = told.reading.load(Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(taking));
    }
}

#[async_trait]
impl ObjectStore for Client {
    async fn put_opts(
        &self,
        location: &StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        read_as_told(&self.told, &payload);
        self.inner.put_opts(location, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        location: &StorePath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        let inner = self.inner.put_multipart_opts(location, opts).await?;
        Ok(Box::new(ClientParts {
            inner,
            told: self.told.clone(),
            parts: location
                .as_ref()
                .ends_with("-merged.vectors")
                .then(|| self.parts.clone()),
        }))
    }
    async fn get_opts(
        &self,
        location: &StorePath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }
    async fn delete(&self, location: &StorePath) -> object_store::Result<()> {
        self.inner.delete(location).await
    }
    fn list(
        &self,
        prefix: Option<&StorePath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }
    async fn list_with_delimiter(
        &self,
        prefix: Option<&StorePath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy(&self, from: &StorePath, to: &StorePath) -> object_store::Result<()> {
        self.inner.copy(from, to).await
    }
    async fn copy_if_not_exists(
        &self,
        from: &StorePath,
        to: &StorePath,
    ) -> object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

#[derive(Debug)]
struct ClientParts {
    inner: Box<dyn MultipartUpload>,
    told: Arc<SpillsTold>,
    parts: Option<Arc<AtomicUsize>>,
}

#[async_trait]
impl MultipartUpload for ClientParts {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        read_as_told(&self.told, &data);
        if let Some(parts) = &self.parts {
            parts.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.put_part(data)
    }
    async fn complete(&mut self) -> object_store::Result<PutResult> {
        self.inner.complete().await
    }
    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}

/// The merged files by what they are, as their names say. An index
/// names its segment, a name drawn at random, as any two merges'
/// differ: that name, and the checksum that sums it, are left blank.
async fn merged_files(store: &dyn Blobs, cluster: &Cluster) -> BTreeMap<String, Vec<u8>> {
    let listed = store.list(&Layout(&cluster.key).prefix()).await.unwrap();
    let keys = listed.into_iter().map(|file| file.key);
    let keys: Vec<String> = keys.filter(|key| key.contains("-merged")).collect();
    let segment = keys.iter().find(|key| key.ends_with(".segment")).unwrap();
    let mut frames = BTreeMap::new();
    for key in &keys {
        let kind = key.rsplit('.').next().unwrap().to_string();
        let mut frame = store.get(key).await.unwrap().to_vec();
        if kind == "index" {
            let name = segment.as_bytes();
            let at = frame.windows(name.len()).position(|bytes| bytes == name);
            let at = at.expect("an index names its segment");
            frame[at..at + name.len()].fill(0);
            frame[15..19].fill(0);
        }
        frames.insert(kind, frame);
    }
    frames
}

/// M8-R1, and that nothing persisted changes. The frames a merge uploads,
/// gathered on disk, are byte for byte those it uploads gathered in memory:
/// its vectors, its segment and its index, at 768 and 8 dimensions, rows
/// without a vector and dead rows among them, and the cache's copy is the
/// same bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_gathered_on_disk_uploads_the_frames_it_would_in_memory_byte_for_byte() {
    let holes = Written {
        has: |n| n % 7 != 0,
        dies: |n| n % 5 == 0,
        ..Written::of(3_000, 768)
    };
    let shapes = [
        holes,
        Written { width: 8, ..holes },
        Written {
            clustered_above: Some(4_000),
            has: |_| true,
            ..holes
        },
    ];
    for shape in shapes {
        let cluster = Cluster::new().await;
        let builds = MergeBuilds::new(1);
        let writer = written_on(cluster.writer().await, &builds, shape).await;
        within(LONG, "the merge", writer.merge()).await.unwrap();
        assert_eq!(writer.last_spilled(), 0, "gathered on no disk");
        let expected = merged_files(&*cluster.blobs, &cluster).await;
        assert!(expected.contains_key("vectors"), "{:?}", expected.keys());

        let cached = Cached::of(shape).await;
        let merge = within(LONG, "the merge", cached.writer.merge()).await;
        assert!(merge.unwrap().is_some());
        assert!(cached.writer.last_spilled() > 0, "gathered in memory");
        let (cluster, store) = (&cached.cluster, &*cached.store);
        let uploaded = merged_files(store, cluster).await;
        let copied = merged_files(&*cluster.blobs, cluster).await;
        assert_eq!(
            uploaded.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (kind, frame) in &expected {
            // Not `assert_eq!`: a frame is far too long to print.
            assert!(uploaded[kind] == *frame, "the uploaded {kind} differs");
            assert!(copied[kind] == *frame, "the cache's {kind} differs");
        }
        cached.all_there(shape).await;
    }
}

/// M8-R1 as production takes it: a frame larger than a part goes up in
/// parts, the store's own client reading them, byte for byte what a merge in
/// memory uploads, as a part lands on a whole number of the pieces it is read
/// in, and as it does not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_larger_than_a_part_goes_up_in_parts_byte_for_byte() {
    const MIB: usize = 1024 * 1024;
    let shape = Written {
        clustered_above: Some(4_000),
        ..Written::of(3_000, 768)
    };
    let cluster = Cluster::new().await;
    let builds = MergeBuilds::new(1);
    let writer = written_on(cluster.writer().await, &builds, shape).await;
    within(LONG, "the merge", writer.merge()).await.unwrap();
    let expected = merged_files(&*cluster.blobs, &cluster).await;
    let frame = expected["vectors"].len();
    for part in [4 * MIB, 8 * MIB, 5 * MIB, 7 * MIB] {
        assert!(frame > part, "{frame} bytes in parts of {part}");
        let transfers = Transfers {
            part_bytes: part,
            ..Transfers::default()
        };
        let told = Arc::new(SpillsTold::default());
        let client = Client::over(told.clone());
        let store: Arc<dyn Blobs> =
            Arc::new(ObjectBlobs::new(client.clone()).transferring(transfers));
        let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 8)
            .fetching(&Transfers {
                part_bytes: part,
                ..Transfers::default()
            })
            .spills_told(told);
        let cached = Cached::over(Cluster::new().await, store, shape, Some(disk)).await;
        let merge = within(LONG, "the merge", cached.writer.merge()).await;
        assert!(merge.unwrap().is_some());
        assert!(cached.writer.last_spilled() > 0, "gathered in memory");
        assert_eq!(
            client.parts.load(Ordering::SeqCst),
            frame.div_ceil(part),
            "{frame} bytes in parts of {part}"
        );
        let uploaded = merged_files(&*cached.store, &cached.cluster).await;
        for (kind, frame) in &expected {
            assert!(uploaded[kind] == *frame, "the uploaded {kind} differs");
        }
        cached.all_there(shape).await;
    }
}

/// The stages a merge past its gather is stopped in.
const STAGES: [MergeStage; 7] = [
    MergeStage::Indexing,
    MergeStage::Vectors(Stage::Sampling),
    MergeStage::Vectors(Stage::Coding),
    MergeStage::Encoding,
    MergeStage::Uploading,
    MergeStage::Loading,
    MergeStage::Summing,
];

/// Clustered, as the stages of a clustered build's stop are asked.
const STOPPED: Written = Written {
    clustered_above: Some(4_000),
    ..Written::of(3_000, WIDTH)
};

/// M8-R3. A merge stopped at any stage past its gather leaves no file of
/// its own in the cache directory, whatever the file had come to.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_at_any_stage_leaves_no_file_of_its_own_on_local_disk() {
    for stage in STAGES {
        let cached = Cached::of(STOPPED).await;
        let before = files(cached.dir());
        cached.writer.stop_merges_at(Some(stage));
        let merged = within(LONG, "the stopped merge", cached.writer.merge()).await;
        assert_eq!(merged.unwrap(), None, "stopped in {stage:?}");
        settled(&format!("stopped in {stage:?}, its files stay"), || {
            files(cached.dir()) == before
        })
        .await;
        assert!(cached.writer.last_spilled() > 0, "gathered in memory");

        cached.writer.stop_merges_at(None);
        let next = within(LONG, "the next merge", cached.writer.merge()).await;
        assert!(next.unwrap().is_some(), "after a stop in {stage:?}");
        cached.all_there(STOPPED).await;
    }
}

/// M8-R3. So too one stopped part way through its gather, with its spill
/// begun, heeded at the next chunk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_part_way_through_its_gather_leaves_no_file_of_its_own() {
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::of(written).await;
    let before = files(cached.dir());
    cached.builds.hold(Some(Holding::GatheringUntilLetGo));
    let merging = cached.merging();
    cached.held().await;
    assert_eq!(spills(cached.dir()).len(), 1, "no spill begun");
    cached.writer.stop_merges();
    cached.builds.hold(None);
    cached.builds.let_one_go();
    let merged = within(LONG, "the stopped merge", merging).await;
    assert_eq!(merged.unwrap().unwrap(), None);
    settled("its files stay", || files(cached.dir()) == before).await;
    cached.all_there(written).await;
}

/// M8-R3. A merge dropped, as a tenant let go drops its task, leaves none
/// either, once its work has ended: whether that is while it builds, or while
/// its file goes up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_dropped_leaves_no_file_of_its_own_on_local_disk() {
    for holding in [Holding::UntilLetGo, Holding::BuiltUntilLetGo] {
        let cached = Cached::of(STOPPED).await;
        let before = files(cached.dir());
        cached.builds.hold(Some(holding));
        let merging = cached.merging();
        cached.held().await;
        assert_eq!(spills(cached.dir()).len(), 1, "no spill begun");
        merging.abort();
        let _ = merging.await;
        cached.builds.hold(None);
        cached.builds.let_one_go();
        settled("its files stay", || files(cached.dir()) == before).await;
        cached.all_there(STOPPED).await;
    }

    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged.vectors");
    let store = cluster.blobs.clone();
    let cached = Cached::over(cluster, store, STOPPED, None).await;
    let before = files(cached.dir());
    paused.armed.store(true, Ordering::SeqCst);
    let merging = cached.merging();
    let stored = within(LONG, "the vectors went up", async {
        while seen.recv().await != Some("stored") {}
    });
    stored.await;
    assert_eq!(spills(cached.dir()).len(), 1, "no spill begun");
    merging.abort();
    let _ = merging.await;
    paused.release.notify_one();
    settled("its files stay", || files(cached.dir()) == before).await;
    cached.all_there(STOPPED).await;
}

/// M8-R3. A build that panics leaves none: what holds the spill goes as the
/// thread unwinds, and the next merge commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_whose_build_panics_leaves_no_file_of_its_own_on_local_disk() {
    let cached = Cached::of(STOPPED).await;
    let before = files(cached.dir());
    cached.writer.panic_merges_in_build(true);
    let ended = within(LONG, "the merge that panics", cached.merging()).await;
    assert!(matches!(ended, Err(_) | Ok(Err(_))), "{ended:?}");
    assert!(cached.writer.last_spilled() > 0, "gathered in memory");
    settled("its files stay", || files(cached.dir()) == before).await;
    cached.writer.panic_merges_in_build(false);
    let next = within(LONG, "the next merge", cached.writer.merge()).await;
    assert!(next.unwrap().is_some());
}

/// A store that looks in the cache as each merged vectors file is put, and
/// keeps every copy of the bytes it is given.
#[derive(Debug)]
struct Looking {
    inner: Arc<dyn Blobs>,
    cache: tempfile::TempDir,
    /// Refuse the merged vectors file.
    refuses: AtomicBool,
    /// How many times the merged vectors file was put.
    puts: AtomicUsize,
    /// Whether a copy of it was in the cache by the time it was put.
    copied_first: AtomicBool,
    /// What was put of it, kept past everything else that holds it.
    kept: Mutex<Vec<(String, Bytes)>>,
}

impl Looking {
    fn over(inner: Arc<dyn Blobs>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            cache: tempfile::tempdir().unwrap(),
            refuses: AtomicBool::new(false),
            puts: AtomicUsize::new(0),
            copied_first: AtomicBool::new(false),
            kept: Mutex::default(),
        })
    }

    /// A writer of `written`, the cache in `self`'s directory over `self`.
    async fn writer(
        self: &Arc<Self>,
        cluster: &mut Cluster,
        builds: &Arc<MergeBuilds>,
        written: Written,
    ) -> Arc<Writer> {
        let store: Arc<dyn Blobs> = self.clone();
        cluster.blobs = Arc::new(CachedBlobs::new(store, self.cache.path()));
        written_on(cluster.writer().await, builds, written).await
    }
}

#[async_trait]
impl Blobs for Looking {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if key.ends_with("-merged.vectors") {
            self.puts.fetch_add(1, Ordering::SeqCst);
            if self.cache.path().join(key).exists() {
                self.copied_first.store(true, Ordering::SeqCst);
            }
            self.kept.lock().unwrap().push((key.into(), bytes.clone()));
            if self.refuses.load(Ordering::SeqCst) {
                return Err(StorageError::Blob {
                    key: key.into(),
                    source: "refused, as a test asks".into(),
                });
            }
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

/// M8-R4, and invariant (iii). The cache takes the file up only once the
/// store has it: at the moment of the put no copy is there, and one the
/// store refuses leaves no file under its key, and no spill.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_file_is_taken_up_by_the_cache_only_once_the_store_has_it() {
    let mut cluster = Cluster::new().await;
    let looking = Looking::over(cluster.blobs.clone());
    let builds = MergeBuilds::new(1);
    let written = Written::of(ROWS, WIDTH);
    let writer = looking.writer(&mut cluster, &builds, written).await;
    let before = files(looking.cache.path());
    let manifest = cluster.committed().await.manifest;

    looking.refuses.store(true, Ordering::SeqCst);
    let refused = within(LONG, "the merge refused", writer.merge()).await;
    assert!(
        matches!(&refused, Err(StorageError::Blob { key, .. }) if key.contains("-merged")),
        "{refused:?}"
    );
    assert_eq!(looking.puts.load(Ordering::SeqCst), 1);
    assert!(writer.last_spilled() > 0, "gathered in memory");
    assert_eq!(
        spills(looking.cache.path()).len(),
        1,
        "bytes kept hold the spill they are of"
    );
    looking.kept.lock().unwrap().clear();
    settled("a file refused stays in the cache", || {
        files(looking.cache.path()) == before
    })
    .await;
    assert_eq!(cluster.committed().await.manifest, manifest);

    looking.refuses.store(false, Ordering::SeqCst);
    let merged = within(LONG, "the merge", writer.merge()).await;
    assert!(merged.unwrap().is_some());
    assert_eq!(looking.puts.load(Ordering::SeqCst), 2);
    assert!(
        !looking.copied_first.load(Ordering::SeqCst),
        "a copy was in the cache before the store had the file"
    );
    let cached = files(looking.cache.path());
    let copies = cached
        .iter()
        .filter(|file| file.ends_with("-merged.vectors"));
    assert_eq!(copies.count(), 1, "{cached:?}");
    assert!(spills(looking.cache.path()).is_empty());
}

/// M8-R4. So too a file put and given up as the cache takes it up, by a
/// merge dropped and so deleting what it uploaded: what takes it up looks
/// before and after, and leaves no copy of a file the store no longer has.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_dropped_as_its_spill_is_taken_up_leaves_no_copy() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 8).spills_told(told.clone());
    let cached = Cached::on(STOPPED, Some(disk)).await;
    let before = files(cached.dir());
    told.taken_up_in.store(1_000, Ordering::SeqCst);
    let merging = cached.merging();
    let prefix = Layout(&cached.cluster.key).prefix();
    within(LONG, "the vectors went up", async {
        loop {
            let listed = cached.store.list(&prefix).await.unwrap();
            if listed
                .iter()
                .any(|file| file.key.ends_with("-merged.vectors"))
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
    assert!(cached.writer.last_spilled() > 0, "gathered in memory");
    merging.abort();
    let _ = merging.await;
    settled("no spill goes", || spills(cached.dir()).is_empty()).await;
    // Past the time it took, with the delete of what was uploaded done.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    settled("its files stay", || files(cached.dir()) == before).await;
}

/// M8-R4, and the frame written once: the cache's copy is the very file the
/// merge gathered, renamed, not written again.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_cache_s_copy_is_the_file_gathered_not_written_again() {
    use std::os::unix::fs::MetadataExt;
    let cached = Cached::of(STOPPED).await;
    cached.builds.hold(Some(Holding::BuiltUntilLetGo));
    let merging = cached.merging();
    cached.held().await;
    let found = spills(cached.dir());
    assert_eq!(found.len(), 1, "no spill begun");
    let gathered = std::fs::metadata(cached.dir().join(&found[0]));
    let gathered = gathered.unwrap().ino();
    cached.builds.hold(None);
    cached.builds.let_one_go();
    let merged = within(LONG, "the merge", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    let all = files(cached.dir());
    let copy = all.iter().find(|file| file.ends_with("-merged.vectors"));
    let copy = std::fs::metadata(cached.dir().join(copy.expect("a copy kept")));
    assert_eq!(copy.unwrap().ino(), gathered, "written again");
}

/// M8-R6. A disk that fills as a merge gathers fails the merge there, before
/// any of it goes up: nothing committed or left, every row as it was, and,
/// once there is room, the next merge commits. So does a cache that cannot
/// begin a spill at all: as a segment whose copy the cache cannot keep fails
/// its build, not held in memory in its place.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_disk_fails_the_merge_where_it_gathers_and_leaves_nothing() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 8).spills_told(told.clone());
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::on(written, Some(disk)).await;
    let before = files(cached.dir());
    let manifest = cached.cluster.committed().await.manifest;
    told.at_most.store(1024 * 1024, Ordering::SeqCst);
    let full = within(LONG, "the merge on a full disk", cached.writer.merge()).await;
    assert!(matches!(&full, Err(StorageError::Blob { .. })), "{full:?}");
    settled("what it wrote stays", || files(cached.dir()) == before).await;
    let prefix = Layout(&cached.cluster.key).prefix();
    let uploaded = cached.store.list(&prefix).await.unwrap();
    assert!(
        !uploaded.iter().any(|file| file.key.contains("-merged")),
        "a merge that could not gather uploaded a file"
    );
    assert_eq!(cached.cluster.committed().await.manifest, manifest);
    cached.all_there(written).await;

    told.at_most.store(u64::MAX, Ordering::SeqCst);
    let merged = within(LONG, "the merge with room", cached.writer.merge()).await;
    assert!(merged.unwrap().is_some());
    assert!(cached.writer.last_spilled() > 0, "gathered in memory");
    cached.all_there(written).await;
}

/// Where CI's Test job mounts a small file system of its own for the rule
/// below.
const SMALL_DISK: &str = "VELLEV_SMALL_DISK";

/// The most a file system may hold for the rule below to fill it.
const SMALL_AT_MOST: u64 = 1024 * 1024 * 1024;

/// M8-R6, on a file system that really fills: the write tells of it, as the
/// error the merge fails on, never as a fault in a mapping, and the room it
/// took comes back with its file. Run in CI's Test job, on a tmpfs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "on a small file system of its own, as CI's Test job mounts"]
async fn a_disk_that_really_fills_fails_the_merge_and_leaves_nothing() {
    use std::io::Write;
    let small = std::env::var(SMALL_DISK).expect("a small file system to fill");
    let cache = tempfile::tempdir_in(&small).unwrap();
    // This fills what it is told to, so it is told only of what is small.
    let (_, size) = disk_room(cache.path()).expect("a file system that says how large it is");
    assert!(
        size <= SMALL_AT_MOST,
        "{small} is no small disk: {size} bytes"
    );
    let mut cluster = Cluster::new().await;
    let store = cluster.blobs.clone();
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 8).spills_told(told.clone());
    cluster.blobs = Arc::new(CachedBlobs::new(store, cache.path()).on(disk));
    let builds = MergeBuilds::new(1);
    let written = Written::of(ROWS, WIDTH);
    let writer = written_on(cluster.writer().await, &builds, written).await;
    let before = files(cache.path());
    // All but 2 MiB of the rest taken, far less than the merge's vectors.
    let ballast = Path::new(&small).join("ballast");
    let mut filling = std::fs::File::create(&ballast).unwrap();
    let mebibyte = vec![1u8; 1024 * 1024];
    while filling.write_all(&mebibyte).is_ok() {}
    let filled = filling.metadata().unwrap().len();
    filling.set_len(filled - 2 * 1024 * 1024).unwrap();
    drop(filling);

    // Asked, the disk says so, and nothing is written to it.
    let manifest = cluster.committed().await.manifest;
    let skipped = within(LONG, "the merge with no room", writer.merge()).await;
    assert_eq!(skipped.unwrap(), None, "nothing to gather its vectors on");
    assert_eq!(
        told.written.load(Ordering::SeqCst),
        0,
        "written to a full disk"
    );
    assert_eq!(files(cache.path()), before);

    // Told otherwise, as one that fills after it is asked, the write says.
    told.free.store(u64::MAX - 1, Ordering::SeqCst);

    let full = within(LONG, "the merge on a full disk", writer.merge()).await;
    let error = full.expect_err("nothing to gather its vectors on");
    assert!(
        matches!(&error, StorageError::Blob { source, .. } if source.to_string().contains("space")),
        "{error}"
    );
    settled("what it wrote stays", || files(cache.path()) == before).await;
    assert_eq!(cluster.committed().await.manifest, manifest);

    told.free.store(SpillsTold::ASK_THE_DISK, Ordering::SeqCst);
    std::fs::remove_file(&ballast).unwrap();
    let merged = within(LONG, "the merge with room", writer.merge()).await;
    assert!(merged.unwrap().is_some());
    assert!(writer.last_spilled() > 0, "gathered in memory");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cache_that_cannot_begin_a_spill_fails_the_merge_before_it_builds_anything() {
    let mut cluster = Cluster::new().await;
    let store = cluster.blobs.clone();
    let written = Written::of(ROWS, WIDTH);
    // Rows flushed through a cache that keeps them, then a cache whose
    // directory is taken away: the vectors stay mapped.
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    cluster.blobs = Arc::new(CachedBlobs::new(store.clone(), &cache));
    let builds = MergeBuilds::new(1);
    let writer = written_on(cluster.writer().await, &builds, written).await;
    std::fs::remove_dir_all(&cache).unwrap();
    std::fs::write(&cache, b"in the way").unwrap();
    let failed = within(LONG, "the merge", writer.merge()).await;
    assert!(
        matches!(&failed, Err(StorageError::Blob { key, .. }) if key.ends_with("-merged.vectors")),
        "{failed:?}"
    );
    // At the gather, not once built and uploaded.
    let uploaded = store.list(&Layout(&cluster.key).prefix()).await.unwrap();
    assert!(!uploaded.iter().any(|file| file.key.contains("-merged")));
    std::fs::remove_file(&cache).unwrap();
    let merged = within(LONG, "the next merge", writer.merge()).await;
    assert!(merged.unwrap().is_some());
    assert!(writer.last_spilled() > 0, "gathered in memory");
}

/// M8-R7. A disk slower than it is waited for fails the merge at the bound
/// [`Disk::within`] sets, as every cache write is bounded, and leaves
/// nothing, however long the write it gave up runs on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_disk_that_stops_answering_fails_the_merge_at_its_bound_and_leaves_nothing() {
    let told = Arc::new(SpillsTold::default());
    let bound = Duration::from_secs(2);
    let disk = Disk::new(bound, 1 << 40, 8).spills_told(told.clone());
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::on(written, Some(disk)).await;
    let before = files(cached.dir());
    let manifest = cached.cluster.committed().await.manifest;
    told.taking.store(8_000, Ordering::SeqCst);
    let begun = std::time::Instant::now();
    let slow = within(LONG, "the merge on a slow disk", cached.writer.merge()).await;
    let took = begun.elapsed();
    assert!(matches!(&slow, Err(StorageError::Blob { .. })), "{slow:?}");
    assert!(took < Duration::from_secs(7), "given up after {took:?}");
    settled("what it wrote stays", || files(cached.dir()) == before).await;
    assert_eq!(cached.cluster.committed().await.manifest, manifest);
    // The write it gave up comes to its end, and leaves no more than that.
    told.taking.store(0, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(8_500)).await;
    assert_eq!(files(cached.dir()), before);
    let merged = within(LONG, "the next merge", cached.writer.merge()).await;
    assert!(merged.unwrap().is_some());
}

/// M8-R7. A stop is heeded while a write waits on the disk, within the two
/// seconds of merge_stop's timed rule, not at the disk's bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_stopped_while_a_write_waits_on_the_disk_gives_up_at_once() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(60), 20 * 1024 * 1024, 8).spills_told(told.clone());
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::on(written, Some(disk)).await;
    let before = files(cached.dir());
    told.taking.store(6_000, Ordering::SeqCst);
    let merging = cached.merging();
    let dir = cached.dir().to_path_buf();
    settled("no spill begun", move || !spills(&dir).is_empty()).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    cached.writer.stop_merges();
    let asked = std::time::Instant::now();
    let merged = within(LONG, "the stopped merge", merging).await;
    let took = asked.elapsed();
    assert_eq!(merged.unwrap().unwrap(), None);
    assert!(took < GIVEN_UP_WITHIN, "a stop took {took:?}");
    settled("what it wrote stays", || files(cached.dir()) == before).await;
    told.taking.store(0, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(6_000)).await;
    assert_eq!(files(cached.dir()), before);
}

/// M8-R13. Nothing waits for a spill's name to go. Removing it is a step on
/// a disk that may have stopped, and a build's thread blocked on that would
/// hold the build, which is waited for to its end, and with it a turn among
/// the node's merges: a disk that stops would stop every merge, and neither
/// a write's bound nor a stop would be kept. So the merge ends, and its turn
/// is free, with the unlink still stalled, as it fails and as it is stopped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_disk_that_stops_answering_as_a_spill_is_removed_holds_up_no_merge() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(2), 1 << 40, 8).spills_told(told.clone());
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::on(written, Some(disk)).await;
    let before = files(cached.dir());
    told.unlinking.store(6_000, Ordering::SeqCst);

    // The merge fails, and the name goes with it, stalled.
    told.at_most.store(1024 * 1024, Ordering::SeqCst);
    let begun = std::time::Instant::now();
    let failed = within(LONG, "the merge", cached.writer.merge()).await;
    let took = begun.elapsed();
    assert!(
        matches!(&failed, Err(StorageError::Blob { .. })),
        "{failed:?}"
    );
    assert!(took < Duration::from_secs(4), "held up for {took:?}");
    assert_eq!(cached.builds.under_way(), 0, "its turn is held");

    // Stopped while a write waits, with the name to go too.
    told.at_most.store(u64::MAX, Ordering::SeqCst);
    told.taking.store(6_000, Ordering::SeqCst);
    let merging = cached.merging();
    let dir = cached.dir().to_path_buf();
    settled("no spill begun", move || !spills(&dir).is_empty()).await;
    tokio::time::sleep(Duration::from_millis(200)).await;
    cached.writer.stop_merges();
    let asked = std::time::Instant::now();
    let merged = within(LONG, "the stopped merge", merging).await;
    let took = asked.elapsed();
    assert_eq!(merged.unwrap().unwrap(), None);
    assert!(took < GIVEN_UP_WITHIN, "a stop took {took:?}");

    told.unlinking.store(0, Ordering::SeqCst);
    told.taking.store(0, Ordering::SeqCst);
    settled("a name stalled never went", || {
        files(cached.dir()) == before
    })
    .await;
}

/// M8-R14. A merge with no room for its vectors, and something to spare, is
/// skipped before anything is written: a full disk is what fails a tenant's
/// open or a flush's copy, and what a merge spills it holds for the whole of
/// its build. Told apart from a failure, it is tried again next round, and
/// what it would have held is reported as it is held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_with_no_room_for_its_vectors_is_skipped_and_writes_nothing() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 8).spills_told(told.clone());
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::on(written, Some(disk)).await;
    let before = files(cached.dir());
    let manifest = cached.cluster.committed().await.manifest;
    let figures = || cached.cluster.blobs.spills().unwrap().figures();
    // Vectors, where each starts and the rows' count, and a frame's header.
    let frame = u64::from(2 * ROWS) * (4 * u64::from(WIDTH) + 8) + 36;

    told.free.store(frame, Ordering::SeqCst);
    let skipped = within(LONG, "the merge with no room", cached.writer.merge()).await;
    assert_eq!(skipped.unwrap(), None);
    assert_eq!(
        told.written.load(Ordering::SeqCst),
        0,
        "written all the same"
    );
    assert_eq!(figures().skipped, 1);
    assert_eq!(figures().failed, 0, "no failure");
    assert_eq!(figures().bytes, 0);
    assert_eq!(files(cached.dir()), before);
    assert_eq!(cached.cluster.committed().await.manifest, manifest);
    cached.all_there(written).await;

    told.free.store(SpillsTold::ASK_THE_DISK, Ordering::SeqCst);
    cached.builds.hold(Some(Holding::GatheringUntilLetGo));
    let merging = cached.merging();
    cached.held().await;
    assert_eq!(
        figures().bytes,
        frame,
        "what it will come to, from its start"
    );
    cached.builds.hold(None);
    cached.builds.let_one_go();
    let merged = within(LONG, "the merge with room", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(figures().bytes, 0, "the cache's, once taken up");
    assert_eq!(figures().skipped, 1);
    cached.all_there(written).await;
}

/// M8-R4. A rename given up takes back only what it put there: a copy of the
/// file that has since taken its place is left, as a link given up leaves
/// one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rename_given_up_leaves_a_copy_put_in_its_place() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 8).spills_told(told.clone());
    let cached = Cached::on(STOPPED, Some(disk)).await;
    told.looks_again_in.store(2_000, Ordering::SeqCst);
    let merging = cached.merging();
    let copy = |dir: &Path| -> Option<String> {
        let all = files(dir).into_iter();
        all.into_iter()
            .find(|file| file.ends_with("-merged.vectors"))
    };
    let dir = cached.dir().to_path_buf();
    let taken = {
        let dir = dir.clone();
        move || copy(&dir).is_some()
    };
    settled("the spill was never taken up", taken).await;
    let key = copy(&dir).unwrap();
    merging.abort();
    let _ = merging.await;
    settled("what was uploaded stays", || !dir.join(&key).exists()).await;
    std::fs::create_dir_all(dir.join(&key).parent().unwrap()).unwrap();
    std::fs::write(dir.join(&key), b"another copy, by a fetch since").unwrap();
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    assert!(
        dir.join(&key).exists(),
        "taken back though it was not its own"
    );
}

/// M8-R12. No async thread waits on the disk for what goes up: a page of a
/// mapping gone from memory is a read from disk on whichever thread touches
/// it, which nothing can give up. So the pieces of a file going up are read
/// by the disk's own steps: a disk that stops answering then fails the merge
/// at its bound, and on a runtime of one thread a timer never comes more
/// than a moment late, as lease renewals must not.
#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn a_disk_that_stops_answering_as_a_file_goes_up_holds_up_no_async_thread() {
    let told = Arc::new(SpillsTold::default());
    let disk = Disk::new(Duration::from_secs(2), 1 << 40, 8).spills_told(told.clone());
    let written = Written::of(ROWS, WIDTH);
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Client::over(told.clone())));
    let cached = Cached::over(Cluster::new().await, store, written, Some(disk)).await;
    let before = files(cached.dir());
    let manifest = cached.cluster.committed().await.manifest;
    let latest = Arc::new(AtomicUsize::new(0));
    let timing = tokio::spawn({
        let latest = latest.clone();
        async move {
            loop {
                let asked = std::time::Instant::now();
                tokio::time::sleep(Duration::from_millis(50)).await;
                let late = asked.elapsed().as_millis() as usize;
                latest.fetch_max(late.saturating_sub(50), Ordering::SeqCst);
            }
        }
    });
    told.reading.store(8_000, Ordering::SeqCst);
    let begun = std::time::Instant::now();
    let slow = within(LONG, "the merge on a slow disk", cached.writer.merge()).await;
    let took = begun.elapsed();
    timing.abort();
    let late = latest.load(Ordering::SeqCst);
    assert!(late <= 250, "a timer came {late} ms late");
    assert!(matches!(&slow, Err(StorageError::Blob { .. })), "{slow:?}");
    assert!(took < Duration::from_secs(7), "given up after {took:?}");
    assert_eq!(cached.cluster.committed().await.manifest, manifest);
    told.reading.store(0, Ordering::SeqCst);
    settled("what it wrote stays", || files(cached.dir()) == before).await;
    let merged = within(LONG, "the next merge", cached.writer.merge()).await;
    assert!(merged.unwrap().is_some());
}

/// The cache directory a process killed part way through a gather is to
/// spill to, as the rule below asks.
const KILLED_IN: &str = "VELLEV_SPILL_KILLED_IN";

/// Not a rule: the process the rule below kills, its merge part way
/// through its gather.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "run by a_spill_left_by_a_process_killed_goes_as_the_cache_is_next_opened"]
async fn a_process_killed_part_way_through_a_gather() {
    let Ok(cache) = std::env::var(KILLED_IN) else {
        return;
    };
    let mut cluster = Cluster::new().await;
    cluster.blobs = Arc::new(CachedBlobs::new(cluster.blobs.clone(), &cache));
    let builds = MergeBuilds::new(1);
    let written = Written::of(ROWS, WIDTH);
    let writer = written_on(cluster.writer().await, &builds, written).await;
    builds.hold(Some(Holding::GatheringUntilLetGo));
    let _merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    let held = || builds.held() == 1;
    settled("the gather was never held", held).await;
    assert_eq!(spills(Path::new(&cache)).len(), 1, "no spill begun");
    std::process::abort();
}

/// M8-R8. A spill left by a process killed outright, whose drop never ran, is
/// gone as soon as the cache is next opened: however new it is, as a restart
/// comes seconds after what it restarts from.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_spill_left_by_a_process_killed_goes_as_the_cache_is_next_opened() {
    let cache = tempfile::tempdir().unwrap();
    let killed = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_process_killed_part_way_through_a_gather",
            "--ignored",
            "--test-threads=1",
        ])
        .env(KILLED_IN, cache.path())
        .output()
        .unwrap();
    assert!(!killed.status.success(), "the process was not killed");
    assert_eq!(
        spills(cache.path()).len(),
        1,
        "{}",
        String::from_utf8_lossy(&killed.stderr)
    );
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let _opened = CachedBlobs::open(store, cache.path()).await;
    assert!(spills(cache.path()).is_empty(), "a restart leaves it");
}

/// M8-R8. Several processes may share a cache directory: one opening it
/// leaves alone a spill another is writing, and that merge commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_spill_under_way_is_left_by_a_cache_opened_beside_it() {
    let written = Written::of(ROWS, WIDTH);
    let cached = Cached::of(written).await;
    cached.builds.hold(Some(Holding::GatheringUntilLetGo));
    let merging = cached.merging();
    cached.held().await;
    assert_eq!(spills(cached.dir()).len(), 1, "no spill begun");
    let _beside = CachedBlobs::open(cached.store.clone(), cached.dir()).await;
    assert_eq!(spills(cached.dir()).len(), 1, "taken from under its merge");
    cached.builds.hold(None);
    cached.builds.let_one_go();
    let merged = within(LONG, "the merge", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    cached.all_there(written).await;
}

/// M8-R9. What a build reads and what goes up are read through a mapping,
/// which outlives the file's name: taken away under a merge, by hand or
/// with the tenant's copies, the merge goes on from what it has mapped, its
/// copy fetched afresh, and every row is what it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_spill_whose_name_goes_under_a_merge_is_read_to_the_end_all_the_same() {
    for holding in [Holding::UntilLetGo, Holding::BuiltUntilLetGo] {
        let cached = Cached::of(STOPPED).await;
        cached.builds.hold(Some(holding));
        let merging = cached.merging();
        cached.held().await;
        let found = spills(cached.dir());
        assert_eq!(found.len(), 1, "no spill begun");
        std::fs::remove_file(cached.dir().join(&found[0])).unwrap();
        cached
            .cluster
            .blobs
            .forget_local("tenant-a/")
            .await
            .unwrap();
        cached.builds.hold(None);
        cached.builds.let_one_go();
        let merged = within(LONG, "the merge", merging).await;
        assert!(merged.unwrap().unwrap().is_some(), "{holding:?}");
        cached.all_there(STOPPED).await;
        assert!(spills(cached.dir()).is_empty());
    }
}

/// M8-R9. Bytes a store keeps on after the merge has ended, and after its
/// column, its upload and its tenant's copy have gone, read as they did: the
/// mapping goes with the last of them, not before.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bytes_a_store_keeps_read_as_they_did_past_everything_else_that_held_them() {
    let mut cluster = Cluster::new().await;
    let looking = Looking::over(cluster.blobs.clone());
    let builds = MergeBuilds::new(1);
    let writer = looking.writer(&mut cluster, &builds, STOPPED).await;
    writer.stop_merges_at(Some(MergeStage::Loading));
    let merged = within(LONG, "the stopped merge", writer.merge()).await;
    assert_eq!(merged.unwrap(), None);
    assert!(writer.last_spilled() > 0, "gathered in memory");
    let cache = looking.cache.path();
    settled("no copy goes", || {
        !files(cache).iter().any(|file| file.contains("-merged"))
    })
    .await;
    drop(writer);
    let kept = std::mem::take(&mut *looking.kept.lock().unwrap());
    assert_eq!(kept.len(), 1);
    assert!(spills(cache).is_empty(), "taken up, then deleted");
    for (key, bytes) in &kept {
        format::verify(key, bytes).unwrap();
    }
}

/// M8-R10. A store with no cache, or one over a cache that does not say so,
/// gathers in memory as it always did: no file, no mapping, and the merge
/// commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_with_no_local_cache_gathers_in_memory() {
    let cluster = Cluster::new().await;
    assert!(cluster.blobs.spills().is_none());
    let dir = tempfile::tempdir().unwrap();
    let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    let on_disk = ObjectBlobs::on_disk(Arc::new(local), dir.path());
    assert!(on_disk.spills().is_none(), "its files are read in place");
    let cached = Arc::new(CachedBlobs::new(cluster.blobs.clone(), dir.path()));
    assert_eq!(cached.spills().is_some(), cfg!(unix));
    let (hiding, _) = {
        let mut over = Cluster::new().await;
        over.blobs = cached.clone();
        Paused::new(&mut over, "nothing")
    };
    assert!(hiding.spills().is_none(), "a store that says nothing");

    let builds = MergeBuilds::new(1);
    let written = Written::of(ROWS, WIDTH);
    let writer = written_on(cluster.writer().await, &builds, written).await;
    let merged = within(LONG, "the merge", writer.merge()).await;
    assert!(merged.unwrap().is_some());
    assert_eq!(writer.last_spilled(), 0);
}

/// M8-R11. A merge of rows with no vectors, whose segment has no vectors
/// file to take the spill up, begins none: a spill never taken up would stay
/// for as long as the segment.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_merge_of_rows_without_vectors_begins_no_spill() {
    let written = Written {
        has: |_| false,
        ..Written::of(4_000, WIDTH)
    };
    let cached = Cached::of(written).await;
    cached.builds.hold(Some(Holding::GatheringUntilLetGo));
    let merging = cached.merging();
    cached.held().await;
    assert!(spills(cached.dir()).is_empty());
    cached.builds.hold(None);
    cached.builds.let_one_go();
    let merged = within(LONG, "the merge", merging).await;
    assert!(merged.unwrap().unwrap().is_some());
    assert_eq!(cached.writer.last_spilled(), 0);
    assert!(spills(cached.dir()).is_empty());
    assert!(!files(cached.dir())
        .iter()
        .any(|f| f.ends_with("-merged.vectors")));
}
