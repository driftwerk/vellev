//! How long a store, a cache and a writer wait before they give up, what
//! a writer does while writes wait on it, and what it leaves behind when it
//! gives up; on a paused clock wherever only the clock is waited on, so that
//! each bound is met exactly.

mod common;

use common::*;
use futures::StreamExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    GetOptions, GetRange, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult, UploadPart,
};
use tokio::time::Instant;
use vellev_storage::{Disk, Transfers};

/// Requests to the store given up after a second; parts of 16 bytes.
const SMALL_PARTS: Transfers = Transfers {
    part_bytes: 16,
    request_within: Duration::from_secs(1),
};

/// Requests given up after a second, and parts so large that every file a
/// test writes goes up whole.
const WHOLE: Transfers = Transfers {
    part_bytes: 1024 * 1024,
    request_within: Duration::from_secs(1),
};

/// What `work` came to, failing the test rather than letting it hang if
/// that took an hour, which on a paused clock means it would never end.
async fn ended<T>(work: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(60 * 60), work)
        .await
        .expect("done within its bound")
}

/// Wait until `holds`, looking again every millisecond for up to an hour.
async fn settled<F: std::future::Future<Output = bool>>(holds: impl Fn() -> F) {
    ended(async {
        while !holds().await {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await;
}

/// How a request for a range of a file goes wrong, once.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Fault {
    /// It is never answered.
    Unanswered,
    /// Half of it comes, and then no more.
    CutShort,
}

/// An object store that counts what is asked of it and goes wrong as told.
#[derive(Debug)]
struct Faulty {
    inner: InMemory,
    /// Uploads of files whose keys hold this are never answered once armed:
    /// a whole file, or every part of one after the first.
    stalled: &'static str,
    armed: AtomicBool,
    whole: AtomicUsize,
    in_parts: AtomicUsize,
    parts: Arc<AtomicUsize>,
    /// Every part after the first fails.
    failing_parts: Arc<AtomicBool>,
    aborts: Arc<AtomicUsize>,
    /// The first abort is never answered.
    silent_abort: Arc<AtomicBool>,
    /// The start of every range asked for, and the version it had to be of.
    ranges: Mutex<Vec<(u64, Option<String>)>>,
    /// The range starting here goes wrong so, the next time it is asked for.
    range_fault: Mutex<Option<(u64, Fault)>>,
    silent_listing: AtomicBool,
    /// No delete is answered, and no question of how large a file is.
    silent_deletes: AtomicBool,
    silent_heads: AtomicBool,
}

impl Faulty {
    fn new(stalled: &'static str) -> Self {
        Self {
            inner: InMemory::new(),
            stalled,
            armed: AtomicBool::new(false),
            whole: AtomicUsize::new(0),
            in_parts: AtomicUsize::new(0),
            parts: Arc::default(),
            failing_parts: Arc::default(),
            aborts: Arc::default(),
            silent_abort: Arc::default(),
            ranges: Mutex::default(),
            range_fault: Mutex::new(None),
            silent_listing: AtomicBool::new(false),
            silent_deletes: AtomicBool::new(false),
            silent_heads: AtomicBool::new(false),
        }
    }

    fn stalls(&self, location: &Path) -> bool {
        self.armed.load(Ordering::SeqCst) && location.as_ref().contains(self.stalled)
    }

    fn starts(&self) -> Vec<u64> {
        let ranges = self.ranges.lock().unwrap();
        ranges.iter().map(|(start, _)| *start).collect()
    }
}

impl std::fmt::Display for Faulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Faulty({})", self.inner)
    }
}

#[derive(Debug)]
struct Parts {
    inner: Box<dyn MultipartUpload>,
    stalls: bool,
    parts: Arc<AtomicUsize>,
    failing: Arc<AtomicBool>,
    aborts: Arc<AtomicUsize>,
    silent_abort: Arc<AtomicBool>,
}

#[async_trait]
impl MultipartUpload for Parts {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        let first = self.parts.fetch_add(1, Ordering::SeqCst) == 0;
        match (first, self.stalls, self.failing.load(Ordering::SeqCst)) {
            (false, true, _) => Box::pin(std::future::pending()),
            (false, _, true) => Box::pin(async {
                Err(object_store::Error::Generic {
                    store: "faulty",
                    source: "connection reset".into(),
                })
            }),
            _ => self.inner.put_part(data),
        }
    }
    async fn complete(&mut self) -> object_store::Result<PutResult> {
        self.inner.complete().await
    }
    async fn abort(&mut self) -> object_store::Result<()> {
        let first = self.aborts.fetch_add(1, Ordering::SeqCst) == 0;
        if first && self.silent_abort.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        self.inner.abort().await
    }
}

#[async_trait]
impl ObjectStore for Faulty {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.whole.fetch_add(1, Ordering::SeqCst);
        if self.stalls(location) {
            std::future::pending::<()>().await;
        }
        self.inner.put_opts(location, payload, opts).await
    }
    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.in_parts.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Parts {
            inner: self.inner.put_multipart_opts(location, opts).await?,
            stalls: self.stalls(location),
            parts: self.parts.clone(),
            failing: self.failing_parts.clone(),
            aborts: self.aborts.clone(),
            silent_abort: self.silent_abort.clone(),
        }))
    }
    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        if options.head && self.silent_heads.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        let start = match &options.range {
            Some(GetRange::Bounded(range)) => range.start,
            _ => 0,
        };
        self.ranges
            .lock()
            .unwrap()
            .push((start, options.if_match.clone()));
        let fault = {
            let mut fault = self.range_fault.lock().unwrap();
            fault
                .take_if(|(at, _)| *at == start)
                .map(|(_, fault)| fault)
        };
        let found = match fault {
            Some(Fault::Unanswered) => std::future::pending().await,
            _ => self.inner.get_opts(location, options).await?,
        };
        if fault != Some(Fault::CutShort) {
            return Ok(found);
        }
        let (meta, range, attributes) = (
            found.meta.clone(),
            found.range.clone(),
            found.attributes.clone(),
        );
        let bytes = found.bytes().await?;
        let half = bytes.slice(..bytes.len() / 2);
        let cut_short =
            futures::stream::once(async move { Ok(half) }).chain(futures::stream::pending());
        Ok(GetResult {
            payload: GetResultPayload::Stream(cut_short.boxed()),
            meta,
            range,
            attributes,
        })
    }
    async fn delete(&self, location: &Path) -> object_store::Result<()> {
        if self.silent_deletes.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        self.inner.delete(location).await
    }
    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> futures::stream::BoxStream<'static, object_store::Result<ObjectMeta>> {
        match self.silent_listing.load(Ordering::SeqCst) {
            true => futures::stream::pending().boxed(),
            false => self.inner.list(prefix),
        }
    }
    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.inner.copy(from, to).await
    }
    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}

/// Blobs that go wrong for the files `picks` picks, as told: their writes
/// fail, or their copies on local disk are waited for until let go; and
/// whose deletes, of any file, are never answered, which no store that
/// keeps to its bounds lets happen.
#[derive(Debug)]
struct Picky {
    inner: Arc<dyn Blobs>,
    picks: fn(&str) -> bool,
    fails: AtomicBool,
    local_waits: AtomicBool,
    deletes_wait: AtomicBool,
    /// Deletes are held, each telling that it waits, until let go on.
    deletes_held: AtomicBool,
    /// No listing is ever answered.
    lists_wait: AtomicBool,
    /// Told each time a copy on local disk begins to be waited for.
    waiting: tokio::sync::Notify,
    /// Lets one wait for a copy on local disk go on.
    go_on: tokio::sync::Notify,
}

impl Picky {
    fn over(cluster: &mut Cluster, picks: fn(&str) -> bool) -> Arc<Self> {
        let picky = Arc::new(Self {
            inner: cluster.blobs.clone(),
            picks,
            fails: AtomicBool::new(false),
            local_waits: AtomicBool::new(false),
            deletes_wait: AtomicBool::new(false),
            deletes_held: AtomicBool::new(false),
            lists_wait: AtomicBool::new(false),
            waiting: tokio::sync::Notify::new(),
            go_on: tokio::sync::Notify::new(),
        });
        cluster.blobs = picky.clone();
        picky
    }
}

#[async_trait]
impl Blobs for Picky {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if (self.picks)(key) && self.fails.load(Ordering::SeqCst) {
            return Err(StorageError::Blob {
                key: key.into(),
                source: "connection reset".into(),
            });
        }
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        if self.lists_wait.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        if self.deletes_wait.load(Ordering::SeqCst) {
            std::future::pending::<()>().await;
        }
        if self.deletes_held.load(Ordering::SeqCst) {
            self.waiting.notify_one();
            self.go_on.notified().await;
        }
        self.inner.delete(key).await
    }
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        if (self.picks)(key) && self.local_waits.load(Ordering::SeqCst) {
            self.waiting.notify_one();
            self.go_on.notified().await;
        }
        self.inner.local(key).await
    }
}

/// Blobs whose writes of one kind of file each take `delay`.
#[derive(Debug)]
struct Slow {
    inner: Arc<dyn Blobs>,
    kind: &'static str,
    delay: Duration,
}

#[async_trait]
impl Blobs for Slow {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        if key.ends_with(self.kind) {
            tokio::time::sleep(self.delay).await;
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

/// A FIFO where a file was, which nothing writes to: whoever opens it to
/// read waits until this is dropped, as on a disk that stopped answering.
#[cfg(unix)]
struct Fifo(std::path::PathBuf);

#[cfg(unix)]
impl Fifo {
    fn instead_of(path: &std::path::Path) -> Self {
        std::fs::remove_file(path).unwrap();
        let made = std::process::Command::new("mkfifo")
            .arg(path)
            .status()
            .unwrap();
        assert!(made.success());
        Self(path.to_path_buf())
    }
}

#[cfg(unix)]
impl Drop for Fifo {
    /// Opening it for both never waits, and lets everyone waiting to read
    /// it in; closing it again tells them there is nothing to read.
    fn drop(&mut self) {
        let _ = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.0);
    }
}

/// Wait for `paused` to hold an upload, skipping the listings it reports.
async fn held(seen: &mut tokio::sync::mpsc::UnboundedReceiver<&'static str>) {
    ended(async { while seen.recv().await != Some("stored") {} }).await;
}

/// Four flushed segments of ten rows each, for a merge to make one of, and
/// a row not yet flushed, for it to flush along.
async fn mergeable(writer: &Writer) {
    writer.commit(vec![create("notes")]).await.unwrap();
    for round in 0..4u8 {
        let writes = (0..10).map(|n| upsert("notes", round * 10 + n, 1));
        writer.commit(writes.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    writer.commit(vec![upsert("notes", 100, 1)]).await.unwrap();
}

/// The keys under the partition that hold `part`.
async fn stored(cluster: &Cluster, part: &str) -> Vec<String> {
    let listed = cluster.blobs.list(&Layout(&cluster.key).prefix()).await;
    let keys = listed.unwrap().into_iter().map(|file| file.key);
    keys.filter(|key| key.contains(part)).collect()
}

/// A frame the cache takes for whole, and a key for it.
fn entry(n: u8) -> (String, Bytes) {
    let key = format!("tenant-a/0/index/{n:020}-x.index");
    (key, format::encode(Kind::Index, &n).unwrap())
}

#[tokio::test(start_paused = true)]
async fn a_file_larger_than_a_part_goes_up_in_parts_and_a_smaller_one_whole() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    let large = Bytes::from((0..100u8).collect::<Vec<_>>());
    blobs.put("large", large.clone()).await.unwrap();
    assert_eq!(store.in_parts.load(Ordering::SeqCst), 1);
    assert_eq!(store.parts.load(Ordering::SeqCst), 7);
    assert_eq!(store.whole.load(Ordering::SeqCst), 0);
    assert_eq!(blobs.get("large").await.unwrap(), large);

    let small = Bytes::from_static(b"sixteen bytes ok");
    blobs.put("small", small).await.unwrap();
    assert_eq!(store.whole.load(Ordering::SeqCst), 1);
    assert_eq!(store.in_parts.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn a_part_that_stalls_is_given_up_within_its_bound_and_the_upload_aborted() {
    let store = Arc::new(Faulty::new("large"));
    store.armed.store(true, Ordering::SeqCst);
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    let started = Instant::now();
    let given_up = ended(blobs.put("large", Bytes::from(vec![7u8; 100])))
        .await
        .unwrap_err();
    assert_eq!(started.elapsed(), SMALL_PARTS.request_within);
    assert!(matches!(&given_up, StorageError::Blob { .. }), "{given_up}");
    assert_eq!(store.aborts.load(Ordering::SeqCst), 1);
    assert!(matches!(
        blobs.get("large").await,
        Err(StorageError::Missing { .. })
    ));
}

#[tokio::test(start_paused = true)]
async fn an_upload_dropped_unfinished_is_aborted() {
    let store = Arc::new(Faulty::new("large"));
    store.armed.store(true, Ordering::SeqCst);
    let blobs = Arc::new(ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS));
    let putting = tokio::spawn({
        let blobs = blobs.clone();
        async move { blobs.put("large", Bytes::from(vec![7u8; 100])).await }
    });
    settled(|| async { store.parts.load(Ordering::SeqCst) == 2 }).await;
    putting.abort();
    assert!(putting.await.unwrap_err().is_cancelled());
    settled(|| async { store.aborts.load(Ordering::SeqCst) == 1 }).await;
}

#[tokio::test(start_paused = true)]
async fn an_abort_given_up_on_part_way_is_made_again() {
    let store = Arc::new(Faulty::new("nothing"));
    store.failing_parts.store(true, Ordering::SeqCst);
    store.silent_abort.store(true, Ordering::SeqCst);
    let blobs = Arc::new(ObjectBlobs::new(store.clone()).transferring(Transfers {
        request_within: Duration::from_secs(60),
        ..SMALL_PARTS
    }));
    let putting = tokio::spawn({
        let blobs = blobs.clone();
        async move { blobs.put("large", Bytes::from(vec![7u8; 100])).await }
    });
    // The second part fails, and the abort that follows is not answered.
    settled(|| async { store.aborts.load(Ordering::SeqCst) == 1 }).await;
    putting.abort();
    assert!(putting.await.unwrap_err().is_cancelled());
    settled(|| async { store.aborts.load(Ordering::SeqCst) == 2 }).await;
}

#[tokio::test(start_paused = true)]
async fn a_file_too_large_for_the_most_parts_is_refused_before_any_of_it_goes_up() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(Transfers {
        part_bytes: 1,
        ..SMALL_PARTS
    });
    let refused = blobs.put("huge", Bytes::from(vec![0u8; 10_001])).await;
    assert!(
        matches!(refused, Err(StorageError::Blob { .. })),
        "{refused:?}"
    );
    assert_eq!(store.whole.load(Ordering::SeqCst), 0);
    assert_eq!(store.in_parts.load(Ordering::SeqCst), 0);
    let fits = Bytes::from(vec![0u8; 10_000]);
    blobs.put("fits", fits).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_file_larger_than_a_part_comes_down_a_range_at_a_time_each_of_the_version_the_first_was()
{
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    let large = Bytes::from((0..100u8).collect::<Vec<_>>());
    blobs.put("large", large.clone()).await.unwrap();
    assert_eq!(blobs.get("large").await.unwrap(), large);
    assert_eq!(store.starts(), [0, 16, 32, 48, 64, 80, 96]);
    let ranges = store.ranges.lock().unwrap();
    let (_, first) = &ranges[0];
    assert_eq!(*first, None, "the first is of whatever version is there");
    let version = ranges[1].1.clone();
    assert!(version.is_some(), "the store tells versions apart");
    assert!(ranges[1..].iter().all(|(_, pinned)| *pinned == version));
}

#[tokio::test(start_paused = true)]
async fn a_range_never_answered_is_given_up_within_its_bound_and_asked_for_again() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    let large = Bytes::from((0..100u8).collect::<Vec<_>>());
    blobs.put("large", large.clone()).await.unwrap();

    *store.range_fault.lock().unwrap() = Some((32, Fault::Unanswered));
    let started = Instant::now();
    assert_eq!(ended(blobs.get("large")).await.unwrap(), large);
    assert_eq!(started.elapsed(), SMALL_PARTS.request_within);
    assert_eq!(store.starts(), [0, 16, 32, 32, 48, 64, 80, 96]);

    // Twice in a row, and the read is given up.
    *store.range_fault.lock().unwrap() = Some((0, Fault::Unanswered));
    let once = tokio::spawn({
        let blobs = blobs.clone();
        async move { blobs.get("large").await }
    });
    settled(|| async { store.range_fault.lock().unwrap().is_none() }).await;
    *store.range_fault.lock().unwrap() = Some((0, Fault::Unanswered));
    assert!(ended(once).await.unwrap().is_err());
}

#[tokio::test(start_paused = true)]
async fn a_range_cut_short_is_gone_on_with_from_the_bytes_that_came() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    let large = Bytes::from((0..100u8).collect::<Vec<_>>());
    blobs.put("large", large.clone()).await.unwrap();

    *store.range_fault.lock().unwrap() = Some((32, Fault::CutShort));
    let started = Instant::now();
    assert_eq!(ended(blobs.get("large")).await.unwrap(), large);
    assert_eq!(started.elapsed(), SMALL_PARTS.request_within);
    assert_eq!(store.starts(), [0, 16, 32, 40, 48, 64, 80, 96]);
}

#[tokio::test]
async fn a_file_with_nothing_in_it_is_read_as_that() {
    let dir = tempfile::tempdir().unwrap();
    let on_disk = Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let stores = [
        ObjectBlobs::new(Arc::new(InMemory::new())),
        ObjectBlobs::on_disk(on_disk, dir.path()),
    ];
    for blobs in stores {
        blobs.put("empty", Bytes::new()).await.unwrap();
        assert_eq!(blobs.get("empty").await.unwrap(), Bytes::new());
        assert!(matches!(
            blobs.get("absent").await,
            Err(StorageError::Missing { .. })
        ));
    }
}

#[tokio::test(start_paused = true)]
async fn a_listing_never_answered_is_given_up_within_its_bound() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    blobs.put("a/b", Bytes::from_static(b"x")).await.unwrap();
    assert_eq!(blobs.list("a/").await.unwrap().len(), 1);

    store.silent_listing.store(true, Ordering::SeqCst);
    let started = Instant::now();
    assert!(ended(blobs.list("a/")).await.is_err());
    assert_eq!(started.elapsed(), SMALL_PARTS.request_within);
}

#[tokio::test(start_paused = true)]
async fn a_delete_never_answered_is_given_up_within_its_bound() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    blobs.put("a/b", Bytes::from_static(b"x")).await.unwrap();

    store.silent_deletes.store(true, Ordering::SeqCst);
    let started = Instant::now();
    assert!(ended(blobs.delete("a/b")).await.is_err());
    assert_eq!(started.elapsed(), SMALL_PARTS.request_within);
    store.silent_deletes.store(false, Ordering::SeqCst);
    blobs.delete("a/b").await.unwrap();
    assert!(blobs.list("a/").await.unwrap().is_empty());
}

#[tokio::test(start_paused = true)]
async fn an_empty_file_whose_size_is_never_told_is_given_up_within_its_bound() {
    let store = Arc::new(Faulty::new("nothing"));
    let blobs = ObjectBlobs::new(store.clone()).transferring(SMALL_PARTS);
    blobs.put("a/empty", Bytes::new()).await.unwrap();
    assert_eq!(blobs.get("a/empty").await.unwrap(), Bytes::new());

    // No range of an empty file can be asked for, so it is asked how large
    // it is, which is waited for as long as any request.
    store.silent_heads.store(true, Ordering::SeqCst);
    let started = Instant::now();
    assert!(ended(blobs.get("a/empty")).await.is_err());
    assert_eq!(started.elapsed(), SMALL_PARTS.request_within);
}

#[test]
fn every_request_a_transfer_may_make_is_counted() {
    // A file of a part goes up whole; a larger one is started, sent in
    // parts, completed, and aborted should that fail.
    assert_eq!(SMALL_PARTS.upload_within(16), Duration::from_secs(1));
    assert_eq!(SMALL_PARTS.upload_within(100), Duration::from_secs(7 + 3));
    // Each range may be asked for twice.
    assert_eq!(SMALL_PARTS.read_within(16), Duration::from_secs(2));
    assert_eq!(SMALL_PARTS.read_within(100), Duration::from_secs(14));
    // A step on local disk is waited for as long as its bytes take.
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 1);
    assert_eq!(disk.within(40 * 1024 * 1024), Duration::from_secs(12));
    // And as long again, at most, for a thread to take it on.
    assert_eq!(
        disk.answers_within(40 * 1024 * 1024),
        Duration::from_secs(24)
    );
}

#[tokio::test]
async fn a_copy_kept_on_local_disk_is_written_whole_whoever_stops_waiting() {
    let dir = tempfile::tempdir().unwrap();
    let cached = CachedBlobs::new(
        Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
        dir.path(),
    );
    let (key, frame) = entry(1);

    // Stop waiting as soon as the write to local disk has begun: hardly
    // ever is it done by then, and whether it is makes no difference.
    let mut putting = Box::pin(cached.put(&key, frame.clone()));
    let _ = futures::poll!(putting.as_mut());
    drop(putting);

    let path = dir.path().join(&key);
    let started = std::time::Instant::now();
    while !path.exists() && started.elapsed() < Duration::from_secs(30) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(std::fs::read(&path).unwrap(), frame);
    let left: Vec<_> = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(left.len(), 1, "{left:?}");
}

#[cfg(unix)]
#[tokio::test]
async fn a_copy_on_local_disk_that_never_answers_is_read_from_the_store_instead() {
    let dir = tempfile::tempdir().unwrap();
    let disk = Disk::new(Duration::from_millis(100), 1024 * 1024, 4);
    let cached = CachedBlobs::new(
        Arc::new(ObjectBlobs::new(Arc::new(InMemory::new()))),
        dir.path(),
    )
    .on(disk);
    let (key, frame) = entry(1);
    cached.put(&key, frame.clone()).await.unwrap();
    let _silent = Fifo::instead_of(&dir.path().join(&key));
    let read = tokio::time::timeout(Duration::from_secs(30), cached.get(&key)).await;
    assert_eq!(read.expect("given up within its bound").unwrap(), frame);
}

#[cfg(unix)]
#[tokio::test]
async fn with_every_thread_for_local_disk_waiting_on_it_the_cache_is_passed_over_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let floor = Duration::from_secs(1);
    let disk = Disk::new(floor, 1024 * 1024, 2);
    let cached = CachedBlobs::new(store.clone(), dir.path()).on(disk);
    let entries = [entry(1), entry(2), entry(3)];
    for (key, frame) in &entries {
        cached.put(key, frame.clone()).await.unwrap();
    }
    let [(first, _), (second, _), (third, kept)] = &entries;
    let silent = [first, second].map(|key| Fifo::instead_of(&dir.path().join(key)));
    // Each is given up and read from the store, its thread waiting still;
    // nothing waits for a thread to keep a copy of what was read.
    let soon = Duration::from_secs(30);
    let both = async { tokio::join!(cached.get(first), cached.get(second)) };
    let (one, other) = tokio::time::timeout(soon, both)
        .await
        .expect("nothing waits for a thread");
    assert!(one.is_ok() && other.is_ok());

    // The third is only in the cache now, which is not asked, and nothing
    // more is asked of the disk, nor any thread of it waited for: both are
    // kept by steps given up.
    store.delete(third).await.unwrap();
    let asked = std::time::Instant::now();
    let passed_over = tokio::time::timeout(soon, cached.get(third)).await;
    assert!(
        matches!(passed_over, Ok(Err(StorageError::Missing { .. }))),
        "{passed_over:?}"
    );
    let refused = tokio::time::timeout(soon, cached.forget_local("tenant-a/")).await;
    assert!(matches!(refused, Ok(Err(_))), "{refused:?}");
    // Nor is a file deleted whose copy cannot be: the copy would be read
    // in its place ever after.
    let (kept_key, kept_frame) = entry(4);
    store.put(&kept_key, kept_frame.clone()).await.unwrap();
    let undeleted = tokio::time::timeout(soon, cached.delete(&kept_key)).await;
    assert!(matches!(undeleted, Ok(Err(_))), "{undeleted:?}");
    assert_eq!(store.get(&kept_key).await.unwrap(), kept_frame);
    assert!(asked.elapsed() < floor, "refused at once");

    // Once the disk answers again, so does the cache.
    drop(silent);
    let started = std::time::Instant::now();
    while cached.get(third).await.is_err() && started.elapsed() < soon {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(cached.get(third).await.unwrap(), *kept);
    cached.forget_local("tenant-a/").await.unwrap();
    assert!(!dir.path().join(third).exists());
}

#[tokio::test]
async fn files_left_part_written_long_ago_are_deleted_as_the_cache_is_opened() {
    let dir = tempfile::tempdir().unwrap();
    let under = dir.path().join("tenant-a/0/vectors");
    std::fs::create_dir_all(&under).unwrap();
    let (stale, fresh, whole) = (
        under.join("x.partial-41-7"),
        under.join("y.partial-42-1"),
        under.join("z.vectors"),
    );
    for path in [&stale, &fresh, &whole] {
        std::fs::write(path, b"bytes").unwrap();
    }
    let long_ago = std::time::SystemTime::now() - Duration::from_secs(2 * 60 * 60);
    for old in [&stale, &whole] {
        let file = std::fs::File::options().write(true).open(old).unwrap();
        file.set_modified(long_ago).unwrap();
    }
    let inner: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    CachedBlobs::open(inner.clone(), dir.path()).await;
    assert!(!stale.exists());
    assert!(fresh.exists(), "another process may be writing it still");
    assert!(whole.exists());
    // A directory that is not there is nothing to sweep.
    CachedBlobs::open(inner, dir.path().join("absent")).await;
}

#[cfg(unix)]
#[test]
fn a_disk_that_never_answers_takes_no_thread_from_those_segments_are_built_on() {
    // One thread for everything the runtime runs off its own: were a step
    // on local disk to wait on it, no segment would ever be built.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let mut cluster = Cluster::new().await;
        let disk = Disk::new(Duration::from_millis(100), 1024 * 1024, 4);
        let cached = Arc::new(CachedBlobs::new(cluster.blobs.clone(), dir.path()).on(disk));
        let (key, frame) = entry(1);
        cached.put(&key, frame.clone()).await.unwrap();
        let _silent = Fifo::instead_of(&dir.path().join(&key));
        assert_eq!(cached.get(&key).await.unwrap(), frame);

        cluster.blobs = cached;
        let writer = cluster.writer().await;
        let merged = tokio::time::timeout(Duration::from_secs(60), async {
            mergeable(&writer).await;
            writer.merge().await
        });
        merged.await.expect("built").unwrap().expect("a merge");
    });
}

/// Blobs over a cache whose copy of each vectors file stops answering the
/// moment it is asked where the copy is.
#[cfg(unix)]
#[derive(Debug)]
struct Silenced {
    inner: CachedBlobs,
    silent: Mutex<Vec<Fifo>>,
}

#[cfg(unix)]
impl std::fmt::Debug for Fifo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fifo({})", self.0.display())
    }
}

#[cfg(unix)]
#[async_trait]
impl Blobs for Silenced {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
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
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        let local = self.inner.local(key).await?;
        if let Some(local) = local.as_ref().filter(|_| key.ends_with(".vectors")) {
            let mut silent = self.silent.lock().unwrap();
            silent.push(Fifo::instead_of(&local.path));
        }
        Ok(local)
    }
}

#[cfg(unix)]
#[tokio::test]
async fn a_flush_whose_vectors_on_local_disk_never_answer_keeps_those_it_holds() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    // The copies lie on a disk of the cache's own, which waits a little
    // over a tenth of a second for a step where the one every store shares
    // waits ten.
    let disk = Disk::new(Duration::from_millis(100), 64 * 1024 * 1024, 4);
    cluster.blobs = Arc::new(Silenced {
        inner: CachedBlobs::new(cluster.blobs.clone(), dir.path()).on(disk),
        silent: Mutex::default(),
    });
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    let started = std::time::Instant::now();
    let flushed = tokio::time::timeout(Duration::from_secs(60), writer.flush()).await;
    flushed.expect("given up within its bound").unwrap();
    let took = started.elapsed();
    assert!(
        took < Duration::from_secs(2),
        "read in place on the disk the copy lies on, and given up as it does: {took:?}"
    );
    assert!(!writer.is_closed());
    let found = writer.partition().read(|state| n_of(state, "notes", 1));
    assert_eq!(found, Some(1));
}

#[tokio::test(start_paused = true)]
async fn a_merge_whose_upload_stalls_holds_no_write_up_and_finishes_when_it_does() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "segment/");
    let writer = Arc::new(cluster.writer().await);
    mergeable(&writer).await;

    // The merged segment's upload is held, and then that of the rows the
    // merge flushes along with it.
    paused.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    for n in [101, 102] {
        held(&mut seen).await;
        let written = ended(writer.commit(vec![upsert("notes", n, 1)])).await;
        assert!(written.is_ok(), "{written:?}");
        assert!(!merging.is_finished());
        paused.release.notify_one();
    }
    ended(merging).await.unwrap().unwrap().expect("a merge");
    assert!(!writer.is_closed());
    let count = |state: &State| state.count(&name("notes"));
    assert_eq!(writer.partition().read(count), 43);
    assert_eq!(
        cluster.open().await.partition().read(count),
        43,
        "what was written meanwhile is there when the partition is opened again"
    );
}

#[tokio::test(start_paused = true)]
async fn a_row_replaced_and_one_deleted_while_a_merge_uploads_are_so_when_opened_again() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, "-merged.segment");
    let writer = Arc::new(cluster.writer().await);
    mergeable(&writer).await;

    paused.armed.store(true, Ordering::SeqCst);
    let merging = tokio::spawn({
        let writer = writer.clone();
        async move { writer.merge().await }
    });
    held(&mut seen).await;
    let changes = vec![upsert("notes", 1, 7), delete("notes", 2)];
    writer.commit(changes).await.unwrap();
    paused.release.notify_one();
    ended(merging).await.unwrap().unwrap().expect("a merge");
    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            assert_eq!(n_of(state, "notes", 1), Some(7));
            assert_eq!(n_of(state, "notes", 2), None);
            assert_eq!(state.count(&name("notes")), 40);
        });
    }
}

#[tokio::test(start_paused = true)]
async fn a_merge_whose_upload_fails_leaves_the_writer_open_and_nothing_it_uploaded() {
    let mut cluster = Cluster::new().await;
    // The merged segment goes up, and then the segment of the rows the
    // merge flushes fails to.
    let picky = Picky::over(&mut cluster, |key| {
        key.ends_with(".segment") && !key.contains("-merged")
    });
    let writer = cluster.writer().await;
    mergeable(&writer).await;

    picky.fails.store(true, Ordering::SeqCst);
    let failed = writer.merge().await.unwrap_err();
    assert!(matches!(&failed, StorageError::Blob { .. }), "{failed}");
    picky.fails.store(false, Ordering::SeqCst);
    assert!(!writer.is_closed(), "nothing was published");
    settled(|| async { stored(&cluster, "-merged").await.is_empty() }).await;
    writer.commit(vec![upsert("notes", 101, 1)]).await.unwrap();
    writer.merge().await.unwrap().expect("a merge");
}

#[tokio::test(start_paused = true)]
async fn a_segment_whose_last_file_fails_to_go_up_takes_the_others_with_it() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |key| key.ends_with("-merged.segment"));
    let writer = cluster.writer().await;
    mergeable(&writer).await;

    picky.fails.store(true, Ordering::SeqCst);
    assert!(writer.merge().await.is_err());
    settled(|| async { stored(&cluster, "-merged").await.is_empty() }).await;
}

#[tokio::test(start_paused = true)]
async fn a_merge_that_fails_while_the_store_answers_no_delete_holds_no_write_up() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |key| {
        key.ends_with(".segment") && !key.contains("-merged")
    });
    let writer = cluster.writer().await;
    mergeable(&writer).await;

    picky.fails.store(true, Ordering::SeqCst);
    picky.deletes_wait.store(true, Ordering::SeqCst);
    let started = Instant::now();
    assert!(ended(writer.merge()).await.is_err());
    let written = ended(writer.commit(vec![upsert("notes", 101, 1)])).await;
    assert!(written.is_ok(), "{written:?}");
    assert_eq!(started.elapsed(), Duration::ZERO, "nothing was waited for");
    assert!(!writer.is_closed() && !writer.stalled());
}

#[tokio::test(start_paused = true)]
async fn a_flush_that_cannot_commit_deletes_the_segment_it_uploaded() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |key| key.ends_with(".vectors"));
    let unreliable = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let writer = Arc::new(cluster.writer_on(unreliable.clone()).await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();

    // The flush has uploaded its segment and waits to load it back when a
    // commit fails, which closes the writer.
    picky.local_waits.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    ended(picky.waiting.notified()).await;
    assert_eq!(stored(&cluster, "segment/").await.len(), 1);
    unreliable
        .refuse_commits
        .store(usize::MAX, Ordering::SeqCst);
    assert!(writer.commit(vec![upsert("notes", 2, 2)]).await.is_err());
    picky.go_on.notify_one();
    assert!(matches!(
        ended(flushing).await.unwrap(),
        Err(StorageError::Closed)
    ));
    settled(|| async { stored(&cluster, "segment/").await.is_empty() }).await;
    assert!(stored(&cluster, "vectors/").await.is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_restore_given_up_before_its_commit_deletes_the_segment_it_uploaded() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |key| key.ends_with(".vectors"));
    let writer = Arc::new(cluster.writer().await);
    let kept = writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap()
        .position;
    let head = cluster.committed().await.manifest.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();

    picky.local_waits.store(true, Ordering::SeqCst);
    let restoring = tokio::spawn({
        let writer = writer.clone();
        async move { writer.restore(Some(&head), kept).await }
    });
    ended(picky.waiting.notified()).await;
    assert_eq!(stored(&cluster, "segment/").await.len(), 1);
    restoring.abort();
    assert!(restoring.await.unwrap_err().is_cancelled());
    settled(|| async { stored(&cluster, "segment/").await.is_empty() }).await;
    picky.local_waits.store(false, Ordering::SeqCst);
    assert!(!writer.is_closed(), "nothing was published");
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_restore_whose_upload_stalls_is_given_up_within_its_bound_and_writes_go_on() {
    let mut cluster = Cluster::new().await;
    let store = Arc::new(Faulty::new("segment/"));
    cluster.blobs = Arc::new(ObjectBlobs::new(store.clone()).transferring(WHOLE));
    let writer = Arc::new(cluster.writer().await);
    let kept = writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap()
        .position;
    let head = cluster.committed().await.manifest.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();

    store.armed.store(true, Ordering::SeqCst);
    let started = Instant::now();
    let given_up = ended(writer.restore(Some(&head), kept)).await.unwrap_err();
    assert_eq!(started.elapsed(), WHOLE.request_within);
    assert!(matches!(&given_up, StorageError::Blob { .. }), "{given_up}");
    store.armed.store(false, Ordering::SeqCst);
    assert!(!writer.is_closed(), "nothing was committed");
    writer.commit(vec![upsert("notes", 3, 3)]).await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn a_restore_whose_read_of_a_replaced_segment_stalls_asks_again_and_finishes() {
    let store = Arc::new(Faulty::new("nothing"));
    let mut cluster = Cluster::new().await;
    cluster.blobs = Arc::new(ObjectBlobs::new(store.clone()).transferring(WHOLE));
    let writer = cluster.writer().await;
    mergeable(&writer).await;
    let kept = cluster.committed().await;
    writer.merge().await.unwrap().expect("a merge");
    writer.commit(vec![upsert("notes", 101, 1)]).await.unwrap();

    // Going back to before the merge reads the segments it replaced, and
    // the first range asked for is never answered.
    *store.range_fault.lock().unwrap() = Some((0, Fault::Unanswered));
    let started = Instant::now();
    ended(writer.restore(kept.manifest.as_ref(), kept.log_position))
        .await
        .unwrap();
    assert_eq!(started.elapsed(), WHOLE.request_within);
    assert_eq!(
        writer.partition().read(|state| state.count(&name("notes"))),
        41
    );
}

#[tokio::test(start_paused = true)]
async fn an_upload_slow_after_it_reached_the_store_is_not_given_up_by_the_writer() {
    // As when the copy kept on local disk is slow to write: the store bounds
    // its own requests, and the writer bounds nothing past them.
    let mut cluster = Cluster::new().await;
    cluster.blobs = Arc::new(Slow {
        inner: cluster.blobs.clone(),
        kind: ".log",
        delay: WHOLE.request_within * 3,
    });
    let writer = cluster.writer().await.transferring(WHOLE);
    ended(writer.commit(vec![create("notes")])).await.unwrap();
    assert!(!writer.is_closed());
}

#[tokio::test(start_paused = true)]
async fn a_commit_left_unanswered_closes_the_writer_once_its_tries_are_given_up() {
    let cluster = Cluster::new().await;
    let unreliable = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let writer = cluster.writer_on(unreliable.clone()).await;
    writer.commit(vec![create("notes")]).await.unwrap();

    unreliable.unanswered.store(true, Ordering::SeqCst);
    let started = Instant::now();
    let given_up = ended(writer.commit(vec![upsert("notes", 1, 1)]))
        .await
        .unwrap_err();
    let tries = TTL / 4 * 3;
    let waits = Duration::from_millis(50 + 100);
    assert_eq!(started.elapsed(), tries + waits);
    assert!(
        matches!(&given_up, StorageError::Coordinator(error) if error.is_retryable()),
        "{given_up}"
    );
    assert!(writer.is_closed());
}

#[tokio::test(start_paused = true)]
async fn a_publish_slow_but_within_its_bounds_is_not_taken_for_stuck() {
    let mut cluster = Cluster::new().await;
    cluster.blobs = Arc::new(Slow {
        inner: cluster.blobs.clone(),
        kind: ".log",
        delay: TTL / 2,
    });
    let unreliable = Arc::new(Unreliable::over(cluster.coordinator.clone()));
    let uploads = Transfers {
        request_within: TTL,
        ..WHOLE
    };
    let writer = cluster.writer_on(unreliable.clone()).await;
    let writer = Arc::new(writer.transferring(uploads));

    // Each try is answered just within its bound, the first two lost.
    *unreliable.slowly.lock().unwrap() = TTL / 4 - Duration::from_millis(1);
    unreliable.lose_responses.store(2, Ordering::SeqCst);
    let committing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(vec![create("notes")]).await }
    });
    tokio::time::sleep(TTL + Duration::from_secs(1)).await;
    assert!(!committing.is_finished(), "still publishing past a lease");
    assert!(!writer.stalled());
    ended(committing).await.unwrap().unwrap();
    assert!(!writer.is_closed());
}

/// Commit `mutations` with a log whose upload reaches the store and is
/// never done with, and see the writer taken for stuck just when all it
/// was allowed has passed: five seconds for what any turn does in memory,
/// however little; `group` for what a commit of so much does there; the
/// log's upload, one request, and its copy to local disk, with the wait
/// for a thread to make it on; and every try to commit, with the waits
/// between.
async fn stuck_only_past_its_bounds(mutations: Vec<Mutation>, group: Duration) {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, ".log");
    let writer = Arc::new(cluster.writer().await.transferring(WHOLE));
    writer.commit(vec![create("notes")]).await.unwrap();
    assert!(!writer.stalled(), "nothing holds it");

    paused.armed.store(true, Ordering::SeqCst);
    let started = Instant::now();
    let _committing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.commit(mutations).await }
    });
    held(&mut seen).await;
    let logs = stored(&cluster, "/log/").await;
    let newest = logs.iter().max().unwrap();
    let bytes = cluster.blobs.get(newest).await.unwrap().len();
    assert!(bytes <= WHOLE.part_bytes, "one request");
    let in_memory = Duration::from_secs(5) + group;
    let log = WHOLE.request_within + Disk::shared().answers_within(bytes);
    let commits = TTL / 4 * 3 + Duration::from_millis(50 + 100);
    tokio::time::sleep_until(started + in_memory + log + commits).await;
    assert!(!writer.stalled(), "not past its bounds yet");
    tokio::time::sleep(Duration::from_millis(1)).await;
    assert!(writer.stalled());
}

#[tokio::test(start_paused = true)]
async fn a_writer_held_past_every_bound_on_what_it_waits_for_is_taken_for_stuck() {
    stuck_only_past_its_bounds(vec![upsert("notes", 1, 1)], Duration::ZERO).await;
}

#[tokio::test(start_paused = true)]
async fn a_writer_committing_a_large_group_is_allowed_as_long_as_so_much_may_take() {
    let rows: Vec<Mutation> = (0..4_000)
        .map(|n| upsert("notes", (n % 200) as u8, n))
        .collect();
    // Two thousand mutations a second, and four mebibytes of them.
    let weight = vellev_storage::Batch::from(rows.clone()).weight();
    let group = Duration::from_millis(weight.rows as u64 * 1_000 / 2_000)
        + Duration::from_millis(weight.bytes as u64 * 1_000 / (4 * 1024 * 1024));
    assert!(group > Duration::from_secs(2));
    stuck_only_past_its_bounds(rows, group).await;
}

#[tokio::test(start_paused = true)]
async fn a_flush_given_up_part_way_leaves_the_writer_closed() {
    let mut cluster = Cluster::new().await;
    let (paused, mut seen) = Paused::new(&mut cluster, ".manifest");
    let writer = Arc::new(cluster.writer().await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();

    paused.armed.store(true, Ordering::SeqCst);
    let flushing = tokio::spawn({
        let writer = writer.clone();
        async move { writer.flush().await }
    });
    held(&mut seen).await;
    flushing.abort();
    assert!(flushing.await.unwrap_err().is_cancelled());
    assert!(writer.is_closed());
    assert!(matches!(
        writer.commit(vec![upsert("notes", 2, 2)]).await,
        Err(StorageError::Closed)
    ));
    assert_eq!(
        stored(&cluster, "segment/").await.len(),
        1,
        "what a commit that was tried names is kept, for it may have been made"
    );
}

#[tokio::test]
async fn a_disk_that_is_busy_and_answers_serves_every_reader_in_its_turn() {
    let dir = tempfile::tempdir().unwrap();
    let store = Cluster::new().await.observed();
    let disk = Disk::new(Duration::from_secs(10), 20 * 1024 * 1024, 2);
    let cached = CachedBlobs::new(store.clone(), dir.path()).on(disk);
    let entries: Vec<(String, Bytes)> = (0..150).map(entry).collect();
    for (key, frame) in &entries {
        cached.put(key, frame.clone()).await.unwrap();
    }
    // Far more readers than the disk has threads: each waits for one.
    let read = join_all(entries.iter().map(|(key, _)| cached.get(key))).await;
    for ((_, frame), read) in entries.iter().zip(read) {
        assert_eq!(read.unwrap(), *frame);
    }
    let found = join_all(entries.iter().map(|(key, _)| cached.local(key))).await;
    assert!(found.into_iter().all(|found| found.unwrap().is_some()));
    assert_eq!(
        store.fetched(),
        Vec::<String>::new(),
        "none was passed over"
    );
}

/// Blobs that note what is asked of them while a writer is held.
#[derive(Debug)]
struct Watched {
    inner: Arc<dyn Blobs>,
    writer: std::sync::OnceLock<std::sync::Weak<Writer>>,
    asked: Mutex<Vec<String>>,
}

impl Watched {
    fn over(cluster: &mut Cluster) -> Arc<Self> {
        let watched = Arc::new(Self {
            inner: cluster.blobs.clone(),
            writer: std::sync::OnceLock::new(),
            asked: Mutex::default(),
        });
        cluster.blobs = watched.clone();
        watched
    }

    fn note(&self, call: &str, key: &str) {
        let writer = self.writer.get().and_then(std::sync::Weak::upgrade);
        if writer.is_some_and(|writer| writer.is_held()) {
            self.asked.lock().unwrap().push(format!("{call} {key}"));
        }
    }

    /// What was asked while the writer was held, since this was last asked.
    fn while_held(&self) -> Vec<String> {
        std::mem::take(&mut self.asked.lock().unwrap())
    }
}

#[async_trait]
impl Blobs for Watched {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.note("put", key);
        self.inner.put(key, bytes).await
    }
    async fn get(&self, key: &str) -> vellev_storage::Result<Bytes> {
        self.note("get", key);
        self.inner.get(key).await
    }
    async fn list(&self, prefix: &str) -> vellev_storage::Result<Vec<Listed>> {
        self.note("list", prefix);
        self.inner.list(prefix).await
    }
    async fn delete(&self, key: &str) -> vellev_storage::Result<()> {
        self.note("delete", key);
        self.inner.delete(key).await
    }
    async fn local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        self.note("local", key);
        self.inner.local(key).await
    }
}

#[tokio::test]
async fn nothing_but_its_head_file_is_asked_of_the_store_or_the_disk_while_a_writer_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    cluster.blobs = Arc::new(CachedBlobs::new(cluster.blobs.clone(), dir.path()));
    let watched = Watched::over(&mut cluster);
    let writer = Arc::new(
        cluster
            .writer()
            .await
            .merging_with(vellev_storage::Merging {
                fanout: 2,
                ..Default::default()
            }),
    );
    watched.writer.set(Arc::downgrade(&writer)).unwrap();
    // The one thing a commit, a checkpoint and a restore each ask for
    // while they hold the writer: that the head they commit be stored.
    let only_its_head = |made: &'static str| {
        let (cluster, watched) = (&cluster, &watched);
        async move {
            let head = cluster.committed().await.manifest.unwrap().0;
            assert_eq!(watched.while_held(), [format!("put {head}")], "{made}");
        }
    };

    let rows = |round: u8| (0..10).map(move |n| upsert("notes", round * 10 + n, 1));
    let first = [create("notes")].into_iter().chain(rows(0)).collect();
    writer.commit(first).await.unwrap();
    only_its_head("a commit").await;
    writer.flush().await.unwrap();
    only_its_head("a flush").await;
    let kept = cluster.committed().await;
    writer.commit(rows(1).collect()).await.unwrap();
    writer.flush().await.unwrap();
    watched.while_held();
    writer.commit(rows(2).collect()).await.unwrap();
    watched.while_held();
    writer
        .merge()
        .await
        .unwrap()
        .expect("two segments to merge");
    only_its_head("a merge").await;

    writer.commit(rows(3).collect()).await.unwrap();
    watched.while_held();
    let restoring = writer.restore(kept.manifest.as_ref(), kept.log_position);
    restoring.await.unwrap();
    only_its_head("a restore").await;

    writer.flush().await.unwrap();
    watched.while_held();
    let deleted = writer.clean(Duration::ZERO).await.unwrap();
    assert!(deleted > 0, "there was something to delete");
    assert_eq!(watched.while_held(), Vec::<String>::new(), "cleaning up");
}

/// Keep the one thread the runtime has for work off its own until what is
/// returned is sent to or dropped.
fn hostage() -> (std::sync::mpsc::Sender<()>, tokio::task::JoinHandle<()>) {
    let (let_go, kept) = std::sync::mpsc::channel::<()>();
    let held = tokio::task::spawn_blocking(move || {
        let _ = kept.recv();
    });
    (let_go, held)
}

/// A runtime of one thread, with one more for work off it.
fn one_thread_for_work() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .max_blocking_threads(1)
        .build()
        .unwrap()
}

/// Long enough for whatever is not waiting on anything to have gone on.
const A_WHILE: Duration = Duration::from_millis(200);

/// Embeds every text alike, once it is let go on, and tells when asked.
#[derive(Debug, Default)]
struct Gated {
    asked: tokio::sync::Notify,
    go_on: tokio::sync::Notify,
    calls: AtomicUsize,
}

#[async_trait]
impl vellev_storage::Vectorize for Gated {
    async fn vectors(
        &self,
        _embedding: &Embedding,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, vellev_storage::Unembedded> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.asked.notify_one();
        self.go_on.notified().await;
        Ok(texts.iter().map(|_| vec![1.0, 0.5]).collect())
    }
}

#[test]
fn a_commit_resolves_encodes_and_applies_its_group_off_the_async_threads() {
    one_thread_for_work().block_on(async {
        let mut cluster = Cluster::new().await;
        let (paused, mut seen) = Paused::new(&mut cluster, "/log/");
        let gated = Arc::new(Gated::default());
        let writer = Arc::new(cluster.writer().await.embedding_with(gated.clone()));
        let mut embedding = schema("notes");
        embedding.embedding = Embedding {
            provider: EmbeddingProvider::Local,
            properties: vec!["title".into()],
            ..embedding.embedding
        };
        let create = Mutation::CreateCollection(Box::new(embedding));
        writer.commit(vec![create]).await.unwrap();
        let before = writer.partition().read(State::position);

        // With the one thread for such work taken, nothing is resolved,
        // so no vector is asked for.
        let (let_go, kept) = hostage();
        paused.armed.store(true, Ordering::SeqCst);
        let unembedded = Mutation::Upsert {
            collection: name("notes"),
            id: id(1),
            properties: json!({ "title": "a note" }).as_object().unwrap().clone(),
            vector: None,
            only_if_version: None,
        };
        let committing = tokio::spawn({
            let writer = writer.clone();
            async move { writer.commit(vec![unembedded]).await }
        });
        tokio::time::sleep(A_WHILE).await;
        assert_eq!(gated.calls.load(Ordering::SeqCst), 0, "resolved on its own");
        let_go.send(()).unwrap();
        kept.await.unwrap();

        // Taken again while the vector is made, the log is not encoded, so
        // nothing is stored.
        gated.asked.notified().await;
        let (let_go, kept) = hostage();
        gated.go_on.notify_one();
        tokio::time::sleep(A_WHILE).await;
        assert!(seen.try_recv().is_err(), "encoded on its own");
        let_go.send(()).unwrap();
        kept.await.unwrap();

        // And taken while the commit is made, the commit is not applied.
        held(&mut seen).await;
        let (let_go, kept) = hostage();
        paused.armed.store(false, Ordering::SeqCst);
        paused.release.notify_one();
        settled(|| async { cluster.committed().await.log_position > before }).await;
        tokio::time::sleep(A_WHILE).await;
        assert_eq!(
            writer.partition().read(State::position),
            before,
            "applied on its own"
        );
        assert!(!committing.is_finished());
        let_go.send(()).unwrap();
        kept.await.unwrap();
        committing.await.unwrap().unwrap();
        assert!(writer.partition().read(State::position) > before);
    });
}

#[test]
fn a_restore_readies_the_state_it_goes_back_to_before_it_holds_the_writer() {
    one_thread_for_work().block_on(async {
        let mut cluster = Cluster::new().await;
        let picky = Picky::over(&mut cluster, |key| key.ends_with(".vectors"));
        let writer = Arc::new(cluster.writer().await);
        let kept = writer
            .commit(vec![create("notes"), upsert("notes", 1, 1)])
            .await
            .unwrap()
            .position;
        let head = cluster.committed().await.manifest.unwrap();
        writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();

        // The last thing a restore asks before it makes ready the state it
        // goes back to is where its segment's vectors lie. With the one
        // thread for such work taken from then on, it waits for it, and
        // not with the writer held.
        picky.local_waits.store(true, Ordering::SeqCst);
        let restoring = tokio::spawn({
            let writer = writer.clone();
            async move { writer.restore(Some(&head), kept).await }
        });
        picky.waiting.notified().await;
        let (let_go, kept) = hostage();
        picky.local_waits.store(false, Ordering::SeqCst);
        picky.go_on.notify_one();
        tokio::time::sleep(A_WHILE).await;
        assert!(!restoring.is_finished(), "it waits for the thread");
        assert!(!writer.is_held());
        let_go.send(()).unwrap();
        kept.await.unwrap();
        restoring.await.unwrap().unwrap();
        assert_eq!(
            writer.partition().read(|state| n_of(state, "notes", 2)),
            None
        );
    });
}

#[tokio::test]
async fn cleaning_up_deletes_so_many_files_at_once_and_the_rest_the_next_time() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    // Half as many again as one call deletes, left by writers long gone
    // below the checkpoint.
    let left = Writer::CLEANED_AT_ONCE * 3 / 2;
    let author = |n: usize| Author {
        generation: Generation(0),
        writer: uuid::Uuid::from_u128(n as u128),
    };
    let layout = Layout(&cluster.key);
    for n in 0..left {
        let key = layout.segment(LogPosition(1), &author(n));
        cluster
            .blobs
            .put(&key, Bytes::from_static(b"x"))
            .await
            .unwrap();
    }
    tokio::time::sleep(Duration::from_millis(5)).await;

    let first = writer.clean(Duration::ZERO).await.unwrap();
    assert_eq!(first, Writer::CLEANED_AT_ONCE);
    let second = writer.clean(Duration::ZERO).await.unwrap();
    assert!(second >= left - first && second < Writer::CLEANED_AT_ONCE);
    assert_eq!(writer.clean(Duration::ZERO).await.unwrap(), 0);
    let author_of = |key: &String| key.contains(&format!("{:020}-{:020}-", 1, 0));
    assert!(!stored(&cluster, "segment/").await.iter().any(author_of));
    let reopened = cluster.open().await;
    let read = |state: &State| (n_of(state, "notes", 1), n_of(state, "notes", 2));
    assert_eq!(reopened.partition().read(read), (Some(1), Some(2)));
}

#[tokio::test(start_paused = true)]
async fn cleaning_up_a_store_that_answers_no_delete_gives_up_and_lets_a_flush_go_on() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |_| false);
    let writer = Arc::new(cluster.writer().await);
    mergeable(&writer).await;
    writer.merge().await.unwrap().expect("a merge");
    writer.flush().await.unwrap();

    picky.deletes_wait.store(true, Ordering::SeqCst);
    let started = Instant::now();
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    // A flush waits for it, and no longer than it asks for deletes.
    tokio::time::sleep(Duration::from_secs(1)).await;
    writer.commit(vec![upsert("notes", 101, 1)]).await.unwrap();
    ended(writer.flush()).await.unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(60));
    assert_eq!(cleaning.await.unwrap().unwrap(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_group_as_large_as_any_is_committed_long_before_its_writer_is_taken_for_stuck() {
    use vellev_storage::{Batch, Weight, GROUP_MOST};
    let cluster = Cluster::new().await;
    let writer = Arc::new(cluster.writer().await);
    writer.commit(vec![create("notes")]).await.unwrap();

    // As many batches of a hundred rows as a group takes, each row with as
    // much text as brings the group to its bound in bytes as well.
    let text = "lorem ipsum ".repeat(GROUP_MOST.bytes / GROUP_MOST.rows / 12 - 8);
    let row = |n: usize| Mutation::Upsert {
        collection: name("notes"),
        id: ObjectId(uuid::Uuid::from_u128(n as u128)),
        properties: json!({ "title": text, "n": n })
            .as_object()
            .unwrap()
            .clone(),
        vector: Some(vec![n as f32, 0.5]),
        only_if_version: None,
    };
    let mut weight = Weight::default();
    let group: Vec<Batch> = (0..GROUP_MOST.rows)
        .step_by(100)
        .map(|from| Batch::from((from..from + 100).map(row).collect::<Vec<_>>()))
        .take_while(|batch| {
            weight = weight + batch.weight();
            weight.within(&GROUP_MOST)
        })
        .collect();
    let weight: Weight = group.iter().map(Batch::weight).sum();
    assert!(weight.rows * 10 >= GROUP_MOST.rows * 9 && weight.bytes * 10 >= GROUP_MOST.bytes * 8);

    let watching = tokio::spawn({
        let writer = writer.clone();
        async move {
            let mut stalled = false;
            while Arc::strong_count(&writer) > 1 {
                stalled |= writer.stalled();
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            stalled
        }
    });

    let committed = writer.commit_batches(group, Default::default()).await;

    assert!(committed.unwrap().iter().all(Result::is_ok));
    drop(writer);
    assert!(!watching.await.unwrap(), "never taken for stuck");
}

#[tokio::test(start_paused = true)]
async fn a_restore_waits_for_cleaning_up_to_end_and_never_commits_what_it_is_deleting() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |_| false);
    let writer = Arc::new(cluster.writer().await);
    // Four segments, the state to go back to, and then one in their place.
    mergeable(&writer).await;
    let kept = cluster.committed().await;
    writer.merge().await.unwrap().expect("a merge");
    writer.flush().await.unwrap();

    // Cleaning up, with no grace, is deleting the four when a restore to
    // the state that lists them is asked for.
    picky.deletes_held.store(true, Ordering::SeqCst);
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    ended(picky.waiting.notified()).await;
    let restoring = tokio::spawn({
        let writer = writer.clone();
        async move {
            let restoring = writer.restore(kept.manifest.as_ref(), kept.log_position);
            restoring.await
        }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(!restoring.is_finished(), "it waits for the clean");
    picky.deletes_held.store(false, Ordering::SeqCst);
    picky.go_on.notify_waiters();
    assert!(ended(cleaning).await.unwrap().unwrap() >= 4);

    // What it would go back to is gone, so it fails, and commits nothing;
    // whatever is committed is all there.
    assert!(ended(restoring).await.unwrap().is_err());
    assert!(!writer.is_closed());
    let count = |state: &State| state.count(&name("notes"));
    let opened = cluster.open().await;
    assert_eq!(opened.partition().read(count), 41);
    assert_eq!(writer.partition().read(count), 41);
}

#[cfg(unix)]
#[tokio::test]
async fn vectors_whose_copy_on_local_disk_never_answers_fail_the_load_within_a_bound() {
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    let store = cluster.observed();
    let disk = Disk::new(Duration::from_millis(100), 1024 * 1024, 4);
    let cached = CachedBlobs::new(store.clone(), dir.path()).on(disk);
    cluster.blobs = Arc::new(cached);
    let writer = cluster.writer().await;
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let vectors = stored(&cluster, "/vectors/").await.remove(0);
    store.fetched();

    // The copy is there and its disk says nothing: the load fails, to be
    // tried again, and the vectors are not fetched to be held in memory.
    let _silent = Fifo::instead_of(&dir.path().join(&vectors));
    let opening = tokio::time::timeout(
        Duration::from_secs(30),
        cluster.open_on(cluster.blobs.clone()),
    );
    let unreached = opening.await.expect("within its bound").unwrap_err();
    assert!(
        matches!(&unreached, StorageError::Blob { key, .. } if *key == vectors),
        "{unreached}"
    );
    assert!(!store.fetched().contains(&vectors));
}

#[tokio::test(start_paused = true)]
async fn cleaning_up_that_never_learns_what_to_delete_gives_up_and_lets_a_flush_go_on() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |_| false);
    let writer = Arc::new(cluster.writer().await);
    mergeable(&writer).await;

    picky.lists_wait.store(true, Ordering::SeqCst);
    let started = Instant::now();
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    tokio::time::sleep(Duration::from_secs(1)).await;
    ended(writer.flush()).await.unwrap();
    assert_eq!(started.elapsed(), Duration::from_secs(60));
    let given_up = cleaning.await.unwrap().unwrap_err();
    assert!(matches!(given_up, StorageError::Blob { .. }), "{given_up}");
    assert!(!writer.is_closed());
}

#[tokio::test(start_paused = true)]
async fn cleaning_up_stops_deleting_once_another_writer_has_the_partition() {
    let mut cluster = Cluster::new().await;
    let picky = Picky::over(&mut cluster, |_| false);
    let writer = Arc::new(cluster.writer().await);
    writer
        .commit(vec![create("notes"), upsert("notes", 1, 1)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    writer.commit(vec![upsert("notes", 2, 2)]).await.unwrap();
    writer.flush().await.unwrap();
    // Litter below the checkpoint, several lots of it.
    let litter = 100;
    let layout = Layout(&cluster.key);
    for n in 0..litter {
        let author = Author {
            generation: Generation(0),
            writer: uuid::Uuid::from_u128(n),
        };
        let key = layout.segment(LogPosition(1), &author);
        let stored = cluster.blobs.put(&key, Bytes::from_static(b"x"));
        stored.await.unwrap();
    }
    let left = || async {
        let litter = |key: &String| key.contains(&format!("{:020}-{:020}-", 1, 0));
        let segments = stored(&cluster, "segment/").await;
        segments.into_iter().filter(litter).count()
    };

    // The first lot is being deleted when another writer takes the lease.
    picky.deletes_held.store(true, Ordering::SeqCst);
    let cleaning = tokio::spawn({
        let writer = writer.clone();
        async move { writer.clean(Duration::ZERO).await }
    });
    ended(picky.waiting.notified()).await;
    let _successor = cluster.writer().await;
    picky.deletes_held.store(false, Ordering::SeqCst);
    picky.go_on.notify_waiters();

    // That lot goes, as it was asked for; no other is.
    let stopped = ended(cleaning).await.unwrap().unwrap_err();
    assert!(
        matches!(stopped, StorageError::Diverged { .. }),
        "{stopped}"
    );
    let left = left().await;
    assert!(
        left >= litter as usize - 16 && left < litter as usize,
        "{left}"
    );
}

/// Blobs whose files come a part at a time: the first part, and never
/// another.
#[derive(Debug)]
struct Stalling(Arc<dyn Blobs>);

#[async_trait]
impl Blobs for Stalling {
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
        self.0.delete(key).await
    }
    fn parts<'a>(
        &'a self,
        key: &'a str,
    ) -> futures::stream::BoxStream<'a, vellev_storage::Result<Bytes>> {
        let first = futures::stream::once(async move {
            let whole = self.0.get(key).await?;
            Ok(whole.slice(..whole.len() / 2))
        });
        first.chain(futures::stream::pending()).boxed()
    }
}

#[tokio::test]
async fn a_fetch_into_the_cache_given_up_part_way_leaves_nothing_part_written() {
    let dir = tempfile::tempdir().unwrap();
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let (key, frame) = entry(1);
    store.put(&key, frame).await.unwrap();
    let cached = Arc::new(CachedBlobs::new(Arc::new(Stalling(store)), dir.path()));
    let under = dir.path().join(&key).parent().unwrap().to_path_buf();
    let part_written = || {
        let listed = std::fs::read_dir(&under).into_iter().flatten().flatten();
        listed
            .filter(|file| file.file_name().to_string_lossy().contains("partial-"))
            .count()
    };
    let soon = |holds: &dyn Fn() -> bool| {
        let started = std::time::Instant::now();
        while !holds() && started.elapsed() < Duration::from_secs(30) {
            std::thread::sleep(Duration::from_millis(5));
        }
        holds()
    };

    // Three callers, each gone once its first part is on disk.
    for _ in 0..3 {
        let fetching = tokio::spawn({
            let (cached, key) = (cached.clone(), key.clone());
            async move { cached.fetch_local(&key).await }
        });
        let begun = tokio::task::spawn_blocking({
            let under = under.clone();
            move || {
                let started = std::time::Instant::now();
                let written = |under: &std::path::Path| {
                    let listed = std::fs::read_dir(under).into_iter().flatten().flatten();
                    let sizes = listed.filter_map(|file| file.metadata().ok());
                    sizes.map(|meta| meta.len()).sum::<u64>() > 0
                };
                while !written(&under) && started.elapsed() < Duration::from_secs(30) {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        });
        begun.await.unwrap();
        assert_eq!(part_written(), 1, "part of it is written");
        fetching.abort();
        assert!(fetching.await.unwrap_err().is_cancelled());
        tokio::task::yield_now().await;
        assert!(soon(&|| part_written() == 0), "nothing of it is left");
    }
    assert!(!dir.path().join(&key).exists());
}
