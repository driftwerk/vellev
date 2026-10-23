mod common;

use common::*;
use vellev_storage::{Merging, Query};

async fn segments_of(cluster: &Cluster) -> Vec<SegmentMeta> {
    let head = cluster.committed().await.manifest.unwrap().0;
    let bytes = cluster.blobs.get(&head).await.unwrap();
    let manifest: Manifest = format::decode(&head, Kind::Manifest, &bytes).unwrap();
    manifest.segments
}

fn vector_of(state: &State, object: u8) -> Option<Vec<f32>> {
    state
        .get(&name("notes"), id(object))
        .unwrap()
        .vector
        .map(<[f32]>::to_vec)
}

#[tokio::test]
async fn a_segment_file_carries_no_vectors_and_its_column_does() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();

    let segment = segments_of(&cluster).await.remove(0);
    let vectors = segment.vectors.clone().expect("rows with vectors");
    assert!(vectors.contains("/vectors/") && vectors.ends_with(".vectors"));
    let bytes = cluster.blobs.get(&segment.key).await.unwrap();
    let file: SegmentFile = format::decode(&segment.key, Kind::Segment, &bytes).unwrap();
    assert_eq!(file.rows.len(), 10);
    assert!(file.rows.iter().all(|row| row.object.vector.is_none()));

    for copy in [
        writer.partition().clone(),
        cluster.open().await.partition().clone(),
    ] {
        copy.read(|state| {
            assert_eq!(vector_of(state, 3), Some(vec![3.0, 0.5]));
            let hits = state
                .search(&name("notes"), &Query::nearest(&[7.0, 0.5], 1))
                .unwrap();
            assert_eq!(hits[0].object.id, id(7));
            assert_eq!(hits[0].vector, &[7.0, 0.5]);
            let listed = state.list(&name("notes"), None, None, 3).unwrap();
            assert!(listed.iter().all(|found| found.vector.is_some()));
        });
    }

    writer.commit(vec![upsert("notes", 3, 30)]).await.unwrap();
    writer
        .partition()
        .read(|state| assert_eq!(vector_of(state, 3), Some(vec![30.0, 0.5])));
    writer.flush().await.unwrap();
    writer.merge().await.unwrap();
    cluster.open().await.partition().read(|state| {
        assert_eq!(vector_of(state, 3), Some(vec![30.0, 0.5]));
        assert_eq!(vector_of(state, 4), Some(vec![4.0, 0.5]));
    });
}

#[tokio::test]
async fn a_segment_without_vectors_has_no_column() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let bare = |object: u8| Mutation::Upsert {
        collection: name("notes"),
        id: id(object),
        properties: json!({ "title": "no vector", "n": 1 })
            .as_object()
            .unwrap()
            .clone(),
        vector: None,
        only_if_version: None,
    };
    writer
        .commit(vec![create("notes"), bare(1), bare(2)])
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let segment = segments_of(&cluster).await.remove(0);
    assert_eq!(segment.vectors, None);
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(vector_of(state, 1), None));
}

#[tokio::test]
async fn a_node_with_a_local_copy_reads_vectors_in_place_and_survives_a_damaged_one() {
    let cluster = Cluster::new().await;
    let writer = cluster.writer().await;
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let vectors = segments_of(&cluster).await.remove(0).vectors.unwrap();

    let dir = tempfile::tempdir().unwrap();
    let store = cluster.observed();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store.clone(), dir.path()));
    let follower = cluster.open_on(cached.clone()).await.unwrap();
    let copy = dir.path().join(&vectors);
    assert!(copy.is_file(), "the column was copied to local disk");
    assert!(store.fetched().contains(&vectors));
    follower
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));

    std::fs::remove_file(&copy).unwrap();
    follower
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));

    std::fs::write(&copy, b"not a vectors file").unwrap();
    let again = cluster.open_on(cached.clone()).await.unwrap();
    assert!(store.fetched().contains(&vectors), "fetched again");
    again
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));
    assert!(format::verify(&vectors, &std::fs::read(&copy).unwrap()).is_ok());
}

#[tokio::test]
async fn a_store_on_local_disk_has_its_own_files_read_in_place() {
    let at = |local: Local| local.path;
    let dir = tempfile::tempdir().unwrap();
    let mut cluster = Cluster::new().await;
    let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path()).unwrap();
    cluster.blobs = Arc::new(ObjectBlobs::on_disk(Arc::new(local), dir.path()));
    let writer = cluster.writer().await;
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let vectors = segments_of(&cluster).await.remove(0).vectors.unwrap();
    assert_eq!(
        cluster.blobs.local(&vectors).await.unwrap().map(at),
        Some(dir.path().join(&vectors))
    );
    assert_eq!(
        cluster.blobs.local("elsewhere").await.unwrap().map(at),
        None
    );
    assert_eq!(cluster.blobs.local("../etc").await.unwrap().map(at), None);
    writer
        .partition()
        .read(|state| assert_eq!(vector_of(state, 2), Some(vec![2.0, 0.5])));
    cluster
        .open()
        .await
        .partition()
        .read(|state| assert_eq!(vector_of(state, 9), Some(vec![9.0, 0.5])));
}

/// Ten notes with vectors flushed into a segment, and the key of its
/// vectors.
async fn flushed(cluster: &Cluster) -> String {
    let writer = cluster.writer().await;
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    segments_of(cluster).await.remove(0).vectors.unwrap()
}

#[tokio::test]
async fn vectors_a_cache_cannot_keep_fail_the_load_rather_than_be_held_in_memory() {
    let cluster = Cluster::new().await;
    let vectors = flushed(&cluster).await;

    // A cache directory that is a file keeps nothing.
    let dir = tempfile::tempdir().unwrap();
    let nowhere = dir.path().join("a-file");
    std::fs::write(&nowhere, b"in the way").unwrap();
    let store = cluster.observed();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store.clone(), &nowhere));
    let unkept = cluster.open_on(cached).await.unwrap_err();
    assert!(
        matches!(&unkept, StorageError::Blob { key, .. } if *key == vectors),
        "to be tried again: {unkept}"
    );
    let fetched = store.fetched();
    let times = fetched.iter().filter(|key| **key == vectors).count();
    assert!(times <= 1, "{fetched:?}");
}

#[tokio::test]
async fn vectors_are_fetched_into_the_cache_and_read_there_never_held_in_memory() {
    let cluster = Cluster::new().await;
    let vectors = flushed(&cluster).await;
    let in_memory = cluster.open().await.partition().read(State::resident);

    let dir = tempfile::tempdir().unwrap();
    let store = cluster.observed();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(store.clone(), dir.path()));
    let fetched_once = |fetched: Vec<String>| {
        let times = fetched.iter().filter(|key| **key == vectors).count();
        assert_eq!(times, 1, "{fetched:?}");
    };
    // The first to open finds no copy: the file is fetched into one.
    let first = cluster.open_on(cached.clone()).await.unwrap();
    fetched_once(store.fetched());
    assert!(format::verify(&vectors, &std::fs::read(dir.path().join(&vectors)).unwrap()).is_ok());
    // The next reads the copy, and asks the store for nothing of it.
    let second = cluster.open_on(cached).await.unwrap();
    assert!(!store.fetched().contains(&vectors));
    for copy in [first, second] {
        let read = |state: &State| (vector_of(state, 6), state.resident());
        let (vector, resident) = copy.partition().read(read);
        assert_eq!(vector, Some(vec![6.0, 0.5]));
        // Ten vectors of two numbers of four bytes.
        assert_eq!(resident + 80, in_memory, "none of them in memory");
    }
}

/// A writer of the rows 0..5, flushed, then 5..10, flushed, on `blobs`
/// over its own cluster; merged too, if `merged`.
async fn built_on(cached: bool, merged: bool) -> (Writer, tempfile::TempDir) {
    let cluster = Cluster::new().await;
    let dir = tempfile::tempdir().unwrap();
    let blobs: Arc<dyn Blobs> = match cached {
        true => Arc::new(CachedBlobs::new(cluster.blobs.clone(), dir.path())),
        false => cluster.blobs.clone(),
    };
    let (lease, committed) = cluster.lease().await;
    let writer = Writer::open(blobs, cluster.coordinator.clone(), lease, &committed)
        .await
        .unwrap()
        .merging_with(Merging {
            fanout: 2,
            ..Merging::default()
        });
    writer.commit(vec![create("notes")]).await.unwrap();
    for half in [0..5, 5..10] {
        let rows = half.map(|n| upsert("notes", n, i64::from(n)));
        writer.commit(rows.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    if merged {
        assert!(
            writer.merge().await.unwrap().is_some(),
            "two segments merge"
        );
        assert_eq!(segments_of(&cluster).await.len(), 1);
    }
    (writer, dir)
}

/// The rule V1 holds a writer to (track 2): a segment it has just built,
/// by a flush or a merge, on a store that keeps copies on local disk, is
/// read from its copy there at once, holding none of its vectors in
/// memory until it is next loaded.
#[tokio::test]
async fn a_segment_just_built_on_a_store_keeping_copies_holds_no_vectors_in_memory() {
    for merged in [false, true] {
        let (held, _) = built_on(false, merged).await;
        let (read_in_place, _dir) = built_on(true, merged).await;
        let resident = |writer: &Writer| writer.partition().read(State::resident);
        // Ten vectors of two numbers of four bytes.
        assert_eq!(
            resident(&read_in_place) + 80,
            resident(&held),
            "merged: {merged}"
        );
        read_in_place.partition().read(|state| {
            assert_eq!(vector_of(state, 7), Some(vec![7.0, 0.5]));
            let hits = state
                .search(&name("notes"), &Query::nearest(&[7.0, 0.5], 1))
                .unwrap();
            assert_eq!(hits[0].vector, &[7.0, 0.5]);
        });
    }
}

/// A segment built on a store meant to keep copies on local disk, whose
/// copy cannot be kept there, is not held in memory instead: the flush or
/// merge fails, said aloud, nothing is committed, and the rows stay in
/// memory where they were, to be flushed once the disk takes them.
#[tokio::test]
async fn a_segment_built_whose_copy_the_cache_cannot_keep_fails_the_build_rather_than_be_held() {
    let cluster = Cluster::new().await;
    // A cache directory that is a file keeps nothing.
    let dir = tempfile::tempdir().unwrap();
    let nowhere = dir.path().join("a-file");
    std::fs::write(&nowhere, b"in the way").unwrap();
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(cluster.blobs.clone(), &nowhere));
    let (lease, committed) = cluster.lease().await;
    let writer = Writer::open(cached, cluster.coordinator.clone(), lease, &committed)
        .await
        .unwrap();
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    let before = cluster.committed().await.manifest;
    let failed = writer.flush().await;
    assert!(
        matches!(&failed, Err(StorageError::Blob { key, .. }) if key.contains("/vectors/")),
        "{failed:?}"
    );
    assert_eq!(
        cluster.committed().await.manifest,
        before,
        "nothing committed"
    );
    writer.partition().read(|state| {
        assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5]));
        assert_eq!(state.rows(), 10);
    });
    // Once the disk takes the copy, the same rows are flushed and read in place.
    std::fs::remove_file(&nowhere).unwrap();
    writer.flush().await.unwrap();
    assert_eq!(segments_of(&cluster).await.len(), 1);
    let (held, _) = built_on(false, false).await;
    let resident = |writer: &Writer| writer.partition().read(State::resident);
    assert!(resident(&writer) < resident(&held));
}

/// A store keeping copies whose copy of a vectors file it has just put is
/// left torn: cut short after it was written, as a disk that lost its end
/// would leave it.
#[derive(Debug)]
struct Tearing {
    inner: CachedBlobs,
    dir: std::path::PathBuf,
}

#[async_trait]
impl Blobs for Tearing {
    async fn put(&self, key: &str, bytes: Bytes) -> vellev_storage::Result<()> {
        self.inner.put(key, bytes).await?;
        if key.ends_with(".vectors") {
            let copy = std::fs::OpenOptions::new()
                .write(true)
                .open(self.dir.join(key))
                .unwrap();
            let length = copy.metadata().unwrap().len();
            copy.set_len(length / 2).unwrap();
        }
        Ok(())
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
        self.inner.local(key).await
    }
    async fn fetch_local(&self, key: &str) -> vellev_storage::Result<Option<Local>> {
        self.inner.fetch_local(key).await
    }
}

/// A segment built whose copy on local disk is torn is read in place from
/// a whole copy fetched afresh, not held in memory as built.
#[tokio::test]
async fn a_segment_built_whose_cached_copy_is_torn_is_read_from_one_fetched_whole() {
    let cluster = Cluster::new().await;
    let dir = tempfile::tempdir().unwrap();
    let tearing: Arc<dyn Blobs> = Arc::new(Tearing {
        inner: CachedBlobs::new(cluster.blobs.clone(), dir.path()),
        dir: dir.path().to_path_buf(),
    });
    let (lease, committed) = cluster.lease().await;
    let writer = Writer::open(tearing, cluster.coordinator.clone(), lease, &committed)
        .await
        .unwrap();
    let rows = (0..10).map(|n| upsert("notes", n, i64::from(n)));
    writer
        .commit([create("notes")].into_iter().chain(rows).collect())
        .await
        .unwrap();
    writer.flush().await.unwrap();
    let (held, _) = built_on(false, false).await;
    let resident = |writer: &Writer| writer.partition().read(State::resident);
    assert_eq!(resident(&writer) + 80, resident(&held), "none of them in memory");
    writer
        .partition()
        .read(|state| assert_eq!(vector_of(state, 6), Some(vec![6.0, 0.5])));
}

/// The cache directory `dir`, and every directory in it, made to take no
/// new file, or to take them again.
#[cfg(unix)]
fn writable(dir: &std::path::Path, writable: bool) {
    use std::os::unix::fs::PermissionsExt;
    let mode = if writable { 0o755 } else { 0o555 };
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            if entry.file_type().unwrap().is_dir() {
                dirs.push(entry.path());
            }
        }
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(mode)).unwrap();
    }
}

/// A merge whose segment's copy the cache cannot keep fails as a flush
/// does: nothing committed, the segments it would have replaced still
/// read, and the merge made once the disk takes the copy.
#[cfg(unix)]
#[tokio::test]
async fn a_merge_whose_copy_the_cache_cannot_keep_fails_and_changes_nothing() {
    let cluster = Cluster::new().await;
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache");
    let cached: Arc<dyn Blobs> = Arc::new(CachedBlobs::new(cluster.blobs.clone(), &cache));
    let (lease, committed) = cluster.lease().await;
    let writer = Writer::open(cached, cluster.coordinator.clone(), lease, &committed)
        .await
        .unwrap()
        .merging_with(Merging {
            fanout: 2,
            ..Merging::default()
        });
    writer.commit(vec![create("notes")]).await.unwrap();
    for half in [0..5, 5..10] {
        let rows = half.map(|n| upsert("notes", n, i64::from(n)));
        writer.commit(rows.collect()).await.unwrap();
        writer.flush().await.unwrap();
    }
    let before = cluster.committed().await.manifest;
    writable(&cache, false);
    let failed = writer.merge().await;
    writable(&cache, true);
    assert!(
        matches!(&failed, Err(StorageError::Blob { key, .. }) if key.contains("/vectors/")),
        "{failed:?}"
    );
    assert_eq!(cluster.committed().await.manifest, before, "nothing committed");
    assert_eq!(segments_of(&cluster).await.len(), 2);
    writer
        .partition()
        .read(|state| assert_eq!(vector_of(state, 7), Some(vec![7.0, 0.5])));
    assert!(writer.merge().await.unwrap().is_some());
    assert_eq!(segments_of(&cluster).await.len(), 1);
}
