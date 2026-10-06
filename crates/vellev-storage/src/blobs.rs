//! Immutable files in object storage. A key is written once and never
//! changes, which is what makes caching them on local disk safe.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::{ObjectStore, PutPayload};

use crate::coordinator::{Generation, LogPosition, PartitionKey};
use crate::format::{verify, Kind};
use crate::{Result, StorageError};

#[async_trait]
pub trait Blobs: fmt::Debug + Send + Sync {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()>;
    async fn get(&self, key: &str) -> Result<Bytes>;
    /// Every file whose key starts with `prefix`.
    async fn list(&self, prefix: &str) -> Result<Vec<Listed>>;
    /// Remove a file; one that is not there is no error.
    async fn delete(&self, key: &str) -> Result<()>;
}

/// A file as a listing shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Listed {
    pub key: String,
    pub modified: chrono::DateTime<chrono::Utc>,
}

/// Who wrote a file: the lease's generation and the writer instance. Two
/// writers never share both, so no two writers can produce the same key, even
/// when one is reopened on a lease that is still current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Author {
    pub generation: Generation,
    pub writer: uuid::Uuid,
}

/// Where a partition's files live.
#[derive(Debug, Clone, Copy)]
pub struct Layout<'a>(pub &'a PartitionKey);

impl Layout<'_> {
    pub fn prefix(&self) -> String {
        format!("{}/{}/", self.0.tenant, self.0.partition.0)
    }

    pub fn log(&self, position: LogPosition, author: &Author) -> String {
        self.key("log", position, author)
    }

    pub fn segment(&self, position: LogPosition, author: &Author) -> String {
        self.key("segment", position, author)
    }

    pub fn manifest(&self, position: LogPosition, author: &Author) -> String {
        self.key("manifest", position, author)
    }

    /// A segment made by merging others at `position`, named apart from the
    /// one a flush at the same position writes.
    pub fn merged(&self, position: LogPosition, author: &Author) -> String {
        self.named("segment", position, author, "-merged")
    }

    fn key(&self, kind: &str, position: LogPosition, author: &Author) -> String {
        self.named(kind, position, author, "")
    }

    fn named(&self, kind: &str, position: LogPosition, author: &Author, suffix: &str) -> String {
        format!(
            "{}{kind}/{:020}-{:020}-{}{suffix}.{kind}",
            self.prefix(),
            position.0,
            author.generation.0,
            author.writer.simple()
        )
    }

    /// What kind of file a key under this partition names and the position
    /// it was written at, or `None` for a key not laid out this way.
    pub(crate) fn parse(&self, key: &str) -> Option<(Kind, LogPosition)> {
        let (kind, name) = key.strip_prefix(&self.prefix())?.split_once('/')?;
        let kind = match kind {
            "log" => Kind::Log,
            "segment" => Kind::Segment,
            "manifest" => Kind::Manifest,
            _ => return None,
        };
        let position = name.get(..20)?.parse().ok()?;
        Some((kind, LogPosition(position)))
    }
}

/// Whether `key` is a plain path: no empty, `.` or `..` parts and no
/// backslash. Keys read out of files are data, and only a plain one names
/// the same object to every store and to the cache.
pub(crate) fn is_plain(key: &str) -> bool {
    !key.contains('\\') && key.split('/').all(|part| !matches!(part, "" | "." | ".."))
}

#[derive(Debug, Clone)]
pub struct ObjectBlobs(Arc<dyn ObjectStore>);

impl ObjectBlobs {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self(store)
    }
}

fn blob_error(key: &str, error: object_store::Error) -> StorageError {
    match error {
        object_store::Error::NotFound { .. } => StorageError::Missing {
            key: key.to_string(),
        },
        other => StorageError::Blob {
            key: key.to_string(),
            source: Box::new(other),
        },
    }
}

#[async_trait]
impl Blobs for ObjectBlobs {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
        self.0
            .put(
                &object_store::path::Path::from(key),
                PutPayload::from_bytes(bytes),
            )
            .await
            .map(|_| ())
            .map_err(|e| blob_error(key, e))
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        let found = self
            .0
            .get(&object_store::path::Path::from(key))
            .await
            .map_err(|e| blob_error(key, e))?;
        found.bytes().await.map_err(|e| blob_error(key, e))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
        use futures::TryStreamExt;
        let under = object_store::path::Path::from(prefix);
        self.0
            .list(Some(&under))
            .map_ok(|meta| Listed {
                key: meta.location.to_string(),
                modified: meta.last_modified,
            })
            .try_collect()
            .await
            .map_err(|e| blob_error(prefix, e))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        match self.0.delete(&object_store::path::Path::from(key)).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(blob_error(key, error)),
        }
    }
}

/// Keeps a copy of every file read or written under `dir`. The cache is an
/// optimisation and never the authority: an entry that fails its checksum is
/// dropped and read from the store again, and a failure to write one is
/// logged and otherwise ignored.
#[derive(Debug, Clone)]
pub struct CachedBlobs {
    inner: Arc<dyn Blobs>,
    dir: PathBuf,
}

static PARTIAL: AtomicU64 = AtomicU64::new(0);

impl CachedBlobs {
    pub fn new(inner: Arc<dyn Blobs>, dir: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            dir: dir.into(),
        }
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        is_plain(key).then(|| self.dir.join(key))
    }

    async fn cached(&self, key: &str, path: &Path) -> Option<Bytes> {
        let bytes = Bytes::from(tokio::fs::read(path).await.ok()?);
        match verify(key, &bytes) {
            Ok(()) => Some(bytes),
            Err(error) => {
                tracing::warn!(key, %error, "dropping a damaged cache entry");
                let _ = tokio::fs::remove_file(path).await;
                None
            }
        }
    }

    async fn remember(&self, key: &str, bytes: &Bytes) {
        let Some(path) = self.path(key) else { return };
        if let Err(error) = write_atomically(&path, bytes).await {
            tracing::warn!(key, %error, "could not cache a file on local disk");
        }
    }
}

/// Written under a name of its own and then renamed, so a reader never sees
/// a partly written file. Nothing is synced: an entry torn by a crash fails
/// its checksum and is fetched again.
async fn write_atomically(path: &Path, bytes: &Bytes) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache path has no parent"))?;
    tokio::fs::create_dir_all(parent).await?;
    let unique = PARTIAL.fetch_add(1, Ordering::Relaxed);
    let partial = path.with_extension(format!("partial-{}-{unique}", std::process::id()));
    let written = async {
        tokio::fs::write(&partial, bytes).await?;
        tokio::fs::rename(&partial, path).await
    }
    .await;
    if written.is_err() {
        let _ = tokio::fs::remove_file(&partial).await;
    }
    written
}

#[async_trait]
impl Blobs for CachedBlobs {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
        self.inner.put(key, bytes.clone()).await?;
        self.remember(key, &bytes).await;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        if let Some(path) = self.path(key) {
            if let Some(cached) = self.cached(key, &path).await {
                return Ok(cached);
            }
        }
        let bytes = self.inner.get(key).await?;
        self.remember(key, &bytes).await;
        Ok(bytes)
    }

    async fn list(&self, prefix: &str) -> Result<Vec<Listed>> {
        self.inner.list(prefix).await
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.inner.delete(key).await?;
        if let Some(path) = self.path(key) {
            let _ = tokio::fs::remove_file(path).await;
        }
        Ok(())
    }
}
