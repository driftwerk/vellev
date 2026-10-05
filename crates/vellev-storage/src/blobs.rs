//! Immutable files in object storage. A key is written once and never
//! changes, which is what makes caching them on local disk safe.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use object_store::{ObjectStore, PutPayload};

use crate::coordinator::{Generation, LogPosition, PartitionKey};
use crate::{Result, StorageError};

#[async_trait]
pub trait Blobs: fmt::Debug + Send + Sync {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()>;
    async fn get(&self, key: &str) -> Result<Bytes>;
}

/// Where a partition's files live. Every name carries the position and the
/// writer's generation, so two writers can never produce the same key.
#[derive(Debug, Clone, Copy)]
pub struct Layout<'a>(pub &'a PartitionKey);

impl Layout<'_> {
    pub fn log(&self, position: LogPosition, generation: Generation) -> String {
        self.key("log", position, generation)
    }

    pub fn segment(&self, position: LogPosition, generation: Generation) -> String {
        self.key("segment", position, generation)
    }

    pub fn manifest(&self, position: LogPosition, generation: Generation) -> String {
        self.key("manifest", position, generation)
    }

    fn key(&self, kind: &str, position: LogPosition, generation: Generation) -> String {
        format!(
            "{}/{}/{kind}/{:020}-{:020}.{kind}",
            self.0.tenant, self.0.partition.0, position.0, generation.0
        )
    }
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
}

/// Keeps a copy of every file read or written under `dir`. The cache is an
/// optimisation: a failure to use it is logged and the store is used instead.
#[derive(Debug, Clone)]
pub struct CachedBlobs {
    inner: Arc<dyn Blobs>,
    dir: PathBuf,
}

impl CachedBlobs {
    pub fn new(inner: Arc<dyn Blobs>, dir: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            dir: dir.into(),
        }
    }

    async fn remember(&self, key: &str, bytes: &Bytes) {
        if let Err(error) = write_atomically(&self.dir.join(key), bytes).await {
            tracing::warn!(key, %error, "could not cache a file on local disk");
        }
    }
}

/// Written under a temporary name and renamed, so a reader never sees a
/// partly written file and a crash leaves no truncated entry behind.
async fn write_atomically(path: &Path, bytes: &Bytes) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache path has no parent"))?;
    tokio::fs::create_dir_all(parent).await?;
    let partial = path.with_extension(format!("partial-{}", std::process::id()));
    tokio::fs::write(&partial, bytes).await?;
    tokio::fs::rename(&partial, path).await
}

#[async_trait]
impl Blobs for CachedBlobs {
    async fn put(&self, key: &str, bytes: Bytes) -> Result<()> {
        self.inner.put(key, bytes.clone()).await?;
        self.remember(key, &bytes).await;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Bytes> {
        if let Ok(cached) = tokio::fs::read(self.dir.join(key)).await {
            return Ok(Bytes::from(cached));
        }
        let bytes = self.inner.get(key).await?;
        self.remember(key, &bytes).await;
        Ok(bytes)
    }
}
