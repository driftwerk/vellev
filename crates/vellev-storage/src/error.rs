use vellev_core::{CollectionName, ObjectError, SchemaError};

use crate::coordinator::CoordinatorError;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error(transparent)]
    Coordinator(#[from] CoordinatorError),
    #[error("object {key} is missing")]
    Missing { key: String },
    #[error("object store failed on {key}: {source}")]
    Blob {
        key: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("{key} is corrupt: {reason}")]
    Corrupt { key: String, reason: String },
    #[error("{key} has format version {found}, newer than the supported {supported}")]
    UnsupportedVersion {
        key: String,
        found: u16,
        supported: u16,
    },
    #[error("collection {0} does not exist")]
    UnknownCollection(CollectionName),
    #[error("collection {0} already exists")]
    CollectionExists(CollectionName),
    #[error(transparent)]
    Schema(#[from] SchemaError),
    #[error(transparent)]
    Object(#[from] ObjectError),
    #[error("the committed history no longer contains {head:?}; reopen the partition")]
    Diverged { head: Option<String> },
    #[error("the writer closed after a write whose outcome is unknown; reopen the partition")]
    Closed,
    #[error("encoding failed: {0}")]
    Encode(String),
}

impl StorageError {
    pub(crate) fn corrupt(key: &str, reason: impl Into<String>) -> Self {
        Self::Corrupt {
            key: key.to_string(),
            reason: reason.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, StorageError>;
