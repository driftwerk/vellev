use std::sync::Arc;

use vellev_core::{CollectionName, ObjectError, ObjectId, SchemaError};

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
    #[error("object {id} does not exist in {collection}")]
    UnknownObject {
        collection: CollectionName,
        id: ObjectId,
    },
    #[error("object {id} already exists in {collection}")]
    ObjectExists {
        collection: CollectionName,
        id: ObjectId,
    },
    #[error("object {id} in {collection} does not have the expected {property:?}")]
    ConditionFailed {
        collection: CollectionName,
        id: ObjectId,
        property: String,
    },
    #[error("{0}")]
    InvalidMutation(String),
    #[error("{0}")]
    InvalidQuery(String),
    #[error("embedding with {model} failed: {reason}")]
    Embedding { model: String, reason: String },
    #[error(transparent)]
    Schema(#[from] SchemaError),
    #[error(transparent)]
    Filter(#[from] vellev_index::filter::FilterError),
    #[error(transparent)]
    Object(#[from] ObjectError),
    #[error("the committed history no longer contains {head:?}; reopen the partition")]
    Diverged { head: Option<String> },
    #[error("the writer closed after a write whose outcome is unknown; open a new writer")]
    Closed,
    #[error("{0} are exhausted")]
    Exhausted(&'static str),
    #[error("encoding failed: {0}")]
    Encode(String),
    #[error("work off the async threads did not finish: {0}")]
    Background(String),
    /// A failure of a commit shared by several callers, as each of them sees it.
    #[error(transparent)]
    Shared(Arc<StorageError>),
}

impl StorageError {
    /// The failure itself, looking through [`StorageError::Shared`].
    pub fn root(&self) -> &StorageError {
        match self {
            Self::Shared(shared) => shared.root(),
            other => other,
        }
    }

    pub(crate) fn corrupt(key: &str, reason: impl Into<String>) -> Self {
        Self::Corrupt {
            key: key.to_string(),
            reason: reason.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, StorageError>;
