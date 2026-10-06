use std::sync::Arc;

use serde::{Deserialize, Serialize};

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
    /// A request refused before, refused again as it was then.
    #[error("{}", .0.message)]
    Refused(Refusal),
}

/// Why a request was refused, kept so that a try of it again is refused
/// alike: what kind of refusal it was, and what it said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Refusal {
    pub kind: Refused,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refused {
    /// Something the request names does not exist.
    NotFound,
    /// What it would do clashes with what is stored.
    Conflict,
    /// It is not a request that can be made.
    Invalid,
}

impl StorageError {
    /// The failure itself, looking through [`StorageError::Shared`].
    pub fn root(&self) -> &StorageError {
        match self {
            Self::Shared(shared) => shared.root(),
            other => other,
        }
    }

    /// What refusal the failure is, if it is the request's own doing and
    /// so would be the same however often the request were made: not a
    /// failure that may pass.
    pub fn refusal(&self) -> Option<Refusal> {
        let kind = match self.root() {
            Self::UnknownCollection(_) | Self::UnknownObject { .. } => Refused::NotFound,
            Self::CollectionExists(_)
            | Self::ObjectExists { .. }
            | Self::ConditionFailed { .. } => Refused::Conflict,
            Self::InvalidMutation(_)
            | Self::InvalidQuery(_)
            | Self::Schema(_)
            | Self::Filter(_)
            | Self::Object(_) => Refused::Invalid,
            Self::Refused(refusal) => return Some(refusal.clone()),
            _ => return None,
        };
        Some(Refusal {
            kind,
            message: self.root().to_string(),
        })
    }

    pub(crate) fn corrupt(key: &str, reason: impl Into<String>) -> Self {
        Self::Corrupt {
            key: key.to_string(),
            reason: reason.into(),
        }
    }
}

pub type Result<T> = std::result::Result<T, StorageError>;
