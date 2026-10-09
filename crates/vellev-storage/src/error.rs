use std::sync::Arc;

use serde::{Deserialize, Serialize};

use vellev_core::{CollectionName, ModelFault, ObjectError, ObjectId, SchemaError};

use crate::coordinator::CoordinatorError;
use crate::mutation::Unembedded;

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
    /// A request for more of something than there may be at the most.
    #[error("{message}")]
    TooMany { over: Over, message: String },
    #[error("{0}")]
    InvalidQuery(String),
    /// Vectors the commit needed could not be made; with `lasting`, not
    /// until that fault is put right.
    #[error("embedding with {model} failed: {reason}")]
    Embedding {
        model: String,
        reason: String,
        lasting: Option<ModelFault>,
    },
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
    #[error("the partition is being restored; nothing was written, try again shortly")]
    Restoring,
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
    /// What the request named that was not there. A release from before
    /// this was kept neither reads nor keeps it, and a refusal that has
    /// been through one says only that something was not found. So does
    /// one that lacked a kind of thing this release has no name for,
    /// kept by a later one: it reads as nothing kept, not as a file that
    /// cannot be read.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "if_known"
    )]
    pub lacking: Option<Lacking>,
    /// What the request asked too much of, and the most it may ask: kept,
    /// and read by other releases, just as `lacking` is.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "if_known"
    )]
    pub over: Option<Over>,
}

/// What was kept, if it is something this release has a name for.
fn if_known<'de, D, T>(kept: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged, bound = "T: Deserialize<'de>")]
    enum Kept<T> {
        Known(T),
        Unknown(serde::de::IgnoredAny),
    }
    Ok(match Option::<Kept<T>>::deserialize(kept)? {
        Some(Kept::Known(known)) => Some(known),
        Some(Kept::Unknown(_)) | None => None,
    })
}

/// What a request asked too much of, and the most it may ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Over {
    pub capped: Capped,
    pub most: u64,
}

/// Something a request may ask for only so much of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capped {
    /// The objects one request deletes by filter.
    Deleted,
}

/// What a request named that was not there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lacking {
    Collection,
    Object,
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
    /// The failure to make vectors with `model`.
    pub fn unembedded(model: String, why: Unembedded) -> Self {
        Self::Embedding {
            model,
            reason: why.reason,
            lasting: why.lasting,
        }
    }

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
        let lacking = match self.root() {
            Self::UnknownCollection(_) => Some(Lacking::Collection),
            Self::UnknownObject { .. } => Some(Lacking::Object),
            _ => None,
        };
        let kind = match self.root() {
            Self::UnknownCollection(_) | Self::UnknownObject { .. } => Refused::NotFound,
            Self::CollectionExists(_)
            | Self::ObjectExists { .. }
            | Self::ConditionFailed { .. } => Refused::Conflict,
            Self::InvalidMutation(_)
            | Self::TooMany { .. }
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
            lacking,
            over: match self.root() {
                Self::TooMany { over, .. } => Some(*over),
                _ => None,
            },
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
