use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error(
        "tenant id {0:?} must be 1-63 characters of a-z, 0-9, '-' or '_', starting with a-z or 0-9"
    )]
    Tenant(String),
    #[error(
        "collection name {0:?} must be 1-128 characters of A-Z, a-z, 0-9 or '_', starting with a letter"
    )]
    Collection(String),
}

/// One organisation or edge. A tenant's data lives under its own storage
/// prefix, so the id is restricted to characters that are safe in an object
/// key and can never contain a path separator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TenantId(String);

impl TenantId {
    pub fn new(id: impl Into<String>) -> Result<Self, IdError> {
        let id = id.into();
        let mut chars = id.chars();
        let first_ok = chars
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit());
        let rest_ok = chars.all(|c| matches!(c, 'a'..='z' | '0'..='9' | '-' | '_'));
        if first_ok && rest_ok && id.len() <= 63 {
            Ok(Self(id))
        } else {
            Err(IdError::Tenant(id))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CollectionName(String);

impl CollectionName {
    pub fn new(name: impl Into<String>) -> Result<Self, IdError> {
        let name = name.into();
        let mut chars = name.chars();
        let first_ok = chars.next().is_some_and(|c| c.is_ascii_alphabetic());
        let rest_ok = chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
        if first_ok && rest_ok && name.len() <= 128 {
            Ok(Self(name))
        } else {
            Err(IdError::Collection(name))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

macro_rules! string_id {
    ($ty:ident) => {
        impl TryFrom<String> for $ty {
            type Error = IdError;
            fn try_from(value: String) -> Result<Self, IdError> {
                Self::new(value)
            }
        }
        impl From<$ty> for String {
            fn from(value: $ty) -> String {
                value.0
            }
        }
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}
string_id!(TenantId);
string_id!(CollectionName);

/// A slice of a tenant's data with its own writer. Every tenant has exactly
/// one today; the key is in the model now so that splitting a large tenant
/// later changes routing, not the storage format.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Partition(pub u32);

impl Partition {
    pub const DEFAULT: Partition = Partition(0);
}

/// Identifies one running node for the lifetime of its process. A restarted
/// node gets a new id, so a lease it held before is never mistaken for its own.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(String);

impl NodeId {
    pub fn generate(host: &str) -> Self {
        Self(format!("{host}-{}", Uuid::new_v4().simple()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for NodeId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Supplied by the client or generated, so ids from an imported store survive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ObjectId(pub Uuid);

impl ObjectId {
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_ids_are_safe_as_storage_prefixes() {
        for ok in ["wf-patchwork", "edge_1", "0abc", "a"] {
            assert!(TenantId::new(ok).is_ok(), "{ok}");
        }
        for bad in ["", "-a", "A", "a/b", "a..b", "a b", "../x", &"a".repeat(64)] {
            assert_eq!(
                TenantId::new(bad),
                Err(IdError::Tenant(bad.to_string())),
                "{bad}"
            );
        }
    }

    #[test]
    fn collection_names_accept_workforce_names_and_reject_paths() {
        for ok in ["WorkforceMemory_bge_base_en_v1_5", "CodeEntity", "a"] {
            assert!(CollectionName::new(ok).is_ok(), "{ok}");
        }
        for bad in ["", "1abc", "_a", "a-b", "a/b", &"a".repeat(129)] {
            assert!(CollectionName::new(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn an_invalid_id_is_refused_when_deserialized() {
        assert!(serde_json::from_str::<TenantId>("\"../other\"").is_err());
        assert_eq!(
            serde_json::from_str::<TenantId>("\"wf-a\"").unwrap(),
            TenantId::new("wf-a").unwrap()
        );
    }
}
