//! What goes over the wire between a client and a node: the bodies of the
//! HTTP API's requests and responses.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use vellev_core::{Collection, CollectionName, ModelFault, ObjectId, Property, TenantId, Version};
use vellev_index::filter::Filter;
use vellev_index::fusion::Fusion;

/// The path every versioned route starts with.
pub const ROOT: &str = "/v1";

/// A request header naming what a tenant's data must have reached before a
/// read is answered, as [`After`] writes it: what a write of the caller's
/// own answered with.
pub const AFTER: &str = "x-vellev-after";

/// A request header choosing a read's [`Consistency`].
pub const CONSISTENCY: &str = "x-vellev-consistency";

/// A request header naming the person a read is for, inside a tenant: a
/// JWT signed with Ed25519 by a key the node trusts, naming the tenant, the
/// person's `identity_id` and `teams`, and when it expires (`exp`).
pub const IDENTITY: &str = "x-vellev-identity";

/// A request header naming a write with a UUID, the same on every try of
/// it, so that a try made after the first was answered is answered as it
/// was rather than made again.
pub const REQUEST: &str = "x-vellev-request";

/// A point in a tenant's history: its incarnation, which a tenant dropped
/// and created again does not share, and a position in it. Written as
/// `incarnation.position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct After {
    pub incarnation: u64,
    pub position: u64,
}

impl After {
    /// The later of two points: within one incarnation the greater
    /// position, otherwise the later incarnation, whose history replaces
    /// the other's.
    pub fn later(self, other: After) -> After {
        match self.incarnation.cmp(&other.incarnation) {
            std::cmp::Ordering::Less => other,
            std::cmp::Ordering::Greater => self,
            std::cmp::Ordering::Equal if other.position > self.position => other,
            std::cmp::Ordering::Equal => self,
        }
    }
}

impl std::fmt::Display for After {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.incarnation, self.position)
    }
}

impl std::str::FromStr for After {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let malformed = || format!("{text:?} is not incarnation.position");
        let (incarnation, position) = text.split_once('.').ok_or_else(malformed)?;
        Ok(After {
            incarnation: incarnation.parse().map_err(|_| malformed())?,
            position: position.parse().map_err(|_| malformed())?,
        })
    }
}

/// How fresh a read must be. A node that writes the tenant always answers
/// with everything committed, whatever is asked.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Consistency {
    /// At least as far as [`AFTER`] says, when it is sent; otherwise as
    /// [`Consistency::Eventual`].
    #[default]
    Session,
    /// Everything committed when the read arrived.
    Strong,
    /// Whatever the node has, at most about a second behind.
    Eventual,
}

impl std::str::FromStr for Consistency {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "session" => Ok(Consistency::Session),
            "strong" => Ok(Consistency::Strong),
            "eventual" => Ok(Consistency::Eventual),
            other => Err(format!(
                "{other:?} is not a consistency: session, strong or eventual"
            )),
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Code {
    // Each is numbered, so that a code added later leaves the others as
    // they were for whoever casts one to a number.
    /// The request is malformed or asks for something that cannot be.
    Invalid = 0,
    Unauthorized = 1,
    /// The key is valid, and does not reach what the request asks for.
    Forbidden = 2,
    NotFound = 3,
    /// The request disagrees with what is stored.
    Conflict = 4,
    /// The tenant is asking more of the node than it may; the request had
    /// no effect and may be sent again after the `Retry-After` it gives.
    Limited = 5,
    /// The node cannot serve this now, and the request has had no effect;
    /// the same request may succeed later or on another node.
    Unavailable = 6,
    /// A write failed at a point where nobody can say whether it took
    /// effect. Read before deciding to send it again.
    Uncertain = 7,
    /// This node cannot use the collection's embedding model until its
    /// setup is put right: the build cannot run the model, the node may
    /// not write where the model is kept, whatever serves it says it has
    /// no such model or will not serve this account, or its vectors are
    /// not the collection's size. The request had no effect. Another node
    /// may be able to do it now, and this one once it is put right, so
    /// the request is to be kept and sent again, never dropped; one that
    /// brings its own vector needs no model.
    ModelUnavailable = 10,
    Internal = 8,
    /// A code this client does not know, from a newer node.
    #[serde(other)]
    Unknown = 9,
}

impl Code {
    /// The HTTP status a failure of this kind is answered with, unless
    /// HTTP has a status of its own for it.
    pub fn status(self) -> u16 {
        match self {
            Code::Invalid => 400,
            Code::Unauthorized => 401,
            Code::Forbidden => 403,
            Code::NotFound => 404,
            Code::Conflict => 409,
            Code::Limited => 429,
            Code::Unavailable | Code::Uncertain => 503,
            // The node's failing, not the caller's, so a server error; and
            // of those the one that says the node cannot do what is asked
            // as it is set up, which HTTP clients and proxies do not send
            // again of their own accord as they do 500, 502, 503 and 504.
            Code::ModelUnavailable => 501,
            Code::Internal | Code::Unknown => 500,
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Problem {
    pub code: Code,
    pub message: String,
    /// On a refusal of a request that asks for more than the node allows:
    /// what it asked too much of, and the most it may ask. A request that
    /// asks for none at all is refused without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<Limit>,
    /// On `not_found`: what the request names that does not exist, when
    /// the node can say. A write refused before and named again is refused
    /// as it was, without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing: Option<Missing>,
    /// On `model_unavailable`: where the fault lies, and so who is to put
    /// it right. It says nothing of what was asked or of what answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
    /// On `conflict` from a write conditional on an object, by `only_if`
    /// or `only_if_version`: the version the object is at now, so that
    /// the caller can decide without reading it again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<Version>,
}

impl Problem {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            limit: None,
            missing: None,
            reason: None,
            version: None,
        }
    }
}

/// Why a node cannot use a collection's embedding model as it is set up.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Reason {
    /// The node's build cannot run the model; a node of another build
    /// may.
    Build,
    /// The node is set up wrong for the model: it may not write where the
    /// model is kept, or what serves the model does not know the node's
    /// key. A node set up right may answer.
    Setup,
    /// What serves the model says it has no such model, or its vectors
    /// are not the collection's size: likely the same of every node,
    /// until the collection or what serves it is changed.
    Provider,
    /// Something this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
}

impl From<ModelFault> for Reason {
    fn from(fault: ModelFault) -> Self {
        match fault {
            ModelFault::Build => Reason::Build,
            ModelFault::Setup => Reason::Setup,
            ModelFault::Provider => Reason::Provider,
            // A fault named since this was written.
            _ => Reason::Unknown,
        }
    }
}

/// Something a request may ask for only so much of, as [`Limits`] names
/// it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Bounded {
    /// Results of a search or a page, asked for as `limit`.
    Results,
    /// Results each side of a hybrid search offers for fusing.
    Candidates,
    /// Candidates scored exactly for each result.
    Oversample,
    /// Bytes of a request's body.
    BodyBytes,
    /// Seconds of a tenant's restore window.
    RestoreWindowSecs,
    /// Objects one request deletes by filter: a delete's `limit`, and
    /// the limits of every delete by filter of one request together, one
    /// without a limit counting for the most.
    Deleted,
    /// Searches, listings and counts a node answers at once. No one
    /// request asks for more of these than it may: a node answering as
    /// many says `unavailable`, to be asked again.
    ReadsAtOnce,
    /// Something this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
}

/// The most of something a request may ask for; at least one is always
/// allowed.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Limit {
    pub name: Bounded,
    pub max: u64,
}

impl Limit {
    pub fn new(name: Bounded, max: u64) -> Self {
        Self { name, max }
    }
}

/// What a request named that does not exist.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Missing {
    /// Also what a tenant's key is told of every other tenant.
    Tenant,
    Collection,
    Object,
    /// One of a tenant's keys.
    Key,
    Backup,
    /// A state of the tenant within its restore window.
    RestorePoint,
    /// The route itself: the node has none for the request. A node from
    /// before a route was added says so too, and one from before this
    /// field says `not_found` without it.
    Route,
    /// Something this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
}

/// The most a caller may ask of the node, each named as [`Bounded`]
/// names it. A node always says all of these; zero stands for one it did
/// not say.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct Limits {
    pub results: u64,
    pub candidates: u64,
    pub oversample: u64,
    pub body_bytes: u64,
    pub restore_window_secs: u64,
    pub deleted: u64,
    pub reads_at_once: u64,
}

impl Limits {
    /// The most of `name` a request may ask for, if the node said and
    /// this client knows it.
    pub fn max(&self, name: Bounded) -> Option<u64> {
        let said = match name {
            Bounded::Results => self.results,
            Bounded::Candidates => self.candidates,
            Bounded::Oversample => self.oversample,
            Bounded::BodyBytes => self.body_bytes,
            Bounded::RestoreWindowSecs => self.restore_window_secs,
            Bounded::Deleted => self.deleted,
            Bounded::ReadsAtOnce => self.reads_at_once,
            Bounded::Unknown => 0,
        };
        (said > 0).then_some(said)
    }
}

/// The body of every response that is not a success.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Failure {
    pub error: Problem,
}

impl From<Problem> for Failure {
    fn from(error: Problem) -> Self {
        Self { error }
    }
}

impl Failure {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            error: Problem::new(code, message),
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema, utoipa::IntoParams))]
#[cfg_attr(feature = "openapi", into_params(parameter_in = Query))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct TenantPage {
    /// What the page before gave as `next`.
    pub after: Option<TenantId>,
    #[cfg_attr(feature = "openapi", param(minimum = 1))]
    pub limit: u32,
}

/// A key that reaches one tenant, just issued. Its secret is shown this
/// once; keep it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IssuedKey {
    pub id: uuid::Uuid,
    /// What to send as the bearer key.
    pub key: String,
    pub created: chrono::DateTime<chrono::Utc>,
}

impl std::fmt::Debug for IssuedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IssuedKey")
            .field("id", &self.id)
            .field("key", &"…")
            .field("created", &self.created)
            .finish()
    }
}

/// A key that reaches one tenant, without its secret.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TenantKey {
    pub id: uuid::Uuid,
    pub created: chrono::DateTime<chrono::Utc>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TenantKeys {
    pub keys: Vec<TenantKey>,
}

/// A state a tenant can be put back to: the commit that made it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RestorePoint {
    pub position: u64,
    pub at: chrono::DateTime<chrono::Utc>,
}

/// How far back a tenant may be restored, and the earliest and latest
/// states it may be put back to, if it has any.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RestorePoints {
    pub window_secs: u64,
    pub earliest: Option<RestorePoint>,
    pub latest: Option<RestorePoint>,
}

/// Which state to put a tenant back to.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestoreTo {
    /// As it stood at this moment.
    At(chrono::DateTime<chrono::Utc>),
    /// As the commit at this position left it.
    Position(u64),
    /// As the backup with this id took it.
    Backup(uuid::Uuid),
}

/// A copy of a tenant kept in the backup store.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Backup {
    pub id: uuid::Uuid,
    pub taken_at: chrono::DateTime<chrono::Utc>,
    /// The position of the tenant's last commit then.
    pub position: u64,
    /// How many files it is kept as, each shared with the backups that
    /// need it too.
    pub files: usize,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Backups {
    pub backups: Vec<Backup>,
}

/// A restore done: the state gone back to, and the commit that put it
/// back.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Restored {
    pub from: RestorePoint,
    /// The tenant's incarnation the restore was made in; see [`After`].
    /// Zero from a node too old to say, which a read treats as reached.
    #[serde(default)]
    pub incarnation: u64,
    pub position: u64,
}

impl Restored {
    /// What a read must have caught up to for this restore to be in it.
    pub fn after(&self) -> After {
        After {
            incarnation: self.incarnation,
            position: self.position,
        }
    }
}

/// How far back a tenant may be restored, in seconds; with none, the
/// node's default.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RestoreWindow {
    pub secs: Option<u64>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Tenants {
    pub tenants: Vec<TenantId>,
    /// What to send as `after` for the next page; absent once a page comes
    /// back with fewer tenants than asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<TenantId>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Collections {
    pub collections: Vec<Collection>,
}

/// An object as a client sends it to be stored whole.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Content {
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub properties: Map<String, Json>,
    /// Left out, the collection embeds the object itself if it is set up
    /// to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// Store it only if an object is stored under the id and is at this
    /// version: `conflict` (409) otherwise, with the version it is at,
    /// and `not_found` (404) if none is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_if_version: Option<Version>,
}

/// Holds when the stored object's `property` equals `equals`, where null
/// means the property has no value.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OnlyIf {
    pub property: String,
    pub equals: Json,
}

/// Properties to merge into a stored object: one sent as null is removed.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Changes {
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub properties: Map<String, Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_if: Option<OnlyIf>,
    /// Only if the stored object is at this version; with `only_if` too,
    /// both must hold. `conflict` (409) otherwise, with the version it
    /// is at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_if_version: Option<Version>,
}

/// An object as it is stored.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Stored {
    pub id: ObjectId,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub properties: Map<String, Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// The version of the object's content, vector included whether or
    /// not it is sent along: what to give as `only_if_version` to write
    /// it only if it is still so. Absent from a node of 0.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<Version>,
}

/// One change. Several sent together are applied as one atomic step, each
/// seeing those before it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Change {
    CreateCollection {
        schema: Box<Collection>,
    },
    AddProperty {
        collection: CollectionName,
        property: Property,
    },
    DropCollection {
        collection: CollectionName,
    },
    /// Store the object whole, replacing any stored under its id. With
    /// `only_if_version`, only if one is stored and is at that version.
    Upsert {
        collection: CollectionName,
        id: ObjectId,
        #[serde(default)]
        #[cfg_attr(feature = "openapi", schema(value_type = Object))]
        properties: Map<String, Json>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vector: Option<Vec<f32>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        only_if_version: Option<Version>,
    },
    /// Store the object only if none is stored under its id.
    Insert {
        collection: CollectionName,
        id: ObjectId,
        #[serde(default)]
        #[cfg_attr(feature = "openapi", schema(value_type = Object))]
        properties: Map<String, Json>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vector: Option<Vec<f32>>,
    },
    /// Merge into a stored object: a property sent as null is removed.
    /// Only if the object passes `only_if` and is at `only_if_version`,
    /// where given.
    Patch {
        collection: CollectionName,
        id: ObjectId,
        #[serde(default)]
        #[cfg_attr(feature = "openapi", schema(value_type = Object))]
        properties: Map<String, Json>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vector: Option<Vec<f32>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        only_if: Option<OnlyIf>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        only_if_version: Option<Version>,
    },
    /// Add `by` to a number, counting no value as zero.
    Increment {
        collection: CollectionName,
        id: ObjectId,
        property: String,
        by: f64,
    },
    /// Add `values` to the end of a text array; with `unique`, only those
    /// not already in it.
    Append {
        collection: CollectionName,
        id: ObjectId,
        property: String,
        values: Vec<String>,
        #[serde(default)]
        unique: bool,
    },
    /// Delete the object; deleting one that is not stored changes
    /// nothing. With `only_if_version`, it must be stored, at that
    /// version.
    Delete {
        collection: CollectionName,
        id: ObjectId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        only_if_version: Option<Version>,
    },
    /// Delete the objects passing `filter`, which must have a condition.
    ///
    /// With no `limit`, every one of them, of which there may be at most
    /// 8,192: with more the request is refused as invalid and nothing is
    /// deleted. With a `limit`, of 1 to 8,192, up to so many of them,
    /// whichever are found first; the outcome says how many, and fewer
    /// than the limit means none were left. The deletes of one request
    /// count together, one with no limit for 8,192, and a request whose
    /// deletes may come to more than 8,192 is refused.
    ///
    /// To delete however many pass, send the delete with a limit until it
    /// deletes fewer than the limit. **Each must be a new request, under
    /// a new `x-vellev-request`:** one sent again under the name it had is
    /// answered as it was, and deletes nothing more. What is deleted in
    /// the end is whatever passes until none does, objects written
    /// meanwhile included, not what passed when the first was sent.
    DeleteWhere {
        collection: CollectionName,
        filter: Filter,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        limit: Option<u32>,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Batch {
    pub changes: Vec<Change>,
}

/// What one change did.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Outcome {
    Applied,
    /// The change was valid and altered nothing.
    Unchanged,
    /// The property's value after an increment.
    Value {
        value: Json,
    },
    /// How many objects a delete by filter removed.
    Deleted {
        count: u64,
    },
    /// An outcome this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
}

/// What a write did, and the position from which it can be read.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Written {
    /// The tenant's incarnation the write was made in; see [`After`]. Zero
    /// from a node too old to say, which a read treats as reached.
    #[serde(default)]
    pub incarnation: u64,
    pub position: u64,
    pub outcomes: Vec<Outcome>,
}

impl Written {
    /// What a read must have caught up to for this write to be in it.
    pub fn after(&self) -> After {
        After {
            incarnation: self.incarnation,
            position: self.position,
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Weighted {
    pub property: String,
    /// How much a match in the property counts beside the others.
    pub weight: f32,
}

/// The objects nearest a vector, or nearest what `text` embeds to.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Nearest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub limit: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    /// How many times `limit` candidates are scored exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oversample: Option<u32>,
    /// Whether each result carries its vector.
    #[serde(default)]
    pub vectors: bool,
}

/// The objects whose searchable text best matches the words of `text`.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Keyword {
    pub text: String,
    /// Every searchable property, equally, when empty.
    #[serde(default)]
    pub properties: Vec<Weighted>,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub limit: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    #[serde(default)]
    pub vectors: bool,
}

/// The objects that best match `text` by meaning and by wording together.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Hybrid {
    pub text: String,
    /// What `text` means. Left out, the collection embeds the text if it
    /// is set up to; failing that, the words alone decide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// The weight of meaning against wording, from 0 to 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub fusion: Fusion,
    #[serde(default)]
    pub properties: Vec<Weighted>,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub limit: u32,
    /// How many results each search offers for fusing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oversample: Option<u32>,
    #[serde(default)]
    pub vectors: bool,
}

/// One result of a search, with the scores of the searches that found it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Found {
    #[serde(flatten)]
    pub object: Stored,
    /// The fused score of a hybrid search; larger is better.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// Under the collection's metric; smaller is nearer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distance: Option<f32>,
    /// For cosine, `(1 + cosine) / 2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub certainty: Option<f32>,
    /// The keyword score; larger is better.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keyword: Option<f32>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Results {
    pub results: Vec<Found>,
    /// Set when a hybrid search could not embed its text and so ranked by
    /// the words alone.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub degraded: bool,
    /// Why, as the search would have been refused had it not fallen back:
    /// `unavailable` if the text may be embedded later, and
    /// `model_unavailable` if the node that answered cannot use the model
    /// as it is set up, where another node may answer in full.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degraded_by: Option<Code>,
}

impl Results {
    /// What a search found, ranked as it was asked to be.
    pub fn found(results: Vec<Found>) -> Self {
        Self {
            results,
            degraded: false,
            degraded_by: None,
        }
    }

    /// What a hybrid search found by the words alone, its text not having
    /// been embedded: `refused` is how it would have been refused had it
    /// not fallen back.
    pub fn by_words(results: Vec<Found>, refused: Code) -> Self {
        Self {
            results,
            degraded: true,
            degraded_by: Some(refused),
        }
    }
}

/// One page of a collection in id order.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Page {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    /// The last id of the page before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<ObjectId>,
    #[cfg_attr(feature = "openapi", schema(minimum = 1))]
    pub limit: u32,
    #[serde(default)]
    pub vectors: bool,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Objects {
    pub objects: Vec<Stored>,
    /// What to send as `after` for the next page; absent on the last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<ObjectId>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct Counting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Counted {
    pub count: u64,
}

// What a caller sends is made with what it must have, and given the rest
// one thing at a time, so that a field added later breaks nobody. What a
// node answers is made whole, by the node.

impl TenantPage {
    pub fn new(after: Option<TenantId>, limit: u32) -> Self {
        Self { after, limit }
    }
}

impl RestoreWindow {
    /// A window of `secs` seconds, or with none the node's default.
    pub fn new(secs: Option<u64>) -> Self {
        Self { secs }
    }
}

impl Content {
    /// An object of these properties, for the collection to embed if it
    /// is set up to.
    pub fn new(properties: Map<String, Json>) -> Self {
        Self {
            properties,
            vector: None,
            only_if_version: None,
        }
    }

    #[must_use]
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Stored only if an object is stored under the id at `version`.
    #[must_use]
    pub fn only_if_version(mut self, version: Version) -> Self {
        self.only_if_version = Some(version);
        self
    }
}

impl Changes {
    /// These properties merged into a stored object.
    pub fn new(properties: Map<String, Json>) -> Self {
        Self {
            properties,
            vector: None,
            only_if: None,
            only_if_version: None,
        }
    }

    #[must_use]
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    #[must_use]
    pub fn only_if(mut self, only_if: OnlyIf) -> Self {
        self.only_if = Some(only_if);
        self
    }

    /// Merged only if the stored object is at `version`.
    #[must_use]
    pub fn only_if_version(mut self, version: Version) -> Self {
        self.only_if_version = Some(version);
        self
    }
}

impl OnlyIf {
    /// Holds when the stored object's `property` equals `equals`.
    pub fn new(property: impl Into<String>, equals: Json) -> Self {
        Self {
            property: property.into(),
            equals,
        }
    }
}

impl Weighted {
    /// A match in `property` counting `weight` beside the others.
    pub fn new(property: impl Into<String>, weight: f32) -> Self {
        Self {
            property: property.into(),
            weight,
        }
    }
}

impl Batch {
    pub fn new(changes: Vec<Change>) -> Self {
        Self { changes }
    }
}

impl Nearest {
    /// The `limit` objects nearest `vector`.
    pub fn to_vector(vector: Vec<f32>, limit: u32) -> Self {
        Self {
            vector: Some(vector),
            ..Self::to(limit)
        }
    }

    /// The `limit` objects nearest what `text` embeds to.
    pub fn to_text(text: impl Into<String>, limit: u32) -> Self {
        Self {
            text: Some(text.into()),
            ..Self::to(limit)
        }
    }

    fn to(limit: u32) -> Self {
        Self {
            vector: None,
            text: None,
            limit,
            filter: None,
            oversample: None,
            vectors: false,
        }
    }

    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    #[must_use]
    pub fn oversample(mut self, oversample: u32) -> Self {
        self.oversample = Some(oversample);
        self
    }

    /// Whether each result carries its vector.
    #[must_use]
    pub fn vectors(mut self, vectors: bool) -> Self {
        self.vectors = vectors;
        self
    }
}

impl Keyword {
    /// The `limit` objects best matching the words of `text`.
    pub fn new(text: impl Into<String>, limit: u32) -> Self {
        Self {
            text: text.into(),
            properties: Vec::new(),
            limit,
            filter: None,
            vectors: false,
        }
    }

    #[must_use]
    pub fn properties(mut self, properties: Vec<Weighted>) -> Self {
        self.properties = properties;
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    #[must_use]
    pub fn vectors(mut self, vectors: bool) -> Self {
        self.vectors = vectors;
        self
    }
}

impl Hybrid {
    /// The `limit` objects best matching `text` by meaning and wording.
    pub fn new(text: impl Into<String>, limit: u32) -> Self {
        Self {
            text: text.into(),
            vector: None,
            alpha: None,
            fusion: Fusion::default(),
            properties: Vec::new(),
            limit,
            candidates: None,
            filter: None,
            oversample: None,
            vectors: false,
        }
    }

    #[must_use]
    pub fn vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }

    #[must_use]
    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = Some(alpha);
        self
    }

    #[must_use]
    pub fn fusion(mut self, fusion: Fusion) -> Self {
        self.fusion = fusion;
        self
    }

    #[must_use]
    pub fn properties(mut self, properties: Vec<Weighted>) -> Self {
        self.properties = properties;
        self
    }

    #[must_use]
    pub fn candidates(mut self, candidates: u32) -> Self {
        self.candidates = Some(candidates);
        self
    }

    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    #[must_use]
    pub fn oversample(mut self, oversample: u32) -> Self {
        self.oversample = Some(oversample);
        self
    }

    #[must_use]
    pub fn vectors(mut self, vectors: bool) -> Self {
        self.vectors = vectors;
        self
    }
}

impl Page {
    /// The first `limit` objects in id order.
    pub fn new(limit: u32) -> Self {
        Self {
            filter: None,
            after: None,
            limit,
            vectors: false,
        }
    }

    #[must_use]
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// The page after the one that ended with `after`.
    #[must_use]
    pub fn after(mut self, after: ObjectId) -> Self {
        self.after = Some(after);
        self
    }

    #[must_use]
    pub fn vectors(mut self, vectors: bool) -> Self {
        self.vectors = vectors;
        self
    }
}

impl Counting {
    pub fn new(filter: Option<Filter>) -> Self {
        Self { filter }
    }
}

impl IssuedKey {
    pub fn new(id: uuid::Uuid, key: String, created: chrono::DateTime<chrono::Utc>) -> Self {
        Self { id, key, created }
    }
}

impl TenantKey {
    pub fn new(id: uuid::Uuid, created: chrono::DateTime<chrono::Utc>) -> Self {
        Self { id, created }
    }
}

impl TenantKeys {
    pub fn new(keys: Vec<TenantKey>) -> Self {
        Self { keys }
    }
}

impl RestorePoint {
    pub fn new(position: u64, at: chrono::DateTime<chrono::Utc>) -> Self {
        Self { position, at }
    }
}

impl RestorePoints {
    pub fn new(
        window_secs: u64,
        earliest: Option<RestorePoint>,
        latest: Option<RestorePoint>,
    ) -> Self {
        Self {
            window_secs,
            earliest,
            latest,
        }
    }
}

impl Backup {
    pub fn new(
        id: uuid::Uuid,
        taken_at: chrono::DateTime<chrono::Utc>,
        position: u64,
        files: usize,
    ) -> Self {
        Self {
            id,
            taken_at,
            position,
            files,
        }
    }
}

impl Backups {
    pub fn new(backups: Vec<Backup>) -> Self {
        Self { backups }
    }
}

impl Restored {
    pub fn new(from: RestorePoint, incarnation: u64, position: u64) -> Self {
        Self {
            from,
            incarnation,
            position,
        }
    }
}

impl Change {
    /// Store `content` whole under `id`, replacing any object stored
    /// there, or only if what is stored is at the version `content`
    /// names.
    pub fn upsert(collection: CollectionName, id: ObjectId, content: Content) -> Self {
        Self::Upsert {
            collection,
            id,
            properties: content.properties,
            vector: content.vector,
            only_if_version: content.only_if_version,
        }
    }

    /// Merge `changes` into the object stored under `id`, on the
    /// conditions `changes` names.
    pub fn patch(collection: CollectionName, id: ObjectId, changes: Changes) -> Self {
        Self::Patch {
            collection,
            id,
            properties: changes.properties,
            vector: changes.vector,
            only_if: changes.only_if,
            only_if_version: changes.only_if_version,
        }
    }

    /// Delete the object stored under `id`, if any.
    pub fn delete(collection: CollectionName, id: ObjectId) -> Self {
        Self::Delete {
            collection,
            id,
            only_if_version: None,
        }
    }

    /// Delete the object stored under `id`, which must be at `version`.
    pub fn delete_if_version(collection: CollectionName, id: ObjectId, version: Version) -> Self {
        Self::Delete {
            collection,
            id,
            only_if_version: Some(version),
        }
    }

    /// Delete every object of `collection` that passes `filter`: refused
    /// if more than 8,192 do.
    pub fn delete_where(collection: CollectionName, filter: Filter) -> Self {
        Self::DeleteWhere {
            collection,
            filter,
            limit: None,
        }
    }

    /// Delete up to `limit` of the objects of `collection` that pass
    /// `filter`, 8,192 at the most; to be sent again, each time as a
    /// request of its own, until fewer than `limit` are deleted.
    pub fn delete_up_to(collection: CollectionName, filter: Filter, limit: u32) -> Self {
        Self::DeleteWhere {
            collection,
            filter,
            limit: Some(limit),
        }
    }
}

impl Tenants {
    pub fn new(tenants: Vec<TenantId>, next: Option<TenantId>) -> Self {
        Self { tenants, next }
    }
}

impl Collections {
    pub fn new(collections: Vec<Collection>) -> Self {
        Self { collections }
    }
}

impl Stored {
    pub fn new(
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
        version: Version,
    ) -> Self {
        Self {
            id,
            properties,
            vector,
            version: Some(version),
        }
    }
}

impl Written {
    pub fn new(incarnation: u64, position: u64, outcomes: Vec<Outcome>) -> Self {
        Self {
            incarnation,
            position,
            outcomes,
        }
    }
}

impl Found {
    /// `object` as a search found it, with no score yet said.
    pub fn new(object: Stored) -> Self {
        Self {
            object,
            score: None,
            distance: None,
            certainty: None,
            keyword: None,
        }
    }
}

impl Objects {
    pub fn new(objects: Vec<Stored>, next: Option<ObjectId>) -> Self {
        Self { objects, next }
    }
}

impl Counted {
    pub fn new(count: u64) -> Self {
        Self { count }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn a_point_in_a_tenants_history_is_written_and_read_back_and_a_new_incarnation_wins() {
        let at = |incarnation, position| After {
            incarnation,
            position,
        };
        assert_eq!(at(7, 42).to_string(), "7.42");
        assert_eq!("7.42".parse::<After>(), Ok(at(7, 42)));
        for bad in ["", "7", "7.", ".42", "7.42.1", "-1.2", "a.b", "7 .42"] {
            assert!(bad.parse::<After>().is_err(), "{bad:?}");
        }
        assert_eq!(at(7, 42).later(at(7, 3)), at(7, 42));
        assert_eq!(at(7, 3).later(at(7, 42)), at(7, 42));
        assert_eq!(at(7, 42).later(at(9, 1)), at(9, 1), "positions start again");
        assert_eq!(at(9, 1).later(at(7, 42)), at(9, 1));
        assert_eq!("strong".parse(), Ok(Consistency::Strong));
        assert!("Strong".parse::<Consistency>().is_err());
        assert_eq!(Consistency::default(), Consistency::Session);
    }

    #[test]
    fn a_change_is_named_by_its_op_and_refuses_what_it_does_not_know() {
        let id = "00000000-0000-0000-0000-000000000001";
        let upsert: Change = serde_json::from_value(json!({
            "op": "upsert", "collection": "Notes", "id": id, "properties": { "n": 1 },
        }))
        .unwrap();
        assert!(
            matches!(&upsert, Change::Upsert { vector: None, properties, .. } if properties["n"] == 1)
        );
        assert_eq!(
            serde_json::to_value(&upsert).unwrap(),
            json!({ "op": "upsert", "collection": "Notes", "id": id, "properties": { "n": 1 } })
        );
        let patch: Change = serde_json::from_value(json!({
            "op": "patch", "collection": "Notes", "id": id,
            "only_if": { "property": "n", "equals": null },
        }))
        .unwrap();
        assert!(matches!(
            patch,
            Change::Patch {
                only_if: Some(OnlyIf {
                    equals: Json::Null,
                    ..
                }),
                ..
            }
        ));
        for unknown in [
            json!({ "op": "upsert", "collection": "Notes", "id": id, "vectr": [1.0] }),
            json!({ "op": "upsert", "collection": "Notes", "id": id, "only_if": { "property": "n", "equals": 1 } }),
            json!({ "op": "explode", "collection": "Notes" }),
            json!({ "op": "delete", "collection": "Notes", "id": "not-an-id" }),
            json!({ "op": "delete", "collection": "no spaces", "id": id }),
        ] {
            assert!(
                serde_json::from_value::<Change>(unknown.clone()).is_err(),
                "{unknown}"
            );
        }
    }

    #[test]
    fn a_search_must_say_how_many_and_a_result_carries_only_the_scores_it_has() {
        assert!(serde_json::from_value::<Nearest>(json!({ "vector": [1.0] })).is_err());
        assert!(serde_json::from_value::<Keyword>(json!({ "text": "fox" })).is_err());
        assert!(serde_json::from_value::<Page>(json!({})).is_err());
        let hybrid: Hybrid = serde_json::from_value(json!({ "text": "fox", "limit": 5 })).unwrap();
        assert_eq!(
            (hybrid.alpha, hybrid.fusion, hybrid.vectors),
            (None, Fusion::RelativeScore, false)
        );

        let found = Found {
            object: Stored {
                id: ObjectId(Default::default()),
                properties: Map::new(),
                vector: None,
                version: None,
            },
            score: None,
            distance: Some(0.5),
            certainty: Some(0.75),
            keyword: None,
        };
        assert_eq!(
            serde_json::to_value(&found).unwrap(),
            json!({
                "id": "00000000-0000-0000-0000-000000000000",
                "properties": {},
                "distance": 0.5,
                "certainty": 0.75,
            })
        );
        assert_eq!(
            serde_json::to_value(Failure::new(Code::NotFound, "gone")).unwrap(),
            json!({ "error": { "code": "not_found", "message": "gone" } })
        );
    }

    #[test]
    fn a_problem_carries_what_it_can_say_and_reads_whatever_a_node_of_any_release_sends() {
        let refused = Failure {
            error: Problem {
                limit: Some(Limit {
                    name: Bounded::Results,
                    max: 1_000,
                }),
                ..Problem::new(Code::Invalid, "limit must be from 1 to 1000, not 5000")
            },
        };
        let written = json!({ "error": {
            "code": "invalid",
            "message": "limit must be from 1 to 1000, not 5000",
            "limit": { "name": "results", "max": 1000 },
        } });
        assert_eq!(serde_json::to_value(&refused).unwrap(), written);
        assert_eq!(serde_json::from_value::<Failure>(written).unwrap(), refused);
        let modelless = Failure {
            error: Problem {
                reason: Some(ModelFault::Setup.into()),
                ..Problem::new(Code::ModelUnavailable, "no")
            },
        };
        assert_eq!(
            serde_json::to_value(&modelless).unwrap(),
            json!({ "error": { "code": "model_unavailable", "message": "no", "reason": "setup" } })
        );
        for (fault, written) in [
            (ModelFault::Build, "build"),
            (ModelFault::Setup, "setup"),
            (ModelFault::Provider, "provider"),
        ] {
            assert_eq!(
                serde_json::to_value(Reason::from(fault)).unwrap(),
                json!(written)
            );
        }
        let degraded = Results::by_words(Vec::new(), Code::ModelUnavailable);
        assert!(degraded.degraded);
        let written =
            json!({ "results": [], "degraded": true, "degraded_by": "model_unavailable" });
        assert_eq!(serde_json::to_value(&degraded).unwrap(), written);
        assert_eq!(
            serde_json::from_value::<Results>(written).unwrap(),
            degraded
        );
        let older: Results =
            serde_json::from_value(json!({ "results": [], "degraded": true })).unwrap();
        assert_eq!((older.degraded, older.degraded_by), (true, None));
        let full = Results::found(Vec::new());
        assert_eq!(
            serde_json::to_value(&full).unwrap(),
            json!({ "results": [] })
        );
        let missing = Failure {
            error: Problem {
                missing: Some(Missing::RestorePoint),
                ..Problem::new(Code::NotFound, "gone")
            },
        };
        assert_eq!(
            serde_json::to_value(&missing).unwrap(),
            json!({ "error": { "code": "not_found", "message": "gone", "missing": "restore_point" } })
        );
        assert_eq!(
            serde_json::to_value(Code::ModelUnavailable).unwrap(),
            json!("model_unavailable")
        );

        let older: Failure =
            serde_json::from_value(json!({ "error": { "code": "not_found", "message": "gone" } }))
                .unwrap();
        assert_eq!(older, Failure::new(Code::NotFound, "gone"));
        let newer: Failure = serde_json::from_value(json!({ "error": {
            "code": "something_new",
            "message": "?",
            "limit": { "name": "something_new", "max": 3 },
            "missing": "something_new",
            "reason": "something_new",
            "hint": "a field from later still",
        } }))
        .unwrap();
        assert_eq!(
            newer.error,
            Problem {
                limit: Some(Limit {
                    name: Bounded::Unknown,
                    max: 3
                }),
                missing: Some(Missing::Unknown),
                reason: Some(Reason::Unknown),
                ..Problem::new(Code::Unknown, "?")
            }
        );
    }

    #[test]
    fn a_request_is_made_with_what_it_must_have_and_given_the_rest() {
        let filter = || Filter::Eq {
            property: "n".into(),
            value: json!(1),
        };
        let id = ObjectId(Default::default());
        let weighted = || vec![Weighted::new("title", 2.0)];
        assert_eq!(
            serde_json::to_value(weighted()).unwrap(),
            json!([{ "property": "title", "weight": 2.0 }])
        );
        let cases = [
            (
                serde_json::to_value(Nearest::to_vector(vec![1.0, 0.0], 5)).unwrap(),
                json!({ "vector": [1.0, 0.0], "limit": 5, "vectors": false }),
            ),
            (
                serde_json::to_value(
                    Nearest::to_text("fox", 5)
                        .filter(filter())
                        .oversample(3)
                        .vectors(true),
                )
                .unwrap(),
                json!({ "text": "fox", "limit": 5, "filter": filter(), "oversample": 3, "vectors": true }),
            ),
            (
                serde_json::to_value(Keyword::new("fox", 5)).unwrap(),
                json!({ "text": "fox", "properties": [], "limit": 5, "vectors": false }),
            ),
            (
                serde_json::to_value(
                    Keyword::new("fox", 5)
                        .properties(weighted())
                        .filter(filter())
                        .vectors(true),
                )
                .unwrap(),
                json!({ "text": "fox", "properties": weighted(), "limit": 5, "filter": filter(), "vectors": true }),
            ),
            (
                serde_json::to_value(Hybrid::new("fox", 5)).unwrap(),
                json!({ "text": "fox", "fusion": Fusion::default(), "properties": [], "limit": 5, "vectors": false }),
            ),
            (
                serde_json::to_value(
                    Hybrid::new("fox", 5)
                        .vector(vec![1.0])
                        .alpha(0.5)
                        .fusion(Fusion::Ranked)
                        .properties(weighted())
                        .candidates(7)
                        .filter(filter())
                        .oversample(3)
                        .vectors(true),
                )
                .unwrap(),
                json!({
                    "text": "fox", "vector": [1.0], "alpha": 0.5, "fusion": Fusion::Ranked,
                    "properties": weighted(), "limit": 5, "candidates": 7, "filter": filter(),
                    "oversample": 3, "vectors": true,
                }),
            ),
            (
                serde_json::to_value(Page::new(5)).unwrap(),
                json!({ "limit": 5, "vectors": false }),
            ),
            (
                serde_json::to_value(Page::new(5).filter(filter()).after(id).vectors(true))
                    .unwrap(),
                json!({ "filter": filter(), "after": id, "limit": 5, "vectors": true }),
            ),
            (
                serde_json::to_value(Content::new(Map::new())).unwrap(),
                json!({ "properties": {} }),
            ),
            (
                serde_json::to_value(Content::new(Map::new()).vector(vec![1.0])).unwrap(),
                json!({ "properties": {}, "vector": [1.0] }),
            ),
            (
                serde_json::to_value(Changes::new(Map::new())).unwrap(),
                json!({ "properties": {} }),
            ),
            (
                serde_json::to_value(
                    Changes::new(Map::new())
                        .vector(vec![1.0])
                        .only_if(OnlyIf::new("n", json!(1))),
                )
                .unwrap(),
                json!({ "properties": {}, "vector": [1.0], "only_if": { "property": "n", "equals": 1 } }),
            ),
            (
                serde_json::to_value(Batch::new(Vec::new())).unwrap(),
                json!({ "changes": [] }),
            ),
            (
                serde_json::to_value(RestoreWindow::new(Some(60))).unwrap(),
                json!({ "secs": 60 }),
            ),
            (
                serde_json::to_value(Counting::new(Some(filter()))).unwrap(),
                json!({ "filter": filter() }),
            ),
            (
                serde_json::to_value(TenantPage::new(None, 5)).unwrap(),
                json!({ "after": null, "limit": 5 }),
            ),
        ];
        for (made, expected) in cases {
            assert_eq!(made, expected);
        }
    }

    #[test]
    fn a_restore_answered_by_a_node_that_says_no_incarnation_reads_as_none() {
        let at = chrono::DateTime::<chrono::Utc>::from_timestamp(1_791_300_000, 0).unwrap();
        let older = json!({ "from": { "position": 4, "at": at }, "position": 5 });
        let restored: Restored = serde_json::from_value(older).unwrap();
        assert_eq!(restored, Restored::new(RestorePoint::new(4, at), 0, 5));
    }

    #[test]
    fn an_answer_is_made_whole() {
        let id = uuid::Uuid::from_u128(7);
        let at = chrono::DateTime::<chrono::Utc>::from_timestamp(1_791_300_000, 0).unwrap();
        let object = ObjectId(id);
        let tenant = TenantId::new("acme").unwrap();
        let version: Version = "0123456789abcdef0123456789abcdef".parse().unwrap();
        let stored = || Stored::new(object, Map::new(), Some(vec![1.0]), version);
        let point = || RestorePoint::new(4, at);
        let cases = [
            (
                serde_json::to_value(IssuedKey::new(id, "vlv_k".into(), at)).unwrap(),
                json!({ "id": id, "key": "vlv_k", "created": at }),
            ),
            (
                serde_json::to_value(TenantKeys::new(vec![TenantKey::new(id, at)])).unwrap(),
                json!({ "keys": [{ "id": id, "created": at }] }),
            ),
            (
                serde_json::to_value(RestorePoints::new(60, Some(point()), None)).unwrap(),
                json!({ "window_secs": 60, "earliest": { "position": 4, "at": at }, "latest": null }),
            ),
            (
                serde_json::to_value(Backups::new(vec![Backup::new(id, at, 4, 9)])).unwrap(),
                json!({ "backups": [{ "id": id, "taken_at": at, "position": 4, "files": 9 }] }),
            ),
            (
                serde_json::to_value(Restored::new(point(), 2, 5)).unwrap(),
                json!({ "from": { "position": 4, "at": at }, "incarnation": 2, "position": 5 }),
            ),
            (
                serde_json::to_value(Tenants::new(vec![tenant.clone()], Some(tenant))).unwrap(),
                json!({ "tenants": ["acme"], "next": "acme" }),
            ),
            (
                serde_json::to_value(Collections::new(Vec::new())).unwrap(),
                json!({ "collections": [] }),
            ),
            (
                serde_json::to_value(Written::new(3, 4, vec![Outcome::Applied])).unwrap(),
                json!({ "incarnation": 3, "position": 4, "outcomes": [{ "outcome": "applied" }] }),
            ),
            (
                serde_json::to_value(Found::new(stored())).unwrap(),
                json!({ "id": object, "properties": {}, "vector": [1.0], "version": version }),
            ),
            (
                serde_json::to_value(Objects::new(vec![stored()], Some(object))).unwrap(),
                json!({ "objects": [{ "id": object, "properties": {}, "vector": [1.0], "version": version }], "next": object }),
            ),
            (
                serde_json::to_value(Counted::new(9)).unwrap(),
                json!({ "count": 9 }),
            ),
        ];
        for (made, expected) in cases {
            assert_eq!(made, expected);
        }
    }

    #[test]
    fn a_delete_by_filter_is_built_with_the_limit_it_is_given_or_none() {
        let notes = CollectionName::new("notes").unwrap();
        let old = || Filter::Lt {
            property: "n".into(),
            value: json!(5),
        };
        let filter = json!({ "lt": { "property": "n", "value": 5 } });
        let unlimited = Change::delete_where(notes.clone(), old());
        assert_eq!(
            unlimited,
            Change::DeleteWhere {
                collection: notes.clone(),
                filter: old(),
                limit: None,
            }
        );
        assert_eq!(
            serde_json::to_value(&unlimited).unwrap(),
            json!({ "op": "delete_where", "collection": "notes", "filter": filter })
        );
        for limit in [1, 500, u32::MAX] {
            let limited = Change::delete_up_to(notes.clone(), old(), limit);
            assert_eq!(
                limited,
                Change::DeleteWhere {
                    collection: notes.clone(),
                    filter: old(),
                    limit: Some(limit),
                }
            );
            assert_eq!(
                serde_json::to_value(&limited).unwrap(),
                json!({ "op": "delete_where", "collection": "notes", "filter": filter,
                    "limit": limit })
            );
        }
    }

    #[test]
    fn a_change_built_from_a_body_carries_every_condition_the_body_names() {
        let notes = CollectionName::new("notes").unwrap();
        let id = ObjectId(uuid::Uuid::from_u128(7));
        let at: Version = "0123456789abcdef0123456789abcdef".parse().unwrap();
        let properties = Map::from_iter([("n".to_string(), json!(1))]);
        let content = Content::new(properties.clone())
            .vector(vec![1.0])
            .only_if_version(at);
        assert_eq!(
            Change::upsert(notes.clone(), id, content),
            Change::Upsert {
                collection: notes.clone(),
                id,
                properties: properties.clone(),
                vector: Some(vec![1.0]),
                only_if_version: Some(at),
            }
        );
        assert_eq!(
            Change::upsert(notes.clone(), id, Content::new(Map::new())),
            Change::Upsert {
                collection: notes.clone(),
                id,
                properties: Map::new(),
                vector: None,
                only_if_version: None,
            }
        );
        let changes = Changes::new(properties.clone())
            .vector(vec![2.0])
            .only_if(OnlyIf::new("n", json!(0)))
            .only_if_version(at);
        assert_eq!(
            Change::patch(notes.clone(), id, changes),
            Change::Patch {
                collection: notes.clone(),
                id,
                properties,
                vector: Some(vec![2.0]),
                only_if: Some(OnlyIf::new("n", json!(0))),
                only_if_version: Some(at),
            }
        );
        assert_eq!(
            Change::delete(notes.clone(), id),
            Change::Delete {
                collection: notes.clone(),
                id,
                only_if_version: None,
            }
        );
        assert_eq!(
            Change::delete_if_version(notes.clone(), id, at),
            Change::Delete {
                collection: notes.clone(),
                id,
                only_if_version: Some(at),
            }
        );
        assert_eq!(
            serde_json::to_value(Change::delete_if_version(notes, id, at)).unwrap(),
            json!({ "op": "delete", "collection": "notes", "id": id, "only_if_version": at })
        );
    }

    #[test]
    fn each_limit_is_told_under_its_own_name_and_one_not_said_is_not_known() {
        let limits = Limits {
            results: 1,
            candidates: 2,
            oversample: 3,
            body_bytes: 4,
            restore_window_secs: 5,
            deleted: 6,
            reads_at_once: 7,
        };
        let told = json!({
            "results": 1,
            "candidates": 2,
            "oversample": 3,
            "body_bytes": 4,
            "restore_window_secs": 5,
            "deleted": 6,
            "reads_at_once": 7,
        });
        assert_eq!(serde_json::to_value(limits).unwrap(), told);
        for (name, max) in [
            (Bounded::Results, Some(1)),
            (Bounded::Candidates, Some(2)),
            (Bounded::Oversample, Some(3)),
            (Bounded::BodyBytes, Some(4)),
            (Bounded::RestoreWindowSecs, Some(5)),
            (Bounded::Deleted, Some(6)),
            (Bounded::ReadsAtOnce, Some(7)),
            (Bounded::Unknown, None),
        ] {
            assert_eq!(limits.max(name), max, "{name:?}");
            let written = serde_json::to_value(name).unwrap();
            assert_eq!(
                written.as_str().and_then(|field| told.get(field)?.as_u64()),
                max,
                "a limit's name is its field: {name:?}"
            );
        }
        // A node of another release may say fewer, or more.
        let fewer: Limits = serde_json::from_value(json!({ "results": 7, "later": 9 })).unwrap();
        assert_eq!(fewer.max(Bounded::Results), Some(7));
        assert_eq!(fewer.max(Bounded::Oversample), None);
    }

    #[test]
    fn every_code_has_its_status_and_a_model_that_cannot_be_used_is_not_one_retried_blindly() {
        let statuses = [
            (Code::Invalid, 400),
            (Code::Unauthorized, 401),
            (Code::Forbidden, 403),
            (Code::NotFound, 404),
            (Code::Conflict, 409),
            (Code::Limited, 429),
            (Code::Unavailable, 503),
            (Code::Uncertain, 503),
            (Code::ModelUnavailable, 501),
            (Code::Internal, 500),
            (Code::Unknown, 500),
        ];
        for (code, status) in statuses {
            assert_eq!(code.status(), status, "{code:?}");
        }
        // As a number, each code is what it was before any was added.
        let numbered = [
            Code::Invalid,
            Code::Unauthorized,
            Code::Forbidden,
            Code::NotFound,
            Code::Conflict,
            Code::Limited,
            Code::Unavailable,
            Code::Uncertain,
            Code::Internal,
            Code::Unknown,
            Code::ModelUnavailable,
        ];
        for (number, code) in numbered.into_iter().enumerate() {
            assert_eq!(code as usize, number, "{code:?}");
        }
    }
}
