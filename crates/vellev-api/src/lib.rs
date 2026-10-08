//! What goes over the wire between a client and a node: the bodies of the
//! HTTP API's requests and responses.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use vellev_core::{Collection, CollectionName, ObjectId, Property, TenantId};
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
    /// The request is malformed or asks for something that cannot be.
    Invalid,
    Unauthorized,
    /// The key is valid, and does not reach what the request asks for.
    Forbidden,
    NotFound,
    /// The request disagrees with what is stored.
    Conflict,
    /// The tenant is asking more of the node than it may; the request had
    /// no effect and may be sent again after the `Retry-After` it gives.
    Limited,
    /// The node cannot serve this now, and the request has had no effect;
    /// the same request may succeed later or on another node.
    Unavailable,
    /// A write failed at a point where nobody can say whether it took
    /// effect. Read before deciding to send it again.
    Uncertain,
    /// This node cannot use the collection's embedding model until its
    /// setup is put right: the build cannot run the model, the node may
    /// not write where the model is kept, whatever serves it says it has
    /// no such model or will not serve this account, or its vectors are
    /// not the collection's size. The request had no effect. Another node
    /// may be able to do it now, and this one once it is put right, so
    /// the request is to be kept and sent again, never dropped; one that
    /// brings its own vector needs no model.
    ModelUnavailable,
    Internal,
    /// A code this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
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
}

impl Problem {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            limit: None,
            missing: None,
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
    /// Something this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
}

/// The most of something a request may ask for; at least one is always
/// allowed.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limit {
    pub name: Bounded,
    pub max: u64,
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
            Bounded::Unknown => 0,
        };
        (said > 0).then_some(said)
    }
}

/// The body of every response that is not a success.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    pub error: Problem,
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
pub struct TenantKey {
    pub id: uuid::Uuid,
    pub created: chrono::DateTime<chrono::Utc>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TenantKeys {
    pub keys: Vec<TenantKey>,
}

/// A state a tenant can be put back to: the commit that made it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestorePoint {
    pub position: u64,
    pub at: chrono::DateTime<chrono::Utc>,
}

/// How far back a tenant may be restored, and the earliest and latest
/// states it may be put back to, if it has any.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
pub struct Backups {
    pub backups: Vec<Backup>,
}

/// A restore done: the state gone back to, and the position of the commit
/// that put it back.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Restored {
    pub from: RestorePoint,
    pub position: u64,
}

/// How far back a tenant may be restored, in seconds; with none, the
/// node's default.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreWindow {
    pub secs: Option<u64>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tenants {
    pub tenants: Vec<TenantId>,
    /// What to send as `after` for the next page; absent once a page comes
    /// back with fewer tenants than asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<TenantId>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Collections {
    pub collections: Vec<Collection>,
}

/// An object as a client sends it to be stored whole.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Content {
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub properties: Map<String, Json>,
    /// Left out, the collection embeds the object itself if it is set up
    /// to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
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
pub struct Changes {
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub properties: Map<String, Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub only_if: Option<OnlyIf>,
}

/// An object as it is stored.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stored {
    pub id: ObjectId,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub properties: Map<String, Json>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
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
    /// Store the object whole, replacing any stored under its id.
    Upsert {
        collection: CollectionName,
        id: ObjectId,
        #[serde(default)]
        #[cfg_attr(feature = "openapi", schema(value_type = Object))]
        properties: Map<String, Json>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vector: Option<Vec<f32>>,
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
    Delete {
        collection: CollectionName,
        id: ObjectId,
    },
    /// Delete every object passing `filter`, which must have a condition.
    DeleteWhere {
        collection: CollectionName,
        filter: Filter,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub changes: Vec<Change>,
}

/// What one change did.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
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
pub struct Results {
    pub results: Vec<Found>,
    /// Set when a hybrid search could not embed its text and so ranked by
    /// the words alone.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub degraded: bool,
}

/// One page of a collection in id order.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
pub struct Objects {
    pub objects: Vec<Stored>,
    /// What to send as `after` for the next page; absent on the last.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<ObjectId>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counting {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counted {
    pub count: u64,
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
                ..Problem::new(Code::Unknown, "?")
            }
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
        };
        let told = json!({
            "results": 1,
            "candidates": 2,
            "oversample": 3,
            "body_bytes": 4,
            "restore_window_secs": 5,
        });
        assert_eq!(serde_json::to_value(limits).unwrap(), told);
        for (name, max) in [
            (Bounded::Results, Some(1)),
            (Bounded::Candidates, Some(2)),
            (Bounded::Oversample, Some(3)),
            (Bounded::BodyBytes, Some(4)),
            (Bounded::RestoreWindowSecs, Some(5)),
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
    }
}
