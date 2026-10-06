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
pub enum Code {
    /// The request is malformed or asks for something that cannot be.
    Invalid,
    Unauthorized,
    /// The key is valid, and does not reach what the request asks for.
    Forbidden,
    NotFound,
    /// The request disagrees with what is stored.
    Conflict,
    /// The node cannot serve this now, and the request has had no effect;
    /// the same request may succeed later or on another node.
    Unavailable,
    /// A write failed at a point where nobody can say whether it took
    /// effect. Read before deciding to send it again.
    Uncertain,
    Internal,
    /// A code this client does not know, from a newer node.
    #[serde(other)]
    Unknown,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Problem {
    pub code: Code,
    pub message: String,
}

/// The body of every response that is not a success.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    pub error: Problem,
}

impl Failure {
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            error: Problem { code, message },
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
            serde_json::to_value(Failure {
                error: Problem {
                    code: Code::NotFound,
                    message: "gone".into()
                }
            })
            .unwrap(),
            json!({ "error": { "code": "not_found", "message": "gone" } })
        );
    }
}
