//! What goes over the wire between a client and a node: the bodies of the
//! HTTP API's requests and responses.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};
use vellev_core::{Collection, CollectionName, ObjectId, Property, TenantId};
use vellev_index::filter::Filter;
use vellev_index::fusion::Fusion;

/// The path every versioned route starts with.
pub const ROOT: &str = "/v1";

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Code {
    /// The request is malformed or asks for something that cannot be.
    Invalid,
    Unauthorized,
    NotFound,
    /// The request disagrees with what is stored.
    Conflict,
    /// The node cannot serve this now; the same request may succeed later
    /// or on another node.
    Unavailable,
    Internal,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TenantPage {
    /// The last tenant of the page before.
    pub after: Option<TenantId>,
    pub limit: usize,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tenants {
    pub tenants: Vec<TenantId>,
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
}

/// What a write did, and the position from which it can be read.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Written {
    pub position: u64,
    pub outcomes: Vec<Outcome>,
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
    pub limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    /// How many times `limit` candidates are scored exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oversample: Option<usize>,
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
    pub limit: usize,
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
    pub limit: usize,
    /// How many results each search offers for fusing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidates: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oversample: Option<usize>,
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
    pub limit: usize,
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
