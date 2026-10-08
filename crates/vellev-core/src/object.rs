use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::{Collection, DataType, Embedding, ObjectId};

/// Why an object, or a vector searched by, is refused. More reasons may come,
/// so a match on one outside this crate needs an arm for the rest:
///
/// ```compile_fail
/// use vellev_core::ObjectError;
///
/// fn reason(error: &ObjectError) -> &'static str {
///     match error {
///         ObjectError::UnknownProperty(_) => "unknown property",
///         ObjectError::Type { .. } => "wrong type",
///         ObjectError::VectorLength { .. } => "wrong length",
///         ObjectError::VectorNotFinite => "not finite",
///         ObjectError::VectorTooLong => "too long",
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ObjectError {
    #[error("property {0:?} is not declared on the collection")]
    UnknownProperty(String),
    #[error("property {property:?} must be {expected}")]
    Type {
        property: String,
        expected: DataType,
    },
    #[error("vector has {got} dimensions, and the collection declares {expected}")]
    VectorLength { expected: u32, got: usize },
    #[error("vector contains a value that is not a finite number")]
    VectorNotFinite,
    #[error("vector is too long: its length must be under 2^62, about 4.6e18")]
    VectorTooLong,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Text(String),
    TextArray(Vec<String>),
    Int(i64),
    Number(f64),
    Bool(bool),
    Date(DateTime<Utc>),
}

impl Value {
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Text(_) => DataType::Text,
            Value::TextArray(_) => DataType::TextArray,
            Value::Int(_) => DataType::Int,
            Value::Number(_) => DataType::Number,
            Value::Bool(_) => DataType::Bool,
            Value::Date(_) => DataType::Date,
        }
    }

    /// Read `json` as `data_type`, or `None` when it is some other type. The
    /// schema decides the reading: a date and a text are both JSON strings.
    fn from_json(data_type: DataType, json: &Json) -> Option<Value> {
        Some(match data_type {
            DataType::Text => Value::Text(json.as_str()?.to_string()),
            DataType::TextArray => Value::TextArray(
                json.as_array()?
                    .iter()
                    .map(|item| item.as_str().map(String::from))
                    .collect::<Option<_>>()?,
            ),
            DataType::Int => Value::Int(json.as_i64()?),
            DataType::Number => Value::Number(json.as_f64().filter(|n| n.is_finite())?),
            DataType::Bool => Value::Bool(json.as_bool()?),
            DataType::Date => Value::Date(
                DateTime::parse_from_rfc3339(json.as_str()?)
                    .ok()?
                    .with_timezone(&Utc),
            ),
        })
    }

    /// Read `json` as the type `collection` declares for `property`; JSON
    /// null is no value.
    pub fn typed(
        collection: &Collection,
        property: &str,
        json: &Json,
    ) -> Result<Option<Value>, ObjectError> {
        let expected = collection
            .property(property)
            .ok_or_else(|| ObjectError::UnknownProperty(property.to_string()))?
            .data_type;
        if json.is_null() {
            return Ok(None);
        }
        Value::from_json(expected, json)
            .map(Some)
            .ok_or_else(|| ObjectError::Type {
                property: property.to_string(),
                expected,
            })
    }

    pub fn to_json(&self) -> Json {
        match self {
            Value::Text(s) => Json::from(s.as_str()),
            Value::TextArray(items) => Json::from(items.clone()),
            Value::Int(n) => Json::from(*n),
            Value::Number(n) => Json::from(*n),
            Value::Bool(b) => Json::from(*b),
            Value::Date(d) => Json::from(d.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)),
        }
    }
}

/// A vector's squared length must be under this, 2^124, so that its length
/// is under 2^62. Then the squared distance between two vectors, at most
/// `(2^62 + 2^62)^2 = 2^126`, and their inner product, at most 2^124, stay
/// within an `f32`, which holds up to nearly 2^128. What the vector indexes
/// estimate from such vectors is bounded as their own documentation says:
/// never NaN, and infinite only where it ranks a row last.
pub const SQUARED_LENGTH_UNDER: f32 = (1u128 << 124) as f32;

impl Embedding {
    /// `vector` if it is one this embedding's collection can hold.
    pub fn checked(&self, vector: Vec<f32>) -> Result<Vec<f32>, ObjectError> {
        self.check(&vector).map(|()| vector)
    }

    /// Whether `vector` is one this embedding's collection can hold, or
    /// search by.
    pub fn check(&self, vector: &[f32]) -> Result<(), ObjectError> {
        let expected = self.dimensions;
        if vector.len() != expected as usize {
            return Err(ObjectError::VectorLength {
                expected,
                got: vector.len(),
            });
        }
        if !vector.iter().all(|v| v.is_finite()) {
            return Err(ObjectError::VectorNotFinite);
        }
        if vector.iter().map(|v| v * v).sum::<f32>() >= SQUARED_LENGTH_UNDER {
            return Err(ObjectError::VectorTooLong);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Object {
    pub id: ObjectId,
    /// Only the properties that have a value. A property that is absent, or
    /// sent as JSON null, is not stored.
    pub properties: BTreeMap<String, Value>,
    #[serde(with = "vector_bytes")]
    pub vector: Option<Vec<f32>>,
}

/// Vectors are stored as little-endian bytes: a quarter of the size of a
/// self-describing array of floats, and exact.
mod vector_bytes {
    use serde::de::Error;
    use serde::{Deserialize, Deserializer, Serializer};
    use serde_bytes::ByteBuf;

    pub fn serialize<S: Serializer>(vector: &Option<Vec<f32>>, s: S) -> Result<S::Ok, S::Error> {
        match vector {
            Some(vector) => s.serialize_some(&ByteBuf::from(
                vector
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<f32>>, D::Error> {
        Option::<ByteBuf>::deserialize(d)?
            .map(|bytes| {
                let chunks = bytes.chunks_exact(4);
                if !chunks.remainder().is_empty() {
                    return Err(D::Error::custom("vector bytes are not a multiple of four"));
                }
                Ok(chunks
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            })
            .transpose()
    }
}

impl Object {
    /// Check `properties` and `vector` against the collection and build the
    /// typed object. Nothing is coerced: a number sent for a text property is
    /// refused, not stringified.
    pub fn from_json(
        collection: &Collection,
        id: ObjectId,
        properties: &serde_json::Map<String, Json>,
        vector: Option<Vec<f32>>,
    ) -> Result<Object, ObjectError> {
        Object {
            id,
            properties: BTreeMap::new(),
            vector: None,
        }
        .patched(collection, properties, vector)
    }

    /// This object with `properties` merged in: a property sent as JSON null
    /// is removed, any other replaces the stored value. `vector` replaces the
    /// stored vector when given.
    pub fn patched(
        &self,
        collection: &Collection,
        properties: &serde_json::Map<String, Json>,
        vector: Option<Vec<f32>>,
    ) -> Result<Object, ObjectError> {
        let merged =
            properties
                .iter()
                .try_fold(self.properties.clone(), |mut merged, (name, json)| {
                    match Value::typed(collection, name, json)? {
                        Some(value) => merged.insert(name.clone(), value),
                        None => merged.remove(name),
                    };
                    Ok(merged)
                })?;
        let vector = match vector {
            Some(vector) => Some(collection.embedding.checked(vector)?),
            None => self.vector.clone(),
        };
        Ok(Object {
            id: self.id,
            properties: merged,
            vector,
        })
    }

    /// The text of this object that a collection embedding as `embedding`
    /// says turns into its vector: the listed properties' values, in
    /// order, a line each.
    pub fn embedded_text(&self, embedding: &Embedding) -> String {
        embedding
            .properties
            .iter()
            .filter_map(|name| self.properties.get(name))
            .flat_map(|value| match value {
                Value::Text(text) => std::slice::from_ref(text),
                Value::TextArray(texts) => texts.as_slice(),
                _ => &[],
            })
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn properties_json(&self) -> serde_json::Map<String, Json> {
        self.properties
            .iter()
            .map(|(name, value)| (name.clone(), value.to_json()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::schema::tests::workforce_memory;

    fn parse(properties: Json, vector: Option<Vec<f32>>) -> Result<Object, ObjectError> {
        Object::from_json(
            &workforce_memory(),
            ObjectId::generate(),
            properties.as_object().unwrap(),
            vector,
        )
    }

    #[test]
    fn a_workforce_memory_row_is_typed_by_the_schema() {
        let row = json!({
            "content": "Deploys go out on Tuesdays",
            "memoryId": "m-1",
            "importance": 0.8,
            "tags": ["date:2026-07-05", "lesson"],
            "deleted": false,
            "accessCount": 3,
            "createdAt": "2026-07-05T09:30:00+01:00",
            "identityId": null
        });
        let object = parse(row, Some(vec![0.1; 768])).unwrap();
        assert_eq!(
            object.properties["content"],
            Value::Text("Deploys go out on Tuesdays".into())
        );
        assert_eq!(object.properties["importance"], Value::Number(0.8));
        assert_eq!(object.properties["accessCount"], Value::Int(3));
        assert_eq!(
            object.properties["tags"],
            Value::TextArray(vec!["date:2026-07-05".into(), "lesson".into()])
        );
        // Dates are normalised to UTC; a null property is simply absent.
        assert_eq!(
            object.properties_json()["createdAt"],
            json!("2026-07-05T08:30:00Z")
        );
        assert!(!object.properties.contains_key("identityId"));
    }

    #[test]
    fn an_integer_is_accepted_where_a_number_is_declared() {
        let object = parse(json!({ "importance": 1 }), None).unwrap();
        assert_eq!(object.properties["importance"], Value::Number(1.0));
    }

    #[test]
    fn wrong_types_are_refused_not_coerced() {
        let cases = [
            (json!({ "content": 42 }), "content", DataType::Text),
            (json!({ "tags": "lesson" }), "tags", DataType::TextArray),
            (json!({ "tags": ["ok", 1] }), "tags", DataType::TextArray),
            (json!({ "accessCount": 1.5 }), "accessCount", DataType::Int),
            (json!({ "accessCount": "3" }), "accessCount", DataType::Int),
            (
                json!({ "importance": "high" }),
                "importance",
                DataType::Number,
            ),
            (json!({ "deleted": "false" }), "deleted", DataType::Bool),
            (
                json!({ "createdAt": "yesterday" }),
                "createdAt",
                DataType::Date,
            ),
            (
                json!({ "createdAt": 1751700000 }),
                "createdAt",
                DataType::Date,
            ),
        ];
        for (row, property, expected) in cases {
            assert_eq!(
                parse(row.clone(), None),
                Err(ObjectError::Type {
                    property: property.into(),
                    expected
                }),
                "{row}"
            );
        }
    }

    #[test]
    fn an_undeclared_property_is_refused() {
        assert_eq!(
            parse(json!({ "colour": "red" }), None),
            Err(ObjectError::UnknownProperty("colour".into()))
        );
    }

    #[test]
    fn a_vector_must_match_the_declared_dimensions_and_be_finite() {
        assert_eq!(
            parse(json!({}), Some(vec![0.0; 1024])),
            Err(ObjectError::VectorLength {
                expected: 768,
                got: 1024
            })
        );
        let mut vector = vec![0.5; 768];
        vector[10] = f32::NAN;
        assert_eq!(
            parse(json!({}), Some(vector)),
            Err(ObjectError::VectorNotFinite)
        );
    }

    #[test]
    fn a_vector_must_be_short_enough_for_every_distance_to_be_finite() {
        let with = |values: &[f32]| {
            let mut vector = vec![0.0; 768];
            vector[..values.len()].copy_from_slice(values);
            parse(json!({}), Some(vector)).map(|object| object.vector.is_some())
        };
        assert_eq!(with(&[4e18]), Ok(true));
        assert_eq!(with(&[3e18, 3e18]), Ok(true));
        // Its squared length, 1.7e38, is finite; its squared distance from
        // its opposite, 6.8e38, is not.
        assert_eq!(with(&[1.3e19]), Err(ObjectError::VectorTooLong));
        assert_eq!(with(&[4e18, 4e18]), Err(ObjectError::VectorTooLong));
        assert_eq!(with(&[1e20]), Err(ObjectError::VectorTooLong));
    }

    #[test]
    fn a_patch_merges_removes_and_keeps_what_it_does_not_mention() {
        let original = parse(
            json!({ "content": "v1", "importance": 0.5, "tags": ["a"] }),
            Some(vec![0.25; 768]),
        )
        .unwrap();
        let patch = json!({ "content": "v2", "importance": null, "deleted": true });
        let patched = original
            .patched(&workforce_memory(), patch.as_object().unwrap(), None)
            .unwrap();
        assert_eq!(patched.id, original.id);
        assert_eq!(patched.properties["content"], Value::Text("v2".into()));
        assert_eq!(patched.properties["deleted"], Value::Bool(true));
        assert_eq!(
            patched.properties["tags"],
            Value::TextArray(vec!["a".into()])
        );
        assert!(!patched.properties.contains_key("importance"));
        assert_eq!(patched.vector, original.vector);

        let revectored = original
            .patched(
                &workforce_memory(),
                &serde_json::Map::new(),
                Some(vec![1.0; 768]),
            )
            .unwrap();
        assert_eq!(revectored.vector, Some(vec![1.0; 768]));
        assert_eq!(revectored.properties, original.properties);

        let wrong = json!({ "importance": "high" });
        assert_eq!(
            original.patched(&workforce_memory(), wrong.as_object().unwrap(), None),
            Err(ObjectError::Type {
                property: "importance".into(),
                expected: DataType::Number
            })
        );
        assert_eq!(
            original.patched(
                &workforce_memory(),
                &serde_json::Map::new(),
                Some(vec![0.0; 3])
            ),
            Err(ObjectError::VectorLength {
                expected: 768,
                got: 3
            })
        );
    }

    #[test]
    fn embedded_text_is_the_listed_properties_in_the_schemas_order() {
        let mut schema = workforce_memory();
        schema.embedding.properties = vec!["source".into(), "tags".into(), "content".into()];
        let object = |properties: serde_json::Value| {
            Object::from_json(
                &schema,
                ObjectId(uuid::Uuid::from_u128(1)),
                properties.as_object().unwrap(),
                None,
            )
            .unwrap()
        };
        let full = object(json!({
            "content": "the body",
            "tags": ["one", "two"],
            "importance": 0.5,
            "owner": "not embedded",
            "source": "A source",
        }));
        assert_eq!(
            full.embedded_text(&schema.embedding),
            "A source\none\ntwo\nthe body"
        );
        assert_eq!(
            object(json!({ "importance": 1 })).embedded_text(&schema.embedding),
            ""
        );
        assert_eq!(
            object(json!({ "tags": [] })).embedded_text(&schema.embedding),
            ""
        );
    }
}
