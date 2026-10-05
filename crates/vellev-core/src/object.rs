use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::{Collection, DataType, ObjectId};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
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
        let mut typed = BTreeMap::new();
        for (name, json) in properties {
            let property = collection
                .property(name)
                .ok_or_else(|| ObjectError::UnknownProperty(name.clone()))?;
            if json.is_null() {
                continue;
            }
            let value =
                Value::from_json(property.data_type, json).ok_or_else(|| ObjectError::Type {
                    property: name.clone(),
                    expected: property.data_type,
                })?;
            typed.insert(name.clone(), value);
        }
        if let Some(vector) = &vector {
            let expected = collection.embedding.dimensions;
            if vector.len() != expected as usize {
                return Err(ObjectError::VectorLength {
                    expected,
                    got: vector.len(),
                });
            }
            if !vector.iter().all(|v| v.is_finite()) {
                return Err(ObjectError::VectorNotFinite);
            }
        }
        Ok(Object {
            id,
            properties: typed,
            vector,
        })
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
}
