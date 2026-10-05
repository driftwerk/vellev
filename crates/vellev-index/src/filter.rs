//! Exact filtering over a collection's properties.
//!
//! A [`Filter`] is what a caller sends. Compiled against a collection it
//! becomes a [`Predicate`], which can be asked two ways: of one object
//! ([`Predicate::matches`], the definition of what a filter means) or of a
//! [`RowIndex`] over many ([`Predicate::select`]). The two always agree.
//!
//! A value that is absent matches no comparison. `ne` is exactly "not `eq`",
//! so it does match an absent value: a row with no tags is not excluded by
//! "tag is not x".

use std::collections::{BTreeMap, HashMap};
use std::ops::{Bound, RangeBounds};

use chrono::{DateTime, Utc};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use vellev_core::{Collection, DataType, Object, ObjectError, Value};

/// How deeply filters may nest; far beyond any real query, and shallow
/// enough that evaluating one cannot exhaust the stack.
pub const MAX_DEPTH: usize = 32;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    /// Equal to `value`; for a text array, contains it.
    Eq {
        property: String,
        value: Json,
    },
    /// Not equal to `value`, or absent; for a text array, does not contain it.
    Ne {
        property: String,
        value: Json,
    },
    Gt {
        property: String,
        value: Json,
    },
    Gte {
        property: String,
        value: Json,
    },
    Lt {
        property: String,
        value: Json,
    },
    Lte {
        property: String,
        value: Json,
    },
    IsNull {
        property: String,
    },
    /// Every one holds; with none, everything matches.
    And(Vec<Filter>),
    /// At least one holds; with none, nothing matches.
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FilterError {
    #[error("property {0:?} is not declared on the collection")]
    UnknownProperty(String),
    #[error("property {0:?} is not filterable")]
    NotFilterable(String),
    #[error("a filter on {property:?} needs a value of type {expected}")]
    Type {
        property: String,
        expected: DataType,
    },
    #[error("{property:?} cannot be compared with null; use is_null")]
    Null { property: String },
    #[error("{property:?} is {data_type} and has no order to compare by")]
    Unordered {
        property: String,
        data_type: DataType,
    },
    #[error("filters nest deeper than {MAX_DEPTH}")]
    TooDeep,
}

/// A single value as filters compare it. Within one property every key has
/// the same variant, so the derived order is the value's own order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    Bool(bool),
    Int(i64),
    Number(Finite),
    Date(DateTime<Utc>),
    Text(String),
}

/// A finite `f64` with a total order, in which the two zeros are one value.
#[derive(Debug, Clone, Copy)]
pub struct Finite(f64);

impl Finite {
    fn new(value: f64) -> Self {
        Self(if value == 0.0 { 0.0 } else { value })
    }
}

impl PartialEq for Finite {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Finite {}

impl PartialOrd for Finite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Finite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Key {
    /// The keys a stored value is found under: one for a scalar, one per
    /// element for a text array.
    fn of(value: &Value) -> Vec<Key> {
        match value {
            Value::Text(text) => vec![Key::Text(text.clone())],
            Value::TextArray(items) => items.iter().cloned().map(Key::Text).collect(),
            Value::Int(n) => vec![Key::Int(*n)],
            Value::Number(n) => vec![Key::Number(Finite::new(*n))],
            Value::Bool(b) => vec![Key::Bool(*b)],
            Value::Date(d) => vec![Key::Date(*d)],
        }
    }
}

/// A filter checked against a collection: every property exists and is
/// filterable, and every value has its property's type.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Eq {
        property: String,
        key: Key,
    },
    Range {
        property: String,
        lower: Bound<Key>,
        upper: Bound<Key>,
    },
    IsNull(String),
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
}

impl Filter {
    pub fn compile(&self, collection: &Collection) -> Result<Predicate, FilterError> {
        self.compiled(collection, MAX_DEPTH)
    }

    fn compiled(&self, collection: &Collection, depth: usize) -> Result<Predicate, FilterError> {
        let deeper = depth.checked_sub(1).ok_or(FilterError::TooDeep)?;
        let all = |filters: &[Filter]| {
            filters
                .iter()
                .map(|f| f.compiled(collection, deeper))
                .collect::<Result<Vec<_>, _>>()
        };
        let range =
            |property: &String, value: &Json, bounds: fn(Key) -> (Bound<Key>, Bound<Key>)| {
                let key = ordered_key(collection, property, value)?;
                let (lower, upper) = bounds(key);
                Ok(Predicate::Range {
                    property: property.clone(),
                    lower,
                    upper,
                })
            };
        match self {
            Filter::Eq { property, value } => equals(collection, property, value),
            Filter::Ne { property, value } => {
                equals(collection, property, value).map(|eq| Predicate::Not(Box::new(eq)))
            }
            Filter::Gt { property, value } => range(property, value, |key| {
                (Bound::Excluded(key), Bound::Unbounded)
            }),
            Filter::Gte { property, value } => range(property, value, |key| {
                (Bound::Included(key), Bound::Unbounded)
            }),
            Filter::Lt { property, value } => range(property, value, |key| {
                (Bound::Unbounded, Bound::Excluded(key))
            }),
            Filter::Lte { property, value } => range(property, value, |key| {
                (Bound::Unbounded, Bound::Included(key))
            }),
            Filter::IsNull { property } => {
                filterable(collection, property).map(|_| Predicate::IsNull(property.clone()))
            }
            Filter::And(filters) => all(filters).map(Predicate::And),
            Filter::Or(filters) => all(filters).map(Predicate::Or),
            Filter::Not(filter) => filter
                .compiled(collection, deeper)
                .map(|p| Predicate::Not(Box::new(p))),
        }
    }
}

fn filterable(collection: &Collection, property: &str) -> Result<DataType, FilterError> {
    let declared = collection
        .property(property)
        .ok_or_else(|| FilterError::UnknownProperty(property.to_string()))?;
    if declared.filterable {
        Ok(declared.data_type)
    } else {
        Err(FilterError::NotFilterable(property.to_string()))
    }
}

/// The key to look `value` up by. A text array is compared by element, so
/// its value is one text, not an array.
fn key(collection: &Collection, property: &str, value: &Json) -> Result<Key, FilterError> {
    let data_type = filterable(collection, property)?;
    let mismatch = || FilterError::Type {
        property: property.to_string(),
        expected: data_type,
    };
    if value.is_null() {
        return Err(FilterError::Null {
            property: property.to_string(),
        });
    }
    if data_type == DataType::TextArray {
        return value
            .as_str()
            .map(|text| Key::Text(text.to_string()))
            .ok_or_else(mismatch);
    }
    match Value::typed(collection, property, value) {
        Ok(Some(typed)) => Key::of(&typed).pop().ok_or_else(mismatch),
        Ok(None) | Err(ObjectError::Type { .. }) => Err(mismatch()),
        Err(_) => Err(FilterError::UnknownProperty(property.to_string())),
    }
}

fn equals(collection: &Collection, property: &str, value: &Json) -> Result<Predicate, FilterError> {
    key(collection, property, value).map(|key| Predicate::Eq {
        property: property.to_string(),
        key,
    })
}

fn ordered_key(collection: &Collection, property: &str, value: &Json) -> Result<Key, FilterError> {
    match filterable(collection, property)? {
        DataType::Int | DataType::Number | DataType::Date => key(collection, property, value),
        data_type => Err(FilterError::Unordered {
            property: property.to_string(),
            data_type,
        }),
    }
}

impl Predicate {
    /// Whether `object` passes the filter.
    pub fn matches(&self, object: &Object) -> bool {
        let keys = |property: &String| object.properties.get(property).map(Key::of);
        match self {
            Predicate::Eq { property, key } => {
                keys(property).is_some_and(|keys| keys.contains(key))
            }
            Predicate::Range {
                property,
                lower,
                upper,
            } => keys(property).is_some_and(|keys| {
                keys.iter()
                    .any(|key| (lower.as_ref(), upper.as_ref()).contains(key))
            }),
            Predicate::IsNull(property) => !object.properties.contains_key(property),
            Predicate::And(all) => all.iter().all(|p| p.matches(object)),
            Predicate::Or(any) => any.iter().any(|p| p.matches(object)),
            Predicate::Not(predicate) => !predicate.matches(object),
        }
    }

    /// The rows of `index` that pass the filter.
    pub fn select(&self, index: &RowIndex) -> RoaringBitmap {
        let property = |name: &String| index.properties.get(name);
        match self {
            Predicate::Eq {
                property: name,
                key,
            } => property(name)
                .and_then(|p| p.values.get(key))
                .cloned()
                .unwrap_or_default(),
            Predicate::Range {
                property: name,
                lower,
                upper,
            } => property(name)
                .map(|p| {
                    p.values
                        .range((lower.as_ref(), upper.as_ref()))
                        .fold(RoaringBitmap::new(), |found, (_, rows)| found | rows)
                })
                .unwrap_or_default(),
            Predicate::IsNull(name) => match property(name) {
                Some(p) => &index.rows - &p.present,
                None => index.rows.clone(),
            },
            Predicate::And(all) => all
                .iter()
                .fold(index.rows.clone(), |found, p| found & p.select(index)),
            Predicate::Or(any) => any
                .iter()
                .fold(RoaringBitmap::new(), |found, p| found | p.select(index)),
            Predicate::Not(predicate) => &index.rows - predicate.select(index),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PropertyIndex {
    values: BTreeMap<Key, RoaringBitmap>,
    present: RoaringBitmap,
}

/// Which rows hold which value, for each filterable property of one
/// collection's rows. Rows are numbered by the caller.
#[derive(Debug, Clone, Default)]
pub struct RowIndex {
    rows: RoaringBitmap,
    properties: HashMap<String, PropertyIndex>,
}

impl RowIndex {
    pub fn rows(&self) -> &RoaringBitmap {
        &self.rows
    }

    /// Record `object` as row `row`, indexing its filterable properties.
    pub fn insert(&mut self, collection: &Collection, row: u32, object: &Object) {
        self.rows.insert(row);
        let filterable = object.properties.iter().filter(|(name, _)| {
            collection
                .property(name)
                .is_some_and(|property| property.filterable)
        });
        for (name, value) in filterable {
            let index = self.properties.entry(name.clone()).or_default();
            index.present.insert(row);
            for key in Key::of(value) {
                index.values.entry(key).or_default().insert(row);
            }
        }
    }
}
