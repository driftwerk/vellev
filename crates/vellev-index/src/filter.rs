//! Exact filtering over a collection's properties.
//!
//! A [`Filter`] is what a caller sends. Compiled against a collection it
//! becomes a [`Predicate`], which can be asked two ways: of one object
//! ([`Predicate::matches`], the definition of what a filter means) or of a
//! [`RowIndex`] over many ([`Predicate::select`]). The two always agree.
//!
//! Text matches the whole value and is never tokenized; a text array matches
//! by element. A value must have its property's type, except that a whole
//! number is accepted where a number is declared.
//!
//! A property with no value matches no comparison. `ne` is exactly "not
//! `eq`", so it does match one: a row with no tags is not excluded by "tag is
//! not x". An empty text array is a value, not the lack of one: `is_null`
//! does not match it.

use std::collections::{BTreeMap, HashMap};
use std::ops::{Bound, RangeBounds};

use chrono::{DateTime, Utc};
use roaring::{MultiOps, RoaringBitmap};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use vellev_core::{Collection, DataType, Object, Value};

/// How deeply filters may nest; far beyond any real query, and shallow
/// enough that evaluating one cannot exhaust the stack.
pub const MAX_DEPTH: usize = 32;

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Filter {
    /// Equal to `value`; for a text array, contains the text `value`.
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
    #[cfg_attr(feature = "openapi", schema(no_recursion))]
    And(Vec<Filter>),
    /// At least one holds; with none, nothing matches.
    #[cfg_attr(feature = "openapi", schema(no_recursion))]
    Or(Vec<Filter>),
    #[cfg_attr(feature = "openapi", schema(no_recursion))]
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
enum Key {
    Bool(bool),
    Int(i64),
    Number(Finite),
    Date(DateTime<Utc>),
    Text(String),
}

/// A finite `f64` with a total order, in which the two zeros are one value.
#[derive(Debug, Clone, Copy)]
struct Finite(f64);

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
/// filterable, and every value has its property's type. It can only be made
/// by [`Filter::compile`].
#[derive(Debug, Clone, PartialEq)]
pub struct Predicate(Node);

#[derive(Debug, Clone, PartialEq)]
enum Node {
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
    And(Vec<Node>),
    Or(Vec<Node>),
    Not(Box<Node>),
}

impl Filter {
    pub fn compile(&self, collection: &Collection) -> Result<Predicate, FilterError> {
        self.compiled(collection, MAX_DEPTH).map(Predicate)
    }

    fn compiled(&self, collection: &Collection, depth: usize) -> Result<Node, FilterError> {
        let deeper = depth.checked_sub(1).ok_or(FilterError::TooDeep)?;
        let all = |filters: &[Filter]| {
            filters
                .iter()
                .map(|f| f.compiled(collection, deeper))
                .collect::<Result<Vec<_>, _>>()
        };
        let equals = |property: &String, value: &Json| {
            key(collection, property, value).map(|key| Node::Eq {
                property: property.clone(),
                key,
            })
        };
        let range =
            |property: &String, value: &Json, bounds: fn(Key) -> (Bound<Key>, Bound<Key>)| {
                let (lower, upper) = bounds(ordered_key(collection, property, value)?);
                Ok(Node::Range {
                    property: property.clone(),
                    lower,
                    upper,
                })
            };
        match self {
            Filter::Eq { property, value } => equals(property, value),
            Filter::Ne { property, value } => {
                equals(property, value).map(|eq| Node::Not(Box::new(eq)))
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
                filterable(collection, property).map(|_| Node::IsNull(property.clone()))
            }
            Filter::And(filters) => all(filters).map(Node::And),
            Filter::Or(filters) => all(filters).map(Node::Or),
            Filter::Not(filter) => filter
                .compiled(collection, deeper)
                .map(|n| Node::Not(Box::new(n))),
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

/// The key to look `value` up by. A text array is
/// compared by element, so the value it takes is one text.
fn key(collection: &Collection, property: &str, value: &Json) -> Result<Key, FilterError> {
    let data_type = filterable(collection, property)?;
    if value.is_null() {
        return Err(FilterError::Null {
            property: property.to_string(),
        });
    }
    let found = match data_type {
        DataType::Text | DataType::TextArray => {
            value.as_str().map(|text| Key::Text(text.to_string()))
        }
        DataType::Int => value.as_i64().map(Key::Int),
        DataType::Number => value
            .as_f64()
            .filter(|n| n.is_finite())
            .map(|n| Key::Number(Finite::new(n))),
        DataType::Bool => value.as_bool().map(Key::Bool),
        DataType::Date => value
            .as_str()
            .and_then(|text| DateTime::parse_from_rfc3339(text).ok())
            .map(|date| Key::Date(date.with_timezone(&Utc))),
    };
    let expected = match data_type {
        DataType::TextArray => DataType::Text,
        scalar => scalar,
    };
    found.ok_or_else(|| FilterError::Type {
        property: property.to_string(),
        expected,
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
        self.0.matches(object)
    }

    /// The rows of `index` that pass the filter.
    pub fn select(&self, index: &RowIndex) -> RoaringBitmap {
        self.0.select(index)
    }

    /// Whether the filter has no condition at all, and so passes everything.
    pub fn is_unconditional(&self) -> bool {
        matches!(&self.0, Node::And(all) if all.is_empty())
    }
}

impl Node {
    fn matches(&self, object: &Object) -> bool {
        let keys = |property: &String| object.properties.get(property).map(Key::of);
        match self {
            Node::Eq { property, key } => keys(property).is_some_and(|keys| keys.contains(key)),
            Node::Range {
                property,
                lower,
                upper,
            } => keys(property).is_some_and(|keys| {
                keys.iter()
                    .any(|key| (lower.as_ref(), upper.as_ref()).contains(key))
            }),
            Node::IsNull(property) => !object.properties.contains_key(property),
            Node::And(all) => all.iter().all(|node| node.matches(object)),
            Node::Or(any) => any.iter().any(|node| node.matches(object)),
            Node::Not(node) => !node.matches(object),
        }
    }

    fn select(&self, index: &RowIndex) -> RoaringBitmap {
        let property = |name: &String| index.properties.get(name);
        match self {
            Node::Eq {
                property: name,
                key,
            } => property(name)
                .and_then(|p| p.values.get(key))
                .cloned()
                .unwrap_or_default(),
            // One union over all the values in range: folding them pairwise
            // copies the growing result once per value.
            Node::Range {
                property: name,
                lower,
                upper,
            } => property(name)
                .map(|p| p.range(lower.as_ref(), upper.as_ref()))
                .unwrap_or_default(),
            Node::IsNull(name) => match property(name) {
                Some(p) => &index.rows - &p.present,
                None => index.rows.clone(),
            },
            Node::And(all) => all
                .iter()
                .fold(index.rows.clone(), |found, n| found & n.select(index)),
            Node::Or(any) => any.iter().map(|node| node.select(index)).union(),
            Node::Not(node) => &index.rows - node.select(index),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PropertyIndex {
    values: BTreeMap<Key, RoaringBitmap>,
    present: RoaringBitmap,
    runs: Option<Runs>,
}

/// How many values, or runs of them, each run of a [`Runs`] level unions.
const RUN: usize = 256;

/// The rows of runs of consecutive values unioned ahead of time, so that a
/// range over many values unions a few of these and the values at its
/// edges. Level 0 unions `run` values, and each level above unions `run`
/// runs of the level below.
#[derive(Debug, Clone)]
struct Runs {
    run: usize,
    /// The first value of each run of level 0.
    firsts: Vec<Key>,
    levels: Vec<Vec<RoaringBitmap>>,
}

impl Runs {
    /// Of `values`, when they can be ranged over and there are enough of
    /// them for a range to gain by it.
    fn of(values: &BTreeMap<Key, RoaringBitmap>, run: usize) -> Option<Self> {
        let ordered = matches!(
            values.keys().next(),
            Some(Key::Int(_) | Key::Number(_) | Key::Date(_))
        );
        if !ordered || values.len() < run * 4 {
            return None;
        }
        let firsts = values.keys().step_by(run).cloned().collect();
        let bitmaps: Vec<&RoaringBitmap> = values.values().collect();
        let mut levels = vec![unions(&bitmaps, run)];
        while let Some(top) = levels.last().filter(|top| top.len() > run) {
            let below: Vec<&RoaringBitmap> = top.iter().collect();
            levels.push(unions(&below, run));
        }
        Some(Self {
            run,
            firsts,
            levels,
        })
    }

    /// The runs of level 0 lying wholly within the bounds, as `a..b`.
    fn within(&self, lower: Bound<&Key>, upper: Bound<&Key>) -> std::ops::Range<usize> {
        let below = |key: &Key| match lower {
            Bound::Included(lower) => key < lower,
            Bound::Excluded(lower) => key <= lower,
            Bound::Unbounded => false,
        };
        let a = self.firsts.partition_point(below);
        // A run lies wholly under the upper bound when the next one starts
        // at or under it, or when the range is open above.
        let b = match upper {
            Bound::Unbounded => self.firsts.len(),
            Bound::Included(upper) | Bound::Excluded(upper) => self
                .firsts
                .partition_point(|key| key <= upper)
                .saturating_sub(1),
        };
        a..b.max(a)
    }

    /// The bitmaps that together hold the rows of level 0 runs `a..b`, the
    /// fewest the levels allow.
    fn cover<'a>(&'a self, level: usize, a: usize, b: usize, out: &mut Vec<&'a RoaringBitmap>) {
        if level == 0 {
            out.extend(&self.levels[0][a..b]);
            return;
        }
        let size = self.run.pow(level as u32);
        let (first, last) = (a.div_ceil(size), b / size);
        if first < last {
            self.cover(level - 1, a, first * size, out);
            out.extend(&self.levels[level][first..last]);
            self.cover(level - 1, last * size, b, out);
        } else {
            self.cover(level - 1, a, b, out);
        }
    }
}

/// The union of each run of `run` bitmaps.
fn unions(bitmaps: &[&RoaringBitmap], run: usize) -> Vec<RoaringBitmap> {
    bitmaps
        .chunks(run)
        .map(|chunk| chunk.iter().copied().union())
        .collect()
}

/// Whether no value lies within the bounds, which a [`BTreeMap`] range
/// refuses.
fn inverted(lower: Bound<&Key>, upper: Bound<&Key>) -> bool {
    match (lower, upper) {
        (Bound::Included(lower), Bound::Included(upper)) => lower > upper,
        (Bound::Included(lower) | Bound::Excluded(lower), Bound::Excluded(upper))
        | (Bound::Excluded(lower), Bound::Included(upper)) => lower >= upper,
        _ => false,
    }
}

impl PropertyIndex {
    /// The rows holding a value within the bounds.
    fn range(&self, lower: Bound<&Key>, upper: Bound<&Key>) -> RoaringBitmap {
        if inverted(lower, upper) {
            return RoaringBitmap::new();
        }
        let values = |lower, upper| self.values.range((lower, upper)).map(|(_, rows)| rows);
        let Some((runs, whole)) = self
            .runs
            .as_ref()
            .map(|runs| (runs, runs.within(lower, upper)))
            .filter(|(_, whole)| !whole.is_empty())
        else {
            return values(lower, upper).union();
        };
        let mut parts = Vec::new();
        runs.cover(runs.levels.len() - 1, whole.start, whole.end, &mut parts);
        parts.extend(values(lower, Bound::Excluded(&runs.firsts[whole.start])));
        // `whole` ends at the last run only when the range is open above.
        if let Some(next) = runs.firsts.get(whole.end) {
            parts.extend(values(Bound::Included(next), upper));
        }
        parts.into_iter().union()
    }
}

/// Which rows hold which value, for each filterable property of one
/// collection's rows. Rows are numbered by the caller.
///
/// Every distinct value of a filterable property is kept as a key, so a
/// property holding long, mostly unique text is cheaper declared not
/// filterable.
#[derive(Debug, Clone, Default)]
pub struct RowIndex {
    rows: RoaringBitmap,
    properties: HashMap<String, PropertyIndex>,
}

impl RowIndex {
    /// About how many bytes the index takes in memory: every row recorded,
    /// and each property's values, with the text of each, the rows holding
    /// each, and the runs over them. Rough, and a walk over every value.
    pub fn resident(&self) -> usize {
        const VALUE: usize = std::mem::size_of::<(Key, RoaringBitmap)>() + 16;
        let bitmap = |rows: &RoaringBitmap| rows.serialized_size();
        let properties: usize = self
            .properties
            .iter()
            .map(|(name, index)| {
                let values: usize = index
                    .values
                    .iter()
                    .map(|(key, rows)| {
                        let text = match key {
                            Key::Text(text) => text.capacity(),
                            _ => 0,
                        };
                        VALUE + text + bitmap(rows)
                    })
                    .sum();
                let runs: usize = index.runs.as_ref().map_or(0, |runs| {
                    runs.levels
                        .iter()
                        .flatten()
                        .map(|rows| std::mem::size_of::<RoaringBitmap>() + bitmap(rows))
                        .sum::<usize>()
                        + runs.firsts.len() * std::mem::size_of::<Key>()
                });
                name.capacity() + bitmap(&index.present) + values + runs
            })
            .sum();
        bitmap(&self.rows) + properties
    }

    /// Every row recorded.
    pub fn rows(&self) -> &RoaringBitmap {
        &self.rows
    }

    /// Ready the index for ranges over many values, as for an index that
    /// takes no more rows: the next insert undoes it.
    pub fn summarise(&mut self) {
        for index in self.properties.values_mut() {
            index.runs = Runs::of(&index.values, RUN);
        }
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
            index.runs = None;
            index.present.insert(row);
            for key in Key::of(value) {
                index.values.entry(key).or_default().insert(row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(n: i64) -> Vec<Bound<Key>> {
        (-1..=n)
            .flat_map(|k| [Bound::Included(Key::Int(k)), Bound::Excluded(Key::Int(k))])
            .chain([Bound::Unbounded])
            .collect()
    }

    /// Small runs, so that every level, partial top runs, and bounds on
    /// every run's first and last values are all reached.
    #[test]
    fn a_range_over_runs_selects_what_it_selects_value_by_value() {
        for n in [16, 17, 63, 64, 65, 257] {
            let values: BTreeMap<Key, RoaringBitmap> = (0..n)
                .map(|k| {
                    (
                        Key::Int(k),
                        RoaringBitmap::from_iter([k as u32, 1_000 + k as u32]),
                    )
                })
                .collect();
            let plain = PropertyIndex {
                values: values.clone(),
                ..PropertyIndex::default()
            };
            let summarised = PropertyIndex {
                runs: Runs::of(&values, 4),
                ..plain.clone()
            };
            assert!(summarised.runs.is_some());
            for lower in bounds(n) {
                for upper in bounds(n) {
                    let (lower, upper) = (lower.as_ref(), upper.as_ref());
                    let expected: RoaringBitmap = values
                        .iter()
                        .filter(|(key, _)| (lower, upper).contains(*key))
                        .map(|(_, rows)| rows)
                        .union();
                    assert_eq!(
                        plain.range(lower, upper),
                        expected,
                        "{n}: {lower:?}..{upper:?}"
                    );
                    assert_eq!(
                        summarised.range(lower, upper),
                        expected,
                        "{n}: {lower:?}..{upper:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn only_values_that_can_be_ranged_over_are_summarised() {
        let text: BTreeMap<Key, RoaringBitmap> = (0..2_000)
            .map(|k| (Key::Text(format!("{k:05}")), RoaringBitmap::from_iter([k])))
            .collect();
        assert!(Runs::of(&text, RUN).is_none());
    }
}
