//! Keyword search over a partition's layers.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use vellev_core::{Bm25, Collection, CollectionName, Object};
use vellev_index::filter::Filter;
use vellev_index::text::{self, Posting, TextIndex};

use crate::state::{Layer, LayerIndex, State};
use crate::{Result, StorageError};

/// The most a match in one property may count against one in another.
pub const MAX_WEIGHT: f32 = 1_000.0;

#[derive(Debug, Clone)]
pub struct Keywords<'a> {
    pub text: &'a str,
    /// The properties searched and how much a match in each counts, up to
    /// [`MAX_WEIGHT`]; every searchable property, equally, when empty.
    pub properties: &'a [(&'a str, f32)],
    pub limit: usize,
    /// Only rows passing this are returned. It has no part in the scores.
    pub filter: Option<&'a Filter>,
}

impl<'a> Keywords<'a> {
    pub fn matching(text: &'a str, limit: usize) -> Self {
        Self {
            text,
            properties: &[],
            limit,
            filter: None,
        }
    }

    pub fn within(self, properties: &'a [(&'a str, f32)]) -> Self {
        Self { properties, ..self }
    }

    pub fn filtered(self, filter: &'a Filter) -> Self {
        Self {
            filter: Some(filter),
            ..self
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scored<'a> {
    pub object: &'a Object,
    pub vector: Option<&'a [f32]>,
    /// The BM25 score; larger is a better match.
    pub score: f32,
}

/// A match ordered so that the greatest is the worst: the one a full list
/// of the best gives up first.
struct Worst<'a>(Scored<'a>);

impl Ord for Worst<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then(self.0.object.id.cmp(&other.0.object.id))
    }
}

impl PartialOrd for Worst<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Worst<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Worst<'_> {}

/// One token looked for in one property.
struct Clause<'a> {
    property: &'a str,
    token: &'a str,
    /// The token's rarity, times the property's weight and how many times
    /// the query repeats the token.
    weight: f32,
    average: f32,
}

impl Clause<'_> {
    fn holding<'a>(&'a self, index: &'a TextIndex) -> impl Iterator<Item = Posting> + 'a {
        index
            .field(self.property)
            .into_iter()
            .flat_map(|field| field.holding(self.token))
    }
}

impl State {
    /// The collection's live objects holding any token of `query.text`,
    /// best match first, ties in object id order. Scores are BM25 under the
    /// collection's parameters, over every live object whatever the filter.
    pub fn keyword(
        &self,
        collection: &CollectionName,
        query: &Keywords<'_>,
    ) -> Result<Vec<Scored<'_>>> {
        let (id, schema) = self.catalog().known(collection)?;
        let properties = searched(schema, query.properties)?;
        let predicate = query
            .filter
            .map(|filter| filter.compile(schema))
            .transpose()?;
        let Some(corpus) = self.corpus(id) else {
            return Ok(Vec::new());
        };
        let layers: Vec<(Layer<'_>, &LayerIndex)> = self
            .layers()
            .filter_map(|layer| layer.indexes.get(&id).map(|index| (layer, index)))
            .collect();
        let repeats = text::terms(query.text).fold(BTreeMap::new(), |mut repeats, token| {
            *repeats.entry(token).or_insert(0u32) += 1;
            repeats
        });
        let clauses: Vec<Clause<'_>> = properties
            .iter()
            .flat_map(|property| repeats.iter().map(move |repeat| (property, repeat)))
            .filter_map(|((property, weight), (token, repeats))| {
                let field = corpus.fields.get(*property)?;
                let clause = Clause {
                    property,
                    token,
                    weight: weight * *repeats as f32,
                    average: field.words as f32 / field.rows as f32,
                };
                let holding: usize = layers
                    .iter()
                    .map(|(layer, index)| {
                        let live = |posting: &Posting| !layer.dead[posting.row as usize];
                        clause.holding(&index.text).filter(live).count()
                    })
                    .sum();
                (holding > 0).then(|| Clause {
                    weight: clause.weight * text::idf(corpus.objects, holding as u64),
                    ..clause
                })
            })
            .collect();
        if clauses.is_empty() {
            return Ok(Vec::new());
        }
        let mut best = BinaryHeap::new();
        for (layer, index) in &layers {
            let allowed = predicate
                .as_ref()
                .map(|predicate| predicate.select(&index.rows));
            scores(schema.bm25, &index.text, &clauses)
                .filter(|(row, _)| !layer.dead[*row as usize])
                .filter(|(row, _)| allowed.as_ref().is_none_or(|rows| rows.contains(*row)))
                .for_each(|(row, score)| {
                    let found = layer.found(row as usize);
                    best.push(Worst(Scored {
                        object: found.object,
                        vector: found.vector,
                        score,
                    }));
                    if best.len() > query.limit {
                        best.pop();
                    }
                });
        }
        Ok(best
            .into_sorted_vec()
            .into_iter()
            .map(|worst| worst.0)
            .collect())
    }
}

/// The properties a query searches, each with its weight.
pub(crate) fn searched<'a>(
    schema: &'a Collection,
    asked: &'a [(&'a str, f32)],
) -> Result<Vec<(&'a str, f32)>> {
    if asked.is_empty() {
        return Ok(schema
            .properties
            .iter()
            .filter(|property| property.searchable)
            .map(|property| (property.name.as_str(), 1.0))
            .collect());
    }
    asked
        .iter()
        .enumerate()
        .map(|(at, (name, weight))| {
            let searchable = schema
                .property(name)
                .is_some_and(|property| property.searchable);
            let refusal = if !searchable {
                Some(format!(
                    "{} has no searchable property {name:?}",
                    schema.name
                ))
            } else if asked[..at].iter().any(|(earlier, _)| earlier == name) {
                Some(format!("{name:?} is searched twice"))
            } else if !(*weight > 0.0 && *weight <= MAX_WEIGHT) {
                Some(format!(
                    "the weight of {name:?} must be above 0 and at most {MAX_WEIGHT}, not {weight}"
                ))
            } else {
                None
            };
            refusal.map_or(Ok((*name, *weight)), |reason| {
                Err(StorageError::InvalidQuery(reason))
            })
        })
        .collect()
}

/// Every row of a layer that matches a clause, ascending, with its score:
/// the clauses' rows merged, each row's contributions added in clause order.
fn scores<'a>(
    bm25: Bm25,
    index: &'a TextIndex,
    clauses: &'a [Clause<'a>],
) -> impl Iterator<Item = (u32, f32)> + 'a {
    let mut cursors: Vec<_> = clauses
        .iter()
        .map(|clause| (clause, clause.holding(index).peekable()))
        .collect();
    std::iter::from_fn(move || {
        let row = cursors
            .iter_mut()
            .filter_map(|(_, cursor)| cursor.peek().map(|posting| posting.row))
            .min()?;
        let score = cursors
            .iter_mut()
            .filter_map(|(clause, cursor)| {
                let posting = cursor.next_if(|posting| posting.row == row)?;
                Some(text::score(
                    bm25,
                    clause.weight,
                    posting.frequency as f32,
                    posting.length as f32,
                    clause.average,
                ))
            })
            .sum();
        Some((row, score))
    })
}
