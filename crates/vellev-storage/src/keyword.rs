//! Keyword search over a partition's layers.

use std::collections::{BTreeMap, HashMap};

use vellev_core::{Collection, CollectionName, Object};
use vellev_index::filter::Filter;
use vellev_index::text::{self, Postings};

use crate::state::{Layer, LayerIndex, State};
use crate::{Result, StorageError};

#[derive(Debug, Clone)]
pub struct Keywords<'a> {
    pub text: &'a str,
    /// The properties searched and how much a match in each counts; every
    /// searchable property, equally, when empty.
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
    /// The BM25 score; larger is a better match.
    pub score: f32,
}

/// One token looked for in one property.
struct Clause<'a> {
    property: &'a str,
    token: &'a str,
    /// The token's rarity, times the property's weight and how many times
    /// the query repeats the token.
    weight: f32,
    average: f32,
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
        let repeats = text::tokens(query.text).fold(BTreeMap::new(), |mut repeats, token| {
            *repeats.entry(token).or_insert(0u32) += 1;
            repeats
        });
        let clauses: Vec<Clause<'_>> = properties
            .iter()
            .flat_map(|property| repeats.iter().map(move |repeat| (property, repeat)))
            .filter_map(|((property, weight), (token, repeats))| {
                let holding: usize = layers
                    .iter()
                    .filter_map(|(layer, index)| {
                        Some((layer, index.text.postings(property, token)?))
                    })
                    .map(|(layer, postings)| live(postings, layer).count())
                    .sum();
                let field = corpus.fields.get(*property)?;
                (holding > 0).then(|| Clause {
                    property,
                    token,
                    weight: text::idf(corpus.objects, holding as u64) * weight * *repeats as f32,
                    average: field.tokens as f32 / field.rows as f32,
                })
            })
            .collect();
        let mut matches: Vec<Scored<'_>> = layers
            .iter()
            .flat_map(|(layer, index)| {
                let allowed = predicate
                    .as_ref()
                    .map(|predicate| predicate.select(&index.rows));
                scores(schema, layer, index, &clauses)
                    .into_iter()
                    .filter(move |(row, _)| allowed.as_ref().is_none_or(|rows| rows.contains(*row)))
                    .map(|(row, score)| Scored {
                        object: &layer.rows[row as usize].object,
                        score,
                    })
            })
            .collect();
        matches.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.object.id.cmp(&b.object.id))
        });
        matches.truncate(query.limit);
        Ok(matches)
    }
}

/// The properties a query searches, each with its weight.
fn searched<'a>(
    schema: &'a Collection,
    asked: &'a [(&'a str, f32)],
) -> Result<Vec<(&'a str, f32)>> {
    let invalid = |reason: String| Err(StorageError::InvalidQuery(reason));
    if asked.is_empty() {
        return Ok(schema
            .properties
            .iter()
            .filter(|property| property.searchable)
            .map(|property| (property.name.as_str(), 1.0))
            .collect());
    }
    for (name, weight) in asked {
        if !schema
            .property(name)
            .is_some_and(|property| property.searchable)
        {
            return invalid(format!(
                "{} has no searchable property {name:?}",
                schema.name
            ));
        }
        if !(weight.is_finite() && *weight > 0.0) {
            return invalid(format!(
                "the weight of {name:?} must be positive, not {weight}"
            ));
        }
    }
    Ok(asked.to_vec())
}

fn live<'a>(postings: &'a Postings, layer: &'a Layer<'_>) -> impl Iterator<Item = (u32, u32)> + 'a {
    postings
        .iter()
        .filter(|(row, _)| !layer.dead[*row as usize])
}

/// The score of every live row of the layer that matches a clause.
fn scores(
    schema: &Collection,
    layer: &Layer<'_>,
    index: &LayerIndex,
    clauses: &[Clause<'_>],
) -> HashMap<u32, f32> {
    let mut scores = HashMap::new();
    for clause in clauses {
        let Some(postings) = index.text.postings(clause.property, clause.token) else {
            continue;
        };
        for (row, frequency) in live(postings, layer) {
            *scores.entry(row).or_insert(0.0) += text::score(
                schema.bm25,
                clause.weight,
                frequency as f32,
                index.text.length(clause.property, row) as f32,
                clause.average,
            );
        }
    }
    scores
}
