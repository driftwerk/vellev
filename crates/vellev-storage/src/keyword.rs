//! Keyword search over a partition's layers.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use vellev_core::{Bm25, Collection, CollectionName, Object};
use vellev_index::filter::Filter;
use vellev_index::text::{self, Cursor, Posting, TextIndex};

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

impl Scored<'_> {
    /// The object's version: of its properties and its stored vector.
    pub fn version(&self) -> vellev_core::Version {
        vellev_core::Version::of(&self.object.properties, self.vector)
    }
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
    fn cursor<'a>(&self, index: &'a TextIndex) -> Option<Cursor<'a>> {
        index.field(self.property)?.cursor(self.token)
    }

    fn score(&self, parameters: Bm25, posting: Posting) -> f32 {
        text::score(
            parameters,
            self.weight,
            posting.frequency as f32,
            posting.length as f32,
            self.average,
        )
    }
}

/// The best matches so far, the worst of them on top.
struct Best<'a> {
    limit: usize,
    heap: BinaryHeap<Worst<'a>>,
}

impl<'a> Best<'a> {
    /// The score a match must reach to be worth considering, once the list
    /// is full: a match scoring the same may still rank above the worst by
    /// id, so it is not ruled out.
    fn threshold(&self) -> Option<f32> {
        (self.heap.len() >= self.limit)
            .then(|| self.heap.peek().map(|worst| worst.0.score))
            .flatten()
    }

    fn offer(&mut self, scored: Scored<'a>) {
        self.heap.push(Worst(scored));
        if self.heap.len() > self.limit {
            self.heap.pop();
        }
    }
}

/// Whether a bound, a sum of up to `terms` parts, `reaches` the threshold,
/// allowing for the rounding of each part and for the sum being taken in
/// another order than the score it bounds.
fn reaches(bound: f32, threshold: Option<f32>, terms: usize) -> bool {
    let slack = 1.0 + (2 * terms + 4) as f32 * f32::EPSILON;
    threshold.is_none_or(|threshold| bound * slack >= threshold)
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
        if query.limit == 0 {
            return Ok(Vec::new());
        }
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
                    .filter_map(|(layer, index)| {
                        let cursor = clause.cursor(&index.text)?;
                        Some(match layer.deaths {
                            0 => cursor.len(),
                            _ => cursor
                                .filter(|posting| !layer.dead[posting.row as usize])
                                .count(),
                        })
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
        let mut best = Best {
            limit: query.limit,
            heap: BinaryHeap::new(),
        };
        for (layer, index) in &layers {
            let allowed = predicate
                .as_ref()
                .map(|predicate| predicate.select(&index.rows));
            let wanted = |row: u32| {
                !layer.dead[row as usize] && allowed.as_ref().is_none_or(|rows| rows.contains(row))
            };
            top(schema.bm25, &index.text, &clauses, wanted, &mut best, layer);
        }
        Ok(best
            .heap
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

/// One clause's walk along a layer's postings, with the most any of them
/// can score.
struct Walk<'a> {
    clause: usize,
    cursor: Cursor<'a>,
    bound: f32,
}

/// Offer `best` every wanted row of a layer that can reach it, scored as
/// the exhaustive search would score it: the clauses' contributions added
/// in clause order. Rows that cannot reach the worst of the best, by the
/// bounds of the postings they are in, are passed over unscored, a block
/// at a time where the blocks allow (block-max WAND).
fn top<'a>(
    parameters: Bm25,
    index: &'a TextIndex,
    clauses: &[Clause<'_>],
    wanted: impl Fn(u32) -> bool,
    best: &mut Best<'a>,
    layer: &Layer<'a>,
) {
    let mut walks: Vec<Walk<'a>> = clauses
        .iter()
        .enumerate()
        .filter_map(|(at, clause)| {
            let cursor = clause.cursor(index)?;
            let bound = cursor.bound(parameters, clause.weight, clause.average);
            Some(Walk {
                clause: at,
                cursor,
                bound,
            })
        })
        .collect();
    let block_at = |walk: &Walk<'a>, row: u32| {
        let clause = &clauses[walk.clause];
        walk.cursor
            .block_at(row, parameters, clause.weight, clause.average)
    };
    loop {
        walks.retain(|walk| walk.cursor.row().is_some());
        walks.sort_by_key(|walk| walk.cursor.row());
        let threshold = best.threshold();
        let mut reach = 0.0;
        let Some(pivot) = walks.iter().position(|walk| {
            reach += walk.bound;
            reaches(reach, threshold, clauses.len())
        }) else {
            break;
        };
        let row = walks[pivot]
            .cursor
            .row()
            .expect("a walk with postings left");
        // Every walk at the pivot's row counts toward it, however far along
        // the bounds it comes, so that a skip past the row passes over all
        // that the row could score.
        let level = pivot
            + walks[pivot + 1..]
                .iter()
                .take_while(|walk| walk.cursor.row() == Some(row))
                .count();
        let blocks: Vec<(u32, f32)> = match threshold {
            Some(_) => walks[..=level]
                .iter()
                .filter_map(|walk| block_at(walk, row))
                .collect(),
            None => Vec::new(),
        };
        let within: f32 = blocks.iter().map(|(_, bound)| bound).sum();
        if !reaches(within, threshold, clauses.len()) {
            let past = blocks
                .iter()
                .map(|(end, _)| end.saturating_add(1))
                .min()
                .unwrap_or(row.saturating_add(1));
            let next = walks
                .get(level + 1)
                .and_then(|walk| walk.cursor.row())
                .map_or(past, |beyond| past.min(beyond));
            walks[..=level]
                .iter_mut()
                .for_each(|walk| walk.cursor.seek(next));
            continue;
        }
        if walks[0].cursor.row() != Some(row) {
            let behind = walks[..pivot]
                .iter_mut()
                .filter(|walk| walk.cursor.row() < Some(row))
                .max_by(|a, b| a.bound.total_cmp(&b.bound))
                .expect("a walk short of the pivot");
            behind.cursor.seek(row);
            continue;
        }
        let at = walks
            .iter()
            .take_while(|walk| walk.cursor.row() == Some(row))
            .count();
        if wanted(row) {
            walks[..at].sort_by_key(|walk| walk.clause);
            let score = walks[..at]
                .iter()
                .map(|walk| {
                    let posting = walk.cursor.posting().expect("a walk at the row");
                    clauses[walk.clause].score(parameters, posting)
                })
                .sum();
            let live = layer.live(row as usize);
            best.offer(Scored {
                object: live.object,
                vector: live.vector,
                score,
            });
        }
        walks[..at]
            .iter_mut()
            .for_each(|walk| walk.cursor.advance());
    }
}
