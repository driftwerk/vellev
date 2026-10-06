//! Nearest-vector search over a partition's layers.

use vellev_core::{Collection, CollectionName, Object, ObjectError};
use vellev_index::filter::{Filter, Predicate};
use vellev_index::vector::{Scorer, VectorIndex};

use crate::format::CollectionId;
use crate::state::{Layer, State, Vectors, SHORTLIST_ABOVE};
use crate::Result;

/// How many times `limit` rows are scored exactly: those the clustered
/// layers estimate nearest, taken together, and as many from each smaller
/// layer. Higher finds more of the true nearest at more cost per query;
/// zero is taken as one.
pub const DEFAULT_OVERSAMPLE: usize = 128;

#[derive(Debug, Clone)]
pub struct Query<'a> {
    pub vector: &'a [f32],
    pub limit: usize,
    /// Only rows passing this are considered, exactly.
    pub filter: Option<&'a Filter>,
    pub oversample: usize,
}

impl<'a> Query<'a> {
    pub fn nearest(vector: &'a [f32], limit: usize) -> Self {
        Self {
            vector,
            limit,
            filter: None,
            oversample: DEFAULT_OVERSAMPLE,
        }
    }

    pub fn filtered(self, filter: &'a Filter) -> Self {
        Self {
            filter: Some(filter),
            ..self
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit<'a> {
    pub object: &'a Object,
    pub vector: &'a [f32],
    /// Under the collection's metric; smaller is nearer.
    pub distance: f32,
    /// For cosine, `(1 + cosine) / 2`.
    pub certainty: Option<f32>,
}

impl State {
    /// The collection's live objects nearest `query.vector`, nearest first,
    /// ties in object id order. Objects without a vector are never returned.
    pub fn search(&self, collection: &CollectionName, query: &Query<'_>) -> Result<Vec<Hit<'_>>> {
        let (id, schema) = self.catalog().known(collection)?;
        comparable(schema, query.vector)?;
        let predicate = query
            .filter
            .map(|filter| filter.compile(schema))
            .transpose()?;
        let scorer = Scorer::new(schema.distance, query.vector);
        let budget = query.limit.saturating_mul(query.oversample.max(1));
        let layers: Vec<Layer<'_>> = self.layers().collect();
        let mut exact = Vec::new();
        let mut estimated = Vec::new();
        for (at, layer) in layers.iter().enumerate() {
            match candidates(layer, id, predicate.as_ref(), query.vector, budget) {
                Candidates::Exact(rows) => exact.extend(rows.into_iter().map(|row| (at, row))),
                Candidates::Estimated(rows) => {
                    estimated.extend(rows.into_iter().map(|(estimate, row)| (estimate, at, row)))
                }
            }
        }
        if estimated.len() > budget {
            estimated.select_nth_unstable_by(budget, |a, b| a.0.total_cmp(&b.0));
            estimated.truncate(budget);
        }
        let estimated = estimated.into_iter().map(|(_, at, row)| (at, row));
        let mut hits: Vec<Hit<'_>> = exact
            .into_iter()
            .chain(estimated)
            .filter_map(|(at, row)| hit(&layers[at], row, &scorer))
            .collect();
        hits.sort_unstable_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then(a.object.id.cmp(&b.object.id))
        });
        hits.truncate(query.limit);
        Ok(hits)
    }
}

/// Whether `vector` can be compared with the collection's.
pub(crate) fn comparable(schema: &Collection, vector: &[f32]) -> Result<()> {
    let expected = schema.embedding.dimensions;
    if vector.len() != expected as usize {
        return Err(ObjectError::VectorLength {
            expected,
            got: vector.len(),
        }
        .into());
    }
    if !vector.iter().all(|value| value.is_finite()) {
        return Err(ObjectError::VectorNotFinite.into());
    }
    Ok(())
}

/// What a layer offers a query.
enum Candidates {
    /// Rows to score exactly.
    Exact(Vec<u32>),
    /// Rows its clustered index estimates nearest, with their estimated
    /// distances, to be scored only if among the nearest of every layer.
    Estimated(Vec<(f32, u32)>),
}

/// The layer's candidates: every allowed live row when there are few,
/// otherwise up to `count` its vector index picks.
fn candidates(
    layer: &Layer<'_>,
    collection: CollectionId,
    predicate: Option<&Predicate>,
    vector: &[f32],
    count: usize,
) -> Candidates {
    let Some(index) = layer.indexes.get(&collection) else {
        return Candidates::Exact(Vec::new());
    };
    let only = predicate.map(|predicate| predicate.select(&index.rows));
    let allowed = only.as_ref().unwrap_or(index.rows.rows());
    let live = || allowed.iter().filter(|row| !layer.dead[*row as usize]);
    let shortlist = index
        .vectors
        .as_ref()
        .filter(|_| many_are_live(allowed.len(), layer, live));
    match shortlist {
        Some(Vectors::Clustered(clustered)) => {
            Candidates::Estimated(clustered.estimates(vector, count, only.as_ref(), layer.dead))
        }
        Some(vectors) => {
            Candidates::Exact(vectors.shortlist(vector, count, only.as_ref(), layer.dead))
        }
        None => Candidates::Exact(live().collect()),
    }
}

/// The layer's live `row` scored exactly, if it has a vector.
fn hit<'a>(layer: &Layer<'a>, row: u32, scorer: &Scorer<'_>) -> Option<Hit<'a>> {
    let found = layer.live(row as usize);
    let vector = found.vector?;
    let distance = scorer.distance(vector);
    Some(Hit {
        object: found.object,
        vector,
        distance,
        certainty: scorer.certainty(distance),
    })
}

/// Whether more than [`SHORTLIST_ABOVE`] of the `allowed` rows are live.
/// The live rows are only counted when the layer has enough dead ones, of
/// any collection, for the answer to be in doubt.
fn many_are_live<I: Iterator<Item = u32>>(
    allowed: u64,
    layer: &Layer<'_>,
    live: impl Fn() -> I,
) -> bool {
    let threshold = SHORTLIST_ABOVE as u64;
    let dead = layer.deaths as u64;
    allowed > threshold
        && (allowed - allowed.min(dead) > threshold || live().count() as u64 > threshold)
}
