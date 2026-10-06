//! Nearest-vector search over a partition's layers.

use vellev_core::{Collection, CollectionName, Object, ObjectError};
use vellev_index::filter::{Filter, Predicate};
use vellev_index::vector::{Scorer, VectorIndex};

use crate::format::CollectionId;
use crate::state::{Layer, State, SHORTLIST_ABOVE};
use crate::Result;

/// How many times `limit` rows a layer shortlists before scoring exactly.
/// Higher finds more of the true nearest at more cost per query; zero is
/// taken as one.
pub const DEFAULT_OVERSAMPLE: usize = 64;

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
        let mut hits: Vec<Hit<'_>> = self
            .layers()
            .flat_map(|layer| nearest(layer, id, &scorer, predicate.as_ref(), query))
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

/// The layer's candidates, each scored exactly: every allowed live row
/// when there are few, otherwise the shortlist its vector index picks.
fn nearest<'a>(
    layer: Layer<'a>,
    collection: CollectionId,
    scorer: &Scorer<'_>,
    predicate: Option<&Predicate>,
    query: &Query<'_>,
) -> Vec<Hit<'a>> {
    let Some(index) = layer.indexes.get(&collection) else {
        return Vec::new();
    };
    let only = predicate.map(|predicate| predicate.select(&index.rows));
    let allowed = only.as_ref().unwrap_or(index.rows.rows());
    let live = || allowed.iter().filter(|row| !layer.dead[*row as usize]);
    let shortlist = index
        .vectors
        .as_ref()
        .filter(|_| many_are_live(allowed.len(), &layer, live));
    let candidates: Vec<u32> = match shortlist {
        Some(vectors) => vectors.shortlist(
            query.vector,
            query.limit.saturating_mul(query.oversample.max(1)),
            only.as_ref(),
            layer.dead,
        ),
        None => live().collect(),
    };
    candidates
        .into_iter()
        .filter_map(|row| {
            let found = layer.live(row as usize);
            let vector = found.vector?;
            let distance = scorer.distance(vector);
            Some(Hit {
                object: found.object,
                vector,
                distance,
                certainty: scorer.certainty(distance),
            })
        })
        .collect()
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
    let dead = layer.dead.iter().filter(|dead| **dead).count() as u64;
    allowed > threshold
        && (allowed - allowed.min(dead) > threshold || live().count() as u64 > threshold)
}
