//! Nearest-vector search over a partition's layers.

use vellev_core::{CollectionName, Object, ObjectError};
use vellev_index::filter::{Filter, Predicate};
use vellev_index::vector::{Scorer, VectorIndex};

use crate::format::CollectionId;
use crate::state::{Layer, State};
use crate::Result;

/// How many times `limit` rows a layer shortlists before scoring exactly.
/// Higher finds more of the true nearest at more cost per query.
pub const DEFAULT_OVERSAMPLE: usize = 64;

/// Up to this many rows a layer scores every one exactly; only beyond it is
/// the compressed shortlist, and with it any approximation, used.
pub(crate) const EXACT_UP_TO: usize = 20_000;

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
        let expected = schema.embedding.dimensions;
        if query.vector.len() != expected as usize {
            return Err(ObjectError::VectorLength {
                expected,
                got: query.vector.len(),
            }
            .into());
        }
        if !query.vector.iter().all(|value| value.is_finite()) {
            return Err(ObjectError::VectorNotFinite.into());
        }
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

/// The layer's candidates, each scored exactly: every allowed row when
/// there are few, otherwise the shortlist its vector index picks.
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
    let candidates: Vec<u32> = match &index.vectors {
        Some(vectors) if allowed.len() > EXACT_UP_TO as u64 => vectors.shortlist(
            query.vector,
            query.limit.saturating_mul(query.oversample),
            only.as_ref(),
            layer.dead,
        ),
        _ => allowed
            .iter()
            .filter(|row| !layer.dead[*row as usize])
            .collect(),
    };
    candidates
        .into_iter()
        .filter_map(|row| {
            let object = &layer.rows[row as usize].object;
            let distance = scorer.distance(object.vector.as_deref()?);
            Some(Hit {
                object,
                distance,
                certainty: scorer.certainty(distance),
            })
        })
        .collect()
}
