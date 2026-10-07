//! Nearest-vector search over a partition's layers.

use vellev_core::{CollectionName, Object};
use vellev_index::filter::{Filter, Predicate};
use vellev_index::vector::{nearest, Scorer, VectorIndex};

use crate::format::CollectionId;
use crate::state::{Layer, State, Vectors, SHORTLIST_ABOVE};
use crate::Result;

/// How many times `limit` rows are scored exactly: those the clustered
/// layers estimate nearest, taken together, with at least `limit` of each,
/// and as many from each smaller layer. Higher finds more of the true nearest at more cost per query;
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
    /// those at one distance in object id order. Objects without a vector
    /// are never returned.
    ///
    /// Where a layer's index shortlists, rows it estimates alike are cut in
    /// object id order too, so that of objects with one vector the lowest
    /// ids are found first. Rows it estimates differently are found, or
    /// missed, as any other near neighbour is, whatever their distance.
    pub fn search(&self, collection: &CollectionName, query: &Query<'_>) -> Result<Vec<Hit<'_>>> {
        let (id, schema) = self.catalog().known(collection)?;
        schema.embedding.check(query.vector)?;
        let predicate = query
            .filter
            .map(|filter| filter.compile(schema))
            .transpose()?;
        let scorer = Scorer::new(schema.distance, query.vector);
        let budget = query.limit.saturating_mul(query.oversample.max(1));
        let layers: Vec<Layer<'_>> = self.layers().collect();
        let mut exact = Vec::new();
        let mut estimated = Vec::new();
        let mut floors = 0;
        for (at, layer) in layers.iter().enumerate() {
            let offered = candidates(layer, id, predicate.as_ref(), query.vector, budget);
            exact.extend(offered.exact.into_iter().map(|row| (at, row)));
            // Each layer's own nearest few are always scored: one whose
            // estimates err more than others' would otherwise crowd them
            // out with rows that only look near.
            if let Some(mut rows) = offered.estimated {
                let rest = nearest(&mut rows, query.limit, |a, b| {
                    layer.id(*a).cmp(&layer.id(*b))
                });
                floors += rows.len();
                exact.extend(rows.into_iter().map(|(_, row)| (at, row)));
                estimated.extend(
                    rest.into_iter()
                        .map(|(estimate, row)| (estimate, (at, row))),
                );
            }
        }
        nearest(
            &mut estimated,
            budget.saturating_sub(floors),
            |(a_at, a), (b_at, b)| layers[*a_at].id(*a).cmp(&layers[*b_at].id(*b)),
        );
        let estimated = estimated.into_iter().map(|(_, found)| found);
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

/// What a layer offers a query.
#[derive(Default)]
struct Candidates {
    /// Rows to score exactly.
    exact: Vec<u32>,
    /// Rows its clustered index estimates nearest, with their estimated
    /// distances, to be scored only if among the nearest of every layer.
    estimated: Option<Vec<(f32, u32)>>,
}

/// The layer's candidates: every allowed live row when there are few,
/// otherwise up to `count` its vector index picks, and besides them every
/// allowed row it has no estimate for.
fn candidates(
    layer: &Layer<'_>,
    collection: CollectionId,
    predicate: Option<&Predicate>,
    vector: &[f32],
    count: usize,
) -> Candidates {
    let Some(index) = layer.indexes.get(&collection) else {
        return Candidates::default();
    };
    let only = predicate.map(|predicate| predicate.select(&index.rows));
    let allowed = only.as_ref().unwrap_or(index.rows.rows());
    let live = || allowed.iter().filter(|row| !layer.dead[*row as usize]);
    let shortlist = index
        .vectors
        .as_ref()
        .filter(|_| many_are_live(allowed.len(), layer, live));
    let Some(vectors) = shortlist else {
        return Candidates {
            exact: live().collect(),
            estimated: None,
        };
    };
    let ties = |a, b| layer.id(a).cmp(&layer.id(b));
    let unmeasured = vectors.unmeasured(only.as_ref(), layer.dead);
    match vectors {
        Vectors::Clustered(clustered) => Candidates {
            exact: unmeasured,
            estimated: Some(clustered.estimates(vector, count, only.as_ref(), layer.dead, &ties)),
        },
        Vectors::Binary(binary) => Candidates {
            exact: binary
                .shortlist(vector, count, only.as_ref(), layer.dead, &ties)
                .into_iter()
                .chain(unmeasured)
                .collect(),
            estimated: None,
        },
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
