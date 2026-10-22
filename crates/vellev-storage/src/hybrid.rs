//! Vector and keyword search fused into one ranking.

use std::collections::HashMap;

use vellev_core::{CollectionName, Object};
use vellev_index::filter::Filter;
use vellev_index::fusion::Fusion;

use crate::keyword::{searched, Keywords};
use crate::search::{Query, DEFAULT_OVERSAMPLE};
use crate::state::State;
use crate::{Result, StorageError, Work};

/// How much of a fused score is the vector search's to give.
pub const DEFAULT_ALPHA: f32 = 0.75;

/// How many results each search offers for fusing, when `limit` asks for
/// no more.
pub const DEFAULT_CANDIDATES: usize = 100;

/// The most rows a hybrid search that names no `oversample` asks each
/// segment that shortlists to score exactly, whatever its limit, unless
/// its fusion pool is larger: see [`default_oversample`].
pub const HYBRID_DEFAULT_RESCORE: usize = 1_024;

/// The oversample of a hybrid search for `limit` results that names none:
/// [`DEFAULT_OVERSAMPLE`], lowered for a large `limit` so that `limit`
/// times it is at most [`HYBRID_DEFAULT_RESCORE`], and never below 1.
pub fn default_oversample(limit: usize) -> usize {
    DEFAULT_OVERSAMPLE.min((HYBRID_DEFAULT_RESCORE / limit.max(1)).max(1))
}

#[derive(Debug, Clone)]
pub struct Hybrid<'a> {
    pub text: &'a str,
    /// What `text` means, when known. Without it the keyword search stands
    /// alone, with the whole weight.
    pub vector: Option<&'a [f32]>,
    /// The vector search's weight, from 0 to 1; the keyword search has the
    /// rest. A search with no weight is not run, though what it would have
    /// been asked is still checked.
    pub alpha: f32,
    pub fusion: Fusion,
    /// As [`Keywords::properties`].
    pub properties: &'a [(&'a str, f32)],
    pub limit: usize,
    /// How many results each search offers for fusing.
    pub candidates: usize,
    pub filter: Option<&'a Filter>,
    /// How many times `limit` rows the vector search scores exactly, and
    /// at least `candidates`: it scales the results asked for, not the
    /// pool fused from. The budget is per layer: each segment that
    /// shortlists scores up to that many, while a segment too small to
    /// shortlist, and the memtable, are scored whole.
    pub oversample: usize,
}

impl<'a> Hybrid<'a> {
    pub fn matching(text: &'a str, vector: Option<&'a [f32]>, limit: usize) -> Self {
        Self {
            text,
            vector,
            alpha: DEFAULT_ALPHA,
            fusion: Fusion::default(),
            properties: &[],
            limit,
            candidates: DEFAULT_CANDIDATES,
            filter: None,
            oversample: DEFAULT_OVERSAMPLE,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Fused<'a> {
    pub object: &'a Object,
    pub vector: Option<&'a [f32]>,
    /// Larger is better.
    pub score: f32,
    /// As [`crate::Hit`], when the vector search offered the object.
    pub distance: Option<f32>,
    pub certainty: Option<f32>,
    /// As [`crate::Scored`], when the keyword search offered the object.
    pub keyword: Option<f32>,
}

impl Fused<'_> {
    /// The object's version: of its properties and its stored vector.
    pub fn version(&self) -> vellev_core::Version {
        vellev_core::Version::of(&self.object.properties, self.vector)
    }
}

impl<'a> Fused<'a> {
    fn of(object: &'a Object, vector: Option<&'a [f32]>) -> Self {
        Self {
            object,
            vector,
            score: 0.0,
            distance: None,
            certainty: None,
            keyword: None,
        }
    }

    /// What orders objects with the same score, as in Weaviate: the
    /// wording's value where there is one, otherwise the vector's.
    fn second(&self) -> f32 {
        self.keyword
            .or(self.distance.map(|distance| 1.0 - distance))
            .unwrap_or(f32::MIN)
    }
}

impl State {
    /// The collection's live objects that best match `query` by meaning and
    /// by wording together, best first. Equal scores are ordered by the
    /// keyword score, or without one the nearness, and then by object id.
    pub fn hybrid(
        &self,
        collection: &CollectionName,
        query: &Hybrid<'_>,
    ) -> Result<Vec<Fused<'_>>> {
        self.hybrid_counted(collection, query, &mut Work::default())
    }

    /// As [`Self::hybrid`], adding what both its legs did to `work`.
    pub fn hybrid_counted(
        &self,
        collection: &CollectionName,
        query: &Hybrid<'_>,
        work: &mut Work,
    ) -> Result<Vec<Fused<'_>>> {
        if !(0.0..=1.0).contains(&query.alpha) {
            return Err(StorageError::InvalidQuery(format!(
                "alpha must be from 0 to 1, not {}",
                query.alpha
            )));
        }
        let (_, schema) = self.catalog().known(collection)?;
        searched(schema, query.properties)?;
        query
            .vector
            .map_or(Ok(()), |vector| schema.embedding.check(vector))?;
        let alpha = query.vector.map_or(0.0, |_| query.alpha);
        let candidates = query.candidates.max(query.limit);
        let nearest = match query.vector {
            // The vector leg offers `candidates` for fusing, but scores
            // exactly only as many rows as the results asked for call for:
            // `limit × oversample`, or the pool itself if that is larger.
            // Scaling the pool instead scored a 12,800-row budget for ten
            // results, nearly every row of a mid-sized collection.
            Some(vector) if alpha > 0.0 => self.search_within(
                collection,
                &Query {
                    vector,
                    limit: candidates,
                    filter: query.filter,
                    oversample: query.oversample,
                },
                query
                    .limit
                    .saturating_mul(query.oversample.max(1))
                    .max(candidates),
                work,
            )?,
            _ => Vec::new(),
        };
        let worded = if alpha < 1.0 {
            let keywords = Keywords {
                text: query.text,
                properties: query.properties,
                limit: candidates,
                filter: query.filter,
            };
            self.keyword_counted(collection, &keywords, work)?
        } else {
            Vec::new()
        };

        let nearness: Vec<f32> = nearest.iter().map(|hit| 1.0 - hit.distance).collect();
        let wording: Vec<f32> = worded.iter().map(|scored| scored.score).collect();
        let by_meaning = nearest
            .iter()
            .zip(query.fusion.shares(alpha, &nearness))
            .map(|(hit, share)| {
                let found = Fused {
                    distance: Some(hit.distance),
                    certainty: hit.certainty,
                    ..Fused::of(hit.object, Some(hit.vector))
                };
                (found, share)
            });
        let by_wording = worded
            .iter()
            .zip(query.fusion.shares(1.0 - alpha, &wording))
            .map(|(scored, share)| {
                let found = Fused {
                    keyword: Some(scored.score),
                    ..Fused::of(scored.object, scored.vector)
                };
                (found, share)
            });
        let mut fused: Vec<Fused<'_>> = by_meaning
            .chain(by_wording)
            .fold(HashMap::new(), |mut fused, (found, share)| {
                let entry = fused
                    .entry(found.object.id)
                    .or_insert(Fused::of(found.object, found.vector));
                *entry = Fused {
                    object: entry.object,
                    vector: entry.vector.or(found.vector),
                    score: entry.score + share,
                    distance: found.distance.or(entry.distance),
                    certainty: found.certainty.or(entry.certainty),
                    keyword: found.keyword.or(entry.keyword),
                };
                fused
            })
            .into_values()
            .collect();
        fused.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.second().total_cmp(&a.second()))
                .then(a.object.id.cmp(&b.object.id))
        });
        fused.truncate(query.limit);
        Ok(fused)
    }
}
