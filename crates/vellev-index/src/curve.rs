//! How many clusters of a clustered index a query scans, as a curve kept
//! with the segment rather than a count fixed in the index when built.
//!
//! A curve is made from sample questions: for each, the ranks, in the
//! question's order of clusters, of the clusters its own nearest rows lie
//! in. It keeps the rank that holds each share of all those rows, from
//! [`FROM`] to all of them in steps of [`STEP`]; the probes a query scans
//! are the rank at the share its source calls for, plus one.

use serde::{Deserialize, Serialize};

/// The least share a curve keeps a rank for.
pub const FROM: f64 = 0.95;
const _: () = assert!(
    POINTS == 101,
    "shares from 19,000 to 20,000 20,000ths in steps of 10"
);
/// The step between the shares a curve keeps.
pub const STEP: f64 = 0.0005;
/// How many shares a curve keeps: [`FROM`] to 1 in steps of [`STEP`].
pub const POINTS: usize = 101;
/// The share of sample questions' nearest rows a curve made from questions
/// asked must hold: recall's target (recall note, §10).
pub const TARGET: f64 = 0.998;

/// What a curve's samples were.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// Rows of the index itself, each asking for its nearest rows, as when
    /// the segment was built and before any question was asked: the
    /// probes scan all their nearest rows' clusters (recall note, R6.3).
    Rows,
    /// Questions asked of the node: the probes hold [`TARGET`] of their
    /// nearest rows.
    Queries,
    /// A source this version does not know, written by a later one: read,
    /// so that the segment it belongs to is, but never used, so that the
    /// segment scans as its index was built to.
    #[serde(other)]
    Unknown,
}

/// A segment's probe curve for one collection's clustered index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Curve {
    pub source: Source,
    /// How many sample questions it was made from.
    pub samples: u32,
    /// How many clusters the index it was made for has.
    pub clusters: u32,
    /// The rank, from 0, holding each share: [`POINTS`] of them.
    pub ranks: Vec<u16>,
}

impl Curve {
    /// The curve of `ranks`, the cluster ranks of `samples` questions'
    /// nearest rows in an index of `clusters`, or `None` if there are none
    /// or one is past the last cluster.
    pub fn of(source: Source, samples: u32, clusters: u32, mut ranks: Vec<u32>) -> Option<Self> {
        if ranks.is_empty()
            || ranks.iter().any(|rank| *rank >= clusters)
            || clusters > u32::from(u16::MAX)
        {
            return None;
        }
        ranks.sort_unstable();
        // Shares counted in 20,000ths, so that each is exact: FROM is
        // 19,000 of them and a step 10.
        let points = (0..POINTS)
            .map(|point| {
                let share = 19_000 + 10 * point;
                let at = (ranks.len() * share).div_ceil(20_000).clamp(1, ranks.len());
                ranks[at - 1] as u16
            })
            .collect();
        Some(Self {
            source,
            samples,
            clusters,
            ranks: points,
        })
    }

    /// How many clusters a query scans by this curve: every cluster that
    /// held a sample's nearest row for a curve of rows, and those holding
    /// [`TARGET`] of them for one of questions; never none, nor more than
    /// the index has.
    pub fn probes(&self) -> usize {
        let share = match self.source {
            Source::Rows | Source::Unknown => 1.0,
            Source::Queries => TARGET,
        };
        let point = (((share - FROM) / STEP).round() as usize).min(POINTS - 1);
        let rank = self.ranks.get(point).copied().unwrap_or(u16::MAX);
        (usize::from(rank) + 1).clamp(1, self.clusters.max(1) as usize)
    }

    /// Whether this curve is one for an index of `clusters`, whole, and
    /// of a source this version knows: a curve that is not is never used.
    pub fn fits(&self, clusters: usize) -> bool {
        self.source != Source::Unknown
            && self.clusters as usize == clusters
            && self.ranks.len() == POINTS
            && self
                .ranks
                .iter()
                .all(|rank| u32::from(*rank) < self.clusters)
            && self.ranks.windows(2).all(|pair| pair[0] <= pair[1])
    }
}
