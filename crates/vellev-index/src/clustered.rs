//! Nearest-vector search over clusters of one-bit codes.
//!
//! The vectors are grouped around centroids found by k-means. A query is
//! compared with every centroid, but with the rows of only the nearest
//! clusters. Each row is kept as a one-bit code of its vector's direction
//! from its centroid, after RaBitQ: the vectors are rotated at random first,
//! which spreads every direction over all the bits evenly, and a factor kept
//! with each code makes the inner product the code estimates unbiased. The
//! query is itself quantized to a few bits for the scan, so that comparing
//! it with a code costs a few AND and popcount instructions per word.
//!
//! How many clusters a query scans is settled when the index is built: the
//! fewest that hold the true nearest ten of sample queries often enough.
//! A query scans further for as long as its shortlist is not full, and one
//! allowed few rows visits just those rows.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize, Serializer};
use vellev_core::Distance;

use crate::curve::{Curve, Source};
use crate::fastscan::{chosen, Blocks, Tables, BLOCK};
use crate::rotation::{Random, Rotation};
use crate::vector::{
    allowed, as_far_from_every_vector, dot, in_row_order, measurable, nearest, normalised,
    or_farthest, short_enough, squared_l2, Scorer, Ties, VectorIndex,
};

/// How many bits the query is quantized to for the scan.
const QUERY_BITS: u32 = 4;
/// How many of its own rows an index is tuned on when built: each asks
/// for its nearest [`TUNING_NEIGHBOURS`], and the clusters they lie in make
/// the index's curve of its rows (recall note, R6.3 and R7).
const TUNING_QUERIES: usize = 400;
/// How many rows a block of the one pass tuning makes over the rows holds:
/// small in tests, so that their rows span many blocks and the blocks'
/// shortlists are merged.
const TUNING_BLOCK: usize = if cfg!(test) { 64 } else { 4096 };
const TUNING_NEIGHBOURS: usize = 10;
const ITERATIONS: usize = 8;
const MOST_CLUSTERS: usize = 4096;
/// How many vectors per cluster k-means is trained on, at most.
const TRAINING_PER_CLUSTER: usize = 64;

/// How a scan of clusters goes: how many clusters it probes at least, and
/// into how many runs it may be split on the read pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Probing {
    /// The clusters it scans at least: as a segment's probe curve says, or
    /// as the index was built to ([`ClusteredIndex::probes`]).
    pub probes: usize,
    /// The runs it may be split into; 1 scans on the calling thread.
    pub parts: usize,
}

/// In what an index is not one of the vectors it is asked to stand for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Unfit {
    #[error("it measures by another metric")]
    Metric,
    #[error("it is of vectors of other dimensions")]
    Dimensions,
    #[error("it is of other rows")]
    Rows,
}

/// Vectors clustered around centroids, each kept as a one-bit code.
#[derive(Clone, Deserialize)]
#[serde(try_from = "Wire")]
pub struct ClusteredIndex {
    metric: Distance,
    dimensions: usize,
    rotation: Rotation,
    /// The centroids in the rotated space, [`Rotation::width`] values each.
    centroids: Vec<f32>,
    clusters: Vec<Cluster>,
    /// Every indexed row in ascending order, and for each its cluster and
    /// its place in it; both follow from the clusters.
    rows: Vec<u32>,
    at: Vec<(u32, u32)>,
    /// The rows with no estimate, in ascending order; they follow from the
    /// clusters too.
    unmeasured: Vec<u32>,
    /// How many clusters a query scans.
    probes: usize,
    /// The curve of the rows the index was tuned on when built: kept with
    /// the segment by the storage layer, and not in the index's file.
    curve: Option<Curve>,
}

impl fmt::Debug for ClusteredIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClusteredIndex")
            .field("metric", &self.metric)
            .field("dimensions", &self.dimensions)
            .field("rows", &self.rows.len())
            .field("clusters", &self.clusters.len())
            .field("probes", &self.probes)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "ClusterWire", into = "ClusterWire")]
struct Cluster {
    rows: Vec<u32>,
    /// The rows' codes, laid out in blocks for the scan.
    blocks: Blocks,
    factors: Vec<Factor>,
}

/// What a code needs besides its bits to stand for a vector `o` in a cluster
/// with centroid `c`: `o`'s distance from `c`, how well the code's unit
/// vector aligns with the direction from `c` to `o`, and that direction's
/// inner product with `c`.
#[derive(Debug, Clone, Copy)]
struct Factor {
    norm: f32,
    align: f32,
    along: f32,
}

impl Factor {
    /// What a row too long to measure is kept with: a distance from its
    /// centroid that is not finite.
    const UNMEASURED: Self = Self {
        norm: f32::INFINITY,
        align: 1.0,
        along: 0.0,
    };

    /// The squared length of the vector `o = c + r` this factor stands for
    /// in a cluster whose centroid `c` has the squared length `centroid`:
    /// `|c|² + 2 |r| (r̂ · c) + |r|²`, in `f64`, where no term overflows. It
    /// is not finite for a row kept as [`Factor::UNMEASURED`].
    fn squared(&self, centroid: f64) -> f64 {
        let norm = f64::from(self.norm);
        centroid + 2.0 * norm * f64::from(self.along) + norm * norm
    }
}

/// The rows of `clusters`, whose centroids are `centroids`, `width` values
/// each, that are too long to measure, in ascending order. Each row's
/// squared length is worked out again from what is stored with it rather
/// than read from a mark: an index built before rows too long to measure
/// were set apart kept them as any other, often at a finite distance from
/// a centroid of their own, and only their length tells them apart.
fn unmeasured(clusters: &[Cluster], centroids: &[f32], width: usize) -> Vec<u32> {
    let mut rows: Vec<u32> = clusters
        .iter()
        .zip(centroids.chunks_exact(width))
        .flat_map(|(members, centroid)| {
            let centroid: f64 = centroid.iter().map(|v| f64::from(*v).powi(2)).sum();
            members
                .rows
                .iter()
                .zip(&members.factors)
                .filter(move |(_, factor)| !short_enough(factor.squared(centroid)))
                .map(|(row, _)| *row)
        })
        .collect();
    rows.sort_unstable();
    rows
}

/// The index on the wire: what cannot be rebuilt from the rest.
#[derive(Serialize, Deserialize)]
struct Wire {
    metric: Distance,
    dimensions: usize,
    rotation: Rotation,
    centroids: Vec<f32>,
    clusters: Vec<Cluster>,
    probes: usize,
}

impl Wire {
    /// What is wrong with the index as read, if its parts do not fit one
    /// another: a file that is corrupt, which would otherwise be found out
    /// by a panic at some later query.
    fn fault(&self) -> Option<String> {
        let width = self.rotation.width;
        if self.dimensions > width {
            return Some(format!(
                "{} dimensions in a rotation {width} wide",
                self.dimensions
            ));
        }
        if self.clusters.is_empty() {
            return Some("no clusters".to_string());
        }
        if !(1..=self.clusters.len()).contains(&self.probes) {
            return Some(format!(
                "{} clusters to scan of {}",
                self.probes,
                self.clusters.len()
            ));
        }
        if self.centroids.len() != self.clusters.len() * width {
            return Some(format!(
                "{} centroid values for {} clusters {width} wide",
                self.centroids.len(),
                self.clusters.len()
            ));
        }
        self.clusters
            .iter()
            .position(|cluster| !cluster.rows.is_empty() && cluster.blocks.words() != width / 64)
            .map(|cluster| format!("cluster {cluster}'s codes are not {width} bits a row"))
    }
}

impl TryFrom<Wire> for ClusteredIndex {
    type Error = String;

    fn try_from(wire: Wire) -> Result<Self, Self::Error> {
        if let Some(fault) = wire.fault() {
            return Err(fault);
        }
        let mut rows: Vec<(u32, (u32, u32))> = wire
            .clusters
            .iter()
            .enumerate()
            .flat_map(|(cluster, members)| {
                members
                    .rows
                    .iter()
                    .enumerate()
                    .map(move |(index, row)| (*row, (cluster as u32, index as u32)))
            })
            .collect();
        rows.sort_unstable();
        let (rows, at) = rows.into_iter().unzip();
        Ok(Self {
            metric: wire.metric,
            dimensions: wire.dimensions,
            unmeasured: unmeasured(&wire.clusters, &wire.centroids, wire.rotation.width),
            rotation: wire.rotation,
            centroids: wire.centroids,
            clusters: wire.clusters,
            rows,
            at,
            probes: wire.probes,
            curve: None,
        })
    }
}

impl Serialize for ClusteredIndex {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Borrowed<'a> {
            metric: Distance,
            dimensions: usize,
            rotation: &'a Rotation,
            centroids: &'a [f32],
            clusters: &'a [Cluster],
            probes: usize,
        }
        Borrowed {
            metric: self.metric,
            dimensions: self.dimensions,
            rotation: &self.rotation,
            centroids: &self.centroids,
            clusters: &self.clusters,
            probes: self.probes,
        }
        .serialize(serializer)
    }
}

/// A cluster on the wire: each column as little-endian bytes.
#[derive(Serialize, Deserialize)]
struct ClusterWire {
    #[serde(with = "le")]
    rows: Vec<u32>,
    #[serde(with = "le")]
    codes: Vec<u64>,
    #[serde(with = "le")]
    norms: Vec<f32>,
    #[serde(with = "le")]
    aligns: Vec<f32>,
    #[serde(with = "le")]
    alongs: Vec<f32>,
}

impl From<Cluster> for ClusterWire {
    fn from(cluster: Cluster) -> Self {
        Self {
            codes: cluster.blocks.codes(),
            rows: cluster.rows,
            norms: cluster.factors.iter().map(|f| f.norm).collect(),
            aligns: cluster.factors.iter().map(|f| f.align).collect(),
            alongs: cluster.factors.iter().map(|f| f.along).collect(),
        }
    }
}

/// A cluster is read back only if each of its rows has all its factors.
impl TryFrom<ClusterWire> for Cluster {
    type Error = String;

    fn try_from(wire: ClusterWire) -> Result<Self, Self::Error> {
        let rows = wire.rows.len();
        if [wire.norms.len(), wire.aligns.len(), wire.alongs.len()] != [rows; 3] {
            return Err(format!("a cluster of {rows} rows without factors for each"));
        }
        let words = wire.codes.len().checked_div(rows).unwrap_or(0);
        if words * rows != wire.codes.len() {
            return Err(format!(
                "a cluster of {rows} rows with {} code words, not whole codes",
                wire.codes.len()
            ));
        }
        let factors = wire
            .norms
            .into_iter()
            .zip(wire.aligns)
            .zip(wire.alongs)
            .map(|((norm, align), along)| Factor { norm, align, along })
            .collect();
        Ok(Self {
            blocks: Blocks::new(&wire.codes, words),
            rows: wire.rows,
            factors,
        })
    }
}

/// A query made ready for one cluster: its difference from the centroid,
/// quantized to [`QUERY_BITS`] bit planes, with what the estimate needs.
struct Probe {
    /// The levels bit by bit, a plane a bit, as a code was first counted
    /// against them: what the tests hold the tables to.
    #[cfg(test)]
    planes: Vec<u64>,
    #[cfg(test)]
    words: usize,
    lower: f32,
    delta: f32,
    sum: f32,
    norm_squared: f32,
    centroid_dot: f32,
    scale: f32,
}

/// A residual's level from its distance above the lowest in steps, `x`:
/// `x.round().clamp(0.0, 15.0)` as an integer, without a call to round
/// where the machine has no instruction for it. `x` is never below zero,
/// as the lowest is subtracted; for such an `x` rounding half away from
/// zero is its whole part, plus one if what is left is a half or more,
/// and both are exact in `f32`. Not a number gives 0 and infinity 15, as
/// before.
#[inline(always)]
fn level_of(x: f32) -> u8 {
    let whole = x as u32;
    let up = u32::from(x - whole as f32 >= 0.5);
    whole.saturating_add(up).min((1 << QUERY_BITS) - 1) as u8
}

/// The levels bit by bit, a plane a bit, `words` words a plane: as a code
/// was first counted against the query.
#[cfg(test)]
fn planes_of(levels: &[u8], words: usize) -> Vec<u64> {
    let mut planes = vec![0u64; words * QUERY_BITS as usize];
    for (word, chunk) in levels.chunks_exact(64).enumerate() {
        let mut bits = [0u64; QUERY_BITS as usize];
        for (at, level) in chunk.iter().enumerate() {
            let level = u64::from(*level);
            for (bit, plane) in bits.iter_mut().enumerate() {
                *plane |= (level >> bit & 1) << at;
            }
        }
        for (bit, plane) in bits.into_iter().enumerate() {
            planes[bit * words + word] = plane;
        }
    }
    planes
}

/// What readying the query for one cluster after another reuses.
#[derive(Default)]
struct Readying {
    residual: Vec<f32>,
    levels: Vec<u8>,
}

impl Probe {
    #[cfg(test)]
    fn new(query: &[f32], centroid: &[f32]) -> Self {
        Self::readied(query, centroid, &mut Readying::default())
    }

    /// The query readied for the cluster whose centroid is `centroid`: each
    /// dimension's level into `space.levels`, where a scan of blocks makes
    /// its tables from them.
    fn readied(query: &[f32], centroid: &[f32], space: &mut Readying) -> Self {
        space.residual.clear();
        space
            .residual
            .extend(query.iter().zip(centroid).map(|(q, c)| q - c));
        let residual = &space.residual;
        let (mut lower, mut upper) = (f32::INFINITY, f32::NEG_INFINITY);
        for value in residual {
            lower = lower.min(*value);
            upper = upper.max(*value);
        }
        let levels = ((1u32 << QUERY_BITS) - 1) as f32;
        let delta = (upper - lower) / levels;
        debug_assert_eq!(residual.len() % 64, 0, "a rotation's width is whole words");
        space.levels.clear();
        space.levels.resize(residual.len(), 0);
        if delta > 0.0 {
            for (level, value) in space.levels.iter_mut().zip(residual) {
                *level = level_of((value - lower) / delta);
            }
        }
        #[cfg(test)]
        let (planes, words) = (
            planes_of(&space.levels, residual.len() / 64),
            residual.len() / 64,
        );
        Self {
            #[cfg(test)]
            planes,
            #[cfg(test)]
            words,
            lower,
            delta,
            sum: residual.iter().sum(),
            norm_squared: dot(residual, residual),
            centroid_dot: dot(query, centroid),
            scale: 1.0 / (residual.len() as f32).sqrt(),
        }
    }

    /// The inner product of the direction a code stands for with the
    /// query's difference from the centroid.
    #[cfg(test)]
    fn inner(&self, code: &[u64], factor: &Factor) -> f32 {
        let (ones, weighted) = counted_in_one_pass(code, &self.planes, self.words);
        self.inner_of(ones, weighted, factor)
    }

    /// As [`Self::inner`], from a code's counts: its set bits, and the sum
    /// of the query's levels over them.
    #[inline(always)]
    fn inner_of(&self, ones: u32, weighted: u32, factor: &Factor) -> f32 {
        let set = self.lower * ones as f32 + self.delta * weighted as f32;
        (2.0 * set - self.sum) * self.scale / factor.align
    }

    #[cfg(test)]
    fn distance(&self, metric: Distance, code: &[u64], factor: &Factor) -> f32 {
        let (ones, weighted) = counted_in_one_pass(code, &self.planes, self.words);
        self.distance_of(metric, ones, weighted, factor)
    }

    /// As [`Self::distance`], from a code's counts.
    #[inline(always)]
    fn distance_of(&self, metric: Distance, ones: u32, weighted: u32, factor: &Factor) -> f32 {
        let inner = self.inner_of(ones, weighted, factor);
        match metric {
            Distance::L2 => {
                factor.norm * factor.norm + self.norm_squared - 2.0 * factor.norm * inner
            }
            Distance::Dot | Distance::Cosine => {
                -(self.centroid_dot + factor.norm * (inner + factor.along))
            }
        }
    }
}

const _: () = assert!(QUERY_BITS == 4, "the scan counts four planes");

/// Every count of a code in one pass over its words: its set bits, and
/// its bits set in each of the query's planes weighted by the plane's
/// place, `(ones, weighted)`. How codes were counted before they were laid
/// out in blocks, and what the tests hold the tables to; the scan itself
/// no longer counts this way, so the popcount dispatch step E chose it by
/// is gone.
#[inline(always)]
#[cfg(test)]
fn counted_in_one_pass(code: &[u64], planes: &[u64], words: usize) -> (u32, u32) {
    let code = &code[..words];
    let (p0, rest) = planes.split_at(words);
    let (p1, rest) = rest.split_at(words);
    let (p2, p3) = rest.split_at(words);
    let p3 = &p3[..words];
    let (mut ones, mut w0, mut w1, mut w2, mut w3) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for i in 0..words {
        let c = code[i];
        ones += c.count_ones();
        w0 += (c & p0[i]).count_ones();
        w1 += (c & p1[i]).count_ones();
        w2 += (c & p2[i]).count_ones();
        w3 += (c & p3[i]).count_ones();
    }
    (ones, w0 + (w1 << 1) + (w2 << 2) + (w3 << 3))
}

#[derive(PartialEq)]
struct Candidate {
    distance: f32,
    row: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.row.cmp(&other.row))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The `count` nearest candidates offered, farthest on top, and those
/// offered as far as the farthest of them that did not fit: which of the
/// candidates at that distance are kept is for `ties` to say.
struct Nearest<'a> {
    count: usize,
    heap: BinaryHeap<Candidate>,
    /// Never more than twice `count`: past that, only the first `count` by
    /// `ties` can be kept, and the rest are let go.
    tied: Vec<u32>,
    ties: Ties<'a>,
}

impl<'a> Nearest<'a> {
    fn new(count: usize, ties: Ties<'a>) -> Self {
        Self {
            count,
            heap: BinaryHeap::with_capacity(count + 1),
            tied: Vec::new(),
            ties,
        }
    }

    fn tie(&mut self, row: u32) {
        self.tied.push(row);
        if self.tied.len() > 2 * self.count {
            self.tied
                .select_nth_unstable_by(self.count, |a, b| (self.ties)(*a, *b));
            self.tied.truncate(self.count);
        }
    }

    fn offer(&mut self, distance: f32, row: u32) {
        let distance = or_farthest(distance);
        if self.heap.len() < self.count {
            self.heap.push(Candidate { distance, row });
            return;
        }
        // Most rows of a long scan are farther than the farthest kept:
        // turned away on a look, without taking the heap's top to change.
        if self
            .heap
            .peek()
            .is_some_and(|worst| distance.total_cmp(&worst.distance).is_gt())
        {
            return;
        }
        let Some(mut worst) = self.heap.peek_mut() else {
            return;
        };
        match distance.total_cmp(&worst.distance) {
            Ordering::Greater => {}
            Ordering::Equal => {
                drop(worst);
                self.tie(row);
            }
            Ordering::Less => {
                let out = std::mem::replace(&mut *worst, Candidate { distance, row });
                drop(worst);
                match self.heap.peek().map(|worst| worst.distance) {
                    Some(worst) if worst.total_cmp(&out.distance).is_eq() => self.tie(out.row),
                    _ => self.tied.clear(),
                }
            }
        }
    }

    fn is_full(&self) -> bool {
        self.heap.len() >= self.count
    }

    /// Whether [`Self::offer`] would turn `distance` away at a look: the
    /// shortlist full and `distance` farther than the farthest kept. A long
    /// scan asks this of most rows, inline, before offering any.
    #[inline(always)]
    fn turns_away(&self, distance: f32) -> bool {
        let distance = or_farthest(distance);
        self.heap.len() >= self.count
            && self
                .heap
                .peek()
                .is_some_and(|worst| distance.total_cmp(&worst.distance).is_gt())
    }

    /// The nearest `count`, those as far as the farthest of them taken in
    /// `ties` order.
    fn estimates(self) -> Vec<(f32, u32)> {
        let Self {
            count,
            heap,
            tied,
            ties,
        } = self;
        let farthest = heap.peek().map(|worst| worst.distance);
        let mut estimates: Vec<(f32, u32)> = heap
            .into_iter()
            .map(|candidate| (candidate.distance, candidate.row))
            .collect();
        estimates.extend(
            farthest
                .into_iter()
                .flat_map(|distance| tied.iter().map(move |row| (distance, *row))),
        );
        nearest(&mut estimates, count, |a, b| ties(*a, *b));
        estimates
    }
}

fn clusters_for(count: usize) -> usize {
    ((count as f64).sqrt() as usize).clamp(1, MOST_CLUSTERS.min(count))
}

/// Which of `centroids` (laid out `width` values each) is nearest `vector`
/// by Euclidean distance.
fn nearest_centroid(centroids: &[f32], width: usize, vector: &[f32]) -> usize {
    centroids
        .chunks_exact(width)
        .map(|centroid| squared_l2(centroid, vector))
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0, |(index, _)| index)
}

/// Lloyd's k-means over `training`, seeded with `k` of them spread evenly
/// through the sample, which is in row order.
fn kmeans(training: &[Vec<f32>], k: usize, width: usize, random: &mut Random) -> Vec<f32> {
    let mut centroids: Vec<f32> = training
        .iter()
        .step_by(training.len() / k)
        .take(k)
        .flat_map(|vector| vector.iter().copied())
        .collect();
    let mut assigned = vec![usize::MAX; training.len()];
    for _ in 0..ITERATIONS {
        let next: Vec<usize> = training
            .par_iter()
            .map(|vector| nearest_centroid(&centroids, width, vector))
            .collect();
        if next == assigned {
            break;
        }
        assigned = next;
        let mut sums = vec![0.0f32; k * width];
        let mut counts = vec![0usize; k];
        for (vector, cluster) in training.iter().zip(&assigned) {
            counts[*cluster] += 1;
            sums[cluster * width..][..width]
                .iter_mut()
                .zip(vector)
                .for_each(|(sum, value)| *sum += value);
        }
        for (cluster, count) in counts.iter().enumerate() {
            let centroid = &mut centroids[cluster * width..][..width];
            match count {
                0 => centroid.copy_from_slice(&training[random.below(training.len())]),
                _ => centroid
                    .iter_mut()
                    .zip(&sums[cluster * width..][..width])
                    .for_each(|(c, sum)| *c = sum / *count as f32),
            }
        }
    }
    centroids
}

impl ClusteredIndex {
    /// Index `vectors`, which must be in ascending row order and all of one
    /// length, for search under `metric`. Returns `None` when there is
    /// nothing to index. Building clusters the vectors, codes each, and
    /// settles how many clusters a query scans; it takes a while for many
    /// vectors and is meant for a background thread.
    pub fn build(metric: Distance, vectors: &[(u32, &[f32])]) -> Option<Self> {
        let dimensions = vectors.first()?.1.len();
        debug_assert!(vectors.windows(2).all(|pair| pair[0].0 < pair[1].0));
        debug_assert!(vectors.iter().all(|(_, vector)| vector.len() == dimensions));
        // Clusters, and how many a query scans, are drawn from the rows that
        // can be measured alone; any other, stored before lengths were
        // limited, would drag them far away. The rows that can are all the
        // index is shaped by, so it comes out the same with or without the
        // others.
        let measured: Vec<(u32, &[f32])> = vectors
            .iter()
            .copied()
            .filter(|(_, vector)| measurable(metric, vector))
            .collect();
        let mut random = Random(measured.len() as u64 ^ ((dimensions as u64) << 32));
        let rotation = Rotation::new(dimensions, random.next());
        let width = rotation.width;
        let rotated = |vector: &[f32]| rotation.apply(&normalised(metric, vector));

        let k = clusters_for(measured.len().max(1));
        let training: Vec<Vec<f32>> = random
            .sample(measured.len(), k * TRAINING_PER_CLUSTER)
            .into_par_iter()
            .map(|at| rotated(measured[at].1))
            .collect();
        let centroids = match training.is_empty() {
            true => vec![0.0; k * width],
            false => kmeans(&training, k, width, &mut random),
        };

        let scale = 1.0 / (width as f32).sqrt();
        let coded: Vec<(usize, Vec<u64>, Factor)> = vectors
            .par_iter()
            .map(|(_, vector)| {
                // Defence in depth: its length, worked out again from its
                // factors, would set such a row apart even if it were coded.
                if !measurable(metric, vector) {
                    return (0, vec![0; width / 64], Factor::UNMEASURED);
                }
                let rotated = rotated(vector);
                let cluster = nearest_centroid(&centroids, width, &rotated);
                let centroid = &centroids[cluster * width..][..width];
                let residual: Vec<f32> = rotated.iter().zip(centroid).map(|(v, c)| v - c).collect();
                let norm = dot(&residual, &residual).sqrt();
                let code: Vec<u64> = residual
                    .chunks_exact(64)
                    .map(|values| {
                        values.iter().enumerate().fold(0u64, |word, (bit, value)| {
                            word | (u64::from(*value > 0.0) << bit)
                        })
                    })
                    .collect();
                let factor = match norm > 0.0 {
                    true => Factor {
                        norm,
                        align: residual.iter().map(|value| value.abs()).sum::<f32>() * scale / norm,
                        along: dot(&residual, centroid) / norm,
                    },
                    false => Factor {
                        norm: 0.0,
                        align: 1.0,
                        along: 0.0,
                    },
                };
                (cluster, code, factor)
            })
            .collect();

        let mut members: Vec<(Vec<u32>, Vec<u64>, Vec<Factor>)> =
            (0..k).map(|_| Default::default()).collect();
        let mut at = Vec::with_capacity(vectors.len());
        for ((row, _), (cluster, code, factor)) in vectors.iter().zip(coded) {
            let (rows, codes, factors) = &mut members[cluster];
            at.push((cluster as u32, rows.len() as u32));
            rows.push(*row);
            codes.extend(code);
            factors.push(factor);
        }
        let clusters: Vec<Cluster> = members
            .into_iter()
            .map(|(rows, codes, factors)| Cluster {
                rows,
                blocks: Blocks::new(&codes, width / 64),
                factors,
            })
            .collect();
        let mut index = Self {
            metric,
            dimensions,
            unmeasured: unmeasured(&clusters, &centroids, width),
            rotation,
            centroids,
            clusters,
            rows: vectors.iter().map(|(row, _)| *row).collect(),
            at,
            probes: 1,
            curve: None,
        };
        let samples = random.sample(measured.len(), TUNING_QUERIES);
        let ranks = index.sample_ranks(&measured, &samples);
        index.curve = Curve::of(
            Source::Rows,
            samples.len() as u32,
            index.clusters.len() as u32,
            ranks,
        );
        // Every cluster a sample's nearest rows lay in, as the curve says;
        // also what a node that knows nothing of curves scans.
        index.probes = index.curve.as_ref().map_or(1, Curve::probes);
        Some(index)
    }

    /// How many clusters a query scans.
    pub fn probes(&self) -> usize {
        self.probes
    }

    /// Whether this is an index of just these vectors' rows, of their
    /// dimensions, measured by `metric`, or else in what it is not: what
    /// one read back from a file must be before it is trusted with them.
    /// An index of other rows would name rows that are not there, and one
    /// of other dimensions or another metric would answer nothing, or
    /// wrongly.
    pub fn fit(&self, metric: Distance, vectors: &[(u32, &[f32])]) -> Result<(), Unfit> {
        let dimensions = vectors.first().map(|(_, vector)| vector.len());
        if self.metric != metric {
            Err(Unfit::Metric)
        } else if dimensions.is_some_and(|dimensions| dimensions != self.dimensions) {
            Err(Unfit::Dimensions)
        } else if !self.rows.iter().eq(vectors.iter().map(|(row, _)| row)) {
            Err(Unfit::Rows)
        } else {
            Ok(())
        }
    }

    /// How many clusters the rows are grouped in.
    pub fn clusters(&self) -> usize {
        self.clusters.len()
    }

    /// The curve of the rows the index was tuned on when it was built:
    /// none for an index read back from a file, whose curve the segment
    /// keeps.
    pub fn rows_curve(&self) -> Option<Curve> {
        self.curve.clone()
    }

    /// Each of `samples`' nearest [`TUNING_NEIGHBOURS`] of `vectors`, the
    /// sample itself left out, ties in row order: in one pass over the
    /// rows for all samples at once, block by block on the rayon pool, so
    /// that the rows are read once however many samples there are.
    fn nearest_of_samples(&self, vectors: &[(u32, &[f32])], samples: &[usize]) -> Vec<Vec<u32>> {
        let asked: Vec<(u32, Scorer<'_>)> = samples
            .iter()
            .map(|at| (vectors[*at].0, Scorer::new(self.metric, vectors[*at].1)))
            .collect();
        let shortlists = |block: &[(u32, &[f32])]| -> Vec<Vec<(f32, u32)>> {
            let mut nearest: Vec<Nearest<'_>> = asked
                .iter()
                .map(|_| Nearest::new(TUNING_NEIGHBOURS, &in_row_order))
                .collect();
            for (row, vector) in block {
                for ((sample, scorer), nearest) in asked.iter().zip(&mut nearest) {
                    if row != sample {
                        let distance = scorer.distance(vector);
                        if !nearest.turns_away(distance) {
                            nearest.offer(distance, *row);
                        }
                    }
                }
            }
            nearest.into_iter().map(Nearest::estimates).collect()
        };
        let merged = vectors.par_chunks(TUNING_BLOCK).map(shortlists).reduce(
            || vec![Vec::new(); asked.len()],
            |a, b| {
                a.into_iter()
                    .zip(b)
                    .map(|(a, b)| {
                        let mut nearest = Nearest::new(TUNING_NEIGHBOURS, &in_row_order);
                        for (distance, row) in a.into_iter().chain(b) {
                            nearest.offer(distance, row);
                        }
                        nearest.estimates()
                    })
                    .collect()
            },
        );
        merged
            .into_iter()
            .map(|mut found| {
                found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                found.truncate(TUNING_NEIGHBOURS);
                found.into_iter().map(|(_, row)| row).collect()
            })
            .collect()
    }

    /// The rank, in each sample's order of clusters, of the cluster each of
    /// its nearest rows lies in.
    fn sample_ranks(&self, vectors: &[(u32, &[f32])], samples: &[usize]) -> Vec<u32> {
        let nearest = self.nearest_of_samples(vectors, samples);
        samples
            .par_iter()
            .zip(nearest)
            .flat_map_iter(|(at, rows)| {
                let mut rank = vec![0u32; self.clusters.len()];
                for (place, (_, cluster)) in self
                    .order(&self.prepared(vectors[*at].1))
                    .into_iter()
                    .enumerate()
                {
                    rank[cluster] = place as u32;
                }
                rows.into_iter()
                    .filter_map(|row| self.position(row))
                    .map(|position| rank[self.at[position].0 as usize])
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn prepared(&self, query: &[f32]) -> Vec<f32> {
        self.rotation.apply(&normalised(self.metric, query))
    }

    /// The clusters nearest the prepared query, nearest first.
    fn order(&self, query: &[f32]) -> Vec<(f32, usize)> {
        let mut order: Vec<(f32, usize)> = self
            .centroids
            .chunks_exact(self.rotation.width)
            .enumerate()
            .map(|(cluster, centroid)| {
                // As rows were assigned to clusters: by how far the
                // centroid is, not by how aligned, so that a cluster whose
                // members lie in several directions, and so whose centroid
                // is short, is not passed over. Only for dot products,
                // where length is the point, is alignment the measure.
                let distance = match self.metric {
                    Distance::L2 | Distance::Cosine => squared_l2(query, centroid),
                    Distance::Dot => -dot(query, centroid),
                };
                (distance, cluster)
            })
            .collect();
        order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        order
    }

    fn position(&self, row: u32) -> Option<usize> {
        self.rows.binary_search(&row).ok()
    }

    fn centroid(&self, cluster: usize) -> &[f32] {
        &self.centroids[cluster * self.rotation.width..][..self.rotation.width]
    }
}

impl VectorIndex for ClusteredIndex {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn unmeasured(&self, only: Option<&RoaringBitmap>, skipped: &[bool]) -> Vec<u32> {
        allowed(&self.unmeasured, only, skipped)
    }

    /// Each estimate is a distance under the index's metric, comparable
    /// with the estimates of other clustered indexes of vectors measured
    /// the same way.
    fn estimates(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
        ties: Ties<'_>,
    ) -> Vec<(f32, u32)> {
        let probing = Probing {
            probes: self.probes,
            parts: 1,
        };
        self.estimates_split(query, count, only, skipped, ties, probing)
    }
}

impl ClusteredIndex {
    /// As [`VectorIndex::estimates`], scanning at least `probes` clusters
    /// rather than the count the index was built with (a segment's probe
    /// curve says how many: see [`crate::curve`]), the scan split into up
    /// to `parts` runs on the rayon pool: the same nearest `count`,
    /// whatever `parts` is, as [`Self::scanned`] says.
    pub fn estimates_split(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
        ties: Ties<'_>,
        probing: Probing,
    ) -> Vec<(f32, u32)> {
        let Probing { probes, parts } = probing;
        if count == 0 || query.len() != self.dimensions {
            return Vec::new();
        }
        // A row with no estimate is never offered: the caller scores it.
        let kept = |row: u32| {
            !skipped.get(row as usize).copied().unwrap_or(false)
                && self.unmeasured.binary_search(&row).is_err()
        };
        let mut nearest = Nearest::new(count, ties);
        // A query as far from every row is estimated alike for each, in
        // whatever cluster, and the caller's order decides.
        if as_far_from_every_vector(self.metric, query) {
            let mut offer = |row: u32| nearest.offer(0.0, row);
            match only {
                Some(only) => only
                    .iter()
                    .filter(|row| kept(*row) && self.position(*row).is_some())
                    .for_each(&mut offer),
                None => self
                    .rows
                    .iter()
                    .copied()
                    .filter(|row| kept(*row))
                    .for_each(&mut offer),
            }
            return nearest.estimates();
        }
        let query = self.prepared(query);
        let probes = probes.clamp(1, self.clusters.len());
        let per_probe = self.rows.len().div_ceil(self.clusters.len()) * probes;
        match only {
            // Fewer rows allowed than a scan would visit: visit just those.
            Some(only) if (only.len() as usize) <= per_probe => {
                self.visited(&query, count, only, skipped, ties)
            }
            _ => self.scanned(
                &query,
                count,
                only,
                skipped,
                ties,
                Probing { probes, parts },
            ),
        }
    }

    /// Whether `row` may be offered: not skipped, and with an estimate.
    fn kept(&self, row: u32, skipped: &[bool]) -> bool {
        !skipped.get(row as usize).copied().unwrap_or(false)
            && self.unmeasured.binary_search(&row).is_err()
    }

    /// The `count` nearest of the rows `only` allows, estimated a cluster
    /// at a time: each cluster met readied once, and each block holding
    /// an allowed row weighed whole, as a scan weighs it, reading its
    /// bytes in order rather than each row's across the block. `query` is
    /// prepared. The rows are offered by cluster rather than in row order,
    /// so the nearest `count` are the same and only offers' order differs.
    fn visited(
        &self,
        query: &[f32],
        count: usize,
        only: &RoaringBitmap,
        skipped: &[bool],
        ties: Ties<'_>,
    ) -> Vec<(f32, u32)> {
        let mut nearest = Nearest::new(count, ties);
        // The allowed rows' places, sorted by cluster by counting, each
        // cluster's kept in the order they were met.
        let mut met: Vec<(u32, u32, u32)> = Vec::new();
        let mut starts = vec![0usize; self.clusters.len() + 1];
        for row in only.iter().filter(|row| self.kept(*row, skipped)) {
            let Some(position) = self.position(row) else {
                continue;
            };
            let (cluster, index) = self.at[position];
            starts[cluster as usize + 1] += 1;
            met.push((cluster, index, row));
        }
        for cluster in 0..self.clusters.len() {
            starts[cluster + 1] += starts[cluster];
        }
        let mut sorted = vec![(0u32, 0u32); met.len()];
        let mut next = starts.clone();
        for (cluster, index, row) in met {
            sorted[next[cluster as usize]] = (index, row);
            next[cluster as usize] += 1;
        }
        let weigh = chosen().1;
        let mut weighed = [0u32; BLOCK];
        let mut space = Readying::default();
        let mut tables = Tables::default();
        for cluster in 0..self.clusters.len() {
            let rows = &sorted[starts[cluster]..starts[cluster + 1]];
            if rows.is_empty() {
                continue;
            }
            let probe = Probe::readied(query, self.centroid(cluster), &mut space);
            tables.fill(&space.levels);
            let members = &self.clusters[cluster];
            let mut weighed_block = usize::MAX;
            for (index, row) in rows {
                let index = *index as usize;
                if index / BLOCK != weighed_block {
                    weighed_block = index / BLOCK;
                    weigh(members.blocks.block(weighed_block), &tables, &mut weighed);
                }
                let distance = probe.distance_of(
                    self.metric,
                    members.blocks.ones(index),
                    weighed[index % BLOCK],
                    &members.factors[index],
                );
                if !nearest.turns_away(distance) {
                    nearest.offer(distance, *row);
                }
            }
        }
        nearest.estimates()
    }

    /// The `count` nearest allowed rows of the clusters nearest the query,
    /// at least [`Self::probes`] of them and more while too few are found,
    /// a block of rows weighed at a time. `query` is prepared. With more
    /// than one of `parts`, the clusters it must probe are split into as
    /// many runs, scanned on the read pool ([`crate::pool::reads`]) each
    /// with its own shortlist,
    /// and the shortlists merged; what it finds is the same nearest
    /// `count`, as the rows offered are, and only offers' order differs.
    fn scanned(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
        ties: Ties<'_>,
        probing: Probing,
    ) -> Vec<(f32, u32)> {
        let Probing { probes, parts } = probing;
        let order = self.order(query);
        let probes = probes.min(order.len());
        let mut scan = Scan::new(self, query, only, skipped);
        let mut nearest = Nearest::new(count, ties);
        let runs = parts.min(probes);
        if runs > 1 {
            let scan_run = |run: usize| {
                let clusters = &order[probes * run / runs..probes * (run + 1) / runs];
                let mut scan = Scan::new(self, query, only, skipped);
                let mut nearest = Nearest::new(count, ties);
                for (_, cluster) in clusters {
                    scan.cluster(*cluster, &mut nearest);
                }
                nearest.estimates()
            };
            let shortlists: Vec<Vec<(f32, u32)>> =
                crate::pool::reads().install(|| (0..runs).into_par_iter().map(scan_run).collect());
            for (distance, row) in shortlists.into_iter().flatten() {
                if !nearest.turns_away(distance) {
                    nearest.offer(distance, row);
                }
            }
        } else {
            for (_, cluster) in &order[..probes] {
                scan.cluster(*cluster, &mut nearest);
            }
        }
        // On past the clusters it must probe, one at a time, while too few
        // rows are found.
        for (_, cluster) in &order[probes..] {
            if nearest.is_full() {
                break;
            }
            scan.cluster(*cluster, &mut nearest);
        }
        nearest.estimates()
    }
}

/// One scan's way through clusters: what it reuses from one to the next.
struct Scan<'a> {
    index: &'a ClusteredIndex,
    query: &'a [f32],
    only: Option<&'a RoaringBitmap>,
    skipped: &'a [bool],
    weigh: crate::fastscan::Weigh,
    weighed: [u32; BLOCK],
    offered: [bool; BLOCK],
    space: Readying,
    tables: Tables,
}

impl<'a> Scan<'a> {
    fn new(
        index: &'a ClusteredIndex,
        query: &'a [f32],
        only: Option<&'a RoaringBitmap>,
        skipped: &'a [bool],
    ) -> Self {
        Self {
            index,
            query,
            only,
            skipped,
            weigh: chosen().1,
            weighed: [0; BLOCK],
            offered: [false; BLOCK],
            space: Readying::default(),
            tables: Tables::default(),
        }
    }

    /// Every allowed row of `cluster` estimated and offered to `nearest`,
    /// a block at a time, in the cluster's order.
    fn cluster(&mut self, cluster: usize, nearest: &mut Nearest<'_>) {
        let index = self.index;
        let probe = Probe::readied(self.query, index.centroid(cluster), &mut self.space);
        self.tables.fill(&self.space.levels);
        let members = &index.clusters[cluster];
        for block in 0..members.blocks.count() {
            let rows = &members.rows[block * BLOCK..members.rows.len().min((block + 1) * BLOCK)];
            let mut any = false;
            for (offer, row) in self.offered.iter_mut().zip(rows) {
                *offer = index.kept(*row, self.skipped)
                    && self.only.is_none_or(|only| only.contains(*row));
                any |= *offer;
            }
            if !any {
                continue;
            }
            (self.weigh)(members.blocks.block(block), &self.tables, &mut self.weighed);
            for (lane, row) in rows.iter().enumerate() {
                if self.offered[lane] {
                    let at = block * BLOCK + lane;
                    let distance = probe.distance_of(
                        index.metric,
                        members.blocks.ones(at),
                        self.weighed[lane],
                        &members.factors[at],
                    );
                    if !nearest.turns_away(distance) {
                        nearest.offer(distance, *row);
                    }
                }
            }
        }
    }
}

/// Columns on the wire as little-endian bytes.
mod le {
    use serde::{Deserializer, Serializer};

    pub trait Word: Sized {
        const SIZE: usize;
        fn to_le(self, out: &mut Vec<u8>);
        fn from_le(bytes: &[u8]) -> Self;
    }

    macro_rules! word {
        ($($t:ty),*) => {$(
            impl Word for $t {
                const SIZE: usize = std::mem::size_of::<$t>();
                fn to_le(self, out: &mut Vec<u8>) {
                    out.extend_from_slice(&self.to_le_bytes());
                }
                fn from_le(bytes: &[u8]) -> Self {
                    <$t>::from_le_bytes(bytes.try_into().expect("a whole word"))
                }
            }
        )*};
    }
    word!(u32, u64, f32);

    pub fn serialize<T: Word + Copy, S: Serializer>(
        words: &[T],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::with_capacity(words.len() * T::SIZE);
        words.iter().for_each(|word| word.to_le(&mut bytes));
        serde_bytes::serialize(&bytes, serializer)
    }

    pub fn deserialize<'de, T: Word, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<T>, D::Error> {
        let bytes: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        if !bytes.len().is_multiple_of(T::SIZE) {
            return Err(serde::de::Error::custom("a column is not whole words"));
        }
        Ok(bytes.chunks_exact(T::SIZE).map(T::from_le).collect())
    }
}

#[cfg(test)]
mod tests {

    /// The scan's estimate as it was first written, word by word in five
    /// passes: what any faster one must equal to the bit.
    fn inner_as_first_written(probe: &Probe, code: &[u64], factor: &Factor) -> f32 {
        let ones: u32 = code.iter().map(|word| word.count_ones()).sum();
        let weighted: u32 = (0..QUERY_BITS)
            .map(|bit| {
                let plane = &probe.planes[bit as usize * probe.words..][..probe.words];
                let count: u32 = code
                    .iter()
                    .zip(plane)
                    .map(|(c, p)| (c & p).count_ones())
                    .sum();
                count << bit
            })
            .sum();
        let set = probe.lower * ones as f32 + probe.delta * weighted as f32;
        (2.0 * set - probe.sum) * probe.scale / factor.align
    }

    /// A query readied for a cluster as first written: what any faster
    /// readying must equal to the bit.
    fn probe_as_first_written(query: &[f32], centroid: &[f32]) -> (Vec<u64>, [u32; 6]) {
        let residual: Vec<f32> = query.iter().zip(centroid).map(|(q, c)| q - c).collect();
        let (mut lower, mut upper) = (f32::INFINITY, f32::NEG_INFINITY);
        for value in &residual {
            lower = lower.min(*value);
            upper = upper.max(*value);
        }
        let levels = ((1u32 << QUERY_BITS) - 1) as f32;
        let delta = (upper - lower) / levels;
        let words = residual.len() / 64;
        let mut planes = vec![0u64; words * QUERY_BITS as usize];
        for (i, value) in residual.iter().enumerate() {
            let level = match delta > 0.0 {
                true => ((value - lower) / delta).round().clamp(0.0, levels) as u32,
                false => 0,
            };
            for bit in 0..QUERY_BITS {
                if level >> bit & 1 == 1 {
                    planes[bit as usize * words + i / 64] |= 1 << (i % 64);
                }
            }
        }
        let sum: f32 = residual.iter().sum();
        let scalars = [
            lower.to_bits(),
            delta.to_bits(),
            sum.to_bits(),
            dot(&residual, &residual).to_bits(),
            dot(query, centroid).to_bits(),
            (1.0 / (residual.len() as f32).sqrt()).to_bits(),
        ];
        (planes, scalars)
    }

    proptest::proptest! {
        /// A query is readied for each cluster as it was, to the bit.
        #[test]
        fn a_query_is_readied_for_a_cluster_as_first_written(
            seed in 0u64..u64::MAX,
            words in proptest::prop_oneof![1usize..20, 32usize..=48],
            flat in proptest::bool::weighted(0.1),
        ) {
            let mut random = Random(seed | 1);
            let width = words * 64;
            let mut unit = || (random.next() >> 40) as f32 / (1u64 << 24) as f32;
            let query: Vec<f32> = (0..width).map(|_| unit() * 2.0 - 1.0).collect();
            // A query on its centroid leaves nothing to quantize.
            let centroid: Vec<f32> = match flat {
                true => query.clone(),
                false => (0..width).map(|_| unit() - 0.5).collect(),
            };
            let probe = Probe::new(&query, &centroid);
            let (planes, scalars) = probe_as_first_written(&query, &centroid);
            proptest::prop_assert_eq!(&probe.planes, &planes);
            let got = [
                probe.lower.to_bits(),
                probe.delta.to_bits(),
                probe.sum.to_bits(),
                probe.norm_squared.to_bits(),
                probe.centroid_dot.to_bits(),
                probe.scale.to_bits(),
            ];
            proptest::prop_assert_eq!(got, scalars);
        }
    }

    proptest::proptest! {
        /// Speed is the scan's only change: every code is estimated as it
        /// was, to the bit.
        #[test]
        fn the_scan_estimates_every_code_as_first_written(
            seed in 0u64..u64::MAX,
            words in proptest::prop_oneof![1usize..20, 32usize..=48],
            align in 0.05f32..1.0,
        ) {
            let mut random = Random(seed | 1);
            let width = words * 64;
            let mut unit = || (random.next() >> 40) as f32 / (1u64 << 24) as f32;
            let query: Vec<f32> = (0..width).map(|_| unit() * 2.0 - 1.0).collect();
            let centroid: Vec<f32> = (0..width).map(|_| unit() - 0.5).collect();
            let probe = Probe::new(&query, &centroid);
            let factor = Factor { norm: 1.0, align, along: 0.0 };
            for _ in 0..8 {
                let code: Vec<u64> = (0..words).map(|_| random.next()).collect();
                let fast = probe.inner(&code, &factor);
                let first = inner_as_first_written(&probe, &code, &factor);
                proptest::prop_assert_eq!(fast.to_bits(), first.to_bits());
            }
        }
    }

    /// The scan as it was before codes were laid out in blocks: every
    /// cluster's rows estimated code by code, in the same order. What the
    /// scan of blocks must equal, estimate for estimate and in order.
    fn estimates_code_by_code(
        index: &ClusteredIndex,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
    ) -> Vec<(f32, u32)> {
        let kept = |row: u32| {
            !skipped.get(row as usize).copied().unwrap_or(false)
                && index.unmeasured.binary_search(&row).is_err()
        };
        let mut nearest = Nearest::new(count, &in_row_order);
        let query = index.prepared(query);
        let words = index.rotation.width / 64;
        let mut code = vec![0u64; words];
        let allowed = |row: u32| kept(row) && only.is_none_or(|only| only.contains(row));
        for (probed, (_, cluster)) in index.order(&query).into_iter().enumerate() {
            if probed >= index.probes && nearest.is_full() {
                break;
            }
            let probe = Probe::new(&query, index.centroid(cluster));
            let members = &index.clusters[cluster];
            for (at, row) in members.rows.iter().enumerate() {
                if allowed(*row) {
                    members.blocks.code(at, &mut code);
                    nearest.offer(
                        probe.distance(index.metric, &code, &members.factors[at]),
                        *row,
                    );
                }
            }
        }
        nearest.estimates()
    }

    /// A level is what rounding gave, for every distance a residual can be
    /// above the lowest: each `f32` from 0 to past 16 near a half step, and
    /// the edges.
    #[test]
    fn a_level_is_the_rounded_distance_as_first_written() {
        let first = |x: f32| x.round().clamp(0.0, 15.0) as u8;
        let mut x = 0.0f32;
        while x < 17.0 {
            assert_eq!(level_of(x), first(x), "{x}");
            for near in [
                x.next_down(),
                x.next_up(),
                x + 0.5,
                (x + 0.5).next_down(),
                (x + 0.5).next_up(),
            ] {
                if near >= 0.0 {
                    assert_eq!(level_of(near), first(near), "{near}");
                }
            }
            x += 1.0 / 64.0;
        }
        for edge in [
            0.0,
            f32::MIN_POSITIVE,
            0.49999997,
            0.5,
            14.5,
            15.0,
            15.000001,
            1e9,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ] {
            assert_eq!(level_of(edge), first(edge), "{edge}");
        }
    }

    proptest::proptest! {
        /// What a shortlist keeps, as every caller cuts it, does not depend
        /// on the order rows are offered in: the nearest `count`, those
        /// alike in row order, whether they come in row order, the reverse
        /// or shuffled, and however many tie. The scan, the visit and a
        /// split scan each offer rows in their own order and rely on it.
        #[test]
        fn a_shortlists_cut_is_the_same_whatever_order_rows_come_in(
            seed in 1u64..u64::MAX,
            rows in 1usize..400,
            distinct in 1u64..12,
            count in 1usize..60,
        ) {
            let mut noise = Noise(seed);
            let offered: Vec<(f32, u32)> = (0..rows as u32)
                .map(|row| (((noise.unit() + 0.5) * distinct as f32).floor(), row))
                .collect();
            let mut want = offered.clone();
            want.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            want.truncate(count);
            let want: Vec<(u32, u32)> = want.iter().map(|(d, row)| (d.to_bits(), *row)).collect();
            let mut shuffled = offered.clone();
            for at in (1..shuffled.len()).rev() {
                let other = ((noise.unit() + 0.5) * (at + 1) as f32) as usize;
                shuffled.swap(at, other.min(at));
            }
            let reversed: Vec<(f32, u32)> = offered.iter().rev().copied().collect();
            for (how, order) in [("in row order", &offered), ("reversed", &reversed), ("shuffled", &shuffled)] {
                let mut nearest = Nearest::new(count, &in_row_order);
                for (distance, row) in order {
                    nearest.offer(*distance, *row);
                }
                proptest::prop_assert_eq!(cut(nearest.estimates(), count), want.clone(), "{}", how);
            }
        }
    }

    /// What a shortlist of `count` says, as every caller reads it: its
    /// nearest `count` by estimate, those alike in row order.
    fn cut(mut found: Vec<(f32, u32)>, count: usize) -> Vec<(u32, u32)> {
        found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        found.truncate(count);
        found.iter().map(|(d, row)| (d.to_bits(), *row)).collect()
    }

    /// The visit of allowed rows as it was: each estimated from its code.
    fn visited_code_by_code(
        index: &ClusteredIndex,
        query: &[f32],
        count: usize,
        only: &RoaringBitmap,
        skipped: &[bool],
    ) -> Vec<(f32, u32)> {
        let kept = |row: u32| {
            !skipped.get(row as usize).copied().unwrap_or(false)
                && index.unmeasured.binary_search(&row).is_err()
        };
        let mut nearest = Nearest::new(count, &in_row_order);
        let query = index.prepared(query);
        let mut code = vec![0u64; index.rotation.width / 64];
        for row in only.iter().filter(|row| kept(*row)) {
            let Some(position) = index.position(row) else {
                continue;
            };
            let (cluster, at) = index.at[position];
            let probe = Probe::new(&query, index.centroid(cluster as usize));
            let members = &index.clusters[cluster as usize];
            members.blocks.code(at as usize, &mut code);
            nearest.offer(
                probe.distance(index.metric, &code, &members.factors[at as usize]),
                row,
            );
        }
        nearest.estimates()
    }

    /// The scan of the clusters a query probes, whatever share of rows is
    /// allowed.
    fn scan_only(
        index: &ClusteredIndex,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
    ) -> Vec<(f32, u32)> {
        index.scanned(
            &index.prepared(query),
            count,
            only,
            skipped,
            &in_row_order,
            Probing {
                probes: index.probes,
                parts: 1,
            },
        )
    }

    /// Each sample row's nearest rows, the row itself left out, found
    /// sample by sample over every row as tuning first found them: what the
    /// one pass over the rows for all samples at once must equal.
    fn neighbours_sample_by_sample(
        index: &ClusteredIndex,
        vectors: &[(u32, &[f32])],
        samples: &[usize],
    ) -> Vec<Vec<u32>> {
        samples
            .iter()
            .map(|at| {
                let (row, query) = vectors[*at];
                let scorer = Scorer::new(index.metric, query);
                let mut all: Vec<(f32, u32)> = vectors
                    .iter()
                    .filter(|(other, _)| *other != row)
                    .map(|(other, vector)| (scorer.distance(vector), *other))
                    .collect();
                all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                all.truncate(TUNING_NEIGHBOURS);
                all.into_iter().map(|(_, row)| row).collect()
            })
            .collect()
    }

    /// Tuning runs on a pool of one thread of its own, apart from rayon's
    /// global pool, where flushes and merges build and where it would take
    /// every core for tens of seconds a segment: it finishes while every
    /// global thread is held busy (recall note, R2.6 and R9.5).
    #[test]
    fn tuning_runs_on_one_thread_of_its_own() {
        assert_eq!(crate::pool::tuning().current_num_threads(), 1);
        let vectors = grouped(5, 3_000, 16, 12, 0.3);
        let index = indexed(Distance::Cosine, &vectors);
        let numbered: Vec<(u32, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(row, v)| (row as u32 * 2, v.as_slice()))
            .collect();
        let (release, held) = std::sync::mpsc::channel::<()>();
        let held = std::sync::Arc::new(std::sync::Mutex::new(held));
        let (started, busy) = std::sync::mpsc::channel::<()>();
        for _ in 0..rayon::current_num_threads() {
            let (held, started) = (held.clone(), started.clone());
            rayon::spawn(move || {
                started.send(()).unwrap();
                let _ = held.lock().unwrap().recv();
            });
        }
        for _ in 0..rayon::current_num_threads() {
            busy.recv().unwrap();
        }
        let samples: Vec<usize> = (0..3_000).step_by(11).collect();
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(|| done.send(index.sample_ranks(&numbered, &samples).len()).unwrap());
            let answer = finished.recv_timeout(std::time::Duration::from_secs(30));
            for _ in 0..rayon::current_num_threads() {
                release.send(()).unwrap();
            }
            assert_eq!(answer, Ok(samples.len() * TUNING_NEIGHBOURS));
        });
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(16))]
        /// Tuning finds each sample row's nearest rows in one pass over the
        /// rows for all samples at once, as a pass for each sample finds
        /// them, ties in row order; and a built index's curve is of their
        /// clusters' ranks, from its rows, scanning every cluster any of
        /// them lay in, as the index's own count does.
        #[test]
        fn one_pass_finds_every_samples_nearest_rows_and_the_curve_holds_them_all(
            seed in 1u64..u64::MAX,
            rows in 30usize..1500,
            dimensions in 2usize..48,
            copies in proptest::prop_oneof![3 => proptest::strategy::Just(1usize), 1 => 2usize..4],
            metric in proptest::sample::select(vec![Distance::Cosine, Distance::L2, Distance::Dot]),
        ) {
            let vectors: Vec<Vec<f32>> = {
                let mut noise = Noise(seed);
                let distinct: Vec<Vec<f32>> =
                    (0..rows.div_ceil(copies)).map(|_| noise.vector(dimensions)).collect();
                (0..rows).map(|n| distinct[n % distinct.len()].clone()).collect()
            };
            let index = indexed(metric, &vectors);
            let numbered: Vec<(u32, &[f32])> = vectors
                .iter()
                .enumerate()
                .map(|(row, v)| (row as u32 * 2, v.as_slice()))
                .collect();
            let samples: Vec<usize> = (0..rows).step_by(7).collect();
            let found = index.nearest_of_samples(&numbered, &samples);
            proptest::prop_assert_eq!(found, neighbours_sample_by_sample(&index, &numbered, &samples));

            let curve = index.rows_curve().expect("a built index has a curve of its rows");
            proptest::prop_assert_eq!(curve.source, crate::curve::Source::Rows);
            proptest::prop_assert_eq!(curve.clusters as usize, index.clusters.len());
            proptest::prop_assert_eq!(curve.samples as usize, rows.min(TUNING_QUERIES));
            proptest::prop_assert_eq!(curve.probes(), index.probes(), "the index scans as its curve says");
            // Built twice from the same rows, the same curve.
            proptest::prop_assert_eq!(indexed(metric, &vectors).rows_curve(), Some(curve));
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(16))]
        /// Probes given, as a segment's curve gives them, scan as an index
        /// built to scan that many does, clamped to one and to every
        /// cluster.
        #[test]
        fn probes_given_scan_as_an_index_built_with_them(
            seed in 1u64..u64::MAX,
            rows in 40usize..1200,
            dimensions in 2usize..64,
            count in 1usize..200,
            probes in 0usize..80,
            share in proptest::option::of(1u32..100),
        ) {
            let vectors: Vec<Vec<f32>> = {
                let mut noise = Noise(seed);
                (0..rows).map(|_| noise.vector(dimensions)).collect()
            };
            let index = indexed(Distance::Cosine, &vectors);
            let built = ClusteredIndex {
                probes: probes.clamp(1, index.clusters.len()),
                ..index.clone()
            };
            let only: Option<RoaringBitmap> = share.map(|share| {
                (0..rows as u32 * 2)
                    .filter(|row| (row.wrapping_mul(2_654_435_761) >> 7) % 100 < share)
                    .collect()
            });
            let mut noise = Noise(seed.rotate_left(9) | 1);
            for _ in 0..4 {
                let query = noise.vector(dimensions);
                let given = index.estimates_split(&query, count, only.as_ref(), &[], &in_row_order, Probing { probes, parts: 1 });
                let as_built = built.estimates(&query, count, only.as_ref(), &[], &in_row_order);
                proptest::prop_assert_eq!(cut(given, count), cut(as_built, count));
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(24))]
        /// A scan split into parts across threads finds what one scan
        /// finds: the same nearest `count`, to the bit, for any number of
        /// parts, filtered or not, and when too few rows are allowed in the
        /// clusters it must probe and it scans on past them.
        #[test]
        fn a_scan_split_finds_what_one_scan_finds(
            seed in 1u64..u64::MAX,
            rows in 40usize..1500,
            dimensions in 2usize..96,
            metric in proptest::sample::select(vec![Distance::Cosine, Distance::L2, Distance::Dot]),
            count in 1usize..300,
            share in proptest::option::of(proptest::prop_oneof![1u32..5, 5u32..100]),
            parts in 2usize..9,
            copies in proptest::prop_oneof![3 => proptest::strategy::Just(1usize), 1 => 2usize..6],
        ) {
            // With copies, rows of one vector tie, across the runs' merge.
            let vectors: Vec<Vec<f32>> = {
                let mut noise = Noise(seed);
                let distinct: Vec<Vec<f32>> =
                    (0..rows.div_ceil(copies)).map(|_| noise.vector(dimensions)).collect();
                (0..rows).map(|n| distinct[n % distinct.len()].clone()).collect()
            };
            let index = indexed(metric, &vectors);
            let mut noise = Noise(seed.rotate_left(17) | 1);
            let only: Option<RoaringBitmap> = share.map(|share| {
                (0..rows as u32 * 2)
                    .filter(|row| (row.wrapping_mul(2_654_435_761) >> 7) % 100 < share)
                    .collect()
            });
            let skipped: Vec<bool> = (0..rows * 2).map(|row| row % 11 == 5).collect();
            for _ in 0..4 {
                let query = index.prepared(&noise.vector(dimensions));
                let one = index.scanned(&query, count, only.as_ref(), &skipped, &in_row_order, Probing { probes: index.probes, parts: 1 });
                let split = index.scanned(&query, count, only.as_ref(), &skipped, &in_row_order, Probing { probes: index.probes, parts });
                proptest::prop_assert_eq!(cut(split, count), cut(one, count));
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(24))]
        /// Laying the codes out in blocks changes no estimate: a scan of
        /// every cluster it probes gives the estimates the code-by-code
        /// scan gives, to the bit and in the same order, filtered or not,
        /// with rows skipped or not, for every metric, and with rows of
        /// one vector tied.
        #[test]
        fn the_scan_of_blocks_estimates_as_code_by_code(
            seed in 1u64..u64::MAX,
            rows in 40usize..1500,
            dimensions in 2usize..160,
            metric in proptest::sample::select(vec![Distance::Cosine, Distance::L2, Distance::Dot]),
            count in 1usize..300,
            share in proptest::option::of(1u32..100),
            skip in proptest::bool::ANY,
            copies in proptest::prop_oneof![3 => proptest::strategy::Just(1usize), 1 => 2usize..6],
        ) {
            // With copies, rows of one vector are estimated alike, so that
            // ties at the shortlist's cut go through the visit's order of
            // offers, by cluster, against the code-by-code row order.
            let vectors: Vec<Vec<f32>> = {
                let mut noise = Noise(seed);
                let distinct: Vec<Vec<f32>> =
                    (0..rows.div_ceil(copies)).map(|_| noise.vector(dimensions)).collect();
                (0..rows).map(|n| distinct[n % distinct.len()].clone()).collect()
            };
            let index = indexed(metric, &vectors);
            let mut noise = Noise(seed.rotate_left(17) | 1);
            // `indexed` numbers the rows 0, 2, 4, ...
            let only: Option<RoaringBitmap> = share.map(|share| {
                (0..rows as u32 * 2)
                    .filter(|row| (row.wrapping_mul(2_654_435_761) >> 7) % 100 < share)
                    .collect()
            });
            let skipped: Vec<bool> = (0..rows * 2)
                .map(|row| skip && row % 7 == 3)
                .collect();
            for _ in 0..4 {
                let query = noise.vector(dimensions);
                let got = scan_only(&index, &query, count, only.as_ref(), &skipped);
                let want = estimates_code_by_code(&index, &query, count, only.as_ref(), &skipped);
                let bits = |found: &[(f32, u32)]| -> Vec<(u32, u32)> {
                    found.iter().map(|(d, row)| (d.to_bits(), *row)).collect()
                };
                proptest::prop_assert_eq!(bits(&got), bits(&want));
                if let Some(only) = &only {
                    // Offered by cluster, not in row order: the same
                    // nearest `count`, as every caller reads them.
                    let got = index.visited(&index.prepared(&query), count, only, &skipped, &in_row_order);
                    let want = visited_code_by_code(&index, &query, count, only, &skipped);
                    proptest::prop_assert_eq!(cut(got, count), cut(want, count));
                }
            }
        }
    }

    use super::*;
    use crate::vector::distance;
    use vellev_core::SQUARED_LENGTH_UNDER;

    struct Noise(u64);

    impl Noise {
        fn unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }

        fn vector(&mut self, dimensions: usize) -> Vec<f32> {
            (0..dimensions).map(|_| self.unit()).collect()
        }
    }

    /// Vectors in `groups` groups about random centres, with a spread.
    fn grouped(
        seed: u64,
        count: usize,
        dimensions: usize,
        groups: usize,
        spread: f32,
    ) -> Vec<Vec<f32>> {
        let mut noise = Noise(seed);
        let centres: Vec<Vec<f32>> = (0..groups).map(|_| noise.vector(dimensions)).collect();
        (0..count)
            .map(|n| {
                let jitter = noise.vector(dimensions);
                centres[n % groups]
                    .iter()
                    .zip(jitter)
                    .map(|(c, e)| c + spread * e)
                    .collect()
            })
            .collect()
    }

    /// Rows written in a repeating pattern, group after group, must not be
    /// sampled in step with it: a group the sample left out would have no
    /// cluster of its own, and a query near it would find none of it. Nor
    /// may a cluster that holds several groups be passed over for its
    /// shorter centroid.
    #[test]
    fn every_group_of_rows_written_in_turn_is_found() {
        let count = 40_000;
        let k = clusters_for(count);
        let stride = count / (k * TRAINING_PER_CLUSTER);
        // More groups than clusters, as many as a multiple of the spacing
        // an evenly spaced sample would have: it would visit only some.
        for groups in [stride * 100, stride * 128] {
            let vectors = grouped(11, count, 32, groups, 0.3);
            let index = indexed(Distance::Cosine, &vectors);
            let (mut found_all, mut all) = (0, 0);
            for (group, query) in vectors.iter().take(groups).enumerate() {
                // As `indexed` numbers them.
                let members: Vec<u32> = (group..count)
                    .step_by(groups)
                    .map(|at| at as u32 * 2)
                    .collect();
                let shortlist = index.shortlist(query, members.len() * 2, None, &[], &in_row_order);
                let found = members.iter().filter(|row| shortlist.contains(row)).count();
                assert!(
                    found * 2 >= members.len(),
                    "{groups} groups: group {group} found {found} of {}",
                    members.len()
                );
                (found_all, all) = (found_all + found, all + members.len());
            }
            assert!(
                found_all * 100 >= all * 95,
                "{groups} groups: found {found_all} of {all}"
            );
        }
    }

    fn indexed(metric: Distance, vectors: &[Vec<f32>]) -> ClusteredIndex {
        let rows: Vec<(u32, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(row, v)| (row as u32 * 2, v.as_slice()))
            .collect();
        ClusteredIndex::build(metric, &rows).unwrap()
    }

    /// Of the true nearest ten of every twentieth vector, how many a
    /// shortlist of `count` holds, out of `total`.
    fn recall(
        index: &ClusteredIndex,
        metric: Distance,
        vectors: &[Vec<f32>],
        count: usize,
    ) -> (usize, usize) {
        let (mut found, mut total) = (0, 0);
        for query in vectors.iter().step_by(vectors.len() / 20) {
            let mut exact: Vec<(f32, u32)> = vectors
                .iter()
                .enumerate()
                .map(|(row, v)| (distance(metric, query, v), row as u32 * 2))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let shortlist = index.shortlist(query, count, None, &[], &in_row_order);
            found += exact[..10]
                .iter()
                .filter(|(_, row)| shortlist.contains(row))
                .count();
            total += 10;
        }
        (found, total)
    }

    /// Vectors with no group structure and lengths spread over a range.
    fn scattered(seed: u64, count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        let mut noise = Noise(seed);
        (0..count)
            .map(|_| {
                let scale = 0.5 + 1.5 * (noise.unit() + 0.5);
                noise
                    .vector(dimensions)
                    .into_iter()
                    .map(|x| x * scale)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_estimated_inner_product_is_unbiased_and_close() {
        let vectors = scattered(8, 3000, 256);
        let index = indexed(Distance::L2, &vectors);
        let query = index.prepared(&scattered(9, 1, 256)[0]);
        let (mut cross, mut square, mut error, mut scale) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (cluster, members) in index.clusters.iter().enumerate() {
            let centroid = index.centroid(cluster);
            let probe = Probe::new(&query, centroid);
            let residual: Vec<f32> = query.iter().zip(centroid).map(|(q, c)| q - c).collect();
            let reach = f64::from(dot(&residual, &residual));
            for (i, row) in members.rows.iter().enumerate() {
                let rotated = index.prepared(&vectors[*row as usize / 2]);
                let from: Vec<f32> = rotated.iter().zip(centroid).map(|(v, c)| v - c).collect();
                let norm = dot(&from, &from).sqrt();
                if norm == 0.0 {
                    continue;
                }
                let exact = f64::from(dot(&from, &residual) / norm);
                let mut code = vec![0u64; members.blocks.words()];
                members.blocks.code(i, &mut code);
                let estimated = f64::from(probe.inner(&code, &members.factors[i]));
                cross += estimated * exact;
                square += exact * exact;
                error += (estimated - exact) * (estimated - exact);
                scale += reach;
            }
        }
        let slope = cross / square;
        assert!((0.97..=1.03).contains(&slope), "slope {slope}");
        // The inner product lies within plus or minus the query's distance
        // from the centroid; the error is a small part of that range.
        let spread = (error / scale).sqrt();
        assert!(spread < 0.1, "error is {spread} of the range");
    }

    #[test]
    fn a_shortlist_holds_the_true_neighbours_of_scattered_vectors() {
        let vectors = scattered(5, 3000, 256);
        for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
            let index = indexed(metric, &vectors);
            let (found, total) = recall(&index, metric, &vectors, 300);
            assert!(
                found * 100 >= total * 90,
                "{metric:?}: {found} of {total} shortlisted"
            );
        }
    }

    #[test]
    fn a_shortlist_holds_the_true_neighbours_under_every_metric() {
        let vectors = grouped(42, 6000, 128, 40, 0.4);
        for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
            let index = indexed(metric, &vectors);
            assert_eq!(index.len(), 6000);
            assert_eq!(index.clusters(), 77);
            let (found, total) = recall(&index, metric, &vectors, 100);
            assert!(
                found * 100 >= total * 95,
                "{metric:?}: {found} of {total} shortlisted"
            );
        }
    }

    #[test]
    fn an_offset_shared_by_all_vectors_costs_nothing() {
        let vectors: Vec<Vec<f32>> = grouped(9, 5000, 96, 25, 0.3)
            .into_iter()
            .map(|v| v.into_iter().map(|x| x + 5.0).collect())
            .collect();
        let index = indexed(Distance::L2, &vectors);
        let (found, total) = recall(&index, Distance::L2, &vectors, 100);
        assert!(found * 100 >= total * 95, "{found} of {total} shortlisted");
    }

    #[test]
    fn probes_are_as_few_as_the_recall_floor_allows() {
        let tight = indexed(Distance::L2, &grouped(1, 5000, 64, 50, 0.05));
        let loose = indexed(Distance::L2, &grouped(1, 5000, 64, 50, 2.0));
        assert!(
            tight.probes() < loose.probes(),
            "{} vs {}",
            tight.probes(),
            loose.probes()
        );
        assert!(tight.probes() >= 1);
        assert!(loose.probes() <= loose.clusters());
    }

    #[test]
    fn a_shortlist_keeps_to_the_rows_it_is_allowed() {
        let vectors = grouped(3, 2000, 64, 10, 0.5);
        let index = indexed(Distance::Cosine, &vectors);
        let query = &vectors[0];

        let only: RoaringBitmap = [0u32, 2, 3, 40, 198, 9999].into_iter().collect();
        let mut allowed = index.shortlist(query, 50, Some(&only), &[], &in_row_order);
        allowed.sort_unstable();
        assert_eq!(allowed, [0, 2, 40, 198], "rows 3 and 9999 are not indexed");

        let mut skipped = vec![false; 4000];
        skipped[0] = true;
        skipped[40] = true;
        let mut kept = index.shortlist(query, 50, Some(&only), &skipped, &in_row_order);
        kept.sort_unstable();
        assert_eq!(kept, [2, 198]);

        let every_other: RoaringBitmap = (0..4000u32).filter(|row| row % 4 == 0).collect();
        let wide = index.shortlist(query, 30, Some(&every_other), &skipped, &in_row_order);
        assert_eq!(wide.len(), 30, "scans on past the usual probes until full");
        assert!(wide
            .iter()
            .all(|row| row % 4 == 0 && *row != 0 && *row != 40));

        let everything = index.shortlist(query, 5000, None, &skipped, &in_row_order);
        assert_eq!(everything.len(), 1998);
        assert_eq!(
            index.shortlist(query, 0, None, &[], &in_row_order),
            Vec::<u32>::new()
        );
        assert_eq!(
            index.shortlist(&query[..10], 5, None, &[], &in_row_order),
            Vec::<u32>::new()
        );
        assert!(ClusteredIndex::build(Distance::L2, &[]).is_none());
    }

    #[test]
    fn a_filter_keeping_few_rows_finds_their_nearest_as_the_scan_would() {
        let vectors = grouped(11, 3000, 64, 30, 0.5);
        let index = indexed(Distance::L2, &vectors);
        let query = &vectors[7];
        let few: RoaringBitmap = (0..6000u32).filter(|row| row % 100 == 0).collect();
        let mut by_rows = index.shortlist(query, 30, Some(&few), &[], &in_row_order);
        by_rows.sort_unstable();
        let mut exact: Vec<(f32, u32)> = few
            .iter()
            .map(|row| {
                (
                    distance(Distance::L2, query, &vectors[row as usize / 2]),
                    row,
                )
            })
            .collect();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0));
        let nearest: std::collections::BTreeSet<u32> =
            exact[..10].iter().map(|(_, row)| *row).collect();
        let found = by_rows.iter().filter(|row| nearest.contains(row)).count();
        assert_eq!(by_rows.len(), 30);
        assert!(found >= 9, "{found} of the nearest ten allowed rows");
    }

    #[test]
    fn a_vector_at_its_centroid_and_tiny_indexes_are_handled() {
        let same: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]; 5];
        let index = indexed(Distance::L2, &same);
        assert!(index.clusters() <= 2);
        let mut rows = index.shortlist(&[1.0, 2.0, 3.0], 10, None, &[], &in_row_order);
        rows.sort_unstable();
        assert_eq!(rows, [0, 2, 4, 6, 8]);
        let one = ClusteredIndex::build(Distance::Cosine, &[(3, &[0.0, 0.0])]).unwrap();
        assert_eq!(one.shortlist(&[1.0, 0.0], 1, None, &[], &in_row_order), [3]);
    }

    /// What a row's factors say of its squared length is what it is: for
    /// rows at an everyday length, and for rows leaning far one way near
    /// the length limit, where each lies far along its centroid.
    #[test]
    fn a_rows_squared_length_is_worked_out_again_from_its_factors() {
        let everyday = scattered(41, 2000, 48);
        let mut noise = Noise(43);
        let lean = noise.vector(48);
        let far: Vec<Vec<f32>> = (0..2000)
            .map(|_| {
                let vector: Vec<f32> = lean.iter().map(|l| l + 0.3 * noise.unit()).collect();
                let scale = (0.9 * SQUARED_LENGTH_UNDER / dot(&vector, &vector)).sqrt();
                vector.into_iter().map(|v| v * scale).collect()
            })
            .collect();
        for vectors in [everyday, far] {
            for metric in [Distance::Dot, Distance::L2] {
                let index = indexed(metric, &vectors);
                assert!(index.unmeasured(None, &[]).is_empty());
                for (at, vector) in vectors.iter().enumerate().step_by(7) {
                    let (cluster, place) = index.at[index.position(at as u32 * 2).unwrap()];
                    let centroid = index.centroid(cluster as usize);
                    let centroid: f64 = centroid.iter().map(|v| f64::from(*v).powi(2)).sum();
                    let factor = index.clusters[cluster as usize].factors[place as usize];
                    let rotated = index.prepared(vector);
                    let exact: f64 = rotated.iter().map(|v| f64::from(*v).powi(2)).sum();
                    let worked = factor.squared(centroid);
                    assert!(
                        (worked - exact).abs() <= 1e-4 * exact,
                        "{metric:?}: {worked} against {exact}"
                    );
                }
            }
        }
    }

    /// Indexes built by the release before, which knew nothing of rows too
    /// long to measure, gave them codes and finite distances from their
    /// centroids: thirty copies each of `[±2.3e18; 32]` far from theirs,
    /// and two hundred of `[1e19; 32]` a cluster of their own, at no
    /// distance from its centroid. Read back, they are recognised from what
    /// was stored, and left to be scored exactly.
    #[test]
    fn indexes_built_before_rows_were_measured_recognise_them_on_load() {
        let fixtures: [(&[u8], usize); 2] = [
            (
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/clustered-main-legacy-rows.msgpack"
                )),
                60,
            ),
            (
                include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/clustered-main-own-cluster.msgpack"
                )),
                200,
            ),
        ];
        for (bytes, long) in fixtures {
            let index: ClusteredIndex = rmp_serde::from_slice(bytes).unwrap();
            assert_eq!(index.len(), 3000 + long);
            let rows: Vec<u32> = (3000..3000 + long as u32).map(|row| row * 2).collect();
            assert_eq!(index.unmeasured(None, &[]), rows, "{long} long rows");
            for query in grouped(31, 3000, 32, 10, 0.4).iter().step_by(300) {
                let shortlist = index.shortlist(query, 100, None, &[], &in_row_order);
                assert!(shortlist.iter().all(|row| *row < 6000));
            }
        }
    }

    /// Rows that cannot be measured shape nothing: the clusters, and how
    /// many a query scans, which is kept with the index and paid by every
    /// query, come out as without them.
    #[test]
    fn rows_too_long_to_measure_leave_the_index_as_it_would_be() {
        let measured = grouped(29, 3000, 32, 10, 0.4);
        let with_long: Vec<Vec<f32>> = measured
            .iter()
            .enumerate()
            .flat_map(|(at, vector)| {
                let long = (at % 50 == 0)
                    .then(|| vec![2.3e18 * if at % 100 == 0 { 1.0 } else { -1.0 }; 32]);
                std::iter::once(vector.clone()).chain(long)
            })
            .collect();
        assert_eq!(with_long.len(), 3060);
        for metric in [Distance::Dot, Distance::L2] {
            let (without, with) = (indexed(metric, &measured), indexed(metric, &with_long));
            assert_eq!(with.unmeasured(None, &[]).len(), 60);
            assert_eq!(
                (with.clusters(), with.probes()),
                (without.clusters(), without.probes()),
                "{metric:?}"
            );
            assert_eq!(with.centroids, without.centroids, "{metric:?}");
        }
    }

    /// Vectors longer than a collection accepts have no estimate: they draw
    /// no centroid to them, take no place in a shortlist, and are left for
    /// the caller to score exactly, read back from the wire as they were.
    #[test]
    fn vectors_too_long_to_measure_are_left_to_be_scored_exactly() {
        let mut vectors = grouped(23, 3000, 32, 10, 0.4);
        vectors.extend(std::iter::repeat_n(vec![2.3e18; 32], 300));
        let long: Vec<u32> = (3000..3300u32).map(|row| row * 2).collect();
        for metric in [Distance::Dot, Distance::L2] {
            let index = indexed(metric, &vectors);
            assert_eq!(index.len(), 3300);
            assert_eq!(index.unmeasured(None, &[]), long, "{metric:?}");
            let mut found = 0;
            for query in vectors[..3000].iter().step_by(300) {
                let shortlist = index.shortlist(query, 100, None, &[], &in_row_order);
                assert_eq!(shortlist.len(), 100);
                assert!(shortlist.iter().all(|row| *row < 6000), "{metric:?}");
                let mut exact: Vec<(f32, u32)> = vectors[..3000]
                    .iter()
                    .enumerate()
                    .map(|(row, v)| (distance(metric, query, v), row as u32 * 2))
                    .collect();
                exact.sort_by(|a, b| a.0.total_cmp(&b.0));
                found += exact[..10]
                    .iter()
                    .filter(|(_, row)| shortlist.contains(row))
                    .count();
            }
            assert!(found >= 90, "{metric:?}: {found} of 100 true neighbours");
            let bytes = rmp_serde::to_vec_named(&index).unwrap();
            let back: ClusteredIndex = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(back.unmeasured(None, &[]), long);
        }
    }

    /// A flood of rows at one distance, as of copies of one vector, holds
    /// no more than twice the count while scanned, and keeps the first of
    /// them by the caller's order.
    #[test]
    fn a_flood_of_ties_is_held_to_twice_the_count() {
        let backwards = |a: u32, b: u32| b.cmp(&a);
        let mut nearest = Nearest::new(3, &backwards);
        for row in 0..10_000 {
            let before = nearest.tied.len();
            nearest.offer(1.0, row);
            let after = nearest.tied.len();
            assert!(after <= 6, "{after} tied");
            // Let go of all but the count at once, not one at a time.
            assert!(before < 6 || after == 3, "{before} tied became {after}");
        }
        let mut rows: Vec<u32> = nearest
            .estimates()
            .into_iter()
            .map(|(_, row)| row)
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, [9997, 9998, 9999]);
    }

    /// Rows tied with the farthest of the nearest are tied no longer once
    /// a nearer row puts it out and the farthest left is nearer still: none
    /// of them may then displace the rows that are kept.
    #[test]
    fn ties_with_a_farthest_that_is_put_out_are_let_go() {
        let mut nearest = Nearest::new(1, &in_row_order);
        nearest.offer(1.0, 5);
        nearest.offer(1.0, 3);
        nearest.offer(0.5, 9);
        assert_eq!(nearest.estimates(), [(0.5, 9)]);
    }

    /// A row put out of the nearest by a nearer one, when the farthest left
    /// is as far as it was, is still tied with it, and the caller's order
    /// may yet prefer it.
    #[test]
    fn a_row_put_out_by_a_nearer_one_stays_tied_with_the_farthest() {
        let backwards = |a: u32, b: u32| b.cmp(&a);
        let mut nearest = Nearest::new(2, &backwards);
        nearest.offer(1.0, 5);
        nearest.offer(1.0, 7);
        nearest.offer(0.5, 9);
        let mut rows: Vec<u32> = nearest
            .estimates()
            .into_iter()
            .map(|(_, row)| row)
            .collect();
        rows.sort_unstable();
        assert_eq!(rows, [7, 9]);
    }

    /// An index is one of just the rows, dimensions and metric it was built
    /// of: whoever reads one back for some vectors asks before trusting it.
    #[test]
    fn an_index_fits_only_the_vectors_it_was_built_of() {
        let vectors = grouped(27, 500, 16, 5, 0.4);
        let rows = |shift: u32, vectors: &'_ [Vec<f32>]| -> Vec<(u32, Vec<f32>)> {
            vectors
                .iter()
                .enumerate()
                .map(|(row, v)| (row as u32 * 2 + shift, v.clone()))
                .collect()
        };
        let index = indexed(Distance::Cosine, &vectors);
        let fit = |metric, rows: &[(u32, Vec<f32>)]| {
            let rows: Vec<(u32, &[f32])> =
                rows.iter().map(|(row, v)| (*row, v.as_slice())).collect();
            index.fit(metric, &rows)
        };
        let wider = grouped(27, 500, 17, 5, 0.4);
        assert_eq!(fit(Distance::Cosine, &rows(0, &vectors)), Ok(()));
        assert_eq!(fit(Distance::Dot, &rows(0, &vectors)), Err(Unfit::Metric));
        assert_eq!(
            fit(Distance::Cosine, &rows(0, &wider)),
            Err(Unfit::Dimensions)
        );
        assert_eq!(fit(Distance::Cosine, &rows(1, &vectors)), Err(Unfit::Rows));
        assert_eq!(
            fit(Distance::Cosine, &rows(0, &vectors[..499])),
            Err(Unfit::Rows)
        );
    }

    /// An index whose stored parts do not fit one another is refused when
    /// read: a width no rotation has, which would otherwise be sized by,
    /// and centroids, codes, factors or a count of clusters to scan that
    /// would otherwise panic at some later query. Whether it is an index of
    /// the vectors it is read for is not told here but by
    /// [`ClusteredIndex::fit`].
    #[test]
    fn an_index_whose_parts_do_not_fit_is_refused_on_load() {
        let index = indexed(Distance::L2, &grouped(21, 1000, 48, 8, 0.4));
        let bytes = rmp_serde::to_vec_named(&index).unwrap();
        type Corrupt = fn(&mut Wire);
        let reread = |corrupt: Corrupt| {
            let mut wire: Wire = rmp_serde::from_slice(&bytes).unwrap();
            corrupt(&mut wire);
            let bytes = rmp_serde::to_vec_named(&wire).unwrap();
            rmp_serde::from_slice::<ClusteredIndex>(&bytes).map(|index| index.len())
        };
        assert_eq!(reread(|_| {}).unwrap(), 1000);
        let corruptions: [(&str, Corrupt); 12] = [
            ("no clusters to scan", |wire| wire.probes = 0),
            ("more clusters to scan than there are", |wire| {
                wire.probes = usize::MAX
            }),
            ("no width", |wire| wire.rotation.width = 0),
            ("a width that is no power of two", |wire| {
                wire.rotation.width = 100
            }),
            ("a width beyond memory", |wire| {
                wire.rotation.width = 1 << 40
            }),
            ("a width the centroids and codes are not of", |wire| {
                wire.rotation.width = 128
            }),
            ("more dimensions than the width", |wire| {
                wire.dimensions = 65
            }),
            ("a centroid cut short", |wire| {
                wire.centroids.pop();
            }),
            ("codes cut short", |wire| {
                let blocks = &wire.clusters[0].blocks;
                let words = blocks.words();
                let shorter: Vec<u64> = blocks
                    .codes()
                    .chunks_exact(words)
                    .flat_map(|code| code[..words - 1].to_vec())
                    .collect();
                wire.clusters[0].blocks = Blocks::new(&shorter, words - 1);
            }),
            ("codes that are not whole rows", |wire| {
                wire.clusters[0].rows.pop();
                wire.clusters[0].factors.pop();
            }),
            ("a row without factors", |wire| {
                wire.clusters[0].factors.pop();
            }),
            ("no clusters", |wire| {
                wire.clusters.clear();
                wire.centroids.clear();
            }),
        ];
        for (what, corrupt) in corruptions {
            assert!(reread(corrupt).is_err(), "an index with {what} was read");
        }
    }

    #[test]
    fn an_index_survives_the_wire() {
        let vectors = grouped(21, 1000, 48, 8, 0.4);
        let index = indexed(Distance::Dot, &vectors);
        let bytes = rmp_serde::to_vec_named(&index).unwrap();
        let back: ClusteredIndex = rmp_serde::from_slice(&bytes).unwrap();
        for query in vectors.iter().step_by(100) {
            let mut a = index.shortlist(query, 20, None, &[], &in_row_order);
            let mut b = back.shortlist(query, 20, None, &[], &in_row_order);
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
        }
        assert_eq!(back.probes(), index.probes());
    }
}
