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

use crate::rotation::{Random, Rotation};
use crate::vector::{
    allowed, as_far_from_every_vector, dot, in_row_order, measurable, nearest, normalised,
    or_farthest, short_enough, squared_l2, Scorer, Ties, VectorIndex,
};

/// The share of the true nearest ten that the clusters scanned must hold,
/// over sample queries, when the index is built.
pub const RECALL_FLOOR: f64 = 0.97;
/// How many bits the query is quantized to for the scan.
const QUERY_BITS: u32 = 4;
const TUNING_QUERIES: usize = 100;
const TUNING_NEIGHBOURS: usize = 10;
const ITERATIONS: usize = 8;
const MOST_CLUSTERS: usize = 4096;
/// How many vectors per cluster k-means is trained on, at most.
const TRAINING_PER_CLUSTER: usize = 64;

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
    codes: Vec<u64>,
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
            .position(|cluster| cluster.codes.len() != cluster.rows.len() * (width / 64))
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
            rows: cluster.rows,
            codes: cluster.codes,
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
        let factors = wire
            .norms
            .into_iter()
            .zip(wire.aligns)
            .zip(wire.alongs)
            .map(|((norm, align), along)| Factor { norm, align, along })
            .collect();
        Ok(Self {
            rows: wire.rows,
            codes: wire.codes,
            factors,
        })
    }
}

/// A query made ready for one cluster: its difference from the centroid,
/// quantized to [`QUERY_BITS`] bit planes, with what the estimate needs.
struct Probe {
    planes: Vec<u64>,
    words: usize,
    /// How this machine counts a code's bits against the planes: with its
    /// popcount instruction where it has one.
    counted: Counted,
    lower: f32,
    delta: f32,
    sum: f32,
    norm_squared: f32,
    centroid_dot: f32,
    scale: f32,
}

impl Probe {
    fn new(query: &[f32], centroid: &[f32]) -> Self {
        let residual: Vec<f32> = query.iter().zip(centroid).map(|(q, c)| q - c).collect();
        let (mut lower, mut upper) = (f32::INFINITY, f32::NEG_INFINITY);
        for value in &residual {
            lower = lower.min(*value);
            upper = upper.max(*value);
        }
        let levels = ((1u32 << QUERY_BITS) - 1) as f32;
        let delta = (upper - lower) / levels;
        let words = residual.len() / 64;
        debug_assert_eq!(residual.len() % 64, 0, "a rotation's width is whole words");
        let mut planes = vec![0u64; words * QUERY_BITS as usize];
        // A word of each plane at a time, each value's level taken as
        // before and its bits laid into the words without a branch: the
        // same planes, readied several times as fast.
        if delta > 0.0 {
            for (word, chunk) in residual.chunks_exact(64).enumerate() {
                let mut bits = [0u64; QUERY_BITS as usize];
                for (at, value) in chunk.iter().enumerate() {
                    let level = ((value - lower) / delta).round().clamp(0.0, levels) as u64;
                    for (bit, plane) in bits.iter_mut().enumerate() {
                        *plane |= (level >> bit & 1) << at;
                    }
                }
                for (bit, plane) in bits.into_iter().enumerate() {
                    planes[bit * words + word] = plane;
                }
            }
        }
        Self {
            planes,
            words,
            counted: counter(),
            lower,
            delta,
            sum: residual.iter().sum(),
            norm_squared: dot(&residual, &residual),
            centroid_dot: dot(query, centroid),
            scale: 1.0 / (residual.len() as f32).sqrt(),
        }
    }

    /// The inner product of the direction a code stands for with the
    /// query's difference from the centroid.
    fn inner(&self, code: &[u64], factor: &Factor) -> f32 {
        let (ones, weighted) = (self.counted)(code, &self.planes, self.words);
        let set = self.lower * ones as f32 + self.delta * weighted as f32;
        (2.0 * set - self.sum) * self.scale / factor.align
    }

    fn distance(&self, metric: Distance, code: &[u64], factor: &Factor) -> f32 {
        let inner = self.inner(code, factor);
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

/// A code's set bits, and its bits set in each of the query's planes
/// weighted by the plane's place: `(ones, weighted)`.
type Counted = fn(&[u64], &[u64], usize) -> (u32, u32);

const _: () = assert!(QUERY_BITS == 4, "the scan counts four planes");

/// Every count of a code in one pass over its words, rather than one pass
/// for its bits and one for each plane: the same integers, so the same
/// estimate to the bit.
#[inline(always)]
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

fn counted_portably(code: &[u64], planes: &[u64], words: usize) -> (u32, u32) {
    counted_in_one_pass(code, planes, words)
}

/// As [`counted_portably`], compiled to use the popcount instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
fn counted_with_popcount(code: &[u64], planes: &[u64], words: usize) -> (u32, u32) {
    counted_in_one_pass(code, planes, words)
}

/// The counter for this machine: decided once per process.
fn counter() -> Counted {
    static COUNTER: std::sync::OnceLock<Counted> = std::sync::OnceLock::new();
    *COUNTER.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("popcnt") {
            // SAFETY: called only where the CPU has just said it has
            // popcount, which is all the function asks.
            return |code, planes, words| unsafe { counted_with_popcount(code, planes, words) };
        }
        counted_portably
    })
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
    // Scratch: a multiple of the usual count, for an experiment.
    let times: f64 = std::env::var("SCRATCH_CLUSTERS_TIMES")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(1.0);
    (((count as f64).sqrt() * times) as usize).clamp(1, (MOST_CLUSTERS * 4).min(count))
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

        let mut clusters: Vec<Cluster> = (0..k)
            .map(|_| Cluster {
                rows: Vec::new(),
                codes: Vec::new(),
                factors: Vec::new(),
            })
            .collect();
        let mut at = Vec::with_capacity(vectors.len());
        for ((row, _), (cluster, code, factor)) in vectors.iter().zip(coded) {
            let members = &mut clusters[cluster];
            at.push((cluster as u32, members.rows.len() as u32));
            members.rows.push(*row);
            members.codes.extend(code);
            members.factors.push(factor);
        }
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
        };
        index.probes = index.tuned(&measured, &mut random);
        Some(index)
    }

    /// How many clusters a query scans.
    pub fn probes(&self) -> usize {
        self.probes
    }

    /// Scratch: the clusters a query would scan were it to scan those whose
    /// centroid lies within `1 + margin` of the nearest's squared distance,
    /// at least `least`.
    pub fn scratch_probes_within(&self, query: &[f32], margin: f32, least: usize) -> usize {
        let order = self.order(&self.prepared(query));
        let first = order.first().map_or(0.0, |(d, _)| *d);
        order
            .iter()
            .take_while(|(d, _)| *d <= first * (1.0 + margin))
            .count()
            .max(least)
    }

    /// Scratch: estimates scanning `probes` clusters, unfiltered.
    pub fn scratch_estimates(&self, query: &[f32], count: usize, probes: usize) -> Vec<(f32, u32)> {
        let mut nearest = Nearest::new(count, &crate::vector::in_row_order);
        let query = self.prepared(query);
        for (probed, (_, cluster)) in self.order(&query).into_iter().enumerate() {
            if probed >= probes && nearest.is_full() {
                break;
            }
            let probe = Probe::new(&query, self.centroid(cluster));
            let members = &self.clusters[cluster];
            for (index, row) in members.rows.iter().enumerate() {
                let code = self.code(members, index);
                nearest.offer(
                    probe.distance(self.metric, code, &members.factors[index]),
                    *row,
                );
            }
        }
        nearest.estimates()
    }

    /// Scratch: this index scanning `probes` clusters at least.
    pub fn with_probes(mut self, probes: usize) -> Self {
        self.probes = probes.clamp(1, self.clusters.len());
        self
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

    /// The fewest clusters whose scan holds [`RECALL_FLOOR`] of the true
    /// nearest [`TUNING_NEIGHBOURS`] of sample queries drawn at random from
    /// `vectors` themselves, each left out of its own neighbours: the rows
    /// that can be measured, as no other is ever offered.
    fn tuned(&self, vectors: &[(u32, &[f32])], random: &mut Random) -> usize {
        let samples = std::env::var("SCRATCH_TUNING_QUERIES").ok().and_then(|q| q.parse().ok()).unwrap_or(TUNING_QUERIES);
        let midpoints = std::env::var("SCRATCH_MIDPOINTS").is_ok_and(|m| m == "1");
        // Scratch: queries are rows, or halfway between two random rows.
        let queries: Vec<(Option<u32>, Vec<f32>)> = match midpoints {
            false => random.sample(vectors.len(), samples).into_iter().map(|at| (Some(vectors[at].0), vectors[at].1.to_vec())).collect(),
            true => (0..samples).map(|_| {
                let (a, b) = (random.below(vectors.len()), random.below(vectors.len()));
                (None, vectors[a].1.iter().zip(vectors[b].1).map(|(x, y)| (x + y) / 2.0).collect())
            }).collect(),
        };
        let ranks: Vec<usize> = queries
            .par_iter()
            .flat_map_iter(|(row, query)| {
                let scorer = Scorer::new(self.metric, query);
                let mut nearest = Nearest::new(TUNING_NEIGHBOURS, &in_row_order);
                for (other, vector) in vectors {
                    if Some(*other) != *row {
                        nearest.offer(scorer.distance(vector), *other);
                    }
                }
                let order = self.order(&self.prepared(query));
                let rank_of = move |cluster: u32| {
                    order
                        .iter()
                        .position(|(_, candidate)| *candidate == cluster as usize)
                        .unwrap_or(order.len())
                };
                nearest
                    .estimates()
                    .into_iter()
                    .map(move |(_, row)| rank_of(self.at[self.position(row).unwrap()].0))
                    .collect::<Vec<_>>()
            })
            .collect();
        if ranks.is_empty() {
            return 1;
        }
        let mut found = vec![0usize; self.clusters.len() + 1];
        for rank in &ranks {
            found[*rank] += 1;
        }
        let floor: f64 = std::env::var("SCRATCH_RECALL_FLOOR").ok().and_then(|f| f.parse().ok()).unwrap_or(RECALL_FLOOR);
        let wanted = (ranks.len() as f64 * floor).ceil() as usize;
        let mut seen = 0;
        for (rank, count) in found.iter().enumerate().take(self.clusters.len()) {
            seen += count;
            if seen >= wanted {
                return rank + 1;
            }
        }
        self.clusters.len()
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

    fn code<'a>(&self, cluster: &'a Cluster, index: usize) -> &'a [u64] {
        let words = self.rotation.width / 64;
        &cluster.codes[index * words..][..words]
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
        let per_probe = self.rows.len().div_ceil(self.clusters.len()) * self.probes;
        match only {
            // Fewer rows allowed than a scan would visit: visit just those,
            // readying the query for each cluster met.
            Some(only) if (only.len() as usize) <= per_probe => {
                let mut probes: Vec<Option<Probe>> = std::iter::repeat_with(|| None)
                    .take(self.clusters.len())
                    .collect();
                for row in only.iter().filter(|row| kept(*row)) {
                    let Some(position) = self.position(row) else {
                        continue;
                    };
                    let (cluster, index) = self.at[position];
                    let probe = probes[cluster as usize]
                        .get_or_insert_with(|| Probe::new(&query, self.centroid(cluster as usize)));
                    let members = &self.clusters[cluster as usize];
                    let code = self.code(members, index as usize);
                    let distance =
                        probe.distance(self.metric, code, &members.factors[index as usize]);
                    nearest.offer(distance, row);
                }
            }
            _ => {
                let allowed = |row: u32| kept(row) && only.is_none_or(|only| only.contains(row));
                for (probed, (_, cluster)) in self.order(&query).into_iter().enumerate() {
                    if probed >= self.probes && nearest.is_full() {
                        break;
                    }
                    let probe = Probe::new(&query, self.centroid(cluster));
                    let members = &self.clusters[cluster];
                    for (index, row) in members.rows.iter().enumerate() {
                        if allowed(*row) {
                            let code = self.code(members, index);
                            let distance =
                                probe.distance(self.metric, code, &members.factors[index]);
                            nearest.offer(distance, *row);
                        }
                    }
                }
            }
        }
        nearest.estimates()
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
            words in 1usize..20,
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
            words in 1usize..20,
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
                let estimated = f64::from(probe.inner(index.code(members, i), &members.factors[i]));
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
        let corruptions: [(&str, Corrupt); 11] = [
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
            ("a code cut short", |wire| {
                wire.clusters[0].codes.pop();
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
