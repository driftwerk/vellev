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
//! A query allowed only some rows scans further until it has its shortlist,
//! or, when it is allowed few, visits just those rows.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use vellev_core::Distance;

use crate::vector::{dot, squared_l2, Scorer, VectorIndex};

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

/// Vectors clustered around centroids, each kept as a one-bit code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteredIndex {
    metric: Distance,
    dimensions: usize,
    rotation: Rotation,
    /// The centroids in the rotated space, [`Rotation::width`] values each.
    centroids: Vec<f32>,
    clusters: Vec<Cluster>,
    /// Every indexed row in ascending order, and for each its cluster and
    /// its place in it.
    rows: Vec<u32>,
    at: Vec<(u32, u32)>,
    /// How many clusters a query scans.
    probes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cluster {
    rows: Vec<u32>,
    #[serde(with = "words")]
    codes: Vec<u64>,
    factors: Vec<Factor>,
}

/// What a code needs besides its bits to stand for a vector `o` in a cluster
/// with centroid `c`: `o`'s distance from `c`, how well the code's unit
/// vector aligns with the direction from `c` to `o`, and that direction's
/// inner product with `c`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Factor {
    norm: f32,
    align: f32,
    along: f32,
}

/// A random orthogonal rotation: random signs followed by a Walsh-Hadamard
/// transform, over the dimensions padded to a power of two.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Rotation {
    width: usize,
    seed: u64,
}

impl Rotation {
    fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            width: dimensions.next_power_of_two().max(64),
            seed,
        }
    }

    fn apply(&self, vector: &[f32]) -> Vec<f32> {
        let mut random = Random(self.seed);
        let mut rotated: Vec<f32> = (0..self.width)
            .map(|i| {
                let sign = if random.next() & 1 == 0 { 1.0 } else { -1.0 };
                vector.get(i).copied().unwrap_or(0.0) * sign
            })
            .collect();
        let mut half = 1;
        while half < self.width {
            for block in rotated.chunks_exact_mut(half * 2) {
                let (low, high) = block.split_at_mut(half);
                for (a, b) in low.iter_mut().zip(high) {
                    let (x, y) = (*a, *b);
                    *a = x + y;
                    *b = x - y;
                }
            }
            half *= 2;
        }
        let scale = 1.0 / (self.width as f32).sqrt();
        rotated.iter_mut().for_each(|value| *value *= scale);
        rotated
    }
}

/// A small deterministic generator; the rotation and the sampling need
/// nothing better, and must come out the same on every load.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// A query made ready for one cluster: its difference from the centroid,
/// quantized to [`QUERY_BITS`] bit planes, with what the estimate needs.
struct Probe {
    planes: Vec<u64>,
    words: usize,
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
        Self {
            planes,
            words,
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
        let ones: u32 = code.iter().map(|word| word.count_ones()).sum();
        let weighted: u32 = (0..QUERY_BITS)
            .map(|bit| {
                let plane = &self.planes[bit as usize * self.words..][..self.words];
                let count: u32 = code
                    .iter()
                    .zip(plane)
                    .map(|(c, p)| (c & p).count_ones())
                    .sum();
                count << bit
            })
            .sum();
        let set = self.lower * ones as f32 + self.delta * weighted as f32;
        let unit = (2.0 * set - self.sum) * self.scale;
        unit / factor.align
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

/// The `count` nearest candidates offered, farthest on top.
struct Nearest {
    count: usize,
    heap: BinaryHeap<Candidate>,
}

impl Nearest {
    fn new(count: usize) -> Self {
        Self {
            count,
            heap: BinaryHeap::with_capacity(count + 1),
        }
    }

    fn offer(&mut self, distance: f32, row: u32) {
        if self.heap.len() < self.count {
            self.heap.push(Candidate { distance, row });
        } else if self
            .heap
            .peek()
            .is_some_and(|worst| distance < worst.distance)
        {
            self.heap.pop();
            self.heap.push(Candidate { distance, row });
        }
    }

    fn is_full(&self) -> bool {
        self.heap.len() >= self.count
    }

    fn rows(self) -> Vec<u32> {
        self.heap
            .into_iter()
            .map(|candidate| candidate.row)
            .collect()
    }
}

fn normalised(metric: Distance, vector: &[f32]) -> Vec<f32> {
    match metric {
        Distance::Cosine => {
            let norm = dot(vector, vector).sqrt();
            match norm > 0.0 && norm.is_finite() {
                true => vector.iter().map(|value| value / norm).collect(),
                false => vector.to_vec(),
            }
        }
        Distance::Dot | Distance::L2 => vector.to_vec(),
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

/// Lloyd's k-means over `training`, from a random choice of `k` of them.
fn kmeans(training: &[Vec<f32>], k: usize, width: usize, random: &mut Random) -> Vec<f32> {
    let mut centroids: Vec<f32> = (0..k)
        .flat_map(|_| training[random.below(training.len())].iter().copied())
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
        let mut random = Random(vectors.len() as u64 ^ (dimensions as u64) << 32);
        let rotation = Rotation::new(dimensions, random.next());
        let width = rotation.width;
        let rotated = |vector: &[f32]| rotation.apply(&normalised(metric, vector));

        let k = clusters_for(vectors.len());
        let stride = (vectors.len() / (k * TRAINING_PER_CLUSTER)).max(1);
        let training: Vec<Vec<f32>> = vectors
            .par_iter()
            .step_by(stride)
            .map(|(_, vector)| rotated(vector))
            .collect();
        let centroids = kmeans(&training, k, width, &mut random);

        let scale = 1.0 / (width as f32).sqrt();
        let coded: Vec<(usize, Vec<u64>, Factor)> = vectors
            .par_iter()
            .map(|(_, vector)| {
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
            rotation,
            centroids,
            clusters,
            rows: vectors.iter().map(|(row, _)| *row).collect(),
            at,
            probes: 1,
        };
        index.probes = index.tuned(vectors);
        Some(index)
    }

    /// How many clusters a query scans.
    pub fn probes(&self) -> usize {
        self.probes
    }

    pub fn clusters(&self) -> usize {
        self.clusters.len()
    }

    /// The fewest clusters whose scan holds [`RECALL_FLOOR`] of the true
    /// nearest [`TUNING_NEIGHBOURS`] of sample queries drawn from the
    /// vectors themselves, each left out of its own neighbours.
    fn tuned(&self, vectors: &[(u32, &[f32])]) -> usize {
        let stride = (vectors.len() / TUNING_QUERIES).max(1);
        let ranks: Vec<usize> = vectors
            .par_iter()
            .step_by(stride)
            .flat_map_iter(|(row, query)| {
                let scorer = Scorer::new(self.metric, query);
                let mut nearest = Nearest::new(TUNING_NEIGHBOURS);
                for (other, vector) in vectors {
                    if other != row {
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
                    .rows()
                    .into_iter()
                    .map(move |row| rank_of(self.at[self.position(row).unwrap()].0))
            })
            .collect();
        if ranks.is_empty() {
            return 1;
        }
        let mut found = vec![0usize; self.clusters.len() + 1];
        for rank in &ranks {
            found[*rank] += 1;
        }
        let wanted = (ranks.len() as f64 * RECALL_FLOOR).ceil() as usize;
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
                let distance = match self.metric {
                    Distance::L2 => squared_l2(query, centroid),
                    Distance::Dot | Distance::Cosine => -dot(query, centroid),
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

    fn shortlist(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
    ) -> Vec<u32> {
        if count == 0 || query.len() != self.dimensions {
            return Vec::new();
        }
        let query = self.prepared(query);
        let kept = |row: u32| !skipped.get(row as usize).copied().unwrap_or(false);
        let mut nearest = Nearest::new(count);
        let per_probe = self.rows.len().div_ceil(self.clusters.len()) * self.probes;
        match only {
            // Fewer rows allowed than a scan would visit: visit just those,
            // readying the query for each cluster met.
            Some(only) if (only.len() as usize) <= per_probe => {
                let mut probes: Vec<Option<Probe>> =
                    (0..self.clusters.len()).map(|_| None).collect();
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
        nearest.rows()
    }
}

/// Codes on the wire as little-endian bytes.
mod words {
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(words: &[u64], serializer: S) -> Result<S::Ok, S::Error> {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        serde_bytes::serialize(&bytes, serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u64>, D::Error> {
        let bytes: Vec<u8> = serde_bytes::deserialize(deserializer)?;
        if !bytes.len().is_multiple_of(8) {
            return Err(serde::de::Error::custom("codes are not whole words"));
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight bytes")))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::distance;

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
            let shortlist = index.shortlist(query, count, None, &[]);
            found += exact[..10]
                .iter()
                .filter(|(_, row)| shortlist.contains(row))
                .count();
            total += 10;
        }
        (found, total)
    }

    #[test]
    fn the_rotation_keeps_lengths_and_inner_products() {
        let mut noise = Noise(5);
        let rotation = Rotation::new(100, 77);
        assert_eq!(rotation.width, 128);
        for _ in 0..5 {
            let (a, b) = (noise.vector(100), noise.vector(100));
            let (ra, rb) = (rotation.apply(&a), rotation.apply(&b));
            assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-4);
            assert!((dot(&a, &a) - dot(&ra, &ra)).abs() < 1e-4);
        }
        assert_eq!(rotation.apply(&[1.0; 100]), rotation.apply(&[1.0; 100]));
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
        let mut allowed = index.shortlist(query, 50, Some(&only), &[]);
        allowed.sort_unstable();
        assert_eq!(allowed, [0, 2, 40, 198], "rows 3 and 9999 are not indexed");

        let mut skipped = vec![false; 4000];
        skipped[0] = true;
        skipped[40] = true;
        let mut kept = index.shortlist(query, 50, Some(&only), &skipped);
        kept.sort_unstable();
        assert_eq!(kept, [2, 198]);

        let every_other: RoaringBitmap = (0..4000u32).filter(|row| row % 4 == 0).collect();
        let wide = index.shortlist(query, 30, Some(&every_other), &skipped);
        assert_eq!(wide.len(), 30, "scans on past the usual probes until full");
        assert!(wide
            .iter()
            .all(|row| row % 4 == 0 && *row != 0 && *row != 40));

        let everything = index.shortlist(query, 5000, None, &skipped);
        assert_eq!(everything.len(), 1998);
        assert_eq!(index.shortlist(query, 0, None, &[]), Vec::<u32>::new());
        assert_eq!(
            index.shortlist(&query[..10], 5, None, &[]),
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
        let mut by_rows = index.shortlist(query, 30, Some(&few), &[]);
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
        let mut rows = index.shortlist(&[1.0, 2.0, 3.0], 10, None, &[]);
        rows.sort_unstable();
        assert_eq!(rows, [0, 2, 4, 6, 8]);
        let one = ClusteredIndex::build(Distance::Cosine, &[(3, &[0.0, 0.0])]).unwrap();
        assert_eq!(one.shortlist(&[1.0, 0.0], 1, None, &[]), [3]);
    }

    #[test]
    fn an_index_survives_the_wire() {
        let vectors = grouped(21, 1000, 48, 8, 0.4);
        let index = indexed(Distance::Dot, &vectors);
        let bytes = rmp_serde::to_vec_named(&index).unwrap();
        let back: ClusteredIndex = rmp_serde::from_slice(&bytes).unwrap();
        for query in vectors.iter().step_by(100) {
            let mut a = index.shortlist(query, 20, None, &[]);
            let mut b = back.shortlist(query, 20, None, &[]);
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b);
        }
        assert_eq!(back.probes(), index.probes());
    }
}
