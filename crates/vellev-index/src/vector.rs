//! Nearest-vector search by a compressed scan and an exact rescore.
//!
//! A [`BinaryIndex`] keeps one bit per dimension of every vector: whether it
//! lies above the mean of the indexed vectors in that dimension. Comparing
//! two such codes costs a few XOR and popcount instructions, so a scan over
//! every code produces a shortlist cheaply, and only the shortlist is scored
//! with the full vectors and the real distance.
//!
//! The shortlist is the only approximate step. It is taken from the rows the
//! caller allows, not filtered afterwards, so a filter never thins it.

use std::borrow::Cow;

use roaring::RoaringBitmap;
use vellev_core::Distance;

const LANES: usize = 16;

/// The dot product, accumulated in independent lanes so that it compiles to
/// wide vector instructions without any being named.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    lanes(a, b, |x, y| x * y)
}

pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    lanes(a, b, |x, y| (x - y) * (x - y))
}

#[inline]
fn lanes(a: &[f32], b: &[f32], term: impl Fn(f32, f32) -> f32) -> f32 {
    let (a_chunks, b_chunks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .map(|(x, y)| term(*x, *y))
        .sum();
    let sums = a_chunks
        .zip(b_chunks)
        .fold([0.0f32; LANES], |mut sums, (x, y)| {
            for lane in 0..LANES {
                sums[lane] += term(x[lane], y[lane]);
            }
            sums
        });
    sums.iter().sum::<f32>() + tail
}

/// The dot product of `a` and `b` together with `b`'s squared length, in one
/// pass over `b`.
fn dot_and_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    let (a_chunks, b_chunks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail = a_chunks
        .remainder()
        .iter()
        .zip(b_chunks.remainder())
        .fold((0.0, 0.0), |(dot, norm), (x, y)| {
            (dot + x * y, norm + y * y)
        });
    let sums = a_chunks
        .zip(b_chunks)
        .fold([(0.0f32, 0.0f32); LANES], |mut sums, (x, y)| {
            for lane in 0..LANES {
                sums[lane].0 += x[lane] * y[lane];
                sums[lane].1 += y[lane] * y[lane];
            }
            sums
        });
    sums.iter()
        .fold(tail, |(dot, norm), (d, n)| (dot + d, norm + n))
}

/// `distance`, or the farthest there is when it is not a number: a sort
/// cannot be trusted to put a NaN last, as the sign it comes with varies.
pub(crate) fn or_farthest(distance: f32) -> f32 {
    if distance.is_nan() {
        f32::INFINITY
    } else {
        distance
    }
}

/// Measures how far vectors are from one query; smaller is nearer. Cosine
/// distance is one minus the cosine, and one for a zero vector, which has no
/// direction. A distance that is not a number is taken as the farthest;
/// vectors no longer than a collection accepts never produce one.
#[derive(Debug, Clone)]
pub struct Scorer<'q> {
    metric: Distance,
    query: &'q [f32],
    query_norm: f32,
}

impl<'q> Scorer<'q> {
    pub fn new(metric: Distance, query: &'q [f32]) -> Self {
        Self {
            metric,
            query,
            query_norm: dot(query, query).sqrt(),
        }
    }

    pub fn distance(&self, vector: &[f32]) -> f32 {
        match self.metric {
            Distance::Cosine => {
                let (dot, norm) = dot_and_norm(self.query, vector);
                // Not finite for a zero vector, or when a length overflows.
                let cosine = dot / (self.query_norm * norm.sqrt());
                if cosine.is_finite() {
                    1.0 - cosine.clamp(-1.0, 1.0)
                } else {
                    1.0
                }
            }
            Distance::Dot => or_farthest(-dot(self.query, vector)),
            Distance::L2 => or_farthest(squared_l2(self.query, vector)),
        }
    }

    /// See [`certainty`].
    pub fn certainty(&self, distance: f32) -> Option<f32> {
        certainty(self.metric, distance)
    }
}

/// How far `a` is from `b` under `metric`.
pub fn distance(metric: Distance, a: &[f32], b: &[f32]) -> f32 {
    Scorer::new(metric, a).distance(b)
}

/// A cosine distance on the scale `(1 + cosine) / 2`, from 0 for opposite to
/// 1 for identical. Other metrics have no such scale.
pub fn certainty(metric: Distance, distance: f32) -> Option<f32> {
    (metric == Distance::Cosine).then_some(1.0 - distance / 2.0)
}

/// `vector` as an index measures it: scaled to unit length for cosine, so
/// that only its direction counts, and as it is otherwise. A zero vector, or
/// one whose length cannot be worked out, is left as it is.
pub(crate) fn normalised(metric: Distance, vector: &[f32]) -> Cow<'_, [f32]> {
    match metric {
        Distance::Cosine => {
            let norm = dot(vector, vector).sqrt();
            match norm > 0.0 && norm.is_finite() {
                true => vector.iter().map(|value| value / norm).collect(),
                false => Cow::Borrowed(vector),
            }
        }
        Distance::Dot | Distance::L2 => Cow::Borrowed(vector),
    }
}

/// A way to pick, cheaply, the rows most likely to be nearest a query.
pub trait VectorIndex {
    /// How many rows are indexed.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Up to `count` rows likely to be nearest `query`, in no particular
    /// order, taken only from `only` when it is given and never from a row
    /// flagged in `skipped`, which is indexed by row.
    fn shortlist(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
    ) -> Vec<u32>;
}

/// One bit per dimension of each vector, for a scan by Hamming distance.
///
/// A row's code says on which side of the mean `m` its vector `o` lies in
/// each dimension, and the query's code the same of the query `q`. The
/// share of bits on which they differ estimates the angle between `o - m`
/// and `q - m` as a share of a half turn: in expectation for codes taken
/// along random directions, and closely enough along the axes. Up to a term
/// the same for every row, each metric's distance is a term of the row's
/// own, kept beside its code, less a multiple of `|o - m| |q - m|` times
/// that angle's cosine. A row's estimate so weighs the angle by the row's
/// length, without which a long vector is lost among short ones.
///
/// The error of that estimate grows with `|o - m| |q - m|` too, so a long
/// vector that looks nearly at right angles to the query may still be
/// nearer than every short one. Each row is ranked by the angle at the near
/// end of the one-standard-error (Wilson) interval about the measured one,
/// which is to say by how near it may well be. That end rises with the
/// Hamming distance, so rows of one length keep the order their Hamming
/// distances give.
///
/// For cosine the vectors and the query are scaled to unit length first,
/// so that only directions count.
#[derive(Debug, Clone)]
pub struct BinaryIndex {
    metric: Distance,
    mean: Vec<f32>,
    words: usize,
    /// The indexed rows in ascending order; a row's code is at its position.
    rows: Vec<u32>,
    codes: Vec<u64>,
    /// Each row's `|o - m|`, and its own term of the distance: `|o - m|²`
    /// for L2, `-m · (o - m)` for the inner products.
    lengths: Vec<f32>,
    terms: Vec<f32>,
    /// By Hamming distance, the cosine of the angle a row is ranked by.
    cosines: Vec<f32>,
}

impl BinaryIndex {
    /// Index `vectors`, which must be in ascending row order and all of one
    /// length, for search under `metric`. Returns `None` when there is
    /// nothing to index.
    pub fn build(metric: Distance, vectors: &[(u32, &[f32])]) -> Option<Self> {
        let dimensions = vectors.first()?.1.len();
        debug_assert!(vectors.windows(2).all(|pair| pair[0].0 < pair[1].0));
        debug_assert!(vectors.iter().all(|(_, vector)| vector.len() == dimensions));
        let count = vectors.len() as f32;
        let mean = vectors
            .iter()
            .fold(vec![0.0f32; dimensions], |mut sum, (_, vector)| {
                sum.iter_mut()
                    .zip(normalised(metric, vector).iter())
                    .for_each(|(sum, value)| *sum += value / count);
                sum
            });
        let words = dimensions.div_ceil(64);
        let mut codes = Vec::with_capacity(vectors.len() * words);
        let (lengths, terms) = vectors
            .iter()
            .map(|(_, vector)| {
                let vector = normalised(metric, vector);
                codes.extend(encode(&mean, &vector));
                let (squared, along) = residual(&mean, &vector);
                let term = match metric {
                    Distance::L2 => squared,
                    Distance::Dot | Distance::Cosine => -along,
                };
                (squared.sqrt(), term)
            })
            .unzip();
        Some(Self {
            metric,
            words,
            rows: vectors.iter().map(|(row, _)| *row).collect(),
            codes,
            lengths,
            terms,
            cosines: (0..=dimensions)
                .map(|hamming| ranked_cosine(hamming, dimensions))
                .collect(),
            mean,
        })
    }

    fn hamming(&self, position: usize, query: &[u64]) -> usize {
        let code = &self.codes[position * self.words..(position + 1) * self.words];
        code.iter()
            .zip(query)
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }
}

/// The squared length of `vector - mean`, and its inner product with `mean`.
fn residual(mean: &[f32], vector: &[f32]) -> (f32, f32) {
    mean.iter()
        .zip(vector)
        .fold((0.0, 0.0), |(squared, along), (mean, value)| {
            let difference = value - mean;
            (squared + difference * difference, along + mean * difference)
        })
}

/// The cosine of the angle a row `hamming` of `dimensions` bits from the
/// query is ranked by: the near end of the Wilson interval, at one standard
/// error, about the share of a half turn those bits measure.
fn ranked_cosine(hamming: usize, dimensions: usize) -> f32 {
    let bits = dimensions as f64;
    let share = hamming as f64 / bits;
    let spread = (share * (1.0 - share) / bits + 0.25 / (bits * bits)).sqrt();
    let near = (share + 0.5 / bits - spread) / (1.0 + 1.0 / bits);
    (std::f64::consts::PI * near.max(0.0)).cos() as f32
}

/// One bit per dimension, set where the vector is above the mean. Built
/// without a branch per dimension, which a coin-flip bit would mispredict
/// half the time.
fn encode<'a>(mean: &'a [f32], vector: &'a [f32]) -> impl Iterator<Item = u64> + 'a {
    vector
        .chunks(64)
        .zip(mean.chunks(64))
        .map(|(values, means)| {
            values
                .iter()
                .zip(means)
                .enumerate()
                .fold(0u64, |word, (bit, (value, mean))| {
                    word | (u64::from(value > mean) << bit)
                })
        })
}

impl VectorIndex for BinaryIndex {
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
        if count == 0 || query.len() != self.mean.len() {
            return Vec::new();
        }
        let query = normalised(self.metric, query);
        let code: Vec<u64> = encode(&self.mean, &query).collect();
        let reach = residual(&self.mean, &query).0.sqrt()
            * match self.metric {
                Distance::L2 => 2.0,
                Distance::Dot | Distance::Cosine => 1.0,
            };
        let kept = |row: u32| !skipped.get(row as usize).copied().unwrap_or(false);
        let estimated = |position: usize| {
            let cosine = self.cosines[self.hamming(position, &code)];
            let estimate = self.terms[position] - reach * self.lengths[position] * cosine;
            (or_farthest(estimate), self.rows[position])
        };
        let mut estimates: Vec<(f32, u32)> = match only {
            None => (0..self.rows.len())
                .filter(|p| kept(self.rows[*p]))
                .map(estimated)
                .collect(),
            // Both are in ascending row order, so one pass finds each allowed
            // row's position without a search per row.
            Some(only) => only
                .iter()
                .filter(|row| kept(*row))
                .scan(0, |position, row| {
                    *position += self.rows[*position..].partition_point(|indexed| *indexed < row);
                    Some((*position, row))
                })
                .filter(|(position, row)| self.rows.get(*position) == Some(row))
                .map(|(position, _)| estimated(position))
                .collect(),
        };
        if estimates.len() > count {
            estimates.select_nth_unstable_by(count, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            estimates.truncate(count);
        }
        estimates.into_iter().map(|(_, row)| row).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator, so tests need no dependency.
    struct Random(u64);

    impl Random {
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

    fn naive(a: &[f32], b: &[f32], term: fn(f32, f32) -> f32) -> f32 {
        a.iter().zip(b).map(|(x, y)| term(*x, *y)).sum()
    }

    #[test]
    fn kernels_agree_with_a_plain_sum_at_every_length() {
        let mut random = Random(7);
        for dimensions in [0, 1, 15, 16, 17, 31, 32, 100, 768, 1024] {
            let (a, b) = (random.vector(dimensions), random.vector(dimensions));
            let close = |x: f32, y: f32| (x - y).abs() <= 1e-4 * (1.0 + y.abs());
            assert!(
                close(dot(&a, &b), naive(&a, &b, |x, y| x * y)),
                "dot at {dimensions}"
            );
            assert!(
                close(squared_l2(&a, &b), naive(&a, &b, |x, y| (x - y) * (x - y))),
                "l2 at {dimensions}"
            );
        }
    }

    #[test]
    fn distances_order_nearer_before_farther_and_cosine_has_a_certainty() {
        let (east, north, west) = ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]);
        let near_east = [2.0, 0.1];
        for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
            assert!(distance(metric, &east, &near_east) < distance(metric, &east, &north));
            assert!(distance(metric, &east, &north) < distance(metric, &east, &west));
        }
        let scaled = [5.0, 0.0];
        assert_eq!(
            certainty(Distance::Cosine, distance(Distance::Cosine, &east, &scaled)),
            Some(1.0)
        );
        assert_eq!(
            certainty(Distance::Cosine, distance(Distance::Cosine, &east, &north)),
            Some(0.5)
        );
        assert_eq!(
            certainty(Distance::Cosine, distance(Distance::Cosine, &east, &west)),
            Some(0.0)
        );
        assert_eq!(distance(Distance::Cosine, &east, &[0.0, 0.0]), 1.0);
        assert_eq!(certainty(Distance::Dot, 0.3), None);
        assert_eq!(certainty(Distance::L2, 0.3), None);
    }

    fn indexed(metric: Distance, vectors: &[Vec<f32>]) -> BinaryIndex {
        let rows: Vec<(u32, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(row, v)| (row as u32 * 2, v.as_slice()))
            .collect();
        BinaryIndex::build(metric, &rows).unwrap()
    }

    /// Vectors about twenty random centres, `spread` from them.
    fn grouped(random: &mut Random, count: usize, dimensions: usize, spread: f32) -> Vec<Vec<f32>> {
        let centres: Vec<Vec<f32>> = (0..20).map(|_| random.vector(dimensions)).collect();
        (0..count)
            .map(|n| {
                let noise = random.vector(dimensions);
                centres[n % 20]
                    .iter()
                    .zip(noise)
                    .map(|(c, e)| c + spread * e)
                    .collect()
            })
            .collect()
    }

    /// Of the true nearest ten of every `step`th vector, nudged, how many a
    /// shortlist of `count` holds, as a share.
    fn recall(metric: Distance, vectors: &[Vec<f32>], step: usize, count: usize) -> f64 {
        let index = indexed(metric, vectors);
        let mut random = Random(99);
        let (mut found, mut total) = (0, 0);
        for query in vectors.iter().step_by(step) {
            let query: Vec<f32> = query.iter().map(|v| v + 0.05 * random.unit()).collect();
            let mut exact: Vec<(f32, u32)> = vectors
                .iter()
                .enumerate()
                .map(|(row, v)| (distance(metric, &query, v), row as u32 * 2))
                .collect();
            exact.sort_by(|a, b| a.0.total_cmp(&b.0));
            let shortlist = index.shortlist(&query, count, None, &[]);
            assert_eq!(shortlist.len(), count.min(vectors.len()));
            found += exact[..10]
                .iter()
                .filter(|(_, row)| shortlist.contains(row))
                .count();
            total += 10;
        }
        found as f64 / total as f64
    }

    #[test]
    fn a_shortlist_holds_the_true_neighbours_of_clustered_vectors() {
        let mut random = Random(42);
        let vectors = grouped(&mut random, 4000, 256, 0.3);
        assert_eq!(indexed(Distance::Cosine, &vectors).len(), 4000);
        let share = recall(Distance::Cosine, &vectors, 200, 160);
        assert!(share >= 0.95, "{share} of the true neighbours shortlisted");
    }

    #[test]
    fn codes_are_taken_about_the_mean_so_an_offset_shared_by_all_vectors_costs_nothing() {
        let mut random = Random(9);
        let vectors: Vec<Vec<f32>> = grouped(&mut random, 4000, 256, 0.3)
            .into_iter()
            .map(|v| v.into_iter().map(|x| x + 5.0).collect())
            .collect();
        let share = recall(Distance::L2, &vectors, 200, 160);
        assert!(share >= 0.9, "{share} of the true neighbours shortlisted");
    }

    /// Lengths spread over a range of sixteen to one, about an offset, as
    /// vectors that are not normalised may be: the shortlist a search at
    /// limit ten takes holds nearly all the true nearest ten.
    #[test]
    fn a_shortlist_holds_the_true_neighbours_of_vectors_of_any_length() {
        let mut random = Random(5);
        let scaled: Vec<Vec<f32>> = grouped(&mut random, 10_000, 128, 0.6)
            .into_iter()
            .map(|v| {
                let length = 0.25 + 3.75 * (random.unit() + 0.5);
                v.into_iter().map(|x| x * length).collect()
            })
            .collect();
        let offset: Vec<Vec<f32>> = scaled
            .iter()
            .map(|v| v.iter().map(|x| x + 1.0).collect())
            .collect();
        for (data, vectors) in [("scaled", &scaled), ("offset", &offset)] {
            for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
                let share = recall(metric, vectors, 100, 1280);
                assert!(share >= 0.99, "{data} {metric:?}: recall@10 {share}");
            }
        }
    }

    #[test]
    fn the_ranked_angle_rises_with_the_hamming_distance_from_none_at_all() {
        for dimensions in [1, 64, 768] {
            let cosines: Vec<f32> = (0..=dimensions)
                .map(|hamming| ranked_cosine(hamming, dimensions))
                .collect();
            assert_eq!(cosines[0], 1.0);
            assert!(cosines.windows(2).all(|pair| pair[1] < pair[0]));
            // Half the bits differing measures a right angle; the rank is
            // nearer than that.
            assert!(cosines[dimensions / 2] > 0.0);
        }
    }

    #[test]
    fn lengths_that_overflow_or_vanish_give_a_finite_cosine_distance() {
        for vector in [[1e30f32, 0.0], [1e-30, 0.0], [0.0, 0.0]] {
            let distance = distance(Distance::Cosine, &vector, &vector);
            assert!(distance.is_finite(), "{vector:?} gave {distance}");
        }
    }

    #[test]
    fn a_distance_that_is_not_a_number_is_the_farthest() {
        let huge = [1e30f32, 1e30];
        assert_eq!(
            distance(Distance::Dot, &huge, &[1e30, -1e30]),
            f32::INFINITY
        );
        assert_eq!(
            distance(Distance::L2, &[f32::INFINITY], &[f32::INFINITY]),
            f32::INFINITY
        );
    }

    #[test]
    fn a_shortlist_keeps_to_the_rows_it_is_allowed() {
        let mut random = Random(3);
        let vectors: Vec<Vec<f32>> = (0..100).map(|_| random.vector(64)).collect();
        let index = indexed(Distance::Cosine, &vectors);
        let query = &vectors[0];
        let shortlist = |count: usize, only: Option<&RoaringBitmap>, skipped: &[bool]| {
            let mut rows = index.shortlist(query, count, only, skipped);
            rows.sort_unstable();
            rows
        };

        let only: RoaringBitmap = [0u32, 2, 3, 40, 198, 999].into_iter().collect();
        assert_eq!(
            shortlist(50, Some(&only), &[]),
            [0, 2, 40, 198],
            "rows 3 and 999 are not indexed"
        );

        let mut skipped = vec![false; 200];
        skipped[0] = true;
        skipped[40] = true;
        assert_eq!(shortlist(50, Some(&only), &skipped), [2, 198]);

        assert_eq!(shortlist(1000, None, &skipped).len(), 98);
        assert_eq!(shortlist(0, None, &[]), Vec::<u32>::new());
        assert_eq!(
            index.shortlist(&query[..10], 5, None, &[]),
            Vec::<u32>::new()
        );
        assert!(BinaryIndex::build(Distance::Cosine, &[]).is_none());
    }
}
