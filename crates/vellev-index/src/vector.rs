//! Nearest-vector search by a compressed scan and an exact rescore.
//!
//! A [`BinaryIndex`] keeps one bit per dimension of every vector: whether it
//! lies above the mean of the indexed vectors in that dimension. Comparing
//! two such codes costs a few XOR and popcount instructions, so a scan over
//! every code produces a shortlist cheaply, and only the shortlist is scored
//! with the full vectors and the real distance.
//!
//! The shortlist is the only approximate step, and it only ever considers
//! rows the caller allows, so a filter cannot cause a miss.

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

/// Measures how far vectors are from one query; smaller is nearer. Cosine
/// distance is one minus the cosine, and one for a zero vector, which has no
/// direction.
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
                let norms = self.query_norm * norm.sqrt();
                if norms > 0.0 {
                    1.0 - (dot / norms).clamp(-1.0, 1.0)
                } else {
                    1.0
                }
            }
            Distance::Dot => -dot(self.query, vector),
            Distance::L2 => squared_l2(self.query, vector),
        }
    }

    /// A cosine distance on the scale `(1 + cosine) / 2`, from 0 for opposite
    /// to 1 for identical. Other metrics have no such scale.
    pub fn certainty(&self, distance: f32) -> Option<f32> {
        (self.metric == Distance::Cosine).then_some(1.0 - distance / 2.0)
    }
}

/// How far `a` is from `b` under `metric`.
pub fn distance(metric: Distance, a: &[f32], b: &[f32]) -> f32 {
    Scorer::new(metric, a).distance(b)
}

/// The certainty of a distance under `metric`; see [`Scorer::certainty`].
pub fn certainty(metric: Distance, distance: f32) -> Option<f32> {
    Scorer::new(metric, &[]).certainty(distance)
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
#[derive(Debug, Clone)]
pub struct BinaryIndex {
    mean: Vec<f32>,
    words: usize,
    /// The indexed rows in ascending order; a row's code is at its position.
    rows: Vec<u32>,
    codes: Vec<u64>,
}

impl BinaryIndex {
    /// Index `vectors`, which must be in ascending row order and all of one
    /// length. Returns `None` when there is nothing to index.
    pub fn build(vectors: &[(u32, &[f32])]) -> Option<Self> {
        let dimensions = vectors.first()?.1.len();
        let count = vectors.len() as f32;
        let mean = vectors
            .iter()
            .fold(vec![0.0f32; dimensions], |mut sum, (_, vector)| {
                sum.iter_mut()
                    .zip(*vector)
                    .for_each(|(sum, value)| *sum += value / count);
                sum
            });
        let words = dimensions.div_ceil(64);
        let mut index = Self {
            mean,
            words,
            rows: vectors.iter().map(|(row, _)| *row).collect(),
            codes: Vec::with_capacity(vectors.len() * words),
        };
        for (_, vector) in vectors {
            let code = index.encode(vector);
            index.codes.extend(code);
        }
        Some(index)
    }

    fn encode(&self, vector: &[f32]) -> Vec<u64> {
        let mut code = vec![0u64; self.words];
        let above = vector
            .iter()
            .zip(&self.mean)
            .enumerate()
            .filter(|(_, (v, mean))| v > mean);
        above.for_each(|(dimension, _)| code[dimension / 64] |= 1 << (dimension % 64));
        code
    }

    fn hamming(&self, position: usize, query: &[u64]) -> u32 {
        let code = &self.codes[position * self.words..(position + 1) * self.words];
        code.iter()
            .zip(query)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
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
        let query = self.encode(query);
        let kept = |row: u32| !skipped.get(row as usize).copied().unwrap_or(false);
        let scored = |position: usize| (self.hamming(position, &query), self.rows[position]);
        let mut nearest: Vec<(u32, u32)> = match only {
            None => (0..self.rows.len())
                .filter(|p| kept(self.rows[*p]))
                .map(scored)
                .collect(),
            Some(only) => only
                .iter()
                .filter(|row| kept(*row))
                .filter_map(|row| self.rows.binary_search(&row).ok())
                .map(scored)
                .collect(),
        };
        if nearest.len() > count {
            nearest.select_nth_unstable(count);
            nearest.truncate(count);
        }
        nearest.into_iter().map(|(_, row)| row).collect()
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

    fn indexed(vectors: &[Vec<f32>]) -> BinaryIndex {
        let rows: Vec<(u32, &[f32])> = vectors
            .iter()
            .enumerate()
            .map(|(row, v)| (row as u32 * 2, v.as_slice()))
            .collect();
        BinaryIndex::build(&rows).unwrap()
    }

    #[test]
    fn a_shortlist_holds_the_true_neighbours_of_clustered_vectors() {
        let mut random = Random(42);
        let centres: Vec<Vec<f32>> = (0..20).map(|_| random.vector(256)).collect();
        let vectors: Vec<Vec<f32>> = (0..4000)
            .map(|n| {
                let noise = random.vector(256);
                centres[n % 20]
                    .iter()
                    .zip(noise)
                    .map(|(c, e)| c + 0.3 * e)
                    .collect()
            })
            .collect();
        let index = indexed(&vectors);
        assert_eq!(index.len(), 4000);

        let mut found = 0;
        for query in vectors.iter().step_by(200) {
            let mut exact: Vec<(f32, u32)> = vectors
                .iter()
                .enumerate()
                .map(|(row, v)| (distance(Distance::Cosine, query, v), row as u32 * 2))
                .collect();
            exact.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let shortlist = index.shortlist(query, 160, None, &[]);
            assert_eq!(shortlist.len(), 160);
            found += exact[..10]
                .iter()
                .filter(|(_, row)| shortlist.contains(row))
                .count();
        }
        assert!(found >= 190, "{found} of 200 true neighbours shortlisted");
    }

    #[test]
    fn a_shortlist_keeps_to_the_rows_it_is_allowed() {
        let mut random = Random(3);
        let vectors: Vec<Vec<f32>> = (0..100).map(|_| random.vector(64)).collect();
        let index = indexed(&vectors);
        let query = &vectors[0];

        let only: RoaringBitmap = [0u32, 2, 3, 40, 198, 999].into_iter().collect();
        let mut allowed = index.shortlist(query, 50, Some(&only), &[]);
        allowed.sort_unstable();
        assert_eq!(allowed, [0, 2, 40, 198], "rows 3 and 999 are not indexed");

        let mut skipped = vec![false; 200];
        skipped[0] = true;
        skipped[40] = true;
        let mut kept = index.shortlist(query, 50, Some(&only), &skipped);
        kept.sort_unstable();
        assert_eq!(kept, [2, 198]);

        let everything = index.shortlist(query, 1000, None, &skipped);
        assert_eq!(everything.len(), 98);
        assert_eq!(index.shortlist(query, 0, None, &[]), Vec::<u32>::new());
        assert_eq!(
            index.shortlist(&query[..10], 5, None, &[]),
            Vec::<u32>::new()
        );
        assert!(BinaryIndex::build(&[]).is_none());
    }
}
