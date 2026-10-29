//! Nearest-vector search by a compressed scan and an exact rescore.
//!
//! A [`BinaryIndex`] keeps one bit per coordinate of every vector, once
//! turned by a random rotation: whether it lies above a centre of the
//! indexed vectors in that coordinate. Comparing
//! two such codes costs a few XOR and popcount instructions, so a scan over
//! every code produces a shortlist cheaply, and only the shortlist is scored
//! with the full vectors and the real distance.
//!
//! The shortlist is the only approximate step. It is taken from the rows the
//! caller allows, not filtered afterwards, so a filter never thins it.

use std::borrow::Cow;
use std::cmp::Ordering;

use rayon::prelude::*;
use roaring::RoaringBitmap;
use vellev_core::{Distance, SQUARED_LENGTH_UNDER};

use crate::rotation::{Random, Rotation};

const LANES: usize = 16;

/// The dot product, accumulated in independent lanes so that it compiles to
/// wide vector instructions without any being named; on x86-64 with AVX2,
/// in those instructions, to the same result bit for bit (see [`f32s`]).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Of other lengths the ways would pair other elements in the tail.
    debug_assert_eq!(a.len(), b.len());
    (f32s::chosen().dot)(a, b)
}

pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    // Of other lengths the ways would pair other elements in the tail.
    debug_assert_eq!(a.len(), b.len());
    (f32s::chosen().squared_l2)(a, b)
}

/// The dot product of `a` and `b` together with `b`'s squared length, in one
/// pass over `b`. Both are summed in the same order, so that of a vector
/// against itself they agree to the last bit; it is not the order [`dot`]
/// sums in. Any shape of the final sum but this one makes the loop several
/// times slower, so a squared length to set beside these must be taken
/// from here too.
fn dot_and_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
    // Of other lengths the ways would pair other elements in the tail.
    debug_assert_eq!(a.len(), b.len());
    (f32s::chosen().dot_and_norm)(a, b)
}

/// The sums over `f32` vectors every distance is made of, in each way this
/// machine can run them, every way giving the portable way's result to the
/// last bit: each lane multiplies, then adds, rounding after each as the
/// portable lanes do, and the lanes and the tail are summed in the same
/// order. Fused multiply-adds round once instead, and so are not used.
pub(crate) mod f32s {
    use super::LANES;

    /// A way of making the three sums.
    #[derive(Clone, Copy)]
    pub(crate) struct Way {
        /// What the rules call it.
        #[cfg_attr(not(test), allow(dead_code))]
        pub name: &'static str,
        pub dot: fn(&[f32], &[f32]) -> f32,
        pub squared_l2: fn(&[f32], &[f32]) -> f32,
        pub dot_and_norm: fn(&[f32], &[f32]) -> (f32, f32),
    }

    /// The portable way, which every other is held to.
    pub(crate) const PORTABLE: Way = Way {
        name: "portable",
        dot: |a, b| lanes(a, b, |x, y| x * y),
        squared_l2: |a, b| lanes(a, b, |x, y| (x - y) * (x - y)),
        dot_and_norm: dot_and_norm_portably,
    };

    /// Every way this machine can run, the portable first.
    pub(crate) fn ways_here() -> Vec<Way> {
        #[allow(unused_mut)]
        let mut ways = vec![PORTABLE];
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            ways.push(Way {
                name: "avx2",
                // SAFETY: handed out only where the CPU has just said it
                // has AVX2, which is all these ask.
                dot: |a, b| unsafe { avx2::dot(a, b) },
                squared_l2: |a, b| unsafe { avx2::squared_l2(a, b) },
                dot_and_norm: |a, b| unsafe { avx2::dot_and_norm(a, b) },
            });
        }
        ways
    }

    /// The way chosen for this machine: the last, so the widest, of
    /// [`ways_here`]. Decided once a process. On aarch64 the portable
    /// way already compiles to NEON, which every aarch64 CPU has.
    pub(crate) fn chosen() -> &'static Way {
        static CHOSEN: std::sync::OnceLock<Way> = std::sync::OnceLock::new();
        CHOSEN.get_or_init(|| *ways_here().last().expect("the portable way is always here"))
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

    fn dot_and_norm_portably(a: &[f32], b: &[f32]) -> (f32, f32) {
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

    /// The sums in AVX2: the sixteen lanes as two registers of eight, each
    /// lane a product (or a difference squared) added after rounding, the
    /// lanes then summed in order in `f32`, and the tail as the portable
    /// way sums it.
    #[cfg(target_arch = "x86_64")]
    mod avx2 {
        use std::arch::x86_64::*;

        use super::LANES;

        /// The sixteen lanes of `term` over the whole chunks of `a` and `b`,
        /// and how many values they took.
        #[target_feature(enable = "avx2")]
        fn lanes(
            a: &[f32],
            b: &[f32],
            term: impl Fn(__m256, __m256) -> __m256,
        ) -> ([f32; LANES], usize) {
            let whole = a.len().min(b.len()) / LANES * LANES;
            let (mut low, mut high) = (_mm256_setzero_ps(), _mm256_setzero_ps());
            let mut at = 0;
            while at < whole {
                // SAFETY: `at + 16 <= whole`, which neither slice is shorter
                // than; the loads ask no alignment.
                let (xl, xh, yl, yh) = unsafe {
                    (
                        _mm256_loadu_ps(a.as_ptr().add(at)),
                        _mm256_loadu_ps(a.as_ptr().add(at + 8)),
                        _mm256_loadu_ps(b.as_ptr().add(at)),
                        _mm256_loadu_ps(b.as_ptr().add(at + 8)),
                    )
                };
                low = _mm256_add_ps(low, term(xl, yl));
                high = _mm256_add_ps(high, term(xh, yh));
                at += LANES;
            }
            let mut sums = [0.0f32; LANES];
            // SAFETY: `sums` holds sixteen values; the stores ask no
            // alignment.
            unsafe {
                _mm256_storeu_ps(sums.as_mut_ptr(), low);
                _mm256_storeu_ps(sums.as_mut_ptr().add(8), high);
            }
            (sums, whole)
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
            let (sums, whole) = lanes(a, b, |x, y| _mm256_mul_ps(x, y));
            let tail: f32 = a[whole..].iter().zip(&b[whole..]).map(|(x, y)| x * y).sum();
            sums.iter().sum::<f32>() + tail
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
            let (sums, whole) = lanes(a, b, |x, y| {
                let d = _mm256_sub_ps(x, y);
                _mm256_mul_ps(d, d)
            });
            let tail: f32 = a[whole..]
                .iter()
                .zip(&b[whole..])
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            sums.iter().sum::<f32>() + tail
        }

        #[target_feature(enable = "avx2")]
        pub(super) fn dot_and_norm(a: &[f32], b: &[f32]) -> (f32, f32) {
            let (dots, whole) = lanes(a, b, |x, y| _mm256_mul_ps(x, y));
            let (norms, _) = lanes(a, b, |_, y| _mm256_mul_ps(y, y));
            let tail = a[whole..]
                .iter()
                .zip(&b[whole..])
                .fold((0.0f32, 0.0f32), |(dot, norm), (x, y)| {
                    (dot + x * y, norm + y * y)
                });
            dots.iter()
                .zip(&norms)
                .fold(tail, |(dot, norm), (d, n)| (dot + d, norm + n))
        }
    }
}

/// As [`dot_and_norm`], in `f64`, where neither sum can overflow or vanish
/// whatever finite values the vectors hold: for a vector whose squared
/// length an `f32` cannot hold in full, being too long or too short.
fn wide_dot_and_norm(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (a_chunks, b_chunks) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail = a_chunks.remainder().iter().zip(b_chunks.remainder()).fold(
        (0.0, 0.0),
        |(dot, norm), (x, y)| {
            let (x, y) = (f64::from(*x), f64::from(*y));
            (dot + x * y, norm + y * y)
        },
    );
    let sums = a_chunks
        .zip(b_chunks)
        .fold([(0.0f64, 0.0f64); LANES], |mut sums, (x, y)| {
            for lane in 0..LANES {
                let (x, y) = (f64::from(x[lane]), f64::from(y[lane]));
                sums[lane].0 += x * y;
                sums[lane].1 += y * y;
            }
            sums
        });
    sums.iter()
        .fold(tail, |(dot, norm), (d, n)| (dot + d, norm + n))
}

/// `distance`, or the farthest there is when it is not a number: a sort
/// cannot be trusted to put a NaN last, as the sign it comes with varies.
/// A negative zero comes back as zero, which a sort by total order would
/// otherwise put before it.
pub(crate) fn or_farthest(distance: f32) -> f32 {
    if distance.is_nan() {
        f32::INFINITY
    } else {
        distance + 0.0
    }
}

/// `vector` scaled to unit length, or `None` when it has no direction:
/// when every value is zero, or one is not finite. A vector whose squared
/// length an `f32` cannot hold in full, being too long or too short, is
/// scaled by its largest value first, so that every other vector has a
/// direction, whatever its length.
pub(crate) fn unit(vector: &[f32]) -> Option<Vec<f32>> {
    let scaled = |vector: &[f32]| {
        let squared = dot(vector, vector);
        squared.is_normal().then(|| {
            let norm = squared.sqrt();
            vector.iter().map(|value| value / norm).collect()
        })
    };
    scaled(vector).or_else(|| {
        let largest = vector
            .iter()
            .fold(0.0f32, |most, value| most.max(value.abs()));
        (largest > 0.0 && largest.is_finite())
            .then(|| {
                vector
                    .iter()
                    .map(|value| value / largest)
                    .collect::<Vec<f32>>()
            })
            .and_then(|vector| scaled(&vector))
    })
}

/// Measures how far vectors are from one query; smaller is nearer. Cosine
/// distance is one minus the cosine, and one for a vector with no direction
/// (see [`unit`]). A distance that is not a number is taken as the
/// farthest; vectors no longer than a collection accepts never produce one.
#[derive(Debug, Clone)]
pub struct Scorer<'q> {
    metric: Distance,
    query: &'q [f32],
    /// What a cosine is worked out from on the query's side.
    measure: Measure,
}

#[derive(Debug, Clone, Copy)]
enum Measure {
    /// The query's squared length, a normal `f32`.
    Squared(f32),
    /// The query's squared length as only an `f64` holds it in full.
    Wide(f64),
    /// The query has no direction.
    Directionless,
}

/// The cosine of the angle between two vectors with the inner product
/// `along` and the squared lengths `a` and `b`. The root of a square is the
/// number squared, to the bit, so of a vector against itself, whose inner
/// product is its squared length summed the same way, it is exactly one.
fn cosine(along: f64, a: f64, b: f64) -> f32 {
    (along / (a * b).sqrt()) as f32
}

impl<'q> Scorer<'q> {
    pub fn new(metric: Distance, query: &'q [f32]) -> Self {
        // Summed as each row's squared length is, so that against itself
        // the query's inner product is its squared length to the bit.
        let (_, squared) = dot_and_norm(query, query);
        let measure = match squared.is_normal() {
            true => Measure::Squared(squared),
            false => match wide_dot_and_norm(query, query).1 {
                wide if wide > 0.0 && wide.is_finite() => Measure::Wide(wide),
                _ => Measure::Directionless,
            },
        };
        Self {
            metric,
            query,
            measure,
        }
    }

    /// The cosine of the angle between the query, of the squared length
    /// `query`, and `vector`, worked out in `f64`: for when either's
    /// squared length is more or less than an `f32` holds in full. `None`
    /// when `vector` has no direction.
    fn wide_cosine(&self, query: f64, vector: &[f32]) -> Option<f32> {
        // A zero vector is the likeliest to come here, and need not be
        // summed again: no value of it has a bit set but its sign. Told so,
        // without a branch a value, rather than by comparing each to zero,
        // which stops at the first that is not and so cannot be vectorised.
        let set = vector
            .iter()
            .fold(0, |set, value| set | (value.to_bits() << 1));
        if set == 0 {
            return None;
        }
        let (along, squared) = wide_dot_and_norm(self.query, vector);
        squared.is_finite().then(|| cosine(along, query, squared))
    }

    pub fn distance(&self, vector: &[f32]) -> f32 {
        match self.metric {
            Distance::Cosine => {
                let cosine = match self.measure {
                    Measure::Squared(query) => {
                        // Straight from the squares where the row's is a
                        // normal `f32` too, as nearly every one is.
                        let (along, squared) = dot_and_norm(self.query, vector);
                        match squared.is_normal() {
                            true => Some(cosine(along.into(), query.into(), squared.into())),
                            false => self.wide_cosine(query.into(), vector),
                        }
                    }
                    Measure::Wide(query) => self.wide_cosine(query, vector),
                    Measure::Directionless => None,
                };
                cosine.map_or(1.0, |cosine| 1.0 - cosine.clamp(-1.0, 1.0))
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
/// that only its direction counts, and as it is otherwise. A vector with no
/// direction is left as it is.
pub(crate) fn normalised(metric: Distance, vector: &[f32]) -> Cow<'_, [f32]> {
    match metric {
        Distance::Cosine => unit(vector).map_or(Cow::Borrowed(vector), Cow::Owned),
        Distance::Dot | Distance::L2 => Cow::Borrowed(vector),
    }
}

/// Whether `query`, measured by `metric`, is as far from every vector: a
/// cosine query with no direction, at distance one from each, or a zero
/// query under the inner product, at distance zero from each.
pub(crate) fn as_far_from_every_vector(metric: Distance, query: &[f32]) -> bool {
    match metric {
        Distance::Cosine => unit(query).is_none(),
        Distance::Dot => query.iter().all(|value| *value == 0.0),
        Distance::L2 => false,
    }
}

/// How to order two rows whose estimated distances are equal.
pub type Ties<'a> = &'a (dyn Fn(u32, u32) -> Ordering + Sync);

/// Rows of equal estimate in row order, for a caller with no order of its
/// own.
pub fn in_row_order(a: u32, b: u32) -> Ordering {
    a.cmp(&b)
}

/// Keep in `rows` the `count` with the nearest estimates, those of equal
/// estimate ordered by `ties`, returning the others. Neither part is in any
/// order.
pub fn nearest<T>(
    rows: &mut Vec<(f32, T)>,
    count: usize,
    ties: impl Fn(&T, &T) -> Ordering,
) -> Vec<(f32, T)> {
    if rows.len() <= count {
        return Vec::new();
    }
    rows.select_nth_unstable_by(count, |a, b| {
        a.0.total_cmp(&b.0).then_with(|| ties(&a.1, &b.1))
    });
    rows.split_off(count)
}

/// A way to pick, cheaply, the rows most likely to be nearest a query.
pub trait VectorIndex {
    /// How many rows are indexed.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Up to `count` rows likely to be nearest `query`, in no particular
    /// order, each with its estimated distance, taken only from `only` when
    /// it is given and never from a row flagged in `skipped`, which is
    /// indexed by row. Where not every row of one estimate fits, those
    /// first by `ties` are taken: rows the index cannot tell apart, such as
    /// those of one vector, come in the caller's order. Estimates order the
    /// rows of one index; they are comparable with another index's only
    /// where the index says so.
    fn estimates(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
        ties: Ties<'_>,
    ) -> Vec<(f32, u32)>;

    /// The rows the index holds but has no estimate for, as their vectors
    /// are too long to measure: only a vector stored before lengths were
    /// limited can be. They are never among the estimates, so they never
    /// take a place a measured row could have; a caller scores them
    /// exactly, taken from `only` and `skipped` as estimates are.
    fn unmeasured(&self, only: Option<&RoaringBitmap>, skipped: &[bool]) -> Vec<u32>;

    /// The rows of [`VectorIndex::estimates`], without their estimates.
    fn shortlist(
        &self,
        query: &[f32],
        count: usize,
        only: Option<&RoaringBitmap>,
        skipped: &[bool],
        ties: Ties<'_>,
    ) -> Vec<u32> {
        self.estimates(query, count, only, skipped, ties)
            .into_iter()
            .map(|(_, row)| row)
            .collect()
    }
}

/// Seeds the rotation and the sample the centre is taken from, which must
/// come out the same on every load.
const ROTATION_SEED: u64 = 0x0B17_C0DE;
/// How many vectors, at most, the centre is the median of.
const CENTRE_SAMPLE: usize = 4096;

/// One bit per coordinate of each vector after a random rotation, for a
/// scan by Hamming distance.
///
/// The rotation makes each bit a sign along a random direction, which meets
/// every axis, rather than along an axis: a vector that lies along a few
/// axes is coded as faithfully as one spread over all of them. Every
/// coordinate of the rotation is coded, the dimensions padded to a power of
/// two, as the padding's bits buy more recall than they cost.
///
/// None of this is free. Against codes along the axes about the mean, over
/// 100,000 rows of 384 dimensions a query takes about 40% longer, 0.70 to
/// 0.73 ms against 0.50 to 0.53, for the padding to 512 bits and each row's
/// factors to load, and the index 31 to 41 ms to build against 19 to 25.
/// It buys recall: on driftwerk's embeddings an oversample of 4 finds 0.990
/// of the true nearest ten, where codes along the axes find 0.881 even at
/// an oversample of 32.
///
/// A row's code says on which side of a centre `m` each coordinate of its
/// rotated vector `o` lies, and the query's code the same of the query `q`.
/// The share of bits on which they differ estimates the angle between
/// `o - m` and `q - m` as a share of a half turn. Whatever `m` is, each
/// metric's distance is, up to a term the same for every row, a term of the
/// row's own, kept beside its code, less a multiple of `|o - m| |q - m|`
/// times that angle's cosine. A row's estimate so weighs the angle by the
/// row's length, without which a long vector is lost among short ones. The
/// centre is the median, coordinate by coordinate, of a sample of the
/// vectors: unlike their mean, it stays among them however far a few lie
/// from the rest, so one such vector cannot spoil every other row's code.
///
/// The error of that estimate grows with `|o - m| |q - m|` too, so a long
/// vector that looks nearly at right angles to the query may still be
/// nearer than every short one. Each row is ranked by the angle at the near
/// end of the one-standard-error (Wilson) interval about the measured one,
/// which is to say by how near it may well be. That end rises with the
/// Hamming distance, so of two rows with the same term and length the one
/// fewer bits away ranks nearer.
///
/// For cosine the vectors and the query are scaled to unit length first,
/// so that only directions count.
///
/// Every coded vector's squared length is under `L`, the write limit with
/// the index's margin, `17/16 · 2^124`, and the query's under `2^124`. In
/// each coordinate at least half the sampled values are at least as large
/// as their median, so the centre's squared length is at most twice their
/// mean square, under `2L`. Then `|o - m| < √L + √(2L) < 2^63.4` and
/// `|o - m|² < 2^126.7`, and `|q - m| < 2^63.3`. The L2 estimate, the first
/// less `2 |q - m| |o - m|` times a cosine, lies between `-2^127.7` and
/// `2^128.3`: it is never `-∞`, but for a row facing away from the query
/// near the limit it may come out as `+∞`, which ranks that row last. The
/// inner products' estimates stay under `2^127.2` either way. No estimate
/// is NaN: an infinity can only be the last sum taken.
#[derive(Debug, Clone)]
pub struct BinaryIndex {
    metric: Distance,
    dimensions: usize,
    rotation: Rotation,
    /// `m`, rotated as the vectors are.
    centre: Vec<f32>,
    words: usize,
    /// The coded rows in ascending order; a row's code is at its position.
    rows: Vec<u32>,
    codes: Vec<u64>,
    /// Each row's own term of the distance, `|o - m|²` for L2 and
    /// `-m · (o - m)` for the inner products, and its `|o - m|`.
    factors: Vec<(f32, f32)>,
    /// The rows too long to code, in ascending order.
    unmeasured: Vec<u32>,
    /// By Hamming distance, the cosine of the angle a row is ranked by.
    cosines: Vec<f32>,
}

/// One row's code and factors, as [`BinaryIndex::code`] works them out.
#[derive(Debug, Clone, PartialEq)]
pub struct Coded {
    words: Vec<u64>,
    factors: (f32, f32),
}

impl BinaryIndex {
    /// Index `vectors`, which must be in ascending row order and all of one
    /// length, for search under `metric`. Returns `None` when there is
    /// nothing to index.
    /// As [`Self::build`], with codes taken about `centre` (in the rotated
    /// space) rather than about the vectors' own median.
    pub fn build_about(
        metric: Distance,
        vectors: &[(u32, &[f32])],
        centre: Vec<f32>,
    ) -> Option<Self> {
        let dimensions = vectors.first()?.1.len();
        debug_assert!(vectors.windows(2).all(|pair| pair[0].0 < pair[1].0));
        debug_assert!(vectors.iter().all(|(_, vector)| vector.len() == dimensions));
        let mut index = Self::empty(metric, dimensions, centre);
        let (measured, unmeasured): (Vec<&(u32, &[f32])>, _) = vectors
            .iter()
            .partition(|(_, vector)| measurable(metric, vector));
        let words = index.words;
        let mut codes = vec![0u64; measured.len() * words];
        let (rotation, centre) = (&index.rotation, &index.centre);
        let factors = codes
            .par_chunks_mut(words)
            .zip(measured.par_iter())
            .map(|(code, (_, vector))| {
                let vector = rotation.apply(&normalised(metric, vector));
                code.iter_mut()
                    .zip(encode(centre, &vector))
                    .for_each(|(slot, word)| *slot = word);
                let (squared, along) = residual(centre, &vector);
                let term = match metric {
                    Distance::L2 => squared,
                    Distance::Dot | Distance::Cosine => -along,
                };
                (term, squared.sqrt())
            })
            .collect();
        index.rows = measured.iter().map(|(row, _)| *row).collect();
        index.codes = codes;
        index.factors = factors;
        index.unmeasured = unmeasured.iter().map(|(row, _)| *row).collect();
        Some(index)
    }

    /// An index of `dimensions`, of no rows yet, its codes taken about
    /// `centre`.
    fn empty(metric: Distance, dimensions: usize, centre: Vec<f32>) -> Self {
        let rotation = Rotation::new(dimensions, ROTATION_SEED);
        debug_assert_eq!(centre.len(), rotation.width);
        let bits = rotation.width;
        Self {
            metric,
            dimensions,
            words: rotation.width / 64,
            rotation,
            centre,
            rows: Vec::new(),
            codes: Vec::new(),
            factors: Vec::new(),
            unmeasured: Vec::new(),
            cosines: (0..=bits)
                .map(|hamming| ranked_cosine(hamming, bits))
                .collect(),
        }
    }

    pub fn build(metric: Distance, vectors: &[(u32, &[f32])]) -> Option<Self> {
        let dimensions = vectors.first()?.1.len();
        let rotation = Rotation::new(dimensions, ROTATION_SEED);
        let measured: Vec<&[f32]> = vectors
            .iter()
            .map(|(_, vector)| *vector)
            .filter(|vector| measurable(metric, vector))
            .collect();
        let sampled: Vec<Vec<f32>> = Random(ROTATION_SEED)
            .sample(measured.len(), CENTRE_SAMPLE)
            .into_par_iter()
            .map(|at| rotation.apply(&normalised(metric, measured[at])))
            .collect();
        let centre = median(rotation.width, sampled.iter().map(Vec::as_slice));
        Self::build_about(metric, vectors, centre)
    }

    /// An index of no rows yet, its codes to be taken about the mean of
    /// `first`: for rows coded one at a time with [`Self::push`] as they
    /// come, such as those in memory since a checkpoint (burst plan F2b).
    /// The vectors, normalised as codes are, are summed in `f64` one value
    /// after another in the order given and rounded to `f32` once, so that
    /// the mean comes out the same to the bit on every machine; and that
    /// mean is rotated once into the space codes are taken in. The cost is
    /// a sum over `first` and one rotation, whatever its length.
    /// `None` when `first` has nothing to measure.
    pub fn about_mean(metric: Distance, first: &[&[f32]]) -> Option<Self> {
        let dimensions = first.first()?.len();
        let mut sums = vec![0.0f64; dimensions];
        let mut counted = 0usize;
        for vector in first.iter().filter(|vector| measurable(metric, vector)) {
            for (sum, value) in sums.iter_mut().zip(normalised(metric, vector).iter()) {
                *sum += f64::from(*value);
            }
            counted += 1;
        }
        if counted == 0 {
            return None;
        }
        let mean: Vec<f32> = sums
            .iter()
            .map(|sum| (sum / counted as f64) as f32)
            .collect();
        // The rotation is linear: the mean rotated once stands for the
        // rotated vectors' mean, at a sum's cost rather than a rotation for
        // every vector.
        let centre = Rotation::new(dimensions, ROTATION_SEED).apply(&mean);
        Some(Self::empty(metric, dimensions, centre))
    }

    /// Code `vector` as row `row`, which must come after every row coded
    /// so far.
    pub fn push(&mut self, row: u32, vector: &[f32]) {
        let coded = self.code(vector);
        self.push_coded(row, coded);
    }

    /// What [`Self::push`] stores for `vector`: its code and factors about
    /// this index's centre, or `None` for a vector too long to measure,
    /// which is scored exactly. Worked out apart from the index, so that it
    /// can be done before whatever holds the index is locked to change it.
    pub fn code(&self, vector: &[f32]) -> Option<Coded> {
        debug_assert_eq!(vector.len(), self.dimensions);
        if !measurable(self.metric, vector) {
            return None;
        }
        let rotated = self.rotation.apply(&normalised(self.metric, vector));
        let (squared, along) = residual(&self.centre, &rotated);
        let term = match self.metric {
            Distance::L2 => squared,
            Distance::Dot | Distance::Cosine => -along,
        };
        Some(Coded {
            words: encode(&self.centre, &rotated).collect(),
            factors: (term, squared.sqrt()),
        })
    }

    /// Store row `row`, which must come after every row taken so far, as
    /// [`Self::code`] coded it about this index's centre.
    pub fn push_coded(&mut self, row: u32, coded: Option<Coded>) {
        match coded {
            None => self.unmeasured.push(row),
            Some(Coded { words, factors }) => {
                debug_assert_eq!(words.len(), self.words);
                self.codes.extend(words);
                self.factors.push(factors);
                self.rows.push(row);
            }
        }
    }

    /// Take row `row`, which must come after every row taken so far, to be
    /// scored exactly beside whatever the codes shortlist, and never
    /// coded: the rows a memtable's centre is taken from (burst plan F2b),
    /// so that reaching the threshold codes none of them.
    pub fn push_exact(&mut self, row: u32) {
        self.unmeasured.push(row);
    }

    /// How many rows have a code.
    pub fn coded(&self) -> usize {
        self.rows.len()
    }

    /// A digest of every code, its factors and its row, and of the rows
    /// scored exactly: equal for two indexes coded alike, bit for bit.
    pub fn digest(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.rows.hash(&mut hasher);
        self.codes.hash(&mut hasher);
        for (term, length) in &self.factors {
            (term.to_bits(), length.to_bits()).hash(&mut hasher);
        }
        self.unmeasured.hash(&mut hasher);
        self.centre
            .iter()
            .map(|x| x.to_bits())
            .for_each(|x| x.hash(&mut hasher));
        hasher.finish()
    }

    /// The centre codes are taken about, in the rotated space.
    pub fn centre(&self) -> &[f32] {
        &self.centre
    }

    /// About how many bytes the index takes in memory.
    pub fn resident(&self) -> usize {
        use std::mem::size_of;
        self.centre.len() * size_of::<f32>()
            + self.rows.capacity() * size_of::<u32>()
            + self.codes.capacity() * size_of::<u64>()
            + self.factors.capacity() * size_of::<(f32, f32)>()
            + self.unmeasured.capacity() * size_of::<u32>()
            + self.cosines.len() * size_of::<f32>()
    }

    fn hamming(&self, position: usize, query: &[u64]) -> usize {
        let code = &self.codes[position * self.words..(position + 1) * self.words];
        code.iter()
            .zip(query)
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum()
    }
}

/// Over how much more than [`SQUARED_LENGTH_UNDER`] a vector's squared
/// length may come out and the vector still be measured. A write sums the
/// squares in one order, the index in another; the two differ by at most
/// the dimensions times an `f32`'s precision, under 2^-10 of the sum, so a
/// sixteenth is ample and every vector a write accepts is measured.
const MEASURABLE_UNDER: f32 = SQUARED_LENGTH_UNDER * (1.0 + 1.0 / 16.0);

/// Whether an index can measure `vector` under `metric`: always under
/// cosine, where only its direction counts, and otherwise when it is no
/// longer than a collection accepts. Only a vector stored before lengths
/// were limited can be longer.
pub(crate) fn measurable(metric: Distance, vector: &[f32]) -> bool {
    metric == Distance::Cosine || short_enough(f64::from(dot(vector, vector)))
}

/// Whether a vector of this squared length can be measured.
pub(crate) fn short_enough(squared: f64) -> bool {
    squared < f64::from(MEASURABLE_UNDER)
}

/// Of `rows`, in ascending order, those in `only` when it is given and not
/// flagged in `skipped`.
pub(crate) fn allowed(rows: &[u32], only: Option<&RoaringBitmap>, skipped: &[bool]) -> Vec<u32> {
    rows.iter()
        .copied()
        .filter(|row| !skipped.get(*row as usize).copied().unwrap_or(false))
        .filter(|row| only.is_none_or(|only| only.contains(*row)))
        .collect()
}

/// The median of `vectors` in each of their `width` coordinates, or the
/// origin when there are none. Unlike the mean, it stays among the vectors
/// however far a few of them lie from the rest.
fn median<'a>(width: usize, vectors: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let vectors: Vec<&[f32]> = vectors.collect();
    (0..width)
        .map(|coordinate| {
            let mut values: Vec<f32> = vectors.iter().map(|vector| vector[coordinate]).collect();
            let middle = values.len() / 2;
            match values.is_empty() {
                true => 0.0,
                false => *values.select_nth_unstable_by(middle, f32::total_cmp).1,
            }
        })
        .collect()
}

/// The squared length of `vector - centre`, and its inner product with
/// `centre`.
fn residual(centre: &[f32], vector: &[f32]) -> (f32, f32) {
    centre
        .iter()
        .zip(vector)
        .fold((0.0, 0.0), |(squared, along), (centre, value)| {
            let difference = value - centre;
            (
                squared + difference * difference,
                along + centre * difference,
            )
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

/// One bit per coordinate, set where the vector is above the centre. Built
/// without a branch per dimension, which a coin-flip bit would mispredict
/// half the time.
fn encode<'a>(centre: &'a [f32], vector: &'a [f32]) -> impl Iterator<Item = u64> + 'a {
    vector
        .chunks(64)
        .zip(centre.chunks(64))
        .map(|(values, centres)| {
            values
                .iter()
                .zip(centres)
                .enumerate()
                .fold(0u64, |word, (bit, (value, centre))| {
                    word | (u64::from(value > centre) << bit)
                })
        })
}

impl VectorIndex for BinaryIndex {
    fn len(&self) -> usize {
        self.rows.len() + self.unmeasured.len()
    }

    fn unmeasured(&self, only: Option<&RoaringBitmap>, skipped: &[bool]) -> Vec<u32> {
        allowed(&self.unmeasured, only, skipped)
    }

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
        // A query as far from every row is estimated alike for each, and
        // the caller's order decides.
        let as_far_from_every_row = as_far_from_every_vector(self.metric, query);
        let query = self.rotation.apply(&normalised(self.metric, query));
        let code: Vec<u64> = encode(&self.centre, &query).collect();
        let reach = residual(&self.centre, &query).0.sqrt()
            * match self.metric {
                Distance::L2 => 2.0,
                Distance::Dot | Distance::Cosine => 1.0,
            };
        let kept = |row: u32| !skipped.get(row as usize).copied().unwrap_or(false);
        let estimated = |position: usize| {
            let estimate = match as_far_from_every_row {
                true => 0.0,
                false => {
                    let (term, length) = self.factors[position];
                    let cosine = self.cosines[self.hamming(position, &code)];
                    or_farthest(term - reach * length * cosine)
                }
            };
            (estimate, self.rows[position])
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
        nearest(&mut estimates, count, |a, b| ties(*a, *b));
        estimates
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

    /// Of the true nearest ten among `vectors` of every `step`th of
    /// `queries`, nudged, how many a shortlist of `count` holds, as a share.
    fn recall(
        metric: Distance,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        step: usize,
        count: usize,
    ) -> f64 {
        let index = indexed(metric, vectors);
        let found: usize = queries
            .par_iter()
            .step_by(step)
            .enumerate()
            .map(|(at, query)| {
                let mut random = Random(99 + at as u64);
                let query: Vec<f32> = query.iter().map(|v| v + 0.05 * random.unit()).collect();
                let mut exact: Vec<(f32, u32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(row, v)| (distance(metric, &query, v), row as u32 * 2))
                    .collect();
                exact.select_nth_unstable_by(9, |a, b| a.0.total_cmp(&b.0));
                // What a caller scores: the shortlist, and the rows the index
                // has no estimate for.
                let mut scored = index.shortlist(&query, count, None, &[], &in_row_order);
                assert_eq!(scored.len(), count.min(index.rows.len()));
                scored.extend(index.unmeasured(None, &[]));
                exact[..10]
                    .iter()
                    .filter(|(_, row)| scored.contains(row))
                    .count()
            })
            .sum();
        found as f64 / (queries.len().div_ceil(step) * 10) as f64
    }

    /// Noise drawn independently along each axis is the one case where
    /// codes along the axes do better than along random directions: on this
    /// draw 0.945, against 0.99 without the rotation. The rotation is kept
    /// for what it does to real embeddings, whose dimensions are anything
    /// but independent: on driftwerk's, recall@10 at oversample 4 rose from
    /// 0.796 to 0.990 under cosine, and from 0.196 to 0.914 under dot.
    #[test]
    fn a_shortlist_holds_the_true_neighbours_of_clustered_vectors() {
        let mut random = Random(42);
        let vectors = grouped(&mut random, 4000, 256, 0.3);
        assert_eq!(indexed(Distance::Cosine, &vectors).len(), 4000);
        let share = recall(Distance::Cosine, &vectors, &vectors, 200, 160);
        assert!(share >= 0.92, "{share} of the true neighbours shortlisted");
    }

    #[test]
    fn codes_are_taken_about_a_centre_so_an_offset_shared_by_all_vectors_costs_nothing() {
        let mut random = Random(9);
        let vectors: Vec<Vec<f32>> = grouped(&mut random, 4000, 256, 0.3)
            .into_iter()
            .map(|v| v.into_iter().map(|x| x + 5.0).collect())
            .collect();
        let share = recall(Distance::L2, &vectors, &vectors, 200, 160);
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
                let share = recall(metric, vectors, vectors, 100, 1280);
                assert!(share >= 0.99, "{data} {metric:?}: recall@10 {share}");
            }
        }
    }

    /// A long vector along one axis, the query's largest, whose inner
    /// product with the query beats every other row's by a third. Its signs
    /// along the axes say nothing of it but in that one axis; after the
    /// rotation every bit sees it.
    #[test]
    fn a_long_vector_along_one_axis_is_shortlisted_as_one_spread_over_all() {
        let mut random = Random(7);
        let normal = |random: &mut Random| -> f32 { (0..12).map(|_| random.unit()).sum() };
        for _ in 0..10 {
            let query: Vec<f32> = (0..256).map(|_| normal(&mut random)).collect();
            let mut vectors: Vec<Vec<f32>> = (0..5000)
                .map(|_| (0..256).map(|_| normal(&mut random)).collect())
                .collect();
            let best = vectors
                .iter()
                .map(|vector| dot(vector, &query))
                .fold(f32::MIN, f32::max);
            let (axis, along) =
                query
                    .iter()
                    .copied()
                    .enumerate()
                    .fold((0, f32::MIN), |most, (axis, value)| {
                        if value > most.1 {
                            (axis, value)
                        } else {
                            most
                        }
                    });
            let mut sparse = vec![0.0; 256];
            sparse[axis] = 1.3 * best / along;
            vectors.push(sparse);
            let index = indexed(Distance::Dot, &vectors);
            let shortlist = index.shortlist(&query, 128, None, &[], &in_row_order);
            assert!(shortlist.contains(&10_000), "best {best}");
        }
    }

    /// Twenty vectors a hundred times as long as the rest, each at nearly a
    /// right angle to the query, with inner products of 40 against the
    /// rest's under 3, over ten draws. Measured, about half of them look at
    /// or past a right angle, and ranked by the measured angle about three
    /// in four are shortlisted; ranked by how near they may be, nearly all.
    #[test]
    fn long_vectors_at_nearly_a_right_angle_are_shortlisted_by_how_near_they_may_be() {
        let found: usize = (31..41)
            .map(|seed| {
                let mut random = Random(seed);
                let query = random.vector(64);
                let mut vectors: Vec<Vec<f32>> = (0..5000).map(|_| random.vector(64)).collect();
                assert!(vectors.iter().all(|vector| dot(vector, &query) < 3.0));
                vectors.extend((0..20).map(|_| {
                    let sideways = random.vector(64);
                    let off = (dot(&sideways, &query) - 0.4) / dot(&query, &query);
                    sideways
                        .iter()
                        .zip(&query)
                        .map(|(s, q)| 100.0 * (s - off * q))
                        .collect()
                }));
                let index = indexed(Distance::Dot, &vectors);
                let shortlist = index.shortlist(&query, 128, None, &[], &in_row_order);
                (5000..5020u32)
                    .filter(|row| shortlist.contains(&(row * 2)))
                    .count()
            })
            .sum();
        assert!(found >= 185, "{found} of 200 long vectors shortlisted");
    }

    /// One vector far from the rest, as long as a collection accepts or, as
    /// one stored before lengths were limited may be, longer: the centre the
    /// codes are taken about stays among the rest, and so does their recall.
    #[test]
    fn a_vector_far_from_the_rest_spoils_no_other_rows_code() {
        for (case, outlier) in [
            ("none", None),
            ("accepted", Some(4e17)),
            ("legacy", Some(1.5e18)),
        ] {
            let mut random = Random(13);
            let mut vectors = grouped(&mut random, 4000, 64, 0.3);
            let queries = vectors.clone();
            vectors.extend(outlier.map(|value| vec![value; 64]));
            for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
                let share = recall(metric, &vectors, &queries, 200, 160);
                assert!(
                    share >= 0.9,
                    "{case} {metric:?}: {share} of the true neighbours"
                );
            }
        }
    }

    /// Vectors and queries at the length limit, leaning one way, so that
    /// the centre is far out too, and facing every way about it: no
    /// estimate is NaN or `-∞`, and only an L2 estimate is ever infinite.
    #[test]
    fn estimates_at_the_length_limit_are_never_nan_or_minus_infinity() {
        let mut random = Random(31);
        let lean = random.vector(64);
        let mut at_limit = || {
            let vector: Vec<f32> = lean.iter().map(|l| l + 0.5 * random.unit()).collect();
            let scale = (0.99 * SQUARED_LENGTH_UNDER / dot(&vector, &vector)).sqrt();
            vector.into_iter().map(|v| v * scale).collect::<Vec<f32>>()
        };
        let vectors: Vec<Vec<f32>> = (0..500).map(|_| at_limit()).collect();
        let queries: Vec<Vec<f32>> = (0..20).map(|_| at_limit()).collect();
        for metric in [Distance::Dot, Distance::L2] {
            let index = indexed(metric, &vectors);
            assert!(index.unmeasured(None, &[]).is_empty());
            for query in queries.iter().chain(&vectors[..20]) {
                for (estimate, _) in index.estimates(query, 500, None, &[], &in_row_order) {
                    assert!(
                        !estimate.is_nan() && estimate != f32::NEG_INFINITY,
                        "{metric:?}"
                    );
                    assert!(estimate.is_finite() || metric == Distance::L2, "{metric:?}");
                }
            }
        }
    }

    /// A write sums a vector's squares one after another, the index in
    /// lanes: at the limit the two can fall either side of it, and every
    /// vector a write accepts must still be measured.
    #[test]
    fn every_vector_a_write_accepts_is_measured() {
        let mut random = Random(29);
        let accepted: Vec<Vec<f32>> = (0..2000)
            .map(|_| {
                let vector = random.vector(768);
                let squared: f32 = vector.iter().map(|v| v * v).sum();
                let scale = (SQUARED_LENGTH_UNDER / squared).sqrt();
                vector.into_iter().map(|v| v * scale).collect::<Vec<f32>>()
            })
            .filter(|vector| vector.iter().map(|v| v * v).sum::<f32>() < SQUARED_LENGTH_UNDER)
            .collect();
        assert!(
            accepted
                .iter()
                .any(|vector| dot(vector, vector) >= SQUARED_LENGTH_UNDER),
            "no vector here sums past the limit in lanes"
        );
        assert!(accepted
            .iter()
            .all(|vector| measurable(Distance::Dot, vector)));
    }

    /// Vectors longer than a collection accepts, though their squared
    /// lengths are just finite, have no code to rank them by. However many
    /// there are, they take no place in a shortlist; they are left for the
    /// caller to score exactly, and the rest are found as ever.
    #[test]
    fn vectors_too_long_to_measure_are_left_to_be_scored_exactly() {
        let mut random = Random(17);
        let mut vectors = grouped(&mut random, 2000, 64, 0.3);
        let queries = vectors.clone();
        vectors.extend(std::iter::repeat_n(vec![2.3e18; 64], 500));
        let long: Vec<u32> = (2000..2500u32).map(|row| row * 2).collect();
        for metric in [Distance::Dot, Distance::L2] {
            let index = indexed(metric, &vectors);
            assert_eq!(index.len(), 2500);
            assert_eq!(index.unmeasured(None, &[]), long, "{metric:?}");
            let only: RoaringBitmap = [0u32, 4000, 4002, 4998].into_iter().collect();
            let mut skipped = vec![false; 5000];
            skipped[4002] = true;
            assert_eq!(index.unmeasured(Some(&only), &skipped), [4000, 4998]);
            let shortlist = index.shortlist(&queries[0], 160, None, &[], &in_row_order);
            assert!(
                shortlist.iter().all(|row| *row < 4000),
                "{metric:?}: a row with no estimate shortlisted"
            );
            let share = recall(metric, &vectors, &queries, 100, 160);
            assert!(share >= 0.9, "{metric:?}: {share} of the true neighbours");
        }
        // Under cosine only a direction counts, and every such vector has one.
        assert!(indexed(Distance::Cosine, &vectors)
            .unmeasured(None, &[])
            .is_empty());
    }

    /// Under cosine only a direction counts, and a vector too long to
    /// square still has one: coded by it, one pointing away from the query
    /// is not shortlisted.
    #[test]
    fn a_vector_too_long_to_square_keeps_its_direction_under_cosine() {
        let unit = normalised(Distance::Cosine, &[3e30, -4e30, 0.0]);
        assert!(
            unit.iter()
                .zip([0.6, -0.8, 0.0])
                .all(|(value, expected)| (value - expected).abs() < 1e-6),
            "{unit:?}"
        );
        assert_eq!(&*normalised(Distance::Cosine, &[0.0, 0.0]), [0.0, 0.0]);
        assert_eq!(&*normalised(Distance::Dot, &[3e30]), [3e30]);

        let mut random = Random(19);
        let mut vectors = grouped(&mut random, 1000, 64, 0.3);
        vectors.push(vectors[0].iter().map(|value| -1e30 * value).collect());
        let index = indexed(Distance::Cosine, &vectors);
        let shortlist = index.shortlist(&vectors[0], 5, None, &[], &in_row_order);
        assert!(!shortlist.contains(&2000));
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
    fn rows_the_index_cannot_tell_apart_are_cut_in_the_callers_order() {
        let mut random = Random(4);
        let mut vectors: Vec<Vec<f32>> = (0..500).map(|_| random.vector(32)).collect();
        // Long enough to be the nearest under every metric.
        let same: Vec<f32> = random.vector(32).into_iter().map(|v| 4.0 * v).collect();
        vectors.extend(std::iter::repeat_n(same.clone(), 300));
        for metric in [Distance::Cosine, Distance::Dot, Distance::L2] {
            let index = indexed(metric, &vectors);
            let backwards = |a: u32, b: u32| b.cmp(&a);
            let mut rows = index.shortlist(&same, 3, None, &[], &backwards);
            rows.sort_unstable();
            assert_eq!(rows, [1594, 1596, 1598], "{metric:?}");
            let mut rows = index.shortlist(&same, 3, None, &[], &in_row_order);
            rows.sort_unstable();
            assert_eq!(rows, [1000, 1002, 1004], "{metric:?}");
        }
    }

    /// A zero query is at an inner product of zero from every vector, and
    /// every such distance is the same zero, not some of them negative.
    #[test]
    fn a_zero_query_is_at_the_same_inner_product_from_every_vector() {
        let zero = [0.0, 0.0];
        let distances: Vec<u32> = [[1.0, 2.0], [-1.0, -2.0], [-1.0, 2.0]]
            .iter()
            .map(|vector| distance(Distance::Dot, &zero, vector).to_bits())
            .collect();
        assert_eq!(distances, [0.0f32.to_bits(); 3]);
    }

    /// Scoring a vector by cosine reads it once, as the inner product does,
    /// and costs little more: a shape of its sums that defeats the
    /// compiler's vectorising has made it three times as slow before. A
    /// zero row, which takes a path of its own, costs little more than any
    /// other row by cosine, and is held to that rather than to the inner
    /// product, as the two differ less from one processor to another: it
    /// has cost three times as much. The vectors fit in the processor's
    /// cache, so that it is the arithmetic that is timed and not the memory
    /// it is read from, and each kind is timed at its fastest of many
    /// rounds, which whatever else the machine is doing slows least.
    /// Timings mean nothing unoptimised, so this runs only in a release
    /// build.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "timings mean nothing unoptimised")]
    fn scoring_by_cosine_costs_little_more_than_by_inner_product() {
        use std::hint::black_box;

        let mut random = Random(59);
        let vectors: Vec<Vec<f32>> = (0..32).map(|_| random.vector(768)).collect();
        let zero = vec![vec![0.0; 768]; 32];
        let query = random.vector(768);
        let fastest = |metric: Distance, vectors: &[Vec<f32>]| {
            let scorer = Scorer::new(metric, &query);
            (0..50)
                .map(|_| {
                    let started = std::time::Instant::now();
                    for vector in vectors.iter().cycle().take(3200) {
                        black_box(scorer.distance(black_box(vector)));
                    }
                    started.elapsed()
                })
                .min()
                .map_or(0.0, |fastest| fastest.as_secs_f64())
        };
        let inner = fastest(Distance::Dot, &vectors);
        let cosine = fastest(Distance::Cosine, &vectors);
        let over_zero = fastest(Distance::Cosine, &zero);
        let (by_cosine, zero_by_cosine) = (cosine / inner, over_zero / cosine);
        println!(
            "a cosine score takes {by_cosine:.2} times an inner product's; over a zero row \
             it takes {zero_by_cosine:.2} times its own over any other, and {:.2} times an \
             inner product's",
            over_zero / inner
        );
        assert!(by_cosine < 3.0, "{by_cosine:.2} times an inner product's");
        assert!(
            zero_by_cosine < 2.5,
            "{zero_by_cosine:.2} times a cosine score over any other row"
        );
    }

    /// A vector is at no cosine distance from itself, and so of certainty
    /// exactly one, over every number of dimensions and at every length:
    /// everyday ones, and those whose squares an `f32` cannot hold in full,
    /// down to values that are not even normal numbers.
    #[test]
    fn a_vector_against_itself_is_at_no_cosine_distance_exactly() {
        let mut random = Random(53);
        let scales = [1e-40, 1e-30, 1e-20, 1e-6, 1.0, 1e6, 1e18, 1e30];
        for dimensions in 3..=1536 {
            for scale in [scales[dimensions % 8], scales[dimensions % 7]] {
                let vector: Vec<f32> = random
                    .vector(dimensions)
                    .into_iter()
                    .map(|value| value * scale)
                    .collect();
                let distance = distance(Distance::Cosine, &vector, &vector);
                assert_eq!(distance, 0.0, "{dimensions} dimensions at {scale}");
                assert_eq!(certainty(Distance::Cosine, distance), Some(1.0));
            }
        }
    }

    #[test]
    fn a_vector_has_a_direction_whatever_its_length_unless_it_is_zero() {
        let east = [1.0, 0.0];
        for vector in [[1e30f32, 0.0], [1e-30, 0.0], [1e-40, 0.0], [3e19, 1e-30]] {
            assert!(
                distance(Distance::Cosine, &vector, &vector) < 1e-6,
                "{vector:?} against itself"
            );
            assert!(
                distance(Distance::Cosine, &east, &vector) < 1e-6,
                "east against {vector:?}"
            );
            assert!(
                distance(Distance::Cosine, &vector, &east) < 1e-6,
                "{vector:?} against east"
            );
        }
        let north_ish = [1e-30, 1e-30];
        assert!(
            (distance(Distance::Cosine, &east, &north_ish) - (1.0 - 0.5f32.sqrt())).abs() < 1e-6
        );
        let directionless = [
            [0.0f32, 0.0],
            [-0.0, 0.0],
            [-0.0, -0.0],
            [f32::INFINITY, 0.0],
            [f32::NAN, 1.0],
        ];
        for vector in directionless {
            assert_eq!(
                distance(Distance::Cosine, &east, &vector),
                1.0,
                "{vector:?}"
            );
            assert_eq!(
                distance(Distance::Cosine, &vector, &east),
                1.0,
                "{vector:?}"
            );
            assert!(unit(&vector).is_none(), "{vector:?}");
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
            let mut rows = index.shortlist(query, count, only, skipped, &in_row_order);
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
            index.shortlist(&query[..10], 5, None, &[], &in_row_order),
            Vec::<u32>::new()
        );
        assert!(BinaryIndex::build(Distance::Cosine, &[]).is_none());
    }

    /// The centre a growing index takes, worked out plainly: the vectors,
    /// normalised as the index normalises them, summed in `f64` one after
    /// another, divided and rounded to `f32` once; and that mean rotated
    /// once, as the index rotates every vector. Rotating the mean rather
    /// than every vector keeps the work at the threshold to a sum.
    fn scalar_mean(metric: Distance, vectors: &[Vec<f32>]) -> Vec<f32> {
        let rotation = Rotation::new(vectors[0].len(), ROTATION_SEED);
        let measured: Vec<Vec<f32>> = vectors
            .iter()
            .filter(|vector| measurable(metric, vector))
            .map(|vector| normalised(metric, vector).into_owned())
            .collect();
        let mean: Vec<f32> = (0..vectors[0].len())
            .map(|at| {
                let mut sum = 0.0f64;
                for vector in &measured {
                    sum += f64::from(vector[at]);
                }
                (sum / measured.len() as f64) as f32
            })
            .collect();
        rotation.apply(&mean)
    }

    #[test]
    fn a_growing_index_is_centred_on_the_scalar_mean_to_the_bit() {
        let mut random = Random(23);
        for (metric, dimensions) in [
            (Distance::Cosine, 768),
            (Distance::L2, 100),
            (Distance::Dot, 64),
        ] {
            let vectors = grouped(&mut random, 1_024, dimensions, 0.4);
            let first: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
            let index = BinaryIndex::about_mean(metric, &first).unwrap();
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(index.centre()),
                bits(&scalar_mean(metric, &vectors)),
                "{metric:?}"
            );
        }
    }

    #[test]
    fn rows_coded_one_at_a_time_rank_as_rows_coded_together_about_the_same_centre() {
        let mut random = Random(29);
        for metric in [Distance::Cosine, Distance::L2, Distance::Dot] {
            let vectors = grouped(&mut random, 3_000, 128, 0.4);
            let first: Vec<&[f32]> = vectors[..1_024].iter().map(Vec::as_slice).collect();
            let mut growing = BinaryIndex::about_mean(metric, &first).unwrap();
            for (row, vector) in vectors.iter().enumerate() {
                growing.push(row as u32 * 3, vector);
            }
            let rows: Vec<(u32, &[f32])> = vectors
                .iter()
                .enumerate()
                .map(|(row, vector)| (row as u32 * 3, vector.as_slice()))
                .collect();
            let together =
                BinaryIndex::build_about(metric, &rows, growing.centre().to_vec()).unwrap();
            assert_eq!(growing.len(), together.len());
            for query in grouped(&mut random, 20, 128, 0.4) {
                let ranked =
                    |index: &BinaryIndex| index.estimates(&query, 200, None, &[], &in_row_order);
                assert_eq!(ranked(&growing), ranked(&together), "{metric:?}");
            }
        }
    }

    #[test]
    fn a_growing_index_counts_its_codes_as_resident() {
        let mut random = Random(31);
        let vectors = grouped(&mut random, 2_000, 768, 0.4);
        let first: Vec<&[f32]> = vectors[..1_024].iter().map(Vec::as_slice).collect();
        let mut index = BinaryIndex::about_mean(Distance::Cosine, &first).unwrap();
        let empty = index.resident();
        for (row, vector) in vectors.iter().enumerate() {
            index.push(row as u32, vector);
        }
        // A bit a rotated dimension and two factors, for each row.
        assert!(index.resident() - empty >= 2_000 * (1_024 / 8 + 8));
    }

    #[test]
    fn rows_pushed_as_exact_are_offered_whole_and_never_ranked() {
        let mut random = Random(37);
        let vectors = grouped(&mut random, 2_000, 128, 0.4);
        let first: Vec<&[f32]> = vectors[..1_024].iter().map(Vec::as_slice).collect();
        let mut index = BinaryIndex::about_mean(Distance::Cosine, &first).unwrap();
        for row in 0..1_024u32 {
            index.push_exact(row);
        }
        for (row, vector) in vectors.iter().enumerate().skip(1_024) {
            index.push(row as u32, vector);
        }
        assert_eq!((index.len(), index.coded()), (2_000, 976));
        assert_eq!(
            index.unmeasured(None, &[]),
            (0..1_024).collect::<Vec<u32>>()
        );
        let ranked = index.estimates(&vectors[5], 2_000, None, &[], &in_row_order);
        assert!(
            ranked.iter().all(|(_, row)| *row >= 1_024),
            "only coded rows are ranked"
        );
        assert_eq!(ranked.len(), 976);
    }

    /// The same sum to the last bit, or both not a number.
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(400))]
        /// Every way of making the `f32` sums this machine runs gives the
        /// portable way's result to the last bit: any length, whole chunks
        /// or not, and values of every size, from those that vanish to
        /// those that overflow, and now and then one that is not finite. A
        /// way that fused its multiply-adds, or summed its lanes in another
        /// order, would not.
        #[test]
        fn every_f32_way_gives_the_portable_sums_to_the_last_bit(
            seed in 1u64..u64::MAX,
            len in 0usize..1100,
            scale in -40i32..40,
            odd in 0u8..20,
        ) {
            let mut noise = seed;
            let mut next = move || {
                noise ^= noise << 13;
                noise ^= noise >> 7;
                noise ^= noise << 17;
                noise
            };
            let value = |next: &mut dyn FnMut() -> u64| {
                let unit = (next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                match next() % 997 {
                    0 if odd == 0 => f32::NAN,
                    1 if odd == 1 => f32::INFINITY,
                    2 => 0.0,
                    3 => -0.0,
                    4 => f32::MIN_POSITIVE / 3.0,
                    _ => unit * 2f32.powi(scale),
                }
            };
            let a: Vec<f32> = (0..len).map(|_| value(&mut next)).collect();
            let b: Vec<f32> = (0..len).map(|_| value(&mut next)).collect();
            let portable = f32s::PORTABLE;
            for way in f32s::ways_here() {
                proptest::prop_assert!(same((way.dot)(&a, &b), (portable.dot)(&a, &b)), "{} dot", way.name);
                proptest::prop_assert!(
                    same((way.squared_l2)(&a, &b), (portable.squared_l2)(&a, &b)),
                    "{} squared_l2", way.name
                );
                let (d, n) = (way.dot_and_norm)(&a, &b);
                let (pd, pn) = (portable.dot_and_norm)(&a, &b);
                proptest::prop_assert!(same(d, pd) && same(n, pn), "{} dot_and_norm", way.name);
            }
        }
    }

    /// The `f32` sums are made the widest way this machine has.
    #[test]
    fn the_f32_way_chosen_is_the_widest_this_machine_has() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            assert_eq!(f32s::chosen().name, "avx2");
            return;
        }
        assert_eq!(f32s::chosen().name, "portable");
    }
}
