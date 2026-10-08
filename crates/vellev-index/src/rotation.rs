//! A random rotation, and the small generator that drives it, shared by the
//! vector indexes. A one-bit code taken after a random rotation is a code
//! along random directions, each of which meets every axis: a vector that
//! lies along a few axes is coded as faithfully as one spread over all.
//!
//! A clustered index is stored with only its rotation's width and seed, and
//! its codes were taken from what [`Rotation::apply`] gave when it was built.
//! Both the rotation and [`Random`] must therefore give the same bits for
//! the same width, seed and input forever: any change to either makes
//! every stored index answer wrongly. The golden tests below hold them to it.

use serde::{Deserialize, Serialize};
use vellev_core::MAX_DIMENSIONS;

/// The narrowest rotation: one word of code.
const NARROWEST: usize = 64;
/// The widest rotation a stored index can have: that of the most dimensions
/// a collection may declare.
const WIDEST: usize = (MAX_DIMENSIONS as usize).next_power_of_two();

/// A random orthogonal rotation: random signs followed by a Walsh-Hadamard
/// transform, over the dimensions padded to a power of two. Stored as its
/// width and seed alone; the signs follow from the seed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "Spec", into = "Spec")]
pub(crate) struct Rotation {
    pub width: usize,
    seed: u64,
    signs: Vec<f32>,
}

/// What a rotation is stored as.
#[derive(Serialize, Deserialize)]
struct Spec {
    width: usize,
    seed: u64,
}

/// A stored rotation is read back only if its width is one [`Rotation::new`]
/// gives a collection's vectors. A file that says otherwise is corrupt, and
/// is refused before anything is sized by what it says.
impl TryFrom<Spec> for Rotation {
    type Error = String;

    fn try_from(Spec { width, seed }: Spec) -> Result<Self, Self::Error> {
        match width.is_power_of_two() && (NARROWEST..=WIDEST).contains(&width) {
            true => Ok(Self::of(width, seed)),
            false => Err(format!(
                "a rotation {width} wide, not a power of two from {NARROWEST} to {WIDEST}"
            )),
        }
    }
}

impl From<Rotation> for Spec {
    fn from(rotation: Rotation) -> Self {
        Self {
            width: rotation.width,
            seed: rotation.seed,
        }
    }
}

impl Rotation {
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self::of(dimensions.next_power_of_two().max(NARROWEST), seed)
    }

    fn of(width: usize, seed: u64) -> Self {
        let mut random = Random(seed);
        let signs = (0..width)
            .map(|_| if random.next() & 1 == 0 { 1.0 } else { -1.0 })
            .collect();
        Self { width, seed, signs }
    }

    pub fn apply(&self, vector: &[f32]) -> Vec<f32> {
        let mut rotated: Vec<f32> = self
            .signs
            .iter()
            .enumerate()
            .map(|(i, sign)| vector.get(i).copied().unwrap_or(0.0) * sign)
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
pub(crate) struct Random(pub u64);

impl Random {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    /// `count` of the positions `0..len`, or all of them if there are no
    /// more, in ascending order. Chosen at random rather than evenly
    /// spaced: rows written in a repeating pattern would otherwise be
    /// sampled in step with it, and whole kinds of vector left out.
    pub fn sample(&mut self, len: usize, count: usize) -> Vec<usize> {
        if count >= len {
            return (0..len).collect();
        }
        let mut chosen = std::collections::HashSet::with_capacity(count);
        while chosen.len() < count {
            chosen.insert(self.below(len));
        }
        let mut chosen: Vec<usize> = chosen.into_iter().collect();
        chosen.sort_unstable();
        chosen
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::dot;

    fn vector(random: &mut Random, dimensions: usize) -> Vec<f32> {
        (0..dimensions)
            .map(|_| (random.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            .collect()
    }

    #[test]
    fn the_rotation_keeps_lengths_and_inner_products() {
        let mut random = Random(5);
        let rotation = Rotation::new(100, 77);
        assert_eq!(rotation.width, 128);
        for _ in 0..5 {
            let (a, b) = (vector(&mut random, 100), vector(&mut random, 100));
            let (ra, rb) = (rotation.apply(&a), rotation.apply(&b));
            assert!((dot(&a, &b) - dot(&ra, &rb)).abs() < 1e-4);
            assert!((dot(&a, &a) - dot(&ra, &ra)).abs() < 1e-4);
        }
        assert_eq!(rotation.apply(&[1.0; 100]), rotation.apply(&[1.0; 100]));
    }

    /// Along one axis, the rotation spreads a vector over every coordinate
    /// equally: each is one over the square root of the width.
    #[test]
    fn the_rotation_spreads_an_axis_over_every_coordinate_equally() {
        let rotation = Rotation::new(64, 3);
        let mut axis = vec![0.0; 64];
        axis[5] = 1.0;
        let rotated = rotation.apply(&axis);
        assert!(
            rotated
                .iter()
                .all(|value| (value.abs() - 0.125).abs() < 1e-7),
            "{rotated:?}"
        );
    }

    /// Stored clustered indexes were coded from exactly these bits.
    #[test]
    fn the_rotation_and_the_generator_give_the_same_bits_forever() {
        let input: Vec<f32> = (0..50).map(|i| (i as f32 - 20.0) / 7.0).collect();
        let bits: Vec<u32> = Rotation::new(50, 42)
            .apply(&input)
            .iter()
            .map(|value| value.to_bits())
            .collect();
        assert_eq!(
            bits,
            [
                3218079745, 1078160823, 1050373558, 3216581779, 1057264203, 1078760009, 1082167881,
                1077262043, 3187671048, 3208342970, 1035393888, 1070596095, 1016219936, 3224895489,
                1079508992, 3209541341, 3205946224, 1066701386, 1044980876, 3230625207, 3163703584,
                1016220064, 1077561636, 1068498944, 3214784220, 3226842844, 3202650700, 1060859318,
                3200253951, 3229239589, 1082991762, 3226543250, 3207144594, 3215682998, 1071195282,
                1066701385, 1071794468, 3211338900, 1057863384, 3218678930, 1060260130, 1067899759,
                1070596096, 3226842842, 1066401792, 3213885438, 3194861280, 1070895690, 1050373559,
                1062656876, 1069697316, 1074565704, 1066701384, 3163703648, 3206545408, 1063855248,
                3203849072, 1076962450, 1065503013, 3190067782, 1059660946, 1072992842, 1060859320,
                1079059602,
            ]
        );
        let mut random = Random(7);
        assert_eq!(
            [random.next(), random.next(), random.next()],
            [
                7191089600892374487,
                309689372594955804,
                16616101746815609346
            ]
        );
        assert_eq!(random.sample(100, 5), [3, 5, 74, 82, 98]);
    }

    /// An FNV-1a hash of `values`' bits, to hold a wide rotation's every
    /// output bit to a golden value in a few characters.
    fn fingerprint(values: &[f32]) -> u64 {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .fold(0xCBF2_9CE4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
            })
    }

    /// At a width of 512, whose root is not exact, the final scale rounds;
    /// at 1,024 the transform runs ten stages. Both give the same bits
    /// forever too.
    #[test]
    fn wide_rotations_give_the_same_bits_forever() {
        for (dimensions, golden) in [(500, 11927336186241100382), (1000, 893120748369387992)] {
            let input: Vec<f32> = (0..dimensions).map(|i| (i as f32 - 200.0) / 13.0).collect();
            let rotation = Rotation::new(dimensions, 1234);
            let rotated = rotation.apply(&input);
            assert_eq!(
                (rotated.len(), fingerprint(&rotated)),
                (rotation.width, golden)
            );
        }
    }

    /// A stored width no rotation of a collection's vectors has is refused,
    /// not sized by: nothing, an odd number, a width too narrow, and one
    /// that would take all memory to give signs to.
    #[test]
    fn a_stored_rotation_of_no_possible_width_is_refused() {
        for width in [0, 1, 32, 100, 16_384, 1 << 40] {
            let bytes = rmp_serde::to_vec_named(&Spec { width, seed: 7 }).unwrap();
            let read = rmp_serde::from_slice::<Rotation>(&bytes);
            assert!(read.is_err(), "a width of {width} was read");
        }
        for width in [64, 1024, 8192] {
            let bytes = rmp_serde::to_vec_named(&Spec { width, seed: 7 }).unwrap();
            let read: Rotation = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(read.apply(&[1.0]).len(), width);
        }
    }

    /// The signs are kept, not stored: a rotation read back is the same.
    #[test]
    fn a_rotation_survives_the_wire_as_its_width_and_seed() {
        let rotation = Rotation::new(100, 77);
        let bytes = rmp_serde::to_vec_named(&rotation).unwrap();
        let spec: Spec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!((spec.width, spec.seed), (128, 77));
        let back: Rotation = rmp_serde::from_slice(&bytes).unwrap();
        let input = vec![0.5; 100];
        assert_eq!(back.apply(&input), rotation.apply(&input));
    }
}
