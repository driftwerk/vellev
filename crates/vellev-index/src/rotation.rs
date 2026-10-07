//! A random rotation, and the small generator that drives it, shared by the
//! vector indexes. A one-bit code taken after a random rotation is a code
//! along random directions, each of which meets every axis: a vector that
//! lies along a few axes is coded as faithfully as one spread over all.

use serde::{Deserialize, Serialize};

/// A random orthogonal rotation: random signs followed by a Walsh-Hadamard
/// transform, over the dimensions padded to a power of two.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Rotation {
    pub width: usize,
    seed: u64,
}

impl Rotation {
    pub fn new(dimensions: usize, seed: u64) -> Self {
        Self {
            width: dimensions.next_power_of_two().max(64),
            seed,
        }
    }

    pub fn apply(&self, vector: &[f32]) -> Vec<f32> {
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

    #[test]
    fn the_rotation_spreads_a_direction_over_every_coordinate() {
        let rotation = Rotation::new(64, 3);
        let rotated = rotation.apply(&[1.0; 64]);
        let largest = rotated.iter().fold(0.0f32, |most, v| most.max(v.abs()));
        let length = dot(&rotated, &rotated).sqrt();
        assert!(
            largest < 0.5 * length,
            "{largest} of {length} in one coordinate"
        );
    }
}
