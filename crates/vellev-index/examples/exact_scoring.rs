//! How long scoring one vector exactly takes under each metric: what every
//! row a search rescores costs, and every row of a layer too small to index.
//!
//!   cargo run --release -p vellev-index --example exact_scoring
//!
//! Environment: VECTORS (100000), DIMS (768), ROUNDS (20). Each metric is
//! timed over every vector, round after round, and the fastest round kept,
//! which is the least disturbed by whatever else the machine is doing.

use std::hint::black_box;
use std::time::{Duration, Instant};

use vellev_core::Distance;
use vellev_index::vector::Scorer;

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

struct Random(u64);

impl Random {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn main() {
    let (count, dimensions, rounds) = (
        setting("VECTORS", 100_000),
        setting("DIMS", 768),
        setting("ROUNDS", 20),
    );
    let mut random = Random(1);
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dimensions).map(|_| random.unit()).collect())
        .collect();
    let query: Vec<f32> = (0..dimensions).map(|_| random.unit()).collect();
    println!("{count} vectors of {dimensions} dimensions, fastest of {rounds} rounds");
    for metric in [Distance::Dot, Distance::L2, Distance::Cosine] {
        let scorer = Scorer::new(metric, &query);
        let fastest = (0..rounds)
            .map(|_| {
                let started = Instant::now();
                for vector in &vectors {
                    black_box(scorer.distance(black_box(vector)));
                }
                started.elapsed()
            })
            .min()
            .unwrap_or(Duration::ZERO);
        println!(
            "{metric:?}: {:.1} ns per score",
            fastest.as_nanos() as f64 / count as f64
        );
    }
}
