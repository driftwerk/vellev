//! Measure the binary shortlist's recall and speed against an exact scan.
//!
//!   cargo run --release -p vellev-index --example vector_recall
//!
//! Environment: VECTORS (100000), DIMS (768), QUERIES (100), SPREAD (1.0).
//!
//! The data is synthetic: unit vectors scattered round random centres, 100 to
//! a centre, with `SPREAD` setting how far. Real embeddings are distributed
//! differently, so read the recall here as a sanity check, not a promise.

use std::time::{Duration, Instant};

use vellev_core::Distance;
use vellev_index::vector::{distance, BinaryIndex, VectorIndex};

fn setting<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

struct Random(u64);

impl Random {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// A standard normal value, by the Box-Muller transform.
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn normalised(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|v| *v /= norm);
    vector
}

fn exact(
    vectors: &[Vec<f32>],
    query: &[f32],
    rows: impl Iterator<Item = u32>,
    k: usize,
) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = rows
        .map(|row| {
            (
                distance(Distance::Cosine, query, &vectors[row as usize]),
                row,
            )
        })
        .collect();
    let k = k.min(scored.len());
    if k < scored.len() {
        scored.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
    }
    scored.truncate(k);
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().map(|(_, row)| row).collect()
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let count: usize = setting("VECTORS", 100_000);
    let dimensions: usize = setting("DIMS", 768);
    let queries: usize = setting("QUERIES", 100);
    let spread: f32 = setting("SPREAD", 1.0);
    const K: usize = 10;

    let mut random = Random(0x5eed);
    let centres: Vec<Vec<f32>> = (0..count.div_ceil(100))
        .map(|_| (0..dimensions).map(|_| random.normal()).collect())
        .collect();
    let mut scattered = |n: usize| {
        let centre = &centres[n % centres.len()];
        normalised(
            centre
                .iter()
                .map(|c| c + spread * random.normal())
                .collect(),
        )
    };
    let started = Instant::now();
    let vectors: Vec<Vec<f32>> = (0..count).map(&mut scattered).collect();
    let asked: Vec<Vec<f32>> = (0..queries).map(|n| scattered(n * 7919)).collect();
    println!(
        "{count} vectors of {dimensions} dims generated in {:.1?}",
        started.elapsed()
    );

    let started = Instant::now();
    let rows: Vec<(u32, &[f32])> = vectors
        .iter()
        .enumerate()
        .map(|(row, v)| (row as u32, v.as_slice()))
        .collect();
    let index = BinaryIndex::build(Distance::Cosine, &rows).expect("there are vectors");
    println!(
        "index built in {:.1?} ({} bytes of codes per vector)\n",
        started.elapsed(),
        dimensions.div_ceil(64) * 8
    );

    let started = Instant::now();
    let truth: Vec<Vec<u32>> = asked
        .iter()
        .map(|query| exact(&vectors, query, 0..count as u32, K))
        .collect();
    let brute = started.elapsed() / queries as u32;
    println!("exact scan of every vector: {brute:.2?} per query\n");
    println!(
        "{:>10} {:>10} {:>12} {:>12}",
        "oversample", "recall@10", "mean", "p99"
    );

    for oversample in [4, 8, 16, 32, 64, 128, 256] {
        let mut found = 0;
        let mut taken = Vec::with_capacity(queries);
        for (query, truth) in asked.iter().zip(&truth) {
            let started = Instant::now();
            let shortlist = index.shortlist(query, K * oversample, None, &[]);
            let nearest = exact(&vectors, query, shortlist.into_iter(), K);
            taken.push(started.elapsed());
            found += truth.iter().filter(|row| nearest.contains(row)).count();
        }
        taken.sort_unstable();
        let mean = taken.iter().sum::<Duration>() / queries as u32;
        println!(
            "{oversample:>10} {:>9.1}% {mean:>12.2?} {:>12.2?}",
            100.0 * found as f64 / (queries * K) as f64,
            percentile(&taken, 0.99),
        );
    }
}
