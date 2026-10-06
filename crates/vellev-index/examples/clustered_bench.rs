//! Build time, recall and query latency of the clustered index against a
//! full scan of one-bit codes, on synthetic grouped vectors.
//!
//! cargo run --release -p vellev-index --example clustered_bench -- [count] [dimensions] [spread]

use std::time::Instant;

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::vector::{distance, BinaryIndex, VectorIndex};

struct Noise(u64);

impl Noise {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let count: usize = args.next().map_or(200_000, |a| a.parse().unwrap());
    let dimensions: usize = args.next().map_or(768, |a| a.parse().unwrap());
    let spread: f32 = args.next().map_or(0.6, |a| a.parse().unwrap());
    let groups = 1_000;
    let mut noise = Noise(1);
    let centres: Vec<Vec<f32>> = (0..groups)
        .map(|_| (0..dimensions).map(|_| noise.unit()).collect())
        .collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|n| {
            centres[n % groups]
                .iter()
                .map(|c| c + spread * noise.unit())
                .collect()
        })
        .collect();
    let rows: Vec<(u32, &[f32])> = vectors
        .iter()
        .enumerate()
        .map(|(row, v)| (row as u32, v.as_slice()))
        .collect();
    println!("{count} vectors of {dimensions} dimensions, {groups} groups spread {spread}");

    let started = Instant::now();
    let clustered = ClusteredIndex::build(Distance::Cosine, &rows).unwrap();
    println!(
        "clustered: built in {:.1?}, {} clusters, {} probes",
        started.elapsed(),
        clustered.clusters(),
        clustered.probes()
    );
    let started = Instant::now();
    let binary = BinaryIndex::build(&rows).unwrap();
    println!("binary: built in {:.1?}", started.elapsed());

    let queries: Vec<&Vec<f32>> = vectors.iter().step_by(count / 50).collect();
    let exact: Vec<Vec<u32>> = queries
        .iter()
        .map(|query| {
            let mut all: Vec<(f32, u32)> = vectors
                .iter()
                .enumerate()
                .map(|(row, v)| (distance(Distance::Cosine, query, v), row as u32))
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            all[..10].iter().map(|(_, row)| *row).collect()
        })
        .collect();
    for oversample in [4usize, 16, 64] {
        for (name, index) in [
            ("clustered", &clustered as &dyn VectorIndex),
            ("binary", &binary as &dyn VectorIndex),
        ] {
            let started = Instant::now();
            let mut found = 0;
            for (query, truth) in queries.iter().zip(&exact) {
                let shortlist = index.shortlist(query, 10 * oversample, None, &[]);
                found += truth.iter().filter(|row| shortlist.contains(row)).count();
            }
            let per_query = started.elapsed() / queries.len() as u32;
            println!(
                "{name:>9} oversample {oversample:>2}: recall@10 {:.3}, {per_query:.2?} per query",
                found as f64 / (queries.len() * 10) as f64
            );
        }
    }
}
