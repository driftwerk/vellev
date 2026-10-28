//! Scratch: the clustered scan's speed on the fixture's real vectors, at
//! several probe counts, one thread.
//!
//!   scan_speed FIXTURE ROWS PROBES,PROBES,...

use std::time::Instant;

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::vector::{in_row_order, VectorIndex};

fn shards(dir: &std::path::Path, most: usize) -> Vec<f32> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.retain(|p| {
        p.extension().is_some_and(|e| e == "f32")
            && !p.file_name().unwrap().to_string_lossy().starts_with("._")
    });
    paths.sort();
    let mut out = Vec::new();
    for p in paths {
        if out.len() >= most {
            break;
        }
        let bytes = std::fs::read(&p).unwrap();
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().unwrap());
    let rows: usize = args.next().unwrap().parse().unwrap();
    let probes: Vec<usize> = args
        .next()
        .unwrap()
        .split(',')
        .map(|p| p.parse().unwrap())
        .collect();
    let d = 768;
    let all = shards(&dir.join("vectors/passages"), rows * d);
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .take(rows)
        .enumerate()
        .map(|(n, v)| (n as u32, v))
        .collect();
    let qall = shards(&dir.join("vectors/queries"), usize::MAX);
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let started = Instant::now();
    let built = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    println!(
        "{} rows, {} clusters, built in {:.1?}",
        live.len(),
        built.clusters(),
        started.elapsed()
    );
    let skipped = vec![false; rows];
    let per_cluster = live.len() as f64 / built.clusters() as f64;
    let checksum = |found: &[(f32, u32)]| {
        found
            .iter()
            .map(|(d, r)| d.to_bits() as u64 ^ *r as u64)
            .fold(0u64, |a, b| a.wrapping_add(b))
    };
    for p in &probes {
        let index = built.clone().with_probes(*p);
        let mut sum = 0u64;
        let mut took = Vec::new();
        for _ in 0..2 {
            took.clear();
            for q in &queries {
                let at = Instant::now();
                let found = index.estimates(q, 1280, None, &skipped, &in_row_order);
                took.push(at.elapsed().as_secs_f64() * 1e3);
                sum = sum.wrapping_add(checksum(&found));
            }
        }
        took.sort_by(f64::total_cmp);
        // Saturated: `threads` threads each asking the questions in turn at
        // once, so every core is busy: throughput, and each query's time
        // while all compete for memory.
        let threads: usize = std::env::var("SCRATCH_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(3);
        let started = Instant::now();
        let mut busy: Vec<f64> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let index = &index;
                    let queries = &queries;
                    let skipped = &skipped;
                    scope.spawn(move || {
                        let mut took = Vec::new();
                        for q in queries.iter().cycle().skip(t * 97).take(queries.len()) {
                            let at = Instant::now();
                            let found = index.estimates(q, 1280, None, skipped, &in_row_order);
                            std::hint::black_box(found);
                            took.push(at.elapsed().as_secs_f64() * 1e3);
                        }
                        took
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let wall = started.elapsed().as_secs_f64();
        busy.sort_by(f64::total_cmp);
        println!(
            "probes {p}, {threads} threads saturated: {:.0} queries a second, each p50 {:.2} ms p99 {:.2} ms",
            busy.len() as f64 / wall,
            busy[busy.len() / 2],
            busy[busy.len() * 99 / 100]
        );
        let mean = took.iter().sum::<f64>() / took.len() as f64;
        println!(
            "probes {p}: mean {mean:.2} ms, p50 {:.2} ms, p99 {:.2} ms, {:.1} ns a code (checksum {sum:x})",
            took[took.len() / 2],
            took[took.len() * 99 / 100],
            mean * 1e6 / (*p as f64 * per_cluster)
        );
    }
}
