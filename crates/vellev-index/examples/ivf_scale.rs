//! Scratch: the clustered index on the first N passages, against their
//! own exact truth: the probes a target needs (tuned on the warm-up
//! questions), the recall and codes a measured question gets, and its
//! single-thread time. How the cost of a recall grows with N.
//!
//!   ivf_scale FIXTURE N

use std::io::BufRead;
use std::time::Instant;

use rayon::prelude::*;
use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::vector::{distance, in_row_order, VectorIndex};

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
    out.truncate(most);
    out
}

fn exact(queries: &[&[f32]], live: &[(u32, &[f32])]) -> Vec<Vec<u32>> {
    queries
        .par_iter()
        .map(|q| {
            let mut best: Vec<(f32, u32)> = live
                .iter()
                .map(|(n, v)| (distance(Distance::Cosine, q, v), *n))
                .collect();
            best.select_nth_unstable_by(10, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            best[..10].iter().map(|(_, n)| *n).collect()
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().unwrap());
    let n: usize = args.next().unwrap().parse().unwrap();
    let d = 768;
    let all = shards(&dir.join("vectors/passages"), n * d);
    let deleted: Vec<bool> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/passages.jsonl")).unwrap())
            .lines()
            .take(n)
            .map(|l| l.unwrap().contains("\"deleted\": true"))
            .collect();
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .enumerate()
        .filter(|(i, _)| !deleted[*i])
        .map(|(i, v)| (i as u32, v))
        .collect();
    let qall = shards(&dir.join("vectors/queries"), usize::MAX);
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let wall = shards(&dir.join("vectors/warmup"), usize::MAX);
    let warmup: Vec<&[f32]> = wall.chunks_exact(d).collect();
    let truth_q = exact(&queries, &live);
    let truth_w = exact(&warmup, &live);
    let index = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    let clusters = index.clusters();
    let mut sizes = vec![0usize; clusters];
    for (row, _) in &live {
        sizes[index.scratch_cluster_of(*row).unwrap()] += 1;
    }
    let ranks = |qs: &[&[f32]], truth: &[Vec<u32>]| -> Vec<Vec<u32>> {
        qs.par_iter()
            .zip(truth)
            .map(|(q, near)| {
                let r = index.scratch_cluster_ranks(q);
                near.iter()
                    .map(|row| r[index.scratch_cluster_of(*row).unwrap()])
                    .collect()
            })
            .collect()
    };
    let warm = ranks(&warmup, &truth_w);
    let measured = ranks(&queries, &truth_q);
    let skipped = vec![false; n];
    println!("n={n}: {} live rows, {clusters} clusters", live.len());
    for target in [0.995, 0.997, 0.998] {
        let mut flat: Vec<u32> = warm.iter().flatten().copied().collect();
        flat.sort_unstable();
        let at = ((flat.len() as f64 * target).ceil() as usize).clamp(1, flat.len());
        let probes = flat[at - 1] as usize + 1;
        let missed: usize = measured
            .iter()
            .map(|r| r.iter().filter(|x| **x as usize >= probes).count())
            .sum();
        let probing = index.clone().with_probes(probes);
        let mut took = Vec::new();
        let mut codes = 0usize;
        for q in &queries {
            let order: Vec<u32> = {
                let r = index.scratch_cluster_ranks(q);
                let mut o: Vec<u32> = (0..clusters as u32).collect();
                o.sort_by_key(|c| r[*c as usize]);
                o
            };
            codes += order[..probes]
                .iter()
                .map(|c| sizes[*c as usize])
                .sum::<usize>();
            let started = Instant::now();
            let short = probing.estimates(q, 1280, None, &skipped, &in_row_order);
            // Rescored exactly, as the storage layer does: every
            // shortlisted row's vector read and scored, the nearest ten kept.
            let mut scored: Vec<(f32, u32)> = short
                .iter()
                .map(|(_, row)| {
                    (
                        distance(Distance::Cosine, q, &all[*row as usize * d..][..d]),
                        *row,
                    )
                })
                .collect();
            let nth = 9.min(scored.len().saturating_sub(1));
            scored.select_nth_unstable_by(nth, |a, b| a.0.total_cmp(&b.0));
            std::hint::black_box(&scored[..10.min(scored.len())]);
            took.push(started.elapsed().as_secs_f64() * 1e3);
        }
        took.sort_by(f64::total_cmp);
        // Saturated: SCRATCH_THREADS threads asking at once, each query
        // estimated and rescored as above.
        let threads: usize = std::env::var("SCRATCH_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(3);
        let started = Instant::now();
        let asked: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (probing, queries, skipped, all) = (&probing, &queries, &skipped, &all);
                    scope.spawn(move || {
                        let mut n = 0;
                        for q in queries.iter().cycle().skip(t * 97).take(queries.len()) {
                            let short = probing.estimates(q, 1280, None, skipped, &in_row_order);
                            let mut scored: Vec<(f32, u32)> = short
                                .iter()
                                .map(|(_, row)| {
                                    (
                                        distance(
                                            Distance::Cosine,
                                            q,
                                            &all[*row as usize * d..][..d],
                                        ),
                                        *row,
                                    )
                                })
                                .collect();
                            let nth = 9.min(scored.len().saturating_sub(1));
                            scored.select_nth_unstable_by(nth, |a, b| a.0.total_cmp(&b.0));
                            std::hint::black_box(&scored);
                            n += 1;
                        }
                        n
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let saturated = asked as f64 / started.elapsed().as_secs_f64();
        println!(
            "n={n} target {target}: {threads} threads saturated, {saturated:.0} queries a second"
        );
        println!(
            "n={n} target {target}: probes {probes} of {clusters} ({:.0}%), recall bound {:.4}, codes a query {}, mean {:.2} ms, p99 {:.2} ms",
            100.0 * probes as f64 / clusters as f64,
            1.0 - missed as f64 / (queries.len() * 10) as f64,
            codes / queries.len(),
            took.iter().sum::<f64>() / took.len() as f64,
            took[took.len() * 99 / 100]
        );
    }
}
