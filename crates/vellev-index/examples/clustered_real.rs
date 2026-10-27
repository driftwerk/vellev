//! Scratch: recall of a clustered index on the benchmark fixture's real
//! vectors, against the scan of more clusters and the shortlist's size.
//!
//!   clustered_real FIXTURE ROWS

use std::io::BufRead;
use std::time::Instant;

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::vector::{distance, in_row_order, VectorIndex};

fn shards(dir: &std::path::Path) -> Vec<f32> {
    let mut paths: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    paths.retain(|p| p.extension().is_some_and(|e| e == "f32") && !p.file_name().unwrap().to_string_lossy().starts_with("._"));
    paths.sort();
    let mut out = Vec::new();
    for p in paths {
        let bytes = std::fs::read(&p).unwrap();
        out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().unwrap());
    let rows: usize = args.next().unwrap().parse().unwrap();
    let d = 768;
    let all = shards(&dir.join("vectors/passages"));
    let vectors: Vec<&[f32]> = all.chunks_exact(d).take(rows).collect();
    // deleted rows are left out, as the truth leaves them out
    let deleted: Vec<bool> = std::io::BufReader::new(std::fs::File::open(dir.join("text/passages.jsonl")).unwrap())
        .lines()
        .take(rows)
        .map(|l| l.unwrap().contains("\"deleted\": true"))
        .collect();
    let live: Vec<(u32, &[f32])> = vectors.iter().enumerate().filter(|(n, _)| !deleted[*n]).map(|(n, v)| (n as u32, *v)).collect();
    let qall = shards(&dir.join("vectors/queries"));
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let started = Instant::now();
    let exact: Vec<Vec<u32>> = std::thread::scope(|scope| {
        let chunks: Vec<_> = queries.chunks(queries.len().div_ceil(4)).map(|qs| {
            let live = &live;
            scope.spawn(move || qs.iter().map(|q| {
                let mut best: Vec<(f32, u32)> = live.iter().map(|(n, v)| (distance(Distance::Cosine, q, v), *n)).collect();
                best.select_nth_unstable_by(10, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let mut top: Vec<_> = best[..10].to_vec();
                top.sort_by(|a, b| a.0.total_cmp(&b.0));
                top.into_iter().map(|(_, n)| n).collect::<Vec<u32>>()
            }).collect::<Vec<_>>())
        }).collect();
        chunks.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    println!("{} live rows, exact answers in {:.1?}", live.len(), started.elapsed());
    let skipped = vec![false; rows];
    for times in ["0.5", "2", "4"] {
    std::env::set_var("SCRATCH_CLUSTERS_TIMES", times);
    let started = Instant::now();
    let index = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    println!("clusters x{times}: built in {:.1?}: {} clusters, {} probes tuned", started.elapsed(), index.clusters(), index.probes());
    let tuned = index.probes();
    for multiple in [1usize, 2, 4, 6, 8, 12, 16] {
        let probing = index.clone().with_probes(tuned * multiple);
        for budget in [1280usize] {
            let mut found = 0;
            let mut top1 = 0;
            let started = Instant::now();
            for (q, truth) in queries.iter().zip(&exact) {
                let short = probing.estimates(q, budget, None, &skipped, &in_row_order);
                let mut scored: Vec<(f32, u32)> = short.iter().map(|(_, n)| (distance(Distance::Cosine, q, vectors[*n as usize]), *n)).collect();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let got: Vec<u32> = scored.iter().take(10).map(|(_, n)| *n).collect();
                found += got.iter().filter(|n| truth.contains(n)).count();
                top1 += usize::from(got.first() == truth.first());
            }
            let each = started.elapsed() / queries.len() as u32;
            println!(
                "probes {:>4} ({multiple:>2}x) budget {budget:>5}: recall@10 {:.4} same-top-1 {:.3} {:.2?} a query",
                tuned * multiple,
                found as f64 / (queries.len() * 10) as f64,
                top1 as f64 / queries.len() as f64,
                each
            );
        }
    }
    }
}
