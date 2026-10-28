//! Scratch: probes tuned on questions seen (the warm-up set) against
//! probes tuned on rows, held out on the measured questions, for one and
//! two clustered segments; and stopping by patience.
//!
//!   clustered_tune FIXTURE

use std::collections::HashMap;
use std::io::BufRead;
use std::time::Instant;

use rayon::prelude::*;
use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
use vellev_index::vector::distance;

fn shards(dir: &std::path::Path) -> Vec<f32> {
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
        let bytes = std::fs::read(&p).unwrap();
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
    }
    out
}

/// Exact nearest 10 live rows of each query, cached on disk.
fn exact(cache: &str, queries: &[&[f32]], live: &[(u32, &[f32])]) -> Vec<Vec<u32>> {
    if let Ok(bytes) = std::fs::read(cache) {
        let ids: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        return ids.chunks(10).map(|c| c.to_vec()).collect();
    }
    let out: Vec<Vec<u32>> = queries
        .par_iter()
        .map(|q| {
            let mut best: Vec<(f32, u32)> = live
                .iter()
                .map(|(n, v)| (distance(Distance::Cosine, q, v), *n))
                .collect();
            best.select_nth_unstable_by(10, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let mut top: Vec<_> = best[..10].to_vec();
            top.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            top.into_iter().map(|(_, n)| n).collect()
        })
        .collect();
    let bytes: Vec<u8> = out.iter().flatten().flat_map(|n| n.to_le_bytes()).collect();
    std::fs::write(cache, bytes).unwrap();
    out
}

/// The smallest probes whose clusters hold `target` of the ranks.
fn probes_for(ranks: &[u32], target: f64) -> usize {
    if ranks.is_empty() {
        return 1;
    }
    let mut sorted = ranks.to_vec();
    sorted.sort_unstable();
    let at = ((ranks.len() as f64 * target).ceil() as usize).clamp(1, ranks.len());
    sorted[at - 1] as usize + 1
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let rows = 1_000_000usize;
    let d = 768;
    let all = shards(&dir.join("vectors/passages"));
    let lines: Vec<serde_json::Value> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/passages.jsonl")).unwrap())
            .lines()
            .take(rows)
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();
    let deleted: Vec<bool> = lines
        .iter()
        .map(|v| v["deleted"].as_bool().unwrap())
        .collect();
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .enumerate()
        .filter(|(n, _)| !deleted[*n])
        .map(|(n, v)| (n as u32, v))
        .collect();
    let qall = shards(&dir.join("vectors/queries"));
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let source = |file: &str| -> Vec<String> {
        std::io::BufReader::new(std::fs::File::open(dir.join(file)).unwrap())
            .lines()
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
                v["source"].as_str().unwrap().to_string()
            })
            .collect()
    };
    let source_q = source("text/queries.jsonl");
    let source_w = source("text/warmup.jsonl");
    let wall = shards(&dir.join("vectors/warmup"));
    let warmup: Vec<&[f32]> = wall.chunks_exact(d).collect();
    let started = Instant::now();
    let tmp = "/Volumes/T9/vellev-scratch/tmp";
    let truth_q = exact(&format!("{tmp}/exact-queries.bin"), &queries, &live);
    let truth_w = exact(&format!("{tmp}/exact-warmup.bin"), &warmup, &live);
    println!("exact answers in {:.1?}", started.elapsed());
    // The published truth for the filtered cases.
    let mut truth_case: HashMap<(usize, String), Vec<u32>> = HashMap::new();
    for l in
        std::io::BufReader::new(std::fs::File::open(dir.join("truth/truth.jsonl")).unwrap()).lines()
    {
        let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
        let ids: Vec<u32> = v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .take(10)
            .map(|i| i.as_u64().unwrap() as u32)
            .collect();
        truth_case.insert(
            (
                v["query"].as_u64().unwrap() as usize,
                v["case"].as_str().unwrap().to_string(),
            ),
            ids,
        );
    }
    let agree = (0..queries.len())
        .filter(|q| truth_case[&(*q, "broad".to_string())] == truth_q[*q])
        .count();
    println!(
        "own exact broad agrees with the published truth on {agree} of {}",
        queries.len()
    );

    let clustered_rows: usize = std::env::var("SCRATCH_CLUSTERED_ROWS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(912_500);
    let layouts: Vec<Vec<(usize, usize)>> = vec![
        vec![(0, clustered_rows)],
        vec![
            (0, clustered_rows / 2),
            (clustered_rows / 2, clustered_rows),
        ],
        vec![(0, 600_000), (600_000, clustered_rows)],
        vec![(0, 300_000), (300_000, 600_000), (600_000, clustered_rows)],
    ];
    let layouts: Vec<Vec<(usize, usize)>> = match std::env::var("SCRATCH_LAYOUTS") {
        Ok(which) => which
            .split(',')
            .map(|i| layouts[i.parse::<usize>().unwrap()].clone())
            .collect(),
        Err(_) => layouts,
    };
    let targets: [f64; 0] = [];
    for layout in layouts {
        println!("== layout {layout:?}, rows past {clustered_rows} scanned exactly");
        let mut segments = Vec::new();
        for (from, to) in &layout {
            let rows: Vec<(u32, &[f32])> = live
                .iter()
                .filter(|(n, _)| (*from..*to).contains(&(*n as usize)))
                .copied()
                .collect();
            let started = Instant::now();
            let index = ClusteredIndex::build(Distance::Cosine, &rows).unwrap();
            println!(
                "segment {from}..{to}: {} live rows, built in {:.1?}, {} clusters, {} probes tuned on rows",
                rows.len(),
                started.elapsed(),
                index.clusters(),
                index.probes()
            );
            segments.push((*from, *to, index));
        }
        // Per segment, the rank of the cluster of each true neighbour held
        // there, for warm-up and measured questions.
        let seg_of = |row: u32| {
            segments
                .iter()
                .position(|(from, to, _)| (*from..*to).contains(&(row as usize)))
        };
        let ranks_of =
            |qs: &[&[f32]], truth: &(dyn Fn(usize) -> Vec<u32> + Sync)| -> Vec<Vec<(usize, u32)>> {
                qs.par_iter()
                    .enumerate()
                    .map(|(q, vector)| {
                        let per: Vec<Vec<u32>> = segments
                            .iter()
                            .map(|(_, _, index)| index.scratch_cluster_ranks(vector))
                            .collect();
                        truth(q)
                            .into_iter()
                            .filter_map(|row| {
                                let s = seg_of(row)?;
                                let cluster = segments[s].2.scratch_cluster_of(row)?;
                                Some((s, per[s][cluster]))
                            })
                            .collect()
                    })
                    .collect()
            };
        let warm = ranks_of(&warmup, &|q| truth_w[q].clone());
        // As a node would know them: each segment's own nearest 10 of each
        // warm-up question, among its rows alone.
        let local: Vec<Vec<u32>> = segments
            .iter()
            .enumerate()
            .map(|(s, (from, to, index))| {
                let rows: Vec<(u32, &[f32])> = live
                    .iter()
                    .filter(|(n, _)| (*from..*to).contains(&(*n as usize)))
                    .copied()
                    .collect();
                let found = exact(
                    &format!("{tmp}/exact-warmup-{from}-{to}.bin"),
                    &warmup,
                    &rows,
                );
                let _ = s;
                warmup
                    .par_iter()
                    .zip(&found)
                    .flat_map_iter(|(vector, near)| {
                        let ranks = index.scratch_cluster_ranks(vector);
                        near.iter()
                            .map(|row| ranks[index.scratch_cluster_of(*row).unwrap()])
                            .collect::<Vec<_>>()
                    })
                    .collect()
            })
            .collect();
        let cases = ["broad", "mid", "narrow", "owner"];
        let measured: Vec<Vec<Vec<(usize, u32)>>> = cases
            .iter()
            .map(|case| ranks_of(&queries, &|q| truth_case[&(q, case.to_string())].clone()))
            .collect();
        // Recall@10's bound when segment s scans probes[s]: neighbours in
        // the exactly scanned rows always count.
        let coverage = |ranks: &[Vec<(usize, u32)>], probes: &[usize]| -> f64 {
            let missed: usize = ranks
                .iter()
                .map(|q| q.iter().filter(|(s, r)| *r as usize >= probes[*s]).count())
                .sum();
            1.0 - missed as f64 / (ranks.len() * 10) as f64
        };
        let subset = |ranks: &[Vec<(usize, u32)>], from: &str| -> Vec<Vec<(usize, u32)>> {
            ranks
                .iter()
                .enumerate()
                .filter(|(q, _)| from == "all" || source_q[*q] == from)
                .map(|(_, r)| r.clone())
                .collect()
        };
        for target in [0.997, 0.998] {
            let tune_local = |from: &str| -> Vec<usize> {
                local
                    .iter()
                    .map(|ranks| {
                        let picked: Vec<u32> = ranks
                            .chunks(10)
                            .enumerate()
                            .filter(|(q, _)| from == "all" || source_w[*q] == from)
                            .flat_map(|(_, r)| r.iter().copied())
                            .collect();
                        probes_for(&picked, target)
                    })
                    .collect()
            };
            for tuned_on in ["all", "squad", "nq"] {
                let probes = tune_local(tuned_on);
                let mut line = format!(
                    "local-truth target {target}, tuned on warm-up {tuned_on}: probes {probes:?};"
                );
                for measured_on in ["all", "squad", "nq"] {
                    line += &format!(
                        " measured {measured_on}: broad {:.4} mid {:.4} owner {:.4};",
                        coverage(&subset(&measured[0], measured_on), &probes),
                        coverage(&subset(&measured[1], measured_on), &probes),
                        coverage(&subset(&measured[3], measured_on), &probes)
                    );
                }
                println!("{line}");
            }
        }
        let rows_tuned: Vec<usize> = segments.iter().map(|(_, _, i)| i.probes()).collect();
        println!(
            "probes tuned on rows {rows_tuned:?}: broad {:.4} mid {:.4} narrow {:.4} owner {:.4}",
            coverage(&measured[0], &rows_tuned),
            coverage(&measured[1], &rows_tuned),
            coverage(&measured[2], &rows_tuned),
            coverage(&measured[3], &rows_tuned)
        );
        for target in targets {
            // Tuned on all 1000 warm-up questions, then on samples of 100,
            // 200 and 400 of them (five draws each) for the spread.
            let tune = |sample: &[usize]| -> Vec<usize> {
                (0..segments.len())
                    .map(|s| {
                        let ranks: Vec<u32> = sample
                            .iter()
                            .flat_map(|q| warm[*q].iter().filter(|(t, _)| *t == s).map(|(_, r)| *r))
                            .collect();
                        probes_for(&ranks, target)
                    })
                    .collect()
            };
            let everyone: Vec<usize> = (0..warmup.len()).collect();
            let probes = tune(&everyone);
            println!(
                "target {target}: warm-up tuned probes {probes:?}: broad {:.4} mid {:.4} narrow {:.4} owner {:.4}",
                coverage(&measured[0], &probes),
                coverage(&measured[1], &probes),
                coverage(&measured[2], &probes),
                coverage(&measured[3], &probes)
            );
            for size in [100usize, 200, 400] {
                let mut seen = Vec::new();
                for draw in 0..5u64 {
                    let mut state = 0x9e3779b97f4a7c15u64.wrapping_mul(draw + 1 + size as u64);
                    let mut pick: Vec<usize> = everyone.clone();
                    for i in (1..pick.len()).rev() {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        pick.swap(i, (state % (i as u64 + 1)) as usize);
                    }
                    let probes = tune(&pick[..size]);
                    seen.push((probes.clone(), coverage(&measured[0], &probes)));
                }
                let lo = seen.iter().map(|s| s.1).fold(1.0, f64::min);
                let hi = seen.iter().map(|s| s.1).fold(0.0, f64::max);
                println!(
                    "    {size} samples: broad {lo:.4}..{hi:.4}, probes {:?}",
                    seen.iter().map(|s| s.0.clone()).collect::<Vec<_>>()
                );
            }
        }
        // Stopping by patience, one segment at a time: probes scanned and
        // the recall of what is scanned, for the broad case.
        if layout.len() == 1 && std::env::var("SCRATCH_PATIENCE").is_ok() {
            let index = &segments[0].2;
            for (least, patience) in [(16, 16), (32, 32), (64, 32), (64, 64), (100, 100)] {
                let started = Instant::now();
                let outcome: Vec<(usize, usize)> = queries
                    .iter()
                    .zip(&truth_q)
                    .map(|(q, truth)| {
                        let (short, probed) = index.scratch_patience(q, 1280, least, patience, 10);
                        let mut scored: Vec<(f32, u32)> = short
                            .iter()
                            .map(|(_, n)| {
                                (distance(Distance::Cosine, q, live_vector(&all, *n)), *n)
                            })
                            .collect();
                        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        let got: Vec<u32> = scored.iter().take(10).map(|(_, n)| *n).collect();
                        // neighbours past the clustered rows count as found
                        let found = truth
                            .iter()
                            .filter(|n| got.contains(n) || **n as usize >= clustered_rows)
                            .count();
                        (found, probed)
                    })
                    .collect();
                let mut probed: Vec<usize> = outcome.iter().map(|o| o.1).collect();
                probed.sort_unstable();
                println!(
                    "patience least {least} quiet {patience}: recall {:.4}, probes mean {:.0} p50 {} p99 {}, {:.2?} a query",
                    outcome.iter().map(|o| o.0).sum::<usize>() as f64 / (queries.len() * 10) as f64,
                    probed.iter().sum::<usize>() as f64 / probed.len() as f64,
                    probed[probed.len() / 2],
                    probed[probed.len() * 99 / 100],
                    started.elapsed() / queries.len() as u32
                );
            }
        }
    }
}

fn live_vector(all: &[f32], row: u32) -> &[f32] {
    &all[row as usize * 768..][..768]
}
