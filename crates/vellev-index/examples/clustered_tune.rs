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
    let mut truth_broad: HashMap<usize, Vec<(u64, f32)>> = HashMap::new();
    for l in
        std::io::BufReader::new(std::fs::File::open(dir.join("truth/truth.jsonl")).unwrap()).lines()
    {
        let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
        if v["case"] != "broad" {
            continue;
        }
        let pairs: Vec<(u64, f32)> = v["ids"]
            .as_array()
            .unwrap()
            .iter()
            .zip(v["scores"].as_array().unwrap())
            .map(|(i, s)| (i.as_u64().unwrap(), s.as_f64().unwrap() as f32))
            .collect();
        truth_broad.insert(v["query"].as_u64().unwrap() as usize, pairs);
    }
    let keyword: Vec<Vec<(u64, f32)>> =
        std::fs::read_to_string("/Volumes/T9/vellev-scratch/tmp/keyword-leg.jsonl")
            .map(|text| {
                text.lines()
                    .map(|l| {
                        let v: serde_json::Value = serde_json::from_str(l).unwrap();
                        v["found"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|p| (p[0].as_u64().unwrap(), p[1].as_f64().unwrap() as f32))
                            .collect()
                    })
                    .collect()
            })
            .unwrap_or_default();
    let fused = |meaning: &[(u64, f32)], q: usize| -> Vec<u64> {
        vellev_index::fusion::fuse(
            vellev_index::fusion::Fusion::RelativeScore,
            0.75,
            meaning,
            &keyword[q],
            10,
        )
        .iter()
        .map(|f| f.key)
        .collect()
    };
    let reference: Vec<Vec<u64>> = (0..keyword.len().min(queries.len()))
        .map(|q| fused(&truth_broad[&q][..100], q))
        .collect();
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
        if !reference.is_empty() {
            let mut settings: Vec<(String, Vec<usize>)> = [0.997, 0.998]
                .iter()
                .map(|target| {
                    (
                        format!("target {target}"),
                        local
                            .iter()
                            .map(|ranks| probes_for(ranks, *target))
                            .collect(),
                    )
                })
                .collect();
            settings.push((
                "every cluster".to_string(),
                segments.iter().map(|(_, _, i)| i.clusters()).collect(),
            ));
            for (target, probes) in settings {
                // The vector leg as the node asks it: 100 candidates, a
                // budget of 1,280 scored exactly, each layer's own nearest
                // 100 by estimate always scored, the rest the nearest
                // estimates across layers; rows past the clustered ones
                // scored whole.
                let found: usize = queries
                    .par_iter()
                    .enumerate()
                    .map(|(q, vector)| {
                        let mut scored: Vec<u32> = live
                            .iter()
                            .filter(|(n, _)| *n as usize >= clustered_rows)
                            .map(|(n, _)| *n)
                            .collect();
                        let mut pooled: Vec<(f32, u32)> = Vec::new();
                        for ((_, _, index), p) in segments.iter().zip(&probes) {
                            let mut estimates = index.scratch_estimates(vector, 1280, *p);
                            estimates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                            scored.extend(estimates.iter().take(100).map(|(_, n)| *n));
                            pooled.extend(estimates.into_iter().skip(100));
                        }
                        let floors = 100 * segments.len();
                        pooled.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        scored.extend(
                            pooled
                                .iter()
                                .take(1280usize.saturating_sub(floors))
                                .map(|(_, n)| *n),
                        );
                        let known: HashMap<u64, f32> = truth_broad[&q].iter().copied().collect();
                        let mut exact: Vec<(u64, f32)> = scored
                            .iter()
                            .map(|n| {
                                let id = *n as u64;
                                let cosine = known.get(&id).copied().unwrap_or_else(|| {
                                    1.0 - distance(Distance::Cosine, vector, live_vector(&all, *n))
                                });
                                (id, cosine)
                            })
                            .collect();
                        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                        exact.dedup_by_key(|p| p.0);
                        exact.truncate(100);
                        let got = fused(&exact, q);
                        reference[q].iter().filter(|id| got.contains(id)).count()
                    })
                    .sum();
                println!(
                    "hybrid at {target}, probes {probes:?}: recall@10 against the exact-leg answer {:.4}",
                    found as f64 / (reference.len() * 10) as f64
                );
            }
        }
        if std::env::var("SCRATCH_ORDERS").is_ok() && layout.len() == 1 {
            let (from, to, index) = &segments[0];
            let rows: Vec<(u32, &[f32])> = live
                .iter()
                .filter(|(n, _)| (*from..*to).contains(&(*n as usize)))
                .copied()
                .collect();
            let local_w = exact(
                &format!("{tmp}/exact-warmup-{from}-{to}.bin"),
                &warmup,
                &rows,
            );
            let mean = |vs: Vec<Vec<f32>>| -> Vec<f32> {
                let mut m = vec![0f32; vs[0].len()];
                for v in &vs {
                    for (a, b) in m.iter_mut().zip(v) {
                        *a += b / vs.len() as f32;
                    }
                }
                m
            };
            let mu_q = mean(warmup.iter().map(|q| index.scratch_prepared(q)).collect());
            let mu_p = mean(
                rows.iter()
                    .step_by(17)
                    .map(|(_, v)| index.scratch_prepared(v))
                    .collect(),
            );
            let sizes: Vec<usize> = {
                let mut s = vec![0usize; index.clusters()];
                for (n, _) in &rows {
                    s[index.scratch_cluster_of(*n).unwrap()] += 1;
                }
                s
            };
            for (label, shift, by_dot) in [
                ("distance", false, false),
                ("inner product", false, true),
                ("distance, shifted", true, false),
                ("inner product, shifted", true, true),
            ] {
                let ranks_of = |q: &[f32]| -> Vec<u32> {
                    let mut r = index.scratch_prepared(q);
                    if shift {
                        for ((x, a), b) in r.iter_mut().zip(&mu_q).zip(&mu_p) {
                            *x += b - a;
                        }
                    }
                    index.scratch_ranks_by(&r, by_dot)
                };
                let warm_ranks: Vec<u32> = warmup
                    .par_iter()
                    .zip(&local_w)
                    .flat_map_iter(|(q, near)| {
                        let ranks = ranks_of(q);
                        near.iter()
                            .map(|row| ranks[index.scratch_cluster_of(*row).unwrap()])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let measured_ranks: Vec<(Vec<u32>, Vec<u32>)> = queries
                    .par_iter()
                    .enumerate()
                    .map(|(q, v)| {
                        let ranks = ranks_of(v);
                        let mut order: Vec<u32> = (0..index.clusters() as u32).collect();
                        order.sort_by_key(|c| ranks[*c as usize]);
                        let near: Vec<u32> = truth_case[&(q, "broad".to_string())]
                            .iter()
                            .filter(|n| (*from..*to).contains(&(**n as usize)))
                            .map(|n| ranks[index.scratch_cluster_of(*n).unwrap()])
                            .collect();
                        (near, order)
                    })
                    .collect();
                for target in [0.997, 0.998] {
                    let p = probes_for(&warm_ranks, target);
                    let missed: usize = measured_ranks
                        .iter()
                        .map(|(near, _)| near.iter().filter(|r| **r as usize >= p).count())
                        .sum();
                    let codes: f64 = measured_ranks
                        .iter()
                        .map(|(_, order)| {
                            order[..p].iter().map(|c| sizes[*c as usize]).sum::<usize>() as f64
                        })
                        .sum::<f64>()
                        / queries.len() as f64;
                    println!(
                        "order by {label}, target {target}: probes {p} of {}, broad {:.4}, codes a query {codes:.0}",
                        index.clusters(),
                        1.0 - missed as f64 / (queries.len() * 10) as f64,
                    );
                }
            }
        }
        if std::env::var("SCRATCH_GUIDED").is_ok() && layout.len() == 1 {
            // Probing guided by questions seen: a measured question's
            // nearest warm-up questions, and the clusters their own nearest
            // rows lay in, ranked by how many; then the rest by centroid.
            let (from, to, index) = &segments[0];
            let rows: Vec<(u32, &[f32])> = live
                .iter()
                .filter(|(n, _)| (*from..*to).contains(&(*n as usize)))
                .copied()
                .collect();
            let local_w = exact(
                &format!("{tmp}/exact-warmup-{from}-{to}.bin"),
                &warmup,
                &rows,
            );
            let sizes: Vec<usize> = {
                let mut s = vec![0usize; index.clusters()];
                for (n, _) in &rows {
                    s[index.scratch_cluster_of(*n).unwrap()] += 1;
                }
                s
            };
            // Each warm-up question's neighbours' clusters.
            let held: Vec<Vec<usize>> = local_w
                .iter()
                .map(|near| {
                    near.iter()
                        .map(|r| index.scratch_cluster_of(*r).unwrap())
                        .collect()
                })
                .collect();
            for (guides, depth) in [(10usize, 10usize), (30, 10), (100, 10), (30, 100)] {
                // with depth 100: the warm-up questions' nearest 100 rows
                let held_deep: Vec<Vec<usize>> = if depth > 10 {
                    warmup
                        .par_iter()
                        .map(|q| {
                            let mut best: Vec<(f32, u32)> = rows
                                .iter()
                                .map(|(n, v)| (distance(Distance::Cosine, q, v), *n))
                                .collect();
                            best.select_nth_unstable_by(depth, |a, b| a.0.total_cmp(&b.0));
                            best[..depth]
                                .iter()
                                .map(|(_, r)| index.scratch_cluster_of(*r).unwrap())
                                .collect()
                        })
                        .collect()
                } else {
                    held.clone()
                };
                let measured_ranks: Vec<(Vec<u32>, Vec<u32>)> = queries
                    .par_iter()
                    .enumerate()
                    .map(|(q, v)| {
                        let base = index.scratch_cluster_ranks(v);
                        let mut nearest_w: Vec<(f32, usize)> = warmup
                            .iter()
                            .enumerate()
                            .map(|(w, wv)| (distance(Distance::Cosine, v, wv), w))
                            .collect();
                        nearest_w.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let mut votes = vec![0usize; index.clusters()];
                        for (_, w) in nearest_w.iter().take(guides) {
                            for c in &held_deep[*w] {
                                votes[*c] += 1;
                            }
                        }
                        let mut order: Vec<u32> = (0..index.clusters() as u32).collect();
                        order.sort_by_key(|c| {
                            (std::cmp::Reverse(votes[*c as usize]), base[*c as usize])
                        });
                        let mut ranks = vec![0u32; index.clusters()];
                        for (r, c) in order.iter().enumerate() {
                            ranks[*c as usize] = r as u32;
                        }
                        let near: Vec<u32> = truth_case[&(q, "broad".to_string())]
                            .iter()
                            .filter(|n| (*from..*to).contains(&(**n as usize)))
                            .map(|n| ranks[index.scratch_cluster_of(*n).unwrap()])
                            .collect();
                        (near, order)
                    })
                    .collect();
                for share in [0.05f64, 0.1, 0.2, 0.3, 0.45, 0.6] {
                    let p = (index.clusters() as f64 * share).ceil() as usize;
                    let missed: usize = measured_ranks
                        .iter()
                        .map(|(near, _)| near.iter().filter(|r| **r as usize >= p).count())
                        .sum();
                    let codes: f64 = measured_ranks
                        .iter()
                        .map(|(_, order)| {
                            order[..p].iter().map(|c| sizes[*c as usize]).sum::<usize>() as f64
                        })
                        .sum::<f64>()
                        / queries.len() as f64;
                    println!(
                        "guided by {guides} warm-up questions' nearest {depth}: probes {p} ({share}), broad {:.4}, codes {codes:.0}",
                        1.0 - missed as f64 / (queries.len() * 10) as f64
                    );
                }
            }
            // The same shares by centroid alone, for comparison.
            let base: Vec<Vec<u32>> = queries
                .par_iter()
                .map(|v| index.scratch_cluster_ranks(v))
                .collect();
            for share in [0.05f64, 0.1, 0.2, 0.3, 0.45, 0.6] {
                let p = (index.clusters() as f64 * share).ceil() as usize;
                let missed: usize = (0..queries.len())
                    .map(|q| {
                        truth_case[&(q, "broad".to_string())]
                            .iter()
                            .filter(|n| (*from..*to).contains(&(**n as usize)))
                            .filter(|n| {
                                base[q][index.scratch_cluster_of(**n).unwrap()] as usize >= p
                            })
                            .count()
                    })
                    .sum();
                println!(
                    "by centroid: probes {p} ({share}), broad {:.4}",
                    1.0 - missed as f64 / (queries.len() * 10) as f64
                );
            }
        }
        if std::env::var("SCRATCH_SOAR").is_ok() && layout.len() == 1 {
            let (from, to, index) = &segments[0];
            let rows: Vec<(u32, &[f32])> = live
                .iter()
                .filter(|(n, _)| (*from..*to).contains(&(*n as usize)))
                .copied()
                .collect();
            for lambda in [0.0f32, 1.0] {
                let started = Instant::now();
                let spills: HashMap<u32, (usize, usize)> = rows
                    .par_iter()
                    .map(|(n, v)| (*n, index.scratch_spill(v, lambda)))
                    .collect();
                let mut sizes = vec![0usize; index.clusters()];
                for (own, spill) in spills.values() {
                    sizes[*own] += 1;
                    sizes[*spill] += 1;
                }
                let plain: Vec<usize> = {
                    let mut s = vec![0usize; index.clusters()];
                    spills.values().for_each(|(own, _)| s[*own] += 1);
                    s
                };
                // Rank of the nearer of a row's two clusters.
                let rank = |ranks: &[u32], row: u32| -> u32 {
                    let (own, spill) = spills[&row];
                    ranks[own].min(ranks[spill])
                };
                let local_w = exact(
                    &format!("{tmp}/exact-warmup-{from}-{to}.bin"),
                    &warmup,
                    &rows,
                );
                let warm_ranks: Vec<u32> = warmup
                    .par_iter()
                    .zip(&local_w)
                    .flat_map_iter(|(v, near)| {
                        let ranks = index.scratch_cluster_ranks(v);
                        near.iter()
                            .map(|row| rank(&ranks, *row))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let measured_ranks: Vec<(Vec<u32>, Vec<u32>)> = queries
                    .par_iter()
                    .enumerate()
                    .map(|(q, v)| {
                        let ranks = index.scratch_cluster_ranks(v);
                        let mut order: Vec<u32> = (0..index.clusters() as u32).collect();
                        order.sort_by_key(|c| ranks[*c as usize]);
                        let near: Vec<u32> = truth_case[&(q, "broad".to_string())]
                            .iter()
                            .filter(|n| (*from..*to).contains(&(**n as usize)))
                            .map(|n| rank(&ranks, *n))
                            .collect();
                        (near, order)
                    })
                    .collect();
                for target in [0.997, 0.998] {
                    let p = probes_for(&warm_ranks, target);
                    let missed: usize = measured_ranks
                        .iter()
                        .map(|(near, _)| near.iter().filter(|r| **r as usize >= p).count())
                        .sum();
                    let codes: f64 = measured_ranks
                        .iter()
                        .map(|(_, order)| {
                            order[..p].iter().map(|c| sizes[*c as usize]).sum::<usize>() as f64
                        })
                        .sum::<f64>()
                        / queries.len() as f64;
                    let p0 = probes_for(&local[0], target);
                    let codes0: f64 = measured_ranks
                        .iter()
                        .map(|(_, order)| {
                            order[..p0]
                                .iter()
                                .map(|c| plain[*c as usize])
                                .sum::<usize>() as f64
                        })
                        .sum::<f64>()
                        / queries.len() as f64;
                    println!(
                        "spill lambda {lambda} (assigned in {:.0?}), target {target}: probes {p} (without spill {p0}), broad {:.4}, codes a query {:.0} (without spill {:.0})",
                        started.elapsed(),
                        1.0 - missed as f64 / (queries.len() * 10) as f64,
                        codes,
                        codes0
                    );
                }
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
