//! Scratch (not for merging): where a clustered search's time goes on this
//! machine. The fixture's first ROWS passages, one index, PROBES clusters,
//! 1,280 shortlisted and rescored, one thread: estimates and rescoring
//! timed apart, and a memory-bandwidth figure for scale. With PROFILE_SECS
//! set, after "profiling" is printed the estimates loop runs that long for
//! a profiler to attach to.
//!
//!   ivf_profile FIXTURE ROWS PROBES

use std::time::{Duration, Instant};

use vellev_core::Distance;
use vellev_index::clustered::{ClusteredIndex, Probing};
use vellev_index::vector::{in_row_order, Scorer};

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

fn stats(mut v: Vec<f64>) -> String {
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    format!(
        "mean {mean:.3} ms, p50 {:.3} ms, p99 {:.3} ms",
        v[v.len() / 2],
        v[v.len() * 99 / 100]
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().unwrap());
    let rows: usize = args.next().unwrap().parse().unwrap();
    let probes: usize = args.next().unwrap().parse().unwrap();
    let d = 768;
    let all = shards(&dir.join("vectors/passages"), rows * d);
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .enumerate()
        .map(|(n, v)| (n as u32, v))
        .collect();
    let qall = shards(&dir.join("vectors/queries"), usize::MAX);
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();

    // Memory bandwidth, one thread: a sum over 1 GiB, three times.
    let big: Vec<u64> = (0..(1usize << 27)).map(|i| i as u64).collect();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let at = Instant::now();
        std::hint::black_box(big.iter().fold(0u64, |a, b| a.wrapping_add(*b)));
        best = best.min(at.elapsed().as_secs_f64());
    }
    println!("one-thread read bandwidth: {:.1} GB/s", (8u64 << 27) as f64 / best / 1e9);
    drop(big);

    let at = Instant::now();
    let index = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    println!(
        "{} rows, {} clusters, built in {:.1?}; {probes} probes",
        live.len(),
        index.clusters(),
        at.elapsed()
    );
    let skipped = vec![false; rows];
    let probing = Probing { probes, parts: 1 };
    let estimate = |q: &[f32]| index.estimates_split(q, 1280, None, &skipped, &in_row_order, probing);
    for _ in 0..200 {
        std::hint::black_box(estimate(queries[0]));
    }
    let (mut est, mut res, mut whole) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..2 {
        for q in &queries {
            let a = Instant::now();
            let short = estimate(q);
            let b = Instant::now();
            let scorer = Scorer::new(Distance::Cosine, q);
            let mut scored: Vec<(f32, u32)> = short
                .iter()
                .map(|(_, row)| (scorer.distance(live[*row as usize].1), *row))
                .collect();
            let nth = 9.min(scored.len() - 1);
            scored.select_nth_unstable_by(nth, |x, y| x.0.total_cmp(&y.0));
            std::hint::black_box(&scored[..10]);
            let c = Instant::now();
            est.push((b - a).as_secs_f64() * 1e3);
            res.push((c - b).as_secs_f64() * 1e3);
            whole.push((c - a).as_secs_f64() * 1e3);
        }
    }
    println!("estimates (scan and shortlist): {}", stats(est));
    println!("rescoring 1,280 exactly:        {}", stats(res));
    println!("whole query:                    {}", stats(whole));
    // The estimates with every cluster's work but almost no rows: one probe,
    // so readying and ranking are seen against scanning.
    let one = Probing { probes: 1, parts: 1 };
    let mut few = Vec::new();
    for q in &queries {
        let a = Instant::now();
        std::hint::black_box(index.estimates_split(q, 1280, None, &skipped, &in_row_order, one));
        few.push(a.elapsed().as_secs_f64() * 1e3);
    }
    println!("estimates at 1 probe (query prepared, clusters ranked, one cluster): {}", stats(few));

    if let Ok(secs) = std::env::var("PROFILE_SECS") {
        let until = Instant::now() + Duration::from_secs(secs.parse().unwrap());
        println!("profiling");
        let mut n = 0usize;
        while Instant::now() < until {
            for q in &queries {
                std::hint::black_box(estimate(q));
                n += 1;
            }
        }
        println!("profiled {n} queries");
    }
}
