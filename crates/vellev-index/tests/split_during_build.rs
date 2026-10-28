//! A bench, not a check, run by hand or by a scratch workflow: a lone
//! reader's reads during an index build on rayon's global pool, unsplit
//! and split on the read pool, and the build's time, against the build's
//! time alone; the arms' order alternating from round to round. What
//! `split_for` declining to split during builds rests on (#146).
//!
//! With `RATE` unset or 0 the reader reads back to back, so a faster arm
//! also reads more during the build; with `RATE` reads a second, it reads
//! on that schedule, each read's time taken from when it was due, so both
//! arms read alike.
//!
//! ```sh
//! VELLEV_FIXTURE=/path/to/fixture RATE=30 cargo test --release -p vellev-index \
//!   --test split_during_build -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use vellev_core::Distance;
use vellev_index::clustered::{ClusteredIndex, Probing};
use vellev_index::vector::in_row_order;

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

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)]
}

#[test]
#[ignore = "a bench on the benchmark fixture: set VELLEV_FIXTURE and run with --ignored"]
fn a_lone_read_during_a_build_unsplit_and_split() {
    let dir = std::path::PathBuf::from(std::env::var("VELLEV_FIXTURE").expect("VELLEV_FIXTURE"));
    let rounds: usize = std::env::var("ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(3);
    let rate: f64 = std::env::var("RATE")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(0.0);
    let (rows, building) = (912_500usize, 300_000usize);
    let d = 768;
    let all = shards(&dir.join("vectors/passages"), rows * d);
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .enumerate()
        .map(|(n, v)| (n as u32, v))
        .collect();
    let built: Vec<(u32, &[f32])> = live[..building].to_vec();
    let qall = shards(&dir.join("vectors/queries"), usize::MAX);
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let index = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    let probes = index.probes();
    let threads = rayon::current_num_threads();
    let skipped = vec![false; rows];
    let search = |q: &[f32], parts: usize| {
        std::hint::black_box(index.estimates_split(
            q,
            1280,
            None,
            &skipped,
            &in_row_order,
            Probing { probes, parts },
        ));
    };
    println!(
        "{rows} rows, {} clusters, {probes} probes, {threads} threads; builds of {building} rows; {}",
        index.clusters(),
        if rate > 0.0 { format!("{rate} reads a second") } else { "reads back to back".to_string() }
    );
    // 0 parts: the build alone, no reads.
    let arms = [0, 1, threads];
    for round in 0..rounds {
        let mut order = arms;
        order.rotate_left(round % arms.len());
        if round % 2 == 1 {
            order.reverse();
        }
        for parts in order {
            let done = AtomicBool::new(false);
            let (took, mut during, late) = std::thread::scope(|scope| {
                let builder = scope.spawn(|| {
                    let at = Instant::now();
                    std::hint::black_box(ClusteredIndex::build(Distance::Cosine, &built));
                    done.store(true, Ordering::Relaxed);
                    at.elapsed().as_secs_f64()
                });
                let (mut times, mut late) = (Vec::new(), 0usize);
                let start = Instant::now();
                for (n, q) in queries.iter().cycle().enumerate() {
                    if parts == 0 || done.load(Ordering::Relaxed) {
                        break;
                    }
                    let due = if rate > 0.0 {
                        let due = start + Duration::from_secs_f64(n as f64 / rate);
                        match due.checked_duration_since(Instant::now()) {
                            Some(wait) => std::thread::sleep(wait),
                            None => late += 1,
                        }
                        due
                    } else {
                        Instant::now()
                    };
                    search(q, parts);
                    times.push(due.elapsed().as_secs_f64() * 1e3);
                }
                (builder.join().unwrap(), times, late)
            });
            if parts == 0 {
                println!("round {round}, build alone: the build took {took:.1} s");
                continue;
            }
            during.sort_by(f64::total_cmp);
            println!(
                "round {round}, {}: {} reads during the build ({late} late), p50 {:.2} ms p99 {:.2} ms; the build took {took:.1} s",
                if parts == 1 { "unsplit" } else { "split" },
                during.len(),
                percentile(&during, 0.5),
                percentile(&during, 0.99),
            );
        }
    }
}
