//! A bench, not a check, run by hand or by a scratch workflow: a lone
//! read's time during an index build on rayon's global pool, unsplit and
//! split on the read pool, the order alternating from round to round, and
//! the build's time. What `split_for` declining to split during builds
//! rests on (#146).
//!
//! ```sh
//! VELLEV_FIXTURE=/path/to/fixture RAYON_NUM_THREADS=3 cargo test --release -p vellev-index \
//!   --test split_during_build -- --ignored --nocapture
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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
    let read = |q: &[f32], parts: usize| {
        let at = Instant::now();
        std::hint::black_box(index.estimates_split(
            q,
            1280,
            None,
            &skipped,
            &in_row_order,
            Probing { probes, parts },
        ));
        at.elapsed().as_secs_f64() * 1e3
    };
    println!(
        "{rows} rows, {} clusters, {probes} probes, {threads} threads; builds of {building} rows",
        index.clusters()
    );
    for round in 0..rounds {
        let order = if round % 2 == 0 {
            [1, threads]
        } else {
            [threads, 1]
        };
        for parts in order {
            let done = AtomicBool::new(false);
            let (took, mut during) = std::thread::scope(|scope| {
                let builder = scope.spawn(|| {
                    let at = Instant::now();
                    std::hint::black_box(ClusteredIndex::build(Distance::Cosine, &built));
                    done.store(true, Ordering::Relaxed);
                    at.elapsed().as_secs_f64()
                });
                let mut times = Vec::new();
                for q in queries.iter().cycle() {
                    if done.load(Ordering::Relaxed) {
                        break;
                    }
                    times.push(read(q, parts));
                }
                (builder.join().unwrap(), times)
            });
            during.sort_by(f64::total_cmp);
            println!(
                "round {round}, {}: {} reads during the build, p50 {:.2} ms p99 {:.2} ms; the build took {took:.1} s",
                if parts == 1 { "unsplit" } else { "split" },
                during.len(),
                percentile(&during, 0.5),
                percentile(&during, 0.99),
            );
        }
    }
}
