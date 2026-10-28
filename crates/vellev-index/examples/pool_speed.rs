//! Scratch: a split read on the read pool against one on rayon's global
//! pool, while an index build runs on the global pool: a lone split
//! query's time, saturated queries a second, and the build's own time.
//!
//!   pool_speed FIXTURE ROWS BUILD_ROWS PROBES

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;
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

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().unwrap());
    let rows: usize = args.next().unwrap().parse().unwrap();
    let build_rows: usize = args.next().unwrap().parse().unwrap();
    let probes: usize = args.next().unwrap().parse().unwrap();
    let d = 768;
    let all = shards(&dir.join("vectors/passages"), rows.max(build_rows) * d);
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .take(rows)
        .enumerate()
        .map(|(n, v)| (n as u32, v))
        .collect();
    let building: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .take(build_rows)
        .enumerate()
        .map(|(n, v)| (n as u32, v))
        .collect();
    let qall = shards(&dir.join("vectors/queries"), usize::MAX);
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let index = ClusteredIndex::build(Distance::Cosine, &live)
        .unwrap()
        .with_probes(probes);
    let skipped = vec![false; rows];
    let threads = rayon::current_num_threads();
    let mode = if std::env::var("SCRATCH_GLOBAL_READS").is_ok() {
        "global pool"
    } else {
        "read pool"
    };
    println!("{mode}: {rows} rows, {threads} threads, probes {probes}");

    let lone = |split: usize, count: usize| -> (f64, f64) {
        let mut took: Vec<f64> = queries
            .iter()
            .cycle()
            .take(count)
            .map(|q| {
                let at = Instant::now();
                std::hint::black_box(index.estimates_split(
                    q,
                    1280,
                    None,
                    &skipped,
                    &in_row_order,
                    split,
                ));
                at.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        took.sort_by(f64::total_cmp);
        (took[took.len() / 2], took[took.len() * 99 / 100])
    };
    let saturated = |split: usize, clients: usize, seconds: f64| -> f64 {
        let stop = AtomicBool::new(false);
        let started = Instant::now();
        let done: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..clients)
                .map(|c| {
                    let (index, queries, skipped, stop) = (&index, &queries, &skipped, &stop);
                    scope.spawn(move || {
                        let mut n = 0;
                        for q in queries.iter().cycle().skip(c * 97) {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            std::hint::black_box(index.estimates_split(
                                q,
                                1280,
                                None,
                                skipped,
                                &in_row_order,
                                split,
                            ));
                            n += 1;
                        }
                        n
                    })
                })
                .collect();
            std::thread::sleep(Duration::from_secs_f64(seconds));
            stop.store(true, Ordering::Relaxed);
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        done as f64 / started.elapsed().as_secs_f64()
    };
    // The build alone.
    let at = Instant::now();
    std::hint::black_box(ClusteredIndex::build(Distance::Cosine, &building));
    let alone = at.elapsed().as_secs_f64();
    let (p50, p99) = lone(threads, 300);
    println!(
        "no build: lone split query p50 {p50:.2} ms p99 {p99:.2} ms; build alone {alone:.1} s"
    );
    // While a build runs on the global pool.
    for (what, split, clients) in [
        ("lone split query", threads, 1usize),
        ("saturated, unsplit", 1, threads),
        ("saturated, split", threads, threads),
    ] {
        let stop = AtomicBool::new(false);
        let (built, measured) = std::thread::scope(|scope| {
            let builder = scope.spawn(|| {
                let at = Instant::now();
                std::hint::black_box(ClusteredIndex::build(Distance::Cosine, &building));
                let took = at.elapsed().as_secs_f64();
                stop.store(true, Ordering::Relaxed);
                took
            });
            std::thread::sleep(Duration::from_millis(500));
            let measured = if clients == 1 {
                let (p50, p99) = lone(split, 300);
                format!("p50 {p50:.2} ms p99 {p99:.2} ms")
            } else {
                format!("{:.0} queries a second", saturated(split, clients, 10.0))
            };
            // Keep the readers busy until the build is done, as traffic would.
            while !stop.load(Ordering::Relaxed) {
                std::hint::black_box(index.estimates_split(
                    queries[0],
                    1280,
                    None,
                    &skipped,
                    &in_row_order,
                    split,
                ));
            }
            (builder.join().unwrap(), measured)
        });
        println!(
            "during a build: {what}: {measured}; the build took {built:.1} s (alone {alone:.1} s)"
        );
    }
}
