//! A bench, not a check, run by hand or by a scratch workflow: reads by
//! several readers at a fixed rate each, with no build and during an index
//! build, the build on a pool at the normal OS priority or niced, the
//! reads unsplit, split in two halves of the threads, or split over all of
//! them; and the build's time. Whether builds at a lower priority let
//! reads win the CPU with no heuristic in `split_for` (#146's follow-up).
//!
//! Each read is timed from when it began, and from when it was due, which
//! a reader that falls behind its schedule makes longer. Nicing works on
//! Linux, where a thread has a priority of its own; elsewhere the niced
//! arms are run as they are, and say so.
//!
//! ```sh
//! VELLEV_FIXTURE=/path/to/fixture READERS="2 4" RATE=15 NICE=10 \
//!   cargo test --release -p vellev-index --test build_priority -- --ignored --nocapture
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
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)]
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Lower the calling thread's priority by `nice`: on Linux, where each
/// thread has its own. Whether it was.
fn niced(nice: i32) -> bool {
    #[cfg(target_os = "linux")]
    {
        // Who 0 is the calling thread on Linux.
        unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = nice;
        false
    }
}

/// A pool of `threads` to build on, its threads niced by `nice`.
fn build_pool(threads: usize, nice: i32) -> (rayon::ThreadPool, bool) {
    let all = std::sync::Arc::new(AtomicBool::new(true));
    let pool = {
        let all = all.clone();
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(move |n| format!("build-nice{nice}-{n}"))
            .start_handler(move |_| {
                if nice != 0 && !niced(nice) {
                    all.store(false, Ordering::Relaxed);
                }
            })
            .build()
            .unwrap()
    };
    // Every thread started, and so niced or not.
    pool.broadcast(|_| ());
    (pool, nice != 0 && all.load(Ordering::Relaxed))
}

/// Reads' times, from when they began and from when they were due, and
/// how many began late.
#[derive(Default)]
struct Reads {
    begun: Vec<f64>,
    due: Vec<f64>,
    late: usize,
}

impl Reads {
    fn add(&mut self, other: Reads) {
        self.begun.extend(other.begun);
        self.due.extend(other.due);
        self.late += other.late;
    }

    fn said(mut self) -> String {
        self.begun.sort_by(f64::total_cmp);
        self.due.sort_by(f64::total_cmp);
        format!(
            "{} reads ({} late): from begun p50 {:.2} ms p99 {:.2} ms, from due p50 {:.2} ms p99 {:.2} ms",
            self.begun.len(),
            self.late,
            percentile(&self.begun, 0.5),
            percentile(&self.begun, 0.99),
            percentile(&self.due, 0.5),
            percentile(&self.due, 0.99),
        )
    }
}

#[test]
#[ignore = "a bench on the benchmark fixture: set VELLEV_FIXTURE and run with --ignored"]
fn reads_during_a_build_at_a_lower_priority() {
    let dir = std::path::PathBuf::from(std::env::var("VELLEV_FIXTURE").expect("VELLEV_FIXTURE"));
    let rounds: usize = env("ROUNDS", 2);
    let rate: f64 = env("RATE", 15.0);
    let nice: i32 = env("NICE", 10);
    let quiet: f64 = env("QUIET_SECS", 8.0);
    let readers: Vec<usize> = std::env::var("READERS")
        .unwrap_or_else(|_| "2 4".to_string())
        .split_whitespace()
        .map(|r| r.parse().unwrap())
        .collect();
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
    let pools = [build_pool(threads, 0), build_pool(threads, nice)];
    println!(
        "{rows} rows, {} clusters, {probes} probes, {threads} threads; builds of {building} rows; \
         {rate} reads a second a reader; nice {nice}: {}",
        index.clusters(),
        if pools[1].1 {
            "applied"
        } else {
            "NOT applied on this platform"
        },
    );

    // `count` readers reading at `rate` each, split into `parts`, until
    // `stop` says so.
    let (index, queries, skipped) = (&index, &queries, &skipped);
    let read = |count: usize, parts: usize, stop: &AtomicBool| -> Reads {
        std::thread::scope(|scope| {
            let readers: Vec<_> = (0..count)
                .map(|reader| {
                    scope.spawn(move || {
                        let mut reads = Reads::default();
                        let start = Instant::now()
                            + Duration::from_secs_f64(reader as f64 / (rate * count as f64));
                        for (n, q) in queries.iter().cycle().skip(reader * 97).enumerate() {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let due = start + Duration::from_secs_f64(n as f64 / rate);
                            match due.checked_duration_since(Instant::now()) {
                                Some(wait) => std::thread::sleep(wait),
                                None => reads.late += 1,
                            }
                            let began = Instant::now();
                            std::hint::black_box(index.estimates_split(
                                q,
                                1280,
                                None,
                                skipped,
                                &in_row_order,
                                Probing { probes, parts },
                            ));
                            reads.begun.push(began.elapsed().as_secs_f64() * 1e3);
                            reads.due.push(due.elapsed().as_secs_f64() * 1e3);
                        }
                        reads
                    })
                })
                .collect();
            let mut all = Reads::default();
            for reader in readers {
                all.add(reader.join().unwrap());
            }
            all
        })
    };

    let splits = {
        let mut splits = vec![1, (threads / 2).max(1), threads];
        splits.dedup();
        splits
    };
    for round in 0..rounds {
        for &count in &readers {
            // Phase one: the reads alone, no build.
            for &parts in &splits {
                let stop = AtomicBool::new(false);
                let reads = std::thread::scope(|scope| {
                    let reading = scope.spawn(|| read(count, parts, &stop));
                    std::thread::sleep(Duration::from_secs_f64(quiet));
                    stop.store(true, Ordering::Relaxed);
                    reading.join().unwrap()
                });
                println!(
                    "round {round}, {count} readers, split {parts}, no build: {}",
                    reads.said()
                );
            }
            // Phase two: during a build, at either priority, the arms'
            // order turned about from round to round.
            let mut arms: Vec<(usize, usize)> = (0..pools.len())
                .flat_map(|pool| splits.iter().map(move |&parts| (pool, parts)))
                .collect();
            let turn = round % arms.len();
            arms.rotate_left(turn);
            if round % 2 == 1 {
                arms.reverse();
            }
            for (pool, parts) in arms {
                let stop = AtomicBool::new(false);
                let (took, reads) = std::thread::scope(|scope| {
                    let reading = scope.spawn(|| read(count, parts, &stop));
                    let at = Instant::now();
                    pools[pool].0.install(|| {
                        std::hint::black_box(ClusteredIndex::build(Distance::Cosine, &built))
                    });
                    let took = at.elapsed().as_secs_f64();
                    stop.store(true, Ordering::Relaxed);
                    (took, reading.join().unwrap())
                });
                println!(
                    "round {round}, {count} readers, split {parts}, during a build at nice {}: {}; the build took {took:.1} s",
                    if pool == 0 { 0 } else { nice },
                    reads.said()
                );
            }
        }
        // And the build alone, at either priority.
        for (pool, (on, _)) in pools.iter().enumerate() {
            let at = Instant::now();
            on.install(|| std::hint::black_box(ClusteredIndex::build(Distance::Cosine, &built)));
            println!(
                "round {round}, build alone at nice {}: {:.1} s",
                if pool == 0 { 0 } else { nice },
                at.elapsed().as_secs_f64()
            );
        }
    }
}
