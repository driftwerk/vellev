//! Scratch: a lone read's time while a rows pass over a 912,500-row index
//! runs, the pass on one tuning thread or on rayon's global pool, and the
//! pass's own time.
//!
//!   tune_speed FIXTURE ROWS PROBES

use std::sync::atomic::{AtomicBool, Ordering};
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
    out.truncate(most);
    out
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
    let index = ClusteredIndex::build(Distance::Cosine, &live)
        .unwrap()
        .with_probes(probes);
    let skipped = vec![false; rows];
    let read = |q: &[f32]| {
        let at = Instant::now();
        std::hint::black_box(index.estimates(q, 1280, None, &skipped, &in_row_order));
        at.elapsed().as_secs_f64() * 1e3
    };
    let mut quiet: Vec<f64> = queries.iter().take(300).map(|q| read(q)).collect();
    quiet.sort_by(f64::total_cmp);
    let mode = if std::env::var("SCRATCH_GLOBAL_TUNING").is_ok() {
        "global pool"
    } else {
        "one tuning thread"
    };
    let done = AtomicBool::new(false);
    let (pass, mut during) = std::thread::scope(|scope| {
        let passing = scope.spawn(|| {
            let at = Instant::now();
            std::hint::black_box(index.curve_of_rows(&live));
            done.store(true, Ordering::Relaxed);
            at.elapsed().as_secs_f64()
        });
        let mut took = Vec::new();
        for q in queries.iter().cycle() {
            if done.load(Ordering::Relaxed) {
                break;
            }
            took.push(read(q));
        }
        (passing.join().unwrap(), took)
    });
    during.sort_by(f64::total_cmp);
    let p = |v: &[f64], q: f64| v[((v.len() as f64 * q) as usize).min(v.len() - 1)];
    println!(
        "{mode}: lone read quiet p50 {:.2} ms p99 {:.2} ms; during the pass ({} reads) p50 {:.2} ms p99 {:.2} ms max {:.2} ms; the pass took {pass:.1} s",
        p(&quiet, 0.5), p(&quiet, 0.99), during.len(), p(&during, 0.5), p(&during, 0.99), during.last().unwrap()
    );
}
