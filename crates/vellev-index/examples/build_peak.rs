//! Scratch (not for merging): how long a clustered build of the fixture's
//! first ROWS passages takes, for `/usr/bin/time -v` to give its peak.
//!
//!   build_peak FIXTURE ROWS

use std::time::Instant;

use vellev_core::Distance;
use vellev_index::clustered::ClusteredIndex;

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
    let mut out = Vec::with_capacity(most);
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
    let d = 768;
    let all = shards(&dir.join("vectors/passages"), rows * d);
    let live: Vec<(u32, &[f32])> = all
        .chunks_exact(d)
        .enumerate()
        .map(|(row, v)| (row as u32, v))
        .collect();
    let begun = Instant::now();
    let index = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    println!(
        "rows {} clusters {} build {:.1} s",
        live.len(),
        index.clusters(),
        begun.elapsed().as_secs_f64()
    );
}
