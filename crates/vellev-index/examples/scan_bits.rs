//! Scratch (not for merging): a clustered scan's estimates with a share of
//! rows dead (skipped), one thread and split in four, on the fixture.
//!
//!   scan_dead FIXTURE ROWS PROBES

use std::time::Instant;

use vellev_core::Distance;
use vellev_index::clustered::{ClusteredIndex, Probing};
use vellev_index::vector::{in_row_order, Skipped};

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
    let index = ClusteredIndex::build(Distance::Cosine, &live).unwrap();
    let dead_at = |share: f64| -> Vec<bool> {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        (0..rows)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 11) as f64 / (1u64 << 53) as f64) < share
            })
            .collect()
    };
    let bits_of = |dead: &[bool]| -> Vec<u64> {
        dead.chunks(64)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u64, |w, (at, d)| w | (u64::from(*d) << at))
            })
            .collect()
    };
    let (one, thirty) = (dead_at(0.01), dead_at(0.30));
    let (one_bits, thirty_bits) = (bits_of(&one), bits_of(&thirty));
    for round in 0..3 {
        for (what, skipped) in [
            ("1% dead, bytes (#157)", Skipped::Bytes(&one)),
            ("1% dead, bits", Skipped::Bits(&one_bits)),
            ("30% dead, bytes (#157)", Skipped::Bytes(&thirty)),
            ("30% dead, bits", Skipped::Bits(&thirty_bits)),
        ] {
            for parts in [1, 4] {
                let probing = Probing { probes, parts };
                for q in &queries[..50] {
                    std::hint::black_box(index.estimates_marked(
                        q,
                        1280,
                        None,
                        skipped,
                        &in_row_order,
                        probing,
                    ));
                }
                let mut took = Vec::new();
                for q in &queries {
                    let at = Instant::now();
                    std::hint::black_box(index.estimates_marked(
                        q,
                        1280,
                        None,
                        skipped,
                        &in_row_order,
                        probing,
                    ));
                    took.push(at.elapsed().as_secs_f64() * 1e3);
                }
                took.sort_by(f64::total_cmp);
                let mean = took.iter().sum::<f64>() / took.len() as f64;
                println!(
                    "round {round}, {what}, parts {parts}: mean {mean:.3} ms, p50 {:.3} ms, p99 {:.3} ms",
                    took[took.len() / 2],
                    took[took.len() * 99 / 100]
                );
            }
        }
    }
}
