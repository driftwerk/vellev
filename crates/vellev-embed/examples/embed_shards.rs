//! Embed a large fixture with a local model in shards that survive a
//! stop: each shard is written whole, synced and then renamed into place,
//! and a run skips the shards already there. So a run on a laptop that
//! sleeps, or is stopped, goes on from the last whole shard.
//!
//! Reads `INPUT` (`{id, text}` per line, ids one after another from any
//! first, as the warm-up questions start after the measured ones) and writes
//! `OUT/NNNNN.f32`: the shard's vectors as little-endian `f32`s, row after
//! row, in the order of the input. A shard is `SHARD` rows, the last one
//! fewer. `PREFIX`, when given, is put before each text, as a query is
//! embedded.
//!
//! ```sh
//! cargo run --release -p vellev-embed --features local --example embed_shards -- \
//!   INPUT OUT [SHARD] [PREFIX] [MODEL] [DIMENSIONS] [CACHE]
//! ```
//!
//! It says how many rows a second it embedded, over the shards it made.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;
use vellev_embed::{Local, Provider, Purpose};

/// Texts given to the model at once.
const BATCH: usize = 256;

fn texts(input: &Path) -> Vec<String> {
    let file = std::fs::File::open(input).unwrap_or_else(|e| panic!("{}: {e}", input.display()));
    BufReader::new(file)
        .lines()
        .scan(None, |next: &mut Option<u64>, line| {
            let row: Value = serde_json::from_str(&line.unwrap()).unwrap();
            let id = row["id"].as_u64().expect("an id");
            let n = *next.get_or_insert(id);
            *next = Some(n + 1);
            assert_eq!(id, n, "rows are in id order, one after another");
            Some(row["text"].as_str().expect("a text").to_string())
        })
        .collect()
}

/// Writes `bytes` to `path` so that it is there whole or not at all.
fn write_whole(path: &Path, bytes: &[u8]) {
    let partial = path.with_extension("partial");
    let mut file = std::fs::File::create(&partial).unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
    std::fs::rename(&partial, path).unwrap();
    std::fs::File::open(path.parent().unwrap())
        .and_then(|dir| dir.sync_all())
        .unwrap();
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().expect("an input"));
    let out = PathBuf::from(args.next().expect("an output directory"));
    let shard: usize = args.next().map_or(10_000, |n| n.parse().unwrap());
    let prefix = args.next().unwrap_or_default();
    let model = args
        .next()
        .unwrap_or_else(|| "BAAI/bge-base-en-v1.5".into());
    let dimensions: u32 = args.next().map_or(768, |d| d.parse().unwrap());
    let cache = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/Volumes/T9/vellev-scratch/models".into()),
    );
    std::fs::create_dir_all(&out).unwrap();
    let texts = texts(&input);
    let local = Local::new(&model, dimensions, &cache).expect("the model");
    let purpose = match prefix.is_empty() {
        true => Purpose::Document,
        false => Purpose::Query,
    };
    let started = std::time::Instant::now();
    let mut embedded = 0usize;
    for (n, rows) in texts.chunks(shard).enumerate() {
        let path = out.join(format!("{n:05}.f32"));
        let expected = rows.len() * dimensions as usize * 4;
        match std::fs::metadata(&path) {
            Ok(meta) if meta.len() as usize == expected => continue,
            Ok(meta) => panic!("{} is {} bytes, not {expected}", path.display(), meta.len()),
            Err(_) => {}
        }
        let mut bytes = Vec::with_capacity(expected);
        for batch in rows.chunks(BATCH) {
            let batch: Vec<String> = batch.iter().map(|t| format!("{prefix}{t}")).collect();
            for vector in local.embed(batch, purpose).await.unwrap() {
                assert_eq!(vector.len(), dimensions as usize);
                for x in vector {
                    bytes.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        write_whole(&path, &bytes);
        embedded += rows.len();
        let seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "shard {n}: {embedded} rows in {seconds:.0} s, {:.1} rows/s",
            embedded as f64 / seconds
        );
    }
    eprintln!("done: {embedded} rows embedded in {:?}", started.elapsed());
}
