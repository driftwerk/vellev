//! Embed a recall fixture with a local model, the way a collection that
//! embeds locally would: documents as they are, queries with the model's
//! query prefix. Reads `rows.jsonl` (`{id, content, ...}`) and
//! `queries.jsonl` (`{text}`) from a directory and writes beside them
//! `memories.jsonl` (`{id, content, vector}`) and `queries.vectors.jsonl`
//! (`{text, vector}`), the formats `recall_fixture.rs` reads.
//!
//! ```sh
//! cargo run --release -p vellev-embed --example recall_fixture -- \
//!   DIR [MODEL] [DIMENSIONS] [QUERY_PREFIX] [CACHE]
//! ```
//!
//! The defaults are bge-base-en-v1.5 at 768 dimensions with its query
//! prefix, as driftwerk's hub embeds memories.

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use serde_json::{json, Value};
use vellev_embed::{Local, Provider, Purpose};

const BATCH: usize = 256;

fn read(path: &PathBuf) -> Vec<Value> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    BufReader::new(file)
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect()
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("a directory"));
    let model = args
        .next()
        .unwrap_or_else(|| "BAAI/bge-base-en-v1.5".into());
    let dimensions: u32 = args.next().map_or(768, |d| d.parse().unwrap());
    let prefix = args
        .next()
        .unwrap_or_else(|| "Represent this sentence for searching relevant passages: ".into());
    let cache = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "/Volumes/T9/vellev-scratch/models".into()),
    );
    let local = Local::new(&model, dimensions, &cache).expect("the model");
    let rows = read(&dir.join("rows.jsonl"));
    let queries = read(&dir.join("queries.jsonl"));
    let started = std::time::Instant::now();
    let mut out = BufWriter::new(std::fs::File::create(dir.join("memories.jsonl")).unwrap());
    for (n, batch) in rows.chunks(BATCH).enumerate() {
        let texts: Vec<String> = batch
            .iter()
            .map(|row| row["content"].as_str().unwrap().to_string())
            .collect();
        let vectors = local.embed(texts, Purpose::Document).await.unwrap();
        for (row, vector) in batch.iter().zip(vectors) {
            let line = json!({ "id": row["id"], "content": row["content"], "vector": vector });
            writeln!(out, "{line}").unwrap();
        }
        if n % 40 == 0 {
            eprintln!(
                "{} of {} rows, {:?}",
                (n + 1) * BATCH,
                rows.len(),
                started.elapsed()
            );
        }
    }
    out.flush().unwrap();
    let mut out = BufWriter::new(std::fs::File::create(dir.join("queries.vectors.jsonl")).unwrap());
    for batch in queries.chunks(BATCH) {
        let texts: Vec<String> = batch
            .iter()
            .map(|query| format!("{prefix}{}", query["text"].as_str().unwrap()))
            .collect();
        let vectors = local.embed(texts, Purpose::Query).await.unwrap();
        for (query, vector) in batch.iter().zip(vectors) {
            let line = json!({ "text": query["text"], "vector": vector });
            writeln!(out, "{line}").unwrap();
        }
    }
    out.flush().unwrap();
    eprintln!("done in {:?}", started.elapsed());
}
