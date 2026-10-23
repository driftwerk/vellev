//! Measure what a segment the writer has just built keeps in memory: its
//! rows are committed, flushed into one segment, then more are committed,
//! flushed and merged with it. After each step the partition's own count
//! (`State::resident`) is printed beside the bytes its vectors take, and the
//! process's anonymous memory as the OS counts it.
//!
//!   cargo run --release -p vellev-storage --example built_resident
//!
//! Environment: OBJECTS (200000), DIMS (384), STORE (a directory to put
//! the store's files in, standing for object storage; `.` if unset), CACHE (a directory to keep
//! the local copies in; a fresh temporary one if unset), HELD=1 to run on a
//! store that keeps no copies, so that every vector is held in memory, for
//! comparison.

use std::sync::Arc;
use std::time::{Duration, Instant};

use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition, Property, TenantId,
};
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Mutation, State, Writer};

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The process's anonymous memory in MiB as the OS counts it: on macOS its
/// physical footprint, which leaves out clean pages mapped from files; on
/// Linux `RssAnon`.
fn anonymous_mib() -> String {
    let pid = std::process::id().to_string();
    if cfg!(target_os = "linux") {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        return status
            .lines()
            .find(|line| line.starts_with("RssAnon:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<u64>().ok())
            .map_or("?".into(), |kb| (kb / 1024).to_string());
    }
    let output = std::process::Command::new("footprint")
        .args(["-p", &pid])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .unwrap_or_default();
    // "Footprint: 123 MB" or "... 1.2 GB" on the summary line.
    output
        .lines()
        .find(|line| line.contains("Footprint:"))
        .map_or("?".into(), |line| line.trim().to_string())
}

fn collection(dimensions: u32) -> Collection {
    Collection {
        name: CollectionName::new("Entities").expect("valid name"),
        properties: vec![Property::new("line", DataType::Int)],
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "supplied".into(),
            dimensions,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let objects = setting("OBJECTS", 200_000) as usize;
    let dimensions = setting("DIMS", 384) as usize;
    let held = std::env::var("HELD").is_ok_and(|held| held == "1");
    let key = PartitionKey {
        tenant: TenantId::new("bench")?,
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await?;
    let temporary = tempfile::tempdir()?;
    let cache = std::env::var("CACHE").map_or(temporary.path().to_path_buf(), Into::into);
    // The store a node writes to, outside the process as object storage
    // is: on local disk, but with no copies of its own to read in place.
    let remote = tempfile::tempdir_in(std::env::var("STORE").unwrap_or_else(|_| ".".into()))?;
    let local = object_store::local::LocalFileSystem::new_with_prefix(remote.path())?;
    let store: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(Arc::new(local)));
    let blobs: Arc<dyn Blobs> = match held {
        true => store,
        false => Arc::new(CachedBlobs::new(store, &cache)),
    };
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("bench"), Duration::from_secs(3600))
        .await?
    else {
        return Err("the lease is held".into());
    };
    let writer = Writer::open(blobs, coordinator.clone(), lease, &committed).await?;
    let schema = collection(dimensions as u32);
    let name = schema.name.clone();
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(schema))])
        .await?;

    let mut seed = 1u64;
    let mut unit = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let vector_mib = |rows: usize| rows * dimensions * 4 / (1024 * 1024);
    let report = |step: &str, rows: usize, writer: &Writer, took: Duration| {
        let resident = writer.partition().read(State::resident) / (1024 * 1024);
        println!(
            "{step:<28} rows {rows:>8}  vectors {:>5} MiB  State::resident {resident:>5} MiB  \
             anonymous {}  ({took:.1?})",
            vector_mib(rows),
            anonymous_mib(),
        );
    };
    println!(
        "{} store, cache {}",
        if held { "no-copies" } else { "cached" },
        cache.display()
    );
    let mut written = 0;
    for half in ["first", "second"] {
        let started = Instant::now();
        for chunk in (0..objects / 2).collect::<Vec<_>>().chunks(1_000) {
            let batch = chunk
                .iter()
                .map(|n| Mutation::Upsert {
                    collection: name.clone(),
                    id: ObjectId::generate(),
                    properties: serde_json::json!({ "line": *n as i64 })
                        .as_object()
                        .cloned()
                        .unwrap_or_default(),
                    vector: Some((0..dimensions).map(|_| unit()).collect()),
                    only_if_version: None,
                })
                .collect();
            writer.commit(batch).await?;
        }
        written += objects / 2;
        report(
            &format!("{half} half committed"),
            written,
            &writer,
            started.elapsed(),
        );
        let started = Instant::now();
        writer.flush().await?;
        report(
            &format!("{half} half flushed"),
            written,
            &writer,
            started.elapsed(),
        );
    }
    let started = Instant::now();
    while writer.merge().await?.is_some() {}
    report("merged", written, &writer, started.elapsed());
    Ok(())
}
