//! Scratch: the storage layer's search on the benchmark fixture, broad and
//! filtered, at the default oversample, for profiling.
//!
//!   fixture_search FIXTURE ROWS FLUSH_ROWS

use std::io::BufRead;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition, Property, TenantId,
};
use vellev_index::filter::Filter;
use vellev_storage::blobs::{Blobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Mutation, Query, Writer};

fn shards(dir: &std::path::Path) -> Vec<f32> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.retain(|p| {
        p.extension().is_some_and(|e| e == "f32")
            && !p.file_name().unwrap().to_string_lossy().starts_with("._")
    });
    paths.sort();
    paths
        .iter()
        .flat_map(|p| {
            let bytes = std::fs::read(p).unwrap();
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().unwrap());
    let rows: usize = args.next().unwrap().parse()?;
    let flush: usize = args.next().unwrap().parse()?;
    let d = 768;
    let all = shards(&dir.join("vectors/passages"));
    let meta: Vec<(i64, bool)> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/passages.jsonl"))?)
            .lines()
            .take(rows)
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
                (
                    v["bucket"].as_i64().unwrap(),
                    v["deleted"].as_bool().unwrap(),
                )
            })
            .collect();
    let qall = shards(&dir.join("vectors/queries"));
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();

    let key = PartitionKey {
        tenant: TenantId::new("bench")?,
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await?;
    let tmp = tempfile::tempdir()?;
    let local = object_store::local::LocalFileSystem::new_with_prefix(tmp.path())?;
    let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::on_disk(Arc::new(local), tmp.path()));
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("bench"), Duration::from_secs(3600))
        .await?
    else {
        return Err("held".into());
    };
    let writer = Writer::open(blobs, coordinator.clone(), lease, &committed).await?;
    let name = CollectionName::new("Passages")?;
    writer
        .commit(vec![Mutation::CreateCollection(Box::new(Collection {
            name: name.clone(),
            properties: vec![
                Property::new("bucket", DataType::Int),
                Property::new("deleted", DataType::Bool),
            ],
            embedding: Embedding {
                provider: EmbeddingProvider::None,
                model: "supplied".into(),
                dimensions: d as u32,
                query_prefix: None,
                document_prefix: None,
                properties: vec![],
            },
            distance: Distance::Cosine,
            bm25: Bm25::default(),
            access_rule: None,
        }))])
        .await?;
    let started = Instant::now();
    for start in (0..rows).step_by(500) {
        let batch = (start..(start + 500).min(rows))
            .map(|n| Mutation::Upsert {
                collection: name.clone(),
                id: ObjectId(uuid::Uuid::from_u128(n as u128)),
                properties: json!({ "bucket": meta[n].0, "deleted": meta[n].1 })
                    .as_object()
                    .unwrap()
                    .clone(),
                vector: Some(all[n * d..(n + 1) * d].to_vec()),
                only_if_version: None,
            })
            .collect();
        writer.commit(batch).await?;
        if (start + 500) % flush == 0 {
            writer.flush().await?;
        }
    }
    writer.flush().await?;
    while writer.merge().await?.is_some() {}
    let shape: Vec<(u64, bool)> = writer
        .partition()
        .read(|s| s.segments().map(|m| (m.rows, m.index.is_some())).collect());
    println!(
        "{rows} rows loaded and merged in {:.1?}: segments {shape:?}",
        started.elapsed()
    );

    let not_deleted = Filter::Eq {
        property: "deleted".into(),
        value: json!(false),
    };
    let mid = Filter::And(vec![
        not_deleted.clone(),
        Filter::Lt {
            property: "bucket".into(),
            value: json!(10),
        },
    ]);
    for (label, filter) in [("broad", &not_deleted), ("mid", &mid)] {
        let mut took = Vec::new();
        for q in &queries {
            let asked = Query::nearest(q, 10).filtered(filter);
            let at = Instant::now();
            let hits = writer
                .partition()
                .read(|state| state.search(&name, &asked).map(|h| h.len()))?;
            took.push(at.elapsed());
            assert!(hits <= 10);
        }
        took.sort();
        println!(
            "{label}: p50 {:.2?} p99 {:.2?}",
            took[took.len() / 2],
            took[took.len() * 99 / 100]
        );
    }
    if std::env::var("SCRATCH_LOOP").is_ok() {
        eprintln!("looping mid for profiling");
        loop {
            for q in &queries {
                let asked = Query::nearest(q, 10).filtered(&mid);
                writer
                    .partition()
                    .read(|state| state.search(&name, &asked).map(|h| h.len()))?;
            }
        }
    }
    Ok(())
}
