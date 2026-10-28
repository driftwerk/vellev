//! Scratch: vellev's keyword leg of the hybrid search, broad, for every
//! measured question, written out for fusing offline.
//!
//!   keyword_leg FIXTURE OUT

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
use vellev_storage::{Keywords, Mutation, Writer};

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
    let out = args.next().unwrap();
    let rows = 1_000_000usize;
    let passages: Vec<(String, bool)> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/passages.jsonl"))?)
            .lines()
            .take(rows)
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
                (
                    v["text"].as_str().unwrap().to_string(),
                    v["deleted"].as_bool().unwrap(),
                )
            })
            .collect();
    let questions: Vec<String> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/queries.jsonl"))?)
            .lines()
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
                v["text"].as_str().unwrap().to_string()
            })
            .collect();
    let key = PartitionKey {
        tenant: TenantId::new("bench")?,
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await?;
    let tmp = tempfile::tempdir_in("/Volumes/T9/vellev-scratch/tmp")?;
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
                Property::new("text", DataType::Text).searchable(),
                Property::new("deleted", DataType::Bool),
            ],
            embedding: Embedding {
                provider: EmbeddingProvider::None,
                model: "supplied".into(),
                dimensions: 768,
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
                properties: json!({ "text": passages[n].0, "deleted": passages[n].1 })
                    .as_object()
                    .unwrap()
                    .clone(),
                vector: None,
                only_if_version: None,
            })
            .collect();
        writer.commit(batch).await?;
        if (start + 500) % 50_000 == 0 {
            writer.flush().await?;
        }
    }
    writer.flush().await?;
    while writer.merge().await?.is_some() {}
    println!("loaded in {:.1?}", started.elapsed());
    let not_deleted = Filter::And(vec![Filter::Eq {
        property: "deleted".into(),
        value: json!(false),
    }]);
    let mut lines = String::new();
    for (q, text) in questions.iter().enumerate() {
        let asked = Keywords {
            text,
            properties: &[],
            limit: 100,
            filter: Some(&not_deleted),
        };
        let found: Vec<(u64, f32)> = writer.partition().read(|state| {
            state.keyword(&name, &asked).map(|hits| {
                hits.iter()
                    .map(|h| (h.object.id.0.as_u128() as u64, h.score))
                    .collect()
            })
        })?;
        lines += &serde_json::to_string(&json!({"query": q, "found": found}))?;
        lines.push('\n');
    }
    std::fs::write(out, lines)?;
    Ok(())
}
