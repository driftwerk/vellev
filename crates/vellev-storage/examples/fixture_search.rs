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
    let meta: Vec<(i64, i64, bool, String)> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/passages.jsonl"))?)
            .lines()
            .take(rows)
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
                (
                    v["bucket"].as_i64().unwrap(),
                    v["owner"].as_i64().unwrap(),
                    v["deleted"].as_bool().unwrap(),
                    v["title"].as_str().unwrap().to_string(),
                )
            })
            .collect();
    let qall = shards(&dir.join("vectors/queries"));
    let queries: Vec<&[f32]> = qall.chunks_exact(d).collect();
    let mut owners = std::collections::HashMap::new();
    for m in &meta {
        owners.entry(m.3.clone()).or_insert(m.1);
    }
    let query_owner: Vec<i64> =
        std::io::BufReader::new(std::fs::File::open(dir.join("text/queries.jsonl"))?)
            .lines()
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(&l.unwrap()).unwrap();
                owners[v["title"].as_str().unwrap()]
            })
            .collect();
    // query, case -> true ids (top 10)
    let mut truth = std::collections::HashMap::new();
    if rows == 1_000_000 {
        for l in
            std::io::BufReader::new(std::fs::File::open(dir.join("truth/truth.jsonl"))?).lines()
        {
            let v: serde_json::Value = serde_json::from_str(&l?)?;
            let ids: Vec<u64> = v["ids"]
                .as_array()
                .unwrap()
                .iter()
                .take(10)
                .map(|i| i.as_u64().unwrap())
                .collect();
            truth.insert(
                (
                    v["query"].as_u64().unwrap() as usize,
                    v["case"].as_str().unwrap().to_string(),
                ),
                ids,
            );
        }
    }

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
                Property::new("bucket", DataType::Int),
                Property::new("owner", DataType::Int),
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
                properties:
                    json!({ "bucket": meta[n].0, "owner": meta[n].1, "deleted": meta[n].2 })
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
    let case_filter = |case: &str, q: usize| match case {
        "broad" => Filter::And(vec![not_deleted.clone()]),
        "mid" | "narrow" => Filter::And(vec![
            not_deleted.clone(),
            Filter::Lt {
                property: "bucket".into(),
                value: json!(if case == "mid" { 10 } else { 1 }),
            },
        ]),
        _ => Filter::And(vec![
            not_deleted.clone(),
            Filter::Eq {
                property: "owner".into(),
                value: json!(query_owner[q]),
            },
        ]),
    };
    let only: Vec<String> = std::env::var("SCRATCH_CASES")
        .unwrap_or("broad,mid,narrow,owner".into())
        .split(',')
        .map(String::from)
        .collect();
    for case in &only {
        let mut took = Vec::new();
        let (mut found, mut asked) = (0usize, 0usize);
        let mut allowed_owner = 0usize;
        for (q, vector) in queries.iter().enumerate() {
            let filter = case_filter(case, q);
            let asked_query = Query::nearest(vector, 10).filtered(&filter);
            let at = Instant::now();
            let ids: Vec<u64> = writer.partition().read(|state| {
                state
                    .search(&name, &asked_query)
                    .map(|h| h.iter().map(|h| h.object.id.0.as_u128() as u64).collect())
            })?;
            took.push(at.elapsed());
            if case == "owner" && q < 1000 {
                allowed_owner += meta
                    .iter()
                    .filter(|m| !m.2 && m.1 == query_owner[q])
                    .count();
            }
            if let Some(want) = truth.get(&(q, case.clone())) {
                asked += want.len();
                found += want.iter().filter(|id| ids.contains(id)).count();
            }
        }
        took.sort();
        if case == "owner" {
            println!(
                "owner: {} allowed rows a query on average",
                allowed_owner / queries.len()
            );
        }
        println!(
            "{case}: recall {:.4} p50 {:.2?} p99 {:.2?}",
            found as f64 / asked.max(1) as f64,
            took[took.len() / 2],
            took[took.len() * 99 / 100]
        );
    }
    if let Ok(cases) = std::env::var("SCRATCH_LOOP") {
        for case in cases.split(',') {
            eprintln!("LOOP {case} start, pid {}", std::process::id());
            let started = Instant::now();
            while started.elapsed() < Duration::from_secs(25) {
                for (q, vector) in queries.iter().enumerate() {
                    let filter = case_filter(case, q);
                    let asked_query = Query::nearest(vector, 10).filtered(&filter);
                    writer
                        .partition()
                        .read(|state| state.search(&name, &asked_query).map(|h| h.len()))?;
                }
            }
            eprintln!("LOOP {case} end");
        }
    }
    Ok(())
}
