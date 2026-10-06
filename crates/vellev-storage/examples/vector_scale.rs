//! Measure vector search through the whole storage path at scale: commit,
//! flush, merge into one clustered segment, then query it, with and
//! without a filter, against an exact answer.
//!
//!   cargo run --release -p vellev-storage --example vector_scale
//!
//! Environment: OBJECTS (1000000), DIMS (768), SEGMENTS (4), QUERIES (50),
//! GROUPS (1000).

use std::sync::Arc;
use std::time::{Duration, Instant};

use object_store::memory::InMemory;
use serde_json::json;
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, NodeId,
    ObjectId, Partition, Property, TenantId,
};
use vellev_index::filter::Filter;
use vellev_index::vector::distance;
use vellev_storage::blobs::ObjectBlobs;
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Mutation, Query, Writer};

fn setting(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

struct Noise(u64);

impl Noise {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn collection(dimensions: u32) -> Collection {
    Collection {
        name: CollectionName::new("Entities").expect("valid name"),
        properties: vec![
            Property::new("content", DataType::Text).searchable(),
            Property::new("line", DataType::Int),
        ],
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

fn resident_mb() -> u64 {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    output
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok())
        .map_or(0, |kb| kb / 1024)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let objects = setting("OBJECTS", 1_000_000) as usize;
    let dimensions = setting("DIMS", 768) as usize;
    let segments = setting("SEGMENTS", 4) as usize;
    let queries = setting("QUERIES", 50) as usize;
    let groups = setting("GROUPS", 1_000) as usize;

    let mut noise = Noise(1);
    let centres: Vec<Vec<f32>> = (0..groups)
        .map(|_| (0..dimensions).map(|_| noise.unit()).collect())
        .collect();
    let vector = |n: usize, noise: &mut Noise| -> Vec<f32> {
        centres[n % groups]
            .iter()
            .map(|c| c + 0.6 * noise.unit())
            .collect()
    };
    let ids: Vec<ObjectId> = (0..objects).map(|_| ObjectId::generate()).collect();

    let key = PartitionKey {
        tenant: TenantId::new("bench")?,
        partition: Partition::DEFAULT,
    };
    let coordinator = Arc::new(MemoryCoordinator::default());
    coordinator.register(&key).await?;
    let blobs = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
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
    println!("{objects} objects of {dimensions} dims in {groups} groups, {segments} segments");

    let started = Instant::now();
    let mut data = Noise(2);
    for (segment, chunk) in ids.chunks(objects.div_ceil(segments)).enumerate() {
        let first = segment * objects.div_ceil(segments);
        for (batch, batch_ids) in chunk.chunks(1_000).enumerate() {
            let start = first + batch * 1_000;
            let mutations = batch_ids
                .iter()
                .enumerate()
                .map(|(i, id)| Mutation::Upsert {
                    collection: name.clone(),
                    id: *id,
                    properties: json!({
                        "content": format!("entity {} in group {}", start + i, (start + i) % groups),
                        "line": (start + i) as i64,
                    })
                    .as_object()
                    .expect("an object")
                    .clone(),
                    vector: Some(vector(start + i, &mut data)),
                })
                .collect();
            writer.commit(mutations).await?;
        }
        let flushing = Instant::now();
        writer.flush().await?;
        println!(
            "segment {segment}: committed and flushed by {:.1?} (flush {:.1?}), rss {} MB",
            started.elapsed(),
            flushing.elapsed(),
            resident_mb()
        );
    }
    let merging = Instant::now();
    let position = writer.merge().await?;
    println!(
        "merge: {:?} in {:.1?}, rss {} MB",
        position,
        merging.elapsed(),
        resident_mb()
    );
    let shape: Vec<(u64, bool)> = writer.partition().read(|state| {
        state
            .segments()
            .map(|segment| (segment.rows, segment.index.is_some()))
            .collect()
    });
    println!("segments (rows, clustered): {shape:?}");

    let mut probe = Noise(3);
    let probes: Vec<Vec<f32>> = (0..queries)
        .map(|q| vector(q * (objects / queries), &mut probe))
        .collect();
    let exact: Vec<Vec<ObjectId>> = {
        let mut regen = Noise(2);
        let all: Vec<Vec<f32>> = (0..objects).map(|n| vector(n, &mut regen)).collect();
        probes
            .iter()
            .map(|query| {
                let mut scored: Vec<(f32, usize)> = all
                    .iter()
                    .enumerate()
                    .map(|(n, v)| (distance(Distance::Cosine, query, v), n))
                    .collect();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0));
                scored[..10].iter().map(|(_, n)| ids[*n]).collect()
            })
            .collect()
    };
    let half = Filter::Lt {
        property: "line".into(),
        value: json!(objects / 2),
    };
    let exact_half: Vec<Vec<ObjectId>> = {
        let mut regen = Noise(2);
        let all: Vec<Vec<f32>> = (0..objects / 2).map(|n| vector(n, &mut regen)).collect();
        probes
            .iter()
            .map(|query| {
                let mut scored: Vec<(f32, usize)> = all
                    .iter()
                    .enumerate()
                    .map(|(n, v)| (distance(Distance::Cosine, query, v), n))
                    .collect();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0));
                scored[..10].iter().map(|(_, n)| ids[*n]).collect()
            })
            .collect()
    };

    for (label, filter, truth) in [
        ("unfiltered", None, &exact),
        ("filtered to half", Some(&half), &exact_half),
    ] {
        for oversample in [16usize, 64] {
            let mut latencies = Vec::with_capacity(queries);
            let mut found = 0;
            for (query, nearest) in probes.iter().zip(truth) {
                let mut asked = Query::nearest(query, 10);
                asked.oversample = oversample;
                if let Some(filter) = filter {
                    asked = asked.filtered(filter);
                }
                let started = Instant::now();
                let hits = writer.partition().read(|state| {
                    state
                        .search(&name, &asked)
                        .map(|hits| hits.iter().map(|hit| hit.object.id).collect::<Vec<_>>())
                })?;
                latencies.push(started.elapsed());
                found += hits.iter().filter(|id| nearest.contains(id)).count();
            }
            latencies.sort();
            println!(
                "{label:>16} oversample {oversample:>2}: recall@10 {:.3}, p50 {:.2?}, p99 {:.2?}",
                found as f64 / (queries * 10) as f64,
                latencies[queries / 2],
                latencies[queries * 99 / 100],
            );
        }
    }
    println!("rss {} MB", resident_mb());
    Ok(())
}
