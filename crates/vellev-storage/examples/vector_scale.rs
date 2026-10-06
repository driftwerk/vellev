//! Measure vector search through the whole storage path at scale: commit,
//! flush, merge into one clustered segment, then query it, with and
//! without a filter, against an exact answer.
//!
//!   cargo run --release -p vellev-storage --example vector_scale
//!
//! Environment: OBJECTS (1000000), DIMS (768), SEGMENTS (4), QUERIES (50),
//! GROUPS (1000). The store is kept in memory with a local copy on disk,
//! so vectors are read in place as a node would read them, or on disk
//! alone with STORE=file, which is what a run too large for memory needs.

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
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::{Acquired, Coordinator, MemoryCoordinator, PartitionKey};
use vellev_storage::{Follower, Mutation, Query, Writer};

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
    // Object `n`'s vector, the same however often it is asked for; `salt`
    // tells a query apart from the object it is near.
    let vector = |n: usize, salt: u64| -> Vec<f32> {
        let mut noise = Noise((n as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt);
        (0..3).for_each(|_| {
            noise.unit();
        });
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
    let dir = tempfile::tempdir()?;
    let memory = Arc::new(ObjectBlobs::new(Arc::new(InMemory::new())));
    let blobs: Arc<dyn Blobs> = match std::env::var("STORE").as_deref() {
        Ok("file") => {
            let local = object_store::local::LocalFileSystem::new_with_prefix(dir.path())?;
            Arc::new(ObjectBlobs::on_disk(Arc::new(local), dir.path()))
        }
        _ => Arc::new(CachedBlobs::new(memory.clone(), dir.path())),
    };
    let Acquired::Lease { lease, committed } = coordinator
        .acquire(&key, &NodeId::generate("bench"), Duration::from_secs(3600))
        .await?
    else {
        return Err("the lease is held".into());
    };
    let writer = Arc::new(Writer::open(blobs, coordinator.clone(), lease, &committed).await?);
    let schema = collection(dimensions as u32);
    let name = schema.name.clone();
    let aside = Collection {
        name: CollectionName::new("WrittenDuringFlushes").expect("valid name"),
        ..schema.clone()
    };
    let during_flushes = aside.name.clone();
    writer
        .commit(vec![
            Mutation::CreateCollection(Box::new(schema)),
            Mutation::CreateCollection(Box::new(aside)),
        ])
        .await?;
    println!("{objects} objects of {dimensions} dims in {groups} groups, {segments} segments");

    let started = Instant::now();
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
                    vector: Some(vector(start + i, 0)),
                })
                .collect();
            writer.commit(mutations).await?;
        }
        // Commit one object at a time while the flush runs, to see how long
        // a commit waits for it.
        let flushing = Instant::now();
        let flush = tokio::spawn({
            let writer = writer.clone();
            async move { writer.flush().await }
        });
        let mut waited = Duration::ZERO;
        let mut during = 0;
        while !flush.is_finished() {
            let n = objects + segment * 1_000_000 + during;
            let commit = Instant::now();
            writer
                .commit(vec![Mutation::Upsert {
                    collection: during_flushes.clone(),
                    id: ObjectId::generate(),
                    properties: json!({ "content": "during", "line": n as i64 })
                        .as_object()
                        .expect("an object")
                        .clone(),
                    vector: Some(vector(n, 7)),
                }])
                .await?;
            waited = waited.max(commit.elapsed());
            during += 1;
        }
        flush.await??;
        writer.clean(Duration::ZERO).await?;
        println!(
            "segment {segment}: committed and flushed by {:.1?} (flush {:.1?}, {during} commits during it, the longest {waited:.1?}), rss {} MB",
            started.elapsed(),
            flushing.elapsed(),
            resident_mb()
        );
    }
    let merging = Instant::now();
    let mut merges = 0;
    while writer.merge().await?.is_some() {
        merges += 1;
        writer.clean(Duration::ZERO).await?;
    }
    println!(
        "{merges} merges in {:.1?}, rss {} MB",
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

    let probes: Vec<Vec<f32>> = (0..queries)
        .map(|q| vector(q * (objects / queries), 3))
        .collect();
    // The true nearest ten of every query among the first `count` objects,
    // found in one pass over the vectors made again, a slice per thread.
    let exact = |count: usize| -> Vec<Vec<ObjectId>> {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let slice = count.div_ceil(threads);
        let mut best: Vec<Vec<(f32, usize)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|t| {
                    let probes = &probes;
                    let vector = &vector;
                    scope.spawn(move || {
                        let (from, to) = (t * slice, ((t + 1) * slice).min(count));
                        let mut best = vec![Vec::<(f32, usize)>::new(); probes.len()];
                        for n in from..to {
                            let v = vector(n, 0);
                            for (q, query) in probes.iter().enumerate() {
                                let d = distance(Distance::Cosine, query, &v);
                                let top = &mut best[q];
                                if top.len() < 10 || d < top[9].0 {
                                    top.push((d, n));
                                    top.sort_by(|a, b| a.0.total_cmp(&b.0));
                                    top.truncate(10);
                                }
                            }
                        }
                        best
                    })
                })
                .collect();
            let mut merged = vec![Vec::<(f32, usize)>::new(); probes.len()];
            for worker in workers {
                for (q, top) in worker.join().expect("a worker").into_iter().enumerate() {
                    merged[q].extend(top);
                }
            }
            merged
        });
        best.iter_mut()
            .map(|top| {
                top.sort_by(|a, b| a.0.total_cmp(&b.0));
                top[..10].iter().map(|(_, n)| ids[*n]).collect()
            })
            .collect()
    };
    let exacting = Instant::now();
    let exact_all = exact(objects);
    let half = Filter::Lt {
        property: "line".into(),
        value: json!(objects / 2),
    };
    let exact_half = exact(objects / 2);
    println!("exact answers in {:.1?}", exacting.elapsed());

    for (label, filter, truth) in [
        ("unfiltered", None, &exact_all),
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

    let committed = coordinator.committed(&key).await?.ok_or("registered")?;
    let empty = tempfile::tempdir()?;
    for (label, cache) in [
        ("an empty cache", empty.path()),
        ("a warm cache", empty.path()),
    ] {
        let cached = Arc::new(CachedBlobs::new(memory.clone(), cache));
        let opening = Instant::now();
        let follower = Follower::open(key.clone(), cached, &committed).await?;
        let rows = follower.partition().read(|state| state.count(&name));
        println!(
            "opened {rows} objects from {label} in {:.1?}",
            opening.elapsed()
        );
    }
    Ok(())
}
