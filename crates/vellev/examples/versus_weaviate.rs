//! Load the same objects into a running vellev and a running Weaviate, and
//! measure both answering the same nearest-vector searches over HTTP,
//! returning whole objects, against the exact answer.
//!
//!   VELLEV_URL=http://127.0.0.1:8080 VELLEV_KEY=dev-key \
//!   WEAVIATE_URL=http://127.0.0.1:58080 \
//!   cargo run --release -p vellev --example versus_weaviate
//!
//! Environment: OBJECTS (1000000), DIMS (768), GROUPS (1000), QUERIES (200),
//! SETTLE (seconds to wait after loading, for vellev to flush and merge and
//! Weaviate to finish indexing; 120), LOAD_WEAVIATE (0 keeps what a run
//! before loaded into Weaviate; 1). Recall is swept on each side: vellev's
//! oversample, and Weaviate's HNSW `ef`.

use std::time::{Duration, Instant};

use serde_json::{json, Value};
use vellev_api::{Content, Nearest};
use vellev_client::{Client, Options};
use vellev_core::{Collection, ObjectId, TenantId};

const CLASS: &str = "VersusWeaviate";
const LIMIT: usize = 10;

fn env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// A small deterministic generator.
struct Random(u64);

impl Random {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn normalised(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|x| *x /= norm);
    vector
}

/// Vectors around `groups` centres, as embeddings of related text are.
fn vectors(objects: usize, dims: usize, groups: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut random = Random(seed);
    let centres: Vec<Vec<f32>> = (0..groups)
        .map(|_| (0..dims).map(|_| random.unit()).collect())
        .collect();
    (0..objects)
        .map(|n| {
            let centre = &centres[n % groups];
            normalised(centre.iter().map(|x| x + 0.6 * random.unit()).collect())
        })
        .collect()
}

fn id(n: usize) -> ObjectId {
    ObjectId(uuid::Uuid::from_u128(0x7e5_0000_0000 + n as u128))
}

/// The true nearest `LIMIT` of each query, by cosine, using every core.
fn exact(vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<ObjectId>> {
    std::thread::scope(|scope| {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        let chunk = queries.len().div_ceil(threads);
        let handles: Vec<_> = queries
            .chunks(chunk)
            .map(|queries| {
                scope.spawn(move || {
                    queries
                        .iter()
                        .map(|query| {
                            let mut scored: Vec<(f32, usize)> = vectors
                                .iter()
                                .enumerate()
                                .map(|(n, v)| (-vellev_index::vector::dot(query, v), n))
                                .collect();
                            scored.select_nth_unstable_by(LIMIT, |a, b| a.0.total_cmp(&b.0));
                            scored.truncate(LIMIT);
                            scored.into_iter().map(|(_, n)| id(n)).collect()
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    })
}

fn properties(n: usize, groups: usize) -> serde_json::Map<String, Value> {
    json!({
        "title": format!("object {n} in group {}", n % groups),
        "n": n as i64,
        "group": format!("g{}", n % groups),
    })
    .as_object()
    .unwrap()
    .clone()
}

/// p50 and p99 of `latencies`, and recall against `truth`.
fn summary(latencies: &mut [Duration], found: &[Vec<ObjectId>], truth: &[Vec<ObjectId>]) -> String {
    latencies.sort();
    let hits: usize = found
        .iter()
        .zip(truth)
        .map(|(found, truth)| found.iter().filter(|id| truth.contains(id)).count())
        .sum();
    format!(
        "recall@10 {:.3}, p50 {:.2?}, p99 {:.2?}",
        hits as f64 / (truth.len() * LIMIT) as f64,
        latencies[latencies.len() / 2],
        latencies[latencies.len() * 99 / 100],
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (objects, dims) = (env("OBJECTS", 1_000_000), env("DIMS", 768));
    let (groups, queries) = (env("GROUPS", 1_000), env("QUERIES", 200));
    let settle = Duration::from_secs(env("SETTLE", 120) as u64);
    let vellev_url = std::env::var("VELLEV_URL")?;
    let key = std::env::var("VELLEV_KEY")?;
    let weaviate_url = std::env::var("WEAVIATE_URL")?;
    println!("{objects} objects of {dims} dims in {groups} groups, {queries} queries");

    let data = vectors(objects, dims, groups, 7);
    let asked = vectors(queries, dims, groups, 7)
        .into_iter()
        .zip(vectors(queries, dims, 1, 99))
        .map(|(near, noise)| normalised(near.iter().zip(noise).map(|(a, b)| a + 0.3 * b).collect()))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let truth = exact(&data, &asked);
    println!("exact answers in {:.1?}", started.elapsed());

    // vellev
    let client = Client::connect(
        &[&vellev_url],
        &key,
        Options::default().timeout(Duration::from_secs(120)),
    )?;
    let tenant = client.tenant(&TenantId::new("versus")?);
    let _ = tenant.drop().await;
    tenant.create().await?;
    let schema: Collection = serde_json::from_value(json!({
        "name": "objects",
        "properties": [
            { "name": "title", "data_type": "text", "searchable": true },
            { "name": "n", "data_type": "int" },
            { "name": "group", "data_type": "text" },
        ],
        "embedding": { "provider": "none", "model": "synthetic", "dimensions": dims },
        "distance": "cosine",
    }))?;
    let records = tenant.collection(&schema.name);
    records.create(&schema).await?;
    let loading = Instant::now();
    records
        .store_all(data.iter().enumerate().map(|(n, vector)| {
            let content = Content::new(properties(n, groups)).vector(vector.clone());
            (id(n), content)
        }))
        .await?;
    println!("vellev: loaded in {:.1?}", loading.elapsed());

    // Weaviate, unless it holds the objects already from a run before.
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let weaviate = weaviate_url.trim_end_matches('/');
    if env("LOAD_WEAVIATE", 1) == 1 {
        let _ = http
            .delete(format!("{weaviate}/v1/schema/{CLASS}"))
            .send()
            .await;
        let class = json!({
            "class": CLASS,
            "vectorizer": "none",
            "vectorIndexConfig": { "distance": "cosine" },
            "properties": [
                { "name": "title", "dataType": ["text"] },
                { "name": "n", "dataType": ["int"] },
                { "name": "group", "dataType": ["text"], "tokenization": "field" },
            ],
        });
        http.post(format!("{weaviate}/v1/schema"))
            .json(&class)
            .send()
            .await?
            .error_for_status()?;
        let loading = Instant::now();
        for chunk in (0..objects).collect::<Vec<_>>().chunks(500) {
            let batch: Vec<Value> = chunk
            .iter()
            .map(|n| json!({ "class": CLASS, "id": id(*n), "properties": properties(*n, groups), "vector": data[*n] }))
            .collect();
            let answer: Value = http
                .post(format!("{weaviate}/v1/batch/objects"))
                .json(&json!({ "objects": batch }))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            if let Some(failed) = answer
                .as_array()
                .and_then(|all| all.iter().find(|o| o["result"]["errors"].is_object()))
            {
                return Err(format!("Weaviate refused an object: {failed}").into());
            }
        }
        println!("weaviate: loaded in {:.1?}", loading.elapsed());
    }

    println!("settling for {settle:?}");
    tokio::time::sleep(settle).await;

    for oversample in [32u32, 64, 128, 256] {
        let mut latencies = Vec::new();
        let mut found = Vec::new();
        for query in &asked {
            let started = Instant::now();
            let results = records
                .nearest(&Nearest::to_vector(query.clone(), LIMIT as u32).oversample(oversample))
                .await?;
            latencies.push(started.elapsed());
            found.push(results.results.iter().map(|r| r.object.id).collect());
        }
        println!(
            "  vellev oversample {oversample:>3}: {}",
            summary(&mut latencies, &found, &truth)
        );
    }

    for ef in [64i64, 128, 256, 512] {
        let mut class: Value = http
            .get(format!("{weaviate}/v1/schema/{CLASS}"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        class["vectorIndexConfig"]["ef"] = json!(ef);
        http.put(format!("{weaviate}/v1/schema/{CLASS}"))
            .json(&class)
            .send()
            .await?
            .error_for_status()?;
        let mut latencies = Vec::new();
        let mut found = Vec::new();
        for query in &asked {
            let vector: Vec<String> = query.iter().map(f32::to_string).collect();
            let graphql = format!(
                "{{ Get {{ {CLASS}(nearVector: {{ vector: [{}] }}, limit: {LIMIT}) {{ title n group _additional {{ id distance }} }} }} }}",
                vector.join(",")
            );
            let started = Instant::now();
            let answer: Value = http
                .post(format!("{weaviate}/v1/graphql"))
                .json(&json!({ "query": graphql }))
                .send()
                .await?
                .json()
                .await?;
            latencies.push(started.elapsed());
            let ids = answer["data"]["Get"][CLASS]
                .as_array()
                .ok_or_else(|| format!("Weaviate answered {answer}"))?
                .iter()
                .map(|found| serde_json::from_value(found["_additional"]["id"].clone()))
                .collect::<Result<Vec<ObjectId>, _>>()?;
            found.push(ids);
        }
        println!(
            "  weaviate ef {ef:>3}: {}",
            summary(&mut latencies, &found, &truth)
        );
    }
    Ok(())
}
