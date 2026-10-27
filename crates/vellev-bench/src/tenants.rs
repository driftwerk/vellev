//! Many tenants (design r2, §3e and §4): the SQuAD recall fixture as 200
//! small tenants and one of 50,000 rows, searched a seeded random tenant
//! per query.
//!
//! The fixture has 118,609 rows. The design's 200 tenants of about 600 rows
//! and one of 50,000 come to 170,000, so the tenants are cut from it
//! independently: small tenant `t` holds rows `t × 593` up to the next 593,
//! and the large tenant the first 50,000. A row may be in a small tenant and
//! in the large one; tenants share nothing, so that changes no measure.
//!
//! Weaviate is run as its own guidance says for small tenants: a
//! multi-tenant class with the `dynamic` index as the headline, and `flat`
//! beside it; default HNSW per tenant is reported, not as the headline.
//! The dynamic index needs Weaviate's asynchronous indexing, so this
//! Weaviate runs with `ASYNC_INDEXING=true`, which is stated.
//!
//! The cost of opening an idle tenant is not measured. Weaviate has no
//! equivalent of vellev letting a tenant go after it is idle (its tenants
//! go cold only when told), so, as the design says, that measure is not
//! published. vellev's idle let-go is still set to the design's 10 minutes.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use eyre::{ensure, eyre, Result, WrapErr};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::fixture::{id_of, uuid_of};
use crate::measure::percentile;
use crate::run::Repeated;

pub const SMALL_TENANTS: usize = 200;
pub const SMALL_ROWS: usize = 593;
pub const LARGE_ROWS: usize = 50_000;
pub const LARGE: &str = "large";
pub const CLASS_PREFIX: &str = "Memories";

/// A row of the recall fixture.
pub struct Memory {
    pub content: String,
    pub vector: Vec<f32>,
}

pub fn read_memories(path: &Path) -> Result<Vec<Memory>> {
    #[derive(Deserialize)]
    struct Line {
        content: String,
        vector: Vec<f32>,
    }
    let file = std::fs::File::open(path).wrap_err_with(|| path.display().to_string())?;
    BufReader::new(file)
        .lines()
        .map(|line| {
            let line: Line = serde_json::from_str(&line?)?;
            Ok(Memory {
                content: line.content,
                vector: line.vector,
            })
        })
        .collect()
}

pub fn read_queries(path: &Path) -> Result<Vec<Vec<f32>>> {
    #[derive(Deserialize)]
    struct Line {
        vector: Vec<f32>,
    }
    let file = std::fs::File::open(path).wrap_err_with(|| path.display().to_string())?;
    BufReader::new(file)
        .lines()
        .map(|line| Ok(serde_json::from_str::<Line>(&line?)?.vector))
        .collect()
}

/// The tenants: their names, and the rows of the fixture each holds.
pub fn tenants(rows: usize) -> Result<Vec<(String, std::ops::Range<usize>)>> {
    ensure!(
        rows >= (SMALL_TENANTS * SMALL_ROWS).max(LARGE_ROWS),
        "{rows} rows are too few for the tenants"
    );
    let mut tenants: Vec<(String, std::ops::Range<usize>)> = (0..SMALL_TENANTS)
        .map(|t| (format!("t{t:03}"), t * SMALL_ROWS..(t + 1) * SMALL_ROWS))
        .collect();
    tenants.push((LARGE.into(), 0..LARGE_ROWS));
    Ok(tenants)
}

/// The tenant each query asks, drawn with a fixed seed: one of the 201,
/// each as likely.
pub fn drawn(queries: usize, tenants: usize) -> Vec<usize> {
    // A small xorshift, so the draw is the same on every runner and release.
    let mut state: u64 = 0x5eed_7e4a_47c3_0001;
    (0..queries)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % tenants as u64) as usize
        })
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct Measured {
    pub index: String,
    pub headline: bool,
    pub load_seconds: f64,
    pub settle_seconds: f64,
    pub p50_ms: Repeated,
    pub p99_ms: Repeated,
    /// memory.stat anon once loaded, settled and searched.
    pub anon_bytes: u64,
}

/// One engine's many tenants: a way to make, load and search them.
#[async_trait::async_trait]
pub trait Tenanted: Send + Sync {
    async fn create(&self, tenants: &[String]) -> Result<()>;
    async fn load(&self, tenant: &str, first_id: usize, rows: &[Memory]) -> Result<()>;
    async fn settled(&self) -> Result<bool>;
    async fn nearest(&self, tenant: &str, vector: &[f32]) -> Result<Vec<u64>>;
}

/// Load every tenant, settle, then ask each query of its drawn tenant,
/// `reps` times.
pub async fn measure(
    engine: &dyn Tenanted,
    memories: &[Memory],
    queries: &[Vec<f32>],
    reps: usize,
    settle_within: Duration,
) -> Result<(f64, f64, Repeated, Repeated)> {
    let tenants = tenants(memories.len())?;
    let names: Vec<String> = tenants.iter().map(|(name, _)| name.clone()).collect();
    engine.create(&names).await?;
    let started = Instant::now();
    for (name, rows) in &tenants {
        for start in rows.clone().step_by(500) {
            let end = (start + 500).min(rows.end);
            engine.load(name, start, &memories[start..end]).await?;
        }
    }
    let load_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    while !engine.settled().await? {
        ensure!(
            started.elapsed() < settle_within,
            "not settled within {settle_within:?}"
        );
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let settle_seconds = started.elapsed().as_secs_f64();
    let asked = drawn(queries.len(), tenants.len());
    let (mut p50s, mut p99s) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let mut took = Vec::with_capacity(queries.len());
        for (vector, &tenant) in queries.iter().zip(&asked) {
            let started = Instant::now();
            let found = engine.nearest(&tenants[tenant].0, vector).await?;
            took.push(started.elapsed().as_secs_f64() * 1e3);
            ensure!(
                found.len() == 10,
                "tenant {} answered {} rows",
                tenants[tenant].0,
                found.len()
            );
        }
        p50s.push(percentile(&took, 50.0));
        p99s.push(percentile(&took, 99.0));
    }
    Ok((
        load_seconds,
        settle_seconds,
        Repeated::of(&p50s),
        Repeated::of(&p99s),
    ))
}

pub struct VellevTenants {
    pub url: String,
    pub key: String,
    pub metrics: String,
    pub http: reqwest::Client,
}

impl VellevTenants {
    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value> {
        let mut request = self
            .http
            .request(method, format!("{}{path}", self.url))
            .bearer_auth(&self.key);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        ensure!(
            status.is_success(),
            "{path}: {status}: {}",
            text.chars().take(500).collect::<String>()
        );
        Ok(if text.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text)?
        })
    }
}

#[async_trait::async_trait]
impl Tenanted for VellevTenants {
    async fn create(&self, tenants: &[String]) -> Result<()> {
        for tenant in tenants {
            self.call(reqwest::Method::PUT, &format!("/v1/tenants/{tenant}"), None)
                .await?;
            let schema = json!({
                "name": CLASS_PREFIX,
                "properties": [{"name": "content", "data_type": "text", "searchable": true, "filterable": false}],
                "embedding": {"provider": "none", "model": "client", "dimensions": crate::fixture::DIMENSIONS},
                "distance": "cosine",
            });
            let path = format!("/v1/tenants/{tenant}/collections/{CLASS_PREFIX}");
            self.call(reqwest::Method::PUT, &path, Some(&schema))
                .await?;
        }
        Ok(())
    }

    async fn load(&self, tenant: &str, first_id: usize, rows: &[Memory]) -> Result<()> {
        let changes: Vec<Value> = rows
            .iter()
            .enumerate()
            .map(|(n, row)| {
                json!({
                    "op": "upsert", "collection": CLASS_PREFIX, "id": uuid_of((first_id + n) as u64),
                    "properties": {"content": row.content}, "vector": row.vector,
                })
            })
            .collect();
        self.call(
            reqwest::Method::POST,
            &format!("/v1/tenants/{tenant}/changes"),
            Some(&json!({ "changes": changes })),
        )
        .await?;
        Ok(())
    }

    async fn settled(&self) -> Result<bool> {
        let text = self
            .http
            .get(format!("{}/metrics", self.metrics))
            .send()
            .await?
            .text()
            .await?;
        let unsettled = crate::engine::Vellev::sample(&text, "vellev_tenants_unsettled")
            .ok_or_else(|| eyre!("the node reports no vellev_tenants_unsettled"))?;
        Ok(unsettled == 0.0)
    }

    async fn nearest(&self, tenant: &str, vector: &[f32]) -> Result<Vec<u64>> {
        let path = format!("/v1/tenants/{tenant}/collections/{CLASS_PREFIX}/nearest");
        let answer = self
            .call(
                reqwest::Method::POST,
                &path,
                Some(&json!({ "vector": vector, "limit": 10 })),
            )
            .await?;
        answer["results"]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| id_of(result["id"].as_str().unwrap_or_default()))
            .collect()
    }
}

/// Weaviate's multi-tenant class with one vector index type.
pub struct WeaviateTenants {
    pub url: String,
    pub http: reqwest::Client,
    /// `dynamic`, `flat` or `hnsw`.
    pub index: String,
}

impl WeaviateTenants {
    pub fn class(&self) -> String {
        format!(
            "{CLASS_PREFIX}{}",
            self.index[..1].to_uppercase() + &self.index[1..]
        )
    }

    async fn call(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value> {
        let mut request = self.http.request(method, format!("{}{path}", self.url));
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        ensure!(
            status.is_success(),
            "{path}: {status}: {}",
            text.chars().take(500).collect::<String>()
        );
        Ok(if text.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text)?
        })
    }
}

#[async_trait::async_trait]
impl Tenanted for WeaviateTenants {
    async fn create(&self, tenants: &[String]) -> Result<()> {
        let class = json!({
            "class": self.class(),
            "vectorizer": "none",
            "vectorIndexType": self.index,
            "vectorIndexConfig": {"distance": "cosine"},
            "multiTenancyConfig": {"enabled": true},
            "properties": [{"name": "content", "dataType": ["text"], "tokenization": "word", "indexSearchable": true, "indexFilterable": false}],
        });
        self.call(reqwest::Method::POST, "/v1/schema", Some(&class))
            .await?;
        let named: Vec<Value> = tenants.iter().map(|name| json!({ "name": name })).collect();
        for chunk in named.chunks(100) {
            let path = format!("/v1/schema/{}/tenants", self.class());
            self.call(reqwest::Method::POST, &path, Some(&json!(chunk)))
                .await?;
        }
        Ok(())
    }

    async fn load(&self, tenant: &str, first_id: usize, rows: &[Memory]) -> Result<()> {
        let objects: Vec<Value> = rows
            .iter()
            .enumerate()
            .map(|(n, row)| {
                json!({
                    "class": self.class(), "tenant": tenant, "id": uuid_of((first_id + n) as u64),
                    "properties": {"content": row.content}, "vector": row.vector,
                })
            })
            .collect();
        let answer = self
            .call(
                reqwest::Method::POST,
                "/v1/batch/objects",
                Some(&json!({ "objects": objects })),
            )
            .await?;
        let failed = answer
            .as_array()
            .ok_or_else(|| eyre!("batch: {answer}"))?
            .iter()
            .filter(|object| !object["result"]["errors"].is_null())
            .count();
        ensure!(failed == 0, "{failed} objects refused: {answer}");
        Ok(())
    }

    async fn settled(&self) -> Result<bool> {
        let nodes = self
            .call(reqwest::Method::GET, "/v1/nodes?output=verbose", None)
            .await?;
        let class = self.class();
        let mut seen = 0;
        for shard in nodes["nodes"]
            .as_array()
            .into_iter()
            .flatten()
            .flat_map(|node| node["shards"].as_array().into_iter().flatten())
        {
            if shard["class"] != json!(class) {
                continue;
            }
            seen += 1;
            let queued = shard["vectorQueueLength"]
                .as_u64()
                .ok_or_else(|| eyre!("a shard with no vectorQueueLength: {shard}"))?;
            let status = shard["vectorIndexingStatus"]
                .as_str()
                .ok_or_else(|| eyre!("a shard with no vectorIndexingStatus: {shard}"))?;
            if queued > 0 || status != "READY" {
                return Ok(false);
            }
        }
        ensure!(seen > 0, "no shard of {class}");
        Ok(true)
    }

    async fn nearest(&self, tenant: &str, vector: &[f32]) -> Result<Vec<u64>> {
        let vector: Vec<String> = vector.iter().map(f32::to_string).collect();
        let class = self.class();
        let query = format!(
            "{{ Get {{ {class}(nearVector: {{vector: [{}]}}, limit: 10, tenant: \"{tenant}\") {{ content _additional {{ id }} }} }} }}",
            vector.join(",")
        );
        let answer = self
            .call(
                reqwest::Method::POST,
                "/v1/graphql",
                Some(&json!({ "query": query })),
            )
            .await?;
        ensure!(
            answer.get("errors").is_none(),
            "GraphQL: {}",
            answer["errors"]
        );
        answer["data"]["Get"][&class]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| id_of(result["_additional"]["id"].as_str().unwrap_or_default()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_hundred_small_tenants_of_593_rows_and_one_of_50000_fit_the_fixture() {
        let cut = tenants(118_609).unwrap();
        assert_eq!(cut.len(), 201);
        assert!(cut[..200].iter().all(|(_, rows)| rows.len() == 593));
        assert_eq!(cut[199].1.end, 118_600);
        assert_eq!(cut[200], (LARGE.to_string(), 0..50_000));
        assert!(tenants(1_000).is_err());
    }

    #[test]
    fn the_draw_is_fixed_and_reaches_every_tenant() {
        let first = drawn(10_000, 201);
        assert_eq!(first, drawn(10_000, 201));
        let mut seen = vec![false; 201];
        first.iter().for_each(|&t| seen[t] = true);
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn each_index_type_has_a_class_of_its_own() {
        let class = |index: &str| {
            WeaviateTenants {
                url: String::new(),
                http: reqwest::Client::new(),
                index: index.into(),
            }
            .class()
        };
        assert_eq!(class("dynamic"), "MemoriesDynamic");
        assert_eq!(class("flat"), "MemoriesFlat");
        assert_eq!(class("hnsw"), "MemoriesHnsw");
    }
}
