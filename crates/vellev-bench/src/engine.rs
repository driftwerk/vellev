//! The two engines as the harness drives them: over HTTP, each with a
//! minimal client of its own (design r2, §6). Both are asked for the same
//! thing and give back the same thing: every answer carries the object's
//! properties, as vellev's always do.

use std::time::Duration;

use async_trait::async_trait;
use eyre::{bail, ensure, eyre, Result, WrapErr};
use serde_json::{json, Value};

use crate::fixture::{id_of, uuid_of, Filter, Row, DIMENSIONS};

pub const COLLECTION: &str = "Passages";
/// A collection left empty, searched to measure what a query costs that
/// finds nothing to score: the client's and the engine's fixed cost.
pub const FLOOR: &str = "Floor";

/// A point on an engine's knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Knob {
    /// vellev with no `oversample` asked: what a caller gets by default.
    VellevDefault,
    /// vellev's `oversample`.
    Oversample(u32),
    /// Weaviate's dynamic `ef`, its default (`ef` = -1).
    WeaviateDynamic,
    /// Weaviate's `ef`, raised to the limit as Weaviate needs.
    Ef(u32),
}

impl Knob {
    pub fn label(&self) -> String {
        match self {
            Knob::VellevDefault => "default".into(),
            Knob::Oversample(n) => format!("oversample={n}"),
            Knob::WeaviateDynamic => "ef=dynamic (default)".into(),
            Knob::Ef(n) => format!("ef={n}"),
        }
    }
}

/// vellev's grid (§5), and its default.
pub const OVERSAMPLES: [u32; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128];
/// Weaviate's grid (§5), each raised to at least `limit`.
pub const EFS: [u32; 9] = [16, 32, 48, 64, 96, 128, 192, 256, 512];

#[async_trait]
pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    /// Make the collection the passages go into, and the empty one the
    /// fixed cost of a query is measured on.
    async fn create(&self) -> Result<()>;
    async fn load(&self, rows: &[Row]) -> Result<()>;
    /// Whether the engine says the writes are all in place: nothing queued
    /// to index, flush or merge.
    async fn settled(&self) -> Result<bool>;
    /// The knob points measured at `limit`, the engine's default first.
    fn points(&self, limit: usize) -> Vec<Knob>;
    /// Put the engine at `knob` for the searches that follow.
    async fn set(&self, knob: Knob) -> Result<()>;
    /// The ids of the nearest `limit` rows to `vector` that pass `filter`.
    async fn nearest(
        &self,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        knob: Knob,
    ) -> Result<Vec<u64>>;
    /// The ids of the best `limit` rows for `text` and `vector` together,
    /// fused at the engine's defaults (alpha 0.75, relative scores), that
    /// pass `filter`.
    async fn hybrid(
        &self,
        text: &str,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        knob: Knob,
    ) -> Result<Vec<u64>>;
    /// A search of the empty collection, as the measured searches are
    /// asked but with nothing to score.
    async fn floor(&self, vector: &[f32]) -> Result<()>;
    /// The engine's settings in effect, read back from it.
    async fn settings(&self) -> Result<Value>;
}

/// At most `n` bytes of `text`, cut at a character.
fn cut(text: &str, n: usize) -> &str {
    let mut end = n.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .pool_max_idle_per_host(256)
        .build()?)
}

/// A request the engine turned away as too busy (429 or 503), rather than
/// failed: counted in a burst, not taken as the run failing.
#[derive(Debug, thiserror::Error)]
#[error("{what}: refused as busy, {status}")]
pub struct Refused {
    pub what: String,
    pub status: reqwest::StatusCode,
}

async fn answered(response: reqwest::Response, what: &str) -> Result<Value> {
    let status = response.status();
    if status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::SERVICE_UNAVAILABLE
    {
        return Err(Refused {
            what: what.into(),
            status,
        }
        .into());
    }
    let text = response.text().await?;
    ensure!(
        status.is_success(),
        "{what}: {status}: {}",
        cut(&text, 2000)
    );
    if text.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).wrap_err_with(|| format!("{what}: {}", cut(&text, 500)))
}

pub struct Vellev {
    url: String,
    metrics: String,
    key: String,
    tenant: String,
    http: reqwest::Client,
}

impl Vellev {
    pub fn new(url: &str, metrics: &str, key: &str, tenant: &str) -> Result<Vellev> {
        Ok(Vellev {
            url: url.trim_end_matches('/').into(),
            metrics: metrics.trim_end_matches('/').into(),
            key: key.into(),
            tenant: tenant.into(),
            http: client()?,
        })
    }

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
        answered(request.send().await?, path).await
    }

    fn filter(filter: Filter) -> Value {
        let mut all = vec![json!({"eq": {"property": "deleted", "value": false}})];
        if let Some(below) = filter.bucket_below {
            all.push(json!({"lt": {"property": "bucket", "value": below}}));
        }
        if let Some(owner) = filter.owner {
            all.push(json!({"eq": {"property": "owner", "value": owner}}));
        }
        json!({ "and": all })
    }

    async fn create_named(&self, name: &str) -> Result<()> {
        let schema = json!({
            "name": name,
            "properties": [
                {"name": "text", "data_type": "text", "searchable": true, "filterable": false},
                {"name": "bucket", "data_type": "int"},
                {"name": "owner", "data_type": "int"},
                {"name": "deleted", "data_type": "bool"},
            ],
            "embedding": {"provider": "none", "model": "client", "dimensions": DIMENSIONS},
            "distance": "cosine",
        });
        let path = format!("/v1/tenants/{}/collections/{name}", self.tenant);
        self.call(reqwest::Method::PUT, &path, Some(&schema))
            .await?;
        Ok(())
    }

    async fn nearest_in(
        &self,
        name: &str,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        knob: Knob,
    ) -> Result<Vec<u64>> {
        let mut body = json!({
            "vector": vector,
            "limit": limit,
            "filter": Vellev::filter(filter),
        });
        match knob {
            Knob::VellevDefault => {}
            Knob::Oversample(n) => body["oversample"] = json!(n),
            other => bail!("vellev has no knob {other:?}"),
        }
        let path = format!("/v1/tenants/{}/collections/{name}/nearest", self.tenant);
        let answer = self.call(reqwest::Method::POST, &path, Some(&body)).await?;
        ensure!(
            answer.get("degraded").is_none(),
            "a degraded answer: {answer}"
        );
        answer["results"]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| id_of(result["id"].as_str().unwrap_or_default()))
            .collect()
    }

    /// The best `limit` rows for the words of `text` that pass `filter`,
    /// with their keyword scores: a hybrid search's keyword leg.
    pub async fn keyword(
        &self,
        text: &str,
        filter: Filter,
        limit: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let body = json!({ "text": text, "limit": limit, "filter": Vellev::filter(filter) });
        let path = format!(
            "/v1/tenants/{}/collections/{COLLECTION}/keyword",
            self.tenant
        );
        let answer = self.call(reqwest::Method::POST, &path, Some(&body)).await?;
        answer["results"]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| {
                let score = result["keyword"]
                    .as_f64()
                    .or(result["score"].as_f64())
                    .ok_or_else(|| eyre!("a keyword result with no score: {result}"))?;
                Ok((
                    id_of(result["id"].as_str().unwrap_or_default())?,
                    score as f32,
                ))
            })
            .collect()
    }

    /// The node's metrics, as Prometheus text.
    pub async fn metrics(&self) -> Result<String> {
        let response = self
            .http
            .get(format!("{}/metrics", self.metrics))
            .send()
            .await?;
        ensure!(
            response.status().is_success(),
            "metrics: {}",
            response.status()
        );
        Ok(response.text().await?)
    }

    /// Fail unless the node runs its Tokio workers and its Rayon pool on
    /// `cpus` threads each, as the run set them (§2).
    pub fn threads_as_set(metrics: &str, cpus: u32) -> Result<()> {
        for pool in ["tokio_workers", "rayon"] {
            let threads = Vellev::sample(metrics, &format!("vellev_threads{{pool=\"{pool}\"}}"));
            ensure!(
                threads == Some(f64::from(cpus)),
                "vellev's {pool} threads are {threads:?}, not the {cpus} set"
            );
        }
        Ok(())
    }

    /// A sample of the metrics, as `name{labels} value`.
    pub fn sample(text: &str, name: &str) -> Option<f64> {
        text.lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '))
            .and_then(|value| value.trim().parse().ok())
    }
}

#[async_trait]
impl Engine for Vellev {
    fn name(&self) -> &'static str {
        "vellev"
    }

    async fn create(&self) -> Result<()> {
        let tenant = format!("/v1/tenants/{}", self.tenant);
        self.call(reqwest::Method::PUT, &tenant, None).await?;
        for name in [COLLECTION, FLOOR] {
            self.create_named(name).await?;
        }
        Ok(())
    }

    async fn load(&self, rows: &[Row]) -> Result<()> {
        let changes: Vec<Value> = rows
            .iter()
            .map(|row| {
                json!({
                    "op": "upsert",
                    "collection": COLLECTION,
                    "id": uuid_of(row.id),
                    "properties": {"text": row.text, "bucket": row.bucket, "owner": row.owner, "deleted": row.deleted},
                    "vector": row.vector,
                })
            })
            .collect();
        let path = format!("/v1/tenants/{}/changes", self.tenant);
        // A write refused while flushes catch up says when to try again.
        loop {
            let response = self
                .http
                .post(format!("{}{path}", self.url))
                .bearer_auth(&self.key)
                .json(&json!({ "changes": changes }))
                .send()
                .await?;
            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS
                || response.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE
            {
                let wait = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
                    .unwrap_or(1);
                tokio::time::sleep(Duration::from_secs(wait.clamp(1, 30))).await;
                continue;
            }
            let written = answered(response, &path).await?;
            let refused = written["outcomes"]
                .as_array()
                .ok_or_else(|| eyre!("no outcomes: {written}"))?
                .iter()
                .filter(|outcome| outcome["outcome"] != "applied")
                .count();
            ensure!(refused == 0, "{refused} rows not applied: {written}");
            return Ok(());
        }
    }

    async fn settled(&self) -> Result<bool> {
        let text = self.metrics().await?;
        let unsettled = Vellev::sample(&text, "vellev_tenants_unsettled")
            .ok_or_else(|| eyre!("the node reports no vellev_tenants_unsettled"))?;
        Ok(unsettled == 0.0)
    }

    fn points(&self, _limit: usize) -> Vec<Knob> {
        std::iter::once(Knob::VellevDefault)
            .chain(OVERSAMPLES.iter().map(|&n| Knob::Oversample(n)))
            .collect()
    }

    async fn set(&self, _knob: Knob) -> Result<()> {
        Ok(())
    }

    async fn nearest(
        &self,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        knob: Knob,
    ) -> Result<Vec<u64>> {
        self.nearest_in(COLLECTION, vector, filter, limit, knob)
            .await
    }

    async fn hybrid(
        &self,
        text: &str,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        knob: Knob,
    ) -> Result<Vec<u64>> {
        let mut body = json!({
            "text": text,
            "vector": vector,
            "limit": limit,
            "filter": Vellev::filter(filter),
        });
        match knob {
            Knob::VellevDefault => {}
            Knob::Oversample(n) => body["oversample"] = json!(n),
            other => bail!("vellev has no knob {other:?}"),
        }
        let path = format!(
            "/v1/tenants/{}/collections/{COLLECTION}/hybrid",
            self.tenant
        );
        let answer = self.call(reqwest::Method::POST, &path, Some(&body)).await?;
        ensure!(
            answer.get("degraded").is_none(),
            "a degraded answer: {answer}"
        );
        answer["results"]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| id_of(result["id"].as_str().unwrap_or_default()))
            .collect()
    }

    async fn floor(&self, vector: &[f32]) -> Result<()> {
        let broad = Filter::of(crate::fixture::Case::Broad, 0);
        let found = self
            .nearest_in(FLOOR, vector, broad, 10, Knob::VellevDefault)
            .await?;
        ensure!(found.is_empty(), "the empty collection answered {found:?}");
        Ok(())
    }

    async fn settings(&self) -> Result<Value> {
        let path = format!("/v1/tenants/{}/collections/{COLLECTION}", self.tenant);
        let schema = self.call(reqwest::Method::GET, &path, None).await?;
        let limits = self.call(reqwest::Method::GET, "/v1/limits", None).await?;
        let metrics = self.metrics().await?;
        let threads =
            |pool: &str| Vellev::sample(&metrics, &format!("vellev_threads{{pool=\"{pool}\"}}"));
        Ok(json!({
            "schema": schema,
            "limits": limits,
            "threads": {
                "tokio_workers": threads("tokio_workers"),
                "rayon": threads("rayon"),
                "reads_at_once": Vellev::sample(&metrics, "vellev_reads_at_once"),
            },
            "default_oversample": "the node's own, no oversample asked",
        }))
    }
}

pub struct Weaviate {
    url: String,
    http: reqwest::Client,
    /// The class's vector index settings, as made.
    index: Value,
}

impl Weaviate {
    /// `index` is the class's `vectorIndexConfig` as made: `{}` for
    /// Weaviate's defaults, or the tuned build's.
    pub fn new(url: &str, index: Value) -> Result<Weaviate> {
        Ok(Weaviate {
            url: url.trim_end_matches('/').into(),
            http: client()?,
            index,
        })
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
        answered(request.send().await?, path).await
    }

    fn filter(filter: Filter) -> String {
        let mut all =
            vec![r#"{path: ["deleted"], operator: Equal, valueBoolean: false}"#.to_string()];
        if let Some(below) = filter.bucket_below {
            all.push(format!(
                r#"{{path: ["bucket"], operator: LessThan, valueInt: {below}}}"#
            ));
        }
        if let Some(owner) = filter.owner {
            all.push(format!(
                r#"{{path: ["owner"], operator: Equal, valueInt: {owner}}}"#
            ));
        }
        format!("{{operator: And, operands: [{}]}}", all.join(", "))
    }

    /// Whether every shard of the passages has nothing queued to index,
    /// as `/v1/nodes?output=verbose` says. A shard that does not say is an
    /// error, not settled: a renamed field must not pass as an empty queue.
    pub fn shards_settled(nodes: &Value) -> Result<bool> {
        let shards: Vec<&Value> = nodes["nodes"]
            .as_array()
            .ok_or_else(|| eyre!("nodes: {nodes}"))?
            .iter()
            .flat_map(|node| node["shards"].as_array().into_iter().flatten())
            .filter(|shard| shard["class"] == COLLECTION)
            .collect();
        ensure!(!shards.is_empty(), "no shard of {COLLECTION}: {nodes}");
        let mut settled = true;
        for shard in shards {
            let queued = shard["vectorQueueLength"]
                .as_u64()
                .ok_or_else(|| eyre!("a shard with no vectorQueueLength: {shard}"))?;
            let status = shard["vectorIndexingStatus"]
                .as_str()
                .ok_or_else(|| eyre!("a shard with no vectorIndexingStatus: {shard}"))?;
            settled &= queued == 0 && status == "READY";
        }
        Ok(settled)
    }

    async fn create_named(&self, name: &str) -> Result<()> {
        let mut index = self.index.clone();
        index["distance"] = json!("cosine");
        let class = json!({
            "class": name,
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": index,
            "properties": [
                {"name": "text", "dataType": ["text"], "tokenization": "word", "indexSearchable": true, "indexFilterable": false},
                {"name": "bucket", "dataType": ["int"], "indexFilterable": true},
                {"name": "owner", "dataType": ["int"], "indexFilterable": true},
                {"name": "deleted", "dataType": ["boolean"], "indexFilterable": true},
            ],
        });
        self.call(reqwest::Method::POST, "/v1/schema", Some(&class))
            .await?;
        Ok(())
    }

    async fn nearest_in(
        &self,
        name: &str,
        vector: &[f32],
        filter: Filter,
        limit: usize,
    ) -> Result<Vec<u64>> {
        let vector: Vec<String> = vector.iter().map(f32::to_string).collect();
        let query = format!(
            "{{ Get {{ {name}(nearVector: {{vector: [{}]}}, limit: {limit}, where: {}) {{ text bucket owner deleted _additional {{ id }} }} }} }}",
            vector.join(","),
            Weaviate::filter(filter),
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
        answer["data"]["Get"][name]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| id_of(result["_additional"]["id"].as_str().unwrap_or_default()))
            .collect()
    }
}

#[async_trait]
impl Engine for Weaviate {
    fn name(&self) -> &'static str {
        "weaviate"
    }

    async fn create(&self) -> Result<()> {
        for name in [COLLECTION, FLOOR] {
            self.create_named(name).await?;
        }
        Ok(())
    }

    async fn load(&self, rows: &[Row]) -> Result<()> {
        let objects: Vec<Value> = rows
            .iter()
            .map(|row| {
                json!({
                    "class": COLLECTION,
                    "id": uuid_of(row.id),
                    "properties": {"text": row.text, "bucket": row.bucket, "owner": row.owner, "deleted": row.deleted},
                    "vector": row.vector,
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
        let failed: Vec<&Value> = answer
            .as_array()
            .ok_or_else(|| eyre!("batch: {answer}"))?
            .iter()
            .filter(|object| !object["result"]["errors"].is_null())
            .collect();
        ensure!(
            failed.is_empty(),
            "{} objects refused: {}",
            failed.len(),
            failed[0]
        );
        Ok(())
    }

    async fn settled(&self) -> Result<bool> {
        let nodes = self
            .call(reqwest::Method::GET, "/v1/nodes?output=verbose", None)
            .await?;
        Weaviate::shards_settled(&nodes)
    }

    fn points(&self, limit: usize) -> Vec<Knob> {
        let mut points = vec![Knob::WeaviateDynamic];
        for ef in EFS {
            let raised = Knob::Ef(ef.max(limit as u32));
            if !points.contains(&raised) {
                points.push(raised);
            }
        }
        points
    }

    async fn set(&self, knob: Knob) -> Result<()> {
        let ef: i64 = match knob {
            Knob::WeaviateDynamic => -1,
            Knob::Ef(n) => i64::from(n),
            other => bail!("Weaviate has no knob {other:?}"),
        };
        let path = format!("/v1/schema/{COLLECTION}");
        let mut class = self.call(reqwest::Method::GET, &path, None).await?;
        class["vectorIndexConfig"]["ef"] = json!(ef);
        self.call(reqwest::Method::PUT, &path, Some(&class)).await?;
        let read = self.call(reqwest::Method::GET, &path, None).await?;
        ensure!(
            read["vectorIndexConfig"]["ef"] == json!(ef),
            "ef did not take: {read}"
        );
        Ok(())
    }

    async fn nearest(
        &self,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        _knob: Knob,
    ) -> Result<Vec<u64>> {
        self.nearest_in(COLLECTION, vector, filter, limit).await
    }

    async fn hybrid(
        &self,
        text: &str,
        vector: &[f32],
        filter: Filter,
        limit: usize,
        _knob: Knob,
    ) -> Result<Vec<u64>> {
        let vector: Vec<String> = vector.iter().map(f32::to_string).collect();
        let query = format!(
            "{{ Get {{ {COLLECTION}(hybrid: {{query: {}, vector: [{}], alpha: 0.75, fusionType: relativeScoreFusion}}, limit: {limit}, where: {}) {{ text bucket owner deleted _additional {{ id }} }} }} }}",
            serde_json::to_string(text)?,
            vector.join(","),
            Weaviate::filter(filter),
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
        answer["data"]["Get"][COLLECTION]
            .as_array()
            .ok_or_else(|| eyre!("no results: {answer}"))?
            .iter()
            .map(|result| id_of(result["_additional"]["id"].as_str().unwrap_or_default()))
            .collect()
    }

    async fn floor(&self, vector: &[f32]) -> Result<()> {
        let broad = Filter::of(crate::fixture::Case::Broad, 0);
        let found = self.nearest_in(FLOOR, vector, broad, 10).await?;
        ensure!(found.is_empty(), "the empty class answered {found:?}");
        Ok(())
    }

    async fn settings(&self) -> Result<Value> {
        let class = self
            .call(
                reqwest::Method::GET,
                &format!("/v1/schema/{COLLECTION}"),
                None,
            )
            .await?;
        let meta = self.call(reqwest::Method::GET, "/v1/meta", None).await?;
        Ok(json!({ "class": class, "meta": meta }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weaviate_ef_is_raised_to_the_limit_and_each_point_measured_once() {
        let weaviate = Weaviate::new("http://x", json!({})).unwrap();
        let at_100 = weaviate.points(100);
        assert_eq!(at_100[0], Knob::WeaviateDynamic);
        assert_eq!(
            at_100[1..],
            [
                Knob::Ef(100),
                Knob::Ef(128),
                Knob::Ef(192),
                Knob::Ef(256),
                Knob::Ef(512)
            ]
        );
        assert_eq!(weaviate.points(10).len(), 1 + EFS.len());
    }

    #[test]
    fn both_engines_filter_deleted_rows_in_every_case() {
        let filter = Filter {
            bucket_below: Some(10),
            owner: Some(3),
        };
        let vellev = Vellev::filter(filter).to_string();
        assert!(
            vellev.contains(r#"{"eq":{"property":"deleted","value":false}}"#),
            "{vellev}"
        );
        assert!(
            vellev.contains(r#"{"lt":{"property":"bucket","value":10}}"#),
            "{vellev}"
        );
        assert!(
            vellev.contains(r#"{"eq":{"property":"owner","value":3}}"#),
            "{vellev}"
        );
        let weaviate = Weaviate::filter(filter);
        assert!(weaviate.contains(r#"{path: ["deleted"], operator: Equal, valueBoolean: false}"#));
        assert!(weaviate.contains(r#"{path: ["bucket"], operator: LessThan, valueInt: 10}"#));
        assert!(weaviate.contains(r#"{path: ["owner"], operator: Equal, valueInt: 3}"#));
    }

    #[test]
    fn a_weaviate_shard_that_does_not_say_its_queue_is_an_error_not_settled() {
        let nodes = |shard: Value| json!({"nodes": [{"shards": [shard]}]});
        let ready =
            json!({"class": COLLECTION, "vectorQueueLength": 0, "vectorIndexingStatus": "READY"});
        assert!(Weaviate::shards_settled(&nodes(ready)).unwrap());
        let queued = json!({"class": COLLECTION, "vectorQueueLength": 3, "vectorIndexingStatus": "INDEXING"});
        assert!(!Weaviate::shards_settled(&nodes(queued)).unwrap());
        let silent = json!({"class": COLLECTION, "vectorIndexingStatus": "READY"});
        assert!(Weaviate::shards_settled(&nodes(silent)).is_err());
        let no_status = json!({"class": COLLECTION, "vectorQueueLength": 0});
        assert!(Weaviate::shards_settled(&nodes(no_status)).is_err());
        let other =
            json!({"class": FLOOR, "vectorQueueLength": 0, "vectorIndexingStatus": "READY"});
        assert!(
            Weaviate::shards_settled(&nodes(other)).is_err(),
            "no shard of the passages"
        );
    }

    #[test]
    fn vellevs_threads_must_be_as_set() {
        let text = "vellev_threads{pool=\"tokio_workers\"} 3\nvellev_threads{pool=\"rayon\"} 3\n";
        assert!(Vellev::threads_as_set(text, 3).is_ok());
        assert!(Vellev::threads_as_set(text, 4).is_err());
        let rayon_off =
            "vellev_threads{pool=\"tokio_workers\"} 3\nvellev_threads{pool=\"rayon\"} 4\n";
        assert!(Vellev::threads_as_set(rayon_off, 3).is_err());
        assert!(
            Vellev::threads_as_set("", 3).is_err(),
            "unreported is not as set"
        );
    }

    #[test]
    fn an_error_body_is_cut_at_a_character() {
        assert_eq!(cut("aé", 2), "a");
        assert_eq!(cut("abc", 10), "abc");
    }

    #[test]
    fn a_metric_is_read_by_its_whole_name() {
        let text = "vellev_tenants_unsettled_x 4\nvellev_tenants_unsettled 0\nvellev_threads{pool=\"rayon\"} 3\n";
        assert_eq!(Vellev::sample(text, "vellev_tenants_unsettled"), Some(0.0));
        assert_eq!(
            Vellev::sample(text, "vellev_threads{pool=\"rayon\"}"),
            Some(3.0)
        );
    }
}
