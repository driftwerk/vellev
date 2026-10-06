//! The same objects and searches through Weaviate and through vellev, the
//! answers compared. Runs only when `VELLEV_WEAVIATE_URL` names a Weaviate
//! to load, such as the one in `docker-compose.dev.yml`.
//!
//! Answers are compared by how many of each top ten are shared and how
//! alike their order is, not for equality, and the scores beside them are
//! reported, so that a difference shows where it lies.
//!
//! Weaviate 1.36's keyword search answers a small limit inexactly: asked
//! for ten it can leave out objects scoring above those it returns, which
//! asked for every match it ranks where BM25 puts them. vellev's keyword
//! search is compared with that full ranking, and with the answer to ten
//! only for the record. Weaviate's hybrid search fuses what its keyword
//! search finds, so it carries some of that error.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use common::*;
use serde_json::{json, Map, Value};
use vellev_api::{Content, Hybrid, Keyword, Nearest, Results};
use vellev_client::{Client, Options};
use vellev_core::{Collection, ObjectId, TenantId};
use vellev_index::filter::Filter;
use vellev_index::fusion::Fusion;

const CLASS: &str = "VellevDifferential";
const OBJECTS: usize = 3_000;
const DIMENSIONS: usize = 64;
const TOPICS: usize = 20;
const QUERIES: usize = 40;
const LIMIT: usize = 10;
const ALPHA: f32 = 0.75;
const OWNERS: [&str; 5] = ["ada", "bob", "cy", "team:core", "team:ops"];
const TAGS: [&str; 5] = [
    "lesson",
    "archived",
    "idea",
    "date:2026-05-07",
    "date:2026-05-08",
];

/// A small deterministic generator, so that every run loads the same.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in [-0.5, 0.5).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }

    fn vector(&mut self, around: &[f32], spread: f32) -> Vec<f32> {
        around.iter().map(|x| x + spread * self.unit()).collect()
    }
}

/// Text, vectors and the properties workforce filters on: objects about
/// one of a few topics each, worded mostly in that topic's words and near
/// that topic's direction.
struct Corpus {
    objects: Vec<(ObjectId, Map<String, Value>, Vec<f32>)>,
    queries: Vec<(String, Vec<f32>)>,
}

fn corpus() -> Corpus {
    let mut random = Random(0x5eed);
    let word = |topic: usize, n: usize| {
        format!(
            "{}{}",
            ["kv", "ra", "lo", "me", "tu"][n % 5],
            topic * 100 + n
        )
    };
    let common: Vec<String> = (0..200).map(|n| format!("w{n}")).collect();
    let centres: Vec<Vec<f32>> = (0..TOPICS)
        .map(|_| random.vector(&[0.0; DIMENSIONS], 2.0))
        .collect();
    let objects = (0..OBJECTS)
        .map(|n| {
            let topic = random.below(TOPICS);
            let length = 8 + random.below(24);
            let content: Vec<String> = (0..length)
                .map(|_| match random.below(3) {
                    0 => common[random.below(common.len())].clone(),
                    _ => word(topic, random.below(15)),
                })
                .collect();
            let tags: Vec<&str> = TAGS
                .iter()
                .copied()
                .filter(|_| random.below(3) == 0)
                .collect();
            let properties = json!({
                "content": content.join(" "),
                "owner": OWNERS[random.below(OWNERS.len())],
                "tags": tags,
                "deleted": random.below(10) == 0,
                "importance": (random.below(1_000) as f64) / 1_000.0,
                "count": random.below(500),
                "createdAt": format!("2026-05-{:02}T09:00:00Z", 1 + random.below(28)),
            });
            let id = serde_json::from_value(json!(id(n as u64))).unwrap();
            let vector = random.vector(&centres[topic], 1.5);
            (id, properties.as_object().unwrap().clone(), vector)
        })
        .collect();
    let queries = (0..QUERIES)
        .map(|_| {
            let topic = random.below(TOPICS);
            let words = 1 + random.below(3);
            let text: Vec<String> = (0..words).map(|_| word(topic, random.below(15))).collect();
            (text.join(" "), random.vector(&centres[topic], 1.0))
        })
        .collect();
    Corpus { objects, queries }
}

/// The collection in vellev's terms; Weaviate's class is made to match.
fn collection() -> Collection {
    serde_json::from_value(json!({
        "name": "memories",
        "properties": [
            { "name": "content", "data_type": "text", "searchable": true, "filterable": false },
            { "name": "owner", "data_type": "text" },
            { "name": "tags", "data_type": "text[]" },
            { "name": "deleted", "data_type": "bool" },
            { "name": "importance", "data_type": "number" },
            { "name": "count", "data_type": "int" },
            { "name": "createdAt", "data_type": "date" },
        ],
        "embedding": { "provider": "none", "model": "client", "dimensions": DIMENSIONS },
        "distance": "cosine",
        "bm25": { "k1": 1.2, "b": 0.75 },
        "access_rule": null,
    }))
    .unwrap()
}

/// The filters compared: none, the one workforce builds for what a user
/// may see, and simpler ones of each kind it is made of.
fn filters() -> Vec<(&'static str, Option<Value>)> {
    let visible = json!({ "and": [
        { "eq": { "property": "deleted", "value": false } },
        { "or": [
            { "eq": { "property": "owner", "value": "ada" } },
            { "eq": { "property": "owner", "value": "team:core" } }
        ] },
        { "or": [
            { "eq": { "property": "tags", "value": "lesson" } },
            { "eq": { "property": "tags", "value": "date:2026-05-07" } }
        ] },
        { "ne": { "property": "tags", "value": "archived" } },
        { "gte": { "property": "importance", "value": 0.4 } }
    ] });
    vec![
        ("none", None),
        (
            "not deleted",
            Some(json!({ "eq": { "property": "deleted", "value": false } })),
        ),
        (
            "importance range",
            Some(json!({ "and": [
            { "gte": { "property": "importance", "value": 0.25 } },
            { "lt": { "property": "importance", "value": 0.75 } }
        ] })),
        ),
        (
            "before a date",
            Some(json!({ "lt": { "property": "createdAt", "value": "2026-05-15T00:00:00Z" } })),
        ),
        ("visible to ada", Some(visible)),
    ]
}

/// A vellev filter as a Weaviate `where`.
fn weaviate_where(filter: &Value) -> Value {
    let (operator, operands) = filter.as_object().unwrap().iter().next().unwrap();
    let nested = |operator: &str| {
        let operands: Vec<Value> = operands
            .as_array()
            .unwrap()
            .iter()
            .map(weaviate_where)
            .collect();
        json!({ "operator": operator, "operands": operands })
    };
    match operator.as_str() {
        "and" => nested("And"),
        "or" => nested("Or"),
        compared => {
            let operator = match compared {
                "eq" => "Equal",
                "ne" => "NotEqual",
                "gt" => "GreaterThan",
                "gte" => "GreaterThanEqual",
                "lt" => "LessThan",
                "lte" => "LessThanEqual",
                other => panic!("no Weaviate operator for {other}"),
            };
            let property = operands["property"].as_str().unwrap();
            let value = &operands["value"];
            let typed = match property {
                "deleted" => "valueBoolean",
                "importance" => "valueNumber",
                "count" => "valueInt",
                "createdAt" => "valueDate",
                _ => "valueText",
            };
            json!({ "operator": operator, "path": [property], typed: value })
        }
    }
}

struct Weaviate {
    url: String,
    http: reqwest::Client,
}

impl Weaviate {
    async fn load(url: &str, corpus: &Corpus) -> Self {
        let weaviate = Self {
            url: url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        };
        let text = |name: &str, searchable: bool| {
            json!({
                "name": name,
                "dataType": ["text"],
                "tokenization": if searchable { "word" } else { "field" },
                "indexSearchable": searchable,
                "indexFilterable": !searchable,
            })
        };
        weaviate
            .http
            .delete(format!("{}/v1/schema/{CLASS}", weaviate.url))
            .send()
            .await
            .unwrap();
        weaviate
            .send("/v1/schema", json!({
                "class": CLASS,
                "vectorizer": "none",
                "vectorIndexType": "flat",
                "vectorIndexConfig": { "distance": "cosine" },
                "invertedIndexConfig": { "bm25": { "k1": 1.2, "b": 0.75 } },
                "properties": [
                    text("content", true),
                    text("owner", false),
                    { "name": "tags", "dataType": ["text[]"], "tokenization": "field", "indexSearchable": false },
                    { "name": "deleted", "dataType": ["boolean"] },
                    { "name": "importance", "dataType": ["number"] },
                    { "name": "count", "dataType": ["int"] },
                    { "name": "createdAt", "dataType": ["date"] },
                ],
            }))
            .await;
        for chunk in corpus.objects.chunks(500) {
            let objects: Vec<Value> = chunk
                .iter()
                .map(|(id, properties, vector)| {
                    json!({ "class": CLASS, "id": id, "properties": properties, "vector": vector })
                })
                .collect();
            let answer = weaviate
                .send("/v1/batch/objects", json!({ "objects": objects }))
                .await;
            let failed: Vec<&Value> = answer
                .as_array()
                .unwrap()
                .iter()
                .filter(|object| object["result"]["errors"].is_object())
                .collect();
            assert!(failed.is_empty(), "Weaviate refused objects: {failed:?}");
        }
        weaviate
    }

    async fn send(&self, path: &str, body: Value) -> Value {
        let answer = self
            .http
            .post(format!("{}{path}", self.url))
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = answer.status();
        let body: Value = answer.json().await.unwrap();
        assert!(status.is_success(), "{path}: {status} {body}");
        body
    }

    /// The first [`LIMIT`] ids found when asking for `limit`, best first,
    /// with the score Weaviate gives each.
    async fn search(
        &self,
        argument: &str,
        limit: usize,
        filter: Option<&Value>,
        fields: &str,
    ) -> Vec<(ObjectId, f32)> {
        let filter = filter
            .map(|filter| format!(", where: {}", graphql(&weaviate_where(filter))))
            .unwrap_or_default();
        let query = format!(
            "{{ Get {{ {CLASS}({argument}, limit: {limit}{filter}) {{ _additional {{ id {fields} }} }} }} }}"
        );
        let answer = self.send("/v1/graphql", json!({ "query": query })).await;
        assert!(answer.get("errors").is_none(), "{query}: {answer}");
        answer["data"]["Get"][CLASS]
            .as_array()
            .unwrap()
            .iter()
            .take(LIMIT)
            .map(|found| {
                let extra = &found["_additional"];
                let score = fields
                    .split(' ')
                    .find_map(|field| scalar(&extra[field]))
                    .unwrap();
                (serde_json::from_value(extra["id"].clone()).unwrap(), score)
            })
            .collect()
    }
}

/// A number Weaviate may give as a JSON number or as a string.
fn scalar(value: &Value) -> Option<f32> {
    match value {
        Value::Number(number) => number.as_f64().map(|n| n as f32),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

/// JSON written as a GraphQL input value: keys unquoted, operators bare.
fn graphql(value: &Value) -> String {
    match value {
        Value::Object(fields) => {
            let fields: Vec<String> = fields
                .iter()
                .map(|(key, value)| match key.as_str() {
                    "operator" => format!("{key}: {}", value.as_str().unwrap()),
                    _ => format!("{key}: {}", graphql(value)),
                })
                .collect();
            format!("{{ {} }}", fields.join(", "))
        }
        Value::Array(items) => {
            let items: Vec<String> = items.iter().map(graphql).collect();
            format!("[{}]", items.join(", "))
        }
        other => other.to_string(),
    }
}

fn vector_literal(vector: &[f32]) -> String {
    let values: Vec<String> = vector.iter().map(f32::to_string).collect();
    format!("[{}]", values.join(", "))
}

/// The ids vellev found, best first, with the score that `score` reads.
fn ids(
    results: Results,
    score: impl Fn(&vellev_api::Found) -> Option<f32>,
) -> Vec<(ObjectId, f32)> {
    results
        .results
        .iter()
        .map(|found| (found.object.id, score(found).unwrap()))
        .collect()
}

/// How alike two answers are.
#[derive(Debug, Default, Clone, Copy)]
struct Agreement {
    /// The share of the shorter answer the other holds too.
    overlap: f64,
    /// Kendall's tau over the objects both hold, 1 when they are ordered
    /// alike.
    order: f64,
    /// The largest difference between the scores both give one object.
    score: f64,
}

fn agreement(ours: &[(ObjectId, f32)], theirs: &[(ObjectId, f32)]) -> Agreement {
    let rank: HashMap<ObjectId, (usize, f32)> = theirs
        .iter()
        .enumerate()
        .map(|(at, (id, score))| (*id, (at, *score)))
        .collect();
    let shared: Vec<(usize, usize, f32, f32)> = ours
        .iter()
        .enumerate()
        .filter_map(|(at, (id, score))| {
            rank.get(id)
                .map(|(other, theirs)| (at, *other, *score, *theirs))
        })
        .collect();
    let shorter = ours.len().min(theirs.len());
    let overlap = match shorter {
        0 => 1.0,
        _ => shared.len() as f64 / shorter as f64,
    };
    let pairs = shared.len() * shared.len().saturating_sub(1) / 2;
    let concordant: i64 = (0..shared.len())
        .flat_map(|i| (i + 1..shared.len()).map(move |j| (i, j)))
        .map(|(i, j)| ((shared[i].1 < shared[j].1) as i64) * 2 - 1)
        .sum();
    let order = match pairs {
        0 => 1.0,
        _ => concordant as f64 / pairs as f64,
    };
    let score = shared
        .iter()
        .map(|(_, _, ours, theirs)| f64::from((ours - theirs).abs()))
        .fold(0.0, f64::max);
    Agreement {
        overlap,
        order,
        score,
    }
}

/// The least a kind of search must agree, on average over every query and
/// filter.
struct Bar {
    overlap: f64,
    order: f64,
}

/// What is compared: vellev's answers to one kind of search with
/// Weaviate's, and how alike they must be, if at all.
struct Kind {
    name: &'static str,
    bar: Option<Bar>,
    found: Vec<Agreement>,
}

impl Kind {
    fn new(name: &'static str, bar: Option<Bar>) -> Self {
        Self {
            name,
            bar,
            found: Vec::new(),
        }
    }
}

#[tokio::test]
async fn vellev_answers_as_weaviate_does() {
    let Ok(url) = std::env::var("VELLEV_WEAVIATE_URL") else {
        eprintln!("VELLEV_WEAVIATE_URL is not set; not comparing with Weaviate");
        return;
    };
    let corpus = corpus();
    let weaviate = Weaviate::load(&url, &corpus).await;

    let served = Served::new().await;
    let options = Options {
        timeout: Duration::from_secs(30),
        batch_bytes: MAX_BODY / 2,
        ..Options::default()
    };
    let client = Client::connect(&[&served.url], KEY, options).unwrap();
    let tenant = client.tenant(&TenantId::new("differential").unwrap());
    tenant.create().await.unwrap();
    let schema = collection();
    let memories = tenant.collection(&schema.name);
    memories.create(&schema).await.unwrap();
    memories
        .store_all(corpus.objects.iter().map(|(id, properties, vector)| {
            let content = Content {
                properties: properties.clone(),
                vector: Some(vector.clone()),
            };
            (*id, content)
        }))
        .await
        .unwrap();

    let bar = |overlap, order| Some(Bar { overlap, order });
    let mut kinds = [
        Kind::new("vector", bar(0.99, 0.95)),
        Kind::new("keyword", bar(0.95, 0.9)),
        Kind::new("keyword, Weaviate at 10", None),
        Kind::new("hybrid", bar(0.9, 0.8)),
    ];
    let mut report = Vec::new();
    for (label, filter) in filters() {
        let parsed: Option<Filter> = filter
            .clone()
            .map(|filter| serde_json::from_value(filter).unwrap());
        let filter = filter.as_ref();
        let mut found: [Vec<Agreement>; 4] = Default::default();
        for (text, vector) in &corpus.queries {
            let nearest = Nearest {
                vector: Some(vector.clone()),
                text: None,
                limit: LIMIT as u32,
                filter: parsed.clone(),
                oversample: None,
                vectors: false,
            };
            let ours = ids(memories.nearest(&nearest).await.unwrap(), |found| {
                found.certainty
            });
            let argument = format!("nearVector: {{ vector: {} }}", vector_literal(vector));
            let theirs = weaviate.search(&argument, LIMIT, filter, "certainty").await;
            found[0].push(agreement(&ours, &theirs));

            let keyword = Keyword {
                text: text.clone(),
                properties: Vec::new(),
                limit: LIMIT as u32,
                filter: parsed.clone(),
                vectors: false,
            };
            let ours = ids(memories.keyword(&keyword).await.unwrap(), |found| {
                found.keyword
            });
            let argument = format!("bm25: {{ query: {} }}", json!(text));
            let every = weaviate.search(&argument, OBJECTS, filter, "score").await;
            found[1].push(agreement(&ours, &every));
            let ten = weaviate.search(&argument, LIMIT, filter, "score").await;
            found[2].push(agreement(&ours, &ten));

            let hybrid = Hybrid {
                text: text.clone(),
                vector: Some(vector.clone()),
                alpha: Some(ALPHA),
                fusion: Fusion::RelativeScore,
                properties: Vec::new(),
                limit: LIMIT as u32,
                candidates: None,
                filter: parsed.clone(),
                oversample: None,
                vectors: false,
            };
            let ours = ids(memories.hybrid(&hybrid).await.unwrap(), |found| found.score);
            let argument = format!(
                "hybrid: {{ query: {}, vector: {}, alpha: {ALPHA}, fusionType: relativeScoreFusion }}",
                json!(text),
                vector_literal(vector)
            );
            let theirs = weaviate.search(&argument, LIMIT, filter, "score").await;
            found[3].push(agreement(&ours, &theirs));
        }
        for (kind, found) in kinds.iter_mut().zip(found) {
            report.push((label, kind.name, summary(&found)));
            kind.found.extend(found);
        }
    }

    eprintln!(
        "{:<18} {:<24} {:>8} {:>8} {:>12}",
        "filter", "search", "overlap", "order", "max score Δ"
    );
    for (label, kind, (overlap, order, score)) in &report {
        eprintln!("{label:<18} {kind:<24} {overlap:>8.3} {order:>8.3} {score:>12.4}");
    }
    for kind in &kinds {
        let Some(bar) = &kind.bar else { continue };
        let (overlap, order, _) = summary(&kind.found);
        assert!(
            overlap >= bar.overlap && order >= bar.order,
            "{}: overlap {overlap:.3} (at least {}), order {order:.3} (at least {})",
            kind.name,
            bar.overlap,
            bar.order
        );
    }
}

/// The mean overlap and order, and the largest score difference.
fn summary(agreements: &[Agreement]) -> (f64, f64, f64) {
    let mean = |value: fn(&Agreement) -> f64| {
        agreements.iter().map(value).sum::<f64>() / agreements.len() as f64
    };
    let score = agreements
        .iter()
        .map(|agreement| agreement.score)
        .fold(0.0, f64::max);
    (mean(|a| a.overlap), mean(|a| a.order), score)
}
