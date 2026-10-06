//! Copying a Weaviate instance into a vellev tenant.
//!
//! Each class becomes a collection of the same name, with what Weaviate
//! knows of its properties, distance and BM25 parameters. Its objects are
//! walked with Weaviate's cursor and stored under the same ids, with their
//! vectors, or without them for a class vellev is to embed itself. The
//! last id stored of each class is written to a state file after every
//! page, so an import that stops resumes where it left off.
//!
//! Weaviate's cursor is not a snapshot, so once every class is copied a
//! catch-up pass copies again each object changed since the import began
//! and deletes each one Weaviate no longer holds. Finally each class's
//! count is compared with Weaviate's.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use eyre::{bail, eyre, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use vellev_api::{Content, Page};
use vellev_client::{Records, Tenant};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider, ObjectId,
    Property,
};

/// A Weaviate to copy from.
#[derive(Debug, Clone)]
pub struct Source {
    pub url: String,
    pub key: Option<String>,
    /// The Weaviate tenant to copy, for classes with multi-tenancy on.
    pub tenant: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Options {
    /// The classes to copy; every class when empty.
    pub classes: Vec<String>,
    /// For each class to be embedded by vellev rather than copied with its
    /// vectors, how.
    pub embeddings: HashMap<String, Embedding>,
    /// A date property to hold each object's creation time in Weaviate.
    pub created_as: Option<String>,
    /// A date property to hold the time each object was last changed.
    pub updated_as: Option<String>,
    /// How many objects are read and stored at a time, and how many ids
    /// listed from vellev to catch up: at most the node's limit on results.
    pub page: usize,
    /// The most objects copied a second, so as not to overload the source.
    pub per_second: Option<f64>,
    /// Where progress is kept, to resume from.
    pub state: PathBuf,
}

/// What an import did, by class.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Report {
    pub classes: BTreeMap<String, Copied>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Copied {
    /// Objects stored by the first pass in this run.
    pub stored: u64,
    /// Objects stored again by the catch-up pass, changed since it began.
    pub changed: u64,
    /// Objects deleted by the catch-up pass, gone from Weaviate.
    pub deleted: u64,
    /// How many objects Weaviate and the collection held at the end.
    pub weaviate: u64,
    pub vellev: u64,
    /// What was mapped but means something different in vellev.
    pub warnings: Vec<String>,
}

/// Copy the classes `options` names from `source` into `tenant`.
pub async fn weaviate(source: &Source, tenant: &Tenant, options: &Options) -> Result<Report> {
    let weaviate = Weaviate::new(source)?;
    let mut state = Progress::load(&options.state)?;
    let classes = weaviate.classes(&options.classes).await?;
    let mut report = Report::default();
    tenant.create().await?;
    let mut plans = Vec::new();
    for class in &classes {
        let sample = weaviate.page(&class.class, None, 1, true).await?;
        let embedding = match options.embeddings.get(&class.class) {
            Some(embedding) => embedding.clone(),
            None => copied_embedding(class, &sample)?,
        };
        let (collection, warnings) = collection(class, embedding, options)?;
        tenant
            .collection(&collection.name)
            .create(&collection)
            .await
            .wrap_err_with(|| format!("creating collection {}", collection.name))?;
        let embedded = options.embeddings.contains_key(&class.class);
        report.classes.insert(
            class.class.clone(),
            Copied {
                warnings,
                ..Copied::default()
            },
        );
        plans.push((
            class.class.clone(),
            tenant.collection(&collection.name),
            embedded,
        ));
    }

    for (class, records, embedded) in &plans {
        let copied = report.classes.get_mut(class).expect("planned");
        let mut after = state.class(class).after.clone();
        let mut pacer = Pacer::new(options.per_second);
        while !state.class(class).done {
            let page = weaviate
                .page(class, after.as_deref(), options.page, !embedded)
                .await?;
            let Some(last) = page.last() else {
                state.class(class).done = true;
                state.save(&options.state)?;
                break;
            };
            let last = last.id.clone();
            let objects = page
                .iter()
                .map(|object| object.content(options, !embedded))
                .collect::<Result<Vec<_>>>()?;
            copied.stored += objects.len() as u64;
            records.store_all(objects).await?;
            state.class(class).after = Some(last.clone());
            after = Some(last);
            state.save(&options.state)?;
            pacer.paced(page.len()).await;
        }
    }

    if !state.caught_up {
        for (class, records, embedded) in &plans {
            let copied = report.classes.get_mut(class).expect("planned");
            catch_up(
                &weaviate,
                class,
                records,
                *embedded,
                state.started,
                options,
                copied,
            )
            .await?;
        }
        state.caught_up = true;
        state.save(&options.state)?;
    }

    for (class, records, _) in &plans {
        let copied = report.classes.get_mut(class).expect("planned");
        copied.weaviate = weaviate.count(class).await?;
        copied.vellev = records.count(None).await?;
        if copied.weaviate != copied.vellev {
            bail!(
                "class {class}: Weaviate holds {} objects and vellev {}",
                copied.weaviate,
                copied.vellev
            );
        }
    }
    Ok(report)
}

/// Store again each object of `class` changed since `started`, and delete
/// each that Weaviate no longer holds.
async fn catch_up(
    weaviate: &Weaviate,
    class: &str,
    records: &Records,
    embedded: bool,
    started: i64,
    options: &Options,
    copied: &mut Copied,
) -> Result<()> {
    let mut held = HashSet::new();
    let mut after: Option<String> = None;
    loop {
        let page = weaviate
            .page(class, after.as_deref(), options.page, !embedded)
            .await?;
        let Some(last) = page.last() else { break };
        after = Some(last.id.clone());
        let changed = page
            .iter()
            .filter(|object| object.updated.unwrap_or(i64::MAX) >= started)
            .map(|object| object.content(options, !embedded))
            .collect::<Result<Vec<_>>>()?;
        held.extend(page.iter().map(|object| object.id.clone()));
        copied.changed += changed.len() as u64;
        records.store_all(changed).await?;
    }
    let mut after = None;
    loop {
        let listed = records
            .list(&Page {
                filter: None,
                after,
                limit: options.page as u32,
                vectors: false,
            })
            .await?;
        for stored in listed.objects {
            if !held.contains(&stored.id.to_string()) {
                records.delete(stored.id).await?;
                copied.deleted += 1;
            }
        }
        match listed.next {
            Some(next) => after = Some(next),
            None => return Ok(()),
        }
    }
}

/// Holds copying to the given rate, near enough.
struct Pacer {
    per_second: Option<f64>,
    started: Instant,
    copied: usize,
}

impl Pacer {
    fn new(per_second: Option<f64>) -> Self {
        Self {
            per_second,
            started: Instant::now(),
            copied: 0,
        }
    }

    async fn paced(&mut self, copied: usize) {
        self.copied += copied;
        if let Some(per_second) = self.per_second.filter(|rate| *rate > 0.0) {
            let due = Duration::from_secs_f64(self.copied as f64 / per_second);
            if let Some(wait) = due.checked_sub(self.started.elapsed()) {
                tokio::time::sleep(wait).await;
            }
        }
    }
}

/// How far an import has come, kept in its state file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Progress {
    /// When the import began, in milliseconds since the epoch: what
    /// changed after it is copied again by the catch-up pass.
    started: i64,
    classes: BTreeMap<String, ClassProgress>,
    caught_up: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ClassProgress {
    /// The last id stored.
    after: Option<String>,
    done: bool,
}

impl Progress {
    fn load(path: &Path) -> Result<Self> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .wrap_err_with(|| format!("reading the import state in {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                started: Utc::now().timestamp_millis(),
                classes: BTreeMap::new(),
                caught_up: false,
            }),
            Err(error) => Err(error).wrap_err_with(|| format!("reading {}", path.display())),
        }
    }

    fn class(&mut self, class: &str) -> &mut ClassProgress {
        self.classes.entry(class.to_string()).or_default()
    }

    /// Written whole under another name and moved into place, so a stop
    /// part way through never leaves half a file.
    fn save(&self, path: &Path) -> Result<()> {
        let partial = path.with_extension("partial");
        std::fs::write(&partial, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&partial, path)
            .wrap_err_with(|| format!("writing the import state to {}", path.display()))
    }
}

/// A class as Weaviate describes it, as far as an import needs.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Class {
    pub class: String,
    #[serde(default)]
    pub properties: Vec<WeaviateProperty>,
    #[serde(default)]
    pub vectorizer: Option<String>,
    #[serde(default)]
    pub vector_index_config: Option<Value>,
    #[serde(default)]
    pub inverted_index_config: Option<Value>,
    #[serde(default)]
    pub vector_config: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WeaviateProperty {
    pub name: String,
    pub data_type: Vec<String>,
    #[serde(default)]
    pub tokenization: Option<String>,
    #[serde(default)]
    pub index_filterable: Option<bool>,
    #[serde(default)]
    pub index_searchable: Option<bool>,
}

/// The collection `class` becomes, embedding as given, with what of it
/// means something different in vellev. Whatever vellev cannot hold is
/// refused, all of it at once.
pub fn collection(
    class: &Class,
    embedding: Embedding,
    options: &Options,
) -> Result<(Collection, Vec<String>)> {
    let mut problems = Vec::new();
    let mut warnings = Vec::new();
    let name = CollectionName::new(&class.class)
        .map_err(|error| eyre!("class {}: {error}", class.class))?;
    if class
        .vector_config
        .as_ref()
        .is_some_and(|named| !named.is_empty())
    {
        problems.push("it has named vectors, and a collection holds one vector".to_string());
    }
    let mut properties: Vec<Property> = class
        .properties
        .iter()
        .filter_map(|property| {
            let data_type = match property.data_type.as_slice() {
                [kind] => match kind.as_str() {
                    "text" | "string" => Some(DataType::Text),
                    "text[]" | "string[]" => Some(DataType::TextArray),
                    "int" => Some(DataType::Int),
                    "number" => Some(DataType::Number),
                    "boolean" => Some(DataType::Bool),
                    "date" => Some(DataType::Date),
                    _ => None,
                },
                _ => None,
            };
            let Some(data_type) = data_type else {
                problems.push(format!(
                    "property {} is {:?}, which vellev cannot hold",
                    property.name, property.data_type
                ));
                return None;
            };
            let text = matches!(data_type, DataType::Text | DataType::TextArray);
            let filterable = property.index_filterable.unwrap_or(true);
            let searchable = text && property.index_searchable.unwrap_or(true);
            let tokenization = property.tokenization.as_deref().unwrap_or("word");
            if text && filterable && tokenization != "field" {
                warnings.push(format!(
                    "property {} is filtered by {tokenization} in Weaviate and by whole value in vellev",
                    property.name
                ));
            }
            Some(Property {
                name: property.name.clone(),
                data_type,
                filterable,
                searchable,
            })
        })
        .collect();
    for time in [&options.created_as, &options.updated_as]
        .into_iter()
        .flatten()
    {
        if properties.iter().any(|property| &property.name == time) {
            problems.push(format!(
                "property {time}, to hold a time, is already a property"
            ));
        }
        properties.push(Property::new(time, DataType::Date));
    }
    let distance = class
        .vector_index_config
        .as_ref()
        .and_then(|config| config.get("distance"))
        .and_then(Value::as_str);
    let distance = match distance {
        None | Some("cosine") => Distance::Cosine,
        Some("dot") => Distance::Dot,
        Some("l2-squared") => Distance::L2,
        Some(other) => {
            problems.push(format!(
                "its distance is {other}, which vellev does not measure"
            ));
            Distance::Cosine
        }
    };
    let bm25 = class
        .inverted_index_config
        .as_ref()
        .and_then(|config| config.get("bm25"))
        .map(|bm25| {
            let value = |name: &str, default: f32| {
                bm25.get(name)
                    .and_then(Value::as_f64)
                    .map_or(default, |value| value as f32)
            };
            let defaults = Bm25::default();
            Bm25 {
                k1: value("k1", defaults.k1),
                b: value("b", defaults.b),
            }
        })
        .unwrap_or_default();
    let collection = Collection {
        name,
        properties,
        embedding,
        distance,
        bm25,
        access_rule: None,
    };
    if let Err(error) = collection.validate() {
        problems.push(error.to_string());
    }
    if !problems.is_empty() {
        bail!(
            "class {} cannot be imported: {}",
            class.class,
            problems.join("; ")
        );
    }
    Ok((collection, warnings))
}

/// How a class copied with its vectors embeds: as the client supplies,
/// with the dimensions its first object's vector has.
fn copied_embedding(class: &Class, sample: &[Object]) -> Result<Embedding> {
    let Some(vector) = sample.first().and_then(|object| object.vector.as_ref()) else {
        bail!(
            "class {} has no vector to copy; say how vellev is to embed it",
            class.class
        );
    };
    Ok(Embedding {
        provider: EmbeddingProvider::None,
        model: format!("weaviate:{}", class.vectorizer.as_deref().unwrap_or("none")),
        dimensions: vector.len() as u32,
        query_prefix: None,
        document_prefix: None,
        properties: Vec::new(),
    })
}

/// An object as Weaviate's REST API gives it.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Object {
    id: String,
    #[serde(default)]
    properties: Map<String, Value>,
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default, deserialize_with = "unix_millis")]
    creation_time_unix: Option<i64>,
    #[serde(
        default,
        rename = "lastUpdateTimeUnix",
        deserialize_with = "unix_millis"
    )]
    updated: Option<i64>,
}

/// Weaviate gives times as numbers or as strings of digits.
fn unix_millis<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<i64>, D::Error> {
    Ok(match Option::<Value>::deserialize(d)? {
        Some(Value::Number(number)) => number.as_i64(),
        Some(Value::String(text)) => text.parse().ok(),
        _ => None,
    })
}

impl Object {
    fn content(&self, options: &Options, vectors: bool) -> Result<(ObjectId, Content)> {
        let id: ObjectId = serde_json::from_value(json!(self.id))
            .wrap_err_with(|| format!("object id {}", self.id))?;
        let mut properties: Map<String, Value> = self
            .properties
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let times = [
            (&options.created_as, self.creation_time_unix),
            (&options.updated_as, self.updated),
        ];
        for (name, millis) in times {
            let (Some(name), Some(millis)) = (name, millis) else {
                continue;
            };
            let at = DateTime::<Utc>::from_timestamp_millis(millis)
                .ok_or_else(|| eyre!("object {} has time {millis}", self.id))?;
            properties.insert(name.clone(), json!(at.to_rfc3339()));
        }
        let vector = self.vector.clone().filter(|_| vectors);
        Ok((id, Content { properties, vector }))
    }
}

struct Weaviate {
    url: String,
    tenant: Option<String>,
    http: reqwest::Client,
}

impl Weaviate {
    fn new(source: &Source) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = &source.key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {key}").parse()?,
            );
        }
        Ok(Self {
            url: source.url.trim_end_matches('/').to_string(),
            tenant: source.tenant.clone(),
            http: reqwest::Client::builder()
                .default_headers(headers)
                .timeout(Duration::from_secs(120))
                .build()?,
        })
    }

    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Value> {
        let answer = self
            .http
            .get(format!("{}{path}", self.url))
            .query(query)
            .send()
            .await
            .wrap_err_with(|| format!("asking Weaviate for {path}"))?;
        let status = answer.status();
        let body: Value = answer.json().await?;
        if !status.is_success() {
            bail!("Weaviate answered {path} with {status}: {body}");
        }
        Ok(body)
    }

    /// The classes named, or every class.
    async fn classes(&self, named: &[String]) -> Result<Vec<Class>> {
        let schema = self.get("/v1/schema", &[]).await?;
        let classes: Vec<Class> = serde_json::from_value(schema["classes"].clone())
            .wrap_err("reading Weaviate's schema")?;
        if named.is_empty() {
            return Ok(classes);
        }
        named
            .iter()
            .map(|name| {
                classes
                    .iter()
                    .find(|class| &class.class == name)
                    .cloned()
                    .ok_or_else(|| eyre!("Weaviate has no class {name}"))
            })
            .collect()
    }

    /// Up to `limit` objects of `class` after the one with id `after`.
    async fn page(
        &self,
        class: &str,
        after: Option<&str>,
        limit: usize,
        vectors: bool,
    ) -> Result<Vec<Object>> {
        let mut query = vec![("class", class.to_string()), ("limit", limit.to_string())];
        query.extend(after.map(|after| ("after", after.to_string())));
        query.extend(vectors.then(|| ("include", "vector".to_string())));
        query.extend(self.tenant.clone().map(|tenant| ("tenant", tenant)));
        let page = self.get("/v1/objects", &query).await?;
        serde_json::from_value(page["objects"].clone())
            .wrap_err_with(|| format!("reading a page of {class}"))
    }

    async fn count(&self, class: &str) -> Result<u64> {
        let tenant = self
            .tenant
            .as_ref()
            .map(|tenant| format!("(tenant: {})", json!(tenant)))
            .unwrap_or_default();
        let query = format!("{{ Aggregate {{ {class}{tenant} {{ meta {{ count }} }} }} }}");
        let answer: Value = self
            .http
            .post(format!("{}/v1/graphql", self.url))
            .json(&json!({ "query": query }))
            .send()
            .await?
            .json()
            .await?;
        answer["data"]["Aggregate"][class][0]["meta"]["count"]
            .as_u64()
            .ok_or_else(|| eyre!("Weaviate did not count {class}: {answer}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(properties: Value, extra: Value) -> Class {
        let mut class = json!({ "class": "Memory", "properties": properties });
        class
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        serde_json::from_value(class).unwrap()
    }

    fn options() -> Options {
        Options {
            classes: vec![],
            embeddings: HashMap::new(),
            created_as: None,
            updated_as: None,
            page: 100,
            per_second: None,
            state: PathBuf::new(),
        }
    }

    fn embedding() -> Embedding {
        Embedding {
            provider: EmbeddingProvider::None,
            model: "weaviate:none".into(),
            dimensions: 4,
            query_prefix: None,
            document_prefix: None,
            properties: vec![],
        }
    }

    #[test]
    fn a_class_becomes_a_collection_with_its_types_indexes_distance_and_bm25() {
        let class = class(
            json!([
                { "name": "content", "dataType": ["text"], "indexFilterable": false },
                { "name": "owner", "dataType": ["text"], "tokenization": "field", "indexSearchable": false },
                { "name": "tags", "dataType": ["text[]"] },
                { "name": "count", "dataType": ["int"] },
                { "name": "importance", "dataType": ["number"] },
                { "name": "deleted", "dataType": ["boolean"] },
                { "name": "createdAt", "dataType": ["date"] },
            ]),
            json!({
                "vectorIndexConfig": { "distance": "dot" },
                "invertedIndexConfig": { "bm25": { "k1": 1.5, "b": 0.5 } },
            }),
        );
        let (collection, warnings) = collection(
            &class,
            embedding(),
            &Options {
                updated_as: Some("updatedAt".into()),
                ..options()
            },
        )
        .unwrap();
        let found: Vec<(&str, DataType, bool, bool)> = collection
            .properties
            .iter()
            .map(|p| (p.name.as_str(), p.data_type, p.filterable, p.searchable))
            .collect();
        assert_eq!(
            found,
            [
                ("content", DataType::Text, false, true),
                ("owner", DataType::Text, true, false),
                ("tags", DataType::TextArray, true, true),
                ("count", DataType::Int, true, false),
                ("importance", DataType::Number, true, false),
                ("deleted", DataType::Bool, true, false),
                ("createdAt", DataType::Date, true, false),
                ("updatedAt", DataType::Date, true, false),
            ]
        );
        assert_eq!(collection.distance, Distance::Dot);
        assert_eq!((collection.bm25.k1, collection.bm25.b), (1.5, 0.5));
        assert_eq!(
            warnings,
            ["property tags is filtered by word in Weaviate and by whole value in vellev"]
        );
    }

    #[test]
    fn what_vellev_cannot_hold_is_refused_all_at_once() {
        let class = class(
            json!([
                { "name": "place", "dataType": ["geoCoordinates"] },
                { "name": "author", "dataType": ["Person"] },
                { "name": "createdAt", "dataType": ["date"] },
            ]),
            json!({
                "vectorIndexConfig": { "distance": "hamming" },
                "vectorConfig": { "title": {} },
            }),
        );
        let refused = collection(
            &class,
            embedding(),
            &Options {
                created_as: Some("createdAt".into()),
                ..options()
            },
        )
        .unwrap_err()
        .to_string();
        for reason in [
            "named vectors",
            "place is [\"geoCoordinates\"]",
            "author is [\"Person\"]",
            "createdAt, to hold a time, is already a property",
            "distance is hamming",
        ] {
            assert!(refused.contains(reason), "{reason} in {refused}");
        }
    }
}
