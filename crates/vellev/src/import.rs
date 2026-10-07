//! Copying a Weaviate instance into a vellev tenant.
//!
//! Each class becomes a collection of the same name, with what Weaviate
//! knows of its properties, distance and BM25 parameters. Its objects are
//! walked with Weaviate's cursor and stored under the same ids, with their
//! vectors, or without them for a class vellev is to embed itself. The
//! last id stored of each class is written to a state file after every
//! page, so an import that stops resumes where it left off.
//!
//! Weaviate's cursor is not a snapshot. Every run therefore ends, for each
//! class copied whole, with a catch-up pass: it stores again each object
//! changed since the class was last caught up (less a margin for the two
//! clocks disagreeing) and deletes each one Weaviate no longer holds. Run
//! again, an import is caught up again, so it can follow Weaviate until
//! writes move over; only a pass made once Weaviate takes no more writes
//! is exact. Each class's count is then compared with Weaviate's.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
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

/// How many times a request to Weaviate is tried before the import stops.
const ATTEMPTS: u32 = 5;

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
    /// For each class with several named vectors, the one to copy.
    pub vectors: HashMap<String, String>,
    /// For each class copied with its vectors whose first objects have
    /// none, how many dimensions its vectors are to have.
    pub dimensions: HashMap<String, u32>,
    /// A date property to hold each object's creation time in Weaviate.
    pub created_as: Option<String>,
    /// A date property to hold the time each object was last changed.
    pub updated_as: Option<String>,
    /// How many objects are read and stored at a time, and how many ids
    /// listed from vellev to catch up: at most the node's limit on results.
    pub page: usize,
    /// The most objects copied a second, so as not to overload the source.
    pub per_second: Option<f64>,
    /// How far Weaviate's clock may be behind this one: a catch-up looks
    /// back this much further for changes.
    pub skew: Duration,
    /// Where progress is kept, to resume from.
    pub state: PathBuf,
}

/// What an import did, by class.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Report {
    pub classes: BTreeMap<String, Copied>,
}

impl Report {
    /// Whether every class holds as many objects in vellev as in Weaviate.
    pub fn matched(&self) -> bool {
        self.classes
            .values()
            .all(|copied| copied.weaviate == copied.vellev)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Copied {
    /// Objects stored by the first pass in this run.
    pub stored: u64,
    /// Objects stored again by the catch-up pass, changed or new.
    pub changed: u64,
    /// Objects deleted by the catch-up pass, gone from Weaviate.
    pub deleted: u64,
    /// How many objects Weaviate and the collection held at the end.
    pub weaviate: u64,
    pub vellev: u64,
    /// What was mapped but means something different in vellev.
    pub warnings: Vec<String>,
}

/// One class to copy, as planned.
struct Plan {
    class: Class,
    records: Records,
    vector: Copy,
    dimensions: u32,
    /// When each object stored by this run's first pass had last changed,
    /// so that the catch-up does not store it again unchanged.
    stored: HashMap<ObjectId, Option<i64>>,
}

/// Which of an object's vectors is copied.
#[derive(Debug, Clone, PartialEq)]
enum Copy {
    /// None: vellev embeds the class.
    Embedded,
    /// The one vector of a class without named vectors.
    Vector,
    Named(String),
}

/// Copy the classes `options` names from `source` into `tenant`, and
/// catch up with what changed in Weaviate meanwhile.
pub async fn weaviate(source: &Source, tenant: &Tenant, options: &Options) -> Result<Report> {
    if options.page == 0 {
        bail!("a page holds at least one object");
    }
    if options.per_second.is_some_and(|rate| rate <= 0.0) {
        bail!("the rate of copying must be more than nothing");
    }
    let mut state = Progress::load(&options.state, source, tenant)?;
    let weaviate = Weaviate::new(source)?;
    let classes = weaviate.classes(&options.classes).await?;
    let named = options.embeddings.keys().chain(options.vectors.keys());
    for named in named.chain(options.dimensions.keys()) {
        if !classes.iter().any(|class| &class.class == named) {
            bail!("{named} is not a class being imported");
        }
    }
    let mut report = Report::default();
    tenant.create().await?;
    let mut plans = Vec::new();
    for class in classes {
        let (plan, collection, warnings) = plan(&weaviate, class, tenant, options).await?;
        plan.records
            .create(&collection)
            .await
            .wrap_err_with(|| format!("creating collection {}", collection.name))?;
        let copied = Copied {
            warnings,
            ..Copied::default()
        };
        report.classes.insert(plan.class.class.clone(), copied);
        plans.push(plan);
    }

    for plan in &mut plans {
        let name = &plan.class.class;
        let copied = report.classes.get_mut(name).expect("planned");
        let vectors = plan.vector != Copy::Embedded;
        let mut pacer = Pacer::new(options.per_second);
        while !state.class(name).done {
            let after = state.class(name).after.clone();
            let page = weaviate
                .page(&plan.class, after.as_deref(), options.page, vectors)
                .await?;
            let Some(last) = page.last() else {
                state.class(name).done = true;
                break;
            };
            let last = last.id.clone();
            let objects = page
                .iter()
                .map(|object| object.content(plan, options))
                .collect::<Result<Vec<_>>>()?;
            for (object, (id, _)) in page.iter().zip(&objects) {
                plan.stored.insert(*id, object.updated);
            }
            copied.stored += objects.len() as u64;
            plan.records.store_all(objects).await?;
            state.class(name).after = Some(last);
            state.save(&options.state)?;
            pacer.paced(page.len()).await;
        }
        state.save(&options.state)?;
    }

    for plan in &plans {
        let name = &plan.class.class;
        let copied = report.classes.get_mut(name).expect("planned");
        let begun = Utc::now().timestamp_millis();
        let since = state.class(name).caught_up_to.unwrap_or(state.started)
            - options.skew.as_millis() as i64;
        catch_up(&weaviate, plan, since, options, copied).await?;
        state.class(name).caught_up_to = Some(begun);
        state.save(&options.state)?;
        copied.weaviate = weaviate.count(&plan.class).await?;
        copied.vellev = plan.records.count(None).await?;
    }
    Ok(report)
}

/// How to copy `class`: the collection it becomes, and what of it means
/// something different in vellev.
async fn plan(
    weaviate: &Weaviate,
    class: Class,
    tenant: &Tenant,
    options: &Options,
) -> Result<(Plan, Collection, Vec<String>)> {
    if class.multi_tenant() && weaviate.tenant.is_none() {
        bail!(
            "class {} has multi-tenancy on; say which Weaviate tenant to copy",
            class.class
        );
    }
    let named: Vec<&String> = class.vector_config.iter().flat_map(Map::keys).collect();
    let vector = match (
        options.embeddings.contains_key(&class.class),
        named.as_slice(),
    ) {
        (true, _) => Copy::Embedded,
        (false, []) => Copy::Vector,
        (false, [only]) => Copy::Named((*only).clone()),
        (false, _) => match options.vectors.get(&class.class) {
            Some(chosen) if named.contains(&chosen) => Copy::Named(chosen.clone()),
            _ => bail!(
                "class {} has named vectors {named:?}; say which to copy",
                class.class
            ),
        },
    };
    let mut vectorless = None;
    let embedding = match options.embeddings.get(&class.class) {
        Some(embedding) => embedding.clone(),
        None => {
            let sample = weaviate.page(&class, None, options.page, true).await?;
            let given = options.dimensions.get(&class.class).copied();
            let (embedding, found) = copied_embedding(&class, &vector, &sample, given)?;
            vectorless = (!found).then(|| {
                format!(
                    "no vector was found in its first objects; it holds vectors of {} dimensions",
                    embedding.dimensions
                )
            });
            embedding
        }
    };
    let (collection, mut warnings) = collection(&class, &vector, embedding, options)?;
    warnings.extend(vectorless);
    let plan = Plan {
        records: tenant.collection(&collection.name),
        dimensions: collection.embedding.dimensions,
        class,
        vector,
        stored: HashMap::new(),
    };
    Ok((plan, collection, warnings))
}

/// Store again each object of the class changed since `since`, or made
/// since the walk passed its id, and delete each that Weaviate no longer
/// holds.
async fn catch_up(
    weaviate: &Weaviate,
    plan: &Plan,
    since: i64,
    options: &Options,
    copied: &mut Copied,
) -> Result<()> {
    let vectors = plan.vector != Copy::Embedded;
    let mut held = HashSet::new();
    let mut after: Option<String> = None;
    let mut pacer = Pacer::new(options.per_second);
    loop {
        let page = weaviate
            .page(&plan.class, after.as_deref(), options.page, false)
            .await?;
        let Some(last) = page.last() else { break };
        after = Some(last.id.clone());
        let mut changed = Vec::new();
        for seen in &page {
            let id = seen.object_id()?;
            held.insert(id);
            let unchanged = plan.stored.get(&id) == Some(&seen.updated);
            if seen.updated.unwrap_or(i64::MAX) >= since && !unchanged {
                let now = weaviate.object(&plan.class, &seen.id, vectors).await?;
                changed.extend(
                    now.map(|object| object.content(plan, options))
                        .transpose()?,
                );
            }
        }
        copied.changed += changed.len() as u64;
        plan.records.store_all(changed).await?;
        pacer.paced(page.len()).await;
    }
    let mut after = None;
    loop {
        let listed = plan
            .records
            .list(&Page {
                filter: None,
                after,
                limit: options.page as u32,
                vectors: false,
            })
            .await?;
        let unseen = listed
            .objects
            .iter()
            .filter(|stored| !held.contains(&stored.id));
        for stored in unseen {
            // Made after the walk passed its id, or gone.
            match weaviate
                .object(&plan.class, &stored.id.to_string(), vectors)
                .await?
            {
                Some(object) => {
                    plan.records
                        .store_all([object.content(plan, options)?])
                        .await?;
                    copied.changed += 1;
                }
                None => {
                    plan.records.delete(stored.id).await?;
                    copied.deleted += 1;
                }
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
        if let Some(per_second) = self.per_second {
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
    /// What is copied from and into, so that a state file is not resumed
    /// against another source or tenant.
    from: String,
    into: String,
    /// When the import began, in milliseconds since the epoch: what
    /// changed after it is copied again by the first catch-up.
    started: i64,
    classes: BTreeMap<String, ClassProgress>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ClassProgress {
    /// The last id stored.
    after: Option<String>,
    done: bool,
    /// When the last catch-up began.
    caught_up_to: Option<i64>,
}

impl Progress {
    fn load(path: &Path, source: &Source, tenant: &Tenant) -> Result<Self> {
        let weaviate_tenant = source
            .tenant
            .as_ref()
            .map(|tenant| format!(" tenant {tenant}"))
            .unwrap_or_default();
        let from = format!("{}{weaviate_tenant}", source.url.trim_end_matches('/'));
        let into = tenant.id().to_string();
        let progress = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice::<Self>(&bytes)
                .wrap_err_with(|| format!("reading the import state in {}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self {
                    from,
                    into,
                    started: Utc::now().timestamp_millis(),
                    classes: BTreeMap::new(),
                })
            }
            Err(error) => {
                return Err(error).wrap_err_with(|| format!("reading {}", path.display()))
            }
        };
        if (&progress.from, &progress.into) != (&from, &into) {
            bail!(
                "{} is the state of an import from {} into {}, not from {from} into {into}",
                path.display(),
                progress.from,
                progress.into
            );
        }
        Ok(progress)
    }

    fn class(&mut self, class: &str) -> &mut ClassProgress {
        self.classes.entry(class.to_string()).or_default()
    }

    /// Written whole and synced under another name, then moved into place,
    /// so a stop part way through never leaves half a file.
    fn save(&self, path: &Path) -> Result<()> {
        let partial = path.with_extension("partial");
        let mut file = std::fs::File::create(&partial)?;
        file.write_all(&serde_json::to_vec_pretty(self)?)?;
        file.sync_all()?;
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
    /// Named vectors, by name.
    #[serde(default)]
    pub vector_config: Option<Map<String, Value>>,
    #[serde(default)]
    pub multi_tenancy_config: Option<Value>,
}

impl Class {
    fn multi_tenant(&self) -> bool {
        self.multi_tenancy_config
            .as_ref()
            .and_then(|config| config.get("enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn named(&self, copy: &Copy, setting: &str) -> Option<&Value> {
        match copy {
            Copy::Named(name) => self.vector_config.as_ref()?.get(name)?.get(setting),
            Copy::Vector | Copy::Embedded => None,
        }
    }
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
    #[serde(default)]
    pub index_range_filters: Option<bool>,
}

/// The collection `class` becomes, embedding as given and measuring as
/// the vector `copy` names does, with what of it means something different
/// in vellev. Whatever vellev cannot hold is refused, all of it at once.
fn collection(
    class: &Class,
    copy: &Copy,
    embedding: Embedding,
    options: &Options,
) -> Result<(Collection, Vec<String>)> {
    let mut problems = Vec::new();
    let mut warnings = Vec::new();
    let name = CollectionName::new(&class.class)
        .map_err(|error| eyre!("class {}: {error}", class.class))?;
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
            let filterable = property.index_filterable.unwrap_or(true)
                || property.index_range_filters.unwrap_or(false);
            let searchable = text && property.index_searchable.unwrap_or(true);
            let tokenization = property.tokenization.as_deref().unwrap_or("word");
            if text && filterable && tokenization != "field" {
                warnings.push(format!(
                    "property {} is filtered by {tokenization} in Weaviate and by whole value in vellev",
                    property.name
                ));
            }
            if searchable && tokenization != "word" {
                warnings.push(format!(
                    "property {} is searched by {tokenization} in Weaviate and by word in vellev",
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
    let index = class
        .named(copy, "vectorIndexConfig")
        .or(class.vector_index_config.as_ref());
    let distance = index
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
/// with the dimensions of the first vector found, or else those `given`,
/// and whether a vector was found. A class whose clients supply vectors,
/// and whose first objects have none, may hold none at all, such as one
/// that keeps a setting: it is given one dimension unless told otherwise.
fn copied_embedding(
    class: &Class,
    copy: &Copy,
    sample: &[Object],
    given: Option<u32>,
) -> Result<(Embedding, bool)> {
    let vectorizer = match class.named(copy, "vectorizer") {
        Some(named) => named
            .as_object()
            .and_then(|vectorizer| vectorizer.keys().next().cloned()),
        None => class.vectorizer.clone(),
    };
    let vectorizer = vectorizer.unwrap_or_else(|| "none".into());
    let found = sample.iter().find_map(|object| object.vector(copy));
    let dimensions = match (found, given) {
        (Some(vector), _) => vector.len() as u32,
        (None, Some(given)) => given,
        (None, None) if vectorizer == "none" => 1,
        (None, None) => bail!(
            "class {} has no vector to copy in its first objects; say how vellev is to embed it, or how many dimensions its vectors have",
            class.class
        ),
    };
    let embedding = Embedding {
        provider: EmbeddingProvider::None,
        model: format!("weaviate:{vectorizer}"),
        dimensions,
        query_prefix: None,
        document_prefix: None,
        properties: Vec::new(),
    };
    Ok((embedding, found.is_some()))
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
    #[serde(default)]
    vectors: Option<HashMap<String, Vec<f32>>>,
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
    fn object_id(&self) -> Result<ObjectId> {
        serde_json::from_value(json!(self.id)).wrap_err_with(|| format!("object id {}", self.id))
    }

    fn vector(&self, copy: &Copy) -> Option<&Vec<f32>> {
        match copy {
            Copy::Embedded => None,
            Copy::Vector => self.vector.as_ref(),
            Copy::Named(name) => self.vectors.as_ref()?.get(name),
        }
    }

    fn content(&self, plan: &Plan, options: &Options) -> Result<(ObjectId, Content)> {
        let id = self.object_id()?;
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
        let vector = self.vector(&plan.vector).cloned();
        if let Some(vector) = vector
            .as_ref()
            .filter(|v| v.len() != plan.dimensions as usize)
        {
            bail!(
                "object {} of {} has a vector of {} dimensions, not {}",
                self.id,
                plan.class.class,
                vector.len(),
                plan.dimensions
            );
        }
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

    /// Weaviate's answer to the request `build` makes, asked again while
    /// it fails in a way that may pass; `None` if it holds nothing there.
    async fn ask(
        &self,
        what: &str,
        build: impl Fn(&reqwest::Client) -> reqwest::RequestBuilder,
    ) -> Result<Option<Value>> {
        let mut wait = Duration::from_millis(500);
        let mut attempt = 1;
        loop {
            let failed = match build(&self.http).send().await {
                Ok(answer) => {
                    let status = answer.status();
                    let text = answer.text().await.unwrap_or_default();
                    if status == reqwest::StatusCode::NOT_FOUND {
                        return Ok(None);
                    }
                    if status.is_success() {
                        return serde_json::from_str(&text)
                            .map(Some)
                            .wrap_err_with(|| format!("reading Weaviate's answer to {what}"));
                    }
                    if !status.is_server_error() {
                        bail!("Weaviate answered {what} with {status}: {text}");
                    }
                    format!("{status}: {text}")
                }
                Err(error) => error.to_string(),
            };
            if attempt == ATTEMPTS {
                bail!("Weaviate failed {what} {ATTEMPTS} times, last with {failed}");
            }
            tokio::time::sleep(wait).await;
            wait *= 2;
            attempt += 1;
        }
    }

    async fn get(&self, path: &str, query: &[(&str, String)]) -> Result<Option<Value>> {
        let url = format!("{}{path}", self.url);
        self.ask(path, |http| http.get(&url).query(query)).await
    }

    /// The classes named, or every class.
    async fn classes(&self, named: &[String]) -> Result<Vec<Class>> {
        let schema = self
            .get("/v1/schema", &[])
            .await?
            .ok_or_else(|| eyre!("Weaviate has no schema"))?;
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

    /// The tenant to name for `class`, if it has multi-tenancy on.
    fn tenant_of(&self, class: &Class) -> Option<(&'static str, String)> {
        self.tenant
            .clone()
            .filter(|_| class.multi_tenant())
            .map(|tenant| ("tenant", tenant))
    }

    /// Up to `limit` objects of `class` after the one with id `after`.
    async fn page(
        &self,
        class: &Class,
        after: Option<&str>,
        limit: usize,
        vectors: bool,
    ) -> Result<Vec<Object>> {
        let mut query = vec![("class", class.class.clone()), ("limit", limit.to_string())];
        query.extend(after.map(|after| ("after", after.to_string())));
        query.extend(vectors.then(|| ("include", "vector".to_string())));
        query.extend(self.tenant_of(class));
        let page = self
            .get("/v1/objects", &query)
            .await?
            .ok_or_else(|| eyre!("Weaviate has no objects of {}", class.class))?;
        serde_json::from_value(page["objects"].clone())
            .wrap_err_with(|| format!("reading a page of {}", class.class))
    }

    /// One object of `class` as it is now, if Weaviate still holds it.
    async fn object(&self, class: &Class, id: &str, vectors: bool) -> Result<Option<Object>> {
        let mut query = Vec::new();
        query.extend(vectors.then(|| ("include", "vector".to_string())));
        query.extend(self.tenant_of(class));
        let path = format!("/v1/objects/{}/{id}", class.class);
        self.get(&path, &query)
            .await?
            .map(|object| {
                serde_json::from_value(object).wrap_err_with(|| format!("reading object {id}"))
            })
            .transpose()
    }

    async fn count(&self, class: &Class) -> Result<u64> {
        let tenant = self
            .tenant_of(class)
            .map(|(_, tenant)| format!("(tenant: {})", json!(tenant)))
            .unwrap_or_default();
        let query = format!(
            "{{ Aggregate {{ {}{tenant} {{ meta {{ count }} }} }} }}",
            class.class
        );
        let url = format!("{}/v1/graphql", self.url);
        let body = json!({ "query": query });
        let answer = self
            .ask("a count", |http| http.post(&url).json(&body))
            .await?
            .unwrap_or_default();
        answer["data"]["Aggregate"][&class.class][0]["meta"]["count"]
            .as_u64()
            .ok_or_else(|| eyre!("Weaviate did not count {}: {answer}", class.class))
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
            vectors: HashMap::new(),
            dimensions: HashMap::new(),
            created_as: None,
            updated_as: None,
            page: 100,
            per_second: None,
            skew: Duration::ZERO,
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
                { "name": "count", "dataType": ["int"], "indexFilterable": false, "indexRangeFilters": true },
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
            &Copy::Vector,
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
            }),
        );
        let refused = collection(
            &class,
            &Copy::Vector,
            embedding(),
            &Options {
                created_as: Some("createdAt".into()),
                ..options()
            },
        )
        .unwrap_err()
        .to_string();
        for reason in [
            "place is [\"geoCoordinates\"]",
            "author is [\"Person\"]",
            "createdAt, to hold a time, is already a property",
            "distance is hamming",
        ] {
            assert!(refused.contains(reason), "{reason} in {refused}");
        }
    }

    #[test]
    fn a_named_vector_measures_as_its_own_index_does_and_other_wording_is_warned_of() {
        let class = class(
            json!([{ "name": "code", "dataType": ["text"], "tokenization": "trigram", "indexFilterable": false }]),
            json!({ "vectorConfig": { "default": {
                "vectorizer": { "text2vec-transformers": {} },
                "vectorIndexConfig": { "distance": "l2-squared" },
            } } }),
        );
        let named = Copy::Named("default".into());
        let (collection, warnings) = collection(&class, &named, embedding(), &options()).unwrap();
        assert_eq!(collection.distance, Distance::L2);
        assert_eq!(
            warnings,
            ["property code is searched by trigram in Weaviate and by word in vellev"]
        );
        let sample: Vec<Object> = serde_json::from_value(json!([
            { "id": id(1), "vectors": {} },
            { "id": id(2), "vectors": { "default": [1.0, 2.0, 3.0] } },
        ]))
        .unwrap();
        let (embedding, found) = copied_embedding(&class, &named, &sample, None).unwrap();
        assert!(found);
        assert_eq!(embedding.dimensions, 3);
        assert_eq!(embedding.model, "weaviate:text2vec-transformers");
    }

    #[test]
    fn a_class_without_vectors_is_copied_with_none_if_its_clients_supply_them() {
        let none = class(json!([]), json!({ "vectorizer": "none" }));
        let (embedding, found) = copied_embedding(&none, &Copy::Vector, &[], None).unwrap();
        assert_eq!((embedding.dimensions, found), (1, false));
        let (embedding, _) = copied_embedding(&none, &Copy::Vector, &[], Some(384)).unwrap();
        assert_eq!(embedding.dimensions, 384);
        let vectorized = class(json!([]), json!({ "vectorizer": "text2vec-transformers" }));
        assert!(copied_embedding(&vectorized, &Copy::Vector, &[], None).is_err());
    }

    fn id(n: u8) -> String {
        format!("00000000-0000-0000-0000-0000000000{n:02}")
    }
}
