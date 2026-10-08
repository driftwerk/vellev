//! Copying a Weaviate instance into a vellev tenant.
//!
//! Each class becomes a collection of the same name, with what Weaviate
//! knows of its properties, distance and BM25 parameters. Its objects are
//! walked with Weaviate's cursor and stored under the same ids, with their
//! vectors, or without them for a class vellev is to embed itself. The
//! last id stored of each class is written to a state file after every
//! page, so an import that stops resumes where it left off.
//!
//! Beside each collection the import keeps a ledger, a collection named
//! `vellev_imported_` and the class: for every id of Weaviate's it has
//! accounted for, when Weaviate says the object last changed, and a hash
//! of the object as the import would store it. An object and its entry
//! are written in one atomic step. The ledger is what tells the import's
//! objects from anyone else's: an object is the import's while the
//! collection holds, under an id the ledger has, exactly what the entry's
//! hash says. The import replaces and deletes its own objects only. It
//! stores a new one only where the collection holds nothing under the id
//! and the ledger has no entry, so what somebody deleted in vellev is not
//! brought back. Whatever else it finds it leaves as it is, and marks the
//! entry as another writer's.
//!
//! Weaviate's cursor is not a snapshot. Every run therefore ends, for each
//! class, with a catch-up: Weaviate's ids are walked beside the ledger's,
//! both in id order, and every object the ledger does not have as Weaviate
//! now does, or that changed within a margin for the two clocks
//! disagreeing, is fetched and settled. Run again, an import is caught up
//! again, so it can follow Weaviate until writes move over; only a pass
//! made once Weaviate takes no more writes is exact.
//!
//! Nothing is deleted while Weaviate is walked. What the ledger has and
//! the walk did not list is only a candidate. Once the walk is over the
//! candidates are counted against what a run may delete, and if they are
//! within it each is asked for twice, a pause apart, with Weaviate ready
//! and the class the one that was imported and its shards loaded, before
//! the import deletes what it stored under that id: [`Copying::remove`]
//! is the one place that does.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use chrono::{DateTime, Utc};
use eyre::{bail, eyre, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tokio::time::Instant;
use vellev_api::{Bounded, Change, Code, Content, Missing, OnlyIf, Page, Stored};
use vellev_client::{Records, Tenant};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider,
    ObjectError, ObjectId, Property,
};

/// How many times an object is settled again because another writer
/// changed it meanwhile, before it is left to that writer.
const ATTEMPTS: u32 = 5;

/// What a class's ledger is called, before the class's name. No class of
/// Weaviate's can have such a name: theirs begin with a capital.
const LEDGER: &str = "vellev_imported_";

/// The collection that says where each class was imported from.
const IMPORTS: &str = "vellev_imports";

/// The version of what the import writes in its own collections. One
/// that reads a later version stops rather than guess at it.
const VERSION: i64 = 1;

/// A Weaviate to copy from.
#[derive(Debug, Clone)]
pub struct Source {
    pub url: String,
    pub key: Option<String>,
    /// The Weaviate tenant to copy, for classes with multi-tenancy on.
    pub tenant: Option<String>,
    /// What this Weaviate is called in the state file and the tenant, for
    /// one whose address changes between runs; its address when not given.
    pub name: Option<String>,
}

impl Source {
    /// What the state file and the tenant know this source as: its
    /// Weaviate tenant too, where what is copied is `tenanted`.
    fn named(&self, tenanted: bool) -> String {
        let weaviate = self
            .name
            .as_deref()
            .unwrap_or(self.url.trim_end_matches('/'));
        match self.tenant.as_ref().filter(|_| tenanted) {
            Some(tenant) => format!("{weaviate} tenant {tenant}"),
            None => weaviate.to_string(),
        }
    }
}

/// How much a run may delete.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Deletes {
    /// Nothing: what Weaviate no longer holds is only counted.
    None,
    /// At most `count` objects of a class, and at most `fraction` of what
    /// its ledger holds, whichever is fewer. A run that finds more gone
    /// deletes none of them.
    UpTo { count: u64, fraction: f64 },
}

/// What a run does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Import,
    /// Change nothing, and compare what the collection holds with what
    /// Weaviate does, object by object: over that many pages begun at
    /// ids picked at random, or over everything.
    Verify {
        pages: Option<usize>,
    },
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
    /// How many objects are read and stored at a time, and how many of the
    /// ledger's entries listed to catch up: at most the node's limit on
    /// results.
    pub page: usize,
    /// The most bytes of objects sent to vellev in one request, near
    /// enough: under the node's limit on a request's body.
    pub batch_bytes: usize,
    /// The most objects asked of Weaviate a second, so as not to overload
    /// it.
    pub per_second: Option<f64>,
    /// How far Weaviate's clock may be behind this one: a catch-up looks
    /// back this much further for changes.
    pub skew: Duration,
    /// Where progress is kept, to resume from.
    pub state: PathBuf,
    pub deletes: Deletes,
    /// How long after Weaviate first answers that it holds no such object
    /// it is asked again, before the object is deleted.
    pub confirm: Duration,
    /// How long a request Weaviate fails, or turns away, is asked again
    /// before the run stops: as long as a restart of it may take.
    pub patience: Duration,
    /// That nothing but an import has written to a collection found
    /// without a ledger: what it lacks is stored and what differs from
    /// Weaviate's is replaced, where otherwise both are left.
    pub sole_writer: bool,
    /// That the class Weaviate has now, made again since the import, is
    /// the one to go on from.
    pub accept_remade: bool,
    /// Pass over an object vellev cannot hold, and report it, where
    /// otherwise the run stops at it.
    pub skip_refused: bool,
    pub mode: Mode,
}

impl Options {
    /// What a run does unless told otherwise, keeping its progress in
    /// `state`.
    pub fn new(state: PathBuf) -> Self {
        Self {
            classes: Vec::new(),
            embeddings: HashMap::new(),
            vectors: HashMap::new(),
            dimensions: HashMap::new(),
            created_as: None,
            updated_as: None,
            page: 1_000,
            batch_bytes: 8 * 1024 * 1024,
            per_second: None,
            skew: Duration::from_secs(300),
            state,
            deletes: Deletes::UpTo {
                count: 1_000,
                fraction: 0.05,
            },
            confirm: Duration::from_secs(5),
            patience: Duration::from_secs(600),
            sole_writer: false,
            accept_remade: false,
            skip_refused: false,
            mode: Mode::Import,
        }
    }
}

/// What an import did, by class.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Report {
    pub classes: BTreeMap<String, Copied>,
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Everything that is the import's is as Weaviate has it.
    Settled,
    /// Weaviate was written to behind the catch-up: run again.
    Behind,
    /// Something is left that another run will not put right by itself.
    Unsettled,
}

impl Report {
    /// How the run ended; with `strict`, anything of another writer's in
    /// a collection leaves it unsettled too.
    pub fn outcome(&self, strict: bool) -> Outcome {
        let classes = || self.classes.values();
        if classes().any(|copied| copied.unsettled() || (strict && !copied.untouched())) {
            Outcome::Unsettled
        } else if classes().any(|copied| !copied.caught_up()) {
            Outcome::Behind
        } else {
            Outcome::Settled
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Copied {
    /// Objects this run stored that the collection did not hold.
    pub stored: u64,
    /// Objects this run stored again, changed in Weaviate.
    pub changed: u64,
    /// Objects this run deleted, gone from Weaviate.
    pub deleted: u64,
    /// Objects of Weaviate's that the collection does not hold as the
    /// import would have them, by the ledger's mark: another writer's,
    /// or deleted by one. The same on every run while they stay so.
    pub kept: u64,
    /// Objects gone from Weaviate that this run left in the collection
    /// because they were not as the import had stored them.
    pub left: u64,
    /// Objects gone from Weaviate that this run did not delete, though
    /// they may be the import's: it was told to delete none, or they are
    /// more than `budget`, what it may delete.
    pub pending: u64,
    pub budget: Option<u64>,
    /// Objects vellev cannot hold, passed over: why, by id.
    pub refused: BTreeMap<String, String>,
    /// Objects changed in Weaviate and too large to replace: the step
    /// that replaces one safely says twice over what it holds.
    pub too_large: Vec<String>,
    /// How many objects Weaviate held at the end, how many of Weaviate's
    /// the ledger had accounted for, and how many the collection held.
    pub weaviate: u64,
    pub imported: u64,
    pub vellev: u64,
    /// What comparing the collection with Weaviate found, if asked for.
    pub verified: Option<Verified>,
    /// What was mapped but means something different in vellev.
    pub warnings: Vec<String>,
}

/// What comparing a collection with Weaviate object by object found.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Verified {
    /// Objects of Weaviate's looked at.
    pub looked: u64,
    /// Those the collection holds exactly as Weaviate does.
    pub same: u64,
    /// Those it holds otherwise, or not at all, though the ledger takes
    /// them for the import's.
    pub different: u64,
    pub missing: u64,
    /// Those the ledger marks as another writer's, and those it has no
    /// entry for.
    pub kept: u64,
    pub unaccounted: u64,
    /// Some of the ids that differ or are missing.
    pub examples: Vec<String>,
}

impl Copied {
    /// Whether the import has accounted for exactly the objects Weaviate
    /// holds, those it passed over or has yet to delete aside. It has
    /// not if Weaviate was written to behind the catch-up; another run
    /// catches up.
    pub fn caught_up(&self) -> bool {
        self.weaviate + self.pending == self.imported + self.refused.len() as u64
    }

    /// Whether something is left that running again does not put right:
    /// more gone than a run may delete, objects vellev cannot hold, or
    /// objects found different by a comparison.
    pub fn unsettled(&self) -> bool {
        let verified = self.verified.as_ref();
        self.budget.is_some()
            || !self.refused.is_empty()
            || !self.too_large.is_empty()
            || verified.is_some_and(|verified| verified.different + verified.missing > 0)
    }

    /// Whether the collection holds as many objects as the import has
    /// accounted for, and none of them is another writer's.
    pub fn untouched(&self) -> bool {
        self.kept == 0 && self.left == 0 && self.vellev == self.imported
    }

    fn count(&mut self, id: ObjectId, settled: Settled) {
        match settled {
            Settled::Stored => self.stored += 1,
            Settled::Replaced => self.changed += 1,
            Settled::Deleted => self.deleted += 1,
            Settled::Left => self.left += 1,
            Settled::TooLarge => self.too_large.push(id.to_string()),
            Settled::Refused(why) => {
                self.refused.insert(id.to_string(), why);
            }
            Settled::Foreign | Settled::Same => {}
        }
    }
}

/// What settling an object did.
#[derive(Debug, Clone, PartialEq)]
enum Settled {
    Stored,
    Replaced,
    Deleted,
    /// It is another writer's, or was deleted by one: the ledger says so.
    Foreign,
    /// Gone from Weaviate and another writer's: left, and forgotten.
    Left,
    /// The collection already held what Weaviate does, or neither held it.
    Same,
    TooLarge,
    Refused(String),
}

/// One class to copy, as planned.
struct Plan {
    class: Class,
    records: Records,
    ledger: Records,
    ledger_name: CollectionName,
    /// The collection as vellev has it, to tell how it reads an object.
    schema: Collection,
    vector: Copy,
    binding: Binding,
}

impl Plan {
    fn vectors(&self) -> bool {
        self.vector != Copy::Embedded
    }

    /// The object as vellev would hold and return it, and the hash of
    /// that: of its properties, and of its vector unless vellev makes it.
    fn held(
        &self,
        id: ObjectId,
        properties: &Map<String, Value>,
        vector: Option<Vec<f32>>,
    ) -> Result<(Content, String), ObjectError> {
        let vector = vector.filter(|_| self.vectors());
        let object = vellev_core::Object::from_json(&self.schema, id, properties, vector)?;
        let properties = object.properties_json();
        let mut hash = Sha256::new();
        let ordered: BTreeMap<&String, &Value> = properties.iter().collect();
        hash.update(serde_json::to_vec(&ordered).expect("JSON values are written as JSON"));
        for part in object.vector.iter().flatten() {
            hash.update(part.to_le_bytes());
        }
        let hash = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&hash.finalize()[..16]);
        let mut content = Content::new(properties);
        content.vector = object.vector;
        Ok((content, hash))
    }

    /// The hash of an object the collection holds; none if it is not one
    /// the collection, as this run knows it, could have been sent.
    fn hash(&self, stored: &Stored) -> Option<String> {
        self.held(stored.id, &stored.properties, stored.vector.clone())
            .ok()
            .map(|(_, hash)| hash)
    }
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

/// An object of Weaviate's as the collection is to hold it.
#[derive(Debug, Clone)]
struct Item {
    id: ObjectId,
    /// When Weaviate says it last changed.
    updated: Option<i64>,
    content: Content,
    hash: String,
}

/// What the ledger says of one id.
#[derive(Debug, Clone, PartialEq)]
struct Entry {
    id: ObjectId,
    updated: Option<i64>,
    hash: String,
    /// That the collection did not hold the object as the import would
    /// have it when the import last looked.
    foreign: bool,
}

impl Entry {
    const PROPERTIES: [(&'static str, DataType); 4] = [
        ("updated", DataType::Int),
        ("hash", DataType::Text),
        ("foreign", DataType::Bool),
        ("version", DataType::Int),
    ];

    fn read(stored: Stored) -> Result<Self> {
        let property = |name: &str| stored.properties.get(name);
        later(property("version"), "the ledger")?;
        let hash = property("hash").and_then(Value::as_str);
        Ok(Self {
            id: stored.id,
            updated: property("updated").and_then(Value::as_i64),
            hash: hash
                .ok_or_else(|| eyre!("the ledger's entry for {} has no hash", stored.id))?
                .to_string(),
            foreign: property("foreign") == Some(&json!(true)),
        })
    }

    /// As the ledger holds it. What is as usual is left out, and costs
    /// nothing: the first version, and an object that is the import's.
    fn properties(item: &Item, foreign: bool) -> Map<String, Value> {
        let updated = item.updated.map(|updated| ("updated", json!(updated)));
        let foreign = foreign.then_some(("foreign", json!(true)));
        let version = (VERSION > 1).then(|| ("version", json!(VERSION)));
        [updated, Some(("hash", json!(item.hash))), foreign, version]
            .into_iter()
            .flatten()
            .map(|(name, value)| (name.to_string(), value))
            .collect()
    }
}

/// Refuse to read what a later release wrote in `what`.
fn later(version: Option<&Value>, what: &str) -> Result<()> {
    match version.and_then(Value::as_i64) {
        Some(version) if version > VERSION => bail!(
            "{what} was written by a later release of the import, as version {version}; this one reads up to {VERSION}"
        ),
        _ => Ok(()),
    }
}

/// Copy the classes `options` names from `source` into `tenant`, and
/// catch up with what changed in Weaviate meanwhile; or, asked only to
/// verify, compare the two.
pub async fn weaviate(source: &Source, tenant: &Tenant, options: &Options) -> Result<Report> {
    if options.page == 0 {
        bail!("a page holds at least one object");
    }
    if options.batch_bytes == 0 {
        bail!("a request holds at least one byte");
    }
    if options
        .per_second
        .is_some_and(|rate| !(rate.is_finite() && rate > 0.0))
    {
        bail!("the rate of copying must be a number more than nothing");
    }
    if let Deletes::UpTo { fraction, .. } = options.deletes {
        if !(0.0..=1.0).contains(&fraction) {
            bail!("the part of a class a run may delete is between nothing and all of it");
        }
    }
    let weaviate = Weaviate::new(source, options.patience)?;
    let classes = weaviate.classes(&options.classes).await?;
    let named = options.embeddings.keys().chain(options.vectors.keys());
    for named in named.chain(options.dimensions.keys()) {
        if !classes.iter().any(|class| &class.class == named) {
            bail!("{named} is not a class being imported");
        }
    }
    let mut report = Report::default();
    if let Mode::Verify { pages } = options.mode {
        for class in classes {
            let (plan, _, _) = prepare(&weaviate, class, tenant, source, options).await?;
            let copying = Copying::new(&weaviate, tenant, &plan, options);
            let copied = Copied {
                verified: Some(copying.verify(pages).await?),
                weaviate: weaviate.count(&plan.class).await?,
                imported: plan.ledger.count(None).await?,
                vellev: plan.records.count(None).await?,
                ..Copied::default()
            };
            report.classes.insert(plan.class.class.clone(), copied);
        }
        return Ok(report);
    }

    let _lock = Progress::lock(&options.state)?;
    let mut state = Progress::load(&options.state, source, tenant)?;
    tenant.create().await?;
    let mut plans = Vec::new();
    for class in classes {
        let (plan, warnings, begins) = prepare(&weaviate, class, tenant, source, options).await?;
        if begins {
            // Whatever the state file says of the class was said of a
            // collection that is no longer there, or of an import that
            // kept no ledger.
            state.classes.remove(&plan.class.class);
        }
        let copied = Copied {
            warnings,
            ..Copied::default()
        };
        report.classes.insert(plan.class.class.clone(), copied);
        plans.push(plan);
    }
    state.save(&options.state)?;

    for plan in &mut plans {
        let copied = report.classes.get_mut(&plan.class.class).expect("planned");
        let copying = Copying::new(&weaviate, tenant, plan, options);
        copying.walk(&mut state, copied).await?;
        if plan.binding.adopting {
            // Every id Weaviate held has its entry: one without from now
            // on is new to Weaviate.
            plan.binding.adopting = false;
            tenant
                .apply(vec![plan.binding.change(&plan.schema.name)])
                .await?;
        }
    }

    for plan in &plans {
        let name = &plan.class.class;
        let copied = report.classes.get_mut(name).expect("planned");
        let copying = Copying::new(&weaviate, tenant, plan, options);
        let begun = Utc::now().timestamp_millis();
        let since = state.class(name).caught_up_to.unwrap_or(state.started)
            - options.skew.as_millis() as i64;
        if copying.catch_up(since, copied).await? {
            state.class(name).caught_up_to = Some(begun);
            state.save(&options.state)?;
        }
        copied.weaviate = weaviate.count(&plan.class).await?;
        copied.imported = plan.ledger.count(None).await?;
        copied.vellev = plan.records.count(None).await?;
    }
    Ok(report)
}

/// End the import of `classes` into `tenant`, or of every class imported
/// into it: drop each ledger, and note that the collection is no longer
/// imported, so that no later run takes it up. Answers with the classes
/// ended.
pub async fn finalise(tenant: &Tenant, classes: &[String]) -> Result<Vec<String>> {
    let imports = tenant.collection(&imports());
    let mut ended = Vec::new();
    // A page small enough for any node: there is an object to a class.
    let mut page = Page::new(10);
    loop {
        let listed = match imports.list(&page).await {
            Ok(listed) => listed,
            Err(error) if error.is_not_found() => break,
            Err(error) => return Err(error.into()),
        };
        for stored in &listed.objects {
            let mut binding = Binding::read(stored)?;
            let Some(class) = binding.class.clone() else {
                continue;
            };
            if !(classes.is_empty() || classes.contains(&class)) {
                continue;
            }
            let name = CollectionName::new(&class).map_err(|error| eyre!("{error}"))?;
            let ledger = ledger_of(&name)?;
            binding.finalised = true;
            binding.adopting = false;
            let mut changes = vec![binding.change(&name)];
            if tenant.collection(&ledger).schema().await?.is_some() {
                changes.push(Change::DropCollection { collection: ledger });
            }
            tenant.apply(changes).await?;
            ended.push(class);
        }
        match listed.next {
            Some(next) => page.after = Some(next),
            None => break,
        }
    }
    if let Some(unknown) = classes.iter().find(|class| !ended.contains(class)) {
        bail!("{unknown} was not imported into this tenant");
    }
    Ok(ended)
}

fn imports() -> CollectionName {
    CollectionName::new(IMPORTS).expect("a valid name")
}

fn ledger_of(collection: &CollectionName) -> Result<CollectionName> {
    CollectionName::new(format!("{LEDGER}{collection}")).map_err(|_| {
        eyre!("class {collection}: its name is too long for a ledger to be kept beside it")
    })
}

fn object_id(id: &str) -> Result<ObjectId> {
    serde_json::from_value(json!(id)).wrap_err_with(|| format!("object id {id}"))
}

/// How to copy `class`, what of it means something different in vellev,
/// and whether its copying begins from the start. Unless the run only
/// verifies, the collection is made if there is none, or else given the
/// properties the class has gained, and its ledger and the note of where
/// it came from are made beside it.
async fn prepare(
    weaviate: &Weaviate,
    class: Class,
    tenant: &Tenant,
    source: &Source,
    options: &Options,
) -> Result<(Plan, Vec<String>, bool)> {
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
    let name = CollectionName::new(&class.class)
        .map_err(|error| eyre!("class {}: {error}", class.class))?;
    let ledger = ledger_of(&name)?;
    let records = tenant.collection(&name);
    let stored = records.schema().await?;
    let mut vectorless = None;
    let embedding = match options.embeddings.get(&class.class) {
        Some(embedding) => embedding.clone(),
        None => {
            let sample = weaviate.page(&class, None, options.page, true).await?;
            // A collection that is there says how long its vectors are,
            // where the first objects no longer do.
            let given = options.dimensions.get(&class.class).copied();
            let given = given.or(stored.as_ref().map(|stored| stored.embedding.dimensions));
            let (embedding, found) = copied_embedding(&class, &vector, &sample, given)?;
            vectorless = (!found && stored.is_none()).then(|| {
                format!(
                    "no vector was found in its first objects; it holds vectors of {} dimensions",
                    embedding.dimensions
                )
            });
            embedding
        }
    };
    let (wanted, mut warnings) = collection(&class, &vector, embedding, options)?;
    warnings.extend(vectorless);
    let bound = tenant
        .collection(&imports())
        .get(Binding::id(&name))
        .await?;
    let bound = bound.as_ref().map(Binding::read).transpose()?;
    let ledgered = tenant.collection(&ledger).schema().await?;
    let identity = weaviate.identity(&class).await?;
    let mut binding = Binding {
        class: Some(class.class.clone()),
        source: source.named(class.multi_tenant()),
        created_as: options.created_as.clone(),
        updated_as: options.updated_as.clone(),
        shards: identity.clone(),
        adopting: false,
        finalised: false,
    };
    let mut changes = Vec::new();
    let mut begins = stored.is_none();
    let schema = match stored {
        None => {
            // A ledger left by a collection since dropped says nothing of
            // the one made now.
            if ledgered.is_some() {
                changes.push(Change::DropCollection {
                    collection: ledger.clone(),
                });
            }
            changes.push(Change::CreateCollection {
                schema: Box::new(wanted.clone()),
            });
            changes.push(Change::CreateCollection {
                schema: Box::new(bookkeeping(ledger.clone(), &Entry::PROPERTIES)),
            });
            wanted
        }
        Some(mut stored) => {
            match &bound {
                Some(bound) => {
                    bound.agrees(&binding, &name, options.accept_remade)?;
                    // A collection with no ledger, or whose ledger began
                    // after it did, is being taken up; one whose ledger
                    // is gone is taken up from the start.
                    binding.adopting = bound.adopting || ledgered.is_none();
                    begins = ledgered.is_none();
                    // An identity is noted the first time it is known,
                    // and kept from then on.
                    if !options.accept_remade {
                        binding.shards = bound.shards.clone().or(identity);
                    }
                }
                None => {
                    binding.adopting = true;
                    begins = true;
                }
            }
            let (gained, differing) = reconcile(&stored, &wanted)
                .wrap_err_with(|| format!("class {} no longer fits its collection", class.class))?;
            warnings.extend(differing);
            for property in gained {
                stored.add_property(property.clone())?;
                changes.push(Change::AddProperty {
                    collection: name.clone(),
                    property,
                });
            }
            changes.extend(grown(
                ledgered,
                bookkeeping(ledger.clone(), &Entry::PROPERTIES),
            ));
            stored
        }
    };
    let noted = tenant.collection(&imports()).schema().await?;
    changes.extend(grown(noted, bookkeeping(imports(), &Binding::PROPERTIES)));
    changes.push(binding.change(&name));
    if options.mode == Mode::Import {
        tenant
            .apply(changes)
            .await
            .wrap_err_with(|| format!("preparing collection {name}"))?;
    } else if begins {
        bail!(
            "class {} has not been imported: there is nothing to compare",
            class.class
        );
    }
    let plan = Plan {
        records,
        ledger: tenant.collection(&ledger),
        ledger_name: ledger,
        schema,
        class,
        vector,
        binding,
    };
    Ok((plan, warnings, begins))
}

/// A collection of the import's own notes: nothing in it is searched,
/// filtered or given a vector.
fn bookkeeping(name: CollectionName, properties: &[(&str, DataType)]) -> Collection {
    Collection {
        name,
        properties: properties
            .iter()
            .map(|(name, data_type)| Property {
                filterable: false,
                ..Property::new(name, *data_type)
            })
            .collect(),
        embedding: Embedding {
            provider: EmbeddingProvider::None,
            model: "vellev:import".into(),
            dimensions: 1,
            query_prefix: None,
            document_prefix: None,
            properties: Vec::new(),
        },
        distance: Distance::Cosine,
        bm25: Bm25::default(),
        access_rule: None,
    }
}

/// The changes that make `wanted`, one of the import's own collections,
/// or give the one `stored` the properties it lacks, so that a release
/// which keeps more in it goes on from what an earlier one made.
fn grown(stored: Option<Collection>, wanted: Collection) -> Vec<Change> {
    let Some(stored) = stored else {
        return vec![Change::CreateCollection {
            schema: Box::new(wanted),
        }];
    };
    let lacked = |property: &Property| stored.property(&property.name).is_none();
    let add = |property: Property| Change::AddProperty {
        collection: wanted.name.clone(),
        property,
    };
    wanted
        .properties
        .iter()
        .filter(|p| lacked(p))
        .cloned()
        .map(add)
        .collect()
}

/// Where a class's collection was imported from and how: kept in the
/// tenant, so that its ledger is never read against another Weaviate or
/// another class of the same name, nor its objects stored two ways.
#[derive(Debug, Clone, PartialEq)]
struct Binding {
    class: Option<String>,
    source: String,
    created_as: Option<String>,
    updated_as: Option<String>,
    /// The names Weaviate gave the class's shards when it made it: a
    /// class made again has others. None for a class with tenants, whose
    /// shards are named after them.
    shards: Option<String>,
    /// That the collection was there before its ledger, and not every id
    /// of Weaviate's has its entry yet.
    adopting: bool,
    /// That the import of the collection has ended.
    finalised: bool,
}

impl Binding {
    const PROPERTIES: [(&'static str, DataType); 8] = [
        ("class", DataType::Text),
        ("source", DataType::Text),
        ("createdAs", DataType::Text),
        ("updatedAs", DataType::Text),
        ("shards", DataType::Text),
        ("adopting", DataType::Bool),
        ("finalised", DataType::Bool),
        ("version", DataType::Int),
    ];

    /// Where a collection's binding is kept: under an id made of its name,
    /// so that there is one.
    fn id(collection: &CollectionName) -> ObjectId {
        let hash = Sha256::digest(collection.as_str());
        ObjectId(uuid::Uuid::from_slice(&hash[..16]).expect("sixteen bytes"))
    }

    /// The change that notes it for `collection`.
    fn change(&self, collection: &CollectionName) -> Change {
        let text = |value: &Option<String>| value.as_ref().map(|value| json!(value));
        let flag = |value: bool| value.then_some(json!(true));
        let values = [
            text(&self.class),
            Some(json!(self.source)),
            text(&self.created_as),
            text(&self.updated_as),
            text(&self.shards),
            flag(self.adopting),
            flag(self.finalised),
            Some(json!(VERSION)),
        ];
        let properties = Self::PROPERTIES.iter().zip(values);
        Change::Upsert {
            collection: imports(),
            id: Self::id(collection),
            properties: properties
                .filter_map(|((name, _), value)| Some((name.to_string(), value?)))
                .collect(),
            vector: None,
        }
    }

    fn read(stored: &Stored) -> Result<Self> {
        let property = |name: &str| stored.properties.get(name);
        later(
            property("version"),
            "the note of where a collection was imported from",
        )?;
        let text = |name: &str| property(name).and_then(Value::as_str).map(String::from);
        let flag = |name: &str| property(name) == Some(&json!(true));
        Ok(Self {
            class: text("class"),
            source: text("source").unwrap_or_default(),
            created_as: text("createdAs"),
            updated_as: text("updatedAs"),
            shards: text("shards"),
            adopting: flag("adopting"),
            finalised: flag("finalised"),
        })
    }

    /// Refuse a run that would copy into `collection` from somewhere
    /// else, or otherwise, than `self` says it was, or after its import
    /// ended.
    fn agrees(&self, now: &Self, collection: &CollectionName, remade: bool) -> Result<()> {
        if self.finalised {
            bail!(
                "the import of collection {collection} was ended: its ledger is gone, and nothing more is imported into it"
            );
        }
        if self.source != now.source {
            bail!(
                "collection {collection} was imported from {}, not from {}; if this is that Weaviate at another address, name it as it was named then",
                self.source,
                now.source
            );
        }
        if (&self.created_as, &self.updated_as) != (&now.created_as, &now.updated_as) {
            bail!(
                "collection {collection} was imported with Weaviate's creation time as {:?} and its time of change as {:?}; an import goes on as it began",
                self.created_as,
                self.updated_as
            );
        }
        match (&self.shards, &now.shards) {
            (Some(was), Some(is)) if was != is && !remade => bail!(
                "Weaviate's class {collection} is not the one that was imported: it has been made again since, its shards {is} where they were {was}. Nothing was changed. If the class as it is now is the one to go on from, say that it was made again"
            ),
            _ => Ok(()),
        }
    }
}

/// What `wanted`, the collection a class would become now, asks of
/// `stored`, the collection it became: the properties it has gained, to
/// add, and what differs that copying goes on despite. What would make
/// the copy wrong is refused, all of it at once: vectors of another size,
/// model or distance, and a property of another type. A property the
/// class no longer has stays in the collection: vellev drops none, and
/// objects stored before still have their values.
fn reconcile(stored: &Collection, wanted: &Collection) -> Result<(Vec<Property>, Vec<String>)> {
    let mut problems = Vec::new();
    let mut warnings = Vec::new();
    let mut gained = Vec::new();
    let (has, asks) = (&stored.embedding, &wanted.embedding);
    if has.dimensions != asks.dimensions {
        problems.push(format!(
            "its vectors have {} dimensions and the collection's {}",
            asks.dimensions, has.dimensions
        ));
    }
    if (&has.provider, &has.model) != (&asks.provider, &asks.model) {
        problems.push(format!(
            "its vectors are made by {} and the collection's by {}",
            asks.model, has.model
        ));
    } else if (has.dimensions == asks.dimensions) && has != asks {
        problems.push("its text is embedded otherwise than the collection's".into());
    }
    if stored.distance != wanted.distance {
        problems.push(format!(
            "its distance is {:?} and the collection's {:?}",
            wanted.distance, stored.distance
        ));
    }
    for property in &wanted.properties {
        match stored.property(&property.name) {
            None => gained.push(property.clone()),
            Some(held) if held.data_type != property.data_type => problems.push(format!(
                "property {} is {} and the collection's {}",
                property.name, property.data_type, held.data_type
            )),
            Some(held) if held != property => warnings.push(format!(
                "property {} is indexed otherwise in the collection than the class now asks; the collection's stays as it is",
                property.name
            )),
            Some(_) => {}
        }
    }
    if stored.bm25 != wanted.bm25 {
        warnings.push(format!(
            "its BM25 is now k1 {} and b {}; the collection's stays k1 {} and b {}",
            wanted.bm25.k1, wanted.bm25.b, stored.bm25.k1, stored.bm25.b
        ));
    }
    if !problems.is_empty() {
        bail!("{}", problems.join("; "));
    }
    Ok((gained, warnings))
}

/// Why vellev would not make a write, as far as settling goes.
#[derive(Debug, Clone, PartialEq)]
enum Refusal {
    /// The object, or its entry, is not as the write took it to be, there
    /// or not; or so the node says, which answers the same of a tenant it
    /// is busy with. Whether another writer got there first shows when
    /// the object is read again.
    Overtaken,
    /// The request is larger than the node takes.
    TooLarge,
    /// The node cannot hold what it was sent.
    Unfit(String),
}

fn refusal(error: &vellev_client::Error) -> Option<Refusal> {
    let large =
        matches!(error, vellev_client::Error::Status { status, .. } if status.as_u16() == 413);
    if large || error.limit_max(Bounded::BodyBytes).is_some() {
        return Some(Refusal::TooLarge);
    }
    match error.code()? {
        Code::Conflict => Some(Refusal::Overtaken),
        Code::NotFound if error.missing() == Some(Missing::Object) => Some(Refusal::Overtaken),
        Code::Invalid => Some(Refusal::Unfit(error.to_string())),
        _ => None,
    }
}

/// One object's part of a request: what settles it.
struct Step {
    item: Item,
    changes: Vec<Change>,
    settled: Settled,
}

impl Step {
    /// How many bytes a request grows by to carry it, near enough.
    fn size(&self) -> usize {
        let size = |change| serde_json::to_vec(change).map_or(0, |written| written.len());
        self.changes.iter().map(size).sum()
    }
}

/// `steps` in runs of at most `most`, each of at most `bytes` bytes, near
/// enough; a step larger than that is a run of its own.
fn batches(steps: Vec<Step>, most: usize, bytes: usize) -> Vec<Vec<Step>> {
    let mut batches: Vec<Vec<Step>> = Vec::new();
    let mut held = 0;
    for step in steps {
        let size = step.size();
        match batches.last_mut() {
            Some(batch) if batch.len() < most && held + size <= bytes => {
                held += size;
                batch.push(step);
            }
            _ => {
                held = size;
                batches.push(vec![step]);
            }
        }
    }
    batches
}

/// What the ledger has that a walk of Weaviate did not list: gone from
/// it, unless Weaviate says otherwise when asked for each. No more are
/// kept than a run may delete, and one over, to tell.
struct Gone {
    entries: Vec<Entry>,
    count: u64,
    /// How many of them the ledger marks as another writer's.
    foreign: u64,
    budget: u64,
}

impl Gone {
    fn push(&mut self, entry: Entry) {
        self.count += 1;
        self.foreign += u64::from(entry.foreign);
        if self.count <= self.budget {
            self.entries.push(entry);
        }
    }
}

/// One class being copied.
struct Copying<'a> {
    weaviate: &'a Weaviate,
    tenant: &'a Tenant,
    plan: &'a Plan,
    options: &'a Options,
}

impl<'a> Copying<'a> {
    fn new(
        weaviate: &'a Weaviate,
        tenant: &'a Tenant,
        plan: &'a Plan,
        options: &'a Options,
    ) -> Self {
        Self {
            weaviate,
            tenant,
            plan,
            options,
        }
    }

    fn data(&self) -> CollectionName {
        self.plan.schema.name.clone()
    }

    /// The object as the collection is to hold it. One it cannot hold
    /// stops the run, or is passed over and reported.
    fn item(&self, object: &Object, copied: &mut Copied) -> Result<Option<Item>> {
        match object.item(self.plan, self.options)? {
            Ok(item) => Ok(Some(item)),
            Err(why) if self.options.skip_refused => {
                copied.refused.insert(object.id.clone(), why);
                Ok(None)
            }
            Err(why) => bail!("object {} of {}: {why}", object.id, self.plan.class.class),
        }
    }

    fn items(&self, page: &[Object], copied: &mut Copied) -> Result<Vec<Item>> {
        let items = page.iter().map(|object| self.item(object, copied));
        Ok(items
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Walk the class from where the state file says the walk had come,
    /// storing each page, or taking it up if the collection was there
    /// before its ledger.
    async fn walk(&self, state: &mut Progress, copied: &mut Copied) -> Result<()> {
        let (plan, options) = (self.plan, self.options);
        let name = &plan.class.class;
        let mut pacer = Pacer::new(options.per_second);
        let mut entries = None;
        while !state.class(name).done {
            let after = state.class(name).after.clone();
            let page = self
                .weaviate
                .page(&plan.class, after.as_deref(), options.page, plan.vectors())
                .await?;
            let Some(last) = page.last() else {
                state.class(name).done = true;
                break;
            };
            let last = last.id.clone();
            let items = self.items(&page, copied)?;
            if plan.binding.adopting {
                let from = after.as_deref().map(object_id).transpose()?;
                let entries =
                    entries.get_or_insert_with(|| Entries::new(&plan.ledger, options.page, from));
                self.adopt(from, object_id(&last)?, items, entries, copied)
                    .await?;
            } else {
                self.store(items, copied).await?;
            }
            state.class(name).after = Some(last);
            state.save(&options.state)?;
            pacer.paced(page.len()).await;
        }
        state.save(&options.state)
    }

    /// Note in the ledger what Weaviate holds under the item's id, and
    /// whether the collection holds it otherwise: in place of the entry
    /// `known` to be there, or else only if there is still none, the
    /// step failing otherwise.
    fn note(&self, item: &Item, known: bool, foreign: bool) -> Change {
        let collection = self.plan.ledger_name.clone();
        let properties = Entry::properties(item, foreign);
        let (id, vector) = (item.id, None);
        match known {
            true => Change::Upsert {
                collection,
                id,
                properties,
                vector,
            },
            false => Change::Insert {
                collection,
                id,
                properties,
                vector,
            },
        }
    }

    /// The note, unless the entry already says as much.
    fn noting(&self, item: &Item, entry: Option<&Entry>, foreign: bool) -> Vec<Change> {
        let said = entry.is_some_and(|entry| {
            (&entry.hash, entry.updated, entry.foreign) == (&item.hash, item.updated, foreign)
        });
        match said {
            true => Vec::new(),
            false => vec![self.note(item, entry.is_some(), foreign)],
        }
    }

    /// Store the object and its entry, if the collection holds nothing
    /// under its id and the ledger no entry; otherwise the step fails.
    fn fresh(&self, item: &Item) -> Vec<Change> {
        let object = Change::Insert {
            collection: self.data(),
            id: item.id,
            properties: item.content.properties.clone(),
            vector: item.content.vector.clone(),
        };
        vec![object, self.note(item, false, false)]
    }

    /// Store the item in place of `stored`, if that is still as it was
    /// read, and note it.
    fn replace(&self, stored: &Stored, item: &Item, known: bool) -> Vec<Change> {
        let mut changes = self.unless_changed(stored);
        changes.push(Change::Upsert {
            collection: self.data(),
            id: item.id,
            properties: item.content.properties.clone(),
            vector: item.content.vector.clone(),
        });
        changes.push(self.note(item, known, false));
        changes
    }

    fn forget(&self, id: ObjectId) -> Change {
        Change::Delete {
            collection: self.plan.ledger_name.clone(),
            id,
        }
    }

    /// Changes that alter nothing and fail, and with them the step they
    /// are part of, unless every property of the object is still as
    /// `stored` has it, those without a value included.
    fn unless_changed(&self, stored: &Stored) -> Vec<Change> {
        let guard = |property: &Property| Change::Patch {
            collection: self.data(),
            id: stored.id,
            properties: Map::new(),
            vector: None,
            only_if: Some(OnlyIf {
                property: property.name.clone(),
                equals: stored
                    .properties
                    .get(&property.name)
                    .cloned()
                    .unwrap_or(Value::Null),
            }),
        };
        self.plan.schema.properties.iter().map(guard).collect()
    }

    async fn entry(&self, id: ObjectId) -> Result<Option<Entry>> {
        let stored = self.plan.ledger.get(id).await?;
        stored.map(Entry::read).transpose()
    }

    /// Whether settling the item takes reading what the collection holds
    /// under its id: not while the ledger says the import stored exactly
    /// this, and it is nobody else's.
    fn reads(item: &Item, entry: Option<&Entry>) -> bool {
        !entry.is_some_and(|entry| entry.hash == item.hash && !entry.foreign)
    }

    /// What brings the collection to hold what Weaviate does under the
    /// item's id, if what it holds, `stored`, is the import's to change,
    /// and the ledger to say what Weaviate holds. While a collection that
    /// had no ledger is `adopted`, what it lacks is not stored and what
    /// differs is not replaced, unless nothing else has written to it.
    fn settling(
        &self,
        item: &Item,
        entry: Option<&Entry>,
        stored: Option<&Stored>,
        adopted: bool,
    ) -> (Vec<Change>, Settled) {
        if !Self::reads(item, entry) {
            return (self.noting(item, entry, false), Settled::Same);
        }
        let held = stored.and_then(|stored| self.plan.hash(stored));
        let ours = entry.is_some_and(|entry| held.as_deref() == Some(entry.hash.as_str()));
        let sole = adopted && self.options.sole_writer;
        match (entry, stored) {
            (None, None) if adopted && !sole => {
                (vec![self.note(item, false, true)], Settled::Foreign)
            }
            (None, None) => (self.fresh(item), Settled::Stored),
            (_, Some(_)) if held.as_deref() == Some(item.hash.as_str()) => {
                (self.noting(item, entry, false), Settled::Same)
            }
            (Some(_), Some(stored)) if ours => {
                (self.replace(stored, item, true), Settled::Replaced)
            }
            (None, Some(stored)) if sole => (self.replace(stored, item, false), Settled::Replaced),
            _ => (self.noting(item, entry, true), Settled::Foreign),
        }
    }

    /// Delete what the import stored under an id Weaviate no longer has,
    /// if the collection still holds it so, and forget the id. The one
    /// place a delete of an object is made.
    fn removing(&self, entry: &Entry, stored: Option<&Stored>) -> (Vec<Change>, Settled) {
        let held = stored.and_then(|stored| self.plan.hash(stored));
        match stored {
            Some(stored) if held.as_deref() == Some(entry.hash.as_str()) => {
                let mut changes = self.unless_changed(stored);
                changes.push(Change::Delete {
                    collection: self.data(),
                    id: entry.id,
                });
                changes.push(self.forget(entry.id));
                (changes, Settled::Deleted)
            }
            Some(_) => (vec![self.forget(entry.id)], Settled::Left),
            None => (vec![self.forget(entry.id)], Settled::Same),
        }
    }

    /// Settle one object: the item Weaviate holds now, or, with none, an
    /// id Weaviate has been found not to hold. A step refused because
    /// the object or its entry is not as it was read is made again from
    /// a new reading, and after enough of that the object is left to
    /// whoever keeps writing it. Refused though both read as before, the
    /// refusal is of something else, and stops the run.
    async fn settle(
        &self,
        id: ObjectId,
        now: Option<&Item>,
        mut entry: Option<Entry>,
        adopted: bool,
    ) -> Result<Settled> {
        let mut before = None;
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                entry = self.entry(id).await?;
            }
            let reads = now.is_none_or(|item| Self::reads(item, entry.as_ref()));
            let stored = match reads {
                true => self.plan.records.get(id).await?,
                false => None,
            };
            let (changes, settled) = match (now, &entry) {
                (Some(item), entry) => {
                    self.settling(item, entry.as_ref(), stored.as_ref(), adopted)
                }
                (None, Some(entry)) => self.removing(entry, stored.as_ref()),
                (None, None) => return Ok(Settled::Same),
            };
            if changes.is_empty() {
                return Ok(settled);
            }
            let refused = match self.tenant.apply(changes).await {
                Ok(_) => return Ok(settled),
                Err(error) => (refusal(&error), error),
            };
            match refused {
                (Some(Refusal::Overtaken), error) => {
                    let read = (entry.clone(), stored);
                    if before.as_ref() == Some(&read) {
                        return Err(error).wrap_err_with(|| format!("settling object {id}"));
                    }
                    before = Some(read);
                }
                (Some(Refusal::TooLarge), _)
                    if matches!(settled, Settled::Replaced | Settled::Deleted) =>
                {
                    return Ok(Settled::TooLarge)
                }
                (Some(Refusal::TooLarge), error) | (Some(Refusal::Unfit(_)), error)
                    if self.options.skip_refused && now.is_some() =>
                {
                    return Ok(Settled::Refused(error.to_string()))
                }
                (_, error) => return Err(error).wrap_err_with(|| format!("settling object {id}")),
            }
        }
        Ok(Settled::Foreign)
    }

    /// Make the steps, several to a request, and answer with what each
    /// did, by id. A request the node refuses is made again a step at a
    /// time, each from a new reading.
    async fn make(
        &self,
        steps: Vec<Step>,
        adopted: bool,
        copied: &mut Copied,
    ) -> Result<Vec<(ObjectId, Settled)>> {
        let mut made = Vec::new();
        for batch in batches(steps, self.options.page, self.options.batch_bytes) {
            let changes: Vec<Change> = batch
                .iter()
                .flat_map(|step| step.changes.iter().cloned())
                .collect();
            let refused = match changes.is_empty() {
                true => None,
                false => self.tenant.apply(changes).await.err(),
            };
            match refused {
                None => made.extend(batch.into_iter().map(|step| (step.item.id, step.settled))),
                Some(error) if refusal(&error).is_some() => {
                    for step in batch {
                        let id = step.item.id;
                        let entry = self.entry(id).await?;
                        let settled = self.settle(id, Some(&step.item), entry, adopted);
                        made.push((id, settled.await?));
                    }
                }
                Some(error) => return Err(error.into()),
            }
        }
        for (id, settled) in &made {
            copied.count(*id, settled.clone());
        }
        Ok(made)
    }

    /// Store a page of objects of a collection the import made, each with
    /// its entry. Where the collection or the ledger already has one of
    /// them the request is refused whole, and its objects are settled one
    /// at a time.
    async fn store(&self, items: Vec<Item>, copied: &mut Copied) -> Result<()> {
        let step = |item: Item| Step {
            changes: self.fresh(&item),
            settled: Settled::Stored,
            item,
        };
        let steps = items.into_iter().map(step).collect();
        self.make(steps, false, copied).await.map(drop)
    }

    /// Take up a page of a collection that had no ledger: give each
    /// object of Weaviate's its entry, by what the collection holds
    /// under its id. `from` is the last id of the page before, `through`
    /// that of this one.
    async fn adopt(
        &self,
        from: Option<ObjectId>,
        through: ObjectId,
        items: Vec<Item>,
        entries: &mut Entries<'_>,
        copied: &mut Copied,
    ) -> Result<()> {
        let mut known = HashMap::new();
        while let Some(entry) = entries.through(through).await? {
            known.insert(entry.id, entry);
        }
        let ids: Vec<ObjectId> = items.iter().map(|item| item.id).collect();
        let mut held = self.held(from, through, &ids, ids.len()).await?;
        let step = |item: Item| {
            let entry = known.remove(&item.id);
            let stored = held.remove(&item.id);
            let (changes, settled) = self.settling(&item, entry.as_ref(), stored.as_ref(), true);
            Step {
                item,
                changes,
                settled,
            }
        };
        let steps = items.into_iter().map(step).collect();
        self.make(steps, true, copied).await.map(drop)
    }

    /// What the collection holds under `ids`, all of them after `from`
    /// and none after `through`: read as one range where they are many
    /// of the `span` objects there, one at a time where they are few.
    async fn held(
        &self,
        from: Option<ObjectId>,
        through: ObjectId,
        ids: &[ObjectId],
        span: usize,
    ) -> Result<HashMap<ObjectId, Stored>> {
        let mut held = HashMap::new();
        if ids.len() * SPARSE < span {
            for id in ids {
                held.extend(
                    self.plan
                        .records
                        .get(*id)
                        .await?
                        .map(|stored| (*id, stored)),
                );
            }
            return Ok(held);
        }
        let mut page = Page::new(u32::try_from(self.options.page).unwrap_or(u32::MAX));
        page.vectors = self.plan.vectors();
        page.after = from;
        loop {
            let listed = self.plan.records.list(&page).await?;
            let past = listed.objects.last().is_none_or(|last| last.id >= through);
            let wanted = listed
                .objects
                .into_iter()
                .filter(|stored| ids.contains(&stored.id));
            held.extend(wanted.map(|stored| (stored.id, stored)));
            match listed.next {
                Some(next) if !past => page.after = Some(next),
                _ => return Ok(held),
            }
        }
    }

    /// What Weaviate holds now under `ids`, all of them among the `span`
    /// objects it listed after `cursor`: asked for as that page again
    /// where they are many of it, and one at a time where they are few
    /// or the page no longer has them. An id it answers nothing for is
    /// left out; one vellev cannot hold is there without its item.
    async fn fetch(
        &self,
        cursor: Option<&str>,
        ids: &[ObjectId],
        span: usize,
        copied: &mut Copied,
    ) -> Result<HashMap<ObjectId, Option<Item>>> {
        let class = &self.plan.class;
        let vectors = self.plan.vectors();
        let mut objects = Vec::new();
        if ids.len() * SPARSE >= span {
            for object in self.weaviate.page(class, cursor, span, vectors).await? {
                if ids.contains(&object.object_id()?) {
                    objects.push(object);
                }
            }
        }
        let mut fetched = HashMap::new();
        for object in &objects {
            fetched.insert(object.object_id()?, self.item(object, copied)?);
        }
        for id in ids {
            if fetched.contains_key(id) {
                continue;
            }
            let object = self.weaviate.object(class, *id, vectors, Asked::Any);
            if let Some(object) = object.await? {
                fetched.insert(*id, self.item(&object, copied)?);
            }
        }
        Ok(fetched)
    }

    /// How many objects of a class whose ledger holds `ledgered` a run
    /// may delete.
    fn budget(&self, ledgered: u64) -> u64 {
        match self.options.deletes {
            Deletes::None => 0,
            Deletes::UpTo { count, fraction } => {
                count.min((ledgered as f64 * fraction).ceil() as u64)
            }
        }
    }

    /// Walk Weaviate's ids beside the ledger's, both in id order, and
    /// settle each object the ledger does not have as Weaviate lists it
    /// and each changed since `since`; then see to what the ledger has
    /// that Weaviate did not list. Whether every one of those was seen
    /// to: if not, the catch-up is to be made again from where the last
    /// one began.
    async fn catch_up(&self, since: i64, copied: &mut Copied) -> Result<bool> {
        let class = &self.plan.class;
        let ledgered = self.plan.ledger.count(None).await?;
        let mut gone = Gone {
            entries: Vec::new(),
            count: 0,
            foreign: 0,
            budget: self.budget(ledgered),
        };
        let mut entries = Entries::new(&self.plan.ledger, self.options.page, None);
        let mut pacer = Pacer::new(self.options.per_second);
        let mut cursor: Option<(String, ObjectId)> = None;
        loop {
            let from = cursor.clone();
            let listing = from.as_ref().map(|(id, _)| id.as_str());
            let page = self
                .weaviate
                .page(class, listing, self.options.page, false)
                .await?;
            if page.is_empty() {
                break;
            }
            let mut wanted = Vec::new();
            for seen in &page {
                let id = seen.object_id()?;
                if let Some((last, _)) = cursor.as_ref().filter(|(_, last)| *last >= id) {
                    bail!(
                        "Weaviate listed {} of {} after {last}: its ids are not in order",
                        seen.id,
                        class.class
                    );
                }
                cursor = Some((seen.id.clone(), id));
                while let Some(unlisted) = entries.before(id).await? {
                    gone.push(unlisted);
                }
                let entry = entries.at(id).await?;
                let noted = entry
                    .as_ref()
                    .is_some_and(|entry| entry.updated == seen.updated);
                if !noted || seen.updated.unwrap_or(i64::MAX) >= since {
                    wanted.push((id, entry));
                } else {
                    copied.kept += u64::from(entry.is_some_and(|entry| entry.foreign));
                }
            }
            let Some((_, through)) = cursor.clone() else {
                break;
            };
            pacer.paced(page.len() + wanted.len()).await;
            let from_id = from.as_ref().map(|(_, id)| *id);
            let listing = (listing, from_id, through, page.len());
            self.revisit(listing, wanted, &mut gone, copied).await?;
        }
        while let Some(unlisted) = entries.next().await? {
            gone.push(unlisted);
        }
        self.remove(gone, copied).await
    }

    /// Fetch what Weaviate holds now under the ids `wanted` of one page
    /// it listed, and settle each. One it listed and no longer has when
    /// asked for is gone, if the ledger has it: seen to with the rest.
    async fn revisit(
        &self,
        (cursor, from, through, span): (Option<&str>, Option<ObjectId>, ObjectId, usize),
        wanted: Vec<(ObjectId, Option<Entry>)>,
        gone: &mut Gone,
        copied: &mut Copied,
    ) -> Result<()> {
        if wanted.is_empty() {
            return Ok(());
        }
        let ids: Vec<ObjectId> = wanted.iter().map(|(id, _)| *id).collect();
        let mut fetched = self.fetch(cursor, &ids, span, copied).await?;
        let mut settling = Vec::new();
        for (id, entry) in wanted {
            match (fetched.remove(&id), entry) {
                (Some(Some(item)), entry) => settling.push((item, entry)),
                (Some(None), _) | (None, None) => {}
                (None, Some(entry)) => gone.push(entry),
            }
        }
        let reads = |(item, entry): &(Item, Option<Entry>)| Self::reads(item, entry.as_ref());
        let read: Vec<ObjectId> = settling
            .iter()
            .filter(|s| reads(s))
            .map(|(item, _)| item.id)
            .collect();
        let mut held = self.held(from, through, &read, span).await?;
        let step = |(item, entry): (Item, Option<Entry>)| {
            let stored = held.remove(&item.id);
            let (changes, settled) = self.settling(&item, entry.as_ref(), stored.as_ref(), false);
            Step {
                item,
                changes,
                settled,
            }
        };
        let steps = settling.into_iter().map(step).collect();
        let made = self.make(steps, false, copied).await?;
        let foreign = |(_, settled): &&(ObjectId, Settled)| *settled == Settled::Foreign;
        copied.kept += made.iter().filter(foreign).count() as u64;
        Ok(())
    }

    /// See to what the ledger has that Weaviate did not list, now that
    /// the walk is over. More than the run may delete, none is: the
    /// answer is then false, and they are reported as pending. Otherwise
    /// each is asked for twice, a pause apart, with Weaviate ready and
    /// the class as it was imported, and only one Weaviate twice says it
    /// does not hold is deleted, if it is the import's. One it does hold
    /// after all is settled as any other.
    async fn remove(&self, gone: Gone, copied: &mut Copied) -> Result<bool> {
        if gone.count == 0 {
            return Ok(true);
        }
        if gone.count > gone.budget {
            copied.pending = gone.count;
            copied.kept += gone.foreign;
            let chosen = self.options.deletes == Deletes::None;
            copied.budget = (!chosen).then_some(gone.budget);
            return Ok(chosen);
        }
        let class = &self.plan.class;
        let mut absent = Vec::new();
        let mut present = Vec::new();
        self.weaviate.steady(class, &self.plan.binding).await?;
        for entry in gone.entries {
            match self
                .weaviate
                .object(class, entry.id, false, Asked::All)
                .await?
            {
                Some(_) => present.push(entry),
                None => absent.push(entry),
            }
        }
        if !absent.is_empty() {
            tokio::time::sleep(self.options.confirm).await;
        }
        for group in absent.chunks(self.options.page) {
            // Weaviate is asked everything before the collection is read,
            // so that nothing of Weaviate's stands between reading an
            // object and deleting it.
            self.weaviate.steady(class, &self.plan.binding).await?;
            let mut confirmed = Vec::new();
            for entry in group {
                match self
                    .weaviate
                    .object(class, entry.id, false, Asked::All)
                    .await?
                {
                    Some(_) => present.push(entry.clone()),
                    None => confirmed.push(entry.clone()),
                }
            }
            self.weaviate.steady(class, &self.plan.binding).await?;
            for entry in confirmed {
                let id = entry.id;
                copied.count(id, self.settle(id, None, Some(entry), false).await?);
            }
        }
        for entry in present {
            let id = entry.id;
            let object = self
                .weaviate
                .object(class, id, self.plan.vectors(), Asked::Any)
                .await?;
            let Some(item) = self.items(object.as_slice(), copied)?.pop() else {
                continue;
            };
            let settled = self.settle(id, Some(&item), Some(entry), false).await?;
            copied.kept += u64::from(settled == Settled::Foreign);
            copied.count(id, settled);
        }
        Ok(true)
    }

    /// Compare what the collection holds with what Weaviate does, object
    /// by object, changing nothing: over `pages` pages of Weaviate's,
    /// each begun after an id picked at random, or over all of them.
    async fn verify(&self, pages: Option<usize>) -> Result<Verified> {
        const EXAMPLES: usize = 20;
        let class = &self.plan.class;
        let mut verified = Verified::default();
        let mut unreadable = Copied::default();
        let mut cursor: Option<String> = None;
        let mut looked = 0;
        loop {
            if pages.is_some() {
                cursor = Some(ObjectId::generate().to_string());
            }
            let page = self
                .weaviate
                .page(
                    class,
                    cursor.as_deref(),
                    self.options.page,
                    self.plan.vectors(),
                )
                .await?;
            let from = cursor.as_deref().map(object_id).transpose()?;
            if let Some(last) = page.last() {
                let through = object_id(&last.id)?;
                let mut entries = Entries::new(&self.plan.ledger, self.options.page, from);
                let mut known = HashMap::new();
                while let Some(entry) = entries.through(through).await? {
                    known.insert(entry.id, entry);
                }
                let options = Options {
                    skip_refused: true,
                    ..self.options.clone()
                };
                let reading = Copying::new(self.weaviate, self.tenant, self.plan, &options);
                let items = reading.items(&page, &mut unreadable)?;
                let ids: Vec<ObjectId> = items.iter().map(|item| item.id).collect();
                let held = self.held(from, through, &ids, ids.len()).await?;
                for item in items {
                    verified.looked += 1;
                    let stored = held.get(&item.id);
                    let hash = stored.and_then(|stored| self.plan.hash(stored));
                    let wrong = match (known.get(&item.id), stored) {
                        _ if hash.as_ref() == Some(&item.hash) => &mut verified.same,
                        (None, _) => &mut verified.unaccounted,
                        (Some(entry), _) if entry.foreign => &mut verified.kept,
                        (Some(_), None) => &mut verified.missing,
                        (Some(_), Some(_)) => &mut verified.different,
                    };
                    *wrong += 1;
                    let wrong = verified.missing + verified.different;
                    let listed = verified.examples.len();
                    if listed < EXAMPLES && wrong as usize > listed {
                        verified.examples.push(item.id.to_string());
                    }
                }
                cursor = Some(last.id.clone());
            }
            looked += 1;
            match pages {
                Some(pages) if looked >= pages => break,
                None if page.is_empty() => break,
                _ => {}
            }
        }
        verified.unaccounted += unreadable.refused.len() as u64;
        verified.looked += unreadable.refused.len() as u64;
        Ok(verified)
    }
}

/// Under how many of the objects of a range the ones wanted are read one
/// at a time rather than with the whole range: one in this many.
const SPARSE: usize = 8;

/// A ledger's entries in id order, read a page at a time. Entries the
/// import writes behind where it has read do not come up again.
struct Entries<'a> {
    ledger: &'a Records,
    limit: u32,
    read: VecDeque<Entry>,
    /// The last id read, and where the next page starts if there is one.
    last: Option<ObjectId>,
    next: Option<ObjectId>,
    ended: bool,
}

impl<'a> Entries<'a> {
    /// The entries after `from`, or all of them.
    fn new(ledger: &'a Records, limit: usize, from: Option<ObjectId>) -> Self {
        Self {
            ledger,
            limit: u32::try_from(limit).unwrap_or(u32::MAX),
            read: VecDeque::new(),
            last: from,
            next: from,
            ended: false,
        }
    }

    /// The next entry, left where it is.
    async fn peek(&mut self) -> Result<Option<&Entry>> {
        while self.read.is_empty() && !self.ended {
            let mut page = Page::new(self.limit);
            page.after = self.next;
            let listed = self.ledger.list(&page).await?;
            for stored in listed.objects {
                let entry = Entry::read(stored)?;
                if self.last.is_some_and(|last| last >= entry.id) {
                    bail!("the ledger listed {} out of order", entry.id);
                }
                self.last = Some(entry.id);
                self.read.push_back(entry);
            }
            self.next = listed.next;
            self.ended = listed.next.is_none();
        }
        Ok(self.read.front())
    }

    async fn next(&mut self) -> Result<Option<Entry>> {
        self.peek().await?;
        Ok(self.read.pop_front())
    }

    /// The next entry, if its id comes before `id`.
    async fn before(&mut self, id: ObjectId) -> Result<Option<Entry>> {
        let comes = self.peek().await?.is_some_and(|entry| entry.id < id);
        Ok(comes.then(|| self.read.pop_front()).flatten())
    }

    /// The next entry, if its id is `id` or comes before it.
    async fn through(&mut self, id: ObjectId) -> Result<Option<Entry>> {
        let comes = self.peek().await?.is_some_and(|entry| entry.id <= id);
        Ok(comes.then(|| self.read.pop_front()).flatten())
    }

    /// The entry for `id`, if it is the next.
    async fn at(&mut self, id: ObjectId) -> Result<Option<Entry>> {
        let is = self.peek().await?.is_some_and(|entry| entry.id == id);
        Ok(is.then(|| self.read.pop_front()).flatten())
    }
}

/// Holds copying to the given rate, near enough. Time spent waiting on
/// anything else earns no burst afterwards.
struct Pacer {
    per_second: Option<f64>,
    /// When the next objects may be asked for.
    due: Option<Instant>,
}

impl Pacer {
    fn new(per_second: Option<f64>) -> Self {
        Self {
            per_second,
            due: None,
        }
    }

    /// How long to wait at `now`, having just copied `copied`.
    fn wait(&mut self, now: Instant, copied: usize) -> Duration {
        let Some(per_second) = self.per_second else {
            return Duration::ZERO;
        };
        let from = self.due.map_or(now, |due| due.max(now));
        let due = Duration::try_from_secs_f64(copied as f64 / per_second)
            .ok()
            .and_then(|took| from.checked_add(took));
        self.due = due.or(self.due);
        due.map_or(Duration::MAX, |due| due.duration_since(now))
    }

    async fn paced(&mut self, copied: usize) {
        let wait = self.wait(Instant::now(), copied);
        if !wait.is_zero() {
            tokio::time::sleep(wait).await;
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
    /// Hold the state file against another run for as long as the
    /// answer is kept: two would write over each other's progress.
    fn lock(path: &Path) -> Result<std::fs::File> {
        let beside = path.with_extension("lock");
        let lock = std::fs::File::create(&beside)
            .wrap_err_with(|| format!("making {}", beside.display()))?;
        match lock.try_lock() {
            Ok(()) => Ok(lock),
            Err(std::fs::TryLockError::WouldBlock) => bail!(
                "another import is running with its state in {}",
                path.display()
            ),
            Err(std::fs::TryLockError::Error(error)) => {
                Err(error).wrap_err_with(|| format!("locking {}", beside.display()))
            }
        }
    }

    fn load(path: &Path, source: &Source, tenant: &Tenant) -> Result<Self> {
        let from = source.named(true);
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
                "{} is the state of an import from {} into {}, not from {from} into {into}; if this is that Weaviate at another address, name it as it was named then",
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
            let tokenization = property.tokenization.as_deref().unwrap_or("word");
            // A whole value, an id most likely, is matched by filters
            // alone: searched by its words it would only skew the scores.
            let searchable =
                text && tokenization != "field" && property.index_searchable.unwrap_or(true);
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
        object_id(&self.id)
    }

    fn vector(&self, copy: &Copy) -> Option<&Vec<f32>> {
        match copy {
            Copy::Embedded => None,
            Copy::Vector => self.vector.as_ref(),
            Copy::Named(name) => self.vectors.as_ref()?.get(name),
        }
    }

    /// The object as the collection is to hold it, or why the collection
    /// cannot hold it. An empty vector is no vector. One with a property
    /// the collection lacks is no fault of the object: the run stops.
    fn item(&self, plan: &Plan, options: &Options) -> Result<Result<Item, String>> {
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
            let Some(at) = DateTime::<Utc>::from_timestamp_millis(millis) else {
                return Ok(Err(format!("its time {millis} is no time")));
            };
            properties.insert(name.clone(), json!(at.to_rfc3339()));
        }
        if let Some(gained) = properties
            .keys()
            .find(|name| plan.schema.property(name).is_none())
        {
            bail!(
                "object {} of {} has a property {gained} that the class did not have when this run began; run again",
                self.id,
                plan.class.class
            );
        }
        let vector = self
            .vector(&plan.vector)
            .filter(|vector| !vector.is_empty());
        let dimensions = plan.schema.embedding.dimensions;
        if let Some(vector) = vector.filter(|vector| vector.len() != dimensions as usize) {
            return Ok(Err(format!(
                "it has a vector of {} dimensions, not the collection's {dimensions}; if the class's vectors are of that size, it must be imported into a collection made anew, saying so",
                vector.len()
            )));
        }
        let held = plan.held(id, &properties, vector.cloned());
        Ok(held
            .map(|(content, hash)| Item {
                id,
                updated: self.updated,
                content,
                hash,
            })
            .map_err(|error| error.to_string()))
    }
}

/// How many of a class's copies must answer for an object.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Asked {
    /// As many as Weaviate asks by itself.
    Any,
    /// Every one: none may be without an object for it to be gone.
    All,
}

struct Weaviate {
    url: String,
    tenant: Option<String>,
    http: reqwest::Client,
    /// How long a request that fails in a way that may pass is asked
    /// again.
    patience: Duration,
}

impl Weaviate {
    fn new(source: &Source, patience: Duration) -> Result<Self> {
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
            patience,
        })
    }

    /// The status and body of Weaviate's answer to the request `build`
    /// makes, asked again, each wait twice the one before up to half a
    /// minute, while it fails in a way that may pass: no answer, one cut
    /// short, a fault of the server's, or being turned away for asking
    /// too much or too slowly.
    async fn answer(
        &self,
        what: &str,
        build: impl Fn(&reqwest::Client) -> reqwest::RequestBuilder,
    ) -> Result<(reqwest::StatusCode, String)> {
        let began = Instant::now();
        let mut wait = Duration::from_millis(500);
        loop {
            let failed = match build(&self.http).send().await {
                Ok(answer) => {
                    let status = answer.status();
                    let passing = status.is_server_error()
                        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status == reqwest::StatusCode::REQUEST_TIMEOUT;
                    match answer.text().await {
                        // An answer cut short says nothing, a 404 least
                        // of all.
                        Err(error) => error.to_string(),
                        Ok(text) if passing => format!("{status}: {text}"),
                        Ok(text) => return Ok((status, text)),
                    }
                }
                Err(error) => error.to_string(),
            };
            if began.elapsed() + wait > self.patience {
                bail!(
                    "Weaviate failed {what} for {} seconds, last with {failed}",
                    began.elapsed().as_secs()
                );
            }
            tokio::time::sleep(wait).await;
            wait = (wait * 2).min(Duration::from_secs(30));
        }
    }

    /// What Weaviate answers; `None` if it holds nothing there.
    async fn ask(
        &self,
        what: &str,
        build: impl Fn(&reqwest::Client) -> reqwest::RequestBuilder,
    ) -> Result<Option<Value>> {
        let (status, text) = self.answer(what, build).await?;
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            bail!("Weaviate answered {what} with {status}: {text}");
        }
        serde_json::from_str(&text)
            .map(Some)
            .wrap_err_with(|| format!("reading Weaviate's answer to {what}"))
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
            .ok_or_else(|| eyre!("Weaviate has no class {}", class.class))?;
        serde_json::from_value(page["objects"].clone())
            .wrap_err_with(|| format!("reading a page of {}", class.class))
    }

    /// One object of `class` as it is now; none if Weaviate holds no such
    /// object, or no such class, or is not yet up enough to know.
    async fn object(
        &self,
        class: &Class,
        id: ObjectId,
        vectors: bool,
        asked: Asked,
    ) -> Result<Option<Object>> {
        let mut query = Vec::new();
        query.extend(vectors.then(|| ("include", "vector".to_string())));
        query.extend((asked == Asked::All).then(|| ("consistency_level", "ALL".to_string())));
        query.extend(self.tenant_of(class));
        let path = format!("/v1/objects/{}/{id}", class.class);
        self.get(&path, &query)
            .await?
            .map(|object| {
                serde_json::from_value(object).wrap_err_with(|| format!("reading object {id}"))
            })
            .transpose()
    }

    /// The class's shards, by name, and whether each is loaded and
    /// serving; none if Weaviate has no such class.
    async fn shards(&self, class: &Class) -> Result<Option<Vec<(String, bool)>>> {
        let path = format!("/v1/schema/{}/shards", class.class);
        let query: Vec<_> = self.tenant_of(class).into_iter().collect();
        let Some(shards) = self.get(&path, &query).await? else {
            return Ok(None);
        };
        let shard = |shard: &Value| {
            let name = shard["name"].as_str()?.to_string();
            Some((name, shard["status"].as_str()? == "READY"))
        };
        let read = shards
            .as_array()
            .map(|shards| shards.iter().map(shard).collect());
        read.flatten().map(Some).ok_or_else(|| {
            eyre!(
                "Weaviate did not list the shards of {}: {shards}",
                class.class
            )
        })
    }

    /// What tells this class from one made again under its name: the
    /// names Weaviate gave its shards. None for a class with tenants,
    /// whose shards are named after them.
    async fn identity(&self, class: &Class) -> Result<Option<String>> {
        if class.multi_tenant() {
            return Ok(None);
        }
        let shards = self.shards(class).await?;
        let shards = shards.ok_or_else(|| eyre!("Weaviate has no class {}", class.class))?;
        let mut names: Vec<String> = shards.into_iter().map(|(name, _)| name).collect();
        names.sort();
        Ok(Some(names.join(",")))
    }

    /// Wait until what Weaviate says of an object of `class` can be
    /// believed: it answers that it is ready, and every shard of the
    /// class is loaded. Refuse if the class is gone, or is not the one
    /// `binding` says was imported. A Weaviate that is starting answers
    /// that it holds no such object for objects it holds.
    async fn steady(&self, class: &Class, binding: &Binding) -> Result<()> {
        let began = Instant::now();
        let url = format!("{}/v1/.well-known/ready", self.url);
        loop {
            let (status, _) = self
                .answer("whether it is ready", |http| http.get(&url))
                .await?;
            let shards = match status.is_success() {
                true => self.shards(class).await?,
                false => Some(vec![(String::new(), false)]),
            };
            let Some(shards) = shards else {
                bail!(
                    "Weaviate no longer has class {}; its objects stay in the collection",
                    class.class
                );
            };
            if shards.iter().all(|(_, serving)| *serving) {
                let mut names: Vec<&str> = shards.iter().map(|(name, _)| name.as_str()).collect();
                names.sort();
                let is = names.join(",");
                return match binding.shards.as_ref().filter(|was| **was != is) {
                    Some(was) if !class.multi_tenant() => bail!(
                        "Weaviate's class {} is not the one that was imported: it has been made again since, its shards {is} where they were {was}; its objects stay in the collection",
                        class.class
                    ),
                    _ => Ok(()),
                };
            }
            if began.elapsed() > self.patience {
                bail!(
                    "Weaviate was not ready to say what it holds of {} for {} seconds; nothing was deleted",
                    class.class,
                    began.elapsed().as_secs()
                );
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
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
            page: 100,
            batch_bytes: 1 << 20,
            skew: Duration::ZERO,
            ..Options::new(PathBuf::new())
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
                { "name": "identityId", "dataType": ["text"], "tokenization": "field" },
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
                ("identityId", DataType::Text, true, false),
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

    fn memory(properties: Vec<Property>) -> Collection {
        Collection {
            name: CollectionName::new("Memory").unwrap(),
            properties,
            embedding: embedding(),
            distance: Distance::Cosine,
            bm25: Bm25::default(),
            access_rule: None,
        }
    }

    #[test]
    fn a_class_that_gained_properties_asks_for_those_and_leaves_what_the_collection_has_besides() {
        let owner = Property::new("owner", DataType::Text);
        let mood = Property::new("mood", DataType::Text).searchable();
        let mut stored = memory(vec![owner.clone(), Property::new("left", DataType::Int)]);
        stored.access_rule = Some(vellev_core::AccessRule {
            owner: "owner".into(),
            team: None,
            team_prefix: String::new(),
        });
        let mut wanted = memory(vec![owner.clone(), mood.clone()]);
        assert_eq!(
            reconcile(&stored, &wanted).unwrap(),
            (vec![mood.clone()], vec![])
        );
        assert_eq!(reconcile(&wanted, &wanted).unwrap(), (vec![], vec![]));

        // What is searched or ranked otherwise is still the same data.
        wanted.properties[0] = owner.searchable();
        wanted.bm25.k1 = 2.0;
        let (gained, warnings) = reconcile(&stored, &wanted).unwrap();
        assert_eq!(gained, [mood]);
        assert_eq!(
            warnings,
            [
                "property owner is indexed otherwise in the collection than the class now asks; the collection's stays as it is",
                "its BM25 is now k1 2 and b 0.75; the collection's stays k1 1.2 and b 0.75",
            ]
        );
    }

    #[test]
    fn a_class_that_no_longer_fits_its_collection_is_refused_for_every_reason_at_once() {
        let stored = memory(vec![Property::new("count", DataType::Int)]);
        let mut wanted = memory(vec![Property::new("count", DataType::Text)]);
        wanted.embedding.dimensions = 8;
        wanted.embedding.model = "weaviate:text2vec-openai".into();
        wanted.distance = Distance::Dot;
        let refused = reconcile(&stored, &wanted).unwrap_err().to_string();
        assert_eq!(
            refused,
            "its vectors have 8 dimensions and the collection's 4; \
             its vectors are made by weaviate:text2vec-openai and the collection's by weaviate:none; \
             its distance is Dot and the collection's Cosine; \
             property count is text and the collection's int"
        );

        // The same model, set up to embed other text, is another model.
        let mut wanted = stored.clone();
        wanted.embedding.document_prefix = Some("passage: ".into());
        let refused = reconcile(&stored, &wanted).unwrap_err().to_string();
        assert_eq!(
            refused,
            "its text is embedded otherwise than the collection's"
        );
    }

    fn plan(properties: Vec<Property>, vector: Copy) -> Plan {
        let client =
            vellev_client::Client::connect(&["http://127.0.0.1:9"], "", Default::default());
        let tenant = client
            .unwrap()
            .tenant(&vellev_core::TenantId::new("unreached").unwrap());
        let schema = memory(properties);
        Plan {
            class: class(json!([]), json!({})),
            records: tenant.collection(&schema.name),
            ledger: tenant.collection(&ledger_of(&schema.name).unwrap()),
            ledger_name: ledger_of(&schema.name).unwrap(),
            schema,
            vector,
            binding: Binding {
                class: Some("Memory".into()),
                source: "http://weaviate".into(),
                created_as: None,
                updated_as: None,
                shards: Some("a".into()),
                adopting: false,
                finalised: false,
            },
        }
    }

    #[test]
    fn an_object_hashes_as_vellev_holds_it_however_it_was_written() {
        let properties = vec![
            Property::new("at", DataType::Date),
            Property::new("score", DataType::Number),
        ];
        let plan = plan(properties, Copy::Vector);
        let id: ObjectId = serde_json::from_value(json!(id(1))).unwrap();
        let hash = |properties: Value, vector: &[f32]| {
            let properties = properties.as_object().unwrap();
            let held = plan.held(id, properties, Some(vector.to_vec()));
            held.map(|(_, hash)| hash)
                .map_err(|error| error.to_string())
        };
        let vector = [1.0, 0.0, 0.0, 0.5];
        let weaviates = json!({ "at": "2026-05-01T11:00:00+02:00", "score": 150 });
        let vellevs = json!({ "at": "2026-05-01T09:00:00Z", "score": 150.0 });
        let hashed = hash(weaviates.clone(), &vector).unwrap();
        assert_eq!(hashed.len(), 22);
        assert_eq!(hash(vellevs, &vector).as_ref(), Ok(&hashed));
        let later = json!({ "at": "2026-05-01T09:00:01Z", "score": 150 });
        assert_ne!(hash(later, &vector).as_ref(), Ok(&hashed));
        assert_ne!(
            hash(weaviates.clone(), &[1.0, 0.0, 0.0, 0.25]).as_ref(),
            Ok(&hashed)
        );

        // What the collection, as this run knows it, could not have been
        // sent is nobody's the import knows of.
        let stored: Stored = serde_json::from_value(json!({
            "id": id,
            "properties": { "added_since": "by another writer" },
        }))
        .unwrap();
        assert_eq!(plan.hash(&stored), None);
    }

    #[test]
    fn an_object_vellev_embeds_hashes_without_the_vector_vellev_made() {
        let plan = plan(vec![Property::new("title", DataType::Text)], Copy::Embedded);
        let id: ObjectId = serde_json::from_value(json!(id(1))).unwrap();
        let properties = json!({ "title": "a red fox" });
        let properties = properties.as_object().unwrap();
        let (sent, hash) = plan.held(id, properties, None).unwrap();
        assert_eq!(sent.vector, None);
        let embedded = plan.held(id, properties, Some(vec![9.0, 1.0, 0.0, 0.0]));
        assert_eq!(embedded.unwrap().1, hash);
    }

    #[test]
    fn an_object_is_read_as_the_collection_holds_one_or_refused_saying_why() {
        let plan = plan(vec![Property::new("title", DataType::Text)], Copy::Vector);
        let item = |object: Value| {
            let object: Object = serde_json::from_value(object).unwrap();
            object.item(&plan, &options())
        };
        let read = item(json!({
            "id": id(1),
            "properties": { "title": "a red fox", "unset": null },
            "vector": [],
            "lastUpdateTimeUnix": "1791489111510",
        }));
        let read = read.unwrap().unwrap();
        assert_eq!(
            read.content.properties,
            *json!({ "title": "a red fox" }).as_object().unwrap()
        );
        assert_eq!(read.content.vector, None, "an empty vector is none");
        assert_eq!(read.updated, Some(1_791_489_111_510));

        // A property the collection lacks is the run's to stop at, not
        // the object's fault.
        let gained = item(json!({ "id": id(2), "properties": { "mood": "glad" } }));
        assert_eq!(
            gained.err().unwrap().to_string(),
            format!("object {} of Memory has a property mood that the class did not have when this run began; run again", id(2))
        );
        let refused = |object: Value| item(object).unwrap().err().unwrap();
        let longer = refused(json!({ "id": id(3), "vector": [1.0, 2.0] }));
        assert!(
            longer.starts_with("it has a vector of 2 dimensions, not the collection's 4"),
            "{longer}"
        );
        let mistyped = refused(json!({ "id": id(4), "properties": { "title": 4 } }));
        assert!(mistyped.contains("title"), "{mistyped}");
    }

    /// What a class of these properties is copied by, and an object of
    /// it as Weaviate and as vellev hold it.
    struct Fixture {
        weaviate: Weaviate,
        tenant: Tenant,
        plan: Plan,
        options: Options,
    }

    impl Fixture {
        fn new(options: Options) -> Self {
            let source = Source {
                url: "http://127.0.0.1:9".into(),
                key: None,
                tenant: None,
                name: None,
            };
            let client =
                vellev_client::Client::connect(&["http://127.0.0.1:9"], "", Default::default());
            let properties = vec![
                Property::new("title", DataType::Text),
                Property::new("mood", DataType::Text),
                Property::new("score", DataType::Number),
            ];
            Self {
                weaviate: Weaviate::new(&source, Duration::ZERO).unwrap(),
                tenant: client
                    .unwrap()
                    .tenant(&vellev_core::TenantId::new("unreached").unwrap()),
                plan: plan(properties, Copy::Vector),
                options,
            }
        }

        fn copying(&self) -> Copying<'_> {
            Copying::new(&self.weaviate, &self.tenant, &self.plan, &self.options)
        }

        fn item(&self, title: &str, updated: i64) -> Item {
            let properties = json!({ "title": title, "score": 0.1 + 0.2 });
            let vector = Some(vec![1.0, 0.0, 0.0, 0.5]);
            let id = serde_json::from_value(json!(id(1))).unwrap();
            let held = self.plan.held(id, properties.as_object().unwrap(), vector);
            let (content, hash) = held.unwrap();
            Item {
                id,
                updated: Some(updated),
                content,
                hash,
            }
        }

        fn stored(&self, title: &str) -> Stored {
            let item = self.item(title, 0);
            serde_json::from_value(json!({
                "id": item.id,
                "properties": item.content.properties,
                "vector": item.content.vector,
            }))
            .unwrap()
        }

        fn entry(&self, title: &str, updated: i64, foreign: bool) -> Entry {
            let item = self.item(title, updated);
            Entry {
                id: item.id,
                updated: item.updated,
                hash: item.hash,
                foreign,
            }
        }
    }

    /// What kind each change is, and of which collection.
    fn kinds(changes: &[Change]) -> Vec<String> {
        let kind = |change: &Change| match change {
            Change::Insert { collection, .. } => format!("insert {collection}"),
            Change::Upsert { collection, .. } => format!("upsert {collection}"),
            Change::Delete { collection, .. } => format!("delete {collection}"),
            Change::Patch {
                only_if: Some(only_if),
                properties,
                vector: None,
                ..
            } if properties.is_empty() => {
                format!("unless {} is not {}", only_if.property, only_if.equals)
            }
            other => format!("{other:?}"),
        };
        changes.iter().map(kind).collect()
    }

    const GUARDS: [&str; 3] = [
        "unless title is not \"fox\"",
        "unless mood is not null",
        "unless score is not 0.30000000000000004",
    ];

    #[test]
    fn an_object_is_stored_replaced_or_left_by_what_the_ledger_and_the_collection_hold() {
        let fixture = Fixture::new(options());
        let copying = fixture.copying();
        let hen = fixture.item("hen", 2);
        let ours = fixture.entry("fox", 1, false);
        let settling = |entry: Option<&Entry>, stored: Option<&Stored>| {
            let (changes, settled) = copying.settling(&hen, entry, stored, false);
            (kinds(&changes), settled)
        };
        let entry = "vellev_imported_Memory";

        // New to Weaviate: stored, if neither is there to be written over.
        assert_eq!(
            settling(None, None),
            (
                vec!["insert Memory".into(), format!("insert {entry}")],
                Settled::Stored
            )
        );
        // The import's, changed in Weaviate: replaced, unless a property
        // of it changes first, one without a value among them.
        let replaced = settling(Some(&ours), Some(&fixture.stored("fox")));
        let mut expected: Vec<String> = GUARDS.iter().map(|guard| guard.to_string()).collect();
        expected.extend(["upsert Memory".to_string(), format!("upsert {entry}")]);
        assert_eq!(replaced, (expected, Settled::Replaced));
        // Another writer's, deleted by one, or never the import's: left,
        // and the ledger says so.
        let foreign = (vec![format!("upsert {entry}")], Settled::Foreign);
        assert_eq!(settling(Some(&ours), Some(&fixture.stored("owl"))), foreign);
        assert_eq!(settling(Some(&ours), None), foreign);
        assert_eq!(
            settling(None, Some(&fixture.stored("owl"))),
            (vec![format!("insert {entry}")], Settled::Foreign)
        );
        // As Weaviate has it already, whoever stored it: the import's.
        assert_eq!(
            settling(None, Some(&fixture.stored("hen"))),
            (vec![format!("insert {entry}")], Settled::Same)
        );
        let marked = fixture.entry("fox", 1, true);
        assert_eq!(
            settling(Some(&marked), Some(&fixture.stored("hen"))),
            (vec![format!("upsert {entry}")], Settled::Same)
        );
        // Once the ledger says an object is another writer's, it is not
        // said again.
        let said = fixture.entry("hen", 2, true);
        assert_eq!(
            settling(Some(&said), Some(&fixture.stored("owl"))),
            (vec![], Settled::Foreign)
        );
    }

    #[test]
    fn an_object_unchanged_is_not_read_and_only_its_time_is_noted_anew() {
        let fixture = Fixture::new(options());
        let copying = fixture.copying();
        let hen = fixture.item("hen", 2);
        let noted = fixture.entry("hen", 2, false);
        let earlier = fixture.entry("hen", 1, false);
        assert!(!Copying::reads(&hen, Some(&noted)));
        assert!(Copying::reads(&hen, None));
        assert!(Copying::reads(&hen, Some(&fixture.entry("fox", 2, false))));
        assert!(Copying::reads(&hen, Some(&fixture.entry("hen", 2, true))));
        let (changes, settled) = copying.settling(&hen, Some(&noted), None, false);
        assert_eq!((kinds(&changes), settled), (vec![], Settled::Same));
        let (changes, settled) = copying.settling(&hen, Some(&earlier), None, false);
        assert_eq!(kinds(&changes), ["upsert vellev_imported_Memory"]);
        assert_eq!(settled, Settled::Same);
        let Change::Upsert { properties, .. } = &changes[0] else {
            unreachable!()
        };
        assert_eq!(
            Value::Object(properties.clone()),
            json!({ "updated": 2, "hash": hen.hash })
        );
    }

    #[test]
    fn a_collection_taken_up_is_given_nothing_it_lacks_unless_nothing_else_wrote_to_it() {
        let hen = |fixture: &Fixture| fixture.item("hen", 2);
        let entry = "vellev_imported_Memory";
        let taken = |fixture: &Fixture, stored: Option<&Stored>| {
            let (changes, settled) = fixture
                .copying()
                .settling(&hen(fixture), None, stored, true);
            (kinds(&changes), settled)
        };
        // Without a ledger nothing tells an object new to Weaviate from
        // one deleted in vellev, nor one changed in Weaviate from one
        // changed in vellev.
        let fixture = Fixture::new(options());
        let withheld = (vec![format!("insert {entry}")], Settled::Foreign);
        assert_eq!(taken(&fixture, None), withheld);
        assert_eq!(taken(&fixture, Some(&fixture.stored("fox"))), withheld);
        let (changes, _) = fixture.copying().settling(&hen(&fixture), None, None, true);
        let Change::Insert { properties, .. } = &changes[0] else {
            unreachable!()
        };
        assert_eq!(properties["foreign"], json!(true));
        assert_eq!(
            taken(&fixture, Some(&fixture.stored("hen"))),
            (vec![format!("insert {entry}")], Settled::Same)
        );

        let sole = Fixture::new(Options {
            sole_writer: true,
            ..options()
        });
        assert_eq!(
            taken(&sole, None),
            (
                vec!["insert Memory".into(), format!("insert {entry}")],
                Settled::Stored
            )
        );
        let mut expected: Vec<String> = GUARDS.iter().map(|guard| guard.to_string()).collect();
        expected.extend(["upsert Memory".to_string(), format!("insert {entry}")]);
        assert_eq!(
            taken(&sole, Some(&sole.stored("fox"))),
            (expected, Settled::Replaced)
        );
    }

    #[test]
    fn only_what_the_import_stored_and_nobody_changed_is_deleted() {
        let fixture = Fixture::new(options());
        let copying = fixture.copying();
        let ours = fixture.entry("fox", 1, false);
        let removing = |stored: Option<&Stored>| {
            let (changes, settled) = copying.removing(&ours, stored);
            (kinds(&changes), settled)
        };
        let entry = "vellev_imported_Memory";
        let mut expected: Vec<String> = GUARDS.iter().map(|guard| guard.to_string()).collect();
        expected.extend(["delete Memory".to_string(), format!("delete {entry}")]);
        assert_eq!(
            removing(Some(&fixture.stored("fox"))),
            (expected, Settled::Deleted)
        );
        assert_eq!(
            removing(Some(&fixture.stored("owl"))),
            (vec![format!("delete {entry}")], Settled::Left)
        );
        assert_eq!(
            removing(None),
            (vec![format!("delete {entry}")], Settled::Same)
        );
    }

    #[test]
    fn a_run_deletes_no_more_than_the_fewer_of_a_count_and_a_part_of_the_ledger() {
        let budget = |deletes: Deletes, ledgered: u64| {
            let fixture = Fixture::new(Options {
                deletes,
                ..options()
            });
            fixture.copying().budget(ledgered)
        };
        let usual = Options::new(PathBuf::new()).deletes;
        assert_eq!(
            usual,
            Deletes::UpTo {
                count: 1_000,
                fraction: 0.05
            }
        );
        assert_eq!(budget(usual, 20), 1);
        assert_eq!(budget(usual, 1_000), 50);
        assert_eq!(budget(usual, 4_200_000), 1_000);
        assert_eq!(budget(usual, 0), 0);
        assert_eq!(budget(Deletes::None, 4_200_000), 0);

        // What is gone is counted whole, and kept only as far as it may
        // be deleted.
        let mut gone = Gone {
            entries: Vec::new(),
            count: 0,
            foreign: 0,
            budget: 2,
        };
        let fixture = Fixture::new(options());
        for foreign in [false, true, true] {
            gone.push(fixture.entry("fox", 1, foreign));
        }
        assert_eq!((gone.count, gone.foreign, gone.entries.len()), (3, 2, 2));
    }

    #[test]
    fn an_objects_hash_is_what_it_was() {
        // The hashes ledgers hold. A change to how an object is written
        // out before it is hashed disowns every imported object.
        let properties = vec![
            Property::new("title", DataType::Text),
            Property::new("tags", DataType::TextArray),
            Property::new("n", DataType::Int),
            Property::new("score", DataType::Number),
            Property::new("seen", DataType::Bool),
            Property::new("at", DataType::Date),
        ];
        let plan = plan(properties, Copy::Vector);
        let id: ObjectId = serde_json::from_value(json!(id(1))).unwrap();
        let hash = |properties: Value, vector: Option<Vec<f32>>| {
            let held = plan.held(id, properties.as_object().unwrap(), vector);
            held.unwrap().1
        };
        let whole = json!({
            "title": "a red fox",
            "tags": ["den", "night"],
            "n": 7,
            "score": 0.30000000000000004,
            "seen": true,
            "at": "2026-05-01T09:00:00.5+02:00",
        });
        assert_eq!(
            hash(whole, Some(vec![1.0, -0.5, 0.25, 3.0e-7])),
            "fYInEaptchJ21Y7XbYfFlg"
        );
        assert_eq!(hash(json!({}), None), "RBNvo1WzZ4oRRq0W9-hkng");
        assert_eq!(
            hash(json!({ "title": "" }), Some(vec![0.0, 0.0, 0.0, 0.0])),
            "sp43dhHq-svTjR4u_J7Faw"
        );
    }

    #[test]
    fn a_ledger_or_a_note_of_a_later_version_is_not_read() {
        let entry = |properties: Value| {
            let stored = json!({ "id": id(1), "properties": properties });
            Entry::read(serde_json::from_value(stored).unwrap())
        };
        let read = entry(json!({ "hash": "h", "updated": 3 })).unwrap();
        assert_eq!((read.updated, read.foreign), (Some(3), false));
        assert!(
            entry(json!({ "hash": "h", "foreign": true }))
                .unwrap()
                .foreign
        );
        assert!(entry(json!({ "hash": "h", "version": 1 })).is_ok());
        let later = entry(json!({ "hash": "h", "version": 2 })).unwrap_err();
        assert!(
            later.to_string().contains("written by a later release"),
            "{later}"
        );
        let note = json!({ "id": id(1), "properties": { "source": "s", "version": 2 } });
        assert!(Binding::read(&serde_json::from_value(note).unwrap()).is_err());
    }

    #[test]
    fn what_the_imports_own_collections_lack_is_added_to_them() {
        let wanted = || {
            bookkeeping(
                CollectionName::new("vellev_imports").unwrap(),
                &Binding::PROPERTIES,
            )
        };
        let made = grown(None, wanted());
        assert!(matches!(made.as_slice(), [Change::CreateCollection { .. }]));
        assert_eq!(grown(Some(wanted()), wanted()), []);
        let mut earlier = wanted();
        earlier.properties.truncate(3);
        let added: Vec<String> = grown(Some(earlier), wanted())
            .iter()
            .map(|change| match change {
                Change::AddProperty { property, .. } => property.name.clone(),
                other => format!("{other:?}"),
            })
            .collect();
        assert_eq!(
            added,
            ["updatedAs", "shards", "adopting", "finalised", "version"]
        );
    }

    #[test]
    fn a_run_is_settled_behind_or_unsettled_by_what_it_left() {
        let report = |copied: Copied| Report {
            classes: [("Memory".to_string(), copied)].into(),
        };
        let level = Copied {
            weaviate: 10,
            imported: 10,
            vellev: 10,
            ..Copied::default()
        };
        let outcome = |copied: Copied| report(copied).outcome(false);
        assert_eq!(outcome(level.clone()), Outcome::Settled);
        // Another writer's objects are no fault of the import's, unless
        // it is asked to be strict.
        let shared = Copied {
            kept: 2,
            vellev: 14,
            ..level.clone()
        };
        assert_eq!(outcome(shared.clone()), Outcome::Settled);
        assert_eq!(report(shared).outcome(true), Outcome::Unsettled);
        let behind = Copied {
            weaviate: 11,
            ..level.clone()
        };
        assert_eq!(outcome(behind.clone()), Outcome::Behind);
        // What it was told not to delete is still to be deleted, and
        // nothing is behind for it.
        let pending = Copied {
            weaviate: 8,
            pending: 2,
            ..level.clone()
        };
        assert_eq!(outcome(pending.clone()), Outcome::Settled);
        let over = Copied {
            budget: Some(1),
            ..pending
        };
        assert_eq!(outcome(over), Outcome::Unsettled);
        let refused = Copied {
            weaviate: 11,
            refused: [(id(1), "why".to_string())].into(),
            ..level.clone()
        };
        assert_eq!(outcome(refused), Outcome::Unsettled);
        let large = Copied {
            too_large: vec![id(1)],
            ..level.clone()
        };
        assert_eq!(outcome(large), Outcome::Unsettled);
        let wrong = Copied {
            verified: Some(Verified {
                different: 1,
                ..Verified::default()
            }),
            ..behind
        };
        assert_eq!(outcome(wrong), Outcome::Unsettled);
    }

    #[test]
    fn objects_go_to_a_request_until_it_is_full_and_a_large_one_goes_alone() {
        let fixture = Fixture::new(options());
        let step = |bytes: usize| {
            let item = fixture.item(&"x".repeat(bytes), 0);
            Step {
                changes: fixture.copying().fresh(&item),
                settled: Settled::Stored,
                item,
            }
        };
        let sizes = |most: usize, bytes: usize| {
            let steps = vec![step(100), step(100), step(100), step(9_000), step(100)];
            batches(steps, most, bytes)
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>()
        };
        assert_eq!(sizes(10, 1 << 20), [5]);
        assert_eq!(sizes(2, 1 << 20), [2, 2, 1]);
        assert_eq!(sizes(10, 2_000), [3, 1, 1]);
    }

    #[test]
    fn time_spent_waiting_on_anything_else_earns_no_burst_afterwards() {
        let begun = Instant::now();
        let second = Duration::from_secs(1);
        let mut pacer = Pacer::new(Some(100.0));
        assert_eq!(pacer.wait(begun, 100), second);
        assert_eq!(pacer.wait(begun, 50), second * 3 / 2, "asked for early");
        let stalled = begun + Duration::from_secs(3_600);
        assert_eq!(pacer.wait(stalled, 100), second);
        assert_eq!(pacer.wait(stalled + second, 1), second / 100);
        assert_eq!(Pacer::new(None).wait(begun, 1_000_000), Duration::ZERO);
        // A rate too slow to wait out is waited on for good, not a panic.
        assert_eq!(
            Pacer::new(Some(f64::MIN_POSITIVE)).wait(begun, 1),
            Duration::MAX
        );
    }
}
