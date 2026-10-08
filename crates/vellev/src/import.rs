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
//! hash says. The import replaces and deletes its own objects only, and
//! deletes one only once Weaviate itself answers that it holds no such
//! object. It stores a new one only where the collection holds nothing
//! under the id and the ledger has no entry, so what somebody deleted in
//! vellev is not brought back. Whatever else it finds it leaves as it is
//! and counts as kept: what another writer stored, changed or deleted.
//!
//! Weaviate's cursor is not a snapshot. Every run therefore ends, for each
//! class, with a catch-up pass: Weaviate's ids are walked beside the
//! ledger's, both in id order, and every object the ledger does not have
//! as Weaviate now does, or that changed within a margin for the two
//! clocks disagreeing, is fetched and settled, as is every one the ledger
//! has and Weaviate did not list. Run again, an import is caught up again,
//! so it can follow Weaviate until writes move over; only a pass made once
//! Weaviate takes no more writes is exact. Each class's count is then
//! compared with the ledger's and the collection's.

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
use vellev_api::{Change, Code, Content, Missing, OnlyIf, Page, Stored};
use vellev_client::{Records, Tenant};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider,
    ObjectError, ObjectId, Property,
};

/// How many times a request to Weaviate is tried before the import stops,
/// and how many times an object is settled again because another writer
/// changed it meanwhile before it is left to that writer.
const ATTEMPTS: u32 = 5;

/// What a class's ledger is called, before the class's name. No class of
/// Weaviate's can have such a name: theirs begin with a capital.
const LEDGER: &str = "vellev_imported_";

/// The collection that says where each class was imported from.
const IMPORTS: &str = "vellev_imports";

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
}

/// What an import did, by class.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Report {
    pub classes: BTreeMap<String, Copied>,
}

impl Report {
    /// Whether every class is caught up and its collection as the import
    /// left it.
    pub fn matched(&self) -> bool {
        self.classes
            .values()
            .all(|copied| copied.caught_up() && copied.untouched())
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
    /// Objects this run would have stored, stored again or deleted, and
    /// left because the collection did not hold them as the import had
    /// left them: another writer's.
    pub kept: u64,
    /// How many objects Weaviate held at the end, how many of Weaviate's
    /// the ledger had accounted for, and how many the collection held.
    pub weaviate: u64,
    pub imported: u64,
    pub vellev: u64,
    /// What was mapped but means something different in vellev.
    pub warnings: Vec<String>,
}

impl Copied {
    /// Whether the import has accounted for as many objects as Weaviate
    /// holds. It has not if Weaviate was written to behind the catch-up;
    /// another run catches up.
    pub fn caught_up(&self) -> bool {
        self.weaviate == self.imported
    }

    /// Whether the collection holds as many objects as the import has
    /// accounted for, and this run found none that was not as the import
    /// had left it. Another run changes nothing about one that does not.
    pub fn untouched(&self) -> bool {
        self.kept == 0 && self.vellev == self.imported
    }

    fn count(&mut self, settled: Settled) {
        match settled {
            Settled::Stored => self.stored += 1,
            Settled::Replaced => self.changed += 1,
            Settled::Deleted => self.deleted += 1,
            Settled::Kept => self.kept += 1,
            Settled::Nothing => {}
        }
    }
}

/// What settling an object did to the collection.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Settled {
    Stored,
    Replaced,
    Deleted,
    /// It was another writer's, and left.
    Kept,
    /// The collection already held what Weaviate does, or neither held it.
    Nothing,
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
}

impl Plan {
    /// The object as vellev would hold and return it, and the hash of
    /// that: of its properties, and of its vector unless vellev makes it.
    fn held(
        &self,
        id: ObjectId,
        properties: &Map<String, Value>,
        vector: Option<Vec<f32>>,
    ) -> Result<(Content, String), ObjectError> {
        let vector = vector.filter(|_| self.vector != Copy::Embedded);
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
struct Item {
    id: ObjectId,
    /// When Weaviate says it last changed.
    updated: Option<i64>,
    content: Content,
    hash: String,
}

impl Item {
    /// How many bytes a request grows by to store it, near enough.
    fn size(&self) -> usize {
        const ENTRY: usize = 512;
        serde_json::to_vec(&self.content).map_or(ENTRY, |content| content.len() + ENTRY)
    }
}

/// What the ledger says of one id.
#[derive(Debug, Clone, PartialEq)]
struct Entry {
    id: ObjectId,
    updated: Option<i64>,
    hash: String,
}

impl Entry {
    fn read(stored: Stored) -> Result<Self> {
        let hash = stored.properties.get("hash").and_then(Value::as_str);
        Ok(Self {
            id: stored.id,
            updated: stored.properties.get("updated").and_then(Value::as_i64),
            hash: hash
                .ok_or_else(|| eyre!("the ledger's entry for {} has no hash", stored.id))?
                .to_string(),
        })
    }
}

/// Copy the classes `options` names from `source` into `tenant`, and
/// catch up with what changed in Weaviate meanwhile.
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
        let (plan, warnings, made) = prepare(&weaviate, class, tenant, source, options).await?;
        if made {
            // Whatever the state file says of the class was said of a
            // collection that is no longer there.
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

    for plan in &plans {
        let name = &plan.class.class;
        let copied = report.classes.get_mut(name).expect("planned");
        let copying = Copying {
            weaviate: &weaviate,
            tenant,
            plan,
            options,
        };
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
            let items = page
                .iter()
                .map(|object| object.item(plan, options))
                .collect::<Result<Vec<_>>>()?;
            copying.store(items, copied).await?;
            state.class(name).after = Some(last);
            state.save(&options.state)?;
            pacer.paced(page.len()).await;
        }
        state.save(&options.state)?;
    }

    for plan in &plans {
        let name = &plan.class.class;
        let copied = report.classes.get_mut(name).expect("planned");
        let copying = Copying {
            weaviate: &weaviate,
            tenant,
            plan,
            options,
        };
        let begun = Utc::now().timestamp_millis();
        let since = state.class(name).caught_up_to.unwrap_or(state.started)
            - options.skew.as_millis() as i64;
        copying.catch_up(since, copied).await?;
        state.class(name).caught_up_to = Some(begun);
        state.save(&options.state)?;
        copied.weaviate = weaviate.count(&plan.class).await?;
        copied.imported = plan.ledger.count(None).await?;
        copied.vellev = plan.records.count(None).await?;
    }
    Ok(report)
}

/// How to copy `class`, what of it means something different in vellev,
/// and whether its collection was made just now. The collection is made
/// if there is none, or else given the properties the class has gained,
/// and its ledger and the note of where it came from are made beside it.
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
    let ledger = CollectionName::new(format!("{LEDGER}{name}")).map_err(|_| {
        eyre!(
            "class {}: its name is too long for a ledger to be kept beside it",
            class.class
        )
    })?;
    let imports = CollectionName::new(IMPORTS).expect("a valid name");
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
    let binding = Binding {
        source: source.named(class.multi_tenant()),
        created_as: options.created_as.clone(),
        updated_as: options.updated_as.clone(),
    };
    let mut changes = Vec::new();
    let made = stored.is_none();
    let schema = match stored {
        None => {
            // A ledger left by a collection since dropped says nothing of
            // the one made now.
            if tenant.collection(&ledger).schema().await?.is_some() {
                changes.push(Change::DropCollection {
                    collection: ledger.clone(),
                });
            }
            changes.push(Change::CreateCollection {
                schema: Box::new(wanted.clone()),
            });
            wanted
        }
        Some(mut stored) => {
            let bound = tenant.collection(&imports).get(Binding::id(&name)).await?;
            if let Some(bound) = bound {
                Binding::read(&bound).agrees(&binding, &name)?;
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
            stored
        }
    };
    changes.extend([
        Change::CreateCollection {
            schema: Box::new(bookkeeping(
                ledger.clone(),
                &[("updated", DataType::Int), ("hash", DataType::Text)],
            )),
        },
        Change::CreateCollection {
            schema: Box::new(bookkeeping(imports.clone(), &Binding::PROPERTIES)),
        },
        Change::Upsert {
            collection: imports,
            id: Binding::id(&name),
            properties: binding.properties(),
            vector: None,
        },
    ]);
    tenant
        .apply(changes)
        .await
        .wrap_err_with(|| format!("preparing collection {name}"))?;
    let plan = Plan {
        records,
        ledger: tenant.collection(&ledger),
        ledger_name: ledger,
        schema,
        class,
        vector,
    };
    Ok((plan, warnings, made))
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

/// Where a class's collection was imported from, and with which of
/// Weaviate's times: kept in the tenant, so that its ledger is never read
/// against another Weaviate, nor its objects stored two ways.
#[derive(Debug, Clone, PartialEq)]
struct Binding {
    source: String,
    created_as: Option<String>,
    updated_as: Option<String>,
}

impl Binding {
    const PROPERTIES: [(&'static str, DataType); 3] = [
        ("source", DataType::Text),
        ("createdAs", DataType::Text),
        ("updatedAs", DataType::Text),
    ];

    /// Where a collection's binding is kept: under an id made of its name,
    /// so that there is one.
    fn id(collection: &CollectionName) -> ObjectId {
        let hash = Sha256::digest(collection.as_str());
        ObjectId(uuid::Uuid::from_slice(&hash[..16]).expect("sixteen bytes"))
    }

    fn properties(&self) -> Map<String, Value> {
        let values = [
            &Some(self.source.clone()),
            &self.created_as,
            &self.updated_as,
        ];
        Self::PROPERTIES
            .iter()
            .zip(values)
            .filter_map(|((name, _), value)| Some((name.to_string(), json!(value.as_ref()?))))
            .collect()
    }

    fn read(stored: &Stored) -> Self {
        let text = |name: &str| {
            let value = stored.properties.get(name).and_then(Value::as_str);
            value.map(String::from)
        };
        Self {
            source: text("source").unwrap_or_default(),
            created_as: text("createdAs"),
            updated_as: text("updatedAs"),
        }
    }

    /// Refuse a run that would copy into `collection` from somewhere
    /// else, or otherwise, than `self` says it was.
    fn agrees(&self, now: &Self, collection: &CollectionName) -> Result<()> {
        if self.source != now.source {
            bail!(
                "collection {collection} was imported from {}, not from {}; if this is that Weaviate at another address, name it as it was named then",
                self.source,
                now.source
            );
        }
        if self != now {
            bail!(
                "collection {collection} was imported with Weaviate's creation time as {:?} and its time of change as {:?}; an import goes on as it began",
                self.created_as,
                self.updated_as
            );
        }
        Ok(())
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

/// Whether a write was refused because an object is no longer as the
/// write took it to be, there or not: another writer got there first.
fn overtaken(error: &vellev_client::Error) -> bool {
    match error.code() {
        Some(Code::Conflict) => true,
        Some(Code::NotFound) => error.missing() == Some(Missing::Object),
        _ => false,
    }
}

/// `items` in runs of at most `most`, each of at most `bytes` bytes, near
/// enough; an item larger than that is a run of its own.
fn batches(items: Vec<Item>, most: usize, bytes: usize) -> Vec<Vec<Item>> {
    let mut batches: Vec<Vec<Item>> = Vec::new();
    let mut held = 0;
    for item in items {
        let size = item.size();
        match batches.last_mut() {
            Some(batch) if batch.len() < most && held + size <= bytes => {
                held += size;
                batch.push(item);
            }
            _ => {
                held = size;
                batches.push(vec![item]);
            }
        }
    }
    batches
}

/// One class being copied.
struct Copying<'a> {
    weaviate: &'a Weaviate,
    tenant: &'a Tenant,
    plan: &'a Plan,
    options: &'a Options,
}

impl Copying<'_> {
    fn data(&self) -> CollectionName {
        self.plan.schema.name.clone()
    }

    /// Note in the ledger what Weaviate holds under the item's id: in
    /// place of the entry `known` to be there, or else only if there is
    /// still none, the step failing otherwise.
    fn note(&self, item: &Item, known: bool) -> Change {
        let collection = self.plan.ledger_name.clone();
        let updated = item.updated.map(|updated| ("updated", json!(updated)));
        let properties = updated
            .into_iter()
            .chain([("hash", json!(item.hash))])
            .map(|(name, value)| (name.to_string(), value))
            .collect();
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

    /// Store the object and its entry, if the collection holds nothing
    /// under its id and the ledger no entry; otherwise the step fails.
    fn fresh(&self, item: &Item) -> [Change; 2] {
        let object = Change::Insert {
            collection: self.data(),
            id: item.id,
            properties: item.content.properties.clone(),
            vector: item.content.vector.clone(),
        };
        [object, self.note(item, false)]
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

    /// Store a page of objects, each with its entry, several to a
    /// request. Where the collection or the ledger already has one of
    /// them the request is refused whole, and its objects are settled one
    /// at a time.
    async fn store(&self, items: Vec<Item>, copied: &mut Copied) -> Result<()> {
        for batch in batches(items, self.options.page, self.options.batch_bytes) {
            let changes = batch.iter().flat_map(|item| self.fresh(item)).collect();
            match self.tenant.apply(changes).await {
                Ok(_) => copied.stored += batch.len() as u64,
                Err(error) if overtaken(&error) => {
                    for item in batch {
                        let entry = self.entry(item.id).await?;
                        copied.count(self.settle(item.id, Some(item), entry).await?);
                    }
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    /// Bring the collection to hold under `id` what Weaviate holds `now`,
    /// if what it holds is the import's to change, and the ledger to say
    /// what Weaviate holds. The step that changes an object fails if the
    /// object's properties changed since they were read; it is then read
    /// and settled again, and after enough of that left to whoever keeps
    /// writing it.
    async fn settle(
        &self,
        id: ObjectId,
        now: Option<Item>,
        mut entry: Option<Entry>,
    ) -> Result<Settled> {
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                entry = self.entry(id).await?;
            }
            let (changes, settled) = self.settling(id, now.as_ref(), entry.as_ref()).await?;
            if changes.is_empty() {
                return Ok(settled);
            }
            match self.tenant.apply(changes).await {
                Ok(_) => return Ok(settled),
                Err(error) if overtaken(&error) => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(Settled::Kept)
    }

    /// The changes that settle `id`, and what they do.
    async fn settling(
        &self,
        id: ObjectId,
        now: Option<&Item>,
        entry: Option<&Entry>,
    ) -> Result<(Vec<Change>, Settled)> {
        match (now, entry) {
            (None, None) => return Ok((Vec::new(), Settled::Nothing)),
            (Some(item), Some(entry)) if item.hash == entry.hash => {
                let noted = item.updated == entry.updated;
                let changes = if noted {
                    vec![]
                } else {
                    vec![self.note(item, true)]
                };
                return Ok((changes, Settled::Nothing));
            }
            _ => {}
        }
        let stored = self.plan.records.get(id).await?;
        let held = stored.as_ref().and_then(|stored| self.plan.hash(stored));
        let ours = entry.is_some_and(|entry| held.as_deref() == Some(entry.hash.as_str()));
        Ok(match (now, entry, &stored) {
            (Some(item), None, None) => (self.fresh(item).into(), Settled::Stored),
            (Some(item), _, Some(stored)) if ours => {
                let mut changes = self.unless_changed(stored);
                changes.push(Change::Upsert {
                    collection: self.data(),
                    id,
                    properties: item.content.properties.clone(),
                    vector: item.content.vector.clone(),
                });
                changes.push(self.note(item, true));
                (changes, Settled::Replaced)
            }
            (Some(item), _, _) => {
                // Another writer's, or deleted by one. The ledger notes
                // what Weaviate holds, so the object is the import's
                // again if the collection comes to hold the same.
                let same = held.as_deref() == Some(item.hash.as_str());
                let settled = if same {
                    Settled::Nothing
                } else {
                    Settled::Kept
                };
                (vec![self.note(item, entry.is_some())], settled)
            }
            (None, _, Some(stored)) if ours => {
                // Weaviate answers the same of an object in a class it no
                // longer has as of one it deleted.
                if !self.weaviate.holds(&self.plan.class).await? {
                    bail!(
                        "Weaviate no longer has class {}; its objects stay in the collection",
                        self.plan.class.class
                    );
                }
                let mut changes = self.unless_changed(stored);
                changes.push(Change::Delete {
                    collection: self.data(),
                    id,
                });
                changes.push(self.forget(id));
                (changes, Settled::Deleted)
            }
            (None, _, Some(_)) => (vec![self.forget(id)], Settled::Kept),
            (None, _, None) => (vec![self.forget(id)], Settled::Nothing),
        })
    }

    /// Fetch what Weaviate holds under `id` now and settle it. An object
    /// Weaviate listed and no longer has when asked for is gone.
    async fn revisit(
        &self,
        id: ObjectId,
        entry: Option<Entry>,
        pacer: &mut Pacer,
        copied: &mut Copied,
    ) -> Result<()> {
        let vectors = self.plan.vector != Copy::Embedded;
        let now = self
            .weaviate
            .object(&self.plan.class, &id.to_string(), vectors)
            .await?
            .map(|object| object.item(self.plan, self.options))
            .transpose()?;
        copied.count(self.settle(id, now, entry).await?);
        pacer.paced(1).await;
        Ok(())
    }

    /// Walk Weaviate's ids beside the ledger's, both in id order, and
    /// revisit each object the ledger does not have as Weaviate lists it,
    /// each changed since `since`, and each the ledger has that Weaviate
    /// did not list.
    async fn catch_up(&self, since: i64, copied: &mut Copied) -> Result<()> {
        let class = &self.plan.class;
        if self.weaviate.count(class).await? == 0 && self.plan.ledger.count(None).await? > 0 {
            bail!(
                "Weaviate holds nothing of class {}, which was imported from it; if that is right, drop the collection, and otherwise check what Weaviate this is: nothing was deleted",
                class.class
            );
        }
        let mut entries = Entries::new(&self.plan.ledger, self.options.page);
        let mut pacer = Pacer::new(self.options.per_second);
        let mut after: Option<(String, ObjectId)> = None;
        loop {
            let from = after.as_ref().map(|(id, _)| id.as_str());
            let page = self
                .weaviate
                .page(class, from, self.options.page, false)
                .await?;
            if page.is_empty() {
                break;
            }
            for seen in &page {
                let id = seen.object_id()?;
                if let Some((last, _)) = after.as_ref().filter(|(_, last)| *last >= id) {
                    bail!(
                        "Weaviate listed {} of {} after {last}: its ids are not in order",
                        seen.id,
                        class.class
                    );
                }
                after = Some((seen.id.clone(), id));
                while let Some(gone) = entries.before(id).await? {
                    self.revisit(gone.id, Some(gone), &mut pacer, copied)
                        .await?;
                }
                let entry = entries.at(id).await?;
                let noted = entry
                    .as_ref()
                    .is_some_and(|entry| entry.updated == seen.updated);
                if !noted || seen.updated.unwrap_or(i64::MAX) >= since {
                    self.revisit(id, entry, &mut pacer, copied).await?;
                }
            }
            pacer.paced(page.len()).await;
        }
        while let Some(gone) = entries.next().await? {
            self.revisit(gone.id, Some(gone), &mut pacer, copied)
                .await?;
        }
        Ok(())
    }
}

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
    fn new(ledger: &'a Records, limit: usize) -> Self {
        Self {
            ledger,
            limit: u32::try_from(limit).unwrap_or(u32::MAX),
            read: VecDeque::new(),
            last: None,
            next: None,
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
        serde_json::from_value(json!(self.id)).wrap_err_with(|| format!("object id {}", self.id))
    }

    fn vector(&self, copy: &Copy) -> Option<&Vec<f32>> {
        match copy {
            Copy::Embedded => None,
            Copy::Vector => self.vector.as_ref(),
            Copy::Named(name) => self.vectors.as_ref()?.get(name),
        }
    }

    /// The object as the collection is to hold it. An empty vector is no
    /// vector.
    fn item(&self, plan: &Plan, options: &Options) -> Result<Item> {
        let id = self.object_id()?;
        let class = &plan.class.class;
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
        if let Some(gained) = properties
            .keys()
            .find(|name| plan.schema.property(name).is_none())
        {
            bail!(
                "object {} of {class} has a property {gained} that the class did not have when this run began; run again",
                self.id
            );
        }
        let vector = self
            .vector(&plan.vector)
            .filter(|vector| !vector.is_empty());
        let dimensions = plan.schema.embedding.dimensions;
        if let Some(vector) = vector.filter(|vector| vector.len() != dimensions as usize) {
            bail!(
                "object {} of {class} has a vector of {} dimensions, not the collection's {dimensions}; if the class's vectors are of that size, drop the collection and import it again, saying so",
                self.id,
                vector.len()
            );
        }
        let (content, hash) = plan
            .held(id, &properties, vector.cloned())
            .map_err(|error| eyre!("object {} of {class}: {error}", self.id))?;
        Ok(Item {
            id,
            updated: self.updated,
            content,
            hash,
        })
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
    /// it fails in a way that may pass, being turned away for asking too
    /// much among them; `None` if it holds nothing there.
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
                    let passing = status.is_server_error()
                        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || status == reqwest::StatusCode::REQUEST_TIMEOUT;
                    match answer.text().await {
                        // An answer cut short says nothing, a 404 least
                        // of all.
                        Err(error) => error.to_string(),
                        Ok(_) if status == reqwest::StatusCode::NOT_FOUND => return Ok(None),
                        Ok(text) if status.is_success() => {
                            return serde_json::from_str(&text)
                                .map(Some)
                                .wrap_err_with(|| format!("reading Weaviate's answer to {what}"))
                        }
                        Ok(text) if passing => format!("{status}: {text}"),
                        Ok(text) => bail!("Weaviate answered {what} with {status}: {text}"),
                    }
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
            .ok_or_else(|| eyre!("Weaviate has no class {}", class.class))?;
        serde_json::from_value(page["objects"].clone())
            .wrap_err_with(|| format!("reading a page of {}", class.class))
    }

    /// Whether Weaviate still has the class.
    async fn holds(&self, class: &Class) -> Result<bool> {
        let path = format!("/v1/schema/{}", class.class);
        Ok(self.get(&path, &[]).await?.is_some())
    }

    /// One object of `class` as it is now; none if Weaviate holds no such
    /// object, or no such class.
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
            batch_bytes: 1 << 20,
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
            ledger: tenant.collection(&schema.name),
            ledger_name: schema.name.clone(),
            schema,
            vector,
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
            object
                .item(&plan, &options())
                .map_err(|error| error.to_string())
        };
        let read = item(json!({
            "id": id(1),
            "properties": { "title": "a red fox", "unset": null },
            "vector": [],
            "lastUpdateTimeUnix": "1791489111510",
        }))
        .unwrap();
        assert_eq!(
            read.content.properties,
            *json!({ "title": "a red fox" }).as_object().unwrap()
        );
        assert_eq!(read.content.vector, None, "an empty vector is none");
        assert_eq!(read.updated, Some(1_791_489_111_510));

        let gained = item(json!({ "id": id(2), "properties": { "mood": "glad" } }));
        assert_eq!(
            gained.err().unwrap(),
            format!("object {} of Memory has a property mood that the class did not have when this run began; run again", id(2))
        );
        let longer = item(json!({ "id": id(3), "vector": [1.0, 2.0] }));
        assert!(longer.err().unwrap().starts_with(&format!(
            "object {} of Memory has a vector of 2 dimensions, not the collection's 4",
            id(3)
        )));
        let mistyped = item(json!({ "id": id(4), "properties": { "title": 4 } }));
        assert!(mistyped
            .err()
            .unwrap()
            .starts_with(&format!("object {} of Memory: ", id(4))));
    }

    #[test]
    fn objects_go_to_a_request_until_it_is_full_and_a_large_one_goes_alone() {
        let item = |n: u8, bytes: usize| {
            let properties = json!({ "title": "x".repeat(bytes) });
            Item {
                id: serde_json::from_value(json!(id(n))).unwrap(),
                updated: None,
                content: Content::new(properties.as_object().unwrap().clone()),
                hash: String::new(),
            }
        };
        let sizes = |most: usize, bytes: usize| {
            let items = vec![
                item(1, 100),
                item(2, 100),
                item(3, 100),
                item(4, 9_000),
                item(5, 100),
            ];
            batches(items, most, bytes)
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
