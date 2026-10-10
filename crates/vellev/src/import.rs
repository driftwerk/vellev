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
//! the import deletes what it stored under that id. [`Copying::removing`]
//! is the one place a delete of an object is made.

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
use vellev_api::{Bounded, Change, Code, Content, Missing, Page, Stored};
use vellev_client::{Records, Tenant};
use vellev_core::{
    Bm25, Collection, CollectionName, DataType, Distance, Embedding, EmbeddingProvider,
    ObjectError, ObjectId, Property, Version,
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
    /// its ledger has or its collection holds, whichever is fewest, and
    /// never the last object the collection holds unless the fraction is
    /// all of it. A run that finds more gone deletes none of them.
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
    /// The most changes sent to vellev in one request: no more than a
    /// node commits together with other writers' requests, or the request
    /// is committed alone and holds theirs up.
    pub batch_changes: usize,
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
    /// without a ledger, at any time since before it was taken up: what
    /// it lacks is stored and what differs from Weaviate's is replaced,
    /// where otherwise both are withheld.
    pub sole_writer: bool,
    /// With `sole_writer`: delete the objects the collection held when it
    /// was taken up that Weaviate did not hold, if they are still as they
    /// were then.
    pub delete_unknown: bool,
    /// That something else has written to a collection found without a
    /// ledger: what was withheld, and what it held that Weaviate did not,
    /// is another writer's, and left for good.
    pub other_writers: bool,
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
            batch_changes: vellev_storage::GROUP_MOST.rows,
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
            delete_unknown: false,
            other_writers: false,
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
    /// Classes passed over because their import was ended.
    pub ended: Vec<String>,
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
    /// Objects gone from Weaviate that stay in the collection because
    /// they were not as the import had stored them: another writer's.
    /// The ledger keeps their entries while the collection holds them,
    /// so the same on every run until it no longer does.
    pub left: u64,
    /// Objects the collection held when it was taken up without a ledger
    /// that Weaviate does not hold: deleted there before there was a
    /// ledger, or another writer's. They stay until the run is told
    /// which.
    pub unknown: u64,
    /// Ids the run was told were another writer's, which neither
    /// Weaviate nor the collection holds now. The ledger keeps their
    /// entries for good, so that what the other writer deleted is not
    /// copied back should Weaviate come to hold the id again.
    pub barred: u64,
    /// Objects of Weaviate's that were not copied, by the ledger's mark:
    /// those a collection lacked, or held otherwise, when it was taken up
    /// without a ledger, there being no telling whether another writer
    /// had deleted or changed them. They are copied by a run told that
    /// nothing else wrote to the collection.
    pub withheld: u64,
    /// Objects gone from Weaviate that this run did not delete, though
    /// they may be the import's: it was told to delete none, or they are
    /// more than `budget`, what it may delete.
    pub pending: u64,
    pub budget: Option<u64>,
    /// Objects vellev cannot hold, passed over: why, by id.
    pub refused: BTreeMap<String, String>,
    /// How many objects Weaviate held at the end, how many of Weaviate's
    /// the ledger had accounted for, those withheld aside, and how many
    /// the collection held. An entry of an id Weaviate does not hold,
    /// as those unknown, left and barred are, is not among the second.
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
    /// Objects looked at: of Weaviate's, or, where only some are looked
    /// at, of those the ledger has.
    pub looked: u64,
    /// Those the collection holds exactly as Weaviate does.
    pub same: u64,
    /// Those it holds otherwise, or not at all, though the ledger takes
    /// them for the import's.
    pub different: u64,
    pub missing: u64,
    /// Those the ledger marks as another writer's or as withheld, those
    /// it has no entry for, and those it takes for the import's that
    /// Weaviate no longer holds.
    pub kept: u64,
    pub withheld: u64,
    pub unaccounted: u64,
    pub gone: u64,
    /// Objects the collection holds that the ledger has no entry for:
    /// counted only when everything is compared.
    pub extra: Option<u64>,
    /// Some of the ids that differ or are missing.
    pub examples: Vec<String>,
}

impl Copied {
    /// Whether the import has accounted for exactly the objects Weaviate
    /// holds, those it passed over or has yet to delete aside. It has
    /// not if Weaviate was written to behind the catch-up; another run
    /// catches up.
    pub fn caught_up(&self) -> bool {
        let accounted = self.imported + self.withheld + self.refused.len() as u64;
        self.weaviate + self.pending == accounted
    }

    /// Whether something is left that running again does not put right:
    /// objects not copied or not deleted, objects nobody has said whose
    /// they are, objects vellev cannot hold, or a comparison that found
    /// differences or nothing to compare.
    pub fn unsettled(&self) -> bool {
        let wrong = |verified: &Verified| {
            verified.different + verified.missing + verified.gone > 0
                || (verified.looked == 0 && self.weaviate > 0)
        };
        self.withheld > 0
            || self.unknown > 0
            || self.pending > 0
            || !self.refused.is_empty()
            || self.verified.as_ref().is_some_and(wrong)
    }

    /// Whether the collection holds as many objects as the import has
    /// accounted for, and none of them is another writer's.
    pub fn untouched(&self) -> bool {
        self.kept == 0 && self.left == 0 && self.vellev == self.imported
    }

    /// How many of the ledger's entries are of nothing the import has
    /// copied from Weaviate, or of nothing Weaviate holds.
    fn unvouched(&self) -> u64 {
        self.withheld + self.unknown + self.left + self.barred
    }

    fn count(&mut self, id: ObjectId, settled: Settled) {
        match settled {
            Settled::Stored => self.stored += 1,
            Settled::Replaced => self.changed += 1,
            Settled::Deleted => self.deleted += 1,
            Settled::Left => self.left += 1,
            Settled::Refused(why) => {
                self.refused.insert(id.to_string(), why);
            }
            Settled::Foreign | Settled::Withheld | Settled::Same => {}
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
    /// It was not copied into a collection taken up without a ledger:
    /// the ledger says so.
    Withheld,
    /// Gone from Weaviate and not the import's to delete: left, its
    /// entry kept and saying so.
    Left,
    /// The collection already held what Weaviate does, or neither held it.
    Same,
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

/// What the ledger says the collection holds under an id.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mark {
    /// What the hash says, as far as the import has seen.
    Stored,
    /// Something else, or nothing: another writer's doing.
    Foreign,
    /// Something else, or nothing, found so when the collection was taken
    /// up without a ledger: the object was not copied, there being no
    /// telling whose doing it was.
    Withheld,
    /// An object Weaviate did not hold, found when the collection was
    /// taken up without a ledger. The hash is of what the collection
    /// held then: Weaviate has never been seen to hold the id.
    Unknown,
    /// Another writer's, on the word of whoever ran the import: what was
    /// withheld or unknown when something else was said to have written
    /// to the collection. Nothing the import is told later makes it
    /// anything else, and its entry has no hash for an object to match.
    Theirs,
}

/// What the ledger says of one id.
#[derive(Debug, Clone)]
struct Entry {
    id: ObjectId,
    /// The version the node gave the entry when it was read, which
    /// writing it anew or forgetting it is conditional on; none for one
    /// yet to be written.
    at: Option<Version>,
    updated: Option<i64>,
    /// Of the object as Weaviate held it when the import last looked.
    hash: String,
    mark: Mark,
    /// Of what the collection held under the id when it was taken up
    /// without a ledger, where that was not what Weaviate held.
    found: Option<String>,
    /// That Weaviate has twice said it holds no such object, and the
    /// object was left in the collection, being nothing the import may
    /// delete.
    gone: bool,
}

/// Two entries are equal that say the same: where each was read does not
/// count, so that one is not written again for saying what it said.
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        let says = |entry: &Self| {
            let found = entry.found.clone();
            (
                entry.id,
                entry.updated,
                entry.hash.clone(),
                entry.mark,
                found,
                entry.gone,
            )
        };
        says(self) == says(other)
    }
}

impl Entry {
    const PROPERTIES: [(&'static str, DataType); 9] = [
        ("updated", DataType::Int),
        ("hash", DataType::Text),
        ("foreign", DataType::Bool),
        ("version", DataType::Int),
        ("withheld", DataType::Bool),
        ("unknown", DataType::Bool),
        ("found", DataType::Text),
        ("gone", DataType::Bool),
        ("theirs", DataType::Bool),
    ];

    /// The entry as it is once the object is said to be another
    /// writer's: with no hash, and nothing of what was found, so that
    /// nothing the collection comes to hold is ever taken for the
    /// import's by it. `gone` says whether Weaviate lists the id.
    fn theirs(&self, updated: Option<i64>, gone: bool) -> Self {
        Self {
            id: self.id,
            at: self.at,
            updated,
            hash: String::new(),
            mark: Mark::Theirs,
            found: None,
            gone,
        }
    }

    /// What the ledger is to say of the item, marked so.
    fn of(item: &Item, mark: Mark) -> Self {
        Self {
            id: item.id,
            at: None,
            updated: item.updated,
            hash: item.hash.clone(),
            mark,
            found: None,
            gone: false,
        }
    }

    /// Whether the entry is of an object Weaviate lists, last changed
    /// at `updated`: then it says all there is to say until Weaviate
    /// changes the object. An entry of an object Weaviate did not hold,
    /// or was found not to, is of another object once it lists the id.
    fn listed(&self, updated: Option<i64>) -> bool {
        (self.updated, self.gone) == (updated, false) && self.mark != Mark::Unknown
    }

    /// The entry as the ledger holds it. One the node gives no version
    /// for is not read: nothing could then tell that it had been written
    /// by another run since.
    fn read(stored: Stored) -> Result<Self> {
        let property = |name: &str| stored.properties.get(name);
        later(property("version"), "the ledger")?;
        let hash = property("hash").and_then(Value::as_str);
        let said = |name: &str| property(name) == Some(&json!(true));
        let Some(at) = stored.version else {
            bail!(
                "the node gave no version for the ledger's entry for {}: it is of a release before 0.2, and the import keeps no ledger on one",
                stored.id
            );
        };
        Ok(Self {
            id: stored.id,
            at: Some(at),
            updated: property("updated").and_then(Value::as_i64),
            hash: hash
                .ok_or_else(|| eyre!("the ledger's entry for {} has no hash", stored.id))?
                .to_string(),
            mark: match ["theirs", "unknown", "withheld", "foreign"].map(said) {
                [true, ..] => Mark::Theirs,
                [_, true, ..] => Mark::Unknown,
                [_, _, true, _] => Mark::Withheld,
                [.., true] => Mark::Foreign,
                _ => Mark::Stored,
            },
            found: property("found").and_then(Value::as_str).map(String::from),
            gone: said("gone"),
        })
    }

    /// As the ledger holds it. What is as usual is left out, and costs
    /// nothing: the first version, and an object that is the import's
    /// and that Weaviate holds.
    fn properties(&self) -> Map<String, Value> {
        let mark = match self.mark {
            Mark::Stored => None,
            Mark::Foreign => Some("foreign"),
            Mark::Withheld => Some("withheld"),
            Mark::Unknown => Some("unknown"),
            Mark::Theirs => Some("theirs"),
        };
        let properties = [
            self.updated.map(|updated| ("updated", json!(updated))),
            Some(("hash", json!(self.hash))),
            mark.map(|mark| (mark, json!(true))),
            (VERSION > 1).then(|| ("version", json!(VERSION))),
            self.found.as_ref().map(|found| ("found", json!(found))),
            self.gone.then_some(("gone", json!(true))),
        ];
        let properties = properties.into_iter().flatten();
        properties
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

impl Options {
    /// Refuse what no run can be told, before anything is asked of
    /// Weaviate or the node.
    fn check(&self) -> Result<()> {
        if self.page == 0 {
            bail!("a page holds at least one object");
        }
        if self.batch_bytes == 0 || self.batch_changes == 0 {
            bail!("a request holds at least one byte, and one change");
        }
        if self
            .per_second
            .is_some_and(|rate| !(rate.is_finite() && rate > 0.0))
        {
            bail!("the rate of copying must be a number more than nothing");
        }
        if self.other_writers && self.sole_writer {
            bail!("either nothing else wrote to the collections, or something did");
        }
        if self.delete_unknown && !self.sole_writer {
            bail!(
                "what a collection held that Weaviate did not is deleted only once nothing but an import is said to have written to it"
            );
        }
        if let Deletes::UpTo { fraction, .. } = self.deletes {
            if !(0.0..=1.0).contains(&fraction) {
                bail!("the part of a class a run may delete is between nothing and all of it");
            }
        }
        Ok(())
    }
}

/// How a run goes about a collection that is already there, by the note
/// of where it was imported from, if there is one, and whether it has a
/// ledger. Answers with whether the collection is being taken up,
/// whether what it holds that Weaviate does not is noted meanwhile, and
/// whether its copying begins from the start.
///
/// What a collection holds that Weaviate does not is noted only on the
/// first taking up of one nothing has ever kept a ledger of, and on that
/// taking up gone on with after a stop. A ledger or a note found without
/// the other says the collection was imported before: what it holds by
/// now may have been written since, and nothing of it is noted. A
/// ledger that is there is believed, whatever it holds: one with no
/// entry is of a collection nothing was imported into.
fn taking_up(bound: Option<&Binding>, ledgered: bool) -> (bool, bool, bool) {
    match bound {
        Some(bound) => {
            let begins = !ledgered;
            let census = bound.census && bound.adopting && !begins;
            (bound.adopting || begins, census, begins)
        }
        None => (true, !ledgered, true),
    }
}

/// Copy the classes `options` names from `source` into `tenant`, and
/// catch up with what changed in Weaviate meanwhile; or, asked only to
/// verify, compare the two.
pub async fn weaviate(source: &Source, tenant: &Tenant, options: &Options) -> Result<Report> {
    options.check()?;
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
            let name = class.class.clone();
            let prepared = prepare(&weaviate, class, tenant, source, options).await?;
            let Some((plan, _, _)) = prepared else {
                report.ended.push(name);
                continue;
            };
            let copying = Copying::new(&weaviate, tenant, &plan, options);
            let mut copied = Copied::default();
            let (verified, stale) = copying.verify(pages, &mut copied).await?;
            copied.verified = Some(verified);
            copied.weaviate = weaviate.count(&plan.class).await?;
            // An entry the next run that copies forgets accounts for
            // nothing of Weaviate's.
            copied.imported = plan.ledger.count(None).await? - copied.unvouched() - stale;
            copied.vellev = plan.records.count(None).await?;
            report.classes.insert(name, copied);
        }
        return Ok(report);
    }

    let _lock = Progress::lock(&options.state)?;
    let mut state = Progress::load(&options.state, source, tenant)?;
    tenant.create().await?;
    let mut plans = Vec::new();
    for class in classes {
        let name = class.class.clone();
        let prepared = prepare(&weaviate, class, tenant, source, options).await?;
        let Some((plan, warnings, begins)) = prepared else {
            report.ended.push(name);
            continue;
        };
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
            let name = &plan.schema.name;
            let noted = tenant.collection(&imports()).get(Binding::id(name)).await?;
            let noted = noted.as_ref().map(Binding::held).transpose()?;
            let Some((_, at)) = noted.filter(|(noted, _)| *noted == plan.binding) else {
                bail!(
                    "the note of where collection {name} was imported from changed under the import: {OTHER_IMPORT}"
                );
            };
            plan.binding.adopting = false;
            plan.binding.census = false;
            plan.binding
                .write(tenant, name, Some(at), Vec::new())
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
        copied.imported = plan.ledger.count(None).await? - copied.unvouched();
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
            let (mut binding, at) = Binding::held(stored)?;
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
            binding.census = false;
            let mut changes = Vec::new();
            if tenant.collection(&ledger).schema().await?.is_some() {
                changes.push(Change::DropCollection { collection: ledger });
            }
            binding.write(tenant, &name, Some(at), changes).await?;
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
/// it came from are made beside it. Nothing, for a class whose import was
/// ended, in a run of every class: it is passed over.
async fn prepare(
    weaviate: &Weaviate,
    class: Class,
    tenant: &Tenant,
    source: &Source,
    options: &Options,
) -> Result<Option<(Plan, Vec<String>, bool)>> {
    let name = CollectionName::new(&class.class)
        .map_err(|error| eyre!("class {}: {error}", class.class))?;
    let bound = tenant
        .collection(&imports())
        .get(Binding::id(&name))
        .await?;
    let bound = bound.as_ref().map(Binding::held).transpose()?;
    let (bound, noted_at) = bound.unzip();
    let ended = bound.as_ref().is_some_and(|bound| bound.finalised);
    if ended && options.classes.is_empty() {
        return Ok(None);
    }
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
    let ledgered = tenant.collection(&ledger).schema().await?;
    let identity = weaviate.identity(&class).await?;
    let mut binding = Binding {
        class: Some(class.class.clone()),
        source: source.named(class.multi_tenant()),
        created_as: options.created_as.clone(),
        updated_as: options.updated_as.clone(),
        shards: identity.clone(),
        adopting: false,
        census: false,
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
            if let Some(bound) = &bound {
                bound.agrees(&binding, &name, options.accept_remade)?;
            }
            (binding.adopting, binding.census, begins) =
                taking_up(bound.as_ref(), ledgered.is_some());
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
    if options.mode == Mode::Import {
        binding.write(tenant, &name, noted_at, changes).await?;
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
    Ok(Some((plan, warnings, begins)))
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
    /// That this is the first taking up of a collection nothing kept a
    /// ledger of before, in which what it holds that Weaviate does not
    /// is noted. No other taking up notes it, and no other run.
    census: bool,
    /// That the import of the collection has ended.
    finalised: bool,
}

impl Binding {
    const PROPERTIES: [(&'static str, DataType); 9] = [
        ("class", DataType::Text),
        ("source", DataType::Text),
        ("createdAs", DataType::Text),
        ("updatedAs", DataType::Text),
        ("shards", DataType::Text),
        ("adopting", DataType::Bool),
        ("finalised", DataType::Bool),
        ("version", DataType::Int),
        ("census", DataType::Bool),
    ];

    /// Where a collection's binding is kept: under an id made of its name,
    /// so that there is one.
    fn id(collection: &CollectionName) -> ObjectId {
        let hash = Sha256::digest(collection.as_str());
        ObjectId(uuid::Uuid::from_slice(&hash[..16]).expect("sixteen bytes"))
    }

    /// The change that notes it for `collection`: over the note that
    /// `was` there, if it is still at that version, or else only if
    /// there is still none.
    fn change(&self, collection: &CollectionName, was: Option<Version>) -> Change {
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
            flag(self.census),
        ];
        let properties = Self::PROPERTIES.iter().zip(values);
        let properties = properties
            .filter_map(|((name, _), value)| Some((name.to_string(), value?)))
            .collect();
        let (collection, id, vector) = (imports(), Self::id(collection), None);
        match was {
            Some(was) => Change::Upsert {
                collection,
                id,
                properties,
                vector,
                only_if_version: Some(was),
            },
            None => Change::Insert {
                collection,
                id,
                properties,
                vector,
            },
        }
    }

    /// The note as the tenant holds it, and the version it is at, which
    /// writing it anew is conditional on.
    fn held(stored: &Stored) -> Result<(Self, Version)> {
        let Some(at) = stored.version else {
            bail!(
                "the node gave no version for the note of where a collection was imported from: it is of a release before 0.2, and the import keeps none on one"
            );
        };
        Ok((Self::read(stored)?, at))
    }

    /// Write the note of `collection`, which was at `was` when read, or
    /// was not there. Refused, something else wrote it meanwhile: an
    /// import that began, or one that was ended.
    async fn write(
        &self,
        tenant: &Tenant,
        collection: &CollectionName,
        was: Option<Version>,
        with: Vec<Change>,
    ) -> Result<()> {
        let mut changes = with;
        changes.push(self.change(collection, was));
        match tenant.apply(changes).await {
            Ok(_) => Ok(()),
            Err(error) if refusal(&error) == Some(Refusal::Overtaken) => {
                Err(error).wrap_err_with(|| {
                    format!(
                        "the note of where collection {collection} was imported from changed under the import: {OTHER_IMPORT}"
                    )
                })
            }
            Err(error) => Err(error).wrap_err_with(|| format!("preparing collection {collection}")),
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
            census: flag("census"),
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
    /// The node does not know a write conditional on a version: it is
    /// of a release before there were any.
    Old,
}

/// Whether a refusal as invalid is of the condition on a version and not
/// of what was to be written: a node of 0.1 refuses a field it does not
/// know by naming it, in one of two ways, and as a release already made
/// it will go on doing so.
fn unversed(code: Code, message: &str) -> bool {
    code == Code::Invalid && message.contains("unknown field") && message.contains(CONDITION)
}

/// The field of a change that makes it conditional on a version.
const CONDITION: &str = "only_if_version";

/// What stops a run that met a node of a release before versions.
const OLD_NODE: &str = "the node refused a write conditional on an object's version, which every node of 0.2 knows: one of an earlier release is among those asked, and the import replaces and deletes nothing through it";

fn refusal(error: &vellev_client::Error) -> Option<Refusal> {
    let large =
        matches!(error, vellev_client::Error::Status { status, .. } if status.as_u16() == 413);
    if large || error.limit_max(Bounded::BodyBytes).is_some() {
        return Some(Refusal::TooLarge);
    }
    let message = match error {
        vellev_client::Error::Api { message, .. } => message.as_str(),
        _ => "",
    };
    match error.code()? {
        code if unversed(code, message) => Some(Refusal::Old),
        Code::Conflict => Some(Refusal::Overtaken),
        Code::NotFound if error.missing() == Some(Missing::Object) => Some(Refusal::Overtaken),
        Code::Invalid => Some(Refusal::Unfit(error.to_string())),
        _ => None,
    }
}

/// One object's part of a request: what settles it.
struct Step {
    item: Item,
    /// What the ledger said of the id when the step was made, if it was
    /// asked.
    noted: Option<Option<Entry>>,
    changes: Vec<Change>,
    settled: Settled,
}

/// What a ledger written by something else means.
const OTHER_IMPORT: &str = "is another import of the tenant running? Two must not run at once";

impl Step {
    /// How many bytes a request grows by to carry it, near enough.
    fn size(&self) -> usize {
        let size = |change| serde_json::to_vec(change).map_or(0, |written| written.len());
        self.changes.iter().map(size).sum()
    }
}

/// `steps` in runs of at most `most`, each of at most `changes` changes
/// and `bytes` bytes, near enough; a step larger than that is a run of
/// its own.
fn batches(steps: Vec<Step>, most: usize, changes: usize, bytes: usize) -> Vec<Vec<Step>> {
    let mut batches: Vec<Vec<Step>> = Vec::new();
    let mut held = (0, 0);
    for step in steps {
        let (count, size) = (step.changes.len(), step.size());
        match batches.last_mut() {
            Some(batch)
                if batch.len() < most && held.0 + count <= changes && held.1 + size <= bytes =>
            {
                held = (held.0 + count, held.1 + size);
                batch.push(step);
            }
            _ => {
                held = (count, size);
                batches.push(vec![step]);
            }
        }
    }
    batches
}

/// What is to be done about an entry of an id Weaviate did not list.
#[derive(Debug, Clone, PartialEq)]
enum Unlisted {
    /// It may be gone: Weaviate is asked, and what the import stored
    /// under it deleted.
    Ask(Entry),
    /// Nothing the import may delete, and Weaviate is not asked: the
    /// entry stays for as long as the collection holds the object, as it
    /// is or, if given, as `now`.
    Keep { entry: Entry, now: Option<Entry> },
}

/// What the ledger has that Weaviate did not list, and may delete if
/// Weaviate says, asked for each, that it holds no such object. No more
/// are kept than a run may delete, and one over, to tell.
struct Gone {
    entries: Vec<Entry>,
    count: u64,
    /// How many of them the ledger marks as another writer's, and as
    /// withheld.
    foreign: u64,
    withheld: u64,
    budget: u64,
    /// Entries to forget, of objects neither Weaviate, by its listing,
    /// nor the collection holds: forgotten only once the walk is over
    /// and Weaviate has said it was ready and the class the one
    /// imported, since a listing that left an id out for any other
    /// reason would have its object stored again as new.
    forgotten: Vec<Change>,
}

impl Gone {
    fn push(&mut self, entry: Entry) {
        self.count += 1;
        self.foreign += u64::from(entry.mark == Mark::Foreign);
        self.withheld += u64::from(entry.mark == Mark::Withheld);
        if self.count <= self.budget {
            self.entries.push(entry);
        }
    }
}

/// What Weaviate says of an id being settled.
#[derive(Clone, Copy)]
enum Now<'a> {
    Holds(&'a Item),
    /// It holds no such object.
    Gone,
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
            let from = after.as_deref().map(object_id).transpose()?;
            let Some(last) = page.last() else {
                if plan.binding.census {
                    // What the collection holds past Weaviate's last id.
                    let entries = entries
                        .get_or_insert_with(|| Entries::new(&plan.ledger, options.page, from));
                    self.adopt(from, None, Vec::new(), entries, copied).await?;
                }
                state.class(name).done = true;
                break;
            };
            let last = last.id.clone();
            let items = self.items(&page, copied)?;
            self.weaviate.steady(&plan.class, &plan.binding).await?;
            if plan.binding.adopting {
                let entries =
                    entries.get_or_insert_with(|| Entries::new(&plan.ledger, options.page, from));
                self.adopt(from, Some(object_id(&last)?), items, entries, copied)
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

    /// Write the entry to the ledger: in place of the one that `was`
    /// there, if that is still at the version it was read at, or else
    /// only if there is still none, the step failing otherwise. Only an
    /// import writes a ledger, so a step that fails so met another's.
    fn note(&self, entry: &Entry, was: Option<&Entry>) -> Change {
        let collection = self.plan.ledger_name.clone();
        let (id, properties, vector) = (entry.id, entry.properties(), None);
        match was {
            Some(was) => Change::Upsert {
                collection,
                id,
                properties,
                vector,
                only_if_version: was.at,
            },
            None => Change::Insert {
                collection,
                id,
                properties,
                vector,
            },
        }
    }

    /// Note in the ledger what Weaviate holds under the item's id, and
    /// what the collection holds by `mark`, unless the entry already
    /// says as much.
    fn noting(&self, item: &Item, entry: Option<&Entry>, mark: Mark) -> Vec<Change> {
        self.renoting(Entry::of(item, mark), entry)
    }

    /// Write `now` in place of the entry there is, unless it says as
    /// much.
    fn renoting(&self, now: Entry, entry: Option<&Entry>) -> Vec<Change> {
        match entry == Some(&now) {
            true => Vec::new(),
            false => vec![self.note(&now, entry)],
        }
    }

    /// Store the object and its entry, if the collection holds nothing
    /// under its id, and the ledger no entry or the one read; otherwise
    /// the step fails.
    fn fresh(&self, item: &Item, entry: Option<&Entry>) -> Vec<Change> {
        let object = Change::Insert {
            collection: self.data(),
            id: item.id,
            properties: item.content.properties.clone(),
            vector: item.content.vector.clone(),
        };
        let stored = Entry::of(item, Mark::Stored);
        vec![object, self.note(&stored, entry)]
    }

    /// Store the item in place of `stored`, and note it. The step fails
    /// unless the object is still at the version it was read at: with
    /// every property and its vector as they were then, and there at
    /// all.
    fn replace(&self, stored: &Stored, item: &Item, was: Option<&Entry>) -> Vec<Change> {
        let object = Change::Upsert {
            collection: self.data(),
            id: item.id,
            properties: item.content.properties.clone(),
            vector: item.content.vector.clone(),
            only_if_version: stored.version,
        };
        vec![object, self.note(&Entry::of(item, Mark::Stored), was)]
    }

    /// Delete `stored` and forget its entry. The step fails unless the
    /// object is still at the version it was read at.
    fn delete(&self, stored: &Stored, entry: &Entry) -> Vec<Change> {
        let object = Change::Delete {
            collection: self.data(),
            id: stored.id,
            only_if_version: stored.version,
        };
        vec![object, self.forget(entry)]
    }

    /// Forget the entry, if it is still at the version it was read at.
    fn forget(&self, entry: &Entry) -> Change {
        Change::Delete {
            collection: self.plan.ledger_name.clone(),
            id: entry.id,
            only_if_version: entry.at,
        }
    }

    /// Stop the run if the ledger's entry for `id` is not the one a
    /// step was made on: only an import writes a ledger, and nothing of
    /// this one's was made between the two readings.
    fn alone(id: ObjectId, read: Option<&Entry>, again: Option<&Entry>) -> Result<()> {
        let at = |entry: Option<&Entry>| entry.map(|entry| entry.at);
        if read != again || at(read) != at(again) {
            bail!("the ledger's entry for {id} changed under the import: {OTHER_IMPORT}");
        }
        Ok(())
    }

    /// Whether the node said what version each object read is at, which
    /// is what replacing or deleting it is conditional on. A node of a
    /// release before versions says none, and a write conditional on
    /// none is not conditional: the run stops rather than make one.
    fn versioned<'s>(read: impl IntoIterator<Item = &'s Stored>) -> Result<()> {
        match read.into_iter().find(|stored| stored.version.is_none()) {
            Some(stored) => bail!(
                "the node gave no version for object {}: it is of a release before 0.2, and the import replaces and deletes only what it can hold to the version it read",
                stored.id
            ),
            None => Ok(()),
        }
    }

    async fn entry(&self, id: ObjectId) -> Result<Option<Entry>> {
        let stored = self.plan.ledger.get(id).await?;
        stored.map(Entry::read).transpose()
    }

    /// Whether settling the item takes reading what the collection holds
    /// under its id: not while the ledger says the import stored exactly
    /// this, and it is nobody else's; nor of what was said to be another
    /// writer's, which is left whatever it is.
    fn reads(item: &Item, entry: Option<&Entry>) -> bool {
        !entry.is_some_and(|entry| {
            entry.mark == Mark::Theirs
                || (&entry.hash, entry.mark, entry.gone) == (&item.hash, Mark::Stored, false)
        })
    }

    /// What brings the collection to hold what Weaviate does under the
    /// item's id, if what it holds, `stored`, is the import's to change,
    /// and the ledger to say what Weaviate holds.
    ///
    /// An object is written over only if the ledger says the import
    /// stored it and it is still as the entry says. While a collection
    /// that had no ledger is `adopted`, an id without an entry is not
    /// vouched for, any more than one whose entry says the object was
    /// withheld or was there unknown to Weaviate: what the collection
    /// lacks under it is not stored and what differs is not replaced,
    /// but marked as withheld, with what the collection held. What is so
    /// marked is stored or replaced, then or by any later run, once
    /// nothing but an import is said to have written to the collection;
    /// once something else is said to have, it is another writer's for
    /// good, and nothing is done to it again.
    fn settling(
        &self,
        item: &Item,
        entry: Option<&Entry>,
        stored: Option<&Stored>,
        adopted: bool,
    ) -> (Vec<Change>, Settled) {
        if let Some(theirs) = entry.filter(|entry| entry.mark == Mark::Theirs) {
            let listed = theirs.theirs(item.updated, false);
            return (self.renoting(listed, entry), Settled::Foreign);
        }
        if !Self::reads(item, entry) {
            return (self.noting(item, entry, Mark::Stored), Settled::Same);
        }
        let held = stored.and_then(|stored| self.plan.hash(stored));
        let mark = entry.map(|entry| entry.mark);
        // Only what the import stored is the import's for being as the
        // entry says: an entry of any other mark says what Weaviate
        // held, or what the collection did, not what the import put
        // there.
        let ours = entry.is_some_and(|entry| {
            entry.mark == Mark::Stored && held.as_deref() == Some(entry.hash.as_str())
        });
        let unvouched = match mark {
            Some(mark) => matches!(mark, Mark::Withheld | Mark::Unknown),
            None => adopted,
        };
        let sole = unvouched && self.options.sole_writer;
        match (entry, stored) {
            // What the collection holds exactly as Weaviate does is the
            // import's, whatever the run is told: there is nothing of
            // another writer's in it to keep.
            (_, Some(_)) if held.as_deref() == Some(item.hash.as_str()) => {
                (self.noting(item, entry, Mark::Stored), Settled::Same)
            }
            _ if unvouched && self.options.other_writers => {
                let theirs = Entry::of(item, Mark::Theirs).theirs(item.updated, false);
                (self.renoting(theirs, entry), Settled::Foreign)
            }
            (Some(_), Some(stored)) if ours => {
                (self.replace(stored, item, entry), Settled::Replaced)
            }
            (None, None) if !adopted => (self.fresh(item, None), Settled::Stored),
            (_, None) if sole => (self.fresh(item, entry), Settled::Stored),
            (_, Some(stored)) if sole => (self.replace(stored, item, entry), Settled::Replaced),
            _ if unvouched => {
                // What the collection held when it was taken up: noted
                // then, and kept through whatever Weaviate does since.
                let found = match entry {
                    Some(entry) if entry.mark == Mark::Unknown => Some(entry.hash.clone()),
                    Some(entry) => entry.found.clone(),
                    None => held,
                };
                let withheld = Entry {
                    found,
                    ..Entry::of(item, Mark::Withheld)
                };
                (self.renoting(withheld, entry), Settled::Withheld)
            }
            _ => (self.noting(item, entry, Mark::Foreign), Settled::Foreign),
        }
    }

    /// Whether what a collection held when it was taken up, that
    /// Weaviate does not hold, is to be deleted: said in so many words,
    /// and only of a collection nothing else is said to have written to.
    fn sweeps(&self) -> bool {
        self.options.sole_writer && self.options.delete_unknown
    }

    /// Whether an entry is of something the collection held when it was
    /// taken up, about which nobody has yet said whose it is.
    fn unsettled(entry: &Entry) -> bool {
        matches!(entry.mark, Mark::Withheld | Mark::Unknown)
    }

    /// Delete what the collection holds under an id the ledger has and
    /// Weaviate no longer does, if it is the import's to delete, and
    /// forget the id. It is the import's if the ledger says the import
    /// stored it and it is still as the entry says. What was there when
    /// the collection was taken up is deleted only when that is asked
    /// for, and only if it is still as it was found then. Anything else
    /// is left, and its entry kept, saying so, or saying that it is
    /// another writer's if that is what the run is told: the ledger
    /// forgets no object the collection holds. The one place a delete of
    /// an object is made.
    fn removing(&self, entry: &Entry, stored: Option<&Stored>) -> (Vec<Change>, Settled) {
        let held = stored.and_then(|stored| self.plan.hash(stored));
        let is = |hash: &str| held.as_deref() == Some(hash);
        let mine = match entry.mark {
            Mark::Stored => is(&entry.hash),
            Mark::Unknown => self.sweeps() && is(&entry.hash),
            Mark::Withheld => self.sweeps() && entry.found.as_deref().is_some_and(is),
            Mark::Foreign | Mark::Theirs => false,
        };
        match stored {
            Some(stored) if mine => (self.delete(stored, entry), Settled::Deleted),
            Some(_) => (self.renoting(self.left(entry), Some(entry)), Settled::Left),
            None => (self.dropping(entry, self.left(entry)), Settled::Same),
        }
    }

    /// The entry of an object left in the collection though Weaviate
    /// does not hold it: saying so, and saying it is another writer's if
    /// it was withheld or unknown and that is what the run is told.
    fn left(&self, entry: &Entry) -> Entry {
        match Self::unsettled(entry) && self.options.other_writers {
            true => entry.theirs(entry.updated, true),
            false => Entry {
                gone: true,
                ..entry.clone()
            },
        }
    }

    /// What becomes of an entry whose object neither Weaviate nor the
    /// collection holds, and which would otherwise be `kept` so:
    /// forgotten, there being nothing left for it to say anything of;
    /// unless the object was said to be another writer's, whose deleting
    /// of it is then not undone if Weaviate comes to hold the id.
    fn dropping(&self, entry: &Entry, kept: Entry) -> Vec<Change> {
        match kept.mark {
            Mark::Theirs => self.renoting(kept, Some(entry)),
            _ => vec![self.forget(entry)],
        }
    }

    /// What is to be done about an entry of an id Weaviate did not list.
    /// One whose object was found gone before and left, or was there
    /// when the collection was taken up, or is another writer's, is not
    /// asked for again: it is kept, unless what was there at the taking
    /// up is now to be deleted. Told that something else wrote to the
    /// collection, whatever was withheld or unknown is kept as another
    /// writer's, at once and however much of it there is: nothing of it
    /// is ever deleted, so there is nothing to ask Weaviate. What was
    /// said to be another writer's on an earlier run is kept likewise,
    /// its entry saying from now on that Weaviate does not hold it. Any
    /// other may be gone, and Weaviate is asked.
    fn unlisted(&self, entry: Entry) -> Unlisted {
        let pre = entry.mark == Mark::Unknown || (entry.mark == Mark::Withheld && entry.gone);
        let theirs = Self::unsettled(&entry) && self.options.other_writers;
        match () {
            _ if theirs => Unlisted::Keep {
                now: Some(entry.theirs(entry.updated, true)),
                entry,
            },
            _ if pre && self.sweeps() => Unlisted::Ask(entry),
            _ if entry.mark == Mark::Theirs && !entry.gone => Unlisted::Keep {
                now: Some(entry.theirs(entry.updated, true)),
                entry,
            },
            _ if pre || entry.gone => Unlisted::Keep { entry, now: None },
            _ => Unlisted::Ask(entry),
        }
    }

    /// See to entries kept though Weaviate did not list their ids, by
    /// whether the collection `holds` each object still: count it, as
    /// unknown or as left, and write the entry anew if it is to say
    /// something else; or, the object gone from the collection too,
    /// forget it, unless it bars the id for being another writer's.
    fn keeping(
        &self,
        entry: &Entry,
        now: Option<&Entry>,
        holds: bool,
        copied: &mut Copied,
    ) -> Vec<Change> {
        let kept = now.unwrap_or(entry);
        if !holds {
            copied.barred += u64::from(kept.mark == Mark::Theirs);
            return self.dropping(entry, kept.clone());
        }
        match Self::unsettled(kept) {
            true => copied.unknown += 1,
            false => copied.left += 1,
        }
        self.renoting(kept.clone(), Some(entry))
    }

    /// See to what [`Self::unlisted`] says of the entries, all of ids
    /// after `from`: those to ask Weaviate about go with the rest that
    /// may be gone, and those kept are counted, rewritten or forgotten
    /// by whether the collection still holds their objects.
    async fn unlist(
        &self,
        from: Option<ObjectId>,
        unlisted: Vec<Entry>,
        gone: &mut Gone,
        copied: &mut Copied,
    ) -> Result<()> {
        let mut kept = Vec::new();
        for entry in unlisted {
            match self.unlisted(entry) {
                Unlisted::Ask(entry) => gone.push(entry),
                Unlisted::Keep { entry, now } => kept.push((entry, now)),
            }
        }
        let Some(through) = kept.last().map(|(entry, _)| entry.id) else {
            return Ok(());
        };
        let ids: Vec<ObjectId> = kept.iter().map(|(entry, _)| entry.id).collect();
        let held = self.present(from, through, &ids).await?;
        let mut changes = Vec::new();
        for (entry, now) in &kept {
            let holds = held.contains(&entry.id);
            changes.extend(self.keeping(entry, now.as_ref(), holds, copied));
        }
        let (forgotten, noted) = changes
            .into_iter()
            .partition(|change| matches!(change, Change::Delete { .. }));
        gone.forgotten.extend::<Vec<Change>>(forgotten);
        self.noted(noted).await?;
        // No more are held than a page of them: Weaviate is checked, as
        // at the end of the walk, and they are forgotten.
        if gone.forgotten.len() >= self.options.page {
            self.forget_unlisted(gone).await?;
        }
        Ok(())
    }

    /// Forget the entries of objects neither Weaviate, by its listing,
    /// nor the collection holds, once Weaviate says it is ready and the
    /// class the one imported.
    async fn forget_unlisted(&self, gone: &mut Gone) -> Result<()> {
        self.weaviate
            .steady(&self.plan.class, &self.plan.binding)
            .await?;
        self.noted(std::mem::take(&mut gone.forgotten)).await
    }

    /// Write ledger entries that do not hang on one another, no more of
    /// them to a request than one may carry. Each is written only if the
    /// ledger still has what was read, or still nothing: refused, it
    /// was written by something else meanwhile, and the run stops rather
    /// than go on beside it.
    async fn noted(&self, notes: Vec<Change>) -> Result<()> {
        let name = &self.plan.ledger_name;
        for notes in notes.chunks(self.options.batch_changes.max(1)) {
            match self.tenant.apply(notes.to_vec()).await {
                Ok(_) => {}
                Err(error) if matches!(refusal(&error), Some(Refusal::Overtaken)) => {
                    return Err(error).wrap_err_with(|| {
                        format!("{name} changed under the import: {OTHER_IMPORT}")
                    });
                }
                Err(error) if refusal(&error) == Some(Refusal::Old) => {
                    return Err(error).wrap_err(OLD_NODE)
                }
                Err(error) => return Err(error).wrap_err_with(|| format!("writing {name}")),
            }
        }
        Ok(())
    }

    /// Which of `ids`, all after `from` and none after `through`, the
    /// collection holds an object under: asked one at a time where they
    /// are few, and as the range they lie in where they are many.
    async fn present(
        &self,
        from: Option<ObjectId>,
        through: ObjectId,
        ids: &[ObjectId],
    ) -> Result<Vec<ObjectId>> {
        let mut present = Vec::new();
        if ids.len() < SPARSE {
            for id in ids {
                present.extend(self.plan.records.get(*id).await?.map(|stored| stored.id));
            }
            return Ok(present);
        }
        let mut page = Page::new(u32::try_from(self.options.page).unwrap_or(u32::MAX));
        page.after = from;
        loop {
            let listed = self.plan.records.list(&page).await?;
            let past = listed.objects.last().is_none_or(|last| last.id >= through);
            let held = listed.objects.iter().map(|stored| stored.id);
            present.extend(held.filter(|id| ids.contains(id)));
            match listed.next {
                Some(next) if !past => page.after = Some(next),
                _ => return Ok(present),
            }
        }
    }

    /// Settle one object: the item Weaviate holds now, or an id Weaviate
    /// has been found not to hold. A step refused because
    /// the object or its entry is not as it was read is made again from
    /// a new reading, and after enough of that the object is left to
    /// whoever keeps writing it. Refused though both read as before, the
    /// refusal is of something else, and stops the run.
    async fn settle(
        &self,
        id: ObjectId,
        now: Now<'_>,
        mut entry: Option<Entry>,
        adopted: bool,
    ) -> Result<Settled> {
        let mut before = None;
        for attempt in 0..ATTEMPTS {
            if attempt > 0 {
                let again = self.entry(id).await?;
                Self::alone(id, entry.as_ref(), again.as_ref())?;
                entry = again;
            }
            let reads = match now {
                Now::Holds(item) => Self::reads(item, entry.as_ref()),
                Now::Gone => true,
            };
            let stored = match reads {
                true => self.plan.records.get(id).await?,
                false => None,
            };
            Self::versioned(&stored)?;
            let (changes, settled) = match (now, &entry) {
                (Now::Holds(item), entry) => {
                    self.settling(item, entry.as_ref(), stored.as_ref(), adopted)
                }
                (Now::Gone, None) => return Ok(Settled::Same),
                (Now::Gone, Some(entry)) => self.removing(entry, stored.as_ref()),
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
                // Nothing a run is told lets it go on past a node too
                // old to hold a write to a version.
                (Some(Refusal::Old), error) => return Err(error).wrap_err(OLD_NODE),
                (Some(Refusal::TooLarge), error) | (Some(Refusal::Unfit(_)), error)
                    if self.options.skip_refused && matches!(now, Now::Holds(_)) =>
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
        let options = self.options;
        let batches = batches(
            steps,
            options.page,
            options.batch_changes,
            options.batch_bytes,
        );
        for batch in batches {
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
                Some(error) if refusal(&error) == Some(Refusal::Old) => {
                    return Err(error).wrap_err(OLD_NODE)
                }
                Some(error) if refusal(&error).is_some() => {
                    for step in batch {
                        let id = step.item.id;
                        let entry = self.entry(id).await?;
                        if let Some(noted) = &step.noted {
                            Self::alone(id, noted.as_ref(), entry.as_ref())?;
                        }
                        let settled = self.settle(id, Now::Holds(&step.item), entry, adopted);
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
        // The ledger was not asked: a page stored before a run was
        // stopped has its entries, and is settled an object at a time.
        let step = |item: Item| Step {
            changes: self.fresh(&item, None),
            settled: Settled::Stored,
            noted: None,
            item,
        };
        let steps = items.into_iter().map(step).collect();
        self.make(steps, false, copied).await.map(drop)
    }

    /// Take up a page of a collection that had no ledger: give each
    /// object of Weaviate's its entry, by what the collection holds
    /// under its id. `from` is the last id of the page before, `through`
    /// that of this one, or nothing past Weaviate's last. On the first
    /// taking up of a collection nothing kept a ledger of, what it holds
    /// there that Weaviate does not is noted too, as it is then: only
    /// what is so noted can ever be deleted for having been there.
    async fn adopt(
        &self,
        from: Option<ObjectId>,
        through: Option<ObjectId>,
        items: Vec<Item>,
        entries: &mut Entries<'_>,
        copied: &mut Copied,
    ) -> Result<()> {
        let mut known = HashMap::new();
        loop {
            let entry = match through {
                Some(through) => entries.through(through).await?,
                None => entries.next().await?,
            };
            let Some(entry) = entry else { break };
            known.insert(entry.id, entry);
        }
        let ids: Vec<ObjectId> = items.iter().map(|item| item.id).collect();
        let mut held = HashMap::new();
        let mut unknown = Vec::new();
        let mut page = Page::new(u32::try_from(self.options.page).unwrap_or(u32::MAX));
        page.vectors = self.plan.vectors();
        page.after = from;
        loop {
            let listed = self.plan.records.list(&page).await?;
            let past = |id: ObjectId| through.is_some_and(|through| id > through);
            let ended = listed.objects.last().is_none_or(|last| past(last.id));
            for stored in listed.objects {
                if ids.contains(&stored.id) {
                    held.insert(stored.id, stored);
                } else if self.plan.binding.census
                    && !past(stored.id)
                    && !known.contains_key(&stored.id)
                {
                    unknown.push(self.census(&stored));
                }
            }
            if unknown.len() >= self.options.page {
                self.note_unknown(std::mem::take(&mut unknown)).await?;
            }
            match listed.next {
                Some(next) if !ended => page.after = Some(next),
                _ => break,
            }
        }
        if !unknown.is_empty() {
            self.note_unknown(unknown).await?;
        }
        Self::versioned(held.values())?;
        let step = |item: Item| {
            let entry = known.remove(&item.id);
            let stored = held.remove(&item.id);
            let (changes, settled) = self.settling(&item, entry.as_ref(), stored.as_ref(), true);
            Step {
                item,
                noted: Some(entry),
                changes,
                settled,
            }
        };
        let steps = items.into_iter().map(step).collect();
        self.make(steps, true, copied).await.map(drop)
    }

    /// Note what the collection holds that Weaviate does not. Each note
    /// is made only where the ledger has none: if it has one after all,
    /// something else is writing the ledger while the collection is
    /// taken up, and the run stops rather than go on beside it.
    async fn note_unknown(&self, notes: Vec<Change>) -> Result<()> {
        let name = &self.plan.schema.name;
        self.noted(notes)
            .await
            .wrap_err_with(|| format!("noting what collection {name} holds that Weaviate does not"))
    }

    /// Note an object the collection holds that Weaviate does not, only
    /// where the ledger has no entry for it: one it has was written by
    /// another run, and is not written over.
    fn census(&self, stored: &Stored) -> Change {
        self.note(&self.unknown(stored), None)
    }

    /// The entry of an object the collection holds that Weaviate does
    /// not, as it is when the collection is taken up. One the collection
    /// as this run knows it could not have been sent hashes as nothing
    /// does, and is never deleted.
    fn unknown(&self, stored: &Stored) -> Entry {
        Entry {
            id: stored.id,
            at: None,
            updated: None,
            hash: self.plan.hash(stored).unwrap_or_default(),
            mark: Mark::Unknown,
            found: None,
            gone: false,
        }
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

    /// How many objects a run may delete of a class whose ledger has
    /// `ledgered` entries and whose collection holds `held` objects: the
    /// fewer of a count and a part of the lesser of the two, and, unless
    /// that part is all of it, never the last object the collection
    /// holds.
    fn budget(&self, ledgered: u64, held: u64) -> u64 {
        match self.options.deletes {
            Deletes::None => 0,
            Deletes::UpTo { count, fraction } => {
                let part = (ledgered.min(held) as f64 * fraction).ceil() as u64;
                let all = if fraction < 1.0 { 1 } else { 0 };
                count.min(part).min(held.saturating_sub(all))
            }
        }
    }

    /// Walk Weaviate's ids beside the ledger's, both in id order, and
    /// settle each object the ledger does not have as Weaviate lists it,
    /// each changed since `since`, and each that was withheld if the
    /// collection is now said to have had no other writer, or to have
    /// had one; then see to what the ledger has that Weaviate did not
    /// list. Whether every one of those was seen to: if not, the
    /// catch-up is to be made again from where the last one began.
    async fn catch_up(&self, since: i64, copied: &mut Copied) -> Result<bool> {
        let class = &self.plan.class;
        let ledgered = self.plan.ledger.count(None).await?;
        let held = self.plan.records.count(None).await?;
        let mut gone = Gone {
            entries: Vec::new(),
            count: 0,
            foreign: 0,
            withheld: 0,
            budget: self.budget(ledgered, held),
            forgotten: Vec::new(),
        };
        let mut entries = Entries::new(&self.plan.ledger, self.options.page, None);
        let mut pacer = Pacer::new(self.options.per_second);
        let mut cursor: Option<(String, ObjectId)> = None;
        let said = self.options.sole_writer || self.options.other_writers;
        loop {
            let from = cursor.clone();
            let from_id = from.as_ref().map(|(_, id)| *id);
            let listing = from.as_ref().map(|(id, _)| id.as_str());
            let page = self
                .weaviate
                .page(class, listing, self.options.page, false)
                .await?;
            if page.is_empty() {
                break;
            }
            let mut wanted = Vec::new();
            let mut unlisted = Vec::new();
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
                while let Some(entry) = entries.before(id).await? {
                    unlisted.push(entry);
                }
                let entry = entries.at(id).await?;
                let mark = entry.as_ref().map(|entry| entry.mark);
                // An entry of an object Weaviate did not hold, or was
                // found not to, is of another object now.
                let noted = entry
                    .as_ref()
                    .is_some_and(|entry| entry.listed(seen.updated));
                let owed = mark == Some(Mark::Withheld) && said;
                if !noted || owed || seen.updated.unwrap_or(i64::MAX) >= since {
                    wanted.push((id, entry));
                } else {
                    let theirs = matches!(mark, Some(Mark::Foreign | Mark::Theirs));
                    copied.kept += u64::from(theirs);
                    copied.withheld += u64::from(mark == Some(Mark::Withheld));
                }
            }
            let Some((_, through)) = cursor.clone() else {
                break;
            };
            self.unlist(from_id, unlisted, &mut gone, copied).await?;
            pacer.paced(page.len() + wanted.len()).await;
            let listing = (listing, from_id, through, page.len());
            self.revisit(listing, wanted, &mut gone, copied).await?;
        }
        let mut from = cursor.map(|(_, id)| id);
        loop {
            let mut unlisted = Vec::new();
            while unlisted.len() < self.options.page {
                let Some(entry) = entries.next().await? else {
                    break;
                };
                unlisted.push(entry);
            }
            let Some(last) = unlisted.last().map(|entry| entry.id) else {
                break;
            };
            self.unlist(from, unlisted, &mut gone, copied).await?;
            from = Some(last);
        }
        // A class made again while it was walked is not gone on from, and
        // one that is not ready is not asked what it does not hold.
        self.forget_unlisted(&mut gone).await?;
        self.remove(gone, copied).await
    }

    /// Fetch what Weaviate holds now under the ids `wanted` of one page
    /// it listed, and settle each. One it listed and no longer has when
    /// asked for is as one it did not list, if the ledger has it.
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
        // What was fetched is stored only if it came of the class that
        // was imported: one made again meanwhile holds other objects
        // under the same ids.
        self.weaviate
            .steady(&self.plan.class, &self.plan.binding)
            .await?;
        let mut settling = Vec::new();
        let mut unlisted = Vec::new();
        for (id, entry) in wanted {
            match (fetched.remove(&id), entry) {
                (Some(Some(item)), entry) => settling.push((item, entry)),
                (Some(None), _) | (None, None) => {}
                (None, Some(entry)) => unlisted.push(entry),
            }
        }
        self.unlist(from, unlisted, gone, copied).await?;
        let reads = |(item, entry): &(Item, Option<Entry>)| Self::reads(item, entry.as_ref());
        let read: Vec<ObjectId> = settling
            .iter()
            .filter(|s| reads(s))
            .map(|(item, _)| item.id)
            .collect();
        let mut held = self.held(from, through, &read, span).await?;
        Self::versioned(held.values())?;
        let step = |(item, entry): (Item, Option<Entry>)| {
            let stored = held.remove(&item.id);
            let (changes, settled) = self.settling(&item, entry.as_ref(), stored.as_ref(), false);
            Step {
                item,
                noted: Some(entry),
                changes,
                settled,
            }
        };
        let steps = settling.into_iter().map(step).collect();
        let made = self.make(steps, false, copied).await?;
        let marked = |mark: Settled| made.iter().filter(|(_, settled)| *settled == mark).count();
        copied.kept += marked(Settled::Foreign) as u64;
        copied.withheld += marked(Settled::Withheld) as u64;
        Ok(())
    }

    /// See to what the ledger has that Weaviate did not list, now that
    /// the walk is over. More than the run may delete, none is: the
    /// answer is then false, and they are reported as pending.
    /// Otherwise each is asked for twice, a pause apart, with Weaviate
    /// ready and the class as it was imported, as whoever calls this has
    /// just seen it to be, and only one Weaviate twice says it does not
    /// hold is deleted, if it is the import's. One it does hold after
    /// all is settled as any other.
    async fn remove(&self, gone: Gone, copied: &mut Copied) -> Result<bool> {
        if gone.count == 0 {
            return Ok(true);
        }
        if gone.count > gone.budget {
            copied.pending = gone.count;
            copied.kept += gone.foreign;
            copied.withheld += gone.withheld;
            let chosen = self.options.deletes == Deletes::None;
            copied.budget = (!chosen).then_some(gone.budget);
            return Ok(chosen);
        }
        let class = &self.plan.class;
        let mut absent = Vec::new();
        let mut present = Vec::new();
        for entry in gone.entries {
            match self.weaviate.held_by_all(class, entry.id).await? {
                true => present.push(entry),
                false => absent.push(entry),
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
                match self.weaviate.held_by_all(class, entry.id).await? {
                    true => present.push(entry.clone()),
                    false => confirmed.push(entry.clone()),
                }
            }
            self.weaviate.steady(class, &self.plan.binding).await?;
            for entry in confirmed {
                let id = entry.id;
                let settled = self.settle(id, Now::Gone, Some(entry), false);
                copied.count(id, settled.await?);
            }
        }
        for entry in present {
            let id = entry.id;
            let object = self
                .weaviate
                .object(class, id, self.plan.vectors(), Asked::Any)
                .await?;
            // It is of the class that was imported, or it is not stored.
            self.weaviate.steady(class, &self.plan.binding).await?;
            let Some(item) = self.items(object.as_slice(), copied)?.pop() else {
                continue;
            };
            let settled = self
                .settle(id, Now::Holds(&item), Some(entry), false)
                .await?;
            copied.kept += u64::from(settled == Settled::Foreign);
            copied.withheld += u64::from(settled == Settled::Withheld);
            copied.count(id, settled);
        }
        Ok(true)
    }

    /// Compare what the collection holds with what Weaviate does, object
    /// by object, changing nothing: over everything Weaviate holds, or
    /// over `pages` pages of the ledger picked at random, and what
    /// Weaviate and the collection hold of the ids on them. A ledger's
    /// pages are picked, not places among all ids there could be, so
    /// that ids that lie close together are looked at as any others.
    /// What the ledger says was withheld, left or unknown is counted in
    /// `copied`, of the whole ledger either way, as a run that copies
    /// counts it: by whether the collection holds it still. Answers too
    /// with how many entries such a run would forget.
    async fn verify(&self, pages: Option<usize>, copied: &mut Copied) -> Result<(Verified, u64)> {
        let mut verified = Verified::default();
        let ledgered = self.plan.ledger.count(None).await? as usize;
        let all = ledgered.div_ceil(self.options.page);
        let random = || u128::from_le_bytes(*ObjectId::generate().0.as_bytes()) as usize;
        let mut picked = match pages {
            Some(pages) => sample(all, pages, random).into_iter().peekable(),
            None => Vec::new().into_iter().peekable(),
        };
        let mut entries = Entries::new(&self.plan.ledger, self.options.page, None);
        let (mut from, mut nth, mut on_page) = (None, 0, 0);
        let (mut left, mut stale) = (Vec::new(), 0);
        while let Some(entry) = entries.next().await? {
            let (id, mark, gone) = (entry.id, entry.mark, entry.gone);
            copied.kept += u64::from(mark == Mark::Theirs && !gone);
            copied.withheld += u64::from(mark == Mark::Withheld && !gone);
            if gone || mark == Mark::Unknown {
                left.push(entry);
            }
            on_page += 1;
            if on_page == self.options.page {
                stale += self.leaving(from, &mut left, copied).await?;
                if picked.next_if_eq(&nth).is_some() {
                    self.compare(from, Some(id), &mut verified).await?;
                }
                (from, nth, on_page) = (Some(id), nth + 1, 0);
            }
        }
        stale += self.leaving(from, &mut left, copied).await?;
        if picked.next_if_eq(&nth).is_some() {
            self.compare(from, None, &mut verified).await?;
        }
        if pages.is_none() {
            self.compare(None, None, &mut verified).await?;
            verified.extra = Some(self.extra().await?);
        }
        Ok((verified, stale))
    }

    /// Count the entries `left`, all of ids after `from`, that are of
    /// nothing Weaviate holds: each as a run that copies counts it, by
    /// whether the collection holds its object still. Answers with how
    /// many of them such a run forgets.
    async fn leaving(
        &self,
        from: Option<ObjectId>,
        left: &mut Vec<Entry>,
        copied: &mut Copied,
    ) -> Result<u64> {
        let Some(through) = left.last().map(|entry| entry.id) else {
            return Ok(0);
        };
        let ids: Vec<ObjectId> = left.iter().map(|entry| entry.id).collect();
        let held = self.present(from, through, &ids).await?;
        let stale = |entry: Entry| self.stale(&entry, held.contains(&entry.id), copied);
        Ok(left.drain(..).map(stale).map(u64::from).sum())
    }

    /// Count an entry kept though Weaviate does not hold its id as a run
    /// that copies does, and say whether that run forgets it: it then
    /// accounts for nothing, and is not counted at all.
    fn stale(&self, entry: &Entry, holds: bool, copied: &mut Copied) -> bool {
        let kept = self.keeping(entry, None, holds, copied);
        kept.iter()
            .any(|change| matches!(change, Change::Delete { .. }))
    }

    /// How many objects the collection holds that the ledger has no
    /// entry for: another writer's, under ids Weaviate never had.
    async fn extra(&self) -> Result<u64> {
        let limit = u32::try_from(self.options.page).unwrap_or(u32::MAX);
        let mut entries = Entries::new(&self.plan.ledger, self.options.page, None);
        let mut page = Page::new(limit);
        let mut extra = 0;
        loop {
            let listed = self.plan.records.list(&page).await?;
            for stored in &listed.objects {
                while entries.before(stored.id).await?.is_some() {}
                extra += u64::from(entries.at(stored.id).await?.is_none());
            }
            match listed.next {
                Some(next) => page.after = Some(next),
                None => return Ok(extra),
            }
        }
    }

    /// Compare the objects Weaviate holds after `from` and no later than
    /// `through`, or to the end, with what the collection holds, and
    /// with what the ledger says of them.
    async fn compare(
        &self,
        from: Option<ObjectId>,
        through: Option<ObjectId>,
        verified: &mut Verified,
    ) -> Result<()> {
        const EXAMPLES: usize = 20;
        let options = Options {
            skip_refused: true,
            ..self.options.clone()
        };
        let reading = Copying::new(self.weaviate, self.tenant, self.plan, &options);
        let mut unreadable = Copied::default();
        let mut entries = Entries::new(&self.plan.ledger, self.options.page, from);
        let mut cursor = from;
        let within = |id: ObjectId| through.is_none_or(|through| id <= through);
        // An entry of an id Weaviate does not list: gone, if the ledger
        // takes Weaviate to hold it.
        let unlisted = |entry: Entry, verified: &mut Verified| {
            let kept = matches!(entry.mark, Mark::Unknown | Mark::Theirs) || entry.gone;
            verified.gone += u64::from(!kept);
            verified.looked += 1;
        };
        loop {
            let after = cursor.map(|id| id.to_string());
            let page = self
                .weaviate
                .page(
                    &self.plan.class,
                    after.as_deref(),
                    self.options.page,
                    self.plan.vectors(),
                )
                .await?;
            let mut items = reading.items(&page, &mut unreadable)?;
            let listed = page.len();
            items.retain(|item| within(item.id));
            let Some(last) = items.last().map(|item| item.id) else {
                break;
            };
            let ids: Vec<ObjectId> = items.iter().map(|item| item.id).collect();
            let held = self.held(cursor, last, &ids, ids.len()).await?;
            for item in items {
                while let Some(entry) = entries.before(item.id).await? {
                    unlisted(entry, verified);
                }
                let entry = entries.at(item.id).await?;
                let stored = held.get(&item.id);
                let hash = stored.and_then(|stored| self.plan.hash(stored));
                let found = match (entry.map(|entry| entry.mark), stored) {
                    _ if hash.as_ref() == Some(&item.hash) => &mut verified.same,
                    (None, _) => &mut verified.unaccounted,
                    (Some(Mark::Foreign | Mark::Theirs), _) => &mut verified.kept,
                    (Some(Mark::Withheld | Mark::Unknown), _) => &mut verified.withheld,
                    (Some(Mark::Stored), None) => &mut verified.missing,
                    (Some(Mark::Stored), Some(_)) => &mut verified.different,
                };
                *found += 1;
                verified.looked += 1;
                let wrong = (verified.missing + verified.different) as usize;
                let shown = verified.examples.len();
                if shown < EXAMPLES && wrong > shown {
                    verified.examples.push(item.id.to_string());
                }
            }
            cursor = Some(last);
            if listed < self.options.page || through.is_some_and(|through| last >= through) {
                break;
            }
        }
        // What the ledger has of the range that Weaviate did not list.
        loop {
            let entry = match through {
                Some(through) => entries.through(through).await?,
                None => entries.next().await?,
            };
            let Some(entry) = entry else { break };
            unlisted(entry, verified);
        }
        let unreadable = unreadable
            .refused
            .keys()
            .filter_map(|id| object_id(id).ok());
        let unreadable = unreadable.filter(|id| within(*id)).count() as u64;
        verified.unaccounted += unreadable;
        verified.looked += unreadable;
        Ok(())
    }
}

/// Which `pages` of `all` to look at, in order: each as likely as any
/// other, by what `random` gives.
fn sample(all: usize, pages: usize, mut random: impl FnMut() -> usize) -> Vec<usize> {
    let mut picked: Vec<usize> = (0..all).collect();
    for kept in 0..pages.min(all) {
        picked.swap(kept, kept + random() % (all - kept));
    }
    picked.truncate(pages);
    picked.sort_unstable();
    picked
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
        let (status, text, _) = self.answered(what, build).await?;
        Ok((status, text))
    }

    /// As [`Self::answer`], and whether the answer came at the first
    /// asking.
    async fn answered(
        &self,
        what: &str,
        build: impl Fn(&reqwest::Client) -> reqwest::RequestBuilder,
    ) -> Result<(reqwest::StatusCode, String, bool)> {
        let began = Instant::now();
        let mut wait = Duration::from_millis(500);
        let mut at_once = true;
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
                        Ok(text) => return Ok((status, text, at_once)),
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
            at_once = false;
        }
    }

    /// Wait until Weaviate answers that it is ready, and say whether it
    /// did at the first asking. Until it is, it answers that it holds no
    /// such class or object for those it holds.
    async fn ready(&self) -> Result<bool> {
        let url = format!("{}/v1/.well-known/ready", self.url);
        let asking = self.answered("whether it is ready", |http| http.get(&url));
        let (status, _, at_once) = asking.await?;
        if !status.is_success() {
            bail!("Weaviate answered whether it is ready with {status}");
        }
        Ok(at_once)
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
        let began = Instant::now();
        loop {
            if let Some(page) = self.get("/v1/objects", &query).await? {
                return serde_json::from_value(page["objects"].clone())
                    .wrap_err_with(|| format!("reading a page of {}", class.class));
            }
            // It has no such class, unless it was not ready to say.
            if self.ready().await? || began.elapsed() > self.patience {
                bail!("Weaviate has no class {}", class.class);
            }
        }
    }

    /// Whether every copy of `class` holds an object under `id`. That
    /// none does is believed only of a Weaviate that says it is ready
    /// straight after: one that was not is asked again.
    async fn held_by_all(&self, class: &Class, id: ObjectId) -> Result<bool> {
        let began = Instant::now();
        loop {
            if self.object(class, id, false, Asked::All).await?.is_some() {
                return Ok(true);
            }
            if self.ready().await? {
                return Ok(false);
            }
            if began.elapsed() > self.patience {
                bail!(
                    "Weaviate was not ready to say whether it holds {id} of {}; nothing was deleted",
                    class.class
                );
            }
        }
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
    /// `binding` says was imported, so that nothing of another class is
    /// stored and nothing deleted for it. A Weaviate that is starting answers
    /// that it holds no such object for objects it holds.
    async fn steady(&self, class: &Class, binding: &Binding) -> Result<()> {
        let began = Instant::now();
        loop {
            self.ready().await?;
            let Some(shards) = self.shards(class).await? else {
                bail!(
                    "Weaviate no longer has class {}; its objects stay in the collection",
                    class.class
                );
            };
            // A class with no shard listed is not one whose shards are
            // all loaded.
            if !shards.is_empty() && shards.iter().all(|(_, serving)| *serving) {
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
                census: false,
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
            // A version of its own for each: no node made these.
            let version = title
                .bytes()
                .fold(1_u128, |all, byte| all * 257 + byte as u128);
            serde_json::from_value(json!({
                "id": item.id,
                "properties": item.content.properties,
                "vector": item.content.vector,
                "version": format!("{version:032x}"),
            }))
            .unwrap()
        }

        /// An entry as it is read from the ledger, at a version.
        fn entry(&self, title: &str, updated: i64, mark: Mark) -> Entry {
            Entry {
                at: self.stored(title).version,
                ..Entry::of(&self.item(title, updated), mark)
            }
        }
    }

    /// What kind each change is, and of which collection; for one to the
    /// ledger, with the mark it gives the entry.
    fn kinds(changes: &[Change]) -> Vec<String> {
        let marked = |properties: &Map<String, Value>| {
            let marks = ["foreign", "withheld", "unknown", "theirs", "found", "gone"];
            let said = marks.iter().filter(|mark| properties.contains_key(**mark));
            said.map(|mark| format!(" {mark}")).collect::<String>()
        };
        let kind = |change: &Change| match change {
            Change::Insert {
                collection,
                properties,
                ..
            } => format!("insert {collection}{}", marked(properties)),
            Change::Upsert {
                collection,
                properties,
                ..
            } => format!("upsert {collection}{}", marked(properties)),
            Change::Delete { collection, .. } => format!("delete {collection}"),
            other => format!("{other:?}"),
        };
        changes.iter().map(kind).collect()
    }

    const ENTRY: &str = "vellev_imported_Memory";
    /// How many cells the table of everything a run can meet has: those
    /// where what a run may do is checked; those where what a later run
    /// may do to what was said to be another writer's is; those among
    /// the first where a run must do something; and those where the last
    /// guard on deleting is asked by itself.
    const SAFETY: usize = 1_380;
    const LATER: usize = 6_400;
    const LIVENESS: usize = 404;
    const DIRECT: usize = 440;

    fn only(changes: &[&str]) -> Vec<String> {
        let changes = changes.iter();
        changes
            .map(|change| change.replace("ENTRY", ENTRY))
            .collect()
    }

    #[test]
    fn an_object_is_stored_replaced_or_left_by_what_the_ledger_and_the_collection_hold() {
        // Nothing else having written to the collection changes none of
        // this: that is said of what was withheld, and of nothing else.
        for sole_writer in [false, true] {
            let fixture = Fixture::new(Options {
                sole_writer,
                ..options()
            });
            let copying = fixture.copying();
            let hen = fixture.item("hen", 2);
            let ours = fixture.entry("fox", 1, Mark::Stored);
            let settling = |entry: Option<&Entry>, stored: Option<&Stored>| {
                let (changes, settled) = copying.settling(&hen, entry, stored, false);
                (kinds(&changes), settled)
            };

            // New to Weaviate: stored, if neither is there to be written
            // over.
            assert_eq!(
                settling(None, None),
                (only(&["insert Memory", "insert ENTRY"]), Settled::Stored)
            );
            // The import's, changed in Weaviate: replaced, unless a
            // property of it changes first, one without a value among
            // them.
            assert_eq!(
                settling(Some(&ours), Some(&fixture.stored("fox"))),
                (only(&["upsert Memory", "upsert ENTRY"]), Settled::Replaced)
            );
            // Another writer's, deleted by one, or never the import's:
            // left, and the ledger says so.
            let foreign = (only(&["upsert ENTRY foreign"]), Settled::Foreign);
            assert_eq!(settling(Some(&ours), Some(&fixture.stored("owl"))), foreign);
            assert_eq!(settling(Some(&ours), None), foreign);
            assert_eq!(
                settling(None, Some(&fixture.stored("owl"))),
                (only(&["insert ENTRY foreign"]), Settled::Foreign)
            );
            // As Weaviate has it already, whoever stored it: the
            // import's.
            assert_eq!(
                settling(None, Some(&fixture.stored("hen"))),
                (only(&["insert ENTRY"]), Settled::Same)
            );
            let marked = fixture.entry("fox", 1, Mark::Foreign);
            assert_eq!(
                settling(Some(&marked), Some(&fixture.stored("hen"))),
                (only(&["upsert ENTRY"]), Settled::Same)
            );
            // Once the ledger says an object is another writer's, it is
            // not said again, and the object is never written over.
            let said = fixture.entry("hen", 2, Mark::Foreign);
            assert_eq!(
                settling(Some(&said), Some(&fixture.stored("owl"))),
                (vec![], Settled::Foreign)
            );
            assert_eq!(settling(Some(&said), None), (vec![], Settled::Foreign));
        }
    }

    #[test]
    fn an_object_unchanged_is_not_read_and_only_its_time_is_noted_anew() {
        let fixture = Fixture::new(options());
        let copying = fixture.copying();
        let hen = fixture.item("hen", 2);
        let noted = fixture.entry("hen", 2, Mark::Stored);
        let earlier = fixture.entry("hen", 1, Mark::Stored);
        assert!(!Copying::reads(&hen, Some(&noted)));
        assert!(Copying::reads(&hen, None));
        let changed = fixture.entry("fox", 2, Mark::Stored);
        assert!(Copying::reads(&hen, Some(&changed)));
        for mark in [Mark::Foreign, Mark::Withheld] {
            assert!(Copying::reads(&hen, Some(&fixture.entry("hen", 2, mark))));
        }
        let (changes, settled) = copying.settling(&hen, Some(&noted), None, false);
        assert_eq!((kinds(&changes), settled), (vec![], Settled::Same));
        let (changes, settled) = copying.settling(&hen, Some(&earlier), None, false);
        assert_eq!(kinds(&changes), only(&["upsert ENTRY"]));
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
    fn a_collection_taken_up_has_what_it_lacks_or_holds_otherwise_withheld() {
        let fixture = Fixture::new(options());
        let hen = fixture.item("hen", 2);
        let taken = |stored: Option<&Stored>| {
            let (changes, settled) = fixture.copying().settling(&hen, None, stored, true);
            (kinds(&changes), settled)
        };
        // Without a ledger nothing tells an object new to Weaviate from
        // one deleted in vellev, nor one changed in Weaviate from one
        // changed in vellev. What the collection held is noted.
        assert_eq!(
            taken(None),
            (only(&["insert ENTRY withheld"]), Settled::Withheld)
        );
        let fox = fixture.stored("fox");
        let (changes, settled) = fixture.copying().settling(&hen, None, Some(&fox), true);
        assert_eq!(
            (kinds(&changes), settled),
            (only(&["insert ENTRY withheld found"]), Settled::Withheld)
        );
        let Change::Insert { properties, .. } = &changes[0] else {
            unreachable!()
        };
        assert_eq!(properties["found"], json!(fixture.item("fox", 0).hash));
        assert_eq!(
            taken(Some(&fixture.stored("hen"))),
            (only(&["insert ENTRY"]), Settled::Same)
        );

        // What was withheld stays so, whatever Weaviate does to it, with
        // what the collection held, and is not said twice.
        let settling = |entry: &Entry, stored: Option<&Stored>| {
            let (changes, settled) = fixture.copying().settling(&hen, Some(entry), stored, false);
            (kinds(&changes), settled)
        };
        let said = fixture.entry("hen", 2, Mark::Withheld);
        assert_eq!(settling(&said, None), (vec![], Settled::Withheld));
        let earlier = Entry {
            found: Some("as it was".into()),
            ..fixture.entry("owl", 1, Mark::Withheld)
        };
        let again = (only(&["upsert ENTRY withheld found"]), Settled::Withheld);
        assert_eq!(settling(&earlier, None), again);
        assert_eq!(settling(&earlier, Some(&fox)), again);
        // What the collection held that Weaviate did not, and Weaviate
        // now holds otherwise, is withheld as what it was found to be.
        // That is what the entry noted then, whatever the collection
        // holds under the id by now.
        let unknown = fixture.entry("fox", 1, Mark::Unknown);
        let owl = fixture.stored("owl");
        let (changes, settled) =
            fixture
                .copying()
                .settling(&hen, Some(&unknown), Some(&owl), false);
        assert_eq!(
            (kinds(&changes), settled),
            (only(&["upsert ENTRY withheld found"]), Settled::Withheld)
        );
        let Change::Upsert { properties, .. } = &changes[0] else {
            unreachable!()
        };
        assert_eq!(properties["found"], json!(unknown.hash));
    }

    #[test]
    fn what_was_withheld_is_stored_or_replaced_once_nothing_else_is_said_to_have_written() {
        let sole = Fixture::new(Options {
            sole_writer: true,
            ..options()
        });
        let hen = sole.item("hen", 2);
        let settling = |entry: Option<&Entry>, stored: Option<&Stored>, adopted: bool| {
            let (changes, settled) = sole.copying().settling(&hen, entry, stored, adopted);
            (kinds(&changes), settled)
        };
        let fox = sole.stored("fox");
        // While the collection is taken up.
        assert_eq!(
            settling(None, None, true),
            (only(&["insert Memory", "insert ENTRY"]), Settled::Stored)
        );
        assert_eq!(
            settling(None, Some(&fox), true),
            (only(&["upsert Memory", "insert ENTRY"]), Settled::Replaced)
        );
        // And by any run after one that withheld it, or found it there
        // unknown to Weaviate.
        for mark in [Mark::Withheld, Mark::Unknown] {
            let entry = sole.entry("owl", 1, mark);
            assert_eq!(
                settling(Some(&entry), None, false),
                (only(&["insert Memory", "upsert ENTRY"]), Settled::Stored)
            );
            assert_eq!(
                settling(Some(&entry), Some(&fox), false),
                (only(&["upsert Memory", "upsert ENTRY"]), Settled::Replaced)
            );
        }
    }

    #[test]
    fn what_was_withheld_is_another_writers_once_something_else_is_said_to_have_written() {
        let others = Fixture::new(Options {
            other_writers: true,
            ..options()
        });
        let hen = others.item("hen", 2);
        let settling = |entry: Option<&Entry>, stored: Option<&Stored>, adopted: bool| {
            let (changes, settled) = others.copying().settling(&hen, entry, stored, adopted);
            (kinds(&changes), settled)
        };
        let fox = others.stored("fox");
        let theirs = |kind: &str| (only(&[kind]), Settled::Foreign);
        assert_eq!(settling(None, None, true), theirs("insert ENTRY theirs"));
        assert_eq!(
            settling(None, Some(&fox), true),
            theirs("insert ENTRY theirs")
        );
        let withheld = Entry {
            found: Some("as it was".into()),
            ..others.entry("hen", 2, Mark::Withheld)
        };
        assert_eq!(
            settling(Some(&withheld), Some(&fox), false),
            theirs("upsert ENTRY theirs")
        );
        // What the collection holds exactly as Weaviate does is not
        // another writer's for that being said: it is the import's, met
        // while the collection is taken up or after.
        let same = others.stored("hen");
        assert_eq!(
            settling(None, Some(&same), true),
            (only(&["insert ENTRY"]), Settled::Same)
        );
        let unknown = others.entry("fox", 1, Mark::Unknown);
        for entry in [&withheld, &unknown] {
            assert_eq!(
                settling(Some(entry), Some(&same), false),
                (only(&["upsert ENTRY"]), Settled::Same)
            );
        }
        // What is new to Weaviate is still stored.
        assert_eq!(
            settling(None, None, false),
            (only(&["insert Memory", "insert ENTRY"]), Settled::Stored)
        );
    }

    #[test]
    fn only_what_the_import_stored_and_nobody_changed_is_deleted() {
        let removing = |fixture: &Fixture, entry: &Entry, stored: Option<&Stored>| {
            let (changes, settled) = fixture.copying().removing(entry, stored);
            (kinds(&changes), settled)
        };
        let fixture = Fixture::new(options());
        let ours = fixture.entry("fox", 1, Mark::Stored);
        let (fox, owl) = (fixture.stored("fox"), fixture.stored("owl"));
        let deleted = (only(&["delete Memory", "delete ENTRY"]), Settled::Deleted);
        assert_eq!(removing(&fixture, &ours, Some(&fox)), deleted);
        // What is left is never forgotten: its entry says Weaviate no
        // longer holds it, once.
        let left = |kind: &str| (only(&[kind]), Settled::Left);
        assert_eq!(
            removing(&fixture, &ours, Some(&owl)),
            left("upsert ENTRY gone")
        );
        let gone = Entry {
            gone: true,
            ..ours.clone()
        };
        assert_eq!(
            removing(&fixture, &gone, Some(&owl)),
            (vec![], Settled::Left)
        );
        let foreign = fixture.entry("hen", 1, Mark::Foreign);
        assert_eq!(
            removing(&fixture, &foreign, Some(&owl)),
            left("upsert ENTRY foreign gone")
        );
        // Not even one that is exactly what its entry hashed: the mark
        // says another writer put it there.
        let matching = fixture.entry("fox", 1, Mark::Foreign);
        assert_eq!(
            removing(&fixture, &matching, Some(&fox)),
            left("upsert ENTRY foreign gone")
        );
        assert_eq!(
            removing(&fixture, &ours, None),
            (only(&["delete ENTRY"]), Settled::Same)
        );

        // What was there when the collection was taken up is not the
        // import's to delete: not for being withheld, not for being as
        // it was found, not for nothing else being said to have written.
        let unknown = fixture.entry("fox", 1, Mark::Unknown);
        let withheld = Entry {
            found: Some(fixture.item("fox", 0).hash),
            ..fixture.entry("hen", 1, Mark::Withheld)
        };
        let sole = Fixture::new(Options {
            sole_writer: true,
            ..options()
        });
        for fixture in [&fixture, &sole] {
            assert_eq!(
                removing(fixture, &unknown, Some(&fox)),
                left("upsert ENTRY unknown gone")
            );
            assert_eq!(
                removing(fixture, &withheld, Some(&fox)),
                left("upsert ENTRY withheld found gone")
            );
        }
        // Asked in so many words, it is deleted if it is still as it was
        // found, and left if it has changed since.
        let sweeping = Fixture::new(Options {
            sole_writer: true,
            delete_unknown: true,
            ..options()
        });
        assert_eq!(removing(&sweeping, &unknown, Some(&fox)), deleted);
        assert_eq!(removing(&sweeping, &withheld, Some(&fox)), deleted);
        assert_eq!(
            removing(&sweeping, &unknown, Some(&owl)),
            left("upsert ENTRY unknown gone")
        );
        assert_eq!(
            removing(&sweeping, &withheld, Some(&owl)),
            left("upsert ENTRY withheld found gone")
        );
        // What was withheld because the collection lacked it was never
        // found as anything.
        let lacked = fixture.entry("hen", 1, Mark::Withheld);
        assert_eq!(
            removing(&sweeping, &lacked, Some(&fox)),
            left("upsert ENTRY withheld gone")
        );
        // And what another writer is known to have written never is.
        assert_eq!(
            removing(&sweeping, &foreign, Some(&owl)),
            left("upsert ENTRY foreign gone")
        );
        // Told that something else wrote, what was there at the taking
        // up is left as another writer's, whatever it is now.
        let others = Fixture::new(Options {
            other_writers: true,
            ..options()
        });
        for entry in [&unknown, &withheld, &lacked] {
            for stored in [&fox, &owl] {
                assert_eq!(
                    removing(&others, entry, Some(stored)),
                    left("upsert ENTRY theirs gone")
                );
            }
        }
    }

    #[test]
    fn a_collection_already_there_is_taken_up_by_what_is_left_of_its_import() {
        let bound = |adopting: bool, census: bool| Binding {
            adopting,
            census,
            ..Fixture::new(options()).plan.binding.clone()
        };
        // (taken up, noting what Weaviate does not hold, from the start)
        // Never imported with a ledger: taken up, and what it holds that
        // Weaviate does not noted.
        assert_eq!(taking_up(None, false), (true, true, true));
        // Its note alone lost: taken up again beside its ledger, and
        // nothing noted.
        assert_eq!(taking_up(None, true), (true, false, true));
        // Imported, and both there: caught up, whatever the ledger
        // holds.
        let imported = bound(false, false);
        assert_eq!(taking_up(Some(&imported), true), (false, false, false));
        // Its ledger lost: taken up from the start, and nothing noted.
        assert_eq!(taking_up(Some(&imported), false), (true, false, true));
        // A first taking up stopped part way goes on, noting still.
        let taking = bound(true, true);
        assert_eq!(taking_up(Some(&taking), true), (true, true, false));
        // Its ledger lost meanwhile, it begins again and notes nothing:
        // what the collection holds by then may have been written since.
        assert_eq!(taking_up(Some(&taking), false), (true, false, true));
        // A later taking up, which noted nothing, goes on so.
        let again = bound(true, false);
        assert_eq!(taking_up(Some(&again), true), (true, false, false));
        // A note that says to note what is found, of no taking up, says
        // nothing.
        let stray = bound(false, true);
        assert_eq!(taking_up(Some(&stray), true), (false, false, false));
    }

    #[test]
    fn a_node_that_does_not_know_a_condition_on_a_version_is_told_from_an_object_it_cannot_hold() {
        // As a node of 0.1 refuses a change with a field it does not
        // know, by either of the ways it reads a body.
        let old = [
            "unknown field `only_if_version`, expected one of `collection`, `id`, `properties`, `vector`",
            "unknown field `changes[3].only_if_version`",
        ];
        for message in old {
            assert!(unversed(Code::Invalid, message), "{message}");
            assert!(!unversed(Code::Conflict, message), "{message}");
        }
        // What a node says of an object it cannot hold is not that.
        for message in [
            "property title: too long",
            "unknown field `colour`",
            "the object is not at the version given as only_if_version",
        ] {
            assert!(!unversed(Code::Invalid, message), "{message}");
        }
    }

    #[test]
    fn the_note_of_a_collection_is_written_only_over_the_one_read_or_where_there_is_none() {
        let fixture = Fixture::new(options());
        let (binding, name) = (&fixture.plan.binding, &fixture.plan.schema.name);
        let at = fixture.stored("fox").version;
        match binding.change(name, at) {
            Change::Upsert {
                collection,
                only_if_version,
                ..
            } => assert_eq!((collection.as_str(), only_if_version), (IMPORTS, at)),
            other => panic!("{other:?}"),
        }
        assert!(at.is_some());
        assert!(matches!(binding.change(name, None), Change::Insert { .. }));
        // One the node gives no version for is not read.
        let noted = |version: Option<&str>| {
            let Change::Insert { id, properties, .. } = binding.change(name, None) else {
                unreachable!()
            };
            let stored = json!({ "id": id, "properties": properties, "version": version });
            Binding::held(&serde_json::from_value(stored).unwrap())
        };
        let (read, _) = noted(Some("000102030405060708090a0b0c0d0e0f")).unwrap();
        assert_eq!(&read, binding);
        let stopped = noted(None).unwrap_err().to_string();
        assert!(stopped.contains("gave no version"), "{stopped}");
    }

    #[test]
    fn a_run_is_refused_what_it_cannot_be_told() {
        let told = |sole_writer: bool, delete_unknown: bool, other_writers: bool| {
            let options = Options {
                sole_writer,
                delete_unknown,
                other_writers,
                ..options()
            };
            options.check().map_err(|refused| refused.to_string())
        };
        for said in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (false, false, true),
        ] {
            assert_eq!(told(said.0, said.1, said.2), Ok(()), "{said:?}");
        }
        let both = told(true, false, true).unwrap_err();
        assert!(both.contains("either nothing else wrote"), "{both}");
        for others in [false, true] {
            let unasked = told(false, true, others).unwrap_err();
            assert!(
                unasked.contains("deleted only once nothing but an import"),
                "{unasked}"
            );
        }
        assert!(told(true, true, true).is_err());
        let sized = |page: usize, batch_bytes: usize, batch_changes: usize| Options {
            page,
            batch_bytes,
            batch_changes,
            ..options()
        };
        assert!(sized(1, 1, 1).check().is_ok());
        for none in [sized(0, 1, 1), sized(1, 0, 1), sized(1, 1, 0)] {
            assert!(none.check().is_err());
        }
    }

    #[test]
    fn an_entry_says_all_there_is_to_say_only_of_an_object_weaviate_lists_unchanged() {
        let fixture = Fixture::new(options());
        for mark in [Mark::Stored, Mark::Foreign, Mark::Withheld, Mark::Theirs] {
            let entry = fixture.entry("fox", 1, mark);
            assert!(entry.listed(Some(1)), "{mark:?}");
            assert!(!entry.listed(Some(2)), "{mark:?}");
            assert!(!entry.listed(None), "{mark:?}");
            let gone = Entry {
                gone: true,
                ..entry
            };
            assert!(!gone.listed(Some(1)), "{mark:?}");
        }
        // What was noted as held by the collection alone is of another
        // object once Weaviate lists the id, whenever that changed.
        let unknown = Entry {
            updated: None,
            ..fixture.entry("fox", 1, Mark::Unknown)
        };
        assert!(!unknown.listed(None));
        assert!(!fixture.entry("fox", 1, Mark::Unknown).listed(Some(1)));
    }

    #[test]
    fn what_a_collection_holds_alone_is_noted_only_where_the_ledger_has_no_entry() {
        let fixture = Fixture::new(options());
        let copying = fixture.copying();
        let fox = fixture.stored("fox");
        let Change::Insert {
            collection,
            id,
            properties,
            ..
        } = copying.census(&fox)
        else {
            panic!("{:?}", copying.census(&fox));
        };
        assert_eq!((collection.as_str(), id), (ENTRY, fox.id));
        assert_eq!(properties.get("unknown"), Some(&json!(true)));
        assert_eq!(
            properties.get("hash"),
            Some(&json!(fixture.item("fox", 0).hash))
        );
    }

    #[test]
    fn what_the_ledger_accounts_for_leaves_out_every_entry_of_nothing_weaviate_holds() {
        let copied = Copied {
            withheld: 1,
            unknown: 20,
            left: 300,
            barred: 4_000,
            kept: 50_000,
            ..Copied::default()
        };
        assert_eq!(copied.unvouched(), 4_321);
    }

    #[test]
    fn a_ledger_entry_that_is_not_the_one_read_stops_the_run() {
        let fixture = Fixture::new(options());
        let read = fixture.entry("fox", 1, Mark::Stored);
        let alone = |read: Option<&Entry>, again: Option<&Entry>| {
            Copying::alone(read.map_or(id_of(&fixture), |entry| entry.id), read, again)
        };
        assert!(alone(None, None).is_ok());
        assert!(alone(Some(&read), Some(&read)).is_ok());
        // Written anew, though it says the same; saying something else;
        // there where there was none; gone.
        let rewritten = Entry {
            at: fixture.stored("owl").version,
            ..read.clone()
        };
        let other = fixture.entry("fox", 1, Mark::Withheld);
        let changed = [
            (Some(&read), Some(&rewritten)),
            (Some(&read), Some(&other)),
            (None, Some(&read)),
            (Some(&read), None),
        ];
        for (read, again) in changed {
            let stopped = alone(read, again).unwrap_err().to_string();
            assert!(
                stopped.contains("is another import of the tenant running?"),
                "{stopped}"
            );
        }
        // Every write of an entry there was says which it was, and one of
        // an entry there was not is made only where there is still none.
        let copying = fixture.copying();
        let gone = Entry {
            gone: true,
            ..read.clone()
        };
        match copying.note(&gone, Some(&read)) {
            Change::Upsert {
                only_if_version, ..
            } => assert_eq!(only_if_version, read.at),
            other => panic!("{other:?}"),
        }
        match copying.forget(&read) {
            Change::Delete {
                only_if_version, ..
            } => assert_eq!(only_if_version, read.at),
            other => panic!("{other:?}"),
        }
        assert!(matches!(copying.note(&gone, None), Change::Insert { .. }));
        assert!(read.at.is_some());
        // One the node gives no version for is not read at all.
        let unversioned = json!({ "id": read.id, "properties": { "hash": "h" } });
        let stopped = Entry::read(serde_json::from_value(unversioned).unwrap()).unwrap_err();
        assert!(stopped.to_string().contains("gave no version"), "{stopped}");
    }

    fn id_of(fixture: &Fixture) -> ObjectId {
        fixture.item("fox", 0).id
    }

    #[test]
    fn a_comparison_counts_what_weaviate_does_not_hold_as_a_run_that_copies_would() {
        let fixture = Fixture::new(options());
        let counted = |entry: &Entry, holds: bool| {
            let mut copied = Copied::default();
            let stale = fixture.copying().stale(entry, holds, &mut copied);
            (stale, copied.left, copied.unknown, copied.barred)
        };
        let left = |mark: Mark| Entry {
            gone: true,
            ..fixture.entry("fox", 1, mark)
        };
        // Left in the collection, it is counted so; gone from there too,
        // its entry is one the next run forgets, and counts for nothing.
        for mark in [Mark::Stored, Mark::Foreign] {
            assert_eq!(counted(&left(mark), true), (false, 1, 0, 0));
            assert_eq!(counted(&left(mark), false), (true, 0, 0, 0));
        }
        // So with what was there when the collection was taken up.
        for entry in [fixture.entry("fox", 1, Mark::Unknown), left(Mark::Withheld)] {
            assert_eq!(counted(&entry, true), (false, 0, 1, 0));
            assert_eq!(counted(&entry, false), (true, 0, 0, 0));
        }
        // What was said to be another writer's keeps its entry, which
        // bars the id.
        let theirs = left(Mark::Stored).theirs(Some(1), true);
        assert_eq!(counted(&theirs, true), (false, 1, 0, 0));
        assert_eq!(counted(&theirs, false), (false, 0, 0, 1));
    }

    #[test]
    fn an_object_is_replaced_or_deleted_only_at_the_version_it_was_read_at() {
        let fixture = Fixture::new(options());
        let copying = fixture.copying();
        let (fox, owl) = (fixture.stored("fox"), fixture.stored("owl"));
        assert!(fox.version.is_some() && fox.version != owl.version);
        let ours = fixture.entry("fox", 1, Mark::Stored);
        let hen = fixture.item("hen", 2);
        let (replacing, _) = copying.settling(&hen, Some(&ours), Some(&fox), false);
        let [Change::Upsert {
            only_if_version, ..
        }, _] = &replacing[..]
        else {
            panic!("{replacing:?}");
        };
        assert_eq!(*only_if_version, fox.version);
        let (deleting, _) = copying.removing(&ours, Some(&fox));
        let [Change::Delete {
            only_if_version, ..
        }, _] = &deleting[..]
        else {
            panic!("{deleting:?}");
        };
        assert_eq!(*only_if_version, fox.version);
        // Read from a node that gives no version, nothing can be held to
        // one, and the run stops.
        let unversioned: Stored =
            serde_json::from_value(json!({ "id": fox.id, "properties": fox.properties })).unwrap();
        assert!(Copying::versioned([&fox, &owl]).is_ok());
        let stopped = Copying::versioned([&fox, &unversioned]).unwrap_err();
        assert!(stopped.to_string().contains("gave no version"), "{stopped}");
    }

    #[test]
    fn an_entry_weaviate_did_not_list_is_asked_about_or_kept_by_what_it_says() {
        let fixture = Fixture::new(options());
        let left = |mark: Mark| Entry {
            gone: true,
            ..fixture.entry("fox", 1, mark)
        };
        let kept = |entry: &Entry| Unlisted::Keep {
            entry: entry.clone(),
            now: None,
        };
        // What may be gone is asked about.
        for mark in [Mark::Stored, Mark::Foreign, Mark::Withheld] {
            let entry = fixture.entry("fox", 1, mark);
            assert_eq!(
                fixture.copying().unlisted(entry.clone()),
                Unlisted::Ask(entry)
            );
        }
        // What was left before is kept, and not asked about again.
        for mark in [Mark::Stored, Mark::Foreign, Mark::Theirs] {
            assert_eq!(fixture.copying().unlisted(left(mark)), kept(&left(mark)));
        }
        // What was said to be another writer's is kept the first time
        // too, saying from then on that Weaviate does not hold it.
        let theirs = fixture.entry("fox", 1, Mark::Theirs).theirs(Some(1), false);
        let gone = Unlisted::Keep {
            entry: theirs.clone(),
            now: Some(theirs.theirs(Some(1), true)),
        };
        assert_eq!(fixture.copying().unlisted(theirs), gone);
        // What was there at the taking up that Weaviate does not hold is
        // kept, whoever is said to be the sole writer, until it is asked
        // to be deleted or said to be another's.
        let sole = Fixture::new(Options {
            sole_writer: true,
            ..options()
        });
        let pre = [fixture.entry("fox", 1, Mark::Unknown), left(Mark::Withheld)];
        for entry in &pre {
            for fixture in [&fixture, &sole] {
                assert_eq!(fixture.copying().unlisted(entry.clone()), kept(entry));
            }
        }
        let sweeping = Fixture::new(Options {
            sole_writer: true,
            delete_unknown: true,
            ..options()
        });
        let others = Fixture::new(Options {
            other_writers: true,
            ..options()
        });
        let withheld = Entry {
            found: Some("as it was".into()),
            ..fixture.entry("fox", 1, Mark::Withheld)
        };
        for entry in pre.iter().chain([&withheld]) {
            let asked = Unlisted::Ask(entry.clone());
            assert_eq!(sweeping.copying().unlisted(entry.clone()), asked);
            // Said to be another writer's, it is at once, without
            // Weaviate being asked, and with nothing left of what it
            // hashed as or was found to be.
            let theirs = Entry {
                hash: String::new(),
                mark: Mark::Theirs,
                found: None,
                gone: true,
                ..entry.clone()
            };
            let now = Unlisted::Keep {
                entry: entry.clone(),
                now: Some(theirs),
            };
            assert_eq!(others.copying().unlisted(entry.clone()), now);
        }
        // Neither makes anything of what another writer is known to
        // have written.
        for fixture in [&sweeping, &others] {
            for mark in [Mark::Foreign, Mark::Theirs] {
                assert_eq!(fixture.copying().unlisted(left(mark)), kept(&left(mark)));
            }
        }
    }

    #[test]
    fn an_entry_kept_is_counted_while_the_collection_holds_its_object_and_forgotten_after() {
        let fixture = Fixture::new(options());
        let keeping = |entry: &Entry, now: Option<&Entry>, holds: bool| {
            let mut copied = Copied::default();
            let changes = fixture.copying().keeping(entry, now, holds, &mut copied);
            (kinds(&changes), copied.left, copied.unknown)
        };
        let left = |mark: Mark| Entry {
            gone: true,
            ..fixture.entry("fox", 1, mark)
        };
        for mark in [Mark::Stored, Mark::Foreign, Mark::Theirs] {
            assert_eq!(keeping(&left(mark), None, true), (vec![], 1, 0));
        }
        for entry in [fixture.entry("fox", 1, Mark::Unknown), left(Mark::Withheld)] {
            assert_eq!(keeping(&entry, None, true), (vec![], 0, 1));
            // The object gone from the collection too, there is nothing
            // left for the entry to say anything of.
            assert_eq!(
                keeping(&entry, None, false),
                (only(&["delete ENTRY"]), 0, 0)
            );
        }
        for mark in [Mark::Stored, Mark::Foreign] {
            assert_eq!(
                keeping(&left(mark), None, false),
                (only(&["delete ENTRY"]), 0, 0)
            );
        }
        // What was said to be another writer's keeps its entry even so:
        // their deleting of it is not undone if Weaviate comes to hold
        // the id. It is no longer counted.
        assert_eq!(keeping(&left(Mark::Theirs), None, false), (vec![], 0, 0));
        let unknown = fixture.entry("fox", 1, Mark::Unknown);
        let theirs = unknown.theirs(unknown.updated, true);
        assert_eq!(
            keeping(&unknown, Some(&theirs), true),
            (only(&["upsert ENTRY theirs gone"]), 1, 0)
        );
        assert_eq!(
            keeping(&unknown, Some(&theirs), false),
            (only(&["upsert ENTRY theirs gone"]), 0, 0)
        );
    }

    /// The ledger's entry once `changes` are made, having been `entry`.
    fn noted(entry: Option<&Entry>, changes: &[Change]) -> Option<Entry> {
        let ledger = CollectionName::new(ENTRY).unwrap();
        let mut entry = entry.cloned();
        for change in changes {
            match change {
                Change::Insert {
                    collection,
                    id,
                    properties,
                    ..
                }
                | Change::Upsert {
                    collection,
                    id,
                    properties,
                    ..
                } if *collection == ledger => {
                    let version = format!("{:032x}", properties.len() + 1);
                    let stored = json!({ "id": id, "properties": properties, "version": version });
                    entry = Some(Entry::read(serde_json::from_value(stored).unwrap()).unwrap());
                }
                Change::Delete { collection, .. } if *collection == ledger => entry = None,
                _ => {}
            }
        }
        entry
    }

    /// What a run is told of who wrote to a collection.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Told {
        Nothing,
        Sole,
        SoleAndDelete,
        Others,
    }

    const TOLD: [Told; 4] = [Told::Nothing, Told::Sole, Told::SoleAndDelete, Told::Others];

    fn told(told: Told) -> Fixture {
        Fixture::new(Options {
            sole_writer: matches!(told, Told::Sole | Told::SoleAndDelete),
            delete_unknown: told == Told::SoleAndDelete,
            other_writers: told == Told::Others,
            ..options()
        })
    }

    /// What a run does about one id: with what Weaviate holds under it,
    /// "hen", or with Weaviate holding nothing; `adopted` says whether
    /// the collection is being taken up.
    fn done(
        fixture: &Fixture,
        entry: Option<&Entry>,
        weaviate: bool,
        stored: Option<&Stored>,
        adopted: bool,
    ) -> Vec<Change> {
        let copying = fixture.copying();
        let Some(entry) = entry.filter(|_| !weaviate) else {
            return match weaviate {
                true => {
                    let hen = fixture.item("hen", 2);
                    copying.settling(&hen, entry, stored, adopted).0
                }
                // Neither Weaviate nor the ledger has the id: nothing
                // looks at it.
                false => Vec::new(),
            };
        };
        match copying.unlisted(entry.clone()) {
            Unlisted::Ask(entry) => copying.removing(&entry, stored).0,
            Unlisted::Keep { entry, now } => {
                let mut copied = Copied::default();
                copying.keeping(&entry, now.as_ref(), stored.is_some(), &mut copied)
            }
        }
    }

    #[test]
    fn nothing_is_deleted_or_written_over_but_what_the_rule_allows_whatever_the_ledger_and_the_run_say(
    ) {
        let plain = Fixture::new(options());
        let hash = |title: &str| plain.item(title, 0).hash;
        // What Weaviate holds under the id, where it holds anything.
        let hen = hash("hen");
        // Every entry there can be: of each mark, found or not to be gone
        // from Weaviate, of what Weaviate holds now or of something it
        // held before; what was withheld, with and without what the
        // collection was found to hold; and what was said to be another
        // writer's.
        let mut entries: Vec<Option<Entry>> = vec![None];
        for gone in [false, true] {
            for (title, updated) in [("fox", 1), ("hen", 2)] {
                for mark in [Mark::Stored, Mark::Foreign, Mark::Withheld, Mark::Unknown] {
                    let entry = Entry {
                        gone,
                        ..plain.entry(title, updated, mark)
                    };
                    entries.push(Some(entry.clone()));
                    if mark == Mark::Withheld {
                        entries.push(Some(Entry {
                            found: Some(hash("owl")),
                            ..entry
                        }));
                    }
                }
            }
            let theirs = plain.entry("fox", 1, Mark::Stored).theirs(Some(1), gone);
            entries.push(Some(theirs));
        }
        // Everything the collection can hold under the id: nothing; what
        // an entry hashes as; what it was found to hold; what Weaviate
        // holds now; something else.
        let held: Vec<Option<Stored>> = [None, Some("fox"), Some("owl"), Some("hen"), Some("emu")]
            .map(|title| title.map(|title| plain.stored(title)))
            .into();
        let is = |stored: Option<&Stored>, hash: &str| {
            stored.and_then(|stored| plain.plan.hash(stored)).as_deref() == Some(hash)
        };
        let has = |changes: &[Change], kind: &str| kinds(changes).iter().any(|k| k == kind);
        // Whether each change to the collection is made only if the
        // object is still at the version it was read at.
        let versioned = |changes: &[Change], stored: Option<&Stored>| {
            let read = stored.and_then(|stored| stored.version);
            changes.iter().all(|change| match change {
                Change::Upsert {
                    collection,
                    only_if_version,
                    ..
                }
                | Change::Delete {
                    collection,
                    only_if_version,
                    ..
                } if collection.as_str() == "Memory" => read.is_some() && *only_if_version == read,
                _ => true,
            })
        };
        // Whether each change to the ledger is made only if it still
        // has the entry that was read, at the version it was read at, or
        // still has none.
        let alone = |changes: &[Change], entry: Option<&Entry>| {
            let read = entry.and_then(|entry| entry.at);
            changes.iter().all(|change| match change {
                Change::Upsert {
                    collection,
                    only_if_version,
                    ..
                }
                | Change::Delete {
                    collection,
                    only_if_version,
                    ..
                } if collection.as_str() == ENTRY => read.is_some() && *only_if_version == read,
                Change::Insert { collection, .. } if collection.as_str() == ENTRY => {
                    entry.is_none()
                }
                _ => true,
            })
        };
        // What makes an object the import's to delete, by the entry and
        // what the collection holds: that the import stored it and it is
        // still so; or that it was there at the taking up and is still
        // as it was found.
        let as_stored = |entry: Option<&Entry>, stored: Option<&Stored>| {
            entry.is_some_and(|entry| entry.mark == Mark::Stored && is(stored, &entry.hash))
        };
        let as_found = |entry: Option<&Entry>, stored: Option<&Stored>| {
            entry.is_some_and(|entry| match entry.mark {
                Mark::Unknown => is(stored, &entry.hash),
                Mark::Withheld => entry
                    .found
                    .as_deref()
                    .is_some_and(|found| is(stored, found)),
                _ => false,
            })
        };
        let fixtures = TOLD.map(told);
        let fixture = |said: Told| &fixtures[TOLD.iter().position(|told| *told == said).unwrap()];
        let (mut safety, mut relapses, mut liveness, mut direct) = (0, 0, 0, 0);
        for entry in &entries {
            for weaviate in [true, false] {
                for stored in &held {
                    for said in TOLD {
                        for adopted in [false, true] {
                            if adopted && !weaviate {
                                continue;
                            }
                            safety += 1;
                            let (entry, stored) = (entry.as_ref(), stored.as_ref());
                            let changes = done(fixture(said), entry, weaviate, stored, adopted);
                            let cell = format!(
                                "{entry:?}, Weaviate holds it: {weaviate}, vellev holds {:?}, \
                                 told {said:?}, taken up: {adopted}: {:?}",
                                stored.map(|stored| &stored.properties["title"]),
                                kinds(&changes)
                            );
                            let mark = entry.map(|entry| entry.mark);
                            let (sole, sweeps) = (
                                matches!(said, Told::Sole | Told::SoleAndDelete),
                                said == Told::SoleAndDelete,
                            );
                            let (as_stored, as_found) =
                                (as_stored(entry, stored), as_found(entry, stored));
                            let unvouched = matches!(mark, Some(Mark::Withheld | Mark::Unknown));
                            let taken_up = entry.is_none() && adopted;
                            // The collection holds exactly what Weaviate
                            // does.
                            let same = weaviate && is(stored, &hen);
                            let after = noted(entry, &changes);

                            // (a) An object is deleted only if the import
                            // stored it and it is still so; or it was
                            // there at the taking up, is still as it was
                            // found, and the run is told both that
                            // nothing else wrote and to delete it.
                            if has(&changes, "delete Memory") {
                                assert!(!weaviate, "{cell}");
                                assert!(as_stored || (sweeps && as_found), "{cell}");
                            }
                            // (b) An object is written over only if the
                            // import stored it and it is still so; or
                            // nothing else is said to have written, and
                            // it was withheld, unknown, or is met with no
                            // entry while the collection is taken up.
                            if has(&changes, "upsert Memory") {
                                assert!(weaviate && stored.is_some(), "{cell}");
                                assert!(as_stored || (sole && (unvouched || taken_up)), "{cell}");
                            }
                            // Nor is anything brought back: an object is
                            // stored where the collection holds none only
                            // if it is new to Weaviate, or on those same
                            // terms.
                            if has(&changes, "insert Memory") {
                                assert!(weaviate && stored.is_none(), "{cell}");
                                let new = entry.is_none() && !adopted;
                                assert!(new || (sole && (unvouched || taken_up)), "{cell}");
                            }
                            // And then only if it is still at the
                            // version it was read at.
                            assert!(versioned(&changes, stored), "{cell}");
                            assert!(alone(&changes, entry), "{cell}");
                            // That a collection is being taken up bears
                            // only on an id the ledger has no entry for.
                            if adopted && entry.is_some() {
                                let caught_up = done(fixture(said), entry, weaviate, stored, false);
                                assert_eq!(kinds(&changes), kinds(&caught_up), "{cell}");
                            }
                            // (d) Without an entry, and outside a taking
                            // up, nothing is deleted or written over.
                            if entry.is_none() && !adopted {
                                assert!(!has(&changes, "delete Memory"), "{cell}");
                                assert!(!has(&changes, "upsert Memory"), "{cell}");
                            }
                            // The ledger forgets no object the collection
                            // holds, unless it deletes it.
                            if has(&changes, &format!("delete {ENTRY}")) {
                                let deleted = has(&changes, "delete Memory");
                                assert!(deleted || stored.is_none(), "{cell}");
                            }
                            // What is another writer's on somebody's word
                            // has nothing done to it, and stays so, even
                            // where it is now exactly what Weaviate
                            // holds; its entry says only whether Weaviate
                            // lists the id.
                            if mark == Some(Mark::Theirs) {
                                let untouched = ["delete Memory", "upsert Memory", "insert Memory"];
                                assert!(
                                    !untouched.iter().any(|kind| has(&changes, kind)),
                                    "{cell}"
                                );
                                let theirs = after.as_ref().expect(&cell);
                                assert_eq!(
                                    (theirs.mark, theirs.hash.as_str(), theirs.gone),
                                    (Mark::Theirs, "", !weaviate),
                                    "{cell}"
                                );
                            }

                            // What the import must do, and not only may.
                            let theirs = mark == Some(Mark::Theirs);
                            let copied = |cell: &str| {
                                let entry = after.as_ref().expect(cell);
                                assert_eq!(
                                    (entry.mark, entry.hash.as_str(), entry.gone),
                                    (Mark::Stored, hen.as_str(), false),
                                    "{cell}"
                                );
                            };
                            let mut owed = false;
                            // What the collection holds exactly as
                            // Weaviate does is the import's, whatever
                            // the run is told and whatever the entry
                            // said, and nothing is written to it.
                            if same && !theirs {
                                owed = true;
                                copied(&cell);
                                assert!(!has(&changes, "upsert Memory"), "{cell}");
                            }
                            // What the import stored, nobody changed and
                            // Weaviate holds otherwise is replaced.
                            if weaviate && as_stored && !same {
                                owed = true;
                                copied(&cell);
                                assert!(has(&changes, "upsert Memory"), "{cell}");
                            }
                            // What was withheld or unknown, or is met
                            // with no entry while the collection is
                            // taken up, is brought to what Weaviate
                            // holds once nothing else is said to have
                            // written.
                            if weaviate && sole && (unvouched || taken_up) && !same {
                                owed = true;
                                copied(&cell);
                                let kind = match stored {
                                    Some(_) => "upsert Memory",
                                    None => "insert Memory",
                                };
                                assert!(has(&changes, kind), "{cell}");
                            }
                            // What is new to Weaviate is stored.
                            if weaviate && entry.is_none() && !adopted && stored.is_none() {
                                owed = true;
                                copied(&cell);
                                assert!(has(&changes, "insert Memory"), "{cell}");
                            }
                            // What the import stored, nobody changed and
                            // Weaviate no longer holds is deleted, unless
                            // it was found gone before and left: that is
                            // not asked about again.
                            let left = entry.is_some_and(|entry| entry.gone);
                            if !weaviate && as_stored && !left {
                                owed = true;
                                assert!(has(&changes, "delete Memory"), "{cell}");
                                assert_eq!(after, None, "{cell}");
                            }
                            // And what was there at the taking up, still
                            // as it was found, once that is asked for.
                            if !weaviate && sweeps && as_found {
                                owed = true;
                                assert!(has(&changes, "delete Memory"), "{cell}");
                                assert_eq!(after, None, "{cell}");
                            }
                            liveness += usize::from(owed);

                            // (c) Once something else is said to have
                            // written, what was withheld or unknown and
                            // is not exactly what Weaviate holds is
                            // another writer's, whatever Weaviate and the
                            // collection hold; and no later run, whatever
                            // it is told and whatever they hold by then,
                            // deletes it, writes over it, brings it back
                            // or takes it for anything else.
                            let said_theirs = unvouched || (taken_up && weaviate);
                            if said != Told::Others || !said_theirs || same {
                                continue;
                            }
                            let theirs = after.expect(&cell);
                            assert_eq!(
                                (theirs.mark, theirs.hash.as_str()),
                                (Mark::Theirs, ""),
                                "{cell}"
                            );
                            assert_eq!(theirs.found, None, "{cell}");
                            for later in TOLD {
                                for weaviate in [true, false] {
                                    for stored in &held {
                                        relapses += 1;
                                        let stored = stored.as_ref();
                                        let later = fixture(later);
                                        let changes =
                                            done(later, Some(&theirs), weaviate, stored, false);
                                        let cell = format!("{cell}; then told {:?}, Weaviate holds it: {weaviate}: {:?}", (later.options.sole_writer, later.options.delete_unknown, later.options.other_writers), kinds(&changes));
                                        for kind in
                                            ["delete Memory", "upsert Memory", "insert Memory"]
                                        {
                                            assert!(!has(&changes, kind), "{cell}");
                                        }
                                        assert!(alone(&changes, Some(&theirs)), "{cell}");
                                        let still = noted(Some(&theirs), &changes).expect(&cell);
                                        assert_eq!(
                                            (still.mark, still.hash.as_str()),
                                            (Mark::Theirs, ""),
                                            "{cell}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // The last guard, asked by itself: whatever brought an entry to
        // be settled as of an id Weaviate does not hold, what the
        // collection holds under it is deleted on the same terms and no
        // others.
        for entry in entries.iter().flatten() {
            for stored in &held {
                for said in TOLD {
                    direct += 1;
                    let stored = stored.as_ref();
                    let copying = fixture(said).copying();
                    let (changes, settled) = copying.removing(entry, stored);
                    let cell = format!(
                        "{entry:?} removed, vellev holds {:?}, told {said:?}: {:?}",
                        stored.map(|stored| &stored.properties["title"]),
                        kinds(&changes)
                    );
                    let sweeps = said == Told::SoleAndDelete;
                    let mine =
                        as_stored(Some(entry), stored) || (sweeps && as_found(Some(entry), stored));
                    let after = noted(Some(entry), &changes);
                    assert_eq!(has(&changes, "delete Memory"), mine, "{cell}");
                    assert_eq!(settled == Settled::Deleted, mine, "{cell}");
                    assert!(versioned(&changes, stored), "{cell}");
                    assert!(alone(&changes, Some(entry)), "{cell}");
                    for kind in ["upsert Memory", "insert Memory"] {
                        assert!(!has(&changes, kind), "{cell}");
                    }
                    if matches!(entry.mark, Mark::Foreign | Mark::Theirs) {
                        assert!(!mine, "{cell}");
                    }
                    let unvouched = matches!(entry.mark, Mark::Withheld | Mark::Unknown);
                    let theirs = entry.mark == Mark::Theirs || (unvouched && said == Told::Others);
                    match (mine, stored, theirs) {
                        (true, ..) => assert_eq!(after, None, "{cell}"),
                        // What is left keeps its entry, saying so.
                        (false, Some(_), _) | (false, None, true) => {
                            let after = after.expect(&cell);
                            assert!(after.gone, "{cell}");
                            let mark = match theirs {
                                true => Mark::Theirs,
                                false => entry.mark,
                            };
                            assert_eq!(after.mark, mark, "{cell}");
                            assert!(!theirs || after.hash.is_empty(), "{cell}");
                        }
                        (false, None, false) => assert_eq!(after, None, "{cell}"),
                    }
                }
            }
        }
        // Every cell was looked at: change the tables above, and say so
        // here.
        assert_eq!(
            (safety, relapses, liveness, direct),
            (SAFETY, LATER, LIVENESS, DIRECT)
        );
    }

    #[test]
    fn a_run_deletes_no_more_than_the_fewer_of_a_count_and_a_part_of_what_is_held() {
        let budget = |deletes: Deletes, ledgered: u64, held: u64| {
            let fixture = Fixture::new(Options {
                deletes,
                ..options()
            });
            fixture.copying().budget(ledgered, held)
        };
        let usual = Options::new(PathBuf::new()).deletes;
        assert_eq!(
            usual,
            Deletes::UpTo {
                count: 1_000,
                fraction: 0.05
            }
        );
        assert_eq!(budget(usual, 20, 20), 1);
        assert_eq!(budget(usual, 21, 21), 2);
        assert_eq!(budget(usual, 1_000, 1_000), 50);
        assert_eq!(budget(usual, 4_200_000, 4_200_000), 1_000);
        assert_eq!(budget(Deletes::None, 4_200_000, 4_200_000), 0);
        // A part of the lesser of what the ledger has and what the
        // collection holds.
        assert_eq!(budget(usual, 1_000, 100), 5);
        assert_eq!(budget(usual, 100, 1_000), 5);
        // Never the last object the collection holds, unless a run may
        // delete all of a class.
        assert_eq!(budget(usual, 2, 2), 1);
        assert_eq!(budget(usual, 40, 2), 1);
        assert_eq!(budget(usual, 1, 1), 0);
        assert_eq!(budget(usual, 40, 1), 0);
        assert_eq!(budget(usual, 0, 0), 0);
        let all = Deletes::UpTo {
            count: 1_000,
            fraction: 1.0,
        };
        assert_eq!((budget(all, 1, 1), budget(all, 20, 20)), (1, 20));

        // What is gone is counted whole, by its marks, and kept only as
        // far as it may be deleted.
        let mut gone = Gone {
            entries: Vec::new(),
            count: 0,
            foreign: 0,
            withheld: 0,
            budget: 3,
            forgotten: Vec::new(),
        };
        let fixture = Fixture::new(options());
        for mark in [Mark::Stored, Mark::Foreign, Mark::Foreign, Mark::Withheld] {
            gone.push(fixture.entry("fox", 1, mark));
        }
        let counted = (gone.count, gone.foreign, gone.withheld);
        assert_eq!((counted, gone.entries.len()), ((4, 2, 1), 3));
    }

    #[test]
    fn the_pages_of_a_ledger_looked_at_are_picked_by_chance_from_all_of_them() {
        let mut given = [7, 0, 5, 2].into_iter().cycle();
        assert_eq!(sample(10, 3, || given.next().unwrap()), [0, 1, 7]);
        // Every page, where as many are asked for as there are, or more.
        assert_eq!(sample(4, 4, || 3), [0, 1, 2, 3]);
        assert_eq!(sample(4, 9, || 3), [0, 1, 2, 3]);
        assert_eq!(sample(0, 3, || 3), [] as [usize; 0]);
        // And each as likely as any other.
        let mut state = 1_usize;
        let mut times = [0; 10];
        for _ in 0..10_000 {
            let random = || {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                state >> 33
            };
            for page in sample(10, 3, random) {
                times[page] += 1;
            }
        }
        assert!(
            times.iter().all(|times| (2_700..3_300).contains(times)),
            "{times:?}"
        );
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
            let version = "000102030405060708090a0b0c0d0e0f";
            let stored = json!({ "id": id(1), "properties": properties, "version": version });
            Entry::read(serde_json::from_value(stored).unwrap())
        };
        let read = entry(json!({ "hash": "h", "updated": 3 })).unwrap();
        assert_eq!((read.updated, read.mark), (Some(3), Mark::Stored));
        let marked = |mark: &str| entry(json!({ "hash": "h", mark: true })).unwrap().mark;
        assert_eq!(marked("foreign"), Mark::Foreign);
        assert_eq!(marked("withheld"), Mark::Withheld);
        assert_eq!(marked("unknown"), Mark::Unknown);
        // An entry is written as it was read.
        let whole =
            json!({ "hash": "h", "updated": 3, "withheld": true, "found": "f", "gone": true });
        let read = entry(whole.clone()).unwrap();
        assert_eq!((read.found.as_deref(), read.gone), (Some("f"), true));
        assert_eq!(Value::Object(read.properties()), whole);
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
            [
                "updatedAs",
                "shards",
                "adopting",
                "finalised",
                "version",
                "census"
            ]
        );
    }

    #[test]
    fn a_run_is_settled_behind_or_unsettled_by_what_it_left() {
        let report = |copied: Copied| Report {
            classes: [("Memory".to_string(), copied)].into(),
            ended: Vec::new(),
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
        // What is gone from Weaviate and was not deleted, whether the
        // run was told to delete nothing or might not delete so much, is
        // the import's and not as Weaviate has it.
        let pending = Copied {
            weaviate: 8,
            pending: 2,
            ..level.clone()
        };
        assert!(pending.caught_up());
        assert_eq!(outcome(pending.clone()), Outcome::Unsettled);
        let over = Copied {
            budget: Some(1),
            ..pending
        };
        assert_eq!(outcome(over), Outcome::Unsettled);
        // Nor has a run settled that did not copy what Weaviate holds.
        let withheld = Copied {
            imported: 7,
            withheld: 3,
            vellev: 7,
            ..level.clone()
        };
        assert!(withheld.caught_up());
        assert_eq!(outcome(withheld), Outcome::Unsettled);
        let refused = Copied {
            weaviate: 11,
            refused: [(id(1), "why".to_string())].into(),
            ..level.clone()
        };
        assert_eq!(outcome(refused), Outcome::Unsettled);
        let wrong = Copied {
            verified: Some(Verified {
                different: 1,
                ..Verified::default()
            }),
            ..behind
        };
        assert_eq!(outcome(wrong), Outcome::Unsettled);
        // A comparison that looked at nothing, of a class that holds
        // something, compared nothing.
        let unseen = Copied {
            verified: Some(Verified::default()),
            ..level.clone()
        };
        assert_eq!(outcome(unseen), Outcome::Unsettled);
        let empty = Copied {
            verified: Some(Verified::default()),
            ..Copied::default()
        };
        assert_eq!(outcome(empty), Outcome::Settled);
    }

    #[test]
    fn objects_go_to_a_request_until_it_is_full_and_a_large_one_goes_alone() {
        let fixture = Fixture::new(options());
        let step = |bytes: usize| {
            let item = fixture.item(&"x".repeat(bytes), 0);
            Step {
                changes: fixture.copying().fresh(&item, None),
                settled: Settled::Stored,
                noted: None,
                item,
            }
        };
        let sizes = |most: usize, changes: usize, bytes: usize| {
            let steps = vec![step(100), step(100), step(100), step(9_000), step(100)];
            batches(steps, most, changes, bytes)
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>()
        };
        let (changes, bytes) = (1_000, 1 << 20);
        assert_eq!(sizes(10, changes, bytes), [5]);
        assert_eq!(sizes(2, changes, bytes), [2, 2, 1]);
        assert_eq!(sizes(10, changes, 2_000), [3, 1, 1]);
        // Each stores an object and notes it: two changes.
        assert_eq!(sizes(10, 6, bytes), [3, 2]);
        assert_eq!(sizes(10, 1, bytes), [1, 1, 1, 1, 1]);
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
