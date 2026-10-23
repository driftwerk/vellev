//! Resolving what callers ask for into what the log records.

use std::collections::{BTreeSet, HashMap};

use vellev_core::{
    Collection, CollectionName, DataType, Embedding, EmbeddingProvider, Object, ObjectError,
    ObjectId, Value, Version,
};

use vellev_index::filter::Predicate;

use crate::catalog::Catalog;
use crate::format::{CollectionId, Op};
use crate::mutation::{Condition, Mutation, Outcome, DELETED_AT_MOST};
use crate::state::{Live, State};
use crate::{Capped, Over, Result, StorageError};

type RowKey = (CollectionId, ObjectId);

/// The changes one commit will make, built up batch by batch on top of a
/// state. A batch sees every batch accepted before it, and a batch that is
/// rejected leaves no trace.
pub(crate) struct Draft<'a> {
    state: &'a State,
    pending: Pending,
}

/// What has been resolved so far: the catalog as it will be, the rows that
/// will be written (`None` for a delete), the ops in order, and how many
/// ops there were after each batch.
struct Pending {
    catalog: Catalog,
    rows: HashMap<RowKey, Option<Object>>,
    ops: Vec<Op>,
    ends: Vec<usize>,
    texts: HashMap<usize, String>,
}

/// An object a commit writes without the vector its collection embeds for
/// it: the last op that writes it, the batch that op came from, and what
/// to embed.
pub(crate) struct Wanted {
    pub op: usize,
    pub batch: usize,
    pub embedding: Embedding,
    pub text: String,
}

/// The text to embed for `object`, if its collection embeds its objects
/// and it has text and no vector.
fn wanting_text(schema: &Collection, object: &Object) -> Option<String> {
    let embeds = schema.embedding.provider != EmbeddingProvider::None;
    (embeds && object.vector.is_none())
        .then(|| object.embedded_text(&schema.embedding))
        .filter(|text| !text.trim().is_empty())
}

/// One batch being resolved on top of a draft, kept apart until it is known
/// to be valid as a whole.
struct Batch<'a> {
    draft: &'a Draft<'a>,
    catalog: Catalog,
    rows: HashMap<RowKey, Option<Object>>,
    ops: Vec<Op>,
    /// The text to embed for each op, by its place in `ops`, that writes
    /// an object of a collection not storing all it embeds, and wants a
    /// vector: the op's object no longer holds all of it.
    texts: HashMap<usize, String>,
}

impl<'a> Draft<'a> {
    pub fn new(state: &'a State) -> Self {
        let pending = Pending {
            catalog: state.catalog().clone(),
            rows: HashMap::new(),
            ops: vec![],
            ends: vec![],
            texts: HashMap::new(),
        };
        Self { state, pending }
    }

    /// Resolve one caller's batch. Either all of it joins the draft, or, on
    /// the first mutation that cannot be applied, none of it does.
    pub fn batch(&mut self, mutations: Vec<Mutation>) -> Result<Vec<Outcome>> {
        let outcomes = self.resolved(mutations);
        self.pending.ends.push(self.pending.ops.len());
        outcomes
    }

    fn resolved(&mut self, mutations: Vec<Mutation>) -> Result<Vec<Outcome>> {
        // Each delete by filter sees what those before it deleted, so the
        // batch may delete as many as all of them may between them: no
        // more than one of them may alone, or the batch is one commit of
        // any size after all.
        let deletes = mutations.iter().filter_map(|mutation| match mutation {
            Mutation::DeleteWhere { .. } => Some(mutation.rows()),
            _ => None,
        });
        let may_delete: usize = deletes.sum();
        if may_delete > DELETED_AT_MOST {
            return Err(too_many_deleted(format!(
                "the deletes by filter of one request may delete {may_delete} objects between \
                 them, each without a limit counting for {DELETED_AT_MOST}, which is the most \
                 one request may delete; send them in requests of their own, or with limits \
                 that come to no more"
            )));
        }
        let mut batch = Batch {
            draft: self,
            catalog: self.pending.catalog.clone(),
            rows: HashMap::new(),
            ops: vec![],
            texts: HashMap::new(),
        };
        let outcomes = mutations
            .into_iter()
            .map(|m| batch.resolve(m))
            .collect::<Result<Vec<_>>>()?;
        let Batch {
            catalog,
            rows,
            ops,
            texts,
            ..
        } = batch;
        let before = self.pending.ops.len();
        self.pending.catalog = catalog;
        self.pending.rows.extend(rows);
        self.pending.ops.extend(ops);
        let texts = texts.into_iter().map(|(op, text)| (before + op, text));
        self.pending.texts.extend(texts);
        Ok(outcomes)
    }

    /// The ops to log, and the objects they leave still to be given a
    /// vector, in the order they are last written. Each is put down to the
    /// batch that first left it with the text it ends with and no vector:
    /// a later batch that only changed something else is not to blame.
    pub fn finish(self) -> (Vec<Op>, Vec<Wanted>) {
        let Pending {
            catalog,
            ops,
            ends,
            texts,
            ..
        } = self.pending;
        let mut wanting: HashMap<RowKey, Wanted> = HashMap::new();
        for (op, written) in ops.iter().enumerate() {
            let batch = ends.partition_point(|end| *end <= op);
            match written {
                Op::Upsert { collection, object } => {
                    let key = (*collection, object.id);
                    // An object of a collection not storing all it embeds
                    // is embedded only from the text its write was given.
                    let text = |schema: &Collection| match schema.has_unstored() {
                        true => texts.get(&op).cloned(),
                        false => wanting_text(schema, object),
                    };
                    let needed = catalog
                        .schema(*collection)
                        .and_then(|schema| Some((schema, text(schema)?)));
                    match needed {
                        Some((schema, text)) => {
                            let since = wanting
                                .remove(&key)
                                .filter(|earlier| earlier.text == text)
                                .map_or(batch, |earlier| earlier.batch);
                            let embedding = schema.embedding.clone();
                            wanting.insert(
                                key,
                                Wanted {
                                    op,
                                    batch: since,
                                    embedding,
                                    text,
                                },
                            );
                        }
                        None => {
                            wanting.remove(&key);
                        }
                    }
                }
                Op::Delete { collection, id } => {
                    wanting.remove(&(*collection, *id));
                }
                Op::CreateCollection { .. }
                | Op::AddProperty { .. }
                | Op::DropCollection { .. } => {}
            }
        }
        let mut wanted: Vec<Wanted> = wanting.into_values().collect();
        wanted.sort_unstable_by_key(|wanted| wanted.op);
        (ops, wanted)
    }
}

fn invalid(reason: String) -> StorageError {
    StorageError::InvalidMutation(reason)
}

/// Refuse a change, without a vector, to any of `changed` that the
/// collection embeds, where it does not store all it embeds: the text to
/// embed again is not all there. Whether it is refused depends on the
/// change and the schema alone, not on the object or the commit.
fn unseen<'a>(schema: &Collection, mut changed: impl Iterator<Item = &'a String>) -> Result<()> {
    if !schema.has_unstored() {
        return Ok(());
    }
    let embedded = &schema.embedding.properties;
    let Some(property) = changed.find(|name| embedded.contains(name)) else {
        return Ok(());
    };
    let unstored: Vec<&str> = schema
        .properties
        .iter()
        .filter(|p| !p.stored)
        .map(|p| p.name.as_str())
        .collect();
    Err(invalid(format!(
        "{} embeds {property:?} with {unstored:?}, which it does not store, so a change to \
         {property:?} must bring its vector, or write the whole object",
        schema.name
    )))
}

/// A request that would delete more by filter than one may, or may.
fn too_many_deleted(message: String) -> StorageError {
    StorageError::TooMany {
        over: Over {
            capped: Capped::Deleted,
            most: DELETED_AT_MOST as u64,
        },
        message,
    }
}

impl Batch<'_> {
    fn row(&self, key: RowKey) -> Option<Live<'_>> {
        match self
            .rows
            .get(&key)
            .or_else(|| self.draft.pending.rows.get(&key))
        {
            Some(pending) => pending.as_ref().map(|object| Live {
                object,
                vector: object.vector.as_deref(),
            }),
            None => self.draft.state.row(key.0, key.1),
        }
    }

    fn collection(&self, name: &CollectionName) -> Result<(CollectionId, Collection)> {
        self.catalog
            .known(name)
            .map(|(id, schema)| (id, schema.clone()))
    }

    fn stored(&self, name: &CollectionName, key: RowKey) -> Result<Object> {
        self.row(key)
            .map(|found| found.to_object())
            .ok_or_else(|| StorageError::UnknownObject {
                collection: name.clone(),
                id: key.1,
            })
    }

    /// The version the object under `key` is at, which there must be one
    /// of, as the batch sees it: a conditional write on an object that is
    /// not there is not found, whatever the condition.
    fn version_of(&self, name: &CollectionName, key: RowKey) -> Result<Version> {
        self.row(key)
            .map(|found| found.version())
            .ok_or_else(|| StorageError::UnknownObject {
                collection: name.clone(),
                id: key.1,
            })
    }

    /// That the object under `key` is at `version`, if a write asks it.
    fn at_version(
        &self,
        name: &CollectionName,
        key: RowKey,
        version: Option<Version>,
    ) -> Result<()> {
        let Some(version) = version else {
            return Ok(());
        };
        let stored = self.version_of(name, key)?;
        if stored != version {
            return Err(StorageError::StaleVersion {
                collection: name.clone(),
                id: key.1,
                stored,
            });
        }
        Ok(())
    }

    /// Write `next` under its key, unless that is what is stored already.
    ///
    /// Where the collection embeds its own objects and the caller
    /// `supplied` no vector, the vector follows the text: the one stored
    /// for the same text if there is one, in this commit or before it, and
    /// otherwise none, to be made before the commit is logged.
    ///
    /// A collection that does not store all it embeds keeps none of that
    /// text: not here, nor in what the commit logs. So a stored row's text
    /// cannot be compared with a `whole` object's, which the caller wrote
    /// in full: that is always embedded afresh, from the text it was given.
    /// One made from a stored row and a change to it has the vector the
    /// row has, as the change is refused before here if it changes what is
    /// embedded.
    fn write(
        &mut self,
        collection: CollectionId,
        schema: &Collection,
        mut next: Object,
        supplied: bool,
        whole: bool,
    ) -> Outcome {
        let key = (collection, next.id);
        let embedding = &schema.embedding;
        let unstored = schema.has_unstored();
        let mut text = None;
        if !supplied && embedding.provider != EmbeddingProvider::None {
            let embedded = next.embedded_text(embedding);
            next.vector = match unstored && whole {
                true => None,
                false => [self.row(key), self.draft.state.row(collection, next.id)]
                    .into_iter()
                    .flatten()
                    .filter(|stored| stored.object.embedded_text(embedding) == embedded)
                    .find_map(|stored| stored.vector.map(<[f32]>::to_vec)),
            };
            text = (unstored && whole && !embedded.trim().is_empty()).then_some(embedded);
        }
        if unstored {
            next = schema.stripped(next);
        }
        if self.row(key).is_some_and(|stored| stored.is(&next)) {
            return Outcome::Unchanged;
        }
        if let Some(text) = text {
            self.texts.insert(self.ops.len(), text);
        }
        self.rows.insert(key, Some(next.clone()));
        self.ops.push(Op::Upsert {
            collection,
            object: next,
        });
        Outcome::Applied
    }

    fn resolve(&mut self, mutation: Mutation) -> Result<Outcome> {
        match mutation {
            Mutation::CreateCollection(schema) => {
                let op = self.catalog.create_collection(schema)?;
                Ok(op.map_or(Outcome::Unchanged, |op| {
                    self.ops.push(op);
                    Outcome::Applied
                }))
            }
            Mutation::AddProperty {
                collection,
                property,
            } => {
                let op = self.catalog.add_property(&collection, property)?;
                Ok(op.map_or(Outcome::Unchanged, |op| {
                    self.ops.push(op);
                    Outcome::Applied
                }))
            }
            Mutation::DropCollection(collection) => {
                let op = self.catalog.drop_collection(&collection)?;
                self.ops.push(op);
                Ok(Outcome::Applied)
            }
            Mutation::Upsert {
                collection,
                id,
                properties,
                vector,
                only_if_version,
            } => {
                let (cid, schema) = self.collection(&collection)?;
                self.at_version(&collection, (cid, id), only_if_version)?;
                let supplied = vector.is_some();
                let object = Object::from_json(&schema, id, &properties, vector)?;
                Ok(self.write(cid, &schema, object, supplied, true))
            }
            Mutation::Insert {
                collection,
                id,
                properties,
                vector,
            } => {
                let (cid, schema) = self.collection(&collection)?;
                if self.row((cid, id)).is_some() {
                    return Err(StorageError::ObjectExists { collection, id });
                }
                let supplied = vector.is_some();
                let object = Object::from_json(&schema, id, &properties, vector)?;
                Ok(self.write(cid, &schema, object, supplied, true))
            }
            Mutation::Patch {
                collection,
                id,
                properties,
                vector,
                only_if,
                only_if_version,
            } => {
                let (cid, schema) = self.collection(&collection)?;
                let stored = self.stored(&collection, (cid, id))?;
                if let Some(Condition { property, equals }) = only_if {
                    let expected = Value::typed(&schema, &property, &equals)?;
                    if stored.properties.get(&property) != expected.as_ref() {
                        return Err(StorageError::ConditionFailed {
                            collection,
                            id,
                            property,
                            stored: stored.version(),
                        });
                    }
                }
                self.at_version(&collection, (cid, id), only_if_version)?;
                let supplied = vector.is_some();
                if !supplied {
                    unseen(&schema, properties.keys())?;
                }
                let patched = stored.patched(&schema, &properties, vector)?;
                Ok(self.write(cid, &schema, patched, supplied, false))
            }
            Mutation::Increment {
                collection,
                id,
                property,
                by,
            } => {
                let (cid, schema) = self.collection(&collection)?;
                let mut object = self.stored(&collection, (cid, id))?;
                let value = incremented(&schema, &object, &property, by)?;
                if by != 0.0 {
                    object.properties.insert(property, value.clone());
                    self.write(cid, &schema, object, false, false);
                }
                Ok(Outcome::Value(value))
            }
            Mutation::Append {
                collection,
                id,
                property,
                values,
                unique,
            } => {
                let (cid, schema) = self.collection(&collection)?;
                unseen(&schema, std::iter::once(&property))?;
                let mut object = self.stored(&collection, (cid, id))?;
                let Some(appended) = appended(&schema, &object, &property, values, unique)? else {
                    return Ok(Outcome::Unchanged);
                };
                object
                    .properties
                    .insert(property, Value::TextArray(appended));
                Ok(self.write(cid, &schema, object, false, false))
            }
            Mutation::Delete {
                collection,
                id,
                only_if_version,
            } => {
                let (cid, _) = self.collection(&collection)?;
                self.at_version(&collection, (cid, id), only_if_version)?;
                if self.row((cid, id)).is_none() {
                    return Ok(Outcome::Unchanged);
                }
                self.delete(cid, id);
                Ok(Outcome::Applied)
            }
            Mutation::DeleteWhere {
                collection,
                filter,
                limit,
            } => {
                let (cid, schema) = self.collection(&collection)?;
                let predicate = filter.compile(&schema)?;
                if predicate.is_unconditional() {
                    return Err(invalid(format!(
                        "a delete from {collection} needs a condition; drop the collection to empty it"
                    )));
                }
                // Only a limit past the most is told the most: one of
                // none is not helped by it.
                let out_of_range = format!("a delete takes a limit of 1 to {DELETED_AT_MOST}");
                match limit {
                    Some(0) => return Err(invalid(out_of_range)),
                    Some(limit) if limit > DELETED_AT_MOST => {
                        return Err(too_many_deleted(out_of_range))
                    }
                    _ => {}
                }
                let (doomed, more) =
                    self.passing(cid, &predicate, limit.unwrap_or(DELETED_AT_MOST));
                if more && limit.is_none() {
                    return Err(too_many_deleted(format!(
                        "more than {DELETED_AT_MOST} objects of {collection} pass the filter, \
                         which is the most one delete takes; give the delete a limit, \
                         and ask again until it deletes fewer than that"
                    )));
                }
                doomed.iter().for_each(|id| self.delete(cid, *id));
                Ok(Outcome::Deleted(doomed.len() as u64))
            }
        }
    }

    fn delete(&mut self, collection: CollectionId, id: ObjectId) {
        self.rows.insert((collection, id), None);
        self.ops.push(Op::Delete { collection, id });
    }

    /// The ids of the collection's rows, as this batch sees them, that pass
    /// `predicate`: stored rows nothing pending has replaced, and pending
    /// rows, a batch's own taking precedence. At most `most` of them,
    /// whichever are found first as the rows happen to be stored, with
    /// whether there are more, found without looking further; those found
    /// are deleted in id order.
    fn passing(
        &self,
        collection: CollectionId,
        predicate: &Predicate,
        most: usize,
    ) -> (BTreeSet<ObjectId>, bool) {
        let pending = &self.draft.pending.rows;
        let overlaid = |id: ObjectId| {
            let key = (collection, id);
            self.rows.contains_key(&key) || pending.contains_key(&key)
        };
        let stored = self
            .draft
            .state
            .matching(collection, predicate)
            .map(|found| found.id())
            .filter(|id| !overlaid(*id));
        let written = pending
            .iter()
            .filter(|(key, _)| !self.rows.contains_key(key))
            .chain(&self.rows)
            .filter(|((c, _), _)| *c == collection)
            .filter_map(|(_, row)| row.as_ref())
            .filter(|object| predicate.matches(object))
            .map(|object| object.id);
        let mut found = stored.chain(written);
        let passing: BTreeSet<ObjectId> = found.by_ref().take(most).collect();
        let more = found.next().is_some();
        (passing, more)
    }
}

fn declared(schema: &Collection, property: &str) -> Result<DataType> {
    schema
        .property(property)
        .map(|p| p.data_type)
        .ok_or_else(|| ObjectError::UnknownProperty(property.to_string()).into())
}

/// The property's value after adding `by`. A stored value always has its
/// declared type, because a property's type never changes, so anything else
/// is the absent case and counts as zero.
fn incremented(schema: &Collection, object: &Object, property: &str, by: f64) -> Result<Value> {
    let out_of_range = || {
        invalid(format!(
            "incrementing {property:?} by {by} leaves its range"
        ))
    };
    match (declared(schema, property)?, object.properties.get(property)) {
        (DataType::Int, current) => {
            let current = match current {
                Some(Value::Int(n)) => *n,
                _ => 0,
            };
            // `i64::MAX as f64` rounds up to 2^63, so this admits exactly the
            // whole numbers the cast below represents without saturating.
            if by.fract() != 0.0 || by.abs() >= i64::MAX as f64 {
                return Err(invalid(format!(
                    "{property:?} is an int and cannot move by {by}"
                )));
            }
            current
                .checked_add(by as i64)
                .map(Value::Int)
                .ok_or_else(out_of_range)
        }
        (DataType::Number, current) => {
            let current = match current {
                Some(Value::Number(n)) => *n,
                _ => 0.0,
            };
            Some(current + by)
                .filter(|n| n.is_finite())
                .map(Value::Number)
                .ok_or_else(out_of_range)
        }
        (other, _) => Err(invalid(format!(
            "{property:?} is {other} and cannot be incremented"
        ))),
    }
}

/// The array after appending `values`, or `None` when that adds nothing.
fn appended(
    schema: &Collection,
    object: &Object,
    property: &str,
    values: Vec<String>,
    unique: bool,
) -> Result<Option<Vec<String>>> {
    if declared(schema, property)? != DataType::TextArray {
        return Err(invalid(format!(
            "{property:?} is not a text array and cannot be appended to"
        )));
    }
    let stored = match object.properties.get(property) {
        Some(Value::TextArray(stored)) => stored.clone(),
        _ => Vec::new(),
    };
    let before = stored.len();
    let appended = values.into_iter().fold(stored, |mut all, value| {
        if !(unique && all.contains(&value)) {
            all.push(value);
        }
        all
    });
    Ok((appended.len() > before).then_some(appended))
}
