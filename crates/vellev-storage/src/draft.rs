//! Resolving what callers ask for into what the log records.

use std::collections::{BTreeSet, HashMap};

use vellev_core::{Collection, CollectionName, DataType, Object, ObjectError, ObjectId, Value};

use vellev_index::filter::Predicate;

use crate::catalog::Catalog;
use crate::format::{CollectionId, Op};
use crate::mutation::{Condition, Mutation, Outcome};
use crate::state::State;
use crate::{Result, StorageError};

type RowKey = (CollectionId, ObjectId);

/// The changes one commit will make, built up batch by batch on top of a
/// state. A batch sees every batch accepted before it, and a batch that is
/// rejected leaves no trace.
pub(crate) struct Draft<'a> {
    state: &'a State,
    pending: Pending,
}

/// What has been resolved so far: the catalog as it will be, the rows that
/// will be written (`None` for a delete), and the ops in order.
struct Pending {
    catalog: Catalog,
    rows: HashMap<RowKey, Option<Object>>,
    ops: Vec<Op>,
}

/// One batch being resolved on top of a draft, kept apart until it is known
/// to be valid as a whole.
struct Batch<'a> {
    draft: &'a Draft<'a>,
    catalog: Catalog,
    rows: HashMap<RowKey, Option<Object>>,
    ops: Vec<Op>,
}

impl<'a> Draft<'a> {
    pub fn new(state: &'a State) -> Self {
        let pending = Pending {
            catalog: state.catalog().clone(),
            rows: HashMap::new(),
            ops: vec![],
        };
        Self { state, pending }
    }

    /// Resolve one caller's batch. Either all of it joins the draft, or, on
    /// the first mutation that cannot be applied, none of it does.
    pub fn batch(&mut self, mutations: Vec<Mutation>) -> Result<Vec<Outcome>> {
        let mut batch = Batch {
            draft: self,
            catalog: self.pending.catalog.clone(),
            rows: HashMap::new(),
            ops: vec![],
        };
        let outcomes = mutations
            .into_iter()
            .map(|m| batch.resolve(m))
            .collect::<Result<Vec<_>>>()?;
        let Batch {
            catalog, rows, ops, ..
        } = batch;
        self.pending.catalog = catalog;
        self.pending.rows.extend(rows);
        self.pending.ops.extend(ops);
        Ok(outcomes)
    }

    pub fn into_ops(self) -> Vec<Op> {
        self.pending.ops
    }
}

fn invalid(reason: String) -> StorageError {
    StorageError::InvalidMutation(reason)
}

impl Batch<'_> {
    fn row(&self, key: RowKey) -> Option<&Object> {
        match self
            .rows
            .get(&key)
            .or_else(|| self.draft.pending.rows.get(&key))
        {
            Some(pending) => pending.as_ref(),
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
            .cloned()
            .ok_or_else(|| StorageError::UnknownObject {
                collection: name.clone(),
                id: key.1,
            })
    }

    /// Write `next` under its key, unless that is what is stored already.
    fn write(&mut self, collection: CollectionId, next: Object) -> Outcome {
        if self.row((collection, next.id)) == Some(&next) {
            return Outcome::Unchanged;
        }
        self.rows.insert((collection, next.id), Some(next.clone()));
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
                self.ops.push(op);
                Ok(Outcome::Applied)
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
            } => {
                let (cid, schema) = self.collection(&collection)?;
                Ok(self.write(cid, Object::from_json(&schema, id, &properties, vector)?))
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
                Ok(self.write(cid, Object::from_json(&schema, id, &properties, vector)?))
            }
            Mutation::Patch {
                collection,
                id,
                properties,
                vector,
                only_if,
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
                        });
                    }
                }
                Ok(self.write(cid, stored.patched(&schema, &properties, vector)?))
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
                    self.write(cid, object);
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
                let mut object = self.stored(&collection, (cid, id))?;
                let Some(appended) = appended(&schema, &object, &property, values, unique)? else {
                    return Ok(Outcome::Unchanged);
                };
                object
                    .properties
                    .insert(property, Value::TextArray(appended));
                Ok(self.write(cid, object))
            }
            Mutation::Delete { collection, id } => {
                let (cid, _) = self.collection(&collection)?;
                if self.row((cid, id)).is_none() {
                    return Ok(Outcome::Unchanged);
                }
                self.delete(cid, id);
                Ok(Outcome::Applied)
            }
            Mutation::DeleteWhere { collection, filter } => {
                let (cid, schema) = self.collection(&collection)?;
                let predicate = filter.compile(&schema)?;
                if predicate.is_unconditional() {
                    return Err(invalid(format!(
                        "a delete from {collection} needs a condition; drop the collection to empty it"
                    )));
                }
                let doomed = self.passing(cid, &predicate);
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
    /// rows, a batch's own taking precedence. In id order, so that the ops a
    /// filter resolves to do not depend on how rows happen to be stored.
    fn passing(&self, collection: CollectionId, predicate: &Predicate) -> BTreeSet<ObjectId> {
        let pending = &self.draft.pending.rows;
        let overlaid = |id: ObjectId| {
            let key = (collection, id);
            self.rows.contains_key(&key) || pending.contains_key(&key)
        };
        let stored = self
            .draft
            .state
            .matching(collection, predicate)
            .map(|object| object.id)
            .filter(|id| !overlaid(*id));
        let written = pending
            .iter()
            .filter(|(key, _)| !self.rows.contains_key(key))
            .chain(&self.rows)
            .filter(|((c, _), _)| *c == collection)
            .filter_map(|(_, row)| row.as_ref())
            .filter(|object| predicate.matches(object))
            .map(|object| object.id);
        stored.chain(written).collect()
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
