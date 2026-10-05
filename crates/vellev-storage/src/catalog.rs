use std::collections::BTreeMap;

use serde_json::{Map, Value as Json};
use vellev_core::{Collection, CollectionName, Object, ObjectId, Property};

use crate::format::{CollectionId, Op};
use crate::{Result, StorageError};

/// A change a caller asks for, by collection name. The writer resolves it
/// against the catalog into an [`Op`], which is what the log records.
#[derive(Debug, Clone, PartialEq)]
pub enum Mutation {
    CreateCollection(Box<Collection>),
    AddProperty {
        collection: CollectionName,
        property: Property,
    },
    DropCollection(CollectionName),
    Upsert {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
    },
    Delete {
        collection: CollectionName,
        id: ObjectId,
    },
}

/// The partition's collections, by name and by id.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Catalog {
    next_id: u64,
    names: BTreeMap<CollectionName, CollectionId>,
    schemas: BTreeMap<CollectionId, Collection>,
}

impl Catalog {
    pub fn restore(next_id: u64, collections: Vec<(CollectionId, Collection)>) -> Self {
        Self {
            next_id,
            names: collections
                .iter()
                .map(|(id, schema)| (schema.name.clone(), *id))
                .collect(),
            schemas: collections.into_iter().collect(),
        }
    }

    pub fn next_id(&self) -> u64 {
        self.next_id
    }

    pub fn collections(&self) -> impl Iterator<Item = (CollectionId, &Collection)> {
        self.schemas.iter().map(|(id, schema)| (*id, schema))
    }

    pub fn contains(&self, id: CollectionId) -> bool {
        self.schemas.contains_key(&id)
    }

    pub fn get(&self, name: &CollectionName) -> Option<(CollectionId, &Collection)> {
        let id = *self.names.get(name)?;
        self.schemas.get(&id).map(|schema| (id, schema))
    }

    fn known(&self, name: &CollectionName) -> Result<(CollectionId, &Collection)> {
        self.get(name)
            .ok_or_else(|| StorageError::UnknownCollection(name.clone()))
    }

    /// Validate `mutation` and turn it into the op to log, applying schema
    /// changes here so later mutations in the same batch see them. `None`
    /// means the mutation changes nothing.
    pub fn resolve(&mut self, mutation: Mutation) -> Result<Option<Op>> {
        let op = match mutation {
            Mutation::CreateCollection(schema) => {
                schema.validate()?;
                if self.names.contains_key(&schema.name) {
                    return Err(StorageError::CollectionExists(schema.name.clone()));
                }
                Op::CreateCollection {
                    id: CollectionId(self.next_id),
                    schema,
                }
            }
            Mutation::AddProperty {
                collection,
                property,
            } => {
                let (id, schema) = self.known(&collection)?;
                if !schema.clone().add_property(property.clone())? {
                    return Ok(None);
                }
                Op::AddProperty {
                    collection: id,
                    property,
                }
            }
            Mutation::DropCollection(collection) => Op::DropCollection {
                collection: self.known(&collection)?.0,
            },
            Mutation::Upsert {
                collection,
                id,
                properties,
                vector,
            } => {
                let (collection, schema) = self.known(&collection)?;
                Op::Upsert {
                    collection,
                    object: Object::from_json(schema, id, &properties, vector)?,
                }
            }
            Mutation::Delete { collection, id } => Op::Delete {
                collection: self.known(&collection)?.0,
                id,
            },
        };
        self.apply(&op)
            .map_err(|reason| StorageError::corrupt("catalog", reason))?;
        Ok(Some(op))
    }

    /// Apply the schema effect of a logged op. An op the catalog cannot apply
    /// means the log does not belong to this state, and the reason is returned.
    pub fn apply(&mut self, op: &Op) -> std::result::Result<(), String> {
        match op {
            Op::CreateCollection { id, schema } => {
                if self.contains(*id) || self.names.contains_key(&schema.name) {
                    return Err(format!("collection {} is created twice", schema.name));
                }
                self.next_id = self.next_id.max(id.0 + 1);
                self.names.insert(schema.name.clone(), *id);
                self.schemas.insert(*id, schema.as_ref().clone());
            }
            Op::AddProperty {
                collection,
                property,
            } => {
                self.schemas
                    .get_mut(collection)
                    .ok_or_else(|| format!("property added to unknown collection {collection:?}"))?
                    .add_property(property.clone())
                    .map_err(|e| e.to_string())?;
            }
            Op::DropCollection { collection } => {
                let schema = self
                    .schemas
                    .remove(collection)
                    .ok_or_else(|| format!("unknown collection {collection:?} is dropped"))?;
                self.names.remove(&schema.name);
            }
            Op::Upsert { collection, .. } | Op::Delete { collection, .. } => {
                if !self.contains(*collection) {
                    return Err(format!("row written to unknown collection {collection:?}"));
                }
            }
        }
        Ok(())
    }
}
