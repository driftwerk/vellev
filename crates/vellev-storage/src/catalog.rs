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
pub(crate) struct Catalog {
    next_id: u64,
    names: BTreeMap<CollectionName, CollectionId>,
    schemas: BTreeMap<CollectionId, Collection>,
}

impl Catalog {
    /// Rebuild a catalog from a manifest, or say why the manifest cannot be
    /// right: ids and names must be unique and below `next_id`, which is what
    /// keeps a dropped collection's id from ever being given out again.
    pub fn restore(
        next_id: u64,
        collections: Vec<(CollectionId, Collection)>,
    ) -> std::result::Result<Self, String> {
        collections.into_iter().try_fold(
            Self {
                next_id,
                ..Self::default()
            },
            |mut catalog, (id, schema)| {
                if id.0 >= next_id {
                    return Err(format!("collection id {} is not below {next_id}", id.0));
                }
                catalog.create(id, schema)?;
                Ok(catalog)
            },
        )
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

    fn create(&mut self, id: CollectionId, schema: Collection) -> std::result::Result<(), String> {
        if self.contains(id) || self.names.contains_key(&schema.name) {
            return Err(format!(
                "collection {} ({}) exists twice",
                schema.name, id.0
            ));
        }
        let after = id.0.checked_add(1).ok_or("collection ids are exhausted")?;
        self.next_id = self.next_id.max(after);
        self.names.insert(schema.name.clone(), id);
        self.schemas.insert(id, schema);
        Ok(())
    }

    /// Validate `mutation` and turn it into the op to log, applying schema
    /// changes here so later mutations in the same batch see them. `None`
    /// means the mutation changes nothing.
    pub fn resolve(&mut self, mutation: Mutation) -> Result<Option<Op>> {
        Ok(Some(match mutation {
            Mutation::CreateCollection(schema) => {
                schema.validate()?;
                if self.names.contains_key(&schema.name) {
                    return Err(StorageError::CollectionExists(schema.name.clone()));
                }
                let id = CollectionId(self.next_id);
                self.create(id, schema.as_ref().clone())
                    .map_err(|_| StorageError::Exhausted("collection ids"))?;
                Op::CreateCollection { id, schema }
            }
            Mutation::AddProperty {
                collection,
                property,
            } => {
                let (id, _) = self.known(&collection)?;
                let added = self
                    .schemas
                    .get_mut(&id)
                    .map(|s| s.add_property(property.clone()));
                if !added.transpose()?.unwrap_or(false) {
                    return Ok(None);
                }
                Op::AddProperty {
                    collection: id,
                    property,
                }
            }
            Mutation::DropCollection(collection) => {
                let (id, _) = self.known(&collection)?;
                self.schemas.remove(&id);
                self.names.remove(&collection);
                Op::DropCollection { collection: id }
            }
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
        }))
    }

    /// Apply the schema effect of a logged op. An op the catalog cannot apply
    /// means the log does not belong to this state, and the reason is returned.
    pub fn apply(&mut self, op: &Op) -> std::result::Result<(), String> {
        let unknown = |id: &CollectionId| format!("collection {} is unknown", id.0);
        match op {
            Op::CreateCollection { id, .. } if id.0 < self.next_id => {
                Err(format!("collection id {} was already given out", id.0))
            }
            Op::CreateCollection { id, schema } => self.create(*id, schema.as_ref().clone()),
            Op::AddProperty {
                collection,
                property,
            } => self
                .schemas
                .get_mut(collection)
                .ok_or_else(|| unknown(collection))?
                .add_property(property.clone())
                .map(|_| ())
                .map_err(|e| e.to_string()),
            Op::DropCollection { collection } => {
                let schema = self
                    .schemas
                    .remove(collection)
                    .ok_or_else(|| unknown(collection))?;
                self.names.remove(&schema.name);
                Ok(())
            }
            Op::Upsert { collection, .. } | Op::Delete { collection, .. } => self
                .contains(*collection)
                .then_some(())
                .ok_or_else(|| unknown(collection)),
        }
    }
}
