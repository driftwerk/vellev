use std::collections::BTreeMap;

use vellev_core::{Collection, CollectionName, Property};

use crate::format::{CollectionId, Op};
use crate::{Result, StorageError};

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

    pub fn known(&self, name: &CollectionName) -> Result<(CollectionId, &Collection)> {
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

    /// Validate a new collection, add it, and return the op to log.
    pub fn create_collection(&mut self, schema: Box<Collection>) -> Result<Op> {
        schema.validate()?;
        if self.names.contains_key(&schema.name) {
            return Err(StorageError::CollectionExists(schema.name.clone()));
        }
        let id = CollectionId(self.next_id);
        self.create(id, schema.as_ref().clone())
            .map_err(|_| StorageError::Exhausted("collection ids"))?;
        Ok(Op::CreateCollection { id, schema })
    }

    /// Add a property and return the op to log, or `None` when the collection
    /// already has exactly that property.
    pub fn add_property(
        &mut self,
        collection: &CollectionName,
        property: Property,
    ) -> Result<Option<Op>> {
        let (id, _) = self.known(collection)?;
        let added = self
            .schemas
            .get_mut(&id)
            .map(|schema| schema.add_property(property.clone()));
        Ok(added
            .transpose()?
            .unwrap_or(false)
            .then_some(Op::AddProperty {
                collection: id,
                property,
            }))
    }

    pub fn drop_collection(&mut self, collection: &CollectionName) -> Result<Op> {
        let (id, _) = self.known(collection)?;
        self.schemas.remove(&id);
        self.names.remove(collection);
        Ok(Op::DropCollection { collection: id })
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
