use serde_json::{Map, Value as Json};
use vellev_core::{Collection, CollectionName, ObjectId, Property, Value};

use crate::coordinator::LogPosition;

/// A change a caller asks for, by collection name. The writer resolves it
/// against the schema and the current data into what the log records: a
/// whole object to write, or a key to delete. Because only the writer
/// resolves, and it does so one batch after another, a read-modify-write
/// such as [`Mutation::Increment`] cannot lose an update.
#[derive(Debug, Clone, PartialEq)]
pub enum Mutation {
    CreateCollection(Box<Collection>),
    AddProperty {
        collection: CollectionName,
        property: Property,
    },
    DropCollection(CollectionName),
    /// Write the whole object, replacing any stored one.
    Upsert {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
    },
    /// Write the object only if none is stored under its id.
    Insert {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
    },
    /// Merge into a stored object: a property sent as JSON null is removed,
    /// and the vector is replaced when one is given.
    Patch {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
        only_if: Option<Condition>,
    },
    /// Add `by` to a numeric property, counting an absent value as zero.
    Increment {
        collection: CollectionName,
        id: ObjectId,
        property: String,
        by: f64,
    },
    /// Add `values` to the end of a text array, counting an absent value as
    /// empty. With `unique`, a value already present is not added again.
    Append {
        collection: CollectionName,
        id: ObjectId,
        property: String,
        values: Vec<String>,
        unique: bool,
    },
    Delete {
        collection: CollectionName,
        id: ObjectId,
    },
}

/// Holds when the stored object's `property` equals `equals`, where JSON
/// null means the property has no value.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub property: String,
    pub equals: Json,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Applied,
    /// The mutation was valid and changed nothing, so nothing was logged.
    Unchanged,
    /// The property's value after an [`Mutation::Increment`].
    Value(Value),
}

/// What one caller's batch did, and the position at which it is visible.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    pub position: LogPosition,
    pub outcomes: Vec<Outcome>,
}
