use serde_json::{Map, Value as Json};
use vellev_core::{Collection, CollectionName, Embedding, ObjectId, Property, RequestId, Value};

use vellev_index::filter::Filter;

use crate::coordinator::{Incarnation, LogPosition};

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
    /// Add `by` to a numeric property, counting an absent value as zero. An
    /// int moves only by whole numbers, which `by` holds exactly up to 2^53.
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
    /// Delete every object that passes `filter`, as the batch sees them:
    /// including rows it wrote earlier, excluding rows it already deleted.
    /// A filter with no condition is refused, so that a caller who built an
    /// empty one by accident does not empty the collection.
    DeleteWhere {
        collection: CollectionName,
        filter: Filter,
    },
}

/// Turns text into vectors for the collections that embed their own
/// objects. A writer with one makes, as part of the commit, the vector of
/// every object written without one, unless the object's embedded text
/// already has a vector stored for it.
#[async_trait::async_trait]
pub trait Vectorize: std::fmt::Debug + Send + Sync {
    /// A vector for each text, in order, as `embedding` embeds what it
    /// stores. No text is blank.
    async fn vectors(
        &self,
        embedding: &Embedding,
        texts: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, String>;
}

/// Holds when the stored object's `property` equals `equals`, where JSON
/// null means the property has no value.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub property: String,
    pub equals: Json,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    Applied,
    /// The mutation was valid and changed nothing, so nothing was logged.
    Unchanged,
    /// The property's value after an [`Mutation::Increment`].
    Value(Value),
    /// How many objects a [`Mutation::DeleteWhere`] deleted.
    Deleted(u64),
}

/// One caller's mutations, made as one atomic step, with the request they
/// came in if the caller named it: a request already answered is answered
/// again as it was, and made no second time.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Batch {
    pub request: Option<RequestId>,
    pub mutations: Vec<Mutation>,
}

impl From<Vec<Mutation>> for Batch {
    fn from(mutations: Vec<Mutation>) -> Self {
        Self {
            request: None,
            mutations,
        }
    }
}

/// What one caller's batch did, and the position at which it is visible.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    /// The partition as registered when the commit was made; with the
    /// position, what a reader needs to have caught up to.
    pub incarnation: Incarnation,
    pub position: LogPosition,
    pub outcomes: Vec<Outcome>,
}
