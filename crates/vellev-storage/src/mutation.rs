use serde_json::{Map, Value as Json};
use vellev_core::{
    Collection, CollectionName, Embedding, ModelFault, ObjectId, Property, RequestId, Value,
    Version,
};

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
    /// Write the whole object, replacing any stored one. With
    /// `only_if_version`, only if one is stored and is at that version.
    Upsert {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
        only_if_version: Option<Version>,
    },
    /// Write the object only if none is stored under its id.
    Insert {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
    },
    /// Merge into a stored object: a property sent as JSON null is removed,
    /// and the vector is replaced when one is given. Only if the stored
    /// object passes `only_if` and is at `only_if_version`, where given;
    /// both must hold.
    Patch {
        collection: CollectionName,
        id: ObjectId,
        properties: Map<String, Json>,
        vector: Option<Vec<f32>>,
        only_if: Option<Condition>,
        only_if_version: Option<Version>,
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
    /// Delete the object, if there is one: deleting none changes nothing.
    /// With `only_if_version`, there must be one, at that version.
    Delete {
        collection: CollectionName,
        id: ObjectId,
        only_if_version: Option<Version>,
    },
    /// Delete every object that passes `filter`, as the batch sees them:
    /// including rows it wrote earlier, excluding rows it already deleted.
    /// A filter with no condition is refused, so that a caller who built an
    /// empty one by accident does not empty the collection.
    ///
    /// With no `limit`, one that more than [`DELETED_AT_MOST`] objects pass
    /// is refused, and deletes nothing: it would be a commit of any size,
    /// made with every other write waiting, and a delete of some and not
    /// the rest is not what was asked for. With a `limit`, of one to
    /// [`DELETED_AT_MOST`], it is asked for: up to so many of the objects
    /// that pass are deleted, whichever are found first, and how many is
    /// told. Fewer than the limit means none were left, so a caller deletes
    /// however many pass by asking again, each time as a request of its
    /// own, under a name of its own, until it is told fewer; each is one
    /// commit, made or not, and a caller that stops part way has only to
    /// go on. What goes in the end is whatever passes until nothing does,
    /// not what passed at first.
    ///
    /// The deletes by filter of one batch count together, each for its
    /// limit and one without for [`DELETED_AT_MOST`], since each sees what
    /// those before it deleted: a batch of them that may delete more than
    /// one may is refused.
    DeleteWhere {
        collection: CollectionName,
        filter: Filter,
        limit: Option<usize>,
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
    ) -> std::result::Result<Vec<Vec<f32>>, Unembedded>;
}

/// Why vectors could not be made, and whether asking again could help.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Unembedded {
    pub reason: String,
    /// Where the fault lies, if nothing will make them until the node or
    /// the collection is set up otherwise.
    pub lasting: Option<ModelFault>,
}

impl Unembedded {
    /// A failure that may pass.
    pub fn for_now(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            lasting: None,
        }
    }

    /// A failure that will not pass until `fault` is put right.
    pub fn for_good(fault: ModelFault, reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            lasting: Some(fault),
        }
    }
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

/// The most objects one [`Mutation::DeleteWhere`] deletes: as many
/// mutations as a commit takes from callers writing together.
pub const DELETED_AT_MOST: usize = 8_192;

/// About how much a commit has to carry: the mutations it is asked to
/// make, and the bytes they take, by which the commits that take long are
/// told from the rest before they are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Weight {
    pub rows: usize,
    pub bytes: usize,
}

impl Weight {
    /// Whether this is within `most` both in rows and in bytes.
    pub fn within(&self, most: &Weight) -> bool {
        self.rows <= most.rows && self.bytes <= most.bytes
    }
}

impl std::ops::Add for Weight {
    type Output = Weight;

    fn add(self, other: Weight) -> Weight {
        Weight {
            rows: self.rows.saturating_add(other.rows),
            bytes: self.bytes.saturating_add(other.bytes),
        }
    }
}

impl std::iter::Sum for Weight {
    fn sum<I: Iterator<Item = Weight>>(weights: I) -> Weight {
        weights.fold(Weight::default(), std::ops::Add::add)
    }
}

/// About how many bytes `value` takes: its text, and a word for anything
/// else.
fn json_bytes(value: &Json) -> usize {
    match value {
        Json::String(text) => text.len(),
        Json::Array(values) => values.iter().map(json_bytes).sum(),
        Json::Object(values) => properties_bytes(values),
        _ => 8,
    }
}

fn properties_bytes(properties: &Map<String, Json>) -> usize {
    properties
        .iter()
        .map(|(name, value)| name.len() + json_bytes(value))
        .sum()
}

impl Mutation {
    /// About how many bytes the mutation takes in the log: its properties
    /// and its vector, each number of which takes five there, and a little
    /// for whatever else it names.
    pub fn bytes(&self) -> usize {
        const NAMED: usize = 64;
        const NUMBER: usize = 5;
        let vector_bytes =
            |vector: &Option<Vec<f32>>| vector.as_ref().map_or(0, |vector| vector.len() * NUMBER);
        NAMED
            + match self {
                Mutation::Upsert {
                    properties, vector, ..
                }
                | Mutation::Insert {
                    properties, vector, ..
                }
                | Mutation::Patch {
                    properties, vector, ..
                } => properties_bytes(properties) + vector_bytes(vector),
                Mutation::Append { values, .. } => values.iter().map(String::len).sum(),
                Mutation::CreateCollection(_)
                | Mutation::AddProperty { .. }
                | Mutation::DropCollection(_)
                | Mutation::Increment { .. }
                | Mutation::Delete { .. }
                | Mutation::DeleteWhere { .. } => 0,
            }
    }
}

impl Mutation {
    /// How many rows the mutation may change: one, or for a delete by
    /// filter as many as one may delete, since how many it will is not
    /// known until it is resolved.
    pub fn rows(&self) -> usize {
        match self {
            Mutation::DeleteWhere { limit, .. } => {
                limit.map_or(DELETED_AT_MOST, |limit| limit.min(DELETED_AT_MOST))
            }
            _ => 1,
        }
    }
}

impl Batch {
    /// About how much the batch is to commit.
    pub fn weight(&self) -> Weight {
        Weight {
            rows: self.mutations.iter().map(Mutation::rows).sum(),
            bytes: self.mutations.iter().map(Mutation::bytes).sum(),
        }
    }
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
