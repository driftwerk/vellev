//! Types shared by every vellev crate: tenants, collections, objects and schema.

mod ids;
mod object;
mod schema;
mod version;

pub use ids::{CollectionName, IdError, NodeId, ObjectId, Partition, RequestId, TenantId};
pub use object::{Object, ObjectError, Value, SQUARED_LENGTH_UNDER};
pub use schema::{
    AccessRule, Bm25, Collection, DataType, Distance, Embedding, EmbeddingProvider, HttpApi,
    ModelFault, Property, SchemaError, MAX_DIMENSIONS,
};
pub use version::{Version, VersionError};
