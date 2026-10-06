//! Types shared by every vellev crate: tenants, collections, objects and schema.

mod ids;
mod object;
mod schema;

pub use ids::{CollectionName, IdError, NodeId, ObjectId, Partition, RequestId, TenantId};
pub use object::{Object, ObjectError, Value};
pub use schema::{
    AccessRule, Bm25, Collection, DataType, Distance, Embedding, EmbeddingProvider, HttpApi,
    Property, SchemaError, MAX_DIMENSIONS,
};
