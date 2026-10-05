//! Turning text into vectors, with the model chosen per collection.

mod bedrock;
mod embedder;
mod http;
#[cfg(feature = "local")]
mod local;
mod registry;

pub use bedrock::{Bedrock, ATTEMPTS as BEDROCK_ATTEMPTS};
pub use embedder::{document, Embedder};
pub use http::Http;
#[cfg(feature = "local")]
pub use local::Local;
pub use registry::{Embedders, Settings};

use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EmbedError {
    #[error("{0}")]
    Unsupported(String),
    #[error("embedding with {model} failed: {reason}")]
    Failed { model: String, reason: String },
    #[error("{model} returned {reason}")]
    Malformed { model: String, reason: String },
}

pub type Result<T> = std::result::Result<T, EmbedError>;

/// A model that turns texts into vectors.
#[async_trait]
pub trait Provider: std::fmt::Debug + Send + Sync {
    /// One vector for each text, in order. No text is blank.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
}
