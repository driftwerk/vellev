//! Turning text into vectors, with the model chosen per collection.

mod bedrock;
mod embedder;
mod http;
#[cfg(feature = "local")]
mod local;
mod registry;

pub use bedrock::Bedrock;
pub use embedder::Embedder;
pub use http::Http;
#[cfg(feature = "local")]
pub use local::Local;
pub use registry::{Embedders, Settings};

use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EmbedError {
    /// This build or this node cannot run the model at all.
    #[error("{0}")]
    Unsupported(String),
    /// Asking the model failed in a way that may pass.
    #[error("embedding with {model} failed: {reason}")]
    Failed { model: String, reason: String },
    #[error("{model} returned {reason}")]
    Malformed { model: String, reason: String },
    /// The model was reached and cannot be used as the collection asks:
    /// whatever serves it refuses this node for good, or does not have
    /// it, or its vectors are not the collection's size.
    #[error("{model} cannot be used: {reason}")]
    Unusable { model: String, reason: String },
}

impl EmbedError {
    /// Whether asking again cannot help until the node or the collection
    /// is set up otherwise. A failure that may be this one text's, or that
    /// may pass, is not lasting.
    pub fn lasting(&self) -> bool {
        match self {
            Self::Unsupported(_) | Self::Unusable { .. } => true,
            Self::Failed { .. } | Self::Malformed { .. } => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, EmbedError>;

/// What texts are embedded for. Someone is waiting on a query, so where a
/// model has a queue, queries go ahead of documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    Document,
    Query,
}

/// A model that turns texts into vectors.
#[async_trait]
pub trait Provider: std::fmt::Debug + Send + Sync {
    /// One vector for each text, in order. No text is blank.
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_model_that_cannot_be_run_or_used_fails_for_good() {
        let (model, reason) = (String::from("m"), String::from("said"));
        let failures = [
            (EmbedError::Unsupported(reason.clone()), true),
            (
                EmbedError::Unusable {
                    model: model.clone(),
                    reason: reason.clone(),
                },
                true,
            ),
            (
                EmbedError::Failed {
                    model: model.clone(),
                    reason: reason.clone(),
                },
                false,
            ),
            (EmbedError::Malformed { model, reason }, false),
        ];
        for (failure, lasting) in failures {
            assert_eq!(failure.lasting(), lasting, "{failure}");
        }
    }
}
