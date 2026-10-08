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
    /// This build cannot run the model at all.
    #[error("{0}")]
    Unsupported(String),
    /// Asking the model failed in a way that may pass.
    #[error("embedding with {model} failed: {reason}")]
    Failed { model: String, reason: String },
    #[error("{model} returned {reason}")]
    Malformed { model: String, reason: String },
    /// The model cannot be used as this node and the collection are set
    /// up: the node may not write where the model is kept, whatever
    /// serves it says in its own code for it that it has no such model or
    /// will not serve this key's account, or its vectors are not the
    /// collection's size.
    #[error("{model} cannot be used: {reason}")]
    Unusable { model: String, reason: String },
}

impl EmbedError {
    /// Whether asking this node again cannot help until its setup is put
    /// right: something settled by how the node is built and set up, or
    /// said outright by whatever serves the model. Nothing else is
    /// lasting, however likely: a failure taken for lasting when it would
    /// have passed is the costlier mistake. So a bare refusal of a key, a
    /// model's input, or a node with no credentials or no network to
    /// fetch a model over, may pass, and is not.
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
