//! Turning text into vectors, with the model chosen per collection.

mod bedrock;
mod embedder;
mod http;
#[cfg(feature = "local")]
mod local;
mod registry;

pub use bedrock::Bedrock;
pub use embedder::{Embedder, FAULT_STANDS};
pub use http::Http;
#[cfg(feature = "local")]
pub use local::Local;
pub use registry::{Embedders, Settings};

use async_trait::async_trait;
use vellev_core::ModelFault;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EmbedError {
    /// This build cannot run the model, though another may: it was built
    /// without local models.
    #[error("{0}")]
    Unsupported(String),
    /// No build of this release runs such a model: its name is not one
    /// the release knows, or the model is not of that size. Whoever asks
    /// for a collection to be made with it has asked for what cannot be.
    /// Met in a collection already there, made by a release that knew
    /// other models, it is this build's fault like any other.
    #[error("{0}")]
    NoSuchModel(String),
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
    Unusable {
        model: String,
        reason: String,
        fault: ModelFault,
    },
}

impl EmbedError {
    /// Whether asking this node again cannot help until its setup is put
    /// right: something settled by how the node is built and set up, or
    /// said outright by whatever serves the model. Nothing else is
    /// lasting, however likely: a failure taken for lasting when it would
    /// have passed is the costlier mistake. So a bare refusal of a key, a
    /// model's input, or a node with no credentials or no network to
    /// fetch a model over, may pass, and is not.
    ///
    /// Where the fault lies, if the failure is lasting.
    pub fn lasting(&self) -> Option<ModelFault> {
        match self {
            Self::Unsupported(_) | Self::NoSuchModel(_) => Some(ModelFault::Build),
            Self::Unusable { fault, .. } => Some(*fault),
            Self::Failed { .. } | Self::Malformed { .. } => None,
        }
    }
}

pub type Result<T> = std::result::Result<T, EmbedError>;

/// `said`, by whatever serves a model, with anything shaped like a key
/// taken out: what is said goes to the node's log, and a refusal may quote
/// the key it refuses, whole or with its middle masked. Only the shapes
/// keys are known to have are taken for one, so that a model's name is
/// left as it is: what follows `Bearer`, an OpenAI key (`sk-`), a Hugging
/// Face token (`hf_` and twenty letters or digits or more) and an AWS
/// access key id (`AKIA` or `ASIA` and sixteen capitals or digits).
pub(crate) fn without_keys(said: &str) -> String {
    const REDACTED: &str = "<redacted>";
    let apart = |c: char| c.is_whitespace() || "\"'`,;:(){}[]=".contains(c);
    let key_shaped = |word: &str| {
        let aws = |prefix: &str| {
            word.strip_prefix(prefix).is_some_and(|rest| {
                rest.len() == 16
                    && rest
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
            })
        };
        let hugging_face = word.strip_prefix("hf_").is_some_and(|rest| {
            rest.len() >= 20 && rest.chars().all(|c| c.is_ascii_alphanumeric())
        });
        (word.starts_with("sk-") && word.len() > 3) || hugging_face || aws("AKIA") || aws("ASIA")
    };
    let mut out = String::with_capacity(said.len());
    let mut after_bearer = false;
    let mut rest = said;
    while !rest.is_empty() {
        let end = rest.find(apart).unwrap_or(rest.len());
        let (word, tail) = rest.split_at(end);
        if !word.is_empty() {
            match after_bearer || key_shaped(word) {
                true => out.push_str(REDACTED),
                false => out.push_str(word),
            }
            after_bearer = word.eq_ignore_ascii_case("bearer");
        }
        let gap = tail.find(|c| !apart(c)).unwrap_or(tail.len());
        out.push_str(&tail[..gap]);
        rest = &tail[gap..];
    }
    out
}

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
    fn what_is_shaped_like_a_key_is_taken_out_and_a_models_name_is_not() {
        for (said, kept) in [
            (
                "Incorrect API key provided: sk-proj-abc123XYZ. Find yours at example.test",
                "Incorrect API key provided: <redacted> Find yours at example.test",
            ),
            (
                r#"{"error":{"message":"Incorrect API key provided: sk-…abcd","code":"invalid_api_key"}}"#,
                r#"{"error":{"message":"Incorrect API key provided: <redacted>","code":"invalid_api_key"}}"#,
            ),
            ("key sk-****abcd refused", "key <redacted> refused"),
            ("key 'sk-ab...yz' refused", "key '<redacted>' refused"),
            (
                "Authorization: Bearer abc.def.ghi was refused",
                "Authorization: Bearer <redacted> was refused",
            ),
            (
                "authorization=bearer\tabc123",
                "authorization=bearer\t<redacted>",
            ),
            (
                "token hf_abcdefghij0123456789ABCD is not valid",
                "token <redacted> is not valid",
            ),
            (
                "The security token AKIAIOSFODNN7EXAMPLE is not valid",
                "The security token <redacted> is not valid",
            ),
            ("id ASIA0123456789ABCDEF, expired", "id <redacted>, expired"),
            // Not keys: models' names, and what only looks a little like one.
            (
                "model text-embedding-3-small not found",
                "model text-embedding-3-small not found",
            ),
            (
                "BAAI/bge-base-en-v1.5 via hf_hub is not sk- shaped",
                "BAAI/bge-base-en-v1.5 via hf_hub is not sk- shaped",
            ),
            (
                "amazon.titan-embed-text-v2:0 in AKIAshort or a task-list",
                "amazon.titan-embed-text-v2:0 in AKIAshort or a task-list",
            ),
            ("", ""),
        ] {
            assert_eq!(without_keys(said), kept, "{said}");
        }
    }

    #[test]
    fn only_a_model_that_cannot_be_run_or_used_fails_for_good() {
        let (model, reason) = (String::from("m"), String::from("said"));
        let failures = [
            (
                EmbedError::Unsupported(reason.clone()),
                Some(ModelFault::Build),
            ),
            (
                EmbedError::NoSuchModel(reason.clone()),
                Some(ModelFault::Build),
            ),
            (
                EmbedError::Unusable {
                    model: model.clone(),
                    reason: reason.clone(),
                    fault: ModelFault::Provider,
                },
                Some(ModelFault::Provider),
            ),
            (
                EmbedError::Failed {
                    model: model.clone(),
                    reason: reason.clone(),
                },
                None,
            ),
            (EmbedError::Malformed { model, reason }, None),
        ];
        for (failure, lasting) in failures {
            assert_eq!(failure.lasting(), lasting, "{failure}");
        }
    }
}
