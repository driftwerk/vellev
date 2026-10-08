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

use std::sync::LazyLock;

use async_trait::async_trait;
use regex::Regex;
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

impl EmbedError {
    // What a model's server says goes into these, and from them to the
    // node's log and sometimes its caller: each is made through here, so
    // that none is made with a key left in it.

    /// A failure of `model` that may pass, as `reason` says.
    pub(crate) fn failed(model: &str, reason: impl std::fmt::Display) -> Self {
        Self::Failed {
            model: model.to_string(),
            reason: without_keys(&reason.to_string()),
        }
    }

    /// An answer of `model` that could not be used, as `reason` says.
    pub(crate) fn malformed(model: &str, reason: impl std::fmt::Display) -> Self {
        Self::Malformed {
            model: model.to_string(),
            reason: without_keys(&reason.to_string()),
        }
    }

    /// `model` being one that cannot be used until `fault` is put right,
    /// as `reason` says.
    pub(crate) fn unusable(model: &str, fault: ModelFault, reason: impl std::fmt::Display) -> Self {
        Self::Unusable {
            model: model.to_string(),
            reason: without_keys(&reason.to_string()),
            fault,
        }
    }
}

/// What stands where a key was.
const REDACTED: &str = "<redacted>";

/// The shapes a key is known by, each with what is put in its place. They
/// are looked for anywhere in what is said, not word by word: a key comes
/// in a path, in markup, or with a full stop after it as readily as alone.
static KEYS: LazyLock<Vec<(Regex, String)>> = LazyLock::new(|| {
    // What an OpenAI or Anthropic key, or a Hugging Face token, is made of.
    const KEY: &str = "[A-Za-z0-9_-]";
    const TOKEN: &str = "[A-Za-z0-9]";
    // How the middle of a key is masked by whoever quotes it back.
    const MASK: &str = r"(?:\u{2026}|\.{3}|\*{2,})";
    let whole = REDACTED.to_string();
    let after = |kept: &str| format!("{kept}{REDACTED}");
    [
        // A JSON web token: three runs of base64url, the first of a header.
        (
            format!(r"\beyJ{KEY}{{4,}}\.{KEY}{{4,}}\.{KEY}{{4,}}"),
            whole.clone(),
        ),
        // What is sent as a bearer or basic credential, if it is as long
        // as one: the word is also said of credentials in passing.
        (
            r"(?i)\b((?:bearer|basic)\s+)[A-Za-z0-9._~+/=-]{16,}".to_string(),
            after("${1}"),
        ),
        // A key or token given in a query string, up to the next part.
        (
            r#"(?i)\b((?:api[_-]?key|access_token|token|key)=)[^&\s"'<>]+"#.to_string(),
            after("${1}"),
        ),
        // OpenAI's and Anthropic's keys, `sk-`, `sk-proj-` and `sk-ant-`
        // among them: masked, or at their length.
        (format!(r"\bsk-{KEY}*{MASK}{KEY}*"), whole.clone()),
        (format!(r"\bsk-{KEY}{{20,}}"), whole.clone()),
        // Hugging Face's tokens, likewise.
        (format!(r"\bhf_{TOKEN}*{MASK}{TOKEN}*"), whole.clone()),
        (format!(r"\bhf_{TOKEN}{{20,}}"), whole.clone()),
        // An AWS access key id.
        (r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b".to_string(), whole),
    ]
    .into_iter()
    .map(|(shape, instead)| (Regex::new(&shape).expect("a key's shape"), instead))
    .collect()
});

/// `said`, by whatever serves a model, with what is shaped like a key
/// taken out, wherever in it the key lies:
///
/// - a JSON web token;
/// - sixteen characters or more of a credential after `Bearer` or `Basic`;
/// - the value of `api_key`, `api-key`, `key`, `token` or `access_token`
///   in a query string;
/// - a key beginning `sk-`, of twenty characters or more after that, or
///   with its middle masked by `…`, `...` or asterisks;
/// - a Hugging Face token, `hf_` and twenty letters and digits or more, or
///   masked likewise;
/// - an AWS access key id, `AKIA` or `ASIA` and sixteen capitals and
///   digits.
///
/// Nothing else is: not a key with no shape of its own, such as an Azure
/// key of thirty-two hex digits or an AWS secret of forty characters,
/// which nothing tells from a hash or an id, and taking which for keys
/// would take much else. A model's name is left as it is.
pub(crate) fn without_keys(said: &str) -> String {
    KEYS.iter()
        .fold(said.to_string(), |said, (shape, instead)| {
            match shape.replace_all(&said, instead.as_str()) {
                std::borrow::Cow::Borrowed(_) => said,
                std::borrow::Cow::Owned(without) => without,
            }
        })
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
    fn what_is_shaped_like_a_key_is_taken_out_wherever_it_lies() {
        const SK: &str = "sk-proj-abcdefghijklmnopqrstuvwxyz012345";
        const ANT: &str = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz";
        const HF: &str = "hf_abcdefghij0123456789ABCD";
        const AWS: &str = "AKIAIOSFODNN7EXAMPLE";
        const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl";
        let cases = [
            // Alone, and as servers quote them back.
            (
                format!("Incorrect API key provided: {SK}."),
                "Incorrect API key provided: <redacted>.".to_string(),
            ),
            (
                format!("key {ANT} refused"),
                "key <redacted> refused".into(),
            ),
            (
                "Incorrect API key provided: sk-\u{2026}abcd".into(),
                "Incorrect API key provided: <redacted>".into(),
            ),
            (
                "key sk-ab...wxyz refused".into(),
                "key <redacted> refused".into(),
            ),
            (
                "key 'sk-****abcd' refused".into(),
                "key '<redacted>' refused".into(),
            ),
            // In a path, in markup, after a full stop.
            (
                format!("GET /v1/keys/{SK} failed"),
                "GET /v1/keys/<redacted> failed".into(),
            ),
            (format!("<{SK}>"), "<<redacted>>".into()),
            (
                format!("key.{SK} is not one"),
                "key.<redacted> is not one".into(),
            ),
            (
                format!("token {HF} is not valid"),
                "token <redacted> is not valid".into(),
            ),
            (format!("token {HF}."), "token <redacted>.".into()),
            (
                "token hf_ab\u{2026}wxyz is not valid".into(),
                "token <redacted> is not valid".into(),
            ),
            (
                "token hf_ab***wxyz is not valid".into(),
                "token <redacted> is not valid".into(),
            ),
            (
                format!("The security token {AWS} is not valid"),
                "The security token <redacted> is not valid".into(),
            ),
            (format!("{AWS}."), "<redacted>.".into()),
            (
                format!("Credential={AWS}/20261008/eu-west-2/bedrock/aws4_request"),
                "Credential=<redacted>/20261008/eu-west-2/bedrock/aws4_request".into(),
            ),
            (
                format!("<AWSAccessKeyId>{AWS}</AWSAccessKeyId>"),
                "<AWSAccessKeyId><redacted></AWSAccessKeyId>".into(),
            ),
            (
                "id ASIA0123456789ABCDEF, expired".into(),
                "id <redacted>, expired".into(),
            ),
            // Sent as a credential, or in a query string.
            (
                "Authorization: Bearer abcdefghij.klmnop_qrst was refused".into(),
                "Authorization: Bearer <redacted> was refused".into(),
            ),
            (
                "authorization: basic dXNlcjpwYXNzd29yZA== refused".into(),
                "authorization: basic <redacted> refused".into(),
            ),
            (
                format!("token {JWT} expired"),
                "token <redacted> expired".into(),
            ),
            (format!("Bearer {JWT}"), "Bearer <redacted>".into()),
            (
                "GET /embed?api_key=abc123&model=m&access_token=xyz failed".into(),
                "GET /embed?api_key=<redacted>&model=m&access_token=<redacted> failed".into(),
            ),
            (
                "https://e.test/v1?key=abc123 refused".into(),
                "https://e.test/v1?key=<redacted> refused".into(),
            ),
            (
                "?api-key=abc&token=def".into(),
                "?api-key=<redacted>&token=<redacted>".into(),
            ),
        ];
        for (said, kept) in cases {
            assert_eq!(without_keys(&said), kept, "{said}");
        }
        // Not keys: left exactly as they were said.
        for said in [
            "Missing bearer token",
            r#"{"token_type":"bearer","expires_in":3600}"#,
            "Bearer realm=\"api\"",
            "use the Bearer scheme, or Basic if you must",
            "model sk-embed-v2 not found",
            "model text-embedding-3-small not found",
            "BAAI/bge-base-en-v1.5 via hf_hub is not sk- shaped",
            "amazon.titan-embed-text-v2:0 in AKIAshort or a task-list",
            "AKIAIOSFODNN7EXAMPLEANDMORE is too long to be one",
            "a risk-assessment of ask-me-anything-about-embeddings-today",
            "max_token=5 and monkey=banana",
            "eyJ is how a token begins, a.b.c is not one",
            "a hash 0123456789abcdef0123456789abcdef has no shape of its own",
            "<redacted>",
            "",
        ] {
            assert_eq!(without_keys(said), said);
        }
    }

    #[test]
    fn a_failure_is_never_made_with_a_key_in_it() {
        let said = "refused sk-proj-abcdefghijklmnopqrstuvwxyz012345";
        let without = "refused <redacted>";
        assert_eq!(
            EmbedError::failed("m", said),
            EmbedError::Failed {
                model: "m".into(),
                reason: without.into()
            }
        );
        assert_eq!(
            EmbedError::malformed("m", said),
            EmbedError::Malformed {
                model: "m".into(),
                reason: without.into()
            }
        );
        assert_eq!(
            EmbedError::unusable("m", ModelFault::Setup, said),
            EmbedError::Unusable {
                model: "m".into(),
                reason: without.into(),
                fault: ModelFault::Setup
            }
        );
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
