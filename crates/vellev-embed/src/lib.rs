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

/// A shape a key is known by. What `found` finds is a key if what it
/// names `key` passes `is_key` and is not followed by a character that
/// `goes_on` it into something longer; whatever it names `kept`, before
/// the key, is left.
struct Shape {
    found: Regex,
    is_key: fn(&str) -> bool,
    goes_on: fn(char) -> bool,
}

/// The shapes keys are known by. They are looked for anywhere in what is
/// said, as it is said and with nothing decoded: a key comes in a path, in
/// markup, before a full stop, and straight after an escape such as `\n`
/// or `\u0026` in a body quoted as JSON, as readily as alone. So no shape
/// asks for anything in particular to come before it: what a key begins
/// with, and how long it is, is what tells it.
static KEYS: LazyLock<Vec<Shape>> = LazyLock::new(|| {
    // What an OpenAI, Anthropic or Google key is made of, and a Hugging
    // Face token.
    const KEY: &str = "[A-Za-z0-9_-]";
    const TOKEN: &str = "[A-Za-z0-9]";
    // How the middle of a key is masked by whoever quotes it back.
    const MASK: &str = r"(?:\u{2026}|\.{3}|\*+)";
    // What a name in a query string, or a scheme, may come straight after:
    // nothing, anything that is not part of a name, or an escape.
    const BEFORE: &str = r"(?:^|[^A-Za-z0-9_]|\\u[0-9A-Fa-f]{4}|\\[nrt]|%[0-9A-Fa-f]{2})";
    let any: fn(&str) -> bool = |_| true;
    let never: fn(char) -> bool = |_| false;
    let key_char: fn(char) -> bool = |c| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    // A run of letters and digits both: a word of prose, however long and
    // hyphenated, has no digit in it.
    let mixed: fn(&str) -> bool = |key| {
        key.chars().any(|c| c.is_ascii_digit()) && key.chars().any(|c| c.is_ascii_alphabetic())
    };
    // A credential and not a word said after the scheme's name: it has a
    // digit, or is long and is not lower-case words joined together.
    let credential: fn(&str) -> bool = |key| {
        let words = |c: char| c.is_ascii_lowercase() || "-_.".contains(c);
        key.chars().any(|c| c.is_ascii_digit()) || (key.len() >= 24 && !key.chars().all(words))
    };
    [
        // A JSON web token: three runs of base64url, the first of a header.
        (
            format!(r"(?P<key>eyJ{KEY}{{4,}}\.{KEY}{{4,}}\.{KEY}{{4,}})"),
            any,
            never,
        ),
        // What is sent as a bearer or basic credential.
        (
            format!(r"(?i)(?P<kept>{BEFORE}(?:bearer|basic)\s+)(?P<key>[A-Za-z0-9._~+/=-]{{16,}})"),
            credential,
            never,
        ),
        // A key or token given in a query string, up to the next part.
        (
            format!(
                r#"(?i)(?P<kept>{BEFORE}(?:api[_-]?key|access_token|token|key)=)(?P<key>[^&\s"'<>\\]+)"#
            ),
            any,
            never,
        ),
        // OpenAI's and Anthropic's keys, `sk-`, `sk-proj-` and `sk-ant-`
        // among them: masked, with something of the key to either side of
        // the mask, or at their length.
        (
            format!(r"(?P<key>sk-(?:{KEY}{{2,}}{MASK}{KEY}*|{KEY}*{MASK}{KEY}{{2,}}))"),
            any,
            never,
        ),
        (format!(r"(?P<key>sk-{KEY}{{20,}})"), mixed, never),
        // Hugging Face's tokens, likewise.
        (
            format!(r"(?P<key>hf_(?:{TOKEN}{{2,}}{MASK}{TOKEN}*|{TOKEN}*{MASK}{TOKEN}{{2,}}))"),
            any,
            never,
        ),
        (format!(r"(?P<key>hf_{TOKEN}{{20,}})"), mixed, never),
        // A Google API key.
        (format!(r"(?P<key>AIza{KEY}{{35}})"), any, key_char),
        // An AWS access key id.
        (
            r"(?P<key>(?:AKIA|ASIA)[A-Z0-9]{16})".to_string(),
            any,
            |c| c.is_ascii_uppercase() || c.is_ascii_digit(),
        ),
    ]
    .into_iter()
    .map(|(found, is_key, goes_on)| Shape {
        found: Regex::new(&found).expect("a key's shape"),
        is_key,
        goes_on,
    })
    .collect()
});

/// `said`, by whatever serves a model, with what is shaped like a key
/// taken out, wherever in it the key lies:
///
/// - a JSON web token;
/// - a credential of sixteen characters or more after `Bearer` or `Basic`,
///   if it has a digit in it, or is of twenty-four or more and not
///   lower-case words joined by `-`, `_` or `.`;
/// - the value of `api_key`, `api-key`, `key`, `token` or `access_token`
///   in a query string, whatever the value is;
/// - a key beginning `sk-`: of twenty characters or more after that, with
///   a letter and a digit among them, or with its middle masked by `…`,
///   `...` or asterisks and two characters or more of it left on a side;
/// - a Hugging Face token, `hf_` and twenty letters and digits or more,
///   some of each, or masked likewise;
/// - a Google API key, `AIza` and thirty-five characters;
/// - an AWS access key id, `AKIA` or `ASIA` and sixteen capitals and
///   digits.
///
/// Nothing else is: not a key with no shape of its own, such as an Azure
/// key of thirty-two hex digits or an AWS secret of forty characters,
/// which nothing tells from a hash or an id, and taking which for keys
/// would take much else. Nor is anything decoded to look inside it. A
/// model's name is left as it is. What is not a key and is taken for one,
/// such as a word of twenty characters with a digit in it that happens to
/// follow `sk-`, costs a detail of what went wrong and no more.
pub(crate) fn without_keys(said: &str) -> String {
    KEYS.iter().fold(said.to_string(), |said, shape| {
        let mut without = String::with_capacity(said.len());
        let mut copied = 0;
        for found in shape.found.captures_iter(&said) {
            let (Some(whole), Some(key)) = (found.get(0), found.name("key")) else {
                continue;
            };
            let goes_on = said[key.end()..].chars().next().is_some_and(shape.goes_on);
            if goes_on || !(shape.is_key)(key.as_str()) {
                continue;
            }
            without.push_str(&said[copied..key.start()]);
            without.push_str(REDACTED);
            copied = whole.end();
        }
        without.push_str(&said[copied..]);
        without
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
        const GOOGLE: &str = "AIzaSyA-0123456789abcdefghijklmnopqrstu";
        const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl";
        assert_eq!(GOOGLE.len(), 4 + 35);
        let r = REDACTED;
        let cases = [
            // Alone, and as servers quote them back.
            (
                format!("Incorrect API key provided: {SK}."),
                format!("Incorrect API key provided: {r}."),
            ),
            (format!("key {ANT} refused"), format!("key {r} refused")),
            (
                "Incorrect API key provided: sk-\u{2026}abcd".into(),
                format!("Incorrect API key provided: {r}"),
            ),
            (
                "key sk-ab...wxyz refused".into(),
                format!("key {r} refused"),
            ),
            (
                "key 'sk-****abcd' refused".into(),
                format!("key '{r}' refused"),
            ),
            ("key sk-ab*Wq9Z refused".into(), format!("key {r} refused")),
            // In a path, in markup, after a full stop.
            (
                format!("GET /v1/keys/{SK} failed"),
                format!("GET /v1/keys/{r} failed"),
            ),
            (format!("<{SK}>"), format!("<{r}>")),
            (
                format!("key.{SK} is not one"),
                format!("key.{r} is not one"),
            ),
            (
                format!("token {HF} is not valid"),
                format!("token {r} is not valid"),
            ),
            (format!("token {HF}."), format!("token {r}.")),
            (
                "token hf_ab\u{2026}wxyz is not valid".into(),
                format!("token {r} is not valid"),
            ),
            (
                "token hf_ab***wxyz is not valid".into(),
                format!("token {r} is not valid"),
            ),
            (
                "token hf_ab*wxyz is not valid".into(),
                format!("token {r} is not valid"),
            ),
            (
                format!("The security token {AWS} is not valid"),
                format!("The security token {r} is not valid"),
            ),
            (format!("{AWS}."), format!("{r}.")),
            (
                format!("Credential={AWS}/20261008/eu-west-2/bedrock/aws4_request"),
                format!("Credential={r}/20261008/eu-west-2/bedrock/aws4_request"),
            ),
            (
                format!("<AWSAccessKeyId>{AWS}</AWSAccessKeyId>"),
                format!("<AWSAccessKeyId>{r}</AWSAccessKeyId>"),
            ),
            (
                "id ASIA0123456789ABCDEF, expired".into(),
                format!("id {r}, expired"),
            ),
            (
                format!("API key not valid: {GOOGLE}."),
                format!("API key not valid: {r}."),
            ),
            // Straight after an escape, a word, or anything else: in a body
            // quoted as JSON, with its backslashes as they were written.
            (
                format!(r"Traceback:\n  File x\n{SK}"),
                format!(r"Traceback:\n  File x\n{r}"),
            ),
            (
                format!(r"line one\n{HF}\nline three"),
                format!(r"line one\n{r}\nline three"),
            ),
            (format!(r"\t{AWS}\r\n"), format!(r"\t{r}\r\n")),
            (
                format!(r"https://e.test/v1?alt=json\u0026key={GOOGLE}"),
                format!(r"https://e.test/v1?alt=json\u0026key={r}"),
            ),
            (
                r"?alt=json\u0026key=abc123\u0026x=1".into(),
                format!(r"?alt=json\u0026key={r}\u0026x=1"),
            ),
            (
                r"failed:\ntoken=abc123".into(),
                format!(r"failed:\ntoken={r}"),
            ),
            (format!("api_key%3D{SK}"), format!("api_key%3D{r}")),
            (
                "redirect%26key=abc123&next=1".into(),
                format!("redirect%26key={r}&next=1"),
            ),
            (format!("prefix_{SK}"), format!("prefix_{r}")),
            (format!("was{SK}"), format!("was{r}")),
            (format!("7{HF}"), format!("7{r}")),
            (format!("x{AWS}"), format!("x{r}")),
            (format!("x{GOOGLE}"), format!("x{r}")),
            (format!("x{JWT}"), format!("x{r}")),
            (format!("密钥{SK}无效"), format!("密钥{r}无效")),
            (format!("密钥{AWS}无效"), format!("密钥{r}无效")),
            // Sent as a credential, or in a query string.
            (
                "Authorization: Bearer abcdefghij.klmnop_qrst7 was refused".into(),
                format!("Authorization: Bearer {r} was refused"),
            ),
            (
                "Bearer ABCDEFGHIJKLMNOPQRSTUVWXYZ refused".into(),
                format!("Bearer {r} refused"),
            ),
            (
                "authorization: basic dXNlcjpwYXNzd29yZA== refused".into(),
                format!("authorization: basic {r} refused"),
            ),
            (
                r"refused:\nBearer abcdefghijklmnop1234".into(),
                format!(r"refused:\nBearer {r}"),
            ),
            (format!("token {JWT} expired"), format!("token {r} expired")),
            (format!("Bearer {JWT}"), format!("Bearer {r}")),
            (
                "GET /embed?api_key=abc123&model=m&access_token=xyz failed".into(),
                format!("GET /embed?api_key={r}&model=m&access_token={r} failed"),
            ),
            (
                "https://e.test/v1?key=abc123 refused".into(),
                format!("https://e.test/v1?key={r} refused"),
            ),
            (
                "?api-key=abc&token=def".into(),
                format!("?api-key={r}&token={r}"),
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
            "Bearer authorization-header-missing",
            "Bearer authentication_is_required_for_this_endpoint",
            "Basic credentials.were.not.given",
            "model sk-embed-v2 not found",
            "model text-embedding-3-small not found",
            "BAAI/bge-base-en-v1.5 via hf_hub is not sk- shaped",
            "amazon.titan-embed-text-v2:0 in AKIAshort or a task-list",
            "AKIAIOSFODNN7EXAMPLEANDMORE is too long to be one",
            "a risk-assessment of ask-me-anything-about-embeddings-today",
            "logs in disk-*.log and task-* queues, risk-...",
            "hf_hub_download and hf_transfer are how models are fetched",
            "max_token=5 and monkey=banana",
            "eyJ is how a token begins, a.b.c is not one",
            "AIza is how a Google key begins",
            "AIzaSyA-0123456789abcdefghijklmnopqrstuvwxyz is too long to be one",
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
