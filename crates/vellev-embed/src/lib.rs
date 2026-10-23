//! Turning text into vectors, with the model chosen per collection.

mod bedrock;
mod embedder;
mod http;
#[cfg(feature = "local")]
mod local;
mod registry;

pub use bedrock::{Bedrock, UNREACHED_FOR};
pub use embedder::{Embedder, DOCUMENTS_WITHIN, FAULT_STANDS, LONGEST_QUERY_WITHIN, QUERY_WITHIN};
pub use http::Http;
#[cfg(feature = "local")]
pub use local::Local;
pub use registry::{
    Embedders, Settings, DEFAULT_LOCAL_MODEL, LOCAL_MODELS, LOCAL_MODEL_IDLE, SHORTEST_KEY,
};

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

/// Whether `run` has a letter and a digit in it: a word of prose, however
/// long and hyphenated, has no digit.
fn mixed(run: &str) -> bool {
    run.chars().any(|c| c.is_ascii_digit()) && run.chars().any(|c| c.is_ascii_alphabetic())
}

/// Whether what begins `sk-` is a key by its length: forty characters or
/// more after that, or twenty with a letter and a digit among them, or
/// twenty after `sk-proj-` or `sk-ant-`.
fn openai_or_anthropic(key: &str) -> bool {
    let rest = &key["sk-".len()..];
    let named = ["proj-", "ant-"]
        .iter()
        .filter_map(|name| rest.strip_prefix(name))
        .any(|rest| rest.len() >= 20);
    named || rest.len() >= 40 || mixed(rest)
}

/// Whether what begins `hf_` is a token by its length: thirty letters and
/// digits or more after that, or twenty with some of each.
fn hugging_face(key: &str) -> bool {
    let rest = &key["hf_".len()..];
    rest.len() >= 30 || mixed(rest)
}

/// Whether what follows `Bearer` or `Basic` is a credential and not words
/// said after the scheme's name: lower-case words joined by `-`, `_` or
/// `.` are the only thing that is not.
fn credential(key: &str) -> bool {
    !key.chars()
        .all(|c| c.is_ascii_lowercase() || "-_.".contains(c))
}

/// The shapes keys are known by. They are looked for anywhere in what is
/// said, as it is said and with nothing decoded: a key comes in a path, in
/// markup, before a full stop, and straight after an escape such as `\n`
/// or `\u0026` in a body quoted as JSON, as readily as alone.
static KEYS: LazyLock<Vec<Shape>> = LazyLock::new(|| {
    // What an OpenAI, Anthropic or Google key is made of, and a Hugging
    // Face token.
    const KEY: &str = "[A-Za-z0-9_-]";
    const TOKEN: &str = "[A-Za-z0-9]";
    // How the middle of a key is masked by whoever quotes it back; and how
    // its end is, which three full stops are not: those end a sentence.
    const MASK: &str = r"(?:\u{2026}|\.{3}|\*+)";
    const HIDDEN: &str = r"(?:\u{2026}|\*+)";
    // The escapes a key, or a name, comes straight after in a body quoted
    // as JSON or in a URL: their last letter is not one of a word.
    const ESCAPE: &str = r"\\[nrt]|\\u[0-9A-Fa-f]{4}|%[0-9A-Fa-f]{2}";
    // What a key known by how it begins may come straight after: anything
    // but a letter, which makes of it the end of a word, as `sk-` is of
    // `task-` and `hf_` of `shf_`.
    let begins = format!("(?P<kept>^|[^A-Za-z]|{ESCAPE})");
    // What a name in a query string, or a scheme, may come straight after:
    // anything that is not part of a name.
    let named = format!("(?:^|[^A-Za-z0-9_]|{ESCAPE})");
    let any: fn(&str) -> bool = |_| true;
    let never: fn(char) -> bool = |_| false;
    [
        // A JSON web token: a header and a body, each the base64url of
        // what begins `{"`, and a signature if it has one.
        (
            format!(r"{begins}(?P<key>eyJ{KEY}{{4,}}\.eyJ{KEY}{{4,}}(?:\.{KEY}+)?)"),
            any,
            never,
        ),
        // What is sent as a bearer or basic credential.
        (
            format!(r"(?i)(?P<kept>{named}(?:bearer|basic)\s+)(?P<key>[A-Za-z0-9._~+/=-]{{16,}})"),
            credential,
            never,
        ),
        // A key or token given in a query string: all of its value, up to
        // the next part, a space, a quote or the end.
        (
            format!(
                r#"(?i)(?P<kept>{named}(?:api[_-]?key|access_token|token|key)=)(?P<key>[^&\s"']+)"#
            ),
            any,
            never,
        ),
        // OpenAI's and Anthropic's keys: masked, with something of the key
        // beside the mask, or at their length.
        (
            format!(r"{begins}(?P<key>sk-(?:{KEY}*{MASK}{KEY}{{2,}}|{KEY}{{2,}}{MASK}))"),
            any,
            never,
        ),
        (
            format!(r"{begins}(?P<key>sk-{KEY}{{20,}})"),
            openai_or_anthropic,
            never,
        ),
        // Hugging Face's tokens, likewise.
        (
            format!(r"{begins}(?P<key>hf_(?:{TOKEN}*{MASK}{TOKEN}{{2,}}|{TOKEN}{{2,}}{HIDDEN}))"),
            any,
            never,
        ),
        (
            format!(r"{begins}(?P<key>hf_{TOKEN}{{20,}})"),
            hugging_face,
            never,
        ),
        // A Google API key. A hyphen or an underscore after it joins it to
        // something else; a letter or a digit makes of it something longer.
        (format!(r"{begins}(?P<key>AIza{KEY}{{35}})"), any, |c| {
            c.is_ascii_alphanumeric()
        }),
        // An AWS access key id.
        (
            format!(r"{begins}(?P<key>(?:AKIA|ASIA)[A-Z0-9]{{16}})"),
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
/// taken out, wherever in it the key lies.
///
/// A key known by how it begins is one anywhere but straight after a
/// letter, unless the letter ends one of the escapes `\n`, `\r`, `\t`,
/// `\uXXXX` and `%XX`. Such keys are:
///
/// - a JSON web token: `eyJ`, a full stop and `eyJ` again, with more
///   after each, and with or without a signature;
/// - a key beginning `sk-`: with forty characters or more after that; or
///   twenty or more, a letter and a digit among them; or twenty or more
///   after `sk-proj-` or `sk-ant-`; or with some of it masked by `…`,
///   `...` or asterisks and two characters or more of it left;
/// - a Hugging Face token, `hf_` and thirty letters and digits or more,
///   or twenty or more with some of each; or masked likewise, though
///   three full stops at its end do not make it one;
/// - a Google API key, `AIza` and thirty-five characters, unless a letter
///   or a digit comes straight after;
/// - an AWS access key id, `AKIA` or `ASIA` and sixteen capitals and
///   digits, unless a capital or a digit comes straight after.
///
/// And two are known by what comes before them, which may itself come
/// after anything but a letter, a digit or an underscore, or after one of
/// those escapes:
///
/// - what follows `Bearer` or `Basic`, of sixteen characters or more,
///   unless it is only lower-case words joined by `-`, `_` or `.`;
/// - the whole value of `api_key`, `api-key`, `apikey`, `key`, `token` or
///   `access_token` in a query string, whatever it is, up to the next
///   `&`, space or quote.
///
/// Nothing else is taken out, and deliberately:
///
/// - a key straight after a letter, `task-…` and `shf_…` being words and
///   no provider gluing a key to one;
/// - a key with no shape of its own, such as an Azure key of thirty-two
///   hex digits or an AWS secret of forty characters, which nothing tells
///   from a hash or an id;
/// - a credential after `Bearer` or `Basic` that is all lower-case
///   letters, which nothing tells from words;
/// - anything that would have to be decoded to be seen.
///
/// A model's name is left as it is. What is not a key and is taken for
/// one, such as a long run of words after `sk-` with a digit in it,
/// costs a detail of what went wrong and no more.
pub(crate) fn without_keys(said: &str) -> String {
    looked_through(said.len());
    KEYS.iter()
        .fold(said.to_string(), |said, shape| without(&said, shape))
}

/// `said` without what is of `shape`. What is found and is not a key
/// after all hides nothing that begins inside it: looking goes on from
/// its next character.
fn without(said: &str, shape: &Shape) -> String {
    let mut without = String::with_capacity(said.len());
    let (mut copied, mut from) = (0, 0);
    loop {
        let found = shape.found.captures_at(said, from);
        // A search reads on to the end of what it finds, or of `said`.
        let read_to = found.as_ref().and_then(|found| found.get(0));
        looked_through(read_to.map_or(said.len(), |whole| whole.end()) - from);
        let Some(found) = found else {
            break;
        };
        let (Some(whole), Some(key)) = (found.get(0), found.name("key")) else {
            break;
        };
        let goes_on = said[key.end()..].chars().next().is_some_and(shape.goes_on);
        if !goes_on && (shape.is_key)(key.as_str()) {
            without.push_str(&said[copied..key.start()]);
            without.push_str(REDACTED);
            copied = key.end();
            from = key.end();
        } else {
            let first = said[whole.start()..].chars().next();
            from = whole.start() + first.map_or(1, char::len_utf8);
        }
    }
    without.push_str(&said[copied..]);
    without
}

/// Counts, in tests and on this thread, `bytes` of what was said as
/// looked through for keys: so a test can hold how much of it is looked
/// through to a number, which no timing does on a loaded machine.
#[cfg(test)]
pub(crate) fn looked_through(bytes: usize) {
    LOOKED_THROUGH_HERE.with(|looked| looked.set(looked.get() + bytes));
}

#[cfg(not(test))]
#[inline(always)]
pub(crate) fn looked_through(_bytes: usize) {}

#[cfg(test)]
thread_local! {
    /// How many bytes [`looked_through`] has counted on this thread.
    pub(crate) static LOOKED_THROUGH_HERE: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Whether `c` is a character a key can be made of, whichever of the
/// [`KEYS`] it is shaped as, the value in a query string apart: a letter
/// or a digit, one of `_-.~+/=`, or what masks one, `*` and `…`.
///
/// So where what is said is cut short for looking through, whatever of a
/// key was left at the cut, too short to be told for one, lies wholly in
/// the run of these that ends there. The value in a query string is
/// taken out whole however little of it there is, cut short or not.
pub(crate) fn of_a_key(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_-.~+/=*\u{2026}".contains(c)
}

/// What texts are embedded for. Someone is waiting on a query, so where a
/// model has a queue, queries go ahead of documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    Document,
    Query,
}

/// What a model that many wait on at once has lately done, for anyone.
/// A write that waits its turn behind others is not given up on while
/// the model is at work, for whomever: only when it has not been, for as
/// long as a model may go without.
///
/// Only a vector made is the model at work. A load is not, but the time
/// a load takes that ends with the model loaded is not held against
/// those who wait: nor that of a load under way, for as long as a load
/// is given. The time of a load that fails, of a run that fails or
/// panics, of anything but loading that makes no vector, is held against
/// them, as it was spent: a load that fails is taken back from the time
/// it was given while it was under way.
#[derive(Debug, Default)]
pub struct Activity(std::sync::Mutex<Acted>);

/// The most loads that ended well, since the last vector, kept to be
/// set against the time a waiter has waited. One more lets the oldest
/// go, which is then held against waiters: they are given up on sooner,
/// never later.
const LOADS_KEPT: usize = 256;

#[derive(Debug, Default)]
struct Acted {
    /// When a vector was last made.
    made: Option<tokio::time::Instant>,
    /// The loads that ended well since then: when each began and ended.
    loaded: std::collections::VecDeque<(tokio::time::Instant, tokio::time::Instant)>,
    /// A load under way: when it began, and when it is no longer given.
    loading: Option<(tokio::time::Instant, tokio::time::Instant)>,
}

impl Activity {
    /// The model has made a vector, for whomever.
    pub fn made(&self) {
        let mut acted = self.0.lock().expect("activity lock");
        acted.made = Some(tokio::time::Instant::now());
        acted.loaded.clear();
    }

    /// The model has begun to load, and is given `at_most` to.
    pub fn loading(&self, at_most: std::time::Duration) {
        let now = tokio::time::Instant::now();
        let until = now.checked_add(at_most).unwrap_or(now + LONGEST);
        self.0.lock().expect("activity lock").loading = Some((now, until));
    }

    /// The load under way has ended with the model loaded: the time it
    /// took is not held against those who wait. It made no vector.
    pub fn loaded(&self) {
        let mut acted = self.0.lock().expect("activity lock");
        if let Some((began, _)) = acted.loading.take() {
            if acted.loaded.len() == LOADS_KEPT {
                acted.loaded.pop_front();
            }
            acted.loaded.push_back((began, tokio::time::Instant::now()));
        }
    }

    /// No load is under way, whether or not one was: one that was ended
    /// without the model loaded, and its time is held against those
    /// who wait.
    pub fn not_loading(&self) {
        self.0.lock().expect("activity lock").loading = None;
    }

    /// How long one who has waited on the model since `since` has waited
    /// with the model not at work: the time since the later of then and
    /// the model's last vector, less what loads have taken that ended
    /// with the model loaded, and what a load under way has taken, up to
    /// what a load is given.
    pub(crate) fn quiet(&self, since: tokio::time::Instant) -> std::time::Duration {
        let acted = self.0.lock().expect("activity lock");
        let now = tokio::time::Instant::now();
        let from = acted.made.map_or(since, |made| made.max(since));
        let taken = |began: tokio::time::Instant, ended: tokio::time::Instant| {
            ended.saturating_duration_since(began.max(from))
        };
        let loaded = acted
            .loaded
            .iter()
            .map(|(began, ended)| taken(*began, *ended));
        let loading = acted
            .loading
            .map(|(began, until)| taken(began, now.min(until)));
        let excused: std::time::Duration = loaded.chain(loading).sum();
        now.saturating_duration_since(from).saturating_sub(excused)
    }

    /// When the load under way is no longer given, if one is.
    #[cfg(test)]
    pub(crate) fn loading_until(&self) -> Option<tokio::time::Instant> {
        self.0
            .lock()
            .expect("activity lock")
            .loading
            .map(|(_, until)| until)
    }
}

/// A time that can be added to any instant a node will see.
const LONGEST: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// A model that turns texts into vectors.
#[async_trait]
pub trait Provider: std::fmt::Debug + Send + Sync {
    /// One vector for each text, in order. No text is blank.
    async fn embed(&self, texts: Vec<String>, purpose: Purpose) -> Result<Vec<Vec<f32>>>;

    /// What the model has lately done for anyone, if it says: one that
    /// many wait on at once, each in turn, says, so that a write waiting
    /// its turn is given up on only when the model has stopped.
    fn activity(&self) -> Option<&Activity> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A load under way is set against the time a waiter has waited only
    /// for what of it comes after the waiter did, and only for as long
    /// as a load is given: one that came after the load was given up on
    /// waits with nothing set against it.
    #[tokio::test(start_paused = true)]
    async fn a_load_under_way_is_set_against_a_wait_only_after_it_and_within_its_time() {
        let secs = std::time::Duration::from_secs;
        let activity = Activity::default();
        let began = tokio::time::Instant::now();
        activity.loading(secs(600));
        // One that came at the start: ten minutes set against its wait.
        tokio::time::advance(secs(700)).await;
        let came = tokio::time::Instant::now();
        assert_eq!(activity.quiet(began), secs(100));
        // One that came at 700 s, past the ten minutes: nothing set
        // against its hundred.
        tokio::time::advance(secs(100)).await;
        assert_eq!(activity.quiet(came), secs(100));
        // One that came at 300 s: three hundred set against its five.
        assert_eq!(activity.quiet(began + secs(300)), secs(500) - secs(300));
    }

    /// A load that ended well is set against the time a waiter has
    /// waited only for what of it came after the waiter did, and after
    /// the model's last vector: what came before is no part of its wait.
    #[tokio::test(start_paused = true)]
    async fn a_load_is_set_against_a_wait_only_for_what_of_it_came_after() {
        let secs = std::time::Duration::from_secs;
        let activity = Activity::default();
        let began = tokio::time::Instant::now();
        // A load of fifty seconds, from nothing.
        activity.loading(secs(600));
        tokio::time::advance(secs(50)).await;
        activity.loaded();
        // A waiter comes ten seconds after it ended, and has waited forty.
        tokio::time::advance(secs(10)).await;
        let came = tokio::time::Instant::now();
        tokio::time::advance(secs(40)).await;
        assert_eq!(activity.quiet(came), secs(40));
        // One that came twenty seconds into the load: thirty of it are
        // set against its wait of eighty.
        assert_eq!(activity.quiet(began + secs(20)), secs(80) - secs(30));
        // A vector made: the load is no part of anyone's wait from then.
        activity.made();
        tokio::time::advance(secs(5)).await;
        assert_eq!(activity.quiet(began), secs(5));
    }

    #[test]
    fn what_is_shaped_like_a_key_is_taken_out_wherever_it_lies() {
        const SK: &str = "sk-proj-abcdefghijklmnopqrstuvwxyz012345";
        const ANT: &str = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz";
        // Hugging Face's tokens as they are: thirty-four letters.
        const HF: &str = "hf_AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh";
        const AWS: &str = "AKIAIOSFODNN7EXAMPLE";
        const GOOGLE: &str = "AIzaSyA-0123456789abcdefghijklmnopqrstu";
        const JWT: &str = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl";
        assert_eq!((HF.len(), GOOGLE.len()), (3 + 34, 4 + 35));
        let r = REDACTED;
        let cases = [
            // Alone, and as servers quote them back.
            (
                format!("Incorrect API key provided: {SK}."),
                format!("Incorrect API key provided: {r}."),
            ),
            (format!("key {ANT} refused"), format!("key {r} refused")),
            (
                format!("Invalid token {HF}."),
                format!("Invalid token {r}."),
            ),
            (
                "Invalid token hf_AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGhIj.".into(),
                format!("Invalid token {r}."),
            ),
            (
                format!("The security token {AWS} is not valid"),
                format!("The security token {r} is not valid"),
            ),
            (
                format!("API key not valid: {GOOGLE}."),
                format!("API key not valid: {r}."),
            ),
            (format!("token {JWT} expired"), format!("token {r} expired")),
            // By their length, whatever they are made of.
            (
                "key sk-abcdefghijklmnopqrstuvwxyzabcdefghijklmn refused".into(),
                format!("key {r} refused"),
            ),
            (
                "key sk-abcdefghij0123456789 refused".into(),
                format!("key {r} refused"),
            ),
            (
                "key sk-proj-abcdefghijklmnopqrst refused".into(),
                format!("key {r} refused"),
            ),
            (
                "key sk-ant-abcdefghijklmnopqrst refused".into(),
                format!("key {r} refused"),
            ),
            (
                "token hf_abcdefghijklmnopqrstuvwxyzabcd refused".into(),
                format!("token {r} refused"),
            ),
            (
                "token hf_abcdefghij0123456789 refused".into(),
                format!("token {r} refused"),
            ),
            (
                "id ASIA0123456789ABCDEF, expired".into(),
                format!("id {r}, expired"),
            ),
            (
                "unsigned eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0In0 token".into(),
                format!("unsigned {r} token"),
            ),
            // At the least each can be: one character fewer is below.
            (
                format!("key sk-{} refused", "a".repeat(40)),
                format!("key {r} refused"),
            ),
            (
                "Bearer abcdefghijklmnoP refused".into(),
                format!("Bearer {r} refused"),
            ),
            // Masked.
            (
                "Incorrect API key provided: sk-\u{2026}abcd".into(),
                format!("Incorrect API key provided: {r}"),
            ),
            (
                "key sk-ab...wxyz refused".into(),
                format!("key {r} refused"),
            ),
            ("key sk-abcd... refused".into(), format!("key {r} refused")),
            (
                "key 'sk-****abcd' refused".into(),
                format!("key '{r}' refused"),
            ),
            ("key sk-ab*Wq9Z refused".into(), format!("key {r} refused")),
            (
                "token hf_ab\u{2026}wxyz is not valid".into(),
                format!("token {r} is not valid"),
            ),
            (
                "token hf_ab...wxyz is not valid".into(),
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
                "token hf_abcd\u{2026} is not valid".into(),
                format!("token {r} is not valid"),
            ),
            (
                "token hf_abcd**** is not valid".into(),
                format!("token {r} is not valid"),
            ),
            // Straight after anything but a letter: in a path, in markup,
            // after a full stop, a quote, an underscore, a digit, other
            // writing.
            (
                format!("GET /v1/keys/{SK} failed"),
                format!("GET /v1/keys/{r} failed"),
            ),
            (format!("<{SK}>"), format!("<{r}>")),
            (format!("auth={HF};"), format!("auth={r};")),
            (
                format!(r#"{{"key":"{SK}"}}"#),
                format!(r#"{{"key":"{r}"}}"#),
            ),
            (
                format!("key.{SK} is not one"),
                format!("key.{r} is not one"),
            ),
            (format!("prefix_{SK}"), format!("prefix_{r}")),
            (format!("7{HF}"), format!("7{r}")),
            (format!("7{JWT}"), format!("7{r}")),
            (format!("_{GOOGLE}"), format!("_{r}")),
            (format!("{AWS}."), format!("{r}.")),
            (
                format!("Credential={AWS}/20261008/eu-west-2/bedrock/aws4_request"),
                format!("Credential={r}/20261008/eu-west-2/bedrock/aws4_request"),
            ),
            (
                format!("<AWSAccessKeyId>{AWS}</AWSAccessKeyId>"),
                format!("<AWSAccessKeyId>{r}</AWSAccessKeyId>"),
            ),
            (format!("密钥{SK}无效"), format!("密钥{r}无效")),
            (format!("密钥{AWS}无效"), format!("密钥{r}无效")),
            (format!("密钥{HF}无效"), format!("密钥{r}无效")),
            // Straight after an escape, whose last letter is not a word's:
            // in a body quoted as JSON, with its backslashes as they were
            // written, or in a URL.
            (
                format!(r"Traceback:\n  File x\n{SK}"),
                format!(r"Traceback:\n  File x\n{r}"),
            ),
            (
                format!(r"line one\n{HF}\nline three"),
                format!(r"line one\n{r}\nline three"),
            ),
            (format!(r"\t{AWS}\r\n"), format!(r"\t{r}\r\n")),
            (format!(r"\r{GOOGLE} "), format!(r"\r{r} ")),
            (format!(r"\u000a{SK}"), format!(r"\u000a{r}")),
            (format!(r"\u003C{JWT}"), format!(r"\u003C{r}")),
            (format!("api_key%3D{SK}"), format!("api_key%3D{r}")),
            (format!("next%3d{HF}"), format!("next%3d{r}")),
            (format!("path%2F{AWS}"), format!("path%2F{r}")),
            // A Google key that something is joined to.
            (format!("{GOOGLE}-x"), format!("{r}-x")),
            (format!("{GOOGLE}_x"), format!("{r}_x")),
            // What is found and is not a key hides none that begins in it.
            (format!("AKIA1{AWS}"), format!("AKIA1{r}")),
            (format!("task-{SK}"), format!("task-{r}")),
            (format!("AIza_{GOOGLE}"), format!("AIza_{r}")),
            // A key that ends where another begins.
            (format!("AKIAIOSFODNN7EXAMPL7{SK}"), format!("{r}{r}")),
            (format!("{GOOGLE}_{AWS}"), format!("{r}_{r}")),
            // Sent as a credential.
            (
                "Authorization: Bearer abcdefghij.klmnop_qrst7 was refused".into(),
                format!("Authorization: Bearer {r} was refused"),
            ),
            (
                "Bearer ABCDEFGHIJKLMNOPQRST refused".into(),
                format!("Bearer {r} refused"),
            ),
            (
                "Bearer abcdefghijklmnopqrsT refused".into(),
                format!("Bearer {r} refused"),
            ),
            (
                "Bearer Authentication-Is-Required-For-This-Endpoint".into(),
                format!("Bearer {r}"),
            ),
            (
                "authorization: basic dXNlcjpwYXNzd29yZA== refused".into(),
                format!("authorization: basic {r} refused"),
            ),
            (
                r"refused:\nBearer abcdefghijklmnop1234".into(),
                format!(r"refused:\nBearer {r}"),
            ),
            (format!("Bearer {JWT}"), format!("Bearer {r}")),
            // Given in a query string: the whole value, whatever is in it.
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
            (
                r"token=abc\/def+SECRETTAIL&x=1".into(),
                format!("token={r}&x=1"),
            ),
            (
                r#"{"url":"https://e.test?apikey=abc<def>ghi","n":1}"#.into(),
                format!(r#"{{"url":"https://e.test?apikey={r}","n":1}}"#),
            ),
            (
                format!(r"https://e.test/v1?alt=json\u0026key={GOOGLE}"),
                format!(r"https://e.test/v1?alt=json\u0026key={r}"),
            ),
            (
                r"?alt=json\u0026key=abc123 was refused".into(),
                format!(r"?alt=json\u0026key={r} was refused"),
            ),
            // Nothing is decoded: an escaped `&` does not end a value.
            (
                r"?alt=json\u0026key=abc123\u0026x=1".into(),
                format!(r"?alt=json\u0026key={r}"),
            ),
            (
                r"failed:\ntoken=abc123".into(),
                format!(r"failed:\ntoken={r}"),
            ),
            (
                "redirect%26key=abc123&next=1".into(),
                format!("redirect%26key={r}&next=1"),
            ),
        ];
        for (said, kept) in cases {
            assert_eq!(without_keys(&said), kept, "{said}");
        }
        // Not keys: left exactly as they were said.
        let glued = [
            format!("was{SK}"),
            format!("x{HF}"),
            format!("x{AWS}"),
            format!("x{GOOGLE}"),
            format!("x{JWT}"),
            format!(r"\\x{SK} and %g{HF} and \u00g{AWS}"),
            // One key id straight after the start of another: the second
            // comes after a letter, as the end of a word does.
            format!("ASIA{AWS}"),
            // One character short of the least each can be.
            format!("key sk-{} refused", "a".repeat(39)),
        ];
        let said = [
            // Straight after a letter, it is the end of a word.
            "task-3f2504e0-4f89-11d3-9a0c-0305e82c3301 failed",
            "risk-assessment-2026-10-08-final",
            "mask-rcnn-resnet50-fpn-v2",
            "disk-0123456789abcdef0123456789abcdef.img is full",
            "desk-0123456789abcdefghijklmnop and kiosk-0123456789abcdefghijklmnop",
            "pod task-runner-5f6d8c9b7d-x2k4q in flask-app-7d9f8b6c5d-abcde restarted",
            "pod vellev-embed-7d9f8b6c5d-x2k4q is not ready",
            "arn:aws:bedrock:eu-west-2:123456789012:foundation-model/amazon.titan-embed-text-v2:0",
            "arn:aws:iam::123456789012:role/task-execution-role-20261008-eu-west-2",
            "arn:aws:ecs:eu-west-2:123456789012:task/vellev/0123456789abcdef0123456789abcdef",
            "s3://vellev-backups/disk-2026-10-08/part-00000-3f2504e04f8911d39a0c0305e82c3301.parquet",
            "s3://bucket/task-0123456789abcdef0123456789/mask-0123456789abcdef0123456789",
            "shf_abcdefghijklmnopqrstuvwxyzABCDEFGH is a name",
            "keyJsonWebTokenParser.eyJsonWebTokenParser failed",
            "a risk-assessment of ask-me-anything-about-embeddings-today",
            "logs in disk-*.log and task-* queues, risk-...",
            // Too short, or of too little, to be one: each by a character.
            "key sk-abcdefghij012345678 refused",
            "key sk-proj-abcdefghijklmnopqrs refused",
            "key sk-ant-abcdefghijklmnopqrs refused",
            "token hf_abcdefghijklmnopqrstuvwxyzabc refused",
            "token hf_abcdefghij012345678 refused",
            "Bearer abcdefghijklmnO refused",
            "AIzaSyA-0123456789abcdefghijklmnopqrst is too short to be one",
            "model sk-embed-v2 not found",
            "model text-embedding-3-small not found",
            "sk-learn-style-estimators are not keys",
            "BAAI/bge-base-en-v1.5 via hf_hub is not sk- shaped",
            "hf_hub_download and hf_transfer are how models are fetched",
            "hf_transfer... and hf_hub",
            "set hf_transferabcdefghijklmnopq to use it",
            "amazon.titan-embed-text-v2:0 in AKIAshort or a task-list",
            "AKIAIOSFODNN7EXAMPLEANDMORE is too long to be one",
            "AIza is how a Google key begins",
            "AIzaSyA-0123456789abcdefghijklmnopqrstuvwxyz is too long to be one",
            "eyJ is how a token begins, a.b.c is not one",
            "eyJhbGciOiJIUzI1NiJ9.c2lnbmF0dXJl is not one either",
            "a hash 0123456789abcdef0123456789abcdef has no shape of its own",
            // Words after a scheme's name, and names that end as a key's does.
            "Missing bearer token",
            r#"{"token_type":"bearer","expires_in":3600}"#,
            "Bearer realm=\"api\"",
            "use the Bearer scheme, or Basic if you must",
            "Bearer authorization-header-missing",
            "Bearer authentication_is_required_for_this_endpoint",
            "Basic credentials.were.not.given",
            "max_token=5 and monkey=banana",
            "<redacted>",
            "",
        ];
        for said in said.into_iter().chain(glued.iter().map(String::as_str)) {
            assert_eq!(without_keys(said), said);
        }
    }

    /// What is dropped where looking for keys ends is the run of these
    /// characters that ends there: a shape whose keys had some other
    /// character in them would leave the start of one behind.
    #[test]
    fn a_key_of_any_shape_is_made_only_of_the_characters_a_key_is_made_of() {
        let keys = [
            "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz_-",
            "sk-ab\u{2026}wxyz",
            "sk-ab...wxyz",
            "sk-****abcd",
            "hf_AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh",
            "hf_abcd\u{2026}",
            "hf_ab***wxyz",
            "AKIAIOSFODNN7EXAMPLE",
            "AIzaSyA-0123456789abcdefghijklmnopqrstu",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl",
            // Every character a credential after `Bearer` or `Basic` has.
            "dXNlcjpwYXNzd29yZA==",
            "ABCdef012._~+/=-ABCdef012",
        ];
        for key in keys {
            // Each is a key, all of it, said after `Bearer` as the last
            // two are only then.
            let said = format!("Bearer {key}");
            assert_eq!(without_keys(&said), format!("Bearer {REDACTED}"), "{key}");
            assert!(key.chars().all(of_a_key), "{key}");
        }
        for not in [' ', '\n', '"', '\'', '&', '?', '<', ':', ',', '\\', '密'] {
            assert!(!of_a_key(not), "{not:?}");
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
