use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{CONTENT_LENGTH, RETRY_AFTER, TRANSFER_ENCODING};
use serde::Deserialize;
use serde_json::json;
use vellev_core::{HttpApi, ModelFault};

use crate::{Activity, EmbedError, Provider, Purpose, Result};

/// Texts per request.
const BATCH: usize = 32;
const ATTEMPTS: u32 = 5;
const FIRST_BACKOFF: Duration = Duration::from_millis(500);
/// The longest an endpoint can make its caller wait before the next try.
const LONGEST_BACKOFF: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one batch of a call to embed may take, every try included,
/// when nothing is said of how it gets on: as for a query.
const DEADLINE: Duration = Duration::from_secs(120);
/// How much of an endpoint's refusal is passed on.
const REFUSAL_CHARS: usize = 300;
/// How many bytes of what an endpoint says are looked through for keys:
/// what is passed on, at four bytes a character at most, and after it
/// more than the length of any key that could begin there and be told
/// for one when cut short.
const LOOKED_THROUGH: usize = 4 * 1024;
/// How many bytes of a refusal are read at all, and so the longest one
/// that is read for the fault it names: many times what any endpoint
/// takes to name one.
const REFUSAL_READ: usize = 64 * 1024;

/// A text-embeddings-inference server or an OpenAI-compatible endpoint.
/// Requests that cannot connect, time out, are throttled, or fail on the
/// server's side are tried again; embedding the same text twice does no
/// harm.
pub struct Http {
    client: reqwest::Client,
    url: String,
    api: HttpApi,
    model: String,
    key: Option<String>,
    /// When the endpoint last answered a batch with its vectors.
    activity: Activity,
    /// How long one batch may take, every try included.
    batch_within: Duration,
    /// What the waits between tries and a batch's bound are timed by.
    clock: std::sync::Arc<dyn crate::clock::Clock>,
}

impl std::fmt::Debug for Http {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Http")
            .field("url", &self.url)
            .field("api", &self.api)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

#[derive(Deserialize)]
struct OpenaiResponse {
    data: Vec<OpenaiEmbedding>,
}

/// A refusal as OpenAI words it: `{"error": {"code": "..."}}`.
#[derive(Deserialize)]
struct OpenaiRefusal {
    error: OpenaiError,
}

#[derive(Deserialize)]
struct OpenaiError {
    code: String,
}

/// The fault a refusal names, in OpenAI's own code for it, if asking again
/// cannot help until someone puts it right: it has no such model, or the
/// key is not one of its keys. An account with nothing left to spend is
/// not one: paying puts that right, with no change to the node, and until
/// then it is throttling like any other. A status alone never names one
/// either: whatever stands in front of an endpoint, such as a firewall's
/// rate rule or a proxy with nothing behind it for a moment, answers 401,
/// 403 or 404 too, and other servers word their refusals in prose, which
/// is not read.
fn fault(api: HttpApi, refusal: &str) -> Option<ModelFault> {
    let refusal = serde_json::from_str::<OpenaiRefusal>(refusal).ok()?;
    match (api, refusal.error.code.as_str()) {
        (HttpApi::Openai, "model_not_found") => Some(ModelFault::Provider),
        (HttpApi::Openai, "invalid_api_key") => Some(ModelFault::Setup),
        _ => None,
    }
}

/// How many characters of a key, at the least, are the start of one: a
/// run shorter than this is no more a key than any word is.
const BEGUN: usize = 4;
/// How many characters of the endpoint's own key in a row, at the least,
/// are taken out wherever they are in what it said: fewer are as likely
/// to be anything else.
const PIECE: usize = 8;

/// What an endpoint said, with its escapes undone: each character, and
/// the bytes of what was said that it stands for.
struct Decoded {
    chars: Vec<char>,
    from: Vec<std::ops::Range<usize>>,
    /// Where an escape begins that what was said ends before the end of,
    /// so that what it stands for cannot be told.
    unfinished: Option<usize>,
}

/// What an escape at the start of some text comes to.
enum Escape {
    /// It stands for this character, and is so many bytes.
    Is(char, usize),
    /// The text ends before it can be told what it stands for.
    Unfinished,
    /// It is no escape.
    Not,
}

/// The value of `digits`, if it is all hex digits, in either case.
fn hex(digits: Option<&str>) -> Option<u32> {
    let digits = digits?;
    digits
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit())
        .then(|| u32::from_str_radix(digits, 16).ok())
        .flatten()
}

/// Whether `text` is all hex digits, and so may be the start of more.
fn begun_hex(text: &str) -> bool {
    text.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// What an escape at the start of `text` comes to: JSON's `\uXXXX`,
/// surrogate pairs included, `\/` and `\\`; and percent-encoding, a
/// character's bytes as `%XX` each, in either case of hex.
fn escape(text: &str) -> Escape {
    let bytes = text.as_bytes();
    match bytes.first() {
        Some(b'\\') => match bytes.get(1) {
            None => Escape::Unfinished,
            Some(b'/') => Escape::Is('/', 2),
            Some(b'\\') => Escape::Is('\\', 2),
            Some(b'u') => {
                let Some(high) = hex(text.get(2..6)) else {
                    let rest = text.get(2..).unwrap_or_default();
                    return match rest.len() < 4 && begun_hex(rest) {
                        true => Escape::Unfinished,
                        false => Escape::Not,
                    };
                };
                if !(0xD800..0xDC00).contains(&high) {
                    return char::from_u32(high).map_or(Escape::Not, |c| Escape::Is(c, 6));
                }
                // A surrogate, of which the other half is to follow.
                let rest = text.get(6..).unwrap_or_default();
                match hex(rest.get(2..6)) {
                    Some(low) if rest.starts_with("\\u") && (0xDC00..0xE000).contains(&low) => {
                        let pair = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                        char::from_u32(pair).map_or(Escape::Not, |c| Escape::Is(c, 12))
                    }
                    _ => {
                        let begun = "\\u".starts_with(rest)
                            || (rest.starts_with("\\u") && rest.len() < 6 && begun_hex(&rest[2..]));
                        match begun {
                            true => Escape::Unfinished,
                            false => Escape::Not,
                        }
                    }
                }
            }
            Some(_) => Escape::Not,
        },
        Some(b'%') => {
            // The bytes of one character, each as `%XX`.
            let mut read = Vec::new();
            let mut at = 0;
            loop {
                let rest = &text[at..];
                match hex(rest.get(1..3)) {
                    Some(byte) if rest.starts_with('%') => {
                        read.push(byte as u8);
                        at += 3;
                    }
                    _ => {
                        // No whole `%XX` here: the text ends partway
                        // through one, or something else comes, short of
                        // a whole character.
                        let partial =
                            rest.starts_with('%') && rest.len() < 3 && begun_hex(&rest[1..]);
                        return match rest.is_empty() || partial {
                            true => Escape::Unfinished,
                            false => Escape::Not,
                        };
                    }
                }
                match std::str::from_utf8(&read) {
                    Ok(decoded) => {
                        let c = decoded.chars().next().expect("a byte read");
                        return Escape::Is(c, at);
                    }
                    Err(error) if error.error_len().is_some() => return Escape::Not,
                    Err(_) => {}
                }
            }
        }
        _ => Escape::Not,
    }
}

/// `text` with its escapes undone.
fn decoded(text: &str) -> Decoded {
    let mut decoded = Decoded {
        chars: Vec::new(),
        from: Vec::new(),
        unfinished: None,
    };
    let mut at = 0;
    while let Some(next) = text[at..].chars().next() {
        let (c, len) = match escape(&text[at..]) {
            Escape::Is(c, len) => (c, len),
            Escape::Unfinished => {
                // The first: what follows is part of it.
                decoded.unfinished.get_or_insert(at);
                (next, next.len_utf8())
            }
            Escape::Not => (next, next.len_utf8()),
        };
        decoded.chars.push(c);
        decoded.from.push(at..at + len);
        at += len;
    }
    decoded
}

/// What is wrong with `key`, to be sent `endpoint`, if it is too short to
/// be one or has space in it, which no key an endpoint gives out does: it
/// would be found in every other word of what the endpoint says, and
/// taken out of all of them.
pub(crate) fn too_short(endpoint: &str, key: &str) -> Option<String> {
    let shortest = crate::registry::SHORTEST_KEY;
    (key.chars().count() < shortest || key.chars().any(char::is_whitespace)).then(|| {
        format!(
            "the key for {endpoint} is shorter than {shortest} characters or has space in \
             it, which no key an endpoint gives out is: leave the key out if the endpoint \
             needs none"
        )
    })
}

/// What stands for a key taken out.
const REDACTED: &str = "<redacted>";

/// The bytes of what was said that stand for runs of [`PIECE`] or more
/// characters of `key`, or all of it if it is shorter, among the
/// characters of `view`: each run as long as it goes, so that runs that
/// overlap are one. Each character of `view` is set against each of the
/// key once, as the end of a run of how long the run before it was and
/// one: so the cost is that of the text's length times the key's,
/// whatever both are made of.
fn runs_of(key: &[char], view: &Decoded) -> Vec<std::ops::Range<usize>> {
    let piece = PIECE.min(key.len());
    // How long a run of the key ends at the character before, at each
    // character of the key; and then at this one.
    let mut before = vec![0usize; key.len() + 1];
    let mut here = vec![0usize; key.len() + 1];
    // How far back from each character the longest run ending there
    // goes.
    let mut back = vec![0usize; view.chars.len()];
    for (at, c) in view.chars.iter().enumerate() {
        for (of, k) in key.iter().enumerate() {
            here[of + 1] = match alike(*c, *k) {
                true => before[of] + 1,
                false => 0,
            };
        }
        back[at] = here.iter().copied().max().unwrap_or_default();
        std::mem::swap(&mut before, &mut here);
    }
    // Every character within a long enough run ending later is covered.
    let mut spans: Vec<std::ops::Range<usize>> = Vec::new();
    let mut reach = 0;
    for at in (0..view.chars.len()).rev() {
        if back[at] >= piece {
            reach = reach.max(back[at]);
        }
        if reach > 0 {
            let from = view.from[at].clone();
            match spans.last_mut() {
                Some(span) if span.start == from.end => span.start = from.start,
                _ => spans.push(from),
            }
            reach -= 1;
        }
    }
    spans.reverse();
    spans
}

/// Whether `a` and `b` are the same, but for the case of ASCII letters.
fn alike(a: char, b: char) -> bool {
    a.eq_ignore_ascii_case(&b)
}

/// What an endpoint said in refusing, as far as it was read.
struct Said {
    /// The start of it: no more than [`REFUSAL_READ`], and no part of a
    /// character.
    text: String,
    /// Whether it came to an end, and was not cut off by an error or by
    /// how much is read. What did may be read for a fault, if it reads
    /// as all of something.
    ended: bool,
    /// Whether it is known to be all the endpoint meant to say: it came
    /// to an end, and the endpoint had said how much it would send. One
    /// that only stops sending has said nothing of whether it was done:
    /// hanging up partway through a key looks the same as having said
    /// all it had to.
    whole: bool,
}

/// Whether `refusing` says how much it will send, so that its end is
/// known for its end and not for a connection lost, as HTTP/1.1 frames
/// what is sent: in chunks, when chunked is the last of its codings,
/// named exactly so in any case; or by its length, when it has no
/// coding, a length beside one being of no account. What has a coding
/// that is not chunked last, or neither, is sent until the endpoint
/// hangs up, as all of HTTP/1.0 is unless it gives a length. Every
/// later version of HTTP frames what it sends, and says when a stream
/// has ended, which is all that is taken here: whether it did is what
/// reading it to its end finds.
fn says_how_much(refusing: &reqwest::Response) -> bool {
    let headers = refusing.headers();
    let codings: Vec<String> = headers
        .get_all(TRANSFER_ENCODING)
        .iter()
        .flat_map(|codings| codings.to_str().unwrap_or("?").split(','))
        .map(|coding| coding.trim().to_ascii_lowercase())
        .filter(|coding| !coding.is_empty())
        .collect();
    let length = headers.contains_key(CONTENT_LENGTH);
    match refusing.version() {
        reqwest::Version::HTTP_09 | reqwest::Version::HTTP_10 => length,
        reqwest::Version::HTTP_11 => match codings.last() {
            Some(last) => last == "chunked",
            None => length,
        },
        _ => true,
    }
}

/// `read` as text, without whatever part of a character it ends with:
/// bytes cut off partway through one are no character, and read as one
/// they would stand between what comes before them and its being seen
/// for the start of a key.
fn heard(mut read: Vec<u8>) -> String {
    // The last byte that begins a character, among the last three, and
    // how many bytes that character is to have.
    let begun = read
        .iter()
        .rev()
        .take(3)
        .position(|byte| byte & 0xC0 != 0x80);
    if let Some(back) = begun {
        let at = read.len() - 1 - back;
        let needs = match read[at] {
            0xF0.. => 4,
            0xE0.. => 3,
            0xC0.. => 2,
            _ => 1,
        };
        if needs > back + 1 {
            read.truncate(at);
        }
    }
    String::from_utf8_lossy(&read).into_owned()
}

/// The start of what `refusing` says in refusing: no more than
/// [`REFUSAL_READ`] is read, however much is sent. So an endpoint that
/// says a great deal costs neither the memory to hold it nor the time to
/// read it for a fault.
async fn said(mut refusing: reqwest::Response) -> Said {
    let says_how_much = says_how_much(&refusing);
    let mut read = Vec::new();
    // It has ended once the endpoint has no more to say: which is not so
    // of what was cut off on its way, or goes on past what is read.
    let mut ended = false;
    while !ended && read.len() <= REFUSAL_READ {
        match refusing.chunk().await {
            Ok(Some(more)) => read.extend_from_slice(&more),
            Ok(None) => ended = true,
            Err(_) => break,
        }
    }
    read.truncate(REFUSAL_READ);
    Said {
        text: heard(read),
        ended,
        whole: ended && says_how_much,
    }
}

#[derive(Deserialize)]
struct OpenaiEmbedding {
    index: usize,
    embedding: Vec<f32>,
}

impl Http {
    /// `key`, unless blank, is sent as a bearer token. An OpenAI-compatible
    /// `url` may be given with or without its `/v1`.
    pub fn new(url: &str, api: HttpApi, model: &str, key: Option<String>) -> Result<Self> {
        Self::through(Self::client(model)?, url, api, model, key)
    }

    /// What endpoints are reached through. Making one reads the
    /// certificates endpoints are trusted by, which is not quick: a node
    /// makes one, and reaches every endpoint through it.
    pub(crate) fn client(model: &str) -> Result<reqwest::Client> {
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(ATTEMPT_TIMEOUT)
            .build()
            .map_err(|error| EmbedError::failed(model, format!("no HTTP client: {error}")))
    }

    /// As [`Self::new`], reached through `client`.
    pub(crate) fn through(
        client: reqwest::Client,
        url: &str,
        api: HttpApi,
        model: &str,
        key: Option<String>,
    ) -> Result<Self> {
        let url = url.trim_end_matches('/');
        let url = match api {
            HttpApi::Tei => url,
            HttpApi::Openai => url.strip_suffix("/v1").unwrap_or(url),
        };
        let key = key.filter(|key| !key.trim().is_empty());
        if let Some(wrong) = key.as_deref().and_then(|key| too_short(url, key)) {
            return Err(EmbedError::unusable(model, ModelFault::Setup, wrong));
        }
        Ok(Self {
            client,
            url: url.to_string(),
            api,
            model: model.to_string(),
            key,
            activity: Activity::default(),
            batch_within: DEADLINE,
            clock: crate::clock::tokio(),
        })
    }

    /// This, giving a batch `within` to be answered in, every try
    /// included, rather than two minutes. It is for tests of that, which
    /// cannot wait two minutes: a node gives every endpoint the same,
    /// and has no setting for it.
    #[doc(hidden)]
    #[must_use]
    pub fn answering_each_batch_within(self, within: Duration) -> Self {
        Self {
            batch_within: within,
            ..self
        }
    }

    /// This, timing the waits between tries and each batch's bound by
    /// `clock` rather than tokio's: for tests that move the clock
    /// themselves.
    #[doc(hidden)]
    #[must_use]
    pub fn timed_by(self, clock: std::sync::Arc<dyn crate::clock::Clock>) -> Self {
        Self { clock, ..self }
    }

    /// `text` without this endpoint's key, whatever its shape. Anything
    /// else shaped like a key is taken out where the failure is made.
    fn redacted(&self, text: String) -> String {
        match &self.key {
            Some(key) => text.replace(key.as_str(), REDACTED),
            None => text,
        }
    }

    /// The start of what an endpoint `said`, to be passed on: without
    /// keys, and no longer than [`REFUSAL_CHARS`]. Keys are taken out
    /// before it is cut to that, since a key cut short is no longer
    /// shaped like one, and what was left of it would be passed on. But
    /// they are looked for only in [`LOOKED_THROUGH`] of it, so that an
    /// endpoint that says a great deal does not cost a great deal; and
    /// that is itself a cut, so only as much of it is looked through as
    /// [`Self::seen_whole`].
    fn opening(&self, said: &str, whole: bool) -> String {
        let seen = &said[..self.seen_whole(said, whole)];
        let without = crate::without_keys(&self.without_own_key(seen));
        without.chars().take(REFUSAL_CHARS).collect()
    }

    /// `said` without any run of [`PIECE`] or more characters of this
    /// endpoint's own key, or all of it if it is shorter: wherever it is,
    /// in whatever case of ASCII letters, and however its characters were
    /// escaped, in JSON or in percent-encoding. The runs are found among
    /// the characters escapes stand for, and what stands for them is
    /// taken out, in one pass.
    fn without_own_key(&self, said: &str) -> String {
        let Some(key) = self.key.as_deref() else {
            return said.to_string();
        };
        let key: Vec<char> = key.chars().collect();
        // Found among the characters escapes stand for, and among those
        // that were said: a key with what reads as an escape in it, as
        // `%41`, is found as it is too.
        let view = decoded(said);
        let as_said = Decoded {
            chars: said.chars().collect(),
            from: said
                .char_indices()
                .map(|(at, c)| at..at + c.len_utf8())
                .collect(),
            unfinished: None,
        };
        let mut spans: Vec<std::ops::Range<usize>> = [&view, &as_said]
            .into_iter()
            .flat_map(|view| runs_of(&key, view))
            .collect();
        spans.sort_by_key(|span| span.start);
        let mut without = String::with_capacity(said.len());
        let mut copied = 0;
        for span in spans {
            if span.end <= copied {
                continue;
            }
            if span.start > copied {
                without.push_str(&said[copied..span.start]);
                without.push_str(REDACTED);
            } else if !without.ends_with(REDACTED) {
                without.push_str(REDACTED);
            }
            copied = span.end;
        }
        without.push_str(&said[copied..]);
        without
    }

    /// How much of the start of `said` is looked through for keys: up to
    /// [`LOOKED_THROUGH`] of it, less whatever at the end of that may be
    /// the start of a key that goes on past it. A key cut there is not
    /// shaped like one, and where a long key before it is taken out, what
    /// was left of it would be drawn into what is passed on.
    ///
    /// That is whatever there is the start of this endpoint's own key,
    /// and then the run of characters a key is made of, if the run goes
    /// on. What is said in fewer bytes than that is all looked through,
    /// if it is known to be the `whole` of what the endpoint said: what
    /// was cut off on its way, or may have been, may end partway through
    /// a key too, and is taken to. And the start of this endpoint's own
    /// key is never passed on, whole or not, if it is as much as is the
    /// start of a key.
    fn seen_whole(&self, said: &str, whole: bool) -> usize {
        let mut end = (0..=said.len().min(LOOKED_THROUGH))
            .rev()
            .find(|&at| said.is_char_boundary(at))
            .unwrap_or_default();
        // Whether what is seen may stop partway through something.
        let cut = end < said.len() || !whole;
        // An escape cut off is no character, as a character's bytes cut
        // off are none: it would stand between what comes before it and
        // its being seen for the start of a key.
        if cut {
            end = decoded(&said[..end]).unfinished.unwrap_or(end);
        }
        if let Some(key) = self.key.as_deref() {
            // What is seen, with its escapes undone, ends with how much of
            // the start of this endpoint's own key: not all of it, which
            // is taken out wherever it is.
            let view = decoded(&said[..end]);
            let key: Vec<char> = key.chars().collect();
            let ends_with = |start: &[char]| {
                start.len() <= view.chars.len()
                    && view.chars[view.chars.len() - start.len()..]
                        .iter()
                        .zip(start)
                        .all(|(c, k)| alike(*c, *k))
            };
            if ends_with(&key) {
                return end;
            }
            let begun = (1..key.len()).rev().find(|of| ends_with(&key[..*of]));
            // Where what is seen is all that was said, a few characters
            // the key begins with are what any word may end with, and
            // stay.
            if let Some(begun) = begun.filter(|begun| cut || *begun >= BEGUN) {
                end = view.from[view.chars.len() - begun].start;
            }
        }
        if !cut {
            return end;
        }
        // Whether the run of a key's characters there goes on: which is
        // not known of what was cut off, and is taken to be so.
        let goes_on = match said[end..].chars().next() {
            Some(next) => crate::of_a_key(next),
            None => !whole,
        };
        if goes_on {
            end = said[..end].trim_end_matches(crate::of_a_key).len();
        }
        end
    }

    fn failed(&self, reason: impl std::fmt::Display) -> EmbedError {
        EmbedError::failed(&self.model, self.redacted(reason.to_string()))
    }

    fn unusable(&self, fault: ModelFault, reason: impl std::fmt::Display) -> EmbedError {
        EmbedError::unusable(&self.model, fault, self.redacted(reason.to_string()))
    }

    fn malformed(&self, reason: String) -> EmbedError {
        EmbedError::malformed(&self.model, self.redacted(reason))
    }

    /// One request, sent until it is answered or refused for good.
    async fn send(&self, texts: &[String]) -> Result<reqwest::Response> {
        let (path, body) = match self.api {
            HttpApi::Tei => ("embed", json!({ "inputs": texts, "truncate": true })),
            HttpApi::Openai => (
                "v1/embeddings",
                json!({ "model": self.model, "input": texts }),
            ),
        };
        let mut backoff = FIRST_BACKOFF;
        let mut attempt = 1;
        loop {
            let request = self.client.post(format!("{}/{path}", self.url)).json(&body);
            let request = match &self.key {
                Some(key) => request.bearer_auth(key),
                None => request,
            };
            let (failure, passing, wait) = match request.send().await {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status();
                    let asked = response
                        .headers()
                        .get(RETRY_AFTER)
                        .and_then(|value| value.to_str().ok()?.parse().ok())
                        .map(Duration::from_secs);
                    let refused = said(response).await;
                    // A refusal cut short is not read for a fault: what
                    // is left of it is not what the endpoint said.
                    let read = || fault(self.api, &refused.text);
                    let lasting = refused.ended.then(read).flatten();
                    let refusal = self.opening(&refused.text, refused.whole);
                    let passing =
                        lasting.is_none() && (status.as_u16() == 429 || status.is_server_error());
                    let wait = asked.unwrap_or(backoff).min(LONGEST_BACKOFF);
                    let said = format!("{status}: {refusal}");
                    let failure = match lasting {
                        Some(fault) => self.unusable(fault, said),
                        None => self.failed(said),
                    };
                    (failure, passing, wait)
                }
                Err(error) => {
                    let passing = error.is_connect() || error.is_timeout();
                    (self.failed(error.without_url()), passing, backoff)
                }
            };
            if !passing || attempt == ATTEMPTS {
                return Err(failure);
            }
            self.clock.sleep(wait).await;
            backoff *= 2;
            attempt += 1;
        }
    }

    async fn batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self.send(texts).await?;
        let unreadable = |error: reqwest::Error| {
            self.malformed(format!("an unreadable answer: {}", error.without_url()))
        };
        let vectors: Vec<Vec<f32>> = match self.api {
            HttpApi::Tei => response.json().await.map_err(unreadable)?,
            HttpApi::Openai => {
                let mut answer: OpenaiResponse = response.json().await.map_err(unreadable)?;
                answer
                    .data
                    .sort_unstable_by_key(|embedding| embedding.index);
                let numbered = answer.data.iter().map(|embedding| embedding.index);
                if !numbered.eq(0..answer.data.len()) {
                    return Err(self.malformed("vectors numbered other than in turn".into()));
                }
                answer.data.into_iter().map(|e| e.embedding).collect()
            }
        };
        if vectors.len() != texts.len() {
            return Err(self.malformed(format!(
                "{} vectors for {} texts",
                vectors.len(),
                texts.len()
            )));
        }
        Ok(vectors)
    }

    /// The vectors of all the texts, a batch at a time. Each batch is
    /// given two minutes, every try included, unless told otherwise. A
    /// batch answered with its vectors is the endpoint at work, and
    /// nothing else is: not a refusal, nor being told to wait and ask
    /// again.
    async fn batches(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let within = self.batch_within;
            let late = || self.failed(format!("no answer within {within:?}"));
            // As a timeout is: the batch is looked at first.
            let made = tokio::select! {
                biased;
                made = self.batch(batch) => made,
                () = self.clock.sleep(within) => Err(late()),
            };
            vectors.extend(made?);
            self.activity.made();
        }
        Ok(vectors)
    }
}

#[async_trait]
impl Provider for Http {
    /// An endpoint keeps no line of this node's making, so the texts are
    /// sent at once, a batch at a time. A call of many batches is not
    /// given up on for how many there are: only when the endpoint has
    /// answered no batch, of this call or any other, for as long as a
    /// model may go without.
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        self.batches(&texts).await
    }

    fn activity(&self) -> Option<&Activity> {
        Some(&self.activity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long key early in what is said is taken out, which draws what
    /// comes later, up to where looking ends, into what is passed on: and
    /// there a key may lie cut short, too short to be told for one.
    /// A refusal cut off on its way, short of where looking for keys
    /// ends, may end partway through a key: what is left of the key is
    /// not shaped like one, and is not passed on. The same words, when
    /// they are all the endpoint said, are passed on whole.
    #[test]
    fn a_refusal_cut_off_partway_through_a_key_passes_on_nothing_of_the_key() {
        let own = "its-own-key-of-no-shape-0123456789";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        // Cut off partway through the endpoint's own key.
        let cut = format!("no such key: {}", &own[..20]);
        assert_eq!(http.opening(&cut, false), "no such key: ");
        // Cut off partway through a key of a shape that is known whole.
        let cut = "invalid key sk-proj-AbCd1234Ef";
        assert_eq!(http.opening(cut, false), "invalid key ");
        // Cut off after a word, which may be a key begun: it goes too.
        assert_eq!(http.opening("401 Unauthorized", false), "401 ");
        // Cut off after a space: nothing was begun.
        assert_eq!(
            http.opening("401 Unauthorized ", false),
            "401 Unauthorized "
        );
        // All of what was said: nothing is begun that does not end.
        assert_eq!(http.opening("401 Unauthorized", true), "401 Unauthorized");
        let whole = format!("no such key: {own}");
        assert_eq!(http.opening(&whole, false), "no such key: <redacted>");
        assert_eq!(http.opening(&whole, true), "no such key: <redacted>");
        // A key of no shape at all, with what is not of a key in it,
        // whole where what was said was cut off: it is taken out whole,
        // and not cut short at its last run of a key's characters.
        let shapeless = "its_own_key,of;no:shape1";
        let http = Http::new(
            "http://e.test",
            HttpApi::Openai,
            "m",
            Some(shapeless.into()),
        )
        .expect("an endpoint");
        let cut = format!("no such key: {shapeless}");
        assert_eq!(http.opening(&cut, false), "no such key: <redacted>");
    }

    /// What is passed on of a failure has no copy of the endpoint's own
    /// key in it, however many, wherever it came from.
    #[test]
    fn a_failure_carries_no_copy_of_the_endpoints_own_key() {
        let own = "its-own-key-of-no-shape-0123456789";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        let said = format!("{own} and again {own}.");
        let made = |error: EmbedError| error.to_string();
        let expected = "<redacted> and again <redacted>.";
        assert!(made(http.failed(&said)).ends_with(expected));
        assert!(made(http.unusable(ModelFault::Setup, &said)).ends_with(expected));
        assert!(made(http.malformed(said.clone())).ends_with(expected));
        assert_eq!(http.opening(&said, true), expected);
    }

    /// Any run of eight characters or more of the endpoint's own key is
    /// taken out wherever it is, and one of seven stays: at every offset
    /// in the key, in prose, in JSON and in a URL.
    #[test]
    fn eight_characters_of_the_own_key_in_a_row_are_taken_out_and_seven_are_not() {
        assert_eq!(PIECE, 8);
        let own = "vk_live_9f3a2b7c1d4e5f60aa77bb88";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        let key: Vec<char> = own.chars().collect();
        let mut cells = 0;
        for length in [7, 8, 9, 16, key.len()] {
            for at in 0..=key.len() - length {
                let piece: String = key[at..at + length].iter().collect();
                for (before, after) in [("key ", " is bad"), ("{\"k\":\"", "\"}"), ("?k=", "&x=1")]
                {
                    let said = format!("{before}{piece}{after}");
                    let passed = http.opening(&said, true);
                    match length < PIECE {
                        true => assert_eq!(passed, said, "{piece}"),
                        false => {
                            assert_eq!(passed, format!("{before}<redacted>{after}"), "{piece}")
                        }
                    }
                    cells += 1;
                }
            }
        }
        assert_eq!(cells, 3 * (26 + 25 + 24 + 17 + 1));
    }

    /// The own key's runs are found however its characters were escaped,
    /// in JSON or in percent-encoding, in either case of hex, mixed, and
    /// across an escape; and in whatever case of ASCII letters.
    #[test]
    fn the_own_key_is_found_however_it_was_escaped_and_whatever_its_case() {
        let own = "ab+cd/ef==gh1234567890XYZ";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        for said in [
            "bad key ab+cd\\/ef==gh1234567890XYZ!",
            "bad key ab+cd\\u002Fef==gh1234567890XYZ!",
            "bad key ab+cd\\u002fef==gh1234567890XYZ!",
            "bad key ab%2Bcd%2Fef%3D%3Dgh1234567890XYZ!",
            "bad key ab%2bcd%2fef%3d%3dgh1234567890XYZ!",
            "bad key ab+cd%2F\\u0065f==gh1234567890XYZ!",
            "bad key \\u0061\\u0062\\u002B\\u0063d/ef==gh1234567890XYZ!",
            "bad key AB+CD/EF==GH1234567890xyz!",
            "bad key Ab+Cd\\/eF==gH1234567890xYz!",
        ] {
            assert_eq!(http.opening(said, true), "bad key <redacted>!", "{said}");
        }
        // Seven characters across an escape stay; eight do not.
        assert_eq!(http.opening("x cd\\/ef== y", true), "x cd\\/ef== y");
        assert_eq!(http.opening("x cd\\/ef==g y", true), "x <redacted> y");
        // Escaped twice over is not undone, and passes.
        let twice = "bad key ab%252Bcd%252Fef%253D%253Dgh1234567890XYZ";
        assert_eq!(
            http.opening(twice, true),
            "bad key ab%252Bcd%252Fef%253D%253D<redacted>"
        );
    }

    /// A key with a character outside the basic plane, which JSON writes
    /// as a surrogate pair, is found so.
    #[test]
    fn the_own_key_is_found_in_a_surrogate_pair() {
        let own = "key-\u{1f511}-0123456789abcdef";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        for said in [
            "bad key-\\uD83D\\uDD11-0123 here",
            "bad key-\\ud83d\\udd11-0123 here",
            "bad key-%F0%9F%94%91-0123 here",
            "bad key-\u{1f511}-0123 here",
        ] {
            assert_eq!(http.opening(said, true), "bad <redacted> here", "{said}");
        }
        // Half a pair is no character, and nothing is taken out for it.
        let half = "bad key-\\uD83D-0123 here";
        assert_eq!(http.opening(half, true), half);
    }

    /// Cut off partway through an escape, what is passed on has neither
    /// the escape nor the start of the key before it.
    #[test]
    fn an_escape_cut_off_is_left_out_with_the_start_of_the_key_before_it() {
        let own = "ab+cd/ef==gh1234567890XYZ";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        for said in [
            "{\"error\":\"bad key ab+cd\\",
            "{\"error\":\"bad key ab+cd\\u",
            "{\"error\":\"bad key ab+cd\\u00",
            "{\"error\":\"bad key ab+cd\\u002",
            "bad key ab%2",
            "bad key ab%",
            "bad key ab%2Bcd%2",
        ] {
            let passed = http.opening(said, false);
            assert!(passed.ends_with("bad key "), "{said}: {passed}");
        }
        // A pair cut off between its halves.
        let own = "key-\u{1f511}-0123456789abcdef";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        for said in [
            "bad key-\\uD83D",
            "bad key-\\uD83D\\",
            "bad key-\\uD83D\\uDD",
        ] {
            assert_eq!(http.opening(said, false), "bad ", "{said}");
        }
    }

    /// The start of the endpoint's own key at the end of all that it
    /// said is left out from four characters on, and three stay.
    #[test]
    fn the_start_of_the_own_key_is_left_out_from_four_characters() {
        assert_eq!(BEGUN, 4);
        let own = "its-own-key-of-no-shape-0123456789";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        assert_eq!(http.opening("bad key: its-", true), "bad key: ");
        assert_eq!(http.opening("bad key: its", true), "bad key: its");
    }

    /// A batch is given two minutes, every try included, by a node: which
    /// has no setting for it.
    #[test]
    fn a_batch_is_given_two_minutes() {
        let http = Http::new("http://e.test", HttpApi::Tei, "m", None).expect("an endpoint");
        assert_eq!(http.batch_within, Duration::from_secs(120));
        // Timed by tokio's clock, as everything is but tests that hand in
        // their own.
        assert_eq!(format!("{:?}", http.clock), "Tokio");
        let told = http.answering_each_batch_within(Duration::from_secs(3));
        assert_eq!(told.batch_within, Duration::from_secs(3));
    }

    /// Bytes that stop partway through a character are heard without
    /// what there is of it, whatever its length, and others as they are.
    #[test]
    fn what_is_cut_off_partway_through_a_character_is_heard_without_it() {
        for whole in ["key", "cl\u{e9}", "sk-A\u{2026}", "\u{5bc6}", "a\u{1f511}"] {
            let bytes = whole.as_bytes();
            assert_eq!(heard(bytes.to_vec()), whole);
            let last = whole.chars().last().unwrap();
            let before = &whole[..whole.len() - last.len_utf8()];
            for cut in 1..last.len_utf8() {
                let cut_off = bytes[..before.len() + cut].to_vec();
                assert_eq!(heard(cut_off), before, "{whole:?} less {cut}");
            }
        }
        assert_eq!(heard(Vec::new()), "");
        // What is no part of any character is not taken for one cut off.
        assert_eq!(
            heard(vec![b'a', 0x80, 0x80, 0x80]),
            "a\u{fffd}\u{fffd}\u{fffd}"
        );
    }

    /// The table of `tests/redaction.rs`, where how what was said was
    /// framed does not matter: only whether it is known to be whole. For
    /// each shape of key, cut off at every byte of it, or with looking
    /// for keys ending at every byte of it, no more of the key is passed
    /// on than less than the start of one.
    #[test]
    fn nothing_passed_on_carries_the_start_of_a_key_wherever_it_is_cut() {
        let keys = [
            (true, "", "its-own-key-of-no-shape-0123456789"),
            (
                true,
                "",
                "cl\u{e9}-secr\u{e8}te-\u{e9}t\u{e9}-0123456789-abcdef",
            ),
            (true, "", "its_own_key,of;no:shape@at!all1"),
            (false, "", "sk-AbCd1234EfGh5678IjKl9012MnOp"),
            (false, "", "sk-proj-AbCd1234EfGh5678IjKl9012MnOp"),
            (false, "", "sk-AbCdEf9\u{2026}XyZ4"),
            (false, "", "hf_AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"),
            (
                false,
                "",
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl",
            ),
            (false, "", "AKIAIOSFODNN7EXAMPLE"),
            (false, "", "ASIAIOSFODNN7EXAMPLE"),
            (false, "", "AIzaSyA-0123456789abcdefghijklmnopqrstu"),
            (false, "Bearer ", "abcdefghij.klmnop_qrst7"),
        ];
        let mut cells = 0;
        for (own, before, key) in keys {
            let own = own.then(|| key.to_string());
            let http = Http::new("http://e.test", HttpApi::Openai, "m", own).expect("an endpoint");
            let start: String = key.chars().take(BEGUN).collect();
            let said = format!("bad key: {before}");
            // Cut off at every byte of the key.
            for cut in 1..key.len() {
                let bytes = [said.as_bytes(), &key.as_bytes()[..cut]].concat();
                let passed = http.opening(&heard(bytes), false);
                assert!(!passed.contains(&start), "{key} cut at {cut}: {passed}");
                assert!(
                    passed.starts_with("bad key:"),
                    "{key} cut at {cut}: {passed}"
                );
                cells += 1;
            }
            // Whole, with looking for keys ending at every byte of it.
            for seen in 0..=key.len() + 1 {
                let padding = "x ".repeat((LOOKED_THROUGH + 1 - seen - said.len()) / 2);
                let all = format!("{padding}{said}{key} and more");
                for whole in [true, false] {
                    let passed = http.opening(&all, whole);
                    assert!(!passed.contains(&start), "{key}, {seen} seen: {passed}");
                    cells += 1;
                }
            }
            // The start of the endpoint's own key is not passed on though
            // it is all that was said: but a few characters it begins
            // with are what any word may end with.
            if http.key.is_some() {
                let begun: String = key.chars().take(BEGUN).collect();
                let whole = format!("{said}{begun}");
                assert_eq!(http.opening(&whole, true), said);
                let fewer: String = key.chars().take(BEGUN - 1).collect();
                let whole = format!("{said}{fewer}");
                assert_eq!(http.opening(&whole, true), whole);
                cells += 2;
            }
        }
        assert_eq!(cells, 1_176, "the table's size");
    }

    #[test]
    fn a_key_cut_short_where_looking_for_keys_ends_is_not_passed_on() {
        let own = "its_own_key,of;no:shape1";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        // What is said first, with a key in it that takes up the rest of
        // `bytes`, and what is passed on of that.
        let value = |bytes: usize| {
            let start = "GET https://e.test/v1?token=";
            let said = format!("{start}{}", "x".repeat(bytes - start.len()));
            (said, format!("{start}<redacted>"))
        };
        let jwt = |bytes: usize| {
            let part = (bytes - "token eyJ.eyJ.".len()) / 3;
            let jwt = format!(
                "eyJ{}.eyJ{}.{}",
                "a".repeat(part),
                "b".repeat(part),
                "c".repeat(bytes - "token eyJ.eyJ.".len() - 2 * part)
            );
            (format!("token {jwt}"), "token <redacted>".to_string())
        };
        // Keys, each with what comes before it that it is known by, if
        // anything, which is not part of it and stays.
        let keys = [
            ("", "sk-proj-AbCd1234EfGh5678IjKl9012MnOp"),
            ("", "hf_AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"),
            ("", "AKIAIOSFODNN7EXAMPLE"),
            ("", "AIzaSyA-0123456789abcdefghijklmnopqrstu"),
            ("", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl"),
            ("", "sk-AbCd\u{2026}MnOp"),
            ("Bearer ", "abcdefghij.klmnop_qrst7"),
            ("", own),
        ];
        for (before, key) in keys {
            let cuts = (1..key.len()).filter(|&at| key.is_char_boundary(at));
            for kept in cuts {
                for early in [value, jwt] {
                    let (first, passed_on) = early(LOOKED_THROUGH - 1 - before.len() - kept);
                    let said = format!("{first} {before}{key} was refused, and so on");
                    assert_eq!(
                        said[..LOOKED_THROUGH],
                        format!("{first} {before}{}", &key[..kept])
                    );
                    assert_eq!(
                        http.opening(&said, true),
                        format!("{passed_on} {before}"),
                        "with {kept} bytes of {key} before the cut"
                    );
                }
            }
        }
        // The three as they were found to be passed on. The token is some
        // 3.9 kilobytes, and the key after it is cut at 22 characters.
        let cut = "sk-proj-AbCd1234EfGh56";
        let (first, passed_on) = value(LOOKED_THROUGH - 1 - cut.len());
        let said = format!("{first} {cut}78IjKl9012MnOp is not a key of ours");
        assert_eq!(http.opening(&said, true), format!("{passed_on} "));
        let said = format!("{first} hf_AbCdEfGhIjKlMnOpQrSt0123456789 is not");
        assert_eq!(http.opening(&said, true), format!("{passed_on} "));
        let (first, passed_on) = jwt(3_900);
        let pad = "é".repeat((LOOKED_THROUGH - 3_900 - 2 - cut.len()) / 2);
        let said = format!("{first} {pad} {cut}78IjKl9012MnOp is not a key of ours");
        assert_eq!(http.opening(&said, true), format!("{passed_on} {pad} "));

        // What ends where looking does is whole, and is kept; so is what
        // no key is made of, however long it runs on.
        let (first, passed_on) = value(LOOKED_THROUGH - " refused".len());
        let said = format!("{first} refused by the endpoint");
        assert_eq!(http.opening(&said, true), format!("{passed_on} refused"));
        let (first, passed_on) = value(LOOKED_THROUGH - 1 - 2 * 3);
        let said = format!("{first} {}", "密".repeat(40));
        assert_eq!(http.opening(&said, true), format!("{passed_on} 密密"));
        // And a word cut short goes, which costs a word.
        let (first, passed_on) = value(LOOKED_THROUGH - " refu".len());
        let said = format!("{first} refused by the endpoint");
        assert_eq!(http.opening(&said, true), format!("{passed_on} "));
    }

    #[test]
    fn a_refusal_is_read_for_a_fault_only_as_openai_words_one() {
        let coded = r#"{"error":{"message":"no","type":"t","code":"invalid_api_key"}}"#;
        assert_eq!(fault(HttpApi::Openai, coded), Some(ModelFault::Setup));
        assert_eq!(fault(HttpApi::Tei, coded), None);
        assert_eq!(fault(HttpApi::Openai, &coded[..coded.len() - 1]), None);
    }

    #[test]
    fn keys_are_looked_for_only_in_the_start_of_what_an_endpoint_says() {
        let key = "sk-proj-AbCd1234EfGh5678IjKl9012MnOp";
        let http = Http::new(
            "http://e.test",
            HttpApi::Openai,
            "m",
            Some("its-own-key-0123".into()),
        )
        .expect("an endpoint");
        // The shapes are made ready when first looked for, which is not
        // what is timed here.
        crate::without_keys("");
        // A key lying across where it is cut, however much is said after:
        // sixteen megabytes of it take no longer than none.
        for kept in [1, 8, 20, key.len()] {
            let start = format!("{} its-own-key-0123 {key} ", "é".repeat(280 - kept));
            let said = format!("{start}{}", "and more ".repeat(16 * 1024 * 1024 / 9));
            let whole = format!("{start}{}", "and more ".repeat(40))
                .replace(key, "<redacted>")
                .replace("its-own-key-0123", "<redacted>");
            let cut: String = whole.chars().take(REFUSAL_CHARS).collect();
            let began = std::time::Instant::now();
            assert_eq!(
                http.opening(&said, true),
                cut,
                "with {kept} of the key before the cut"
            );
            let took = began.elapsed();
            assert!(took < Duration::from_millis(50), "it took {took:?}");
        }
        // Nor does what is found over and over and is never a key.
        for never in ["ask-", "-sk-", " hf_", ".eyJ", "AKIA", " bearer ", "-key="] {
            let said = never.repeat(4 * 1024 * 1024 / never.len());
            let began = std::time::Instant::now();
            let opening = http.opening(&said, true);
            let took = began.elapsed();
            assert!(took < Duration::from_millis(50), "{never} took {took:?}");
            assert!(opening.chars().count() <= REFUSAL_CHARS);
        }
        // What is said is cut where a character ends, wherever that is.
        for bytes in LOOKED_THROUGH - 3..=LOOKED_THROUGH + 3 {
            let said = format!("{}{}", "a".repeat(bytes % 4), "密".repeat(bytes / 3));
            let cut: String = said.chars().take(REFUSAL_CHARS).collect();
            assert_eq!(http.opening(&said, true), cut);
        }
        assert_eq!(http.opening("", true), "");
    }

    /// Looking for the endpoint's own key costs the length of what was
    /// said times the key's, whatever both are made of: a key of a
    /// thousand `a`s against four kilobytes of them, which set against
    /// each other one run at a time cost 1.45 seconds in a release build
    /// and 34 in a debug one, take milliseconds in a release build and
    /// under a second in a debug one.
    #[test]
    fn looking_for_the_own_key_costs_no_more_than_the_text_times_the_key() {
        for (key, said) in [
            ("a".repeat(1_000), "a".repeat(LOOKED_THROUGH)),
            (
                format!("sk-{}", "x7".repeat(1_000)),
                format!("{} and more", "x7".repeat(2_000)),
            ),
        ] {
            let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(key.clone()))
                .expect("an endpoint");
            let began = std::time::Instant::now();
            let passed = http.opening(&said, true);
            let took = began.elapsed();
            // Well short of the 34 seconds a debug build took before, on
            // a busy machine.
            assert!(took < Duration::from_secs(5), "it took {took:?}");
            assert!(!passed.contains(&key[..PIECE]), "{passed}");
        }
    }

    /// A key with what reads as an escape in it, as `%41`, is found as it
    /// was said as well as as it reads.
    #[test]
    fn a_key_with_what_reads_as_an_escape_in_it_is_found_as_said() {
        let own = "ab%41cd%2Fef-0123456789xyz";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        assert_eq!(
            http.opening("key ab%41cd%2F is bad", true),
            "key <redacted> is bad"
        );
        let in_json = r#"{"error":"bad key ab%41cd%2Fef-0123"}"#;
        assert_eq!(
            http.opening(in_json, true),
            r#"{"error":"bad key <redacted>"}"#
        );
        // Seven characters of it as said stay, as any word may share them.
        assert_eq!(
            http.opening("key ab%41cd is bad", true),
            "key ab%41cd is bad"
        );
    }

    /// A key with a backslash in it, which JSON writes as two, is found
    /// so, and as it is.
    #[test]
    fn a_key_with_a_backslash_in_it_is_found_as_json_writes_it() {
        let own = "abc\\def-0123456789xyz";
        let http = Http::new("http://e.test", HttpApi::Openai, "m", Some(own.into()))
            .expect("an endpoint");
        let in_json = r#"{"error":"bad key abc\\def-0123"}"#;
        assert_eq!(
            http.opening(in_json, true),
            r#"{"error":"bad key <redacted>"}"#
        );
        assert_eq!(
            http.opening("bad key abc\\def-0123 here", true),
            "bad key <redacted> here"
        );
    }

    /// A key too short to be one, or with space in it, is refused where
    /// an endpoint is made: said by its endpoint, never by the key.
    #[test]
    fn an_endpoint_is_not_made_with_a_key_too_short_to_be_one() {
        for short in [
            "1",
            "40",
            "e",
            "ab",
            "the",
            "0123456789abcde",
            "sk-abcdef 0123456789",
        ] {
            let made = Http::new("http://e.test/", HttpApi::Openai, "m", Some(short.into()));
            assert_eq!(
                made.unwrap_err(),
                EmbedError::Unusable {
                    model: "m".into(),
                    reason: "the key for http://e.test is shorter than 16 characters or has \
                             space in it, which no key an endpoint gives out is: leave the key \
                             out if the endpoint needs none"
                        .into(),
                    fault: ModelFault::Setup,
                }
            );
        }
        assert!(Http::new(
            "http://e.test",
            HttpApi::Openai,
            "m",
            Some("0123456789abcdef".into())
        )
        .is_ok());
        assert!(Http::new("http://e.test", HttpApi::Openai, "m", Some("  ".into())).is_ok());
        assert!(Http::new("http://e.test", HttpApi::Openai, "m", None).is_ok());
    }
}
