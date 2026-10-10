use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::RETRY_AFTER;
use serde::Deserialize;
use serde_json::json;
use vellev_core::{HttpApi, ModelFault};

use crate::{EmbedError, Progress, Provider, Purpose, Result};

/// Texts per request.
const BATCH: usize = 32;
const ATTEMPTS: u32 = 5;
const FIRST_BACKOFF: Duration = Duration::from_millis(500);
/// The longest an endpoint can make its caller wait before the next try.
const LONGEST_BACKOFF: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one call to embed may take, every batch and try included,
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

/// The start of what `refusing` says in refusing, and whether that is all
/// of it: no more than [`REFUSAL_READ`] is read, however much is sent. So
/// an endpoint that says a great deal costs neither the memory to hold it
/// nor the time to read it for a fault. One that stops coming partway is
/// not all of it either.
async fn said(mut refusing: reqwest::Response) -> (String, bool) {
    let mut read = Vec::new();
    // It is all of it once the endpoint has no more to say: which is
    // not so of what was cut off on its way, or goes on past what is
    // read.
    let mut whole = false;
    while !whole && read.len() <= REFUSAL_READ {
        match refusing.chunk().await {
            Ok(Some(more)) => read.extend_from_slice(&more),
            Ok(None) => whole = true,
            Err(_) => break,
        }
    }
    read.truncate(REFUSAL_READ);
    (String::from_utf8_lossy(&read).into_owned(), whole)
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
        Ok(Self {
            client,
            url: url.to_string(),
            api,
            model: model.to_string(),
            key: key.filter(|key| !key.trim().is_empty()),
        })
    }

    /// `text` without this endpoint's key, whatever its shape. Anything
    /// else shaped like a key is taken out where the failure is made.
    fn redacted(&self, text: String) -> String {
        match &self.key {
            Some(key) => text.replace(key.as_str(), "<redacted>"),
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
    fn opening(&self, said: &str) -> String {
        let seen = &said[..self.seen_whole(said)];
        let without = crate::without_keys(&self.redacted(seen.to_string()));
        without.chars().take(REFUSAL_CHARS).collect()
    }

    /// How much of the start of `said` is looked through for keys: up to
    /// [`LOOKED_THROUGH`] of it, less whatever at the end of that may be
    /// the start of a key that goes on past it. A key cut there is not
    /// shaped like one, and where a long key before it is taken out, what
    /// was left of it would be drawn into what is passed on.
    ///
    /// That is whatever there is the start of this endpoint's own key,
    /// and then the run of characters a key is made of, if the run goes
    /// on. What is said in fewer bytes than that is all looked through.
    fn seen_whole(&self, said: &str) -> usize {
        let mut end = (0..=said.len().min(LOOKED_THROUGH))
            .rev()
            .find(|&at| said.is_char_boundary(at))
            .unwrap_or_default();
        if end == said.len() {
            return end;
        }
        // The longest start of the key that what is seen ends with: not
        // all of the key, which is taken out wherever it is whole.
        let of_its_own = self.key.as_deref().and_then(|key| {
            let starts = key.char_indices().rev().map(|(at, _)| &key[..at]);
            starts
                .filter(|start| !start.is_empty())
                .find(|start| said[..end].ends_with(start))
        });
        end -= of_its_own.map_or(0, str::len);
        if said[end..].chars().next().is_some_and(crate::of_a_key) {
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

    /// One request, sent until it is answered or refused for good. An
    /// endpoint that answers at all, though it be to say it is busy, has
    /// not stopped: which is said through `progress`, so that the tries
    /// it asks for are not taken for its saying nothing.
    async fn send(
        &self,
        texts: &[String],
        progress: Option<&Progress>,
    ) -> Result<reqwest::Response> {
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
                    if let Some(progress) = progress {
                        progress.made();
                    }
                    let status = response.status();
                    let asked = response
                        .headers()
                        .get(RETRY_AFTER)
                        .and_then(|value| value.to_str().ok()?.parse().ok())
                        .map(Duration::from_secs);
                    let (refusal, whole) = said(response).await;
                    // A refusal cut short is not read for a fault: what
                    // is left of it is not what the endpoint said.
                    let lasting = whole.then(|| fault(self.api, &refusal)).flatten();
                    let refusal = self.opening(&refusal);
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
            tokio::time::sleep(wait).await;
            backoff *= 2;
            attempt += 1;
        }
    }

    async fn batch(&self, texts: &[String], progress: Option<&Progress>) -> Result<Vec<Vec<f32>>> {
        let response = self.send(texts, progress).await?;
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

    async fn batches(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            vectors.extend(self.batch(batch, None).await?);
        }
        Ok(vectors)
    }
}

#[async_trait]
impl Provider for Http {
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        tokio::time::timeout(DEADLINE, self.batches(&texts))
            .await
            .unwrap_or_else(|_| Err(self.failed(format!("no answer within {DEADLINE:?}"))))
    }

    fn says_how_it_gets_on(&self) -> bool {
        true
    }

    /// An endpoint keeps no line of this node's making, so the texts are
    /// sent at once, a batch at a time, whoever's they are. Each batch
    /// answered is somewhere got: a call of many batches is given up on
    /// when the endpoint stops answering them, and not for how many
    /// there are or how slowly an endpoint others are asking much of
    /// answers each.
    async fn embed_saying(
        &self,
        texts: Vec<String>,
        _purpose: Purpose,
        _tenant: &str,
        progress: Progress,
    ) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            vectors.extend(self.batch(batch, Some(&progress)).await?);
            progress.made();
        }
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long key early in what is said is taken out, which draws what
    /// comes later, up to where looking ends, into what is passed on: and
    /// there a key may lie cut short, too short to be told for one.
    #[test]
    fn a_key_cut_short_where_looking_for_keys_ends_is_not_passed_on() {
        let own = "its own key, of no shape";
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
                        http.opening(&said),
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
        assert_eq!(http.opening(&said), format!("{passed_on} "));
        let said = format!("{first} hf_AbCdEfGhIjKlMnOpQrSt0123456789 is not");
        assert_eq!(http.opening(&said), format!("{passed_on} "));
        let (first, passed_on) = jwt(3_900);
        let pad = "é".repeat((LOOKED_THROUGH - 3_900 - 2 - cut.len()) / 2);
        let said = format!("{first} {pad} {cut}78IjKl9012MnOp is not a key of ours");
        assert_eq!(http.opening(&said), format!("{passed_on} {pad} "));

        // What ends where looking does is whole, and is kept; so is what
        // no key is made of, however long it runs on.
        let (first, passed_on) = value(LOOKED_THROUGH - " refused".len());
        let said = format!("{first} refused by the endpoint");
        assert_eq!(http.opening(&said), format!("{passed_on} refused"));
        let (first, passed_on) = value(LOOKED_THROUGH - 1 - 2 * 3);
        let said = format!("{first} {}", "密".repeat(40));
        assert_eq!(http.opening(&said), format!("{passed_on} 密密"));
        // And a word cut short goes, which costs a word.
        let (first, passed_on) = value(LOOKED_THROUGH - " refu".len());
        let said = format!("{first} refused by the endpoint");
        assert_eq!(http.opening(&said), format!("{passed_on} "));
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
            Some("its-own-key".into()),
        )
        .expect("an endpoint");
        // The shapes are made ready when first looked for, which is not
        // what is timed here.
        crate::without_keys("");
        // A key lying across where it is cut, however much is said after:
        // sixteen megabytes of it take no longer than none.
        for kept in [1, 8, 20, key.len()] {
            let start = format!("{} its-own-key {key} ", "é".repeat(285 - kept));
            let said = format!("{start}{}", "and more ".repeat(16 * 1024 * 1024 / 9));
            let whole = format!("{start}{}", "and more ".repeat(40))
                .replace(key, "<redacted>")
                .replace("its-own-key", "<redacted>");
            let cut: String = whole.chars().take(REFUSAL_CHARS).collect();
            let began = std::time::Instant::now();
            assert_eq!(
                http.opening(&said),
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
            let opening = http.opening(&said);
            let took = began.elapsed();
            assert!(took < Duration::from_millis(50), "{never} took {took:?}");
            assert!(opening.chars().count() <= REFUSAL_CHARS);
        }
        // What is said is cut where a character ends, wherever that is.
        for bytes in LOOKED_THROUGH - 3..=LOOKED_THROUGH + 3 {
            let said = format!("{}{}", "a".repeat(bytes % 4), "密".repeat(bytes / 3));
            let cut: String = said.chars().take(REFUSAL_CHARS).collect();
            assert_eq!(http.opening(&said), cut);
        }
        assert_eq!(http.opening(""), "");
    }
}
