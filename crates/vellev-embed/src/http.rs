use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::RETRY_AFTER;
use serde::Deserialize;
use serde_json::json;
use vellev_core::{HttpApi, ModelFault};

use crate::{EmbedError, Provider, Purpose, Result};

/// Texts per request.
const BATCH: usize = 32;
const ATTEMPTS: u32 = 5;
const FIRST_BACKOFF: Duration = Duration::from_millis(500);
/// The longest an endpoint can make its caller wait before the next try.
const LONGEST_BACKOFF: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one call to embed may take, every batch and try included.
const DEADLINE: Duration = Duration::from_secs(120);
/// How much of an endpoint's refusal is passed on.
const REFUSAL_CHARS: usize = 300;
/// How many bytes of what an endpoint says are looked through for keys:
/// what is passed on, at four bytes a character at most, and after it
/// more than the length of any key that could begin there and be told
/// for one when cut short.
const LOOKED_THROUGH: usize = 4 * 1024;

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

#[derive(Deserialize)]
struct OpenaiEmbedding {
    index: usize,
    embedding: Vec<f32>,
}

impl Http {
    /// `key`, unless blank, is sent as a bearer token. An OpenAI-compatible
    /// `url` may be given with or without its `/v1`.
    pub fn new(url: &str, api: HttpApi, model: &str, key: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(ATTEMPT_TIMEOUT)
            .build()
            .map_err(|error| EmbedError::failed(model, format!("no HTTP client: {error}")))?;
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
    /// endpoint that says a great deal does not cost a great deal.
    fn opening(&self, said: &str) -> String {
        let end = (0..=said.len().min(LOOKED_THROUGH))
            .rev()
            .find(|&at| said.is_char_boundary(at))
            .unwrap_or_default();
        let without = crate::without_keys(&self.redacted(said[..end].to_string()));
        without.chars().take(REFUSAL_CHARS).collect()
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
                    let refusal = response.text().await.unwrap_or_default();
                    let lasting = fault(self.api, &refusal);
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

    async fn batches(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            vectors.extend(self.batch(batch).await?);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
