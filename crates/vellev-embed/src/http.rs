use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::RETRY_AFTER;
use serde::Deserialize;
use serde_json::json;
use vellev_core::HttpApi;

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
            .map_err(|error| EmbedError::Failed {
                model: model.to_string(),
                reason: format!("no HTTP client: {error}"),
            })?;
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

    fn redacted(&self, text: String) -> String {
        match &self.key {
            Some(key) => text.replace(key.as_str(), "<redacted>"),
            None => text,
        }
    }

    fn failed(&self, reason: impl std::fmt::Display) -> EmbedError {
        EmbedError::Failed {
            model: self.model.clone(),
            reason: self.redacted(reason.to_string()),
        }
    }

    fn unusable(&self, reason: impl std::fmt::Display) -> EmbedError {
        EmbedError::Unusable {
            model: self.model.clone(),
            reason: self.redacted(reason.to_string()),
        }
    }

    fn malformed(&self, reason: String) -> EmbedError {
        EmbedError::Malformed {
            model: self.model.clone(),
            reason,
        }
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
                    let refusal = self.redacted(response.text().await.unwrap_or_default());
                    let refusal: String = refusal.chars().take(REFUSAL_CHARS).collect();
                    let passing = status.as_u16() == 429 || status.is_server_error();
                    let wait = asked.unwrap_or(backoff).min(LONGEST_BACKOFF);
                    let said = format!("{status}: {refusal}");
                    // Refused the key, or has no such model: the node is
                    // set up wrong for this endpoint, whatever the text.
                    // Any other refusal may be about the texts sent.
                    let failure = match status.as_u16() {
                        401 | 403 | 404 => self.unusable(said),
                        _ => self.failed(said),
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
