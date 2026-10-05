use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use vellev_core::HttpApi;

use crate::{EmbedError, Provider, Result};

/// Texts per request.
const BATCH: usize = 32;
const ATTEMPTS: u32 = 4;
const FIRST_BACKOFF: Duration = Duration::from_millis(250);
const TIMEOUT: Duration = Duration::from_secs(60);
/// How much of an endpoint's refusal is passed on.
const REFUSAL_CHARS: usize = 300;

/// A text-embeddings-inference server or an OpenAI-compatible endpoint.
/// Requests that fail to arrive, are throttled, or fail on the server's
/// side are tried again.
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
    /// `key`, if any, is sent as a bearer token.
    pub fn new(url: &str, api: HttpApi, model: &str, key: Option<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(TIMEOUT)
                .build()
                .unwrap_or_default(),
            url: url.trim_end_matches('/').to_string(),
            api,
            model: model.to_string(),
            key,
        }
    }

    fn failed(&self, reason: impl std::fmt::Display) -> EmbedError {
        let reason = reason.to_string();
        let reason = match &self.key {
            Some(key) => reason.replace(key.as_str(), "<redacted>"),
            None => reason,
        };
        EmbedError::Failed {
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
            let (failure, lasting) = match request.send().await {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status();
                    let refusal: String = response
                        .text()
                        .await
                        .unwrap_or_default()
                        .chars()
                        .take(REFUSAL_CHARS)
                        .collect();
                    let passing = status.as_u16() == 429 || status.is_server_error();
                    (self.failed(format!("{status}: {refusal}")), !passing)
                }
                Err(error) => (self.failed(error.without_url()), false),
            };
            if lasting || attempt == ATTEMPTS {
                return Err(failure);
            }
            tokio::time::sleep(backoff).await;
            backoff *= 2;
            attempt += 1;
        }
    }

    async fn batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let response = self.send(texts).await?;
        let malformed = |error: reqwest::Error| EmbedError::Malformed {
            model: self.model.clone(),
            reason: format!("an unreadable answer: {}", error.without_url()),
        };
        match self.api {
            HttpApi::Tei => response.json().await.map_err(malformed),
            HttpApi::Openai => {
                let mut answer: OpenaiResponse = response.json().await.map_err(malformed)?;
                answer
                    .data
                    .sort_unstable_by_key(|embedding| embedding.index);
                Ok(answer.data.into_iter().map(|e| e.embedding).collect())
            }
        }
    }
}

#[async_trait]
impl Provider for Http {
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let embedded = self.batch(batch).await?;
            if embedded.len() != batch.len() {
                return Err(EmbedError::Malformed {
                    model: self.model.clone(),
                    reason: format!("{} vectors for {} texts", embedded.len(), batch.len()),
                });
            }
            vectors.extend(embedded);
        }
        Ok(vectors)
    }
}
