use std::time::Duration;

use async_trait::async_trait;
use aws_config::retry::RetryConfig;
use aws_config::timeout::TimeoutConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::ProvideErrorMetadata;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use futures::future::try_join_all;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{OnceCell, Semaphore};

use crate::{EmbedError, Provider, Purpose, Result};

const MODEL: &str = "amazon.titan-embed-text-v2";
const DIMENSIONS: [u32; 3] = [256, 512, 1024];
/// Calls in flight at once: Titan takes one text per call.
const CONCURRENCY: usize = 10;
/// Tries per call, throttling and server errors being retried.
const ATTEMPTS: u32 = 5;
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30);
/// How long one text may take, every try included.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// Titan v2 takes up to 8,192 tokens; characters beyond this are dropped.
const MAX_CHARS: usize = 30_000;

/// Amazon Titan Text Embeddings v2 on Bedrock, normalised, with the node's
/// own AWS credentials.
#[derive(Debug)]
pub struct Bedrock {
    region: String,
    model: String,
    dimensions: u32,
    client: OnceCell<Client>,
    permits: Semaphore,
}

#[derive(Deserialize)]
struct Answer {
    embedding: Vec<f32>,
}

impl Bedrock {
    pub fn new(region: &str, model: &str, dimensions: u32) -> Result<Self> {
        if !model.starts_with(MODEL) {
            return Err(EmbedError::Unsupported(format!(
                "{model} is not {MODEL}, the one Bedrock model that can be embedded with"
            )));
        }
        if !DIMENSIONS.contains(&dimensions) {
            return Err(EmbedError::Unsupported(format!(
                "{model} has 256, 512 or 1024 dimensions, not {dimensions}"
            )));
        }
        Ok(Self {
            region: region.to_string(),
            model: model.to_string(),
            dimensions,
            client: OnceCell::new(),
            permits: Semaphore::new(CONCURRENCY),
        })
    }

    /// As [`Self::new`], through a client already configured, which should
    /// have these [`Self::retries`] and [`Self::timeouts`].
    pub fn with_client(client: Client, model: &str, dimensions: u32) -> Result<Self> {
        Ok(Self {
            client: OnceCell::new_with(Some(client)),
            ..Self::new("", model, dimensions)?
        })
    }

    pub fn retries() -> RetryConfig {
        RetryConfig::standard().with_max_attempts(ATTEMPTS)
    }

    pub fn timeouts() -> TimeoutConfig {
        TimeoutConfig::builder()
            .operation_attempt_timeout(ATTEMPT_TIMEOUT)
            .operation_timeout(CALL_TIMEOUT)
            .build()
    }

    async fn client(&self) -> &Client {
        self.client
            .get_or_init(|| async {
                let config = aws_config::defaults(BehaviorVersion::latest())
                    .region(Region::new(self.region.clone()))
                    .retry_config(Self::retries())
                    .timeout_config(Self::timeouts())
                    .load()
                    .await;
                Client::new(&config)
            })
            .await
    }

    fn failed(&self, reason: impl std::fmt::Display) -> EmbedError {
        EmbedError::Failed {
            model: self.model.clone(),
            reason: reason.to_string(),
        }
    }

    async fn one(&self, text: &str) -> Result<Vec<f32>> {
        let _permit = self.permits.acquire().await.map_err(|e| self.failed(e))?;
        let text: String = text.chars().take(MAX_CHARS).collect();
        let body = json!({
            "inputText": text,
            "dimensions": self.dimensions,
            "normalize": true,
        });
        let answer = self
            .client()
            .await
            .invoke_model()
            .model_id(&self.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body.to_string()))
            .send()
            .await
            .map_err(|error| {
                let said = match (error.code(), error.message()) {
                    (Some(code), Some(message)) => format!("{code}: {message}"),
                    (Some(said), None) | (None, Some(said)) => said.to_string(),
                    (None, None) => error.to_string(),
                };
                // Bedrock says the region has no such model: nothing passes
                // until the collection or the region is another. Access
                // denied is not taken so, being also what is said while a
                // permission just granted spreads; nor a refused input,
                // which may be this text's alone.
                let lasting = error
                    .as_service_error()
                    .is_some_and(|refused| refused.is_resource_not_found_exception());
                match lasting {
                    true => EmbedError::Unusable {
                        model: self.model.clone(),
                        reason: said,
                    },
                    false => self.failed(said),
                }
            })?;
        serde_json::from_slice::<Answer>(answer.body().as_ref())
            .map(|answer| answer.embedding)
            .map_err(|error| EmbedError::Malformed {
                model: self.model.clone(),
                reason: format!("an unreadable answer: {error}"),
            })
    }
}

#[async_trait]
impl Provider for Bedrock {
    async fn embed(&self, texts: Vec<String>, _purpose: Purpose) -> Result<Vec<Vec<f32>>> {
        try_join_all(texts.iter().map(|text| self.one(text))).await
    }
}
