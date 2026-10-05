use async_trait::async_trait;
use aws_config::retry::RetryConfig;
use aws_config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::DisplayErrorContext;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client;
use futures::future::try_join_all;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{OnceCell, Semaphore};

use crate::{EmbedError, Provider, Result};

/// Calls in flight at once: Titan takes one text per call.
const CONCURRENCY: usize = 10;
/// Tries per call, throttling and server errors being retried.
pub const ATTEMPTS: u32 = 5;
/// Titan v2 takes up to 8,192 tokens; characters beyond this are dropped.
const MAX_CHARS: usize = 30_000;

/// Amazon Titan Text Embeddings on Bedrock, normalised, with the node's
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
    pub fn new(region: &str, model: &str, dimensions: u32) -> Self {
        Self {
            region: region.to_string(),
            model: model.to_string(),
            dimensions,
            client: OnceCell::new(),
            permits: Semaphore::new(CONCURRENCY),
        }
    }

    /// As [`Self::new`], through a client already configured.
    pub fn with_client(client: Client, model: &str, dimensions: u32) -> Self {
        Self {
            client: OnceCell::new_with(Some(client)),
            ..Self::new("", model, dimensions)
        }
    }

    async fn client(&self) -> &Client {
        self.client
            .get_or_init(|| async {
                let config = aws_config::defaults(BehaviorVersion::latest())
                    .region(Region::new(self.region.clone()))
                    .retry_config(RetryConfig::standard().with_max_attempts(ATTEMPTS))
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
            .map_err(|error| self.failed(DisplayErrorContext(&error)))?;
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
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        try_join_all(texts.iter().map(|text| self.one(text))).await
    }
}
