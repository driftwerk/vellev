//! Client for the vellev HTTP API.

use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("request to vellev failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("vellev returned {0}")]
    Status(reqwest::StatusCode),
}

#[derive(Debug, Clone)]
pub struct Client {
    base_url: String,
    http: reqwest::Client,
}

impl Client {
    /// `base_url` is the node or load balancer address, e.g. `http://vellev:8080`.
    pub fn new(base_url: &str, timeout: Duration) -> Result<Self, Error> {
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::builder().timeout(timeout).build()?,
        })
    }

    /// Whether the node's process is up.
    pub async fn live(&self) -> Result<(), Error> {
        self.check("/health/live").await
    }

    /// Whether the node can serve requests.
    pub async fn ready(&self) -> Result<(), Error> {
        self.check("/health/ready").await
    }

    async fn check(&self, path: &str) -> Result<(), Error> {
        let resp = self
            .http
            .get(format!("{}{path}", self.base_url))
            .send()
            .await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(Error::Status(resp.status()))
        }
    }
}
