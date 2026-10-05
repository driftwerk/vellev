//! Building a node from what its operator configures.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use eyre::{bail, eyre, Result, WrapErr};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use object_store::ObjectStore;
use vellev_core::NodeId;
use vellev_embed::Embedders;
use vellev_node::{Node, Settings};
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::PostgresCoordinator;

#[derive(Debug, Clone, clap::Args)]
pub struct Config {
    /// Postgres, for the record of who writes each tenant and how far.
    #[arg(long, env = "VELLEV_DATABASE_URL", hide_env_values = true)]
    pub database_url: String,
    #[arg(long, env = "VELLEV_DATABASE_CONNECTIONS", default_value_t = 8)]
    pub database_connections: u32,
    /// Where data is kept: `s3://bucket`, with the usual AWS variables for
    /// region, endpoint and credentials; `file:///path`; or `memory://`.
    #[arg(long, env = "VELLEV_STORE")]
    pub store: String,
    /// A local directory to keep fetched data files in.
    #[arg(long, env = "VELLEV_CACHE_DIR")]
    pub cache_dir: Option<PathBuf>,
    /// The keys callers may present, separated by commas.
    #[arg(
        long,
        env = "VELLEV_API_KEYS",
        hide_env_values = true,
        value_delimiter = ','
    )]
    pub api_keys: Vec<String>,
    /// The most results a search or a page may ask for.
    #[arg(long, env = "VELLEV_MAX_LIMIT", default_value_t = 1_000)]
    pub max_limit: usize,
    /// Where local embedding models are kept once fetched.
    #[arg(
        long,
        env = "VELLEV_MODEL_DIR",
        default_value = "/var/lib/vellev/models"
    )]
    pub model_dir: PathBuf,
    /// A bearer token for each embedding endpoint that wants one, as
    /// `url=token`, separated by commas. The token is all that follows the
    /// first `=`.
    #[arg(
        long,
        env = "VELLEV_EMBEDDING_KEYS",
        hide_env_values = true,
        value_delimiter = ','
    )]
    pub embedding_keys: Vec<String>,
    /// How long the node's hold on a tenant lasts, in seconds.
    #[arg(long, env = "VELLEV_LEASE_SECS", default_value_t = 30)]
    pub lease_secs: u64,
    /// A tenant with this many rows written since its last checkpoint is
    /// flushed.
    #[arg(long, env = "VELLEV_FLUSH_ROWS", default_value_t = 50_000)]
    pub flush_rows: usize,
}

fn store(url: &str) -> Result<Arc<dyn ObjectStore>> {
    if url == "memory://" {
        return Ok(Arc::new(InMemory::new()));
    }
    if let Some(path) = url.strip_prefix("file://") {
        std::fs::create_dir_all(path).wrap_err_with(|| format!("creating {path}"))?;
        return Ok(Arc::new(LocalFileSystem::new_with_prefix(path)?));
    }
    if url.starts_with("s3://") {
        return Ok(Arc::new(AmazonS3Builder::from_env().with_url(url).build()?));
    }
    bail!("{url} is not an s3://, file:// or memory:// store")
}

impl Config {
    /// The keys callers may present. A node that nobody could call is a
    /// mistake, so none is refused.
    pub fn keys(&self) -> Result<Arc<[String]>> {
        let keys: Arc<[String]> = self
            .api_keys
            .iter()
            .map(|key| key.trim().to_string())
            .filter(|key| !key.is_empty())
            .collect();
        if keys.is_empty() {
            bail!("no API key is configured; set VELLEV_API_KEYS");
        }
        Ok(keys)
    }

    fn embedding_keys(&self) -> Result<HashMap<String, String>> {
        self.embedding_keys
            .iter()
            .filter(|pair| !pair.trim().is_empty())
            .map(|pair| {
                let (url, token) = pair
                    .split_once('=')
                    .ok_or_else(|| eyre!("an embedding key must be given as url=token"))?;
                Ok((
                    url.trim().trim_end_matches('/').to_string(),
                    token.trim().to_string(),
                ))
            })
            .collect()
    }

    pub async fn node(&self) -> Result<Node> {
        let coordinator =
            PostgresCoordinator::connect(&self.database_url, self.database_connections)
                .await
                .wrap_err("connecting to Postgres")?;
        let blobs: Arc<dyn Blobs> = Arc::new(ObjectBlobs::new(store(&self.store)?));
        let blobs = match &self.cache_dir {
            Some(dir) => Arc::new(CachedBlobs::new(blobs, dir)),
            None => blobs,
        };
        let embedders = Embedders::new(vellev_embed::Settings {
            cache: self.model_dir.clone(),
            keys: self.embedding_keys()?,
        });
        let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "vellev".into());
        let settings = Settings {
            lease: Duration::from_secs(self.lease_secs),
            flush_rows: self.flush_rows,
            ..Settings::new(NodeId::generate(&host))
        };
        Ok(Node::new(
            Arc::new(coordinator),
            blobs,
            Arc::new(embedders),
            settings,
        ))
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[derive(Parser)]
    struct Parsed {
        #[command(flatten)]
        config: Config,
    }

    fn config(extra: &[&str]) -> Config {
        let required = [
            "vellev",
            "--database-url",
            "postgres://db",
            "--store",
            "memory://",
        ];
        Parsed::parse_from(required.iter().chain(extra)).config
    }

    #[test]
    fn a_node_nobody_could_call_is_refused() {
        assert!(config(&[]).keys().is_err());
        assert!(config(&["--api-keys", " , "]).keys().is_err());
        let keys = config(&["--api-keys", "one, two,,"]).keys().unwrap();
        assert_eq!(*keys, ["one".to_string(), "two".to_string()]);
    }

    #[test]
    fn an_embedding_key_belongs_to_the_endpoint_it_is_given_for() {
        let keyed = config(&[
            "--embedding-keys",
            "https://api.example.test/v1/=sk-a=b,http://tei:80=t",
        ]);
        let keys = keyed.embedding_keys().unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys["https://api.example.test/v1"], "sk-a=b");
        assert_eq!(keys["http://tei:80"], "t");
        assert!(config(&["--embedding-keys", "no-token"])
            .embedding_keys()
            .is_err());
        assert!(config(&[]).embedding_keys().unwrap().is_empty());
    }

    #[test]
    fn a_store_is_named_by_its_url() {
        assert!(store("memory://").is_ok());
        let dir = std::env::temp_dir().join(format!("vellev-store-{}", std::process::id()));
        assert!(store(&format!("file://{}", dir.display())).is_ok());
        assert!(dir.is_dir());
        std::fs::remove_dir_all(dir).unwrap();
        for unknown in ["", "ftp://host/path", "bucket"] {
            assert!(store(unknown).is_err(), "{unknown}");
        }
        assert_eq!(config(&[]).max_limit, 1_000);
    }
}
