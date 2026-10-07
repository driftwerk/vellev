//! Building a node from what its operator configures.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use eyre::{bail, eyre, Result, WrapErr};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use vellev_core::NodeId;
use vellev_embed::Embedders;
use vellev_node::{Node, Settings};

use crate::server::Limits;
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::PostgresCoordinator;
use vellev_storage::format::{DEFAULT_MAX_FORMAT, FORMAT_VERSION};

#[derive(Clone, clap::Args)]
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
    /// A local directory to keep fetched data files in. Vectors are read
    /// in place from it; a node without one holds every vector in memory.
    #[arg(long, env = "VELLEV_CACHE_DIR")]
    pub cache_dir: Option<PathBuf>,
    /// The public keys whose identity tokens are trusted, as KID=X separated
    /// by commas, X being the Ed25519 key in unpadded base64url as a JWK's
    /// `x` holds it. A read of a collection with an access rule made with a
    /// tenant's key sees only what the token's person may.
    #[arg(long, env = "VELLEV_IDENTITY_KEYS", value_delimiter = ',')]
    pub identity_keys: Vec<String>,
    /// The admin keys, separated by commas: each reaches every tenant and
    /// alone may create, drop and list tenants and issue their keys. A key
    /// that reaches one tenant is issued through the API.
    #[arg(
        long,
        env = "VELLEV_API_KEYS",
        hide_env_values = true,
        value_delimiter = ','
    )]
    pub api_keys: Vec<String>,
    /// The most results a search or a page may ask for.
    #[arg(long, env = "VELLEV_MAX_LIMIT", default_value_t = 1_000, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_limit: u32,
    /// The most bytes a request's body may have.
    #[arg(long, env = "VELLEV_MAX_BODY_BYTES", default_value_t = 32 * 1024 * 1024)]
    pub max_body_bytes: usize,
    /// Where local embedding models are kept once fetched.
    #[arg(
        long,
        env = "VELLEV_MODEL_DIR",
        default_value = "/var/lib/vellev/models"
    )]
    pub model_dir: PathBuf,
    /// The embedding endpoints a collection may name, separated by commas.
    /// Those given a key below may be named too.
    #[arg(long, env = "VELLEV_EMBEDDING_URLS", value_delimiter = ',')]
    pub embedding_urls: Vec<String>,
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
    /// Where other nodes can reach this one, so that a request for a tenant
    /// this node holds can be sent on to it: this node's own address, never
    /// a load balancer's. By default this host's name and the port
    /// listened on.
    #[arg(long, env = "VELLEV_ADVERTISE_URL")]
    pub advertise_url: Option<String>,
    /// How long the node's hold on a tenant lasts, in seconds.
    #[arg(long, env = "VELLEV_LEASE_SECS", default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..=3600))]
    pub lease_secs: u64,
    /// A tenant with this many rows written since its last checkpoint is
    /// flushed.
    #[arg(long, env = "VELLEV_FLUSH_ROWS", default_value_t = 50_000)]
    pub flush_rows: usize,
    /// How far back a tenant may be restored unless it is set otherwise for
    /// it, in seconds. What a restore that far back needs is kept.
    #[arg(long, env = "VELLEV_RESTORE_WINDOW_SECS", default_value_t = 14 * 24 * 60 * 60)]
    pub restore_window_secs: u64,
    /// Where tenants are backed up to, as `VELLEV_STORE` names a store;
    /// best a bucket of its own. Without one nothing is backed up.
    #[arg(long, env = "VELLEV_BACKUP_STORE")]
    pub backup_store: Option<String>,
    /// How often each tenant is backed up, in seconds.
    #[arg(long, env = "VELLEV_BACKUP_EVERY_SECS", default_value_t = 24 * 60 * 60)]
    pub backup_every_secs: u64,
    /// How long a backup is kept, in seconds; a tenant's newest always is.
    #[arg(long, env = "VELLEV_BACKUPS_KEPT_SECS", default_value_t = 35 * 24 * 60 * 60)]
    pub backups_kept_secs: u64,
    /// How much longer than a tenant's restore window its files that
    /// nothing reaches any more, such as segments a merge replaced, are
    /// kept before they are deleted, in seconds.
    #[arg(long, env = "VELLEV_GRACE_SECS", default_value_t = 60 * 60)]
    pub grace_secs: u64,
    /// How long a tenant this node writes for may go unused before it is
    /// let go, its memory given back, in seconds.
    #[arg(long, env = "VELLEV_TENANT_IDLE_SECS", default_value_t = 15 * 60)]
    pub tenant_idle_secs: u64,
    /// The most queries a second one tenant may ask of a node; unlimited
    /// if unset. A tenant asking more is answered 429 with Retry-After.
    #[arg(long, env = "VELLEV_TENANT_QUERIES_PER_SECOND")]
    pub tenant_queries_per_second: Option<f64>,
    /// The most objects a second one tenant may write through a node;
    /// unlimited if unset.
    #[arg(long, env = "VELLEV_TENANT_WRITES_PER_SECOND")]
    pub tenant_writes_per_second: Option<f64>,
    /// About how many megabytes the tenants a node holds may take in
    /// memory together; past it the least recently used are let go.
    /// Unbounded if unset. Vectors read in place from the cache directory
    /// do not count, being the operating system's to page out.
    #[arg(long, env = "VELLEV_MEMORY_MB")]
    pub memory_mb: Option<usize>,
    /// The newest file format nodes may write, so that the release before
    /// can still read everything if this one is rolled back. Raise it once
    /// every node runs the release that reads it and rollback is no longer
    /// wanted; see docs/compatibility.md.
    #[arg(
        long,
        env = "VELLEV_MAX_FORMAT",
        default_value_t = DEFAULT_MAX_FORMAT,
        value_parser = clap::value_parser!(u16).range(1..=i64::from(FORMAT_VERSION))
    )]
    pub max_format: u16,
    /// How long a node told to stop keeps serving while it reports not
    /// ready, in seconds, so that whatever routes requests to it stops
    /// before it closes its port: Docker Swarm's routing mesh and load
    /// balancers that check `/health/ready` need a few seconds. ECS stops
    /// sending before it signals, so none is needed there.
    #[arg(long, env = "VELLEV_STOP_DELAY_SECS", default_value_t = 0)]
    pub stop_delay_secs: u64,
    /// How long requests in flight are then given to finish, in seconds.
    /// Keep the delay and this under the platform's own grace period (ECS
    /// `stopTimeout`, Swarm `stop_grace_period`), so that the node lets go
    /// of its tenants before it is killed.
    #[arg(long, env = "VELLEV_DRAIN_SECS", default_value_t = 20)]
    pub drain_secs: u64,
}

/// The store a URL names. A bucket is used whole: a path within one would
/// be ignored, and two nodes given different paths would write the same
/// files.
fn store(url: &str) -> Result<ObjectBlobs> {
    if url == "memory://" {
        return Ok(ObjectBlobs::new(Arc::new(InMemory::new())));
    }
    if let Some(path) = url.strip_prefix("file://") {
        if !path.starts_with('/') {
            bail!("{url} must name an absolute path, as file:///path");
        }
        std::fs::create_dir_all(path).wrap_err_with(|| format!("creating {path}"))?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
        return Ok(ObjectBlobs::on_disk(store, path));
    }
    if let Some(bucket) = url.strip_prefix("s3://") {
        if bucket.is_empty() || bucket.trim_end_matches('/').contains('/') {
            bail!("{url} must name a bucket and nothing within it");
        }
        let store = AmazonS3Builder::from_env().with_url(url).build()?;
        return Ok(ObjectBlobs::new(Arc::new(store)));
    }
    bail!("{url} is not an s3://, file:// or memory:// store")
}

fn endpoint(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
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
                Ok((endpoint(url), token.trim().to_string()))
            })
            .collect()
    }

    /// What callers may ask of the node, and the store it keeps data in.
    /// Everything that can be wrong without reaching another machine is
    /// found wrong here.
    pub fn checked(&self) -> Result<(Limits, ObjectBlobs)> {
        let keyed = self.embedding_keys()?;
        let embedding_urls = self
            .embedding_urls
            .iter()
            .map(|url| endpoint(url))
            .filter(|url| !url.is_empty())
            .chain(keyed.into_keys())
            .collect();
        if self.backup_store.as_deref() == Some(self.store.as_str()) {
            bail!("tenants cannot be backed up to the store they are kept in");
        }
        if let Some(url) = &self.advertise_url {
            let parsed = reqwest::Url::parse(url).ok();
            let usable = parsed.is_some_and(|url| {
                matches!(url.scheme(), "http" | "https")
                    && url.host().is_some()
                    && url.path() == "/"
            });
            if !usable {
                bail!("{url} is not an http(s) address other nodes could reach this one at");
            }
        }
        let limits = Limits {
            results: self.max_limit as usize,
            body: self.max_body_bytes,
            embedding_urls,
        };
        Ok((limits, store(&self.store)?))
    }

    /// A node over `store`, once Postgres can be reached, reachable by
    /// other nodes at `advertise_url` or else at this host on `port`.
    pub async fn node(&self, store: ObjectBlobs, port: u16) -> Result<Node> {
        let coordinator =
            PostgresCoordinator::connect(&self.database_url, self.database_connections)
                .await
                .wrap_err("connecting to Postgres")?;
        let blobs: Arc<dyn Blobs> = Arc::new(store);
        let blobs = match &self.cache_dir {
            Some(dir) => Arc::new(CachedBlobs::new(blobs, dir)),
            None => blobs,
        };
        let embedders = Embedders::new(vellev_embed::Settings {
            cache: self.model_dir.clone(),
            keys: self.embedding_keys()?,
        });
        let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "vellev".into());
        let address = self
            .advertise_url
            .clone()
            .unwrap_or_else(|| format!("http://{host}:{port}"));
        let settings = Settings {
            lease: Duration::from_secs(self.lease_secs),
            flush_rows: self.flush_rows,
            restore_window: Duration::from_secs(self.restore_window_secs),
            backups: self
                .backup_store
                .as_deref()
                .map(self::store)
                .transpose()
                .wrap_err("opening the backup store")?
                .map(|store| Arc::new(store) as Arc<dyn Blobs>),
            backup_every: Duration::from_secs(self.backup_every_secs),
            backups_kept: Duration::from_secs(self.backups_kept_secs),
            grace: Duration::from_secs(self.grace_secs),
            max_format: self.max_format,
            tenant_idle: Duration::from_secs(self.tenant_idle_secs),
            queries_per_second: self.tenant_queries_per_second,
            writes_per_second: self.tenant_writes_per_second,
            memory: self
                .memory_mb
                .map(|megabytes| megabytes.saturating_mul(1024 * 1024)),
            ..Settings::new(NodeId::reachable_at(address.trim_end_matches('/'), &host))
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

        let (limits, _) = keyed.checked().unwrap();
        assert_eq!(limits.embedding_urls.len(), 2);
        let named = config(&["--embedding-urls", "http://tei:80/, http://other"]);
        let (limits, _) = named.checked().unwrap();
        assert_eq!(
            *limits.embedding_urls,
            ["http://tei:80".to_string(), "http://other".to_string()]
        );
        let nowhere = ["vellev", "--database-url", "x", "--store", "nowhere"];
        assert!(Parsed::parse_from(nowhere).config.checked().is_err());
    }

    #[test]
    fn a_store_is_named_by_its_url() {
        assert!(store("memory://").is_ok());
        let dir = std::env::temp_dir().join(format!("vellev-store-{}", std::process::id()));
        assert!(store(&format!("file://{}", dir.display())).is_ok());
        assert!(dir.is_dir());
        std::fs::remove_dir_all(dir).unwrap();
        for unknown in [
            "",
            "ftp://host/path",
            "bucket",
            "file://host/path",
            "s3://",
            "s3://bucket/prefix",
        ] {
            assert!(store(unknown).is_err(), "{unknown}");
        }
        assert!(store("s3://bucket").is_ok() && store("s3://bucket/").is_ok());
        assert_eq!(config(&[]).max_limit, 1_000);
        let zero = [
            "vellev",
            "--database-url",
            "x",
            "--store",
            "memory://",
            "--max-limit",
            "0",
        ];
        assert!(Parsed::try_parse_from(zero).is_err());
    }
}
