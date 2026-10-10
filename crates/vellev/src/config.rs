//! Building a node from what its operator configures.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use eyre::{bail, eyre, Result, WrapErr};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use vellev_core::NodeId;
use vellev_embed::Embedders;
use vellev_node::{Node, RequestBounds, Settings};

use crate::server::Limits;
use object_store::aws::AmazonS3ConfigKey;
use object_store::{BackoffConfig, ClientConfigKey, RetryConfig};
use vellev_storage::blobs::{Blobs, CachedBlobs, ObjectBlobs};
use vellev_storage::coordinator::{Bounded, PostgresCoordinator};
use vellev_storage::format::{DEFAULT_MAX_FORMAT, FORMAT_VERSION};
use vellev_storage::{Disk, Transfers};

#[derive(Clone, clap::Args)]
pub struct Config {
    /// Postgres, for the record of who writes each tenant and how far.
    #[arg(long, env = "VELLEV_DATABASE_URL", hide_env_values = true)]
    pub database_url: String,
    #[arg(long, env = "VELLEV_DATABASE_CONNECTIONS", default_value_t = 8)]
    pub database_connections: u32,
    /// How long any one request to Postgres may go unanswered, in seconds,
    /// before it is given up as failed. A commit gets the lesser of this and
    /// a quarter of the lease. To bound statements on the server too, set
    /// `statement_timeout` for the role, or with `options` in the URL.
    #[arg(long, env = "VELLEV_DATABASE_TIMEOUT_SECS", default_value_t = 10, value_parser = clap::value_parser!(u64).range(1..=3600))]
    pub database_timeout_secs: u64,
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
    /// The most reads answered at once; more are turned away, to be asked
    /// again. Each takes one of the runtime's 512 threads for work off the
    /// async ones, of which loading and building take up to 128 and
    /// commits need the rest.
    #[arg(long, env = "VELLEV_READS_AT_ONCE", default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..=320))]
    pub reads_at_once: u32,
    /// How many milliseconds a read waits for its turn, when as many are
    /// being answered as are at once, before it is turned away.
    #[arg(long, env = "VELLEV_READS_WAIT_MS", default_value_t = 500, value_parser = clap::value_parser!(u64).range(0..=60_000))]
    pub reads_wait_ms: u64,
    /// How many whole vectors files are read or fetched into the cache
    /// directory at once, the rest waiting their turn.
    #[arg(long, env = "VELLEV_CACHE_LOADS", default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=64))]
    pub cache_loads: u32,
    /// Where local embedding models are kept once fetched, unless
    /// `HF_HOME` is set, which what fetches them uses instead.
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
    /// Whether this node looks for tenants due a backup of its own accord.
    /// Off, it backs a tenant up only when asked to, and purges removed
    /// tenants as any node does: for a cluster whose backups are to be
    /// taken by some of its nodes and not others.
    #[arg(long, env = "VELLEV_BACKUPS_SCHEDULED", default_value_t = true, action = clap::ArgAction::Set)]
    pub backups_scheduled: bool,
    /// How long a node's claim to back a tenant up or restore it lasts
    /// unless renewed, in seconds: a node that dies holding one blocks the
    /// tenant's backups and restores for no longer. A third of it must
    /// outlast one request to object storage, tries and all
    /// (`VELLEV_STORE_TIMEOUT_SECS` and the rest), and the half minute the
    /// store is given to apply a request the node gave up, so it is at
    /// least three times the two together: 10 minutes 51 seconds as a node
    /// is set up by default.
    #[arg(long, env = "VELLEV_BACKUP_CLAIM_SECS", default_value_t = 15 * 60, value_parser = clap::value_parser!(u64).range(1..=24 * 60 * 60))]
    pub backup_claim_secs: u64,
    /// How long after a tenant is removed its files, backups and cached
    /// copies are deleted, in seconds; it cannot be made again until then.
    /// With a backup store it is no sooner than `VELLEV_BACKUP_CLAIM_SECS`.
    #[arg(long, env = "VELLEV_PURGE_AFTER_SECS", default_value_t = 10 * 60)]
    pub purge_after_secs: u64,
    /// How long one try of a request to object storage may go unanswered, in
    /// seconds, before the time a part takes at
    /// `VELLEV_STORE_MIN_BYTES_PER_SEC` is added. A request that fails in a
    /// way worth trying again is tried up to three times more, a second at
    /// most between tries, and after a timeout only if it is safe to
    /// repeat; it is given up once all that could have run out.
    #[arg(long, env = "VELLEV_STORE_TIMEOUT_SECS", default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..=3600))]
    pub store_timeout_secs: u64,
    /// The slowest the link to object storage is expected to carry a part,
    /// in bytes a second: a try that moves a part slower than this, past
    /// `VELLEV_STORE_TIMEOUT_SECS`, is given up. By default a mebibyte a
    /// second, about 8 Mbit/s, below which a node is not supported.
    #[arg(long, env = "VELLEV_STORE_MIN_BYTES_PER_SEC", default_value_t = 1024 * 1024, value_parser = clap::value_parser!(u64).range(1..))]
    pub store_min_bytes_per_sec: u64,
    /// The size of the parts a file larger than it goes up to object storage
    /// in, and comes down in, in MiB: from 5 to 5120, as S3 takes for every
    /// part but the last. A file may have at most 10,000 parts, so 16 MiB
    /// parts carry files up to 156 GiB.
    #[arg(long, env = "VELLEV_UPLOAD_PART_MIB", default_value_t = 16, value_parser = clap::value_parser!(u64).range(5..=5120))]
    pub upload_part_mib: u64,
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

/// How a node talks to object storage: how large the parts are that a large
/// file goes up and comes down in, and how long one try of a request may
/// take, which is `floor` and as long again as a part takes at
/// `bytes_per_second`, the slowest the link is expected to be.
#[derive(Debug, Clone, Copy)]
struct StoreClient {
    floor: Duration,
    bytes_per_second: u64,
    part_bytes: usize,
}

/// How many times the store client tries a request again after the first
/// try fails in a way worth trying again. It tries again after a timeout
/// only a request that is safe to repeat.
const STORE_RETRIES: u32 = 3;
/// The longest wait between two tries.
const STORE_BACKOFF: Duration = Duration::from_secs(1);

impl StoreClient {
    /// How long one try of a request may take: a part moved at the slowest
    /// the link is expected to be, past the floor. Every request gets as
    /// long, the client having one timeout for all.
    fn try_within(&self) -> Duration {
        let part = self.part_bytes as u64 * 1_000 / self.bytes_per_second.max(1);
        self.floor + Duration::from_millis(part)
    }

    /// How the client tries a request again: [`STORE_RETRIES`] times, never
    /// starting a try once every try could have run out.
    fn retry(&self) -> RetryConfig {
        RetryConfig {
            backoff: BackoffConfig {
                init_backoff: Duration::from_millis(100),
                max_backoff: STORE_BACKOFF,
                base: 2.0,
            },
            max_retries: STORE_RETRIES as usize,
            retry_timeout: self.transfers().request_within,
        }
    }

    /// `builder` set up to give each try up after [`Self::try_within`], to
    /// connect within five seconds, and to try again as [`Self::retry`]
    /// says. Every other setting it was given, from the environment or
    /// otherwise, stays as it was.
    fn configured(&self, builder: AmazonS3Builder) -> AmazonS3Builder {
        let millis = |duration: Duration| format!("{}ms", duration.as_millis());
        let client = |key| AmazonS3ConfigKey::Client(key);
        builder
            .with_config(client(ClientConfigKey::Timeout), millis(self.try_within()))
            .with_config(
                client(ClientConfigKey::ConnectTimeout),
                millis(Duration::from_secs(5)),
            )
            .with_retry(self.retry())
    }

    /// Transfers in parts of `part_bytes`, each request given up once every
    /// try the client may make of it, and every wait between, could have run
    /// out: it is reached when each try of a request safe to repeat times
    /// out.
    fn transfers(&self) -> Transfers {
        Transfers {
            part_bytes: self.part_bytes,
            request_within: self.try_within() * (STORE_RETRIES + 1) + STORE_BACKOFF * STORE_RETRIES,
        }
    }
}

impl Default for StoreClient {
    fn default() -> Self {
        Self {
            floor: Duration::from_secs(30),
            bytes_per_second: 1024 * 1024,
            part_bytes: 16 * 1024 * 1024,
        }
    }
}

/// The variables of `env` that are text, as the process's own are given:
/// one that is not is nothing a store is set up by, and is passed over.
fn text(
    env: impl IntoIterator<Item = (OsString, OsString)>,
) -> impl Iterator<Item = (String, String)> {
    env.into_iter()
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
}

/// An S3 store's builder for `url`, set up from the `AWS_` variables of
/// `env`, as `AmazonS3Builder::from_env` sets one up from the process's, and
/// then as `client` says.
fn s3(
    url: &str,
    client: StoreClient,
    env: impl IntoIterator<Item = (String, String)>,
) -> AmazonS3Builder {
    let from_env = env
        .into_iter()
        .filter(|(key, _)| key.starts_with("AWS_"))
        .filter_map(|(key, value)| Some((key.to_ascii_lowercase().parse().ok()?, value)))
        .fold(AmazonS3Builder::new(), |builder, (key, value)| {
            builder.with_config(key, value)
        });
    client.configured(from_env).with_url(url)
}

/// The store a URL names. A bucket is used whole: a path within one would
/// be ignored, and two nodes given different paths would write the same
/// files.
fn store(url: &str, client: StoreClient) -> Result<ObjectBlobs> {
    let transfers = client.transfers();
    if url == "memory://" {
        return Ok(ObjectBlobs::new(Arc::new(InMemory::new())).transferring(transfers));
    }
    if let Some(path) = url.strip_prefix("file://") {
        if !path.starts_with('/') {
            bail!("{url} must name an absolute path, as file:///path");
        }
        std::fs::create_dir_all(path).wrap_err_with(|| format!("creating {path}"))?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(path)?);
        return Ok(ObjectBlobs::on_disk(store, path).transferring(transfers));
    }
    if let Some(bucket) = url.strip_prefix("s3://") {
        if bucket.is_empty() || bucket.trim_end_matches('/').contains('/') {
            bail!("{url} must name a bucket and nothing within it");
        }
        let store = s3(url, client, text(std::env::vars_os())).build()?;
        return Ok(ObjectBlobs::new(Arc::new(store)).transferring(transfers));
    }
    bail!("{url} is not an s3://, file:// or memory:// store")
}

fn endpoint(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

impl Config {
    fn store_client(&self) -> StoreClient {
        StoreClient {
            floor: Duration::from_secs(self.store_timeout_secs),
            bytes_per_second: self.store_min_bytes_per_sec,
            part_bytes: usize::try_from(self.upload_part_mib.saturating_mul(1024 * 1024))
                .unwrap_or(usize::MAX),
        }
    }

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
        if self.purge_after_secs < 2 * self.lease_secs {
            bail!("VELLEV_PURGE_AFTER_SECS must be at least two leases, so a holder lets go first");
        }
        if self
            .backup_store
            .as_deref()
            .map(|url| url.trim_end_matches('/'))
            == Some(self.store.trim_end_matches('/'))
        {
            bail!("tenants cannot be backed up to the store they are kept in");
        }
        self.settings(NodeId::generate("checked"), None)?;
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
            reads: self.reads_at_once as usize,
            reads_wait: Duration::from_millis(self.reads_wait_ms),
        };
        Ok((limits, store(&self.store, self.store_client())?))
    }

    /// The disk the cache directory is on: the one every store shares,
    /// reading and writing as many whole large files at once as is set.
    fn cache_disk(&self) -> Result<Disk> {
        let disk = Disk::shared().clone();
        let disk = disk
            .loading_at_once(self.cache_loads as usize)
            .wrap_err("VELLEV_CACHE_LOADS")?;
        Ok(disk.fetching(&self.store_client().transfers()))
    }

    /// A node over `store`, once Postgres can be reached, reachable by
    /// other nodes at `advertise_url` or else at this host on `port`.
    pub async fn node(&self, store: ObjectBlobs, port: u16) -> Result<Node> {
        let within = Duration::from_secs(self.database_timeout_secs);
        let coordinator =
            PostgresCoordinator::connect(&self.database_url, self.database_connections, within)
                .await
                .wrap_err("connecting to Postgres")?;
        let blobs: Arc<dyn Blobs> = Arc::new(store);
        let blobs = match &self.cache_dir {
            Some(dir) => Arc::new(CachedBlobs::open(blobs, dir).await.on(self.cache_disk()?)),
            None => blobs,
        };
        let embedders = Embedders::new(vellev_embed::Settings {
            cache: self.model_dir.clone(),
            keys: self.embedding_keys()?,
            ..Default::default()
        });
        let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "vellev".into());
        let address = self
            .advertise_url
            .clone()
            .unwrap_or_else(|| format!("http://{host}:{port}"));
        let backups = self
            .backup_store
            .as_deref()
            .map(|url| self::store(url, self.store_client()))
            .transpose()
            .wrap_err("opening the backup store")?
            .map(|store| Arc::new(store) as Arc<dyn Blobs>);
        let node = NodeId::reachable_at(address.trim_end_matches('/'), &host);
        Ok(Node::new(
            Arc::new(Bounded::new(Arc::new(coordinator), within)),
            blobs,
            Arc::new(embedders),
            self.settings(node, backups)?,
        ))
    }

    /// How `node` is to run, backing tenants up to `backups` if anywhere:
    /// refused if a node could not run so.
    fn settings(&self, node: NodeId, backups: Option<Arc<dyn Blobs>>) -> Result<Settings> {
        let settings = Settings {
            lease: Duration::from_secs(self.lease_secs),
            flush_rows: self.flush_rows,
            restore_window: Duration::from_secs(self.restore_window_secs),
            backups,
            backup_every: Duration::from_secs(self.backup_every_secs),
            backups_kept: Duration::from_secs(self.backups_kept_secs),
            backups_scheduled: self.backups_scheduled,
            backup_claim: Duration::from_secs(self.backup_claim_secs),
            purge_after: Duration::from_secs(self.purge_after_secs),
            transfers: self.store_client().transfers(),
            request_bounds: RequestBounds {
                store: self.store_client().transfers().request_within,
                coordinator: Duration::from_secs(self.database_timeout_secs),
            },
            grace: Duration::from_secs(self.grace_secs),
            max_format: self.max_format,
            tenant_idle: Duration::from_secs(self.tenant_idle_secs),
            queries_per_second: self.tenant_queries_per_second,
            writes_per_second: self.tenant_writes_per_second,
            memory: self
                .memory_mb
                .map(|megabytes| megabytes.saturating_mul(1024 * 1024)),
            ..Settings::new(node)
        };
        settings
            .check()
            .wrap_err("VELLEV_BACKUP_CLAIM_SECS cannot be as it is set")?;
        Ok(settings)
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

    #[test]
    fn the_cache_reads_as_many_large_files_at_once_as_it_is_told_to() {
        assert_eq!(config(&[]).cache_disk().unwrap().loads_at_once(), 2);
        let five = config(&["--cache-loads", "5"]);
        assert_eq!(five.cache_disk().unwrap().loads_at_once(), 5);
    }

    #[test]
    fn a_turn_at_the_cache_is_waited_for_longer_than_a_part_takes_as_the_store_is_set_up() {
        // The longer the store is given to answer in, the longer one
        // fetching a file may go without getting on, and the longer
        // those behind it must wait before they give it up.
        let mut last = Duration::ZERO;
        for secs in ["30", "120", "900"] {
            let config = config(&["--store-timeout-secs", secs]);
            let part = config.store_client().transfers().part_within();
            let still = config.cache_disk().unwrap().loading_still_for();
            assert!(still > part, "{secs}: {still:?} for a part of {part:?}");
            assert!(still > last, "{secs}: {still:?}");
            last = still;
        }
        assert!(last > Duration::from_secs(2 * 4 * 900));
    }

    #[test]
    fn reads_are_answered_as_many_at_once_and_wait_as_long_as_the_node_is_told() {
        let (limits, _) = config(&[]).checked().unwrap();
        assert_eq!(limits.reads, 256);
        assert_eq!(limits.reads_wait, Duration::from_millis(500));
        let set = config(&["--reads-at-once", "7", "--reads-wait-ms", "20"]);
        let (limits, _) = set.checked().unwrap();
        assert_eq!(limits.reads, 7);
        assert_eq!(limits.reads_wait, Duration::from_millis(20));
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
    fn an_s3_store_keeps_every_client_setting_its_environment_gives() {
        let env = [
            ("AWS_ALLOW_HTTP", "true"),
            ("AWS_ENDPOINT", "http://127.0.0.1:59090"),
            ("AWS_PROXY_URL", "http://proxy.internal:3128"),
            ("AWS_ALLOW_INVALID_CERTIFICATES", "true"),
            ("AWS_ACCESS_KEY_ID", "key"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "eu-west-1"),
            ("HOME", "/ignored"),
        ]
        .map(|(key, value)| (key.to_string(), value.to_string()));
        let client = StoreClient::default();
        let builder = s3("s3://bucket", client, env);
        let setting = |key| builder.get_config_value(&AmazonS3ConfigKey::Client(key));
        assert_eq!(setting(ClientConfigKey::AllowHttp).as_deref(), Some("true"));
        assert_eq!(
            setting(ClientConfigKey::ProxyUrl).as_deref(),
            Some("http://proxy.internal:3128")
        );
        assert_eq!(
            setting(ClientConfigKey::AllowInvalidCertificates).as_deref(),
            Some("true")
        );
        assert_eq!(
            setting(ClientConfigKey::Timeout).as_deref(),
            Some("46000ms")
        );
        assert_eq!(
            setting(ClientConfigKey::ConnectTimeout).as_deref(),
            Some("5000ms")
        );
        assert_eq!(
            builder
                .get_config_value(&AmazonS3ConfigKey::Endpoint)
                .as_deref(),
            Some("http://127.0.0.1:59090")
        );
        assert!(builder.build().is_ok(), "a plain http endpoint is allowed");
    }

    #[test]
    fn a_try_is_given_long_enough_for_a_part_at_the_slowest_link_and_a_request_for_every_try() {
        let client = StoreClient {
            floor: Duration::from_secs(30),
            bytes_per_second: 2 * 1024 * 1024,
            part_bytes: 64 * 1024 * 1024,
        };
        assert_eq!(client.try_within(), Duration::from_secs(30 + 32));
        assert_eq!(
            client.transfers().request_within,
            Duration::from_secs(4 * 62 + 3)
        );
        assert_eq!(
            StoreClient::default().transfers(),
            Transfers::default(),
            "a store made without settings bounds its requests as a node's does"
        );
        let retry = client.retry();
        assert_eq!(retry.max_retries, 3);
        assert_eq!(retry.retry_timeout, client.transfers().request_within);
        assert_eq!(retry.backoff.max_backoff, Duration::from_secs(1));
    }

    #[test]
    fn a_variable_of_the_environment_that_is_not_text_is_passed_over() {
        #[cfg(unix)]
        let odd = {
            use std::os::unix::ffi::OsStringExt;
            OsString::from_vec(vec![0xff, 0xfe])
        };
        #[cfg(not(unix))]
        let odd = OsString::from("plain");
        let env = [
            (OsString::from("AWS_REGION"), OsString::from("eu-west-1")),
            (OsString::from("AWS_SESSION_TOKEN"), odd.clone()),
            (odd, OsString::from("value")),
        ];
        let kept: Vec<(String, String)> = text(env).collect();
        assert!(kept.contains(&("AWS_REGION".into(), "eu-west-1".into())));
        assert!(kept.len() <= 2);
    }

    /// A stand-in for S3 that fails every request, counting them.
    async fn failing_store() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counting = asked.clone();
        let router = axum::Router::new().fallback(move || {
            counting.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            async { axum::http::StatusCode::SERVICE_UNAVAILABLE }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        (url, asked)
    }

    #[tokio::test]
    async fn an_s3_store_tries_a_failing_request_as_often_as_it_is_set_up_to() {
        let (endpoint, asked) = failing_store().await;
        let env = [
            ("AWS_ALLOW_HTTP", "true"),
            ("AWS_ENDPOINT", endpoint.as_str()),
            ("AWS_ACCESS_KEY_ID", "key"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "eu-west-1"),
        ]
        .map(|(key, value)| (key.to_string(), value.to_string()));
        let store = s3("s3://bucket", StoreClient::default(), env)
            .build()
            .unwrap();
        let blobs = ObjectBlobs::new(Arc::new(store));
        assert!(blobs.delete("key").await.is_err());
        assert_eq!(
            asked.load(std::sync::atomic::Ordering::SeqCst),
            1 + STORE_RETRIES as usize
        );
    }

    #[test]
    fn a_node_backs_up_of_its_own_accord_unless_told_not_to() {
        let scheduled = |args: &[&str]| {
            let config = config(args);
            let node = NodeId::generate("node");
            config.settings(node, None).unwrap().backups_scheduled
        };
        assert!(scheduled(&[]));
        assert!(!scheduled(&["--backups-scheduled", "false"]));
        assert!(scheduled(&["--backups-scheduled", "true"]));
    }

    #[test]
    fn a_backup_claim_too_short_to_hold_or_too_long_is_refused() {
        assert!(config(&[]).checked().is_ok());
        assert_eq!(config(&[]).backup_claim_secs, 900);
        let claimed = |secs: &str| config(&["--backup-claim-secs", secs]).checked();
        // Three times what one request to the store may take, as the node
        // is set up to make them, and the store is given to apply one the
        // node gave up.
        let request = StoreClient::default().transfers().request_within;
        let least = 3 * (request + vellev_node::LANDED_AFTER).as_secs();
        assert_eq!(
            least,
            10 * 60 + 51,
            "as the README and the release notes say"
        );
        assert_eq!(
            request,
            vellev_node::RequestBounds::default().store,
            "the one default the node and the store client share"
        );
        assert!(claimed(&least.to_string()).is_ok());
        let refused = format!(
            "{:#}",
            claimed(&(least - 1).to_string()).map(drop).unwrap_err()
        );
        assert!(refused.contains("VELLEV_BACKUP_CLAIM_SECS"), "{refused}");
        assert!(refused.contains("a third of it"), "{refused}");
        let too_long = [
            "vellev",
            "--database-url",
            "x",
            "--store",
            "memory://",
            "--backup-claim-secs",
            "86401",
        ];
        assert!(Parsed::try_parse_from(too_long).is_err());
    }

    #[test]
    fn a_store_is_named_by_its_url() {
        assert!(store("memory://", StoreClient::default()).is_ok());
        let dir = std::env::temp_dir().join(format!("vellev-store-{}", std::process::id()));
        assert!(store(&format!("file://{}", dir.display()), StoreClient::default()).is_ok());
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
            assert!(store(unknown, StoreClient::default()).is_err(), "{unknown}");
        }
        assert!(store("s3://bucket", StoreClient::default()).is_ok());
        assert!(store("s3://bucket/", StoreClient::default()).is_ok());
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
