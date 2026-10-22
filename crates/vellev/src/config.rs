//! Building a node from what its operator configures.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use eyre::{bail, eyre, Result, WrapErr};
use object_store::aws::{AmazonS3Builder, S3ConditionalPut};
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
    /// The node's own local embedding model, which a collection may
    /// always name as the one it embeds with. A place is kept for it
    /// among those loaded, and it stays loaded. Empty, the node has none.
    #[arg(
        long,
        env = "VELLEV_LOCAL_MODEL",
        default_value = vellev_embed::DEFAULT_LOCAL_MODEL
    )]
    pub local_model: String,
    /// The other local embedding models collections may be made with,
    /// by name and separated by commas.
    #[arg(long, env = "VELLEV_LOCAL_MODELS_ALLOWED", value_delimiter = ',')]
    pub local_models_allowed: Vec<String>,
    /// The regions collections may embed through Bedrock in, separated
    /// by commas: by default the node's own, `AWS_REGION` or
    /// `AWS_DEFAULT_REGION`, if it has one. The node's calls to Bedrock
    /// are shared equally among them.
    #[arg(long, env = "VELLEV_BEDROCK_REGIONS", value_delimiter = ',')]
    pub bedrock_regions: Option<Vec<String>>,
    /// The most local embedding models loaded at once, the node's own
    /// among them. Each takes about a gigabyte while it runs.
    #[arg(long, env = "VELLEV_LOCAL_MODELS", default_value_t = vellev_embed::LOCAL_MODELS as u64, value_parser = clap::value_parser!(u64).range(1..=64))]
    pub local_models: u64,
    /// How long a local embedding model other than the node's own may go
    /// unused before it is let go, in seconds.
    #[arg(long, env = "VELLEV_LOCAL_MODEL_IDLE_SECS", default_value_t = vellev_embed::LOCAL_MODEL_IDLE.as_secs(), value_parser = clap::value_parser!(u64).range(1..))]
    pub local_model_idle_secs: u64,
    /// How long an embedding model is given to answer a search, in
    /// seconds; a search it has not answered by then is `unavailable`.
    /// Two minutes at the most, which is what it was before it was ten
    /// seconds: a search kept longer holds one of the node's reads for
    /// longer than a client waits.
    #[arg(long, env = "VELLEV_EMBED_QUERY_SECS", default_value_t = vellev_embed::QUERY_WITHIN.as_secs(), value_parser = clap::value_parser!(u64).range(1..=120))]
    pub embed_query_secs: u64,
    /// How long an embedding model may go without making a vector for
    /// anyone, while it is not loading, before a write that waits on it
    /// is `unavailable`, in seconds: not how long the write may take.
    #[arg(long, env = "VELLEV_EMBED_DOCUMENTS_SECS", default_value_t = vellev_embed::DOCUMENTS_WITHIN.as_secs(), value_parser = clap::value_parser!(u64).range(1..))]
    pub embed_documents_secs: u64,
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
    /// A tenant with more live rows with a vector than this written since
    /// its last checkpoint is flushed, at most once every
    /// `VELLEV_MEMTABLE_FLUSH_EVERY_SECS`: rows not yet flushed have no
    /// vector index, and a search scores each of them exactly.
    #[arg(long, env = "VELLEV_MEMTABLE_VECTORS", default_value_t = vellev_storage::SHORTLIST_ABOVE)]
    pub memtable_vectors: usize,
    #[arg(long, env = "VELLEV_MEMTABLE_FLUSH_EVERY_SECS", default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    pub memtable_flush_every_secs: u64,
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
    /// kept before they are deleted, in seconds: at least 900, fifteen
    /// minutes, and at least three times what one request to object
    /// storage may take with its tries, or the node does not start. It is
    /// all that keeps a file a restore within the window names from being
    /// deleted, where a file is written a little before it is committed
    /// (its upload, its copy to local disk and three tries to commit), a
    /// restore waits up to a minute for a clean-up to settle what it
    /// deletes, and the clocks of the node, the store and Postgres differ.
    /// The least covers a node set up near its defaults and no more: raise
    /// this with `VELLEV_LEASE_SECS` and `VELLEV_DATABASE_TIMEOUT_SECS`,
    /// and with both, as one alone may change nothing, to more than one
    /// request to the store plus three times the lesser of a quarter of
    /// the lease and the database timeout, plus some twenty seconds for
    /// the file's copy to local disk, with room to spare. Nor does it
    /// cover a head file of more than one part (`VELLEV_UPLOAD_PART_MIB`),
    /// a reader still fetching a segment of several parts under a short
    /// restore window of the tenant's own, a clock wrong by more than it,
    /// or a flush or merge that takes longer than it. Cleaning up by the
    /// commits Postgres still holds instead of by time is DW-681, with
    /// DW-668.
    #[arg(long, env = "VELLEV_GRACE_SECS", default_value_t = 60 * 60, value_parser = clap::value_parser!(u64).range(vellev_node::LEAST_GRACE.as_secs()..))]
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
    /// How long requests in flight are then given to finish, in seconds,
    /// and, once the node has let go of its tenants, as long again at the
    /// most for the aborts of uploads the stop cut off. Keep the delay and
    /// this under the platform's own grace period (ECS `stopTimeout`, Swarm
    /// `stop_grace_period`), so that the node lets go of its tenants before
    /// it is killed; the delay and twice this, so that it aborts what it
    /// cut off too.
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
/// then as `client` says. A file is created only once by `If-None-Match: *`
/// whatever `AWS_CONDITIONAL_PUT` says: not by a lock table of its own, nor
/// not at all.
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
    client
        .configured(from_env)
        .with_url(url)
        .with_conditional_put(S3ConditionalPut::ETagMatch)
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

/// The node's own region of Amazon's, as the AWS tools read it: from
/// `AWS_REGION`, or `AWS_DEFAULT_REGION` if that is unset or blank. Not
/// from a profile, nor from the machine the node runs on.
fn own_region(aws_region: Option<String>, aws_default_region: Option<String>) -> Option<String> {
    [aws_region, aws_default_region]
        .into_iter()
        .flatten()
        .map(|region| region.trim().to_string())
        .find(|region| !region.is_empty())
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

    /// How the node embeds, which is refused if it could not embed so.
    fn embedding(&self) -> Result<vellev_embed::Settings> {
        let var = |name| std::env::var(name).ok();
        self.embedding_in(own_region(var("AWS_REGION"), var("AWS_DEFAULT_REGION")))
    }

    /// How a node whose own region of Amazon's is `own` embeds.
    fn embedding_in(&self, own: Option<String>) -> Result<vellev_embed::Settings> {
        let named = |names: &[String]| -> Vec<String> {
            let names = names.iter().map(|name| name.trim());
            names
                .filter(|name| !name.is_empty())
                .map(String::from)
                .collect()
        };
        let settings = vellev_embed::Settings {
            cache: self.model_dir.clone(),
            keys: self.embedding_keys()?,
            local_model: named(std::slice::from_ref(&self.local_model)).pop(),
            local_models_allowed: named(&self.local_models_allowed),
            bedrock_regions: match &self.bedrock_regions {
                Some(regions) => named(regions),
                None => named(&own.into_iter().collect::<Vec<_>>()),
            },
            local_models: usize::try_from(self.local_models).unwrap_or(usize::MAX),
            local_model_idle: Duration::from_secs(self.local_model_idle_secs),
            query_within: Duration::from_secs(self.embed_query_secs),
            documents_within: Duration::from_secs(self.embed_documents_secs),
            ..Default::default()
        };
        settings.checked().map_err(|wrong| eyre!(wrong))?;
        Ok(settings)
    }

    /// What callers may ask of the node, and the store it keeps data in.
    /// Everything that can be wrong without reaching another machine is
    /// found wrong here.
    pub fn checked(&self) -> Result<(Limits, ObjectBlobs)> {
        let keyed = self.embedding_keys()?;
        self.embedding()?;
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
        let embedders = Embedders::new(self.embedding()?);
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
            memtable_vectors: self.memtable_vectors,
            memtable_flush_every: Duration::from_secs(self.memtable_flush_every_secs),
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
        if let Some(problem) = settings.request_bounds.grace_problem(settings.grace) {
            bail!("VELLEV_GRACE_SECS cannot be as it is set: {problem}");
        }
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
    fn a_node_embeds_as_it_is_told_and_runs_its_own_local_model_alone_unless_told() {
        let seconds = Duration::from_secs;
        let untold = config(&[]).embedding_in(None).unwrap();
        assert_eq!(untold.cache, PathBuf::from("/var/lib/vellev/models"));
        assert!(untold.keys.is_empty());
        assert_eq!(untold.local_model.as_deref(), Some("BAAI/bge-base-en-v1.5"));
        assert!(untold.local_models_allowed.is_empty());
        assert!(untold.bedrock_regions.is_empty());
        assert_eq!(
            (untold.local_models, untold.local_model_idle),
            (2, seconds(900))
        );
        assert_eq!(untold.query_within, seconds(10));
        assert_eq!(untold.documents_within, seconds(60));
        assert_eq!(untold.fault_stands, vellev_embed::FAULT_STANDS);

        let told = config(&[
            "--model-dir",
            "/models",
            "--embedding-keys",
            "http://tei:80=t0123456789abcdef",
            "--local-model",
            "bge-small-en-v1.5",
            "--local-models-allowed",
            "BAAI/bge-large-en-v1.5, bge-base-en-v1.5,,",
            "--bedrock-regions",
            "eu-west-2, us-east-1,",
            "--local-models",
            "3",
            "--local-model-idle-secs",
            "60",
            "--embed-query-secs",
            "4",
            "--embed-documents-secs",
            "120",
        ])
        .embedding_in(Some("ap-south-1".into()))
        .unwrap();
        assert_eq!(told.cache, PathBuf::from("/models"));
        assert_eq!(told.keys["http://tei:80"], "t0123456789abcdef");
        assert_eq!(told.local_model.as_deref(), Some("bge-small-en-v1.5"));
        assert_eq!(
            told.local_models_allowed,
            ["BAAI/bge-large-en-v1.5", "bge-base-en-v1.5"]
        );
        assert_eq!(told.bedrock_regions, ["eu-west-2", "us-east-1"]);
        assert_eq!((told.local_models, told.local_model_idle), (3, seconds(60)));
        assert_eq!(told.query_within, seconds(4));
        assert_eq!(told.documents_within, seconds(120));
        assert!(config(&["--embedding-keys", "no-token"])
            .embedding_in(None)
            .is_err());

        // The node's own region is read as the AWS tools read it: a blank
        // `AWS_REGION` falls through to `AWS_DEFAULT_REGION`.
        let some = |region: &str| Some(region.to_string());
        assert_eq!(
            own_region(some("eu-west-2"), some("us-east-1")),
            some("eu-west-2")
        );
        assert_eq!(own_region(some(" "), some("us-east-1")), some("us-east-1"));
        assert_eq!(own_region(None, some(" us-east-1 ")), some("us-east-1"));
        assert_eq!(own_region(some(""), some("")), None);
        assert_eq!(own_region(None, None), None);
        // Told of no regions, a node embeds through Bedrock in its own,
        // if it has one; and told of none in so many words, in none.
        let own = config(&[]).embedding_in(Some("eu-west-2".into())).unwrap();
        assert_eq!(own.bedrock_regions, ["eu-west-2"]);
        let none = config(&["--bedrock-regions", ""]).embedding_in(Some("eu-west-2".into()));
        assert!(none.unwrap().bedrock_regions.is_empty());
        // Told to have no model of its own, a node has none.
        let none = config(&["--local-model", ""]).embedding_in(None).unwrap();
        assert_eq!(none.local_model, None);

        // A node cannot be told to hold no model, or to give no time.
        for nothing in [
            "--local-models",
            "--local-model-idle-secs",
            "--embed-query-secs",
            "--embed-documents-secs",
        ] {
            let required = ["vellev", "--database-url", "x", "--store", "memory://"];
            let asked = required.into_iter().chain([nothing, "0"]);
            assert!(Parsed::try_parse_from(asked).is_err(), "{nothing}");
        }
        // Nor to keep a search waiting for longer than two minutes.
        let required = ["vellev", "--database-url", "x", "--store", "memory://"];
        let asked = |secs| required.into_iter().chain(["--embed-query-secs", secs]);
        assert!(Parsed::try_parse_from(asked("120")).is_ok());
        assert!(Parsed::try_parse_from(asked("121")).is_err());
    }

    /// What a node could not do as it is told is refused when it starts,
    /// and not found out by the collections made on it.
    #[test]
    fn a_node_told_to_run_local_models_it_cannot_does_not_start() {
        let wrong = |told: &[&str]| {
            let refused = config(told).embedding_in(None).unwrap_err();
            refused.to_string()
        };
        // It is found with whatever else is wrong with how a node is
        // told to run, before anything is reached.
        assert!(config(&["--local-model", "BAAI/bge-base-en-v15"])
            .checked()
            .is_err());
        assert!(config(&[]).checked().is_ok());
        // A slip in a name: its own model's, or another's.
        assert_eq!(
            wrong(&["--local-model", "BAAI/bge-base-en-v15"]),
            "BAAI/bge-base-en-v15 is not a local model this release runs"
        );
        assert_eq!(
            wrong(&[
                "--local-models-allowed",
                "bge-small-en-v1.5,acme/unheard-of"
            ]),
            "acme/unheard-of is not a local model this release runs"
        );
        // More models than could ever be loaded.
        assert_eq!(
            wrong(&[
                "--local-models",
                "1",
                "--local-models-allowed",
                "bge-small-en-v1.5"
            ]),
            "a node with room for one local model has room for its own alone, \
             BAAI/bge-base-en-v1.5: it cannot also run bge-small-en-v1.5"
        );
        let two = [
            "--local-models",
            "2",
            "--local-models-allowed",
            "bge-small-en-v1.5",
        ];
        assert!(config(&two).embedding_in(None).is_ok());
    }

    #[test]
    fn an_embedding_key_belongs_to_the_endpoint_it_is_given_for() {
        let keyed = config(&[
            "--embedding-keys",
            "https://api.example.test/v1/=sk-a=b0123456789abcdef,http://tei:80=t0123456789abcdef",
        ]);
        let keys = keyed.embedding_keys().unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(
            keys["https://api.example.test/v1"],
            "sk-a=b0123456789abcdef"
        );
        assert_eq!(keys["http://tei:80"], "t0123456789abcdef");
        // A key too short to be one: the node does not start.
        let short = config(&["--embedding-keys", "http://tei:80=t"]);
        let refused = short.checked().unwrap_err().to_string();
        assert!(
            refused.contains("the key for http://tei:80 is shorter"),
            "{refused}"
        );
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

    /// A stand-in for S3 that answers its requests with `answers` in turn,
    /// and every one after them with the last, counting them.
    async fn answering(
        answers: Vec<axum::http::StatusCode>,
    ) -> (ObjectBlobs, Arc<std::sync::atomic::AtomicUsize>) {
        use std::sync::atomic::Ordering;
        let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counting = asked.clone();
        let router = axum::Router::new().fallback(move || {
            let turn = counting.fetch_add(1, Ordering::SeqCst);
            let answer = answers[turn.min(answers.len() - 1)];
            async move { answer }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
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
        (ObjectBlobs::new(Arc::new(store)), asked)
    }

    #[tokio::test]
    async fn a_create_that_landed_and_was_answered_unavailable_is_tried_again_and_found_there() {
        use axum::http::StatusCode;
        use vellev_storage::blobs::Created;
        // The create lands and its answer is lost to a 503; the client
        // tries it again, and the store refuses it, the file being there.
        let answers = vec![
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::PRECONDITION_FAILED,
        ];
        let (blobs, asked) = answering(answers).await;
        let created = blobs.put_new("link", bytes::Bytes::from_static(b"A")).await;
        assert_eq!(created.unwrap(), Created::AlreadyThere);
        assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_create_refused_while_another_is_under_way_is_taken_as_maybe_there() {
        use axum::http::StatusCode;
        use vellev_storage::blobs::Created;
        // S3 answers 409 while another create of the key is under way,
        // which may yet fail: nothing need be there.
        let (blobs, asked) = answering(vec![StatusCode::CONFLICT]).await;
        let created = blobs.put_new("link", bytes::Bytes::from_static(b"A")).await;
        assert_eq!(created.unwrap(), Created::AlreadyThere);
        assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn an_s3_store_creates_only_with_if_none_match_whatever_its_environment_says() {
        for given in ["disabled", "dynamo:locks", "etag"] {
            let env = [("AWS_CONDITIONAL_PUT", given), ("AWS_REGION", "eu-west-1")]
                .map(|(key, value)| (key.to_string(), value.to_string()));
            let builder = s3("s3://bucket", StoreClient::default(), env);
            let conditional = builder.get_config_value(&AmazonS3ConfigKey::ConditionalPut);
            assert_eq!(conditional.as_deref(), Some("etag"), "{given}");
        }
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
    fn a_grace_below_fifteen_minutes_or_three_requests_to_the_store_is_refused() {
        assert_eq!(config(&[]).grace_secs, 60 * 60);
        assert!(config(&[]).checked().is_ok());
        // Fifteen minutes at the least, which is more than three requests
        // to the store as a node is set up to make them: refused as soon
        // as it is read.
        let least = vellev_node::RequestBounds::default()
            .least_grace()
            .as_secs();
        assert_eq!(least, 15 * 60, "as the README and the release notes say");
        assert!(config(&["--grace-secs", &least.to_string()])
            .checked()
            .is_ok());
        let parsed = |secs: u64| {
            let (secs, required) = (secs.to_string(), ["vellev", "--database-url", "x"]);
            let rest = ["--store", "memory://", "--grace-secs", &secs];
            Parsed::try_parse_from(required.iter().chain(&rest)).map(drop)
        };
        assert!(parsed(least).is_ok());
        let refused = parsed(least - 1).unwrap_err().to_string();
        assert!(refused.contains("--grace-secs"), "{refused}");
        assert!(refused.contains("899 is not in 900.."), "{refused}");
        assert!(parsed(0).is_err());

        // A store given longer to answer in takes longer over a request,
        // and three of them are then the least: with a backup claim long
        // enough for that store, so that nothing but the grace is refused.
        let graced = |secs: u64| {
            let (secs, slow) = (secs.to_string(), ["--store-timeout-secs", "120"]);
            let rest = ["--backup-claim-secs", "3600", "--grace-secs", &secs];
            let config = config(&[slow.as_slice(), rest.as_slice()].concat());
            let request = config.store_client().transfers().request_within;
            (config.checked().map(drop), request)
        };
        let (checked, request) = graced(60 * 60);
        assert!(checked.is_ok());
        let least = 3 * request.as_secs();
        assert_eq!(least, 27 * 60 + 21);
        assert!(graced(least).0.is_ok());
        let refused = format!("{:#}", graced(least - 1).0.unwrap_err());
        assert!(refused.contains("VELLEV_GRACE_SECS"), "{refused}");
        assert!(refused.contains("at least 1641s"), "{refused}");
        assert!(!refused.contains("VELLEV_BACKUP_CLAIM_SECS"), "{refused}");
        let refused = format!("{:#}", graced(15 * 60).0.unwrap_err());
        assert!(refused.contains("VELLEV_GRACE_SECS"), "{refused}");
        // And a backup claim too short is still said to be that.
        let claimed = config(&["--backup-claim-secs", "600"]).checked();
        let refused = format!("{:#}", claimed.map(drop).unwrap_err());
        assert!(refused.contains("VELLEV_BACKUP_CLAIM_SECS"), "{refused}");
        assert!(!refused.contains("VELLEV_GRACE_SECS"), "{refused}");
        // With both wrong it is the grace that is spoken of, as by
        // `Settings::check`.
        let slow = ["--store-timeout-secs", "120", "--grace-secs", "900"];
        let both = config(&slow).settings(NodeId::generate("node"), None);
        let refused = format!("{:#}", both.map(drop).unwrap_err());
        assert!(refused.contains("VELLEV_GRACE_SECS"), "{refused}");
        assert!(!refused.contains("VELLEV_BACKUP_CLAIM_SECS"), "{refused}");
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
