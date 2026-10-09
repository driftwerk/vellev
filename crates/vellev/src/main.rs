use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use eyre::{Result, WrapErr};
use tokio::net::TcpListener;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Notify;
use vellev::config::Config;
use vellev::server::App;
use vellev_api::{Adopt, RestoreTo};
use vellev_core::{Embedding, TenantId};
use vellev_storage::blobs::ObjectBlobs;

#[derive(Parser)]
#[command(name = "vellev", version, about = "Vellev vector database")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args)]
struct Serve {
    #[arg(long, env = "VELLEV_LISTEN", default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    #[command(flatten)]
    config: Config,
}

#[derive(Subcommand)]
enum Command {
    /// Run a vellev node.
    Serve(Box<Serve>),
    /// Print the API's OpenAPI description.
    Openapi,
    /// Check that a node is live; exits non-zero when it is not. Container
    /// health checks run this, so the image needs no curl or shell.
    Health {
        #[arg(
            long,
            env = "VELLEV_HEALTH_URL",
            default_value = "http://127.0.0.1:8080"
        )]
        url: String,
        #[arg(long, default_value_t = 2)]
        timeout_secs: u64,
    },
    /// Copy data from elsewhere into a tenant.
    #[command(subcommand)]
    Import(Import),
    /// Back a tenant up, and put it back as it was; needs an admin key.
    Backup(Box<Backup>),
}

#[derive(clap::Args)]
struct Backup {
    /// A vellev node of the cluster.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    node: String,
    /// An admin key.
    #[arg(long, env = "VELLEV_API_KEY")]
    key: String,
    #[arg(long, value_parser = |id: &str| TenantId::new(id))]
    tenant: TenantId,
    #[command(subcommand)]
    action: BackupAction,
}

#[derive(Subcommand)]
enum BackupAction {
    /// Back the tenant up now.
    Take,
    /// List the tenant's backups.
    List,
    /// Show how far back the tenant may be restored.
    Points,
    /// Put the tenant back as it stood at a moment, a commit, or a backup.
    #[command(group(clap::ArgGroup::new("to").required(true)))]
    Restore {
        /// A moment in its restore window, as RFC 3339.
        #[arg(long, group = "to")]
        at: Option<chrono::DateTime<chrono::Utc>>,
        /// The position of a commit in its restore window.
        #[arg(long, group = "to")]
        position: Option<u64>,
        /// The id of one of its backups.
        #[arg(long, group = "to")]
        backup: Option<uuid::Uuid>,
        /// The id of a backup kept under its name, to take up again a
        /// tenant this cluster has lost: the tenant is made, with the
        /// restore window it had.
        #[arg(long, group = "to")]
        adopt: Option<uuid::Uuid>,
        /// Give the tenant adopted the keys the backup kept: those it had
        /// when the backup was taken, one revoked since among them.
        #[arg(long, requires = "adopt")]
        with_keys: bool,
    },
    /// Set how far back the tenant may be restored.
    Window {
        /// Seconds; with none, the cluster's default.
        secs: Option<u64>,
    },
}

/// How long a backup or a restore through the command line may take.
const BACKUP_TIMEOUT: Duration = Duration::from_secs(60 * 60);

#[derive(Subcommand)]
enum Import {
    /// Copy a Weaviate instance's classes into a vellev tenant, resuming
    /// where an import that stopped left off.
    Weaviate(Box<ImportWeaviate>),
    /// End the import of a tenant's classes, once nothing more is to be
    /// brought from Weaviate: drop each ledger, which frees what it takes
    /// on the node, and note that the collection is no longer imported,
    /// so that no later run changes it.
    Finalise {
        /// The vellev node.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        to: String,
        /// The vellev API key.
        #[arg(long, env = "VELLEV_API_KEY")]
        key: String,
        #[arg(long, value_parser = |id: &str| TenantId::new(id))]
        tenant: TenantId,
        /// A class to end the import of; every imported class when none
        /// is named.
        #[arg(long = "class")]
        classes: Vec<String>,
    },
}

/// What `import weaviate` exits with when Weaviate was written to behind
/// the catch-up, and another run is all it takes.
const BEHIND: u8 = 3;

#[derive(clap::Args)]
struct ImportWeaviate {
    /// The Weaviate to copy from.
    #[arg(long)]
    from: String,
    /// The Weaviate API key, if it needs one.
    #[arg(long, env = "VELLEV_IMPORT_FROM_KEY")]
    from_key: Option<String>,
    /// The Weaviate tenant to copy, for classes with multi-tenancy on.
    #[arg(long)]
    from_tenant: Option<String>,
    /// What to call this Weaviate, where its address changes between runs:
    /// an import goes on only from the Weaviate it began from, which it
    /// knows by this name, or by its address when none is given.
    #[arg(long)]
    from_name: Option<String>,
    /// The vellev node to copy into.
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    to: String,
    /// The vellev API key.
    #[arg(long, env = "VELLEV_API_KEY")]
    key: String,
    #[arg(long, value_parser = |id: &str| TenantId::new(id))]
    tenant: TenantId,
    /// A class to copy; every class when none is named.
    #[arg(long = "class")]
    classes: Vec<String>,
    /// CLASS=FILE: embed the class in vellev as the embedding in FILE says,
    /// rather than copy its vectors.
    #[arg(long = "embedding", value_parser = class_embedding)]
    embeddings: Vec<(String, Embedding)>,
    /// CLASS=NAME: copy the class's named vector NAME, where it has several.
    #[arg(long = "vector", value_parser = class_vector)]
    vectors: Vec<(String, String)>,
    /// CLASS=N: the dimensions of the class's vectors, where its first
    /// objects have none to tell by.
    #[arg(long = "dimensions", value_parser = class_dimensions)]
    dimensions: Vec<(String, u32)>,
    /// A date property to hold each object's creation time in Weaviate.
    #[arg(long)]
    created_as: Option<String>,
    /// A date property to hold the time each object last changed.
    #[arg(long)]
    updated_as: Option<String>,
    /// How many objects to copy at a time; at most the node's limit on
    /// results, which it is unless given, or 1000 where the node does not
    /// say.
    #[arg(long)]
    page: Option<usize>,
    /// The most bytes of objects to send the node in one request, near
    /// enough; under its limit on a request's body.
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    batch_bytes: usize,
    /// The most objects to ask of Weaviate a second.
    #[arg(long)]
    per_second: Option<f64>,
    /// How far Weaviate's clock may be behind this machine's, in seconds:
    /// a catch-up looks back this much further for changes.
    #[arg(long, default_value_t = 300)]
    skew_secs: u64,
    /// Where progress is kept; vellev-import-TENANT.json by default.
    #[arg(long)]
    state: Option<PathBuf>,
    /// Delete nothing: what Weaviate no longer holds is only counted, as
    /// pending.
    #[arg(long)]
    no_delete: bool,
    /// The most objects of a class a run may delete. A run that finds
    /// more gone from Weaviate deletes none of them and fails, since a
    /// Weaviate emptied, restored or mistaken for another looks the same.
    #[arg(long, default_value_t = 1_000)]
    max_deletes: u64,
    /// The most of a class a run may delete, as a part of what was
    /// imported of it, between 0 and 1; the fewer of this and
    /// --max-deletes holds.
    #[arg(long, default_value_t = 0.05)]
    max_delete_fraction: f64,
    /// How many seconds after Weaviate first says it holds no such object
    /// it is asked again, before the object is deleted.
    #[arg(long, default_value_t = 5)]
    confirm_secs: u64,
    /// How many seconds a request Weaviate fails or turns away is asked
    /// again before the run stops: as long as a restart of it may take.
    #[arg(long, default_value_t = 600)]
    patience_secs: u64,
    /// Say that nothing but an import has written to the collections
    /// found without a ledger, such as those an earlier release imported:
    /// what they lack is stored and what differs from Weaviate's is
    /// replaced, where otherwise both are left alone.
    #[arg(long)]
    sole_writer: bool,
    /// Say that a class Weaviate has made again since it was imported is
    /// the one to go on from.
    #[arg(long)]
    accept_remade: bool,
    /// Pass over an object vellev cannot hold and report it, where
    /// otherwise the run stops at it.
    #[arg(long)]
    skip_refused: bool,
    /// Fail too when a collection holds anything that is not as the
    /// import left it: another writer's objects, or its deletions.
    #[arg(long)]
    strict: bool,
    /// Change nothing, and compare what each collection holds with what
    /// Weaviate does, object by object: over this many pages begun at
    /// ids picked at random, 20 unless given.
    #[arg(long, num_args = 0..=1, default_missing_value = "20")]
    verify: Option<usize>,
    /// As --verify, over every object.
    #[arg(long, conflicts_with = "verify")]
    verify_all: bool,
}

/// A class's name and the embedding in the file named after `=`.
fn class_embedding(given: &str) -> Result<(String, Embedding)> {
    let (class, path) = given
        .split_once('=')
        .ok_or_else(|| eyre::eyre!("expected CLASS=FILE, got {given}"))?;
    let read = || -> Result<Embedding> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) };
    let embedding = read().wrap_err_with(|| format!("reading the embedding in {path}"))?;
    Ok((class.to_string(), embedding))
}

/// A class's name and the number after `=`.
fn class_dimensions(given: &str) -> Result<(String, u32)> {
    let (class, dimensions) = class_vector(given)?;
    Ok((class, dimensions.parse()?))
}

/// A class's name and the name after `=`.
fn class_vector(given: &str) -> Result<(String, String)> {
    let (class, vector) = given
        .split_once('=')
        .ok_or_else(|| eyre::eyre!("expected CLASS=NAME, got {given}"))?;
    Ok((class.to_string(), vector.to_string()))
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    match Cli::parse().command {
        Command::Serve(serve) => {
            let Serve { listen, config } = *serve;
            tracing_subscriber::fmt().json().init();
            let (limits, store) = config.checked()?;
            let trusted = vellev::identity::IdentityKeys::parse(&config.identity_keys)?;
            let app = App::new(config.keys()?, limits).trusting(trusted);
            let listener = TcpListener::bind(listen).await?;
            tracing::info!(%listen, version = env!("CARGO_PKG_VERSION"), "vellev listening");
            let port = listener.local_addr()?.port();
            let starting = tokio::spawn(start(config.clone(), store, app.clone(), port));

            let told = Arc::new(Notify::new());
            let stop = told.clone();
            let (stopping, delay) = (app.clone(), Duration::from_secs(config.stop_delay_secs));
            let served = vellev::server::serve(listener, app.clone(), async move {
                shutdown_signal().await;
                stopping.stopping();
                tracing::info!(?delay, "stopping: not ready, still serving");
                tokio::time::sleep(delay).await;
                stop.notify_one();
            });
            let drain = Duration::from_secs(config.drain_secs);
            tokio::select! {
                served = served => served?,
                () = async { told.notified().await; tokio::time::sleep(drain).await } => {
                    tracing::warn!(?drain, "stopping with requests still in flight");
                }
            }
            starting.abort();
            if let Some(node) = app.started() {
                node.shutdown().await;
            }
            tracing::info!("vellev stopped");
            Ok(ExitCode::SUCCESS)
        }
        Command::Openapi => {
            println!("{}", vellev::server::openapi().to_pretty_json()?);
            Ok(ExitCode::SUCCESS)
        }
        Command::Backup(backup) => {
            let Backup {
                node,
                key,
                tenant,
                action,
            } = *backup;
            let options = vellev_client::Options::default().timeout(BACKUP_TIMEOUT);
            let client = vellev_client::Client::connect(&[&node], &key, options)?;
            let tenant = client.tenant(&tenant);
            let answer = match action {
                BackupAction::Take => serde_json::to_value(tenant.back_up().await?)?,
                BackupAction::List => serde_json::to_value(tenant.backups().await?)?,
                BackupAction::Points => serde_json::to_value(tenant.restore_points().await?)?,
                BackupAction::Restore {
                    at,
                    position,
                    backup,
                    adopt,
                    with_keys,
                } => {
                    let adopt = adopt.map(|backup| Adopt { backup, with_keys });
                    let to = at
                        .map(RestoreTo::At)
                        .or(position.map(RestoreTo::Position))
                        .or(backup.map(RestoreTo::Backup))
                        .or(adopt.map(RestoreTo::Adopt))
                        .ok_or_else(|| {
                            eyre::eyre!("restore to --at, --position, --backup or --adopt")
                        })?;
                    serde_json::to_value(tenant.restore(to).await?)?
                }
                BackupAction::Window { secs } => {
                    tenant.set_restore_window(secs).await?;
                    serde_json::to_value(tenant.restore_points().await?)?
                }
            };
            println!("{}", serde_json::to_string_pretty(&answer)?);
            Ok(ExitCode::SUCCESS)
        }
        Command::Import(Import::Weaviate(import)) => {
            let ImportWeaviate {
                from,
                from_key,
                from_tenant,
                from_name,
                to,
                key,
                tenant,
                classes,
                embeddings,
                vectors,
                dimensions,
                created_as,
                updated_as,
                page,
                batch_bytes,
                per_second,
                skew_secs,
                state,
                no_delete,
                max_deletes,
                max_delete_fraction,
                confirm_secs,
                patience_secs,
                sole_writer,
                accept_remade,
                skip_refused,
                strict,
                verify,
                verify_all,
            } = *import;
            let source = vellev::import::Source {
                url: from,
                key: from_key,
                tenant: from_tenant,
                name: from_name,
            };
            let client = vellev_client::Client::connect(&[&to], &key, Default::default())?;
            let page = match page {
                Some(page) => page,
                None => {
                    let results = client
                        .limits()
                        .await?
                        .and_then(|limits| limits.max(vellev_client::api::Bounded::Results));
                    results.map_or(1_000, |results| results as usize)
                }
            };
            let state =
                state.unwrap_or_else(|| PathBuf::from(format!("vellev-import-{tenant}.json")));
            let deletes = match no_delete {
                true => vellev::import::Deletes::None,
                false => vellev::import::Deletes::UpTo {
                    count: max_deletes,
                    fraction: max_delete_fraction,
                },
            };
            let mode = match (verify, verify_all) {
                (None, false) => vellev::import::Mode::Import,
                (pages, _) => vellev::import::Mode::Verify { pages },
            };
            let options = vellev::import::Options {
                classes,
                embeddings: embeddings.into_iter().collect(),
                vectors: vectors.into_iter().collect(),
                dimensions: dimensions.into_iter().collect(),
                created_as,
                updated_as,
                page,
                batch_bytes,
                per_second,
                skew: Duration::from_secs(skew_secs),
                deletes,
                confirm: Duration::from_secs(confirm_secs),
                patience: Duration::from_secs(patience_secs),
                sole_writer,
                accept_remade,
                skip_refused,
                mode,
                ..vellev::import::Options::new(state)
            };
            let report =
                vellev::import::weaviate(&source, &client.tenant(&tenant), &options).await?;
            for (class, copied) in &report.classes {
                for warning in &copied.warnings {
                    eprintln!("{class}: {warning}");
                }
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            let classes = |of: &dyn Fn(&vellev::import::Copied) -> bool| {
                let named = report.classes.iter().filter(|(_, copied)| of(copied));
                let named: Vec<&str> = named.map(|(class, _)| class.as_str()).collect();
                (!named.is_empty()).then(|| named.join(", "))
            };
            for class in &report.ended {
                eprintln!("{class}: its import was ended; nothing was done");
            }
            if let Some(withheld) = classes(&|copied| copied.withheld > 0) {
                eprintln!("{withheld}: some of Weaviate's objects were NOT copied: see `withheld`. The collection was there before the import kept a ledger of it, as one an earlier release imported is, and held nothing, or something else, under their ids; nothing tells an object Weaviate gained or changed since from one another writer deleted or changed in vellev. If nothing but an import has written to the collection, run again with --sole-writer, which stores and replaces them, and deletes what Weaviate no longer holds; otherwise they stay as they are");
            }
            if let Some(over) = classes(&|copied| copied.budget.is_some()) {
                eprintln!("{over}: more objects are gone from Weaviate than a run may delete, so none was deleted: see `pending` and `budget`. A Weaviate that was emptied, restored or is not the one imported looks like this. If they are truly gone, raise --max-deletes and --max-delete-fraction for one run");
            }
            if let Some(pending) = classes(&|copied| copied.pending > 0 && copied.budget.is_none())
            {
                eprintln!("{pending}: objects gone from Weaviate were not deleted, as asked: see `pending`");
            }
            if let Some(refused) = classes(&|copied| !copied.refused.is_empty()) {
                eprintln!(
                    "{refused}: some objects vellev cannot hold were passed over: see `refused`"
                );
            }
            if let Some(large) = classes(&|copied| !copied.too_large.is_empty()) {
                eprintln!("{large}: some objects changed in Weaviate are too large to replace safely in one request, and were left as they are: see `too_large`");
            }
            if let Some(wrong) = classes(&|copied| {
                let verified = copied.verified.as_ref();
                verified.is_some_and(|verified| verified.different + verified.missing > 0)
            }) {
                eprintln!("{wrong}: the collection does not hold some of the import's objects as Weaviate does: see `verified`");
            }
            if let Some(unseen) = classes(&|copied| {
                let verified = copied.verified.as_ref();
                verified.is_some_and(|verified| verified.looked == 0) && copied.weaviate > 0
            }) {
                eprintln!("{unseen}: nothing was compared, though Weaviate holds objects: the ledger has none of them");
            }
            if let Some(behind) = classes(&|copied| !copied.caught_up()) {
                eprintln!("{behind}: Weaviate holds a different number of objects than the import has accounted for; it was written to during the run; run again to catch up");
            }
            if let Some(touched) = classes(&|copied| !copied.untouched()) {
                eprintln!("{touched}: the collection holds objects that are not as the import left them, written, changed or deleted by something else; they were left as they are, and running again will not change that");
            }
            Ok(match report.outcome(strict) {
                vellev::import::Outcome::Settled => ExitCode::SUCCESS,
                vellev::import::Outcome::Behind => ExitCode::from(BEHIND),
                vellev::import::Outcome::Unsettled => ExitCode::FAILURE,
            })
        }
        Command::Import(Import::Finalise {
            to,
            key,
            tenant,
            classes,
        }) => {
            let client = vellev_client::Client::connect(&[&to], &key, Default::default())?;
            let ended = vellev::import::finalise(&client.tenant(&tenant), &classes).await?;
            println!("{}", serde_json::to_string_pretty(&ended)?);
            Ok(ExitCode::SUCCESS)
        }
        Command::Health { url, timeout_secs } => {
            let client = vellev_client::Client::new(&url, Duration::from_secs(timeout_secs))?;
            match client.live().await {
                Ok(()) => Ok(ExitCode::SUCCESS),
                Err(err) => {
                    eprintln!("{err}");
                    Ok(ExitCode::FAILURE)
                }
            }
        }
    }
}

/// Reach Postgres, however long it takes, and then serve from the node.
/// A node that gave up would be restarted into the same outage.
async fn start(config: Config, store: ObjectBlobs, app: App, port: u16) {
    let mut wait = Duration::from_secs(1);
    loop {
        match config.node(store.clone(), port).await {
            Ok(node) => {
                app.start(Arc::new(node));
                tracing::info!("vellev ready");
                return;
            }
            Err(error) => tracing::warn!(error = %format!("{error:#}"), ?wait, "not ready yet"),
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(Duration::from_secs(10));
    }
}

/// SIGTERM is what ECS and Swarm send on a rolling deploy; SIGINT is ctrl-c.
async fn shutdown_signal() {
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}
