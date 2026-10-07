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
}

#[derive(Subcommand)]
enum Import {
    /// Copy a Weaviate instance's classes into a vellev tenant, resuming
    /// where an import that stopped left off.
    Weaviate(Box<ImportWeaviate>),
}

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
    /// results.
    #[arg(long, default_value_t = 100)]
    page: usize,
    /// The most objects to copy a second.
    #[arg(long)]
    per_second: Option<f64>,
    /// How far Weaviate's clock may be behind this machine's, in seconds:
    /// a catch-up looks back this much further for changes.
    #[arg(long, default_value_t = 300)]
    skew_secs: u64,
    /// Where progress is kept; vellev-import-TENANT.json by default.
    #[arg(long)]
    state: Option<PathBuf>,
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
        Command::Import(Import::Weaviate(import)) => {
            let ImportWeaviate {
                from,
                from_key,
                from_tenant,
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
                per_second,
                skew_secs,
                state,
            } = *import;
            let source = vellev::import::Source {
                url: from,
                key: from_key,
                tenant: from_tenant,
            };
            let options = vellev::import::Options {
                classes,
                embeddings: embeddings.into_iter().collect(),
                vectors: vectors.into_iter().collect(),
                dimensions: dimensions.into_iter().collect(),
                created_as,
                updated_as,
                page,
                per_second,
                skew: Duration::from_secs(skew_secs),
                state: state
                    .unwrap_or_else(|| PathBuf::from(format!("vellev-import-{tenant}.json"))),
            };
            let client = vellev_client::Client::connect(&[&to], &key, Default::default())?;
            let report =
                vellev::import::weaviate(&source, &client.tenant(&tenant), &options).await?;
            for (class, copied) in &report.classes {
                for warning in &copied.warnings {
                    eprintln!("{class}: {warning}");
                }
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
            if report.matched() {
                return Ok(ExitCode::SUCCESS);
            }
            eprintln!("some classes hold a different number of objects in vellev than in Weaviate; run again to catch up");
            Ok(ExitCode::FAILURE)
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
