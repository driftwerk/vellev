use std::net::SocketAddr;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use eyre::Result;
use object_store::ObjectStore;
use tokio::net::TcpListener;
use tokio::signal::unix::{signal, SignalKind};
use tokio::sync::Notify;
use vellev::config::Config;
use vellev::server::App;

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
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    match Cli::parse().command {
        Command::Serve(serve) => {
            let Serve { listen, config } = *serve;
            tracing_subscriber::fmt().json().init();
            let (limits, store) = config.checked()?;
            let app = App::new(config.keys()?, limits);
            let listener = TcpListener::bind(listen).await?;
            tracing::info!(%listen, version = env!("CARGO_PKG_VERSION"), "vellev listening");
            let starting = tokio::spawn(start(config.clone(), store, app.clone()));

            let told = Arc::new(Notify::new());
            let stop = told.clone();
            let served = vellev::server::serve(listener, app.clone(), async move {
                shutdown_signal().await;
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
async fn start(config: Config, store: Arc<dyn ObjectStore>, app: App) {
    let mut wait = Duration::from_secs(1);
    loop {
        match config.node(store.clone()).await {
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
