use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use eyre::Result;
use tokio::net::TcpListener;
use tokio::signal::unix::{signal, SignalKind};

#[derive(Parser)]
#[command(name = "vellev", version, about = "Vellev vector database")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a vellev node.
    Serve {
        #[arg(long, env = "VELLEV_LISTEN", default_value = "0.0.0.0:8080")]
        listen: SocketAddr,
    },
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
        Command::Serve { listen } => {
            tracing_subscriber::fmt().json().init();
            let listener = TcpListener::bind(listen).await?;
            tracing::info!(%listen, version = env!("CARGO_PKG_VERSION"), "vellev listening");
            vellev::server::serve(listener, shutdown_signal()).await?;
            tracing::info!("vellev stopped");
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

/// SIGTERM is what ECS and Swarm send on a rolling deploy; SIGINT is ctrl-c.
async fn shutdown_signal() {
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}
