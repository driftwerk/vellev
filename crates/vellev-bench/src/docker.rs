//! Bringing each engine up and down in Docker, pinned as the design says
//! (§2): the engine on CPUs 0-2, with three CPUs and 10 GiB of memory, and
//! only one engine up at a time. vellev needs Postgres for its commit
//! pointer, so Postgres runs under the same CPUs and inside the same 10 GiB,
//! and is taken down with vellev.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use eyre::{bail, ensure, eyre, Result, WrapErr};

pub const NETWORK: &str = "vellev-bench";
pub const VELLEV_KEY: &str = "bench-key";
pub const VELLEV_PORT: u16 = 18080;
pub const VELLEV_METRICS_PORT: u16 = 19464;
pub const WEAVIATE_PORT: u16 = 18081;

/// How an engine is pinned.
#[derive(Debug, Clone)]
pub struct Pinning {
    pub cpuset: String,
    pub cpus: u32,
    pub memory_mb: u64,
}

/// Postgres's share of vellev's memory.
const POSTGRES_MB: u64 = 512;

fn docker(args: &[&str]) -> Result<String> {
    let output = Command::new("docker")
        .args(args)
        .output()
        .wrap_err("docker")?;
    ensure!(
        output.status.success(),
        "docker {}: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Remove `name`, whether or not it is there.
fn removed(name: &str) {
    let _ = Command::new("docker")
        .args(["rm", "-f", "-v", name])
        .output();
}

/// The image files are cleared with: its own `rm`, run as root, since what
/// an engine wrote is owned by the user the engine ran as.
const CLEARER: &str =
    "postgres:16-alpine@sha256:0cefaabd882e42bcb2a2f3dc56d3ce4770560ba62fd7f197f1a2e008fd5731ab";

/// Remove `dir` and everything in it, whoever wrote it.
pub fn cleared(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let mount = format!("{}:/cleared", dir.display());
    docker(&[
        "run",
        "--rm",
        "--user",
        "0",
        "--entrypoint",
        "sh",
        "-v",
        &mount,
        CLEARER,
        "-c",
        "rm -rf /cleared/* /cleared/.[!.]* /cleared/..?* 2>/dev/null; true",
    ])?;
    std::fs::remove_dir_all(dir).wrap_err_with(|| dir.display().to_string())
}

/// A directory the engine's containers write to, made afresh and open to
/// the user the image runs as.
fn fresh(dir: &Path) -> Result<()> {
    cleared(dir)?;
    std::fs::create_dir_all(dir)?;
    let status = Command::new("chmod").args(["0777"]).arg(dir).status()?;
    ensure!(status.success(), "chmod {}", dir.display());
    Ok(())
}

/// The image's digest as Docker resolved it.
pub fn digest(image: &str) -> Result<String> {
    let digests = docker(&[
        "image",
        "inspect",
        "-f",
        "{{json .RepoDigests}} {{.Id}}",
        image,
    ])?;
    Ok(digests)
}

pub fn pull(image: &str) -> Result<()> {
    docker(&["pull", "-q", image]).map(|_| ())
}

/// A container the engine runs in.
#[derive(Debug, Clone)]
pub struct Container {
    pub name: String,
}

impl Container {
    pub fn id(&self) -> Result<String> {
        docker(&["inspect", "-f", "{{.Id}}", &self.name])
    }

    pub fn restart(&self) -> Result<()> {
        docker(&["restart", "-t", "60", &self.name]).map(|_| ())
    }

    pub fn stop(&self) -> Result<()> {
        docker(&["stop", "-t", "60", &self.name]).map(|_| ())
    }

    pub fn start(&self) -> Result<()> {
        docker(&["start", &self.name]).map(|_| ())
    }

    pub fn logs(&self) -> String {
        Command::new("docker")
            .args(["logs", "--tail", "200", &self.name])
            .output()
            .map(|out| {
                format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                )
            })
            .unwrap_or_default()
    }

    /// The container's cgroup directory (cgroup v2).
    pub fn cgroup(&self) -> Result<PathBuf> {
        let id = self.id()?;
        let candidates = [
            PathBuf::from(format!("/sys/fs/cgroup/system.slice/docker-{id}.scope")),
            PathBuf::from(format!("/sys/fs/cgroup/docker/{id}")),
        ];
        candidates
            .into_iter()
            .find(|dir| dir.join("memory.stat").exists())
            .ok_or_else(|| eyre!("no cgroup v2 directory for {}", self.name))
    }
}

/// Drop the runner's page cache, so that what an engine reads next comes
/// from the disk.
pub fn drop_caches() -> Result<()> {
    let status = Command::new("sudo")
        .args(["sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"])
        .status()
        .wrap_err("sudo")?;
    ensure!(status.success(), "dropping the page cache failed");
    Ok(())
}

/// Empty `dir` and leave it open to the user the image runs as.
pub fn emptied(dir: &Path) -> Result<()> {
    fresh(dir)
}

/// vellev and the Postgres it keeps its commit pointer in.
pub struct VellevUp {
    pub vellev: Container,
    pub postgres: Container,
    pub store: PathBuf,
    pub cache: PathBuf,
}

impl VellevUp {
    pub fn containers(&self) -> [&Container; 2] {
        [&self.vellev, &self.postgres]
    }
}

fn network() {
    let _ = Command::new("docker")
        .args(["network", "create", NETWORK])
        .output();
}

pub fn vellev_up(
    image: &str,
    postgres: &str,
    data: &Path,
    pinning: &Pinning,
    env: &[(String, String)],
) -> Result<VellevUp> {
    network();
    removed("bench-vellev");
    removed("bench-postgres");
    let store = data.join("store");
    let cache = data.join("cache");
    let pgdata = data.join("postgres");
    for dir in [&store, &cache, &pgdata] {
        fresh(dir)?;
    }
    ensure!(
        pinning.memory_mb > POSTGRES_MB * 2,
        "too little memory to share with Postgres"
    );
    let cpuset = format!("--cpuset-cpus={}", pinning.cpuset);
    let cpus = format!("--cpus={}", pinning.cpus);
    docker(&[
        "run",
        "-d",
        "--name",
        "bench-postgres",
        "--network",
        NETWORK,
        &cpuset,
        &cpus,
        &format!("--memory={POSTGRES_MB}m"),
        "-e",
        "POSTGRES_USER=vellev",
        "-e",
        "POSTGRES_PASSWORD=vellev",
        "-e",
        "POSTGRES_DB=vellev",
        "-v",
        &format!("{}:/var/lib/postgresql/data", pgdata.display()),
        postgres,
    ])?;
    let mut args: Vec<String> = ["run", "-d", "--name", "bench-vellev", "--network", NETWORK]
        .map(String::from)
        .to_vec();
    args.extend([
        cpuset,
        cpus,
        format!("--memory={}m", pinning.memory_mb - POSTGRES_MB),
        "-p".into(),
        format!("{VELLEV_PORT}:8080"),
        "-p".into(),
        format!("{VELLEV_METRICS_PORT}:9464"),
        "-v".into(),
        format!("{}:/bench/store", store.display()),
        "-v".into(),
        format!("{}:/bench/cache", cache.display()),
    ]);
    let mut all_env: Vec<(String, String)> = vec![
        (
            "VELLEV_DATABASE_URL".into(),
            "postgres://vellev:vellev@bench-postgres:5432/vellev".into(),
        ),
        ("VELLEV_STORE".into(), "file:///bench/store".into()),
        ("VELLEV_CACHE_DIR".into(), "/bench/cache".into()),
        ("VELLEV_API_KEYS".into(), VELLEV_KEY.into()),
        ("VELLEV_METRICS_LISTEN".into(), "0.0.0.0:9464".into()),
        ("VELLEV_LOCAL_MODEL".into(), String::new()),
        ("TOKIO_WORKER_THREADS".into(), pinning.cpus.to_string()),
        ("RAYON_NUM_THREADS".into(), pinning.cpus.to_string()),
    ];
    all_env.extend(env.iter().cloned());
    for (key, value) in &all_env {
        args.push("-e".into());
        args.push(format!("{key}={value}"));
    }
    args.push(image.into());
    docker(&args.iter().map(String::as_str).collect::<Vec<_>>())?;
    Ok(VellevUp {
        vellev: Container {
            name: "bench-vellev".into(),
        },
        postgres: Container {
            name: "bench-postgres".into(),
        },
        store,
        cache,
    })
}

pub fn weaviate_up(
    image: &str,
    data: &Path,
    pinning: &Pinning,
    env: &[(String, String)],
) -> Result<Container> {
    removed("bench-weaviate");
    fresh(data)?;
    let mut args: Vec<String> = vec![
        "run".into(),
        "-d".into(),
        "--name".into(),
        "bench-weaviate".into(),
        format!("--cpuset-cpus={}", pinning.cpuset),
        format!("--cpus={}", pinning.cpus),
        format!("--memory={}m", pinning.memory_mb),
        "-p".into(),
        format!("{WEAVIATE_PORT}:8080"),
        "-v".into(),
        format!("{}:/var/lib/weaviate", data.display()),
    ];
    let mut all_env: Vec<(String, String)> = [
        ("PERSISTENCE_DATA_PATH", "/var/lib/weaviate"),
        ("AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED", "true"),
        ("DEFAULT_VECTORIZER_MODULE", "none"),
        ("CLUSTER_HOSTNAME", "node1"),
        ("DISABLE_TELEMETRY", "true"),
        ("LIMIT_RESOURCES", "true"),
    ]
    .map(|(key, value)| (key.to_string(), value.to_string()))
    .to_vec();
    all_env.push(("GOMAXPROCS".into(), pinning.cpus.to_string()));
    all_env.extend(env.iter().cloned());
    for (key, value) in &all_env {
        args.push("-e".into());
        args.push(format!("{key}={value}"));
    }
    args.push(image.into());
    docker(&args.iter().map(String::as_str).collect::<Vec<_>>())?;
    Ok(Container {
        name: "bench-weaviate".into(),
    })
}

pub fn down(containers: &[&Container], data: &Path) -> Result<()> {
    for container in containers {
        removed(&container.name);
    }
    cleared(data)
}

/// Wait until `url` answers 2xx, at most `within`.
pub async fn ready(url: &str, within: Duration, container: &Container) -> Result<Duration> {
    let started = Instant::now();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    while started.elapsed() < within {
        if let Ok(response) = http.get(url).send().await {
            if response.status().is_success() {
                return Ok(started.elapsed());
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    bail!("{url} not ready within {within:?}:\n{}", container.logs())
}
