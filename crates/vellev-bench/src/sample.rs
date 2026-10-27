//! Memory and disk, sampled every second while an engine runs (design r2,
//! §4): anonymous memory (`memory.stat` anon) reported apart from the file
//! cache, `memory.peak` labelled as including the cache, and the bytes
//! used on the runner's disk, each at its high-water mark in each phase.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use eyre::{eyre, Result, WrapErr};
use serde::Serialize;

/// The figures of one phase, in bytes.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq)]
pub struct PhaseMemory {
    /// The most anonymous memory seen: the engine's own.
    pub anon_peak: u64,
    /// The most file cache seen.
    pub file_peak: u64,
    /// The most bytes used on the runner's disk.
    pub disk_used_peak: u64,
    /// Samples taken.
    pub samples: u64,
}

/// One reading of the engine's containers and the runner's disk.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq)]
pub struct Reading {
    pub anon: u64,
    pub file: u64,
    /// `memory.peak`, summed over the containers: it includes the cache.
    pub peak_including_cache: u64,
    pub disk_used: u64,
}

/// `anon` and `file` of a cgroup's `memory.stat`.
pub fn memory_stat(text: &str) -> Result<(u64, u64)> {
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(name)?.strip_prefix(' '))
            .and_then(|value| value.trim().parse::<u64>().ok())
            .ok_or_else(|| eyre!("memory.stat has no {name}"))
    };
    Ok((field("anon")?, field("file")?))
}

/// The bytes used on the file system holding `path`, as `df` says.
pub fn disk_used(path: &Path) -> Result<u64> {
    let output = std::process::Command::new("df")
        .args(["-B1", "--output=used"])
        .arg(path)
        .output()
        .wrap_err("df")?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .nth(1)
        .and_then(|line| line.trim().parse().ok())
        .ok_or_else(|| eyre!("df: {text}"))
}

/// Read the containers' cgroups and the disk once.
pub fn read(cgroups: &[PathBuf], disk: &Path) -> Result<Reading> {
    let mut reading = Reading {
        disk_used: disk_used(disk)?,
        ..Reading::default()
    };
    for cgroup in cgroups {
        let stat = std::fs::read_to_string(cgroup.join("memory.stat"))
            .wrap_err_with(|| cgroup.display().to_string())?;
        let (anon, file) = memory_stat(&stat)?;
        reading.anon += anon;
        reading.file += file;
        reading.peak_including_cache += std::fs::read_to_string(cgroup.join("memory.peak"))
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0);
    }
    Ok(reading)
}

/// Samples taken every second, each counted towards the phase the run is
/// in when it is taken.
pub struct Sampler {
    phase: Arc<Mutex<String>>,
    phases: Arc<Mutex<BTreeMap<String, PhaseMemory>>>,
    last: Arc<Mutex<Reading>>,
    task: tokio::task::JoinHandle<()>,
}

impl Sampler {
    /// Sample the containers whose cgroups are `cgroups` and the disk
    /// holding `disk`, starting in `phase`. A container restarted keeps
    /// its cgroup directory, so the paths stay good.
    pub fn start(cgroups: Vec<PathBuf>, disk: PathBuf, phase: &str) -> Sampler {
        let phase = Arc::new(Mutex::new(phase.to_string()));
        let phases: Arc<Mutex<BTreeMap<String, PhaseMemory>>> = Arc::default();
        let last: Arc<Mutex<Reading>> = Arc::default();
        let task = {
            let (phase, phases, last) = (phase.clone(), phases.clone(), last.clone());
            tokio::spawn(async move {
                loop {
                    // A container between a stop and a start has no cgroup
                    // to read: that second is not counted.
                    if let Ok(reading) = read(&cgroups, &disk) {
                        *last.lock().expect("last") = reading;
                        let name = phase.lock().expect("phase").clone();
                        let mut phases = phases.lock().expect("phases");
                        let seen = phases.entry(name).or_default();
                        seen.anon_peak = seen.anon_peak.max(reading.anon);
                        seen.file_peak = seen.file_peak.max(reading.file);
                        seen.disk_used_peak = seen.disk_used_peak.max(reading.disk_used);
                        seen.samples += 1;
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            })
        };
        Sampler {
            phase,
            phases,
            last,
            task,
        }
    }

    pub fn phase(&self, name: &str) {
        *self.phase.lock().expect("phase") = name.to_string();
    }

    pub fn last(&self) -> Reading {
        *self.last.lock().expect("last")
    }

    pub fn stop(self) -> BTreeMap<String, PhaseMemory> {
        self.task.abort();
        self.phases.lock().expect("phases").clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anon_and_file_are_read_by_their_whole_names() {
        let stat = "anon_thp 5\nanon 1024\nfile_mapped 7\nfile 2048\nkernel 9\n";
        assert_eq!(memory_stat(stat).unwrap(), (1024, 2048));
        assert!(memory_stat("anon_thp 5\nfile 2\n").is_err());
    }
}
