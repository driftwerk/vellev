//! What the process and its cgroup really hold in memory, as Linux's files
//! say: the measure the node's own reckoning of its tenants (its memory
//! budget's) is checked against (resident accounting, PR 1).
//!
//! jemalloc counts only what the node's Rust code allocates. The local
//! embedding model, onnxruntime's, allocates through the C library's
//! `malloc`, about 1 GB of it, which only the process's and the cgroup's
//! figures show. Each file is read as the metrics are, and a figure whose
//! file is not there (macOS, or no cgroup v2) is not reported at all.

use std::path::{Path, PathBuf};

/// Where the process's and its cgroup's memory are read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryFiles {
    /// The process's status, as `/proc/self/status`.
    status: PathBuf,
    /// Its cgroup's (v2) directory, holding `memory.current` and the rest,
    /// if it has one.
    cgroup: Option<PathBuf>,
}

/// What the process and its cgroup hold now, as far as their files say.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Memory {
    /// The process's resident anonymous memory (`RssAnon`), in bytes.
    pub anon: Option<u64>,
    /// Its resident file-backed memory (`RssFile`), in bytes: mapped
    /// vectors and the binary among it.
    pub file: Option<u64>,
    /// The cgroup's memory now (`memory.current`).
    pub current: Option<u64>,
    /// The cgroup's limit (`memory.max`), unless it has none.
    pub max: Option<u64>,
    /// How often the cgroup was throttled, met its limit, ran out and had a
    /// process killed for it (`memory.events`), as many of those as it says.
    pub events: Vec<(&'static str, u64)>,
    /// Its anonymous and file-backed memory (`memory.stat`), in bytes.
    pub stat: Vec<(&'static str, u64)>,
}

/// The events of `memory.events` reported.
const EVENTS: [&str; 4] = ["high", "max", "oom", "oom_kill"];
/// The kinds of memory of `memory.stat` reported.
const KINDS: [&str; 2] = ["anon", "file"];

impl MemoryFiles {
    /// This process's files: its status, and the cgroup `/proc/self/cgroup`
    /// names under `/sys/fs/cgroup`, if it names one of cgroup v2.
    pub fn of_this_process() -> Self {
        let cgroup = std::fs::read_to_string("/proc/self/cgroup")
            .ok()
            .and_then(|named| v2_cgroup(&named).map(str::to_string))
            .map(|path| Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/')));
        Self {
            status: PathBuf::from("/proc/self/status"),
            cgroup,
        }
    }

    /// The files at these paths: a process's status, and its cgroup's
    /// directory if it has one.
    pub fn at(status: PathBuf, cgroup: Option<PathBuf>) -> Self {
        Self { status, cgroup }
    }

    /// What the files say now.
    pub fn read(&self) -> Memory {
        let status = std::fs::read_to_string(&self.status).ok();
        let kib = |name: &str| status.as_deref().and_then(|text| status_kib(text, name));
        let mut memory = Memory {
            anon: kib("RssAnon:").map(|kib| kib * 1024),
            file: kib("RssFile:").map(|kib| kib * 1024),
            ..Memory::default()
        };
        let Some(cgroup) = &self.cgroup else {
            return memory;
        };
        let read = |name: &str| std::fs::read_to_string(cgroup.join(name)).ok();
        memory.current = read("memory.current").and_then(|text| text.trim().parse().ok());
        // "max" where there is no limit.
        memory.max = read("memory.max").and_then(|text| text.trim().parse().ok());
        if let Some(text) = read("memory.events") {
            memory.events = keyed(&text, &EVENTS);
        }
        if let Some(text) = read("memory.stat") {
            memory.stat = keyed(&text, &KINDS);
        }
        memory
    }
}

/// The path of the cgroup v2 a process is in, as `/proc/self/cgroup`
/// gives it: the line `0::PATH`.
fn v2_cgroup(named: &str) -> Option<&str> {
    named.lines().find_map(|line| line.strip_prefix("0::"))
}

/// The figure in kB on the status line `name`, as `RssAnon:   2000 kB`.
fn status_kib(status: &str, name: &str) -> Option<u64> {
    let line = status.lines().find_map(|line| line.strip_prefix(name))?;
    line.split_whitespace().next()?.parse().ok()
}

/// Each of `wanted` that a file of `key value` lines has, in that order.
fn keyed(text: &str, wanted: &[&'static str]) -> Vec<(&'static str, u64)> {
    wanted
        .iter()
        .filter_map(|key| {
            let value = text.lines().find_map(|line| {
                let (named, value) = line.split_once(' ')?;
                (named == *key).then(|| value.trim().parse().ok())?
            })?;
            Some((*key, value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cgroup_v2_path_is_the_line_of_hierarchy_zero() {
        assert_eq!(v2_cgroup("0::/\n"), Some("/"));
        assert_eq!(
            v2_cgroup("12:memory:/old\n0::/system.slice/vellev.service\n"),
            Some("/system.slice/vellev.service")
        );
        assert_eq!(v2_cgroup("12:memory:/old\n"), None);
    }

    #[test]
    fn a_status_figure_is_read_in_kib_by_its_whole_name() {
        let status = "VmRSS:\t 3000 kB\nRssAnon:\t 2000 kB\nRssFile:\t 900 kB\n";
        assert_eq!(status_kib(status, "RssAnon:"), Some(2000));
        assert_eq!(status_kib(status, "RssFile:"), Some(900));
        assert_eq!(status_kib(status, "RssShmem:"), None);
    }

    #[test]
    fn only_the_keys_wanted_are_read_and_a_key_is_never_a_prefix_of_another() {
        let events = "low 0\nhigh 3\nmax 4\noom 1\noom_kill 2\noom_group_kill 9\n";
        assert_eq!(
            keyed(events, &EVENTS),
            vec![("high", 3), ("max", 4), ("oom", 1), ("oom_kill", 2)]
        );
        assert_eq!(keyed("anon 5\n", &KINDS), vec![("anon", 5)]);
    }
}
