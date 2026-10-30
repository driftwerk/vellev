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
    /// The most it has held (`memory.peak`, kernels from 5.19).
    pub peak: Option<u64>,
    /// The cgroup's own limit (`memory.max`), unless it has none. A lower
    /// limit an ancestor sets binds it too, and is not this.
    pub max: Option<u64>,
    /// Its own high mark (`memory.high`), past which it is throttled and
    /// reclaimed from, unless it has none.
    pub high: Option<u64>,
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
            .and_then(|named| cgroup_dir(Path::new("/sys/fs/cgroup"), &named));
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
            anon: kib("RssAnon:").and_then(|kib| kib.checked_mul(1024)),
            file: kib("RssFile:").and_then(|kib| kib.checked_mul(1024)),
            ..Memory::default()
        };
        let Some(cgroup) = &self.cgroup else {
            return memory;
        };
        let read = |name: &str| std::fs::read_to_string(cgroup.join(name)).ok();
        // Each is "max" where there is no bound, and so is left out.
        let bytes = |name: &str| read(name).and_then(|text| text.trim().parse().ok());
        memory.current = bytes("memory.current");
        memory.peak = bytes("memory.peak");
        memory.max = bytes("memory.max");
        memory.high = bytes("memory.high");
        if let Some(text) = read("memory.events") {
            memory.events = keyed(&text, &EVENTS);
        }
        if let Some(text) = read("memory.stat") {
            memory.stat = keyed(&text, &KINDS);
        }
        memory
    }
}

/// The directory of the cgroup v2 a process is in, under `root`, where
/// cgroup v2 is mounted, from what `/proc/self/cgroup` says (`named`): the
/// line `0::PATH`. None where memory is not on cgroup v2: a host with v1
/// alone has no such line, and a hybrid host has one but keeps memory on
/// v1 (a line `N:...memory...:PATH`), so that v2 has no memory files.
pub fn cgroup_dir(root: &Path, named: &str) -> Option<PathBuf> {
    let on_v1 = named.lines().any(|line| {
        let mut fields = line.splitn(3, ':');
        let (_, controllers) = (fields.next(), fields.next().unwrap_or(""));
        controllers
            .split(',')
            .any(|controller| controller == "memory")
    });
    if on_v1 {
        return None;
    }
    let path = v2_cgroup(named)?;
    Some(root.join(path.trim_start_matches('/')))
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

    /// Review of #165, M2: where a process's cgroup v2 directory is, from
    /// what `/proc/self/cgroup` says, under the cgroup v2 mount `root`.
    #[test]
    fn the_cgroup_directory_is_found_only_where_memory_is_on_cgroup_v2() {
        let root = Path::new("/sys/fs/cgroup");
        // A container in a cgroup namespace of its own.
        assert_eq!(cgroup_dir(root, "0::/\n"), Some(root.to_path_buf()));
        // A systemd service or scope, as on a host.
        assert_eq!(
            cgroup_dir(root, "0::/system.slice/x.scope\n"),
            Some(root.join("system.slice/x.scope"))
        );
        // A hybrid host: v2 is mounted beside v1, but memory is v1's, so
        // v2 has no memory files to read.
        let hybrid = "12:memory:/user.slice\n11:cpu,cpuacct:/user.slice\n0::/user.slice/x.scope\n";
        assert_eq!(cgroup_dir(root, hybrid), None);
        // A host with v1 alone.
        let v1 = "12:memory:/docker/abc\n4:pids:/docker/abc\n1:name=systemd:/docker/abc\n";
        assert_eq!(cgroup_dir(root, v1), None);
        // Nothing said.
        assert_eq!(cgroup_dir(root, ""), None);
    }

    /// On Linux, this process's own files are read: its anonymous and
    /// file-backed memory always, and its cgroup's memory where cgroup v2
    /// is mounted.
    #[cfg(target_os = "linux")]
    #[test]
    fn this_process_is_read_as_linux_has_it() {
        let memory = MemoryFiles::of_this_process().read();
        assert!(memory.anon.is_some_and(|bytes| bytes > 0), "{memory:?}");
        assert!(memory.file.is_some(), "{memory:?}");
        // Not in the root cgroup, which has no memory.current of its own.
        let own = std::fs::read_to_string("/proc/self/cgroup")
            .ok()
            .and_then(|named| cgroup_dir(Path::new("/sys/fs/cgroup"), &named))
            .map(|dir| dir.join("memory.current"));
        if own.is_some_and(|current| current.exists()) {
            assert!(memory.current.is_some_and(|bytes| bytes > 0), "{memory:?}");
        }
    }

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
