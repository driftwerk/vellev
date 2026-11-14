//! The node gives back memory it has freed (vellev memory note, fix 1).
//!
//! A flush or a merge copies a run's vectors several times over and frees
//! the copies, leaving a little behind among them. Under glibc's malloc
//! that left the process holding its high-water mark: 4.3 GiB of anonymous
//! memory at rest against 0.9 GiB in use, until a 1M-row load was killed
//! for memory. This holds the node's allocator to giving such memory back.
//!
//! The test reads the whole process's memory, so it is the only test in
//! this binary: another test's allocations would count against it. A test
//! that panics with `RUST_BACKTRACE=1`, for one, leaves the symbols std
//! read to print its backtrace allocated for the rest of the process,
//! 134 MiB for the allocator tests' binary.

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

/// The node binary's jemalloc options, exported here as the binary does.
#[cfg(target_os = "linux")]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static OPTIONS: vellev::allocator::Options =
    vellev::allocator::Options(vellev::allocator::OPTIONS.as_ptr());

/// A size_t statistic or setting of jemalloc's, as of the last epoch
/// advanced; usize::MAX if it cannot be read.
#[cfg(target_os = "linux")]
fn statistic(name: &str) -> usize {
    let name = format!("{name}\0");
    // SAFETY: every name asked is jemalloc's, of a size_t statistic.
    unsafe { tikv_jemalloc_ctl::raw::read::<usize>(name.as_bytes()) }.unwrap_or(usize::MAX)
}

/// Bytes in pages freed and not yet given back, in every arena: what
/// jemalloc's decay still has to purge.
#[cfg(target_os = "linux")]
fn dirty() -> u64 {
    let _ = tikv_jemalloc_ctl::epoch::advance();
    let pages = statistic("stats.arenas.4096.pdirty") as u64;
    pages.saturating_mul(statistic("arenas.page") as u64)
}

/// What jemalloc holds, for a failure to show.
#[cfg(target_os = "linux")]
fn jemalloc() -> String {
    const MIB: usize = 1024 * 1024;
    // One epoch for every figure, so that they add up as of one moment.
    let _ = tikv_jemalloc_ctl::epoch::advance();
    let pages = statistic("stats.arenas.4096.pdirty") as u64;
    let dirty = pages.saturating_mul(statistic("arenas.page") as u64);
    format!(
        "jemalloc: allocated {} MiB, active {} MiB, metadata {} MiB, dirty {} MiB, resident {} MiB, retained {} MiB, {} background threads",
        statistic("stats.allocated") / MIB,
        statistic("stats.active") / MIB,
        statistic("stats.metadata") / MIB,
        dirty / MIB as u64,
        statistic("stats.resident") / MIB,
        statistic("stats.retained") / MIB,
        statistic("stats.background_thread.num_threads"),
    )
}

/// The process's anonymous resident memory, in bytes.
#[cfg(target_os = "linux")]
fn anon() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let kb: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("RssAnon:"))
        .and_then(|rest| rest.split_whitespace().next()?.parse().ok())
        .unwrap();
    kb * 1024
}

#[cfg(target_os = "linux")]
#[test]
fn memory_freed_among_what_is_kept_is_given_back() {
    const MIB: u64 = 1024 * 1024;
    let before = anon();
    // About 900 MiB in 3 KiB pieces, as a flush's copies of its vectors.
    let pieces: Vec<Vec<u8>> = (0..300_000)
        .map(|n| vec![(n % 251) as u8; 3 * 1024])
        .collect();
    assert!(anon() > before + 800 * MIB, "the pieces are resident");
    // One in sixteen kept, scattered among the rest, as a segment is.
    let kept: Vec<Vec<u8>> = pieces
        .into_iter()
        .enumerate()
        .filter(|(n, _)| n % 16 == 0)
        .map(|(_, piece)| piece)
        .collect();
    let held = kept.iter().map(|piece| piece.len() as u64).sum::<u64>();
    // An allocator that hands out small pieces from pages of several keeps
    // a page any one of its pieces is still in: jemalloc's pages for 3 KiB
    // pieces hold four, so one kept in sixteen keeps one page in four, four
    // times what is kept (about 220 MiB here; 239 MiB measured at rest with
    // the rest of the process: 227 MiB of pages in use and 14 MiB of
    // jemalloc's own). glibc kept all 890 MiB.
    let bound = before + 4 * held + 64 * MIB;
    // Judged at rest: jemalloc gives freed pages back gradually, over its
    // decay time of 10 s, and anonymous memory falls with them. Read part
    // way down, it holds pages it is about to give back; so the test waits
    // until none are left to give back, or the deadline.
    let freed = std::time::Instant::now();
    let deadline = freed + std::time::Duration::from_secs(60);
    while dirty() > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let rest = anon();
    eprintln!(
        "{} MiB anonymous at rest, {:.1} s after the free (bound {} MiB); {}",
        rest / MIB,
        freed.elapsed().as_secs_f64(),
        bound / MIB,
        jemalloc()
    );
    assert!(
        rest <= bound,
        "{} MiB anonymous at rest, {:.1} s after the free ({} MiB before, bound {} MiB) with {} MiB kept: what was freed was not given back; {}",
        rest / MIB,
        freed.elapsed().as_secs_f64(),
        before / MIB,
        bound / MIB,
        held / MIB,
        jemalloc()
    );
    drop(kept);
}
