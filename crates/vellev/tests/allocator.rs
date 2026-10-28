//! The node gives back memory it has freed (vellev memory note, fix 1).
//!
//! A flush or a merge copies a run's vectors several times over and frees
//! the copies, leaving a little behind among them. Under glibc's malloc
//! that left the process holding its high-water mark: 4.3 GiB of anonymous
//! memory at rest against 0.9 GiB in use, until a 1M-row load was killed
//! for memory. This holds the node's allocator to giving such memory back.

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

/// The node binary's jemalloc options, exported here as the binary does.
#[cfg(target_os = "linux")]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static OPTIONS: vellev::allocator::Options =
    vellev::allocator::Options(vellev::allocator::OPTIONS.as_ptr());

/// One test at a time: the memory test reads the whole process's memory,
/// which another test's allocations, such as the symbol list below, would
/// count against it.
#[cfg(target_os = "linux")]
static ALONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// What this test binary's own export of the options gives jemalloc: the
/// background thread, which only the options turn on.
#[cfg(target_os = "linux")]
#[test]
fn the_background_thread_that_gives_memory_back_is_on() {
    assert_eq!(tikv_jemalloc_ctl::opt::background_thread::read(), Ok(true));
}

/// The node binary itself carries the export, so that it is not only this
/// test, which exports its own copy, that turns the background thread on:
/// `_rjem_malloc_conf` is defined in the built `vellev`, as initialised
/// data, which is where a static pointer jemalloc reads lives. A release
/// profile with `strip = true` would remove the symbol table this reads,
/// and fail this test though the export were there.
#[cfg(target_os = "linux")]
#[test]
fn the_node_binary_gives_jemalloc_its_options() {
    let _alone = ALONE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    use std::io::BufRead;
    let binary = env!("CARGO_BIN_EXE_vellev");
    // Read a line at a time, so that the listing is never held whole.
    let mut nm = std::process::Command::new("nm")
        .arg("--defined-only")
        .arg(binary)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("nm, from binutils");
    let listing = std::io::BufReader::new(nm.stdout.take().expect("nm's output"));
    let export = listing
        .lines()
        .map_while(Result::ok)
        .find(|line| line.ends_with(" _rjem_malloc_conf"));
    let _ = nm.kill();
    let _ = nm.wait();
    let export = export.unwrap_or_else(|| panic!("{binary} does not export _rjem_malloc_conf"));
    let kind = export.split_whitespace().nth(1).unwrap_or_default();
    assert!(
        matches!(kind, "D" | "d" | "R" | "r"),
        "_rjem_malloc_conf is not initialised data in {binary}: {export}"
    );
}

/// What jemalloc says it holds, for a failure to show.
#[cfg(target_os = "linux")]
fn jemalloc() -> String {
    use tikv_jemalloc_ctl::{arenas, epoch, raw, stats};
    let _ = epoch::advance();
    // SAFETY: the name is jemalloc's, of a size_t statistic.
    let threads =
        unsafe { raw::read::<usize>(b"stats.background_thread.num_threads\0") }.unwrap_or(0);
    format!(
        "jemalloc: allocated {} MiB, resident {} MiB, retained {} MiB, {threads} background threads, {} arenas",
        stats::allocated::read().unwrap_or(0) >> 20,
        stats::resident::read().unwrap_or(0) >> 20,
        stats::retained::read().unwrap_or(0) >> 20,
        arenas::narenas::read().unwrap_or(0),
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
    // Scratch: another thread allocates and frees about 100 MiB of large
    // blocks while this test runs, as a flush on another pool would.
    let started = std::time::Instant::now();
    let other = std::thread::spawn(|| {
        let blocks: Vec<Vec<u8>> = (0..25).map(|n| vec![n as u8; 4 * 1024 * 1024]).collect();
        std::thread::sleep(std::time::Duration::from_millis(500));
        drop(blocks);
        // And the original pattern: nm's whole listing taken at once, and
        // copied into a string, on this other thread.
        let listed = std::process::Command::new("nm")
            .arg("--defined-only")
            .arg(env!("CARGO_BIN_EXE_vellev"))
            .output()
            .expect("nm");
        let text = String::from_utf8_lossy(&listed.stdout).into_owned();
        let size = (listed.stdout.capacity() + text.capacity()) >> 20;
        drop((listed, text));
        size
    });
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
    // times what is kept (about 220 MiB here; 238 MiB measured with the
    // rest of the process). glibc kept all 890 MiB.
    let bound = before + 4 * held + 64 * MIB;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while anon() > bound && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    let listing = other.join().unwrap();
    panic!(
        "scratch report: nm's listing took {listing} MiB on the other thread; {} MiB anonymous ({} MiB before, bound {} MiB) {:.1} s in; {}",
        anon() / MIB,
        before / MIB,
        bound / MIB,
        started.elapsed().as_secs_f64(),
        jemalloc()
    );
    #[allow(unreachable_code)]
    assert!(
        anon() <= bound,
        "{} MiB anonymous ({} MiB before, bound {} MiB) with {} MiB kept: what was freed was not given back; {}",
        anon() / MIB,
        before / MIB,
        bound / MIB,
        held / MIB,
        jemalloc()
    );
    drop(kept);
}
