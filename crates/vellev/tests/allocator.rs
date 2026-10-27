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
    // a page any one of its pieces is still in: one kept in sixteen leaves
    // about a quarter of jemalloc's pages for 3 KiB pieces in use. What is
    // asked is that most of what was freed goes back, at least half; glibc
    // gave back none of it.
    let freed = 300_000 * 3 * 1024 - held;
    let bound = before + held + freed / 2;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while anon() > bound && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    assert!(
        anon() <= bound,
        "{} MiB anonymous with {} MiB kept of {} MiB: not half of what was freed was given back",
        anon() / MIB,
        held / MIB,
        (held + freed) / MIB
    );
    drop(kept);
}
