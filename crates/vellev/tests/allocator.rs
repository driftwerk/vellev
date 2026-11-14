//! The node runs on jemalloc with the options that give freed memory back
//! (vellev memory note, fix 1): the background thread, in this test binary
//! and in the node binary itself. That memory is given back is held in
//! `tests/memory.rs`, a binary of its own.

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: vellev::allocator::Allocator = vellev::allocator::Allocator::new();

/// The node binary's jemalloc options, exported here as the binary does.
#[cfg(target_os = "linux")]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static OPTIONS: vellev::allocator::Options =
    vellev::allocator::Options(vellev::allocator::OPTIONS.as_ptr());

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
    // Found, nm is stopped and its status says nothing; not found, nm
    // must have read the whole binary, or the export may be there unseen.
    if export.is_some() {
        let _ = nm.kill();
    }
    let status = nm.wait().expect("nm's status");
    let export = export.unwrap_or_else(|| {
        assert!(status.success(), "nm could not list {binary}: {status}");
        panic!("{binary} does not export _rjem_malloc_conf")
    });
    let kind = export.split_whitespace().nth(1).unwrap_or_default();
    assert!(
        matches!(kind, "D" | "d" | "R" | "r"),
        "_rjem_malloc_conf is not initialised data in {binary}: {export}"
    );
}
