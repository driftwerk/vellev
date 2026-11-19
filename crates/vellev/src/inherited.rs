//! The node's secrets, handed over by a supervisor on an inherited pipe
//! (`vellev serve --secrets-fd <n>`) rather than in its environment.
//!
//! A process's starting environment is readable by every other process of
//! the same user (`ps -E`), so a supervisor on someone's own computer (Desk
//! Helper) writes the database URL, which holds Postgres's password, and the
//! admin keys to a pipe instead. The payload is its length (4 bytes,
//! big-endian), then one JSON object holding `VELLEV_DATABASE_URL` and
//! `VELLEV_API_KEYS` (comma-separated), as the environment would. Both are
//! required, and with a pipe it is their only source: either one also set
//! on the command line or in the environment is refused.
//!
//! The descriptor must be a pipe the node inherited (not close-on-exec);
//! once taken it is made close-on-exec, and blocking. The supervisor writes the payload once
//! and closes its end before the node starts: the node reads the frame and
//! then the pipe's end, so bytes written after the frame are refused, and
//! a write end still open elsewhere, like a supervisor that never writes,
//! is an error at the deadline rather than a node waiting for ever. Errors
//! name one of the two variables or a position, never a value, nor a name
//! the node doesn't take.

use std::collections::BTreeMap;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::time::Duration;

use eyre::{bail, eyre, Result};

use crate::config::Config;

/// The most a payload may hold.
pub const MAX_BYTES: usize = 64 * 1024;
/// How long the node waits for its supervisor's payload.
pub const READ_WITHIN: Duration = Duration::from_secs(10);

const DATABASE_URL: &str = "VELLEV_DATABASE_URL";
const API_KEYS: &str = "VELLEV_API_KEYS";

/// What a supervisor handed over. `Debug` shows which were given, never
/// their values.
#[derive(Default)]
pub struct Inherited {
    pub database_url: Option<String>,
    pub api_keys: Vec<String>,
}

impl std::fmt::Debug for Inherited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inherited")
            .field("database_url", &self.database_url.is_some())
            .field("api_keys", &self.api_keys.len())
            .finish()
    }
}

impl Inherited {
    /// Reads the descriptor `fd`, handed down by a supervisor for this
    /// alone, and closes it.
    ///
    /// The number is looked at without being taken: one that isn't open,
    /// isn't a pipe, or is close-on-exec (so was opened in this process,
    /// as Rust and tokio open every descriptor, rather than inherited) is
    /// refused and left as it was.
    ///
    /// # Safety
    ///
    /// If `fd` names an open pipe without close-on-exec, nothing else in
    /// this process may own it: it is taken, and closed once read.
    pub unsafe fn from_descriptor(fd: RawFd) -> Result<Self> {
        // SAFETY: passed on from this function's own contract.
        let owned = unsafe { Self::adopt(fd) }?;
        Self::read(owned, READ_WITHIN)
    }

    /// Checks `fd` by number, then takes it and makes it close-on-exec, so
    /// nothing the node starts inherits it.
    ///
    /// # Safety
    ///
    /// As [`Inherited::from_descriptor`].
    unsafe fn adopt(fd: RawFd) -> Result<OwnedFd> {
        if fd < 3 {
            bail!("--secrets-fd {fd} is one of the standard streams");
        }
        // SAFETY: `fcntl(F_GETFD)` and `fstat` read about a number, take
        // nothing, and fail harmlessly on one that isn't open.
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags == -1 {
            bail!("--secrets-fd {fd} isn't open");
        }
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: as above; `fstat` only writes `st`.
        if unsafe { libc::fstat(fd, st.as_mut_ptr()) } != 0 {
            bail!("--secrets-fd {fd} can't be looked at");
        }
        // SAFETY: `fstat` succeeded, so `st` is written.
        let mode = unsafe { st.assume_init() }.st_mode;
        if mode & libc::S_IFMT != libc::S_IFIFO {
            bail!("--secrets-fd {fd} doesn't name a pipe");
        }
        if flags & libc::FD_CLOEXEC != 0 {
            bail!(
                "--secrets-fd {fd} is close-on-exec, so it was opened in this process \
                 rather than inherited from a supervisor"
            );
        }
        // SAFETY: an open pipe without close-on-exec, which the caller
        // promises nothing else in this process owns.
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        // SAFETY: sets a flag on the descriptor just taken.
        if unsafe { libc::fcntl(owned.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
            bail!(
                "--secrets-fd {fd} can't be made close-on-exec: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(owned)
    }

    /// Reads a framed payload from the pipe `fd` within `within`, and
    /// closes it.
    pub fn read(fd: OwnedFd, within: Duration) -> Result<Self> {
        use std::os::unix::fs::FileTypeExt;
        let mut file = std::fs::File::from(fd);
        let is_pipe = file
            .metadata()
            .map_err(|e| eyre!("the secrets' descriptor can't be read: {e}"))?
            .file_type()
            .is_fifo();
        if !is_pipe {
            bail!("the secrets' descriptor isn't a pipe");
        }
        // A pipe made non-blocking (the flag belongs to the open pipe, so a
        // supervisor's choice is shared) would fail the read at once rather
        // than wait for the deadline: it is made blocking.
        let raw = file.as_raw_fd();
        // SAFETY: reads and sets the status flags of the descriptor owned
        // by `file`.
        let blocking = unsafe {
            let flags = libc::fcntl(raw, libc::F_GETFL);
            flags != -1
                && (flags & libc::O_NONBLOCK == 0
                    || libc::fcntl(raw, libc::F_SETFL, flags & !libc::O_NONBLOCK) != -1)
        };
        if !blocking {
            bail!(
                "the secrets' descriptor can't be made blocking: {}",
                std::io::Error::last_os_error()
            );
        }
        // Read on a thread of its own, so a supervisor that never writes is
        // an error at the deadline rather than a node that never starts.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(Self::read_frame(&mut file));
        });
        let body = match rx.recv_timeout(within) {
            Ok(Ok(body)) => body,
            Ok(Err(Unread::TooLong)) => bail!("the secrets are over {MAX_BYTES} bytes"),
            Ok(Err(Unread::Short)) => bail!("the secrets ended before their length"),
            Ok(Err(Unread::After)) => {
                bail!("more was written after the secrets; the supervisor writes them once")
            }
            Ok(Err(Unread::Io(e))) => bail!("reading the secrets: {e}"),
            Err(_) => bail!(
                "the secrets didn't arrive within {within:?}, with their writer's end \
                 closed after them"
            ),
        };
        Self::parse(&body)
    }

    /// The framed payload, and then the end of the pipe: the supervisor
    /// writes it once and closes its end, so anything after it was written
    /// by something else.
    fn read_frame(file: &mut std::fs::File) -> std::result::Result<Vec<u8>, Unread> {
        let short = |e: std::io::Error| match e.kind() {
            std::io::ErrorKind::UnexpectedEof => Unread::Short,
            _ => Unread::Io(e),
        };
        let mut header = [0u8; 4];
        file.read_exact(&mut header).map_err(short)?;
        let len = u32::from_be_bytes(header) as usize;
        if len > MAX_BYTES {
            return Err(Unread::TooLong);
        }
        let mut body = vec![0u8; len];
        file.read_exact(&mut body).map_err(short)?;
        let mut after = [0u8; 1];
        loop {
            match file.read(&mut after) {
                Ok(0) => return Ok(body),
                Ok(_) => return Err(Unread::After),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Unread::Io(e)),
            }
        }
    }

    fn parse(body: &[u8]) -> Result<Self> {
        // serde_json's messages can quote what they read: only the position
        // is kept.
        let raw: BTreeMap<String, String> = serde_json::from_slice(body).map_err(|e| {
            eyre!(
                "the secrets aren't a JSON object of strings (line {}, column {})",
                e.line(),
                e.column()
            )
        })?;
        let mut inherited = Self::default();
        for (name, value) in raw {
            // A name is repeated only once it is known to be one of the
            // two: a supervisor that writes the wrong shape may put a
            // secret where a name belongs.
            match name.as_str() {
                DATABASE_URL => {
                    if value.trim().is_empty() {
                        bail!("{DATABASE_URL} was handed over empty");
                    }
                    inherited.database_url = Some(value);
                }
                API_KEYS => {
                    inherited.api_keys = value
                        .split(',')
                        .map(|k| k.trim().to_string())
                        .filter(|k| !k.is_empty())
                        .collect();
                    if inherited.api_keys.is_empty() {
                        bail!("{API_KEYS} was handed over with no key in it");
                    }
                }
                _ => bail!(
                    "the secrets hold a name that isn't a secret the node takes on a pipe \
                     (only {DATABASE_URL} and {API_KEYS}; the name isn't repeated)"
                ),
            }
        }
        if inherited.database_url.is_none() {
            bail!("the secrets don't hold {DATABASE_URL}");
        }
        if inherited.api_keys.is_empty() {
            bail!("the secrets don't hold {API_KEYS}");
        }
        Ok(inherited)
    }

    /// Makes what was handed over the node's database URL and keys. With a
    /// pipe it is their only source: either one also given on the command
    /// line or in the environment, both readable by other processes, is
    /// refused rather than overridden, so a supervisor that still leaks
    /// them is found out.
    pub fn apply(self, config: &mut Config) -> Result<()> {
        if !config.database_url.trim().is_empty() {
            bail!(
                "{DATABASE_URL} is also set on the command line or in the environment; \
                 with --secrets-fd it comes only on the pipe"
            );
        }
        if config.api_keys.iter().any(|k| !k.trim().is_empty()) {
            bail!(
                "{API_KEYS} is also set on the command line or in the environment; \
                 with --secrets-fd they come only on the pipe"
            );
        }
        config.database_url = self.database_url.unwrap_or_default();
        config.api_keys = self.api_keys;
        Ok(())
    }
}

/// Why a frame couldn't be read.
enum Unread {
    TooLong,
    Short,
    After,
    Io(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::Duration;

    fn framed(payload: &[u8]) -> Vec<u8> {
        let mut out = (payload.len() as u32).to_be_bytes().to_vec();
        out.extend_from_slice(payload);
        out
    }

    fn through_pipe(bytes: &[u8]) -> eyre::Result<Inherited> {
        let (reader, mut writer) = std::io::pipe().unwrap();
        writer.write_all(bytes).unwrap();
        drop(writer);
        Inherited::read(reader.into(), Duration::from_secs(5))
    }

    #[test]
    fn the_database_url_and_keys_are_read_from_the_pipe() {
        let read = through_pipe(&framed(
            br#"{"VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db","VELLEV_API_KEYS":" k1-fixture, ,k2-fixture"}"#,
        ))
        .unwrap();
        assert_eq!(
            read.database_url.as_deref(),
            Some("postgres://u:pw-fixture@127.0.0.1:1/db")
        );
        assert_eq!(read.api_keys, ["k1-fixture", "k2-fixture"]);
        let shown = format!("{read:?}");
        assert!(
            !shown.contains("pw-fixture") && !shown.contains("k1-fixture"),
            "{shown}"
        );
    }

    const BOTH: &[u8] =
        br#"{"VELLEV_DATABASE_URL":"postgres://from-pipe","VELLEV_API_KEYS":"k-fixture"}"#;

    /// A config as clap leaves it with neither secret on the command line
    /// or in the environment.
    fn bare_config() -> Config {
        let mut config = crate::config::tests::config(&[]);
        config.database_url.clear();
        config.api_keys.clear();
        config
    }

    #[test]
    fn they_become_the_nodes_database_and_keys() {
        let mut config = bare_config();
        through_pipe(&framed(BOTH))
            .unwrap()
            .apply(&mut config)
            .unwrap();
        assert_eq!(config.database_url, "postgres://from-pipe");
        assert_eq!(config.api_keys, ["k-fixture"]);
    }

    /// With a pipe, the pipe is the only source: a database URL or keys
    /// also given on the command line or in the environment (both readable
    /// by other processes) are refused, naming the variable, not the value.
    #[test]
    fn with_a_pipe_neither_may_also_come_from_the_command_line_or_environment() {
        let mut url = bare_config();
        url.database_url = "postgres://u:pw-fixture@h/db".into();
        let mut keys = bare_config();
        keys.api_keys = vec!["env-key-fixture".into()];
        for (mut config, named) in [(url, "VELLEV_DATABASE_URL"), (keys, "VELLEV_API_KEYS")] {
            let err = format!(
                "{:#}",
                through_pipe(&framed(BOTH))
                    .unwrap()
                    .apply(&mut config)
                    .unwrap_err()
            );
            assert!(err.contains(named), "{err}");
            assert!(
                !err.contains("pw-fixture") && !err.contains("env-key-fixture"),
                "{err}"
            );
        }
    }

    /// A pipe that leaves either secret out is refused, rather than the
    /// node starting on whatever else it was given.
    #[test]
    fn the_pipe_must_hand_over_both() {
        for (payload, missing) in [
            (
                &br#"{"VELLEV_API_KEYS":"k-fixture"}"#[..],
                "VELLEV_DATABASE_URL",
            ),
            (
                br#"{"VELLEV_DATABASE_URL":"postgres://db"}"#,
                "VELLEV_API_KEYS",
            ),
            (br#"{}"#, "VELLEV_DATABASE_URL"),
        ] {
            let err = format!("{:#}", through_pipe(&framed(payload)).unwrap_err());
            assert!(err.contains(missing), "{err}");
        }
    }

    /// Anything after the frame is refused: the supervisor writes one
    /// payload and closes its end, so more bytes mean something else wrote
    /// to the pipe.
    #[test]
    fn bytes_after_the_frame_are_refused() {
        let mut twice = framed(BOTH);
        twice.extend(framed(br#"{"VELLEV_API_KEYS":"forged-fixture"}"#));
        let err = format!("{:#}", through_pipe(&twice).unwrap_err());
        assert!(err.contains("after"), "{err}");
        assert!(!err.contains("forged-fixture"), "{err}");
    }

    /// A name the node doesn't take is refused, and is not repeated: a
    /// supervisor that writes the wrong shape may put a secret where a
    /// name belongs.
    #[test]
    fn anything_but_the_two_names_is_refused_without_repeating_it() {
        for payload in [
            &br#"{"AWS_SECRET_ACCESS_KEY":"s-fixture"}"#[..],
            br#"{"postgres://u:pw-fixture@127.0.0.1:1/db":"x"}"#,
            br#"{"postgres://u:pw-fixture@127.0.0.1:1/db":""}"#,
        ] {
            let err = format!("{:#}", through_pipe(&framed(payload)).unwrap_err());
            assert!(err.contains("isn't a secret the node takes"), "{err}");
            for secret in ["AWS_SECRET_ACCESS_KEY", "s-fixture", "pw-fixture"] {
                assert!(!err.contains(secret), "{err}");
            }
        }
    }

    /// Keys handed over on the pipe that come to none once trimmed are
    /// refused, rather than leaving the command line's or environment's
    /// keys in force.
    #[test]
    fn a_key_list_that_is_empty_once_trimmed_is_refused() {
        for keys in [" ", ",", " , ,"] {
            let payload = format!(r#"{{"VELLEV_API_KEYS":"{keys}"}}"#);
            let err = format!(
                "{:#}",
                through_pipe(&framed(payload.as_bytes())).unwrap_err()
            );
            assert!(err.contains("VELLEV_API_KEYS"), "{err}");
        }
    }

    #[test]
    fn malformed_or_short_payloads_are_refused_without_repeating_them() {
        let mut short = framed(br#"{"VELLEV_API_KEYS":"k-fixture"}"#);
        short.truncate(short.len() - 2);
        for (bad, reason) in [
            (short, "ended before their length"),
            (vec![0, 0], "ended before their length"),
            (
                framed(br#"{"VELLEV_API_KEYS": 12345678}"#),
                "line 1, column",
            ),
            (framed(b"VELLEV_API_KEYS=k-fixture"), "line 1, column"),
            (
                ((MAX_BYTES as u32) + 1).to_be_bytes().to_vec(),
                "over 65536 bytes",
            ),
        ] {
            let err = through_pipe(&bad).unwrap_err().to_string();
            assert!(err.contains(reason), "{reason}: {err}");
            assert!(
                !err.contains("k-fixture") && !err.contains("12345678"),
                "{err}"
            );
        }
    }

    /// A supervisor that never writes, or whose write end stays open after
    /// the payload (so the end of it can't be confirmed), is an error at
    /// the deadline, never a node waiting for ever.
    #[test]
    fn a_silent_pipe_or_a_held_write_end_is_refused_at_the_deadline() {
        let (reader, writer) = std::io::pipe().unwrap();
        let err = Inherited::read(reader.into(), Duration::from_millis(200))
            .unwrap_err()
            .to_string();
        assert!(err.contains("within"), "{err}");
        drop(writer);
        let (reader, mut writer) = std::io::pipe().unwrap();
        writer.write_all(&framed(BOTH)).unwrap();
        let err = Inherited::read(reader.into(), Duration::from_millis(200))
            .unwrap_err()
            .to_string();
        assert!(err.contains("within"), "{err}");
        drop(writer);
    }

    #[test]
    fn only_a_pipe_is_read() {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&framed(br#"{"VELLEV_API_KEYS":"k"}"#))
            .unwrap();
        use std::io::Seek;
        file.rewind().unwrap();
        let err = Inherited::read(file.into(), Duration::from_secs(1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("pipe"), "{err}");
    }

    /// The descriptor-number rules: what `from_descriptor` takes, refuses
    /// and leaves alone. Whether a descriptor is still open is told by its
    /// inode, never by its number alone, which a test running alongside
    /// could reuse.
    mod descriptor {
        use super::*;
        use std::os::fd::{AsRawFd, IntoRawFd};

        fn inode(fd: RawFd) -> Option<(libc::dev_t, libc::ino_t)> {
            let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
            // SAFETY: `fstat` only writes `st`, and fails harmlessly on a
            // number that isn't open.
            (unsafe { libc::fstat(fd, st.as_mut_ptr()) } == 0).then(|| {
                // SAFETY: `fstat` succeeded, so `st` is written.
                let st = unsafe { st.assume_init() };
                (st.st_dev, st.st_ino)
            })
        }

        fn fd_flags(fd: RawFd) -> libc::c_int {
            // SAFETY: `F_GETFD` reads a flag; harmless on any number.
            unsafe { libc::fcntl(fd, libc::F_GETFD) }
        }

        /// As a supervisor's `dup2` leaves it in the child.
        fn inherited(fd: &impl AsRawFd) {
            // SAFETY: clears close-on-exec on a descriptor this test owns.
            assert_eq!(unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, 0) }, 0);
        }

        fn take(fd: RawFd) -> Result<Inherited> {
            // SAFETY: each test hands over only a descriptor it gives up,
            // or one the checks refuse before taking it.
            unsafe { Inherited::from_descriptor(fd) }
        }

        #[test]
        fn a_standard_stream_is_refused() {
            for fd in [0, 1, 2, -1] {
                assert!(take(fd).is_err(), "{fd}");
            }
        }

        #[test]
        fn a_number_that_isnt_open_is_refused() {
            let err = format!("{:#}", take(i32::MAX).unwrap_err());
            assert!(err.contains("isn't open"), "{err}");
        }

        /// A descriptor with close-on-exec set was opened in this process
        /// (Rust and tokio open every one so), not handed down by a
        /// supervisor: it belongs to something else here, and is refused
        /// and left open.
        #[test]
        fn a_pipe_this_process_opened_is_refused_and_left_open() {
            let (reader, mut writer) = std::io::pipe().unwrap();
            writer.write_all(&framed(BOTH)).unwrap();
            drop(writer);
            let before = inode(reader.as_raw_fd());
            let err = format!("{:#}", take(reader.as_raw_fd()).unwrap_err());
            assert!(err.contains("inherited"), "{err}");
            assert_eq!(inode(reader.as_raw_fd()), before);
            drop(reader);
        }

        #[test]
        fn an_inherited_descriptor_that_isnt_a_pipe_is_refused_and_left_open() {
            let file = tempfile::tempfile().unwrap();
            inherited(&file);
            let before = inode(file.as_raw_fd());
            let err = format!("{:#}", take(file.as_raw_fd()).unwrap_err());
            assert!(err.contains("pipe"), "{err}");
            assert_eq!(inode(file.as_raw_fd()), before);
            drop(file);
        }

        /// Once taken, the descriptor is close-on-exec, so nothing the node
        /// might start inherits it.
        #[test]
        fn a_taken_descriptor_is_not_passed_on() {
            let (reader, _writer) = std::io::pipe().unwrap();
            inherited(&reader);
            let fd = reader.into_raw_fd();
            // SAFETY: the test gives `fd` up here.
            let owned = unsafe { Inherited::adopt(fd) }.unwrap();
            assert_ne!(fd_flags(owned.as_raw_fd()) & libc::FD_CLOEXEC, 0);
        }

        #[test]
        fn an_inherited_pipe_is_read_and_closed() {
            let (reader, mut writer) = std::io::pipe().unwrap();
            writer.write_all(&framed(BOTH)).unwrap();
            drop(writer);
            inherited(&reader);
            let pipe = inode(reader.as_raw_fd());
            let fd = reader.into_raw_fd();
            let read = take(fd).unwrap();
            assert_eq!(read.api_keys, ["k-fixture"]);
            // Closed: the number is free, or reused by something else.
            assert!(inode(fd).is_none_or(|now| Some(now) != pipe));
        }

        /// A supervisor whose pipe is non-blocking (shared with the node,
        /// as the flag belongs to the open pipe) is still waited for until
        /// the deadline, not failed at once.
        #[test]
        fn a_non_blocking_pipe_is_waited_for() {
            let (reader, mut writer) = std::io::pipe().unwrap();
            // SAFETY: sets a flag on a descriptor this test owns.
            unsafe {
                let flags = libc::fcntl(reader.as_raw_fd(), libc::F_GETFL);
                assert_eq!(
                    libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK),
                    0
                );
            }
            let late = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                writer.write_all(&framed(BOTH)).unwrap();
            });
            let read = Inherited::read(reader.into(), Duration::from_secs(5)).unwrap();
            assert_eq!(read.api_keys, ["k-fixture"]);
            late.join().unwrap();
        }
    }
}
