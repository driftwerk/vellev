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
//! The descriptor must be a pipe. The supervisor writes the payload once
//! and closes its end before the node starts, and no other copy of the
//! write end may survive anywhere, in other children of the supervisor
//! included: the node reads the frame and then the pipe's end, so bytes
//! written after the frame, or a name given twice in it, are refused, and a
//! write end still open elsewhere, like a supervisor that never writes, is
//! an error at the deadline (saying which) rather than a node waiting for
//! ever. The node then exits refused, and its supervisor sees a start that
//! failed. Errors name one of the two variables or a position, never a
//! value, nor a name the node doesn't take.
//!
//! Only these two take the pipe: embedding keys and an object store's
//! credentials still come from the environment. On Linux another process
//! of the same user can open `/proc/<pid>/fd/<n>` in the moment before the
//! node reads, so the pipe shortens the exposure the environment had but
//! does not close it: such a process can read the payload, and can even
//! replace it whole, undetected. Only a partial read or bytes added once
//! the node reads fail the start. macOS has no such path.

use std::collections::BTreeSet;
use std::io::Read;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::time::Duration;

use eyre::{bail, eyre, Result};

use crate::config::Config;

/// The most a payload may hold: a conservative cap, so that with its 4-byte
/// length it fits a default fresh pipe (16 KiB on macOS, 64 KiB on Linux),
/// as the supervisor writes it all before the node starts to read. A
/// Linux user over `pipe-user-pages-soft` gets smaller pipes; supervisors
/// should keep the payload far below this (Desk Helper's is a few hundred
/// bytes).
pub const MAX_BYTES: usize = 16 * 1024 - 4;
const _: () = assert!(MAX_BYTES + 4 <= 16 * 1024, "a frame must fit a fresh pipe");
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
    /// Reads the descriptor `fd`, handed to this process for this alone,
    /// and closes it.
    pub fn from_descriptor(fd: RawFd) -> Result<Self> {
        if fd < 3 {
            bail!("--secrets-fd {fd} is one of the standard streams");
        }
        use std::os::unix::fs::FileTypeExt;
        // Looked at without being owned, so a number that names something
        // else (nothing was handed over on it) is left open.
        // SAFETY: the `File` is never dropped, so the descriptor isn't
        // closed; `fstat` on a number that isn't open fails harmlessly.
        let peek = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
        if !peek.metadata().is_ok_and(|m| m.file_type().is_fifo()) {
            bail!("--secrets-fd {fd} doesn't name a pipe");
        }
        // SAFETY: a pipe handed to this process to be read here, once;
        // nothing else owns it.
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        Self::read(owned, READ_WITHIN)
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
        // Read on a thread of its own, so a supervisor that never writes is
        // an error at the deadline rather than a node that never starts.
        let (tx, rx) = std::sync::mpsc::channel();
        // Set once the frame is in, so a pipe that never ends after it is
        // told from one nothing was written to.
        let framed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let framed_in = framed.clone();
        std::thread::spawn(move || {
            let read = Self::read_frame(&mut file, &framed_in);
            // Closed before the answer is sent, so it is closed by the time
            // this returns, not some time after.
            drop(file);
            let _ = tx.send(read);
        });
        let body = match rx.recv_timeout(within) {
            Ok(Ok(body)) => body,
            Ok(Err(Unread::TooLong)) => bail!(
                "the secrets' length is over the most the secrets may hold ({MAX_BYTES} bytes)"
            ),
            Ok(Err(Unread::Short)) => bail!("the secrets ended before their length"),
            Ok(Err(Unread::After)) => {
                bail!(
                    "more was written after the secrets' framed length: the supervisor writes \
                     them once, after their exact length"
                )
            }
            Ok(Err(Unread::Io(e))) => bail!("reading the secrets: {e}"),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                if framed.load(std::sync::atomic::Ordering::Acquire) =>
            {
                bail!(
                    "the secrets arrived, but the pipe's write end is still open elsewhere \
                     after {within:?}: the supervisor, and every child of it, must close it \
                     once they are written"
                )
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => bail!(
                "the secrets didn't arrive within {within:?}, nor did the pipe's write end \
                 close"
            ),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                bail!("reading the secrets stopped without an answer")
            }
        };
        Self::parse(&body)
    }

    /// The framed payload, and then the end of the pipe: the supervisor
    /// writes it once and closes its end, so anything after it was written
    /// by something else.
    fn read_frame(
        file: &mut std::fs::File,
        framed: &std::sync::atomic::AtomicBool,
    ) -> std::result::Result<Vec<u8>, Unread> {
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
        framed.store(true, std::sync::atomic::Ordering::Release);
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
        let Pairs(raw) = serde_json::from_slice(body).map_err(|e| {
            eyre!(
                "the secrets aren't a JSON object of strings (line {}, column {})",
                e.line(),
                e.column()
            )
        })?;
        let mut inherited = Self::default();
        let mut seen = BTreeSet::new();
        for (name, value) in raw {
            let known = name == DATABASE_URL || name == API_KEYS;
            if known && !seen.insert(name.clone()) {
                bail!("{name} was handed over twice");
            }
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

/// A JSON object of strings, its names and values in the order given,
/// repeats kept, so a name given twice is seen rather than the last one
/// silently winning.
struct Pairs(Vec<(String, String)>);

impl<'de> serde::Deserialize<'de> for Pairs {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Visit;
        impl<'de> serde::de::Visitor<'de> for Visit {
            type Value = Pairs;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object of strings")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Pairs, A::Error> {
                let mut pairs = Vec::new();
                while let Some(pair) = map.next_entry::<String, String>()? {
                    pairs.push(pair);
                }
                Ok(Pairs(pairs))
            }
        }
        d.deserialize_map(Visit)
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
        assert!(err.contains("more was written"), "{err}");
        assert!(!err.contains("forged-fixture"), "{err}");
        for stray in 1..=3 {
            let mut trailer = framed(BOTH);
            trailer.extend(std::iter::repeat_n(b'x', stray));
            let err = format!("{:#}", through_pipe(&trailer).unwrap_err());
            assert!(err.contains("more was written"), "{stray}: {err}");
        }
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
    /// refused as such, beside a good database URL.
    #[test]
    fn a_key_list_that_is_empty_once_trimmed_is_refused() {
        for keys in [" ", ",", " , ,"] {
            let payload =
                format!(r#"{{"VELLEV_DATABASE_URL":"postgres://db","VELLEV_API_KEYS":"{keys}"}}"#);
            let err = format!(
                "{:#}",
                through_pipe(&framed(payload.as_bytes())).unwrap_err()
            );
            assert!(
                err.contains("VELLEV_API_KEYS was handed over with no key in it"),
                "{err}"
            );
        }
    }

    /// A database URL handed over blank is refused as such, not left for
    /// `checked()` to call missing.
    #[test]
    fn a_blank_database_url_is_refused() {
        for url in ["", " ", "\\t "] {
            let payload =
                format!(r#"{{"VELLEV_DATABASE_URL":"{url}","VELLEV_API_KEYS":"k-fixture"}}"#);
            let err = format!(
                "{:#}",
                through_pipe(&framed(payload.as_bytes())).unwrap_err()
            );
            assert!(
                err.contains("VELLEV_DATABASE_URL was handed over empty"),
                "{err}"
            );
        }
    }

    /// A name given twice is refused, as anything after the frame is: the
    /// supervisor writes each once. Only a known name is repeated.
    #[test]
    fn a_name_given_twice_is_refused() {
        let url_twice = br#"{"VELLEV_DATABASE_URL":"postgres://u:a-fixture@h/db","VELLEV_API_KEYS":"k","VELLEV_DATABASE_URL":"postgres://u:b-fixture@h/db"}"#;
        let err = format!("{:#}", through_pipe(&framed(url_twice)).unwrap_err());
        assert!(
            err.contains("VELLEV_DATABASE_URL was handed over twice"),
            "{err}"
        );
        assert!(
            !err.contains("a-fixture") && !err.contains("b-fixture"),
            "{err}"
        );
        let twice = br#"{"VELLEV_DATABASE_URL":"postgres://db","VELLEV_API_KEYS":"a-fixture","VELLEV_API_KEYS":"b-fixture"}"#;
        let err = format!("{:#}", through_pipe(&framed(twice)).unwrap_err());
        assert!(
            err.contains("VELLEV_API_KEYS was handed over twice"),
            "{err}"
        );
        assert!(
            !err.contains("a-fixture") && !err.contains("b-fixture"),
            "{err}"
        );
        let unknown = br#"{"postgres://u:pw-fixture@h/db":"x","postgres://u:pw-fixture@h/db":"y"}"#;
        let err = format!("{:#}", through_pipe(&framed(unknown)).unwrap_err());
        assert!(!err.contains("pw-fixture"), "{err}");
    }

    /// Blank values clap leaves from an empty or comma-only variable count
    /// as unset, as they do for `keys()` and `checked()`.
    #[test]
    fn blank_values_from_the_environment_are_not_a_second_source() {
        for (url, keys) in [("", vec![""]), ("  ", vec![" ", ""]), ("", vec![])] {
            let mut config = bare_config();
            config.database_url = url.into();
            config.api_keys = keys.into_iter().map(String::from).collect();
            through_pipe(&framed(BOTH))
                .unwrap()
                .apply(&mut config)
                .unwrap();
            assert_eq!(config.database_url, "postgres://from-pipe");
            assert_eq!(config.api_keys, ["k-fixture"]);
        }
    }

    /// A payload of exactly the most is taken, one byte more is refused.
    /// (That the most fits a default fresh pipe is held by the constant's
    /// own compile-time check, not shown here.)
    #[test]
    fn the_most_a_payload_may_hold_is_taken_and_a_byte_more_refused() {
        let mut most = BOTH.to_vec();
        most.resize(MAX_BYTES, b' ');
        let read = through_pipe(&framed(&most)).unwrap();
        assert_eq!(read.api_keys, ["k-fixture"]);
        let mut over = most;
        over.push(b' ');
        let err = format!("{:#}", through_pipe(&framed(&over)).unwrap_err());
        assert!(err.contains("over the most the secrets may hold"), "{err}");
    }

    /// A length shorter than the payload is reported as the frame's length
    /// being wrong, not only as more being written.
    #[test]
    fn a_length_shorter_than_the_payload_is_named() {
        let mut short = (BOTH.len() as u32 - 1).to_be_bytes().to_vec();
        short.extend_from_slice(BOTH);
        let err = format!("{:#}", through_pipe(&short).unwrap_err());
        assert!(err.contains("framed length"), "{err}");
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
            // serde_json's own message would quote the whole string.
            (
                framed(br#""postgres://u:pw-fixture@h/db""#),
                "line 1, column",
            ),
            (
                ((MAX_BYTES as u32) + 1).to_be_bytes().to_vec(),
                "over the most the secrets may hold",
            ),
        ] {
            let err = format!("{:#}", through_pipe(&bad).unwrap_err());
            assert!(err.contains(reason), "{reason}: {err}");
            assert!(
                !err.contains("k-fixture")
                    && !err.contains("12345678")
                    && !err.contains("pw-fixture"),
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
        let silent = Inherited::read(reader.into(), Duration::from_secs(1))
            .unwrap_err()
            .to_string();
        assert!(silent.contains("didn't arrive within"), "{silent}");
        drop(writer);
        // The payload is in, but the pipe never ends: the error says so,
        // as this is a supervisor (or a child of it) still holding a copy
        // of the write end, not one that never wrote.
        let (reader, mut writer) = std::io::pipe().unwrap();
        writer.write_all(&framed(BOTH)).unwrap();
        let held = Inherited::read(reader.into(), Duration::from_secs(1))
            .unwrap_err()
            .to_string();
        assert!(held.contains("write end is still open"), "{held}");
        assert!(!held.contains("didn't arrive"), "{held}");
        drop(writer);
        // Only part of the payload, and the pipe held: the secrets did not
        // arrive.
        let (reader, mut writer) = std::io::pipe().unwrap();
        let whole = framed(BOTH);
        writer.write_all(&whole[..whole.len() - 3]).unwrap();
        let part = Inherited::read(reader.into(), Duration::from_secs(1))
            .unwrap_err()
            .to_string();
        assert!(part.contains("didn't arrive within"), "{part}");
        drop(writer);
    }

    /// Closed by the time the read returns, not some time after on the
    /// reader's thread: many times over, so a close that races the return
    /// is seen. Told by the pipe's inode, as the number may be reused.
    #[test]
    fn the_pipe_is_closed_by_the_time_the_read_returns() {
        use std::os::fd::{AsRawFd, OwnedFd};
        use std::os::unix::fs::MetadataExt;
        let inode = |fd: RawFd| {
            std::fs::metadata(format!("/dev/fd/{fd}"))
                .ok()
                .map(|m| (m.dev(), m.ino()))
        };
        for _ in 0..500 {
            let (reader, mut writer) = std::io::pipe().unwrap();
            writer.write_all(&framed(BOTH)).unwrap();
            drop(writer);
            // A second read end, held until the check, keeps the pipe in
            // being, so no pipe made alongside can come to share its inode.
            let pin = reader.try_clone().unwrap();
            let reader = OwnedFd::from(reader);
            let fd = reader.as_raw_fd();
            let pipe = inode(fd);
            assert!(pipe.is_some());
            Inherited::read(reader, Duration::from_secs(5)).unwrap();
            assert!(inode(fd).is_none_or(|now| Some(now) != pipe));
            drop(pin);
        }
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
}
