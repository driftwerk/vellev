//! The node's secrets, handed over by a supervisor on an inherited pipe
//! (`vellev serve --secrets-fd <n>`) rather than in its environment.
//!
//! A process's starting environment is readable by every other process of
//! the same user (`ps -E`), so a supervisor on someone's own computer (Desk
//! Helper) writes the database URL, which holds Postgres's password, and the
//! admin keys to a pipe instead. The payload is its length (4 bytes,
//! big-endian), then one JSON object holding `VELLEV_DATABASE_URL` and
//! `VELLEV_API_KEYS` (comma-separated), as the environment would. They take
//! the place of whatever the command line or the environment gave.
//!
//! The descriptor must be a pipe. The read stops once the framed length is
//! in, so a copy of the write end held open elsewhere can't make it wait,
//! and a supervisor that never writes is an error at the deadline. Errors
//! name a variable or a position, never a value.

use std::collections::BTreeMap;
use std::io::Read;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
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
        std::thread::spawn(move || {
            let read = (|| -> std::io::Result<Vec<u8>> {
                let mut header = [0u8; 4];
                file.read_exact(&mut header)?;
                let len = u32::from_be_bytes(header) as usize;
                if len > MAX_BYTES {
                    return Err(std::io::Error::other("too long"));
                }
                let mut body = vec![0u8; len];
                file.read_exact(&mut body)?;
                Ok(body)
            })();
            let _ = tx.send(read);
        });
        let body = match rx.recv_timeout(within) {
            Ok(Ok(body)) => body,
            Ok(Err(e)) if e.to_string() == "too long" => {
                bail!("the secrets are over {MAX_BYTES} bytes")
            }
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                bail!("the secrets ended before their length")
            }
            Ok(Err(e)) => bail!("reading the secrets: {e}"),
            Err(_) => bail!("the secrets didn't arrive within {within:?}"),
        };
        Self::parse(&body)
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
            if value.is_empty() {
                bail!("{name} was handed over empty");
            }
            match name.as_str() {
                DATABASE_URL => inherited.database_url = Some(value),
                API_KEYS => {
                    inherited.api_keys = value
                        .split(',')
                        .map(|k| k.trim().to_string())
                        .filter(|k| !k.is_empty())
                        .collect();
                }
                _ => bail!("{name:?} isn't a secret the node takes on a pipe"),
            }
        }
        Ok(inherited)
    }

    /// Puts what was handed over in place of what the command line or the
    /// environment gave.
    pub fn apply(self, config: &mut Config) {
        if let Some(url) = self.database_url {
            config.database_url = url;
        }
        if !self.api_keys.is_empty() {
            config.api_keys = self.api_keys;
        }
    }
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
            (&br#"{"VELLEV_API_KEYS":"k-fixture"}"#[..], "VELLEV_DATABASE_URL"),
            (br#"{"VELLEV_DATABASE_URL":"postgres://db"}"#, "VELLEV_API_KEYS"),
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
            let err = format!("{:#}", through_pipe(&framed(payload.as_bytes())).unwrap_err());
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
            (framed(br#"{"VELLEV_API_KEYS": 12345678}"#), "line 1, column"),
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

    #[test]
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
}
