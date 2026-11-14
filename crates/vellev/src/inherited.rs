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
    pub database_url: String,
    pub api_keys: Vec<String>,
}

impl std::fmt::Debug for Inherited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inherited")
            .field("database_url", &!self.database_url.is_empty())
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
                    return Err(std::io::Error::from(std::io::ErrorKind::InvalidData));
                }
                let mut body = vec![0u8; len];
                file.read_exact(&mut body)?;
                Ok(body)
            })();
            let _ = tx.send(read);
        });
        let body = match rx.recv_timeout(within) {
            Ok(Ok(body)) => body,
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::InvalidData => {
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
        let Pairs(raw) = serde_json::from_slice(body).map_err(|e| {
            eyre!(
                "the secrets aren't a JSON object of strings, each named once (line {}, column {})",
                e.line(),
                e.column()
            )
        })?;
        let (mut database_url, mut api_keys) = (None, None);
        for (name, value) in raw {
            if value.is_empty() {
                bail!("{name} was handed over empty");
            }
            let seen = match name.as_str() {
                DATABASE_URL => database_url.replace(value).is_some(),
                API_KEYS => api_keys.replace(value).is_some(),
                _ => bail!("{name:?} isn't a secret the node takes on a pipe"),
            };
            if seen {
                bail!("{name} was handed over twice");
            }
        }
        // Both are required: a supervisor that drops one must not leave the
        // node on whatever its environment or command line holds.
        let Some(database_url) = database_url else {
            bail!("{DATABASE_URL} wasn't handed over");
        };
        let Some(api_keys) = api_keys else {
            bail!("{API_KEYS} wasn't handed over");
        };
        let api_keys: Vec<String> = api_keys
            .split(',')
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .collect();
        if api_keys.is_empty() {
            bail!("{API_KEYS} was handed over with no key in it");
        }
        Ok(Self {
            database_url,
            api_keys,
        })
    }

    /// Puts what was handed over in place of what the command line or the
    /// environment gave.
    pub fn apply(self, config: &mut Config) {
        config.database_url = self.database_url;
        config.api_keys = self.api_keys;
    }
}

/// A JSON object's members in the order given, with none dropped: serde's
/// maps keep only the last of a repeated name.
struct Pairs(Vec<(String, String)>);

impl<'de> serde::Deserialize<'de> for Pairs {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Pairs;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object of strings")
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
        d.deserialize_map(V)
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
            br#"{"VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db","VELLEV_API_KEYS":"k1-fixture,k2-fixture"}"#,
        ))
        .unwrap();
        assert_eq!(read.database_url, "postgres://u:pw-fixture@127.0.0.1:1/db");
        assert_eq!(read.api_keys, ["k1-fixture", "k2-fixture"]);
        let shown = format!("{read:?}");
        assert!(
            !shown.contains("pw-fixture") && !shown.contains("k1-fixture"),
            "{shown}"
        );
    }

    #[test]
    fn they_replace_what_the_command_line_and_environment_gave() {
        let mut config = crate::config::tests::config(&[]);
        let read = through_pipe(&framed(
            br#"{"VELLEV_DATABASE_URL":"postgres://from-pipe","VELLEV_API_KEYS":"k-fixture"}"#,
        ))
        .unwrap();
        read.apply(&mut config);
        assert_eq!(config.database_url, "postgres://from-pipe");
        assert_eq!(config.api_keys, ["k-fixture"]);
    }

    #[test]
    fn anything_but_the_two_names_is_refused() {
        let err = through_pipe(&framed(br#"{"AWS_SECRET_ACCESS_KEY":"s-fixture"}"#))
            .unwrap_err()
            .to_string();
        assert!(err.contains("AWS_SECRET_ACCESS_KEY"), "{err}");
        assert!(!err.contains("s-fixture"), "{err}");
    }

    #[test]
    fn malformed_or_short_payloads_are_refused_without_repeating_them() {
        let mut short = framed(
            br#"{"VELLEV_DATABASE_URL":"postgres://x-fixture","VELLEV_API_KEYS":"k-fixture"}"#,
        );
        short.truncate(short.len() - 2);
        for bad in [
            short,
            framed(br#"{"VELLEV_API_KEYS": 12345678}"#),
            framed(b"VELLEV_API_KEYS=k-fixture"),
            ((MAX_BYTES as u32) + 1).to_be_bytes().to_vec(),
        ] {
            let err = through_pipe(&bad).unwrap_err().to_string();
            assert!(
                !err.contains("k-fixture") && !err.contains("12345678"),
                "{err}"
            );
        }
    }

    #[test]
    fn a_silent_pipe_times_out_and_a_held_write_end_does_not_hang() {
        let (reader, writer) = std::io::pipe().unwrap();
        let err = Inherited::read(reader.into(), Duration::from_millis(200))
            .unwrap_err()
            .to_string();
        assert!(err.contains("within"), "{err}");
        drop(writer);
        let (reader, mut writer) = std::io::pipe().unwrap();
        writer
            .write_all(&framed(
                br#"{"VELLEV_DATABASE_URL":"postgres://x-fixture","VELLEV_API_KEYS":"k-fixture"}"#,
            ))
            .unwrap();
        let read = Inherited::read(reader.into(), Duration::from_secs(5)).unwrap();
        assert_eq!(read.api_keys, ["k-fixture"]);
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

    const URL_FIXTURE: &str = r#""VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db""#;

    /// A supervisor that drops a secret must not leave the node on whatever
    /// the environment or command line happens to hold: both names are
    /// required, and the refusal names the one missing, never a value.
    #[test]
    fn a_payload_missing_a_secret_is_refused() {
        let keys = r#""VELLEV_API_KEYS":"k-fixture""#;
        for (payload, missing) in [
            ("{}".to_string(), "VELLEV_DATABASE_URL"),
            (format!("{{{keys}}}"), "VELLEV_DATABASE_URL"),
            (format!("{{{URL_FIXTURE}}}"), "VELLEV_API_KEYS"),
        ] {
            let err = through_pipe(&framed(payload.as_bytes()))
                .unwrap_err()
                .to_string();
            assert!(err.contains(missing), "{err}");
            assert!(!err.contains("fixture"), "{err}");
        }
    }

    /// Keys that are all blanks are no keys: the node isn't left on the
    /// environment's.
    #[test]
    fn a_key_list_with_no_key_in_it_is_refused() {
        for blank in [",", " ", " , ,"] {
            let payload = format!(r#"{{{URL_FIXTURE},"VELLEV_API_KEYS":"{blank}"}}"#);
            let err = through_pipe(&framed(payload.as_bytes()))
                .unwrap_err()
                .to_string();
            assert!(err.contains("VELLEV_API_KEYS"), "{err}");
            assert!(!err.contains("pw-fixture"), "{err}");
        }
    }

    /// Two values for one name are ambiguous, not "the last one".
    #[test]
    fn a_name_given_twice_is_refused() {
        let payload = format!(
            r#"{{{URL_FIXTURE},"VELLEV_API_KEYS":"k1-fixture","VELLEV_API_KEYS":"k2-fixture"}}"#
        );
        let err = through_pipe(&framed(payload.as_bytes()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("VELLEV_API_KEYS"), "{err}");
        assert!(
            !err.contains("k1-fixture") && !err.contains("k2-fixture"),
            "{err}"
        );
    }

    /// What `--secrets-fd` may name: not a standard stream, not a closed
    /// number, and not a file, which is left open for whoever owns it.
    #[test]
    fn a_descriptor_that_is_not_a_handed_over_pipe_is_refused_and_left_open() {
        use std::os::fd::AsRawFd;
        for fd in [0, 1, 2] {
            let err = Inherited::from_descriptor(fd).unwrap_err().to_string();
            assert!(err.contains("standard stream"), "{err}");
        }
        let err = Inherited::from_descriptor(1_000_000)
            .unwrap_err()
            .to_string();
        assert!(err.contains("pipe"), "{err}");
        let file = tempfile::tempfile().unwrap();
        let err = Inherited::from_descriptor(file.as_raw_fd())
            .unwrap_err()
            .to_string();
        assert!(err.contains("pipe"), "{err}");
        assert!(file.metadata().is_ok(), "the file's descriptor was closed");
    }

    /// An oversize payload is told apart by its kind, not by its words.
    #[test]
    fn an_oversize_payload_is_named_as_such() {
        let err = through_pipe(&((MAX_BYTES as u32) + 1).to_be_bytes())
            .unwrap_err()
            .to_string();
        assert!(err.contains("over"), "{err}");
    }
}
