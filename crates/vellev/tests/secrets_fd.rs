//! `vellev serve --secrets-fd 3` as a supervisor really starts it: the
//! read end of a pipe put on descriptor 3 across an exec (as `dup2` leaves
//! it, not close-on-exec), with the payload written and the write end
//! closed before the node starts.
#![cfg(unix)]

use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::process::{Command, Output};

fn framed(payload: &[u8]) -> Vec<u8> {
    let mut out = (payload.len() as u32).to_be_bytes().to_vec();
    out.extend_from_slice(payload);
    out
}

/// Held from making a pipe until the child using it has started: on macOS
/// a pipe is made close-on-exec only after it is made, so a test starting
/// a child alongside could inherit another's write end, and that node
/// would never see its pipe end (as a supervisor's `SPAWNS` guards).
static SPAWNS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Starts the node with `payload` on descriptor 3 and only `env` in its
/// environment, and waits for it to stop.
fn serve(payload: &[u8], env: &[(&str, &str)]) -> Output {
    let spawns = SPAWNS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (reader, mut writer) = std::io::pipe().unwrap();
    writer.write_all(&framed(payload)).unwrap();
    drop(writer);
    let from = reader.as_raw_fd();
    let mut command = Command::new(env!("CARGO_BIN_EXE_vellev"));
    command
        .args(["serve", "--store", "memory://", "--secrets-fd", "3"])
        .env_clear()
        .env("VELLEV_DATABASE_TIMEOUT_SECS", "2")
        .envs(env.iter().copied());
    // SAFETY: only `dup2` and `fcntl`, which are async-signal-safe, run
    // between fork and exec.
    unsafe {
        command.pre_exec(move || {
            let placed = if from == 3 {
                libc::fcntl(3, libc::F_SETFD, 0)
            } else {
                libc::dup2(from, 3)
            };
            if placed == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    drop(reader);
    drop(spawns);
    child.wait_with_output().unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A payload the node refuses fails the start with status 1, which a
/// supervisor tells from clap's 2 (a node too old for the flag), and the
/// reason names what is missing.
#[test]
fn a_refused_payload_fails_the_start_saying_why() {
    let output = serve(br#"{"VELLEV_API_KEYS":"k-fixture"}"#, &[]);
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let err = stderr(&output);
    assert!(err.contains("don't hold VELLEV_DATABASE_URL"), "{err}");
    assert!(!err.contains("k-fixture"), "{err}");
}

/// A secret also in the environment is refused, naming the variable only.
#[test]
fn a_secret_also_in_the_environment_is_refused() {
    let output = serve(
        br#"{"VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db","VELLEV_API_KEYS":"k-fixture"}"#,
        &[("VELLEV_API_KEYS", "env-key-fixture")],
    );
    assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
    let err = stderr(&output);
    assert!(err.contains("VELLEV_API_KEYS is also set"), "{err}");
    for secret in ["pw-fixture", "k-fixture", "env-key-fixture"] {
        assert!(!err.contains(secret), "{err}");
    }
}

/// A good payload is taken: the node gets past the pipe to whatever it
/// does next (here, a database it can't reach), and neither secret shows
/// in what it says.
#[test]
fn a_good_payload_is_taken_and_never_repeated() {
    let output = serve(
        br#"{"VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db","VELLEV_API_KEYS":"k-fixture"}"#,
        &[],
    );
    let err = stderr(&output);
    assert!(!output.status.success(), "{err}");
    assert!(!err.contains("secrets"), "{err}");
    assert!(!err.contains("--secrets-fd"), "{err}");
    for secret in ["pw-fixture", "k-fixture"] {
        assert!(!err.contains(secret), "{err}");
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains(secret),
            "stdout"
        );
    }
}
