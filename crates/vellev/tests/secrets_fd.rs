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
/// environment, and waits for it to stop, for a minute at most: a node
/// that wrongly took the payload would wait for its database for ever, so
/// it is then killed (and reaped) and the test fails, rather than hanging.
fn serve(payload: &[u8], env: &[(&str, &str)]) -> Output {
    let mut child = start(payload, env);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let stopped = loop {
        if child.try_wait().unwrap().is_some() {
            break true;
        }
        if std::time::Instant::now() >= deadline {
            break false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    if !stopped {
        let _ = child.kill();
    }
    let output = child.wait_with_output().unwrap();
    assert!(stopped, "the node didn't stop: {}", stderr(&output));
    output
}

/// Starts the node with `payload` on descriptor 3, listening on a port of
/// its own, with only `env` in its environment.
fn start(payload: &[u8], env: &[(&str, &str)]) -> std::process::Child {
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
        .env("VELLEV_LISTEN", "127.0.0.1:0")
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
    child
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

/// A good payload is taken: the node gets past the pipe, its checks and
/// its keys to listening (then it is stopped, as it would wait for its
/// database for ever), and neither secret shows in anything it said.
#[test]
fn a_good_payload_is_taken_and_never_repeated() {
    use std::io::BufRead;
    let mut child = start(
        br#"{"VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db","VELLEV_API_KEYS":"k-fixture"}"#,
        &[],
    );
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut said = Vec::new();
    let listening = loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(left) {
            Ok(line) => {
                let done = line.contains("vellev listening");
                said.push(line);
                if done {
                    break true;
                }
            }
            Err(_) => break false,
        }
    };
    let _ = child.kill();
    let output = child.wait_with_output().unwrap();
    said.extend(rx.try_iter());
    let err = stderr(&output);
    assert!(listening, "never listened: {err} {said:?}");
    let said = said.join("\n");
    for secret in ["pw-fixture", "k-fixture"] {
        assert!(!said.contains(secret), "stdout: {said}");
        assert!(!err.contains(secret), "stderr: {err}");
    }
}
