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

/// What a reader of one of the node's outputs sends: each line as it
/// comes, then how the read ended.
enum Said {
    Line(String),
    End(std::io::Result<()>),
}

/// Reads `from` line by line on a thread of its own, so a child saying a
/// lot can't block on a full pipe. Lines are read as bytes and shown
/// lossily, so one that isn't UTF-8 stops nothing.
fn lines(from: impl std::io::Read + Send + 'static) -> std::sync::mpsc::Receiver<Said> {
    use std::io::BufRead;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut from = std::io::BufReader::new(from);
        let mut line = Vec::new();
        loop {
            line.clear();
            let said = match from.read_until(b'\n', &mut line) {
                Ok(0) => Said::End(Ok(())),
                Ok(_) => Said::Line(String::from_utf8_lossy(&line).trim_end().to_owned()),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => Said::End(Err(e)),
            };
            let ended = matches!(said, Said::End(_));
            if tx.send(said).is_err() || ended {
                break;
            }
        }
    });
    rx
}

/// The rest of what `rx` says, after `got`, to its end: waited for ten
/// seconds at most, as the node has stopped and its ends close with it
/// unless something else kept a copy. Short of the end, it says why, with
/// what did come.
fn rest(
    rx: &std::sync::mpsc::Receiver<Said>,
    mut got: Vec<String>,
) -> Result<Vec<String>, (Vec<String>, String)> {
    use std::sync::mpsc::RecvTimeoutError;
    let end = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match rx.recv_timeout(end.saturating_duration_since(std::time::Instant::now())) {
            Ok(Said::Line(line)) => got.push(line),
            Ok(Said::End(Ok(()))) => return Ok(got),
            Ok(Said::End(Err(e))) => return Err((got, format!("reading it failed: {e}"))),
            Err(RecvTimeoutError::Timeout) => {
                return Err((got, "it stayed open after the node ended".into()));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err((got, "its reader died".into()));
            }
        }
    }
}

/// All of one output, or the test fails saying why it isn't.
fn whole(said: Result<Vec<String>, (Vec<String>, String)>, name: &str) -> String {
    match said {
        Ok(lines) => lines.join("\n"),
        Err((lines, why)) => panic!("the node's {name}: {why}; it said: {}", lines.join("\n")),
    }
}

/// Whatever of one output came, for a failure's message.
fn shown(said: &Result<Vec<String>, (Vec<String>, String)>) -> String {
    match said {
        Ok(lines) => lines.join("\n"),
        Err((lines, why)) => format!("{} [{why}]", lines.join("\n")),
    }
}

/// Starts the node with `payload` on descriptor 3 and only `env` in its
/// environment, and waits a minute at most for it to stop: a node that
/// wrongly took the payload would wait for its database for ever, so it is
/// then killed (and reaped) and the test fails, rather than hanging. Its
/// output is read while it runs, so a node saying a lot can't block on a
/// full pipe, and then to its end, for up to ten seconds each.
fn serve(payload: &[u8], env: &[(&str, &str)]) -> Output {
    let mut child = start(payload, env);
    let out = lines(child.stdout.take().unwrap());
    let err = lines(child.stderr.take().unwrap());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let (status, stopped) = loop {
        match child.try_wait() {
            Ok(Some(status)) => break (status, true),
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Ok(None) => {
                let _ = child.kill();
                break (child.wait().unwrap(), false);
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("waiting on the node failed: {e}");
            }
        }
    };
    let out = rest(&out, Vec::new());
    let err = rest(&err, Vec::new());
    assert!(
        stopped,
        "the node didn't stop within a minute and was killed: stderr: {} stdout: {}",
        shown(&err),
        shown(&out)
    );
    Output {
        status,
        stdout: whole(out, "stdout").into_bytes(),
        stderr: whole(err, "stderr").into_bytes(),
    }
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
    let mut child = start(
        br#"{"VELLEV_DATABASE_URL":"postgres://u:pw-fixture@127.0.0.1:1/db","VELLEV_API_KEYS":"k-fixture"}"#,
        &[],
    );
    let out = lines(child.stdout.take().unwrap());
    let err = lines(child.stderr.take().unwrap());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut said = Vec::new();
    let (listening, ended) = loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match out.recv_timeout(left) {
            Ok(Said::Line(line)) => {
                let done = line.contains("vellev listening");
                said.push(line);
                if done {
                    break (true, None);
                }
            }
            Ok(Said::End(how)) => break (false, Some(how)),
            Err(_) => break (false, None),
        }
    };
    let _ = child.kill();
    child.wait().unwrap();
    // Everything it said, to the end, not only what came before listening:
    // a secret logged after it would be missed otherwise.
    let said = match ended {
        Some(Ok(())) => Ok(said),
        Some(Err(e)) => Err((said, format!("reading it failed: {e}"))),
        None => rest(&out, said),
    };
    let err = rest(&err, Vec::new());
    assert!(
        listening,
        "never listened: stderr: {} stdout: {}",
        shown(&err),
        shown(&said)
    );
    let said = whole(said, "stdout");
    let err = whole(err, "stderr");
    for secret in ["pw-fixture", "k-fixture"] {
        assert!(!said.contains(secret), "stdout: {said}");
        assert!(!err.contains(secret), "stderr: {err}");
    }
}
