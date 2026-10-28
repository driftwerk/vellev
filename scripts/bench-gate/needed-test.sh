#!/usr/bin/env bash
# The rule the benchmark gates are run by: any change that can alter the
# node the gate builds, the harness, or the fixture it reads needs them;
# anything else does not. Run before each gate decides.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
failed=0
expect() {
  local want="$1"
  shift
  local got
  got="$(printf '%s\n' "$@" | "$here/needed.sh")"
  if [ "$got" != "$want" ]; then
    echo "expected '$want' for: $*, got '$got'" >&2
    failed=1
  fi
}
for path in \
  crates/vellev/src/server.rs crates/vellev-node/src/lib.rs crates/vellev-core/src/lib.rs \
  crates/vellev-storage/src/state.rs crates/vellev-index/src/clustered.rs crates/vellev-bench/src/main.rs \
  crates/vellev-embed/src/lib.rs crates/vellev-api/src/lib.rs crates/vellev-client/src/lib.rs \
  Dockerfile .dockerignore Cargo.toml Cargo.lock rust-toolchain.toml .cargo/config.toml \
  scripts/bench-fixture/SHA256SUMS scripts/bench-gate/needed.sh .github/workflows/bench-gate.yml; do
  expect needed "$path"
  expect needed README.md "$path"
done
expect "not needed" README.md
expect "not needed" docs/design.md .github/workflows/ci.yml scripts/recall-fixture/make.py
expect "not needed" Dockerfile.notes/x crates.md
expect "not needed"
exit "$failed"
