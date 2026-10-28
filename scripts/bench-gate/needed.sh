#!/usr/bin/env bash
# Whether a change needs the benchmark gates: reads the paths it changes,
# one per line, and prints "needed" if any can alter what the gates measure
# (the node they build, the harness, the toolchain, the fixture they read,
# or the gates themselves), else "not needed". The gates run on every pull
# request and decide by this, so they can be required checks: a check
# filtered out by paths would never report.
set -euo pipefail
answer="not needed"
while IFS= read -r path; do
  case "$path" in
    crates/* | Cargo.toml | Cargo.lock | rust-toolchain.toml | .cargo/* | \
      Dockerfile | .dockerignore | scripts/bench-fixture/* | scripts/bench-gate/* | \
      .github/workflows/bench-gate.yml)
      answer=needed
      ;;
  esac
done
echo "$answer"
