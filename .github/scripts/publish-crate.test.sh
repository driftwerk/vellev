#!/usr/bin/env bash
# Runs publish-crate.sh --check against a crates.io index and downloads
# served from fixtures, one case per answer crates.io can give.
set -euo pipefail

cd "$(dirname "$0")/../.."
script=.github/scripts/publish-crate.sh
crate=vellev-core
version=$(cargo metadata --no-deps --format-version 1 |
  jq -er --arg name "$crate" '.packages[] | select(.name == $name) | .version')
release=1111111111111111111111111111111111111111
other=2222222222222222222222222222222222222222

work=$(mktemp -d)
served="$work/served"
mkdir -p "$served"
server=
cleanup() {
  if [ -n "$server" ]; then
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT

python3 - "$served" "$work/port" <<'EOF' &
import functools, http.server, os, sys

class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

server = http.server.ThreadingHTTPServer(
    ("127.0.0.1", 0), functools.partial(Quiet, directory=sys.argv[1])
)
with open(sys.argv[2] + ".tmp", "w") as port:
    port.write(str(server.server_port))
os.replace(sys.argv[2] + ".tmp", sys.argv[2])
server.serve_forever()
EOF
server=$!
for _ in $(seq 1 50); do
  [ -s "$work/port" ] && break
  sleep 0.1
done
[ -s "$work/port" ] || { echo "the fixture server did not start" >&2; exit 1; }
base="http://127.0.0.1:$(cat "$work/port")"

# Serves the index entries given, a version each, and the package of
# `version` with the VCS record given (none when empty).
serve() {
  local vcs="$1" package="$work/package/$crate-$version" entries="$served/index/${crate:0:2}/${crate:2:2}" vers
  shift
  rm -rf "${served:?}"/* "$work/package"
  if [ "$#" -gt 0 ]; then
    mkdir -p "$entries"
    for vers in "$@"; do
      printf '{"name":"%s","vers":"%s"}\n' "$crate" "$vers"
    done >"$entries/$crate"
  fi
  mkdir -p "$package" "$served/crates/$crate"
  printf '[package]\nname = "%s"\n' "$crate" >"$package/Cargo.toml"
  [ -z "$vcs" ] || printf '%s' "$vcs" >"$package/.cargo_vcs_info.json"
  tar -czf "$served/crates/$crate/$crate-$version.crate" -C "$work/package" "$crate-$version"
}

failures=0

# Runs the check against `index` and expects its exit status and a line of
# its output.
expect() {
  local case="$1" index="$2" status="$3" line="$4" output got=0
  output=$(GITHUB_SHA=$release CRATES_IO_INDEX="$index" CRATES_IO_DOWNLOADS="$base/crates" \
    "$script" --check "$crate" 2>&1) || got=$?
  if [ "$got" = "$status" ] && grep -qF -- "$line" <<<"$output"; then
    echo "ok: $case"
  else
    printf 'FAILED: %s (exit %s, expected %s with "%s")\n%s\n' \
      "$case" "$got" "$status" "$line" "$output"
    failures=$((failures + 1))
  fi
}

clean() {
  printf '{"git":{"sha1":"%s"},"path_in_vcs":"crates/vellev-core"}' "$1"
}

serve ""
expect "a crate crates.io has never had" "$base/index" 0 "is new to crates.io"

serve "" 0.0.1 0.0.2
expect "other versions only" "$base/index" 0 "$crate $version is not on crates.io yet"

serve "$(clean $release)" "$version"
expect "this version from this commit" "$base/index" 0 "$crate $version from $release is already on crates.io"

serve "{\"git\":{\"sha1\":\"$release\",\"dirty\":false}}" 0.0.1 "$version"
expect "this version from this commit, recorded clean" "$base/index" 0 "is already on crates.io"

serve "{\"git\":{\"sha1\":\"$release\",\"dirty\":true},\"path_in_vcs\":\"crates/vellev-core\"}" "$version"
expect "this version from this commit with uncommitted changes" "$base/index" 1 \
  "was packaged from $release with uncommitted changes"

serve "$(clean $other)" "$version"
expect "this version from another commit" "$base/index" 1 "was built from $other, not $release"

serve "" "$version"
expect "this version with no VCS record" "$base/index" 1 "its package names no commit"

serve '{"path_in_vcs":"crates/vellev-core"}' "$version"
expect "this version with no commit in its VCS record" "$base/index" 1 "its package names no commit"

expect "an index that does not answer" "http://127.0.0.1:1" 1 "the crates.io index of $crate answered"

[ "$failures" = 0 ] || { echo "$failures case(s) failed" >&2; exit 1; }
