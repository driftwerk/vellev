#!/usr/bin/env bash
# Publishes one workspace crate and fails unless crates.io ends up serving
# the workspace version built from this commit. A version crates.io already
# has counts as published only when its package records this commit and no
# uncommitted changes, which is what a re-run after a partial release finds.
#
# With --check it publishes nothing, and fails unless the version is absent
# from crates.io or there from this commit: run over every crate before the
# first is published, it refuses a release that could only end partial.
set -euo pipefail

check=false
if [ "${1:-}" = "--check" ]; then
  check=true
  shift
fi
crate="$1"
commit="${GITHUB_SHA:?GITHUB_SHA must name the commit being released}"
index="${CRATES_IO_INDEX:-https://index.crates.io}"
downloads="${CRATES_IO_DOWNLOADS:-https://static.crates.io/crates}"

version=$(cargo metadata --no-deps --format-version 1 |
  jq -er --arg name "$crate" '.packages[] | select(.name == $name) | .version')

lower=$(printf '%s' "$crate" | tr '[:upper:]' '[:lower:]')
case ${#lower} in
  1) index_path="1/$lower" ;;
  2) index_path="2/$lower" ;;
  3) index_path="3/${lower:0:1}/$lower" ;;
  *) index_path="${lower:0:2}/${lower:2:2}/$lower" ;;
esac

# The crate's index entries, a JSON line per version, and nothing for a
# crate crates.io has never had; fails on any other answer.
index_entries() {
  local body status
  body=$(mktemp)
  status=$(curl -sS -o "$body" -w '%{http_code}' "$index/$index_path") || status=failed
  case $status in
    200) cat "$body" ;;
    404) ;;
    *) echo "::error::the crates.io index of $crate answered $status" >&2 ;;
  esac
  rm -f "$body"
  [ "$status" = 200 ] || [ "$status" = 404 ]
}

has_version() {
  jq -e --arg v "$version" 'select(.vers == $v)' <<<"$1" >/dev/null
}

in_index() {
  local entries
  entries=$(index_entries) && has_version "$entries"
}

# The published package's VCS record: the commit it was packaged from, and
# `dirty` when that tree had uncommitted changes.
published_vcs() {
  curl -fsSL "$downloads/$crate/$crate-$version.crate" |
    tar -xzOf - "$crate-$version/.cargo_vcs_info.json" |
    jq -ec '.git | select(.sha1 | type == "string")'
}

require_this_commit() {
  local vcs found
  vcs=$(published_vcs) || {
    echo "::error::$crate $version is on crates.io but its package names no commit"
    exit 1
  }
  found=$(jq -r '.sha1' <<<"$vcs")
  if jq -e '.dirty == true' <<<"$vcs" >/dev/null; then
    echo "::error::$crate $version on crates.io was packaged from $found with uncommitted changes; bump the version"
    exit 1
  fi
  if [ "$found" != "$commit" ]; then
    echo "::error::$crate $version on crates.io was built from $found, not $commit; bump the version"
    exit 1
  fi
  echo "$crate $version from $commit is already on crates.io"
  exit 0
}

if [ "$check" = true ]; then
  entries=$(index_entries)
  if [ -z "$entries" ]; then
    echo "::notice::$crate is new to crates.io: its first publish needs a CARGO_REGISTRY_TOKEN with the publish-new scope"
  elif has_version "$entries"; then
    require_this_commit
  fi
  echo "$crate $version is not on crates.io yet"
  exit 0
fi

if in_index; then
  require_this_commit
fi

if ! output=$(cargo publish -p "$crate" 2>&1); then
  echo "$output"
  if grep -q "already exists" <<<"$output"; then
    require_this_commit
  fi
  exit 1
fi
echo "$output"

for _ in $(seq 1 30); do
  if in_index; then
    echo "$crate $version is live on crates.io"
    exit 0
  fi
  sleep 10
done

echo "::error::$crate $version did not appear in the crates.io index"
exit 1
