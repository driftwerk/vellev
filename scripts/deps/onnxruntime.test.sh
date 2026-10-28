#!/usr/bin/env bash
# The rules onnxruntime.sh is held to. Fetches the x86_64 archive once from
# ort-sys's own CDN, then serves copies of it, good and bad, from files.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
script=$here/onnxruntime.sh
pins=$here/onnxruntime.tsv
target=x86_64-unknown-linux-gnu
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export RETRIES=0
failed=0

pass() { echo "ok: $*"; }
fail() {
  echo "FAILED: $*" >&2
  failed=1
}

# Run a command, expecting it to succeed, or to be refused saying `why`.
expect() {
  local what=$1
  shift
  if "$@" >"$work/out" 2>&1; then
    pass "$what"
  else
    fail "$what: $(cat "$work/out")"
  fi
}
refused() {
  local what=$1 why=$2
  shift 2
  if "$@" >"$work/out" 2>&1; then
    fail "$what: it succeeded: $(cat "$work/out")"
  elif grep -q -- "$why" "$work/out"; then
    pass "$what"
  else
    fail "$what: refused, but not as $why: $(cat "$work/out")"
  fi
}
gone() {
  if [ -e "$1" ]; then
    fail "something is left in $1"
  else
    pass "nothing is left of it"
  fi
}

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Pins as given, with one field of the x86_64 pin replaced.
pinned_but() {
  local field=$1 value=$2
  awk -F'\t' -v OFS='\t' -v target="$target" -v field="$field" -v value="$value" \
    '$1 == target { $field = value } { print }' "$pins" >"$work/pins.tsv"
  echo "$work/pins.tsv"
}

url=$(awk -F'\t' -v target="$target" '$1 == target { print $4 }' "$pins")
archive_sha=$(awk -F'\t' -v target="$target" '$1 == target { print $2 }' "$pins")

## Checked against the locked ort-sys.
expect "the pins are the locked ort-sys's own" "$script" check
dist=$(cd "$here/../.." && cargo metadata --format-version 1 --locked |
  jq -er '.packages[] | select(.name == "ort-sys") | .manifest_path' |
  xargs dirname)/build/download/dist.tsv
refused "an archive pinned to another hash" "but the locked ort-sys downloads" \
  env PINS="$(pinned_but 2 "${archive_sha//e/f}")" "$script" check
refused "an archive pinned from another url" "but the locked ort-sys downloads" \
  env PINS="$(pinned_but 4 "${url//1.28.0/1.27.0}")" "$script" check
grep -v "^$target	none	" "$dist" >"$work/dist.tsv"
refused "a target the locked ort-sys does not build" "has no $target build" \
  env DIST="$work/dist.tsv" "$script" check

## Fetched, and taken only as pinned.
mkdir -p "$work/served"
curl -fsSL --retry 5 -o "$work/served/good.tar.lzma2" "$url"
[ "$(sha256 "$work/served/good.tar.lzma2")" = "$archive_sha" ] || {
  echo "the CDN's archive is not the pinned one" >&2
  exit 1
}
cp "$work/served/good.tar.lzma2" "$work/served/tampered.tar.lzma2"
printf 'x' | dd of="$work/served/tampered.tar.lzma2" bs=1 seek=4096 conv=notrunc 2>/dev/null
served=file://$work/served

expect "the pinned archive unpacks to the pinned library" \
  env SOURCES="$served/good.tar.lzma2" "$script" fetch "$target" "$work/lib"
library_sha=$(awk -F'\t' -v target="$target" '$1 == target { print $3 }' "$pins")
if [ -f "$work/lib/libonnxruntime.a" ] && [ "$(sha256 "$work/lib/libonnxruntime.a")" = "$library_sha" ] &&
  [ "$(find "$work/lib" -mindepth 1 | wc -l)" -eq 1 ]; then
  pass "the library alone is in the directory given"
else
  fail "the library is not what was pinned, or not alone: $(ls -la "$work/lib")"
fi

refused "an archive that is not the pinned one" "no source gave the pinned archive" \
  env SOURCES="$served/tampered.tar.lzma2" "$script" fetch "$target" "$work/tampered"
gone "$work/tampered"

expect "a source that fails, then one that gives the pinned archive" \
  env SOURCES="$served/missing.tar.lzma2 $served/tampered.tar.lzma2 $served/good.tar.lzma2" \
  "$script" fetch "$target" "$work/fallback"

refused "an archive that unpacks to a library not pinned" "a library that is not the pinned one" \
  env PINS="$(pinned_but 3 "${library_sha//0/1}")" SOURCES="$served/good.tar.lzma2" \
  "$script" fetch "$target" "$work/unpinned"
gone "$work/unpinned"

# An archive, made here and pinned, holding more than the library.
mkdir -p "$work/more"
cp "$work/lib/libonnxruntime.a" "$work/more/"
echo extra >"$work/more/extra"
tar -c -f - -C "$work/more" libonnxruntime.a extra |
  xz --compress --stdout --format=raw --lzma2=dict=64MiB >"$work/served/more.tar.lzma2"
refused "an archive holding more than the library" "unpacks to more than libonnxruntime.a" \
  env PINS="$(pinned_but 2 "$(sha256 "$work/served/more.tar.lzma2")")" \
  SOURCES="$served/more.tar.lzma2" "$script" fetch "$target" "$work/more-out"
gone "$work/more-out"

exit "$failed"
