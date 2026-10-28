#!/usr/bin/env bash
# The ONNX Runtime that local embedding links, from a pinned copy rather
# than from whatever ort-sys's build script would download.
#
#   onnxruntime.sh fetch TARGET DIR   unpack TARGET's pinned library into DIR,
#                                     for ORT_LIB_LOCATION=DIR
#   onnxruntime.sh download TARGET FILE
#                                     save TARGET's pinned archive as FILE
#   onnxruntime.sh check              fail unless the pins are the locked
#                                     ort-sys's own
#   onnxruntime.sh mirror             upload the pinned archives to the deps
#                                     release (a maintainer, once a version)
#
# A fetch tries our release first and ort-sys's own CDN after, and takes an
# archive only if its SHA-256 is pinned, and the library only if what it
# unpacks to is pinned too: two sources, one artifact.
#
# PINS, DIST, METADATA, FEATURES, SOURCES, RETRIES and REPOSITORY may be
# set to test it.
set -euo pipefail
# A failure in $(...) fails what it is in, not only the subshell.
shopt -s inherit_errexit

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PINS=${PINS:-$here/onnxruntime.tsv}
REPOSITORY=${REPOSITORY:-driftwerk/vellev}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

die() {
  echo "onnxruntime.sh: $*" >&2
  exit 1
}

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# The pin of TARGET: archive sha256, library sha256 and ort-sys's url.
pin() {
  local found
  found=$(grep -v '^#' "$PINS" | awk -F'\t' -v target="$1" '$1 == target { print $2, $3, $4 }')
  [ -n "$found" ] || die "no pin for $1 in $PINS"
  [ "$(wc -l <<<"$found")" -eq 1 ] || die "more than one pin for $1 in $PINS"
  echo "$found"
}

# ONNX Runtime's version, as ort-sys's url names it (ms@1.28.0).
version_of() {
  local version=${1#*/ms@}
  version=${version%%/*}
  [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "no version in $1"
  echo "$version"
}

release_of() {
  echo "deps-onnxruntime-$(version_of "$1")"
}

# Where an archive may be had, in order: our release, then ort-sys's CDN.
sources() {
  local target=$1 url=$2
  if [ -n "${SOURCES:-}" ]; then
    echo "${SOURCES//\{target\}/$target}" | tr ' ' '\n'
  else
    echo "https://github.com/$REPOSITORY/releases/download/$(release_of "$url")/$target.tar.lzma2"
    echo "$url"
  fi
}

# Fetch TARGET's pinned archive to WORK/TARGET.tar.lzma2 and unpack its
# pinned library into DIR.
fetch() {
  local target=$1 dir=$2 archive_sha library_sha url work source got
  local pinned
  pinned=$(pin "$target") || exit 1
  read -r archive_sha library_sha url <<<"$pinned"
  work=$WORK/$target
  rm -rf "$work"
  mkdir -p "$work"
  got=
  for source in $(sources "$target" "$url"); do
    rm -f "$work/archive"
    # A transfer that stalls, as pyke's CDN's have from CI runners, is cut
    # off and tried again rather than waited out.
    if ! curl -fsSL --retry "${RETRIES:-5}" --retry-connrefused --connect-timeout 20 --max-time 300 \
      --speed-limit 50000 --speed-time 30 -o "$work/archive" "$source"; then
      echo "onnxruntime.sh: could not fetch $source" >&2
      continue
    fi
    if [ "$(sha256 "$work/archive")" != "$archive_sha" ]; then
      echo "onnxruntime.sh: $source is not the pinned archive" >&2
      continue
    fi
    got=$source
    break
  done
  [ -n "$got" ] || die "no source gave the pinned archive for $target"
  # A raw LZMA2 stream, as ort-sys reads it with a 64 MiB dictionary, of a
  # tar holding the static library alone.
  mkdir -p "$work/unpacked"
  xz --decompress --stdout --format=raw --lzma2=dict=64MiB "$work/archive" |
    tar -x -f - -C "$work/unpacked" --no-same-owner --no-same-permissions
  [ "$(find "$work/unpacked" -mindepth 1 | wc -l)" -eq 1 ] &&
    [ -f "$work/unpacked/libonnxruntime.a" ] ||
    die "$target's archive unpacks to more than libonnxruntime.a"
  [ "$(sha256 "$work/unpacked/libonnxruntime.a")" = "$library_sha" ] ||
    die "$target's archive unpacks to a library that is not the pinned one"
  mkdir -p "$dir"
  mv "$work/unpacked/libonnxruntime.a" "$dir/libonnxruntime.a"
  mv "$work/archive" "$WORK/$target.tar.lzma2"
  echo "onnxruntime.sh: $target's pinned ONNX Runtime $(version_of "$url"), from $got, in $dir"
}

# The workspace's metadata, as Cargo.lock resolves it, read once.
metadata() {
  if [ ! -f "$WORK/metadata.json" ]; then
    if [ -n "${METADATA:-}" ]; then
      cat "$METADATA" >"$WORK/metadata.json"
    else
      (cd "$here/../.." && cargo metadata --format-version 1 --locked) >"$WORK/metadata.json"
    fi
  fi
  cat "$WORK/metadata.json"
}

# The locked ort-sys's package id.
ort_sys() {
  local id
  id=$(metadata | jq -r '.packages[]? | select(.name == "ort-sys") | .id')
  [ -n "$id" ] || die "no ort-sys in the locked dependencies"
  [ "$(wc -l <<<"$id")" -eq 1 ] || die "more than one ort-sys is locked"
  echo "$id"
}

# ort-sys's dist.tsv, of the version Cargo.lock holds.
locked_dist() {
  local id manifest
  id=$(ort_sys) || exit 1
  manifest=$(metadata | jq -er --arg id "$id" '.packages[] | select(.id == $id) | .manifest_path')
  echo "$(dirname "$manifest")/build/download/dist.tsv"
}

# The features ort-sys is built with, across the workspace.
features() {
  local id
  if [ -n "${FEATURES:-}" ]; then
    echo "$FEATURES"
  else
    id=$(ort_sys) || exit 1
    metadata | jq -r --arg id "$id" '.resolve.nodes[] | select(.id == $id) | .features[]'
  fi
}

# ort-sys's features that make it download an ONNX Runtime other than the
# one of feature set "none" (its build/download/resolve.rs), which is all
# that is pinned.
OTHER_BUILDS="training cuda tensorrt openvino onednn directml nnapi coreml xnnpack rocm acl armnn
tvm migraphx rknpu vitis cann qnn webgpu azure nvrtx vsinpu"

check() {
  local dist built feature target archive_sha library_sha url theirs checked=0
  built=$(features) || exit 1
  for feature in $built; do
    for other in $OTHER_BUILDS; do
      [ "$feature" != "$other" ] ||
        die "ort-sys is built with $feature, which downloads another ONNX Runtime than the pinned one"
    done
  done
  if [ -n "${DIST:-}" ]; then
    dist=$DIST
  else
    dist=$(locked_dist) || exit 1
  fi
  [ -f "$dist" ] || die "no $dist"
  while IFS=$'\t' read -r target archive_sha library_sha url; do
    [[ -z $target || $target == \#* ]] && continue
    theirs=$(awk -F'\t' -v target="$target" '$1 == target && $2 == "none" { print $3, $4 }' "$dist")
    [ -n "$theirs" ] || die "the locked ort-sys has no $target build"
    [ "$theirs" = "$url $archive_sha" ] ||
      die "$target is pinned to $url $archive_sha, but the locked ort-sys downloads $theirs: fetch and pin its archive"
    checked=$((checked + 1))
  done <"$PINS"
  [ "$checked" -gt 0 ] || die "no pins in $PINS"
  echo "onnxruntime.sh: $checked pins are the locked ort-sys's own"
}

# TARGET's pinned archive, as fetched and checked, saved as FILE.
download() {
  fetch "$1" "$WORK/unpacked-$1" >&2
  cp "$WORK/$1.tar.lzma2" "$2"
}

# The release must exist; it is made by hand, not as the latest release.
mirror() {
  local target archive_sha library_sha url
  while IFS=$'\t' read -r target archive_sha library_sha url; do
    [[ -z $target || $target == \#* ]] && continue
    SOURCES=$url download "$target" "$WORK/$target.upload"
    mv "$WORK/$target.upload" "$WORK/$target.tar.lzma2"
    gh release upload --repo "$REPOSITORY" "$(release_of "$url")" "$WORK/$target.tar.lzma2"
  done <"$PINS"
}

case "${1:-}" in
  fetch)
    [ $# -eq 3 ] || die "usage: onnxruntime.sh fetch TARGET DIR"
    fetch "$2" "$3"
    ;;
  download)
    [ $# -eq 3 ] || die "usage: onnxruntime.sh download TARGET FILE"
    download "$2" "$3"
    ;;
  check) check ;;
  mirror) mirror ;;
  *) die "usage: onnxruntime.sh fetch TARGET DIR | download TARGET FILE | check | mirror" ;;
esac
