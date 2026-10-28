#!/usr/bin/env bash
# The paths changed between two commits, one per line, for needed.sh. A
# rename is listed as both its old path and its new one (--no-renames), so
# a file moved out of what the gates measure still counts as changing it.
set -euo pipefail
git diff --no-renames --name-only "$1" "$2"
