#!/usr/bin/env bash
# The step 3 bench (merge note, §3 and §9), every arm one after another on
# this machine. Each arm is two processes run as the image runs (uid
# 10001, no capabilities, the default limits), each in a systemd scope of
# its own where systemd can make one: the node (step3_arm), and the load
# generator (step3_load), which finds the node through a file the node
# writes. The node's memory and CPU are then its own. Linux only.
#
#   scripts/step3-bench/run.sh FIXTURE OUT DATA [ARMS...]
#
# FIXTURE is the bench fixture unpacked (text/, vectors/), as the
# benchmark workflow unpacks it; OUT gets each arm's JSON and metrics, the
# machine's description, summary.md and verdict.json; DATA holds each
# arm's store while it runs, deleted after. ARMS defaults to "a b c d b2":
# b2 is (b) run again, its spread from (b) the noise floor.
# STEP3_* variables pass through to the bench (see crates/vellev/tests/
# step3_bench.rs); STEP3_SOAK_SECS below 1800 is a smoke, never compared.
#
# Needs root, through sudo or as root, to drop to uid 10001, and every
# path given where uid 10001 can reach it: not under a home it may not
# enter, such as a GitHub runner's $RUNNER_TEMP.
set -euo pipefail

fixture=$(realpath "$1")
out=$(realpath -m "$2")
data=$(realpath -m "$3")
shift 3
arms=("${@:-a b c d b2}")
read -r -a arms <<< "${arms[*]}"

as_root() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

mkdir -p "$out"
as_root install -d -o 10001 -g 10001 "$data"
as_root chmod -R a+rX "$fixture"
# Uid 10001 reaches all three, or the run stops here, saying which.
for path in "$fixture/vectors/passages" "$out" "$data"; do
  if ! as_root setpriv --reuid=10001 --regid=10001 --clear-groups test -x "$path"; then
    echo "uid 10001 cannot reach $path: give paths it can (see the top of this script)" >&2
    exit 1
  fi
done

# What the machine is, as the run found it.
{
  echo "## uname"; uname -a
  echo "## lscpu"; lscpu
  echo "## nproc"; nproc
  echo "## memory"; free -m
  echo "## cgroup"
  cg=$(sed -n 's/^0:://p' /proc/self/cgroup)
  for f in cpu.max cpu.weight memory.max cpuset.cpus.effective; do
    printf '%s: ' "$f"; cat "/sys/fs/cgroup$cg/$f" 2>/dev/null || echo "(none)"
  done
  echo "## steal since boot, all and per CPU, as the kernel accounts it"
  awk '/^cpu/{print $1, $9}' /proc/stat
  echo "## virtualisation"; systemd-detect-virt 2>&1 || true
  echo "## the kernel on steal"; as_root dmesg 2>/dev/null | grep -i steal || echo "(nothing)"
  echo "## limits"; ulimit -a
  echo "## disk"; df -h "$data" "$out"
} > "$out/machine.txt" 2>&1
steal() { awk '/^cpu /{print $9}' /proc/stat; }
steal_before=$(steal)
# The commit the bench is built from, stamped into every arm's reports:
# arms built from different commits are not compared.
sha=$(git -C "$(dirname "$0")" rev-parse HEAD)
if ! git -C "$(dirname "$0")" diff --quiet HEAD; then sha="$sha-dirty"; fi
echo "## commit: $sha" >> "$out/machine.txt"

# The bench, built once, optimised, as the image is.
export CARGO_INCREMENTAL=0
bin=$(cargo test --release --locked -p vellev --test step3_bench --no-run --message-format=json \
  | jq -r 'select(.reason == "compiler-artifact" and .target.name == "step3_bench" and .executable != null) | .executable' \
  | tail -1)
test -x "$bin"
cp "$bin" "$out/step3_bench"
chmod a+rx "$out/step3_bench"
as_root chown -R 10001:10001 "$out"

# Whether systemd can make a scope here, so that each process's memory
# peak and CPU are its own.
scoped=0
if command -v systemd-run > /dev/null \
  && as_root systemd-run --scope --quiet -p MemoryAccounting=yes true 2> /dev/null; then
  scoped=1
fi
echo "## systemd scopes: $scoped" | as_root tee -a "$out/machine.txt" > /dev/null

# One process of an arm, as uid 10001 with no capabilities, in a scope of
# its own where one can be made, its output to a log of its own (written
# as root, as the output directory is uid 10001's): started in the
# background, its pid the caller's $!.
start() {
  local name=$1 test=$2 log=$3
  local envs=(STEP3_ARM="$arm" STEP3_FIXTURE="$fixture" STEP3_OUT="$out" STEP3_DATA="$data"
              STEP3_AS_IMAGE=1 STEP3_SHA="$sha" STEP3_SCOPE="$scoped" RUST_BACKTRACE=1)
  local name_value
  while IFS='=' read -r name_value value; do
    case "$name_value" in
      STEP3_ARM|STEP3_FIXTURE|STEP3_OUT|STEP3_DATA|STEP3_AS_IMAGE|STEP3_SHA|STEP3_SCOPE) ;;
      STEP3_*) envs+=("$name_value=$value") ;;
    esac
  done < <(env)
  local run=(setpriv --reuid=10001 --regid=10001 --clear-groups --inh-caps=-all
             env -i HOME=/tmp PATH=/usr/bin:/bin "${envs[@]}"
             "$out/step3_bench" --ignored --exact "$test" --nocapture --test-threads 1)
  if [ "$scoped" = 1 ]; then
    run=(systemd-run --scope --quiet --unit="step3-$name-$arm-$$" -p MemoryAccounting=yes
         "${run[@]}")
  fi
  as_root bash -c 'exec "$@" > "$0" 2>&1' "$log" "${run[@]}" &
}

# Each arm: the node, then its load generator, then both waited for. If
# either fails, the other is stopped, the arm is said to have failed, and
# the others still run.
failed=()
for arm in "${arms[@]}"; do
  echo "== arm $arm"
  as_root rm -f "$out/ready-$arm.json" "$out/load-$arm.json" "$out/arm-$arm.json"
  start node step3_arm "$out/arm-$arm.log"
  node=$!
  start load step3_load "$out/load-$arm.log"
  load=$!
  ok=1
  if ! wait -n -p first "$node" "$load"; then
    ok=0
    as_root kill "$node" "$load" 2> /dev/null || true
  fi
  if [ "$first" = "$node" ]; then rest=$load; else rest=$node; fi
  wait "$rest" || ok=0
  # Its store, whatever became of the arm: one killed part way leaves its
  # store behind, and the next arm would run out of disk.
  as_root rm -rf "$data/arm-$arm"
  tail -n 30 "$out/arm-$arm.log" "$out/load-$arm.log" || true
  if [ "$ok" = 0 ]; then failed+=("$arm"); fi
done

echo "## steal over the run (ticks): $(( $(steal) - steal_before ))" | as_root tee -a "$out/machine.txt"
as_root env STEP3_OUT="$out" "$out/step3_bench" --ignored --exact step3_compare --nocapture \
  | as_root tee "$out/compare.log"
as_root chmod -R a+rX "$out"
cat "$out/summary.md"
if [ "${#failed[@]}" -gt 0 ]; then
  echo "arms that failed: ${failed[*]}" >&2
  exit 1
fi
