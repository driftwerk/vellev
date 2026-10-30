#!/usr/bin/env bash
# The step 3 bench (merge note, §3 and §9), every arm one after another on
# this machine, each in a process of its own run as the image runs: uid
# 10001, no capabilities, the default limits. Linux only.
#
#   scripts/step3-bench/run.sh FIXTURE OUT DATA [ARMS...]
#
# FIXTURE is the bench fixture unpacked (text/, vectors/), as the
# benchmark workflow unpacks it; OUT gets each arm's JSON and metrics, the
# machine's description, summary.md and verdict.json; DATA holds each
# arm's store while it runs, deleted after. ARMS defaults to "a b c d".
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
arms=("${@:-a b c d}")
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
  echo "## limits"; ulimit -a
  echo "## disk"; df -h "$data" "$out"
} > "$out/machine.txt" 2>&1
steal() { awk '/^cpu /{print $9}' /proc/stat; }
steal_before=$(steal)

# The bench, built once, optimised, as the image is.
export CARGO_INCREMENTAL=0
bin=$(cargo test --release --locked -p vellev --test step3_bench --no-run --message-format=json \
  | jq -r 'select(.reason == "compiler-artifact" and .target.name == "step3_bench" and .executable != null) | .executable' \
  | tail -1)
test -x "$bin"
cp "$bin" "$out/step3_bench"
chmod a+rx "$out/step3_bench"
as_root chown -R 10001:10001 "$out"

# Each arm: in a cgroup of its own where systemd can make one, so that its
# memory peak is the arm's; as uid 10001 with no capabilities either way.
failed=()
for arm in "${arms[@]}"; do
  echo "== arm $arm"
  envs=(STEP3_ARM="$arm" STEP3_FIXTURE="$fixture" STEP3_OUT="$out" STEP3_DATA="$data"
        STEP3_AS_IMAGE=1 RUST_BACKTRACE=1)
  while IFS='=' read -r name value; do
    case "$name" in STEP3_ARM|STEP3_FIXTURE|STEP3_OUT|STEP3_DATA|STEP3_AS_IMAGE) ;;
      STEP3_*) envs+=("$name=$value") ;; esac
  done < <(env)
  run=(setpriv --reuid=10001 --regid=10001 --clear-groups --inh-caps=-all
       env -i HOME=/tmp PATH=/usr/bin:/bin "${envs[@]}"
       "$out/step3_bench" --ignored --exact step3_arm --nocapture --test-threads 1)
  if command -v systemd-run > /dev/null && as_root systemd-run --scope --quiet -p MemoryAccounting=yes true 2> /dev/null; then
    run=(systemd-run --scope --quiet -p MemoryAccounting=yes "${run[@]}")
  fi
  # An arm that fails is said, and the others still run.
  as_root "${run[@]}" 2>&1 | as_root tee "$out/arm-$arm.log" || failed+=("$arm")
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
