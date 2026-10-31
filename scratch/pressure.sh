#!/bin/bash
# Scratch: main and the M8 head under a memory limit. Not for merging.
set -u
mkdir -p measured run-root
rows=$1; dims=768; runs=${2:-3}
V=$(( rows * (4 * dims + 8) + 36 ))
one() { # arm limit label run
  local arm=$1 limit=$2 label=$3 run=$4 out=measured/$1-$rows-$3-$4.txt
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  sudo env ROOT=$PWD/run-root ROWS=$rows DIMS=$dims READER_MB=200 \
    systemd-run --scope --quiet -p MemoryMax=$limit -p MemorySwapMax=0 ./bin-$arm > "$out" 2>&1
  local code=$?
  echo "exit=$code" >> "$out"
  echo "$arm $label run $run limit=$limit exit=$code $(grep '^merge ' "$out" | cut -c1-90)"
  sudo rm -rf run-root/* 2>/dev/null
  sudo dmesg 2>/dev/null | grep -c "oom-kill" | sed "s/^/oom-kill lines so far: /"
}
# With no limit to speak of: what the after arm comes to.
one after 15G free 0
line=$(grep -h '^merge ' measured/after-$rows-free-0.txt)
base=$(echo "$line" | sed -E 's/.*base_anon=([0-9.]+).*/\1/')
peak=$(echo "$line" | sed -E 's/.*anon_peak=([0-9.]+).*/\1/')
# base_anon and anon_peak are the group's anon in MB, at the merge's start
# and at most: so the limit is baseline + what the after arm adds + 150 MB.
X=$(python3 -c "print(int(($peak + 150) * 1000000))")
XV=$(( X + V ))
echo "rows=$rows V=$V base_anon=$base anon_peak=$peak X=$X XV=$XV" | tee measured/limits-$rows.txt
one before 15G free 0
for free in 1 2; do one after 15G free $free; one before 15G free $free; done
for run in $(seq 1 "$runs"); do
  if [ $((run % 2)) -eq 1 ]; then arms="before after"; else arms="after before"; fi
  for arm in $arms; do
    one $arm $X x $run
    one $arm $XV xv $run
  done
done
