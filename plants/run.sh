#!/bin/bash
# Each line of a plants list: name | patches | expect | cargo test args;...
# Plants are applied to committed code and undone by checking out what is
# committed, one at a time, on the head this branch carries.
set -u
list=$1
mkdir -p plant-logs
summary=plant-logs/summary-$(basename "$list").md
echo "head $(git rev-parse HEAD), plants on $(git rev-parse HEAD~1)" > "$summary"
echo "| plant | expected | came to | rules failed |" >> "$summary"
echo "|---|---|---|---|" >> "$summary"
while IFS='|' read -r name patches expect runs; do
  [ -z "$name" ] && continue
  name=$(echo $name); expect=$(echo $expect)
  log=plant-logs/$name.log; : > "$log"
  applied=yes
  for p in $patches; do git apply plants/$p.patch || { echo "patch $p failed" >> "$log"; applied=no; }; done
  status=pass
  IFS=';' read -ra list_runs <<< "$runs"
  for run in "${list_runs[@]}"; do
    timeout 1800 bash -c "cargo test --locked $run --test-threads=2" >> "$log" 2>&1
    code=$?
    if [ $code -eq 124 ]; then status=HUNG; elif [ $code -ne 0 ] && [ "$status" != HUNG ]; then status=fail; fi
  done
  [ "$applied" = no ] && status=NOT-APPLIED
  failed=$(grep -E '^test .* FAILED$' "$log" | sed -E 's/^test (.*) \.\.\. FAILED$/\1/' | sort -u | paste -sd, -)
  signal=$(grep -E 'signal: [0-9]+' "$log" | head -1 | sed -E 's/.*(signal: [0-9]+[^)]*).*/\1/')
  ran=$(grep -cE '^test .* (ok|FAILED)$' "$log")
  echo "| $name | $expect | $status ($ran rules ran) | $failed $signal |" >> "$summary"
  git checkout -- crates
done < "$list"
cat "$summary" >> "$GITHUB_STEP_SUMMARY"
cat "$summary"
