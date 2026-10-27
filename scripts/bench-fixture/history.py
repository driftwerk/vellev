"""The benchmark's history (design r2, §7) and its soft gate (§8).

    python3 history.py ROOT RESULTS DAY RUN

ROOT is a checkout of the `benchmark-history` branch. Each night's
results.json is kept whole as runs/DAY-RUN.json, and one line is added
to series.jsonl: vellev's default p99 (broad, limit 10) and Weaviate's
image digest. A digest that changed is marked as a step, not a
regression. The soft gate trips when vellev's default p99 is more than
25% over the median of the 7 nights before; it opens an issue and never
fails the run. Prints what it found as JSON.
"""

import json
import statistics
import sys
from pathlib import Path

WINDOW = 7
SOFT = 1.25


def default_p99(results):
    for point in results["engines"]["vellev"]["points"]:
        if point["limit"] == 10 and point["case"] == "broad" and point["knob"] == "vellev_default":
            return point["p99_ms"]["median"]
    raise ValueError("no default broad point at limit 10")


def append(root, results, day, run):
    root = Path(root)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    with open(root / "runs" / f"{day}-{run}.json", "w") as f:
        json.dump(results, f, indent=1)
    series_path = root / "series.jsonl"
    before = []
    if series_path.exists():
        with open(series_path) as f:
            before = [json.loads(line) for line in f if line.strip()]
    p99 = default_p99(results)
    digest = results["engines"]["weaviate"]["image"]["digest"]
    window = [entry["vellev_default_broad_p99_ms"] for entry in before[-WINDOW:]]
    median = statistics.median(window) if len(window) == WINDOW else None
    entry = {
        "day": day,
        "run": run,
        "rows": results["rows"],
        "vellev_default_broad_p99_ms": p99,
        "weaviate_digest": digest,
        "weaviate_step": bool(before) and before[-1]["weaviate_digest"] != digest,
        "gate_passed": results["gate"]["passed"],
        "rolling_median_ms": median,
        "soft_gate_tripped": median is not None and p99 > SOFT * median,
    }
    with open(series_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


if __name__ == "__main__":
    root, path, day, run = sys.argv[1:5]
    with open(path) as f:
        found = append(root, json.load(f), day, run)
    print(json.dumps(found))
