"""A synthetic fixture laid out as the real one is, for running the
benchmark harness end to end where the real fixture is not to be fetched:
the harness's own CI check. Its numbers say nothing about either engine.

    python3 synthetic.py OUT [--rows 5000] [--queries 100]

Writes OUT/text, OUT/vectors and OUT/truth (with truth.py), as the
release unpacks. Vectors are unit vectors around 64 random centres, so
that nearest neighbours mean something; the attributes are given as
prepare.py gives them.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

import truth

WORDS = "fox den hen river stone cloud lamp field winter harbour copper violin".split()


def unit(rows):
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype("<f4")


def main(out, rows, queries, seed=7):
    out = Path(out)
    rng = np.random.default_rng(seed)
    words = random.Random(seed)
    centres = rng.standard_normal((64, truth.DIMENSIONS))
    which = rng.integers(0, 64, rows)
    vectors = unit(centres[which] + 0.6 * rng.standard_normal((rows, truth.DIMENSIONS)))
    deleted = random.Random("bench-deleted")
    (out / "text").mkdir(parents=True, exist_ok=True)
    titles = [f"article {n // 5}" for n in range(rows)]
    with open(out / "text" / "passages.jsonl", "w") as f:
        for n in range(rows):
            row = {
                "id": n,
                "text": " ".join(words.choice(WORDS) for _ in range(40)),
                "title": titles[n],
                "url": f"https://example.invalid/{n // 5}",
                "bucket": n % 100,
                "owner": hashlib.sha256(titles[n].encode()).digest()[0] % 8,
                "deleted": deleted.random() < 0.05,
            }
            f.write(json.dumps(row) + "\n")
    for name, first in (("queries", 0), ("warmup", queries)):
        (out / "vectors" / name).mkdir(parents=True, exist_ok=True)
        asked = rng.integers(0, rows, queries)
        unit(vectors[asked] + 0.3 * rng.standard_normal((queries, truth.DIMENSIONS))).tofile(
            out / "vectors" / name / "00000.f32"
        )
        with open(out / "text" / f"{name}.jsonl", "w") as f:
            for n, at in enumerate(asked):
                f.write(json.dumps({"id": first + n, "text": " ".join(words.choice(WORDS) for _ in range(8)),
                                    "source": "synthetic", "title": titles[at]}) + "\n")
    (out / "vectors" / "passages").mkdir(parents=True, exist_ok=True)
    for shard, start in enumerate(range(0, rows, 10_000)):
        vectors[start : start + 10_000].tofile(out / "vectors" / "passages" / f"{shard:05}.f32")
    truth.main(out / "text", out / "vectors", out / "truth", slice_rows=rows, checked=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()
    main(args.out, args.rows, args.queries)
