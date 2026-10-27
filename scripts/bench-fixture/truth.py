"""The benchmark fixture's truth (design §3c): for every measured query and
every filter case, the exact top K rows by cosine, scored in f64, with
equal scores ordered by row id. Every case leaves out deleted rows.

    python3 truth.py TEXT VECTORS OUT [--slice 100000]

TEXT holds passages.jsonl and queries.jsonl, as prepare.py writes them;
VECTORS holds passages/ and queries/, NNNNN.f32 shards of 768 little-endian
f32s a row, as embed_shards writes them. It writes to OUT:

  truth.jsonl        {query, case, ids, scores}, by query, then case
  truth-<N>.jsonl    the same over the first N passages only: the slice a
                     pull request is checked on, with a truth of its own
  truth.json         what was computed, and how it was checked

The cases:

  broad   every row
  mid     bucket < 10
  narrow  bucket < 1
  owner   rows of the owner of the article the query was asked of

For the owner case the design says only "1 of 8 by a hash of the
article". The owner is the query's own article's: a search within one
person's data, which holds the answer, as a hub searches one owner's
memories. It keeps about an eighth of the rows, the share an owner has.

Before writing, CHECKED queries chosen with a fixed seed are recomputed
by `check`, a brute force that shares no code with `exact_top`: no chunks,
no partial sort, the filters worked out row by row from the text again.
"""

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

K = 100
DIMENSIONS = 768
CASES = ("broad", "mid", "narrow", "owner")
CHECKED = 20


@dataclass
class Meta:
    bucket: np.ndarray
    owner: np.ndarray
    deleted: np.ndarray


def allowed(meta, start, stop, query_owner):
    """Per case, which rows in [start, stop) each query may have: a
    `queries x rows` mask, or one row mask for all queries."""
    live = ~meta.deleted[start:stop]
    bucket = meta.bucket[start:stop]
    owner = meta.owner[start:stop]
    return {
        "broad": live,
        "mid": live & (bucket < 10),
        "narrow": live & (bucket < 1),
        "owner": live[None, :] & (owner[None, :] == query_owner[:, None]),
    }


def best(ids, scores, k):
    """The first k of (ids, scores) by score, high first, then by id."""
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def exact_top(queries, chunks, meta, query_owner, k):
    """For each case, per query, the exact top k (ids, scores) over the
    rows `chunks` gives as (first id, vectors)."""
    q = np.asarray(queries, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    kept = {case: [(np.empty(0, np.int64), np.empty(0)) for _ in q] for case in CASES}
    for start, vectors in chunks:
        v = np.asarray(vectors, np.float64)
        norms = np.linalg.norm(v, axis=1)
        scores = (q @ v.T) / norms[None, :]
        ids = np.arange(start, start + len(v), dtype=np.int64)
        for case, mask in allowed(meta, start, start + len(v), query_owner).items():
            for n in range(len(q)):
                row = mask if mask.ndim == 1 else mask[n]
                s, i = scores[n][row], ids[row]
                if len(s) > k:
                    # Everything at or above the k-th score, ties included,
                    # so that ties are settled by id below, not by the sort.
                    kth = np.partition(s, len(s) - k)[len(s) - k]
                    keep = s >= kth
                    s, i = s[keep], i[keep]
                old_i, old_s = kept[case][n]
                kept[case][n] = best(np.concatenate([old_i, i]), np.concatenate([old_s, s]), k)
    return {
        case: [([int(x) for x in i], [float(x) for x in s]) for i, s in per]
        for case, per in kept.items()
    }


def check(queries, vectors, meta, query_owner, got, k, which):
    """Recompute the truth of the queries `which` one at a time, the slow
    way, and fail if any id or its order differs from `got`, or a score
    by more than f64 rounding."""
    norms = [float(np.dot(r, r)) ** 0.5 for r in (np.asarray(row, np.float64) for row in vectors)]
    for n in which:
        q = np.asarray(queries[n], np.float64)
        qn = float(np.dot(q, q)) ** 0.5
        scores = [float(np.dot(np.asarray(row, np.float64), q)) / (norm * qn) for row, norm in zip(vectors, norms)]
        for case in CASES:
            passing = []
            for i in range(len(vectors)):
                if meta.deleted[i]:
                    continue
                b, o = int(meta.bucket[i]), int(meta.owner[i])
                if case == "mid" and not b < 10:
                    continue
                if case == "narrow" and not b < 1:
                    continue
                if case == "owner" and o != int(query_owner[n]):
                    continue
                passing.append((-scores[i], i))
            want = sorted(passing)[:k]
            ids, s = got[case][n]
            assert list(ids) == [i for _, i in want], f"query {n} {case}: ids differ"
            assert np.allclose(s, [-x for x, _ in want], rtol=0, atol=1e-12), f"query {n} {case}: scores differ"


def read_meta(path):
    bucket, owner, deleted, titles = [], [], [], {}
    with open(path) as f:
        for n, line in enumerate(f):
            row = json.loads(line)
            assert row["id"] == n, "passages are in id order"
            bucket.append(row["bucket"])
            owner.append(row["owner"])
            deleted.append(row["deleted"])
            titles.setdefault(row["title"], row["owner"])
    meta = Meta(np.array(bucket), np.array(owner), np.array(deleted, bool))
    return meta, titles


def shards(directory):
    return sorted(Path(directory).glob("*.f32"))


def read_rows(directory, stop=None):
    parts = [np.fromfile(p, "<f4").reshape(-1, DIMENSIONS) for p in shards(directory)]
    rows = np.concatenate(parts)
    return rows if stop is None else rows[:stop]


def chunked(rows, size):
    for start in range(0, len(rows), size):
        yield start, rows[start : start + size]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, got, count):
    with open(path, "w") as f:
        for n in range(count):
            for case in CASES:
                ids, scores = got[case][n]
                f.write(json.dumps({"query": n, "case": case, "ids": ids, "scores": scores}) + "\n")


def main(text, vectors, out, k=K, slice_rows=100_000, chunk=50_000, checked=CHECKED):
    text, vectors, out = Path(text), Path(vectors), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta, owners = read_meta(text / "passages.jsonl")
    with open(text / "queries.jsonl") as f:
        asked = [json.loads(line) for line in f]
    query_owner = np.array([owners[q["title"]] for q in asked])
    queries = read_rows(vectors / "queries")
    rows = read_rows(vectors / "passages")
    assert len(queries) == len(asked) and len(rows) == len(meta.bucket)
    sample = sorted(random.Random("bench-truth-check").sample(range(len(asked)), min(checked, len(asked))))
    report = {"k": k, "cases": list(CASES), "owner_case": "query_article_owner", "checked_queries": sample}
    for name, stop in (("truth.jsonl", len(rows)), (f"truth-{slice_rows}.jsonl", min(slice_rows, len(rows)))):
        part = rows[:stop]
        part_meta = Meta(meta.bucket[:stop], meta.owner[:stop], meta.deleted[:stop])
        got = exact_top(queries, chunked(part, chunk), part_meta, query_owner, k)
        check(queries, part, part_meta, query_owner, got, k, sample)
        write(out / name, got, len(asked))
        live = ~part_meta.deleted
        report[name] = {
            "rows": int(stop),
            "sha256": sha256(out / name),
            "selectivity": {
                "broad": float(live.mean()),
                "mid": float((live & (part_meta.bucket < 10)).mean()),
                "narrow": float((live & (part_meta.bucket < 1)).mean()),
                "owner": float(np.mean([(live & (part_meta.owner == o)).mean() for o in query_owner])),
            },
            "fewest_ids": {case: min(len(got[case][n][0]) for n in range(len(asked))) for case in CASES},
        }
        print(name, json.dumps(report[name]), file=sys.stderr)
    with open(out / "truth.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("vectors")
    parser.add_argument("out")
    parser.add_argument("--slice", type=int, default=100_000)
    args = parser.parse_args()
    main(args.text, args.vectors, args.out, slice_rows=args.slice)
