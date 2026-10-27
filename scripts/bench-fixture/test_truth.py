"""The rules the benchmark's truth is held to (design §3c), on tiny
hand-made rows. Run from the repository root:

    python3 -m unittest discover -s scripts/bench-fixture -p 'test_*.py'
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import truth

D = 4


def rows(n, bucket=None, owner=None, deleted=()):
    """Row metadata as truth reads it: bucket, owner and deleted per id."""
    return truth.Meta(
        bucket=np.array(bucket if bucket is not None else [i % 100 for i in range(n)]),
        owner=np.array(owner if owner is not None else [0] * n),
        deleted=np.array([i in set(deleted) for i in range(n)]),
    )


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def chunks(vectors, size):
    return [(start, vectors[start : start + size]) for start in range(0, len(vectors), size)]


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


class Rules(unittest.TestCase):
    def top(self, vectors, queries, meta, query_owner=None, k=3, size=1000):
        query_owner = np.array(query_owner if query_owner is not None else [0] * len(queries))
        return truth.exact_top(np.array(queries, np.float32), chunks(np.array(vectors, np.float32), size), meta, query_owner, k)

    def test_equal_scores_are_ordered_by_row_id(self):
        same = unit(1, 1, 0, 0)
        vectors = [unit(0, 0, 1, 0), same, same, same, unit(0, 0, 0, 1)]
        got = self.top(vectors, [unit(1, 0, 0, 0)], rows(5))
        self.assertEqual(got["broad"][0][0], [1, 2, 3])

    def test_a_tie_across_a_chunk_boundary_is_still_ordered_by_row_id(self):
        same = unit(1, 1, 0, 0)
        vectors = [unit(0, 0, 1, 0)] + [same] * 6
        for size in (1, 2, 3, 4, 1000):
            got = self.top(vectors, [unit(1, 0, 0, 0)], rows(7), size=size)
            self.assertEqual(got["broad"][0][0], [1, 2, 3], f"chunks of {size}")

    def test_many_rows_tied_at_the_kth_score_give_the_lowest_ids(self):
        # A partial sort picks arbitrarily among equal scores: the truth may not.
        rng = np.random.default_rng(1)
        n = 500
        top = unit(1, 1, 0, 0)
        tied = set(rng.choice(n, 200, replace=False).tolist())
        vectors = [top if i in tied else unit(rng.random() * 0.1, 1, rng.random(), 0) for i in range(n)]
        for size in (64, 1000):
            got = self.top(vectors, [unit(1, 0, 0, 0)], rows(n), k=5, size=size)
            self.assertEqual(got["broad"][0][0], sorted(tied)[:5], f"chunks of {size}")

    def test_deleted_rows_are_never_in_any_case(self):
        vectors = [unit(1, 0, 0, 0), unit(1, 0.1, 0, 0), unit(1, 0.2, 0, 0), unit(1, 0.3, 0, 0)]
        meta = rows(4, bucket=[0, 0, 0, 0], deleted=[0, 2])
        got = self.top(vectors, [unit(1, 0, 0, 0)], meta, k=4)
        for case in truth.CASES:
            self.assertEqual(got[case][0][0], [1, 3], case)

    def test_each_filter_keeps_only_its_rows(self):
        n = 30
        vectors = [unit(1, i / n, 0, 0) for i in range(n)]
        meta = rows(n, bucket=[i * 7 % 100 for i in range(n)], owner=[i % 8 for i in range(n)])
        got = self.top(vectors, [unit(1, 0, 0, 0)], meta, query_owner=[3], k=n)
        self.assertEqual(sorted(got["broad"][0][0]), list(range(n)))
        self.assertEqual(sorted(got["mid"][0][0]), [i for i in range(n) if i * 7 % 100 < 10])
        self.assertEqual(sorted(got["narrow"][0][0]), [i for i in range(n) if i * 7 % 100 < 1])
        self.assertEqual(sorted(got["owner"][0][0]), [i for i in range(n) if i % 8 == 3])

    def test_the_owner_case_follows_each_querys_own_owner(self):
        vectors = [unit(1, 0, 0, 0)] * 4
        meta = rows(4, owner=[0, 1, 0, 1])
        got = self.top(vectors, [unit(1, 0, 0, 0)] * 2, meta, query_owner=[1, 0], k=4)
        self.assertEqual(got["owner"][0][0], [1, 3])
        self.assertEqual(got["owner"][1][0], [0, 2])

    def test_fewer_rows_than_k_gives_every_row_that_passes(self):
        got = self.top([unit(1, 0, 0, 0), unit(0, 1, 0, 0)], [unit(1, 0, 0, 0)], rows(2), k=100)
        self.assertEqual(got["broad"][0][0], [0, 1])
        self.assertEqual(got["narrow"][0][0], [0])

    def test_scores_are_cosine_whatever_a_vectors_length(self):
        vectors = [[3, 0, 0, 0], [1, 1, 0, 0], [0.01, 0, 0, 0]]
        got = self.top(vectors, [[2, 0, 0, 0]], rows(3), k=3)
        ids, scores = got["broad"][0]
        self.assertEqual(ids, [0, 2, 1])
        self.assertAlmostEqual(scores[0], 1.0, places=12)
        self.assertAlmostEqual(scores[2], 2 ** -0.5, places=7)

    def test_scores_are_summed_in_f64(self):
        # Differ only past f32's precision once summed: f32 sums would tie.
        a = np.full(D * 256, 1.0, np.float32)
        b = a.copy()
        b[0] = np.float32(1.0 + 2 ** -20)
        q = a.copy()
        q[0] = 2.0
        got = truth.exact_top(
            np.array([q]), [(0, np.array([a, b]))], rows(2), np.array([0]), 2
        )
        self.assertEqual(got["broad"][0][0], [1, 0])

    def test_the_independent_check_passes_a_right_truth_and_fails_a_wrong_one(self):
        rng = np.random.default_rng(7)
        vectors = rng.standard_normal((50, D)).astype(np.float32)
        queries = rng.standard_normal((3, D)).astype(np.float32)
        meta = rows(50, owner=[i % 8 for i in range(50)])
        owners = np.array([1, 2, 3])
        got = truth.exact_top(queries, chunks(vectors, 7), meta, owners, 5)
        truth.check(queries, vectors, meta, owners, got, 5, [0, 2])
        wrong = {case: [(list(ids), list(scores)) for ids, scores in per] for case, per in got.items()}
        ids, scores = wrong["mid"][2]
        ids[1], ids[2] = ids[2], ids[1]
        with self.assertRaises(AssertionError):
            truth.check(queries, vectors, meta, owners, wrong, 5, [0, 2])


class Files(unittest.TestCase):
    """The whole tool on files laid out as the fixture is."""

    def test_writes_one_line_per_query_and_case_and_a_slice_of_its_own(self):
        rng = np.random.default_rng(3)
        n, q = 25, 4
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "text").mkdir()
            (tmp / "vectors" / "passages").mkdir(parents=True)
            (tmp / "vectors" / "queries").mkdir(parents=True)
            titles = [f"t{i % 6}" for i in range(n)]
            with open(tmp / "text" / "passages.jsonl", "w") as f:
                for i in range(n):
                    row = {"id": i, "text": "x", "title": titles[i], "url": "u",
                           "bucket": i % 100, "owner": i % 6 % 8, "deleted": i == 4}
                    f.write(json.dumps(row) + "\n")
            with open(tmp / "text" / "queries.jsonl", "w") as f:
                for i in range(q):
                    f.write(json.dumps({"id": i, "text": "q", "source": "squad", "title": f"t{i + 1}"}) + "\n")
            vectors = rng.standard_normal((n, 768)).astype("<f4")
            vectors[:10].tofile(tmp / "vectors" / "passages" / "00000.f32")
            vectors[10:20].tofile(tmp / "vectors" / "passages" / "00001.f32")
            vectors[20:].tofile(tmp / "vectors" / "passages" / "00002.f32")
            rng.standard_normal((q, 768)).astype("<f4").tofile(tmp / "vectors" / "queries" / "00000.f32")
            truth.main(tmp / "text", tmp / "vectors", tmp / "out", k=5, slice_rows=12, chunk=8, checked=2)
            whole = read(tmp / "out" / "truth.jsonl")
            part = read(tmp / "out" / "truth-12.jsonl")
            self.assertEqual(len(whole), q * len(truth.CASES))
            self.assertEqual([(r["query"], r["case"]) for r in whole],
                             [(i, c) for i in range(q) for c in truth.CASES])
            self.assertTrue(all(max(r["ids"], default=-1) < 12 for r in part))
            self.assertTrue(all(4 not in r["ids"] for r in whole + part))
            for r in whole:
                if r["case"] == "owner":
                    self.assertTrue(all((i % 6) % 8 == (r["query"] + 1) % 8 for i in r["ids"]))
            with open(tmp / "out" / "truth.json") as f:
                self.assertEqual(json.load(f)["owner_case"], "query_article_owner")


if __name__ == "__main__":
    unittest.main()
