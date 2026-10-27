"""The rules the benchmark history and the soft gate are held to (design
r2, §7 and §8), on hand-made results. Run with the other fixture tests."""

import json
import tempfile
import unittest
from pathlib import Path

import history


def results(p99s, digest="sha256:a", gate=True):
    """A results.json with vellev's default broad p99 at limit 10 as given."""
    return {
        "rows": 1000000,
        "engines": {
            "vellev": {"points": [
                {"limit": 10, "case": "broad", "knob": "vellev_default", "p99_ms": {"median": p99s}},
                {"limit": 10, "case": "broad", "knob": {"oversample": 4}, "p99_ms": {"median": 0.1}},
            ]},
            "weaviate": {"image": {"digest": digest}},
        },
        "gate": {"passed": gate},
    }


class History(unittest.TestCase):
    def appended(self, root, day, p99, digest="sha256:a"):
        return history.append(root, results(p99, digest), day, f"run-{day}")

    def test_each_night_is_kept_whole_and_one_line_added_to_the_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.appended(root, "2026-10-01", 2.0)
            self.appended(root, "2026-10-02", 2.1)
            self.assertTrue((root / "runs" / "2026-10-01-run-2026-10-01.json").exists())
            series = [json.loads(line) for line in open(root / "series.jsonl")]
            self.assertEqual([s["vellev_default_broad_p99_ms"] for s in series], [2.0, 2.1])

    def test_the_soft_gate_trips_past_a_quarter_over_the_7_day_rolling_median(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n, p99 in enumerate([2.0, 2.0, 2.0, 1.0, 9.0, 2.0, 2.0]):
                self.appended(root, f"2026-10-0{n + 1}", p99)
            self.assertFalse(self.appended(root, "2026-10-08", 2.5)["soft_gate_tripped"], "2.5 is 1.25 x 2.0, not over")
            tripped = self.appended(root, "2026-10-09", 2.51)
            self.assertTrue(tripped["soft_gate_tripped"])
            self.assertEqual(tripped["rolling_median_ms"], 2.0)

    def test_too_short_a_history_trips_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.appended(root, "2026-10-01", 1.0)
            self.assertFalse(self.appended(root, "2026-10-02", 100.0)["soft_gate_tripped"])

    def test_a_weaviate_digest_change_is_marked_as_a_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(self.appended(root, "2026-10-01", 2.0, "sha256:a")["weaviate_step"])
            self.assertFalse(self.appended(root, "2026-10-02", 2.0, "sha256:a")["weaviate_step"])
            self.assertTrue(self.appended(root, "2026-10-03", 2.0, "sha256:b")["weaviate_step"])


if __name__ == "__main__":
    unittest.main()
