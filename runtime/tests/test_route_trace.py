from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deepseek_v4_ssd.route_trace import RouteTraceRecorder, analyze_handoff


class RouteTraceTests(unittest.TestCase):
    def test_trace_preserves_each_prefill_chunk_histogram(self):
        recorder = RouteTraceRecorder(1, 4, 2, 10)
        with recorder.phase("prefill"):
            recorder.record(0, np.array([0, 1, 0, 2]))
            recorder.record(0, np.array([1, 1, 1, 3]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            recorder.write(path)
            trace = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            trace["prefill_chunk_histograms"],
            [[[2, 1, 1, 0], [0, 3, 0, 1]]],
        )

    def test_trace_measures_prefill_hot_set_decode_coverage(self):
        recorder = RouteTraceRecorder(2, 4, 2, 10)
        with recorder.phase("prefill"):
            recorder.record(0, np.array([0, 1, 0, 2]))
            recorder.record(1, np.array([3, 2, 3, 1]))
        with recorder.phase("decode"):
            recorder.record(0, np.array([0, 3]))
            recorder.record_residency(0, [0, 3], [3])
            recorder.record(0, np.array([0, 2]))
            recorder.record_residency(0, [0, 2], [0])
            recorder.record(1, np.array([3, 0]))
            recorder.record_residency(1, [3, 0], [0])
            recorder.record(1, np.array([3, 1]))
            recorder.record_residency(1, [3, 1], [3])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            recorder.write(path)
            trace = json.loads(path.read_text(encoding="utf-8"))

        result = analyze_handoff(trace, hot_per_layer=(1,), decode_tokens=2)
        policy = result["policies"][0]
        self.assertEqual(policy["used_slots"], 2)
        self.assertEqual(policy["slot_bytes"], 20)
        self.assertEqual(policy["covered_routes"], 4)
        self.assertEqual(policy["total_routes"], 8)
        self.assertEqual(policy["route_coverage"], 0.5)
        self.assertEqual(policy["baseline_miss_reads"], 4)
        self.assertEqual(policy["handoff_recoverable_miss_reads"], 2)
        self.assertEqual(policy["handoff_recoverable_miss_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
