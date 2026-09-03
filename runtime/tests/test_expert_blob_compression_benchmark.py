from __future__ import annotations

import unittest

from Scripts.benchmark_expert_blob_compression import (
    _AppleCompression,
    _gate,
    _percentile,
    _sample_specs,
)


class ExpertBlobCompressionBenchmarkTests(unittest.TestCase):
    def test_p95_uses_nearest_rank(self):
        self.assertEqual(_percentile(list(range(1, 10)), 0.95), 9)

    def test_native_codec_round_trip_and_gate(self):
        source = bytes(range(256)) * 32
        codec = _AppleCompression("lzfse")
        compressed = codec.encode(source)

        self.assertEqual(codec.decode(compressed, len(source)), source)
        self.assertEqual(
            _sample_specs(2_611_200)[-1],
            ("full_expert", 2_611_200, "single_expert_blob"),
        )
        row = {
            "exact": True,
            "normal_comparison": {
                "passed_5_percent_time_gate": True,
                "passed_5_percent_peak_memory_gate": True,
            },
            "metal_comparison": {
                "passed_5_percent_time_gate": True,
                "passed_5_percent_peak_memory_gate": True,
            },
        }
        self.assertTrue(_gate([row])["passed"])
        row["metal_comparison"]["passed_5_percent_time_gate"] = False
        self.assertFalse(_gate([row])["passed"])


if __name__ == "__main__":
    unittest.main()
