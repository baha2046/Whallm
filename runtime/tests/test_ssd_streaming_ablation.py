from __future__ import annotations

import mmap
import os
import tempfile
import unittest

from research.ssd_streaming_ablation import summarize
from pathlib import Path


class StreamingAblationTests(unittest.TestCase):

    def test_speed_signal_rejects_changed_output_swap_and_excess_memory(self):
        def row(wave, variant):
            return {"wave": wave, "variant": variant, "swapout_delta": 0,
                    "metrics": {"token_sha256": "same", "request_seconds": 10 if variant == "control" else 8,
                                "time_to_first_token_seconds": 5, "decode_tokens_per_second": 10 if variant == "control" else 12,
                                "expert_bytes_read": 100, "peak_memory_bytes": 200}}
        rows = [row(wave, mode) for wave in range(2) for mode in ("control", "dynamic")]
        self.assertTrue(summarize(rows)["variants"]["dynamic"]["screening_speed_signal"])
        for key, value in (("token_sha256", "different"), ("peak_memory_bytes", 2_000_000_000)):
            original = rows[1]["metrics"][key]
            rows[1]["metrics"][key] = value
            self.assertFalse(summarize(rows)["variants"]["dynamic"]["screening_speed_signal"])
            rows[1]["metrics"][key] = original
        rows[1]["swapout_delta"] = 1
        self.assertFalse(summarize(rows)["variants"]["dynamic"]["screening_speed_signal"])

    def test_one_pair_is_not_a_speed_signal(self):
        self.assertEqual(summarize([]), {"pairs": [], "variants": {}})


if __name__ == "__main__":
    unittest.main()
