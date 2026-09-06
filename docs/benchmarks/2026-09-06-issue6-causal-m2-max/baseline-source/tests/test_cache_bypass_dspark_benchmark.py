from __future__ import annotations

import unittest

from Scripts.benchmark_cache_bypass_dspark import _candidate_gate


def _summary(
    *,
    request: float = 100.0,
    decode: float = 10.0,
    memory: float = 100.0,
    disk_per_token: float = 100.0,
    speculative_per_committed: float = 100.0,
) -> dict:
    return {
        "request_seconds": request,
        "decode_tokens_per_second": decode,
        "peak_memory_bytes": memory,
        "request_process_disk_bytes_per_generated_token": disk_per_token,
        "dspark_speculative_expert_bytes_per_committed_token": (
            speculative_per_committed
        ),
    }


class CacheBypassDSparkBenchmarkTests(unittest.TestCase):
    def test_candidate_gate_accepts_all_predeclared_thresholds(self):
        result = _candidate_gate(
            _summary(),
            _summary(request=94, decode=10.6, memory=114, disk_per_token=104),
        )

        self.assertTrue(result["candidate_gate_passed"])

    def test_candidate_gate_rejects_request_regression(self):
        result = _candidate_gate(
            _summary(),
            _summary(request=101, decode=11, memory=100, disk_per_token=100),
        )

        self.assertFalse(result["candidate_gate_passed"])
        self.assertFalse(
            result["criteria"]["request_time_at_least_5_percent_lower"]
        )

    def test_adaptive_gate_compares_speculative_bytes_with_fixed(self):
        result = _candidate_gate(
            _summary(),
            _summary(
                request=94,
                decode=10.6,
                memory=114,
                disk_per_token=104,
                speculative_per_committed=106,
            ),
            fixed=_summary(speculative_per_committed=100),
        )

        self.assertFalse(result["candidate_gate_passed"])
        self.assertFalse(
            result["criteria"][
                "speculative_bytes_per_committed_no_more_than_5_percent_above_fixed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
