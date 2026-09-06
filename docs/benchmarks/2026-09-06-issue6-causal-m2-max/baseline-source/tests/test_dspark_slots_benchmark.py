from __future__ import annotations

import unittest

from Scripts.benchmark_dspark_slots import _evaluate_gate


def _changes(
    *,
    peak: float = -0.01,
    draft: float = 0.50,
    speculative: float = 0.02,
    request: float = 0.05,
) -> dict[str, float | None]:
    return {
        "peak_memory_bytes": peak,
        "dspark_draft_expert_bytes_per_committed_token": draft,
        "dspark_speculative_expert_bytes_per_committed_token": speculative,
        "request_seconds": request,
    }


class DSparkSlotsBenchmarkGateTests(unittest.TestCase):
    def test_long_decode_thresholds_are_inclusive(self):
        result = _evaluate_gate(
            profile="long_decode",
            all_exact=True,
            max_tokens=128,
            change=_changes(),
        )

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_long_decode_stops_on_draft_read_regression(self):
        result = _evaluate_gate(
            profile="long_decode",
            all_exact=True,
            max_tokens=128,
            change=_changes(draft=0.8317152103559872),
        )

        self.assertFalse(result["passed"])
        self.assertFalse(
            result["checks"]["draft_read_increase_at_most_fifty_percent"]
        )
        self.assertTrue(
            result["checks"]["speculative_read_increase_at_most_two_percent"]
        )

    def test_exploratory_profile_requires_only_exactness_and_memory_direction(self):
        result = _evaluate_gate(
            profile="exploratory",
            all_exact=True,
            max_tokens=32,
            change=_changes(peak=-0.001, draft=2.0, speculative=1.0, request=1.0),
        )

        self.assertTrue(result["passed"])
        self.assertIsNone(result["thresholds"])


if __name__ == "__main__":
    unittest.main()
