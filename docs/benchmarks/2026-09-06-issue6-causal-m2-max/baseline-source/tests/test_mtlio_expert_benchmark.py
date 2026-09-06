from __future__ import annotations

import unittest

from Scripts.benchmark_mtlio_experts import (
    DEFAULT_COUNTS,
    METHODS,
    _integration_decision,
    _native_gate,
)


def _row(method: str, count: int) -> dict:
    reference = f"sha-{method}-{count}"
    row = {
        "method": method,
        "count": count,
        "status": "complete",
        "error": None,
        "exact": True,
        "candidate_sha256": reference,
        "reference_sha256": reference,
    }
    if method == "mtlio_shared":
        row.update(
            {
                "shared_event_ordering": True,
                "event_signaled_value": 1,
                "gpu_visible_seconds": 0.2,
                "gpu_copy_seconds": 0.1,
                "storage_mode": "shared",
                "cpu_visible_destination": True,
                "direct_cpu_sha256": reference,
            }
        )
    elif method == "mtlio_private":
        row.update(
            {
                "shared_event_ordering": True,
                "event_signaled_value": 1,
                "gpu_visible_seconds": 0.2,
                "gpu_copy_seconds": 0.1,
                "storage_mode": "private",
                "cpu_visible_destination": False,
                "explicit_gpu_validation_copy": True,
                "validation_copy_required_for_cpu_access": True,
            }
        )
    return row


def _rows() -> list[dict]:
    return [
        _row(method, count) for method in METHODS for count in DEFAULT_COUNTS
    ]


class MTLIOExpertBenchmarkTests(unittest.TestCase):
    def test_native_gate_accepts_exact_event_ordered_matrix_and_safe_cancel(self):
        cancellation = {
            "status": "cancelled",
            "destination_admitted": False,
            "candidate_sha256": None,
        }

        result = _native_gate(_rows(), cancellation)

        self.assertTrue(result["passed"])

    def test_native_gate_rejects_shared_event_or_byte_mismatch(self):
        rows = _rows()
        shared = next(row for row in rows if row["method"] == "mtlio_shared")
        shared["event_signaled_value"] = 0
        private = next(row for row in rows if row["method"] == "mtlio_private")
        private["candidate_sha256"] = "different"

        result = _native_gate(
            rows,
            {
                "status": "complete",
                "destination_admitted": True,
                "exact": True,
                "candidate_sha256": "same",
                "reference_sha256": "same",
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(
            result["criteria"][
                "shared_and_private_event_gpu_visibility_exact"
            ]
        )
        self.assertFalse(result["criteria"]["all_completed_ranges_byte_exact"])

    def test_integration_stops_when_public_mlx_dependency_handoff_is_missing(self):
        decision = _integration_decision(
            {"passed": True},
            {"passed": False},
            {"cache_state_balanced": False},
        )

        self.assertFalse(decision["continue_to_runtime_prototype"])
        self.assertEqual(
            decision["outcome"],
            "stop_no_supported_mtlio_to_mlx_dependency_handoff",
        )


if __name__ == "__main__":
    unittest.main()
