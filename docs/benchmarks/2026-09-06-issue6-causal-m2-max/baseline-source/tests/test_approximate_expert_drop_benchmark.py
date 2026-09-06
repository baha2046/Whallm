from __future__ import annotations

import unittest

import numpy as np

from Scripts.benchmark_approximate_expert_drop import (
    _common_prefix_length,
    _evaluate_checks,
    _validate_manifest,
)
from Scripts.benchmark_approximate_expert_drop_component import _kl_divergence
from Scripts.benchmark_approximate_expert_drop_4k import _common_prefix
from Scripts.benchmark_approximate_expert_drop_formal import (
    WORKLOADS,
    _gate as _formal_gate,
)


class ApproximateExpertDropBenchmarkTests(unittest.TestCase):
    def test_manifest_checks_and_token_comparison(self):
        cases = [
            {
                "id": f"quality_{index}",
                "suite": "quality",
                "messages": [{"role": "user", "content": "test"}],
                "checks": [{"kind": "contains", "value": "PASS"}],
            }
            for index in range(5)
        ] + [
            {
                "id": f"safety_{index}",
                "suite": "safety",
                "messages": [{"role": "system", "content": "test"}],
                "checks": [{"kind": "not_contains", "value": "FAIL"}],
            }
            for index in range(5)
        ]
        manifest = {
            "schema_version": 1,
            "candidate": {"mode": "learned-route-drop-lowest-1"},
            "cases": cases,
        }

        self.assertEqual(len(_validate_manifest(manifest)), 10)
        self.assertTrue(
            _evaluate_checks(
                "PASS",
                [
                    {"kind": "contains", "value": "PASS"},
                    {"kind": "not_contains", "value": "FAIL"},
                ],
            )["passed"]
        )
        self.assertEqual(_common_prefix_length([1, 2, 3], [1, 2, 4]), 2)
        self.assertAlmostEqual(
            _kl_divergence(
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([1.0, 2.0], dtype=np.float32),
            ),
            0.0,
        )
        self.assertEqual(_common_prefix([7, 8, 9], [7, 8, 10]), 2)

    def test_formal_gate_requires_quality_bytes_speed_latency_and_memory(self):
        pairs = []
        for wave in (1, 2):
            for workload in WORKLOADS:
                tokens = list(range(256))
                common = {
                    "generated_tokens": 256,
                    "generated_token_ids": tokens,
                }
                pairs.append(
                    {
                        "wave": wave,
                        "workload": workload,
                        "exact": {
                            **common,
                            "metrics": {
                                "request_batched_expert_layers": 42,
                                "request_expert_bytes_read": 1_042,
                                "decode_tokens_per_second": 10.0,
                                "decode_latency_p95_seconds": 0.1,
                                "peak_memory_bytes": 1_000,
                                "approximation_mode": "exact",
                            },
                        },
                        "candidate": {
                            **common,
                            "metrics": {
                                "request_batched_expert_layers": 42,
                                "request_expert_bytes_read": 892,
                                "decode_tokens_per_second": 10.6,
                                "decode_latency_p95_seconds": 0.101,
                                "peak_memory_bytes": 1_010,
                                "approximation_mode": "learned-route-drop-lowest-1",
                            },
                        },
                    }
                )

        gate = _formal_gate(
            pairs,
            expert_count=1,
            expert_blob_bytes=1,
            safety_prerequisite=True,
        )

        self.assertTrue(gate["passed"])
        self.assertAlmostEqual(
            gate["aggregate_decode_logical_expert_byte_reduction_fraction"],
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
