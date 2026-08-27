from __future__ import annotations

import unittest

import numpy as np

from Scripts.benchmark_dspark_router_predictor import (
    FEATURE_SOURCES,
    PREDICTED_TOP_K,
    _aggregate_gate,
    _stable_topk,
    _summarize_candidate,
    _summarize_layer,
)


WORKLOADS = (
    "repeated",
    "code",
    "zh_technical",
    "mixed_math",
    "tool_like",
)


def _candidate(feature_source: str, top_k: int, *, recovered: bool) -> dict:
    target = [[0, 1, 2, 3, 4, 5]] * 5
    predicted = (
        [list(range(top_k))] * 5
        if recovered
        else [list(range(6, 6 + top_k))] * 5
    )
    layers = [
        _summarize_layer(
            layer=layer,
            target_routes=target,
            predicted_routes=predicted,
            expert_count=64,
            expert_blob_bytes=10,
        )
        for layer in range(3, 43)
    ]
    return _summarize_candidate(
        feature_source=feature_source,
        predicted_top_k=top_k,
        layers=layers,
    )


def _worker(workload: str, *, passing_candidate: bool) -> dict:
    return {
        "workload": workload,
        "structural_gate": {"passed": True},
        "target_route_trace": {"scored_assignments": 1_200},
        "candidates": [
            _candidate(
                feature_source,
                top_k,
                recovered=(
                    passing_candidate
                    and feature_source == FEATURE_SOURCES[0]
                ),
            )
            for feature_source in FEATURE_SOURCES
            for top_k in PREDICTED_TOP_K
        ],
    }


class DSparkRouterPredictorBenchmarkTests(unittest.TestCase):
    def test_stable_topk_resolves_score_ties_by_lower_expert_id(self):
        scores = np.array([1.0, 4.0, 4.0, 2.0, 4.0], dtype=np.float32)

        self.assertEqual(_stable_topk(scores, 3), [1, 2, 4])

    def test_layer_summary_closes_useful_wasted_and_missed_unions(self):
        result = _summarize_layer(
            layer=3,
            target_routes=[[0, 1, 2, 3, 4, 5]] * 5,
            predicted_routes=[[0, 1, 2, 3, 4, 6]] * 5,
            expert_count=8,
            expert_blob_bytes=10,
        )

        self.assertEqual(result["recovered_assignments"], 25)
        self.assertEqual(result["useful_union_experts"], 5)
        self.assertEqual(result["wasted_union_experts"], 1)
        self.assertEqual(result["missed_union_experts"], 1)
        self.assertTrue(result["accounting_exact"])

    def test_gate_selects_smallest_eligible_top_k(self):
        rows = [_worker(name, passing_candidate=True) for name in WORKLOADS]

        gate = _aggregate_gate(rows)

        self.assertTrue(gate["structural"]["passed"])
        self.assertTrue(gate["continuation_passed"])
        self.assertEqual(gate["winner"]["feature_source"], FEATURE_SOURCES[0])
        self.assertEqual(gate["winner"]["predicted_top_k"], 6)
        self.assertEqual(
            gate["decision"],
            "continue_isolated_scratch_prefetch_gate",
        )

    def test_gate_stops_direct_transfer_when_recall_is_zero(self):
        rows = [_worker(name, passing_candidate=False) for name in WORKLOADS]

        gate = _aggregate_gate(rows)

        self.assertTrue(gate["structural"]["passed"])
        self.assertFalse(gate["continuation_passed"])
        self.assertIsNone(gate["winner"])
        self.assertEqual(
            gate["decision"],
            "stop_direct_frozen_router_transfer_training_required",
        )


if __name__ == "__main__":
    unittest.main()
