from __future__ import annotations

import unittest

import numpy as np

from Scripts.benchmark_dspark_candidate_paths import (
    _accepted_prefix,
    _aggregate_gate,
    _is_feasible_improvement,
    _selector_sweep,
    _stable_topk,
)


def _candidate(
    identifier: str,
    tokens: list[int],
    *,
    log_probability: float,
    accepted: int,
    requested: int,
    missing: int,
) -> dict:
    return {
        "id": identifier,
        "tokens": tokens,
        "joint_log_probability": log_probability,
        "accepted_draft_tokens": accepted,
        "requested_hash_experts": requested,
        "missing_hash_experts": missing,
    }


def _worker_row(workload: str, *, feasible: bool) -> dict:
    return {
        "workload": workload,
        "selector_feasible_improvement": feasible,
        "structural_gate": {"passed": True},
        "configuration": {
            "temperature": 0,
            "top_p": 1,
            "first_speculative_round_only": True,
            "block_size": 5,
            "branch_width": 4,
            "beam_width": 8,
            "persistent_prompt_cache": False,
            "layer_major_prefill": False,
            "hash_prefetch": False,
            "adaptive_block": False,
            "fallback": False,
            "page_cache_probe": False,
            "expert_file_cache_policy": "cached",
        },
    }


class DSparkCandidatePathsBenchmarkTests(unittest.TestCase):
    def test_stable_topk_orders_score_then_lower_token_id(self):
        scores = np.array([1.0, 3.0, 3.0, 2.0, 3.0], dtype=np.float32)

        self.assertEqual(_stable_topk(scores, 3), [1, 2, 4])

    def test_accepted_prefix_stops_at_first_mismatch(self):
        self.assertEqual(_accepted_prefix([10, 11, 99, 13], [10, 11, 12, 13]), 2)

    def test_storage_weight_can_select_a_feasible_path(self):
        baseline = _candidate(
            "baseline",
            [1, 1, 1, 1, 1],
            log_probability=-1.0,
            accepted=3,
            requested=10,
            missing=5,
        )
        alternative = _candidate(
            "alternative",
            [1, 1, 2, 1, 1],
            log_probability=-1.2,
            accepted=3,
            requested=9,
            missing=4,
        )

        selections = _selector_sweep([baseline, alternative], baseline)

        probability_only = next(
            row
            for row in selections
            if row["requested_weight"] == 0 and row["missing_weight"] == 0
        )
        self.assertEqual(probability_only["selected_candidate_id"], "baseline")
        self.assertTrue(
            any(row["selector_feasible_improvement"] for row in selections)
        )

    def test_lower_acceptance_is_not_a_feasible_storage_improvement(self):
        baseline = _candidate(
            "baseline",
            [1, 1, 1, 1, 1],
            log_probability=-1.0,
            accepted=4,
            requested=10,
            missing=5,
        )
        alternative = _candidate(
            "alternative",
            [1, 1, 2, 1, 1],
            log_probability=-1.2,
            accepted=2,
            requested=8,
            missing=3,
        )

        self.assertFalse(_is_feasible_improvement(alternative, baseline))

    def test_aggregate_gate_continues_only_after_a_feasible_workload(self):
        workloads = (
            "repeated",
            "code",
            "zh_technical",
            "mixed_math",
            "tool_like",
        )
        stopped = _aggregate_gate(
            [_worker_row(name, feasible=False) for name in workloads]
        )
        continued = _aggregate_gate(
            [
                _worker_row(name, feasible=name == "code")
                for name in workloads
            ]
        )

        self.assertEqual(stopped["decision"], "stop_current_markov_beam_candidate")
        self.assertEqual(
            continued["decision"],
            "continue_to_multi_round_runtime_prototype",
        )
        self.assertEqual(
            continued["continuation"]["selector_feasible_workloads"],
            ["code"],
        )


if __name__ == "__main__":
    unittest.main()
