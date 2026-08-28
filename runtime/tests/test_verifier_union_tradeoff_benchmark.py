from __future__ import annotations

import unittest

from Scripts.benchmark_verifier_union_tradeoff import (
    _partition,
    _shape_check,
    _tradeoff_decision,
)


def _row(mode: str) -> dict:
    metrics = {
        "dspark_rounds": 1,
        "dspark_proposed_tokens": 5,
        "dspark_accepted_tokens": 5,
        "dspark_committed_tokens": 6,
        "dspark_replay_expert_bytes_read": 0,
        "dspark_block_verification_rounds": 0,
        "dspark_sequential_verification_rounds": 0,
        "dspark_hybrid_verification_rounds": 0,
        "dspark_last_verification_mode": mode,
        "dspark_verification_expert_union_calls": (
            258 if mode == "sequential" else 43
        ),
        "dspark_verification_routed_expert_assignments": 1_548,
        "dspark_hybrid_attention_layers": 0,
        "dspark_hybrid_attention_token_calls": 0,
        "dspark_hybrid_ffn_token_calls": 0,
        "dspark_hybrid_moe_token_calls": 0,
    }
    round_key = {
        "sequential": "dspark_sequential_verification_rounds",
        "block": "dspark_block_verification_rounds",
        "hybrid": "dspark_hybrid_verification_rounds",
    }[mode]
    metrics[round_key] = 1
    if mode == "hybrid":
        metrics.update(
            dspark_hybrid_attention_layers=43,
            dspark_hybrid_attention_token_calls=258,
            dspark_hybrid_ffn_token_calls=258,
            dspark_hybrid_moe_token_calls=258,
        )
    return {
        "id": f"fixture-{mode}",
        "mode": "grouped" if mode == "block" else mode,
        "metrics": metrics,
    }


def _summary(
    *,
    calls: float,
    assignments: float = 1_548,
    experts: float,
    reused: float,
    target_bytes: float,
    verification_seconds: float,
) -> dict:
    return {
        "dspark_verification_expert_union_calls": calls,
        "dspark_verification_routed_expert_assignments": assignments,
        "dspark_verification_expert_union_experts": experts,
        "dspark_verification_expert_union_reused_assignments": reused,
        "dspark_target_expert_bytes_read": target_bytes,
        "dspark_verification_seconds": verification_seconds,
    }


class VerifierUnionTradeoffBenchmarkTests(unittest.TestCase):
    def test_shape_gate_accepts_all_three_predeclared_modes(self):
        for mode in ("sequential", "block", "hybrid"):
            with self.subTest(mode=mode):
                self.assertTrue(_shape_check(_row(mode))["exact"])

    def test_shape_gate_rejects_hybrid_per_token_acquisition(self):
        row = _row("hybrid")
        row["metrics"]["dspark_verification_expert_union_calls"] = 258

        result = _shape_check(row)

        self.assertFalse(result["exact"])
        self.assertFalse(result["criteria"]["union_calls_exact"])

    def test_partition_requires_a_closed_zero_failure_identity(self):
        metrics = {
            "logical": 100,
            "classified": 100,
            "resident": 20,
            "nonresident": 80,
            "unclassified": 0,
            "failures": 0,
        }

        result = _partition(
            metrics,
            logical_key="logical",
            classified_key="classified",
            resident_key="resident",
            nonresident_key="nonresident",
            unclassified_key="unclassified",
            failures_key="failures",
        )

        self.assertTrue(result["accounting_exact"])

    def test_tradeoff_marks_material_grouped_qmm_loss_after_union_passes(self):
        summaries = {
            "sequential": _summary(
                calls=258,
                experts=1_548,
                reused=0,
                target_bytes=1_000,
                verification_seconds=2.0,
            ),
            "grouped": _summary(
                calls=43,
                experts=900,
                reused=648,
                target_bytes=900,
                verification_seconds=1.0,
            ),
            "hybrid": _summary(
                calls=43,
                experts=900,
                reused=648,
                target_bytes=950,
                verification_seconds=1.3,
            ),
        }

        result = _tradeoff_decision(summaries)

        self.assertTrue(result["hybrid_union_contract_passed"])
        self.assertTrue(
            result["grouped_execution_advantage_material_at_20_percent"]
        )
        self.assertTrue(
            result[
                "dedupe_insufficient_to_overcome_token_shaped_execution_cost"
            ]
        )


if __name__ == "__main__":
    unittest.main()
