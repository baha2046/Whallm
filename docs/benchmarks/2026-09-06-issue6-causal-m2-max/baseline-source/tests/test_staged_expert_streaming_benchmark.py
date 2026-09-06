from __future__ import annotations

import unittest
from pathlib import Path

from Scripts.benchmark_staged_expert_streaming import (
    _decision,
    _staged_regions,
)
from deepseek_v4_ssd.manifest import (
    EXPERT_BLOB_SIZE,
    EXPERT_REGIONS,
    InstalledModel,
    Tensor,
)


def _model() -> InstalledModel:
    regions = tuple(
        Tensor(name, dtype, shape, offset, length)
        for name, dtype, shape, offset, length in EXPERT_REGIONS
    )
    return InstalledModel(
        root=Path("/fixture"),
        model_id="fixture",
        revision="fixture",
        layer_count=43,
        expert_count=256,
        selected_expert_count=6,
        expert_blob_size=EXPERT_BLOB_SIZE,
        common_tensors=(),
        expert_regions=regions,
    )


def _summary(rows: int, hidden: float, median: float, p95: float) -> dict:
    return {
        "rows": rows,
        "w2_read_hidden_fraction_median": hidden,
        "complete_median_change_fraction": median,
        "complete_p95_change_fraction": p95,
    }


class StagedExpertStreamingBenchmarkTests(unittest.TestCase):
    def test_staged_region_budget_preserves_every_canonical_byte(self):
        w13, w2 = _staged_regions(_model())

        self.assertEqual(sum(region.length for region in w13.values()), 8_912_896)
        self.assertEqual(sum(region.length for region in w2.values()), 4_456_448)
        self.assertEqual(
            sum(region.length for region in (*w13.values(), *w2.values())),
            EXPERT_BLOB_SIZE,
        )
        self.assertEqual(set(w13).union(w2), {row[0] for row in EXPERT_REGIONS})

    def test_decision_continues_when_one_block_shape_passes_every_gate(self):
        result = _decision(
            {"passed": True},
            [
                _summary(1, 0.9, -0.5, -0.5),
                _summary(4, 0.25, -0.06, 0.04),
                _summary(8, 0.19, -0.20, -0.20),
            ],
        )

        self.assertTrue(result["continue_to_runtime_token_hash_prototype"])
        self.assertTrue(result["shape_candidates"][0]["passed"])
        self.assertFalse(result["shape_candidates"][1]["passed"])

    def test_decision_stops_on_correctness_or_end_to_end_regression(self):
        summaries = [
            _summary(4, 0.80, -0.04, -0.20),
            _summary(8, 0.80, -0.20, 0.06),
        ]

        performance_stop = _decision({"passed": True}, summaries)
        correctness_stop = _decision(
            {"passed": False},
            [_summary(4, 0.80, -0.20, -0.20)],
        )

        self.assertFalse(
            performance_stop["continue_to_runtime_token_hash_prototype"]
        )
        self.assertFalse(
            correctness_stop["continue_to_runtime_token_hash_prototype"]
        )


if __name__ == "__main__":
    unittest.main()
