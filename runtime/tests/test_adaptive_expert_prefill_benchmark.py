from __future__ import annotations

import unittest

from Scripts.benchmark_adaptive_expert_prefill import (
    EXPECTED_LAYER_MAJOR_TOKENS,
    _eligibility_gate,
    _summarize_trace,
)


def _trace(unions: list[int], *, experts: int = 10) -> dict:
    histograms = []
    chunk_histograms = []
    assignments = EXPECTED_LAYER_MAJOR_TOKENS * 2
    for union in unions:
        histogram = [0] * experts
        for index in range(union):
            histogram[index] = 1
        histogram[0] += assignments - union
        histograms.append(histogram)
        chunk_histograms.append([list(histogram)])
    histograms.append([0] * experts)
    chunk_histograms.append([])
    return {
        "format": 1,
        "layer_count": len(histograms),
        "expert_count": experts,
        "selected_expert_count": 2,
        "expert_blob_size": 100,
        "prefill_histograms": histograms,
        "prefill_chunk_histograms": chunk_histograms,
    }


def _workload(name: str, summary: dict) -> dict:
    metrics = {
        "prompt_token_sha256": f"prompt-{name}",
        "generated_token_ids": [1, 2],
        "token_sha256": f"output-{name}",
        "prompt_tokens": 4_096,
        "layer_major_prefill_tokens": 4_095,
    }
    return {
        "name": name,
        "reference": {"metrics": dict(metrics)},
        "trace": {"metrics": dict(metrics)},
        "route_summary": summary,
    }


class AdaptiveExpertPrefillBenchmarkTests(unittest.TestCase):
    def test_trace_summary_applies_full_and_selective_thresholds(self):
        summary = _summarize_trace(
            _trace([6, 8, 9]),
            expected_layers=4,
            expert_count=10,
            selected_experts=2,
            expert_blob_bytes=100,
        )

        self.assertEqual(summary["active_expert_layers"], 3)
        self.assertEqual(summary["non_layer_major_layers"], [3])
        self.assertEqual(summary["union_experts"]["median"], 8)
        seventy = summary["thresholds"]["70_percent"]
        ninety = summary["thresholds"]["90_percent"]
        self.assertEqual(seventy["full_layers"], 2)
        self.assertEqual(seventy["selective_layers"], 1)
        self.assertEqual(ninety["full_layers"], 0)
        self.assertEqual(ninety["read_experts"], 23)

    def test_gate_continues_when_three_workloads_save_at_least_ten_percent(self):
        summaries = [
            _summarize_trace(
                _trace([7] * 42),
                expected_layers=43,
                expert_count=10,
                selected_experts=2,
                expert_blob_bytes=100,
            )
            for _ in range(5)
        ]
        result = _eligibility_gate(
            [_workload(f"w{index}", summary) for index, summary in enumerate(summaries)]
        )

        self.assertTrue(result["correctness"]["passed"])
        self.assertTrue(result["continuation_passed"])
        self.assertEqual(result["eligible_thresholds"], [0.7, 0.8, 0.9])

    def test_gate_stops_on_token_mismatch(self):
        summary = _summarize_trace(
            _trace([7] * 42),
            expected_layers=43,
            expert_count=10,
            selected_experts=2,
            expert_blob_bytes=100,
        )
        workloads = [_workload(f"w{index}", summary) for index in range(5)]
        workloads[0]["trace"]["metrics"]["generated_token_ids"] = [9, 9]

        result = _eligibility_gate(workloads)

        self.assertFalse(result["correctness"]["passed"])
        self.assertFalse(result["continuation_passed"])


if __name__ == "__main__":
    unittest.main()
