from __future__ import annotations

import unittest

from Scripts.benchmark_adaptive_expert_prefill_runtime import (
    _percentile,
    _runtime_gate,
)


class AdaptiveExpertPrefillRuntimeBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _row(mode: str, wave: int, ttft: float) -> dict:
        adaptive = mode == "adaptive"
        read_experts = 70 if adaptive else 0
        misses = 3
        blob = 24
        full = 2 * 50
        return {
            "mode": mode,
            "wave": wave,
            "sequence": 1 if (mode == "adaptive") == (wave == 1) else 2,
            "configuration": {
                "adaptive_expert_prefill_threshold": 0.9 if adaptive else None,
                "batched_expert_prefill": True,
                "dspark_enabled": False,
                "expert_file_cache_policy": "bypass",
                "layer_major_prefill": True,
                "persistent_prompt_cache": False,
                "staged_expert_streaming": False,
            },
            "prompt_token_sha256": "prompt",
            "generated_tokens": 2,
            "generated_token_ids": [1, 2],
            "token_sha256": "tokens",
            "metrics": {
                "request_seconds": ttft + 0.2,
                "external_wall_seconds": ttft + 0.3,
                "time_to_first_token_seconds": ttft,
                "peak_memory_bytes": 95 if adaptive else 100,
                "request_expert_cache_misses": misses,
                "request_expert_bytes_read": (
                    (read_experts if adaptive else full) + misses
                )
                * blob,
                "request_adaptive_prefill_planned_layers": 2 if adaptive else 0,
                "request_adaptive_prefill_full_layers": 0,
                "request_adaptive_prefill_selective_layers": 2 if adaptive else 0,
                "request_adaptive_prefill_union_experts": (
                    read_experts if adaptive else 0
                ),
                "request_adaptive_prefill_read_experts": (
                    read_experts if adaptive else 0
                ),
                "request_adaptive_prefill_bytes_read": (
                    read_experts * blob if adaptive else 0
                ),
                "request_adaptive_prefill_avoided_bytes": (
                    (full - read_experts) * blob if adaptive else 0
                ),
            },
        }

    def test_percentile_interpolates_two_run_p95(self):
        self.assertEqual(_percentile([10.0, 20.0], 0.95), 19.5)

    def test_gate_continues_an_exact_faster_candidate(self):
        rows = [
            self._row("adaptive", 1, 8.0),
            self._row("control", 1, 10.0),
            self._row("control", 2, 10.0),
            self._row("adaptive", 2, 8.0),
        ]

        gate = _runtime_gate(
            rows,
            max_tokens=2,
            active_layers=2,
            expert_count=50,
            expert_blob_bytes=24,
            threshold_invariant=True,
        )

        self.assertTrue(gate["correctness"]["passed"])
        self.assertTrue(gate["performance"]["passed"])
        self.assertEqual(
            gate["decision"],
            "continue_full_multiworkload_threshold_screen_default_off",
        )

    def test_gate_stops_an_exact_candidate_with_slower_ttft(self):
        rows = [
            self._row("adaptive", 1, 12.0),
            self._row("control", 1, 10.0),
            self._row("control", 2, 10.0),
            self._row("adaptive", 2, 12.0),
        ]

        gate = _runtime_gate(
            rows,
            max_tokens=2,
            active_layers=2,
            expert_count=50,
            expert_blob_bytes=24,
            threshold_invariant=True,
        )

        self.assertTrue(gate["correctness"]["passed"])
        self.assertFalse(gate["performance"]["passed"])
        self.assertFalse(
            gate["performance"]["criteria"][
                "median_ttft_improves_at_least_5_percent"
            ]
        )
        self.assertEqual(
            gate["decision"],
            "stop_runtime_candidate_keep_default_off",
        )


if __name__ == "__main__":
    unittest.main()
