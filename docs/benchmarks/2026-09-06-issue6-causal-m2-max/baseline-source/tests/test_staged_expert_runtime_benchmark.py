from __future__ import annotations

import unittest

from Scripts.benchmark_staged_expert_runtime import _runtime_gate


EXPERT_BYTES = 15
W13_BYTES = 10
W2_BYTES = 5


def _row(
    wave: int,
    mode: str,
    sequence: int,
    *,
    request: float = 10.0,
    decode: float = 10.0,
    p95: float = 0.1,
    memory: float = 100.0,
    logical_bytes: int = 1_000,
    evictions: int = 10,
) -> dict:
    staged = mode == "staged"
    reads = 2 if staged else 0
    return {
        "wave": wave,
        "mode": mode,
        "sequence": sequence,
        "configuration": {
            "staged_expert_streaming": staged,
            "ready_expert_decode": True,
            "expert_file_cache_policy": "bypass",
            "layer_major_prefill": False,
            "persistent_prompt_cache": False,
            "dspark_enabled": False,
        },
        "prompt_token_sha256": "prompt",
        "generated_tokens": 32,
        "token_sha256": "output",
        "metrics": {
            "request_seconds": request,
            "external_wall_seconds": request + 0.01,
            "time_to_first_token_seconds": 5.0,
            "decode_tokens_per_second": decode,
            "decode_latency_p50_seconds": 0.08,
            "decode_latency_p95_seconds": p95,
            "peak_memory_bytes": memory,
            "request_expert_cache_hits": 20,
            "request_expert_cache_misses": 30,
            "request_expert_evictions": evictions,
            "request_expert_bytes_read": logical_bytes,
            "request_staged_expert_reads": reads,
            "request_staged_w13_bytes_read": reads * W13_BYTES,
            "request_staged_w2_bytes_read": reads * W2_BYTES,
            "request_staged_read_seconds": 1.0 if staged else 0.0,
            "request_staged_w2_wait_seconds": 0.2 if staged else 0.0,
            "request_staged_first_stage_submit_seconds": (
                0.1 if staged else 0.0
            ),
        },
    }


def _rows(
    *,
    candidate_request: float = 9.4,
    candidate_decode: float = 10.6,
) -> list[dict]:
    rows = []
    orders = (
        ("control", "staged"),
        ("staged", "control"),
        ("control", "staged"),
        ("staged", "control"),
    )
    for wave, order in enumerate(orders, start=1):
        for sequence, mode in enumerate(order, start=1):
            rows.append(
                _row(
                    wave,
                    mode,
                    sequence,
                    request=(candidate_request if mode == "staged" else 10.0),
                    decode=(candidate_decode if mode == "staged" else 10.0),
                    p95=(0.104 if mode == "staged" else 0.1),
                    memory=(104.0 if mode == "staged" else 100.0),
                )
            )
    return rows


class StagedExpertRuntimeBenchmarkTests(unittest.TestCase):
    def test_gate_accepts_exact_outputs_and_predeclared_performance_thresholds(self):
        result = _runtime_gate(
            _rows(),
            max_tokens=32,
            expert_blob_bytes=EXPERT_BYTES,
            w13_bytes=W13_BYTES,
            w2_bytes=W2_BYTES,
        )

        self.assertTrue(result["correctness"]["passed"])
        self.assertTrue(result["performance"]["passed"])
        self.assertEqual(
            result["decision"],
            "continue_multi_workload_long_decode_default_off",
        )

    def test_gate_stops_a_correct_but_slow_candidate(self):
        result = _runtime_gate(
            _rows(candidate_request=10.1, candidate_decode=9.9),
            max_tokens=32,
            expert_blob_bytes=EXPERT_BYTES,
            w13_bytes=W13_BYTES,
            w2_bytes=W2_BYTES,
        )

        self.assertTrue(result["correctness"]["passed"])
        self.assertFalse(result["performance"]["passed"])
        self.assertEqual(
            result["decision"],
            "stop_runtime_candidate_keep_default_off",
        )

    def test_gate_rejects_increased_logical_bytes_and_broken_split_accounting(self):
        rows = _rows()
        staged = next(row for row in rows if row["mode"] == "staged")
        staged["metrics"]["request_expert_bytes_read"] = 1_001
        staged["metrics"]["request_staged_w2_bytes_read"] -= 1

        result = _runtime_gate(
            rows,
            max_tokens=32,
            expert_blob_bytes=EXPERT_BYTES,
            w13_bytes=W13_BYTES,
            w2_bytes=W2_BYTES,
        )

        self.assertFalse(result["correctness"]["passed"])
        self.assertFalse(
            result["correctness"]["criteria"][
                "logical_expert_bytes_never_increased"
            ]
        )
        self.assertFalse(
            result["correctness"]["criteria"][
                "staged_read_accounting_closes"
            ]
        )


if __name__ == "__main__":
    unittest.main()
