from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts"))

from benchmark_dspark_prompt_cache import (  # noqa: E402
    EXPECTED_OUTPUT_TOKEN_SHA256,
    _gate,
)


def _row(source: str, reused: int, main: int, draft: int) -> dict:
    return {
        "token_sha256": EXPECTED_OUTPUT_TOKEN_SHA256,
        "main_logical_expert_bytes": main,
        "draft_logical_expert_bytes": draft,
        "combined_logical_expert_bytes": main + draft,
        "metrics": {
            "dspark_prompt_cache_source": source,
            "prompt_cache_reused_tokens": reused,
            "prompt_cache_write_errors": 0,
        },
    }


class DSparkPromptCacheBenchmarkTests(unittest.TestCase):
    def test_gate_requires_sources_tokens_hashes_and_byte_reduction(self):
        result = _gate(
            [
                _row("none", 0, 80, 20),
                _row("memory", 127, 30, 10),
                _row("persistent", 127, 30, 10),
            ],
            128,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["memory_combined_byte_change_fraction"], -0.6)

    def test_gate_rejects_a_half_restored_or_oversized_candidate(self):
        result = _gate(
            [
                _row("none", 0, 80, 20),
                _row("memory", 126, 30, 10),
                _row("persistent", 127, 50, 10),
            ],
            128,
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["criteria"]["reused_token_counts_exact"])
        self.assertFalse(
            result["criteria"][
                "persistent_combined_logical_expert_bytes_reduced_at_least_50_percent"
            ]
        )


if __name__ == "__main__":
    unittest.main()
