from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts"))

from benchmark_block_prompt_cache import (  # noqa: E402
    SHARED_CHECKPOINT_TOKENS,
    _gate,
)


def _run(reused: int, tokens: list[int], errors: int = 0) -> dict:
    return {
        "generated_token_ids": tokens,
        "token_sha256": "same" if tokens == [7] else "different",
        "metrics": {
            "prompt_cache_reused_tokens": reused,
            "prompt_cache_write_errors": errors,
        },
    }


def _bundle(metadata: str = "meta", data: str = "data", hits: int = 1) -> dict:
    return {
        "metadata_sha256": metadata,
        "data_sha256": data,
        "access": {"reuseCount": hits},
    }


class BlockPromptCacheBenchmarkTests(unittest.TestCase):
    def test_gate_requires_restart_partial_match_parity_and_immutable_payload(self):
        result = _gate(
            [
                _run(0, [1]),
                _run(SHARED_CHECKPOINT_TOKENS, [7]),
                _run(0, [7]),
            ],
            442,
            _bundle(),
            _bundle(),
            {"contract": True},
        )

        self.assertTrue(result["passed"])

    def test_gate_rejects_wrong_reuse_changed_payload_and_output(self):
        result = _gate(
            [
                _run(0, [1]),
                _run(SHARED_CHECKPOINT_TOKENS - 1, [8]),
                _run(0, [7]),
            ],
            442,
            _bundle(),
            _bundle(metadata="changed", hits=0),
            {"contract": False},
        )

        self.assertFalse(result["passed"])
        self.assertFalse(
            result["criteria"]["restart_branch_reuses_exact_checkpoint"]
        )
        self.assertFalse(result["criteria"]["shared_metadata_immutable"])
        self.assertFalse(
            result["criteria"]["persistent_and_cold_branch_tokens_exact"]
        )


if __name__ == "__main__":
    unittest.main()
