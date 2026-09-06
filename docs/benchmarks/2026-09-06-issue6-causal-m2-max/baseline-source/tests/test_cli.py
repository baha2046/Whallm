from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deepseek_v4_ssd.cli import (
    _read_prompt,
    _select_approximation_mode,
    _token_sha256,
)


class CLITests(unittest.TestCase):
    def test_approximation_default_is_model_aware_and_exact_can_override(self):
        self.assertEqual(
            _select_approximation_mode(
                None,
                is_qwen=False,
                dspark_enabled=False,
            ),
            "learned-route-drop-lowest-1",
        )
        for is_qwen, dspark_enabled in ((True, False), (False, True)):
            with self.subTest(is_qwen=is_qwen, dspark_enabled=dspark_enabled):
                self.assertEqual(
                    _select_approximation_mode(
                        None,
                        is_qwen=is_qwen,
                        dspark_enabled=dspark_enabled,
                    ),
                    "exact",
                )
                with self.assertRaises(ValueError):
                    _select_approximation_mode(
                        "learned-route-drop-lowest-1",
                        is_qwen=is_qwen,
                        dspark_enabled=dspark_enabled,
                    )

        self.assertEqual(
            _select_approximation_mode(
                "exact",
                is_qwen=False,
                dspark_enabled=False,
            ),
            "exact",
        )

    def test_prompt_file_and_token_hash_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.txt"
            path.write_text("fixed prompt", encoding="utf-8")

            self.assertEqual(_read_prompt(None, str(path)), "fixed prompt")
        self.assertEqual(
            _token_sha256([1, 2, 3]),
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d",
        )


if __name__ == "__main__":
    unittest.main()
