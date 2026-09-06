import hashlib
import unittest

from Scripts.benchmark_research_baseline import validate_run


class BaselineValidationTest(unittest.TestCase):
    def test_rejects_corrupt_tokens_and_wrong_mode(self):
        ids = list(range(256))
        metrics = dict(generated_token_ids=ids,
                       token_sha256=hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(),
                       prompt_token_sha256="prompt", approximation_mode="exact",
                       mtp_enabled=False, prompt_cache_reused_tokens=0)
        prompt = {"prompt_token_sha256": "prompt"}
        validate_run(metrics, prompt)
        for change in ({"token_sha256": "wrong"}, {"approximation_mode": "drop"},
                       {"generated_token_ids": ids[:-1]}, {"prompt_cache_reused_tokens": 1}):
            with self.assertRaises(ValueError):
                validate_run(metrics | change, prompt)
