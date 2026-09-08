"""Numerical/budget gates for the new research verifier's expert stage."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import StreamingExperts
from research.qwen_short_block import union_experts
from runtime.tests.test_ssd_prefill_pipeline import fixture


class ShortBlockTests(unittest.TestCase):
    def test_one_union_matches_tokenwise_experts_without_changing_router_order(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            rng = np.random.default_rng(239)
            with ExpertCache(model, slots=32, read_workers=2) as cache:
                expert = StreamingExperts(0, cache)
                for block in (2, 4):
                    value = mx.array(rng.normal(0,.2,(1,block,128)), mx.bfloat16)
                    indices = mx.array(rng.integers(0,128,(1,block,4)), mx.int32)
                    expected = mx.concatenate([expert(value[:,t:t+1],indices[:,t:t+1]) for t in range(block)],axis=1)
                    mx.eval(expected)
                    with patch.object(cache,'get_many',wraps=cache.get_many) as acquire:
                        actual = union_experts(value,indices,cache,0)
                        self.assertEqual(acquire.call_count,1)
                    self.assertTrue(mx.isfinite(actual).all().item())
                    self.assertTrue(mx.array_equal(expected,actual).item())

    def test_oversize_block_rejected_before_acquisition(self):
        with self.assertRaisesRegex(ValueError,'1..4'):
            union_experts(mx.zeros((1,5,128)),mx.zeros((1,5,4),mx.int32),None,0)


if __name__ == '__main__':
    unittest.main()
