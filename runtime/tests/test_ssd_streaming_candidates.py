"""Native-weight checks for research-only reader and union patches."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import SparseMoE
from runtime.tests.test_qwen_resident_block import fixture
from research.ssd_streaming_hotness import install_cost_hotness


class StreamingCandidateTests(unittest.TestCase):

    def test_cost_hotness_rebuilds_prices_without_stale_heap_entries(self):
        from runtime.tests.test_expert_eviction_policy import fixture as small_fixture
        from contextlib import ExitStack
        methods = ("__init__", "_touch", "_eviction_rank", "_decay_if_needed", "_read_expert_into_pool")
        with ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            for name in methods:
                stack.enter_context(patch.object(ExpertCache, name, getattr(ExpertCache, name)))
            model = small_fixture(Path(directory), layers=2)
            install_cost_hotness()
            with ExpertCache(model, slots=4) as cache:
                cache.get_many(0, [0, 1])
                cache.get_many(1, [0, 1])
                cache._cost_sum = [1.0, 4.0]
                cache._cost_count = [1, 1]
                cache._clock = 32
                cache._decay_if_needed()
                self.assertLess(cache._cost_active[0], cache._cost_active[1])
                self.assertEqual(cache._layer_reserve, 0)
                self.assertEqual(len(cache._heap), cache.resident_count)
                cache.get_many(1, [2])
                self.assertEqual(cache.resident_count, 4)
                self.assertTrue(all(item[0] == cache._eviction_rank(cache._entries[(item[3], item[4])])
                                    for item in cache._heap if (item[3], item[4]) in cache._entries and
                                    item[2] == cache._entries[(item[3], item[4])].version))


if __name__ == "__main__":
    unittest.main()
