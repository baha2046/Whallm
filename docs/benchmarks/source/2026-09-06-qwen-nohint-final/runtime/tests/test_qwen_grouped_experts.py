import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.expert_cache import QwenBatchedExperts, QwenExpertWeights
from deepseek_v4_ssd.model import RuntimeConfig


class GroupedExpertsTests(unittest.TestCase):
    def weights(self):
        rng = np.random.default_rng(20260906)
        gate, gate_scales = mx.quantize(mx.array(rng.normal(0, .1, (16, 128, 64)), dtype=mx.bfloat16),
            group_size=32, bits=4, mode="mxfp4")
        down, down_scales = mx.quantize(mx.array(rng.normal(0, .1, (16, 64, 64)), dtype=mx.bfloat16),
            group_size=32, bits=4, mode="mxfp4")
        return QwenBatchedExperts(gate, gate_scales, down, down_scales)

    def test_grouping_preserves_each_expert_and_short_input_fallback(self):
        weights = self.weights()
        calls = []
        cache = SimpleNamespace(current_batched=lambda layer: weights, record_gather_qmm=calls.append)
        experts = qwen.StreamingExperts(0, cache)
        self.assertFalse(RuntimeConfig().qwen_grouped_experts)
        self.assertFalse(experts.grouped_prefill)
        rng = np.random.default_rng(61)
        for length, top_k in ((1, 10), (6, 10), (8, 8), (128, 10), (1024, 10)):
            with self.subTest(length=length, top_k=top_k):
                value = mx.array(rng.normal(0, .2, (1, length, 64)), dtype=mx.bfloat16)
                indices = mx.array((np.arange(length)[:, None] * 7 + np.arange(top_k)[None, :] * 3) % 16, dtype=mx.int32)[None]
                experts.grouped_prefill = False
                expected = experts(value, indices)
                mx.eval(expected)
                experts.grouped_prefill = True
                with patch.object(qwen, "_gather_sort", wraps=qwen._gather_sort) as sort:
                    actual = experts(value, indices)
                    mx.eval(actual)
                    self.assertEqual(sort.call_count, int(length * top_k >= 64))
                self.assertEqual(actual.shape, (1, length, top_k, 64))
                self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(calls, [2] * 10)

    def test_individual_expert_decode_keeps_original_path(self):
        batched = self.weights()
        individual = [QwenExpertWeights(batched.gate_up[i], batched.gate_up_scales[i],
                      batched.down[i], batched.down_scales[i]) for i in range(16)]
        acquisitions = []
        def get_many(layer, ids):
            acquisitions.append((layer, ids))
            return SimpleNamespace(individual_weights=individual, slots={i: i for i in range(16)})
        cache = SimpleNamespace(current_batched=lambda layer: None, get_many=get_many)
        experts = qwen.StreamingExperts(3, cache)
        value = mx.full((1, 1, 64), .25, dtype=mx.bfloat16)
        indices = mx.array([[[9, 2, 7, 1]]])
        expected = experts(value, indices)
        mx.eval(expected)
        experts.grouped_prefill = True
        with patch.object(qwen, "_gather_sort", side_effect=AssertionError("Decode was grouped")):
            actual = experts(value, indices)
            mx.eval(actual)
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(acquisitions, [(3, [9, 2, 7, 1])] * 2)


if __name__ == "__main__":
    unittest.main()
