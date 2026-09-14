from types import SimpleNamespace
from unittest.mock import patch
import unittest
import mlx.core as mx
from deepseek_v4_ssd.qwen4_exp import StreamingExperts


class QwenReadyDecodeTests(unittest.TestCase):
    def test_ready_order_does_not_change_router_order_or_output_shape(self):
        cache = SimpleNamespace(ready_expert_decode=True, current_batched=lambda _: None,
                    iter_ready=lambda layer, ids: iter([(2, 20), (0, 3), (1, 8)]))
        experts = StreamingExperts(0, cache)
        value = mx.array([[1., 2., 3., 4.]])
        with patch.object(StreamingExperts, '_one', side_effect=lambda value, weight: value * weight):
            actual = experts(value, mx.array([[1, 2, 0]]))
        self.assertEqual(actual.shape, (1, 3, 4))
        self.assertTrue(mx.array_equal(actual, value[:, None, :] * mx.array([8, 20, 3])[None, :, None]).item())
