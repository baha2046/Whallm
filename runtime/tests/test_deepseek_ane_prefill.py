from types import SimpleNamespace
from unittest.mock import patch
import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from deepseek_v4_ssd.ane_prefill import install_deepseek_ane_prefill


class DeepSeekANETests(unittest.TestCase):
    def test_quantized_gpu_decode_ane_prefill_and_failure_fallback(self):
        mx.random.seed(16)
        linear = nn.QuantizedLinear.from_linear(nn.Linear(32, 512, bias=False), group_size=32, bits=8, mode='mxfp8')
        original = mx.array(linear.weight)
        x = mx.random.normal((1, 1024, 32)).astype(mx.float16)
        projections = []
        class Projection:
            spatial = 1024
            input_channels = 32
            output_channels = 256
            fail = False
            def __init__(self, weight): self.weight = weight
            def evaluate(self, value):
                if self.fail: raise RuntimeError('test device failure')
                return value @ self.weight.T
            def close(self): pass
        def create(weight, spatial):
            result = Projection(weight)
            projections.append(result)
            return result
        model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(wq_b=linear))])
        with patch('deepseek_v4_ssd.ane_prefill._ANELibrary.open', return_value=SimpleNamespace(create=create)):
            controller = install_deepseek_ane_prefill(model, True, 0.5)
        self.assertTrue(controller.active)
        wrapper = model.layers[0].attn.wq_b
        self.assertTrue(mx.array_equal(linear.weight, original).item())
        self.assertTrue(mx.array_equal(wrapper(x[:, :1]), linear(x[:, :1])).item())
        np.testing.assert_allclose(np.array(wrapper(x)), np.array(linear(x)), rtol=0.01, atol=0.005)
        projections[0].fail = True
        self.assertTrue(mx.array_equal(wrapper(x), linear(x)).item())
        self.assertFalse(controller.active)

    def test_compile_failure_does_not_publish_partial_wrappers(self):
        first, second = nn.Linear(32, 512, bias=False), nn.Linear(32, 512, bias=False)
        model = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(wq_b=x)) for x in (first, second)])
        with patch('deepseek_v4_ssd.ane_prefill._ANELibrary.open', side_effect=RuntimeError('unavailable')):
            controller = install_deepseek_ane_prefill(model, True, 0.5)
        self.assertFalse(controller.active)
        self.assertIs(model.layers[0].attn.wq_b, first)
        self.assertIs(model.layers[1].attn.wq_b, second)
