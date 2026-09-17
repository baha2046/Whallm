"""V4.1 layer layout, real quantized matmuls, and request-local buffer ownership."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from deepseek_v4_ssd.cancellation import cancellation_scope, GenerationCancelled
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import _StreamingSwitchGLU
from runtime.tests.test_prefill_slot_release import fixture


class V41LayerBufferTests(unittest.TestCase):
    def test_quantized_projections_match_individual_experts_across_buffer_reuse(self):
        mx.random.seed(17)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experts').mkdir()
            regions = []
            for layer in range(4):
                data = bytearray()
                for expert in range(3):
                    offset = 0
                    for name in ('w1', 'w2', 'w3'):
                        weight, scale = mx.quantize(mx.random.normal((128, 128)) * 0.05,
                                                   group_size=32, bits=4, mode='mxfp4')
                        for suffix, array, dtype, shape in (
                            ('weight', weight, 'I8', (128, 64)),
                            ('scale', scale, 'F8_E8M0', (128, 4)),
                        ):
                            raw = np.array(array).tobytes()
                            if layer == expert == 0:
                                regions.append(Tensor(f'{name}.{suffix}', dtype, shape, offset, len(raw)))
                            data.extend(raw)
                            offset += len(raw)
                (root / f'experts/layer_{layer:02d}.bin').write_bytes(data)
            model = InstalledModel(root, 'fixture', 'fixture', 4, 3, 2, offset, (), tuple(regions),
                                   model_kind='deepseek-v4.1')
            x = mx.random.normal((1, 33, 128))
            indices = mx.array([[[i % 3, (i + 1) % 3] for i in range(33)]], mx.uint32)
            with ExpertCache(model, slots=3, read_workers=1) as cache:
                projection = _StreamingSwitchGLU(0, cache, lambda up, gate: up * nn.silu(gate))
                buffers = []
                with cache.reuse_layer_buffers():
                    for layer in range(4):
                        with cache.batched_layer(layer) as batch:
                            buffers.append(id(cache._active_prefetch_trace.job.packed))
                            if layer < 3:
                                cache.prefetch_layer(layer + 1)
                            # NumPy's view exposes actual byte strides, not the logical shape.
                            for value in (batch.w13, batch.w13_scales, batch.w2, batch.w2_scales):
                                self.assertTrue(np.asarray(value).flags.c_contiguous)
                            individual = cache.get_many(layer, [0, 1, 2]).individual_weights
                            for expert, weights in enumerate(individual):
                                for name in ('w1', 'w2', 'w3', 'w13', 'w1_scales', 'w2_scales', 'w3_scales', 'w13_scales'):
                                    self.assertTrue(mx.array_equal(getattr(batch, name)[expert], getattr(weights, name)).item())
                            actual = projection._gather_qmm(x, indices, batch)
                            expected = []
                            for token in range(33):
                                outputs = []
                                for expert in indices[0, token].tolist():
                                    weights = individual[expert]
                                    def qmm(value, w, s):
                                        return mx.quantized_matmul(value, w, s, transpose=True,
                                                                   group_size=32, bits=4, mode='mxfp4')
                                    up = qmm(x[:, token], weights.w3, weights.w3_scales)
                                    gate = qmm(x[:, token], weights.w1, weights.w1_scales)
                                    outputs.append(qmm(up * nn.silu(gate), weights.w2, weights.w2_scales))
                                expected.append(mx.stack(outputs, axis=1))
                            expected = mx.stack(expected, axis=1)
                            self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
                self.assertEqual(len(set(buffers)), 2)
                self.assertIsNone(cache._layer_buffers)
                self.assertFalse(cache._prefetched_layers)

    def test_cancelled_read_releases_pool_and_next_request_can_run(self):
        with TemporaryDirectory() as directory:
            model = replace(fixture(Path(directory)), model_kind='deepseek-v4.1')
            with ExpertCache(model, slots=2, read_workers=1) as cache:
                cancelled = threading.Event()
                original = cache._read_expert_ids
                def read(*args):
                    result = original(*args)
                    cancelled.set()
                    return result
                with patch.object(cache, '_read_expert_ids', side_effect=read):
                    with cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
                        with cache.reuse_layer_buffers(), cache.batched_layer(0):
                            self.fail('partial layer was exposed')
                self.assertIsNone(cache._layer_buffers)
                self.assertIsNone(cache._batched_layer)
                self.assertFalse(cache._prefetched_layers)
                with cache.reuse_layer_buffers(), cache.batched_layer(0) as batch:
                    self.assertEqual(batch.w1_scales[:, 0, 0].tolist(), [4, 28])
