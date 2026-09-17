from __future__ import annotations

import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel, Tensor, QWEN_EXPERT_REGIONS
from deepseek_v4_ssd.model import layer_major_prefill
from deepseek_v4_ssd.model_support.qwen import _qwen_layer_major_prefill


def fixture(root):
    (root / 'experts').mkdir()
    names = ('w1.weight', 'w1.scale', 'w2.weight', 'w2.scale', 'w3.weight', 'w3.scale')
    regions = tuple(Tensor(name, 'I8' if name.endswith('weight') else 'F8_E8M0',
                           (1, 4), index * 4, 4) for index, name in enumerate(names))
    (root / 'experts/layer_00.bin').write_bytes(bytes(range(48)))
    return InstalledModel(root=root, model_id='fixture', revision='fixture',
                          layer_count=1, expert_count=2, selected_expert_count=1,
                          expert_blob_size=24, common_tensors=(), expert_regions=regions)


class PrefillSlotReleaseTests(unittest.TestCase):
    def test_releases_real_buffers_and_refills_without_reloading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            for staged in (False, True):
                for policy in ('lru', 'lfu'):
                    with self.subTest(staged=staged, policy=policy), ExpertCache(
                        model, slots=2, read_workers=1, staged_expert_streaming=staged,
                        eviction_policy=policy,
                    ) as cache:
                        resident = cache.get_many(0, [0, 1])
                        expected = resident.individual_weights[0].w1.tolist()
                        del resident
                        before = mx.get_active_memory()
                        old_pool = weakref.ref(cache._pool)
                        pool_type = type(cache._pool)
                        descriptors = list(cache._descriptors)
                        metrics = cache.metrics_snapshot()
                        cache.release_prefill_slots()
                        self.assertIsNone(old_pool())
                        self.assertGreaterEqual(before - mx.get_active_memory(), 48)
                        self.assertEqual(cache.resident_count, 0)
                        self.assertEqual(len(cache._free_slots), 2)
                        self.assertEqual(cache._layer_counts, [0])
                        self.assertEqual(cache._heap, [])
                        self.assertEqual(cache._descriptors, descriptors)
                        self.assertEqual(cache.metrics_snapshot(), metrics)
                        self.assertIs(type(cache._pool), pool_type)
                        cache.release_prefill_slots()  # Already empty is safe.
                        resident = cache.get_many(0, [0, 1])
                        self.assertEqual(resident.individual_weights[0].w1.tolist(), expected)
                        self.assertEqual(cache.metrics.misses, metrics.misses + 2)
                        cache.get_many(0, [0])
                        self.assertEqual(cache.metrics.hits, metrics.hits + 1)
                        del resident

    def test_rejects_release_while_experts_are_in_use(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                cache.get_many(0, [0])
                with cache.pin_layer(0):
                    with self.assertRaisesRegex(RuntimeError, 'in use'):
                        cache.release_prefill_slots()
                with cache.batched_layer(0):
                    with self.assertRaisesRegex(RuntimeError, 'in use'):
                        cache.release_prefill_slots()
                self.assertEqual(cache.resident_count, 1)

    def test_qwen_pool_releases_and_refills(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'experts').mkdir()
            (root / 'experts/layer_00.bin').write_bytes(bytes(2 * 2_611_200))
            model = InstalledModel(root=root, model_id='fixture', revision='fixture',
                layer_count=1, expert_count=2, selected_expert_count=1,
                expert_blob_size=2_611_200, common_tensors=(),
                expert_regions=tuple(Tensor(*region) for region in QWEN_EXPERT_REGIONS),
                model_kind='qwen3.8-flash-next')
            for options in ({},):
                with self.subTest(options=options), ExpertCache(model, slots=4, **options) as cache:
                    cache.get_many(0, [0, 1])
                    old_pool = weakref.ref(cache._pool)
                    pool_type = type(cache._pool)
                    before = mx.get_active_memory()
                    cache.release_prefill_slots()
                    self.assertIsNone(old_pool())
                    self.assertIs(type(cache._pool), pool_type)
                    self.assertGreaterEqual(before - mx.get_active_memory(), 2 * model.expert_blob_size)
                    if hasattr(cache._pool, 'arenas'):
                        self.assertEqual(cache._pool.arenas, {})
                        self.assertEqual(cache._pool.grouped, {})
                    resident = cache.get_many(0, [0, 1])
                    self.assertEqual(resident.individual_weights[0].gate_up[0, :4].tolist(), [0] * 4)
                    self.assertEqual(cache.resident_count, 2)
                    del resident
                    import numpy as np
                    with cache.reuse_layer_buffers(), cache.batched_layer(0) as batch:
                        for value in (batch.gate_up, batch.gate_up_scales, batch.down, batch.down_scales):
                            self.assertTrue(np.asarray(value).flags.c_contiguous)
                    self.assertIsNone(cache._layer_buffers)

    def test_waits_for_orphaned_prefetch_before_discarding_its_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                entered, finish, done = threading.Event(), threading.Event(), threading.Event()
                original = cache._read_expert_ids
                def delayed(*args):
                    entered.set()
                    if not finish.wait(3):
                        raise TimeoutError('test read did not resume')
                    result = original(*args)
                    done.set()
                    return result
                with patch.object(cache, '_read_expert_ids', side_effect=delayed):
                    cache.prefetch_layer(0)
                    self.assertTrue(entered.wait(3))
                    # Mimic the queued next layer left by an aborted prefill.
                    timer = threading.Timer(0.03, finish.set)
                    timer.start()
                    try:
                        cache.release_prefill_slots()
                    finally:
                        finish.set()
                        timer.join()
                self.assertTrue(done.is_set())
                self.assertEqual(cache._prefetched_layers, {})
                with cache.batched_layer(0) as batched:
                    self.assertEqual(batched.w1_scales[:, 0, 0].tolist(), [4, 28])

    def test_release_happens_before_embeddings_only_for_batched_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                for batched in (False, True):
                    cache.get_many(0, [0, 1])
                    def embed(_):
                        self.assertEqual(cache.resident_count, 0 if batched else 2)
                        raise ValueError('stop after checking prefill entry')
                    model = SimpleNamespace(model=SimpleNamespace(
                        pipeline_layers=[object()], embed_tokens=embed))
                    with self.assertRaisesRegex(ValueError, 'stop after checking'):
                        layer_major_prefill(model, [1], [None], 128, cache,
                                            batched_experts=batched)
                    self.assertIsNone(cache._layer_buffers)
                    self.assertFalse(cache._prefetched_layers)
                cache.get_many(0, [0])
                layer_major_prefill(model, [], [None], 128, cache)
                self.assertEqual(cache.resident_count, 1)

    def test_qwen_releases_before_embeddings_and_preserves_prompt_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                cache.get_many(0, [0, 1])
                state = object()
                prompt_cache = [state]
                def embed(_):
                    self.assertEqual(cache.resident_count, 0)
                    self.assertIs(prompt_cache[0], state)
                    raise ValueError('checked Qwen prefill entry')
                model = SimpleNamespace(model=SimpleNamespace(layers=[object()], embed_tokens=embed))
                with self.assertRaisesRegex(ValueError, 'checked Qwen'):
                    _qwen_layer_major_prefill(model, [1], prompt_cache, 128, cache)
                self.assertIsNone(cache._layer_buffers)
                self.assertFalse(cache._prefetched_layers)


class LayerBufferPoolTests(unittest.TestCase):
    def test_slot_major_arenas_keep_their_layout_and_dynamic_expert_count(self):
        import copy
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            with ExpertCache(model, slots=2, read_workers=1) as cache:
                for count in (1, 2):
                    proxy = copy.copy(cache._pool)
                    proxy._model = replace(model, expert_count=count)
                    arena = mx.zeros((count * model.expert_blob_size // 4,), dtype=mx.uint32)
                    mx.eval(arena)
                    buffer = memoryview(arena).cast('B')
                    for expert in range(count):
                        for region, view in zip(model.expert_regions, proxy.write_views(buffer, expert)):
                            start = expert * model.expert_blob_size + region.offset
                            view[:] = bytes(range(start, start + region.length))
                    weights = proxy.batched(arena)
                    self.assertEqual(weights.w1_scales[:, 0, 0].tolist(), [4, 28][:count])
                    self.assertEqual(weights.w13_scales[:, :, 0].tolist(), [[20, 4], [44, 28]][:count])

    def test_layer_buffers_are_reused_between_layers_and_released_after_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                with cache.reuse_layer_buffers():
                    with cache.batched_layer(0) as first:
                        mx.eval(first.w1_scales)
                        first_buffer = cache._active_prefetch_trace.job.packed
                        self.assertEqual(cache._layer_buffers, [])
                    self.assertEqual(len(cache._layer_buffers), 1)
                    self.assertIs(cache._layer_buffers[0], first_buffer)
                    with cache.batched_layer(0) as second:
                        self.assertIs(cache._active_prefetch_trace.job.packed, first_buffer)
                        self.assertEqual(second.w1_scales[:, 0, 0].tolist(), [4, 28])
                    cache.prefetch_layer(0)
                    self.assertEqual(cache._layer_buffers, [])
                # Request cleanup must drain pending prefetches before releasing storage.
                self.assertIsNone(cache._layer_buffers)
                self.assertFalse(cache._prefetched_layers)
                cache.discard_prefetched_layers()  # Generation's later cleanup is harmless.
                self.assertIsNone(cache._layer_buffers)
                with cache.batched_layer(0) as third:
                    self.assertEqual(third.w1_scales[:, 0, 0].tolist(), [4, 28])

    def test_cancelled_prefill_does_not_repopulate_pool_during_generation_cleanup(self):
        from deepseek_v4_ssd.cancellation import GenerationCancelled
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                with self.assertRaises(GenerationCancelled):
                    with cache.reuse_layer_buffers():
                        cache.prefetch_layer(0)
                        raise GenerationCancelled()
                self.assertFalse(cache._prefetched_layers)
                self.assertIsNone(cache._layer_buffers)
                cache.discard_prefetched_layers()
                self.assertIsNone(cache._layer_buffers)
                with cache.reuse_layer_buffers(), cache.batched_layer(0) as batch:
                    self.assertEqual(batch.w1_scales[:, 0, 0].tolist(), [4, 28])

    def test_batched_regions_are_contiguous_per_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1) as cache:
                with cache.batched_layer(0) as batched:
                    packed = cache._active_prefetch_trace.job.packed
                    mx.eval(batched.w13, batched.w2, batched.w13_scales, batched.w2_scales)
                    # Expert-major copies of each batched region, in slot order:
                    # w13 = w3 then w1 for expert 0, then expert 1; then w2; scales alike.
                    raw = memoryview(packed).cast('B').tobytes()
                    self.assertEqual(list(raw), [16, 17, 18, 19, 0, 1, 2, 3, 40, 41, 42, 43, 24, 25, 26, 27,
                                                 8, 9, 10, 11, 32, 33, 34, 35,
                                                 20, 21, 22, 23, 4, 5, 6, 7, 44, 45, 46, 47, 28, 29, 30, 31,
                                                 12, 13, 14, 15, 36, 37, 38, 39])
                    self.assertEqual(batched.w13.shape, (2, 2, 1))
                    self.assertEqual(batched.w13_scales[:, :, 0].tolist(), [[20, 4], [44, 28]])
                    self.assertEqual(batched.w2_scales[:, 0, 0].tolist(), [12, 36])
