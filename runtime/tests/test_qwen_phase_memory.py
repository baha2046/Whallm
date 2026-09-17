import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd.cancellation import GenerationCancelled
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.generation import _route_phase
from runtime.tests.test_prefill_slot_release import fixture


def model_fixture(root):
    model = fixture(root)
    for layer in range(1, 4):
        (root / f'experts/layer_{layer:02d}.bin').write_bytes(bytes(range(48)))
    return replace(model, layer_count=4)


class QwenPhaseMemoryTests(unittest.TestCase):
    def test_shrink_releases_buffers_and_grow_preserves_residents(self):
        for policy in ('lru', 'lfu', 'route'):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                with ExpertCache(model_fixture(Path(directory)), slots=8, read_workers=1,
                                 eviction_policy=policy) as cache:
                    for layer in range(4):
                        cache.get_many(layer, [0, 1])
                    descriptors = list(cache._descriptors)
                    first = cache._pool._slots[0]
                    expected = cache.get_many(0, [0]).individual_weights[0].w1.tolist()
                    before = mx.get_active_memory()
                    cache.enable_phase_memory()
                    with cache.phase_memory_request():
                        with _route_phase(cache, 'prefill'):
                            self.assertEqual(cache.slots, 4)
                            self.assertEqual(cache.resident_count, 4)
                            self.assertGreaterEqual(before - mx.get_active_memory(), 4 * 24)
                            self.assertIs(cache._pool._slots[0], first)
                            self.assertEqual(cache._layer_counts, [2, 2, 0, 0])
                        with _route_phase(cache, 'prefill'):
                            self.assertEqual(cache._phase_resize_count, 1)
                        with _route_phase(cache, 'decode'):
                            self.assertEqual(cache.slots, 8)
                            self.assertIs(cache._pool._slots[0], first)
                            self.assertEqual(cache.resident_count, 4)
                            self.assertEqual(len(cache._free_slots), 4)
                            self.assertEqual(cache.get_many(0, [0]).individual_weights[0].w1.tolist(), expected)
                            for layer in range(2, 4):
                                cache.get_many(layer, [0, 1])
                    self.assertEqual(cache.resident_count, 8)
                    self.assertEqual(cache._phase_resize_count, 2)
                    self.assertEqual(cache._descriptors, descriptors)
                    self.assertEqual(sum(cache._layer_counts), cache.resident_count)
                    if policy == 'route':
                        self.assertEqual(cache._route_policy.capacity, 8)

    def test_cancel_exception_and_next_request_restore_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory)), slots=8, read_workers=1) as cache:
                cache.enable_phase_memory()
                for error in (GenerationCancelled, ValueError):
                    with self.assertRaises(error), cache.phase_memory_request():
                        with _route_phase(cache, 'prefill'):
                            cache.get_many(0, [0, 1])
                            raise error('injected')
                    self.assertEqual(cache.slots, 8)
                    self.assertFalse(cache._phase_memory_active)
                with cache.phase_memory_request():
                    with _route_phase(cache, 'prefill'):
                        self.assertEqual(cache.slots, 4)
                self.assertEqual(cache.slots, 8)

    def test_noop_when_disabled_or_budget_cannot_spare_a_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            model = model_fixture(Path(directory))
            for slots, enabled in ((8, False), (2, True), (1, True)):
                with ExpertCache(model, slots=slots, read_workers=1) as cache:
                    if enabled:
                        cache.enable_phase_memory()
                    with cache.phase_memory_request(), _route_phase(cache, 'prefill'):
                        self.assertEqual(cache.slots, slots)
                    self.assertEqual(cache._phase_resize_count, 0)

    def test_cannot_resize_pinned_or_batched_experts_or_exceed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory)), slots=8, read_workers=1) as cache:
                cache.enable_phase_memory()
                for owner in (cache.pin_layer(0), cache.batched_layer(0)):
                    with owner, self.assertRaisesRegex(RuntimeError, 'in use'):
                        cache._resize_phase_slots(4)
                with self.assertRaises(ValueError):
                    cache._resize_phase_slots(9)
                self.assertEqual(cache.slots, 8)
                with cache.phase_memory_request():
                    with self.assertRaisesRegex(RuntimeError, 'already active'):
                        with cache.phase_memory_request():
                            pass

    def test_closing_ready_iterator_on_a_cache_hit_drains_missing_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory)), slots=8, read_workers=1) as cache:
                cache.enable_phase_memory()
                cache.get_many(0, [0])
                entered, finish, done = threading.Event(), threading.Event(), threading.Event()
                original = cache._read_expert_into_slot
                def delayed(*args):
                    entered.set()
                    if not finish.wait(3):
                        raise TimeoutError('read did not resume')
                    result = original(*args)
                    done.set()
                    return result
                with patch.object(cache, '_read_expert_into_slot', side_effect=delayed):
                    ready = cache.iter_ready(0, [0, 1])
                    self.assertEqual(next(ready)[0], 0)
                    self.assertTrue(entered.wait(3))
                    timer = threading.Timer(0.03, finish.set)
                    timer.start()
                    try:
                        ready.close()
                    finally:
                        finish.set()
                        timer.join()
                self.assertTrue(done.is_set())
                self.assertNotIn((0, 1), cache._entries)
                with cache.phase_memory_request(), _route_phase(cache, 'prefill'):
                    self.assertEqual(cache.slots, 4)
                    cache.get_many(0, [0, 1])
                self.assertEqual(cache.slots, 8)

    def test_running_prefetch_is_drained_before_gpu_sync_and_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(model_fixture(Path(directory)), slots=8, read_workers=1) as cache:
                cache.enable_phase_memory()
                entered, finish, done = threading.Event(), threading.Event(), threading.Event()
                original_read, original_sync = cache._read_expert_ids, mx.synchronize
                def delayed(*args):
                    entered.set()
                    if not finish.wait(3):
                        raise TimeoutError('read did not resume')
                    result = original_read(*args)
                    done.set()
                    return result
                def synchronize(*args, **kwargs):
                    self.assertTrue(done.is_set())
                    self.assertEqual(cache.slots, 8)
                    return original_sync(*args, **kwargs)
                with patch.object(cache, '_read_expert_ids', side_effect=delayed):
                    cache.prefetch_layer(0)
                    self.assertTrue(entered.wait(3))
                    timer = threading.Timer(0.03, finish.set)
                    timer.start()
                    try:
                        with patch('deepseek_v4_ssd.expert_cache.mx.synchronize', side_effect=synchronize):
                            cache._resize_phase_slots(4)
                    finally:
                        finish.set()
                        timer.join()
                self.assertEqual(cache._prefetched_layers, {})
                self.assertEqual(cache.slots, 4)
                self.assertEqual(cache.get_many(0, [0]).individual_weights[0].w1.tolist(), [[0x03020100]])


if __name__ == '__main__':
    unittest.main()
