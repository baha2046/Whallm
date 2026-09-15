from __future__ import annotations

import threading
import tempfile
import unittest
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd import cancellation
from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope, wait_for_futures
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.model import layer_major_prefill
from runtime.tests.test_prefill_slot_release import fixture


class PrefillCancellationTests(unittest.TestCase):
    def test_cancelled_reads_drain_running_writes_and_cancel_queued_work(self):
        cancelled, released, waiting = threading.Event(), threading.Event(), threading.Event()
        running, queued = Future(), Future()
        running.set_running_or_notify_cancel()
        queued.add_done_callback(lambda _: released.set())

        def finish_write():
            if not waiting.wait(3):
                running.set_exception(TimeoutError('wait never started'))
                return
            cancelled.set()
            if released.wait(3):
                running.set_result('finished write')
            else:
                running.set_exception(TimeoutError('queued read was not cancelled'))

        worker = threading.Thread(target=finish_write)
        worker.start()
        original_wait = cancellation.wait
        def observe_wait(*args, **kwargs):
            waiting.set()
            return original_wait(*args, **kwargs)
        try:
            with patch.object(cancellation, 'wait', side_effect=observe_wait), \
                 cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
                wait_for_futures([running, queued])
            self.assertTrue(queued.cancelled())
            self.assertEqual(running.result(), 'finished write')
        finally:
            worker.join(3)
        # A worker's own error must not be mistaken for a polling timeout.
        failed = Future()
        failed.set_exception(TimeoutError('read failed'))
        with self.assertRaisesRegex(TimeoutError, 'read failed'):
            wait_for_futures([failed])

    def test_expert_reads_cancel_without_leaking_slots_and_allow_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1, separate_prefill_io=True) as cache:
                cancelled = threading.Event()
                original = cache._read_expert_into_slot
                def read(*args):
                    result = original(*args)
                    cancelled.set()
                    return result
                with patch.object(cache, '_read_expert_into_slot', side_effect=read):
                    with cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
                        cache.get_many(0, [0, 1])
                self.assertEqual(cache.resident_count, 0)
                self.assertEqual(len(cache._free_slots), 2)
                self.assertEqual(len(cache.get_many(0, [0, 1]).individual_weights), 2)

    def test_cancelled_batched_read_never_exposes_partial_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with ExpertCache(fixture(Path(directory)), slots=2, read_workers=1, separate_prefill_io=True) as cache:
                cancelled = threading.Event()
                original = cache._read_expert_ids
                def read(*args):
                    result = original(*args)
                    cancelled.set()
                    return result
                with patch.object(cache, '_read_expert_ids', side_effect=read):
                    with cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
                        with cache.batched_layer(0):
                            self.fail('cancelled read exposed weights')
                self.assertIsNone(cache._batched_layer)
                self.assertEqual(cache._prefetched_layers, {})
                with cache.batched_layer(0) as weights:
                    self.assertIsNotNone(weights)

    def test_v41_stops_before_next_layer_and_fresh_cache_recovers(self):
        from runtime.tests.test_v41_prompt_cache import tiny_model
        model = tiny_model()
        cancelled = threading.Event()
        first = model.model.layers[0]
        def layer(*args):
            result = first(*args)
            cancelled.set()
            return result
        layer.engram = first.engram
        with patch.object(model.model, 'layers', [layer, *model.model.layers[1:]]), \
             patch.object(model.model.layers[1], 'attn', side_effect=AssertionError('ran next layer')):
            with cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
                model(mx.array([[1, 2]]), cache=model.make_cache())
        mx.synchronize()
        output = model(mx.array([[1, 2]]), cache=model.make_cache())
        mx.eval(output)
        self.assertEqual(output.shape[-1], 16)

    def test_v4_stops_at_chunk_boundary_and_can_run_again(self):
        for phase in ('attention', 'moe', 'fallback'):
            for step in (2, 4):
                with self.subTest(phase=phase, step=step):
                    cancelled = threading.Event()
                    calls, pinned = [], []

                    def compute(name, value):
                        calls.append(name)
                        if name == phase:
                            cancelled.set()
                        return value + 1

                    @contextmanager
                    def pin(layer):
                        pinned.append(layer)
                        try:
                            yield
                        finally:
                            pinned.remove(layer)

                    class Fallback:
                        def __call__(self, value, *_):
                            return compute('fallback', value)

                    layer = Fallback() if phase == 'fallback' else SimpleNamespace(
                        attn_hc=lambda value: (value, None, None),
                        attn_norm=lambda value: value,
                        attn=lambda value, **_: compute('attention', value),
                        ffn_hc=lambda value: (value, None, None),
                        ffn_norm=lambda value: value,
                        ffn=lambda value, _: compute('moe', value),
                    )
                    model = SimpleNamespace(model=SimpleNamespace(
                        embed_tokens=lambda inputs: inputs[..., None].astype(mx.float32),
                        pipeline_layers=[layer, layer],
                        args=SimpleNamespace(hc_mult=1, sliding_window=4),
                    ))
                    def run():
                        layer_major_prefill(model, [1, 2, 3, 4],
                                            [SimpleNamespace(offset=0) for _ in range(2)],
                                            step, SimpleNamespace(pin_layer=pin),
                                            moe_step_size=step, batched_experts=False)

                    with patch('deepseek_v4_ssd.model.deepseek_v4.create_attention_mask', return_value=None), \
                         patch('deepseek_v4_ssd.model.deepseek_v4.hc_expand', side_effect=lambda value, *_: value):
                        with cancellation_scope(cancelled), self.assertRaises(GenerationCancelled):
                            run()
                        self.assertEqual(calls.count(phase), 1)
                        self.assertEqual(pinned, [])
                        # A cancelled request must not poison the next request's scope.
                        calls.clear()
                        run()
                        self.assertGreater(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
