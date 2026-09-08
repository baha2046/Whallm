import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from threading import Event
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd import qwen_block_generation as block
from deepseek_v4_ssd.cancellation import cancellation_scope, GenerationCancelled
from deepseek_v4_ssd.model import RuntimeConfig


class Cache:
    def __init__(self):
        self.keys = mx.array([], mx.int32)

    @property
    def state(self):
        return [self.keys]


def model(inputs, cache):
    cache[0].keys = mx.concatenate([cache[0].keys, inputs.reshape(-1)])
    return mx.where(mx.arange(7)[None, None, :] == ((inputs+1) % 7)[..., None], 0., -10.)


def verify(model, inputs, cache):
    return mx.concatenate([model(inputs[:, t:t+1], cache=cache) for t in range(inputs.shape[1])], axis=1)


class BlockGenerationTests(unittest.TestCase):
    def run_case(self, draft, maximum, eos=(), sampler=None):
        cache = [Cache()]
        stats = dict(rounds=0, proposed_tokens=0, accepted_tokens=0, memory_fallbacks=0)
        kwargs = dict(history=[0], prompt_cache=cache, max_tokens=maximum,
                      sampler=sampler or (lambda x: mx.argmax(x, axis=-1)),
                      logits_processors=[], prefill_step_size=4,
                      prompt_progress_callback=lambda *x: None, eos_token_ids=set(eos), stats=stats)
        with patch.object(block, 'verify_union', verify), patch.object(
            block, 'propose', side_effect=lambda history, maximum: draft(history)[:maximum]):
            output = [t for t, _ in block.block_tokens(model, [0], **kwargs)]
        return output, cache[0].keys.tolist(), stats

    def test_accept_reject_eos_and_length_commit_only_consumed_tokens(self):
        for draft in [lambda h: [(h[-1]+i) % 7 for i in (1,2,3)],
                      lambda h: [6,6,6], lambda h: [(h[-1]+1) % 7,6,6]]:
            for maximum in range(1, 10):
                for eos in [(), (1,), (2,), (3,), (4,)]:
                    output, state, stats = self.run_case(draft, maximum, eos)
                    expected = [(i+1) % 7 for i in range(maximum)]
                    if eos and eos[0] in expected:
                        expected = expected[:expected.index(eos[0])+1]
                    self.assertEqual(output, expected)
                    self.assertEqual(state, [0]+(expected[:-1] if expected[-1] in eos else expected))

    def test_sampling_uses_one_target_draw_per_emitted_token(self):
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=4.)
        mx.random.seed(814)
        reference = self.run_case(lambda _: [], 20, sampler=sampler)
        mx.random.seed(814)
        candidate = self.run_case(lambda h: [(h[-1]+i) % 7 for i in (1,2,3)], 20, sampler=sampler)
        self.assertEqual(reference[:2], candidate[:2])
        self.assertGreater(candidate[2]['rounds'], 0)

    def test_memory_guard_and_cancellation(self):
        with patch.object(block, 'MAX_FORK_BYTES', -1):
            _, _, stats = self.run_case(lambda _: [1,2,3], 8)
            self.assertGreater(stats['memory_fallbacks'], 0)
            self.assertEqual(stats['rounds'], 0)
        event = Event()
        event.set()
        with cancellation_scope(event), self.assertRaises(GenerationCancelled):
            self.run_case(lambda _: [1,2,3], 8)

    def test_catalog_compatibility_and_types(self):
        from deepseek_v4_ssd.model_manager import _parse_runtime, ModelCatalogError
        data = asdict(RuntimeConfig())
        data.pop('qwen_short_block')
        self.assertFalse(_parse_runtime(data, 'runtime', 'qwen3.8-flash-next').qwen_short_block)
        for value in (True, False):
            self.assertEqual(_parse_runtime({**data, 'qwen_short_block': value}, 'runtime',
                                           'qwen3.8-flash-next').qwen_short_block, value)
        with self.assertRaises(ModelCatalogError):
            _parse_runtime({**data, 'qwen_short_block': 'true'}, 'runtime', 'qwen3.8-flash-next')

    def test_cancel_during_verification_does_not_commit_private_state(self):
        cache = [Cache()]
        event = Event()
        def cancel(model, inputs, private):
            result = verify(model, inputs, private)
            event.set()
            return result
        with cancellation_scope(event), patch.object(block, 'propose', return_value=[1,2,3]), \
             patch.object(block, 'verify_union', cancel), self.assertRaises(GenerationCancelled):
            list(block.block_tokens(model, [0], history=[0], prompt_cache=cache,
                max_tokens=8, sampler=lambda x: mx.argmax(x, axis=-1), logits_processors=[],
                prefill_step_size=4, prompt_progress_callback=lambda *x: None,
                eos_token_ids=set(), stats=dict(rounds=0, proposed_tokens=0, accepted_tokens=0)))
        self.assertEqual(cache[0].keys.tolist(), [])

    def test_verifier_failure_keeps_original_state(self):
        cache = [Cache()]
        def fail(model, inputs, private):
            verify(model, inputs, private)
            raise OSError('injected read failure')
        with patch.object(block, 'propose', return_value=[1,2,3]), \
             patch.object(block, 'verify_union', fail), self.assertRaises(OSError):
            list(block.block_tokens(model, [0], history=[0], prompt_cache=cache,
                max_tokens=8, sampler=lambda x: mx.argmax(x, axis=-1), logits_processors=[],
                prefill_step_size=4, prompt_progress_callback=lambda *x: None,
                eos_token_ids=set(), stats=dict(rounds=0, proposed_tokens=0, accepted_tokens=0)))
        self.assertEqual(cache[0].keys.tolist(), [])

    def test_large_arena_count_splits_kernel_arguments_without_repacking(self):
        from dataclasses import replace
        from runtime.tests.test_qwen_resident_block import fixture
        from deepseek_v4_ssd.expert_cache import ExpertCache
        from deepseek_v4_ssd.qwen4_exp import StreamingExperts
        from deepseek_v4_ssd.qwen_resident_block import BoundedArenaPool, resident_experts
        class SingleSlotPages(BoundedArenaPool):
            PAGE_SLOTS = 1
        with tempfile.TemporaryDirectory() as directory:
            installed = fixture(Path(directory))
            file = installed.root/'experts/layer_00.bin'
            file.write_bytes(file.read_bytes()*2)
            installed = replace(installed, expert_count=24)
            with ExpertCache(installed, slots=40, qwen_short_block=True) as candidate, \
                 ExpertCache(installed, slots=40) as reference:
                candidate._pool = SingleSlotPages(installed, 40)
                values = mx.ones((1,4,2560), mx.bfloat16)*.125
                routes = mx.arange(24).reshape(1,4,6)
                expert = StreamingExperts(0, reference)
                expected = mx.concatenate([expert(values[:,t:t+1], routes[:,t:t+1]) for t in range(4)], axis=1)
                mx.eval(expected)
                actual = resident_experts(values, routes, candidate, 0)
                self.assertEqual(len(candidate._pool.arenas), 24)
                self.assertTrue(mx.array_equal(actual, expected).item())

    def test_production_pool_and_native_verifier_preserve_frequency(self):
        from runtime.tests.test_qwen_resident_block import fixture
        from deepseek_v4_ssd.expert_cache import ExpertCache
        from deepseek_v4_ssd.qwen4_exp import StreamingExperts
        from deepseek_v4_ssd.qwen_resident_block import resident_experts
        with tempfile.TemporaryDirectory() as directory:
            installed = fixture(Path(directory))
            with ExpertCache(installed, slots=4096, qwen_short_block=True) as candidate, \
                 ExpertCache(installed, slots=4096) as reference:
                self.assertTrue(candidate.qwen_short_block_active)
                values = mx.ones((1,4,2560), mx.bfloat16)*0.2
                routes = mx.array([[[0,1,2,3], [3,1,4,5], [5,6,0,1], [7,0,1,8]]])
                expert = StreamingExperts(0, reference)
                expected = mx.concatenate([expert(values[:,t:t+1], routes[:,t:t+1]) for t in range(4)], axis=1)
                mx.eval(expected)
                actual = resident_experts(values, routes, candidate, 0)
                self.assertTrue(mx.array_equal(actual, expected).item())
                self.assertEqual(candidate._entries[(0,1)].frequency, 4)
                self.assertEqual(candidate.metrics.bytes_read, 9*installed.expert_blob_size)


if __name__ == '__main__':
    unittest.main()
