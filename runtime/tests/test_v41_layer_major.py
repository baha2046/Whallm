from contextlib import nullcontext
from types import SimpleNamespace
import unittest
import mlx.core as mx
from runtime.tests.test_v41_prompt_cache import tiny_model
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_support.deepseek_v41 import (
    _deepseek_v41_prefill, _deepseek_v41_layer_major_prefill)


class V41LayerMajorTests(unittest.TestCase):
    def test_shared_attention_partial_groups_engram_and_continuation(self):
        model = tiny_model()
        for prefix in ([], [3, 4, 5]):
            for step in (2, 3, 5):
                for batched in (False, True):
                    reference, candidate = model.make_cache(), model.make_cache()
                    if prefix:
                        _deepseek_v41_prefill(model, prefix, reference, 2)
                        _deepseek_v41_prefill(model, prefix, candidate, 2)
                    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    events = []
                    experts = SimpleNamespace(
                        release_prefill_slots=lambda: events.append('release'),
                        batched_layer=lambda i: (events.append(('layer', i)) or nullcontext()),
                        prefetch_layer=lambda i: events.append(('prefetch', i)))
                    config = RuntimeConfig(batched_expert_prefill=batched, v41_next_layer_prefetch=True)
                    _deepseek_v41_prefill(model, tokens, reference, step)
                    _deepseek_v41_layer_major_prefill(model, tokens, candidate, step, experts, config)
                    for token in (10, 11, 12):
                        expected = model(mx.array([[token]]), reference)
                        actual = model(mx.array([[token]]), candidate)
                        mx.eval(expected, actual)
                        self.assertTrue(mx.array_equal(actual, expected).item(), (prefix, step, batched))
                    self.assertEqual(candidate[0].offset, reference[0].offset)
                    for a, b in zip(candidate[0].state, reference[0].state):
                        self.assertTrue(mx.array_equal(a, b).item())
                    self.assertEqual(events.count('release'), int(batched))
                    self.assertEqual(events.count(('prefetch', 1)), int(batched))

    def test_ced_rebuilds_all_windows_and_skips_decoder_prefix(self):
        from dataclasses import replace
        from deepseek_v4_ssd.deepseek_v41.model import Model
        from deepseek_v4_ssd.deepseek_v41_ssd import DeepSeekV41ForCausalLM
        base = tiny_model()
        args = replace(base.args, n_layers=7, compress_ratios=(2, 2, 1, 1, 1, 1, 1), max_seq_len=64)
        model = DeepSeekV41ForCausalLM(Model(args, token_map=list(range(16))))
        tokens = [i % 16 for i in range(39)]
        for step in (2, 3, 5):
            for packed in (False, True):
                model.packed_kv = model.packed_index = packed
                reference, candidate = model.make_cache(), model.make_cache()
                experts = SimpleNamespace()
                control = RuntimeConfig(batched_expert_prefill=False)
                ced = RuntimeConfig(batched_expert_prefill=False, v41_ced_prefill=True)
                _deepseek_v41_layer_major_prefill(model, tokens, reference, step, experts, control)
                _deepseek_v41_layer_major_prefill(model, tokens, candidate, step, experts, ced)
                self.assertGreater(model.ced_skipped_layer_tokens, 0)
                for a, b in zip(candidate[0].state, reference[0].state):
                    self.assertTrue(mx.array_equal(a, b).item(), (step, packed))
                for token in (4, 5, 6):
                    expected = model(mx.array([[token]]), reference)
                    actual = model(mx.array([[token]]), candidate)
                    self.assertTrue(mx.array_equal(actual, expected).item(), (step, packed))
