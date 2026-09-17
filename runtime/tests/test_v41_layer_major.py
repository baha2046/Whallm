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
                    config = RuntimeConfig(batched_expert_prefill=batched, v41_next_layer_prefetch=True,
                                           moe_prefill_step_size=step)
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
                control = RuntimeConfig(batched_expert_prefill=False, moe_prefill_step_size=step)
                ced = RuntimeConfig(batched_expert_prefill=False, v41_ced_prefill=True,
                                    moe_prefill_step_size=step)
                _deepseek_v41_layer_major_prefill(model, tokens, reference, step, experts, control)
                _deepseek_v41_layer_major_prefill(model, tokens, candidate, step, experts, ced)
                self.assertGreater(model.ced_skipped_layer_tokens, 0)
                for a, b in zip(candidate[0].state, reference[0].state):
                    self.assertTrue(mx.array_equal(a, b).item(), (step, packed))
                for token in (4, 5, 6):
                    expected = model(mx.array([[token]]), reference)
                    actual = model(mx.array([[token]]), candidate)
                    self.assertTrue(mx.array_equal(actual, expected).item(), (step, packed))

    def test_moe_batches_and_dspark_seed_match_token_major_prefill(self):
        from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
        from deepseek_v4_ssd.model import forward_with_hidden
        model = tiny_model()
        draft = DSpark(model.args, block_size=3, noise_token_id=15,
                       target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        model.dspark = draft
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        prefilled = model.make_cache()
        _, hidden = forward_with_hidden(model, mx.array([tokens]), prefilled, draft.target_layers)
        draft.prefill_context(hidden, 0)
        expected_state = draft.cache_state()
        def close(actual, expected, tolerance):
            actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
            finite = mx.isfinite(expected)
            return mx.allclose(mx.where(finite, actual, 0), mx.where(finite, expected, 0),
                               atol=tolerance, rtol=tolerance).item()

        # A whole-layer MoE batch runs different matmul kernels than the
        # reference's small chunks, so only per-chunk batches are bit-exact. The
        # fake-quantized caches can then land one FP8/FP4 code apart, so the
        # inexact cases are checked through the continuation logits and a
        # coarse draft-context tolerance instead of exact cache equality.
        from deepseek_v4_ssd.model_support import get_support
        support = get_support("deepseek-v4.1")
        for step, moe_step, exact in ((3, 3, True), (4, 4, True), (3, 0, False), (5, 2, False)):
            with self.subTest(step=step, moe_step=moe_step):
                reference = support.clone_cache(prefilled)
                candidate = model.make_cache()
                experts = SimpleNamespace(release_prefill_slots=lambda: None,
                                          batched_layer=lambda i: nullcontext(),
                                          prefetch_layer=lambda i: None,
                                          release_layer_buffers=lambda: None)
                config = RuntimeConfig(moe_prefill_step_size=moe_step)
                draft.reset_cache()
                _deepseek_v41_layer_major_prefill(model, tokens, candidate, step, experts, config)
                self.assertEqual(candidate[0].offset, reference[0].offset)
                if exact:
                    for a, b in zip(candidate[0].state, reference[0].state):
                        self.assertTrue(mx.array_equal(a, b).item())
                actual_state = draft.cache_state()
                self.assertEqual([offset for _, offset in actual_state], [offset for _, offset in expected_state])
                for (actual, _), (expected, _) in zip(actual_state, expected_state):
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertTrue(close(actual, expected, 1e-4 if exact else 0.25))
                for token in (12, 13):
                    expected = model(mx.array([[token]]), reference)
                    actual = model(mx.array([[token]]), candidate)
                    self.assertTrue(mx.array_equal(actual, expected).item() if exact else close(actual, expected, 0.02))

    def test_dspark_request_prefills_layer_major_and_finishes_with_the_last_token(self):
        from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
        from deepseek_v4_ssd.dspark import generate_tokens
        from deepseek_v4_ssd.model_support import get_support
        model = tiny_model()
        draft = DSpark(model.args, block_size=3, noise_token_id=15,
                       target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        model.dspark = draft
        support = get_support("deepseek-v4.1")
        self.assertTrue(support.layer_major_prefill_seeds_dspark)
        config = RuntimeConfig(dspark_enabled=True, v41_next_layer_prefetch=True, v41_ced_prefill=True)
        support.validate_config(config)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        reference = model.make_cache()
        expected = [int(t) for t, _, _ in generate_tokens(prompt, model, draft, reference, max_tokens=6,
                    prefill_step_size=4, temperature=0, top_p=1, fallback_enabled=False)]
        candidate = model.make_cache()
        experts = SimpleNamespace(release_prefill_slots=lambda: None,
                                  batched_layer=lambda i: nullcontext(),
                                  prefetch_layer=lambda i: None,
                                  release_layer_buffers=lambda: None)
        draft.reset_cache()
        support.prefill(model, prompt[:-1], candidate, 4, experts, config)
        self.assertEqual(candidate[0].offset, len(prompt) - 1)
        actual = [int(t) for t, _, _ in generate_tokens(prompt, model, draft, candidate, max_tokens=6,
                  prefill_step_size=4, temperature=0, top_p=1, fallback_enabled=False,
                  prefilled_tokens=len(prompt) - 1)]
        self.assertEqual(actual, expected)
