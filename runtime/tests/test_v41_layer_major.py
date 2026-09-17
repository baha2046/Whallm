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
                        reuse_layer_buffers=nullcontext,
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

    def test_dspark_capture_matches_chunked_forward_and_continuation(self):
        from deepseek_v4_ssd.model import forward_with_hidden
        from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
        from deepseek_v4_ssd.dspark import generate_tokens
        model = tiny_model()
        targets = (1, 2, 3)
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        reference, candidate = model.make_cache(), model.make_cache()
        chunks = []
        for begin in range(0, len(prompt) - 1, 3):
            _, hidden = forward_with_hidden(model, mx.array([prompt[begin:min(begin + 3, len(prompt)-1)]]),
                                            reference, targets)
            chunks.append(hidden)
        expected = mx.concatenate(chunks, axis=1)
        actual = _deepseek_v41_layer_major_prefill(
            model, prompt[:-1], candidate, 3, SimpleNamespace(),
            RuntimeConfig(batched_expert_prefill=False), targets)
        self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
        draft = DSpark(model.args, block_size=3, noise_token_id=15,
                       target_layers=targets, markov_rank=8, expert_count=2, topk=1)
        kwargs = dict(max_tokens=8, prefill_step_size=3, temperature=0, top_p=1,
                      fallback_enabled=False)
        cold = list(generate_tokens(prompt, model, draft, model.make_cache(), **kwargs))
        draft.reset_cache()
        draft.prefill_context(actual, 0)
        warm = list(generate_tokens(prompt, model, draft, candidate,
                                    prefilled_tokens=len(prompt)-1, **kwargs))
        self.assertEqual([int(x[0]) for x in cold], [int(x[0]) for x in warm])

    def test_four_residual_streams_preserve_staggered_mixes(self):
        from dataclasses import replace
        from deepseek_v4_ssd.deepseek_v41.model import Model
        from deepseek_v4_ssd.deepseek_v41_ssd import DeepSeekV41ForCausalLM
        args = replace(tiny_model().args, hc_mult=4)
        core = Model(args, token_map=list(range(16)))
        for layer in core.layers:
            for side in ('attn', 'ffn'):
                name = f'hc_{side}_fn'
                setattr(layer, name, mx.random.normal(getattr(layer, name).shape) * 0.02)
                setattr(layer, f'hc_{side}_scale', mx.array([0.3, 0.2, 0.1]))
        model = DeepSeekV41ForCausalLM(core)
        reference, candidate = model.make_cache(), model.make_cache()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        _deepseek_v41_prefill(model, tokens, reference, 3)
        _deepseek_v41_layer_major_prefill(model, tokens, candidate, 3, SimpleNamespace(),
                                        RuntimeConfig(batched_expert_prefill=False))
        for token in (10, 11, 12):
            expected = model(mx.array([[token]]), reference)
            actual = model(mx.array([[token]]), candidate)
            self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())
            self.assertTrue(mx.array_equal(mx.argmax(actual, axis=-1), mx.argmax(expected, axis=-1)).item())
        for a, b in zip(candidate[0].state, reference[0].state):
            self.assertTrue(mx.allclose(a, b, atol=1e-5, rtol=1e-5).item())

    def test_configured_moe_step_size_does_not_change_attention_chunks(self):
        events = []
        class Layer:
            engram = None
            def forward_attention(self, h, mix, start, *_):
                events.append(('attention', h.shape[1]))
                return h, mix
            def forward_ffn(self, h, mix):
                events.append(('ffn', h.shape[1]))
                return h, mix
        state = SimpleNamespace(offset=0, ensure_capacity=lambda _: None)
        core = SimpleNamespace(layers=[Layer()], hc_mult=1, engram_hasher=None,
            args=SimpleNamespace(engram_layer_ids=(), kv_source_layers=()),
            embed=lambda ids: ids[..., None].astype(mx.float32))
        _deepseek_v41_layer_major_prefill(SimpleNamespace(model=core), list(range(7)),
            [SimpleNamespace(cache=state)], 2, SimpleNamespace(),
            RuntimeConfig(batched_expert_prefill=False, moe_prefill_step_size=3))
        self.assertEqual([n for phase, n in events if phase == 'attention'], [2, 2, 2, 1])
        self.assertEqual([n for phase, n in events if phase == 'ffn'], [3, 3, 1])
        self.assertEqual(state.offset, 7)

    def test_ffn_batch_cap_tail_and_prefetch_before_attention(self):
        events = []
        class Layer:
            engram = None
            def forward_attention(self, h, mix, start, *_):
                events.append(('attention', start, h.shape[1]))
                return h + 1, mix
            def forward_ffn(self, h, mix):
                events.append(('ffn', h.shape[1]))
                return h * 2, mix
        state = SimpleNamespace(offset=0, ensure_capacity=lambda _: None)
        core = SimpleNamespace(layers=[Layer(), Layer()], hc_mult=1, engram_hasher=None,
            args=SimpleNamespace(engram_layer_ids=(), kv_source_layers=()),
            embed=lambda ids: ids[..., None].astype(mx.float32))
        experts = SimpleNamespace(release_prefill_slots=lambda: None,
            reuse_layer_buffers=nullcontext, batched_layer=lambda _: nullcontext(),
            prefetch_layer=lambda i: events.append(('prefetch', i)))
        model = SimpleNamespace(model=core)
        tokens = list(range(4097))
        hidden = _deepseek_v41_layer_major_prefill(model, tokens, [SimpleNamespace(cache=state)],
            1024, experts, RuntimeConfig(v41_next_layer_prefetch=True), target_layers=(1,))
        self.assertEqual(events[0], ('prefetch', 1))
        self.assertEqual([event[1] for event in events if event[0] == 'ffn'], [4096, 1, 4096, 1])
        self.assertEqual([event[2] for event in events if event[0] == 'attention'], [1024]*4 + [1] + [1024]*4 + [1])
        self.assertEqual(hidden[0, -1, 0].item(), (4096 + 1) * 2)
        self.assertEqual(state.offset, 4097)
