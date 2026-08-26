from __future__ import annotations

import importlib
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models import deepseek_v4
from mlx_lm.models.cache import CacheList

from deepseek_v4_ssd.expert_cache import (
    BatchedExperts,
    CacheMetrics,
    ExpertCache,
    ExpertWeights,
    ResidentExperts,
    _ReadLimiter,
)
from deepseek_v4_ssd.dspark import (
    DraftResult,
    VerificationMetrics,
    _confidence_prefix_length,
    _should_fallback,
    _verify,
    generate_tokens,
)
from deepseek_v4_ssd.fp8_cache import CorrectPoolingCache, MXFP8PoolingCache
from deepseek_v4_ssd.generation import (
    GenerationOptions,
    ModelRuntime,
    RuntimeMetrics,
    _RawEvalCacheList,
    _decode_cache_state,
    _encode_cache_state,
    _persistence_cache_state,
    _restore_persistence_cache,
)
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    _ORIGINAL_SPARSE_POOLED_ATTENTION,
    _StreamingSwitchGLU,
    _configure_memory_limits,
    _correct_compressor,
    _select_moe_step_size,
    _select_prefill_step_size,
    _sparse_pooled_attention,
    _stable_topk_indices,
    eval_prompt_cache,
    forward_with_hidden,
    layer_major_prefill,
    verification_forward_with_hidden,
)

_mlx_lm_generate = importlib.import_module("mlx_lm.generate")


class ModelRuntimeTests(unittest.TestCase):
    def test_model_load_and_request_share_cross_thread_stream(self):
        load_stream = None
        pending = None
        original_stream = _mlx_lm_generate.generation_stream

        def fake_load_model(_installed, _config):
            nonlocal load_stream
            load_stream = mx.default_stream(mx.gpu)
            return object(), SimpleNamespace(close=lambda: None)

        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1, layer_major_prefill=False)
        with (
            patch("deepseek_v4_ssd.generation.load_model", side_effect=fake_load_model),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda *_args, **_kwargs: [1],
                ),
            ),
        ):
            runtime = ModelRuntime(installed, config)

        self.assertEqual(runtime._generation_stream, load_stream)
        response = SimpleNamespace(
            text="OK",
            token=1,
            prompt_tokens=1,
            generation_tokens=1,
            finish_reason="stop",
        )
        pieces = []
        errors = []

        def fake_stream_generate(*_args, **_kwargs):
            nonlocal pending
            with mx.stream(_mlx_lm_generate.generation_stream):
                if pending is not None:
                    mx.eval(pending)
                pending = mx.ones((1,)) + 1
                yield response

        def generate():
            try:
                pieces.extend(runtime.stream("test", GenerationOptions(max_tokens=1)))
            except Exception as error:
                errors.append(error)

        with (
            patch("deepseek_v4_ssd.generation.make_prompt_cache", return_value=[]),
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
        ):
            for _ in range(2):
                thread = threading.Thread(target=generate)
                thread.start()
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual([piece.text for piece in pieces], ["OK", "OK"])
        self.assertIs(_mlx_lm_generate.generation_stream, original_stream)

    def test_generation_tracks_phases_and_evaluates_cache_state(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1, layer_major_prefill=False)
        cache = SimpleNamespace(state=(mx.array([1]),))
        responses = [
            SimpleNamespace(
                text="A",
                token=1,
                prompt_tokens=4,
                generation_tokens=1,
                finish_reason=None,
            ),
            SimpleNamespace(
                text="B",
                token=2,
                prompt_tokens=4,
                generation_tokens=2,
                finish_reason="stop",
            ),
        ]
        received_options = {}
        route_phases = []
        observed_phases = []

        class ExpertCache:
            active_phase = None

            @contextmanager
            def trace_routes(self, phase):
                route_phases.append(phase)
                self.active_phase = phase
                try:
                    yield
                finally:
                    self.active_phase = None

            def close(self):
                pass

        expert_cache = ExpertCache()

        def fake_stream_generate(*_args, **options):
            received_options.update(options)
            for response in responses:
                observed_phases.append(expert_cache.active_phase)
                yield response

        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), expert_cache),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda *_args, **_kwargs: [1, 2, 3, 4],
                ),
            ),
            patch("deepseek_v4_ssd.generation.make_prompt_cache", return_value=[cache]),
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval") as evaluate,
        ):
            runtime = ModelRuntime(installed, config)
            pieces = list(runtime.stream("test", GenerationOptions(max_tokens=2)))

        metrics = runtime.metrics.snapshot()
        self.assertEqual([piece.text for piece in pieces], ["A", "B"])
        self.assertEqual(received_options["prompt_cache"], [cache])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(route_phases, ["prefill", "decode", "decode"])
        self.assertEqual(observed_phases, ["prefill", "decode"])
        self.assertEqual(metrics["runtime_prompt_tokens"], 4)
        self.assertEqual(metrics["runtime_generation_tokens"], 2)
        self.assertEqual(metrics["accumulated_generation_tokens"], 2)
        self.assertEqual(metrics["completed_request_count"], 1)
        self.assertEqual(metrics["cache_state_eval_count"], 2)
        self.assertGreaterEqual(metrics["time_to_first_token_seconds"], 0)
        self.assertGreaterEqual(metrics["prefill_tokens_per_second"], 0)
        self.assertGreaterEqual(metrics["decode_seconds"], 0)

    def test_generation_reuses_a_complete_prompt_prefix(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1, layer_major_prefill=False)
        cache = SimpleNamespace(state=(mx.array([1]),))
        prompts = {
            "first": [1, 2],
            "continued": [1, 2, 42, 3],
            "branch": [9, 10],
        }
        received_prompts = []

        def fake_stream_generate(_model, _tokenizer, prompt, **_options):
            received_prompts.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                prompt_tokens=len(prompt),
                generation_tokens=1,
                finish_reason="length",
            )

        tokenizer = SimpleNamespace(
            bos_token=None,
            encode=lambda prompt, **_options: prompts[prompt],
        )
        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "deepseek_v4_ssd.generation.make_prompt_cache",
                return_value=[cache],
            ) as make_cache,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval"),
        ):
            runtime = ModelRuntime(installed, config)
            list(runtime.stream("first", GenerationOptions(max_tokens=1)))
            pieces = list(
                runtime.stream("continued", GenerationOptions(max_tokens=1))
            )
            reused_metrics = runtime.metrics.snapshot()
            list(runtime.stream("branch", GenerationOptions(max_tokens=1)))

        self.assertEqual(received_prompts, [[1, 2], [3], [9, 10]])
        self.assertEqual(make_cache.call_count, 2)
        self.assertEqual(pieces[0].prompt_tokens, 4)
        self.assertEqual(reused_metrics["prompt_cache_reused_tokens"], 3)
        self.assertEqual(runtime.metrics.snapshot()["prompt_cache_reused_tokens"], 0)

    def test_generation_keeps_two_independent_prompt_cache_timelines(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(
            prefill_step_size=1,
            prompt_cache_entries=2,
            prompt_cache_memory_gib=1,
            layer_major_prefill=False,
        )
        prompts = {
            "first": [1, 2],
            "branch": [9, 10],
            "continued": [1, 2, 42, 3],
        }
        received_prompts = []

        def fake_stream_generate(_model, _tokenizer, prompt, **_options):
            received_prompts.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                prompt_tokens=len(prompt),
                generation_tokens=1,
                finish_reason="length",
            )

        tokenizer = SimpleNamespace(
            bos_token=None,
            encode=lambda prompt, **_options: prompts[prompt],
        )
        caches = [[SimpleNamespace(state=(), nbytes=1)] for _ in range(2)]
        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ),
            patch(
                "deepseek_v4_ssd.generation.make_prompt_cache",
                side_effect=caches,
            ) as make_cache,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval"),
        ):
            runtime = ModelRuntime(installed, config)
            list(runtime.stream("first", GenerationOptions(max_tokens=1)))
            list(runtime.stream("branch", GenerationOptions(max_tokens=1)))
            list(runtime.stream("continued", GenerationOptions(max_tokens=1)))

        self.assertEqual(received_prompts, [[1, 2], [9, 10], [3]])
        self.assertEqual(make_cache.call_count, 2)

    def test_warm_prompt_leaves_one_token_for_the_first_request(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(
            prefill_step_size=1,
            prompt_cache_entries=1,
            prompt_cache_memory_gib=1,
            layer_major_prefill=False,
        )
        prompts = {"warm": [1, 2, 3], "request": [1, 2, 3, 4]}
        received_prompts = []
        cache = [SimpleNamespace(state=(), nbytes=1)]

        def fake_stream_generate(_model, _tokenizer, prompt, **_options):
            received_prompts.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                generation_tokens=1,
                finish_reason="length",
            )

        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
            ),
            patch(
                "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    bos_token=None,
                    encode=lambda prompt, **_options: prompts[prompt],
                ),
            ),
            patch("deepseek_v4_ssd.generation.make_prompt_cache", return_value=cache),
            patch("deepseek_v4_ssd.generation.layer_major_prefill") as prefill,
            patch(
                "deepseek_v4_ssd.generation.stream_generate",
                side_effect=fake_stream_generate,
            ),
            patch("deepseek_v4_ssd.generation.mx.eval"),
        ):
            runtime = ModelRuntime(installed, config)
            warmed = runtime.warm_prompt("warm")
            list(runtime.stream("request", GenerationOptions(max_tokens=1)))

        self.assertEqual(warmed, 2)
        self.assertEqual(prefill.call_args.args[1], [1, 2])
        self.assertEqual(received_prompts, [[3, 4]])
        self.assertEqual(runtime.metrics.snapshot()["prompt_cache_reused_tokens"], 2)

    def test_persistent_prompt_cache_survives_restart(self):
        class Cache:
            def __init__(self):
                self._state = (mx.array([7]), mx.empty((0,), dtype=mx.float32))
                self.nbytes = 4

            @property
            def state(self):
                return self._state

            @state.setter
            def state(self, value):
                self._state = value

        prompts = {
            "first": [1, 2],
            "continued": [1, 2, 42, 3],
        }
        received = []

        def generate(_model, _tokenizer, prompt, **_options):
            received.append(prompt)
            yield SimpleNamespace(
                text="A",
                token=42,
                generation_tokens=1,
                finish_reason="length",
            )

        with tempfile.TemporaryDirectory() as directory:
            installed = SimpleNamespace(
                root=Path("/tmp/tokenizer"),
                revision="fixture-revision",
            )
            config = SimpleNamespace(
                prefill_step_size=1,
                layer_major_prefill=False,
                prompt_cache_entries=2,
                prompt_cache_memory_gib=1,
                persistent_prompt_cache=True,
                persistent_prompt_cache_entries=8,
                prompt_cache_directory=directory,
            )
            tokenizer = SimpleNamespace(
                bos_token=None,
                encode=lambda prompt, **_options: prompts[prompt],
            )
            with (
                patch(
                    "deepseek_v4_ssd.generation.load_model",
                    return_value=(object(), SimpleNamespace(close=lambda: None)),
                ),
                patch(
                    "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "deepseek_v4_ssd.generation.make_prompt_cache",
                    side_effect=lambda _model: [Cache()],
                ),
                patch(
                    "deepseek_v4_ssd.generation.stream_generate",
                    side_effect=generate,
                ),
            ):
                first = ModelRuntime(installed, config)
                list(first.stream("first", GenerationOptions(max_tokens=1)))
                first.close()

                second = ModelRuntime(installed, config)
                list(second.stream("continued", GenerationOptions(max_tokens=1)))
                reused = second.metrics.snapshot()["prompt_cache_reused_tokens"]
                second.close()

        self.assertEqual(received, [[1, 2], [3]])
        self.assertEqual(reused, 3)

    def test_decode_rate_includes_cache_evaluation(self):
        metrics = RuntimeMetrics()
        metrics.start(4, 0, 1, False, CacheMetrics())
        metrics.record_token(1, 1.0, 0.25)
        metrics.record_token(2, 2.0, 0.5)
        metrics.finish(CacheMetrics())

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["decode_model_step_seconds"], 2.0)
        self.assertEqual(snapshot["decode_cache_eval_seconds"], 0.5)
        self.assertEqual(snapshot["decode_end_to_end_seconds"], 2.5)
        self.assertEqual(snapshot["decode_model_step_tokens_per_second"], 0.5)
        self.assertEqual(snapshot["decode_tokens_per_second"], 0.4)
        self.assertEqual(snapshot["decode_latency_p50_seconds"], 2.5)
        self.assertEqual(snapshot["decode_latency_p95_seconds"], 2.5)

    def test_output_tokens_accumulate_across_requests(self):
        metrics = RuntimeMetrics()
        metrics.start(4, 0, 1, False, CacheMetrics())
        metrics.record_token(2, 1.0, 0.0)
        self.assertEqual(metrics.snapshot()["accumulated_generation_tokens"], 2)
        metrics.finish(CacheMetrics())

        metrics.start(4, 0, 1, False, CacheMetrics())
        metrics.record_token(3, 1.0, 0.0)
        self.assertEqual(metrics.snapshot()["accumulated_generation_tokens"], 5)
        metrics.finish(CacheMetrics())

        self.assertEqual(metrics.snapshot()["accumulated_generation_tokens"], 5)


class MemoryLimitTests(unittest.TestCase):
    def test_automatic_memory_limit_uses_metals_recommended_maximum(self):
        maximum = 40_200_896_512
        with (
            patch(
                "deepseek_v4_ssd.model.mx.device_info",
                return_value={"max_recommended_working_set_size": maximum},
            ),
            patch("deepseek_v4_ssd.model.mx.set_memory_limit") as set_memory_limit,
            patch("deepseek_v4_ssd.model.mx.set_wired_limit") as set_wired_limit,
        ):
            selected = _configure_memory_limits(RuntimeConfig())

        self.assertEqual(selected, maximum)
        set_memory_limit.assert_called_once_with(maximum)
        set_wired_limit.assert_called_once_with(maximum)

    def test_explicit_memory_limit_caps_wired_memory_at_metals_maximum(self):
        maximum = 40_200_896_512
        requested = 48 * 1024**3
        with (
            patch(
                "deepseek_v4_ssd.model.mx.device_info",
                return_value={"max_recommended_working_set_size": maximum},
            ),
            patch("deepseek_v4_ssd.model.mx.set_memory_limit") as set_memory_limit,
            patch("deepseek_v4_ssd.model.mx.set_wired_limit") as set_wired_limit,
        ):
            selected = _configure_memory_limits(RuntimeConfig(memory_limit_gib=48))

        self.assertEqual(selected, requested)
        set_memory_limit.assert_called_once_with(requested)
        set_wired_limit.assert_called_once_with(maximum)


class PrefillTests(unittest.TestCase):
    def test_runtime_uses_1152_expert_slots_by_default(self):
        self.assertEqual(RuntimeConfig().slots, 1_152)
        self.assertEqual(RuntimeConfig().read_workers, 4)
        self.assertEqual(RuntimeConfig().memory_limit_gib, 0)
        self.assertEqual(RuntimeConfig().dspark_slots, 768)
        self.assertTrue(RuntimeConfig().ready_expert_decode)

    def test_automatic_step_size_uses_larger_chunks_for_long_prompts(self):
        self.assertEqual(_select_prefill_step_size(0, 1_000), 128)
        self.assertEqual(_select_prefill_step_size(0, 4_000), 256)
        self.assertEqual(_select_prefill_step_size(0, 14_000), 1_024)
        self.assertEqual(_select_prefill_step_size(192, 14_000), 192)
        self.assertEqual(_select_moe_step_size(0, 14_000), 4_096)
        self.assertEqual(_select_moe_step_size(2_048, 14_000), 2_048)

    def test_metrics_report_layer_major_prefill_token_count(self):
        metrics = RuntimeMetrics()
        metrics.start(4_098, 0, 1_024, True, CacheMetrics())

        self.assertEqual(metrics.snapshot()["layer_major_prefill_tokens"], 4_097)

    def test_layer_major_prefill_finishes_one_layer_before_the_next(self):
        calls = []

        class Cache:
            def __init__(self):
                self.offset = 0

            @property
            def state(self):
                raise AssertionError("layer-major prefill must evaluate raw cache arrays")

        class Layer:
            def __init__(self, index):
                self.index = index

            def __call__(self, hidden, _mask, cache, input_ids):
                calls.append((self.index, input_ids.reshape(-1).tolist()))
                cache.offset += input_ids.shape[1]
                return hidden + 1

        class PinnedCache:
            def pin_layer(self, _layer):
                from contextlib import nullcontext

                return nullcontext()

        core = SimpleNamespace(
            embed_tokens=lambda inputs: inputs[..., None].astype(mx.float32),
            pipeline_layers=[Layer(0), Layer(1)],
            args=SimpleNamespace(hc_mult=1, sliding_window=4),
        )
        model = SimpleNamespace(model=core)
        caches = [Cache(), Cache()]
        with patch(
            "deepseek_v4_ssd.model.deepseek_v4.create_attention_mask",
            return_value=None,
        ):
            layer_major_prefill(model, [1, 2, 3, 4], caches, 2, PinnedCache())

        self.assertEqual(
            calls,
            [(0, [1, 2]), (0, [3, 4]), (1, [1, 2]), (1, [3, 4])],
        )

    def test_layer_major_prefill_matches_standard_next_token_logits(self):
        from contextlib import nullcontext

        arguments = deepseek_v4.ModelArgs(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            qk_rope_head_dim=16,
            head_dim=16,
            o_groups=1,
            o_lora_rank=16,
            hc_mult=1,
            compress_ratios=[0, 0],
        )
        model = deepseek_v4.Model(arguments)
        standard_cache = model.make_cache()
        layer_major_cache = model.make_cache()
        prompt = mx.array([[1, 2, 3, 4]])
        model(prompt, cache=standard_cache)

        expert_cache = SimpleNamespace(pin_layer=lambda _layer: nullcontext())
        layer_major_prefill(
            model,
            prompt.reshape(-1).tolist(),
            layer_major_cache,
            2,
            expert_cache,
        )
        expected = model(mx.array([[5]]), cache=standard_cache)
        actual = model(mx.array([[5]]), cache=layer_major_cache)
        mx.eval(expected, actual)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

class DSparkTests(unittest.TestCase):
    def test_hardware_scheduler_falls_back_when_dspark_costs_more(self):
        draft = DraftResult([2], [mx.zeros((5,))], [0.8], 0.2)

        self.assertTrue(_should_fallback(0.1, draft, 0, 0.2))
        self.assertFalse(_should_fallback(0.3, draft, 1, 0.1))

    def test_confidence_scheduler_can_skip_the_whole_draft(self):
        self.assertEqual(_confidence_prefix_length([0.4, 0.9], 0.6), 0)
        self.assertEqual(_confidence_prefix_length([0.8, 0.5], 0.6), 1)

    def test_next_round_receives_every_committed_hidden_state(self):
        calls = []

        class DSpark:
            target_layers = (0,)

            def reset_cache(self):
                pass

            def prefill_context(self, _hidden, _offset):
                pass

            def draft(
                self,
                _model,
                _anchor,
                hidden,
                start_pos,
                _temperature,
                _top_p,
                _threshold,
            ):
                calls.append((hidden.shape[1], start_pos))
                return DraftResult(
                    [2, 3],
                    [mx.zeros((5,)), mx.zeros((5,))],
                    [1.0, 1.0],
                    0.0,
                )

        def forward(_model, inputs, _cache, _layers):
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            hidden = mx.ones((1, inputs.shape[1], 1))
            return logits, hidden

        def verify(_model, tokens, _cache, _layers):
            logits = mx.zeros((1, len(tokens), 5))
            for index, token in enumerate((2, 3, 4)):
                logits[0, index, token] = 1
            hidden = mx.ones((1, len(tokens), 1))
            return logits, hidden, [None], VerificationMetrics()

        with (
            patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward),
            patch("deepseek_v4_ssd.dspark._target_sequence", side_effect=verify),
            patch("deepseek_v4_ssd.dspark._should_fallback", return_value=False),
        ):
            list(
                generate_tokens(
                    [0],
                    object(),
                    DSpark(),
                    [None],
                    max_tokens=6,
                    prefill_step_size=1,
                    temperature=0,
                    top_p=1,
                )
            )

        self.assertEqual(calls, [(1, 1), (3, 2)])

    def test_rejected_draft_replays_only_the_committed_prefix(self):
        replayed = []

        class DSpark:
            target_layers = (0,)

            def reset_cache(self):
                pass

            def prefill_context(self, _hidden, _offset):
                pass

            def draft(self, *_args):
                return DraftResult(
                    [2, 4],
                    [mx.zeros((5,)), mx.zeros((5,))],
                    [1.0, 1.0],
                    0.0,
                )

        def forward(_model, inputs, _cache, _layers):
            values = inputs.reshape(-1).tolist()
            if len(values) > 1:
                replayed.append(values)
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            return logits, mx.ones((1, inputs.shape[1], 1))

        def verify(_model, tokens, _cache, _layers):
            logits = mx.zeros((1, len(tokens), 5))
            logits[0, 0, 2] = 1
            logits[0, 1, 3] = 1
            return logits, mx.ones((1, len(tokens), 1)), [None], VerificationMetrics()

        with (
            patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward),
            patch("deepseek_v4_ssd.dspark._target_sequence", side_effect=verify),
            patch("deepseek_v4_ssd.dspark._should_fallback", return_value=True),
        ):
            list(
                generate_tokens(
                    [0],
                    object(),
                    DSpark(),
                    [None],
                    max_tokens=4,
                    prefill_step_size=1,
                    temperature=0,
                    top_p=1,
                )
            )

        self.assertEqual(replayed, [[1, 2]])

    def test_verification_batches_moe_without_changing_sequential_results(self):
        mx.random.seed(7)
        arguments = deepseek_v4.ModelArgs(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            q_lora_rank=16,
            qk_rope_head_dim=16,
            head_dim=16,
            o_groups=1,
            o_lora_rank=16,
            hc_mult=1,
            compress_ratios=[0, 4, 128],
            num_hash_layers=0,
            sliding_window=8,
        )
        model = deepseek_v4.Model(arguments)
        sequential_cache = model.make_cache()
        verification_cache = model.make_cache()
        prefix = mx.array([[1, 2, 3]])
        forward_with_hidden(model, prefix, sequential_cache, (1, 2))
        forward_with_hidden(model, prefix, verification_cache, (1, 2))

        tokens = [4, 5, 6, 7, 8]
        sequential_logits = []
        sequential_hidden = []
        for token in tokens:
            logits, hidden = forward_with_hidden(
                model,
                mx.array([[token]]),
                sequential_cache,
                (1, 2),
            )
            mx.eval(logits, hidden)
            sequential_logits.append(logits)
            sequential_hidden.append(hidden)
        expected_logits = mx.concatenate(sequential_logits, axis=1)
        expected_hidden = mx.concatenate(sequential_hidden, axis=1)
        actual_logits, actual_hidden, verified_cache, metrics = (
            verification_forward_with_hidden(
                model,
                mx.array([tokens]),
                verification_cache,
                (1, 2),
            )
        )
        mx.eval(expected_logits, expected_hidden, actual_logits, actual_hidden)

        self.assertEqual(
            mx.argmax(actual_logits, axis=-1).tolist(),
            mx.argmax(expected_logits, axis=-1).tolist(),
        )
        self.assertLess(mx.mean(mx.abs(actual_logits - expected_logits)).item(), 0.01)
        self.assertLess(mx.mean(mx.abs(actual_hidden - expected_hidden)).item(), 0.01)
        self.assertEqual(metrics.per_position_cache_copies, 0)
        self.assertEqual(metrics.state_fetch_count, 0)
        self.assertEqual(metrics.block_attention_layers, 3)

        retained_cache = model.make_cache()
        forward_with_hidden(model, prefix, retained_cache, (1, 2))
        expected, _ = forward_with_hidden(
            model, mx.array([[9]]), retained_cache, (1, 2)
        )
        actual, _ = forward_with_hidden(
            model, mx.array([[9]]), verification_cache, (1, 2)
        )
        mx.eval(expected, actual)
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 0.005)

        expected, _ = forward_with_hidden(
            model, mx.array([[10]]), sequential_cache, (1, 2)
        )
        actual, _ = forward_with_hidden(
            model, mx.array([[10]]), verified_cache, (1, 2)
        )
        mx.eval(expected, actual)
        self.assertEqual(
            mx.argmax(actual, axis=-1).tolist(),
            mx.argmax(expected, axis=-1).tolist(),
        )
        self.assertLess(mx.mean(mx.abs(actual - expected)).item(), 0.01)

    def test_greedy_verification_stops_at_the_first_mismatch(self):
        draft = DraftResult(
            tokens=[2, 4],
            logprobs=[mx.zeros((5,)), mx.zeros((5,))],
            confidence=[0.9, 0.8],
            seconds=0.01,
        )
        target = mx.array(
            [
                [-4.0, -4.0, 0.0, -4.0, -4.0],
                [-4.0, -4.0, -4.0, 0.0, -4.0],
                [-4.0, 0.0, -4.0, -4.0, -4.0],
            ]
        )

        accepted, token, _ = _verify(draft, target, temperature=0)

        self.assertEqual(accepted, 1)
        self.assertEqual(token, 3)

    def test_sampling_verification_accepts_identical_distributions(self):
        distribution = mx.array([-float("inf"), 0.0, -float("inf")])
        draft = DraftResult([1], [distribution], [1.0], 0.01)
        target = mx.stack([distribution, distribution])

        accepted, token, _ = _verify(draft, target, temperature=1)

        self.assertEqual(accepted, 1)
        self.assertEqual(token, 1)

class CacheMetricsTests(unittest.TestCase):
    def test_delta_reports_only_the_current_request(self):
        before = CacheMetrics(hits=10, misses=4, evictions=2, bytes_read=100)
        after = CacheMetrics(hits=13, misses=9, evictions=6, bytes_read=500)

        delta = after.delta(before)

        self.assertEqual(delta.hits, 3)
        self.assertEqual(delta.misses, 5)
        self.assertEqual(delta.evictions, 4)
        self.assertEqual(delta.bytes_read, 400)


class ExpertCacheTests(unittest.TestCase):
    def test_read_limiter_caps_aggregate_preadv_rate(self):
        limiter = _ReadLimiter(1_000_000_000)
        with (
            patch("deepseek_v4_ssd.expert_cache.os.preadv", return_value=500_000_000),
            patch(
                "deepseek_v4_ssd.expert_cache.time.perf_counter",
                side_effect=[10.0, 10.25],
            ),
            patch("deepseek_v4_ssd.expert_cache.time.sleep") as sleep,
        ):
            count = limiter.preadv(1, [memoryview(bytearray(1))], 0)

        self.assertEqual(count, 500_000_000)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.25)

    def test_cache_reads_an_independent_expert_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            dspark_experts = root / "dspark/experts"
            dspark_experts.mkdir(parents=True)
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            (root / "experts/layer_00.bin").write_bytes(bytes(15))
            dspark_experts.joinpath("layer_00.bin").write_bytes(bytes(range(15)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=1,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(
                model,
                slots=1,
                read_workers=1,
                layer_count=1,
                expert_directory=dspark_experts,
            ) as cache:
                ready = dict(cache.iter_ready(0, [0]))

            self.assertEqual(cache.metrics.bytes_read, 15)
            self.assertEqual(
                ready[0].w1.view(mx.uint8).tolist(),
                [[0, 1, 2, 3]],
            )
            self.assertEqual(
                ready[0].w13.view(mx.uint8).tolist(),
                [[10, 11, 12, 13], [0, 1, 2, 3]],
            )
            self.assertEqual(cache.metrics.upload_seconds, 0.0)

    def test_lfu_cache_reuses_and_evicts_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            blob_size = 15
            (root / "experts/layer_00.bin").write_bytes(bytes(range(blob_size * 2)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=2,
                selected_expert_count=1,
                expert_blob_size=blob_size,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=1, read_workers=1) as cache:
                first = cache.get_many(0, [0])
                cache.get_many(0, [0])
                second = cache.get_many(0, [1])
                self.assertEqual(cache.metrics.hits, 1)
                self.assertEqual(cache.metrics.misses, 2)
                self.assertEqual(cache.metrics.evictions, 1)
                self.assertEqual(cache.metrics.bytes_read, blob_size * 2)
                self.assertEqual(first.slots[0], second.slots[1])
                individual = cache.get_many(0, [1])
                self.assertEqual(len(individual.individual_weights), 1)
                self.assertEqual(individual.individual_weights[0].w1.shape, (1, 1))

    def test_layer_limits_keep_resident_experts_across_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            for layer in range(2):
                (root / f"experts/layer_{layer:02d}.bin").write_bytes(bytes(range(45)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=2,
                expert_count=3,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=4, read_workers=1) as cache:
                cache.get_many(0, [0])
                cache.get_many(0, [1])
                cache.get_many(0, [2])
                cache.get_many(1, [0])
                cache.get_many(1, [1])
                misses = cache.metrics.misses
                cache.get_many(1, [0])
                cache.get_many(0, [1])
                self.assertEqual(cache.metrics.misses, misses)
                self.assertEqual(cache.resident_count, 4)

    def test_pinned_prefill_layer_keeps_its_experts_resident(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 1), 4, 1),
                Tensor("w2.weight", "I8", (1, 4), 5, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 1), 9, 1),
                Tensor("w3.weight", "I8", (1, 4), 10, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 1), 14, 1),
            )
            for layer in range(2):
                (root / f"experts/layer_{layer:02d}.bin").write_bytes(bytes(range(60)))
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=2,
                expert_count=4,
                selected_expert_count=1,
                expert_blob_size=15,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=4, read_workers=1) as cache:
                for _ in range(8):
                    cache.get_many(0, [0, 1])
                with cache.pin_layer(1):
                    cache.get_many(1, [0])
                    cache.get_many(1, [1])
                    cache.get_many(1, [2])
                    misses = cache.metrics.misses
                    cache.get_many(1, [0])

                self.assertEqual(cache.metrics.misses, misses)

    def test_batched_layer_uses_one_strided_expert_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            regions = (
                Tensor("w1.weight", "I8", (1, 4), 0, 4),
                Tensor("w1.scale", "F8_E8M0", (1, 4), 4, 4),
                Tensor("w2.weight", "I8", (1, 4), 8, 4),
                Tensor("w2.scale", "F8_E8M0", (1, 4), 12, 4),
                Tensor("w3.weight", "I8", (1, 4), 16, 4),
                Tensor("w3.scale", "F8_E8M0", (1, 4), 20, 4),
            )
            blob_size = 24
            source = bytes(range(blob_size * 2))
            (root / "experts/layer_00.bin").write_bytes(source)
            model = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=2,
                selected_expert_count=1,
                expert_blob_size=blob_size,
                common_tensors=(),
                expert_regions=regions,
            )

            with ExpertCache(model, slots=1, read_workers=1) as cache:
                cache.prefetch_layer(0)
                with cache.batched_layer(0) as batched:
                    mx.eval(batched.w1, batched.w3_scales)
                    self.assertIs(cache.current_batched(0), batched)
                    self.assertEqual(batched.w1.shape, (2, 1, 1))
                    self.assertEqual(
                        batched.w1_scales[:, 0, 0].tolist(), [4, 28]
                    )
                self.assertIsNone(cache.current_batched(0))
                self.assertEqual(cache.metrics.bytes_read, len(source))

class MXFP8PoolingCacheTests(unittest.TestCase):
    def test_generation_cache_state_exposes_raw_quantized_arrays(self):
        class QuantizedOnlyCache(MXFP8PoolingCache):
            @property
            def state(self):
                raise AssertionError("generation must not rebuild a BF16 cache")

        cache = QuantizedOnlyCache(ratio=4)
        cache.update(mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16))

        arrays = _RawEvalCacheList(cache).state
        mx.eval(*arrays)

        self.assertGreater(len(arrays), 0)

    def test_persistence_round_trip_keeps_quantized_chunks(self):
        class QuantizedOnlyCache(MXFP8PoolingCache):
            @property
            def state(self):
                raise AssertionError("persistence must not rebuild a BF16 cache")

        source = QuantizedOnlyCache(ratio=4)
        source.update(mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16))
        eval_prompt_cache([CacheList(source)])

        arrays = {}
        schema = _encode_cache_state(
            _persistence_cache_state([CacheList(source)]),
            arrays,
        )
        schema = json.loads(json.dumps(schema))
        decoded = _decode_cache_state(schema, arrays)
        restored = MXFP8PoolingCache(ratio=4)
        _restore_persistence_cache([CacheList(restored)], decoded)
        eval_prompt_cache([CacheList(restored)])

        query = mx.random.uniform(shape=(1, 3, 2, 64)).astype(mx.float32)
        expected = source.quantized_matmul(query)
        actual = restored.quantized_matmul(query)
        mx.eval(expected, actual)

        self.assertEqual(restored.offset, source.offset)
        self.assertEqual(restored.nbytes, source.nbytes)
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-6)

    def test_completed_chunks_use_less_memory_than_bfloat16(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16)
        restored = cache.update_and_fetch(source)
        mx.eval(restored)

        self.assertEqual(restored.shape, source.shape)
        self.assertEqual(cache.offset, 64)
        self.assertLess(cache.nbytes, source.nbytes)
        self.assertLess(mx.mean(mx.abs(restored - source)).item(), 0.02)
        state = cache.state
        self.assertEqual(cache.offset, 64)
        self.assertEqual(state[2].shape, source.shape)

    def test_quantized_chunks_multiply_without_fetching_complete_cache(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16)
        cache.update(source)
        query = mx.random.uniform(shape=(1, 3, 2, 64)).astype(mx.float32)

        actual = cache.quantized_matmul(query)
        restored = cache._fetch(1, 64, mx.bfloat16)
        expected = query @ restored[:, None].swapaxes(-1, -2).astype(mx.float32)
        mx.eval(actual, expected)

        self.assertEqual(actual.shape, (1, 3, 2, 130))
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 0.05)

    def test_fp4_index_view_matches_the_fp8_index_shape(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16)
        cache.update(source)
        query = mx.random.uniform(shape=(1, 3, 2, 64)).astype(mx.float32)

        actual = cache.index_matmul(query)
        expected = cache.quantized_matmul(query)
        mx.eval(actual, expected)

        self.assertEqual(actual.shape, expected.shape)
        relative_error = mx.mean(mx.abs(actual - expected)) / mx.mean(mx.abs(expected))
        self.assertLess(relative_error.item(), 0.1)

    def test_packed_chunks_are_reused_until_the_cache_changes(self):
        cache = MXFP8PoolingCache(ratio=4)
        cache.update(mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16))

        packed = cache._packed()
        self.assertIs(cache._packed(), packed)

        cache.update(mx.random.uniform(shape=(1, 64, 64)).astype(mx.bfloat16))
        self.assertIsNot(cache._packed(), packed)

    def test_gather_reads_selected_rows_across_quantized_chunks(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 130, 64)).astype(mx.bfloat16)
        cache.update(source)
        indices = mx.array([[[0, 63, 64], [65, 128, 129]]])

        actual = cache.gather(indices)
        restored = cache._fetch(1, 64, mx.bfloat16)
        expected = mx.take_along_axis(
            mx.broadcast_to(restored[:, None], (1, 2, 130, 64)),
            mx.broadcast_to(indices[..., None], (1, 2, 3, 64)),
            axis=2,
        )
        mx.eval(actual, expected)

        self.assertEqual(actual.shape, (1, 2, 3, 64))
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_gather_reads_pending_rows_before_the_first_chunk(self):
        cache = MXFP8PoolingCache(ratio=4)
        source = mx.random.uniform(shape=(1, 32, 64)).astype(mx.bfloat16)
        cache.update(source)
        indices = mx.array([[[0, 15, 31]]])

        actual = cache.gather(indices)
        expected = mx.take_along_axis(
            source[:, None],
            mx.broadcast_to(indices[..., None], (1, 1, 3, 64)),
            axis=2,
        )
        mx.eval(actual, expected)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_sparse_attention_matches_complete_cache(self):
        cache = MXFP8PoolingCache(ratio=4)
        cache.update(mx.random.uniform(shape=(1, 600, 64)).astype(mx.bfloat16))
        query = mx.random.uniform(shape=(1, 2, 2, 64)).astype(mx.bfloat16)
        local = mx.random.uniform(shape=(1, 1, 3, 64)).astype(mx.bfloat16)
        topk = mx.array([[[0, 63, 511], [64, 512, 599]]])
        sinks = mx.zeros((2,), dtype=mx.float32)

        actual = _sparse_pooled_attention(
            query,
            local,
            cache,
            topk,
            None,
            None,
            0.125,
            sinks,
        )
        complete = cache.fetch(1, 64, mx.bfloat16)
        expected = _ORIGINAL_SPARSE_POOLED_ATTENTION(
            query,
            local,
            complete,
            topk,
            None,
            None,
            0.125,
            sinks,
        )
        mx.eval(actual, expected)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_ratio_four_chunking_matches_one_prefill(self):
        arguments = deepseek_v4.ModelArgs(hidden_size=32)
        compressor = deepseek_v4.Compressor(arguments, compress_ratio=4, head_dim=64)
        source = mx.random.uniform(shape=(1, 8, 32)).astype(mx.bfloat16)

        chunked_cache = CorrectPoolingCache(4)
        _correct_compressor(compressor, source[:, :4], chunked_cache, 0)
        chunked = _correct_compressor(compressor, source[:, 4:], chunked_cache, 4)
        complete = _correct_compressor(compressor, source, CorrectPoolingCache(4), 0)
        mx.eval(chunked, complete)

        self.assertEqual(chunked.shape, complete.shape)
        self.assertLess(mx.max(mx.abs(chunked - complete)).item(), 1e-5)


class MXFP4Tests(unittest.TestCase):
    def test_ready_experts_keep_router_selection_order(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )

        cache = SimpleNamespace(
            ready_expert_decode=True,
            iter_ready=lambda _layer, _selected: iter(
                [(0, experts[0]), (1, experts[1])]
            ),
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        source = mx.ones((1, 1, 32), dtype=mx.bfloat16)
        selected = mx.array([[[1, 0]]])

        actual = switch(source, selected)
        reference_cache = SimpleNamespace(
            get_many=lambda _layer, _selected: ResidentExperts(
                tuple(experts), {0: 0, 1: 1}
            )
        )
        expected = _StreamingSwitchGLU(
            0,
            reference_cache,
            lambda up, _gate: up,
        )(source, selected)
        mx.eval(actual, expected)

        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_individual_experts_run_in_selected_order(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )

        def get_many(_layer, _experts):
            return ResidentExperts(tuple(experts), {0: 0, 1: 1})

        cache = SimpleNamespace(
            model=SimpleNamespace(expert_count=2),
            get_many=get_many,
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        def reference(weights):
            def restore(weight, scales):
                return mx.dequantize(
                    weight,
                    scales,
                    group_size=32,
                    bits=4,
                    mode="mxfp4",
                )

            def run(source):
                hidden = source @ restore(weights.w3, weights.w3_scales).T
                return hidden @ restore(weights.w2, weights.w2_scales).T

            return run

        for length in (1, 2):
            with self.subTest(length=length):
                source = mx.ones((1, length, 32), dtype=mx.bfloat16)
                selected = mx.array([[[1, 0] for _ in range(length)]])
                actual = switch(source, selected)
                expected = mx.concatenate(
                    [
                        mx.stack(
                            [
                                reference(experts[expert])(source[:, token : token + 1])
                                for expert in (1, 0)
                            ],
                            axis=-2,
                        )
                        for token in range(length)
                    ],
                    axis=1,
                )
                mx.eval(actual, expected)
                self.assertEqual(actual.shape, (1, length, 2, 32))
                self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)

    def test_topk_resolves_score_ties_by_position(self):
        scores = mx.array([[1.0, 1.0, 1.0, 0.5]])
        indices = _stable_topk_indices(scores, 2)
        self.assertEqual(indices.tolist(), [[0, 1]])

    def test_batched_experts_match_individual_experts(self):
        def quantized(value: float):
            return mx.quantize(
                mx.full((32, 32), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(value)
            w2, w2_scales = quantized(value + 0.125)
            w3, w3_scales = quantized(value + 0.25)
            experts.append(
                ExpertWeights(w1, w1_scales, w2, w2_scales, w3, w3_scales)
            )
        batched = BatchedExperts(
            *(
                mx.stack([getattr(expert, name) for expert in experts])
                for name in (
                    "w1",
                    "w1_scales",
                    "w2",
                    "w2_scales",
                    "w3",
                    "w3_scales",
                )
            ),
            w13=mx.stack(
                [mx.concatenate([expert.w3, expert.w1], axis=0) for expert in experts]
            ),
            w13_scales=mx.stack(
                [
                    mx.concatenate([expert.w3_scales, expert.w1_scales], axis=0)
                    for expert in experts
                ]
            ),
        )
        calls = []
        cache = SimpleNamespace(
            current_batched=lambda _layer: batched,
            record_gather_qmm=lambda count=3: calls.append(count),
        )
        switch = _StreamingSwitchGLU(0, cache, lambda up, _gate: up)
        source = mx.ones((1, 2, 32), dtype=mx.bfloat16)
        selected = mx.array([[[1, 0], [0, 1]]])

        actual = switch(source, selected)
        reference_cache = SimpleNamespace(
            get_many=lambda _layer, _experts: ResidentExperts(
                tuple(experts), {0: 0, 1: 1}
            )
        )
        expected = _StreamingSwitchGLU(
            0, reference_cache, lambda up, _gate: up
        )(source, selected)
        mx.eval(actual, expected)

        self.assertEqual(calls, [2])
        self.assertLess(mx.max(mx.abs(actual - expected)).item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
