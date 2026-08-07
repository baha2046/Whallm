from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from mlx_lm.models import deepseek_v4

from deepseek_v4_ssd.expert_cache import ExpertCache, ExpertWeights, ResidentExperts
from deepseek_v4_ssd.fp8_cache import CorrectPoolingCache, MXFP8PoolingCache
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import (
    _ORIGINAL_SPARSE_POOLED_ATTENTION,
    _StreamingSwitchGLU,
    _correct_compressor,
    _sparse_pooled_attention,
    _stable_topk_indices,
)


class ModelRuntimeTests(unittest.TestCase):
    def test_model_load_and_request_share_cross_thread_stream(self):
        load_stream = None

        def fake_load_model(_installed, _config):
            nonlocal load_stream
            load_stream = mx.default_stream(mx.gpu)
            return object(), SimpleNamespace(close=lambda: None)

        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1)
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
            value = mx.ones((1,)) + 1
            mx.async_eval(value)
            mx.eval(value)
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
            thread = threading.Thread(target=generate)
            thread.start()
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual([piece.text for piece in pieces], ["OK"])

    def test_generation_tracks_phases_and_evaluates_cache_state(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1)
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

        def fake_stream_generate(*_args, **options):
            received_options.update(options)
            yield from responses

        with (
            patch(
                "deepseek_v4_ssd.generation.load_model",
                return_value=(object(), SimpleNamespace(close=lambda: None)),
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
        self.assertEqual(metrics["runtime_prompt_tokens"], 4)
        self.assertEqual(metrics["runtime_generation_tokens"], 2)
        self.assertEqual(metrics["cache_state_eval_count"], 2)
        self.assertGreaterEqual(metrics["time_to_first_token_seconds"], 0)
        self.assertGreaterEqual(metrics["decode_seconds"], 0)

    def test_generation_reuses_a_complete_prompt_prefix(self):
        installed = SimpleNamespace(root=Path("/tmp/tokenizer"))
        config = SimpleNamespace(prefill_step_size=1)
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


class ExpertCacheTests(unittest.TestCase):
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

class MXFP8PoolingCacheTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
