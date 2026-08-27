from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.generation import _qwen_layer_major_prefill
from deepseek_v4_ssd.manifest import InstalledModel, NGram, QWEN_EXPERT_REGIONS, Tensor
from deepseek_v4_ssd.qwen4_exp import (
    ModelArgs,
    NGramStore,
    PLELayer,
    QSAAttention,
    RMSNormGated,
    SparseMoE,
    ngram_ids,
    qsa_causal_block_mask,
)
from deepseek_v4_ssd.tool_codec import (
    QwenToolCodec,
    QwenToolStreamParser,
    ToolCall,
    ToolChoice,
)


class FakeTokenizer:
    def __init__(self):
        self.arguments = None

    def apply_chat_template(self, messages, **arguments):
        self.arguments = (messages, arguments)
        return "qwen prompt"


class QwenTests(unittest.TestCase):
    descriptor = NGram(
        "ngram.bin",
        "F8_E4M3",
        4,
        2,
        8,
        tuple(range(0, 160, 10)),
        tuple(range(11, 27)),
    )

    def test_ngram_hash_keeps_cross_chunk_context(self):
        tokens = np.array([[4, 5, 6, 7]], dtype=np.int64)
        multipliers = np.array([1, 3, 5], dtype=np.int64)
        complete = ngram_ids(
            np.concatenate([np.array([[99, 99]]), tokens], axis=1),
            multipliers,
            self.descriptor,
            eos_token_id=99,
        )[:, -4:]
        first = ngram_ids(
            np.array([[99, 99, 4, 5]]),
            multipliers,
            self.descriptor,
            eos_token_id=99,
        )[:, -2:]
        second = ngram_ids(
            np.array([[4, 5, 6, 7]]),
            multipliers,
            self.descriptor,
            eos_token_id=99,
        )[:, -2:]

        np.testing.assert_array_equal(np.concatenate([first, second], axis=1), complete)

    def test_ngram_store_copies_only_requested_rows(self):
        descriptor = NGram(
            "ngram.bin", "F8_E4M3", 2, 2, 3, (0,) * 16, (3,) * 16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.array(
                [
                    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44,
                    0x48, 0x4C, 0xB8, 0xBC, 0xC0, 0xC4,
                ],
                dtype=np.uint8,
            ).tofile(path)
            store = NGramStore(path, descriptor, 0.5)

            result = store.lookup(np.array([[1, 4]]))
            mx.eval(result)

            np.testing.assert_array_equal(
                np.array(result.astype(mx.float32)),
                np.array([[[0.5, 0.75], [-0.5, -0.75]]], dtype=np.float32),
            )

    def test_ple_matches_across_prompt_chunks(self):
        descriptor = NGram(
            "ngram.bin", "F8_E4M3", 2, 1, 100, (0,) * 16, (100,) * 16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.resize(
                np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), 200
            ).tofile(path)
            args = ModelArgs(
                hidden_size=4,
                hc_count=2,
                hc_lowrank=2,
                ple_embed_dim=32,
                ple_conv_kernel_size=2,
                ngram_size=3,
                heads_per_ngram=8,
                eos_token_id=99,
            )
            layer = PLELayer(args, NGramStore(path, descriptor))
            tokens = mx.array([[1, 2, 3, 4]])
            hidden = mx.arange(32, dtype=mx.float32).reshape(1, 4, 8) / 100
            complete = layer(hidden, tokens, ArraysCache(size=4))
            cache = ArraysCache(size=4)
            chunked = mx.concatenate(
                [
                    layer(hidden[:, :2], tokens[:, :2], cache),
                    layer(hidden[:, 2:], tokens[:, 2:], cache),
                ],
                axis=1,
            )
            mx.eval(complete, chunked)

            np.testing.assert_allclose(
                np.array(chunked), np.array(complete), rtol=0, atol=1e-5
            )

    def test_qsa_causal_mask_selects_only_complete_blocks(self):
        np.testing.assert_array_equal(
            qsa_causal_block_mask(np.array([2, 3, 7]), 3),
            np.array(
                [
                    [False, False, False],
                    [True, False, False],
                    [True, True, False],
                ]
            ),
        )

    def test_gated_delta_norm_uses_sigmoid(self):
        norm = RMSNormGated(2, 1e-6)
        value = mx.array([[[3.0, 4.0]]])
        gate = mx.array([[[1.0, -1.0]]])
        result = norm(value, gate)
        mx.eval(result)

        gate_array = np.array(gate)
        expected = np.array(value / np.sqrt(12.5 + 1e-6)) * (
            1 / (1 + np.exp(-gate_array))
        )
        np.testing.assert_allclose(np.array(result), expected, rtol=0, atol=1e-6)

    def test_qsa_chunked_prefill_matches_one_pass(self):
        mx.random.seed(0)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=4,
            indexer_compress_ratio=2,
            partial_rotary_factor=0.5,
            max_position_embeddings=64,
        )
        hidden = mx.arange(96, dtype=mx.float32).reshape(1, 6, 16) / 100
        attention = QSAAttention(args)
        complete = attention(hidden, None)
        cache = CacheList(KVCache(), KVCache())
        chunked = mx.concatenate(
            [attention(hidden[:, :3], cache), attention(hidden[:, 3:], cache)],
            axis=1,
        )
        mx.eval(complete, chunked)

        np.testing.assert_allclose(
            np.array(chunked), np.array(complete), rtol=0, atol=1e-6
        )

    def test_qsa_decode_uses_both_caches(self):
        mx.random.seed(0)
        args = ModelArgs(
            hidden_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=4,
            indexer_compress_ratio=2,
            partial_rotary_factor=0.5,
            max_position_embeddings=64,
        )
        hidden = mx.arange(96, dtype=mx.float32).reshape(1, 6, 16) / 100
        attention = QSAAttention(args)
        complete = attention(hidden, None)[:, -1:]
        cache = CacheList(KVCache(), KVCache())
        _ = attention(hidden[:, :5], cache)
        decoded = attention(hidden[:, 5:], cache)
        mx.eval(complete, decoded)

        np.testing.assert_allclose(
            np.array(decoded), np.array(complete), rtol=0, atol=3e-4
        )
        self.assertEqual(cache[0].offset, 6)
        self.assertEqual(cache[1].offset, 6)

    def test_qwen_codec_and_stream_parser_produce_same_call(self):
        tokenizer = FakeTokenizer()
        codec = QwenToolCodec(tokenizer)
        prompt = codec.encode(
            [{"role": "user", "content": "Weather?"}],
            "thinking",
            [{"type": "function", "function": {"name": "weather"}}],
            ToolChoice("required"),
            "medium",
        )
        self.assertEqual(prompt, "qwen prompt")
        self.assertTrue(tokenizer.arguments[1]["enable_thinking"])
        self.assertEqual(tokenizer.arguments[1]["reasoning_effort"], "medium")

        raw = (
            "plan</think>Summary\n\n<tool_call>\n<function=weather>\n"
            "<parameter=city>\nTaipei\n</parameter>\n"
            "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
        )
        turn = codec.parse(raw, "thinking")
        parser = QwenToolStreamParser("thinking")
        deltas = [delta for character in raw for delta in parser.feed(character)]
        deltas.extend(parser.finish())
        call = ToolCall(
            next(delta.tool_name for delta in deltas if delta.tool_name),
            "".join(delta.arguments for delta in deltas if delta.tool_index == 0),
        )

        self.assertEqual(turn.reasoning_content, "plan")
        self.assertEqual(turn.content, "Summary")
        self.assertEqual(turn.tool_calls, (ToolCall("weather", '{"city":"Taipei","days":2}'),))
        self.assertTrue(parser.matches((call,)))
        self.assertTrue(parser.matches(turn.tool_calls))

    def test_qwen_codec_adds_forced_function_instruction(self):
        tokenizer = FakeTokenizer()
        QwenToolCodec(tokenizer).encode(
            [{"role": "user", "content": "Weather?"}],
            "chat",
            [{"type": "function", "function": {"name": "weather"}}],
            ToolChoice("function", "weather"),
        )

        messages = tokenizer.arguments[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn('Call the "weather" tool', messages[0]["content"])

    def test_qwen_codec_rejects_malformed_xml(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            QwenToolCodec(FakeTokenizer()).parse(
                "<tool_call><function=weather>", "chat"
            )

    def test_qwen_expert_layout_loads_fused_gate_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experts = root / "experts"
            experts.mkdir()
            (experts / "layer_00.bin").write_bytes(bytes(2_611_200))
            regions = tuple(Tensor(*region) for region in QWEN_EXPERT_REGIONS)
            installed = InstalledModel(
                root=root,
                model_id="fixture",
                revision="fixture",
                layer_count=1,
                expert_count=1,
                selected_expert_count=1,
                expert_blob_size=2_611_200,
                common_tensors=(Tensor("fixture", "U8", (1,), 0, 1),),
                expert_regions=regions,
                model_kind="qwen3.8-flash-next",
                format_version=2,
            )
            with ExpertCache(installed, slots=1, read_workers=1) as cache:
                weights = cache.get_many(0, [0]).individual_weights[0]
                mx.eval(weights.gate_up, weights.down)

                self.assertEqual(weights.gate_up.shape, (1_280, 320))
                self.assertEqual(weights.gate_up_scales.shape, (1_280, 80))
                self.assertEqual(weights.down.shape, (2_560, 80))
                self.assertEqual(weights.down_scales.shape, (2_560, 20))

    def test_qwen_layer_major_prefill_reads_each_expert_layer_once(self):
        class Layer:
            layer_type = "full_attention"

            def __init__(self):
                self.calls = 0

            def __call__(self, hidden, _tokens, _mask, _cache):
                self.calls += 1
                return hidden + 1

        class Cache:
            def __init__(self):
                self.layers = []

            @contextmanager
            def batched_layer(self, layer):
                self.layers.append(layer)
                yield None

        layers = [Layer(), Layer()]
        core = SimpleNamespace(
            args=SimpleNamespace(hidden_size=1, hc_count=2),
            layers=layers,
            embed_tokens=lambda tokens: tokens[..., None].astype(mx.float32),
        )
        cache = Cache()

        _qwen_layer_major_prefill(
            SimpleNamespace(model=core), [1, 2, 3, 4, 5], [None, None], 2, cache
        )

        self.assertEqual(cache.layers, [0, 1])
        self.assertEqual([layer.calls for layer in layers], [3, 3])

    def test_short_qwen_prefill_does_not_load_a_complete_expert_layer(self):
        class Cache:
            def __init__(self):
                self.batched_layers = []

            def current_batched(self, _layer):
                return None

            @contextmanager
            def batched_layer(self, layer):
                self.batched_layers.append(layer)
                yield None

        cache = Cache()
        args = ModelArgs(
            hidden_size=4,
            num_experts=2,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=2,
        )
        moe = SparseMoE(args, 0, cache)
        moe.experts = lambda value, indices: mx.zeros((*indices.shape, value.shape[-1]))

        result = moe(mx.ones((1, 5, args.hidden_size)))
        mx.eval(result)

        self.assertEqual(cache.batched_layers, [])


if __name__ == "__main__":
    unittest.main()
