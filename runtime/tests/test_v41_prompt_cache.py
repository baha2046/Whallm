from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from deepseek_v4_ssd.deepseek_v41.config import ModelArgs
from deepseek_v4_ssd.deepseek_v41.model import Model
from deepseek_v4_ssd.deepseek_v41_ssd import DeepSeekV41ForCausalLM, DeepSeekV41PromptCache
from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model_support.state import _encode_cache_state, _decode_cache_state


def tiny_model():
    mx.random.seed(41)
    args = ModelArgs(vocab_size=16, dim=64, n_layers=4, moe_inter_dim=32,
                     n_heads=2, head_dim=32, rope_head_dim=4, q_lora_rank=32,
                     o_lora_rank=32, o_groups=2, window_size=4,
                     n_routed_experts=2, n_shared_experts=1, n_activated_experts=1,
                     compress_ratios=(2, 2, 1, 1), kv_source_layers=(0, 2),
                     index_source_layers=(0, 2), index_n_heads=2, index_head_dim=32,
                     index_topk=32, original_seq_len=32, max_seq_len=32, hc_mult=1,
                     engram_layer_ids=(1,), engram_num_embeddings=(59,),
                     engram_vocab_size=16, engram_n_heads=1, engram_head_dim=32,
                     engram_compressed_vocab_size=16)
    core = Model(args, token_map=list(range(16)))
    # Nonzero Engram weights make a lost token history affect continuation logits.
    core.layers[1].engram.embed.weight = mx.random.normal((59, 32)) * 0.1
    core.layers[1].engram.embed.scale = mx.ones((59, 1))
    mx.eval(core.parameters())
    return DeepSeekV41ForCausalLM(core)


class V41PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.support = get_support('deepseek-v4.1')

    def test_clone_and_disk_round_trip_continue_at_partial_group_and_ring_wrap(self):
        original = [DeepSeekV41PromptCache(self.model.model.make_cache(max_seq_len=4, dtype=mx.bfloat16))]
        prefix = mx.array([[1, 2, 3, 4, 5, 6, 7]])
        mx.eval(self.model(prefix, cache=original))
        cloned = self.support.clone_cache(original)
        state = self.support.snapshot_cache(original)
        arrays = {}
        schema = _encode_cache_state(state, arrays)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'cache.safetensors')
            mx.save_safetensors(path, arrays, {'state': json.dumps(schema)})
            saved, metadata = mx.load(path, return_metadata=True)
            restored = self.model.make_cache()
            self.support.restore_cache(restored, _decode_cache_state(json.loads(metadata['state']), saved))
        original_ids = original[0].cache.engram_ids.copy()
        for tokens in ([[8]], [[9, 10, 11]], [[12]]):
            expected = self.model(mx.array(tokens), cache=original)
            for branch in (cloned, restored):
                actual = self.model(mx.array(tokens), cache=branch)
                mx.eval(expected, actual)
                np.testing.assert_array_equal(np.array(actual), np.array(expected))
        # Independent branches must never overwrite the shared saved prefix.
        np.testing.assert_array_equal(np.array(state[0]['engram_ids']), original_ids)
        self.assertEqual(state[0]['offset'], 7)
        self.assertEqual(restored[0].offset, 12)
        self.assertEqual(restored[0].cache.max_seq_len, 16)
        self.assertGreater(restored[0].nbytes, sum(a.nbytes for a in restored[0].state))

    def test_history_is_required_and_branches_are_independent(self):
        prefix = self.model.make_cache()
        mx.eval(self.model(mx.array([[1, 2, 3, 4, 5, 6, 7]]), cache=prefix))
        for component in ('engram', 'compressor'):
            with self.subTest(component=component):
                correct = self.support.clone_cache(prefix)
                broken = self.support.clone_cache(prefix)
                if component == 'engram':
                    broken[0].cache.engram_ids[:] = 0
                else:
                    broken[0].cache.layers[0].comp_state.kv_state[:] = 0
                expected = self.model(mx.array([[8, 9, 10, 11, 12]]), cache=correct)
                actual = self.model(mx.array([[8, 9, 10, 11, 12]]), cache=broken)
                mx.eval(expected, actual)
                self.assertFalse(np.array_equal(np.array(expected), np.array(actual)))
                self.assertEqual(prefix[0].offset, 7)
                self.assertEqual(prefix[0].cache.engram_ids[0, 6], 7)

    def test_runtime_memory_disk_off_and_cancelled_branch(self):
        backend = Tokenizer(models.WordLevel({str(i): i for i in range(16)}, unk_token='0'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0')
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        options = GenerationOptions(max_tokens=3, temperature=0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(asdict(self.model.args)))
            installed = SimpleNamespace(root=root, model_kind='deepseek-v4.1',
                                        model_id='fixture/v41', revision='fixture',
                                        format_version=3, maximum_context=32)
            def runtime(disk=False, enabled=True):
                config = RuntimeConfig(prompt_cache_entries=2 if enabled else 0,
                                       persistent_prompt_cache=disk,
                                       prompt_cache_directory=root / 'cache', prefill_step_size=3)
                with patch('deepseek_v4_ssd.generation.load_model',
                           return_value=(self.model, SimpleNamespace(close=lambda: None))), \
                     patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained', return_value=tokenizer):
                    return ModelRuntime(installed, config)
            cold = runtime(enabled=False)
            expected = [p.token for p in cold.stream(prompt, options)]
            self.assertEqual(cold.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            self.assertFalse(cold._prompt_caches)
            cold.close()
            memory = runtime()
            self.assertEqual([p.token for p in memory.stream(prompt, options)], expected)
            self.assertEqual([p.token for p in memory.stream(prompt, options)], expected)
            self.assertGreater(memory.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            self.assertIsNone(memory._prompt_cache_directory)
            self.assertFalse((root / 'cache').exists())
            memory.close()
            warm = runtime(disk=True)
            self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
            self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
            self.assertGreater(warm.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            interrupted = warm.stream(prompt + [9], GenerationOptions(max_tokens=8, temperature=0))
            next(interrupted)
            interrupted.close()
            self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
            warm.close()
            reopened = runtime(disk=True)
            self.assertTrue(reopened._persistent_prompt_caches)
            self.assertEqual([p.token for p in reopened.stream(prompt, options)], expected)
            self.assertGreater(reopened.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            reopened.close()

    def test_rejects_invalid_state_without_changing_target(self):
        cache = self.model.make_cache()
        for field, value in (('offset', 33), ('capacity', 0), ('version', 2),
                             ('layers', []), ('engram_ids', None)):
            with self.subTest(field=field):
                state = self.support.snapshot_cache(cache)
                state[0][field] = value
                with self.assertRaises(ValueError):
                    self.support.restore_cache(cache, state)
                self.assertEqual(cache[0].offset, 0)


if __name__ == '__main__':
    unittest.main()
