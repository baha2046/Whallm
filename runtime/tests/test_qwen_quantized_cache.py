import copy
import unittest
import mlx.core as mx
from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
from deepseek_v4_ssd.model import _cache_arrays
from mlx_lm.models.cache import CacheList


class QSAQuantizedCacheTests(unittest.TestCase):
    def test_storage_restoration_trim_and_continuation(self):
        mx.random.seed(974)
        value = mx.random.normal((1, 2, 256, 128)).astype(mx.bfloat16)
        for bits in (4, 8):
            cache = QSAQuantizedCache(bits, 128)
            result, _ = cache.update_and_fetch(value, value)
            expected = mx.dequantize(*mx.quantize(value, group_size=32, bits=bits), group_size=32, bits=bits)
            self.assertTrue(mx.array_equal(result, expected).item())
            self.assertLess(cache.nbytes, 2 * value.nbytes)
            self.assertEqual(len(_cache_arrays([CacheList(cache)])), 6)
            restored = QSAQuantizedCache(bits, 128)
            restored.restore_persistence_state(cache.persistence_state())
            clone = copy.deepcopy(cache)
            for other in (cache, restored, clone):
                self.assertEqual(other.trim(3), 3)
            append = mx.ones((1, 2, 3, 128), mx.bfloat16)
            reference, _ = cache.update_and_fetch(append, append)
            for other in (restored, clone):
                actual, _ = other.update_and_fetch(append, append)
                self.assertTrue(mx.array_equal(actual, reference).item())
            with self.assertRaises(ValueError):
                QSAQuantizedCache(12 - bits, 128).restore_persistence_state(cache.persistence_state())

    def test_attention_continues_after_packed_cache_round_trip(self):
        from deepseek_v4_ssd.qwen4_exp import ModelArgs, QSAAttention
        from mlx_lm.models.cache import KVCache
        args = ModelArgs(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=32, indexer_n_heads=2, indexer_kv_heads=1,
                         indexer_head_dim=32, indexer_budget=4, indexer_compress_ratio=2,
                         partial_rotary_factor=0.5, max_position_embeddings=64)
        attention = QSAAttention(args)
        cache = CacheList(QSAQuantizedCache(8, 32), QSAQuantizedCache(4, 32))
        hidden = mx.random.normal((1, 12, 32))
        attention(hidden[:, :7], cache)
        clone = copy.deepcopy(cache)
        actual = attention(hidden[:, 7:], cache)
        restored = attention(hidden[:, 7:], clone)
        self.assertTrue(mx.array_equal(actual, restored).item())
        self.assertTrue(mx.all(mx.isfinite(actual)).item())
        self.assertEqual(cache[0].offset, 12)
        self.assertEqual(cache[1].offset, 12)
