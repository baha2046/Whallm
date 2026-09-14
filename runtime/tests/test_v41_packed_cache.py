import unittest
import mlx.core as mx
from deepseek_v4_ssd.deepseek_v41.packed_cache import PackedRows
from deepseek_v4_ssd.deepseek_v41.fakequant import (
    fake_quant_fp4_e4m3, fake_quant_fp4_ue8m0, fake_quant_fp8_ue8m0)
from runtime.tests.test_v41_prompt_cache import tiny_model
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model_support.state import _encode_cache_state, _decode_cache_state


class PackedCacheTests(unittest.TestCase):
    def test_native_formats_round_trip_match_qat_and_use_less_memory(self):
        mx.random.seed(941)
        value = mx.random.normal((1, 23, 64)) * 2.3
        for kind, quant in (('fp4_e4m3', fake_quant_fp4_e4m3),
                            ('fp4_ue8m0', fake_quant_fp4_ue8m0),
                            ('fp8_ue8m0', fake_quant_fp8_ue8m0)):
            cache = PackedRows(value.shape, kind)
            cache.write((slice(None), slice(None)), value)
            expected = quant(value).astype(mx.bfloat16)
            self.assertTrue(mx.array_equal(cache[:], expected).item(), kind)
            self.assertLess(cache.nbytes, expected.nbytes)
            grown = cache.grow(40)
            self.assertTrue(mx.array_equal(grown[:, :23], expected).item())
            restored = PackedRows(grown.shape, kind).restore(grown.state())
            self.assertTrue(mx.array_equal(restored[:, :23], expected).item())

    def test_model_continuation_clone_and_serialization_preserve_native_values(self):
        model = tiny_model()
        support = get_support('deepseek-v4.1')
        for kv, index in ((True, False), (False, True), (True, True)):
            model.packed_kv = model.packed_index = False
            reference = model.make_cache()
            model.packed_kv, model.packed_index = kv, index
            candidate = model.make_cache()
            for tokens in ([1, 2, 3], [4, 5], [6, 7, 8, 9, 10]):
                a = model(mx.array([tokens]), reference)
                b = model(mx.array([tokens]), candidate)
                self.assertTrue(mx.array_equal(a, b).item(), (kv, index, tokens))
            saved = support.snapshot_cache(candidate)
            arrays = {}
            encoded = _encode_cache_state(saved, arrays)
            restored = model.make_cache()
            support.restore_cache(restored, _decode_cache_state(encoded, arrays))
            clone = support.clone_cache(candidate)
            for token in (11, 12, 13):
                a = model(mx.array([[token]]), reference)
                for c in (candidate, clone, restored):
                    self.assertTrue(mx.array_equal(a, model(mx.array([[token]]), c)).item())
            self.assertLess(candidate[0].nbytes, reference[0].nbytes)
