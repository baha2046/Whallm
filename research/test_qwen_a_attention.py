"""Boundary and causality checks for the research-only direct selected kernel."""
import unittest
from types import SimpleNamespace

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from qwen_a_qsa_candidate import build_direct


class DirectAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = staticmethod(build_direct())

    def arrays(self, length, context, offset):
        mx.random.seed(1729 + context)
        args = qwen.ModelArgs(num_attention_heads=4,num_key_value_heads=2,head_dim=32,
            indexer_n_heads=2,indexer_head_dim=16)
        norm = qwen.GroupRMSNorm(16,None,1e-6)
        proxy=SimpleNamespace(args=args,indexer=SimpleNamespace(k_layernorm=norm))
        inputs=dict(query=mx.random.normal((1,4,length,32)).astype(mx.bfloat16),
            key=mx.random.normal((1,2,context,32)).astype(mx.bfloat16),
            value=mx.random.normal((1,2,context,32)).astype(mx.bfloat16),
            index_query=mx.random.normal((1,length,2,16)).astype(mx.bfloat16),
            raw_index_keys=mx.random.normal((1,context,16)).astype(mx.bfloat16),offset=offset)
        return proxy,inputs

    def assert_matches(self, proxy, inputs):
        expected=qwen.QSAAttention._bounded_attention(proxy,**inputs)
        actual=self.candidate(proxy,**inputs)
        error=mx.abs(expected.astype(mx.float32)-actual.astype(mx.float32))
        self.assertTrue(bool(mx.isfinite(actual).all().item()))
        self.assertTrue(bool((error<=0.03125+0.01*mx.abs(expected.astype(mx.float32))).all().item()))
        return actual

    def test_dense_and_sparse_budget_boundaries(self):
        for context in (1,3,4,17,2047,2048,2049,2052):
            for length in (1,min(context,7)):
                with self.subTest(context=context,length=length):
                    self.assert_matches(*self.arrays(length,context,context-length))

    def test_future_values_are_not_visible(self):
        proxy,inputs=self.arrays(7,2052,0)
        first=self.assert_matches(proxy,inputs)
        changed=dict(inputs)
        changed["value"]=mx.concatenate([inputs["value"][:,:,:7],
            mx.full((1,2,2045,32),100,dtype=mx.bfloat16)],axis=2)
        second=self.assert_matches(proxy,changed)
        self.assertTrue(bool(mx.array_equal(first,second).item()))


if __name__ == "__main__":
    unittest.main()
