import unittest
from unittest.mock import patch
from types import SimpleNamespace
import mlx.core as mx
from deepseek_v4_ssd.deepseek_v41.config import ModelArgs
from deepseek_v4_ssd.deepseek_v41.indexer import Indexer


class CandidateIndexTests(unittest.TestCase):
    def test_later_indexer_scores_only_source_candidates(self):
        args = ModelArgs(dim=32, q_lora_rank=32, n_layers=3, head_dim=32,
            rope_head_dim=4, index_n_heads=2, index_head_dim=32,
            compress_ratios=(1, 1, 1), kv_source_layers=(0,),
            index_source_layers=(0, 2), candidate_source_layer=0,
            candidate_topk_blocks=2, candidate_block_size=4, index_topk=4)
        mx.random.seed(319)
        source, later = Indexer(args, 0), Indexer(args, 2)
        x = mx.random.normal((1, 3, 32))
        keys = mx.random.normal((1, 64, 32))
        cos, sin = mx.ones((67, 2)), mx.zeros((67, 2))
        shared = SimpleNamespace(candidates=None, candidate_indices=None)
        source(x, x, 61, 5, cos, sin, keys, shared)
        reference = later(x, x, 61, 5, cos, sin, keys, shared)
        mx.eval(reference)
        later.candidate_only = True
        with patch.object(mx, 'einsum', wraps=mx.einsum) as dot:
            actual = later(x, x, 61, 5, cos, sin, keys, shared)
            mx.eval(actual)
        self.assertTrue(mx.array_equal(actual, reference).item())
        self.assertEqual(dot.call_args.args[2].shape, (1, 3, 8, 32))
        # No source candidates: the ordinary indexer remains available.
        shared.candidate_indices = None
        actual = later(x, x, 61, 5, cos, sin, keys, shared)
        self.assertTrue(mx.array_equal(actual, reference).item())
