from Scripts.archived_evidence import enable_archived_research
enable_archived_research()
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mlx.core as mx
import numpy as np
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import StreamingExperts
from research.ssd_prefill_pipeline import qwen_pipeline
from research.ssd_prefill_retained import RetainedBanks
from runtime.tests.test_ssd_prefill_pipeline import fixture


class RetainedTests(unittest.TestCase):
    def test_chunks_reuse_exact_weights_and_layer_restart_reads_again(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d));budget=model.expert_count*model.expert_blob_size
            rng=np.random.default_rng(932)
            with ExpertCache(model,slots=64) as cache, RetainedBanks(model,32,budget) as banks:
                expected_experts=StreamingExperts(0,cache);seen=set()
                for iteration in range(2):
                    with banks.layer(0):
                        seen.clear()
                        for offset in (0,16,8):
                            routes=np.arange(offset,offset+48,dtype=np.int32).reshape(1,12,4)
                            value=mx.array(rng.normal(0,.2,(1,12,128)),mx.bfloat16)
                            # Prefill's actual reference is the batched-layer path.
                            with cache.batched_layer(0):
                                expected=expected_experts(value,mx.array(routes));mx.eval(expected)
                            actual=qwen_pipeline(value,mx.array(routes),banks,0);mx.eval(actual)
                            self.assertTrue(mx.array_equal(expected,actual).item())
                            wanted=set(routes.reshape(-1).tolist())
                            self.assertEqual(sum(r['bytes_read'] for r in banks.rows),len(wanted-seen)*model.expert_blob_size)
                            seen|=wanted
                            self.assertLessEqual(len(banks.pages)*32,model.expert_count)

    def test_consumer_error_drains_then_next_layer_scope_recovers(self):
        from research.ssd_prefill_pipeline import plan_waves
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d));waves=plan_waves(np.arange(64),128,32)
            with RetainedBanks(model,32,128*model.expert_blob_size) as banks:
                with self.assertRaisesRegex(RuntimeError,'consumer'):
                    with banks.layer(0):banks.run(0,waves,lambda *a:(_ for _ in ()).throw(RuntimeError('consumer')))
                self.assertFalse(banks._running.locked());self.assertIsNone(banks.fd)
                with banks.layer(0):banks.run(0,waves,lambda *a:None)
                self.assertEqual(sum(r['bytes_read'] for r in banks.rows),64*model.expert_blob_size)

    def test_cancel_before_submit_allocates_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with RetainedBanks(model,32,128*model.expert_blob_size) as banks:
                with self.assertRaisesRegex(RuntimeError,'cancel'):
                    with banks.layer(0):banks.run(0,[],lambda *a:None,cancelled=lambda:(_ for _ in ()).throw(RuntimeError('cancel')))
                self.assertEqual(banks.pages,{})
                self.assertIsNone(banks.fd)


if __name__=='__main__':unittest.main()
