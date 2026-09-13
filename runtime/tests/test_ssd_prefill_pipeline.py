"""Failure/lifetime and numerical gates for the isolated research pipeline."""
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import StreamingExperts
from research.ssd_prefill_pipeline import ExpertBanks, plan_waves, preadv_exact, qwen_pipeline


def fixture(root, count=128):
    specs = [('gate_up.weight','U32',(256,16)), ('gate_up.scale','U8',(256,4)),
             ('down.weight','U32',(128,16)), ('down.scale','U8',(128,4))]
    regions, offset = [], 0
    for name,dtype,shape in specs:
        length = int(np.prod(shape)) * (4 if dtype=='U32' else 1)
        regions.append(Tensor(name,dtype,shape,offset,length))
        offset += length
    model = InstalledModel(root,'fixture','fixture',1,count,4,offset,(),tuple(regions),model_kind='qwen3.8-flash-next')
    (root/'experts').mkdir()
    rng = np.random.default_rng(19)
    with (root/'experts/layer_00.bin').open('wb') as f:
        for expert in range(count):
            # Finite packed FP4 data, nonzero scale, distinct bytes per expert.
            for region in regions:
                f.write(rng.integers(0,256,region.length,dtype=np.uint8).tobytes()
                        if region.name.endswith('.weight') else bytes([117+expert%3])*region.length)
    return model


class PipelineTests(unittest.TestCase):
    def test_scoped_dispatch_restores_hooks_after_exception_and_falls_back_without_banks(self):
        from deepseek_v4_ssd import qwen4_exp
        from deepseek_v4_ssd.model_support import qwen as qwen_support
        from research.ssd_prefill_pipeline_run import install_pipeline
        original = qwen4_exp.StreamingExperts.__call__
        original_prefill = qwen_support._qwen_layer_major_prefill
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with install_pipeline('pipeline'):
                self.assertIsNot(qwen4_exp.StreamingExperts.__call__, original)
                raise RuntimeError('stop')
        self.assertIs(qwen4_exp.StreamingExperts.__call__, original)
        self.assertIs(qwen_support._qwen_layer_major_prefill, original_prefill)
        with tempfile.TemporaryDirectory() as d:
            model = fixture(Path(d))
            cache = SimpleNamespace(model=model, file_cache_policy='cached', _read_limiter=None,
                expert_directory=model.root / 'experts')
            with patch.object(qwen_support, '_qwen_layer_major_prefill', return_value='original') as fallback, \
                 patch('research.ssd_prefill_pipeline_run.ExpertBanks', side_effect=AssertionError('allocated')):
                with install_pipeline('pipeline') as state:
                    self.assertEqual(qwen_support._qwen_layer_major_prefill(None,[0]*2048,None,1024,cache),'original')
                    cache.file_cache_policy='bypass'
                    self.assertEqual(qwen_support._qwen_layer_major_prefill(None,[0]*2,None,1024,cache),'original')
                    self.assertTrue(all(not e['eligible'] for e in state['events']))
                self.assertEqual(fallback.call_count,2)

    def test_plan_preserves_duplicates_order_and_sparse_tail(self):
        indices=np.array([[[8,1,8],[3,9,1]]],np.int32)
        waves=plan_waves(indices,10,2)
        seen=[]
        for ids,pos,local in waves:
            self.assertLessEqual(len(ids),2)
            self.assertEqual(np.asarray(ids)[local].tolist(),indices.reshape(-1)[pos].tolist())
            seen.extend(pos.tolist())
        self.assertEqual(sorted(seen),list(range(indices.size)))
        self.assertEqual([w[0] for w in waves],[(1,3),(8,9)])
        self.assertEqual(plan_waves(np.array([],np.int32),10,2),[])
        for invalid in [np.array([10]),np.array([-1]),np.array([1.1])]:
            with self.assertRaises(ValueError): plan_waves(invalid,10,2)

    def test_partial_reads_eintr_and_eof(self):
        with tempfile.TemporaryFile() as f:
            f.write(b'abcdefgh');f.flush()
            a,b=bytearray(3),bytearray(5)
            real=os.preadv
            calls=[0]
            def short(fd,views,offset):
                calls[0]+=1
                if calls[0]==1: raise InterruptedError()
                return real(fd,[views[0][:2]],offset)
            with patch('research.ssd_prefill_pipeline.os.preadv',short):
                self.assertEqual(preadv_exact(f.fileno(),[memoryview(a),memoryview(b)],0),8)
            self.assertEqual(a+b,b'abcdefgh')
            with self.assertRaises(EOFError): preadv_exact(f.fileno(),[memoryview(bytearray(9))],0)

    def test_budget_rejects_before_allocation_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with patch('research.ssd_prefill_pipeline.mx.empty',side_effect=AssertionError('allocated')):
                with self.assertRaises(ValueError): ExpertBanks(model,4,7*model.expert_blob_size)
            banks=ExpertBanks(model,4,8*model.expert_blob_size)
            banks.close();banks.close()
            with self.assertRaises(RuntimeError): banks.run(0,[],lambda *a:None)

    def test_two_banks_overlap_and_reuse_only_after_consumer_fence(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            second_read=threading.Event()
            def reader(fd,views,offset):
                if offset==4*model.expert_blob_size: second_read.set()
                return preadv_exact(fd,views,offset)
            with ExpertBanks(model,4,8*model.expert_blob_size,reader) as banks:
                waves=plan_waves(np.arange(13,dtype=np.int32),model.expert_count,4)
                def consume(weights,pos,local):
                    self.assertTrue(second_read.wait(5),'read-ahead never submitted')
                    return weights.gate_up.sum()
                banks.run(0,waves,consume)
                self.assertEqual(len(banks.buffers),2)
                for i,row in enumerate(banks.rows):
                    self.assertEqual(row['bytes_read'],len(waves[i][0])*model.expert_blob_size)
                    if i>=2:
                        self.assertGreaterEqual(row['read_submit'],banks.rows[i-2]['consumer_fence_complete'])

    def test_failure_and_cancellation_drain_then_allow_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            waves=plan_waves(np.arange(12,dtype=np.int32),model.expert_count,4)
            with ExpertBanks(model,4,8*model.expert_blob_size) as banks:
                def fail(*a): raise RuntimeError('consumer failed')
                with self.assertRaisesRegex(RuntimeError,'consumer failed'): banks.run(0,waves,fail)
                calls=[0]
                def cancel():
                    calls[0]+=1
                    if calls[0]==3: raise RuntimeError('cancelled')
                with self.assertRaisesRegex(RuntimeError,'cancelled'): banks.run(0,waves,lambda *a:None,cancel)
                seen=[]
                banks.run(0,waves,lambda _,p,l:seen.extend(p.tolist()))
                self.assertEqual(sorted(seen),list(range(12)))

    def test_read_failure_is_not_admitted_as_ready(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            (model.root/'experts/layer_00.bin').write_bytes(b'')
            waves=plan_waves(np.array([0,5]),128,4)
            with ExpertBanks(model,4,8*model.expert_blob_size) as banks:
                with self.assertRaises(EOFError): banks.run(0,waves,lambda *a:self.fail('consumed failed read'))

    def test_recursive_consumer_rejected_and_trace_storage_stays_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            waves=plan_waves(np.arange(12,dtype=np.int32),128,4)
            with ExpertBanks(model,4,model.expert_count*model.expert_blob_size) as banks:
                def consume(*a):
                    with self.assertRaisesRegex(RuntimeError,'concurrent'):
                        banks.run(0,waves,lambda *a:None)
                for _ in range(3):
                    banks.run(0,waves,consume)
                    self.assertEqual(len(banks.rows),len(waves))

    def test_workspace_and_hidden_guard_reject_before_routed_output_allocation(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with ExpertBanks(model,4,model.expert_count*model.expert_blob_size) as banks:
                value=mx.zeros((1,1024,128));indices=mx.zeros((1,1024,4),mx.int32)
                wrong=mx.zeros((1,1024,64))
                mx.eval(value,indices,wrong)
                with patch('research.ssd_prefill_pipeline.mx.zeros',side_effect=AssertionError('output allocated')):
                    with self.assertRaisesRegex(ValueError,'workspace'):
                        qwen_pipeline(value,indices,banks,0)
                    with self.assertRaisesRegex(ValueError,'hidden'):
                        qwen_pipeline(wrong,indices,banks,0)

    def test_qmm_output_and_weighted_router_order_exact_across_bank_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            rng=np.random.default_rng(221)
            with ExpertBanks(model,4,model.expert_count*model.expert_blob_size) as banks, \
                 ExpertCache(model,slots=4,read_workers=2) as cache:
                experts=StreamingExperts(0,cache);experts.grouped_prefill=True
                for n in [1,7,32,128]:
                    with self.subTest(tokens=n):
                        value=mx.array(rng.normal(0,.2,(1,n,128)),mx.bfloat16)
                        indices=mx.array(rng.integers(0,128,(1,n,4)),mx.int32)
                        scores=mx.array(rng.uniform(0,1,(1,n,4)),mx.bfloat16)
                        with cache.batched_layer(0):
                            expected=experts(value,indices);mx.eval(expected)
                        actual=qwen_pipeline(value,indices,banks,0);mx.eval(actual)
                        self.assertTrue(mx.isfinite(actual).all().item())
                        self.assertTrue(mx.array_equal(actual,expected).item())
                        self.assertTrue(mx.array_equal((actual*scores[...,None]).sum(-2),(expected*scores[...,None]).sum(-2)).item())
                empty=qwen_pipeline(mx.zeros((1,0,128)),mx.zeros((1,0,4),mx.int32),banks,0)
                self.assertEqual(empty.shape,(1,0,4,128))


if __name__=='__main__': unittest.main()
