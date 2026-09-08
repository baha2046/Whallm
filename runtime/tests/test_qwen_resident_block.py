"""Native canonical-shape correctness and bounded allocation tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import mlx.core as mx
import numpy as np
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.qwen4_exp import StreamingExperts
from research.qwen_resident_block import BoundedArenaPool, direct_project, install_resident_block


def fixture(root):
    specs = [('gate_up.weight','U32',(1280,320)),('gate_up.scale','U8',(1280,80)),
             ('down.weight','U32',(2560,80)),('down.scale','U8',(2560,20))]
    regions, offset = [],0
    for name,dtype,shape in specs:
        length = int(np.prod(shape)) * (4 if dtype=='U32' else 1)
        regions.append(Tensor(name,dtype,shape,offset,length)); offset += length
    model = InstalledModel(root,'fixture','fixture',1,12,4,offset,(),tuple(regions),model_kind='qwen3.8-flash-next')
    (root/'experts').mkdir()
    rng = np.random.default_rng(813)
    with (root/'experts/layer_00.bin').open('wb') as f:
        for expert in range(12):
            for region in regions:
                f.write(rng.integers(0,256,region.length,dtype=np.uint8).tobytes()
                    if region.name.endswith('.weight') else bytes([117+expert%3])*region.length)
    return model


class ResidentBlockTests(unittest.TestCase):
    def test_native_cross_page_duplicates_and_reordered_rows_exact(self):
        with tempfile.TemporaryDirectory() as d:
            model = fixture(Path(d))
            class SmallPages(BoundedArenaPool):
                PAGE_SLOTS = 4
            with ExpertCache(model,slots=12) as reference, ExpertCache(model,slots=12) as candidate:
                candidate._pool = SmallPages(model,12)
                # More than one page, repeat/reorder experts within and across tokens.
                rng = np.random.default_rng(942)
                for block in (2,4):
                    value = mx.array(rng.normal(0,.2,(1,block,2560)),mx.bfloat16)
                    route = np.array([[[7,0,7,9],[1,8,0,5],[9,5,8,7],[3,1,3,0]]],np.int32)[:,:block]
                    indices = mx.array(route)
                    expert = StreamingExperts(0,reference)
                    expected = mx.concatenate([expert(value[:,t:t+1],indices[:,t:t+1]) for t in range(block)],axis=1)
                    mx.eval(expected)
                    from research.qwen_resident_block import resident_experts
                    rows=[]
                    with patch.object(candidate,'get_many',wraps=candidate.get_many) as acquire:
                        actual = resident_experts(value,indices,candidate,0,rows)
                    self.assertEqual(acquire.call_count,1)
                    self.assertGreater(rows[0]['arena_count'],1)
                    self.assertEqual(rows[0]['packed_bytes'],0)
                    self.assertTrue(mx.isfinite(actual).all().item())
                    self.assertTrue(mx.array_equal(actual,expected).item())
                    for slot in (e.slot for e in candidate._entries.values()):
                        page,index = candidate._pool._location(slot)
                        self.assertEqual(np.asarray(candidate._pool._slots[slot]).ctypes.data,
                            np.asarray(candidate._pool.arenas[page]).ctypes.data+index*model.expert_blob_size)

    def test_sparse_reservation_rejected_before_allocation_and_hooks_restored(self):
        with tempfile.TemporaryDirectory() as d:
            model = fixture(Path(d))
            pool = BoundedArenaPool(model,1152)
            with self.assertRaisesRegex(ValueError,'reservation'):
                pool.prepare([0,128,256,384,512,640,768,896,1024])
            self.assertEqual(pool.arenas,{})
            with self.assertRaisesRegex(ValueError,'physical slot'):
                pool.prepare([-1])
            from research import qwen_short_block
            original_init,original_union=ExpertCache.__init__,qwen_short_block.union_experts
            with self.assertRaisesRegex(RuntimeError,'abort'):
                with install_resident_block():
                    raise RuntimeError('abort')
            self.assertIs(ExpertCache.__init__,original_init)
            self.assertIs(qwen_short_block.union_experts,original_union)

    def test_bad_native_input_fails_before_kernel(self):
        with self.assertRaisesRegex(ValueError,'BF16'):
            direct_project(mx.zeros((1,5,2560)),None,[],4)

    def test_native_reads_updated_slot_without_repacking(self):
        with tempfile.TemporaryDirectory() as d:
            model = fixture(Path(d))
            pool = BoundedArenaPool(model,4)
            blob = (model.root/'experts/layer_00.bin').read_bytes()[:model.expert_blob_size]
            pool.store([0],[blob])
            value = mx.ones((1,1,2560),mx.bfloat16)
            old = direct_project(value,pool,[0],1)
            mx.eval(old)
            self.assertTrue(mx.any(old != 0).item())
            pointer = np.asarray(pool.arenas[0]).ctypes.data
            # Fence above permits the same allocation to hold a different expert.
            pool.store([0],[bytes(model.expert_blob_size)])
            new = direct_project(value,pool,[0],1)
            mx.eval(new)
            self.assertTrue(mx.all(new == 0).item())
            self.assertEqual(np.asarray(pool.arenas[0]).ctypes.data,pointer)


if __name__ == '__main__': unittest.main()
