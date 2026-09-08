import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import RuntimeConfig


def fixture(root, layers=1):
    regions = tuple(Tensor(f'w{i}.{kind}', dtype, shape, (i-1)*5+offset, length)
                    for i in (1, 2, 3)
                    for kind, dtype, shape, offset, length in
                    [('weight','I8',(1,4),0,4), ('scale','F8_E8M0',(1,1),4,1)])
    (root/'experts').mkdir()
    for layer in range(layers):
        (root/'experts'/f'layer_{layer:02d}.bin').write_bytes(bytes(range(60)))
    return InstalledModel(root,'fixture','fixture',layers,4,1,15,(),regions)


class EvictionPolicyTests(unittest.TestCase):
    def test_legacy_catalog_defaults_and_explicit_policy_validation(self):
        from deepseek_v4_ssd.model_manager import _parse_runtime, ModelCatalogError
        legacy=asdict(RuntimeConfig())
        for key in ('expert_eviction_policy','qwen_grouped_decode','qwen_grouped_experts'):
            legacy.pop(key)
        self.assertEqual(_parse_runtime(legacy,'runtime','qwen3.8-flash-next').expert_eviction_policy,'lfu')
        self.assertEqual(_parse_runtime({**legacy,'expert_eviction_policy':'lru'},'runtime','qwen3.8-flash-next').expert_eviction_policy,'lru')
        for invalid in ('future',None,True,[]):
            with self.assertRaises(ModelCatalogError):
                _parse_runtime({**legacy,'expert_eviction_policy':invalid},'runtime','qwen3.8-flash-next')

    def test_default_lfu_and_recent_lru_choose_different_victims(self):
        self.assertEqual(RuntimeConfig().expert_eviction_policy, 'lfu')
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            for policy, retained in [('lfu',0),('lru',1)]:
                with ExpertCache(model, slots=2, eviction_policy=policy) as cache:
                    cache.get_many(0,[0]*10)
                    cache.get_many(0,[1])
                    cache.get_many(0,[2])
                    self.assertEqual(set(cache._entries), {(0,retained),(0,2)})
                    self.assertEqual(cache.metrics.bytes_read,45)
            with self.assertRaisesRegex(ValueError, 'eviction policy'):
                ExpertCache(model, eviction_policy='future')

    def test_state_fork_preserves_bits_and_owns_distinct_storage(self):
        import ctypes
        import numpy as np
        from types import SimpleNamespace
        from deepseek_v4_ssd.model import _clone_layer_cache
        for dtype, storage, bits in [
            (mx.bfloat16,mx.uint16,[0,0x8000,1,0x8001,0x7fc1]),
            (mx.float16,mx.uint16,[0,0x8000,1,0x8001,0x7e01]),
            (mx.float32,mx.uint32,[0,0x80000000,1,0x80000001,0x7fc00001]),
        ]:
            value=mx.array(bits,storage).view(dtype).reshape(1,1,1,-1)
            mx.eval(value)
            original=SimpleNamespace(keys=value,values=value)
            clone,copied=_clone_layer_cache(original);mx.eval(*copied)
            before=np.asarray(value.view(storage)).copy()
            self.assertTrue(np.array_equal(np.asarray(clone.keys.view(storage)),before))
            def address(array):return ctypes.addressof(ctypes.c_ubyte.from_buffer(array))
            self.assertNotEqual(address(value),address(clone.keys))
            self.assertNotEqual(address(clone.values),address(clone.keys))
            # Simulate an in-place consumer write after the GPU fence.
            ctypes.memset(address(clone.keys),0,clone.keys.nbytes)
            self.assertTrue(np.array_equal(np.asarray(original.keys.view(storage)),before))

    def test_lru_protects_requested_and_speculatively_pinned_keys(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with ExpertCache(model, slots=3, eviction_policy='lru') as cache:
                cache.get_many(0,[0,1,2])
                cache._speculative_pinned_keys.add((0,0))
                cache.get_many(0,[1,3])
                self.assertEqual(set(cache._entries), {(0,0),(0,1),(0,3)})
                cache._speculative_pinned_keys.clear()
                cache.get_many(0,[2])
                self.assertNotIn((0,0),cache._entries)

    def test_layer_reserve_survives_lru_and_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d),layers=2)
            with ExpertCache(model, slots=4, eviction_policy='lru') as cache:
                cache.get_many(0,[0])
                cache.get_many(1,[0,1,2])
                for _ in range(80):
                    cache.get_many(1,[1])
                self.assertGreater(cache._last_decay,0)
                self.assertLess(len(cache._heap), max(cache.slots*8,64)+cache.slots)
                cache.get_many(1,[3])
                self.assertIn((0,0),cache._entries)
                self.assertIn((1,1),cache._entries)
                self.assertNotIn((1,0),cache._entries)

    def test_read_failure_releases_slots_and_can_retry_in_lru(self):
        with tempfile.TemporaryDirectory() as d:
            model=fixture(Path(d))
            with ExpertCache(model, slots=2, eviction_policy='lru') as cache:
                cache.get_many(0,[0,1])
                with patch.object(cache,'_read_expert_into_slot',side_effect=OSError('injected')):
                    with self.assertRaisesRegex(OSError,'injected'):
                        cache.get_many(0,[2])
                self.assertNotIn((0,2),cache._entries)
                cache.get_many(0,[2])
                self.assertEqual(cache.resident_count,2)
                self.assertEqual(len(set(e.slot for e in cache._entries.values())),2)

    def test_frequency_preserving_native_block_reads_each_expert_once(self):
        from runtime.tests.test_qwen_resident_block import fixture as native_fixture
        from research.qwen_resident_block import BoundedArenaPool, resident_experts
        with tempfile.TemporaryDirectory() as d:
            model=native_fixture(Path(d))
            outputs=[]
            for preserve in (False,True):
                with ExpertCache(model,slots=4) as cache:
                    cache._pool=BoundedArenaPool(model,4)
                    values=mx.ones((1,2,2560),mx.bfloat16)
                    routes=mx.array([[[0,1],[0,2]]],mx.int32)
                    output=resident_experts(values,routes,cache,0,preserve_frequency=preserve)
                    mx.eval(output);outputs.append(output)
                    self.assertEqual(cache.metrics.bytes_read,3*model.expert_blob_size)
                    self.assertEqual(cache._entries[(0,0)].frequency,2 if preserve else 1)
                    self.assertEqual(cache.metrics.routed_expert_assignments,4 if preserve else 3)
            self.assertTrue(mx.array_equal(*outputs).item())


if __name__=='__main__':unittest.main()
