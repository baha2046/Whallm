"""Real layout bytes, phase isolation and reader resource failures."""
import mmap
import os
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import (
    InstalledModel, Tensor, EXPERT_REGIONS, DEEPSEEK_V41_EXPERT_REGIONS, QWEN_EXPERT_REGIONS,
)
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.prefill_io import PrefillReader


class PrefillIOTests(unittest.TestCase):
    def test_three_native_layouts_batched_and_demand_reads_are_byte_exact(self):
        for kind, specs in (("deepseek-v4", EXPERT_REGIONS),
                            ("deepseek-v4.1", DEEPSEEK_V41_EXPERT_REGIONS),
                            ("qwen3.8-flash-next", QWEN_EXPERT_REGIONS)):
            with self.subTest(model=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'experts').mkdir()
                regions = tuple(Tensor(*r) for r in specs)
                size = sum(r.length for r in regions)
                data = bytes(range(251)) * ((3 * size + 250) // 251)
                data = data[:3 * size]
                (root / 'experts/layer_00.bin').write_bytes(data)
                model = InstalledModel(root, 'fixture', 'fixture', 1, 3, 1,
                                       size, (), regions, model_kind=kind)
                with ExpertCache(model, slots=3, separate_prefill_io=True,
                                 prefetch_read_workers=1) as cache:
                    self.assertNotEqual(cache._descriptors, cache._prefill_reader.descriptors)
                    with cache.batched_layer(0, experts=[0, 2]):
                        packed = memoryview(cache._active_prefetch_trace.job.packed).cast('B')
                        try:
                            for expert in (0, 2):
                                parts = cache._pool.write_views(packed, expert)
                                self.assertEqual(b''.join(map(bytes, parts)), data[expert*size:(expert+1)*size])
                                for part in parts: part.release()
                        finally:
                            packed.release()
                    with cache.trace_routes('prefill'):
                        cache.get_many(0, [1])
                    stats = cache.prefill_io_snapshot()
                    self.assertEqual(stats['direct_bytes'] + stats['copied_bytes'], 3 * size)
                    cache.get_many(0, [2])
                    self.assertEqual(cache.prefill_io_snapshot(), stats)
                    for expert in (1, 2):
                        slot = cache._entries[(0, expert)].slot
                        parts = [cache._pool.writable_region(slot, r.name) for r in regions]
                        self.assertEqual(b''.join(map(bytes, parts)), data[expert*size:(expert+1)*size])
                        for part in parts: part.release()
                    self.assertLessEqual(stats['staging_buffers'], cache.read_workers)
                    if kind == 'deepseek-v4': self.assertGreater(stats['direct_bytes'], 0)
                    else: self.assertGreater(stats['copied_bytes'], 0)
                    descriptors = list(cache._prefill_reader.descriptors)
                for fd in descriptors:
                    with self.assertRaises(OSError): os.fstat(fd)

    def test_direct_short_read_can_finish_through_aligned_copy_and_eof_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, mmap.mmap(-1, 2*mmap.PAGESIZE) as buffer:
            root = Path(directory)
            data = bytes(range(251)) * 300
            (root/'layer_00.bin').write_bytes(data)
            reader = PrefillReader(root, 1, 2*mmap.PAGESIZE)
            view = memoryview(buffer)
            try:
                with patch.object(reader, '_preadv', wraps=reader._preadv) as read:
                    self.assertEqual(reader.read(0, [view], 0), len(view))
                    self.assertEqual(bytes(view), data[:len(view)])
                    self.assertEqual(read.call_count, 1)
                tail = memoryview(bytearray(81))
                try:
                    reader.read(0, [tail], len(data)-81)
                    self.assertEqual(bytes(tail), data[-81:])
                    with self.assertRaises(EOFError): reader.read(0, [tail], len(data)-40)
                finally: tail.release()
                original = reader._preadv
                with patch.object(reader, '_preadv', side_effect=lambda fd, parts, pos:
                                  os.preadv(fd, [parts[0][:17]], pos)):
                    self.assertEqual(reader.read(0, [view], 0), 17)
                part = view[17:]
                try: reader.read(0, [part], 17)
                finally: part.release()
                self.assertEqual(bytes(view), data[:len(view)])
            finally:
                view.release()
                reader.close()

    def test_unaligned_short_read_cancel_and_open_failure_release_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'layer_00.bin').write_bytes(bytes(3*mmap.PAGESIZE))
            reader = PrefillReader(root, 1, mmap.PAGESIZE)
            part = memoryview(bytearray(100))
            try:
                with patch.object(reader, '_preadv', return_value=17):
                    with self.assertRaisesRegex(OSError, 'Unaligned short'): reader.read(0, [part], 1)
                event = threading.Event(); event.set()
                with cancellation_scope(event):
                    with self.assertRaises(GenerationCancelled): reader.read(0, [part], 1)
            finally:
                part.release(); reader.close()
            opened = []
            original = os.open
            def tracked(*args):
                fd = original(*args); opened.append(fd); return fd
            with patch('deepseek_v4_ssd.prefill_io.os.open', side_effect=tracked):
                with self.assertRaises(FileNotFoundError): PrefillReader(root, 2, 100)
            for fd in opened:
                with self.assertRaises(OSError): os.fstat(fd)

    def test_catalog_migrates_disabled_retired_settings_and_validates_new_option(self):
        from deepseek_v4_ssd.model_manager import _parse_runtime, ModelCatalogError
        config = asdict(RuntimeConfig())
        config.pop('separate_prefill_io')
        config.update(dspark_hash_prefetch=False, dspark_adaptive_block=False,
                      dspark_hybrid_verification=False, adaptive_expert_prefill_threshold=None)
        for kind in ('deepseek-v4', 'deepseek-v4.1', 'qwen3.8-flash-next'):
            self.assertTrue(_parse_runtime(config, 'runtime', kind).separate_prefill_io)
            self.assertFalse(_parse_runtime({**config, 'separate_prefill_io':False}, 'runtime', kind).separate_prefill_io)
        for key, value in (('dspark_hash_prefetch', True), ('adaptive_expert_prefill_threshold', .8),
                           ('separate_prefill_io', 'true')):
            with self.assertRaises(ModelCatalogError):
                _parse_runtime({**config, key:value}, 'runtime', 'deepseek-v4')
