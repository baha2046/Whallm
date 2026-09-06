"""Slot ownership, page order and per-request phase tests for grouped Decode."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.expert_cache import ExpertCache, _QwenArenaSlotPool
from deepseek_v4_ssd.generation import _prompt_cache_contract, _qwen_decode_request
from deepseek_v4_ssd.manifest import InstalledModel, QWEN_EXPERT_REGIONS, Tensor
from deepseek_v4_ssd.model import RuntimeConfig, load_model
from deepseek_v4_ssd.qwen4_exp import StreamingExperts


def fixture(root):
    regions = tuple(Tensor(*region) for region in QWEN_EXPERT_REGIONS)
    model = InstalledModel(
        root=root, model_id="fixture", revision="fixture", layer_count=1,
        expert_count=4, selected_expert_count=2, expert_blob_size=2_611_200,
        common_tensors=(Tensor("fixture", "U8", (1,), 0, 1),),
        expert_regions=regions, model_kind="qwen3.8-flash-next", format_version=2,
    )
    rng = np.random.default_rng(381)
    blobs = []
    for _ in range(4):
        blob = bytearray(model.expert_blob_size)
        for region in regions:
            payload = (rng.integers(0, 256, region.length, dtype=np.uint8).tobytes()
                       if region.name.endswith("weight") else bytes([118]) * region.length)
            blob[region.offset:region.offset + region.length] = payload
        blobs.append(bytes(blob))
    (root / "experts").mkdir()
    (root / "experts/layer_00.bin").write_bytes(b"".join(blobs))
    return model


class QwenGroupedDecodeTests(unittest.TestCase):
    def test_growth_plan_bounds_unused_reservation_and_maps_every_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            common_bytes = 9 * 1024**3
            model = replace(model, common_tensors=(Tensor("resident", "U8", (9, 1024**3), 0, common_bytes),))
            pool = _QwenArenaSlotPool(model, 4096)
            self.assertEqual(sum(pool.page_counts), 4096)
            for page, (start, count) in enumerate(zip(pool.page_starts, pool.page_counts)):
                self.assertEqual(pool._location(start), (page, 0))
                self.assertEqual(pool._location(start + count - 1), (page, count - 1))
                self.assertLessEqual(count, 1024)
                if page:
                    self.assertLessEqual(count * model.expert_blob_size,
                                         .15 * (common_bytes + start * model.expert_blob_size))
            self.assertFalse(pool.arenas)

    def test_pages_eviction_order_duplicates_and_exact_output(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            # Force two pages in a small fixture, including partial final page.
            with patch.object(_QwenArenaSlotPool, "PAGE_SLOTS", 2), \
                 ExpertCache(model, slots=3, read_workers=1, qwen_grouped_decode=True) as cache:
                experts = StreamingExperts(0, cache)
                value = mx.array(np.random.default_rng(1).normal(size=(1, 1, 2560)), dtype=mx.bfloat16)
                with cache.qwen_decode_request():
                    cache.set_qwen_decode_prefill(1)
                    with cache.qwen_decode_step(1):
                        self.assertFalse(cache.qwen_decode_active)
                    # Request 3 experts to cross pages, then evict and revisit.
                    for selected in ([2, 0, 1], [3, 2, 3], [0, 1, 2]):
                        indices = mx.array([[selected]], dtype=mx.uint32)
                        reference = experts(value, indices)
                        mx.eval(reference)
                        with cache.qwen_decode_step(1):
                            actual = experts(value, indices)
                            mx.eval(actual)
                        self.assertTrue(mx.isfinite(actual).all().item())
                        self.assertTrue(mx.array_equal(actual, reference).item())
                    self.assertEqual(len(cache._pool.arenas), 2)
                    self.assertEqual(cache._pool.arenas[1].shape[0], 1)
                    self.assertGreater(cache.metrics.evictions, 0)
                    self.assertGreater(cache.metrics.gather_qmm_calls, 0)
                    for slot, array in enumerate(cache._pool._slots):
                        page, index = divmod(slot, 2)
                        self.assertEqual(np.asarray(array).ctypes.data,
                                         np.asarray(cache._pool.arenas[page]).ctypes.data + index * model.expert_blob_size)

    def test_one_token_prefill_chunk_and_request_reset_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            with ExpertCache(model, slots=2, read_workers=1, qwen_grouped_decode=True) as cache:
                for count, chunks in ((130, [128, 1, 1]), (1, [1]), (3, [2, 1])):
                    with self.assertRaisesRegex(ValueError, "cancelled"):
                        with cache.qwen_decode_request():
                            cache.set_qwen_decode_prefill(count)
                            for chunk in chunks:
                                with cache.qwen_decode_step(chunk):
                                    self.assertFalse(cache.qwen_decode_active)
                            with cache.qwen_decode_step(1):
                                self.assertTrue(cache.qwen_decode_active)
                                raise ValueError("cancelled")
                    self.assertFalse(cache.qwen_decode_active)
                    self.assertFalse(cache._qwen_decode_request)
                    self.assertIsNone(cache._qwen_prefill_remaining)
                with cache.qwen_decode_step(1):
                    self.assertFalse(cache.qwen_decode_active)

    def test_runtime_request_drains_generation_stream_on_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            model = fixture(Path(directory))
            with ExpertCache(model, slots=2, read_workers=1, qwen_grouped_decode=True) as cache:
                runtime = SimpleNamespace(config=RuntimeConfig(qwen_grouped_decode=True),
                                          expert_cache=cache, _generation_stream=object())
                with patch("deepseek_v4_ssd.generation.mx.synchronize") as sync:
                    with self.assertRaisesRegex(ValueError, "cancelled"):
                        with _qwen_decode_request(runtime):
                            raise ValueError("cancelled")
                    sync.assert_called_once_with(runtime._generation_stream)
                self.assertFalse(cache._qwen_decode_request)

    def test_configuration_rejects_unsupported_models_and_mtp_before_loading(self):
        self.assertFalse(RuntimeConfig().qwen_grouped_decode)
        for is_qwen, mtp in ((False, False), (True, True)):
            with self.assertRaisesRegex(ValueError, "requires Qwen with MTP disabled"):
                load_model(SimpleNamespace(is_qwen=is_qwen),
                           RuntimeConfig(qwen_grouped_decode=True, mtp_enabled=mtp))

    def test_experimental_prompt_cache_is_isolated_without_changing_legacy_contract(self):
        model = SimpleNamespace(root="/missing", revision="fixture", model_id="fixture")
        baseline = _prompt_cache_contract(model, RuntimeConfig())
        candidate = _prompt_cache_contract(model, RuntimeConfig(qwen_grouped_decode=True))
        self.assertNotIn("qwenGroupedDecode", baseline)
        self.assertTrue(candidate.pop("qwenGroupedDecode"))
        self.assertEqual(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
