from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd.dspark import generate_tokens
from deepseek_v4_ssd.expert_cache import CacheMetrics
from deepseek_v4_ssd.generation import (
    ModelRuntime,
    RuntimeMetrics,
    _DSparkPromptCacheEntry,
    _PromptCacheEntry,
)


class _FixtureCache:
    def __init__(self, value: int = 0):
        self._state = (mx.array([value], dtype=mx.int32),)
        self.meta_state = ""

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


class _CloseableCache:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FixtureDSpark:
    target_layers = (40, 41, 42)
    layers = (object(), object(), object())

    def __init__(self):
        self.expert_cache = _CloseableCache()
        self.reset_count = 0

    def reset_cache(self):
        self.reset_count += 1


def _context_state(seed: int):
    return tuple(
        ((mx.array([seed + layer], dtype=mx.int32),), seed + layer)
        for layer in range(3)
    )


def _runtime(directory: Path | None = None, *, entries: int = 2):
    runtime = object.__new__(ModelRuntime)
    dspark = _FixtureDSpark()
    runtime.installed = SimpleNamespace(revision="fixture-revision")
    runtime.config = SimpleNamespace(
        prompt_cache_entries=entries,
        prompt_cache_memory_gib=1,
        persistent_prompt_cache_entries=8,
    )
    runtime.metrics = RuntimeMetrics()
    runtime.model = SimpleNamespace(dspark=dspark)
    runtime.expert_cache = _CloseableCache()
    runtime._prompt_caches = []
    runtime._persistent_prompt_caches = []
    runtime._dspark_prompt_caches = []
    runtime._persistent_dspark_prompt_caches = []
    runtime._prompt_cache_directory = directory
    return runtime, dspark


def _entry(value: int = 7, tokens: list[int] | None = None):
    return _DSparkPromptCacheEntry(
        [_FixtureCache(value)],
        _context_state(value * 10),
        list(tokens or [1, 2]),
        "fixture-revision",
        (40, 41, 42),
    )


class DSparkPromptCacheTests(unittest.TestCase):
    def test_metrics_expose_last_reuse_source(self):
        metrics = RuntimeMetrics()
        metrics.start(
            3,
            2,
            1,
            False,
            CacheMetrics(),
            dspark_enabled=True,
            dspark_prompt_cache_source="memory",
        )

        self.assertEqual(
            metrics.snapshot()["dspark_prompt_cache_source"],
            "memory",
        )

    def test_memory_hit_returns_independent_atomic_clone(self):
        runtime, dspark = _runtime()
        stored = _entry()
        runtime._store_dspark_prompt_cache(stored)

        acquired, source = runtime._acquire_dspark_prompt_cache([1, 2, 3], dspark)
        acquired.cache[0].state = (mx.array([99], dtype=mx.int32),)
        acquired.context_state = _context_state(900)

        self.assertEqual(source, "memory")
        self.assertIsNot(acquired.cache, stored.cache)
        self.assertEqual(stored.cache[0].state[0].tolist(), [7])
        self.assertEqual(stored.context_state[0][0][0].tolist(), [70])

    def test_memory_contract_rejects_prefix_revision_layers_and_missing_context(self):
        runtime, dspark = _runtime()
        cases = [
            (_entry(tokens=[9, 2]), [1, 2, 3]),
            (
                _DSparkPromptCacheEntry(
                    [_FixtureCache()],
                    _context_state(1),
                    [1, 2],
                    "wrong-revision",
                    (40, 41, 42),
                ),
                [1, 2, 3],
            ),
            (
                _DSparkPromptCacheEntry(
                    [_FixtureCache()],
                    _context_state(1),
                    [1, 2],
                    "fixture-revision",
                    (39, 40, 41),
                ),
                [1, 2, 3],
            ),
            (
                _DSparkPromptCacheEntry(
                    [_FixtureCache()],
                    (None, *_context_state(1)[1:]),
                    [1, 2],
                    "fixture-revision",
                    (40, 41, 42),
                ),
                [1, 2, 3],
            ),
        ]
        with patch(
            "deepseek_v4_ssd.generation.make_prompt_cache",
            side_effect=lambda _model: [_FixtureCache()],
        ):
            for candidate, prompt in cases:
                runtime._dspark_prompt_caches = [candidate]
                acquired, source = runtime._acquire_dspark_prompt_cache(prompt, dspark)
                self.assertEqual(source, "none")
                self.assertEqual(acquired.tokens, [])

    def test_normal_and_dspark_persistent_namespaces_do_not_cross_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime, dspark = _runtime(directory)
            runtime._persist_prompt_cache(
                _PromptCacheEntry([_FixtureCache(3)], [1, 2])
            )
            runtime._persist_dspark_prompt_cache(_entry())

            normal = runtime._scan_persistent_prompt_caches()
            atomic = runtime._scan_persistent_dspark_prompt_caches(dspark)

        self.assertEqual(len(normal), 1)
        self.assertEqual(normal[0].format, 4)
        self.assertEqual(len(atomic), 1)
        self.assertEqual(atomic[0].format, 3)

    def test_persistent_atomic_bundle_survives_runtime_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, _ = _runtime(directory)
            first._persist_dspark_prompt_cache(_entry())
            metadata_path = next(directory.glob("*.dspark.v3.json"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            second, dspark = _runtime(directory)
            second._persistent_dspark_prompt_caches = (
                second._scan_persistent_dspark_prompt_caches(dspark)
            )
            with patch(
                "deepseek_v4_ssd.generation.make_prompt_cache",
                side_effect=lambda _model: [_FixtureCache()],
            ):
                acquired, source = second._acquire_dspark_prompt_cache(
                    [1, 2, 3], dspark
                )

        self.assertEqual(source, "persistent")
        self.assertEqual(acquired.cache[0].state[0].tolist(), [7])
        self.assertEqual(acquired.context_state[2][0][0].tolist(), [72])
        self.assertEqual(metadata["mode"], "dspark")
        self.assertEqual(metadata["revision"], "fixture-revision")
        self.assertEqual(metadata["targetLayers"], [40, 41, 42])
        self.assertEqual(metadata["tokens"], [1, 2])

    def test_malformed_partial_and_wrong_contract_bundles_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime, dspark = _runtime(directory)
            (directory / "malformed.json").write_text("{", encoding="utf-8")
            partial = {
                "format": 3,
                "mode": "dspark",
                "revision": "fixture-revision",
                "targetLayers": [40, 41, 42],
                "tokens": [1, 2],
                "data": "missing.safetensors",
            }
            (directory / "partial.json").write_text(
                json.dumps(partial), encoding="utf-8"
            )
            for name, key, value in (
                ("wrong-revision", "revision", "other"),
                ("wrong-layers", "targetLayers", [39, 40, 41]),
                ("wrong-mode", "mode", "normal"),
            ):
                metadata = dict(partial)
                metadata["data"] = f"{name}.safetensors"
                metadata[key] = value
                (directory / metadata["data"]).write_bytes(b"invalid")
                (directory / f"{name}.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
            invalid_state = dict(partial)
            invalid_state["data"] = "invalid-state.safetensors"
            (directory / invalid_state["data"]).write_bytes(b"invalid")
            (directory / "invalid-state.json").write_text(
                json.dumps(invalid_state), encoding="utf-8"
            )

            descriptors = runtime._scan_persistent_dspark_prompt_caches(dspark)
            runtime._persistent_dspark_prompt_caches = descriptors
            with patch(
                "deepseek_v4_ssd.generation.make_prompt_cache",
                side_effect=lambda _model: [_FixtureCache()],
            ):
                acquired, source = runtime._acquire_dspark_prompt_cache(
                    [1, 2, 3], dspark
                )

        self.assertEqual(len(descriptors), 1)
        self.assertEqual(source, "none")
        self.assertEqual(acquired.tokens, [])
        self.assertEqual(runtime._dspark_prompt_caches, [])

    def test_eviction_and_close_release_complete_atomic_entries(self):
        runtime, dspark = _runtime(entries=1)
        runtime._store_dspark_prompt_cache(_entry(1, [1]))
        runtime._store_dspark_prompt_cache(_entry(2, [2]))

        self.assertEqual(len(runtime._dspark_prompt_caches), 1)
        self.assertEqual(runtime._dspark_prompt_caches[0].tokens, [2])
        self.assertEqual(runtime._dspark_prompt_caches[0].cache[0].state[0].tolist(), [2])
        self.assertEqual(
            runtime._dspark_prompt_caches[0].context_state[0][0][0].tolist(),
            [20],
        )

        runtime.close()

        self.assertEqual(runtime._dspark_prompt_caches, [])
        self.assertEqual(dspark.reset_count, 1)
        self.assertTrue(dspark.expert_cache.closed)
        self.assertTrue(runtime.expert_cache.closed)

    def test_resumed_prefill_skips_reset_and_snapshots_only_aligned_suffix(self):
        forwarded = []
        prefills = []
        snapshots = []

        class DSpark:
            target_layers = (0,)

            def __init__(self):
                self.reset_count = 0

            def reset_cache(self):
                self.reset_count += 1

            def prefill_context(self, hidden, offset):
                prefills.append((hidden.shape[1], offset))

            def cache_state(self):
                return ((mx.array([3]),),)

        dspark = DSpark()

        def forward(_model, inputs, _cache, _layers):
            forwarded.append(inputs.reshape(-1).tolist())
            logits = mx.zeros((1, inputs.shape[1], 5))
            logits[..., 1] = 1
            return logits, mx.ones((1, inputs.shape[1], 1))

        with patch("deepseek_v4_ssd.model.forward_with_hidden", side_effect=forward):
            outputs = list(
                generate_tokens(
                    [0, 1, 2, 3],
                    object(),
                    dspark,
                    [object()],
                    max_tokens=1,
                    prefill_step_size=8,
                    temperature=0,
                    top_p=1,
                    prefilled_tokens=2,
                    record_prefill_snapshot=lambda *values: snapshots.append(values),
                )
            )

        self.assertEqual(dspark.reset_count, 0)
        self.assertEqual(forwarded, [[2], [3]])
        self.assertEqual(prefills, [(1, 2), (1, 3)])
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0][0], 3)
        self.assertEqual(outputs[0][0], 1)


if __name__ == "__main__":
    unittest.main()
