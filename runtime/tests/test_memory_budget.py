from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.memory_budget import resolve_cache_budgets
from deepseek_v4_ssd.model_manager import validate_runtime_config, ModelSpec, ModelDefaults
from deepseek_v4_ssd.server import _catalog_overrides


class MemoryBudgetTests(unittest.TestCase):
    def installed(self, blob=2_611_200, selected=10):
        return SimpleNamespace(expert_blob_size=blob, selected_expert_count=selected,
                               dspark_block_size=5)

    def test_legacy_settings_remain_unchanged(self):
        config = RuntimeConfig(slots=900, mtp_slots=100, dspark_slots=300)
        self.assertIs(resolve_cache_budgets(self.installed(), config), config)

    def test_uses_actual_model_blob_size_and_never_rounds_up(self):
        for blob in (2_611_200, 13_369_344, 18_800_640):
            budget = 8 * 1024**3
            result = resolve_cache_budgets(self.installed(blob), RuntimeConfig(expert_cache_bytes=budget))
            self.assertLessEqual(result.slots * blob, budget)
            self.assertGreater((result.slots + 1) * blob, budget)

    def test_auxiliary_budgets_are_independent_and_only_active_when_enabled(self):
        blob = self.installed().expert_blob_size
        config = RuntimeConfig(expert_cache_bytes=100 * blob, mtp_cache_bytes=20 * blob,
                               dspark_cache_bytes=60 * blob)
        off = resolve_cache_budgets(self.installed(), config)
        self.assertEqual((off.slots, off.mtp_slots, off.dspark_slots), (100, 32, 768))
        on = resolve_cache_budgets(self.installed(), replace(config, mtp_enabled=True, dspark_enabled=True))
        self.assertEqual((on.slots, on.mtp_slots, on.dspark_slots), (100, 20, 60))
        self.assertEqual(resolve_cache_budgets(self.installed(), on), on)

    def test_model_runtime_passes_resolved_capacity_to_loader(self):
        from deepseek_v4_ssd.generation import ModelRuntime
        installed = self.installed(selected=6)
        budget = 100 * installed.expert_blob_size + 1
        with patch('deepseek_v4_ssd.generation.load_model', side_effect=RuntimeError('stop before loading')) as loader:
            with self.assertRaisesRegex(RuntimeError, 'stop before loading'):
                ModelRuntime(installed, RuntimeConfig(expert_cache_bytes=budget))
        self.assertEqual(loader.call_args.args[1].slots, 100)
        self.assertEqual(loader.call_args.args[1].expert_cache_bytes, budget)

    def test_explicit_slot_flag_overrides_catalog_memory_budget(self):
        spec = ModelSpec('qwen3.8-flash-next-fp8', None, '/tmp/model', 'qwen3.8-flash-next',
                         RuntimeConfig(expert_cache_bytes=8 * 1024**3, mtp_cache_bytes=512 * 1024**2),
                         ModelDefaults(8192, 0.7, 0.8, 20))
        args = SimpleNamespace(prompt_cache=None)
        inherited = _catalog_overrides([spec], args, RuntimeConfig(), set())[0]
        self.assertEqual(inherited.runtime.expert_cache_bytes, 8 * 1024**3)
        changed = _catalog_overrides([spec], args, RuntimeConfig(slots=900), {'slots'})[0]
        self.assertEqual(changed.runtime.slots, 900)
        self.assertIsNone(changed.runtime.expert_cache_bytes)
        self.assertEqual(changed.runtime.mtp_cache_bytes, 512 * 1024**2)

    def test_invalid_or_insufficient_budgets_fail_before_model_loading(self):
        for value in (0, -1, True, 1.5, float('nan'), 2**51):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_config(RuntimeConfig(expert_cache_bytes=value))
        blob = self.installed().expert_blob_size
        with self.assertRaisesRegex(ValueError, 'at least'):
            resolve_cache_budgets(self.installed(), RuntimeConfig(expert_cache_bytes=10 * blob - 1))
        exact = resolve_cache_budgets(self.installed(), RuntimeConfig(expert_cache_bytes=10 * blob))
        self.assertEqual(exact.slots, 10)


if __name__ == '__main__':
    unittest.main()
