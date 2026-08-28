from __future__ import annotations

import threading
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import (
    ModelCatalogError,
    ModelDefaults,
    ModelLoadFailed,
    ModelManager,
    ModelSpec,
    parse_model_catalog,
)


def raw_model(
    model_kind: str,
    *,
    alias: str | None = None,
    path: str | None = None,
) -> dict:
    qwen = model_kind == "qwen3.8-flash-next"
    return {
        "id": (
            "qwen3.8-flash-next-fp8"
            if qwen
            else "deepseek-v4-flash-0731"
        ),
        "alias": alias,
        "path": path or f"/tmp/{model_kind}",
        "model_kind": model_kind,
        "runtime": asdict(RuntimeConfig()),
        "defaults": {
            "max_tokens": 272_000,
            "temperature": 1.0 if qwen else 0.2,
            "top_p": 0.95 if qwen else 0.98,
            "top_k": 20 if qwen else 0,
        },
        "warmup_prompt_path": None,
    }


def spec(model_kind: str, alias: str | None = None) -> ModelSpec:
    return parse_model_catalog(
        {"version": 1, "models": [raw_model(model_kind, alias=alias)]}
    )[0]


class FakeMetrics:
    def __init__(self):
        self.tokens = 0
        self.requests = 0

    def snapshot(self):
        return {
            "accumulated_generation_tokens": self.tokens,
            "completed_request_count": self.requests,
        }


class FakeRuntime:
    def __init__(self, model_id: str, events: list[str]):
        self.model_id = model_id
        self.events = events
        self.metrics = FakeMetrics()
        self.config = RuntimeConfig()
        self.installed = SimpleNamespace(
            root=f"/tmp/{model_id}",
            has_dspark=False,
        )
        self.expert_cache = SimpleNamespace(
            resident_count=0,
            metrics=SimpleNamespace(
                bytes_read=0,
                read_seconds=0,
                pack_seconds=0,
                eviction_seconds=0,
                routing_sync_seconds=0,
                hit_rate=0,
                hits=0,
                misses=0,
            ),
        )

    def close(self):
        self.events.append(f"close:{self.model_id}")


class ModelCatalogTests(unittest.TestCase):
    def test_empty_catalog_is_valid(self):
        self.assertEqual(parse_model_catalog({"version": 1, "models": []}), ())

    def test_aliases_are_trimmed_and_model_list_keeps_catalog_order(self):
        models = parse_model_catalog(
            {
                "version": 1,
                "models": [
                    raw_model("deepseek-v4", alias="  work-model  "),
                    raw_model("qwen3.8-flash-next", alias="chat-model"),
                ],
            }
        )
        manager = ModelManager(models, clear_cache=lambda: None)
        self.assertEqual(models[0].alias, "work-model")
        self.assertEqual(
            [item["id"] for item in manager.models()],
            [
                "deepseek-v4-flash-0731",
                "work-model",
                "qwen3.8-flash-next-fp8",
                "chat-model",
            ],
        )
        self.assertEqual(manager.models()[0]["owned_by"], manager.models()[1]["owned_by"])

    def test_empty_and_self_aliases_do_not_add_duplicate_models(self):
        empty = spec("deepseek-v4", "  ")
        same = spec("deepseek-v4", "deepseek-v4-flash-0731")
        self.assertIsNone(empty.alias)
        self.assertEqual(
            [item["id"] for item in ModelManager([same]).models()],
            ["deepseek-v4-flash-0731"],
        )

    def test_alias_cannot_conflict_with_another_id_or_alias(self):
        cases = (
            [
                raw_model("deepseek-v4", alias="qwen3.8-flash-next-fp8"),
                raw_model("qwen3.8-flash-next"),
            ],
            [
                raw_model("deepseek-v4", alias="shared"),
                raw_model("qwen3.8-flash-next", alias="shared"),
            ],
        )
        for models in cases:
            with self.subTest(models=models), self.assertRaises(ModelCatalogError):
                parse_model_catalog({"version": 1, "models": models})

    def test_merged_runtime_fields_are_validated(self):
        cases = (
            ("deepseek-v4", "dspark_prompt_cache", True),
            ("deepseek-v4", "dspark_fallback_enabled", False),
            ("deepseek-v4", "expert_file_cache_policy", "cold"),
            ("qwen3.8-flash-next", "staged_expert_streaming", True),
            ("qwen3.8-flash-next", "adaptive_expert_prefill_threshold", 0.8),
        )
        for model_kind, name, value in cases:
            model = raw_model(model_kind)
            model["runtime"][name] = value
            with self.subTest(name=name), self.assertRaises(ModelCatalogError):
                parse_model_catalog({"version": 1, "models": [model]})

    def test_layer_major_prefill_threshold_must_be_positive(self):
        model = raw_model("deepseek-v4")
        model["runtime"]["layer_major_prefill_threshold"] = 0
        with self.assertRaises(ModelCatalogError):
            parse_model_catalog({"version": 1, "models": [model]})


class ModelManagerTests(unittest.TestCase):
    def test_listing_and_status_do_not_load_a_runtime(self):
        loads = []
        manager = ModelManager(
            [spec("deepseek-v4")],
            runtime_loader=lambda model: loads.append(model.id),
            clear_cache=lambda: None,
        )
        self.assertEqual(manager.models()[0]["id"], "deepseek-v4-flash-0731")
        status = manager.status_snapshot()
        self.assertEqual(loads, [])
        self.assertIsNone(status["loaded_model"])
        self.assertIsNone(status["model"])
        self.assertIsNone(status["runtime"])
        self.assertEqual(status["performance"]["generation_tokens"], 0)
        self.assertEqual(status["performance"]["prompt_cache_write_error"], "")
        self.assertEqual(
            status["performance"]["dspark_last_verification_layer_seconds"],
            (),
        )

    def test_first_request_loads_once_and_alias_reuses_the_runtime(self):
        events = []
        loads = []

        def load(model):
            loads.append(model.id)
            return FakeRuntime(model.id, events)

        manager = ModelManager(
            [spec("deepseek-v4", "work-model")],
            runtime_loader=load,
            clear_cache=lambda: events.append("clear"),
        )
        with manager.request("work-model") as first:
            self.assertEqual(first.name, "work-model")
        with manager.request("deepseek-v4-flash-0731") as second:
            self.assertIs(first.runtime, second.runtime)
        self.assertEqual(loads, ["deepseek-v4-flash-0731"])

    def test_manual_load_and_unload_use_the_same_runtime(self):
        events = []
        manager = ModelManager(
            [spec("deepseek-v4", "work-model")],
            runtime_loader=lambda model: FakeRuntime(model.id, events),
            clear_cache=lambda: events.append("clear"),
        )

        manager.load("work-model")
        self.assertEqual(
            manager.status_snapshot()["loaded_model"],
            "deepseek-v4-flash-0731",
        )
        manager.unload("deepseek-v4-flash-0731")

        self.assertIsNone(manager.status_snapshot()["loaded_model"])
        self.assertEqual(events, ["close:deepseek-v4-flash-0731", "clear"])

    def test_switch_closes_and_clears_before_loading_the_next_runtime(self):
        events = []

        def load(model):
            events.append(f"load:{model.id}")
            return FakeRuntime(model.id, events)

        manager = ModelManager(
            [spec("deepseek-v4"), spec("qwen3.8-flash-next")],
            runtime_loader=load,
            clear_cache=lambda: events.append("clear"),
        )
        with manager.request("deepseek-v4-flash-0731"):
            pass
        with manager.request("qwen3.8-flash-next-fp8"):
            pass
        self.assertEqual(
            events,
            [
                "load:deepseek-v4-flash-0731",
                "close:deepseek-v4-flash-0731",
                "clear",
                "load:qwen3.8-flash-next-fp8",
            ],
        )

    def test_failed_load_can_be_retried(self):
        attempts = 0

        def load(model):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary failure")
            return FakeRuntime(model.id, [])

        manager = ModelManager(
            [spec("deepseek-v4")],
            runtime_loader=load,
            clear_cache=lambda: None,
        )
        with self.assertRaises(ModelLoadFailed):
            with manager.request("deepseek-v4-flash-0731"):
                pass
        self.assertIsNone(manager.status_snapshot()["loading_model"])
        with manager.request("deepseek-v4-flash-0731"):
            pass
        self.assertEqual(attempts, 2)

    def test_loading_status_is_visible_without_waiting_for_the_model(self):
        entered = threading.Event()
        release = threading.Event()

        def load(model):
            entered.set()
            release.wait(timeout=2)
            return FakeRuntime(model.id, [])

        manager = ModelManager(
            [spec("deepseek-v4")],
            runtime_loader=load,
            clear_cache=lambda: None,
        )
        def run_request():
            with manager.request("deepseek-v4-flash-0731"):
                pass

        request = threading.Thread(target=run_request)
        request.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            status = manager.status_snapshot()
            self.assertIsNone(status["loaded_model"])
            self.assertEqual(status["loading_model"], "deepseek-v4-flash-0731")
            self.assertEqual(status["performance"]["generation_tokens"], 0)
        finally:
            release.set()
            request.join(timeout=1)

    def test_cumulative_counts_continue_after_a_model_switch(self):
        manager = ModelManager(
            [spec("deepseek-v4"), spec("qwen3.8-flash-next")],
            runtime_loader=lambda model: FakeRuntime(model.id, []),
            clear_cache=lambda: None,
        )
        with manager.request("deepseek-v4-flash-0731") as request:
            request.runtime.metrics.tokens = 7
            request.runtime.metrics.requests = 1
        with manager.request("qwen3.8-flash-next-fp8"):
            pass

        performance = manager.status_snapshot()["performance"]
        self.assertEqual(performance["accumulated_generation_tokens"], 7)
        self.assertEqual(performance["completed_request_count"], 1)

    def test_requests_wait_until_the_streaming_scope_finishes(self):
        events = []

        def load(model):
            events.append(f"load:{model.id}")
            return FakeRuntime(model.id, events)

        manager = ModelManager(
            [spec("deepseek-v4"), spec("qwen3.8-flash-next")],
            runtime_loader=load,
            clear_cache=lambda: events.append("clear"),
        )
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def first_request():
            with manager.request("deepseek-v4-flash-0731"):
                first_entered.set()
                release_first.wait(timeout=2)

        def second_request():
            with manager.request("qwen3.8-flash-next-fp8"):
                second_entered.set()

        first = threading.Thread(target=first_request)
        second = threading.Thread(target=second_request)
        first.start()
        self.assertTrue(first_entered.wait(timeout=1))
        second.start()
        self.assertFalse(second_entered.wait(timeout=0.1))
        self.assertEqual(events, ["load:deepseek-v4-flash-0731"])

        status = manager.status_snapshot()
        self.assertEqual(status["loaded_model"], "deepseek-v4-flash-0731")

        release_first.set()
        self.assertTrue(second_entered.wait(timeout=1))
        first.join(timeout=1)
        second.join(timeout=1)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(
            events,
            [
                "load:deepseek-v4-flash-0731",
                "close:deepseek-v4-flash-0731",
                "clear",
                "load:qwen3.8-flash-next-fp8",
            ],
        )


if __name__ == "__main__":
    unittest.main()
