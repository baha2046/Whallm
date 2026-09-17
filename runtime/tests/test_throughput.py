from __future__ import annotations

import gzip
import hashlib
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from test_server import FakeRuntime
from deepseek_v4_ssd.cancellation import check_cancelled
from deepseek_v4_ssd.generation import GeneratedPiece
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.server import OpenAIServer
from deepseek_v4_ssd.throughput import CONTEXT_LENGTHS, CONTEXT_TYPES, CORPUS_DIRECTORY, prompt_tokens


class BenchmarkRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.prompt = []
        self.entered = threading.Event()
        self.config = RuntimeConfig(slots=2304)
        self.installed = SimpleNamespace(maximum_context=262144, is_qwen=False)
        self.metrics = SimpleNamespace(snapshot=lambda: {
            "time_to_first_token_seconds": 0.5,
            "decode_tokens_per_second": 4.0,
            "prefill_tokens_per_second": len(self.prompt) / 0.5,
            "prompt_cache_reused_tokens": 0,
        })
        self.finished = threading.Event()
        self.slow = False
        self.fail = False

    def _encode_prompt(self, text):
        return list(text.encode())

    def stream(self, prompt, options):
        self.prompt = prompt
        self.entered.set()
        self.options = options
        self.benchmark_config = self.config
        try:
            if self.fail:
                raise ValueError("test generation failure")
            while self.slow:
                check_cancelled()
                time.sleep(0.01)
            yield GeneratedPiece("a", 12, len(prompt), 1, None)
            yield GeneratedPiece("b", 13, len(prompt), 2, "stop")
        finally:
            self.finished.set()


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.runtime = BenchmarkRuntime()
        self.original_config = self.runtime.config
        self.loads = 0
        def load(_):
            self.loads += 1
            return self.runtime
        self.manager = ModelManager([
            ModelSpec("deepseek-v4-flash-0731", None, "/tmp/model", "deepseek-v4",
                      RuntimeConfig(), ModelDefaults(272000, 0.2, 0.98, 0))
        ], runtime_loader=load, clear_cache=lambda: None)
        self.server = OpenAIServer(("127.0.0.1", 0), self.manager, api_key="secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/benchmark/throughput"

    def tearDown(self):
        self.runtime.slow = False
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()

    def request(self, **changes):
        payload = dict(model="deepseek-v4-flash-0731", context_length=1024, generation_length=128)
        payload.update(changes)
        return Request(self.url, data=json.dumps(payload).encode(), headers={
            "Authorization": "Bearer secret", "Content-Type": "application/json"})

    def test_both_assets_use_original_prefixes_and_match_manifest(self):
        manifest = json.loads((CORPUS_DIRECTORY / "manifest.json").read_text())
        total_bytes = 0
        for context in CONTEXT_TYPES:
            path = CORPUS_DIRECTORY / f"{context}.txt.gz"
            total_bytes += path.stat().st_size
            data = gzip.decompress(path.read_bytes())
            expected_hash = hashlib.sha256(data).hexdigest()
            self.assertEqual(expected_hash, manifest["contexts"][context]["sha256"])
            self.assertEqual(len(data), manifest["contexts"][context]["utf8_bytes"])
            original = self.runtime._encode_prompt(data.decode())
            for length in CONTEXT_LENGTHS:
                tokens, corpus_hash = prompt_tokens(self.runtime, length, context)
                self.assertEqual(len(tokens), length)
                self.assertEqual(tokens, original[:length])
                self.assertEqual(corpus_hash, expected_hash)
        self.assertLess(total_bytes, 750_000)

    def test_short_corpus_is_rejected_instead_of_repeated(self):
        with patch.object(self.runtime, "_encode_prompt", return_value=[1, 2, 3]):
            for context in CONTEXT_TYPES:
                with self.assertRaisesRegex(ValueError, "only 3 tokens.*1024 were requested"):
                    prompt_tokens(self.runtime, 1024, context)
        with self.assertRaisesRegex(ValueError, "Choose Code or Novel"):
            prompt_tokens(self.runtime, 1024, "../../invalid")

    def test_novel_selection_reaches_generation_and_result(self):
        with urlopen(self.request(benchmark_context="novel")) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        expected, expected_hash = prompt_tokens(self.runtime, 1024, "novel")
        self.assertEqual(self.runtime.prompt, expected)
        self.assertEqual(result["benchmark_context"], "novel")
        self.assertEqual(result["corpus_sha256"], expected_hash)

    def test_auto_load_and_request_scoped_results_restore_settings(self):
        with patch("deepseek_v4_ssd.app_memory.physical_footprint", return_value=123456), \
             patch("deepseek_v4_ssd.app_memory.app_process_ids", return_value=(11, 22)):
            with urlopen(self.request()) as response:
                body = response.read().decode()
        events = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertEqual(self.loads, 1)
        self.assertEqual(len(self.runtime.prompt), 1024)
        self.assertIsInstance(self.runtime.prompt, list)
        self.assertEqual(self.runtime.options.max_tokens, 128)
        self.assertEqual(self.runtime.benchmark_config.prompt_cache_entries, 0)
        self.assertFalse(self.runtime.benchmark_config.persistent_prompt_cache)
        self.assertFalse(self.runtime.benchmark_config.dspark_prompt_cache)
        self.assertIs(self.runtime.config, self.original_config)
        self.assertEqual(result["generation_tokens"], 2)  # EOS is not reported as 128.
        self.assertEqual(result["context_tokens"], 1024)
        self.assertEqual(result["benchmark_context"], "code")
        self.assertEqual(result["slots"], self.original_config.slots)
        self.assertEqual(result["corpus_sha256"], prompt_tokens(self.runtime, 1024)[1])
        self.assertEqual(result["ttft_ms"], 500)
        self.assertEqual(result["tpot_ms"], 250)
        self.assertEqual(result["peak_app_memory_bytes"], 246912)
        self.assertEqual(result["memory_scope"], "app")
        self.assertNotIn("peak_memory_bytes", result)
        self.assertEqual(result["output_token_sha256"], hashlib.sha256(b"12\n13\n").hexdigest())
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_memory_sampling_covers_loading_and_stops_after_result(self):
        from deepseek_v4_ssd.app_memory import AppMemorySampler
        samplers = []
        def create_sampler():
            sampler = AppMemorySampler(reader=lambda _: 10, pids=(11, 22), interval=60)
            samplers.append(sampler)
            return sampler
        # The 'loading' SSE event is sent before the model manager is entered.
        original_sse = self.server.RequestHandlerClass._sse
        def observe_sse(handler, event):
            if event.get("phase") == "loading":
                sampler = samplers[-1]
                self.assertTrue(sampler._thread.is_alive())
                sampler._reader = lambda _: 100
                sampler.sample()
                sampler._reader = lambda _: 10
            if "result" in event:
                self.assertIsNone(samplers[-1]._thread)
            return original_sse(handler, event)
        with patch("deepseek_v4_ssd.server.AppMemorySampler", side_effect=create_sampler), \
             patch.object(self.server.RequestHandlerClass, "_sse", observe_sse):
            with urlopen(self.request()) as response:
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertEqual(result["peak_app_memory_bytes"], 200)
        self.assertIsNone(samplers[-1]._thread)

    def test_unavailable_memory_does_not_fail_generation(self):
        with patch("deepseek_v4_ssd.app_memory.physical_footprint", return_value=None):
            with urlopen(self.request()) as response:
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertIsNone(result["peak_app_memory_bytes"])
        self.assertNotIn("peak_memory_bytes", result)

    def test_rejects_invalid_lengths_and_auth_before_loading(self):
        for changes in [dict(context_length=True), dict(context_length=200000),
                        dict(generation_length=256), dict(generation_length="128"),
                        dict(benchmark_context="other"), dict(benchmark_context=None),
                        dict(benchmark_context=["code"])]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.request(**changes))
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        request = self.request()
        request.remove_header("Authorization")
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()
        self.assertEqual(self.loads, 0)

    def test_generation_error_restores_config_and_returns_stream_error(self):
        self.runtime.fail = True
        with urlopen(self.request()) as response:
            body = response.read().decode()
        self.assertIn('"error"', body)
        self.assertIn("test generation failure", body)
        self.assertIs(self.runtime.config, self.original_config)
        self.assertFalse(self.server.metrics.snapshot()["generating"])

    def test_missing_asset_returns_a_stream_error(self):
        with patch("deepseek_v4_ssd.throughput.prompt_tokens", side_effect=FileNotFoundError("Missing benchmark context")):
            with urlopen(self.request()) as response:
                body = response.read().decode()
        self.assertIn('"error"', body)
        self.assertIn("Missing benchmark context", body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        self.assertIs(self.runtime.config, self.original_config)

    def test_context_limit_rejected_without_generation(self):
        self.runtime.installed.maximum_context = 1100
        with urlopen(self.request()) as response:
            body = response.read().decode()
        self.assertIn("context limit", body)
        self.assertFalse(self.runtime.finished.is_set())

    def test_disconnect_cancels_prefill_and_restores_config(self):
        self.runtime.slow = True
        connection = socket.create_connection(self.server.server_address)
        data = self.request().data
        connection.sendall((
            "POST /api/benchmark/throughput HTTP/1.1\r\nHost: localhost\r\n"
            "Authorization: Bearer secret\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n\r\n").encode() + data)
        self.assertTrue(self.runtime.entered.wait(3))
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        # The App immediately asks to unload after cancellation. The endpoint
        # must wait for generation to unwind before closing the runtime.
        unload = Request(
            self.url.replace("api/benchmark/throughput", "api/models/unload"),
            data=json.dumps({"model": "deepseek-v4-flash-0731"}).encode(),
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        with urlopen(unload, timeout=5) as response:
            self.assertEqual(response.status, 200)
        self.assertTrue(self.runtime.finished.wait(3))
        self.assertIs(self.runtime.config, self.original_config)
        self.assertTrue(self.runtime.closed)
        self.assertIsNone(self.manager.status_snapshot()["loaded_model"])
        self.assertFalse(self.server.metrics.snapshot()["generating"])
