from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_v4_ssd.generation import GeneratedPiece, THINK_START, encode_chat
from deepseek_v4_ssd.server import OpenAIServer


class FakeRuntime:
    model_id = "deepseek-ai/DeepSeek-V4-Flash-0731"
    config = SimpleNamespace(
        slots=1024,
        read_workers=4,
        prefill_step_size=32,
        fp8_kv_cache=True,
    )
    installed = SimpleNamespace(root=Path("/tmp/model"))
    expert_cache = SimpleNamespace(
        metrics=SimpleNamespace(
            hits=3,
            misses=1,
            hit_rate=0.75,
            bytes_read=1_048_576,
            read_seconds=0.25,
            pack_seconds=0.1,
            eviction_seconds=0.01,
            routing_sync_seconds=0.05,
            evictions=0,
        ),
        resident_count=3,
    )

    def __init__(self):
        self.generation_gate = None
        self.generation_entered = threading.Event()

    def encode_chat(self, messages, thinking_mode="chat"):
        return encode_chat(messages, thinking_mode)

    def stream(self, prompt, options):
        if self.generation_gate is not None:
            self.generation_entered.set()
            self.generation_gate.wait(timeout=2)
        text = "plan</think>Hello" if prompt.endswith(THINK_START) else "Hello"
        midpoint = max(1, len(text) // 2)
        yield GeneratedPiece(text[:midpoint], 10, 5, 1, None)
        yield GeneratedPiece(text[midpoint:], 11, 5, 2, "stop")


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = OpenAIServer(
            ("127.0.0.1", 0),
            FakeRuntime(),
            api_key="secret",
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, *, method="GET", body=None, authenticated=True):
        headers = {}
        if authenticated:
            headers["Authorization"] = "Bearer secret"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urlopen(request) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            try:
                return error.code, error.headers, error.read()
            finally:
                error.close()

    def test_official_chat_subset_encoding(self):
        prompt = encode_chat(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ]
        )
        self.assertEqual(
            prompt,
            "<｜begin▁of▁sentence｜>Be brief.<｜User｜>Hi"
            "<｜Assistant｜></think>Hello<｜end▁of▁sentence｜>"
            "<｜User｜>Again<｜Assistant｜></think>",
        )

    def test_models_requires_bearer_key(self):
        status, _, body = self.request("/v1/models", authenticated=False)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["code"], "invalid_api_key")

        status, _, body = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"][0]["id"], "deepseek-v4-flash-0731")

    def test_chat_completion_and_reasoning(self):
        status, _, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking_mode": "thinking",
            },
        )
        response = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "plan")
        self.assertEqual(response["usage"]["total_tokens"], 7)

    def test_streaming_chat_uses_openai_sse_shape(self):
        status, headers, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking_mode": "thinking",
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        text = body.decode()
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertIn('"role":"assistant"', text)
        self.assertIn('"reasoning_content":"p"', text)
        self.assertIn('"reasoning_content":"lan"', text)
        self.assertIn('"content":"Hello"', text)
        self.assertIn('"usage":{"prompt_tokens":5', text)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))

    def test_text_completion(self):
        status, _, body = self.request(
            "/v1/completions",
            method="POST",
            body={"model": "deepseek-v4-flash-0731", "prompt": "Say"},
        )
        response = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["text"], "Hello")

    def test_status_reports_live_performance_metrics(self):
        self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        status, _, body = self.request("/api/status")
        performance = json.loads(body)["performance"]
        self.assertEqual(status, 200)
        self.assertFalse(performance["generating"])
        self.assertEqual(performance["generation_tokens"], 2)
        self.assertGreaterEqual(performance["tokens_per_second"], 0)
        self.assertEqual(performance["ssd_bytes_read"], 1_048_576)
        self.assertEqual(performance["expert_pack_seconds"], 0.1)
        self.assertEqual(performance["expert_eviction_seconds"], 0.01)
        self.assertEqual(performance["routing_sync_seconds"], 0.05)
        self.assertEqual(performance["active_parameters_cache"]["hit_rate"], 0.75)
        self.assertEqual(performance["active_parameters_cache"]["resident_slots"], 3)

    def test_status_responds_while_generation_is_busy(self):
        gate = threading.Event()
        runtime = self.server.runtime
        runtime.generation_gate = gate
        runtime.generation_entered.clear()
        request_thread = threading.Thread(
            target=lambda: self.request(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
        )
        request_thread.start()
        try:
            self.assertTrue(runtime.generation_entered.wait(timeout=1))
            status, _, body = self.request("/api/status")
            self.assertEqual(status, 200)
            self.assertIn("performance", json.loads(body))
        finally:
            gate.set()
            request_thread.join(timeout=1)
            runtime.generation_gate = None

    def test_settings_update_and_request_validation(self):
        status, _, body = self.request(
            "/api/settings",
            method="PUT",
            body={"max_tokens": 12, "temperature": 0.5, "top_p": 0.9},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["max_tokens"], 12)

        status, _, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": [{"type": "function"}],
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "tools")

        status, _, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
                "logprobs": {},
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "logprobs")

    def test_root_describes_the_api_server(self):
        status, _, body = self.request("/", authenticated=False)
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(body),
            {"name": "DeepSeekV4SSD", "status": "ok", "api_base": "/v1"},
        )


if __name__ == "__main__":
    unittest.main()
