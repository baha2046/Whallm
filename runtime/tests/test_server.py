from __future__ import annotations

import json
import threading
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_v4_ssd.generation import GenerationOptions, GeneratedPiece, THINK_START
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import (
    ModelDefaults,
    ModelManager,
    ModelSpec,
)
from deepseek_v4_ssd.server import (
    APIError,
    OpenAIServer,
    ServerDefaults,
    _options,
    _parser,
    _reasoning_settings,
)
from deepseek_v4_ssd.tool_codec import AssistantTurn, ToolCall


class FakeRuntime:
    model_id = "deepseek-ai/DeepSeek-V4-Flash-0731"
    config = SimpleNamespace(
        slots=1024,
        read_workers=4,
        prefetch_read_workers=2,
        prefill_step_size=32,
        moe_prefill_step_size=0,
        fp8_kv_cache=True,
        layer_major_prefill=True,
        layer_major_prefill_threshold=1_024,
        batched_expert_prefill=True,
        prompt_cache_entries=2,
        prompt_cache_memory_gib=8,
        persistent_prompt_cache=True,
        fp4_index_cache=True,
        expert_page_cache_probe=False,
        expert_file_cache_policy="cached",
        dspark_prompt_cache=False,
        dspark_hash_prefetch=False,
        dspark_adaptive_block=False,
        dspark_fallback_enabled=True,
        dspark_sequential_verification=False,
        dspark_hybrid_verification=False,
        dspark_confidence_threshold=0.6,
        dspark_slots=768,
        power_saving_limit_gbps=None,
    )
    installed = SimpleNamespace(
        root=Path("/tmp/model"),
        is_qwen=False,
        has_dspark=False,
    )
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
        direct_io_alignment=0,
    )
    metrics = SimpleNamespace(
        snapshot=lambda: {
            "runtime_prompt_tokens": 5,
            "runtime_generation_tokens": 2,
            "accumulated_generation_tokens": 2,
            "prompt_cache_reused_tokens": 3,
            "completed_request_count": 1,
            "request_seconds": 0.75,
            "time_to_first_token_seconds": 0.5,
            "prefill_tokens_per_second": 4.0,
            "decode_seconds": 0.25,
            "decode_tokens_per_second": 4.0,
            "cache_state_eval_seconds": 0.01,
            "cache_state_eval_count": 2,
            "request_prefill_step_size": 512,
            "layer_major_prefill": True,
            "request_expert_cache_hit_rate": 0.9,
            "request_expert_cache_hits": 90,
            "request_expert_cache_misses": 10,
            "request_expert_evictions": 4,
            "request_expert_bytes_read": 1_024,
            "request_expert_read_seconds": 0.2,
            "request_ssd_read_bytes_per_second": 5_120.0,
            "request_routing_sync_seconds": 0.3,
        }
    )

    def __init__(self):
        self.generation_gate = None
        self.generation_entered = threading.Event()
        self.pause_after_chunks = None
        self.chunk_paused = threading.Event()
        self.chunk_gate = threading.Event()
        self.response_chunks = None
        self.parsed_turn = AssistantTurn("Hello", "", ())
        self.last_messages = None
        self.last_tools = None
        self.last_tool_choice = None
        self.last_thinking_mode = None
        self.last_reasoning_effort = None
        self.last_options = None
        self.last_parse_text = None
        self.parse_error = False
        self.closed = False

    def encode_chat(
        self,
        messages,
        thinking_mode="chat",
        tools=None,
        tool_choice=None,
        reasoning_effort="low",
    ):
        self.last_messages = messages
        self.last_tools = tools
        self.last_tool_choice = tool_choice
        self.last_thinking_mode = thinking_mode
        self.last_reasoning_effort = reasoning_effort
        return "prompt" + (THINK_START if thinking_mode == "thinking" else "")

    def parse_chat(self, text, thinking_mode):
        self.last_parse_text = text
        if self.parse_error:
            raise ValueError("invalid tool call")
        return self.parsed_turn

    def stream(self, prompt, options):
        self.last_options = options
        if self.generation_gate is not None:
            self.generation_entered.set()
            self.generation_gate.wait(timeout=2)
        chunks = self.response_chunks
        if chunks is None:
            text = "plan</think>Hello" if prompt.endswith(THINK_START) else "Hello"
            midpoint = max(1, len(text) // 2)
            chunks = [text[:midpoint], text[midpoint:]]
        for index, chunk in enumerate(chunks, 1):
            finish = "stop" if index == len(chunks) else None
            yield GeneratedPiece(chunk, 9 + index, 5, index, finish)
            if index == self.pause_after_chunks:
                self.chunk_paused.set()
                self.chunk_gate.wait(timeout=2)

    def close(self):
        self.closed = True


class QwenServerSettingsTests(unittest.TestCase):
    def test_qwen_reasoning_effort_maps_to_checkpoint_values(self):
        cases = {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": "xhigh",
        }
        for effort, native in cases.items():
            with self.subTest(effort=effort):
                self.assertEqual(
                    _reasoning_settings({"reasoning_effort": effort}, qwen=True),
                    ("thinking", native),
                )

    def test_qwen_thinking_mode_uses_or_rule(self):
        cases = (
            ({}, False, "chat"),
            ({"thinking_mode": "chat"}, False, "chat"),
            ({"thinking_mode": "thinking"}, False, "thinking"),
            ({"reasoning_effort": "none"}, False, "chat"),
            (
                {"thinking_mode": "chat", "reasoning_effort": "medium"},
                False,
                "thinking",
            ),
            ({"reasoning": {"effort": "none"}}, True, "chat"),
            (
                {"thinking_mode": "chat", "reasoning": {"effort": "high"}},
                True,
                "thinking",
            ),
        )
        for payload, responses_api, expected in cases:
            with self.subTest(payload=payload, responses_api=responses_api):
                mode, _ = _reasoning_settings(
                    payload,
                    responses_api=responses_api,
                    qwen=True,
                )
                self.assertEqual(mode, expected)


class QwenSamplingServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FakeRuntime()
        cls.runtime.installed = SimpleNamespace(
            root=Path("/tmp/qwen-model"),
            is_qwen=True,
            has_dspark=False,
        )
        cls.model_manager = ModelManager(
            [
                ModelSpec(
                    id="qwen3.8-flash-next-fp8",
                    alias=None,
                    path="/tmp/qwen-model",
                    model_kind="qwen3.8-flash-next",
                    runtime=RuntimeConfig(),
                    defaults=ModelDefaults(262_144, 0.7, 0.8, 20),
                )
            ],
            runtime_loader=lambda _: cls.runtime,
            clear_cache=lambda: None,
        )
        cls.server = OpenAIServer(("127.0.0.1", 0), cls.model_manager)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.model_manager.close()

    def request(self, path, body):
        request = Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request) as response:
                return response.status, response.read()
        except HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    def test_qwen_endpoints_select_mode_sampling_options(self):
        cases = (
            (
                "/v1/chat/completions",
                {
                    "model": "qwen3.8-flash-next-fp8",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "thinking_mode": "chat",
                },
                GenerationOptions(
                    max_tokens=262_144,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=1.5,
                    repetition_penalty=1.0,
                ),
            ),
            (
                "/v1/responses",
                {
                    "model": "qwen3.8-flash-next-fp8",
                    "input": "Hi",
                    "reasoning": {"effort": "high"},
                },
                GenerationOptions(
                    max_tokens=262_144,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                ),
            ),
            (
                "/v1/completions",
                {
                    "model": "qwen3.8-flash-next-fp8",
                    "prompt": "Hi",
                    "thinking_mode": "thinking",
                    "reasoning_effort": "high",
                },
                GenerationOptions(
                    max_tokens=262_144,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=1.5,
                    repetition_penalty=1.0,
                ),
            ),
        )
        for path, payload, expected in cases:
            with self.subTest(path=path):
                status, body = self.request(path, payload)
                self.assertEqual(status, 200, body)
                self.assertEqual(self.runtime.last_options, expected)

    def test_qwen_request_sampling_fields_override_mode_defaults(self):
        status, body = self.request(
            "/v1/chat/completions",
            {
                "model": "qwen3.8-flash-next-fp8",
                "messages": [{"role": "user", "content": "Hi"}],
                "thinking_mode": "thinking",
                "temperature": 0,
                "top_p": 0.6,
                "top_k": 7,
                "min_p": 0.4,
                "presence_penalty": 0,
                "repetition_penalty": 1.2,
            },
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(
            self.runtime.last_options,
            GenerationOptions(
                max_tokens=262_144,
                temperature=0,
                top_p=0.6,
                top_k=7,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
            ),
        )

    def test_qwen_request_rejects_nonzero_presence_penalty(self):
        status, body = self.request(
            "/v1/chat/completions",
            {
                "model": "qwen3.8-flash-next-fp8",
                "messages": [{"role": "user", "content": "Hi"}],
                "presence_penalty": 1.5,
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "presence_penalty")


class ServerArgumentTests(unittest.TestCase):
    def test_model_and_catalog_are_optional_and_mutually_exclusive(self):
        arguments = _parser().parse_args([])
        self.assertIsNone(arguments.model)
        self.assertIsNone(arguments.model_catalog)
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                ["--model", "/tmp/model", "--model-catalog", "/tmp/catalog.json"]
            )

    def test_dspark_prompt_cache_is_research_opt_in(self):
        arguments = _parser().parse_args(["--model", "/tmp/model"])
        self.assertFalse(arguments.dspark_prompt_cache)

        arguments = _parser().parse_args(
            ["--model", "/tmp/model", "--dspark", "--dspark-prompt-cache"]
        )
        self.assertTrue(arguments.dspark_prompt_cache)

    def test_expert_page_cache_probe_is_research_opt_in(self):
        arguments = _parser().parse_args(["--model", "/tmp/model"])
        self.assertFalse(arguments.expert_page_cache_probe)

        arguments = _parser().parse_args(
            ["--model", "/tmp/model", "--expert-page-cache-probe"]
        )
        self.assertTrue(arguments.expert_page_cache_probe)

    def test_expert_file_cache_policy_defaults_to_cached(self):
        arguments = _parser().parse_args(["--model", "/tmp/model"])
        self.assertEqual(arguments.expert_file_cache_policy, "cached")

        arguments = _parser().parse_args(
            [
                "--model",
                "/tmp/model",
                "--expert-file-cache-policy",
                "bypass",
            ]
        )
        self.assertEqual(arguments.expert_file_cache_policy, "bypass")

    def test_layer_major_prefill_threshold_defaults_to_1024(self):
        arguments = _parser().parse_args(["--model", "/tmp/model"])
        self.assertEqual(arguments.layer_major_prefill_threshold, 1_024)
        arguments = _parser().parse_args(
            [
                "--model",
                "/tmp/model",
                "--layer-major-prefill-threshold",
                "2048",
            ]
        )
        self.assertEqual(arguments.layer_major_prefill_threshold, 2_048)

    def test_power_saving_limit_uses_fixed_values(self):
        self.assertIsNone(
            _parser().parse_args(["--model", "/tmp/model"]).power_saving_limit_gbps
        )
        self.assertEqual(
            _parser().parse_args(
                ["--model", "/tmp/model", "--power-saving-limit-gbps", "0.5"]
            ).power_saving_limit_gbps,
            0.5,
        )
        with self.assertRaises(SystemExit):
            _parser().parse_args(
                ["--model", "/tmp/model", "--power-saving-limit-gbps", "0.75"]
            )


class ServerTests(unittest.TestCase):
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    def test_generation_defaults_match_app_defaults(self):
        options = _options({}, ServerDefaults())
        self.assertEqual(options, GenerationOptions())
        self.assertEqual(
            _options({}, ServerDefaults(), thinking_mode="thinking"),
            GenerationOptions(),
        )

    def test_generation_token_limit_is_272000(self):
        self.assertEqual(_options({"max_tokens": 272_000}, ServerDefaults()).max_tokens, 272_000)
        with self.assertRaises(APIError):
            _options({"max_tokens": 272_001}, ServerDefaults())

    @classmethod
    def setUpClass(cls):
        cls.runtime = FakeRuntime()
        cls.model_manager = ModelManager(
            [
                ModelSpec(
                    id="deepseek-v4-flash-0731",
                    alias="work-model",
                    path="/tmp/model",
                    model_kind="deepseek-v4",
                    runtime=RuntimeConfig(),
                    defaults=ModelDefaults(272_000, 0.2, 0.98, 0),
                )
            ],
            runtime_loader=lambda _: cls.runtime,
            clear_cache=lambda: None,
        )
        cls.server = OpenAIServer(
            ("127.0.0.1", 0),
            cls.model_manager,
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
        cls.model_manager.close()

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

    def test_models_requires_bearer_key(self):
        status, _, body = self.request("/v1/models", authenticated=False)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["code"], "invalid_api_key")

        status, _, body = self.request("/v1/models")
        self.assertEqual(status, 200)
        models = json.loads(body)["data"]
        self.assertEqual(
            [model["id"] for model in models],
            ["deepseek-v4-flash-0731", "work-model"],
        )
        self.assertEqual(models[0]["owned_by"], models[1]["owned_by"])

    def test_model_load_and_unload_require_bearer_key(self):
        runtime = FakeRuntime()
        manager = ModelManager(
            [
                ModelSpec(
                    id="deepseek-v4-flash-0731",
                    alias="work-model",
                    path="/tmp/model",
                    model_kind="deepseek-v4",
                    runtime=RuntimeConfig(),
                    defaults=ModelDefaults(272_000, 0.2, 0.98, 0),
                )
            ],
            runtime_loader=lambda _: runtime,
            clear_cache=lambda: None,
        )
        server = OpenAIServer(("127.0.0.1", 0), manager, api_key="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, model, *, authenticated=True):
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["Authorization"] = "Bearer secret"
            value = Request(
                base + path,
                data=json.dumps({"model": model}).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(value) as response:
                    return response.status, response.read()
            except HTTPError as error:
                try:
                    return error.code, error.read()
                finally:
                    error.close()

        try:
            status, body = request(
                "/api/models/load", "work-model", authenticated=False
            )
            self.assertEqual(status, 401)
            self.assertEqual(json.loads(body)["error"]["code"], "invalid_api_key")

            status, body = request("/api/models/load", "work-model")
            self.assertEqual(status, 200)
            self.assertEqual(
                json.loads(body)["loaded_model"], "deepseek-v4-flash-0731"
            )

            status, body = request(
                "/api/models/unload", "deepseek-v4-flash-0731"
            )
            self.assertEqual(status, 200)
            self.assertIsNone(json.loads(body)["loaded_model"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            manager.close()

    def test_generation_responses_preserve_the_requested_alias(self):
        chat_status, _, chat_body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "work-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        completion_status, _, completion_body = self.request(
            "/v1/completions",
            method="POST",
            body={"model": "work-model", "prompt": "Hi"},
        )
        response_status, _, response_body = self.request(
            "/v1/responses",
            method="POST",
            body={"model": "work-model", "input": "Hi"},
        )
        self.assertEqual((chat_status, completion_status, response_status), (200, 200, 200))
        self.assertEqual(json.loads(chat_body)["model"], "work-model")
        self.assertEqual(json.loads(completion_body)["model"], "work-model")
        self.assertEqual(json.loads(response_body)["model"], "work-model")

    def test_streaming_events_preserve_the_requested_alias(self):
        requests = (
            (
                "/v1/chat/completions",
                {
                    "model": "work-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            ),
            (
                "/v1/completions",
                {"model": "work-model", "prompt": "Hi", "stream": True},
            ),
        )
        for path, payload in requests:
            with self.subTest(path=path):
                status, _, body = self.request(path, method="POST", body=payload)
                events = [
                    json.loads(line[6:])
                    for line in body.decode().splitlines()
                    if line.startswith("data: {")
                ]
                self.assertEqual(status, 200)
                self.assertTrue(events)
                self.assertTrue(all(event["model"] == "work-model" for event in events))

        status, _, body = self.request(
            "/v1/responses",
            method="POST",
            body={"model": "work-model", "input": "Hi", "stream": True},
        )
        events = [
            json.loads(line[6:])
            for line in body.decode().splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(status, 200)
        response_events = [event["response"] for event in events if "response" in event]
        self.assertTrue(response_events)
        self.assertTrue(
            all(response["model"] == "work-model" for response in response_events)
        )

    def test_unknown_model_uses_model_not_found(self):
        status, _, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "unknown",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["code"], "model_not_found")

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

    def test_chat_completion_accepts_reasoning_effort(self):
        status, _, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_effort": "high",
            },
        )
        response = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["message"]["reasoning_content"], "plan")
        self.assertEqual(self.runtime.last_thinking_mode, "thinking")
        self.assertEqual(self.runtime.last_reasoning_effort, "high")

    def test_chat_rejects_final_assistant_history(self):
        status, _, body = self.request(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "messages.1.role")

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

    def test_chat_returns_tool_calls_and_accepts_tool_results(self):
        runtime = self.runtime
        runtime.response_chunks = ["raw", " tool", " output"]
        runtime.parsed_turn = AssistantTurn(
            "",
            "",
            (ToolCall("get_weather", '{"city":"Taipei"}'),),
        )
        try:
            status, _, body = self.request(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [self.tool],
                    "tool_choice": "required",
                },
            )
            response = json.loads(body)
            call = response["choices"][0]["message"]["tool_calls"][0]
            self.assertEqual(status, 200)
            self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual(call["function"]["name"], "get_weather")
            self.assertEqual(call["function"]["arguments"], '{"city":"Taipei"}')
            self.assertTrue(call["id"].startswith("call_"))
            self.assertEqual(runtime.last_parse_text, "raw tool output")
            self.assertEqual(runtime.last_tool_choice.mode, "required")

            runtime.parsed_turn = AssistantTurn("It is sunny.", "", ())
            status, _, _ = self.request(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "messages": [
                        {"role": "user", "content": "Weather?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [call],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Sunny",
                        },
                    ],
                    "tools": [self.tool],
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(runtime.last_messages[-1]["role"], "tool")
            self.assertEqual(runtime.last_messages[-1]["content"], "Sunny")
        finally:
            runtime.response_chunks = None
            runtime.parsed_turn = AssistantTurn("Hello", "", ())

    def test_streaming_tool_call_handles_fragments_and_parse_errors(self):
        runtime = self.runtime
        raw = (
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Taipei'
            '</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
        )
        runtime.response_chunks = list(raw)
        runtime.parsed_turn = AssistantTurn(
            "",
            "",
            (ToolCall("get_weather", '{"city": "Taipei"}'),),
        )
        try:
            status, _, body = self.request(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [self.tool],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            text = body.decode()
            self.assertEqual(status, 200)
            self.assertEqual(runtime.last_parse_text, raw)
            self.assertIn('"tool_calls":[{"index":0,', text)
            self.assertIn('"id":"call_', text)
            self.assertIn('"name":"get_weather"', text)
            self.assertIn('"finish_reason":"tool_calls"', text)
            self.assertIn('"usage":{"prompt_tokens":5', text)
            self.assertTrue(text.endswith("data: [DONE]\n\n"))
            streamed_calls = [
                call
                for line in text.splitlines()
                if line.startswith("data: {")
                for choice in json.loads(line[6:]).get("choices", [])
                for call in choice.get("delta", {}).get("tool_calls", [])
            ]
            self.assertEqual(
                "".join(call["function"].get("arguments", "") for call in streamed_calls),
                '{"city": "Taipei"}',
            )

            runtime.parse_error = True
            status, _, body = self.request(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [self.tool],
                    "stream": True,
                },
            )
            text = body.decode()
            self.assertEqual(status, 200)
            self.assertIn('"code":"invalid_tool_call"', text)
            self.assertTrue(text.endswith("data: [DONE]\n\n"))
        finally:
            runtime.parse_error = False
            runtime.response_chunks = None
            runtime.parsed_turn = AssistantTurn("Hello", "", ())

    def test_streaming_tool_call_emits_before_generation_finishes(self):
        runtime = self.runtime
        runtime.response_chunks = [
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n',
            '<｜DSML｜parameter name="city" string="true">Taipei'
            '</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        ]
        runtime.parsed_turn = AssistantTurn(
            "",
            "",
            (ToolCall("get_weather", '{"city":"Taipei"}'),),
        )
        runtime.pause_after_chunks = 1
        runtime.chunk_paused.clear()
        runtime.chunk_gate.clear()
        tool_delta_received = threading.Event()
        client_errors = []

        def consume_stream():
            body = json.dumps(
                {
                    "model": "deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [self.tool],
                    "stream": True,
                }
            ).encode()
            request = Request(
                self.base + "/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": "Bearer secret",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urlopen(request) as response:
                    for line in response:
                        if b'"tool_calls"' in line:
                            tool_delta_received.set()
                        if line == b"data: [DONE]\n":
                            break
            except Exception as error:
                client_errors.append(error)

        client = threading.Thread(target=consume_stream)
        client.start()
        try:
            self.assertTrue(runtime.chunk_paused.wait(timeout=1))
            self.assertTrue(
                tool_delta_received.wait(timeout=0.5),
                "Tool call was buffered until generation finished.",
            )
        finally:
            runtime.chunk_gate.set()
            client.join(timeout=2)
            runtime.pause_after_chunks = None
            runtime.response_chunks = None
            runtime.parsed_turn = AssistantTurn("Hello", "", ())
        self.assertFalse(client.is_alive())
        self.assertEqual(client_errors, [])

    def test_text_completion(self):
        status, _, body = self.request(
            "/v1/completions",
            method="POST",
            body={"model": "deepseek-v4-flash-0731", "prompt": "Say"},
        )
        response = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(response["choices"][0]["text"], "Hello")

    def test_responses_text(self):
        status, _, body = self.request(
            "/v1/responses",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "instructions": "Answer briefly.",
                "input": "Hi",
            },
        )
        response = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["type"], "message")
        self.assertEqual(response["output"][0]["content"][0]["text"], "Hello")
        self.assertEqual(response["usage"]["total_tokens"], 7)

    def test_responses_accepts_final_assistant_history(self):
        status, _, body = self.request(
            "/v1/responses",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Inspect this."}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "phase": "commentary",
                        "content": [
                            {"type": "output_text", "text": "I am checking it."}
                        ],
                    },
                ],
            },
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(
            self.runtime.last_messages[-1],
            {"role": "assistant", "content": "I am checking it."},
        )

    def test_responses_maps_codex_reasoning_effort(self):
        cases = (
            ("none", "chat", "low"),
            ("minimal", "thinking", "low"),
            ("low", "thinking", "low"),
            ("medium", "thinking", "low"),
            ("high", "thinking", "high"),
            ("xhigh", "thinking", "max"),
            ("max", "thinking", "max"),
        )
        for effort, thinking_mode, native_effort in cases:
            with self.subTest(effort=effort):
                status, _, body = self.request(
                    "/v1/responses",
                    method="POST",
                    body={
                        "model": "deepseek-v4-flash-0731",
                        "input": "Hi",
                        "reasoning": {"effort": effort, "summary": "auto"},
                    },
                )
                self.assertEqual(status, 200, body)
                self.assertEqual(self.runtime.last_thinking_mode, thinking_mode)
                self.assertEqual(
                    self.runtime.last_reasoning_effort,
                    native_effort,
                )

    def test_responses_rejects_invalid_reasoning_effort(self):
        status, _, body = self.request(
            "/v1/responses",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "input": "Hi",
                "reasoning": {"effort": "extreme"},
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "reasoning.effort")

    def test_responses_streams_text_and_tool_calls(self):
        status, headers, body = self.request(
            "/v1/responses",
            method="POST",
            body={
                "model": "deepseek-v4-flash-0731",
                "input": "Hi",
                "stream": True,
            },
        )
        events = [
            json.loads(line[6:])
            for line in body.decode().splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        self.assertEqual(events[0]["type"], "response.created")
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(
            "".join(
                event["delta"]
                for event in events
                if event["type"] == "response.output_text.delta"
            ),
            "Hello",
        )

        runtime = self.runtime
        raw = (
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">Taipei'
            '</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
        )
        runtime.response_chunks = list(raw)
        runtime.parsed_turn = AssistantTurn(
            "",
            "",
            (ToolCall("get_weather", '{"city": "Taipei"}'),),
        )
        response_tool = {"type": "function", **self.tool["function"]}
        try:
            status, _, body = self.request(
                "/v1/responses",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "input": "Weather?",
                    "tools": [response_tool],
                    "stream": True,
                },
            )
            events = [
                json.loads(line[6:])
                for line in body.decode().splitlines()
                if line.startswith("data: {")
            ]
            self.assertEqual(status, 200)
            self.assertEqual(
                "".join(
                    event["delta"]
                    for event in events
                    if event["type"]
                    == "response.function_call_arguments.delta"
                ),
                '{"city": "Taipei"}',
            )
            completed = events[-1]["response"]
            call = completed["output"][0]
            self.assertEqual(call["type"], "function_call")
            self.assertEqual(call["name"], "get_weather")

            runtime.response_chunks = ["It is sunny."]
            runtime.parsed_turn = AssistantTurn("It is sunny.", "", ())
            status, _, body = self.request(
                "/v1/responses",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "input": [
                        {"role": "user", "content": "Weather?"},
                        call,
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": "Sunny",
                        },
                    ],
                    "tools": [response_tool],
                },
            )
            response = json.loads(body)
            self.assertEqual(status, 200)
            self.assertEqual(response["output"][0]["content"][0]["text"], "It is sunny.")
            self.assertEqual(runtime.last_messages[-1]["role"], "tool")
        finally:
            runtime.response_chunks = None
            runtime.parsed_turn = AssistantTurn("Hello", "", ())

    def test_responses_tool_error_ends_for_codex(self):
        runtime = self.runtime
        runtime.response_chunks = ["I am checking the project."]
        runtime.parse_error = True
        try:
            status, _, body = self.request(
                "/v1/responses",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "input": "Review the project.",
                    "tools": [{"type": "function", **self.tool["function"]}],
                    "stream": True,
                },
            )
            events = [
                json.loads(line[6:])
                for line in body.decode().splitlines()
                if line.startswith("data: {")
            ]
            self.assertEqual(status, 200)
            self.assertEqual(events[-2]["type"], "error")
            self.assertEqual(events[-1]["type"], "response.completed")
            failed = events[-1]["response"]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"]["code"], "invalid_tool_call")
            self.assertIn(
                "Whallm could not complete the request",
                "".join(
                    event["delta"]
                    for event in events
                    if event["type"] == "response.output_text.delta"
                ),
            )
        finally:
            runtime.response_chunks = None
            runtime.parse_error = False

    def test_responses_accepts_codex_namespace_tools(self):
        runtime = self.runtime
        runtime.response_chunks = [
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="multi_agent_v1__spawn_agent">\n',
            '<｜DSML｜parameter name="task" string="true">inspect'
            '</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>',
        ]
        runtime.parsed_turn = AssistantTurn(
            "",
            "",
            (ToolCall("multi_agent_v1__spawn_agent", '{"task":"inspect"}'),),
        )
        try:
            status, _, body = self.request(
                "/v1/responses",
                method="POST",
                body={
                    "model": "deepseek-v4-flash-0731",
                    "input": "Inspect the project.",
                    "tools": [
                        {
                            "type": "namespace",
                            "name": "multi_agent_v1",
                            "description": "Agent tools.",
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "spawn_agent",
                                    "description": "Spawn an agent.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {"task": {"type": "string"}},
                                        "required": ["task"],
                                    },
                                }
                            ],
                        },
                        {"type": "web_search"},
                    ],
                    "stream": True,
                },
            )
            events = [
                json.loads(line[6:])
                for line in body.decode().splitlines()
                if line.startswith("data: {")
            ]
            self.assertEqual(status, 200)
            call = events[-1]["response"]["output"][0]
            self.assertEqual(call["type"], "function_call")
            self.assertEqual(call["namespace"], "multi_agent_v1")
            self.assertEqual(call["name"], "spawn_agent")
        finally:
            runtime.response_chunks = None
            runtime.parsed_turn = AssistantTurn("Hello", "", ())

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
        payload = json.loads(body)
        performance = payload["performance"]
        self.assertEqual(status, 200)
        self.assertIsNone(payload["runtime"]["power_saving_limit_gbps"])
        self.assertFalse(payload["runtime"]["expert_page_cache_probe"])
        self.assertFalse(payload["runtime"]["dspark_prompt_cache"])
        self.assertEqual(payload["runtime"]["layer_major_prefill_threshold"], 1_024)
        self.assertEqual(payload["runtime"]["expert_file_cache_policy"], "cached")
        self.assertEqual(
            payload["runtime"]["expert_file_direct_io_alignment_bytes"],
            0,
        )
        self.assertFalse(performance["generating"])
        self.assertEqual(performance["generation_tokens"], 2)
        self.assertGreaterEqual(performance["tokens_per_second"], 0)
        self.assertEqual(performance["ssd_bytes_read"], 1_048_576)
        self.assertEqual(performance["expert_pack_seconds"], 0.1)
        self.assertEqual(performance["expert_eviction_seconds"], 0.01)
        self.assertEqual(performance["routing_sync_seconds"], 0.05)
        self.assertEqual(performance["time_to_first_token_seconds"], 0.5)
        self.assertEqual(performance["prefill_tokens_per_second"], 4.0)
        self.assertEqual(performance["decode_tokens_per_second"], 4.0)
        self.assertEqual(performance["accumulated_generation_tokens"], 2)
        self.assertEqual(performance["completed_request_count"], 1)
        self.assertEqual(performance["request_ssd_read_bytes_per_second"], 5_120.0)
        self.assertEqual(performance["prompt_cache_reused_tokens"], 3)
        self.assertEqual(performance["cache_state_eval_count"], 2)
        self.assertEqual(performance["request_prefill_step_size"], 512)
        self.assertTrue(performance["layer_major_prefill"])
        self.assertEqual(performance["request_expert_bytes_read"], 1_024)
        self.assertEqual(performance["request_expert_cache_hit_rate"], 0.9)
        self.assertEqual(performance["active_parameters_cache"]["hit_rate"], 0.75)
        self.assertEqual(performance["active_parameters_cache"]["resident_slots"], 3)

    def test_status_poll_does_not_write_access_log(self):
        output = StringIO()
        with redirect_stderr(output):
            status, _, _ = self.request("/api/status")

        self.assertEqual(status, 200)
        self.assertEqual(output.getvalue(), "")

    def test_status_responds_while_generation_is_busy(self):
        gate = threading.Event()
        runtime = self.runtime
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
            self.assertTrue(json.loads(body)["performance"]["generating"])
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
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"]["code"], "not_found")

        status, _, body = self.request("/api/settings")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"]["code"], "not_found")

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
        self.assertEqual(json.loads(body)["error"]["param"], "tools.0.function")

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
            {"name": "Whallm", "status": "ok", "api_base": "/v1"},
        )


class EmptyCatalogServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = ModelManager([], clear_cache=lambda: None)
        cls.server = OpenAIServer(("127.0.0.1", 0), cls.manager)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.manager.close()

    def request(self, path, *, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urlopen(request) as response:
                return response.status, response.read()
        except HTTPError as error:
            try:
                return error.code, error.read()
            finally:
                error.close()

    def test_empty_catalog_starts_with_empty_models_and_zero_status(self):
        status, body = self.request("/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"], [])

        status, body = self.request("/api/status")
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIsNone(payload["loaded_model"])
        self.assertIsNone(payload["loading_model"])
        self.assertIsNone(payload["model"])
        self.assertIsNone(payload["runtime"])
        self.assertEqual(payload["performance"]["generation_tokens"], 0)

    def test_model_load_failure_returns_500_and_allows_retry(self):
        attempts = 0

        def fail(_):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("load failed")

        manager = ModelManager(
            [
                ModelSpec(
                    id="deepseek-v4-flash-0731",
                    alias=None,
                    path="/tmp/missing",
                    model_kind="deepseek-v4",
                    runtime=RuntimeConfig(),
                    defaults=ModelDefaults(272_000, 0.2, 0.98, 0),
                )
            ],
            runtime_loader=fail,
            clear_cache=lambda: None,
        )
        server = OpenAIServer(("127.0.0.1", 0), manager)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        body = json.dumps(
            {
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        ).encode()
        try:
            for _ in range(2):
                request = Request(
                    base + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(HTTPError) as caught:
                    urlopen(request)
                error = caught.exception
                try:
                    self.assertEqual(error.code, 500)
                    self.assertEqual(
                        json.loads(error.read())["error"]["code"],
                        "model_load_failed",
                    )
                finally:
                    error.close()
            self.assertEqual(attempts, 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            manager.close()

    def test_health_and_status_respond_while_a_model_is_loading(self):
        entered = threading.Event()
        release = threading.Event()
        request_errors = []

        def load(_):
            entered.set()
            release.wait(timeout=2)
            return FakeRuntime()

        manager = ModelManager(
            [
                ModelSpec(
                    id="deepseek-v4-flash-0731",
                    alias=None,
                    path="/tmp/model",
                    model_kind="deepseek-v4",
                    runtime=RuntimeConfig(),
                    defaults=ModelDefaults(272_000, 0.2, 0.98, 0),
                )
            ],
            runtime_loader=load,
            clear_cache=lambda: None,
        )
        server = OpenAIServer(("127.0.0.1", 0), manager)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        body = json.dumps(
            {
                "model": "deepseek-v4-flash-0731",
                "messages": [{"role": "user", "content": "Hi"}],
            }
        ).encode()

        def generate():
            try:
                request = Request(
                    base + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=2) as response:
                    response.read()
            except Exception as error:
                request_errors.append(error)

        generation_thread = threading.Thread(target=generate)
        generation_thread.start()
        try:
            self.assertTrue(entered.wait(timeout=1))
            with urlopen(base + "/healthz", timeout=1) as response:
                self.assertEqual(json.loads(response.read()), {"status": "ok"})
            with urlopen(base + "/api/status", timeout=1) as response:
                status = json.loads(response.read())
            self.assertIsNone(status["loaded_model"])
            self.assertEqual(
                status["loading_model"], "deepseek-v4-flash-0731"
            )
        finally:
            release.set()
            generation_thread.join(timeout=2)
            server.shutdown()
            server.server_close()
            server_thread.join()
            manager.close()
        self.assertFalse(generation_thread.is_alive())
        self.assertEqual(request_errors, [])


if __name__ == "__main__":
    unittest.main()
