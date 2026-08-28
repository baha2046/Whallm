from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_api",
    PROJECT_ROOT / "Scripts" / "benchmark_api.py",
)
assert SPEC is not None and SPEC.loader is not None
benchmark_api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark_api)


class FakeTokenizer:
    bos_token = None

    def encode(self, text, *, add_special_tokens):
        tokens = [ord(character) for character in text]
        return [-1, *tokens] if add_special_tokens else tokens

    def decode(self, tokens, **_):
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return f"<user>{messages[0]['content']}</user><assistant>"


class FakeBenchmarkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json({"data": [{"id": "test-model"}]})
            return
        if self.path == "/api/status":
            with self.server.state_lock:
                performance = dict(self.server.performance)
            self.send_json(
                {
                    "model": "test-model",
                    "model_path": "/tmp/test-model",
                    "performance": performance,
                }
            )
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(length))
        with self.server.state_lock:
            self.server.performance.update(
                generating=True,
                active_memory_bytes=2 * benchmark_api.GIB,
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(0.05)
        self.wfile.write(b'data: {"choices":[{"text":"Hello"}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")
        with self.server.state_lock:
            self.server.performance.update(
                generating=False,
                runtime_prompt_tokens=32,
                runtime_generation_tokens=4,
                prompt_cache_reused_tokens=0,
                time_to_first_token_seconds=1.5,
                prefill_tokens_per_second=20.0,
                decode_tokens_per_second=2.0,
                active_memory_bytes=benchmark_api.GIB,
            )

    def send_json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


class BenchmarkAPITests(unittest.TestCase):
    def test_builds_an_exact_prompt(self):
        prompt, tokens = benchmark_api.build_exact_prompt(
            FakeTokenizer(),
            {
                "role": "user",
                "content": "API benchmark test-run." + (" test" * 1_024),
            },
            1_024,
        )

        self.assertEqual(len(tokens), 1_024)
        self.assertTrue(prompt.startswith("<user>API benchmark test-run."))
        self.assertTrue(prompt.endswith("</user><assistant>"))

    def test_loads_exact_speed_bench_mixed_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "throughput_1k.jsonl"
            rows = [
                {
                    "question_id": "skip",
                    "category": "high_entropy",
                    "messages": [{"role": "user", "content": "x" * 2_000}],
                },
                *[
                    {
                        "question_id": f"mixed-{index}",
                        "category": "mixed",
                        "sub_category": "test",
                        "source": "fixture",
                        "src_id": str(index),
                        "messages": [
                            {"role": "user", "content": character * 2_000}
                        ],
                    }
                    for index, character in enumerate(("a", "b"), 1)
                ],
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            actual_path, samples = benchmark_api.load_speed_bench_samples(
                Path(directory),
                FakeTokenizer(),
                1_024,
                2,
            )

        self.assertEqual(actual_path, path)
        self.assertEqual(
            [sample["question_id"] for sample in samples],
            ["mixed-1", "mixed-2"],
        )
        self.assertTrue(all(len(sample["tokens"]) == 1_024 for sample in samples))
        self.assertTrue(
            all(
                sample["prompt"].endswith("</user><assistant>")
                for sample in samples
            )
        )

    def test_runs_api_request_and_formats_peak_and_p95(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeBenchmarkHandler)
        server.state_lock = threading.Lock()
        server.performance = {
            "generating": False,
            "runtime_prompt_tokens": 0,
            "runtime_generation_tokens": 0,
            "prompt_cache_reused_tokens": 0,
            "time_to_first_token_seconds": 0,
            "prefill_tokens_per_second": 0,
            "decode_tokens_per_second": 0,
            "active_memory_bytes": benchmark_api.GIB,
        }
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            result = benchmark_api.run_request(
                base_url,
                benchmark_api.status_endpoint(base_url),
                "",
                "test-model",
                "test prompt",
                4,
                0.01,
                5,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(result["output_text"], "Hello")
        self.assertEqual(result["input_tokens"], 32)
        self.assertEqual(result["ttft_seconds"], 1.5)
        self.assertEqual(result["prefill_tokens_per_second"], 20.0)
        self.assertEqual(result["decode_tokens_per_second"], 2.0)
        self.assertEqual(result["peak_active_memory_bytes"], 2 * benchmark_api.GIB)

        runs = []
        for value in range(1, 21):
            runs.append(
                {
                    "target_input_tokens": 1_024,
                    "input_tokens": 1_024,
                    "wall_seconds": value,
                    "ttft_seconds": value,
                    "prefill_tokens_per_second": value,
                    "decode_tokens_per_second": value,
                    "peak_active_memory_bytes": value * benchmark_api.GIB,
                }
            )
        summaries = benchmark_api.summarize(runs)
        table = benchmark_api.render_ascii_table(summaries)
        self.assertEqual(summaries[0]["metrics"]["ttft_seconds"]["peak"], 20)
        self.assertEqual(summaries[0]["metrics"]["ttft_seconds"]["p95"], 19)
        self.assertEqual(summaries[0]["metrics"]["wall_seconds"]["p95"], 19)
        self.assertIn("| Actual", table)
        self.assertIn("| Maximum", table)
        self.assertIn("| P95", table)
        self.assertIn("| Total time (s)", table)
        self.assertNotIn("| Peak", table)

    def test_uses_the_model_selected_in_whallm_for_server_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "modelID": "deepseek-ai/DeepSeek-V4-Flash-0731",
                        "modelKind": "deepseek-v4",
                    }
                ),
                encoding="utf-8",
            )
            saved = {
                "modelPath": str(model_path),
                "publicModel": "deepseek-v4-flash-0731",
                "slots": 1_152,
                "powerSavingLimitGBps": 0.75,
            }
            with mock.patch.object(
                benchmark_api,
                "saved_server_configuration",
                return_value=saved,
            ):
                actual_path, public_model, configuration = (
                    benchmark_api.launch_configuration(None, None)
                )

        self.assertEqual(actual_path, model_path.resolve())
        self.assertEqual(public_model, "deepseek-v4-flash-0731")
        self.assertEqual(configuration["slots"], 1_152)
        command = benchmark_api.server_command(
            actual_path,
            public_model,
            "127.0.0.1",
            11_434,
            configuration,
        )
        self.assertIn("--slots", command)
        self.assertNotIn("--power-saving-limit-gbps", command)

    def test_finds_the_requested_model_in_the_whallm_model_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory)
            model_path = model_root / "deepseek-v4-flash-0731.dsv4"
            model_path.mkdir()
            (model_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "modelID": "deepseek-ai/DeepSeek-V4-Flash-0731",
                        "modelKind": "deepseek-v4",
                    }
                ),
                encoding="utf-8",
            )
            preferences = {"modelLibraryRoot": str(model_root)}
            with mock.patch.object(
                benchmark_api,
                "app_preferences",
                return_value=preferences,
            ):
                actual_path, public_model, configuration = (
                    benchmark_api.launch_configuration(
                        None,
                        "deepseek-v4-flash-0731",
                    )
                )

        self.assertEqual(actual_path, model_path.resolve())
        self.assertEqual(public_model, "deepseek-v4-flash-0731")
        self.assertEqual(configuration, {})

    def test_uses_the_qwen_api_model_id_and_new_app_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory)
            model_path = model_root / "qwen3.8-flash-next.dsv4"
            model_path.mkdir()
            (model_path / "manifest.json").write_text(
                json.dumps(
                    {
                        "modelID": "Qwen/Qwen3.8-Flash-Next-FP8",
                        "modelKind": "qwen3.8-flash-next",
                    }
                ),
                encoding="utf-8",
            )
            preferences = {
                "modelLibraryRoot": str(model_root),
                "serverConfiguration": json.dumps(
                    {"powerSavingLimitGBps": 2}
                ).encode(),
                "modelAdvancedSettings.qwen3.8-flash-next": json.dumps(
                    {"slots": 900}
                ).encode(),
            }
            with mock.patch.object(
                benchmark_api,
                "app_preferences",
                return_value=preferences,
            ):
                actual_path, public_model, configuration = (
                    benchmark_api.launch_configuration(
                        None,
                        "qwen3.8-flash-next-fp8",
                    )
                )

        self.assertEqual(actual_path, model_path.resolve())
        self.assertEqual(public_model, "qwen3.8-flash-next-fp8")
        self.assertEqual(configuration["slots"], 900)
        self.assertEqual(configuration["powerSavingLimitGBps"], 2)

    def test_stops_a_server_started_by_the_benchmark(self):
        arguments = SimpleNamespace(
            base_url="http://127.0.0.1:11434/v1",
            model=None,
            model_path=None,
            server_start_timeout=5,
        )
        process = mock.Mock()
        log = mock.Mock()
        with (
            mock.patch.object(benchmark_api, "server_is_listening", return_value=False),
            mock.patch.object(
                benchmark_api,
                "local_server_address",
                return_value=("127.0.0.1", 11_434),
            ),
            mock.patch.object(
                benchmark_api,
                "launch_configuration",
                return_value=(Path("/model"), "test-model", {}),
            ),
            mock.patch.object(
                benchmark_api,
                "server_command",
                return_value=["python", "-m", "server"],
            ),
            mock.patch.object(
                benchmark_api,
                "start_local_server",
                return_value=(process, log),
            ),
            mock.patch.object(benchmark_api, "wait_for_server"),
            mock.patch.object(
                benchmark_api,
                "run_connected_benchmark",
                return_value=0,
            ) as run_connected,
            mock.patch.object(benchmark_api, "stop_local_server") as stop_server,
        ):
            result = benchmark_api.run_benchmark(arguments)

        self.assertEqual(result, 0)
        self.assertTrue(run_connected.call_args.args[-1])
        stop_server.assert_called_once_with(process, log)


if __name__ == "__main__":
    unittest.main()
