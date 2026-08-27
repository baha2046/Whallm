#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import plistlib
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_INPUT_TOKENS = (1_024, 4_096, 16_384, 32_768)
GIB = 1024**3
APP_PREFERENCES_DOMAIN = "com.deepseekv4ssd.app"
DEFAULT_PUBLIC_MODELS = {
    "deepseek-v4": "deepseek-v4-flash-0731",
    "qwen3.8-flash-next": "Qwen/Qwen3.8-Flash-Next-FP8",
}


class BenchmarkError(RuntimeError):
    pass


def parse_token_count(value: str) -> int:
    text = value.strip().upper().replace("_", "")
    multiplier = 1
    if text.endswith("K"):
        text = text[:-1]
        multiplier = 1_024
    try:
        result = int(text) * multiplier
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid token count: {value}") from error
    if result < 16:
        raise argparse.ArgumentTypeError("Token counts must be at least 16")
    return result


def api_endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def status_endpoint(base_url: str) -> str:
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, f"{path}/api/status", "", ""))


def request_headers(api_key: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def error_message(error: Exception) -> str:
    if isinstance(error, HTTPError):
        body = error.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(body).get("error", {}).get("message")
        except json.JSONDecodeError:
            message = None
        detail = message or error.reason
        return f"HTTP {error.code}: {detail}"
    if isinstance(error, URLError):
        return f"Unable to connect: {error.reason}"
    return f"{type(error).__name__}: {error}"


def get_json(url: str, api_key: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers=request_headers(api_key))
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise BenchmarkError(f"{url}: {error_message(error)}") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{url}: the response must be a JSON object")
    return value


def available_models(base_url: str, api_key: str, timeout: float) -> list[str]:
    payload = get_json(api_endpoint(base_url, "models"), api_key, timeout)
    models = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not models:
        raise BenchmarkError("The API did not return an available model")
    return models


def select_model(models: list[str], requested: str | None) -> str:
    if requested:
        if requested not in models:
            names = ", ".join(models)
            raise BenchmarkError(f"Model '{requested}' is unavailable. Available: {names}")
        return requested
    if not sys.stdin.isatty():
        if len(models) == 1:
            return models[0]
        raise BenchmarkError("Use --model when standard input is not interactive")

    print("Available models:")
    for index, model in enumerate(models, 1):
        print(f"  {index}. {model}")
    while True:
        choice = input("Select model [1]: ").strip() or "1"
        try:
            return models[int(choice) - 1]
        except (ValueError, IndexError):
            print(f"Enter a number from 1 to {len(models)}.")


def server_address(base_url: str) -> tuple[str, int]:
    parts = urlsplit(base_url)
    if not parts.hostname:
        raise BenchmarkError(f"Invalid API base URL: {base_url}")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as error:
        raise BenchmarkError(f"Invalid API base URL: {base_url}") from error
    return parts.hostname, port


def server_is_listening(base_url: str, timeout: float = 0.5) -> bool:
    host, port = server_address(base_url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def local_server_address(base_url: str) -> tuple[str, int]:
    parts = urlsplit(base_url)
    host, port = server_address(base_url)
    if (
        parts.scheme != "http"
        or host not in {"127.0.0.1", "::1", "localhost"}
        or parts.path.rstrip("/") != "/v1"
    ):
        raise BenchmarkError(
            "Automatic Server startup requires a local http://HOST:PORT/v1 URL."
        )
    return ("127.0.0.1" if host == "localhost" else host), port


def app_preferences() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["defaults", "export", APP_PREFERENCES_DOMAIN, "-"],
            check=True,
            capture_output=True,
        )
        preferences = plistlib.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, plistlib.InvalidFileException):
        return {}
    return preferences if isinstance(preferences, dict) else {}


def saved_server_configuration(
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preferences = preferences if preferences is not None else app_preferences()
    raw = preferences.get("serverConfiguration")
    try:
        configuration = json.loads(raw) if isinstance(raw, (bytes, str)) else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        configuration = {}
    if not isinstance(configuration, dict):
        configuration = {}
    if not configuration.get("modelPath"):
        configuration["modelPath"] = preferences.get("selectedModelPath")
    return configuration


def saved_advanced_configuration(
    preferences: dict[str, Any],
    model_kind: str,
) -> dict[str, Any]:
    raw = preferences.get(f"modelAdvancedSettings.{model_kind}")
    try:
        configuration = json.loads(raw) if isinstance(raw, (bytes, str)) else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return configuration if isinstance(configuration, dict) else {}


def read_installed_manifest(model_path: Path) -> dict[str, Any]:
    manifest_path = model_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(
            f"Unable to read installed model manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise BenchmarkError(f"Installed model manifest is invalid: {manifest_path}")
    return manifest


def default_public_model(manifest: dict[str, Any]) -> str:
    model_kind = manifest.get("modelKind") or "deepseek-v4"
    public_model = DEFAULT_PUBLIC_MODELS.get(model_kind) or manifest.get("modelID")
    if not isinstance(public_model, str) or not public_model:
        raise BenchmarkError("The installed model manifest does not include a model ID")
    return public_model


def known_model_ids(manifest: dict[str, Any]) -> set[str]:
    values = {default_public_model(manifest), manifest.get("modelID")}
    return {value for value in values if isinstance(value, str) and value}


def find_installed_model(model_root: Path, model: str) -> Path:
    try:
        children = sorted(
            (
                path
                for path in model_root.iterdir()
                if path.is_dir()
                and not path.name.startswith(".")
                and path.suffix != ".partial"
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        children = []
    available = []
    for candidate in [model_root, *children]:
        try:
            manifest = read_installed_manifest(candidate)
            model_ids = known_model_ids(manifest)
        except BenchmarkError:
            continue
        available.extend(sorted(model_ids))
        if model in model_ids:
            return candidate.resolve()
    detail = f" Available: {', '.join(dict.fromkeys(available))}." if available else ""
    raise BenchmarkError(
        f"Installed model '{model}' was not found in '{model_root}'.{detail} "
        "Install it in Whallm or use --model-path PATH."
    )


def launch_configuration(
    requested_model_path: Path | None,
    requested_model: str | None,
) -> tuple[Path, str, dict[str, Any]]:
    preferences = app_preferences()
    saved = saved_server_configuration(preferences)
    raw_path = requested_model_path
    saved_path = saved.get("modelPath")
    if raw_path is None and requested_model and saved_path:
        try:
            saved_manifest = read_installed_manifest(
                Path(saved_path).expanduser().resolve()
            )
            saved_ids = known_model_ids(saved_manifest)
            if isinstance(saved.get("publicModel"), str):
                saved_ids.add(saved["publicModel"])
            if requested_model in saved_ids:
                raw_path = Path(saved_path)
        except BenchmarkError:
            pass
    if raw_path is None and requested_model:
        model_root = Path(
            preferences.get("modelLibraryRoot") or Path.home() / ".dsmodel"
        ).expanduser().resolve()
        raw_path = find_installed_model(model_root, requested_model)
    if raw_path is None:
        raw_path = saved_path
    if not raw_path:
        raise BenchmarkError(
            "No Server is running and no installed model is selected. "
            "Select a model in Whallm or use --model-path PATH."
        )
    model_path = Path(raw_path).expanduser().resolve()
    manifest = read_installed_manifest(model_path)
    uses_saved_settings = bool(
        saved_path and Path(saved_path).expanduser().resolve() == model_path
    )
    public_model = (
        saved.get("publicModel") if uses_saved_settings else None
    ) or default_public_model(manifest)
    if requested_model:
        known_ids = known_model_ids(manifest) | {public_model}
        if requested_model not in known_ids:
            raise BenchmarkError(
                f"Model '{requested_model}' does not match installed model "
                f"'{public_model}'."
            )
        public_model = requested_model
    configuration = (
        saved
        if uses_saved_settings
        else saved_advanced_configuration(
            preferences,
            manifest.get("modelKind") or "deepseek-v4",
        )
    )
    return model_path, public_model, configuration


def server_command(
    model_path: Path,
    public_model: str,
    host: str,
    port: int,
    configuration: dict[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "deepseek_v4_ssd.server",
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--public-model",
        public_model,
    ]
    options = (
        ("slots", "--slots"),
        ("readWorkers", "--read-workers"),
        ("memoryLimitGiB", "--memory-limit-gib"),
        ("prefillStepSize", "--prefill-step-size"),
        ("promptCacheEntries", "--prompt-cache-entries"),
        ("promptCacheMemoryGiB", "--prompt-cache-memory-gib"),
        ("dsparkSlots", "--dspark-slots"),
        ("dsparkConfidenceThreshold", "--dspark-confidence-threshold"),
        ("defaultMaxTokens", "--default-max-tokens"),
        ("defaultTemperature", "--default-temperature"),
        ("defaultTopP", "--default-top-p"),
        ("defaultTopK", "--default-top-k"),
    )
    for key, option in options:
        if configuration.get(key) is not None:
            command += [option, str(configuration[key])]
    if configuration.get("layerMajorPrefill") is False:
        command.append("--no-layer-major-prefill")
    if configuration.get("bf16KVCache"):
        command.append("--bf16-kv-cache")
    if configuration.get("dsparkEnabled"):
        command.append("--dspark")
    power_limit = configuration.get("powerSavingLimitGBps")
    if power_limit in {0.5, 1, 2, 3, 5, 10, 25}:
        command += ["--power-saving-limit-gbps", str(power_limit)]
    warmup_path = configuration.get("warmupPromptPath")
    if warmup_path:
        command += ["--warmup-prompt-file", str(warmup_path)]
    return command


def start_local_server(
    project_root: Path,
    command: list[str],
    api_key: str,
) -> tuple[subprocess.Popen[str], TextIO]:
    environment = os.environ.copy()
    runtime_path = str(project_root / "runtime")
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{runtime_path}{os.pathsep}{current_python_path}"
        if current_python_path
        else runtime_path
    )
    if api_key:
        environment["DEEPSEEK_API_KEY"] = api_key
    else:
        environment.pop("DEEPSEEK_API_KEY", None)
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        log.close()
        raise BenchmarkError(f"Unable to start Server: {error}") from error
    return process, log


def server_log_tail(log: TextIO, limit: int = 4_000) -> str:
    log.flush()
    log.seek(0)
    return log.read()[-limit:].strip()


def wait_for_server(
    process: subprocess.Popen[str],
    log: TextIO,
    base_url: str,
    api_key: str,
    timeout: float,
) -> list[str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = server_log_tail(log)
            suffix = f"\n{detail}" if detail else ""
            raise BenchmarkError(f"Server stopped during startup.{suffix}")
        try:
            return available_models(base_url, api_key, 2)
        except BenchmarkError as error:
            last_error = str(error)
        time.sleep(0.25)
    detail = server_log_tail(log) or last_error
    suffix = f"\n{detail}" if detail else ""
    raise BenchmarkError(f"Server did not become ready within {timeout:g} seconds.{suffix}")


def stop_local_server(process: subprocess.Popen[str], log: TextIO) -> None:
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    log.close()


def encode_prompt(tokenizer: Any, text: str) -> list[int]:
    add_special_tokens = tokenizer.bos_token is None or not text.startswith(
        tokenizer.bos_token
    )
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def build_exact_prompt(
    tokenizer: Any,
    target_tokens: int,
    marker: str,
) -> tuple[str, list[int]]:
    source = f"API benchmark {marker}." + (" test" * (target_tokens + 128))
    source_tokens = list(tokenizer.encode(source, add_special_tokens=False))
    body_tokens = target_tokens
    for _ in range(12):
        prompt = tokenizer.decode(
            source_tokens[:body_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        tokens = encode_prompt(tokenizer, prompt)
        if len(tokens) == target_tokens:
            return prompt, tokens
        body_tokens += target_tokens - len(tokens)
        if body_tokens < 1 or body_tokens > len(source_tokens):
            break
    raise BenchmarkError(f"Unable to construct a {target_tokens}-token prompt")


def load_tokenizer(model_path: str) -> Any:
    try:
        import logging

        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from transformers import AutoTokenizer
    except ImportError as error:
        raise BenchmarkError(
            "Transformers is unavailable. Run this script with .venv/bin/python."
        ) from error
    tokenizer_path = Path(model_path).expanduser() / "tokenizer"
    if not tokenizer_path.is_dir():
        raise BenchmarkError(f"Tokenizer directory is missing: {tokenizer_path}")
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as error:
        raise BenchmarkError(f"Unable to load tokenizer: {error}") from error


def stream_completion(
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    timeout: float,
    result: dict[str, Any],
    done: threading.Event,
) -> None:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_output_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": True,
        }
    ).encode()
    request = Request(
        url,
        data=payload,
        headers=request_headers(api_key, json_body=True),
        method="POST",
    )
    started = time.perf_counter()
    first_output_at = None
    output = []
    try:
        with urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                for choice in event.get("choices", []):
                    text = choice.get("text", "")
                    if text:
                        if first_output_at is None:
                            first_output_at = time.perf_counter()
                        output.append(text)
        result.update(
            {
                "output_text": "".join(output),
                "wall_seconds": time.perf_counter() - started,
                "client_first_output_seconds": (
                    first_output_at - started if first_output_at is not None else 0.0
                ),
            }
        )
    except Exception as error:
        result["error"] = error_message(error)
    finally:
        done.set()


def run_request(
    base_url: str,
    status_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    poll_interval: float,
    timeout: float,
) -> dict[str, Any]:
    initial = get_json(status_url, api_key, timeout)
    if initial.get("performance", {}).get("generating"):
        raise BenchmarkError("The server is busy. Wait for the current request to finish.")

    result: dict[str, Any] = {}
    done = threading.Event()
    worker = threading.Thread(
        target=stream_completion,
        args=(
            api_endpoint(base_url, "completions"),
            api_key,
            model,
            prompt,
            max_output_tokens,
            timeout,
            result,
            done,
        ),
        daemon=True,
    )
    worker.start()
    active_memory_samples = []
    poll_error = None
    while not done.is_set():
        try:
            status = get_json(status_url, api_key, min(timeout, 5))
            performance = status.get("performance", {})
            if performance.get("generating"):
                active_memory_samples.append(
                    int(performance.get("active_memory_bytes", 0))
                )
        except BenchmarkError as error:
            poll_error = error
            break
        done.wait(poll_interval)
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise BenchmarkError("The API request did not finish before the timeout")
    if poll_error is not None:
        raise poll_error
    if "error" in result:
        raise BenchmarkError(f"Completion request failed: {result['error']}")

    final = get_json(status_url, api_key, timeout)
    performance = final.get("performance", {})
    active_memory_samples.append(int(performance.get("active_memory_bytes", 0)))
    result.update(
        {
            "input_tokens": int(performance.get("runtime_prompt_tokens", 0)),
            "output_tokens": int(performance.get("runtime_generation_tokens", 0)),
            "prompt_cache_reused_tokens": int(
                performance.get("prompt_cache_reused_tokens", 0)
            ),
            "ttft_seconds": float(
                performance.get("time_to_first_token_seconds", 0)
            ),
            "prefill_tokens_per_second": float(
                performance.get("prefill_tokens_per_second", 0)
            ),
            "decode_tokens_per_second": float(
                performance.get("decode_tokens_per_second", 0)
            ),
            "peak_active_memory_bytes": max(active_memory_samples, default=0),
        }
    )
    return result


def nearest_rank_p95(values: list[float]) -> float:
    if not values:
        raise ValueError("P95 requires at least one value")
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.95) - 1]


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for target in sorted({int(run["target_input_tokens"]) for run in runs}):
        group = [run for run in runs if run["target_input_tokens"] == target]
        metrics = {}
        for key in (
            "ttft_seconds",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "peak_active_memory_bytes",
        ):
            values = [float(run[key]) for run in group]
            metrics[key] = {"peak": max(values), "p95": nearest_rank_p95(values)}
        actual = [int(run["input_tokens"]) for run in group]
        summaries.append(
            {
                "target_input_tokens": target,
                "actual_input_tokens_min": min(actual),
                "actual_input_tokens_max": max(actual),
                "run_count": len(group),
                "metrics": metrics,
            }
        )
    return summaries


def input_label(tokens: int) -> str:
    return f"{tokens // 1_024}K" if tokens % 1_024 == 0 else f"{tokens:,}"


def render_ascii_table(summaries: list[dict[str, Any]]) -> str:
    headers = (
        "Input",
        "Actual",
        "Stat",
        "TTFT (s)",
        "Prefill tok/s",
        "Decode tok/s",
        "Memory (GiB)",
    )
    rows = []
    for summary in summaries:
        minimum = summary["actual_input_tokens_min"]
        maximum = summary["actual_input_tokens_max"]
        actual = f"{minimum:,}" if minimum == maximum else f"{minimum:,}-{maximum:,}"
        metrics = summary["metrics"]
        for statistic in ("peak", "p95"):
            rows.append(
                (
                    input_label(summary["target_input_tokens"]),
                    actual,
                    "Peak" if statistic == "peak" else "P95",
                    f"{metrics['ttft_seconds'][statistic]:.2f}",
                    f"{metrics['prefill_tokens_per_second'][statistic]:.1f}",
                    f"{metrics['decode_tokens_per_second'][statistic]:.1f}",
                    f"{metrics['peak_active_memory_bytes'][statistic] / GIB:.2f}",
                )
            )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def line(values: tuple[str, ...]) -> str:
        return "|" + "|".join(
            f" {value:<{width}} " for value, width in zip(values, widths)
        ) + "|"

    return "\n".join([border, line(headers), border, *(line(row) for row in rows), border])


def command_output(project_root: Path, *command: str) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_metadata(project_root: Path) -> dict[str, Any]:
    diff = command_output(project_root, "git", "diff", "--binary", "HEAD")
    status = command_output(project_root, "git", "status", "--porcelain")
    return {
        "commit": command_output(project_root, "git", "rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "working_tree_diff_sha256": (
            hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None
        ),
        "benchmark_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }


def environment_metadata(project_root: Path, base_url: str) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "hardware_model": command_output(project_root, "sysctl", "-n", "hw.model"),
        "unified_memory_bytes": command_output(
            project_root, "sysctl", "-n", "hw.memsize"
        ),
        "python": platform.python_version(),
        "api_base_url": base_url,
    }


def installed_model_metadata(model_path: str) -> dict[str, Any]:
    manifest_path = Path(model_path) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"manifest_available": False}
    return {
        "manifest_available": True,
        "model_id": manifest.get("modelID"),
        "model_kind": manifest.get("modelKind"),
        "revision": manifest.get("revision"),
        "format_version": manifest.get("formatVersion"),
    }


def save_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def default_output_path(project_root: Path, model: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d-%H%M%S")
    model_name = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return project_root / "docs" / "benchmarks" / f"{timestamp}-api-{model_name}.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a Whallm model through its local API."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
        help="OpenAI API base URL (default: %(default)s)",
    )
    parser.add_argument("--model", help="Model ID; prompts for a model when omitted")
    parser.add_argument(
        "--model-path",
        type=Path,
        help=(
            "Installed model used to start Server; defaults to the model selected "
            "in Whallm"
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_token_count,
        default=list(DEFAULT_INPUT_TOKENS),
        metavar="TOKENS",
        help="Input sizes (default: 1K 4K 16K 32K)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Runs per input size (default: %(default)s)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=64,
        help="Output token limit per run (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Status polling interval in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3_600,
        help="Timeout for one request in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=float,
        default=900,
        help="Time to wait for an automatically started Server (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, help="JSON artifact path")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List API models and exit",
    )
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be at least 1")
    if arguments.max_output_tokens < 2:
        parser.error("--max-output-tokens must be at least 2")
    if arguments.poll_interval <= 0:
        parser.error("--poll-interval must be greater than 0")
    if arguments.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if arguments.server_start_timeout <= 0:
        parser.error("--server-start-timeout must be greater than 0")
    arguments.sizes = list(dict.fromkeys(arguments.sizes))
    return arguments


def run_connected_benchmark(
    arguments: argparse.Namespace,
    project_root: Path,
    api_key: str,
    base_url: str,
    server_started: bool,
) -> int:
    models = available_models(base_url, api_key, arguments.timeout)
    if arguments.list_models:
        print("\n".join(models))
        return 0
    model = select_model(models, arguments.model)
    status_url = status_endpoint(base_url)
    status = get_json(status_url, api_key, arguments.timeout)
    if status.get("model") != model:
        raise BenchmarkError(
            f"Start '{model}' in Whallm before running this benchmark."
        )
    model_path = status.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise BenchmarkError("The status response does not include model_path")
    if arguments.model_path:
        expected_path = arguments.model_path.expanduser().resolve()
        if Path(model_path).expanduser().resolve() != expected_path:
            raise BenchmarkError(
                f"Server is using '{model_path}', not requested model "
                f"'{expected_path}'. Stop the current Server and try again."
            )
    tokenizer = load_tokenizer(model_path)
    output_path = (
        arguments.output.expanduser().resolve()
        if arguments.output
        else default_output_path(project_root, model)
    )
    session_id = uuid.uuid4().hex
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "evidence_kind": "exploratory API benchmark",
        "status": "running",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source": source_metadata(project_root),
        "environment": environment_metadata(project_root, base_url),
        "installed_model": installed_model_metadata(model_path),
        "server": {
            "model": status.get("model"),
            "source_model": status.get("source_model"),
            "started_by_benchmark": server_started,
            "runtime_configuration": status.get("runtime", {}),
        },
        "workload": {
            "endpoint": "/v1/completions",
            "target_input_tokens": arguments.sizes,
            "runs_per_size": arguments.runs,
            "max_output_tokens": arguments.max_output_tokens,
            "temperature": 0,
            "top_p": 1,
            "poll_interval_seconds": arguments.poll_interval,
            "prompt": "unique prefix followed by repeated ' test' tokens",
            "prompt_cache_state": "fresh unique input; prompt cache not cleared",
            "filesystem_cache_state": "not purged",
            "memory_metric": (
                "maximum performance.active_memory_bytes observed during each request"
            ),
            "p95_method": "nearest rank across runs",
        },
        "evidence_limits": [
            "The API does not expose exact generated token IDs.",
            "output_text_sha256 hashes streamed text, not generated token IDs.",
            "Memory is MLX active memory, not process RSS.",
        ],
        "runs": [],
        "summary": [],
    }
    save_artifact(output_path, artifact)

    total_requests = len(arguments.sizes) * arguments.runs
    print(f"Model: {model}")
    print(f"Plan: {len(arguments.sizes)} input sizes x {arguments.runs} runs")
    print(f"Requests: {total_requests}; output limit: {arguments.max_output_tokens} tokens")
    print(f"Result file: {output_path}")
    if arguments.runs < 20:
        print("Note: nearest-rank P95 equals Peak when fewer than 20 runs are used.")

    request_index = 0
    try:
        for target_tokens in arguments.sizes:
            for run_index in range(1, arguments.runs + 1):
                request_index += 1
                marker = f"{session_id}-{target_tokens}-{run_index}"
                prompt, prompt_tokens = build_exact_prompt(
                    tokenizer,
                    target_tokens,
                    marker,
                )
                print(
                    f"[{request_index}/{total_requests}] "
                    f"{input_label(target_tokens)} run {run_index}/{arguments.runs}...",
                    flush=True,
                )
                result = run_request(
                    base_url,
                    status_url,
                    api_key,
                    model,
                    prompt,
                    arguments.max_output_tokens,
                    arguments.poll_interval,
                    arguments.timeout,
                )
                if result["input_tokens"] != target_tokens:
                    raise BenchmarkError(
                        "Tokenizer mismatch: "
                        f"requested {target_tokens}, server reported {result['input_tokens']}"
                    )
                output_text = result.pop("output_text")
                result.update(
                    {
                        "target_input_tokens": target_tokens,
                        "run": run_index,
                        "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt_token_sha256": token_sha256(prompt_tokens),
                        "output_text_sha256": hashlib.sha256(
                            output_text.encode()
                        ).hexdigest(),
                    }
                )
                artifact["runs"].append(result)
                artifact["summary"] = summarize(artifact["runs"])
                save_artifact(output_path, artifact)
                print(
                    "  "
                    f"TTFT {result['ttft_seconds']:.2f}s; "
                    f"Prefill {result['prefill_tokens_per_second']:.1f} tok/s; "
                    f"Decode {result['decode_tokens_per_second']:.1f} tok/s; "
                    f"Memory {result['peak_active_memory_bytes'] / GIB:.2f} GiB"
                )
    except BaseException as error:
        artifact["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        artifact["error"] = str(error)
        save_artifact(output_path, artifact)
        raise

    artifact["status"] = "completed"
    artifact["completed_at"] = datetime.now().astimezone().isoformat()
    artifact["summary"] = summarize(artifact["runs"])
    save_artifact(output_path, artifact)
    print()
    print(render_ascii_table(artifact["summary"]))
    print(f"\nSaved: {output_path}")
    return 0


def run_benchmark(arguments: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = arguments.base_url.rstrip("/")
    process = None
    log = None
    server_started = False
    try:
        if not server_is_listening(base_url):
            host, port = local_server_address(base_url)
            model_path, public_model, configuration = launch_configuration(
                arguments.model_path,
                arguments.model,
            )
            command = server_command(
                model_path,
                public_model,
                host,
                port,
                configuration,
            )
            print(f"Server is not running. Starting {public_model}...", flush=True)
            process, log = start_local_server(project_root, command, api_key)
            started_at = time.monotonic()
            wait_for_server(
                process,
                log,
                base_url,
                api_key,
                arguments.server_start_timeout,
            )
            server_started = True
            print(f"Server ready in {time.monotonic() - started_at:.1f}s.", flush=True)
        return run_connected_benchmark(
            arguments,
            project_root,
            api_key,
            base_url,
            server_started,
        )
    finally:
        if process is not None and log is not None:
            print("Stopping Server...", flush=True)
            stop_local_server(process, log)


def main() -> int:
    arguments = parse_arguments()
    try:
        return run_benchmark(arguments)
    except KeyboardInterrupt:
        print(
            "\nBenchmark interrupted. Completed runs, if any, remain in the result file.",
            file=sys.stderr,
        )
        return 130
    except BenchmarkError as error:
        print(f"Benchmark stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
