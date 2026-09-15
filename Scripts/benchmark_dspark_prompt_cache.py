from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx

from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.model import RuntimeConfig


EXPECTED_OUTPUT_TOKEN_SHA256 = (
    "03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def _runtime_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_state(project_root: Path) -> dict:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
        "working_tree_dirty": bool(
            _command_output(["git", "status", "--porcelain"], project_root)
        ),
        "tracked_diff_sha256_at_run": hashlib.sha256(diff).hexdigest(),
        "runtime_python_tree_sha256": _runtime_tree_sha256(
            project_root / "runtime" / "deepseek_v4_ssd"
        ),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _config(cache_root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        persistent_prompt_cache=True,
        prompt_cache_directory=str(cache_root),
        layer_major_prefill=False,
        dspark_enabled=True,
        dspark_prompt_cache=True,
        dspark_fallback_enabled=False,
        expert_file_cache_policy="bypass",
    )


def _run_request(runtime: ModelRuntime, prompt: str, run_id: str) -> dict:
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    started = time.perf_counter()
    pieces = list(
        runtime.stream(
            prompt,
            GenerationOptions(max_tokens=8, temperature=0, top_p=1),
        )
    )
    wall_seconds = time.perf_counter() - started
    tokens = [int(piece.token) for piece in pieces]
    metrics = runtime.metrics.snapshot()
    main_bytes = int(metrics["request_expert_bytes_read"])
    draft_bytes = int(metrics["dspark_draft_expert_bytes_read"])
    return {
        "id": run_id,
        "generated_token_ids": tokens,
        "token_sha256": _token_sha256(tokens),
        "wall_seconds": wall_seconds,
        "main_logical_expert_bytes": main_bytes,
        "draft_logical_expert_bytes": draft_bytes,
        "combined_logical_expert_bytes": main_bytes + draft_bytes,
        "metrics": metrics,
    }


def _gate(runs: list[dict], expected_prompt_tokens: int) -> dict:
    first, memory, persistent = runs
    expected_reused = expected_prompt_tokens - 1
    source_exact = [
        first["metrics"]["dspark_prompt_cache_source"] == "none",
        memory["metrics"]["dspark_prompt_cache_source"] == "memory",
        persistent["metrics"]["dspark_prompt_cache_source"] == "persistent",
    ]
    reused_exact = [
        int(first["metrics"]["prompt_cache_reused_tokens"]) == 0,
        int(memory["metrics"]["prompt_cache_reused_tokens"]) == expected_reused,
        int(persistent["metrics"]["prompt_cache_reused_tokens"])
        == expected_reused,
    ]
    output_exact = [
        run["token_sha256"] == EXPECTED_OUTPUT_TOKEN_SHA256 for run in runs
    ]
    first_bytes = first["combined_logical_expert_bytes"]
    byte_checks = [
        run["main_logical_expert_bytes"]
        <= first["main_logical_expert_bytes"]
        and run["draft_logical_expert_bytes"]
        <= first["draft_logical_expert_bytes"]
        and run["combined_logical_expert_bytes"] <= first_bytes * 0.5
        for run in (memory, persistent)
    ]
    criteria = {
        "reuse_sources_exact": all(source_exact),
        "reused_token_counts_exact": all(reused_exact),
        "all_output_token_hashes_exact": all(output_exact),
        "memory_combined_logical_expert_bytes_reduced_at_least_50_percent": (
            byte_checks[0]
        ),
        "persistent_combined_logical_expert_bytes_reduced_at_least_50_percent": (
            byte_checks[1]
        ),
        "all_cache_writes_succeeded": all(
            int(run["metrics"]["prompt_cache_write_errors"]) == 0
            for run in runs
        ),
    }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "memory_combined_byte_change_fraction": (
            memory["combined_logical_expert_bytes"] / first_bytes - 1
            if first_bytes
            else None
        ),
        "persistent_combined_byte_change_fraction": (
            persistent["combined_logical_expert_bytes"] / first_bytes - 1
            if first_bytes
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate atomic target and DSpark prompt-context snapshots"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    manifest_path = Path(arguments.prompt_manifest).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    prompt_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_entry = next(
        (
            entry
            for entry in prompt_manifest["prompts"]
            if entry["name"] == "repeated"
        ),
        None,
    )
    if prompt_entry is None:
        parser.error("prompt manifest does not contain repeated")
    prompt_path = manifest_path.parent / prompt_entry["file"]
    if _sha256(prompt_path) != prompt_entry["text_sha256"]:
        raise RuntimeError("prompt text SHA-256 does not match manifest")
    prompt = prompt_path.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="dspark-prompt-cache-") as temporary:
        cache_root = Path(temporary)
        runtime_a = ModelRuntime.open(str(model), _config(cache_root))
        try:
            prompt_tokens = runtime_a._encode_prompt(prompt)
            if _token_sha256(prompt_tokens) != prompt_entry["prompt_token_sha256"]:
                raise RuntimeError("prompt token SHA-256 does not match manifest")
            runs = [
                _run_request(runtime_a, prompt, "runtime-a-first"),
                _run_request(runtime_a, prompt, "runtime-a-memory"),
            ]
            installed = runtime_a.installed
            target_layers = list(installed.dspark_target_layer_ids)
            revision = installed.revision
        finally:
            runtime_a.close()

        mx.clear_cache()
        gc.collect()
        runtime_b = ModelRuntime.open(str(model), _config(cache_root))
        try:
            runs.append(_run_request(runtime_b, prompt, "runtime-b-persistent"))
        finally:
            runtime_b.close()

        cache_directory = cache_root / revision
        metadata_paths = sorted(cache_directory.glob("*.dspark.v3.json"))
        if len(metadata_paths) != 1:
            raise RuntimeError(
                f"expected one DSpark prompt snapshot, found {len(metadata_paths)}"
            )
        cache_metadata_path = metadata_paths[0]
        cache_metadata = json.loads(
            cache_metadata_path.read_text(encoding="utf-8")
        )
        cache_data_path = cache_directory / cache_metadata["data"]
        metadata_contract = {
            "mode_exact": cache_metadata.get("mode") == "dspark",
            "format_exact": int(cache_metadata.get("format", 0)) == 3,
            "revision_exact": cache_metadata.get("revision") == revision,
            "target_layers_exact": cache_metadata.get("targetLayers")
            == target_layers,
            "tokens_exact": cache_metadata.get("tokens") == prompt_tokens[:-1],
            "data_file_present": cache_data_path.is_file(),
        }
        cache_bundle = {
            "metadata": cache_metadata,
            "metadata_sha256": _sha256(cache_metadata_path),
            "metadata_bytes": cache_metadata_path.stat().st_size,
            "data_sha256": _sha256(cache_data_path),
            "data_bytes": cache_data_path.stat().st_size,
            "contract": metadata_contract,
            "contract_passed": all(metadata_contract.values()),
        }

    gate = _gate(runs, len(prompt_tokens))
    gate["criteria"]["persistent_metadata_contract_exact"] = cache_bundle[
        "contract_passed"
    ]
    gate["passed"] = all(gate["criteria"].values())
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_kind": "functional integration gate",
        "formal_performance_result": False,
        "source": _source_state(project_root),
        "checkpoint": {
            "model_path": str(model),
            "model_id": prompt_manifest["model_id"],
            "revision": revision,
            "installed_manifest_sha256": _sha256(model / "manifest.json"),
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
            "chip": _command_output(["sysctl", "-n", "machdep.cpu.brand_string"], project_root),
            "memory_bytes": _command_output(["sysctl", "-n", "hw.memsize"], project_root),
        },
        "experiment": {
            "workload": "repeated",
            "prompt_file": prompt_entry["file"],
            "prompt_tokens": len(prompt_tokens),
            "prompt_token_sha256": prompt_entry["prompt_token_sha256"],
            "prompt_manifest_sha256": _sha256(manifest_path),
            "max_output_tokens": 8,
            "temperature": 0,
            "top_p": 1,
            "batch_size": 1,
            "dspark_prompt_cache": True,
            "persistent_prompt_cache": True,
            "dspark_fallback_enabled": False,
            "layer_major_prefill": False,
            "expert_file_cache_policy": "bypass",
            "cache_directory": "new isolated temporary directory",
            "run_order": [run["id"] for run in runs],
            "expected_output_token_sha256": EXPECTED_OUTPUT_TOKEN_SHA256,
        },
        "runs": runs,
        "persistent_bundle": cache_bundle,
        "gate": gate,
        "decision": (
            "atomic_prompt_context_reuse_functionally_validated_default_off"
            if gate["passed"]
            else "stop_atomic_prompt_context_reuse_candidate"
        ),
        "evidence_limits": [
            "This is a three-request functional gate, not a repeated-wave performance result.",
            "Request time and TTFT are descriptive because model reload, memory-cache state, and operating-system state are not balanced.",
            "Expert bytes are logical runtime reads; they are not physical SSD byte counters.",
            "The gate covers one checkpoint, one exact prompt prefix, batch size 1, one process lifetime, and one restart boundary.",
            "Passing does not enable DSpark, prompt-context reuse, or expert-file bypass by default.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    if not gate["passed"]:
        raise SystemExit("DSpark prompt-context snapshot gate failed")


if __name__ == "__main__":
    main()
