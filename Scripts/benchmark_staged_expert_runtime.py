from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


MODE_ORDERS = (
    ("control", "staged"),
    ("staged", "control"),
    ("control", "staged"),
    ("staged", "control"),
)
SUMMARY_METRICS = (
    "request_seconds",
    "external_wall_seconds",
    "time_to_first_token_seconds",
    "decode_tokens_per_second",
    "decode_latency_p50_seconds",
    "decode_latency_p95_seconds",
    "peak_memory_bytes",
    "request_expert_cache_hits",
    "request_expert_cache_misses",
    "request_expert_evictions",
    "request_expert_bytes_read",
    "request_staged_expert_reads",
    "request_staged_w13_bytes_read",
    "request_staged_w2_bytes_read",
    "request_staged_read_seconds",
    "request_staged_w2_wait_seconds",
    "request_staged_first_stage_submit_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(
        ",".join(map(str, tokens)).encode("utf-8")
    ).hexdigest()


def _runtime_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
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


def _sysctl(name: str, cwd: Path) -> str | None:
    return _command_output(["sysctl", "-n", name], cwd)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _change_fraction(control: float, candidate: float) -> float | None:
    return (candidate - control) / control if control else None


def _mode_summary(rows: list[dict]) -> dict:
    return {
        key: statistics.median(float(row["metrics"][key]) for row in rows)
        for key in SUMMARY_METRICS
    }


def _paired_changes(rows: list[dict]) -> list[dict]:
    pairs = []
    for wave in sorted({int(row["wave"]) for row in rows}):
        by_mode = {
            row["mode"]: row for row in rows if int(row["wave"]) == wave
        }
        control = by_mode["control"]
        staged = by_mode["staged"]
        pairs.append(
            {
                "wave": wave,
                "order": [
                    row["mode"]
                    for row in sorted(
                        by_mode.values(), key=lambda item: int(item["sequence"])
                    )
                ],
                "token_hash_exact": (
                    control["token_sha256"] == staged["token_sha256"]
                ),
                "logical_expert_bytes_not_increased": (
                    int(staged["metrics"]["request_expert_bytes_read"])
                    <= int(control["metrics"]["request_expert_bytes_read"])
                ),
                "evictions_not_increased": (
                    int(staged["metrics"]["request_expert_evictions"])
                    <= int(control["metrics"]["request_expert_evictions"])
                ),
                "change_fraction": {
                    key: _change_fraction(
                        float(control["metrics"][key]),
                        float(staged["metrics"][key]),
                    )
                    for key in (
                        "request_seconds",
                        "external_wall_seconds",
                        "decode_tokens_per_second",
                        "decode_latency_p95_seconds",
                        "peak_memory_bytes",
                        "request_expert_bytes_read",
                        "request_expert_evictions",
                    )
                },
            }
        )
    return pairs


def _runtime_gate(
    rows: list[dict],
    *,
    max_tokens: int,
    expert_blob_bytes: int,
    w13_bytes: int,
    w2_bytes: int,
) -> dict:
    control = [row for row in rows if row["mode"] == "control"]
    staged = [row for row in rows if row["mode"] == "staged"]
    pairs = _paired_changes(rows)
    token_hashes = {row["token_sha256"] for row in rows}
    prompt_hashes = {row["prompt_token_sha256"] for row in rows}

    staged_byte_closure = []
    for row in staged:
        metrics = row["metrics"]
        reads = int(metrics["request_staged_expert_reads"])
        staged_byte_closure.append(
            reads > 0
            and int(metrics["request_staged_w13_bytes_read"])
            == reads * w13_bytes
            and int(metrics["request_staged_w2_bytes_read"])
            == reads * w2_bytes
            and int(metrics["request_staged_w13_bytes_read"])
            + int(metrics["request_staged_w2_bytes_read"])
            == reads * expert_blob_bytes
        )

    correctness_criteria = {
        "four_complete_mode_pairs": len(control) == 4 and len(staged) == 4,
        "all_prompt_token_hashes_exact": len(prompt_hashes) == 1,
        "all_output_token_hashes_exact": len(token_hashes) == 1,
        "all_runs_reached_output_budget": all(
            int(row["generated_tokens"]) == max_tokens for row in rows
        ),
        "all_pairs_have_exact_tokens": all(
            pair["token_hash_exact"] for pair in pairs
        ),
        "logical_expert_bytes_never_increased": all(
            pair["logical_expert_bytes_not_increased"] for pair in pairs
        ),
        "expert_evictions_never_increased": all(
            pair["evictions_not_increased"] for pair in pairs
        ),
        "control_has_no_staged_reads": all(
            int(row["metrics"]["request_staged_expert_reads"]) == 0
            and int(row["metrics"]["request_staged_w13_bytes_read"]) == 0
            and int(row["metrics"]["request_staged_w2_bytes_read"]) == 0
            for row in control
        ),
        "staged_read_accounting_closes": all(staged_byte_closure),
        "configuration_exact": all(
            row["configuration"]
            == {
                "staged_expert_streaming": row["mode"] == "staged",
                "ready_expert_decode": True,
                "expert_file_cache_policy": "bypass",
                "layer_major_prefill": False,
                "persistent_prompt_cache": False,
                "dspark_enabled": False,
            }
            for row in rows
        ),
    }
    correctness_passed = all(correctness_criteria.values())

    paired_medians = {
        key: statistics.median(
            float(pair["change_fraction"][key]) for pair in pairs
        )
        for key in (
            "request_seconds",
            "external_wall_seconds",
            "decode_tokens_per_second",
            "decode_latency_p95_seconds",
            "peak_memory_bytes",
            "request_expert_bytes_read",
            "request_expert_evictions",
        )
    }
    performance_criteria = {
        "request_or_decode_improves_at_least_5_percent": (
            paired_medians["request_seconds"] <= -0.05
            or paired_medians["decode_tokens_per_second"] >= 0.05
        ),
        "decode_p95_regresses_no_more_than_5_percent": (
            paired_medians["decode_latency_p95_seconds"] <= 0.05
        ),
        "peak_memory_increases_no_more_than_5_percent": (
            paired_medians["peak_memory_bytes"] <= 0.05
        ),
    }
    performance_passed = correctness_passed and all(
        performance_criteria.values()
    )
    return {
        "shared_prompt_token_sha256": next(iter(prompt_hashes), None),
        "shared_output_token_sha256": next(iter(token_hashes), None),
        "correctness": {
            "criteria": correctness_criteria,
            "passed": correctness_passed,
        },
        "performance": {
            "paired_median_change_fraction": paired_medians,
            "criteria": performance_criteria,
            "passed": performance_passed,
        },
        "paired_runs": pairs,
        "mode_medians": {
            "control": _mode_summary(control),
            "staged": _mode_summary(staged),
        },
        "decision": (
            "continue_multi_workload_long_decode_default_off"
            if performance_passed
            else "stop_runtime_candidate_keep_default_off"
        ),
    }


def _source_state(project_root: Path, protocol: Path) -> dict:
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
        "predeclared_protocol_sha256": _sha256(protocol),
    }


def _run_worker(arguments: argparse.Namespace) -> None:
    import mlx.core as mx

    from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
    from deepseek_v4_ssd.model import RuntimeConfig

    model = Path(arguments.model).expanduser().resolve()
    prompt_path = Path(arguments.prompt_file).expanduser().resolve()
    output = Path(arguments.worker_output).expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    staged = arguments.mode == "staged"
    config = RuntimeConfig(
        layer_major_prefill=False,
        persistent_prompt_cache=False,
        dspark_enabled=False,
        expert_file_cache_policy="bypass",
        ready_expert_decode=True,
        staged_expert_streaming=staged,
    )
    runtime = ModelRuntime.open(str(model), config)
    try:
        prompt_tokens = runtime._encode_prompt(prompt)
        prompt_token_sha256 = _token_sha256(prompt_tokens)
        if len(prompt_tokens) != arguments.expected_prompt_tokens:
            raise RuntimeError(
                "worker prompt token count does not match the manifest"
            )
        if prompt_token_sha256 != arguments.expected_prompt_token_sha256:
            raise RuntimeError("worker prompt token hash does not match the manifest")
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
        started = time.perf_counter()
        pieces = list(
            runtime.stream(
                prompt,
                GenerationOptions(
                    max_tokens=arguments.max_tokens,
                    temperature=0,
                    top_p=1,
                ),
            )
        )
        external_wall_seconds = time.perf_counter() - started
        generated_token_ids = [int(piece.token) for piece in pieces]
        snapshot = runtime.metrics.snapshot()
        metrics = {
            key: (
                external_wall_seconds
                if key == "external_wall_seconds"
                else snapshot[key]
            )
            for key in SUMMARY_METRICS
        }
        row = {
            "mode": arguments.mode,
            "configuration": {
                "staged_expert_streaming": staged,
                "ready_expert_decode": config.ready_expert_decode,
                "expert_file_cache_policy": config.expert_file_cache_policy,
                "layer_major_prefill": config.layer_major_prefill,
                "persistent_prompt_cache": config.persistent_prompt_cache,
                "dspark_enabled": config.dspark_enabled,
            },
            "prompt_tokens": len(prompt_tokens),
            "prompt_token_sha256": prompt_token_sha256,
            "generated_tokens": len(generated_token_ids),
            "generated_token_ids": generated_token_ids,
            "token_sha256": _token_sha256(generated_token_ids),
            "expert_file_direct_io_alignment_bytes": (
                runtime.expert_cache.direct_io_alignment
            ),
            "expert_resident_slots": runtime.expert_cache.resident_count,
            "metrics": metrics,
        }
    finally:
        runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_repeated_prompt(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt = next(
        (item for item in manifest["prompts"] if item["name"] == "repeated"),
        None,
    )
    if prompt is None:
        raise ValueError("prompt manifest does not contain repeated")
    prompt = dict(prompt)
    path = manifest_path.parent / prompt["file"]
    if not path.is_file():
        raise ValueError(f"prompt file is missing: {path}")
    if _sha256(path) != prompt["text_sha256"]:
        raise ValueError(f"prompt text hash mismatch: {path}")
    if int(prompt["target_tokens"]) != 128:
        raise ValueError("repeated prompt is not the predeclared 128 tokens")
    prompt["path"] = str(path)
    prompt["model_id"] = manifest["model_id"]
    prompt["revision"] = manifest["revision"]
    return prompt


def _run_fresh_process(
    *,
    project_root: Path,
    python: Path,
    model: Path,
    prompt: dict,
    mode: str,
    wave: int,
    sequence: int,
    max_tokens: int,
    raw_directory: Path,
) -> dict:
    run_id = f"repeated-w{wave:02d}-{sequence:02d}-{mode}"
    result_path = raw_directory / f"{run_id}.json"
    stdout_path = raw_directory / f"{run_id}.stdout.txt"
    stderr_path = raw_directory / f"{run_id}.stderr.txt"
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(model),
        "--prompt-file",
        prompt["path"],
        "--expected-prompt-tokens",
        str(prompt["target_tokens"]),
        "--expected-prompt-token-sha256",
        prompt["prompt_token_sha256"],
        "--mode",
        mode,
        "--max-tokens",
        str(max_tokens),
        "--worker-output",
        str(result_path),
    ]
    environment = os.environ.copy()
    runtime = str(project_root / "runtime")
    environment["PYTHONPATH"] = (
        runtime
        if not environment.get("PYTHONPATH")
        else runtime + os.pathsep + environment["PYTHONPATH"]
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    if completed.returncode or not result_path.is_file():
        raise RuntimeError(
            f"{run_id} failed ({completed.returncode}):\n{stderr_text[-2_000:]}"
        )
    row = json.loads(result_path.read_text(encoding="utf-8"))
    row.update(
        {
            "id": run_id,
            "wave": wave,
            "sequence": sequence,
            "raw_process": {
                "command": command,
                "returncode": completed.returncode,
                "result_sha256": _sha256(result_path),
                "stdout_sha256": _sha256(stdout_path),
                "stderr_sha256": _sha256(stderr_path),
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        }
    )
    return row


def _run_orchestrator(arguments: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    protocol = project_root / "research" / "STAGED_EXPERT_STREAMING_2026-08-27.md"
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if raw_directory.exists() and any(raw_directory.iterdir()):
        raise ValueError(f"raw directory must be new or empty: {raw_directory}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt = _load_repeated_prompt(prompt_manifest)
    installed_manifest = json.loads((model / "manifest.json").read_text())
    expert_blob_bytes = int(installed_manifest["expertBlobSize"])
    w13_bytes = 8_912_896
    w2_bytes = 4_456_448
    if w13_bytes + w2_bytes != expert_blob_bytes:
        raise ValueError("predeclared staged split does not match expert blob size")

    rows = []
    total = sum(len(order) for order in MODE_ORDERS)
    completed = 0
    for wave, order in enumerate(MODE_ORDERS, start=1):
        for sequence, mode in enumerate(order, start=1):
            completed += 1
            print(
                f"[{completed}/{total}] wave={wave} mode={mode}",
                flush=True,
            )
            rows.append(
                _run_fresh_process(
                    project_root=project_root,
                    python=python,
                    model=model,
                    prompt=prompt,
                    mode=mode,
                    wave=wave,
                    sequence=sequence,
                    max_tokens=arguments.max_tokens,
                    raw_directory=raw_directory,
                )
            )

    gate = _runtime_gate(
        rows,
        max_tokens=arguments.max_tokens,
        expert_blob_bytes=expert_blob_bytes,
        w13_bytes=w13_bytes,
        w2_bytes=w2_bytes,
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "exploratory_staged_expert_runtime_abba",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": _sysctl("hw.model", project_root),
            "chip": _sysctl("machdep.cpu.brand_string", project_root),
            "memory_bytes": int(_sysctl("hw.memsize", project_root) or 0),
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
        },
        "checkpoint": {
            "installed_model": str(model),
            "model_id": installed_manifest["modelID"],
            "revision": installed_manifest["revision"],
            "installed_manifest_sha256": _sha256(model / "manifest.json"),
            "layer_count": installed_manifest["layerCount"],
            "expert_count": installed_manifest["expertCount"],
            "selected_expert_count": installed_manifest["selectedExpertCount"],
            "expert_blob_bytes": expert_blob_bytes,
        },
        "experiment": {
            "objective": (
                "Test a default-off split-slot runtime that overlaps each direct "
                "expert miss's w2 read with first-stage MLX work"
            ),
            "workload": "repeated",
            "prompt_file": prompt["file"],
            "prompt_text_sha256": prompt["text_sha256"],
            "prompt_tokens": prompt["target_tokens"],
            "prompt_token_sha256": prompt["prompt_token_sha256"],
            "max_tokens": arguments.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "waves": len(MODE_ORDERS),
            "orders": [list(order) for order in MODE_ORDERS],
            "fresh_process_per_run": True,
            "control": "one fused expert slot and the existing ready-expert path",
            "candidate": "fixed split w13/w2 slot with bounded concurrent w2 reads",
            "w13_bytes": w13_bytes,
            "w2_bytes": w2_bytes,
            "cache_state": (
                "Darwin expert-file bypass descriptor; no OS-wide cache purge"
            ),
            "defaults_changed": False,
        },
        "gate": gate,
        "runs": rows,
        "limitations": [
            "single short 128-token repeated-prompt workload",
            "four pairs are exploratory and not a formal performance result",
            "F_NOCACHE residency behavior is not a physical SSD byte counter",
            "no DSpark, long decode, multi-workload, power, or ANE measurement",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "correctness_passed": gate["correctness"]["passed"],
                "performance_passed": gate["performance"]["passed"],
                "decision": gate["decision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the default-off staged expert runtime prototype"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", required=True)
    worker.add_argument("--prompt-file", required=True)
    worker.add_argument("--expected-prompt-tokens", required=True, type=int)
    worker.add_argument("--expected-prompt-token-sha256", required=True)
    worker.add_argument("--mode", choices=("control", "staged"), required=True)
    worker.add_argument("--max-tokens", type=int, required=True)
    worker.add_argument("--worker-output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt-manifest", required=True)
    run.add_argument("--raw-directory", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--max-tokens", type=int, default=32)
    return parser


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")
    if arguments.command == "worker":
        _run_worker(arguments)
    else:
        _run_orchestrator(arguments)


if __name__ == "__main__":
    main()
