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


MODE_ORDERS = (("adaptive", "control"), ("control", "adaptive"))
SUMMARY_METRICS = (
    "request_seconds",
    "external_wall_seconds",
    "time_to_first_token_seconds",
    "peak_memory_bytes",
    "request_expert_cache_hits",
    "request_expert_cache_misses",
    "request_expert_evictions",
    "request_expert_bytes_read",
    "request_expert_read_seconds",
    "request_routing_sync_seconds",
    "request_batched_expert_layers",
    "request_gather_qmm_calls",
    "request_prefetched_layer_hits",
    "request_adaptive_prefill_planned_layers",
    "request_adaptive_prefill_full_layers",
    "request_adaptive_prefill_selective_layers",
    "request_adaptive_prefill_union_experts",
    "request_adaptive_prefill_read_experts",
    "request_adaptive_prefill_bytes_read",
    "request_adaptive_prefill_avoided_bytes",
    "request_adaptive_prefill_plan_seconds",
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


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _load_repeated_prompt(manifest_path: Path) -> tuple[dict, dict]:
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
    if int(prompt["target_tokens"]) != 4_096:
        raise ValueError("repeated prompt is not the predeclared 4,096 tokens")
    prompt["path"] = str(path)
    return manifest, prompt


def _repeated_decisions_are_threshold_invariant(eligibility: dict) -> bool:
    workload = next(
        item for item in eligibility["workloads"] if item["name"] == "repeated"
    )
    thresholds = workload["route_summary"]["thresholds"]
    signatures = {
        tuple(
            (decision["layer"], decision["mode"], decision["read_experts"])
            for decision in threshold["decisions"]
        )
        for threshold in thresholds.values()
    }
    return len(signatures) == 1


def _paired_changes(rows: list[dict]) -> list[dict]:
    pairs = []
    for wave in sorted({int(row["wave"]) for row in rows}):
        by_mode = {
            row["mode"]: row for row in rows if int(row["wave"]) == wave
        }
        control = by_mode["control"]
        adaptive = by_mode["adaptive"]
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
                    control["token_sha256"] == adaptive["token_sha256"]
                ),
                "change_fraction": {
                    key: _change_fraction(
                        float(control["metrics"][key]),
                        float(adaptive["metrics"][key]),
                    )
                    for key in (
                        "request_seconds",
                        "external_wall_seconds",
                        "time_to_first_token_seconds",
                        "peak_memory_bytes",
                        "request_expert_bytes_read",
                    )
                },
            }
        )
    return pairs


def _runtime_gate(
    rows: list[dict],
    *,
    max_tokens: int,
    active_layers: int,
    expert_count: int,
    expert_blob_bytes: int,
    threshold_invariant: bool,
) -> dict:
    controls = [row for row in rows if row["mode"] == "control"]
    candidates = [row for row in rows if row["mode"] == "adaptive"]
    pairs = _paired_changes(rows)
    full_prefill_experts = active_layers * expert_count

    def closes(row: dict) -> bool:
        metrics = row["metrics"]
        misses = int(metrics["request_expert_cache_misses"])
        if row["mode"] == "control":
            expected_experts = full_prefill_experts + misses
        else:
            expected_experts = (
                int(metrics["request_adaptive_prefill_read_experts"]) + misses
            )
        return (
            int(metrics["request_expert_bytes_read"])
            == expected_experts * expert_blob_bytes
        )

    correctness_criteria = {
        "two_complete_reversed_pairs": (
            len(controls) == 2
            and len(candidates) == 2
            and [pair["order"] for pair in pairs]
            == [["adaptive", "control"], ["control", "adaptive"]]
        ),
        "repeated_decisions_identical_at_all_three_thresholds": (
            threshold_invariant
        ),
        "all_prompt_token_hashes_exact": len(
            {row["prompt_token_sha256"] for row in rows}
        )
        == 1,
        "all_output_token_ids_and_hashes_exact": (
            len({tuple(row["generated_token_ids"]) for row in rows}) == 1
            and len({row["token_sha256"] for row in rows}) == 1
            and all(pair["token_hash_exact"] for pair in pairs)
        ),
        "all_runs_reached_output_budget": all(
            int(row["generated_tokens"]) == max_tokens for row in rows
        ),
        "all_actual_expert_bytes_close_by_blob": all(closes(row) for row in rows),
        "adaptive_batched_bytes_close": all(
            int(row["metrics"]["request_adaptive_prefill_bytes_read"])
            == int(row["metrics"]["request_adaptive_prefill_read_experts"])
            * expert_blob_bytes
            for row in candidates
        ),
        "control_has_no_adaptive_decisions": all(
            int(row["metrics"]["request_adaptive_prefill_planned_layers"]) == 0
            and int(row["metrics"]["request_adaptive_prefill_read_experts"]) == 0
            and int(row["metrics"]["request_adaptive_prefill_bytes_read"]) == 0
            and int(row["metrics"]["request_adaptive_prefill_avoided_bytes"]) == 0
            for row in controls
        ),
        "adaptive_decisions_cover_all_expert_layers": all(
            int(row["metrics"]["request_adaptive_prefill_planned_layers"])
            == active_layers
            and int(row["metrics"]["request_adaptive_prefill_full_layers"])
            + int(row["metrics"]["request_adaptive_prefill_selective_layers"])
            == active_layers
            for row in candidates
        ),
        "adaptive_read_set_covers_union_within_capacity": all(
            0
            < int(row["metrics"]["request_adaptive_prefill_union_experts"])
            <= int(row["metrics"]["request_adaptive_prefill_read_experts"])
            <= full_prefill_experts
            for row in candidates
        ),
        "adaptive_avoided_bytes_close": all(
            int(row["metrics"]["request_adaptive_prefill_avoided_bytes"])
            == (
                full_prefill_experts
                - int(row["metrics"]["request_adaptive_prefill_read_experts"])
            )
            * expert_blob_bytes
            for row in candidates
        ),
        "configuration_exact": all(
            row["configuration"]
            == {
                "adaptive_expert_prefill_threshold": (
                    0.9 if row["mode"] == "adaptive" else None
                ),
                "batched_expert_prefill": True,
                "dspark_enabled": False,
                "expert_file_cache_policy": "bypass",
                "layer_major_prefill": True,
                "persistent_prompt_cache": False,
                "staged_expert_streaming": False,
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
            "time_to_first_token_seconds",
            "peak_memory_bytes",
            "request_expert_bytes_read",
        )
    }
    control_ttft_p95 = _percentile(
        [float(row["metrics"]["time_to_first_token_seconds"]) for row in controls],
        0.95,
    )
    adaptive_ttft_p95 = _percentile(
        [
            float(row["metrics"]["time_to_first_token_seconds"])
            for row in candidates
        ],
        0.95,
    )
    ttft_p95_change = _change_fraction(control_ttft_p95, adaptive_ttft_p95)
    performance_criteria = {
        "median_expert_bytes_fall_at_least_10_percent": (
            paired_medians["request_expert_bytes_read"] <= -0.10
        ),
        "median_ttft_improves_at_least_5_percent": (
            paired_medians["time_to_first_token_seconds"] <= -0.05
        ),
        "repeated_ttft_p95_regresses_no_more_than_5_percent": (
            ttft_p95_change is not None and ttft_p95_change <= 0.05
        ),
        "median_peak_memory_increases_no_more_than_5_percent": (
            paired_medians["peak_memory_bytes"] <= 0.05
        ),
    }
    performance_passed = correctness_passed and all(
        performance_criteria.values()
    )
    if not correctness_passed:
        decision = "stop_runtime_candidate_on_correctness"
    elif performance_passed:
        decision = "continue_full_multiworkload_threshold_screen_default_off"
    else:
        decision = "stop_runtime_candidate_keep_default_off"
    return {
        "correctness": {
            "criteria": correctness_criteria,
            "passed": correctness_passed,
        },
        "performance": {
            "paired_median_change_fraction": paired_medians,
            "control_ttft_p95_seconds": control_ttft_p95,
            "adaptive_ttft_p95_seconds": adaptive_ttft_p95,
            "ttft_p95_change_fraction": ttft_p95_change,
            "criteria": performance_criteria,
            "passed": performance_passed,
        },
        "paired_runs": pairs,
        "decision": decision,
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
    adaptive = arguments.mode == "adaptive"
    config = RuntimeConfig(
        adaptive_expert_prefill_threshold=(0.9 if adaptive else None),
        batched_expert_prefill=True,
        dspark_enabled=False,
        expert_file_cache_policy="bypass",
        layer_major_prefill=True,
        persistent_prompt_cache=False,
        staged_expert_streaming=False,
    )
    runtime = ModelRuntime.open(str(model), config)
    try:
        prompt_tokens = runtime._encode_prompt(prompt)
        prompt_token_sha256 = _token_sha256(prompt_tokens)
        if len(prompt_tokens) != arguments.expected_prompt_tokens:
            raise RuntimeError("worker prompt token count does not match manifest")
        if prompt_token_sha256 != arguments.expected_prompt_token_sha256:
            raise RuntimeError("worker prompt token hash does not match manifest")
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
                "adaptive_expert_prefill_threshold": (
                    config.adaptive_expert_prefill_threshold
                ),
                "batched_expert_prefill": config.batched_expert_prefill,
                "dspark_enabled": config.dspark_enabled,
                "expert_file_cache_policy": config.expert_file_cache_policy,
                "layer_major_prefill": config.layer_major_prefill,
                "persistent_prompt_cache": config.persistent_prompt_cache,
                "staged_expert_streaming": config.staged_expert_streaming,
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
    run_id = f"repeated-4k-w{wave:02d}-{sequence:02d}-{mode}"
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
    eligibility_path = Path(arguments.eligibility_artifact).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    protocol = project_root / "research" / "ADAPTIVE_EXPERT_PREFILL_2026-08-27.md"
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if raw_directory.exists() and any(raw_directory.iterdir()):
        raise ValueError(f"raw directory must be new or empty: {raw_directory}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_source, prompt = _load_repeated_prompt(prompt_manifest)
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    threshold_invariant = _repeated_decisions_are_threshold_invariant(eligibility)
    installed_manifest = json.loads((model / "manifest.json").read_text())
    layer_count = int(installed_manifest["layerCount"])
    expert_count = int(installed_manifest["expertCount"])
    expert_blob_bytes = int(installed_manifest["expertBlobSize"])

    rows = []
    total = sum(len(order) for order in MODE_ORDERS)
    completed = 0
    for wave, order in enumerate(MODE_ORDERS, start=1):
        for sequence, mode in enumerate(order, start=1):
            completed += 1
            print(f"[{completed}/{total}] wave={wave} mode={mode}", flush=True)
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
        active_layers=layer_count - 1,
        expert_count=expert_count,
        expert_blob_bytes=expert_blob_bytes,
        threshold_invariant=threshold_invariant,
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "adaptive_expert_prefill_repeated_4k_two_pair_gate",
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
            "layer_count": layer_count,
            "expert_count": expert_count,
            "selected_expert_count": installed_manifest["selectedExpertCount"],
            "expert_blob_bytes": expert_blob_bytes,
        },
        "experiment": {
            "objective": (
                "Stop or continue the adaptive prefill runtime before the full "
                "five-workload threshold matrix"
            ),
            "prompt_manifest": str(prompt_manifest),
            "prompt_manifest_sha256": _sha256(prompt_manifest),
            "prompt_model_id": prompt_source["model_id"],
            "prompt_revision": prompt_source["revision"],
            "eligibility_artifact": str(eligibility_path),
            "eligibility_artifact_sha256": _sha256(eligibility_path),
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
            "candidate_threshold": 0.9,
            "thresholds_share_repeated_decisions": threshold_invariant,
            "expert_file_cache_policy": "bypass",
            "persistent_prompt_cache": False,
            "dspark": False,
            "defaults_changed": False,
            "cache_state": (
                "Darwin expert-file bypass descriptor; no OS-wide cache purge"
            ),
        },
        "gate": gate,
        "runs": rows,
        "limitations": [
            "one workload and two pairs are exploratory, not a formal result",
            "the full threshold-by-workload screen stops if this guard fails",
            "F_NOCACHE behavior is not a physical SSD byte counter",
            "no power, ANE, multi-request, or long-context measurement",
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
        description="Gate the default-off adaptive expert prefill runtime"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", required=True)
    worker.add_argument("--prompt-file", required=True)
    worker.add_argument("--expected-prompt-tokens", required=True, type=int)
    worker.add_argument("--expected-prompt-token-sha256", required=True)
    worker.add_argument("--mode", choices=("control", "adaptive"), required=True)
    worker.add_argument("--max-tokens", type=int, required=True)
    worker.add_argument("--worker-output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt-manifest", required=True)
    run.add_argument("--eligibility-artifact", required=True)
    run.add_argument("--raw-directory", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--max-tokens", type=int, default=2)
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
