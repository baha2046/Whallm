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
from pathlib import Path


WORKLOADS = ("repeated", "code", "zh_technical", "mixed_math", "tool_like")
THRESHOLDS = (0.7, 0.8, 0.9)
EXPECTED_PROMPT_TOKENS = 4_096
EXPECTED_LAYER_MAJOR_TOKENS = 4_095


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_prompts(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {prompt["name"]: prompt for prompt in manifest["prompts"]}
    prompts = []
    for name in WORKLOADS:
        if name not in by_name:
            raise ValueError(f"prompt manifest is missing {name}")
        prompt = dict(by_name[name])
        path = manifest_path.parent / prompt["file"]
        if not path.is_file():
            raise ValueError(f"prompt file is missing: {path}")
        if _sha256(path) != prompt["text_sha256"]:
            raise ValueError(f"prompt text hash mismatch: {path}")
        if int(prompt["target_tokens"]) != EXPECTED_PROMPT_TOKENS:
            raise ValueError(f"{name} is not an exact 4,096-token prompt")
        prompt["path"] = str(path)
        prompts.append(prompt)
    return manifest, prompts


def _threshold_key(threshold: float) -> str:
    return f"{int(round(threshold * 100))}_percent"


def _summarize_trace(
    trace: dict,
    *,
    expected_layers: int,
    expert_count: int,
    selected_experts: int,
    expert_blob_bytes: int,
) -> dict:
    if int(trace.get("format", 0)) != 1:
        raise ValueError("unsupported route trace format")
    if int(trace["layer_count"]) != expected_layers:
        raise ValueError("route trace layer count mismatch")
    if int(trace["expert_count"]) != expert_count:
        raise ValueError("route trace expert count mismatch")
    if int(trace["selected_expert_count"]) != selected_experts:
        raise ValueError("route trace selected-expert count mismatch")
    if int(trace["expert_blob_size"]) != expert_blob_bytes:
        raise ValueError("route trace expert blob size mismatch")

    histograms = trace["prefill_histograms"]
    chunk_histograms = trace["prefill_chunk_histograms"]
    if len(histograms) != expected_layers:
        raise ValueError("prefill histogram layer count mismatch")
    if len(chunk_histograms) != expected_layers:
        raise ValueError("prefill chunk histogram layer count mismatch")
    expected_assignments = EXPECTED_LAYER_MAJOR_TOKENS * selected_experts
    active_layers = []
    non_layer_major_layers = []
    for layer, (histogram, chunks) in enumerate(
        zip(histograms, chunk_histograms, strict=True)
    ):
        if len(histogram) != expert_count:
            raise ValueError("prefill histogram expert count mismatch")
        if any(int(count) < 0 for count in histogram):
            raise ValueError("prefill histogram contains a negative count")
        combined = [0] * expert_count
        for chunk in chunks:
            if len(chunk) != expert_count:
                raise ValueError("prefill chunk histogram expert count mismatch")
            if any(int(count) < 0 for count in chunk):
                raise ValueError("prefill chunk histogram contains a negative count")
            for expert, count in enumerate(chunk):
                combined[expert] += int(count)
            if sum(int(count) for count in chunk) % selected_experts:
                raise ValueError("prefill chunk is not token-aligned")
        if combined != [int(count) for count in histogram]:
            raise ValueError("prefill chunks do not close to the cumulative histogram")
        layer_major_chunks = [
            chunk
            for chunk in chunks
            if sum(int(count) for count in chunk) == expected_assignments
        ]
        if not layer_major_chunks:
            non_layer_major_layers.append(layer)
            continue
        if len(layer_major_chunks) != 1:
            raise ValueError(
                f"layer {layer} has {len(layer_major_chunks)} layer-major chunks"
            )
        layer_major = [int(count) for count in layer_major_chunks[0]]
        assignments = sum(layer_major)
        union = sum(bool(count) for count in layer_major)
        active_layers.append(
            {
                "layer": layer,
                "assignments": assignments,
                "histogram": layer_major,
                "union_experts": union,
                "union_fraction": union / expert_count,
                "empty_experts": expert_count - union,
                "other_prefill_chunk_assignments": [
                    sum(int(count) for count in chunk)
                    for chunk in chunks
                    if chunk is not layer_major_chunks[0]
                ],
            }
        )
    if len(active_layers) != expected_layers - 1:
        raise ValueError("expected exactly 42 layer-major expert histograms")
    if non_layer_major_layers != [expected_layers - 1]:
        raise ValueError("only the final layer may lack a layer-major chunk")

    full_experts = len(active_layers) * expert_count
    baseline_bytes = full_experts * expert_blob_bytes
    thresholds = {}
    for threshold in THRESHOLDS:
        decisions = []
        selected_total = 0
        full_layers = selective_layers = 0
        for layer in active_layers:
            full = layer["union_fraction"] > threshold
            read_experts = expert_count if full else layer["union_experts"]
            selected_total += read_experts
            full_layers += int(full)
            selective_layers += int(not full)
            decisions.append(
                {
                    "layer": layer["layer"],
                    "mode": "full" if full else "selective",
                    "union_experts": layer["union_experts"],
                    "read_experts": read_experts,
                }
            )
        estimated_bytes = selected_total * expert_blob_bytes
        thresholds[_threshold_key(threshold)] = {
            "threshold": threshold,
            "full_layers": full_layers,
            "selective_layers": selective_layers,
            "read_experts": selected_total,
            "avoided_experts": full_experts - selected_total,
            "estimated_bytes": estimated_bytes,
            "estimated_byte_change_fraction": (
                estimated_bytes / baseline_bytes - 1
            ),
            "decisions": decisions,
        }
    unions = [layer["union_experts"] for layer in active_layers]
    return {
        "active_expert_layers": len(active_layers),
        "non_layer_major_layers": non_layer_major_layers,
        "expected_assignments_per_layer": expected_assignments,
        "full_read_experts": full_experts,
        "full_read_bytes": baseline_bytes,
        "union_experts": {
            "minimum": min(unions),
            "median": statistics.median(unions),
            "maximum": max(unions),
            "mean": statistics.fmean(unions),
        },
        "layers": active_layers,
        "thresholds": thresholds,
    }


def _eligibility_gate(workloads: list[dict]) -> dict:
    correctness = {
        "all_reference_trace_prompt_hashes_exact": all(
            workload["reference"]["metrics"]["prompt_token_sha256"]
            == workload["trace"]["metrics"]["prompt_token_sha256"]
            for workload in workloads
        ),
        "all_reference_trace_output_tokens_exact": all(
            workload["reference"]["metrics"]["generated_token_ids"]
            == workload["trace"]["metrics"]["generated_token_ids"]
            and workload["reference"]["metrics"]["token_sha256"]
            == workload["trace"]["metrics"]["token_sha256"]
            for workload in workloads
        ),
        "all_prompts_exactly_4096_tokens": all(
            int(workload["trace"]["metrics"]["prompt_tokens"])
            == EXPECTED_PROMPT_TOKENS
            for workload in workloads
        ),
        "all_layer_major_paths_cover_4095_tokens": all(
            int(workload["trace"]["metrics"]["layer_major_prefill_tokens"])
            == EXPECTED_LAYER_MAJOR_TOKENS
            for workload in workloads
        ),
        "all_traces_have_42_exact_expert_layers": all(
            workload["route_summary"]["active_expert_layers"] == 42
            and workload["route_summary"]["non_layer_major_layers"] == [42]
            for workload in workloads
        ),
    }
    threshold_summary = {}
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        changes = [
            float(
                workload["route_summary"]["thresholds"][key][
                    "estimated_byte_change_fraction"
                ]
            )
            for workload in workloads
        ]
        passing = sum(change <= -0.10 for change in changes)
        threshold_summary[key] = {
            "threshold": threshold,
            "workload_change_fraction": {
                workload["name"]: change
                for workload, change in zip(workloads, changes, strict=True)
            },
            "median_change_fraction": statistics.median(changes),
            "workloads_saving_at_least_10_percent": passing,
            "continuation_eligible": passing >= 3,
        }
    correctness_passed = all(correctness.values())
    eligible = [
        summary
        for summary in threshold_summary.values()
        if summary["continuation_eligible"]
    ]
    return {
        "correctness": {
            "criteria": correctness,
            "passed": correctness_passed,
        },
        "thresholds": threshold_summary,
        "continuation_passed": correctness_passed and bool(eligible),
        "eligible_thresholds": [
            summary["threshold"] for summary in eligible
        ],
        "decision": (
            "continue_default_off_runtime_threshold_screen"
            if correctness_passed and eligible
            else "stop_no_route_union_byte_eligibility"
        ),
    }


def _run(
    *,
    project_root: Path,
    python: Path,
    model: Path,
    prompt: dict,
    mode: str,
    sequence: int,
    raw_directory: Path,
    max_tokens: int,
) -> dict:
    run_id = f"{prompt['name']}-{sequence:02d}-{mode}"
    metrics_path = raw_directory / f"{run_id}.metrics.json"
    trace_path = raw_directory / f"{run_id}.routes.json"
    stdout_path = raw_directory / f"{run_id}.stdout.txt"
    stderr_path = raw_directory / f"{run_id}.stderr.txt"
    command = [
        str(python),
        "-m",
        "deepseek_v4_ssd.cli",
        "--model",
        str(model),
        "--prompt-file",
        prompt["path"],
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--no-persistent-prompt-cache",
        "--expert-file-cache-policy",
        "bypass",
        "--metrics-json",
        str(metrics_path),
        *(
            ("--expert-route-trace", str(trace_path))
            if mode == "trace"
            else ()
        ),
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
    if completed.returncode or not metrics_path.is_file():
        raise RuntimeError(
            f"{run_id} failed ({completed.returncode}):\n{stderr_text[-2_000:]}"
        )
    if (mode == "trace") != trace_path.is_file():
        raise RuntimeError(f"{run_id} route trace presence is incorrect")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("prompt_token_sha256") != prompt["prompt_token_sha256"]:
        raise RuntimeError(f"{run_id} prompt token hash mismatch")
    return {
        "id": run_id,
        "mode": mode,
        "sequence": sequence,
        "command": command,
        "metrics": metrics,
        "metrics_sha256": _sha256(metrics_path),
        "route_trace": (
            json.loads(trace_path.read_text(encoding="utf-8"))
            if mode == "trace"
            else None
        ),
        "route_trace_sha256": _sha256(trace_path) if mode == "trace" else None,
        "stdout": stdout_text,
        "stdout_sha256": _sha256(stdout_path),
        "stderr": stderr_text,
        "stderr_sha256": _sha256(stderr_path),
        "returncode": completed.returncode,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure 4K prefill route-union eligibility at 70/80/90%"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-tokens", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.max_tokens < 2:
        parser.error("--max-tokens must be at least 2")

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    manifest_path = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    protocol = project_root / "research" / "ADAPTIVE_EXPERT_PREFILL_2026-08-27.md"
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if raw_directory.exists() and any(raw_directory.iterdir()):
        parser.error(f"raw directory must be new or empty: {raw_directory}")
    if output.exists():
        parser.error(f"output already exists: {output}")
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_manifest, prompts = _load_prompts(manifest_path)
    installed_manifest = json.loads((model / "manifest.json").read_text())
    layer_count = int(installed_manifest["layerCount"])
    expert_count = int(installed_manifest["expertCount"])
    selected_experts = int(installed_manifest["selectedExpertCount"])
    expert_blob_bytes = int(installed_manifest["expertBlobSize"])

    workloads = []
    total = len(prompts) * 2
    completed = 0
    for workload_index, prompt in enumerate(prompts):
        order = (
            ("reference", "trace")
            if workload_index % 2 == 0
            else ("trace", "reference")
        )
        by_mode = {}
        for sequence, mode in enumerate(order, start=1):
            completed += 1
            print(
                f"[{completed}/{total}] workload={prompt['name']} mode={mode}",
                flush=True,
            )
            by_mode[mode] = _run(
                project_root=project_root,
                python=python,
                model=model,
                prompt=prompt,
                mode=mode,
                sequence=sequence,
                raw_directory=raw_directory,
                max_tokens=arguments.max_tokens,
            )
        trace = by_mode["trace"]["route_trace"]
        if trace is None:
            raise RuntimeError("trace run did not return route data")
        workloads.append(
            {
                "name": prompt["name"],
                "prompt_file": prompt["file"],
                "prompt_text_sha256": prompt["text_sha256"],
                "prompt_token_sha256": prompt["prompt_token_sha256"],
                "order": list(order),
                "reference": by_mode["reference"],
                "trace": by_mode["trace"],
                "route_summary": _summarize_trace(
                    trace,
                    expected_layers=layer_count,
                    expert_count=expert_count,
                    selected_experts=selected_experts,
                    expert_blob_bytes=expert_blob_bytes,
                ),
            }
        )

    gate = _eligibility_gate(workloads)
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "adaptive_expert_prefill_route_union_eligibility",
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
            "selected_expert_count": selected_experts,
            "expert_blob_bytes": expert_blob_bytes,
        },
        "experiment": {
            "objective": (
                "Determine whether exact 4K layer route unions justify a "
                "default-off selective expert-read prototype"
            ),
            "prompt_manifest": str(manifest_path),
            "prompt_manifest_sha256": _sha256(manifest_path),
            "prompt_model_id": prompt_manifest["model_id"],
            "prompt_revision": prompt_manifest["revision"],
            "workloads": list(WORKLOADS),
            "prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "layer_major_tokens": EXPECTED_LAYER_MAJOR_TOKENS,
            "max_tokens": arguments.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "thresholds": list(THRESHOLDS),
            "expert_file_cache_policy": "bypass",
            "persistent_prompt_cache": False,
            "dspark": False,
            "route_trace_timing_interpretable": False,
        },
        "gate": gate,
        "workloads": workloads,
        "limitations": [
            "route-union byte estimate assigns zero cost to planning and scattered reads",
            "trace synchronization invalidates reference/trace timing comparison",
            "bypass residency behavior is not a physical SSD byte counter",
            "no selective runtime, warm-cache, power, ANE, or multi-request measurement",
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
                "eligible_thresholds": gate["eligible_thresholds"],
                "decision": gate["decision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
