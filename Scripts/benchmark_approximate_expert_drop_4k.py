from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .benchmark_dspark_adaptive import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )
except ImportError:
    from benchmark_dspark_adaptive import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )


WORKLOADS = ("repeated", "code", "zh_technical", "mixed_math", "tool_like")
MINIMUM_WORKLOAD_AGREEMENT = 0.90
MINIMUM_AGGREGATE_AGREEMENT = 0.95
MINIMUM_COMMON_PREFIX = 16
MINIMUM_DECODE_BYTE_REDUCTION = 0.10
MAXIMUM_PEAK_RSS_INCREASE = 0.05


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode("utf-8")).hexdigest()


def _load_prompts(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported 4K prompt manifest")
    prompts = []
    for row in manifest.get("prompts", []):
        if row.get("name") not in WORKLOADS or row.get("target_tokens") != 4096:
            raise ValueError("4K prompt manifest contains an invalid workload")
        prompt_path = path.parent / row["file"]
        if _sha256(prompt_path) != row["text_sha256"]:
            raise ValueError(f"prompt text hash mismatch: {row['name']}")
        prompts.append({**row, "path": str(prompt_path)})
    if tuple(row["name"] for row in prompts) != WORKLOADS:
        raise ValueError("4K prompts are not in the required workload order")
    return manifest, prompts


def _run_worker(arguments: argparse.Namespace) -> None:
    import mlx.core as mx

    from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
    from deepseek_v4_ssd.model import RuntimeConfig

    prompt_path = Path(arguments.prompt_file).expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    output = Path(arguments.worker_output).expanduser().resolve()
    config = RuntimeConfig(
        batched_expert_prefill=True,
        dspark_enabled=False,
        expert_file_cache_policy="bypass",
        layer_major_prefill=True,
        persistent_prompt_cache=False,
    )
    with ModelRuntime.open(arguments.model, config) as runtime:
        prompt_tokens = runtime._encode_prompt(prompt)
        if len(prompt_tokens) != arguments.expected_prompt_tokens:
            raise ValueError("prompt token count does not match manifest")
        if _token_sha256(prompt_tokens) != arguments.expected_prompt_token_sha256:
            raise ValueError("prompt token hash does not match manifest")
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
                    approximation_mode=(
                        "learned-route-drop-lowest-1"
                        if arguments.mode == "candidate"
                        else "exact"
                    ),
                ),
            )
        )
        wall_seconds = time.perf_counter() - started
        tokens = [int(piece.token) for piece in pieces]
        snapshot = runtime.metrics.snapshot()
        metric_names = (
            "request_seconds",
            "time_to_first_token_seconds",
            "decode_tokens_per_second",
            "decode_latency_p50_seconds",
            "decode_latency_p95_seconds",
            "peak_memory_bytes",
            "request_expert_bytes_read",
            "request_expert_read_seconds",
            "request_expert_cache_hits",
            "request_expert_cache_misses",
            "request_expert_evictions",
            "request_batched_expert_layers",
            "request_prefetched_layer_hits",
            "approximation_mode",
        )
        row = {
            "mode": arguments.mode,
            "prompt_tokens": len(prompt_tokens),
            "prompt_token_sha256": _token_sha256(prompt_tokens),
            "generated_tokens": len(tokens),
            "generated_token_ids": tokens,
            "token_sha256": _token_sha256(tokens),
            "output_text": "".join(piece.text for piece in pieces),
            "external_wall_seconds": wall_seconds,
            "metrics": {name: snapshot[name] for name in metric_names},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def _run_fresh_process(
    script: Path,
    *,
    project_root: Path,
    model: Path,
    prompt: dict[str, Any],
    mode: str,
    max_tokens: int,
    raw_directory: Path,
    run_label: str = "4k32",
) -> dict[str, Any]:
    run_id = f"{prompt['name']}-{run_label}-{mode}"
    result_path = raw_directory / f"{run_id}.json"
    stdout_path = raw_directory / f"{run_id}.stdout.txt"
    stderr_path = raw_directory / f"{run_id}.stderr.txt"
    command = [
        sys.executable,
        str(script),
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
    runtime_path = str(project_root / "runtime")
    environment["PYTHONPATH"] = (
        runtime_path
        if not environment.get("PYTHONPATH")
        else runtime_path + os.pathsep + environment["PYTHONPATH"]
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
    if completed.returncode or not result_path.is_file():
        error = stderr_path.read_text(encoding="utf-8")[-2_000:]
        raise RuntimeError(f"{run_id} failed ({completed.returncode}):\n{error}")
    row = json.loads(result_path.read_text(encoding="utf-8"))
    row.update(
        {
            "id": run_id,
            "workload": prompt["name"],
            "raw": {
                "result_sha256": _sha256(result_path),
                "stdout_sha256": _sha256(stdout_path),
                "stderr_sha256": _sha256(stderr_path),
            },
        }
    )
    return row


def _common_prefix(first: list[int], second: list[int]) -> int:
    count = 0
    for left, right in zip(first, second):
        if left != right:
            break
        count += 1
    return count


def _gate(
    pairs: list[dict[str, Any]],
    *,
    expert_count: int,
    expert_blob_bytes: int,
    safety_prerequisite: bool,
) -> dict[str, Any]:
    total_matches = 0
    total_aligned = 0
    exact_decode_bytes = 0
    candidate_decode_bytes = 0
    peak_changes = []
    comparisons = []
    for pair in pairs:
        exact = pair["exact"]
        candidate = pair["candidate"]
        aligned = min(len(exact["generated_token_ids"]), len(candidate["generated_token_ids"]))
        matches = sum(
            left == right
            for left, right in zip(exact["generated_token_ids"], candidate["generated_token_ids"])
        )
        agreement = matches / aligned if aligned else 0.0
        prefix = _common_prefix(exact["generated_token_ids"], candidate["generated_token_ids"])
        exact_prefill = (
            int(exact["metrics"]["request_batched_expert_layers"])
            * expert_count
            * expert_blob_bytes
        )
        candidate_prefill = (
            int(candidate["metrics"]["request_batched_expert_layers"])
            * expert_count
            * expert_blob_bytes
        )
        exact_decode = int(exact["metrics"]["request_expert_bytes_read"]) - exact_prefill
        candidate_decode = int(candidate["metrics"]["request_expert_bytes_read"]) - candidate_prefill
        peak_change = (
            int(candidate["metrics"]["peak_memory_bytes"])
            / int(exact["metrics"]["peak_memory_bytes"])
            - 1
        )
        total_matches += matches
        total_aligned += aligned
        exact_decode_bytes += exact_decode
        candidate_decode_bytes += candidate_decode
        peak_changes.append(peak_change)
        comparisons.append(
            {
                "workload": pair["workload"],
                "common_prefix_tokens": prefix,
                "aligned_token_agreement_fraction": agreement,
                "exact_decode_logical_expert_bytes": exact_decode,
                "candidate_decode_logical_expert_bytes": candidate_decode,
                "decode_logical_expert_byte_reduction_fraction": (
                    1 - candidate_decode / exact_decode
                ),
                "peak_memory_change_fraction": peak_change,
            }
        )
    aggregate_agreement = total_matches / total_aligned
    aggregate_byte_reduction = 1 - candidate_decode_bytes / exact_decode_bytes
    criteria = {
        "all_runs_generated_32_tokens": all(
            run[mode]["generated_tokens"] == 32
            for run in pairs
            for mode in ("exact", "candidate")
        ),
        "all_workload_token_agreements_at_least_90_percent": all(
            row["aligned_token_agreement_fraction"] >= MINIMUM_WORKLOAD_AGREEMENT
            for row in comparisons
        ),
        "aggregate_token_agreement_at_least_95_percent": (
            aggregate_agreement >= MINIMUM_AGGREGATE_AGREEMENT
        ),
        "all_common_prefixes_at_least_16_tokens": all(
            row["common_prefix_tokens"] >= MINIMUM_COMMON_PREFIX
            for row in comparisons
        ),
        "all_runs_have_42_full_layer_prefill_reads": all(
            run[mode]["metrics"]["request_batched_expert_layers"] == 42
            for run in pairs
            for mode in ("exact", "candidate")
        ),
        "aggregate_decode_logical_bytes_reduced_at_least_10_percent": (
            aggregate_byte_reduction >= MINIMUM_DECODE_BYTE_REDUCTION
        ),
        "all_peak_memory_increases_at_most_5_percent": all(
            change <= MAXIMUM_PEAK_RSS_INCREASE for change in peak_changes
        ),
        "phase_6a_safety_prerequisite_passed": safety_prerequisite,
    }
    return {
        "criteria": criteria,
        "aggregate_token_agreement_fraction": aggregate_agreement,
        "aggregate_decode_logical_expert_byte_reduction_fraction": aggregate_byte_reduction,
        "comparisons": comparisons,
        "passed": all(criteria.values()),
    }


def _source_state(project_root: Path, protocol: Path, prompt_manifest: Path) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=project_root, stdout=subprocess.PIPE, check=True
    ).stdout
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
        "working_tree_dirty": bool(_command_output(["git", "status", "--porcelain"], project_root)),
        "tracked_diff_sha256_at_run": hashlib.sha256(diff).hexdigest(),
        "runtime_python_tree_sha256": _runtime_tree_sha256(project_root / "runtime" / "deepseek_v4_ssd"),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256_at_run": _sha256(protocol),
        "prompt_manifest_sha256": _sha256(prompt_manifest),
    }


def _run(arguments: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    protocol = Path(arguments.protocol).expanduser().resolve()
    safety_artifact = Path(arguments.safety_artifact).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    manifest, prompts = _load_prompts(prompt_manifest)
    safety = json.loads(safety_artifact.read_text(encoding="utf-8"))
    safety_passed = bool(
        safety.get("gate", {}).get("criteria", {}).get("candidate_safety_all_passed")
    )
    installed = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
    raw_directory.mkdir(parents=True, exist_ok=True)
    pairs = []
    for index, prompt in enumerate(prompts):
        order = ("exact", "candidate") if index % 2 == 0 else ("candidate", "exact")
        rows = {}
        for mode in order:
            print(f"{prompt['name']} {mode}", flush=True)
            rows[mode] = _run_fresh_process(
                script,
                project_root=project_root,
                model=model,
                prompt=prompt,
                mode=mode,
                max_tokens=arguments.max_tokens,
                raw_directory=raw_directory,
            )
        pairs.append({"workload": prompt["name"], "order": list(order), **rows})
    gate = _gate(
        pairs,
        expert_count=int(installed["expertCount"]),
        expert_blob_bytes=int(installed["expertBlobSize"]),
        safety_prerequisite=safety_passed,
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "approximate_expert_drop_five_workload_4k32_pilot",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol, prompt_manifest),
        "installed_model": {
            "model_id": installed["modelID"],
            "revision": installed["revision"],
            "manifest_sha256": _sha256(model / "manifest.json"),
            "expert_count": installed["expertCount"],
            "expert_blob_bytes": installed["expertBlobSize"],
        },
        "method": {
            "workloads": list(WORKLOADS),
            "prompt_tokens": 4096,
            "max_tokens": arguments.max_tokens,
            "temperature": 0,
            "top_p": 1,
            "fresh_process_per_run": True,
            "expert_file_cache_policy": "bypass",
            "persistent_prompt_cache": False,
            "dspark": False,
            "timing_is_exploratory": True,
        },
        "prompt_manifest": {"path": str(prompt_manifest), "revision": manifest["revision"]},
        "safety_prerequisite": {"path": str(safety_artifact), "sha256": _sha256(safety_artifact)},
        "pairs": pairs,
        "gate": gate,
        "decision": {
            "phase_6c": "pass" if gate["passed"] else "stop_candidate",
            "continue_to_phase_6d": gate["passed"],
            "runtime_changed": False,
        },
        "limits": [
            "One pair per workload is a pilot, not a formal performance result.",
            "Token agreement is not a semantic capability score.",
            "Logical expert bytes are not physical SSD counters.",
            "The unchanged candidate reuses the Phase 6A short safety prerequisite.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 6C five-workload 4K/32 pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", required=True)
    worker.add_argument("--prompt-file", required=True)
    worker.add_argument("--expected-prompt-tokens", type=int, required=True)
    worker.add_argument("--expected-prompt-token-sha256", required=True)
    worker.add_argument("--mode", choices=("exact", "candidate"), required=True)
    worker.add_argument("--max-tokens", type=int, required=True)
    worker.add_argument("--worker-output", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt-manifest", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--safety-artifact", required=True)
    run.add_argument("--raw-directory", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-tokens", type=int, default=32)
    return parser


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "worker":
        _run_worker(arguments)
    else:
        if arguments.max_tokens != 32:
            parser.error("Phase 6C requires exactly 32 output tokens")
        _run(arguments)


if __name__ == "__main__":
    main()
