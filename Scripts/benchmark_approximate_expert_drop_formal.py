from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

try:
    from .benchmark_approximate_expert_drop_4k import (
        WORKLOADS,
        _common_prefix,
        _load_prompts,
        _run_fresh_process,
        _run_worker,
    )
    from .benchmark_common import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )
except ImportError:
    from benchmark_approximate_expert_drop_4k import (
        WORKLOADS,
        _common_prefix,
        _load_prompts,
        _run_fresh_process,
        _run_worker,
    )
    from benchmark_common import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )


OUTPUT_TOKENS = 256
WAVES = 2
MINIMUM_PAIR_AGREEMENT = 0.90
MINIMUM_AGGREGATE_AGREEMENT = 0.95
MINIMUM_COMMON_PREFIX = 64
MINIMUM_DECODE_BYTE_REDUCTION = 0.10
MINIMUM_MEDIAN_DECODE_THROUGHPUT_IMPROVEMENT = 0.05
MAXIMUM_WORKLOAD_MEDIAN_P95_REGRESSION = 0.02
MAXIMUM_PEAK_MEMORY_INCREASE = 0.05


def _source_state(
    project_root: Path,
    protocol: Path,
    prompt_manifest: Path,
) -> dict[str, Any]:
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
        "worker_script_sha256": _sha256(
            project_root / "Scripts" / "benchmark_approximate_expert_drop_4k.py"
        ),
        "protocol_sha256_at_run": _sha256(protocol),
        "prompt_manifest_sha256": _sha256(prompt_manifest),
    }


def _pair_comparison(
    pair: dict[str, Any],
    *,
    expert_count: int,
    expert_blob_bytes: int,
) -> dict[str, Any]:
    exact = pair["exact"]
    candidate = pair["candidate"]
    aligned = min(
        len(exact["generated_token_ids"]),
        len(candidate["generated_token_ids"]),
    )
    matches = sum(
        left == right
        for left, right in zip(
            exact["generated_token_ids"],
            candidate["generated_token_ids"],
        )
    )
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
    exact_decode_bytes = int(exact["metrics"]["request_expert_bytes_read"]) - exact_prefill
    candidate_decode_bytes = (
        int(candidate["metrics"]["request_expert_bytes_read"]) - candidate_prefill
    )
    exact_decode_rate = float(exact["metrics"]["decode_tokens_per_second"])
    candidate_decode_rate = float(candidate["metrics"]["decode_tokens_per_second"])
    exact_p95 = float(exact["metrics"]["decode_latency_p95_seconds"])
    candidate_p95 = float(candidate["metrics"]["decode_latency_p95_seconds"])
    exact_peak = int(exact["metrics"]["peak_memory_bytes"])
    candidate_peak = int(candidate["metrics"]["peak_memory_bytes"])
    return {
        "wave": pair["wave"],
        "workload": pair["workload"],
        "common_prefix_tokens": _common_prefix(
            exact["generated_token_ids"],
            candidate["generated_token_ids"],
        ),
        "aligned_tokens": aligned,
        "aligned_token_matches": matches,
        "aligned_token_agreement_fraction": matches / aligned if aligned else 0.0,
        "exact_decode_logical_expert_bytes": exact_decode_bytes,
        "candidate_decode_logical_expert_bytes": candidate_decode_bytes,
        "decode_logical_expert_byte_reduction_fraction": (
            1 - candidate_decode_bytes / exact_decode_bytes
        ),
        "decode_throughput_change_fraction": candidate_decode_rate / exact_decode_rate - 1,
        "decode_p95_change_fraction": candidate_p95 / exact_p95 - 1,
        "peak_memory_change_fraction": candidate_peak / exact_peak - 1,
    }


def _gate(
    pairs: list[dict[str, Any]],
    *,
    expert_count: int,
    expert_blob_bytes: int,
    safety_prerequisite: bool,
) -> dict[str, Any]:
    comparisons = [
        _pair_comparison(
            pair,
            expert_count=expert_count,
            expert_blob_bytes=expert_blob_bytes,
        )
        for pair in pairs
    ]
    exact_decode_bytes = sum(
        row["exact_decode_logical_expert_bytes"] for row in comparisons
    )
    candidate_decode_bytes = sum(
        row["candidate_decode_logical_expert_bytes"] for row in comparisons
    )
    aggregate_byte_reduction = 1 - candidate_decode_bytes / exact_decode_bytes
    aggregate_matches = sum(row["aligned_token_matches"] for row in comparisons)
    aggregate_aligned = sum(row["aligned_tokens"] for row in comparisons)
    aggregate_agreement = aggregate_matches / aggregate_aligned
    throughput_changes = [
        row["decode_throughput_change_fraction"] for row in comparisons
    ]
    workload_p95 = {
        workload: statistics.median(
            row["decode_p95_change_fraction"]
            for row in comparisons
            if row["workload"] == workload
        )
        for workload in WORKLOADS
    }
    criteria = {
        "all_20_runs_generated_256_tokens": all(
            pair[mode]["generated_tokens"] == OUTPUT_TOKENS
            for pair in pairs
            for mode in ("exact", "candidate")
        ),
        "all_runtime_modes_reported_correctly": all(
            pair["exact"]["metrics"]["approximation_mode"] == "exact"
            and pair["candidate"]["metrics"]["approximation_mode"]
            == "learned-route-drop-lowest-1"
            for pair in pairs
        ),
        "all_pair_token_agreements_at_least_90_percent": all(
            row["aligned_token_agreement_fraction"] >= MINIMUM_PAIR_AGREEMENT
            for row in comparisons
        ),
        "aggregate_token_agreement_at_least_95_percent": (
            aggregate_agreement >= MINIMUM_AGGREGATE_AGREEMENT
        ),
        "all_common_prefixes_at_least_64_tokens": all(
            row["common_prefix_tokens"] >= MINIMUM_COMMON_PREFIX
            for row in comparisons
        ),
        "all_runs_have_42_full_layer_prefill_reads": all(
            pair[mode]["metrics"]["request_batched_expert_layers"] == 42
            for pair in pairs
            for mode in ("exact", "candidate")
        ),
        "aggregate_decode_logical_bytes_reduced_at_least_10_percent": (
            aggregate_byte_reduction >= MINIMUM_DECODE_BYTE_REDUCTION
        ),
        "median_decode_throughput_improved_at_least_5_percent": (
            statistics.median(throughput_changes)
            >= MINIMUM_MEDIAN_DECODE_THROUGHPUT_IMPROVEMENT
        ),
        "every_workload_median_decode_p95_regression_at_most_2_percent": all(
            change <= MAXIMUM_WORKLOAD_MEDIAN_P95_REGRESSION
            for change in workload_p95.values()
        ),
        "all_peak_memory_increases_at_most_5_percent": all(
            row["peak_memory_change_fraction"] <= MAXIMUM_PEAK_MEMORY_INCREASE
            for row in comparisons
        ),
        "phase_6a_safety_prerequisite_passed": safety_prerequisite,
    }
    return {
        "criteria": criteria,
        "aggregate_token_agreement_fraction": aggregate_agreement,
        "aggregate_decode_logical_expert_byte_reduction_fraction": (
            aggregate_byte_reduction
        ),
        "exact_decode_logical_expert_bytes_per_output_token": (
            exact_decode_bytes / aggregate_aligned
        ),
        "candidate_decode_logical_expert_bytes_per_output_token": (
            candidate_decode_bytes / aggregate_aligned
        ),
        "median_decode_throughput_change_fraction": statistics.median(
            throughput_changes
        ),
        "workload_median_decode_p95_change_fraction": workload_p95,
        "comparisons": comparisons,
        "passed": all(criteria.values()),
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
        safety.get("gate", {}).get("criteria", {}).get(
            "candidate_safety_all_passed"
        )
    )
    installed = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
    raw_directory.mkdir(parents=True, exist_ok=True)
    pairs = []
    for wave in range(1, WAVES + 1):
        for index, prompt in enumerate(prompts):
            exact_first = (index + wave) % 2 == 1
            order = ("exact", "candidate") if exact_first else ("candidate", "exact")
            rows = {}
            for mode in order:
                print(f"wave {wave} {prompt['name']} {mode}", flush=True)
                rows[mode] = _run_fresh_process(
                    script,
                    project_root=project_root,
                    model=model,
                    prompt=prompt,
                    mode=mode,
                    max_tokens=OUTPUT_TOKENS,
                    raw_directory=raw_directory,
                    run_label=f"4k256-wave{wave}",
                )
            pairs.append(
                {
                    "wave": wave,
                    "workload": prompt["name"],
                    "order": list(order),
                    **rows,
                }
            )
    gate = _gate(
        pairs,
        expert_count=int(installed["expertCount"]),
        expert_blob_bytes=int(installed["expertBlobSize"]),
        safety_prerequisite=safety_passed,
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "approximate_expert_drop_five_workload_4k256_formal_gate",
        "formal_performance_result": True,
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
            "waves": WAVES,
            "prompt_tokens": 4096,
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "top_p": 1,
            "fresh_process_per_run": True,
            "reversed_order_between_waves": True,
            "expert_file_cache_policy": "bypass",
            "persistent_prompt_cache": False,
            "dspark": False,
            "candidate_runtime_api": "learned-route-drop-lowest-1",
        },
        "prompt_manifest": {
            "path": str(prompt_manifest),
            "revision": manifest["revision"],
        },
        "safety_prerequisite": {
            "path": str(safety_artifact),
            "sha256": _sha256(safety_artifact),
        },
        "pairs": pairs,
        "gate": gate,
        "manual_quality_review": {
            "status": "required_after_machine_gate",
            "scope": "Compare exact and candidate text for all five workloads.",
        },
        "decision": {
            "phase_6e_machine_gate": "pass" if gate["passed"] else "stop_candidate",
            "candidate_can_be_adopted": False,
        },
        "limits": [
            "Decode logical expert bytes subtract the fixed 42-layer full Prefill reads.",
            "Logical bytes are not physical SSD counters.",
            "The page cache was not purged between fresh processes.",
            "The machine gate does not replace manual capability review.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6E five-workload 4K/256 formal gate"
    )
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
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "worker":
        if arguments.max_tokens != OUTPUT_TOKENS:
            raise ValueError("Phase 6E worker requires 256 output tokens")
        _run_worker(arguments)
    else:
        _run(arguments)


if __name__ == "__main__":
    main()
