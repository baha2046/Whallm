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


WORKLOAD_ORDER = (
    "repeated",
    "code",
    "zh_technical",
    "mixed_math",
    "tool_like",
)
RUN_PATTERN = ("fixed", "adaptive", "adaptive", "fixed")
MODE_ARGUMENTS = {
    "normal": (),
    "fixed": ("--dspark",),
    "adaptive": (
        "--dspark",
        "--dspark-adaptive-block",
    ),
}
ROW_METRICS = (
    "generated_token_ids",
    "expert_file_cache_policy",
    "expert_file_direct_io_alignment_bytes",
    "prompt_token_sha256",
    "token_sha256",
    "prompt_tokens",
    "generated_tokens",
    "request_seconds",
    "time_to_first_token_seconds",
    "decode_tokens_per_second",
    "peak_memory_bytes",
    "request_expert_bytes_read",
    "request_process_disk_bytes_read",
    "request_expert_page_cache_probe_calls",
    "request_expert_page_cache_probe_failures",
    "request_expert_page_cache_classified_bytes",
    "request_expert_page_cache_resident_bytes_before_read",
    "request_expert_page_cache_nonresident_bytes_before_read",
    "request_expert_page_cache_unclassified_bytes",
    "request_expert_page_cache_resident_fraction_before_read",
    "request_expert_page_cache_nonresident_fraction_before_read",
    "request_expert_page_cache_nonresident_bytes_per_generated_token",
    "request_expert_evictions",
    "dspark_rounds",
    "dspark_proposed_tokens",
    "dspark_accepted_tokens",
    "dspark_committed_tokens",
    "dspark_output_budget_trimmed_tokens",
    "dspark_fallback",
    "dspark_fallback_would_trigger_rounds",
    "dspark_fallback_triggered_rounds",
    "dspark_fallback_cost_ratios",
    "dspark_last_fallback_target_step_seconds",
    "dspark_last_fallback_speculative_seconds",
    "dspark_last_fallback_break_even_seconds",
    "dspark_last_fallback_cost_ratio",
    "dspark_round_trace",
    "dspark_block_verification_rounds",
    "dspark_sequential_verification_rounds",
    "dspark_hybrid_verification_rounds",
    "dspark_hybrid_attention_layers",
    "dspark_hybrid_attention_token_calls",
    "dspark_hybrid_ffn_token_calls",
    "dspark_hybrid_moe_token_calls",
    "dspark_last_verification_mode",
    "dspark_last_sequential_position_seconds",
    "dspark_last_hybrid_verification_positions",
    "dspark_draft_seconds",
    "dspark_verification_seconds",
    "dspark_verification_expert_union_calls",
    "dspark_verification_routed_expert_assignments",
    "dspark_verification_expert_union_experts",
    "dspark_verification_expert_union_reused_assignments",
    "dspark_verification_expert_union_reuse_rate",
    "dspark_verification_expert_union_misses",
    "dspark_target_expert_bytes_read",
    "dspark_verification_expert_bytes_read",
    "dspark_replay_expert_bytes_read",
    "dspark_target_expert_read_seconds",
    "dspark_last_verification_expert_union_layer_ids",
    "dspark_last_verification_expert_assignments_by_layer",
    "dspark_last_verification_expert_union_by_layer",
    "dspark_last_verification_expert_misses_by_layer",
    "dspark_last_verification_expert_bytes_read",
    "dspark_hash_prefetch_requested_experts",
    "dspark_hash_prefetch_cache_resident_experts",
    "dspark_hash_prefetch_experts_read",
    "dspark_hash_prefetch_bytes_read",
    "dspark_hash_prefetch_useful_bytes",
    "dspark_hash_prefetch_wasted_bytes",
    "dspark_hash_prefetch_page_cache_classified_bytes",
    "dspark_hash_prefetch_page_cache_resident_bytes_before_read",
    "dspark_hash_prefetch_page_cache_nonresident_bytes_before_read",
    "dspark_hash_prefetch_page_cache_unclassified_bytes",
    "dspark_hash_prefetch_useful_page_cache_resident_bytes_before_read",
    "dspark_hash_prefetch_useful_page_cache_nonresident_bytes_before_read",
    "dspark_hash_prefetch_useful_page_cache_unclassified_bytes",
    "dspark_hash_prefetch_wasted_page_cache_resident_bytes_before_read",
    "dspark_hash_prefetch_wasted_page_cache_nonresident_bytes_before_read",
    "dspark_hash_prefetch_wasted_page_cache_unclassified_bytes",
    "dspark_hash_prefetch_useful_rate",
    "dspark_hash_prefetch_wait_seconds",
    "dspark_adaptive_block_decisions",
    "dspark_adaptive_block_original_tokens",
    "dspark_adaptive_block_selected_tokens",
    "dspark_adaptive_block_trimmed_tokens",
    "dspark_adaptive_block_expected_committed",
    "dspark_adaptive_block_requested_hash_experts",
    "dspark_adaptive_block_resident_hash_experts",
    "dspark_adaptive_block_missing_hash_experts",
    "dspark_adaptive_block_predicted_hash_bytes",
    "dspark_adaptive_block_predicted_hash_bytes_per_committed_token",
    "dspark_adaptive_block_high_confidence_full_decisions",
    "dspark_adaptive_block_storage_score_decisions",
    "dspark_adaptive_block_full_block_decisions",
    "dspark_adaptive_block_selected_length_counts",
    "dspark_adaptive_block_plan_seconds",
    "dspark_last_adaptive_block_candidate_tokens",
    "dspark_last_adaptive_block_expected_committed",
    "dspark_last_adaptive_block_requested_hash_experts",
    "dspark_last_adaptive_block_resident_hash_experts",
    "dspark_last_adaptive_block_missing_hash_experts",
    "dspark_last_adaptive_block_predicted_hash_bytes",
    "dspark_last_adaptive_block_scores",
    "dspark_last_adaptive_block_selected_score",
    "dspark_last_adaptive_block_selection_reason",
    "dspark_last_adaptive_block_full_commit_fraction",
    "dspark_last_hash_prefetch_union_by_layer",
    "dspark_draft_expert_bytes_read",
    "dspark_draft_page_cache_probe_calls",
    "dspark_draft_page_cache_probe_failures",
    "dspark_draft_page_cache_classified_bytes",
    "dspark_draft_page_cache_resident_bytes_before_read",
    "dspark_draft_page_cache_nonresident_bytes_before_read",
    "dspark_draft_page_cache_unclassified_bytes",
    "dspark_draft_page_cache_nonresident_bytes_per_committed_token",
    "dspark_speculative_expert_bytes_read",
    "dspark_draft_expert_bytes_per_committed_token",
    "dspark_target_expert_bytes_per_committed_token",
    "dspark_speculative_expert_bytes_per_committed_token",
    "dspark_expert_cache_hit_rate",
    "dspark_expert_cache_hits",
    "dspark_expert_cache_misses",
    "dspark_expert_evictions",
    "dspark_expert_bytes_read",
    "dspark_expert_read_seconds",
    "dspark_expert_upload_seconds",
    "dspark_expert_pack_seconds",
    "dspark_expert_resident_slots",
    "dspark_expert_capacity_slots",
)
SUMMARY_METRICS = (
    "request_seconds",
    "time_to_first_token_seconds",
    "decode_tokens_per_second",
    "peak_memory_bytes",
    "request_expert_bytes_read",
    "request_process_disk_bytes_read",
    "request_expert_evictions",
    "dspark_rounds",
    "dspark_proposed_tokens",
    "dspark_accepted_tokens",
    "dspark_committed_tokens",
    "dspark_output_budget_trimmed_tokens",
    "dspark_last_fallback_cost_ratio",
    "dspark_draft_seconds",
    "dspark_verification_seconds",
    "dspark_target_expert_bytes_read",
    "dspark_replay_expert_bytes_read",
    "dspark_hash_prefetch_requested_experts",
    "dspark_hash_prefetch_experts_read",
    "dspark_hash_prefetch_bytes_read",
    "dspark_hash_prefetch_useful_bytes",
    "dspark_hash_prefetch_wasted_bytes",
    "dspark_hash_prefetch_useful_rate",
    "dspark_hash_prefetch_bytes_per_committed_token",
    "dspark_hash_prefetch_wait_seconds",
    "dspark_adaptive_block_decisions",
    "dspark_adaptive_block_original_tokens",
    "dspark_adaptive_block_selected_tokens",
    "dspark_adaptive_block_trimmed_tokens",
    "dspark_adaptive_block_predicted_hash_bytes",
    "dspark_adaptive_block_high_confidence_full_decisions",
    "dspark_adaptive_block_storage_score_decisions",
    "dspark_adaptive_block_full_block_decisions",
    "dspark_adaptive_block_plan_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_tree_sha256(runtime_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(runtime_root.rglob("*.py")):
        digest.update(path.relative_to(runtime_root).as_posix().encode())
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


def _change_fraction(control: float, candidate: float) -> float | None:
    return (candidate - control) / control if control else None


def _median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row["metrics"][key]) for row in rows)


def _mode_summary(rows: list[dict]) -> dict[str, float]:
    return {key: _median(rows, key) for key in SUMMARY_METRICS}


def _adaptive_decision_summary(rows: list[dict]) -> dict:
    adaptive = [row for row in rows if row["mode"] == "adaptive"]
    length_counts: dict[int, int] = {}
    for row in adaptive:
        for length, count in (
            row["metrics"]["dspark_adaptive_block_selected_length_counts"] or ()
        ):
            length = int(length)
            length_counts[length] = length_counts.get(length, 0) + int(count)
    decisions = sum(
        int(row["metrics"]["dspark_adaptive_block_decisions"] or 0)
        for row in adaptive
    )
    selected_tokens = sum(
        int(row["metrics"]["dspark_adaptive_block_selected_tokens"] or 0)
        for row in adaptive
    )
    return {
        "runs": len(adaptive),
        "fallback_runs": sum(bool(row["metrics"]["dspark_fallback"]) for row in adaptive),
        "decisions": decisions,
        "high_confidence_full_decisions": sum(
            int(
                row["metrics"][
                    "dspark_adaptive_block_high_confidence_full_decisions"
                ]
                or 0
            )
            for row in adaptive
        ),
        "storage_score_decisions": sum(
            int(
                row["metrics"]["dspark_adaptive_block_storage_score_decisions"]
                or 0
            )
            for row in adaptive
        ),
        "full_block_decisions": sum(
            int(row["metrics"]["dspark_adaptive_block_full_block_decisions"] or 0)
            for row in adaptive
        ),
        "selected_length_counts": [
            [length, count] for length, count in sorted(length_counts.items())
        ],
        "mean_selected_tokens_per_decision": (
            selected_tokens / decisions if decisions else 0.0
        ),
        "output_budget_trimmed_tokens": sum(
            int(row["metrics"]["dspark_output_budget_trimmed_tokens"] or 0)
            for row in adaptive
        ),
    }


def _workload_summary(name: str, rows: list[dict]) -> dict:
    modes = {row["mode"] for row in rows}
    prompt_hashes = {
        row["metrics"]["prompt_token_sha256"] for row in rows
    }
    output_hashes = {row["metrics"]["token_sha256"] for row in rows}
    generated_counts = {row["metrics"]["generated_tokens"] for row in rows}
    output_consistent = (
        len(prompt_hashes) == 1
        and len(output_hashes) == 1
        and len(generated_counts) == 1
    )
    summary = {
        "name": name,
        "greedy_parity": (
            output_consistent if {"fixed", "adaptive"}.issubset(modes) else None
        ),
        "normal_fixed_greedy_parity": (
            output_consistent if {"normal", "fixed"}.issubset(modes) else None
        ),
        "normal_adaptive_greedy_parity": (
            output_consistent if {"normal", "adaptive"}.issubset(modes) else None
        ),
        "within_run_set_output_consistency": output_consistent,
        "prompt_token_sha256": sorted(prompt_hashes),
        "output_token_sha256": sorted(output_hashes),
        "generated_token_counts": sorted(generated_counts),
        "mode_medians": {
            mode: _mode_summary([row for row in rows if row["mode"] == mode])
            for mode in MODE_ARGUMENTS
            if any(row["mode"] == mode for row in rows)
        },
        "adaptive_decision_summary": _adaptive_decision_summary(rows),
    }
    fixed = [row for row in rows if row["mode"] == "fixed"]
    adaptive = [row for row in rows if row["mode"] == "adaptive"]
    if len(fixed) == 2 and len(adaptive) == 2:
        fixed_summary = summary["mode_medians"]["fixed"]
        adaptive_summary = summary["mode_medians"]["adaptive"]
        summary.update(
            {
                "fixed_medians": fixed_summary,
                "adaptive_medians": adaptive_summary,
                "adaptive_change_fraction": {
                    key: _change_fraction(fixed_summary[key], adaptive_summary[key])
                    for key in SUMMARY_METRICS
                },
                "paired_adaptive_change_fraction": {
                    key: [
                        _change_fraction(
                            float(fixed[index]["metrics"][key]),
                            float(adaptive[index]["metrics"][key]),
                        )
                        for index in range(2)
                    ]
                    for key in SUMMARY_METRICS
                },
            }
        )
    normal = [row for row in rows if row["mode"] == "normal"]
    for candidate_mode, candidates in (
        ("fixed", fixed),
        ("adaptive", adaptive),
    ):
        if len(normal) != 1 or len(candidates) != 1:
            continue
        reference_tokens = normal[0]["metrics"]["generated_token_ids"]
        candidate_tokens = candidates[0]["metrics"]["generated_token_ids"]
        common_length = min(len(reference_tokens), len(candidate_tokens))
        mismatches = [
            index
            for index, (reference, candidate) in enumerate(
                zip(
                    reference_tokens[:common_length],
                    candidate_tokens[:common_length],
                    strict=True,
                )
            )
            if reference != candidate
        ]
        if len(reference_tokens) != len(candidate_tokens):
            mismatches.extend(
                range(common_length, max(len(reference_tokens), len(candidate_tokens)))
            )
        first_mismatch = mismatches[0] if mismatches else None
        summary[f"normal_{candidate_mode}_token_comparison"] = {
            "exact": not mismatches,
            "first_mismatch_index": first_mismatch,
            "mismatch_count": len(mismatches),
            "first_mismatch_normal_token": (
                reference_tokens[first_mismatch]
                if first_mismatch is not None
                and first_mismatch < len(reference_tokens)
                else None
            ),
            f"first_mismatch_{candidate_mode}_token": (
                candidate_tokens[first_mismatch]
                if first_mismatch is not None
                and first_mismatch < len(candidate_tokens)
                else None
            ),
        }
    return summary


def _aggregate_summary(workloads: list[dict]) -> dict:
    if not all("adaptive_change_fraction" in workload for workload in workloads):
        adaptive = [
            workload["adaptive_decision_summary"] for workload in workloads
        ]
        length_counts: dict[int, int] = {}
        for summary in adaptive:
            for length, count in summary["selected_length_counts"]:
                length_counts[length] = length_counts.get(length, 0) + count
        decisions = sum(summary["decisions"] for summary in adaptive)
        return {
            "all_workloads_greedy_parity": None,
            "all_workloads_normal_fixed_greedy_parity": (
                all(
                    workload["normal_fixed_greedy_parity"]
                    for workload in workloads
                )
                if all(
                    workload["normal_fixed_greedy_parity"] is not None
                    for workload in workloads
                )
                else None
            ),
            "all_workloads_normal_adaptive_greedy_parity": (
                all(
                    workload["normal_adaptive_greedy_parity"]
                    for workload in workloads
                )
                if all(
                    workload["normal_adaptive_greedy_parity"] is not None
                    for workload in workloads
                )
                else None
            ),
            "all_workloads_within_run_set_output_consistency": all(
                workload["within_run_set_output_consistency"]
                for workload in workloads
            ),
            "workloads_measured": len(workloads),
            "adaptive_decisions": decisions,
            "adaptive_high_confidence_full_decisions": sum(
                summary["high_confidence_full_decisions"] for summary in adaptive
            ),
            "adaptive_storage_score_decisions": sum(
                summary["storage_score_decisions"] for summary in adaptive
            ),
            "adaptive_full_block_decisions": sum(
                summary["full_block_decisions"] for summary in adaptive
            ),
            "adaptive_selected_length_counts": [
                [length, count] for length, count in sorted(length_counts.items())
            ],
            "adaptive_fallback_runs": sum(
                summary["fallback_runs"] for summary in adaptive
            ),
        }
    changes = {}
    for key in SUMMARY_METRICS:
        values = [
            workload["adaptive_change_fraction"][key]
            for workload in workloads
            if workload["adaptive_change_fraction"][key] is not None
        ]
        changes[key] = statistics.median(values) if values else None
    return {
        "all_workloads_greedy_parity": all(
            workload["greedy_parity"] for workload in workloads
        ),
        "all_workloads_normal_fixed_greedy_parity": (
            all(workload["normal_fixed_greedy_parity"] for workload in workloads)
            if all(
                workload["normal_fixed_greedy_parity"] is not None
                for workload in workloads
            )
            else None
        ),
        "all_workloads_normal_adaptive_greedy_parity": (
            all(
                workload["normal_adaptive_greedy_parity"]
                for workload in workloads
            )
            if all(
                workload["normal_adaptive_greedy_parity"] is not None
                for workload in workloads
            )
            else None
        ),
        "workloads_measured": len(workloads),
        "workloads_with_lower_request_logical_expert_bytes": sum(
            workload["adaptive_change_fraction"]["request_expert_bytes_read"] < 0
            for workload in workloads
        ),
        "workloads_with_lower_target_logical_expert_bytes": sum(
            workload["adaptive_change_fraction"][
                "dspark_target_expert_bytes_read"
            ]
            < 0
            for workload in workloads
        ),
        "workloads_with_lower_wasted_prefetch_bytes": sum(
            (
                workload["adaptive_change_fraction"][
                    "dspark_hash_prefetch_wasted_bytes"
                ]
                is not None
                and workload["adaptive_change_fraction"][
                    "dspark_hash_prefetch_wasted_bytes"
                ]
                < 0
            )
            for workload in workloads
        ),
        "median_adaptive_change_fraction_across_workloads": changes,
    }


def _load_prompts(manifest_path: Path, names: tuple[str, ...]) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    by_name = {prompt["name"]: prompt for prompt in manifest["prompts"]}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"prompt manifest is missing: {', '.join(missing)}")
    prompts = []
    for name in names:
        prompt = dict(by_name[name])
        path = manifest_path.parent / prompt["file"]
        if not path.is_file():
            raise ValueError(f"prompt file is missing: {path}")
        if _sha256(path) != prompt["text_sha256"]:
            raise ValueError(f"prompt text hash mismatch: {path}")
        prompt["path"] = str(path)
        prompts.append(prompt)
    return prompts


def _run(
    *,
    project_root: Path,
    python: Path,
    model: Path,
    prompt: dict,
    mode: str,
    run_id: str,
    raw_directory: Path,
    max_tokens: int,
    resume: bool,
    dspark_fallback_enabled: bool,
    layer_major_prefill_enabled: bool,
    sequential_verification_enabled: bool,
    hybrid_verification_enabled: bool,
    hybrid_hash_prefetch_enabled: bool,
    hash_prefetch_enabled: bool | None = None,
    dspark_slots: int | None = None,
    expert_page_cache_probe_enabled: bool = False,
    expert_file_cache_policy: str = "cached",
) -> dict:
    metrics_path = raw_directory / f"{run_id}.metrics.json"
    stdout_path = raw_directory / f"{run_id}.stdout.txt"
    stderr_path = raw_directory / f"{run_id}.stderr.txt"
    if not (resume and metrics_path.is_file()):
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
            expert_file_cache_policy,
            *(
                ("--expert-page-cache-probe",)
                if expert_page_cache_probe_enabled
                else ()
            ),
            *(("--no-layer-major-prefill",) if not layer_major_prefill_enabled else ()),
            *MODE_ARGUMENTS[mode],
            *(
                ("--dspark-hash-prefetch",)
                if (
                    hash_prefetch_enabled
                    if hash_prefetch_enabled is not None
                    else mode != "normal"
                    and not sequential_verification_enabled
                    and (
                        not hybrid_verification_enabled
                        or hybrid_hash_prefetch_enabled
                    )
                )
                else ()
            ),
            *(
                ("--dspark-sequential-verification",)
                if mode != "normal" and sequential_verification_enabled
                else ()
            ),
            *(
                ("--dspark-hybrid-verification",)
                if mode != "normal" and hybrid_verification_enabled
                else ()
            ),
            *(
                ("--dspark-confidence-threshold", "0")
                if mode != "normal"
                else ()
            ),
            *(
                ("--dspark-slots", str(dspark_slots))
                if mode != "normal" and dspark_slots is not None
                else ()
            ),
            *(
                ("--no-dspark-fallback",)
                if mode != "normal" and not dspark_fallback_enabled
                else ()
            ),
            "--metrics-json",
            str(metrics_path),
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
        if completed.returncode:
            tail = stderr_path.read_text(encoding="utf-8")[-2_000:]
            raise RuntimeError(f"{run_id} failed ({completed.returncode}):\n{tail}")
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    if metrics.get("prompt_token_sha256") != prompt["prompt_token_sha256"]:
        raise RuntimeError(f"{run_id} prompt token hash does not match the manifest")
    return {
        "id": run_id,
        "workload": prompt["name"],
        "mode": mode,
        "raw_metrics_sha256": _sha256(metrics_path),
        "metrics": {key: metrics.get(key) for key in ROW_METRICS},
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed/adaptive DSpark calibration or decision survey"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=list(WORKLOAD_ORDER),
        metavar="NAME",
    )
    parser.add_argument(
        "--run-pattern",
        nargs="+",
        choices=tuple(MODE_ARGUMENTS),
        default=list(RUN_PATTERN),
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--no-dspark-fallback", action="store_true")
    parser.add_argument("--no-layer-major-prefill", action="store_true")
    parser.add_argument(
        "--sequential-verification",
        action="store_true",
        help=(
            "use the sequential target oracle for DSpark modes and omit hash "
            "prefetch"
        ),
    )
    parser.add_argument(
        "--hybrid-verification",
        action="store_true",
        help=(
            "use token-shaped target math with one MoE expert-union "
            "acquisition per layer for DSpark modes and omit hash prefetch"
        ),
    )
    parser.add_argument(
        "--hybrid-hash-prefetch",
        action="store_true",
        help="compose exact hash-layer prefetch with hybrid verification",
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens < 3:
        parser.error("--max-tokens must be at least 3")
    if arguments.sequential_verification and arguments.hybrid_verification:
        parser.error(
            "--sequential-verification and --hybrid-verification are mutually exclusive"
        )
    if arguments.hybrid_hash_prefetch and not arguments.hybrid_verification:
        parser.error("--hybrid-hash-prefetch requires --hybrid-verification")

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    workload_order = tuple(arguments.workloads)
    run_pattern = tuple(arguments.run_pattern)
    prompts = _load_prompts(prompt_manifest, workload_order)
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    timed_total = len(prompts) * len(run_pattern)
    timed_completed = 0
    for prompt in prompts:
        if not arguments.skip_warmup:
            warmup_id = f"{prompt['name']}-warmup-fixed"
            print(f"[warmup] {prompt['name']} fixed", flush=True)
            _run(
                project_root=project_root,
                python=python,
                model=model,
                prompt=prompt,
                mode="fixed",
                run_id=warmup_id,
                raw_directory=raw_directory,
                max_tokens=arguments.max_tokens,
                resume=arguments.resume,
                dspark_fallback_enabled=not arguments.no_dspark_fallback,
                layer_major_prefill_enabled=(
                    not arguments.no_layer_major_prefill
                ),
                sequential_verification_enabled=(
                    arguments.sequential_verification
                ),
                hybrid_verification_enabled=arguments.hybrid_verification,
                hybrid_hash_prefetch_enabled=arguments.hybrid_hash_prefetch,
            )
        for sequence, mode in enumerate(run_pattern, start=1):
            timed_completed += 1
            run_id = f"{prompt['name']}-{sequence:02d}-{mode}"
            print(
                f"[{timed_completed}/{timed_total}] {prompt['name']} {mode}",
                flush=True,
            )
            row = _run(
                project_root=project_root,
                python=python,
                model=model,
                prompt=prompt,
                mode=mode,
                run_id=run_id,
                raw_directory=raw_directory,
                max_tokens=arguments.max_tokens,
                resume=arguments.resume,
                dspark_fallback_enabled=not arguments.no_dspark_fallback,
                layer_major_prefill_enabled=(
                    not arguments.no_layer_major_prefill
                ),
                sequential_verification_enabled=(
                    arguments.sequential_verification
                ),
                hybrid_verification_enabled=arguments.hybrid_verification,
                hybrid_hash_prefetch_enabled=arguments.hybrid_hash_prefetch,
            )
            row["sequence"] = sequence
            runs.append(row)

    summaries = [
        _workload_summary(
            prompt["name"],
            [row for row in runs if row["workload"] == prompt["name"]],
        )
        for prompt in prompts
    ]
    aggregate = _aggregate_summary(summaries)
    hybrid_candidate_mode = (
        "adaptive"
        if arguments.hybrid_verification and "adaptive" in run_pattern
        else "fixed"
    )
    hybrid_parity_key = (
        "all_workloads_normal_adaptive_greedy_parity"
        if hybrid_candidate_mode == "adaptive"
        else "all_workloads_normal_fixed_greedy_parity"
    )
    installed_manifest = json.loads((model / "manifest.json").read_text())
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": (
            "dspark_sequential_verification_oracle"
            if arguments.sequential_verification
            else "dspark_hybrid_hash_adaptive_validation"
            if arguments.hybrid_hash_prefetch and hybrid_candidate_mode == "adaptive"
            else "dspark_hybrid_hash_prefetch_validation"
            if arguments.hybrid_hash_prefetch
            else "dspark_hybrid_verification_validation"
            if arguments.hybrid_verification
            else (
                "exploratory_adaptive_block_abba"
                if run_pattern == RUN_PATTERN
                else "exploratory_adaptive_block_decision_survey"
            )
        ),
        "formal_performance_result": False,
        "source": _source_state(project_root),
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
            "layer_count": installed_manifest["layerCount"],
            "expert_count": installed_manifest["expertCount"],
            "selected_expert_count": installed_manifest["selectedExpertCount"],
            "expert_blob_bytes": installed_manifest["expertBlobSize"],
            "dspark_block_size": installed_manifest["dspark"]["blockSize"],
        },
        "experiment": {
            "objective": (
                "Validate storage-aware adaptive block selection composed with "
                "exact hash-layer prefetch and token-shaped hybrid verification"
                if arguments.hybrid_hash_prefetch
                and hybrid_candidate_mode == "adaptive"
                else "Validate exact hash-layer prefetch composed with token-shaped "
                "target math and one MoE expert-union acquisition per layer"
                if arguments.hybrid_hash_prefetch
                else "Validate token-shaped target math with one MoE "
                "expert-union acquisition per layer against normal target decode"
                if arguments.hybrid_verification
                else "Measure fixed five-token hash prefetch and/or the "
                "default-off storage-aware adaptive selector on selected R0 domains"
            ),
            "control": (
                "fixed DSpark with sequential target verification"
                if arguments.sequential_verification
                else "normal target decode"
                if arguments.hybrid_verification
                else "fixed five-token DSpark with hash exact prefetch"
            ),
            "candidate": (
                "adaptive DSpark prefix with sequential target verification"
                if arguments.sequential_verification
                else "Storage-aware adaptive DSpark prefix with hybrid exact "
                "verification and hash-layer prefetch"
                if arguments.hybrid_hash_prefetch
                and hybrid_candidate_mode == "adaptive"
                else "DSpark hybrid verification with exact hash-layer prefetch"
                if arguments.hybrid_hash_prefetch
                else "DSpark with token-shaped hybrid target verification and "
                "one expert-union acquisition per layer"
                if arguments.hybrid_verification
                else "adaptive DSpark prefix with hash exact prefetch"
            ),
            "workload_order": list(workload_order),
            "run_pattern_per_workload": list(run_pattern),
            "warmup_per_workload": not arguments.skip_warmup,
            "fresh_process_per_run": True,
            "os_page_cache": "not purged",
            "prompt_cache": "persistent cache disabled",
            "prompt_tokens": sorted({prompt["target_tokens"] for prompt in prompts}),
            "max_output_tokens": arguments.max_tokens,
            "batch_size": 1,
            "temperature": 0,
            "top_p": 1,
            "dspark_confidence_threshold": 0,
            "dspark_fallback_enabled": not arguments.no_dspark_fallback,
            "dspark_hash_prefetch_enabled": (
                any(mode != "normal" for mode in run_pattern)
                and not arguments.sequential_verification
                and (
                    not arguments.hybrid_verification
                    or arguments.hybrid_hash_prefetch
                )
            ),
            "dspark_sequential_verification": (
                arguments.sequential_verification
            ),
            "dspark_hybrid_verification": arguments.hybrid_verification,
            "dspark_hybrid_hash_prefetch": arguments.hybrid_hash_prefetch,
            "layer_major_prefill_enabled": (
                not arguments.no_layer_major_prefill
            ),
            "adaptive_full_block_commit_fraction_floor": 0.90,
            "main_expert_slots": 1152,
            "dspark_expert_slots": 768,
            "hash_prefetch_scratch_slots": 108,
        },
        "prompt_manifest_sha256": _sha256(prompt_manifest),
        "workloads": [
            {key: value for key, value in prompt.items() if key != "path"}
            for prompt in prompts
        ],
        "runs": runs,
        "workload_summaries": summaries,
        "aggregate_summary": aggregate,
        "correctness_gate": (
            {
                "candidate_mode": hybrid_candidate_mode,
                "normal_candidate_token_parity_passed": aggregate[
                    hybrid_parity_key
                ],
                "decision": (
                    "continue_multi_workload_validation"
                    if aggregate[hybrid_parity_key]
                    else "stop_and_isolate_remaining_verifier_difference"
                ),
            }
            if arguments.hybrid_verification
            else None
        ),
        "evidence_limits": [
            (
                "This is one ABBA wave per workload, not a formal repeated-wave result."
                if run_pattern == RUN_PATTERN
                else "This is a decision survey, not a paired fixed/adaptive performance result."
            ),
            "Prompt and output lengths are recorded in experiment; a survey run is behavioral evidence, not a paired timing result.",
            (
                "Each workload has an untimed same-prompt warmup, but the operating-system page cache is not controlled or purged."
                if not arguments.skip_warmup
                else "No same-prompt warmup was run, and the operating-system page cache is not controlled or purged."
            ),
            "Process disk counters include all reads attributed to the process and do not isolate physical expert-file SSD traffic.",
            "Adaptive selection only models the three exact target hash layers and still computes the complete five-position DSpark draft.",
            *(
                [
                    "DSpark fallback was disabled as an explicit research control; would-trigger counters must not be interpreted as production performance."
                ]
                if arguments.no_dspark_fallback
                else []
            ),
            *(
                [
                    "Sequential verification is a correctness oracle, not a block-verification performance candidate; hash prefetch is disabled in this mode."
                ]
                if arguments.sequential_verification
                else []
            ),
            *(
                [
                    "Hybrid verification evaluates target math tokenwise after one expert-union acquisition per layer and is a correctness candidate; timing is not adoption evidence."
                ]
                if arguments.hybrid_verification
                else []
            ),
            *(
                [
                    "Exact hash-layer prefetch is composed only for correctness and byte-accounting validation; process disk counters do not isolate physical expert-file reads."
                ]
                if arguments.hybrid_hash_prefetch
                else []
            ),
            "Timing, process-disk, and peak-memory changes remain exploratory until repeated waves establish stable paired results."
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
