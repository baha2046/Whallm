from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from .benchmark_dspark_adaptive import (
        _command_output,
        _load_prompts,
        _package_version,
        _run,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )
except ImportError:
    from benchmark_dspark_adaptive import (
        _command_output,
        _load_prompts,
        _package_version,
        _run,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )


EXPECTED_OUTPUT_TOKEN_SHA256 = (
    "03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3"
)
MODES = ("normal", "sequential", "grouped", "hybrid")
DSPARK_MODES = ("sequential", "grouped", "hybrid")
PERFORMANCE_WAVES = (
    ("normal", "sequential", "grouped", "hybrid"),
    ("sequential", "grouped", "hybrid", "normal"),
    ("grouped", "hybrid", "normal", "sequential"),
    ("hybrid", "normal", "sequential", "grouped"),
)
OBSERVER_PATTERN = DSPARK_MODES
SUMMARY_METRICS = (
    "request_seconds",
    "time_to_first_token_seconds",
    "decode_tokens_per_second",
    "peak_memory_bytes",
    "request_expert_bytes_read",
    "request_process_disk_bytes_read",
    "dspark_rounds",
    "dspark_proposed_tokens",
    "dspark_accepted_tokens",
    "dspark_committed_tokens",
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
)


def _change_fraction(control: float, candidate: float) -> float | None:
    return (candidate - control) / control if control else None


def _mode_summary(rows: list[dict]) -> dict:
    summary: dict[str, float | int | None] = {}
    for key in SUMMARY_METRICS:
        values = [row["metrics"].get(key) for row in rows]
        numeric = [float(value) for value in values if value is not None]
        summary[key] = statistics.median(numeric) if numeric else None
    committed = summary["dspark_committed_tokens"]
    verification = summary["dspark_verification_seconds"]
    target_bytes = summary["dspark_target_expert_bytes_read"]
    summary["dspark_verification_seconds_per_committed_token"] = (
        verification / committed if verification is not None and committed else 0.0
    )
    summary["dspark_target_expert_bytes_per_committed_token"] = (
        target_bytes / committed if target_bytes is not None and committed else 0.0
    )
    summary["runs"] = len(rows)
    return summary


def _mode_arguments(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown verifier mode: {mode}")
    return {
        "mode": "normal" if mode == "normal" else "fixed",
        "sequential_verification_enabled": mode == "sequential",
        "hybrid_verification_enabled": mode == "hybrid",
    }


def _run_mode(
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
    probe: bool,
) -> dict:
    arguments = _mode_arguments(mode)
    row = _run(
        project_root=project_root,
        python=python,
        model=model,
        prompt=prompt,
        run_id=run_id,
        raw_directory=raw_directory,
        max_tokens=max_tokens,
        resume=resume,
        dspark_fallback_enabled=False,
        layer_major_prefill_enabled=False,
        hybrid_hash_prefetch_enabled=False,
        hash_prefetch_enabled=False,
        expert_page_cache_probe_enabled=probe,
        expert_file_cache_policy="bypass",
        **arguments,
    )
    row["mode"] = mode
    return row


def _shape_check(row: dict) -> dict:
    mode = row["mode"]
    metrics = row["metrics"]
    if mode == "normal":
        criteria = {
            "dspark_rounds_zero": int(metrics["dspark_rounds"]) == 0,
            "union_calls_zero": int(
                metrics["dspark_verification_expert_union_calls"]
            )
            == 0,
        }
        return {"id": row["id"], "mode": mode, "criteria": criteria, "exact": all(criteria.values())}

    expected_round_key = {
        "sequential": "dspark_sequential_verification_rounds",
        "grouped": "dspark_block_verification_rounds",
        "hybrid": "dspark_hybrid_verification_rounds",
    }[mode]
    expected_verification_mode = {
        "sequential": "sequential",
        "grouped": "block",
        "hybrid": "hybrid",
    }[mode]
    expected_calls = 258 if mode == "sequential" else 43
    criteria = {
        "one_round": int(metrics["dspark_rounds"]) == 1,
        "five_proposed": int(metrics["dspark_proposed_tokens"]) == 5,
        "five_accepted": int(metrics["dspark_accepted_tokens"]) == 5,
        "six_committed": int(metrics["dspark_committed_tokens"]) == 6,
        "zero_replay_bytes": int(metrics["dspark_replay_expert_bytes_read"]) == 0,
        "mode_round_exact": int(metrics[expected_round_key]) == 1,
        "last_mode_exact": metrics["dspark_last_verification_mode"]
        == expected_verification_mode,
        "union_calls_exact": int(
            metrics["dspark_verification_expert_union_calls"]
        )
        == expected_calls,
        "assignments_exact": int(
            metrics["dspark_verification_routed_expert_assignments"]
        )
        == 1_548,
    }
    if mode == "hybrid":
        criteria.update(
            {
                "hybrid_attention_layers_exact": int(
                    metrics["dspark_hybrid_attention_layers"]
                )
                == 43,
                "hybrid_attention_token_calls_exact": int(
                    metrics["dspark_hybrid_attention_token_calls"]
                )
                == 258,
                "hybrid_ffn_token_calls_exact": int(
                    metrics["dspark_hybrid_ffn_token_calls"]
                )
                == 258,
                "hybrid_moe_token_calls_exact": int(
                    metrics["dspark_hybrid_moe_token_calls"]
                )
                == 258,
            }
        )
    return {
        "id": row["id"],
        "mode": mode,
        "criteria": criteria,
        "exact": all(criteria.values()),
    }


def _partition(
    metrics: dict,
    *,
    logical_key: str,
    classified_key: str,
    resident_key: str,
    nonresident_key: str,
    unclassified_key: str,
    failures_key: str,
) -> dict:
    logical = int(metrics[logical_key])
    classified = int(metrics[classified_key])
    resident = int(metrics[resident_key])
    nonresident = int(metrics[nonresident_key])
    unclassified = int(metrics[unclassified_key])
    failures = int(metrics[failures_key])
    exact = bool(
        failures == 0
        and unclassified == 0
        and resident + nonresident == classified
        and classified + unclassified == logical
    )
    return {
        "logical_bytes": logical,
        "classified_bytes": classified,
        "resident_bytes_before_read": resident,
        "nonresident_bytes_before_read": nonresident,
        "unclassified_bytes": unclassified,
        "probe_failures": failures,
        "accounting_exact": exact,
    }


def _observer_accounting(row: dict) -> dict:
    metrics = row["metrics"]
    main = _partition(
        metrics,
        logical_key="request_expert_bytes_read",
        classified_key="request_expert_page_cache_classified_bytes",
        resident_key="request_expert_page_cache_resident_bytes_before_read",
        nonresident_key="request_expert_page_cache_nonresident_bytes_before_read",
        unclassified_key="request_expert_page_cache_unclassified_bytes",
        failures_key="request_expert_page_cache_probe_failures",
    )
    draft = _partition(
        metrics,
        logical_key="dspark_draft_expert_bytes_read",
        classified_key="dspark_draft_page_cache_classified_bytes",
        resident_key="dspark_draft_page_cache_resident_bytes_before_read",
        nonresident_key="dspark_draft_page_cache_nonresident_bytes_before_read",
        unclassified_key="dspark_draft_page_cache_unclassified_bytes",
        failures_key="dspark_draft_page_cache_probe_failures",
    )
    return {
        "id": row["id"],
        "mode": row["mode"],
        "main": main,
        "draft": draft,
        "accounting_exact": main["accounting_exact"] and draft["accounting_exact"],
    }


def _tradeoff_decision(summaries: dict[str, dict]) -> dict:
    sequential = summaries["sequential"]
    grouped = summaries["grouped"]
    hybrid = summaries["hybrid"]
    hybrid_bytes_vs_sequential = _change_fraction(
        sequential["dspark_target_expert_bytes_read"],
        hybrid["dspark_target_expert_bytes_read"],
    )
    hybrid_time_vs_grouped = _change_fraction(
        grouped["dspark_verification_seconds"],
        hybrid["dspark_verification_seconds"],
    )
    union_criteria = {
        "hybrid_one_call_per_layer": int(
            hybrid["dspark_verification_expert_union_calls"]
        )
        == 43,
        "hybrid_preserves_all_assignments": int(
            hybrid["dspark_verification_routed_expert_assignments"]
        )
        == 1_548,
        "hybrid_reuses_assignments": int(
            hybrid["dspark_verification_expert_union_reused_assignments"]
        )
        > 0,
        "hybrid_target_bytes_no_more_than_5_percent_above_sequential": (
            hybrid_bytes_vs_sequential is not None
            and hybrid_bytes_vs_sequential <= 0.05
        ),
    }
    grouped_execution_material = bool(
        hybrid_time_vs_grouped is not None and hybrid_time_vs_grouped >= 0.20
    )
    return {
        "hybrid_union_call_reduction_vs_sequential": _change_fraction(
            sequential["dspark_verification_expert_union_calls"],
            hybrid["dspark_verification_expert_union_calls"],
        ),
        "hybrid_union_experts_change_vs_sequential": _change_fraction(
            sequential["dspark_verification_expert_union_experts"],
            hybrid["dspark_verification_expert_union_experts"],
        ),
        "hybrid_target_bytes_change_vs_sequential": hybrid_bytes_vs_sequential,
        "hybrid_verification_time_change_vs_sequential": _change_fraction(
            sequential["dspark_verification_seconds"],
            hybrid["dspark_verification_seconds"],
        ),
        "hybrid_verification_time_change_vs_grouped": hybrid_time_vs_grouped,
        "grouped_target_bytes_change_vs_hybrid": _change_fraction(
            hybrid["dspark_target_expert_bytes_read"],
            grouped["dspark_target_expert_bytes_read"],
        ),
        "union_contract_criteria": union_criteria,
        "hybrid_union_contract_passed": all(union_criteria.values()),
        "grouped_execution_advantage_material_at_20_percent": (
            grouped_execution_material
        ),
        "dedupe_insufficient_to_overcome_token_shaped_execution_cost": (
            all(union_criteria.values()) and grouped_execution_material
        ),
    }


def _source_state(project_root: Path, contract: Path) -> dict:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    helper = project_root / "Scripts" / "benchmark_dspark_adaptive.py"
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
        "benchmark_helper_sha256": _sha256(helper),
        "cache_bypass_contract_sha256": _sha256(contract),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential, grouped, and hybrid target verification "
            "under the validated expert-file bypass policy"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-bypass-contract", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workload", default="repeated")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens != 8:
        parser.error("this predeclared diagnostic requires --max-tokens 8")

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    contract_path = Path(arguments.cache_bypass_contract).expanduser().resolve()
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not contract["summary"]["contract_passed"]:
        parser.error("cache-bypass contract did not pass")
    prompt = _load_prompts(prompt_manifest, (arguments.workload,))[0]
    if int(prompt["target_tokens"]) != 128:
        parser.error("this predeclared diagnostic requires a 128-token prompt")
    manifest = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    performance_runs: list[dict] = []
    total = sum(len(wave) for wave in PERFORMANCE_WAVES)
    completed = 0
    for wave_index, wave in enumerate(PERFORMANCE_WAVES, start=1):
        for sequence, mode in enumerate(wave, start=1):
            completed += 1
            print(
                f"[performance {completed}/{total}] wave={wave_index} mode={mode}",
                flush=True,
            )
            row = _run_mode(
                project_root=project_root,
                python=python,
                model=model,
                prompt=prompt,
                mode=mode,
                run_id=(
                    f"{prompt['name']}-union-w{wave_index:02d}-"
                    f"{sequence:02d}-{mode}"
                ),
                raw_directory=raw_directory,
                max_tokens=arguments.max_tokens,
                resume=arguments.resume,
                probe=False,
            )
            row.update(wave=wave_index, sequence=sequence, track="performance")
            performance_runs.append(row)

    observer_runs: list[dict] = []
    for sequence, mode in enumerate(OBSERVER_PATTERN, start=1):
        print(f"[observer {sequence}/{len(OBSERVER_PATTERN)}] mode={mode}", flush=True)
        row = _run_mode(
            project_root=project_root,
            python=python,
            model=model,
            prompt=prompt,
            mode=mode,
            run_id=f"{prompt['name']}-union-observer-{sequence:02d}-{mode}",
            raw_directory=raw_directory,
            max_tokens=arguments.max_tokens,
            resume=arguments.resume,
            probe=True,
        )
        row.update(sequence=sequence, track="observer")
        observer_runs.append(row)

    all_runs = [*performance_runs, *observer_runs]
    token_hashes = sorted({row["metrics"]["token_sha256"] for row in all_runs})
    token_gate = token_hashes == [EXPECTED_OUTPUT_TOKEN_SHA256] and all(
        row["metrics"]["generated_token_ids"]
        == all_runs[0]["metrics"]["generated_token_ids"]
        for row in all_runs[1:]
    )
    cache_policy_gate = all(
        row["metrics"]["expert_file_cache_policy"] == "bypass"
        and int(row["metrics"]["expert_file_direct_io_alignment_bytes"]) == 4_096
        for row in all_runs
    )
    shape_checks = [_shape_check(row) for row in all_runs]
    shape_gate = all(item["exact"] for item in shape_checks)
    summaries = {
        mode: _mode_summary([row for row in performance_runs if row["mode"] == mode])
        for mode in MODES
    }
    repetition_gate = all(
        summaries[mode]["runs"] == len(PERFORMANCE_WAVES) for mode in MODES
    )
    observer_accounting = [_observer_accounting(row) for row in observer_runs]
    observer_gate = all(item["accounting_exact"] for item in observer_accounting)
    tradeoff = _tradeoff_decision(summaries)
    validity_gate = bool(
        token_gate
        and cache_policy_gate
        and shape_gate
        and repetition_gate
        and observer_gate
    )
    if not validity_gate:
        status = "verifier_union_tradeoff_invalid"
    elif tradeoff[
        "dedupe_insufficient_to_overcome_token_shaped_execution_cost"
    ]:
        status = "union_dedupe_confirmed_but_token_shaped_execution_cost_material"
    elif tradeoff["hybrid_union_contract_passed"]:
        status = "union_dedupe_confirmed_without_material_grouped_qmm_loss"
    else:
        status = "hybrid_union_contract_failed"

    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "verifier_union_tradeoff_diagnostic",
        "formal_performance_result": False,
        "source": _source_state(project_root, contract_path),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": _sysctl("hw.model", project_root),
            "chip": _sysctl("machdep.cpu.brand_string", project_root),
            "memory_bytes": int(_sysctl("hw.memsize", project_root) or 0),
            "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
        },
        "checkpoint": {
            "installed_model": str(model),
            "model_id": manifest["modelID"],
            "revision": manifest["revision"],
            "layer_count": int(manifest["layerCount"]),
            "selected_experts": int(manifest["selectedExpertCount"]),
            "expert_blob_bytes": int(manifest["expertBlobSize"]),
        },
        "experiment": {
            "workload": arguments.workload,
            "prompt_file": prompt["file"],
            "prompt_tokens": prompt["target_tokens"],
            "max_output_tokens": arguments.max_tokens,
            "expected_output_token_sha256": EXPECTED_OUTPUT_TOKEN_SHA256,
            "performance_waves": [list(wave) for wave in PERFORMANCE_WAVES],
            "observer_pattern": list(OBSERVER_PATTERN),
            "fresh_process_per_run": True,
            "persistent_prompt_cache": False,
            "layer_major_prefill": False,
            "expert_file_cache_policy": "bypass",
            "performance_probe_enabled": False,
            "observer_probe_enabled": True,
            "hash_prefetch": False,
            "adaptive_block": False,
            "fallback_enabled": False,
            "temperature": 0,
            "top_p": 1,
        },
        "prompt_manifest_sha256": _sha256(prompt_manifest),
        "cache_bypass_contract": {
            "path": str(contract_path),
            "sha256": _sha256(contract_path),
            "status": contract["decision"]["status"],
        },
        "performance_runs": performance_runs,
        "observer_runs": observer_runs,
        "summary": {
            "all_run_token_ids_exact": token_gate,
            "output_token_sha256": token_hashes,
            "cache_policy_and_alignment_exact": cache_policy_gate,
            "complete_four_run_repetition": repetition_gate,
            "shape_checks": shape_checks,
            "shape_gate_passed": shape_gate,
            "observer_accounting": observer_accounting,
            "observer_accounting_exact": observer_gate,
            "mode_medians": summaries,
            "tradeoff": tradeoff,
        },
        "decision": {
            "status": status,
            "validity_gate_passed": validity_gate,
            "hybrid_union_contract_passed": tradeoff[
                "hybrid_union_contract_passed"
            ],
            "grouped_execution_advantage_material_at_20_percent": tradeoff[
                "grouped_execution_advantage_material_at_20_percent"
            ],
            "adopt_as_default": False,
        },
        "evidence_limits": [
            "This diagnostic uses one 128-token high-confidence workload and one accepted five-token draft block per run.",
            "The grouped verifier is already rejected by multi-workload low-margin correctness evidence.",
            "Grouped-versus-hybrid timing measures all execution-shape differences and does not isolate routed QMM as a sole cause.",
            "F_NOCACHE does not purge pre-existing resident pages or expose storage-controller/device caches.",
            "Observer timings are excluded because mincore instrumentation adds overhead.",
            "This result diagnoses acquisition and execution shape; it cannot enable DSpark or any verifier by default.",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    if not validity_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
