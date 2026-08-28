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
    from .benchmark_hash_prefetch_page_cache import (
        _accounting_exact,
        _run_accounting,
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
    from benchmark_hash_prefetch_page_cache import (
        _accounting_exact,
        _run_accounting,
    )


PERFORMANCE_WAVES = (
    ("normal", "fixed", "adaptive"),
    ("fixed", "adaptive", "normal"),
    ("adaptive", "normal", "fixed"),
)
OBSERVER_PATTERN = ("normal", "fixed", "adaptive")
PERFORMANCE_METRICS = (
    "request_seconds",
    "time_to_first_token_seconds",
    "decode_tokens_per_second",
    "peak_memory_bytes",
    "request_expert_bytes_read",
    "request_process_disk_bytes_read",
    "request_expert_evictions",
    "dspark_committed_tokens",
    "dspark_draft_expert_bytes_read",
    "dspark_target_expert_bytes_read",
    "dspark_speculative_expert_bytes_read",
    "dspark_draft_expert_bytes_per_committed_token",
    "dspark_target_expert_bytes_per_committed_token",
    "dspark_speculative_expert_bytes_per_committed_token",
    "dspark_hash_prefetch_bytes_read",
    "dspark_hash_prefetch_useful_bytes",
    "dspark_hash_prefetch_wasted_bytes",
    "dspark_hash_prefetch_useful_rate",
)


def _change_fraction(control: float, candidate: float) -> float | None:
    return (candidate - control) / control if control else None


def _mode_summary(rows: list[dict]) -> dict:
    summary = {}
    for key in PERFORMANCE_METRICS:
        values = [row["metrics"].get(key) for row in rows]
        values = [float(value) for value in values if value is not None]
        summary[key] = statistics.median(values) if values else None
    generated = [int(row["metrics"]["generated_tokens"]) for row in rows]
    disk = [row["metrics"].get("request_process_disk_bytes_read") for row in rows]
    disk_per_token = [
        float(value) / tokens
        for value, tokens in zip(disk, generated, strict=True)
        if value is not None and tokens
    ]
    summary["request_process_disk_bytes_per_generated_token"] = (
        statistics.median(disk_per_token) if disk_per_token else None
    )
    summary["runs"] = len(rows)
    return summary


def _candidate_gate(
    normal: dict,
    candidate: dict,
    *,
    fixed: dict | None = None,
) -> dict:
    changes = {
        "request_seconds": _change_fraction(
            normal["request_seconds"], candidate["request_seconds"]
        ),
        "decode_tokens_per_second": _change_fraction(
            normal["decode_tokens_per_second"],
            candidate["decode_tokens_per_second"],
        ),
        "peak_memory_bytes": _change_fraction(
            normal["peak_memory_bytes"], candidate["peak_memory_bytes"]
        ),
        "request_process_disk_bytes_per_generated_token": _change_fraction(
            normal["request_process_disk_bytes_per_generated_token"],
            candidate["request_process_disk_bytes_per_generated_token"],
        ),
    }
    criteria = {
        "request_time_at_least_5_percent_lower": (
            changes["request_seconds"] is not None
            and changes["request_seconds"] <= -0.05
        ),
        "decode_throughput_at_least_5_percent_higher": (
            changes["decode_tokens_per_second"] is not None
            and changes["decode_tokens_per_second"] >= 0.05
        ),
        "peak_memory_no_more_than_15_percent_higher": (
            changes["peak_memory_bytes"] is not None
            and changes["peak_memory_bytes"] <= 0.15
        ),
        "process_disk_per_token_no_more_than_5_percent_higher": (
            changes["request_process_disk_bytes_per_generated_token"]
            is not None
            and changes["request_process_disk_bytes_per_generated_token"]
            <= 0.05
        ),
    }
    if fixed is not None:
        adaptive_bytes = candidate[
            "dspark_speculative_expert_bytes_per_committed_token"
        ]
        fixed_bytes = fixed[
            "dspark_speculative_expert_bytes_per_committed_token"
        ]
        speculative_change = _change_fraction(fixed_bytes, adaptive_bytes)
        changes[
            "speculative_bytes_per_committed_token_vs_fixed"
        ] = speculative_change
        criteria[
            "speculative_bytes_per_committed_no_more_than_5_percent_above_fixed"
        ] = speculative_change is not None and speculative_change <= 0.05
    return {
        "change_fraction": changes,
        "criteria": criteria,
        "candidate_gate_passed": all(criteria.values()),
    }


def _observer_accounting(row: dict) -> dict:
    original_mode = row["mode"]
    accounting_input = dict(row)
    accounting_input["mode"] = (
        "normal" if original_mode == "normal" else "hash"
    )
    accounting = _run_accounting(accounting_input)
    check = dict(accounting)
    check["mode"] = "normal" if original_mode == "normal" else "hash"
    accounting["mode"] = original_mode
    accounting["accounting_exact"] = _accounting_exact(check)
    return accounting


def _source_state(project_root: Path, contract: Path) -> dict:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    helper = project_root / "Scripts" / "benchmark_dspark_adaptive.py"
    accounting_helper = (
        project_root / "Scripts" / "benchmark_hash_prefetch_page_cache.py"
    )
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
        "accounting_helper_sha256": _sha256(accounting_helper),
        "cache_bypass_contract_sha256": _sha256(contract),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated normal/fixed/adaptive DSpark waves under the "
            "validated expert-file cache-bypass policy"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-bypass-contract", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workload", default="random_hex")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens < 3:
        parser.error("--max-tokens must be at least 3")

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    contract_path = Path(arguments.cache_bypass_contract).expanduser().resolve()
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    contract = json.loads(contract_path.read_text())
    if not contract["summary"]["contract_passed"]:
        parser.error("cache-bypass contract did not pass")
    prompt = _load_prompts(prompt_manifest, (arguments.workload,))[0]
    manifest = json.loads((model / "manifest.json").read_text())
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    performance_runs = []
    total_performance_runs = sum(len(wave) for wave in PERFORMANCE_WAVES)
    completed_runs = 0
    for wave_index, wave in enumerate(PERFORMANCE_WAVES, start=1):
        for sequence, mode in enumerate(wave, start=1):
            completed_runs += 1
            print(
                f"[performance {completed_runs}/{total_performance_runs}] "
                f"wave={wave_index} mode={mode}",
                flush=True,
            )
            row = _run(
                project_root=project_root,
                python=python,
                model=model,
                prompt=prompt,
                mode=mode,
                run_id=(
                    f"{prompt['name']}-performance-w{wave_index:02d}-"
                    f"{sequence:02d}-{mode}"
                ),
                raw_directory=raw_directory,
                max_tokens=arguments.max_tokens,
                resume=arguments.resume,
                dspark_fallback_enabled=False,
                layer_major_prefill_enabled=False,
                sequential_verification_enabled=False,
                hybrid_verification_enabled=True,
                hybrid_hash_prefetch_enabled=True,
                expert_page_cache_probe_enabled=False,
                expert_file_cache_policy="bypass",
            )
            row["wave"] = wave_index
            row["sequence"] = sequence
            row["track"] = "performance"
            performance_runs.append(row)

    observer_runs = []
    for sequence, mode in enumerate(OBSERVER_PATTERN, start=1):
        print(
            f"[observer {sequence}/{len(OBSERVER_PATTERN)}] mode={mode}",
            flush=True,
        )
        row = _run(
            project_root=project_root,
            python=python,
            model=model,
            prompt=prompt,
            mode=mode,
            run_id=f"{prompt['name']}-observer-{sequence:02d}-{mode}",
            raw_directory=raw_directory,
            max_tokens=arguments.max_tokens,
            resume=arguments.resume,
            dspark_fallback_enabled=False,
            layer_major_prefill_enabled=False,
            sequential_verification_enabled=False,
            hybrid_verification_enabled=True,
            hybrid_hash_prefetch_enabled=True,
            expert_page_cache_probe_enabled=True,
            expert_file_cache_policy="bypass",
        )
        row["sequence"] = sequence
        row["track"] = "observer"
        observer_runs.append(row)

    all_runs = [*performance_runs, *observer_runs]
    token_sequences = [row["metrics"]["generated_token_ids"] for row in all_runs]
    all_exact = all(tokens == token_sequences[0] for tokens in token_sequences[1:])
    cache_policy_exact = all(
        row["metrics"]["expert_file_cache_policy"] == "bypass"
        and int(row["metrics"]["expert_file_direct_io_alignment_bytes"]) > 0
        for row in all_runs
    )
    mode_summaries = {
        mode: _mode_summary(
            [row for row in performance_runs if row["mode"] == mode]
        )
        for mode in OBSERVER_PATTERN
    }
    fixed_gate = _candidate_gate(
        mode_summaries["normal"],
        mode_summaries["fixed"],
    )
    adaptive_gate = _candidate_gate(
        mode_summaries["normal"],
        mode_summaries["adaptive"],
        fixed=mode_summaries["fixed"],
    )
    observer_accounting = [
        _observer_accounting(row) for row in observer_runs
    ]
    observer_exact = all(
        row["accounting_exact"] for row in observer_accounting
    )
    complete_repetition = all(
        mode_summaries[mode]["runs"] == len(PERFORMANCE_WAVES)
        for mode in OBSERVER_PATTERN
    )
    correctness_gate = bool(
        all_exact and cache_policy_exact and complete_repetition
    )
    if not correctness_gate or not observer_exact:
        status = "stop_cache_bypass_full_model_gate"
    elif fixed_gate["candidate_gate_passed"] or adaptive_gate[
        "candidate_gate_passed"
    ]:
        status = "candidate_requires_broader_adoption_gate"
    else:
        status = "performance_candidates_rejected_under_cache_bypass"

    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "cache_bypass_dspark_repeated_comparison",
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
            "expert_blob_bytes": int(manifest["expertBlobSize"]),
        },
        "experiment": {
            "workload": arguments.workload,
            "prompt_file": prompt["file"],
            "prompt_tokens": prompt["target_tokens"],
            "max_output_tokens": arguments.max_tokens,
            "performance_waves": [list(wave) for wave in PERFORMANCE_WAVES],
            "observer_pattern": list(OBSERVER_PATTERN),
            "fresh_process_per_run": True,
            "persistent_prompt_cache": False,
            "layer_major_prefill": False,
            "expert_file_cache_policy": "bypass",
            "performance_probe_enabled": False,
            "observer_probe_enabled": True,
            "dspark_target_verification": "hybrid v3 token-shaped target math",
            "hash_prefetch": True,
            "adaptive_block": "adaptive mode only",
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
            "all_run_token_ids_exact": all_exact,
            "output_token_sha256": sorted(
                {row["metrics"]["token_sha256"] for row in all_runs}
            ),
            "cache_policy_and_alignment_exact": cache_policy_exact,
            "complete_three_run_repetition": complete_repetition,
            "mode_medians": mode_summaries,
            "fixed_gate": fixed_gate,
            "adaptive_gate": adaptive_gate,
            "observer_accounting": observer_accounting,
            "observer_accounting_exact": observer_exact,
        },
        "decision": {
            "status": status,
            "correctness_gate_passed": correctness_gate,
            "observer_accounting_gate_passed": observer_exact,
            "fixed_candidate_gate_passed": fixed_gate[
                "candidate_gate_passed"
            ],
            "adaptive_candidate_gate_passed": adaptive_gate[
                "candidate_gate_passed"
            ],
            "adopt_as_default": False,
        },
        "evidence_limits": [
            "F_NOCACHE prevents measured nonresident expert ranges from entering the VM page cache but does not purge pre-existing resident pages.",
            "The observer probe changes timing; observer timings are excluded from performance medians.",
            "Process disk counters are process-wide and remain separate from expert logical bytes.",
            "Fallback is disabled to observe the complete research candidate rather than serving defaults.",
            "This single-workload result cannot establish broad runtime adoption.",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    if not correctness_gate or not observer_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
