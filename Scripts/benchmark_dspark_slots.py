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


RUN_PATTERN = ("control", "candidate", "candidate", "control")
LONG_DECODE_GATE = {
    "minimum_output_tokens": 128,
    "maximum_peak_memory_change_fraction": -0.01,
    "maximum_draft_expert_bytes_per_committed_token_change_fraction": 0.50,
    "maximum_speculative_expert_bytes_per_committed_token_change_fraction": 0.02,
    "maximum_request_seconds_change_fraction": 0.05,
}
SLOT_METRICS = (
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
    "dspark_draft_expert_bytes_read",
    "dspark_speculative_expert_bytes_read",
    "dspark_draft_expert_bytes_per_committed_token",
    "dspark_speculative_expert_bytes_per_committed_token",
    "dspark_expert_cache_hit_rate",
    "dspark_expert_cache_hits",
    "dspark_expert_cache_misses",
    "dspark_expert_evictions",
    "dspark_expert_bytes_read",
    "dspark_expert_read_seconds",
    "dspark_expert_resident_slots",
    "dspark_expert_capacity_slots",
)


def _median(rows: list[dict], key: str) -> float:
    return statistics.median(float(row["metrics"][key]) for row in rows)


def _fraction(control: float, candidate: float) -> float | None:
    return (candidate - control) / control if control else None


def _mode_summary(rows: list[dict]) -> dict:
    return {key: _median(rows, key) for key in SLOT_METRICS}


def _evaluate_gate(
    *,
    profile: str,
    all_exact: bool,
    max_tokens: int,
    change: dict[str, float | None],
) -> dict:
    checks = {
        "all_run_token_ids_exact": all_exact,
        "peak_memory_lower": (
            change["peak_memory_bytes"] is not None
            and change["peak_memory_bytes"] < 0
        ),
    }
    thresholds = None
    if profile == "long_decode":
        thresholds = dict(LONG_DECODE_GATE)
        checks.update(
            {
                "minimum_output_tokens": (
                    max_tokens >= LONG_DECODE_GATE["minimum_output_tokens"]
                ),
                "peak_memory_reduction_at_least_one_percent": (
                    change["peak_memory_bytes"] is not None
                    and change["peak_memory_bytes"]
                    <= LONG_DECODE_GATE[
                        "maximum_peak_memory_change_fraction"
                    ]
                ),
                "draft_read_increase_at_most_fifty_percent": (
                    change[
                        "dspark_draft_expert_bytes_per_committed_token"
                    ]
                    is not None
                    and change[
                        "dspark_draft_expert_bytes_per_committed_token"
                    ]
                    <= LONG_DECODE_GATE[
                        "maximum_draft_expert_bytes_per_committed_token_change_fraction"
                    ]
                ),
                "speculative_read_increase_at_most_two_percent": (
                    change[
                        "dspark_speculative_expert_bytes_per_committed_token"
                    ]
                    is not None
                    and change[
                        "dspark_speculative_expert_bytes_per_committed_token"
                    ]
                    <= LONG_DECODE_GATE[
                        "maximum_speculative_expert_bytes_per_committed_token_change_fraction"
                    ]
                ),
                "request_time_increase_at_most_five_percent": (
                    change["request_seconds"] is not None
                    and change["request_seconds"]
                    <= LONG_DECODE_GATE[
                        "maximum_request_seconds_change_fraction"
                    ]
                ),
            }
        )
    return {
        "profile": profile,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _source_state(project_root: Path) -> dict:
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
    }


def _run_configuration(
    *,
    project_root: Path,
    python: Path,
    model: Path,
    prompt: dict,
    label: str,
    slots: int,
    run_id: str,
    raw_directory: Path,
    max_tokens: int,
    resume: bool,
    fallback_enabled: bool,
) -> dict:
    row = _run(
        project_root=project_root,
        python=python,
        model=model,
        prompt=prompt,
        mode="fixed",
        run_id=run_id,
        raw_directory=raw_directory,
        max_tokens=max_tokens,
        resume=resume,
        dspark_fallback_enabled=fallback_enabled,
        layer_major_prefill_enabled=False,
        sequential_verification_enabled=False,
        hybrid_verification_enabled=True,
        hybrid_hash_prefetch_enabled=False,
        dspark_slots=slots,
    )
    row["mode"] = label
    row["dspark_slots"] = slots
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the complete DSpark graph with its default and a reduced "
            "independent expert-cache capacity"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workload", default="storage_sentence")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--control-slots", type=int, default=768)
    parser.add_argument("--candidate-slots", type=int, default=96)
    parser.add_argument("--fallback-enabled", action="store_true")
    parser.add_argument("--skip-normal-reference", action="store_true")
    parser.add_argument(
        "--single-pair",
        action="store_true",
        help="run one control/candidate pair instead of the default ABBA wave",
    )
    parser.add_argument(
        "--gate-profile",
        choices=("exploratory", "long_decode"),
        default="exploratory",
        help="record and enforce the predeclared evaluation gate",
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens < 3:
        parser.error("--max-tokens must be at least 3")
    if arguments.control_slots <= arguments.candidate_slots:
        parser.error("--control-slots must be greater than --candidate-slots")
    if arguments.gate_profile == "long_decode":
        if arguments.max_tokens < LONG_DECODE_GATE["minimum_output_tokens"]:
            parser.error(
                "--gate-profile long_decode requires --max-tokens >= "
                f"{LONG_DECODE_GATE['minimum_output_tokens']}"
            )
        if arguments.skip_normal_reference:
            parser.error(
                "--gate-profile long_decode requires a fresh normal reference"
            )
        if arguments.fallback_enabled:
            parser.error(
                "--gate-profile long_decode requires fallback to remain disabled"
            )

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    prompt = _load_prompts(prompt_manifest, (arguments.workload,))[0]
    manifest = json.loads((model / "manifest.json").read_text())
    dspark = manifest.get("dspark")
    if not isinstance(dspark, dict):
        parser.error("installed model has no DSpark descriptor")
    one_block_working_set = (
        int(dspark["layerCount"])
        * int(dspark["blockSize"])
        * int(manifest["selectedExpertCount"])
    )
    if arguments.candidate_slots < one_block_working_set:
        parser.error(
            "--candidate-slots must hold the worst-case assignments for one "
            f"complete draft block ({one_block_working_set})"
        )

    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_pattern = ("control", "candidate") if arguments.single_pair else RUN_PATTERN
    runs = []
    if not arguments.skip_normal_reference:
        print("[reference] normal target decode", flush=True)
        reference = _run(
            project_root=project_root,
            python=python,
            model=model,
            prompt=prompt,
            mode="normal",
            run_id=f"{prompt['name']}-normal-reference",
            raw_directory=raw_directory,
            max_tokens=arguments.max_tokens,
            resume=arguments.resume,
            dspark_fallback_enabled=arguments.fallback_enabled,
            layer_major_prefill_enabled=False,
            sequential_verification_enabled=False,
            hybrid_verification_enabled=True,
            hybrid_hash_prefetch_enabled=False,
        )
        reference["sequence"] = 0
        runs.append(reference)

    slots_by_label = {
        "control": arguments.control_slots,
        "candidate": arguments.candidate_slots,
    }
    for sequence, label in enumerate(run_pattern, start=1):
        slots = slots_by_label[label]
        print(f"[{sequence}/{len(run_pattern)}] {label} {slots} slots", flush=True)
        row = _run_configuration(
            project_root=project_root,
            python=python,
            model=model,
            prompt=prompt,
            label=label,
            slots=slots,
            run_id=f"{prompt['name']}-{sequence:02d}-{label}-{slots}",
            raw_directory=raw_directory,
            max_tokens=arguments.max_tokens,
            resume=arguments.resume,
            fallback_enabled=arguments.fallback_enabled,
        )
        row["sequence"] = sequence
        runs.append(row)

    control = [row for row in runs if row["mode"] == "control"]
    candidate = [row for row in runs if row["mode"] == "candidate"]
    control_summary = _mode_summary(control)
    candidate_summary = _mode_summary(candidate)
    token_sequences = [row["metrics"]["generated_token_ids"] for row in runs]
    all_exact = all(tokens == token_sequences[0] for tokens in token_sequences[1:])
    output_hashes = sorted({row["metrics"]["token_sha256"] for row in runs})
    prompt_hashes = sorted(
        {row["metrics"]["prompt_token_sha256"] for row in runs}
    )
    change = {
        key: _fraction(control_summary[key], candidate_summary[key])
        for key in SLOT_METRICS
    }
    pair_count = min(len(control), len(candidate))
    paired_change = {
        key: [
            _fraction(
                float(control[index]["metrics"][key]),
                float(candidate[index]["metrics"][key]),
            )
            for index in range(pair_count)
        ]
        for key in SLOT_METRICS
    }
    expert_blob_bytes = int(manifest["expertBlobSize"])
    control_capacity = arguments.control_slots * expert_blob_bytes
    candidate_capacity = arguments.candidate_slots * expert_blob_bytes
    gate = _evaluate_gate(
        profile=arguments.gate_profile,
        all_exact=all_exact,
        max_tokens=arguments.max_tokens,
        change=change,
    )
    peak_memory_lower = gate["checks"]["peak_memory_lower"]
    if gate["passed"]:
        decision = (
            "long_decode_gate_passed"
            if arguments.gate_profile == "long_decode"
            else "continue_reduced_cache_validation"
        )
    else:
        decision = "stop_reduced_cache_candidate"
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "dspark_reduced_expert_cache_pilot",
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
            "model_id": manifest["modelID"],
            "revision": manifest["revision"],
            "expert_blob_bytes": expert_blob_bytes,
            "dspark_layer_count": int(dspark["layerCount"]),
            "dspark_block_size": int(dspark["blockSize"]),
            "selected_experts_per_token": int(manifest["selectedExpertCount"]),
        },
        "experiment": {
            "hypothesis": (
                f"A {arguments.candidate_slots}-slot independent DSpark cache can "
                "retain one worst-case complete draft working set while reducing "
                "resident memory without changing greedy target output."
            ),
            "workload": arguments.workload,
            "prompt_file": prompt["file"],
            "prompt_tokens": prompt["target_tokens"],
            "max_output_tokens": arguments.max_tokens,
            "run_pattern": list(run_pattern),
            "normal_reference": not arguments.skip_normal_reference,
            "fresh_process_per_run": True,
            "persistent_prompt_cache": False,
            "layer_major_prefill": False,
            "target_verification": "hybrid v3 token-shaped target math",
            "hash_prefetch": False,
            "adaptive_block": False,
            "fallback_enabled": arguments.fallback_enabled,
            "temperature": 0,
            "top_p": 1,
            "control_slots": arguments.control_slots,
            "candidate_slots": arguments.candidate_slots,
            "worst_case_one_block_assignments": one_block_working_set,
            "control_payload_capacity_bytes": control_capacity,
            "candidate_payload_capacity_bytes": candidate_capacity,
            "payload_capacity_reduction_bytes": control_capacity
            - candidate_capacity,
            "payload_capacity_reduction_fraction": 1
            - candidate_capacity / control_capacity,
            "os_page_cache": "not purged or controlled",
            "gate_profile": arguments.gate_profile,
        },
        "prompt_manifest_sha256": _sha256(prompt_manifest),
        "runs": runs,
        "summary": {
            "prompt_token_sha256": prompt_hashes,
            "output_token_sha256": output_hashes,
            "all_run_token_ids_exact": all_exact,
            "control_medians": control_summary,
            "candidate_medians": candidate_summary,
            "candidate_change_fraction": change,
            "paired_candidate_change_fraction": paired_change,
            "candidate_peak_memory_lower": peak_memory_lower,
            "predeclared_gate": gate,
        },
        "decision": {
            "status": decision,
            "correctness_gate_passed": all_exact,
            "memory_direction_gate_passed": peak_memory_lower,
            "predeclared_gate_passed": gate["passed"],
            "adopt_as_default": False,
            "next_gate": (
                "Repeat with controlled cache state before any default change."
                if decision == "long_decode_gate_passed"
                else (
                    "Repeat across multiple workloads and longer decode with controlled "
                    "cache state; require stable memory reduction, exact tokens, and an "
                    "acceptable expert-read/request-time tradeoff."
                    if decision == "continue_reduced_cache_validation"
                    else "Do not expand this slot candidate."
                )
            ),
        },
        "evidence_limits": [
            (
                "This is one unreplicated control/candidate pair on one workload, not a formal performance result."
                if arguments.single_pair
                else "This is one exploratory ABBA pilot on one workload, not a formal performance result."
            ),
            "The operating-system page cache is neither purged nor controlled.",
            "Process disk counters include every read attributed to the process and are not expert-file physical I/O.",
            "Fallback is disabled unless explicitly requested so the pilot can observe the complete research trace.",
            "Payload capacity is a hard upper bound; peak MLX memory reflects only slots actually materialized during each run.",
            "No timing, memory, or I/O difference from this pilot authorizes a default change.",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
