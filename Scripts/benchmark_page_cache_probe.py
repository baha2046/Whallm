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
    from .benchmark_common import (
        _command_output,
        _load_prompts,
        _package_version,
        _run,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )
except ImportError:
    from benchmark_common import (
        _command_output,
        _load_prompts,
        _package_version,
        _run,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )


RUN_PATTERN = ("control", "probe", "probe", "control")
SUMMARY_METRICS = (
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
)


def _fraction(control: float, candidate: float) -> float | None:
    return (candidate - control) / control if control else None


def _median(rows: list[dict], key: str) -> float | None:
    values = [row["metrics"][key] for row in rows]
    if any(value is None for value in values):
        return None
    return statistics.median(float(value) for value in values)


def _mode_summary(rows: list[dict]) -> dict[str, float | None]:
    return {key: _median(rows, key) for key in SUMMARY_METRICS}


def _probe_accounting(row: dict) -> dict:
    metrics = row["metrics"]
    logical = int(metrics["request_expert_bytes_read"])
    classified = int(metrics["request_expert_page_cache_classified_bytes"])
    resident = int(
        metrics["request_expert_page_cache_resident_bytes_before_read"]
    )
    nonresident = int(
        metrics["request_expert_page_cache_nonresident_bytes_before_read"]
    )
    unclassified = int(
        metrics["request_expert_page_cache_unclassified_bytes"]
    )
    return {
        "id": row["id"],
        "logical_expert_bytes": logical,
        "classified_bytes": classified,
        "resident_bytes_before_read": resident,
        "nonresident_bytes_before_read": nonresident,
        "unclassified_bytes": unclassified,
        "probe_calls": int(metrics["request_expert_page_cache_probe_calls"]),
        "probe_failures": int(
            metrics["request_expert_page_cache_probe_failures"]
        ),
        "classified_partition_exact": resident + nonresident == classified,
        "logical_coverage_exact": classified + unclassified == logical,
        "process_disk_bytes_read": metrics["request_process_disk_bytes_read"],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the research-only expert-file page-cache residency probe "
            "against logical byte accounting"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workload", default="code")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens < 1:
        parser.error("--max-tokens must be positive")

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
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for sequence, label in enumerate(RUN_PATTERN, start=1):
        print(f"[{sequence}/{len(RUN_PATTERN)}] {label}", flush=True)
        row = _run(
            project_root=project_root,
            python=python,
            model=model,
            prompt=prompt,
            mode="normal",
            run_id=f"{prompt['name']}-{sequence:02d}-{label}",
            raw_directory=raw_directory,
            max_tokens=arguments.max_tokens,
            resume=arguments.resume,
            dspark_fallback_enabled=True,
            layer_major_prefill_enabled=False,
            sequential_verification_enabled=False,
            expert_page_cache_probe_enabled=label == "probe",
        )
        row["mode"] = label
        row["sequence"] = sequence
        runs.append(row)

    controls = [row for row in runs if row["mode"] == "control"]
    probes = [row for row in runs if row["mode"] == "probe"]
    control_summary = _mode_summary(controls)
    probe_summary = _mode_summary(probes)
    token_sequences = [row["metrics"]["generated_token_ids"] for row in runs]
    all_exact = all(tokens == token_sequences[0] for tokens in token_sequences[1:])
    accounting = [_probe_accounting(row) for row in probes]
    accounting_exact = all(
        row["classified_partition_exact"]
        and row["logical_coverage_exact"]
        and row["probe_failures"] == 0
        and row["unclassified_bytes"] == 0
        for row in accounting
    )
    probe_observer_change = {
        key: (
            _fraction(control_summary[key], probe_summary[key])
            if control_summary[key] is not None and probe_summary[key] is not None
            else None
        )
        for key in SUMMARY_METRICS
    }
    decision = (
        "probe_contract_passed"
        if all_exact and accounting_exact
        else "stop_page_cache_proxy"
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "expert_file_page_cache_residency_probe_validation",
        "formal_performance_result": False,
        "source": _source_state(project_root),
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
            "hypothesis": (
                "A pre-read mincore probe can classify every logical expert-file "
                "byte as page-resident or page-nonresident without changing greedy "
                "target output."
            ),
            "workload": arguments.workload,
            "prompt_file": prompt["file"],
            "prompt_tokens": prompt["target_tokens"],
            "max_output_tokens": arguments.max_tokens,
            "run_pattern": list(RUN_PATTERN),
            "fresh_process_per_run": True,
            "persistent_prompt_cache": False,
            "layer_major_prefill": False,
            "dspark": False,
            "temperature": 0,
            "top_p": 1,
            "os_page_cache": "not purged or controlled",
        },
        "prompt_manifest_sha256": _sha256(prompt_manifest),
        "runs": runs,
        "summary": {
            "all_run_token_ids_exact": all_exact,
            "output_token_sha256": sorted(
                {row["metrics"]["token_sha256"] for row in runs}
            ),
            "control_medians": control_summary,
            "probe_medians": probe_summary,
            "probe_observer_change_fraction": probe_observer_change,
            "probe_accounting": accounting,
            "all_probe_accounting_exact": accounting_exact,
        },
        "decision": {
            "status": decision,
            "correctness_gate_passed": all_exact,
            "accounting_gate_passed": accounting_exact,
            "adopt_as_default": False,
            "next_gate": (
                "Use the proxy in a DSpark hash-prefetch useful/wasted-byte trace; "
                "retain process disk counters and controlled cache-state labels."
                if decision == "probe_contract_passed"
                else "Do not use this proxy until classification is exact."
            ),
        },
        "evidence_limits": [
            "mincore reports virtual-memory page residency before each read, not physical SSD bytes.",
            "APFS, storage-controller caches, read coalescing, and concurrent I/O remain outside this probe.",
            "The operating-system page cache is neither purged nor controlled.",
            "The control/probe timing difference measures observer overhead only and is not runtime adoption evidence.",
            "Process disk counters remain process-wide and cannot be attributed to one expert file.",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
