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


RUN_PATTERN = ("normal", "hash", "hash", "normal")


def _partition(
    *,
    logical: int,
    classified: int,
    resident: int,
    nonresident: int,
    unclassified: int,
) -> dict:
    return {
        "logical_bytes": logical,
        "classified_bytes": classified,
        "resident_bytes_before_read": resident,
        "nonresident_bytes_before_read": nonresident,
        "unclassified_bytes": unclassified,
        "classified_partition_exact": resident + nonresident == classified,
        "logical_coverage_exact": classified + unclassified == logical,
    }


def _run_accounting(row: dict) -> dict:
    metrics = row["metrics"]
    main = _partition(
        logical=int(metrics["request_expert_bytes_read"]),
        classified=int(metrics["request_expert_page_cache_classified_bytes"]),
        resident=int(
            metrics["request_expert_page_cache_resident_bytes_before_read"]
        ),
        nonresident=int(
            metrics["request_expert_page_cache_nonresident_bytes_before_read"]
        ),
        unclassified=int(
            metrics["request_expert_page_cache_unclassified_bytes"]
        ),
    )
    result = {
        "id": row["id"],
        "mode": row["mode"],
        "main_target": main,
        "main_probe_calls": int(
            metrics["request_expert_page_cache_probe_calls"]
        ),
        "main_probe_failures": int(
            metrics["request_expert_page_cache_probe_failures"]
        ),
        "process_disk_bytes_read": metrics["request_process_disk_bytes_read"],
    }
    if row["mode"] != "hash":
        return result

    draft = _partition(
        logical=int(metrics["dspark_draft_expert_bytes_read"]),
        classified=int(metrics["dspark_draft_page_cache_classified_bytes"]),
        resident=int(
            metrics["dspark_draft_page_cache_resident_bytes_before_read"]
        ),
        nonresident=int(
            metrics["dspark_draft_page_cache_nonresident_bytes_before_read"]
        ),
        unclassified=int(metrics["dspark_draft_page_cache_unclassified_bytes"]),
    )
    hash_prefetch = _partition(
        logical=int(metrics["dspark_hash_prefetch_bytes_read"]),
        classified=int(
            metrics["dspark_hash_prefetch_page_cache_classified_bytes"]
        ),
        resident=int(
            metrics[
                "dspark_hash_prefetch_page_cache_resident_bytes_before_read"
            ]
        ),
        nonresident=int(
            metrics[
                "dspark_hash_prefetch_page_cache_nonresident_bytes_before_read"
            ]
        ),
        unclassified=int(
            metrics["dspark_hash_prefetch_page_cache_unclassified_bytes"]
        ),
    )
    useful = _partition(
        logical=int(metrics["dspark_hash_prefetch_useful_bytes"]),
        classified=(
            int(
                metrics[
                    "dspark_hash_prefetch_useful_page_cache_resident_bytes_before_read"
                ]
            )
            + int(
                metrics[
                    "dspark_hash_prefetch_useful_page_cache_nonresident_bytes_before_read"
                ]
            )
        ),
        resident=int(
            metrics[
                "dspark_hash_prefetch_useful_page_cache_resident_bytes_before_read"
            ]
        ),
        nonresident=int(
            metrics[
                "dspark_hash_prefetch_useful_page_cache_nonresident_bytes_before_read"
            ]
        ),
        unclassified=int(
            metrics[
                "dspark_hash_prefetch_useful_page_cache_unclassified_bytes"
            ]
        ),
    )
    wasted = _partition(
        logical=int(metrics["dspark_hash_prefetch_wasted_bytes"]),
        classified=(
            int(
                metrics[
                    "dspark_hash_prefetch_wasted_page_cache_resident_bytes_before_read"
                ]
            )
            + int(
                metrics[
                    "dspark_hash_prefetch_wasted_page_cache_nonresident_bytes_before_read"
                ]
            )
        ),
        resident=int(
            metrics[
                "dspark_hash_prefetch_wasted_page_cache_resident_bytes_before_read"
            ]
        ),
        nonresident=int(
            metrics[
                "dspark_hash_prefetch_wasted_page_cache_nonresident_bytes_before_read"
            ]
        ),
        unclassified=int(
            metrics[
                "dspark_hash_prefetch_wasted_page_cache_unclassified_bytes"
            ]
        ),
    )
    result.update(
        {
            "draft": draft,
            "draft_probe_calls": int(
                metrics["dspark_draft_page_cache_probe_calls"]
            ),
            "draft_probe_failures": int(
                metrics["dspark_draft_page_cache_probe_failures"]
            ),
            "hash_prefetch": hash_prefetch,
            "useful": useful,
            "wasted": wasted,
            "logical_useful_rate": metrics["dspark_hash_prefetch_useful_rate"],
            "nonresident_useful_rate": (
                useful["nonresident_bytes_before_read"]
                / hash_prefetch["nonresident_bytes_before_read"]
                if hash_prefetch["nonresident_bytes_before_read"]
                else None
            ),
            "hash_useful_wasted_partition_exact": (
                useful["logical_bytes"] + wasted["logical_bytes"]
                == hash_prefetch["logical_bytes"]
                and useful["resident_bytes_before_read"]
                + wasted["resident_bytes_before_read"]
                == hash_prefetch["resident_bytes_before_read"]
                and useful["nonresident_bytes_before_read"]
                + wasted["nonresident_bytes_before_read"]
                == hash_prefetch["nonresident_bytes_before_read"]
                and useful["unclassified_bytes"] + wasted["unclassified_bytes"]
                == hash_prefetch["unclassified_bytes"]
            ),
        }
    )
    return result


def _accounting_exact(row: dict) -> bool:
    partitions = [row["main_target"]]
    if row["mode"] == "hash":
        partitions.extend(
            [row["draft"], row["hash_prefetch"], row["useful"], row["wasted"]]
        )
        if row["draft_probe_failures"] or not row[
            "hash_useful_wasted_partition_exact"
        ]:
            return False
    return (
        row["main_probe_failures"] == 0
        and all(
            partition["classified_partition_exact"]
            and partition["logical_coverage_exact"]
            and partition["unclassified_bytes"] == 0
            for partition in partitions
        )
    )


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
            "Compose the expert-file page-cache proxy with exact DSpark hash "
            "prefetch and classify useful/wasted pre-read residency"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workload", default="code")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--single-pair", action="store_true")
    parser.add_argument("--require-useful-and-wasted", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens < 3:
        parser.error("--max-tokens must be at least 3")

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

    run_pattern = ("normal", "hash") if arguments.single_pair else RUN_PATTERN
    runs = []
    for sequence, label in enumerate(run_pattern, start=1):
        print(f"[{sequence}/{len(run_pattern)}] {label}", flush=True)
        row = _run(
            project_root=project_root,
            python=python,
            model=model,
            prompt=prompt,
            mode="normal" if label == "normal" else "fixed",
            run_id=f"{prompt['name']}-{sequence:02d}-{label}",
            raw_directory=raw_directory,
            max_tokens=arguments.max_tokens,
            resume=arguments.resume,
            dspark_fallback_enabled=False,
            layer_major_prefill_enabled=False,
            sequential_verification_enabled=False,
            hybrid_verification_enabled=True,
            hybrid_hash_prefetch_enabled=True,
            expert_page_cache_probe_enabled=True,
        )
        row["mode"] = label
        row["sequence"] = sequence
        runs.append(row)

    token_sequences = [row["metrics"]["generated_token_ids"] for row in runs]
    all_exact = all(tokens == token_sequences[0] for tokens in token_sequences[1:])
    accounting = [_run_accounting(row) for row in runs]
    accounting_exact = all(_accounting_exact(row) for row in accounting)
    hash_rows = [row for row in accounting if row["mode"] == "hash"]
    useful_and_wasted_observed = all(
        row["useful"]["logical_bytes"] > 0
        and row["wasted"]["logical_bytes"] > 0
        for row in hash_rows
    )
    logical_useful_rates = [float(row["logical_useful_rate"]) for row in hash_rows]
    nonresident_useful_rates = [
        float(row["nonresident_useful_rate"])
        for row in hash_rows
        if row["nonresident_useful_rate"] is not None
    ]
    decision = (
        "hash_prefetch_proxy_composition_passed"
        if all_exact
        and accounting_exact
        and (
            useful_and_wasted_observed
            or not arguments.require_useful_and_wasted
        )
        else "stop_page_cache_proxy_composition"
    )
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "hash_prefetch_page_cache_residency_composition",
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
                "The exact hash-prefetch logical useful/wasted partition remains "
                "closed when each read is also classified by pre-read page residency."
            ),
            "workload": arguments.workload,
            "prompt_file": prompt["file"],
            "prompt_tokens": prompt["target_tokens"],
            "max_output_tokens": arguments.max_tokens,
            "run_pattern": list(run_pattern),
            "fresh_process_per_run": True,
            "persistent_prompt_cache": False,
            "layer_major_prefill": False,
            "dspark_target_verification": "hybrid v3 token-shaped target math",
            "hash_prefetch": True,
            "adaptive_block": False,
            "fallback_enabled": False,
            "page_cache_probe": True,
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
            "all_accounting_exact": accounting_exact,
            "useful_and_wasted_observed": useful_and_wasted_observed,
            "useful_and_wasted_required": arguments.require_useful_and_wasted,
            "accounting": accounting,
            "logical_hash_prefetch_useful_rate": {
                "minimum": min(logical_useful_rates),
                "median": statistics.median(logical_useful_rates),
                "maximum": max(logical_useful_rates),
            },
            "nonresident_hash_prefetch_useful_rate": {
                "minimum": min(nonresident_useful_rates),
                "median": statistics.median(nonresident_useful_rates),
                "maximum": max(nonresident_useful_rates),
            },
        },
        "decision": {
            "status": decision,
            "correctness_gate_passed": all_exact,
            "accounting_gate_passed": accounting_exact,
            "useful_and_wasted_coverage_gate_passed": (
                useful_and_wasted_observed
                or not arguments.require_useful_and_wasted
            ),
            "adopt_as_default": False,
            "next_gate": (
                "Run repeated, explicitly labeled cache-state waves before using "
                "the proxy to judge a performance candidate."
                if decision == "hash_prefetch_proxy_composition_passed"
                else "Do not use the proxy for useful/wasted conclusions."
            ),
        },
        "evidence_limits": [
            "mincore nonresident bytes are a page-cache-miss proxy, not physical SSD bytes.",
            "The operating-system page cache is neither purged nor controlled.",
            "Fallback is disabled to observe the complete research trace.",
            "The probe adds observer overhead; timing is not adoption evidence.",
            "Process disk counters remain process-wide and cannot be partitioned as useful or wasted expert traffic.",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
