from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx

from deepseek_v4_ssd.io_metrics import page_cache_residency_snapshot


METHODS = ("preadv", "mtlio_bytes", "mtlio_shared", "mtlio_private")
DEFAULT_COUNTS = (1, 6, 32, 128)
OFFICIAL_REFERENCES = (
    "https://developer.apple.com/documentation/metal/mtliocommandbuffer",
    "https://developer.apple.com/documentation/metal/mtliocommandqueue",
    "https://developer.apple.com/documentation/metal/resource-loading",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
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


def _residency(path: Path, length: int) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        snapshot = page_cache_residency_snapshot(descriptor, 0, length)
    finally:
        os.close(descriptor)
    if snapshot is None:
        return {
            "available": False,
            "requested_bytes": length,
            "resident_bytes": None,
            "nonresident_bytes": None,
        }
    classification = snapshot.classify(length)
    return {
        "available": True,
        "requested_bytes": length,
        "resident_bytes": classification.resident_bytes,
        "nonresident_bytes": classification.nonresident_bytes,
        "resident_fraction": classification.resident_bytes / length,
        "page_size_bytes": snapshot.page_size,
    }


def _native_gate(
    rows: list[dict[str, Any]],
    cancellation: dict[str, Any],
    expected_counts: tuple[int, ...] = DEFAULT_COUNTS,
) -> dict[str, Any]:
    expected_pairs = {(method, count) for method in METHODS for count in expected_counts}
    observed_pairs = {(row.get("method"), row.get("count")) for row in rows}
    complete_exact = all(
        row.get("status") == "complete"
        and row.get("error") is None
        and row.get("exact") is True
        and row.get("candidate_sha256") == row.get("reference_sha256")
        for row in rows
    )
    buffer_rows = [
        row
        for row in rows
        if row.get("method") in {"mtlio_shared", "mtlio_private"}
    ]
    shared_rows = [row for row in rows if row.get("method") == "mtlio_shared"]
    private_rows = [row for row in rows if row.get("method") == "mtlio_private"]
    shared_event_visibility = all(
        row.get("shared_event_ordering") is True
        and int(row.get("event_signaled_value", 0)) >= 1
        and row.get("gpu_visible_seconds") is not None
        and row.get("gpu_copy_seconds") is not None
        for row in buffer_rows
    )
    shared_direct_exact = all(
        row.get("storage_mode") == "shared"
        and row.get("cpu_visible_destination") is True
        and row.get("direct_cpu_sha256") == row.get("reference_sha256")
        for row in shared_rows
    )
    private_copy_explicit = all(
        row.get("storage_mode") == "private"
        and row.get("cpu_visible_destination") is False
        and row.get("explicit_gpu_validation_copy") is True
        and row.get("validation_copy_required_for_cpu_access") is True
        for row in private_rows
    )
    cancellation_complete = (
        cancellation.get("status") == "complete"
        and cancellation.get("destination_admitted") is True
        and cancellation.get("exact") is True
        and cancellation.get("candidate_sha256")
        == cancellation.get("reference_sha256")
    )
    cancellation_rejected = (
        cancellation.get("status") == "cancelled"
        and cancellation.get("destination_admitted") is False
        and cancellation.get("candidate_sha256") is None
    )
    criteria = {
        "method_count_matrix_complete": observed_pairs == expected_pairs,
        "all_completed_ranges_byte_exact": complete_exact,
        "shared_and_private_event_gpu_visibility_exact": (
            bool(buffer_rows) and shared_event_visibility
        ),
        "shared_destination_direct_cpu_view_exact": (
            bool(shared_rows) and shared_direct_exact
        ),
        "private_cpu_validation_copy_explicit": (
            bool(private_rows) and private_copy_explicit
        ),
        "cancellation_destination_admission_safe": (
            cancellation_complete or cancellation_rejected
        ),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = {(row["method"], int(row["count"])): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    for count in sorted({int(row["count"]) for row in rows}):
        baseline = by_pair.get(("preadv", count))
        if baseline is None:
            continue
        for method in METHODS[1:]:
            candidate = by_pair.get((method, count))
            if candidate is None:
                continue
            comparison: dict[str, Any] = {
                "count": count,
                "method": method,
                "baseline_method": "preadv",
                "different_installed_layer_files": (
                    candidate.get("file") != baseline.get("file")
                ),
                "io_seconds_change_fraction": (
                    float(candidate["io_seconds"])
                    / max(float(baseline["io_seconds"]), 1e-12)
                    - 1
                ),
                "aggregate_throughput_change_fraction": (
                    float(candidate["aggregate_bytes_per_second"])
                    / max(float(baseline["aggregate_bytes_per_second"]), 1e-12)
                    - 1
                ),
            }
            if count in {1, 6}:
                comparison["latency_p50_change_fraction"] = (
                    float(candidate["latency_p50_seconds"])
                    / max(float(baseline["latency_p50_seconds"]), 1e-12)
                    - 1
                )
                comparison["latency_p95_change_fraction"] = (
                    float(candidate["latency_p95_seconds"])
                    / max(float(baseline["latency_p95_seconds"]), 1e-12)
                    - 1
                )
            comparisons.append(comparison)
    return {
        "cache_state_balanced": False,
        "cache_control": "not_purged_unique_layer_files_mincore_observed",
        "claim_boundary": (
            "exploratory wall time only; methods use different installed layer "
            "files and page-cache state is observed but not controlled"
        ),
        "comparisons": comparisons,
        "eligible_for_runtime_adoption": False,
    }


class _DLPackCallProbe:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __dlpack__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": list(args), "kwargs": dict(kwargs)})
        raise RuntimeError("capture DLPack consumer call")


def _mlx_handoff_audit() -> dict[str, Any]:
    package = Path(mx.__file__).resolve().parent
    include = package / "include" / "mlx"
    array_header = include / "array.h"
    allocator_header = include / "allocator.h"
    event_header = include / "event.h"
    metal_event_header = include / "backend" / "metal" / "event.h"
    headers = (array_header, allocator_header, event_header, metal_event_header)
    for header in headers:
        if not header.is_file():
            raise RuntimeError(f"required installed MLX header is missing: {header}")

    array_text = array_header.read_text(encoding="utf-8")
    allocator_text = allocator_header.read_text(encoding="utf-8")
    event_text = event_header.read_text(encoding="utf-8")
    metal_event_text = metal_event_header.read_text(encoding="utf-8")

    dlpack_roundtrip = mx.from_dlpack(mx.array([1, 2, 3]), copy=False)
    dlpack_roundtrip_exact = dlpack_roundtrip.tolist() == [1, 2, 3]
    probe = _DLPackCallProbe()
    try:
        mx.from_dlpack(probe, copy=False)
    except (RuntimeError, TypeError):
        pass

    python_metal_attributes = sorted(
        name
        for name in dir(mx.metal)
        if any(term in name.lower() for term in ("buffer", "event", "fence"))
    )
    cpp_raw_pointer_candidate = (
        "Build an array from a raw pointer" in array_text
        and "make_buffer(void* ptr, size_t size)" in allocator_text
    )
    cpp_event_import = (
        "Event(MTL" in event_text
        or "EventImpl(MTL" in metal_event_text
        or "SharedEvent*" in event_text
    )
    dlpack_consumer_supplied_dependency = any(
        bool(call["args"]) or bool(call["kwargs"]) for call in probe.calls
    )
    criteria = {
        "public_python_direct_metal_resource_import": bool(
            python_metal_attributes
        ),
        "public_cpp_raw_pointer_no_copy_candidate": cpp_raw_pointer_candidate,
        "public_external_mtlshared_event_import": cpp_event_import,
        "dlpack_consumer_supplies_stream_or_event_dependency": (
            dlpack_consumer_supplied_dependency
        ),
        "resource_and_dependency_conjunction": False,
    }
    criteria["resource_and_dependency_conjunction"] = bool(
        (
            criteria["public_python_direct_metal_resource_import"]
            or criteria["public_cpp_raw_pointer_no_copy_candidate"]
        )
        and (
            criteria["public_external_mtlshared_event_import"]
            or criteria[
                "dlpack_consumer_supplies_stream_or_event_dependency"
            ]
        )
    )
    return {
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "python_core_extension": str(Path(mx.__file__).resolve()),
        "python_from_dlpack_present": hasattr(mx, "from_dlpack"),
        "python_mlx_metal_resource_or_event_attributes": python_metal_attributes,
        "metal_dlpack_roundtrip_exact": dlpack_roundtrip_exact,
        "observed_from_dlpack_provider_calls": probe.calls,
        "installed_header_sha256": {
            str(header.relative_to(package)): _sha256(header) for header in headers
        },
        "criteria": criteria,
        "passed": criteria["resource_and_dependency_conjunction"],
        "result": (
            "public raw-pointer/DLPack resource candidate exists, but the "
            "installed public surface exposes no external MTLSharedEvent "
            "import and the observed DLPack consumer call supplies no stream "
            "or event dependency"
        ),
        "evidence_limit": (
            "This audits the installed MLX 0.32.0 Python surface and shipped "
            "public headers; it does not prove that a future MLX release or "
            "unsupported allocator/backend cast cannot implement a bridge."
        ),
    }


def _integration_decision(
    native_gate: dict[str, Any],
    handoff_audit: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    if not native_gate["passed"]:
        return {
            "continue_to_runtime_prototype": False,
            "outcome": "stop_native_correctness_or_lifetime_gate_failed",
        }
    if not handoff_audit["passed"]:
        return {
            "continue_to_runtime_prototype": False,
            "outcome": "stop_no_supported_mtlio_to_mlx_dependency_handoff",
        }
    if not timing["cache_state_balanced"]:
        return {
            "continue_to_runtime_prototype": False,
            "outcome": "defer_uncontrolled_page_cache_timing",
        }
    return {
        "continue_to_runtime_prototype": True,
        "outcome": "continue_to_isolated_runtime_prototype",
    }


def _source_state(project_root: Path, swift_source: Path) -> dict[str, Any]:
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
        "orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "native_probe_sha256": _sha256(swift_source),
    }


def _compile_probe(
    project_root: Path,
    swift_source: Path,
    executable: Path,
    developer_directory: Path,
) -> dict[str, Any]:
    executable.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["DEVELOPER_DIR"] = str(developer_directory)
    command = [
        "xcrun",
        "swiftc",
        "-swift-version",
        "5",
        "-O",
        str(swift_source),
        "-o",
        str(executable),
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "executable_sha256": _sha256(executable),
    }


def _run_probe(
    executable: Path,
    expert_file: Path,
    blob_size: int,
    count: int,
    method: str,
    raw_path: Path,
) -> dict[str, Any]:
    total_bytes = blob_size * count
    before = _residency(expert_file, total_bytes)
    command = [
        str(executable),
        "--file",
        str(expert_file),
        "--blob-size",
        str(blob_size),
        "--count",
        str(count),
        "--start-expert",
        "0",
        "--method",
        method,
        "--read-workers",
        "4",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw_path.write_text(completed.stdout, encoding="utf-8")
    row = json.loads(completed.stdout)
    row["stderr"] = completed.stderr
    row["page_cache_before"] = before
    row["page_cache_after"] = _residency(expert_file, total_bytes)
    row["raw_output"] = {
        "path": str(raw_path),
        "sha256": _sha256(raw_path),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate native MTLIO expert loading and MLX handoff gates"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--layer-start", type=int, default=20)
    parser.add_argument(
        "--developer-directory",
        default="/Applications/Xcode-26.6.0.app/Contents/Developer",
    )
    arguments = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("the MTLIO native gate requires Darwin")

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    developer_directory = Path(arguments.developer_directory).expanduser().resolve()
    swift_source = project_root / "Scripts" / "mtlio_expert_probe.swift"
    executable = raw_directory / "mtlio_expert_probe"
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = model / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blob_size = int(manifest["expertBlobSize"])
    layer_count = int(manifest["layerCount"])
    expert_count = int(manifest["expertCount"])
    required_layers = len(METHODS) * len(DEFAULT_COUNTS) + 1
    if arguments.layer_start < 0 or arguments.layer_start + required_layers > layer_count:
        parser.error("selected layer range does not fit the installed model")
    if max(DEFAULT_COUNTS) > expert_count:
        parser.error("benchmark count exceeds installed expert count")

    compile_result = _compile_probe(
        project_root,
        swift_source,
        executable,
        developer_directory,
    )
    rows: list[dict[str, Any]] = []
    execution_order: list[dict[str, Any]] = []
    layer = arguments.layer_start
    for count_index, count in enumerate(DEFAULT_COUNTS):
        rotated = METHODS[count_index:] + METHODS[:count_index]
        for method in rotated:
            expert_file = model / "experts" / f"layer_{layer:02d}.bin"
            raw_path = raw_directory / f"{len(rows):02d}-{method}-{count}.json"
            print(
                f"[{len(rows) + 1}/{len(METHODS) * len(DEFAULT_COUNTS)}] "
                f"method={method} count={count} layer={layer}",
                flush=True,
            )
            row = _run_probe(
                executable,
                expert_file,
                blob_size,
                count,
                method,
                raw_path,
            )
            row["layer"] = layer
            rows.append(row)
            execution_order.append(
                {"ordinal": len(rows), "method": method, "count": count, "layer": layer}
            )
            layer += 1

    cancellation_file = model / "experts" / f"layer_{layer:02d}.bin"
    cancellation = _run_probe(
        executable,
        cancellation_file,
        blob_size,
        1,
        "mtlio_cancel",
        raw_directory / "16-mtlio-cancel-1.json",
    )
    cancellation["layer"] = layer
    execution_order.append(
        {"ordinal": 17, "method": "mtlio_cancel", "count": 1, "layer": layer}
    )

    native_gate = _native_gate(rows, cancellation)
    timing = _timing_summary(rows)
    handoff = _mlx_handoff_audit()
    decision = _integration_decision(native_gate, handoff, timing)
    environment = dict(os.environ)
    environment["DEVELOPER_DIR"] = str(developer_directory)
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "native_mtlio_ownership_and_copy_path_gate",
        "formal_performance_result": False,
        "source": _source_state(project_root, swift_source),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": _command_output(["sysctl", "-n", "hw.model"], project_root),
            "chip": _command_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], project_root
            ),
            "memory_bytes": int(
                _command_output(["sysctl", "-n", "hw.memsize"], project_root) or 0
            ),
            "xcode": subprocess.check_output(
                ["xcodebuild", "-version"], env=environment, text=True
            ).strip(),
            "sdk_version": subprocess.check_output(
                ["xcrun", "--show-sdk-version"], env=environment, text=True
            ).strip(),
            "developer_directory": str(developer_directory),
            "os_page_cache": "not purged",
        },
        "installed_model": {
            "path": str(model),
            "revision": manifest["revision"],
            "manifest_sha256": _sha256(manifest_path),
            "layer_count": layer_count,
            "expert_count": expert_count,
            "expert_blob_size": blob_size,
        },
        "protocol": {
            "counts": list(DEFAULT_COUNTS),
            "methods": list(METHODS),
            "read_workers": 4,
            "method_order": execution_order,
            "range_selection": "expert ranges 0..<count from one unique installed layer per row",
            "cache_control": "not purged; mincore observed before and after each row",
            "official_api_references": list(OFFICIAL_REFERENCES),
        },
        "compile": compile_result,
        "runs": rows,
        "cancellation": cancellation,
        "native_gate": native_gate,
        "exploratory_timing": timing,
        "mlx_handoff_audit": handoff,
        "decision": decision,
        "evidence_limits": [
            "mincore page residency is not physical SSD traffic",
            "MTLIO owns its file handle and this probe does not apply F_NOCACHE to it",
            "different unique layer files reduce direct reuse but do not balance cache state",
            "the GPU blit validates event visibility and bytes, not target-model compute overlap",
            "no ANE concurrency or energy measurement is included",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "decision": decision}, indent=2))
    if not native_gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
