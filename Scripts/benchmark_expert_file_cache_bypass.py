from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import json
import mmap
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

try:
    from .benchmark_common import (
        _command_output,
        _package_version,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )
except ImportError:
    from benchmark_common import (
        _command_output,
        _package_version,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )

from deepseek_v4_ssd.io_metrics import (
    configure_expert_file_cache_policy,
    page_cache_residency_snapshot,
    process_disk_io_snapshot,
)


def _residency(descriptor: int, offset: int, length: int) -> dict:
    snapshot = page_cache_residency_snapshot(descriptor, offset, length)
    if snapshot is None:
        raise RuntimeError("page-cache residency probe is unavailable")
    classification = snapshot.classify(length)
    return {
        "classified_bytes": classification.classified_bytes,
        "resident_bytes": classification.resident_bytes,
        "nonresident_bytes": classification.nonresident_bytes,
        "unclassified_bytes": classification.unclassified_bytes,
        "probe_calls": classification.probe_calls,
        "probe_failures": classification.probe_failures,
    }


def _aligned_read(
    descriptor: int,
    offset: int,
    length: int,
    alignment: int,
) -> dict:
    buffer = mmap.mmap(-1, length, access=mmap.ACCESS_WRITE)
    view = memoryview(buffer)
    address = ctypes.addressof(ctypes.c_char.from_buffer(view))
    if address % alignment or offset % alignment or length % alignment:
        view.release()
        buffer.close()
        raise RuntimeError("benchmark direct-read range is not aligned")
    disk_before = process_disk_io_snapshot()
    started = time.perf_counter()
    count = os.preadv(descriptor, [view], offset)
    elapsed = time.perf_counter() - started
    disk_after = process_disk_io_snapshot()
    digest = hashlib.sha256(view[:count]).hexdigest()
    view.release()
    buffer.close()
    disk_delta = (
        disk_after.delta(disk_before)
        if disk_before is not None and disk_after is not None
        else None
    )
    return {
        "bytes_read": count,
        "sha256": digest,
        "seconds": elapsed,
        "destination_address_alignment_bytes": alignment,
        "process_disk_bytes_read": (
            disk_delta.bytes_read if disk_delta is not None else None
        ),
        "process_disk_bytes_written": (
            disk_delta.bytes_written if disk_delta is not None else None
        ),
    }


def _range_contract_passed(row: dict, blob_size: int) -> bool:
    return bool(
        row["before"]["resident_bytes"] == 0
        and row["before"]["nonresident_bytes"] == blob_size
        and row["bypass_read"]["bytes_read"] == blob_size
        and row["after_bypass"]["resident_bytes"] == 0
        and row["after_bypass"]["nonresident_bytes"] == blob_size
        and row["cached_read"]["bytes_read"] == blob_size
        and row["after_cached"]["resident_bytes"] == blob_size
        and row["after_cached"]["nonresident_bytes"] == 0
        and row["bypass_read"]["sha256"] == row["cached_read"]["sha256"]
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
        "xnu_contract_revision": (
            "f6217f891ac0bb64f3d375211650a4c1ff8ca1ea"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate aligned Darwin F_NOCACHE expert reads against cached "
            "reads and pre/post mincore residency"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.sample_count < 1:
        parser.error("--sample-count must be positive")
    if sys.platform != "darwin":
        parser.error("this contract requires Darwin")

    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = model / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    blob_size = int(manifest["expertBlobSize"])
    layer_count = int(manifest["layerCount"])
    expert_count = int(manifest["expertCount"])
    filesystem = os.statvfs(model / "experts")
    alignment = int(filesystem.f_frsize or filesystem.f_bsize)

    layout_aligned = bool(
        blob_size % alignment == 0
        and all(
            int(region["offset"]) % alignment == 0
            and int(region["length"]) % alignment == 0
            for region in manifest["expertRegions"]
        )
    )
    if not layout_aligned:
        raise RuntimeError("installed expert layout is not direct-I/O aligned")

    rows = []
    scanned = 0
    for layer in reversed(range(layer_count)):
        expert_file = model / "experts" / f"layer_{layer:02d}.bin"
        cached_descriptor = os.open(expert_file, os.O_RDONLY)
        bypass_descriptor = os.open(expert_file, os.O_RDONLY)
        try:
            configure_expert_file_cache_policy(bypass_descriptor, "bypass")
            for expert in range(expert_count):
                scanned += 1
                offset = expert * blob_size
                before = _residency(cached_descriptor, offset, blob_size)
                if before["resident_bytes"] != 0:
                    continue
                index = len(rows) + 1
                print(
                    f"[{index}/{arguments.sample_count}] "
                    f"layer={layer} expert={expert}",
                    flush=True,
                )
                bypass_read = _aligned_read(
                    bypass_descriptor,
                    offset,
                    blob_size,
                    alignment,
                )
                after_bypass = _residency(
                    cached_descriptor,
                    offset,
                    blob_size,
                )
                cached_read = _aligned_read(
                    cached_descriptor,
                    offset,
                    blob_size,
                    alignment,
                )
                after_cached = _residency(
                    cached_descriptor,
                    offset,
                    blob_size,
                )
                row = {
                    "layer": layer,
                    "expert": expert,
                    "file": expert_file.name,
                    "offset": offset,
                    "length": blob_size,
                    "before": before,
                    "bypass_read": bypass_read,
                    "after_bypass": after_bypass,
                    "cached_read": cached_read,
                    "after_cached": after_cached,
                }
                row["contract_passed"] = _range_contract_passed(
                    row,
                    blob_size,
                )
                rows.append(row)
                if len(rows) == arguments.sample_count:
                    break
        finally:
            os.close(bypass_descriptor)
            os.close(cached_descriptor)
        if len(rows) == arguments.sample_count:
            break
    if len(rows) != arguments.sample_count:
        raise RuntimeError(
            f"measured only {len(rows)} fully nonresident expert ranges"
        )

    contract_passed = all(row["contract_passed"] for row in rows)
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "expert_file_cache_bypass_contract",
        "formal_performance_result": False,
        "source": _source_state(project_root),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": _sysctl("hw.model", project_root),
            "chip": _sysctl("machdep.cpu.brand_string", project_root),
            "memory_bytes": int(_sysctl("hw.memsize", project_root) or 0),
            "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
            "filesystem_block_size_bytes": alignment,
            "filesystem_preferred_transfer_size_bytes": int(
                filesystem.f_bsize
            ),
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
        },
        "checkpoint": {
            "installed_model": str(model),
            "model_id": manifest["modelID"],
            "revision": manifest["revision"],
            "manifest_sha256": _sha256(manifest_path),
            "expert_blob_bytes": blob_size,
            "expert_layout_aligned": layout_aligned,
        },
        "experiment": {
            "cache_bypass": "F_NOCACHE=1 and F_RDAHEAD=0",
            "cached_control": "default descriptor flags",
            "selection": (
                "just-in-time fully nonresident ranges in reverse layer order; "
                "recheck after every cached control read"
            ),
            "sample_count": arguments.sample_count,
            "ranges_scanned": scanned,
            "same_range_order": ["bypass", "cached"],
        },
        "ranges": rows,
        "summary": {
            "ranges_passed": sum(row["contract_passed"] for row in rows),
            "ranges_measured": len(rows),
            "all_byte_hashes_exact": all(
                row["bypass_read"]["sha256"]
                == row["cached_read"]["sha256"]
                for row in rows
            ),
            "all_bypass_reads_left_nonresident": all(
                row["after_bypass"]["resident_bytes"] == 0 for row in rows
            ),
            "all_cached_reads_became_resident": all(
                row["after_cached"]["resident_bytes"] == blob_size
                for row in rows
            ),
            "contract_passed": contract_passed,
        },
        "decision": {
            "status": (
                "cache_bypass_contract_passed"
                if contract_passed
                else "stop_cache_bypass_candidate"
            ),
            "run_full_model_wave": contract_passed,
            "adopt_as_default": False,
        },
        "evidence_limits": [
            "F_NOCACHE does not purge pages resident before descriptor configuration.",
            "mincore reports VM page residency, not storage-controller or device-cache state.",
            "Process disk counters remain process-wide rather than expert-file-specific.",
            "This contract validates aligned installed expert ranges, not full-model speed.",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    if not contract_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
