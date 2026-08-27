from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models import deepseek_v4

from deepseek_v4_ssd.expert_cache import _fused_slot_regions
from deepseek_v4_ssd.io_metrics import (
    configure_expert_file_cache_policy,
    page_cache_residency_snapshot,
)
from deepseek_v4_ssd.manifest import InstalledModel, Tensor


ROW_COUNTS = (1, 4, 8)
DEFAULT_SAMPLES = 12


@dataclass(frozen=True)
class _Weights:
    w13: mx.array
    w13_scales: mx.array
    w2: mx.array
    w2_scales: mx.array


@dataclass(frozen=True)
class _ReadResult:
    bytes_read: int
    calls: int
    started: float
    finished: float

    @property
    def seconds(self) -> float:
        return self.finished - self.started


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(view: memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(view.cast("B"))
    return digest.hexdigest()


def _array_sha256(array: mx.array) -> str:
    return _bytes_sha256(memoryview(array).cast("B"))


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


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(np.ceil(len(ordered) * fraction)) - 1))
    return ordered[index]


def _view_array(packed: mx.array, region: Tensor) -> mx.array:
    value = packed[region.offset : region.offset + region.length]
    value = value.reshape(region.shape)
    if region.dtype == "I8":
        value = value.view(mx.int8)
    return value.view(mx.uint32) if region.name.endswith(".weight") else value


def _staged_regions(model: InstalledModel) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    source = {region.name: region for region in model.expert_regions}
    w13: dict[str, Tensor] = {}
    offset = 0
    for name in ("w3.weight", "w1.weight", "w3.scale", "w1.scale"):
        region = source[name]
        w13[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    w2: dict[str, Tensor] = {}
    offset = 0
    for name in ("w2.weight", "w2.scale"):
        region = source[name]
        w2[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    return w13, w2


class _Arenas:
    def __init__(self, model: InstalledModel) -> None:
        self.model = model
        self.full_regions = _fused_slot_regions(model)
        self.w13_regions, self.w2_regions = _staged_regions(model)
        self.full_size = model.expert_blob_size
        self.w13_size = sum(region.length for region in self.w13_regions.values())
        self.w2_size = sum(region.length for region in self.w2_regions.values())
        if self.w13_size + self.w2_size != self.full_size:
            raise ValueError("staged arena sizes do not equal one expert blob")

        self.full = mx.empty((self.full_size,), dtype=mx.uint8)
        self.w13 = mx.empty((self.w13_size,), dtype=mx.uint8)
        self.w2 = mx.empty((self.w2_size,), dtype=mx.uint8)
        mx.eval(self.full, self.w13, self.w2)
        self.full_memory = memoryview(self.full)
        self.w13_memory = memoryview(self.w13)
        self.w2_memory = memoryview(self.w2)

        self.full_weights = _Weights(
            w13=_view_array(self.full, self.full_regions["w13.weight"]),
            w13_scales=_view_array(self.full, self.full_regions["w13.scale"]),
            w2=_view_array(self.full, self.full_regions["w2.weight"]),
            w2_scales=_view_array(self.full, self.full_regions["w2.scale"]),
        )
        self.staged_weights = _Weights(
            w13=_view_array(self.w13, self._combined_w13_weight()),
            w13_scales=_view_array(self.w13, self._combined_w13_scale()),
            w2=_view_array(self.w2, self.w2_regions["w2.weight"]),
            w2_scales=_view_array(self.w2, self.w2_regions["w2.scale"]),
        )

    def _combined_w13_weight(self) -> Tensor:
        first = self.w13_regions["w3.weight"]
        second = self.w13_regions["w1.weight"]
        return Tensor(
            "w13.weight",
            first.dtype,
            (first.shape[0] + second.shape[0], *first.shape[1:]),
            first.offset,
            first.length + second.length,
        )

    def _combined_w13_scale(self) -> Tensor:
        first = self.w13_regions["w3.scale"]
        second = self.w13_regions["w1.scale"]
        return Tensor(
            "w13.scale",
            first.dtype,
            (first.shape[0] + second.shape[0], *first.shape[1:]),
            first.offset,
            first.length + second.length,
        )

    @staticmethod
    def _slice(view: memoryview, region: Tensor) -> memoryview:
        return view[region.offset : region.offset + region.length]

    def full_views(self) -> list[memoryview]:
        return [
            self._slice(self.full_memory, self.full_regions[region.name])
            for region in self.model.expert_regions
        ]

    def staged_w13_groups(self) -> tuple[list[memoryview], list[memoryview]]:
        return (
            [
                self._slice(self.w13_memory, self.w13_regions["w1.weight"]),
                self._slice(self.w13_memory, self.w13_regions["w1.scale"]),
            ],
            [
                self._slice(self.w13_memory, self.w13_regions["w3.weight"]),
                self._slice(self.w13_memory, self.w13_regions["w3.scale"]),
            ],
        )

    def staged_w2_views(self) -> list[memoryview]:
        return [
            self._slice(self.w2_memory, self.w2_regions["w2.weight"]),
            self._slice(self.w2_memory, self.w2_regions["w2.scale"]),
        ]

    def region_hashes(self, staged: bool) -> dict[str, str]:
        if staged:
            return {
                **{
                    name: _bytes_sha256(self._slice(self.w13_memory, region))
                    for name, region in self.w13_regions.items()
                },
                **{
                    name: _bytes_sha256(self._slice(self.w2_memory, region))
                    for name, region in self.w2_regions.items()
                },
            }
        return {
            name: _bytes_sha256(self._slice(self.full_memory, region))
            for name, region in self.full_regions.items()
            if name in {source.name for source in self.model.expert_regions}
        }

    def alignment_contract(self, alignment: int) -> dict[str, Any]:
        views = {
            "full": self.full_memory,
            "w13": self.w13_memory,
            "w2": self.w2_memory,
        }
        addresses = {
            name: ctypes.addressof(ctypes.c_char.from_buffer(view))
            for name, view in views.items()
        }
        return {
            "alignment_bytes": alignment,
            "addresses": addresses,
            "all_arena_addresses_aligned": all(
                address % alignment == 0 for address in addresses.values()
            ),
            "all_source_regions_aligned": all(
                region.offset % alignment == 0 and region.length % alignment == 0
                for region in self.model.expert_regions
            ),
            "all_destination_regions_aligned": all(
                region.offset % alignment == 0 and region.length % alignment == 0
                for region in (
                    *self.full_regions.values(),
                    *self.w13_regions.values(),
                    *self.w2_regions.values(),
                )
            ),
        }


def _pread_exact(descriptor: int, views: list[memoryview], offset: int) -> _ReadResult:
    pending = list(views)
    position = offset
    total = 0
    calls = 0
    started = time.perf_counter()
    while pending:
        count = os.preadv(descriptor, pending, position)
        calls += 1
        if count <= 0:
            raise EOFError(f"expert file ended early at offset {position}")
        total += count
        position += count
        consumed = count
        while pending and consumed >= len(pending[0]):
            consumed -= len(pending[0])
            pending.pop(0)
        if pending and consumed:
            pending[0] = pending[0][consumed:]
    return _ReadResult(total, calls, started, time.perf_counter())


def _read_full(
    descriptor: int,
    arenas: _Arenas,
    expert_offset: int,
) -> _ReadResult:
    return _pread_exact(descriptor, arenas.full_views(), expert_offset)


def _read_w13(
    descriptor: int,
    arenas: _Arenas,
    expert_offset: int,
) -> _ReadResult:
    source = {region.name: region for region in arenas.model.expert_regions}
    w1_views, w3_views = arenas.staged_w13_groups()
    started = time.perf_counter()
    w1 = _pread_exact(
        descriptor,
        w1_views,
        expert_offset + source["w1.weight"].offset,
    )
    w3 = _pread_exact(
        descriptor,
        w3_views,
        expert_offset + source["w3.weight"].offset,
    )
    return _ReadResult(
        w1.bytes_read + w3.bytes_read,
        w1.calls + w3.calls,
        started,
        time.perf_counter(),
    )


def _read_w2(
    descriptor: int,
    arenas: _Arenas,
    expert_offset: int,
) -> _ReadResult:
    source = {region.name: region for region in arenas.model.expert_regions}
    return _pread_exact(
        descriptor,
        arenas.staged_w2_views(),
        expert_offset + source["w2.weight"].offset,
    )


def _mxfp4(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )


def _first_stage(
    x: mx.array,
    weights: _Weights,
    activation: deepseek_v4.LimitedSwiGLU,
) -> tuple[mx.array, float, float, float]:
    started = time.perf_counter()
    projected = _mxfp4(x, weights.w13, weights.w13_scales)
    up, gate = mx.split(projected, 2, axis=-1)
    hidden = activation(up, gate)
    compute_started = time.perf_counter()
    mx.eval(hidden)
    compute_finished = time.perf_counter()
    return hidden, compute_finished - started, compute_started, compute_finished


def _down_stage(hidden: mx.array, weights: _Weights) -> tuple[mx.array, float]:
    started = time.perf_counter()
    output = _mxfp4(hidden, weights.w2, weights.w2_scales).astype(mx.float32)
    mx.eval(output)
    return output, time.perf_counter() - started


def _residency(
    descriptor: int,
    offset: int,
    length: int,
) -> dict[str, Any]:
    snapshot = page_cache_residency_snapshot(descriptor, offset, length)
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


def _control_run(
    descriptor: int,
    arenas: _Arenas,
    expert_offset: int,
    x: mx.array,
    activation: deepseek_v4.LimitedSwiGLU,
) -> tuple[dict[str, Any], mx.array]:
    residency_before = _residency(
        descriptor,
        expert_offset,
        arenas.model.expert_blob_size,
    )
    started = time.perf_counter()
    read = _read_full(descriptor, arenas, expert_offset)
    hidden, first_seconds, _, _ = _first_stage(
        x,
        arenas.full_weights,
        activation,
    )
    output, down_seconds = _down_stage(hidden, arenas.full_weights)
    complete_seconds = time.perf_counter() - started
    residency_after = _residency(
        descriptor,
        expert_offset,
        arenas.model.expert_blob_size,
    )
    return (
        {
            "method": "full_slot_control",
            "bytes_read": read.bytes_read,
            "preadv_calls": read.calls,
            "read_seconds": read.seconds,
            "w13_compute_seconds": first_seconds,
            "down_compute_seconds": down_seconds,
            "complete_seconds": complete_seconds,
            "output_sha256": _array_sha256(output),
            "page_cache_before": residency_before,
            "page_cache_after": residency_after,
        },
        output,
    )


def _staged_run(
    descriptor: int,
    arenas: _Arenas,
    expert_offset: int,
    x: mx.array,
    activation: deepseek_v4.LimitedSwiGLU,
    executor: ThreadPoolExecutor,
) -> tuple[dict[str, Any], mx.array]:
    residency_before = _residency(
        descriptor,
        expert_offset,
        arenas.model.expert_blob_size,
    )
    started = time.perf_counter()
    w13_read = _read_w13(descriptor, arenas, expert_offset)
    read_started = threading.Event()

    def read_w2() -> _ReadResult:
        read_started.set()
        return _read_w2(descriptor, arenas, expert_offset)

    future = executor.submit(read_w2)
    read_started.wait()
    hidden, first_seconds, compute_started, compute_finished = _first_stage(
        x,
        arenas.staged_weights,
        activation,
    )
    wait_started = time.perf_counter()
    w2_read = future.result()
    wait_seconds = time.perf_counter() - wait_started
    output, down_seconds = _down_stage(hidden, arenas.staged_weights)
    complete_seconds = time.perf_counter() - started
    overlap = max(
        0.0,
        min(w2_read.finished, compute_finished)
        - max(w2_read.started, compute_started),
    )
    residency_after = _residency(
        descriptor,
        expert_offset,
        arenas.model.expert_blob_size,
    )
    return (
        {
            "method": "staged_overlap",
            "bytes_read": w13_read.bytes_read + w2_read.bytes_read,
            "preadv_calls": w13_read.calls + w2_read.calls,
            "w13_read_bytes": w13_read.bytes_read,
            "w2_read_bytes": w2_read.bytes_read,
            "w13_read_seconds": w13_read.seconds,
            "w2_read_seconds": w2_read.seconds,
            "w13_compute_seconds": first_seconds,
            "w2_wait_seconds": wait_seconds,
            "down_compute_seconds": down_seconds,
            "w2_w13_overlap_seconds": overlap,
            "w2_read_hidden_fraction": (
                overlap / w2_read.seconds if w2_read.seconds > 0 else 0.0
            ),
            "complete_seconds": complete_seconds,
            "output_sha256": _array_sha256(output),
            "page_cache_before": residency_before,
            "page_cache_after": residency_after,
        },
        output,
    )


def _make_input(rows: int) -> mx.array:
    values = mx.arange(rows * 4096, dtype=mx.float32)
    values = ((values % 257) - 128) / 128
    array = values.reshape(1, rows, 4096).astype(mx.bfloat16)
    mx.eval(array)
    return array


def _find_nonresident_expert(
    descriptors: list[int],
    model: InstalledModel,
    used: set[tuple[int, int]],
) -> tuple[int, int, dict[str, Any]]:
    for layer, descriptor in enumerate(descriptors):
        for expert in range(model.expert_count):
            key = (layer, expert)
            if key in used:
                continue
            offset = expert * model.expert_blob_size
            snapshot = _residency(
                descriptor,
                offset,
                model.expert_blob_size,
            )
            if snapshot["available"] and snapshot["resident_bytes"] == 0:
                used.add(key)
                return layer, expert, snapshot
    raise RuntimeError("no fully nonresident installed expert range remains")


def _correctness_gate(
    pairs: list[dict[str, Any]],
    model: InstalledModel,
    arenas: _Arenas,
    alignment: dict[str, Any],
) -> dict[str, Any]:
    criteria = {
        "all_full_and_staged_bytes_exact": all(
            pair["control"]["bytes_read"] == model.expert_blob_size
            and pair["staged"]["bytes_read"] == model.expert_blob_size
            and pair["staged"]["w13_read_bytes"] == arenas.w13_size
            and pair["staged"]["w2_read_bytes"] == arenas.w2_size
            for pair in pairs
        ),
        "all_six_region_hashes_exact": all(
            pair["region_hashes_exact"] for pair in pairs
        ),
        "all_float32_output_hashes_exact": all(
            pair["output_exact"] for pair in pairs
        ),
        "fixed_candidate_arena_bytes_equal_full_slot": (
            arenas.w13_size + arenas.w2_size == arenas.full_size
        ),
        "all_direct_io_alignment_contracts_exact": all(
            bool(alignment[key])
            for key in (
                "all_arena_addresses_aligned",
                "all_source_regions_aligned",
                "all_destination_regions_aligned",
            )
        ),
        "all_bypass_ranges_remain_nonresident": all(
            observation["available"]
            and observation["resident_bytes"] == 0
            for pair in pairs
            for method in ("control", "staged")
            for observation in (
                pair[method]["page_cache_before"],
                pair[method]["page_cache_after"],
            )
        ),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


def _shape_summary(pairs: list[dict[str, Any]], rows: int) -> dict[str, Any]:
    selected = [pair for pair in pairs if pair["rows"] == rows]
    control = [pair["control"]["complete_seconds"] for pair in selected]
    staged = [pair["staged"]["complete_seconds"] for pair in selected]
    hidden = [pair["staged"]["w2_read_hidden_fraction"] for pair in selected]
    wait = [pair["staged"]["w2_wait_seconds"] for pair in selected]
    overlap = [pair["staged"]["w2_w13_overlap_seconds"] for pair in selected]
    control_median = statistics.median(control)
    staged_median = statistics.median(staged)
    control_p95 = _percentile(control, 0.95)
    staged_p95 = _percentile(staged, 0.95)
    return {
        "rows": rows,
        "sample_count": len(selected),
        "control_complete_median_seconds": control_median,
        "staged_complete_median_seconds": staged_median,
        "complete_median_change_fraction": (
            staged_median / control_median - 1 if control_median else None
        ),
        "control_complete_p95_seconds": control_p95,
        "staged_complete_p95_seconds": staged_p95,
        "complete_p95_change_fraction": (
            staged_p95 / control_p95 - 1 if control_p95 else None
        ),
        "w2_read_hidden_fraction_median": statistics.median(hidden),
        "w2_wait_median_seconds": statistics.median(wait),
        "w2_w13_overlap_median_seconds": statistics.median(overlap),
        "w13_read_median_seconds": statistics.median(
            pair["staged"]["w13_read_seconds"] for pair in selected
        ),
        "w2_read_median_seconds": statistics.median(
            pair["staged"]["w2_read_seconds"] for pair in selected
        ),
        "w13_compute_median_seconds": statistics.median(
            pair["staged"]["w13_compute_seconds"] for pair in selected
        ),
        "down_compute_median_seconds": statistics.median(
            pair["staged"]["down_compute_seconds"] for pair in selected
        ),
    }


def _decision(
    correctness: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for summary in summaries:
        if summary["rows"] not in {4, 8}:
            continue
        criteria = {
            "w2_read_hidden_at_least_20_percent": (
                summary["w2_read_hidden_fraction_median"] >= 0.20
            ),
            "complete_median_improves_at_least_5_percent": (
                summary["complete_median_change_fraction"] <= -0.05
            ),
            "complete_p95_regression_at_most_5_percent": (
                summary["complete_p95_change_fraction"] <= 0.05
            ),
        }
        candidates.append(
            {
                "rows": summary["rows"],
                "criteria": criteria,
                "passed": all(criteria.values()),
            }
        )
    continue_to_runtime = bool(
        correctness["passed"] and any(candidate["passed"] for candidate in candidates)
    )
    return {
        "shape_candidates": candidates,
        "continue_to_runtime_token_hash_prototype": continue_to_runtime,
        "outcome": (
            "continue_to_isolated_runtime_token_hash_prototype"
            if continue_to_runtime
            else "stop_fixed_arena_staged_overlap_gate_not_met"
        ),
    }


def _source_state(project_root: Path) -> dict[str, Any]:
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
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "expert_cache_sha256": _sha256(
            project_root / "runtime" / "deepseek_v4_ssd" / "expert_cache.py"
        ),
        "model_runtime_sha256": _sha256(
            project_root / "runtime" / "deepseek_v4_ssd" / "model.py"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fixed-arena w13 compute overlap with a direct w2 read"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    arguments = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("the staged direct-I/O gate requires Darwin")
    if arguments.samples < 3:
        parser.error("--samples must be at least 3")

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(arguments.model).expanduser().resolve()
    output_path = Path(arguments.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = InstalledModel.open(model_path)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    activation = deepseek_v4.LimitedSwiGLU(float(config["swiglu_limit"]))
    arenas = _Arenas(model)
    filesystem = os.statvfs(model_path / "experts")
    alignment_bytes = int(filesystem.f_frsize or filesystem.f_bsize)
    alignment = arenas.alignment_contract(alignment_bytes)
    if not all(
        alignment[key]
        for key in (
            "all_arena_addresses_aligned",
            "all_source_regions_aligned",
            "all_destination_regions_aligned",
        )
    ):
        raise RuntimeError("fixed arenas do not satisfy direct-I/O alignment")

    descriptors: list[int] = []
    try:
        for layer in range(model.layer_count):
            descriptor = os.open(
                model_path / "experts" / f"layer_{layer:02d}.bin",
                os.O_RDONLY,
            )
            try:
                configure_expert_file_cache_policy(descriptor, "bypass")
            except Exception:
                os.close(descriptor)
                raise
            descriptors.append(descriptor)

        used: set[tuple[int, int]] = set()
        warm_layer, warm_expert, warm_residency = _find_nonresident_expert(
            descriptors,
            model,
            used,
        )
        warm_offset = warm_expert * model.expert_blob_size
        _read_full(descriptors[warm_layer], arenas, warm_offset)
        _read_w13(descriptors[warm_layer], arenas, warm_offset)
        _read_w2(descriptors[warm_layer], arenas, warm_offset)
        inputs = {rows: _make_input(rows) for rows in ROW_COUNTS}
        input_hashes = {
            rows: _array_sha256(inputs[rows].astype(mx.float32)) for rows in ROW_COUNTS
        }
        for rows in ROW_COUNTS:
            for weights in (arenas.full_weights, arenas.staged_weights):
                for _ in range(2):
                    hidden, _, _, _ = _first_stage(inputs[rows], weights, activation)
                    _down_stage(hidden, weights)

        pairs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            total_pairs = len(ROW_COUNTS) * arguments.samples
            for rows in ROW_COUNTS:
                for sample in range(arguments.samples):
                    layer, expert, selected_residency = _find_nonresident_expert(
                        descriptors,
                        model,
                        used,
                    )
                    descriptor = descriptors[layer]
                    expert_offset = expert * model.expert_blob_size
                    staged_first = sample % 2 == 1
                    print(
                        f"[{len(pairs) + 1}/{total_pairs}] rows={rows} "
                        f"layer={layer} expert={expert} "
                        f"order={'staged-control' if staged_first else 'control-staged'}",
                        flush=True,
                    )
                    if staged_first:
                        staged, staged_output = _staged_run(
                            descriptor,
                            arenas,
                            expert_offset,
                            inputs[rows],
                            activation,
                            executor,
                        )
                        control, control_output = _control_run(
                            descriptor,
                            arenas,
                            expert_offset,
                            inputs[rows],
                            activation,
                        )
                    else:
                        control, control_output = _control_run(
                            descriptor,
                            arenas,
                            expert_offset,
                            inputs[rows],
                            activation,
                        )
                        staged, staged_output = _staged_run(
                            descriptor,
                            arenas,
                            expert_offset,
                            inputs[rows],
                            activation,
                            executor,
                        )
                    full_hashes = arenas.region_hashes(staged=False)
                    staged_hashes = arenas.region_hashes(staged=True)
                    pair = {
                        "rows": rows,
                        "sample": sample,
                        "layer": layer,
                        "expert": expert,
                        "expert_offset": expert_offset,
                        "order": (
                            "staged-control" if staged_first else "control-staged"
                        ),
                        "selected_page_cache": selected_residency,
                        "control": control,
                        "staged": staged,
                        "full_region_sha256": full_hashes,
                        "staged_region_sha256": staged_hashes,
                        "region_hashes_exact": full_hashes == staged_hashes,
                        "output_exact": (
                            control["output_sha256"] == staged["output_sha256"]
                        ),
                        "complete_change_fraction": (
                            staged["complete_seconds"]
                            / control["complete_seconds"]
                            - 1
                        ),
                    }
                    pairs.append(pair)
                    del control_output, staged_output
    finally:
        for descriptor in descriptors:
            os.close(descriptor)

    correctness = _correctness_gate(pairs, model, arenas, alignment)
    summaries = [_shape_summary(pairs, rows) for rows in ROW_COUNTS]
    decision = _decision(correctness, summaries)
    manifest_path = model_path / "manifest.json"
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "fixed_arena_staged_w13_w2_overlap_gate",
        "formal_performance_result": False,
        "source": _source_state(project_root),
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
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "os_page_cache": "fully nonresident just-in-time selections; bypass descriptor",
            "storage_cache_limit": "F_NOCACHE does not prove device-controller cache state",
        },
        "installed_model": {
            "path": str(model_path),
            "revision": model.revision,
            "manifest_sha256": _sha256(manifest_path),
            "expert_blob_size": model.expert_blob_size,
            "expert_regions": [
                {
                    "name": region.name,
                    "dtype": region.dtype,
                    "shape": list(region.shape),
                    "offset": region.offset,
                    "length": region.length,
                }
                for region in model.expert_regions
            ],
        },
        "protocol": {
            "row_counts": list(ROW_COUNTS),
            "samples_per_shape": arguments.samples,
            "control": "one preadv into current fused full slot, then w13 and w2 eval",
            "candidate": "two w13 preadv calls, background w2 preadv overlapped with w13 eval, then w2 eval",
            "method_order": "alternating control-staged and staged-control",
            "file_cache_policy": "F_NOCACHE and F_RDAHEAD disabled",
            "warmup_iterations_per_shape_and_layout": 2,
            "input_float32_sha256_by_rows": {
                str(rows): digest for rows, digest in input_hashes.items()
            },
            "swiglu_limit": float(config["swiglu_limit"]),
            "optimistic_route_distribution": (
                "all rows use one expert; 4/8 rows are an upper bound, not an observed route trace"
            ),
        },
        "fixed_arenas": {
            "full_slot_bytes": arenas.full_size,
            "w13_bytes": arenas.w13_size,
            "w2_bytes": arenas.w2_size,
            "candidate_total_bytes": arenas.w13_size + arenas.w2_size,
            "alignment": alignment,
        },
        "warmup_expert": {
            "layer": warm_layer,
            "expert": warm_expert,
            "page_cache_before": warm_residency,
        },
        "pairs": pairs,
        "correctness_gate": correctness,
        "shape_summaries": summaries,
        "decision": decision,
        "evidence_limits": [
            "isolated single-expert compute, not a complete routed MoE layer",
            "4/8 rows intentionally over-concentrate routes on one expert",
            "no attention, route reduction, cache eviction, or full-model token hash",
            "mincore plus F_NOCACHE is not a physical-device byte counter",
            "no MTLIO, ANE concurrency, energy, or end-to-end decode throughput",
        ],
    }
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "correctness": correctness["passed"],
                "decision": decision,
            },
            indent=2,
        )
    )
    if not correctness["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
