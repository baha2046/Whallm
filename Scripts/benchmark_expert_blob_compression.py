from __future__ import annotations

import argparse
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from deepseek_v4_ssd.io_metrics import (
    configure_expert_file_cache_policy,
    page_cache_residency_snapshot,
)
from deepseek_v4_ssd.manifest import InstalledModel

try:
    from .benchmark_dspark_adaptive import (
        _command_output,
        _package_version,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )
except ImportError:
    from benchmark_dspark_adaptive import (
        _command_output,
        _package_version,
        _runtime_tree_sha256,
        _sha256,
        _sysctl,
    )


CODECS = {"lz4": 0x100, "lzfse": 0x801}
FIXED_SIZES = (64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024)


class _AppleCompression:
    def __init__(self, codec: str) -> None:
        try:
            algorithm = CODECS[codec]
        except KeyError as error:
            raise ValueError(f"unsupported codec: {codec}") from error
        self.codec = codec
        self.algorithm = algorithm
        self.library = ctypes.CDLL("/usr/lib/libcompression.dylib")
        self.library.compression_encode_scratch_buffer_size.argtypes = [ctypes.c_int]
        self.library.compression_encode_scratch_buffer_size.restype = ctypes.c_size_t
        self.library.compression_decode_scratch_buffer_size.argtypes = [ctypes.c_int]
        self.library.compression_decode_scratch_buffer_size.restype = ctypes.c_size_t
        arguments = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.library.compression_encode_buffer.argtypes = arguments
        self.library.compression_encode_buffer.restype = ctypes.c_size_t
        self.library.compression_decode_buffer.argtypes = arguments
        self.library.compression_decode_buffer.restype = ctypes.c_size_t
        self.encode_scratch_bytes = int(
            self.library.compression_encode_scratch_buffer_size(algorithm)
        )
        self.decode_scratch_bytes = int(
            self.library.compression_decode_scratch_buffer_size(algorithm)
        )

    @staticmethod
    def _address(buffer: bytearray) -> int:
        return ctypes.addressof(ctypes.c_ubyte.from_buffer(buffer))

    def encode(self, source: bytes) -> bytes:
        source_buffer = bytearray(source)
        destination = bytearray(len(source) * 2 + 4096)
        scratch = bytearray(self.encode_scratch_bytes)
        count = self.library.compression_encode_buffer(
            self._address(destination),
            len(destination),
            self._address(source_buffer),
            len(source_buffer),
            self._address(scratch) if scratch else None,
            self.algorithm,
        )
        if not count:
            raise RuntimeError(f"{self.codec} compression failed")
        return bytes(destination[:count])

    def decode(self, source: bytes, output_bytes: int) -> bytes:
        source_buffer = bytearray(source)
        destination = bytearray(output_bytes)
        scratch = bytearray(self.decode_scratch_bytes)
        count = self.decode_into(source_buffer, destination, scratch)
        if count != output_bytes:
            raise RuntimeError(f"{self.codec} decompression returned {count} bytes")
        return bytes(destination)

    def decode_into(
        self,
        source: bytearray,
        destination: bytearray,
        scratch: bytearray,
    ) -> int:
        return int(
            self.library.compression_decode_buffer(
                self._address(destination),
                len(destination),
                self._address(source),
                len(source),
                self._address(scratch) if scratch else None,
                self.algorithm,
            )
        )


def _sample_specs(blob_size: int) -> list[tuple[str, int, str]]:
    rows = [
        (f"fixed_{size // 1024}kib", size, "contiguous_layer_range")
        for size in FIXED_SIZES
    ]
    rows.append(("full_expert", blob_size, "single_expert_blob"))
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def _read_exact(descriptor: int, destination: bytearray) -> None:
    view = memoryview(destination)
    total = 0
    try:
        while total < len(view):
            count = os.preadv(descriptor, [view[total:]], total)
            if count <= 0:
                raise EOFError("benchmark file ended early")
            total += count
    finally:
        view.release()


def _residency(descriptor: int, length: int) -> dict[str, int | bool | None]:
    snapshot = page_cache_residency_snapshot(descriptor, 0, length)
    if snapshot is None:
        return {"available": False, "resident_bytes": None, "nonresident_bytes": None}
    classification = snapshot.classify(length)
    return {
        "available": True,
        "resident_bytes": classification.resident_bytes,
        "nonresident_bytes": classification.nonresident_bytes,
    }


class _MetalLoad:
    def __init__(self) -> None:
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.iterations = 0
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            import mlx.core as mx

            left = mx.ones((2048, 2048), dtype=mx.float16)
            right = mx.ones((2048, 2048), dtype=mx.float16)
            mx.eval(left, right)
            self.ready.set()
            while not self.stop.is_set():
                mx.eval(left @ right)
                self.iterations += 1
        except BaseException as error:
            self.error = error
            self.ready.set()

    def __enter__(self) -> _MetalLoad:
        self.thread.start()
        if not self.ready.wait(timeout=30):
            raise TimeoutError("Metal load did not start")
        if self.error is not None:
            raise RuntimeError("Metal load failed") from self.error
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("Metal load did not stop")
        if self.error is not None:
            raise RuntimeError("Metal load failed") from self.error


def _worker(arguments: argparse.Namespace) -> dict[str, Any]:
    path = Path(arguments.worker_path)
    descriptor = os.open(path, os.O_RDONLY)
    configure_expert_file_cache_policy(descriptor, "bypass")
    codec = (
        _AppleCompression(arguments.worker_codec)
        if arguments.worker_mode == "compressed"
        else None
    )
    source_size = int(arguments.worker_output_bytes)
    read_size = path.stat().st_size
    read_buffer = bytearray(read_size)
    output_buffer = bytearray(source_size)
    scratch = bytearray(codec.decode_scratch_bytes if codec is not None else 0)
    elapsed_samples: list[float] = []
    read_samples: list[float] = []
    decode_samples: list[float] = []
    before = _residency(descriptor, read_size)
    metal_context = _MetalLoad() if arguments.worker_metal else None

    def measure() -> None:
        started = time.perf_counter()
        _read_exact(descriptor, read_buffer)
        read_finished = time.perf_counter()
        if codec is not None:
            count = codec.decode_into(read_buffer, output_buffer, scratch)
            if count != source_size:
                raise RuntimeError(
                    f"{codec.codec} decompression returned {count} bytes"
                )
        finished = time.perf_counter()
        elapsed_samples.append(finished - started)
        read_samples.append(read_finished - started)
        decode_samples.append(finished - read_finished)

    try:
        if metal_context is not None:
            metal_context.__enter__()
        measure()
        elapsed_samples.clear()
        read_samples.clear()
        decode_samples.clear()
        for _ in range(arguments.samples):
            measure()
    finally:
        if metal_context is not None:
            metal_context.__exit__(None, None, None)
        os.close(descriptor)

    digest = hashlib.sha256(output_buffer if codec is not None else read_buffer).hexdigest()
    elapsed_p50 = statistics.median(elapsed_samples)
    decode_p50 = statistics.median(decode_samples)
    return {
        "mode": arguments.worker_mode,
        "metal": bool(arguments.worker_metal),
        "samples": arguments.samples,
        "source_bytes": source_size,
        "stored_bytes": read_size,
        "sha256": digest,
        "page_cache_before": before,
        "elapsed_p50_seconds": elapsed_p50,
        "elapsed_p95_seconds": _percentile(elapsed_samples, 0.95),
        "read_p50_seconds": statistics.median(read_samples),
        "decode_p50_seconds": decode_p50,
        "decode_output_bytes_per_second": (
            source_size / decode_p50 if decode_p50 else None
        ),
        "temporary_buffer_bytes": (
            read_size + len(scratch) if codec is not None else 0
        ),
        "working_buffer_bytes": source_size + read_size + len(scratch),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "metal_iterations": metal_context.iterations if metal_context else 0,
    }


def _write_bypass(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    configure_expert_file_cache_policy(descriptor, "bypass")
    try:
        view = memoryview(data)
        total = 0
        while total < len(view):
            count = os.write(descriptor, view[total:])
            if count <= 0:
                raise OSError("benchmark file write made no progress")
            total += count
        os.fsync(descriptor)
        view.release()
    finally:
        os.close(descriptor)


def _run_worker(
    script: Path,
    *,
    mode: str,
    path: Path,
    output_bytes: int,
    samples: int,
    metal: bool,
    codec: str | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--worker-mode",
        mode,
        "--worker-path",
        str(path),
        "--worker-output-bytes",
        str(output_bytes),
        "--samples",
        str(samples),
    ]
    if metal:
        command.append("--worker-metal")
    if codec is not None:
        command.extend(("--worker-codec", codec))
    return json.loads(subprocess.check_output(command, text=True))


def _comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    improvement = 1 - (
        candidate["elapsed_p50_seconds"] / baseline["elapsed_p50_seconds"]
    )
    peak_change = (
        candidate["peak_rss_bytes"] / baseline["peak_rss_bytes"] - 1
    )
    return {
        "elapsed_p50_improvement_fraction": improvement,
        "peak_rss_change_fraction": peak_change,
        "passed_5_percent_time_gate": improvement >= 0.05,
        "passed_5_percent_peak_memory_gate": peak_change <= 0.05,
    }


def _gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    criteria = {
        "all_decompressed_hashes_exact": all(row["exact"] for row in rows),
        "all_normal_cases_improve_at_least_5_percent": all(
            row["normal_comparison"]["passed_5_percent_time_gate"] for row in rows
        ),
        "all_metal_cases_improve_at_least_5_percent": all(
            row["metal_comparison"]["passed_5_percent_time_gate"] for row in rows
        ),
        "all_peak_rss_increases_at_most_5_percent": all(
            row[mode]["passed_5_percent_peak_memory_gate"]
            for row in rows
            for mode in ("normal_comparison", "metal_comparison")
        ),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


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
        "runtime_python_tree_sha256": _runtime_tree_sha256(
            project_root / "runtime" / "deepseek_v4_ssd"
        ),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
    }


def _model_description(model: InstalledModel) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "revision": model.revision,
        "model_kind": model.model_kind,
        "layer_count": model.layer_count,
        "expert_count": model.expert_count,
        "expert_blob_bytes": model.expert_blob_size,
        "expert_quantization": (
            {
                "mode": model.expert_quantization.mode,
                "bits": model.expert_quantization.bits,
                "group_size": model.expert_quantization.group_size,
            }
            if model.expert_quantization is not None
            else {
                "mode": "checkpoint_native_fp4",
                "storage_dtype": "I8",
            }
        ),
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
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["model"], row["size_name"], row["codec"]) for row in rows})
    for model, size_name, codec in keys:
        selected = [
            row
            for row in rows
            if (row["model"], row["size_name"], row["codec"])
            == (model, size_name, codec)
        ]
        result.append(
            {
                "model": model,
                "size_name": size_name,
                "source_bytes": selected[0]["source_bytes"],
                "source_kind": selected[0]["source_kind"],
                "codec": codec,
                "expert_samples": len(selected),
                "compression_ratio_median": statistics.median(
                    row["compression_ratio"] for row in selected
                ),
                "decode_output_gbps_median": statistics.median(
                    row["candidate_normal"]["decode_output_bytes_per_second"]
                    for row in selected
                )
                / 1_000_000_000,
                "normal_elapsed_improvement_median_fraction": statistics.median(
                    row["normal_comparison"]["elapsed_p50_improvement_fraction"]
                    for row in selected
                ),
                "metal_elapsed_improvement_median_fraction": statistics.median(
                    row["metal_comparison"]["elapsed_p50_improvement_fraction"]
                    for row in selected
                ),
                "peak_rss_change_max_fraction": max(
                    row[comparison]["peak_rss_change_fraction"]
                    for row in selected
                    for comparison in ("normal_comparison", "metal_comparison")
                ),
                "temporary_buffer_bytes_median": int(
                    statistics.median(
                        row["candidate_normal"]["temporary_buffer_bytes"]
                        for row in selected
                    )
                ),
                "all_hashes_exact": all(row["exact"] for row in selected),
                "all_normal_time_gates_passed": all(
                    row["normal_comparison"]["passed_5_percent_time_gate"]
                    for row in selected
                ),
                "all_metal_time_gates_passed": all(
                    row["metal_comparison"]["passed_5_percent_time_gate"]
                    for row in selected
                ),
                "all_peak_memory_gates_passed": all(
                    row[comparison]["passed_5_percent_peak_memory_gate"]
                    for row in selected
                    for comparison in ("normal_comparison", "metal_comparison")
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Apple Compression against installed expert bytes"
    )
    parser.add_argument("--deepseek-model")
    parser.add_argument("--qwen-model")
    parser.add_argument("--output")
    parser.add_argument("--scratch-directory")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--codecs", nargs="+", choices=tuple(CODECS), default=list(CODECS))
    parser.add_argument("--expert-samples", type=int, default=3)
    parser.add_argument("--worker-mode", choices=("baseline", "compressed"))
    parser.add_argument("--worker-path")
    parser.add_argument("--worker-output-bytes", type=int)
    parser.add_argument("--worker-codec", choices=tuple(CODECS))
    parser.add_argument("--worker-metal", action="store_true")
    arguments = parser.parse_args()

    if arguments.worker_mode is not None:
        if arguments.worker_path is None or arguments.worker_output_bytes is None:
            parser.error("worker mode requires a path and output size")
        if arguments.worker_mode == "compressed" and arguments.worker_codec is None:
            parser.error("compressed worker mode requires a codec")
        print(json.dumps(_worker(arguments), separators=(",", ":")))
        return

    if sys.platform != "darwin":
        parser.error("this benchmark requires Darwin")
    if not arguments.deepseek_model or not arguments.qwen_model:
        parser.error("both installed model paths are required")
    if not arguments.output or not arguments.scratch_directory:
        parser.error("output and scratch directory are required")
    if arguments.samples < 3:
        parser.error("--samples must be at least 3")
    if arguments.expert_samples != 3:
        parser.error("this formal benchmark requires exactly 3 expert samples")

    project_root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    output = Path(arguments.output).expanduser().resolve()
    scratch = Path(arguments.scratch_directory).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    models = {
        "deepseek_fp4": InstalledModel.open(arguments.deepseek_model),
        "qwen_mxfp4": InstalledModel.open(arguments.qwen_model),
    }
    rows: list[dict[str, Any]] = []

    for model_name, model in models.items():
        maximum_sample = max(FIXED_SIZES)
        required_experts = (maximum_sample + model.expert_blob_size - 1) // model.expert_blob_size
        last_start = model.expert_count - required_experts
        expert_ids = (0, last_start // 2, last_start)
        layer_path = model.root / "experts" / "layer_00.bin"
        descriptor = os.open(layer_path, os.O_RDONLY)
        try:
            for expert in expert_ids:
                for size_name, source_bytes, source_kind in _sample_specs(
                    model.expert_blob_size
                ):
                    offset = expert * model.expert_blob_size
                    source = os.pread(descriptor, source_bytes, offset)
                    if len(source) != source_bytes:
                        raise EOFError("installed expert layer ended early")
                    stem = f"{model_name}-{expert}-{size_name}"
                    source_path = scratch / f"{stem}.raw"
                    _write_bypass(source_path, source)
                    source_hash = hashlib.sha256(source).hexdigest()
                    baseline = {
                        metal: _run_worker(
                            script,
                            mode="baseline",
                            path=source_path,
                            output_bytes=source_bytes,
                            samples=arguments.samples,
                            metal=metal,
                        )
                        for metal in (False, True)
                    }
                    for codec_name in arguments.codecs:
                        codec = _AppleCompression(codec_name)
                        compressed = codec.encode(source)
                        compressed_path = scratch / f"{stem}.{codec_name}"
                        _write_bypass(compressed_path, compressed)
                        candidate = {
                            metal: _run_worker(
                                script,
                                mode="compressed",
                                path=compressed_path,
                                output_bytes=source_bytes,
                                samples=arguments.samples,
                                metal=metal,
                                codec=codec_name,
                            )
                            for metal in (False, True)
                        }
                        exact = all(
                            result["sha256"] == source_hash
                            for result in candidate.values()
                        )
                        row = {
                            "model": model_name,
                            "layer": 0,
                            "expert": expert,
                            "size_name": size_name,
                            "source_kind": source_kind,
                            "source_bytes": source_bytes,
                            "source_sha256": source_hash,
                            "codec": codec_name,
                            "compressed_bytes": len(compressed),
                            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                            "compression_ratio": len(compressed) / source_bytes,
                            "encode_scratch_bytes": codec.encode_scratch_bytes,
                            "decode_scratch_bytes": codec.decode_scratch_bytes,
                            "baseline_normal": baseline[False],
                            "baseline_metal": baseline[True],
                            "candidate_normal": candidate[False],
                            "candidate_metal": candidate[True],
                            "normal_comparison": _comparison(
                                baseline[False], candidate[False]
                            ),
                            "metal_comparison": _comparison(
                                baseline[True], candidate[True]
                            ),
                            "exact": exact,
                        }
                        rows.append(row)
                        print(
                            f"{model_name} expert={expert} {size_name} {codec_name}",
                            flush=True,
                        )
        finally:
            os.close(descriptor)

    gate = _gate(rows)
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "expert_blob_compression_microbenchmark",
        "formal_performance_result": True,
        "source": _source_state(project_root),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": _sysctl("hw.model", project_root),
            "chip": _sysctl("machdep.cpu.brand_string", project_root),
            "memory_bytes": int(_sysctl("hw.memsize", project_root) or 0),
            "python_version": platform.python_version(),
            "mlx_version": _package_version("mlx"),
            "mlx_lm_version": _package_version("mlx-lm"),
            "compression_library": "/usr/lib/libcompression.dylib",
        },
        "models": {
            name: _model_description(model) for name, model in models.items()
        },
        "method": {
            "codecs": arguments.codecs,
            "sizes_bytes": list(FIXED_SIZES),
            "includes_full_expert": True,
            "expert_samples_per_model": arguments.expert_samples,
            "timing_samples_per_case": arguments.samples,
            "file_cache_policy": "Darwin F_NOCACHE with read-ahead disabled",
            "baseline": "read canonical bytes into a preallocated destination",
            "candidate": "read compressed bytes then decode into a preallocated destination",
            "metal_contention": "repeated 2048x2048 FP16 MLX matrix multiplication",
            "peak_memory": "fresh worker process ru_maxrss",
        },
        "summary": _summary(rows),
        "gate": gate,
        "decision": {
            "phase_4": "pass" if gate["passed"] else "stop",
            "continue_to_phase_5": gate["passed"],
            "runtime_changed": False,
        },
        "limits": [
            "The fixed 4 MiB Qwen sample spans adjacent expert blobs because one Qwen expert blob is smaller than 4 MiB.",
            "The benchmark uses layer 0 and three expert positions per model.",
            "F_NOCACHE avoids retaining benchmark reads but does not prove physical device bytes for every call.",
            "The Metal load is a synthetic contention control, not model inference.",
        ],
        "rows": rows,
        "raw_directory": str(scratch),
    }
    with output.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
