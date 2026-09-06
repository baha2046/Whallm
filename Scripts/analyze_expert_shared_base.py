"""Fixed-anchor low-rank storage/error screen on installed Qwen MXFP4 weights.

This is an optimistic weight-space diagnostic, not a LorExperts reproduction,
activation-quality test, compressed kernel, or speed measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def spectrum_errors(matrix: np.ndarray, singular_values: np.ndarray, ranks: list[int]) -> dict:
    energy = np.square(singular_values.astype(np.float64))
    remaining = np.maximum(0, energy.sum() - np.r_[0., np.cumsum(energy)])
    norm2 = float(np.square(matrix.astype(np.float64)).sum())
    return {str(rank): float(np.sqrt(remaining[min(rank, len(energy))] / norm2))
            for rank in ranks}


def storage_bytes(rank: int, group_size: int, base_bytes: int) -> float:
    # FP16 factors for gate_up (1280 x 2560) and down (2560 x 640).
    return base_bytes / group_size + 2 * rank * (1280 + 2560 + 2560 + 640)


def read_expert(model, layer: int, expert: int) -> tuple[dict, str]:
    import mlx.core as mx

    path = model.root / f"experts/layer_{layer:02d}.bin"
    with path.open("rb") as file:
        file.seek(expert * model.expert_blob_size)
        blob = file.read(model.expert_blob_size)
    if len(blob) != model.expert_blob_size:
        raise ValueError("truncated expert blob")
    arrays = {}
    for region in model.expert_regions:
        dtype = np.uint32 if region.dtype == "U32" else np.uint8
        values = np.frombuffer(blob, dtype=dtype, count=region.length // np.dtype(dtype).itemsize,
                               offset=region.offset).reshape(region.shape)
        arrays[region.name] = mx.array(values)
    result = {}
    for name in ("gate_up", "down"):
        dense = mx.dequantize(arrays[name + ".weight"], arrays[name + ".scale"],
                              group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
        mx.eval(dense)
        result[name] = np.asarray(dense).copy()
    return result, hashlib.sha256(blob).hexdigest()


def main() -> None:
    from deepseek_v4_ssd.manifest import InstalledModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = InstalledModel.open(args.model.expanduser().resolve())
    if not model.is_qwen:
        raise ValueError("this screen is only defined for Qwen MXFP4 shapes")
    trace = json.loads(args.trace.read_text())
    ranks = [0, 16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 1024, 1280]
    result = {"schema_version": 1, "evidence_kind": "installed_weight_fixed_anchor_spectral_screen",
              "formal_performance_result": False,
              "manifest_sha256": hashlib.sha256((model.root / "manifest.json").read_bytes()).hexdigest(),
              "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "numpy_version": np.__version__,
              "model_id": model.model_id, "checkpoint_revision": model.revision,
              "selection": "Layers 0,23,47; anchor is most frequent expert in first 128 code Decode steps; member co-occurs most often with anchor. Frozen before reading weights.",
              "ranks": ranks, "cases": [],
              "storage_assumptions": {"group_size": 8, "factor_dtype": "FP16", "native_blob_bytes": model.expert_blob_size,
                                      "native_anchor_amortized": True, "metadata_bytes": "excluded; optimistic lower bound"},
              "limits": ["One pair in each of three layers, chosen from a single synthetic code workload.",
                         "Fixed unaligned anchor; no permutation alignment, clustering search, activation whitening or fine-tuning.",
                         "SVD error is the best unweighted rank-r error for that fixed residual; factor rounding excluded.",
                         "Group-of-eight storage is a hypothetical amortization, not eight measured members.",
                         "No capability, generated-output, SSD latency or compressed-kernel speed result.",
                         "No rejection of the full LorExperts method or of activation-aware compression."]}
    for layer in (0, 23, 47):
        events = trace["decode_routes"][layer]
        if len(events) < 128:
            raise ValueError("need at least 128 calibration Decode steps")
        counts = Counter(e for event in events[:128] for e in event)
        anchor = min(counts, key=lambda e: (-counts[e], e))
        joint = Counter(e for event in events[:128] if anchor in event for e in event if e != anchor)
        member = min(joint, key=lambda e: (-joint[e], e))
        print(f"LAYER {layer} anchor {anchor} member {member}", flush=True)
        base, base_hash = read_expert(model, layer, anchor)
        target, target_hash = read_expert(model, layer, member)
        case = {"layer": layer, "anchor": anchor, "member": member,
                "anchor_blob_sha256": base_hash, "member_blob_sha256": target_hash,
                "calibration_cooccurrences": joint[member],
                "heldout_same_code_cooccurrences": sum(anchor in e and member in e for e in events[128:]),
                "projections": {}}
        for name in ("gate_up", "down"):
            matrix = target[name]
            delta = matrix - base[name]
            print(f"  SVD {name} {matrix.shape}", flush=True)
            singular = np.linalg.svd(delta, compute_uv=False)
            own_singular = np.linalg.svd(matrix, compute_uv=False)
            valid_ranks = [r for r in ranks if r <= min(matrix.shape)]
            case["projections"][name] = {
                "shape": list(matrix.shape),
                "weight_cosine": float(np.vdot(matrix, base[name]) / (np.linalg.norm(matrix) * np.linalg.norm(base[name]))),
                "residual_norm_over_weight_norm": float(np.linalg.norm(delta) / np.linalg.norm(matrix)),
                "best_fixed_anchor_relative_frobenius_error": spectrum_errors(matrix, singular, valid_ranks),
                "best_zero_anchor_relative_frobenius_error": spectrum_errors(matrix, own_singular, valid_ranks),
            }
        result["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["storage_curve"] = [{"rank": r,
                                "native_anchor_average_bytes": storage_bytes(r, 8, model.expert_blob_size),
                                "fp16_anchor_average_bytes": storage_bytes(r, 8, 2 * (1280*2560 + 2560*640)),
                                "native_anchor_reduction_fraction": 1 - storage_bytes(r, 8, model.expert_blob_size) / model.expert_blob_size}
                               for r in ranks if r <= 640]
    result["status"] = "completed"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
