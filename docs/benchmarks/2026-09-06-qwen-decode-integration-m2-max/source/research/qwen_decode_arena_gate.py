"""Actual 1,152-slot arena gate, including sparse indices and slot overwrite."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from deepseek_v4_ssd.manifest import InstalledModel
from qwen_decode_arena import ArenaSlotPool
from qwen_decode_candidates import grouped_resident
from qwen_a_replay import benchmark


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    model = InstalledModel.open(args.model)
    data = mx.load(str(args.capture))
    mx.eval(data)
    experts = sorted(int(k.split("_")[1]) for k in data if k.endswith("_gate_up"))
    slots = [0, 7, 63, 127, 255, 511, 767, 1023, 1100, 1151]
    assert len(experts) == len(slots)
    names = {"gate_up.weight": "gate_up", "gate_up.scale": "gate_up_scales", "down.weight": "down", "down.scale": "down_scales"}
    blobs = []
    for expert in experts:
        blob = bytearray(model.expert_blob_size)
        for region in model.expert_regions:
            payload = memoryview(data[f"expert_{expert}_{names[region.name]}"]).cast("B")
            assert len(payload) == region.length
            blob[region.offset:region.offset + region.length] = payload
        blobs.append(bytes(blob))
    pool = ArenaSlotPool(model, 1152)
    pool.store(slots, blobs)
    mapping = dict(zip(experts, slots))
    indices = mx.array([mapping[int(e)] for e in data["indices"].reshape(-1).tolist()], dtype=mx.uint32).reshape(data["indices"].shape)
    fn = lambda: grouped_resident(data["value"], indices, pool.grouped)
    actual = fn()
    mx.eval(actual)
    result = {"kind": "strided slot arena component gate, not full-model acceptance", "slots": slots,
              "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              "tensor_exact": bool(mx.array_equal(actual, data["output"]).item()),
              "finite": bool(mx.isfinite(actual).all().item()),
              "max_abs_error": float(mx.abs(actual.astype(mx.float32)-data["output"].astype(mx.float32)).max().item())}
    if result["tensor_exact"] and result["finite"]:
        result["timing"] = benchmark(fn)
        # Overwrite one used physical slot, verify fresh views see new bytes,
        # then restore and require exact original output again.
        pool.store([slots[0]], [blobs[1]])
        changed = fn()
        mx.eval(changed)
        result["overwrite_visible"] = not bool(mx.array_equal(actual, changed).item())
        pool.store([slots[0]], [blobs[0]])
        restored = fn()
        mx.eval(restored)
        result["restore_exact"] = bool(mx.array_equal(actual, restored).item())
    result["peak_memory_bytes"] = mx.get_peak_memory()
    result["screen_pass"] = all(result.get(k, False) for k in ("tensor_exact", "finite", "overwrite_visible", "restore_exact"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
