"""Bounded QK/PV attribution after the direct candidate's exact-token rejection.

New hybrids keep one original grouped matmul and its four-query shape. This
tests whether either direct kernel can retain tensor-exact results and useful
speed by itself. No full-model retry of the rejected two-kernel candidate.
Mechanism: IO-aware attention, https://arxiv.org/abs/2205.14135.
"""
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from qwen_a_qsa_candidate import build_direct
from qwen_a_replay import benchmark


def build_hybrid(mode):
    candidate = build_direct()
    direct = candidate.__globals__["_direct"]
    kernels = inspect.getclosurevars(direct).nonlocals
    qk, pv = kernels["qk"], kernels["pv"]

    def hybrid(query, key, value, selected, valid):
        _, hq, ql, d = query.shape
        hk, kl = key.shape[1:3]
        ns = selected.shape[1]
        template = [("InT", query.dtype), ("HQ", hq), ("HK", hk), ("QL", ql),
                    ("KL", kl), ("NS", ns), ("D", d), ("R", hq // hk)]
        if mode == "direct-qk-original-pv":
            scores = qk(inputs=[query, key, selected, valid], template=template,
                        grid=(32, ns, ql * hk), threadgroup=(32, 4, 1),
                        output_shapes=[(ql, hq, ns)], output_dtypes=[query.dtype])[0]
            weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(query.dtype)
        else:
            parts = []
            for start in range(0, ql, 4):
                end = min(start + 4, ql)
                k = mx.take(key[0].transpose(1, 0, 2), selected[start:end], axis=0).transpose(0, 2, 1, 3)
                q = query[0, :, start:end].transpose(1, 0, 2).reshape(end-start, hk, hq//hk, d)
                scores = (q[..., None, :] @ k[:, :, None].swapaxes(-1, -2)).squeeze(-2) * (d**-0.5)
                scores = mx.where(valid[start:end, None, None, :], scores, mx.finfo(scores.dtype).min)
                parts.append(mx.softmax(scores.astype(mx.float32), axis=-1).astype(query.dtype).reshape(end-start, hq, ns))
            weights = mx.concatenate(parts, axis=0)
        if mode == "original-qk-direct-pv":
            return pv(inputs=[weights, value, selected], template=template,
                      grid=(32, d, ql * hk), threadgroup=(32, 4, 1),
                      output_shapes=[query.shape], output_dtypes=[query.dtype])[0]
        parts = []
        for start in range(0, ql, 4):
            end = min(start + 4, ql)
            v = mx.take(value[0].transpose(1, 0, 2), selected[start:end], axis=0).transpose(0, 2, 1, 3)
            w = weights[start:end].reshape(end-start, hk, hq//hk, ns)
            current = (w[..., None, :] @ v[:, :, None]).squeeze(-2).reshape(end-start, hq, d)
            parts.append(current.transpose(1, 0, 2)[None])
        return mx.concatenate(parts, axis=2)

    candidate.__globals__["_direct"] = hybrid
    return candidate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    result = {"kind": "component-attribution-not-performance-acceptance", "full_model_run": False,
              "predeclared_gate": "tensor-exact, finite, >=30% component reduction before any full-model screen",
              "rows": []}
    path = args.output / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    profile = json.loads((args.captures / "profile.json").read_text())
    sample = next(s for s in profile["samples"] if s["section"] == "qsa" and s["phase"] == "prefill")
    data = mx.load(str(args.captures / sample["path"]))
    mx.eval(data)
    norm = qwen.GroupRMSNorm(128, None, 1e-6)
    norm.weight = data["index_norm_weight"]
    proxy = SimpleNamespace(args=qwen.ModelArgs(), indexer=SimpleNamespace(k_layernorm=norm))
    inputs = {k: v for k, v in data.items() if k not in ("output", "index_norm_weight")}
    inputs["offset"] = sample["offset"]
    original = qwen.QSAAttention._bounded_attention
    reference = original(proxy, **inputs)
    mx.eval(reference)
    for mode in ("direct-qk-original-pv", "original-qk-direct-pv"):
        candidate = build_hybrid(mode)
        output = candidate(proxy, **inputs)
        mx.eval(output)
        row = {"mode": mode, "sample": sample, "tensor_exact": bool(mx.array_equal(reference, output).item()),
               "finite": bool(mx.isfinite(output).all().item()),
               "different_elements": int((reference != output).sum().item()),
               "max_abs_error": float(mx.abs(reference.astype(mx.float32)-output.astype(mx.float32)).max().item())}
        # Time only the exact hybrid; numerical failures stop immediately.
        if row["tensor_exact"] and row["finite"]:
            for name, function in (("control_1", original), ("candidate_1", candidate),
                                   ("candidate_2", candidate), ("control_2", original)):
                row[name] = benchmark(lambda: function(proxy, **inputs))
            a = (row["control_1"]["median_seconds"] + row["control_2"]["median_seconds"]) / 2
            b = (row["candidate_1"]["median_seconds"] + row["candidate_2"]["median_seconds"]) / 2
            row["component_reduction_fraction"] = 1 - b / a
        row["screen_pass"] = row["tensor_exact"] and row["finite"] and row.get("component_reduction_fraction", 0) >= .30
        result["rows"].append(row)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
