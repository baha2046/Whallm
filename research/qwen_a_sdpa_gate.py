"""Real-input numerical screening and paired component timing for QSA SDPA."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from qwen_a_qsa_candidate import build_sdpa, build_direct
from qwen_a_replay import benchmark


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("sdpa","direct"), default="sdpa")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    candidate = build_sdpa() if args.method == "sdpa" else build_direct()
    profile = json.loads((args.captures / "profile.json").read_text())
    rows = []
    for sample in profile["samples"]:
        if sample["section"] != "qsa":
            continue
        data = mx.load(str(args.captures / sample["path"]))
        mx.eval(data)
        norm = qwen.GroupRMSNorm(128, None, 1e-6)
        norm.weight = data["index_norm_weight"]
        proxy = SimpleNamespace(args=qwen.ModelArgs(), indexer=SimpleNamespace(k_layernorm=norm))
        inputs = {k:v for k,v in data.items() if k not in ("output","index_norm_weight")}
        inputs["offset"] = sample["offset"]
        control = qwen.QSAAttention._bounded_attention(proxy, **inputs)
        result = candidate(proxy, **inputs)
        mx.eval(control,result)
        delta = mx.abs(control.astype(mx.float32)-result.astype(mx.float32))
        bound = 0.03125 + 0.01*mx.abs(control.astype(mx.float32))
        row = dict(sample=sample, max_abs_error=delta.max().item(), mean_abs_error=delta.mean().item(),
            output_exact=mx.array_equal(result,control).item(),
            numerical_gate=bool((mx.isfinite(result).all() & (delta<=bound).all()).item()),
            violating_elements=int((delta>bound).sum().item()))
        if row["numerical_gate"]:
            for name,fn in [("control_1",qwen.QSAAttention._bounded_attention),("candidate_1",candidate),("candidate_2",candidate),("control_2",qwen.QSAAttention._bounded_attention)]:
                row[name]=benchmark(lambda:fn(proxy,**inputs))
        rows.append(row)
        print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(dict(evidence_kind="QSA component screening", method=args.method, formal_performance_result=False,
        tolerance=dict(atol=0.03125,rtol=0.01),rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
