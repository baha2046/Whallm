"""Numerical/memory/time gate for Grouped-KV retile, no production mutation."""
import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from qwen_a_qsa_candidate import build_grouped_tile
from qwen_a_replay import benchmark


def measure(fn):
    mx.clear_cache()
    mx.reset_peak_memory()
    result = benchmark(fn)
    result["peak_memory_bytes"] = mx.get_peak_memory()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    profile=json.loads((args.captures/"profile.json").read_text())
    sample=next(s for s in profile["samples"] if s["section"]=="qsa" and s["phase"]=="prefill")
    data=mx.load(str(args.captures/sample["path"]))
    mx.eval(data)
    norm=qwen.GroupRMSNorm(128,None,1e-6)
    norm.weight=data["index_norm_weight"]
    proxy=SimpleNamespace(args=qwen.ModelArgs(),indexer=SimpleNamespace(k_layernorm=norm))
    inputs={k:v for k,v in data.items() if k not in ("output","index_norm_weight")}
    inputs["offset"]=sample["offset"]
    reference=qwen.QSAAttention._bounded_attention(proxy,**inputs)
    mx.eval(reference)
    rows=[]
    for tile in (16,64):
        candidate=build_grouped_tile(tile)
        result=candidate(proxy,**inputs)
        mx.eval(result)
        error=mx.abs(result.astype(mx.float32)-reference.astype(mx.float32))
        row=dict(tile=tile,output_exact=mx.array_equal(result,reference).item(),max_abs_error=error.max().item(),
            numerical_gate=bool((mx.isfinite(result).all() & (error<=0.03125+0.01*mx.abs(reference.astype(mx.float32))).all()).item()))
        del result,error
        if row["numerical_gate"]:
            for name,fn in [("control_1",qwen.QSAAttention._bounded_attention),("candidate_1",candidate),("candidate_2",candidate),("control_2",qwen.QSAAttention._bounded_attention)]:
                row[name]=measure(lambda:fn(proxy,**inputs))
        rows.append(row)
        print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(dict(evidence_kind="post-Grouped-KV tile component gate",formal_performance_result=False,sample=sample,rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
