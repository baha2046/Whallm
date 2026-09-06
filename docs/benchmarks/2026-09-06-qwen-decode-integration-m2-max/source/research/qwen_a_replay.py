"""Replay captured Qwen inputs; ideal pooled-key reuse is a cost oracle only.

No production cache is changed. The oracle removes repeated pooling for one
fixed captured input, so its speed is an optimistic component limit, not runtime
performance. QSA micro-block basis: https://arxiv.org/abs/2608.30320.
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_update
from deepseek_v4_ssd import qwen4_exp as qwen


def benchmark(fn, repeats=9):
    for _ in range(3):
        result = fn()
        mx.eval(result)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        times.append(time.perf_counter() - start)
    return dict(median_seconds=statistics.median(times), samples_seconds=times)


def pool(proxy, raw):
    a = proxy.args
    blocks = raw.shape[1] // a.indexer_compress_ratio
    pooled = raw[:, :blocks*a.indexer_compress_ratio].reshape(
        1, blocks, a.indexer_compress_ratio, a.indexer_head_dim
    ).astype(mx.float32).mean(axis=2).astype(raw.dtype)
    pooled = proxy.indexer.k_layernorm(pooled)
    if blocks:
        pooled = qwen._apply_partial_rope(pooled,
            (mx.arange(blocks)*a.indexer_compress_ratio)[None],
            int(a.head_dim*a.partial_rotary_factor), a.rope_theta)
    return pooled


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    profile = json.loads((args.captures / "profile.json").read_text())
    rows = []
    for sample in profile["samples"]:
        data = mx.load(str(args.captures / sample["path"]))
        mx.eval(data)
        if sample["section"] == "gdn":
            inputs = {k:v for k,v in data.items() if k not in ("output","final_state")}
            row = dict(sample=sample, baseline=benchmark(lambda:gated_delta_update(**inputs)))
        else:
            norm = qwen.GroupRMSNorm(128, None, 1e-6)
            norm.weight = data["index_norm_weight"]
            proxy = SimpleNamespace(args=qwen.ModelArgs(), indexer=SimpleNamespace(k_layernorm=norm))
            proxy._probe_pooled = pool(proxy, data["raw_index_keys"])
            mx.eval(proxy._probe_pooled)
            # Reuse the exact attention body; only replace preprocessing with a
            # precomputed result for this single fixed-input component oracle.
            source = textwrap.dedent(inspect.getsource(qwen.QSAAttention._bounded_attention))
            start = source.index("    pooled = raw_index_keys")
            end = source.index("    outputs = []", start)
            source = source[:start] + "    pooled = self._probe_pooled\n" + source[end:]
            namespace = dict(vars(qwen))
            exec(compile(source, "<pooled-key-cost-oracle>", "exec"), namespace)
            oracle = namespace["_bounded_attention"]
            inputs = {k:v for k,v in data.items() if k not in ("output","index_norm_weight")}
            inputs["offset"] = sample["offset"]
            output = oracle(proxy, **inputs)
            baseline_output = qwen.QSAAttention._bounded_attention(proxy, **inputs)
            equal = bool(mx.array_equal(output, baseline_output).item())
            saved_equal = bool(mx.array_equal(baseline_output, data["output"]).item())
            if not equal or not saved_equal:
                raise ValueError("fixed-input QSA output parity failed")
            row = dict(sample=sample, pooled=benchmark(lambda:pool(proxy,data["raw_index_keys"])), output_exact=equal, capture_output_exact=saved_equal)
            row["control_1"] = benchmark(lambda:qwen.QSAAttention._bounded_attention(proxy, **inputs))
            row["oracle_1"] = benchmark(lambda:oracle(proxy, **inputs))
            row["oracle_2"] = benchmark(lambda:oracle(proxy, **inputs))
            row["control_2"] = benchmark(lambda:qwen.QSAAttention._bounded_attention(proxy, **inputs))
            row["ideal_saved_seconds"] = statistics.median(row[k]["median_seconds"] for k in ("control_1","control_2"))-statistics.median(row[k]["median_seconds"] for k in ("oracle_1","oracle_2"))
        rows.append(row)
        print(sample["path"], json.dumps({k:v for k,v in row.items() if k not in ("sample",)}, default=str), flush=True)
    args.output.write_text(json.dumps(dict(evidence_kind="fixed-input component timings and ideal cache oracle", formal_performance_result=False, rows=rows),indent=2)+"\n")


if __name__ == "__main__":
    main()
