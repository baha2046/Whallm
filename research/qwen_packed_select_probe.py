"""Same-packed QSA component experiment; production runtime is never patched.

Run with PYTHONPATH=runtime .venv/bin/python research/qwen_packed_select_probe.py
--output scratch/<new-directory>. Synthetic inputs use the installed architecture.
"""
import argparse
import hashlib
import inspect
import json
import platform
import statistics
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import QuantizedKVCache
from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
from qwen_a_replay import benchmark


def selected_rows(packed, selected, bits):
    # Keep the final dimension contiguous and move all three components together.
    parts = [mx.take(a[0].transpose(1, 0, 2), selected, axis=0) for a in packed]
    return mx.dequantize(*parts, group_size=32, bits=bits)


def make_attention():
    source = textwrap.dedent(inspect.getsource(qwen.QSAAttention._bounded_attention))
    for name in ("key", "value"):
        old = f"mx.take({name}[0].transpose(1, 0, 2), selected, axis=0)"
        assert source.count(old) == 1
        source = source.replace(old, f"_select(self._packed_{name}, selected, self._bits)")
    namespace = dict(vars(qwen), _select=selected_rows)
    exec(compile(source, "<same-packed-select-first>", "exec"), namespace)
    return namespace["_bounded_attention"], source


def peak(fn):
    mx.synchronize()
    mx.clear_cache()
    initial = mx.get_active_memory()
    mx.reset_peak_memory()
    result = fn()
    mx.eval(result)
    return {"initial_active_bytes": initial,
            "peak_extra_bytes": mx.get_peak_memory() - initial}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    candidate, source = make_attention()
    (args.output / "candidate.py.txt").write_text(source)
    result = {"evidence_kind": "synthetic_same_packed_QSA_component", "formal_performance_result": False,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "platform": platform.platform(), "device": mx.device_info(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "attention_source_sha256": hashlib.sha256(inspect.getsource(qwen.QSAAttention._bounded_attention).encode()).hexdigest(),
              "limits": ["Synthetic tensors, not full-model generation or BF16 quality validation.",
                         "Quantization/update cost excluded equally; full-history indexer included in QSA timings.",
                         "Normal MLX lazy evaluation; no synchronization inserted before gather.",
                         "Packed main K/V only; raw index history remains the same in both arms."],
              "rows": []}
    config = qwen.ModelArgs()
    proxy = SimpleNamespace(args=config, indexer=SimpleNamespace(k_layernorm=qwen.GroupRMSNorm(128, None, 1e-6)))
    for bits in (4, 8):
        for length, queries in [(2047,1),(2048,1),(2049,1),(2050,1),(2051,1),(2052,1),
                                (8192,1),(16385,1),(32768,1),(8192,4),(8192,16),(8192,128)]:
            mx.random.seed(915 + length + queries)
            k = mx.random.normal((1,2,length,256)).astype(mx.bfloat16)
            v = mx.random.normal(k.shape).astype(mx.bfloat16)
            cache = QSAQuantizedCache(bits, 256)
            packed_k, packed_v = QuantizedKVCache.update_and_fetch(cache, k, v)
            query = mx.random.normal((1,24,queries,256)).astype(mx.bfloat16)
            iq = mx.random.normal((1,queries,4,128)).astype(mx.bfloat16)
            raw = mx.random.normal((1,length,128)).astype(mx.bfloat16)
            mx.eval(packed_k, packed_v, query, iq, raw, proxy.indexer.k_layernorm.weight)
            del k, v, cache
            proxy._packed_key, proxy._packed_value, proxy._bits = packed_k, packed_v, bits
            placeholder = SimpleNamespace(shape=(1,2,length,256))
            offset = length - queries

            def baseline():
                key, value = [mx.dequantize(*p, group_size=32, bits=bits) for p in (packed_k,packed_v)]
                return qwen.QSAAttention._bounded_attention(proxy, query, key, value, iq, raw, offset)

            def alternative():
                return candidate(proxy, query, placeholder, placeholder, iq, raw, offset)

            # Explicit duplicate, clipped-tail and reversed-row test, separate from timing.
            ids = mx.array([[length-1,0,length-1,1, min(2048,length-1)]], mx.int32)
            exact_rows = []
            for packed in (packed_k,packed_v):
                full = mx.dequantize(*packed, group_size=32, bits=bits)
                expected = mx.take(full[0].transpose(1,0,2), ids, axis=0)
                actual = selected_rows(packed, ids, bits)
                exact_rows.append(bool(mx.array_equal(expected, actual).item()))
            reference, actual = baseline(), alternative()
            mx.eval(reference, actual)
            exact = bool(mx.array_equal(reference, actual).item())
            row = {"bits":bits,"history":length,"queries":queries,
                   "selected_rows_exact":all(exact_rows), "attention_exact":exact,
                   "finite":bool(mx.isfinite(actual).all().item()),
                   "output_sha256":hashlib.sha256(memoryview(actual.astype(mx.float32)).tobytes()).hexdigest(),
                   "full_dequant_kv_bytes":2*2*length*256*2,
                   "selected_row_occurrences":queries*(length if length<=2048 else 2052)}
            if not all(exact_rows) or not exact or not row["finite"]:
                result["rows"].append(row)
                (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
                raise RuntimeError(f"same-packed exactness gate failed: {row}")
            if length == 16385:
                mx.export_to_dot(str(args.output/f"baseline-{bits}.dot"), baseline())
                mx.export_to_dot(str(args.output/f"candidate-{bits}.dot"), alternative())
            for label, fn in (("control_1",baseline),("candidate_1",alternative),
                              ("candidate_2",alternative),("control_2",baseline)):
                row[label] = benchmark(fn, repeats=15)
            a=statistics.mean(row[x]["median_seconds"] for x in ("control_1","control_2"))
            b=statistics.mean(row[x]["median_seconds"] for x in ("candidate_1","candidate_2"))
            row.update(control_ms=a*1000,candidate_ms=b*1000,reduction_percent=100*(1-b/a),
                       control_memory=peak(baseline),candidate_memory=peak(alternative))
            result["rows"].append(row)
            (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps({k:row[k] for k in ("bits","history","queries","attention_exact","control_ms","candidate_ms","reduction_percent","control_memory","candidate_memory")}),flush=True)


if __name__ == "__main__":
    main()
