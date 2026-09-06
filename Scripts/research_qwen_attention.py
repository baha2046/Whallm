"""Process-local Qwen component observer or QSA query-chunk experiment.

No installed dependency or runtime source is edited. Chunking is not FlashAttention.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path


def chunk_source(source: str, size: int) -> str:
    if size not in (4, 8, 16, 32):
        raise ValueError("pre-registered chunk sizes are 4, 8, 16, 32")
    tree = ast.parse(textwrap.dedent(source))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "query_chunk" for t in n.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant) or assignments[0].value.value != 4:
        raise ValueError("QSA source changed; audit before using this experiment")
    assignments[0].value = ast.Constant(value=size)
    return ast.unparse(ast.fix_missing_locations(tree))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--query-chunk", type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import mlx.core as mx
    from deepseek_v4_ssd import qwen4_exp as qwen
    from deepseek_v4_ssd.cli import main as cli_main

    calls, originals = [], []
    cls = qwen.QSAAttention
    source = chunk_source(inspect.getsource(cls._bounded_attention), args.query_chunk)
    if args.query_chunk != 4:
        namespace = {}
        exec(compile(source, "<research QSA chunk>", "exec"), vars(qwen), namespace)
        originals.append((cls, "_bounded_attention", cls._bounded_attention))
        cls._bounded_attention = namespace["_bounded_attention"]

    def install(cls, method, label, length_axis):
        original = getattr(cls, method)
        originals.append((cls, method, original))

        def observe(self, *values, **kw):
            tensors = [x for x in values if isinstance(x, mx.array)]
            mx.eval(*tensors)
            mx.synchronize()
            start = time.perf_counter()
            result = original(self, *values, **kw)
            mx.eval(result)
            mx.synchronize()
            calls.append(dict(component=label, length=values[0].shape[length_axis],
                              seconds=time.perf_counter()-start,
                              layer=getattr(self, "layer", None)))
            return result

        setattr(cls, method, observe)

    if args.observe:
        install(qwen.QSAAttention, "_bounded_attention", "qsa_core_nested", 2)
        install(qwen.QSAAttention, "__call__", "qsa_total", 1)
        install(qwen.GatedDeltaNet, "__call__", "gated_delta_total", 1)
        install(qwen.SparseMoE, "__call__", "moe_total", 1)
    sys.argv = ["qwen attention research", *(args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args)]
    status = "failed"
    try:
        cli_main()
        status = "completed"
    finally:
        for cls, name, method in reversed(originals):
            setattr(cls, name, method)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(dict(status=status, query_chunk=args.query_chunk,
            observe=args.observe, source_sha256=hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            candidate_source_sha256=hashlib.sha256(source.encode()).hexdigest(), calls=calls,
            limits=["Observer synchronization changes overlap; not ordinary component latency.",
                    "qsa_core_nested is included in qsa_total; never add them.",
                    "MoE Prefill timer excludes the full-layer weight read outside this call.",
                    "Cache-state work not needed for the returned output can occur outside a timer.",
                    "Changing query chunk is a work-granularity experiment, not a fused FlashAttention implementation."]
        ), indent=2)+"\n")


if __name__ == "__main__":
    main()
