"""Research-only Qwen Prefill expert grouping using existing MLX-LM helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sorted_expert_output(value, indices, batched, sorting_hint=True):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    source, selected, inverse = _gather_sort(mx.expand_dims(value, (-2, -3)), indices)
    projected = mx.gather_qmm(source, batched.gate_up, batched.gate_up_scales,
                              rhs_indices=selected, transpose=True, group_size=32,
                              bits=4, mode="mxfp4", sorted_indices=sorting_hint)
    gate, up = mx.split(projected, 2, axis=-1)
    output = mx.gather_qmm(nn.silu(gate)*up, batched.down, batched.down_scales,
                           rhs_indices=selected, transpose=True, group_size=32,
                           bits=4, mode="mxfp4", sorted_indices=sorting_hint)
    return _scatter_unsort(output, inverse, indices.shape).squeeze(-2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--no-sorting-hint", action="store_true")
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from deepseek_v4_ssd import qwen4_exp as qwen
    from deepseek_v4_ssd.cli import main as cli_main

    original = qwen.StreamingExperts.__call__
    modified_calls = 0

    def grouped(self, value, indices):
        nonlocal modified_calls
        batched = self.cache.current_batched(self.layer)
        if not isinstance(batched, qwen.QwenBatchedExperts) or indices.size < 64:
            return original(self, value, indices)
        output = sorted_expert_output(value, indices, batched, not args.no_sorting_hint)
        self.cache.record_gather_qmm(2)
        modified_calls += 1
        return output

    qwen.StreamingExperts.__call__ = grouped
    sys.argv = ["Qwen sorted-expert research", *(args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args)]
    status = "failed"
    try:
        cli_main()
        status = "completed"
    finally:
        qwen.StreamingExperts.__call__ = original
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(dict(status=status, modified_calls=modified_calls,
            sorting_hint=not args.no_sorting_hint,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(qwen.__file__).read_bytes()).hexdigest(),
            limits=["Grouping changes kernel selection and may change floating-point results.",
                    "Uses existing gather/scatter copies, not the fused ScatterMoE implementation.",
                    "Only the batched Prefill path changes; ordinary one-token expert loading is preserved."]
        ), indent=2)+"\n")


if __name__ == "__main__":
    main()
