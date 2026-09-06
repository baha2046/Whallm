"""Compare sorted and unsorted Qwen Prefill outputs while returning the control."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import mlx.core as mx
    from deepseek_v4_ssd import qwen4_exp as qwen
    from deepseek_v4_ssd.cli import main as cli_main
    from research_qwen_sorted_experts import sorted_expert_output

    original = qwen.StreamingExperts.__call__
    cases = []
    seen = set()

    def compare(self, value, indices):
        reference = original(self, value, indices)
        if self.layer not in (0, 23, 47) or self.layer in seen or indices.size < 64:
            return reference
        batched = self.cache.current_batched(self.layer)
        if not isinstance(batched, qwen.QwenBatchedExperts):
            return reference
        seen.add(self.layer)
        mx.eval(reference)
        flat = indices.flatten()
        order = mx.argsort(flat)
        inverse = mx.argsort(order)
        assert bool(mx.all(flat[order][inverse] == flat).item())
        grouped_rows = value.reshape(-1, value.shape[-1])[order // indices.shape[-1]]
        expected_rows = mx.repeat(value.reshape(-1, value.shape[-1]), indices.shape[-1], axis=0)
        assert bool(mx.all(grouped_rows[inverse] == expected_rows).item())
        case = dict(layer=self.layer, shape=list(reference.shape), assignment_roundtrip_exact=True, variants=[])
        for hint in (True, False):
            candidate = sorted_expert_output(value, indices, batched, hint)
            mx.eval(candidate)
            delta = candidate.astype(mx.float32)-reference.astype(mx.float32)
            norm2 = mx.sum(mx.square(reference.astype(mx.float32)))
            case["variants"].append(dict(sorting_hint=hint,
                finite=bool(mx.all(mx.isfinite(candidate)).item()),
                exact_fraction=float(mx.mean((reference==candidate).astype(mx.float32)).item()),
                max_absolute_error=float(mx.max(mx.abs(delta)).item()),
                relative_l2_error=float(mx.sqrt(mx.sum(mx.square(delta))/norm2).item())))
        cases.append(case)
        print(f"Checked expert layer {self.layer}", file=sys.stderr, flush=True)
        return reference

    qwen.StreamingExperts.__call__ = compare
    sys.argv = ["grouping diagnosis", *(args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args)]
    status = "failed"
    try:
        cli_main()
        assert seen == {0, 23, 47}
        status = "completed"
    finally:
        qwen.StreamingExperts.__call__ = original
        args.record.write_text(json.dumps(dict(status=status, cases=cases,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            helper_sha256=hashlib.sha256(Path(__file__).with_name("research_qwen_sorted_experts.py").read_bytes()).hexdigest(),
            limits=["Three first Prefill chunks from one code input; not capability evaluation.",
                    "Returns only control outputs; timing is invalidated by duplicate compute.",
                    "Small output errors alone do not prove a grouping implementation correct for every shape."]
        ), indent=2)+"\n")


if __name__ == "__main__":
    main()
