"""Real routed-expert replay: exact tensors first, ABBA time second."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.expert_cache import ResidentExperts, QwenExpertWeights, QwenBatchedExperts
from qwen_a_replay import benchmark
from qwen_decode_candidates import single_row, grouped_resident


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    result = {"kind": "resident component screen; grouped oracle excludes packing", "formal_performance_result": False,
              "predeclared_gate": "finite tensor-exact on all three captures; no full-model retry on mandatory failure", "rows": []}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for sample in json.loads((args.captures / "profile.json").read_text())["samples"]:
        data = mx.load(str(args.captures / sample["path"]))
        experts = sample["experts"]
        individuals = tuple(QwenExpertWeights(**{name: data[f"expert_{e}_{name}"] for name in
                         ("gate_up", "gate_up_scales", "down", "down_scales")}) for e in experts)
        resident = ResidentExperts(individuals, {e: i for i, e in enumerate(experts)})
        cache = SimpleNamespace(current_batched=lambda layer: None, get_many=lambda layer, ids: resident)
        proxy = qwen.StreamingExperts(sample["layer"], cache)
        value, indices = data["value"], data["indices"]
        baseline = lambda: qwen.StreamingExperts.__call__(proxy, value, indices)
        grouped = QwenBatchedExperts(**{name: mx.stack([getattr(w, name) for w in individuals]) for name in
                                     ("gate_up", "gate_up_scales", "down", "down_scales")})
        mapped = mx.array([resident.slots[int(e)] for e in indices.reshape(-1).tolist()]).reshape(indices.shape)
        mx.eval(value, indices, mapped, *vars(grouped).values())
        reference = baseline()
        mx.eval(reference)
        if not mx.array_equal(reference, data["output"]).item():
            raise RuntimeError("capture replay differs from original output")
        for name, fn in [("single-row", lambda: single_row(proxy, value, indices)),
                         ("grouped-resident-oracle", lambda: grouped_resident(value, mapped, grouped))]:
            actual = fn()
            mx.eval(actual)
            row = dict(sample=sample, candidate=name, finite=bool(mx.isfinite(actual).all().item()),
                       tensor_exact=bool(mx.array_equal(reference, actual).item()),
                       different_elements=int((reference != actual).sum().item()),
                       max_abs_error=float(mx.abs(reference.astype(mx.float32)-actual.astype(mx.float32)).max().item()))
            if row["finite"] and row["tensor_exact"]:
                for label, func in [("control_1", baseline), ("candidate_1", fn), ("candidate_2", fn), ("control_2", baseline)]:
                    row[label] = benchmark(func)
                a = (row["control_1"]["median_seconds"] + row["control_2"]["median_seconds"]) / 2
                b = (row["candidate_1"]["median_seconds"] + row["candidate_2"]["median_seconds"]) / 2
                row["saved_seconds"] = a-b
                row["reduction_fraction"] = 1-b/a
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
