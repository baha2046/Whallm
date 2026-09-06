"""Evaluate the pre-registered causal cache on current Qwen observer traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from simulate_qwen_lfu import simulate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    if baseline["status"] != "completed":
        raise ValueError("baseline and observer parity checks must finish first")
    cases = []
    for run in baseline["runs"]:
        if run["kind"] != "observer":
            continue
        path = Path(run["trace_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["trace_sha256"]:
            raise ValueError("trace changed since the baseline")
        trace = json.loads(path.read_text())
        current = simulate(trace, 4096, "current")
        recorded = sum(v for layer in trace["decode_misses"] for token in layer for v in token)
        if recorded != current["misses"]:
            raise ValueError(f'{run["case"]}: current replay {current["misses"]} != recorded {recorded}')
        candidate = simulate(trace, 4096, "prefill_guided_24")
        fraction = 1 - candidate["misses"] / current["misses"] if current["misses"] else 0
        cases.append({"case": run["case"], "trace_sha256": run["trace_sha256"],
                      "recorded_misses": recorded, "current": current, "candidate": candidate,
                      "miss_reduction_fraction": fraction,
                      "logical_byte_reduction": (current["misses"] - candidate["misses"]) * trace["expert_blob_size"]})
        print(run["case"], f"miss change {-fraction:.2%}", flush=True)
    if len(cases) != 5:
        raise ValueError("expected five observer workloads")
    median = statistics.median(c["miss_reduction_fraction"] for c in cases if not c["case"].startswith("repeated-"))
    passed = median >= .05 and min(c["miss_reduction_fraction"] for c in cases) >= -.02
    result = {"schema_version": 1, "evidence_kind": "current_route_offline_policy_screen",
              "formal_performance_result": False, "source": baseline["source"],
              "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "simulator_sha256": hashlib.sha256(Path(__file__).with_name("simulate_qwen_lfu.py").read_bytes()).hexdigest(),
              "slots": 4096, "cases": cases,
              "gate": {"non_repeated_median_miss_reduction": median, "passed": passed,
                       "minimum_median": .05, "maximum_per_workload_regression": .02},
              "decision": "eligible_for_runtime_timing" if passed else "stop_this_policy_at_4096_slots",
              "limits": ["No candidate runtime speed measurement.",
                         "Synthetic prompts and 256 output tokens; not long-session validation.",
                         "Candidate uses Prefill and observed Decode only; no future route oracle.",
                         "Miss savings cannot be read as latency savings."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
