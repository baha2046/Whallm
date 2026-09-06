"""Sequential Decode ABBA stop gate with exact tokens and resource limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--order", choices=("ABBA", "BAAB"), default="ABBA")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "order": args.order, "runs": [], "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
               "gates": {"exact_tokens": True, "decode_improvement_min": .05, "ttft_regression_max": .05,
                         "p95_regression_max": .10, "peak_memory_regression_max": .15, "expert_bytes_regression_max": .05}}
    output = args.output / "summary.json"
    def save():
        output.write_text(json.dumps(summary, indent=2) + "\n")
    save()
    reference = None
    for ordinal, letter in enumerate(args.order):
        mode = "control" if letter == "A" else "arena"
        name = f"{ordinal+1}-{mode}"
        metrics, status = args.output / f"{name}.json", args.output / f"{name}-status.json"
        cmd = ["/usr/bin/time", "-l", sys.executable, "research/qwen_decode_run.py", "--mode", mode, "--status", str(status), "--",
               "--model", args.model, "--prompt-file", str(args.prompt), "--max-tokens", str(args.tokens),
               "--temperature", "0", "--top-p", "1", "--top-k", "0", "--slots", "1152",
               "--no-persistent-prompt-cache", "--metrics-json", str(metrics)]
        print(f"START {name}", flush=True)
        start = time.monotonic()
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "runtime:research"}, stdout=log, stderr=subprocess.STDOUT)
        row = dict(mode=mode, command=cmd, metrics=metrics.name, status=status.name, exit_code=process.returncode,
                   client_wall_seconds=time.monotonic()-start)
        summary["runs"].append(row)
        if process.returncode:
            summary["status"] = "failed-process"; save(); return
        data, observed = json.loads(metrics.read_text()), json.loads(status.read_text())
        if not observed["ane"] or any(not a["active"] or a["fallbacks"] for a in observed["ane"]):
            summary["status"] = "failed-ane-contract"; save(); return
        if mode == "arena" and not observed["arena_state"]["grouped_calls"]:
            summary["status"] = "failed-candidate-not-exercised"; save(); return
        if reference is None:
            reference = data
        row["prompt_exact"] = reference["prompt_token_sha256"] == data["prompt_token_sha256"]
        row["output_exact"] = reference["generated_token_ids"] == data["generated_token_ids"]
        if not row["prompt_exact"] or not row["output_exact"]:
            row["first_divergence_zero_based"] = next((i for i, pair in enumerate(zip(reference["generated_token_ids"], data["generated_token_ids"])) if pair[0] != pair[1]), None)
            summary["status"] = "rejected-exact-token-gate"; save(); return
        print(f"END {name}: {data['decode_tokens_per_second']:.3f} tok/s; exact", flush=True)
        save()
    metrics_by_mode = {m: [json.loads((args.output / r["metrics"]).read_text()) for r in summary["runs"] if r["mode"] == m] for m in ("control", "arena")}
    keys = ("decode_tokens_per_second", "time_to_first_token_seconds", "request_seconds", "decode_latency_p95_seconds", "peak_memory_bytes", "expert_bytes_read")
    medians = {mode: {k: statistics.median(r[k] for r in runs) for k in keys} for mode, runs in metrics_by_mode.items()}
    changes = {k: medians["arena"][k]/medians["control"][k]-1 for k in keys}
    summary["medians"], summary["fraction_changes"] = medians, changes
    failures = [k for k, cap in (("time_to_first_token_seconds", .05), ("decode_latency_p95_seconds", .10), ("peak_memory_bytes", .15), ("expert_bytes_read", .05)) if changes[k] > cap]
    if failures:
        summary["status"] = "rejected-regression-gate"
    elif changes["decode_tokens_per_second"] < .05:
        summary["status"] = "inconclusive-below-5-percent-stop"
    else:
        summary["status"] = "passed-first-wave-not-adoption"
    summary["failed_metrics"] = failures
    save()
    print(json.dumps({k:summary[k] for k in ("status", "fraction_changes", "failed_metrics")}), flush=True)


if __name__ == "__main__":
    main()
