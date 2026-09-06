"""Cold minimal-request gate: do not hide allocation growth behind a full cache."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "prompt": "Hi", "slots": 4096, "max_tokens": args.tokens,
               "kind": "cold allocation edge gate, not Decode throughput measurement",
               "gates": {"output_exact": True, "peak_memory_regression_max": .15, "expert_bytes_regression_max": .05}, "runs": []}
    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    for mode in ("control", "arena"):
        metrics = args.output / f"{mode}.json"
        cmd = ["/usr/bin/time", "-l", sys.executable, "research/qwen_decode_cross_arena_cold_run.py", "--mode", "control", "--status", str(args.output/f"{mode}-status.json"), "--",
               "--model", args.model, "--prompt", "Hi", "--max-tokens", str(args.tokens), "--slots", "4096", "--temperature", "0", "--top-p", "1", "--top-k", "0",
               "--no-persistent-prompt-cache", "--metrics-json", str(metrics)]
        if mode == "arena":
            cmd.append("--qwen-grouped-decode")
        print(f"START cold {mode}", flush=True)
        with (args.output/f"{mode}.log").open("w") as log:
            process = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "runtime:research"}, stdout=log, stderr=subprocess.STDOUT)
        summary["runs"].append({"mode": mode, "command": cmd, "exit_code": process.returncode})
        if process.returncode:
            summary["status"] = "failed-process"; save(); return
        save()
    a, b = [json.loads((args.output/f"{mode}.json").read_text()) for mode in ("control", "arena")]
    summary["prompt_tokens"] = a["prompt_tokens"]
    summary["output_tokens"] = a["generated_tokens"]
    summary["output_exact"] = a["generated_token_ids"] == b["generated_token_ids"]
    summary["fraction_changes"] = {key: b[key]/a[key]-1 for key in ("peak_memory_bytes", "expert_bytes_read", "seconds", "time_to_first_token_seconds")}
    summary["peak_memory_bytes"] = {"control": a["peak_memory_bytes"], "arena": b["peak_memory_bytes"]}
    summary["failed_metrics"] = [k for k, cap in (("peak_memory_bytes", .15), ("expert_bytes_read", .05)) if summary["fraction_changes"][k] > cap]
    summary["status"] = "passed-cold-edge" if summary["output_exact"] and not summary["failed_metrics"] else "rejected-cold-edge"
    statuses = [json.loads((args.output/f"{m}-status.json").read_text()) for m in ("control", "arena")]
    summary["load_allocation"] = [s["load_allocation"] for s in statuses]
    assert all(x["floor_satisfied"] for s in statuses for x in s["load_allocation"])
    summary["cross_arena_calls"] = [s["cross_arena_calls"] for s in statuses]
    if args.tokens > 1 and not statuses[1]["cross_arena_calls"]:
        summary["status"] = "failed-candidate-not-exercised"
    if a["generated_tokens"] != args.tokens or b["generated_tokens"] != args.tokens:
        summary["status"] = "failed-output-length"
    if any(not x["active"] or x["fallbacks"] for s in statuses for x in s["ane"]):
        summary["status"] = "failed-ane"
    save()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
