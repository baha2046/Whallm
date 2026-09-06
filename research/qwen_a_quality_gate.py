"""Sequential fresh-process exact-token screening; stop at the first failure.

This is a correctness screen, not a repeated performance acceptance experiment.
Use the same five historical prompt files but record actual Qwen token hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"kind": "exact-token-screen-not-performance-acceptance", "pairs": [], "status": "running"}

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    env = {**os.environ, "PYTHONPATH": "runtime:research"}
    save()
    for workload in ("mixed_math", "tool_like", "zh_technical", "repeated", "code"):
        prompt = Path("docs/benchmarks/prompts/2026-08-26-adaptive-4096") / f"{workload}-4096.txt"
        pair = {"workload": workload, "prompt_file_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(), "runs": []}
        result["pairs"].append(pair)
        for mode in ("control", "direct-prefill"):
            name = f"{workload}-{mode}"
            metrics_path = args.output / f"{name}.json"
            status_path = args.output / f"{name}-status.json"
            cmd = ["/usr/bin/time", "-l", sys.executable, "research/qwen_a_run.py", "--mode", mode,
                   "--status", str(status_path), "--", "--model", args.model, "--prompt-file", str(prompt),
                   "--max-tokens", "256", "--temperature", "0", "--top-p", "1", "--top-k", "0",
                   "--slots", "1152", "--no-persistent-prompt-cache", "--metrics-json", str(metrics_path)]
            print(f"START {name}", flush=True)
            start = time.monotonic()
            with (args.output / f"{name}.log").open("w") as log:
                process = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            record = {"mode": mode, "command": cmd, "client_wall_seconds": time.monotonic() - start,
                      "exit_code": process.returncode, "metrics": metrics_path.name, "status_file": status_path.name}
            pair["runs"].append(record)
            if process.returncode != 0:
                result["status"] = "failed-process"
                save()
                return
            status = json.loads(status_path.read_text())
            if not status["ane"] or not all(s["active"] and s["fallbacks"] == 0 for s in status["ane"]):
                result["status"] = "failed-ane-contract"
                save()
                return
            print(f"END {name} {record['client_wall_seconds']:.2f}s", flush=True)
            save()
        control, candidate = [json.loads((args.output / run["metrics"]).read_text()) for run in pair["runs"]]
        a, b = control["generated_token_ids"], candidate["generated_token_ids"]
        pair["prompt_exact"] = control["prompt_token_sha256"] == candidate["prompt_token_sha256"]
        pair["output_exact"] = a == b
        pair["first_divergence_zero_based"] = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else None)
        print(f"PARITY {workload}: {pair['output_exact']}", flush=True)
        if not pair["prompt_exact"] or not pair["output_exact"]:
            result["status"] = "rejected-exact-token-gate"
            save()
            return
        save()
    result["status"] = "passed-quality-screen-performance-unvalidated"
    save()


if __name__ == "__main__":
    main()
