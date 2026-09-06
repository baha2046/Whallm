"""Bounded broader exact-token screen, no performance adoption from one pair."""
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
    summary = {"status": "running", "kind": "exact-token coverage, single pairs not performance adoption", "pairs": []}
    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    cases = [(128, 64, w) for w in ("zh_technical", "tool_like", "mixed_math", "repeated")]
    cases += [(4096, 256, w) for w in ("tool_like", "zh_technical", "mixed_math", "code", "repeated")]
    for length, tokens, workload in cases:
        prompt = Path(f"docs/benchmarks/prompts/2026-08-26-adaptive-{length}/{workload}-{length}.txt")
        pair = {"workload": workload, "prompt_length_label": length, "output_limit": tokens,
                "prompt_file_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(), "runs": []}
        summary["pairs"].append(pair)
        for mode in ("control", "arena"):
            name = f"{workload}-{length}-{mode}"
            metrics, status = args.output / f"{name}.json", args.output / f"{name}-status.json"
            cmd = ["/usr/bin/time", "-l", sys.executable, "research/qwen_decode_run.py", "--mode", mode, "--status", str(status), "--",
                   "--model", args.model, "--prompt-file", str(prompt), "--max-tokens", str(tokens), "--temperature", "0",
                   "--top-p", "1", "--top-k", "0", "--slots", "1152", "--no-persistent-prompt-cache", "--metrics-json", str(metrics)]
            print(f"START {name}", flush=True)
            start = time.monotonic()
            with (args.output / f"{name}.log").open("w") as log:
                process = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "runtime:research"}, stdout=log, stderr=subprocess.STDOUT)
            pair["runs"].append(dict(mode=mode, command=cmd, metrics=metrics.name, status=status.name,
                                     exit_code=process.returncode, client_wall_seconds=time.monotonic()-start))
            if process.returncode:
                summary["status"] = "failed-process"; save(); return
            observed = json.loads(status.read_text())
            if not observed["ane"] or any(not x["active"] or x["fallbacks"] for x in observed["ane"]):
                summary["status"] = "failed-ane-contract"; save(); return
            if mode == "arena" and not observed["arena_state"]["grouped_calls"]:
                summary["status"] = "failed-candidate-not-exercised"; save(); return
            save()
        a, b = [json.loads((args.output / r["metrics"]).read_text()) for r in pair["runs"]]
        pair["prompt_exact"] = a["prompt_token_sha256"] == b["prompt_token_sha256"]
        pair["output_exact"] = a["generated_token_ids"] == b["generated_token_ids"]
        pair["actual_prompt_tokens"] = a["prompt_tokens"]
        pair["first_divergence_zero_based"] = next((i for i, (x, y) in enumerate(zip(a["generated_token_ids"], b["generated_token_ids"])) if x != y),
                                                   min(a["generated_tokens"], b["generated_tokens"]) if a["generated_tokens"] != b["generated_tokens"] else None)
        print(f"PARITY {workload}-{length}: {pair['output_exact']}", flush=True)
        if not pair["prompt_exact"] or not pair["output_exact"]:
            summary["status"] = "rejected-exact-token-gate"; save(); return
        save()
    summary["status"] = "passed-exact-coverage-not-performance-adoption"
    save()


if __name__ == "__main__":
    main()
