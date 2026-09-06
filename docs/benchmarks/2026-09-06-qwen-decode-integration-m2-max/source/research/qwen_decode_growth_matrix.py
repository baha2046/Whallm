"""Growth-plan integration acceptance matrix; completed 4096 short waves are not repeated; stop at the first failed wave."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    short = "docs/benchmarks/prompts/2026-08-26-adaptive-128/code-128.txt"
    cases = [("code143-1152-abba", short, 64, 1152, "ABBA")]
    for workload in ("tool_like", "zh_technical", "mixed_math", "code", "repeated"):
        prompt = ("docs/benchmarks/prompts/2026-09-06-qwen-decode-zh-extended.txt" if workload == "zh_technical"
                  else f"docs/benchmarks/prompts/2026-08-26-adaptive-4096/{workload}-4096.txt")
        for order in ("ABBA", "BAAB"):
            cases.append((f"{workload}-4096-{order.lower()}", prompt, 256, 4096, order))
    summary = {"status": "running", "cases": cases, "waves": [], "stop_on_any_failed_wave": True,
               "runner_sha256": hashlib.sha256(Path("research/qwen_decode_integrated_abba.py").read_bytes()).hexdigest()}
    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    for name, prompt, tokens, slots, order in cases:
        cmd = [sys.executable, "research/qwen_decode_integrated_abba.py", "--output", str(args.output/name),
               "--model", args.model, "--prompt", prompt, "--tokens", str(tokens), "--slots", str(slots), "--order", order]
        print(f"WAVE {name}", flush=True)
        process = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "runtime:research"})
        path = args.output / name / "summary.json"
        wave = json.loads(path.read_text()) if path.exists() else {}
        row = {"name": name, "status": wave.get("status"), "exit_code": process.returncode, "command": cmd}
        summary["waves"].append(row)
        if process.returncode or row["status"] != "passed-integrated-wave-default-off":
            summary["status"] = "stopped-at-failed-wave"; save(); return
        for run in wave["runs"]:
            metrics = json.loads((path.parent / run["metrics"]).read_text())
            if metrics["generated_tokens"] != tokens:
                summary["status"] = "stopped-incomplete-output-coverage"; save(); return
        save()
    summary["status"] = "passed-local-integration-matrix"
    save()


if __name__ == "__main__":
    main()
