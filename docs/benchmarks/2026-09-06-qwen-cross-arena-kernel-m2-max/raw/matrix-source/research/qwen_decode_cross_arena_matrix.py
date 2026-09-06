"""Sequential acceptance gates for the B cross-arena prototype; stop on failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--first-wave", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = json.loads(args.first_wave.read_text())
    if first["status"] != "passed-cross-arena-wave-research-only":
        raise RuntimeError("first long-tool ABBA must pass")
    for name, sha in first["research_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"first wave source drift: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for tokens in (1, 2, 3, 4, 8, 16):
        tasks.append((f"cold-{tokens}", "qwen_decode_cross_arena_cold.py", ["--tokens", str(tokens)], "passed-cold-edge"))
    tasks.append(("lifecycle", "qwen_decode_cross_arena_lifecycle.py", [], "passed-lifecycle-gate"))
    short = "docs/benchmarks/prompts/2026-08-26-adaptive-128/code-128.txt"
    waves = [("tool-reverse", "docs/benchmarks/prompts/2026-08-26-adaptive-4096/tool_like-4096.txt", 256, 4096, "BAAB")]
    for slots in (4096, 1152):
        for order in ("ABBA", "BAAB"):
            waves.append((f"code143-{slots}-{order.lower()}", short, 64, slots, order))
    for workload in ("zh_technical", "mixed_math", "code", "repeated"):
        prompt = ("docs/benchmarks/prompts/2026-09-06-qwen-decode-zh-extended.txt" if workload == "zh_technical"
                  else f"docs/benchmarks/prompts/2026-08-26-adaptive-4096/{workload}-4096.txt")
        for order in ("ABBA", "BAAB"):
            waves.append((f"{workload}-{order.lower()}", prompt, 256, 4096, order))
    for name, prompt, tokens, slots, order in waves:
        tasks.append((name, "qwen_decode_cross_arena_abba.py", ["--prompt", prompt, "--tokens", str(tokens), "--slots", str(slots), "--order", order], "passed-cross-arena-wave-research-only"))
    summary = {"status": "running", "first_wave": str(args.first_wave), "tasks": tasks, "completed": [],
               "first_wave_sha256": hashlib.sha256(args.first_wave.read_bytes()).hexdigest(),
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("research").glob("qwen_decode_cross_arena*.py")}}
    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    for name, runner, options, expected in tasks:
        command = [sys.executable, "research/" + runner, "--model", args.model, "--output", str(args.output / name), *options]
        print("START", name, flush=True)
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, env={**os.environ, "PYTHONPATH": "runtime:research"}, stdout=log, stderr=subprocess.STDOUT)
        path = args.output / name / "summary.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        row = {"name": name, "exit_code": result.returncode, "status": data.get("status"), "command": command}
        summary["completed"].append(row)
        if result.returncode or row["status"] != expected:
            summary["status"] = "stopped-at-failed-gate"; save(); return
        if runner.endswith("abba.py"):
            requested = int(options[options.index("--tokens") + 1])
            if any(json.loads((path.parent / r["metrics"]).read_text())["generated_tokens"] != requested for r in data["runs"]):
                summary["status"] = "stopped-incomplete-output-coverage"; save(); return
        print("PASS", name, flush=True)
        save()
    summary["status"] = "passed-local-research-matrix-unadopted"
    save()


if __name__ == "__main__":
    main()
