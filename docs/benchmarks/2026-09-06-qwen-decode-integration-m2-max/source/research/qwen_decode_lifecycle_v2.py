"""Fresh control/candidate processes: warm cache, cancellation and restart parity."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def worker(args):
    import mlx.core as mx
    from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
    from deepseek_v4_ssd.model import RuntimeConfig

    config = RuntimeConfig(slots=args.slots, qwen_grouped_decode=args.mode == "arena",
                           prompt_cache_directory=str(args.cache), persistent_prompt_cache=True)
    result = {"mode": args.mode, "stage": args.stage, "config": asdict(config), "runs": [], "status": "running"}
    output = args.output / f"{args.mode}-{args.stage}.json"
    runtime = ModelRuntime.open(args.model, config)
    prompt = Path("docs/benchmarks/prompts/2026-08-26-adaptive-128/code-128.txt").read_text()
    branch = prompt + "\nAlso explain the time complexity of each operation."
    cases = [("restart", prompt, 32, None)] if args.stage == "restart" else [
        ("cold", prompt, 32, None), ("memory-repeat", prompt, 32, None),
        ("branch-eos", branch, 32, None), ("decode-cancel", prompt, 32, 5), ("after-cancel", prompt, 32, None),
        ("one-output", prompt, 1, None), ("after-one-output", prompt, 32, None),
    ]
    try:
        for name, text, limit, cancel in cases:
            start = time.perf_counter()
            tokens = []
            stream = runtime.stream(text, GenerationOptions(max_tokens=limit, temperature=0, top_p=1, top_k=0))
            try:
                for piece in stream:
                    tokens.append(int(piece.token))
                    if cancel and len(tokens) >= cancel:
                        break
            finally:
                stream.close()
            state = runtime.expert_cache
            row = {"name": name, "token_ids": tokens, "token_sha256": digest(tokens),
                   "prompt_sha256": digest(runtime._encode_prompt(text)), "cancelled": cancel is not None,
                   "client_seconds": time.perf_counter()-start, "metrics": runtime.metrics.snapshot(),
                   "active_memory_bytes": mx.get_active_memory(), "peak_memory_bytes": mx.get_peak_memory(),
                   "resident_experts": state.resident_count,
                   "dispatch_reset": not state.qwen_decode_active and not state._qwen_decode_request and state._qwen_prefill_remaining is None}
            result["runs"].append(row)
            output.write_text(json.dumps(result, indent=2) + "\n")
            if cancel and len(tokens) != cancel:
                raise RuntimeError("EOS prevented actual cancellation coverage")
            if not row["dispatch_reset"]:
                raise RuntimeError("Decode state leaked across requests")
            if name in ("memory-repeat", "restart") and not row["metrics"]["prompt_cache_reused_tokens"]:
                raise RuntimeError(f"{name} did not reuse prompt cache")
            print(f"{args.mode} {name}: {len(tokens)} outputs; reused {row['metrics']['prompt_cache_reused_tokens']}", flush=True)
        result["ane"] = runtime.model.ane_prefill.snapshot()
        if not result["ane"]["active"] or result["ane"]["fallbacks"]:
            raise RuntimeError("ANE contract failed")
        result["status"] = "completed"
    finally:
        runtime.close()
        result["cache_files"] = [str(p.relative_to(args.cache)) for p in args.cache.rglob("*") if p.is_file()]
        output.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4096)
    parser.add_argument("--mode", choices=("control", "arena"))
    parser.add_argument("--stage", choices=("seed", "restart"))
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.mode:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"kind": "exact lifecycle gate, not performance acceptance", "status": "running", "processes": [], "pairs": [],
               "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("runtime/deepseek_v4_ssd").glob("*.py")}}
    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    for stage in ("seed", "restart"):
        for mode in ("control", "arena"):
            cmd = [sys.executable, __file__, "--model", args.model, "--output", str(args.output), "--slots", str(args.slots),
                   "--mode", mode, "--stage", stage, "--cache", str(args.output / f"{mode}-cache")]
            print(f"START {mode}-{stage}", flush=True)
            with (args.output / f"{mode}-{stage}.log").open("w") as log:
                completed = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "runtime:research"}, stdout=log, stderr=subprocess.STDOUT)
            summary["processes"].append({"mode": mode, "stage": stage, "command": cmd, "exit_code": completed.returncode})
            if completed.returncode:
                summary["status"] = "failed-process"; save(); return
            save()
        a, b = [json.loads((args.output / f"{mode}-{stage}.json").read_text()) for mode in ("control", "arena")]
        for left, right in zip(a["runs"], b["runs"], strict=True):
            pair = {"stage": stage, "name": left["name"], "output_exact": left["token_ids"] == right["token_ids"],
                    "prompt_exact": left["prompt_sha256"] == right["prompt_sha256"],
                    "reused_tokens_equal": left["metrics"]["prompt_cache_reused_tokens"] == right["metrics"]["prompt_cache_reused_tokens"],
                    "expert_bytes_equal": left["metrics"]["request_expert_bytes_read"] == right["metrics"]["request_expert_bytes_read"]}
            summary["pairs"].append(pair)
            if not all(pair[k] for k in ("output_exact", "prompt_exact", "reused_tokens_equal", "expert_bytes_equal")):
                summary["status"] = "rejected-lifecycle-parity"; save(); return
        save()
    summary["status"] = "passed-lifecycle-gate"
    save()
    print(summary["status"], flush=True)


if __name__ == "__main__":
    main()
