"""Fresh-process Qwen baseline and separate route-observer checks for research B."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(metrics: dict, prompt: dict) -> None:
    ids = metrics["generated_token_ids"]
    expected = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
    if metrics["token_sha256"] != expected or len(ids) != 256:
        raise ValueError(f"output token count/hash mismatch: count={len(ids)}, hash_valid={metrics['token_sha256'] == expected}")
    if metrics["prompt_token_sha256"] != prompt["prompt_token_sha256"]:
        raise ValueError("prompt tokens differ from the prepared manifest")
    if metrics["approximation_mode"] != "exact" or metrics["mtp_enabled"]:
        raise ValueError("baseline must use exact mode without MTP")
    if metrics["prompt_cache_reused_tokens"]:
        raise ValueError("baseline unexpectedly reused prompt state")


def prepare_prompts(root: Path, model: Path, directory: Path) -> list[dict]:
    from transformers import AutoTokenizer
    from prepare_r0_prompts import SEEDS, encode_prompt, token_sha256

    tokenizer = AutoTokenizer.from_pretrained(model / "tokenizer", trust_remote_code=True)
    directory.mkdir()
    prompts = []
    for name, seed in SEEDS.items():
        source = root / f"docs/benchmarks/prompts/2026-09-01-ssd-cache-oracle/{name}.txt"
        original = source.read_text()
        marker = "<|im_start|>user\n"
        if not original.startswith(marker):
            raise ValueError("expected the existing Qwen chat prompt")
        body = original[len(marker):]
        context = seed
        while len(tokenizer.encode(context, add_special_tokens=False)) < 4200:
            context += context
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        for target in (1024, 4096):
            count = max(1, target - len(encode_prompt(tokenizer, original)) - 128)
            prefix = tokenizer.decode(context_ids[:count], clean_up_tokenization_spaces=False)
            padding = 0
            for _ in range(20):
                text = marker + "Background examples:\n" + prefix + "\nPadding:" + " test" * padding + "\n\nTask:\n" + body
                tokens = encode_prompt(tokenizer, text)
                if len(tokens) == target:
                    break
                padding += target - len(tokens)
                if padding < 0:
                    raise ValueError("background exceeded the prompt budget")
            else:
                raise ValueError("could not build the requested prompt length")
            path = directory / f"{name}-{target}.txt"
            path.write_text(text)
            prompts.append({"name": name, "target_tokens": target, "file": path.name,
                            "prompt_token_sha256": token_sha256(tokens),
                            "source": str(source.relative_to(root)), "source_sha256": digest(source),
                            "text_sha256": digest(path)})
    (directory / "manifest.json").write_text(json.dumps({"method": "Existing long-answer chat tasks with repeated background padding; thinking disabled", "prompts": prompts}, indent=2) + "\n")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = args.model.expanduser().resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(root / "runtime"))
    prompts_dir = output / "prompts"
    prompts = prepare_prompts(root, model, prompts_dir)
    source_files = sorted((root / "runtime/deepseek_v4_ssd").glob("*.py"))
    source_files += [Path(__file__), root / "Scripts/prepare_r0_prompts.py",
                     root / "research/PREFILL_DECODE_B_ROUND1_2026-09-05.md"]
    artifact = {
        "schema_version": 1, "evidence_kind": "current_baseline_and_observer_validation",
        "formal_performance_result": False, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                   "files_sha256": {str(p.relative_to(root)): digest(p) for p in source_files}},
        "environment": {"platform": platform.platform(), "python": sys.version,
                        "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
                        "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy")}},
        "installed_model": {"path": str(model), "manifest_sha256": digest(model / "manifest.json")},
        "cache_state": {"process": "fresh_per_request", "expert_cache": "empty",
                        "persistent_prompt_cache": "disabled", "os_page_cache": "not purged"},
        "limits": ["Existing long-answer chat tasks with synthetic background padding; not held-out capability evaluation.",
                   "Single untraced run per case except the code-4096 repeat; no speed adoption claim.",
                   "Observer runs are separate from timing baselines.",
                   "CLI first text and process wall include model loading; runtime TTFT does not.",
                   "CLI output chunk gaps are not exact per-token streaming latency.",
                   "Logical expert bytes and summed read timers do not identify physical SSD traffic or exposed wait."],
        "runs": [],
    }
    plan = [(p, "baseline") for p in prompts]
    plan += [(next(p for p in prompts if p["name"] == "code" and p["target_tokens"] == 4096), "repeat")]
    plan += [(p, "observer") for p in prompts if p["target_tokens"] == 4096]
    controls = {}
    try:
        for number, (prompt, kind) in enumerate(plan, 1):
            key = f'{prompt["name"]}-{prompt["target_tokens"]}'
            prefix = output / f"{number:02d}-{key}-{kind}"
            metrics_path = prefix.with_suffix(".json")
            trace_path = prefix.with_suffix(".route.json")
            command = [sys.executable, "-m", "deepseek_v4_ssd.cli", "--model", str(model),
                       "--prompt-file", str(prompts_dir / prompt["file"]),
                       "--max-tokens", "256", "--temperature", "0", "--top-p", "1", "--top-k", "0",
                       "--approximation", "exact", "--slots", "4096", "--read-workers", "4",
                       "--prefetch-read-workers", "2", "--memory-limit-gib", "48",
                       "--ane-prefill-ratio", "0.25", "--no-persistent-prompt-cache",
                       "--metrics-json", str(metrics_path)]
            if kind == "observer":
                command += ["--expert-route-trace", str(trace_path)]
            print(f"START {number}/{len(plan)} {key} {kind}", flush=True)
            started = time.perf_counter()
            emissions = []
            with prefix.with_suffix(".stderr").open("wb") as err, prefix.with_suffix(".txt").open("wb") as out:
                proc = subprocess.Popen(["/usr/bin/time", "-l", *command], cwd=root, env=env,
                                        stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL)
                assert proc.stdout is not None
                while data := os.read(proc.stdout.fileno(), 65536):
                    emissions.append(time.perf_counter() - started)
                    out.write(data)
                returncode = proc.wait()
            if returncode:
                raise RuntimeError(f"{key} {kind} exited {returncode}; see {prefix}.stderr")
            wall = time.perf_counter() - started
            metrics = json.loads(metrics_path.read_text())
            validate_run(metrics, prompt)
            if kind == "baseline":
                controls[key] = metrics["token_sha256"]
            elif metrics["token_sha256"] != controls[key]:
                raise ValueError(f"{key}: {kind} differs from baseline output")
            rss = re.search(r"(\d+)\s+maximum resident set size", prefix.with_suffix(".stderr").read_text())
            artifact["runs"].append({"case": key, "kind": kind, "command": command,
                                     "prompt_file_sha256": digest(prompts_dir / prompt["file"]),
                                     "metrics": metrics, "metrics_path": str(metrics_path),
                                     "metrics_sha256": digest(metrics_path),
                                     "trace_path": str(trace_path) if kind == "observer" else None,
                                     "trace_sha256": digest(trace_path) if kind == "observer" else None,
                                     "process_wall_seconds": wall,
                                     "process_first_text_seconds": emissions[0] if emissions else None,
                                     "process_max_output_chunk_gap_seconds": max((b-a for a,b in zip(emissions, emissions[1:])), default=0),
                                     "max_rss_bytes": int(rss[1]) if rss else None})
            (output / "baseline.json").write_text(json.dumps(artifact, indent=2) + "\n")
            print(f'DONE {number}/{len(plan)} TTFT={metrics["time_to_first_token_seconds"]:.2f}s decode={metrics["decode_tokens_per_second"]:.2f} hash={metrics["token_sha256"][:12]}', flush=True)
        artifact["status"] = "completed"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "baseline.json").write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
