"""Local research runner. Opt-in patches exist only in a fresh worker process."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "v4": (Path.home() / ".dsmodel/deepseek-v4-flash-0731.dsv4", 1152),
    "qwen": (Path.home() / ".dsmodel/qwen3.8-flash-next.dsv4", 3072),
}
VARIANTS = ("route", "control", "dynamic", "lfu", "hot", "dynamic_hot", "cost_hot", "split_io")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_hash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def swapouts():
    text = subprocess.check_output(["vm_stat"], text=True)
    return int(next(line.split(":")[1].strip().rstrip(".")
                    for line in text.splitlines() if line.startswith("Swapouts:")))


def apply_variant(variant):
    """Keep pinning, slot lifetime, routing and math in the production code."""
    from deepseek_v4_ssd.expert_cache import ExpertCache
    sys.argv += ["--separate-prefill-io" if variant == "split_io" else "--no-separate-prefill-io"]
    original_init = ExpertCache.__init__

    def initialize(self, *args, **kwargs):
        if variant in ("lfu", "hot", "dynamic_hot"):
            kwargs["eviction_policy"] = "lfu"
        if variant == "route":
            kwargs["eviction_policy"] = "route"
        original_init(self, *args, **kwargs)
        if variant in ("dynamic", "dynamic_hot"):
            self._layer_reserve = 0

    if variant in ("route", "dynamic", "lfu", "hot", "dynamic_hot"):
        ExpertCache.__init__ = initialize
    if variant in ("hot", "dynamic_hot"):
        def decay(self):
            # Assignment clock, approximately 16 full decode tokens.
            interval = 16 * self.layer_count * self.model.selected_expert_count
            if self._clock - self._last_decay < interval:
                return
            self._last_decay = self._clock
            self._heap = []
            for (layer, expert), entry in self._entries.items():
                entry.frequency = max(1, entry.frequency // 2)
                entry.version += 1
                self._heap.append((self._eviction_rank(entry), entry.last_access,
                                   entry.version, layer, expert))
            heapq.heapify(self._heap)
        ExpertCache._decay_if_needed = decay
    if variant == "cost_hot":
        from ssd_streaming_hotness import install_cost_hotness
        install_cost_hotness()


def prepare_prompt(model_name, target, workload, output):
    from transformers import AutoTokenizer
    model, _ = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model / "tokenizer", trust_remote_code=True)
    source = ROOT / f"docs/benchmarks/prompts/2026-09-05-qwen-research-baseline/{workload}-{target}.txt"
    text = source.read_text()
    if model_name == "v4":
        text = text.removeprefix("<|im_start|>user\n")
        text = text.split("<|im_end|>")[0]
        body = text
        padding = 0
        for _ in range(20):
            text = ((tokenizer.bos_token or "") + "User: Reference padding:" +
                    " context" * padding + "\n" + body + "\nAssistant:")
            size = len(tokenizer.encode(text, add_special_tokens=False))
            if size == target:
                break
            padding += target - size
            if padding < 0:
                raise ValueError("V4 source exceeds requested prompt size")
        else:
            raise ValueError("could not produce exact V4 prompt length")
    output.write_text(text)
    special = tokenizer.bos_token is None or not text.startswith(tokenizer.bos_token)
    ids = tokenizer.encode(text, add_special_tokens=special)
    return {"path": str(output), "source": str(source.relative_to(ROOT)),
            "text_sha256": digest(output), "token_sha256": token_hash(ids),
            "actual_tokens": len(ids), "nominal_tokens": target, "workload": workload}


def summarize(rows):
    controls_exact = len({r["metrics"]["token_sha256"] for r in rows
                          if r["variant"] == "control"}) <= 1
    pairs = []
    for wave in sorted({r["wave"] for r in rows}):
        group = [r for r in rows if r["wave"] == wave]
        controls = [r for r in group if r["variant"] == "control"]
        if len(controls) != 1:
            continue
        control = controls[0]
        for row in group:
            if row["variant"] == "control":
                continue
            a, b = control["metrics"], row["metrics"]
            exact = a["token_sha256"] == b["token_sha256"]
            changes = {}
            for field in ("request_seconds", "time_to_first_token_seconds",
                          "decode_tokens_per_second", "expert_bytes_read", "peak_memory_bytes"):
                if a[field]:
                    changes[field] = 100 * (b[field] / a[field] - 1)
            pairs.append({"wave": wave, "variant": row["variant"], "token_exact": exact,
                          "swapout_free": row["swapout_delta"] == control["swapout_delta"] == 0,
                          "change_percent": changes,
                          "peak_extra_bytes": b["peak_memory_bytes"] - a["peak_memory_bytes"]})
            if (row.get("effective_worker", {}).get("peak_rss_bytes") is not None and
                    control.get("effective_worker", {}).get("peak_rss_bytes") is not None):
                pairs[-1]["rss_extra_bytes"] = (row["effective_worker"]["peak_rss_bytes"] -
                                                control["effective_worker"]["peak_rss_bytes"])
    summary = {}
    for variant in sorted({p["variant"] for p in pairs}):
        items = [p for p in pairs if p["variant"] == variant]
        medians = {key: statistics.median(p["change_percent"][key] for p in items)
                   for key in items[0]["change_percent"]}
        valid = controls_exact and all(p["token_exact"] and p["swapout_free"] and
                    p["peak_extra_bytes"] <= 1_000_000_000 and
                    p.get("rss_extra_bytes", 0) <= 1_000_000_000 for p in items)
        if variant in ("route", "dynamic", "lfu", "hot", "dynamic_hot", "cost_hot"):
            speed = medians["decode_tokens_per_second"] >= 5
        elif variant.startswith("spec_"):
            speed = medians["request_seconds"] <= -5
        else:
            speed = (medians["request_seconds"] <= -5 or
                     medians["time_to_first_token_seconds"] <= -5)
        summary[variant] = {"pairs": len(items), "valid": valid,
                            "median_change_percent": medians,
                            "pair_range_percent": {key: [min(p["change_percent"][key] for p in items),
                                                          max(p["change_percent"][key] for p in items)] for key in medians},
                            "screening_speed_signal": len(items) >= 2 and valid and speed}
    return {"pairs": pairs, "variants": summary}


def suite(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "runtime"), TOKENIZERS_PARALLELISM="false")
    model, slots = MODELS[args.model]
    prompt = prepare_prompt(args.model, args.input_tokens, args.workload, output / "prompt.txt")
    artifact = {
        "evidence_kind": "full_model_screening" if not args.observer else "route_observer",
        "formal_performance_result": False, "status": "running", "model": args.model,
        "environment": {"platform": platform.platform(), "python": sys.version,
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "chip": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))},
        "source": {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "sha256": {str(p.relative_to(ROOT)): digest(p)
                              for p in sorted((ROOT / "runtime").rglob("*.py")) +
                              sorted((ROOT / "research").glob("ssd_streaming*.py"))}},
        "installed_model": {"path": str(model), "manifest_sha256": digest(model / "manifest.json")},
        "prompt": prompt, "slots": slots,
        "cache_state": {"process": "fresh", "expert_slots": "empty", "prompt_cache": "off",
                        "os_page_cache": "not purged; reverse order on alternate waves",
                        "conditioning": "identical control request before every timed worker" if args.prime else "none"},
        "limits": ["Initial screening, not a general performance claim.",
                   "Read bytes are logical expert bytes, not physical SSD traffic.",
                   "Observer runs must not be used for timing conclusions."],
        "runs": [],
    }
    for relative in artifact["source"]["sha256"]:
        snapshot = output / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, snapshot)
    variants = args.variants.split(",")
    if any(v not in VARIANTS for v in variants) or variants.count("control") != 1:
        raise ValueError("variants must contain one control and known unique candidates")
    if len(variants) != len(set(variants)):
        raise ValueError("duplicate variant")
    waves = 1 if args.observer else args.waves
    path = output / "summary.json"
    def save():
        artifact["comparison"] = summarize(artifact["runs"]) if not args.observer else None
        path.write_text(json.dumps(artifact, indent=2) + "\n")
    save()
    try:
        for wave in range(waves):
            order = variants if wave % 2 == 0 else list(reversed(variants))
            for variant in order:
                prefix = output / f"{wave}-{variant}"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", variant,
                           "--model", str(model), "--prompt-file", prompt["path"],
                           "--max-tokens", str(args.output_tokens), "--temperature", "0",
                           "--top-p", "1", "--top-k", "0", "--approximation", "exact",
                           "--slots", str(slots), "--expert-eviction-policy", "lru",
                           "--read-workers", "4", "--prefetch-read-workers", "2",
                           "--memory-limit-gib", "48", "--prompt-cache", "off",
                           "--metrics-json", str(prefix.with_suffix(".json"))]
                if args.observer:
                    command += ["--expert-route-trace", str(prefix.with_suffix(".route.json"))]
                if args.prime:
                    primer_swap = swapouts()
                    primer = list(command)
                    primer[primer.index("--worker") + 1] = "control"
                    primer[primer.index("--metrics-json") + 1] = str(prefix) + ".primer.json"
                    print(f"PRIME {args.model} wave={wave} {variant}", flush=True)
                    with Path(str(prefix) + ".primer.stdout").open("w") as out, Path(str(prefix) + ".primer.stderr").open("w") as err:
                        subprocess.run(primer, cwd=ROOT, env=env, stdout=out, stderr=err,
                                       stdin=subprocess.DEVNULL, timeout=1800, check=True)
                    primer_metrics = json.loads(Path(str(prefix) + ".primer.json").read_text())
                    if (primer_metrics["generated_tokens"] != args.output_tokens or
                            primer_metrics["prompt_token_sha256"] != prompt["token_sha256"] or
                            swapouts() != primer_swap):
                        raise ValueError("conditioning request failed token/memory checks")
                print(f"START {args.model} wave={wave} {variant}", flush=True)
                before = swapouts()
                started = time.time()
                with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err,
                                            stdin=subprocess.DEVNULL, timeout=1800)
                if result.returncode:
                    raise RuntimeError(f"{prefix.name} failed ({result.returncode}); see stderr")
                metrics = json.loads(prefix.with_suffix(".json").read_text())
                metrics["combined_expert_bytes_read"] = (metrics["expert_bytes_read"] +
                    metrics.get("mtp_expert_bytes_read", 0) + metrics.get("dspark_draft_expert_bytes_read", 0))
                if metrics["token_sha256"] != token_hash(metrics["generated_token_ids"]):
                    raise ValueError("invalid output token hash")
                if metrics["prompt_token_sha256"] != prompt["token_sha256"]:
                    raise ValueError("prompt token hash changed")
                if metrics["generated_tokens"] != args.output_tokens:
                    raise ValueError("output stopped early; use another prompt for equal-work timing")
                row = {"wave": wave, "variant": variant, "command": command,
                       "started_unix": started, "process_seconds": time.time() - started,
                       "swapout_delta": swapouts() - before, "metrics": metrics,
                       "effective_worker": json.loads(Path(str(prefix.with_suffix(".json")) + ".effective.json").read_text()),
                       "metrics_sha256": digest(prefix.with_suffix(".json"))}
                artifact["runs"].append(row)
                if Path(str(prefix.with_suffix(".json")) + ".reader.json").exists():
                    row["reader_counts"] = json.loads(Path(str(prefix.with_suffix(".json")) + ".reader.json").read_text())
                save()
                print(f"DONE {variant}: TTFT={metrics['time_to_first_token_seconds']:.2f}s "
                      f"decode={metrics['decode_tokens_per_second']:.2f} tok/s", flush=True)
        artifact["status"] = "complete"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        variant = sys.argv[2]
        if variant not in VARIANTS:
            raise ValueError("unknown variant")
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        apply_variant(variant)
        if "--expert-route-trace" in sys.argv:
            from ssd_streaming_observer import install_observer
            install_observer()
        effective_path = Path(sys.argv[sys.argv.index("--metrics-json") + 1] + ".effective.json")
        effective = {"variant": variant, "argv": sys.argv[1:],
            "prefill_file_policy": "bypass_direct_with_aligned_fallback" if variant == "split_io" else "cached",
            "layer_reserve": "dynamic_marginal_value" if variant == "route" else (0 if variant in ("dynamic", "dynamic_hot", "cost_hot") else "production"),
            "frequency_half_life_decode_tokens": 16 if variant in ("hot", "dynamic_hot", "cost_hot") else "production",
            "route_history_half_life_tokens": [16, 256] if variant == "route" else None,
            "cost_weight_range": [0.25, 4.0] if variant == "cost_hot" else None}
        effective_path.write_text(json.dumps(effective, indent=2) + "\n")
        from deepseek_v4_ssd.cli import main
        try:
            main()
        finally:
            effective["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            effective_path.write_text(json.dumps(effective, indent=2) + "\n")
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--model", choices=MODELS, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--variants", default="control,dynamic,lfu,hot")
        parser.add_argument("--input-tokens", type=int, choices=(1024, 4096), default=1024)
        parser.add_argument("--output-tokens", type=int, default=64)
        parser.add_argument("--workload", choices=("code", "zh_technical", "mixed_math", "tool_like"), default="code")
        parser.add_argument("--waves", type=int, default=2)
        parser.add_argument("--observer", action="store_true")
        parser.add_argument("--prime", action="store_true", help="run the same control request before each timed worker")
        suite(parser.parse_args())
