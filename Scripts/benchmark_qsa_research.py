"""Pre-registered QSA or sorted-expert Prefill research, with reversed pairs."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("pilot", "paired"))
    parser.add_argument("--candidate", type=int, choices=(8, 16, 32))
    parser.add_argument("--sorted-experts", action="store_true")
    args = parser.parse_args()
    if args.phase == "paired" and args.candidate is None and not args.sorted_experts:
        parser.error("paired phase needs the selected candidate")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_path = root / "docs/benchmarks/2026-09-05-qwen-research-baseline-m5-pro.json"
    baseline = json.loads(baseline_path.read_text())
    controls = {r["case"]: r for r in baseline["runs"] if r["kind"] == "baseline" and r["case"].endswith("-4096")}
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = root / ("research/QWEN_GROUPED_EXPERTS_2026-09-06.md" if args.sorted_experts else "research/QWEN_ATTENTION_GRANULARITY_2026-09-06.md")
    (output / "protocol-at-start.md").write_bytes(protocol.read_bytes())
    files = list((root / "runtime/deepseek_v4_ssd").glob("*.py")) + [Path(__file__), root / "Scripts/research_qwen_attention.py", protocol]
    if args.sorted_experts:
        files.append(root / "Scripts/research_qwen_sorted_experts.py")
    artifact = dict(status="running", formal_performance_result=False, phase=args.phase,
                    candidate_kind="sorted_experts_without_hint" if args.sorted_experts else "qsa_query_chunk",
                    started_at=datetime.now(timezone.utc).isoformat(),
                    source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                                files_sha256={str(p.relative_to(root)): sha(p) for p in files}),
                    environment=dict(platform=platform.platform(), python=sys.version,
                                     chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                                     memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
                                     packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy")}),
                    installed_model=baseline["installed_model"], baseline_sha256=sha(baseline_path),
                    cache_state=baseline["cache_state"], runs=[],
                    limits=["Existing synthetic prompts; not held-out adoption validation.",
                            "OS page cache not purged; paired order reduces but cannot remove system variation.",
                            "All candidates are process-local; no runtime defaults changed."])
    assert sha(Path(artifact["installed_model"]["path"]) / "manifest.json") == artifact["installed_model"]["manifest_sha256"]
    if args.phase == "pilot":
        count = 32
        plan = [(0, "code-4096", size) for size in ((4, 0, 0, 4) if args.sorted_experts else (4, 8, 16, 32, 4))]
    else:
        count = 256
        candidate_size = 0 if args.sorted_experts else args.candidate
        plan = [(wave, case, size) for wave in (0, 1) for case in controls
                for size in ((4, candidate_size) if wave == 0 else (candidate_size, 4))]
    try:
        for index, (wave, case, size) in enumerate(plan, 1):
            label = "sorted-no-hint" if size == 0 else f"chunk{size}"
            stem = output / f"{index:02d}-{case}-{label}"
            command = controls[case]["command"].copy()
            command.append("--no-qwen-grouped-experts")
            command[0] = sys.executable
            command[command.index("--max-tokens") + 1] = str(count)
            command[command.index("--metrics-json") + 1] = str(stem.with_suffix(".json"))
            record = stem.with_suffix(".record.json")
            cli_arguments = command[3:]
            command = [command[0], str(root / "Scripts/research_qwen_attention.py"),
                       "--record", str(record), "--query-chunk", str(size), "--", *cli_arguments]
            if size == 0:
                command = [command[0], str(root / "Scripts/research_qwen_sorted_experts.py"),
                           "--record", str(record), "--no-sorting-hint", "--", *cli_arguments]
            print(f"START {index}/{len(plan)} {case} {label}", flush=True)
            started = time.perf_counter()
            with stem.with_suffix(".txt").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
                subprocess.run(["/usr/bin/time", "-l", *command], cwd=root,
                               env=os.environ | {"PYTHONPATH": str(root / "runtime")},
                               stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=True)
            elapsed = time.perf_counter()-started
            metrics = json.loads(stem.with_suffix(".json").read_text())
            ids = metrics["generated_token_ids"]
            assert metrics["token_sha256"] == hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
            assert metrics["approximation_mode"] == "exact" and not metrics["mtp_enabled"] and metrics["prompt_cache_reused_tokens"] == 0
            assert metrics["prompt_token_sha256"] == controls[case]["metrics"]["prompt_token_sha256"]
            parity = ids == controls[case]["metrics"]["generated_token_ids"][:count]
            rss = re.search(r"(\d+)\s+maximum resident set size", stem.with_suffix(".stderr").read_text())
            artifact["runs"].append(dict(wave=wave, case=case, chunk=size, command=command,
                                         variant="control" if size == 4 else "candidate",
                                         process_seconds=elapsed, max_rss_bytes=int(rss[1]) if rss else None,
                                         metrics=metrics, token_parity=parity,
                                         record=json.loads(record.read_text()), metrics_sha256=sha(stem.with_suffix(".json"))))
            (output / "results.json").write_text(json.dumps(artifact, indent=2)+"\n")
            print(f'DONE TTFT={metrics["time_to_first_token_seconds"]:.2f} decode={metrics["decode_tokens_per_second"]:.2f} parity={parity}', flush=True)
            if not parity and (size == 4 or args.phase == "paired"):
                raise ValueError("output parity failed; stop this dependent experiment")
        artifact["status"] = "completed"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(artifact, indent=2)+"\n")


if __name__ == "__main__":
    main()
