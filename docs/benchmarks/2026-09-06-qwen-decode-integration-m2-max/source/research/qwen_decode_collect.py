"""Publish immutable Decode research evidence after the bounded gate finishes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quality = json.loads((args.attempt / "quality/summary.json").read_text())
    extension_path = args.attempt / "zh-extended-summary.json"
    extension = json.loads(extension_path.read_text()) if extension_path.exists() else None
    if quality["status"] == "running":
        parser.error("quality gate still running")
    if quality["status"].startswith("passed-") and extension is None:
        parser.error("long Chinese extension has not been recorded")
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for p in sorted(args.attempt.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(args.attempt)
        row = {"path": str(rel), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "included": p.suffix in (".json", ".log", ".py")}
        if row["included"]:
            target = args.output / "raw" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        else:
            row["local_evidence_path"] = str(p.resolve())
        artifacts.append(row)
    waves = {name: json.loads((args.attempt / name / "summary.json").read_text()) for name in ("code143-abba", "code143-baab")}
    coverage = []
    for pair in quality["pairs"] + ([extension] if extension else []):
        row = {k: v for k, v in pair.items() if k != "runs"}
        row["runs"] = []
        for run in pair["runs"]:
            p = args.attempt / "quality" / run["metrics"]
            if not p.exists():
                row["runs"].append(run)
                continue
            metrics = json.loads(p.read_text())
            selected = {k: metrics[k] for k in ("prompt_tokens", "generated_tokens", "prompt_token_sha256", "token_sha256",
                        "request_seconds", "time_to_first_token_seconds", "decode_tokens_per_second", "decode_latency_p95_seconds",
                        "peak_memory_bytes", "expert_bytes_read", "request_process_disk_bytes_read")}
            log = p.with_suffix(".log").read_text()
            match = re.search(r"(\d+)\s+maximum resident set size", log)
            selected["max_rss_bytes"] = int(match[1]) if match else None
            row["runs"].append({**run, "observed": selected})
        coverage.append(row)
    passed = quality["status"].startswith("passed-") and extension and extension.get("output_exact") and extension.get("long_decode_covered")
    status = "SCOPED_RESEARCH_CANDIDATE_NOT_ADOPTED" if passed else "STOPPED_AT_BROADER_GATE"
    summary = {"schema_version": 1, "status": status, "adopted": False,
               "provenance": json.loads((args.attempt / "provenance.json").read_text()),
               "candidate_provenance": json.loads((args.attempt / "candidate-provenance.json").read_text()),
               "final_source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("research").glob("qwen_decode_*.py"))},
               "repeated_performance_scope": "code143/64, fresh process, M2 Max, 1152 slots, persistent cache disabled, OS cache not purged",
               "waves": waves, "quality_status": quality["status"], "coverage": coverage,
               "remaining_adoption_gates": ["Repeated long-context and non-code workload performance", "Multi-request warm/restart cache lifecycle", "App 4096 slots", "M5 Pro hardware replication", "Production integration and regression suite"],
               "limits": ["Single-pair coverage timings are not repeated performance acceptance", "Synthetic prompts establish parity, not task-quality improvement", "No power or energy improvement measured", "Model weights, top-10 router, read/eviction policy, runtime defaults unchanged"],
               "artifacts": artifacts}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in artifacts:
        if row["included"]:
            assert hashlib.sha256((args.output / "raw" / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    print(json.dumps({"status": status, "files_indexed": len(artifacts), "output": str(args.output)}))


if __name__ == "__main__":
    main()
