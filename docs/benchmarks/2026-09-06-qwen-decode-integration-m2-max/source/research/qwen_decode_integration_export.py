"""Seal all integration attempts, including rejected layouts and cache evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts = [args.source / f"attempt-{i:02d}" for i in (2, 3, 4)]
    latest = attempts[-1]
    matrix = json.loads((latest / "formal-matrix/summary.json").read_text())
    if matrix["status"] == "running":
        parser.error("cannot seal a running matrix")
    args.output.mkdir(parents=True, exist_ok=False)
    index, waves, cold = [], [], []
    snapshots = {sha(p) for root in attempts for p in root.glob("source-v*/*.py")}
    for root in attempts:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(args.source)
            entry = {"bytes": path.stat().st_size, "sha256": sha(path)}
            if any(part.endswith("-cache") for part in relative.parts):
                entry["local_only"] = str(path.resolve())
            else:
                target = args.output / "raw" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                entry["file"] = str(target.relative_to(args.output))
            index.append(entry)
            if path.name != "summary.json":
                continue
            value = json.loads(path.read_text())
            if "medians" in value:
                row = {"path": str(relative), "status": value["status"], "slots": value["slots"],
                       "medians": value["medians"], "fraction_changes": value["fraction_changes"]}
                by_mode = {m: [json.loads((path.parent / r["metrics"]).read_text()) for r in value["runs"] if r["mode"] == m]
                           for m in ("control", "arena")}
                reference = by_mode["control"][0]
                row.update(prompt_tokens=reference["prompt_tokens"], output_tokens=reference["generated_tokens"], token_sha256=reference["token_sha256"])
                row["all_tokens_exact"] = all(x["generated_token_ids"] == reference["generated_token_ids"] and x["prompt_token_sha256"] == reference["prompt_token_sha256"]
                                              for runs in by_mode.values() for x in runs)
                row["candidate_exercised"] = min(x["request_gather_qmm_calls"] for x in by_mode["arena"]) > max(x["request_gather_qmm_calls"] for x in by_mode["control"])
                row["client_generation_seconds"] = {m: statistics.median(x["seconds"] for x in runs) for m, runs in by_mode.items()}
                row["client_generation_fraction_change"] = row["client_generation_seconds"]["arena"] / row["client_generation_seconds"]["control"] - 1
                for filename, digest in value["source_sha256"].items():
                    assert sha(Path(filename)) == digest or digest in snapshots, filename
                assert row["candidate_exercised"] and row["all_tokens_exact"]
                waves.append(row)
            elif "peak_memory_bytes" in value and "gates" in value:
                cold.append({"path": str(relative), "status": value["status"], "outputs": value.get("output_tokens", value["max_tokens"]),
                             "peak_memory_bytes": value["peak_memory_bytes"], "fraction_changes": value["fraction_changes"], "output_exact": value["output_exact"]})
    sources = [*Path("runtime/deepseek_v4_ssd").glob("*.py"), *Path("research").glob("qwen_decode_*.py"),
               *Path("research").glob("qwen_a_*.py"), Path("runtime/tests/test_qwen_grouped_decode.py"),
               Path("runtime/tests/test_model_manager.py"), Path("Sources/DeepSeekV4SSDApp/ServerController.swift"),
               Path("Tests/DeepSeekV4SSDAppTests/ServerConfigurationTests.swift")]
    for path in sources:
        target = args.output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        index.append({"file": str(target.relative_to(args.output)), "bytes": target.stat().st_size, "sha256": sha(target)})
    lifecycle = json.loads((latest / "lifecycle4096/summary.json").read_text())
    edges = json.loads((latest / "cold-edges-summary.json").read_text())
    result = {"status": "RESEARCH_DEFAULT_OFF", "matrix_status": matrix["status"], "lifecycle_status": lifecycle["status"],
              "cold_edges_status": edges["status"], "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "provenance": json.loads((attempts[0] / "integration-provenance.json").read_text()),
              "attempt_dispositions": {"attempt-02": "2048-slot pages rejected: cold peak +29.85%; remaining long matrix intentionally stopped",
                                       "attempt-03": "512-slot pages stopped: short Decode +3.28%, below 5% gate",
                                       "attempt-04": matrix["status"]},
              "waves": waves, "cold_gates": cold, "index": index,
              "limits": ["M2 Max only; M5 Pro not measured", "Greedy sampled token parity does not prove universal equality or model capability",
                         "No energy measurement", "attempt-02 lifecycle4096 v1 did not reach actual cancellation; its v2 and attempt-04 enforce five outputs before cancellation",
                         "Swift: 61 passed, 3 missing-DeepSeek-fixture skips, 4 Keychain-dependent cases not completed"]}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"matrix_status": matrix["status"], "files": len(index), "performance_waves": len(waves), "cold_pairs": len(cold)}))


if __name__ == "__main__":
    main()
