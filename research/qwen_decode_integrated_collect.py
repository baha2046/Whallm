"""Export immutable integration evidence, retaining failures and source snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = []
    for path in sorted(args.source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(args.source)
        if any(part.endswith("-cache") for part in relative.parts):
            # Persistent cache binary data stay local; metadata and hashes are audited.
            index.append({"local_only": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)})
            continue
        target = args.output / "raw" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        index.append({"file": str(target.relative_to(args.output)), "bytes": target.stat().st_size, "sha256": sha(target)})
    sources = [*Path("runtime/deepseek_v4_ssd").glob("*.py"), *Path("research").glob("qwen_decode_integrated_*.py"),
               Path("research/qwen_decode_lifecycle.py"), Path("research/qwen_decode_lifecycle_v2.py"),
               Path("research/qwen_a_run.py"), Path("runtime/tests/test_qwen_grouped_decode.py"),
               Path("Sources/DeepSeekV4SSDApp/ServerController.swift"), Path("tests/DeepSeekV4SSDAppTests/ServerConfigurationTests.swift")]
    for path in sources:
        target = args.output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        index.append({"file": str(target.relative_to(args.output)), "bytes": target.stat().st_size, "sha256": sha(target)})
    matrix = json.loads((args.source / "formal-matrix/summary.json").read_text())
    lifecycle = json.loads((args.source / "lifecycle4096-v2/summary.json").read_text())
    summary = {"status": "LOCAL_INTEGRATION_REVIEW_DEFAULT_OFF", "matrix_status": matrix["status"],
               "lifecycle_status": lifecycle["status"], "waves": [], "index": index,
               "provenance": json.loads((args.source / "integration-provenance.json").read_text()),
               "limits": ["M2 Max only; M5 Pro not measured", "greedy token parity is not a capability benchmark or universal numerical proof",
                          "no energy measurement", "lifecycle v1 had early EOS instead of cancellation; v2 enforces five outputs before cancellation",
                          "Swift Keychain-dependent tests blocked; missing DeepSeek fixture tests skipped"]}
    paths = [args.source / "code143-4096-abba/summary.json"]
    paths += [args.source / "formal-matrix" / w["name"] / "summary.json" for w in matrix["waves"]]
    for path in paths:
        wave = json.loads(path.read_text())
        row = {"name": path.parent.name, "status": wave["status"], "slots": wave["slots"],
               "fraction_changes": wave.get("fraction_changes"), "medians": wave.get("medians")}
        if wave["status"] == "passed-integrated-wave-default-off":
            by_mode = {mode: [json.loads((path.parent / run["metrics"]).read_text()) for run in wave["runs"] if run["mode"] == mode]
                       for mode in ("control", "arena")}
            row["client_generation_seconds"] = {mode: statistics.median(x["seconds"] for x in runs) for mode, runs in by_mode.items()}
            row["client_generation_fraction_change"] = row["client_generation_seconds"]["arena"] / row["client_generation_seconds"]["control"] - 1
            row["prompt_tokens"] = by_mode["control"][0]["prompt_tokens"]
            row["output_tokens"] = by_mode["control"][0]["generated_tokens"]
            row["token_sha256"] = by_mode["control"][0]["token_sha256"]
            row["candidate_exercised"] = min(x["request_gather_qmm_calls"] for x in by_mode["arena"]) > max(x["request_gather_qmm_calls"] for x in by_mode["control"])
            assert row["candidate_exercised"]
            for relative, digest in wave["source_sha256"].items():
                current = Path(relative)
                snapshots = [args.source / version / current.name for version in ("source-v2", "source-v3")]
                assert sha(current) == digest or any(p.exists() and sha(p) == digest for p in snapshots), relative
        summary["waves"].append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "matrix": summary["matrix_status"], "lifecycle": summary["lifecycle_status"], "evidence_files": len(index)}))


if __name__ == "__main__":
    main()
