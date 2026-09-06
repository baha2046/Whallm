"""Seal a new B research artifact; preserve failed gates and source snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index = []
    for source in sorted(args.attempt.rglob("*")):
        relative = source.relative_to(args.attempt)
        if not source.is_file() or relative.parts[0] == "mlx" or "__pycache__" in relative.parts:
            continue
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        # Persistent cache binaries stay local; record hash and size for auditing.
        local_only = any(p.endswith("-cache") for p in relative.parts) and source.suffix not in (".json", ".jsonl")
        record = {"path": str(relative), "sha256": digest, "bytes": source.stat().st_size, "local_only": local_only}
        if local_only:
            record["local_path"] = str(source.resolve())
        else:
            target = args.output / "raw" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        index.append(record)
    summaries = {}
    for name in ("tool-abba", "matrix"):
        path = args.attempt / name / "summary.json"
        if path.exists():
            summaries[name] = json.loads(path.read_text())
    if any(s.get("status") == "running" for s in summaries.values()):
        raise RuntimeError("do not seal an active attempt")
    summary = {"status": "RESEARCH_ONLY_UNADOPTED", "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "first_wave_artifact": "raw/tool-abba/summary.json", "matrix_artifact": "raw/matrix/summary.json",
               "attempt": str(args.attempt.resolve()), "summaries": summaries, "index": index,
               "limits": ["M2 Max only", "No energy measurement", "Kernel not installed in production runtime", "No gate relaxation"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"indexed_files": len(index), "status": summary["status"]}), flush=True)


if __name__ == "__main__":
    main()
