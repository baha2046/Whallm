"""Independently recompute completed research gates from raw measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    root = args.attempt
    output = root / "final-audit.json"
    if output.exists():
        raise RuntimeError("new audit output required")
    assert read(root / "matrix/summary.json")["status"] == "passed-local-research-matrix-unadopted"
    paths = [root / "tool-abba/summary.json", *sorted((root / "matrix").glob("*/summary.json"))]
    groups = {}
    waves = []
    generated = 0
    for path in paths:
        summary = read(path)
        if "order" not in summary:
            continue
        assert summary["status"] == "passed-cross-arena-wave-research-only"
        for name, sha in summary["source_sha256"].items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
        for name in ("research/qwen_decode_cross_arena.py", "research/qwen_decode_cross_arena_run.py",
                     "research/qwen_decode_cross_arena_abba.py", "research/kernels/qwen_cross_arena_qmv.h"):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == summary["research_sha256"][name]
        runs = {"control": [], "arena": []}
        for run in summary["runs"]:
            metrics, status = read(path.parent / run["metrics"]), read(path.parent / run["status"])
            requested = int(run["command"][run["command"].index("--max-tokens") + 1])
            assert metrics["generated_tokens"] == len(metrics["generated_token_ids"]) == requested
            assert "--no-persistent-prompt-cache" in run["command"]
            assert status["status"] == "completed"
            assert status["ane"] and all(a["active"] and not a["fallbacks"] for a in status["ane"])
            assert status["cross_arena_calls"] == (48 * requested if run["mode"] == "arena" else 0)
            key = (metrics["prompt_token_sha256"], requested)
            if key not in groups:
                groups[key] = metrics["generated_token_ids"]
            assert groups[key] == metrics["generated_token_ids"]
            generated += requested
            runs[run["mode"]].append(metrics)
        assert all(len(r) == 2 for r in runs.values())
        changes = {}
        for key, recorded in summary["fraction_changes"].items():
            change = statistics.median(r[key] for r in runs["arena"]) / statistics.median(r[key] for r in runs["control"]) - 1
            assert abs(change - recorded) < 1e-12
            changes[key] = change
        assert changes["decode_tokens_per_second"] >= .05
        for key, limit in (("time_to_first_token_seconds", .05), ("decode_latency_p95_seconds", .10),
                           ("peak_memory_bytes", .15), ("expert_bytes_read", .05)):
            assert changes[key] <= limit
        assert changes["expert_bytes_read"] == 0
        waves.append({"path": str(path.relative_to(root)), "changes": changes})
    assert len(waves) == 14
    cold = []
    for tokens in (1, 2, 3, 4, 8, 16):
        path = root / "matrix" / f"cold-{tokens}"
        a, b = read(path / "control.json"), read(path / "arena.json")
        assert a["generated_token_ids"] == b["generated_token_ids"]
        assert a["generated_tokens"] == b["generated_tokens"] == tokens
        assert a["prompt_token_sha256"] == b["prompt_token_sha256"]
        assert a["expert_bytes_read"] == b["expert_bytes_read"]
        peak = b["peak_memory_bytes"] / a["peak_memory_bytes"] - 1
        assert peak <= .15
        cold.append({"tokens": tokens, "peak_change": peak})
    lifecycle = root / "matrix/lifecycle"
    assert read(lifecycle / "summary.json")["status"] == "passed-lifecycle-gate"
    for mode in ("control", "arena"):
        seed, restart = read(lifecycle / f"{mode}-seed.json"), read(lifecycle / f"{mode}-restart.json")
        rows = {r["name"]: r for r in seed["runs"] + restart["runs"]}
        base = rows["cold"]["token_ids"]
        for name in ("memory-repeat", "after-cancel", "after-one-output", "restart"):
            assert rows[name]["token_ids"] == base
        assert rows["decode-cancel"]["token_ids"] == base[:5]
        assert rows["one-output"]["token_ids"] == base[:1]
        assert all(r["dispatch_reset"] for r in rows.values())
    for name in ("layer0.json", "layer23.json", "layer47.json", "boundaries.json"):
        assert read(root / name)["screen_pass"]
    result = {"status": "passed-independent-audit", "formal_waves": len(waves), "formal_requests": 4 * len(waves),
              "formal_generated_tokens": generated, "cold_requests": 12, "lifecycle_requests": 16,
              "cross_wave_output_groups": len(groups), "waves": waves, "cold": cold,
              "lifecycle_prefix_consistency": True,
              "scope": "M2 Max research only; no production adoption, M5 Pro or energy result"}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("waves", "cold")}), flush=True)


if __name__ == "__main__":
    main()
