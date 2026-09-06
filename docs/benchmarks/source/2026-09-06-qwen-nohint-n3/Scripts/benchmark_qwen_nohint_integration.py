"""N3: exercise the real runtime toggle and previously saved cache states."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark_research_baseline import digest
from prepare_r0_prompts import token_sha256


def observed_cli(record, cli_args):
    from deepseek_v4_ssd import qwen4_exp as qwen
    from deepseek_v4_ssd.cli import main as cli_main
    from deepseek_v4_ssd.generation import ModelRuntime
    original_sort, original_close = qwen._gather_sort, ModelRuntime.close
    calls = 0
    state = None

    def counted_sort(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sort(*args, **kwargs)

    def close(runtime):
        nonlocal state
        controller = getattr(runtime.model, "ane_prefill", None)
        state = controller.snapshot() if controller is not None else None
        return original_close(runtime)

    qwen._gather_sort, ModelRuntime.close = counted_sort, close
    status = "failed"
    try:
        sys.argv = ["N3 runtime toggle", *cli_args]
        cli_main()
        status = "completed"
    finally:
        qwen._gather_sort, ModelRuntime.close = original_sort, original_close
        record.write_text(json.dumps(dict(status=status, sorting_calls=calls, ane_state=state,
            script_sha256=digest(Path(__file__)), qwen_source_sha256=digest(Path(qwen.__file__))), indent=2)+"\n")


def performance_summary(runs):
    if len(runs) != 8:
        raise ValueError("need all eight N3 performance runs")
    pairs = []
    for case in ("code-1024", "code-16384"):
        for wave in (0, 1):
            rows = {r["variant"]: r for r in runs if r["case"] == case and r["wave"] == wave}
            if set(rows) != {"control", "candidate"}:
                raise ValueError("missing N3 pair")
            a, b = rows["control"]["metrics"], rows["candidate"]["metrics"]
            pairs.append(dict(case=case, wave=wave,
                ttft_reduction=1-b["time_to_first_token_seconds"]/a["time_to_first_token_seconds"],
                decode_change=b["decode_tokens_per_second"]/a["decode_tokens_per_second"]-1,
                p95_change=b["decode_latency_p95_seconds"]/a["decode_latency_p95_seconds"]-1,
                mlx_memory_change=b["peak_memory_bytes"]/a["peak_memory_bytes"]-1,
                rss_change=rows["candidate"]["max_rss_bytes"]/rows["control"]["max_rss_bytes"]-1))
    keys = ("ttft_reduction", "decode_change", "p95_change", "mlx_memory_change", "rss_change")
    per_case = [dict(case=case, **{k: statistics.median(r[k] for r in pairs if r["case"] == case) for k in keys})
                for case in ("code-1024", "code-16384")]
    checks = dict(ttft=all(r["ttft_reduction"] >= .05 for r in per_case),
        decode=all(r["decode_change"] >= -.05 for r in per_case),
        p95=all(r["p95_change"] <= .10 for r in per_case),
        memory=all(r["mlx_memory_change"] <= .05 and r["rss_change"] <= .05 for r in per_case))
    return dict(pairs=pairs, per_case=per_case, checks=checks, passed=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n1", type=Path)
    parser.add_argument("--n2", type=Path)
    parser.add_argument("--cli-record", type=Path)
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cli_record:
        observed_cli(args.cli_record, args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args)
        return
    if not (args.output and args.n1 and args.n2):
        parser.error("--output, --n1 and --n2 are required")
    from analyze_qwen_nohint_validation import summarize
    n1, n2 = json.loads(args.n1.read_text()), json.loads(args.n2.read_text())
    assert summarize(n1)["passed"] and n2["status"] == "completed"
    root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = [*sorted((root / "runtime/deepseek_v4_ssd").glob("*.py")), Path(__file__),
        root / "Scripts/benchmark_qwen_nohint_cache.py", root / "Scripts/research_qwen_sorted_experts.py",
        root / "Scripts/analyze_qwen_nohint_validation.py", root / "Scripts/benchmark_research_baseline.py",
        root / "Scripts/prepare_r0_prompts.py", root / "research/QWEN_NOHINT_INTEGRATION_2026-09-06.md",
        root / "runtime/tests/test_qwen_grouped_experts.py",
        root / "runtime/tests/test_qwen_nohint_validation.py",
        root / "Native/ANEBridge/WhallmANE.m", root / "Native/ANEBridge/LICENSE",
        root / "Scripts/build-ane-bridge.sh", root / ".build/native/libWhallmANE.dylib"]
    source = {str(p.relative_to(root)): digest(p) for p in files}
    for path in files:
        dest = output / "source-at-start" / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    artifact = dict(status="running", formal_performance_result=False, started_at=datetime.now(timezone.utc).isoformat(),
        n1_sha256=digest(args.n1), n2_sha256=digest(args.n2), installed_model=n1["installed_model"],
        source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), files_sha256=source),
        environment=dict(platform=platform.platform(), python=sys.version,
            chip=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
            memory_bytes=int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
            packages={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "numpy")}),
        cache_state=n1["cache_state"], performance_runs=[], cache_workers=[],
        limits=["Integration regression check on the existing N1 code prompts; not a new capability evaluation.",
                "Only two reversed pairs per length, OS page cache not purged.",
                "Counts existing sort helper invocations; does not replace expert math."])
    assert digest(Path(n1["installed_model"]["path"]) / "manifest.json") == n1["installed_model"]["manifest_sha256"]

    def save():
        (output / "results.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+"\n")

    def verify_source():
        assert all(digest(root / name) == sha for name, sha in source.items())

    try:
        for wave in (0, 1):
            for case in ("code-1024", "code-16384"):
                pair = {}
                old = next(r for r in n1["runs"] if r["case"] == case and r["variant"] == "control")
                prompt = next(p for p in n1["prompts"] if p["case"] == case)
                for variant in (("control", "candidate") if wave == 0 else ("candidate", "control")):
                    verify_source()
                    assert digest(Path(prompt["path"])) == prompt["text_sha256"]
                    name = f"{wave}-{case}-{variant}"
                    stem = output / name
                    cli = old["command"][3:].copy()
                    for flag, value in (("--max-tokens", 256), ("--metrics-json", stem.with_suffix(".json"))):
                        cli[cli.index(flag)+1] = str(value)
                    cli.append("--qwen-grouped-experts" if variant == "candidate" else "--no-qwen-grouped-experts")
                    record = stem.with_suffix(".record.json")
                    command = [sys.executable, str(Path(__file__).resolve()), "--cli-record", str(record), "--", *cli]
                    print(f"START performance {len(artifact['performance_runs'])+1}/8 {name}", flush=True)
                    with stem.with_suffix(".txt").open("w") as stdout, stem.with_suffix(".stderr").open("w") as stderr:
                        subprocess.run(["/usr/bin/time", "-l", *command], cwd=root,
                            env=os.environ | {"PYTHONPATH": str(root / "runtime")},
                            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=True)
                    metrics, observation = json.loads(stem.with_suffix(".json").read_text()), json.loads(record.read_text())
                    rss = re.search(r"(\d+)\s+maximum resident set size", stem.with_suffix(".stderr").read_text())
                    row = dict(wave=wave, case=case, variant=variant, command=command, metrics=metrics,
                        observation=observation, max_rss_bytes=int(rss[1]) if rss else None,
                        text=stem.with_suffix(".txt").read_text(), text_sha256=digest(stem.with_suffix(".txt")),
                        metrics_sha256=digest(stem.with_suffix(".json")))
                    artifact["performance_runs"].append(row)
                    save()
                    assert metrics["generated_token_ids"] == old["metrics"]["generated_token_ids"][:256]
                    assert metrics["token_sha256"] == token_sha256(metrics["generated_token_ids"])
                    assert metrics["prompt_token_sha256"] == prompt["token_sha256"]
                    assert metrics["generated_tokens"] == 256 and metrics["prompt_tokens"] == prompt["length"]
                    assert metrics["approximation_mode"] == "exact" and not metrics["mtp_enabled"] and metrics["prompt_cache_reused_tokens"] == 0
                    assert metrics["qwen_grouped_experts"] == (variant == "candidate")
                    assert observation["status"] == "completed"
                    assert observation["script_sha256"] == source["Scripts/benchmark_qwen_nohint_integration.py"]
                    assert observation["qwen_source_sha256"] == source["runtime/deepseek_v4_ssd/qwen4_exp.py"]
                    assert observation["sorting_calls"] == (192 if case.endswith("1024") else 768) * (variant == "candidate")
                    assert observation["ane_state"]["active"] and not observation["ane_state"]["fallbacks"]
                    pair[variant] = row
                    print(f'DONE TTFT={metrics["time_to_first_token_seconds"]:.2f} decode={metrics["decode_tokens_per_second"]:.2f}', flush=True)
                a, b = pair["control"]["metrics"], pair["candidate"]["metrics"]
                assert all(a[k] == b[k] for k in ("generated_token_ids", "request_expert_bytes_read", "request_gather_qmm_calls"))
        artifact["performance_summary"] = performance_summary(artifact["performance_runs"])
        save()
        assert artifact["performance_summary"]["passed"], "N3 performance gate did not pass"
        cold = {b: next(w["result"]["runs"][0] for w in n2["workers"] if w["name"] == f"cold-control-{b}") for b in ("A", "B")}
        for writer in ("control", "candidate"):
            seed = next(w["result"]["plan"]["seed_copy"] for w in n2["workers"] if w["name"] == f"seed-{writer}")
            previous = Path(seed)
            for variant in ("candidate", "control"):
                verify_source()
                name = f"cache-{writer}-seed-{variant}-reader"
                cache = output / f"{name}-data"
                shutil.copytree(previous, cache)
                plan = dict(model=n1["installed_model"]["path"], manifest_sha256=n1["installed_model"]["manifest_sha256"],
                    variant=variant, integrated=True, action="restart", branches=["B", "A"], suite=n2["suite"],
                    cache_root=str(cache), seed_copy="unused")
                plan_path, result_path = output / f"{name}.plan.json", output / f"{name}.json"
                plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n")
                command = [sys.executable, str(root / "Scripts/benchmark_qwen_nohint_cache.py"),
                    "--worker-plan", str(plan_path), "--output", str(result_path)]
                print(f"START {name}", flush=True)
                with (output / f"{name}.stdout").open("w") as stdout, (output / f"{name}.stderr").open("w") as stderr:
                    completed = subprocess.run(command, cwd=root, env=os.environ | {"PYTHONPATH": str(root / "runtime")},
                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                result = json.loads(result_path.read_text()) if result_path.exists() else dict(status="failed", runs=[])
                artifact["cache_workers"].append(dict(name=name, command=command, result=result))
                save()
                assert completed.returncode == 0 and result["status"] == "completed", name
                assert result["config"]["qwen_grouped_experts"] == (variant == "candidate")
                for row in result["runs"]:
                    ref = cold[row["branch"]]
                    assert row["generated_token_ids"] == ref["generated_token_ids"] and row["text"] == ref["text"]
                    assert row["metrics"]["prompt_cache_reused_tokens"] >= 1024
                    assert (row["modified_calls"] > 0 if variant == "candidate" else row["modified_calls"] == 0)
                previous = cache
                print(f"DONE {name}", flush=True)
        artifact["status"] = "completed"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = str(error)
        raise
    finally:
        artifact["ended_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
