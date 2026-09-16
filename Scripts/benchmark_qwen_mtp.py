#!/usr/bin/env python3
from __future__ import annotations
if __package__:
    from .archived_evidence import archived_path
else:
    from archived_evidence import archived_path

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from statistics import median


WORKLOADS = ("code", "mixed_math", "repeated", "tool_like", "zh_technical")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        text=True,
    ).strip()


def _run(
    model: Path,
    prompt: Path,
    max_tokens: int,
    mtp: bool,
    mtp_slots: int,
    metrics_path: Path,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "deepseek_v4_ssd.cli",
        "--model",
        str(model),
        "--prompt-file",
        str(prompt),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--no-persistent-prompt-cache",
        "--metrics-json",
        str(metrics_path),
    ]
    if mtp:
        command.extend(("--mtp", "--mtp-slots", str(mtp_slots)))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "runtime"
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    keys = (
        "prompt_tokens",
        "generated_tokens",
        "generated_token_ids",
        "prompt_token_sha256",
        "token_sha256",
        "seconds",
        "tokens_per_second",
        "request_seconds",
        "time_to_first_token_seconds",
        "decode_tokens_per_second",
        "request_expert_bytes_read",
        "peak_memory_bytes",
        "decode_latency_p95_seconds",
        "mtp_rounds",
        "mtp_proposed_tokens",
        "mtp_accepted_tokens",
        "mtp_committed_tokens",
        "mtp_rejected_tokens",
        "mtp_acceptance_rate",
        "mtp_average_accepted_length",
        "mtp_draft_seconds",
        "mtp_verification_seconds",
        "mtp_replay_seconds",
        "mtp_fallback",
        "mtp_fallback_rounds",
        "mtp_expert_bytes_read",
    )
    return {key: metrics[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare normal and MTP greedy Qwen output tokens"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
    )
    parser.add_argument("--prompt-suffix", default="128")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--mtp-slots", type=int, default=32)
    parser.add_argument(
        "--workloads",
        default=",".join(WORKLOADS),
        help="Comma-separated workload names",
    )
    parser.add_argument(
        "--run-order",
        default="control,candidate",
        help="Comma-separated control and candidate runs",
    )
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.prompts is None:
        arguments.prompts = archived_path("docs/benchmarks/prompts/2026-08-26-adaptive-128")

    if arguments.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if arguments.mtp_slots < 10:
        parser.error("--mtp-slots must be at least 10")
    workloads = tuple(part.strip() for part in arguments.workloads.split(","))
    if not workloads or any(part not in WORKLOADS for part in workloads):
        parser.error(f"--workloads accepts: {','.join(WORKLOADS)}")
    run_order = tuple(part.strip() for part in arguments.run_order.split(","))
    if not run_order or any(
        part not in {"control", "candidate"} for part in run_order
    ):
        parser.error("--run-order accepts only control and candidate")
    if not {"control", "candidate"}.issubset(run_order):
        parser.error("--run-order must include control and candidate")
    formal_order = ("control", "candidate", "candidate", "control")
    if arguments.formal and run_order != formal_order:
        parser.error("--formal requires control,candidate,candidate,control")

    results = []
    with tempfile.TemporaryDirectory(prefix="qwen-mtp-") as directory:
        temporary = Path(directory)
        for name in workloads:
            prompt = arguments.prompts / f"{name}-{arguments.prompt_suffix}.txt"
            if not prompt.is_file():
                parser.error(f"prompt does not exist: {prompt}")
            runs = []
            for index, mode in enumerate(run_order, start=1):
                metrics = _run(
                    arguments.model,
                    prompt,
                    arguments.max_tokens,
                    mode == "candidate",
                    arguments.mtp_slots,
                    temporary / f"{name}-{mode}-{index}.json",
                )
                runs.append({"mode": mode, "metrics": metrics})
            normal = next(
                run["metrics"] for run in runs if run["mode"] == "control"
            )
            mtp = next(
                run["metrics"] for run in runs if run["mode"] == "candidate"
            )
            reference_tokens = runs[0]["metrics"]["generated_token_ids"]
            token_ids_equal = all(
                run["metrics"]["generated_token_ids"] == reference_tokens
                for run in runs[1:]
            )
            results.append(
                {
                    "name": name,
                    "prompt": str(prompt),
                    "prompt_sha256": _sha256(prompt),
                    "token_ids_equal": token_ids_equal,
                    "normal": normal,
                    "mtp": mtp,
                    "runs": runs,
                }
            )

    passed = all(result["token_ids_equal"] for result in results)
    artifact = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_kind": (
            (
                "qwen_mtp_formal_performance"
                if workloads == WORKLOADS
                else "qwen_mtp_formal_performance_stop_gate"
            )
            if arguments.formal
            else "qwen_mtp_greedy_output_parity"
        ),
        "formal_performance_result": arguments.formal,
        "status": "passed" if passed else "failed",
        "source": {
            "commit": _git("rev-parse", "HEAD"),
            "working_tree_dirty": bool(_git("status", "--porcelain")),
        },
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "conditions": {
            "model": str(arguments.model.resolve()),
            "prompts": str(arguments.prompts),
            "workloads": list(workloads),
            "maximum_output_tokens": arguments.max_tokens,
            "mtp_slots": arguments.mtp_slots,
            "temperature": 0,
            "persistent_prompt_cache": False,
            "run_order": list(run_order),
            "process_state": "fresh process for every run",
            "os_page_cache": "uncontrolled; alternating run order limits order bias",
            "evidence_limit": (
                (
                    "This formal stop gate uses the selected workloads only. "
                    "The remaining workloads are not required after a stop decision."
                    if workloads != WORKLOADS
                    else "This matrix validates greedy output parity."
                )
                if arguments.formal
                else "This matrix validates greedy output parity. Its single-run "
                "timings are not a formal performance result."
            ),
        },
        "results": results,
        "summary": {
            "workloads": len(results),
            "matching_workloads": sum(
                result["token_ids_equal"] for result in results
            ),
            "all_output_tokens_equal": passed,
            "mtp_proposed_tokens": sum(
                run["metrics"]["mtp_proposed_tokens"]
                for result in results
                for run in result["runs"]
                if run["mode"] == "candidate"
            ),
            "mtp_accepted_tokens": sum(
                run["metrics"]["mtp_accepted_tokens"]
                for result in results
                for run in result["runs"]
                if run["mode"] == "candidate"
            ),
        },
    }
    proposed = artifact["summary"]["mtp_proposed_tokens"]
    accepted = artifact["summary"]["mtp_accepted_tokens"]
    artifact["summary"]["mtp_acceptance_rate"] = (
        accepted / proposed if proposed else 0.0
    )
    if arguments.formal:
        comparisons = []
        for result in results:
            controls = [
                run["metrics"]
                for run in result["runs"]
                if run["mode"] == "control"
            ]
            candidates = [
                run["metrics"]
                for run in result["runs"]
                if run["mode"] == "candidate"
            ]

            def middle(items: list[dict], key: str) -> float:
                return median(float(item[key]) for item in items)

            def improvement(control: float, candidate: float) -> float:
                return (control - candidate) / control * 100 if control else 0.0

            def increase(control: float, candidate: float) -> float:
                return (candidate - control) / control * 100 if control else 0.0

            control_request = middle(controls, "request_seconds")
            candidate_request = middle(candidates, "request_seconds")
            control_decode = middle(controls, "decode_tokens_per_second")
            candidate_decode = middle(candidates, "decode_tokens_per_second")
            control_ttft = middle(controls, "time_to_first_token_seconds")
            candidate_ttft = middle(candidates, "time_to_first_token_seconds")
            control_peak = middle(controls, "peak_memory_bytes")
            candidate_peak = middle(candidates, "peak_memory_bytes")
            control_p95 = middle(controls, "decode_latency_p95_seconds")
            candidate_p95 = middle(candidates, "decode_latency_p95_seconds")
            control_bytes = median(
                item["request_expert_bytes_read"] / item["generated_tokens"]
                for item in controls
            )
            candidate_bytes = median(
                (
                    item["request_expert_bytes_read"]
                    + item["mtp_expert_bytes_read"]
                )
                / item["generated_tokens"]
                for item in candidates
            )
            comparison = {
                "name": result["name"],
                "request_improvement_percent": improvement(
                    control_request, candidate_request
                ),
                "decode_throughput_improvement_percent": increase(
                    control_decode, candidate_decode
                ),
                "ttft_increase_percent": increase(control_ttft, candidate_ttft),
                "peak_memory_increase_percent": increase(
                    control_peak, candidate_peak
                ),
                "expert_bytes_per_generated_token_increase_percent": increase(
                    control_bytes, candidate_bytes
                ),
                "p95_token_latency_increase_percent": increase(
                    control_p95, candidate_p95
                ),
            }
            result["comparison"] = comparison
            comparisons.append(comparison)

        summary = artifact["summary"]
        for key in (
            "request_improvement_percent",
            "decode_throughput_improvement_percent",
            "ttft_increase_percent",
            "peak_memory_increase_percent",
            "expert_bytes_per_generated_token_increase_percent",
            "p95_token_latency_increase_percent",
        ):
            summary[f"paired_median_{key}"] = median(
                comparison[key] for comparison in comparisons
            )
        request_improvement = summary[
            "paired_median_request_improvement_percent"
        ]
        expert_increase = summary[
            "paired_median_expert_bytes_per_generated_token_increase_percent"
        ]
        counters_reconcile = all(
            metrics["mtp_proposed_tokens"]
            == metrics["mtp_accepted_tokens"] + metrics["mtp_rejected_tokens"]
            and metrics["mtp_committed_tokens"]
            == metrics["mtp_accepted_tokens"] + metrics["mtp_rounds"]
            for result in results
            for run in result["runs"]
            if run["mode"] == "candidate"
            for metrics in (run["metrics"],)
        )
        gates = {
            "correctness": passed,
            "mtp_counters_reconcile": counters_reconcile,
            "request_improvement_at_least_5_percent": request_improvement >= 5,
            "decode_improvement_at_least_5_percent": summary[
                "paired_median_decode_throughput_improvement_percent"
            ] >= 5,
            "ttft_increase_at_most_5_percent": summary[
                "paired_median_ttft_increase_percent"
            ] <= 5,
            "peak_memory_increase_at_most_15_percent": summary[
                "paired_median_peak_memory_increase_percent"
            ] <= 15,
            "expert_bytes_gate": expert_increase <= 5 or request_improvement >= 10,
            "p95_latency_increase_at_most_10_percent": summary[
                "paired_median_p95_token_latency_increase_percent"
            ] <= 10,
        }
        summary["adoption_gates"] = gates
        summary["decision"] = "adopt" if all(gates.values()) else "stop"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit("Qwen MTP output parity failed")


if __name__ == "__main__":
    main()
