from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import resource
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache, _route_phase
from deepseek_v4_ssd.model import RuntimeConfig, forward_with_hidden

try:
    from .benchmark_approximate_expert_drop import (
        _apply_candidate,
        _validate_manifest,
    )
    from .benchmark_dspark_adaptive import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )
except ImportError:
    from benchmark_approximate_expert_drop import _apply_candidate, _validate_manifest
    from benchmark_dspark_adaptive import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )


KL_LIMIT = 0.02
MINIMUM_LOGICAL_BYTE_REDUCTION = 0.10
MAXIMUM_PEAK_RSS_INCREASE = 0.05


class _VariableRouteRecorder:
    def __init__(self, layer_count: int) -> None:
        self.routes: list[list[np.ndarray]] = [[] for _ in range(layer_count)]

    @contextmanager
    def phase(self, _name: str):
        yield

    def record(self, layer: int, selected: np.ndarray) -> None:
        self.routes[layer].append(np.asarray(selected, dtype=np.int32).copy())

    def record_residency(self, *_: object) -> None:
        return

    def record_prefetch_event(self, **_: object) -> None:
        return


def _route_payload(recorder: _VariableRouteRecorder) -> tuple[list[dict[str, Any]], str]:
    digest = hashlib.sha256()
    layers = []
    for layer, chunks in enumerate(recorder.routes):
        if not chunks:
            raise ValueError(f"layer {layer} did not record routes")
        values = np.concatenate([chunk.reshape(-1) for chunk in chunks])
        per_token = int(chunks[0].shape[-1])
        if any(int(chunk.shape[-1]) != per_token for chunk in chunks):
            raise ValueError(f"layer {layer} changed route width")
        digest.update(layer.to_bytes(2, "little"))
        digest.update(per_token.to_bytes(2, "little"))
        digest.update(values.tobytes())
        layers.append(
            {
                "layer": layer,
                "experts_per_token": per_token,
                "assignments": int(values.size),
                "unique_experts": int(np.unique(values).size),
            }
        )
    return layers, digest.hexdigest()


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum(dtype=np.float64)))


def _kl_divergence(exact: np.ndarray, candidate: np.ndarray) -> float:
    exact64 = np.asarray(exact, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    log_exact = exact64 - _logsumexp(exact64)
    log_candidate = candidate64 - _logsumexp(candidate64)
    probabilities = np.exp(log_exact)
    return float(np.sum(probabilities * (log_exact - log_candidate)))


def _run_mode(arguments: argparse.Namespace) -> dict[str, Any]:
    suite_path = Path(arguments.suite).expanduser().resolve()
    manifest = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = _validate_manifest(manifest)
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    mode_directory = raw_directory / arguments.worker_mode
    mode_directory.mkdir(parents=True, exist_ok=True)
    config = RuntimeConfig(
        persistent_prompt_cache=False,
        layer_major_prefill=False,
        dspark_enabled=False,
        expert_file_cache_policy="bypass",
    )
    rows = []
    with ModelRuntime.open(arguments.model, config) as runtime:
        if arguments.worker_mode == "candidate":
            _apply_candidate(runtime, 6, 5)
        for case in cases:
            runtime._prompt_caches.clear()
            prompt = runtime.encode_chat(case["messages"], thinking_mode="chat")
            tokens = runtime._encode_prompt(prompt)
            prompt_cache = _make_prompt_cache(runtime.model)
            recorder = _VariableRouteRecorder(runtime.installed.layer_count)
            runtime.expert_cache._route_trace = recorder
            before = runtime.expert_cache.metrics_snapshot()
            with _route_phase(runtime.expert_cache, "prefill"):
                logits, hidden = forward_with_hidden(
                    runtime.model,
                    mx.array([tokens], dtype=mx.int32),
                    prompt_cache,
                    (0,),
                )
            mx.eval(logits, hidden)
            values = np.asarray(logits[0, -1].astype(mx.float32), dtype=np.float32)
            if values.ndim != 1:
                raise ValueError("next-token logits are not one-dimensional")
            metrics = runtime.expert_cache.metrics_snapshot().delta(before)
            layers, route_hash = _route_payload(recorder)
            logits_path = mode_directory / f"{case['id']}.logits.npy"
            routes_path = mode_directory / f"{case['id']}.routes.json"
            np.save(logits_path, values, allow_pickle=False)
            with routes_path.open("w", encoding="utf-8") as file:
                json.dump(layers, file, separators=(",", ":"))
                file.write("\n")
            rows.append(
                {
                    "id": case["id"],
                    "suite": case["suite"],
                    "prompt_tokens": len(tokens),
                    "logits_file": str(logits_path),
                    "logits_sha256": _sha256(logits_path),
                    "logits_finite": bool(np.isfinite(values).all()),
                    "next_token_top1": int(np.argmax(values)),
                    "routes_file": str(routes_path),
                    "routes_sha256": _sha256(routes_path),
                    "route_values_sha256": route_hash,
                    "route_layers": layers,
                    "metrics": asdict(metrics),
                }
            )
            runtime.expert_cache._route_trace = None
            del logits, hidden, prompt_cache
            mx.clear_cache()
        installed = {
            "model_id": runtime.installed.model_id,
            "revision": runtime.installed.revision,
            "manifest_sha256": _sha256(runtime.installed.root / "manifest.json"),
        }
    return {
        "mode": arguments.worker_mode,
        "installed_model": installed,
        "rows": rows,
        "aggregate_logical_expert_bytes": sum(
            int(row["metrics"]["bytes_read"]) for row in rows
        ),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _run_worker(
    script: Path,
    *,
    mode: str,
    model: Path,
    suite: Path,
    raw_directory: Path,
) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            sys.executable,
            str(script),
            "--model",
            str(model),
            "--suite",
            str(suite),
            "--raw-directory",
            str(raw_directory),
            "--worker-mode",
            mode,
        ],
        text=True,
    )
    return json.loads(output)


def _component_gate(
    exact: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if [row["id"] for row in exact["rows"]] != [
        row["id"] for row in candidate["rows"]
    ]:
        raise ValueError("exact and candidate cases do not align")
    comparisons = []
    for exact_row, candidate_row in zip(exact["rows"], candidate["rows"]):
        exact_logits = np.load(exact_row["logits_file"], allow_pickle=False)
        candidate_logits = np.load(candidate_row["logits_file"], allow_pickle=False)
        divergence = _kl_divergence(exact_logits, candidate_logits)
        comparisons.append(
            {
                "id": exact_row["id"],
                "suite": exact_row["suite"],
                "exact_top1": exact_row["next_token_top1"],
                "candidate_top1": candidate_row["next_token_top1"],
                "top1_equal": (
                    exact_row["next_token_top1"]
                    == candidate_row["next_token_top1"]
                ),
                "exact_to_candidate_kl_divergence": divergence,
                "kl_within_limit": divergence <= KL_LIMIT,
            }
        )
    exact_bytes = int(exact["aggregate_logical_expert_bytes"])
    candidate_bytes = int(candidate["aggregate_logical_expert_bytes"])
    byte_reduction = 1 - candidate_bytes / exact_bytes
    peak_rss_change = (
        int(candidate["peak_rss_bytes"]) / int(exact["peak_rss_bytes"]) - 1
    )
    route_contract = all(
        all(layer["experts_per_token"] == 6 for layer in exact_row["route_layers"])
        and all(
            layer["experts_per_token"] == (6 if layer["layer"] < 3 else 5)
            for layer in candidate_row["route_layers"]
        )
        for exact_row, candidate_row in zip(exact["rows"], candidate["rows"])
    )
    criteria = {
        "all_logits_finite": all(
            row["logits_finite"] for result in (exact, candidate) for row in result["rows"]
        ),
        "all_next_token_top1_equal": all(row["top1_equal"] for row in comparisons),
        "all_kl_divergences_at_most_0_02": all(
            row["kl_within_limit"] for row in comparisons
        ),
        "route_width_contract_exact": route_contract,
        "logical_expert_bytes_reduced_at_least_10_percent": (
            byte_reduction >= MINIMUM_LOGICAL_BYTE_REDUCTION
        ),
        "peak_rss_increase_at_most_5_percent": (
            peak_rss_change <= MAXIMUM_PEAK_RSS_INCREASE
        ),
    }
    return comparisons, {
        "criteria": criteria,
        "logical_expert_byte_reduction_fraction": byte_reduction,
        "peak_rss_change_fraction": peak_rss_change,
        "passed": all(criteria.values()),
    }


def _source_state(project_root: Path, protocol: Path, suite: Path) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    return {
        "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
        "working_tree_dirty": bool(
            _command_output(["git", "status", "--porcelain"], project_root)
        ),
        "tracked_diff_sha256_at_run": hashlib.sha256(diff).hexdigest(),
        "runtime_python_tree_sha256": _runtime_tree_sha256(
            project_root / "runtime" / "deepseek_v4_ssd"
        ),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "protocol_sha256_at_run": _sha256(protocol),
        "suite_manifest_sha256": _sha256(suite),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6B fresh-worker component gate"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output")
    parser.add_argument("--worker-mode", choices=("exact", "candidate"))
    arguments = parser.parse_args()

    if arguments.worker_mode is not None:
        print(json.dumps(_run_mode(arguments), separators=(",", ":")))
        return
    if not arguments.protocol or not arguments.output:
        parser.error("parent mode requires --protocol and --output")

    project_root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve()
    model = Path(arguments.model).expanduser().resolve()
    suite = Path(arguments.suite).expanduser().resolve()
    protocol = Path(arguments.protocol).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    raw_directory.mkdir(parents=True, exist_ok=True)
    exact = _run_worker(
        script,
        mode="exact",
        model=model,
        suite=suite,
        raw_directory=raw_directory,
    )
    candidate = _run_worker(
        script,
        mode="candidate",
        model=model,
        suite=suite,
        raw_directory=raw_directory,
    )
    comparisons, gate = _component_gate(exact, candidate)
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "approximate_expert_drop_component_gate",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol, suite),
        "installed_model": exact["installed_model"],
        "method": {
            "fresh_worker_per_mode": True,
            "case_order_identical": True,
            "expert_cache_starts_empty_per_mode": True,
            "expert_file_cache_policy": "bypass",
            "timing_is_not_a_gate": True,
        },
        "exact": exact,
        "candidate": candidate,
        "comparisons": comparisons,
        "gate": gate,
        "decision": {
            "phase_6b": "pass" if gate["passed"] else "stop_candidate",
            "continue_to_phase_6c": gate["passed"],
            "runtime_changed": False,
        },
        "limits": [
            "Ten short prompts do not establish broad capability or safety.",
            "Logical expert bytes are runtime reads, not physical SSD counters.",
            "The component gate does not make a performance claim.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
