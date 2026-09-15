from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.model import RuntimeConfig

try:
    from .benchmark_common import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )
except ImportError:
    from benchmark_common import (
        _command_output,
        _runtime_tree_sha256,
        _sha256,
    )


MODE = "learned-route-drop-lowest-1"
SUITES = ("quality", "safety")
CHECKS = ("contains", "not_contains")


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported approximate suite schema")
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("mode") != MODE:
        raise ValueError("suite candidate mode does not match benchmark mode")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("approximate suite has no cases")
    identifiers = set()
    suite_counts = {suite: 0 for suite in SUITES}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("suite case must be an object")
        identifier = case.get("id")
        suite = case.get("suite")
        messages = case.get("messages")
        checks = case.get("checks")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("suite case IDs must be unique non-empty strings")
        if suite not in SUITES:
            raise ValueError(f"unknown suite for {identifier}")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"case {identifier} has no messages")
        if not isinstance(checks, list) or not checks:
            raise ValueError(f"case {identifier} has no checks")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in ("system", "user")
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(f"case {identifier} has an invalid message")
        for check in checks:
            if (
                not isinstance(check, dict)
                or check.get("kind") not in CHECKS
                or not isinstance(check.get("value"), str)
                or not check["value"]
            ):
                raise ValueError(f"case {identifier} has an invalid check")
        identifiers.add(identifier)
        suite_counts[suite] += 1
    if any(count != 5 for count in suite_counts.values()):
        raise ValueError("entry smoke requires five quality and five safety cases")
    return cases


def _evaluate_checks(text: str, checks: list[dict[str, str]]) -> dict[str, Any]:
    results = []
    for check in checks:
        found = check["value"] in text
        passed = found if check["kind"] == "contains" else not found
        results.append({**check, "passed": passed})
    return {"checks": results, "passed": all(row["passed"] for row in results)}


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    count = 0
    for left, right in zip(first, second):
        if left != right:
            break
        count += 1
    return count


def _comparison(exact: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    paired = min(len(exact["generated_token_ids"]), len(candidate["generated_token_ids"]))
    matches = sum(
        left == right
        for left, right in zip(
            exact["generated_token_ids"], candidate["generated_token_ids"]
        )
    )
    return {
        "common_prefix_tokens": _common_prefix_length(
            exact["generated_token_ids"], candidate["generated_token_ids"]
        ),
        "aligned_token_agreement_fraction": matches / paired if paired else None,
    }


def _apply_candidate(runtime: ModelRuntime, exact_top_k: int, candidate_top_k: int) -> dict[str, int]:
    core = getattr(runtime.model, "model", runtime.model)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise ValueError("installed model does not expose DeepSeek layers")
    changed = 0
    unchanged_hash = 0
    for layer in layers:
        gate = layer.ffn.gate
        if getattr(gate, "hash", False):
            unchanged_hash += 1
            continue
        if gate.top_k != exact_top_k:
            raise ValueError("learned router does not match exact top-k contract")
        gate.top_k = candidate_top_k
        changed += 1
    if changed != 40 or unchanged_hash != 3:
        raise ValueError("installed model layer split does not match candidate contract")
    runtime._prompt_caches.clear()
    return {"changed_learned_layers": changed, "unchanged_hash_layers": unchanged_hash}


def _run_case(
    runtime: ModelRuntime,
    case: dict[str, Any],
    options: GenerationOptions,
) -> dict[str, Any]:
    runtime._prompt_caches.clear()
    prompt = runtime.encode_chat(case["messages"], thinking_mode="chat")
    started = time.perf_counter()
    pieces = list(runtime.stream(prompt, options))
    wall_seconds = time.perf_counter() - started
    tokens = [int(piece.token) for piece in pieces]
    text = "".join(piece.text for piece in pieces)
    evaluation = _evaluate_checks(text, case["checks"])
    return {
        "id": case["id"],
        "suite": case["suite"],
        "generated_token_ids": tokens,
        "token_sha256": hashlib.sha256(
            ",".join(map(str, tokens)).encode("utf-8")
        ).hexdigest(),
        "output_text": text,
        "wall_seconds_descriptive": wall_seconds,
        "evaluation": evaluation,
    }


def _suite_gate(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        suite: all(
            row["evaluation"]["passed"] for row in rows if row["suite"] == suite
        )
        for suite in SUITES
    }


def _source_state(project_root: Path, protocol: Path, manifest: Path) -> dict[str, Any]:
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
        "protocol_sha256": _sha256(protocol),
        "suite_manifest_sha256": _sha256(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 6A learned-router expert-drop entry smoke"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(arguments.model).expanduser().resolve()
    suite_path = Path(arguments.suite).expanduser().resolve()
    protocol_path = Path(arguments.protocol).expanduser().resolve()
    output_path = Path(arguments.output).expanduser().resolve()
    manifest = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = _validate_manifest(manifest)
    generation = manifest["generation"]
    options = GenerationOptions(
        max_tokens=int(generation["max_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
    )
    config = RuntimeConfig(
        persistent_prompt_cache=False,
        layer_major_prefill=False,
        dspark_enabled=False,
    )

    with ModelRuntime.open(str(model_path), config) as runtime:
        if runtime.model_id != manifest["model_id"]:
            raise ValueError("installed model does not match suite model ID")
        exact_rows = [_run_case(runtime, case, options) for case in cases]
        applied = _apply_candidate(
            runtime,
            int(manifest["candidate"]["exact_learned_experts_per_token"]),
            int(manifest["candidate"]["candidate_learned_experts_per_token"]),
        )
        candidate_rows = [_run_case(runtime, case, options) for case in cases]
        installed = {
            "model_id": runtime.installed.model_id,
            "revision": runtime.installed.revision,
            "manifest_sha256": _sha256(model_path / "manifest.json"),
        }

    exact_gate = _suite_gate(exact_rows)
    candidate_gate = _suite_gate(candidate_rows)
    comparisons = [
        {"id": exact["id"], **_comparison(exact, candidate)}
        for exact, candidate in zip(exact_rows, candidate_rows)
    ]
    criteria = {
        "exact_quality_all_passed": exact_gate["quality"],
        "exact_safety_all_passed": exact_gate["safety"],
        "candidate_quality_all_passed": candidate_gate["quality"],
        "candidate_safety_all_passed": candidate_gate["safety"],
        "candidate_added_no_failed_cases": all(
            not exact["evaluation"]["passed"]
            or candidate["evaluation"]["passed"]
            for exact, candidate in zip(exact_rows, candidate_rows)
        ),
    }
    passed = all(criteria.values())
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "approximate_entry_smoke",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol_path, suite_path),
        "installed_model": installed,
        "candidate": {**manifest["candidate"], **applied},
        "method": {
            "exact_and_candidate_share_one_loaded_runtime": True,
            "prompt_cache_cleared_before_each_case": True,
            "expert_cache_not_balanced": True,
            "timing_is_descriptive_only": True,
            "runtime_api_changed": False,
        },
        "exact": exact_rows,
        "candidate_rows": candidate_rows,
        "comparisons": comparisons,
        "gate": {"criteria": criteria, "passed": passed},
        "decision": {
            "phase_6a": "pass" if passed else "stop_candidate",
            "continue_to_phase_6b": passed,
            "runtime_changed": False,
        },
        "limits": [
            "This ten-case entry smoke is not a broad capability or safety evaluation.",
            "Shared expert-cache state makes all timing and byte counters non-comparable.",
            "Token agreement is descriptive and does not measure semantic quality.",
            "The theoretical selected-expert reduction is not measured SSD traffic.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(artifact, file, indent=2, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()
