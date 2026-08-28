from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.dspark import (
    _expected_committed_tokens,
    _hash_expert_prefetch_plan,
)
from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    _select_prefill_step_size,
    eval_prompt_cache,
    forward_with_hidden,
)


WORKLOAD_ORDER = (
    "repeated",
    "code",
    "zh_technical",
    "mixed_math",
    "tool_like",
)
SELECTOR_WEIGHTS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
BLOCK_SIZE = 5
BRANCH_WIDTH = 4
BEAM_WIDTH = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(
        ",".join(map(str, tokens)).encode("utf-8")
    ).hexdigest()


def _runtime_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _command_output(command: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sysctl(name: str, cwd: Path) -> str | None:
    return _command_output(["sysctl", "-n", name], cwd)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_state(project_root: Path, protocol: Path) -> dict:
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
    }


def _stable_topk(values: np.ndarray, count: int) -> list[int]:
    """Return score-descending, token-ID-ascending exact top-k indices."""
    scores = np.asarray(values)
    if scores.ndim != 1:
        raise ValueError("stable top-k requires a one-dimensional score row")
    if not 1 <= count <= scores.size:
        raise ValueError("stable top-k count is out of range")
    if not np.isfinite(scores).all():
        raise ValueError("stable top-k scores must be finite")
    if count == scores.size:
        candidates = np.arange(scores.size, dtype=np.int64)
    else:
        partition = np.argpartition(scores, scores.size - count)[-count:]
        cutoff = scores[partition].min()
        greater = np.flatnonzero(scores > cutoff)
        ties = np.flatnonzero(scores == cutoff)
        candidates = np.concatenate([greater, ties[: count - greater.size]])
    order = np.lexsort((candidates, -scores[candidates]))
    return [int(value) for value in candidates[order][:count]]


def _accepted_prefix(candidate: list[int], target: list[int]) -> int:
    accepted = 0
    for proposed, expected in zip(candidate, target):
        if proposed != expected:
            break
        accepted += 1
    return accepted


def _is_feasible_improvement(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["tokens"] != baseline["tokens"]
        and int(candidate["accepted_draft_tokens"]) >= 1
        and int(candidate["accepted_draft_tokens"])
        >= int(baseline["accepted_draft_tokens"])
        and int(candidate["requested_hash_experts"])
        <= int(baseline["requested_hash_experts"])
        and int(candidate["missing_hash_experts"])
        <= int(baseline["missing_hash_experts"])
        and (
            int(candidate["requested_hash_experts"])
            < int(baseline["requested_hash_experts"])
            or int(candidate["missing_hash_experts"])
            < int(baseline["missing_hash_experts"])
        )
    )


def _dominates(left: dict, right: dict, *, acceptance_aware: bool) -> bool:
    no_worse = (
        float(left["joint_log_probability"])
        >= float(right["joint_log_probability"])
        and int(left["requested_hash_experts"])
        <= int(right["requested_hash_experts"])
        and int(left["missing_hash_experts"])
        <= int(right["missing_hash_experts"])
    )
    strict = (
        float(left["joint_log_probability"])
        > float(right["joint_log_probability"])
        or int(left["requested_hash_experts"])
        < int(right["requested_hash_experts"])
        or int(left["missing_hash_experts"])
        < int(right["missing_hash_experts"])
    )
    if acceptance_aware:
        no_worse = no_worse and (
            int(left["accepted_draft_tokens"])
            >= int(right["accepted_draft_tokens"])
        )
        strict = strict or (
            int(left["accepted_draft_tokens"])
            > int(right["accepted_draft_tokens"])
        )
    return bool(no_worse and strict)


def _pareto_frontier(candidates: list[dict], *, acceptance_aware: bool) -> list[str]:
    return [
        candidate["id"]
        for candidate in candidates
        if not any(
            other is not candidate
            and _dominates(other, candidate, acceptance_aware=acceptance_aware)
            for other in candidates
        )
    ]


def _selector_sweep(candidates: list[dict], baseline: dict) -> list[dict]:
    selections = []
    for requested_weight in SELECTOR_WEIGHTS:
        for missing_weight in SELECTOR_WEIGHTS:
            scored = [
                (
                    float(candidate["joint_log_probability"])
                    - requested_weight
                    * int(candidate["requested_hash_experts"])
                    - missing_weight * int(candidate["missing_hash_experts"]),
                    candidate,
                )
                for candidate in candidates
            ]
            score, selected = min(
                scored,
                key=lambda item: (-item[0], tuple(item[1]["tokens"])),
            )
            selections.append(
                {
                    "requested_weight": requested_weight,
                    "missing_weight": missing_weight,
                    "selected_candidate_id": selected["id"],
                    "selected_tokens": selected["tokens"],
                    "score": score,
                    "different_from_baseline": selected["id"] != baseline["id"],
                    "selector_feasible_improvement": _is_feasible_improvement(
                        selected, baseline
                    ),
                }
            )
    return selections


def _draft_features(dspark, main_model, anchor: int, main_hidden, start_pos: int):
    main_x = dspark._main_x(main_hidden)
    input_ids = mx.full((1, dspark.block_size), dspark.noise_token_id, mx.int32)
    input_ids[:, 0] = anchor
    hidden = main_model.model.embed_tokens(input_ids)
    hidden = mx.broadcast_to(
        hidden[:, :, None, :],
        (*hidden.shape[:2], dspark.hc_mult, hidden.shape[-1]),
    )
    hidden = mx.contiguous(hidden)
    for layer in dspark.layers:
        hidden = layer(hidden, input_ids, main_x, start_pos)
    head_hidden = dspark.hc_head(hidden)
    base_logits = main_model.lm_head(dspark.norm(head_hidden)).astype(mx.float32)
    mx.eval(head_hidden, base_logits)
    return head_hidden, base_logits


class _TransitionCache:
    def __init__(self, dspark, base_logits):
        self.dspark = dspark
        self.base_logits = base_logits
        self.rows: dict[tuple[int, int], np.ndarray] = {}

    def get_many(self, position: int, previous_tokens: list[int]) -> list[np.ndarray]:
        missing = sorted(
            {
                int(token)
                for token in previous_tokens
                if (position, int(token)) not in self.rows
            }
        )
        if missing:
            previous = mx.array(missing, dtype=mx.int32)
            bias, _ = self.dspark.markov_head(previous)
            logits = self.base_logits[:, position] + bias
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            mx.eval(logprobs)
            values = np.asarray(logprobs, dtype=np.float32)
            for token, row in zip(missing, values):
                self.rows[(position, token)] = row
        return [self.rows[(position, int(token))] for token in previous_tokens]


def _sequential_greedy_path(
    transitions: _TransitionCache,
    anchor: int,
    block_size: int,
) -> list[int]:
    tokens = []
    previous = anchor
    for position in range(block_size):
        row = transitions.get_many(position, [previous])[0]
        token = _stable_topk(row, 1)[0]
        tokens.append(token)
        previous = token
    return tokens


def _beam_paths(
    transitions: _TransitionCache,
    anchor: int,
    block_size: int,
    branch_width: int,
    beam_width: int,
) -> list[tuple[list[int], float]]:
    beam: list[tuple[list[int], float]] = [([], 0.0)]
    for position in range(block_size):
        previous = [path[-1] if path else anchor for path, _ in beam]
        rows = transitions.get_many(position, previous)
        expanded = []
        for (path, joint), row in zip(beam, rows):
            for token in _stable_topk(row, branch_width):
                expanded.append(([*path, token], joint + float(row[token])))
        beam = sorted(expanded, key=lambda item: (-item[1], tuple(item[0])))[
            :beam_width
        ]
    return beam


def _path_distribution(
    transitions: _TransitionCache,
    anchor: int,
    tokens: list[int],
) -> tuple[float, list[float], list[list[int]], bool]:
    previous = [anchor, *tokens[:-1]]
    rows = transitions.get_many(0, [previous[0]])
    for position in range(1, len(tokens)):
        rows.extend(transitions.get_many(position, [previous[position]]))
    values = [float(row[token]) for row, token in zip(rows, tokens)]
    top_tokens = [_stable_topk(row, BRANCH_WIDTH) for row in rows]
    return (
        sum(values),
        values,
        top_tokens,
        all(token in top for token, top in zip(tokens, top_tokens)),
    )


def _path_confidence(dspark, head_hidden, anchor: int, tokens: list[int]) -> list[float]:
    previous = mx.array([[anchor, *tokens[:-1]]], dtype=mx.int32)
    embeddings = dspark.markov_head.markov_w1(previous)
    confidence = mx.sigmoid(dspark.confidence_head(head_hidden, embeddings))[0]
    mx.eval(confidence)
    return [float(value) for value in confidence.tolist()]


def _context_arrays(state) -> list[mx.array]:
    arrays = []
    for layer in state or ():
        if layer is None:
            continue
        values, _ = layer
        arrays.extend(values)
    return arrays


def _seed_first_round(runtime: ModelRuntime, prompt_tokens: list[int]):
    dspark = runtime.model.dspark
    prompt_cache = _make_prompt_cache(runtime.model)
    dspark.reset_cache()
    processed = 0
    step_size = _select_prefill_step_size(
        runtime.config.prefill_step_size,
        len(prompt_tokens),
    )
    while len(prompt_tokens) - processed > 1:
        count = min(step_size, len(prompt_tokens) - processed - 1)
        inputs = mx.array(prompt_tokens[processed : processed + count])[None]
        final_logits, final_hidden = forward_with_hidden(
            runtime.model,
            inputs,
            prompt_cache,
            dspark.target_layers,
        )
        dspark.prefill_context(final_hidden, processed)
        mx.eval(final_logits, final_hidden)
        processed += count
        mx.clear_cache()
    final_logits, final_hidden = forward_with_hidden(
        runtime.model,
        mx.array([[prompt_tokens[-1]]], dtype=mx.int32),
        prompt_cache,
        dspark.target_layers,
    )
    dspark.prefill_context(final_hidden, processed)
    mx.eval(final_logits, final_hidden)
    first = int(mx.argmax(final_logits[:, -1], axis=-1).item())
    logits, main_hidden = forward_with_hidden(
        runtime.model,
        mx.array([[first]], dtype=mx.int32),
        prompt_cache,
        dspark.target_layers,
    )
    mx.eval(logits, main_hidden)
    eval_prompt_cache(prompt_cache, logits, main_hidden)
    anchor = int(mx.argmax(logits[:, -1], axis=-1).item())
    return prompt_cache, first, anchor, main_hidden, step_size


def _target_truth(
    main_model,
    prompt_cache,
    target_layers: tuple[int, ...],
    anchor: int,
    length: int,
) -> list[int]:
    current = anchor
    truth = []
    for _ in range(length):
        logits, hidden = forward_with_hidden(
            main_model,
            mx.array([[current]], dtype=mx.int32),
            prompt_cache,
            target_layers,
        )
        eval_prompt_cache(prompt_cache, logits, hidden)
        current = int(mx.argmax(logits[:, -1], axis=-1).item())
        truth.append(current)
    return truth


def _worker_config() -> RuntimeConfig:
    return RuntimeConfig(
        persistent_prompt_cache=False,
        layer_major_prefill=False,
        dspark_enabled=True,
        dspark_confidence_threshold=0.0,
        dspark_hash_prefetch=False,
        dspark_adaptive_block=False,
        dspark_fallback_enabled=False,
        expert_page_cache_probe=False,
        expert_file_cache_policy="cached",
    )


def _run_worker(arguments: argparse.Namespace) -> None:
    model = Path(arguments.model).expanduser().resolve()
    prompt_path = Path(arguments.prompt_file).expanduser().resolve()
    output = Path(arguments.worker_output).expanduser().resolve()
    if _sha256(prompt_path) != arguments.expected_text_sha256:
        raise RuntimeError("prompt text SHA-256 does not match manifest")
    prompt = prompt_path.read_text(encoding="utf-8")
    runtime = ModelRuntime.open(str(model), _worker_config())
    try:
        prompt_tokens = runtime._encode_prompt(prompt)
        prompt_hash = _token_sha256(prompt_tokens)
        if len(prompt_tokens) != arguments.expected_prompt_tokens:
            raise RuntimeError("prompt token count does not match manifest")
        if prompt_hash != arguments.expected_prompt_token_sha256:
            raise RuntimeError("prompt token SHA-256 does not match manifest")
        dspark = runtime.model.dspark
        if dspark is None:
            raise RuntimeError("installed model did not load DSpark")
        if dspark.block_size != BLOCK_SIZE:
            raise RuntimeError(
                f"expected DSpark block size {BLOCK_SIZE}, got {dspark.block_size}"
            )

        prompt_cache, first, anchor, main_hidden, step_size = _seed_first_round(
            runtime, prompt_tokens
        )
        oracle_cache = copy.deepcopy(prompt_cache)
        eval_prompt_cache(oracle_cache)

        context_state = dspark.cache_state()
        context_arrays = _context_arrays(context_state)
        if context_arrays:
            mx.eval(*context_arrays)
        actual_started = time.perf_counter()
        actual = dspark.draft(
            runtime.model,
            anchor,
            main_hidden,
            len(prompt_tokens),
            0,
            1,
            0,
        )
        actual_wall_seconds = time.perf_counter() - actual_started
        dspark.restore_cache_state(context_state)
        head_hidden, base_logits = _draft_features(
            dspark,
            runtime.model,
            anchor,
            main_hidden,
            len(prompt_tokens),
        )
        transitions = _TransitionCache(dspark, base_logits)
        reconstructed = _sequential_greedy_path(
            transitions,
            anchor,
            BLOCK_SIZE,
        )
        beam = _beam_paths(
            transitions,
            anchor,
            BLOCK_SIZE,
            BRANCH_WIDTH,
            BEAM_WIDTH,
        )
        paths = {tuple(tokens) for tokens, _ in beam}
        paths.add(tuple(reconstructed))

        expert_blob_bytes = int(runtime.installed.expert_blob_size)
        candidates = []
        hash_layer_sets = []
        for tokens_tuple in paths:
            tokens = list(tokens_tuple)
            joint, transition_values, top_tokens, in_branch = _path_distribution(
                transitions,
                anchor,
                tokens,
            )
            confidence = _path_confidence(dspark, head_hidden, anchor, tokens)
            plan = _hash_expert_prefetch_plan(runtime.model, [anchor, *tokens])
            requested_keys = plan.keys_for_draft_tokens(BLOCK_SIZE)
            resident_keys = runtime.expert_cache.resident_expert_keys(
                plan.experts_by_layer
            )
            missing_keys = requested_keys - resident_keys
            layer_ids = [int(layer.layer) for layer in plan.layers]
            hash_layer_sets.append(layer_ids)
            candidates.append(
                {
                    "tokens": tokens,
                    "is_baseline": tokens == reconstructed,
                    "joint_log_probability": joint,
                    "transition_log_probabilities": transition_values,
                    "branch_top_tokens": top_tokens,
                    "all_transitions_in_branch_top_k": in_branch,
                    "confidence": confidence,
                    "expected_committed_tokens": _expected_committed_tokens(
                        confidence, BLOCK_SIZE
                    ),
                    "hash_layers": [
                        {
                            "layer": int(layer.layer),
                            "routes_by_position": [
                                list(routes) for routes in layer.routes_by_position
                            ],
                            "expert_union": list(layer.expert_union),
                        }
                        for layer in plan.layers
                    ],
                    "requested_hash_experts": len(requested_keys),
                    "resident_hash_experts": len(resident_keys),
                    "missing_hash_experts": len(missing_keys),
                    "requested_hash_bytes": len(requested_keys)
                    * expert_blob_bytes,
                    "missing_hash_bytes": len(missing_keys) * expert_blob_bytes,
                }
            )

        candidates.sort(
            key=lambda candidate: (
                -float(candidate["joint_log_probability"]),
                tuple(candidate["tokens"]),
            )
        )
        for index, candidate in enumerate(candidates):
            candidate["id"] = f"candidate-{index:02d}"

        target_truth = _target_truth(
            runtime.model,
            oracle_cache,
            dspark.target_layers,
            anchor,
            BLOCK_SIZE + 1,
        )
        for candidate in candidates:
            accepted = _accepted_prefix(candidate["tokens"], target_truth)
            plan = _hash_expert_prefetch_plan(
                runtime.model,
                [anchor, *candidate["tokens"]],
            )
            requested_keys = plan.keys_for_draft_tokens(BLOCK_SIZE)
            useful_keys = plan.useful_keys(accepted)
            wasted_keys = requested_keys - useful_keys
            committed = [*candidate["tokens"][:accepted], target_truth[accepted]]
            candidate.update(
                {
                    "accepted_draft_tokens": accepted,
                    "committed_tokens": committed,
                    "committed_prefix_matches_target_truth": (
                        committed == target_truth[: accepted + 1]
                    ),
                    "realized_useful_hash_experts": len(useful_keys),
                    "realized_wasted_hash_experts": len(wasted_keys),
                    "realized_useful_hash_bytes": len(useful_keys)
                    * expert_blob_bytes,
                    "realized_wasted_hash_bytes": len(wasted_keys)
                    * expert_blob_bytes,
                }
            )

        baseline = next(
            candidate for candidate in candidates if candidate["is_baseline"]
        )
        confidence_error = max(
            (
                abs(left - right)
                for left, right in zip(actual.confidence, baseline["confidence"])
            ),
            default=0.0,
        )
        selections = _selector_sweep(candidates, baseline)
        feasible_selections = [
            selection
            for selection in selections
            if selection["selector_feasible_improvement"]
        ]
        criteria = {
            "prompt_contract_exact": (
                len(prompt_tokens) == arguments.expected_prompt_tokens
                and prompt_hash == arguments.expected_prompt_token_sha256
            ),
            "dspark_block_size_exact": dspark.block_size == BLOCK_SIZE,
            "actual_draft_has_full_block": len(actual.tokens) == BLOCK_SIZE,
            "reconstructed_baseline_tokens_exact": actual.tokens == reconstructed,
            "reconstructed_baseline_confidence_close": confidence_error <= 1e-5,
            "candidate_paths_unique_and_complete": (
                len(candidates) == len({tuple(row["tokens"]) for row in candidates})
                and all(len(row["tokens"]) == BLOCK_SIZE for row in candidates)
            ),
            "all_candidate_transitions_in_branch_top_k": all(
                row["all_transitions_in_branch_top_k"] for row in candidates
            ),
            "all_hash_layers_exact": all(
                layers == [0, 1, 2] for layers in hash_layer_sets
            ),
            "sequential_target_oracle_has_bonus_position": (
                len(target_truth) == BLOCK_SIZE + 1
            ),
            "all_committed_prefixes_target_exact": all(
                row["committed_prefix_matches_target_truth"]
                for row in candidates
            ),
        }
        worker = {
            "workload": arguments.workload,
            "prompt_file": str(prompt_path),
            "prompt_tokens": len(prompt_tokens),
            "prompt_token_sha256": prompt_hash,
            "configuration": {
                "temperature": 0,
                "top_p": 1,
                "first_speculative_round_only": True,
                "block_size": BLOCK_SIZE,
                "branch_width": BRANCH_WIDTH,
                "beam_width": BEAM_WIDTH,
                "prefill_step_size": step_size,
                "main_expert_slots": runtime.config.slots,
                "dspark_expert_slots": runtime.config.dspark_slots,
                "persistent_prompt_cache": False,
                "layer_major_prefill": False,
                "hash_prefetch": False,
                "adaptive_block": False,
                "fallback": False,
                "page_cache_probe": False,
                "expert_file_cache_policy": "cached",
            },
            "seed_tokens": {
                "first_target_token": first,
                "anchor": anchor,
            },
            "target_truth_after_anchor": target_truth,
            "target_truth_sha256": _token_sha256(target_truth),
            "actual_dspark_draft": {
                "tokens": actual.tokens,
                "confidence": actual.confidence,
                "reported_seconds": actual.seconds,
                "external_wall_seconds": actual_wall_seconds,
            },
            "reconstructed_baseline_candidate_id": baseline["id"],
            "reconstructed_baseline_tokens": reconstructed,
            "baseline_confidence_max_abs_error": confidence_error,
            "expert_blob_bytes": expert_blob_bytes,
            "candidates": candidates,
            "candidate_pareto_frontier": _pareto_frontier(
                candidates, acceptance_aware=False
            ),
            "acceptance_aware_oracle_frontier": _pareto_frontier(
                candidates, acceptance_aware=True
            ),
            "selector_sweep": selections,
            "selector_feasible_improvement": bool(feasible_selections),
            "feasible_selector_rows": feasible_selections,
            "structural_gate": {
                "criteria": criteria,
                "passed": all(criteria.values()),
            },
        }
    finally:
        runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(worker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_prompt_manifest(path: Path) -> tuple[dict, list[dict]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in source["prompts"]}
    if tuple(entries) != WORKLOAD_ORDER:
        raise ValueError("prompt manifest workload order does not match protocol")
    prompts = []
    for name in WORKLOAD_ORDER:
        entry = entries[name]
        prompt_path = path.parent / entry["file"]
        if not prompt_path.is_file():
            raise ValueError(f"prompt file does not exist: {prompt_path}")
        if _sha256(prompt_path) != entry["text_sha256"]:
            raise ValueError(f"prompt text SHA-256 mismatch: {prompt_path}")
        prompts.append({**entry, "path": str(prompt_path.resolve())})
    return source, prompts


def _run_fresh_process(
    *,
    project_root: Path,
    python: Path,
    model: Path,
    prompt: dict,
    raw_directory: Path,
) -> dict:
    run_id = f"{prompt['name']}-128-first-round"
    result_path = raw_directory / f"{run_id}.json"
    stdout_path = raw_directory / f"{run_id}.stdout.txt"
    stderr_path = raw_directory / f"{run_id}.stderr.txt"
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "worker",
        "--model",
        str(model),
        "--workload",
        prompt["name"],
        "--prompt-file",
        prompt["path"],
        "--expected-text-sha256",
        prompt["text_sha256"],
        "--expected-prompt-tokens",
        str(prompt["target_tokens"]),
        "--expected-prompt-token-sha256",
        prompt["prompt_token_sha256"],
        "--worker-output",
        str(result_path),
    ]
    environment = os.environ.copy()
    runtime_path = str(project_root / "runtime")
    environment["PYTHONPATH"] = (
        runtime_path
        if not environment.get("PYTHONPATH")
        else runtime_path + os.pathsep + environment["PYTHONPATH"]
    )
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    if completed.returncode or not result_path.is_file():
        raise RuntimeError(
            f"{run_id} failed ({completed.returncode}):\n{stderr_text[-2_000:]}"
        )
    row = json.loads(result_path.read_text(encoding="utf-8"))
    row.update(
        {
            "id": run_id,
            "raw_process": {
                "command": command,
                "returncode": completed.returncode,
                "result_sha256": _sha256(result_path),
                "stdout_sha256": _sha256(stdout_path),
                "stderr_sha256": _sha256(stderr_path),
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        }
    )
    return row


def _aggregate_gate(rows: list[dict]) -> dict:
    feasible = [
        row["workload"]
        for row in rows
        if row["selector_feasible_improvement"]
    ]
    criteria = {
        "five_fresh_workers_complete": (
            len(rows) == len(WORKLOAD_ORDER)
            and tuple(row["workload"] for row in rows) == WORKLOAD_ORDER
        ),
        "all_worker_structural_gates_pass": all(
            row["structural_gate"]["passed"] for row in rows
        ),
        "all_configurations_exact": all(
            row["configuration"]["temperature"] == 0
            and row["configuration"]["top_p"] == 1
            and row["configuration"]["first_speculative_round_only"]
            and row["configuration"]["block_size"] == BLOCK_SIZE
            and row["configuration"]["branch_width"] == BRANCH_WIDTH
            and row["configuration"]["beam_width"] == BEAM_WIDTH
            and not row["configuration"]["persistent_prompt_cache"]
            and not row["configuration"]["layer_major_prefill"]
            and not row["configuration"]["hash_prefetch"]
            and not row["configuration"]["adaptive_block"]
            and not row["configuration"]["fallback"]
            and not row["configuration"]["page_cache_probe"]
            and row["configuration"]["expert_file_cache_policy"] == "cached"
            for row in rows
        ),
    }
    structurally_valid = all(criteria.values())
    continuation_passed = structurally_valid and bool(feasible)
    return {
        "structural": {
            "criteria": criteria,
            "passed": structurally_valid,
        },
        "continuation": {
            "selector_feasible_workloads": feasible,
            "selector_feasible_workload_count": len(feasible),
            "passed": continuation_passed,
        },
        "decision": (
            "invalid_stop_and_debug"
            if not structurally_valid
            else "continue_to_multi_round_runtime_prototype"
            if continuation_passed
            else "stop_current_markov_beam_candidate"
        ),
    }


def _run_orchestrator(arguments: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    protocol = project_root / "research" / "DSPARK_CANDIDATE_PATHS_2026-08-27.md"
    python = Path(arguments.python).expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if raw_directory.exists() and any(raw_directory.iterdir()):
        raise ValueError(f"raw directory must be new or empty: {raw_directory}")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    raw_directory.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_source, prompts = _load_prompt_manifest(prompt_manifest)
    installed_manifest = json.loads((model / "manifest.json").read_text())

    rows = []
    for index, prompt in enumerate(prompts, start=1):
        print(
            f"[{index}/{len(prompts)}] workload={prompt['name']}",
            flush=True,
        )
        rows.append(
            _run_fresh_process(
                project_root=project_root,
                python=python,
                model=model,
                prompt=prompt,
                raw_directory=raw_directory,
            )
        )
    gate = _aggregate_gate(rows)
    artifact = {
        "schema_version": 1,
        "recorded_at": datetime.datetime.now().astimezone().isoformat(),
        "evidence_kind": "dspark_storage_aware_candidate_path_first_round_gate",
        "formal_performance_result": False,
        "source": _source_state(project_root, protocol),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_model": _sysctl("hw.model", project_root),
            "chip": _sysctl("machdep.cpu.brand_string", project_root),
            "memory_bytes": int(_sysctl("hw.memsize", project_root) or 0),
            "python": platform.python_version(),
            "mlx": _package_version("mlx"),
            "mlx_lm": _package_version("mlx-lm"),
            "transformers": _package_version("transformers"),
        },
        "checkpoint": {
            "installed_model": str(model),
            "model_id": installed_manifest["modelID"],
            "revision": installed_manifest["revision"],
            "installed_manifest_sha256": _sha256(model / "manifest.json"),
            "layer_count": installed_manifest["layerCount"],
            "expert_count": installed_manifest["expertCount"],
            "selected_expert_count": installed_manifest["selectedExpertCount"],
            "expert_blob_bytes": installed_manifest["expertBlobSize"],
        },
        "experiment": {
            "objective": (
                "Find a selector-visible coherent DSpark path that preserves "
                "greedy acceptance while reducing exact hash-route storage"
            ),
            "prompt_manifest": str(prompt_manifest),
            "prompt_manifest_sha256": _sha256(prompt_manifest),
            "prompt_model_id": prompt_source["model_id"],
            "prompt_revision": prompt_source["revision"],
            "workloads": list(WORKLOAD_ORDER),
            "prompt_tokens": 128,
            "temperature": 0,
            "top_p": 1,
            "first_speculative_round_only": True,
            "block_size": BLOCK_SIZE,
            "branch_width": BRANCH_WIDTH,
            "beam_width": BEAM_WIDTH,
            "selector_weights": list(SELECTOR_WEIGHTS),
            "fresh_process_per_workload": True,
            "sampling_supported": False,
            "defaults_changed": False,
            "cache_state": (
                "normal cached descriptors; LFU residency captured before target truth; "
                "no OS-wide cache purge"
            ),
        },
        "gate": gate,
        "runs": rows,
        "limitations": [
            "five short first rounds are a feasibility gate, not a speed result",
            "only exact hash-router layers 0-2 are storage-scored",
            "LFU snapshot misses are not OS residency or physical SSD bytes",
            "realized acceptance is never an input to selector scores",
            "sampling is prohibited until selector proposal probabilities are derived",
            "no steady-state, long-context, power, ANE, or multi-request measurement",
        ],
    }
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "structural_passed": gate["structural"]["passed"],
                "continuation_passed": gate["continuation"]["passed"],
                "selector_feasible_workloads": gate["continuation"][
                    "selector_feasible_workloads"
                ],
                "decision": gate["decision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate storage-aware coherent DSpark candidate paths"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", required=True)
    worker.add_argument("--workload", choices=WORKLOAD_ORDER, required=True)
    worker.add_argument("--prompt-file", required=True)
    worker.add_argument("--expected-text-sha256", required=True)
    worker.add_argument("--expected-prompt-tokens", required=True, type=int)
    worker.add_argument("--expected-prompt-token-sha256", required=True)
    worker.add_argument("--worker-output", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt-manifest", required=True)
    run.add_argument("--raw-directory", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--python", default=sys.executable)
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    if parsed.command == "worker":
        _run_worker(parsed)
    else:
        _run_orchestrator(parsed)
