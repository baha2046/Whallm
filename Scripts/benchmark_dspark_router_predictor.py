from __future__ import annotations

import argparse
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
from mlx_lm.models import deepseek_v4

from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    _select_prefill_step_size,
    eval_prompt_cache,
    forward_with_hidden,
    sequential_verification_forward_with_hidden,
)


WORKLOAD_ORDER = (
    "repeated",
    "code",
    "zh_technical",
    "mixed_math",
    "tool_like",
)
FEATURE_SOURCES = (
    "dspark_layer_0_head",
    "dspark_layer_1_head",
    "dspark_layer_2_head",
    "dspark_final_norm",
)
PREDICTED_TOP_K = (6, 12, 24, 48)
MAX_CONTINUATION_TOP_K = 24
BLOCK_SIZE = 5
HASH_LAYER_COUNT = 3


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
    scores = np.asarray(values)
    if scores.ndim != 1:
        raise ValueError("stable top-k requires one-dimensional scores")
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
    eval_prompt_cache(prompt_cache, logits, main_hidden)
    anchor = int(mx.argmax(logits[:, -1], axis=-1).item())
    return prompt_cache, first, anchor, main_hidden, step_size


def _tap_dspark_features(dspark, main_model, anchor, main_hidden, start_pos):
    main_x = dspark._main_x(main_hidden)
    input_ids = mx.full((1, dspark.block_size), dspark.noise_token_id, mx.int32)
    input_ids[:, 0] = anchor
    hidden = main_model.model.embed_tokens(input_ids)
    hidden = mx.broadcast_to(
        hidden[:, :, None, :],
        (*hidden.shape[:2], dspark.hc_mult, hidden.shape[-1]),
    )
    hidden = mx.contiguous(hidden)
    features = {}
    for index, layer in enumerate(dspark.layers):
        hidden = layer(hidden, input_ids, main_x, start_pos)
        head = dspark.hc_head(hidden)
        mx.eval(hidden, head)
        features[f"dspark_layer_{index}_head"] = head
    features["dspark_final_norm"] = dspark.norm(
        features["dspark_layer_2_head"]
    )
    base_logits = main_model.lm_head(features["dspark_final_norm"]).astype(
        mx.float32
    )
    mx.eval(base_logits, *features.values())
    return features, base_logits


def _greedy_markov_tokens(dspark, base_logits, anchor: int) -> list[int]:
    previous = anchor
    tokens = []
    for position in range(dspark.block_size):
        bias, _ = dspark.markov_head(mx.array([previous], dtype=mx.int32))
        logits = base_logits[:, position] + bias
        mx.eval(logits)
        row = np.asarray(logits, dtype=np.float32)[0]
        token = _stable_topk(row, 1)[0]
        tokens.append(token)
        previous = token
    return tokens


def _corrected_router_rankings(feature, layer) -> list[list[int]]:
    gate = layer.ffn.gate
    if getattr(gate, "hash", False):
        raise ValueError("learned-router predictor received a hash router")
    normalized = layer.ffn_norm(feature)
    logits = normalized @ gate.weight.T
    scores = deepseek_v4._score_func(logits.astype(mx.float32), gate.scoring_func)
    corrected = scores + gate.e_score_correction_bias
    mx.eval(corrected)
    values = np.asarray(corrected, dtype=np.float32)[0]
    maximum = max(PREDICTED_TOP_K)
    return [_stable_topk(row, maximum) for row in values]


def _summarize_layer(
    *,
    layer: int,
    target_routes: list[list[int]],
    predicted_routes: list[list[int]],
    expert_count: int,
    expert_blob_bytes: int,
) -> dict:
    if len(target_routes) != BLOCK_SIZE or len(predicted_routes) != BLOCK_SIZE:
        raise ValueError("predictor layer does not contain five aligned positions")
    target_sets = [set(route) for route in target_routes]
    predicted_sets = [set(route) for route in predicted_routes]
    if any(len(route) != 6 for route in target_sets):
        raise ValueError("target route does not contain six unique experts")
    predicted_k = len(predicted_routes[0])
    if any(len(route) != predicted_k for route in predicted_sets):
        raise ValueError("predicted route is not a unique fixed top-k set")
    if any(
        expert < 0 or expert >= expert_count
        for route in (*target_sets, *predicted_sets)
        for expert in route
    ):
        raise ValueError("route contains an invalid expert ID")
    recovered = sum(
        len(target.intersection(predicted))
        for target, predicted in zip(target_sets, predicted_sets)
    )
    full_sets = sum(
        target.issubset(predicted)
        for target, predicted in zip(target_sets, predicted_sets)
    )
    target_union = set().union(*target_sets)
    predicted_union = set().union(*predicted_sets)
    useful = target_union.intersection(predicted_union)
    wasted = predicted_union - target_union
    missed = target_union - predicted_union
    accounting_exact = (
        len(useful) + len(missed) == len(target_union)
        and len(useful) + len(wasted) == len(predicted_union)
        and recovered <= BLOCK_SIZE * 6
    )
    return {
        "layer": layer,
        "target_assignments": BLOCK_SIZE * 6,
        "recovered_assignments": recovered,
        "full_set_positions": full_sets,
        "positions": BLOCK_SIZE,
        "target_union_experts": len(target_union),
        "predicted_union_experts": len(predicted_union),
        "useful_union_experts": len(useful),
        "wasted_union_experts": len(wasted),
        "missed_union_experts": len(missed),
        "target_union_bytes": len(target_union) * expert_blob_bytes,
        "predicted_union_bytes": len(predicted_union) * expert_blob_bytes,
        "useful_union_bytes": len(useful) * expert_blob_bytes,
        "wasted_union_bytes": len(wasted) * expert_blob_bytes,
        "missed_union_bytes": len(missed) * expert_blob_bytes,
        "assignment_recall": recovered / (BLOCK_SIZE * 6),
        "full_set_position_rate": full_sets / BLOCK_SIZE,
        "union_recall": len(useful) / len(target_union) if target_union else 0.0,
        "useful_rate": (
            len(useful) / len(predicted_union) if predicted_union else 0.0
        ),
        "accounting_exact": accounting_exact,
    }


def _summarize_candidate(
    *,
    feature_source: str,
    predicted_top_k: int,
    layers: list[dict],
) -> dict:
    totals = {
        key: sum(int(layer[key]) for layer in layers)
        for key in (
            "target_assignments",
            "recovered_assignments",
            "full_set_positions",
            "positions",
            "target_union_experts",
            "predicted_union_experts",
            "useful_union_experts",
            "wasted_union_experts",
            "missed_union_experts",
            "target_union_bytes",
            "predicted_union_bytes",
            "useful_union_bytes",
            "wasted_union_bytes",
            "missed_union_bytes",
        )
    }
    totals.update(
        {
            "feature_source": feature_source,
            "predicted_top_k": predicted_top_k,
            "learned_layers": len(layers),
            "assignment_recall": (
                totals["recovered_assignments"] / totals["target_assignments"]
            ),
            "full_set_position_rate": (
                totals["full_set_positions"] / totals["positions"]
            ),
            "union_recall": (
                totals["useful_union_experts"] / totals["target_union_experts"]
            ),
            "useful_rate": (
                totals["useful_union_experts"]
                / totals["predicted_union_experts"]
            ),
            "accounting_exact": all(layer["accounting_exact"] for layer in layers),
            "layers": layers,
        }
    )
    return totals


def _worker_config(route_output: Path) -> RuntimeConfig:
    return RuntimeConfig(
        persistent_prompt_cache=False,
        layer_major_prefill=False,
        dspark_enabled=True,
        dspark_confidence_threshold=0.0,
        dspark_hash_prefetch=False,
        dspark_adaptive_block=False,
        dspark_fallback_enabled=False,
        expert_route_trace=str(route_output),
        expert_page_cache_probe=False,
        expert_file_cache_policy="cached",
    )


def _run_worker(arguments: argparse.Namespace) -> None:
    model = Path(arguments.model).expanduser().resolve()
    prompt_path = Path(arguments.prompt_file).expanduser().resolve()
    route_output = Path(arguments.route_output).expanduser().resolve()
    output = Path(arguments.worker_output).expanduser().resolve()
    if _sha256(prompt_path) != arguments.expected_text_sha256:
        raise RuntimeError("prompt text SHA-256 does not match manifest")
    prompt = prompt_path.read_text(encoding="utf-8")
    runtime = ModelRuntime.open(str(model), _worker_config(route_output))
    try:
        prompt_tokens = runtime._encode_prompt(prompt)
        prompt_hash = _token_sha256(prompt_tokens)
        if len(prompt_tokens) != arguments.expected_prompt_tokens:
            raise RuntimeError("prompt token count does not match manifest")
        if prompt_hash != arguments.expected_prompt_token_sha256:
            raise RuntimeError("prompt token SHA-256 does not match manifest")
        dspark = runtime.model.dspark
        if dspark is None or dspark.block_size != BLOCK_SIZE:
            raise RuntimeError("installed model does not expose block-5 DSpark")
        prompt_cache, first, anchor, main_hidden, step_size = _seed_first_round(
            runtime, prompt_tokens
        )

        state = dspark.cache_state()
        state_arrays = _context_arrays(state)
        if state_arrays:
            mx.eval(*state_arrays)
        actual = dspark.draft(
            runtime.model,
            anchor,
            main_hidden,
            len(prompt_tokens),
            0,
            1,
            0,
        )
        dspark.restore_cache_state(state)
        features, base_logits = _tap_dspark_features(
            dspark,
            runtime.model,
            anchor,
            main_hidden,
            len(prompt_tokens),
        )
        reconstructed = _greedy_markov_tokens(dspark, base_logits, anchor)

        verification_inputs = mx.array([[anchor, *actual.tokens]], dtype=mx.int32)
        with runtime.expert_cache.trace_routes("decode"):
            logits, hidden, _, verification_metrics = (
                sequential_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    prompt_cache,
                    dspark.target_layers,
                )
            )
        mx.eval(logits, hidden)
        recorder = runtime.expert_cache._route_trace
        if recorder is None:
            raise RuntimeError("route recorder was not installed")
        recorder.write(route_output)
        trace = json.loads(route_output.read_text(encoding="utf-8"))

        layer_count = int(trace["layer_count"])
        expert_count = int(trace["expert_count"])
        selected_count = int(trace["selected_expert_count"])
        expert_blob_bytes = int(trace["expert_blob_size"])
        target_layers = []
        for layer in range(HASH_LAYER_COUNT, layer_count):
            routes = trace["decode_routes"][layer]
            target_layers.append(
                {
                    "layer": layer,
                    "anchor_route": routes[0] if routes else [],
                    "aligned_draft_routes": routes[1:],
                }
            )

        predictions = {}
        candidates = []
        for feature_source in FEATURE_SOURCES:
            feature = features[feature_source]
            feature_predictions = []
            for target in target_layers:
                layer_index = int(target["layer"])
                layer = runtime.model.model.pipeline_layers[layer_index]
                top48 = _corrected_router_rankings(feature, layer)
                feature_predictions.append(
                    {
                        "layer": layer_index,
                        "top_48_by_position": top48,
                    }
                )
            predictions[feature_source] = feature_predictions
            for predicted_top_k in PREDICTED_TOP_K:
                layer_rows = []
                for target, predicted in zip(target_layers, feature_predictions):
                    layer_rows.append(
                        _summarize_layer(
                            layer=int(target["layer"]),
                            target_routes=target["aligned_draft_routes"],
                            predicted_routes=[
                                row[:predicted_top_k]
                                for row in predicted["top_48_by_position"]
                            ],
                            expert_count=expert_count,
                            expert_blob_bytes=expert_blob_bytes,
                        )
                    )
                candidates.append(
                    _summarize_candidate(
                        feature_source=feature_source,
                        predicted_top_k=predicted_top_k,
                        layers=layer_rows,
                    )
                )

        expected_route_positions = BLOCK_SIZE + 1
        criteria = {
            "prompt_contract_exact": (
                len(prompt_tokens) == arguments.expected_prompt_tokens
                and prompt_hash == arguments.expected_prompt_token_sha256
            ),
            "full_draft_exact": len(actual.tokens) == BLOCK_SIZE,
            "reconstructed_draft_tokens_exact": actual.tokens == reconstructed,
            "sequential_verification_shape_exact": (
                verification_metrics.verification_mode == "sequential"
                and verification_metrics.sequential_verification_positions
                == expected_route_positions
            ),
            "trace_contract_exact": (
                layer_count == 43
                and expert_count == 256
                and selected_count == 6
                and expert_blob_bytes == runtime.installed.expert_blob_size
            ),
            "learned_layer_range_exact": (
                [row["layer"] for row in target_layers]
                == list(range(HASH_LAYER_COUNT, layer_count))
            ),
            "six_target_positions_per_layer": all(
                len(row["anchor_route"]) == selected_count
                and len(row["aligned_draft_routes"]) == BLOCK_SIZE
                and all(
                    len(route) == selected_count
                    for route in row["aligned_draft_routes"]
                )
                for row in target_layers
            ),
            "prediction_matrix_complete": (
                set(predictions) == set(FEATURE_SOURCES)
                and all(
                    len(rows) == layer_count - HASH_LAYER_COUNT
                    and all(
                        len(position) == max(PREDICTED_TOP_K)
                        and len(set(position)) == max(PREDICTED_TOP_K)
                        for row in rows
                        for position in row["top_48_by_position"]
                    )
                    for rows in predictions.values()
                )
            ),
            "all_candidate_accounting_exact": all(
                candidate["accounting_exact"] for candidate in candidates
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
                "prefill_step_size": step_size,
                "feature_sources": list(FEATURE_SOURCES),
                "predicted_top_k": list(PREDICTED_TOP_K),
                "scored_target_layers": [HASH_LAYER_COUNT, layer_count - 1],
                "alignment": "dspark_position_i_to_target_draft_input_t_i",
                "anchor_routes_scored": False,
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
            "draft_tokens": actual.tokens,
            "draft_token_sha256": _token_sha256(actual.tokens),
            "reconstructed_draft_tokens": reconstructed,
            "target_route_trace": {
                "path": str(route_output),
                "sha256": _sha256(route_output),
                "captured_positions_per_layer": expected_route_positions,
                "scored_positions_per_layer": BLOCK_SIZE,
                "scored_assignments": (
                    (layer_count - HASH_LAYER_COUNT) * BLOCK_SIZE * selected_count
                ),
                "layers": target_layers,
            },
            "feature_predictions": predictions,
            "candidates": candidates,
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


def _aggregate_candidate(rows: list[dict], feature_source: str, top_k: int) -> dict:
    workload_candidates = []
    for row in rows:
        candidate = next(
            item
            for item in row["candidates"]
            if item["feature_source"] == feature_source
            and int(item["predicted_top_k"]) == top_k
        )
        workload_candidates.append((row["workload"], candidate))
    layer_rows = []
    for layer in range(HASH_LAYER_COUNT, 43):
        matches = [
            next(item for item in candidate["layers"] if item["layer"] == layer)
            for _, candidate in workload_candidates
        ]
        target = sum(int(item["target_assignments"]) for item in matches)
        recovered = sum(int(item["recovered_assignments"]) for item in matches)
        layer_rows.append(
            {
                "layer": layer,
                "target_assignments": target,
                "recovered_assignments": recovered,
                "assignment_recall": recovered / target if target else 0.0,
            }
        )
    totals = {
        key: sum(int(candidate[key]) for _, candidate in workload_candidates)
        for key in (
            "target_assignments",
            "recovered_assignments",
            "full_set_positions",
            "positions",
            "target_union_experts",
            "predicted_union_experts",
            "useful_union_experts",
            "wasted_union_experts",
            "missed_union_experts",
            "target_union_bytes",
            "predicted_union_bytes",
            "useful_union_bytes",
            "wasted_union_bytes",
            "missed_union_bytes",
        )
    }
    workload_recall = [
        {
            "workload": workload,
            "assignment_recall": float(candidate["assignment_recall"]),
            "union_recall": float(candidate["union_recall"]),
            "useful_rate": float(candidate["useful_rate"]),
        }
        for workload, candidate in workload_candidates
    ]
    assignment_recall = totals["recovered_assignments"] / totals["target_assignments"]
    union_recall = totals["useful_union_experts"] / totals["target_union_experts"]
    useful_rate = totals["useful_union_experts"] / totals["predicted_union_experts"]
    layers_at_75 = sum(item["assignment_recall"] >= 0.75 for item in layer_rows)
    criteria = {
        "top_k_at_most_24": top_k <= MAX_CONTINUATION_TOP_K,
        "assignment_recall_at_least_80_percent": assignment_recall >= 0.80,
        "every_workload_recall_at_least_70_percent": all(
            item["assignment_recall"] >= 0.70 for item in workload_recall
        ),
        "at_least_30_layers_recall_at_least_75_percent": layers_at_75 >= 30,
        "union_recall_at_least_80_percent": union_recall >= 0.80,
        "useful_rate_at_least_50_percent": useful_rate >= 0.50,
        "all_worker_accounting_exact": all(
            candidate["accounting_exact"] for _, candidate in workload_candidates
        ),
    }
    return {
        "feature_source": feature_source,
        "predicted_top_k": top_k,
        **totals,
        "assignment_recall": assignment_recall,
        "full_set_position_rate": (
            totals["full_set_positions"] / totals["positions"]
        ),
        "union_recall": union_recall,
        "useful_rate": useful_rate,
        "layers_at_least_75_percent_recall": layers_at_75,
        "workloads": workload_recall,
        "layers": layer_rows,
        "eligibility": {
            "criteria": criteria,
            "passed": all(criteria.values()),
        },
    }


def _aggregate_gate(rows: list[dict]) -> dict:
    structural_criteria = {
        "five_fresh_workers_complete": (
            len(rows) == len(WORKLOAD_ORDER)
            and tuple(row["workload"] for row in rows) == WORKLOAD_ORDER
        ),
        "all_worker_structural_gates_pass": all(
            row["structural_gate"]["passed"] for row in rows
        ),
        "six_thousand_scored_labels": sum(
            int(row["target_route_trace"]["scored_assignments"])
            for row in rows
        )
        == 6_000,
    }
    structural_passed = all(structural_criteria.values())
    candidates = [
        _aggregate_candidate(rows, feature_source, top_k)
        for feature_source in FEATURE_SOURCES
        for top_k in PREDICTED_TOP_K
    ]
    eligible = [
        candidate for candidate in candidates if candidate["eligibility"]["passed"]
    ]
    winner = (
        min(
            eligible,
            key=lambda candidate: (
                int(candidate["predicted_top_k"]),
                -float(candidate["useful_rate"]),
                -float(candidate["assignment_recall"]),
                FEATURE_SOURCES.index(candidate["feature_source"]),
            ),
        )
        if structural_passed and eligible
        else None
    )
    return {
        "structural": {
            "criteria": structural_criteria,
            "passed": structural_passed,
        },
        "candidates": candidates,
        "eligible_candidates": [
            {
                "feature_source": candidate["feature_source"],
                "predicted_top_k": candidate["predicted_top_k"],
            }
            for candidate in eligible
        ],
        "winner": (
            {
                "feature_source": winner["feature_source"],
                "predicted_top_k": winner["predicted_top_k"],
                "assignment_recall": winner["assignment_recall"],
                "union_recall": winner["union_recall"],
                "useful_rate": winner["useful_rate"],
            }
            if winner is not None
            else None
        ),
        "continuation_passed": winner is not None,
        "decision": (
            "invalid_stop_and_debug"
            if not structural_passed
            else "continue_isolated_scratch_prefetch_gate"
            if winner is not None
            else "stop_direct_frozen_router_transfer_training_required"
        ),
    }


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
    route_path = raw_directory / f"{run_id}.routes.json"
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
        "--route-output",
        str(route_path),
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
    if completed.returncode or not result_path.is_file() or not route_path.is_file():
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
                "route_trace_sha256": _sha256(route_path),
                "stdout_sha256": _sha256(stdout_path),
                "stderr_sha256": _sha256(stderr_path),
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
        }
    )
    return row


def _run_orchestrator(arguments: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    model = Path(arguments.model).expanduser().resolve()
    prompt_manifest = Path(arguments.prompt_manifest).expanduser().resolve()
    raw_directory = Path(arguments.raw_directory).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    protocol = (
        project_root / "research" / "DSPARK_LEARNED_ROUTER_PREDICTOR_2026-08-27.md"
    )
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
    inference_config = json.loads((model / "inference" / "config.json").read_text())

    rows = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{len(prompts)}] workload={prompt['name']}", flush=True)
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
        "evidence_kind": "dspark_hidden_learned_router_frozen_transfer_gate",
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
            "inference_config_sha256": _sha256(model / "inference" / "config.json"),
            "layer_count": installed_manifest["layerCount"],
            "hash_layer_count": inference_config["n_hash_layers"],
            "learned_layer_count": (
                installed_manifest["layerCount"] - inference_config["n_hash_layers"]
            ),
            "expert_count": installed_manifest["expertCount"],
            "selected_expert_count": installed_manifest["selectedExpertCount"],
            "expert_blob_bytes": installed_manifest["expertBlobSize"],
            "dspark_block_size": inference_config["dspark_block_size"],
            "dspark_markov_rank": inference_config["dspark_markov_rank"],
        },
        "experiment": {
            "objective": (
                "Measure whether untrained DSpark hidden features transferred through "
                "frozen target routers justify learned-layer scratch prefetch"
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
            "feature_sources": list(FEATURE_SOURCES),
            "predicted_top_k": list(PREDICTED_TOP_K),
            "alignment": "dspark_position_i_to_target_draft_input_t_i",
            "anchor_routes_scored": False,
            "scored_target_layers": [HASH_LAYER_COUNT, 42],
            "fresh_process_per_workload": True,
            "predictor_training": False,
            "prefetch_executed": False,
            "defaults_changed": False,
        },
        "gate": gate,
        "runs": rows,
        "limitations": [
            "offline first-round route-label gate, not a timing result",
            "anchor routes are captured but excluded from predictor scoring",
            "logical union bytes are labels, not executed reads or physical SSD bytes",
            "no low-confidence, replay, multi-round, long-context, or held-out training data",
            "a failing direct-transfer baseline does not rule out trained route signal",
            "no power, ANE, serving concurrency, or target-I/O overlap measurement",
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
                "continuation_passed": gate["continuation_passed"],
                "winner": gate["winner"],
                "decision": gate["decision"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate direct DSpark-hidden transfer through frozen target routers"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--model", required=True)
    worker.add_argument("--workload", choices=WORKLOAD_ORDER, required=True)
    worker.add_argument("--prompt-file", required=True)
    worker.add_argument("--expected-text-sha256", required=True)
    worker.add_argument("--expected-prompt-tokens", required=True, type=int)
    worker.add_argument("--expected-prompt-token-sha256", required=True)
    worker.add_argument("--route-output", required=True)
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
