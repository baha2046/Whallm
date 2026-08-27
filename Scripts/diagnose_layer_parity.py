from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx_lm.models import deepseek_v4

from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    eval_prompt_cache,
    forward_with_hidden,
    sequential_verification_forward_with_hidden,
    verification_forward_with_hidden,
)


_CAPTURED_STAGES = ("post_attention", "ffn_input", "post_layer")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_tree_sha256(runtime_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(runtime_root.rglob("*.py")):
        digest.update(path.relative_to(runtime_root).as_posix().encode())
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


def _token_sha256(tokens: list[int]) -> str:
    return _sha256_bytes(",".join(map(str, tokens)).encode())


def _token_metrics_record(document: dict, workload: str) -> dict:
    if "generated_token_ids" in document:
        return document
    candidates = [
        run["metrics"]
        for run in document.get("runs", [])
        if run.get("workload") == workload
        and run.get("mode") == "normal"
        and "generated_token_ids" in run.get("metrics", {})
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "token metrics must contain one normal run for the selected workload"
        )
    return candidates[0]


def _top_two(logits: np.ndarray) -> dict:
    values = np.asarray(logits, dtype=np.float32)
    indices = np.argpartition(values, -2)[-2:]
    indices = indices[np.argsort(values[indices])[::-1]]
    first, second = (int(index) for index in indices)
    return {
        "token_ids": [first, second],
        "logits": [float(values[first]), float(values[second])],
        "margin": float(values[first] - values[second]),
    }


def _logit_comparison(sequential: np.ndarray, block: np.ndarray) -> dict:
    sequential_values = np.asarray(sequential, dtype=np.float32)
    block_values = np.asarray(block, dtype=np.float32)
    difference = np.abs(sequential_values - block_values)
    return {
        "sequential": _top_two(sequential),
        "block": _top_two(block),
        "top_token_match": int(np.argmax(sequential_values))
        == int(np.argmax(block_values)),
        "exact": bool(np.array_equal(sequential_values, block_values)),
        "maximum_absolute_logit_delta": float(difference.max()),
        "mean_absolute_logit_delta": float(difference.mean(dtype=np.float64)),
    }


def _array_comparison(sequential: np.ndarray, block: np.ndarray) -> dict:
    sequential_values = sequential.astype(np.float64, copy=False)
    block_values = block.astype(np.float64, copy=False)
    difference = np.abs(sequential_values - block_values)
    squared = np.square(sequential_values - block_values)
    return {
        "exact": bool(np.array_equal(sequential, block)),
        "elements": int(difference.size),
        "nonzero_elements": int(np.count_nonzero(difference)),
        "maximum_absolute_delta": float(difference.max(initial=0.0)),
        "mean_absolute_delta": float(difference.mean()) if difference.size else 0.0,
        "root_mean_square_delta": (
            float(math.sqrt(squared.mean())) if squared.size else 0.0
        ),
        "sequential_root_mean_square": (
            float(math.sqrt(np.square(sequential_values).mean()))
            if sequential_values.size
            else 0.0
        ),
    }


def _route_comparison(
    sequential_indices: np.ndarray,
    sequential_scores: np.ndarray,
    block_indices: np.ndarray,
    block_scores: np.ndarray,
) -> dict:
    sequential_ids = [int(value) for value in sequential_indices.reshape(-1)]
    block_ids = [int(value) for value in block_indices.reshape(-1)]
    sequential_weights = [float(value) for value in sequential_scores.reshape(-1)]
    block_weights = [float(value) for value in block_scores.reshape(-1)]
    sequential_map = dict(zip(sequential_ids, sequential_weights, strict=True))
    block_map = dict(zip(block_ids, block_weights, strict=True))
    shared = sorted(set(sequential_ids) & set(block_ids))
    common_score_delta = [
        abs(sequential_map[expert] - block_map[expert]) for expert in shared
    ]
    return {
        "ordered_exact": sequential_ids == block_ids,
        "set_exact": set(sequential_ids) == set(block_ids),
        "shared_experts": len(shared),
        "sequential_expert_ids": sequential_ids,
        "block_expert_ids": block_ids,
        "sequential_scores": sequential_weights,
        "block_scores": block_weights,
        "maximum_absolute_common_expert_score_delta": (
            max(common_score_delta) if common_score_delta else None
        ),
    }


def _materialize(array: mx.array) -> np.ndarray:
    return np.array(np.asarray(array.astype(mx.float32)), copy=True)


def _capture_verification(
    run: Callable[[], tuple],
) -> tuple[np.ndarray, str, list[dict]]:
    """Capture layer boundaries without changing the verifier's cache policy."""
    entries: list[dict] = []
    active_entry: dict | None = None
    original_block_call = deepseek_v4.DeepseekV4Block.__call__
    original_gate_call = deepseek_v4.MoEGate.__call__

    def capture_gate(self, x: mx.array, input_ids: mx.array | None = None):
        nonlocal active_entry
        indices, scores = original_gate_call(self, x, input_ids)
        if active_entry is None:
            raise RuntimeError("router capture occurred outside a decoder layer")
        active_entry["route_indices"] = indices
        active_entry["route_scores"] = scores
        return indices, scores

    def capture_block(self, hidden, mask, cache, input_ids):
        nonlocal active_entry
        residual = hidden
        value, post, combine = self.attn_hc(hidden)
        value = self.attn(self.attn_norm(value), mask=mask, cache=cache)
        hidden = deepseek_v4.hc_expand(value, residual, post, combine)

        entry = {
            "layer": int(self.ffn.switch_mlp.layer),
            "attention_type": type(self.attn).__name__,
            "compress_ratio": int(getattr(self.attn, "compress_ratio", 0)),
            "post_attention": hidden,
        }
        entries.append(entry)

        residual = hidden
        value, post, combine = self.ffn_hc(hidden)
        value = self.ffn_norm(value)
        entry["ffn_input"] = value
        previous_entry = active_entry
        active_entry = entry
        try:
            value = self.ffn(value, input_ids)
        finally:
            active_entry = previous_entry
        hidden = deepseek_v4.hc_expand(value, residual, post, combine)
        entry["post_layer"] = hidden
        return hidden

    deepseek_v4.DeepseekV4Block.__call__ = capture_block
    deepseek_v4.MoEGate.__call__ = capture_gate
    try:
        result = run()
        logits, hidden = result[:2]
        mx.eval(logits, hidden)
        arrays = [
            entry[name]
            for entry in entries
            for name in (*_CAPTURED_STAGES, "route_indices", "route_scores")
        ]
        if arrays:
            mx.eval(*arrays)
        materialized = []
        for entry in entries:
            materialized.append(
                {
                    "layer": entry["layer"],
                    "attention_type": entry["attention_type"],
                    "compress_ratio": entry["compress_ratio"],
                    **{
                        stage: _materialize(entry[stage])
                        for stage in _CAPTURED_STAGES
                    },
                    "route_indices": np.array(
                        np.asarray(entry["route_indices"].astype(mx.int32)),
                        copy=True,
                    ),
                    "route_scores": _materialize(entry["route_scores"]),
                }
            )
        logits_values = _materialize(logits)
        mode = result[3].verification_mode
        return logits_values, mode, materialized
    finally:
        deepseek_v4.DeepseekV4Block.__call__ = original_block_call
        deepseek_v4.MoEGate.__call__ = original_gate_call


def _run_verification(run: Callable[[], tuple]) -> tuple[np.ndarray, str]:
    result = run()
    logits, hidden = result[:2]
    mx.eval(logits, hidden)
    return _materialize(logits), result[3].verification_mode


def _sequential_layers(
    entries: list[dict],
    layer_count: int,
    position_count: int,
) -> list[dict]:
    if len(entries) != layer_count * position_count:
        raise RuntimeError(
            "sequential capture count does not match layers times positions"
        )
    layers = []
    for layer in range(layer_count):
        positions = [
            entries[position * layer_count + layer]
            for position in range(position_count)
        ]
        if any(entry["layer"] != layer for entry in positions):
            raise RuntimeError("sequential layer capture order is inconsistent")
        layers.append(
            {
                "layer": layer,
                "attention_type": positions[0]["attention_type"],
                "compress_ratio": positions[0]["compress_ratio"],
                **{
                    stage: np.concatenate(
                        [entry[stage] for entry in positions], axis=1
                    )
                    for stage in _CAPTURED_STAGES
                },
                "route_indices": np.concatenate(
                    [entry["route_indices"] for entry in positions], axis=1
                ),
                "route_scores": np.concatenate(
                    [entry["route_scores"] for entry in positions], axis=1
                ),
            }
        )
    return layers


def _block_layers(entries: list[dict], layer_count: int) -> list[dict]:
    if len(entries) != layer_count:
        raise RuntimeError("block capture count does not match model layers")
    if [entry["layer"] for entry in entries] != list(range(layer_count)):
        raise RuntimeError("block layer capture order is inconsistent")
    return entries


def _layer_comparisons(
    sequential_layers: list[dict],
    block_layers: list[dict],
    position_count: int,
) -> list[dict]:
    comparisons = []
    for sequential, block in zip(sequential_layers, block_layers, strict=True):
        layer = sequential["layer"]
        if block["layer"] != layer:
            raise RuntimeError("sequential and block layer indices do not align")
        positions = []
        for position in range(position_count):
            positions.append(
                {
                    "position": position,
                    **{
                        stage: _array_comparison(
                            sequential[stage][0, position],
                            block[stage][0, position],
                        )
                        for stage in _CAPTURED_STAGES
                    },
                    "router": _route_comparison(
                        sequential["route_indices"][0, position],
                        sequential["route_scores"][0, position],
                        block["route_indices"][0, position],
                        block["route_scores"][0, position],
                    ),
                }
            )
        comparisons.append(
            {
                "layer": layer,
                "attention_type": sequential["attention_type"],
                "compress_ratio": sequential["compress_ratio"],
                "positions": positions,
            }
        )
    return comparisons


def _first_layer(layers: list[dict], position: int, predicate: Callable[[dict], bool]):
    return next(
        (layer["layer"] for layer in layers if predicate(layer["positions"][position])),
        None,
    )


def _layer_summary(layers: list[dict], position_count: int) -> dict:
    positions = []
    for position in range(position_count):
        positions.append(
            {
                "position": position,
                "first_post_attention_nonexact_layer": _first_layer(
                    layers,
                    position,
                    lambda item: not item["post_attention"]["exact"],
                ),
                "first_ffn_input_nonexact_layer": _first_layer(
                    layers,
                    position,
                    lambda item: not item["ffn_input"]["exact"],
                ),
                "first_router_order_mismatch_layer": _first_layer(
                    layers,
                    position,
                    lambda item: not item["router"]["ordered_exact"],
                ),
                "first_router_set_mismatch_layer": _first_layer(
                    layers,
                    position,
                    lambda item: not item["router"]["set_exact"],
                ),
                "first_post_layer_nonexact_layer": _first_layer(
                    layers,
                    position,
                    lambda item: not item["post_layer"]["exact"],
                ),
                "maximum_post_attention_absolute_delta": max(
                    layer["positions"][position]["post_attention"][
                        "maximum_absolute_delta"
                    ]
                    for layer in layers
                ),
                "maximum_ffn_input_absolute_delta": max(
                    layer["positions"][position]["ffn_input"][
                        "maximum_absolute_delta"
                    ]
                    for layer in layers
                ),
                "maximum_post_layer_absolute_delta": max(
                    layer["positions"][position]["post_layer"][
                        "maximum_absolute_delta"
                    ]
                    for layer in layers
                ),
            }
        )
    return {"positions": positions}


def _broadcast_embeddings(model, inputs: mx.array) -> mx.array:
    core = model.model
    hidden = core.embed_tokens(inputs)
    hidden = mx.broadcast_to(
        hidden[:, :, None, :],
        (*hidden.shape[:2], core.args.hc_mult, hidden.shape[-1]),
    )
    return mx.contiguous(hidden)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential and two-token block target states at every "
            "decoder layer"
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--token-metrics", required=True)
    parser.add_argument("--anchor-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    manifest_path = Path(arguments.prompt_manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_entry = next(
        (
            prompt
            for prompt in manifest["prompts"]
            if prompt["name"] == arguments.workload
        ),
        None,
    )
    if prompt_entry is None:
        parser.error("--workload is not present in --prompt-manifest")
    prompt_path = manifest_path.parent / prompt_entry["file"]
    prompt = prompt_path.read_text(encoding="utf-8")

    token_metrics_path = Path(arguments.token_metrics).expanduser().resolve()
    token_metrics = json.loads(token_metrics_path.read_text(encoding="utf-8"))
    token_record = _token_metrics_record(token_metrics, arguments.workload)
    output_tokens = [int(token) for token in token_record["generated_token_ids"]]
    anchor_index = arguments.anchor_index
    if not 0 <= anchor_index < len(output_tokens) - 1:
        parser.error("--anchor-index must leave at least one following token")

    runtime = ModelRuntime.open(
        arguments.model,
        RuntimeConfig(
            persistent_prompt_cache=False,
            layer_major_prefill=False,
        ),
    )
    try:
        prompt_tokens = runtime._encode_prompt(prompt)
        if _token_sha256(prompt_tokens) != prompt_entry["prompt_token_sha256"]:
            raise RuntimeError("runtime prompt token hash does not match manifest")
        if token_record["prompt_token_sha256"] != prompt_entry["prompt_token_sha256"]:
            raise RuntimeError("token metrics prompt hash does not match manifest")

        target_layers = runtime.installed.dspark_target_layer_ids
        cache = _make_prompt_cache(runtime.model)
        step_size = 1_024
        processed = 0
        with mx.stream(runtime._generation_stream):
            while len(prompt_tokens) - processed > 1:
                count = min(step_size, len(prompt_tokens) - processed - 1)
                _, hidden = forward_with_hidden(
                    runtime.model,
                    mx.array([prompt_tokens[processed : processed + count]]),
                    cache,
                    target_layers,
                )
                mx.eval(hidden)
                processed += count

            predicted_prefix = []
            inputs = [prompt_tokens[-1], *output_tokens[:anchor_index]]
            for token in inputs:
                logits, hidden = forward_with_hidden(
                    runtime.model,
                    mx.array([[token]], dtype=mx.int32),
                    cache,
                    target_layers,
                )
                mx.eval(logits, hidden)
                predicted_prefix.append(int(mx.argmax(logits[0, -1]).item()))
            eval_prompt_cache(cache)

            expected_prefix = output_tokens[: anchor_index + 1]
            if predicted_prefix != expected_prefix:
                raise RuntimeError(
                    "common prefix reconstruction did not preserve exact tokens"
                )

            anchor = output_tokens[anchor_index]
            following = output_tokens[anchor_index + 1]
            verification_inputs = mx.array(
                [[anchor, following]],
                dtype=mx.int32,
            )

            reference_sequential_values, sequential_mode = _run_verification(
                lambda: sequential_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                )
            )
            gc.collect()
            mx.clear_cache()
            (
                captured_sequential_values,
                captured_sequential_mode,
                sequential_entries,
            ) = _capture_verification(
                lambda: sequential_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                )
            )
            sequential_capture_exact = bool(
                np.array_equal(reference_sequential_values, captured_sequential_values)
            )
            if sequential_mode != captured_sequential_mode:
                raise RuntimeError("sequential capture changed verifier mode")
            if not sequential_capture_exact:
                raise RuntimeError("layer capture changed sequential verifier logits")
            gc.collect()
            mx.clear_cache()

            reference_block_values, block_mode = _run_verification(
                lambda: verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                )
            )
            gc.collect()
            mx.clear_cache()
            captured_block_values, captured_block_mode, block_entries = (
                _capture_verification(
                    lambda: verification_forward_with_hidden(
                        runtime.model,
                        verification_inputs,
                        cache,
                        target_layers,
                    )
                )
            )
            block_capture_exact = bool(
                np.array_equal(reference_block_values, captured_block_values)
            )
            if block_mode != captured_block_mode:
                raise RuntimeError("block capture changed verifier mode")
            if not block_capture_exact:
                raise RuntimeError("layer capture changed block verifier logits")
            gc.collect()
            mx.clear_cache()

            layer_count = len(runtime.model.model.pipeline_layers)
            sequential_layers = _sequential_layers(
                sequential_entries,
                layer_count,
                verification_inputs.shape[1],
            )
            block_layers = _block_layers(block_entries, layer_count)
            layers = _layer_comparisons(
                sequential_layers,
                block_layers,
                verification_inputs.shape[1],
            )

            sequential_embeddings = mx.concatenate(
                [
                    _broadcast_embeddings(
                        runtime.model,
                        verification_inputs[:, position : position + 1],
                    )
                    for position in range(verification_inputs.shape[1])
                ],
                axis=1,
            )
            block_embeddings = _broadcast_embeddings(runtime.model, verification_inputs)
            mx.eval(sequential_embeddings, block_embeddings)
            embedding_comparisons = [
                {
                    "position": position,
                    **_array_comparison(
                        _materialize(sequential_embeddings)[0, position],
                        _materialize(block_embeddings)[0, position],
                    ),
                }
                for position in range(verification_inputs.shape[1])
            ]

        git_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
        ).stdout
        git_status = _command_output(["git", "status", "--porcelain"], project_root)
        artifact = {
            "schema_version": 1,
            "recorded_at": datetime.datetime.now().astimezone().isoformat(),
            "evidence_kind": "dspark_layer_parity_diagnostic",
            "formal_performance_result": False,
            "source": {
                "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
                "working_tree_dirty": bool(git_status),
                "tracked_diff_sha256_at_run": _sha256_bytes(git_diff),
                "runtime_python_tree_sha256": _runtime_tree_sha256(
                    project_root / "runtime"
                ),
                "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
            },
            "environment": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "mac_model": _command_output(
                    ["sysctl", "-n", "hw.model"], project_root
                ),
                "chip": _command_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], project_root
                ),
                "memory_bytes": int(
                    _command_output(["sysctl", "-n", "hw.memsize"], project_root)
                    or 0
                ),
                "python": platform.python_version(),
                "mlx": importlib.metadata.version("mlx"),
                "mlx_lm": importlib.metadata.version("mlx-lm"),
                "transformers": importlib.metadata.version("transformers"),
            },
            "checkpoint": {
                "installed_model": str(Path(arguments.model).expanduser().resolve()),
                "model_id": runtime.installed.model_id,
                "revision": runtime.installed.revision,
                "layer_count": layer_count,
                "dspark_target_layer_ids": list(target_layers),
            },
            "workload": {
                "name": arguments.workload,
                "prompt_tokens": len(prompt_tokens),
                "prompt_manifest": str(manifest_path.relative_to(project_root)),
                "prompt_manifest_sha256": _sha256(manifest_path),
                "prompt_token_sha256": prompt_entry["prompt_token_sha256"],
                "reference_token_metrics": str(
                    token_metrics_path.relative_to(project_root)
                ),
                "reference_output_token_sha256": token_record["token_sha256"],
                "anchor_index": anchor_index,
                "anchor_token": anchor,
                "following_reference_token": following,
            },
            "prefix_reconstruction": {
                "expected_tokens": expected_prefix,
                "predicted_tokens": predicted_prefix,
                "exact": True,
            },
            "verification": {
                "positions": int(verification_inputs.shape[1]),
                "sequential_mode": sequential_mode,
                "block_mode": block_mode,
                "instrumentation_preserved_sequential_logits_exactly": (
                    sequential_capture_exact
                ),
                "instrumentation_preserved_block_logits_exactly": block_capture_exact,
            },
            "embeddings": embedding_comparisons,
            "summary": _layer_summary(layers, verification_inputs.shape[1]),
            "layers": layers,
            "final_logits": [
                {
                    "position": position,
                    **_logit_comparison(
                        captured_sequential_values[0, position],
                        captured_block_values[0, position],
                    ),
                }
                for position in range(verification_inputs.shape[1])
            ],
            "post_run_audit": {
                "prefix_reconstruction_exact": True,
                "captured_sequential_layers": len(sequential_entries),
                "captured_block_layers": len(block_entries),
                "all_layers_and_positions_captured": (
                    len(sequential_entries)
                    == layer_count * verification_inputs.shape[1]
                    and len(block_entries) == layer_count
                ),
                "instrumentation_preserved_reference_logits": (
                    sequential_capture_exact and block_capture_exact
                ),
            },
            "evidence_limits": [
                (
                    "This diagnostic compares one exact common cache state and one "
                    "two-token block."
                ),
                (
                    "Layer deltas identify the first observed numerical boundary but "
                    "do not prove one kernel is solely causal."
                ),
                (
                    "Router equality covers selected expert IDs and selected weights, "
                    "not every unselected router logit."
                ),
                (
                    "No timing or expert-I/O value from this diagnostic is performance "
                    "evidence."
                ),
            ],
        }
        output = Path(arguments.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output}")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
