from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx_lm.models import deepseek_v4

import deepseek_v4_ssd.model as runtime_model
from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    eval_prompt_cache,
    forward_with_hidden,
    hybrid_verification_forward_with_hidden,
    sequential_verification_forward_with_hidden,
)
from diagnose_layer_parity import (
    _array_comparison,
    _command_output,
    _logit_comparison,
    _materialize,
    _route_comparison,
    _run_verification,
    _runtime_tree_sha256,
    _sha256,
    _sha256_bytes,
    _token_metrics_record,
    _token_sha256,
)


_STAGE_ORDER = (
    "input_hidden",
    "ffn_hc_collapsed",
    "ffn_hc_post",
    "ffn_hc_combine",
    "ffn_norm",
    "shared_output",
    "routed_selected_outputs",
    "routed_reduced",
    "moe_output",
    "post_layer",
)


def _capture_layer0_ffn(
    run: Callable[[], tuple],
    layer0,
) -> tuple[np.ndarray, str, list[dict]]:
    """Capture layer 0 FFN sub-boundaries without changing cache policy."""
    entries: list[dict] = []
    pending_hc: dict | None = None
    pending_expand: dict | None = None
    original_hc_call = deepseek_v4.HyperConnection.__call__
    original_moe_call = deepseek_v4.DeepseekV4MoE.__call__
    original_hc_expand = deepseek_v4.hc_expand

    def capture_hc(self, hidden):
        nonlocal pending_hc
        collapsed, post, combine = original_hc_call(self, hidden)
        if self is layer0.ffn_hc:
            if pending_hc is not None:
                raise RuntimeError("layer 0 FFN HyperConnection capture overlapped")
            pending_hc = {
                "input_length": int(hidden.shape[1]),
                "stages": {
                    "input_hidden": hidden,
                    "ffn_hc_collapsed": collapsed,
                    "ffn_hc_post": post,
                    "ffn_hc_combine": combine,
                },
            }
        return collapsed, post, combine

    def capture_moe(self, x: mx.array, input_ids: mx.array):
        nonlocal pending_hc, pending_expand
        if self is not layer0.ffn:
            return original_moe_call(self, x, input_ids)
        if pending_hc is None or pending_expand is not None:
            raise RuntimeError("layer 0 FFN capture state is inconsistent")
        if getattr(self, "sharding_group", None) is not None:
            raise ValueError("SSD expert streaming supports one Apple Silicon device")

        entry = pending_hc
        pending_hc = None
        stages = entry["stages"]
        stages["ffn_norm"] = x

        indices, scores = self.gate(x, input_ids)
        shared = self.shared_experts(x)
        mx.async_eval(shared)
        cache = self.switch_mlp.cache
        current_batched = getattr(cache, "current_batched", None)
        batched = (
            current_batched(self.switch_mlp.layer)
            if callable(current_batched)
            else None
        )
        if batched is None or getattr(cache, "route_trace_enabled", False):
            started = time.perf_counter()
            mx.eval(indices)
            cache.record_routing_sync(time.perf_counter() - started)
        routed_selected = self.switch_mlp(x, indices)
        routed_reduced = runtime_model._route_reduce(routed_selected, scores)
        output = routed_reduced + shared

        entry["route_indices"] = indices
        entry["route_scores"] = scores
        stages["shared_output"] = shared
        stages["routed_selected_outputs"] = routed_selected
        stages["routed_reduced"] = routed_reduced
        stages["moe_output"] = output
        pending_expand = entry
        entries.append(entry)
        return output

    def capture_expand(value, residual, post, combine):
        nonlocal pending_expand
        output = original_hc_expand(value, residual, post, combine)
        if pending_expand is not None:
            pending_expand["stages"]["post_layer"] = output
            pending_expand = None
        return output

    deepseek_v4.HyperConnection.__call__ = capture_hc
    deepseek_v4.DeepseekV4MoE.__call__ = capture_moe
    deepseek_v4.hc_expand = capture_expand
    try:
        result = run()
        logits, hidden = result[:2]
        mx.eval(logits, hidden)
        if pending_hc is not None or pending_expand is not None:
            raise RuntimeError("layer 0 FFN capture did not finish")
        arrays = [
            array
            for entry in entries
            for array in (
                *[entry["stages"][stage] for stage in _STAGE_ORDER],
                entry["route_indices"],
                entry["route_scores"],
            )
        ]
        if arrays:
            mx.eval(*arrays)
        materialized = [
            {
                "input_length": entry["input_length"],
                "stages": {
                    stage: _materialize(entry["stages"][stage])
                    for stage in _STAGE_ORDER
                },
                "route_indices": np.array(
                    np.asarray(entry["route_indices"].astype(mx.int32)),
                    copy=True,
                ),
                "route_scores": _materialize(entry["route_scores"]),
            }
            for entry in entries
        ]
        return _materialize(logits), result[3].verification_mode, materialized
    finally:
        deepseek_v4.HyperConnection.__call__ = original_hc_call
        deepseek_v4.DeepseekV4MoE.__call__ = original_moe_call
        deepseek_v4.hc_expand = original_hc_expand


def _combine_sequential(entries: list[dict], positions: int) -> dict:
    if len(entries) != positions:
        raise RuntimeError("sequential layer 0 FFN capture count is inconsistent")
    if any(entry["input_length"] != 1 for entry in entries):
        raise RuntimeError("sequential layer 0 FFN capture was not token-shaped")
    return {
        "stages": {
            stage: np.concatenate(
                [entry["stages"][stage] for entry in entries],
                axis=1,
            )
            for stage in _STAGE_ORDER
        },
        "route_indices": np.concatenate(
            [entry["route_indices"] for entry in entries],
            axis=1,
        ),
        "route_scores": np.concatenate(
            [entry["route_scores"] for entry in entries],
            axis=1,
        ),
    }


def _hybrid_block(entries: list[dict], positions: int) -> dict:
    if len(entries) != 1 or entries[0]["input_length"] != positions:
        raise RuntimeError("hybrid layer 0 FFN capture was not block-shaped")
    return entries[0]


def _component_comparisons(
    sequential: dict,
    hybrid: dict,
    positions: int,
) -> list[dict]:
    comparisons = []
    for position in range(positions):
        stages = {
            stage: _array_comparison(
                sequential["stages"][stage][0, position],
                hybrid["stages"][stage][0, position],
            )
            for stage in _STAGE_ORDER
        }
        comparisons.append(
            {
                "position": position,
                "first_nonexact_stage": next(
                    (stage for stage in _STAGE_ORDER if not stages[stage]["exact"]),
                    None,
                ),
                "stages": stages,
                "router": _route_comparison(
                    sequential["route_indices"][0, position],
                    sequential["route_scores"][0, position],
                    hybrid["route_indices"][0, position],
                    hybrid["route_scores"][0, position],
                ),
            }
        )
    return comparisons


def _clear_between_runs() -> None:
    gc.collect()
    mx.clear_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential and hybrid layer 0 FFN component states"
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
        processed = 0
        with mx.stream(runtime._generation_stream):
            while len(prompt_tokens) - processed > 1:
                count = min(1_024, len(prompt_tokens) - processed - 1)
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

            verification_inputs = mx.array(
                [[output_tokens[anchor_index], output_tokens[anchor_index + 1]]],
                dtype=mx.int32,
            )
            positions = int(verification_inputs.shape[1])
            layer0 = runtime.model.model.pipeline_layers[0]

            reference_sequential_values, sequential_mode = _run_verification(
                lambda: sequential_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                )
            )
            _clear_between_runs()
            (
                captured_sequential_values,
                captured_sequential_mode,
                sequential_entries,
            ) = _capture_layer0_ffn(
                lambda: sequential_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                ),
                layer0,
            )
            sequential_capture_exact = bool(
                np.array_equal(
                    reference_sequential_values,
                    captured_sequential_values,
                )
            )
            if sequential_mode != captured_sequential_mode:
                raise RuntimeError("sequential capture changed verifier mode")
            if not sequential_capture_exact:
                raise RuntimeError("FFN capture changed sequential verifier logits")
            _clear_between_runs()

            reference_hybrid_values, hybrid_mode = _run_verification(
                lambda: hybrid_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                )
            )
            _clear_between_runs()
            (
                captured_hybrid_values,
                captured_hybrid_mode,
                hybrid_entries,
            ) = _capture_layer0_ffn(
                lambda: hybrid_verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                ),
                layer0,
            )
            hybrid_capture_exact = bool(
                np.array_equal(reference_hybrid_values, captured_hybrid_values)
            )
            if hybrid_mode != captured_hybrid_mode:
                raise RuntimeError("hybrid capture changed verifier mode")
            if not hybrid_capture_exact:
                raise RuntimeError("FFN capture changed hybrid verifier logits")

            sequential = _combine_sequential(sequential_entries, positions)
            hybrid = _hybrid_block(hybrid_entries, positions)
            components = _component_comparisons(sequential, hybrid, positions)

        git_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
        ).stdout
        git_status = _command_output(["git", "status", "--porcelain"], project_root)
        dependency_path = Path(__file__).with_name("diagnose_layer_parity.py")
        artifact = {
            "schema_version": 1,
            "recorded_at": datetime.datetime.now().astimezone().isoformat(),
            "evidence_kind": "dspark_layer0_ffn_component_diagnostic",
            "formal_performance_result": False,
            "source": {
                "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
                "working_tree_dirty": bool(git_status),
                "tracked_diff_sha256_at_run": _sha256_bytes(git_diff),
                "runtime_python_tree_sha256": _runtime_tree_sha256(
                    project_root / "runtime"
                ),
                "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
                "layer_diagnostic_dependency_sha256": _sha256(dependency_path),
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
                "layer_count": len(runtime.model.model.pipeline_layers),
                "selected_experts": runtime.installed.selected_expert_count,
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
                "anchor_token": output_tokens[anchor_index],
                "following_reference_token": output_tokens[anchor_index + 1],
            },
            "prefix_reconstruction": {
                "expected_tokens": expected_prefix,
                "predicted_tokens": predicted_prefix,
                "exact": True,
            },
            "verification": {
                "positions": positions,
                "sequential_mode": sequential_mode,
                "hybrid_mode": hybrid_mode,
                "instrumentation_preserved_sequential_logits_exactly": (
                    sequential_capture_exact
                ),
                "instrumentation_preserved_hybrid_logits_exactly": (
                    hybrid_capture_exact
                ),
            },
            "stage_order": list(_STAGE_ORDER),
            "components": components,
            "final_logits": [
                {
                    "position": position,
                    **_logit_comparison(
                        reference_sequential_values[0, position],
                        reference_hybrid_values[0, position],
                    ),
                }
                for position in range(positions)
            ],
            "post_run_audit": {
                "prefix_reconstruction_exact": True,
                "captured_sequential_layer0_ffn_calls": len(sequential_entries),
                "captured_hybrid_layer0_ffn_calls": len(hybrid_entries),
                "all_components_captured": all(
                    all(stage in entry["stages"] for stage in _STAGE_ORDER)
                    for entry in [*sequential_entries, *hybrid_entries]
                ),
                "instrumentation_preserved_reference_logits": (
                    sequential_capture_exact and hybrid_capture_exact
                ),
            },
            "evidence_limits": [
                (
                    "This diagnostic compares one exact common cache state and one "
                    "two-token hybrid FFN block."
                ),
                (
                    "The first non-exact stage identifies an observed numerical "
                    "boundary, not necessarily one independently causal kernel."
                ),
                (
                    "Shared and routed expert comparisons use their production "
                    "token-shaped and grouped block execution paths."
                ),
                "No timing or expert-I/O value is performance evidence.",
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
