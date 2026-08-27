from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx_lm.models import deepseek_v4
from mlx_lm.models.cache import CacheList

from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    eval_prompt_cache,
    forward_with_hidden,
    sequential_verification_forward_with_hidden,
    verification_forward_with_hidden,
)
from diagnose_layer_parity import (
    _block_layers,
    _capture_verification,
    _command_output,
    _layer_comparisons,
    _layer_summary,
    _logit_comparison,
    _materialize,
    _run_verification,
    _runtime_tree_sha256,
    _sequential_layers,
    _sha256,
    _sha256_bytes,
    _token_metrics_record,
    _token_sha256,
)


def _mask_cache(cache):
    return cache[0] if isinstance(cache, CacheList) else cache


def _run_hybrid_verification(
    run: Callable[[], tuple],
    *,
    window_size: int,
    capture_layers: bool,
) -> tuple[np.ndarray, str, list[dict]]:
    """Run token-shaped attention and one block-shaped FFN per layer."""
    entries: list[dict] = []
    active_entry: dict | None = None
    attention_token_calls = 0
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

    def hybrid_block(self, hidden, _block_mask, cache, input_ids):
        nonlocal active_entry, attention_token_calls
        input_length = int(hidden.shape[1])
        if input_length < 1:
            raise RuntimeError("hybrid verifier received an empty block")

        attention_outputs = []
        mask_kinds = []
        for position in range(input_length):
            token_hidden = hidden[:, position : position + 1]
            residual = token_hidden
            value, post, combine = self.attn_hc(token_hidden)
            normalized = self.attn_norm(value)
            token_mask = deepseek_v4.create_attention_mask(
                token_hidden[:, :, 0, :],
                _mask_cache(cache),
                window_size=window_size,
                return_array=True,
            )
            value = self.attn(normalized, mask=token_mask, cache=cache)
            token_hidden = deepseek_v4.hc_expand(
                value,
                residual,
                post,
                combine,
            )
            if cache is None:
                mx.eval(token_hidden)
            else:
                eval_prompt_cache([cache], token_hidden)
            attention_outputs.append(token_hidden)
            attention_token_calls += 1
            mask_kinds.append(
                "array"
                if isinstance(token_mask, mx.array)
                else "none"
                if token_mask is None
                else "symbolic"
            )

        hidden = (
            attention_outputs[0]
            if input_length == 1
            else mx.concatenate(attention_outputs, axis=1)
        )
        entry = {
            "layer": int(self.ffn.switch_mlp.layer),
            "attention_type": type(self.attn).__name__,
            "compress_ratio": int(getattr(self.attn, "compress_ratio", 0)),
            "attention_input_length": 1,
            "attention_token_calls": input_length,
            "attention_mask_kinds": tuple(mask_kinds),
            "ffn_input_length": input_length,
            "post_attention": hidden,
        }

        residual = hidden
        value, post, combine = self.ffn_hc(hidden)
        value = self.ffn_norm(value)
        entry["ffn_input"] = value
        previous_entry = active_entry
        active_entry = entry if capture_layers else None
        try:
            value = self.ffn(value, input_ids)
        finally:
            active_entry = previous_entry
        hidden = deepseek_v4.hc_expand(value, residual, post, combine)
        entry["post_layer"] = hidden
        if capture_layers:
            entries.append(entry)
        return hidden

    deepseek_v4.DeepseekV4Block.__call__ = hybrid_block
    if capture_layers:
        deepseek_v4.MoEGate.__call__ = capture_gate
    try:
        result = run()
        logits, hidden = result[:2]
        mx.eval(logits, hidden)
        expected_calls = len(entries) * int(logits.shape[1])
        if capture_layers and attention_token_calls != expected_calls:
            raise RuntimeError("hybrid attention capture count is inconsistent")
        arrays = [
            entry[name]
            for entry in entries
            for name in (
                "post_attention",
                "ffn_input",
                "post_layer",
                "route_indices",
                "route_scores",
            )
        ]
        if arrays:
            mx.eval(*arrays)
        materialized = [
            {
                "layer": entry["layer"],
                "attention_type": entry["attention_type"],
                "compress_ratio": entry["compress_ratio"],
                "attention_input_length": entry["attention_input_length"],
                "attention_token_calls": entry["attention_token_calls"],
                "attention_mask_kinds": list(entry["attention_mask_kinds"]),
                "ffn_input_length": entry["ffn_input_length"],
                "post_attention": _materialize(entry["post_attention"]),
                "ffn_input": _materialize(entry["ffn_input"]),
                "post_layer": _materialize(entry["post_layer"]),
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
        deepseek_v4.DeepseekV4Block.__call__ = original_block_call
        deepseek_v4.MoEGate.__call__ = original_gate_call


def _expert_union_summary(profile) -> dict:
    layers = [
        {
            "call": call,
            "layer": item.layer,
            "routed_expert_assignments": item.routed_expert_assignments,
            "unique_experts": item.unique_experts,
            "cache_misses": item.cache_misses,
        }
        for call, item in enumerate(profile.layers)
    ]
    return {
        "calls": len(layers),
        "routed_expert_assignments": profile.routed_expert_assignments,
        "unique_experts": profile.unique_experts,
        "cache_misses": profile.cache_misses,
        "layers": layers,
    }


def _clear_between_runs() -> None:
    gc.collect()
    mx.clear_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential and block verification with a token-attention, "
            "block-FFN hybrid"
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
            positions = int(verification_inputs.shape[1])
            layer_count = len(runtime.model.model.pipeline_layers)
            window_size = int(runtime.model.model.args.sliding_window)

            with runtime.expert_cache.capture_expert_unions() as sequential_profile:
                reference_sequential_values, sequential_mode = _run_verification(
                    lambda: sequential_verification_forward_with_hidden(
                        runtime.model,
                        verification_inputs,
                        cache,
                        target_layers,
                    )
                )
            sequential_union = _expert_union_summary(sequential_profile)
            _clear_between_runs()

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
                np.array_equal(
                    reference_sequential_values,
                    captured_sequential_values,
                )
            )
            if sequential_mode != captured_sequential_mode:
                raise RuntimeError("sequential capture changed verifier mode")
            if not sequential_capture_exact:
                raise RuntimeError("layer capture changed sequential verifier logits")
            _clear_between_runs()

            with runtime.expert_cache.capture_expert_unions() as block_profile:
                reference_block_values, block_mode = _run_verification(
                    lambda: verification_forward_with_hidden(
                        runtime.model,
                        verification_inputs,
                        cache,
                        target_layers,
                    )
                )
            block_union = _expert_union_summary(block_profile)
            _clear_between_runs()

            with runtime.expert_cache.capture_expert_unions() as hybrid_profile:
                (
                    reference_hybrid_values,
                    hybrid_mode,
                    _,
                ) = _run_hybrid_verification(
                    lambda: verification_forward_with_hidden(
                        runtime.model,
                        verification_inputs,
                        cache,
                        target_layers,
                    ),
                    window_size=window_size,
                    capture_layers=False,
                )
            hybrid_union = _expert_union_summary(hybrid_profile)
            _clear_between_runs()

            (
                captured_hybrid_values,
                captured_hybrid_mode,
                hybrid_entries,
            ) = _run_hybrid_verification(
                lambda: verification_forward_with_hidden(
                    runtime.model,
                    verification_inputs,
                    cache,
                    target_layers,
                ),
                window_size=window_size,
                capture_layers=True,
            )
            hybrid_capture_exact = bool(
                np.array_equal(reference_hybrid_values, captured_hybrid_values)
            )
            if hybrid_mode != captured_hybrid_mode:
                raise RuntimeError("hybrid capture changed verifier mode")
            if not hybrid_capture_exact:
                raise RuntimeError("layer capture changed hybrid verifier logits")
            _clear_between_runs()

            sequential_layers = _sequential_layers(
                sequential_entries,
                layer_count,
                positions,
            )
            hybrid_layers = _block_layers(hybrid_entries, layer_count)
            layers = _layer_comparisons(
                sequential_layers,
                hybrid_layers,
                positions,
            )

        hybrid_union_layer_order_exact = [
            item["layer"] for item in hybrid_union["layers"]
        ] == list(range(layer_count))
        hybrid_retains_block_ffn_union = (
            hybrid_union["calls"] == layer_count
            and hybrid_union_layer_order_exact
            and all(
                item["routed_expert_assignments"]
                == positions * runtime.installed.selected_expert_count
                for item in hybrid_union["layers"]
            )
        )
        sequential_vs_hybrid = [
            {
                "position": position,
                **_logit_comparison(
                    reference_sequential_values[0, position],
                    reference_hybrid_values[0, position],
                ),
            }
            for position in range(positions)
        ]
        token_id_gate_passed = all(
            item["top_token_match"] for item in sequential_vs_hybrid
        )

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
            "evidence_kind": "dspark_hybrid_verifier_diagnostic",
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
                "layer_count": layer_count,
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
                "anchor_token": anchor,
                "following_reference_token": following,
            },
            "prefix_reconstruction": {
                "expected_tokens": expected_prefix,
                "predicted_tokens": predicted_prefix,
                "exact": True,
            },
            "hybrid_contract": {
                "attention_execution": "one_token_at_a_time_per_layer",
                "attention_cache": "one_shared_layer_cache_advanced_per_token",
                "ffn_execution": "one_complete_block_per_layer",
                "expert_union": "one_acquisition_per_layer",
                "underlying_verification_mode": hybrid_mode,
            },
            "verification": {
                "positions": positions,
                "sequential_mode": sequential_mode,
                "block_mode": block_mode,
                "hybrid_mode": hybrid_mode,
                "instrumentation_preserved_sequential_logits_exactly": (
                    sequential_capture_exact
                ),
                "instrumentation_preserved_hybrid_logits_exactly": (
                    hybrid_capture_exact
                ),
            },
            "expert_union": {
                "sequential": sequential_union,
                "block": block_union,
                "hybrid": hybrid_union,
                "hybrid_layer_order_exact": hybrid_union_layer_order_exact,
                "hybrid_retains_block_ffn_union": hybrid_retains_block_ffn_union,
            },
            "summary": _layer_summary(layers, positions),
            "layers": layers,
            "final_logits": {
                "sequential_vs_block": [
                    {
                        "position": position,
                        **_logit_comparison(
                            reference_sequential_values[0, position],
                            reference_block_values[0, position],
                        ),
                    }
                    for position in range(positions)
                ],
                "sequential_vs_hybrid": sequential_vs_hybrid,
                "block_vs_hybrid": [
                    {
                        "position": position,
                        **_logit_comparison(
                            reference_block_values[0, position],
                            reference_hybrid_values[0, position],
                        ),
                    }
                    for position in range(positions)
                ],
            },
            "gate": {
                "case_token_id_parity_passed": token_id_gate_passed,
                "case_full_logit_parity_passed": all(
                    item["exact"] for item in sequential_vs_hybrid
                ),
                "block_ffn_expert_union_preserved": (
                    hybrid_retains_block_ffn_union
                ),
                "decision": (
                    "advance_to_multi_workload_hybrid_validation"
                    if token_id_gate_passed and hybrid_retains_block_ffn_union
                    else "isolate_ffn_hyperconnection_and_routed_qmm"
                ),
            },
            "post_run_audit": {
                "prefix_reconstruction_exact": True,
                "captured_sequential_layers": len(sequential_entries),
                "captured_hybrid_layers": len(hybrid_entries),
                "all_layers_and_positions_captured": (
                    len(sequential_entries) == layer_count * positions
                    and len(hybrid_entries) == layer_count
                ),
                "hybrid_attention_token_calls": sum(
                    item["attention_token_calls"] for item in hybrid_entries
                ),
                "hybrid_ffn_block_calls": len(hybrid_entries),
                "instrumentation_preserved_reference_logits": (
                    sequential_capture_exact and hybrid_capture_exact
                ),
            },
            "evidence_limits": [
                (
                    "This diagnostic compares one exact common cache state and one "
                    "two-token block."
                ),
                (
                    "A passing token-ID gate would establish sufficiency only for "
                    "this reproduction, not general verifier equivalence."
                ),
                (
                    "The hybrid intentionally evaluates caches between attention "
                    "tokens and is not a performance implementation."
                ),
                (
                    "Expert-union cache misses depend on run order; call shape and "
                    "assignment counts, not misses, establish retained union work."
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
