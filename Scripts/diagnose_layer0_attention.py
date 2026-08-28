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

from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    eval_prompt_cache,
    forward_with_hidden,
    sequential_verification_forward_with_hidden,
    verification_forward_with_hidden,
)
from diagnose_layer_parity import (
    _array_comparison,
    _command_output,
    _logit_comparison,
    _materialize,
    _run_verification,
    _runtime_tree_sha256,
    _sha256,
    _sha256_bytes,
    _token_metrics_record,
    _token_sha256,
)


_STAGE_ORDER = (
    "input_hidden",
    "attn_hc_collapsed",
    "attn_hc_post",
    "attn_hc_combine",
    "attn_norm",
    "q_wq_a",
    "q_norm",
    "q_wq_b",
    "q_head_rms",
    "q_rope",
    "kv_wkv",
    "kv_norm",
    "kv_rope_new",
    "attention_output",
    "inverse_rope",
    "grouped_projection_input",
    "wo_a",
    "wo_b_input",
    "wo_b_output",
    "post_attention",
)


def _normalize_q(array: mx.array) -> mx.array:
    return array.transpose(0, 2, 1, 3)


def _normalize_grouped(array: mx.array) -> mx.array:
    return array.transpose(0, 2, 1, 3)


def _capture_attention_verification(
    run: Callable[[], tuple],
) -> tuple[np.ndarray, str, list[dict]]:
    """Capture layer 0 attention sub-boundaries with production cache behavior."""
    entries: list[dict] = []
    original_block_call = deepseek_v4.DeepseekV4Block.__call__

    def capture_block(self, hidden, mask, cache, input_ids):
        layer = int(self.ffn.switch_mlp.layer)
        if layer != 0:
            return original_block_call(self, hidden, mask, cache, input_ids)
        if not isinstance(self.attn, deepseek_v4.LocalAttention):
            raise RuntimeError("layer 0 is not LocalAttention")

        attention = self.attn
        entry = {
            "input_length": int(hidden.shape[1]),
            "cache_before": {
                "offset": int(cache.offset),
                "index": int(cache._idx),
            },
            "stages": {"input_hidden": hidden},
            "mask": mask,
        }
        stages = entry["stages"]

        residual = hidden
        collapsed, post, combine = self.attn_hc(hidden)
        stages["attn_hc_collapsed"] = collapsed
        stages["attn_hc_post"] = post
        stages["attn_hc_combine"] = combine
        normalized = self.attn_norm(collapsed)
        stages["attn_norm"] = normalized

        batch, length, _ = normalized.shape
        offset = cache.offset
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset

        q_wq_a = attention.wq_a(normalized)
        stages["q_wq_a"] = q_wq_a
        q_norm = attention.q_norm(q_wq_a)
        stages["q_norm"] = q_norm
        q_wq_b = attention.wq_b(q_norm)
        stages["q_wq_b"] = q_wq_b
        q_head_rms = q_wq_b.reshape(
            batch,
            length,
            attention.n_heads,
            attention.head_dim,
        )
        q_head_rms = mx.fast.rms_norm(
            q_head_rms,
            None,
            attention.config.rms_norm_eps,
        )
        stages["q_head_rms"] = q_head_rms
        q = q_head_rms.transpose(0, 2, 1, 3)
        q = attention.rope(q, offset)
        stages["q_rope"] = _normalize_q(q)

        kv_wkv = attention.wkv(normalized)
        stages["kv_wkv"] = kv_wkv
        kv_norm = attention.kv_norm(kv_wkv)
        stages["kv_norm"] = kv_norm
        kv_new = kv_norm.reshape(batch, 1, length, attention.head_dim)
        kv_new = attention.rope(kv_new, offset)
        stages["kv_rope_new"] = _normalize_q(kv_new)

        fetched, _ = cache.update_and_fetch(
            kv_new,
            mx.zeros((batch, 1, length, 0)),
        )
        entry["cache_fetched_raw"] = fetched
        entry["cache_fetched_temporal"] = cache._temporal_order(fetched)
        entry["cache_after"] = {
            "offset": int(cache.offset),
            "index": int(cache._idx),
        }

        output = deepseek_v4.scaled_dot_product_attention(
            q,
            fetched,
            fetched,
            cache=cache,
            scale=attention.scale,
            mask=mask,
            sinks=attention.attn_sink.astype(q.dtype),
        )
        stages["attention_output"] = _normalize_q(output)
        output = attention.rope(output, offset, inverse=True)
        stages["inverse_rope"] = _normalize_q(output)

        grouped = output.reshape(
            batch,
            attention.o_groups,
            -1,
            length,
            attention.head_dim,
        )
        grouped = grouped.transpose(0, 1, 3, 2, 4).flatten(-2)
        stages["grouped_projection_input"] = _normalize_grouped(grouped)
        projected = attention.wo_a(grouped)
        stages["wo_a"] = _normalize_grouped(projected)
        projected = projected.transpose(0, 2, 1, 3).flatten(-2)
        stages["wo_b_input"] = projected
        projected = attention.wo_b(projected)
        stages["wo_b_output"] = projected

        hidden = deepseek_v4.hc_expand(
            projected,
            residual,
            post,
            combine,
        )
        stages["post_attention"] = hidden
        entries.append(entry)

        residual = hidden
        value, post, combine = self.ffn_hc(hidden)
        value = self.ffn(self.ffn_norm(value), input_ids)
        return deepseek_v4.hc_expand(value, residual, post, combine)

    deepseek_v4.DeepseekV4Block.__call__ = capture_block
    try:
        result = run()
        logits, hidden = result[:2]
        mx.eval(logits, hidden)
        arrays = [
            array
            for entry in entries
            for array in (
                *[entry["stages"][stage] for stage in _STAGE_ORDER],
                entry["mask"],
                entry["cache_fetched_raw"],
                entry["cache_fetched_temporal"],
            )
            if isinstance(array, mx.array)
        ]
        mx.eval(*arrays)
        materialized = []
        for entry in entries:
            materialized.append(
                {
                    "input_length": entry["input_length"],
                    "cache_before": entry["cache_before"],
                    "cache_after": entry["cache_after"],
                    "stages": {
                        stage: _materialize(entry["stages"][stage])
                        for stage in _STAGE_ORDER
                    },
                    "mask": (
                        _materialize(entry["mask"])
                        if isinstance(entry["mask"], mx.array)
                        else entry["mask"]
                    ),
                    "cache_fetched_raw": _materialize(
                        entry["cache_fetched_raw"]
                    ),
                    "cache_fetched_temporal": _materialize(
                        entry["cache_fetched_temporal"]
                    ),
                }
            )
        return _materialize(logits), result[3].verification_mode, materialized
    finally:
        deepseek_v4.DeepseekV4Block.__call__ = original_block_call


def _combine_sequential(entries: list[dict], positions: int) -> dict:
    if len(entries) != positions:
        raise RuntimeError("sequential layer 0 capture count does not match positions")
    if any(entry["input_length"] != 1 for entry in entries):
        raise RuntimeError("sequential layer 0 capture was not token-shaped")
    return {
        stage: np.concatenate(
            [entry["stages"][stage] for entry in entries],
            axis=1,
        )
        for stage in _STAGE_ORDER
    }


def _block_stages(entries: list[dict], positions: int) -> dict:
    if len(entries) != 1 or entries[0]["input_length"] != positions:
        raise RuntimeError("block layer 0 capture was not block-shaped")
    return entries[0]["stages"]


def _component_comparisons(
    sequential: dict,
    block: dict,
    positions: int,
) -> list[dict]:
    comparisons = []
    for position in range(positions):
        stages = {
            stage: _array_comparison(
                sequential[stage][0, position],
                block[stage][0, position],
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
            }
        )
    return comparisons


def _mask_summary(mask: np.ndarray | str | None) -> dict:
    if mask is None:
        return {"kind": "none"}
    if isinstance(mask, str):
        return {"kind": "symbolic", "value": mask}
    if mask.dtype == np.bool_:
        allowed = int(np.count_nonzero(mask))
    else:
        allowed = int(np.count_nonzero(np.isfinite(mask)))
    return {
        "kind": "array",
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "finite_or_true_elements": allowed,
        "elements": int(mask.size),
    }


def _cache_shape_summary(
    sequential: np.ndarray,
    block: np.ndarray,
) -> dict:
    return {
        "sequential_shape": list(sequential.shape),
        "block_shape": list(block.shape),
        "directly_comparable": sequential.shape == block.shape,
    }


def _temporal_cache_suffix_comparison(
    sequential: np.ndarray,
    block: np.ndarray,
) -> dict:
    shared_tokens = min(sequential.shape[2], block.shape[2])
    return {
        **_cache_shape_summary(sequential, block),
        "shared_suffix_tokens": shared_tokens,
        "shared_suffix": _array_comparison(
            sequential[:, :, -shared_tokens:, :],
            block[:, :, -shared_tokens:, :],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare sequential and block layer 0 attention component states"
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
    prompt = (manifest_path.parent / prompt_entry["file"]).read_text(
        encoding="utf-8"
    )

    token_metrics_path = Path(arguments.token_metrics).expanduser().resolve()
    token_document = json.loads(token_metrics_path.read_text(encoding="utf-8"))
    token_record = _token_metrics_record(token_document, arguments.workload)
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
            prefix_inputs = [prompt_tokens[-1], *output_tokens[:anchor_index]]
            for token in prefix_inputs:
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
            ) = _capture_attention_verification(
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
                raise RuntimeError("component capture changed sequential logits")
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
                _capture_attention_verification(
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
                raise RuntimeError("component capture changed block logits")
            gc.collect()
            mx.clear_cache()

            sequential_stages = _combine_sequential(sequential_entries, positions)
            block_stages = _block_stages(block_entries, positions)
            components = _component_comparisons(
                sequential_stages,
                block_stages,
                positions,
            )

            final_sequential_cache = sequential_entries[-1]
            final_block_cache = block_entries[0]
            cache_comparison = {
                "sequential_updates": [
                    {
                        "input_length": entry["input_length"],
                        "update_path": "in_place",
                        "cache_before": entry["cache_before"],
                        "cache_after": entry["cache_after"],
                        "mask": _mask_summary(entry["mask"]),
                    }
                    for entry in sequential_entries
                ],
                "block_update": {
                    "input_length": final_block_cache["input_length"],
                    "update_path": "concatenate",
                    "cache_before": final_block_cache["cache_before"],
                    "cache_after": final_block_cache["cache_after"],
                    "mask": _mask_summary(final_block_cache["mask"]),
                },
                "final_raw_fetched_cache": _cache_shape_summary(
                    final_sequential_cache["cache_fetched_raw"],
                    final_block_cache["cache_fetched_raw"],
                ),
                "final_temporal_order_cache": _temporal_cache_suffix_comparison(
                    final_sequential_cache["cache_fetched_temporal"],
                    final_block_cache["cache_fetched_temporal"],
                ),
            }

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
            "evidence_kind": "dspark_layer0_attention_component_diagnostic",
            "formal_performance_result": False,
            "source": {
                "commit": _command_output(["git", "rev-parse", "HEAD"], project_root),
                "working_tree_dirty": bool(git_status),
                "tracked_diff_sha256_at_run": _sha256_bytes(git_diff),
                "runtime_python_tree_sha256": _runtime_tree_sha256(
                    project_root / "runtime"
                ),
                "diagnostic_script_sha256": _sha256(Path(__file__).resolve()),
                "layer_diagnostic_dependency_sha256": _sha256(
                    Path(__file__).resolve().with_name("diagnose_layer_parity.py")
                ),
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
                "layer": 0,
                "attention_type": type(
                    runtime.model.model.pipeline_layers[0].attn
                ).__name__,
                "sliding_window": int(runtime.model.model.args.sliding_window),
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
                "positions": positions,
                "sequential_mode": sequential_mode,
                "block_mode": block_mode,
                "instrumentation_preserved_sequential_logits_exactly": (
                    sequential_capture_exact
                ),
                "instrumentation_preserved_block_logits_exactly": block_capture_exact,
            },
            "stage_order": list(_STAGE_ORDER),
            "components": components,
            "cache_update": cache_comparison,
            "final_logits": [
                {
                    "position": position,
                    **_logit_comparison(
                        captured_sequential_values[0, position],
                        captured_block_values[0, position],
                    ),
                }
                for position in range(positions)
            ],
            "post_run_audit": {
                "prefix_reconstruction_exact": True,
                "captured_sequential_layer0_calls": len(sequential_entries),
                "captured_block_layer0_calls": len(block_entries),
                "all_components_captured": all(
                    set(entry["stages"]) == set(_STAGE_ORDER)
                    for entry in (*sequential_entries, *block_entries)
                ),
                "instrumentation_preserved_reference_logits": (
                    sequential_capture_exact and block_capture_exact
                ),
            },
            "evidence_limits": [
                (
                    "This diagnostic compares one exact cache state and one "
                    "two-token block."
                ),
                (
                    "The first non-exact captured tensor identifies an observed "
                    "boundary, not necessarily one independently causal kernel."
                ),
                (
                    "Raw rotating-cache layout and temporal-order cache comparisons "
                    "describe different update paths, not performance."
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
