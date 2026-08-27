from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.generation import ModelRuntime, _make_prompt_cache
from deepseek_v4_ssd.model import (
    RuntimeConfig,
    _fork_prompt_cache,
    eval_prompt_cache,
    forward_with_hidden,
    verification_forward_with_hidden,
)


def _token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


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


def _top_two(logits: mx.array) -> dict:
    values = np.asarray(logits.astype(mx.float32))
    indices = np.argpartition(values, -2)[-2:]
    indices = indices[np.argsort(values[indices])[::-1]]
    first, second = (int(index) for index in indices)
    return {
        "token_ids": [first, second],
        "logits": [float(values[first]), float(values[second])],
        "margin": float(values[first] - values[second]),
    }


def _comparison(sequential: mx.array, block: mx.array) -> dict:
    sequential_values = np.asarray(sequential.astype(mx.float32))
    block_values = np.asarray(block.astype(mx.float32))
    difference = np.abs(sequential_values - block_values)
    return {
        "sequential": _top_two(sequential),
        "block": _top_two(block),
        "top_token_match": int(np.argmax(sequential_values))
        == int(np.argmax(block_values)),
        "maximum_absolute_logit_delta": float(difference.max()),
        "mean_absolute_logit_delta": float(difference.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sequential and block target logits from one cache state"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--token-metrics", required=True)
    parser.add_argument("--anchor-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()

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

            sequential_cache, sequential_arrays = _fork_prompt_cache(cache)
            if sequential_arrays:
                mx.eval(*sequential_arrays)
            anchor = output_tokens[anchor_index]
            following = output_tokens[anchor_index + 1]
            sequential_first, _ = forward_with_hidden(
                runtime.model,
                mx.array([[anchor]], dtype=mx.int32),
                sequential_cache,
                target_layers,
            )
            sequential_second, _ = forward_with_hidden(
                runtime.model,
                mx.array([[following]], dtype=mx.int32),
                sequential_cache,
                target_layers,
            )
            mx.eval(sequential_first, sequential_second)

            block_logits, _, _, _ = verification_forward_with_hidden(
                runtime.model,
                mx.array([[anchor, following]], dtype=mx.int32),
                cache,
                target_layers,
            )
            mx.eval(block_logits)

        expected_prefix = output_tokens[: anchor_index + 1]
        artifact = {
            "schema_version": 1,
            "recorded_at": datetime.datetime.now().astimezone().isoformat(),
            "evidence_kind": "dspark_block_parity_diagnostic",
            "formal_performance_result": False,
            "model": str(Path(arguments.model).expanduser().resolve()),
            "model_id": runtime.installed.model_id,
            "revision": runtime.installed.revision,
            "workload": arguments.workload,
            "prompt_tokens": len(prompt_tokens),
            "prompt_token_sha256": prompt_entry["prompt_token_sha256"],
            "reference_token_metrics": str(token_metrics_path),
            "reference_output_token_sha256": token_record["token_sha256"],
            "anchor_index": anchor_index,
            "anchor_token": anchor,
            "following_reference_token": following,
            "prefix_reconstruction": {
                "expected_tokens": expected_prefix,
                "predicted_tokens": predicted_prefix,
                "exact": predicted_prefix == expected_prefix,
            },
            "position_zero": _comparison(
                sequential_first[0, -1],
                block_logits[0, 0],
            ),
            "position_one": _comparison(
                sequential_second[0, -1],
                block_logits[0, 1],
            ),
            "evidence_limits": [
                "This diagnostic compares one cache state and one two-token block.",
                "It does not establish a safe numerical tolerance for every workload.",
                "Timing and expert-I/O counters from this diagnostic are not performance evidence."
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
