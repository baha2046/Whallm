from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from .generation import (
    APPROXIMATION_MODES,
    EXACT_APPROXIMATION_MODE,
    LEARNED_ROUTE_DROP_LOWEST_1,
    GenerationOptions,
    ModelRuntime,
)
from .io_metrics import EXPERT_FILE_CACHE_POLICIES
from .manifest import InstalledModel
from .model import (
    RuntimeConfig,
    _POWER_SAVING_LIMITS_GBPS,
    _select_moe_step_size,
)


def _read_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt_file is None:
        assert prompt is not None
        return prompt
    return Path(prompt_file).read_text(encoding="utf-8")


def _token_sha256(tokens) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def _select_approximation_mode(
    requested: str | None,
    *,
    is_qwen: bool,
    dspark_enabled: bool,
) -> str:
    mode = requested
    if mode is None:
        mode = (
            EXACT_APPROXIMATION_MODE
            if is_qwen or dspark_enabled
            else LEARNED_ROUTE_DROP_LOWEST_1
        )
    if mode != EXACT_APPROXIMATION_MODE and (is_qwen or dspark_enabled):
        raise ValueError("--approximation is not supported by Qwen or DSpark")
    return mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an installed model")
    parser.add_argument("--model", required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=272_000)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--approximation", choices=sorted(APPROXIMATION_MODES))
    parser.add_argument("--slots", type=int, default=1_152)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--prefetch-read-workers", type=int, default=2)
    parser.add_argument(
        "--power-saving-limit-gbps",
        type=float,
        choices=_POWER_SAVING_LIMITS_GBPS,
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=int,
        default=0,
        help="MLX memory limit in GiB; 0 selects the model-safe automatic limit",
    )
    parser.add_argument("--prefill-step-size", type=int, default=0)
    parser.add_argument("--moe-prefill-step-size", type=int, default=0)
    parser.add_argument("--no-layer-major-prefill", action="store_true")
    parser.add_argument("--layer-major-prefill-threshold", type=int, default=1_024)
    parser.add_argument("--no-batched-expert-prefill", action="store_true")
    parser.add_argument(
        "--qwen-grouped-experts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="group Qwen Prefill rows by expert (default: on for Qwen when MTP is off)",
    )
    parser.add_argument(
        "--no-ane-prefill",
        action="store_true",
        help="use the original GPU Prefill path",
    )
    parser.add_argument("--ane-prefill-ratio", type=float, default=0.25)
    parser.add_argument(
        "--qwen-next-layer-prefetch",
        action="store_true",
        help="research only: prefetch the next Qwen expert layer during compute",
    )
    parser.add_argument("--prompt-cache-entries", type=int, default=2)
    parser.add_argument("--prompt-cache-memory-gib", type=int, default=8)
    parser.add_argument("--no-persistent-prompt-cache", action="store_true")
    parser.add_argument("--prompt-cache-directory")
    parser.add_argument("--bf16-kv-cache", action="store_true")
    parser.add_argument("--no-fp4-index-cache", action="store_true")
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="enable experimental Qwen MTP speculative decoding",
    )
    parser.add_argument("--mtp-slots", type=int, default=32)
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument(
        "--dspark-prompt-cache",
        action="store_true",
        help="experimentally reuse an atomic target and DSpark prompt snapshot",
    )
    parser.add_argument(
        "--dspark-hash-prefetch",
        action="store_true",
        help="experimentally prefetch exact target hash-layer experts",
    )
    parser.add_argument(
        "--dspark-adaptive-block",
        action="store_true",
        help="experimentally select a storage-aware DSpark draft prefix",
    )
    parser.add_argument(
        "--no-dspark-fallback",
        action="store_true",
        help="research only: continue DSpark after its wall-time stop gate",
    )
    parser.add_argument(
        "--dspark-sequential-verification",
        action="store_true",
        help="research oracle: verify each DSpark target position sequentially",
    )
    parser.add_argument(
        "--dspark-hybrid-verification",
        action="store_true",
        help=(
            "experimental verifier: token-shaped target math with one expert "
            "union acquisition per layer"
        ),
    )
    parser.add_argument("--dspark-slots", type=int, default=768)
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--metrics-json")
    parser.add_argument("--expert-route-trace")
    parser.add_argument(
        "--expert-page-cache-probe",
        action="store_true",
        help=(
            "research-only pre-read mincore classification for expert-file "
            "page residency; not a physical SSD byte counter"
        ),
    )
    parser.add_argument(
        "--expert-file-cache-policy",
        choices=EXPERT_FILE_CACHE_POLICIES,
        default="cached",
        help=(
            "research-only expert descriptor policy; bypass uses Darwin "
            "F_NOCACHE with read-ahead disabled"
        ),
    )
    parser.add_argument("--no-ready-expert-decode", action="store_true")
    arguments = parser.parse_args()
    try:
        prompt_text = _read_prompt(arguments.prompt, arguments.prompt_file)
    except OSError as error:
        parser.error(f"cannot read --prompt-file: {error}")
    if arguments.max_tokens < 1:
        parser.error("--max-tokens must be greater than zero")
    try:
        installed = InstalledModel.open(arguments.model)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if installed.is_qwen and arguments.dspark:
        parser.error("Qwen3.8-Flash-Next does not support --dspark")
    try:
        approximation_mode = _select_approximation_mode(
            arguments.approximation,
            is_qwen=installed.is_qwen,
            dspark_enabled=arguments.dspark,
        )
    except ValueError as error:
        parser.error(str(error))
    if arguments.mtp and not installed.is_qwen:
        parser.error("--mtp is supported only by Qwen3.8-Flash-Next")
    if arguments.mtp and not installed.has_mtp:
        parser.error("--mtp requires an installed MTP sidecar")
    temperature = arguments.temperature
    top_p = arguments.top_p
    top_k = arguments.top_k
    if temperature is None:
        temperature = 1.0 if installed.is_qwen else 0.2
    if top_p is None:
        top_p = 0.95 if installed.is_qwen else 0.98
    if top_k is None:
        top_k = 20 if installed.is_qwen else 0
    if not 0 <= temperature <= 2:
        parser.error("--temperature must be between zero and two")
    if not 0 < top_p <= 1:
        parser.error("--top-p must be greater than zero and at most one")
    if not 0 <= top_k <= 248_320:
        parser.error("--top-k must be between zero and 248320")
    if arguments.slots < 6:
        parser.error("--slots must be at least 6")
    if arguments.read_workers < 1:
        parser.error("--read-workers must be greater than zero")
    if arguments.prefetch_read_workers < 1:
        parser.error("--prefetch-read-workers must be greater than zero")
    if arguments.memory_limit_gib < 0:
        parser.error("--memory-limit-gib must be zero or greater")
    if arguments.prefill_step_size < 0:
        parser.error("--prefill-step-size must be zero or greater")
    if arguments.moe_prefill_step_size < 0:
        parser.error("--moe-prefill-step-size must be zero or greater")
    if arguments.layer_major_prefill_threshold < 1:
        parser.error("--layer-major-prefill-threshold must be greater than zero")
    if arguments.prompt_cache_entries < 1:
        parser.error("--prompt-cache-entries must be greater than zero")
    if arguments.prompt_cache_memory_gib < 1:
        parser.error("--prompt-cache-memory-gib must be greater than zero")
    if not 0 <= arguments.dspark_confidence_threshold <= 1:
        parser.error("--dspark-confidence-threshold must be between zero and one")
    if arguments.dspark_slots < 30:
        parser.error("--dspark-slots must be at least 30")
    if arguments.mtp_slots < 10:
        parser.error("--mtp-slots must be at least 10")
    if arguments.dspark_prompt_cache and not arguments.dspark:
        parser.error("--dspark-prompt-cache requires --dspark")
    if arguments.dspark_hash_prefetch and not arguments.dspark:
        parser.error("--dspark-hash-prefetch requires --dspark")
    if arguments.dspark_adaptive_block and not arguments.dspark:
        parser.error("--dspark-adaptive-block requires --dspark")
    if arguments.no_dspark_fallback and not arguments.dspark:
        parser.error("--no-dspark-fallback requires --dspark")
    if arguments.dspark_sequential_verification and not arguments.dspark:
        parser.error("--dspark-sequential-verification requires --dspark")
    if arguments.dspark_sequential_verification and arguments.dspark_hash_prefetch:
        parser.error(
            "--dspark-sequential-verification cannot use --dspark-hash-prefetch"
        )
    if arguments.dspark_hybrid_verification and not arguments.dspark:
        parser.error("--dspark-hybrid-verification requires --dspark")
    if (
        arguments.dspark_hybrid_verification
        and arguments.dspark_sequential_verification
    ):
        parser.error(
            "--dspark-hybrid-verification and "
            "--dspark-sequential-verification are mutually exclusive"
        )
    if arguments.dspark and arguments.expert_route_trace:
        parser.error("--expert-route-trace currently requires DSpark to be disabled")
    if arguments.qwen_grouped_experts and not installed.is_qwen:
        parser.error("--qwen-grouped-experts requires Qwen3.8-Flash-Next")
    if arguments.qwen_next_layer_prefetch and not installed.is_qwen:
        parser.error("--qwen-next-layer-prefetch requires Qwen3.8-Flash-Next")
    if arguments.qwen_next_layer_prefetch and arguments.no_layer_major_prefill:
        parser.error("--qwen-next-layer-prefetch requires layer-major Prefill")
    if not 0 <= arguments.ane_prefill_ratio <= 1:
        parser.error("--ane-prefill-ratio must be between 0 and 1")
    config = RuntimeConfig(
        slots=arguments.slots,
        read_workers=arguments.read_workers,
        prefetch_read_workers=arguments.prefetch_read_workers,
        memory_limit_gib=arguments.memory_limit_gib,
        prefill_step_size=arguments.prefill_step_size,
        moe_prefill_step_size=arguments.moe_prefill_step_size,
        fp8_kv_cache=not arguments.bf16_kv_cache,
        layer_major_prefill=not arguments.no_layer_major_prefill,
        layer_major_prefill_threshold=arguments.layer_major_prefill_threshold,
        batched_expert_prefill=not arguments.no_batched_expert_prefill,
        ane_prefill=not arguments.no_ane_prefill,
        ane_prefill_ratio=arguments.ane_prefill_ratio,
        qwen_next_layer_prefetch=arguments.qwen_next_layer_prefetch,
        qwen_grouped_experts=(
            installed.is_qwen if arguments.qwen_grouped_experts is None
            else arguments.qwen_grouped_experts
        ),
        prompt_cache_entries=arguments.prompt_cache_entries,
        prompt_cache_memory_gib=arguments.prompt_cache_memory_gib,
        persistent_prompt_cache=not arguments.no_persistent_prompt_cache,
        prompt_cache_directory=arguments.prompt_cache_directory,
        fp4_index_cache=not arguments.no_fp4_index_cache,
        mtp_enabled=arguments.mtp,
        mtp_slots=arguments.mtp_slots,
        dspark_enabled=arguments.dspark,
        dspark_prompt_cache=arguments.dspark_prompt_cache,
        dspark_hash_prefetch=arguments.dspark_hash_prefetch,
        dspark_adaptive_block=arguments.dspark_adaptive_block,
        dspark_fallback_enabled=not arguments.no_dspark_fallback,
        dspark_sequential_verification=(
            arguments.dspark_sequential_verification
        ),
        dspark_hybrid_verification=arguments.dspark_hybrid_verification,
        dspark_slots=arguments.dspark_slots,
        dspark_confidence_threshold=arguments.dspark_confidence_threshold,
        expert_route_trace=arguments.expert_route_trace,
        expert_page_cache_probe=arguments.expert_page_cache_probe,
        expert_file_cache_policy=arguments.expert_file_cache_policy,
        ready_expert_decode=not arguments.no_ready_expert_decode,
        power_saving_limit_gbps=arguments.power_saving_limit_gbps,
    )
    runtime = ModelRuntime.open(arguments.model, config)
    prompt_token_sha256 = _token_sha256(runtime._encode_prompt(prompt_text))

    started = time.perf_counter()
    generated = 0
    generated_token_ids: list[int] = []
    prompt_tokens = 0
    try:
        for response in runtime.stream(
            prompt_text,
            GenerationOptions(
                max_tokens=arguments.max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                approximation_mode=approximation_mode,
            ),
        ):
            sys.stdout.write(response.text)
            sys.stdout.flush()
            generated += 1
            generated_token_ids.append(int(response.token))
            prompt_tokens = response.prompt_tokens
    finally:
        elapsed = time.perf_counter() - started
        metrics = runtime.expert_cache.metrics
        runtime_metrics = runtime.metrics.snapshot()
        prefill_kernel_tokens = runtime_metrics["layer_major_prefill_tokens"]
        selected_moe_step_size = (
            _select_moe_step_size(
                config.moe_prefill_step_size,
                prefill_kernel_tokens,
            )
            if prefill_kernel_tokens
            else 0
        )
        selected_attention_step_size = runtime_metrics["request_prefill_step_size"]
        attention_chunk_sizes = [
            min(selected_attention_step_size, prefill_kernel_tokens - start)
            for start in range(
                0,
                prefill_kernel_tokens,
                selected_attention_step_size or 1,
            )
        ]
        moe_chunk_sizes = [
            min(selected_moe_step_size, prefill_kernel_tokens - start)
            for start in range(0, prefill_kernel_tokens, selected_moe_step_size or 1)
        ]
        result = {
            "generated_tokens": generated,
            "generated_token_ids": generated_token_ids,
            "prompt_token_sha256": prompt_token_sha256,
            "token_sha256": _token_sha256(generated_token_ids),
            "prompt_tokens": prompt_tokens,
            "seconds": elapsed,
            "tokens_per_second": generated / elapsed if elapsed else 0.0,
            "expert_cache_hit_rate": metrics.hit_rate,
            "expert_cache_hits": metrics.hits,
            "expert_cache_misses": metrics.misses,
            "expert_resident_slots": runtime.expert_cache.resident_count,
            "expert_bytes_read": metrics.bytes_read,
            "expert_page_cache_probe_calls": metrics.page_cache_probe_calls,
            "expert_page_cache_probe_failures": (
                metrics.page_cache_probe_failures
            ),
            "expert_page_cache_classified_bytes": (
                metrics.page_cache_classified_bytes
            ),
            "expert_page_cache_resident_bytes_before_read": (
                metrics.page_cache_resident_bytes_before_read
            ),
            "expert_page_cache_nonresident_bytes_before_read": (
                metrics.page_cache_nonresident_bytes_before_read
            ),
            "expert_page_cache_unclassified_bytes": (
                metrics.page_cache_unclassified_bytes
            ),
            "expert_page_cache_resident_fraction_before_read": (
                metrics.page_cache_resident_bytes_before_read
                / metrics.page_cache_classified_bytes
                if metrics.page_cache_classified_bytes
                else None
            ),
            "expert_page_cache_nonresident_fraction_before_read": (
                metrics.page_cache_nonresident_bytes_before_read
                / metrics.page_cache_classified_bytes
                if metrics.page_cache_classified_bytes
                else None
            ),
            "expert_bytes_per_token": (
                metrics.bytes_read / (prompt_tokens + generated)
                if prompt_tokens + generated
                else 0.0
            ),
            "expert_read_seconds": metrics.read_seconds,
            "expert_upload_seconds": metrics.upload_seconds,
            "expert_pack_seconds": metrics.pack_seconds,
            "expert_eviction_seconds": metrics.eviction_seconds,
            "routing_sync_seconds": metrics.routing_sync_seconds,
            "expert_evictions": metrics.evictions,
            "peak_memory_bytes": mx.get_peak_memory(),
            "fp8_kv_cache": config.fp8_kv_cache,
            "prefill_step_size": config.prefill_step_size,
            "layer_major_prefill_threshold": config.layer_major_prefill_threshold,
            "moe_prefill_step_size": config.moe_prefill_step_size,
            "selected_moe_prefill_step_size": selected_moe_step_size,
            "prefill_attention_chunk_sizes": attention_chunk_sizes,
            "prefill_moe_chunk_sizes": moe_chunk_sizes,
            "batched_expert_prefill": config.batched_expert_prefill,
            "qwen_grouped_experts": bool(
                config.qwen_grouped_experts
                and installed.is_qwen
                and not config.mtp_enabled
            ),
            "ane_prefill": config.ane_prefill,
            "ane_prefill_ratio": config.ane_prefill_ratio,
            "qwen_next_layer_prefetch": config.qwen_next_layer_prefetch,
            "fp4_index_cache": config.fp4_index_cache,
            "ready_expert_decode": config.ready_expert_decode,
            "expert_page_cache_probe": config.expert_page_cache_probe,
            "expert_file_cache_policy": config.expert_file_cache_policy,
            "expert_file_direct_io_alignment_bytes": (
                runtime.expert_cache.direct_io_alignment
            ),
            "power_saving_limit_gbps": config.power_saving_limit_gbps,
            "mtp_available": runtime.installed.has_mtp,
            "mtp_enabled": config.mtp_enabled and runtime.installed.has_mtp,
            "mtp_slots": config.mtp_slots,
            "dspark_enabled": config.dspark_enabled and runtime.installed.has_dspark,
            "dspark_prompt_cache": config.dspark_prompt_cache,
            "dspark_hash_prefetch": config.dspark_hash_prefetch,
            "dspark_adaptive_block": config.dspark_adaptive_block,
            "dspark_fallback_enabled": config.dspark_fallback_enabled,
            "dspark_sequential_verification": (
                config.dspark_sequential_verification
            ),
            "dspark_hybrid_verification": config.dspark_hybrid_verification,
            "dspark_hash_prefetch_scratch_slots": (
                runtime.expert_cache.speculative_slots
            ),
            "dspark_slots": config.dspark_slots,
            **runtime_metrics,
        }
        sys.stderr.write("\n" + json.dumps(result, indent=2) + "\n")
        if arguments.metrics_json:
            with open(arguments.metrics_json, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
                file.write("\n")
        runtime.close()


if __name__ == "__main__":
    main()
