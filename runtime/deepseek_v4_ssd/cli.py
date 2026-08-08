from __future__ import annotations

import argparse
import json
import sys
import time

import mlx.core as mx

from .generation import GenerationOptions, ModelRuntime
from .model import RuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepSeek-V4 from an installed model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=272_000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--slots", type=int, default=512)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--prefetch-read-workers", type=int, default=2)
    parser.add_argument("--prefill-step-size", type=int, default=0)
    parser.add_argument("--moe-prefill-step-size", type=int, default=0)
    parser.add_argument("--no-layer-major-prefill", action="store_true")
    parser.add_argument("--no-batched-expert-prefill", action="store_true")
    parser.add_argument("--prompt-cache-entries", type=int, default=2)
    parser.add_argument("--prompt-cache-memory-gib", type=int, default=8)
    parser.add_argument("--no-persistent-prompt-cache", action="store_true")
    parser.add_argument("--prompt-cache-directory")
    parser.add_argument("--bf16-kv-cache", action="store_true")
    parser.add_argument("--no-fp4-index-cache", action="store_true")
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument("--dspark-slots", type=int, default=256)
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--metrics-json")
    parser.add_argument("--expert-route-trace")
    parser.add_argument("--no-ready-expert-decode", action="store_true")
    arguments = parser.parse_args()
    if arguments.max_tokens < 1:
        parser.error("--max-tokens must be greater than zero")
    if not 0 <= arguments.temperature <= 2:
        parser.error("--temperature must be between zero and two")
    if not 0 < arguments.top_p <= 1:
        parser.error("--top-p must be greater than zero and at most one")
    if arguments.slots < 6:
        parser.error("--slots must be at least 6")
    if arguments.read_workers < 1:
        parser.error("--read-workers must be greater than zero")
    if arguments.prefetch_read_workers < 1:
        parser.error("--prefetch-read-workers must be greater than zero")
    if arguments.prefill_step_size < 0:
        parser.error("--prefill-step-size must be zero or greater")
    if arguments.moe_prefill_step_size < 0:
        parser.error("--moe-prefill-step-size must be zero or greater")
    if arguments.prompt_cache_entries < 1:
        parser.error("--prompt-cache-entries must be greater than zero")
    if arguments.prompt_cache_memory_gib < 1:
        parser.error("--prompt-cache-memory-gib must be greater than zero")
    if not 0 <= arguments.dspark_confidence_threshold <= 1:
        parser.error("--dspark-confidence-threshold must be between zero and one")
    if arguments.dspark_slots < 30:
        parser.error("--dspark-slots must be at least 30")
    if arguments.dspark and arguments.expert_route_trace:
        parser.error("--expert-route-trace currently requires DSpark to be disabled")

    config = RuntimeConfig(
        slots=arguments.slots,
        read_workers=arguments.read_workers,
        prefetch_read_workers=arguments.prefetch_read_workers,
        prefill_step_size=arguments.prefill_step_size,
        moe_prefill_step_size=arguments.moe_prefill_step_size,
        fp8_kv_cache=not arguments.bf16_kv_cache,
        layer_major_prefill=not arguments.no_layer_major_prefill,
        batched_expert_prefill=not arguments.no_batched_expert_prefill,
        prompt_cache_entries=arguments.prompt_cache_entries,
        prompt_cache_memory_gib=arguments.prompt_cache_memory_gib,
        persistent_prompt_cache=not arguments.no_persistent_prompt_cache,
        prompt_cache_directory=arguments.prompt_cache_directory,
        fp4_index_cache=not arguments.no_fp4_index_cache,
        dspark_enabled=arguments.dspark,
        dspark_slots=arguments.dspark_slots,
        dspark_confidence_threshold=arguments.dspark_confidence_threshold,
        expert_route_trace=arguments.expert_route_trace,
        ready_expert_decode=not arguments.no_ready_expert_decode,
    )
    runtime = ModelRuntime.open(arguments.model, config)

    started = time.perf_counter()
    generated = 0
    prompt_tokens = 0
    try:
        for response in runtime.stream(
            arguments.prompt,
            GenerationOptions(
                max_tokens=arguments.max_tokens,
                temperature=arguments.temperature,
                top_p=arguments.top_p,
            ),
        ):
            sys.stdout.write(response.text)
            sys.stdout.flush()
            generated += 1
            prompt_tokens = response.prompt_tokens
    finally:
        elapsed = time.perf_counter() - started
        metrics = runtime.expert_cache.metrics
        result = {
            "generated_tokens": generated,
            "prompt_tokens": prompt_tokens,
            "seconds": elapsed,
            "tokens_per_second": generated / elapsed if elapsed else 0.0,
            "expert_cache_hit_rate": metrics.hit_rate,
            "expert_cache_hits": metrics.hits,
            "expert_cache_misses": metrics.misses,
            "expert_resident_slots": runtime.expert_cache.resident_count,
            "expert_bytes_read": metrics.bytes_read,
            "expert_bytes_per_token": (
                metrics.bytes_read / (prompt_tokens + generated)
                if prompt_tokens + generated
                else 0.0
            ),
            "expert_read_seconds": metrics.read_seconds,
            "expert_pack_seconds": metrics.pack_seconds,
            "expert_eviction_seconds": metrics.eviction_seconds,
            "routing_sync_seconds": metrics.routing_sync_seconds,
            "expert_evictions": metrics.evictions,
            "peak_memory_bytes": mx.get_peak_memory(),
            "fp8_kv_cache": config.fp8_kv_cache,
            "prefill_step_size": config.prefill_step_size,
            "moe_prefill_step_size": config.moe_prefill_step_size,
            "batched_expert_prefill": config.batched_expert_prefill,
            "fp4_index_cache": config.fp4_index_cache,
            "ready_expert_decode": config.ready_expert_decode,
            "dspark_enabled": config.dspark_enabled and runtime.installed.has_dspark,
            "dspark_slots": config.dspark_slots,
            **runtime.metrics.snapshot(),
        }
        sys.stderr.write("\n" + json.dumps(result, indent=2) + "\n")
        if arguments.metrics_json:
            with open(arguments.metrics_json, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
                file.write("\n")
        runtime.close()


if __name__ == "__main__":
    main()
