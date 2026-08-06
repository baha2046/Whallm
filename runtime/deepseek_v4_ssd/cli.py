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
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--prefill-step-size", type=int, default=32)
    parser.add_argument("--bf16-kv-cache", action="store_true")
    parser.add_argument("--metrics-json")
    arguments = parser.parse_args()
    if arguments.max_tokens < 1:
        parser.error("--max-tokens must be greater than zero")
    if arguments.slots < 6:
        parser.error("--slots must be at least 6")
    if arguments.read_workers < 1:
        parser.error("--read-workers must be greater than zero")
    if arguments.prefill_step_size < 1:
        parser.error("--prefill-step-size must be greater than zero")

    config = RuntimeConfig(
        slots=arguments.slots,
        read_workers=arguments.read_workers,
        prefill_step_size=arguments.prefill_step_size,
        fp8_kv_cache=not arguments.bf16_kv_cache,
    )
    runtime = ModelRuntime.open(arguments.model, config)

    started = time.perf_counter()
    generated = 0
    prompt_tokens = 0
    try:
        for response in runtime.stream(
            arguments.prompt,
            GenerationOptions(max_tokens=arguments.max_tokens),
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
        }
        sys.stderr.write("\n" + json.dumps(result, indent=2) + "\n")
        if arguments.metrics_json:
            with open(arguments.metrics_json, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
                file.write("\n")
        runtime.close()


if __name__ == "__main__":
    main()
