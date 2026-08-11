from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import CacheList, make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from .dspark import DraftResult, VerificationMetrics, generate_tokens
from .expert_cache import CacheMetrics
from .fp8_cache import MXFP8PoolingCache
from .manifest import InstalledModel
from .model import (
    RuntimeConfig,
    _cache_arrays,
    _select_prefill_step_size,
    eval_prompt_cache,
    layer_major_prefill,
    load_model,
)
from .tool_codec import AssistantTurn, ToolChoice, ToolCodec

THINK_START = "<think>"
THINK_END = "</think>"


def _route_phase(expert_cache, phase: str):
    trace_routes = getattr(expert_cache, "trace_routes", None)
    return trace_routes(phase) if callable(trace_routes) else nullcontext()


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int = 272_000
    temperature: float = 0.2
    top_p: float = 0.98


@dataclass(frozen=True)
class GeneratedPiece:
    text: str
    token: int
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


class _RawEvalCacheList(CacheList):
    @property
    def state(self):
        return _cache_arrays([self])

    @state.setter
    def state(self, value):
        for cache, saved in zip(self.caches, value):
            cache.state = saved


def _make_prompt_cache(model: Any):
    cache = make_prompt_cache(model)
    for layer_cache in cache:
        if isinstance(layer_cache, CacheList):
            layer_cache.__class__ = _RawEvalCacheList
    return cache


@dataclass
class _PromptCacheEntry:
    cache: Any
    tokens: list[int]


@dataclass(frozen=True)
class _PersistentPromptCacheEntry:
    tokens: list[int]
    path: Path
    format: int


_PROMPT_CACHE_FORMAT = 2
_SUPPORTED_PROMPT_CACHE_FORMATS = frozenset((1, _PROMPT_CACHE_FORMAT))


def _encode_cache_state(value: Any, arrays: dict[str, mx.array]) -> Any:
    if isinstance(value, mx.array):
        if value.size == 0:
            return {
                "empty_array": {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).rsplit(".", 1)[-1],
                }
            }
        name = f"state_{len(arrays)}"
        arrays[name] = value
        return {"array": name}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "items": [_encode_cache_state(item, arrays) for item in value],
            "tuple": isinstance(value, tuple),
        }
    if isinstance(value, dict):
        return {
            "mapping": [
                [str(key), _encode_cache_state(item, arrays)]
                for key, item in value.items()
            ]
        }
    raise TypeError(f"unsupported prompt cache state value: {type(value).__name__}")


def _decode_cache_state(value: Any, arrays: dict[str, mx.array]) -> Any:
    if isinstance(value, dict) and "empty_array" in value:
        description = value["empty_array"]
        dtype = getattr(mx, description["dtype"])
        return mx.empty(tuple(description["shape"]), dtype=dtype)
    if isinstance(value, dict) and "array" in value:
        return arrays[value["array"]]
    if isinstance(value, dict) and "items" in value:
        items = [_decode_cache_state(item, arrays) for item in value["items"]]
        return tuple(items) if value.get("tuple") else items
    if isinstance(value, dict) and "mapping" in value:
        return {
            key: _decode_cache_state(item, arrays)
            for key, item in value["mapping"]
        }
    return value


def _persistence_item(item: Any) -> dict[str, Any]:
    if isinstance(item, MXFP8PoolingCache):
        return {
            "kind": "mxfp8_pooling",
            "value": item.persistence_state(),
        }
    return {
        "kind": "state",
        "value": item.state,
        "meta": getattr(item, "meta_state", ""),
    }


def _persistence_cache_state(cache: Any) -> list[dict[str, Any]]:
    state = []
    for layer_cache in cache:
        if isinstance(layer_cache, CacheList):
            state.append(
                {
                    "kind": "cache_list",
                    "items": [_persistence_item(item) for item in layer_cache.caches],
                }
            )
        else:
            state.append(_persistence_item(layer_cache))
    return state


def _restore_persistence_item(target: Any, saved: dict[str, Any]) -> None:
    kind = saved.get("kind")
    if kind == "mxfp8_pooling":
        if not isinstance(target, MXFP8PoolingCache):
            raise ValueError("prompt cache type does not match the saved cache")
        target.restore_persistence_state(saved["value"])
        return
    if kind != "state":
        raise ValueError(f"unsupported prompt cache item kind: {kind}")
    target.state = saved["value"]
    meta = saved.get("meta", "")
    if meta not in (None, ""):
        target.meta_state = meta


def _restore_persistence_cache(cache: Any, state: list[dict[str, Any]]) -> None:
    if len(cache) != len(state):
        raise ValueError("prompt cache layer count does not match")
    for target, saved in zip(cache, state):
        if saved.get("kind") == "cache_list":
            if not isinstance(target, CacheList):
                raise ValueError("prompt cache structure does not match")
            items = saved["items"]
            if len(target.caches) != len(items):
                raise ValueError("prompt cache item count does not match")
            for target_item, saved_item in zip(target.caches, items):
                _restore_persistence_item(target_item, saved_item)
        else:
            _restore_persistence_item(target, saved)


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._time_to_first_token_seconds = 0.0
        self._decode_seconds = 0.0
        self._decode_cache_eval_seconds = 0.0
        self._cache_state_eval_seconds = 0.0
        self._cache_state_eval_count = 0
        self._prompt_cache_snapshot_seconds = 0.0
        self._prompt_cache_serialize_seconds = 0.0
        self._prompt_cache_write_seconds = 0.0
        self._prompt_cache_write_errors = 0
        self._prompt_cache_write_error = ""
        self._decode_latency_seconds: list[float] = []
        self._prompt_tokens = 0
        self._generation_tokens = 0
        self._prompt_cache_reused_tokens = 0
        self._prefill_step_size = 0
        self._layer_major_prefill = False
        self._request_started = 0.0
        self._request_seconds = 0.0
        self._request_active = False
        self._completed_request_count = 0
        self._expert_before = CacheMetrics()
        self._expert_request = CacheMetrics()
        self._dspark_enabled = False
        self._dspark_rounds = 0
        self._dspark_proposed_tokens = 0
        self._dspark_accepted_tokens = 0
        self._dspark_committed_tokens = 0
        self._dspark_draft_seconds = 0.0
        self._dspark_verification_seconds = 0.0
        self._dspark_cache_fork_seconds = 0.0
        self._dspark_cache_replay_seconds = 0.0
        self._dspark_cache_eval_count = 0
        self._dspark_cache_eval_bytes = 0
        self._dspark_cache_fork_layers = 0
        self._dspark_per_position_cache_copies = 0
        self._dspark_state_fetch_count = 0
        self._dspark_block_attention_layers = 0
        self._dspark_last_layer_seconds: tuple[float, ...] = ()
        self._dspark_confidence_sum = 0.0
        self._dspark_confidence_count = 0
        self._dspark_last_proposed_tokens = 0
        self._dspark_last_accepted_tokens = 0
        self._dspark_last_draft_seconds = 0.0
        self._dspark_last_verification_seconds = 0.0
        self._dspark_fallback = False
        self._dspark_cache = None
        self._dspark_expert_before = CacheMetrics()
        self._dspark_expert_request = CacheMetrics()

    def start(
        self,
        prompt_tokens: int,
        reused_tokens: int,
        prefill_step_size: int,
        layer_major_prefill_enabled: bool,
        expert_before: CacheMetrics,
        dspark_enabled: bool = False,
        dspark_cache=None,
    ) -> None:
        with self._lock:
            self._time_to_first_token_seconds = 0.0
            self._decode_seconds = 0.0
            self._decode_cache_eval_seconds = 0.0
            self._cache_state_eval_seconds = 0.0
            self._cache_state_eval_count = 0
            self._prompt_cache_snapshot_seconds = 0.0
            self._prompt_cache_serialize_seconds = 0.0
            self._prompt_cache_write_seconds = 0.0
            self._prompt_cache_write_errors = 0
            self._prompt_cache_write_error = ""
            self._decode_latency_seconds = []
            self._prompt_tokens = prompt_tokens
            self._generation_tokens = 0
            self._prompt_cache_reused_tokens = reused_tokens
            self._prefill_step_size = prefill_step_size
            self._layer_major_prefill = layer_major_prefill_enabled
            self._request_started = time.perf_counter()
            self._request_seconds = 0.0
            self._request_active = True
            self._expert_before = expert_before
            self._expert_request = CacheMetrics()
            self._dspark_enabled = dspark_enabled
            self._dspark_rounds = 0
            self._dspark_proposed_tokens = 0
            self._dspark_accepted_tokens = 0
            self._dspark_committed_tokens = 0
            self._dspark_draft_seconds = 0.0
            self._dspark_verification_seconds = 0.0
            self._dspark_cache_fork_seconds = 0.0
            self._dspark_cache_replay_seconds = 0.0
            self._dspark_cache_eval_count = 0
            self._dspark_cache_eval_bytes = 0
            self._dspark_cache_fork_layers = 0
            self._dspark_per_position_cache_copies = 0
            self._dspark_state_fetch_count = 0
            self._dspark_block_attention_layers = 0
            self._dspark_last_layer_seconds = ()
            self._dspark_confidence_sum = 0.0
            self._dspark_confidence_count = 0
            self._dspark_last_proposed_tokens = 0
            self._dspark_last_accepted_tokens = 0
            self._dspark_last_draft_seconds = 0.0
            self._dspark_last_verification_seconds = 0.0
            self._dspark_fallback = False
            self._dspark_cache = dspark_cache
            self._dspark_expert_before = (
                dspark_cache.metrics_snapshot()
                if dspark_cache is not None
                else CacheMetrics()
            )
            self._dspark_expert_request = CacheMetrics()

    def record(
        self,
        response,
        step_seconds: float,
        cache_state_eval_seconds: float,
    ) -> None:
        self.record_token(
            response.generation_tokens,
            step_seconds,
            cache_state_eval_seconds,
        )

    def record_token(
        self,
        generation_tokens: int,
        step_seconds: float,
        cache_state_eval_seconds: float,
    ) -> None:
        with self._lock:
            if generation_tokens == 1:
                self._time_to_first_token_seconds = (
                    time.perf_counter() - self._request_started
                )
            else:
                self._decode_seconds += step_seconds
                self._decode_cache_eval_seconds += cache_state_eval_seconds
                self._decode_latency_seconds.append(
                    step_seconds + cache_state_eval_seconds
                )
            self._cache_state_eval_seconds += cache_state_eval_seconds
            self._cache_state_eval_count += 1
            self._generation_tokens = generation_tokens

    def record_prompt_cache_snapshot(self, seconds: float) -> None:
        with self._lock:
            self._prompt_cache_snapshot_seconds += seconds

    def record_prompt_cache_serialize(self, seconds: float) -> None:
        with self._lock:
            self._prompt_cache_serialize_seconds += seconds

    def record_prompt_cache_write(self, seconds: float) -> None:
        with self._lock:
            self._prompt_cache_write_seconds += seconds

    def record_prompt_cache_write_error(self, error: Exception) -> None:
        with self._lock:
            self._prompt_cache_write_errors += 1
            self._prompt_cache_write_error = f"{type(error).__name__}: {error}"

    def record_dspark_round(
        self,
        draft: DraftResult,
        accepted: int,
        verification_seconds: float,
        verification: VerificationMetrics,
    ) -> None:
        with self._lock:
            self._dspark_rounds += 1
            self._dspark_proposed_tokens += len(draft.tokens)
            self._dspark_accepted_tokens += accepted
            self._dspark_committed_tokens += accepted + 1
            self._dspark_draft_seconds += draft.seconds
            self._dspark_verification_seconds += verification_seconds
            self._dspark_cache_fork_seconds += verification.cache_fork_seconds
            self._dspark_cache_replay_seconds += verification.cache_replay_seconds
            self._dspark_cache_eval_count += verification.cache_eval_count
            self._dspark_cache_eval_bytes += verification.cache_eval_bytes
            self._dspark_cache_fork_layers += verification.cache_fork_layers
            self._dspark_per_position_cache_copies += (
                verification.per_position_cache_copies
            )
            self._dspark_state_fetch_count += verification.state_fetch_count
            self._dspark_block_attention_layers += (
                verification.block_attention_layers
            )
            self._dspark_last_layer_seconds = verification.layer_seconds
            self._dspark_confidence_sum += sum(draft.confidence)
            self._dspark_confidence_count += len(draft.confidence)
            self._dspark_last_proposed_tokens = len(draft.tokens)
            self._dspark_last_accepted_tokens = accepted
            self._dspark_last_draft_seconds = draft.seconds
            self._dspark_last_verification_seconds = verification_seconds

    def record_dspark_fallback(self) -> None:
        with self._lock:
            self._dspark_fallback = True

    def finish(self, expert_after: CacheMetrics) -> None:
        with self._lock:
            self._request_seconds = time.perf_counter() - self._request_started
            self._request_active = False
            self._expert_request = expert_after.delta(self._expert_before)
            if self._dspark_cache is not None:
                self._dspark_expert_request = self._dspark_cache.metrics_snapshot().delta(
                    self._dspark_expert_before
                )
            if self._generation_tokens > 0:
                self._completed_request_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            decode_tokens = max(0, self._generation_tokens - 1)
            request_seconds = self._request_seconds
            if self._request_active:
                request_seconds = time.perf_counter() - self._request_started
            dspark_expert = self._dspark_expert_request
            if self._request_active and self._dspark_cache is not None:
                dspark_expert = self._dspark_cache.metrics_snapshot().delta(
                    self._dspark_expert_before
                )
            prefill_tokens = max(
                0, self._prompt_tokens - self._prompt_cache_reused_tokens
            )
            decode_end_to_end_seconds = (
                self._decode_seconds + self._decode_cache_eval_seconds
            )
            decode_latencies = sorted(self._decode_latency_seconds)

            def percentile(fraction: float) -> float:
                if not decode_latencies:
                    return 0.0
                position = (len(decode_latencies) - 1) * fraction
                lower = int(position)
                upper = min(lower + 1, len(decode_latencies) - 1)
                weight = position - lower
                return (
                    decode_latencies[lower] * (1 - weight)
                    + decode_latencies[upper] * weight
                )

            return {
                "runtime_prompt_tokens": self._prompt_tokens,
                "runtime_generation_tokens": self._generation_tokens,
                "prompt_cache_reused_tokens": self._prompt_cache_reused_tokens,
                "completed_request_count": self._completed_request_count,
                "request_seconds": request_seconds,
                "time_to_first_token_seconds": self._time_to_first_token_seconds,
                "prefill_tokens_per_second": (
                    prefill_tokens / self._time_to_first_token_seconds
                    if self._time_to_first_token_seconds
                    else 0.0
                ),
                "layer_major_prefill_tokens": (
                    max(0, prefill_tokens - 1)
                    if self._layer_major_prefill
                    else 0
                ),
                "decode_seconds": self._decode_seconds,
                "decode_model_step_seconds": self._decode_seconds,
                "decode_cache_eval_seconds": self._decode_cache_eval_seconds,
                "decode_end_to_end_seconds": decode_end_to_end_seconds,
                "decode_tokens_per_second": (
                    decode_tokens / decode_end_to_end_seconds
                    if decode_end_to_end_seconds
                    else 0.0
                ),
                "decode_end_to_end_tokens_per_second": (
                    decode_tokens / decode_end_to_end_seconds
                    if decode_end_to_end_seconds
                    else 0.0
                ),
                "decode_model_step_tokens_per_second": (
                    decode_tokens / self._decode_seconds if self._decode_seconds else 0.0
                ),
                "decode_latency_p50_seconds": percentile(0.5),
                "decode_latency_p95_seconds": percentile(0.95),
                "cache_state_eval_seconds": self._cache_state_eval_seconds,
                "cache_state_eval_count": self._cache_state_eval_count,
                "prompt_cache_snapshot_seconds": (
                    self._prompt_cache_snapshot_seconds
                ),
                "prompt_cache_serialize_seconds": (
                    self._prompt_cache_serialize_seconds
                ),
                "prompt_cache_write_seconds": self._prompt_cache_write_seconds,
                "prompt_cache_write_errors": self._prompt_cache_write_errors,
                "prompt_cache_write_error": self._prompt_cache_write_error,
                "request_prefill_step_size": self._prefill_step_size,
                "layer_major_prefill": self._layer_major_prefill,
                "request_expert_cache_hit_rate": self._expert_request.hit_rate,
                "request_expert_cache_hits": self._expert_request.hits,
                "request_expert_cache_misses": self._expert_request.misses,
                "request_expert_evictions": self._expert_request.evictions,
                "request_expert_bytes_read": self._expert_request.bytes_read,
                "request_expert_read_seconds": self._expert_request.read_seconds,
                "request_expert_upload_seconds": self._expert_request.upload_seconds,
                "request_expert_pack_seconds": self._expert_request.pack_seconds,
                "request_ssd_read_bytes_per_second": (
                    self._expert_request.bytes_read
                    / self._expert_request.read_seconds
                    if self._expert_request.read_seconds
                    else 0.0
                ),
                "request_routing_sync_seconds": (
                    self._expert_request.routing_sync_seconds
                ),
                "request_batched_expert_layers": self._expert_request.batched_layers,
                "request_gather_qmm_calls": self._expert_request.gather_qmm_calls,
                "request_prefetched_layer_hits": (
                    self._expert_request.prefetched_layer_hits
                ),
                "dspark_enabled": self._dspark_enabled,
                "dspark_fallback": self._dspark_fallback,
                "dspark_rounds": self._dspark_rounds,
                "dspark_proposed_tokens": self._dspark_proposed_tokens,
                "dspark_accepted_tokens": self._dspark_accepted_tokens,
                "dspark_committed_tokens": self._dspark_committed_tokens,
                "dspark_rejected_tokens": (
                    self._dspark_proposed_tokens - self._dspark_accepted_tokens
                ),
                "dspark_acceptance_rate": (
                    self._dspark_accepted_tokens / self._dspark_proposed_tokens
                    if self._dspark_proposed_tokens
                    else 0.0
                ),
                "dspark_average_accepted_length": (
                    self._dspark_accepted_tokens / self._dspark_rounds
                    if self._dspark_rounds
                    else 0.0
                ),
                "dspark_average_committed_length": (
                    self._dspark_committed_tokens / self._dspark_rounds
                    if self._dspark_rounds
                    else 0.0
                ),
                "dspark_average_confidence": (
                    self._dspark_confidence_sum / self._dspark_confidence_count
                    if self._dspark_confidence_count
                    else 0.0
                ),
                "dspark_draft_seconds": self._dspark_draft_seconds,
                "dspark_verification_seconds": self._dspark_verification_seconds,
                "dspark_cache_fork_seconds": self._dspark_cache_fork_seconds,
                "dspark_cache_replay_seconds": self._dspark_cache_replay_seconds,
                "dspark_verification_cache_eval_count": (
                    self._dspark_cache_eval_count
                ),
                "dspark_verification_cache_eval_bytes": (
                    self._dspark_cache_eval_bytes
                ),
                "dspark_cache_fork_layers": self._dspark_cache_fork_layers,
                "dspark_per_position_cache_copies": (
                    self._dspark_per_position_cache_copies
                ),
                "dspark_verification_state_fetch_count": (
                    self._dspark_state_fetch_count
                ),
                "dspark_block_attention_layers": (
                    self._dspark_block_attention_layers
                ),
                "dspark_last_verification_layer_seconds": (
                    self._dspark_last_layer_seconds
                ),
                "dspark_last_proposed_tokens": self._dspark_last_proposed_tokens,
                "dspark_last_accepted_tokens": self._dspark_last_accepted_tokens,
                "dspark_last_draft_seconds": self._dspark_last_draft_seconds,
                "dspark_last_verification_seconds": (
                    self._dspark_last_verification_seconds
                ),
                "dspark_seconds_per_output_token": (
                    (self._dspark_draft_seconds + self._dspark_verification_seconds)
                    / (self._dspark_accepted_tokens + self._dspark_rounds)
                    if self._dspark_rounds
                    else 0.0
                ),
                "dspark_expert_cache_hit_rate": dspark_expert.hit_rate,
                "dspark_expert_cache_hits": dspark_expert.hits,
                "dspark_expert_cache_misses": dspark_expert.misses,
                "dspark_expert_evictions": dspark_expert.evictions,
                "dspark_expert_bytes_read": dspark_expert.bytes_read,
                "dspark_expert_read_seconds": dspark_expert.read_seconds,
                "dspark_expert_upload_seconds": dspark_expert.upload_seconds,
                "dspark_expert_pack_seconds": dspark_expert.pack_seconds,
                "dspark_expert_resident_slots": (
                    self._dspark_cache.resident_count
                    if self._dspark_cache is not None
                    else 0
                ),
                "dspark_expert_capacity_slots": (
                    self._dspark_cache.slots
                    if self._dspark_cache is not None
                    else 0
                ),
                "active_memory_bytes": mx.get_active_memory(),
                "cache_memory_bytes": mx.get_cache_memory(),
                "peak_memory_bytes": mx.get_peak_memory(),
            }


class ModelRuntime:
    """Keep one installed model resident and serialize all generation."""

    def __init__(self, installed: InstalledModel, config: RuntimeConfig):
        self.installed = installed
        self.config = config
        self.metrics = RuntimeMetrics()
        self._codec: ToolCodec | None = None
        self._prompt_caches: list[_PromptCacheEntry] = []
        self._persistent_prompt_caches: list[_PersistentPromptCacheEntry] = []
        self._prompt_cache_directory: Path | None = None
        self._generation_lock = threading.Lock()
        self._generation_stream = mx.new_thread_unsafe_stream(mx.gpu)
        with mx.stream(self._generation_stream):
            self.model, self.expert_cache = load_model(installed, config)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    installed.root / "tokenizer",
                    trust_remote_code=True,
                )
            except Exception:
                dspark = getattr(self.model, "dspark", None)
                if dspark is not None:
                    dspark.expert_cache.close()
                self.expert_cache.close()
                raise
        self._prompt_cache_directory = self._open_prompt_cache_directory()
        if self._prompt_cache_directory is not None:
            self._persistent_prompt_caches = self._scan_persistent_prompt_caches()
        # ponytail: the lock serializes graph evaluation as required by the
        # cross-thread MLX stream and remains correct for batch size 1.

    @classmethod
    def open(
        cls,
        model_path: str,
        config: RuntimeConfig = RuntimeConfig(),
    ) -> ModelRuntime:
        return cls(InstalledModel.open(model_path), config)

    @property
    def model_id(self) -> str:
        return self.installed.model_id

    def encode_chat(
        self,
        messages: list[dict[str, Any]],
        thinking_mode: str = "chat",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = ToolChoice(),
        reasoning_effort: str = "low",
    ) -> str:
        if self._codec is None:
            self._codec = ToolCodec.open(self.installed.root)
        return self._codec.encode(
            messages,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
        )

    def parse_chat(self, text: str, thinking_mode: str) -> AssistantTurn:
        if self._codec is None:
            self._codec = ToolCodec.open(self.installed.root)
        return self._codec.parse(text, thinking_mode)

    def stream(
        self,
        prompt: str,
        options: GenerationOptions,
    ) -> Iterator[GeneratedPiece]:
        sampler = make_sampler(
            temp=options.temperature,
            top_p=options.top_p,
        )
        with self._generation_lock:
            with mx.stream(self._generation_stream):
                prompt_tokens = self._encode_prompt(prompt)
                dspark = getattr(self.model, "dspark", None)
                entry = (
                    _PromptCacheEntry(_make_prompt_cache(self.model), [])
                    if dspark is not None
                    else self._acquire_prompt_cache(prompt_tokens)
                )
                prompt_cache = entry.cache
                cache_tokens = entry.tokens
                reused_tokens = len(cache_tokens)
                generation_prompt = prompt_tokens[reused_tokens:]
                cache_tokens.extend(generation_prompt)
                step_size = _select_prefill_step_size(
                    getattr(self.config, "prefill_step_size", 128),
                    len(generation_prompt),
                )
                use_layer_major = bool(
                    getattr(self.config, "layer_major_prefill", True)
                    and len(generation_prompt) >= 4_096
                )
                self.metrics.start(
                    len(prompt_tokens),
                    reused_tokens,
                    step_size,
                    use_layer_major,
                    self._expert_metrics(),
                    dspark_enabled=dspark is not None,
                    dspark_cache=(dspark.expert_cache if dspark is not None else None),
                )
                completed = False
                prefill_persist_entry = None
                try:
                    if dspark is not None:
                        yield from self._stream_dspark(
                            prompt_tokens,
                            prompt_cache,
                            dspark,
                            options,
                            step_size,
                        )
                        completed = True
                        return
                    if use_layer_major:
                        with _route_phase(self.expert_cache, "prefill"):
                            layer_major_prefill(
                                self.model,
                                generation_prompt[:-1],
                                prompt_cache,
                                step_size,
                                self.expert_cache,
                                getattr(self.config, "moe_prefill_step_size", 0),
                                getattr(self.config, "batched_expert_prefill", True),
                            )
                        snapshot_started = time.perf_counter()
                        prefill_persist_entry = _PromptCacheEntry(
                            copy.deepcopy(prompt_cache),
                            list(prompt_tokens[:-1]),
                        )
                        self.metrics.record_prompt_cache_snapshot(
                            time.perf_counter() - snapshot_started
                        )
                        self._store_prompt_cache(
                            prefill_persist_entry,
                            persist=False,
                        )
                        generation_prompt = generation_prompt[-1:]
                    responses = iter(
                        stream_generate(
                            self.model,
                            self.tokenizer,
                            generation_prompt,
                            max_tokens=options.max_tokens,
                            sampler=sampler,
                            prompt_cache=prompt_cache,
                            prefill_step_size=step_size,
                        )
                    )
                    first_response = True
                    while True:
                        started = time.perf_counter()
                        try:
                            phase = "prefill" if first_response else "decode"
                            with _route_phase(self.expert_cache, phase):
                                response = next(responses)
                        except StopIteration:
                            break
                        first_response = False
                        step_seconds = time.perf_counter() - started
                        cache_started = time.perf_counter()
                        eval_prompt_cache(prompt_cache)
                        cache_seconds = time.perf_counter() - cache_started
                        self.metrics.record(response, step_seconds, cache_seconds)
                        if response.finish_reason != "stop":
                            cache_tokens.append(int(response.token))
                        if response.finish_reason is not None:
                            completed = True
                        yield GeneratedPiece(
                            text=response.text,
                            token=response.token,
                            prompt_tokens=len(prompt_tokens),
                            generation_tokens=response.generation_tokens,
                            finish_reason=response.finish_reason,
                        )
                finally:
                    self.metrics.finish(self._expert_metrics())
                    if completed and dspark is None:
                        if prefill_persist_entry is not None:
                            self._persist_prompt_cache(prefill_persist_entry)
                        self._store_prompt_cache(entry, persist=True)

    def _stream_dspark(
        self,
        prompt_tokens: list[int],
        prompt_cache,
        dspark,
        options: GenerationOptions,
        step_size: int,
    ) -> Iterator[GeneratedPiece]:
        tokenizer = TokenizerWrapper(self.tokenizer)
        detokenizer = tokenizer.detokenizer
        responses = iter(
            generate_tokens(
                prompt_tokens,
                self.model,
                dspark,
                prompt_cache,
                max_tokens=options.max_tokens,
                prefill_step_size=step_size,
                temperature=options.temperature,
                top_p=options.top_p,
                confidence_threshold=getattr(
                    self.config,
                    "dspark_confidence_threshold",
                    0.6,
                ),
                record_round=self.metrics.record_dspark_round,
                record_fallback=self.metrics.record_dspark_fallback,
            )
        )
        generation_tokens = 0
        last_token = 0
        pending_text = ""
        finish_reason = "length"
        while generation_tokens < options.max_tokens:
            started = time.perf_counter()
            try:
                token, _, _ = next(responses)
            except StopIteration:
                break
            step_seconds = time.perf_counter() - started
            generation_tokens += 1
            last_token = token
            if token in tokenizer.eos_token_ids:
                finish_reason = "stop"
                break
            detokenizer.add_token(token)
            self.metrics.record_token(
                generation_tokens,
                step_seconds,
                0.0,
            )
            if generation_tokens == options.max_tokens:
                pending_text = detokenizer.last_segment
                break
            yield GeneratedPiece(
                text=detokenizer.last_segment,
                token=token,
                prompt_tokens=len(prompt_tokens),
                generation_tokens=generation_tokens,
                finish_reason=None,
            )
        detokenizer.finalize()
        final_text = detokenizer.last_segment
        if pending_text and not final_text.startswith(pending_text):
            final_text = pending_text + final_text
        self.metrics.record_token(generation_tokens, 0.0, 0.0)
        yield GeneratedPiece(
            text=final_text,
            token=last_token,
            prompt_tokens=len(prompt_tokens),
            generation_tokens=generation_tokens,
            finish_reason=finish_reason,
        )

    def warm_prompt(self, prompt: str) -> int:
        tokens = self._encode_prompt(prompt)
        if len(tokens) < 2:
            return 0
        with self._generation_lock:
            with mx.stream(self._generation_stream):
                cache = _make_prompt_cache(self.model)
                step_size = _select_prefill_step_size(
                    getattr(self.config, "prefill_step_size", 128),
                    len(tokens) - 1,
                )
                layer_major_prefill(
                    self.model,
                    tokens[:-1],
                    cache,
                    step_size,
                    self.expert_cache,
                    getattr(self.config, "moe_prefill_step_size", 0),
                    getattr(self.config, "batched_expert_prefill", True),
                )
                self._store_prompt_cache(
                    _PromptCacheEntry(cache, tokens[:-1]),
                    persist=True,
                )
        return len(tokens) - 1

    def _acquire_prompt_cache(self, prompt_tokens: list[int]) -> _PromptCacheEntry:
        matches = [
            entry
            for entry in self._prompt_caches
            if len(entry.tokens) < len(prompt_tokens)
            and prompt_tokens[: len(entry.tokens)] == entry.tokens
        ]
        if matches:
            entry = max(matches, key=lambda item: len(item.tokens))
            return _PromptCacheEntry(copy.deepcopy(entry.cache), list(entry.tokens))
        persistent = [
            entry
            for entry in self._persistent_prompt_caches
            if len(entry.tokens) < len(prompt_tokens)
            and prompt_tokens[: len(entry.tokens)] == entry.tokens
        ]
        if persistent:
            entry = max(persistent, key=lambda item: len(item.tokens))
            loaded = self._load_persistent_prompt_cache(entry)
            if loaded is not None:
                return loaded
        return _PromptCacheEntry(_make_prompt_cache(self.model), [])

    def _store_prompt_cache(
        self,
        entry: _PromptCacheEntry,
        *,
        persist: bool = False,
    ) -> None:
        self._prompt_caches = [
            cached for cached in self._prompt_caches if cached.tokens != entry.tokens
        ]
        self._prompt_caches.insert(0, entry)
        maximum = max(1, int(getattr(self.config, "prompt_cache_entries", 2)))
        memory_limit = max(
            1,
            int(getattr(self.config, "prompt_cache_memory_gib", 8)),
        ) * 1024**3
        while len(self._prompt_caches) > maximum:
            self._prompt_caches.pop()
        while len(self._prompt_caches) > 1 and self._prompt_cache_bytes() > memory_limit:
            self._prompt_caches.pop()
        if persist:
            self._persist_prompt_cache(entry)

    def _prompt_cache_bytes(self) -> int:
        return sum(
            sum(int(getattr(cache, "nbytes", 0)) for cache in entry.cache)
            for entry in self._prompt_caches
        )

    def _expert_metrics(self) -> CacheMetrics:
        snapshot = getattr(self.expert_cache, "metrics_snapshot", None)
        return snapshot() if snapshot is not None else CacheMetrics()

    def _encode_prompt(self, prompt: str) -> list[int]:
        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
        )
        return list(
            self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        )

    def _open_prompt_cache_directory(self) -> Path | None:
        if not getattr(self.config, "persistent_prompt_cache", True):
            return None
        revision = getattr(self.installed, "revision", None)
        if not revision:
            return None
        configured = getattr(self.config, "prompt_cache_directory", None)
        root = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".dsmodel" / "prompt-cache"
        )
        directory = root / revision
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return directory

    def _scan_persistent_prompt_caches(self) -> list[_PersistentPromptCacheEntry]:
        directory = self._prompt_cache_directory
        if directory is None:
            return []
        entries = []
        for metadata_path in sorted(
            directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                data_path = directory / metadata["data"]
                cache_format = int(metadata.get("format", 0))
                if (
                    cache_format in _SUPPORTED_PROMPT_CACHE_FORMATS
                    and metadata.get("revision") == self.installed.revision
                    and data_path.is_file()
                ):
                    entries.append(
                        _PersistentPromptCacheEntry(
                            [int(token) for token in metadata["tokens"]],
                            data_path,
                            cache_format,
                        )
                    )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        maximum = max(
            1,
            int(getattr(self.config, "persistent_prompt_cache_entries", 8)),
        )
        return entries[:maximum]

    def _load_persistent_prompt_cache(
        self,
        entry: _PersistentPromptCacheEntry,
    ) -> _PromptCacheEntry | None:
        try:
            arrays, metadata = mx.load(entry.path, return_metadata=True)
            schema = json.loads(metadata["state"])
            state = _decode_cache_state(schema, arrays)
            cache = _make_prompt_cache(self.model)
            if entry.format == 1:
                if len(cache) != len(state):
                    return None
                for target, saved in zip(cache, state):
                    target.state = saved
            else:
                _restore_persistence_cache(cache, state)
            eval_prompt_cache(cache)
            return _PromptCacheEntry(cache, list(entry.tokens))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _persist_prompt_cache(self, entry: _PromptCacheEntry) -> None:
        directory = self._prompt_cache_directory
        if directory is None or not entry.tokens:
            return
        digest = hashlib.sha256(
            json.dumps(entry.tokens, separators=(",", ":")).encode()
        ).hexdigest()
        stem = f"{digest}.v{_PROMPT_CACHE_FORMAT}"
        data_path = directory / f"{stem}.safetensors"
        metadata_path = directory / f"{stem}.json"
        if data_path.exists() and metadata_path.exists():
            return
        serialize_started = time.perf_counter()
        arrays: dict[str, mx.array] = {}
        state = _persistence_cache_state(entry.cache)
        schema = _encode_cache_state(state, arrays)
        mx.eval(*arrays.values())
        self.metrics.record_prompt_cache_serialize(
            time.perf_counter() - serialize_started
        )
        metadata = {
            "format": _PROMPT_CACHE_FORMAT,
            "revision": self.installed.revision,
            "tokens": entry.tokens,
            "data": data_path.name,
        }

        def save() -> None:
            write_started = time.perf_counter()
            temporary_data = data_path.with_name(data_path.stem + ".tmp.safetensors")
            temporary_metadata = metadata_path.with_name(
                metadata_path.stem + ".tmp.json"
            )
            try:
                mx.save_safetensors(
                    temporary_data,
                    arrays,
                    metadata={"state": json.dumps(schema, separators=(",", ":"))},
                )
                temporary_metadata.write_text(
                    json.dumps(metadata, separators=(",", ":")),
                    encoding="utf-8",
                )
                os.replace(temporary_data, data_path)
                os.replace(temporary_metadata, metadata_path)
                maximum = max(
                    1,
                    int(
                        getattr(
                            self.config,
                            "persistent_prompt_cache_entries",
                            8,
                        )
                    ),
                )
                saved = sorted(
                    directory.glob("*.json"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                for stale_metadata in saved[maximum:]:
                    try:
                        stale = json.loads(
                            stale_metadata.read_text(encoding="utf-8")
                        )
                        stale_data = directory / stale["data"]
                        if stale_data.parent == directory:
                            stale_data.unlink(missing_ok=True)
                        stale_metadata.unlink(missing_ok=True)
                    except (KeyError, OSError, json.JSONDecodeError):
                        continue
            except Exception as error:
                self.metrics.record_prompt_cache_write_error(error)
                for path in (temporary_data, temporary_metadata):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            finally:
                self.metrics.record_prompt_cache_write(
                    time.perf_counter() - write_started
                )

        save()

    def close(self) -> None:
        self._prompt_caches.clear()
        dspark = getattr(self.model, "dspark", None)
        if dspark is not None:
            dspark.expert_cache.close()
        self.expert_cache.close()

    def __enter__(self) -> ModelRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
