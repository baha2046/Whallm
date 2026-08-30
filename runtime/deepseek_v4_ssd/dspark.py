from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import deepseek_v4
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.hyper_connection import HyperConnection, HyperHead, hc_expand
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

from .expert_cache import (
    CacheMetrics,
    ExpertCache,
    ExpertUnionProfile,
    SpeculativePrefetchMetrics,
)


class _ResidentMoE(nn.Module):
    def __init__(
        self,
        config: deepseek_v4.ModelArgs,
        layer: int,
        expert_cache: ExpertCache,
    ):
        super().__init__()
        self.gate = deepseek_v4.MoEGate(config, layer)
        from .model import _StreamingSwitchGLU

        self.switch_mlp = _StreamingSwitchGLU(
            layer - 43,
            expert_cache,
            deepseek_v4.LimitedSwiGLU(config.swiglu_limit),
        )
        self.shared_experts = deepseek_v4.DeepseekV4MLP(
            config,
            intermediate_size=(
                config.moe_intermediate_size * config.n_shared_experts
            ),
        )

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        from .model import _streaming_moe

        return _streaming_moe(self, x, input_ids)


class _DSparkAttention(deepseek_v4.LocalAttention):
    def __init__(self, config: deepseek_v4.ModelArgs, layer: int):
        super().__init__(config, layer)
        self.context_cache = RotatingKVCache(max_size=config.sliding_window)

    def reset(self) -> None:
        self.context_cache = RotatingKVCache(max_size=self.config.sliding_window)

    def cache_state(self):
        cache = self.context_cache
        if cache.keys is None:
            return None
        return (
            tuple(value + mx.zeros((), value.dtype) for value in cache.state),
            cache.meta_state,
        )

    def restore_cache_state(self, state) -> None:
        self.reset()
        if state is None:
            return
        values, metadata = state
        self.context_cache.state = values
        self.context_cache.meta_state = metadata

    def _kv(self, x: mx.array, offset: int) -> mx.array:
        batch, length, _ = x.shape
        kv = self.kv_norm(self.wkv(x)).reshape(batch, 1, length, self.head_dim)
        return self.rope(kv, offset)

    def prefill_context(self, main_x: mx.array, offset: int) -> None:
        kv = self._kv(main_x, offset)
        self.context_cache.update_and_fetch(
            kv,
            mx.zeros((*kv.shape[:-1], 0), dtype=kv.dtype),
        )

    def __call__(
        self,
        x: mx.array,
        main_x: mx.array,
        start_pos: int,
    ) -> mx.array:
        batch, length, _ = x.shape
        main_kv = self._kv(main_x, start_pos)
        context_kv, _ = self.context_cache.update_and_fetch(
            main_kv,
            mx.zeros((*main_kv.shape[:-1], 0), dtype=main_kv.dtype),
        )
        draft_offset = start_pos + main_x.shape[1]
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(batch, length, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        q = self.rope(q.transpose(0, 2, 1, 3), draft_offset)
        draft_kv = self._kv(x, draft_offset)
        kv = mx.concatenate([context_kv, draft_kv], axis=2)
        output = deepseek_v4.scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        output = self.rope(output, draft_offset, inverse=True)
        output = output.reshape(batch, self.o_groups, -1, length, self.head_dim)
        output = output.transpose(0, 1, 3, 2, 4).flatten(-2)
        output = self.wo_a(output)
        output = output.transpose(0, 2, 1, 3).flatten(-2)
        return self.wo_b(output)


class _DSparkBlock(nn.Module):
    def __init__(
        self,
        config: deepseek_v4.ModelArgs,
        layer: int,
        expert_cache: ExpertCache,
    ):
        super().__init__()
        self.attn = _DSparkAttention(config, layer)
        self.ffn = _ResidentMoE(config, layer + 43, expert_cache)
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        hidden: mx.array,
        input_ids: mx.array,
        main_x: mx.array,
        start_pos: int,
    ) -> mx.array:
        residual = hidden
        value, post, combine = self.attn_hc(hidden)
        value = self.attn(self.attn_norm(value), main_x, start_pos)
        hidden = hc_expand(value, residual, post, combine)
        residual = hidden
        value, post, combine = self.ffn_hc(hidden)
        value = self.ffn(self.ffn_norm(value), input_ids)
        return hc_expand(value, residual, post, combine)


class _MarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def __call__(self, token: mx.array) -> tuple[mx.array, mx.array]:
        embedding = self.markov_w1(token)
        return self.markov_w2(embedding.astype(mx.float32)), embedding


class _ConfidenceHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, 1, bias=False)

    def __call__(self, hidden: mx.array, markov: mx.array) -> mx.array:
        return self.proj(mx.concatenate([hidden, markov], axis=-1).astype(mx.float32))[
            ..., 0
        ]


@dataclass(frozen=True)
class DraftResult:
    tokens: list[int]
    logprobs: list[mx.array]
    confidence: list[float]
    seconds: float


@dataclass(frozen=True)
class VerificationMetrics:
    """Measurements for one target verification transaction, including replay."""

    verification_mode: str = "single"
    sequential_verification_positions: int = 0
    sequential_position_seconds: tuple[float, ...] = ()
    hybrid_verification_positions: int = 0
    hybrid_attention_layers: int = 0
    hybrid_attention_token_calls: int = 0
    hybrid_ffn_token_calls: int = 0
    hybrid_moe_token_calls: int = 0
    committed_tokens: int | None = None
    output_budget_trimmed_tokens: int = 0
    fallback_target_step_seconds: float = 0.0
    fallback_speculative_seconds: float = 0.0
    fallback_break_even_seconds: float = 0.0
    fallback_cost_ratio: float = 0.0
    fallback_would_trigger: bool = False
    fallback_triggered: bool = False
    cache_fork_seconds: float = 0.0
    cache_replay_seconds: float = 0.0
    layer_seconds: tuple[float, ...] = ()
    cache_eval_count: int = 0
    cache_eval_bytes: int = 0
    cache_fork_layers: int = 0
    per_position_cache_copies: int = 0
    state_fetch_count: int = 0
    block_attention_layers: int = 0
    routed_expert_assignments: int = 0
    expert_union_experts: int = 0
    expert_union_misses: int = 0
    expert_bytes_read: int = 0
    verification_expert_bytes_read: int = 0
    replay_expert_bytes_read: int = 0
    expert_read_seconds: float = 0.0
    layer_expert_union_layer_ids: tuple[int, ...] = ()
    layer_routed_expert_assignments: tuple[int, ...] = ()
    layer_expert_union_counts: tuple[int, ...] = ()
    layer_expert_union_misses: tuple[int, ...] = ()
    hash_prefetch_requested_experts: int = 0
    hash_prefetch_cache_resident_experts: int = 0
    hash_prefetch_experts_read: int = 0
    hash_prefetch_bytes_read: int = 0
    hash_prefetch_useful_bytes: int = 0
    hash_prefetch_wasted_bytes: int = 0
    hash_prefetch_page_cache_classified_bytes: int = 0
    hash_prefetch_page_cache_resident_bytes_before_read: int = 0
    hash_prefetch_page_cache_nonresident_bytes_before_read: int = 0
    hash_prefetch_page_cache_unclassified_bytes: int = 0
    hash_prefetch_useful_page_cache_resident_bytes_before_read: int = 0
    hash_prefetch_useful_page_cache_nonresident_bytes_before_read: int = 0
    hash_prefetch_useful_page_cache_unclassified_bytes: int = 0
    hash_prefetch_wasted_page_cache_resident_bytes_before_read: int = 0
    hash_prefetch_wasted_page_cache_nonresident_bytes_before_read: int = 0
    hash_prefetch_wasted_page_cache_unclassified_bytes: int = 0
    hash_prefetch_read_seconds: float = 0.0
    hash_prefetch_wait_seconds: float = 0.0
    hash_prefetch_plan_seconds: float = 0.0
    hash_prefetch_layer_ids: tuple[int, ...] = ()
    hash_prefetch_layer_union_counts: tuple[int, ...] = ()
    adaptive_block_original_tokens: int = 0
    adaptive_block_selected_tokens: int = 0
    adaptive_block_selected_expected_committed: float = 0.0
    adaptive_block_selected_requested_hash_experts: int = 0
    adaptive_block_selected_resident_hash_experts: int = 0
    adaptive_block_selected_missing_hash_experts: int = 0
    adaptive_block_selected_predicted_hash_bytes: int = 0
    adaptive_block_selected_score: float = 0.0
    adaptive_block_selection_reason: str = ""
    adaptive_block_full_commit_fraction: float = 0.0
    adaptive_block_plan_seconds: float = 0.0
    adaptive_block_candidate_tokens: tuple[int, ...] = ()
    adaptive_block_candidate_expected_committed: tuple[float, ...] = ()
    adaptive_block_candidate_requested_hash_experts: tuple[int, ...] = ()
    adaptive_block_candidate_resident_hash_experts: tuple[int, ...] = ()
    adaptive_block_candidate_missing_hash_experts: tuple[int, ...] = ()
    adaptive_block_candidate_predicted_hash_bytes: tuple[int, ...] = ()
    adaptive_block_candidate_scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class HashExpertLayerRoutes:
    layer: int
    routes_by_position: tuple[tuple[int, ...], ...]
    expert_union: tuple[int, ...]


@dataclass(frozen=True)
class HashExpertPrefetchPlan:
    layers: tuple[HashExpertLayerRoutes, ...] = ()
    resolve_seconds: float = 0.0

    @property
    def experts_by_layer(self) -> dict[int, tuple[int, ...]]:
        return {layer.layer: layer.expert_union for layer in self.layers}

    def keys_for_draft_tokens(self, draft_tokens: int) -> set[tuple[int, int]]:
        """Return hash-route keys for anchor plus a draft prefix."""
        if draft_tokens < 0:
            raise ValueError("draft token count must be zero or greater")
        keys: set[tuple[int, int]] = set()
        input_count = draft_tokens + 1
        for layer in self.layers:
            for routes in layer.routes_by_position[:input_count]:
                keys.update((layer.layer, expert) for expert in routes)
        return keys

    def useful_keys(self, accepted_draft_tokens: int) -> set[tuple[int, int]]:
        return self.keys_for_draft_tokens(accepted_draft_tokens)

    def prefix(self, draft_tokens: int) -> HashExpertPrefetchPlan:
        """Trim a full-block plan to anchor plus ``draft_tokens`` inputs."""
        if draft_tokens < 0:
            raise ValueError("draft token count must be zero or greater")
        input_count = draft_tokens + 1
        layers: list[HashExpertLayerRoutes] = []
        for layer in self.layers:
            routes = layer.routes_by_position[:input_count]
            union = tuple(
                dict.fromkeys(expert for position in routes for expert in position)
            )
            layers.append(HashExpertLayerRoutes(layer.layer, routes, union))
        return HashExpertPrefetchPlan(tuple(layers), self.resolve_seconds)


@dataclass(frozen=True)
class AdaptiveBlockCandidate:
    draft_tokens: int
    expected_committed_tokens: float
    requested_hash_experts: int
    resident_hash_experts: int
    missing_hash_experts: int
    predicted_hash_bytes: int
    score: float


@dataclass(frozen=True)
class AdaptiveBlockDecision:
    original_tokens: int = 0
    selected_tokens: int = 0
    candidates: tuple[AdaptiveBlockCandidate, ...] = ()
    selection_reason: str = ""
    full_commit_fraction: float = 0.0
    plan_seconds: float = 0.0

    @property
    def selected_candidate(self) -> AdaptiveBlockCandidate | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.draft_tokens == self.selected_tokens
            ),
            None,
        )


def _expected_committed_tokens(confidence: list[float], length: int) -> float:
    """Estimate accepted draft tokens plus the correction or bonus token."""
    expected = 1.0
    survival = 1.0
    for value in confidence[:length]:
        survival *= min(1.0, max(0.0, value))
        expected += survival
    return expected


def _adaptive_candidate_lengths(length: int) -> tuple[int, ...]:
    if length < 1:
        return ()
    return tuple(sorted({size for size in (1, 2, 4, length) if size <= length}))


# The first five-workload calibration over-trimmed drafts that the target later
# accepted completely. Keep a full block when confidence predicts at least 90%
# utilization; lower-confidence blocks still use the storage score below.
_ADAPTIVE_FULL_BLOCK_COMMIT_FRACTION_FLOOR = 0.90


def _select_adaptive_block(
    draft: DraftResult,
    plan: HashExpertPrefetchPlan,
    resident_keys: set[tuple[int, int]] | frozenset[tuple[int, int]],
    expert_blob_size: int,
) -> AdaptiveBlockDecision:
    """Choose a draft prefix by expected commits per missing hash expert.

    The first prototype deliberately scores only the exact, observable hash-layer
    routes. It does not pretend to predict the learned-router layers.
    """
    if expert_blob_size < 1:
        raise ValueError("expert blob size must be greater than zero")
    started = time.perf_counter()
    original_tokens = len(draft.tokens)
    lengths = _adaptive_candidate_lengths(original_tokens)
    if not lengths or not plan.layers:
        return AdaptiveBlockDecision(
            original_tokens=original_tokens,
            selected_tokens=original_tokens,
            plan_seconds=plan.resolve_seconds + time.perf_counter() - started,
        )

    candidates: list[AdaptiveBlockCandidate] = []
    for length in lengths:
        keys = plan.keys_for_draft_tokens(length)
        resident = len(keys.intersection(resident_keys))
        missing = len(keys) - resident
        expected = _expected_committed_tokens(draft.confidence, length)
        candidates.append(
            AdaptiveBlockCandidate(
                draft_tokens=length,
                expected_committed_tokens=expected,
                requested_hash_experts=len(keys),
                resident_hash_experts=resident,
                missing_hash_experts=missing,
                predicted_hash_bytes=missing * expert_blob_size,
                score=expected / max(1, missing),
            )
        )
    storage_selected = max(
        candidates,
        key=lambda candidate: (
            candidate.score,
            candidate.expected_committed_tokens,
            candidate.draft_tokens,
        ),
    )
    full_candidate = candidates[-1]
    full_commit_fraction = (
        full_candidate.expected_committed_tokens
        / (full_candidate.draft_tokens + 1)
    )
    if full_commit_fraction >= _ADAPTIVE_FULL_BLOCK_COMMIT_FRACTION_FLOOR:
        selected = full_candidate
        selection_reason = "high_confidence_full"
    else:
        selected = storage_selected
        selection_reason = "storage_score"
    return AdaptiveBlockDecision(
        original_tokens=original_tokens,
        selected_tokens=selected.draft_tokens,
        candidates=tuple(candidates),
        selection_reason=selection_reason,
        full_commit_fraction=full_commit_fraction,
        plan_seconds=plan.resolve_seconds + time.perf_counter() - started,
    )


def _truncate_draft(draft: DraftResult, length: int) -> DraftResult:
    if not 0 <= length <= len(draft.tokens):
        raise ValueError("draft prefix length is out of range")
    return DraftResult(
        tokens=draft.tokens[:length],
        logprobs=draft.logprobs[:length],
        confidence=draft.confidence[:length],
        seconds=draft.seconds,
    )


def _hash_expert_prefetch_plan(
    main_model,
    token_ids: list[int],
) -> HashExpertPrefetchPlan:
    """Resolve the exact checkpoint tid2eid routes for target hash layers."""
    if not token_ids:
        return HashExpertPrefetchPlan()
    started = time.perf_counter()
    core = getattr(main_model, "model", main_model)
    layers = getattr(core, "pipeline_layers", None)
    if layers is None:
        layers = getattr(main_model, "layers", ())
    layers = tuple(layers)
    args = getattr(main_model, "args", getattr(core, "args", None))
    hash_layer_count = int(getattr(args, "num_hash_layers", 0))
    if hash_layer_count < 1:
        return HashExpertPrefetchPlan()
    if len(layers) < hash_layer_count:
        raise RuntimeError("main model does not expose every configured hash layer")

    inputs = mx.array(token_ids, dtype=mx.int32)
    planned: list[HashExpertLayerRoutes] = []
    for position in range(hash_layer_count):
        layer = layers[position]
        gate = getattr(getattr(layer, "ffn", None), "gate", None)
        if gate is None or not getattr(gate, "hash", False):
            raise RuntimeError(f"main model layer {position} is not a hash router")
        table = getattr(gate, "tid2eid", None)
        if table is None:
            raise RuntimeError(f"main model layer {position} has no tid2eid table")
        selected = table[inputs]
        mx.eval(selected)
        selected_array = np.asarray(selected, dtype=np.int32)
        if selected_array.ndim != 2 or selected_array.shape[0] != len(token_ids):
            raise RuntimeError("hash router produced an incompatible expert table")
        routes = tuple(
            tuple(int(expert) for expert in row) for row in selected_array.tolist()
        )
        union = tuple(dict.fromkeys(expert for row in routes for expert in row))
        switch_mlp = getattr(getattr(layer, "ffn", None), "switch_mlp", None)
        layer_id = int(getattr(switch_mlp, "layer", position))
        planned.append(HashExpertLayerRoutes(layer_id, routes, union))
    return HashExpertPrefetchPlan(
        tuple(planned),
        time.perf_counter() - started,
    )


class DSparkModel(nn.Module):
    def __init__(
        self,
        config: deepseek_v4.ModelArgs,
        block_size: int,
        noise_token_id: int,
        target_layers: tuple[int, ...],
        markov_rank: int,
        expert_cache: ExpertCache,
    ):
        super().__init__()
        self.block_size = block_size
        self.noise_token_id = noise_token_id
        self.target_layers = target_layers
        self.main_proj = nn.Linear(
            config.hidden_size * len(target_layers),
            config.hidden_size,
            bias=False,
        )
        self.main_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            _DSparkBlock(config, layer, expert_cache) for layer in range(3)
        ]
        self.expert_cache = expert_cache
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = HyperHead(config)
        self.markov_head = _MarkovHead(config.vocab_size, markov_rank)
        self.confidence_head = _ConfidenceHead(config.hidden_size + markov_rank)
        self.hc_mult = config.hc_mult

    def reset_cache(self) -> None:
        for layer in self.layers:
            layer.attn.reset()

    def cache_state(self):
        return tuple(layer.attn.cache_state() for layer in self.layers)

    def restore_cache_state(self, state) -> None:
        for layer, value in zip(self.layers, state or (None,) * len(self.layers)):
            layer.attn.restore_cache_state(value)

    def _main_x(self, main_hidden: mx.array) -> mx.array:
        return self.main_norm(self.main_proj(main_hidden))

    def prefill_context(self, main_hidden: mx.array, offset: int) -> None:
        main_x = self._main_x(main_hidden)
        for layer in self.layers:
            layer.attn.prefill_context(main_x, offset)

    def draft(
        self,
        main_model,
        anchor: int,
        main_hidden: mx.array,
        start_pos: int,
        temperature: float,
        top_p: float,
        confidence_threshold: float,
    ) -> DraftResult:
        started = time.perf_counter()
        main_x = self._main_x(main_hidden)
        input_ids = mx.full((1, self.block_size), self.noise_token_id, mx.int32)
        input_ids[:, 0] = anchor
        hidden = main_model.model.embed_tokens(input_ids)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (*hidden.shape[:2], self.hc_mult, hidden.shape[-1]),
        )
        hidden = mx.contiguous(hidden)
        for layer in self.layers:
            hidden = layer(hidden, input_ids, main_x, start_pos)
        head_hidden = self.hc_head(hidden)
        base_logits = main_model.lm_head(self.norm(head_hidden)).astype(mx.float32)

        previous = mx.array([anchor], dtype=mx.int32)
        token_arrays: list[mx.array] = []
        distributions: list[mx.array] = []
        markov: list[mx.array] = []
        for position in range(self.block_size):
            bias, embedding = self.markov_head(previous)
            logprobs = sampling_logprobs(
                base_logits[:, position] + bias,
                temperature,
                top_p,
            )
            token = (
                mx.argmax(logprobs, axis=-1)
                if temperature == 0
                else mx.random.categorical(logprobs)
            )
            token_arrays.append(token)
            distributions.append(logprobs[0])
            markov.append(embedding)
            previous = token.astype(mx.int32)

        confidence = mx.sigmoid(
            self.confidence_head(head_hidden, mx.stack(markov, axis=1))
        )[0]
        mx.eval(confidence, *token_arrays, *(distributions if temperature else ()))
        tokens = [int(token.item()) for token in token_arrays]
        if temperature == 0:
            empty = mx.array([], dtype=mx.float32)
            distributions = [empty] * len(tokens)
        values = [float(value) for value in confidence.tolist()]
        if confidence_threshold > 0:
            keep = _confidence_prefix_length(values, confidence_threshold)
            tokens = tokens[:keep]
            distributions = distributions[:keep]
            values = values[:keep]
        return DraftResult(
            tokens=tokens,
            logprobs=distributions,
            confidence=values,
            seconds=time.perf_counter() - started,
        )


def sampling_logprobs(
    logits: mx.array,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    min_p: float = 0.0,
) -> mx.array:
    if temperature == 0:
        return logits
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if 0 < top_p < 1:
        logprobs = apply_top_p(logprobs, top_p)
    if min_p:
        logprobs = apply_min_p(logprobs, min_p)
    if 0 < top_k < logprobs.shape[-1]:
        logprobs = apply_top_k(logprobs, top_k)
    scaled = logprobs / max(temperature, 1e-5)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def sanitize_dspark_weights(
    model: DSparkModel,
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    remapped: dict[str, mx.array] = {}
    head_prefixes = (
        "norm.",
        "markov_head.",
        "confidence_head.",
    )
    for key, value in weights.items():
        parts = key.split(".", 2)
        if len(parts) != 3 or parts[0] != "mtp":
            continue
        layer = int(parts[1])
        rest = parts[2]
        if rest.startswith(("main_proj.", "main_norm.")):
            target = rest
        elif rest.startswith(head_prefixes):
            target = rest
        elif rest == "hc_head_fn":
            target = "hc_head.fn"
        elif rest == "hc_head_base":
            target = "hc_head.base"
        elif rest == "hc_head_scale":
            target = "hc_head.scale"
        else:
            target = f"layers.{layer}.{rest}"
        target = target.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        for section in ("attn", "ffn"):
            for parameter in ("fn", "base", "scale"):
                target = target.replace(
                    f".hc_{section}_{parameter}",
                    f".{section}_hc.{parameter}",
                )
        for source, destination in (
            ("w1", "gate_proj"),
            ("w2", "down_proj"),
            ("w3", "up_proj"),
        ):
            target = target.replace(
                f".shared_experts.{source}.",
                f".shared_experts.{destination}.",
            )
        remapped[target] = value

    for key in list(remapped):
        if not key.endswith(".scale"):
            continue
        weight_key = key.removesuffix(".scale") + ".weight"
        weight = remapped.get(weight_key)
        if weight is None:
            continue
        scale = remapped.pop(key)
        if weight.dtype == mx.uint8:
            remapped[weight_key] = weight.view(mx.uint32)
            remapped[key + "s"] = mx.repeat(mx.repeat(scale, 4, -1), 128, 0)

    for layer in range(len(model.layers)):
        prefix = f"layers.{layer}.attn.wo_a"
        for suffix in ("weight", "scales", "biases"):
            key = f"{prefix}.{suffix}"
            if key in remapped and remapped[key].ndim == 2:
                remapped[key] = remapped[key].reshape(
                    model.layers[layer].attn.o_groups,
                    model.layers[layer].attn.o_lora_rank,
                    -1,
                )
    return remapped


def load_dspark_model(
    main_model,
    args: deepseek_v4.ModelArgs,
    common_weights: dict[str, mx.array],
    expert_cache: ExpertCache,
    *,
    block_size: int,
    noise_token_id: int,
    target_layers: tuple[int, ...],
    markov_rank: int,
) -> DSparkModel:
    config = replace(
        args,
        num_hidden_layers=3,
        num_hash_layers=0,
        compress_ratios=[0, 0, 0],
    )
    model = DSparkModel(
        config,
        block_size,
        noise_token_id,
        target_layers,
        markov_rank,
        expert_cache,
    )
    weights = sanitize_dspark_weights(model, common_weights)
    quantization = deepseek_v4.make_quantization_config(model)
    quantization["main_proj"] = {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }

    def class_predicate(path: str, module: nn.Module):
        if path in quantization:
            return quantization[path]
        if not hasattr(module, "to_quantized"):
            return False
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=quantization["group_size"],
        bits=quantization["bits"],
        mode=quantization["mode"],
        class_predicate=class_predicate,
    )
    model.eval()
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def generate_tokens(
    prompt: list[int],
    main_model,
    dspark: DSparkModel,
    prompt_cache,
    *,
    max_tokens: int,
    prefill_step_size: int,
    temperature: float,
    top_p: float,
    confidence_threshold: float = 0.0,
    record_round: Callable[
        [DraftResult, int, float, VerificationMetrics], None
    ]
    | None = None,
    record_fallback: Callable[[], None] | None = None,
    target_expert_cache: ExpertCache | None = None,
    hash_prefetch: bool = False,
    adaptive_block: bool = False,
    fallback_enabled: bool = True,
    sequential_verification: bool = False,
    hybrid_verification: bool = False,
    prefilled_tokens: int = 0,
    record_prefill_snapshot: Callable[[int, object, object], None] | None = None,
) -> Iterator[tuple[int, mx.array, bool]]:
    """Yield tokens after target-model verification."""
    if not prompt or max_tokens < 1:
        return
    if not 0 <= prefilled_tokens < len(prompt):
        raise ValueError(
            "prefilled DSpark prompt tokens must be a strict prompt prefix"
        )
    if adaptive_block and target_expert_cache is None:
        raise ValueError("adaptive block scheduling requires the target expert cache")
    if sequential_verification and hash_prefetch:
        raise ValueError(
            "sequential verification cannot use speculative hash prefetch"
        )
    if sequential_verification and hybrid_verification:
        raise ValueError(
            "sequential and hybrid verification are mutually exclusive"
        )
    from .model import (
        _hybrid_forward_with_hidden,
        _sequential_forward_with_hidden,
        eval_prompt_cache,
        forward_with_hidden,
    )

    if prefilled_tokens == 0:
        dspark.reset_cache()
    final_logits = final_hidden = None
    processed = prefilled_tokens
    while len(prompt) - processed > 1:
        count = min(prefill_step_size, len(prompt) - processed - 1)
        inputs = mx.array(prompt[processed : processed + count])[None]
        final_logits, final_hidden = forward_with_hidden(
            main_model,
            inputs,
            prompt_cache,
            dspark.target_layers,
        )
        dspark.prefill_context(final_hidden, processed)
        mx.eval(final_logits, final_hidden)
        processed += count
        mx.clear_cache()
    if record_prefill_snapshot is not None and processed > prefilled_tokens:
        record_prefill_snapshot(processed, prompt_cache, dspark.cache_state())
    final_logits, final_hidden = forward_with_hidden(
        main_model,
        mx.array([[prompt[-1]]], dtype=mx.int32),
        prompt_cache,
        dspark.target_layers,
    )
    dspark.prefill_context(final_hidden, processed)
    mx.eval(final_logits, final_hidden)
    assert final_logits is not None and final_hidden is not None

    logprobs = sampling_logprobs(final_logits[:, -1], temperature, top_p)[0]
    anchor = _sample(logprobs, temperature)
    yield anchor, logprobs, False
    generated = 1
    if generated >= max_tokens:
        return

    position = len(prompt)
    target_started = time.perf_counter()
    logits, main_hidden = forward_with_hidden(
        main_model,
        mx.array([[anchor]], dtype=mx.int32),
        prompt_cache,
        dspark.target_layers,
    )
    logprobs = sampling_logprobs(logits[:, -1], temperature, top_p)[0]
    anchor = _sample(logprobs, temperature)
    mx.eval(main_hidden, logprobs)
    target_step_seconds = time.perf_counter() - target_started
    yield anchor, logprobs, False
    generated += 1

    while generated < max_tokens:
        draft = dspark.draft(
            main_model,
            anchor,
            main_hidden,
            position,
            temperature,
            top_p,
            confidence_threshold,
        )
        position += main_hidden.shape[1]
        output_budget_draft_tokens = max(0, max_tokens - generated - 1)
        output_budget_trimmed_tokens = max(
            0,
            len(draft.tokens) - output_budget_draft_tokens,
        )
        if output_budget_trimmed_tokens:
            draft = _truncate_draft(draft, output_budget_draft_tokens)
        verify_started = time.perf_counter()
        expert_before = (
            target_expert_cache.metrics_snapshot()
            if target_expert_cache is not None
            else CacheMetrics()
        )
        full_verification_inputs = [anchor, *draft.tokens]
        hash_plan = (
            _hash_expert_prefetch_plan(main_model, full_verification_inputs)
            if (hash_prefetch or adaptive_block)
            and draft.tokens
            and target_expert_cache is not None
            else HashExpertPrefetchPlan()
        )
        adaptive_decision = AdaptiveBlockDecision()
        if adaptive_block and draft.tokens and target_expert_cache is not None:
            adaptive_decision = _select_adaptive_block(
                draft,
                hash_plan,
                target_expert_cache.resident_expert_keys(
                    hash_plan.experts_by_layer
                ),
                target_expert_cache.model.expert_blob_size,
            )
            draft = _truncate_draft(draft, adaptive_decision.selected_tokens)
            hash_plan = hash_plan.prefix(adaptive_decision.selected_tokens)
        verification_inputs = [anchor, *draft.tokens]
        prefetch = (
            target_expert_cache.speculative_prefetch(hash_plan.experts_by_layer)
            if hash_prefetch
            and hash_plan.layers
            and target_expert_cache is not None
            else nullcontext(None)
        )
        capture = (
            target_expert_cache.capture_expert_unions()
            if target_expert_cache is not None
            else nullcontext(None)
        )
        exact_prefetch = None
        with prefetch as exact_prefetch:
            with capture as expert_union:
                if draft.tokens:
                    (
                        logits,
                        verified_hidden,
                        verified_cache,
                        verification_metrics,
                    ) = (
                        _sequential_target_sequence(
                            main_model,
                            verification_inputs,
                            prompt_cache,
                            dspark.target_layers,
                        )
                        if sequential_verification
                        else _hybrid_target_sequence(
                            main_model,
                            verification_inputs,
                            prompt_cache,
                            dspark.target_layers,
                        )
                        if hybrid_verification
                        else _target_sequence(
                            main_model,
                            verification_inputs,
                            prompt_cache,
                            dspark.target_layers,
                        )
                    )
                else:
                    logits, verified_hidden = forward_with_hidden(
                        main_model,
                        mx.array([[anchor]], dtype=mx.int32),
                        prompt_cache,
                        dspark.target_layers,
                    )
                    verified_cache = None
                    verification_metrics = VerificationMetrics(
                        verification_mode=(
                            "sequential"
                            if sequential_verification
                            else "hybrid"
                            if hybrid_verification
                            else "single"
                        ),
                        sequential_verification_positions=(
                            1 if sequential_verification else 0
                        ),
                        hybrid_verification_positions=(
                            1 if hybrid_verification else 0
                        ),
                    )
            expert_after_verification = (
                target_expert_cache.metrics_snapshot()
                if target_expert_cache is not None
                else CacheMetrics()
            )
            target_logprobs = sampling_logprobs(logits[0], temperature, top_p)
            mx.eval(target_logprobs, verified_hidden)

            accepted, next_token, next_logprobs = _verify(
                draft,
                target_logprobs,
                temperature,
            )
            if draft.tokens:
                assert verified_cache is not None
                if accepted == len(draft.tokens):
                    prompt_cache[:] = verified_cache
                else:
                    replay_started = time.perf_counter()
                    replay_inputs = mx.array(
                        [[anchor, *draft.tokens[:accepted]]],
                        dtype=mx.int32,
                    )
                    if sequential_verification:
                        replay_logits, replay_hidden, _, _, _ = (
                            _sequential_forward_with_hidden(
                                main_model,
                                replay_inputs,
                                prompt_cache,
                                dspark.target_layers,
                            )
                        )
                    elif hybrid_verification:
                        replay_logits, replay_hidden, _, _, _ = (
                            _hybrid_forward_with_hidden(
                                main_model,
                                replay_inputs,
                                prompt_cache,
                                dspark.target_layers,
                            )
                        )
                        verified_hidden = replay_hidden
                    else:
                        replay_logits, replay_hidden = forward_with_hidden(
                            main_model,
                            replay_inputs,
                            prompt_cache,
                            dspark.target_layers,
                        )
                        mx.eval(replay_logits, replay_hidden)
                        eval_prompt_cache(prompt_cache)
                    verification_metrics = replace(
                        verification_metrics,
                        cache_replay_seconds=time.perf_counter() - replay_started,
                    )
        hash_prefetch_metrics = (
            exact_prefetch.metrics(hash_plan.useful_keys(accepted))
            if exact_prefetch is not None
            else SpeculativePrefetchMetrics()
        )
        expert_after_round = (
            target_expert_cache.metrics_snapshot()
            if target_expert_cache is not None
            else CacheMetrics()
        )
        verification_expert = expert_after_verification.delta(expert_before)
        round_expert = expert_after_round.delta(expert_before)
        profile = (
            expert_union
            if isinstance(expert_union, ExpertUnionProfile)
            else ExpertUnionProfile()
        )
        adaptive_selected = adaptive_decision.selected_candidate
        verification_metrics = replace(
            verification_metrics,
            committed_tokens=min(
                accepted + 1,
                max_tokens - generated,
            ),
            output_budget_trimmed_tokens=output_budget_trimmed_tokens,
            routed_expert_assignments=profile.routed_expert_assignments,
            expert_union_experts=profile.unique_experts,
            expert_union_misses=profile.cache_misses,
            expert_bytes_read=round_expert.bytes_read,
            verification_expert_bytes_read=verification_expert.bytes_read,
            replay_expert_bytes_read=(
                round_expert.bytes_read - verification_expert.bytes_read
            ),
            expert_read_seconds=round_expert.read_seconds,
            layer_expert_union_layer_ids=tuple(
                layer.layer for layer in profile.layers
            ),
            layer_routed_expert_assignments=tuple(
                layer.routed_expert_assignments for layer in profile.layers
            ),
            layer_expert_union_counts=tuple(
                layer.unique_experts for layer in profile.layers
            ),
            layer_expert_union_misses=tuple(
                layer.cache_misses for layer in profile.layers
            ),
            hash_prefetch_requested_experts=(
                hash_prefetch_metrics.requested_experts
            ),
            hash_prefetch_cache_resident_experts=(
                hash_prefetch_metrics.cache_resident_experts
            ),
            hash_prefetch_experts_read=hash_prefetch_metrics.experts_read,
            hash_prefetch_bytes_read=hash_prefetch_metrics.bytes_read,
            hash_prefetch_useful_bytes=hash_prefetch_metrics.useful_bytes,
            hash_prefetch_wasted_bytes=hash_prefetch_metrics.wasted_bytes,
            hash_prefetch_page_cache_classified_bytes=(
                hash_prefetch_metrics.page_cache_classified_bytes
            ),
            hash_prefetch_page_cache_resident_bytes_before_read=(
                hash_prefetch_metrics.page_cache_resident_bytes_before_read
            ),
            hash_prefetch_page_cache_nonresident_bytes_before_read=(
                hash_prefetch_metrics.page_cache_nonresident_bytes_before_read
            ),
            hash_prefetch_page_cache_unclassified_bytes=(
                hash_prefetch_metrics.page_cache_unclassified_bytes
            ),
            hash_prefetch_useful_page_cache_resident_bytes_before_read=(
                hash_prefetch_metrics.useful_page_cache_resident_bytes_before_read
            ),
            hash_prefetch_useful_page_cache_nonresident_bytes_before_read=(
                hash_prefetch_metrics.useful_page_cache_nonresident_bytes_before_read
            ),
            hash_prefetch_useful_page_cache_unclassified_bytes=(
                hash_prefetch_metrics.useful_page_cache_unclassified_bytes
            ),
            hash_prefetch_wasted_page_cache_resident_bytes_before_read=(
                hash_prefetch_metrics.wasted_page_cache_resident_bytes_before_read
            ),
            hash_prefetch_wasted_page_cache_nonresident_bytes_before_read=(
                hash_prefetch_metrics.wasted_page_cache_nonresident_bytes_before_read
            ),
            hash_prefetch_wasted_page_cache_unclassified_bytes=(
                hash_prefetch_metrics.wasted_page_cache_unclassified_bytes
            ),
            hash_prefetch_read_seconds=hash_prefetch_metrics.read_seconds,
            hash_prefetch_wait_seconds=hash_prefetch_metrics.wait_seconds,
            hash_prefetch_plan_seconds=(
                hash_plan.resolve_seconds if hash_prefetch else 0.0
            ),
            hash_prefetch_layer_ids=tuple(
                layer.layer for layer in hash_plan.layers if hash_prefetch
            ),
            hash_prefetch_layer_union_counts=tuple(
                len(layer.expert_union)
                for layer in hash_plan.layers
                if hash_prefetch
            ),
            adaptive_block_original_tokens=adaptive_decision.original_tokens,
            adaptive_block_selected_tokens=adaptive_decision.selected_tokens,
            adaptive_block_selected_expected_committed=(
                adaptive_selected.expected_committed_tokens
                if adaptive_selected is not None
                else 0.0
            ),
            adaptive_block_selected_requested_hash_experts=(
                adaptive_selected.requested_hash_experts
                if adaptive_selected is not None
                else 0
            ),
            adaptive_block_selected_resident_hash_experts=(
                adaptive_selected.resident_hash_experts
                if adaptive_selected is not None
                else 0
            ),
            adaptive_block_selected_missing_hash_experts=(
                adaptive_selected.missing_hash_experts
                if adaptive_selected is not None
                else 0
            ),
            adaptive_block_selected_predicted_hash_bytes=(
                adaptive_selected.predicted_hash_bytes
                if adaptive_selected is not None
                else 0
            ),
            adaptive_block_selected_score=(
                adaptive_selected.score
                if adaptive_selected is not None
                else 0.0
            ),
            adaptive_block_selection_reason=adaptive_decision.selection_reason,
            adaptive_block_full_commit_fraction=(
                adaptive_decision.full_commit_fraction
            ),
            adaptive_block_plan_seconds=adaptive_decision.plan_seconds,
            adaptive_block_candidate_tokens=tuple(
                candidate.draft_tokens
                for candidate in adaptive_decision.candidates
            ),
            adaptive_block_candidate_expected_committed=tuple(
                candidate.expected_committed_tokens
                for candidate in adaptive_decision.candidates
            ),
            adaptive_block_candidate_requested_hash_experts=tuple(
                candidate.requested_hash_experts
                for candidate in adaptive_decision.candidates
            ),
            adaptive_block_candidate_resident_hash_experts=tuple(
                candidate.resident_hash_experts
                for candidate in adaptive_decision.candidates
            ),
            adaptive_block_candidate_missing_hash_experts=tuple(
                candidate.missing_hash_experts
                for candidate in adaptive_decision.candidates
            ),
            adaptive_block_candidate_predicted_hash_bytes=tuple(
                candidate.predicted_hash_bytes
                for candidate in adaptive_decision.candidates
            ),
            adaptive_block_candidate_scores=tuple(
                candidate.score for candidate in adaptive_decision.candidates
            ),
        )
        verification_seconds = time.perf_counter() - verify_started

        fallback_would_trigger = _should_fallback(
            target_step_seconds,
            draft,
            accepted,
            verification_seconds,
        )
        fallback = fallback_enabled and fallback_would_trigger
        fallback_speculative_seconds = draft.seconds + verification_seconds
        fallback_break_even_seconds = target_step_seconds * (accepted + 1)
        verification_metrics = replace(
            verification_metrics,
            fallback_target_step_seconds=target_step_seconds,
            fallback_speculative_seconds=fallback_speculative_seconds,
            fallback_break_even_seconds=fallback_break_even_seconds,
            fallback_cost_ratio=(
                fallback_speculative_seconds / fallback_break_even_seconds
                if fallback_break_even_seconds
                else 0.0
            ),
            fallback_would_trigger=fallback_would_trigger,
            fallback_triggered=fallback,
        )
        if record_round is not None:
            record_round(
                draft,
                accepted,
                verification_seconds,
                verification_metrics,
            )
        if not draft.tokens:
            target_step_seconds = min(
                target_step_seconds,
                verification_seconds,
            )
        for index in range(accepted):
            if generated >= max_tokens:
                return
            yield draft.tokens[index], target_logprobs[index], True
            generated += 1
        if generated >= max_tokens:
            return
        yield next_token, next_logprobs, False
        generated += 1

        main_hidden = verified_hidden[:, : accepted + 1]
        anchor = next_token
        if fallback:
            if record_fallback is not None:
                record_fallback()
            while generated < max_tokens:
                logits, _ = forward_with_hidden(
                    main_model,
                    mx.array([[anchor]], dtype=mx.int32),
                    prompt_cache,
                    dspark.target_layers,
                )
                logprobs = sampling_logprobs(
                    logits[:, -1], temperature, top_p
                )[0]
                anchor = _sample(logprobs, temperature)
                mx.eval(logprobs)
                yield anchor, logprobs, False
                generated += 1
                if generated % 256 == 0:
                    mx.clear_cache()
            return
        if generated % 256 == 0:
            mx.clear_cache()


def _target_sequence(main_model, tokens, cache, target_layers):
    from .model import verification_forward_with_hidden

    return verification_forward_with_hidden(
        main_model,
        mx.array([tokens], dtype=mx.int32),
        cache,
        target_layers,
    )


def _sequential_target_sequence(main_model, tokens, cache, target_layers):
    from .model import sequential_verification_forward_with_hidden

    return sequential_verification_forward_with_hidden(
        main_model,
        mx.array([tokens], dtype=mx.int32),
        cache,
        target_layers,
    )


def _hybrid_target_sequence(main_model, tokens, cache, target_layers):
    from .model import hybrid_verification_forward_with_hidden

    return hybrid_verification_forward_with_hidden(
        main_model,
        mx.array([tokens], dtype=mx.int32),
        cache,
        target_layers,
    )


def _confidence_prefix_length(confidence: list[float], threshold: float) -> int:
    keep = 0
    while keep < len(confidence) and confidence[keep] >= threshold:
        keep += 1
    return keep


def _should_fallback(
    target_step_seconds: float,
    draft: DraftResult,
    accepted: int,
    verification_seconds: float,
) -> bool:
    speculative_seconds = draft.seconds + verification_seconds
    return speculative_seconds > target_step_seconds * (accepted + 1)


def _verify(
    draft: DraftResult,
    target_logprobs: mx.array,
    temperature: float,
) -> tuple[int, int, mx.array]:
    if temperature == 0:
        greedy_tokens = mx.argmax(
            target_logprobs[: len(draft.tokens) + 1],
            axis=-1,
        ).tolist()
        for index, (token, target_token) in enumerate(
            zip(draft.tokens, greedy_tokens)
        ):
            if token != target_token:
                return index, int(target_token), target_logprobs[index]
        bonus_index = len(draft.tokens)
        return bonus_index, int(greedy_tokens[bonus_index]), target_logprobs[bonus_index]

    accepted = 0
    for index, (token, draft_logprobs) in enumerate(
        zip(draft.tokens, draft.logprobs)
    ):
        target = target_logprobs[index]
        probability = mx.minimum(
            1.0,
            mx.exp(target[token] - draft_logprobs[token]),
        )
        if float(mx.random.uniform().item()) > float(probability.item()):
            difference = mx.maximum(
                mx.exp(target) - mx.exp(draft_logprobs),
                0,
            )
            total = difference.sum()
            correction = mx.where(
                total > 0,
                mx.log(difference / total),
                target,
            )
            return accepted, _sample(correction, temperature), correction
        accepted += 1
    bonus = target_logprobs[len(draft.tokens)]
    return accepted, _sample(bonus, temperature), bonus


def _sample(logprobs: mx.array, temperature: float) -> int:
    if temperature == 0:
        return int(mx.argmax(logprobs).item())
    return int(mx.random.categorical(logprobs).item())
