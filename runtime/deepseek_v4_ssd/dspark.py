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


# The first five-workload calibration over-trimmed drafts that the target later
# accepted completely. Keep a full block when confidence predicts at least 90%
# utilization; lower-confidence blocks still use the storage score below.
_ADAPTIVE_FULL_BLOCK_COMMIT_FRACTION_FLOOR = 0.90


def _truncate_draft(draft: DraftResult, length: int) -> DraftResult:
    if not 0 <= length <= len(draft.tokens):
        raise ValueError("draft prefix length is out of range")
    return DraftResult(
        tokens=draft.tokens[:length],
        logprobs=draft.logprobs[:length],
        confidence=draft.confidence[:length],
        seconds=draft.seconds,
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
    fallback_enabled: bool = True,
    sequential_verification: bool = False,
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
    from .model import (
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
        verification_inputs = [anchor, *draft.tokens]
        capture = (
            target_expert_cache.capture_expert_unions()
            if target_expert_cache is not None
            else nullcontext(None)
        )
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
                        else "single"
                    ),
                    sequential_verification_positions=(
                        1 if sequential_verification else 0
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
                else:
                    replay_logits, replay_hidden = forward_with_hidden(
                        main_model, replay_inputs, prompt_cache, dspark.target_layers,
                    )
                    mx.eval(replay_logits, replay_hidden)
                    eval_prompt_cache(prompt_cache)
                verification_metrics = replace(
                    verification_metrics,
                    cache_replay_seconds=time.perf_counter() - replay_started,
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
    if hasattr(main_model, "forward_with_hidden"):
        from .model_support import get_support
        branch = get_support("deepseek-v4.1").clone_cache(cache)
        logits, hidden = main_model.forward_with_hidden(mx.array([tokens], mx.int32), branch, target_layers)
        return logits, hidden, branch, VerificationMetrics()
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
