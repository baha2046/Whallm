"""Research-only Qwen verifier: token-shaped dense/state, grouped expert union.

Input tokens must already be proposed. This module is NOT a drafter or an
approximate output path. Caller owns a forked cache and verifies acceptance.
"""
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from deepseek_v4_ssd.cancellation import check_cancelled
from deepseek_v4_ssd.expert_cache import QwenBatchedExperts
from deepseek_v4_ssd.qwen4_exp import create_ssm_mask


def union_experts(value, indices, expert_cache, layer, stats=None):
    """One union acquisition, then copy only selected resident views for QMM.

    Copies count against the budget/timer. No claim of zero-copy execution.
    Nothing can evict the resident sources until the returned result is eval'd.
    """
    if value.shape[0] != 1 or not 1 <= value.shape[1] <= 4:
        raise ValueError('only batch-one blocks of 1..4 tokens are supported')
    mx.eval(value, indices)
    raw = np.asarray(indices)
    unique = np.unique(raw)
    if len(unique) > expert_cache.slots:
        raise ValueError('expert union exceeds slot budget')
    if len(unique) * expert_cache.model.expert_blob_size > 1_000_000_000:
        raise ValueError('packed union alone exceeds the extra-memory limit')
    started = time.perf_counter()
    resident = expert_cache.get_many(layer, unique.tolist())
    weights = [resident.individual_weights[resident.slots[int(e)]] for e in unique]
    fields = ('gate_up', 'gate_up_scales', 'down', 'down_scales')
    packed = QwenBatchedExperts(*(mx.stack([getattr(w, field) for w in weights]) for field in fields))
    selected = mx.array(np.searchsorted(unique, raw).astype(np.uint32))
    source = mx.expand_dims(value, (-2, -3))
    projected = mx.gather_qmm(source, packed.gate_up, packed.gate_up_scales, rhs_indices=selected,
        transpose=True, group_size=32, bits=4, mode='mxfp4', sorted_indices=False)
    gate, up = mx.split(projected, 2, axis=-1)
    result = mx.gather_qmm(nn.silu(gate) * up, packed.down, packed.down_scales, rhs_indices=selected,
        transpose=True, group_size=32, bits=4, mode='mxfp4', sorted_indices=False).squeeze(-2)
    # Fence before a subsequent layer can evict/rewrite the resident sources.
    mx.eval(result)
    if stats is not None:
        stats.append(dict(layer=layer, unique_experts=len(unique), assignments=int(raw.size),
            packed_bytes=sum(x.nbytes for x in vars(packed).values()), acquire_pack_qmm_seconds=time.perf_counter()-started))
    return result


def _dense_and_route(layer, hidden, tokens, cache):
    if layer.ple is not None:
        hidden = hidden + layer.ple(hidden, tokens, cache)
    mixed, residual, injection = layer.attn_hyper_connection(hidden)
    result = (layer.linear_attn(mixed, None, cache) if layer.layer_type == 'linear_attention'
        else layer.self_attn(mixed, cache))
    hidden = residual + (result[..., None, :] * injection[..., None]).reshape(*residual.shape)
    mixed, residual, injection = layer.mlp_hyper_connection(hidden)
    mlp = layer.mlp
    probabilities = mx.softmax(mlp.gate(mixed), axis=-1, precise=True)
    indices = mx.argpartition(probabilities, kth=-mlp.top_k, axis=-1)[..., -mlp.top_k:]
    scores = mx.take_along_axis(probabilities, indices, axis=-1)
    if mlp.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    shared = mx.sigmoid(mlp.shared_expert_gate(mixed)) * mlp.shared_expert(mixed)
    mx.eval(mixed, residual, injection, indices, scores, shared)
    return mixed, residual, injection, indices, scores, shared


def verify_union(model, inputs, cache, stats=None):
    """Consume a proposed short block; preserve every dense operation's T=1.

    Cache is mutated. Use a private fork and discard it after rejection; replay
    only accepted positions through the original target path before committing.
    """
    if inputs.ndim != 2 or inputs.shape[0] != 1 or not 2 <= inputs.shape[1] <= 4:
        raise ValueError('expected [1,2..4] proposed input tokens')
    core = model.model
    if len(cache) != len(core.layers):
        raise ValueError('cache layer count mismatch')
    ids = [inputs[:, t:t+1] for t in range(inputs.shape[1])]
    hidden = [mx.tile(core.embed_tokens(x), (1, 1, core.args.hc_count)) for x in ids]
    # The current gate has no ragged batch or padding mask. Fail explicitly.
    if any(create_ssm_mask(h[..., :core.args.hidden_size], cache[0]) is not None for h in hidden):
        raise ValueError('masked/ragged verification is not supported')
    for i, (layer, layer_cache) in enumerate(zip(core.layers, cache)):
        check_cancelled()
        staged = [_dense_and_route(layer, h, token, layer_cache) for h, token in zip(hidden, ids)]
        values = mx.concatenate([s[0] for s in staged], axis=1)
        indices = mx.concatenate([s[3] for s in staged], axis=1)
        routed = union_experts(values, indices, model._expert_cache, i, stats)
        next_hidden = []
        for t, (_, residual, injection, _, scores, shared) in enumerate(staged):
            part = routed[:, t:t+1]
            result = (part * scores[..., None].astype(part.dtype)).sum(axis=-2) + shared
            next_hidden.append(residual + (result[..., None, :] * injection[..., None]).reshape(*residual.shape))
        mx.eval(*next_hidden)
        hidden = next_hidden
    logits = [model.lm_head(core.hyper_connection_mixer(h)) for h in hidden]
    mx.eval(*logits)
    return mx.concatenate(logits, axis=1)
