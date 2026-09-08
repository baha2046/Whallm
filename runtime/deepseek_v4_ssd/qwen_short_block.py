"""Exact Qwen short-block verifier; callers own acceptance and private state."""

import mlx.core as mx

from .cancellation import check_cancelled
from .qwen4_exp import create_ssm_mask


from .qwen_resident_block import resident_experts as union_experts


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
