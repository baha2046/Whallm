from __future__ import annotations

import json
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import deepseek_v4

from .expert_cache import ExpertCache
from .fp8_cache import CorrectPoolingCache, MXFP8PoolingCache
from .manifest import InstalledModel, Tensor

_ORIGINAL_SPARSE_POOLED_ATTENTION = deepseek_v4._sparse_pooled_attention


@dataclass(frozen=True)
class RuntimeConfig:
    slots: int = 1024
    read_workers: int = 4
    prefill_step_size: int = 128
    fp8_kv_cache: bool = True
    memory_limit_gib: int = 48


class _EmptySwitchGLU(nn.Module):
    def __init__(self, *_: object, activation: nn.Module, **__: object):
        super().__init__()
        self.activation = activation


class _StreamingSwitchGLU(nn.Module):
    def __init__(self, layer: int, cache: ExpertCache, activation: nn.Module):
        super().__init__()
        self.layer = layer
        self.cache = cache
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        selected = np.asarray(indices, dtype=np.int32)
        resident = self.cache.get_many(self.layer, selected.reshape(-1).tolist())
        if x.shape[0] == 1 and x.shape[1] == 1:
            outputs = []
            for expert in selected.reshape(-1):
                weights = resident.individual_weights[resident.slots[int(expert)]]
                up = _mxfp4(x, weights.w3, weights.w3_scales)
                gate = _mxfp4(x, weights.w1, weights.w1_scales)
                hidden = self.activation(up, gate)
                outputs.append(_mxfp4(hidden, weights.w2, weights.w2_scales))
            return mx.stack(outputs, axis=-2)

        flat_selected = selected.reshape(-1)
        order = np.argsort(flat_selected, kind="stable")
        boundaries = np.flatnonzero(np.diff(flat_selected[order])) + 1
        flat_x = x.reshape(-1, x.shape[-1])
        outputs = []
        for positions in np.split(order, boundaries):
            expert = int(flat_selected[positions[0]])
            weights = resident.individual_weights[resident.slots[expert]]
            source = mx.take(
                flat_x,
                mx.array(positions // selected.shape[-1]),
                axis=0,
            )
            up = _mxfp4(source, weights.w3, weights.w3_scales)
            gate = _mxfp4(source, weights.w1, weights.w1_scales)
            hidden = self.activation(up, gate)
            outputs.append(_mxfp4(hidden, weights.w2, weights.w2_scales))
        grouped = mx.concatenate(outputs, axis=0)
        restored = mx.take(grouped, mx.array(np.argsort(order)), axis=0)
        return restored.reshape(*selected.shape, -1)


def _mxfp4(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )


def _streaming_moe(self, x: mx.array, input_ids: mx.array) -> mx.array:
    if self.sharding_group is not None:
        raise ValueError("SSD expert streaming supports one Apple Silicon device")
    indices, scores = self.gate(x, input_ids)
    started = time.perf_counter()
    mx.eval(indices)
    self.switch_mlp.cache.metrics.routing_sync_seconds += time.perf_counter() - started
    shared = self.shared_experts(x)
    mx.async_eval(shared)
    routed = self.switch_mlp(x, indices)
    routed = (routed * scores[..., None].astype(routed.dtype)).sum(-2)
    return routed + shared


def _correct_compressor(self, x: mx.array, pool_cache, offset) -> mx.array:
    batch = x.shape[0]
    kv = self.wkv(x)
    gate = self.wgate(x)
    if pool_cache is None:
        usable = kv.shape[1] // self.compress_ratio * self.compress_ratio
        ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
        pool_base = offset
    else:
        ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(kv, gate, offset)

    if ready_kv.size == 0:
        new_pooled = mx.zeros((batch, 0, self.head_dim), dtype=x.dtype)
    else:
        compress = (
            deepseek_v4._overlap_compress_kv
            if self.overlap
            else deepseek_v4._simple_compress_kv
        )
        windows_kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
        windows_gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))
        previous_kv = previous_gate = None
        if self.overlap and pool_cache is not None:
            previous_kv, previous_gate = pool_cache.previous_window()
        if previous_kv is not None:
            windows_kv = mx.concatenate([previous_kv, windows_kv], axis=1)
            windows_gate = mx.concatenate([previous_gate, windows_gate], axis=1)
            new_pooled = compress(windows_kv, windows_gate, self.ape, self.head_dim)[:, 1:]
        else:
            new_pooled = compress(windows_kv, windows_gate, self.ape, self.head_dim)
        if self.overlap and pool_cache is not None:
            pool_cache.store_previous_window(windows_kv, windows_gate)
        new_pooled = self.norm(new_pooled)
        new_pooled = self.rope(new_pooled[:, None], offset=pool_base).squeeze(1)

    if isinstance(pool_cache, MXFP8PoolingCache):
        pool_cache.update(new_pooled)
        return pool_cache
    return pool_cache.update_and_fetch(new_pooled) if pool_cache is not None else new_pooled


def _stable_topk_indices(scores: mx.array, count: int) -> mx.array:
    selected = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
    threshold = mx.min(mx.take_along_axis(scores, selected, axis=-1), axis=-1, keepdims=True)
    size = scores.shape[-1]
    positions = mx.arange(size, dtype=mx.uint32)
    region = mx.where(
        scores > threshold,
        0,
        mx.where(scores == threshold, 1, 2),
    ).astype(mx.uint32)
    keys = region * size + positions
    return mx.sort(mx.argpartition(keys, kth=count - 1, axis=-1)[..., :count], axis=-1)


def _correct_indexer(
    self,
    x: mx.array,
    q_residual: mx.array,
    position_rope,
    pool_cache,
    offset,
):
    batch, length, _ = x.shape
    pooled = self.compressor(x, pool_cache, offset)
    pooled_length = pooled.offset if isinstance(pooled, MXFP8PoolingCache) else pooled.shape[1]
    if pooled_length == 0:
        return None
    query = self.wq_b(q_residual).reshape(batch, length, self.n_heads, self.head_dim)
    query = position_rope(query.transpose(0, 2, 1, 3), offset)
    query = query.astype(mx.float32)
    scores = (
        pooled.quantized_matmul(query)
        if isinstance(pooled, MXFP8PoolingCache)
        else query @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
    )
    scores = mx.maximum(scores, 0) * self.scale
    weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
    scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
    mask = pool_cache.make_mask(length, offset) if pool_cache is not None else None
    if mask is not None:
        scores = mx.where(
            mask if mask.ndim == 3 else mask[None],
            scores,
            mx.finfo(scores.dtype).min,
        )
    count = min(self.index_topk, pooled_length)
    return _stable_topk_indices(scores, count)


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled,
    topk: mx.array,
    local_mask: mx.array | None,
    pooled_mask: mx.array | None,
    scale: float,
    sinks: mx.array | None,
) -> mx.array:
    if not isinstance(pooled, MXFP8PoolingCache):
        return _ORIGINAL_SPARSE_POOLED_ATTENTION(
            q,
            local_kv,
            pooled,
            topk,
            local_mask,
            pooled_mask,
            scale,
            sinks,
        )

    pooled_selected = pooled.gather(topk)
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = deepseek_v4._apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    query = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = query @ pooled_selected.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = deepseek_v4._apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer,
        mx.logsumexp(pooled_scores, -1, keepdims=True),
    )

    local_weights, pooled_weights = deepseek_v4._split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )
    output = local_weights @ local_kv
    pooled_weights = pooled_weights.transpose(0, 2, 1, 3)
    output += (pooled_weights @ pooled_selected).transpose(0, 2, 1, 3)
    return output.astype(q.dtype)


def load_model(
    installed_model: InstalledModel,
    config: RuntimeConfig = RuntimeConfig(),
):
    mx.set_memory_limit(config.memory_limit_gib * 1024**3)
    mx.set_wired_limit(config.memory_limit_gib * 1024**3)
    mx.set_cache_limit(1024**3)

    with (installed_model.root / "config.json").open("rb") as file:
        raw_config = json.load(file)
    args = deepseek_v4.ModelArgs.from_dict(raw_config)

    deepseek_v4.SwitchGLU = _EmptySwitchGLU
    deepseek_v4.PoolingCache = (
        MXFP8PoolingCache if config.fp8_kv_cache else CorrectPoolingCache
    )
    deepseek_v4.Compressor.__call__ = _correct_compressor
    deepseek_v4.Indexer.__call__ = _correct_indexer
    deepseek_v4._sparse_pooled_attention = _sparse_pooled_attention
    model = deepseek_v4.Model(args)
    cache = ExpertCache(installed_model, config.slots, config.read_workers)
    try:
        for layer_index, layer in enumerate(model.layers):
            layer.ffn.switch_mlp = _StreamingSwitchGLU(
                layer_index,
                cache,
                deepseek_v4.LimitedSwiGLU(args.swiglu_limit),
            )
        deepseek_v4.DeepseekV4MoE.__call__ = _streaming_moe

        weights = model.sanitize(_load_common_weights(installed_model))
        quantization = deepseek_v4.make_quantization_config(model)

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
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        return model, cache
    except Exception:
        cache.close()
        raise


def _load_common_weights(installed_model: InstalledModel) -> dict[str, mx.array]:
    path = installed_model.root / "common.bin"
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    weights: dict[str, mx.array] = {}
    for tensor in installed_model.common_tensors:
        weights[tensor.name] = _tensor_from_buffer(mapped, tensor)
    return weights


def _tensor_from_buffer(buffer: np.memmap, tensor: Tensor) -> mx.array:
    numpy_dtype, mlx_view = {
        "BOOL": (np.bool_, None),
        "F16": (np.float16, None),
        "BF16": (np.uint16, mx.bfloat16),
        "F32": (np.float32, None),
        "F64": (np.float64, None),
        "I8": (np.int8, None),
        "I16": (np.int16, None),
        "I32": (np.int32, None),
        "I64": (np.int64, None),
        "U8": (np.uint8, None),
        "U16": (np.uint16, None),
        "U32": (np.uint32, None),
        "U64": (np.uint64, None),
        "F8_E4M3": (np.uint8, None),
        "F8_E8M0": (np.uint8, None),
    }.get(tensor.dtype, (None, None))
    if numpy_dtype is None:
        raise ValueError(f"unsupported tensor dtype {tensor.dtype} for {tensor.name}")
    item_size = np.dtype(numpy_dtype).itemsize
    if tensor.length != int(np.prod(tensor.shape)) * item_size:
        raise ValueError(f"tensor byte count does not match its shape: {tensor.name}")
    view = np.ndarray(
        tensor.shape,
        dtype=numpy_dtype,
        buffer=buffer,
        offset=tensor.offset,
    )
    result = mx.array(view)
    return result.view(mlx_view) if mlx_view is not None else result
