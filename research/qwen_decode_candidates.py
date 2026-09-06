"""Research-only Decode candidates; preserve weights and routed expert set.

Mechanism: small-batch dispatch/data-movement costs, DeepSpeed Inference §III,
https://arxiv.org/pdf/2207.00032. Neither candidate is a production default.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from deepseek_v4_ssd import qwen4_exp as qwen


def single_row(self, value, indices):
    """Avoid redundant one-row gathers; preserve QMM and sorted expert order."""
    if value.size // value.shape[-1] != 1 or self.cache.current_batched(self.layer) is not None:
        return qwen.StreamingExperts.__call__(self, value, indices)
    selected = np.asarray(indices, dtype=np.int32)
    flat = selected.reshape(-1)
    if len(set(flat.tolist())) != flat.size:
        return qwen.StreamingExperts.__call__(self, value, indices)
    resident = self.cache.get_many(self.layer, flat.tolist())
    order = np.argsort(flat, kind="stable")
    source = value.reshape(1, -1)
    outputs = [self._one(source, resident.individual_weights[resident.slots[int(flat[i])]]) for i in order]
    grouped = mx.concatenate(outputs, axis=0)
    restored = mx.take(grouped, mx.array(np.argsort(order)), axis=0)
    return restored.reshape(*selected.shape, -1)


def grouped_resident(value, indices, weights):
    """Preassembled resident oracle; packing is explicitly outside replay timing.

    A runtime implementation would need a contiguous slot arena without per-step
    copies and validated alias/lifetime handling. This oracle is not that runtime.
    """
    source = mx.expand_dims(value, (-2, -3))
    projected = mx.gather_qmm(source, weights.gate_up, weights.gate_up_scales,
                              rhs_indices=indices, transpose=True, group_size=32, bits=4, mode="mxfp4")
    gate, up = mx.split(projected, 2, axis=-1)
    output = mx.gather_qmm(nn.silu(gate) * up, weights.down, weights.down_scales,
                           rhs_indices=indices, transpose=True, group_size=32, bits=4, mode="mxfp4")
    return output.squeeze(-2)
