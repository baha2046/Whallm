"""Research-only Qwen slot arena for copy-free resident grouped QMM.

Keep canonical blob regions and existing reads/eviction. One allocation permits
strided batch views indexed by physical slot. Lifetime/alias tests and exact
full-model gates are mandatory before adoption. No production import exists.
"""
from dataclasses import replace

import mlx.core as mx
import numpy as np
from deepseek_v4_ssd import expert_cache as cache_module
from deepseek_v4_ssd import qwen4_exp as qwen
from qwen_decode_candidates import grouped_resident

OriginalSlotPool = cache_module._SlotPool
OriginalExperts = qwen.StreamingExperts.__call__


class ArenaSlotPool(OriginalSlotPool):
    def __init__(self, model, slots):
        super().__init__(model, slots)
        if not model.is_qwen:
            raise ValueError("research arena supports Qwen only")
        self.arena = None
        self.grouped = None

    def prepare(self, slots):
        if self.arena is None:
            self.arena = mx.empty((len(self._slots), self._model.expert_blob_size // 4), dtype=mx.uint32)
            mx.eval(self.arena)
            # Reuse the production region/view contract, changing only batch count.
            proxy = OriginalSlotPool(replace(self._model, expert_count=len(self._slots)), 0)
            self.grouped = proxy.batched(self.arena)
            mx.eval(*vars(self.grouped).values())
        created = []
        for slot in slots:
            if self._slots[slot] is None:
                self._slots[slot] = self.arena[slot].view(mx.uint8)
                created.append(self._slots[slot])
        if created:
            mx.eval(*created)
        for slot in slots:
            if self._views[slot] is None:
                self._views[slot] = memoryview(self._slots[slot])


def install():
    state = {"single_model_calls": 0, "active": False, "grouped_calls": 0, "original_calls": 0}
    original_model = qwen.Model.__call__

    def model(self, input_ids, cache=None):
        if input_ids.size == 1:
            state["single_model_calls"] += 1
        state["active"] = input_ids.size == 1 and state["single_model_calls"] > 1
        try:
            return original_model(self, input_ids, cache)
        finally:
            state["active"] = False

    def experts(self, value, indices):
        pool = self.cache._pool
        if not state["active"] or value.size // value.shape[-1] != 1 or self.cache.current_batched(self.layer) is not None:
            state["original_calls"] += 1
            return OriginalExperts(self, value, indices)
        if not isinstance(pool, ArenaSlotPool):
            raise RuntimeError("candidate requires its own arena")
        selected = np.asarray(indices, dtype=np.int32)
        flat = selected.reshape(-1).tolist()
        # Materializing router indices waits for preceding dependent GPU work.
        # Keep get_many's reserve/read/eviction ownership unchanged.
        self.cache.get_many(self.layer, flat)
        with self.cache._lock:
            physical = [self.cache._entries[(self.layer, int(e))].slot for e in flat]
            if any(not pool._loaded[s] for s in physical):
                raise RuntimeError("unloaded physical slot")
        mapped = mx.array(physical, dtype=mx.uint32).reshape(indices.shape)
        state["grouped_calls"] += 1
        return grouped_resident(value, mapped, pool.grouped)

    cache_module._SlotPool = ArenaSlotPool
    qwen.Model.__call__ = model
    qwen.StreamingExperts.__call__ = experts
    return state
