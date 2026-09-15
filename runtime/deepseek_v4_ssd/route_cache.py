"""Bounded routing history and marginal-value budgets, independent of weight residency.

Caller holds ExpertCache's lock. These estimates change retention only, never routing.
"""
from __future__ import annotations

import heapq
import math
import numpy as np


class RouteCachePolicy:
    short_half_life = 16
    long_half_life = 256
    rebalance_tokens = 16

    def __init__(self, layers: int, experts: int, slots: int, top_k: int):
        if min(layers, experts, slots, top_k) < 1 or top_k > experts:
            raise ValueError("invalid route cache dimensions")
        self.layers, self.experts, self.top_k = layers, experts, top_k
        self.capacity = min(slots, layers * experts)
        self.short = np.zeros((layers, experts), dtype=np.float64)
        self.long = np.zeros_like(self.short)
        self.prefill = np.zeros_like(self.short)
        self.scores = np.zeros_like(self.short)
        self.prefill_tokens = np.zeros(layers, dtype=np.int64)
        self.decode_tokens = np.zeros(layers, dtype=np.int64)
        self.request_decode_tokens = np.zeros(layers, dtype=np.int64)
        self.read_cost = np.zeros(layers, dtype=np.float64)
        self.read_samples = np.zeros(layers, dtype=np.int64)
        self.quotas = np.zeros(layers, dtype=np.int64)
        self.rebalances = 0
        self._last_rebalance = 0
        self.rebalance()

    def begin_prefill(self):
        # Current prompt is a weak prior; it never floods either decode history.
        self.prefill.fill(0)
        self.prefill_tokens.fill(0)
        self.request_decode_tokens.fill(0)

    def observe(self, layer: int, selected, phase: str) -> bool:
        ids = np.asarray(selected, dtype=np.int32).reshape(-1)
        if not 0 <= layer < self.layers or np.any(ids < 0) or np.any(ids >= self.experts):
            raise ValueError("invalid routed expert")
        if not ids.size:
            return False
        # A normal route has top_k entries per position. Union-only callers still
        # contribute demand, but cannot reconstruct missing assignment multiplicity.
        tokens = max(1, math.ceil(ids.size / self.top_k))
        if phase == "prefill":
            self.prefill[layer] += np.bincount(ids, minlength=self.experts)
            self.prefill_tokens[layer] += tokens
            return False
        for history, half_life in ((self.short, self.short_half_life), (self.long, self.long_half_life)):
            decay = 2 ** (-1 / half_life)
            age = tokens - 1 - np.arange(ids.size) // self.top_k
            weights = (1 - decay) * np.power(decay, age)
            history[layer] *= decay ** tokens
            history[layer] += np.bincount(ids, weights=weights, minlength=self.experts)
        self.decode_tokens[layer] += tokens
        self.request_decode_tokens[layer] += tokens
        total = int(self.decode_tokens.sum())
        if total - self._last_rebalance >= self.rebalance_tokens * self.layers:
            self._last_rebalance = total
            self.rebalance()
            return True
        return False

    def record_read(self, layer: int, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            return
        if self.read_samples[layer]:
            self.read_cost[layer] += 0.05 * (seconds - self.read_cost[layer])
        else:
            self.read_cost[layer] = seconds
        self.read_samples[layer] += 1

    def rebalance(self):
        prior = self.prefill / np.maximum(self.prefill_tokens[:, None], 1)
        prior *= (0.1 * np.exp2(-self.request_decode_tokens / 32))[:, None]
        rates = 0.75 * self.short + 0.25 * self.long + prior
        observed = self.read_cost[self.read_samples > 0]
        mean = float(np.median(observed)) if observed.size else 1.0
        cost = np.where(self.read_samples > 0, self.read_cost, mean)
        cost = np.clip(cost / max(mean, 1e-9), 0.5, 2.0)
        self.scores[:] = rates * cost[:, None]
        # Each layer's sorted scores form a diminishing marginal-benefit curve.
        # A heap grants each next slot to the largest remaining predicted saving.
        order = np.sort(self.scores, axis=1)[:, ::-1]
        self.quotas.fill(0)
        heap = [(-float(order[layer, 0]), 0, layer) for layer in range(self.layers)]
        heapq.heapify(heap)
        for _ in range(self.capacity):
            _, rank, layer = heapq.heappop(heap)
            self.quotas[layer] += 1
            rank += 1
            if rank < self.experts:
                # Equal benefits distribute capacity across layers before deepening one.
                heapq.heappush(heap, (-float(order[layer, rank]), rank, layer))
        self.rebalances += 1

    def snapshot(self):
        return {
            "short_half_life_tokens": self.short_half_life,
            "long_half_life_tokens": self.long_half_life,
            "rebalance_tokens": self.rebalance_tokens,
            "rebalances": self.rebalances,
            "history_survives_eviction": True,
            "history_scope": "loaded model lifetime; includes observed speculative/replay routes",
            "metadata_array_bytes": sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray)),
            "layers": [{"layer": layer, "target_slots": int(self.quotas[layer]),
                "decode_observed_tokens": int(self.decode_tokens[layer]),
                "prefill_observed_tokens": int(self.prefill_tokens[layer]),
                "history_experts": int(np.count_nonzero(self.short[layer] + self.long[layer])),
                "read_cost_seconds": float(self.read_cost[layer]),
                "read_samples": int(self.read_samples[layer]),
                "predicted_retained_value": float(np.sort(self.scores[layer])[::-1][:self.quotas[layer]].sum())}
                for layer in range(self.layers)],
        }
