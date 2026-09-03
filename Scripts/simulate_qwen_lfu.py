#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    frequency: int
    last_access: int
    version: int = 0


class Cache:
    def __init__(self, slots: int, layer_count: int, policy: str) -> None:
        self.slots = slots
        self.policy = policy
        self.layer_reserve = (slots // layer_count) // 2
        self.layer_counts = [0] * layer_count
        self.entries: dict[tuple[int, int], Entry] = {}
        self.heap: list[tuple[int, int, int, int, int]] = []
        self.clock = 0
        self.last_decay = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.decays = 0
        self._future_uses: dict[tuple[int, int], deque[int]] = {}
        self._prefill_guided_layer_caps: list[int] | None = None
        self._observed_frequency: dict[tuple[int, int], int] = {}
        self._decode_started = False

    def configure_prefill_guided(
        self,
        histograms: list[list[int]],
        minimum_per_layer: int,
    ) -> None:
        if len(histograms) != len(self.layer_counts):
            raise ValueError("prefill histogram layer count does not match the cache")
        if self.slots < minimum_per_layer * len(self.layer_counts):
            raise ValueError("slots cannot hold the minimum per-layer allocation")
        if self.slots > sum(len(row) for row in histograms):
            raise ValueError("slots exceed the experts described by the histograms")
        ranked = [sorted(map(int, row), reverse=True) for row in histograms]
        if any(len(row) <= minimum_per_layer for row in ranked):
            raise ValueError("prefill histogram cannot satisfy the layer allocation")
        caps = [minimum_per_layer] * len(self.layer_counts)
        next_scores = [
            (-row[minimum_per_layer], layer)
            for layer, row in enumerate(ranked)
        ]
        heapq.heapify(next_scores)
        for _ in range(self.slots - minimum_per_layer * len(caps)):
            _, layer = heapq.heappop(next_scores)
            caps[layer] += 1
            if caps[layer] < len(ranked[layer]):
                heapq.heappush(next_scores, (-ranked[layer][caps[layer]], layer))
        self._prefill_guided_layer_caps = caps
        self._observed_frequency = {
            (layer, expert): int(count)
            for layer, row in enumerate(histograms)
            for expert, count in enumerate(row)
            if count
        }

    def begin_decode(self) -> None:
        self._decode_started = True

    def set_future_events(self, events: list[tuple[int, list[int]]]) -> None:
        future_uses: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        for position, (layer, expert_ids) in enumerate(events):
            for expert in set(expert_ids):
                future_uses[(layer, expert)].append(position)
        self._future_uses = dict(future_uses)

    def _begin_event(self, position: int, layer: int, expert_ids: list[int]) -> None:
        if self.policy != "belady_oracle":
            return
        for expert in set(expert_ids):
            uses = self._future_uses[(layer, expert)]
            if not uses or uses[0] != position:
                raise RuntimeError("Belady replay position does not match the trace")
            uses.popleft()

    def _touch(self, key: tuple[int, int], entry: Entry, count: int) -> None:
        self.clock += 1
        entry.frequency += count
        entry.last_access = self.clock
        entry.version += 1
        heapq.heappush(
            self.heap,
            (entry.frequency, entry.last_access, entry.version, *key),
        )

    def _decay(self) -> None:
        for entry in self.entries.values():
            entry.frequency = max(1, entry.frequency // 2)
            entry.version += 1
        self.heap = [
            (entry.frequency, entry.last_access, entry.version, layer, expert)
            for (layer, expert), entry in self.entries.items()
        ]
        heapq.heapify(self.heap)
        self.decays += 1

    def _evict(self, protected: set[tuple[int, int]]) -> None:
        if self.policy == "lru":
            self._evict_lru(protected)
            return
        if self.policy == "prefill_guided_24":
            self._evict_prefill_guided(protected)
            return
        if self.policy == "belady_oracle":
            self._evict_belady(protected)
            return
        protected_items = []
        reserved_items = []
        key = None
        try:
            while self.heap:
                item = heapq.heappop(self.heap)
                frequency, last_access, version, layer, expert = item
                candidate = (layer, expert)
                entry = self.entries.get(candidate)
                if entry is None or (
                    entry.frequency,
                    entry.last_access,
                    entry.version,
                ) != (frequency, last_access, version):
                    continue
                if candidate in protected:
                    protected_items.append(item)
                    continue
                if self.layer_counts[layer] <= self.layer_reserve:
                    reserved_items.append(item)
                    continue
                key = candidate
                break
            if key is None:
                if not reserved_items:
                    raise RuntimeError("no simulated expert can be evicted")
                item = reserved_items.pop(0)
                key = (item[3], item[4])
        finally:
            for item in (*protected_items, *reserved_items):
                heapq.heappush(self.heap, item)
        del self.entries[key]
        self.layer_counts[key[0]] -= 1
        self.evictions += 1

    def _evict_lru(self, protected: set[tuple[int, int]]) -> None:
        # ponytail: O(cache slots) is sufficient for an offline simulator.
        candidates = [
            (entry.last_access, key)
            for key, entry in self.entries.items()
            if key not in protected
            and self.layer_counts[key[0]] > self.layer_reserve
        ]
        if not candidates:
            candidates = [
                (entry.last_access, key)
                for key, entry in self.entries.items()
                if key not in protected
            ]
        if not candidates:
            raise RuntimeError("no simulated expert can be evicted")
        _, key = min(candidates)
        del self.entries[key]
        self.layer_counts[key[0]] -= 1
        self.evictions += 1

    def _evict_prefill_guided(
        self,
        protected: set[tuple[int, int]],
    ) -> None:
        caps = self._prefill_guided_layer_caps
        if caps is None:
            raise RuntimeError("prefill-guided cache is not configured")
        current_layer = next(iter(protected))[0]
        candidates = []
        if self.layer_counts[current_layer] >= caps[current_layer]:
            candidates = [
                (self._observed_frequency.get(key, 0), entry.last_access, key)
                for key, entry in self.entries.items()
                if key not in protected and key[0] == current_layer
            ]
        if not candidates:
            candidates = [
                (self._observed_frequency.get(key, 0), entry.last_access, key)
                for key, entry in self.entries.items()
                if key not in protected and self.layer_counts[key[0]] > caps[key[0]]
            ]
        if not candidates:
            candidates = [
                (self._observed_frequency.get(key, 0), entry.last_access, key)
                for key, entry in self.entries.items()
                if key not in protected
            ]
        if not candidates:
            raise RuntimeError("no simulated expert can be evicted")
        _, _, key = min(candidates)
        del self.entries[key]
        self.layer_counts[key[0]] -= 1
        self.evictions += 1

    def _evict_belady(self, protected: set[tuple[int, int]]) -> None:
        # This oracle sees the full future and intentionally ignores layer reserve.
        candidates = []
        for key, entry in self.entries.items():
            if key in protected:
                continue
            uses = self._future_uses.get(key)
            next_use = uses[0] if uses else float("inf")
            candidates.append((next_use, -entry.last_access, key))
        if not candidates:
            raise RuntimeError("no simulated expert can be evicted")
        _, _, key = max(candidates)
        del self.entries[key]
        self.layer_counts[key[0]] -= 1
        self.evictions += 1

    def access(
        self,
        layer: int,
        expert_ids: list[int],
        event_position: int | None = None,
    ) -> None:
        if event_position is not None:
            self._begin_event(event_position, layer, expert_ids)
        frequencies = Counter(expert_ids)
        if self.policy == "prefill_guided_24" and self._decode_started:
            for expert, count in frequencies.items():
                key = (layer, expert)
                self._observed_frequency[key] = (
                    self._observed_frequency.get(key, 0) + count
                )
        unique = sorted(frequencies)
        protected = {(layer, expert) for expert in unique}
        missing = []
        for expert in unique:
            entry = self.entries.get((layer, expert))
            if entry is None:
                self.misses += 1
                missing.append(expert)
            else:
                self.hits += 1
                self._touch((layer, expert), entry, frequencies[expert])
        for expert in missing:
            if len(self.entries) == self.slots:
                self._evict(protected)
            entry = Entry(frequencies[expert], 0)
            self.entries[(layer, expert)] = entry
            self.layer_counts[layer] += 1
            self._touch((layer, expert), entry, 0)
        if self.policy == "current" and (
            self.clock - self.last_decay >= max(self.slots * 8, 64)
        ):
            self.last_decay = self.clock
            self._decay()

    def finish_token(self, token: int) -> None:
        if self.policy == "prefill_guided_24" and token % 24 == 0:
            self._observed_frequency = {
                key: max(1, frequency // 2)
                for key, frequency in self._observed_frequency.items()
            }
            self.decays += 1
            return
        if not self.policy.startswith("token_"):
            return
        interval = int(self.policy.removeprefix("token_"))
        if token % interval == 0:
            self._decay()


def simulate(trace: dict, slots: int, policy: str) -> dict:
    if slots < 1:
        raise ValueError("slot count must be greater than zero")
    layer_count = int(trace["layer_count"])
    if layer_count < 1:
        raise ValueError("layer count must be greater than zero")
    routes = trace["decode_routes"]
    if len(routes) != layer_count:
        raise ValueError("decode route layer count does not match the trace")
    token_counts = {len(layer) for layer in routes}
    if len(token_counts) != 1:
        raise ValueError("decode route layers have different token counts")
    token_count = token_counts.pop()
    cache = Cache(slots, layer_count, policy)
    events = [
        (layer, routes[layer][token_index])
        for token_index in range(token_count)
        for layer in range(layer_count)
    ]
    if policy == "belady_oracle":
        cache.set_future_events(events)
    selected_count = int(trace.get("selected_expert_count", 1))
    if slots < selected_count:
        raise ValueError("slot count must hold one routed expert event")
    if any(len(expert_ids) != selected_count for _, expert_ids in events):
        raise ValueError("decode route has an invalid selection count")
    if policy == "prefill_guided_24":
        cache.configure_prefill_guided(
            trace["prefill_histograms"],
            selected_count,
        )
    prefill_cache_accesses = trace.get("prefill_cache_accesses")
    prefill_cache_calls = 0
    if prefill_cache_accesses is not None:
        for access in prefill_cache_accesses:
            layer = int(access["layer"])
            histogram = access["histogram"]
            if not 0 <= layer < layer_count:
                raise ValueError("prefill cache access has an invalid layer")
            if len(histogram) != int(trace.get("expert_count", len(histogram))):
                raise ValueError("prefill cache histogram has an invalid size")
            if any(int(count) < 0 for count in histogram):
                raise ValueError("prefill cache histogram has a negative count")
            selected = [
                expert
                for expert, count in enumerate(histogram)
                for _ in range(int(count))
            ]
            if not selected:
                raise ValueError("prefill cache access is empty")
            cache.access(layer, selected)
            prefill_cache_calls += 1
    else:
        prefill_chunks = trace.get("prefill_chunk_histograms", [])
        handoff_chunks = []
        if prefill_chunks:
            handoff_chunks = [
                index
                for index, histogram in enumerate(prefill_chunks[0])
                if sum(histogram) == selected_count
            ]
            for layer_chunks in prefill_chunks:
                if [
                    index
                    for index, histogram in enumerate(layer_chunks)
                    if sum(histogram) == selected_count
                ] != handoff_chunks:
                    raise ValueError("prefill handoff chunks differ by layer")
            for chunk_index in handoff_chunks:
                for layer, layer_chunks in enumerate(prefill_chunks):
                    histogram = layer_chunks[chunk_index]
                    selected = [
                        expert
                        for expert, count in enumerate(histogram)
                        for _ in range(count)
                    ]
                    cache.access(layer, selected)
                    prefill_cache_calls += 1
    cache.begin_decode()
    initial_hits = cache.hits
    initial_misses = cache.misses
    initial_evictions = cache.evictions
    initial_decays = cache.decays
    for event_position, (layer, expert_ids) in enumerate(events):
        cache.access(layer, expert_ids, event_position)
        token_index = event_position // layer_count
        if (event_position + 1) % layer_count:
            continue
        cache.finish_token(token_index + 1)
    return {
        "policy": policy,
        "prefill_cache_calls": prefill_cache_calls,
        "decode_tokens": token_count,
        "hits": cache.hits - initial_hits,
        "misses": cache.misses - initial_misses,
        "evictions": cache.evictions - initial_evictions,
        "decays": cache.decays - initial_decays,
        "resident_experts": len(cache.entries),
        "uses_future_routes": policy == "belady_oracle",
    }


def summarize(
    trace: dict,
    trace_sha256: str,
    slots: int,
    target_tokens_per_second: float | None = None,
) -> dict:
    if target_tokens_per_second is not None and target_tokens_per_second <= 0:
        raise ValueError("target tokens per second must be greater than zero")
    policies = ["current", "token_128", "token_512", "token_1024", "never"]
    results = [simulate(trace, slots, policy) for policy in policies]
    current = results[0]
    replacement_results = [
        current,
        simulate(trace, slots, "lru"),
        simulate(trace, slots, "belady_oracle"),
    ]
    causal_results = (
        [simulate(trace, slots, "prefill_guided_24")]
        if "prefill_histograms" in trace
        else []
    )
    blob_size = int(trace["expert_blob_size"])
    recorded_misses = sum(
        value
        for layer in trace.get("decode_misses", [])
        for token in layer
        for value in token
    )
    for result in [*results, *replacement_results[1:], *causal_results]:
        result["expert_bytes"] = result["misses"] * blob_size
        result["expert_bytes_per_token"] = (
            result["expert_bytes"] / result["decode_tokens"]
            if result["decode_tokens"]
            else 0.0
        )
        demand_accesses = result["hits"] + result["misses"]
        result["demand_hit_rate"] = (
            result["hits"] / demand_accesses if demand_accesses else 0.0
        )
        if target_tokens_per_second is not None:
            result["required_expert_bandwidth_gbps"] = (
                result["expert_bytes_per_token"]
                * target_tokens_per_second
                / 1_000_000_000
            )
        result["miss_change_fraction"] = (
            result["misses"] / current["misses"] - 1
            if current["misses"]
            else 0.0
        )
    oracle = replacement_results[-1]
    return {
        "schema_version": 3,
        "evidence_kind": "expert_cache_trace_simulation",
        "trace_sha256": trace_sha256,
        "slots": slots,
        "layer_reserve": (slots // int(trace["layer_count"])) // 2,
        "recorded_decode_misses": recorded_misses,
        "current_simulated_misses": current["misses"],
        "current_matches_recorded": current["misses"] == recorded_misses,
        "policies": results,
        "replacement_policies": replacement_results,
        "causal_policies": causal_results,
        "belady_oracle": {
            "scope": "prefill_handoff_and_decode",
            "uses_future_routes": True,
            "ignores_layer_reserve": True,
            "miss_reduction_upper_bound_fraction": (
                (current["misses"] - oracle["misses"]) / current["misses"]
                if current["misses"]
                else 0.0
            ),
        },
        "target_tokens_per_second": target_tokens_per_second,
    }


def self_test() -> None:
    trace = {
        "layer_count": 2,
        "expert_blob_size": 16,
        "decode_routes": [[[0], [1], [0]], [[0], [1], [0]]],
        "decode_misses": [[[1], [1], [0]], [[1], [1], [0]]],
    }
    result = summarize(trace, "fixture", 4)
    assert result["current_matches_recorded"]
    assert result["current_simulated_misses"] == 4
    assert result["policies"][0]["hits"] == 2
    replacements = {row["policy"]: row for row in result["replacement_policies"]}
    assert replacements["lru"]["misses"] == 4
    assert replacements["belady_oracle"]["misses"] == 4

    oracle_trace = {
        "layer_count": 1,
        "expert_blob_size": 10,
        "decode_routes": [[[0], [1], [2], [0], [1]]],
        "decode_misses": [[[1], [1], [1], [1], [1]]],
    }
    oracle_result = summarize(oracle_trace, "oracle-fixture", 2, 5.0)
    oracle_replacements = {
        row["policy"]: row for row in oracle_result["replacement_policies"]
    }
    assert oracle_replacements["lru"]["misses"] == 5
    assert oracle_replacements["belady_oracle"]["misses"] == 4
    assert oracle_replacements["belady_oracle"]["expert_bytes_per_token"] == 8
    assert oracle_replacements["belady_oracle"][
        "required_expert_bandwidth_gbps"
    ] == 0.00000004


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--slots", type=int, default=1_152)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-tokens-per-second", type=float)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if arguments.trace is None:
        parser.error("trace is required unless --self-test is used")
    raw = arguments.trace.read_bytes()
    result = summarize(
        json.loads(raw),
        hashlib.sha256(raw).hexdigest(),
        arguments.slots,
        arguments.target_tokens_per_second,
    )
    encoded = json.dumps(result, indent=2) + "\n"
    if arguments.output is None:
        print(encoded, end="")
        return
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, arguments.output)


if __name__ == "__main__":
    main()
