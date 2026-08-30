#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
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

    def access(self, layer: int, expert_ids: list[int]) -> None:
        frequencies = Counter(expert_ids)
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
        if not self.policy.startswith("token_"):
            return
        interval = int(self.policy.removeprefix("token_"))
        if token % interval == 0:
            self._decay()


def simulate(trace: dict, slots: int, policy: str) -> dict:
    layer_count = int(trace["layer_count"])
    routes = trace["decode_routes"]
    if len(routes) != layer_count:
        raise ValueError("decode route layer count does not match the trace")
    token_counts = {len(layer) for layer in routes}
    if len(token_counts) != 1:
        raise ValueError("decode route layers have different token counts")
    token_count = token_counts.pop()
    cache = Cache(slots, layer_count, policy)
    selected_count = int(trace.get("selected_expert_count", 1))
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
    initial_hits = cache.hits
    initial_misses = cache.misses
    initial_evictions = cache.evictions
    initial_decays = cache.decays
    for token_index in range(token_count):
        for layer in range(layer_count):
            cache.access(layer, routes[layer][token_index])
        cache.finish_token(token_index + 1)
    return {
        "policy": policy,
        "prefill_handoff_calls": len(handoff_chunks) * layer_count,
        "decode_tokens": token_count,
        "hits": cache.hits - initial_hits,
        "misses": cache.misses - initial_misses,
        "evictions": cache.evictions - initial_evictions,
        "decays": cache.decays - initial_decays,
        "resident_experts": len(cache.entries),
    }


def summarize(trace: dict, trace_sha256: str, slots: int) -> dict:
    policies = ["current", "token_128", "token_512", "token_1024", "never"]
    results = [simulate(trace, slots, policy) for policy in policies]
    current = results[0]
    blob_size = int(trace["expert_blob_size"])
    recorded_misses = sum(
        value
        for layer in trace.get("decode_misses", [])
        for token in layer
        for value in token
    )
    for result in results:
        result["expert_bytes"] = result["misses"] * blob_size
        result["miss_change_fraction"] = (
            result["misses"] / current["misses"] - 1
            if current["misses"]
            else 0.0
        )
    return {
        "schema_version": 1,
        "evidence_kind": "qwen_lfu_decay_trace_simulation",
        "trace_sha256": trace_sha256,
        "slots": slots,
        "layer_reserve": (slots // int(trace["layer_count"])) // 2,
        "recorded_decode_misses": recorded_misses,
        "current_simulated_misses": current["misses"],
        "current_matches_recorded": current["misses"] == recorded_misses,
        "policies": results,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--slots", type=int, default=1_152)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        return
    if arguments.trace is None:
        parser.error("trace is required unless --self-test is used")
    raw = arguments.trace.read_bytes()
    result = summarize(json.loads(raw), hashlib.sha256(raw).hexdigest(), arguments.slots)
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
