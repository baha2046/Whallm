from __future__ import annotations

import heapq
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .manifest import InstalledModel


@dataclass(frozen=True)
class ExpertWeights:
    w1: mx.array
    w1_scales: mx.array
    w2: mx.array
    w2_scales: mx.array
    w3: mx.array
    w3_scales: mx.array


@dataclass(frozen=True)
class ResidentExperts:
    weights: ExpertWeights
    slots: dict[int, int]


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    pack_seconds: float = 0.0
    eviction_seconds: float = 0.0
    routing_sync_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class _Entry:
    slot: int
    frequency: int
    last_access: int
    version: int = 0


class _SlotPool:
    """Keep one packed Metal array in each fixed expert slot."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._regions = {region.name: region for region in model.expert_regions}
        self._slots: list[mx.array | None] = [None] * slots

    def select(self, slots: list[int]) -> ExpertWeights:
        arrays = [self._slots[slot] for slot in slots]
        if any(array is None for array in arrays):
            raise RuntimeError("expert slot is empty")

        def array(name: str) -> mx.array:
            region = self._regions[name]
            selected = []
            for packed in arrays:
                value = packed[region.offset : region.offset + region.length]
                value = value.reshape(region.shape)
                if region.dtype == "I8":
                    value = value.view(mx.int8)
                if name.endswith(".weight"):
                    value = value.view(mx.uint32)
                selected.append(value)
            return mx.stack(selected)

        return ExpertWeights(
            w1=array("w1.weight"),
            w1_scales=array("w1.scale"),
            w2=array("w2.weight"),
            w2_scales=array("w2.scale"),
            w3=array("w3.weight"),
            w3_scales=array("w3.scale"),
        )

    def store(self, slots: list[int], blobs: list[bytes]) -> None:
        arrays = [mx.array(np.frombuffer(blob, dtype=np.uint8)) for blob in blobs]
        mx.eval(*arrays)
        for slot, array in zip(slots, arrays):
            self._slots[slot] = array


class ExpertCache:
    """A fixed-size, layer-aware LFU slot pool for routed expert weights."""

    def __init__(
        self,
        installed_model: InstalledModel,
        slots: int = 1024,
        read_workers: int = 4,
    ) -> None:
        if slots < installed_model.selected_expert_count:
            raise ValueError("slot count must hold at least one token's routed experts")
        if read_workers < 1:
            raise ValueError("read worker count must be greater than zero")
        self.model = installed_model
        self.slots = slots
        self.metrics = CacheMetrics()
        self._pool = _SlotPool(installed_model, slots)
        self._entries: dict[tuple[int, int], _Entry] = {}
        self._free_slots = list(reversed(range(slots)))
        self._layer_heaps: list[list[tuple[int, int, int, int, int]]] = [
            [] for _ in range(installed_model.layer_count)
        ]
        self._layer_counts = [0] * installed_model.layer_count
        base = slots // installed_model.layer_count
        self._layer_reserve = base // 2
        self._clock = 0
        self._last_decay = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=read_workers)
        self._descriptors: list[int] = []
        try:
            for layer in range(installed_model.layer_count):
                self._descriptors.append(
                    os.open(
                        installed_model.root / f"experts/layer_{layer:02d}.bin",
                        os.O_RDONLY,
                    )
                )
        except Exception:
            for descriptor in self._descriptors:
                os.close(descriptor)
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for descriptor in self._descriptors:
            os.close(descriptor)
        self._descriptors.clear()

    def __enter__(self) -> ExpertCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def resident_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def get_many(self, layer: int, expert_ids: list[int]) -> ResidentExperts:
        if not 0 <= layer < self.model.layer_count:
            raise ValueError(f"invalid layer {layer}")
        frequencies = Counter(expert_ids)
        unique = sorted(frequencies)
        if len(unique) > self.slots:
            raise ValueError(
                f"prefill selected {len(unique)} experts, but the cache has {self.slots} slots"
            )
        for expert in unique:
            if not 0 <= expert < self.model.expert_count:
                raise ValueError(f"invalid expert {expert}")

        missing: list[int] = []
        protected = {(layer, expert) for expert in unique}
        with self._lock:
            for expert in unique:
                entry = self._entries.get((layer, expert))
                if entry is None:
                    self.metrics.misses += 1
                    missing.append(expert)
                else:
                    self.metrics.hits += 1
                    self._touch(layer, expert, entry, frequencies[expert])

        started = time.perf_counter()
        futures = {
            expert: self._executor.submit(self._read_blob, layer, expert) for expert in missing
        }
        blobs = {expert: future.result() for expert, future in futures.items()}
        elapsed = time.perf_counter() - started if missing else 0.0

        with self._lock:
            self.metrics.bytes_read += len(missing) * self.model.expert_blob_size
            self.metrics.read_seconds += elapsed
            assigned_slots: list[int] = []
            assigned_blobs: list[bytes] = []
            for expert, blob in blobs.items():
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    eviction_started = time.perf_counter()
                    victim_key, victim = self._evict(protected)
                    self.metrics.eviction_seconds += time.perf_counter() - eviction_started
                    slot = victim.slot
                    del self._entries[victim_key]
                    self._layer_counts[victim_key[0]] -= 1
                    self.metrics.evictions += 1
                assigned_slots.append(slot)
                assigned_blobs.append(blob)
                entry = _Entry(slot, frequencies[expert], 0)
                self._entries[(layer, expert)] = entry
                self._layer_counts[layer] += 1
                self._touch(layer, expert, entry, 0)
            if assigned_slots:
                self._pool.store(assigned_slots, assigned_blobs)
            self._decay_if_needed()
            physical_slots = [self._entries[(layer, expert)].slot for expert in unique]
            pack_started = time.perf_counter()
            weights = self._pool.select(physical_slots)
            self.metrics.pack_seconds += time.perf_counter() - pack_started
            return ResidentExperts(
                weights,
                {expert: slot for slot, expert in enumerate(unique)},
            )

    def _touch(self, layer: int, expert: int, entry: _Entry, count: int) -> None:
        self._clock += 1
        entry.frequency += count
        entry.last_access = self._clock
        entry.version += 1
        heapq.heappush(
            self._layer_heaps[layer],
            (entry.frequency, entry.last_access, entry.version, layer, expert),
        )

    def _evict(self, protected: set[tuple[int, int]]) -> tuple[tuple[int, int], _Entry]:
        candidates = [
            victim
            for layer in range(self.model.layer_count)
            if (victim := self._pop_layer_victim(layer, protected)) is not None
        ]
        if not candidates:
            raise RuntimeError("no expert cache slot can be evicted")
        preferred = [
            victim
            for victim in candidates
            if self._layer_counts[victim[0][0]] > self._layer_reserve
        ]
        chosen = min(
            preferred or candidates,
            key=lambda victim: (victim[1].frequency, victim[1].last_access),
        )
        for key, entry in candidates:
            if key != chosen[0]:
                heapq.heappush(
                    self._layer_heaps[key[0]],
                    (entry.frequency, entry.last_access, entry.version, *key),
                )
        return chosen

    def _pop_layer_victim(
        self,
        layer: int,
        protected: set[tuple[int, int]],
    ) -> tuple[tuple[int, int], _Entry] | None:
        heap = self._layer_heaps[layer]
        held: list[tuple[int, int, int, int, int]] = []
        try:
            while heap:
                item = heapq.heappop(heap)
                frequency, last_access, version, _, expert = item
                key = (layer, expert)
                entry = self._entries.get(key)
                if entry is None or (
                    entry.frequency,
                    entry.last_access,
                    entry.version,
                ) != (frequency, last_access, version):
                    continue
                if key in protected:
                    held.append(item)
                    continue
                return key, entry
            return None
        finally:
            for item in held:
                heapq.heappush(heap, item)

    def _decay_if_needed(self) -> None:
        if self._clock - self._last_decay < max(self.slots * 8, 64):
            return
        self._last_decay = self._clock
        self._layer_heaps = [[] for _ in range(self.model.layer_count)]
        for (layer, expert), entry in self._entries.items():
            entry.frequency = max(1, entry.frequency // 2)
            entry.version += 1
            heapq.heappush(
                self._layer_heaps[layer],
                (entry.frequency, entry.last_access, entry.version, layer, expert),
            )

    def _read_blob(self, layer: int, expert: int) -> bytes:
        offset = expert * self.model.expert_blob_size
        remaining = self.model.expert_blob_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.pread(self._descriptors[layer], remaining, offset)
            if not chunk:
                raise EOFError(f"expert file ended early at layer {layer}, expert {expert}")
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
