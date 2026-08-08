from __future__ import annotations

import heapq
import os
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

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
    individual_weights: tuple[ExpertWeights, ...]
    slots: dict[int, int]


@dataclass(frozen=True)
class BatchedExperts:
    w1: mx.array
    w1_scales: mx.array
    w2: mx.array
    w2_scales: mx.array
    w3: mx.array
    w3_scales: mx.array


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
    batched_layers: int = 0
    gather_qmm_calls: int = 0
    prefetched_layer_hits: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def delta(self, before: CacheMetrics) -> CacheMetrics:
        return CacheMetrics(
            hits=self.hits - before.hits,
            misses=self.misses - before.misses,
            evictions=self.evictions - before.evictions,
            bytes_read=self.bytes_read - before.bytes_read,
            read_seconds=self.read_seconds - before.read_seconds,
            pack_seconds=self.pack_seconds - before.pack_seconds,
            eviction_seconds=self.eviction_seconds - before.eviction_seconds,
            routing_sync_seconds=(
                self.routing_sync_seconds - before.routing_sync_seconds
            ),
            batched_layers=self.batched_layers - before.batched_layers,
            gather_qmm_calls=self.gather_qmm_calls - before.gather_qmm_calls,
            prefetched_layer_hits=(
                self.prefetched_layer_hits - before.prefetched_layer_hits
            ),
        )


@dataclass
class _Entry:
    slot: int
    frequency: int
    last_access: int
    version: int = 0


@dataclass(frozen=True)
class _LayerRead:
    buffer: bytearray
    futures: tuple[Future[float], ...]


class _SlotPool:
    """Keep one packed Metal array in each fixed expert slot."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._regions = {region.name: region for region in model.expert_regions}
        self._slots: list[mx.array | None] = [None] * slots

    def select_individual(self, slots: list[int]) -> tuple[ExpertWeights, ...]:
        arrays = [self._slots[slot] for slot in slots]
        if any(array is None for array in arrays):
            raise RuntimeError("expert slot is empty")
        return tuple(
            ExpertWeights(
                w1=self._array(array, "w1.weight"),
                w1_scales=self._array(array, "w1.scale"),
                w2=self._array(array, "w2.weight"),
                w2_scales=self._array(array, "w2.scale"),
                w3=self._array(array, "w3.weight"),
                w3_scales=self._array(array, "w3.scale"),
            )
            for array in arrays
        )

    def _array(self, packed: mx.array, name: str) -> mx.array:
        region = self._regions[name]
        value = packed[region.offset : region.offset + region.length]
        value = value.reshape(region.shape)
        if region.dtype == "I8":
            value = value.view(mx.int8)
        return value.view(mx.uint32) if name.endswith(".weight") else value

    def store(self, slots: list[int], blobs: list[bytes]) -> None:
        arrays = [mx.array(np.frombuffer(blob, dtype=np.uint8)) for blob in blobs]
        mx.eval(*arrays)
        for slot, array in zip(slots, arrays):
            self._slots[slot] = array

    def batched(self, packed: mx.array) -> BatchedExperts:
        return BatchedExperts(
            w1=self._batched_array(packed, "w1.weight"),
            w1_scales=self._batched_array(packed, "w1.scale"),
            w2=self._batched_array(packed, "w2.weight"),
            w2_scales=self._batched_array(packed, "w2.scale"),
            w3=self._batched_array(packed, "w3.weight"),
            w3_scales=self._batched_array(packed, "w3.scale"),
        )

    def _batched_array(self, packed: mx.array, name: str) -> mx.array:
        region = self._regions[name]
        if (
            self._model.expert_blob_size % 4
            or region.offset % 4
            or region.shape[-1] % 4
        ):
            raise ValueError("batched expert regions must be 4-byte aligned")
        packed_shape = (*region.shape[:-1], region.shape[-1] // 4)
        shape = (self._model.expert_count, *packed_shape)
        row_strides = []
        stride = 1
        for size in reversed(packed_shape):
            row_strides.append(stride)
            stride *= size
        value = mx.as_strided(
            packed,
            shape=shape,
            strides=(self._model.expert_blob_size // 4, *reversed(row_strides)),
            offset=region.offset // 4,
        )
        return value if name.endswith(".weight") else value.view(mx.uint8)


class ExpertCache:
    """A fixed-size, layer-aware LFU slot pool for routed expert weights."""

    def __init__(
        self,
        installed_model: InstalledModel,
        slots: int = 512,
        read_workers: int = 4,
        prefetch_read_workers: int = 2,
        layer_count: int | None = None,
        expert_directory: Path | None = None,
    ) -> None:
        if slots < installed_model.selected_expert_count:
            raise ValueError("slot count must hold at least one token's routed experts")
        if read_workers < 1:
            raise ValueError("read worker count must be greater than zero")
        if prefetch_read_workers < 1:
            raise ValueError("prefetch read worker count must be greater than zero")
        self.model = installed_model
        self.slots = slots
        self.read_workers = read_workers
        self.prefetch_read_workers = min(read_workers, prefetch_read_workers)
        self.layer_count = layer_count or installed_model.layer_count
        self.expert_directory = expert_directory or installed_model.root / "experts"
        if self.layer_count < 1:
            raise ValueError("expert cache layer count must be greater than zero")
        self.metrics = CacheMetrics()
        self._pool = _SlotPool(installed_model, slots)
        self._entries: dict[tuple[int, int], _Entry] = {}
        self._free_slots = list(reversed(range(slots)))
        self._heap: list[tuple[int, int, int, int, int]] = []
        self._layer_counts = [0] * self.layer_count
        base = slots // self.layer_count
        self._layer_reserve = base // 2
        self._clock = 0
        self._last_decay = 0
        self._pinned_layers: set[int] = set()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=read_workers)
        self._prefetched_layers: dict[int, _LayerRead] = {}
        self._batched_layer: tuple[int, BatchedExperts] | None = None
        self._descriptors: list[int] = []
        try:
            for layer in range(self.layer_count):
                self._descriptors.append(
                    os.open(
                        self.expert_directory / f"layer_{layer:02d}.bin",
                        os.O_RDONLY,
                    )
                )
        except Exception:
            for descriptor in self._descriptors:
                os.close(descriptor)
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise

    def close(self) -> None:
        for job in self._prefetched_layers.values():
            for future in job.futures:
                future.cancel()
        self._prefetched_layers.clear()
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

    def metrics_snapshot(self) -> CacheMetrics:
        with self._lock:
            return replace(self.metrics)

    def record_routing_sync(self, seconds: float) -> None:
        with self._lock:
            self.metrics.routing_sync_seconds += seconds

    def record_gather_qmm(self, calls: int = 3) -> None:
        with self._lock:
            self.metrics.gather_qmm_calls += calls

    def current_batched(self, layer: int) -> BatchedExperts | None:
        current = self._batched_layer
        return current[1] if current is not None and current[0] == layer else None

    def prefetch_layer(self, layer: int) -> None:
        if not 0 <= layer < self.layer_count:
            return
        with self._lock:
            if layer not in self._prefetched_layers:
                length = self.model.expert_count * self.model.expert_blob_size
                buffer = bytearray(length)
                ranges = []
                step = (
                    (length + self.prefetch_read_workers - 1)
                    // self.prefetch_read_workers
                    + 4095
                ) // 4096 * 4096
                for start in range(0, length, step):
                    ranges.append((start, min(start + step, length)))
                self._prefetched_layers[layer] = _LayerRead(
                    buffer,
                    tuple(
                        self._executor.submit(
                            self._read_into,
                            layer,
                            buffer,
                            start,
                            end,
                        )
                        for start, end in ranges
                    ),
                )

    @contextmanager
    def batched_layer(self, layer: int, enabled: bool = True):
        if not enabled:
            yield None
            return
        self.prefetch_layer(layer)
        with self._lock:
            job = self._prefetched_layers.pop(layer)
        was_ready = all(future.done() for future in job.futures)
        elapsed = max(future.result() for future in job.futures)
        blob = job.buffer
        packed = mx.array(np.frombuffer(blob, dtype=np.uint32))
        mx.eval(packed)
        del blob
        batched = self._pool.batched(packed)
        with self._lock:
            self.metrics.bytes_read += self.model.expert_count * self.model.expert_blob_size
            self.metrics.read_seconds += elapsed
            self.metrics.batched_layers += 1
            if was_ready:
                self.metrics.prefetched_layer_hits += 1
        self._batched_layer = (layer, batched)
        try:
            yield batched
        finally:
            self._batched_layer = None

    @contextmanager
    def pin_layer(self, layer: int):
        with self._lock:
            self._pinned_layers.add(layer)
        try:
            yield
        finally:
            with self._lock:
                self._pinned_layers.discard(layer)

    def get_many(self, layer: int, expert_ids: list[int]) -> ResidentExperts:
        if not 0 <= layer < self.layer_count:
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
            individual_weights = self._pool.select_individual(physical_slots)
            self.metrics.pack_seconds += time.perf_counter() - pack_started
            return ResidentExperts(
                individual_weights,
                {expert: slot for slot, expert in enumerate(unique)},
            )

    def _touch(self, layer: int, expert: int, entry: _Entry, count: int) -> None:
        self._clock += 1
        entry.frequency += count
        entry.last_access = self._clock
        entry.version += 1
        heapq.heappush(
            self._heap,
            (entry.frequency, entry.last_access, entry.version, layer, expert),
        )

    def _evict(self, protected: set[tuple[int, int]]) -> tuple[tuple[int, int], _Entry]:
        protected_items: list[tuple[int, int, int, int, int]] = []
        pinned_items: list[tuple[int, int, int, int, int]] = []
        reserved_items: list[tuple[int, int, int, int, int]] = []
        try:
            while self._heap:
                item = heapq.heappop(self._heap)
                frequency, last_access, version, layer, expert = item
                key = (layer, expert)
                entry = self._entries.get(key)
                if entry is None or (
                    entry.frequency,
                    entry.last_access,
                    entry.version,
                ) != (frequency, last_access, version):
                    continue
                if key in protected:
                    protected_items.append(item)
                    continue
                if layer in self._pinned_layers:
                    pinned_items.append(item)
                    continue
                if self._layer_counts[layer] <= self._layer_reserve:
                    reserved_items.append(item)
                    continue
                return key, entry
            candidates = reserved_items or pinned_items
            if not candidates:
                raise RuntimeError("no expert cache slot can be evicted")
            item = candidates.pop(0)
            key = (item[3], item[4])
            return key, self._entries[key]
        finally:
            for item in (*protected_items, *pinned_items, *reserved_items):
                heapq.heappush(self._heap, item)

    def _decay_if_needed(self) -> None:
        if self._clock - self._last_decay < max(self.slots * 8, 64):
            return
        self._last_decay = self._clock
        self._heap = []
        for (layer, expert), entry in self._entries.items():
            entry.frequency = max(1, entry.frequency // 2)
            entry.version += 1
            heapq.heappush(
                self._heap,
                (entry.frequency, entry.last_access, entry.version, layer, expert),
            )

    def _read_blob(self, layer: int, expert: int) -> bytes:
        offset = expert * self.model.expert_blob_size
        return self._read_exact(layer, offset, self.model.expert_blob_size)

    def _read_into(
        self,
        layer: int,
        buffer: bytearray,
        start: int,
        end: int,
    ) -> float:
        started = time.perf_counter()
        view = memoryview(buffer)
        position = start
        maximum = 256 * 1024**2
        while position < end:
            stop = min(position + maximum, end)
            count = os.preadv(
                self._descriptors[layer],
                [view[position:stop]],
                position,
            )
            if count <= 0:
                raise EOFError(
                    f"expert file ended early at layer {layer}, offset {position}"
                )
            position += count
        return time.perf_counter() - started

    def _read_exact(self, layer: int, offset: int, length: int) -> bytes:
        remaining = length
        chunks: list[bytes] = []
        while remaining:
            chunk = os.pread(self._descriptors[layer], remaining, offset)
            if not chunk:
                raise EOFError(
                    f"expert file ended early at layer {layer}, offset {offset}"
                )
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
