from __future__ import annotations

import heapq
import os
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from .manifest import InstalledModel, Tensor


@dataclass(frozen=True)
class ExpertWeights:
    w1: mx.array
    w1_scales: mx.array
    w2: mx.array
    w2_scales: mx.array
    w3: mx.array
    w3_scales: mx.array
    w13: mx.array | None = None
    w13_scales: mx.array | None = None


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
    w13: mx.array | None = None
    w13_scales: mx.array | None = None


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    upload_seconds: float = 0.0
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
            upload_seconds=self.upload_seconds - before.upload_seconds,
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
    packed: mx.array
    futures: tuple[Future[float], ...]


class _ReadLimiter:
    """Limit aggregate preadv throughput across all read workers."""

    def __init__(self, bytes_per_second: int) -> None:
        if bytes_per_second <= 0:
            raise ValueError("read limit must be greater than zero")
        self._bytes_per_second = bytes_per_second
        self._lock = threading.Lock()

    def preadv(
        self,
        descriptor: int,
        views: list[memoryview],
        offset: int,
    ) -> int:
        with self._lock:
            started = time.perf_counter()
            count = os.preadv(descriptor, views, offset)
            delay = count / self._bytes_per_second - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
            return count


def _fused_slot_regions(model: InstalledModel) -> dict[str, Tensor]:
    source = {region.name: region for region in model.expert_regions}
    required = {
        "w1.weight",
        "w1.scale",
        "w2.weight",
        "w2.scale",
        "w3.weight",
        "w3.scale",
    }
    if set(source) != required:
        raise ValueError("expert blob does not contain the required regions")
    w1 = source["w1.weight"]
    w3 = source["w3.weight"]
    w1_scale = source["w1.scale"]
    w3_scale = source["w3.scale"]
    if w1.dtype != w3.dtype or w1.shape[1:] != w3.shape[1:]:
        raise ValueError("w1 and w3 weights cannot use one fused projection")
    if w1_scale.dtype != w3_scale.dtype or w1_scale.shape[1:] != w3_scale.shape[1:]:
        raise ValueError("w1 and w3 scales cannot use one fused projection")

    regions = {}
    offset = 0

    def add(name: str, template: Tensor) -> None:
        nonlocal offset
        regions[name] = Tensor(
            name,
            template.dtype,
            template.shape,
            offset,
            template.length,
        )
        offset += template.length

    add("w3.weight", w3)
    add("w1.weight", w1)
    regions["w13.weight"] = Tensor(
        "w13.weight",
        w3.dtype,
        (w3.shape[0] + w1.shape[0], *w3.shape[1:]),
        0,
        w3.length + w1.length,
    )
    add("w2.weight", source["w2.weight"])
    scale_offset = offset
    add("w3.scale", w3_scale)
    add("w1.scale", w1_scale)
    regions["w13.scale"] = Tensor(
        "w13.scale",
        w3_scale.dtype,
        (w3_scale.shape[0] + w1_scale.shape[0], *w3_scale.shape[1:]),
        scale_offset,
        w3_scale.length + w1_scale.length,
    )
    add("w2.scale", source["w2.scale"])
    if offset != model.expert_blob_size:
        raise ValueError("fused expert slot size does not match the manifest")
    return regions


class _SlotPool:
    """Keep each expert slot in a directly writable Metal buffer."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._source_regions = {
            region.name: region for region in model.expert_regions
        }
        self._regions = _fused_slot_regions(model)
        self._slots: list[mx.array | None] = [None] * slots
        self._views: list[memoryview | None] = [None] * slots
        self._loaded = bytearray(slots)

    def prepare(self, slots: list[int]) -> None:
        created = []
        for slot in slots:
            if self._slots[slot] is None:
                array = mx.empty((self._model.expert_blob_size,), dtype=mx.uint8)
                self._slots[slot] = array
                created.append(array)
        if created:
            mx.eval(*created)
            for slot in slots:
                array = self._slots[slot]
                if self._views[slot] is None and array is not None:
                    self._views[slot] = memoryview(array)

    def select_individual(self, slots: list[int]) -> tuple[ExpertWeights, ...]:
        arrays = []
        for slot in slots:
            array = self._slots[slot]
            if not self._loaded[slot] or array is None:
                raise RuntimeError("expert slot is empty")
            arrays.append(array)
        return tuple(
            ExpertWeights(
                w1=self._array(array, "w1.weight"),
                w1_scales=self._array(array, "w1.scale"),
                w2=self._array(array, "w2.weight"),
                w2_scales=self._array(array, "w2.scale"),
                w3=self._array(array, "w3.weight"),
                w3_scales=self._array(array, "w3.scale"),
                w13=self._array(array, "w13.weight"),
                w13_scales=self._array(array, "w13.scale"),
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

    def store(self, slots: list[int], blobs: list[bytes]) -> float:
        started = time.perf_counter()
        self.prepare(slots)
        for slot, blob in zip(slots, blobs):
            if len(blob) != self._model.expert_blob_size:
                raise ValueError("expert blob size does not match the manifest")
            source = memoryview(blob)
            for name, region in self._source_regions.items():
                self.writable_region(slot, name)[:] = source[
                    region.offset : region.offset + region.length
                ]
            self._loaded[slot] = 1
        return time.perf_counter() - started

    def writable_region(self, slot: int, name: str) -> memoryview:
        region = self._regions[name]
        view = self._views[slot]
        if view is None:
            raise RuntimeError("expert slot buffer is not prepared")
        return view[region.offset : region.offset + region.length]

    def write_views(self, buffer: memoryview, slot: int) -> list[memoryview]:
        base = slot * self._model.expert_blob_size
        return [
            buffer[
                base + self._regions[region.name].offset :
                base + self._regions[region.name].offset + region.length
            ]
            for region in self._model.expert_regions
        ]

    def mark_loaded(self, slot: int) -> None:
        self._loaded[slot] = 1

    def mark_empty(self, slot: int) -> None:
        self._loaded[slot] = 0

    def batched(self, packed: mx.array) -> BatchedExperts:
        return BatchedExperts(
            w1=self._batched_array(packed, "w1.weight"),
            w1_scales=self._batched_array(packed, "w1.scale"),
            w2=self._batched_array(packed, "w2.weight"),
            w2_scales=self._batched_array(packed, "w2.scale"),
            w3=self._batched_array(packed, "w3.weight"),
            w3_scales=self._batched_array(packed, "w3.scale"),
            w13=self._batched_array(packed, "w13.weight"),
            w13_scales=self._batched_array(packed, "w13.scale"),
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
        slots: int = 1_152,
        read_workers: int = 4,
        prefetch_read_workers: int = 2,
        layer_count: int | None = None,
        expert_directory: Path | None = None,
        route_trace_path: str | Path | None = None,
        ready_expert_decode: bool = False,
        read_limiter: _ReadLimiter | None = None,
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
        self.ready_expert_decode = ready_expert_decode
        self._read_limiter = read_limiter
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
        self._route_trace_path = route_trace_path
        if route_trace_path is not None:
            from .route_trace import RouteTraceRecorder

            self._route_trace = RouteTraceRecorder(
                self.layer_count,
                installed_model.expert_count,
                installed_model.selected_expert_count,
                installed_model.expert_blob_size,
            )
        else:
            self._route_trace = None
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
        if self._route_trace is not None and self._route_trace_path is not None:
            self._route_trace.write(self._route_trace_path)

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

    @property
    def route_trace_enabled(self) -> bool:
        return self._route_trace is not None

    @contextmanager
    def trace_routes(self, phase: str):
        if self._route_trace is None:
            yield
            return
        with self._route_trace.phase(phase):
            yield

    def record_routes(self, layer: int, selected: np.ndarray) -> None:
        if self._route_trace is not None:
            self._route_trace.record(layer, selected)

    def _record_residency(
        self,
        layer: int,
        selected: list[int],
        missing: list[int],
    ) -> None:
        if self._route_trace is not None:
            self._route_trace.record_residency(layer, selected, missing)

    def current_batched(self, layer: int) -> BatchedExperts | None:
        current = self._batched_layer
        return current[1] if current is not None and current[0] == layer else None

    def prefetch_layer(self, layer: int) -> None:
        if not 0 <= layer < self.layer_count:
            return
        with self._lock:
            if layer not in self._prefetched_layers:
                length = self.model.expert_count * self.model.expert_blob_size
                if length % 4:
                    raise ValueError("batched expert layer must be 4-byte aligned")
                packed = mx.empty((length // 4,), dtype=mx.uint32)
                mx.eval(packed)
                ranges = []
                step = (
                    self.model.expert_count + self.prefetch_read_workers - 1
                ) // self.prefetch_read_workers
                for start in range(0, self.model.expert_count, step):
                    ranges.append(
                        (start, min(start + step, self.model.expert_count))
                    )
                self._prefetched_layers[layer] = _LayerRead(
                    packed,
                    tuple(
                        self._executor.submit(
                            self._read_expert_range,
                            layer,
                            packed,
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
        packed = job.packed
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
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, missing)

        started = time.perf_counter()
        futures = {
            expert: self._executor.submit(
                self._read_expert_into_slot,
                layer,
                expert,
                assigned[expert],
            )
            for expert in missing
        }
        try:
            for future in futures.values():
                future.result()
        except Exception:
            for future in futures.values():
                future.cancel()
            for future in futures.values():
                try:
                    future.result()
                except Exception:
                    pass
            with self._lock:
                self._release_slots(layer, assigned)
            raise
        elapsed = time.perf_counter() - started if missing else 0.0

        with self._lock:
            self.metrics.bytes_read += len(missing) * self.model.expert_blob_size
            self.metrics.read_seconds += elapsed
            for slot in assigned.values():
                self._pool.mark_loaded(slot)
            self._decay_if_needed()
            physical_slots = [self._entries[(layer, expert)].slot for expert in unique]
            pack_started = time.perf_counter()
            individual_weights = self._pool.select_individual(physical_slots)
            self.metrics.pack_seconds += time.perf_counter() - pack_started
            return ResidentExperts(
                individual_weights,
                {expert: slot for slot, expert in enumerate(unique)},
            )

    def iter_ready(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> Iterator[tuple[int, ExpertWeights]]:
        """Yield resident and newly read experts without waiting for the slowest read."""
        if not 0 <= layer < self.layer_count:
            raise ValueError(f"invalid layer {layer}")
        frequencies = Counter(expert_ids)
        unique = sorted(frequencies)
        if len(unique) > self.slots:
            raise ValueError("selected experts exceed the expert slot count")
        if any(not 0 <= expert < self.model.expert_count for expert in unique):
            raise ValueError("selected experts contain an invalid expert ID")

        resident: list[tuple[int, int]] = []
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
                    resident.append((expert, entry.slot))
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, missing)

        started = time.perf_counter()
        futures = {
            self._executor.submit(
                self._read_expert_into_slot,
                layer,
                expert,
                assigned[expert],
            ): expert
            for expert in missing
        }
        for expert, slot in resident:
            pack_started = time.perf_counter()
            weights = self._pool.select_individual([slot])[0]
            with self._lock:
                self.metrics.pack_seconds += time.perf_counter() - pack_started
            yield expert, weights

        ready_at = started
        completed: set[int] = set()
        try:
            for future in as_completed(futures):
                expert = futures[future]
                finished = future.result()
                ready_at = max(ready_at, finished)
                completed.add(expert)
                with self._lock:
                    slot = assigned[expert]
                    self._pool.mark_loaded(slot)
                    pack_started = time.perf_counter()
                    weights = self._pool.select_individual([slot])[0]
                    self.metrics.pack_seconds += time.perf_counter() - pack_started
                yield expert, weights
        finally:
            incomplete = {
                expert: assigned[expert]
                for expert in missing
                if expert not in completed
            }
            if incomplete:
                for future, expert in futures.items():
                    if expert in incomplete:
                        future.cancel()
                for future, expert in futures.items():
                    if expert in incomplete:
                        try:
                            future.result()
                        except Exception:
                            pass
                with self._lock:
                    self._release_slots(layer, incomplete)

        with self._lock:
            self.metrics.bytes_read += len(completed) * self.model.expert_blob_size
            self.metrics.read_seconds += ready_at - started if missing else 0.0
            self._decay_if_needed()

    def _reserve_slots(
        self,
        layer: int,
        missing: list[int],
        frequencies: Counter,
        protected: set[tuple[int, int]],
    ) -> dict[int, int]:
        assigned = {}
        for expert in missing:
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
                self._pool.mark_empty(slot)
            entry = _Entry(slot, frequencies[expert], 0)
            self._entries[(layer, expert)] = entry
            self._layer_counts[layer] += 1
            self._touch(layer, expert, entry, 0)
            assigned[expert] = slot
        self._pool.prepare(list(assigned.values()))
        return assigned

    def _release_slots(self, layer: int, assigned: dict[int, int]) -> None:
        for expert, slot in assigned.items():
            entry = self._entries.get((layer, expert))
            if entry is None or entry.slot != slot:
                continue
            del self._entries[(layer, expert)]
            self._layer_counts[layer] -= 1
            self._pool.mark_empty(slot)
            self._free_slots.append(slot)

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

    def _read_expert_into_slot(self, layer: int, expert: int, slot: int) -> float:
        views = [
            self._pool.writable_region(slot, region.name)
            for region in self.model.expert_regions
        ]
        self._pread_views(
            layer,
            views,
            expert * self.model.expert_blob_size,
        )
        return time.perf_counter()

    def _read_expert_range(
        self,
        layer: int,
        packed: mx.array,
        start: int,
        end: int,
    ) -> float:
        started = time.perf_counter()
        view = memoryview(packed).cast("B")
        for expert in range(start, end):
            self._pread_views(
                layer,
                self._pool.write_views(view, expert),
                expert * self.model.expert_blob_size,
            )
        return time.perf_counter() - started

    def _pread_views(
        self,
        layer: int,
        views: list[memoryview],
        offset: int,
    ) -> None:
        pending = list(views)
        position = offset
        while pending:
            if self._read_limiter is None:
                count = os.preadv(self._descriptors[layer], pending, position)
            else:
                count = self._read_limiter.preadv(
                    self._descriptors[layer], pending, position
                )
            if count <= 0:
                raise EOFError(
                    f"expert file ended early at layer {layer}, offset {position}"
                )
            consumed = count
            while pending and consumed >= len(pending[0]):
                consumed -= len(pending[0])
                pending.pop(0)
            if pending and consumed:
                pending[0] = pending[0][consumed:]
            position += count
