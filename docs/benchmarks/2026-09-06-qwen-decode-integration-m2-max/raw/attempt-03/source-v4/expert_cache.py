from __future__ import annotations

import ctypes
import heapq
import os
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from .io_metrics import (
    EXPERT_FILE_CACHE_POLICIES,
    PageCacheReadClassification,
    configure_expert_file_cache_policy,
    page_cache_residency_snapshot,
)
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


@dataclass(frozen=True)
class QwenExpertWeights:
    gate_up: mx.array
    gate_up_scales: mx.array
    down: mx.array
    down_scales: mx.array


@dataclass(frozen=True)
class QwenBatchedExperts:
    gate_up: mx.array
    gate_up_scales: mx.array
    down: mx.array
    down_scales: mx.array


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
    expert_union_calls: int = 0
    routed_expert_assignments: int = 0
    expert_union_experts: int = 0
    expert_union_misses: int = 0
    speculative_prefetch_rounds: int = 0
    speculative_prefetch_requested_experts: int = 0
    speculative_prefetch_cache_resident_experts: int = 0
    speculative_prefetch_experts_read: int = 0
    speculative_prefetch_bytes_read: int = 0
    speculative_prefetch_read_seconds: float = 0.0
    speculative_prefetch_wait_seconds: float = 0.0
    speculative_scratch_hits: int = 0
    staged_expert_reads: int = 0
    staged_w13_bytes_read: int = 0
    staged_w2_bytes_read: int = 0
    staged_read_seconds: float = 0.0
    staged_w2_wait_seconds: float = 0.0
    staged_first_stage_submit_seconds: float = 0.0
    adaptive_prefill_planned_layers: int = 0
    adaptive_prefill_full_layers: int = 0
    adaptive_prefill_selective_layers: int = 0
    adaptive_prefill_union_experts: int = 0
    adaptive_prefill_read_experts: int = 0
    adaptive_prefill_bytes_read: int = 0
    adaptive_prefill_avoided_bytes: int = 0
    adaptive_prefill_plan_seconds: float = 0.0
    page_cache_probe_calls: int = 0
    page_cache_probe_failures: int = 0
    page_cache_classified_bytes: int = 0
    page_cache_resident_bytes_before_read: int = 0
    page_cache_nonresident_bytes_before_read: int = 0
    page_cache_unclassified_bytes: int = 0
    speculative_prefetch_page_cache_classified_bytes: int = 0
    speculative_prefetch_page_cache_resident_bytes_before_read: int = 0
    speculative_prefetch_page_cache_nonresident_bytes_before_read: int = 0
    speculative_prefetch_page_cache_unclassified_bytes: int = 0

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
            expert_union_calls=(
                self.expert_union_calls - before.expert_union_calls
            ),
            routed_expert_assignments=(
                self.routed_expert_assignments
                - before.routed_expert_assignments
            ),
            expert_union_experts=(
                self.expert_union_experts - before.expert_union_experts
            ),
            expert_union_misses=(
                self.expert_union_misses - before.expert_union_misses
            ),
            speculative_prefetch_rounds=(
                self.speculative_prefetch_rounds
                - before.speculative_prefetch_rounds
            ),
            speculative_prefetch_requested_experts=(
                self.speculative_prefetch_requested_experts
                - before.speculative_prefetch_requested_experts
            ),
            speculative_prefetch_cache_resident_experts=(
                self.speculative_prefetch_cache_resident_experts
                - before.speculative_prefetch_cache_resident_experts
            ),
            speculative_prefetch_experts_read=(
                self.speculative_prefetch_experts_read
                - before.speculative_prefetch_experts_read
            ),
            speculative_prefetch_bytes_read=(
                self.speculative_prefetch_bytes_read
                - before.speculative_prefetch_bytes_read
            ),
            speculative_prefetch_read_seconds=(
                self.speculative_prefetch_read_seconds
                - before.speculative_prefetch_read_seconds
            ),
            speculative_prefetch_wait_seconds=(
                self.speculative_prefetch_wait_seconds
                - before.speculative_prefetch_wait_seconds
            ),
            speculative_scratch_hits=(
                self.speculative_scratch_hits - before.speculative_scratch_hits
            ),
            staged_expert_reads=(
                self.staged_expert_reads - before.staged_expert_reads
            ),
            staged_w13_bytes_read=(
                self.staged_w13_bytes_read - before.staged_w13_bytes_read
            ),
            staged_w2_bytes_read=(
                self.staged_w2_bytes_read - before.staged_w2_bytes_read
            ),
            staged_read_seconds=(
                self.staged_read_seconds - before.staged_read_seconds
            ),
            staged_w2_wait_seconds=(
                self.staged_w2_wait_seconds - before.staged_w2_wait_seconds
            ),
            staged_first_stage_submit_seconds=(
                self.staged_first_stage_submit_seconds
                - before.staged_first_stage_submit_seconds
            ),
            adaptive_prefill_planned_layers=(
                self.adaptive_prefill_planned_layers
                - before.adaptive_prefill_planned_layers
            ),
            adaptive_prefill_full_layers=(
                self.adaptive_prefill_full_layers
                - before.adaptive_prefill_full_layers
            ),
            adaptive_prefill_selective_layers=(
                self.adaptive_prefill_selective_layers
                - before.adaptive_prefill_selective_layers
            ),
            adaptive_prefill_union_experts=(
                self.adaptive_prefill_union_experts
                - before.adaptive_prefill_union_experts
            ),
            adaptive_prefill_read_experts=(
                self.adaptive_prefill_read_experts
                - before.adaptive_prefill_read_experts
            ),
            adaptive_prefill_bytes_read=(
                self.adaptive_prefill_bytes_read
                - before.adaptive_prefill_bytes_read
            ),
            adaptive_prefill_avoided_bytes=(
                self.adaptive_prefill_avoided_bytes
                - before.adaptive_prefill_avoided_bytes
            ),
            adaptive_prefill_plan_seconds=(
                self.adaptive_prefill_plan_seconds
                - before.adaptive_prefill_plan_seconds
            ),
            page_cache_probe_calls=(
                self.page_cache_probe_calls - before.page_cache_probe_calls
            ),
            page_cache_probe_failures=(
                self.page_cache_probe_failures
                - before.page_cache_probe_failures
            ),
            page_cache_classified_bytes=(
                self.page_cache_classified_bytes
                - before.page_cache_classified_bytes
            ),
            page_cache_resident_bytes_before_read=(
                self.page_cache_resident_bytes_before_read
                - before.page_cache_resident_bytes_before_read
            ),
            page_cache_nonresident_bytes_before_read=(
                self.page_cache_nonresident_bytes_before_read
                - before.page_cache_nonresident_bytes_before_read
            ),
            page_cache_unclassified_bytes=(
                self.page_cache_unclassified_bytes
                - before.page_cache_unclassified_bytes
            ),
            speculative_prefetch_page_cache_classified_bytes=(
                self.speculative_prefetch_page_cache_classified_bytes
                - before.speculative_prefetch_page_cache_classified_bytes
            ),
            speculative_prefetch_page_cache_resident_bytes_before_read=(
                self.speculative_prefetch_page_cache_resident_bytes_before_read
                - before.speculative_prefetch_page_cache_resident_bytes_before_read
            ),
            speculative_prefetch_page_cache_nonresident_bytes_before_read=(
                self.speculative_prefetch_page_cache_nonresident_bytes_before_read
                - before.speculative_prefetch_page_cache_nonresident_bytes_before_read
            ),
            speculative_prefetch_page_cache_unclassified_bytes=(
                self.speculative_prefetch_page_cache_unclassified_bytes
                - before.speculative_prefetch_page_cache_unclassified_bytes
            ),
        )


@dataclass(frozen=True)
class ExpertUnionLayerMetrics:
    layer: int
    routed_expert_assignments: int
    unique_experts: int
    cache_misses: int


@dataclass
class ExpertUnionProfile:
    """Capture the per-layer expert union for one model forward."""

    layers: list[ExpertUnionLayerMetrics] = field(default_factory=list)

    @property
    def routed_expert_assignments(self) -> int:
        return sum(layer.routed_expert_assignments for layer in self.layers)

    @property
    def unique_experts(self) -> int:
        return sum(layer.unique_experts for layer in self.layers)

    @property
    def cache_misses(self) -> int:
        return sum(layer.cache_misses for layer in self.layers)


@dataclass
class _Entry:
    slot: int
    frequency: int
    last_access: int
    version: int = 0


@dataclass(frozen=True)
class _LayerRead:
    packed: mx.array
    futures: tuple[Future[_SpeculativeRead], ...]
    experts: tuple[int, ...]
    submitted: float


@dataclass(frozen=True)
class _SpeculativeRead:
    started: float
    finished: float
    page_cache: PageCacheReadClassification = PageCacheReadClassification()


@dataclass
class _ActivePrefetchTrace:
    layer: int
    job: _LayerRead
    reads: tuple[_SpeculativeRead, ...]
    deadline: float
    future_wait_seconds: float
    used_experts: set[int] = field(default_factory=set)
    compute_submit: float | None = None


class StagedReadyExpert:
    """Expose w13-ready weights while keeping partial slot admission private."""

    def __init__(
        self,
        cache: ExpertCache,
        expert: int,
        weights: ExpertWeights,
        slot: int,
        w13_read: _SpeculativeRead | None = None,
        w2_future: Future[_SpeculativeRead] | None = None,
    ) -> None:
        self._cache = cache
        self.expert = expert
        self.weights = weights
        self.slot = slot
        self.w13_read = w13_read
        self._w2_future = w2_future
        self.w2_read: _SpeculativeRead | None = None
        self.finished = w2_future is None

    def finish_w2(self) -> None:
        if self.finished:
            return
        if self._w2_future is None:
            raise RuntimeError("staged expert has no w2 future")
        wait_started = time.perf_counter()
        read = self._w2_future.result()
        waited = time.perf_counter() - wait_started
        cache = self._cache
        with cache._lock:
            cache._pool.mark_loaded(self.slot)
            cache.metrics.staged_w2_wait_seconds += waited
        self.w2_read = read
        self.finished = True


@dataclass(frozen=True)
class SpeculativePrefetchMetrics:
    requested_experts: int = 0
    cache_resident_experts: int = 0
    experts_read: int = 0
    bytes_read: int = 0
    useful_bytes: int = 0
    wasted_bytes: int = 0
    read_seconds: float = 0.0
    wait_seconds: float = 0.0
    page_cache_classified_bytes: int = 0
    page_cache_resident_bytes_before_read: int = 0
    page_cache_nonresident_bytes_before_read: int = 0
    page_cache_unclassified_bytes: int = 0
    useful_page_cache_resident_bytes_before_read: int = 0
    useful_page_cache_nonresident_bytes_before_read: int = 0
    useful_page_cache_unclassified_bytes: int = 0
    wasted_page_cache_resident_bytes_before_read: int = 0
    wasted_page_cache_nonresident_bytes_before_read: int = 0
    wasted_page_cache_unclassified_bytes: int = 0


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


def _qwen_slot_regions(model: InstalledModel) -> dict[str, Tensor]:
    regions = {region.name: region for region in model.expert_regions}
    if set(regions) != {
        "gate_up.weight",
        "gate_up.scale",
        "down.weight",
        "down.scale",
    }:
        raise ValueError("Qwen expert blob does not contain the required regions")
    if sum(region.length for region in regions.values()) != model.expert_blob_size:
        raise ValueError("Qwen expert slot size does not match the manifest")
    return regions


def _staged_slot_regions(
    model: InstalledModel,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    source = {region.name: region for region in model.expert_regions}
    w13: dict[str, Tensor] = {}
    offset = 0
    for name in ("w3.weight", "w1.weight"):
        region = source[name]
        w13[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    first = w13["w3.weight"]
    second = w13["w1.weight"]
    w13["w13.weight"] = Tensor(
        "w13.weight",
        first.dtype,
        (first.shape[0] + second.shape[0], *first.shape[1:]),
        first.offset,
        first.length + second.length,
    )
    scale_offset = offset
    for name in ("w3.scale", "w1.scale"):
        region = source[name]
        w13[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    first_scale = w13["w3.scale"]
    second_scale = w13["w1.scale"]
    w13["w13.scale"] = Tensor(
        "w13.scale",
        first_scale.dtype,
        (
            first_scale.shape[0] + second_scale.shape[0],
            *first_scale.shape[1:],
        ),
        scale_offset,
        first_scale.length + second_scale.length,
    )

    w2: dict[str, Tensor] = {}
    offset = 0
    for name in ("w2.weight", "w2.scale"):
        region = source[name]
        w2[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    if (
        sum(region.length for name, region in w13.items() if not name.startswith("w13"))
        + sum(region.length for region in w2.values())
        != model.expert_blob_size
    ):
        raise ValueError("staged expert regions do not preserve the blob size")
    return w13, w2


class _SlotPool:
    """Keep each expert slot in a directly writable Metal buffer."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._source_regions = {
            region.name: region for region in model.expert_regions
        }
        self._regions = (
            _qwen_slot_regions(model) if model.is_qwen else _fused_slot_regions(model)
        )
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
        if self._model.is_qwen:
            return tuple(
                QwenExpertWeights(
                    gate_up=self._array(array, "gate_up.weight"),
                    gate_up_scales=self._array(array, "gate_up.scale"),
                    down=self._array(array, "down.weight"),
                    down_scales=self._array(array, "down.scale"),
                )
                for array in arrays
            )
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
        if region.dtype == "U32":
            return value.view(mx.uint32).reshape(region.shape)
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
        if self._model.is_qwen:
            return QwenBatchedExperts(
                gate_up=self._batched_array(packed, "gate_up.weight"),
                gate_up_scales=self._batched_array(packed, "gate_up.scale"),
                down=self._batched_array(packed, "down.weight"),
                down_scales=self._batched_array(packed, "down.scale"),
            )
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
        if region.dtype == "U32":
            shape = (self._model.expert_count, *region.shape)
            row_strides = []
            stride = 1
            for size in reversed(region.shape):
                row_strides.append(stride)
                stride *= size
            return mx.as_strided(
                packed,
                shape=shape,
                strides=(
                    self._model.expert_blob_size // 4,
                    *reversed(row_strides),
                ),
                offset=region.offset // 4,
            )
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


class _StagedSlotPool(_SlotPool):
    """Split each direct slot into fixed w13 and w2 Metal-visible arrays."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._source_regions = {
            region.name: region for region in model.expert_regions
        }
        self._regions = _fused_slot_regions(model)
        self._w13_regions, self._w2_regions = _staged_slot_regions(model)
        self._w13_size = sum(
            region.length
            for name, region in self._w13_regions.items()
            if not name.startswith("w13")
        )
        self._w2_size = sum(region.length for region in self._w2_regions.values())
        self._w13_slots: list[mx.array | None] = [None] * slots
        self._w2_slots: list[mx.array | None] = [None] * slots
        self._w13_views: list[memoryview | None] = [None] * slots
        self._w2_views: list[memoryview | None] = [None] * slots
        self._loaded = bytearray(slots)

    def prepare(self, slots: list[int]) -> None:
        created = []
        for slot in slots:
            if self._w13_slots[slot] is None:
                self._w13_slots[slot] = mx.empty(
                    (self._w13_size,),
                    dtype=mx.uint8,
                )
                self._w2_slots[slot] = mx.empty(
                    (self._w2_size,),
                    dtype=mx.uint8,
                )
                created.extend(
                    [self._w13_slots[slot], self._w2_slots[slot]]
                )
        if created:
            mx.eval(*created)
            for slot in slots:
                w13 = self._w13_slots[slot]
                w2 = self._w2_slots[slot]
                if self._w13_views[slot] is None and w13 is not None:
                    self._w13_views[slot] = memoryview(w13)
                if self._w2_views[slot] is None and w2 is not None:
                    self._w2_views[slot] = memoryview(w2)

    @staticmethod
    def _array_from(
        packed: mx.array,
        region: Tensor,
    ) -> mx.array:
        value = packed[region.offset : region.offset + region.length]
        value = value.reshape(region.shape)
        if region.dtype == "I8":
            value = value.view(mx.int8)
        return value.view(mx.uint32) if region.name.endswith(".weight") else value

    def _weights(self, slot: int, require_loaded: bool) -> ExpertWeights:
        w13 = self._w13_slots[slot]
        w2 = self._w2_slots[slot]
        if w13 is None or w2 is None or (require_loaded and not self._loaded[slot]):
            raise RuntimeError("staged expert slot is empty")
        return ExpertWeights(
            w1=self._array_from(w13, self._w13_regions["w1.weight"]),
            w1_scales=self._array_from(w13, self._w13_regions["w1.scale"]),
            w2=self._array_from(w2, self._w2_regions["w2.weight"]),
            w2_scales=self._array_from(w2, self._w2_regions["w2.scale"]),
            w3=self._array_from(w13, self._w13_regions["w3.weight"]),
            w3_scales=self._array_from(w13, self._w13_regions["w3.scale"]),
            w13=self._array_from(w13, self._w13_regions["w13.weight"]),
            w13_scales=self._array_from(w13, self._w13_regions["w13.scale"]),
        )

    def select_individual(self, slots: list[int]) -> tuple[ExpertWeights, ...]:
        return tuple(self._weights(slot, require_loaded=True) for slot in slots)

    def select_staged_individual(
        self,
        slots: list[int],
    ) -> tuple[ExpertWeights, ...]:
        return tuple(self._weights(slot, require_loaded=False) for slot in slots)

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
        if name in self._w13_regions:
            region = self._w13_regions[name]
            view = self._w13_views[slot]
        else:
            region = self._w2_regions[name]
            view = self._w2_views[slot]
        if view is None:
            raise RuntimeError("staged expert slot buffer is not prepared")
        return view[region.offset : region.offset + region.length]

    def mark_loaded(self, slot: int) -> None:
        self._loaded[slot] = 1

    def mark_empty(self, slot: int) -> None:
        self._loaded[slot] = 0


class _QwenArenaSlotPool(_SlotPool):
    """Canonical blobs in bounded arenas; views never flatten over 2**31 words."""

    PAGE_SLOTS = 512

    def __init__(self, model: InstalledModel, slots: int) -> None:
        if not model.is_qwen or model.expert_blob_size % 4:
            raise ValueError("Qwen arena requires aligned Qwen expert blobs")
        super().__init__(model, slots)
        self.page_slots = min(self.PAGE_SLOTS, (2**31 - 1) // (model.expert_blob_size // 4))
        if self.page_slots < 1:
            raise ValueError("Qwen expert blob exceeds MLX's view dimension limit")
        self.arenas: dict[int, mx.array] = {}
        self.grouped: dict[int, QwenBatchedExperts] = {}

    def prepare(self, slots: list[int]) -> None:
        for page in sorted({slot // self.page_slots for slot in slots}):
            if page not in self.arenas:
                count = min(self.page_slots, len(self._slots) - page * self.page_slots)
                arena = mx.empty((count, self._model.expert_blob_size // 4), dtype=mx.uint32)
                mx.eval(arena)
                proxy = _SlotPool(replace(self._model, expert_count=count), 0)
                grouped = proxy.batched(arena)
                mx.eval(*vars(grouped).values())
                self.arenas[page], self.grouped[page] = arena, grouped
        created = []
        for slot in slots:
            if self._slots[slot] is None:
                page, index = divmod(slot, self.page_slots)
                words = self._model.expert_blob_size // 4
                # Ordinary array[index] slicing can copy; CPU writes must alias.
                self._slots[slot] = mx.as_strided(
                    self.arenas[page], shape=(words,), strides=(1,), offset=index * words
                ).view(mx.uint8)
                created.append(self._slots[slot])
        if created:
            mx.eval(*created)
        for slot in slots:
            if self._views[slot] is None:
                page, index = divmod(slot, self.page_slots)
                base = np.asarray(self.arenas[page]).ctypes.data
                array = self._slots[slot]
                if np.asarray(array).ctypes.data != base + index * self._model.expert_blob_size:
                    raise RuntimeError("Qwen slot view does not alias its arena")
                self._views[slot] = memoryview(array)

    def grouped_for_slots(self, slots: list[int]):
        pages: dict[int, tuple[list[int], list[int]]] = {}
        for position, slot in enumerate(slots):
            if not self._loaded[slot]:
                raise RuntimeError("grouped Qwen weights contain an unloaded slot")
            page, index = divmod(slot, self.page_slots)
            positions, indices = pages.setdefault(page, ([], []))
            positions.append(position)
            indices.append(index)
        return [(self.grouped[page], indices, positions) for page, (positions, indices) in pages.items()]


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
        page_cache_probe: bool = False,
        file_cache_policy: str = "cached",
        staged_expert_streaming: bool = False,
        qwen_grouped_decode: bool = False,
    ) -> None:
        if qwen_grouped_decode and (not installed_model.is_qwen or staged_expert_streaming):
            raise ValueError("grouped Decode requires Qwen without staged streaming")
        if slots < installed_model.selected_expert_count:
            raise ValueError("slot count must hold at least one token's routed experts")
        if read_workers < 1:
            raise ValueError("read worker count must be greater than zero")
        if prefetch_read_workers < 1:
            raise ValueError("prefetch read worker count must be greater than zero")
        if file_cache_policy not in EXPERT_FILE_CACHE_POLICIES:
            raise ValueError(
                f"unknown expert-file cache policy: {file_cache_policy}"
            )
        self.model = installed_model
        self.slots = slots
        self.read_workers = read_workers
        self.prefetch_read_workers = min(read_workers, prefetch_read_workers)
        self.layer_count = layer_count or installed_model.layer_count
        self.expert_directory = expert_directory or installed_model.root / "experts"
        self.ready_expert_decode = ready_expert_decode
        self._read_limiter = read_limiter
        self.page_cache_probe = page_cache_probe
        self.file_cache_policy = file_cache_policy
        self.staged_expert_streaming = staged_expert_streaming
        self.qwen_grouped_decode = qwen_grouped_decode
        self.qwen_decode_active = False
        self._qwen_decode_request = False
        self._qwen_prefill_remaining: int | None = None
        filesystem = os.statvfs(self.expert_directory)
        self.direct_io_alignment = (
            int(filesystem.f_frsize or filesystem.f_bsize)
            if file_cache_policy == "bypass"
            else 0
        )
        if self.direct_io_alignment < 0:
            raise ValueError("direct-I/O alignment cannot be negative")
        if self.layer_count < 1:
            raise ValueError("expert cache layer count must be greater than zero")
        self.metrics = CacheMetrics()
        self._pool = (
            _StagedSlotPool(installed_model, slots)
            if staged_expert_streaming
            else (_QwenArenaSlotPool if qwen_grouped_decode else _SlotPool)(installed_model, slots)
        )
        self._entries: dict[tuple[int, int], _Entry] = {}
        self._free_slots = list(reversed(range(slots)))
        self._heap: list[tuple[int, int, int, int, int]] = []
        self._layer_counts = [0] * self.layer_count
        base = slots // self.layer_count
        self._layer_reserve = base // 2
        self._clock = 0
        self._last_decay = 0
        self._pinned_layers: set[int] = set()
        self._speculative_pinned_keys: set[tuple[int, int]] = set()
        self._expert_union_profiles: list[ExpertUnionProfile] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=read_workers)
        self._staged_w2_executor = (
            ThreadPoolExecutor(max_workers=read_workers)
            if staged_expert_streaming
            else None
        )
        self._prefetched_layers: dict[int, _LayerRead] = {}
        self.speculative_slots = 0
        self._speculative_pool: _SlotPool | None = None
        self._active_speculative_prefetch: _SpeculativeExpertPrefetch | None = None
        self._batched_layer: tuple[int, BatchedExperts] | None = None
        self._active_prefetch_trace: _ActivePrefetchTrace | None = None
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
                descriptor = os.open(
                    self.expert_directory / f"layer_{layer:02d}.bin",
                    os.O_RDONLY,
                )
                try:
                    configure_expert_file_cache_policy(
                        descriptor,
                        self.file_cache_policy,
                    )
                except Exception:
                    os.close(descriptor)
                    raise
                self._descriptors.append(descriptor)
        except Exception:
            for descriptor in self._descriptors:
                os.close(descriptor)
            self._executor.shutdown(wait=False, cancel_futures=True)
            if self._staged_w2_executor is not None:
                self._staged_w2_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            raise

    def close(self) -> None:
        speculative = self._active_speculative_prefetch
        if speculative is not None:
            speculative.close()
        for job in self._prefetched_layers.values():
            for future in job.futures:
                future.cancel()
        self._prefetched_layers.clear()
        self._executor.shutdown(wait=True)
        if self._staged_w2_executor is not None:
            self._staged_w2_executor.shutdown(wait=True)
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

    @contextmanager
    def qwen_decode_request(self):
        """Scope the final Prefill token and Decode dispatch to one request."""
        if not self.qwen_grouped_decode:
            yield
            return
        if self._qwen_decode_request:
            raise RuntimeError("Qwen Decode requests must be serialized")
        self._qwen_decode_request = True
        self._qwen_prefill_remaining = None
        try:
            yield
        finally:
            self.qwen_decode_active = False
            self._qwen_decode_request = False
            self._qwen_prefill_remaining = None

    def set_qwen_decode_prefill(self, token_count: int) -> None:
        if not self._qwen_decode_request or token_count < 1:
            raise RuntimeError("Qwen Decode requires a request and its uncached prompt")
        self._qwen_prefill_remaining = token_count

    @contextmanager
    def qwen_decode_step(self, token_count: int):
        self.qwen_decode_active = (
            self._qwen_decode_request and token_count == 1 and self._qwen_prefill_remaining == 0
        )
        if self._qwen_decode_request and self._qwen_prefill_remaining is not None:
            self._qwen_prefill_remaining = max(0, self._qwen_prefill_remaining - token_count)
        try:
            yield
        finally:
            self.qwen_decode_active = False

    def qwen_grouped_weights(
        self, layer: int, expert_ids: list[int]
    ):
        if not self.qwen_decode_active or not isinstance(self._pool, _QwenArenaSlotPool):
            raise RuntimeError("grouped Qwen weights require an active Decode step")
        # The caller materializes dependent router indices before any slot reuse.
        # Preserve the existing reservation, reads, metrics and LFU decisions.
        self.get_many(layer, expert_ids)
        with self._lock:
            physical = [self._entries[(layer, expert)].slot for expert in expert_ids]
            return self._pool.grouped_for_slots(physical)

    def resident_expert_keys(
        self,
        experts_by_layer: dict[int, list[int] | tuple[int, ...]],
    ) -> frozenset[tuple[int, int]]:
        """Snapshot which requested experts are currently in the main LFU cache."""
        requested = {
            (layer, expert)
            for layer, experts in experts_by_layer.items()
            for expert in experts
        }
        with self._lock:
            return frozenset(requested.intersection(self._entries))

    def metrics_snapshot(self) -> CacheMetrics:
        with self._lock:
            return replace(self.metrics)

    def record_routing_sync(self, seconds: float) -> None:
        with self._lock:
            self.metrics.routing_sync_seconds += seconds

    def record_gather_qmm(self, calls: int = 3) -> None:
        with self._lock:
            self.metrics.gather_qmm_calls += calls

    def record_staged_first_stage_submit(self, seconds: float) -> None:
        with self._lock:
            self.metrics.staged_first_stage_submit_seconds += seconds

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
        active = self._active_prefetch_trace
        if active is not None and active.layer == layer:
            active.used_experts.update(
                int(expert) for expert in np.asarray(selected).reshape(-1)
            )
        if self._route_trace is not None:
            self._route_trace.record(layer, selected)

    @contextmanager
    def capture_expert_unions(self) -> Iterator[ExpertUnionProfile]:
        """Capture the unique expert set used by each layer in one forward."""
        profile = ExpertUnionProfile()
        with self._lock:
            self._expert_union_profiles.append(profile)
        try:
            yield profile
        finally:
            with self._lock:
                for index, active in enumerate(self._expert_union_profiles):
                    if active is profile:
                        del self._expert_union_profiles[index]
                        break

    def _record_expert_union_locked(
        self,
        layer: int,
        routed_expert_assignments: int,
        unique_experts: int,
        cache_misses: int,
    ) -> None:
        self.metrics.expert_union_calls += 1
        self.metrics.routed_expert_assignments += routed_expert_assignments
        self.metrics.expert_union_experts += unique_experts
        self.metrics.expert_union_misses += cache_misses
        if not self._expert_union_profiles:
            return
        metrics = ExpertUnionLayerMetrics(
            layer=layer,
            routed_expert_assignments=routed_expert_assignments,
            unique_experts=unique_experts,
            cache_misses=cache_misses,
        )
        for profile in self._expert_union_profiles:
            profile.layers.append(metrics)

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

    def record_compute_submit(self, layer: int) -> None:
        active = self._active_prefetch_trace
        if (
            active is not None
            and active.layer == layer
            and active.compute_submit is None
        ):
            active.compute_submit = time.perf_counter()

    def prefetch_layer(
        self,
        layer: int,
        experts: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if not 0 <= layer < self.layer_count:
            return
        selected = (
            tuple(range(self.model.expert_count))
            if experts is None
            else tuple(sorted(set(experts)))
        )
        if not selected:
            raise ValueError("batched expert prefetch cannot be empty")
        if any(not 0 <= expert < self.model.expert_count for expert in selected):
            raise ValueError("batched expert prefetch contains an invalid expert")
        with self._lock:
            existing = self._prefetched_layers.get(layer)
            if existing is not None:
                if existing.experts != selected:
                    raise RuntimeError(
                        "batched expert layer is already prefetched with a "
                        "different expert set"
                    )
                return
            length = self.model.expert_count * self.model.expert_blob_size
            if length % 4:
                raise ValueError("batched expert layer must be 4-byte aligned")
            packed = mx.empty((length // 4,), dtype=mx.uint32)
            mx.eval(packed)
            step = (
                len(selected) + self.prefetch_read_workers - 1
            ) // self.prefetch_read_workers
            batches = [
                selected[start : start + step]
                for start in range(0, len(selected), step)
            ]
            submitted = time.perf_counter()
            self._prefetched_layers[layer] = _LayerRead(
                packed,
                tuple(
                    self._executor.submit(
                        self._read_expert_ids,
                        layer,
                        packed,
                        batch,
                    )
                    for batch in batches
                ),
                selected,
                submitted,
            )

    @contextmanager
    def batched_layer(
        self,
        layer: int,
        enabled: bool = True,
        experts: list[int] | tuple[int, ...] | None = None,
        adaptive: bool = False,
    ):
        if not enabled:
            yield None
            return
        self.prefetch_layer(layer, experts)
        with self._lock:
            job = self._prefetched_layers.pop(layer)
        deadline = time.perf_counter()
        was_ready = all(future.done() for future in job.futures)
        wait_started = time.perf_counter()
        reads = tuple(future.result() for future in job.futures)
        future_wait_seconds = time.perf_counter() - wait_started
        elapsed = max(
            (read.finished - read.started for read in reads),
            default=0.0,
        )
        packed = job.packed
        batched = self._pool.batched(packed)
        with self._lock:
            self.metrics.bytes_read += len(job.experts) * self.model.expert_blob_size
            if adaptive:
                self.metrics.adaptive_prefill_bytes_read += (
                    len(job.experts) * self.model.expert_blob_size
                )
            self.metrics.read_seconds += elapsed
            self.metrics.batched_layers += 1
            if was_ready:
                self.metrics.prefetched_layer_hits += 1
        self._batched_layer = (layer, batched)
        self._active_prefetch_trace = _ActivePrefetchTrace(
            layer=layer,
            job=job,
            reads=reads,
            deadline=deadline,
            future_wait_seconds=future_wait_seconds,
        )
        try:
            yield batched
        finally:
            active = self._active_prefetch_trace
            if active is not None and self._route_trace is not None:
                self._route_trace.record_prefetch_event(
                    layer=active.layer,
                    requested_experts=len(active.job.experts),
                    used_experts=len(active.used_experts),
                    read_submit=active.job.submitted,
                    read_start=min(read.started for read in active.reads),
                    read_complete=max(read.finished for read in active.reads),
                    expert_deadline=active.deadline,
                    compute_submit=active.compute_submit,
                    future_wait_seconds=active.future_wait_seconds,
                )
            self._active_prefetch_trace = None
            self._batched_layer = None

    def record_adaptive_prefill_decision(
        self,
        *,
        union_experts: int,
        read_experts: int,
        full_layer: bool,
        plan_seconds: float,
    ) -> None:
        if not 0 < union_experts <= self.model.expert_count:
            raise ValueError("adaptive prefill union is outside expert capacity")
        if not union_experts <= read_experts <= self.model.expert_count:
            raise ValueError("adaptive prefill read set does not cover its union")
        with self._lock:
            self.metrics.adaptive_prefill_planned_layers += 1
            self.metrics.adaptive_prefill_full_layers += int(full_layer)
            self.metrics.adaptive_prefill_selective_layers += int(not full_layer)
            self.metrics.adaptive_prefill_union_experts += union_experts
            self.metrics.adaptive_prefill_read_experts += read_experts
            self.metrics.adaptive_prefill_avoided_bytes += (
                self.model.expert_count - read_experts
            ) * self.model.expert_blob_size
            self.metrics.adaptive_prefill_plan_seconds += plan_seconds

    @contextmanager
    def pin_layer(self, layer: int):
        with self._lock:
            self._pinned_layers.add(layer)
        try:
            yield
        finally:
            with self._lock:
                self._pinned_layers.discard(layer)

    def configure_speculative_scratch(self, slots: int) -> None:
        """Allocate a fixed scratch pool used only by speculative verification."""
        if slots < 0:
            raise ValueError("speculative scratch slots must be zero or greater")
        with self._lock:
            if self._active_speculative_prefetch is not None:
                raise RuntimeError("cannot resize active speculative scratch")
            if self._speculative_pool is not None:
                if slots != self.speculative_slots:
                    raise RuntimeError("speculative scratch is already configured")
                return
            self.speculative_slots = slots
            if slots == 0:
                return
            pool = (
                _StagedSlotPool(self.model, slots)
                if self.staged_expert_streaming
                else _SlotPool(self.model, slots)
            )
            self._speculative_pool = pool
        pool.prepare(list(range(slots)))

    @contextmanager
    def speculative_prefetch(
        self,
        experts_by_layer: dict[int, list[int] | tuple[int, ...]],
    ) -> Iterator[_SpeculativeExpertPrefetch]:
        """Read exact speculative experts into scratch without LFU admission."""
        handle = self._begin_speculative_prefetch(experts_by_layer)
        try:
            yield handle
        finally:
            handle.close()

    def speculative_prefetch_active(self, layer: int) -> bool:
        with self._lock:
            active = self._active_speculative_prefetch
            return active is not None and active.covers_layer(layer)

    def _begin_speculative_prefetch(
        self,
        experts_by_layer: dict[int, list[int] | tuple[int, ...]],
    ) -> _SpeculativeExpertPrefetch:
        requested: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for layer, experts in experts_by_layer.items():
            if not 0 <= layer < self.layer_count:
                raise ValueError(f"invalid layer {layer}")
            for expert in experts:
                if not 0 <= expert < self.model.expert_count:
                    raise ValueError(f"invalid expert {expert}")
                key = (layer, expert)
                if key not in seen:
                    requested.append(key)
                    seen.add(key)

        with self._lock:
            if self._active_speculative_prefetch is not None:
                raise RuntimeError("a speculative prefetch is already active")
            pool = self._speculative_pool
            if pool is None:
                raise RuntimeError("speculative scratch is not configured")
            resident = {key for key in requested if key in self._entries}
            missing = [key for key in requested if key not in resident]
            if len(missing) > self.speculative_slots:
                raise ValueError(
                    f"speculative prefetch needs {len(missing)} scratch slots, "
                    f"but only {self.speculative_slots} are configured"
                )
            self._speculative_pinned_keys.update(resident)
            scratch_slots = {key: slot for slot, key in enumerate(missing)}
            for slot in scratch_slots.values():
                pool.mark_empty(slot)
            handle = _SpeculativeExpertPrefetch(
                self,
                tuple(requested),
                frozenset(resident),
                scratch_slots,
            )
            self._active_speculative_prefetch = handle

        try:
            handle.start()
        except Exception:
            handle.close()
            raise
        return handle

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
        scratch: list[int] = []
        resident: list[int] = []
        protected = {(layer, expert) for expert in unique}
        with self._lock:
            speculative = self._active_speculative_prefetch
            for expert in unique:
                entry = self._entries.get((layer, expert))
                if entry is None:
                    self.metrics.misses += 1
                    if (
                        speculative is not None
                        and speculative.has_scratch(layer, expert)
                    ):
                        scratch.append(expert)
                    else:
                        missing.append(expert)
                else:
                    self.metrics.hits += 1
                    self._touch(layer, expert, entry, frequencies[expert])
                    resident.append(expert)
            self._record_expert_union_locked(
                layer,
                len(expert_ids),
                len(unique),
                len(missing) + len(scratch),
            )
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, [*missing, *scratch])

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
        scratch_weights = (
            speculative.scratch_weights(layer, scratch)
            if scratch and speculative is not None
            else ()
        )

        with self._lock:
            self.metrics.bytes_read += len(missing) * self.model.expert_blob_size
            self.metrics.read_seconds += elapsed
            for slot in assigned.values():
                self._pool.mark_loaded(slot)
            self._decay_if_needed()
            pack_started = time.perf_counter()
            main_experts = [*resident, *missing]
            main_weights = self._pool.select_individual(
                [self._entries[(layer, expert)].slot for expert in main_experts]
            )
            self.metrics.pack_seconds += time.perf_counter() - pack_started
            weights_by_expert = dict(zip(main_experts, main_weights))
            weights_by_expert.update(zip(scratch, scratch_weights))
            individual_weights = tuple(weights_by_expert[expert] for expert in unique)
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
            self._record_expert_union_locked(
                layer,
                len(expert_ids),
                len(unique),
                len(missing),
            )
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

    def iter_staged_ready(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> Iterator[StagedReadyExpert]:
        """Yield w13-ready split slots and admit each only after w2 succeeds."""
        pool = self._pool
        w2_executor = self._staged_w2_executor
        if (
            not self.staged_expert_streaming
            or not isinstance(pool, _StagedSlotPool)
            or w2_executor is None
        ):
            raise RuntimeError("staged expert streaming is not configured")
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
            self._record_expert_union_locked(
                layer,
                len(expert_ids),
                len(unique),
                len(missing),
            )
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, missing)

        started = time.perf_counter()
        begin_futures = {
            self._executor.submit(
                self._begin_staged_expert_read,
                layer,
                expert,
                assigned[expert],
            ): expert
            for expert in missing
        }
        for expert, slot in resident:
            pack_started = time.perf_counter()
            weights = pool.select_individual([slot])[0]
            with self._lock:
                self.metrics.pack_seconds += time.perf_counter() - pack_started
            yield StagedReadyExpert(self, expert, weights, slot)

        ready_at = started
        completed: set[int] = set()
        handles: dict[int, StagedReadyExpert] = {}
        try:
            for future in as_completed(begin_futures):
                expert = begin_futures[future]
                w13_read, w2_future = future.result()
                slot = assigned[expert]
                pack_started = time.perf_counter()
                weights = pool.select_staged_individual([slot])[0]
                with self._lock:
                    self.metrics.pack_seconds += (
                        time.perf_counter() - pack_started
                    )
                handle = StagedReadyExpert(
                    self,
                    expert,
                    weights,
                    slot,
                    w13_read,
                    w2_future,
                )
                handles[expert] = handle
                yield handle
                if not handle.finished:
                    handle.finish_w2()
                if handle.w2_read is None:
                    raise RuntimeError("staged expert completed without w2 read")
                ready_at = max(ready_at, handle.w2_read.finished)
                completed.add(expert)
        finally:
            for future, expert in begin_futures.items():
                if expert not in completed:
                    future.cancel()
            for future, expert in begin_futures.items():
                if expert in completed:
                    continue
                try:
                    _, pending_w2 = future.result()
                except Exception:
                    continue
                pending_w2.cancel()
                try:
                    pending_w2.result()
                except Exception:
                    pass
            for expert, handle in handles.items():
                if expert in completed or handle._w2_future is None:
                    continue
                handle._w2_future.cancel()
                try:
                    handle._w2_future.result()
                except Exception:
                    pass
            incomplete = {
                expert: assigned[expert]
                for expert in missing
                if expert not in completed
            }
            if incomplete:
                with self._lock:
                    self._release_slots(layer, incomplete)

        with self._lock:
            completed_count = len(completed)
            self.metrics.bytes_read += completed_count * self.model.expert_blob_size
            self.metrics.read_seconds += ready_at - started if missing else 0.0
            self.metrics.staged_expert_reads += completed_count
            self.metrics.staged_w13_bytes_read += completed_count * pool._w13_size
            self.metrics.staged_w2_bytes_read += completed_count * pool._w2_size
            self.metrics.staged_read_seconds += (
                ready_at - started if missing else 0.0
            )
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
        speculative_items: list[tuple[int, int, int, int, int]] = []
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
                if key in self._speculative_pinned_keys:
                    speculative_items.append(item)
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
            for item in (
                *protected_items,
                *speculative_items,
                *pinned_items,
                *reserved_items,
            ):
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
        return self._read_expert_into_pool(self._pool, layer, expert, slot).finished

    def _begin_staged_expert_read(
        self,
        layer: int,
        expert: int,
        slot: int,
    ) -> tuple[_SpeculativeRead, Future[_SpeculativeRead]]:
        pool = self._pool
        executor = self._staged_w2_executor
        if not isinstance(pool, _StagedSlotPool) or executor is None:
            raise RuntimeError("staged expert streaming is not configured")
        w13_read = self._read_staged_w13_into_pool(
            pool,
            layer,
            expert,
            slot,
        )
        w2_future = executor.submit(
            self._read_staged_w2_into_pool,
            pool,
            layer,
            expert,
            slot,
        )
        return w13_read, w2_future

    def _read_staged_w13_into_pool(
        self,
        pool: _StagedSlotPool,
        layer: int,
        expert: int,
        slot: int,
    ) -> _SpeculativeRead:
        source = {region.name: region for region in self.model.expert_regions}
        base = expert * self.model.expert_blob_size
        started = time.perf_counter()
        page_cache = self._pread_views(
            layer,
            [
                pool.writable_region(slot, "w1.weight"),
                pool.writable_region(slot, "w1.scale"),
            ],
            base + source["w1.weight"].offset,
        )
        page_cache += self._pread_views(
            layer,
            [
                pool.writable_region(slot, "w3.weight"),
                pool.writable_region(slot, "w3.scale"),
            ],
            base + source["w3.weight"].offset,
        )
        return _SpeculativeRead(
            started,
            time.perf_counter(),
            page_cache,
        )

    def _read_staged_w2_into_pool(
        self,
        pool: _StagedSlotPool,
        layer: int,
        expert: int,
        slot: int,
    ) -> _SpeculativeRead:
        source = {region.name: region for region in self.model.expert_regions}
        base = expert * self.model.expert_blob_size
        started = time.perf_counter()
        page_cache = self._pread_views(
            layer,
            [
                pool.writable_region(slot, "w2.weight"),
                pool.writable_region(slot, "w2.scale"),
            ],
            base + source["w2.weight"].offset,
        )
        return _SpeculativeRead(
            started,
            time.perf_counter(),
            page_cache,
        )

    def _read_expert_into_pool(
        self,
        pool: _SlotPool,
        layer: int,
        expert: int,
        slot: int,
    ) -> _SpeculativeRead:
        started = time.perf_counter()
        views = [
            pool.writable_region(slot, region.name)
            for region in self.model.expert_regions
        ]
        page_cache = self._pread_views(
            layer,
            views,
            expert * self.model.expert_blob_size,
        )
        return _SpeculativeRead(started, time.perf_counter(), page_cache)

    def _read_expert_ids(
        self,
        layer: int,
        packed: mx.array,
        experts: tuple[int, ...],
    ) -> _SpeculativeRead:
        started = time.perf_counter()
        view = memoryview(packed).cast("B")
        for expert in experts:
            self._pread_views(
                layer,
                self._pool.write_views(view, expert),
                expert * self.model.expert_blob_size,
            )
        return _SpeculativeRead(started, time.perf_counter())

    def _pread_views(
        self,
        layer: int,
        views: list[memoryview],
        offset: int,
    ) -> PageCacheReadClassification:
        pending = list(views)
        position = offset
        page_cache = PageCacheReadClassification()
        while pending:
            if self.file_cache_policy == "bypass":
                self._validate_direct_read(position, pending)
            sample = None
            requested = sum(len(view) for view in pending)
            if self.page_cache_probe:
                sample = page_cache_residency_snapshot(
                    self._descriptors[layer],
                    position,
                    requested,
                )
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
            if self.page_cache_probe:
                page_cache += (
                    sample.classify(count)
                    if sample is not None
                    else PageCacheReadClassification.unavailable(count)
                )
            consumed = count
            while pending and consumed >= len(pending[0]):
                consumed -= len(pending[0])
                pending.pop(0)
            if pending and consumed:
                pending[0] = pending[0][consumed:]
            position += count
        if self.page_cache_probe:
            with self._lock:
                self.metrics.page_cache_probe_calls += page_cache.probe_calls
                self.metrics.page_cache_probe_failures += (
                    page_cache.probe_failures
                )
                self.metrics.page_cache_classified_bytes += (
                    page_cache.classified_bytes
                )
                self.metrics.page_cache_resident_bytes_before_read += (
                    page_cache.resident_bytes
                )
                self.metrics.page_cache_nonresident_bytes_before_read += (
                    page_cache.nonresident_bytes
                )
                self.metrics.page_cache_unclassified_bytes += (
                    page_cache.unclassified_bytes
                )
        return page_cache

    def _validate_direct_read(
        self,
        offset: int,
        views: list[memoryview],
    ) -> None:
        alignment = self.direct_io_alignment
        if alignment < 1:
            raise RuntimeError("cache-bypass read has no direct-I/O alignment")
        if offset % alignment:
            raise RuntimeError(
                f"cache-bypass file offset {offset} is not {alignment}-byte aligned"
            )
        for view in views:
            address = ctypes.addressof(ctypes.c_char.from_buffer(view))
            if address % alignment:
                raise RuntimeError(
                    "cache-bypass destination address is not "
                    f"{alignment}-byte aligned"
                )
            if len(view) % alignment:
                raise RuntimeError(
                    "cache-bypass iovec length is not "
                    f"{alignment}-byte aligned"
                )


class _SpeculativeExpertPrefetch:
    """One exact prefetch transaction backed by the cache's scratch pool."""

    def __init__(
        self,
        cache: ExpertCache,
        requested: tuple[tuple[int, int], ...],
        resident: frozenset[tuple[int, int]],
        scratch_slots: dict[tuple[int, int], int],
    ) -> None:
        self._cache = cache
        self._requested = requested
        self._resident = resident
        self._scratch_slots = scratch_slots
        self._futures: dict[tuple[int, int], Future[_SpeculativeRead]] = {}
        self._reads: dict[tuple[int, int], _SpeculativeRead] = {}
        self._wait_seconds = 0.0
        self._read_time_recorded = False
        self._closed = False

    def start(self) -> None:
        cache = self._cache
        pool = cache._speculative_pool
        if pool is None:
            raise RuntimeError("speculative scratch is not configured")
        with cache._lock:
            cache.metrics.speculative_prefetch_rounds += 1
            cache.metrics.speculative_prefetch_requested_experts += len(
                self._requested
            )
            cache.metrics.speculative_prefetch_cache_resident_experts += len(
                self._resident
            )
        for (layer, expert), slot in self._scratch_slots.items():
            self._futures[(layer, expert)] = cache._executor.submit(
                cache._read_expert_into_pool,
                pool,
                layer,
                expert,
                slot,
            )

    def covers_layer(self, layer: int) -> bool:
        return any(key[0] == layer for key in self._requested)

    def has_scratch(self, layer: int, expert: int) -> bool:
        return (layer, expert) in self._scratch_slots

    def scratch_weights(
        self,
        layer: int,
        experts: list[int],
    ) -> tuple[ExpertWeights, ...]:
        if self._closed:
            raise RuntimeError("speculative prefetch is closed")
        keys = [(layer, expert) for expert in experts]
        if any(key not in self._scratch_slots for key in keys):
            raise RuntimeError("requested expert is not in speculative scratch")
        wait_started = time.perf_counter()
        self._finish_reads(keys)
        waited = time.perf_counter() - wait_started
        self._wait_seconds += waited
        cache = self._cache
        pool = cache._speculative_pool
        if pool is None:
            raise RuntimeError("speculative scratch is not configured")
        pack_started = time.perf_counter()
        weights = pool.select_individual(
            [self._scratch_slots[key] for key in keys]
        )
        with cache._lock:
            cache.metrics.speculative_prefetch_wait_seconds += waited
            cache.metrics.speculative_scratch_hits += len(keys)
            cache.metrics.pack_seconds += time.perf_counter() - pack_started
        return weights

    def metrics(
        self,
        useful_keys: set[tuple[int, int]] | frozenset[tuple[int, int]],
    ) -> SpeculativePrefetchMetrics:
        read_keys = set(self._reads)
        useful = read_keys.intersection(useful_keys)
        wasted = read_keys.difference(useful_keys)
        read_seconds = 0.0
        if self._reads:
            read_seconds = max(read.finished for read in self._reads.values()) - min(
                read.started for read in self._reads.values()
            )
        blob_size = self._cache.model.expert_blob_size

        def page_bytes(
            keys: set[tuple[int, int]],
            name: str,
        ) -> int:
            return sum(
                int(getattr(self._reads[key].page_cache, name)) for key in keys
            )

        return SpeculativePrefetchMetrics(
            requested_experts=len(self._requested),
            cache_resident_experts=len(self._resident),
            experts_read=len(read_keys),
            bytes_read=len(read_keys) * blob_size,
            useful_bytes=len(useful) * blob_size,
            wasted_bytes=len(wasted) * blob_size,
            read_seconds=read_seconds,
            wait_seconds=self._wait_seconds,
            page_cache_classified_bytes=page_bytes(
                read_keys, "classified_bytes"
            ),
            page_cache_resident_bytes_before_read=page_bytes(
                read_keys, "resident_bytes"
            ),
            page_cache_nonresident_bytes_before_read=page_bytes(
                read_keys, "nonresident_bytes"
            ),
            page_cache_unclassified_bytes=page_bytes(
                read_keys, "unclassified_bytes"
            ),
            useful_page_cache_resident_bytes_before_read=page_bytes(
                useful, "resident_bytes"
            ),
            useful_page_cache_nonresident_bytes_before_read=page_bytes(
                useful, "nonresident_bytes"
            ),
            useful_page_cache_unclassified_bytes=page_bytes(
                useful, "unclassified_bytes"
            ),
            wasted_page_cache_resident_bytes_before_read=page_bytes(
                wasted, "resident_bytes"
            ),
            wasted_page_cache_nonresident_bytes_before_read=page_bytes(
                wasted, "nonresident_bytes"
            ),
            wasted_page_cache_unclassified_bytes=page_bytes(
                wasted, "unclassified_bytes"
            ),
        )

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self._finish_reads(list(self._futures))
        except BaseException as caught:
            error = caught
            for future in self._futures.values():
                future.cancel()
            for future in self._futures.values():
                try:
                    future.result()
                except BaseException:
                    pass
        finally:
            cache = self._cache
            pool = cache._speculative_pool
            with cache._lock:
                self._record_read_time_locked()
                cache._speculative_pinned_keys.difference_update(self._resident)
                if cache._active_speculative_prefetch is self:
                    cache._active_speculative_prefetch = None
            if pool is not None:
                for slot in self._scratch_slots.values():
                    pool.mark_empty(slot)
            self._closed = True
        if error is not None:
            raise error

    def _finish_reads(self, keys: list[tuple[int, int]]) -> None:
        cache = self._cache
        pool = cache._speculative_pool
        if pool is None:
            raise RuntimeError("speculative scratch is not configured")
        completed: list[tuple[tuple[int, int], _SpeculativeRead]] = []
        for key in keys:
            if key in self._reads:
                continue
            read = self._futures[key].result()
            pool.mark_loaded(self._scratch_slots[key])
            self._reads[key] = read
            completed.append((key, read))
        if completed:
            bytes_read = len(completed) * cache.model.expert_blob_size
            page_cache = PageCacheReadClassification()
            for _, read in completed:
                page_cache += read.page_cache
            with cache._lock:
                cache.metrics.bytes_read += bytes_read
                cache.metrics.speculative_prefetch_experts_read += len(completed)
                cache.metrics.speculative_prefetch_bytes_read += bytes_read
                metrics = cache.metrics
                metrics.speculative_prefetch_page_cache_classified_bytes += (
                    page_cache.classified_bytes
                )
                metrics.speculative_prefetch_page_cache_resident_bytes_before_read += (
                    page_cache.resident_bytes
                )
                metrics.speculative_prefetch_page_cache_nonresident_bytes_before_read += (
                    page_cache.nonresident_bytes
                )
                metrics.speculative_prefetch_page_cache_unclassified_bytes += (
                    page_cache.unclassified_bytes
                )

    def _record_read_time_locked(self) -> None:
        if self._read_time_recorded or not self._reads:
            return
        elapsed = max(read.finished for read in self._reads.values()) - min(
            read.started for read in self._reads.values()
        )
        self._cache.metrics.read_seconds += elapsed
        self._cache.metrics.speculative_prefetch_read_seconds += elapsed
        self._read_time_recorded = True
