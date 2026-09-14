"""Research-only direct resident MXFP4 short-block expert execution."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import time
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from bisect import bisect_right
from dataclasses import replace
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.expert_cache import ExpertCache, _SlotPool, QwenBatchedExperts
from deepseek_v4_ssd.cancellation import check_cancelled


# Historical implementation: removed from production on 2026-09-14.
class _QwenArenaSlotPool(_SlotPool):
    """Canonical blobs in bounded arenas; views never flatten over 2**31 words."""

    PAGE_SLOTS = 1024

    def __init__(self, model: InstalledModel, slots: int) -> None:
        if not model.is_qwen or model.expert_blob_size % 4:
            raise ValueError("Qwen arena requires aligned Qwen expert blobs")
        super().__init__(model, slots)
        self.page_slots = min(self.PAGE_SLOTS, (2**31 - 1) // (model.expert_blob_size // 4))
        if self.page_slots < 1:
            raise ValueError("Qwen expert blob exceeds MLX's view dimension limit")
        # Bound unused reservation relative to common tensors and filled pages.
        # Use 90% of canonical common bytes as a conservative planning floor,
        # and a 14% growth budget with 32-slot rounding. Cold gates remain required.
        # The first page is capped separately: one full Qwen step uses 480 slots.
        common_floor = sum(tensor.length for tensor in model.common_tensors) * 9 // 10
        self.page_starts: list[int] = []
        self.page_counts: list[int] = []
        start = 0
        while start < slots:
            limit = self.page_slots
            if start:
                budget = 14 * (common_floor + start * model.expert_blob_size) // (100 * model.expert_blob_size)
                limit = min(limit, max(1, (budget + 31) // 32 * 32))
            count = min(limit, slots - start)
            self.page_starts.append(start)
            self.page_counts.append(count)
            start += count
        self.arenas: dict[int, mx.array] = {}
        self.grouped: dict[int, QwenBatchedExperts] = {}

    def _location(self, slot: int) -> tuple[int, int]:
        page = bisect_right(self.page_starts, slot) - 1
        return page, slot - self.page_starts[page]

    def prepare(self, slots: list[int]) -> None:
        for page in sorted({self._location(slot)[0] for slot in slots}):
            if page not in self.arenas:
                count = self.page_counts[page]
                arena = mx.empty((count, self._model.expert_blob_size // 4), dtype=mx.uint32)
                mx.eval(arena)
                proxy = _SlotPool(replace(self._model, expert_count=count), 0)
                grouped = proxy.batched(arena)
                mx.eval(*vars(grouped).values())
                self.arenas[page], self.grouped[page] = arena, grouped
        created = []
        for slot in slots:
            if self._slots[slot] is None:
                page, index = self._location(slot)
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
                page, index = self._location(slot)
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
            page, index = self._location(slot)
            positions, indices = pages.setdefault(page, ([], []))
            positions.append(position)
            indices.append(index)
        return [(self.grouped[page], indices, positions) for page, (positions, indices) in pages.items()]

class BoundedArenaPool(_QwenArenaSlotPool):
    PAGE_SLOTS = 128
    MAX_UNUSED_BYTES = 127 * 2611200

    def __init__(self, model, slots):
        if model.expert_blob_size != 2611200 or not model.is_qwen or not 1 <= slots <= 1152:
            raise ValueError('resident prototype requires canonical Qwen and <=1152 slots')
        expected = [('gate_up.weight','U32',(1280,320),0,1638400),
                    ('gate_up.scale','U8',(1280,80),1638400,102400),
                    ('down.weight','U32',(2560,80),1740800,819200),
                    ('down.scale','U8',(2560,20),2560000,51200)]
        actual = sorted((r.name,r.dtype,tuple(r.shape),r.offset,r.length) for r in model.expert_regions)
        if actual != sorted(expected):
            raise ValueError('native kernel requires exact canonical region layout')
        super().__init__(model, slots)

    def prepare(self, slots):
        if any(s < 0 or s >= len(self._slots) for s in slots):
            raise ValueError('physical slot outside pool')
        pages = set(self.arenas) | {self._location(s)[0] for s in slots}
        initialized = sum(x is not None for x in self._slots)
        initialized += sum(self._slots[s] is None for s in set(slots))
        unused = (sum(self.page_counts[p] for p in pages) - initialized) * self._model.expert_blob_size
        if unused > self.MAX_UNUSED_BYTES:
            raise ValueError('arena unused reservation exceeds bounded prefix budget')
        super().prepare(slots)


@lru_cache(None)
def kernel(pages, down):
    if not 1 <= pages <= 16:
        raise ValueError('native arena argument count exceeds prototype limit')
    names = [f'arena{i}' for i in range(pages)]
    k, n, offset, scale = (640,2560,1740800,2560000) if down else (2560,1280,0,1638400)
    select = '\n'.join(f'case {i}: base = arena{i}; break;' for i in range(pages))
    impl = 'fp_qmv_fast_impl' if k % 512 == 0 else 'fp_qmv_impl'
    source = f'''
    uint assignment = threadgroup_position_in_grid.z;
    const device uint32_t* base = arena0;
    switch (locations[3 * assignment]) {{ {select} }}
    const device uint8_t* blob = reinterpret_cast<const device uint8_t*>(base)
        + ulong(locations[3 * assignment + 1]) * 2611200ul;
    const device uint32_t* w = reinterpret_cast<const device uint32_t*>(blob + {offset});
    const device uint8_t* scales = blob + {scale};
    const device T* x = inp + {'assignment * 640' if down else 'locations[3 * assignment + 2] * 2560'};
    {impl}<T,32,4>(w,scales,x,out + assignment * {n},{k},{n},
        uint3(0,threadgroup_position_in_grid.y,0),
        simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
    '''
    return mx.fast.metal_kernel(name=f'qwen_resident_block_{pages}_{down}',
        input_names=['inp','locations',*names], output_names=['out'],
        header=(Path(__file__).parent/'kernels/qwen_cross_arena_qmv.h').read_text(),
        source=source, ensure_row_contiguous=True)


def direct_project(value, pool, slots, top_k):
    if value.ndim != 3 or value.shape[0] != 1 or not 1 <= value.shape[1] <= 4 or value.shape[2] != 2560 or value.dtype != mx.bfloat16:
        raise ValueError('expected BF16 [1,1..4,2560] input')
    if not 1 <= top_k <= 10 or len(slots) != value.shape[1] * top_k:
        raise ValueError('assignment count mismatch')
    if any(s < 0 or s >= len(pool._slots) or not pool._loaded[s] for s in slots):
        raise ValueError('assignment references unloaded physical slot')
    locations = [pool._location(s) for s in slots]
    pages = sorted({p for p,_ in locations})
    mapping = {p:i for i,p in enumerate(pages)}
    loc = mx.array([[mapping[p],s,i//top_k] for i,(p,s) in enumerate(locations)],mx.uint32)
    arenas = [pool.arenas[p] for p in pages]
    def project(x,down):
        n = 2560 if down else 1280
        return kernel(len(pages),down)(inputs=[x,loc,*arenas],template=[('T',value.dtype)],
            grid=(32,n//4,len(slots)),threadgroup=(32,2,1),
            output_shapes=[(1,value.shape[1],top_k,1,n)],output_dtypes=[value.dtype])[0]
    gate,up = mx.split(project(value,False),2,axis=-1)
    return project(nn.silu(gate)*up,True).squeeze(-2)


def resident_experts(value, indices, expert_cache, layer, stats=None, *, preserve_frequency=False):
    if not isinstance(expert_cache._pool,BoundedArenaPool):
        raise ValueError('bounded arena pool required')
    check_cancelled()
    mx.eval(value,indices)
    raw = np.asarray(indices)
    if raw.ndim != 3 or raw.shape[:2] != value.shape[:2] or raw.dtype.kind not in 'iu':
        raise ValueError('invalid route shape or dtype')
    unique = np.unique(raw)
    if len(unique) > expert_cache.slots or np.any(unique >= expert_cache.model.expert_count) or np.any(unique < 0):
        raise ValueError('expert union outside budget or model')
    started = time.perf_counter()
    # get_many already reads each unique miss once. Passing assignments retains
    # repeated uses in LFU without acquiring or storing any duplicate blobs.
    requested = raw.reshape(-1).tolist() if preserve_frequency else unique.tolist()
    expert_cache.get_many(layer,requested)
    with expert_cache._lock:
        slots = [expert_cache._entries[(layer,int(e))].slot for e in raw.reshape(-1)]
        output = direct_project(value,expert_cache._pool,slots,raw.shape[-1])
        mx.eval(output)  # Finish reads before eviction/rewrite can acquire the lock.
    expert_cache.record_gather_qmm(2)
    if stats is not None:
        stats.append(dict(layer=layer,unique_experts=len(unique),assignments=int(raw.size),
            packed_bytes=0,arena_count=len(expert_cache._pool.arenas),
            acquire_native_seconds=time.perf_counter()-started,
            frequency_preserved=preserve_frequency))
    return output


@contextmanager
def install_resident_block(*, preserve_frequency=False):
    """Fresh research process only; restore both hooks even on exception."""
    import research.qwen_short_block as block
    original = ExpertCache.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        try:
            if self.staged_expert_streaming or self.qwen_grouped_decode or self._entries:
                raise ValueError('incompatible cache mode')
            self._pool = BoundedArenaPool(self.model,self.slots)
        except BaseException:
            self.close()
            raise
    def execute(value, indices, expert_cache, layer, stats=None):
        return resident_experts(value, indices, expert_cache, layer, stats,
                                preserve_frequency=preserve_frequency)
    with patch.object(ExpertCache,'__init__',initialize), patch.object(block,'union_experts',execute):
        yield
