"""Research-only two-bank Qwen expert pipeline; production never imports this.

Workers only touch prepared memoryviews. The consumer must finish all GPU reads
before a bank is resubmitted. This first prototype uses an explicit MLX fence;
it is not a native external-event bridge or a full-model performance result.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from deepseek_v4_ssd.expert_cache import _SlotPool
from deepseek_v4_ssd.cancellation import check_cancelled


def preadv_exact(fd, views, offset):
    """Finish short reads without allocating an expert-sized staging copy."""
    pending = list(views)
    total = 0
    while pending:
        try:
            n = os.preadv(fd, pending, offset + total)
        except InterruptedError:
            continue
        if n <= 0:
            raise EOFError(f"short expert file at offset {offset + total}")
        total += n
        while pending and n >= len(pending[0]):
            n -= len(pending.pop(0))
        if pending and n:
            pending[0] = pending[0][n:]
    return total


def plan_waves(indices, expert_count, bank_experts):
    if bank_experts < 1 or expert_count < 1:
        raise ValueError("positive expert and bank counts required")
    raw = np.asarray(indices)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("integer routing indices required")
    flat = raw.reshape(-1)
    if flat.size and (flat.min() < 0 or flat.max() >= expert_count):
        raise ValueError("expert index outside manifest")
    order = np.argsort(flat, kind="stable")
    unique, starts = np.unique(flat[order], return_index=True)
    waves = []
    for start in range(0, len(unique), bank_experts):
        end = min(start + bank_experts, len(unique))
        positions = order[starts[start]:starts[end] if end < len(unique) else len(flat)]
        ids = unique[start:end]
        local = np.searchsorted(ids, flat[positions]).astype(np.uint32)
        waves.append((tuple(map(int, ids)), positions.astype(np.int32), local))
    return waves


def workspace_bound(model, tokens, top_k, bank_experts):
    """Conservative explicit array allowance, not a measured process footprint.

    Includes three full routed outputs, gathered input, projected gate/up and
    nonlinear intermediates at FP32 worst case, and CPU/GPU routing metadata.
    Common model, original input, and final output are not hidden in this count.
    MLX allocator/runtime overhead still needs a fresh-process measurement.
    """
    regions = {r.name: r for r in model.expert_regions}
    hidden = regions['down.weight'].shape[0]
    intermediate = regions['gate_up.weight'].shape[0] // 2
    assignments = tokens * top_k
    return {
        'banks_bytes': 2 * bank_experts * model.expert_blob_size,
        'temporary_array_allowance': assignments * (4 * hidden + 6 * intermediate) * 4,
        'routing_allowance': assignments * 64 + model.expert_count * 64,
        'baseline_layer_payload_bytes': model.expert_count * model.expert_blob_size,
    }


class ExpertBanks:
    def __init__(self, model, bank_experts, budget_bytes, reader=preadv_exact):
        if not model.is_qwen:
            raise ValueError('initial pipeline supports Qwen only')
        if bank_experts < 1 or 2 * bank_experts * model.expert_blob_size > budget_bytes:
            raise ValueError('two banks exceed fixed payload budget')
        self.model = model
        self.bank_experts = bank_experts
        self.reader = reader
        self.rows = []
        self.closed = False
        self._running = threading.Lock()
        bank_model = replace(model, expert_count=bank_experts)
        self.pool = _SlotPool(bank_model, 0)
        self.buffers = [mx.empty((bank_experts * model.expert_blob_size // 4,), mx.uint32) for _ in range(2)]
        mx.eval(*self.buffers)
        self.weights = [self.pool.batched(buf) for buf in self.buffers]
        mx.eval(*[v for ws in self.weights for v in vars(ws).values()])
        self.views = [[self.pool.write_views(memoryview(buf).cast('B'), i)
                       for i in range(bank_experts)] for buf in self.buffers]
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _read(self, fd, bank, ids, row):
        row['read_start'] = time.perf_counter()
        row['bytes_read'] = 0
        try:
            for position, expert in enumerate(ids):
                row['bytes_read'] += self.reader(fd, self.views[bank][position], expert * self.model.expert_blob_size)
        finally:
            row['read_complete'] = time.perf_counter()

    def run(self, layer, waves, consume, cancelled=check_cancelled):
        if self.closed:
            raise RuntimeError('pipeline is closed')
        if not 0 <= layer < self.model.layer_count:
            raise ValueError('layer outside manifest')
        for ids, _, _ in waves:
            if not ids or len(ids) > self.bank_experts or len(set(ids)) != len(ids):
                raise ValueError('invalid expert wave')
            if min(ids) < 0 or max(ids) >= self.model.expert_count:
                raise ValueError('expert outside manifest')
        if not self._running.acquire(blocking=False):
            raise RuntimeError('concurrent pipeline consumption is unsupported')
        pending = {}
        fd = None
        stream = mx.default_stream(mx.default_device())
        try:
            if self.closed:
                raise RuntimeError('pipeline is closed')
            # Keep instrumentation bounded when a caller reuses banks across
            # arbitrarily many layers/requests. Readers from the last run were
            # drained before its lock was released.
            self.rows.clear()
            cancelled()
            if not waves:
                return
            fd = os.open(self.model.root / 'experts' / f'layer_{layer:02d}.bin', os.O_RDONLY)

            def submit(i):
                cancelled()
                bank = i % 2
                row = dict(layer=layer, wave=i, bank=bank, expert_ids=list(waves[i][0]), read_submit=time.perf_counter())
                self.rows.append(row)
                pending[i] = (self.executor.submit(self._read, fd, bank, waves[i][0], row), row)

            for i in range(min(2, len(waves))):
                submit(i)
            for i, (ids, positions, local) in enumerate(waves):
                cancelled()
                future, row = pending[i]
                row['wait_start'] = time.perf_counter()
                future.result()
                row['consume_start'] = time.perf_counter()
                # consumer may enqueue lazy/asynchronous work but cannot retain
                # unmaterialized bank-dependent outputs beyond this fence.
                output = consume(self.weights[i % 2], positions, local)
                if output is not None:
                    mx.eval(output)
                mx.synchronize(stream)
                row['consumer_fence_complete'] = time.perf_counter()
                del pending[i]
                if i + 2 < len(waves):
                    submit(i + 2)
        finally:
            # On consumer/read/cancellation failure, drain before closing fd or
            # permitting a second invocation to overwrite either shared bank.
            try:
                for future, _ in pending.values():
                    try:
                        future.result()
                    except Exception:
                        pass
                mx.synchronize(stream)
            finally:
                if fd is not None:
                    os.close(fd)
                self._running.release()

    def close(self):
        with self._running:
            if not self.closed:
                self.executor.shutdown(wait=True)
                self.closed = True
                self.views.clear()
                self.weights.clear()
                self.buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def qwen_pipeline(value, indices, banks, layer):
    if value.ndim != 3 or indices.ndim != 3 or value.shape[:2] != indices.shape[:2]:
        raise ValueError('expected matching [batch,tokens,hidden] and [batch,tokens,k]')
    if value.shape[0] != 1 or not indices.shape[-1]:
        raise ValueError('batch one and nonzero top-k required')
    n, k, hidden = value.shape[1], indices.shape[-1], value.shape[-1]
    expected_hidden = next(r.shape[0] for r in banks.model.expert_regions if r.name == 'down.weight')
    if hidden != expected_hidden:
        raise ValueError('input hidden dimension differs from manifest')
    budget = workspace_bound(banks.model, n, k, banks.bank_experts)
    if sum(budget[key] for key in ('banks_bytes','temporary_array_allowance','routing_allowance')) > budget['baseline_layer_payload_bytes']:
        raise ValueError('candidate workspace exceeds replaced layer payload allowance')
    mx.eval(value, indices)
    waves = plan_waves(np.asarray(indices), banks.model.expert_count, banks.bank_experts)
    output = mx.zeros((n * k, hidden), dtype=value.dtype)
    flat_value = value.reshape(n, hidden)

    def consume(weights, positions, local):
        nonlocal output
        pos = mx.array(positions, mx.int32)
        source = mx.take(flat_value, pos // k, axis=0)[:, None, None, :]
        selected = mx.array(local, mx.uint32)[:, None]
        projected = mx.gather_qmm(source, weights.gate_up, weights.gate_up_scales,
            rhs_indices=selected, transpose=True, group_size=32, bits=4, mode='mxfp4', sorted_indices=False)
        gate, up = mx.split(projected, 2, axis=-1)
        part = mx.gather_qmm(nn.silu(gate) * up, weights.down, weights.down_scales,
            rhs_indices=selected, transpose=True, group_size=32, bits=4, mode='mxfp4', sorted_indices=False)
        output[pos] = part.reshape(len(positions), hidden)
        mx.async_eval(output)
        return output

    banks.run(layer, waves, consume)
    return output.reshape(1, n, k, hidden)
