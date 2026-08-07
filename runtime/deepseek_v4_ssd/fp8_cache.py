from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import PoolingCache


class CorrectPoolingCache(PoolingCache):
    """Keep the previous raw window required by ratio-4 overlap compression."""

    def __init__(self, ratio: int):
        super().__init__(ratio)
        self.previous_window_kv: mx.array | None = None
        self.previous_window_gate: mx.array | None = None

    def previous_window(self):
        return self.previous_window_kv, self.previous_window_gate

    def store_previous_window(self, kv: mx.array, gate: mx.array):
        self.previous_window_kv = kv[:, -1:]
        self.previous_window_gate = gate[:, -1:]

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder > 0 else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder > 0 else None
        return (
            buf_kv,
            buf_gate,
            self.pooled,
            self.previous_window_kv,
            self.previous_window_gate,
        )

    @state.setter
    def state(self, value):
        if len(value) == 3:
            buf_kv, buf_gate, pooled = value
            previous_kv = previous_gate = None
        elif len(value) == 5:
            buf_kv, buf_gate, pooled, previous_kv, previous_gate = value
        else:
            raise ValueError("pooling cache state must have 3 or 5 values")
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        self.pooled = pooled
        self.previous_window_kv = previous_kv
        self.previous_window_gate = previous_gate

    @property
    def nbytes(self):
        total = super().nbytes
        if self.previous_window_kv is not None:
            total += self.previous_window_kv.nbytes + self.previous_window_gate.nbytes
        return total


class MXFP8PoolingCache(CorrectPoolingCache):
    """Store completed compressed-attention cache chunks as MXFP8."""

    chunk_size = 64

    def __init__(self, ratio: int):
        super().__init__(ratio)
        self._chunks: list[tuple[mx.array, mx.array]] = []
        self._packed_cache: tuple[mx.array, mx.array] | None = None
        self._pending: mx.array | None = None
        self._length = 0
        self._last_shape: tuple[int, int] | None = None
        self.pooled = None

    @property
    def offset(self):
        return self._length

    @property
    def shape(self):
        batch, width = self._last_shape or (1, 0)
        return batch, self._length, width

    def __getitem__(self, key):
        batch, width = self._last_shape
        dtype = self._pending.dtype if self._pending is not None else mx.bfloat16
        return self.fetch(batch, width, dtype)[key]

    def update_and_fetch(self, px: mx.array):
        self.update(px)
        return self.fetch(px.shape[0], px.shape[-1], px.dtype)

    def update(self, px: mx.array) -> None:
        if px.shape[1] > 0:
            self._last_shape = (px.shape[0], px.shape[-1])
            self._length += px.shape[1]
            self._pending = (
                px
                if self._pending is None
                else mx.concatenate([self._pending, px], axis=1)
            )
            ready = self._pending.shape[1] // self.chunk_size * self.chunk_size
            if ready:
                self._packed_cache = None
                completed = self._pending[:, :ready]
                self._pending = self._pending[:, ready:] if ready < self._pending.shape[1] else None
                for start in range(0, ready, self.chunk_size):
                    self._chunks.append(
                        mx.quantize(
                            completed[:, start : start + self.chunk_size],
                            group_size=32,
                            bits=8,
                            mode="mxfp8",
                        )
                    )

    def quantized_matmul(self, query: mx.array) -> mx.array:
        scores = []
        packed = self._packed()
        if packed is not None:
            scores.append(
                mx.quantized_matmul(
                    query,
                    *packed,
                    transpose=True,
                    group_size=32,
                    bits=8,
                    mode="mxfp8",
                )
            )
        if self._pending is not None:
            scores.append(query @ self._pending[:, None].swapaxes(-1, -2).astype(query.dtype))
        if not scores:
            return mx.zeros((*query.shape[:-1], 0), dtype=query.dtype)
        return scores[0] if len(scores) == 1 else mx.concatenate(scores, axis=-1)

    def gather(self, indices: mx.array) -> mx.array:
        batch, length, _ = indices.shape

        def take_rows(source: mx.array, rows: mx.array) -> mx.array:
            source = mx.broadcast_to(
                source[:, None],
                (batch, length, *source.shape[1:]),
            )
            return mx.take_along_axis(
                source,
                mx.broadcast_to(
                    rows[..., None],
                    (*rows.shape, source.shape[-1]),
                ),
                axis=2,
            )

        completed = None
        completed_length = len(self._chunks) * self.chunk_size
        packed = self._packed()
        if packed is not None:
            weights, scales = packed
            rows = mx.clip(indices, 0, completed_length - 1)
            completed = mx.dequantize(
                take_rows(weights, rows),
                take_rows(scales, rows),
                group_size=32,
                bits=8,
                mode="mxfp8",
            )

        if self._pending is None:
            return completed
        pending_rows = mx.clip(
            indices - completed_length,
            0,
            self._pending.shape[1] - 1,
        )
        pending = take_rows(self._pending, pending_rows)
        if completed is None:
            return pending
        return mx.where((indices >= completed_length)[..., None], pending, completed)

    def _packed(self) -> tuple[mx.array, mx.array] | None:
        if not self._chunks:
            return None
        if self._packed_cache is None:
            weights = [weight for weight, _ in self._chunks]
            scales = [scale for _, scale in self._chunks]
            self._packed_cache = (
                weights[0] if len(weights) == 1 else mx.concatenate(weights, axis=1),
                scales[0] if len(scales) == 1 else mx.concatenate(scales, axis=1),
            )
        return self._packed_cache

    def fetch(self, batch: int, width: int, dtype):
        return self._fetch(batch, width, dtype)

    def _fetch(self, batch: int, width: int, dtype):
        arrays = [
            mx.dequantize(*chunk, group_size=32, bits=8, mode="mxfp8")
            for chunk in self._chunks
        ]
        if self._pending is not None:
            arrays.append(self._pending)
        if arrays:
            return arrays[0] if len(arrays) == 1 else mx.concatenate(arrays, axis=1)
        return mx.zeros((batch, 0, width), dtype=dtype)

    def make_mask(self, length: int = 1, offset: int = 0):
        if self._length == 0 or length == 1:
            return None
        pool_index = mx.arange(self._length)
        query_index = mx.arange(offset + 1, offset + length + 1)
        return pool_index < query_index[:, None] // self.ratio

    @property
    def state(self):
        buf_kv = self.buf_kv[:, : self.remainder] if self.remainder > 0 else None
        buf_gate = self.buf_gate[:, : self.remainder] if self.remainder > 0 else None
        pooled = None
        if self._length:
            batch, width = self._last_shape
            dtype = self._pending.dtype if self._pending is not None else mx.bfloat16
            pooled = self._fetch(batch, width, dtype)
        return (
            buf_kv,
            buf_gate,
            pooled,
            self.previous_window_kv,
            self.previous_window_gate,
        )

    @state.setter
    def state(self, value):
        if len(value) == 3:
            buf_kv, buf_gate, pooled = value
            previous_kv = previous_gate = None
        elif len(value) == 5:
            buf_kv, buf_gate, pooled, previous_kv, previous_gate = value
        else:
            raise ValueError("MXFP8 pooling cache state must have 3 or 5 values")
        self.remainder = 0
        self.buf_kv = self.buf_gate = None
        self._chunks = []
        self._packed_cache = None
        self._pending = None
        self._length = 0
        self._last_shape = None
        self.pooled = None
        self.previous_window_kv = previous_kv
        self.previous_window_gate = previous_gate
        if buf_kv is not None:
            self.accumulate_windows(buf_kv, buf_gate, 0)
        if pooled is not None:
            self.update_and_fetch(pooled)

    def is_trimmable(self):
        return self._length == 0

    def size(self):
        return self._length

    def empty(self):
        return self._length == 0 and self.remainder == 0

    @property
    def nbytes(self):
        total = sum(weight.nbytes + scale.nbytes for weight, scale in self._chunks)
        if self._pending is not None:
            total += self._pending.nbytes
        if self.buf_kv is not None:
            total += self.buf_kv.nbytes + self.buf_gate.nbytes
        if self.previous_window_kv is not None:
            total += self.previous_window_kv.nbytes + self.previous_window_gate.nbytes
        return total
