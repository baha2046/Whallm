"""Incremental state for DeepSeek-V4.1 decode and chunked prefill.

Per layer:

* **window ring** — every layer: a circular buffer of ``window_size`` KV
  entries (position p at slot p % window), values already FP8 fake-quantized;
* **compressed KV** — kv_source layers only: one FP4-fake-quantized latent per
  complete group, append-only. Consumer layers hold a *reference* to their
  source's buffer, so a source's write is immediately visible downstream;
* **compressor partial group** — ratio>1 sources: fp32 kv/score rows of the
  open group;
* **index keys** — layers that are both kv and index sources: the FP4-fake-
  quantized index-key cache, read by every index source below them.

Model-level: the engram compressed-token-id history, and the global offset.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .compressor import CompressorState
from .config import ModelArgs
from .packed_cache import PackedRows


class LayerCache:
    def __init__(self, bsz: int, args: ModelArgs, layer_id: int, max_seq_len: int,
                 dtype=mx.float32, packed_kv=False, packed_index=False):
        self.window = args.window_size
        self.ratio = args.compress_ratio(layer_id)
        self.is_kv_source = layer_id in args.kv_source_layers
        self.dtype = dtype

        self.packed_kv, self.packed_index = packed_kv, packed_index
        self.win_kv = (PackedRows((bsz, self.window, args.head_dim), 'fp8_ue8m0', dtype)
                       if packed_kv else mx.zeros((bsz, self.window, args.head_dim), dtype=dtype))

        self.comp_kv = None
        self.comp_state = None
        self.index_k = None
        if self.is_kv_source:
            n_comp = max_seq_len // self.ratio
            self.comp_kv = (PackedRows((bsz, n_comp, args.head_dim), 'fp4_e4m3', dtype)
                            if packed_kv else mx.zeros((bsz, n_comp, args.head_dim), dtype=dtype))
            if self.ratio > 1:
                self.comp_state = CompressorState(bsz, self.ratio, args.head_dim)
            if layer_id in args.index_source_layers:
                self.index_k = (PackedRows((bsz, n_comp, args.index_head_dim), 'fp4_ue8m0', dtype)
                                if packed_index else mx.zeros((bsz, n_comp, args.index_head_dim), dtype=dtype))

    # ---- window ring ----

    def window_chrono(self, pos: int) -> mx.array:
        """The cached window KV in chronological order: positions
        [pos - Wp, pos) where Wp = min(pos, window). [b, Wp, head_dim]."""
        w = self.window
        wp = min(pos, w)
        if wp == 0:
            return self.win_kv[:, :0]
        first = pos - wp
        slots = (first + mx.arange(wp)) % w
        return self.win_kv[:, slots]

    def write_window(self, pos: int, kv: mx.array):
        """Write chunk KV at positions [pos, pos+n) into the ring."""
        n = kv.shape[1]
        keep = min(n, self.window)
        tail = kv[:, n - keep:]
        slots = (pos + n - keep + mx.arange(keep)) % self.window
        if isinstance(self.win_kv, PackedRows):
            self.win_kv.write((slice(None), slots), tail)
        else:
            self.win_kv[:, slots] = tail.astype(self.dtype)


class ModelCache:
    def __init__(self, args: ModelArgs, bsz: int = 1, max_seq_len: int | None = None,
                 dtype=mx.float32, packed_kv=False, packed_index=False):
        self.args = args
        self.packed_kv, self.packed_index = packed_kv, packed_index
        self.max_seq_len = max_seq_len or min(args.max_seq_len, 4096)
        self.offset = 0
        self.layers = [LayerCache(bsz, args, i, self.max_seq_len, dtype, packed_kv, packed_index)
                       for i in range(args.n_layers)]
        self.engram_ids = (np.zeros((bsz, self.max_seq_len), dtype=np.int64)
                           if args.engram_layer_ids else None)

    def ensure_capacity(self, required: int):
        if required <= self.max_seq_len:
            return
        capacity = self.max_seq_len
        while capacity < required:
            next_capacity = min(max(capacity * 2, required), self.args.max_seq_len)
            if next_capacity <= capacity:
                raise ValueError("DeepSeek V4.1 cache exceeds the configured context")
            capacity = next_capacity
        for layer in self.layers:
            for name in ('comp_kv', 'index_k'):
                old = getattr(layer, name)
                if old is None:
                    continue
                rows = capacity // layer.ratio
                if isinstance(old, PackedRows):
                    expanded = old.grow(rows)
                else:
                    expanded = mx.zeros((old.shape[0], rows, old.shape[-1]), dtype=old.dtype)
                    expanded[:, :old.shape[1]] = old
                setattr(layer, name, expanded)
        if self.engram_ids is not None:
            expanded_ids = np.zeros((self.engram_ids.shape[0], capacity), dtype=np.int64)
            expanded_ids[:, : self.engram_ids.shape[1]] = self.engram_ids
            self.engram_ids = expanded_ids
        self.max_seq_len = capacity
