"""Research-only cross-arena MXFP4 QMV; MLX 0.32.0 arithmetic retained.

Hypothesis: one dispatch across resident buffers reduces small-batch overhead.
Mechanism source: DeepSpeed Inference, https://arxiv.org/abs/2207.00032.
Neither the paper nor source adaptation establishes numerical parity or speed.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@lru_cache(None)
def kernel(pages, down):
    names = [f"arena{i}" for i in range(pages)]
    # Canonical checkpoint layout; this prototype is intentionally shape-specific.
    k, n, offset, scale = (640, 2560, 1740800, 2560000) if down else (2560, 1280, 0, 1638400)
    select = "\n".join(f"case {i}: base = arena{i}; break;" for i in range(pages))
    impl = "fp_qmv_fast_impl" if k % 512 == 0 else "fp_qmv_impl"
    source = f"""
    uint expert = threadgroup_position_in_grid.z;
    const device uint32_t* base = arena0;
    switch (locations[2 * expert]) {{ {select} }}
    const device uint8_t* blob = reinterpret_cast<const device uint8_t*>(base)
        + ulong(locations[2 * expert + 1]) * 2611200ul;
    const device uint32_t* w = reinterpret_cast<const device uint32_t*>(blob + {offset});
    const device uint8_t* scales = blob + {scale};
    const device T* x = inp + {('expert * 640' if down else '0')};
    {impl}<T, 32, 4>(w, scales, x, out + expert * {n}, {k}, {n},
        uint3(0, threadgroup_position_in_grid.y, 0),
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
    """
    return mx.fast.metal_kernel(
        name=f"qwen_cross_arena_{pages}_{'down' if down else 'gate'}",
        input_names=["inp", "locations", *names], output_names=["out"],
        header=(Path(__file__).parent / "kernels/qwen_cross_arena_qmv.h").read_text(),
        source=source, ensure_row_contiguous=True,
    )


def cross_arena(value, pool, slots):
    if value.size != 2560 or value.dtype != mx.bfloat16:
        raise ValueError("prototype requires one BF16 2560-wide input")
    locations = [pool._location(int(slot)) for slot in slots]
    pages = sorted({page for page, _ in locations})
    page_map = {page: i for i, page in enumerate(pages)}
    loc = mx.array([[page_map[p], local] for p, local in locations], mx.uint32)
    arenas = [pool.arenas[p] for p in pages]
    count = len(slots)
    def project(x, down):
        n = 2560 if down else 1280
        return kernel(len(pages), down)(
            inputs=[x, loc, *arenas], template=[("T", value.dtype)],
            grid=(32, n // 4, count), threadgroup=(32, 2, 1),
            output_shapes=[(1, 1, count, 1, n)], output_dtypes=[value.dtype],
        )[0]
    projected = project(value, False)
    gate, up = mx.split(projected, 2, axis=-1)
    return project(nn.silu(gate) * up, True).squeeze(-2)


def install():
    """Only called explicitly by the research runner; runtime remains unchanged."""
    from deepseek_v4_ssd.qwen4_exp import StreamingExperts
    original = StreamingExperts.__call__
    def call(self, value, indices):
        if (self.cache.qwen_decode_active and value.size == 2560
                and self.cache.current_batched(self.layer) is None):
            selected = np.asarray(indices, dtype=np.int32).reshape(-1).tolist()
            self.cache.get_many(self.layer, selected)
            with self.cache._lock:
                slots = [self.cache._entries[(self.layer, e)].slot for e in selected]
                output = cross_arena(value, self.cache._pool, slots)
            self.cache.record_gather_qmm(2)
            return output
        return original(self, value, indices)
    StreamingExperts.__call__ = call
