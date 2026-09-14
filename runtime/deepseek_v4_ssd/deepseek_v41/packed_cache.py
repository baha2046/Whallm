"""Native V4.1 activation formats stored as bytes, not floating-point simulations."""
from __future__ import annotations
import mlx.core as mx
from .fakequant import (_blockify, _ue8m0_scale, _e2m1_round,
                        FP4_MAX_INV, FP8_MAX_INV, _E2M1_LUT)


class PackedRows:
    def __init__(self, shape, kind, dtype=mx.bfloat16):
        self.shape = tuple(shape)
        self.kind = kind
        self.dtype = dtype
        self.block = 16 if kind == 'fp4_e4m3' else 32
        self.bits = 8 if kind == 'fp8_ue8m0' else 4
        if kind not in ('fp8_ue8m0', 'fp4_e4m3', 'fp4_ue8m0') or shape[-1] % self.block:
            raise ValueError('Invalid V4.1 packed cache format or width')
        self.payload = mx.zeros((*shape[:-1], shape[-1] * self.bits // 8), mx.uint8)
        self.scales = mx.zeros((*shape[:-1], shape[-1] // self.block), mx.uint8)

    @property
    def nbytes(self):
        return self.payload.nbytes + self.scales.nbytes

    def write(self, index, value):
        """Quantize the pre-QAT activation exactly once, using checkpoint scale rules."""
        blocks = _blockify(value.astype(mx.float32), self.block)
        amax = mx.max(mx.abs(blocks), axis=-1, keepdims=True)
        if self.kind == 'fp4_e4m3':
            encoded_scale = mx.to_fp8(mx.maximum(amax, 6.0 * 2.0**-9) / 6.0)
            scale = mx.from_fp8(encoded_scale, mx.float32)
        else:
            floor, inverse = ((1e-4, FP8_MAX_INV) if self.bits == 8
                              else (6.0 * 2.0**-126, FP4_MAX_INV))
            scale = _ue8m0_scale(mx.maximum(amax, floor), inverse)
            encoded_scale = ((scale.view(mx.uint32) >> 23) & 255).astype(mx.uint8)
        scaled = blocks / scale
        if self.bits == 8:
            payload = mx.to_fp8(mx.clip(scaled, -448.0, 448.0)).reshape(value.shape)
        else:
            rounded = _e2m1_round(mx.clip(scaled, -6.0, 6.0))
            code = mx.sum((mx.abs(rounded)[..., None] > _E2M1_LUT).astype(mx.uint8), axis=-1).astype(mx.uint8)
            code = (code | ((rounded < 0).astype(mx.uint8) << 3)).reshape(value.shape)
            payload = code[..., ::2] | (code[..., 1::2] << 4)
        self.payload[index] = payload
        self.scales[index] = encoded_scale[..., 0]

    def __getitem__(self, index):
        payload, scales = self.payload[index], self.scales[index]
        if self.kind == 'fp4_e4m3':
            scale = mx.from_fp8(scales, mx.float32)
        else:
            scale = (scales.astype(mx.uint32) << 23).view(mx.float32)
        if self.bits == 8:
            value = mx.from_fp8(payload, mx.float32)
        else:
            codes = mx.stack([payload & 15, payload >> 4], axis=-1)
            codes = codes.reshape(*payload.shape[:-1], payload.shape[-1] * 2)
            value = _E2M1_LUT[(codes & 7).astype(mx.int32)] * mx.where(codes & 8, -1., 1.)
        return (value.reshape(*value.shape[:-1], value.shape[-1] // self.block, self.block) * scale[..., None]).reshape(value.shape).astype(self.dtype)

    def grow(self, capacity):
        enlarged = PackedRows((self.shape[0], capacity, self.shape[2]), self.kind, self.dtype)
        enlarged.payload[:, :self.shape[1]] = self.payload
        enlarged.scales[:, :self.shape[1]] = self.scales
        return enlarged

    def state(self):
        return dict(format=self.kind, payload=self.payload, scales=self.scales)

    def restore(self, state):
        if not isinstance(state, dict) or set(state) != {'format', 'payload', 'scales'} or state['format'] != self.kind:
            raise ValueError('V4.1 packed cache format does not match')
        for name in ('payload', 'scales'):
            saved, expected = state[name], getattr(self, name)
            if not isinstance(saved, mx.array) or saved.shape != expected.shape or saved.dtype != mx.uint8:
                raise ValueError('Invalid V4.1 packed cache array')
        self.payload, self.scales = mx.array(state['payload']), mx.array(state['scales'])
        return self
