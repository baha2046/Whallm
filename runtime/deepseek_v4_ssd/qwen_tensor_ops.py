"""Pure Qwen tensor candidate: no mutable cache, weight or RNG capture."""
import mlx.core as mx


def group_rms_norm(value, weight, eps):
    normalized = value.astype(mx.float32) * mx.rsqrt(
        mx.mean(mx.square(value.astype(mx.float32)), axis=-1, keepdims=True) + eps)
    return (normalized * weight).astype(value.dtype)


compiled_group_rms_norm = mx.compile(group_rms_norm)
