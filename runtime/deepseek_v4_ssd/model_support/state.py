"""Shared MLX cache serialization; model packages choose whether to expose it."""
from __future__ import annotations
import copy
from typing import Any
import mlx.core as mx
from mlx_lm.models.cache import CacheList, make_prompt_cache
from ..model import _cache_arrays
from ..fp8_cache import MXFP8PoolingCache
from ..qwen_quantized_cache import QSAQuantizedCache

class _RawEvalCacheList(CacheList):
    @property
    def state(self):
        return _cache_arrays([self])

    @state.setter
    def state(self, value):
        for cache, saved in zip(self.caches, value):
            cache.state = saved


def _make_prompt_cache(model: Any):
    cache = make_prompt_cache(model)
    for layer_cache in cache:
        if isinstance(layer_cache, CacheList):
            layer_cache.__class__ = _RawEvalCacheList
    return cache


def _clone_cache_state(value: Any) -> Any:
    if isinstance(value, mx.array):
        return value + mx.zeros((), value.dtype)
    if isinstance(value, tuple):
        return tuple(_clone_cache_state(item) for item in value)
    if isinstance(value, list):
        return [_clone_cache_state(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_cache_state(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _cache_state_arrays(value: Any) -> list[mx.array]:
    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in _cache_state_arrays(item)]
    if isinstance(value, dict):
        return [array for item in value.values() for array in _cache_state_arrays(item)]
    return []


def _cache_state_nbytes(value: Any) -> int:
    return sum(int(array.nbytes) for array in _cache_state_arrays(value))


def _encode_cache_state(value: Any, arrays: dict[str, mx.array]) -> Any:
    if isinstance(value, mx.array):
        if value.size == 0:
            return {
                "empty_array": {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).rsplit(".", 1)[-1],
                }
            }
        name = f"state_{len(arrays)}"
        arrays[name] = value
        return {"array": name}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return {
            "items": [_encode_cache_state(item, arrays) for item in value],
            "tuple": isinstance(value, tuple),
        }
    if isinstance(value, dict):
        return {
            "mapping": [
                [str(key), _encode_cache_state(item, arrays)]
                for key, item in value.items()
            ]
        }
    raise TypeError(f"unsupported prompt cache state value: {type(value).__name__}")


def _decode_cache_state(value: Any, arrays: dict[str, mx.array]) -> Any:
    if isinstance(value, dict) and "empty_array" in value:
        description = value["empty_array"]
        dtype = getattr(mx, description["dtype"])
        return mx.empty(tuple(description["shape"]), dtype=dtype)
    if isinstance(value, dict) and "array" in value:
        return arrays[value["array"]]
    if isinstance(value, dict) and "items" in value:
        items = [_decode_cache_state(item, arrays) for item in value["items"]]
        return tuple(items) if value.get("tuple") else items
    if isinstance(value, dict) and "mapping" in value:
        return {
            key: _decode_cache_state(item, arrays)
            for key, item in value["mapping"]
        }
    return value


def _persistence_item(item: Any) -> dict[str, Any]:
    if isinstance(item, QSAQuantizedCache):
        return {"kind": "qsa_quantized", "value": item.persistence_state()}
    if isinstance(item, MXFP8PoolingCache):
        return {
            "kind": "mxfp8_pooling",
            "value": item.persistence_state(),
        }
    return {
        "kind": "state",
        "value": item.state,
        "meta": getattr(item, "meta_state", ""),
    }


def _persistence_cache_state(cache: Any) -> list[dict[str, Any]]:
    state = []
    for layer_cache in cache:
        if isinstance(layer_cache, CacheList):
            state.append(
                {
                    "kind": "cache_list",
                    "items": [_persistence_item(item) for item in layer_cache.caches],
                }
            )
        else:
            state.append(_persistence_item(layer_cache))
    return state


def _restore_persistence_item(target: Any, saved: dict[str, Any]) -> None:
    kind = saved.get("kind")
    if kind == "qsa_quantized":
        if not isinstance(target, QSAQuantizedCache):
            raise ValueError("QSA prompt cache format does not match")
        target.restore_persistence_state(saved["value"])
        return
    if isinstance(target, QSAQuantizedCache):
        raise ValueError("QSA prompt cache format does not match")
    if kind == "mxfp8_pooling":
        if not isinstance(target, MXFP8PoolingCache):
            raise ValueError("prompt cache type does not match the saved cache")
        target.restore_persistence_state(saved["value"])
        return
    if kind != "state":
        raise ValueError(f"unsupported prompt cache item kind: {kind}")
    target.state = saved["value"]
    meta = saved.get("meta", "")
    if meta not in (None, ""):
        target.meta_state = meta


def _restore_persistence_cache(cache: Any, state: list[dict[str, Any]]) -> None:
    if len(cache) != len(state):
        raise ValueError("prompt cache layer count does not match")
    for target, saved in zip(cache, state):
        if saved.get("kind") == "cache_list":
            if not isinstance(target, CacheList):
                raise ValueError("prompt cache structure does not match")
            items = saved["items"]
            if len(target.caches) != len(items):
                raise ValueError("prompt cache item count does not match")
            for target_item, saved_item in zip(target.caches, items):
                _restore_persistence_item(target_item, saved_item)
        else:
            _restore_persistence_item(target, saved)


make_cache = _make_prompt_cache
persistence_cache_state = _persistence_cache_state
restore_persistence_cache = _restore_persistence_cache
