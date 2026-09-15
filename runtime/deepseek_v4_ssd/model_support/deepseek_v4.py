from __future__ import annotations

from ..expert_layout import FusedMXFP4Layout
from .base import ModelSupport
from contextlib import contextmanager


class DeepSeekV4Support(ModelSupport):
    expert_layout = FusedMXFP4Layout()

    def manifest_contract(self, raw):
        return _deepseek_contract(raw)

    def load(self, installed, config, raw_config, weights, read_limiter):
        return _load(installed, config, raw_config, weights, read_limiter)

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        from ..model import layer_major_prefill
        return layer_major_prefill(
            model, tokens, cache, step_size, expert_cache,
            getattr(config, "moe_prefill_step_size", 0),
            getattr(config, "batched_expert_prefill", True),
        )

    def open_codec(self, root, tokenizer):
        from ..tool_codec import ENCODER_SHA256, ToolCodec
        return ToolCodec.from_encoder(root / "encoding/encoding_dsv4.py", ENCODER_SHA256)

    def make_tool_stream_parser(self, thinking_mode):
        from ..tool_codec import ToolStreamParser
        return ToolStreamParser(thinking_mode)

    @contextmanager
    def approximation(self, model, mode):
        if mode == "exact":
            yield
            return
        if mode != "learned-route-drop-lowest-1":
            raise ValueError(f"unsupported approximation mode: {mode}")
        if getattr(model, "dspark", None) is not None:
            raise ValueError("approximation mode is not supported with DSpark")
        core = getattr(model, "model", model)
        layers = getattr(core, "layers", None)
        if layers is None:
            raise ValueError("installed model does not expose DeepSeek layers")
        changed = []
        hash_layers = 0
        try:
            for layer in layers:
                gate = layer.ffn.gate
                if getattr(gate, "hash", False):
                    if gate.top_k != 6:
                        raise ValueError("hash router does not match exact top-k contract")
                    hash_layers += 1
                    continue
                if gate.top_k != 6:
                    raise ValueError("learned router does not match exact top-k contract")
                gate.top_k = 5
                changed.append(gate)
            if len(changed) != 40 or hash_layers != 3:
                raise ValueError("installed model layer split does not match approximation contract")
            yield
        finally:
            for gate in changed:
                gate.top_k = 6

def _deepseek_contract(raw: dict) -> dict:
    from ..manifest import (MODEL_ID, REVISION, LAYER_COUNT, EXPERT_COUNT, SELECTED_EXPERT_COUNT, EXPERT_BLOB_SIZE, EXPERT_REGIONS)
    expected = (
        raw.get("modelID") == MODEL_ID
        and raw.get("revision") == REVISION
        and raw.get("layerCount") == LAYER_COUNT
        and raw.get("expertCount") == EXPERT_COUNT
        and raw.get("selectedExpertCount") == SELECTED_EXPERT_COUNT
        and raw.get("expertBlobSize") == EXPERT_BLOB_SIZE
    )
    if not expected:
        raise ValueError("installed model does not match the pinned model contract")
    dspark = raw.get("dspark")
    if dspark is not None and not (
        dspark.get("layerCount") == 3
        and dspark.get("blockSize") == 5
        and dspark.get("noiseTokenID") == 128_799
        and dspark.get("targetLayerIDs") == [40, 41, 42]
        and dspark.get("markovRank") == 256
    ):
        raise ValueError("installed model has an invalid DSpark contract")
    required = {
        "common.bin",
        "config.json",
        "encoding/encoding_dsv4.py",
        "tokenizer/tokenizer.json",
        *(f"experts/layer_{layer:02d}.bin" for layer in range(LAYER_COUNT)),
    }
    actual = {item.get("path") for item in raw.get("files", [])}
    allowed = set(required)
    allowed.update(actual.intersection(
        {"generation_config.json", "tokenizer/tokenizer_config.json", "inference/config.json"}
    ))
    if dspark is not None:
        dspark_files = {
            "dspark/common.bin",
            "inference/config.json",
            *(f"dspark/experts/layer_{layer:02d}.bin" for layer in range(3)),
        }
        required.update(dspark_files)
        allowed.update(dspark_files)
    return {
        "required": required,
        "allowed": allowed,
        "model_kind": "deepseek-v4",
        "layer_count": LAYER_COUNT,
        "expert_count": EXPERT_COUNT,
        "selected_expert_count": SELECTED_EXPERT_COUNT,
        "expert_blob_size": EXPERT_BLOB_SIZE,
        "maximum_context": 1_048_576,
        "expert_regions": EXPERT_REGIONS,
    }


def _load(installed_model, config, raw_config, common_weights, read_limiter):
    from ..model import (deepseek_v4, MXFP8PoolingCache, CorrectPoolingCache,
        _EmptySwitchGLU, _StreamingSwitchGLU, _correct_compressor, _correct_indexer,
        _sparse_pooled_attention, ExpertCache, _streaming_moe, _load_dspark, nn, mx)
    args = deepseek_v4.ModelArgs.from_dict(raw_config)

    deepseek_v4.SwitchGLU = _EmptySwitchGLU
    deepseek_v4.PoolingCache = (
        MXFP8PoolingCache if config.fp8_kv_cache else CorrectPoolingCache
    )
    MXFP8PoolingCache.fp4_index = config.fp4_index_cache
    deepseek_v4.Compressor.__call__ = _correct_compressor
    deepseek_v4.Indexer.__call__ = _correct_indexer
    deepseek_v4._sparse_pooled_attention = _sparse_pooled_attention
    model = deepseek_v4.Model(args)
    cache = ExpertCache(
        installed_model,
        config.slots,
        config.read_workers,
        config.prefetch_read_workers,
        route_trace_path=config.expert_route_trace,
        ready_expert_decode=config.ready_expert_decode,
        read_limiter=read_limiter,
        page_cache_probe=config.expert_page_cache_probe,
        file_cache_policy=config.expert_file_cache_policy,
        separate_prefill_io=getattr(config, "separate_prefill_io", True),
        eviction_policy=config.expert_eviction_policy,
        staged_expert_streaming=config.staged_expert_streaming,
    )
    try:
        if config.staged_expert_streaming:
            if not config.ready_expert_decode:
                raise ValueError(
                    "staged expert streaming requires ready expert decode"
                )
            if config.dspark_enabled:
                raise ValueError(
                    "staged expert streaming prototype does not support DSpark"
                )
        if getattr(config, "dspark_prompt_cache", False):
            if not config.dspark_enabled:
                raise ValueError("DSpark prompt cache requires DSpark")
            if not installed_model.has_dspark:
                raise ValueError("installed model does not contain DSpark")
        if not config.dspark_fallback_enabled and not config.dspark_enabled:
            raise ValueError("disabling DSpark fallback requires DSpark")
        if config.dspark_sequential_verification:
            if not config.dspark_enabled:
                raise ValueError("DSpark sequential verification requires DSpark")
            if not installed_model.has_dspark:
                raise ValueError("installed model does not contain DSpark")
        for layer_index, layer in enumerate(model.layers):
            layer.ffn.switch_mlp = _StreamingSwitchGLU(
                layer_index,
                cache,
                deepseek_v4.LimitedSwiGLU(args.swiglu_limit),
            )
        deepseek_v4.DeepseekV4MoE.__call__ = _streaming_moe

        weights = model.sanitize(common_weights)
        quantization = deepseek_v4.make_quantization_config(model)

        def class_predicate(path: str, module: nn.Module):
            if path in quantization:
                return quantization[path]
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization["mode"],
            class_predicate=class_predicate,
        )
        model.eval()
        model.load_weights(list(weights.items()), strict=False)
        model.dspark = None
        if config.dspark_enabled and installed_model.has_dspark:
            model.dspark = _load_dspark(
                installed_model, model, args, config, read_limiter
            )
        mx.eval(model.parameters())
        from ..ane_prefill import install_deepseek_ane_prefill
        model.ane_prefill = install_deepseek_ane_prefill(model, config.deepseek_ane_prefill, config.ane_prefill_ratio)
        return model, cache
    except Exception:
        cache.close()
        raise
