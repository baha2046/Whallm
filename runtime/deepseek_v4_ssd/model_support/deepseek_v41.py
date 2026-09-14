from __future__ import annotations

import mlx.core as mx
from ..cancellation import check_cancelled
from ..expert_layout import FusedMXFP4Layout
from .base import ModelSupport


class DeepSeekV41Support(ModelSupport):
    expert_layout = FusedMXFP4Layout()

    @property
    def option_error_name(self):
        return "DeepSeek V4.1"

    def manifest_contract(self, raw):
        return _deepseek_v41_contract(raw)

    def load(self, installed, config, raw_config, weights, read_limiter):
        from ..deepseek_v41_ssd import load
        return load(installed, config, raw_config, weights, read_limiter)

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        if not getattr(config, "v41_layer_major_prefill", False):
            return _deepseek_v41_prefill(model, tokens, cache, step_size)
        return _deepseek_v41_layer_major_prefill(
            model, tokens, cache, step_size, expert_cache, config)

    def open_codec(self, root, tokenizer):
        from ..tool_codec import V41_ENCODER_SHA256, ToolCodec
        return ToolCodec.from_encoder(root / "encoding/encoding.py", V41_ENCODER_SHA256)

    def make_tool_stream_parser(self, thinking_mode):
        from ..tool_codec import DeepSeekV41ToolStreamParser
        return DeepSeekV41ToolStreamParser(thinking_mode)

    @staticmethod
    def _model_cache(cache):
        from ..deepseek_v41_ssd import DeepSeekV41PromptCache
        if len(cache) != 1 or not isinstance(cache[0], DeepSeekV41PromptCache):
            raise ValueError("DeepSeek V4.1 requires its singleton model cache")
        return cache[0]

    def clone_cache(self, cache):
        from copy import copy
        source = self._model_cache(cache)
        target = copy(source)
        target.restore_persistence_state(source.persistence_state())
        return [target]

    def snapshot_cache(self, cache):
        return [self._model_cache(cache).persistence_state()]

    def restore_cache(self, cache, state):
        target = self._model_cache(cache)
        if not isinstance(state, list) or len(state) != 1:
            raise ValueError("DeepSeek V4.1 requires one saved model cache")
        target.restore_persistence_state(state[0])

def _deepseek_v41_contract(raw: dict) -> dict:
    from ..manifest import (DEEPSEEK_V41_MODEL_ID, DEEPSEEK_V41_REVISION, DEEPSEEK_V41_EXPERT_REGIONS)
    layer_count = 40
    expert_count = 384
    expert_blob_size = 18_800_640
    expected = (
        raw.get("modelKind") == "deepseek-v4.1"
        and raw.get("modelID") == DEEPSEEK_V41_MODEL_ID
        and raw.get("revision") == DEEPSEEK_V41_REVISION
        and raw.get("layerCount") == layer_count
        and raw.get("expertCount") == expert_count
        and raw.get("selectedExpertCount") == 6
        and raw.get("expertBlobSize") == expert_blob_size
        and raw.get("maximumContext") == 1_048_576
        and raw.get("mtp") is None
        and raw.get("ngram") is None
        and raw.get("expertQuantization") is None
    )
    if not expected:
        raise ValueError("installed model does not match the pinned V4.1 contract")
    tables = (
        (1, 384_006_168),
        (14, 384_016_682),
    )
    expected_engram = {
        "tables": [
            {
                "layer": layer,
                "weightFile": f"engram/layer_{layer:02d}.weight.bin",
                "scaleFile": f"engram/layer_{layer:02d}.scale.bin",
                "rows": rows,
                "dimension": 256,
                "blockSize": 32,
            }
            for layer, rows in tables
        ]
    }
    if raw.get("engram") != expected_engram:
        raise ValueError("installed model has an invalid V4.1 engram contract")
    required = {
        "common.bin",
        "config.json",
        "encoding/encoding.py",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        *(f"experts/layer_{layer:02d}.bin" for layer in range(layer_count)),
        *(f"engram/layer_{layer:02d}.weight.bin" for layer, _ in tables),
        *(f"engram/layer_{layer:02d}.scale.bin" for layer, _ in tables),
    }
    dspark = raw.get("dspark")
    if dspark is not None:
        expected_dspark = {"layerCount": 3, "blockSize": 5, "noiseTokenID": 128799,
                           "targetLayerIDs": [37, 38, 39], "markovRank": 256}
        if (not isinstance(dspark, dict) or any(dspark.get(k) != v for k, v in expected_dspark.items())
                or len(dspark.get("commonTensors", [])) != 97):
            raise ValueError("invalid V4.1 DSpark contract")
        required.update({"dspark/common.bin", "inference/config.json", *(f"dspark/experts/layer_{i:02d}.bin" for i in range(3))})
    files = {item.get("path"): item.get("size") for item in raw.get("files", [])}
    if any(
        files.get(f"experts/layer_{layer:02d}.bin")
        != expert_count * expert_blob_size
        for layer in range(layer_count)
    ):
        raise ValueError("installed V4.1 expert layer has an invalid size")
    for layer, rows in tables:
        if files.get(f"engram/layer_{layer:02d}.weight.bin") != rows * 256:
            raise ValueError("installed V4.1 engram weight table has an invalid size")
        if files.get(f"engram/layer_{layer:02d}.scale.bin") != rows * 8:
            raise ValueError("installed V4.1 engram scale table has an invalid size")
    return {
        "required": required,
        "allowed": required,
        "model_kind": "deepseek-v4.1",
        "layer_count": layer_count,
        "expert_count": expert_count,
        "selected_expert_count": 6,
        "expert_blob_size": expert_blob_size,
        "maximum_context": 1_048_576,
        "expert_regions": DEEPSEEK_V41_EXPERT_REGIONS,
        "engram": expected_engram,
    }


def _deepseek_v41_prefill(
    model,
    token_ids: list[int],
    cache,
    step_size: int,
) -> None:
    for start in range(0, len(token_ids), step_size):
        check_cancelled()
        tokens = mx.array(token_ids[start : start + step_size])[None]
        logits = model(tokens, cache=cache)
        mx.eval(logits)


def _deepseek_v41_layer_major_prefill(model, token_ids, cache, step_size,
                                      expert_cache, config):
    """Read each expert layer once while retaining per-chunk shared attention state."""
    from contextlib import nullcontext
    import numpy as np
    from ..deepseek_v41.model import SharedState
    from ..deepseek_v41.hyper_connections import make_identity_pre_mix

    if not token_ids:
        return
    core = model.model
    state = cache[0].cache
    start_pos = state.offset
    count = len(token_ids)
    state.ensure_capacity(start_pos + count)
    batched = config.batched_expert_prefill
    if batched:
        expert_cache.release_prefill_slots()
    inputs = mx.array(token_ids)[None]
    hashes = None
    if core.engram_hasher is not None:
        hashes = mx.array(core.engram_hasher(np.array(inputs, dtype=np.int64),
                                            start_pos, state.engram_ids))
    elif core.args.engram_layer_ids:
        raise RuntimeError("model has Engram layers but no token map")
    hidden = core.embed(inputs)
    hidden = mx.broadcast_to(hidden[:, :, None, :], (1, count, core.hc_mult, hidden.shape[-1]))
    chunks = list(range(0, count, step_size))
    shared_chunks = [SharedState() for _ in chunks]
    pre_mix = make_identity_pre_mix(1, count, core.hc_mult)
    hidden_start = 0
    last_source = max(core.args.kv_source_layers, default=len(core.layers) - 1)
    skipped = 0
    for layer_id, layer in enumerate(core.layers):
        check_cancelled()
        outputs, mixes = [], []
        keep_from = 0
        if config.v41_ced_prefill and layer_id > last_source:
            # Later layers share encoder KV. Their only token-local history is SWA.
            # Retain the entire dependency cone needed to rebuild every final window.
            required = (len(core.layers) - layer_id) * (core.args.window_size - 1) + 1
            keep_from = max(0, count - required) // step_size * step_size
            if layer.engram is not None:
                raise ValueError("CED tail replay cannot skip an Engram layer")
        with expert_cache.batched_layer(layer_id) if batched else nullcontext():
            for chunk_index, begin in enumerate(chunks):
                check_cancelled()
                end = min(count, begin + step_size)
                if begin < keep_from:
                    skipped += end - begin
                    continue
                h = hidden[:, begin - hidden_start:end - hidden_start]
                if layer.engram is not None:
                    h = layer.engram(h, hashes[:, begin:end, layer.engram.layer_hash_index])
                h, mix = layer(h, pre_mix[:, begin - hidden_start:end - hidden_start], start_pos + begin,
                               state, shared_chunks[chunk_index])
                if (batched and config.v41_next_layer_prefetch and begin == keep_from
                        and layer_id + 1 < len(core.layers)):
                    expert_cache.prefetch_layer(layer_id + 1)
                mx.eval(h, mix)
                outputs.append(h)
                mixes.append(mix)
        hidden = mx.concatenate(outputs, axis=1)
        hidden_start = keep_from
        pre_mix = mx.concatenate(mixes, axis=1)
        mx.eval(hidden, pre_mix)
    state.offset = start_pos + count
    model.ced_skipped_layer_tokens = skipped
