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
        return _deepseek_v41_prefill(model, tokens, cache, step_size)

    def open_codec(self, root, tokenizer):
        from ..tool_codec import V41_ENCODER_SHA256, ToolCodec
        return ToolCodec.from_encoder(root / "encoding/encoding.py", V41_ENCODER_SHA256)

    def make_tool_stream_parser(self, thinking_mode):
        from ..tool_codec import DeepSeekV41ToolStreamParser
        return DeepSeekV41ToolStreamParser(thinking_mode)

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
        and raw.get("dspark") is None
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
