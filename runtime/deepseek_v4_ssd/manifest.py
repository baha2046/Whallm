from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
QWEN_MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
QWEN_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"

# Keep these names for callers that use the DeepSeek format 1 contract.
MODEL_ID = DEEPSEEK_MODEL_ID
REVISION = DEEPSEEK_REVISION
LAYER_COUNT = 43
EXPERT_COUNT = 256
SELECTED_EXPERT_COUNT = 6
EXPERT_BLOB_SIZE = 13_369_344
EXPERT_REGIONS = (
    ("w1.weight", "I8", (2_048, 2_048), 0, 4_194_304),
    ("w1.scale", "F8_E8M0", (2_048, 128), 4_194_304, 262_144),
    ("w2.weight", "I8", (4_096, 1_024), 4_456_448, 4_194_304),
    ("w2.scale", "F8_E8M0", (4_096, 64), 8_650_752, 262_144),
    ("w3.weight", "I8", (2_048, 2_048), 8_912_896, 4_194_304),
    ("w3.scale", "F8_E8M0", (2_048, 128), 13_107_200, 262_144),
)
QWEN_EXPERT_REGIONS = (
    ("gate_up.weight", "U32", (1_280, 320), 0, 1_638_400),
    ("gate_up.scale", "U8", (1_280, 80), 1_638_400, 102_400),
    ("down.weight", "U32", (2_560, 80), 1_740_800, 819_200),
    ("down.scale", "U8", (2_560, 20), 2_560_000, 51_200),
)
QWEN_NGRAM_HEAD_OFFSETS = (
    0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165,
    120_000_228, 140_000_297, 160_000_374, 180_000_455, 200_000_548,
    220_000_655, 240_000_802, 260_000_955, 280_001_114, 300_001_275,
)
QWEN_NGRAM_HEAD_VOCAB_SIZES = (
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
    20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
    20_000_153, 20_000_159, 20_000_161, 20_000_171,
)


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True)
class ExpertQuantization:
    mode: str
    bits: int
    group_size: int
    conversion_version: int


@dataclass(frozen=True)
class NGram:
    file: str
    dtype: str
    row_bytes: int
    shard_count: int
    shard_row_count: int
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]


@dataclass(frozen=True)
class InstalledModel:
    root: Path
    model_id: str
    revision: str
    layer_count: int
    expert_count: int
    selected_expert_count: int
    expert_blob_size: int
    common_tensors: tuple[Tensor, ...]
    expert_regions: tuple[Tensor, ...]
    model_kind: str = "deepseek-v4"
    format_version: int = 1
    maximum_context: int = 1_048_576
    expert_quantization: ExpertQuantization | None = None
    ngram: NGram | None = None
    dspark_layer_count: int = 0
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 0
    dspark_common_tensors: tuple[Tensor, ...] = ()

    @property
    def has_dspark(self) -> bool:
        return self.dspark_layer_count > 0

    @property
    def is_qwen(self) -> bool:
        return self.model_kind == "qwen3.8-flash-next"

    @classmethod
    def open(cls, root: str | Path) -> InstalledModel:
        root = Path(root).expanduser().resolve()
        with (root / "manifest.json").open("rb") as file:
            raw = json.load(file)

        if raw.get("formatVersion") == 2:
            contract = _qwen_contract(raw)
        elif raw.get("formatVersion") == 1:
            contract = _deepseek_contract(raw)
        else:
            raise ValueError("installed model has an unsupported manifest format")

        files = {item["path"]: item["size"] for item in raw.get("files", [])}
        if len(files) != len(raw.get("files", [])):
            raise ValueError("installed manifest contains duplicate file paths")
        missing = contract["required"].difference(files)
        if missing:
            raise ValueError(f"installed model is missing {sorted(missing)[0]}")
        if set(files) != contract["allowed"]:
            raise ValueError("installed manifest contains an unexpected file")
        for path, size in files.items():
            target = (root / path).resolve()
            if root not in target.parents or not target.is_file():
                raise ValueError(f"installed model has an unsafe or missing file: {path}")
            if target.stat().st_size != size:
                raise ValueError(f"installed file size does not match the manifest: {path}")

        expected_layer_size = contract["expert_count"] * contract["expert_blob_size"]
        for layer in range(contract["layer_count"]):
            path = f"experts/layer_{layer:02d}.bin"
            if files[path] != expected_layer_size:
                raise ValueError(f"installed expert layer has an invalid size: {path}")

        def tensors(key: str) -> tuple[Tensor, ...]:
            return tuple(
                Tensor(
                    name=item["name"],
                    dtype=item["dtype"],
                    shape=tuple(item["shape"]),
                    offset=item["offset"],
                    length=item["length"],
                )
                for item in raw[key]
            )

        common_tensors = tensors("commonTensors")
        expert_regions = tensors("expertRegions")
        actual_regions = tuple(
            (item.name, item.dtype, item.shape, item.offset, item.length)
            for item in expert_regions
        )
        if actual_regions != contract["expert_regions"]:
            raise ValueError("installed expert blob layout does not match the model contract")
        _validate_tensors(common_tensors, files["common.bin"], "common")

        dspark = raw.get("dspark")
        dspark_common_tensors: tuple[Tensor, ...] = ()
        if dspark is not None:
            dspark_common_tensors = tuple(
                Tensor(
                    name=item["name"],
                    dtype=item["dtype"],
                    shape=tuple(item["shape"]),
                    offset=item["offset"],
                    length=item["length"],
                )
                for item in dspark["commonTensors"]
            )
            _validate_tensors(
                dspark_common_tensors,
                files["dspark/common.bin"],
                "DSpark common",
                prefix="mtp.",
            )
            for layer in range(3):
                path = f"dspark/experts/layer_{layer:02d}.bin"
                if files[path] != expected_layer_size:
                    raise ValueError(f"installed DSpark expert layer has an invalid size: {path}")

        return cls(
            root=root,
            model_id=raw["modelID"],
            revision=raw["revision"],
            layer_count=contract["layer_count"],
            expert_count=contract["expert_count"],
            selected_expert_count=contract["selected_expert_count"],
            expert_blob_size=contract["expert_blob_size"],
            common_tensors=common_tensors,
            expert_regions=expert_regions,
            model_kind=contract["model_kind"],
            format_version=raw["formatVersion"],
            maximum_context=contract["maximum_context"],
            expert_quantization=contract.get("expert_quantization"),
            ngram=contract.get("ngram"),
            dspark_layer_count=dspark["layerCount"] if dspark is not None else 0,
            dspark_block_size=dspark["blockSize"] if dspark is not None else 0,
            dspark_noise_token_id=dspark["noiseTokenID"] if dspark is not None else 0,
            dspark_target_layer_ids=(
                tuple(dspark["targetLayerIDs"]) if dspark is not None else ()
            ),
            dspark_markov_rank=dspark["markovRank"] if dspark is not None else 0,
            dspark_common_tensors=dspark_common_tensors,
        )


def _deepseek_contract(raw: dict) -> dict:
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


def _qwen_contract(raw: dict) -> dict:
    quantization = raw.get("expertQuantization") or {}
    ngram = raw.get("ngram") or {}
    expected = (
        raw.get("modelKind") == "qwen3.8-flash-next"
        and raw.get("modelID") == QWEN_MODEL_ID
        and raw.get("revision") == QWEN_REVISION
        and raw.get("layerCount") == 48
        and raw.get("expertCount") == 512
        and raw.get("selectedExpertCount") == 10
        and raw.get("expertBlobSize") == 2_611_200
        and raw.get("maximumContext") == 262_144
        and quantization
        == {"bits": 4, "conversionVersion": 2, "groupSize": 32, "mode": "mxfp4"}
        and ngram.get("file") == "ngram.bin"
        and ngram.get("dtype") == "F8_E4M3"
        and ngram.get("rowBytes") == 160
        and ngram.get("shardCount") == 128
        and ngram.get("shardRowCount") == 2_500_012
        and tuple(ngram.get("headOffsets", [])) == QWEN_NGRAM_HEAD_OFFSETS
        and tuple(ngram.get("headVocabSizes", [])) == QWEN_NGRAM_HEAD_VOCAB_SIZES
        and raw.get("dspark") is None
    )
    if not expected:
        raise ValueError("installed model does not match the pinned Qwen model contract")
    required = {
        "common.bin",
        "ngram.bin",
        "config.json",
        "generation_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        *(f"experts/layer_{layer:02d}.bin" for layer in range(48)),
    }
    file_sizes = {item.get("path"): item.get("size") for item in raw.get("files", [])}
    if file_sizes.get("ngram.bin") != 128 * 2_500_012 * 160:
        raise ValueError("installed Qwen N-gram file has an invalid size")
    return {
        "required": required,
        "allowed": required,
        "model_kind": "qwen3.8-flash-next",
        "layer_count": 48,
        "expert_count": 512,
        "selected_expert_count": 10,
        "expert_blob_size": 2_611_200,
        "maximum_context": 262_144,
        "expert_regions": QWEN_EXPERT_REGIONS,
        "expert_quantization": ExpertQuantization("mxfp4", 4, 32, 2),
        "ngram": NGram(
            "ngram.bin",
            "F8_E4M3",
            160,
            128,
            2_500_012,
            tuple(ngram["headOffsets"]),
            tuple(ngram["headVocabSizes"]),
        ),
    }


def _validate_tensors(
    tensors: tuple[Tensor, ...],
    file_size: int,
    label: str,
    *,
    prefix: str | None = None,
) -> None:
    names = {item.name for item in tensors}
    if not tensors or len(names) != len(tensors):
        raise ValueError(f"installed {label} tensors are empty or have duplicate names")
    if any(
        (prefix is not None and not item.name.startswith(prefix))
        or item.offset < 0
        or item.length < 0
        or item.offset + item.length > file_size
        for item in tensors
    ):
        raise ValueError(f"installed {label} tensor table is invalid")
