from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
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


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


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

    @classmethod
    def open(cls, root: str | Path) -> InstalledModel:
        root = Path(root).expanduser().resolve()
        with (root / "manifest.json").open("rb") as file:
            raw = json.load(file)

        expected = (
            raw.get("formatVersion") == 1
            and raw.get("modelID") == MODEL_ID
            and raw.get("revision") == REVISION
            and raw.get("layerCount") == LAYER_COUNT
            and raw.get("expertCount") == EXPERT_COUNT
            and raw.get("selectedExpertCount") == SELECTED_EXPERT_COUNT
            and raw.get("expertBlobSize") == EXPERT_BLOB_SIZE
        )
        if not expected:
            raise ValueError("installed model does not match the pinned model contract")

        files = {item["path"]: item["size"] for item in raw["files"]}
        required = {
            "common.bin",
            "config.json",
            "encoding/encoding_dsv4.py",
            "tokenizer/tokenizer.json",
        }
        required.update(f"experts/layer_{layer:02d}.bin" for layer in range(LAYER_COUNT))
        missing = required.difference(files)
        if missing:
            raise ValueError(f"installed model is missing {sorted(missing)[0]}")
        for path, size in files.items():
            target = (root / path).resolve()
            if root not in target.parents or not target.is_file():
                raise ValueError(f"installed model has an unsafe or missing file: {path}")
            if target.stat().st_size != size:
                raise ValueError(f"installed file size does not match the manifest: {path}")
        expected_layer_size = EXPERT_COUNT * EXPERT_BLOB_SIZE
        for layer in range(LAYER_COUNT):
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
        if actual_regions != EXPERT_REGIONS:
            raise ValueError("installed expert blob layout does not match the model contract")
        common_names = {item.name for item in common_tensors}
        if not common_tensors or len(common_names) != len(common_tensors):
            raise ValueError("installed common tensors are empty or have duplicate names")
        common_size = files["common.bin"]
        if any(
            item.offset < 0
            or item.length < 0
            or item.offset + item.length > common_size
            for item in common_tensors
        ):
            raise ValueError("installed common tensor is outside common.bin")

        return cls(
            root=root,
            model_id=raw["modelID"],
            revision=raw["revision"],
            layer_count=raw["layerCount"],
            expert_count=raw["expertCount"],
            selected_expert_count=raw["selectedExpertCount"],
            expert_blob_size=raw["expertBlobSize"],
            common_tensors=common_tensors,
            expert_regions=expert_regions,
        )
