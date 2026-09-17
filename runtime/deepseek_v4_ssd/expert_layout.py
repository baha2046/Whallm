"""Expert blob layouts shared by model packages; SSD ownership stays in ExpertCache."""
from __future__ import annotations
from typing import TYPE_CHECKING
from .manifest import Tensor
if TYPE_CHECKING:
    from .manifest import InstalledModel

def _fused_slot_regions(model: InstalledModel) -> dict[str, Tensor]:
    source = {region.name: region for region in model.expert_regions}
    required = {
        "w1.weight",
        "w1.scale",
        "w2.weight",
        "w2.scale",
        "w3.weight",
        "w3.scale",
    }
    if set(source) != required:
        raise ValueError("expert blob does not contain the required regions")
    w1 = source["w1.weight"]
    w3 = source["w3.weight"]
    w1_scale = source["w1.scale"]
    w3_scale = source["w3.scale"]
    if w1.dtype != w3.dtype or w1.shape[1:] != w3.shape[1:]:
        raise ValueError("w1 and w3 weights cannot use one fused projection")
    if w1_scale.dtype != w3_scale.dtype or w1_scale.shape[1:] != w3_scale.shape[1:]:
        raise ValueError("w1 and w3 scales cannot use one fused projection")

    regions = {}
    offset = 0

    def add(name: str, template: Tensor) -> None:
        nonlocal offset
        regions[name] = Tensor(
            name,
            template.dtype,
            template.shape,
            offset,
            template.length,
        )
        offset += template.length

    add("w3.weight", w3)
    add("w1.weight", w1)
    regions["w13.weight"] = Tensor(
        "w13.weight",
        w3.dtype,
        (w3.shape[0] + w1.shape[0], *w3.shape[1:]),
        0,
        w3.length + w1.length,
    )
    add("w2.weight", source["w2.weight"])
    scale_offset = offset
    add("w3.scale", w3_scale)
    add("w1.scale", w1_scale)
    regions["w13.scale"] = Tensor(
        "w13.scale",
        w3_scale.dtype,
        (w3_scale.shape[0] + w1_scale.shape[0], *w3_scale.shape[1:]),
        scale_offset,
        w3_scale.length + w1_scale.length,
    )
    add("w2.scale", source["w2.scale"])
    if offset != model.expert_blob_size:
        raise ValueError("fused expert slot size does not match the manifest")
    return regions


def _qwen_slot_regions(model: InstalledModel) -> dict[str, Tensor]:
    regions = {region.name: region for region in model.expert_regions}
    if set(regions) != {
        "gate_up.weight",
        "gate_up.scale",
        "down.weight",
        "down.scale",
    }:
        raise ValueError("Qwen expert blob does not contain the required regions")
    if sum(region.length for region in regions.values()) != model.expert_blob_size:
        raise ValueError("Qwen expert slot size does not match the manifest")
    return regions


def batched_layer_layout(
    regions: dict[str, Tensor],
    batched: tuple[str, ...],
    blob_size: int,
    expert_count: int,
) -> dict[str, tuple[int, int, int]]:
    """Place every slot region in a region-major layer buffer.

    The batched regions must partition the slot exactly. The layer buffer stores
    each batched region as ``expert_count`` consecutive copies of its slot bytes,
    so its ``[expert_count, ...]`` view is contiguous and ``gather_qmm`` reads it
    without copying the whole layer first. Returns ``name -> (base, stride,
    inner)``: expert ``e`` of ``name`` starts at ``base + e * stride + inner``.
    """
    spans = sorted((regions[name].offset, regions[name].length, name) for name in batched)
    expected = 0
    for offset, length, name in spans:
        if offset != expected:
            raise ValueError(f"batched expert region {name} does not partition the slot")
        expected += length
    if expected != blob_size:
        raise ValueError("batched expert regions do not cover the expert slot")
    bases = {}
    base = 0
    for offset, length, name in spans:
        bases[name] = (base, offset, length)
        base += expert_count * length
    layout = {}
    for name, region in regions.items():
        for owner_base, owner_offset, owner_length in bases.values():
            if (owner_offset <= region.offset
                    and region.offset + region.length <= owner_offset + owner_length):
                layout[name] = (owner_base, owner_length, region.offset - owner_offset)
                break
        else:
            raise ValueError(f"expert slot region {name} is outside every batched region")
    return layout


class FusedMXFP4Layout:
    regions = staticmethod(_fused_slot_regions)
    # Contiguous per-layer arrays: the fused w13 pair and w2, weights and scales.
    batched_regions = ("w13.weight", "w2.weight", "w13.scale", "w2.scale")

    @staticmethod
    def individual(arrays, read):
        from .expert_cache import ExpertWeights
        return tuple(ExpertWeights(
            w1=read(array, "w1.weight"), w1_scales=read(array, "w1.scale"),
            w2=read(array, "w2.weight"), w2_scales=read(array, "w2.scale"),
            w3=read(array, "w3.weight"), w3_scales=read(array, "w3.scale"),
            w13=read(array, "w13.weight"), w13_scales=read(array, "w13.scale"),
        ) for array in arrays)

    @staticmethod
    def batched(packed, read):
        from .expert_cache import BatchedExperts
        return BatchedExperts(
            w1=read(packed, "w1.weight"), w1_scales=read(packed, "w1.scale"),
            w2=read(packed, "w2.weight"), w2_scales=read(packed, "w2.scale"),
            w3=read(packed, "w3.weight"), w3_scales=read(packed, "w3.scale"),
            w13=read(packed, "w13.weight"), w13_scales=read(packed, "w13.scale"),
        )


class GateUpMXFP4Layout:
    regions = staticmethod(_qwen_slot_regions)
    batched_regions = ("gate_up.weight", "gate_up.scale", "down.weight", "down.scale")

    @staticmethod
    def individual(arrays, read):
        from .expert_cache import QwenExpertWeights
        return tuple(QwenExpertWeights(
            gate_up=read(array, "gate_up.weight"), gate_up_scales=read(array, "gate_up.scale"),
            down=read(array, "down.weight"), down_scales=read(array, "down.scale"),
        ) for array in arrays)

    @staticmethod
    def batched(packed, read):
        from .expert_cache import QwenBatchedExperts
        return QwenBatchedExperts(
            gate_up=read(packed, "gate_up.weight"), gate_up_scales=read(packed, "gate_up.scale"),
            down=read(packed, "down.weight"), down_scales=read(packed, "down.scale"),
        )
