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


class FusedMXFP4Layout:
    regions = staticmethod(_fused_slot_regions)

    @staticmethod
    def layer_regions(model):
        """Contiguous fused projections across experts, without an SSD repack.

        w1/w3 are aliases of w13; the batched matmul consumes the contiguous
        fused array, while these aliases preserve the individual tensor values.
        """
        source = _fused_slot_regions(model)
        regions, strides = {}, {}
        offset = 0
        for name in ("w13.weight", "w2.weight", "w13.scale", "w2.scale"):
            region = source[name]
            regions[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
            strides[name] = region.length
            if name.startswith("w13"):
                suffix = name.split(".")[1]
                for alias in (f"w3.{suffix}", f"w1.{suffix}"):
                    part = source[alias]
                    regions[alias] = Tensor(alias, part.dtype, part.shape,
                                           offset + part.offset - region.offset, part.length)
                    strides[alias] = region.length
            offset += model.expert_count * region.length
        if offset != model.expert_count * model.expert_blob_size:
            raise ValueError("batched expert layout does not preserve the layer size")
        return regions, strides

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
