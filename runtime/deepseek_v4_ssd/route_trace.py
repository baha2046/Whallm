from __future__ import annotations

import argparse
import json
import os
from array import array
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class RouteTraceRecorder:
    """Record compact prefill histograms and exact decode routes."""

    FORMAT = 1

    def __init__(
        self,
        layer_count: int,
        expert_count: int,
        selected_expert_count: int,
        expert_blob_size: int,
    ) -> None:
        self.layer_count = layer_count
        self.expert_count = expert_count
        self.selected_expert_count = selected_expert_count
        self.expert_blob_size = expert_blob_size
        self._phase: str | None = None
        self._prefill = np.zeros((layer_count, expert_count), dtype=np.uint64)
        self._prefill_chunks = [[] for _ in range(layer_count)]
        self._decode = [array("H") for _ in range(layer_count)]
        self._decode_misses = [array("B") for _ in range(layer_count)]

    @contextmanager
    def phase(self, name: str):
        if name not in ("prefill", "decode"):
            raise ValueError(f"invalid route trace phase: {name}")
        previous = self._phase
        self._phase = name
        try:
            yield
        finally:
            self._phase = previous

    def record(self, layer: int, selected: np.ndarray) -> None:
        if self._phase is None:
            return
        if not 0 <= layer < self.layer_count:
            raise ValueError(f"invalid route trace layer: {layer}")
        values = np.asarray(selected, dtype=np.int64).reshape(-1)
        if values.size % self.selected_expert_count:
            raise ValueError("route trace selection count is not token-aligned")
        if values.size and (values.min() < 0 or values.max() >= self.expert_count):
            raise ValueError("route trace contains an invalid expert ID")
        if self._phase == "prefill":
            histogram = np.bincount(
                values,
                minlength=self.expert_count,
            ).astype(np.uint64)
            self._prefill[layer] += histogram
            self._prefill_chunks[layer].append(histogram.tolist())
        else:
            self._decode[layer].extend(values.tolist())

    def record_residency(
        self,
        layer: int,
        selected: list[int],
        missing: list[int],
    ) -> None:
        if self._phase != "decode":
            return
        missing_set = set(missing)
        self._decode_misses[layer].extend(
            expert in missing_set for expert in selected
        )

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.FORMAT,
            "layer_count": self.layer_count,
            "expert_count": self.expert_count,
            "selected_expert_count": self.selected_expert_count,
            "expert_blob_size": self.expert_blob_size,
            "prefill_histograms": self._prefill.tolist(),
            "prefill_chunk_histograms": self._prefill_chunks,
            "decode_routes": [
                [
                    list(routes[start : start + self.selected_expert_count])
                    for start in range(0, len(routes), self.selected_expert_count)
                ]
                for routes in self._decode
            ],
            "decode_misses": [
                [
                    list(misses[start : start + self.selected_expert_count])
                    for start in range(0, len(misses), self.selected_expert_count)
                ]
                for misses in self._decode_misses
            ],
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, separators=(",", ":"))
            file.write("\n")
        os.replace(temporary, destination)


def analyze_handoff(
    trace: dict,
    hot_per_layer: tuple[int, ...] = (6, 8),
    decode_tokens: int = 256,
) -> dict:
    layer_count = int(trace["layer_count"])
    expert_count = int(trace["expert_count"])
    selected_count = int(trace["selected_expert_count"])
    blob_size = int(trace["expert_blob_size"])
    prefill = trace["prefill_histograms"]
    decode = trace["decode_routes"]
    decode_misses = trace.get("decode_misses")
    if len(prefill) != layer_count or len(decode) != layer_count:
        raise ValueError("route trace layer count does not match its data")

    policies = []
    for requested_hot in hot_per_layer:
        if requested_hot < 1:
            raise ValueError("hot expert count must be greater than zero")
        covered = 0
        total = 0
        baseline_miss_reads = 0
        recoverable_miss_reads = 0
        used_slots = 0
        per_layer = []
        for layer in range(layer_count):
            if len(prefill[layer]) != expert_count:
                raise ValueError("prefill histogram has an invalid expert count")
            ranked = sorted(
                (expert for expert, count in enumerate(prefill[layer]) if count),
                key=lambda expert: (-prefill[layer][expert], expert),
            )
            hot = set(ranked[:requested_hot])
            routes = decode[layer][:decode_tokens]
            misses = (
                decode_misses[layer][:decode_tokens]
                if decode_misses is not None
                else None
            )
            layer_total = sum(len(route) for route in routes)
            if any(len(route) != selected_count for route in routes):
                raise ValueError("decode route has an invalid selection count")
            layer_covered = sum(
                expert in hot for route in routes for expert in route
            )
            if misses is not None:
                if len(misses) != len(routes):
                    raise ValueError("decode miss data does not match decode routes")
                for route, flags in zip(routes, misses):
                    if len(flags) != selected_count:
                        raise ValueError("decode miss data has an invalid selection count")
                    missed = {
                        expert for expert, flag in zip(route, flags) if flag
                    }
                    baseline_miss_reads += len(missed)
                    recoverable_miss_reads += len(missed & hot)
            used_slots += len(hot)
            covered += layer_covered
            total += layer_total
            per_layer.append(
                {
                    "layer": layer,
                    "hot_experts": sorted(hot),
                    "coverage": layer_covered / layer_total if layer_total else 0.0,
                }
            )
        policies.append(
            {
                "hot_per_layer": requested_hot,
                "used_slots": used_slots,
                "slot_bytes": used_slots * blob_size,
                "decode_tokens": decode_tokens,
                "covered_routes": covered,
                "total_routes": total,
                "route_coverage": covered / total if total else 0.0,
                "covered_activation_bytes_upper_bound": covered * blob_size,
                "baseline_miss_reads": baseline_miss_reads,
                "baseline_miss_bytes": baseline_miss_reads * blob_size,
                "handoff_recoverable_miss_reads": recoverable_miss_reads,
                "handoff_recoverable_miss_bytes": recoverable_miss_reads
                * blob_size,
                "handoff_recoverable_miss_rate": (
                    recoverable_miss_reads / baseline_miss_reads
                    if baseline_miss_reads
                    else 0.0
                ),
                "per_layer": per_layer,
            }
        )
    return {"format": 1, "policies": policies}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate prefill hot experts against decode routes"
    )
    parser.add_argument("trace")
    parser.add_argument("--hot-per-layer", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--decode-tokens", type=int, default=256)
    arguments = parser.parse_args()
    if arguments.decode_tokens < 1:
        parser.error("--decode-tokens must be greater than zero")
    with Path(arguments.trace).open("r", encoding="utf-8") as file:
        trace = json.load(file)
    if trace.get("format") != RouteTraceRecorder.FORMAT:
        parser.error("unsupported route trace format")
    print(
        json.dumps(
            analyze_handoff(
                trace,
                tuple(arguments.hot_per_layer),
                arguments.decode_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
