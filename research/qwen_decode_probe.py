"""Original Decode path with wall boundaries and bounded real-input captures.

No extra eval on measured calls. Router materialization includes preceding lazy
GPU work; graph construction is not isolated GPU time. Captured calls add eval
and file I/O and are excluded along with each layer's first four single calls.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import textwrap
import time

import mlx.core as mx
from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.ane_prefill import ANEPrefillController


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cli or args.cli[0] != "--" or "--no-persistent-prompt-cache" not in args.cli:
        parser.error("ordinary CLI with persistent cache disabled must follow --")
    args.output.mkdir(exist_ok=False, parents=True)
    records, samples, ane = [], [], []
    counts = Counter()
    close = ANEPrefillController.close

    def close_probe(controller):
        ane.append(controller.snapshot())
        return close(controller)

    ANEPrefillController.close = close_probe

    def record(self, value, indices, resident, result, start, routed, loaded, built, byte_before):
        if value.size // value.shape[-1] != 1:
            return result
        ordinal = counts[self.layer]
        counts[self.layer] += 1
        row = dict(layer=self.layer, ordinal=ordinal, router_materialization_seconds=routed-start,
                   expert_acquisition_seconds=loaded-routed, graph_construction_seconds=built-loaded,
                   bytes_read=self.cache.metrics.bytes_read-byte_before, measured=ordinal >= 4)
        if ordinal == 1 and self.layer in (0, 23, 47):
            arrays = dict(value=value, indices=indices, output=result)
            for expert, slot in resident.slots.items():
                for name in ("gate_up", "gate_up_scales", "down", "down_scales"):
                    arrays[f"expert_{expert}_{name}"] = getattr(resident.individual_weights[slot], name)
            mx.eval(*arrays.values())
            path = args.output / f"expert-layer-{self.layer}.safetensors"
            mx.save_safetensors(str(path), arrays)
            samples.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                layer=self.layer, ordinal=ordinal, experts=sorted(resident.slots)))
            row["capture"] = True
        records.append(row)
        return result

    source = textwrap.dedent(inspect.getsource(qwen.StreamingExperts.__call__))
    source = source.replace("    selected = np.asarray(indices, dtype=np.int32)",
                            "    _started = _clock()\n    selected = np.asarray(indices, dtype=np.int32)\n    _routed = _clock()\n    _byte_before = self.cache.metrics.bytes_read")
    source = source.replace("    flat = selected.reshape(-1)", "    _loaded = _clock()\n    flat = selected.reshape(-1)")
    source = source.replace("    return restored.reshape(*selected.shape, -1)",
                            "    result = restored.reshape(*selected.shape, -1)\n    return _record(self, value, indices, resident, result, _started, _routed, _loaded, _clock(), _byte_before)")
    namespace = {**vars(qwen), "_clock": time.perf_counter, "_record": record}
    exec(compile(source, "<qwen-decode-observer>", "exec"), namespace)
    qwen.StreamingExperts.__call__ = namespace["__call__"]
    sys.argv = ["deepseek_v4_ssd.cli", *args.cli[1:]]
    status = "failed"
    try:
        runpy.run_module("deepseek_v4_ssd.cli", run_name="__main__")
        status = "completed"
    finally:
        totals = defaultdict(float)
        for row in records:
            if row["measured"]:
                for name in ("router_materialization_seconds", "expert_acquisition_seconds", "graph_construction_seconds", "bytes_read"):
                    totals[name] += row[name]
        result = dict(status=status, formal_performance_result=False, command=args.cli[1:],
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(), ane=ane,
                      records=records, samples=samples, totals=dict(totals),
                      measured_layer_calls=sum(r["measured"] for r in records),
                      limits=["First four single-token calls per layer excluded; initial one may be final Prefill token.",
                              "Wall boundaries partition this Python call only; router wait includes earlier lazy GPU work.",
                              "Capture adds synchronization and I/O; whole probe is diagnostic only."])
        (args.output / "profile.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
