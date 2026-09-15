"""Actual expert reads and unchanged per-expert math, fixed Qwen layer replay.

No route prediction, grouped GEMM, or extra waits inside the measured layer.
The optional signposts are observer-only and excluded from speed comparisons.
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.qwen4_exp import StreamingExperts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--signpost", type=Path)
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--variant", choices=("control", "resident_batch"), default="control")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    events = []
    native = None
    if args.signpost:
        native = ctypes.CDLL(str(args.signpost.resolve()))
        native.probe_event.argtypes = [ctypes.c_char_p]
        native.probe_event.restype = ctypes.c_uint64

    def event(label):
        if native:
            before = time.perf_counter_ns()
            stamp = native.probe_event(label.encode())
            after = time.perf_counter_ns()
            events.append(dict(label=label, before_ns=before, native_ns=stamp, after_ns=after))

    model = InstalledModel.open(Path.home()/".dsmodel/qwen3.8-flash-next.dsv4")
    layer = 23
    ids = [3, 17, 44, 91, 128, 203, 277, 341, 409, 500]
    selected = [277, 3, 409, 44, 17, 500, 91, 341, 128, 203]
    mx.random.seed(915)
    value = mx.random.normal((1,1,2560)).astype(mx.bfloat16)
    scores = mx.softmax(mx.random.normal((1,1,10)))
    indices = mx.array([[selected]], mx.int32)
    mx.eval(value,scores,indices)
    # Initialize signpost once before reader threads can use it.
    event("clock-initialized")
    # Warm kernels and the selected file ranges, not the measured cache entries.
    with ExpertCache(model,slots=10,ready_expert_decode=True,eviction_policy="lru") as warm:
        weights = warm.get_many(layer, ids)
        for _ in range(5):
            outs = [StreamingExperts._one(value,w) for w in weights.individual_weights]
            mx.eval(outs)
    result = dict(evidence_kind="fixed_layer_real_weights_synthetic_hidden", formal_performance_result=False,
                  commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  device=mx.device_info(),platform=platform.platform(),pid=os.getpid(),
                  layer=layer,selected=selected,seed=915,read_workers=4,slots=10,variant=args.variant,
                  manifest_sha256=hashlib.sha256((model.root/"manifest.json").read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cache_state="Selected file ranges primed; each repetition uses a fresh fixed-capacity expert cache.",
                  limits=["Fixed layer and synthetic hidden/router scores, not full-model generation.",
                          "10 slots suffice for this layer; no cross-layer eviction in measured span.",
                          "Observer signposts perturb timings; compare only uninstrumented runs for speed.",
                          "Final eval joins all consumers before cache closure; individual slot last-use not isolated."], rows=[])
    reference = None
    original_read = ExpertCache._read_expert_into_slot
    current_tag = "setup"
    def read(cache, layer, expert, slot):
        event(f"{current_tag}:read-start:{expert}")
        finished = original_read(cache, layer, expert, slot)
        event(f"{current_tag}:read-done:{expert}")
        return finished
    if native:
        ExpertCache._read_expert_into_slot = read
    for wave in range(args.waves):
        for resident_count in ([10,4,0] if wave%2==0 else [0,4,10]):
            current_tag = f"w{wave}-r{resident_count}"
            with ExpertCache(model,slots=10,read_workers=4,ready_expert_decode=True,eviction_policy="lru") as cache:
                if resident_count:
                    cache.get_many(layer,ids[:resident_count])
                mx.synchronize()
                initial_bytes=cache.metrics.bytes_read
                event(f"{current_tag}:layer-start")
                started=time.perf_counter_ns()
                outputs={}
                pending=[]
                def submit(items):
                    for eid,_ in items:
                        event(f"{current_tag}:submit-start:{eid}")
                    mx.async_eval([tensor for _,tensor in items])
                    for eid,_ in items:
                        event(f"{current_tag}:submit-end:{eid}")
                for expert,weights in cache.iter_ready(layer,selected):
                    event(f"{current_tag}:ready:{expert}")
                    output=StreamingExperts._one(value,weights)
                    if args.variant == "resident_batch" and len(outputs) < resident_count:
                        pending.append((expert,output))
                        if len(pending) == resident_count:
                            submit(pending)
                    else:
                        submit([(expert,output)])
                    outputs[expert]=output
                routed=mx.stack([outputs[e] for e in selected],axis=-2)
                output=(routed*scores[...,None].astype(routed.dtype)).sum(axis=-2)
                event(f"{current_tag}:join-start")
                mx.eval(output)
                finished=time.perf_counter_ns()
                event(f"{current_tag}:layer-done")
                digest=hashlib.sha256(memoryview(output.astype(mx.float32)).tobytes()).hexdigest()
                if reference is None:
                    reference=output
                assert bool(mx.array_equal(reference,output).item()), "ready-order changes altered output"
                result["rows"].append(dict(tag=current_tag,resident=resident_count,misses=10-resident_count,
                                           seconds=(finished-started)/1e9,output_sha256=digest,
                                           bytes_read=cache.metrics.bytes_read-initial_bytes))
    result["events"]=events
    result["medians_ms"]={str(n):1000*statistics.median(r["seconds"] for r in result["rows"] if r["resident"]==n) for n in (10,4,0)}
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(pid=result["pid"],medians_ms=result["medians_ms"],events=len(events),all_output_exact=True)),flush=True)


if __name__=="__main__":
    main()
