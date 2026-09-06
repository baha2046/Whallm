"""Fixed-token replay: incremental decode versus fresh layer-major prefill.

Run with the bundled Python and an isolated runtime PYTHONPATH. No sampling,
model-weight edits, persistent prompt-cache reuse, or performance claims.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import mlx.core as mx
import numpy as np
from deepseek_v4_ssd.generation import ModelRuntime, _qwen_layer_major_prefill
from deepseek_v4_ssd.model import RuntimeConfig


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def cache_arrays(cache):
    result = {}
    def visit(value, name):
        if isinstance(value, mx.array):
            result[name] = value
        elif isinstance(value, (list, tuple)):
            for i, part in enumerate(value):
                visit(part, f"{name}.{i}")
    for i, layer in enumerate(cache):
        if hasattr(layer, "caches"):
            for j, part in enumerate(layer.caches):
                visit(part.state, f"layer{i:02d}.branch{j}")
        else:
            visit(layer.state, f"layer{i:02d}")
    return result


def stats(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    delta = a - b
    return dict(max_abs=float(mx.max(mx.abs(delta)).item()),
                rms=float(mx.sqrt(mx.mean(delta * delta)).item()),
                relative_l2=float((mx.sqrt(mx.sum(delta * delta)) /
                                   mx.maximum(mx.sqrt(mx.sum(a*a)), 1e-12)).item()))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=int, default=[0, 128, 512, 1280])
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    config = RuntimeConfig(slots=8192, prefill_step_size=128, persistent_prompt_cache=False)
    write(args.output / "environment.json", dict(platform=platform.platform(), config=asdict(config),
          input_case=str(args.case.resolve()), kind="fixed-prefix numerical diagnosis; no sampling",
          limitations=["Same runtime on both routes, not an independent full-model reference.",
                       "Forced IDs originate in a failed candidate, not a new baseline generation.",
                       "Different reduction order can cause floating-point differences; no exact-equality gate."]))
    request = json.loads((args.case / "request.json").read_text())
    forced = [r["token"] for line in (args.case / "tokens.jsonl").read_text().splitlines()
              if (r := json.loads(line))["finish"] is None]
    assert max(args.checkpoints) <= len(forced)
    runtime = ModelRuntime.open("/Users/yanun/.dsmodel/qwen3.8-flash-next.dsv4", config)
    started = time.monotonic()
    try:
        with mx.stream(runtime._generation_stream):
            prompt = runtime.encode_chat(request["messages"], "thinking", reasoning_effort="medium")
            tokens = runtime._encode_prompt(prompt)
            write(args.output / "tokens.json", dict(prompt=tokens, forced=forced[:max(args.checkpoints)],
                  prompt_text=prompt, prompt_sha256=hashlib.sha256(bytes(prompt, "utf-8")).hexdigest()))

            def prefill(ids):
                cache = runtime.model.make_cache()
                _qwen_layer_major_prefill(runtime.model, ids[:-1], cache, 128, runtime.expert_cache)
                logits = runtime.model(mx.array([ids[-1:]]), cache=cache)[:, -1]
                mx.eval(logits, *cache_arrays(cache).values())
                return cache, logits

            def save(n, cache, logits):
                mx.save_safetensors(str(args.output / f"cache-{n}.safetensors"), cache_arrays(cache))
                mx.save_safetensors(str(args.output / f"logits-{n}.safetensors"), {"logits": logits})
                print(json.dumps(dict(phase="incremental", generated_tokens=n,
                      elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)

            cache, logits = prefill(tokens)
            if 0 in args.checkpoints:
                save(0, cache, logits)
            for n in range(1, max(args.checkpoints)+1):
                logits = runtime.model(mx.array([[forced[n-1]]]), cache=cache)[:, -1]
                mx.eval(logits, *cache_arrays(cache).values())
                if n in args.checkpoints:
                    save(n, cache, logits)
                elif n % 128 == 0:
                    print(json.dumps(dict(phase="incremental", progress=n)), flush=True)
            del cache, logits
            results = []
            for n in args.checkpoints:
                print(json.dumps(dict(phase="fresh-prefill", generated_tokens=n)), flush=True)
                cache, actual = prefill(tokens + forced[:n])
                expected = mx.load(str(args.output / f"logits-{n}.safetensors"))["logits"]
                mx.save_safetensors(str(args.output / f"fresh-logits-{n}.safetensors"), {"logits":actual})
                old_state = mx.load(str(args.output / f"cache-{n}.safetensors"))
                now = cache_arrays(cache)
                state_stats = {k:stats(old_state[k], now[k]) for k in now if old_state[k].shape == now[k].shape}
                ep, ap = mx.softmax(expected.astype(mx.float32)), mx.softmax(actual.astype(mx.float32))
                ei, ai = int(mx.argmax(expected).item()), int(mx.argmax(actual).item())
                row = dict(generated_tokens=n, prefix_tokens=len(tokens)+n,
                           logits=stats(expected,actual), incremental_top1=ei, fresh_top1=ai,
                           incremental_top1_text=runtime.tokenizer.decode([ei]), fresh_top1_text=runtime.tokenizer.decode([ai]),
                           probability_l1=float(mx.sum(mx.abs(ep-ap)).item()),
                           incremental_top1_probability=float(ep[0,ei].item()),
                           fresh_top1_probability=float(ap[0,ai].item()), state=state_stats)
                results.append(row)
                write(args.output / "results.json", results)
                print(json.dumps({k:v for k,v in row.items() if k != "state"},ensure_ascii=False),flush=True)
                del cache, actual, expected, old_state, now, ep, ap
            write(args.output / "complete.json", dict(elapsed_seconds=time.monotonic()-started,
                  checkpoints=args.checkpoints, status="diagnostic complete; requires interpretation"))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
