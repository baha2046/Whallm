"""Isolated 2x2 mode/sampling diagnosis on the frozen release runtime.

This driver deliberately uses GenerationOptions directly: the release API does
not expose all controls. No production sampler changes or automatic recovery.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import mlx.core as mx
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.model import RuntimeConfig
from issue6_reproduce import LONG


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sustained_loop(tokens):
    for length in range(1, 65):
        count = max(8, (128 + length - 1) // length)
        size = length * count
        if len(tokens) >= size and tokens[-size:] == tokens[-length:] * count:
            return dict(period_tokens=length, repeated_tokens=size, repetitions=count)
    return None


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["chat", "thinking"], required=True)
    p.add_argument("--profile", choices=["chat", "thinking"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    thinking = args.profile == "thinking"
    options = GenerationOptions(max_tokens=args.max_tokens, temperature=1.0 if thinking else 0.7,
                                top_p=0.95 if thinking else 0.8, top_k=20, min_p=0.0,
                                presence_penalty=0.0 if thinking else 1.5, repetition_penalty=1.0)
    config = RuntimeConfig(slots=8192, persistent_prompt_cache=False, prefill_step_size=128)
    write(args.output / "configuration.json", dict(mode=args.mode, sampling_profile=args.profile,
          seed=args.seed, options=asdict(options), runtime=asdict(config),
          limitations=["Release logits processors retain their original 20-token window.",
                       "Fixed seed is a same-runtime control, not cross-backend equivalence.",
                       "2048-token diagnostic cap is not full novel quality acceptance.",
                       "Raw runtime invocation isolates mode/sampling from HTTP validation.",
                       "Research stop after >=128 repetitive tokens and >=8 repeats; not a production feature."]))
    started = time.monotonic()
    runtime = ModelRuntime.open("/Users/yanun/.dsmodel/qwen3.8-flash-next.dsv4", config)
    iterator = None
    try:
        prompt = runtime.encode_chat([dict(role="user", content=LONG)], args.mode, reasoning_effort="medium")
        prompt_ids = runtime._encode_prompt(prompt)
        write(args.output / "prompt.json", dict(text=prompt, tokens=prompt_ids))
        with mx.stream(runtime._generation_stream):
            mx.random.seed(args.seed)
        iterator = runtime.stream(prompt, options)
        text, ids, finish, loop = "", [], None, None
        first = None
        with (args.output / "tokens.jsonl").open("w") as output:
            for piece in iterator:
                if first is None:
                    first = time.monotonic() - started
                text += piece.text
                output.write(json.dumps(asdict(piece), ensure_ascii=False) + "\n")
                output.flush()
                finish = piece.finish_reason
                if finish is None:
                    ids.append(piece.token)
                if len(ids) % 128 == 0 or finish:
                    print(json.dumps(dict(mode=args.mode, profile=args.profile, seed=args.seed,
                          generated_tokens=len(ids), elapsed_seconds=round(time.monotonic()-started,2))), flush=True)
                    (args.output / "raw.txt").write_text(text)
                loop = sustained_loop(ids)
                if loop or time.monotonic() - started > 720:
                    break
        iterator.close()
        mx.synchronize(runtime._generation_stream)
        turn = runtime.parse_chat(text, args.mode)
        (args.output / "raw.txt").write_text(text)
        (args.output / "content.txt").write_text(turn.content)
        (args.output / "reasoning.txt").write_text(turn.reasoning_content)
        write(args.output / "result.json", dict(finish_reason=finish,
              state="research-stopped-loop" if loop else "complete" if finish else "research-timeout",
              detected_loop=loop, generated_tokens=len(ids), prompt_tokens=len(prompt_ids),
              content_chars=len(turn.content), reasoning_chars=len(turn.reasoning_content),
              elapsed_seconds=time.monotonic()-started, first_text_seconds=first,
              raw_sha256=hashlib.sha256(text.encode()).hexdigest(),
              token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    finally:
        if iterator is not None:
            iterator.close()
        mx.synchronize(runtime._generation_stream)
        runtime.close()


if __name__ == "__main__":
    main()
