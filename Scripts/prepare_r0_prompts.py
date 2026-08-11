from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from deepseek_v4_ssd.manifest import MODEL_ID, REVISION


SEEDS = {
    "repeated": " test",
    "code": "\ndef parse(items):\n    return [item.strip() for item in items if item]\n",
    "zh_technical": "在每一層中，runtime 讀取 routed expert，然後執行矩陣乘法。",
    "mixed_math": "Given x = 17, compute x*x + 3*x - 5 and explain each step. ",
    "tool_like": (
        'User: Inspect the runtime metrics.\nTool: {"ttft": 24.7, "unit": "s"}\n'
        "Assistant: Compare the measured phases and state the evidence limit.\n"
    ),
}


def encode_prompt(tokenizer, text: str) -> list[int]:
    add_special_tokens = tokenizer.bos_token is None or not text.startswith(
        tokenizer.bos_token
    )
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def build_prompt(tokenizer, seed: str, target_tokens: int) -> tuple[str, list[int]]:
    source = seed
    source_tokens = list(tokenizer.encode(source, add_special_tokens=False))
    while len(source_tokens) < target_tokens + 64:
        source += source
        source_tokens = list(tokenizer.encode(source, add_special_tokens=False))

    body_tokens = target_tokens
    for _ in range(8):
        prompt = tokenizer.decode(
            source_tokens[:body_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        tokens = encode_prompt(tokenizer, prompt)
        if len(tokens) == target_tokens:
            return prompt, tokens
        body_tokens += target_tokens - len(tokens)
        if body_tokens < 1 or body_tokens > len(source_tokens):
            break
    raise RuntimeError(f"cannot construct a {target_tokens}-token prompt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed R0 prompt artifacts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[4_096, 8_192, 14_363],
    )
    arguments = parser.parse_args()
    if any(tokens < 2 for tokens in arguments.tokens):
        parser.error("--tokens values must be at least 2")

    model = Path(arguments.model).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model / "tokenizer",
        trust_remote_code=True,
    )
    prompts = []
    for name, seed in SEEDS.items():
        for target_tokens in arguments.tokens:
            text, tokens = build_prompt(tokenizer, seed, target_tokens)
            path = output / f"{name}-{target_tokens}.txt"
            path.write_text(text, encoding="utf-8")
            prompts.append(
                {
                    "name": name,
                    "target_tokens": target_tokens,
                    "file": path.name,
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "prompt_token_sha256": token_sha256(tokens),
                }
            )

    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "method": "repeat seed, decode a token prefix, then verify runtime encoding",
        "seeds": SEEDS,
        "prompts": prompts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
