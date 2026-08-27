from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from deepseek_v4_ssd.manifest import MODEL_ID, REVISION


FILLER = (
    "Background reference: SSD expert routing, cache residency, verification, "
    "and deterministic decoding are being audited. "
)

CASES = (
    (
        "storage_sentence",
        "\n\nWrite one short sentence about storage.",
    ),
    (
        "creative_metaphor",
        "\n\nWrite one original seven-word metaphor about memory. Output only the metaphor.",
    ),
    (
        "balanced_choice",
        "\n\nChoose exactly one of alpha or beta. Both choices are equally valid. "
        "Reply with only the chosen word.\nAnswer:",
    ),
    (
        "multilingual_choice",
        "\n\nWrite one short sentence, choosing freely between English, Traditional "
        "Chinese, Japanese, and Spanish. Output only the sentence.\nAnswer:",
    ),
    (
        "random_hex",
        "\n\nReturn exactly 32 different eight-digit hexadecimal strings, one per "
        "line, with no explanation.\n",
    ),
)


def encode_prompt(tokenizer, text: str) -> list[int]:
    add_special_tokens = tokenizer.bos_token is None or not text.startswith(
        tokenizer.bos_token
    )
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def token_sha256(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def build_prompt_with_suffix(
    tokenizer,
    filler: str,
    suffix: str,
    target_tokens: int,
) -> tuple[str, list[int], str]:
    source = filler
    source_tokens = list(tokenizer.encode(source, add_special_tokens=False))
    while len(source_tokens) < target_tokens + 256:
        source += filler
        source_tokens = list(tokenizer.encode(source, add_special_tokens=False))

    suffix_tokens = list(tokenizer.encode(suffix, add_special_tokens=False))
    estimated_body_tokens = max(1, target_tokens - len(suffix_tokens) - 1)
    separator_padding = (
        "",
        " ",
        "  ",
        "   ",
        "\n",
        "\n ",
        " \n",
        "\n\n",
        "\n\n ",
        " \n\n",
        "\t",
    )
    offsets = sorted(range(-64, 65), key=lambda value: (abs(value), value))
    for offset in offsets:
        body_tokens = estimated_body_tokens + offset
        if body_tokens < 1 or body_tokens > len(source_tokens):
            continue
        prefix = tokenizer.decode(
            source_tokens[:body_tokens],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for padding in separator_padding:
            prompt = prefix + padding + suffix
            tokens = encode_prompt(tokenizer, prompt)
            if len(tokens) == target_tokens:
                if not prompt.endswith(suffix):
                    raise RuntimeError("constructed prompt lost its requested suffix")
                return prompt, tokens, padding
    raise RuntimeError(
        f"cannot construct a {target_tokens}-token prompt ending with {suffix!r}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare exact prompts for adaptive-block workload discovery"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=4_096)
    arguments = parser.parse_args()
    if arguments.tokens < 256:
        parser.error("--tokens must be at least 256")

    model = Path(arguments.model).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model / "tokenizer",
        trust_remote_code=True,
    )

    prompts = []
    for name, suffix in CASES:
        text, tokens, padding = build_prompt_with_suffix(
            tokenizer,
            FILLER,
            suffix,
            arguments.tokens,
        )
        path = output / f"{name}-{arguments.tokens}.txt"
        path.write_text(text, encoding="utf-8")
        prompts.append(
            {
                "name": name,
                "target_tokens": arguments.tokens,
                "file": path.name,
                "suffix": suffix,
                "separator_padding": padding,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_token_sha256": token_sha256(tokens),
            }
        )

    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "method": (
            "decode an exact filler token prefix, append a fixed instruction "
            "suffix, then verify runtime encoding"
        ),
        "filler": FILLER,
        "prompts": prompts,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
