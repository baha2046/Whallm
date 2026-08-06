from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterator

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from .manifest import InstalledModel
from .model import RuntimeConfig, load_model

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"
THINK_START = "<think>"
THINK_END = "</think>"


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class GeneratedPiece:
    text: str
    token: int
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


def encode_chat(messages: list[dict[str, Any]], thinking_mode: str = "chat") -> str:
    """Encode the supported OpenAI message subset for DeepSeek-V4."""
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode must be 'chat' or 'thinking'")

    prompt = BOS
    for index, message in enumerate(messages):
        role = message["role"]
        content = message.get("content") or ""
        if role == "system":
            prompt += content
        elif role in {"user", "developer"}:
            prompt += USER + content
            next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
            if next_role == "assistant" or next_role is None:
                prompt += ASSISTANT
                prompt += THINK_START if thinking_mode == "thinking" else THINK_END
        elif role == "assistant":
            if thinking_mode == "thinking":
                prompt += (message.get("reasoning_content") or "") + THINK_END
            prompt += content + EOS
        else:
            raise ValueError(f"unsupported message role: {role}")
    return prompt


class ModelRuntime:
    """Keep one installed model resident and serialize all generation."""

    def __init__(self, installed: InstalledModel, config: RuntimeConfig):
        self.installed = installed
        self.config = config
        self._generation_lock = threading.Lock()
        self._generation_stream = mx.new_thread_unsafe_stream(mx.gpu)
        with mx.stream(self._generation_stream):
            self.model, self.expert_cache = load_model(installed, config)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    installed.root / "tokenizer",
                    trust_remote_code=True,
                )
            except Exception:
                self.expert_cache.close()
                raise
        # ponytail: the lock serializes graph evaluation as required by the
        # cross-thread MLX stream and remains correct for batch size 1.

    @classmethod
    def open(
        cls,
        model_path: str,
        config: RuntimeConfig = RuntimeConfig(),
    ) -> ModelRuntime:
        return cls(InstalledModel.open(model_path), config)

    @property
    def model_id(self) -> str:
        return self.installed.model_id

    def encode_chat(
        self,
        messages: list[dict[str, Any]],
        thinking_mode: str = "chat",
    ) -> str:
        return encode_chat(messages, thinking_mode)

    def stream(
        self,
        prompt: str,
        options: GenerationOptions,
    ) -> Iterator[GeneratedPiece]:
        sampler = make_sampler(
            temp=options.temperature,
            top_p=options.top_p,
        )
        with self._generation_lock:
            with mx.stream(self._generation_stream):
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=options.max_tokens,
                    sampler=sampler,
                    prefill_step_size=self.config.prefill_step_size,
                ):
                    yield GeneratedPiece(
                        text=response.text,
                        token=response.token,
                        prompt_tokens=response.prompt_tokens,
                        generation_tokens=response.generation_tokens,
                        finish_reason=response.finish_reason,
                    )

    def close(self) -> None:
        self.expert_cache.close()

    def __enter__(self) -> ModelRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
