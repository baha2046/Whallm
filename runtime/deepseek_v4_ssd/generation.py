from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from .manifest import InstalledModel
from .model import RuntimeConfig, load_model
from .tool_codec import AssistantTurn, ToolChoice, ToolCodec

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


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._time_to_first_token_seconds = 0.0
        self._decode_seconds = 0.0
        self._cache_state_eval_seconds = 0.0
        self._cache_state_eval_count = 0
        self._prompt_tokens = 0
        self._generation_tokens = 0
        self._prompt_cache_reused_tokens = 0

    def start(self, prompt_tokens: int, reused_tokens: int) -> None:
        with self._lock:
            self._time_to_first_token_seconds = 0.0
            self._decode_seconds = 0.0
            self._cache_state_eval_seconds = 0.0
            self._cache_state_eval_count = 0
            self._prompt_tokens = prompt_tokens
            self._generation_tokens = 0
            self._prompt_cache_reused_tokens = reused_tokens

    def record(
        self,
        response,
        step_seconds: float,
        cache_state_eval_seconds: float,
    ) -> None:
        with self._lock:
            if response.generation_tokens == 1:
                self._time_to_first_token_seconds = step_seconds
            else:
                self._decode_seconds += step_seconds
            self._cache_state_eval_seconds += cache_state_eval_seconds
            self._cache_state_eval_count += 1
            self._generation_tokens = response.generation_tokens

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            decode_tokens = max(0, self._generation_tokens - 1)
            return {
                "runtime_prompt_tokens": self._prompt_tokens,
                "runtime_generation_tokens": self._generation_tokens,
                "prompt_cache_reused_tokens": self._prompt_cache_reused_tokens,
                "time_to_first_token_seconds": self._time_to_first_token_seconds,
                "decode_seconds": self._decode_seconds,
                "decode_tokens_per_second": (
                    decode_tokens / self._decode_seconds if self._decode_seconds else 0.0
                ),
                "cache_state_eval_seconds": self._cache_state_eval_seconds,
                "cache_state_eval_count": self._cache_state_eval_count,
            }


class ModelRuntime:
    """Keep one installed model resident and serialize all generation."""

    def __init__(self, installed: InstalledModel, config: RuntimeConfig):
        self.installed = installed
        self.config = config
        self.metrics = RuntimeMetrics()
        self._codec: ToolCodec | None = None
        self._prompt_cache = None
        self._prompt_cache_tokens: list[int] = []
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = ToolChoice(),
    ) -> str:
        if self._codec is None:
            self._codec = ToolCodec.open(self.installed.root)
        return self._codec.encode(messages, thinking_mode, tools, tool_choice)

    def parse_chat(self, text: str, thinking_mode: str) -> AssistantTurn:
        if self._codec is None:
            self._codec = ToolCodec.open(self.installed.root)
        return self._codec.parse(text, thinking_mode)

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
                prompt_tokens = self._encode_prompt(prompt)
                reused_tokens = 0
                if (
                    self._prompt_cache is not None
                    and len(self._prompt_cache_tokens) < len(prompt_tokens)
                    and prompt_tokens[: len(self._prompt_cache_tokens)]
                    == self._prompt_cache_tokens
                ):
                    reused_tokens = len(self._prompt_cache_tokens)
                else:
                    self._prompt_cache = make_prompt_cache(self.model)
                    self._prompt_cache_tokens = []
                generation_prompt = prompt_tokens[reused_tokens:]
                self._prompt_cache_tokens.extend(generation_prompt)
                responses = iter(
                    stream_generate(
                        self.model,
                        self.tokenizer,
                        generation_prompt,
                        max_tokens=options.max_tokens,
                        sampler=sampler,
                        prompt_cache=self._prompt_cache,
                        prefill_step_size=self.config.prefill_step_size,
                    )
                )
                self.metrics.start(len(prompt_tokens), reused_tokens)
                try:
                    while True:
                        started = time.perf_counter()
                        try:
                            response = next(responses)
                        except StopIteration:
                            break
                        step_seconds = time.perf_counter() - started
                        cache_started = time.perf_counter()
                        mx.eval([cache.state for cache in self._prompt_cache])
                        cache_seconds = time.perf_counter() - cache_started
                        self.metrics.record(response, step_seconds, cache_seconds)
                        if response.finish_reason != "stop":
                            self._prompt_cache_tokens.append(int(response.token))
                        yield GeneratedPiece(
                            text=response.text,
                            token=response.token,
                            prompt_tokens=len(prompt_tokens),
                            generation_tokens=response.generation_tokens,
                            finish_reason=response.finish_reason,
                        )
                except Exception:
                    self._prompt_cache = None
                    self._prompt_cache_tokens = []
                    raise

    def _encode_prompt(self, prompt: str) -> list[int]:
        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
        )
        return list(
            self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        )

    def close(self) -> None:
        self.expert_cache.close()

    def __enter__(self) -> ModelRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
