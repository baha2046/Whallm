from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit

from .generation import GenerationOptions, GeneratedPiece, ModelRuntime, THINK_END
from .manifest import InstalledModel
from .model import RuntimeConfig, _POWER_SAVING_LIMITS_GBPS
from .tool_codec import ToolChoice, ToolStreamDelta, ToolStreamParser

MAX_REQUEST_BYTES = 1_048_576
MAX_GENERATION_TOKENS = 272_000
PUBLIC_MODEL = "deepseek-v4-flash-0731"


class APIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        param: str | None = None,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ):
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code
        self.error_type = error_type

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class ServerDefaults:
    max_tokens: int = 272_000
    temperature: float = 0.2
    top_p: float = 0.98
    top_k: int = 0


class GenerationMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generating = False
        self._tokens = 0
        self._tokens_per_second = 0.0
        self._first_token_at = 0.0

    def start(self) -> None:
        with self._lock:
            self._generating = True
            self._tokens = 0
            self._tokens_per_second = 0.0
            self._first_token_at = 0.0

    def record(self, tokens: int) -> None:
        now = time.perf_counter()
        with self._lock:
            self._tokens = tokens
            if tokens == 1 or self._first_token_at == 0:
                self._first_token_at = now
                return
            elapsed = now - self._first_token_at
            if elapsed > 0:
                self._tokens_per_second = (tokens - 1) / elapsed

    def finish(self) -> None:
        with self._lock:
            self._generating = False

    def snapshot(self) -> dict[str, bool | int | float]:
        with self._lock:
            return {
                "generating": self._generating,
                "generation_tokens": self._tokens,
                "tokens_per_second": self._tokens_per_second,
            }


class OpenAIServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 32

    def __init__(
        self,
        address: tuple[str, int],
        runtime: ModelRuntime,
        *,
        public_model: str = PUBLIC_MODEL,
        api_key: str | None = None,
        defaults: ServerDefaults = ServerDefaults(),
    ):
        self.runtime = runtime
        self.public_model = public_model
        self.api_key = api_key
        self.defaults = defaults
        self.metrics = GenerationMetrics()
        super().__init__(address, OpenAIHandler)

    def track(self, pieces: Iterator[GeneratedPiece]) -> Iterator[GeneratedPiece]:
        self.metrics.start()
        try:
            for piece in pieces:
                self.metrics.record(piece.generation_tokens)
                yield piece
        finally:
            self.metrics.finish()


class ReasoningParser:
    def __init__(self, enabled: bool):
        self.reasoning = enabled
        self.pending = ""

    def feed(self, text: str) -> tuple[str, str]:
        if not self.reasoning:
            return "", text
        combined = self.pending + text
        if THINK_END in combined:
            reasoning, content = combined.split(THINK_END, 1)
            self.pending = ""
            self.reasoning = False
            return reasoning, content
        keep = min(len(THINK_END) - 1, len(combined))
        if keep == len(combined):
            self.pending = combined
            return "", ""
        self.pending = combined[-keep:] if keep else ""
        return combined[:-keep] if keep else combined, ""

    def finish(self) -> tuple[str, str]:
        pending, self.pending = self.pending, ""
        return (pending, "") if self.reasoning else ("", pending)


class OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Whallm/1"

    @property
    def app(self) -> OpenAIServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        self._run(self._get)

    def do_POST(self) -> None:
        self._run(self._post)

    def do_PUT(self) -> None:
        self._run(self._put)

    def _run(self, action) -> None:
        try:
            action()
        except APIError as error:
            self._json(error.status, error.body())
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            sys.stderr.write(f"request failed: {error}\n")
            if self.close_connection:
                return
            self._json(
                500,
                APIError(
                    "The server could not complete the request.",
                    status=500,
                    code="internal_error",
                    error_type="server_error",
                ).body(),
            )

    def _get(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._json(
                200,
                {
                    "name": "Whallm",
                    "status": "ok",
                    "api_base": "/v1",
                },
            )
        elif path == "/favicon.ico":
            self._bytes(204, b"", "image/x-icon")
        elif path == "/healthz":
            self._json(200, {"status": "ok"})
        elif path == "/api/status":
            self._json(200, self._status())
        elif path == "/api/settings":
            self._authorize()
            self._json(200, asdict(self.app.defaults))
        elif path == "/v1/models":
            self._authorize()
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.app.public_model,
                            "object": "model",
                            "created": 0,
                            "owned_by": self.app.runtime.model_id.split("/", 1)[0],
                        }
                    ],
                },
            )
        else:
            raise APIError("Route not found.", status=404, code="not_found")

    def _post(self) -> None:
        path = urlsplit(self.path).path
        if path == "/v1/chat/completions":
            self._authorize()
            payload = self._request_json()
            self._chat(payload)
        elif path == "/v1/responses":
            self._authorize()
            payload = self._request_json()
            self._responses(payload)
        elif path == "/v1/completions":
            self._authorize()
            payload = self._request_json()
            self._completion(payload)
        else:
            raise APIError("Route not found.", status=404, code="not_found")

    def _put(self) -> None:
        if urlsplit(self.path).path != "/api/settings":
            raise APIError("Route not found.", status=404, code="not_found")
        self._authorize()
        payload = self._request_json()
        unknown = set(payload).difference(
            {"max_tokens", "temperature", "top_p", "top_k"}
        )
        if unknown:
            name = sorted(unknown)[0]
            raise APIError(f"Unknown setting: {name}.", param=name)
        options = _options(payload, self.app.defaults)
        self.app.defaults = ServerDefaults(
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
        )
        self._json(200, asdict(self.app.defaults))

    def _chat(self, payload: dict[str, Any]) -> None:
        options, stream = self._common(payload)
        tools, tool_choice = _tool_request(payload)
        messages = _messages(payload.get("messages"))
        thinking_mode, reasoning_effort = _reasoning_settings(
            payload,
            qwen=getattr(getattr(self.app.runtime, "installed", None), "is_qwen", False),
        )
        prompt = self.app.runtime.encode_chat(
            messages,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
        )
        request_id = "chatcmpl-" + uuid.uuid4().hex
        pieces = self.app.track(self.app.runtime.stream(prompt, options))
        tool_calling = bool(tools) and tool_choice.mode != "none"
        if stream:
            stream_options = payload.get("stream_options") or {}
            if tool_calling:
                self._stream_tool_chat(
                    pieces,
                    request_id,
                    thinking_mode,
                    bool(stream_options.get("include_usage", False)),
                )
                return
            self._stream_chat(
                pieces,
                request_id,
                thinking_mode == "thinking",
                bool(stream_options.get("include_usage", False)),
            )
            return

        if tool_calling:
            raw, prompt_tokens, generated, finish = _collect_raw(pieces)
            try:
                turn = self.app.runtime.parse_chat(raw, thinking_mode)
            except Exception as error:
                raise APIError(
                    "The model returned an invalid tool call.",
                    status=500,
                    code="invalid_tool_call",
                    error_type="server_error",
                ) from error
            tool_calls = _response_tool_calls(turn.tool_calls)
            message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.content or None,
            }
            if turn.reasoning_content:
                message["reasoning_content"] = turn.reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._json(
                200,
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.app.public_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool_calls else finish,
                        }
                    ],
                    "usage": _usage(prompt_tokens, generated),
                },
            )
            return

        text, reasoning, prompt_tokens, generated, finish = _collect(
            pieces,
            thinking_mode == "thinking",
        )
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning:
            message["reasoning_content"] = reasoning
        self._json(
            200,
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.app.public_model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish,
                    }
                ],
                "usage": _usage(prompt_tokens, generated),
            },
        )

    def _responses(self, payload: dict[str, Any]) -> None:
        request = dict(payload)
        if "max_output_tokens" in request:
            request["max_tokens"] = request["max_output_tokens"]
        options, stream = self._common(request)
        messages = _response_messages(payload)
        tools, tool_choice, response_tools = _response_tool_request(payload)
        thinking_mode, reasoning_effort = _reasoning_settings(
            payload,
            responses_api=True,
            qwen=getattr(getattr(self.app.runtime, "installed", None), "is_qwen", False),
        )
        _validate_response_request(payload)
        prompt = self.app.runtime.encode_chat(
            messages,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
        )
        request_id = "resp_" + uuid.uuid4().hex
        pieces = self.app.track(self.app.runtime.stream(prompt, options))
        tool_calling = bool(tools) and tool_choice.mode != "none"
        if stream:
            self._stream_response(
                pieces,
                request_id,
                payload,
                options,
                thinking_mode,
                tool_calling,
                response_tools,
            )
            return

        if tool_calling:
            raw, prompt_tokens, generated, _ = _collect_raw(pieces)
            try:
                turn = self.app.runtime.parse_chat(raw, thinking_mode)
            except Exception as error:
                raise APIError(
                    "The model returned an invalid tool call.",
                    status=500,
                    code="invalid_tool_call",
                    error_type="server_error",
                ) from error
            output = _response_output(
                turn.content,
                turn.reasoning_content,
                include_empty=False,
            )
            output.extend(_response_function_calls(turn.tool_calls, response_tools))
        else:
            text, reasoning, prompt_tokens, generated, _ = _collect(
                pieces,
                thinking_mode == "thinking",
            )
            output = _response_output(text, reasoning)
        response = _response_object(
            request_id,
            self.app.public_model,
            payload,
            options,
            output,
            _response_usage(prompt_tokens, generated),
        )
        self._json(200, response)

    def _stream_response(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        payload: dict[str, Any],
        options: GenerationOptions,
        thinking_mode: str,
        tool_calling: bool,
        response_tools: dict[str, dict[str, str]],
    ) -> None:
        self._start_sse()
        output: list[dict[str, Any]] = []
        sequence = 0
        prompt_tokens = generated = 0
        reasoning_item: dict[str, Any] | None = None
        reasoning_index = -1
        reasoning_text = ""
        message_item: dict[str, Any] | None = None
        message_index = -1
        message_text = ""
        calls: dict[int, tuple[int, dict[str, Any]]] = {}

        def send(event_type: str, **values: Any) -> None:
            nonlocal sequence
            self._sse({"type": event_type, "sequence_number": sequence, **values})
            sequence += 1

        def start_reasoning() -> None:
            nonlocal reasoning_item, reasoning_index
            if reasoning_item is not None:
                return
            reasoning_index = len(output)
            reasoning_item = {
                "id": "rs_" + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [],
                "status": "in_progress",
            }
            output.append(reasoning_item)
            send(
                "response.output_item.added",
                output_index=reasoning_index,
                item=dict(reasoning_item),
            )
            send(
                "response.reasoning_summary_part.added",
                item_id=reasoning_item["id"],
                output_index=reasoning_index,
                summary_index=0,
                part={"type": "summary_text", "text": ""},
            )

        def finish_reasoning() -> None:
            if reasoning_item is None or reasoning_item["status"] == "completed":
                return
            part = {"type": "summary_text", "text": reasoning_text}
            reasoning_item["summary"] = [part]
            reasoning_item["status"] = "completed"
            send(
                "response.reasoning_summary_text.done",
                item_id=reasoning_item["id"],
                output_index=reasoning_index,
                summary_index=0,
                text=reasoning_text,
            )
            send(
                "response.reasoning_summary_part.done",
                item_id=reasoning_item["id"],
                output_index=reasoning_index,
                summary_index=0,
                part=part,
            )
            send(
                "response.output_item.done",
                output_index=reasoning_index,
                item=dict(reasoning_item),
            )

        def start_message() -> None:
            nonlocal message_item, message_index
            if message_item is not None:
                return
            finish_reasoning()
            message_index = len(output)
            message_item = _response_message("", status="in_progress")
            output.append(message_item)
            send(
                "response.output_item.added",
                output_index=message_index,
                item=dict(message_item),
            )
            send(
                "response.content_part.added",
                item_id=message_item["id"],
                output_index=message_index,
                content_index=0,
                part=_response_text(""),
            )

        def finish_message() -> None:
            if message_item is None or message_item["status"] == "completed":
                return
            text_part = _response_text(message_text)
            message_item["content"] = [text_part]
            message_item["status"] = "completed"
            send(
                "response.output_text.done",
                item_id=message_item["id"],
                output_index=message_index,
                content_index=0,
                text=message_text,
                logprobs=[],
            )
            send(
                "response.content_part.done",
                item_id=message_item["id"],
                output_index=message_index,
                content_index=0,
                part=text_part,
            )
            send(
                "response.output_item.done",
                output_index=message_index,
                item=dict(message_item),
            )

        def send_delta(delta: ToolStreamDelta) -> None:
            nonlocal reasoning_text, message_text
            if delta.reasoning_content:
                start_reasoning()
                reasoning_text += delta.reasoning_content
                send(
                    "response.reasoning_summary_text.delta",
                    item_id=reasoning_item["id"],
                    output_index=reasoning_index,
                    summary_index=0,
                    delta=delta.reasoning_content,
                )
            if delta.content:
                start_message()
                message_text += delta.content
                send(
                    "response.output_text.delta",
                    item_id=message_item["id"],
                    output_index=message_index,
                    content_index=0,
                    delta=delta.content,
                    logprobs=[],
                )
            if delta.tool_index is None:
                return
            if delta.tool_name is not None and delta.tool_index not in calls:
                finish_reasoning()
                finish_message()
                item = _response_tool_call(
                    delta.tool_name,
                    "",
                    response_tools,
                    status="in_progress",
                )
                output_index = len(output)
                output.append(item)
                calls[delta.tool_index] = (output_index, item)
                send(
                    "response.output_item.added",
                    output_index=output_index,
                    item=dict(item),
                )
            if delta.arguments and delta.tool_index in calls:
                output_index, item = calls[delta.tool_index]
                item["arguments"] += delta.arguments
                send(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=output_index,
                    delta=delta.arguments,
                )

        created = _response_object(
            request_id,
            self.app.public_model,
            payload,
            options,
            [],
            None,
            status="in_progress",
        )
        send("response.created", response=created)
        raw_parts: list[str] = []
        if tool_calling:
            factory = getattr(self.app.runtime, "make_tool_stream_parser", None)
            parser = (
                factory(thinking_mode)
                if callable(factory)
                else ToolStreamParser(thinking_mode)
            )
            for piece in pieces:
                raw_parts.append(piece.text)
                prompt_tokens = piece.prompt_tokens
                generated = piece.generation_tokens
                for delta in parser.feed(piece.text):
                    send_delta(delta)
            for delta in parser.finish():
                send_delta(delta)
            try:
                turn = self.app.runtime.parse_chat("".join(raw_parts), thinking_mode)
            except Exception:
                send(
                    "error",
                    code="invalid_tool_call",
                    message="The model returned an invalid tool call.",
                    param=None,
                )
                return
            if not parser.matches(turn.tool_calls):
                send(
                    "error",
                    code="invalid_tool_call",
                    message="The streamed tool call failed validation.",
                    param=None,
                )
                return
            for index, call in enumerate(turn.tool_calls):
                if index not in calls:
                    send_delta(ToolStreamDelta(tool_index=index, tool_name=call.name))
                    send_delta(ToolStreamDelta(tool_index=index, arguments=call.arguments))
        else:
            parser = ReasoningParser(thinking_mode == "thinking")
            for piece in pieces:
                prompt_tokens = piece.prompt_tokens
                generated = piece.generation_tokens
                reasoning, content = parser.feed(piece.text)
                send_delta(ToolStreamDelta(reasoning_content=reasoning, content=content))
            reasoning, content = parser.finish()
            send_delta(ToolStreamDelta(reasoning_content=reasoning, content=content))

        finish_reasoning()
        if message_item is None and not calls:
            start_message()
        finish_message()
        for output_index, item in calls.values():
            item["status"] = "completed"
            send(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=output_index,
                arguments=item["arguments"],
            )
            send(
                "response.output_item.done",
                output_index=output_index,
                item=dict(item),
            )
        completed = _response_object(
            request_id,
            self.app.public_model,
            payload,
            options,
            output,
            _response_usage(prompt_tokens, generated),
        )
        send("response.completed", response=completed)

    def _completion(self, payload: dict[str, Any]) -> None:
        options, stream = self._common(payload)
        if payload.get("tools") not in (None, []):
            raise APIError("tools are only supported for chat completions.", param="tools")
        if payload.get("tool_choice") not in (None, "none"):
            raise APIError(
                "tool_choice is only supported for chat completions.",
                param="tool_choice",
            )
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise APIError("prompt must be a string.", param="prompt")
        request_id = "cmpl-" + uuid.uuid4().hex
        pieces = self.app.track(self.app.runtime.stream(prompt, options))
        if stream:
            self._stream_completion(pieces, request_id)
            return
        text, _, prompt_tokens, generated, finish = _collect(pieces, False)
        self._json(
            200,
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.app.public_model,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finish,
                    }
                ],
                "usage": _usage(prompt_tokens, generated),
            },
        )

    def _common(self, payload: dict[str, Any]) -> tuple[GenerationOptions, bool]:
        model = payload.get("model")
        if model != self.app.public_model:
            raise APIError(
                f"model must be '{self.app.public_model}'.",
                param="model",
                code="model_not_found",
            )
        unsupported = {
            "response_format": payload.get("response_format"),
            "stop": payload.get("stop"),
        }
        for name, value in unsupported.items():
            if value is not None and value != []:
                raise APIError(f"{name} is not supported.", param=name)
        for name in ("frequency_penalty", "presence_penalty"):
            if payload.get(name) not in (None, 0, 0.0):
                raise APIError(f"{name} is not supported.", param=name)
        if payload.get("seed") is not None:
            raise APIError("seed is not supported.", param="seed")
        if payload.get("logprobs") not in (None, False):
            raise APIError("logprobs is not supported.", param="logprobs")
        if payload.get("echo") not in (None, False):
            raise APIError("echo is not supported.", param="echo")
        n = payload.get("n", 1)
        if type(n) is not int or n != 1:
            raise APIError("n must be 1.", param="n")
        stream = payload.get("stream", False)
        if type(stream) is not bool:
            raise APIError("stream must be a boolean.", param="stream")
        stream_options = payload.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, dict):
            raise APIError("stream_options must be an object.", param="stream_options")
        if isinstance(stream_options, dict):
            include_usage = stream_options.get("include_usage", False)
            if type(include_usage) is not bool:
                raise APIError(
                    "stream_options.include_usage must be a boolean.",
                    param="stream_options.include_usage",
                )
        return _options(payload, self.app.defaults), stream

    def _stream_chat(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        thinking: bool,
        include_usage: bool,
    ) -> None:
        self._start_sse()
        created = int(time.time())
        base = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.app.public_model,
        }
        self._sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        parser = ReasoningParser(thinking)
        prompt_tokens = generated = 0
        finish = None
        for piece in pieces:
            prompt_tokens = piece.prompt_tokens
            generated = piece.generation_tokens
            finish = piece.finish_reason or finish
            reasoning, content = parser.feed(piece.text)
            if reasoning:
                self._sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]})
            if content:
                self._sse({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        reasoning, content = parser.finish()
        if reasoning:
            self._sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]})
        if content:
            self._sse({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        self._sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish or "stop"}]})
        if include_usage:
            self._sse({**base, "choices": [], "usage": _usage(prompt_tokens, generated)})
        self._sse_done()

    def _stream_tool_chat(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        thinking_mode: str,
        include_usage: bool,
    ) -> None:
        self._start_sse()
        created = int(time.time())
        base = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": self.app.public_model,
        }
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        factory = getattr(self.app.runtime, "make_tool_stream_parser", None)
        parser = (
            factory(thinking_mode)
            if callable(factory)
            else ToolStreamParser(thinking_mode)
        )
        raw_parts = []
        prompt_tokens = generated = 0
        finish = "stop"

        def send(delta: ToolStreamDelta) -> None:
            value: dict[str, Any]
            if delta.reasoning_content:
                value = {"reasoning_content": delta.reasoning_content}
            elif delta.content:
                value = {"content": delta.content}
            elif delta.tool_index is not None:
                function: dict[str, str] = {"arguments": delta.arguments}
                call: dict[str, Any] = {
                    "index": delta.tool_index,
                    "function": function,
                }
                if delta.tool_name is not None:
                    call_id = "call_" + uuid.uuid4().hex
                    call.update({"id": call_id, "type": "function"})
                    function["name"] = delta.tool_name
                value = {"tool_calls": [call]}
            else:
                return
            self._sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": value, "finish_reason": None}
                    ],
                }
            )

        for piece in pieces:
            raw_parts.append(piece.text)
            prompt_tokens = piece.prompt_tokens
            generated = piece.generation_tokens
            finish = piece.finish_reason or finish
            for delta in parser.feed(piece.text):
                send(delta)
        for delta in parser.finish():
            send(delta)

        raw = "".join(raw_parts)
        try:
            turn = self.app.runtime.parse_chat(raw, thinking_mode)
        except Exception:
            self._sse(
                {
                    "error": {
                        "message": "The model returned an invalid tool call.",
                        "type": "server_error",
                        "param": None,
                        "code": "invalid_tool_call",
                    }
                }
            )
            self._sse_done()
            return
        if not parser.matches(turn.tool_calls):
            if parser.streamed_tool_count:
                self._sse(
                    {
                        "error": {
                            "message": "The streamed tool call failed validation.",
                            "type": "server_error",
                            "param": None,
                            "code": "invalid_tool_call",
                        }
                    }
                )
                self._sse_done()
                return
            for index, call in enumerate(_response_tool_calls(turn.tool_calls)):
                self._sse(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": index, **call}]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if turn.tool_calls else finish,
                    }
                ],
            }
        )
        if include_usage:
            self._sse(
                {
                    **base,
                    "choices": [],
                    "usage": _usage(prompt_tokens, generated),
                }
            )
        self._sse_done()

    def _stream_completion(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
    ) -> None:
        self._start_sse()
        created = int(time.time())
        finish = None
        for piece in pieces:
            finish = piece.finish_reason or finish
            self._sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.app.public_model,
                    "choices": [
                        {
                            "index": 0,
                            "text": piece.text,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        self._sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": self.app.public_model,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": finish or "stop",
                    }
                ],
            }
        )
        self._sse_done()

    def _status(self) -> dict[str, Any]:
        config = self.app.runtime.config
        installed = self.app.runtime.installed
        cache = self.app.runtime.expert_cache
        cache_metrics = cache.metrics
        return {
            "status": "ready",
            "model": self.app.public_model,
            "source_model": self.app.runtime.model_id,
            "model_path": str(installed.root),
            "requires_api_key": self.app.api_key is not None,
            "runtime": {
                "slots": config.slots,
                "read_workers": config.read_workers,
                "prefetch_read_workers": getattr(
                    config, "prefetch_read_workers", 1
                ),
                "power_saving_limit_gbps": getattr(
                    config, "power_saving_limit_gbps", None
                ),
                "prefill_step_size": config.prefill_step_size,
                "moe_prefill_step_size": getattr(
                    config, "moe_prefill_step_size", 0
                ),
                "layer_major_prefill": getattr(config, "layer_major_prefill", False),
                "batched_expert_prefill": getattr(
                    config, "batched_expert_prefill", False
                ),
                "prompt_cache_entries": getattr(config, "prompt_cache_entries", 1),
                "prompt_cache_memory_gib": getattr(
                    config, "prompt_cache_memory_gib", 0
                ),
                "persistent_prompt_cache": getattr(
                    config, "persistent_prompt_cache", False
                ),
                "fp4_index_cache": getattr(config, "fp4_index_cache", False),
                "dspark_available": getattr(installed, "has_dspark", False),
                "dspark_enabled": bool(
                    getattr(getattr(self.app.runtime, "model", None), "dspark", None)
                ),
                "dspark_confidence_threshold": getattr(
                    config, "dspark_confidence_threshold", 0.6
                ),
                "dspark_slots": getattr(config, "dspark_slots", 768),
                "kv_cache": "MXFP8" if config.fp8_kv_cache else "BF16",
            },
            "performance": {
                **self.app.metrics.snapshot(),
                **self.app.runtime.metrics.snapshot(),
                "ssd_bytes_read": cache_metrics.bytes_read,
                "ssd_read_seconds": cache_metrics.read_seconds,
                "expert_pack_seconds": cache_metrics.pack_seconds,
                "expert_eviction_seconds": cache_metrics.eviction_seconds,
                "routing_sync_seconds": cache_metrics.routing_sync_seconds,
                "active_parameters_cache": {
                    "hit_rate": cache_metrics.hit_rate,
                    "hits": cache_metrics.hits,
                    "misses": cache_metrics.misses,
                    "resident_slots": cache.resident_count,
                    "capacity_slots": config.slots,
                },
            },
        }

    def _authorize(self) -> None:
        expected = self.app.api_key
        if expected is None:
            return
        value = self.headers.get("Authorization", "")
        supplied = value[7:] if value.startswith("Bearer ") else ""
        if not secrets.compare_digest(supplied, expected):
            raise APIError(
                "Provide a valid Bearer API key.",
                status=401,
                code="invalid_api_key",
                error_type="authentication_error",
            )

    def _request_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise APIError("Content-Length is required.", status=411)
        try:
            length = int(raw_length)
        except ValueError as error:
            raise APIError("Content-Length is invalid.", status=400) from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise APIError("Request body is too large.", status=413)
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise APIError("Request body must be valid JSON.") from error
        if not isinstance(value, dict):
            raise APIError("Request body must be a JSON object.")
        return value

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self._bytes(status, body, "application/json; charset=utf-8")

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.client_address[0]} - {format % args}\n")


def _validate_response_request(payload: dict[str, Any]) -> None:
    for name in ("previous_response_id", "conversation"):
        if payload.get(name) is not None:
            raise APIError(f"{name} is not supported.", param=name)
    for name in ("store", "background"):
        if payload.get(name) not in (None, False):
            raise APIError(f"{name} is not supported.", param=name)
    parallel = payload.get("parallel_tool_calls")
    if parallel is not None and type(parallel) is not bool:
        raise APIError(
            "parallel_tool_calls must be a boolean.",
            param="parallel_tool_calls",
        )
    if payload.get("truncation") not in (None, "disabled"):
        raise APIError("truncation is not supported.", param="truncation")
    text = payload.get("text")
    if text not in (None, {}):
        if not isinstance(text, dict) or text.get("format", {"type": "text"}) != {
            "type": "text"
        }:
            raise APIError("Only plain text output is supported.", param="text.format")


_REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "low",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def _reasoning_settings(
    payload: dict[str, Any],
    *,
    responses_api: bool = False,
    qwen: bool = False,
) -> tuple[str, str]:
    thinking_mode = payload.get("thinking_mode")
    if thinking_mode is not None:
        if thinking_mode not in {"chat", "thinking"}:
            raise APIError(
                "thinking_mode must be 'chat' or 'thinking'.",
                param="thinking_mode",
            )
    if responses_api:
        reasoning = payload.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, dict):
            raise APIError("reasoning must be an object.", param="reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        param = "reasoning.effort"
    else:
        effort = payload.get("reasoning_effort")
        param = "reasoning_effort"
    if effort is not None and not isinstance(effort, str):
        raise APIError(f"{param} must be a string.", param=param)
    if effort not in {None, "none", *_REASONING_EFFORT_MAP}:
        raise APIError(f"{param} is invalid.", param=param)
    if thinking_mode is None:
        thinking_mode = "chat" if effort in {None, "none"} else "thinking"
    native_effort = (
        {
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": "xhigh",
        }.get(effort, "low")
        if qwen
        else _REASONING_EFFORT_MAP.get(effort, "low")
    )
    return thinking_mode, native_effort


def _response_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise APIError("instructions must be a string.", param="instructions")
        messages.append({"role": "developer", "content": instructions})

    value = payload.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return _messages(messages, allow_assistant_final=True)
    if not isinstance(value, list) or not value:
        raise APIError("input must be a string or a non-empty array.", param="input")

    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": pending_calls[:]}
            )
            pending_calls.clear()

    for index, raw in enumerate(value):
        param = f"input.{index}"
        if not isinstance(raw, dict):
            raise APIError("Each input item must be an object.", param=param)
        item_type = raw.get("type")
        if item_type == "function_call":
            call_id = raw.get("call_id")
            name = raw.get("name")
            namespace = raw.get("namespace")
            arguments = raw.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                raise APIError("call_id must be a non-empty string.", param=f"{param}.call_id")
            _validate_function_name(name, f"{param}.name")
            if namespace is not None:
                _validate_function_name(namespace, f"{param}.namespace")
            name = _response_internal_tool_name(name, namespace)
            if not isinstance(arguments, str):
                raise APIError("arguments must be a JSON string.", param=f"{param}.arguments")
            pending_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue
        if item_type == "reasoning":
            continue
        flush_calls()
        if item_type == "function_call_output":
            call_id = raw.get("call_id")
            output = raw.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise APIError("call_id must be a non-empty string.", param=f"{param}.call_id")
            if not isinstance(output, str):
                raise APIError("output must be a string.", param=f"{param}.output")
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": output}
            )
            continue
        if item_type not in (None, "message"):
            raise APIError("Only text messages and function calls are supported.", param=param)
        messages.append({"role": raw.get("role"), "content": raw.get("content")})
    flush_calls()
    return _messages(messages, allow_assistant_final=True)


def _response_tool_request(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], ToolChoice, dict[str, dict[str, str]]]:
    raw_tools = payload.get("tools")
    response_tools: dict[str, dict[str, str]] = {}
    if raw_tools is None:
        tools = None
    elif not isinstance(raw_tools, list):
        raise APIError("tools must be an array.", param="tools")
    else:
        tools = []

        def add_tool(raw: dict[str, Any], param: str, namespace: str | None = None) -> None:
            kind = raw.get("type")
            if kind != "function":
                raise APIError("Namespace tools must be functions.", param=param)
            name = raw.get("name")
            _validate_function_name(name, f"{param}.name")
            internal_name = _response_internal_tool_name(name, namespace)
            description = raw.get("description", "")
            if not isinstance(description, str):
                raise APIError("Tool description must be a string.", param=f"{param}.description")
            function = {
                field: raw[field]
                for field in ("description", "parameters", "strict")
                if field in raw
            }
            function["name"] = internal_name
            tools.append({"type": "function", "function": function})
            response_tools[internal_name] = {
                "type": kind,
                "name": name,
                **({"namespace": namespace} if namespace else {}),
            }

        for index, raw in enumerate(raw_tools):
            param = f"tools.{index}"
            if not isinstance(raw, dict):
                raise APIError("Each tool must be an object.", param=param)
            kind = raw.get("type")
            if kind == "function":
                add_tool(raw, param)
                continue
            if kind == "namespace":
                namespace = raw.get("name")
                _validate_function_name(namespace, f"{param}.name")
                namespace_tools = raw.get("tools")
                if not isinstance(namespace_tools, list):
                    raise APIError("Namespace tools must be an array.", param=f"{param}.tools")
                for child_index, child in enumerate(namespace_tools):
                    child_param = f"{param}.tools.{child_index}"
                    if not isinstance(child, dict):
                        raise APIError("Each namespace tool must be an object.", param=child_param)
                    add_tool(child, child_param, namespace)
                continue
            if kind == "web_search":
                continue
            raise APIError(
                "This tool type is not supported.",
                param=param,
            )
    choice = payload.get("tool_choice", "auto")
    if isinstance(choice, dict) and choice.get("type") == "function":
        selected = next(
            (
                internal_name
                for internal_name, tool in response_tools.items()
                if tool["type"] == choice.get("type")
                and tool["name"] == choice.get("name")
                and tool.get("namespace") == choice.get("namespace")
            ),
            None,
        )
        choice = {"type": "function", "function": {"name": selected}}
    request = {"tools": tools, "tool_choice": choice}
    parsed_tools, parsed_choice = _tool_request(request)
    return parsed_tools, parsed_choice, response_tools


def _response_text(text: str) -> dict[str, Any]:
    return {"type": "output_text", "annotations": [], "logprobs": [], "text": text}


def _response_message(text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [_response_text(text)],
    }


def _response_output(
    text: str,
    reasoning: str,
    *,
    include_empty: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if reasoning:
        output.append(
            {
                "id": "rs_" + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "status": "completed",
            }
        )
    if text or (include_empty and not reasoning):
        output.append(_response_message(text))
    return output


def _response_internal_tool_name(name: str, namespace: str | None) -> str:
    return f"{namespace}__{name}" if namespace else name


def _response_tool_call(
    internal_name: str,
    arguments: str,
    response_tools: dict[str, dict[str, str]],
    *,
    status: str,
) -> dict[str, Any]:
    tool = response_tools.get(
        internal_name,
        {"type": "function", "name": internal_name},
    )
    item: dict[str, Any] = {
        "id": "fc_" + uuid.uuid4().hex,
        "call_id": "call_" + uuid.uuid4().hex,
        "type": "function_call",
        "name": tool["name"],
        "arguments": arguments,
        "status": status,
    }
    if "namespace" in tool:
        item["namespace"] = tool["namespace"]
    return item


def _response_function_calls(
    calls,
    response_tools: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    tools = response_tools or {}
    return [
        _response_tool_call(call.name, call.arguments, tools, status="completed")
        for call in calls
    ]


def _response_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _response_object(
    request_id: str,
    model: str,
    payload: dict[str, Any],
    options: GenerationOptions,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    now = time.time()
    return {
        "id": request_id,
        "object": "response",
        "created_at": now,
        "status": status,
        "completed_at": now if status == "completed" else None,
        "error": None,
        "incomplete_details": None,
        "instructions": payload.get("instructions"),
        "max_output_tokens": options.max_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": payload.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": payload.get("reasoning"),
        "store": False,
        "temperature": options.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": payload.get("tools") or [],
        "top_p": options.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": payload.get("metadata") or {},
    }


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(f"{name} must be a number.", param=name)
    result = float(value)
    if not minimum <= result <= maximum:
        raise APIError(
            f"{name} must be between {minimum:g} and {maximum:g}.",
            param=name,
        )
    return result


def _options(payload: dict[str, Any], defaults: ServerDefaults) -> GenerationOptions:
    max_tokens = payload.get(
        "max_completion_tokens",
        payload.get("max_output_tokens", payload.get("max_tokens", defaults.max_tokens)),
    )
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise APIError("max_tokens must be an integer.", param="max_tokens")
    if not 1 <= max_tokens <= MAX_GENERATION_TOKENS:
        raise APIError(
            f"max_tokens must be between 1 and {MAX_GENERATION_TOKENS}.",
            param="max_tokens",
        )
    temperature = _number(
        payload.get("temperature", defaults.temperature),
        "temperature",
        minimum=0,
        maximum=2,
    )
    top_p = _number(
        payload.get("top_p", defaults.top_p),
        "top_p",
        minimum=0.000001,
        maximum=1,
    )
    top_k = payload.get("top_k", defaults.top_k)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 248_320:
        raise APIError("top_k must be between 0 and 248320.", param="top_k")
    return GenerationOptions(max_tokens, temperature, top_p, top_k)


def _tool_request(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], ToolChoice]:
    value = payload.get("tools")
    if value is None:
        tools: list[dict[str, Any]] = []
    elif not isinstance(value, list):
        raise APIError("tools must be an array.", param="tools")
    else:
        tools = []
        names = set()
        for index, raw in enumerate(value):
            param = f"tools.{index}"
            if not isinstance(raw, dict) or raw.get("type") != "function":
                raise APIError("Each tool must be a function.", param=param)
            function = raw.get("function")
            if not isinstance(function, dict):
                raise APIError("Tool function must be an object.", param=f"{param}.function")
            name = function.get("name")
            _validate_function_name(name, f"{param}.function.name")
            if name in names:
                raise APIError("Tool function names must be unique.", param=f"{param}.function.name")
            names.add(name)
            description = function.get("description")
            if description is not None and not isinstance(description, str):
                raise APIError(
                    "Tool description must be a string.",
                    param=f"{param}.function.description",
                )
            parameters = function.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise APIError(
                    "Tool parameters must be an object.",
                    param=f"{param}.function.parameters",
                )
            tools.append(raw)

    raw_choice = payload.get("tool_choice", "auto")
    if isinstance(raw_choice, str) and raw_choice in {"auto", "none", "required"}:
        choice = ToolChoice(raw_choice)
    elif isinstance(raw_choice, dict):
        function = raw_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if raw_choice.get("type") != "function":
            raise APIError("tool_choice must select a function.", param="tool_choice")
        _validate_function_name(name, "tool_choice.function.name")
        if name not in {tool["function"]["name"] for tool in tools}:
            raise APIError(
                "tool_choice must name a provided tool.",
                param="tool_choice.function.name",
            )
        choice = ToolChoice("function", name)
    else:
        raise APIError(
            "tool_choice must be auto, none, required, or a function.",
            param="tool_choice",
        )
    if not tools and choice.mode in {"required", "function"}:
        raise APIError("tool_choice requires at least one tool.", param="tool_choice")
    return tools, choice


def _validate_function_name(value: Any, param: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is None:
        raise APIError(
            "Function name must use 1 to 64 letters, numbers, underscores, or hyphens.",
            param=param,
        )


def _assistant_tool_calls(value: Any, param: str) -> list[dict[str, Any]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise APIError("tool_calls must be an array.", param=param)
    result = []
    ids = set()
    for index, raw in enumerate(value):
        item_param = f"{param}.{index}"
        if not isinstance(raw, dict) or raw.get("type") != "function":
            raise APIError("Each tool call must be a function.", param=item_param)
        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in ids:
            raise APIError("Tool call IDs must be unique strings.", param=f"{item_param}.id")
        function = raw.get("function")
        if not isinstance(function, dict):
            raise APIError("Tool call function must be an object.", param=f"{item_param}.function")
        name = function.get("name")
        _validate_function_name(name, f"{item_param}.function.name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise APIError(
                "Tool call arguments must be a JSON string.",
                param=f"{item_param}.function.arguments",
            )
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise APIError(
                "Tool call arguments must contain valid JSON.",
                param=f"{item_param}.function.arguments",
            ) from error
        if not isinstance(decoded, dict):
            raise APIError(
                "Tool call arguments must contain a JSON object.",
                param=f"{item_param}.function.arguments",
            )
        ids.add(call_id)
        result.append(raw)
    return result


def _messages(
    value: Any,
    *,
    allow_assistant_final: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise APIError("messages must be a non-empty array.", param="messages")
    result = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise APIError("Each message must be an object.", param=f"messages.{index}")
        role = raw.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise APIError(
                "Message role must be system, developer, user, assistant, or tool.",
                param=f"messages.{index}.role",
            )
        tool_calls = (
            _assistant_tool_calls(raw.get("tool_calls"), f"messages.{index}.tool_calls")
            if role == "assistant"
            else []
        )
        content_value = raw.get("content")
        content = (
            ""
            if role == "assistant" and content_value is None and tool_calls
            else _message_text(content_value, f"messages.{index}.content")
        )
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if role == "tool":
            call_id = raw.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise APIError(
                    "tool_call_id must be a non-empty string.",
                    param=f"messages.{index}.tool_call_id",
                )
            message["tool_call_id"] = call_id
        reasoning = raw.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant" or not isinstance(reasoning, str):
                raise APIError(
                    "reasoning_content must be a string on an assistant message.",
                    param=f"messages.{index}.reasoning_content",
                )
            message["reasoning_content"] = reasoning
        result.append(message)

    pending_tool_ids: set[str] = set()
    for index, message in enumerate(result):
        role = message["role"]
        if role == "assistant":
            if pending_tool_ids:
                raise APIError(
                    "Provide every tool result before the next assistant message.",
                    param=f"messages.{index}",
                )
            pending_tool_ids = {
                call["id"] for call in message.get("tool_calls", [])
            }
        elif role == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending_tool_ids:
                raise APIError(
                    "tool_call_id must match the previous assistant tool call.",
                    param=f"messages.{index}.tool_call_id",
                )
            pending_tool_ids.remove(call_id)
        elif pending_tool_ids:
            raise APIError(
                "Provide every tool result before the next message.",
                param=f"messages.{index}",
            )
    if pending_tool_ids:
        raise APIError(
            "Provide every tool result before requesting another response.",
            param="messages",
        )
    allowed_final_roles = {"user", "developer", "tool"}
    if allow_assistant_final:
        allowed_final_roles.add("assistant")
    if result[-1]["role"] not in allowed_final_roles:
        raise APIError(
            "The final message must have the user, developer, or tool role.",
            param=f"messages.{len(result) - 1}.role",
        )
    return result


def _message_text(value: Any, param: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict) or item.get("type") not in {
                "text",
                "input_text",
                "output_text",
            }:
                raise APIError("Only text message content is supported.", param=param)
            text = item.get("text")
            if not isinstance(text, str):
                raise APIError("Message text must be a string.", param=param)
            parts.append(text)
        return "".join(parts)
    raise APIError("Message content must be text.", param=param)


def _collect(
    pieces: Iterator[GeneratedPiece],
    thinking: bool,
) -> tuple[str, str, int, int, str]:
    parser = ReasoningParser(thinking)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    prompt_tokens = generated = 0
    finish = None
    for piece in pieces:
        prompt_tokens = piece.prompt_tokens
        generated = piece.generation_tokens
        finish = piece.finish_reason or finish
        reasoning, content = parser.feed(piece.text)
        reasoning_parts.append(reasoning)
        content_parts.append(content)
    reasoning, content = parser.finish()
    reasoning_parts.append(reasoning)
    content_parts.append(content)
    return (
        "".join(content_parts),
        "".join(reasoning_parts),
        prompt_tokens,
        generated,
        finish or "stop",
    )


def _collect_raw(
    pieces: Iterator[GeneratedPiece],
) -> tuple[str, int, int, str]:
    parts = []
    prompt_tokens = generated = 0
    finish = None
    for piece in pieces:
        parts.append(piece.text)
        prompt_tokens = piece.prompt_tokens
        generated = piece.generation_tokens
        finish = piece.finish_reason or finish
    return "".join(parts), prompt_tokens, generated, finish or "stop"


def _response_tool_calls(calls) -> list[dict[str, Any]]:
    return [
        {
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }
        for call in calls
    ]


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the installed model with an OpenAI-compatible API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--public-model")
    parser.add_argument("--slots", type=int, default=1_152)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--prefetch-read-workers", type=int, default=2)
    parser.add_argument(
        "--power-saving-limit-gbps",
        type=float,
        choices=_POWER_SAVING_LIMITS_GBPS,
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=int,
        default=0,
        help="MLX memory limit in GiB; 0 selects the model-safe automatic limit",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=0,
        help="prompt chunk size; 0 selects 128, 256, or 1,024 automatically",
    )
    parser.add_argument("--moe-prefill-step-size", type=int, default=0)
    parser.add_argument("--no-layer-major-prefill", action="store_true")
    parser.add_argument("--no-batched-expert-prefill", action="store_true")
    parser.add_argument("--prompt-cache-entries", type=int, default=2)
    parser.add_argument("--prompt-cache-memory-gib", type=int, default=8)
    parser.add_argument("--no-persistent-prompt-cache", action="store_true")
    parser.add_argument("--prompt-cache-directory")
    parser.add_argument("--warmup-prompt-file")
    parser.add_argument("--bf16-kv-cache", action="store_true")
    parser.add_argument("--no-fp4-index-cache", action="store_true")
    parser.add_argument("--no-ready-expert-decode", action="store_true")
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument("--dspark-slots", type=int, default=768)
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--default-max-tokens", type=int, default=272_000)
    parser.add_argument("--default-temperature", type=float)
    parser.add_argument("--default-top-p", type=float)
    parser.add_argument("--default-top-k", type=int)
    return parser


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "::1", "localhost"} and not arguments.api_key:
        parser.error("--api-key is required when --host is not local")
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if arguments.slots < 6:
        parser.error("--slots must be at least 6")
    if arguments.read_workers < 1:
        parser.error("--read-workers must be greater than zero")
    if arguments.prefetch_read_workers < 1:
        parser.error("--prefetch-read-workers must be greater than zero")
    if arguments.memory_limit_gib < 0:
        parser.error("--memory-limit-gib must be zero or greater")
    if arguments.prefill_step_size < 0:
        parser.error("--prefill-step-size must be zero or greater")
    if arguments.moe_prefill_step_size < 0:
        parser.error("--moe-prefill-step-size must be zero or greater")
    if arguments.prompt_cache_entries < 1:
        parser.error("--prompt-cache-entries must be greater than zero")
    if arguments.prompt_cache_memory_gib < 1:
        parser.error("--prompt-cache-memory-gib must be greater than zero")
    if not 0 <= arguments.dspark_confidence_threshold <= 1:
        parser.error("--dspark-confidence-threshold must be between zero and one")
    if arguments.dspark_slots < 30:
        parser.error("--dspark-slots must be at least 30")
    try:
        installed = InstalledModel.open(arguments.model)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if installed.is_qwen and arguments.dspark:
        parser.error("Qwen3.8-Flash-Next does not support --dspark")
    public_model = arguments.public_model or installed.model_id
    if not public_model:
        parser.error("--public-model must not be empty")
    default_temperature = arguments.default_temperature
    default_top_p = arguments.default_top_p
    default_top_k = arguments.default_top_k
    if default_temperature is None:
        default_temperature = 1.0 if installed.is_qwen else 0.2
    if default_top_p is None:
        default_top_p = 0.95 if installed.is_qwen else 0.98
    if default_top_k is None:
        default_top_k = 20 if installed.is_qwen else 0
    try:
        options = _options(
            {
                "max_tokens": arguments.default_max_tokens,
                "temperature": default_temperature,
                "top_p": default_top_p,
                "top_k": default_top_k,
            },
            ServerDefaults(),
        )
        defaults = ServerDefaults(
            options.max_tokens,
            options.temperature,
            options.top_p,
            options.top_k,
        )
    except APIError as error:
        parser.error(str(error))
    config = RuntimeConfig(
        slots=arguments.slots,
        read_workers=arguments.read_workers,
        prefetch_read_workers=arguments.prefetch_read_workers,
        memory_limit_gib=arguments.memory_limit_gib,
        prefill_step_size=arguments.prefill_step_size,
        moe_prefill_step_size=arguments.moe_prefill_step_size,
        fp8_kv_cache=not arguments.bf16_kv_cache,
        layer_major_prefill=not arguments.no_layer_major_prefill,
        batched_expert_prefill=not arguments.no_batched_expert_prefill,
        prompt_cache_entries=arguments.prompt_cache_entries,
        prompt_cache_memory_gib=arguments.prompt_cache_memory_gib,
        persistent_prompt_cache=not arguments.no_persistent_prompt_cache,
        prompt_cache_directory=arguments.prompt_cache_directory,
        fp4_index_cache=not arguments.no_fp4_index_cache,
        ready_expert_decode=not arguments.no_ready_expert_decode,
        dspark_enabled=arguments.dspark,
        dspark_slots=arguments.dspark_slots,
        dspark_confidence_threshold=arguments.dspark_confidence_threshold,
        power_saving_limit_gbps=arguments.power_saving_limit_gbps,
    )
    print(f"Loading {arguments.model}...", flush=True)
    runtime = ModelRuntime.open(arguments.model, config)
    server = None
    try:
        if arguments.warmup_prompt_file:
            try:
                with open(arguments.warmup_prompt_file, encoding="utf-8") as file:
                    warmup_prompt = file.read()
            except OSError as error:
                parser.error(f"cannot read --warmup-prompt-file: {error}")
            print("Warming prompt cache...", flush=True)
            warmed = runtime.warm_prompt(warmup_prompt)
            print(f"Warmed {warmed} prompt tokens.", flush=True)
        server = OpenAIServer(
            (arguments.host, arguments.port),
            runtime,
            public_model=public_model,
            api_key=arguments.api_key,
            defaults=defaults,
        )
        print(f"Ready: http://{arguments.host}:{server.server_port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        if server is not None:
            server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
