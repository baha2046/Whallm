from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ENCODER_SHA256 = (
    "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"
)


@dataclass(frozen=True)
class ToolChoice:
    mode: str = "auto"
    name: str | None = None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    reasoning_content: str
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class ToolStreamDelta:
    reasoning_content: str = ""
    content: str = ""
    tool_index: int | None = None
    tool_name: str | None = None
    arguments: str = ""


class ToolStreamParser:
    """Incrementally separate DeepSeek text and DSML tool calls."""

    _tool_start = "\n\n<｜DSML｜tool_calls>"
    _tools_end = "</｜DSML｜tool_calls>"
    _invoke_start = "<｜DSML｜invoke"
    _invoke_end = "</｜DSML｜invoke>"
    _parameter_start = "<｜DSML｜parameter"
    _parameter_end = "</｜DSML｜parameter>"

    def __init__(self, thinking_mode: str):
        self._state = "reasoning" if thinking_mode == "thinking" else "content"
        self._buffer = ""
        self._tool_index = -1
        self._parameter_is_string = False
        self._names: list[str] = []
        self._arguments: list[str] = []
        self.failed = False

    def feed(self, text: str) -> tuple[ToolStreamDelta, ...]:
        if self.failed:
            return ()
        self._buffer += text
        deltas: list[ToolStreamDelta] = []
        while not self.failed:
            before = (self._state, self._buffer)
            if self._state == "reasoning":
                self._read_text("</think>", "content", deltas, reasoning=True)
            elif self._state == "content":
                self._read_text(self._tool_start, "tool", deltas)
            elif self._state == "tool":
                self._read_tool(deltas)
            elif self._state == "parameter_or_end":
                self._read_parameter_or_end(deltas)
            elif self._state == "parameter_value":
                self._read_parameter_value(deltas)
            else:
                break
            if before == (self._state, self._buffer):
                break
        return tuple(deltas)

    def finish(self) -> tuple[ToolStreamDelta, ...]:
        if self.failed:
            return ()
        if self._state == "content" and self._buffer:
            delta = ToolStreamDelta(content=self._buffer)
            self._buffer = ""
            return (delta,)
        return ()

    def matches(self, calls: tuple[ToolCall, ...]) -> bool:
        if (
            self.failed
            or len(calls) != len(self._names)
            or (calls and self._state != "done")
        ):
            return False
        for index, call in enumerate(calls):
            if call.name != self._names[index]:
                return False
            try:
                if json.loads(call.arguments) != json.loads(self._arguments[index]):
                    return False
            except json.JSONDecodeError:
                return False
        return True

    @property
    def streamed_tool_count(self) -> int:
        return len(self._names)

    def _read_text(
        self,
        marker: str,
        next_state: str,
        deltas: list[ToolStreamDelta],
        *,
        reasoning: bool = False,
    ) -> None:
        position = self._buffer.find(marker)
        if position >= 0:
            self._emit_text(self._buffer[:position], deltas, reasoning)
            self._buffer = self._buffer[position + len(marker) :]
            self._state = next_state
            return
        keep = self._marker_suffix_length(self._buffer, marker)
        ready = self._buffer[:-keep] if keep else self._buffer
        self._buffer = self._buffer[-keep:] if keep else ""
        self._emit_text(ready, deltas, reasoning)

    @staticmethod
    def _emit_text(
        text: str,
        deltas: list[ToolStreamDelta],
        reasoning: bool,
    ) -> None:
        if text:
            deltas.append(
                ToolStreamDelta(
                    reasoning_content=text if reasoning else "",
                    content="" if reasoning else text,
                )
            )

    def _read_tool(self, deltas: list[ToolStreamDelta]) -> None:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return
        if self._wait_for_marker(self._tools_end):
            if self._buffer.startswith(self._tools_end):
                self._buffer = self._buffer[len(self._tools_end) :]
                self._state = "done"
            return
        if self._invoke_start.startswith(self._buffer):
            return
        if not self._buffer.startswith(self._invoke_start):
            self.failed = True
            return
        end = self._buffer.find(">\n")
        if end < 0:
            return
        header = self._buffer[: end + 1]
        match = re.fullmatch(r'<｜DSML｜invoke name="([A-Za-z0-9_-]{1,64})">', header)
        if match is None:
            self.failed = True
            return
        self._buffer = self._buffer[end + 2 :]
        self._tool_index += 1
        name = match.group(1)
        self._names.append(name)
        self._arguments.append("")
        self._state = "parameter_or_end"
        deltas.append(ToolStreamDelta(tool_index=self._tool_index, tool_name=name))

    def _read_parameter_or_end(self, deltas: list[ToolStreamDelta]) -> None:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return
        if self._wait_for_marker(self._invoke_end):
            if self._buffer.startswith(self._invoke_end):
                self._buffer = self._buffer[len(self._invoke_end) :]
                self._append_arguments("}", deltas, empty="{}")
                self._state = "tool"
            return
        if self._parameter_start.startswith(self._buffer):
            return
        if not self._buffer.startswith(self._parameter_start):
            self.failed = True
            return
        end = self._buffer.find(">")
        if end < 0:
            return
        header = self._buffer[: end + 1]
        match = re.fullmatch(
            r'<｜DSML｜parameter name="([^"]+)" string="(true|false)">',
            header,
        )
        if match is None:
            self.failed = True
            return
        self._buffer = self._buffer[end + 1 :]
        self._parameter_is_string = match.group(2) == "true"
        prefix = (
            ("{" if not self._arguments[self._tool_index] else ", ")
            + json.dumps(match.group(1), ensure_ascii=False)
            + ": "
            + ('"' if self._parameter_is_string else "")
        )
        self._append_arguments(prefix, deltas)
        self._state = "parameter_value"

    def _read_parameter_value(self, deltas: list[ToolStreamDelta]) -> None:
        position = self._buffer.find(self._parameter_end)
        if position >= 0:
            self._append_value(self._buffer[:position], deltas)
            self._buffer = self._buffer[position + len(self._parameter_end) :]
            if self._parameter_is_string:
                self._append_arguments('"', deltas)
            self._state = "parameter_or_end"
            return
        keep = self._marker_suffix_length(self._buffer, self._parameter_end)
        ready = self._buffer[:-keep] if keep else self._buffer
        self._buffer = self._buffer[-keep:] if keep else ""
        self._append_value(ready, deltas)

    def _append_value(
        self,
        value: str,
        deltas: list[ToolStreamDelta],
    ) -> None:
        if not value:
            return
        fragment = (
            json.dumps(value, ensure_ascii=False)[1:-1]
            if self._parameter_is_string
            else value
        )
        self._append_arguments(fragment, deltas)

    def _append_arguments(
        self,
        fragment: str,
        deltas: list[ToolStreamDelta],
        *,
        empty: str | None = None,
    ) -> None:
        if empty is not None and not self._arguments[self._tool_index]:
            fragment = empty
        self._arguments[self._tool_index] += fragment
        deltas.append(
            ToolStreamDelta(tool_index=self._tool_index, arguments=fragment)
        )

    def _wait_for_marker(self, marker: str) -> bool:
        return self._buffer.startswith(marker) or marker.startswith(self._buffer)

    @staticmethod
    def _marker_suffix_length(text: str, marker: str) -> int:
        maximum = min(len(text), len(marker) - 1)
        for length in range(maximum, 0, -1):
            if text.endswith(marker[:length]):
                return length
        return 0


class ToolCodec:
    """Use the pinned DeepSeek-V4 encoder through one small interface."""

    def __init__(self, encoding: ModuleType):
        self._encoding = encoding

    @classmethod
    def open(cls, model_root: Path) -> ToolCodec:
        path = model_root / "encoding" / "encoding_dsv4.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ENCODER_SHA256:
            raise RuntimeError("DeepSeek-V4 encoder checksum does not match")
        spec = importlib.util.spec_from_file_location(
            f"_deepseek_v4_encoding_{abs(hash(path))}",
            path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load DeepSeek-V4 encoder: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "encode_messages") or not hasattr(
            module,
            "parse_message_from_completion_text",
        ):
            raise RuntimeError("DeepSeek-V4 encoder is missing required functions")
        return cls(module)

    def encode(
        self,
        messages: list[dict[str, Any]],
        thinking_mode: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = ToolChoice(),
    ) -> str:
        prepared = copy.deepcopy(messages)
        active_tools = [] if tool_choice.mode == "none" else list(tools or [])
        if active_tools:
            instruction = self._choice_instruction(tool_choice)
            target = next(
                (message for message in prepared if message["role"] == "system"),
                None,
            )
            if target is None:
                target = {"role": "system", "content": ""}
                prepared.insert(0, target)
            if instruction:
                content = target.get("content") or ""
                target["content"] = f"{content}\n\n{instruction}".strip()
            target["tools"] = active_tools
        return self._encoding.encode_messages(
            prepared,
            thinking_mode=thinking_mode,
        )

    def parse(self, text: str, thinking_mode: str) -> AssistantTurn:
        eos = self._encoding.eos_token
        complete = text if text.endswith(eos) else text + eos
        parsed = self._encoding.parse_message_from_completion_text(
            complete,
            thinking_mode=thinking_mode,
        )
        calls = tuple(
            ToolCall(
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in parsed.get("tool_calls") or []
        )
        return AssistantTurn(
            content=parsed.get("content") or "",
            reasoning_content=parsed.get("reasoning_content") or "",
            tool_calls=calls,
        )

    @staticmethod
    def _choice_instruction(choice: ToolChoice) -> str:
        if choice.mode == "required":
            return "Call one or more tools before you give a final answer."
        if choice.mode == "function":
            return f'Call the "{choice.name}" tool before you give a final answer.'
        return ""
