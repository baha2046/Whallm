# OpenAI-compatible server

The server keeps one installed model resident. It processes one generation
request at a time. This matches the current batch size 1 runtime.

## Start the macOS app

```sh
cd /Users/Shared/Project/test/llm_ssd
make run
```

Use the SwiftUI app to select the installed model, configure the runtime, and
start the server. The default address is `http://127.0.0.1:11434`.

Start the server without the app when needed:

```sh
DEEPSEEK_API_KEY=local-key make server
```

The server allows a local address without an API key. The server requires
`--api-key` or `DEEPSEEK_API_KEY` when `--host` is not local.

Use `--prefill-step-size 0` to select 128, 256, or 1,024 tokens automatically.
Layer-major prefill is enabled for requests with at least 4,096 uncached
tokens. The default layer-local MoE tile is 4,096 tokens. The runtime uses a
strided routed expert layer and batched `gather_qmm` during this path. Use
`--no-batched-expert-prefill` or `--no-layer-major-prefill` only for comparison.

The server keeps two prompt cache timelines within an 8 GiB limit. It keeps up
to eight persistent cache entries under `~/.dsmodel/prompt-cache/`. Use
`--no-persistent-prompt-cache` to disable disk cache. Use
`--warmup-prompt-file PATH` to populate a fixed UTF-8 prompt prefix during
startup.

The runtime uses four workers for individual routed expert reads. It uses two
workers for full-layer prefetch. Use `--prefetch-read-workers` to change the
second value. The indexer uses an FP4 cache by default. Use
`--no-fp4-index-cache` for the MXFP8 comparison path.

## OpenAI Python client

Install the OpenAI client in the environment that calls the server:

```sh
python -m pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="local-key",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash-0731",
    messages=[{"role": "user", "content": "法國的首都是哪裡？"}],
    max_tokens=32,
    temperature=0,
)

print(response.choices[0].message.content)
```

Use the Responses API when your client expects typed output items:

```python
response = client.responses.create(
    model="deepseek-v4-flash-0731",
    instructions="Answer briefly.",
    input="法國的首都是哪裡？",
)

print(response.output_text)
```

## curl

```sh
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Authorization: Bearer local-key' \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

Add `"stream": true` to receive `text/event-stream` chunks. Add
`"stream_options": {"include_usage": true}` to receive a final usage chunk.

## Endpoints

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /api/status`
- `GET /api/settings`
- `PUT /api/settings`

The `/api/settings` endpoint changes `max_tokens`, `temperature`, and `top_p`
defaults in memory. A server restart restores the command-line defaults.

## Supported request fields

- `model`
- `messages` with text content, assistant `tool_calls`, and `role: tool` results
- `prompt` for text completions
- `max_tokens` and `max_completion_tokens`
- `temperature`
- `top_p`
- `stream`
- `stream_options.include_usage`
- `n`, when its value is `1`
- `thinking_mode`, with `chat` or `thinking`
- `tools`, with OpenAI function definitions
- `tool_choice`, with `auto`, `none`, `required`, or one named function

`/v1/responses` accepts a text `input` or an array of text message items. It
also accepts `instructions`, `max_output_tokens`, `reasoning.effort`, function
`tools`, Codex namespace tools, and `function_call_output` items. Set
`stream: true` to receive typed
Responses API events, including `response.output_text.delta`,
`response.function_call_arguments.delta`, and `response.completed`.

The response adds `reasoning_content` when `thinking_mode` is `thinking` and
the model finishes a reasoning block. This is a DeepSeek extension.

The server returns a tool request in `message.tool_calls`. The client must run
the function and send its result in a later `role: tool` message. The server
does not run functions or external commands.

When a request uses both `stream: true` and `tools`, the server sends text and
reasoning as the model generates them. It sends the function name when the
DeepSeek tool block starts. It then sends `arguments` as SSE fragments. The
official parser validates the complete response at the end. If validation
fails, the stream sends an error event and then `[DONE]`.

## Current limits

- The server supports one text generation at a time.
- Message content supports strings and OpenAI text content parts.
- Images, audio, `response_format`, `stop`, and logprobs are not
  supported.
- `/v1/responses` is stateless. It does not support `previous_response_id`,
  `conversation`, `store`, or `background`. Send earlier output items again in
  `input` when you continue a Tool call.
- `/v1/responses` supports function tools and Codex namespace tools. The server
  ignores hosted `web_search` declarations because the local runtime cannot
  execute them. It rejects other unsupported built-in tools.
- The server returns an OpenAI error object when a request uses an unsupported
  field.
- The request body limit is 1 MiB.
- The maximum requested output is 272,000 tokens.

`GET /api/status` reports both cumulative expert cache values and values for
the latest request. Request fields include the selected prefill step, whether
layer-major prefill ran, expert cache hits and misses, expert evictions, expert
bytes read, SSD read time, and routing synchronization time.

DeepSeek-V4 uses the official encoder stored in the installed model. The model
installer pins this file to the same revision as the model weights. The Runtime
checks its SHA-256 before it loads the file.
