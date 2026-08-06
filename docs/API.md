# OpenAI-compatible server

The server keeps one installed model resident. It processes one generation
request at a time. This matches the current batch size 1 runtime.

## Start the macOS app

```sh
cd /Users/Shared/Project/test/llm_ssd
make run
```

Use the SwiftUI app to select the installed model, configure the runtime, and
start the server. The default address is `http://127.0.0.1:8000`.

Start the server without the app when needed:

```sh
DEEPSEEK_API_KEY=local-key make server
```

The server allows a local address without an API key. The server requires
`--api-key` or `DEEPSEEK_API_KEY` when `--host` is not local.

## OpenAI Python client

Install the OpenAI client in the environment that calls the server:

```sh
python -m pip install openai
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
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

## curl

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
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
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /api/status`
- `GET /api/settings`
- `PUT /api/settings`

The `/api/settings` endpoint changes `max_tokens`, `temperature`, and `top_p`
defaults in memory. A server restart restores the command-line defaults.

## Supported request fields

- `model`
- `messages` with text content
- `prompt` for text completions
- `max_tokens` and `max_completion_tokens`
- `temperature`
- `top_p`
- `stream`
- `stream_options.include_usage`
- `n`, when its value is `1`
- `thinking_mode`, with `chat` or `thinking`

The response adds `reasoning_content` when `thinking_mode` is `thinking` and
the model finishes a reasoning block. This is a DeepSeek extension.

## Current limits

- The server supports one text generation at a time.
- Message content supports strings and OpenAI text content parts.
- Images, audio, tools, `response_format`, `stop`, and logprobs are not
  supported.
- The server returns an OpenAI error object when a request uses an unsupported
  field.
- The request body limit is 1 MiB.
- The maximum requested output is 32,768 tokens.

DeepSeek-V4 uses its official special-token message format because the pinned
tokenizer has no Jinja chat template.
