# OpenAI 相容 API

server 預設監聽 `http://127.0.0.1:11434`。
server 載入一個 installed model。
server 一次只執行一個 generation request。

本 API 是相容子集。
本 API 不是 OpenAI API 的完整實作。

## 啟動 server

使用 APP：

```sh
make run
```

使用 command line：

```sh
DEEPSEEK_API_KEY=local-key make server
```

`MODEL`、`HOST` 和 `PORT` 可以由 Make 變數覆寫。

```sh
MODEL=/path/to/model.dsv4 HOST=127.0.0.1 PORT=11434 make server
```

## 驗證與網路邊界

本機 host 可以不設定 API key。
非本機 host 必須設定 `--api-key` 或 `DEEPSEEK_API_KEY`。

需要驗證的 request 使用：

```http
Authorization: Bearer local-key
```

| Endpoint | 設定 key 後是否驗證 |
| --- | --- |
| `/v1/*` | 是 |
| `GET /api/settings` | 是 |
| `PUT /api/settings` | 是 |
| `GET /` | 否 |
| `GET /healthz` | 否 |
| `GET /api/status` | 否 |

`/api/status` 會回傳本機 `model_path`。
server 不提供 TLS、CORS 或 rate limit。
請勿把 server 直接暴露到不受信任的網路。

## Endpoint

| Method | Path | 用途 |
| --- | --- | --- |
| `GET` | `/` | 回傳 server 名稱和 API base。 |
| `GET` | `/favicon.ico` | 回傳空的 `204` response。 |
| `GET` | `/healthz` | 回傳基本存活狀態。 |
| `GET` | `/v1/models` | 回傳目前公開 model ID。 |
| `POST` | `/v1/chat/completions` | Chat Completions 相容子集。 |
| `POST` | `/v1/responses` | Responses 相容子集。 |
| `POST` | `/v1/completions` | Text Completions 相容子集。 |
| `GET` | `/api/status` | 回傳 APP 和 profiling 使用的 runtime 狀態。 |
| `GET` | `/api/settings` | 讀取目前 generation 預設值。 |
| `PUT` | `/api/settings` | 修改目前 process 的 generation 預設值。 |

未知 route 回傳 `404`。

## 共用 request 欄位

| 欄位 | 規則 |
| --- | --- |
| `model` | 必須等於 server 的公開 model ID。預設值是 `deepseek-v4-flash-0731`。 |
| `max_tokens` | 1 至 272,000。預設值是 272,000。 |
| `temperature` | 0 至 2。預設值是 0.2。 |
| `top_p` | 0.000001 至 1。預設值是 0.98。 |
| `stream` | 必須是 boolean。 |
| `stream_options.include_usage` | 必須是 boolean。只影響 streaming response。 |
| `n` | 只接受 `1`。 |

`max_completion_tokens` 可以取代 Chat Completions 的 `max_tokens`。
`max_output_tokens` 可以取代 Responses 的 `max_tokens`。

server 明確拒絕下列 request：

- `response_format` 不是 `null` 或 `[]`。
- `stop` 不是 `null` 或 `[]`。
- 非零的 `frequency_penalty`。
- 非零的 `presence_penalty`。
- 任何 `seed`。
- `logprobs: true`。
- `echo: true`。
- `n` 不等於 `1`。

未列出的未知欄位可能被忽略。
client 不應依賴未知欄位。

## Chat Completions

### Request

`messages` 必須是非空 array。

支援下列 role：

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

最後一個 message 必須是 `user`、`developer` 或 `tool`。

`content` 可以是 string。
`content` 也可以是 text part array。
支援的 part type 是 `text`、`input_text` 和 `output_text`。

assistant message 可以包含 `tool_calls`。
tool message 必須包含對應的 `tool_call_id`。
client 必須先提供所有 tool result，才能要求下一個 response。

### Python 範例

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

### curl 範例

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

### Response

非 streaming response 使用 `chat.completion` shape。
response 包含 `choices` 和 `usage`。

thinking mode 完成後，assistant message 可以包含 `reasoning_content`。
`reasoning_content` 是 DeepSeekV4SSD extension。

## Responses

### 支援的 input

`input` 可以是 string。
`input` 也可以是非空 item array。

支援下列 item：

- text `message`
- `function_call`
- `function_call_output`
- `reasoning`

server 接受 replay 的 `reasoning` item，但不把該 item 加入 prompt。
client 必須在後續 `input` 中重送需要的歷史 message 和 tool item。

`instructions` 必須是 string。
server 會把 `instructions` 轉成 `developer` message。

### Python 範例

```python
response = client.responses.create(
    model="deepseek-v4-flash-0731",
    instructions="Answer briefly.",
    input="法國的首都是哪裡？",
    reasoning={"effort": "high"},
)

print(response.output_text)
```

### Stateless 限制

Responses endpoint 不保存 server-side conversation。

server 拒絕：

- 非空的 `previous_response_id`。
- 非空的 `conversation`。
- `store: true`。
- `background: true`。
- `truncation` 不是 `disabled`。
- `text.format` 不是 plain text。

`metadata` 會原樣放入 response。
`parallel_tool_calls` 只接受 boolean。
該欄位不會讓本機 generation 並行。

## Text Completions

`POST /v1/completions` 需要 string `prompt`。
response 使用 `text_completion` shape。

Text Completions 不支援 `tools`。
Text Completions 只接受 `tool_choice: none` 或省略該欄位。

## Reasoning effort

Chat Completions 使用 `reasoning_effort`。
Responses 使用 `reasoning.effort`。

| API effort | 預設 thinking mode | DeepSeek encoder effort |
| --- | --- | --- |
| 省略、`none` | `chat` | `low` |
| `minimal`、`low`、`medium` | `thinking` | `low` |
| `high` | `thinking` | `high` |
| `xhigh`、`max` | `thinking` | `max` |

`thinking_mode` 可以是 `chat` 或 `thinking`。
明確的 `thinking_mode` 只覆寫 mode。
server 仍會驗證並傳送 encoder effort。

## Tool

### Chat Completions tool

Chat Completions 支援 OpenAI function tool shape。

`tool_choice` 支援：

- `auto`
- `none`
- `required`
- 指定一個 function

function name 必須使用 1 至 64 個字母、數字、底線或連字號。
function name 在同一個 request 中必須唯一。

### Responses tool

Responses 支援：

- top-level function tool
- Codex namespace tool
- `function_call_output`

server 會忽略 `web_search` tool declaration。
server 不會執行 web search。
server 會拒絕其他不支援的 built-in tool。

server 只產生 tool request。
client 必須執行 tool。
client 必須把 tool result 傳回 server。

## Streaming

設定 `stream: true` 後，server 使用
`text/event-stream; charset=utf-8`。

Chat Completions 會傳送 `chat.completion.chunk`。
Text Completions 會傳送 `text_completion` chunk。
Chat Completions 和 Text Completions 的最後一個 SSE frame 是：

```text
data: [DONE]
```

設定 `stream_options.include_usage: true` 後，Chat Completions 會在結束前
傳送 usage chunk。

Responses 會傳送 typed event。
主要 event 包含：

- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.output_text.delta`
- `response.function_call_arguments.delta`
- `response.output_item.done`
- `response.completed`

Responses 不傳送 `[DONE]`。
server 會在 `response.completed` 或 `error` 後關閉連線。

tool streaming 會在 generation 過程中傳送 function name 和 arguments fragment。
server 會在 generation 結束時驗證完整 tool block。
Chat Completions 驗證失敗時會傳送 error object，然後傳送 `[DONE]`。
Responses 驗證失敗時會傳送 `error` event，然後關閉連線。

## APP 設定 API

`GET /api/settings` 回傳：

```json
{
  "max_tokens": 272000,
  "temperature": 0.2,
  "top_p": 0.98
}
```

`PUT /api/settings` 可以更新一個或多個欄位。
server 會拒絕未知設定。
設定只存在目前 process 的記憶體。
server restart 會還原 command-line 預設值。

## Status API

`GET /api/status` 回傳三類資料。

| 區域 | 內容 |
| --- | --- |
| root | model ID、checkpoint model ID、installed model path 和 key 狀態。 |
| `runtime` | slot、worker、省電模式、prefill、cache 和 DSpark 設定。 |
| `performance` | generation、時間、記憶體、SSD 和 expert cache 指標。 |

`performance` 同時包含累計值和最近一次 request 值。
欄位語意請見[效能與瓶頸](PERFORMANCE.md)。

`runtime.power_saving_limit_gbps` 是 0.5、1、2、3、5、10、25 或 `null`。
`null` 代表 routed expert SSD 讀取速度沒有限制。

## Error

錯誤使用 OpenAI error object shape。

```json
{
  "error": {
    "message": "model must be 'deepseek-v4-flash-0731'.",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

常見 status code：

| Status | 條件 |
| ---: | --- |
| 400 | JSON、欄位或 request 狀態無效。 |
| 401 | Bearer API key 無效。 |
| 404 | route 不存在。 |
| 411 | 缺少 `Content-Length`。 |
| 413 | request body 超過限制。 |
| 500 | runtime 或 model output 發生內部錯誤。 |

## 限制

- request body 上限是 1 MiB。
- request 必須包含 `Content-Length`。
- server 不支援 chunked request body。
- requested output 上限是 272,000 token。
- output 上限不是已驗證 context 長度。
- API 只支援文字。
- API 不支援 image、audio、logprobs、stop 和 structured output。
- server 一次只執行一個 generation request。
- server 不執行 tool、web search 或外部 command。
