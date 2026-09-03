# OpenAI 相容 API

server 預設監聽 `http://127.0.0.1:11434`。
server 啟動時讀取初始 model catalog。
APP 可以更新未載入模型的 model catalog entry。
server 啟動時不載入模型權重。
第一個 generation request 會載入指定的 installed model。
client 也可以明確載入或卸載 installed model。
server 一次只保留一個載入的 installed model。
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

`--model` 是選用參數。
`--model-catalog` 也是選用參數。
兩個參數不能同時使用。
server 可以在沒有 installed model 的狀態下啟動。

APP 使用版本化 JSON model catalog 啟動 server。
下列範例包含一個 installed model：

```json
{
  "version": 1,
  "models": [{
    "id": "deepseek-v4-flash-0731",
    "alias": "work-model",
    "path": "/path/to/model.dsv4",
    "model_kind": "deepseek-v4",
    "runtime": {},
    "defaults": {
      "max_tokens": 272000,
      "temperature": 0.2,
      "top_p": 0.98,
      "top_k": 0
    },
    "warmup_prompt_path": null
  }]
}
```

實際的 `runtime` object 必須包含全部 `RuntimeConfig` snake-case 欄位。
`--public-model` 只適用於舊的 `--model` 流程。
該參數會設定該 installed model 的 Alias。

固定的 API model ID 如下：

- `deepseek-v4-flash-0731`
- `qwen3.8-flash-next-fp8`

server 會移除 Alias 前後的空白。
空字串代表沒有 Alias。
Alias 可以等於自己的 API model ID。
Alias 不可等於其他模型的 API model ID 或 Alias。

研究用途可以使用 `--expert-file-cache-policy bypass` 啟動 Python server。
正式使用和 APP 使用預設值 `cached`。
`bypass` 只套用到 expert-file descriptor。
`bypass` 不會清除整個系統的 cache。

Atomic DSpark prefix reuse 是另一個預設停用的研究設定。
啟用時必須同時使用 `--dspark --dspark-prompt-cache`。
server 會拒絕沒有 `--dspark` 的 `--dspark-prompt-cache`。
APP 預設不會啟用此設定。

Qwen MTP 是預設停用的 speculative decoding prototype。
直接啟動 server 時，可以使用 `--mtp`。
Model catalog 可以設定 `mtp_enabled=true` 和 `mtp_slots>=10`。
MTP 支援 Qwen 的 Temperature、Top P、Top K、Min P 和 logit processor。
Installed model 必須包含 MTP sidecar。
Qwen3.8 的 Advanced Settings 可以啟用 MTP。
APP 只會在 installed model 包含 MTP sidecar 時啟用 MTP。

## 驗證與網路邊界

本機 host 可以不設定 API key。
非本機 host 必須設定 `--api-key` 或 `DEEPSEEK_API_KEY`。

server 的 `--log-level` 支援 `debug`、`info` 和 `error`，預設為 `info`。
`info` 顯示一般 request access log；`error` 只顯示 HTTP 4xx／5xx request；`debug`
除了 access log，還會把每個已解析 JSON request body 完整寫到 Server Log。
Debug 不會記錄 Authorization header，但 JSON 仍可能包含 prompt、檔案路徑、tool result
或其他敏感內容。

需要驗證的 request 使用：

```http
Authorization: Bearer local-key
```

| Endpoint | 設定 key 後是否驗證 |
| --- | --- |
| `/v1/*` | 是 |
| `POST /api/models/configure` | 是 |
| `POST /api/models/load` | 是 |
| `POST /api/models/unload` | 是 |
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
| `GET` | `/v1/models` | 回傳目前 model catalog 內的 API model ID 和 Alias。 |
| `POST` | `/v1/chat/completions` | Chat Completions 相容子集。 |
| `POST` | `/v1/responses` | Responses 相容子集。 |
| `POST` | `/v1/completions` | Text Completions 相容子集。 |
| `POST` | `/api/models/configure` | 更新未載入模型的 model catalog entry。 |
| `POST` | `/api/models/load` | 載入指定模型。必要時先卸載目前模型。 |
| `POST` | `/api/models/unload` | 卸載指定模型。 |
| `GET` | `/api/status` | 回傳 APP 和 profiling 使用的 runtime 狀態。 |

未知 route 回傳 `404`。
`GET /api/settings` 和 `PUT /api/settings` 也回傳 `404`。

`GET /v1/models` 不會載入模型權重。
OpenAI 相容的 `data` array 中，每個 installed model 先列出 API model ID。
如果 Alias 與 API model ID 不同，server 接著列出 Alias。
兩個項目使用相同的 `owned_by`。

同一個 response 也包含 Codex model metadata 使用的 `models` array。
每個 API model ID 和 Alias 都有對應的 `slug`，並宣告 context window、text input、
local shell tool 與 base instructions。Codex provider 的 base URL 含 `/v1` 時，
`GET /v1/models?client_version=...` 可直接反序列化這個 array，不需使用 fallback
model metadata。額外的 `models` 欄位不改變標準 `data` array。

## 模型設定、載入與卸載

`POST /api/models/load` 和 `POST /api/models/unload` 使用下列 JSON body：

```json
{
  "model": "deepseek-v4-flash-0731"
}
```

`model` 可以是 API model ID 或 Alias。
`POST /api/models/load` 可以加入 `configuration`。
`configuration` 必須是該模型的完整 model catalog entry。
server 會先驗證並更新 entry，再載入模型。

`POST /api/models/configure` 使用下列 JSON body：

```json
{
  "configuration": {
    "id": "deepseek-v4-flash-0731",
    "alias": "work-model",
    "path": "/path/to/model.dsv4",
    "model_kind": "deepseek-v4",
    "runtime": {},
    "defaults": {
      "max_tokens": 272000,
      "temperature": 0.2,
      "top_p": 0.98,
      "top_k": 0
    },
    "warmup_prompt_path": null
  }
}
```

實際的 `runtime` object 必須包含全部 `RuntimeConfig` snake-case 欄位。
Loaded 或 Loading 的模型不能更新 entry。
未載入模型的更新會在下次載入時生效。
三個模型管理 endpoint 成功時都回傳與 `GET /api/status` 相同的資料。
載入另一個模型時，server 會先關閉目前的 runtime。
卸載不是目前已載入的模型時，server 不會變更目前的 runtime。
載入和卸載會等待目前的完整 streaming request 結束。

## 共用 request 欄位

| 欄位 | 規則 |
| --- | --- |
| `model` | 必須是目前 model catalog 內的 API model ID 或 Alias。名稱比對區分大小寫。 |
| `max_tokens` | 1 至 272,000。預設值是 272,000。 |
| `temperature` | 0 至 2。DeepSeek 預設 0.2。Qwen 依模式使用 0.7 或 1.0。 |
| `top_p` | 0.000001 至 1。DeepSeek 預設 0.98。Qwen 依模式使用 0.8 或 0.95。 |
| `top_k` | 0 或正整數。DeepSeek 預設 0。Qwen 兩種模式都使用 20。 |
| `approximation` | 選用 object。一般 DeepSeek 未提供時使用 `learned-route-drop-lowest-1`。Qwen 和 DSpark 使用 `exact`。 |
| `stream` | 必須是 boolean。 |
| `stream_options.include_usage` | 必須是 boolean。只影響 streaming response。 |
| `n` | 只接受 `1`。 |

`max_completion_tokens` 可以取代 Chat Completions 的 `max_tokens`。
`max_output_tokens` 可以取代 Responses 的 `max_tokens`。

### DeepSeek approximate mode

一般 DeepSeek request 預設使用 `learned-route-drop-lowest-1`。
Client 不需要加入額外欄位。
Response 會明確回報實際 mode。

Client 也可以明確指定這個 mode：

```json
{
  "approximation": {
    "mode": "learned-route-drop-lowest-1"
  }
}
```

這個 mode 只在單次 request 內把 40 個 learned-routing layers 從 top-6 改為
top-5。
三個 hash-routing layers 保持 top-6。
Request 完成或失敗後，runtime 會還原 learned router。

Client 可以明確切回 exact mode：

```json
{
  "approximation": {
    "mode": "exact"
  }
}
```

Qwen 和啟用 DSpark 的 DeepSeek 在沒有 `approximation` 欄位時使用 exact mode。
`approximation` 必須只包含 string `mode`。
未知 mode 會回傳 HTTP 400。
Qwen 和 DSpark 不支援 approximate mode。

直接使用 Python CLI 時，一般 DeepSeek 也預設使用 approximate mode。
下列參數可以切回 exact：

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt "Hello" \
  --approximation exact
```

Exact 和 approximate memory prompt cache 使用不同 mode key。
Exact request 不會重用 approximate cache。
Approximate cache 不會寫入 persistent prompt cache。
Persistent prefill checkpoint 是當下 cache state 的 immutable snapshot；後續 generation
不會改寫它。只剩一個 prompt token 時仍可安全 replay。

所有一般 response 和 streaming event 都包含實際的
`approximation.mode`。
這個欄位是 Whallm extension。
它不是 OpenAI API 標準欄位。

### Qwen 模式取樣

Whallm 只對 Qwen 套用模式取樣配置。
參數來自 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices)。

| 模式 | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 思考 | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| 非思考 | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |

Qwen 使用 OR 規則判定模式。

- `thinking_mode` 是 `thinking` 時，server 使用思考模式。
- Chat Completions 的 `reasoning_effort` 不是 `none` 時，server 使用思考模式。
- Responses 的 `reasoning.effort` 不是 `none` 時，server 使用思考模式。
- 其他 request 使用非思考模式。
- Text Completions 固定使用非思考模式。

`thinking_mode: "chat"` 不會抵銷非 `none` 的 reasoning effort。

明確的 `temperature`、`top_p` 和 `top_k` 會覆寫模式配置。
`max_tokens` 的優先序不變。
DeepSeek 仍使用 model catalog 預設值。

正數 `temperature` 會使用 categorical sampling。
明確的 `temperature: 0` 會使用 greedy sampling。

`min_p`、`presence_penalty` 和 `repetition_penalty` 是 Qwen 內部配置。
API 不支援公開的 `min_p` 或 `repetition_penalty` request 欄位。
API 仍會拒絕非零的 `presence_penalty` request 欄位。
Request 內的 `presence_penalty: 0` 不會停用 Qwen 內部配置。

目前 [pinned `mlx-lm` 的 presence processor](https://github.com/Blaizzy/mlx-lm/blob/5c10538136b9038b9626c134612b08afc18d697a/mlx_lm/sample_utils.py#L315-L338)
只檢查 generation call
可見的最近 20 個 token。
processor 不會檢查 prompt cache 已重用的 prefix。
因此，這個 processor 不會檢查完整 prompt。
`repetition_penalty=1.0` 是中性值。
Whallm 不會為這個值建立 processor。

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

所有一般 response 和 streaming event 的 `model` 欄位會回傳 request 使用的名稱。
所有 generation response 和 streaming event 的 `approximation.mode` 會回傳實際 mode。
API model ID 和 Alias 會共用同一個 runtime。
切換 API model ID 時，server 會先關閉舊 runtime，然後載入新 runtime。
其他 generation request 會等待目前的完整 streaming request 結束。
`GET /healthz`、`GET /v1/models` 和 `GET /api/status` 不會等待 generation lock。

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
`reasoning_content` 是 Whallm extension。

## Responses

### 支援的 input

`input` 可以是 string。
`input` 也可以是非空 item array。

支援下列 item：

- text `message`
- `function_call`
- `function_call_output`
- `reasoning`

`message.content` 可以是 string。
`message.content` 也可以是 `text`、`input_text` 或 `output_text` item array。
server 會依原順序合併 array 中的文字。
server 會拒絕 image、file 和其他非文字 content item。

text `message` 支援 `system`、`developer`、`user` 和 `assistant` role。
Qwen codec 會把所有 `system` 和 `developer` message 合併成一個開頭的
`system` message。

`function_call.arguments` 必須是 JSON string。
該 JSON string 必須包含一個 object。
`function_call_output.output` 必須是 string。
`function_call` 和對應的 `function_call_output` 必須使用相同的 `call_id`。

server 接受 replay 的 `reasoning` item，但不把該 item 加入 prompt。
client 必須在後續 `input` 中重送需要的歷史 message 和 tool item。
`input` 可以用 `assistant` message 作為最後一個 item。
這可讓 client 重送先前的 model output。

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
Qwen Text Completions 固定使用非思考模式取樣配置。
`thinking_mode` 和 reasoning effort 不會變更這項行為。

## Reasoning effort

Chat Completions 使用 `reasoning_effort`。
Responses 使用 `reasoning.effort`。

| API effort | 預設 thinking mode | DeepSeek encoder effort | Qwen effort |
| --- | --- | --- | --- |
| 省略、`none` | `chat` | `low` | `low` |
| `minimal`、`low` | `thinking` | `low` | `low` |
| `medium` | `thinking` | `low` | `medium` |
| `high` | `thinking` | `high` | `xhigh` |
| `xhigh`、`max` | `thinking` | `max` | `xhigh` |

`thinking_mode` 可以是 `chat` 或 `thinking`。
Qwen 使用前述 OR 規則。
DeepSeek 保留既有行為。
DeepSeek 的明確 `thinking_mode` 會覆寫 mode。
server 仍會驗證並傳送 encoder effort。

Codex 的 `model_reasoning_effort` 是 Responses API 設定。
Codex client 會把該設定傳成 Responses 的 `reasoning.effort`。
Whallm 不新增 `model_reasoning_effort` HTTP 欄位。
格式請見 [OpenAI Codex 設定參考](https://developers.openai.com/codex/config-reference/)
和 [Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)。

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

Codex namespace 與 function name 合併後如果超過 64 個字元，server 會在
Qwen／DeepSeek prompt 內使用固定的 64 字元 hash Alias。
Streaming event、完整 response 和後續 `function_call` replay 仍使用 client 原本的
`namespace` 和 `name`。
這可避免新版 Codex 內建 app tools 因完整名稱過長而讓整個 request 在 generation
之前失敗。

server 會忽略 `web_search` tool declaration。
server 不會執行 web search。
server 會拒絕其他不支援的 built-in tool。

server 只產生 tool request。
client 必須執行 tool。
client 必須把 tool result 傳回 server。

Qwen 的 Codex tool-first 模式不需要新增 request 欄位。
當 Responses request 同時符合下列條件時，server 會把這一輪的 `auto`
視為 `required`：

- model 是 Qwen。
- tools 包含 Codex 的 `exec_command`。
- input 尚未包含 `function_call_output`。

client 傳回 `function_call_output` 後，後續輪維持 `auto`，因此 model 可以繼續呼叫
tool，也可以輸出最終答案。一般 function tools 不會啟用 Codex tool-first 模式。

Responses 的 `required` 和指定 function 都會驗證完整 generation 結果。
`required` 至少要產生一個 tool call；指定 function 必須產生該 function 的 call。
第一次不符合時，server 只會重試一次。重試會使用相同 request 和取樣設定，並加入
一段要求立即呼叫 tool 的 developer instruction。第一次的無效文字不會傳給 client。

第二次仍沒有符合要求的 call 時，response 以 `tool_choice_not_satisfied` 失敗。
第二次無法解析 tool call 時，response 以 `invalid_tool_call` 失敗。server 不會進行
第三次 generation。

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
server 會在 `response.completed` 後關閉連線。

Responses 的 tool streaming 會先保留單次 generation 的輸出，並在 generation 結束時
驗證完整 tool block。驗證後的完整 parser 結果是 SSE 的唯一依據，不會再交給
streaming parser 重複判定。這同時套用 `auto`、`required`、指定 function 和 Qwen
Codex tool-first 首輪。
驗證通過後才傳送該次的文字、function name 和 arguments；若 `required` 或指定
function 的第一次結果不符合，client 只會看到第二次的有效結果。
DeepSeek 使用 DSML。Qwen 使用官方 XML tool-call 格式。
Chat Completions 驗證失敗時會傳送 error object，然後傳送 `[DONE]`。
Responses 驗證失敗時會傳送 `error` event。
接著，server 會傳送一段可見的錯誤文字和 `response.completed`，然後關閉連線。
最後一個 response 的 `status` 是 `failed`。
`error.code` 是 `invalid_tool_call` 或 `tool_choice_not_satisfied`。
這個終止方式用於目前 Codex client 相容性。

## Status API

`GET /api/status` 回傳三類資料。

| 區域 | 內容 |
| --- | --- |
| root | 載入中與已載入的 API model ID、checkpoint model ID、installed model path 和 key 狀態。 |
| `runtime` | slot、worker、省電模式、prefill、cache 和 DSpark 設定。 |
| `performance` | generation、時間、記憶體、SSD 和 expert cache 指標。 |

`performance` 同時包含累計值和最近一次 request 值。
`performance.approximation_mode` 是最近一次或目前 request 的實際 mode。
`accumulated_generation_tokens` 是目前 server process 產生的 output token 總數。
`completed_request_count` 和 `accumulated_generation_tokens` 不會因模型切換而歸零。
欄位語意請見[效能與瓶頸](PERFORMANCE.md)。

`loaded_model` 是已載入的 API model ID。
`loading_model` 是載入中的 API model ID。
兩個欄位都是 nullable。
沒有載入模型時，`model`、`source_model`、`model_path` 和 `runtime` 是 `null`。
此時 `performance` 保留固定 shape，並使用零值。

`runtime.power_saving_limit_gbps` 是 0.5、1、2、3、5、10、25 或 `null`。
`runtime.layer_major_prefill_threshold` 是 DeepSeek 啟用 layer-major Prefill 所需的最少未快取 token 數。
預設值是 1,024。
`runtime.ane_prefill` 表示 Qwen 是否要求 private ANE Prefill 路線。
Qwen App catalog 的預設值是 `true`。
`runtime.ane_prefill_ratio` 是分配給 ANE 的 `q_proj` output channels 比例。
值域是 0 到 1，預設值是 0.25。
Runtime 會把比例對齊到 256 個 output channels。
`runtime.ane_prefill_status.active` 表示目前 loaded model 的 ANE projection 可用。
`requested_ratio` 和 `active_ratio` 分別表示設定比例與對齊後比例。
`ane_channels` 和 `gpu_channels` 表示實際 output channels 分配。
`error` 保存第一次 private interface 錯誤。
`evaluations` 和 `fallbacks` 是目前 loaded model 的累計值。
ANE 錯誤不會讓 request 失敗；runtime 會改用原本的 GPU Prefill。
`null` 代表 routed expert SSD 讀取速度沒有限制。
`runtime.expert_page_cache_probe` 表示 research-only pre-read `mincore` probe 是否啟用；
預設為 `false`。`performance.request_expert_page_cache_*` 將本次 logical expert reads
分為讀取前 resident、nonresident 與 unclassified bytes，並回報 calls／failures。
Nonresident 是 expert-file-specific page-cache-miss proxy，不是 physical SSD bytes。
`performance.dspark_draft_page_cache_*` 與
`performance.dspark_hash_prefetch_*_page_cache_*` 分別保存 draft 與 exact hash-prefetch
useful／wasted partition。Probe 本身會改變 timing，不應在服務模式預設開啟。
`runtime.expert_file_cache_policy` 是 `cached` 或 `bypass`；預設為 `cached`。
`runtime.expert_file_direct_io_alignment_bytes` 是目前 expert descriptors 要求的
alignment，cached mode 為 0，本次 APFS bypass mode 為 4,096。Darwin bypass mode
會對 main 與 DSpark expert descriptors 設定 `F_NOCACHE` 並停用 read-ahead；它不清除
啟用前已 resident 的 pages。若 destination、offset 或 iovec length 不符合 alignment，
runtime 會拒絕該 read。
`performance.request_staged_expert_reads`、`request_staged_w13_bytes_read`、
`request_staged_w2_bytes_read`、`request_staged_read_seconds`、
`request_staged_w2_wait_seconds` 與 `request_staged_first_stage_submit_seconds`
只供 stopped split-slot research prototype 計帳。一般 server 全部回傳 0；本 API、CLI
與 APP 都沒有啟用 `staged_expert_streaming` 的介面。First-stage submit 是 CPU graph
submission wall，不是 GPU kernel duration。
`performance.request_adaptive_prefill_planned_layers`、
`request_adaptive_prefill_full_layers`、`request_adaptive_prefill_selective_layers`、
`request_adaptive_prefill_union_experts`、`request_adaptive_prefill_read_experts`、
`request_adaptive_prefill_bytes_read`、`request_adaptive_prefill_avoided_bytes` 與
`request_adaptive_prefill_plan_seconds` 只供 stopped adaptive prefill research
prototype 計帳。一般 server 全部回傳 0；本 API、CLI 與 APP 都沒有啟用 internal
`adaptive_expert_prefill_threshold` 的介面。Adaptive batched bytes 只在 read futures
成功後累加，且已包含於 `request_expert_bytes_read`；avoided bytes 是相對 42 層
full-layer logical budget，不是 physical SSD bytes。
`runtime.dspark_prompt_cache` 表示 atomic target-KV + DSpark-context prefix
reuse 是否啟用，預設為 `false`，且需要 DSpark。它使用獨立 format-3 namespace，不會讀取
一般 target-only prompt entries。`performance.dspark_prompt_cache_source` 回報最近一次
request 的 `disabled`、`none`、`memory` 或 `persistent`；啟用 gate 只把後三者視為 reuse
contract 狀態。`performance.prompt_cache_reused_tokens` 同時回報實際重用的 prefix 長度。
一般 target-only persistent cache 預設使用 format 5；它會驗證 model／RoPE／KV／attention
contract 與 content-addressed token-block chain，並可在 suffix 分岔時重用已保存的 bounded
prefill checkpoint。這不需要額外 API flag。舊 normal format 1／2／4 不再載入；format 4
失效可避免升級後重用曾被後續 generation 改寫的 Qwen checkpoint。
`runtime.dspark_hash_prefetch` 表示實驗性 exact prefetch 是否啟用；
`runtime.dspark_hash_prefetch_scratch_slots` 是 main model verification scratch 的
expert blob slot 數，停用時為 0。`runtime.dspark_adaptive_block` 表示 storage-aware
DSpark verification prefix selector 是否啟用。兩個實驗開關都預設為 `false`，也都
需要 DSpark；adaptive selector 另要求 installed model 的 target hash layers。
`runtime.dspark_fallback_enabled` 預設為 `true`。`--no-dspark-fallback` 只供隔離的
研究控制使用；它不代表建議的服務設定，也不會把 would-trigger round 變成效能證據。
`runtime.dspark_sequential_verification` 表示逐 token target verification oracle 是否啟用；
預設為 `false`，只供 correctness diagnosis。`--dspark-sequential-verification` 需要
DSpark，且不能與 `--dspark-hash-prefetch` 同時使用，避免 speculative scratch 改變
ready-expert execution path。
`runtime.dspark_hybrid_verification` 表示 token-shaped hybrid target verifier 是否啟用；
預設為 `false`，需要 DSpark，且與 sequential oracle 互斥。Hybrid 可與 hash prefetch
或 adaptive selector 組合；它逐 token 執行 target math，但每層只 acquire 一次
expert union。

`performance.dspark_block_verification_rounds`、
`performance.dspark_sequential_verification_rounds` 與
`performance.dspark_hybrid_verification_rounds` 分開記錄 verifier mode。
`performance.dspark_hybrid_attention_layers`、
`performance.dspark_hybrid_attention_token_calls`、
`performance.dspark_hybrid_ffn_token_calls`、
`performance.dspark_hybrid_moe_token_calls` 與
`performance.dspark_last_hybrid_verification_positions` 證明 hybrid token-shaped path
實際執行。`performance.dspark_last_verification_mode`、
`performance.dspark_last_sequential_position_seconds` 與 round trace 另保存最近一輪
mode、sequential／hybrid position 數。

`performance.dspark_verification_expert_union_calls` 是本次 request 的 target
verification／replay `get_many` acquisition 總次數。它必須和
`dspark_verification_routed_expert_assignments`、
`dspark_verification_expert_union_experts`、reuse、misses、target bytes 與 read time
一起解讀。Calls 減少只證明 acquisition batching；不等於 expert bytes 或 target math
一定更快。

## Error

錯誤使用 OpenAI error object shape。

```json
{
  "error": {
    "message": "The model 'unknown-model' does not exist.",
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
| 500 | 模型載入、runtime 或 model output 發生內部錯誤。 |

未知模型使用 `model_not_found`。
模型載入失敗使用 `model_load_failed`。
載入失敗後，下一個 request 可以再次嘗試載入模型。

## 限制

- request body 上限是 1 MiB。
- request 必須包含 `Content-Length`。
- server 不支援 chunked request body。
- requested output 上限是 272,000 token。
- output 上限不是已驗證 context 長度。
- API 只支援文字。
- Qwen 不支援 vision、video 或 DSpark。
- Qwen MTP prototype 支援 greedy 和 categorical sampling，且預設停用。
- Sampled MTP 尚未取得完整 installed model 的正式效能結果。
- API 不支援 image、audio、logprobs、stop 和 structured output。
- server 一次只執行一個 generation request。
- server 不執行 tool、web search 或外部 command。
