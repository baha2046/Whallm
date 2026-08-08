<div align="right">
  <a href="README.md">English</a> · <strong>繁體中文</strong>
</div>

# DeepSeekV4SSD

![DeepSeekV4SSD APP 畫面](docs/assets/deepseekv4ssd-app.png)

DeepSeekV4SSD 可在 Apple Silicon Mac 上執行 `DeepSeek-V4-Flash-0731`。
DeepSeekV4SSD 會把 common tensor 保留在統一記憶體中。DeepSeekV4SSD 會從高速
SSD 讀取 routed expert。

> [!IMPORTANT]
> DeepSeekV4SSD 是實驗性軟體。Mac 至少需要 64 GiB 統一記憶體。SSD 約需
> 160 GiB 可用空間。APP 不包含模型權重。

## 實測效能

下列結果使用 `v1.0.2` runtime，並於 2026-08-08 完成實測。測試機器使用
Apple M5 Pro、18 核心 CPU、20 核心 GPU 和 64 GiB 統一記憶體。runtime 使用
512 個主模型 slot、4 個 read worker、MXFP8 KV cache、batch size 1 和 greedy
decode。測試停用 DSpark。

| 測試 | Prefill | Decode | 第一個 token 等待時間 | 完成時長 | MLX 峰值記憶體 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5-token prompt，首次生成 32 個 token | — | 5.66 Tok/s | 2.12 s | 7.84 s | 15.05 GiB |
| 同一個 runtime 第二次執行相同 request | — | 6.41 Tok/s | 1.20 s | 6.28 s | 15.05 GiB |
| 4,096-token 重複 token prompt，生成 1 個 token | 144.53 Tok/s | — | 28.34 s | 28.34 s | 15.56 GiB |

測試 SSD 使用 4 個 read worker 時，direct read 速度是 14.30 GiB/s。這些結果
是參考值，不是效能保證。prompt 內容、SSD 速度和 cache 狀態都會改變效能。
[完整驗證紀錄](docs/VALIDATION.md)包含更多實測結果和測試細節。

## 功能

- APP 可以下載、安裝、驗證和修復支援的模型。
- APP 可以繼續中斷的模型下載。
- 預設模型資料夾是 `~/.dsmodel/`。使用者也可以選擇其他資料夾。
- APP 可以在 Mac 上啟動 OpenAI 相容 API server。
- API 支援 `/v1/responses`、`/v1/chat/completions` 和 `/v1/completions`。
- API 支援文字、推理內容和 function Tool call 的流式回覆。
- APP 會顯示 prefill、decode、token、記憶體、SSD 和 cache 指標。
- runtime 會在每個 routed expert ready 後立即開始計算。
- 每個指標會顯示即時值、最低值、平均值和最高值。
- APP 支援英文、簡體中文和繁體中文。
- APP 會透過 GitHub Releases 和 Sparkle 接收簽章更新。

## 系統需求

| 項目 | 需求 |
| --- | --- |
| Mac | Apple Silicon |
| macOS | macOS 15 或更新版本 |
| 統一記憶體 | 64 GiB 或更多 |
| 儲存空間 | 約 160 GiB 可用空間 |
| 模型儲存裝置 | 高速內接或外接 SSD |
| 網路 | 下載模型和檢查更新時需要網路 |

SSD 速度會直接影響 token 生成速度。外接模型 SSD 建議使用 Thunderbolt 或
USB4 連線。

## 安裝 APP

1. 從 [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest)
   下載最新的簽章與公證 ZIP。
2. 解壓縮 `DeepSeekV4SSD-macOS-arm64.zip`。
3. 把 `DeepSeekV4SSD.app` 移到「應用程式」資料夾。
4. 開啟 APP。

APP 會從 GitHub Releases 檢查更新。使用者也可以從 APP 選單選擇
「檢查更新⋯」。

## 安裝模型

1. 開啟 DeepSeekV4SSD。
2. 保留預設模型資料夾 `~/.dsmodel/`，或選擇其他資料夾。
3. 選擇是否要同時安裝 DSpark。
4. 選擇「下載模型」。
5. 等待下載和安裝完成。

主模型約使用 145 GiB。DSpark 會增加約 10.12 GiB。使用者可以停止下載，
之後再繼續下載。APP 會保留已驗證的部分資料。

如果使用者已經有 installed model，請選擇 installed model 資料夾或上一層
資料夾。APP 會自動偵測有效的 installed model。

## 啟動本機 server

1. 選擇 installed model。
2. 檢查 server 和 runtime 設定。
3. 選擇「啟動 Server」。
4. 使用測試對話，或連接其他 API client。

預設 server 位址是：

```text
http://127.0.0.1:11434
```

預設 OpenAI base URL 是：

```text
http://127.0.0.1:11434/v1
```

本機 server 不需要 API key。非本機 Host 必須設定 API key。請勿把 server
直接公開到網際網路。

## 呼叫 API

### Chat Completions

```sh
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "deepseek-v4-flash-0731",
    "messages": [
      {"role": "user", "content": "請簡短說明天空為什麼是藍色。"}
    ],
    "stream": true,
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.98
  }'
```

### Responses API

```sh
curl -N http://127.0.0.1:11434/v1/responses \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "deepseek-v4-flash-0731",
    "instructions": "請簡短回答。",
    "input": "請簡短說明天空為什麼是藍色。",
    "stream": true,
    "max_output_tokens": 256
  }'
```

支援的 endpoint 包含：

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

API 支援 OpenAI function tool。API client 必須執行 function，然後把結果傳回
server。server 不會執行 tool 或外部指令。

[API 指南](docs/API.md)包含 request 欄位、Python 範例、Codex tool 支援和目前
限制。

## DSpark

DSpark 是可選的 speculative decoding 模組。安裝 DSpark 會讓模型資料夾增加
約 10.12 GiB。安裝 DSpark 不會自動啟用 DSpark。

使用者可以在 runtime 設定中啟用「使用 DSpark」。預設 DSpark cache 使用
256 個獨立的 SSD slot。當 normal decode 較快時，runtime 可以在該 request
的後續生成停用 DSpark。

DSpark 的速度取決於 prompt、SSD 和 cache 狀態。DSpark 不會讓每個 request
都變快。停用 DSpark 不會刪除 DSpark 檔案。選擇「移除 DSpark」可以釋放
DSpark 使用的儲存空間。

## 指標

APP 會顯示測試對話和 API request 的下列指標：

- Prefill Tok/s
- Decode Tok/s
- Input Tokens
- Output Tokens
- Memory usage
- SSD read speed
- Cache Hit rate
- First Token wait time
- Completion time

每個指標會顯示即時值、最低值、平均值和最高值。歷史資料會記錄所有 request
的每秒取樣值。選擇「清除指標歷史」可以重設最低值、平均值和最高值。

## 隱私與網路連線

推論會在使用者的 Mac 上執行。prompt 和生成文字會留在本機 runtime。呼叫
API 的 client 可能會把資料傳送到其他位置。

APP 會在下列工作中使用網路：

- 從 Hugging Face 下載模型。
- 從 GitHub Releases 檢查和下載 APP 更新。
- 在使用者設定的 Host 和 Port 接收 API request。

除非其他裝置需要連線，否則請保留預設 Host `127.0.0.1`。使用非本機 Host
前，請設定強度足夠的 API key。

## 疑難排解

### 模型下載中斷

開啟相同的模型資料夾。選擇「繼續下載」。APP 會重用已驗證的部分資料。

### 模型遺失或損壞

選擇「驗證完整模型」。如果 APP 回報損壞檔案，請選擇「驗證並修復」。APP
只會重新下載遺失或損壞的資料。

### 第一個 token 等待很久

冷 expert cache 和很長的 input 會增加第一個 token 等待時間。cache 變熱後，
後續 request 可能會變快。請查看 Prefill Tok/s、SSD read speed 和 Cache Hit
rate。

### 記憶體用量太高

請先停止 server，再修改 runtime 設定。使用者可以降低 Prompt cache GiB、
降低 Prompt cache entries、停用 DSpark，或降低 DSpark slots。DSpark 至少
需要 30 個 slot。

### API request 等待很久

目前 runtime 一次只處理一個生成 request。後續 request 必須等待目前 request
完成。很長的 input 也會增加 prefill 時間。

### server 無法啟動

請驗證 selected model。請確認設定的 Port 可用。如果其他 APP 已使用 `11434`，
請修改 Port。

## 目前限制

- runtime 只支援固定 revision 的 `DeepSeek-V4-Flash-0731`。
- runtime 一次只處理一個生成 request。
- API 不支援圖片、音訊、logprobs、`response_format` 和 `stop`。
- request 的最大 output 是 272,000 tokens。
- 很長的 output 需要更多 KV cache 記憶體。
- 效能取決於 SSD 速度、input 長度和 cache 狀態。

DeepSeekV4SSD 與 DeepSeek 沒有從屬關係。下載和使用模型前，請先閱讀模型條款。
