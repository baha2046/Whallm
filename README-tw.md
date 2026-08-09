# DeepSeekV4SSD

[![English](https://img.shields.io/badge/English-Click-yellow)](README.md)
[![繁體中文](https://img.shields.io/badge/繁體中文-點擊查看-orange)](README-tw.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](README-cn.md)
[![日本語](https://img.shields.io/badge/日本語-クリック-青)](README-ja.md)
[![한국어](https://img.shields.io/badge/한국어-클릭-yellow)](README-ko.md)

DeepSeekV4SSD 讓 M 系列 Mac 使用約 30 GB 記憶體執行
`DeepSeek-V4-Flash-0731` 的全部 284B 參數，並從 SSD 串流 routed expert
（啟發自 [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare)）。

![DeepSeekV4SSD APP 畫面](docs/assets/deepseekv4ssd-app.png)

## Benchmark

測試機器是 MacBook Pro。MacBook Pro 配備 Apple M5 Pro、18 核心 CPU、
20 核心 GPU 和 64 GiB 統一記憶體。測試停用 DSpark。

| 測試 | Prefill | Decode | 峰值記憶體 |
| --- | ---: | ---: | ---: |
| Codex request，14,000 個 input token | 180 Tok/s | 6.5 Tok/s | 30 GB |
| 4,096-token prompt，生成 1 個 token | 144.53 Tok/s | — | 15.56 GiB |
| 短 prompt，同一個 runtime 第二次執行 | — | 6.41 Tok/s | 15.05 GiB |

前兩項測試分別使用 `v1.0.3` 和 `v1.0.2` runtime。prompt 內容、
SSD 速度和 cache 狀態會改變效能。[完整驗證紀錄](docs/VALIDATION.md)
包含完整測試資料。

## 如何使用

**下載 APP → 開啟 APP → 下載 167 GB 全參數模型 → 啟動 server →
在 APP 對話或連接 Codex**

1. 從 [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest)
   下載最新的 `DeepSeekV4SSD-macOS-arm64.zip`。
2. 解壓縮 ZIP。開啟 `DeepSeekV4SSD.app`。
3. 選擇「下載模型」。預設安裝包含 DSpark。installed model 約使用
   167 GB。使用者可以停止下載，之後再繼續下載。
4. 模型 ready 後，選擇「啟動 server」。
5. 使用 APP 的對話功能，或使用下方設定連接 Codex。

預設本機 server 位址是 `http://127.0.0.1:11434`。

## 使用需求

| 項目 | 需求 |
| --- | --- |
| Mac | Apple Silicon M 系列 Mac |
| macOS | macOS 15 或更新版本 |
| 統一記憶體 | 64 GiB 或更多 |
| 可用儲存空間 | 約 172 GB（160 GiB） |
| 模型儲存裝置 | 高速內接、Thunderbolt 或 USB4 SSD |
| 網路 | 下載模型和 APP 更新時需要網路 |

> [!IMPORTANT]
> DeepSeekV4SSD 是實驗性軟體。APP 不包含模型權重。除非其他裝置必須
> 連線，否則請保留預設本機 server 位址。

## Codex `config.toml` 配置

先在 DeepSeekV4SSD 啟動 server。然後把下列設定加入
`~/.codex/config.toml`：

```toml
model = "deepseek-v4-flash-0731"
model_provider = "deepseek-v4-ssd"
model_reasoning_effort = "high"

[model_providers.deepseek-v4-ssd]
name = "DeepSeekV4SSD"
base_url = "http://127.0.0.1:11434/v1"
wire_api = "responses"
requires_openai_auth = false
```

儲存檔案後，請重新啟動 Codex。本機位址不需要 API key。provider 設定
必須放在使用者層級的設定檔。[Codex 官方配置參考](https://developers.openai.com/codex/config-reference/)
包含其他設定。

## 其他技術細節

### 運作方式

- main model 具有 284B 總參數。每個 token 約會啟用 13B 參數。
- runtime 會把 common tensor 保留在統一記憶體中。
- routed expert 使用 checkpoint 原生 FP4 權重。runtime 會在需要時從
  SSD 讀取 routed expert。
- runtime 使用 FP8 KV cache 和有容量上限的 expert cache 控制記憶體用量。
- installed model 必須通過固定 checkpoint revision 的驗證。

### 模型儲存空間與 DSpark

- main model 約使用 145 GiB。
- DSpark 會增加約 10.12 GiB。預設下載會安裝 DSpark。
- 安裝 DSpark 不會啟用 DSpark。使用者可以在 runtime 設定中啟用
  「使用 DSpark」來測試 speculative decoding。
- 使用者可以移除 DSpark。移除 DSpark 不需要重新安裝 main model。

### OpenAI 相容 server

server 支援下列 endpoint：

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Responses API 支援 Codex tool 和 OpenAI function tool。API client 必須執行
tool，然後把結果傳回 server。[API 指南](docs/API.md)包含 request 欄位、
範例和目前限制。

### 指標與隱私

APP 會顯示 prefill 速度、decode 速度、token 數量、記憶體用量、SSD 讀取速度、
cache hit rate、第一個 token 等待時間和完成時間。

runtime 會在 Mac 上執行推論。prompt 和生成文字會保留在本機 runtime。
連接的 API client 仍可能把資料傳送到其他位置。APP 會使用網路下載模型、
檢查更新和接收已設定的 API request。

### 目前限制

- runtime 只支援固定 checkpoint revision 的 `DeepSeek-V4-Flash-0731`。
- runtime 一次只處理一個生成 request。
- API 不支援圖片、音訊、logprobs、`response_format` 和 `stop`。
- 很長的 input 和 output 需要更多 KV cache 記憶體。
- 效能取決於 SSD 速度、input 長度和 cache 狀態。

[運行時研究](docs/RUNTIME_RESEARCH_2026-08-07.md)和
[實作計畫](docs/IMPLEMENTATION_PLAN.md)包含模型合約、runtime 設計和已量測的
工程決策。

DeepSeekV4SSD 與 DeepSeek 沒有從屬關係。下載和使用模型前，請先閱讀模型條款。
