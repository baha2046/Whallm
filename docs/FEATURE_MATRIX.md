# 命令列與 UI 功能對照

核對日期：2026-09-13。基準為 `develop`（`0204409`）加目前未提交修改，
不是已發布的 v1.1.6 成品。表格依目前程式入口逐項核對，不代表所有模型都已完成實機驗證。

- **CLI**：`python -m deepseek_v4_ssd.cli`，直接生成一次。
- **Server**：`python -m deepseek_v4_ssd.server`，啟動 HTTP 服務。
- **JSON**：可透過 model catalog 或模型 configure API 設定，沒有同名命令列旗標。
- **API**：可用 `curl` 呼叫；不代表有獨立 CLI 子命令。
- UI 欄標示「固定」代表程式有使用該功能，但沒有可操作的控制項。
- 模型進階設定在下次載入生效；Loaded／Loading 模型不可修改。其他未載入模型可修改。

## 本次變更

Throughput 頁面的測試流程、數據定義與限制見 [Throughput](THROUGHPUT.md)。

| 功能 | 命令列 | UI | 預設／行為 |
| --- | --- | --- | --- |
| 長輸入前釋放舊 expert Slots | CLI／Server／Throughput 共用 runtime | 自動；無新開關 | V4 batched layer-major 與 Qwen layer-major 在建立輸入暫存前釋放；保留模型和 Prompt Cache；V4.1 無相同重疊配置。原始碼已修正，尚未發布 |
| 吞吐量測試 | API：`POST /api/benchmark/throughput` | Throughput | Code／Novel 下拉選單；不重複補長的內建素材（gzip 合計約 698 KB），三種 tokenizer 均覆蓋 200K；自動載入模型，整輪結束後 App 自動卸載（原始碼已實作，尚未發布）；1K–200K 輸入多選；128／1024／4096 輸出上限；逐筆結果與取消；結果保留當次 slots；可複製純文字／JSON／Markdown 表格 |
| 關閉跨請求快取 | CLI／Server：`--prompt-cache off` | Model → Advanced Settings → Prompt cache → 不使用 | 每次重新處理輸入；單次生成仍需 KV state |
| 記憶體快取 | CLI／Server：`--prompt-cache memory` | 同上 → 記憶體 | **新預設**；不建立、讀取或寫入磁碟快取 |
| 磁碟快取 | CLI／Server：`--prompt-cache disk` | 同上 → 磁碟 | 記憶體重用加磁碟保存，可跨重啟恢復 |
| catalog 與 CLI 同時設定 | Server：明確 CLI 參數優先 | App 產生各模型的 JSON | CLI 沒給的欄位保留 JSON；覆寫後重新驗證 |
| Log Level | Server：`--log-level debug\|info\|error` | **Log 頁面** | Info；下次啟動 server 生效 |
| Qwen 一次確認最多四個 token | CLI／Server：`--qwen-short-block`／`--no-qwen-short-block` | Qwen → Advanced Settings | **新預設關閉**；已明確儲存的選擇保留 |

DeepSeek V4／V4.1 與 Qwen 均提供記憶體／磁碟 Prompt Cache 選項。
Qwen MTP 不使用一般 prompt cache；DeepSeek DSpark 的跨請求重用另由實驗旗標控制。
這裡的 prompt cache 與「專家資料快取」是不同設定。

## 安裝與模型管理

| 功能 | 命令列 | UI | 說明 |
| --- | --- | --- | --- |
| 查看模型 checkpoint 合約 | `dsv4-repack inspect` | 無完整合約檢視 | Model 頁面顯示名稱、目錄、容量與安裝狀態 |
| 產生安裝計畫 | `dsv4-repack plan --output` | 無計畫檔匯出 | App 在安裝流程內使用計畫 |
| 下載 checkpoint 並重排安裝 | `dsv4-repack repack --output [--plan]` | Model → 下載 | 模型套件決定安裝方式 |
| Qwen 已重排模型直接下載 | 無對應的 `dsv4-repack` 子命令 | Qwen → 下載 | 與從原始 checkpoint 重排的 CLI 路徑不同 |
| 指定模型種類 | repack 的 `--model` | Model 頁面的模型列 | DeepSeek V4／V4.1／Qwen |
| 安裝進度、續傳 | repack 安裝紀錄與續傳檔 | 進度、容量、速度、剩餘時間、恢復 | UI 一次只下載一個模型 |
| 取消安裝 | 結束安裝程序 | Stop Current Operation | 保留可續傳資料 |
| 校驗模型 | `dsv4-repack verify --model` | Verify and repair | UI 另會下載缺少或損壞的資料 |
| 修復安裝 | 重跑 `repack`，沿用安裝計畫與續傳 | Verify and repair | CLI 的 `verify` 本身只校驗 |
| 安裝 DSpark | `dsv4-repack install-dspark` | DeepSeek 新安裝包含 DSpark | 安裝不等於啟用 |
| 安裝 Qwen MTP | `dsv4-repack install-mtp` | Qwen 的 MTP 下載按鈕 | 保留主模型 |
| 選擇模型目錄 | CLI／Server：`--model PATH`；catalog 的 `path` | Select Model Folder | UI 顯示空間、可用性與修復狀態 |
| 模型 Alias | Server：`--public-model` 或 JSON `alias` | Model → Advanced Settings → Alias | `--public-model` 僅限單模型啟動 |
| 查看模型清單 | API：`GET /v1/models` | Model 頁面 | Server 可用空 catalog 啟動 |
| 主動載入／卸載 | API：`POST /api/models/load`、`/unload` | Model → Load／Unload | 服務同時只載入一個模型 |
| 修改未載入模型設定 | API：`POST /api/models/configure` | Model → Advanced Settings | 不用停止整個服務 |

## 服務、生成與聊天

| 功能 | 命令列 | UI | 說明 |
| --- | --- | --- | --- |
| 啟動／停止服務 | Server 或 `make server`；Ctrl+C 停止 | Server → Start／Stop | 啟動服務不等於載入模型 |
| 多模型 catalog | Server：`--model-catalog` | 自動產生 | 與 `--model` 互斥 |
| 監聽位址／Port | Server：`--host`、`--port` | Server 頁面 | 執行中鎖定，重啟生效 |
| API key | Server：`--api-key` 或 `DEEPSEEK_API_KEY` | Server 頁面 | UI 存入 Keychain；非本機位址必須設 key |
| 讀取服務狀態 | API：`/healthz`、`/api/status` | Server／Metric 頁面 | 未公開額外設定 API |
| 直接輸入一段 prompt | CLI：`--prompt` | Chat 的訊息欄 | CLI 是單次生成；UI 包含對話歷史 |
| 從 UTF-8 檔讀 prompt | CLI：`--prompt-file` | 無直接匯入聊天檔按鈕 | 另有 warmup 檔案設定 |
| 串流聊天 | API：`/v1/chat/completions` | Chat 頁面 | UI 顯示文字、思考與工具呼叫 |
| Responses API | API：`/v1/responses` | 無專用編輯器 | 由外部 client 使用 |
| Text Completions API | API：`/v1/completions` | 無專用編輯器 | CLI 可直接處理文字 prompt |
| 取消正在生成的回答 | CLI：Ctrl+C；API：中斷串流連線 | Stop Generating | UI 離開 Chat 頁面不會取消 |
| 保存／清除對話歷史 | Client 自行管理；Server 不保存完整對話 | Chat 自動保存／Clear Chat | 清除 UI 聊天不等於刪除 prompt cache |
| 工具定義、tool choice、工具結果 | API request 欄位 | 僅顯示工具呼叫 | **App 不執行工具**，也沒有工具定義編輯器 |
| 思考模式／reasoning effort | API request 欄位 | 顯示思考內容；無完整模式控制項 | 支援值與模型差異見 API 文件 |
| Exact／近似模式 | CLI：`--approximation`；JSON `defaults.approximation_mode`；API request 欄位 | DeepSeek V4 → Use approximate mode | 預設關閉（Exact）；DSpark 啟用時停用 |
| 最大輸出 token | CLI：`--max-tokens`；Server：`--default-max-tokens`；API | Model → Advanced Settings → Max tokens | App 各模型預設 8192；保留已保存設定；仍受模型 context 上限限制 |
| Temperature／Top P／Top K | CLI：`--temperature`／`--top-p`／`--top-k`；Server：`--default-temperature`／`--default-top-p`／`--default-top-k`；API | DeepSeek V4 與 Qwen 的進階設定可調 | Qwen 關閉 Use adaptive sampling 後使用手動值；V4.1 UI 不提供調整 |
| Qwen 自動取樣 | JSON `defaults.qwen_adaptive_sampling` | Use adaptive sampling | 預設開啟，依聊天／思考模式選值；關閉後使用已儲存的 T／P／K，API 明確值仍優先 |
| 暖機 prompt | Server：`--warmup-prompt-file`；JSON `warmup_prompt_path` | Model → Advanced Settings → Warmup prompt | catalog 必須逐模型設定檔案 |

## Runtime 與快取設定

| 功能／欄位 | CLI | Server 命令列 | UI |
| --- | --- | --- | --- |
| 專家快取容量 `slots` | `--slots` | `--slots` | Slots；App 預設 V4 1152、V4.1 1152、Qwen 3072 |
| 讀取工作數 `read_workers` | `--read-workers` | 同左 | Read workers |
| 預讀工作數 `prefetch_read_workers` | `--prefetch-read-workers` | 同左 | Prefetch read workers；預設 2，至少 1 |
| MLX 記憶體限額 | `--memory-limit-gib` | 同左 | Memory limit GiB；0 為自動 |
| 輸入分批大小 | `--prefill-step-size` | 同左 | Prefill step size |
| MoE 輸入分批大小 | `--moe-prefill-step-size` | 同左 | MoE prefill step size；預設 0（自動），不可為負 |
| 按層處理輸入 | `--no-layer-major-prefill` | 同左 | 支援模型的 Use layer-major prefill |
| 按層處理門檻 | `--layer-major-prefill-threshold` | 同左 | DeepSeek V4 可調；Qwen 使用模型門檻 |
| Expert 合批輸入 | `--no-batched-expert-prefill` | 同左 | 固定開啟，沒有獨立控制項 |
| Qwen Prefill acceleration | `--qwen-grouped-experts`／`--no-qwen-grouped-experts` | 同左 | Qwen 開關；預設開啟，MTP 開啟時停用 |
| ANE Prefill | `--no-ane-prefill` | 同左 | 沒有獨立開關；Qwen 提供 ANE Prefill share |
| ANE 分工比例 | `--ane-prefill-ratio` | 同左 | Qwen ANE Prefill share；0 表示 GPU only |
| FP8／BF16 KV cache | `--bf16-kv-cache` | 同左 | DeepSeek V4 可選 BF16；其他模型固定 |
| FP4 index cache | `--no-fp4-index-cache` | 同左 | 固定開啟，沒有控制項 |
| Ready expert decode | `--no-ready-expert-decode` | 同左 | 固定開啟，沒有控制項 |
| LRU／LFU 專家資料淘汰方式 | `--expert-eviction-policy` | 同左 | Expert cache eviction 選單：LRU／LFU；App 預設 LRU，CLI 預設 LFU |
| Qwen 四 token 合批 | `--qwen-short-block`／`--no-qwen-short-block` | 同左 | Qwen 開關；新預設關閉 |
| Prompt cache 模式 | `--prompt-cache off\|memory\|disk` | 同左 | 不使用／記憶體／磁碟；新預設記憶體 |
| Prompt cache 筆數 | `--prompt-cache-entries` | 同左 | Prompt cache entries；不使用時隱藏 |
| Prompt cache 記憶體限額 | `--prompt-cache-memory-gib` | 同左 | Prompt cache GiB；不使用時隱藏 |
| 磁碟快取開關 | `--persistent-prompt-cache`／`--no-persistent-prompt-cache` | 同左 | 整合在三種快取模式中 |
| 磁碟快取目錄 | `--prompt-cache-directory` | 同左 | 使用預設目錄，沒有自訂路徑控制項 |
| 磁碟快取筆數 `persistent_prompt_cache_entries` | 無旗標 | JSON | 固定 8，沒有控制項 |
| SSD 讀取限速 | `--power-saving-limit-gbps` | 同左 | Advanced → Power Saving Mode |
| Qwen MTP | `--mtp` | 同左 | Use MTP；需先安裝 sidecar |
| MTP 快取容量 | `--mtp-slots` | 同左 | MTP slots |
| DeepSeek DSpark | `--dspark` | 同左 | Use DSpark；需先安裝 |
| DSpark 快取容量 | `--dspark-slots` | 同左 | DSpark slots |
| DSpark 信心門檻 | `--dspark-confidence-threshold` | 同左 | DSpark confidence threshold |

`Server` 欄的 runtime 旗標也可覆寫 catalog 中對應欄位。這些覆寫會套到 catalog
內所有模型；需要逐模型差異時，直接編輯各 entry。表中的開關只表示入口存在，
實際作用仍受模型、MTP／DSpark、輸入大小與資料格式限制。

## 實驗與診斷入口

以下功能不因有設定入口就成為已驗證的加速方案。正式量測與停止原因見研究文件。

| 功能／欄位 | CLI | Server 命令列／JSON | UI |
| --- | --- | --- | --- |
| Qwen 下一層預讀 | `--qwen-next-layer-prefetch` | JSON `qwen_next_layer_prefetch` | 固定關閉 |
| Qwen grouped Decode | `--qwen-grouped-decode` | JSON `qwen_grouped_decode` | 固定關閉 |
| DSpark prompt cache | `--dspark-prompt-cache` | 同左 | 固定關閉 |
| DSpark hash 預讀 | `--dspark-hash-prefetch` | 同左 | 固定關閉 |
| DSpark adaptive block | `--dspark-adaptive-block` | 同左 | 固定關閉 |
| DSpark 停止保護 | `--no-dspark-fallback` 關閉保護 | 同左 | 保護固定開啟 |
| DSpark 逐 token 驗證 | `--dspark-sequential-verification` | 同左 | 固定關閉 |
| DSpark 混合驗證 | `--dspark-hybrid-verification` | 同左 | 固定關閉 |
| 專家檔案頁面快取探測 | `--expert-page-cache-probe` | 同左 | 固定關閉 |
| 專家檔案 cached／bypass | `--expert-file-cache-policy` | 同左 | 固定 cached |
| 路由追蹤 | `--expert-route-trace` | JSON `expert_route_trace` | 無控制項 |
| Split-slot prototype | 無旗標 | JSON `staged_expert_streaming` 可解析，受相容檢查 | 固定關閉 |
| Adaptive expert prefill prototype | 無旗標 | JSON `adaptive_expert_prefill_threshold` 可解析，受相容檢查 | 固定 null |

最後兩項沒有直接 CLI 開關，屬已停止的內部研究原型；JSON 型別接受欄位不等於完整模型驗證。

## Log、效能與 App 功能

| 功能 | 命令列 | UI | 說明 |
| --- | --- | --- | --- |
| Log Level | Server：`--log-level` | Log 頁面 | Debug／Info／Error；執行中鎖定 |
| 讀取 server log | 終端 stdout／stderr | Log 頁面 | Debug 可能包含完整 request 內容 |
| 生成指標匯出 | CLI：`--metrics-json` | 無 JSON 匯出按鈕 | UI 有 Metric 頁面與聊天即時指標 |
| 即時記憶體、讀取、快取、生成速度 | API：`GET /api/status` | Metric 頁面 | 個別指標支援範圍依 runtime 模式而異 |
| 效能歷史圖與清除 | 沒有同等的內建 CLI 圖表 | Metric 頁面 | UI 保存觀察到的歷史資料 |
| SSD 讀取基準測試 | `dsv4-repack benchmark` | 無 | 與完整模型生成 benchmark 不同 |
| 固定工作負載 API benchmark | `Scripts/benchmark_api.py`、`make benchmark-qwen`／`benchmark-dsv4` | 無 | 可記錄多次結果 |
| App 語言 | 無產品 CLI 控制項 | Settings → Language | 英文、繁中、簡中 |
| 檢查／安裝 App 更新 | 無 Sparkle CLI | Settings → Software Updates | stable／dev 來源、自動檢查開關；[更新流程](UPDATES.md) |
| 原始碼／Release／模型外部連結 | 可自行開啟網址 | Settings／Model 頁面 | 不是 runtime API |
| 本機編譯／測試 | `make build`、`make test`、Python unittest | 無 | 開發流程 |
| 本機封裝／正式發布 | `make package`、`make release VERSION=…` | 無 | 需遵守專案封裝與發布驗證流程 |

## 核對來源

- [Server 參數與 catalog 覆寫](../runtime/deepseek_v4_ssd/server.py)
- [直接生成 CLI](../runtime/deepseek_v4_ssd/cli.py)
- [RuntimeConfig](../runtime/deepseek_v4_ssd/model.py)
- [API 與 model catalog 合約](API.md)
- [App 設定畫面](../Sources/DeepSeekV4SSDApp/ContentView.swift)
- [App 偏好與預設值](../Sources/DeepSeekV4SSDApp/ServerController.swift)
- [App 傳入 runtime 的設定](../Sources/DeepSeekV4SSDApp/ModelLibrary.swift)
- [模型能力與預設](../Sources/DeepSeekRepack/Resources/ModelPackages.json)
- [安裝 CLI](../Sources/dsv4-repack/main.swift)
