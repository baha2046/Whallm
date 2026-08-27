# 架構與目前實作

Whallm 在 Apple Silicon 上執行固定的
`DeepSeek-V4-Flash-0731` 或 `Qwen3.8-Flash-Next-FP8` checkpoint。
runtime 將 common tensor 保留在統一記憶體。
runtime 只在 router 選到 routed expert 時讀取 expert blob。

本文件描述目前程式碼。
本文件不描述研究中的預期設計。

## 元件

```text
DeepSeek checkpoint or Qwen artifact production
  -> Swift checkpoint inspector
  -> repack plan
  -> resumable repacker
  -> installed model + manifest

Published Qwen installed model artifact
  -> resumable installed file downloader
  -> complete SHA-256 verification
  -> installed model + manifest

installed model
  -> Python MLX runtime
  -> OpenAI-compatible server
  -> SwiftUI APP or API client
```

| 元件 | 責任 |
| --- | --- |
| `DeepSeekRepack` | 檢查 checkpoint、建立 repack plan、下載 published installed model、安裝、驗證和 repair。 |
| `dsv4-repack` | 提供 `inspect`、`plan`、`repack`、`verify`、`install-dspark` 和 `benchmark`。 |
| `deepseek_v4_ssd` | 載入 installed model、執行推論、管理 cache 和記錄指標。 |
| `deepseek_v4_ssd.server` | 提供 OpenAI 相容 API 和 APP 專用 API。 |
| `DeepSeekV4SSDApp` | 管理模型、啟動 server、顯示對話和效能。 |

Qwen 使用 manifest format 2。
format 2 新增 `modelKind`、`maximumContext`、`expertQuantization` 和 `ngram`。
DeepSeek 保留 manifest format 1。現有 installed model 不需要轉換。

Qwen 的完整合約和資料路徑請見 [Qwen 支援](QWEN.md)。

## Checkpoint 合約

程式碼固定下列合約。

| 欄位 | 值 |
| --- | ---: |
| Model ID | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| main model layers | 43 |
| routed experts per layer | 256 |
| selected experts per token | 6 |
| hidden size | 4,096 |
| expert intermediate size | 2,048 |
| checkpoint maximum position | 1,048,576 |
| expert dtype | checkpoint-native FP4 |

官方模型頁記錄 284B total parameters、約 13B active parameters 和 1M
context。[官方模型頁](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
提供外部規格。

`ModelContract.validate` 會檢查模型 shape 和架構欄位。
repacker 會拒絕不相容的 checkpoint。

1,048,576 是 checkpoint 合約上限。
1,048,576 不是本專案的本機驗證長度。
目前的長 context 證據請見[驗證紀錄](VALIDATION.md)。

## Installed model

App 的 DeepSeek 安裝固定包含 DSpark。
使用者可以在安裝後移除 DSpark。

```text
deepseek-v4-flash-0731.dsv4/
  manifest.json
  common.bin
  config.json
  generation_config.json
  inference/config.json              # DSpark 安裝時必需
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  encoding/encoding_dsv4.py
  experts/
    layer_00.bin
    ...
    layer_42.bin
  dspark/                             # 可選
    common.bin
    experts/
      layer_00.bin
      layer_01.bin
      layer_02.bin
```

### 大小

目前 verified installed model 具有下列大小。

| 資料 | Bytes | GiB |
| --- | ---: | ---: |
| main model packed weights | 156,015,738,880 | 145.30 |
| main model `common.bin` | 8,846,000,128 | 8.24 |
| 每個 expert blob | 13,369,344 | 0.01245 |
| 每個 main model expert layer | 3,422,552,064 | 3.19 |
| DSpark packed weights | 10,862,841,600 | 10.12 |
| 含 DSpark 的 54 個 manifest files | 166,884,980,648 | 155.42 |

`manifest.json` 不包含在 `manifest.files` 計數中。
沒有 DSpark 的 manifest 需要 49 個 main model files。
`inference/config.json` 可以作為第 50 個可選 main model file 保留。

App 使用下列固定值執行下載前空間檢查。

| model kind | Bytes | 值的來源 |
| --- | ---: | --- |
| DeepSeek | 166,878,580,480 | 固定 checkpoint revision 的 repack plan `installedBytes` |
| Qwen | 125,291,490,955 | 固定 installed model revision 的 manifest files 合計 |

App 啟動時不會為了取得這些值建立 repack plan 或下載遠端 manifest。
checkpoint revision 或 installed model revision 變更時，App 必須同步更新固定值。

### Expert blob

repacker 依下列順序寫入每個 expert blob。

1. `w1.weight`
2. `w1.scale`
3. `w2.weight`
4. `w2.scale`
5. `w3.weight`
6. `w3.scale`

repacker 不轉換這些值。
repacker 只把 checkpoint byte range 複製到標準位置。

runtime slot 會把 `w3` 和 `w1` 放在相鄰區域。
這個 slot layout 讓 runtime 可以建立 fused `w13` view。
`preadv` 會把一個 expert blob 的各區域直接寫入對應 view。

## 安裝與驗證

DeepSeek App 安裝和 Qwen artifact 產生使用 repack 流程。
流程如下。

1. inspector 下載 `config.json` 和 safetensors index。
2. inspector 使用 HTTP Range 讀取每個 shard 的 safetensors header。
3. planner 檢查每個 routed expert 的 dtype、shape 和大小。
4. planner 建立 checkpoint byte range 到 installed model byte range 的對應。
5. common tensor 和 N-gram 使用最多八個並行下載工作與 8 MiB chunk。
6. Qwen expert weight 使用兩個並行工作。每個工作合併最多 64 MiB 的連續 range。
7. Qwen 每個 checkpoint shard 的 expert scale 只下載一次。
8. repacker 使用 receipt 記錄已完成 chunk 的 digest。
9. repacker 對每個 installed file 計算 SHA-256。
10. repacker 寫入 manifest，然後以原子 move 完成安裝。

Qwen App 不執行 FP8 到 MXFP4 轉換。
Qwen App 直接下載 published installed model。
Qwen App 使用四個並行 file 工作和 8 MiB range。
Qwen App 使用 receipt 記錄完成的 file。
Qwen App 驗證 manifest 合約、file size 和每個 file 的 SHA-256。
完整驗證通過後，App 才完成原子 move。

repair 會先做完整 audit。
DeepSeek repair 只重新下載失敗 file 涉及的 checkpoint byte range。
Qwen repair 直接重新下載失敗的 installed file。

### 完整性邊界

不同入口執行不同層級的檢查。

| 入口 | 檢查 |
| --- | --- |
| APP 模型掃描 | manifest 合約、路徑、一般檔案和 file size。 |
| APP 完整驗證 | manifest 合約、file size 和每個 file 的 SHA-256。 |
| `dsv4-repack verify` | manifest 合約、file size 和每個 file 的 SHA-256。 |
| Python runtime 載入 | 固定合約、必要 file、safe path、file size 和 expert layout。 |

Python runtime 啟動時不重新計算 155 GiB 的 SHA-256。
使用者應在安裝後或懷疑損壞時執行完整驗證。

## Runtime 預設值

| 設定 | 預設值 | 說明 |
| --- | ---: | --- |
| `slots` | 1,152 | main model expert cache 容量。 |
| `read_workers` | 4 | 個別 expert blob 讀取工作數。 |
| `prefetch_read_workers` | 2 | full-layer prefetch 工作數。 |
| `power_saving_limit_gbps` | `null` | routed expert SSD 聚合讀取速度上限。`null` 代表無限制。 |
| `prefill_step_size` | 0 | 由 prompt 長度自動選擇。 |
| `moe_prefill_step_size` | 0 | 4K 以上自動使用 4,096-token tile。 |
| `layer_major_prefill` | `true` | 只在至少 4,096 個未快取 token 時啟用。 |
| `batched_expert_prefill` | `true` | full-layer prefill 使用 `gather_qmm`。 |
| `fp8_kv_cache` | `true` | 已完成的 compressed cache chunk 使用 MXFP8。 |
| `fp4_index_cache` | `true` | indexer cache 使用 MXFP4 view。 |
| `ready_expert_decode` | `true` | decode 依 expert ready 時間提交運算。 |
| `prompt_cache_entries` | 2 | 記憶體 prompt cache timeline 數。 |
| `prompt_cache_memory_gib` | 8 | 記憶體 prompt cache 上限。 |
| persistent cache entries | 8 | revision 專用的磁碟 cache 上限。 |
| `memory_limit_gib` | 0 | 0 使用模型安全自動上限。Qwen 自動上限不超過 48 GiB。DeepSeek 使用 Metal 建議上限。正值設定 MLX memory limit，wired limit 不超過 Metal 建議上限。 |
| `dspark_enabled` | `false` | DSpark 預設停用。 |
| `dspark_slots` | 768 | DSpark 使用獨立 expert cache。 |

APP 的省電模式 Slider 支援 500 MB/s、1、2、3、5、10、25 GB/s 和無限制。
選擇速度上限時，APP 會傳送 `--power-saving-limit-gbps`。
server 只接受 0.5、1、2、3、5、10 或 25 GB/s。
選擇無限制時，APP 不會傳送這個參數。
限速器會序列化 routed expert 的 `preadv` 呼叫，並在每次讀取後等待。
此限速不包含啟動時讀取的 common tensor。
專案尚未量測各速度上限的耗電量與 generation 效能。

自動 prefill step 如下。

| 未快取 prompt token | Step |
| ---: | ---: |
| 少於 1,024 | 128 |
| 1,024 至 4,095 | 256 |
| 4,096 或更多 | 1,024 |

## Prefill 資料路徑

少於 4,096 個未快取 token 時，runtime 使用 mlx-lm 的 chunk-major path。

4,096 個或更多未快取 token 時，runtime 使用 layer-major path。

layer-major path 執行下列工作。

1. runtime 依 step 執行一層的 attention。
2. runtime 預取下一個 routed expert layer。
3. runtime 把目前層的 256 個 routed expert 讀入連續 MLX buffer。
4. runtime 依 4,096-token MoE tile 執行 batched `gather_qmm`。
5. runtime 完成目前層後才進入下一層。
6. runtime 保留最後一個 prompt token 給一般 generator。

Prefill 與 Decode 共用一般 file descriptor。
M1 曾讓 full-layer Prefill 使用獨立的 `F_NOCACHE` file descriptor。
正式研究沒有通過全部採用條件。
runtime 已移除 M1 prototype 和相關開關。

cache-only prefill 不執行最後一層 MoE。
因此 4K layer-major prefill 通常記錄 42 個 batched expert layers。

Qwen 使用不同門檻。
少於 128 個未快取 token 時，Qwen 使用 selected expert cache。
短 prompt 不會建立 48 個完整 expert layer buffer。
128 個或更多未快取 token 時，Qwen 使用 layer-major path。
該 path 每次只保留一個完整 expert layer buffer。

## Decode 資料路徑

每個 main model layer 執行下列工作。

1. Metal 計算 attention、router 和 shared expert。
2. CPU 取得六個 routed expert ID。
3. expert cache 回傳 resident slot，或提交缺少的 `preadv`。
4. ready expert decode 先提交 resident 或先讀完的 expert 運算。
5. runtime 依原始 router 順序重組 routed expert output。
6. runtime 把 routed output 和 shared expert output 相加。

ready expert decode 不修改 router 選擇。
ready expert decode 可能修改 cache admission 的完成順序。

## Cache

### Expert cache

expert cache 使用全域 LFU heap。
相同 frequency 時，expert cache 使用最近存取時間決定順序。
每層保留最小 slot 配額。
prefill 可以暫時 pin 一層。

1,152 個 slot 的 expert payload 容量是 14.34 GiB。
slot 會在首次需要時配置。
runtime 不會在啟動時配置全部 slot。

### KV cache

短 sliding-window cache 保留 BF16。
完成的 compressed-attention chunk 使用 MXFP8。
indexer 預設另外建立 MXFP4 index cache。
很長的 context 仍會增加 KV cache 和暫存資料。

### Prompt cache

runtime 預設保留兩個記憶體 timeline。
runtime 把記憶體用量限制在 8 GiB。

runtime 預設把最多八個完成 entry 寫到：

```text
~/.dsmodel/prompt-cache/<checkpoint-revision>/
```

persistent cache format 2 直接儲存 quantized cache arrays。
runtime 會忽略無法載入的 cache entry。

## DSpark

DSpark 是可選功能。
App 下載的 DeepSeek installed model 固定包含 DSpark weights。
runtime 預設不啟用 DSpark。

目前 DSpark 合約如下。

| 欄位 | 值 |
| --- | ---: |
| layers | 3 |
| block size | 5 |
| target layer IDs | 40、41、42 |
| Markov rank | 256 |
| noise token ID | 128,799 |
| expert slots | 768 |

DSpark 使用獨立 expert cache。
768 個 DSpark slot 的 expert payload 容量是 9.56 GiB。

目前 verifier 一次處理完整 verification block。
verifier 在每個 round 建立一次 cache fork。
若 draft 被拒絕，runtime 只 replay committed prefix。
runtime 會在 speculative cost 高於 target baseline 時 fallback。

DSpark path 不使用一般 prompt cache reuse。
目前量測沒有證明 DSpark 具有淨加速。
詳細決策請見[研究結論](RESEARCH.md)。

## Server 與 APP

server 讓一個 installed model 保持載入。
`ModelRuntime` 使用 lock 序列化 generation。
因此 server 一次只執行一個 generation request。

`ThreadingHTTPServer` 仍可在 generation 期間回應 `/api/status`。
其他 generation request 會等待 runtime lock。

APP 使用獨立 Python process 啟動 server。
APP 每秒讀取 `/api/status`。
APP 使用 process RSS 顯示記憶體。
Server 的 Model list 使用 `NavigationStack` 顯示每個 model kind 的獨立進階設定頁。
獨立進階設定頁不會新增 sidebar 項目。
使用者選擇進階設定圖示時，APP 從右側推入頁面。
頁面標頭顯示 model 名稱和返回指示。
APP 將 Generate 和 Runtime 設定依 model kind 分開儲存。
使用者選擇模型時，APP 將該模型的設定載入 `ServerConfiguration`。
Power Saving Mode 是所有模型共用的設定。
APP 使用 bundle domain `com.deepseekv4ssd.app` 的 `UserDefaults` 保留 Server、各模型的進階設定、Power Saving Mode、模型、語言、目前頁面與測試對話設定。
APP 使用 macOS Keychain 的 `com.deepseekv4ssd.app` service 與 `server-api-key` account 保留 API key。
APP 啟動時會依目前 APP 位置重新取得 runtime 路徑。

## 目前限制

- runtime 只支援固定 checkpoint revision。
- runtime 只支援一個 Apple Silicon device。
- generation batch size 是 1。
- server 一次只執行一個 generation request。
- full-model sampling parity 尚未記錄在目前驗證 artifact。
- 本專案沒有驗證 1M context。
- MTLIO、custom Metal expert kernel 和 learned prefetch predictor 尚未整合。
- DSpark 可執行，但 DSpark 預設停用。

## 主要程式碼

- Checkpoint 合約：[`Sources/DeepSeekRepack/Model.swift`](../Sources/DeepSeekRepack/Model.swift)
- Checkpoint 讀取：[`Sources/DeepSeekRepack/Checkpoint.swift`](../Sources/DeepSeekRepack/Checkpoint.swift)
- Repacker 與完整驗證：[`Sources/DeepSeekRepack/Repacker.swift`](../Sources/DeepSeekRepack/Repacker.swift)
- Runtime 設定與 model path：[`runtime/deepseek_v4_ssd/model.py`](../runtime/deepseek_v4_ssd/model.py)
- Expert cache：[`runtime/deepseek_v4_ssd/expert_cache.py`](../runtime/deepseek_v4_ssd/expert_cache.py)
- Generation 與 prompt cache：[`runtime/deepseek_v4_ssd/generation.py`](../runtime/deepseek_v4_ssd/generation.py)
- DSpark：[`runtime/deepseek_v4_ssd/dspark.py`](../runtime/deepseek_v4_ssd/dspark.py)
- Server：[`runtime/deepseek_v4_ssd/server.py`](../runtime/deepseek_v4_ssd/server.py)
