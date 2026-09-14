# 架構與目前實作

Whallm 在 Apple Silicon 上執行固定的
`DeepSeek-V4-Flash-0731`、`DeepSeek-V4.1-Flash` 或
`Qwen3.8-Flash-Next-FP8` checkpoint。
runtime 將 common tensor 保留在統一記憶體。
runtime 只在 router 選到 routed expert 時讀取 expert blob。

本文件描述目前程式碼。
本文件不描述研究中的預期設計。

2026-09-12 已接入 [模型支援套件](MODEL_PACKAGES.md)：Swift `ModelPackage` 與
Python `ModelSupport` 分別處理安裝及 runtime 差異，`ModelPackages.json` 共用
模型身分、App 顯示與設定資料。`ExpertCache` 由套件選擇 expert layout，
共用讀取和 slot 管理；詳細責任、相容範圍及新增模型步驟見該文件。

Model Advanced Settings 的專家資料保留設定沿用既有預設。
三模型新增或補上 UI 的加速路徑見 [三模型加速功能](MODEL_ACCELERATION.md)。
文字候選四 token 驗證與常駐專家 Decode 合批已移除；Prefill 合批與 DSpark／MTP 保留。
狀態位元複製修正繼續生效。

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
  -> model catalog
  -> ModelManager
  -> Python MLX runtime on first generation request
  -> OpenAI-compatible server
  -> SwiftUI APP or API client
```

| 元件 | 責任 |
| --- | --- |
| `DeepSeekRepack` | 檢查 checkpoint、建立 repack plan、下載 published installed model、安裝、驗證和 repair。 |
| `dsv4-repack` | 提供 `inspect`、`plan`、`repack`、`verify`、`install-dspark` 和 `benchmark`。 |
| `deepseek_v4_ssd` | 載入 installed model、執行推論、管理 cache 和記錄指標。 |
| `deepseek_v4_ssd.model_manager` | 驗證 model catalog、延遲載入一個 runtime、切換模型並序列化 generation request。 |
| `deepseek_v4_ssd.server` | 提供 OpenAI 相容 API 和 APP 專用 API。 |
| `DeepSeekV4SSDApp` | 管理模型、啟動 server、顯示對話和效能。 |

Qwen 使用 manifest format 2。
format 2 新增 `modelKind`、`maximumContext`、`expertQuantization` 和 `ngram`。
DeepSeek V4 Flash 0731 保留 manifest format 1。
DeepSeek V4.1 使用 manifest format 3，加入固定的 `engram` table descriptor。
現有 installed model 不需要轉換。

Qwen 的完整合約和資料路徑請見 [Qwen 支援](QWEN.md)。
DeepSeek V4.1 的完整合約和資料路徑請見
[DeepSeek V4.1 支援](DEEPSEEK_V41.md)。

## Checkpoint 合約

程式碼固定下列合約。

| 欄位 | 值 |
| --- | ---: |
| checkpoint model ID | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| API model ID | `deepseek-v4-flash-0731` |
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

Python runtime 載入 installed model 時不重新計算 155 GiB 的 SHA-256。
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
| `layer_major_prefill` | `true` | 控制 DeepSeek layer-major Prefill。 |
| `layer_major_prefill_threshold` | 1,024 | DeepSeek 啟用 layer-major Prefill 所需的最少未快取 token 數。APP 可設定此值。 |
| `batched_expert_prefill` | `true` | full-layer prefill 使用 `gather_qmm`。 |
| `qwen_next_layer_prefetch` | `false` | Qwen next-layer prefetch 研究開關。APP model catalog 固定設為關閉。 |
| `qwen_grouped_decode` | `false` | Qwen resident grouped QMM 實驗開關；需要 MTP 關閉。CLI 可 opt-in，APP catalog 固定關閉。目前每個 arena 最多 1,024 slots，後續依 common bytes 與已配置 slots 規劃大小；request 結束時同步 generation stream。 |
| `ane_prefill` | `true` | Qwen 使用 private `AppleNeuralEngine.framework` 執行固定 shape projection。DeepSeek 不使用此設定。 |
| `ane_prefill_ratio` | 0.25 | Qwen 分配給 ANE 的 `q_proj` output channels 比例。值域是 0 到 1。 |
| `fp8_kv_cache` | `true` | 已完成的 compressed cache chunk 使用 MXFP8。 |
| `fp4_index_cache` | `true` | indexer cache 使用 MXFP4 view。 |
| `ready_expert_decode` | `true` | decode 依 expert ready 時間提交運算。 |
| `staged_expert_streaming` | `false` | Internal research-only split `w13`／`w2` slot prototype；需要 ready-expert decode，拒絕 DSpark，且不提供 CLI／server／APP opt-in。 |
| `adaptive_expert_prefill_threshold` | `null` | Internal stopped research prototype；只接受 0.7／0.8／0.9，需要 layer-major batched prefill，拒絕 DSpark／staged composition，且沒有 CLI／server／APP opt-in。 |
| `expert_page_cache_probe` | `false` | Research-only `mincore` pre-read page-residency classification；不是 physical SSD counter。 |
| `expert_file_cache_policy` | `cached` | Expert descriptor policy；research-only `bypass` 使用 Darwin `F_NOCACHE` 並停用 read-ahead。 |
| `expert_eviction_policy` | `lfu` | 固定容量 expert cache 的淘汰排序，可選 `lru`；保留相同每層配額、pinning、in-flight 保護與 heap 清理。CLI／server／catalog 可 opt-in，舊 catalog 省略時仍用 LFU，App UI 預設不變。 |
| `prompt_cache_entries` | 2 | 記憶體 prompt cache timeline 數；0 停用跨請求快取（也不讀寫磁碟快取）。 |
| `persistent_prompt_cache` | `false` | 預設僅使用記憶體；`true` 額外保存磁碟快取。 |
| `prompt_cache_directory` | `null` | 磁碟模式的根目錄；null 使用 `~/.dsmodel/prompt-cache`。 |
| `prompt_cache_memory_gib` | 8 | 記憶體 prompt cache 上限。 |
| persistent cache entries | 8 | normal 和 DSpark 各自的 revision 專用磁碟 payload 上限。normal format 5 依 reuse count 和 access recency 執行 eviction。 |
| `memory_limit_gib` | 0 | 0 使用模型安全自動上限。Qwen 自動上限不超過 48 GiB。DeepSeek 使用 Metal 建議上限。正值設定 MLX memory limit，wired limit 不超過 Metal 建議上限。 |
| `mtp_enabled` | `false` | 啟用 Qwen MTP speculative decoding prototype。需要 installed MTP sidecar。 |
| `mtp_slots` | 32 | Qwen MTP 使用獨立 expert cache。最小值是 10。 |
| `dspark_enabled` | `false` | DSpark 預設停用。 |
| `dspark_prompt_cache` | `false` | 實驗性原子 target KV + DSpark context prefix reuse；需要 DSpark。 |
| `dspark_slots` | 768 | DSpark 使用獨立 expert cache。 |
| `dspark_hash_prefetch` | `false` | 實驗性 target hash-layer exact prefetch；需要 DSpark。 |
| `dspark_adaptive_block` | `false` | 實驗性 storage-aware verification prefix selector；需要 DSpark 與 target hash layers。 |
| `dspark_fallback_enabled` | `true` | speculative wall-time 超過 autoregressive break-even 時停止 DSpark；研究控制才可停用。 |
| `dspark_sequential_verification` | `false` | 逐 token target verification correctness oracle；需要 DSpark，且不能與 hash prefetch 同時啟用。 |
| `dspark_hybrid_verification` | `false` | 逐 token target math、每層一次 expert-union acquisition 的 correctness candidate；需要 DSpark，且與 sequential oracle 互斥。 |

APP 的省電模式 Slider 支援 500 MB/s、1、2、3、5、10、25 GB/s 和無限制。
APP 會把選定值寫入 model catalog 的每個 `runtime` object。
server 只接受 0.5、1、2、3、5、10、25 GB/s 或 `null`。
`null` 代表無限制。
限速器會序列化 routed expert 的 `preadv` 呼叫，並在每次讀取後等待。
此限速不包含啟動時讀取的 common tensor。
專案尚未量測各速度上限的耗電量與 generation 效能。

`--expert-page-cache-probe` 在每次 expert `preadv` 前用 `mincore` 以 OS VM page
粒度分類實際 read range（本次 M2 Max 是 16 KiB）。Partial boundary pages 以 byte
overlap 計算，並保存
resident、nonresident、unclassified bytes 與 failure count。Main cache、DSpark cache
及 exact hash-prefetch useful/wasted partition 都使用同一套 accounting。Probe 會增加
`mmap`／`mincore`／`munmap` observer overhead，預設關閉。Nonresident 只表示讀取前頁面
不在 VM core；APFS、storage-controller cache 與實體 device bytes 不在此合約內。

`--expert-file-cache-policy cached|bypass` 同時套用到 main 與 DSpark expert
descriptors。預設 `cached` 不改 descriptor flags。Darwin research-only `bypass` 設定
`F_NOCACHE=1`、`F_RDAHEAD=0`，並在每次 read 前驗證 destination address、file offset
與 iovec length 都符合 filesystem allocation-block alignment；目前 APFS installed model
是 4,096 bytes。CLI metrics 與 `/api/status` 回報 policy 及 alignment。Bypass 不會清除
設定前已 resident 的 pages，也不構成 system-wide cold-cache 宣告。六個 installed
expert ranges 的 byte／residency contract 通過，但 4K／32 repeated gate 的 fixed 與
adaptive candidates 都未達 request-time／throughput 門檻，因此預設沒有改變。

自動 prefill step 如下。

| 未快取 prompt token | Step |
| ---: | ---: |
| 少於 1,024 | 128 |
| 1,024 至 4,095 | 256 |
| 4,096 或更多 | 1,024 |

## Prefill 資料路徑

未快取 token 數少於 `layer_major_prefill_threshold` 時，DeepSeek runtime 使用 mlx-lm 的 chunk-major path。

未快取 token 數達到 `layer_major_prefill_threshold` 時，DeepSeek runtime 使用 layer-major path。
預設門檻是 1,024。

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

`qwen_grouped_experts` 已採用為 Qwen 預設。開啟時，Qwen 在 MTP 關閉且有 batched expert buffer、
expert 分配數至少 64 的情況下，先依 expert 排序輸入，再執行原本兩次 MXFP4 QMM，最後還原順序。
兩次 QMM 都明確使用 `sorted_indices=False`，保留已驗證的運算路徑。
Individual-expert Decode 與 MTP 不使用這個排序路徑，現有 prompt-cache format 5 不變。
CLI 提供 `--qwen-grouped-experts`／`--no-qwen-grouped-experts`；model catalog／configure API
可選填同名 boolean，省略時依 model kind 補值：Qwen 為 `true`，DeepSeek 為 `false`。
App 的 Qwen 進階設定提供「Prefill 加速」開關，預設開啟。
`ModelAdvancedSettings.qwenGroupedExperts` 保存選擇，並編碼為 catalog 的 `qwen_grouped_experts`；
舊 App 設定省略時仍採用 Qwen 預設，明確的 `false` 會保留。
未載入模型沿用 configure 同步流程，下次載入生效；MTP 開啟或 layer-major prefill 關閉時停用開關。
目前驗證與適用範圍見 [Qwen 支援](QWEN.md)。

Qwen 的 `ane_prefill` 預設開啟。
只有 1,024-token chunk 會進入 ANE 路線。
`ane_prefill_ratio` 預設是 0.25。
Runtime 會把比例對齊到 256 個 output channels。
預設由 GPU 計算前 9,216 個 output channels。
ANE 同時計算最後 3,072 個 output channels。
runtime 只串接兩個結果，不執行跨裝置 reduction。
比例 0 使用完整 GPU projection。
比例 1 使用完整 ANE projection。
其他 chunk、Decode 和 ANE 停用狀態會使用原本的完整 GPU projection。

Runtime 會動態載入 private `AppleNeuralEngine.framework`。
Runtime 會檢查必要的 Objective-C class 和 selector。
Compile、load 或 evaluate 回傳錯誤時，runtime 會停用該 loaded model 的 ANE 路線。
Evaluate 失敗的當次 projection 會用原本的完整 GPU projection 重算。
private framework 沒有 OS 相容性保證。
該 path 每次只保留一個完整 expert layer buffer。

Stopped adaptive prefill prototype 在當層 attention 完成後只計算一次 router／shared
expert，形成每層 union，再依 0.7／0.8／0.9 threshold 選 full 或 selective read。
Selective read 仍配置完整 256-row fixed Metal-visible buffer，把六個 canonical regions
寫入原 expert row offset，並執行相同 batched `gather_qmm`。這條 path 明確失去目前的
next-layer prefetch overlap。`repeated` 4K runtime gate 雖把 request expert bytes 降低
66.78%，TTFT paired median 卻回退 16.87%，所以預設仍執行 full-layer path。

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

runtime 與 App 預設只使用記憶體快取，不建立或讀寫磁碟快取目錄。
App 的 Model Advanced Settings 提供「不使用／記憶體／磁碟」；修改在下次載入模型時生效。
舊 App 設定沒有模式欄位時遷移為記憶體；既有磁碟檔案不刪除。
`--prompt-cache off|memory|disk` 提供相同選擇；off 以 `prompt_cache_entries=0` 停用
跨請求保存與重用，單次生成所需的 KV state 仍存在。DeepSeek V4.1 不支援此功能。

runtime 預設保留兩個記憶體 timeline。
runtime 把記憶體用量限制在 8 GiB。
每個記憶體 entry 都包含 `approximation_mode`。
Exact request 只重用 exact entry。
`learned-route-drop-lowest-1` request 只重用相同 mode 的 entry。
Approximate entry 不會寫入 persistent prompt cache。

選擇磁碟模式後，runtime 把最多八個 entry 寫到：

```text
~/.dsmodel/prompt-cache/<model-id>/<checkpoint-revision>/format-<manifest-format>/
```

一般 target-only persistent cache 使用 format 5。每個 identity 從完整 cache contract
開始，依 128-token blocks 建立 SHA-256 parent chain；terminal block key 同時命名 immutable
metadata 與 quantized safetensors payload。Contract 包含 model ID、checkpoint revision、
完整 canonical `config.json` SHA-256、顯式 RoPE 欄位、KV／index format、attention
implementation／window／compression settings、state schema 與 block size。Scanner 會重算
contract、所有 block descriptors 與 terminal key；任一欄位不符即忽略。舊 normal format
1／2 因缺少足以證明相容性的 contract 而不再載入；format 4 也不再載入，避免重用可能
含有 mutable-state alias 的舊 prefill checkpoint。

非 layer-major prefill 會暫存最多兩個輸入快照，完成請求後加入受筆數限制的記憶體快取；
只有磁碟模式另外寫入磁碟。這讓記憶體模式也能重用短輸入的最後一個 token 前的狀態。
兩個快照分別是：第一個完成的 prefill
chunk 與最後一個 `prompt[:-1]` checkpoint；完成 request timeline 仍照常保存。Checkpoint
會在 prefill callback 當下深複製並 materialize cache state，避免 Qwen `ArraysCache` 的可變
state list 被後續 final-token evaluation 或 decode 改寫。新 prompt 即使只剩一個 token 要
replay，也能選擇最長的已保存安全 prefix。相同 contract／tokens 只建立
一組 immutable data／metadata；reuse count 與 last-access time 寫在獨立 sidecar，不改寫
content-addressed payload。Normal eviction 先保留 reuse count 較高者，再比較 access
recency，因此重複的 system／tool prefix 優先於一次性 suffix。Payload 是 cumulative cache
checkpoint；目前沒有把每層 KV 切成可獨立 dedupe 的 delta objects。

當 `dspark_prompt_cache=true` 時，runtime 使用獨立的 format 3 namespace；同一 entry 原子
保存 target cache、三個 DSpark context states、token prefix、revision 與 target layers。
Format 5 scanner 忽略 DSpark bundle，format 3 scanner 也忽略一般 entry。Runtime 會忽略
無法載入、缺少任一 context、revision／target layers 不符或只有半份檔案的 cache entry。
兩個 namespace 各自套用 entry 上限與 eviction，不會互相 prune。

### Exact 與選用近似模式

一般 DeepSeek API 和 command-line request 預設使用 `exact`。
App 的 Use approximate mode 預設關閉；開啟後 catalog 的
`defaults.approximation_mode` 會改為 `learned-route-drop-lowest-1`。
Client 可以明確覆寫 mode。
Qwen 和啟用 DSpark 的 DeepSeek 預設使用 `exact`。
`GenerationOptions.approximation_mode` 的內部安全預設仍是 `exact`；server 和 CLI
會依 model kind 與 DSpark 狀態選擇實際預設。
Runtime 持有 generation lock 時，會暫時把 40 個 learned router 從 top-6 改為 top-5。
三個 hash router 保持 top-6。
Runtime 使用 `finally` 還原 learned router，因此正常完成、錯誤和中止都會回到 top-6。
V4.1 與 Qwen 也可明確啟用 top-k 減一；預設皆為 Exact。
DSpark 和 MTP 會拒絕這個 mode。
App request 沿用模型設定；DSpark 開啟時固定使用 Exact。

## DSpark

DSpark 是可選功能。
App 下載的 DeepSeek installed model 固定包含 DSpark weights。
runtime 預設不啟用 DSpark。

以下為 V4 的 DSpark 合約。V4.1 的 128 專家、top-3 合約見
[三模型加速功能](MODEL_ACCELERATION.md#v41-dspark-安裝)。

| 欄位 | 值 |
| --- | ---: |
| layers | 3 |
| block size | 5 |
| target layer IDs | 40、41、42 |
| Markov rank | 256 |
| noise token ID | 128,799 |
| expert slots | 768 |

一次 DSpark backbone 會產生五個固定 position logits；每個 position 再由前一個候選
token 經 rank-256 Markov embedding／projection 加上 conditional bias。現行 runtime
逐位置取 argmax 或 sample，因此 `DraftResult` 仍只有一條路徑。Research-only
branch-4／beam-8 script 已證明可由同一次 backbone 組成 coherent Markov paths，但五組
first-round gate 有 0/5 acceptance-preserving storage improvements，candidate 已停止。
Runtime、CLI、server 與 App 都沒有 multi-candidate path setting。Sampling path selection
也未實作，因 arbitrary selector 會改變 proposal probability。

Layers 3--42 learned-router prefetch 也沒有 runtime path。Research baseline 曾把三個
DSpark layer heads 與 final norm 直接送入各 target layer 的 frozen `ffn_norm`／router；
6,000-label gate 的最佳 top-24 只有 17.15% assignment recall 與 11.15% useful union
rate，所以 direct transfer 已停止。Runtime 不建立 learned-layer scratch、probation 或
deadline pinning；未來若有 trained predictor，必須使用獨立 namespace／artifact contract
並先通過 executed useful／wasted bytes 與 exact output gate。

Transformers `config.json` 的 `num_nextn_predict_layers=1` 不會被 pinned official
inference graph 使用。該 graph 依 `inference/config.json` 建立三個 `mtp.*` stages，
從 `mtp.0` 的 target-hidden adapter 進入，依序執行三層，再由 `mtp.2` 的 norm、
HyperHead、Markov head 與 confidence head 輸出。Checkpoint index 與 installed
manifest 都沒有同時擁有 input adapter 與 output head 的 self-contained stage。
因此目前 runtime 沒有 faithful native MTP-1 模式，也不把裁層視為
checkpoint-equivalent。完整證據見
[`MTP-1 contract audit`](benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json)。

DSpark 使用獨立 expert cache。
768 個 DSpark slot 的 expert payload 容量是 9.56 GiB。
Main 與 DSpark expert cache 目前都以 `(layer, expert)` 作 key，但相同數字指向不同目錄與
不同 checkpoint weights；在沒有 model namespace、slot ownership 與 fence 測試前，
不能直接共用同一個 expert cache。普通 prompt cache 不保存 layers 40–42 hidden taps 或
三個 DSpark attention contexts，因此仍不能只恢復 target KV。Default-off
`dspark_prompt_cache` 另以一個 atomic entry 保存 target KV 與三個 context states，且只在
兩者已消耗相同 `prompt[:-1]` prefix 後 admission。128-token installed-model gate 已通過
同程序 memory hit 與重啟後 persistent hit；這不改變獨立 768-slot expert cache。

96-slot reduced-cache research candidate 可容納一個完整 draft 的最壞 90 assignments。
三組 4K／32 pilot 保持 exact tokens 且降低 observed peak memory；4K／128
`random_hex` 也保持 exact，但 draft expert bytes/committed token 增加 83.17%，超過
預先宣告的 +50% 停止線。96-slot candidate 已停止，768-slot 預設不變。詳見
[`state ownership audit`](../research/DSPARK_STATE_OWNERSHIP_2026-08-27.md)。

預設 block verifier 一次處理完整 verification block。
verifier 在每個 round 建立一次 cache fork。
每層把 block 內 routed expert assignments 合併成 unique expert union；
每個 expert blob 最多取得一次，且同 expert 的 token rows 集中計算。
若 draft 被拒絕，runtime 只 replay committed prefix。
runtime 會在 speculative cost 高於 target baseline 時 fallback。
`--no-dspark-fallback` 可在隔離研究中繼續執行，但不改變 would-trigger 診斷；
正式 runtime 預設與建議值都維持 fallback 啟用。

`--dspark-sequential-verification` 會保留單一 round-level cache fork，但依 target
autoregressive shape 一次只執行一個 verification token；若 draft 被拒絕，committed
prefix 也逐 token replay。它是 correctness oracle，不做 block attention／MoE union
加速。此模式明確禁止 hash exact prefetch，因目前 speculative scratch 會切換
ready-expert execution path，無法再把結果只歸因於 verifier shape。2026-08-26 的
`random_hex` 4K／32 fixed 與 adaptive oracle 都逐 token 精確匹配 sequential-prefill
reference；這把該 workload 的既有分歧縮小到 block-shaped verification boundary，
但不代表 oracle 是效能候選。

2026-08-27 的 layer-wise diagnostic 進一步比較同一 common prefix 下的 sequential 與
two-token block。Input embeddings 完全相同，但第 0 層 `LocalAttention` 後已出現
非 exact hidden states；第 0 層 router 選擇仍完全相同，learned router 到第 12／16 層
才依 position 首次改變 expert set。這是目前最早觀測到的 correctness boundary，尚未
證明 `LocalAttention` 內某個子算子是唯一原因。逐層資料位於
[`layer parity diagnostic`](benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json)。

Layer 0 component diagnostic 再拆開 HyperConnection、Q／KV projection、RoPE、rotating
cache、attention output 與 output projection。兩個位置的 input hidden、collapsed
attention input、attention norm、`wq_a`、`q_norm`、`wkv`、KV norm 與新 KV RoPE 都
exact。嚴格逐值比較最先在 HyperConnection `post`／`combine` 看到極小 shape-dependent
差異；`wq_b` 也在 exact `q_norm` input 下非 exact，因此不是只有 cache state 的差異。
Sequential 使用兩次 one-token in-place cache update 且沒有 mask；block 使用一次
two-token concatenate update 與 2x129 mask。兩者 raw fetched cache 長度不同，但
temporal-order cache 的共同 128-token suffix 完全一致。這些同時改變的 execution shapes
仍無法證明單一 causal kernel。Component 資料位於
[`layer 0 attention component diagnostic`](benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json)。

FFN component follow-up 又顯示 layer 0 router IDs／scores、routed selected outputs、
routed reduction 與最終 MoE output exact；FFN HyperConnection `post`／`combine` 先出現
shape-dependent 差異。Shared expert output 各有一個值不同，但在 MoE target dtype
邊界被消除。資料位於
[`layer 0 FFN component diagnostic`](benchmarks/2026-08-27-dspark-layer0-ffn-component-diagnostic-m2-max.json)。

`--dspark-hybrid-verification` 是預設關閉的 verifier candidate。它保留每 round 一次
cache fork；每層的 attention、FFN HyperConnection、router、shared expert、routed
expert math 與 final expand 都依 autoregressive one-token shape 執行。Runtime 會先收集
該層所有 token 的 selected expert IDs，以一次 `get_many` acquire expert union，再用
同一批 resident weights 逐 token 計算 routed experts。因此它保留 union I/O 去重，
但不保留原 block verifier 的 grouped multi-row QMM。Rejected committed prefix 也走
相同 hybrid path。Hybrid 與 sequential oracle 互斥；hybrid 可與 hash exact prefetch
及 adaptive selector 組合。

候選分三步收斂。V1 只讓 attention token-shaped；單一 exact cache state 恢復 near-tie
top token，但 `random_hex` 4K／32 多 round 在 index 15 再次分歧。V2 再讓 FFN
HyperConnection token-shaped；五組 discovery workloads 中 3 組 exact，
`storage_sentence` 與 `multilingual_choice` 仍分別在 index 16／6 分歧。V3 把 MoE math
也改為 token-shaped、仍每層只 acquire 一次 union；五組 4K workloads 的 normal／fixed
greedy tokens 全部 exact。V3 correctness artifact 位於
[`hybrid v3 five-workload gate`](benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json)。
這是 32-output-token decision survey（`balanced_choice` 在 5 tokens EOS），不是 sampling
proof、長 decode proof 或 performance adoption evidence。

Metrics 以 `dspark_verification_expert_union_calls` 累計 target verification／replay
實際執行的 `get_many` 次數，並以 assignments、union experts、reuse、misses、bytes 與
read time 分開描述 acquisition。128-token `repeated` 的四波 gate 在相同 accepted
5-token block 上量到 sequential／grouped／hybrid calls 為 258／43／43。Hybrid 相對
sequential target bytes -12.31%，但 verification time +17.34%；相對 grouped
verification time +87.59%。這確認 one-per-layer acquisition，卻也顯示目前所有
token-shaped target execution 的 aggregate cost 是 material。Grouped 與 hybrid 還會
產生不同內部 union，因此 timing 不能單獨歸因於 QMM。Grouped 仍因既有 low-margin
correctness failure 停止，hybrid 仍只是 default-off correctness implementation。

`--dspark-hash-prefetch` 啟用實驗性 exact prefetch。前三個 main model
router 直接以 checkpoint 的 `tid2eid[token_id]` 查表；draft block 完成後，runtime
會依 block 內第一次使用的位置建立 per-layer expert union。主 LFU cache 已 resident
的 expert 在 transaction 期間暫時 pin；其餘 expert 讀入獨立 verification scratch，
不做 LFU admission。scratch 在目前固定 5-token DSpark block 下最多配置 108 個
expert blob slots，並在 initial verification 與 rejected-prefix replay 之間重用。
此功能預設關閉。2026-08-26 的單一 full-model greedy smoke 已通過 output token
hash parity 與 logical-byte accounting，但不是正式速度結果。
2026-08-27 與 hybrid v3 組合後，五組 4K normal／fixed outputs 仍全部 exact；每組
`useful + wasted = hash_prefetch_bytes_read`，useful rate 範圍是 23.53% 至 56.23%。
Artifact 位於
[`hybrid v3 + hash`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json)。

`--dspark-adaptive-block` 會在 DSpark 產生完整草稿後，對 1、2、4 與 checkpoint
最大 block size（目前為 5）建立候選 prefix。runtime 將 confidence 當成 conditional
survival probability，計算預期 committed tokens；再以 checkpoint hash routes 與主 LFU
cache 的即時 resident snapshot，計算每個候選的 missing hash experts。第一版 score 是：

```text
expected committed tokens / max(1, missing hash experts)
```

送入 selector 前，runtime 會先把 draft 限制為最多
`remaining output tokens - 1`；保留的一個位置供 bonus 或 correction token 使用。
這個 output-budget truncation 與 adaptive score truncation 分開計量。
若完整候選的預期 committed tokens 除以 `draft tokens + 1` 至少為 0.90，校準後的
護欄會直接保留完整 block；否則才使用上述 storage score。

selector 只使用前三個可 exact lookup 的 hash layers，不估計其餘 40 個 learned-router
layers。選擇 prefix 不會省下 DSpark 本身的完整 5-position forward，只改變 target
verification、exact prefetch 與可能的 round 數。既有 confidence threshold 會先做
hard prefix truncation，output budget 再限制可驗證長度，adaptive selector 最後從
留下的長度建立候選。此功能同樣預設關閉。

2026-08-27 的 hybrid v3 + hash + adaptive 五組 survey 共有 63 個 adaptive decisions：
selected length 1／2／3／4／5 分別出現 56／4／1／1／1 次。Normal、fixed 與 adaptive
的完整 output token 序列逐組 exact；adaptive hash-prefetch useful rate 是 65.22% 至
91.31%，但每層 union assignment reuse rate 降到 15.59% 至 23.98%。這顯示縮短 block
可減少 rejected-only prefetch，同時犧牲 block 內 expert reuse。Artifact 位於
[`hybrid v3 + hash + adaptive`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json)。
本 survey 沒有 warmup、沒有控制 OS page cache、每模式只有一 run；不得用 request
time、process disk bytes 或 peak memory 宣稱採用。

DSpark path 不讀取一般 prompt cache；只有明確啟用 `--dspark-prompt-cache` 時才讀取獨立
atomic DSpark namespace。預設仍每次執行完整 prompt prefill。
目前量測沒有證明 DSpark 具有淨加速。
詳細決策請見[研究結論](RESEARCH.md)。

## Server 與 APP

APP 在啟動 server 前建立 version 1 model catalog。
model catalog 只包含啟動時可用的 installed model。
damaged model 不會進入 model catalog。
APP 使用權限 `0600` 的暫存檔把 model catalog 傳給 server。
APP 在 server 停止或啟動失敗後刪除暫存檔。

server 啟動時驗證初始 model catalog。
server 啟動時不建立 `ModelRuntime`。
`GET /v1/models` 也不建立 `ModelRuntime`。
第一個 generation request 會載入 request 指定的 installed model。
`POST /api/models/configure` 可以更新未載入模型的 model catalog entry。
`POST /api/models/load` 也可以明確載入指定的 installed model。
APP 手動載入模型時，會把最新 model catalog entry 一起傳給 server。
`POST /api/models/unload` 可以明確卸載指定的 installed model。
server 讓該 installed model 保持載入。
API model ID 和該模型的 Alias 共用同一個 runtime。

`ModelManager` 持有 generation lock。
generation lock 涵蓋模型切換、模型載入、warmup 和完整 streaming request。
手動載入和卸載也使用 generation lock。
其他 generation request 會依序等待。
切換模型時，`ModelManager` 先關閉舊 runtime。
`ModelManager` 接著釋放引用、執行 Python GC、清除 MLX cache，然後載入新 runtime。
server 一次只保留一個載入的 runtime。
模型載入失敗後，下一個 request 可以再次嘗試載入。

`ThreadingHTTPServer` 可在載入或 generation 期間回應 `/healthz`、
`/v1/models` 和 `/api/status`。

APP 使用獨立 Python process 啟動 server。
APP 每秒讀取 `/api/status`。
server 不會把 APP 的 `/api/status` polling 寫入 access log。
APP 使用 process RSS 顯示記憶體。
APP 只建立目前顯示的頁面。未顯示的頁面不會參與 SwiftUI layout。
Server 頁面只包含 server 狀態、系統檢查和 server 設定。
Server 可以使用空 model catalog 啟動。
Model 頁面包含 Loaded 區塊、模型清單、模型資料夾、下載、驗證、repair、DSpark 和模型進階設定。
Loaded 區塊位於 Model 區塊上方。
已載入模型只顯示在 Loaded 區塊。
每個模型列可以手動載入或卸載模型。
Model 頁面的選擇只控制模型管理和進階設定。
該選擇不控制 server 載入的模型。
每個模型的進階設定頁提供選用的 Alias。
APP 會驗證 Alias，並自動儲存有效的變更。
server 執行期間，APP 只會停用 Loaded 或 Loading 模型的 Alias 和模型進階設定。
APP 會把未載入模型的有效變更同步到 server。
這些變更會在模型下次載入時生效。
server 執行期間完成下載後，使用者必須重新啟動 server。

APP 將 Generate 和 Runtime 設定依 model kind 分開儲存。
Power Saving Mode 是所有模型共用的設定。
`ServerConfiguration` 只保存 server 設定。
`ServerConfiguration` 不保存 installed model path 或 Alias。
Server 頁面的 Log 層級會保存到 `ServerConfiguration`，並在下次啟動 server 時以
`--log-level` 傳入。預設 Info；Debug 額外輸出完整 JSON request body；Error 只保留
HTTP 4xx／5xx access log。
升級時，APP 會把舊的自訂 `publicModel` 遷移到當時所選模型的 Alias。
舊的 Qwen 預設名稱不會遷移成 Alias。
Chat 頁面只列出目前 server model catalog 內的模型。
Chat 頁面對每個 installed model 只顯示一個選項。
Alias 存在時，Chat 頁面使用 Alias 作為 request 名稱，並同時顯示 API model ID。
`ChatSession` 會合併 50 ms 內收到的 streaming deltas，再發布一次訊息更新。
Generation 完成、失敗或停止時，`ChatSession` 會先發布剩餘 deltas。
這個設計減少長回覆的文字重排與自動捲動次數。
切換 Chat 模型不會清除對話。
`ContentView` 持有 Chat 回覆任務。切換頁面不會中止正在進行的回覆。
使用者按下 Stop Generating 或關閉 `ContentView` 時，APP 會取消回覆任務。
每個 assistant 訊息會保存該 request 使用的模型名稱。
Metric 頁面顯示載入中或已載入的 API model ID。
APP 偵測到模型切換時會清除效能歷史。
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
- MTLIO、custom Metal expert kernel 和 learned prefetch predictor 尚未整合；native
  MTLIO bytes/shared/private、shared-event 與 cancellation gate 已通過，但 installed
  MLX 0.32.0 沒有支援 external `MTLSharedEvent` dependency handoff，因此停止 runtime
  integration。Hash-layer exact prefetch 只有預設關閉的 prototype；4K／32 explicit
  bypass-policy repeated gate 已拒絕目前 fixed hybrid + hash 候選。
- Internal staged `w13`／`w2` split-slot prototype 的 byte／token correctness gate 通過，
  但 128／32 四波 request +2.17%、Decode -4.26%，因此 candidate 停止、預設維持
  `false`，且不暴露在 CLI、server 或 APP。
- Internal adaptive expert prefill prototype 的 route、selected-row、batched-byte 與
  output correctness 通過，但 `repeated` 4K TTFT paired median +16.87%、p95 +17.02%；
  70%／80%／90% 在此 workload 的決策相同且都失敗，因此維持 full-layer 預設。
- DSpark 可執行，但 DSpark 預設停用。

## 主要程式碼

- Checkpoint 合約：[`Sources/DeepSeekRepack/Model.swift`](../Sources/DeepSeekRepack/Model.swift)
- Checkpoint 讀取：[`Sources/DeepSeekRepack/Checkpoint.swift`](../Sources/DeepSeekRepack/Checkpoint.swift)
- Repacker 與完整驗證：[`Sources/DeepSeekRepack/Repacker.swift`](../Sources/DeepSeekRepack/Repacker.swift)
- Runtime 設定與 model path：[`runtime/deepseek_v4_ssd/model.py`](../runtime/deepseek_v4_ssd/model.py)
- Expert cache：[`runtime/deepseek_v4_ssd/expert_cache.py`](../runtime/deepseek_v4_ssd/expert_cache.py)
- Process disk-I/O metrics：[`runtime/deepseek_v4_ssd/io_metrics.py`](../runtime/deepseek_v4_ssd/io_metrics.py)
- Generation 與 prompt cache：[`runtime/deepseek_v4_ssd/generation.py`](../runtime/deepseek_v4_ssd/generation.py)
- DSpark：[`runtime/deepseek_v4_ssd/dspark.py`](../runtime/deepseek_v4_ssd/dspark.py)
- Server：[`runtime/deepseek_v4_ssd/server.py`](../runtime/deepseek_v4_ssd/server.py)
- Model manager：[`runtime/deepseek_v4_ssd/model_manager.py`](../runtime/deepseek_v4_ssd/model_manager.py)
