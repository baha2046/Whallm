# SSD Streaming 瓶頸研究與執行計畫

狀態：Active

本文件整理「研究 SSD 串流瓶頸」對話。
本文件也把對話方向對照目前程式碼和量測。
對話內容是研究輸入。
對話內容不是本專案的速度證據。

## 結論

本研究的主要目標不是單獨提高 SSD 峰值速度。
本研究的主要目標是降低每個輸出 token 暴露在關鍵路徑上的 expert blob 搬移時間。

```text
                         routed expert route
                                  │
                                  ▼
┌──────────────┐   miss   ┌─────────────────┐   ready   ┌──────────────┐
│ Expert cache │─────────▶│ SSD expert blob │──────────▶│ Metal compute│
└──────┬───────┘          └─────────────────┘           └──────────────┘
       │ hit
       ▼
  fixed slot

改善點：
  1. 每次 miss 少讀 bytes。
  2. 每個 token 少發生 miss。
  3. 用計算遮蔽仍然必要的讀取。
```

可使用下列概念模型。

```text
expert bytes/token
≈ layers × selected experts × miss rate × expert blob bytes
  + wasted prefetch bytes

exposed stall/layer
≈ max(0, read time + reconstruction time - overlap window)
```

Hit rate 不是唯一指標。
預取可以提高表面 hit rate，也可以增加無用讀取。
每個正式實驗必須同時記錄下列項目。

- demand hit rate。
- expert bytes/token。
- wasted prefetch bytes。
- ready-before-deadline rate。
- 暴露的 read wait。
- output token hash。

## 對話提出的研究方向

### 1. 降低每次 miss 的 bytes

第一個方向是直接使用 kernel 可執行格式。
installed model 不應在 miss 後重新轉置或展開完整精度權重。

第二個方向是分段讀取 routed expert。
DeepSeek 可以先讀 `w1`／`w3`，再讀 `w2`。
前段計算可以嘗試遮蔽後段讀取。

第三個方向是壓縮或位元切片。
這個方向必須依實際 expert blob 格式決定。
一般壓縮不能先假設對 FP4、FP8 和 BF16 有相同收益。

壓縮候選必須通過下列損益條件。

```text
raw time = S / Bssd
compressed time = rS / Bssd + S / Bdecode

串行路徑要獲利時：
Bdecode > Bssd / (1 - r)
```

`S` 是原始 expert blob bytes。
`r` 是壓縮後比例。
`Bssd` 是實際讀取頻寬。
`Bdecode` 是解壓縮輸出頻寬。

Apple Silicon 使用統一記憶體。
CPU 解壓縮、Metal compute 和資料寫入會共用記憶體頻寬。
因此單獨的解壓縮 microbenchmark 不能證明端到端收益。

### 2. 降低 miss 次數

全域 LRU 不理解 layer 位置和 forward 週期。
本專案目前也不是使用全域 LRU。
本專案使用逐層保留量的全域 LFU。

對話提出下列候選。

- 將一個 token／layer 的 routed expert 視為一個 event。
- 在 event 完成前保護全部 selected expert。
- 比較 current／stale generation。
- 使用 request-local reuse。
- 對一次性 miss 執行 admission control。
- 使用每個 byte 可避免的暴露停頓作為 cache value。

所有 causal policy 都必須和 Belady oracle 比較。
Belady oracle 可以看到完整未來 route。
Belady oracle 只表示理論下限。
Belady oracle 不能直接放入 runtime。

### 3. 遮蔽必要的 miss

預取必須受 overlap window 限制。
預取候選不能只依預測機率排序。

```text
prefetch budget bytes
≈ effective SSD bandwidth × available overlap window
```

每個預取還需要 deadline。
deadline 是 routed expert 開始計算前的最晚完成時間。

MTP／DSpark 可以提供較長的預取時間。
但是較長的 draft 也會擴大 routed expert union。
這類候選應使用下列指標。

```text
accepted tokens / nonresident expert bytes
```

### 4. Exact 與 approximate 必須分開

Cache policy、讀取排程和無損壓縮可以維持 exact output。
位元切片、cache-aware routing 和 expert 替換可能改變 output。
這些候選必須使用明確的 approximate mode。
這些候選也需要獨立的品質和安全門檻。

## 與目前專案的對照

| 對話方向 | 目前狀態 | 決定 |
| --- | --- | --- |
| Kernel 可執行 expert blob | 已採用 | DeepSeek routed expert 使用 checkpoint-native FP4。installed model 保存 canonical expert blob。 |
| 全域 LRU | 不適用 | Runtime 已使用全域 LFU、逐層保留量和同 event 保護。 |
| Route trace | 已實作 | Trace 保存 Prefill histogram、實際 Prefill cache access、Decode route 和真實 miss。Trace 會同步 router index，所以 trace wall time 不是速度結果。 |
| `w1`／`w3` 與 `w2` 分段串流 | 已測試並停止 | Full-model candidate 的 Decode throughput 是 -4.26%。只有不同 kernel 或不同 overlap window 才能重啟。 |
| MTLIO direct path | 已查核並停止 | Native byte gate 通過。MLX 0.32.0 沒有支援 external Metal event handoff。 |
| Selective Prefill read | 已測試並停止 | Expert bytes 降低，但 `repeated` 4K 的 TTFT paired median 增加 16.87%。 |
| Physical SSD bytes | 目前無法歸因 | Runtime 有 logical bytes、process disk bytes 和 page-cache residency proxy。這些值都不能改稱 physical SSD bytes。 |
| 一般壓縮 expert blob | 尚未量測 | 必須先量測實際 FP4／FP8 expert blob。不能從 BF16 研究直接推論。 |
| Cache replacement oracle | Phase 1 已通過 | 五種 workload 的 current replay 全部符合 runtime miss。四種非 repeated workload 的 Belady miss 改善是 41.77% 到 46.16%。 |
| Deadline-ready trace | 尚未完成 | 只有部分 ready/read 指標。缺少統一 deadline replay。 |

外部研究提供方向，不提供本專案速度結論。
[LLM in a Flash](https://arxiv.org/abs/2312.11514) 將重點放在減少搬移 bytes 和增加連續讀取大小。
[MoE-Infinity](https://arxiv.org/abs/2401.14361) 使用 request-level activation trace 來改善 expert cache 和 prefetch。

## 執行規則

Agent 每次只執行第一個標示「下一步」的階段。
Agent 不得跳到後續 runtime prototype。

所有 runtime 候選預設關閉。
Quick gate 和 formal gate 都通過後，Agent 才能討論修改預設值。

原始 trace 和暫存輸出放在 `scratch/`。
通過驗證的 machine-readable 結果放在 `docs/benchmarks/`。
每個 formal artifact 必須記錄 commit、裝置、模型 revision、workload、slot 數、cache 狀態和 output token hash。

Repeated workload 只能作 regression control。
Repeated workload 不能代表一般 expert reuse。

## Phase 1：Cache replacement 上限

狀態：**完成。2026-09-01 跨 workload gate 通過。**

### 問題

只改 cache replacement policy，最多可以降低多少 expert bytes/token？

### 已完成的工具改動

[`Scripts/simulate_qwen_lfu.py`](../Scripts/simulate_qwen_lfu.py) 現在比較下列 policy。

- 目前 LFU。
- LRU。
- Belady oracle。

Simulator 以 token-major、layer-major 順序重播 Decode route。
Simulator 將同一個 token／layer 視為一個 event。
Simulator 在 event 期間保護全部 selected expert。

Belady oracle 可以看到完整未來 route。
Belady oracle 不使用逐層保留量。
因此 Belady 結果是寬鬆的理論下限。

每個 policy 輸出下列欄位。

- `hits` 和 `misses`。
- `demand_hit_rate`。
- `expert_bytes`。
- `expert_bytes_per_token`。
- `required_expert_bandwidth_gbps`。

### 執行方式

```sh
.venv/bin/python Scripts/simulate_qwen_lfu.py \
  scratch/routes.json \
  --slots 1152 \
  --target-tokens-per-second 10 \
  --output scratch/cache-oracle.json
```

`--target-tokens-per-second` 必須使用實驗預先宣告的目標。
數值 `10` 只是命令格式範例。

### 輸入

至少使用五種 workload。
Workload 必須包含 code、技術中文、mixed math、tool-like 和 repeated control。
每個 workload 至少保存 256 個 Decode steps。

### Correctness gate

每個 trace 的 `current_matches_recorded` 必須為 `true`。
每個 trace 必須保存固定 prompt hash 和 output token hash。
任何 replay 不一致都會停止此階段。

### 決策 gate

- 如果所有非 repeated workload 的 Belady 改善都小於 5%，停止 replacement-only 方向。
- 如果任一非 repeated workload 的 Belady 改善至少 10%，進入 Phase 2。
- 如果改善介於 5% 和 10%，先增加 route 長度和 workload，不修改 runtime。
- 即使 miss 降低，目標速度所需頻寬仍高於可用頻寬時，cache replacement 不能單獨解決瓶頸。

### Artifact

正式輸出位置：

```text
docs/benchmarks/2026-09-01-expert-cache-oracle-m5-pro.json
```

### 2026-09-01 formal gate

本輪使用 Qwen3.8-Flash-Next-FP8 installed model。
每個 workload 保存 256 個 Decode steps。
每個 workload 使用 1,152 個 expert slot。
目標速度投影是每秒 10 個 token。

| Workload | 目前 LFU miss | LRU 改善 | Belady 改善 | 目前 LFU GB/s | Belady GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 78,565 | 11.48% | 45.92% | 8.01 | 4.33 |
| mixed math | 69,688 | 8.26% | 43.52% | 7.11 | 4.01 |
| tool-like | 76,776 | 8.11% | 41.77% | 7.83 | 4.56 |
| 技術中文 | 65,883 | 9.14% | 46.16% | 6.72 | 3.62 |
| repeated control | 36,141 | 24.12% | 64.96% | 3.69 | 1.29 |

五個 current replay 都和 runtime 記錄的 Decode miss 完全相同。
四個非 repeated workload 的 Belady 改善中位數是 44.72%。
四個非 repeated workload 的 LRU 改善中位數是 8.70%。

本結果通過 Phase 1 決策 gate。
Phase 2 可以研究 causal cache policy。
本結果不支持直接用 LRU 取代目前 LFU。
本結果也不是 runtime throughput 證據。

完整結果位於
[`docs/benchmarks/2026-09-01-expert-cache-oracle-m5-pro.json`](../docs/benchmarks/2026-09-01-expert-cache-oracle-m5-pro.json)。

固定 prompts 與 token hashes 位於
[`docs/benchmarks/prompts/2026-09-01-ssd-cache-oracle/manifest.json`](../docs/benchmarks/prompts/2026-09-01-ssd-cache-oracle/manifest.json)。

原始 trace、runtime metrics 和 replay 位於：

```text
scratch/ssd-cache-oracle-2026-09-01/
```

第一次 repeated replay 暴露 Prefill trace 缺口。
短提示沒有使用 layer-major Prefill。
短提示把多 token routed expert 送入一般 expert cache。
舊 simulator 只重播單 token handoff。

Trace 現在記錄實際 Prefill cache access。
修正後 repeated replay 的 runtime miss 和模擬 miss 都是 36,141。
這項 trace 變更只在明確啟用 `--expert-route-trace` 時生效。

### 2026-08-31 smoke replay

本輪先重播既有 Qwen repeated control。
輸入是 1,024 個 Decode steps 和 1,152 個 slot。
這個 smoke replay 不是跨 workload gate。

| Policy | Decode miss | 相對目前 LFU | Expert bytes/token | 10 token/s 所需頻寬 |
| --- | ---: | ---: | ---: | ---: |
| 目前 LFU | 114,821 | 0% | 292.79 MB | 2.93 GB/s |
| LRU | 146,413 | +27.51% | 373.35 MB | 3.73 GB/s |
| Belady oracle | 72,801 | -36.60% | 185.64 MB | 1.86 GB/s |

目前 LFU replay 和 runtime trace 的 miss 完全相同。
LRU 明顯較差。
Belady oracle 顯示此單一 repeated trace 仍有 replacement 上限。
這個結果不能證明 causal policy 可以取得相同改善。

暫存輸出位於：

```text
scratch/ssd-streaming-cache-oracle/qwen-repeated128-output1025.json
```

輸入 trace SHA-256 是
`01e3de5405396033188e88186629b701bc1215a3684d30680af72db4028b9857`。
暫存輸出 SHA-256 是
`16a2b4e22c45dc6c1d43ac62c0e2a470904ea4132a7c2d9441d0bd11cb6a14a5`。

## Phase 2：Causal cache policy

狀態：**完成。2026-09-01 quick gate 未通過。**

只有 Phase 1 通過時才執行。

先在離線 simulator 加入 forward-aware eviction 和 admission control。
不要先修改 `ExpertCache`。

Candidate 必須使用和 Phase 1 相同的 trace。
Candidate 不能看到未來 route。
Candidate 必須保持 event atomic。

Quick gate 要求 candidate 取得至少一半的 Belady bytes 改善。
Formal gate 要求非 repeated workload 的 expert bytes/token 中位數至少改善 5%。
Formal gate 也要求任何 workload 不得回退超過 2%。

### 2026-09-01 quick gate

本輪先篩選下列 causal policy。

- 移除逐層保留量。
- 使用 10 個暫存 slot 的 admission control。
- 每層固定 slot。
- 每層 slot 上限。
- 使用 Prefill route 次數分配每層 slot。
- 使用 Prefill 與已完成 Decode 次數決定 eviction。

Admission control 的最佳 screening 結果低於 5%。
每層固定 24 個 slot 讓 repeated control 的 miss 增加 33.94%。
這兩個方向在 quick screening 停止。

最佳候選是 `prefill_guided_24`。
候選只使用 Prefill 和已完成 Decode route。
候選不查看未來 Decode route。
候選每 24 個完成的 Decode token 衰減一次使用次數。

| Workload | 目前 LFU miss | 候選 miss | 候選改善 | 取得 Belady 改善 |
| --- | ---: | ---: | ---: | ---: |
| code | 78,565 | 63,750 | 18.86% | 41.06% |
| mixed math | 69,688 | 56,891 | 18.36% | 42.20% |
| tool-like | 76,776 | 62,732 | 18.29% | 43.80% |
| 技術中文 | 65,883 | 53,683 | 18.52% | 40.11% |
| repeated control | 36,141 | 26,220 | 27.45% | 42.26% |

四種非 repeated workload 的改善中位數是 18.44%。
四種非 repeated workload 取得 Belady 改善的中位數是 41.63%。
Quick gate 要求至少 50%。
因此 Phase 2 停止。

候選通過原訂的 5% formal 數值門檻。
所有 workload 都沒有回退。
但是 quick gate 先失敗，所以本輪不建立 runtime prototype。

完整結果位於
[`docs/benchmarks/2026-09-01-expert-cache-causal-policy-m5-pro.json`](../docs/benchmarks/2026-09-01-expert-cache-causal-policy-m5-pro.json)。

原始 replay 位於：

```text
scratch/ssd-cache-phase2-2026-09-01/
```

## Phase 3：Deadline-ready trace

狀態：**完成。2026-09-01 aggregate gate 通過，候選保持 default-off。**

只有 causal cache policy 不能達到目標頻寬時才執行。

Trace 必須新增下列時間。

- read submit。
- read complete。
- compute submit。
- expert deadline。
- prefetch used 或 wasted。

本階段先量測現有 prefetch。
本階段不先加入新的 predictor。

下一個候選只能依可用 overlap window 配置 prefetch bytes。
Formal gate 必須同時改善 ready-before-deadline rate 和暴露 read wait。
Wasted prefetch bytes 不得增加超過 5%。

### 2026-09-01 baseline

Runtime 現在會在明確啟用 `--expert-route-trace` 時記錄下列資料。

- worker future 提交前的 `read submit`。
- 最早的 worker `read start`。
- 最晚的 worker `read complete`。
- `batched_layer` 等待 expert 前的 `expert deadline`。
- 第一個使用該層輸出的 `mx.eval` 前的 `compute submit`。
- 該層請求與實際使用的 unique expert。
- logical useful bytes 與 logical wasted bytes。

本輪重跑 Phase 1 的五種固定內容。
四種長提示使用 layer-major Prefill。
每種長提示都產生 48 個 batched prefetch event。
repeated control 不使用 layer-major Prefill，所以沒有 batched prefetch event。

| Workload | Ready rate | 實際 overlap p50 | 暴露 read wait p50 | Wasted bytes |
| --- | ---: | ---: | ---: | ---: |
| code | 0% | 0.043 ms | 120.1 ms | 54.89% |
| mixed math | 0% | 0.044 ms | 119.0 ms | 52.84% |
| tool-like | 0% | 0.045 ms | 118.8 ms | 52.75% |
| 技術中文 | 0% | 0.044 ms | 120.7 ms | 53.96% |

四種長提示共有 192 個 event。
所有 event 都未在 deadline 前完成。
實際 overlap window 的 p50 是 0.044 ms。
暴露 read wait 的 p50 是 119.7 ms。
logical wasted bytes 比例是 53.61%。

前一層 `compute submit` 到下一層 `expert deadline` 的視窗介於 59.7 ms
和 120.0 ms。
該視窗的 p50 是 79.9 ms。
該視窗的 p95 是 93.9 ms。

這些資料表示 Qwen 現在幾乎在 deadline 才提交該層讀取。
Qwen 目前沒有使用前一層 compute 期間的可用視窗。

下一個實驗只測試一個 default-off 候選。
候選在目前層第一次提交 compute 時，提交下一層 expert read。
候選不加入 route predictor。
候選不改變預設行為。
候選必須使用同一組五種內容通過本節的 formal gate。

完整結果位於
[`docs/benchmarks/2026-09-01-prefetch-deadline-baseline-m5-pro.json`](../docs/benchmarks/2026-09-01-prefetch-deadline-baseline-m5-pro.json)。

原始 trace 與 runtime metrics 位於：

```text
scratch/ssd-prefetch-deadline-2026-09-01/
```

本結果只描述 opt-in trace observer。
本結果不是正式 throughput 結果。
Logical expert bytes 不是 physical SSD bytes。

### 2026-09-01 next-layer prefetch gate

候選使用 `--qwen-next-layer-prefetch` 明確啟用。
候選只在 Qwen layer-major Prefill 生效。
候選在 layer N 第一次提交 compute 前，提交 layer N+1 的完整 expert read。
候選重用既有 batched expert buffer 和 read worker。
候選不使用 route predictor。

本輪使用和 baseline 相同的五種固定內容。
所有 prompt hash 和 output token hash 都相同。

| Workload | Ready event | 暴露 read wait 改善 | Wasted bytes 增加 |
| --- | ---: | ---: | ---: |
| code | 0／48 | 19.36% | 0% |
| mixed math | 1／48 | 33.91% | 0% |
| tool-like | 0／48 | 27.46% | 0% |
| 技術中文 | 0／48 | 29.07% | 0% |

Aggregate ready-before-deadline rate 從 0% 提高到 0.52%。
Aggregate 暴露 read wait 降低 27.41%。
四種長提示的暴露 read wait 都改善。
Logical wasted bytes 沒有增加。

候選通過原訂 aggregate formal gate。
但是 192 個 event 只有 1 個在 deadline 前完成。
三種長提示的 ready event 仍是 0。
候選因此只保留為 default-off research option。

本輪同時觀察到 read duration p50 增加 17.50%。
Trace observer wall time 不能用來判斷正式 throughput。
repeated control 沒有啟動候選，但 observer wall time 仍明顯漂移。
因此本輪不改變 runtime default。

完整結果位於
[`docs/benchmarks/2026-09-01-qwen-next-layer-prefetch-gate-m5-pro.json`](../docs/benchmarks/2026-09-01-qwen-next-layer-prefetch-gate-m5-pro.json)。

候選原始 trace 與 runtime metrics 位於：

```text
scratch/ssd-prefetch-next-layer-2026-09-01/
```

## Phase 4：實際 expert blob 壓縮

狀態：**完成。2026-09-01 gate 未通過。**

本階段只執行 microbenchmark。
本階段不接 runtime。

分別量測 DeepSeek FP4 expert blob 和 Qwen expert blob。
每個 artifact 必須記錄原始格式。
候選可以包含 Apple Compression framework 提供的 codec。
本階段不新增第三方 dependency。

測試必須包含 64 KiB、256 KiB、1 MiB、4 MiB 和完整 expert blob。
測試必須同時量測壓縮比例、解碼輸出頻寬、暫存記憶體和 byte hash。

只有下列條件全部成立時才進入 Phase 5。

- 解壓縮後的 expert blob byte hash 完全相同。
- `compressed read + decode` 的暴露時間至少改善 5%。
- Metal compute 同時執行時仍通過 5% 門檻。
- Peak memory 不增加超過 5%。

本機 M5 Pro microbenchmark 使用 Apple Compression 的 LZ4 和 LZFSE。
測試包含兩個 installed model、五種讀取尺寸和每個 model 三個 expert 位置。
每個 timing case 使用 9 個樣本。
所有 60 個解壓縮結果都通過 byte hash。
所有讀取前頁面都顯示為 nonresident。

完整 expert blob 的中位數如下。
時間改善的正值代表較快。負值代表較慢。

| Model | Codec | 儲存減少 | Decode output | 無 Metal 時間改善 | 有 Metal 時間改善 | Peak RSS 最大變化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek FP4 | LZ4 | 2.43% | 21.10 GB/s | -69.91% | -71.24% | +1.15% |
| DeepSeek FP4 | LZFSE | 7.11% | 1.29 GB/s | -974.74% | -1036.27% | -0.29% |
| Qwen MXFP4 | LZ4 | 2.47% | 26.11 GB/s | +0.36% | -5.08% | +0.36% |
| Qwen MXFP4 | LZFSE | 4.92% | 1.22 GB/s | -564.51% | -591.75% | +0.88% |

Peak memory gate 通過。
無 Metal 和有 Metal 的 5% 時間 gate 都未通過。
因此 exact track 在 Phase 4 停止。
Runtime 沒有接入壓縮，也沒有新增 dependency。

正式 artifact 位於
[`docs/benchmarks/2026-09-01-expert-blob-compression-m5-pro.json`](../docs/benchmarks/2026-09-01-expert-blob-compression-m5-pro.json)。
原始 worker 檔案位於：

```text
scratch/expert-blob-compression-2026-09-01/
```

## Phase 5：Ready／compact／cold cache

狀態：**Blocked。Phase 4 gate 未通過。**

只有 Phase 4 通過時才建立 default-off runtime prototype。

```text
┌────────────────────┐
│ Ready expert       │  fixed slot，可直接執行
└─────────▲──────────┘
          │ promote
┌─────────┴──────────┐
│ Compact expert     │  RAM，需要重建
└─────────▲──────────┘
          │ read
┌─────────┴──────────┐
│ Cold expert blob   │  SSD
└────────────────────┘
```

Prototype 必須使用固定記憶體預算。
Prototype 必須在長 context 下保留 OS headroom。
Prototype 不得讓可重建 expert 觸發大量 swap。

## Phase 6：Approximate 方向

狀態：**進行中。Phase 6A 完成，Phase 6B 下一步。**

本階段包含位元切片、cache-aware routing、expert 替換和 shared base 加 residual。
本階段會改變模型行為或需要訓練。

Agent 不得把本階段合併到 exact runtime。
Agent 必須先定義 API mode、品質集、安全集和停止條件。

使用者已在 2026-09-01 明確要求繼續 approximate 研究。
Phase 6A 已定義 API mode、五個品質 case、五個安全 case 和停止條件。
第一個 candidate 是 `learned-route-drop-lowest-1`。
它只把 40 個 learned-routing layer 從 top-6 改為 top-5。
三個 hash-routing layer 保持 top-6。

Exact 與 candidate 都通過 10/10 entry smoke。
10 組 output token 也完全相同。
這個結果只允許進入 Phase 6B component gate。
這個結果不是 capability、SSD bytes 或速度結論。

研究合約位於
[`APPROXIMATE_MODE_2026-09-01.md`](APPROXIMATE_MODE_2026-09-01.md)。
Artifact 位於
[`docs/benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json`](../docs/benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json)。

## Handoff

Phase 4 已完成，而且 gate 未通過。
下一個 Agent 不得建立 Phase 5 ready／compact／cold runtime cache。
Exact track 沒有下一個可執行階段。
Phase 6A 已完成。
下一個 Agent 只執行 `APPROXIMATE_MODE_2026-09-01.md` 的 Phase 6B。
下一個 Agent 必須使用 fresh worker 量測 next-token logit drift、route count 和 logical expert bytes。
下一個 Agent 不得建立 runtime、API、CLI 或 APP 開關。
