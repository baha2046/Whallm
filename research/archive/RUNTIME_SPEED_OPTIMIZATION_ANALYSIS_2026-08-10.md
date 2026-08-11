# Runtime 速度最佳化推論分析

> [!WARNING]
> 本文件已於 2026-08-11 封存。
> 本文件記錄已完成的推論分析，不描述目前 runtime。
> 目前狀態請使用[研究結論](../../docs/RESEARCH.md)與[效能文件](../../docs/PERFORMANCE.md)。

日期：2026-08-10
最後更新：2026-08-11

## 結論

本文件根據現有程式碼、架構文件、效能文件、研究結論和驗證紀錄，
推論 DeepSeekV4SSD runtime 的加速方向。

本文件是推論分析，不是正式效能結果。
R0 baseline 和 R1 boundary capture 已完成。
R2a 已拒絕 2,048-slot 候選設定。
R2b 已拒絕 2-worker 候選設定。
R2b 已拒絕 8-worker 候選設定。
R2c 已拒絕 4-prefetch-worker 候選設定。
R2c 已拒絕 1-prefetch-worker 候選設定。
R2 設定 sweep 已完成。
R3 已拒絕目前 DSpark 預設啟用。
M1 已完成 code 4K 探索性 ABBA。
M1 已完成 Tool-like 4K 獨立探索性 ABBA。
M1 沒有通過記憶體接受條件。
兩種 workload 都確認 pilot 等級的效能訊號。
M1 的五輪正式效能評估已完成。
主要 workload request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
Repeated Decode p95 regression 是 3.01%。
M1 沒有通過全部採用條件。
專案已拒絕並移除 M1。
P3 `gather_qmm` tile profiling 已完成。
P3 的原始 GPU time 占比是 12.77%。
依涵蓋率校正後是 13.99%。
P3 沒有通過 15% profile gate。
專案沒有建立 isolated MLX build。
P4 attention 類型區塊探索已完成。
512-step 候選改變 39/42 個 MoE layer 的 expert route。
P4 沒有通過正確性 gate。
runtime 已移除 P4 prototype。
D2 routed expert top-6 profile 已完成。
dispatch、shader sample 和涵蓋率校正結果是 Decode wall time 的
0.121%、0.456% 和 0.972%。
D2 沒有通過 3% profile gate。
D3 CSA row trace 已完成。
五種 context 合併後的 row 交集比例中位數是 85.94%。
`gather` 的涵蓋率校正占比最多是 Decode wall time 的 0.34%。
D3 沒有通過 90% row gate 或 5% profile gate。
D3 沒有建立 prototype。
目前計畫沒有其他 active 實驗。

## 目前 Runtime 的事實

### 硬體環境

| 項目 | 值 |
|---|---|
| Mac | MacBook Pro `Mac17,8` |
| SoC | Apple M5 Pro |
| CPU / GPU | 18 cores / 20 cores |
| 統一記憶體 | 64 GB |
| 模型磁碟 | 內接 APFS SSD，FileVault 開啟 |

### 模型合約

| 項目 | 值 |
|---|---|
| Model ID | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| main model layers | 43 |
| routed experts per layer | 256 |
| selected experts per token | 6 |
| hidden size | 4,096 |
| expert intermediate size | 2,048 |
| expert blob size | 13,369,344 bytes (12.75 MiB) |
| expert dtype | checkpoint-native FP4 |

### 目前量測結果（2026-08-10 R0）

R0 使用五種 prompt。
每種 prompt 都有 4,096、8,192 和 14,363 token。
每個 run 都產生 256 個 output token。

| Prompt token | TTFT 中位數 | Prefill 中位數 | Decode 中位數 | Expert bytes 中位數 | Misses 中位數 | Evictions 中位數 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 21.611 秒 | 189.53 Tok/s | 8.00 Tok/s | 473.970 GB | 24,700 | 23,548 |
| 8,192 | 39.124 秒 | 209.39 Tok/s | 7.76 Tok/s | 477.393 GB | 24,956 | 23,804 |
| 14,363 | 67.466 秒 | 212.89 Tok/s | 7.43 Tok/s | 482.593 GB | 25,345 | 24,193 |

每種 prompt 和長度只有一個 run。
跨 prompt p95 表示 workload 差異。
跨 prompt p95 不表示重複執行的變異。

完整 baseline、route profile、Metal capture 和 CPU profile 位於
[`docs/benchmarks/2026-08-10-r0-m5-pro.json`](../../docs/benchmarks/2026-08-10-r0-m5-pro.json)。

### 已實作的優化

下列優化已實作並驗證：

| 優化 | 證據 |
|---|---|
| Layer-major prefill（4K+） | 14K 結果降低 expert bytes 約 94.3% |
| Batched `gather_qmm` | 4K/14K 配對、fixture parity |
| Direct slot read（`preadv`） | 目前程式碼和 full-model run |
| Ready expert decode | 五組配對 hash，改善中位數 12.9% |
| MXFP8 compressed KV cache | cache tests 和 8K greedy token |
| MXFP4 index cache | shape、gather 和 parity tests |
| Persistent prompt cache v2 | restart 和 round-trip tests |
| Automatic prefill step | 128/256/1024 三級 |

### 已測試但未採用

| 項目 | 結果 |
|---|---|
| 單一 contiguous slot arena | 比目前慢 10.1% |
| Custom greedy generator | 比 mlx-lm generator 慢 7% |
| Prefill-to-decode handoff | 單 prompt 證據不足 |
| DSpark 預設啟用 | R3 request 增加 32.58%；總 throughput 降低 24.60% |
| M1 作為記憶體最佳化 | Process peak footprint 增加 1.77 MiB；沒有達到至少降低 5% 的接受條件 |
| P1 平均分配 prefill 區塊 | TTFT 增加 19.8%，被拒絕 |
| P2 略過 CSA indexer | 配對中位數改善 1.41%，低於 3% 門檻 |
| P3 `gather_qmm` tile | 12.77% 與 13.99% 都低於 15% profile gate；未建立候選。 |
| P4 attention 類型區塊 | 512-step 候選改變 39/42 個 MoE layer 的 expert route；prototype 已移除。 |
| D2 routed expert top-6 | 0.121%、0.456% 和 0.972% 都低於 3% profile gate；未建立候選。 |
| D3 CSA row 重用 | Row 交集比例中位數是 85.94%；`gather` 校正占比最多是 0.34%；未建立候選。 |

## 瓶頸推論

### TTFT 分解

R0 的 TTFT 中位數是 21.611、39.124 和 67.466 秒。
對應長度是 4K、8K 和 14K。

R1 的 Tool-like Prefill GPU busy 如下。

- 4K：60.7%。
- 14K：70.2%。

長 Prefill 的 GPU busy 比 4K Prefill 高。
因此，目前資料不支持「Prefill 只受 SSD 限制」。

Expert read timer 與 GPU 計算會重疊。
Routing sync 也包含 upstream graph work。
目前不能把這些 boundary 直接相加。

### Decode 分解

Repeated 4K 的 Decode 是 12.22 Tok/s。
Repeated 4K 有 1,535 次 expert miss 和 383 次 eviction。

四種非 repeated 4K prompt 的 Decode 是 6.44 至 8.28 Tok/s。
這些 prompt 有 24,307 至 28,765 次 expert miss。
這些 prompt 有 23,155 至 27,613 次 eviction。

Repeated prompt 的 route 分布較集中。
Repeated prompt 不能代表一般 Decode 的 expert cache 行為。

R1 的 Tool-like Decode GPU busy 是 35.5% 至 37.1%。
Tool-like Decode GPU idle 是 7.968 至 8.114 秒。
CPU profile 同時有 6.406 至 6.464 秒的 `preadv` leaf sample。
這些 sample 是跨 thread 的 running time，不是 wall time。

目前證據指向一般 Decode 的 SSD 與 slot 壓力。
目前證據不支持「expert cache 不是 Decode 瓶頸」。

### 已知限制

| 限制 | 影響 |
|---|---|
| Batch size = 1 | 無法用 batch 隱藏 I/O 延遲 |
| 單一 generation request | server lock 序列化 |
| 每種 prompt 和長度只有一個 baseline run | 無法估計重複執行的變異 |
| R1 Metal trace 沒有 shader timing sample | R1 無法分開 attention、routed expert 和 shared expert 的 GPU time。P3 已補上 `gather_qmm` shader timing。 |
| request phase alignment 是推定值 | Metal boundary 有對齊誤差 |
| DSpark 預設關閉 | R3 只測試 code 4K 和 64 個 output token；結果足以拒絕目前候選，但不表示所有 DSpark 設計都無效。 |

## 優化方向

### 方向一：Metal boundary capture（已完成）

**目的**：確認 GPU 上 attention、MoE、shared expert 各佔多少時間。

**結果**：R1 trace 可分開 GPU busy、GPU idle 和 CPU sample。
R1 trace 沒有 shader timing sample。
P3 已使用 `Metal GPU Counters` 補上 `gather_qmm` shader timing。

R1 已 capture repeated 4K、Tool-like 4K 和 Tool-like 14K。
Tool-like Decode 的 GPU idle boundary 較長。
CPU profile 沒有顯示 MLX 圖走訪或 identity index 穩定超過 5%。

因此，D1 和 P5 的 profile gate 都沒有通過。
目前不實作 D1 或 P5。

P3 量到 `gather_qmm` 是 request 的 12.77%。
依 shader 涵蓋率校正後是 13.99%。
P3 沒有通過 15% profile gate。
P3 已停止，且沒有建立 MLX fork。

如果其他候選需要 shader-level 證據，研究必須使用 `Metal GPU Counters` 重跑。
Metal capture 開啟時的 wall time 不是正式效能結果。

### 方向二：Worker A/B（已完成）

**目的**：確認 `read_workers` 和 `prefetch_read_workers` 的最佳值。

**理由**：目前預設 `read_workers=4`、`prefetch_read_workers=2`、`slots=1,152`。
Microbenchmark 顯示 SSD direct mode 在 2/4/8 workers 間差異不大（12.94/12.89/12.95 GiB/s），
但這是 I/O 層面。端到端結果可能不同。

R2a 已比較 1,152 與 2,048 slots。

| Prompt | 1,152 slots | 2,048 slots | Decode 差異 |
|---|---:|---:|---:|
| Repeated 4K | 14.42 Tok/s | 13.95 Tok/s | -3.31% |
| Code 4K | 8.25 Tok/s | 2.39 Tok/s | -71.09% |

Code 的 2,048-slot 設定減少 21.73% expert bytes。
但是，routing synchronization boundary 從 9.829 秒增加到 111.488 秒。
MLX peak memory 增加 11.156 GiB。

2,048 slots 已拒絕。
目前保留 1,152 slots。
完整結果位於
[`docs/benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

R2b 已使用 code 4K prompt 比較 4 與 2 個 read worker。

| Read workers | Decode 中位數 | Expert read 中位數 |
|---:|---:|---:|
| 4 | 8.29 Tok/s | 28.042 秒 |
| 2 | 7.94 Tok/s | 29.435 秒 |

2 workers 讓 Decode 降低 4.20%。
兩個配對結果都是負值。
Expert read timer 增加 4.97%。

2 workers 已拒絕。
目前保留 4 個 read worker。
完整結果位於
[`docs/benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

R2b 也已比較 4 與 8 個 read worker。

| Read workers | Decode 中位數 | Expert read 中位數 |
|---:|---:|---:|
| 4 | 8.27 Tok/s | 28.202 秒 |
| 8 | 8.29 Tok/s | 27.875 秒 |

8 workers 只讓 Decode 增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
這個結果低於 3% 採用門檻。

8 workers 已拒絕。
目前保留 4 個 read worker。
完整結果位於
[`docs/benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

R2c 已比較 2 與 4 個 prefetch worker。

| Prefetch workers | TTFT 中位數 | Expert read 中位數 |
|---:|---:|---:|
| 2 | 21.956 秒 | 14.478 秒 |
| 4 | 22.061 秒 | 12.009 秒 |

4 workers 讓 expert read timer 降低 17.05%。
但是，TTFT 增加 0.48%。
這個結果低於 3% 採用門檻。

4 prefetch workers 已拒絕。
目前保留 2 個 prefetch worker。
完整結果位於
[`docs/benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

R2c 也已比較 2 與 1 個 prefetch worker。

| Prefetch workers | TTFT 中位數 | Expert read | Prefetch hits 中位數 |
|---:|---:|---:|---:|
| 2 | 22.037 秒 | 14.531 秒 | 39.5 |
| 1 | 23.740 秒 | 19.394 秒 | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
Expert read timer 增加 33.46%。

1 prefetch worker 已拒絕。
目前保留 2 個 prefetch worker。
完整結果位於
[`docs/benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

R2 設定 sweep 已完成。
目前保留 `slots=1152`、`read_workers=4` 和
`prefetch_read_workers=2`。

所有候選設定都沒有達到 3% 採用門檻。

### 方向三：DSpark 重新評估（已完成）

R3 使用 code 4K prompt 和 64 個 output token。
R3 的執行順序是 normal、DSpark、DSpark、normal。

| Mode | Request 中位數 | TTFT 中位數 | 總 throughput | 回報的 Decode | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Normal | 31.152 秒 | 22.072 秒 | 2.054 Tok/s | 6.94 Tok/s | 23.274 GiB |
| DSpark | 41.301 秒 | 33.722 秒 | 1.549 Tok/s | 8.32 Tok/s | 27.257 GiB |

DSpark 讓 request 增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都接受全部 5 個 draft token。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
Runtime 的成本 gate 因此判定 speculative work 比 normal step 昂貴。

回報的 Decode boundary 增加 19.82%。
這個 boundary 不包含較慢的 DSpark prefill。
DSpark 也沒有在整個 Decode phase 持續運作。
因此，這個數字不是端到端 DSpark 收益。

目前 DSpark prefill 以區塊呼叫 `forward_with_hidden`。
DSpark prefill 不使用 main model 的 `layer_major_prefill`。
R3 的 DSpark run 也記錄 0 個 batched expert layer 和 0 次 `gather_qmm`。
Normal run 記錄 42 個 batched expert layer 和 84 次 `gather_qmm`。

修改 DSpark prefill 可能降低 TTFT。
但是，這個修改不能解決第一個 speculative round 的 fallback。
目前證據不足以投入 DSpark 專用 prefill prototype。

R3 已停止。
研究沒有擴大到五個 prompt 和 256 個 output token。
DSpark 預設值維持停用。

完整結果位於
[`docs/benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

### 方向四：MLX 圖走訪成本（停止）

**目的**：確認 MLX graph dispatch 或圖走訪是否佔 decode step 的顯著比例。

**理由**：研究計畫提到兩個 MLX 實驗：
1. 依 routed expert 的 row 數選擇 `gather_qmm` tile
2. 分開測試 MLX 圖走訪成本與 `gather_qmm` identity index cache

R1 CPU profile 沒有顯示圖走訪穩定超過 decode step 的 5%。
D1 profile gate 沒有通過。
目前不實作圖走訪修改。

如果新的 CPU profile 顯示圖走訪超過 5%，再啟動獨立修改。

**決策規則**：
- 必須先確認 CPU profile
- 任何修改必須維持 greedy token 序列相同
- 改善必須超過 3% 採用門檻

### 方向五：檔案快取策略（正式評估完成；已拒絕）

目前 checkout 已移除 `--prefill-no-file-cache` prototype。
歷史 prototype 為 full-layer Prefill 使用另一組 `F_NOCACHE` file descriptor。
Decode 使用一般 file descriptor。
單元測試已通過。
Code 4K one-token full-model smoke test 已通過。
Code 4K、256-output-token 探索性 ABBA 已完成。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Control | 52.508 秒 | 21.884 秒 | 187.17 Tok/s | 8.33 Tok/s | 23.296 GiB |
| M1 | 50.239 秒 | 20.221 秒 | 202.56 Tok/s | 8.50 Tok/s | 23.297 GiB |

M1 讓 request 改善 4.32%。
M1 讓 TTFT 改善 7.60%。
兩個 request 配對都改善超過 4%。
四個 output token hash 都相同。

Process peak footprint 增加 1.77 MiB。
M1 沒有達到至少降低 5% 的記憶體接受條件。
測試沒有出現可重現的 memory pressure。
因此，M1 的記憶體假設尚未成立。

Tool-like 4K 獨立重跑結果如下。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Control | 62.574 秒 | 22.860 秒 | 179.33 Tok/s | 6.42 Tok/s | 23.296 GiB |
| M1 | 60.380 秒 | 20.966 秒 | 195.38 Tok/s | 6.47 Tok/s | 23.297 GiB |

M1 讓 Tool-like request 改善 3.51%。
M1 讓 Tool-like TTFT 改善 8.29%。
兩個 request 配對分別改善 2.34% 和 4.63%。
四個 output token hash 都相同。

Code 與 Tool-like 的 request 中位數都改善超過 3%。
兩種 workload 的四個 request 配對都同向改善。
這個結果確認 pilot 等級的效能訊號。

Tool-like system swap 在序列中增加 2.13 MiB。
這個 system-wide 變化不能歸因於 M1。
後續正式效能測試因此繼續監控 swap。

**目的**：避免 Prefill 的 full-layer read 長時間占用系統 page cache。

**理由**：研究計畫提到 Prefill 用 `F_NOCACHE`、Decode 保留一般快取。
Prefill 的 full-layer batched read 會佔用系統 page cache，影響後續 decode 的 SSD 讀取。

**已完成工作**：

1. Prefill 階段對獨立 expert file descriptor 設定 `F_NOCACHE`
2. Decode 階段保留一般檔案快取
3. 比較 TTFT、Prefill、Decode 和 expert read timer
4. 比較 process footprint 與 system page-cache 計數

**決策規則**：

- 必須使用多 prompt ABBA
- 改善必須超過 3% 採用門檻
- 必須記錄 OS page cache 狀態
- `phys_footprint` 必須降低至少 5%，或消除可重現的 memory pressure
- Prefill 與 Decode 都不可降低超過 2%

**目前決定**：

- 專案不把 M1 採用為記憶體最佳化
- 專案拒絕預設啟用 M1
- runtime 已移除 M1 prototype

正式效能評估 Wave 1 已完成。
正式效能評估 Wave 2 也已完成。
正式效能評估 Wave 3 也已完成。
正式效能評估 Wave 4 也已完成。
正式效能評估 Wave 5 也已完成。
五輪四個主要 workload 的 request 改善中位數是 3.53%。
TTFT 改善中位數是 7.99%。
Prefill throughput 增加中位數是 8.69%。
Decode throughput 增加中位數是 0.23%。
全部 50 個 request 配對都改善。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
主要 workload 的 Decode p95 regression 最大值是 1.71%。
Repeated Decode p95 regression 是 3.01%。
System-wide compressor 的不增加條件沒有得到證明。

Wave 1 完整結果位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

Wave 3 完整結果位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-wave3-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-wave3-m5-pro.json)。

Wave 4 與前四輪階段性完整結果位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-wave4-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-wave4-m5-pro.json)
和
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-progress-wave4-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-progress-wave4-m5-pro.json)。

Wave 5 與正式統計位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)
和
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

完整結果位於
[`docs/benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)
和
[`docs/benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

## 不建議的方向

以下方向已測試或推論不適合，不建議優先投入：

| 方向 | 原因 |
|---|---|
| 單一 contiguous slot arena | 比目前慢 10.1% |
| Custom greedy generator | 比 mlx-lm generator 慢 7% |
| Prefill-to-decode handoff | 單 prompt 證據不足 |
| 平均分配 prefill 區塊 | TTFT 增加 19.8%，被拒絕 |
| 略過 CSA indexer | 收益不穩定，低於 3% 門檻 |
| DSpark 直接預設啟用 | R3 request 增加 32.58%，總 throughput 降低 24.60%，且兩個 run 都 fallback。 |
| P3 `gather_qmm` tile | shader timing 占比低於 15% profile gate |
| P4 attention 類型區塊 | 候選改變 expert route，無法隔離 attention step 的效能。 |
| D2 專用 routed expert top-6 | dispatch 與 GPU time 都低於 3% profile gate。 |
| D3 CSA row 重用 | Row 重疊率與 `gather` 占比都低於各自的 profile gate。 |
| Custom Metal kernel | 目標 section 必須先通過該研究項目的 profile gate |
| MTLIO bridge | 尚未驗證 MLX buffer ownership |

## 執行順序

```
R0: 建立可重現 baseline（4K/8K/14K）〔完成〕
  ↓
R1: Metal boundary capture〔完成；缺少 shader timing〕
  ↓
R2a: Slot A/B〔完成；2,048 slots 已拒絕〕
  ↓
R2b-1: 2/4 Worker A/B〔完成；2 workers 已拒絕〕
  ↓
R2b-2: 4/8 Worker A/B〔完成；8 workers 已拒絕〕
  ↓
R2c-1: 2/4 Prefetch Worker A/B〔完成；4 workers 已拒絕〕
  ↓
R2c-2: 1/2 Prefetch Worker A/B〔完成；1 worker 已拒絕〕
  ↓
R3: DSpark 探索性 A/B〔完成；目前候選已拒絕〕
  ↓
R4/M1: 分階段 macOS 檔案快取〔完成；候選已拒絕並移除〕
  ↓
P3: `gather_qmm` tile profiling〔完成；profile gate 未通過〕
  ↓
P4: attention 類型區塊〔完成；正確性 gate 未通過〕
  ↓
D2: routed expert top-6 profiling〔完成；profile gate 未通過〕
  ↓
D3: CSA row trace〔完成；兩個 gate 都未通過〕
```

R0 和 R1 已完成。
R2b 與 R2c 使用 R0 的固定 prompt。
R3 使用 R0 的固定 code prompt。
M1 不需要 shader-level 證據。
P3 shader-level 證據已完成。
P3 沒有通過 15% profile gate。
P4 沒有通過正確性 gate。
D2 沒有通過 3% profile gate。
D3 沒有通過 90% row gate 或 5% profile gate。
其他 custom Metal kernel 仍需要各自的 shader-level 證據。

## 實驗規則

所有實驗必須遵守：

1. 使用至少五個代表性 prompt
2. 每個 prompt 產生至少 256 個 token（TTFT 測試除外）
3. A/B 執行順序要交錯
4. 分開 fresh runtime、warm graph 和 OS page-cache 狀態
5. 保存每次 run，不只保存平均值
6. 報告 median 和 p95
7. 比較完整 token hash
8. 記錄 peak memory 和 process RSS
9. 記錄 expert bytes、read timer、hit、miss 和 eviction
10. 任何改善不得讓短 prompt 或低記憶體設定發生未說明的 regression

## 與現有研究計畫的關係

本文件補充
[`RUNTIME_PREFILL_DECODE_RESEARCH_PLAN_2026-08-10.md`](RUNTIME_PREFILL_DECODE_RESEARCH_PLAN_2026-08-10.md)。

本文件提供推論分析，現有計畫提供實驗流程。
本文件不取代現有計畫。
本文件不取代 R0、R1、R2、R3、R4 的實驗。

現有計畫的 P3（gather_qmm tile 選擇）、P4（attention 類型選擇區塊）、
D2（top-6 選擇）和 D3（CSA row 重用）已完成。
M1 已完成，且目前候選已拒絕並移除。
D1 與 P5 已停止，等待新的 profile 證據。

本文件的優化方向與現有計畫對齊。
目前計畫沒有其他 active 實驗。

## 一手來源

- [DeepSeek-V4-Flash-0731 config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX gather_qmm](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)
- [Apple Metal System Trace](https://developer.apple.com/documentation/xcode/metal-developer-workflows)
- [DSpark paper](https://arxiv.org/html/2607.05147)
