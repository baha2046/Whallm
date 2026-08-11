# 研究結論與決策

本文件整合 2026-08-07 至 2026-08-11 的研究。
本文件只保留目前仍成立的結論。
日期型原始研究已移到 [`research/archive`](../research/archive/README.md)。

已完成的詳細實驗計畫與推論分析已封存：

- [`RUNTIME_PREFILL_DECODE_RESEARCH_PLAN_2026-08-10.md`](../research/archive/RUNTIME_PREFILL_DECODE_RESEARCH_PLAN_2026-08-10.md)
- [`RUNTIME_SPEED_OPTIMIZATION_ANALYSIS_2026-08-10.md`](../research/archive/RUNTIME_SPEED_OPTIMIZATION_ANALYSIS_2026-08-10.md)

這兩份文件保存研究假設、停止條件和執行順序。
這兩份文件不描述目前 runtime。

## 總結

目前 runtime 的基礎方向正確。

- main model 使用固定 checkpoint 合約。
- common tensor 保留在統一記憶體。
- routed expert 使用 checkpoint-native FP4。
- runtime 依需求從 SSD 讀取 expert blob。
- 長 prefill 使用 layer-major 和 batched `gather_qmm`。
- decode 使用 ready expert path。
- compressed KV cache 使用 MXFP8。
- prompt cache 保存完整 prefix state。
- DSpark 可安裝和執行，但 DSpark 預設停用。

R0 已建立可重現的多 prompt profiling artifact。
R2a 已拒絕 2,048-slot 候選設定。
R2b 已拒絕 2-worker 候選設定。
R2b 已拒絕 8-worker 候選設定。
R2c 已拒絕 4-prefetch-worker 候選設定。
R2c 已拒絕 1-prefetch-worker 候選設定。
R2 設定 sweep 已完成。
R3 已拒絕目前 DSpark 預設啟用。
M1 的五輪正式效能評估已完成。
正式評估包含 100 個有效量測 run 和 50 個配對。
四個主要 workload 的 request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。
System-wide compressor 的不增加條件也沒有得到證明。
專案拒絕預設啟用 M1。
runtime 已移除 M1 prototype。
P3 `gather_qmm` tile profiling 已完成。
P3 量到的 `gather_qmm` GPU time 是 Prefill 的 12.77%。
依 GPU interval 涵蓋率校正後是 13.99%。
兩個結果都低於 15% profile gate。
P3 已停止。
專案沒有建立 isolated MLX build。
P4 attention 類型區塊探索已完成。
512-step 候選讓 39/42 個 MoE layer 的 expert route 改變。
至少 32,075 個 expert assignment 改變。
專案拒絕 P4，並移除 prototype。
D2 routed expert top-6 profile 已完成。
Python dispatch 占 Decode wall time 的 0.121%。
shader sample 占 0.456%。
依 Shader Timeline 涵蓋率校正後是 0.972%。
三個結果都低於 3% profile gate。
D2 已停止，且沒有建立 prototype。
D3 CSA row trace 已完成。
五種 context 合併後的 row 交集比例中位數是 85.94%。
4K、8K 和 14K 的中位數分別是 78.71%、70.51% 和 62.30%。
`MXFP8PoolingCache.gather` 的涵蓋率校正核心占比是 Decode 的 0.25%。
包含通用 clip 和 select shader 後是 0.34%。
兩個 D3 gate 都沒有通過。
D3 已停止，且沒有建立 prototype。
目前研究計畫沒有其他已通過 profile gate 的候選項目。

## 證據分類

| 類別 | 可以支持的結論 |
| --- | --- |
| 目前程式碼 | 功能、預設值、資料路徑和限制。 |
| 自動測試 | fixture 或 mock path 的正確性。 |
| 本機配對量測 | 指定硬體、prompt 和設定的速度差異。 |
| 外部一手來源 | 外部 API、格式、模型合約或論文結果。 |
| 研究假設 | 值得測試的方向。不能寫成預期收益。 |

外部模型的改善百分比不能套用到本專案。
估計值不能放入正式結果表。

## 目前候選項目

P1、P2、M1 與 P4 prototype 已移除。

| ID | 開關 | 目前證據 | 決定 |
| --- | --- | --- | --- |
| P1 | 已移除 | TTFT 增加 19.8%；40 層的 Prefill route histogram 不同 | 拒絕。保留目前固定 step。 |
| P2 | 已移除 | 兩輪 repeated prompt ABBA 的配對中位數改善 1.41%；安靜環境的第二輪慢 0.96% | 拒絕。保留完整 CSA indexer。 |
| R2a | `--slots` | Repeated Decode 降低 3.31%；code Decode 降低 71.09% | 拒絕 2,048 slots。保留 1,152 slots。 |
| R2b | `--read-workers` | Code Decode 降低 4.20%；兩個配對結果都是負值 | 拒絕 2 workers。保留 4 workers。 |
| R2b | `--read-workers` | Code Decode 增加 0.25%；低於 3% 門檻 | 拒絕 8 workers。保留 4 workers。 |
| R2c | `--prefetch-read-workers` | Expert read 降低 17.05%，但 TTFT 增加 0.48% | 拒絕 4 workers。保留 2 workers。 |
| R2c | `--prefetch-read-workers` | TTFT 增加 7.73%；prefetch hit 中位數降到 0.5 | 拒絕 1 worker。保留 2 workers。 |
| R3 | `--dspark` | Request 增加 32.58%；總 throughput 降低 24.60%；兩個 run 都 fallback | 拒絕目前候選。DSpark 預設停用。 |
| M1 | 已移除 | 正式 request 改善中位數是 3.53%，但 Repeated Decode p95 regression 是 3.01%；compressor 不增加條件沒有得到證明 | 拒絕預設啟用。移除 prototype。 |
| P3 | 無 | `gather_qmm` GPU time 是 Prefill 的 12.77%；涵蓋率校正後是 13.99% | 未通過 15% profile gate。停止，不建立 MLX fork。 |
| P4 | 已移除 | 512-step 候選改變 39/42 個 MoE layer 的 expert route；至少 3.11% 的 assignment 不同 | 未通過正確性 gate。拒絕，不執行正式 ABBA。 |
| D2 | 無 | dispatch 是 Decode 的 0.121%；shader sample 是 0.456%；涵蓋率校正後是 0.972% | 未通過 3% profile gate。停止，不建立 prototype。 |
| D3 | 無 | 五種 context 的 row 交集比例中位數是 85.94%；`gather` 涵蓋率校正占比最多是 Decode 的 0.34% | 未通過 90% row gate 與 5% profile gate。停止，不建立 prototype。 |

此表同時列出探索性結果和 M1 正式結果。

## R0 與 R1 結果

R0 已完成五種 prompt 與三種長度的 baseline。
每個 run 都產生 256 個 output token。
每個 prompt 和長度目前只有一個 run。

| Prompt token | TTFT 中位數 | Prefill 中位數 | Decode 中位數 | Misses 中位數 | Evictions 中位數 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 21.611 s | 189.53 Tok/s | 8.00 Tok/s | 24,700 | 23,548 |
| 8,192 | 39.124 s | 209.39 Tok/s | 7.76 Tok/s | 24,956 | 23,804 |
| 14,363 | 67.466 s | 212.89 Tok/s | 7.43 Tok/s | 25,345 | 24,193 |

Repeated 4K 的 Decode 是 12.22 Tok/s。
四種非 repeated 4K prompt 的 Decode 是 6.44 至 8.28 Tok/s。
Repeated prompt 不能代表一般 Decode 的 SSD 與 slot 壓力。

R1 已完成 GPU busy 與 GPU idle boundary capture。
Tool-like Decode 的 GPU busy 是 35.5% 至 37.1%。
Tool-like Decode 的 GPU idle 是 7.968 至 8.114 秒。
這個結果支持先測 slot 和 worker 設定。

Metal System Trace 沒有提供 shader timing sample。
R1 不能把 GPU time 分配到 attention、routed expert 和 shared expert。
CPU profile 也沒有讓 D1 或 P5 通過 5% profile gate。

P3 另外使用 `Metal GPU Counters` capture Tool-like 4K Prefill。
目前 kernel 是 `mxfp4_gather_qmm_rhs_nax`。
kernel 使用 BM64、BN64 和 BK64。
trace 包含 42 個 expert layer burst。
runtime metrics 記錄 84 次 `gather_qmm` call。

`gather_qmm` 的 shader sample duration 是 2.966 秒。
這個時間是 request 的 12.77%。
shader sample work 涵蓋 Python GPU busy 的 91.28%。
依涵蓋率校正後，`gather_qmm` 是 request 的 13.99%。
兩個結果都低於 P3 的 15% profile gate。

同一個固定 prompt 的 R0 route trace 顯示，
24.53% 的 expert run 是空的。
27.37% 的 expert run 只有 1 至 15 rows。
BM64 的理論 padding 是配置 rows 的 24.43%。
這個分布支持 tile 浪費假設。
但是，P3 的端到端占比沒有通過 profile gate。
因此，專案停止 P3，且不建立 isolated MLX build。

完整資料位於
[`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)。
P3 shader profile 位於
[`benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json`](benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json)。

## P4 attention 類型區塊結果

P4 使用 Tool-like 4K prompt 和一個 output token。
P4 分別測試 local、CSA 和 dense compressed attention 的
512、2,048 和 4,096 step。
目前全域 Control step 是 1,024。

512 step 在單次 profile 中降低 CSA 和 dense compressed attention phase。
但是，Control request 在研究期間漂移 7.91%。
單次 request 結果不能作為正式速度結論。

組合的 512-step 候選產生相同 output token hash。
但是，該候選改變 39/42 個 MoE layer 的 expert route。
expert histogram 的 L1 distance 是 64,150。
至少 32,075 個 assignment 改變。
這是 1,031,940 個 assignment 的 3.11%。

CSA、dense compressed 和 local 的單項 512-step 候選也分別改變
8、-1 和 7 個 expert blob 的讀取量。
因此，P4 phase time 混入不同的 expert I/O。
P4 沒有通過正確性 gate。
專案沒有執行正式 ABBA。
runtime 已移除 P4 prototype，並保留單一全域 step。

完整資料位於
[`benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json`](benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json)。

## D2 routed expert top-6 profile

D2 使用 Tool-like 4K prompt 和 64 個 output token。
runtime metrics 記錄 63 個 Decode token。
每個 Decode forward 有 40 個非 hash-routed layer。

| 指標 | 結果 |
| --- | ---: |
| `_expert_select` Python dispatch | 13.252 ms |
| Dispatch / Decode wall | 0.121% |
| Decode GPU busy | 4.559 s |
| Shader Timeline 涵蓋率 | 46.88% |
| `_expert_select` shader sample | 58.834 ms |
| Shader sample / Decode wall | 0.456% |
| 涵蓋率校正後 / Decode wall | 0.972% |

D2 的停止門檻是 Decode wall time 的 3%。
dispatch、原始 shader sample 和涵蓋率校正結果都低於門檻。
專案停止 D2，且沒有建立專用 top-6 prototype。
runtime 沒有變更。

完整資料位於
[`benchmarks/2026-08-11-d2-top6-profile-m5-pro.json`](benchmarks/2026-08-11-d2-top6-profile-m5-pro.json)。

## D3 CSA row trace

D3 使用 Tool-like prompt。
D3 分別記錄 512、2,048、4,096、8,192 和 14,363 token context。
每組產生 64 個 output token。
每組比較 21 個 CSA layer 的 63 個相鄰 Decode transition。

交集比例使用下列定義。

```text
前一個 token 與目前 token 的共同 selected row / 目前 token 的 selected row
```

| Context | 交集比例中位數 | Jaccard 中位數 |
| ---: | ---: | ---: |
| 512 | 100.00% | 100.00% |
| 2,048 | 99.02% | 98.07% |
| 4,096 | 78.71% | 64.90% |
| 8,192 | 70.51% | 54.45% |
| 14,363 | 62.30% | 45.25% |

五種 context 的合併交集比例中位數是 85.94%。
D3 的 row gate 是 90%。
512 與 2,048 context 會選到幾乎全部可用 pooled row。
4K 以上的 sparse CSA path 都沒有通過 row gate。

D3 使用 production-shape microbenchmark 對應 `gather` shader label。
D3 再把 label 套用到既有的 D2 Tool-like 4K Metal trace。

| `gather` 指標 | Decode wall time 占比 |
| --- | ---: |
| 核心 shader sample | 0.117% |
| 核心 shader，依涵蓋率校正 | 0.250% |
| 包含通用 clip/select shader sample | 0.158% |
| 包含通用 clip/select shader，依涵蓋率校正 | 0.338% |

D3 的 `gather` gate 是 5%。
核心結果與包含通用 shader 的結果都低於門檻。
D3 沒有通過兩個 gate。
專案停止 D3，且沒有建立 row 重用 prototype。
runtime 沒有變更。

完整資料位於
[`benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json`](benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json)。

## R2a slot 結果

R2a 使用 4K prompt 和 256 個 output token。
每個 prompt 的順序是 1,152、2,048、2,048、1,152 slots。

| Prompt | 1,152 slots | 2,048 slots | Decode 差異 |
| --- | ---: | ---: | ---: |
| Repeated | 14.42 Tok/s | 13.95 Tok/s | -3.31% |
| Code | 8.25 Tok/s | 2.39 Tok/s | -71.09% |

Code 的 2,048-slot 設定減少 expert bytes、miss 和 eviction。
但是，Code 的 routing synchronization boundary 增加到 111.488 秒。
Code 的 MLX peak memory 增加 11.156 GiB。
Swap 沒有增加。

測試在 code ABBA 完成後觸發停止條件。
後面三種 prompt 沒有繼續執行。
這不是正式五個 workload、五輪 ABBA 結果。
這個結果足以拒絕 2,048-slot 候選設定。

完整資料位於
[`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

## R2b 2-worker 結果

R2b 使用 code 4K prompt 和 256 個 output token。
執行順序是 4、2、2、4 read workers。

| Read workers | Decode 中位數 | Expert read 中位數 |
| ---: | ---: | ---: |
| 4 | 8.29 Tok/s | 28.042 s |
| 2 | 7.94 Tok/s | 29.435 s |

2 workers 讓 Decode 降低 4.20%。
兩個配對差異是 -4.12% 和 -4.27%。
Expert read timer 增加 4.97%。
Expert bytes、miss、eviction 和 token hash 都相同。

這不是正式多 workload 結果。
這個結果足以拒絕 2-worker 候選設定。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

### 8-worker 結果

R2b 使用相同的 code 4K prompt 比較 4 與 8 個 read worker。

| Read workers | Decode 中位數 | Expert read 中位數 |
| ---: | ---: | ---: |
| 4 | 8.27 Tok/s | 28.202 s |
| 8 | 8.29 Tok/s | 27.875 s |

8 workers 讓 Decode 增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
這個結果低於 3% 採用門檻。

8-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

## R2c 4-prefetch-worker 結果

R2c 使用 code 4K prompt 和一個 output token。
執行順序是 2、4、4、2 prefetch workers。

| Prefetch workers | TTFT 中位數 | Expert read 中位數 |
| ---: | ---: | ---: |
| 2 | 21.956 s | 14.478 s |
| 4 | 22.061 s | 12.009 s |

4 workers 讓 expert read timer 降低 17.05%。
但是，TTFT 增加 0.48%。
兩個配對改善的方向不一致。

4-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

### 1-prefetch-worker 結果

R2c 使用相同的 code 4K prompt 比較 2 與 1 個 prefetch worker。

| Prefetch workers | TTFT 中位數 | Expert read 中位數 | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: |
| 2 | 22.037 s | 14.531 s | 39.5 |
| 1 | 23.740 s | 19.394 s | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
兩個配對結果都是負值。
Expert read timer 增加 33.46%。

1-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。
R2 設定 sweep 已完成。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

## R3 DSpark 結果

R3 使用 code 4K prompt 和 64 個 output token。
執行順序是 normal、DSpark、DSpark、normal。

| Mode | Request 中位數 | TTFT 中位數 | 總 throughput | 回報的 Decode | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal | 31.152 s | 22.072 s | 2.054 Tok/s | 6.94 Tok/s | 23.274 GiB |
| DSpark | 41.301 s | 33.722 s | 1.549 Tok/s | 8.32 Tok/s | 27.257 GiB |

DSpark 讓 request 增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都接受全部 5 個 draft token。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
回報的 Decode boundary 不包含較慢的 DSpark prefill。
因此，回報的 Decode 改善不是端到端收益。

四個 output token hash 都相同。
R3 已觸發停止條件。
研究沒有擴大到五個 prompt 和 256 個 output token。
目前 DSpark 預設值維持停用。

完整資料位於
[`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

## M1 Prefill file-cache 結果

M1 使用 code 4K prompt 和 256 個 output token。
執行順序是 Control、M1、M1、Control。
M1 只對 full-layer Prefill file descriptor 設定 `F_NOCACHE`。
Decode 保留一般 file descriptor。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 52.508 s | 21.884 s | 187.17 Tok/s | 8.33 Tok/s | 23.296 GiB |
| M1 | 50.239 s | 20.221 s | 202.56 Tok/s | 8.50 Tok/s | 23.297 GiB |

M1 讓 request 改善 4.32%。
M1 讓 TTFT 改善 7.60%。
兩個 request 配對都改善超過 4%。
四個 output token hash 都相同。

Process peak footprint 增加 1.77 MiB。
這個差異可視為沒有變化。
M1 沒有達到至少降低 5% 的記憶體接受條件。
測試也沒有出現可重現的 memory pressure。
因此，M1 的記憶體假設尚未成立。
專案不把 M1 採用為記憶體最佳化。

本次測試只有一種 workload 和一輪 ABBA。
本次測試不能支持正式效能採用。
兩個 request 配對仍顯示一致的效能訊號。
專案當時暫時保留 opt-in prototype。
後續 Tool-like 4K 獨立效能重跑確認相同方向。
這個 pilot 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)。

### Tool-like 4K 獨立重跑

Tool-like 重跑使用相同 ABBA 順序和 256 個 output token。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 62.574 s | 22.860 s | 179.33 Tok/s | 6.42 Tok/s | 23.296 GiB |
| M1 | 60.380 s | 20.966 s | 195.38 Tok/s | 6.47 Tok/s | 23.297 GiB |

M1 讓 request 改善 3.51%。
M1 讓 TTFT 改善 8.29%。
兩個 request 配對分別改善 2.34% 和 4.63%。
四個 output token hash 都相同。
Process peak footprint 增加 0.98 MiB。

Code 與 Tool-like 的 request 中位數都改善超過 3%。
兩種 workload 的四個 request 配對都同向改善。
這個結果確認 pilot 等級的效能訊號。

System swap 在序列中增加 2.13 MiB。
這個 system-wide 變化不能歸因於 M1。
後續正式測試因此繼續監控 swap。

兩種 workload 都只有一輪 ABBA。
這兩個 pilot 當時沒有 bootstrap 信賴區間。
專案因此進入正式多 workload 重複 ABBA。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

### 正式效能評估 Wave 1

Wave 1 使用五個 R0 4K prompt。
每個 prompt 執行一個不納入統計的 warm-up 和一輪 ABBA。
Wave 1 共有 20 個有效量測 run。

四個主要 workload 的 request 改善中位數是 3.54%。
TTFT 改善中位數是 8.18%。
Prefill throughput 增加中位數是 8.91%。
Decode throughput 增加中位數是 0.35%。
八個主要 workload request 配對都改善。

Repeated Decode 降低 2.93%。
後續 wave 因此繼續確認這個結果。
Wave 1 當時沒有 p95 或 95% bootstrap 信賴區間。
因此，Wave 1 當時不能支持正式採用決定。

正式序列的 system swap 降低 8.00 MiB。
Process peak footprint 沒有實質變化。
所有 output token hash 都一致。
測試沒有觸發停止條件。

Wave 1 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

### 正式效能評估 Wave 2

Wave 2 使用輪替後的五個 R0 4K prompt 順序。
Wave 2 共有 20 個有效量測 run。

四個主要 workload 的 request 改善中位數是 3.62%。
TTFT 改善中位數是 8.05%。
Prefill throughput 增加中位數是 8.75%。
Decode throughput 增加中位數是 0.32%。
八個主要 workload request 配對都改善。

Wave 1 的 Repeated Decode regression 沒有在 Wave 2 重現。
Repeated Decode 在 Wave 2 增加 3.64%。
Process peak footprint 沒有實質變化。
System swap 降低 8.00 MiB。
System-wide compressor 增加 209.67 MiB。
這個 system-wide 變化不能歸因於 M1。

第一次 Code warm-up 後，runner 的 Swap parser 發生錯誤。
錯誤發生在任何正式量測 run 開始前。
專案排除第一次 warm-up，並從新的 Code warm-up 完整重跑。

### 正式效能評估 Wave 3

Wave 3 使用再次輪替的五個 R0 4K prompt 順序。
Wave 3 共有 20 個有效量測 run。

四個主要 workload 的 request 改善中位數是 3.71%。
TTFT 改善中位數是 8.09%。
Prefill throughput 增加中位數是 8.76%。
Decode throughput 增加中位數是 1.03%。
八個主要 workload request 配對都改善。

所有 output token hash 都與 Wave 1 相同。
Process peak footprint 沒有實質變化。
System swap 沒有變化。
System-wide compressor 增加 244.77 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有觸發停止條件。

### 正式效能評估 Wave 4

Wave 4 使用再次輪替的五個 R0 4K prompt 順序。
Wave 4 共有 20 個有效量測 run。

四個主要 workload 的 request 改善中位數是 3.32%。
TTFT 改善中位數是 7.82%。
Prefill throughput 增加中位數是 8.49%。
Decode throughput 增加中位數是 0.56%。
八個主要 workload request 配對都改善。

所有 output token hash 都與 Wave 1 相同。
Process peak footprint 沒有實質變化。
System swap 沒有變化。
System-wide compressor 增加 340.41 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有觸發停止條件。

### 正式效能評估 Wave 5 與最終決定

Wave 5 再次輪替五個 R0 4K prompt。
Wave 5 共有 20 個有效量測 run。
第一次序列在 Tool-like B1 啟動時失去執行 session。
專案排除整個不完整序列，並完整重跑 Wave 5。

Wave 5 的主要 workload request 改善中位數是 3.15%。
TTFT 改善中位數是 7.58%。
Prefill throughput 增加中位數是 8.24%。
Decode throughput 增加中位數是 0.12%。
全部十個 request 配對都改善。

五輪共有 100 個有效量測 run 和 50 個配對。
全部 50 個 request 配對都改善。
主要 workload request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
主要 workload 的 Decode p95 regression 最大值是 1.71%。
這些效能條件通過。

Repeated Decode p95 regression 是 3.01%。
這個結果沒有通過 2% 上限。
Process peak footprint 的配對中位數只增加 0.32 MiB。
每一輪的 system swap 都沒有增加。
部分 wave 的 system-wide compressor 總量增加。
這個增加不能歸因於 M1，但證據也沒有證明 compressor 不增加。

M1 沒有通過全部正式採用條件。
專案拒絕預設啟用 M1。
runtime 已移除 M1 prototype。

Wave 5 完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)。

正式統計與決定位於
[`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

## 已採用

| 決策 | 目前實作 | 證據 |
| --- | --- | --- |
| Canonical expert blob | repacker 固定 `w1/w2/w3` 和 scales 的 byte layout。 | 合約測試、完整 SHA-256。 |
| Resumable repack | 8 MiB chunk、receipt、digest 和 repair。 | Swift tests 和完整 installed model。 |
| Layer-major prefill | 4,096 個未快取 token 起啟用。 | 14K 歷史結果把 expert bytes 降低約 94.3%。 |
| Batched layer-local MoE | full-layer buffer、strided view 和 `gather_qmm`。 | 4K/14K 歷史配對、fixture parity。 |
| 自動 prefill step | 128、256、1,024。 | 目前程式碼和 tests。 |
| MXFP8 compressed cache | 完成的 64-row chunk 使用 MXFP8。 | cache tests 和 8K 歷史 greedy token。 |
| MXFP4 index cache | index scoring 預設使用 MXFP4 view。 | shape、gather 和 parity tests。 |
| Persistent prompt cache v2 | 直接保存 quantized cache arrays。 | restart 和 round-trip tests。 |
| Direct slot read | `preadv` 直接寫入 MLX slot view。 | 目前程式碼和 full-model run。 |
| Ready expert decode | resident 和先讀完的 expert 先提交 compute。 | 五組配對 hash；改善中位數 12.9%。 |
| 分開 decode 指標 | model step、cache eval、end-to-end、p50 和 p95。 | 目前 status 和 metrics artifact。 |
| DSpark round transaction | 每 round 一次 cache fork；拒絕時只 replay committed prefix。 | DSpark tests 和目前程式碼。 |

## 已測試但未採用

### 單一 contiguous slot arena

一組 4K prompt 和 256-token decode 使用相同 token hash。
單一 15 GB buffer 是 5.80 Tok/s。
每個 slot 一個 direct buffer 是 6.39 Tok/s。

單一 buffer 在該測試慢 10.1%。
runtime 保留每個 slot 一個 direct buffer。

這個結論只適用於目前 ready expert 設計。
未來若 MLX buffer ownership 改變，必須重新量測。

### Custom greedy generator

custom greedy generator 是 6.05 Tok/s。
mlx-lm generator 是 6.51 Tok/s。
兩個 path 產生相同 token。

runtime 保留 mlx-lm generator。

### Prefill-to-decode handoff

一組 4K repeated prompt 和 64-token decode 產生下列離線結果。

| Hot set | Route coverage | Baseline miss bytes covered |
| ---: | ---: | ---: |
| top-6 per layer | 69.1% | 9.0% |
| top-8 per layer | 78.1% | 14.8% |
| top-10 per layer | 83.0% | 21.4% |

route coverage 很高，但目前 LFU 已保留多數 hot expert。
單一 prompt 的 miss coverage 不足以支持實作 handoff。

handoff 保留為延後研究。
下一次研究必須使用多個 domain 和 256-token output。

### DSpark 預設啟用

舊 runtime 的三組測試中，normal decode 中位數是 5.39 Tok/s。
DSpark 中位數是 5.14 Tok/s。

目前 checkout 的 R3 探索性 ABBA 也沒有淨收益。
R3 的 request 增加 32.58%。
R3 的總 throughput 降低 24.60%。
兩個 DSpark run 都在第一個 speculative round 後 fallback。

因此，runtime 不預設啟用 DSpark。

## DSpark 目前狀態

舊研究指出兩個主要 verification 問題。

1. 每個 position 複製 43 層 cache。
2. hot loop 讀取 persistence `state`，並重建完整 BF16 pooled cache。

目前程式碼已取代這兩個 path。

- verifier 一次處理完整 token block。
- verifier 在每個 round 建立一次 cache fork。
- verifier 使用 raw cache arrays 做 evaluation。
- verifier 完整接受時直接 commit fork。
- verifier 拒絕時只 replay anchor 和 accepted prefix。
- metrics 記錄 fork、replay、cache bytes 和 per-layer time。

舊 `DSPARK_80_PERCENT_OPTIMIZATION_RESEARCH_2026-08-09.md` 的
per-position bottleneck 不再描述目前程式碼。

目前仍有下列限制。

- generation 使用 batch size 1。
- server 序列化 generation。
- DSpark 不使用一般 prompt cache reuse。
- DSpark prefill 不使用 main model 的 layer-major path。
- DSpark 使用獨立 768-slot expert cache。
- DSpark attention 尚未以目前硬體完成 GPU section profiling。
- 目前沒有五個 prompt 的 current-checkout paired speed matrix。

R3 在單一 code 4K workload 已觸發停止條件。
目前不需要為預設啟用執行五個 prompt speed matrix。

DeepSeek 論文的 60–85% 是 production serving 結果。
該結果使用 matched aggregate throughput 和多 request scheduler。
該結果不是單一 Apple Silicon request 的倍率。

## 延後研究

| 方向 | 目前狀態 | 啟動條件 |
| --- | --- | --- |
| Metal boundary capture | R1 boundary 與 P3 shader timing 已完成 | 新候選需要不同 section 或不同 workload 時才重跑。 |
| `slots` sweep | 2,048 slots 已拒絕 | 保留目前 1,152-slot 預設值。 |
| `read_workers` sweep | 2 與 8 workers 已拒絕 | 保留目前 4-worker 預設值。 |
| Prefetch worker sweep | 1 與 4 workers 已拒絕 | 保留目前 2-worker 預設值。 |
| w13/w2 staged streaming | 尚未實作 | capture 顯示 expert read wait 或單 expert critical path。 |
| Prefill-to-decode handoff | 單 prompt 離線證據不足 | 多 domain miss coverage 穩定且不增加 eviction。 |
| Same-layer predictor | 尚未訓練 | route trace 先證明 ready recall 上限。 |
| Hash-layer exact prefetch | 尚未實作 | 三個 hash layer 的可用 ID 時點明確。 |
| MTLIO bridge | 尚未實作 | native microbenchmark 先證明 shared/private buffer 優勢。 |
| P3 `gather_qmm` tile | Profile gate 未通過 | 只有新的 workload 或 upstream kernel 證據讓占比達到 15% 才重啟。 |
| P4 attention 類型區塊 | 正確性 gate 未通過；prototype 已移除 | 新設計必須保留 expert route。 |
| D2 routed expert top-6 | Profile gate 未通過 | 新的 workload 或 upstream 證據必須先讓占比達到 3%。 |
| D3 CSA row 重用 | Row gate 與 profile gate 都未通過 | 新的 workload 或 cache 設計必須先讓 row 重疊率達到 90%，且 `gather` 占比達到 5%。 |
| Custom Metal kernel | 尚未實作 | 某個穩定 section 先通過該研究項目的 profile gate。 |
| DSpark 預設啟用 | R3 已拒絕目前候選 | DSpark 實作改變後，新的探索性 ABBA 先通過 request、throughput 與 memory gate。 |
| M1 Prefill file-cache | 正式評估完成；候選已拒絕並移除 | 只有新的設計與新的 profile 證據才重新研究。 |
| Approximate MoE | 不在 exact mode | API 必須明確標示 output parity 不成立。 |

## MTLIO、zero-copy 與 custom Metal

Apple 提供 `MTLIOCommandQueue`。
MLX 提供 unified memory、compile 和 custom Metal 機制。

這些 API 只能證明平台具有機制。
這些 API 不能證明本專案可以直接包裝外部 `MTLBuffer`。
這些 API 也不能證明 external SSD 會更快。

MTLIO prototype 必須先驗證：

1. MLX array 和 `MTLBuffer` 的 ownership。
2. CPU write 和 GPU read 的 synchronization。
3. shared buffer 和 private buffer 的 copy path。
4. cancel、priority 和 queue depth。
5. 相同 expert bytes 和相同 output token hash。
6. 端到端 wall time，不只 I/O command time。

未完成上述驗證前，MTLIO 保持研究假設。

## 下一輪研究順序

### R0：保存可重現 baseline

已完成。
R0 保存 4K、8K 和 14K prompt artifact。
R0 保存可重建規則、raw metrics JSON 和完整 output token hash。

### R1：分解目前 TTFT

已完成 boundary capture。
R1 已對 4K repeated、4K Tool-like 和 14K Tool-like 執行 Metal capture。
R1 已分開 GPU busy、GPU idle 和 CPU `preadv` sample。
R1 trace 沒有 shader timing sample。
R1 trace 不能分開 attention、batched MoE 和 shared expert。

### R2：端到端 worker 與 slot A/B

R2a slot 探索性 ABBA 已完成。
2,048 slots 已拒絕。
目前保留 1,152 slots。

R2b 已完成 2/4 read-worker 探索性 ABBA。
2 workers 已拒絕。
R2b 也已完成 4/8 read-worker 探索性 ABBA。
8 workers 已拒絕。
目前保留 4 個 read worker。

R2c 已完成 2/4 prefetch-worker 探索性 ABBA。
4 workers 已拒絕。
R2c 也已完成 1/2 prefetch-worker 探索性 ABBA。
1 worker 已拒絕。

R2 已完成。
目前保留 `slots=1152`、`read_workers=4` 和
`prefetch_read_workers=2`。

每次只改一個設定。
每組比較使用相同 token hash。

### R3：重新評估 DSpark

R3 已完成 code 4K、64-output-token 探索性 ABBA。
Request 增加 32.58%。
總 throughput 降低 24.60%。
Peak footprint 增加 3.983 GiB。
兩個 DSpark run 都 fallback。

R3 已停止。
研究沒有執行五個 prompt 和 256 個 output token 的正式 ABBA。
DSpark 預設值維持停用。

### R4：只處理已確認的 section

R4 的第一個候選是 M1 分階段 macOS 檔案快取。
M1 正式效能測試已完成。
候選沒有通過 Repeated Decode regression 與 compressor 條件。
專案已拒絕並移除 M1。

P3 `gather_qmm` tile profiling 已完成。
P3 沒有通過 15% profile gate。
P4 attention 類型區塊探索已完成。
P4 沒有通過正確性 gate，且 prototype 已移除。
D2 routed expert top-6 profile 已完成。
D2 沒有通過 3% profile gate。
D3 CSA row trace 已完成。
D3 沒有通過 90% row gate 與 5% profile gate。
D3 沒有建立 prototype。
目前計畫中的 active trace 與 profile 工作都已完成。
D1 和 P5 仍等待新的 profile 證據。
新的記憶體最佳化研究需要新的 process-memory 證據。
之後再評估 staged streaming 或 MTLIO prototype。
如果 GPU operation 是主要 section，再測 compile 或 custom Metal kernel。
如果 cache eval 是主要 section，再修改 cache layout。

研究不得先假設瓶頸。

## 一手來源

- [固定 checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [固定 checkpoint tensor index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json)
- [DeepSeek reference inference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py)
- [safetensors metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX `gather_qmm`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)
- [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [Apple Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)
- [Apple Metal developer workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows)
- [DSpark paper](https://arxiv.org/html/2607.05147)
- [DeepSpec reference repository](https://github.com/deepseek-ai/DeepSpec)

外部主張的逐項查核位於
[`research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md`](../research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md)。
