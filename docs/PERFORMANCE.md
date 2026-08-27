# 效能與瓶頸

本文件定義 runtime 指標和瓶頸檢測方法。
量測結果請見[驗證紀錄](VALIDATION.md)。

## 先固定測試狀態

同一個 prompt 也可能得到不同速度。
下列狀態會改變結果。

- 第一次 MLX graph compile 或 warm graph。
- 作業系統 page cache。
- expert cache 的 resident slot。
- 記憶體 prompt cache。
- persistent prompt cache。
- prompt 長度和 routed expert 分布。
- output 長度。
- SSD、FileVault 和背景 I/O。

每個結果必須標示 fresh 或 warm 狀態。
「cold」只能用於有明確清除方法的 cache。
如果作業系統 page cache 沒有清除，請寫 `not purged`。

## 指標定義

### Request 時間

`RuntimeMetrics` 在 prompt tokenization 和 prompt cache acquisition 之後開始計時。
因此 runtime TTFT 不包含 tokenizer 時間。
runtime TTFT 也不包含 persistent prompt cache 載入時間。

`time_to_first_token_seconds` 從 runtime request 開始計到第一個 output token
完成 cache evaluation。

第一個 output token 出現前，App 使用即時 `request_seconds` 顯示
First Token wait time。
第一個 output token 出現後，App 改為顯示最終的
`time_to_first_token_seconds`。
效能歷史只記錄每個 request 的最終 First Token wait time。

`request_seconds` 在 generation 結束時停止。
`request_seconds` 不包含 response 結束後的 persistent cache serialize 和 write。

外部 client wall time 可能包含更多工作。
正式報告應同時記錄 client wall time 和 runtime request time。

### Prefill

runtime 使用下列公式：

```text
uncached_prompt_tokens = runtime_prompt_tokens - prompt_cache_reused_tokens
prefill_tokens_per_second = uncached_prompt_tokens / time_to_first_token_seconds
```

這個 prefill 指標包含第一個 output token 的工作。
這個 prefill 指標不是純 attention kernel throughput。

`request_prefill_step_size` 是本次 request 實際使用的 step。
`layer_major_prefill` 表示本次 request 是否進入 layer-major path。
`layer_major_prefill_tokens` 是 layer-major kernel 實際處理的 token 數。

`request_batched_expert_layers` 記錄 full-layer batched MoE 次數。
`request_gather_qmm_calls` 記錄 batched `gather_qmm` call 次數。

CLI 的 `prompt_token_sha256` 記錄完整 prompt token ID hash。
CLI 的 `token_sha256` 記錄完整 output token ID hash。
CLI 也會記錄 attention 與 MoE 的實際 Prefill 區塊大小。

### Decode

第一個 output token 計入 TTFT。
decode token 數使用：

```text
decode_tokens = max(0, runtime_generation_tokens - 1)
```

`accumulated_generation_tokens` 是目前 server process 產生的 output token 總數。
這個欄位包含進行中的 request。
server restart 會把這個欄位重設為 0。

主要 decode 指標如下。

| 欄位 | 意義 |
| --- | --- |
| `decode_model_step_seconds` | 取得後續 token 的 generator step wall time。這個時間可以包含 expert I/O 和 MLX evaluation。 |
| `decode_cache_eval_seconds` | step 後的 prompt cache array evaluation。 |
| `decode_end_to_end_seconds` | model step 加 cache evaluation。 |
| `decode_tokens_per_second` | decode token 除以 end-to-end seconds。 |
| `decode_model_step_tokens_per_second` | decode token 除以 model step seconds。 |
| `decode_latency_p50_seconds` | 每 token model step 加 cache evaluation 的 p50。 |
| `decode_latency_p95_seconds` | 每 token model step 加 cache evaluation 的 p95。 |

`performance.tokens_per_second` 是 server streaming 的即時速率。
正式結果應使用 `decode_tokens_per_second`。

### Expert I/O

| 欄位 | 意義 |
| --- | --- |
| `request_expert_bytes_read` | 本次 request 的 expert bytes。包含 full-layer batched read 和 slot miss。 |
| `request_expert_read_seconds` | 每個 read batch 的 critical wall time 加總。 |
| `request_ssd_read_bytes_per_second` | bytes 除以 expert read timer。 |
| `request_expert_cache_hits` | 個別 slot lookup hit。 |
| `request_expert_cache_misses` | 個別 slot lookup miss。 |
| `request_expert_evictions` | 本次 request 的 slot eviction。 |
| `request_prefetched_layer_hits` | 進入 batched layer 時，所有 prefetch worker 已完成的 layer 數。 |

full-layer batched prefill 不增加 expert cache hit 和 miss。
因此 expert cache hit rate 不能代表 prefill 的全部 I/O。

prefetch 可以和前一層 GPU work 重疊。
因此 expert read timer 不能直接從 request time 扣除。

歷史 M1 artifact 的 `prefill_no_file_cache` 記錄當時的實驗組別。
`false` 代表 Control。
`true` 代表 full-layer Prefill 使用另一組 `F_NOCACHE` file descriptor。
目前 CLI 不再輸出這個欄位。

### Routing synchronization

`request_routing_sync_seconds` 量測 `mx.eval(indices)` 的 boundary。
MLX 使用 lazy evaluation。
這個 boundary 可以等待 attention、router 和其他 upstream graph work。

請勿把 `request_routing_sync_seconds` 稱為純 router kernel time。
純 GPU 分解需要 Metal capture 或 GPU counter。

### Memory

| 欄位 | 來源 |
| --- | --- |
| `active_memory_bytes` | `mx.get_active_memory()` |
| `cache_memory_bytes` | `mx.get_cache_memory()` |
| `peak_memory_bytes` | `mx.get_peak_memory()` |
| APP memory usage | runtime Python process RSS |

四個值具有不同語意。
報告必須寫出指標名稱。

`mx.clear_cache()` 只清理 MLX cache memory。
`mx.clear_cache()` 不會移除仍被引用的 active array。
`mx.clear_cache()` 也不代表 process RSS 立即下降。

### Prompt cache

| 欄位 | 意義 |
| --- | --- |
| `prompt_cache_reused_tokens` | 本次 request 重用的 prefix token。 |
| `prompt_cache_snapshot_seconds` | layer-major prefill snapshot 時間。 |
| `prompt_cache_serialize_seconds` | quantized cache state serialize 時間。 |
| `prompt_cache_write_seconds` | safetensors 和 metadata write 時間。 |
| `prompt_cache_write_errors` | Write failure 數。 |

prompt cache write 在最後一個 token 之後執行。
client generator 必須正常結束，runtime 才能保存完成 entry。

### DSpark

`dspark_accepted_tokens` 只計算接受的 draft token。
`dspark_committed_tokens` 也包含每個 round 的 bonus 或 correction token。

```text
average_accepted_length = accepted_draft_tokens / rounds
average_committed_length = committed_tokens / rounds
```

DSpark throughput 比較必須使用 committed token。
DSpark 報告也必須列出 draft、verification、cache fork、cache replay 和 fallback。

## 目前瓶頸證據

2026-08-10 的 R0 baseline 使用五種固定 prompt。
每種 prompt 都測試 4,096、8,192 和 14,363 token。
每個 run 都產生 256 個 output token。

| Prompt token | TTFT 中位數 | Prefill 中位數 | Decode 中位數 | Expert bytes 中位數 | Misses 中位數 | Evictions 中位數 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 21.611 s | 189.53 Tok/s | 8.00 Tok/s | 473.970 GB | 24,700 | 23,548 |
| 8,192 | 39.124 s | 209.39 Tok/s | 7.76 Tok/s | 477.393 GB | 24,956 | 23,804 |
| 14,363 | 67.466 s | 212.89 Tok/s | 7.43 Tok/s | 482.593 GB | 25,345 | 24,193 |

每種 prompt 和長度只有一個 run。
這些中位數表示跨 prompt 的中心值。
這些結果不表示重複執行的變異。

4K repeated prompt 的 Decode 是 12.22 Tok/s。
4K repeated prompt 有 1,535 次 miss 和 383 次 eviction。
四種非 repeated prompt 的 Decode 是 6.44 至 8.28 Tok/s。
四種非 repeated prompt 有 24,307 至 28,765 次 miss。
四種非 repeated prompt 有 23,155 至 27,613 次 eviction。

因此，repeated prompt 低估一般 Decode 的 SSD 與 slot 壓力。
後續效能決策不能只使用 repeated prompt。

### Metal boundary 結果

R1 對 repeated 4K、Tool-like 4K 和 Tool-like 14K 執行
`Metal System Trace`。

| Workload | Prefill GPU busy | Decode GPU busy | Decode GPU idle |
| --- | ---: | ---: | ---: |
| Repeated 4K | 57.6% | 57.8% | 2.604 s |
| Tool-like 4K | 60.7% | 35.5% | 7.968 s |
| Tool-like 14K | 70.2% | 37.1% | 8.114 s |

Tool-like Decode 有較長的 GPU idle boundary。
Tool-like Decode 同時有較多 expert miss、expert read 和 `preadv` CPU sample。
因此，R2 先測試 slot 與 worker 設定。

14K Tool-like Prefill 的 GPU busy 是 70.2%。
這個結果高於 4K Tool-like 的 60.7%。
目前不應假設長 Prefill 只受 SSD 限制。

R1 Metal System Trace 沒有提供 shader timing sample。
R1 資料不能把 GPU time 分配到 attention、routed expert 和 shared expert。
request 結束時間是依最後一個 Python GPU interval 推定。
capture wall time 不是正式效能結果。

CPU profile 沒有顯示 MLX 圖走訪或 identity index 穩定超過 5%。
因此，D1 和 P5 的 profile gate 都沒有通過。

完整資料位於
[`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)。

### P3 `gather_qmm` shader profile

P3 使用 `Metal GPU Counters` capture Tool-like 4K Prefill。
這次 capture 產生 shader timeline。
capture 開啟時的 wall time 不是 production 效能結果。

目前 `gather_qmm` kernel 是
`mxfp4_gather_qmm_rhs_nax`。
kernel 使用 BM64、BN64 和 BK64。

| 指標 | 結果 |
| --- | ---: |
| Request | 23.220 秒 |
| Python GPU busy | 13.306 秒 |
| Shader sample work | 12.146 秒 |
| `gather_qmm` shader sample | 2.966 秒 |
| `gather_qmm` / Request | 12.77% |
| 涵蓋率校正後 `gather_qmm` / Request | 13.99% |

shader sample work 涵蓋 Python GPU busy 的 91.28%。
涵蓋率校正假設未分配的 GPU busy 使用相同 shader 組成。

trace 包含 42 個 expert layer burst。
runtime metrics 記錄 84 次 `gather_qmm` call。
shader timeline 有 108 個 `gather_qmm` interval。
一個 logical call 可以被 profiler 分成多個 interval。

同一個固定 prompt 的 R0 route trace 顯示，平均是 95.98 rows/expert。
24.53% 的 expert run 是空的。
27.37% 的 expert run 只有 1 至 15 rows。
BM64 的理論 padding 是配置 rows 的 24.43%。

P3 的停止門檻是 Prefill 的 15%。
原始結果與涵蓋率校正結果都低於門檻。
P3 已停止。
專案沒有建立 isolated MLX build，也沒有修改 runtime。

完整資料位於
[`benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json`](benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json)。

### P4 attention step pilot

P4 使用 Tool-like 4K prompt 和一個 output token。
P4 依 attention 類型測試 512、2,048 和 4,096 step。
目前全域 Control step 是 1,024。

較大的 step 增加 temporary memory，且沒有穩定改善 phase time。
512 step 在單次 run 中降低 CSA 和 dense compressed attention phase。
但是，Control request 在研究期間漂移 7.91%。
這些單次 request time 不是正式速度結果。

組合的 512-step 候選改變 39/42 個 MoE layer 的 expert route。
至少 32,075 個 assignment 改變。
差異下限是 3.11%。
候選的 expert bytes 也減少 6 個 expert blob。

P4 沒有通過正確性 gate。
專案沒有執行正式 ABBA。
runtime 已移除 P4 prototype，並保留單一全域 step。

完整資料位於
[`benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json`](benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json)。

### D2 routed expert top-6 profile

D2 使用 Tool-like 4K prompt 和 64 個 output token。
runtime metrics 記錄 63 個 Decode token。
D2 先量測 `_expert_select` Python dispatch。
D2 再使用 `Metal GPU Counters` 量測相同工作負載。

| 指標 | 結果 |
| --- | ---: |
| Decode wall | 12.916 s |
| `_expert_select` Python dispatch | 13.252 ms |
| Dispatch / Decode wall | 0.121% |
| Decode GPU busy | 4.559 s |
| Shader sample work | 2.137 s |
| Shader Timeline 涵蓋率 | 46.88% |
| `_expert_select` shader sample | 58.834 ms |
| Shader sample / Decode wall | 0.456% |
| 涵蓋率校正後 / Decode wall | 0.972% |

一個 bfloat16、256-to-6 microbenchmark 確認 6 個 `_expert_select` shader label。
microbenchmark 只用來對應 shader label。
microbenchmark 的同步 wall time 不用於 profile gate。

涵蓋率校正假設未分配的 GPU busy 使用相同 shader 組成。
D2 的停止門檻是 Decode wall time 的 3%。
dispatch、原始 shader sample 和涵蓋率校正結果都低於門檻。
D2 已停止。
專案沒有建立專用 top-6 prototype，也沒有修改 runtime。

完整資料位於
[`benchmarks/2026-08-11-d2-top6-profile-m5-pro.json`](benchmarks/2026-08-11-d2-top6-profile-m5-pro.json)。

### D3 CSA row 與 `gather` profile

D3 使用 Tool-like prompt 和 64 個 output token。
D3 分別記錄 512、2,048、4,096、8,192 和 14,363 token context。
每個 context 有 63 個相鄰 Decode transition。
每個 transition 比較 21 個 CSA layer。

| Context | 交集比例中位數 | Jaccard 中位數 |
| ---: | ---: | ---: |
| 512 | 100.00% | 100.00% |
| 2,048 | 99.02% | 98.07% |
| 4,096 | 78.71% | 64.90% |
| 8,192 | 70.51% | 54.45% |
| 14,363 | 62.30% | 45.25% |

五種 context 的合併交集比例中位數是 85.94%。
這個結果低於 90% row gate。
Row trace 會同步 selected row array。
因此，row trace 的 wall time 不是速度結果。

D3 使用 microbenchmark 對應 `MXFP8PoolingCache.gather` shader label。
D3 使用既有的 D2 Tool-like 4K Metal trace 計算 Decode 占比。

| 指標 | 結果 |
| --- | ---: |
| Decode wall | 12.916 s |
| Shader Timeline 涵蓋率 | 46.88% |
| 核心 `gather` shader sample | 15.118 ms |
| 核心 sample / Decode wall | 0.117% |
| 核心 sample，依涵蓋率校正 | 0.250% |
| 包含通用 clip/select shader sample | 20.467 ms |
| 包含通用 shader，依涵蓋率校正 | 0.338% |

通用 clip 和 select shader 也可能出現在其他 runtime 操作。
因此，0.338% 不是純 `gather` 結果。
核心結果與包含通用 shader 的結果都低於 5% profile gate。

D3 沒有通過兩個 gate。
D3 已停止。
專案沒有建立 row 重用 prototype，也沒有修改 runtime。

完整資料位於
[`benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json`](benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json)。

### R2a slot 結果

R2a 使用探索性 ABBA 比較 1,152 與 2,048 slots。

| Prompt | 1,152 slots | 2,048 slots | Decode 差異 | MLX peak 差異 |
| --- | ---: | ---: | ---: | ---: |
| Repeated 4K | 14.42 Tok/s | 13.95 Tok/s | -3.31% | +2.901 GiB |
| Code 4K | 8.25 Tok/s | 2.39 Tok/s | -71.09% | +11.156 GiB |

Code 的 2,048-slot 設定把 expert bytes 降低 21.73%。
Miss 降低 31.34%。
Eviction 降低 36.77%。
但是，expert read timer 沒有降低。
Routing synchronization boundary 從 9.829 秒增加到 111.488 秒。

這個結果顯示較高的 cache hit rate 不等於較高的端到端速度。
2,048-slot 候選設定已拒絕。
目前預設值維持 1,152 slots。

完整資料位於
[`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

### R2b 2-worker 結果

R2b 使用 code 4K prompt 執行一輪 ABBA。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8.29 Tok/s | 110.58 ms | 172.12 ms | 28.042 s |
| 2 | 7.94 Tok/s | 116.54 ms | 182.41 ms | 29.435 s |

2 workers 讓 Decode 降低 4.20%。
兩個配對結果都是負值。
Expert bytes、miss、eviction 和 routing boundary 沒有明顯差異。
Expert read timer 增加 4.97%。

2-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

### R2b 8-worker 結果

R2b 使用相同的 code 4K prompt 執行一輪 ABBA。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8.27 Tok/s | 110.88 ms | 173.60 ms | 28.202 s |
| 8 | 8.29 Tok/s | 110.61 ms | 173.13 ms | 27.875 s |

8 workers 只讓 Decode 增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
這個結果低於 3% 採用門檻。

8-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

### R2c 4-prefetch-worker 結果

R2c 使用 code 4K prompt 和一個 output token 執行一輪 ABBA。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read |
| ---: | ---: | ---: | ---: |
| 2 | 21.956 s | 186.56 Tok/s | 14.478 s |
| 4 | 22.061 s | 185.67 Tok/s | 12.009 s |

4 workers 讓 expert read timer 降低 17.05%。
但是，TTFT 增加 0.48%。
這個結果顯示 expert read timer 不是可以直接相加的 request boundary。

4-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

### R2c 1-prefetch-worker 結果

R2c 使用相同的 code 4K prompt 執行一輪 ABBA。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 22.037 s | 185.87 Tok/s | 14.531 s | 39.5 |
| 1 | 23.740 s | 172.54 Tok/s | 19.394 s | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
Expert read timer 增加 33.46%。
Prefetch hit 中位數從 39.5 降到 0.5。

1-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。
R2 設定 sweep 已完成。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

### R3 DSpark 結果

R3 使用 code 4K prompt 和 64 個 output token 執行一輪 ABBA。
R3 比較 normal generation 與 DSpark。

| Mode | Request 中位數 | TTFT 中位數 | 總 throughput | 回報的 Decode | Main model expert bytes | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 31.152 s | 22.072 s | 2.054 Tok/s | 6.94 Tok/s | 239.659 GB | 23.274 GiB |
| DSpark | 41.301 s | 33.722 s | 1.549 Tok/s | 8.32 Tok/s | 316.666 GB | 27.257 GiB |

DSpark 讓 request 增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都接受全部 5 個 draft token。
兩個 DSpark run 都提交 6 個 token。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
Runtime 的成本 gate 因此判定 speculative work 比 normal step 昂貴。

回報的 Decode boundary 增加 19.82%。
這個 boundary 從第一個 token 之後開始。
這個 boundary 不包含較慢的 DSpark prefill。
DSpark 也沒有在整個 Decode phase 持續運作。
因此，端到端決策必須使用 request 與總 throughput。

Normal run 記錄 42 個 batched expert layer 和 84 次 `gather_qmm`。
DSpark run 的兩個欄位都是 0。
DSpark 的 main model expert bytes 增加 32.13%。
DSpark 的 main model cache miss 增加 230.16%。

四個 output token hash 都相同。
R3 已觸發停止條件。
研究沒有執行正式五個 prompt 測試。
目前 DSpark 預設值維持停用。

完整資料位於
[`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

### M1 Prefill file-cache 歷史結果

正式研究已結束。
runtime 已移除 M1 prototype。

M1 使用 code 4K prompt 和 256 個 output token 執行一輪 ABBA。
Control 使用一般 file descriptor。
M1 為 full-layer Prefill 使用獨立 `F_NOCACHE` file descriptor。
Decode 仍使用一般 file descriptor。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 52.508 s | 21.884 s | 187.17 Tok/s | 8.33 Tok/s | 27.945 s | 23.296 GiB |
| M1 | 50.239 s | 20.221 s | 202.56 Tok/s | 8.50 Tok/s | 24.542 s | 23.297 GiB |

M1 讓 request 改善 4.32%。
M1 讓 TTFT 改善 7.60%。
M1 讓 Prefill throughput 增加 8.22%。
M1 讓 Decode throughput 增加 2.02%。
M1 讓 expert read timer 降低 12.18%。

兩個 request 配對都改善超過 4%。
四個 output token hash 都相同。
Expert bytes、miss 和 eviction 都相同。

Process peak footprint 只增加 1.77 MiB。
M1 沒有達到至少降低 5% 的記憶體接受條件。
測試也沒有出現可重現的 memory pressure。
因此，專案不把 M1 採用為記憶體最佳化。

System pagein 計數降低 61.88%。
這個計數包含其他 process，也受到 ABBA 順序影響。
這個計數不能取代 process footprint。

本次測試只有一種 workload 和一輪 ABBA。
作業系統 page cache 沒有清除。
本次測試沒有 bootstrap 信賴區間。
Code pilot 本身不能支持正式效能採用。
這個 pilot 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)。

#### Tool-like 4K 獨立重跑

Tool-like 重跑使用相同設定和 ABBA 順序。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 62.574 s | 22.860 s | 179.33 Tok/s | 6.42 Tok/s | 36.282 s | 23.296 GiB |
| M1 | 60.380 s | 20.966 s | 195.38 Tok/s | 6.47 Tok/s | 33.173 s | 23.297 GiB |

M1 讓 request 改善 3.51%。
M1 讓 TTFT 改善 8.29%。
M1 讓 Prefill throughput 增加 8.95%。
M1 讓 Decode throughput 增加 0.75%。
M1 讓 expert read timer 降低 8.57%。

兩個 request 配對分別改善 2.34% 和 4.63%。
兩個配對都是同向改善。
四個 output token hash 都相同。
Process peak footprint 只增加 0.98 MiB。

Code 與 Tool-like 的 request 中位數分別改善 4.32% 和 3.51%。
兩種 workload 的 TTFT 分別改善 7.60% 和 8.29%。
這個結果確認 pilot 等級的效能訊號。

System swap 在完整序列中增加 2.13 MiB。
這個 system-wide 變化不能歸因於 M1。
後續正式測試因此繼續監控 swap。

兩種 workload 都只有一輪 ABBA。
這兩個 pilot 當時沒有 bootstrap 信賴區間。
專案因此進入正式多 workload 重複 ABBA。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

#### 正式效能評估 Wave 1

Wave 1 使用五個 R0 4K prompt。
每個 prompt 有一個不納入統計的 Control warm-up。
每個 prompt 接著執行一輪 ABBA。
Wave 1 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 3.03% | 8.23% | 8.98% | -2.93% |
| Code | 4.54% | 8.55% | 9.34% | 1.73% |
| 繁體中文技術文字 | 3.40% | 8.12% | 8.84% | 0.13% |
| Mixed math | 3.69% | 8.24% | 8.98% | 0.57% |
| Tool-like | 2.70% | 7.97% | 8.66% | -0.32% |

四個主要 workload 的 request 改善中位數是 3.54%。
TTFT 改善中位數是 8.18%。
Prefill throughput 增加中位數是 8.91%。
Decode throughput 增加中位數是 0.35%。
八個主要 workload request 配對都改善。

四個主要 workload 的 process peak footprint 增加中位數是 0.36 MiB。
正式序列的 system swap 降低 8.00 MiB。
測試沒有出現 thermal 或 performance warning。

Repeated Decode 在本輪降低 2.93%。
後續 wave 因此繼續確認這個 regression。
Wave 1 單獨不能支持正式採用。
Wave 1 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

#### 正式效能評估 Wave 2

Wave 2 輪替 prompt 順序，並完成 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.63% | 11.05% | 12.48% | 3.64% |
| Code | 3.78% | 7.83% | 8.49% | 0.93% |
| 繁體中文技術文字 | 3.66% | 8.26% | 9.01% | 0.47% |
| Mixed math | 3.57% | 8.51% | 9.30% | 0.17% |
| Tool-like | 2.61% | 7.77% | 8.43% | -0.36% |

四個主要 workload 的 request 改善中位數是 3.62%。
TTFT 改善中位數是 8.05%。
Prefill throughput 增加中位數是 8.75%。
Decode throughput 增加中位數是 0.32%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 降低 8.00 MiB。
System-wide compressor 增加 209.67 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

#### 正式效能評估 Wave 3

Wave 3 再次輪替 prompt 順序，並完成 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.93% | 11.87% | 13.45% | 3.25% |
| Code | 4.08% | 7.71% | 8.35% | 1.53% |
| 繁體中文技術文字 | 4.28% | 8.68% | 9.50% | 1.22% |
| Mixed math | 3.34% | 6.97% | 7.52% | 0.84% |
| Tool-like | 2.07% | 8.48% | 9.17% | -1.64% |

四個主要 workload 的 request 改善中位數是 3.71%。
TTFT 改善中位數是 8.09%。
Prefill throughput 增加中位數是 8.76%。
Decode throughput 增加中位數是 1.03%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 沒有變化。
System-wide compressor 增加 244.77 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

#### 正式效能評估 Wave 4

Wave 4 再次輪替 prompt 順序，並完成 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.76% | 10.82% | 12.18% | 4.34% |
| Code | 3.31% | 5.81% | 6.17% | 1.53% |
| 繁體中文技術文字 | 3.56% | 7.95% | 8.63% | 0.48% |
| Mixed math | 3.17% | 7.75% | 8.42% | 0.01% |
| Tool-like | 3.32% | 7.90% | 8.56% | 0.63% |

四個主要 workload 的 request 改善中位數是 3.32%。
TTFT 改善中位數是 7.82%。
Prefill throughput 增加中位數是 8.49%。
Decode throughput 增加中位數是 0.56%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 沒有變化。
System-wide compressor 增加 340.41 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

#### 正式效能評估 Wave 5

Wave 5 再次輪替 prompt 順序，並完成 20 個有效量測 run。
第一次序列在 Tool-like B1 啟動時失去執行 session。
專案排除整個不完整序列，並從新的 warm-up 完整重跑。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 8.36% | 12.44% | 14.24% | 3.89% |
| Code | 2.85% | 6.26% | 6.68% | 0.51% |
| 繁體中文技術文字 | 3.44% | 9.36% | 10.32% | -0.69% |
| Mixed math | 3.68% | 8.83% | 9.66% | 0.14% |
| Tool-like | 2.36% | 6.32% | 6.82% | 0.10% |

四個主要 workload 的 request 改善中位數是 3.15%。
TTFT 改善中位數是 7.58%。
Prefill throughput 增加中位數是 8.24%。
Decode throughput 增加中位數是 0.12%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 降低 8.56 MiB。
System-wide compressor 增加 1,236.50 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)。

#### 五輪正式結果

五輪共有 100 個有效量測 run 和 50 個配對。
全部 50 個 request 配對都改善。

四個主要 workload 的 request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
TTFT 改善中位數是 7.99%。
Prefill throughput 增加中位數是 8.69%。
Decode throughput 增加中位數是 0.23%。
Expert read timer 改善中位數是 10.88%。

主要 workload 的 Decode p95 regression 最大值是 1.71%。
這個結果通過 2% 條件。
Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。

Process peak footprint 的配對中位數增加 0.32 MiB。
配對 p95 增加 19.24 MiB。
每一輪的 system swap 都沒有增加。
部分 wave 的 system-wide compressor 總量增加。
這個總量包含測試兩組與其他 process，因此不能歸因於 M1。
這組證據仍沒有證明 compressor 不增加。

M1 沒有通過全部正式採用條件。
專案拒絕預設啟用 M1。
runtime 已移除 M1 prototype。

完整結果位於
[`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

## 瓶頸判讀

| 觀察 | 先檢查 | 不應直接下的結論 |
| --- | --- | --- |
| TTFT 高，`batched_expert_layers=42` | expert bytes、read timer、prefetch hits、Metal capture | 所有 TTFT 都是 SSD。 |
| Expert bytes 高，read throughput 低 | direct SSD microbenchmark、背景 I/O、worker A/B | 增加 worker 一定更快。 |
| Cache miss 和 eviction 都高 | slot 數、prompt 類型、每層 reserve | 提高 hit rate 一定提高端到端速度。 |
| Routing sync 高 | 前一個 `mx.eval` boundary、attention 和 MoE graph | router kernel 本身很慢。 |
| Decode cache eval 高 | cache array 數、context、MXFP8 chunk、persistence path | KV cache 量化必定較快。 |
| `gather_qmm_calls=0` 但長 prefill | layer-major 和 batched flag、未快取 token 數 | MLX 不支援 `gather_qmm`。 |
| Prompt reuse 高但 TTFT 仍高 | persistent cache load、suffix 長度、第一個 decode step | prompt cache 沒有效果。 |
| DSpark acceptance 高但速度低 | committed length、draft time、verify time、fallback | acceptance 高等於 throughput 高。 |

## 檢測流程

### 1. 建立 baseline

先固定 checkpoint、commit、prompt 和 output token 數。
先停用 persistent prompt cache。
先使用 greedy generation。

R0 可以用下列命令建立固定 prompt 檔。

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/prepare_r0_prompts.py \
  --model /path/to/model.dsv4 \
  --output scratch/r0-profile/prompts
```

prompt manifest 會保存 prompt token 數、文字 SHA-256 和
`prompt_token_sha256`。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file fixed-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --top-p 1 \
  --no-persistent-prompt-cache \
  --metrics-json baseline.json
```

### 2. 一次只改一個設定

runtime 提供下列比較開關。

| 問題 | 比較設定 |
| --- | --- |
| Layer-major 收益 | `--no-layer-major-prefill` |
| Batched MoE 收益 | `--no-batched-expert-prefill` |
| Attention step | `--prefill-step-size N` |
| MoE tile | `--moe-prefill-step-size N` |
| 一般 read concurrency | `--read-workers N` |
| Full-layer prefetch concurrency | `--prefetch-read-workers N` |
| Slot 壓力 | `--slots N` |
| KV cache | `--bf16-kv-cache` |
| Index cache | `--no-fp4-index-cache` |
| Ready expert decode | `--no-ready-expert-decode` |
| Prompt persistence | `--no-persistent-prompt-cache` |
| DSpark | `--dspark` |

每組 A/B 必須產生相同 greedy token hash。
如果 output token 不同，速度比較無效。

### 3. 分開 SSD microbenchmark

```sh
swift run dsv4-repack benchmark \
  --model /path/to/model.dsv4 \
  --samples 32
```

microbenchmark 只回答 expert blob read 能力。
microbenchmark 不包含 router、GPU compute、slot admission 或 MLX graph。

### 4. 記錄 routed expert trace

CLI 可以記錄 prefill histogram、decode route 和真實 miss。
目前 trace 要求 DSpark 停用。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file fixed-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --expert-route-trace routes.json

PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.route_trace \
  routes.json \
  --hot-per-layer 6 8 \
  --decode-tokens 256
```

route coverage 是 counterfactual 分析。
route coverage 不等於實際 speedup。

route trace 的 `prefill_chunk_histograms` 保存每個 Prefill MoE 區塊的
rows/expert。
route trace 會同步 router index。
因此 route trace 的 wall time 不是正式效能結果。

### 5. 使用 Metal capture

Metal capture 用來分解 GPU operation 和 CPU/GPU wait。
capture 開啟時的 wall time 不是正式效能結果。

capture 應回答下列問題。

- attention、routed MoE 和 shared expert 各使用多少 GPU time。
- `gather_qmm` 是否使用預期 shape 和排序 path。
- graph 是否有大量小 operation 或空檔。
- CPU `preadv`、router index materialization 和 GPU submission 是否重疊。
- 某一個 section 是否穩定超過 request time 的 5%。

下列命令會啟動一個獨立的 capture run。
capture 的 metrics 不可作為正式 wall-time 結果。

```sh
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output scratch/r0-profile/repeated-4096.trace \
  --env PYTHONPATH=runtime \
  --target-stdout /dev/null \
  --launch -- .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file scratch/r0-profile/prompts/repeated-4096.txt \
  --max-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --no-persistent-prompt-cache \
  --metrics-json scratch/r0-profile/repeated-4096-capture.json
```

只有 capture 證明 section 足夠大時，才評估 custom Metal kernel。

## API input size benchmark

`Scripts/benchmark_api.py` 透過 Whallm API 執行 input size benchmark。
預設 input size 是 1K、4K、16K 和 32K token。
腳本使用 installed model 的 tokenizer 建立精確 token 數的 raw completion prompt。
腳本也會確認 server 回報相同的 input token 數。

```sh
make benchmark-dsv4
make benchmark-qwen
```

Python CLI 的預設值是每個 input size 執行 5 次。
目前兩個 Make entry 明確指定每個 input size 執行 3 次。

腳本會先重用目前的 Server。
本機 Server 未執行時，腳本會從 Whallm model folder 找出指定的 installed model，
並使用 App 已儲存的設定啟動 Server。
測試完成後，腳本只會停止自己啟動的 Server。
腳本不會停止 App 或其他 process 啟動的 Server。

`--model-path` 可以指定另一個 installed model。
自動啟動只支援本機 `http://HOST:PORT/v1` URL。
`--server-start-timeout` 的預設值是 900 秒。
API key 從 `OPENAI_API_KEY` environment variable 讀取。

```sh
OPENAI_API_KEY='<local-key>' \
  .venv/bin/python Scripts/benchmark_api.py \
  --model-path /Volumes/Models/deepseek-v4-flash-0731.dsv4 \
  --model deepseek-v4-flash-0731 \
  --runs 5
```

```sh
.venv/bin/python Scripts/benchmark_api.py \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --runs 5 \
  --max-output-tokens 64
```

每個 run 記錄最終 TTFT、Prefill tok/s、Decode tok/s 和 request 期間最高的
`performance.active_memory_bytes`。
每個 input size 的 Peak 和 P95 使用個別 run 的結果計算。
P95 使用 nearest-rank 方法。
少於 20 個 run 時，nearest-rank P95 會等於 Peak。

腳本會在終端顯示 ASCII 表格。
腳本也會把每個 run、source、environment、runtime configuration、cache state
和 output text hash 寫入 `docs/benchmarks/`。
artifact 的 `server.started_by_benchmark` 會記錄 Server 是否由腳本啟動。
每個 run 使用不同的 prompt prefix。
腳本不會清除 prompt cache 或作業系統 page cache。
Memory 是 MLX active memory，不是 process RSS。
API 不提供生成 token ID，所以 artifact 不宣稱具有正式 output token hash。

## A/B 驗收

正式 A/B 使用下列規則。

1. 使用至少五個代表性 prompt。
2. 每個 prompt 產生至少 256 個 token，除非測試目標是 TTFT。
3. A/B 執行順序要交錯。
4. 分開 fresh runtime、warm graph 和 OS page-cache 狀態。
5. 保存每次 run，不只保存平均值。
6. 報告 median 和 p95。
7. 比較完整 token hash。
8. 記錄 peak memory 和 process RSS。
9. 記錄 expert bytes、read timer、hit、miss 和 eviction。
10. 任何改善不得讓短 prompt 或低記憶體設定發生未說明的 regression。

## 目前不成立的結論

- 官方 DSpark 的 60–85% 不是本機 batch size 1 的預期倍率。
- Apple unified memory 不代表任意 `MTLBuffer` 可以由 Python MLX zero-copy 使用。
- MTLIO API 存在不代表本 runtime 會更快。
- `mx.compile` 可用不代表所有 graph 都會融合或加速。
- 高 route coverage 不代表 handoff policy 會降低實際 miss。
- 更大的 slot cache 不代表更快。slot 也會增加記憶體和 eviction scan 成本。

目前研究狀態請見[研究結論](RESEARCH.md)。
