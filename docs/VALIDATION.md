# 驗證紀錄

本文件分開記錄目前驗證和歷史量測。
目前自動測試使用 base commit `9e7f2f377662e59a44275ff632f5c434adc2d5c4`
和本文件描述的未提交工作樹變更。
完整 installed model 驗證和正式量測使用 commit
`57e440e48b79fb0399beae7e9965481f74b35b25`。
每個探索性 artifact 另行記錄 working tree source hash。
歷史量測只用來說明演進和決策。

機器可讀的目前結果如下。

- [`benchmarks/2026-08-10-m5-pro.json`](benchmarks/2026-08-10-m5-pro.json)：clean checkout 的單一 repeated prompt 驗證。
- [`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)：目前 checkout 的 R0 多 prompt baseline、route profile 和 Metal capture。
- [`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)：R2a 1,152/2,048-slot 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)：R2b 4/2 read-worker 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)：R2b 4/8 read-worker 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)：R2c 2/4 prefetch-worker TTFT 探索性 ABBA。
- [`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)：R2c 2/1 prefetch-worker TTFT 探索性 ABBA。
- [`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)：R3 normal/DSpark 端到端探索性 ABBA。
- [`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)：M1 Prefill file-cache 端到端探索性 ABBA。
- [`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)：M1 Tool-like 4K 獨立探索性 ABBA。
- [`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)：M1 正式效能評估 Wave 5。
- [`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)：M1 五輪正式統計與最終決定。
- [`benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json`](benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json)：P3 `gather_qmm` shader profile gate。
- [`benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json`](benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json)：P4 attention step 探索、記憶體與 expert route 比較。
- [`benchmarks/2026-08-11-d2-top6-profile-m5-pro.json`](benchmarks/2026-08-11-d2-top6-profile-m5-pro.json)：D2 routed expert top-6 dispatch 與 shader profile gate。
- [`benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json`](benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json)：D3 相鄰 Decode CSA row 與 `gather` profile gate。
- [`benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json`](benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json)：Qwen 完整安裝、API、4K prompt cache 與 packaged App 驗證。

## 證據標記

| 標記 | 意義 |
| --- | --- |
| 目前驗證 | 本次使用目前程式碼重跑。 |
| 歷史量測 | 舊版本留下的本機結果。沒有使用目前程式碼重跑。 |
| 單元測試 | 驗證局部邏輯。單元測試不代表 full-model 速度或品質。 |
| 未驗證 | 程式可能允許該設定，但本專案沒有完整證據。 |

## 正式量測環境

| 項目 | 值 |
| --- | --- |
| 日期 | 2026-08-10 至 2026-08-11 |
| Commit | `57e440e48b79fb0399beae7e9965481f74b35b25` |
| Mac | MacBook Pro `Mac17,8` |
| SoC | Apple M5 Pro |
| CPU / GPU | 18 cores / 20 cores |
| 統一記憶體 | 64 GB |
| Metal | Metal 4 |
| 模型磁碟 | 內接 APFS SSD，Apple Fabric，FileVault 開啟 |
| macOS | 27.0 build `26A5388g` |
| Xcode | 27.0 build `27A5218g` |
| Python | 3.14.6 |
| MLX | 0.32.0 |
| mlx-lm package | 0.31.3 |
| mlx-lm source | `Blaizzy/mlx-lm@5c10538136b9038b9626c134612b08afc18d697a` |
| Transformers | 5.12.1 |

目前 mlx-lm 來源是固定 fork。
目前 mlx-lm 來源不是相同版本的 upstream release。

## 自動測試

2026-08-27 使用當時的工作樹執行 Swift 測試。
2026-08-28 在 Codex 和 Qwen request 格式修正後，重新執行 Python 測試。

```sh
make test
PYTHONPATH=runtime .venv/bin/python -m unittest discover -s runtime/tests -v
```

下表合併每個 suite 的最近一次結果。

| Suite | 通過 | 失敗 | 略過 |
| --- | ---: | ---: | ---: |
| Swift `DeepSeekRepackTests` | 16 | 0 | 0 |
| Swift `DeepSeekV4SSDAppTests` | 39 | 0 | 3 |
| Python runtime 與 server | 124 | 0 | 0 |
| 合計 | 179 | 0 | 3 |

三個 App 測試因本機沒有完整 installed model 而略過。
2026-08-28 的目前工作樹已執行 `make package`。
該 package 包含 2026-08-28 的 `QwenToolCodec` 修正。
App 版本是 `1.1.0`，build 是 `1.1.1`。
Apple notary submission `1994e20c-5298-46c5-bca6-852e8cfdb1cc`
的狀態是 `Accepted`。
本機 App 和解壓後的 ZIP 都通過 Developer ID、stapled ticket 和
Gatekeeper 檢查。
ZIP SHA-256 是
`da6a638ad738da811f2ab5a62c1b7c0e7d1d99f29475ed59664c9e1c06e0fd50`。
本次驗證沒有建立 Git tag 或 GitHub Release。
兩個 App 都包含英文、簡體中文和繁體中文 localization。
隔離啟動檢查禁止 App 讀取專案 `.build` 目錄。
隔離啟動檢查也禁止 App 讀取 Swift resource bundle。
三種 localization 都直接從 `Contents/Resources` 載入。
App 在每種語言下都持續執行，且沒有在 `L10n` 初始化時停止。

測試覆蓋下列關鍵行為。

- repack plan 和 expert blob layout。
- 無效 expert shape 和缺少 tensor 的拒絕行為。
- 8 MiB bounded copy、續傳 receipt、SHA failure 和 repair。
- APP installed model 掃描、server 設定、設定儲存、Keychain 和測試對話儲存。
- Power Saving Mode Slider 的 legend 節點對齊。
- MXFP4 個別與 batched expert output。
- ready expert decode 的 router 順序。
- MXFP8 cache chunk、gather、index 和 persistence round-trip。
- layer-major prefill 和一般 path 的 next-token logits。
- 記憶體與 persistent prompt cache reuse。
- Output token 的 process 累計值。
- DSpark greedy、sampling、verification、replay 和 fallback 邏輯。
- Chat Completions、Responses、Text Completions、tool 和 SSE。
- SPEED-Bench prompt 的精確 input token 數和完整 chat template 尾端。
- Codex Responses request 到 Qwen chat template 的完整 codec 格式轉換。
- 空 model catalog、API model ID、Alias、未知模型和模型載入失敗重試。
- 延遲載入、runtime 重用、關閉後切換、generation request 排隊和累計計數。
- 模型載入期間的 `/healthz` 和 `/api/status` 回應。
- APP Alias 儲存與遷移、model catalog、Chat 模型選擇和訊息模型名稱。
- 未載入、載入中與已載入的 status decoding，以及模型切換後清除 Metric 歷史。

這些測試多數使用 fixture 或 mock model。
這些測試不取代 full-model benchmark。
多模型生命週期測試使用 fake runtime。
目前工作區沒有兩個完整 installed model。
本次驗證沒有執行真實雙模型切換或 RSS 量測。

2026-08-28 也使用目前 installed Qwen tokenizer 驗證完整格式轉換路徑。
驗證路徑是 Codex Responses request、server normalization、`QwenToolCodec` 和
Qwen `chat_template.jinja`。
驗證內容包含 `instructions`、`developer` message、text content item array、
後置 instruction、沒有 user 的 history、namespace tool、`web_search`、
巢狀 arguments、`function_call_output` 和 Qwen 保留標記。
role sequence 矩陣涵蓋 255 組 server 接受的非 tool history。
installed tokenizer 也通過含 tool history 的 prompt boundary 檢查。
這次驗證沒有載入完整模型權重，也沒有執行 generation。

## Installed model 完整驗證

目前驗證執行：

```sh
swift run dsv4-repack verify \
  --model scratch/deepseek-v4-flash-0731.dsv4
```

結果是通過。
驗證會重新計算每個 manifest file 的 SHA-256。

| 欄位 | 結果 |
| --- | ---: |
| checkpoint revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| manifest files | 54 |
| manifest file bytes | 166,884,980,648 |
| main model packed weight bytes | 156,015,738,880 |
| common tensor count | 1,564 |
| main model `common.bin` | 8,846,000,128 bytes |
| DSpark packed weight bytes | 10,862,841,600 |
| file size 驗證 | 通過 |
| SHA-256 驗證 | 通過 |

## 目前 SSD microbenchmark

目前驗證執行：

```sh
swift run dsv4-repack benchmark \
  --model scratch/deepseek-v4-flash-0731.dsv4 \
  --samples 32
```

每列讀取 32 個 expert blob。
每個 expert blob 是 13,369,344 bytes。

`direct` 對 file descriptor 設定 `F_NOCACHE`。
`cached` 先讀取一次相同 expert blob，然後重複讀取該 blob。

| Mode | Workers | GiB/s | Mean read time |
| --- | ---: | ---: | ---: |
| Direct | 1 | 10.76 | 1.16 ms |
| Direct | 2 | 12.94 | 1.90 ms |
| Direct | 4 | 12.89 | 3.69 ms |
| Direct | 8 | 12.95 | 6.87 ms |
| Cached | 1 | 25.21 | 0.49 ms |
| Cached | 2 | 42.87 | 0.57 ms |
| Cached | 4 | 60.40 | 0.79 ms |
| Cached | 8 | 72.79 | 1.28 ms |

本次 direct throughput 在 2、4 和 8 workers 間很接近。
8 workers 的 mean read time 最高。
目前 runtime 保留 4 個一般 read workers 和 2 個 prefetch workers。
這個預設值來自多次端到端量測，不只來自這次 microbenchmark。

## 目前端到端量測

本次量測使用一個新 `ModelRuntime`。
expert cache 在開始時是空的。
persistent prompt cache 已停用。
作業系統 page cache 沒有清除。
完整 SHA-256 和 SSD microbenchmark 在本次量測前已執行。

prompt 產生方式如下。

1. 重複字串 ` test`。
2. tokenizer 不加入 special token。
3. 保留前 4,096 個 token。
4. tokenizer 把 token 解碼成 prompt。

generation 使用 batch size 1、`temperature=0`、`top_p=1` 和 64 個 output token。
DSpark 已停用。

| 指標 | 結果 |
| --- | ---: |
| Runtime prompt tokens | 4,096 |
| Generated tokens | 64 |
| Wall time | 30.055 s |
| Time to first token | 24.703 s |
| Prefill | 165.81 Tok/s |
| Decode end-to-end time | 5.348 s |
| Decode | 11.78 Tok/s |
| Decode p50 | 74.30 ms/token |
| Decode p95 | 132.20 ms/token |
| Expert bytes read | 156,688,711,680 |
| Expert read timer | 15.648 s |
| Routing synchronization boundary | 2.367 s |
| Batched expert layers | 42 |
| `gather_qmm` calls | 84 |
| Prefetched layer hits | 41 |
| Expert cache hits / misses | 15,802 / 968 |
| Expert evictions | 0 |
| MLX active memory | 20.67 GiB |
| MLX peak memory | 20.67 GiB |
| Token SHA-256 | `90760d2b...fdd9b9` |

完整 token hash 是
`90760d2bc22d73915dfadf7cfa5b064514d4a56d92cf2868c24614d6ddfdd9b9`。

### 量測限制

- 這是一次端到端量測。這個結果不是 throughput 保證。
- `prefill_tokens_per_second` 使用未快取 token 除以 time to first token。
- expert read timer 和 GPU work 可以重疊。兩者不能直接相加。
- routing synchronization timer 包含產生 router index 之前的 pending graph work。
- expert cache hit rate 不包含 full-layer batched prefill 讀取。
- MLX peak memory 不等於 APP 顯示的 process RSS。

指標的完整語意請見[效能與瓶頸](PERFORMANCE.md)。

## R0 多 prompt baseline

R0 使用五種固定 prompt。
prompt 類型是 repeated、code、繁體中文技術文字、mixed math 和 Tool-like。
每種 prompt 都有 4,096、8,192 和 14,363 token。
每個 run 都建立新的 process。
expert cache 和 prompt cache 在開始時都是空的。
persistent prompt cache 和 DSpark 已停用。
每個 run 使用 greedy generation，並產生 256 個 output token。
作業系統 page cache 沒有清除。

下表是每個長度跨五種 prompt 的中位數。

| Prompt token | TTFT | Prefill | Decode | Expert bytes | Expert read | Misses | Evictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 21.611 s | 189.53 Tok/s | 8.00 Tok/s | 473.970 GB | 28.844 s | 24,700 | 23,548 |
| 8,192 | 39.124 s | 209.39 Tok/s | 7.76 Tok/s | 477.393 GB | 27.116 s | 24,956 | 23,804 |
| 14,363 | 67.466 s | 212.89 Tok/s | 7.43 Tok/s | 482.593 GB | 27.931 s | 25,345 | 24,193 |

| Prompt token | MLX peak | Maximum RSS | Peak footprint |
| ---: | ---: | ---: | ---: |
| 4,096 | 22.969 GiB | 22.897 GiB | 23.297 GiB |
| 8,192 | 23.014 GiB | 22.904 GiB | 23.360 GiB |
| 14,363 | 22.939 GiB | 22.918 GiB | 23.267 GiB |

R0 只對每種 prompt 和長度執行一次。
因此，跨 prompt p95 只表示 workload 差異。
跨 prompt p95 不表示重複執行的變異。

### Repeated prompt 的證據邊界

4K repeated prompt 與四種非 repeated prompt 的 Decode 行為不同。

| Workload | Decode | Expert misses | Evictions | Expert bytes |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 12.22 Tok/s | 1,535 | 383 | 164.269 GB |
| 四種非 repeated prompt 的範圍 | 6.44–8.28 Tok/s | 24,307–28,765 | 23,155–27,613 | 468.716–528.316 GB |

Repeated prompt 的 route 分布也較集中。
Repeated 4K 的每個 Prefill 區塊有 84 個 active expert 中位數。
其他 4K prompt 的中位數是 177 至 202。
因此，repeated prompt 不能代表一般 Decode 的 SSD 與 slot 壓力。

### R1 Metal boundary capture

下表來自獨立的 `Metal System Trace` 診斷 run。
capture 使用 64 個 output token。
capture 的 wall time 不是正式效能結果。

| Workload | Prefill GPU busy | Decode GPU busy | Decode GPU idle |
| --- | ---: | ---: | ---: |
| Repeated 4K | 57.6% | 57.8% | 2.604 s |
| Tool-like 4K | 60.7% | 35.5% | 7.968 s |
| Tool-like 14K | 70.2% | 37.1% | 8.114 s |

Tool-like Decode 的 GPU idle 時間較長。
同一批 capture 也記錄較多 expert misses 和 `preadv` CPU samples。
這個結果支持先測 slot 和 worker 設定。

Metal System Trace 沒有提供 shader timing sample。
目前 capture 不能把 GPU time 分配到 attention、routed expert 和 shared expert。
request 結束時間是依最後一個 Python GPU interval 推定。
CPU sampled running time 是跨 thread 的總和，不是 wall time。

完整設定、prompt hash、output token hash、route profile 和限制位於
[`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)。

## R2a slot 探索性 ABBA

R2a 比較 `slots=1152` 與 `slots=2048`。
其他 runtime 設定保持不變。
每個 run 使用新的 process、4K prompt 和 256 個 output token。

每個 prompt 的順序是 A、B、B、A。
A 是 1,152 slots。
B 是 2,048 slots。

| Prompt | Slots | Decode 中位數 | Expert bytes | Misses | Evictions | Routing sync | MLX peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated 4K | 1,152 | 14.42 Tok/s | 164.269 GB | 1,535 | 383 | 8.906 s | 22.969 GiB |
| Repeated 4K | 2,048 | 13.95 Tok/s | 162.264 GB | 1,385 | 0 | 9.226 s | 25.870 GiB |
| Code 4K | 1,152 | 8.25 Tok/s | 468.716 GB | 24,307 | 23,155 | 9.829 s | 22.969 GiB |
| Code 4K | 2,048 | 2.39 Tok/s | 366.868 GB | 16,689 | 14,641 | 111.488 s | 34.125 GiB |

2,048 slots 讓 repeated Decode 降低 3.31%。
2,048 slots 讓 code Decode 降低 71.09%。
Code 的兩個 2,048-slot run 分別是 3.49 和 1.28 Tok/s。
同一組 ABBA 的兩個 1,152-slot run 是 8.46 和 8.05 Tok/s。

Code 的 expert bytes 降低 21.73%。
Code 的 miss 降低 31.34%。
Code 的 eviction 降低 36.77%。
但是，expert read timer 增加 2.01%。
Routing synchronization boundary 從 9.829 秒增加到 111.488 秒。
這個 boundary 包含 pending upstream graph work。
這個 boundary 不是純 router 時間。

Code 的 MLX peak memory 增加 11.156 GiB。
測試前後的 system swap 都是 361.12 MiB。
八個完整 run 的 output token hash 都在相同 prompt 內一致。

R2a 在 code ABBA 完成後觸發停止條件。
後面三種 prompt 沒有繼續執行。
這不是正式五個 workload、五輪 ABBA 結果。
這個探索性結果足以拒絕 2,048-slot 候選設定。
目前預設值維持 1,152 slots。

完整資料位於
[`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

## R2b 2-worker 探索性 ABBA

R2b 使用 code 4K prompt。
R2b 比較 `read_workers=4` 與 `read_workers=2`。
每個 run 使用 1,152 slots、2 個 prefetch worker 和 256 個 output token。
執行順序是 4、2、2、4 workers。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read | External wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.29 Tok/s | 110.58 ms | 172.12 ms | 28.042 s | 57.96 s |
| 2 | 7.94 Tok/s | 116.54 ms | 182.41 ms | 29.435 s | 59.34 s |

2 workers 讓 Decode 中位數降低 4.20%。
兩個配對差異是 -4.12% 和 -4.27%。
Decode p50 增加 5.39%。
Decode p95 增加 5.98%。
Expert read timer 增加 4.97%。

Expert bytes、miss、eviction 和 token hash 都相同。
Routing synchronization boundary 和 peak memory 沒有明顯差異。
測試前後的 system swap 都是 337.12 MiB。

這是一個 workload、一輪 ABBA 的探索性結果。
這個結果足以拒絕 2-worker 候選設定。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

## R2b 8-worker 探索性 ABBA

R2b 使用相同的 code 4K prompt。
R2b 比較 `read_workers=4` 與 `read_workers=8`。
執行順序是 4、8、8、4 workers。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read | External wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8.27 Tok/s | 110.88 ms | 173.60 ms | 28.202 s | 58.36 s |
| 8 | 8.29 Tok/s | 110.61 ms | 173.13 ms | 27.875 s | 58.19 s |

8 workers 讓 Decode 中位數增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
Expert read timer 降低 1.16%。
這些差異低於 3% 採用門檻。

Expert bytes、miss、eviction 和 token hash 都相同。
Peak memory 沒有明顯差異。
測試前後的 system swap 都是 337.12 MiB。

這是一個 workload、一輪 ABBA 的探索性結果。
8-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

## R2c 4-prefetch-worker 探索性 ABBA

R2c 使用 code 4K prompt 和一個 output token。
R2c 比較 `prefetch_read_workers=2` 與 `prefetch_read_workers=4`。
執行順序是 2、4、4、2 workers。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 21.956 s | 186.56 Tok/s | 14.478 s | 40 |
| 4 | 22.061 s | 185.67 Tok/s | 12.009 s | 41 |

4 workers 讓 expert read timer 降低 17.05%。
但是，4 workers 讓 TTFT 增加 0.48%。
兩個 TTFT 配對改善是 -1.01% 和 0.05%。
結果低於 3% 採用門檻。

Expert bytes、`gather_qmm` call 和 token hash 都相同。
測試前後的 system swap 都是 337.12 MiB。

這是一個 workload、一輪 ABBA 的探索性結果。
4-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

## R2c 1-prefetch-worker 探索性 ABBA

R2c 使用相同的 code 4K prompt 和一個 output token。
R2c 比較 `prefetch_read_workers=2` 與 `prefetch_read_workers=1`。
執行順序是 2、1、1、2 workers。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 22.037 s | 185.87 Tok/s | 14.531 s | 39.5 |
| 1 | 23.740 s | 172.54 Tok/s | 19.394 s | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
兩個 TTFT 配對改善是 -7.68% 和 -7.78%。
Expert read timer 增加 33.46%。
Prefetch hit 中位數從 39.5 降到 0.5。

Expert bytes、`gather_qmm` call 和 token hash 都相同。
測試前後的 system swap 都是 337.12 MiB。

1-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。
R2 設定 sweep 已完成。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

## R3 DSpark 探索性 ABBA

R3 使用 code 4K prompt 和 64 個 output token。
R3 比較 normal generation 與 DSpark。
執行順序是 normal、DSpark、DSpark、normal。
每個 run 都使用新的 process。
每個 run 都停用 persistent prompt cache。

| Mode | Request | TTFT | 總 throughput | 回報的 Decode | Main model expert bytes | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 31.152 s | 22.072 s | 2.054 Tok/s | 6.94 Tok/s | 239.659 GB | 23.274 GiB |
| DSpark | 41.301 s | 33.722 s | 1.549 Tok/s | 8.32 Tok/s | 316.666 GB | 27.257 GiB |

DSpark 讓 request 中位數增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都完成一個 speculative round。
每個 round 提議 5 個 token，接受 5 個 token，並提交 6 個 token。
兩個 run 的 acceptance 都是 100%。
兩個 run 之後都觸發 fallback。

回報的 Decode 從 6.94 增加到 8.32 Tok/s。
這個指標只涵蓋第一個 token 之後的 boundary。
這個指標不包含 DSpark 增加的 TTFT。
DSpark 也在一個 round 後停止 speculative decoding。
因此，這個 Decode 指標不能推翻 request 與總 throughput 的結果。

Normal run 記錄 42 個 batched expert layer 和 84 次 `gather_qmm`。
DSpark run 的兩個欄位都是 0。
DSpark 的 main model expert bytes 增加 32.13%。
DSpark 的 main model cache miss 增加 230.16%。

四個 output token hash 都相同。
測試前後的 system swap 都是 337.12 MiB。
System compressor pages 從 82,443 增加到 114,791。
作業系統 page cache 沒有清除。

R3 達到停止條件。
研究沒有擴大到五個 prompt 和 256 個 output token。
這個結果足以拒絕目前 DSpark 預設啟用。
DSpark 預設值維持停用。

完整資料位於
[`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

## M1 full-model smoke test

本節到正式結論記錄已移除 prototype 的歷史測試。
目前 runtime 不提供 M1 開關。

M1 smoke test 使用 code 4K prompt 和一個 output token。
當時的 runtime 啟用 `--prefill-no-file-cache`。
Persistent prompt cache 和 DSpark 都停用。

| 指標 | 結果 |
| --- | ---: |
| TTFT | 20.412 s |
| Prefill | 200.66 Tok/s |
| Expert bytes | 149.991 GB |
| Expert read | 11.574 s |
| Batched expert layers | 42 |
| `gather_qmm` calls | 84 |
| Prefetched layer hits | 41 |
| Peak footprint | 20.352 GiB |

Output token hash 是
`2a7185fd600ecd7bd4b998bc7dbf460bd0dd2f1b9529ebdf0f662a48be746642`。
這個 hash 與既有 code 4K one-token control 相同。

這次 smoke test 只驗證 full-model 路徑可以執行。
這次 smoke test 沒有同輪 control。
這次 smoke test 不是效能結果。
後續探索性 ABBA 位於下一節。

## M1 Prefill file-cache 探索性 ABBA

M1 使用 code 4K prompt 和 256 個 output token。
M1 比較預設 file descriptor 與 Prefill 專用 `F_NOCACHE` file descriptor。
執行順序是 Control、M1、M1、Control。
每個 run 都使用新的 process。
Persistent prompt cache 和 DSpark 都停用。
作業系統 page cache 沒有清除。

| Mode | Request | TTFT | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 52.508 s | 21.884 s | 187.17 Tok/s | 8.33 Tok/s | 27.945 s | 23.296 GiB |
| M1 | 50.239 s | 20.221 s | 202.56 Tok/s | 8.50 Tok/s | 24.542 s | 23.297 GiB |

M1 讓 request 中位數改善 4.32%。
M1 讓 TTFT 改善 7.60%。
M1 讓 Prefill throughput 增加 8.22%。
M1 讓 Decode throughput 增加 2.02%。
M1 讓 expert read timer 降低 12.18%。

兩個 request 配對分別改善 4.18% 和 4.46%。
兩個 TTFT 配對分別改善 7.27% 和 7.93%。
四個 output token hash 都相同。
Expert bytes、cache hit、miss 和 eviction 都相同。

M1 的 process peak footprint 增加 1.77 MiB。
這個差異是 0.0074%，可視為沒有變化。
MLX peak memory 也沒有實質變化。
測試沒有出現可重現的 memory pressure。
測試前後的 system swap 都是 337.12 MiB。

M1 沒有達到預先定義的記憶體接受條件。
M1 的記憶體假設尚未成立。
專案不把 M1 採用為記憶體最佳化。
這個 pilot 當時沒有改變 runtime 預設值。

System pagein 計數的配對中位數降低 61.88%。
這個計數包含其他 process，也受到執行順序影響。
這個計數不等於 expert bytes，也不能證明 process 記憶體降低。

外部 wall time 的第一個配對增加 0.83%。
外部 wall time 的第二個配對改善 4.03%。
本次測試只有一種 workload 和一輪 ABBA。
本次測試沒有 bootstrap 信賴區間。
因此，本次測試不能支持正式效能採用。

兩個 runtime request 配對都有超過 3% 的同向改善。
專案當時暫時保留 opt-in prototype。
後續 Tool-like 4K 獨立效能重跑位於下一節。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)。

## M1 Tool-like 4K 獨立探索性 ABBA

這次重跑使用 Tool-like 4K prompt 和 256 個 output token。
執行順序是 Control、M1、M1、Control。
其他設定與 code 4K pilot 相同。

| Mode | Request | TTFT | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 62.574 s | 22.860 s | 179.33 Tok/s | 6.42 Tok/s | 36.282 s | 23.296 GiB |
| M1 | 60.380 s | 20.966 s | 195.38 Tok/s | 6.47 Tok/s | 33.173 s | 23.297 GiB |

M1 讓 request 中位數改善 3.51%。
M1 讓 TTFT 改善 8.29%。
M1 讓 Prefill throughput 增加 8.95%。
M1 讓 Decode throughput 增加 0.75%。
M1 讓 expert read timer 降低 8.57%。

兩個 request 配對分別改善 2.34% 和 4.63%。
兩個 TTFT 配對分別改善 6.31% 和 10.15%。
外部 wall time 中位數改善 3.15%。
四個 output token hash 都相同。
Expert bytes、cache hit、miss 和 eviction 都相同。

M1 的 process peak footprint 增加 0.98 MiB。
MLX peak memory 沒有實質變化。
System swap 在 B2 後增加 2.13 MiB。
System swap 在 A2 後維持相同數值。
System swap 是 system-wide 指標。
本次測試不能把這個變化歸因於 M1。

A1 的 CLI 已完成，並保存完整 metrics 和 time output。
量測包裝指令在 CLI 結束後使用 zsh 保留變數，因此回傳錯誤。
A1 的 runtime 和 process 指標有效。
A1 的 post-run system snapshot 比其他 run 晚。
System pagein 比較只保留為診斷資料。

Code 與 Tool-like pilot 的 request 中位數都改善超過 3%。
兩種 workload 的四個 request 配對都同向改善。
這個結果確認 pilot 等級的效能訊號。

兩種 workload 都只有一輪 ABBA。
這兩個 pilot 當時沒有 bootstrap 信賴區間。
專案因此進入正式多 workload 重複 ABBA。
正式測試監控 swap，並套用既有停止條件。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

## M1 正式效能評估：Wave 1

Wave 1 使用全部五個 R0 4K prompt。
Prompt 順序是 Repeated、Code、繁體中文技術文字、Mixed math、Tool-like。
每個 prompt 先執行一個 Control warm-up。
五個 warm-up 都不納入統計。
每個 prompt 接著執行一輪 Control、M1、M1、Control。
Wave 1 共有 20 個有效量測 run。

第一個 runner 在 Repeated B2 後發生 `awk` 比較錯誤。
Runner 在第一個 Repeated A2 產生 93 個 token 時停止。
專案移出第一次 Repeated 序列的完整檔案和部分 A2 檔案。
正式序列接著從新的 Repeated warm-up 完整重跑。
下表和 artifact 都不包含第一次序列。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 3.03% | 8.23% | 8.98% | -2.93% | 21.91% | 0.72 MiB |
| Code | 4.54% | 8.55% | 9.34% | 1.73% | 12.34% | 4.01 MiB |
| 繁體中文技術文字 | 3.40% | 8.12% | 8.84% | 0.13% | 10.74% | 0.53 MiB |
| Mixed math | 3.69% | 8.24% | 8.98% | 0.57% | 10.85% | 0.20 MiB |
| Tool-like | 2.70% | 7.97% | 8.66% | -0.32% | 8.24% | 0.05 MiB |

四個主要 workload 的 request 改善中位數是 3.54%。
TTFT 改善中位數是 8.18%。
Prefill throughput 增加中位數是 8.91%。
Decode throughput 增加中位數是 0.35%。
Expert read timer 改善中位數是 10.80%。
外部 wall time 改善中位數是 3.26%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
每個 workload 的五個 output token hash 都相同。
Expert bytes、cache hit、miss 和 eviction 在同一 workload 內都相同。

四個主要 workload 的 process peak footprint 增加中位數是 0.36 MiB。
最大的 workload 差異是 Code 增加 4.01 MiB。
這個差異是 0.017%。
MLX peak memory 沒有實質變化。
正式序列的 system swap 從 291.25 MiB 降到 283.25 MiB。
Compressor 使用量從 106,138 pages 降到 102,407 pages。
測試沒有出現 thermal 或 performance warning。

Repeated Decode 在 Wave 1 降低 2.93%。
這個結果高於 2% regression 門檻。
但是，Wave 1 只有兩個 Repeated candidate run。
目前沒有五輪 p95 或 95% bootstrap 信賴區間。
專案必須在後續 wave 繼續檢查這個 regression。

Wave 1 單獨不能支持正式採用決定。
Wave 1 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

## M1 正式效能評估：Wave 2

Wave 2 輪替 prompt 順序。
順序是 Code、繁體中文技術文字、Mixed math、Tool-like、Repeated。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
Wave 2 共有 20 個有效量測 run。

第一次 Code warm-up 完成後，runner 的 Swap parser 使用了 macOS `awk` 不接受的變數名稱。
這個錯誤發生在任何正式量測 run 開始前。
專案保留並排除第一次 warm-up。
Parser 修正並通過相等值與增加值檢查後，完整 Wave 2 從新的 Code warm-up 重跑。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 7.63% | 11.05% | 12.48% | 3.64% | 23.24% | 1.60 MiB |
| Code | 3.78% | 7.83% | 8.49% | 0.93% | 11.34% | -0.72 MiB |
| 繁體中文技術文字 | 3.66% | 8.26% | 9.01% | 0.47% | 10.72% | -1.01 MiB |
| Mixed math | 3.57% | 8.51% | 9.30% | 0.17% | 10.69% | 0.01 MiB |
| Tool-like | 2.61% | 7.77% | 8.43% | -0.36% | 8.04% | 0.63 MiB |

四個主要 workload 的 request 改善中位數是 3.62%。
TTFT 改善中位數是 8.05%。
Prefill throughput 增加中位數是 8.75%。
Decode throughput 增加中位數是 0.32%。
Expert read timer 改善中位數是 10.70%。
外部 wall time 改善中位數是 3.26%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
Runtime Python tree SHA-256 也與 Wave 1 相同。

四個主要 workload 的 process peak footprint 差異中位數是 -0.36 MiB。
最大的 workload 增加是 Repeated 的 1.60 MiB。
這些 process 差異沒有實質變化。
正式序列的 system swap 從 283.25 MiB 降到 275.25 MiB。
System-wide compressor 增加 209.67 MiB。
這個增加主要發生在 warm-up 與 Control 區段。
本次測試不能把 system-wide compressor 變化歸因於 M1。
測試沒有出現 thermal 或 performance warning。

Wave 1 的 Repeated Decode regression 沒有在 Wave 2 重現。
Repeated Decode 在 Wave 2 增加 3.64%。

## M1 正式效能評估：Wave 3

Wave 3 再次輪替 prompt 順序。
順序是繁體中文技術文字、Mixed math、Tool-like、Repeated、Code。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
Wave 3 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 7.93% | 11.87% | 13.45% | 3.25% | 23.77% | -11.73 MiB |
| Code | 4.08% | 7.71% | 8.35% | 1.53% | 11.53% | 0.04 MiB |
| 繁體中文技術文字 | 4.28% | 8.68% | 9.50% | 1.22% | 11.82% | 0.24 MiB |
| Mixed math | 3.34% | 6.97% | 7.52% | 0.84% | 10.72% | -1.48 MiB |
| Tool-like | 2.07% | 8.48% | 9.17% | -1.64% | 8.23% | 0.99 MiB |

四個主要 workload 的 request 改善中位數是 3.71%。
TTFT 改善中位數是 8.09%。
Prefill throughput 增加中位數是 8.76%。
Decode throughput 增加中位數是 1.03%。
Expert read timer 改善中位數是 11.12%。
外部 wall time 改善中位數是 3.35%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
Runtime Python tree SHA-256 也與 Wave 1 相同。

四個主要 workload 的 process peak footprint 增加中位數是 0.14 MiB。
最大的 workload 增加是 Tool-like 的 0.99 MiB。
這些 process 差異沒有實質變化。
正式序列的 system swap 維持 275.25 MiB。
System-wide compressor 增加 244.77 MiB。
本次測試不能把 system-wide compressor 變化歸因於 M1。
測試沒有出現 thermal 或 performance warning。

Tool-like Decode 在 Wave 3 降低 1.64%。
這個差異低於 2% regression 門檻。

## M1 正式效能評估：Wave 4

Wave 4 再次輪替 prompt 順序。
順序是 Mixed math、Tool-like、Repeated、Code、繁體中文技術文字。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
Wave 4 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 7.76% | 10.82% | 12.18% | 4.34% | 23.05% | -2.16 MiB |
| Code | 3.31% | 5.81% | 6.17% | 1.53% | 10.36% | 10.54 MiB |
| 繁體中文技術文字 | 3.56% | 7.95% | 8.63% | 0.48% | 11.09% | -2.28 MiB |
| Mixed math | 3.17% | 7.75% | 8.42% | 0.01% | 10.65% | 13.66 MiB |
| Tool-like | 3.32% | 7.90% | 8.56% | 0.63% | 9.02% | -0.45 MiB |

四個主要 workload 的 request 改善中位數是 3.32%。
TTFT 改善中位數是 7.82%。
Prefill throughput 增加中位數是 8.49%。
Decode throughput 增加中位數是 0.56%。
Expert read timer 改善中位數是 10.51%。
外部 wall time 改善中位數是 2.90%。

八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
Runtime Python tree SHA-256 也與 Wave 1 相同。

四個主要 workload 的 process peak footprint 增加中位數是 5.05 MiB。
最大的 workload 增加是 Mixed math 的 13.66 MiB。
這些 process 差異沒有實質變化。
正式序列的 system swap 維持 275.25 MiB。
System-wide compressor 增加 340.41 MiB。
本次測試不能把 system-wide compressor 變化歸因於 M1。
測試沒有出現 thermal 或 performance warning。

## M1 正式效能評估：Wave 5

Wave 5 的 prompt 順序是 Tool-like、Repeated、Code、繁體中文技術文字、Mixed math。
每個 prompt 仍使用一個不納入統計的 Control warm-up 和一輪 ABBA。
有效序列共有 20 個量測 run。

第一次序列在 Tool-like B1 啟動時失去執行 session。
Tool-like A1 已完成，但 B1 沒有完成，也沒有寫入 metrics。
專案保存並排除整個不完整序列。
專案接著從新的 Tool-like warm-up 完整重跑 Wave 5。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput | Expert read 改善 | Peak footprint 增加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Repeated | 8.36% | 12.44% | 14.24% | 3.89% | 23.84% | -0.34 MiB |
| Code | 2.85% | 6.26% | 6.68% | 0.51% | 10.98% | 17.00 MiB |
| 繁體中文技術文字 | 3.44% | 9.36% | 10.32% | -0.69% | 10.85% | -9.09 MiB |
| Mixed math | 3.68% | 8.83% | 9.66% | 0.14% | 10.88% | -6.55 MiB |
| Tool-like | 2.36% | 6.32% | 6.82% | 0.10% | 9.22% | -3.77 MiB |

四個主要 workload 的 request 改善中位數是 3.15%。
TTFT 改善中位數是 7.58%。
Prefill throughput 增加中位數是 8.24%。
Decode throughput 增加中位數是 0.12%。
八個主要 workload request 配對都改善。
包含 Repeated 後，十個 request 配對都改善。
所有 output token hash 都與前四輪相同。

四個主要 workload 的 process peak footprint 差異中位數是降低 5.16 MiB。
最大的 workload 增加是 Code 的 17.00 MiB。
正式序列的 system swap 降低 8.56 MiB。
System-wide compressor 增加 1,236.50 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)。

### 五輪正式結論

五輪共有 100 個有效量測 run 和 50 個配對。
四十個主要 workload request 配對都改善。
包含 Repeated 後，五十個 request 配對都改善。
所有正確性與執行停止條件都通過。

四個主要 workload 的正式結果如下。

| 指標 | 結果 |
| --- | ---: |
| Request 改善中位數 | 3.53% |
| Request 95% bootstrap 信賴區間 | 3.27% 至 3.76% |
| TTFT 改善中位數 | 7.99% |
| Prefill throughput 增加中位數 | 8.69% |
| Decode throughput 增加中位數 | 0.23% |
| Expert read 改善中位數 | 10.88% |
| 外部 wall time 改善中位數 | 3.25% |

Request 中位數與信賴區間通過效能條件。
主要 workload 的 Decode p95 regression 最大值是 1.71%。
這個結果通過 2% 條件。

Repeated 是 regression control。
Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。

Process peak footprint 的配對中位數增加 0.32 MiB。
配對 p95 增加 19.24 MiB。
配對最大值增加 21.67 MiB。
這些結果通過 256 MiB 條件。
每一輪的 system swap 都沒有增加。

部分 wave 的 system-wide compressor 總量增加。
Snapshot 包含 warm-up、Control、M1 和其他 process。
因此，專案不能把增加歸因於 M1。
但是，這組證據沒有證明 compressor 不增加。

M1 沒有通過全部正式採用條件。
專案拒絕預設啟用 M1。
專案依研究計畫移除 M1 prototype。
目前 Prefill 與 Decode 共用一般 file descriptor。

完整正式結果位於
[`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

## 歷史 full-model 結果

以下結果來自舊版本的本機紀錄。
以下結果沒有在本次文件整理中全部重跑。

### Release 摘要

| Runtime | 測試 | Prefill | Decode | 記錄的峰值記憶體 |
| --- | --- | ---: | ---: | ---: |
| v1.0.3 | 14,000-token Codex request | 180 Tok/s | 6.5 Tok/s | 30 GB |
| v1.0.2 | 4,096-token prompt，1 output | 144.53 Tok/s | — | 15.56 GiB |
| v1.0.2 | 短 prompt，同 runtime 第二次 | — | 6.41 Tok/s | 15.05 GiB |

這些 row 使用不同 prompt、output 長度和 cache 狀態。
這些 row 不能互相比較。
repo 沒有保存三個 release row 的完整 raw artifact。

### Prefill 演進

早期 32-token chunk 測試使用 deterministic repeated prompt。

| Context | KV cache | Time | Expert bytes | 結果 |
| ---: | --- | ---: | ---: | --- |
| 4,096 | BF16 | 124.0 s | 572.9 GB | greedy token 是 ` test` |
| 8,192 | MXFP8 | 264.1 s | 1,340.2 GB | greedy token 是 ` test` |
| 8,192 | BF16 | 264.6 s | 1,377.7 GB | greedy token 是 ` test` |

128-token chunk 和 grouped routed expert path 後的結果如下。

| Context | KV cache | Time | Expert bytes | 結果 |
| ---: | --- | ---: | ---: | --- |
| 4,096 | BF16 | 30.74 s | 219.1 GB | 與早期 greedy token 相同 |
| 8,192 | MXFP8 | 65.47 s | 488.8 GB | 與早期 greedy token 相同 |

14,363-token Tool-like prompt 的歷史 path 比較如下。

| Path | Time | Expert bytes |
| --- | ---: | ---: |
| Chunk-major，512-token step | 386.65 s | 2,290.6 GB |
| Layer-major，512-token step | 105.81 s | 131.5 GB |
| Batched layer-local | 76.54 s | 150.41 GB |

layer-major path 相對 chunk-major path 減少約 94.3% expert bytes。
batched layer-local path 再減少大量個別 matrix operation。
這些數字來自同類 Tool-like prompt，但 repo 沒有保存完整 prompt artifact。

### Prompt cache

一組同 process 的 4K continuation 重用 4,097 / 4,098 prompt token。
time to first token 從 20.89 秒降到 0.56 秒。

persistent prompt cache v2 的短 prompt artifact 是 6.46 MB。
寫入時間是 4.4 ms。
round-trip 測試還原 62 個 quantized pooling cache item。

目前自動測試仍覆蓋 persistent cache restart 和 quantized round-trip。

### Ready expert decode

五組 4,096-token prompt 和 256-token output 使用配對 A/B。
prompt 類型是 repeated text、code、繁體中文技術文字、English prose 和 mixed math。

| 結果 | 歷史量測 |
| --- | ---: |
| Median Decode Tok/s 改善 | 12.9% |
| Minimum 改善 | 8.1% |
| Maximum 改善 | 14.0% |
| Aggregate decode-time 降低 | 10.5% |
| 相同 256-token hash | 5 / 5 |

2,000-token stability pair 的 baseline 是 10.67 Tok/s。
ready expert decode 是 11.92 Tok/s。
兩個 run 的完整 token hash 相同。

原始本機 artifact 位於被忽略的 `scratch/expert-streaming-experiment`。
這些數字不是 release throughput 保證。

### DSpark

2026-08-09 的舊 runtime 有三組配對量測。
normal decode 中位數是 5.39 Tok/s。
DSpark 中位數是 5.14 Tok/s。
DSpark 在該測試中慢約 4.6%。

該測試早於目前的 round-level cache fork 和 block verification。
該測試不能代表目前 DSpark 速度。

較新的 16-token smoke pair 產生相同 token hash。
DSpark 第一輪接受 0 個 draft token，然後正確 fallback。

目前 checkout 的 R3 探索性 ABBA 使用 code 4K prompt。
R3 的 request 增加 32.58%。
R3 的總 throughput 降低 24.60%。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
因此，runtime 繼續預設停用 DSpark。

## 驗證矩陣

### Qwen 第一版驗證結果

2026-08-27 已執行官方 FP8 header inspection。
inspection 沒有下載 tensor payload。
結果為 48 層、每層 512 個 routed expert、每個 expert blob 2,611,200 bytes，
181,906,343,706 checkpoint tensor payload bytes，
以及 125,268,506,112 installed weight bytes。

Swift 輕量測試已覆蓋 format 2 planner、vision/MTP 排除、MXFP4 固定向量、
Qwen expert 轉換、installed artifact file 續傳與損壞 layer repair。Python 輕量測試已覆蓋 QSA、N-gram、PLE 資料路徑、
Qwen expert layout、layer-major prefill 和 XML tool parser。

2026-08-27 Qwen 第一版工作樹的自動測試結果如下：

| Suite | 通過 | 失敗 | 略過 |
| --- | ---: | ---: | ---: |
| Swift `DeepSeekRepackTests` | 16 | 0 | 0 |
| Swift `DeepSeekV4SSDAppTests` | 39 | 0 | 3 |
| Python runtime 與 server | 113 | 0 | 0 |
| 合計 | 168 | 0 | 3 |

三個 Swift App 測試需要現有的完整 DeepSeek installed model。
使用者已把該模型移到外接硬碟。
本次驗證沒有存取外接硬碟，並略過這三項。

Swift App 測試也確認空的模型庫仍提供 DeepSeek 與 Qwen 選項。
Swift App 測試確認模型下載會比較所需空間與可用空間。
Swift App 測試確認 First Token wait time 會在 Prefill 期間即時更新。
效能歷史只記錄每個 request 的最終 First Token wait time。
Python 測試使用 fake API server 驗證 API benchmark 的精確 input token、
指標收集、Peak、P95 和 ASCII 表格。
Python 測試也驗證 installed model discovery、Server 自動啟動和 Server 清理。
Python 測試確認 APP 的 `/api/status` polling 不會寫入 access log。
本機 Qwen installed model 已通過 Server 自動啟動、API ready 和自動清理檢查。
Swift repack 測試確認 direct installed artifact file 可以從現有 file size 續傳，
並在完成時驗證 SHA-256。

Whallm 更名工作樹已執行 `make package`。
流程建立 `dist/Whallm.app` 和 `dist/Whallm-macOS-arm64.zip`。
App 與解壓後的 App 都通過簽章、三種 localization 和隔離啟動檢查。

Published installed model 位於
[`Yanun/Qwen3.8-Flash-Next-MXFP4`](https://huggingface.co/Yanun/Qwen3.8-Flash-Next-MXFP4)。
程式碼固定 commit `753d0aa57059fad70a5f7e6cc249f25df56bbd34`。
Hugging Face API 列出 61 個 repository files。
遠端 `manifest.json` 的 SHA-256 是
`3f4cb52a88335591cfb8233778eb56396e9d2756c153485e4dcfb0a12daed0ce`。
這個 SHA-256 與本機 `manifest.json` 相同。
遠端 `common.bin` 的 1 MiB range 回傳 HTTP 206 和 1,048,576 bytes。
遠端 range 的 SHA-256 與本機相同。
遠端 `config.json` 的 SHA-256 也與本機相同。
本次驗證沒有重新執行完整 125 GB 的 App direct download。

Qwen installed model 位於內建 SSD。
完整 SHA-256 驗證已通過 57 個 manifest files。
manifest files 合計是 125,291,490,955 bytes。
common tensor 共 1,069 個。

完整模型 API 驗證結果如下：

| 項目 | 結果 |
| --- | --- |
| thinking 關閉 | 回覆 `OK`，`stop`。 |
| thinking 開啟 | reasoning 與答案 `4` 正確，`stop`。 |
| forced tool call | `get_weather`，`{"city":"Taipei"}`。 |
| streaming tool call | function 名稱與 arguments 和完整 response 相同。 |
| packaged App | App、ZIP、三種 localization、隔離啟動和 runtime import 通過。 |

CLI 也使用預設 `memory_limit_gib=0` 執行完整模型。
runtime 套用 48 GiB 自動上限。
5-token prompt 產生 ` Paris`。
該 request 從 SSD 讀取 4,658,380,800 bytes 的 routed expert。
MLX peak memory 是 13,048,438,112 bytes。
request 結束時的 active memory 是 13,040,928,562 bytes。
修正前的 MLX peak memory 是 74,069,529,060 bytes。
修正前後的 output token SHA-256 相同。

4K 使用 4,096-token repeated prompt、greedy generation 和 1 個 output token。
runtime memory limit 是 48 GiB。

| Cache state | 重用 prompt tokens | Wall time | Expert bytes read | Output token SHA-256 |
| --- | ---: | ---: | ---: | --- |
| Cold | 0 | 65.42 s | 66,081,638,400 | `6dfb9763…b3b2bd1` |
| Warm | 4,095 | 0.214 s | 0 | `6dfb9763…b3b2bd1` |

兩次 output token ID 都是 `1228`。
兩次完整 SHA-256 都是
`6dfb97632210ac38a071667cf8be7df83a16178e12f1248e45b2a3d24b3b2bd1`。
這些時間是一個本機驗證結果，不是效能保證。
完整 artifact 位於
[`benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json`](benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json)。

| 能力 | 狀態 | 證據邊界 |
| --- | --- | --- |
| 固定 checkpoint 合約 | 通過 | 程式檢查、測試和完整 manifest 驗證。 |
| Installed model SHA-256 | 通過 | Qwen 目前 57 個 manifest files。 |
| Repack 續傳與 repair | 通過 | Fixture 自動測試。 |
| MXFP4 routed expert | 通過 | Dequantized reference 和 batched parity 單元測試。 |
| Layer-major prefill parity | 通過 | Fixture next-token logits 和歷史 full-model token。 |
| MXFP8 cache | 通過 | 單元測試、8K 歷史 greedy token。 |
| Persistent prompt cache | 通過 | Restart 和 quantized round-trip 測試。 |
| Ready expert decode | 通過 | 五組歷史 hash 和目前單元測試。 |
| API、tool、SSE | 通過 | Python server tests，包括無效 tool call 的 Codex 終止 event。 |
| DSpark 邏輯 | 通過 | Fixture greedy、sampling、replay 和 fallback tests。 |
| DSpark 淨加速 | 未通過 | 目前 checkout 的 R3 探索性 ABBA 已觸發停止條件。 |
| 14K context | 歷史通過 | Tool-like prompt，greedy，1 output token。 |
| 1M context | 未驗證 | checkpoint 合約值不等於本機證據。 |
| Full-model sampling parity | 未驗證 | 目前沒有固定 reference artifact。 |
| Model quality benchmark | 未執行 | 本專案驗證 runtime，不宣稱品質評分。 |

## 重跑規則

每個正式 performance row 必須保存：

1. commit、dependency、checkpoint revision、硬體和 OS。
2. prompt 產生方式、input token 和 output token。
3. batch、temperature、top-p 和停止條件。
4. slot、worker、prefill、KV cache、prompt cache 和 DSpark 設定。
5. fresh/warm compile、OS page cache 和 expert cache 狀態。
6. wall time、TTFT、decode time、SSD bytes、read timer 和 memory 指標。
7. 完整 output token hash。

設定比較必須使用相同 output token 數。
greedy 比較必須使用相同 token hash。
正式 A/B 應至少使用五個 prompt，並交錯執行順序。

## 重跑 command

記錄版本：

```sh
git rev-parse HEAD
sw_vers
.venv/bin/python --version
.venv/bin/python -m pip show mlx mlx-lm transformers
```

驗證 installed model：

```sh
swift run dsv4-repack verify --model /path/to/model.dsv4
```

量測 expert blob I/O：

```sh
swift run dsv4-repack benchmark \
  --model /path/to/model.dsv4 \
  --samples 32
```

CLI 可以使用 `--metrics-json` 保存 runtime 指標。
CLI 也會保存完整 prompt token ID 的 `prompt_token_sha256`。
CLI 也會保存完整 output token ID 的 `token_sha256`。
hash 輸入是以逗號連接的十進位 token ID。
`Scripts/prepare_r0_prompts.py` 可以建立 4K、8K 和 14K 固定 prompt 檔。
工具會把 prompt hash 和重建規則寫入 manifest。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file fixed-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --top-p 1 \
  --no-persistent-prompt-cache \
  --metrics-json result.json
```

不要只保存 `tokens_per_second`。
請同時保存 prompt token、`prompt_token_sha256`、generated token、
`token_sha256` 和 runtime metrics。
