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
- Hash exact prefetch 與 storage-aware adaptive block 已有預設關閉 prototype；adaptive 原始 score policy 已被拒絕，0.90 高信心護欄修正版已與 hybrid exact verifier 完成五組 correctness／logical-byte composition gate，但尚未通過正式 performance gate。
- Research-only expert-file page-cache probe 已通過 full-model contract 與 hash-prefetch useful/wasted composition gate；4K `balanced_choice` 的 logical／nonresident useful rate 分別是 23.53%／29.65%。它是 default-off `mincore` proxy，不是 physical SSD counter。
- Sequential oracle 與逐層／component 診斷定位 block-shape drift；default-off hybrid v3 將 target math 維持 one-token shape、每層只 acquire 一次 expert union，五組 4K normal／fixed token parity 全部通過。它沒有保留 grouped multi-row QMM，也沒有 speed adoption evidence。
- Pinned checkpoint 的 MTP-1 contract audit 已完成：official inference graph 需要三個
  `mtp.*` stages，沒有 self-contained one-stage checkpoint path；因此拒絕把 runtime
  裁層標成 faithful native MTP-1。
- DSpark state ownership 已盤點；96-slot independent-cache candidate 在三組 4K／32
  workload 保持 exact tokens 且降低 peak memory，但 4K／128 長解碼讓 draft expert
  bytes/committed token 增加 83.17%，超過預先宣告的 +50% 停止線。候選已停止，預設
  仍是 768。
- Default-off atomic DSpark prompt-context snapshot 已實作；128-token gate 的 memory／
  restart-persistent hits 都 exact 重用 127 tokens，combined logical expert bytes 分別
  -92.23%／-65.59%。這是 functional gate，不是採用級效能結果。
- Native MTLIO 1／6／32／128 expert gate 的 CPU bytes、shared/private buffers、
  shared-event GPU visibility 與 cancellation admission 全部 byte-exact；但 MLX 0.32.0
  公開介面沒有 external `MTLSharedEvent` dependency handoff，runtime integration 已停止。
- Fixed-arena staged `w13`／`w2` overlap gate 通過 optimistic 4／8-row continuation；
  default-off split-slot runtime 的四波 128／32 exact，但 request +2.17%、Decode -4.26%，
  因此 candidate 已停止且不暴露為 runtime 介面。
- Adaptive full-layer／selective prefill 的五類 4K route gate 讓 70%／80%／90% 全部
  byte-eligible；fixed-buffer runtime 也保持 token／selected-row／byte exact，卻在
  `repeated` 4K 讓 TTFT paired median +16.87%、p95 +17.02%。三 threshold 的該
  workload decisions 相同，因此 candidate 已停止，full-layer 預設不變。
- DSpark coherent Markov beam 已完成 greedy-only first-round gate。五組 baseline 都被
  target 接受 5/5；storage weights 在 `code`／`tool_like` 選到較小 hash union 時，
  acceptance 分別降到 4／1。0/5 workloads 通過 continuation gate，因此 branch-4／
  beam-8 candidate 已停止，sampling 仍禁止。
- Learned-router fixed-label baseline 已完成 6,000 個 layers 3--42 assignments。四個
  DSpark feature taps 的最佳 top-24 direct transfer 只有 17.15% assignment recall、
  33.03% union recall 與 11.15% useful rate；0/40 layers 達 75%。Direct frozen-router
  transfer 已停止，後續需要明確的 dataset／training prerequisite。

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
除 2026-08-26 新增的 storage-aware DSpark prototypes 外，目前研究計畫沒有其他已
通過 profile gate 的候選項目。這兩個 prototype 尚未通過正式 performance gate。

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
| HYB | `--dspark --dspark-hybrid-verification` | V1／V2 分別在多 round 與 2/5 workloads 失敗；V3 的 token-shaped target math 加每層一次 expert union 在五組 4K／32 normal／fixed 全部 exact | 保留預設關閉 correctness candidate；尚未證明 grouped-QMM 或淨加速。 |
| HXP | `--dspark --dspark-hash-prefetch` | 與 hybrid v3 組合後，五組 4K normal／fixed 全部 exact，logical `useful + wasted` 對帳；useful rate 23.53%–56.23% | 保留預設關閉 prototype；仍缺 explicit-cache-state repeated throughput／memory gate。已有 expert-file pre-read residency proxy；true physical device bytes 仍需更低層 tracer。 |
| PCR | `--expert-page-cache-probe` | `code` 128/32 ABBA exact 且 90.323 GB/run 全分類；4K `balanced_choice` useful／wasted residency partitions exact，1.618 GB wasted bytes 為 pre-read nonresident | 保留 research-only default-off proxy；不可稱為 physical SSD bytes。下一步需 explicit cache-state repeated wave。 |
| ABS | `--dspark --dspark-adaptive-block` | Hybrid v3 + hash 五組 normal／fixed／adaptive 全部 exact；63 decisions 中 56 次選 1 token，prefetch useful rate 65.22%–91.31%，union reuse 降到 15.59%–23.98% | 拒絕原始 score policy；保留修正版預設關閉。Tradeoff 已量到，尚未採用速度、memory 或 power 結果。 |
| PATH | 無；research script only | Branch-4／beam-8 coherent Markov lattice 在五組 128-token fresh workers 通過 actual-draft reconstruction、sequential target truth 與 exact hash-route gates；五組 baseline 都接受 5/5。`code`／`tool_like` 的 storage-selected alternatives 少 17／16 requested blobs，但 acceptance 降到 4／1 | 0/5 通過 acceptance-preserving storage gate；停止目前 candidate，不建立 runtime／CLI／server／APP 開關。Sampling proposal 尚未證明。 |
| RPRED | 無；research script only | 五組 first rounds 擷取 6,000 個 learned-router labels；四個 DSpark taps × top-6／12／24／48 的 assignment 與 union bytes 全閉合。最佳 eligible-size final-norm top-24 僅 17.15% assignment／33.03% union recall、11.15% useful rate、0/40 layers 達 75% | 停止 direct frozen-router transfer；不建立 scratch prefetch、probation 或 deadline pinning。重啟需要新 data-rights／collection protocol、train/validation/held-out split 與 immutable trained predictor artifact。 |
| MTP1 | 無 | Pinned config／reference graph／checkpoint index／installed payload audit：`num_nextn_predict_layers=1` 不由 inference graph 使用；`mtp.0` 只有 input adapter、`mtp.2` 只有 output heads，三層都不是 self-contained | 拒絕 runtime stage truncation。只有新的一層 checkpoint 或另行訓練／蒸餾，才能重啟此 baseline。 |
| DSC96 | `--dspark-slots 96` | 三組 4K／32 exact 且 peak memory -0.40% 至 -3.31%；4K／128 normal／768／96 exact、peak -7.35%、all speculative bytes/committed +0.91%，但 draft bytes/committed +83.17% | 未通過預先宣告的 +50% draft-read gate；停止 96-slot candidate，保留 768-slot 預設。 |
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

## Native MTP-1 checkpoint contract audit

2026-08-27 使用完整 installed model、pinned official configs、reference graph 與
checkpoint index 稽核 PLAN 的 minimum-cost MTP-1 baseline。Transformers metadata 是
`num_nextn_predict_layers=1`，但 official inference contract 是 `n_mtp_layers=3`、
block size 5；reference graph 不讀取前者，而是從 `mtp.0` 進入、執行全部三層、
由 `mtp.2` 輸出。

Checkpoint index 有 4,705 個 `mtp.*` tensors，stage 0／1／2 分別有
1,568／1,565／1,572 個。`mtp.0` 的 target-hidden input adapter 完整但沒有 output
heads；`mtp.2` 的 output heads 完整但沒有 input adapter。沒有 self-contained stage。
Installed tensor set 與 checkpoint index exact，10,862,841,600-byte DSpark payload 的
四個檔案也全部通過 manifest SHA-256。

因此 pinned checkpoint 不暴露 faithful native MTP-1。跳過 stage、把 `mtp.0` 接到
`mtp.2` heads、替換 main-model head 或改變 block contract 都會離開官方 reference
graph；runtime truncation 已拒絕。這不否定另行訓練的一層 drafter，也不是 timing 或
quality 結果。研究說明與 raw evidence 分別位於
[`MTP-1 contract audit`](../research/MTP1_CONTRACT_AUDIT_2026-08-27.md) 與
[`machine-readable artifact`](benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json)。

## DSpark state ownership 與 reduced cache

Main expert cache 與 DSpark expert cache 使用不同 weight directories，但現有 key 都是
`(layer, expert)`；直接共用會把同名 key 對到不同 tensors。普通 prompt cache 只保存
target KV，不保存 layers 40–42 hidden taps 或 DSpark 的三個 attention contexts，所以
DSpark 不能安全套用一般 prefix reuse。完整 ownership map 位於
[`state ownership audit`](../research/DSPARK_STATE_OWNERSHIP_2026-08-27.md)。

後續 default-off candidate 建立獨立 format-3 namespace，把 target KV 與三個 DSpark
contexts 在相同 `prompt[:-1]` boundary 原子保存。八個 runtime tests 與兩個 decision
tests 通過 cloning、mutation isolation、revision／layer／prefix contract、namespace、
restart、malformed／partial bundle、eviction 與 close lifetime。Installed `repeated`
128/8 gate 的 `none → memory → persistent` sources 與 0／127／127 reused tokens 全部 exact；
三條 output hash 相同。Memory／restart hits 的 combined logical expert bytes 是
2,286,157,824／10,120,593,408，相對首次 29,412,556,800 分別降低 92.23%／65.59%。
這只通過 functional integration gate；選項、DSpark 與 hybrid 都維持預設關閉。詳見
[`atomic snapshot protocol`](../research/DSPARK_PROMPT_CONTEXT_SNAPSHOT_2026-08-27.md) 與
[`artifact`](benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json)。

第一個 reduced-state pilot 不改 graph，只把獨立 DSpark cache 從 768 限到 96 slots。
96 高於單輪最壞 90 assignments。單一 `storage_sentence` 4K／32 的 normal reference 與
768/96/96/768 全部產生 exact token IDs；96-slot peak memory 中位數少 0.883 GiB，
但 draft expert reads 多 0.722 GiB、misses 增加 27.23%，並出現 175 evictions。Request
time 的兩個 paired 方向相反，OS page cache 未控制，因此不採用 timing，也不更改
768-slot 預設。Artifact 位於
[`768 vs 96 slots`](benchmarks/2026-08-27-dspark-slots-768-vs-96-storage-4k32-m2-max.json)。

後續 `multilingual_choice` 與 `random_hex` 4K／32 single pairs 也保持 exact，且 output
hash 都匹配既有 hybrid v3 reference。三 workload 的 peak-memory change 是 -0.40% 至
-3.31%，draft-read change 是 +16.56% 至 +27.23%，all speculative-read change 是
+0.216% 至 +0.515%；96-slot evictions 是 80 至 175。組合 artifact 位於
[`three-workload summary`](benchmarks/2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json)。

在執行前固定的 4K／128 `random_hex` gate 要求 exact tokens、peak memory 至少 -1%、
draft expert bytes/committed token 不超過 +50%、all speculative bytes/committed token
不超過 +2%，以及 request time 不超過 +5%。Fresh normal、768、96 三條 128-token
序列 hash 都是
`aa163ef3caef62b9e177d651ccceb1c05d80df9aad0183a4df09e57a387b03b9`。
96 slots 的 peak memory 降低 7.35%，all speculative bytes/committed token 增加 0.91%，
但 draft bytes/committed token 增加 83.17%，misses 由 309 增至 566，並產生 470
evictions。因此 DSC96 未通過預先宣告的 gate，停止擴大此候選；單一 pair 的 -0.14%
request-time change 不是效能證據。Artifact 位於
[`4K/128 long-decode stop`](benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json)。

## Expert-file page-cache residency proxy

`--expert-page-cache-probe` 在每次 expert `preadv` 前以 `mincore` 分類該 range 的 VM
page residency。它保留四種互不混用的量：logical expert bytes、pre-read resident／
nonresident expert-file bytes、process-wide disk bytes，以及無法分類的 bytes。Probe
預設關閉，因 system-call observer overhead 會改變 timing；nonresident 也不是 physical
SSD bytes。

`code` 128/32 control/probe/probe/control 的四條 token sequence exact。兩個 probe runs
各將 90,323,288,064 bytes 全部分類，6,756 calls、零 failure、零 unclassified；median
是 32.10% resident、67.90% nonresident。接著的短 hybrid v3 + hash composition 把
main、draft、prefetch、useful／wasted partitions 全部閉合，但該 workload 的 prefetch
100% useful。

預先宣告的 4K `balanced_choice` normal/hash coverage gate 同時觀察到 useful 與 wasted，
並保持 exact output：802,160,640 useful logical bytes 中 681,705,472 是 nonresident；
2,607,022,080 wasted logical bytes 中 1,617,559,552 是 nonresident。Logical useful rate
是 23.53%，nonresident useful rate 是 29.65%。這證明 proxy 能揭露 logical accounting
看不到的 page-cache state，但仍不支援速度或採用結論。研究說明與 artifacts 位於
[`page-cache proxy audit`](../research/PAGE_CACHE_RESIDENCY_PROXY_2026-08-27.md)、
[`probe contract`](benchmarks/2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json)、
[`short composition`](benchmarks/2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json)
與
[`4K useful/wasted coverage`](benchmarks/2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json)。

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
| Block prompt cache v4 | 以完整 RoPE／KV／attention contract 與 128-token SHA-256 chain 命名 immutable cumulative cache checkpoints；suffix 分岔可回退到 bounded prefill checkpoint，frequency-aware sidecar 保留高重用 prefix。 | Restart partial-match、immutable sharing、contract rejection、eviction、MXFP8 snapshot-isolation tests，以及 installed-model functional gate。 |
| Direct slot read | `preadv` 直接寫入 MLX slot view。 | 目前程式碼和 full-model run。 |
| Ready expert decode | resident 和先讀完的 expert 先提交 compute。 | 五組配對 hash；改善中位數 12.9%。 |
| 分開 decode 指標 | model step、cache eval、end-to-end、p50 和 p95。 | 目前 status 和 metrics artifact。 |
| DSpark round transaction | 每 round 一次 cache fork；拒絕時只 replay committed prefix。 | DSpark tests 和目前程式碼。 |
| Verification expert union profiler | 記錄每層 assignments、unique experts、miss、verification／replay bytes 和 committed-token 成本。 | Expert cache、DSpark metrics tests 和目前程式碼。 |
| Process disk-I/O counter | request 前後讀取 Darwin process disk-I/O counter。 | metrics delta test 和 macOS 本機 API probe。 |

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
- routed expert path 以每層 token block 的 unique expert union 呼叫一次
  `get_many`，再把同 expert 的 token rows 集中計算。
- 預設關閉的 hybrid verifier 改以 one-token shape 執行 attention、FFN
  HyperConnection、router、shared／routed expert math；它仍先收集該層所有 route IDs，
  只呼叫一次 `get_many`，但不做 grouped multi-row expert QMM。
- metrics 分開記錄 union 前 assignments、union unique experts、cache miss、
  initial verification bytes 和 replay bytes。
- metrics 以 speculative round 的 committed tokens 計算 draft、target 與兩者合計的
  logical expert bytes／committed token。
- request metrics 另記錄 Darwin 對整個 process 計帳的 disk read／write bytes，
  作為 process-wide disk activity signal；它不能歸因到 expert file。
- 實驗性 `--dspark-hash-prefetch` 直接查 checkpoint `tid2eid[token_id]`，建立前三個
  main model layers 的 exact per-layer expert union。
- exact prefetch 只 pin 已 resident 的主 LFU entry；missing expert 進入獨立
  verification scratch，rejected-prefix replay 可重用而不做 LFU admission。
- metrics 直接依 accepted prefix 將實際 scratch read 分成 useful 與 wasted bytes，
  並記錄 target 等待 prefetch future 的 wall time。
- 實驗性 `--dspark-adaptive-block` 對 1、2、4、5-token prefixes 計算 confidence
  survival，完整候選預期利用率至少 0.90 時保留完整 block，否則以當下 missing exact
  hash experts 決定 verification prefix。
- target verification 前會保留一個 bonus／correction 位置，把 draft 限制為
  `remaining output tokens - 1`；這個 output-budget cap 與 adaptive trimming 分開計量。
- adaptive metrics 保存 selected／trimmed tokens、預期 commits、candidate unions、
  resident／missing counts、predicted bytes、score、selection reason、output-budget trim
  與 planning time；committed tokens 不再包含不會輸出的 suffix。

2026-08-26 五類 128-token 首輪 ABBA 否決了原始 storage-only policy：全部 output
hash parity 通過，但跨 workload request 中位數增加 2.013%、Decode 降低 14.843%，
整個 request logical bytes 沒有任何 workload 改善。第二輪加入 0.90 高信心護欄後，
五類 adaptive 都選完整 5-token block，logical counters 與 fixed 完全相同；request 與
Decode 的跨 workload 差異中位數分別是 +0.0495% 與 +0.2164%，只證明 regression
已消除。另一組低信心 A/B/B/A 仍使用 storage score，保持 output parity 並降低
deterministic logical bytes，但不足以採用 timing。詳細 evidence 位於三個
2026-08-26 adaptive calibration／repair artifacts。

後續五類 4K／256-output decision survey 重建相同 R0 prompts，五個 prompt 與 output
token hash 全部匹配歷史 fixed baseline。合計 54 個 decisions 全部走 high-confidence
full-block guard，storage-score decision 為 0；53 次選 5 tokens，唯一一次選 1 token
是中文最後一輪經 output-budget cap 後的完整 block。四個 workload 最終 fallback，
中文 workload 完成 43 rounds 而未 fallback。因 candidate 與 fixed selection 沒有行為
差異，paired A/B/B/A 依 stop gate 未執行。這是 correctness 與 selector-behavior
evidence，不是速度結果。

為滿足低／中 confidence gate，後續固定五個新的 4K instruction-suffix prompts。
五類 16-output screening 全部命中 storage score，也全部在第一 round 觸發正常
fallback。`storage_sentence` 重跑的 fallback cost ratio 是 1.301；research-only
no-fallback control 可在相同 16-token hash 下完成 8 rounds，但其中 5 rounds 正常應停止。

`random_hex` 的 no-fallback 4K／256 run 有 86 decisions、66 次 storage-score，卻未通過
fixed／adaptive output parity。32-token exact-ID 重現顯示 fixed 與 adaptive 都在 index
15 離開 sequential-prefill reference，並在 index 17 互相分歧；layer-major normal 另在
index 6 離開 sequential reference。共同 cache prefix 的 diagnostic 在分歧位置量到
sequential top-2 margin 0.125，而 block top logits 打平。這是 block-shape／低精度 near-tie
敏感性的 correctness stop，不是速度或 byte 改善。所有 no-fallback timing 與 logical-byte
差異都已標記無效；目前不採用未經全 workload 界定的 logit-margin heuristic。

逐 token target verification oracle 隨後完成。它在同一 round-level cache fork 上逐
token 驗證，且 rejection 後也逐 token replay；為隔離 verifier shape，oracle 禁止
hash prefetch scratch。`random_hex` 4K／32 fixed oracle 的 13/13 rounds 與 adaptive
oracle 的 17/17 rounds 都回到 sequential-prefill reference 的完整 token hash
`e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`，沒有分歧。
Adaptive oracle 仍執行 16 次 storage-score decisions，
所以成功不來自 selector 被繞過。這項證據排除 adaptive prefix policy 與 sequential
replay wiring，並把目前 correctness boundary 縮小到 block-shaped target verification；
oracle 本身沒有 verification batching，因此不是效能候選。

2026-08-27 的逐層 follow-up 在同一 exact common prefix 上捕獲 86 個 sequential layer
positions 與 43 個 block layers，且 instrumentation 前後 logits 完全一致。兩個位置的
embeddings exact，但第 0 層 `LocalAttention` 後最大 delta 已分別是
0.0009765625／0.00048828125；此時 router IDs、順序與 selected weights 仍 exact。
Position 1 到第 12 層、position 0 到第 16 層才首次改變 expert set。這表示 router 會
放大已存在的 hidden-state 差異，卻不是最早起點。該證據只定位到 layer 0 attention
branch，尚未隔離其中的 projection、attention kernel 或 output path；因此仍不重啟
效能 gate，也不採用 tolerance heuristic。原始資料位於
[`layer parity diagnostic`](benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json)。

同日 component follow-up 進一步顯示：input hidden、HyperConnection collapsed value、
attention norm、`wq_a`、`q_norm` 與 KV projection／norm／RoPE 都 exact；嚴格 stage order
最先在 HyperConnection `post`／`combine` 出現極小差異，而 `wq_b` 也在 exact input 下
非 exact。Sequential 與 block 分別走 one-token in-place/no-mask 與 two-token
concatenate/2x129-mask cache path，但最後 temporal-order cache 的共同 128-token suffix
exact。這排除 KV projection 與共同 cache suffix 是本次重現的最早差異來源，卻仍不能
在同時改變的 HyperConnection、Q、mask/cache layout 與 SDPA shapes 中指定單一原因。
下一個 correctness gate 是 one-token attention／block-shaped FFN-MoE hybrid，不是調整
selector。原始資料位於
[`layer 0 component diagnostic`](benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json)。

Hybrid follow-up 已完成。V1 的 one-token attention 在單一 exact cache state 恢復
near-tie top token，但 `random_hex` 多 round 仍在 index 15 分歧。FFN component
diagnostic 顯示 layer 0 router、routed outputs 與 MoE output exact，FFN HyperConnection
`post/combine` 先不同。V2 把 FFN HyperConnection token-shaped 後只通過五組中的三組；
`storage_sentence`／`multilingual_choice` 分別在 index 16／6 分歧。

V3 進一步讓 router、shared 與 routed expert math token-shaped，同時每層先 acquire
一次完整 expert union。五組 4K normal／fixed token 序列全部 exact。與 hash exact
prefetch 組合後五組仍 exact，fixed prefetch useful rate 是 23.53%–56.23%；再加入
adaptive selector 後，normal／fixed／adaptive 仍全部 exact。63 個 adaptive decisions
有 56 個選 1 token；prefetch useful rate 提升到 65.22%–91.31%，expert-union reuse
rate 則降到 15.59%–23.98%。這證明 correctness、selector behavior 與 logical-byte
tradeoff，不證明 request throughput、physical expert-file I/O、memory 或 power 改善。
V3、hash composition 與 adaptive composition artifacts 分別是
[`hybrid v3`](benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json)、
[`hybrid v3 + hash`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json)
與
[`hybrid v3 + hash + adaptive`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json)。

Explicit descriptor-policy follow-up 已完成。六個 fully nonresident installed expert
ranges 通過 aligned `F_NOCACHE`／cached byte 與 residency contract。接著以
`random_hex` 4K／32 執行三波 normal／fixed／adaptive Latin square，另以三個 observer
runs 驗證 page-residency accounting；12 條 output exact，所有 observer partitions
閉合。Fixed hybrid + hash 相對 normal 的 request／Decode 中位數是
+18.19%／-59.67%，adaptive hybrid + hash 是 +5.02%／-25.58%；兩者都未達預先登記的
request -5% 與 Decode +5% 門檻。Adaptive 雖把 speculative logical bytes/committed
token 降低 42.96%，仍未抵銷 token-shaped target math。這個 exact candidate gate
拒絕目前 fixed 與 adaptive candidates；不是對所有 verifier 或所有 workload 的否決。
Artifacts 位於
[`installed-range contract`](benchmarks/2026-08-27-expert-file-cache-bypass-contract-m2-max.json)
與
[`full-model gate`](benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json)。

Verifier-union tradeoff gate 隨後在 128-token high-confidence workload 上固定同一個
完整接受的 5-token block，執行四波 sequential／grouped／hybrid 與三個 observer。
19 條 outputs exact；acquisition calls 是 258／43／43。Hybrid 相對 sequential 的
target bytes -12.31%，但 verification time +17.34%；相對 grouped +87.59%。因此
one-per-layer union/dedupe plumbing 確認成立，卻不足以抵銷目前 token-shaped target
execution 的 aggregate cost。Grouped 較快不推翻其既有 low-margin correctness
rejection；而且 grouped／hybrid internal unions 不同，timing 不能歸因於 QMM alone。
Artifact 位於
[`verifier union tradeoff`](benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json)。

舊 `DSPARK_80_PERCENT_OPTIMIZATION_RESEARCH_2026-08-09.md` 的
per-position bottleneck 不再描述目前程式碼。

目前仍有下列限制。

- generation 使用 batch size 1。
- server 序列化 generation。
- DSpark 不讀取一般 prompt cache；default-off atomic namespace 可另行重用 target +
  context snapshot。
- DSpark prefill 不使用 main model 的 layer-major path。
- DSpark 使用獨立 768-slot expert cache。
- DSpark attention 尚未以目前硬體完成 GPU section profiling。
- 目前有五個 128-token／8-output-token 的 current-checkout exploratory ABBA，以及
  一輪五類 4K／256-output selector survey；後者沒有 storage-score decision，因此沒有
  執行無策略差異的正式 paired speed matrix。
- 原 block-shaped verifier 在低信心 `random_hex` 相對 sequential target parity 失敗；
  hybrid v3 已在五組 4K／32 correctness survey 恢復 exact tokens，也確認每層一次
  union acquisition；但短 repeated gate 的 verification 比 sequential 慢 17.34%、
  比 grouped 慢 87.59%，目前沒有 target verification 淨加速。
- process disk-I/O counter 不是 expert file 專用。Default-off `mincore` probe 現在可分解
  expert read 前的 resident／nonresident pages；default-cached descriptor policy 另有
  research-only aligned bypass control。兩者仍都不是 physical device-byte counter。
- hash-layer exact prefetch 與 adaptive block prototype 都預設關閉。4K／32 repeated
  bypass-policy gate 已記錄 peak memory、process disk bytes 與 observer labels，但目前
  fixed／adaptive candidates 都未通過 request／Decode gate；power、queue depth、GPU idle
  與 true physical expert I/O 仍未量測。
- adaptive score 只涵蓋三個 exact hash layers；其餘 40 層、hidden I/O、GPU idle、
  memory pressure 與 full-draft compute 尚未納入 cost model。
- Current DSpark path experiment 只保留 coherent first-order Markov beam。五個短 first
  rounds 的 baseline 全部接受 5/5，沒有 acceptance-preserving alternative；這不是低信心
  rescue、multi-round、sampled proposal 或 learned-router predictor 的證據。
- Direct learned-router transfer 只測試四個未訓練 DSpark taps，並明確排除 anchor routes
  的 scoring。低 recall／useful rate 不等於 trained reducer 沒有訊號；目前也不是執行
  prefetch、page residency 或 physical SSD byte evidence。

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
| w13/w2 staged streaming | Fixed-arena gate 通過；tested runtime schedule 停止 | 四波 full-model exact，但 request +2.17%、Decode -4.26%。只在不同 first-stage kernel／I/O schedule 可實質放大 one-row overlap window 時重啟；不要重跑相同 split-slot candidate。 |
| Adaptive expert prefill | Five-workload route eligibility 通過；tested post-attention runtime schedule 停止 | Selected rows 及 outputs exact、request expert bytes -66.78%，但 `repeated` 4K TTFT +16.87%、p95 +17.02%。保留 full-layer 預設；只在能恢復 overlap 的 exact 設計或另行量測 predictor 出現時重啟。 |
| Prefill-to-decode handoff | 單 prompt 離線證據不足 | 多 domain miss coverage 穩定且不增加 eviction。 |
| Learned-router predictor | 6,000-label direct-transfer baseline 完成且停止 | Top-24 最佳只有 17.15% assignment recall／11.15% useful rate。重啟先定義 data rights、擴大 route traces、train/validation/held-out split，並保存 trained reducer/predictor hash；不直接建立 prefetch。 |
| Atomic DSpark prompt-context reuse | Exact memory／restart functional gate 通過，default off | 若評估採用，先做多 workload repeated waves、平衡 expert-cache state，並把 acquisition/load 納入外部 wall time。 |
| Hybrid v3 verifier | Five-workload greedy correctness 與 one-per-layer union contract 通過；短 repeated gate 顯示 aggregate token-shaped execution cost material，且 4K／32 end-to-end candidate 已拒絕 | 新候選必須先在五 workload 保持 exact，同時恢復更多 grouped execution；不要重跑相同 v3。 |
| Hash-layer exact prefetch | 五組 4K correctness 通過；4K／32 bypass-policy repeated gate 的 fixed hybrid + hash request +18.19%、Decode -59.67%，候選拒絕且預設關閉 | 只有 verifier／prefetch architecture 改變且先有較低 target-math 成本時才重啟；新 gate 仍需多 workload、observer labels 與 attributable I/O。 |
| Storage-aware adaptive block | 五組 correctness 通過；4K／32 bypass-policy repeated gate 雖降低 speculative bytes/committed 42.96%，request +5.02%、Decode -25.58%，候選拒絕且預設關閉 | 先改善 target verifier 或 cost model；重啟時加入多 workload、queue depth、GPU idle、memory pressure 與 attributable I/O。 |
| Storage-aware candidate path | Coherent branch-4／beam-8 first-round gate 結構 exact，但 0/5 workloads 在降低 hash storage 時維持 baseline acceptance | 不重跑相同 gate。只有 materially different candidate generator 或 trained selector/drafter 才重啟；sampled mode 先證明完整 selected-path proposal probability。 |
| Adaptive expert prefill | Route eligibility 與 selected-row runtime correctness 通過；post-attention schedule 停止 | 不重跑相同 70/80/90 matrix。只有 exact planning／I/O 可與現有 layer pipeline overlap，或 predictor 有獨立 useful／wasted byte gate 時才重啟。 |
| MTLIO bridge | Native mechanism gate 通過；MLX handoff 停止 | 只在 MLX 公開支援 external Metal resource + event dependency 後重啟；再跑 repeated balanced cache waves，不能沿用本次 exploratory timing。 |
| P3 `gather_qmm` tile | Profile gate 未通過 | 只有新的 workload 或 upstream kernel 證據讓占比達到 15% 才重啟。 |
| P4 attention 類型區塊 | 正確性 gate 未通過；prototype 已移除 | 新設計必須保留 expert route。 |
| D2 routed expert top-6 | Profile gate 未通過 | 新的 workload 或 upstream 證據必須先讓占比達到 3%。 |
| D3 CSA row 重用 | Row gate 與 profile gate 都未通過 | 新的 workload 或 cache 設計必須先讓 row 重疊率達到 90%，且 `gather` 占比達到 5%。 |
| Custom Metal kernel | 尚未實作 | 某個穩定 section 先通過該研究項目的 profile gate。 |
| DSpark 預設啟用 | R3 已拒絕目前候選 | DSpark 實作改變後，新的探索性 ABBA 先通過 request、throughput 與 memory gate。 |
| M1 Prefill file-cache | 正式評估完成；候選已拒絕並移除 | 只有新的設計與新的 profile 證據才重新研究。 |
| Approximate MoE | 不在 exact mode | API 必須明確標示 output parity 不成立。 |

## Staged `w13`／`w2` expert streaming

2026-08-27 的 fixed-arena gate 保留 canonical expert blob 的六個 regions 與總
13,369,344-byte budget，但把 destination 固定拆成 8,912,896-byte `w13` 和
4,456,448-byte `w2` arrays。36 組 paired samples 的 region／float32 output hashes、
alignment 與 bypass nonresidency 全部 exact。4-row complete wall 中位數 -10.74%、p95
-10.24%、hidden read 80.98%；8-row分別是 -13.67%、-12.67%、87.90%。這兩個 shape
把所有 rows 放在同一個 routed expert，只是 optimistic component gate。

通過後建立 internal-only `RuntimeConfig(staged_expert_streaming=True)` prototype：direct
miss 的 `w13` 先讀入固定 split slot，`w2` 由另一個 bounded executor 讀取，first-stage
MLX graph 先提交，特定 `w2` future 成功後才提交 down QMM 並把 slot 標為 loaded。錯誤／
cancel 會釋放 partial slot；resident hits 不重讀；full-layer batched prefill 不變。設定
預設 `false`、需要 ready-expert decode、拒絕 DSpark，且沒有 CLI／server／APP opt-in。

完整 installed model 使用 128-token `repeated` prompt、32 greedy outputs、四波
control/staged fresh-process pairs。八條 output token SHA-256 全部是
`06de0413ef6f0d6f67a8328b6cfbea9cf97fe28689b6503ad9975367a59d3fa7`；每個 staged
run 的 1,024 次 reads 精確閉合 9,126,805,504 `w13` + 4,563,402,752 `w2` bytes。
Logical expert bytes 40,121,401,344、evictions 1,849、peak MLX memory
24,665,121,392 bytes 在每一 pair 都不增加。

但 paired median request 是 +2.17%，Decode throughput -4.26%，decode p95 +2.12%。
Correctness、p95 與 memory guards 通過，必要的 request 或 throughput 5% 改善失敗；
decision 是 `stop_runtime_candidate_keep_default_off`。完整 protocol 與 artifacts 位於
[`staged research`](../research/STAGED_EXPERT_STREAMING_2026-08-27.md)、
[`fixed-arena result`](benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json) 與
[`runtime result`](benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json)。
兩個 gate 都是 exploratory、非正式效能，也不是 physical SSD byte measurement。

## Adaptive full-layer／selective expert prefill

五種 4,096-token prompts 的 exact route-union gate 隔離每個 expert layer 的
24,570 assignments；所有 reference／trace token hashes 一致。70%／80%／90% threshold
估計的跨 workload prefill expert-byte變化中位數是 -23.19%／-33.38%／-34.24%，三者
都通過至少三個 workload 節省 10% 的 continuation gate。這只是 logical-byte upper
bound，不包含 synchronization、scatter 或 overlap cost。

Internal prototype 保留 full 256-row fixed buffer 與相同 `gather_qmm`；當層 attention
完成後計算一次 router/shared tensors，把 exact union rows 寫回原 expert offsets，再完成
原 route reduction／HyperConnection math。`adaptive_expert_prefill_threshold` 預設
`None`、只有 0.7／0.8／0.9、拒絕 DSpark／staged composition，且無 public opt-in。

`repeated` 4K 的兩組反向 fresh-process pairs 全部輸出 `[1950, 1950]`，token SHA-256
是 `dd22c238773ee5642280c221b7a7a51094e6dcec6882cce0f749378ee01e4891`。Candidate
42 層全 selective，3,309 rows 精確讀取 44,239,159,296 adaptive batched bytes；加 393
ordinary misses 後 request bytes 是 49,493,311,488，較 control 149,001,338,880 少
66.78%。Peak MLX memory 也少 15.78%。

但 TTFT paired median +16.87%、p95 +17.02%；control 41 個 next-layer prefetch hits，
candidate 0 個，routing sync wall 中位數也從 0.121 s 增至 2.061 s。這只支持
route-planning、scatter 與 lost-overlap schedule 的組合成本太高，不支持把 slowdown
歸給單一 component。`repeated` 在三個 threshold 的 decisions 完全相同，因此每個
threshold 都已失敗 per-workload p95 guard；完整矩陣與 cached observer 依 stop rule
取消。Artifacts 位於
[`research protocol`](../research/ADAPTIVE_EXPERT_PREFILL_2026-08-27.md)、
[`route gate`](benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json)
與 [`runtime stop`](benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json)。

## DSpark storage-aware coherent candidate paths

現有 DSpark 不是五個獨立 position marginals。一次三層 backbone 先產生固定
position logits；低秩 Markov head 再依前一個候選 token 改變下一個 conditional
distribution。因此 research script 可在不重跑 DSpark backbone 的情況下建立 coherent
first-order Markov beam。Joint score 使用 normalized log probability；greedy target
truth 一律由 cloned target cache 逐 token 產生。這不是 DFlash2 selector 的重製。

Branch 4／beam 8 gate 對五個 128-token prompts 各跑一個 fresh process、只量第一個
speculative round。五個 reconstructed greedy paths 都與實際 `DSparkModel.draft` token／
confidence 一致，所有 top-4 transition、hash layers 0--2 route、target truth 與 committed
prefix gates 通過。五個 baseline 都被 target 完整接受 5/5。

64 組 probability／requested-union／snapshot-miss 權重中，`code` 有 40 組選到少
17 requested、16 missing blobs 的 alternative，但 acceptance 從 5 降為 4；`tool_like`
有 37 組選到少 16 requested、14 missing blobs，acceptance 降為 1。其餘三類只選
baseline。0/5 workloads 同時維持 acceptance 並降低 storage，所以 decision 是
`stop_current_markov_beam_candidate`。Artifact 位於
[`candidate-path first-round gate`](benchmarks/2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json)，
SHA-256 是
`de2ee382fff19f030a3c5150fe1251c352e51d6e5928fb2bfa7c5e40211d29ce`。

這是 candidate-feasibility、不是 timing。沒有 runtime setting，sampling 也未啟用：
任意 storage selector 會改變 proposal distribution，必須先導出 selected-path probability
並完成 rejection correctness proof。完整 protocol／stop decision 位於
[`research/DSPARK_CANDIDATE_PATHS_2026-08-27.md`](../research/DSPARK_CANDIDATE_PATHS_2026-08-27.md)。

## DSpark hidden 到 target learned-router baseline

五個 128-token fresh workers 各捕捉第一個完整 DSpark draft。Target label 由 sequential
verifier 對 `[anchor, t_1, ..., t_5]` 的 one-token calls 產生；scoring 將 DSpark
position `i` 對齊 target 處理 draft input `t_i`，所以 layers 3--42 共 6,000 個 top-6
assignments。Anchor routes 另保存但不計入 predictor bytes。

No-training candidates 將三個 DSpark layer outputs 分別經既有 `hc_head` collapse，另加
final output norm，共四個 feature sources。每個 target layer 再套用該層 frozen
`ffn_norm`、router weight、score function 與 correction bias，量 top-6／12／24／48。
所有 prompt、draft reconstruction、route shape、prediction uniqueness 與 useful／wasted／
missed union identities 通過。

| Best feature | k | Assignment recall | Union recall | Useful rate | Layers >=75% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dspark_final_norm` | 6 | 6.27% | 12.18% | 13.50% | 0/40 |
| `dspark_final_norm` | 12 | 10.13% | 20.56% | 12.40% | 0/40 |
| `dspark_final_norm` | 24 | 17.15% | 33.03% | 11.15% | 0/40 |
| `dspark_final_norm` | 48 | 28.93% | 51.26% | 10.03% | 0/40 |

最佳允許 size top-24 的 per-workload assignment recall 介於 12.42% 與 22.42%，遠低於
80% aggregate、70% workload floor、30-layer 與 50% useful-rate gates。Top-48 已超過
允許的 prefetch fanout，仍約 90% predicted union 不在 target union。Decision 是
`stop_direct_frozen_router_transfer_training_required`。Artifact 位於
[`frozen-transfer gate`](benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json)，
SHA-256 是
`4eb2574b7a9e6b613784288833a3d6ff327e71af145e6d822c2b501d387f7739`。

Runtime 不新增 learned-layer scratch prefetch、probation 或 deadline pinning。這個負面
baseline 不否定 trainable signal；重啟需要新的 data-rights／collection protocol、更大
dataset、train／validation／held-out splits 與 immutable trained reducer/predictor hash。
完整 protocol 位於
[`research/DSPARK_LEARNED_ROUTER_PREDICTOR_2026-08-27.md`](../research/DSPARK_LEARNED_ROUTER_PREDICTOR_2026-08-27.md)。

## MTLIO、zero-copy 與 custom Metal

2026-08-27 的 standalone Swift gate 已完成 native mechanism 驗證。它以 canonical
13,369,344-byte expert ranges 比較四-worker `preadv`、MTLIO `loadBytes`、shared
`MTLBuffer` 與 private `MTLBuffer`，每種執行 1／6／32／128 ranges。16/16 rows 的
candidate/reference SHA-256 exact；shared/private 都以 external shared event signal 1
排序 GPU validation blit，取消 probe 回傳 `cancelled` 且 destination 未 admission。

Shared buffer 的 direct CPU hash exact，不需要 CPU copy。Private buffer 必須 blit 到
shared validation buffer 才能由 CPU hash。32／128 aggregate 的 MTLIO bytes/shared
相對不同 layer-file `preadv` 都在 ±0.78% 內；六 expert p95 反而增加 51.44%／52.38%。
本次每列開始前至少 99.9885% range pages nonresident，但 OS page cache 沒有 purge、
沒有 repeated balanced waves，因此時間只能作 exploratory direction，不是正式效能或
physical-SSD 結果。

Installed MLX 0.32.0 audit 找到 public C++ raw-pointer no-copy candidate 與 Metal
DLPack round trip，但 `mx.metal` 沒有 direct resource/event import，public `Event` 也
不能 adopt external `MTLSharedEvent`。Observed `from_dlpack` provider call 沒有收到
stream/event dependency。這表示 native bytes 與 Metal queue 同步成立，卻不能透過目前
公開 API 把相同 dependency 安全交給 MLX GPU work；allocator/backend internal cast 不在
支援範圍。

因此 runtime MTLIO integration 停止，四-worker `preadv` slot ownership 維持不變。
完整 protocol 與 artifact 位於
[`native gate`](../research/MTLIO_EXPERT_STREAMING_2026-08-27.md) 與
[`machine-readable result`](benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json)。
只在 MLX 公開提供 external resource + event/stream handoff 後重啟；屆時仍須 balanced
cache waves、端到端 target-model overlap 與至少 10%／p95 5% gate。

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

2026-08-26 的 adaptive follow-up 已完成五類 4K／256 decision survey。
五個 output hash 均匹配歷史 fixed R0；54 個 decisions 全部選完整 block，沒有
storage-score decision。因 control 與 candidate 沒有策略差異，stop gate 阻止後續
A/B/B/A。下一次重新啟動 paired gate 前，必須先建立可重現且能多輪觸發 storage score
的 workload。

低信心 discovery 隨後建立了該 workload，但 long-decode exact parity gate 失敗。
`random_hex` fixed／adaptive 在 256 tokens 產生不同 hash；32-token trace 也顯示 block
verification 與 sequential target 在 near-tie token 分歧。因此 R3 follow-up 再次停止，
且 no-fallback research control 不得作為服務設定或效能證據。

逐 token verification oracle 的 fixed 與 adaptive 4K／32 follow-up 都精確匹配
sequential-prefill reference；adaptive run 仍有 16 次 storage-score decisions。這證明
selector 與 sequential replay wiring 可保持該案例的 target token parity，但 oracle
移除了 block verification 的 batching 價值，不能推翻 R3 stop decision。下一輪只處理
block-shaped verifier correctness，不量測或宣稱 oracle speedup。2026-08-27 layer-wise
diagnostic 已顯示 exact embeddings 在第 0 層 attention branch 後首次產生差異，router
到後續層才改變 expert set。Component diagnostic 又排除 exact input、attention norm、
KV projection 與共同 128-token cache suffix，並觀測到 HyperConnection `post/combine`、
`wq_b`、mask/cache layout 與 attention execution shape 的差異。

後續 hybrid v1／v2 分別排除只有 one-token attention、再加 one-token FFN
HyperConnection 仍不足；hybrid v3 以完整 token-shaped target math 加每層一次 expert
union，在五組 4K／32 normal／fixed 全部恢復 exact IDs。Hash 與 adaptive composition
也通過相同五組 correctness gate。後續 explicit expert-file bypass-policy gate 已完成
三波 fixed／adaptive performance runs 與獨立 observer wave；12 條 exact，但 fixed 與
adaptive 的 request／Decode 都比 normal 差並未達預先門檻。R3 的「不預設啟用」決策
因此不變。下一步不再重複量測同一候選；只有 target verifier 或 predictor architecture
實質改變後，才以多 workload gate 重啟。

Greedy-only multi-candidate follow-up 也已完成。它從同一次 DSpark backbone 的 Markov
conditionals 建立 branch-4／beam-8 coherent paths；五組 128-token first rounds 的
baseline 都接受 5/5，任何 storage-selected alternative 都降低 acceptance，0/5 通過。
因此不建立 multi-round path runtime。下一個不需新 drafter training 的 executable
方向是固定 learned-router route-trace dataset 與 top-k recall／useful-wasted byte gate。
該 gate 也已完成：6,000 labels 的最佳 top-24 direct transfer 只有 17.15% assignment
recall 與 11.15% useful rate，因此不建立 learned-layer prefetch。後續 predictor 工作
需要使用者明確授權的資料收集／訓練 scope。

後續 block-granular persistent-prefix milestone 也已完成。Normal persistent format 4 把
model revision、完整 config hash、RoPE、KV format、attention mode 與 128-token block chain
納入 content key；第一與最後完成的 non-layer-major prefill checkpoints 提供 bounded
partial match。453-token suffix-branch gate 在重啟後精確重用 128／442 個共同 prefix tokens，
與 isolated cold branch 產生相同 token `36363`／SHA-256；7,002,850-byte shared payload
與 metadata hash 未改變，reuse sidecar 由 0 增為 1。第一次 full-model run 曾抓到可變
MXFP8 chunk list 汙染舊 snapshot，修正為 list snapshot 後才重跑通過。這是 functional
gate，不是 balanced speed result；format 4 分享 cumulative checkpoints，尚未實作
per-layer KV delta dedupe。完整 protocol 與結果位於
[`BLOCK_PROMPT_CACHE_2026-08-27.md`](../research/BLOCK_PROMPT_CACHE_2026-08-27.md)。

最後的 prerequisite closure audit 將剩餘九個方向分開收斂。完整 Xcode 的
`xctrace`／`metal`／`coremlcompiler` 與系統 observation tools 都存在，但 installed
manifest 沒有 separately named dense drafter tensors；checkout 也沒有 trained／Core ML
candidate、approved data/training/approximate-mode contract、current `.trace`、attributable
physical-device-byte field 或 storage-native model project spec。`.venv` 沒有
`coremltools`／PyTorch／datasets／accelerate。現有 6,000 labels 只屬 direct-transfer
feasibility，不能冒充 rights-approved training dataset。

因此 physical-I/O 與 GPU trace 方向以 instrumentation／measurement boundary 關閉；
probation／deadline、dense drafter 與 ANE 以 upstream training candidate 關閉；native
zero-allocation loop 等待 trace-selected boundary；router locality、sub-FP4/drop 與
storage-native model 分別以 non-equivalent training／approximate-mode／separate-project
prerequisite 關閉。這些是 scoped stop/defer，不是宣稱方向永遠不可行。Audit artifact
SHA-256 是
`213087121a7dc7b0c39d1e15066c92f470dfaf6c5326dfcf9889a6d8a6fafcfa`；完整 reopening
rules 位於
[`PLAN_PREREQUISITE_CLOSURE_2026-08-27.md`](../research/PLAN_PREREQUISITE_CLOSURE_2026-08-27.md)。

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
Tested staged streaming schedule 已完成並停止；只有 first-stage kernel 或 I/O schedule
實質改變時才重啟。MTLIO prototype 等待 MLX public external-event handoff prerequisite，
不重複目前 native gate。
Tested adaptive prefill post-attention schedule 也已完成並停止；full-layer prefetch 保持
預設。只有能恢復 exact overlap 的不同設計或有 useful／wasted byte 證據的 predictor
才重啟。
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
- [DFlash paper](https://arxiv.org/abs/2602.06036)
- [DFlash official repository](https://github.com/z-lab/dflash)
- [DFlash2 official model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- [Apple Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [Apple Recurrent Drafter repository](https://github.com/apple/ml-recurrent-drafter)
- [Apple ReDrafter MLX experiment](https://github.com/apple/ml-recurrent-drafter/blob/main/recurrent_drafting/mlx/experiments/README.md)
- [HyperDFlash paper](https://arxiv.org/abs/2606.26744)
- [ReMoE router fine-tuning paper](https://arxiv.org/abs/2605.27081)

外部主張的逐項查核位於
[`research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md`](../research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md)。
