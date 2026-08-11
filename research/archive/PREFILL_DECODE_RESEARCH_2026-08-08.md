# Prefill 與 decode 效能研究

> [!WARNING]
> 本文件是 2026-08-08 的歷史研究。估計值不是目前效能結果。
> 請以[目前效能指南](../../docs/PERFORMANCE.md)和
> [驗證紀錄](../../docs/VALIDATION.md)為準。

Checked: 2026-08-08

## 結論

DSpark 改為下一階段的第一優先。

完整順序記錄在 [DSpark-first 實作計畫](DSPARK_FIRST_PLAN_2026-08-08.md)。

先補 DSpark 所需的最小 profiling。

目前資料足以說明 prefill 已不再只受 SSD 限制。

目前資料不足以判定 decode 的主要瓶頸。

一般 prefill 最佳化延後。

一般 decode 最佳化也延後。

第一個工作是固定 DSpark configuration 與 `mtp.*` tensor contract。

第二個工作是讓 installed model 可以選擇包含 DSpark。

第三個工作是加入 cache checkpoint、commit 與 rollback。

若 profiling 支持本文假設，14K prefill 的工作目標是從 76.54 秒降至 60–70 秒。

若 profiling 支持本文假設，decode 的工作目標是從 14.14 Tok/s 提升至 16–18 Tok/s。

這兩個範圍是工程假設。

這兩個範圍不是量測結果。

## 範圍

本文只研究 batch size 1。

本文以 greedy decode 為正確性基線。

本文保留 main model 的 prefill 與 decode 研究結果。

DSpark 是下一階段的主要範圍。

DSpark 的詳細計畫記錄在 [DSpark-first 實作計畫](DSPARK_FIRST_PLAN_2026-08-08.md)。

本文不重複提出下列已完成項目：

- layer-major prefill。
- 1,024-token attention step。
- 4,096-token MoE tile。
- full-layer prefetch。
- strided full-layer `gather_qmm`。
- FP4 lightning-index cache。
- MXFP8 KV cache。
- in-memory 與 persistent prompt cache。
- decode 的 direct six-expert path。

## 目前基線

4,097-token varied prompt 從 grouped fallback 的 44.03 秒降至 28.02 秒。

這是 36.4% 的 TTFT 改善。

14,363-token prompt 從 105.81 秒降至 76.54 秒。

這是 27.7% 的 TTFT 改善。

14,363-token prefill 讀取 150.41 GB routed expert bytes。

其中 SSD read time 是 11.85 秒。

因此，完全移除這段 read time 的上限只占總時間 15.5%。

剩餘時間包含 attention、lightning indexer、mHC、shared expert、routed expert、graph 建立、GPU 執行與同步。

目前 14,363-token prefill 執行 504 次 `gather_qmm`。

目前 batched 路徑在每個 MoE tile 仍呼叫 `mx.eval(indices)`。

MLX 使用 lazy evaluation。

`eval()` 會執行尚未完成的整個相依 graph。

因此，現有 `routing_sync_seconds` 不是純 router 時間。

MLX 也說明每次 graph evaluation 有固定成本。

極小 graph 與極大 graph 都可能較慢。[MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html#when-to-evaluate)

目前 decode 只有 request-level 指標。

目前指標不能分開 attention、lightning indexer、MoE、shared expert、mHC 與 cache-state evaluation。

因此，14.14 Tok/s 只能作為 end-to-end 基線。

## 瓶頸假設

### Prefill

**H1：batched MoE 中的 router 同步造成可見成本。**

full-layer expert 已在 MoE 執行前載入。

batched `gather_qmm` 可直接使用 MLX `indices`。

這條路徑不需要先把 `indices` 轉成 NumPy。

因此，`mx.eval(indices)` 可能是多餘的 CPU-GPU 同步點。

但是，必須先單獨量測這個同步點。

不能直接移除這個同步點。

**H2：full-layer buffer 可能發生一次約 3.42 GB 的複製。**

目前每層先建立 `bytearray`。

runtime 再呼叫 `mx.array(np.frombuffer(blob, dtype=np.uint32))`。

目前資料沒有證明這個轉換是 no-copy。

MLX PR #2875 已加入 C++ raw-pointer array constructor。

該 constructor 會在可行時建立 no-copy array。[MLX PR #2875](https://github.com/ml-explore/mlx/pull/2875)

這個 API 只證明 shared-buffer 方案可研究。

這個 API 不證明 Python 的 `mx.array(np.frombuffer(...))` 是 no-copy。

**H3：prefill 的剩餘時間可能由 GPU kernels 與 host dispatch 共同造成。**

DeepSeek-V4 的官方系統使用 fused kernels。

官方資料指出小型 kernel 的 host orchestration 可限制吞吐量。[DeepSeek-V4, section 3.2](https://arxiv.org/html/2606.19348#S3.SS2)

這個結果來自不同硬體。

本 runtime 必須用 Metal capture 確認。

### Decode

**H4：routed expert miss 與 SSD wait 可能限制短 context decode。**

每個 token 在每個 MoE layer 選六個 routed experts。

目前 direct path 對每個 expert 執行 `w1`、`w3` 與 `w2`。

因此，每個 MoE layer 與每個 token 執行 18 次 quantized matrix multiplication。

目前每層也需要把 `indices` 同步到 CPU，才能查詢 expert cache。

**H5：attention 與 lightning indexer 可能隨 context 增長而限制 decode。**

DeepSeek-V4 使用異質 KV cache。

它分開管理 compressed cache、SWA state 與 incomplete compression state。[DeepSeek-V4, section 3.5.1](https://arxiv.org/html/2606.19348#S3.SS5.SSS1)

官方模型用 FP4 執行 lightning-index attention。

官方模型用 BF16 儲存 RoPE dimensions，並用 FP8 儲存其餘 KV dimensions。[DeepSeek-V4, attention precision](https://arxiv.org/html/2606.19348#S2.SS3.SSS3)

目前 runtime 已實作 FP4 index 與 MXFP8 cache。

但是，runtime 尚未分開量測 index matmul、stable top-k、gather/dequantize、softmax 與 value reduction。

**H6：cache-state evaluation 可能在長輸出累積可見成本。**

runtime 在每個輸出 token 後評估完整 prompt cache state。

這個動作保護長輸出的 Metal resource 使用。

目前只有累計時間。

目前沒有 2K 或 12K output 的 TPOT 分布。

**H7：LFU 可能不符合 MoE 的 layer access order。**

Apple SpecMD 指出 MoE access 依 layer 順序前進。

該研究指出 LRU 與 LFU 可能造成 collision miss。

該研究的 Least-Stale 結果來自其他模型與硬體。[Apple SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching), [paper](https://arxiv.org/html/2602.03921)

因此，本文只建議先做本地 trace simulator。

本文不建議直接替換 LFU。

## 必須先補的 profiling

### 量測方法

建立不改變執行順序的 trace hooks。

每個 phase 記錄 CPU wall time。

每個 phase 也記錄完成該 phase 的同步時間。

同步時間與 CPU submission time 必須分開。

短基準執行五次。

中型基準執行三次。

12K-output 穩定性基準先執行一次。

若 12K-output 基準出現異常，再重複執行兩次。

短基準報告 median 與 p95。

中型基準報告 median。

代表性的 4K 與 14K 短基準分開記錄 cold page cache 與 warm page cache。

其他基準使用 warm page cache。

測試記錄必須說明 page cache 的準備方法。

每個基準記錄 process resident memory、MLX active memory、MLX cache memory 與 MLX peak memory。

MLX 說明 active memory 不包含 cached buffers。[MLX active memory](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.get_active_memory.html)

因此，不能只用一個 MLX memory counter 判定實際 memory pressure。

### Prefill phases

每層記錄下列 phase：

1. attention mHC 與 norm。
2. attention 與 cache update。
3. lightning index matmul。
4. stable top-k。
5. packed KV gather 與 dequantize。
6. sparse softmax 與 value reduction。
7. router graph submission。
8. `mx.eval(indices)` wait。
9. shared expert。
10. routed `gather_qmm` 的 `w1`、`w3` 與 `w2`。
11. route restore 與 score reduction。
12. FFN mHC。
13. full-layer SSD read。
14. `bytearray` 到 MLX array 的轉換。

另外記錄 graph evaluation 次數、Metal dispatch 次數、command buffer 數量與 copy kernels。

### Decode phases

每個 token 與每層記錄下列 phase：

1. attention。
2. lightning indexer。
3. routed MoE。
4. shared expert。
5. mHC。
6. router CPU sync。
7. expert cache lookup。
8. expert SSD read wait。
9. cache-state evaluation。
10. token sampling 與文字處理。

每個 token 記錄 expert bytes、read wait、hit、miss、eviction 與 selected expert IDs。

每個 request 記錄 TPOT p50 與 TPOT p95。

每個 request 也記錄 context 長度對 attention 與 indexer 的斜率。

### 基準矩陣

快速比較使用 4K、8K 與 14K context。

每個快速比較產生 256 output tokens。

每個快速比較執行五次。

長輸出趨勢使用 4K 與 14K context。

每個長輸出趨勢產生 2K output tokens。

每個長輸出趨勢執行三次。

穩定性測試使用 14K context 與 12K output tokens。

穩定性測試先執行一次。

若結果出現 memory growth、效能突降或錯誤，再重複執行兩次。

每個輸出長度都必須保留 greedy token parity。

### Profiling 門檻

profiling 版本相對於無 profiling 版本的 median total time 差異必須小於 1%。

所有 phase 的 accounted wall time 必須涵蓋 total time 的至少 90%。

Metal capture 只截取代表性區段。

不要 capture 完整 12K output。

MLX 可建立 `.gputrace`。

Xcode Dependencies view 可顯示 operation 相依關係。[MLX Metal Debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)

Apple Metal System Trace 可同時顯示 CPU、GPU 與 memory timeline。[Apple Metal developer workflows](https://developer.apple.com/documentation/Xcode/Metal-developer-workflows)

GPU counters 只用於 capture 顯示的主要 kernels。[Apple GPU counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)

## DSpark 完成後的次要順序

本節不是下一階段的主路線。

下一階段先執行 [DSpark-first 實作計畫](DSPARK_FIRST_PLAN_2026-08-08.md)。

### R0：加入 profiling，預估 1–2 個工作日

R0 只加入量測。

R0 不改變 runtime 執行順序。

R0 交付 phase metrics、Metal capture 與基準指令。

### R1：建立 decode 基線，預估 1–2 個工作日

R1 先完成 256-output 快速比較。

R1 再完成 2K-output 趨勢測試。

R1 最後執行一次 12K-output 穩定性測試。

### P1：驗證低風險 prefill 改善，預估 2–4 個工作日

P1 測試 batched router sync。

P1 也確認 full-layer MLX array 是否複製。

### D1：驗證 decode MoE 改善，預估 3–5 個工作日

D1 先建立 expert trace simulator。

D1 只有在 profiling 通過門檻後，才建立 fused `w13` prototype。

### P2：驗證 compile 與 kernel 融合，預估 2–5 個工作日

P2 只處理 capture 顯示的主要瓶頸。

### P3：評估 native I/O，預估 1–2 週

P3 只有在 copy 或 I/O 仍占總時間至少 10% 時開始。

以上時間是工程估算。

以上時間不包含完整 12K-output 重複測試。

## 可實作項目

### P1a：測試 batched MoE 的 router sync

先在 batched prefill path 加入可切換實驗。

實驗 path 讓 `indices` 保持 lazy。

實驗 path 直接把 `indices` 傳給 `_gather_sort` 與 `gather_qmm`。

fallback 與 decode path 保留 `mx.eval(indices)`。

這個實驗不能改變 route IDs、route scores 或 output。

預期影響是 3–10% prefill 改善。

這是未量測假設。

主要風險是 graph 變大、峰值 memory 增加或同步延後到更昂貴的位置。

接受門檻是 4,097-token 與 14,363-token median TTFT 至少改善 3%。

接受門檻也要求 peak memory 不增加超過 5%。

### P1b：確認 full-layer MLX array 是否複製

在 `bytearray` 建立前後記錄 process memory 與 MLX memory。

在 `mx.array(np.frombuffer(...))` 前後記錄時間與 Metal copy kernels。

保留 source buffer，直到 GPU 完成該層。

先確認 Python path 是否複製。

若 Python path 複製，再做最小 C++ extension prototype。

prototype 使用 MLX raw-pointer constructor。

prototype 不得修改 installed model 格式。

預期影響上限受 11.85 秒 SSD read 與尚未量測的 copy time 限制。

主要風險是 buffer lifetime、alignment 與 GPU read 時發生 overwrite。

接受門檻是移除一個約 3.42 GB/layer 的 copy。

接受門檻也要求 14K median TTFT 至少改善 3%。

### D1a：建立 decode expert trace simulator

trace simulator 讀取真實 selected expert IDs。

simulator 重播目前 slot count、layer reserve 與 miss cost。

simulator 比較 LFU、Least-Stale 與只調整 layer reserve 的簡單 policy。

simulator 不改 router。

simulator 不做 expert substitution。

simulator 不做 expert drop。

預期影響先不給固定百分比。

SpecMD 的 10.7–34.7% TTFT 改善不能套用到本 runtime。

進入實作的門檻是三種 context 與三種 output 長度都減少至少 15% miss bytes。

進入實作的門檻也要求沒有一個基準增加 miss bytes。

正式實作的接受門檻是 median Tok/s 至少改善 5%。

正式實作的 p95 TPOT 不得惡化超過 2%。

### D1b：repack fused `w13`

把每個 routed expert 的 `w1` 與 `w3` 合併成一個 packed region。

每個 expert 先用一次 quantized matrix multiplication 計算 fused `w13`。

runtime 再把結果分成 gate 與 up。

`w2` 保持第二次 quantized matrix multiplication。

decode 的每層 matrix call 數可從 18 降至 12。

這個數字是 call-count 上限。

這個數字不是 33% end-to-end 改善。

這項工作需要新的 expert blob layout、manifest version 與 repack migration。

主要風險是 MXFP4 scale alignment、layout copy、numerical order 與 installed model 相容性。

進入實作的門檻是 routed MoE 占 decode TPOT 至少 25%。

microbenchmark 接受門檻是 routed MoE median latency 至少改善 15%。

end-to-end 接受門檻是 decode Tok/s 至少改善 5%。

### P2a：compile 固定且純 MLX 的區段

優先測試 stable top-k 後處理、route reduction、mHC elementwise 區段與 sparse attention 後處理。

不要把 SSD read、NumPy conversion 或 Python cache mutation 放進 compiled function。

MLX 要求 compiled function 沒有 side effects。

input shape 改變時，MLX 會重新 compile。[MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)

因此，先為 decode shape 與標準 prefill shape 建立固定 entry。

預期影響是 2–8%。

這是未量測假設。

進入實作的門檻是 capture 顯示小型 elementwise kernels 與 dispatch 占 total time 至少 10%。

接受門檻是該 phase 至少改善 20%，且 end-to-end 至少改善 3%。

### P2b：融合 sparse attention 後處理

只在 capture 支持時建立 `mx.fast.metal_kernel`。

候選融合範圍是 packed row gather、MXFP8 dequantize、mask、sink、softmax 與 value reduction。

不要先重寫已優化的 quantized matrix multiplication。

MLX 支持由 Python 建立與重用 custom Metal kernel。[MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

MLX 預設使用 safe math。

mask softmax 需要保留 `-inf` 行為。

主要風險是 top-k tie、mask、sink、accumulation order 與 non-contiguous input copy。

進入實作的門檻是 sparse attention 後處理占 TPOT 或 TTFT 至少 15%。

接受門檻是 phase latency 至少改善 30%。

接受門檻也要求 end-to-end 至少改善 5%。

### P3：Metal I/O prototype

Apple Metal I/O 可直接把檔案資料載入 GPU buffer。

Metal I/O 也可使用 shared events 與 GPU work 同步。[Apple Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading), [MTLIOCommandQueue](https://developer.apple.com/documentation/metal/mtliocommandqueue)

目前 MLX Python path 不公開接管任意 `MTLBuffer` 的簡單介面。

因此，這項工作需要 native extension 或 MLX 變更。

只在 full-layer copy 或 SSD wait 仍占 total time 至少 10% 時研究。

主要風險是 MLX allocator ownership、resource lifetime、synchronization 與 portability。

接受門檻是 expert bytes 不增加，且相關 read/copy phase 至少改善 20%。

### 第一優先：DSpark

DSpark 用 parallel backbone 產生 draft block。

DSpark 再用 lightweight sequential head 建立 token dependency。

DSpark 也用 confidence head 選擇 verification prefix。[DSpark paper](https://arxiv.org/html/2607.05147#S3)

DeepSeek-V4 部署使用三個 MoE draft layers、SWA 128 與最大 block size 5。[DSpark deployment](https://arxiv.org/html/2607.05147#S5.SS1)

目前 installed model 排除約 10.12 GiB `mtp.*` weights。

目前 MXFP8 pooling cache 在已有 compressed entries 後不能 trim。

因此，DSpark 需要 repack 變更、draft execution、verification、cache checkpoint/restore 與 acceptance metrics。

DSpark 現在開始實作。

第一步先固定 DSpark configuration 與 `mtp.*` tensor contract。

第二步加入 DSpark 安裝格式與 cache transaction。

第三步移植 DSpark model 與 greedy scheduler。

最後量測 draft cost、verification cost 與 acceptance。

不要把論文 speedup 直接套用到 Apple Silicon。

完整階段與門檻記錄在 [DSpark-first 實作計畫](DSPARK_FIRST_PLAN_2026-08-08.md)。

## 不優先項目

不要重做 MLX-LM 已有的 next-token `async_eval` pipeline。

新增 GPU stream 也不是預設方向。

Apple Metal 說明 command buffer submission 太頻繁可能造成 CPU stall。[Metal command-buffer best practices](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html)

只有 Metal System Trace 顯示可填補的 GPU idle gap 時，才測試額外 stream。

Accelerate BNNS 目前也不是優先方向。

BNNSGraph 是 CPU-based neural network graph。

BNNSGraph 可做 graph-level fusion 與減少 copy。[Apple BNNS](https://developer.apple.com/documentation/accelerate/bnns-library/)

目前主要 matrix operations 已在 MLX Metal 路徑。

只有 profiling 顯示大型 CPU matrix work，或 MLX 缺少必要 kernel 時，才評估 Accelerate。

## 分階段驗證門檻

### Gate 0：可信 profiling

- profiling overhead 小於 1%。
- phase accounting 至少 90%。
- 256-output 基準各執行五次。
- 2K-output 基準各執行三次。
- 14K context 與 12K output 的穩定性測試至少執行一次。
- 短基準報告 median 與 p95。
- 中型基準報告 median。
- 代表性短基準分開記錄 cold page cache 與 warm page cache。

Gate 0 未通過時，不接受新的效能結論。

### Gate 1：低風險 graph 與 copy 改善

- 完成 batched router-sync 實驗。
- 完成 full-layer copy/no-copy 判定。
- 所有 greedy tokens 相同。
- 所有 route IDs 與 route scores 相同。
- 所有 cache offsets 與 cache state shape 相同。
- peak memory 增加不超過 5%。
- 目標 phase 至少改善 10%。
- end-to-end median 至少改善 3%。

### Gate 2：decode cache policy 與 fused `w13`

- trace simulator 先通過。
- policy 不得修改 router output。
- policy 不得增加任何基準的 miss bytes。
- fused `w13` 通過完整 expert microbenchmark。
- BF16 reference 誤差不超過現有 MXFP4 path。
- greedy output 相同。
- decode median Tok/s 至少改善 5%。
- p95 TPOT 不得惡化超過 2%。

### Gate 3：custom Metal kernel 或 Metal I/O

- Metal capture 先證明該 phase 是 material bottleneck。
- kernel 有獨立 reference test。
- top-k tie 行為相同。
- mask、sink 與 chunk-boundary 行為相同。
- 100 次重複執行沒有 race 或 memory growth。
- 12K output 沒有 resource growth。
- end-to-end median 至少改善 5%。

### DSpark 發布門檻

- 安裝流程可選擇包含 `mtp.*`。
- non-speculative path 保持不變。
- cache 支持 checkpoint 與 rollback。
- 第一版驗證 greedy correctness。
- sampling correctness 延後至第二版。
- 分開報告 draft latency、verification latency、accepted length 與 rejection rate。
- code、Tool-like 與 open chat workload 分開。
- 只有 net Tok/s 為正且 memory 在限制內時，才預設啟用。

## 引用與本地證據

- [本地 runtime research](RUNTIME_RESEARCH_2026-08-07.md)
- [本地 runtime speed research](RUNTIME_SPEED_RESEARCH_2026-08-07.md)
- [本地 validation](../../docs/VALIDATION.md)
- [本地 layer-major prefill](../../runtime/deepseek_v4_ssd/model.py)
- [本地 expert cache](../../runtime/deepseek_v4_ssd/expert_cache.py)
- [本地 MXFP8 pooling cache](../../runtime/deepseek_v4_ssd/fp8_cache.py)
- [本地 generation path](../../runtime/deepseek_v4_ssd/generation.py)
- [MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [MLX Metal Debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX PR #2875](https://github.com/ml-explore/mlx/pull/2875)
- [Apple Metal developer workflows](https://developer.apple.com/documentation/Xcode/Metal-developer-workflows)
- [Apple Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)
- [Apple BNNS](https://developer.apple.com/documentation/accelerate/bnns-library/)
- [DeepSeek-V4 paper](https://arxiv.org/html/2606.19348)
- [DSpark paper](https://arxiv.org/html/2607.05147)
- [Apple SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching)
- [SpecMD paper](https://arxiv.org/html/2602.03921)
