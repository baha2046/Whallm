# Routed expert streaming 研究

> [!WARNING]
> 本文件是 2026-08-09 的歷史研究。外部論文結果只代表研究假設。
> 請以[目前研究結論](../../docs/RESEARCH.md)和
> [驗證紀錄](../../docs/VALIDATION.md)為準。

Checked: 2026-08-09

## 結論

本專案不應先做完整 route tree。完整 tree 只改善猜測，不會減少
expert blob 的大小，也不會改變目前 `get_many()` 等待所有 read 完成的問題。

目前最有價值的研究順序如下：

1. 建立 **prefill-to-decode slot handoff**。這個方法只改 cache 的 admission
   和 pinning。這個方法不改 router，也不增加 SSD bytes。這是本專案最特有的
   高報酬方向。
2. 建立 **per-expert ready scheduler**。目前 runtime 等待所有 missing expert
   完成後才開始 routed expert compute。讓已完成的 expert 先進入計算，可以直接
   重疊 SSD 與 GPU 工作。這個方法不需要預測器。
3. 做 **w13/w2 staged streaming**。把一個 expert 的第一階段權重和第二階段
   權重分開讀取。runtime 在執行 w1/w3 時讀取 w2。這個方法特別適合目前的
   batch size 1 direct path。
4. 做 **MTLIO native prototype**。Metal 官方 API 可以把檔案直接載入
   `MTLBuffer`，也支援獨立 I/O queue、priority、shared event 和取消。這個
   方法可能移除目前的 host buffer 與 MLX upload，但需要 native bridge。
5. 做 **same-layer pre-attention predictor**。這個方法比 token-to-token
   transition predictor 更接近真正的 router input。這個方法需要用
   DeepSeek-V4 trace 訓練 predictor，不能直接套用其他模型的準確率。

以上五個方向都可以保持 native router 和 greedy output 不變。預測錯誤只會
造成額外 read 或 cache pollution。

`AcceptMoE` 應列為另外的非精確模式。它的主要收益來自限制 verifier 的 expert
eligibility set。這個 mask 會改變 target router 的分布。因此，它不符合本專案
目前的 greedy parity 要求。

## 目前架構與限制

本專案的固定模型 contract 是 43 個 main model layers、每層 256 個 routed
experts、每個 token 選 6 個 routed experts。模型也有 3 個 hash-routed layers。
這些值來自固定 revision 的官方 configuration 和本地 validation record。

- [官方 DeepSeek-V4 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
- [本地 model contract](../../Sources/DeepSeekRepack/Model.swift)
- [本地 validation record](../../docs/VALIDATION.md)

每個 installed routed expert blob 是 12.75 MiB。blob 目前依序放置：

```text
w1.weight + w1.scale + w2.weight + w2.scale + w3.weight + w3.scale
```

本地 cache 使用 512 個固定 slot。每個 slot 保存一個完整 expert blob。cache
使用一個 global LFU heap，並保留每層 reserve。[expert cache](../../runtime/deepseek_v4_ssd/expert_cache.py#L171-L211)

目前 decode 的流程是：

```text
native gate
  -> mx.eval(indices)
  -> get_many()
  -> 等待所有 missing read
  -> mx.array() 和 mx.eval()
  -> 執行 selected experts
```

`get_many()` 會先提交所有 missing read，再對每個 future 呼叫
`future.result()`。runtime 之後才把 blob 存入 slot。[get_many](../../runtime/deepseek_v4_ssd/expert_cache.py#L330-L390)

`_SlotPool.store()` 目前會建立 NumPy view，再建立 MLX array，最後同步
`mx.eval()`。[slot store](../../runtime/deepseek_v4_ssd/expert_cache.py#L131-L135)

layer-major prefill 的 batched path 會讀取整層 256 個 expert，建立一個
`BatchedExperts` view。context 結束時，runtime 只清除 `_batched_layer`。它不會
把這個完整層中的 prompt hot experts promotion 到一般固定 slot。[batched layer](../../runtime/deepseek_v4_ssd/expert_cache.py#L293-L318)

這是本研究最重要的專案特有缺口。

目前實測基線如下：

- 8K optimized prefill：65.47 秒，expert bytes 488.8 GB，cache hit rate
  58.7%。
- 14K batched layer-local prefill：76.54 秒，expert bytes 150.41 GB。
- 8K run 的 SSD read 約 14.0 秒。
- 4 個 read workers 的 direct SSD benchmark 是 14.30 GiB/s。
- 1,024 到 1,536 slots 的已測試變更沒有改善總時間。2,048 slots 因 memory
  pressure 變慢。

來源：[optimized prefill](../../docs/VALIDATION.md)、
[batched layer-local measurements](../../docs/VALIDATION.md)、
[SSD measurements](../../docs/VALIDATION.md)、
[已測試的 slot sweep](RUNTIME_RESEARCH_2026-08-07.md#implementation-progress)。

因此，本研究不再建議單純增加 slot 數，也不再建議 layer-order eviction。

## 已排除的方向

下列方向已經做過、測過，或已有本專案研究結論。本文件不把它們當成新方案：

- layer-major prefill。
- 1,024-token attention step。
- 4,096-token MoE tile。
- full-layer prefetch。
- strided full-layer `gather_qmm`。
- FP4 lightning-index cache。
- MXFP8 KV cache。
- in-memory 和 persistent prompt cache。
- decode direct six-expert path。
- 增加 slot 數。
- layer-order eviction。
- 只用相鄰 token 或 route transition 的機率預測。
- 只增加 GPU stream。

這些項目的結果記錄在 [prefill/decode research](PREFILL_DECODE_RESEARCH_2026-08-08.md)
和 [runtime speed research](RUNTIME_SPEED_RESEARCH_2026-08-07.md)。

DSpark 已有獨立的 implementation plan。DSpark 仍然值得單獨評估，但本文件只
把它當作 verifier-stage 的使用場景，不重複 DSpark 的整體實作計畫。

## 評估標準

每個 output-preserving 方案都必須符合下列條件：

1. native router 仍決定最後的 selected expert IDs。
2. prediction 只能提供 read hint、slot admission 或 read priority。
3. prediction miss 必須退回 native on-demand read。
4. required read 的優先權必須高於 speculative read。
5. expert bytes、slot state 和 GPU buffer lifetime 必須可追蹤。
6. greedy token ID 必須和目前 baseline 完全相同。

研究階段先記錄這些指標：

- 每個 layer、每個 token 的 selected expert IDs。
- predicted recall@6、ready recall@6 和 prediction miss bytes。
- cache hit、miss、eviction、prefetch hit 和 bytes read。
- read queue wait、GPU wait、slot promotion copy time。
- w13 ready time、w2 ready time 和 routed expert compute time。
- MTLIO command buffer status、cancel count 和 shared-event wait time。
- p50 和 p95 TPOT。
- greedy token parity。

## 方向一：prefill-to-decode slot handoff

### 研究想法

在 prefill 期間收集每層的 expert activation histogram。prefill 完成後，
runtime 將 prompt hot experts 放入一般 active expert slots，並在前幾個 decode
token 暫時 pin 這些 slots。

這個方法不是猜下一個 expert。這個方法使用已經發生的 native routing。
runtime 不修改 route 結果。runtime 只改變哪一些 expert 被保留。

ELDR 的核心觀察是：prefill expert activation 和 decode expert activation
在多個 MoE model 之間有明顯 correlation。ELDR 將 per-layer prefill counts
整理成 expert signature，並用 signature 進行 decode worker locality routing。
ELDR 保持 model、gate、kernel 和 batching 不變，因此 output 不變。[ELDR abstract and invariance](https://arxiv.org/html/2607.00466#S1)

另一篇 activation-pattern 研究也在 DeepSeek-V3、Llama-4 Maverick 和
Qwen3-230B-A22B 上觀察到 domain-specific activation，以及 prefill 和 decode
activation 的 correlation。[activation patterns](https://arxiv.org/html/2604.23150#S3.SS3)

這些數字不能直接套用到 DeepSeek-V4-Flash-0731。它們只支持「值得對本模型
建立 trace」這個研究假設。

### 本專案的適配方式

本專案可以把 ELDR 的 worker routing 改成 local slot policy：

```text
prefill route IDs
  -> per-layer histogram
  -> top-H hot set
  -> promote from current BatchedExperts to fixed slots
  -> pin for first N decode tokens
  -> native route miss falls back to normal cache policy
```

`batched_layer()` 已經在 layer-local path 讀取完整 256-expert layer。完整層的
packed array 在 context 內仍然可用。runtime 可以在 context 結束前，將 hot set
的 raw byte ranges 複製到一般 slot。這樣不需要再次從 SSD 讀取 expert。

一個 layer 的候選數量和 memory 成本如下：

| 每層 promotion 數量 | 所需 logical experts | 估計 slot memory |
| ---: | ---: | ---: |
| 6 | 258 | 3.21 GiB |
| 8 | 344 | 4.28 GiB |
| 10 | 430 | 5.35 GiB |

512 slots 可以容納每層 6 個 expert 的完整 handoff set。每層 8 個候選仍會
留下 168 個 dynamic slots。第一版應測試 `H=6` 和 `H=8`，不要直接 pin 全層。

runtime 必須把 prompt signature 一起放入 prompt cache metadata。signature 可以
先使用 `uint16[43][256]`，大小約 22 KiB。persistent cache 必須加入：

- checkpoint revision。
- signature format version。
- prompt cache block token range。
- per-layer histogram 或已計算 hot set。
- partial prefix hit 的 block-level merge rule。

ELDR 也指出，prefix cache 若只保存 KV 而不保存 activation signature，partial
prefix hit 會產生不完整 signature。因此，本專案必須讓 signature 和 prompt
cache entry 同步失效。[ELDR prefix-cache coherence](https://arxiv.org/html/2607.00466#S3.SS5)

### 收益

- 可避免 prompt 完成後第一批 decode 的 cold slot miss。
- 不需要 route predictor，也不需要額外 SSD bytes。
- 可重用 layer-major prefill 已經載入的完整 expert layer。
- 可以和 current LFU 共存。pin 只在短時間生效。
- native router 和 greedy output 完全不變。

這個方向的收益上限取決於 prefill-to-decode correlation。對完全不同的 prompt
或 prompt route 幾乎均勻的 layer，收益可能接近零。對長 prompt、同一 domain 或
重複 prompt，收益可能明顯。

### Trade-off

- promotion 需要 GPU copy。每層 6 個 expert 約 76.5 MiB，43 層約 3.21 GiB。
- promotion 會暫時增加 active memory。copy 完成後才能釋放 full-layer packed
  array。
- pin 太多候選會壓縮 dynamic slots，可能降低長 decode 的 hit rate。
- 目前 layer-major cache-only prefill 排除最後一個 prompt token，且最後一層
  不執行 MoE。signature 必須標記 coverage，不能假設 43 層都完整。
- prompt signature 需要新的 persistent metadata 和 invalidation rule。
- ELDR 的結果來自多 worker serving。batch size 1 的 local slot 效益必須用
  DeepSeek-V4 trace 驗證。

### 建議 gate

先不改 runtime。先收集真實 route trace，離線比較：

- current LFU。
- prefill top-H pinning。
- top-H pinning 加短期 expiry。
- top-H pinning 加 native route feedback。

只有在 4K、8K、14K prompt 和至少 2K output 都達到下列條件時才實作：

- handoff promotion 不增加 SSD bytes。
- first 256 decode token 的 miss bytes 降低至少 15%。
- 沒有一個 benchmark 的 total time 增加超過 2%。
- greedy output 的 token IDs 完全相同。

## 方向二：same-layer pre-attention predictor

### 研究想法

在同一 layer 的 attention 開始前，使用 pre-attention normalized activation
預測同一 layer 的 MoE expert。GPU 執行 attention 時，CPU 或另一個低成本 GPU
path 開始讀 predicted experts。attention 完成後，native gate 仍然決定真正的
selected experts。

`Pre-Attention Expert Prediction and Prefetching` 提出每層的 lightweight
predictor。論文使用兩個 linear layers 和 ranking-aware loss。論文的理由是
pre-attention activation 比前一層 activation 更接近同層 router decision，並且
可以處理第一層 prefetch。[paper motivation and method](https://arxiv.org/html/2511.10676#S1)

論文在 DeepSeek-V2-Lite、Qwen3-30B 和 Phi-mini-MoE 報告 93.03%、94.69% 和
97.62% 的 exact-match prediction。這些結果不是 DeepSeek-V4 結果。論文也指出
用 10 個候選取代 6 個候選時，DeepSeek-V2-Lite 的 hit rate 提升到 98.65%，但
I/O 增加約 67%。[reported accuracy and I/O trade-off](https://arxiv.org/html/2511.10676#S5.SS2)

SpecPrefetch 使用另一種較小的 low-rank adapter。adapter 只預測 next-layer
expert priority，native router 仍負責最後 routing。論文明確將 prediction 只
用於 asynchronous transfer，因此 prediction miss 不改變 output。[SpecPrefetch native-router contract](https://arxiv.org/html/2607.24787#S3.SS1)

### 本專案的適配方式

本專案的 DeepSeek-V4 gate 位於 attention 後的 MoE block。same-layer predictor
需要使用每層 attention 前的 normalized activation `X`，並為每一個 non-hash
layer 訓練 predictor。

前三個 layer 是 hash-routed layer。這三層不應使用 learned predictor。token ID
和官方 `tid2eid` table 已經提供 exact route；它們應使用下一節的 exact hash
prefetch。

對其餘 layer，第一個 prototype 應採用：

```text
pre-attention X
  -> predictor top-H candidates
  -> low-priority SSD prefetch
  -> native attention
  -> native gate top-6
  -> hit uses resident slot; miss uses high-priority on-demand read
```

predictor 必須使用 DeepSeek-V4-Flash-0731 的 pinned checkpoint 和實際 prompt
trace 訓練。不能使用其他 model 的 predictor weights。

### 兩階段預取，避免污染 active slot

預測器不應直接把所有候選寫入 active slot。runtime 應分成兩個階段：

```text
low-confidence candidate
  -> bounded staging buffer 或作業系統 page cache
  -> native router 確認
  -> active slot

high-confidence candidate
  -> 可用 active slot
  -> native router 確認或丟棄
```

本地測量顯示，4 個 workers 的 cached read 平均是 0.73 ms。direct read 平均是
3.30 ms。因此，即使預取結果只進入作業系統 page cache，native route miss 的
後續讀取仍可能縮短。[SSD measurements](../../docs/VALIDATION.md)

這個設計把兩種資源分開管理：

- active slot 保存已確認或高 reuse 的 expert。
- staging buffer 保存尚未確認的 candidate。

錯誤預取仍會消耗 SSD bandwidth 和 page-cache memory。但是，錯誤預取不會直接
evict active slot。第一版應使用有上限的 staging byte budget，並且讓 required
read 可以取消或略過 speculative copy。

如果 per-layer two-linear predictor 的 common memory 太大，可以先比較：

- per-layer two-linear predictor。
- shared low-rank adapter。
- CPU-only linear predictor。
- 只保留 top-1 或 top-2 high-confidence candidate。

### 收益

- same-layer signal 比 route transition 更接近真正的 routing input。
- 可以從第一個 non-hash layer 開始，不必等待上一層 route。
- prediction 可在 attention 計算期間取得 SSD overlap window。
- native router 不變。錯誤只增加 read 或 cache pollution。
- 可以和 MTLIO low-priority queue 組合。

如果 DeepSeek-V4 的 recall 接近論文報告值，這個方向可能比 token transition
predictor 更有用。實際收益仍受 SSD read time、attention duration 和 slot
eviction 限制。

### Trade-off

- 需要收集 route labels 和離線訓練流程。
- per-layer predictor 會增加 common tensor memory。論文的 two-linear architecture
  使用 2,048 hidden dimension；DeepSeek-V4 的 4,096 hidden dimension 可能產生
  可見的 predictor weights。
- predictor activation clone、CPU inference 和 synchronization 會消耗 overlap
  window。
- high candidate count 會增加 SSD bytes。低 recall 會污染 slot。
- hash layer 不適合直接使用 learned predictor。
- 論文使用 NVIDIA GPU 和其他 model。Apple Silicon、MXFP4、43-layer DeepSeek-V4
  必須重新測量。

### 建議 gate

先離線計算每層：

- recall@6。
- recall@8。
- top-1 hit rate。
- prediction latency。
- ready recall，也就是 expert 在 native gate 到達前是否已 ready。
- extra SSD bytes。

建議進入 native prototype 的門檻是 non-hash layers 的 recall@6 至少 85%、
ready recall 至少 70%、extra SSD bytes 不超過 15%。prototype 必須保持所有
greedy token IDs 不變。

## 方向三：Apple MTLIOCommandQueue 直接 SSD 到 MTLBuffer

### 官方能力

Apple 的 Metal resource loading 文件說明，Metal 3 可以使用專用 I/O queue，
在可用時直接將檔案載入 GPU resources。文件也說明 unified memory Apple
Silicon 是目標使用情境之一。[Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)

`MTLIOCommandBuffer.load(buffer:offset:size:sourceHandle:sourceHandleOffset:)`
可以把檔案範圍直接載入 `MTLBuffer`。[load buffer](https://developer.apple.com/documentation/metal/mtliocommandbuffer/load%28_%3Aoffset%3Asize%3Asourcehandle%3Asourcehandleoffset%3A%29)

官方 API 也提供：

- concurrent `MTLIOCommandQueue`。
- `maxCommandsInFlight` 和 `maxCommandBufferCount`。
- high、normal 和 low queue priority。[queue descriptor](https://developer.apple.com/documentation/metal/mtliocommandqueuedescriptor)
- `signalEvent` 和 `waitForEvent`。
- completion handler、status 和 error。
- `tryCancel()` 和 `MTLIOStatus.cancelled`。[I/O command buffer](https://developer.apple.com/documentation/metal/mtliocommandbuffer)

### 本專案的適配方式

每一個 physical expert slot 需要對應一個固定 MTLBuffer range。對 expert
`(layer, expert)`，其 source offset 是：

```text
expert * 13,369,344 bytes
```

native bridge 可以建立兩個 queue：

```text
requiredQueue: high priority
  exact on-demand expert reads

speculativeQueue: low priority
  predictor prefetch and handoff candidates
```

每次 slot fill 建立一個 I/O command buffer：

1. load expert blob 到 slot 的 MTLBuffer offset 0。
2. signal shared event `readyValue`。
3. MLX/Metal compute command buffer wait for `readyValue`。
4. compute 完成後才允許 slot reuse。

如果 speculative candidate 已經不需要，可以呼叫 `tryCancel()`。completion
handler 必須檢查 `complete`、`cancelled` 和 `error`，不能只依賴 Python future
完成。

### 為什麼這可能比目前路徑好

目前 path 是：

```text
os.pread -> Python bytes -> bytearray -> NumPy view -> mx.array -> mx.eval
```

MTLIO prototype 可以測試：

```text
MTLIO file handle -> MTLBuffer slot -> quantized matmul
```

這個方向不需要預測器。它也不會增加 expert bytes。若 host staging copy 或
MLX upload 是主要成本，收益可能高於單純提升 prediction recall。

### 主要阻礙

MLX Python API 沒有本專案可直接使用的「把任意外部 MTLBuffer 包成
`mx.array`」介面。此方向需要 Swift、Objective-C++ 或 C++ extension，並且要
確認：

- MTLBuffer 的 storage mode 是否符合 MLX 使用方式。
- buffer offset 和 MXFP4/E8M0 alignment 是否符合要求。
- MLX array 是否會再次複製 MTLBuffer。
- `gather_qmm` 的 strided view 是否可直接指向 slot buffer。
- shared event 是否能和 MLX 的 command queue 正確同步。
- MTLIO file handle 是否能穩定讀取 installed model 的外部 SSD files。

Apple 文件保證 API 的 resource loading contract。文件沒有保證 MLX 能對
任意 MTLBuffer 建立 zero-copy array。因此，這個方向必須先做 native
microbenchmark，不能先改 Python cache。

### 收益與 trade-off

優點：

- 不依賴 route prediction accuracy。
- 可以對 required read 使用 high priority。
- 可以取消 stale speculative read。
- 可以用 shared event 讓 I/O 和 GPU compute 同步。
- native router、expert bytes 和 greedy output 不變。

缺點：

- 需要跨 Swift/Python/MLX 的 native bridge。
- queue priority 是 queue-level policy，必須用多個 queue 管理 required 和
  speculative work。
- slot reuse 會有 command buffer lifetime race。
- unified memory 可能令 direct load 的收益低於離散 GPU。
- MTLIO 直接載入 GPU resource 不代表 MLX graph 會自動接受該 resource。

8K optimized run 的 SSD read 約 14.0 秒，總時間約 65.5 秒。即使完整移除
這段等待，該測試的總時間上限也只是約 21.4%。這是上限，不是 MTLIO 的預測
收益。[local 8K measurement](../../docs/VALIDATION.md)

### 建議 gate

先建立不接 MLX 的 native benchmark，對 32 個 12.75 MiB ranges 比較：

- `pread` 到 host buffer。
- MTLIO 到 shared MTLBuffer。
- MTLIO 到 private MTLBuffer。
- required queue 和 speculative queue 的 queue depth。
- cancel 前後的 SSD bandwidth。

只有在 MTLIO 的 destination buffer 可被 MLX 使用，且 Metal capture 證實移除
一個可見 copy 或 wait，才進入 model integration。

## 方向四：w13/w2 staged streaming 與 variable-granularity slot cache

### 研究想法

每個 routed expert 的執行順序是：

```text
w3(up) + w1(gate)
  -> activation
  -> w2(down)
```

本地 direct path 正是依序執行這三次 MXFP4 matmul。[decode expert path](../../runtime/deepseek_v4_ssd/model.py#L249-L259)

目前 blob 把 w2 放在 w1 和 w3 中間。可以把 installed expert layout 改成：

```text
w13 region: w1.weight + w1.scale + w3.weight + w3.scale = 8.50 MiB
w2 region:  w2.weight + w2.scale                         = 4.25 MiB
```

總 bytes 仍是 12.75 MiB。這不是模型量化，也不是 expert pruning。

### pipeline

對每一個 missing expert，runtime 可以使用兩階段 pipeline：

```text
read w13(A)
  -> w1/w3 matmul(A)
  -> activation(A)
  -> read w2(A) concurrently with w13(B) or compute(B)
  -> w2 matmul(A)
```

第一版不要把 w1 和 w3 的 arithmetic 融合成新 kernel。第一版只重排 bytes、
分割 read range、並保持目前三次 `quantized_matmul` 的順序。這樣比較容易保留
greedy output。

ProMoE 的原始研究也把 expert transfer 切成 chunk，並使用 early preemption
和 reordered inference 來減少 transfer 在 critical path 的時間。它的硬體和
本專案不同，但它支持「read task 不應該等整批完成」這個系統假設。[ProMoE chunked prefetch](https://arxiv.org/html/2410.22134#S4.SS3)

### variable-granularity slot cache

目前每個 slot 固定保存一個完整 blob。新 cache 改用兩個 byte-budget arena：

```text
w13 arena: map[(layer, expert)] -> w13 slot
w2 arena:  map[(layer, expert)] -> w2 slot
```

兩個 arena 共用一個 memory budget，但可以有不同 capacity。runtime 可以：

- 保留高 reuse 的 w13。
- 對低 reuse 的 w2 使用 ephemeral buffer。
- 只在 w13 hit、w2 miss 時讀 4.25 MiB。
- 只在 w2 hit、w13 miss 時讀 8.50 MiB。
- 對目前正在計算的 w13、activation 和 w2 加 temporary pin。

第一版只對 batch size 1 direct decode 使用。現有 full-layer batched prefill
保持完整 layer packed path，避免同時改兩條 execution path。

### 收益

- w2 read 可以和 w13 compute 重疊。
- partial hit 時只讀缺少的 region。
- memory budget 可以依 region size 和 reuse 分配，不再被完整 blob stride
  綁定。
- SSD bytes 在 cold miss 不增加。
- 保持原始 FP4 bytes 和原始三個 matmul，具備 output parity 的可能。

本專案的單層 direct six-expert compute 約 0.56 ms。每個 w2 region 約 4.25 MiB。
本地 direct SSD 速度是 14.30 GiB/s。這表示單次 w2 read 的純頻寬時間約為
0.29 ms；是否能被 w13 compute 隱藏，必須用 Metal capture 測量。[direct SSD and direct expert measurements](../../docs/VALIDATION.md)

### Trade-off

- repack plan、manifest、installed model format 和 verifier 都要增加 version。
- 每個 expert 可能需要兩個 read task，增加 syscall 或 MTLIO command 數量。
- w13 和 w2 可能被不同 request 共享。錯誤的 arena budget 會造成 partial
  miss，反而增加 read。
- activation buffer 必須在 w2 ready 前保持有效。
- slot eviction 不能覆寫正在被 GPU 使用的 w13 或 w2。
- full-layer gather 需要新的 strided layout validation，不能只修改 direct path
  而忽略 prefill。
- 若 w13 compute 太短，staging 只增加複雜度，不會隱藏 read time。

### 建議 gate

先用現有 expert trace 和 microbenchmark 建立四種比較：

1. 完整 blob read，現有 direct path。
2. w13 read 完成後再 read w2。
3. w13 和 w2 read 同時提交，但 w2 compute 延後。
4. w13/w2 partial arena，使用相同 byte budget。

進入 runtime prototype 的條件：

- decode routed MoE phase 至少占 TPOT 的 20%。
- staged path 不增加 cold expert bytes。
- w2 wait 至少降低 20%。
- median decode Tok/s 至少改善 5%。
- greedy token IDs 完全相同。

## 方向五：per-expert ready scheduler、early preemption 和 reordered compute

這是支援前四個方向的 runtime 基礎。

目前 `get_many()` 等待所有 missing futures 後才呼叫 `_pool.store()`。如果六個
selected experts 中有一個 read 較慢，其他五個 ready expert 也必須等待。

新的 scheduler 可以：

1. 先查詢 resident experts。
2. 為每個 missing expert 建立獨立 read task。
3. 某一個 expert ready 後立即填入 temporary slot。
4. 立即開始該 expert 的 w1/w3 compute。
5. 保存每個 expert 的 output。
6. 最後依 selected expert 順序執行 weighted reduction。

計算順序可以改變，但 reduction 應保持 deterministic selected order。第一版
不要改變 reduction tree，也不要讓 cache eviction 覆寫仍被 GPU 使用的 slot。

ProMoE 的原始實驗指出，naive prefetch 會延後 missing expert handling，而
early preemption 和 reordered inference 可以顯著改善 critical path。[ProMoE ablation](https://arxiv.org/html/2410.22134#S5.SS4)

本方向的優點是：

- 不需要 learned predictor。
- 不需要改 router。
- 不需要增加 SSD bytes。
- 可直接和 MTLIO required queue、w13/w2 staged read 組合。

主要風險是 MLX lazy evaluation、slot lifetime 和 output accumulation order。
這個方向應先做 trace-driven scheduler prototype，再改真正的 Metal path。

## 方向六：DeepSeek-V4 hash-layer exact prefetch

官方 DeepSeek-V4 configuration 定義 3 個 hash-routed layers。本地
`MoEGate` 對 hash layer 使用 token ID 和 `tid2eid` table。這些 layer 的 selected
expert IDs 不需要預測，只要 token ID 已知就可以 exact lookup。[official checkpoint configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)

對 batch size 1 decode，runtime 可以在 token 開始時：

```text
known input token ID
  -> exact tid2eid lookup for layers 0, 1, 2
  -> requiredQueue prefetch
  -> normal model forward
```

這個方向不會產生 prediction miss，也不會改 native gate output。它只覆蓋
43 層中的 3 層，因此預期收益有限，但可以作為低風險 baseline。

## 方向七：AcceptMoE 作為非精確模式

AcceptMoE 針對 speculative decoding verification block 建立 layer- and
block-specific eligible expert set。它用 commitment probability 排序 expert
demand，並可依 expert residency 移除低 demand 的 nonresident experts。[AcceptMoE selector](https://arxiv.org/html/2608.02989#S4.SS1)

論文報告，在 physical expert offloading 下，AcceptMoE 將 host-to-device
traffic 降低 73.6% 到 77.1%，並取得約 2.06x 相對 Standard SD 的 throughput。
[AcceptMoE offloading results](https://arxiv.org/html/2608.02989#S6)

但是，AcceptMoE 對 router logits 加入 eligibility mask。論文明確指出，這個
mask 會改變 target router distribution，因此它是 approximation，而不是
distribution-preserving optimization。[AcceptMoE output caveat](https://arxiv.org/html/2608.02989#S4.SS1)

分類如下：

| 用法 | SSD traffic | router/output | 本專案結論 |
| --- | --- | --- | --- |
| native route + AcceptMoE eligibility mask | 大幅下降 | 改變 | 只可作 opt-in approximate mode |
| native route + AcceptMoE priority ordering | 可能改善 ready time | 不變 | 可作 speculative scheduler 實驗，但不會得到完整 traffic reduction |
| native route + AcceptMoE residency-aware cache admission | 可能改善 slot hit | 不變 | 可以加入 DSpark verifier trace simulator |

本專案的 greedy exact mode 不應啟用 eligibility mask。若未來加入 approximate
mode，API 必須明確顯示 output parity 不再保證。

## 綜合優先順序

| 優先級 | 研究 | 需要訓練 | 需要 native bridge | 是否保持 router/output | 主要收益來源 |
| --- | --- | ---: | ---: | --- | --- |
| P0 | route trace 和 ready-time instrumentation | 否 | 否 | 是 | 讓所有方向可量測 |
| P1 | prefill-to-decode slot handoff | 否 | 否 | 是 | 不增加 SSD bytes 的 cache hit |
| P1 | per-expert ready scheduler | 否 | 否 | 是 | SSD 和 compute overlap |
| P1 | w13/w2 staged streaming | 否 | 否 | 是 | partial read 和 w2 overlap |
| P2 | MTLIO direct SSD-to-MTLBuffer | 否 | 是 | 是 | 移除 host staging 和提高 I/O control |
| P2 | same-layer pre-attention predictor 加兩階段預取 | 是 | 否 | 是 | 在 attention 期間預熱，不先污染 active slot |
| P3 | hash-layer exact prefetch | 否 | 否 | 是 | 3 個 layer 的零錯誤 prefetch |
| P4 | AcceptMoE eligibility mask | 否 | 可能 | 否 | 減少 verifier expert union 和 traffic |

這些收益不能直接相加。MTLIO、ready scheduler 和 w13/w2 是資料搬移層的
不同實作。same-layer predictor 和 prefill signature 主要改善 ready time 或
cache admission。AcceptMoE 則改變模型運算條件。

## 最小實驗計畫

### E0：只加 trace，不改執行順序

**Implemented:** CLI 現在可以使用 `--expert-route-trace PATH`。runtime 只在啟用
這個選項時記錄 prefill histogram 和 decode route。正常路徑不會把 batched
route 搬到 CPU。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model scratch/deepseek-v4-flash-0731.dsv4 \
  --prompt "PROMPT" \
  --max-tokens 256 \
  --expert-route-trace scratch/expert-routes.json

PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.route_trace \
  scratch/expert-routes.json \
  --hot-per-layer 6 8 \
  --decode-tokens 256
```

simulator 會報告 slot 數、slot bytes、route coverage、baseline miss bytes 和
handoff recoverable miss bytes。runtime 會在每次 decode route 記錄當時的真實
cache miss。recoverable miss bytes 仍是 counterfactual。handoff 改變 eviction
後，實際 cache state 可能不同。

對每個 token、layer 記錄：

- router selected IDs。
- prefill histogram。
- slot state。
- per-expert read start、ready、store 和 compute start。
- w13/w2 region access。

trace 版本相對 baseline 的 total time 差異必須小於 1%。

### E1：offline handoff simulator

**Preliminary result:** 2026-08-09 的第一個實驗使用 4,096-token repeated prompt、
64-token greedy output、512 slots 和停用 DSpark。兩次 trace 的 SHA-256 完全相同。

| Policy | Slot bytes | Route coverage | Baseline miss bytes covered |
| --- | ---: | ---: | ---: |
| top-6 per layer | 3.21 GiB | 69.1% | 9.0% |
| top-8 per layer | 4.20 GiB | 78.1% | 14.8% |
| top-10 per layer | 5.17 GiB | 83.0% | 21.4% |

這個結果不支持立即實作 handoff。top-6 和 top-8 的 route coverage 很高，但
current LFU 已經命中大部分熱門 expert。top-10 只留下約 97 個 dynamic slots，
可能造成更多 eviction。這個結論只適用於目前的 repeated prompt 和前 63 個
decode forwards。下一步仍需測試不同 domain 和 256-token output。

使用真實 prefill histogram 和後續 decode route，重播：

- current LFU。
- top-6 pin。
- top-8 pin。
- top-8 pin 加 expiry。
- signature metadata partial-prefix merge。

先以 miss bytes 和 first-256-token TPOT 排序。不要先改 cache implementation。

### E2：native MTLIO microbenchmark

先不接 MLX。比較 host `pread`、MTLIO shared buffer 和 MTLIO private buffer。
量測 required/speculative priority、cancel、queue depth 和 shared event wait。

### E3：staged direct-expert prototype

**Implemented first stage:** ready expert decode 現在預設啟用。CLI 和 server 可以
使用 `--no-ready-expert-decode` 建立 baseline。runtime 會先提交所有 missing
expert reads。resident expert 和先完成讀取的 expert 會先提交 GPU compute。
runtime 最後仍依 native router 順序建立 output。

第一組配對 A/B 使用相同 4,096-token repeated prompt 和 64-token greedy output：

| Mode | Decode Tok/s | Decode time | Token SHA-256 |
| --- | ---: | ---: | --- |
| baseline | 10.26 | 6.14 s | `90760d2b...fdd9b9` |
| ready expert decode | 11.87 | 5.31 s | `90760d2b...fdd9b9` |

ready expert decode 在這一組 A/B 改善 15.7% Decode Tok/s，並減少 13.6% decode
time。這個結果只有一次配對。兩次 run 的 prefill SSD time 明顯不同，因此不能
比較 total request time。runtime 會讓 ready expert 的 cache admission order
跟隨 read completion order。這個順序可能改變後續 LFU state，但不改 native
router 或本次 greedy output。

後續五組 4,096-token prompt、256-token output A/B 使用 repeated、code、繁體中文
技術內容、English prose 和 mixed math。五組 token hash 全部相同。

| 結果 | 數值 |
| --- | ---: |
| Decode Tok/s improvement median | 12.9% |
| Minimum improvement | 8.1% |
| Maximum improvement | 14.0% |
| Aggregate decode time reduction | 10.5% |

2,000-token stability A/B 使用 repeated prompt。baseline 是 10.67 Tok/s。ready
expert decode 是 11.92 Tok/s，改善 11.7%。兩個 run 都生成完整 2,000 tokens，
token hash 完全相同。ready run 沒有 resource error。這些結果通過預設啟用 gate。

只改 batch size 1 direct path。保留 full-layer batched prefill。先禁止
variable arena eviction，使用固定 w13/w2 temporary buffers 確認 pipeline。

### E4：predictor offline validation

對 DeepSeek-V4-Flash-0731 重新訓練 same-layer predictor 或 low-rank adapter。
先報 recall 和 ready recall，再決定是否加入 runtime。模擬器必須分開比較
「candidate 直接占用 active slot」和「candidate 先進入 bounded staging」兩種
policy。

### E5：完整 parity matrix

每個候選方案都比較：

- cold page cache 和 warm page cache。
- 4K、8K 和 14K context。
- 256 output 和 2K output。
- p50 和 p95 TPOT。
- expert bytes、SSD read time、cache hit rate、peak memory。
- output token ID、final text 和 cache state。

## 最終建議

若只能選一個新研究，選 **prefill-to-decode slot handoff**。它最貼合本專案
的 active slot 設計。它使用已載入的 full-layer data，不需要預測器，不需要
增加 SSD traffic，也不改 output。它的風險主要是 promotion copy 和 prompt
signature coverage，兩者都可以用 offline simulator 先量測。

若 handoff 的 correlation 不足，第二選擇是 **w13/w2 staged streaming 加
per-expert ready scheduler**。這組合不依賴 prediction accuracy，並直接處理
目前 `get_many()` 等待全部 read 的 critical path。

MTLIO 應先做 native microbenchmark。MTLIO 是可能改變資料搬移成本的高上限
方案，但 MLX buffer ownership 是主要未知數。same-layer predictor 應在 trace
證明 ready recall 足夠後再訓練。AcceptMoE 只應作明確標示的 approximate
verifier mode。
