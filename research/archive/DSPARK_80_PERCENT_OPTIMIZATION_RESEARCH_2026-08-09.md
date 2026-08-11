# DSpark 80% 速度目標最佳化研究

> [!WARNING]
> 本文件是 2026-08-09 的歷史研究。文件描述的 per-position cache copy
> 已由目前的 round-level cache fork 取代。請以
> [目前研究結論](../../docs/RESEARCH.md)和
> [驗證紀錄](../../docs/VALIDATION.md)為準。

日期：2026-08-09

## 結論

目前 DSpark 還沒有達到本機的 80% 速度目標。最新三次測試的 normal Decode 中位數是 5.39 Tok/s。DSpark 中位數是 5.14 Tok/s。DSpark 目前比 normal Decode 慢約 4.6%。

論文中的「提升 60%–85%」不是 batch size 1 Apple Silicon 的固定倍率。該數字是 DeepSeek-V4-Flash production live traffic 中，DSpark-5 與 MTP-1 在相同 aggregate throughput 下的 per-user generation speed 比較。論文沒有把這個結果定義成單一請求的 Decode Tok/s。

本機若把 80% 定義成 `5.39 * 1.8`，目標是約 9.70 Tok/s。最新測試的 DSpark 每回合接受 5 個 draft token，並提交 1 個 target bonus token。因此，一個回合最多提交 6 個 token。每回合總時間必須小於：

`6 / 9.70 = 0.619 秒`

目前實測回合約 0.84–1.24 秒。相同 acceptance 下，回合時間必須降低約 26%–50%。這是 batch size 1 Apple Silicon 的 stretch target，不是論文 production 結果的直接重現。

最高優先工作是移除 target verification 的 per-position cache materialization 和 cache copy。現在一個 6-token verification block 最多執行 43 層 × 6 個 position 的 `mx.eval(output, layer_cache.state)`。FP8 cache 的 `state` 會解量化並串接所有 compressed cache chunks。runtime 也會為每個 position 複製各層 cache。這兩項工作比 confidence threshold 更可能造成主要延遲。

建議順序如下：

1. 先新增 per-round、per-layer、per-cache 的計時與計數。
2. 先禁止 verification hot loop 讀取 persistence `state`。
3. 以 round-level copy-on-write fork 取代每個 position 的 cache checkpoint。
4. 驗證 cache transaction 正確後，再做 block attention 和 batched MoE。
5. 最後才調整 draft head、SSD residency、confidence scheduler 和 block size。

以下預估值是工程排序用的範圍。各項改善不能直接相加。每一項都必須以同一組 prompt 和相同輸出 token ID 驗證。

## 1. 官方結果的正確解讀

### 1.1 論文的 80% 不是單請求 80%

DSpark 論文的延遲模型是：

`L = (T_draft + T_verify) / tau`

其中 `tau` 是每個回合的期望 committed token 數。論文的 production 結果有以下條件：

- 比較對象是 MTP-1。
- 結果來自 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro 的 live traffic。
- scheduler 同時管理多個 active requests。
- scheduler 依硬體實測的 SPS(B) 和 confidence prefix survival 選擇 verification budget。
- V4-Flash 在 80 tok/s/user 的 SLA 下，aggregate throughput 提升約 51%。
- 在 matched practical aggregate capacities 下，V4-Flash 的 per-user generation speed 提升約 60%–85%。
- 論文也報告 120 tok/s/user SLA 的名目 +661%，但該點處於很小 baseline batch，論文明確說不適合作為代表性倍率。
- 論文描述低於約 200 個 Flash concurrent requests 時，verification budget 會從 MTP-1 的 2 增加到約 4–6；負載升高時 budget 會下降。

因此，本專案不能把「官方 80%」直接轉成單一 Apple Silicon request 的 1.8 倍。若要驗證相同 claim，runtime 必須支援並量測 concurrent requests、aggregate throughput、per-user latency、SLA 和 dynamic verification budget。

論文沒有公布本專案所需的 Apple Silicon、SSD、MLX 或 memory layout 條件。研究文件不能假設論文使用的 GPU 型號或記憶體配置。

### 1.2 官方程式的可用證據

官方 DeepSpec repository 是 reference training/evaluation stack。它不是 DeepSeek production serving runtime。它可以用來確認演算法和 cache commit 行為。

DeepSpec evaluator 將 `draft_token_count + 1` 個 token 一次送入 target model，接著計算 longest accepted prefix、correction 或 bonus token。每回合結束後，它用 `crop` 將 target cache 保留在 committed position。DSpark draft cache 也用 `crop` 移除 speculative suffix。

官方 DeepSeek-V4-Flash-DSpark model code 使用 DSpark sparse attention。它預先建立 top-k index，並在 decode block 中一次處理 query KV。該程式不是本專案 MLX SSD runtime 的直接 drop-in implementation，但它指出目前 generic dense attention 可能不是最終成本。

官方 vLLM DSpark implementation 使用以下 production 技術：

- preallocated top-k index buffer；
- paged KV cache 和 fused KV insertion；
- flattened variable-length request batch；
- fixed query layout 和 CUDA graph；
- batch 內一次處理多個 request 的 draft block；
- 可選的 draft top-k，避免對完整 vocabulary 重複套用 Markov bias。

官方 MLX-LM speculative path 將整個 verification block 放入一次 target forward。它在回合結束後使用 `trim`，而不是為每個 token 建立完整 cache copy。MLX `RotatingKVCache` 已有可裁切的 logical offset/index，但本專案的 DeepSeek compressed/index cache 還需要 transaction API。

## 2. 本機測試與指標定義

### 2.1 最新測試

以下數值來自本次 runtime 實測。測試使用相同 prompt、greedy decode 和相同輸出設定。

| 路徑 | 三次 Tok/s | 中位數 | 其他觀察 |
| --- | --- | ---: | --- |
| normal Decode | 5.11、5.39、5.52 | 5.39 | baseline |
| DSpark | 6.64、5.14、4.45 | 5.14 | 三次 acceptance 都是 100% |

DSpark 三次測試平均每回合提出 5 個 draft token。draft 時間約 0.18–0.53 秒/request。verification 時間約 8.70–13.35 秒/request。以完整請求的回合數換算，主要回合約 0.84–1.24 秒。第二次測試有 fallback。第三次測試雖然沒有 fallback，但速度更低。這表示目前的 runtime decision 受單次 target baseline、cache 狀態和 warm-up 影響。

### 2.2 accepted length 和 committed length 不同

目前 metrics 的 `dspark_average_accepted_length` 是：

`accepted_draft_tokens / rounds`

它不包含每回合的 target bonus/correction token。論文的 `tau` 和 throughput 計算包含該 token。為了避免錯誤比較，metrics 應新增：

`committed_tokens_per_round = (accepted_draft_tokens + rounds) / rounds`

在沒有 EOS 或 max-token 截斷時，完整接受 5 個 draft token 的 committed length 是 6，而不是 5。現有 `dspark_seconds_per_output_token` 已使用：

`(draft_seconds + verification_seconds) / (accepted_draft_tokens + rounds)`

該欄位較接近論文的 latency 定義。報告必須同時列出 draft acceptance、committed length 和 seconds per committed token。

### 2.3 Decode Tok/s 和 request latency

本專案的 `decode_tokens_per_second` 與 DSpark 的 `dspark_seconds_per_output_token` 不應直接視為同一個欄位。兩條路徑的前兩個 token、prefill、target anchor step、fallback 和 cache persistence 成本不同。

每次報告至少列出：

- warm steady-state Decode Tok/s；
- `request_seconds / generation_tokens`；
- `time_to_first_token_seconds`；
- `draft_seconds`；
- `verification_seconds`；
- committed tokens per round；
- fallback 次數；
- main model 和 DSpark expert cache read/hit/miss；
- exact greedy token ID equality。

## 3. 本機瓶頸證據

### 3.1 Verification 每個 position 都同步 cache

目前 [verification_forward_with_hidden](../../runtime/deepseek_v4_ssd/model.py#L588-L654) 依 layer 和 position 執行：

```text
for each of 43 target layers:
    for each token in [anchor, draft tokens]:
        attention(token, cache)
        mx.eval(output, layer_cache.state)
        copy checkpoint for rollback
```

6-token input 會產生最多 258 次 position-level cache evaluation。每次 loop 也呼叫 `create_attention_mask` 和一次 attention。

### 3.2 FP8 `state` 會重建完整 pooled cache

在目前預設 `fp8_kv_cache=True` 時，[MXFP8PoolingCache.state](../../runtime/deepseek_v4_ssd/fp8_cache.py#L228-L264) 會呼叫 `_fetch`。`_fetch` 會：

1. 對所有 `_chunks` 執行 FP8 dequantize。
2. 把所有解量化 chunks concatenate。
3. 再回傳 pending cache 和 previous window metadata。

因此，verification hot loop 的 `mx.eval(output, layer_cache.state)` 不只是讀取 offset。它可能在每個 position 重新建立完整 compressed cache view。這是目前最強的 P0 證據。

Inference hot loop 不應讀取 persistence `state`。runtime 應新增只回傳運算需要資料的 `eval_arrays()` 或等效 transaction view。該 API 不得呼叫完整 `_fetch`，也不得在每個 token 解量化全部 chunks。

`state` 可以保留給 prompt cache persistence、debug dump 和 request 結束時的 checkpoint。它不應是 verification 的同步介面。

### 3.3 Per-position cache copy 成本高

目前 [_copy_layer_cache](../../runtime/deepseek_v4_ssd/model.py#L657-L680) 會對每一個 checkpoint：

- `copy.deepcopy(layer_cache)`；
- 複製 keys、values、buffer、previous window arrays；
- 對複製的 arrays 呼叫 `mx.eval`。

該函式在每一層、每一個 rollback position 建立資料。它的工作量與 43 層和 draft length 相乘。這不是必要的 speculative decoding 語意。官方 DeepSpec 和 MLX-LM 都使用 crop/trim 或 cache commit。

### 3.4 Verification 的 MoE 沒有使用 batched `gather_qmm`

目前 [_StreamingSwitchGLU](../../runtime/deepseek_v4_ssd/model.py#L196-L310) 只有在 `ExpertCache.current_batched()` 有值時才使用 `mx.gather_qmm`。該 context 主要由 prefill 使用。Verification path 沒有建立 batched layer context，因此最新 metrics 的 `request_gather_qmm_calls` 是 0。

在 multi-token verification 中，目前程式會：

1. 先把 token route 依 expert 分組。
2. 依 unique expert 逐組切出 source。
3. 對每一個 expert 執行 W1、W3、activation、W2 的 `quantized_matmul`。
4. 再把各組結果 concatenate 和 restore order。

43 層的 unique expert 數量會使 Metal operation 數量快速增加。這個成本可能大於數學 FLOPs。

目前 [_streaming_moe](../../runtime/deepseek_v4_ssd/model.py#L325-L336) 也會先同步 `indices`。該 routing sync 必須獨立計時，不能和 MoE matmul 混在一起。

### 3.5 Draft head 造成 host sync

目前 [DSparkModel.draft](../../runtime/deepseek_v4_ssd/dspark.py#L230-L289) 在每個 Markov position 呼叫 `mx.argmax(...).item()` 或 `mx.random.categorical(...).item()`。這是 Markov recurrence 的必要順序，但每次 `.item()` 都可能造成 CPU/GPU synchronization。

目前每個 position 也先做完整 `logsumexp`。Greedy API 不需要完整 normalized log probability。若 API 不要求 logprobs，greedy draft 和 greedy verifier 可以只保留 argmax 路徑。該變更必須保留 logits dtype、tie rule 和 token ID equality。

### 3.6 DSpark attention 與官方 path 不同

目前 [_DSparkAttention](../../runtime/deepseek_v4_ssd/dspark.py#L46-L116) 會將 context KV 和 draft KV concatenate，然後呼叫 generic `scaled_dot_product_attention`。官方 DeepSeek-V4 DSpark code 使用 top-k sparse attention，並預先建立 index。vLLM 也使用 preallocated top-k buffer、fused KV insertion 和 paged cache。

這是 P3 工作。P0/P1 尚未證明前，不應直接改寫 attention。先用 profiler 確認 dense attention 是否佔據回合時間。

### 3.7 SSD 和 context KV 需要分開量測

目前 DSpark 有獨立 ExpertCache。這可以避免 DSpark routed expert 直接驅逐 main model 的 slot，但每個 round 仍可能引入 SSD read、pack 和 eviction。

目前 [DSpark attention](../../runtime/deepseek_v4_ssd/dspark.py#L76-L101) 的 `prefill_context` 和 `__call__` 都會執行 `_kv` 和 context cache update。這可能是設計需要，也可能在某些回合重算同一段 main context。需要以 `wkv` invocation、context cache length 和 accepted position 記錄確認，不能先假設是 bug。官方 vLLM 將 context KV precompute 和 query block 分開，該設計可作為比較基準。

### 3.8 目前無法直接重現 production scheduler

[ModelRuntime](../../runtime/deepseek_v4_ssd/generation.py#L353-L385) 以 `_generation_lock` serialize generation。[verification_forward_with_hidden](../../runtime/deepseek_v4_ssd/model.py#L594-L600) 也只接受 batch size 1。

因此目前不能重現論文的多 request Hardware-Aware Prefix Scheduler。batch size 1 的 confidence threshold 只能縮短 draft prefix。它不能增加 acceptance，也不能把單一請求變成 production scheduler。

## 4. P0：先建立可證明的計時基線

在修改數學路徑前，先加入 per-round trace。每個 round 至少記錄以下欄位：

- `round_id`、start position、draft length、accepted draft length；
- `committed_tokens = accepted + 1`；
- draft wall time 和 verification wall time；
- attention、FFN、MoE routing sync、MoE matmul、cache state fetch、cache copy、rollback/replay 的時間；
- `mx.eval` 次數和評估 array bytes；
- 每層 cache length、FP8 `_chunks` 數、pending length、packed cache 是否重建；
- main model/DSpark expert cache hit、miss、eviction、bytes read、read seconds、pack seconds；
- round 前後 peak memory；
- fallback reason；
- target logits argmax 和 final emitted token ID。

測試要分成 cold、warm compile、warm SSD cache 三種狀態。每種狀態至少使用 5 個 prompt、每個 prompt 產生 256 個 token，並交錯執行 normal 和 DSpark。第一個請求不能代表 steady state。結果要報 median 和 p95。

P0 的完成條件如下：

- normal 和 DSpark 的 greedy token ID 完全相同；
- `state_fetch_seconds` 和 `cache_copy_seconds` 可單獨看到；
- 每個 round 的 committed length 可與論文公式對照；
- 能分辨 SSD、MoE、attention、cache 和 host sync 的成本。

## 5. P1：round-level cache transaction

### 5.1 低風險第一版：完整 fork，拒絕時重播 prefix

第一版不要立刻實作任意 prefix delta commit。保留 round 起點的 base cache，建立一份 round-level copy-on-write fork：

1. Round 開始時，base cache 保持不變。
2. Target 先沿用目前的 sequential verification。所有 speculative writes 都進入 fork。每層只在必要邊界評估 cache arrays。
3. 若 draft 全部接受，直接 adopt fork。
4. 若只接受 prefix，discard 完整 fork。
5. 從未修改的 base 建立新 fork，對 `[anchor, accepted draft tokens]` 執行 cache-only prefix replay，再 adopt 新 fork。
6. 若 acceptance 很低，直接使用 normal decode 或現有安全路徑，避免 fork 加 replay 反而變慢。

目前測試是 100% acceptance。因此完整接受路徑可以先證明「一次 fork、沒有 per-position copy」的收益。P2 再將 sequential verification 改成 block verification。低 acceptance 的 replay 成本在後續階段再最佳化。

### 5.2 Transaction API 的資料要求

每個 main model layer 的 transaction 必須保存：

- committed logical length；
- write cursor 或 ring index；
- pending raw KV/gate；
- completed compressed chunk 數；
- FP8 和 FP4 packed cache pointer；
- previous window KV/gate；
- mask cache 所需 metadata；
- `CacheList` 內所有子 cache 的同一個 transaction boundary。

對 `RotatingKVCache`，若 verification 只寫入未使用的尾端，rollback 可只恢復 offset/index。若 verification 會繞回 ring buffer 並覆寫 committed data，runtime 必須改用 touched-block copy-on-write 或 round fork。不能只保存一個整數 offset。

對 `MXFP8PoolingCache`，commit/rollback 必須能截短 `_pending`、`_chunks`、`_index_chunks` 和 `_length`，並在必要時使 `_packed_cache`、`_packed_index_cache` 失效。rollback 不能重新呼叫 `state` setter 來解量化和重建全部 chunks。

不要使用 shallow Python copy 取代 transaction。MLX array 的 graph value 可能仍被兩個 cache 物件共用。任何可變 metadata 和 write target 都必須經過明確的 ownership 檢查。

### 5.3 P1 的 hot-path 規則

- verification 不呼叫 `layer_cache.state`。
- verification 不呼叫 `_copy_layer_cache`。
- verification 不在每個 token 呼叫 `mx.eval`。
- transaction 只在 round begin、round commit、round rollback/replay 評估必要 arrays。
- debug mode 可以保留完整 state compare，但 production hot path 必須關閉。

P1 的成功條件：

- greedy token ID 與目前 reference 完全相同；
- 100% acceptance workload 不再建立每-position checkpoint；
- `cache_copy_bytes` 和 per-position `state_fetch` 接近 0；
- verification median 至少降低 30%，或明確證明其他成本佔比超過 70%；
- peak memory 不超過目前 memory limit。

停止條件：cache rollback 產生任何 token mismatch、compressed cache metadata 不一致、或 full fork 的記憶體超過專案限制。此時保留現有安全路徑，不能用不完整 rollback 取代它。

## 6. P2：一次 block verification

P1 證明 transaction 正確後，再把 target attention 從 per-position loop 改成 block query。目標是讓 target 對 `[anchor, draft]` 使用一次 layer forward 和一次 batched attention。FFN 已經對整個 `hidden` block 執行；主要缺口是 attention cache update。

DeepSeek-V4 index/compressed attention 需要特別處理以下項目：

- query position 的 causal mask；
- local window 和 pooled index mask；
- block 內 token 之間的可見性；
- cache append 順序；
- FP8 pending window 和 compression ratio；
- `CacheList` 內不同 cache 的相同 offset。

先以 `block_size=5` 和 batch size 1 實作。每一層先比較 batched path 與目前 sequential reference 的 logits、hidden、cache metadata。確認 parity 後才開啟 production flag。

Greedy path 可以使用 target `argmax` 直接驗證 draft token。只有 sampling 或 API 要求 logprobs 時，才計算完整 normalized logprobs。這可以移除 verification 內不必要的 `logsumexp`。

P2 的目標是讓 verification p50 降到 0.45–0.50 秒，使完整回合有機會低於 0.619 秒。該數字是本機 stretch target。若 batched attention 不能保持 token equality，回到 P1，不得以降低精度掩蓋 mismatch。

## 7. P3：verification MoE 使用 slot-aware `gather_qmm`

目前 `_SlotPool` 已將每個 slot 保存成一個 packed Metal array，但 `_StreamingSwitchGLU` 在 verification 中仍逐 expert 執行三個 `quantized_matmul`。建議保留 SSD slot cache，並增加 block verification 專用的 batched path：

1. 將 slot pool 組成一個連續的 `[slot_count, expert_blob]` Metal array，或建立可證明不複製的 slot slice/reshape view。
2. `ExpertCache.get_many` 同時回傳 logical expert ID 到 physical slot ID 的 mapping。
3. Gate indices 只同步一次。
4. 將 physical slot indices 傳給 W1、W3、W2 的 `mx.gather_qmm`。
5. 每層保留三個主要 quantized matmul dispatch，不依 unique expert 數量增加 dispatch。
6. 只讀取當前 route 使用的 slot，不載入整層 256 個 expert。

這個設計保留目前 memory-bounded SSD streaming。它不要求 DSpark routed expert 全部常駐，也不要求 main model 和 DSpark 共用同一個 cache。

需要用 Metal allocation 和 operation trace 驗證 slice/reshape 沒有偷偷建立完整 copy。若 MLX `gather_qmm` 不能接受 slot dimension 的 physical index，先保留現有 per-expert path，並記錄原因。

P3 的成功條件：

- verification metrics 的 `request_gather_qmm_calls` 從 0 變成每層固定數量；
- quantized matmul dispatch 不再隨 unique expert 數量線性增加；
- target token ID、hidden 和 cache parity 不變；
- MoE verification time 至少降低 20%。

## 8. P4：降低 draft 和 attention 成本

### 8.1 Draft Markov loop

Markov head 的 token dependency 不能完全移除。可以先做以下低風險修改：

- greedy 時移除完整 `sampling_logprobs` normalization；
- 將每個 position 的 `.item()` 改成一次 batch/stream evaluation，或使用固定 block 的 compiled path；
- 只有 sampling path 才保留 top-p 和 random categorical；
- 只有 metrics 或 API 要求時才保留每個 token 的 logprobs；
- 用固定 `block_size=5` 預先配置 token、confidence 和 Markov buffers。

預估 draft time 可降低 20%–40%，但這不是主要第一優先，因為最新測試顯示 verification 成本更大。若 draft acceptance 下降超過 5 個百分點或總 TPS 下降，立即回退。

### 8.2 Context KV 和官方 sparse attention

先記錄每回合 `_kv`、context cache length、top-k index 以及 DSpark expert read。若確認 main context KV 被重複 projection，將 context KV precompute 與 draft query 分開保存。

接著才評估將 generic dense attention 改成官方 DeepSeek-V4 DSpark 的 sparse attention。該工作可能需要 MLX custom Metal kernel。它不能只將 Python `scaled_dot_product_attention` 替換成另一個名稱，必須同時驗證 top-k index、window mask、position offset 和 output parity。

## 9. P5：SSD expert cache 與 memory trade-off

比較以下三個模式：

1. DSpark common tensor 和 routed expert 全部常駐。
2. DSpark common tensor 常駐，routed expert 使用目前獨立 ExpertCache。
3. DSpark 與 main model 使用共享 slot pool，但保留 layer reserve 和 eviction priority。

每個模式都記錄：

- DSpark `hit_rate`、miss、eviction；
- bytes read、read seconds、pack seconds；
- main model 的 hit rate 和後續正常 decode miss；
- peak memory；
- cold/warm SSD 的 p50/p95 round time。

不要只用 SSD bandwidth 決定模式。DSpark verification 可能把專家載入 main model cache，造成下一個 normal token 的 cache miss。若 DSpark read time 超過 round time 的 5%，先做 layer prefetch 或調整 admission priority。若 DSpark 常駐使 peak memory 接近 limit，保留獨立 SSD cache。

## 10. P6：confidence 與 block length

論文的 confidence 是條件接受機率。完整 prefix survival 是各 position confidence 的累積乘積。論文 production scheduler 另外使用實測 SPS(B)，而不是只使用固定 threshold。論文也使用 Sequential Temperature Scaling 校準 raw confidence。

本專案目前 batch size 1。`dspark_confidence_threshold=0.6` 只能把 draft 變短。它不能直接接受 token。最新 workload 的 acceptance 已是 100%，因此 threshold 不會帶來品質收益，反而可能減少 committed token。

在 P1–P4 完成後，建立 batch size 1 的成本表：

| proposal length | draft time | verification time | expected committed tokens | expected tokens/sec |
| ---: | ---: | ---: | ---: | ---: |
| 0 | measure | measure | measure | measure |
| 1 | measure | measure | measure | measure |
| 2 | measure | measure | measure | measure |
| 3 | measure | measure | measure | measure |
| 4 | measure | measure | measure | measure |
| 5 | measure | measure | measure | measure |

選擇使 `expected committed tokens / measured round time` 最大的 prefix length。只在代表性 prompt 上計算 confidence calibration。若 ECE 高，先做 STS，再使用 cumulative survival 做決策。

不要在 verification 未最佳化前調大 block size。先測 3、5、7，但只以 end-to-end committed Tok/s 決定。acceptance rate 不能單獨決定 block size。

## 11. P7：若要複製 production 80%，需要 concurrency

若目標是論文的 production claim，而不是單請求 9.70 Tok/s，runtime 需要新增另一個 scope：

- request queue 和多 request scheduler；
- variable-length draft block flattening；
- batch-aware SPS(B) profile；
- request-level KV slot allocation 和 commit/rollback；
- dynamic verification budget；
- aggregate throughput、per-user throughput、SLA latency 的同時量測。

這個 scope 不應和 M2 batch size 1、M3 SSD streaming、M4 DSpark decision 混在同一次微最佳化。單 request path 先達到穩定結果，再決定是否投入 production scheduler。

## 12. 建議的驗收門檻

每個 phase 都使用以下共同門檻：

- greedy output 的完整 token ID 與 reference 完全相同；
- 固定 checkpoint revision 和固定 tensor layout；
- warm steady-state median 不劣於 phase 前；
- p95 不因 SSD、cache rollback 或 host sync 顯著惡化；
- peak memory 不超過專案設定；
- request metrics 能區分 draft、verification、cache、MoE 和 SSD。

本機 80% stretch target 的硬門檻是：

`normal median 5.39 Tok/s -> DSpark median >= 9.70 Tok/s`

若 committed length 不是 6，則使用實測 committed length 重新計算每回合時間上限：

`round_time_limit = committed_tokens_per_round / 9.70`

只有在至少 5 個代表性 prompt、每個 prompt 256 output tokens、warm steady state、greedy parity 都通過時，才可以說 DSpark 在本機接近該目標。任何單次 6.64 Tok/s 的結果都不能代表 80% improvement。

## 13. 官方來源

研究只使用官方或原作者來源。以下連結是 2026-08-09 使用的入口；`main` 分支可能在未來變更。

### DeepSeek DSpark

- [DSpark paper](https://arxiv.org/html/2607.05147)：延遲公式、confidence prefix survival、Hardware-Aware Prefix Scheduler、production 結果和限制。
- [DeepSpec official repository](https://github.com/deepseek-ai/DeepSpec)：reference training/evaluation stack。
- [DeepSpec target verifier](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/eval/base_evaluator.py)：一次 block verify、acceptance、correction/bonus、cache crop。
- [DeepSpec DSpark draft operations](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/eval/dspark/draft_ops.py)：parallel draft block 和 draft cache crop。
- [DeepSeek-V4-Flash-DSpark model code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/aa22cb07426656189b2573b8e77a9b7333b8ae0f/inference/model.py)：DSpark sparse attention、top-k index 和官方模型結構。

### vLLM official implementation

- [vLLM DeepSeek-V4 DSpark model](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/nvidia/dspark.py)：fused attention、preallocated index、KV insertion 和 DSpark layers。
- [vLLM DSpark speculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)：parallel draft、Markov sampling、optional draft top-k 和 batch buffers。
- [vLLM DFlash speculator](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py)：flattened query layout、preallocated buffers 和 graph dispatch。
- [vLLM speculative decoding guide](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)：硬體、traffic 和 sampling 對實際收益的限制。

### MLX official implementation

- [MLX-LM speculative generation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)：target block forward、cache trim 和 draft/target cache 分離。
- [MLX-LM cache implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)：`RotatingKVCache.trim`、offset 和 trimmable cache。
- [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)：Apple Silicon unified memory 行為。

## 14. 本機來源

- [verification implementation](../../runtime/deepseek_v4_ssd/model.py#L588-L680)
- [FP8 pooling cache](../../runtime/deepseek_v4_ssd/fp8_cache.py#L228-L285)
- [streaming MoE](../../runtime/deepseek_v4_ssd/model.py#L196-L336)
- [DSpark draft and attention](../../runtime/deepseek_v4_ssd/dspark.py#L46-L289)
- [DSpark verification loop and fallback](../../runtime/deepseek_v4_ssd/dspark.py#L497-L613)
- [runtime DSpark metrics](../../runtime/deepseek_v4_ssd/generation.py#L97-L350)
- [runtime batch-size and generation lock](../../runtime/deepseek_v4_ssd/generation.py#L353-L385)
- [existing runtime optimization notes](DSPARK_RUNTIME_OPTIMIZATION_2026-08-08.md)
