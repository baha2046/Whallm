# Prefill 與 decode runtime 最佳化研究

> [!WARNING]
> 本文件是 2026-08-09 的歷史實作紀錄。測試數量和部分預設值已變更。
> 請以[目前研究結論](../../docs/RESEARCH.md)和
> [驗證紀錄](../../docs/VALIDATION.md)為準。

日期：2026-08-09

## 結論

目前 runtime 已經有有效的 layer-major prefill、batched expert prefill、MXFP8 KV cache，以及 ready expert decode。

下一輪不應先重寫 attention 或加入複雜的 expert predictor。

下一輪應先移除可確認的同步與資料複製。

建議順序如下。

1. runtime 應使用原始量化快取陣列做 `mx.eval()`。
2. runtime 不應在一般生成路徑讀取 `MXFP8PoolingCache.state`。
3. batched prefill 應延後 routed expert index 的 CPU 同步。
4. 預設 `read_workers` 應先回到 4。
5. runtime 應補上完整的端到端 decode 指標。
6. runtime 應用原生 MLX buffer 移除 full-layer expert 的 staging copy。
7. decode 應保留 ready expert 路徑。
8. DSpark 應繼續預設停用。

本機方向性測試顯示，原始快取陣列可讓 4K TTFT 降低約 2.4%。

同一個測試讓 metrics 合計的 decode 工作時間降低約 4.2%。

這個結果只有一次 prototype 測試。

正式變更仍需要 paired benchmark。

## 研究範圍

本報告分析目前工作樹。

目前工作樹以 commit `5e0ec7f` 為基礎。

目前工作樹也包含尚未提交的 runtime 變更。

本報告沒有修改 runtime 程式。

測試環境如下。

| 項目 | 值 |
|---|---:|
| 裝置 | Apple M5 Pro，20-core GPU，64 GB unified memory |
| 系統建議工作集 | 約 51.84 GiB |
| MLX | 0.32.0 |
| mlx-lm | commit `5c10538136b9038b9626c134612b08afc18d697a` |
| checkpoint revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| main model layer | 43 |
| routed expert | 每層 256 個 |
| selected expert | 每 token 6 個 |
| expert blob | 12.75 MiB |
| 測試模式 | batch size 1、greedy decode |

本機 Python 測試共 55 個。

所有測試都通過。

執行命令如下。

```sh
PYTHONPATH=runtime .venv/bin/python -m unittest discover -s runtime/tests -p 'test_*.py' -v
```

## 目前資料路徑

### Prefill

```text
layer expert file
  -> 3.1875 GiB bytearray
  -> mx.array copy
  -> 3 個 gather_qmm
  -> cache.state 評估
  -> 下一層
```

[layer_major_prefill](../../runtime/deepseek_v4_ssd/model.py#L64) 會逐層執行 attention 和 MoE。

[ExpertCache.prefetch_layer](../../runtime/deepseek_v4_ssd/expert_cache.py#L307) 會先配置一個 full-layer `bytearray`。

[ExpertCache.batched_layer](../../runtime/deepseek_v4_ssd/expert_cache.py#L337) 再用 `mx.array(np.frombuffer(...))` 建立 MLX array。

本機 mutation test 顯示，這個 Python 路徑會複製資料。

每個 full expert layer 是 3.1875 GiB。

42 個 batched layer 會讀取 133.875 GiB 的 expert 資料。

### Decode

```text
gate
  -> mx.eval(indices)
  -> CPU cache lookup
  -> SSD read for misses
  -> ready expert qmm
  -> cache.state 評估
  -> token
```

[_streaming_moe](../../runtime/deepseek_v4_ssd/model.py#L334) 會在每一層執行 `mx.eval(indices)`。

[iter_ready](../../runtime/deepseek_v4_ssd/expert_cache.py#L440) 已經可以讓 resident expert 和已完成讀取的 expert 先開始計算。

[ModelRuntime.stream](../../runtime/deepseek_v4_ssd/generation.py#L477) 會在每個輸出後評估所有 `cache.state`。

## 本機量測

### 1. 原始量化快取陣列

目前 [MXFP8PoolingCache.state](../../runtime/deepseek_v4_ssd/fp8_cache.py#L249) 會呼叫 `_fetch()`。

`_fetch()` 會解量化所有 FP8 chunk。

`_fetch()` 也會串接完整的歷史快取。

一般生成路徑只需要完成尚未完成的快取 graph。

一般生成路徑不需要建立 persistence state。

目前 DSpark 路徑已經有 [eval_prompt_cache](../../runtime/deepseek_v4_ssd/model.py#L659)。

這個函式會直接評估原始快取陣列。

本機 prototype 在單一程序內讓一般生成路徑使用相同做法。

測試使用 4,096-token repeated prompt。

測試產生 64 個 greedy token。

測試停用 persistent prompt cache。

所有輸出 token 的雜湊相同。

| 變體 | TTFT | request wall time | decode 工作時間 | cache state 評估 |
|---|---:|---:|---:|---:|
| control 中位數，2 次 | 33.60 s | 39.21 s | 5.60 s | 0.545 s |
| 原始快取陣列，1 次 | 32.78 s | 38.15 s | 5.37 s | 0.366 s |
| 方向性差異 | -2.4% | -2.7% | -4.2% | -32.8% |

`decode 工作時間` 是 `decode_seconds + cache_state_eval_seconds`。

這個合計包含第一個 token 的 cache state 評估。

因此，這個合計只用來比較兩個變體。

目前顯示的 `decode_tokens_per_second` 不包含 `cache_state_eval_seconds`。

[RuntimeMetrics.record_token](../../runtime/deepseek_v4_ssd/generation.py#L210) 會把兩段時間分開記錄。

[RuntimeMetrics.snapshot](../../runtime/deepseek_v4_ssd/generation.py#L277) 只用 `decode_seconds` 計算 decode Tok/s。

因此，目前 Decode Tok/s 高於使用者實際取得 token 的速度。

### 2. Batched prefill 的 routed expert index 同步

`gather_qmm` 可以直接使用 MLX array 形式的 routed expert index。

batched prefill 不需要先把 routed expert index 送到 CPU。

一般 decode 仍需要 CPU index。

route trace 也仍需要 CPU index。

本機 prototype 只跳過 42 個 batched layer 的 CPU 同步。

最後一層仍保留 CPU 同步。

測試使用 4,096-token prompt。

測試產生 16 個 greedy token。

三次輸出 token 的雜湊相同。

| 變體 | TTFT | request wall time | SSD read | routing sync |
|---|---:|---:|---:|---:|
| prototype 1 | 32.54 s | 34.70 s | 12.54 s | 0.80 s |
| control | 35.69 s | 37.90 s | 12.98 s | 1.43 s |
| prototype 2 | 35.70 s | 37.89 s | 13.64 s | 0.80 s |

這個 prototype 穩定降低了已記錄的 `routing sync` 時間。

這個 prototype 沒有穩定降低 TTFT。

SSD read 在三次測試之間有明顯波動。

此外，`mx.eval(indices)` 可能同時完成上游 lazy graph。

因此，`routing sync` 不是可以直接相加的獨立成本。

正式變更需要至少 5 組 paired benchmark。

### 3. Greedy decode 的完整 logprobs

目前 pinned mlx-lm 會先計算：

```python
logprobs = logits - mx.logsumexp(logits, keepdims=True)
sampled = sampler(logprobs)
```

greedy decode 只需要 `argmax(logits)`。

減去同一列的常數不會改變 `argmax`。

本機 process-only prototype 跳過 `logsumexp`。

prototype 的 Decode Tok/s 是 8.07。

前後兩次 control 的 Decode Tok/s 是 7.76 和 7.99。

三次輸出 token 的雜湊相同。

這個改善小於本機 I/O 波動。

runtime 可以在完成前兩項工作後再實作 greedy-only generator。

### 4. Expert slot 與 read worker

目前 [RuntimeConfig](../../runtime/deepseek_v4_ssd/model.py#L24) 的預設值是 1,152 slots 和 8 個 read worker。

1,152 slots 需要 14.34 GiB 的 expert slot 容量。

512 slots 需要 6.375 GiB 的 expert slot 容量。

兩個設定相差 7.97 GiB。

既有 [SSD microbenchmark](../../docs/VALIDATION.md) 結果如下。

| read worker | 直接讀取速度 |
|---:|---:|
| 1 | 10.86 GiB/s |
| 2 | 14.04 GiB/s |
| 4 | 14.30 GiB/s |
| 8 | 13.25 GiB/s |

4 個 read worker 在這台機器上最快。

目前資料不支持把 8 設成預設值。

slot 數量則是速度和記憶體的取捨。

runtime 應提供至少兩個明確 profile。

- Memory profile 可以使用 512 slots。
- Throughput profile 可以使用 1,152 slots。

runtime 不應把 1,152 slots 視為所有 prompt 的固定最佳值。

### 5. Ready expert decode

目前 ready expert 路徑是有效的最佳化。

既有 [5-prompt paired benchmark](../../docs/VALIDATION.md) 的改善中位數是 12.9%。

每一組 greedy token 雜湊都相同。

2,000-token repeated prompt 也從 10.67 Tok/s 提升到 11.92 Tok/s。

runtime 應保留 [ready expert decode](../../runtime/deepseek_v4_ssd/model.py#L218)。

### 6. Prefill-to-decode handoff

既有 [route trace](EXPERT_STREAMING_RESEARCH_2026-08-09.md#e1offline-handoff-simulator) 顯示，top-6 handoff 有 69.1% route coverage。

但是，top-6 handoff 只能避免 9.0% 的實際 miss read。

top-10 handoff 有 83.0% route coverage。

但是，top-10 handoff 只能避免 21.4% 的實際 miss read。

route coverage 不等於 miss reduction。

因此，runtime 不應先做複雜的 handoff predictor。

Apple 的 SpecMD 研究也指出，expert temporal locality 會依模型、工作負載和硬體改變。[Apple SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching)

## 建議變更

### P0：先做

#### P0.1 分開 runtime evaluation 與 persistence state

runtime 應新增明確的快取介面。

建議介面如下。

```python
cache.eval_arrays()
cache.persistence_state()
```

`eval_arrays()` 應回傳原始 FP8 chunk、FP4 index chunk、pending array，以及其他目前狀態。

`persistence_state()` 才能解量化或轉換儲存格式。

runtime 應在下列位置使用 `eval_arrays()`。

- [layer_major_prefill](../../runtime/deepseek_v4_ssd/model.py#L64)
- [ModelRuntime.stream](../../runtime/deepseek_v4_ssd/generation.py#L477)
- DSpark verification

runtime 不應直接改變 `state` property 的 persistence contract。

這種改法可以避免載入舊 prompt cache 時產生語意差異。

#### P0.2 修正 decode 指標

runtime 應新增下列指標。

- `decode_end_to_end_seconds`
- `decode_end_to_end_tokens_per_second`
- `expert_upload_seconds`
- `expert_compute_seconds`
- `prompt_cache_snapshot_seconds`
- `prompt_cache_write_seconds`

`decode_end_to_end_seconds` 應包含第二個 token 之後的每 token 快取評估。

第一個 token 的快取評估應計入 TTFT。

使用者介面應顯示端到端 Decode Tok/s。

舊的純 model step 指標可以保留。

舊指標應改名為 `decode_model_step_tokens_per_second`。

#### P0.3 延後 batched prefill 的 CPU 同步

runtime 應只在下列情況執行 `mx.eval(indices)`。

- `current_batched(layer)` 是 `None`。
- runtime 正在記錄 route trace。
- 後續 cache lookup 需要 CPU expert ID。

batched prefill 應保持 routed expert index 在 MLX graph 中。

runtime 應使用 5 組 paired benchmark 驗證這項變更。

#### P0.4 將預設 `read_workers` 設為 4

這個變更只適用於目前 M5 Pro 目標。

runtime 應保留使用者設定。

runtime 應分開調整 `read_workers` 和 `prefetch_read_workers`。

#### P0.5 避免 TTFT 包含 persistence 轉換

layer-major prefill 目前會在第一個輸出前呼叫 [_store_prompt_cache](../../runtime/deepseek_v4_ssd/generation.py#L540)。

[_persist_prompt_cache](../../runtime/deepseek_v4_ssd/generation.py#L822) 會在送交 writer 前讀取所有 `item.state`。

對 MXFP8 cache 而言，這會建立完整的解量化快取。

runtime 應先建立 immutable snapshot。

runtime 應直接把量化 snapshot 交給 writer thread。

writer thread 應只做檔案寫入。

runtime 不應讓 writer thread 在未受控的 MLX stream 執行格式轉換。

更好的做法是直接儲存量化 chunk 和必要 metadata。

這個格式變更必須提高 prompt cache format version。

DeepSeek-V4 的正式設計也使用 on-disk prefix cache。[DeepSeek-V4 inference framework](https://arxiv.org/html/2606.19348#S3.SS5)

正式設計會分開處理壓縮項目、SWA 項目和不完整 tail。

本專案也應保留這個區分。

### P1：P0 穩定後執行

#### P1.1 移除 expert staging copy

目前 full-layer prefill 先讀入 3.1875 GiB `bytearray`。

runtime 再把資料複製到 MLX array。

decode 也有相同問題。

[_SlotPool.store](../../runtime/deepseek_v4_ssd/expert_cache.py#L132) 會把每個 12.75 MiB expert blob 複製到新的 MLX array。

MLX 已經合併 C++ no-copy array constructor。[MLX no-copy array change](https://github.com/ml-explore/mlx/pull/2875)

MLX 也支持 C++ 和 Metal custom extension。[MLX custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)

建議實作順序如下。

1. native extension 使用 MLX allocator 配置目標 buffer。
2. `preadv` 直接把 expert layer 或 expert blob 讀入目標 buffer。
3. MLX array 保持該 buffer 的生命週期。
4. prefill 的 `gather_qmm` 直接使用 full-layer array。
5. decode 的 direct qmm 直接使用 slot array。

runtime 應先完成這個版本。

runtime 之後才應評估 Metal I/O command queue。[Apple Metal resource loading](https://developer.apple.com/documentation/metal/resource-loading)

Metal I/O 需要原生 Metal bridge。

Metal I/O 也需要明確的 event 同步。

#### P1.2 降低 `mx.clear_cache()` 頻率

4K layer-major prefill 目前最多會呼叫約 214 次 `mx.clear_cache()`。

這個數字包含 172 個 attention chunk 和 42 個 MoE layer。

runtime 應比較三種策略。

- 每個 chunk 清除。
- 每個 layer 清除。
- 只有 active memory 超過門檻時清除。

MLX memory cache 會保留可重用的 free memory。[MLX memory cache](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)

`mx.clear_cache()` 會清空該 memory cache。[MLX clear_cache](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clear_cache.html)

過度清除可能增加後續配置成本。

但是，降低清除頻率也可能提高 peak memory。

runtime 應以速度和 peak memory 一起決定策略。

#### P1.3 加入 greedy-only generator

greedy-only generator 應直接執行 `argmax(logits)`。

greedy-only generator 不應建立完整 logprobs。

sampling 模式仍應使用完整 logprobs。

這項變更必須保留 EOS、detokenizer、prompt progress 和 async next-token 行為。

#### P1.4 只編譯形狀穩定的小函式

MLX `compile()` 可以融合 elementwise operation。[MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)

輸入 shape、dtype 或輸入數量改變時，MLX 可能重新編譯。

runtime 應先測試 decode 的固定 shape 函式。

runtime 不應先編譯包含動態 expert residency 的整個 Python loop。

### P2：需要資料格式變更

#### P2.1 建立連續 expert slot arena

目前 [_SlotPool](../../runtime/deepseek_v4_ssd/expert_cache.py#L100) 使用 `list[mx.array | None]`。

每個 slot 是獨立 array。

這個格式不適合對 physical slot 直接執行 batched `gather_qmm`。

runtime 可以建立單一固定 arena。

runtime 也可以維護 `expert ID -> physical slot` 映射。

這個格式可讓多 token verification 直接使用 resident expert。

runtime 必須先比較 single-token direct qmm 和 arena `gather_qmm`。

如果 single-token `gather_qmm` 較慢，runtime 應保留目前 direct qmm 路徑。

#### P2.2 合併 w1 與 w3

目前每個 expert 會分開執行 w3、w1 和 w2 matmul。

runtime 可以在 repack 時把 w1 和 w3 放入同一個連續區段。

runtime 可以用一次 qmm 產生 up 和 gate。

runtime 也可以先讀 w1/w3，再於計算時讀 w2。

這個設計需要新的 expert blob format version。

這個設計也需要 greedy token parity 測試。

#### P2.3 只在 profile 支持時寫 custom Metal kernel

DeepSeek-V4 正式 runtime 使用 fused kernel。

DeepSeek-V4 報告也指出，Python host invocation 會限制小 kernel 的速度。[DeepSeek-V4 host overhead](https://arxiv.org/html/2606.19348#S3.SS2.SSS1)

但是，本專案目前主要成本仍包含 SSD read、buffer copy 和 expert qmm。

runtime 應先使用 Metal capture 找出實際 kernel gap。[MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)

只有 attention 或 route kernel 佔明顯比例時，runtime 才應加入 custom Metal kernel。

DeepSeek-V4 的 sparse attention 也需要 KV layout 和 kernel 一起設計。[DeepSeek-V4 sparse attention](https://arxiv.org/html/2606.19348#S3.SS5.SSS1)

runtime 不應只替換其中一半。

## DSpark 判斷

目前工作樹已經使用 round-level cache fork。

目前工作樹也會一次驗證完整 draft block。

[verification_forward_with_hidden](../../runtime/deepseek_v4_ssd/model.py#L597) 不再為每個位置建立 cache checkpoint。

這是正確方向。

本次 current-runtime 診斷使用 5-token prompt 和 64-token output。

normal 路徑的 Decode Tok/s 是 7.76。

預設 DSpark 在第一回合自動退回 normal decode。

強制提出 5 個 draft token 時，DSpark 接受 2 個 draft token。

該回合共提交 3 個 token。

draft 使用 0.129 秒。

verification 使用 1.048 秒。

該回合平均使用 0.392 秒產生一個 output token。

runtime 因此正確觸發 fallback。

這是單次診斷。

這個結果不應取代完整 DSpark benchmark。

DSpark 論文報告的 60% 至 85% 改善來自 production traffic 和 aggregate throughput。[DSpark paper](https://arxiv.org/abs/2607.05147)

該數字不是 batch size 1 Apple Silicon 的固定倍率。

DSpark 應繼續預設停用。

下一個 DSpark 工作應依序執行下列項目。

1. [_verify](../../runtime/deepseek_v4_ssd/dspark.py#L664) 應一次計算整個 block 的 greedy `argmax`。
2. verification 應使用連續 expert slot arena。
3. rejection replay 應只重算必要前綴。
4. runtime 應用 5 種 prompt 和 256-token output 做 paired benchmark。

## 不建議現在執行的工作

- 不要把 8 個 read worker 設為固定最佳值。
- 不要只靠 route coverage 增加 handoff slot。
- 不要先加入未驗證的 expert predictor。
- 不要先以 MTLIO 取代所有 `preadv`。
- 不要先重寫整個 attention kernel。
- 不要用近似 expert routing 換取速度。
- 不要用只包含 model step 的 Decode Tok/s 判斷使用者速度。

## 驗證計畫

每一項最佳化都應使用相同的 paired benchmark。

每一組應交替執行 control 和 candidate。

每一組至少應執行 5 次。

測試矩陣如下。

| 維度 | 測試值 |
|---|---|
| prompt 長度 | 321、4,096、14,336 token |
| output 長度 | 1、256、2,000 token |
| prompt 類型 | repeated、English、Traditional Chinese、code、math |
| cache 狀態 | cold process、warm compile、warm SSD page cache |
| 模式 | normal、DSpark |

每次測試都應記錄下列資料。

- greedy token ID hash
- TTFT
- 端到端 Decode Tok/s
- p50 和 p95 latency
- SSD bytes read
- SSD read seconds
- expert upload seconds
- routing sync seconds
- expert compute seconds
- cache evaluation seconds
- prompt cache snapshot seconds
- active memory
- cache memory
- peak memory

candidate 必須滿足下列條件。

1. greedy token ID 必須完全相同。
2. candidate 不得提高 OOM 風險。
3. candidate 的改善必須大於 paired run 的波動。
4. candidate 不得讓短 prompt 或長 output 出現明顯 regression。
5. DSpark 只有在 p50 改善且 p95 不退步時才能預設啟用。

## 建議執行順序

### 第一批

1. 新增端到端 decode 指標。
2. 一般生成路徑改用原始快取陣列。
3. batched prefill 延後 CPU routed expert index 同步。
4. 將 M5 Pro 預設 `read_workers` 改成 4。
5. 執行完整 paired benchmark。

### 第二批

1. persistent prompt cache 改存量化 chunk。
2. prefill 和 decode 改用 no-copy MLX buffer。
3. 比較每 chunk、每 layer 和 memory-threshold `clear_cache()`。
4. 加入 greedy-only generator。

### 第三批

1. 建立連續 expert slot arena。
2. 評估 w1/w3 fused repack。
3. 重新量測 DSpark block verification。
4. 使用 Metal capture 決定是否需要 custom kernel。

這個順序先處理已確認的同步和複製。

這個順序也保留目前 memory-bounded SSD streaming 設計。

## 實作結果

本節記錄 2026-08-09 的 runtime 修改與實機驗證。

### 已完成

runtime 已完成下列修改。

1. runtime 使用原始快取陣列做 `mx.eval`。
2. runtime 不再為一般生成重建完整 BF16 pooling cache。
3. persistent prompt cache v2 直接儲存 MXFP8 與 MXFP4 chunk。
4. runtime 在最後一個 token 之後才寫入 persistent prompt cache。
5. runtime 分開記錄 model step、cache eval 和端到端 decode 時間。
6. runtime 記錄 decode p50、decode p95、active memory、cache memory 和 peak memory。
7. batched prefill 不再先把 routed expert index 同步到 CPU。
8. M5 Pro 的預設 `read_workers` 已改成 4。
9. runtime 依 cache memory 用量呼叫 `mx.clear_cache()`。
10. SSD 使用 `preadv` 直接寫入 MLX expert slot buffer。
11. full-layer prefill 使用可直接寫入的連續 MLX buffer。
12. runtime 在讀取時重排 w1 和 w3。
13. batched prefill 用一次 `gather_qmm` 計算 w1 和 w3。
14. DSpark verification 一次計算整個 block 的 greedy `argmax`。
15. DSpark rejection replay 只重算 committed prefix。

### 4K prefill 實機結果

測試使用 4,096-token prompt 和 1-token output。

| 指標 | 結果 |
|---|---:|
| TTFT | 28.89 秒 |
| Prefill | 141.76 Tok/s |
| SSD read | 14.48 秒 |
| Routing sync | 0.078 秒 |
| Batched expert layer | 42 |
| `gather_qmm` call | 84 |
| Active memory | 24.65 GB |
| Peak memory | 36.77 GB |

84 次 `gather_qmm` 代表每個 batched expert layer 使用 2 次 call。

未合併的路徑每個 layer 需要 3 次 call。

### Decode 實機結果

測試使用相同 prompt 和 256-token greedy output。

| slot buffer | 端到端 Decode Tok/s | token hash |
|---|---:|---|
| 單一 15 GB buffer | 5.80 | `85b0e122...a5af42` |
| 每個 slot 一個 direct buffer | 6.39 | `85b0e122...a5af42` |

每個 slot 一個 direct buffer 提高 10.1%。

兩條路徑產生相同 token。

runtime 因此沒有採用單一 15 GB buffer。

單一 Metal buffer 會讓 CPU 寫入與 GPU 讀取共用同一個資源。

這個行為會降低 ready-expert overlap。

### Greedy generator 實機結果

custom greedy generator 是 6.05 Tok/s。

既有 mlx-lm generator 是 6.51 Tok/s。

兩條路徑產生相同 token。

runtime 因此保留既有 mlx-lm generator。

runtime 透過 `_RawEvalCacheList` 讓既有 generator 評估原始快取陣列。

### Persistent prompt cache v2

真實模型的短 prompt cache 檔案是 6.46 MB。

寫入時間是 4.4 ms。

runtime 在最後一個 token 之後執行寫入。

真實模型的載入測試還原了 62 個 quantized pooling cache item。

載入後的 quantized cache 大小是 6.13 MB。

### DSpark

16-token 配對測試產生相同 token hash。

DSpark 第一輪提出 2 個 token。

DSpark 接受 0 個 token。

runtime 正確 fallback 到 normal decode。

DSpark 維持預設關閉。

### 未採用項目

runtime 沒有採用單一 contiguous slot arena。

實機測試顯示這個設計降低 decode 速度。

runtime 沒有採用 custom greedy generator。

實機測試顯示這個設計降低 decode 速度。

runtime 沒有加入 custom Metal kernel。

目前的量測沒有證明 custom Metal kernel 可以改善主要瓶頸。

runtime 沒有加入 handoff predictor。

route trace 顯示 handoff predictor 的預期收益不足。

### 測試結果

- Python：61 個測試通過。
- Swift：20 個測試通過。
- 4K prefill：通過。
- 256-token decode：通過。
- persistent prompt cache v2 真實檔案 round-trip：通過。
- DSpark fallback 與 token hash：通過。

### Codex effort API

Responses API 支援 `reasoning.effort`。

Chat Completions 支援 `reasoning_effort`。

runtime 將 Codex effort 映射到 DeepSeek-V4 官方 encoder 的 `low`、`high` 和
`max`。

真實 encoder prompt 驗證已通過。

### 測試 APP

測試 APP 版本是 1.1.1。

APP bundle 位於 `dist/DeepSeekV4SSD.app`。

ZIP 位於 `dist/DeepSeekV4SSD-macOS-arm64.zip`。

APP 已通過 `codesign --verify --deep --strict`。

ZIP 已通過完整性檢查。

ZIP SHA-256 是 `990cbf6791c6757d707d1c5843289e21f4eb0f8ad2d300e4f15cc070ce96ed19`。

測試 APP 使用 ad hoc 簽章。

測試 APP 沒有 Apple notarization。
