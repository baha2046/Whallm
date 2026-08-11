# Runtime Prefill 與 Decode 最佳化研究計畫

> [!WARNING]
> 本文件已於 2026-08-11 封存。
> 本文件記錄已完成的研究計畫，不描述目前 runtime。
> 目前狀態請使用[研究結論](../../docs/RESEARCH.md)與[效能文件](../../docs/PERFORMANCE.md)。

日期：2026-08-10
最後更新：2026-08-11

## 結論

本計畫已完成 P1 與 P2 的探索性評估。
P1 平均分配 Prefill 區塊會降低速度，並改變 Prefill routed expert histogram。
P2 的兩輪 ABBA 沒有穩定收益。
P2 的配對中位數改善是 1.41%。
P1 與 P2 都沒有達到採用門檻。
runtime 已移除兩個 prototype。

R0 baseline 和 R1 boundary capture 已完成。
R0 顯示 repeated prompt 不能代表一般 Decode。
R1 顯示 Tool-like Decode 有較長的 GPU idle boundary。
R1 沒有讓 D1 或 P5 通過 5% profile gate。

R2a 已拒絕 2,048-slot 候選設定。
R2b 已拒絕 2-worker 候選設定。
R2b 已拒絕 8-worker 候選設定。
R2c 已拒絕 4-prefetch-worker 候選設定。
R2c 已拒絕 1-prefetch-worker 候選設定。
R2 設定 sweep 已完成。
R3 已拒絕目前 DSpark 預設啟用。
M1 的五輪正式效能評估已完成。
正式評估包含 100 個有效量測 run 和 50 個配對。
主要 workload 的 request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。
System-wide compressor 的不增加條件也沒有得到證明。
專案拒絕預設啟用 M1。
runtime 已移除 M1 prototype。
P3 `gather_qmm` tile profiling 已完成。
P3 的原始 GPU time 占比是 12.77%。
依涵蓋率校正後是 13.99%。
兩個結果都低於 15% profile gate。
P3 已停止，且沒有建立 isolated MLX build。
P4 attention 類型區塊探索已完成。
512-step 候選改變 39/42 個 MoE layer 的 expert route。
至少 32,075 個 expert assignment 改變。
P4 沒有通過正確性 gate。
runtime 已移除 P4 prototype。
D2 routed expert top-6 profile 已完成。
dispatch、shader sample 和涵蓋率校正結果分別是 Decode wall time 的
0.121%、0.456% 和 0.972%。
三個結果都低於 3% profile gate。
D2 已停止，且沒有建立 prototype。
D3 CSA row trace 已完成。
五種 context 合併後的 row 交集比例中位數是 85.94%。
`MXFP8PoolingCache.gather` 的涵蓋率校正核心占比是 Decode 的 0.25%。
包含通用 clip 和 select shader 後是 0.34%。
D3 沒有通過 90% row gate 或 5% profile gate。
D3 已停止，且沒有建立 prototype。
目前計畫中的 active trace 與 profile 工作都已完成。
後續 MLX 實驗仍必須先通過各自的 profile gate。

目前不執行 MLX 圖走訪或 `gather_qmm` identity index 修改。
D1 和 P5 必須等待新的 profile 證據。

Decode 的後續研究必須先取得 profile。

1. 如果 MLX 圖走訪占用明顯時間，測試獨立的圖走訪修改。
2. 如果 routed expert 的 top-k 選擇占用明顯時間，才測試專用 top-6 選擇。
3. 如果相鄰 Decode token 的 CSA 選取 row 高度重疊，才測試已解量化 row 的重用。

M1 已測試分階段檔案快取策略。
Process footprint 證據不支持 M1 的記憶體假設。
正式結果也沒有通過全部採用條件。

所有外部效能數字都來自其他模型或其他硬體。這些數字只用來建立假設。本專案必須用固定 checkpoint 重新量測。

## 目標與限制

主要目標如下。

1. 降低 TTFT。
2. 提高 Prefill Tok/s。
3. 提高端到端 Decode Tok/s。
4. 降低 Decode p50 與 p95 token latency。

次要目標如下。

1. 維持或降低作業系統層級的 peak memory。
2. 維持或降低 MLX peak memory。
3. 不增加 swap。

本計畫使用下列固定條件。

- checkpoint revision 不變。
- installed model 格式不變，除非獨立實驗明確要求格式變更。
- batch size 是 1。
- greedy token 序列必須相同。
- DSpark 預設關閉。
- prompt cache 在原始效能基準中關閉。
- 每個候選項目必須可以獨立開啟或關閉。

## 已跳過的研究主題

本計畫已檢查下列文件。

- [Runtime research](RUNTIME_RESEARCH_2026-08-07.md)
- [Runtime speed research](RUNTIME_SPEED_RESEARCH_2026-08-07.md)
- [Prefill 與 Decode 效能研究](PREFILL_DECODE_RESEARCH_2026-08-08.md)
- [Prefill 與 Decode Runtime 最佳化研究](PREFILL_DECODE_RUNTIME_OPTIMIZATION_2026-08-09.md)
- [Routed expert streaming 研究](EXPERT_STREAMING_RESEARCH_2026-08-09.md)
- [DSpark Runtime 最佳化研究](DSPARK_RUNTIME_OPTIMIZATION_2026-08-08.md)
- [DSpark 80% 目標研究](DSPARK_80_PERCENT_OPTIMIZATION_RESEARCH_2026-08-09.md)

下列主題已完成、已測試、已拒絕，或已有專案研究結論。本計畫不重新研究這些主題。

| 類別 | 已跳過的主題 |
|---|---|
| Prefill 結構 | layer-major Prefill、1,024-token attention step、4,096-token MoE tile、full-layer prefetch、batched full-layer `gather_qmm` |
| Expert 資料路徑 | `preadv` 直接寫入 MLX buffer、full-layer 連續 buffer、read-time `w1`/`w3` 重排、融合 `w13`、`w13`/`w2` staged streaming |
| Decode | direct six-expert path、ready expert scheduler、單一 15 GB expert arena、custom greedy generator、額外 GPU stream |
| Expert cache | layer-order eviction、route transition predictor、same-layer predictor、Prefill-to-Decode slot handoff |
| Attention 與快取 | MXFP8 pooling cache、FP4 lightning-index cache、一般 sparse-attention 融合、固定形狀 padded tail |
| Prompt 重用 | in-memory prompt cache、persistent prompt cache、壓縮快取序列化 |
| Native I/O | MTLIO、完整 native I/O queue 設計 |
| 其他 | 一般 `mx.compile` 探索、一般 custom Metal kernel、common tensor 量化 |

「平均分配區塊」不等於先前的固定形狀 padded tail。平均分配不加入 padding。平均分配也不增加區塊數。

「依 attention 類型選擇區塊」不等於先前的全域 step sweep。這個實驗會分開量測 `compress_ratio=0`、`4` 和 `128`。

「Decode 選取 row 重用」不等於重新研究完整 sparse-attention 融合。這個實驗只重用相鄰 Decode token 已經解量化的 row。

## 目前 Runtime 的事實

### 固定模型

目前 installed model 有下列固定值。

| 項目 | 值 |
|---|---:|
| main model layer | 43 |
| routed expert / layer | 256 |
| selected routed expert / token | 6 |
| hash-routed layer | 3 |
| expert blob | 12.75 MiB |
| `hidden_size` | 4,096 |
| `moe_intermediate_size` | 2,048 |
| `index_topk` | 512 |
| sliding window | 128 |
| `compress_ratio=0` layer | 2 |
| `compress_ratio=4` layer | 21 |
| `compress_ratio=128` layer | 20 |

這些值來自固定 revision 的
[checkpoint config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)和
[目前架構](../../docs/ARCHITECTURE.md)。

### Prefill 路徑

Runtime 對至少 4K 的新 prompt 使用 layer-major Prefill。Runtime 對 attention 使用 1,024-token step。Runtime 對 MoE 使用 4,096-token step。

目前兩個 loop 都使用固定 step 和最後餘數。

```text
for start in range(0, token_count, step):
    end = min(start + step, token_count)
```

程式位置如下。

- [attention step 選擇](../../runtime/deepseek_v4_ssd/model.py#L54)
- [MoE step 選擇](../../runtime/deepseek_v4_ssd/model.py#L64)
- [attention loop](../../runtime/deepseek_v4_ssd/model.py#L145)
- [MoE loop](../../runtime/deepseek_v4_ssd/model.py#L182)

這個規則會產生很小的最後區塊。例如，4,097-token MoE 輸入會變成 4,096 和 1。第二個區塊仍然需要執行一次 full-layer `gather_qmm` 路徑。

每個 4,096-token MoE 區塊有 24,576 個 routed row。平均每個 routed expert 有 96 個 row。

```text
4,096 tokens * 6 selected experts / 256 experts = 96 rows/expert
```

目前 batched Prefill 的 `w13` 和 `w2` 每層執行兩次 `gather_qmm`。[batched routed expert path](../../runtime/deepseek_v4_ssd/model.py#L271)

### Decode 路徑

目前 Decode 不使用 batched `gather_qmm`。Runtime 會對六個 selected routed experts 分別執行 `quantized_matmul`。[direct Decode path](../../runtime/deepseek_v4_ssd/model.py#L226)

每個 routed expert 會執行融合 `w13` 和 `w2` 兩個量化矩陣運算。43 個 main model layer 每個 token 最多會排入 516 個 routed expert 量化矩陣運算。這個數字不含 common tensor 的運算。

```text
43 layers * 6 selected experts * 2 quantized matrix operations = 516
```

因此，`gather_qmm` identity index 快取不會直接改善目前的 Decode。

Runtime 已使用 ready expert scheduler。Runtime 會在 expert blob 準備完成後開始該 routed expert 的計算。

非 hash-routed layer 仍使用 `mx.argpartition`，從 256 個 routed expert 中選出 6 個。這個選擇每個 Decode token 執行 40 次。這個路徑位於 pinned `mlx-lm` 的 `deepseek_v4._expert_select`。

### 記憶體與 I/O

目前預設值是 1,152 個 slot 和 4 個 read worker。[RuntimeConfig](../../runtime/deepseek_v4_ssd/model.py#L30)

1,152 個 slot 的 expert blob 容量約為 14.34 GiB。

Runtime 對每個 expert layer 開啟一個一般唯讀 file descriptor。Runtime 未設定 `F_NOCACHE`。[file descriptor 建立](../../runtime/deepseek_v4_ssd/expert_cache.py#L355)

Runtime 使用 `os.preadv` 直接讀入 slot 或 full-layer buffer。[expert read](../../runtime/deepseek_v4_ssd/expert_cache.py#L786)

Swift direct SSD benchmark 已使用 `F_NOCACHE`。[ExpertIOBenchmark](../../Sources/DeepSeekRepack/ExpertIOBenchmark.swift#L72)

### 已記錄的基準

2026-08-09 的文件記錄下列結果。

| 工作負載 | 指標 | 結果 |
|---|---|---:|
| 4,096-token prompt，1-token output | TTFT | 28.89 秒 |
| 同上 | Prefill | 141.76 Tok/s |
| 同上 | SSD read | 14.48 秒 |
| 同上 | peak memory | 36.77 GB |
| 4,096-token prompt，256-token output | direct Decode | 6.39 Tok/s |

這些數字是歷史基準。正式研究必須先在目前 commit `57e440e` 重新建立基準。

## 新方向與優先順序

| 順序 | ID | 目標 | 方向 | 目前結果或 gate |
|---:|---|---|---|---|
| 1 | P1 | Prefill | 平均分配 attention 與 MoE 區塊 | 拒絕。速度降低，route histogram 改變。 |
| 2 | P2 | Prefill、短 context Decode | 略過完整集合的 CSA indexer 計算 | 拒絕。收益不穩定且低於門檻。 |
| 3 | R2 | Prefill、Decode | Slot 與 worker A/B | 完成。保留 1,152 slots、4 read workers 和 2 prefetch workers。 |
| 4 | R3 | Decode | Current-checkout DSpark 探索性 A/B | 拒絕。Request 增加 32.58%，兩個 run 都 fallback。 |
| 5 | M1 | Prefill、記憶體 | 分階段設定 macOS 檔案快取 | 正式評估完成；候選已拒絕並移除。 |
| 6 | P3 | Prefill | 依 rows/expert 選擇 `gather_qmm` tile | 停止。12.77% 與 13.99% 都低於 15% profile gate。 |
| 7 | D1 | Decode | 降低 MLX 圖走訪成本 | Profile gate 未通過 |
| 8 | P4 | Prefill | 依 attention 類型選擇區塊 | 拒絕。512-step 候選改變 39/42 個 MoE layer 的 expert route。 |
| 9 | D2 | Decode | 專用 routed expert top-6 選擇 | 停止。0.121%、0.456% 和 0.972% 都低於 3% profile gate。 |
| 10 | D3 | Decode | 重用相鄰 token 已解量化的 CSA row | 停止。Row 交集比例中位數是 85.94%；`gather` 校正占比最多是 0.34%。 |
| 11 | P5 | Prefill | 快取 `gather_qmm` identity index | Profile gate 未通過 |

## 目前實作狀態

目前 checkout 不保留 P1、P2、M1 或 P4 prototype。

| ID | 開關 | 已完成 | 尚未完成 |
|---|---|---|---|
| P1 | 已移除 | 一輪 ABBA 與 route trace | 不再實作 |
| P2 | 已移除 | 兩輪 ABBA 與多 context smoke | 不再實作 |
| M1 | 已移除 | 五輪正式 ABBA、p95 與 bootstrap 統計 | 不再實作目前設計 |
| P3 | 無 | Metal GPU Counters profile、kernel 與 rows/expert 分布 | Profile gate 未通過，不建立候選。 |
| P4 | 已移除 | attention 類型 profile、step pilot、記憶體與 expert route 比較 | 正確性 gate 未通過，不執行正式 ABBA。 |
| D2 | 無 | Python dispatch、Metal GPU Counters profile 和 256-to-6 microbenchmark | Profile gate 未通過，不建立候選。 |
| D3 | 無 | 五種 context row trace、production-shape `gather` microbenchmark 和 D2 shader 對應 | Row gate 與 profile gate 都未通過，不建立候選。 |

CLI 現在會把完整 prompt 與 output token ID 的 SHA-256 寫入 metrics JSON。
CLI 使用逗號連接的十進位 token ID 作為 hash 輸入。
CLI 也會記錄 layer-major kernel token 數與兩種區塊大小。
route trace 也會保存每個 Prefill MoE 區塊的 row histogram。
`Scripts/prepare_r0_prompts.py` 會建立固定 R0 prompt 與 manifest。

### 2026-08-10 探索性 ABBA

本次測試使用 base commit `57e440e` 加上目前 prototype 修改。
每個 run 都建立新的 runtime process。
作業系統 page cache 沒有清除。
persistent prompt cache 已停用。
generation 使用 batch size 1、greedy、`temperature=0`、`top_p=1` 和一個 output token。

本次測試只使用 repeated prompt。
每個候選項目只執行一輪 ABBA。
每個 mode 有兩個 run。
下表的 median 是兩個 run 的中間值。

| ID | Mode | API prompt token | Kernel token | TTFT | Prefill | Expert read | MLX peak memory |
|---|---|---:|---:|---:|---:|---:|---:|
| P1 | Control | 4,098 | 4,097 | 23.396 秒 | 175.44 Tok/s | 14.672 秒 | 21.256 GB |
| P1 | Balanced | 4,098 | 4,097 | 28.032 秒 | 146.24 Tok/s | 13.879 秒 | 20.614 GB |
| P2 | Control | 4,096 | 4,095 | 23.370 秒 | 175.30 Tok/s | 14.809 秒 | 21.374 GB |
| P2 | Skip | 4,096 | 4,095 | 22.367 秒 | 183.14 Tok/s | 14.338 秒 | 21.353 GB |

P1 的 TTFT 增加 19.8%。
P1 的 Prefill Tok/s 降低 16.6%。
P1 的 expert bytes 也從 149.242 GB 變成 148.279 GB。
後續 route trace 顯示 43 層中有 40 層的 Prefill histogram 不同。
route histogram 至少有 32,588 個 assignment 不同。
總 assignment 是 1,032,960 個。
差異下限是 3.15%。

P2 的 TTFT 降低 4.3%。
P2 的 Prefill Tok/s 增加 4.5%。
P2 的 expert read time 同時降低 0.471 秒。
因此，本次測試不能把全部差異歸因於 CSA indexer。
P2 每個 run 略過 42 次完整集合 indexer call。

八個 run 的 output token hash 都是
`3f5f3806e425deac33023e9764d08ac98397e6f1bc8599743cbb15a9cfdf8929`。
原始 metrics 位於被忽略的
`scratch/runtime-prefill-decode-experiment/`。

本次結果不是正式效能結果。
本次結果缺少五輪 ABBA、多種 prompt、route parity、`phys_footprint` 和 Metal capture。

P1 已拒絕。
P1 prototype 已移除。
P2 已拒絕。
P2 prototype 已移除。

### P2 第二輪 ABBA

執行第二輪前，使用者停止其他 SSD 與 GPU 活動。
第二輪仍使用 4,096-token repeated prompt。
測試順序是 Control、P2、P2、Control。

| Mode | TTFT 中間值 | Prefill 中間值 | Expert read 中間值 |
|---|---:|---:|---:|
| Control | 21.082 秒 | 194.35 Tok/s | 14.107 秒 |
| P2 | 21.284 秒 | 192.45 Tok/s | 14.075 秒 |

P2 的 TTFT 增加 0.96%。
P2 的 Prefill Tok/s 降低 0.98%。
Expert read 只降低 0.032 秒。
四個 output token hash 相同。

兩輪共有四個配對。
四個 TTFT 改善值是 -2.43%、0.47%、2.36% 和 6.18%。
配對中位數改善是 1.41%。
這個結果低於 3% 採用門檻。
這個結果也不穩定。

原始 metrics 使用 `p2-control-*`、`p2-skip-*` 和 `p2-r2-*` 檔名。
檔案位於被忽略的 `scratch/runtime-prefill-decode-experiment/`。

## 實驗 R0：建立可信的基準

### 狀態

R0 baseline 已完成。
R1 已完成 GPU busy、GPU idle 和 CPU sample 的 boundary capture。

R0 使用 repeated、code、繁體中文技術文字、mixed math 和 Tool-like prompt。
每種 prompt 都測試 4,096、8,192 和 14,363 token。
每個 run 都產生 256 個 output token。

| Prompt token | TTFT 中位數 | Prefill 中位數 | Decode 中位數 | Misses 中位數 | Evictions 中位數 |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 21.611 秒 | 189.53 Tok/s | 8.00 Tok/s | 24,700 | 23,548 |
| 8,192 | 39.124 秒 | 209.39 Tok/s | 7.76 Tok/s | 24,956 | 23,804 |
| 14,363 | 67.466 秒 | 212.89 Tok/s | 7.43 Tok/s | 25,345 | 24,193 |

Repeated 4K 的 Decode 是 12.22 Tok/s。
四種非 repeated 4K prompt 的 Decode 是 6.44 至 8.28 Tok/s。
因此，後續 A/B 必須使用一般 prompt。

R1 的 Tool-like Decode GPU busy 是 35.5% 至 37.1%。
Tool-like Decode GPU idle 是 7.968 至 8.114 秒。
R2a 已拒絕 2,048 slots。
R2b 已拒絕 2 workers。
R2b 已拒絕 8 workers。
R2c 已拒絕 4 prefetch workers。
R2c 已拒絕 1 prefetch worker。
R2 已完成。
R3 已拒絕目前 DSpark 預設啟用。
M1 記憶體接受條件未通過。
M1 Tool-like 4K 效能重跑已完成。
M1 五輪正式效能評估已完成。
M1 候選已拒絕並移除。
P3、P4、D2 與 D3 已完成。
目前計畫沒有其他 active trace 或 profile 工作。

完整結果位於
[`docs/benchmarks/2026-08-10-r0-m5-pro.json`](../../docs/benchmarks/2026-08-10-r0-m5-pro.json)。

### 目的

R0 要分開 GPU、CPU、SSD 和記憶體成本。後續實驗不可只比較總時間。

### 工作

1. 固定 checkpoint、token ID、環境變數和 RuntimeConfig。
2. 關閉 DSpark、in-memory prompt cache 和 persistent prompt cache。
3. 記錄每個 phase 的 wall time。
4. 使用 Metal System Trace 或同等工具記錄 kernel、command buffer 和 GPU idle gap。
5. 記錄每個 MoE 區塊的 routed row 分布。
6. 記錄 CSA indexer、`MXFP8PoolingCache.gather` 和 routed expert top-k 的時間。
7. 同時記錄 MLX 與作業系統記憶體。

目前工具狀態如下。

- CLI 已保存 prompt 與 output token hash。
- prompt 準備工具已保存文字檔、hash 和重建規則。
- route trace 已保存每個 Prefill MoE 區塊的 row histogram。
- Metal capture 使用 macOS 的 `Metal System Trace` template。

### 必要指標

Prefill 指標如下。

- TTFT。
- Prefill Tok/s。
- attention、CSA indexer、MoE、common tensor 與 output head 時間。
- `gather_qmm` call 數量。
- 每個區塊的最小、平均、p95 和最大 rows/expert。
- expert bytes、read time、read bandwidth 和 prefetch hit。
- CPU graph build 與 `mx.eval` 時間。

Decode 指標如下。

- 端到端 Tok/s。
- model step p50 與 p95。
- cache eval p50 與 p95。
- routed expert read wait。
- routed expert top-k 時間。
- CSA indexer 時間。
- `MXFP8PoolingCache.gather` 時間。
- MLX 圖走訪與 command-buffer 提交時間。

記憶體指標如下。

- `mx.get_active_memory()`。
- `mx.get_cache_memory()`。
- `mx.get_peak_memory()`。
- `footprint` 的 `phys_footprint`。
- process resident size。
- `vm_stat` 的 compressor 與 swap 變化。

MLX 指標不能單獨作為記憶體驗收依據。一個 MLX 0.32.0 的外部 streaming case 曾回報 46 GB 的 MLX peak 與 110 GB 的 `phys_footprint`。這是單一外部案例，不代表本 Runtime 一定有相同行為。[MLX issue #3896](https://github.com/ml-explore/mlx/issues/3896)

### Profile 證據限制

R1 Metal System Trace 沒有提供 shader timing sample。
R1 資料不能把 GPU time 分配到 attention、routed expert 和 shared expert。
request 結束時間是依最後一個 Python GPU interval 推定。
CPU sampled running time 是跨 thread 的總和，不是 wall time。

CPU profile 沒有顯示 MLX 圖走訪或 identity index 穩定超過 5%。
因此，D1 和 P5 的 profile gate 都沒有通過。
目前不實作 D1 或 P5。

如果後續候選項目需要 shader-level 證據，研究必須使用
`Metal GPU Counters` 重新 capture。
重新 capture 前必須先建立安靜的 SSD 與 GPU 測試環境。

### 完成條件

R0 已產生可重跑的 JSON 記錄。
R0 artifact 已保存 token ID hash、commit、MLX 版本、macOS 版本和硬體類型。

## 實驗 R2：Slot 與 Worker A/B

### 假設

一般 4K prompt 有 23,155 至 27,613 次 expert eviction。
一般 4K prompt 的 Decode 是 6.44 至 8.28 Tok/s。
較多 slot 可能降低 expert read 與 GPU idle。
較多 slot 也會增加記憶體容量和 eviction scan 成本。

### R2a：Slot 探索性 ABBA

R2a 已完成並停止。
A 使用目前預設的 `slots=1152`。
B 使用 `slots=2048`。
兩組都保留 `read_workers=4` 和 `prefetch_read_workers=2`。

2,048 slots 的 expert blob 容量比 1,152 slots 多 11.156 GiB。
這是容量差，不是已量測的 process footprint 差。

原計畫使用五個 R0 4K prompt。
Repeated prompt 只作為控制組。
一般 workload 結論只使用 code、繁體中文技術文字、mixed math 和 Tool-like prompt。
每個 prompt 執行一輪 ABBA。
因此，第一輪是探索性測試，不是正式效能結果。

每個 run 必須符合下列條件。

1. 建立新的 process。
2. 產生 256 個 output token。
3. 停用 persistent prompt cache 和 DSpark。
4. 不清除作業系統 page cache，但記錄執行順序。
5. 記錄 output token hash、Decode、expert bytes、read time、hit、miss、eviction、RSS、footprint、compressor 和 swap。

停止條件是 B 的一般 prompt 配對中位數改善不到 3%。
如果任何 output token hash 不同，該比較無效。
如果 swap 明顯增加，停止測試。

Repeated prompt 的 2,048-slot Decode 降低 3.31%。
Code prompt 的 2,048-slot Decode 降低 71.09%。

| Prompt | 1,152 slots | 2,048 slots | Decode 差異 |
|---|---:|---:|---:|
| Repeated | 14.42 Tok/s | 13.95 Tok/s | -3.31% |
| Code | 8.25 Tok/s | 2.39 Tok/s | -71.09% |

Code 的 expert bytes 降低 21.73%。
Code 的 miss 降低 31.34%。
Code 的 eviction 降低 36.77%。
但是，Code 的 routing synchronization boundary 從 9.829 秒增加到 111.488 秒。
Code 的 MLX peak memory 增加 11.156 GiB。

Code ABBA 已觸發停止條件。
研究沒有執行後面三種 prompt。
八個完整 run 的 output token hash 都在相同 prompt 內一致。
Swap 沒有增加。

R2a 不是正式五個 workload、五輪 ABBA 結果。
R2a 足以拒絕 2,048-slot 候選設定。
目前預設值維持 1,152 slots。

2,048 slots 不符合目前預設值的 256 MiB 記憶體增加門檻。

完整資料位於
[`docs/benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

### R2b：Worker A/B

R2b 保留 `slots=1152`。
R2b 已完成 `read_workers=2/4` 的 code 4K 探索性 ABBA。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read |
|---:|---:|---:|---:|---:|
| 4 | 8.29 Tok/s | 110.58 ms | 172.12 ms | 28.042 秒 |
| 2 | 7.94 Tok/s | 116.54 ms | 182.41 ms | 29.435 秒 |

2 workers 讓 Decode 降低 4.20%。
兩個配對差異是 -4.12% 和 -4.27%。
Expert read timer 增加 4.97%。
Expert bytes、miss、eviction 和 token hash 都相同。
Swap 沒有增加。

2-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`docs/benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

R2b 也已完成 `read_workers=4/8` 的 code 4K 探索性 ABBA。

| Read workers | Decode 中位數 | Expert read |
|---:|---:|---:|
| 4 | 8.27 Tok/s | 28.202 秒 |
| 8 | 8.29 Tok/s | 27.875 秒 |

8 workers 讓 Decode 增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
這個結果低於 3% 採用門檻。

8-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`docs/benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

### R2c：Prefetch Worker A/B

R2c 使用 code 4K prompt 和一個 output token。
R2c 的主要指標是 TTFT、Prefill Tok/s、expert read timer 和 prefetch hit。
R2c 保留 `slots=1152` 和 `read_workers=4`。

R2c 已完成 `prefetch_read_workers=2/4` 的探索性 ABBA。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read |
|---:|---:|---:|---:|
| 2 | 21.956 秒 | 186.56 Tok/s | 14.478 秒 |
| 4 | 22.061 秒 | 185.67 Tok/s | 12.009 秒 |

4 workers 讓 expert read timer 降低 17.05%。
但是，TTFT 增加 0.48%。
兩個 TTFT 配對改善是 -1.01% 和 0.05%。
結果低於 3% 採用門檻。

4-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`docs/benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

R2c 也已完成 `prefetch_read_workers=1/2` 的探索性 ABBA。

| Prefetch workers | TTFT 中位數 | Expert read | Prefetch hits 中位數 |
|---:|---:|---:|---:|
| 2 | 22.037 秒 | 14.531 秒 | 39.5 |
| 1 | 23.740 秒 | 19.394 秒 | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
兩個配對改善是 -7.68% 和 -7.78%。
Expert read timer 增加 33.46%。

1-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`docs/benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

R2 設定 sweep 已完成。
目前保留 `slots=1152`、`read_workers=4` 和
`prefetch_read_workers=2`。
每次只改一個設定。
每個正式比較都遵守本文件的 A/B 方法。

## 實驗 R3：DSpark 探索性 A/B

R3 已完成。
R3 已拒絕目前 DSpark 預設啟用。

第一輪使用 code 4K prompt 和 64 個 output token。
A 使用 normal generation。
B 使用 `--dspark`。
執行順序是 A、B、B、A。

每個 run 都建立新的 process。
每個 run 都停用 persistent prompt cache。
R3 保存 main model 與 DSpark 的 expert bytes、read timer、cache miss、
acceptance、committed length、verification、fork、replay、fallback、
peak memory 和完整 output token hash。

| Mode | Request 中位數 | TTFT 中位數 | 總 throughput | 回報的 Decode | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Normal | 31.152 秒 | 22.072 秒 | 2.054 Tok/s | 6.94 Tok/s | 23.274 GiB |
| DSpark | 41.301 秒 | 33.722 秒 | 1.549 Tok/s | 8.32 Tok/s | 27.257 GiB |

DSpark 讓 request 增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都接受 5 個 draft token，並提交 6 個 token。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
回報的 Decode boundary 不包含較慢的 DSpark prefill。
這個 Decode 改善不是端到端收益。

四個 output token hash 都相同。
Swap 沒有增加。
R3 已觸發停止條件。
研究沒有擴大到五個 prompt 和 256 個 output token。

完整資料位於
[`docs/benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

DSpark 預設值維持停用。
只有 DSpark 實作改變後，才重新執行探索性 ABBA。

## 實驗 P1：平均分配 Prefill 區塊

### 結果

P1 已拒絕。
runtime 已移除 P1 prototype。
原因是速度降低 19.8%，且 Prefill route histogram 改變。

### 假設

固定 step 會留下極小的最後區塊。極小區塊的 GPU 使用率較低。極小 MoE 區塊也會讓每個 routed expert 的 row 數過少。

平均分配使用下列規則。

```text
chunk_count = ceil(token_count / target_step)
chunk sizes differ by at most one token
max(chunk_size) <= target_step
```

範例如下。

| token 數 | 現有 MoE 區塊 | 平均分配後 |
|---:|---|---|
| 4,097 | 4,096 + 1 | 2,049 + 2,048 |
| 8,193 | 4,096 + 4,096 + 1 | 2,731 + 2,731 + 2,731 |
| 14,363 | 4,096 + 4,096 + 4,096 + 2,075 | 3,591 + 3,591 + 3,591 + 3,590 |

這個規則不增加區塊數。這個規則不增加最大區塊大小。

一個外部 MLX Swift LM issue 在其他模型上回報，退化尾端 case 的 Prefill 最多改善約 9%。這不是本專案的預期值。[MLX Swift LM issue #466](https://github.com/ml-explore/mlx-swift-lm/issues/466)

### 實作邊界

1. 先只加入純整數區塊規劃函式。
2. attention 與 MoE 共用相同規則，但保留不同 target step。
3. 不加入 padding。
4. 不改 cache offset、mask 或 token 順序。

### 測試

測試 Prefill kernel 實際收到的 token 數。API generation path 會保留最後一個 token。因此，測試記錄必須同時寫入 API prompt token 數與 Prefill kernel token 數。

必要邊界如下。

- 1,023、1,024、1,025。
- 2,047、2,048、2,049。
- 4,095、4,096、4,097。
- 8,191、8,192、8,193。
- 14,363。

### 停止條件

如果 4,097 和 8,193 的 paired median Prefill 改善都小於 2%，停止 P1。

如果任一主要 case 的 peak memory 增加超過 256 MiB，拒絕 P1。

預估工期是 1 至 2 個工作日。

## 實驗 P2：略過完整集合的 CSA indexer 計算

### 結果

P2 已拒絕。
runtime 已移除 P2 prototype。
兩輪 ABBA 的配對中位數改善是 1.41%。
安靜環境的第二輪慢 0.96%。

### 程式事實

`_correct_indexer` 先更新 indexer compressor。接著，Runtime 計算 query projection、index score、weights 和 top-k。[indexer path](../../runtime/deepseek_v4_ssd/model.py#L441)

CSA attention 在 pooled row 數不超過 `index_topk` 時使用完整 pooled 集合。該分支不使用 top-k score。Pinned `mlx-lm` 仍會先呼叫 indexer。

對固定模型，`index_topk` 是 512。`compress_ratio=4` 的 pooled row 數在前 2,048 token 不超過 512。

`compress_ratio=128` layer 使用 dense compressed attention。這些 layer 不使用 lightning indexer。P2 不適用於這些 layer。

### 假設

Runtime 必須繼續更新 indexer compressor。Runtime 可以略過 query projection、score、weights 和 top-k。Runtime 可以直接回傳升序的完整 row index。

這個修改不改變 selected row 集合。這個修改也不改變 selected row 順序。

### 探索性測試

本研究執行了一次 control-first 探索性測試。測試使用 4,096-token 重複 prompt、64-token greedy output 和關閉的 persistent prompt cache。

| 指標 | Control | P2 prototype | 差異 |
|---|---:|---:|---:|
| TTFT | 23.465 秒 | 22.114 秒 | -5.8% |
| Prefill | 174.56 Tok/s | 185.22 Tok/s | +6.1% |
| SSD read | 16.178 秒 | 15.404 秒 | -0.774 秒 |
| MLX peak memory | 22.199 GB | 22.199 GB | 0 |
| token SHA-256 | 相同 | 相同 | 相同 |

Prototype 略過 42 次完整集合 indexer call。Prototype 仍執行 1,407 次一般 indexer call。

這次測試不能證明 5.8% 改善。Prototype 在第二個 process 執行。SSD read 同時減少 0.774 秒。這個順序會受到檔案快取和溫度影響。

這次測試也不能證明 Decode 改善。4K context 的 Decode 不會進入 P2 fast path。任何 Decode 差異都屬於測試干擾。

### 正式測試

1. 使用 ABBA 次序執行至少 5 對測試。
2. 分開測試 512、2,048、4,096 和 8,192 context。
3. 對每個 `compress_ratio=4` layer 比較 top-k index。
4. 比較最終 logits、greedy token ID 和 cache state。
5. 單獨記錄略過的 call 數量。

### 停止條件

如果 2K 與 4K Prefill 的 paired median 改善都小於 2%，停止 P2。

如果任何 top-k index 或 greedy token 不同，拒絕 P2。

預估工期是 1 個工作日。

## 實驗 P3：依 rows/expert 選擇 `gather_qmm` tile

### 假設

目前 4,096-token MoE 區塊平均有 96 rows/expert。這個形狀可能適合較寬的 `gather_qmm` tile。

MLX PR #3918 在 M3 Max 和另一個 256-expert 模型上回報，rows/expert 為 64 至 128 時，kernel 改善 19% 至 22%。該 PR 的 32K Prefill 改善是 6.2%。該模型使用不同 shape 與 group size。PR 已關閉。維護者沒有接受該實作。[MLX PR #3918](https://github.com/ml-explore/mlx/pull/3918)

MLX issue #3925 也指出，M5 Neural Accelerator path 的固定 64-row tile 在短 routed expert run 會浪費計算。[MLX issue #3925](https://github.com/ml-explore/mlx/issues/3925)

### 工作

1. 先用 Metal capture 確認目前實際 kernel。
2. 如果 Runtime 使用一般 `gather_qmm_rhs`，測試 #3918 的 tile 選擇原理。
3. 如果 Runtime 使用 `gather_qmm_rhs_nax`，測試 #3925 的 BM 選擇原理。
4. 分開測試 `w13` 與 `w2` shape。
5. 使用目前固定 step 的區塊。
6. 只建立 isolated MLX build。不要先替換正式 dependency。

### 正確性

每個 kernel shape 必須逐元素比較。每個端到端 case 必須產生相同 token ID。

### 停止條件

如果 `gather_qmm` GPU time 少於 Prefill 的 15%，停止 P3。

如果 isolated kernel 改善小於 10%，停止端到端整合。

如果端到端 Prefill 改善小於 3%，不採用 MLX fork。

### 結果

P3 使用 Tool-like 4K prompt 和一個 output token。
P3 使用 `Metal System Trace` 與 `Metal GPU Counters`。
目前 kernel 是 `mxfp4_gather_qmm_rhs_nax`。
kernel 使用 BM64、BN64 和 BK64。

Runtime metrics 記錄 42 個 batched expert layer。
Runtime metrics 記錄 84 次 `gather_qmm` call。
shader timeline 涵蓋 42 個 expert layer burst。

`gather_qmm` shader sample duration 是 2.966 秒。
這個時間是 23.220 秒 request 的 12.77%。
shader sample work 涵蓋 Python GPU busy 的 91.28%。
依涵蓋率校正後，`gather_qmm` 是 request 的 13.99%。

同一個固定 prompt 的 R0 route trace 顯示，平均是 95.98 rows/expert。
24.53% 的 expert run 是空的。
27.37% 的 expert run 只有 1 至 15 rows。
BM64 的理論 padding 是配置 rows 的 24.43%。

rows/expert 分布支持 tile 浪費假設。
但是，兩個 GPU time 占比都低於 15% profile gate。
P3 已觸發停止條件。
專案沒有建立 isolated MLX build。
專案也沒有修改 runtime。

完整資料位於
[`docs/benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json`](../../docs/benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json)。

預估工期是 3 至 5 個工作日。

## 實驗 D1：降低 MLX 圖走訪成本

### 狀態

R1 CPU profile 沒有讓 D1 通過 5% gate。
D1 目前停止。

### 假設

目前 Decode 每個 token 會建立並評估大型 MLX graph。CPU 可能花費明顯時間走訪 graph 和準備 command buffer。

MLX PR #3920 將圖走訪修改與 `gather_qmm` identity index 快取放在同一個 PR。作者在另一個 MoE 模型上回報 2.0% 至 2.8% Decode 改善、3.6% 8K Prefill 改善，以及不變的 peak memory。維護者認為實作品質不足，並關閉 PR。[MLX PR #3920](https://github.com/ml-explore/mlx/pull/3920)

本專案不可直接採用該 PR。本專案必須把兩個修改拆開。

### D1a：圖走訪

1. profile `eval_impl`、degree bookkeeping、stream bookkeeping 和 Python thread。
2. 只有 CPU 圖走訪占 Decode wall time 至少 5% 時才建立 prototype。
3. 保留 graph traversal order。
4. 使用 bounded data structure。
5. 執行 thread safety、transform 和 trace 測試。

### D1b：command-buffer 成本

一個外部 MLX Swift LM issue 回報 command-buffer accounting 與 commit policy 可改善另一個 MoE 模型的 Decode。該 issue 尚未提供可接受的正式 patch。[MLX Swift LM issue #466](https://github.com/ml-explore/mlx-swift-lm/issues/466)

1. 先量測每個 token 的 command buffer 數量。
2. 量測 CPU commit wait 與 GPU idle gap。
3. 只有兩者合計占 Decode wall time 至少 5% 時才繼續。
4. 在可審查的 upstream patch 出現前，不自行複製 issue 中的設計。

### 停止條件

如果 CPU 圖走訪與 command-buffer 提交合計少於 Decode wall time 的 5%，停止 D1。

如果 256-token Decode 的 paired median 改善小於 3%，不採用 MLX fork。

預估工期是 3 至 5 個工作日。等待 upstream patch 的時間不計入工期。

## 實驗 M1：分階段設定 macOS 檔案快取

### 目前狀態

目前 checkout 已移除 `--prefill-no-file-cache` prototype。
歷史 prototype 為 full-layer Prefill 開啟獨立 file descriptor。
歷史 prototype 對 Prefill file descriptor 設定 `F_NOCACHE`。
Decode 使用一般 file descriptor。

M1 的單元測試已通過。
M1 的 code 4K one-token full-model smoke test 已通過。
Smoke test 的 output token hash 與既有 control 相同。
M1 已完成 code 4K、256-output-token 探索性 ABBA。
四個 output token hash 都相同。
這次 pilot 只比較 A 和 B。
這次 pilot 沒有執行 C 或 D。

M1 讓 request 中位數改善 4.32%。
M1 讓 TTFT 改善 7.60%。
M1 讓 Decode throughput 增加 2.02%。
M1 讓 expert read timer 降低 12.18%。

Process peak footprint 增加 1.77 MiB。
這個差異可視為沒有變化。
測試沒有出現可重現的 memory pressure。
M1 沒有通過預先定義的記憶體接受條件。
M1 的正式記憶體擴大測試已停止。

本次測試只有一種 workload 和一輪 ABBA。
本次測試不能支持正式效能採用。
兩個 request 配對都有超過 4% 的同向改善。
Tool-like 4K 獨立效能重跑已確認相同方向。

完整資料位於
[`docs/benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)。

Tool-like 4K 重跑也使用 256 個 output token 和一輪 ABBA。
M1 讓 request 中位數改善 3.51%。
M1 讓 TTFT 改善 8.29%。
M1 讓 Decode throughput 增加 0.75%。
M1 讓 expert read timer 降低 8.57%。
兩個 request 配對分別改善 2.34% 和 4.63%。
四個 output token hash 都相同。

Tool-like process peak footprint 增加 0.98 MiB。
System swap 在序列中增加 2.13 MiB。
System swap 是 system-wide 指標。
本次測試不能把這個變化歸因於 M1。

Code 與 Tool-like 的 request 中位數都改善超過 3%。
兩種 workload 的四個 request 配對都同向改善。
這個結果確認 pilot 等級的效能訊號。

完整資料位於
[`docs/benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

### 假設

Runtime 已用約 14.34 GiB slot 保存 routed expert。macOS 也可能在檔案快取保存相同 expert bytes。

Prefill 對每個 layer 執行一次大型順序 read。Decode 則可能再次使用最近讀過的 individual expert blob。兩個 phase 不應自動使用相同檔案快取策略。

Apple `fcntl(2)` 定義下列功能。

- `F_NOCACHE` 可關閉或開啟 data caching。
- `F_RDAHEAD` 可關閉或開啟 read ahead。
- `F_RDADVISE` 可送出非同步 read 建議。

來源：[Apple `fcntl(2)` manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)

### 實驗組

| 組別 | Prefill descriptor | Decode descriptor |
|---|---|---|
| A | 現有一般快取 | 現有一般快取 |
| B | `F_NOCACHE` | 現有一般快取 |
| C | `F_NOCACHE` + `F_RDAHEAD` on | 現有一般快取 |
| D | `F_NOCACHE` | `F_NOCACHE` |

D 是負向控制。D 可能增加 Decode SSD miss 成本。

Runtime 應使用分開的 Prefill 與 Decode descriptor。實驗不可在同一 descriptor 中途切換政策，除非測試證明政策只影響該 descriptor。

### 測試

1. 分開記錄 warm file-cache 與可確認的 cold file-cache case。
2. 如果系統沒有可靠清除 file cache，該 case 必須標示為 warm 或 unknown。
3. 記錄 `phys_footprint`、resident size、compressor、swap、SSD bytes 和 Decode miss。
4. 在相同 prompt 後執行 256-token Decode。

### 接受條件

M1 必須降低 `phys_footprint` 至少 5%，或消除可重現的 memory pressure。

Prefill 與 Decode 的速度都不可降低超過 2%。

### 評估結果

Code 4K pilot 沒有達到記憶體接受條件。
專案不把 M1 採用為記憶體最佳化。

Code 4K pilot 顯示一致的 request 與 TTFT 效能訊號。
專案當時暫時保留 opt-in prototype。
Tool-like 4K 獨立重跑已確認相同方向。
這個重跑沒有改變原本的記憶體接受條件。

正式多 workload 重複 ABBA 已完成。
M1 沒有通過全部正確性、效能和記憶體條件。
專案已拒絕並移除 M1。

### 正式效能評估設計

Code 與 Tool-like pilot 只用來決定是否擴大研究。
正式統計不納入這兩個 pilot。

正式評估使用五個 R0 4K prompt。
Repeated prompt 是控制 workload。
Code、繁體中文技術文字、mixed math 和 Tool-like 是主要 workload。

每個 prompt 執行五輪 ABBA。
每個 run 都建立新的 process，並產生 256 個 output token。
Persistent prompt cache 和 DSpark 必須停用。
作業系統 page cache 不清除時，cache 狀態必須標示為 `not purged`。
每個 prompt 的第一個 Control run 是 warm-up。
Warm-up 不納入正式統計。

每一輪必須涵蓋全部五個 prompt。
不同輪次必須輪替 prompt 順序。
正式結果不得只保留較快的輪次。

正式評估使用 request time、TTFT、Prefill throughput 和 Decode throughput。
正式評估也記錄 process footprint、MLX peak memory、compressor 和 swap。
主要決策使用四個主要 workload 的 paired delta。
Repeated prompt 只檢查 regression。

任何 output token hash 不同時，停止並診斷正確性。
出現 thermal 或 performance warning 時，該輪無效。
System swap 在兩個連續 candidate run 增加時，停止目前輪次並先診斷。

### 正式效能評估進度

Wave 1 已在 2026-08-11 完成。
Prompt 順序是 Repeated、Code、繁體中文技術文字、Mixed math、Tool-like。
每個 prompt 的 warm-up 都不納入統計。
Wave 1 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 3.03% | 8.23% | 8.98% | -2.93% |
| Code | 4.54% | 8.55% | 9.34% | 1.73% |
| 繁體中文技術文字 | 3.40% | 8.12% | 8.84% | 0.13% |
| Mixed math | 3.69% | 8.24% | 8.98% | 0.57% |
| Tool-like | 2.70% | 7.97% | 8.66% | -0.32% |

四個主要 workload 的 request 改善中位數是 3.54%。
八個主要 workload request 配對都改善。
所有 output token hash 都一致。
正式序列的 system swap 降低 8.00 MiB。
測試沒有觸發停止條件。

Repeated Decode 在本輪降低 2.93%。
後續 wave 因此繼續檢查這個 regression。
Wave 1 當時沒有正式 p95 或 95% bootstrap 信賴區間。
Wave 1 當時不能支持正式採用決定。

完整資料位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

Wave 2 也已在 2026-08-11 完成。
Prompt 順序是 Code、繁體中文技術文字、Mixed math、Tool-like、Repeated。
Wave 2 的 20 個正式量測 run 都有效。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.63% | 11.05% | 12.48% | 3.64% |
| Code | 3.78% | 7.83% | 8.49% | 0.93% |
| 繁體中文技術文字 | 3.66% | 8.26% | 9.01% | 0.47% |
| Mixed math | 3.57% | 8.51% | 9.30% | 0.17% |
| Tool-like | 2.61% | 7.77% | 8.43% | -0.36% |

Wave 2 的主要 workload request 改善中位數是 3.62%。
Wave 2 的八個主要 workload request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
System swap 降低 8.00 MiB。
測試沒有觸發停止條件。

Wave 3 也已在 2026-08-11 完成。
Prompt 順序是繁體中文技術文字、Mixed math、Tool-like、Repeated、Code。
Wave 3 的 20 個正式量測 run 都有效。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.93% | 11.87% | 13.45% | 3.25% |
| Code | 4.08% | 7.71% | 8.35% | 1.53% |
| 繁體中文技術文字 | 4.28% | 8.68% | 9.50% | 1.22% |
| Mixed math | 3.34% | 6.97% | 7.52% | 0.84% |
| Tool-like | 2.07% | 8.48% | 9.17% | -1.64% |

Wave 3 的主要 workload request 改善中位數是 3.71%。
Wave 3 的八個主要 workload request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
System swap 沒有變化。
測試沒有觸發停止條件。

Wave 4 也已在 2026-08-11 完成。
Prompt 順序是 Mixed math、Tool-like、Repeated、Code、繁體中文技術文字。
Wave 4 的 20 個正式量測 run 都有效。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.76% | 10.82% | 12.18% | 4.34% |
| Code | 3.31% | 5.81% | 6.17% | 1.53% |
| 繁體中文技術文字 | 3.56% | 7.95% | 8.63% | 0.48% |
| Mixed math | 3.17% | 7.75% | 8.42% | 0.01% |
| Tool-like | 3.32% | 7.90% | 8.56% | 0.63% |

Wave 4 的主要 workload request 改善中位數是 3.32%。
Wave 4 的八個主要 workload request 配對都改善。
所有 output token hash 都與 Wave 1 相同。
System swap 沒有變化。
測試沒有觸發停止條件。

Wave 5 也已在 2026-08-11 完成。
Prompt 順序是 Tool-like、Repeated、Code、繁體中文技術文字、Mixed math。
Wave 5 的 20 個正式量測 run 都有效。
第一次不完整序列已保存並排除。
完整 Wave 5 從新的 warm-up 重跑。

Wave 5 的主要 workload request 改善中位數是 3.15%。
TTFT 改善中位數是 7.58%。
Prefill throughput 增加中位數是 8.24%。
Decode throughput 增加中位數是 0.12%。
Wave 5 的全部十個 request 配對都改善。

五輪共有 100 個有效量測 run 和 50 個配對。
全部 50 個 request 配對都改善。
主要 workload request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
主要 workload 的 Decode p95 regression 最大值是 1.71%。

Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。
Process peak footprint 的配對中位數增加 0.32 MiB。
每一輪的 system swap 都沒有增加。
部分 wave 的 system-wide compressor 總量增加。
這個增加不能歸因於 M1。
但是，證據沒有證明 compressor 不增加。

M1 沒有通過全部正式採用條件。
專案拒絕預設啟用 M1。
runtime 已移除 M1 prototype。

Wave 5 完整資料位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)。

正式統計與決定位於
[`docs/benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](../../docs/benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

## 實驗 P4：依 attention 類型選擇 Prefill 區塊

### 狀態

P4 探索已完成。
候選沒有通過正確性 gate。
runtime 已移除 P4 prototype。

### 假設

目前所有 layer 共用同一個 attention step。三種 attention 類型的成本不同。

- `compress_ratio=0` 使用 local attention。
- `compress_ratio=4` 使用 CSA 與 lightning indexer。
- `compress_ratio=128` 使用 dense compressed attention。

因此，一個全域 step 不一定是三種 attention 類型的最佳值。

### 工作

1. 先完成 R0 的 layer-type profile。
2. 以目前固定 step 作為 Control。
3. 測試每種類型的 512、1,024、2,048 和 4,096 target step。
4. 每個候選項目只改變一種 attention 類型的 step。
5. 記錄每種類型的 peak temporary memory。
6. 不同 layer 可以使用不同 step。相同類型必須先使用相同 step。

### 風險

較大的 local attention 區塊可能增加 attention matrix 成本。較小區塊會增加 graph evaluation 次數。

### 停止條件

如果任一 attention 類型都沒有至少 5% 的 phase 改善，停止 P4。

如果端到端 Prefill 改善小於 2%，保留單一全域 step。

### 結果

P4 使用 Tool-like 4K prompt 和一個 output token。
Control 使用全域 1,024 step。
P4 分別測試 local、CSA 和 dense compressed attention 的
512、2,048 和 4,096 step。

512 step 在單次 profile 中降低 CSA 和 dense compressed attention phase。
但是，Control request 在研究期間漂移 7.91%。
單次 request time 不能作為正式速度結論。

組合的 512-step 候選產生相同 output token hash。
但是，該候選改變 39/42 個 MoE layer 的 expert route。
expert histogram 的 L1 distance 是 64,150。
至少 32,075 個 assignment 改變。
差異下限是 3.11%。

CSA、dense compressed 和 local 的單項 512-step 候選也分別改變
8、-1 和 7 個 expert blob 的讀取量。
因此，候選的 attention phase time 混入不同的 expert I/O。

P4 沒有通過正確性 gate。
專案沒有執行正式 ABBA。
專案拒絕 P4，並保留單一全域 step。

完整資料位於
[`docs/benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json`](../../docs/benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json)。

預估工期是 2 至 4 個工作日。

## 實驗 D2：專用 routed expert top-6 選擇

### 狀態

D2 profile 已完成。
D2 沒有通過 3% profile gate。
專案沒有建立 prototype。

### 假設

非 hash-routed layer 每個 Decode token 會對 256 個 score 執行 `mx.argpartition`。固定模型只需要 6 個 index。

一個外部 MLX Swift LM issue 在另一個 256-expert、top-8 模型上回報，專用 router top-k 可改善短 context Decode 1.9%。該修改尚未成為本專案可直接使用的實作。[MLX Swift LM issue #466](https://github.com/ml-explore/mlx-swift-lm/issues/466)

### 工作

1. 先量測 `_expert_select` 的 GPU time 和 dispatch time。
2. 如果該時間占 Decode wall time 至少 3%，建立 isolated prototype。
3. Prototype 必須保留 selected index 的集合與順序。
4. Prototype 必須保留 routed weight 的 dtype 與值。
5. Hash-routed layer 不變。

D2 只研究固定的 256-to-6 選擇。D2 不重新開啟一般 custom Metal kernel 研究。

### 停止條件

如果 routed expert top-k 少於 Decode wall time 的 3%，停止 D2。

如果 index 順序或 greedy token 不同，拒絕 D2。

### 結果

D2 使用 Tool-like 4K prompt 和 64 個 output token。
runtime metrics 記錄 63 個 Decode token。

`_expert_select` Python dispatch 共花 13.252 ms。
這是 Decode wall time 的 0.121%。

Metal GPU Counters capture 量到 4.559 秒 Decode GPU busy。
Shader Timeline 涵蓋 GPU busy 的 46.88%。
`_expert_select` shader sample 是 58.834 ms。
這是 Decode wall time 的 0.456%。
依涵蓋率校正後是 0.972%。

一個 production-dtype bfloat16、256-to-6 microbenchmark 確認
6 個 `_expert_select` shader label。
microbenchmark 的同步 wall time 不參與 profile gate。

dispatch、原始 shader sample 和涵蓋率校正結果都低於 3%。
D2 已觸發停止條件。
runtime 沒有變更。

完整資料位於
[`docs/benchmarks/2026-08-11-d2-top6-profile-m5-pro.json`](../../docs/benchmarks/2026-08-11-d2-top6-profile-m5-pro.json)。

預估工期是 2 至 4 個工作日。

## 實驗 D3：重用相鄰 Decode token 的 CSA row

### 狀態

D3 trace 與 shader profile 已完成。
D3 沒有通過兩個 trace gate。
專案沒有建立 prototype。

### 程式事實

`MXFP8PoolingCache.gather` 會取得 selected row，然後對 MXFP8 row 執行解量化。[MXFP8 gather](../../runtime/deepseek_v4_ssd/fp8_cache.py#L161)

目前每個 Decode token 都重新執行這個工作。

### 假設

相鄰 Decode token 的 512 個 selected pooled row 可能高度重疊。Runtime 可以保留前一個 token 已解量化的 row。Runtime 只需解量化新 row。

本機 trace 已測試這個假設。

### Trace gate

1. 對 512、2K、4K、8K 和 14K context 記錄 selected row。
2. 計算相鄰 token 的交集比例與 Jaccard index。
3. 分開記錄每個 `compress_ratio=4` layer。
4. 記錄 `MXFP8PoolingCache.gather` 的 Decode 時間比例。

只有同時符合下列條件時才建立 prototype。

- selected row 的 median 交集比例至少 90%。
- `MXFP8PoolingCache.gather` 至少占 Decode wall time 的 5%。

### Prototype 限制

1. 每層 cache 上限是 512 rows 加 metadata。
2. Runtime 必須在 pending row 轉成 MXFP8 chunk 時失效相關項目。
3. Runtime 不可把完整 pooled cache 解量化為 BF16。
4. Runtime 必須保持 selected row 順序。

### 停止條件

如果 trace gate 未通過，停止 D3。

如果 prototype 的額外 lookup 成本抵銷解量化收益，停止 D3。

### 結果

D3 使用 Tool-like prompt 和 64 個 output token。
D3 對每個 context 比較 63 個相鄰 Decode transition。
每個 transition 包含 21 個 `compress_ratio=4` layer。

| Context | 交集比例中位數 | Jaccard 中位數 |
|---:|---:|---:|
| 512 | 100.00% | 100.00% |
| 2,048 | 99.02% | 98.07% |
| 4,096 | 78.71% | 64.90% |
| 8,192 | 70.51% | 54.45% |
| 14,363 | 62.30% | 45.25% |

五種 context 合併後的交集比例中位數是 85.94%。
這個結果低於 90% row gate。

D3 使用 production-shape microbenchmark 對應 `gather` shader label。
D3 再使用既有的 D2 Tool-like 4K Metal trace。
核心 `gather` shader sample 占 Decode wall time 的 0.117%。
依 Shader Timeline 涵蓋率校正後是 0.250%。
包含通用 clip 和 select shader 後是 0.158%。
依涵蓋率校正後是 0.338%。
所有結果都低於 5% profile gate。

D3 已觸發停止條件。
runtime 沒有變更。

完整資料位於
[`docs/benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json`](../../docs/benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json)。

預估 trace 是 2 個工作日。預估 prototype 是 3 至 5 個工作日。

## 實驗 P5：快取 `gather_qmm` identity index

### 狀態

R1 CPU profile 沒有顯示 identity index 建立成本穩定超過 5%。
P5 目前停止。

### 假設

目前 batched Prefill 只提供 `rhs_indices`。MLX 會為缺少的 lhs index 建立 identity row index。

MLX PR #3920 的作者回報，快取該 identity index 可改善另一個 MoE 模型的 8K Prefill。該 PR 已關閉。[MLX PR #3920](https://github.com/ml-explore/mlx/pull/3920)

### 工作

1. 從 #3920 分離 identity index 修改。
2. 使用 shape、dtype、device 和 stream 建立完整 key。
3. 設定 bounded cache。
4. 在 tracing 和 function transform 中停用快取。
5. 只測試 Prefill。不要把結果寫成 Decode 改善。

### 停止條件

如果 CPU profile 沒有 identity index 建立成本，停止 P5。

如果 8K Prefill 改善小於 2%，不採用 MLX fork。

預估工期是 1 至 2 個工作日。

## 基準矩陣

### Prompt

使用三組固定 token ID。

1. Varied：自然語言、程式碼和數字混合。
2. Repeated：只用於區塊邊界壓力測試。
3. Conversation：含固定 shared prefix，但停用 prompt cache。

不能只使用 repeated prompt。Repeated prompt 可能產生不代表一般工作負載的 routed expert 分布。

### Prefill 長度

必要長度如下。

- 321。
- 2,047、2,048、2,049。
- 4,095、4,096、4,097。
- 8,191、8,192、8,193。
- 14,363。

每個記錄必須清楚標示 API prompt token 數與 Prefill kernel token 數。

### Decode 長度

| context | output token | 用途 |
|---:|---:|---|
| 512 | 256 | 短 context Decode |
| 2,048 | 256 | P2 邊界與短 CSA |
| 4,096 | 256 | 目前主要基準 |
| 8,192 | 256 | 長 context steady state |
| 14,363 | 256 | 目前長 Prefill 基準的後續 Decode |
| 4,096 | 2,000 | 記憶體成長與長時間穩定性 |

## A/B 方法

1. 每個候選項目使用相同 process 模式。
2. 使用 ABBA 次序。
3. 每個 workload 至少執行 5 個 ABBA round。這代表至少 20 次 measured run。
4. 第一個 warm-up 不計入結果。
5. 記錄環境溫度、power mode 和其他 GPU process。
6. Prefill 與 Decode 分開報告。
7. cold、warm 和 unknown file-cache case 分開報告。
8. 不把單次最佳結果當成結論。

報告使用 paired delta。報告同時列出 median、p50、p95 和 95% bootstrap 信賴區間。

## 驗收條件

### 正確性

每個候選項目必須符合下列條件。

- 所有 greedy token ID 相同。
- token ID SHA-256 相同。
- routed expert index 相同。只對會修改 routing 的候選項目要求逐層比較。
- CSA selected row index 相同。只對會修改 indexer 的候選項目要求逐層比較。
- cache offset、remainder 和有效長度相同。
- kernel replacement 必須通過逐元素測試。

### 效能

正式採用的候選項目必須符合下列條件。

- 目標 workload 的 paired median 改善至少 3%。
- 95% bootstrap 信賴區間的下界高於 0。
- 非目標主要 workload 的 p95 regression 不超過 2%。

1% 至 3% 的改善只可保留為 upstream patch 或實驗分支。這類修改不可增加 Runtime 複雜度。

### 記憶體

正式採用的候選項目必須符合下列條件。

- `phys_footprint` 沒有可重現的增加。
- median `phys_footprint` 增加不超過 256 MiB。
- MLX peak memory 沒有可重現的增加。
- compressor 與 swap 沒有增加。
- 長時間 Decode 沒有線性記憶體成長。

## 執行順序與工期

| 階段 | 工作 | 預估工期 |
|---|---|---:|
| 0 | R0 基準與 R1 boundary capture | 完成 |
| 1 | P1 與 P2 評估 | 完成；兩項都拒絕 |
| 2 | R2a slot A/B | 完成；2,048 slots 已拒絕 |
| 3 | R2b 2/4 worker A/B | 完成；2 workers 已拒絕 |
| 4 | R2b 4/8 worker A/B | 完成；8 workers 已拒絕 |
| 5 | R2c 2/4 prefetch-worker A/B | 完成；4 workers 已拒絕 |
| 6 | R2c 1/2 prefetch-worker A/B | 完成；1 worker 已拒絕 |
| 7 | R3 DSpark 探索性 A/B | 完成；目前候選已拒絕 |
| 8 | M1 分階段檔案快取 | 完成；候選已拒絕並移除 |
| 9 | P3 `gather_qmm` tile | 完成；profile gate 未通過，未建立候選 |
| 10 | P4 attention 類型區塊 | 完成；正確性 gate 未通過，prototype 已移除 |
| 11 | D2 top-6 trace 與 prototype | 完成；profile gate 未通過，未建立 prototype |
| 12 | D3 CSA row trace；通過 gate 後做 prototype | 完成；兩個 gate 都未通過，未建立 prototype |

D1 和 P5 目前停止。
D1 和 P5 必須先取得新的 profile 證據。
目前計畫沒有其他 active 實驗。

## 交付物

每個實驗必須產生下列交付物。

1. 一份獨立設計說明。
2. 通過 profile gate 後，一個可開關的 prototype。
3. 單元測試與端到端 token parity 測試。
4. 原始 JSON 或 CSV benchmark 記錄。
5. Metal capture 或 CPU profile 摘要。
6. Prefill、Decode 和記憶體比較表。
7. 採用、拒絕或等待 upstream 的明確決定。

最終報告必須分開列出已量測結果、工程推論和外部案例。最終報告不可把外部模型的改善比例當成本專案結果。

## 外部一手資料

- [DeepSeek-V4 paper](https://arxiv.org/html/2606.19348v1)
- [MLX 0.32.0 release](https://github.com/ml-explore/mlx/releases/tag/v0.32.0)
- [MLX PR #3918：rows-per-expert tile](https://github.com/ml-explore/mlx/pull/3918)
- [MLX issue #3925：M5 `gather_qmm_rhs_nax` tile](https://github.com/ml-explore/mlx/issues/3925)
- [MLX PR #3920：圖走訪與 identity index](https://github.com/ml-explore/mlx/pull/3920)
- [MLX Swift LM issue #466：外部效能研究索引](https://github.com/ml-explore/mlx-swift-lm/issues/466)
- [Apple `fcntl(2)` manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
- [MLX issue #3896：作業系統 memory footprint 與 MLX 指標差異](https://github.com/ml-explore/mlx/issues/3896)
