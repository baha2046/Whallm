# Decode 提交與 Qwen packed cache：選項 2、3 實測

使用者在 GPT 6 Pro 討論後指定先研究 **2、3**。本輪以
`872d6fd7741a40ac25abe7423044d4ffa92a0c90` 為起點；已包含 route cache 與
Prefill／Decode 分開讀取。沒有將兩個候選混在同一次請求，也沒有修改 runtime 或 App 預設。

環境：Apple M5 Pro、64 GiB、macOS 26.6.2／25G83、MLX 0.32.2。
本輪結論：**選項 3 有明確的同 packed 模式 Decode 改善訊號；選項 2 未達延伸門檻。**
這是一次有條件的研究結果，不是跨模型、跨工作量或普通 BF16 的通用加速結論。

## 1. Decode：先分清楚空檔與可省時間

固定 Qwen 第 23 層，使用 installed model 的真實 MXFP4 expert weights，
隱藏向量與 router scores 為固定亂數。10 個 expert IDs、順序、4 個讀取 workers 固定。
每次建立 10-slot cache，預先放入 10／4／0 個指定專家，再執行一個 token。
所選檔案區段先讀過，OS cache 未清除；這不是冷 SSD 或完整模型比較。

保留每個 expert 的兩次 QMM、SwiGLU、最後 stack → scores → sum 的順序。
結尾 `mx.eval` 等所有使用者完成才釋放 cache；本輪沒有跨層覆寫 slot。

### 1.1 原生時間軸

`Metal System Trace` 加上 `Points of Interest`，對齊 637 個讀取／就緒／提交事件，
取得目標程序 179 個 GPU 活動區間。原版每層約有 11–12 個 GPU 活動區間。
這些區間的標籤是 command buffer，沒有辨識每個 expert 的個別 kernel 或最後使用 slot 的時刻。

| 本層命中 | 觀察器下單層耗時 | 有 ready 工作時的內部空檔上限 |
| --- | ---: | ---: |
| 10／10 | 1.056 ms | 0.162 ms |
| 4／10 | 1.628 ms | 0.424 ms |
| 0／10 | 2.056 ms | 0.292 ms |

計算方式：對每個 expert 取「讀取完成或本層開始」到 `async_eval` 開始的區間，
取聯集，再與目標程序沒有 GPU 活動的區間相交。表中排除第一段 GPU 工作開始前、
以及最後一段 GPU 工作完成後的時間，避免把初始準備與結尾回報全算成中途停頓。

**上限不是加速預測。** 它還包含必要的 CPU 建圖、排程與觀察器成本，
也不表示整顆 GPU 閒置。637 個 signpost 呼叫耗時中位數約 0.958 µs；
原生 stamp 與 trace 時鐘的 offset 全距 22.167 µs，因此不做次微秒因果判讀。
觀察器明顯拉長測量時間，不能用它和未開觀察器的候選比速度。

### 1.2 有限候選：只合併已常駐專家的提交

把本層開始時就常駐的專家各自建圖，放在同一次 `mx.async_eval`。
需要讀取的專家仍在到達時各別提交，不等最慢的讀取，不增加 slots，不改 QMM 形狀。

未開觀察器，以 A–B–B–A 四個新程序，每個命中條件每程序 32 次：

| 本層命中 | 原版 | 合併常駐提交 | 單層時間縮短 |
| --- | ---: | ---: | ---: |
| 10／10 | 0.618 ms | 0.490 ms | 20.75% |
| 4／10 | 0.961 ms | 0.942 ms | 1.93% |
| 0／10 | 1.150 ms | 1.120 ms | 2.66% |

384 次輸出 hash 全部相同。0 命中時候選實際沒有合併工作，仍出現 2.66% 差距，
它是本輪的波動參考；因此不能採信 4 命中時的 1.93% 為收益。
高命中是正向部件訊號，實際模型收益仍取決於命中分布與讀取是否遮住這項成本。

另用相同修正版觀察器，分別重錄原版與候選，各 4 次：
10 命中的 GPU 活動區間從 **11 → 2**，4 命中從 **12 → 9**，0 命中維持 **12 → 12**。
這次有原生證據證明實際 GPU 工作批次減少，不只是 Python 呼叫數下降。
區間標籤仍是 command buffer，不代表 QMM kernel 數目同樣減少。

## 2. Qwen packed cache：先選資料，再還原

目前 Qwen 的主 K/V packed 選項使用 **8-bit affine、group size 32**；
索引 cache 的選項另外使用 4-bit。兩者預設皆關閉。
本次主 K/V 的 4-bit 表格只是額外研究模式，不是 App 已提供的主 K/V 設定。

原版 `QSAQuantizedCache.update_and_fetch` 先把 K/V 歷史全部解量化，
`QSAAttention._bounded_attention` 再選所需列。
候選對 payload、scales、biases 使用相同 IDs 選列，再解量化；保留後續
QK、mask、softmax、PV 的程式，indexer 的全歷史計算也保留。

### 2.1 完整暫存確實是需要檢查的成本

MLX 0.32.2 的 `quantize_impl` 先分配 `out.nbytes()`，解量化工作數由輸出大小決定，
再另外執行 gather；沒有在此入口根據下游 selected IDs 縮小輸出。
見 [MLX 原始碼](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp#L185-L287)。
本機輸出的 lazy graph 也保留這個先後順序，且分別計量的 peak active memory 支持
確有可減少的暫存。這些證據不等於量到了實際 DRAM bytes 或每條 kernel 的執行時間。

16,385 個歷史 token，2 KV heads、head dimension 256、BF16：
單是完整 K+V 還原結果就有 **32.002 MiB**；Decode 最多選 2,052 列，
包含尾部占位，對應結果約 4.008 MiB。實際 peak 還包含 gather、indexer、scores 等暫存，
所以不能直接把這兩個容量的差當成全部節省量。

### 2.2 相同 packed 模式的部件比較

使用真實 Qwen attention 形狀、固定亂數輸入，沿用既有 benchmark helper，
每組 A–B–B–A，每段預熱 3 次、測量 15 次。
固定已量化的輸入，兩邊都不計 cache append／quantize；保留整個 QSA indexer 與 attention。
沒有在原版 gather 前插入強制同步。

24 組全部保持 selected K/V 與 attention 輸出一致、有限值：

- 4／8 bits；歷史 2,047、2,048、2,049–2,052、8,192、16,385、32,768。
- 尾部四種餘數、重複／倒序 selected IDs。
- 8,192 歷史下的 1／4／16／128 queries；後三者仍依原版每 4 queries 一批。

目前可用主 K/V 設定的 **8-bit、單 query**：

| 歷史 | 原版 QSA | 先選列 QSA | 時間縮短 | 額外 peak：原版 → 候選 |
| --- | ---: | ---: | ---: | ---: |
| 8,192 | 0.699 ms | 0.584 ms | 16.57% | 47.10 → 26.44 MiB |
| 16,385 | 1.196 ms | 0.759 ms | 36.52% | 107.89 → 46.17 MiB |
| 32,768 | 1.698 ms | 1.120 ms | 34.03% | 174.94 → 85.28 MiB |

額外 peak 是該次 QSA 之前 active memory 以上的最高值；每個 arm 的輸入常駐量相同，
先同步、清除 allocator cache，再重設 peak。它不是整個 App 或完整模型的 peak。

4-bit 的 16,385／32,768 單 query 分別縮短 40.81%／41.63%，仍只作研究參考。
8-bit、8,192 歷史、4／16／128 queries 縮短 10.17%／3.98%／12.49%。
本輪沒有出現「多 query 一定更慢」，但 128-query 的額外 peak 幾乎相同，
不能保證 Prefill 也有 Decode 同等的時間或記憶體收益。
完整請求候選先限於單 query；不改 Prefill 多 query 路徑。

## 3. 完整請求檢查

兩個方向已分別完成 A–B–B–A，共 8 次完整請求。fresh process、LRU 3072 slots、4 個讀取 workers、
greedy 64 outputs、Prompt Cache off、目前的分開 Prefill I/O 開啟。

- 提交候選：繁中技術 4K，使用普通 BF16 K/V。
- 先選列候選：code 16K，主 K/V packed 8-bit 開啟、index packed 關閉。

輸出 tokens 有差異、有新增 swapout，或額外 MLX／RSS peak 超過 1 GB，停止採信速度。
兩對比較的 request 縮短至少 5% 或 Decode throughput 增加至少 5%，才視為延伸訊號。
系統檔案快取未清除且沒有每次相同的 primer，因此此矩陣仍不是冷 SSD 或最終採用證據。

### 3.1 實際結果

所有請求都完成 64 outputs；同一方向的四次 token hash 完全相同，
要求讀取的 expert bytes 也相同，8 次全部沒有新增系統 swapout。
這不是兩種不同 prompt 之間的 token 一致比較。

| 方向 | Decode 第一對 | Decode 反向第二對 | Decode 變化中位數 | Request 時間變化中位數 | 結論 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2：常駐專家合併提交，4K 繁中、普通 K/V | +5.96% | +1.24% | **+3.60%** | −0.19% | 未達 5% 延伸線 |
| 3：先選列，16K code、packed 8-bit K/V | +15.12% | +15.49% | **+15.31%** | −1.45% | 通過本輪延伸線 |

變化是每對「候選 ÷ 原版 − 1」；Decode 正值較快、時間負值較短。
每個方向只有兩對，區間是觀察值範圍，不是信賴區間。

選項 2 的 Decode 原版／候選依序為 9.677／10.253、10.055／10.179 tok/s。
MLX peak 僅增加 2,560 bytes，RSS 也未增加到門檻；停止原因是收益小且不穩，
不是數值、slots 或記憶體問題。
候選記錄 3,120 個單 query expert 呼叫，平均命中 7.13／10、全命中約 15.61%；
這包括 runtime 的單 query 準備呼叫，不能直接當成純 Decode 的命中率。

選項 3 的兩對原版／候選為 9.380／10.799、8.554／9.879 tok/s。
MLX peak 約 **18.515 → 18.450 GB**，每對少約 64.9 MB；RSS peak 幾乎不變。
這符合減少暫存的方向，沒有減少持久保存的 packed cache。
每次候選實際使用 780 次先選列 attention，包含符合條件的單 query 準備／尾端呼叫。

16K 的 Prefill 約 106–111 秒，Decode 只有約 6 秒，因此 Decode 的 15% 改善，
在這個短輸出請求中只變成約 1.45% 的整次請求縮短。
不能把它宣稱為「Prefill 快 15%」或「整個請求快 15%」。

### 3.2 驗證與分析修正

- 24 組同 packed 部件數值檢查通過；384 次提交部件輸出 hash 相同。
- 兩個方向各 4 次完整生成通過 token 一致、讀取量一致與 swapout 檢查。
- 既有 `test_qwen_quantized_cache.py` 2 項、`test_qwen_ready_decode.py` 1 項通過。
- 原生 trace、target-only CSV、source snapshots 與每次輸出 hash 均保存。

沿用的 SSD summary helper 原本只對未知名稱檢查 Prefill／request 改善，
會漏掉本研究事先寫明的 Decode 門檻。已修正研究 runner 的分類，重新計算兩份 summary。
原始 summary 另存 `result-original-summary.json`；**耗時、token 與配對百分比都未修改**。

### 3.3 後續條件

1. **選項 2 本輪停止整合。** 保留可重播時間軸與原型；只有在新的高命中工作量、
   或能單獨證明更大提交空檔時，才重開。不要以 20.75% 的全命中單層數字取代完整模型結果。
2. **選項 3 保留為現有 packed 8-bit 選項內的候選。** 下一輪先補繁中／工具／code、
   4K／16K／32K 與較長輸出，記錄相同 packed 的 tokens、cache trim／持久化／續聊一致性、
   request／Decode 分布與 RSS／MLX peak。輸出或 cache 行為不一致、swapout、記憶體增加超過門檻就停止。
3. **是否建議使用者開啟 packed 是另一個問題。** 要另外與普通 BF16 比較任務品質與完整請求成本。
   這輪沒有做該品質比較，不因此更改預設，也不擴充成主 K/V 4-bit 設定。

## 4. 重現與證據位置

原始資料：`scratch/prefill-decode-2-3-2026-09-15/`。
此目錄保存來源副本、實際輸出 token hashes、DOT graphs、原生 trace、signpost XML、
目標程序 GPU CSV、CPU 事件、各次耗時與記憶體資料。

- `packed-01/result.json`：24 組同 packed 部件測量。
- `submit-baseline/result.json`：未開觀察器的原版基準。
- `submit-observer/result.json`、`submit-analysis.json`：對齊時間軸與空檔上限。
- `submit-abba-summary.json` 與四個 `submit-abba-*/result.json`：合併常駐提交的部件比較。
- `full-packed/`、`full-submit/`：分開的完整請求比較。
- `trace-control/`、`trace-resident_batch/`：原生 GPU 工作批次比較。

可放入版本控制的機器可讀證據見
[benchmark 入口](../docs/benchmarks/2026-09-15-qwen-submit-packed/README.md)。

研究入口：

```sh
PYTHONPATH=runtime .venv/bin/python research/qwen_packed_select_probe.py --output scratch/<new-packed-run>
PYTHONPATH=runtime .venv/bin/python research/decode_submission_probe.py --output scratch/<new-submit-run> --variant resident_batch
PYTHONPATH=runtime .venv/bin/python research/qwen_two_three_run.py --candidate packed_select --output scratch/<new-full-packed-run>
PYTHONPATH=runtime .venv/bin/python research/qwen_two_three_run.py --candidate resident_batch --output scratch/<new-full-submit-run>
```

原生追蹤需先編譯 `research/decode_probe_signpost.c` 為 dylib，再用 `xctrace record`
選 `Metal System Trace`＋`Points of Interest`，傳入 `--signpost <dylib>`。
使用 `research/analyze_decode_submission.py` 分析匯出的目標 GPU 與 signpost XML。

## 5. 邊界

這輪主要研究 Qwen，沒有新增 V4／V4.1 效能結論。
同 packed 輸出一致，不代表 packed 與普通 BF16 的模型品質相同。
未修改產品預設，未打包或發布，也沒有把單層加速乘上層數當成整機收益。
