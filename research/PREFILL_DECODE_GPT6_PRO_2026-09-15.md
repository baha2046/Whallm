# Prefill／Decode 技術整理與 GPT 6 Pro 討論

日期：2026-09-15。這是程式碼查核與研究討論，不是新的效能測量。
目前 runtime 仍以 [docs](../docs/README.md) 為準；本文不代表採用或執行新方案。

## 結論

已在使用者指定的 GPT 6 Pro 完成兩輪討論，並以本機程式碼補正第一輪假設。
建議優先順序是 **M5 實際 kernel 查核 → Decode 已就緒工作的提交時機 → Qwen packed cache 選列**。
這是按驗證成本與目前預設路徑的相關性排序，不是已量出的瓶頸排名。

Prefill 的關鍵是保留資料提前到達，並確認預讀是否拖慢同期 GPU；
Decode 的關鍵是每個必要 expert 何時到達、何時真正算完。
更低位元格式、更多預讀、更多 ANE 分工，都沒有必然加速的證據。
完整的三個後續選項與停止條件見第 9 節。

## 1. 查核範圍

- 工作樹起點：`6cdbdce2fba0085c035c982be1d18bfc7387f3bf`，查核前乾淨。
- v1.1.7 release source：`062624ee974febb0b0ee07e228552998f260389d`。
- 本機：Apple M5 Pro、64 GiB、macOS 26.6.2（25G83）。
- 本機 Python 套件 metadata：MLX 0.32.2、mlx-lm 0.31.3。
  [requirements](../requirements.txt) 固定 MLX 0.32.2，mlx-lm 則固定 Blaizzy fork commit
  `5c10538136b9038b9626c134612b08afc18d697a`；不能只以版本名稱識別 fork。
- 透過使用者指定的 in-app browser 進行討論，送出時模型選擇器顯示 **6 Pro**。
- [GPT 討論連結](https://chatgpt.com/c/6aa8a3ff-7ca0-83e8-b41c-91890bf8bc8c)。
- 本次未修改 runtime、預設設定或安裝模型，未執行完整模型 benchmark。

### 證據衝突

[README 的參考表](../README.md#benchmarks) 已列出 V4.1 在 M2 Max 的 1K–16K 結果，
但 [V4.1 文件](../docs/DEEPSEEK_V41.md#current-boundary) 與
[加速功能文件](../docs/MODEL_ACCELERATION.md#驗證範圍) 仍記錄完整模型未驗證。
README 明確說表格缺 build revision 與 cache state。
因此目前可說「有速度參考紀錄」，不能說「完全沒有執行紀錄」，也不能把表格視為新加速選項的完整驗證。
第一輪給 GPT 的摘要沿用了 docs 的限制；後續討論補正此差異。

## 2. 專案如何執行模型

Swift 負責安裝、驗證與 App；Python／MLX 負責模型計算。
目前一次處理一個生成請求，batch size 為 1。
common tensor 常駐統一記憶體；routed expert 的權重放 SSD，需要時讀到固定大小的 slot。
slot 是一塊可供 Metal 使用的 MLX 記憶體，不是整個模型的副本。

| 合約 | DeepSeek V4 | Qwen3.8 Flash Next | DeepSeek V4.1 |
| --- | ---: | ---: | ---: |
| 層數 | 43 | 48 | 40 |
| 每層 routed experts | 256 | 512 | 384 |
| 每 token 選取數 | 6 | 10 | 6 |
| hidden width | 4096 | 2560 | 5120 |
| expert intermediate width | 2048 | 640 | 2304 |
| 每 expert bytes | 13,369,344 | 2,611,200 | 18,800,640 |
| App 新設定 slots | 1152 | 3072 | 1152 |
| installed expert 格式 | 原生 FP4 | 由 FP8 轉成 MXFP4 | 原生 FP4 |
| 額外大型查表 | — | 約 51.20 GB N-gram | 約 202.76 GB Engram |

來源：[架構](../docs/ARCHITECTURE.md)、[Qwen](../docs/QWEN.md)、
[V4.1](../docs/DEEPSEEK_V41.md)、[模型描述](../Sources/DeepSeekRepack/Resources/ModelPackages.json)。
已保存的自訂 slots 不會被新預設覆蓋。

只算 routed experts，假設所有 lookup 都 miss、每個 blob 都讀取一次：
V4 每 token 約 3.45 GB、Qwen 約 1.25 GB、V4.1 約 4.51 GB。
這是 `層數 × top-k × blob bytes` 的容量推算，不是實測 SSD 流量；
不含 common tensor、查表、KV、預讀浪費或 speculation，也未扣除 slot／OS cache 命中。

| 純 expert payload 容量推算（GiB） | V4 | Qwen | V4.1 |
| --- | ---: | ---: | ---: |
| 單個 full-layer buffer | 3.188 | 1.245 | 6.724 |
| 當層＋下一層 | 6.375 | 2.490 | 13.447 |
| 全部設定 slots 配滿 | 14.344 | 7.471 | 20.171 |

這些是容量，不是同時發生的 runtime peak；Prefill 會釋放 slots，不能把三列直接相加。

## 3. 現行路徑與值得注意的差異

### Prefill：讀取整段輸入

- **V4**：未快取輸入達 1024 tokens 時使用按層 Prefill。
  當層資料可在 attention 前開始預讀，下一層在當層 MoE 前排入；
  attention 分 chunk，MoE 使用最高 4096-token 自動 tile。
  cache-only 最後一層略過 MoE，因此典型 4K 有 42 個 full expert layers。
- **Qwen**：128 tokens 起使用按層 Prefill；完整 expert layer 只讀一次供該層 chunks 使用。
  下一層預讀有開關，預設關閉，不能把 V4 的 41/42 預讀命中直接套用到 Qwen。
- **Qwen 專家排序**：先將同 expert 的輸入放一起，算完還原；兩次
  `gather_qmm` 刻意保持 `sorted_indices=False`。
  先前開提示的候選曾在繁中任務出現可重現退步，不能直接改成 true。
- **V4 與 Qwen**：長輸入的 batched Prefill 前釋放舊 slots，避免 full-layer buffer 與舊權重重疊。
  模型及 prompt cache 保留；這不是速度提升證據。
- **V4.1**：按層 Prefill、候選索引與 CED 已有實作，預設關閉。
  CED 只在保留共享 KV 與必要前文後略過後段舊 token；不能任意截短輸入。
  開啟按層 batched 路線時，現行程式也會呼叫 `release_prefill_slots`。

程式入口：[V4 Prefill](../runtime/deepseek_v4_ssd/model.py)、
[Qwen Prefill](../runtime/deepseek_v4_ssd/model_support/qwen.py)、
[V4.1 Prefill](../runtime/deepseek_v4_ssd/model_support/deepseek_v41.py)。

### Decode：逐個生成 token

- `iter_ready` 先交出已常駐 expert，miss 用工作執行緒讀取，完成一個就交出一個。
- 各 expert 的運算可由 `mx.async_eval` 提交；最後依 router 原始順序組回結果再加權。
- 現行 pool 每個 slot 是獨立 MLX array，按需配置；不是歷史 cross-arena grouped Decode。
- 每次取得 router indices 的 CPU 轉換可能需要等待上游 GPU；等待時長與 Python 呼叫成本需分開。
- App 新設定採 LRU；共用 runtime／舊 catalog 的 fallback 仍可能為 LFU。
  所有比較都要保存實際設定，不能寫成全專案只有一種淘汰方式。

程式入口：[expert cache](../runtime/deepseek_v4_ssd/expert_cache.py)、
[Qwen experts](../runtime/deepseek_v4_ssd/qwen4_exp.py)、
[App defaults](../Sources/DeepSeekV4SSDApp/ServerController.swift)、
[App catalog 編碼](../Sources/DeepSeekV4SSDApp/ModelLibrary.swift)。

### 記憶體與查表

`preadv` 寫入 slot 的 memoryview，已避免先建立完整 Python expert bytes 再複製。
不能因為換成 MTLIO 就假設消除了一次現存的完整 CPU→GPU 拷貝。
full-layer 預讀仍會分配完整 MLX array 並 `mx.eval`，讀取完成後才交給 QMM。

Qwen QSA 新的低位元 cache 會先解量化回歷史陣列，再由 attention 選取列。
「保存較小」與「計算時只處理被選中的資料」尚未等同。
Qwen N-gram 目前經 NumPy 選列、複製、FP8 解碼，再轉為 MLX BF16；
這條 CPU／GPU 交接值得量測，但尚無成本占比。
V4.1 的 PackedRows 保留原生格式，取出時還原；仍需檢查 attention 的取用範圍。

來源：[QSA cache](../runtime/deepseek_v4_ssd/qwen_quantized_cache.py)、
[QSA 與 N-gram](../runtime/deepseek_v4_ssd/qwen4_exp.py)、
[V4.1 PackedRows](../runtime/deepseek_v4_ssd/deepseek_v41/packed_cache.py)。

此處的「尚無成本占比」指目前版本、目標 workload 的完整等待時間。
不能漏掉舊資料：2026-09-06 M2 Max 的同步診斷曾量到 Qwen N-gram lookup
0.139 秒，未達該次 5% 入場線；同期的 QSA 歷史 pooled-key cache 理想化測試也未顯示可用收益。
所以這兩項不是從未研究的新方向，只能在更長 context、不同晶片或新 packed cache
等條件讓占比改變時重新評估。
見 [Qwen 第一輪歸因](QWEN_PREFILL_DECODE_RESEARCH_2026-09-06.md#第一輪-attribution-與-h2-reopening)。

### ANE 與 GPU Neural Accelerator

專案既有 ANE 分工採私有 framework，只用 batch 1、1024-token projection，
output channels 分給 GPU 與 ANE，最後串接。
Qwen 預設 ANE 25%；DeepSeek 的 ANE 新開關預設關閉。
CPU array 轉換、FP16 捨入、等待與串接都要算進成本。

M5 GPU core 裡的 Neural Accelerator 與 ANE 是不同硬體。
Apple 的 TensorOps 可在 GPU shader 中混用矩陣運算與一般運算，並提供分塊及片上暫存能力。
但沒有資料支持把 CPU/GPU/ANE 的快取視為完全相同，或假定 DRAM 存取不會互相影響。
來源：[Apple M5 GPU 技術說明](https://developer.apple.com/videos/play/tech-talks/111432/)、
[Metal TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)、
[MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)。

Apple 標示 M5 Pro 記憶體頻寬最高 307 GB/s，M2 Max 為 400 GB/s。
較新晶片與較高記憶體頻寬是兩個不同條件；不同 workload 的勝負仍需量測。
來源：[M5 Pro 規格](https://www.apple.com/au/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)、
[M2 Max 規格](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/)。

版本也要分清：Apple 在 macOS 26 更新加入的 4／8-bit integer tensor，
不等於 FP4／FP8 加 E8M0 scale plane；後者由 WWDC26 文件列為 macOS 27 能力。
本機仍是 26.6.2。因此「直接把 MXFP4 交給新格式 TensorOps」不能視為目前可用。
本機可研究既有的解碼與矩陣路徑，但仍需實際 SDK／kernel 查核。
來源：[Apple TensorOps 格式與版本](https://developer.apple.com/videos/play/wwdc2026/330/)。

## 4. MLX dispatch 的新查核線索

MLX 0.32.2 上游已包含 NAX 的量化矩陣路徑，但仍依 shape、排序提示、dtype、
OS、硬體及編譯設定分派。現行 Qwen 排序後保留 M=1 且不送排序提示，
從上游判斷式推導，典型 Prefill 呼叫可能走 gather QMV。
V4 則會送出排序提示。這是**來源推論**，尚未以本機 Metal capture 確認 kernel。

因此第一個便宜查核是記錄真實 shape、dtype、strides 與 kernel 名稱，
確認不同模型是否使用不同路徑。不能宣稱「MLX 完全沒用到 M5 加速器」，
也不能跳過 Qwen 舊品質反例直接切換提示。

來源：[MLX 0.32.2 quantized.cpp](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp)、
[NAX availability](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp)、
[舊 Qwen 提示／無提示實驗](QWEN_GROUPED_EXPERTS_2026-09-06.md)。

## 5. 歷史結果不能省略的條件

| 方向 | 歷史結果 | 本輪的解讀 |
| --- | --- | --- |
| V4 selective Prefill | 少讀約 25%，TTFT oracle 改善 −5.21% | 少讀可能損失原本的跨層重疊 |
| Qwen no-hint Prefill | M5 Pro 1K／16K 首次回覆快 7.59%／13.30% | 已採用；第一輪對話誤寫 M2 Max，後續補正 |
| Qwen direct selected-row QSA | component 約快 36%，但生成分歧 | 更換 reduction 次序需重新驗證 |
| Qwen cross-arena Decode | M2 Max Decode 快 9.59–16.42% | 已從現行 runtime 移除，不是今天可直接開啟的功能 |
| Qwen bounded MTP | acceptance 96.30%，Decode 快 40.01%，request 只快 2.48% | 每輪停頓與 TTFT 抵銷收益，P95 退步 |
| w13／w2 分段串流 | Decode 慢 4.26% | 沒有不同運算窗口或 kernel，不重跑同版本 |
| MTLIO | native byte／事件順序通過，MLX 交接未過 | 先查 0.32.2 的公開 ownership／event 支援，再談整合 |
| LZ4／LZFSE | 壓縮收益不足以抵消解碼 | FP4 bytes 已緊密，不因 SSD 慢就推定壓縮有利 |

來源：[研究結論](../docs/RESEARCH.md)、[Qwen MTP](../docs/QWEN.md)、
[no-hint 整合](QWEN_NOHINT_INTEGRATION_2026-09-06.md)、
[MTLIO](MTLIO_EXPERT_STREAMING_2026-08-27.md)、
[SSD 瓶頸研究](SSD_STREAMING_BOTTLENECKS_2026-08-31.md)。

## 6. 量測邊界

Prefill／Decode 都要從同一時間軸找真正使下一步等待的工作，
不能直接相加 GPU time、I/O time、CPU wait，因為它們可能重疊。
理想化單個重疊階段可寫成 `max(讀取完成時間, 輸入就緒時間) + 後續計算`；
完整請求還要考慮共享頻寬造成兩邊同時變慢。

每個候選需保存：commit、套件、晶片、SSD／路徑類型、prompt／output hash、
輸入及輸出長度、sampling、cache 狀態、slots、預讀設定、wall time、TTFT、
Decode p50／p95、logical bytes、process disk bytes、MLX peak、RSS、swap。
OS cache 未清除就寫 `not purged`；process disk bytes 不能當 expert 專屬實體 SSD 流量。
只有主模型 wall time 的實測改善才能列為專案加速結果。

目前 runtime 的 `prefill_tokens_per_second` 使用 Prefill token 數除以 TTFT，
不能當作純 GPU Prefill 計算吞吐。runtime TTFT 又不含 tokenization 與磁碟 prompt-cache 載入；
面向使用者的完整等待時間，應另外在請求入口及第一個可見 token 計時。
來源：[generation metrics](../runtime/deepseek_v4_ssd/generation.py)。

驗證順序應為：原始碼／實際 dispatch → 小型真實 shape → 數值與狀態 → 完整交錯 A/B。
Greedy token 相同仍不等於所有 logits 或 sampling 分布相同；
近似模式、額外量化與新捨入路徑必須分開記錄。
詳見 [效能指標與比較方法](../docs/PERFORMANCE.md)。

### 先找上限，再決定是否寫 kernel

設某段工作占沒有重疊的總時間比例為 `f`，該段加速 `s` 倍，
在其他條件完全不變的簡化模型下，總加速上限為 `1 / ((1-f) + f/s)`。
不能把某個同步 probe 的 section 百分比直接代入，因為 probe 可能改變原本排程。
若刪掉某段計算後只是讓原本被遮住的 I/O 顯露，實際收益會更小。

Prefill 的可用讀取窗口，要以「下一層最早開始需要該 buffer 的時間」定義。
預讀更早可能搶走目前 GPU 的 DRAM 頻寬；更晚則可能造成等待。
因此讀取完成的 hit 計數只回答是否趕上，不回答預讀期間是否拖慢 GPU。

Decode 要保存每層 expert 的 miss 數、每個讀取完成時間、GPU 提交與完成時間。
同樣的總命中率，若慢讀集中在最後一個必要 expert，尾端延遲仍可能很大。
kernel 合批若要等待尚未到達的 expert，就可能抵消減少提交的好處。
保留 expert 原始加權順序、slot 使用到 GPU 完成才釋放，是任何候選的必要條件。

## 7. 可比較的研究方向

下列是待驗證的問題，並非已採用 TODO；每項都需先證明目標成本足夠大。

| 方向 | 適用範圍與這次要問的差異 | 最便宜的第一步 | 停止條件 |
| --- | --- | --- | --- |
| M5 真正使用哪個 kernel | 分開 V4／Qwen、Prefill／Decode；確認目前形狀與 no-hint 設定的分派 | 保存真實 shape／dtype／strides，擷取實際 kernel 名稱 | 已走合適路徑，或相關運算占比不足；不直接重開舊 hint 候選 |
| Packed cache 先選列再還原 | Qwen 新 QSA packed 路線；相對「先還原全歷史」減少暫存 | 固定相同 packed bytes 與 selected IDs，比較兩條解碼路徑 | 任一值／mask／tie 不符，或選列本身成本抵銷收益 |
| 預讀依使用時間安排 | 保留 V4 full-layer 策略；另看 Qwen／V4.1 的可選跨層預讀 | 比較 GPU 單獨運算與重疊讀取下的延長、read deadline 與 peak | 讀取趕上但 GPU 變慢，總時間沒改善，或壓縮／換頁增加 |
| V4.1 CED／候選索引 | 新接入的模型固有結構；先核對完整 checkpoint，再逐項開啟 | 記錄實際略過 layer-tokens、候選使用／fallback 次數與保留前文 | logits／續寫／共享 KV 不符；實際幾乎沒有可略過工作 |
| Decode 只優化已就緒工作 | 不等 miss 才合批；先比較單 expert 計算與就緒資料提交成本 | 按 miss 數分類時間軸，找出 GPU 空檔與每 expert kernel 成本 | 必須複製整批權重、改原加權次序或等慢讀，才能得到局部收益 |
| QSA 僅 Decode 的選列 kernel | 舊 Prefill kernel 失敗，單 token capture 曾 exact；這是不同適用範圍 | 先在多種真實 Decode captures 核對 tensor-exact | 任一數值／路由分歧，或端到端占比不足；不能以舊單 capture 放行 |
| 有條件 ANE 分工 | 新 M5 GPU baseline 下，重新看現行固定 25% 是否合適 | 同 shape 比較 GPU-only 與現有分工，包含 CPU 轉換、同步及串接 | projection 變快但 TTFT 未改善，或品質／峰值不過 |
| 確定地址的查表預讀 | 尤其 V4.1 Engram；地址可從已知 token 推得，不必猜 expert router | 先確認地址何時可知、實際 page wait 與重複 row 比例 | lookup 占比不足，或與 expert 讀取競爭使總時間增加 |

Qwen 額外量化相對預設已可能改變輸出；「先選列再還原」即使與 packed 對照完全相同，
也不能宣稱與未量化預設等價。ANE FP16 與不同矩陣 reduction 路徑同樣需獨立品質驗證。

「先選列再還原」應先從 Decode 的單 query 檢查：歷史很長、選取上限約 2048 tokens
時，才有明確可省資料範圍。Prefill 一次有很多 queries，同一歷史列可能被不同 query
重複選中；若改成反覆解量化同一列，可能比原本整段還原一次更貴。
因此要記錄 selected IDs 的聯集與重複次數，分開比較 Decode 與 Prefill，不能共用一個收益推算。
負責決定 selected IDs 的 indexer 仍可能需要完整歷史；此候選不會自動省掉那部分。

更低 bit 的 expert、少選 expert、改 router 或訓練預測器屬另一條研究線，
需要獨立資料集與品質決策，不包含在這次低風險量測建議中。

## 8. GPT 討論與交叉核對

第一輪 GPT 6 Pro 完整回覆提出九個方向；核心建議是：

1. 保留完整、提前的 Prefill 讀取，先找預讀與 GPU 是否互相拖慢。
2. Decode 看 expert 到達時間與提交空檔；不要為了合批等待未到的權重。
3. 優先檢查 packed cache 的選列／解量化順序，再看 V4.1 結構上可省的工作，最後才擴大 ANE。
4. 相同容量的 cache 策略應比較實際省下的等待，而非只比較命中率。
5. MTLIO 必須先補公開 MLX 交接合約；後端看得到 event 類型不代表 Python 已能安全使用。

本機查核補回了四個會影響排序的條件：

- Qwen packed cache 預設關閉；其內部變快不能直接叫做預設模式加速。
- Qwen 一次 QMM 的輸入分塊大小，不能用整個 prompt 長度代替。
  例如一段 4096-token 輸入若分成 1024-token chunks，平均每 expert 的單 chunk assignments 為 20，
  而不是用整段計算的 80；真實路由分布還需另記錄。
- QSA 歷史索引與 N-gram 有舊的停止證據，不可當成全新方向。
- V4.1 有 README 參考表、預設 1152 slots，且新按層 batched 路徑會釋放 slots。

第二輪已完成。GPT 接受補正，撤回把 V4 預讀條件套用到 Qwen 的判斷，
不建議重啟舊 pooled-key cache／N-gram 方向；並將預設路徑的查核排到 packed cache 前面。
它也指出 Prefill 先選列可能重複解量化，與本機對 query_chunk=4 的分析一致。

這兩輪是模型建議與來源查核，沒有產生新的 runtime 效能證據。
完整問答保留於上方 ChatGPT 連結，包含第一輪原文與第二輪補正。

## 9. 三個後續選項

**後續決定：使用者指定先研究 2、3。** 已開始
[分開的實測](DECODE_SUBMISSION_PACKED_SELECT_2026-09-15.md)，
以下的「推薦先選 1」保留為當時建議，不限制這次授權。

### 選項 1：確認 M5 目前實際使用的 kernel（建議先做）

- **最小驗證**：固定現行 Qwen 預設、一層與一個代表性 1024-token chunk，
  保存兩次 expert QMM 的 shape、dtype、strides、排序旗標與 Metal kernel 名稱。
  只辨識，不改旗標或運算。
- **優點**：不引入數值變化，直接確認目前「可能走 QMV」的推論，與一般使用者設定相關。
- **限制**：知道走哪個 kernel，還不能證明它低效，也不會立即提供加速。
- **停止條件**：若不是推測路徑，撤回假設；若確實是 QMV，這一步也只完成辨識。
  是否改 kernel 需再看暴露成本，不能據此打開已失敗的排序提示。
- **產物**：真實輸入形狀、實際 kernel 證據、環境與設定，不附未測的加速百分比。

### 選項 2：檢查 Decode 已就緒工作的提交空檔

- **最小驗證**：固定一層與命中集合，對齊 expert 讀取完成、`async_eval` 呼叫、
  GPU kernel 起訖，以及 slot 最後一次使用完成。
- **優點**：直接針對目前 Decode；先保持個別 expert 的原 kernel、shape 與合併順序。
- **限制**：少一次 `async_eval` 不等於少一次 Metal 提交，量測本身也可能干擾排程。
- **停止條件**：已就緒資料沒有因提交延後造成等待，就不寫合併提交候選。
  若必須等待未到的 expert 或擴大權重常駐區，則超出這個候選的範圍。
- **產物**：可重播的事件時間軸與可消除空檔的上限；不是命中率表格。

### 選項 3：檢查 packed cache 先選列的收益

- **最小驗證**：固定同一個 packed 模式，先用超過 2048-token 的歷史與 Decode 單 query，
  確認完整解量化暫存是否真的形成、耗時及大小；再比較相同 selected IDs 的資料準備。
- **優點**：具體減少可能不需要的還原工作，可先保持 QK／softmax／PV 不動。
- **限制**：目前預設關閉；Prefill 多 query 可能重複還原相同列，indexer 的全歷史需求仍在。
- **停止條件**：沒有可省的全量工作、選列重複成本抵銷收益，或所選 K/V 無法逐位元一致。
- **產物**：同 packed 模式的數值、時間與峰值比較；若繼續，再做對預設 BF16 的獨立品質與完整請求比較。

推薦先選 **1**，因為它能在不改變模型行為的情況下排除最基本的硬體使用誤判。
選項 2、3 保留為下一階段候選，不同時混入第一輪。
本節保留最初建議；後續已依使用者選擇開始 2、3，見本節開頭的實測連結。

### 可以對外宣稱到哪裡

即使選項 3 通過，也只能說可選 packed 模式在指定條件下變快。
它相對自己的舊實作等價，不代表相對預設 BF16 品質不變，亦可能仍比預設慢。
要宣稱一般使用者受益，必須另外說明適用設定與範圍，並完成對預設模式的品質及請求速度比較。
