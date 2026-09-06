# Prefill／Decode 新一輪研究

日期：2026-09-05。核對 commit：`7dc9cf8f050c75def77c0563cd7b8ac03f2d8435`。
初稿完成專案盤點、既有結果重算與論文初步查核。
使用者已於 2026-09-05 選 B；之後的新量測見 [B 第一輪](PREFILL_DECODE_B_ROUND1_2026-09-05.md)、
[H2 計時](GATED_DELTA_PREFILL_2026-09-06.md)、[QSA 批次](QWEN_ATTENTION_GRANULARITY_2026-09-06.md)
與 [expert 分組](QWEN_GROUPED_EXPERTS_2026-09-06.md)。runtime 程式尚未改動。

## 先讀這些結論

Prefill 是讀懂輸入的階段；Decode 是接著逐字產生回覆的階段。
目前值得追求兩種收益：縮短第一次看到文字的等待，以及縮短後續每個 token 的等待。
兩者要分開量測，也要一起看整個 request。

最初盤點得到四個研究判斷，以下保留當時的推論，不能覆蓋上述後續量測：

1. **Prefill 優先找重複搬移與中間資料。** Qwen Grouped-KV 的成功，支持繼續檢查記憶體流量；單純少讀 expert 的舊實作反而失去運算與讀取的重疊。
2. **Decode 保留一條低成本重啟路線。** 舊 causal cache 候選減少 18.44% miss，因相對理想上限的門檻而停止，還沒有 runtime 速度結論。
3. **另開真正不同的假設。** 檢索式草稿、非前綴文件重用、分群共用 expert 權重，都值得先做小實驗。論文適用條件與本機差異見下文。
4. **先決定如何對待品質差異。** 改變權重、路由或文件狀態重用，不能只用文字「看起來正常」決定採用。

## 專案目前有什麼

目前行為以 [docs 入口](../docs/README.md)、[架構](../docs/ARCHITECTURE.md)、
[Qwen 合約](../docs/QWEN.md)和程式碼為準。研究筆記保存各次實驗的條件與理由。

| 部分 | 現況與研究意義 |
| --- | --- |
| 安裝與 App | Swift repacker 產生 installed model／manifest，SwiftUI App 管理模型，Python／MLX runtime 提供 API。安裝、修復、完整性驗證已有基礎。 |
| 記憶體與 SSD | common tensor 常駐統一記憶體；routed expert 以 expert blob 從 SSD 載入固定 slot。CPU、GPU、ANE 共享記憶體資源，不能直接套用獨立 GPU 的搬移成本。 |
| DeepSeek | 43 個 main model layers，每層 256 個 routed experts；原始 top-6。一般 API／CLI 預設使用 `learned-route-drop-lowest-1`；前三個 hash-routing layers 保持原路由，後續 learned-routing layers 少用最低分的一個 expert。可指定 `exact`，DSpark 維持 exact。 |
| Qwen | 48 層，包含 QSA、Gated DeltaNet、top-10 experts、N-gram／PLE 等元件。API 名稱帶 FP8，但 installed routed experts 使用 MXFP4；不能把它視為 FP8 expert 再壓到 FP4 的研究。 |
| Prefill | 長輸入按層處理，完整讀取一層 expert，搭配分批矩陣運算；DeepSeek 有下一層預取。Qwen 已採用 Grouped-KV QSA，ANE 僅分擔固定 1,024-token shape 的 QSA projection。 |
| Decode | expert cache、依 expert 就緒時間提交運算；DSpark／Qwen MTP 已有研究實作，預設關閉。 |
| Prompt cache | normal persistent format 5；保存可重用狀態，包含可變狀態的獨立 snapshot。不是只有 attention K/V。 |
| 基準設定 | DeepSeek 1,152 slots；Qwen App 預設 4,096 slots、Use MTP false；Qwen ANE ratio 0.25。實驗另外記錄實際設定，不能只寫「預設」。 |

程式碼核對入口：
[RuntimeConfig](../runtime/deepseek_v4_ssd/model.py)、
[generation 與 cache format](../runtime/deepseek_v4_ssd/generation.py)、
[API 預設](../runtime/deepseek_v4_ssd/server.py)、
[Qwen](../runtime/deepseek_v4_ssd/qwen4_exp.py)、
[App 每模型設定](../Sources/DeepSeekV4SSDApp/ServerController.swift)。
Qwen 的 GatedDeltaNet 來自 `mlx_lm.models.qwen3_5`，本輪未查完該依賴的 kernel 實作。

本機查到 Apple M5 Pro、64 GiB 記憶體；這只識別本次研究機器，不構成新的速度測試。

## 最近測速：重算原始 runs，避免只看最高值

來源為 9 月 4 日的 [DeepSeek API artifact](../docs/benchmarks/2026-09-04-130402-api-deepseek-v4-flash-0731.json)
與 [Qwen API artifact](../docs/benchmarks/2026-09-04-132210-api-qwen3-8-flash-next-fp8.json)。
以下各欄獨立取三筆 runs 的中位數，單位為秒或 token/s。

| 模型 | 輸入 tokens | 首字等待 | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek | 1,024 | 17.30 | 59.20 | 7.62 |
| DeepSeek | 2,048 | 17.53 | 116.84 | 7.20 |
| DeepSeek | 8,192 | 40.01 | 204.74 | 7.36 |
| DeepSeek | 16,384 | 78.55 | 208.59 | 7.08 |
| Qwen | 1,024 | 15.17 | 67.49 | 10.36 |
| Qwen | 2,048 | 24.29 | 84.31 | 9.78 |
| Qwen | 8,192 | 74.73 | 109.63 | 9.92 |
| Qwen | 16,384 | 147.19 | 111.32 | 9.44 |

條件：64-token 回覆、greedy、SPEED-Bench mixed，每個長度使用三個不同題目。
source commit 是 `b25ae07` 加 dirty diff，並非本次 HEAD。
所有 runs 回報 prompt reuse 為 0，但 cache 沒有逐次清除，OS page cache 是 `not purged`。
DeepSeek 設 1,152 slots，Qwen 設 4,096 slots；兩者 speculative decoding 關閉。
artifact 沒保存每個 request 的 `approximation_mode`，因此不能把 DeepSeek 這組數字宣稱為 exact baseline。
Qwen 設定中的 ANE active 不能證明每次 request 都有使用 ANE，還需逐次 dispatch 計數。

這組 16K／64 workload 的首字等待占 client wall time 中位數：DeepSeek 89.81%、Qwen 95.21%。
**觀察：這種長輸入、短回覆的體驗主要受首字等待影響。**
它不代表短輸入、長回覆的瓶頸，也不能用來判定 SSD、QSA 或 Gated DeltaNet 各自占多少時間。

重算方法、原檔 SHA-256、完整設定和限制保存在
[證據摘要 JSON](PREFILL_DECODE_EVIDENCE_2026-09-05.json)。
API 原檔只有 output **text** hash；16K 已有執行與測速紀錄，但沒有同等強度的逐 token 正確性證據。
原檔 p95 只有三筆不同題目，nearest-rank 等於最大值，不能當穩定的尾端延遲分布。

## 成功、失敗，以及沒有被排除的可能

以下均為已有紀錄，本輪沒有重跑；M2 Max 與 M5 Pro 結果分開理解。

| 經驗 | 已有證據 | 本輪汲取的教訓／重啟條件 |
| --- | --- | --- |
| Qwen Grouped-KV：已採用 | M5 Pro、4,577-token、四次交錯測試；TTFT 85.36 → 44.88 秒、peak memory 14.79 → 12.84 GB；四次只比對首個 output token。[artifact](../docs/benchmarks/2026-09-02-qwen-qsa-grouped-kv-adoption-quick-gate-m5-pro.json) | 去除 KV heads 的重複展開很有效。這是 quick gate，不能宣稱所有工作或長輸出都加快一倍。 |
| DeepSeek 少用一個 learned expert：已採用 | M5 Pro、五種 4K／256、10 pairs；Decode 中位改善 8.37%、logical bytes 減少 15.14%；10 pairs 都是 256/256 tokens 相同。[artifact](../docs/benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json) | 在已測集合能得到收益；它仍是 approximate，有限樣本一致不證明普遍等價，也不授權任意擴大 drop。 |
| Prompt reuse：功能有效 | format-5 的 Qwen restart／snapshot 修復見 [Qwen 驗證](../docs/QWEN.md)。較早有很大的 warm TTFT 降幅。 | 重用對重複輸入有價值；重跑時需用新 format，包含重啟、分支與續寫，不能只測相同 prompt 的首 token。 |
| Selective Prefill：候選停止 | M2 Max、repeated 4K；bytes −66.78%、memory −15.78%，TTFT 卻 +16.87%。[artifact](../docs/benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json) | 少讀不等於少等。這個候選失去下一層預取；不同的提早規劃／重疊方式仍未被排除。 |
| 拆開讀取 `w13`／`w2`：候選停止 | M2 Max、128／32、四波；Decode −4.26%。[artifact](../docs/benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json) | 局部重疊收益抵不過完整流程的額外成本。重啟需要不同 kernel 或排程。 |
| LZ4／LZFSE 無損壓縮：停止 | M5 Pro、60 cases 都 byte-exact；同時有 Metal 工作時，未通過讀取加解壓的時間門檻。[artifact](../docs/benchmarks/2026-09-01-expert-blob-compression-m5-pro.json) | 壓縮後 bytes 較少，仍需付解壓與資源競爭。未排除直接在壓縮表示上運算，但那需要新 kernel。 |
| Causal cache：未做 runtime 測試 | Qwen、1,152 slots；四個非 repeated workloads miss 中位 −18.44%、沒有 workload regression；只取得 Belady 理想改善的 41.63%，低於當時 50% quick gate。[artifact](../docs/benchmarks/2026-09-01-expert-cache-causal-policy-m5-pro.json) | 這是研究篩選停止，並非實測變慢。現在 App 預設 4,096 slots，須先重新 replay，不能沿用改善比例。可提議新的時間成本 gate；不可事後把舊結果改標通過。 |
| Qwen MTP：Decode 快但體驗退步 | M5 Pro、code 4K／64；原 verifier Decode +65.07%，TTFT +13.10%、request +7.34%、p95 +153.13%。bounded 版本仍未過 request／p95 gate。[合約及結果](../docs/QWEN.md) | 接受率或平均 Decode 不是完整目標。檢索式草稿可移除 draft model 成本，仍需處理 verifier 停頓。 |
| Qwen ANE：已實作，收益未確立 | M5 Pro、4,097-token ABBA 首 token 一致，反向順序結果接近相同。[artifact](../docs/benchmarks/2026-09-02-qwen-private-ane-prefill-exploratory-m5-pro.json) | 要區分「有執行」與「有淨收益」；記錄 compile、dispatch、共享頻寬、整段時間與 cache 狀態。 |
| MTLIO／learned router：前提不足 | 舊 MLX 0.32.0 缺公開 external event handoff；6,000-label frozen-router transfer recall 很低。[歷史 tracker](PLAN_EXECUTION_STATUS_2026-08-27.md) | 不是永久平台限制或 predictor 不可能。須先驗證介面是否改變，或提出新特徵／訓練資料，再重啟。 |

三個資料新舊差異要特別處理：舊 tracker 仍談 format 4；SSD 研究尾段仍寫 Phase 6B 下一步；Qwen 早期合約只列 4K 驗證。
目前應分別讀成：程式已用 format 5、approximate 已完成並採用、16K 有 API 測速但沒有同等 token parity 證據。
這些差異不構成新的效能結論。新一輪入口是 [research README](README.md)。

## 論文支持的研究方向

下表保留每項由論文啟發的初始假設；後續量測與適用條件見各輪報告，不能把假設直接當成已採用成果。
論文只能支持機制與研究動機，不能預先證明本專案會加速；文末列出版日期、查核深度與限制。
「新」表示在本輪搜尋的 `PLAN.md`、`research/`、`docs/` 與 runtime 沒找到同名專項 gate，並非宣稱全世界首次提出。

| ID／目標 | 論文依據與本地假設 | 最小實驗與停止條件 |
| --- | --- | --- |
| H1 Prefill：融合 QSA 的選取、softmax、加權運算 | [P1] 減少中間讀寫和改善工作分配。Qwen 現在每四個 query 分批，以多個 MLX operations 完成；假設可保留 selected indices／mask，減少中間張量與提交成本。屬現有方向的新候選。 | 先量 QSA 在最新完整 request 的時間；固定現有 query chunk，比較一個邊界。若可移除時間不足 5%、路由／長輸出不一致或整體沒快，停止。不是重新測已失敗的 chunk 8。 |
| H2 Prefill：Gated DeltaNet 的分塊執行 | [P2] 提供 gated delta rule 的 chunkwise 計算法。假設目前 Qwen 依賴中的投影、狀態更新或同步仍有可消除成本；不能假設套件沒做分塊。 | 先讀 installed mlx-lm 實作並量各部分；若已有同樣方法或時間占比太小，停止。改分塊須驗證完整 recurrent state 與後續 Decode。 |
| H3 Decode：用實際等待成本評估快取 | [P3] 的多因素 expert cache 支持同時考慮使用紀錄與載入成本。以舊 causal 候選為起點，假設 −18.44% miss 在目前模型／slot 設定下仍可轉成收益。 | 先重新核對 replay 與現在路由；新題目測時間、替換成本、p95。理想模型只作上限，通過依據改為實際 request／Decode。若 miss 降但暴露等待不降，停止。 |
| H4 Decode：expert blob 排列與成批讀取 | [P4] 支持重用與較大的連續讀取。假設高共現 experts 的實體位置有時能降低 I/O 呼叫／等待；本案已有 canonical blob，研究的是跨 blob 排列。 | 先用不同題目的 route trace 計算能合併的 bytes、額外讀取量，再對小型獨立 sidecar 量測；SSD 隨機讀與順序讀接近、或 read amplification 吃掉收益即停。先不重打包完整 installed model。 |
| H5 Prefill＋Decode：共同分配有限記憶體 | [P5] 跨記憶體層成本配置、[P6] prefix reuse。新假設：同一固定總預算下，expert slots、prompt states、臨時 buffer 的邊際收益，比單獨增加 slots 更重要。 | 先用人工固定的兩三種配置找交換曲線，不先寫自動控制器。prefix 無重用與多輪分支分開測；加上 serialize／reload 成本。若只有把記憶體需求推給 OS compressor，停止。 |
| H6 Decode：檢索式草稿 | [P7][P8] 用文字續段作候選，再由 target 驗證；新假設：code／tool 的重複片段可省掉 DSpark／MTP 的 draft I/O。 | 先用固定、來源清楚的小語料；只查已知 prefix，禁止把測試未來答案放進索引。先單一路徑 greedy、2／4-token 草稿；量 accepted tokens、target expert union、查找與修復成本。即使免 draft model，verifier 若仍比逐字生成貴就停。Qwen recurrent state 不能直接套 tree attention。 |
| H7 Prefill：非前綴文件 cache 重用 | [P9] CacheBlend 重用獨立文件並重算少數 token。新假設：同一文件在不同位置出現，仍可能省 Prefill；這與已實作的相同 prefix reuse 不同。 | 先列出每層完整狀態如何依賴前文，特別是 Qwen DeltaNet／convolution／N-gram 與 DeepSeek compressed state。若只能拼 K/V 卻不能處理其他狀態，停止直接移植；近似重算須獨立品質評估。 |
| H8 Prefill＋Decode：低於目前 FP4 的混合表示 | [P3] 支持對較不重要的 miss 使用較低精度；[P10] 顯示 expert 預算也能用於 speculative verifier，但會交換品質。舊構想尚未等於本機 sub-FP4 成功。 | 先比對 installed FP4 對照與單層 2／3-bit 候選的誤差、實際 bytes、解碼 kernel 成本。必須加上 scales／metadata，不能拿 FP16 當主要對照。若沒更小、更快，或品質失敗，停止。 |
| H9 Prefill＋Decode：expert 共用部分權重 | [P11] shared base＋delta；[P12] 進一步分群，避免所有 experts 共用單一 base 的限制。舊 PLAN 提過抽象方向，新一輪有具體方法和反例。假設少量常駐 base 加小 delta 可減少 SSD 負擔。 | 先各取早／中／晚層，量目前量化權重的殘差與共現；用不同資料驗證。計入 base 常駐排擠 slots、非線性 gate、額外運算；若只有 BF16 空間變小、相對 FP4 不省，停止。 |
| H10 新模型：讓路由更早可知 | [P13] Pre-gated MoE 共同設計路由與執行；[P12] 提供分群共享表示。假設可從小模型驗證 SSD 成本導向的路由／權重設計；屬新模型，不能沿用原 checkpoint 等價名稱。 | 先做小規模 teacher／student 或 routing 實驗，明訂資料、計算與品質預算。要與同參數／同 active compute 對照；若必須顯著犧牲能力才有 locality，停止。 |
| H11 Prefill：按 expert 組織運算（9 月 6 日新增） | [P15] ScatterMoE、[P16] MegaBlocks 支持分組運算與搬移成本的取捨。Qwen 的 unsorted Prefill 與同專案 DeepSeek 已排序的路徑有差異。 | 先用現有 MLX 工具排序／還原，分開測排序提示與數值；候選若改變 tokens，停止精確採用。再量相反順序的五類／256-token pairs，不能將不同運算路徑的數值差異當作品質已通過。 |

另外保留兩個研究分支，但本輪不列為單人對話的優先候選：
依 [P5]，跨請求合併運算可以攤薄讀權重成本，適合離線批次；須先接受排隊延遲與修改目前序列化 generation 的範圍。
依 [P14]，CPU／GPU 工作分配值得建立本機成本曲線，但 Apple 統一記憶體沒有同樣的 PCIe 搬移收益；不能把其他平台的倍數套過來。

## 如何讓小實驗回答問題

以下數值是**新一輪擬議門檻**，不是論文結論，也不會回頭改寫舊實驗的判定。

| 階段 | 工作 | 完成／停止 |
| --- | --- | --- |
| R0 盤點 | 核對程式、分類舊結果、重算最近 API artifact、建立論文與假設對照。 | 本輪完成；證據 JSON 與本文件可供重查。 |
| R1 選研究範圍 | 使用者選 A／B／C，記錄允許的品質與資源取捨。 | 已選 B；可做精確候選和近似候選的小實驗，尚未同意品質退步或預設採用。 |
| R2 新基準 | 固定當前 commit／依賴／installed model、明確 approximation mode；補逐 token hash 和分段時間。 | 基準可重跑，時間和 cache accounting 能對上才進下一階段。 |
| R3 最小反證 | 每個候選先做單層／離線／小模型檢驗；比較「最好情況能省多少」與新增成本。 | 上限都不足 5% 的完整時間收益，停止該候選；不擴大 prototype。 |
| R4 單一候選 | 一次接入一個預設關閉候選，先驗證行為與有限 workload。 | 正確性不符先停止；符合才量完整 request，避免同時改兩個方向。 |
| R5 正式比較 | 至少五個 AB／BA pairs／主要 workload，保留獨立 holdout 題目與原始 runs。 | 主要時間指標至少改善 5%，報區間及每題結果；低於門檻或區間跨零視為未解決。 |
| R6 採用或停止 | 採用時同步 docs、測試與對照資料；停止時寫明適用條件和重啟所需新證據。 | 未採用候選維持關閉；新模型／新量化格式使用獨立身份與 cache namespace。 |

R2 的主要組合：code、繁中技術、數學、tool；repeated 只作回歸控制。
先 1K／4K 輸入 × 256 輸出，再為勝出候選測 16K／64、1K／1,024 和多輪分支。
16K 先補目前版本正確性基準；不同模型使用各自 tokenizer，保存原文與 token hash。
精確路線先與 installed model 的 `exact` 比；另保留一般 DeepSeek 預設模式的實際體驗對照。
kernel 的數學等價不保證浮點 bitwise 等價；若 greedy tokens 改變，不能以「同公式」放行。

每次量：client 首字等待／wall time、runtime TTFT、Decode、p50／p95／最大輸出停頓、
未快取／重用 tokens、logical expert bytes、暴露讀取等待、MLX memory、process RSS 與系統記憶體壓力。
read timer 與 GPU work 可能重疊，不能直接相加；logical bytes 不等於 physical SSD bytes。
第一次編譯、模型載入、warm graph、memory prefix hit、restart prefix hit 分開記錄。
有 observer 的 profiling wave 與無 observer 的正式 timing wave 分開。

R5 擬議保護條件：另一階段的速度與首字等待不回退超過 5%，每 workload 的 p95 不增加超過 10%，總記憶體在同一預算內。
這些一般門檻不取代既有較嚴格的專項合約；例如 expert-drop 擴充仍遵守原本 2% p95 限制，除非使用者另行決定。
對 approximate 候選，token 一致率只作診斷；還需要可執行 code、數學正確率、tool schema／參數正確率、繁中能力與長文依賴評估。
品質可容忍退步幅度要在看到候選結果前決定；不能沿用十題 smoke 作為通用能力證明。

可重用工具：`Scripts/benchmark_api.py`、`Scripts/benchmark_qwen_mtp.py`、
`Scripts/benchmark_approximate_expert_drop_formal.py`、`Scripts/benchmark_block_prompt_cache.py`、
`runtime/deepseek_v4_ssd/route_trace.py`。先核對各自 CLI，不把 API 的 text hash 當 token hash。
新 raw runs 放 `scratch/`；正式量測放 `docs/benchmarks/`，附 commit、環境、設定、cache 狀態、prompt／output token hash。
本輪重算資料屬研究盤點，留在 `research/`，沒有冒充新的正式 benchmark。

## 論文查核表

本輪查核日為 2026-09-05。全文查核指閱讀相關方法／限制段落，不代表已重現論文。
只讀摘要的方向需在 R3 前補讀方法與實驗設定。所有外部速度數字均不作 Whallm 預測值。

| 編號 | 論文與版本 | 支持範圍／證據限制 |
| --- | --- | --- |
| P1 | [FlashAttention-2](https://arxiv.org/abs/2307.08691v1)，2023，摘要 | 支持減少中間讀寫與改善工作分配；A100 attention，不是 Qwen QSA 的可直接替換 kernel。 |
| P2 | [Gated Delta Networks](https://arxiv.org/html/2412.06464v1)，2024，§3.2 | 支持分塊 gated delta rule；訓練演算法的硬體效率不證明目前 MLX Prefill 有缺失。 |
| P3 | [HOBBIT](https://arxiv.org/html/2411.01433v2)，2024，§3.2–3.4 | 支持混合精度 miss、預取與多因素 cache。它的模型／精度／硬體與本案不同，尤其本案 experts 已為 FP4。 |
| P4 | [LLM in a flash](https://arxiv.org/abs/2312.11514v3)，2024 版本，摘要 | 支持 flash 成本模型、重用與連續讀；神經元稀疏方法不能直接當成 MoE expert 稀疏。 |
| P5 | [FlexGen](https://arxiv.org/abs/2303.06865v2)，2023，摘要 | 支持有限資源的存放／存取共同配置。主要目標為能接受延遲的批次 throughput，不是單人聊天。 |
| P6 | [SGLang](https://arxiv.org/abs/2312.07104v2)，2024 版本，摘要 | 支持 prefix state 重用與結構化工作流程；本案已有 cache，收益須來自新增可重用範圍或較好的預算配置。 |
| P7 | [REST](https://arxiv.org/html/2311.08252v2)，2024 版本，§3.2、§4、Limitations | 無須訓練 drafter，使用檢索候選和 target 驗證；收益依語料／領域，原實驗是 7B／13B，未證明 offloaded MoE 或 Qwen recurrent state 的收益。 |
| P8 | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)，2022 首稿，摘要 | 支持 target 驗證／修正可保持分布的原理；本機 shape、state、sampling 實作須另驗，不能直接借用等價聲明。 |
| P9 | [CacheBlend](https://arxiv.org/html/2405.16444v3)，2024 系列，§3–4 | 非前綴 KV 融合和選擇性重算；是品質近似，並非逐 token exact，也未覆蓋本案混合 attention／recurrent state。 |
| P10 | [MoE-Spec](https://arxiv.org/abs/2602.16052v1)，2026，摘要 | 限制 verifier expert 數，交換品質與延遲。省略 target experts 會改變被驗證的分布，不能宣稱對原 target lossless。 |
| P11 | [D²-MoE](https://arxiv.org/html/2502.17298v1)，2025，方法與 inference 評估 | shared base＋低秩 delta；速度表是 Mixtral、batch 64 等條件，本案需比 native FP4、batch 1，完整計入常駐 base。 |
| P12 | [LorExperts／BTExperts](https://arxiv.org/html/2608.07814v1)，2026-08-07 預印本，方法與 §7 限制 | 分群 base＋delta，提醒單一共享基底在 expert 多時失效；攤薄運算取決於路由，校準也有成本。本案兩 checkpoint 均未驗證。 |
| P13 | [Pre-gated MoE](https://arxiv.org/abs/2308.12066)，2023 首稿，摘要 | 支持提前路由的模型／系統共同設計；需要模型改造，不能冒充固定 checkpoint 的純 runtime 調整。 |
| P14 | [Fiddler](https://arxiv.org/abs/2402.07033v3)，2025 版本，摘要 | CPU／GPU 配置須依工作量與資料移動成本選擇；Apple 統一記憶體必須重新建立成本對照。 |
| P15 | [ScatterMoE](https://arxiv.org/html/2403.08245v1)，2024，§2.1–2.2；9 月 6 日補查 | 分組、還原與避免輸入副本的融合方法；本案用 MLX／MXFP4，並非其 Triton kernel。 |
| P16 | [MegaBlocks](https://arxiv.org/abs/2211.15841)，2022 首稿，摘要；9 月 6 日補查 | 支持不丟 token 的分塊稀疏運算方向；訓練吞吐結果不是本機 batch-1 推論收益。 |

## 研究範圍決策紀錄

使用者已選 B。以下保留當時的三個選項，供查閱研究範圍；不是再次等待選擇。

| 選項 | 接著做什麼 | 好處 | 代價／取捨 |
| --- | --- | --- | --- |
| A 保持輸出一致 | 先 H1／H2 Prefill 診斷、H3 cache、H6 小型檢索草稿；以 installed exact 為對照。 | 結果較容易驗證，可保留既有模型能力。 | 可能受到目前模型結構與 verifier 成本限制；不能靠低位元或更激進 drop 換速度。 |
| B 同時研究精確與近似（建議） | 精確候選先做一個，另做 H9 共用權重或 H8 低位元的單層可行性；先比較速度／品質曲線，再決定可接受範圍。 | 保留近期改善，也能找較大的突破；不需要立即訓練新模型。 | 研究與評估工作較多；進入完整 approximate 模型前，還要決定品質容忍幅度。選 B 不代表直接採用。 |
| C 以新模型設計為主 | H10：先定小模型、資料、計算預算，再驗證提前路由或分群共享架構；既有 runtime 作成本參考。 | 能處理固定 checkpoint 難以改變的根本成本。 | 工期最長、成果不確定；得到的是新模型，需要重新評估能力。 |

交接狀態：R0／R1、R2 的 Qwen 1K／4K 第一批基準，以及 H3／H9 初步篩選完成。
H3 舊候選在 4,096 slots 的 misses 增加；H9 未對齊固定 base 的誤差不利，均未接入 runtime。
完整結果與下一步見 [B 第一輪](PREFILL_DECODE_B_ROUND1_2026-09-05.md)。
使用者於 2026-09-06 明確允許恢復 GPU 研究；H2 接續狀態見
[Gated Delta 計時](GATED_DELTA_PREFILL_2026-09-06.md)。
DeepSeek 保留於後續輪次；這是實驗順序，不縮減原研究範圍。
H8／H9 仍停在離線／單層研究；會改變浮點結果的 expert 開提示候選已進入完整模型品質初測。
使用者於 2026-09-06 指定「任務正確率不下降」；該候選在
[Q1A](QWEN_QUALITY_Q1_2026-09-06.md) 因可重現繁中回退停止，未採用。
當前交接順序以 [研究入口](README.md) 為準。
無提示 expert 排序的 N1 新題目／長生成與 N2 快取互讀已通過；
預設關閉的 [N3 runtime 整合](QWEN_NOHINT_INTEGRATION_2026-09-06.md) 也已通過。
使用者於 2026-09-06 選擇預設開啟，目前已採用為 Qwen 預設。
