# Qwen expert 計算：先按 expert 排序

目前結論：無排序提示的候選完成五類 4K／256-token、兩輪相反順序配對，
主要四類 TTFT 改善中位數 **15.79%**，全部輸出與 logical expert bytes 一致，通過本輪篩選。
這是 M5 Pro 上既有合成輸入的結果。
後續 [N1 新題目／長生成](QWEN_NOHINT_VALIDATION_2026-09-06.md) 的 32 次執行通過，
1K／16K 首次回覆改善 7.04%／13.46%；[N2 快取檢查](QWEN_NOHINT_CACHE_2026-09-06.md) 也通過。
預設關閉的 runtime／CLI 選項已通過 [N3 整合對照](QWEN_NOHINT_INTEGRATION_2026-09-06.md)：
1K／16K 首次回覆改善 7.59%／13.30%，輸出與快取互讀一致。使用者後續選擇預設開啟，目前已採用。
開啟排序提示的路線有較大的單次速度訊號，但未通過「任務正確率不下降」門檻。
[12 題初測](QWEN_TASK_ACCURACY_2026-09-06.md) 曾無觀察到退步；
[Q1A 擴充評估](QWEN_QUALITY_Q1_2026-09-06.md) 在繁中候補名單題出現原版對／候選錯，
反向重跑仍重現，已停止此版本的擴測與採用。這不改寫無提示候選的既有結果。

## 假設與目前證據

4K code 的同步 observer 量到 MoE Prefill 約 19.8 秒，是本輪已量組件中最大的一項。
它不含外部 full-layer 讀取，不能把時間直接叫做 SSD 等待。
原版 Qwen 的 batched Prefill 直接將未排序的 assignments 傳入 `gather_qmm`。
同專案的 DeepSeek 已使用 MLX-LM 的 `_gather_sort`／`_scatter_unsort` 與 `sorted_indices`。
因此先研究：將分配到相同 expert 的輸入放在一起，能否改善 Qwen Prefill 的計算效率。

論文依據是 [ScatterMoE §2.1–2.2](https://arxiv.org/html/2403.08245v1)
與 [MegaBlocks](https://arxiv.org/abs/2211.15841)：按 expert 組織運算與資料搬移的取捨。
本候選只重用現有 MLX 排序／還原工具，會建立額外輸入副本；不是這兩篇論文 kernel 的重現。
論文不證明 native MXFP4、Apple GPU、本模型的收益。

## 事前規則

先獨立測 code 4K／32，使用與 QSA 實驗相同的正常設定與輸入。
只改 batched Prefill、assignments 至少 64 時的 expert 計算順序，還原後才按原 scores 加總。
路由、top-k、expert bytes、gate/up 順序、量化、Decode expert 路線都維持原合約。
必須完整 output token hash 相同，否則停止精確路線，不以「公式相同」放行。
預期 192 次 Prefill expert 呼叫經過候選，記錄實際計數核對。
通過後，與控制做相反順序的重複測試，再進入五類／256-token 評估；
速度／記憶體門檻沿用本輪 QSA 計畫，兩個候選先分開，不混在同一個實驗。

## 初次篩選與後續診斷

第一次 code 4K／32 已完成，192 次 Prefill 呼叫經過候選。
TTFT 為 24.364 秒，但與控制從第 5 個 token 起分歧，因此**沒有通過精確輸出條件**。
這只是速度信號，不是可採用的加速結果；不同文字也尚不能判定能力變好或變差。
metrics 的 `approximation_mode=exact` 表示未啟用既有 expert-drop 模式，
不能代替研究候選的 token parity 檢查；更換運算路徑仍可能產生浮點差異。
初次候選腳本的完整原文與 hash 保存在 `scratch/qwen-grouped-experts-2026-09-06/first-candidate-source.py`。

接著先固定 layer 0／23／47 的第一個 Prefill chunk：核對排序／還原的 expert IDs 與輸入列完全相等；
比較開／關 `sorted_indices` 兩個版本與原始輸出的相對誤差。
這個 observer 一律回傳原始輸出，不把候選誤差帶入後續模型。
若關閉提示即可消除誤差，再測其完整輸出及速度；若仍不同，先定位差異，不進入品質採用。

本輪查閱的 [MLX v0.32.0 原始碼](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/quantized.cpp)
在 `GatherQMM::eval_gpu` 對右側已排序、M=1、B≥16、B/E≥4 使用不同的 grouped 路徑；
目前 1,024×10 assignments／512 experts 的形狀符合此條件。
這支持把排序提示與運算路徑分開診斷，但尚未以 GPU trace 證明本機 wheel 的內部 dispatch。
上述上游版本檔案已保存 URL 對應的 raw 內容；安裝 wheel 的版本是 0.32.0。

QSA 成對測試已完成；以下接續 expert 三層診斷與新候選。

## 三層診斷與無提示候選

layer 0／23／47 的 expert 分配與輸入列經過排序、還原後均完全相等。
開啟排序提示時，三層的輸出元素一致率分別為 99.8382%／99.9845%／99.9871%，
相對 L2 誤差約 0.00914%／0.00332%／0.00214%。
關閉提示、仍保留排序時，三層均 100% 相同、最大絕對誤差為 0。
診斷始終回傳原始結果，其完整 32-token 輸出也與控制相同。
這支持差異與提示選擇的運算路徑有關；仍不是每個形狀都正確的普遍證明。

接著測無提示候選，code 4K／32 的單次 TTFT 為 32.642 秒，輸出完全相同。
進一步的 control／candidate／candidate／control 四次測試，TTFT 為
39.461／33.019／31.952／37.531 秒；兩對改善 16.33%／14.87%，四條輸出完全相同。
因此啟動原訂五類／256-token、兩輪相反順序、20 個獨立 process 的完整篩選。
其通過後仍需 held-out 與長輸入／cache contract 驗證，不能直接當成預設值採用。

無提示版本沒有將 candidate 換成更低精度權重，也沒有少算 expert；
仍須用完整 outputs 和 logical bytes 核對，不單憑這個設計意圖放行。
benchmark 共用 driver 的 `chunk=0` 只是 sorted-expert 候選識別值，實際 QSA query chunk 仍為 4；
`candidate_kind`、`variant` 與完整命令另行記錄，以免誤讀。

保存 [初次失敗輸出](../docs/benchmarks/2026-09-06-qwen-grouped-experts-initial-m5-pro.json)
與 [三層數值診斷／無提示單次結果](../docs/benchmarks/2026-09-06-qwen-grouped-experts-numerical-diagnosis-m5-pro.json)。

## 無提示候選的完整篩選結果

20 個 fresh-process requests、每個 256 tokens 全部完成。
10 對 outputs 與舊基準相同；10 對 logical expert bytes 和 gather_qmm 呼叫數也全部相同。
每次候選都有 192 次 batched Prefill expert 呼叫，排序提示均為關閉。

| 題目 | TTFT 改善 | Decode 變化 | Decode p95 變化 |
| --- | ---: | ---: | ---: |
| repeated 控制 | +11.47% | −0.13% | −0.24% |
| 程式碼 | +15.73% | −2.20% | +1.05% |
| 繁中技術 | +16.18% | +0.10% | −0.18% |
| 數學 | +15.08% | −0.20% | +0.53% |
| tool 格式 | +15.85% | −0.25% | +0.01% |

每列是同題兩對的相對變化中位數；主要四類 TTFT 中位數為 15.79%，超過 5% 門檻。
逐題 Decode 均未退步超過 5%，p95 均未增加超過 10%；MLX peak memory 近乎不變，
process RSS 各題中位變化介於 −0.20% 和 +0.20%，通過記憶體條件。
這是 Prefill 改善，不是 Decode 加速；程式碼題目的 Decode 小幅回退仍須在較長生成中另驗。
每題只有兩對、OS cache 未 purge，且題目曾用於選候選，因此不標為正式採用效能結論。

- [20 組完整量測](../docs/benchmarks/2026-09-06-qwen-grouped-experts-nohint-paired-runs-m5-pro.json)。
- [門檻計算](../docs/benchmarks/2026-09-06-qwen-grouped-experts-nohint-paired-gate-m5-pro.json)。
- [執行前規則快照](../docs/benchmarks/2026-09-06-qwen-grouped-experts-nohint-protocol.md)。
- [反向順序短測試](../docs/benchmarks/2026-09-06-qwen-grouped-experts-nohint-pilot-m5-pro.json)。

當輪留下的新題目、1K／16K 輸入、較長生成和 prompt-cache 分支／重啟檢查，
後續已由 N1／N2 完成；code 長生成沒有重現原先的小幅 Decode 回退。
本節的 4K 結果來自研究 process 替換，當時未改 runtime；目前的整合狀態以文首與 N3 為準。
完整 [第二輪證據索引](../docs/benchmarks/2026-09-06-prefill-round2-index.json) 保存 artifact hashes 與檢查結果。
收尾通過 9 個 CPU 單元測試、Python 編譯、文件連結與 diff 檢查；兩個方向共 40 組配對資料的
output hashes 和 logical bytes 均核對完成。沒有重跑完整 App／runtime 測試，也沒有常駐 benchmark process。

## 若必須改用任務品質判定

使用者於 2026-09-06 指定「任務正確率不下降」，未授權可容忍的下降幅度。
輸出文字不同不能換算成品質下降比例；上面的實作／數值診斷已完成。
本輪已依 [品質門檻](QWEN_TASK_ACCURACY_2026-09-06.md) 固定題目、評分規則與停止條件，
完成 Q0 初測，並在 Q1A 因可重現回退停止目前的開提示版本。

評估方法的論文依據可包括 [HumanEval](https://arxiv.org/abs/2107.03374) 的程式功能正確性、
[GSM8K](https://arxiv.org/abs/2110.14168) 的數學答案正確性，與
[C-Eval](https://arxiv.org/abs/2305.08322) 的中文知識／理解評估。
這些各有涵蓋範圍：C-Eval 不能單獨代表繁中對話品質；工具呼叫還需本專案 schema／參數測試。
它們是測量方法的來源，不是本候選已通過的證據。
對照與候選必須使用相同的 held-out 題目與生成條件，按能力類別分開報告，
不以某一類的提升抵銷另一類的退步，也不把少量題目零回退宣稱成普遍品質保證。

## 保留的新方向：只排序索引，減少輸入複製

目前 `_gather_sort` 會按 top-k 展開並複製輸入；Qwen top-10 在 1,024-token chunk 有 10,240 個 assignments。
可另研究保留原始輸入，用已排序的 `lhs_indices`／`rhs_indices` 存取，最後才還原 expert 輸出。
這對應 ScatterMoE §2.2 避免 grouping copy 的動機，但本案先用現有 MLX `gather_qmm`，不是其融合 kernel。
是否減少時間或記憶體尚未量測，也可能改變 dispatch，仍須完整數值與時間對照。
它與本輪已在跑的「排序並複製、關閉提示」候選分開，不能在同一批測試途中替換實作。
