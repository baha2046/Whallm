# 研究入口

2026-09-15 最新決定：[移除範圍與歷史保存](archive/SSD_DIRECTIONS_RETIRED_2026-09-15.md)。
第 3 項已接入三模型；下方較早的「待實作／保留原型」描述是當時狀態。

2026-09-15 新增 [完整路由感知快取研究](ROUTE_AWARE_CACHE_2026-09-15.md)。
跨淘汰歷史、長短期統計、逐層容量分配已接入工作目錄，
目前功能契約見 [路由感知快取](../docs/ROUTE_AWARE_CACHE.md)。

目前 runtime 行為以 [docs](../docs/README.md) 為準。
本目錄保存研究問題、實驗與尚未採用的方向。

## 2026-09-15：GPT 討論後的選項 2、3

使用者指定先研究 [Decode 提交與 Qwen packed cache 先選列](DECODE_SUBMISSION_PACKED_SELECT_2026-09-15.md)。
兩個方向已分開完成部件與 8 次完整請求測量：Qwen 16K packed 8-bit 先選列的
Decode 兩對提高 15.12–15.49%；4K 常駐合併提交平均僅 3.60%，本輪停止整合。
候選只存在獨立研究程序，尚未採用為 runtime 或 App 預設；完整條件與限制見報告。

## 2026-09-15：本機 SSD streaming 六項實驗

使用者已選擇本機 V4＋Qwen。依 [執行表與比較規則](SSD_STREAMING_ABLATION_2026-09-15.md)
依指定對話的六項實驗驗證統計、快取、讀取、排列與推測解碼聯集預讀。
[實測報告與原始資料](../docs/benchmarks/2026-09-15-ssd-streaming-ablation/README.md) 已收斂：
V4 direct Prefill 的 4K request 縮短 6.35%；其餘訊號、退步、輸出差異及 swapout 排除個別列出。
App 預設不變。GPU 等待 I/O 的逐層歸因仍未完成；不把前景等待時間當成 GPU idle。

[Prefill／Decode 技術整理與 GPT 6 Pro 討論](PREFILL_DECODE_GPT6_PRO_2026-09-15.md)
整理目前三模型的硬體、記憶體、kernel 路徑與舊實驗限制，並列出可比較的後續量測。
這是研究建議，不修改下方既有執行階段，也不代表已採用或測得加速。

## 2026-09-12：模組化 runtime 與自有 mlx-lm fork

[設計研究與分階段 TODO](MODULAR_RUNTIME_2026-09-12.md) 比較輕量封裝、模型支援套件、
細粒度組件框架；使用者已選定模型支援套件並完成第一輪改造，
目前結構見 [模型支援套件](../docs/MODEL_PACKAGES.md)。
自有 fork 已建立，本機相容來源通過 389 項 runtime 測試及 1 項 V4 小模型測試，
尚未提交/推送或切換 dependency；目前狀態見 [fork 紀錄](../docs/MLX_LM_FORK.md)。

## 2026-09-07：SSD Streaming 架構原型

最新 [功能實作結果](SSD_FEATURE_IMPLEMENTATION_RESULTS_2026-09-07.md)：runtime
增加可選 LRU，完整 request 縮短 3.17%，未升預設。保留次數的 warm B4 verifier
成本縮短 14.85%、logical reads 少 20.53%，全 logits／state exact、MLX peak
增加 252 MB；尚未接入聊天生成。另修正 state clone 的負零位元，367 tests 通過。

前一輪 [三方向第一輪 gate](SSD_THREE_DIRECTIONS_GATE_2026-09-07.md) 已完成；
[原始結果及 191 份 source／raw 索引](../docs/benchmarks/2026-09-07-ssd-three-directions/summary.json)：
sparse fill + 整層 QMM 在 8K／16K 的 request 時間縮短 2.86%／1.36%，
數值與 +1 GB 通過但速度未達 5%，未採用。多字合批保留頻率、同容量 LRU 的
CPU 重播少讀有潛力，尚非 runtime 加速／峰值驗收。360 + 3 tests 通過。
結論與限制已同步 [docs/RESEARCH.md](../docs/RESEARCH.md)。

前一輪 [8K／16K 長輸入實測](SSD_RETAINED_LONG_RESULTS_2026-09-07.md) 已完成：
兩長度 full logits/state exact；各八次有效交錯測量，8K request 1.00393x，
幾乎持平；16K 0.95716x、時間增加 4.48%，速度 gate 拒絕。
峰值增幅均低於 +1 GB；受換頁干擾的首輪與完整重測資料分開保留，
112 項證據稽核通過。production／App 預設未改。

前一輪 [長 Prefill／ready Decode 實作結果](SSD_RETAINED_READY_RESULTS_2026-09-07.md)：
每層保留 pages、跨 chunk 重用已完成。4K／8K 數值與 state exact；
4K request 1.0907x、TTFT 1.1059x、MLX／RSS peak 增幅不到 1 MB。
當輪 8K 只有正確性，16K 尚未驗證，現由上述實測更新。Decode ready groups／shared overlap 數值通過，
但三版測速在第 10 次原版對照出現 system swapout，效能 gate 停止，未採用。
前置 [設計研究](LONG_PREFILL_DECODE_PIPELINE_2026-09-07.md) 保存逐 chunk 重讀問題
及 route 推算；它不覆蓋本次實測結果。各版均未改 App 預設。

後續 [直接 resident 短 block](QWEN_RESIDENT_BLOCK_RESULTS_2026-09-07.md) 已完成：
消除權重重打包，數值/state 正確、+1 GB 內；兩字 0.9374x、四字 1.0530x。
四字只通過 verifier 初篩，logical reads 沒下降，未接入正式生成。

後續 [穩定性與短 block](SSD_STABILITY_AND_BLOCK_2026-09-07.md) 已完成：
Prefill 通過 23 對 lifecycle 情境並封存 source；新短 block verifier 的 logits/state
exact，但速度成本為 0.944x / 1.001x，沒有減少 logical reads，第一候選停止。
不要在沒有新設計下重跑相同 2 / 4-token 打包版本或直接加上 drafter。

[雙緩衝 expert Prefill 流水線](SSD_PREFILL_PIPELINE_2026-09-07.md)
已實作研究版、取消／buffer lifetime 測試與真實權重 component gate。
使用者將記憶體條件更新為峰值最多增加 1 GB；原始零增幅結果保留。
143 / 1024-token 的完整模型 observer 均通過 48 層 tensor 與生成 token 一致性。
速度、冷／暖啟動限制和目前 gate 狀態以該文件及其證據檔為準；尚未採用為 App 預設。

## 新一輪 Prefill／Decode 研究

要接續 2026-09-05 的速度研究，先讀
[專案盤點、成敗經驗與論文支持的研究計畫](PREFILL_DECODE_RESEARCH_2026-09-05.md)。
最近測速的重算資料在 [證據摘要](PREFILL_DECODE_EVIDENCE_2026-09-05.json)。

目前：使用者於 2026-09-05 選 B。R0／R1、R2 的 Qwen 第一批基準與 H3／H9 初步篩選完成。
H3 舊規則增加 misses；H9 未對齊固定 base 的誤差不利，兩個候選均未接入 runtime。
使用者於 2026-09-06 允許繼續使用 GPU；H2 Prefill 時間診斷完成。
4K code 的 recurrent kernel 約 0.55 秒，投入排序低於 5% 門檻，暫緩完整改寫。
完整 attention 與 expert 計時已完成。
[QSA 查詢批次](QWEN_ATTENTION_GRANULARITY_2026-09-06.md) 的 20 組配對改善 2.64%，未達門檻；
[expert 排序、不開提示](QWEN_GROUPED_EXPERTS_2026-09-06.md) 的 20 組配對改善 15.79%，
輸出／讀取量一致，通過當輪篩選；後續延伸驗證已完成，見下文。
開啟排序提示的路線有較大的單次速度訊號，但會改變文字。
使用者已決定「任務正確率不下降」；[固定答案初測](QWEN_TASK_ACCURACY_2026-09-06.md) 完成，
兩版各 10/12，各類均無觀察到退步，但不代表可採用。
[Q1A 擴充評估](QWEN_QUALITY_Q1_2026-09-06.md) 在第 27/32 題出現繁中回退，
反向重跑仍是原版對／候選錯。開提示版本未通過品質門檻，已停止擴測與 Q1B。
一題數學的原版截斷另列未定，不以其他題的表現抵銷回退。
既有無提示候選的 [N1 延伸驗證](QWEN_NOHINT_VALIDATION_2026-09-06.md) 已通過：
四類新題目、1K／16K 輸入、1,024-token 生成與兩輪反向配對，32 次輸出與讀取量一致。
首次回覆改善中位數為 7.04%／13.46%，其餘保護條件也通過。
[N2 prompt-cache 分支／重啟一致性](QWEN_NOHINT_CACHE_2026-09-06.md) 的 18 次回答與
122 個共用 state tensors 也通過。當時預設關閉的整合已通過
[N3 runtime 對照與快取互讀](QWEN_NOHINT_INTEGRATION_2026-09-06.md)：
八次效能與八次快取請求輸出一致，1K／16K 首次回覆改善 7.59%／13.30%，220 個相關測試通過。
使用者於 2026-09-06 決定預設開啟，已採用為 Qwen 預設；CLI 與 model catalog 保留關閉方式。
採用與回歸檢查見 [目前 Qwen 文件](../docs/QWEN.md) 和 [驗證紀錄](../docs/VALIDATION.md)。
使用者已允許持續使用 GPU，無須逐次詢問。
H2 的證據限制與重啟條件見
[Gated Delta 計時](GATED_DELTA_PREFILL_2026-09-06.md)。
步驟、數字、證據限制與執行紀錄見
[B 路線第一輪](PREFILL_DECODE_B_ROUND1_2026-09-05.md)。

## 既有研究

| 文件 | 用途與日期限制 |
| --- | --- |
| [Qwen runtime](QWEN_RUNTIME_OPTIMIZATION_2026-08-29.md) | Prefill、QSA、MTP 的既有結果與停止條件。 |
| [SSD streaming](SSD_STREAMING_BOTTLENECKS_2026-08-31.md) | cache、預取與壓縮實驗。文末 Phase 6B 狀態較舊，approximate 目前狀態改讀下列文件及 docs。 |
| [Approximate mode](APPROXIMATE_MODE_2026-09-01.md) | DeepSeek expert-drop 合約與評估。 |
| [舊 PLAN tracker](PLAN_EXECUTION_STATUS_2026-08-27.md) | 21 個方向在 8 月 27 日的狀態；format 4、approximate 尚未開始等敘述屬當時紀錄，不覆蓋目前程式。 |
| [外部主張查核](EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md) | 舊研究的外部事實與本機證據邊界。新論文查核見本輪研究計畫。 |
| [Archive](archive/README.md) | 歷史推論與已被取代的計畫，供查原因，不能當目前預設。 |

新一輪沒有改寫舊實驗的通過／停止判定。重啟時需記錄新假設、對照條件與新的採用門檻。
