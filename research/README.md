# 研究入口

目前 runtime 行為以 [docs](../docs/README.md) 為準。
本目錄保存研究問題、實驗與尚未採用的方向。

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
