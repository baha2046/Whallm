# Qwen grouped Decode 整合與採用 gate

2026-09-06。使用者要求持續推進 Decode，遇到真正的決策才停下。
本文件接續 [第一輪 Decode 研究](QWEN_DECODE_RESEARCH_2026-09-06.md)。
Runtime 已新增 default-off 整合；尚未改變 APP 預設，未宣稱已通過全部採用 gate。

**使用者已選 B：維持原門檻，研究跨分區 kernel；未採用。** 大區塊的 cold memory 不合格；小區塊速度
不足；依常駐量規劃的版本通過短 code 與 cold／lifecycle，但長工具 Decode +4.58%，
未達預定 5% gate，矩陣已停止。完整 artifacts 見
[455 份證據索引](../docs/benchmarks/2026-09-06-qwen-decode-integration-m2-max/summary.json)。
新工作接續於 [跨區塊 kernel 研究](QWEN_DECODE_KERNEL_2026-09-06.md)，舊失敗結果保持不變。
該新 kernel 已通過 M2 Max 的 14 個正式波次與 cold／lifecycle gate，仍為
research-only；它不改變本文件分頁 QMM 的失敗判定，也尚未接到原 CLI flag。

## 假設、來源與邊界

[DeepSpeed Inference §III](https://arxiv.org/pdf/2207.00032) 支持檢查小 batch 的
kernel dispatch 和資料搬移；[MoE-Infinity](https://arxiv.org/abs/2401.14361)
支持依實際 expert 活動量評估 cache。論文不證明 MLX 的 alias 正確性、Qwen
數值一致性或 M2 Max 速度。這些全部使用本機 component／完整模型 gate 驗證。
本輪固定 checkpoint、MXFP4 bytes、top-10、LFU、greedy 與 ANE requested=true／0.25。

## 本輪實作與失敗經驗

- 把研究 monkeypatch 移到 default-off `RuntimeConfig.qwen_grouped_decode`，拒絕
  非 Qwen／MTP，CLI 提供開關，APP 固定關閉，舊 catalog 缺少新欄位時相容為 false。
- 每個 request 記錄剩餘 Prefill tokens；僅以單 token 呼叫次數區分的原型不適合
  多輪請求，也不能涵蓋中間恰為一個 token 的 Prefill chunk。生成器取消後同步其
  generation stream，再允許下一次 request 使用 slots。
- 4,096-slot 單 arena 實測失敗：MLX `as_strided` 內部 flatten 為 2,673,868,800
  uint32 words，超過 32-bit shape 上限。失敗 source 保留；改為最多 2,048 slots
  的分區 arena。跨區塊各自 grouped QMM，最後恢復原 expert 次序再由既有程式加權。
- Candidate prompt-cache contract 另加標記隔離；一般模式的既有 contract 不變。

## 執行前停止條件

先檢查實際 4,096 slots 的 sparse indices、跨區塊 overwrite／restore 與 tensor-exact。
接著檢查 request lifecycle：同程序重複、分支 prompt 取消、取消後重用、單輸出 request、
全新程序 persistent cache restart。每個 matched case 的 prompt／output token IDs、
cache reuse 和 logical expert bytes 均須相同，ANE 必須 active 且無 fallback。

效能以 fresh-process ABBA 與 BAAB，persistent cache 關閉、OS cache 不清除，
分別評估短 code 與五類 4K／256 workloads（中文使用補充 prompt，避免九個 token 的
提早 EOS）。每輪 Decode median 改善至少 5%；TTFT 不超過 +5%、p95 +10%、
MLX peak +15%、logical expert bytes +5%。任何 exact mismatch 立即停；未過速度門檻
也不得以平均其他 workload 掩蓋。M2 Max 與歷史 M5 Pro 數值不混合。

## 目前結果

4,096-slot component：跨頁 positions 包含 0、2,047、3,290、4,095。Tensor-exact、
finite、overwrite visible、restore exact 全通過。一次 component median 0.435 ms，
不是正式端到端速度結果。原單 arena failure 另存，不覆寫失敗紀錄。

短 code／143 input／64 outputs／4,096 slots 第一輪 ABBA：四份 output IDs 相同，
Decode +7.84%、request -2.67%、TTFT -0.03%、p95 -15.77%、MLX peak -0.20%、
logical expert bytes 相同。這是第一輪結果，仍需反向順序與其他 workload gate。

短 code 的 4,096-slot BAAB 另取得 Decode +10.04%、request -4.11%、TTFT -1.12%、
p95 -15.45%，output IDs／expert bytes 相同。1,152-slot 整合 ABBA 的 Decode +10.30%、
request -1.39%、TTFT +3.45%、p95 -7.14%，通過原 gate；速度不能只引用最好一輪。

Lifecycle v1 雖然配對一致，分支 prompt 卻在一個 output 就 EOS，沒有實際測到取消。
保留該 diagnostic，另以 v2 明確要求輸出五個 token 才取消。v2 共 16 次 request：
cold、memory repeat、branch EOS、Decode cancel、after cancel、one output、after one
output 各一對，加上 fresh-process restart 一對，全部 token／reuse／bytes 一致。
每條路徑的 cold／warm／cancel prefix／單輸出／restart 也相互一致。Memory repeat
及 restart 均重用 142 tokens；candidate active memory 從約 20.840 到 20.849 GB，
control 從 20.881 到 20.891 GB，未見每次 request 累增 arena。

Python 全套原 303 項通過，補上 cache-contract 與分區預算邊界後，最新為 305 項通過。
Swift 使用完整 `/Applications/Xcode.app` 後成功編譯。全套執行遇到既有
`AppKeychain.readAPIKey`／`SecItemCopyMatching` 等待，sample 已保留；排除四項
會存取 Keychain 的案例後，64 項執行、61 通過、三項因缺少完整 DeepSeek installed
model fixture 而 skipped、零失敗。不能寫成完整 Swift suite 全通過。

## 極短冷請求揭露的停止條件

工具長文兩輪全部 exact：Decode +7.80%／+9.59%，request -2.99%／-2.79%。
補充中文第一輪四次 4,155／256 也全部 exact：Decode +9.84%、request -3.52%。
但這些請求會填滿 expert cache，未覆蓋冷啟動只產生一個 token 的 allocation 行為。

在中文第一輪結束的 wave 邊界暫停 matrix，補跑 fresh-process `Hi`，實際 1 input／
1 output、4,096 slots，兩個 outputs 相同、expert bytes 相同、resident slots 同為 693。
原 2,048-slot arena 的 MLX peak 為 15,372,815,368 bytes，control 11,839,322,356 bytes，
**+29.85%，超過原 15% gate**。因此該設計停止採用；未繼續把後面正面平均值補齊。
數學／code／repeated 的長重複矩陣，以及中文反向波，仍未完成，不能報成完整通過。

[PagedAttention，SOSP 2023](https://arxiv.org/abs/2309.06180) 說明動態區塊配置可降低
KV cache 的記憶體浪費。本輪將「減少未使用的預先配置」作為 **expert arena 的研究
推論**，不是引用該論文證明 Qwen QMM 效能。改成 512-slot 區塊後，真實 4,096-slot
component 的 tensor／overwrite／restore 全 exact；極短冷請求 peak 為 12,698,062,232
bytes，相對相同 control **+7.25%**，bytes／output 相同，通過 cold memory gate。
該單次短請求時間 +7.85%（約數十毫秒），不能宣稱 cold latency 改善。

固定 512-slot 區塊的短 code／4,096 slots ABBA 全部 exact，但 Decode 只 +3.28%，
request +0.59%、TTFT +2.85%、p95 +0.75%，低於 5% speed gate，停止採用。
Grouped QMM 次數由 2,048-slot 版本的 10,522 增至 23,566（control 192 次均來自
Prefill），支持較碎分區增加 dispatch 工作量的源碼解釋，並非量到純 GPU compute。

## 依常駐量規劃的後續候選

使用同一個 PagedAttention 啟發的配置方向：第一區塊最多 1,024 slots，後續以
canonical common bytes 的 90% 作保守規劃下限，加上先前已配置的 expert bytes，
每區塊取其 14% 預算，再向上對齊 32 slots，且仍不超過 1,024 slots。
這些數字是本專案的工程護欄，不是論文宣稱的最佳參數。
本 checkpoint／4,096 slots 得到 `[1024, 640, 736, 832, 864]`。

獨立載入 observer 實測：common manifest 9,895,397,146 bytes，規劃下限
8,905,857,431 bytes；control／candidate 模型剛載入的 active memory 均為
9,896,625,952 bytes，下限成立。它不是其他 checkpoint 的通用保證。
4,096-slot component 的 tensor／overwrite／restore 仍 exact，極短 cold request
peak +7.26%，output／bytes 相同。2／3／4／8／16 output 的獨立 cold pairs 也全通過，
最高 peak 增幅 +11.50%。16 次 lifecycle requests／restart 全 exact，reuse／bytes 一致。
短 code／4,096 slots ABBA／BAAB 的 Decode +6.68%／+5.33%，request -3.35%／-2.41%，
TTFT -1.64%／-0.85%，p95 -10.47%／-8.16%，都通過。

CLI 1,152-slot code ABBA 的 Decode +9.14%，但 request 只 -0.23%，TTFT +4.95%，
接近 +5% 上限；這一配置只有一個 wave，不構成完整採用證據。
4,539-input／256-output 工具 ABBA 的四次輸出完全相同；Decode +4.58%、request
-1.93%、TTFT -0.60%、p95 -10.46%、MLX peak -0.20%、expert bytes 相同。
這不是數值失敗，但 **低於預定 5% speed gate**，因此停止後續 matrix。
新版本的工具 BAAB、其餘四類長 workloads、1,152-slot 反向 wave 和 M5 Pro 均未完成。
未把已停止設計的其他 workload 數字搬來當作新版本結果。

必須同時符合原門檻，不能為了採用而放寬 +15% memory 或 +5% Decode 改善要求。
預設仍關閉。

## 需要使用者決定的取捨

下列是下一階段提案，不是已授權的新 gate，也不是宣稱已可開預設。
原 gate 與失敗記錄維持不變。若選 A 或 C，新標準必須另行記錄並用新驗證確認。

| 選項 | 優點 | 代價與待驗證事項 |
| --- | --- | --- |
| A：接受部分情境 Decode 未滿 5%，改以整段請求改善與尾延遲為主（建議） | 保留目前記憶體護欄與 exact outputs；目前已有 request 約 1.9–3.4% 的指定情境改善 | 收益較小；需另訂標準、使用新的 workloads 補驗，不能將舊未過 gate 改寫成通過 |
| B：維持原門檻，轉研究跨分區 kernel／dispatch | 保留 RAM 與品質要求，嘗試消除多區塊額外呼叫 | 需要更大實作，收益與 numerical parity 未知；DeepSpeed Inference 只支持研究機制，仍需 tensor-exact gate |
| C：接受更多預留 RAM，續驗 2,048-slot 大區塊 | 已量到 App 容量下部分 workloads Decode 約 7.8–10.0% 改善 | Hi／1 output 已多 3.53 GB MLX peak（+29.85%），不是其他 workload 的上限；需補 cold 長度、其餘長 workload 及硬體驗證 |

建議 A 是因為使用者關心實際使用的整段時間；它不表示目前證據已足以採用。
本輪未量測能耗，也未做新的模型能力評分。
