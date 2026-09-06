# Qwen 無提示排序：預設關閉的整合驗證

此文件先固定 N3；只有 N1 與 N2 都通過後才套用整合草稿、執行以下測試。
目前不代表已完成整合或已更改預設。

## 範圍

把已測的排序／還原接入 Qwen `StreamingExperts`，保留原本兩次 MXFP4 QMM、
`sorted_indices=False` 和至少 64 assignments 的條件。
新增預設關閉的 `RuntimeConfig.qwen_grouped_experts` 與 CLI 開／關參數。
只在 Qwen、MTP 關閉且使用 batched Prefill 時作用；一般逐字生成維持原路徑。
DeepSeek 的行為、slot 預設與 Qwen 的 4,096 slots／MTP 關閉預設不變。

先測開／關時每個 expert 的結果完全相同，涵蓋短輸入、64 assignments 邊界、
128／1,024-token chunk 和一般 individual-expert Decode。
再跑相關 Qwen、runtime、CLI 與 prompt-cache 測試。測試失敗先修正，不接模型測速。

## N3 模型驗證

整合程式碼固定後，使用 N1 已固定的程式碼 1K／16K prompts。
每個長度兩輪相反順序、每次 256-token 生成，共八個 fresh-process requests。
這是整合回歸檢查；四類新題目與長生成的廣度由 N1 提供，不把這八次當成全新能力測驗。
設定沿用 N1：greedy、thinking／MTP 關閉、4,096 slots、48 GiB、ANE 比例 0.25，
prompt cache 停用、OS page cache 不清除。

CLI 直接開／關 runtime 功能，不替換 expert 計算函式。
只在研究 wrapper 計數排序 helper 的實際呼叫，並在關閉 model 前記錄 ANE 實際狀態。
每個控制須為零次排序，候選 1K／16K 須為 192／768 次。
完整輸出須與同對、以及 N1 對應輸出的前 256 tokens 相同；logical expert bytes／QMM 次數也須相同。

逐長度取兩對相對變化中位數：首次回覆至少改善 5%，Decode 不得退步超過 5%，
p95 不得增加超過 10%，MLX peak／process RSS 不得增加超過 5%。
任何輸出或讀取合約失敗立即停止；速度門檻失敗先解釋原因，不直接採用預設。

接著由實際整合版本分別讀取 N2 原版與研究候選保存的共用快取，各執行 B、A。
再讓關閉功能的原版路徑於新 process 讀取上述整合版本寫入的分支快取，各執行 B、A。
四個 processes、八個快取 requests，每次最多 128 tokens。
輸出需與 N2 未快取對照相同，實際 reuse、format 5 與 checkpoint 內容不變條件沿用 N2。

所有檢查通過後，將來源、命令、輸入、版本、快取狀態與輸出 hashes 保存至 `docs/benchmarks/`，
更新目前 runtime 文件，再提供是否改成預設的決策。此輪不打包、不發布。
