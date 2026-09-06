# Qwen 無提示排序：預設關閉的整合驗證

N3 已完成並通過；使用者後續於 2026-09-06 決定預設開啟，已採用為 Qwen 預設。
本文件的「預設關閉」描述 N3 執行當時狀態，目前設定以 [docs/QWEN](../docs/QWEN.md) 為準。
以下範圍與門檻在執行前固定；原始版本保存在結果索引中。

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

## 結果與目前狀態

M5 Pro 上的八次效能執行及八次快取執行均通過。各長度兩輪相反順序的中位數：

| 輸入長度 | 首次回覆時間縮短 | Decode 速度變化 | 每 token p95 延遲變化 |
| --- | ---: | ---: | ---: |
| 1,024 tokens | 7.59% | +0.17% | −0.99% |
| 16,384 tokens | 13.30% | −0.14% | +0.27% |

MLX peak 與 RSS 變化均小於 0.01%，通過預定門檻。
每次完整 256-token 輸出與同對、N1 前 256 tokens 相同；讀取量與 QMM 次數相同。
原版沒有排序呼叫，開啟時 1K／16K 分別為 192／768 次。
16K 執行的 ANE evaluation 為 180 次且沒有 fallback；1K evaluation 為零。

整合版本成功讀取 N2 兩種版本保存的共用快取，關閉功能後也成功讀取整合版本寫入的分支。
八次快取回答均與 N2 未快取對照相同，重用至少 1,024 tokens，共用 checkpoint 內容未改變。
N1、N2、N3 累計 66 次模型請求完成；這不含 N2 兩次只建立共用快取的 warm-up。

保存 [完整結果](../docs/benchmarks/2026-09-06-qwen-nohint-n3-integration-m5-pro.json)、
[來源索引](../docs/benchmarks/2026-09-06-qwen-nohint-n3-index.json) 與
[最後程式與測試核對](../docs/benchmarks/2026-09-06-qwen-nohint-final-verification.json)。

整合後檢查發現舊 model catalog 會因缺少新欄位而被拒絕，已修正為可省略新 boolean 欄位，
省略時仍為 false；其他既有必填欄位與型別檢查維持原規則。
此修正發生在 N3 量測後，只改設定解析；量測使用的 model、expert 計算、CLI、生成、快取與 ANE 程式
仍與 N3 固定來源逐檔相同。另有 80 個模型管理／API 測試通過，累計 220 個相關測試。

數字限於本機、固定題目、指定設定；每個長度只有兩對，OS page cache 未清除。
N1 長生成全部到上限，驗證的是輸出與運算一致，並非完成任務的正確率成績。
本輪沒有證明所有硬體、輸入或未來 MLX 版本都同樣受益。

採用決策已完成：使用者選擇 Qwen 預設開啟。CLI／單模型 server 可用
`--no-qwen-grouped-experts` 關閉；model catalog／configure 可明確設為 false。
舊 App 省略此欄位時自動採用 Qwen 新預設；4,096 slots、MTP 關閉與 DeepSeek 行為不變。
[採用驗證](../docs/benchmarks/2026-09-06-qwen-grouped-experts-default-on.json) 與原始 N3 量測分開保存。
