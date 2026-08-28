# Whallm 文件

本目錄只放目前有效的文件。
Whallm 的舊名稱是 DeepSeekV4SSD。

最後核對日期是 2026-08-28。
Qwen 支援核對版本是目前工作樹。這些變更尚未建立 commit。
DeepSeek checkpoint revision 是
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`。
Qwen FP8 checkpoint revision 是
`bcd9f01ddc9cff2316eb84281bebcd5b058bddce`。
Qwen installed model revision 是
`753d0aa57059fad70a5f7e6cc249f25df56bbd34`。

## 文件入口

| 文件 | 內容 |
| --- | --- |
| [架構](ARCHITECTURE.md) | checkpoint 合約、installed model、runtime 資料路徑、cache 與目前限制。 |
| [API](API.md) | OpenAI 相容 endpoint、欄位、串流、驗證與錯誤。 |
| [驗證](VALIDATION.md) | 目前測試、完整 SHA-256、SSD 量測、端到端量測與歷史基準。 |
| [效能與瓶頸](PERFORMANCE.md) | 指標定義、瓶頸判讀、A/B 方法與 profiling 流程。 |
| [研究結論](RESEARCH.md) | 已採用、未採用、延後研究與 DSpark 決策。 |
| [Qwen 支援](QWEN.md) | Qwen checkpoint、installed model、runtime、API 和目前驗證邊界。 |

專案使用方式仍以根目錄的 [README](../README.md) 為入口。

## 目前狀態

| 里程碑 | 狀態 | 目前證據 |
| --- | --- | --- |
| M1：checkpoint 與 repack | 完成 | repack plan、續傳、repair、manifest 與完整 SHA-256 驗證。 |
| M2：正確 reference decode | 完成 | batch size 1、greedy decode、4K 測試與 MXFP4 單元測試。 |
| M3：SSD expert streaming | 完成 | 固定 slot、LFU、`preadv`、SSD microbenchmark 與 runtime 指標。 |
| M4：throughput 與 context | 完成 | chunked prefill、MXFP8 KV cache、8K/14K 歷史測試與 DSpark 決策。 |

M4 完成不代表 runtime 已驗證 1M context。
本專案只驗證文件列出的測試長度。

Qwen text model 第一版已完成。
官方 FP8 checkpoint 轉換、installed model SHA-256、對話、thinking、tool call、
greedy 4K、prompt cache 和舊安裝路徑的 packaged App 驗證都已通過。
目前 direct published installed model 下載路徑已通過 file 續傳單元測試。
目前尚未重新執行完整 125 GB 的 App direct download。

App 的 Model 頁面固定顯示 DeepSeek 與 Qwen。
使用者可以管理尚未安裝的模型。
Server 頁面只顯示 server 狀態、系統檢查和 server 設定。
啟動按鈕不依賴 Model 頁面的選擇。
server 可以使用空 model catalog 啟動。
Model 頁面的標題列提供模型資料夾按鈕。
模型卡片會顯示模型資料夾路徑與可用空間。
每個未安裝模型的列提供下載按鈕。
每個模型的進階設定頁提供 Alias 欄位。
有效的 Alias 變更會自動儲存。
使用者可以在安裝模型前設定 Alias。
server 執行期間，App 會停用 Alias 和模型進階設定。
DeepSeek 下載固定包含 DSpark。模型列不提供排除 DSpark 的選項。
App 使用 pinned revision 的固定 installed model 大小執行下載前空間檢查。
App 啟動時不會為了取得下載大小連線到 Hugging Face。
每個模型列的「進階設定」動作使用 Tertiary icon button。
按鈕提供 40 pt 點擊區域、Tooltip 和 VoiceOver 名稱。
模型資料夾可用空間不足時，App 會停用下載按鈕。
App 會在模型列顯示停用原因、所需空間與可用空間。
游標停留在停用按鈕上時，App 也會顯示相同原因。
下載期間，模型列會依序顯示目前階段、百分比、完成容量、下載速度與剩餘時間。
下載模型時，使用者可以選擇另一個已安裝模型並啟動 Server。
Server 執行時，App 可以下載另一個尚未安裝的模型。
App 一次只執行一個模型下載。
server 啟動時只讀取 installed model 清單。
第一個 generation request 會載入指定模型。
server 一次只保留一個載入的模型。
Chat 頁面的模型選單只顯示 server 啟動時可用的 installed model。
使用者切換 Chat 模型時，App 會保留對話。
Metric 頁面顯示目前載入或載入中的 API model ID。
模型切換時，App 會清除舊模型的效能歷史。
「系統檢查」固定顯示在 Server 頁面。
該列使用硬體 SF Symbol 和狀態圖示顯示 Apple Silicon、記憶體和高速 SSD 檢查。

## 可信度規則

文件使用下列四種證據。

1. **目前程式碼**：描述目前介面、預設值和資料路徑。
2. **目前測試**：描述自動測試已覆蓋的行為。
3. **本機量測**：描述指定硬體、設定和 cache 狀態下的結果。
4. **研究假設**：描述尚未實作或尚未量測的方向。

外部文件只能證明外部 API、格式或論文結果。
外部文件不能證明本專案的速度。

若文件與程式碼衝突，請先以目前程式碼為準。
接著請更新文件和測試。

## 歷史研究

日期型研究草稿已移到 [`research/archive`](../research/archive/README.md)。
歷史研究保留原始推論與量測。
歷史研究不描述目前 runtime。

目前的研究查核位於
[`research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md`](../research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md)。

## 專案用語

本專案固定使用下列用語。

- **checkpoint**：固定 revision 的 Hugging Face 模型與 safetensors shards。
- **common tensor**：runtime 保留在記憶體的非 routed expert tensor。
- **routed expert**：一個模型專用 expert。DeepSeek 使用 `w1`、`w2`、`w3`。
  Qwen 使用 fused `gate_up` 和 `down`。
- **expert blob**：一個 routed expert 的標準封裝 bytes。
- **repack plan**：checkpoint byte range 到 installed model byte range 的完整對應。
- **installed model**：由 repack plan 產生並驗證的本機目錄。
- **API model ID**：一個 model kind 的固定且區分大小寫的 API 名稱。
- **Alias**：API model ID 的選用 request 名稱。Alias 區分大小寫。
- **manifest**：定義 installed model 與完整性資料的 JSON 檔案。
- **slot**：可放置一個 expert blob 的固定 Metal 可見記憶體區域。
- **main model**：43 個目標模型層。main model 不包含 DSpark。
- **DSpark**：儲存在 `mtp.*` 的可選 speculative decoding 模組。
- **N-gram store**：Qwen 使用的 read-only `ngram.bin` row store。
- **model kind**：manifest 用來選擇 DeepSeek 或 Qwen runtime 的欄位。
