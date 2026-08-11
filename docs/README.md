# DeepSeekV4SSD 文件

本目錄只放目前有效的文件。

最後核對日期是 2026-08-11。
核對版本是 commit `57e440e48b79fb0399beae7e9965481f74b35b25`。
checkpoint revision 是
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`。

## 文件入口

| 文件 | 內容 |
| --- | --- |
| [架構](ARCHITECTURE.md) | checkpoint 合約、installed model、runtime 資料路徑、cache 與目前限制。 |
| [API](API.md) | OpenAI 相容 endpoint、欄位、串流、驗證與錯誤。 |
| [驗證](VALIDATION.md) | 目前測試、完整 SHA-256、SSD 量測、端到端量測與歷史基準。 |
| [效能與瓶頸](PERFORMANCE.md) | 指標定義、瓶頸判讀、A/B 方法與 profiling 流程。 |
| [研究結論](RESEARCH.md) | 已採用、未採用、延後研究與 DSpark 決策。 |

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
- **routed expert**：一組 `w1`、`w2`、`w3` 與 scales。
- **expert blob**：一個 routed expert 的標準封裝 bytes。
- **repack plan**：checkpoint byte range 到 installed model byte range 的完整對應。
- **installed model**：由 repack plan 產生並驗證的本機目錄。
- **manifest**：定義 installed model 與完整性資料的 JSON 檔案。
- **slot**：可放置一個 expert blob 的固定 Metal 可見記憶體區域。
- **main model**：43 個目標模型層。main model 不包含 DSpark。
- **DSpark**：儲存在 `mtp.*` 的可選 speculative decoding 模組。
