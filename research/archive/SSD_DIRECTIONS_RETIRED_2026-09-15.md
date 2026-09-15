# 2026-09-15 移除的 SSD 實驗

依使用者對十個方向的決定，保留第 1／2 項路由感知快取，將第 3 項接入三模型。
第 8 項不實作；第 4、5、6、7、9、10 項不再列為待實作工作。

| 方向 | 本次移除內容 |
| --- | --- |
| 4 共啟用排列 | 排列學習、模型副本重排與合併讀取的研究入口 |
| 5 自適應讀取 | 正式 runtime 內的密度門檻、路由規劃與相應統計／設定；Qwen 研究補丁 |
| 6 預測式預讀 | frozen-router 預測基線工具；沒有已採用的自學預測器 |
| 7 推測解碼感知預讀 | hash 預讀、獨立 scratch、storage-aware adaptive block、V4 hybrid 專家聯集驗證，以及 Qwen 額外聯集預讀補丁 |
| 9 專用 GPU 格式 | 取消方向；沒有已接入的專用離線 tile／scale 格式可刪 |
| 10 Metal I/O | 獨立 Swift probe 及其 benchmark 入口；沒有 MLX runtime 整合可刪 |

一般下一層預讀、必要的模型 repack、基本 DSpark／MTP、逐 token 驗證與一般專家 pinning 保留。
舊 Server 設定中已關閉的移除欄位可讀取後忽略；若設定啟用已移除功能，會明確報錯。

## 歷史保存

- [移除前程式與第一批工具](ssd-directions-retired-2026-09-15.tar.gz)
- [相依的舊診斷工具](ssd-directions-retired-diagnostics-2026-09-15.tar.gz)
- [後續移除工具與測試](ssd-directions-retired-tools-2026-09-15.tar.gz)
- [原始實驗的正式記錄](../../docs/benchmarks/2026-09-15-ssd-streaming-ablation/README.md)

壓縮檔中的程式不會被正常 App、CLI、Server 或測試載入，也不代表目前支援的功能。
它們是移除時的保存副本；要重現特定歷史數字，應使用該次 benchmark 自己的原始碼快照與設定。
目前行為以 [docs](../../docs/README.md) 為準。
