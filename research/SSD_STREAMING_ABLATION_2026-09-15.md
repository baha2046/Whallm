# 本機 SSD streaming 六項實驗

使用者於 2026-09-15 選擇本機 V4＋Qwen，並明確指定原對話
「如果由我來排 Whallm 下一輪實驗」的六項，逐項實作並量測。
起點 `6cdbdce2fba0085c035c982be1d18bfc7387f3bf`；M5 Pro、64 GiB、MLX 0.32.2。
既有未提交的 GPT 6 Pro 討論文件保留。這份文件記錄本輪執行，不改寫舊實驗結論。

本輪已收斂；目前結果以 [六項實測報告](../docs/benchmarks/2026-09-15-ssd-streaming-ablation/README.md)
為準。研究程式可重現各候選，App 預設不變。

## 比較規則

- 同模型、同輸入、greedy、固定輸出上限；V4 1152 slots、Qwen 3072 slots。
- 每次獨立程序，prompt cache 關閉，expert slots 初始為空。
- OS page cache 不清空；反向配對並記錄 run order，不能稱為冷 SSD benchmark。
- 先收集獨立 route observer，再做無 observer 的時間比較。
- 初篩使用 code 1K／64 output tokens；通過者擴至另一題目及 4K／128。
- 每項獨立和原版比較；輸出 token hash 不同即停止速度採用，保留失敗資料。
- 速度初篩至少 AB／BA 兩對；記錄 TTFT、request、decode tok/s、logical bytes、peak MLX。
- 正向訊號需換題目、拉長輸出；後續 `--prime` 在每次計時前執行相同的原版請求。
  Primer 不列入速度統計，也不把 OS page cache 宣稱為完全相同或已清空。
- 改善門檻 5%，記憶體最多增加 1 GB、不得出現新增 swapout。初篩不是普遍速度保證。
- 減少 logical bytes、component 速度、模擬 route replay 都分開報告，不能當成生成加速。
- 原始資料與暫存模型放 `scratch/ssd-ablation-2026-09-15/`；驗收結果才進 docs/benchmarks。
- 改動先保留在 research runner，通過擴測前不改 App 預設或 installed-model 格式。

## 執行表

| # | 建議 | 本輪可驗證的改動 | 下一步／狀態 |
| --- | --- | --- | --- |
| 1 | 每層統計 | route misses/token、logical bytes、命中率、讀取耗時、前景等待權重時間 | V4 43 層、Qwen 48 層完成；每層 bytes 加總符合 request counters；GPU idle 尚未量測 |
| 2 | 動態＋路由熱門度快取 | 移除每層最低保留量；LRU 對照 LFU／16-token 衰減；再加入每層實測 miss cost 加權 | V4 LFU 4K Decode +5.24%，兩輪 +1.75–8.74%；其他版本沒有穩定改善，停止採用 |
| 3 | Prefill／Decode 分開讀取 | Prefill 使用獨立 bypass descriptor，Decode 保留 cached | V4 direct 4K TTFT −14.65%、request −6.35%；Qwen staging request −3.12%；contiguous 一輪 swapout 排除，未採用 |
| 4 | 自適應部分／整層 | 保持原 QMM，路由確定後按聯集密度選讀取 | V4 1K request +21.41%；Qwen 4K +6.34%；讀取量減少卻更慢，不採用 |
| 5 | 共啟用磁碟排列 | 前半段 route 訓練排列，後半段檢查連續讀取與額外 bytes | warm／bypass 真權重 I/O 已完成；尚無超過原 ID 順序合併讀取的 5% 訊號，停止整模型改排 |
| 6 | 推測解碼 expert 聯集 | V4 DSpark sequential／hybrid／hash；Qwen MTP sequential／union／prefetch | Qwen 全請求更慢；V4 與預設 Prefill 的輸出不一致。改比相同輸出的 DSpark 版本，聯集及預讀仍未加速 |

## 停止與交接

每項保留實作入口、命令、資料路徑、失敗原因與數字。某個候選失敗只代表此實作／條件，
不否定整個方向。功能或能力不支援時寫明未量測，不能填 0% 或沿用歷史百分比。
第 1 項 GPU 等待 I/O 尚未完成歸因，保留 null；其餘本輪候選依上表結案。
這是有界初篩，不代表已測完所有 workload 或已通過 production 採用。

## 實作細節與本輪限制

目前 cache 已是跨層共用；`dynamic` 測試只是拿掉每層最低保留量。
`cost_hot` 才進一步以「衰減的使用次數 × 過去觀察到的每層讀取耗時」分配共用 slots。
讀取耗時以全層平均正規化並限制在 0.25–4 倍；每 16 個 Decode token 左右一起更新價格與 heap，
避免舊排序失效。這是可測候選，不代表該分數已能準確預測快取價值。
本輪熱門度跟隨常駐 `_Entry`，淘汰後不保留。跨淘汰的全專家長短期路由統計、
依逐層快取效益調整配額尚未實作；本輪淘汰策略的結果不能代表完整路由感知快取。

後續更新：使用者另要求完整功能，已另外實作
[路由感知快取](../docs/ROUTE_AWARE_CACHE.md)。以上保留六項實驗當時的範圍。

第 5 項使用獨立檔案，不覆寫 installed model。bypass 的 24 個 method/layer/wave 起點
檔案 resident bytes 均為 0；仍只是 I/O component。Qwen 的一個第一次 individual-read 樣本明顯較慢，
因此不能把 learned 相對該值的約 43% 差距當成改排加速。
真正隔離「排列」的對照是 identity-coalesced，兩模型均未出現可延伸的改善。

第 6 項先用 1K／32 tokens 檢查完整生成：逐 token 驗證、既有聯集驗證、聯集預讀分開比較。
V4 使用 948 main slots＋96 DSpark slots，hash-prefetch 額外保留 108 scratch slots；
Qwen 使用 2980 main slots＋32 MTP slots，預讀額外保留 60 scratch slots。
總 expert slot 預算不超過原版，另外記錄實際 peak。每個模型的三個推測版本使用相同 main cache 大小。
另外測 V4 1152 main＋768 DSpark slots 的記憶體取捨，仍未改善。
V4 普通生成關閉整層 Prefill、以及 DSpark 門檻 1 且實際提出 0 candidates，均重現
DSpark 的完整輸出 hash；差異和既有 Prefill／主模型路徑有關，不歸因於本輪聯集預讀。

`F_NOCACHE`／`F_RDAHEAD` 是針對開啟的 file descriptor 設定；本輪分開開啟 Prefill／Decode descriptor，
不使用改變整個檔案行為的 `F_GLOBAL_NOCACHE`。
API 語意依 [Apple XNU header](https://github.com/apple/darwin-xnu/blob/main/bsd/sys/fcntl.h)。
