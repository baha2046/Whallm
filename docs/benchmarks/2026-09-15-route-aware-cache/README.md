# 完整路由感知快取：V4／Qwen 實機對照

這一輪測試新增的 production `route` 策略，與先前只記錄常駐項目頻率的
[六項實驗](../2026-09-15-ssd-streaming-ablation/README.md) 分開。
功能與啟用方式見 [路由感知快取](../../ROUTE_AWARE_CACHE.md)。

## 最終結果

下表為兩個同輪相對變化的中位數。完整請求時間負值表示等待變短。

| 模型／輸入 | Decode tokens/s | 完整請求時間 | 首字時間 | Expert 讀取量 |
| --- | ---: | ---: | ---: | ---: |
| V4／1K 程式題 | +17.57% | -10.54% | -5.01% | +0.57% |
| V4／4K 繁中 | +6.69% | +0.18% | +4.69% | +2.23% |
| Qwen／1K 程式題 | +15.76% | -9.87% | -6.34% | +1.05% |
| Qwen／4K 繁中 | +2.34% | -0.26% | +0.51% | +0.64% |

**結論：短輸入有正向訊號；長輸入尚無穩定的整體收益。保留手動選項，不改原預設。**

- V4 1K Decode 兩輪為 +10.56%／+24.58%；Qwen 1K 為 +10.05%／+21.47%。
- V4 4K 為 +19.79%／−6.42%，波動大；Qwen 4K 為 +6.43%／−1.75%。
- V4 4K 數值上達到 Decode +5% 初篩門檻，但完整請求接近持平，不能稱為穩定收益。
- 四組都請求了更多 expert bytes，不能把生成加速說成少讀了相同比例的 SSD 資料。
- 同輪百分比先取中位數，與先取兩輪秒數中位數再相除可能略有差異；接近零的結果視為持平。

16 次計時生成均與同組 LRU 輸出 token hash 完全相同，沒有新增系統 swapout。
MLX 峰值差異小於 0.01 MB；程序 RSS 的最大增加約 6.83 MB，均通過 +1 GB 限制。
路由統計陣列為 V4 354,320 bytes／Qwen 788,736 bytes。這不包含 Python 候選 heap。

V4／Qwen 連續請求驗證各 4 組通過：首次、重複、換題，以及取消／釋放槽位後重試。
完整輸出均與 LRU 對照的前 32 tokens 一致；重複請求只重設 Prefill 統計，
長短期歷史持續累積，手動釋放權重槽位也不會清掉歷史。

每組 `*-layers.csv` 保存逐層目標／實際容量、歷史與成本，
各組 JSON 保存輸入／輸出 hash、原始測量、配置及每輪比較。
V4 最後一層原有的按層 Prefill 只建立 KV，略過 FFN；該層只有首次回傳前的
2 個實際路由位置。其他受測層為輸入長度＋1，包含生成器提前計算的位置。
證據檢查依原有執行路徑核對，不替沒有執行的路由補造資料。
所有數字是本機這四個有限負載的結果，不外推到其他硬體、Slots 或對話長度。

[彙總與檔案 SHA-256](summary.json)、[V4 生命週期](v4-lifecycle.json)、
[Qwen 生命週期](qwen-lifecycle.json)、[完整原始封存](raw-evidence.tar.gz)。

## 條件

- 2026-09-15，Apple M5 Pro、64 GiB；Python 3.14、MLX 0.32.2、mlx-lm 0.31.3。
- 基底 commit `6cdbdce2fba0085c035c982be1d18bfc7387f3bf` 加本次工作目錄修改。
  每組保留 Python source snapshot 和 SHA-256；原始封存另含 runtime／App diff。
- V4 43 層、256 專家、top-6、1152 slots；Qwen 48 層、512 專家、top-10、3072 slots。
  Installed model 的路徑和 manifest hash 記錄在各組 JSON。
- 工作負載為程式題 1024 輸入／128 輸出、繁中技術題 4096 輸入／128 輸出。
  Greedy、exact、4 個讀取工作、2 個預讀工作、MLX 限額 48 GiB；其他沿用 CLI 設定。
- 對照固定 LRU。同容量，兩輪順序 LRU→route、route→LRU。
  每次各用新程序、空 expert slots、關閉 prompt cache；計時前先跑相同 LRU 請求。
  沒有清除 macOS 檔案快取，因此不是冷 SSD 測試。
- 先檢查輸入／輸出 token hash、無新增系統 swapout、MLX／RSS 峰值差額不超過 +1 GB。
  再看 Decode 是否達到事前設定的 +5% 初篩門檻。兩輪不能代表統計顯著或所有負載。

## 指標解讀

速度以兩個同輪相對變化的中位數呈現。時間和讀取量負值較好；tokens/s 正值較好。
讀取量是程式請求的 expert bytes，包含 Prefill，並非 SSD 實體讀取量。
`expert_eviction_seconds` 只包含挑選被淘汰專家的時間；不能當成策略的全部 CPU 成本。
`metadata_array_bytes` 只計歷史陣列，不包含 Python heap、權重或整個程序。
本輪比較整套 route 策略與 LRU，沒有分離長短期統計、讀取成本和逐層配額各自的貢獻。
每次計時都從空歷史開始；連續請求另做功能驗證，不代表已測出長期學習的速度收益。

## 重現

```sh
.venv/bin/python research/ssd_streaming_ablation.py \
  --model qwen --output scratch/route-cache-rerun/qwen-code \
  --variants control,route --input-tokens 1024 --output-tokens 128 \
  --workload code --prime
```

V4 改 `--model v4`；長輸入改 `--input-tokens 4096 --workload zh_technical`。
每次使用新的輸出目錄；同一台機器一次只跑一個模型測試。

## 證據範圍

功能測試與 App 設定測試另存 `runtime-tests.log`／`swift-tests.log`。
生命週期檢查重用同一個已載入模型，測重複請求、換題、取消、釋放槽位、重試；
與本輪 LRU 輸出的前 32 tokens 核對，不拿這些流程時間宣稱加速。

MTP／DSpark 的階段統計已接入；完整 sidecar 的路由感知速度不在本輪測量範圍。
V4.1 完整 checkpoint 未安裝，不宣稱已測。原始 App 已能編譯及通過設定測試，
沒有重新封裝或發布。

初版另有 V4／Qwen 1K 診斷結果，淘汰搜尋調整後四組全部重跑；
初版留在 `excluded/before-eligible-layer-scan/`，不混入最終版。
初次 Qwen 準備流程因發現兩段 Prefill 重設問題而中止，當時尚未執行 route。
修正後全部重跑；該份部分對照留在封存 `excluded/`，未納入比較。
