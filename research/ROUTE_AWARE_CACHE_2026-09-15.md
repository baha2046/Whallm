# 完整路由感知快取

使用者要求補齊跨淘汰的全專家歷史、長短期熱門度與逐層效益分配。
沿用當前固定 slots、canonical bytes、pinning 與讀取生命週期；新增 `route` 選項，舊預設保留。

## 研究依據與實作假設

[TinyLFU](https://arxiv.org/abs/1512.00727v2) 將近期使用統計與常駐資料分開，
用歷史評估保留價值。本實作借用此分離觀念；專家總數已知且小，使用精確陣列，
不需要 Bloom sketch。本版不是 TinyLFU admission：需要的權重仍必須載入既有 slots。
[MoE 快取評估研究](https://arxiv.org/abs/2608.07911v4) 指出逐次重播與實際聯集取得的
差異會扭曲結論，因此以真正生成為驗收，不將離線命中率當速度。

- 全專家短／長期 EMA：半衰期 16／256 個該層觀察到的 token；被淘汰後仍保留。
- Prefill 單獨統計，最多提供 0.1 權重的機率先驗，隨 Decode 衰減；不沖掉 Decode 歷史。
- 分數：0.75 短期＋0.25 長期＋先驗，乘上該層平滑讀取成本（相對中位數限 0.5–2 倍）。
- 每約 16 個完整 Decode tokens 重算各層的邊際收益，分配固定總 slots。
  配額是軟目標，正在使用／被 pin 的權重優先；不能為追求配額破壞可用性。
- 更新分數時重建各層的 resident heap，避免舊分數失效。淘汰先查看超額層，
  只有該組全部被保護才查看下一組。僅改權重保留，不改模型路由或數學。
- 歷史存活於 loaded model lifetime；不存磁碟，卸載後清除。推測／重播算觀察到的路由，
  不是只計已接受 tokens。完整 V4.1 checkpoint 仍不在本機測試範圍。

## 驗收

1. 淘汰後歷史、短長期轉移、Prefill 隔離、配額總和／熱層轉移、pinning、取消／失敗重試。
2. CLI、server catalog、App 選項與舊偏好相容；不改預設、不發布。
3. 本機 V4 1152／Qwen 3072，與 LRU 同容量、greedy、fresh process、prompt cache off。
   1K code／128 output、4K 繁中／128 output，各 AB／BA，先檢查輸出一致。
4. 記錄 request、Decode、bytes、peak、swapout、配額與歷史；所有 runtime 測試。
   不一致或新增 swapout 不採用速度數字；沒有 5% 收益也保留功能與負面結果，不假稱加速。

## 實作紀錄

- `route_cache.py` 保存固定大小的全專家短／長期陣列、輸入先驗與逐層目標容量。
- `ExpertCache` 沿用既有槽位與讀取生命週期，加入策略觀察和逐層淘汰候選。
- `ModelRuntime` 每個請求只重設一次輸入統計；兩段 Prefill、MTP／DSpark 和
  prompt warmup 都有階段邊界。歷史包括實際觀察到的推測與重播工作。
- App 新增選項，CLI／Server／catalog 接受 `route`；舊設定及預設保留。
- 首輪 V4／Qwen 1K 對照發現全層候選檢查的成本。改為按優先組別搜尋，
  用逐一檢查所有 resident entries 的測試核對淘汰結果，再固定版本重跑四組。
  初次資料留在 scratch 的 `before-eligible-layer-scan/`，不混入最終版本比較。

目前：功能及本輪驗收完成。Python 440 項、Swift 31 項設定測試通過；
16 次計時生成輸出一致、無新增 swapout，V4／Qwen 連續請求與取消重試亦通過。
1K 有正向訊號，4K 完整等待時間接近持平，保留手動選項及原預設。
完整 [量測、範圍與原始證據](../docs/benchmarks/2026-09-15-route-aware-cache/README.md)。
目前功能契約見 [docs](../docs/ROUTE_AWARE_CACHE.md)。
