# Qwen attention 與 expert 計算：第二輪

使用者已允許持續使用 GPU，之後無須逐次詢問。
品質容忍幅度仍需另行決策；本輪先測不刻意改變計算結果的候選。

## 事前規則

1. 先以同一個 code 4K／32 做組件 observer，核對前一輪正常輸出的完整 token hash。
   計時區分完整 GatedDeltaNet、QSA（含 core）、MoE；core 是 QSA 的子集，不重複加總。
   Prefill MoE 不含外部 full-layer 讀取；同步會改變重疊，僅供投入排序。
2. 若 QSA 有明顯成本，試目前每次 4 個 query 改成 8／16／32。
   只改工作批次，不改 key 選取、causal mask、softmax 精度、量化或 model defaults。
   這是 [FlashAttention-2 §3.2–3.3](https://arxiv.org/html/2307.08691v1)
   所討論工作分配與 block size 取捨的本機假設；不是論文 kernel 的重現，也沒有保證會更快。
3. 先 code 4K／32 篩選；所有候選完整 tokens 必須與同題控制一致。
   不一致即停止該候選進入精確路線，不將它偷偷視為通過的近似方案。
4. 最佳可用候選再跑五類 4K／256，兩組相反順序成對比較，observer 關閉。
   入選要求主要四類 TTFT 改善中位數至少 5%；逐題 Decode 不退步超過 5%、p95 不增加超過 10%，
   峰值記憶體不增加超過 5%，全部 output token hashes 相同。
   repeated 只作控制。這批使用既有合成輸入，不是最終採用的獨立資料；通過後再做 held-out 與長輸入驗證。
5. 固定 Qwen 4,096 slots、exact、MTP off、48 GiB、ANE ratio 0.25、4／2 workers。
   每次獨立 process，prompt cache 不重用，OS cache 不 purge；順序與原始資料完整保留。
6. 若入口篩選只顯示誤差或回退，先量是哪一段成本抵銷；不否定整個融合／工作分配方向。

狀態：組件計時待執行。工具只在研究 process 中替換函式，不改 runtime 或已安裝依賴。
