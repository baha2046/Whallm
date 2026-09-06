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

## 結果

組件 observer 通過 code 4K／32 token parity；Prefill 的完整 GatedDeltaNet 約 1.366 秒、
MoE 約 19.799 秒、QSA 約 6.423 秒（其中 core 6.055 秒）。
這些是同步干擾下的分段時間，不是可直接相加的正常延遲占比。

入口五次 code 4K／32 的 TTFT 為：chunk 4 的控制 37.407／37.260 秒，
chunk 8／16／32 分別 36.825／36.377／37.007 秒。
所有 outputs 一致，選擇 chunk 16 進入原訂的五類／256-token 成對測試。

完整 20 組測試完成，10 對各 256 tokens 全部相同，也與前一輪基準相符。
下表為每題兩個相反順序配對的相對變化中位數；不是不同題目絕對秒數直接平均。

| 題目 | TTFT 改善 | Decode 變化 | Decode p95 變化 |
| --- | ---: | ---: | ---: |
| repeated 控制 | +2.74% | +0.65% | −1.66% |
| 程式碼 | +2.45% | +1.08% | −3.09% |
| 繁中技術 | +2.74% | +1.78% | +0.25% |
| 數學 | +2.69% | +0.34% | +0.11% |
| tool 格式 | +2.59% | +1.06% | −0.49% |

主要四題 TTFT 改善中位數 **2.64%**，低於 5% 門檻；Decode、p95、MLX memory／process RSS 保護條件通過。
因此停止 chunk 16 的採用，不改 runtime 預設。
這不是「完全沒變快」，而是改善幅度未達事先要求；仍只有兩對／題，且輸入不是獨立品質評估。
也不能因此否定 H1 的融合方向：本候選只改查詢批次，沒有省去 selected KV 和中間 scores／weights 的搬移。

下一步轉向 [expert 分組](QWEN_GROUPED_EXPERTS_2026-09-06.md) 的輸出差異診斷。
若重啟 QSA，需提出真正減少中間讀寫或其他新成本模型，不能只重試同一個 chunk 16。

## 保存與驗證

- [組件計時](../docs/benchmarks/2026-09-06-qwen-attention-moe-component-profile-m5-pro.json)。
- [五次篩選](../docs/benchmarks/2026-09-06-qwen-qsa-chunk-pilot-m5-pro.json)。
- [20 組完整結果](../docs/benchmarks/2026-09-06-qwen-qsa-chunk-paired-runs-m5-pro.json) 與 [門檻計算](../docs/benchmarks/2026-09-06-qwen-qsa-chunk-paired-gate-m5-pro.json)。
- [執行前規則](../docs/benchmarks/2026-09-06-qwen-qsa-chunk-protocol.md)。

工具只在研究 process 中替換函式，不改 runtime 或已安裝依賴。
原始檔案放 `scratch/qwen-attention-2026-09-06/`、`scratch/qsa-pilot-2026-09-06/` 與 `scratch/qsa-paired-2026-09-06/`。
檢查了每條 output／prompt hash、expert mode、MTP、cache reuse、20 組完整性，
以及研究 source 轉換／拒絕未完成量測的兩個 CPU 單元測試。
