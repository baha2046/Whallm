# Qwen Prefill：先量 Gated Delta 的成本

2026-09-06 使用者已明確允許繼續使用 GPU，解除前一輪暫停。
本輪沿用 B 路線，不改模型或 App 預設。

## 事先固定的方法

依據 [Gated Delta Networks §3.2](https://arxiv.org/html/2412.06464v1)，
逐步遞推可以改寫為分塊矩陣計算。論文支持演算法方向，不能證明本機收益。
先測現有運算，避免只因看到逐步迴圈就投入改写。

- 使用上一輪的 code 4K 輸入，先跑正常 CLI，再跑獨立 observer，各輸出 32 tokens。
- Qwen 4,096 slots、exact、MTP off、48 GiB、4 read workers、2 prefetch workers、ANE ratio 0.25。
- 每個 request 獨立 process，persistent prompt cache off，OS page cache 不清除。
- 32 個輸出 token 必須一致，且與上一輪同題 256-token 基準的前 32 個一致。
- Observer 只在此 process 包裝已安裝的 `gated_delta_kernel`；先等待輸入就緒，再計時至輸出就緒。
- 同形狀首次呼叫另外標記，不能把編譯成本當作反覆運算成本。
- 同步會改變重疊與排程，故 observer 耗時只供組件診斷，不當正常延遲或優化收益。
- 每種 Prefill 形狀保留一組真實輸入，生成結束後才存檔，以便單層原型重播。
- 先用非首次 Prefill 呼叫的總時間／正常 TTFT 作投入排序參考；這不是嚴格可加總的瓶頸占比。
  若低於 5%，暫緩完整改寫；若達到 5%，先做實際形狀的單層比較，仍需完整生成確認收益。
- 原型若改變 greedy tokens，不以公式相同為由放行；需區分數值誤差與能力評估。

目前狀態：正常基準與 observer 待執行。raw 輸出放 `scratch/gated-delta-2026-09-06/`。
