# Qwen Prefill：先量 Gated Delta 的成本

2026-09-06 使用者已明確允許繼續使用 GPU，解除前一輪暫停。
本輪沿用 B 路線，不改模型或 App 預設。

## 事先固定的方法

依據 [Gated Delta Networks §3.2](https://arxiv.org/html/2412.06464v1)，
逐步遞推可以改寫為分塊矩陣計算。論文支持演算法方向，不能證明本機收益。
先測現有運算，避免只因看到逐步迴圈就投入改寫。

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

## 結果

正常基準、observer 與正常重跑均完成，各 32 tokens，全部與 9 月 5 日同題的前 32 tokens 相同。
runtime Python 檔案 hashes 與上一輪一致；模型 manifest、observer 依賴與保存輸入的 hashes 已核對。
使用 M5 Pro／64 GiB、MLX 0.32.0、mlx-lm 0.31.3。

| 測試 | TTFT（秒） | 用途 |
| --- | ---: | --- |
| 正常基準 | 69.910 | 無分段 observer |
| 分段 observer | 44.209 | 組件計時；會改變同步與重疊 |
| 正常重跑 | 43.981 | 因時間差距大而追加，檢查判斷是否依賴第一次較慢的基準 |

不能將 69.910 → 44.209 秒當成加速結果，這兩次沒有候選演算法。
OS page cache 未清除，編譯、載入與系統狀態也未完整控制，因此不歸因某一原因。

Observer 記錄 1,332 次呼叫，其中 144 次為 Prefill：
108 次 T=1,024、36 次 T=1,023，合計處理每個 recurrent layer 的 4,095 個位置。
真實形狀為 q/k `[1,T,16,128]`、v `[1,T,48,128]`；輸入 BF16，state FP32。
其餘 1,188 次 T=1 包含最後一個 prompt 位置及 Decode，不當純 Decode 組件統計。

- Prefill 呼叫總計 **0.5528 秒**。
- 扣除兩種形狀的首次呼叫後，142 次共 **0.5437 秒**。
- 非首次 T=1,024／1,023 呼叫的中位數分別為 **3.819／3.700 ms**。
- 0.5437 秒相對兩次正常 TTFT 為 **0.78%／1.24%**，皆低於事前設定的 5% 投入排序門檻。

因此，本案例暫緩完整 chunkwise kernel 改寫，未實作或採用候選。
此比例來自受同步干擾的組件計時，**不是嚴格瓶頸占比，也不是速度改善的硬上限**。
它足以支持本輪的投入排序，不足以否定 H2、論文方法或更長輸入的潛力。
三次 32-token 一致也不代表長生成或其他輸入已驗證。

## 下一步與重啟條件

接著優先量 QSA attention、完整 GatedDeltaNet（含投影）與 expert 計算的成本，
同樣先核對輸出，再判斷是否進入 H1 或其他原型。
這些仍是待測問題；目前不能因 recurrent kernel 很小，就斷言剩下的時間都在 SSD 或某個矩陣乘法。
H1 的研究依據與限制沿用總計畫 P1（FlashAttention-2）；需要讀方法並核對 QSA 結構後才設計替換。

H2 若在更長輸入、另一個 chunk 設定或不同硬體上量到明顯成本，可重啟單層分塊原型。
重啟時仍需量輸出與 recurrent state、同預算下的額外記憶體，最後才量完整生成。

本輪 GPU 指令已執行完畢，沒有常駐測試程序；使用者本輪已允許 GPU 研究，未要求再次暫停。

## 保存與驗證

- [機器可讀結果](../docs/benchmarks/2026-09-06-qwen-gated-delta-prefill-diagnosis-m5-pro.json)：三次完整 metrics／命令、逐呼叫計時、來源／模型／輸入 hashes 與限制。
- [執行前規則快照](../docs/benchmarks/2026-09-06-qwen-gated-delta-protocol.md)：保留量測開始時的原文 hash。
- [observer 工具](../Scripts/profile_gated_delta_prefill.py)：只在獨立 process 包裝現有函式，不修改 runtime 或已安裝依賴。
- `scratch/gated-delta-2026-09-06/`：stdout／stderr、規則／來源快照，以及兩組真實組件輸入。

驗證包含 Python 編譯、三次 32-token hash／exact／MTP-off／prompt-cache 檢查、
144 次 Prefill／1,332 次總呼叫檢查、來源與樣本 hash 核對。
沒有執行完整 App／runtime 測試；runtime 程式未改動。
