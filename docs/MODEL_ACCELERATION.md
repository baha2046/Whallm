# 三模型加速功能（2026-09-14 原始碼）

本次補齊模型比較中可實作的項目，並加入 Model → Advanced Settings。
已完成 local build `1.1.7 / 1.1.7d2`，尚未發布；不代表完整 checkpoint 已測得速度提升。
成品檢查見 [本次打包紀錄](VALIDATION.md#2026-09-14三模型加速-local-build)。

## 功能與設定

| 功能 | DeepSeek V4 | DeepSeek V4.1 | Qwen3.8 |
| --- | --- | --- | --- |
| 專家讀取完成即開始運算 | 已有；新增 UI | 已有；新增 UI | 新增；保留 router 原始輸出順序 |
| 按層處理 Prefill | 已有 | 新增；預設關閉 | 已有 |
| Prefill 專家合批 | 已有；新增 UI | 新增 | 已有；新增獨立 UI |
| 下一層專家預讀 | 既有讀取排程 | 新增；需按層、合批 | 既有路徑新增 UI |
| KV 低位元保存 | 既有 FP8 | 新增原生 FP8 視窗／FP4 壓縮 KV | 新增 QSA 8-bit 保存 |
| 索引低位元保存 | 既有 FP4 | 新增原生 FP4 | 新增 QSA 4-bit 保存 |
| 只計算候選區塊的索引分數 | 不適用此訓練架構 | 新增 | 不適用 V4.1 的跨層候選共享方式 |
| CED：略過後段不再需要的舊 token | 不適用此訓練架構 | 新增；需按層 Prefill | 不適用此訓練架構 |
| ANE 分擔 query projection | 新增；預設關閉 | 新增；預設關閉 | 既有 |
| 訓練好的候選模型、主模型確認 | 既有 DSpark | 新增 DSpark 安裝與執行 | 既有 MTP |
| 少選一個專家的近似模式 | 既有 | 新增 | 新增 |

新增的低位元保存、候選索引、CED、下一層預讀與 DeepSeek ANE 都預設關閉。
Ready-expert Decode 與 Prefill 合批沿用開啟預設。
三個模型的近似模式預設皆為 **Exact／關閉**；保留使用者明確儲存的開關。
近似模式將 learned router 的 top-k 減一；V4 hash router 保持原值。
結束、取消或例外時恢復 top-k。DSpark／MTP 與近似模式不能同時使用。

兩項功能已從 runtime、CLI、設定與 UI 移除：

- 從已有文字提出候選、一次確認最多四個 token。
- 生成時將多個常駐專家合批計算。

舊 catalog 中 `qwen_short_block=false`、`qwen_grouped_decode=false` 可讀取並忽略；
設為 true 或錯誤型別則拒絕。歷史研究與 benchmark 素材保留，不能當成目前功能。
Prefill 的專家合批與使用訓練權重的 DSpark／MTP 繼續保留。

## CLI、API 與 UI

以下新旗標同時支援 CLI 與 server；可用 `--no-…` 關閉。
JSON 使用對應的 snake_case 欄位。UI 變更在下次載入模型時生效。

| 旗標 | 模型 | UI |
| --- | --- | --- |
| `--v41-layer-major-prefill` | V4.1 | 按層處理輸入 |
| `--v41-packed-kv` | V4.1 | 壓縮 KV 快取 |
| `--v41-packed-index` | V4.1 | 壓縮索引快取 |
| `--v41-candidate-index` | V4.1 | 只計算候選區塊的索引 |
| `--v41-ced-prefill` | V4.1 | 略過不再需要的舊 token |
| `--v41-next-layer-prefetch` | V4.1 | 提前載入下一層專家 |
| `--qwen-quantized-kv` | Qwen | 壓縮 KV 快取 |
| `--qwen-quantized-index` | Qwen | 壓縮索引快取 |
| `--deepseek-ane-prefill` | V4／V4.1 | 使用 ANE 處理輸入 |
| `--no-ready-expert-decode` | 全部 | 專家載入完成就開始計算 |
| `--no-batched-expert-prefill` | 全部 | 合批處理輸入的專家運算 |
| `--qwen-next-layer-prefetch` | Qwen | 提前載入下一層專家；server 使用 JSON |

V4.1 按層 Prefill 同時需要既有 `layer_major_prefill` 主開關；CED 又依賴按層處理。
DSpark 使用自己的 Prefill 流程，因此 V4.1 DSpark 與按層 Prefill／CED／下一層預讀
互斥。UI 停用相依控制項，catalog 保留使用者偏好但只送出當下有效組合；
直接設定不相容的 CLI／JSON 組合會報錯。

## 格式與正確性

V4.1 保存原生量化資料，而不是先還原、再重複量化：視窗為 E4M3／UE8M0、
壓縮 KV 為 E2M1／E4M3、索引為 E2M1／UE8M0。
小模型的 logits、續寫快取、序列化恢復與既有 fake-quant 路徑完全相同。
Packed cache 使用狀態格式 2，舊 BF16 狀態仍用格式 1；快取身分記錄格式選擇。

Qwen 使用 MLX 的 8-bit／4-bit affine cache，僅改動 QSA，DeltaNet 狀態不變。
這是額外有損量化，可能改變索引選擇與輸出。保存的資料較小，但目前 attention
會建立完整歷史的暫時解量化視圖；不能宣稱峰值記憶體或速度一定改善。
格式、位元數、維度與 offset 隨快取保存，設定不同的快取不能混用。

V4.1 候選索引只在來源已提供有效候選且候選數足夠時使用；否則回到完整索引計算。
CED 保留每層最後視窗所依賴的前文範圍，按原有 chunk 邊界裁切，避免改變運算分組。
共享 KV 來源層仍處理完整輸入；後段才逐層略過舊 token，不宣稱等同官方的效能數字。

DeepSeek ANE 分割每層 `wq_b` 的輸出通道，沿用 `ane_prefill_ratio`（預設 0.25）。
只有 batch 1、1024 token 使用 ANE；其他尺寸、Decode 與失敗情況回到原 GPU 計算。
所有層編譯完成後才替換投影；ANE 使用 FP16，可能改變捨入與輸出。

## V4.1 DSpark 安裝

App 的安裝 DSpark 動作與 `dsv4-repack install-dspark --model …` 依模型分流。
V4.1 主模型仍可獨立安裝；選用 DSpark 時增加 `dspark/common.bin`、三個
`dspark/experts/layer_XX.bin` 與 `inference/config.json`。修復會保留已安裝 DSpark。

固定 revision 的 DSpark 為 3 層、每層 128 專家、top-3、block size 5、
noise token 128799、target layers 37／38／39、Markov rank 256；共有 97 個 common tensors。
它使用 V4.1 的 staggered hyper-connections、主模型 embedding/head、獨立專家快取。
主模型提供 target layer 進入 block 前的隱藏值；候選由既有接受／拒絕流程驗證。
未接受的分支不會覆寫主模型快取。

V4.1 不開放 V4 的 hash 預讀、自適應候選長度、逐 token／混合驗證與跨請求
DSpark prompt cache 實驗開關；一般候選驗證、信心門檻與速度回退保護可用。

來源：[固定 revision 的官方推論程式](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/inference)。
已核對實際 safetensors headers 的 97 個 common tensor 名稱、形狀及 FP8 scale 配對。
原始 header 摘錄位於 `scratch/2026-09-14-model-features/`，不包含模型權重。

## 驗證範圍

新增測試涵蓋七層 CED 的視窗重建、共享索引、Engram 歷史、packed cache 成長與還原、
QSA packed cache 續寫、專家完成順序、ANE 失敗回退、DSpark 候選驗證與獨立狀態，
以及 Swift 安裝配置、設定預設與互斥設定。

另在 Apple M5 Pro 的真實 ANE 上執行一個 32→512 投影，batch 1、1024 token、ANE 分工 50%；
確認 ANE active 且完成一次運算，與 GPU 的最大絕對差為 0.002398。
這是小投影的可執行性檢查，不是完整模型速度或品質測試。

尚未下載或執行完整 V4.1 DSpark 權重，也未跑三個完整模型的速度／品質比較。
已完成本機封裝、ad hoc 簽章、App／ZIP 隔離啟動及包內 12 項功能測試；未公證或發布。
