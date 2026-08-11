# DSpark-first 實作計畫

> [!WARNING]
> 本文件是 2026-08-08 的歷史計畫。DSpark 安裝與 runtime 已經實作。
> 請以[目前研究結論](../../docs/RESEARCH.md)和
> [驗證紀錄](../../docs/VALIDATION.md)為準。

Checked: 2026-08-08

## 決定

DSpark 改為下一階段的第一優先。

一般 prefill 與 decode 最佳化不再先做。

runtime 只先加入 DSpark 所需的基線量測。

第一個 DSpark 版本支援 batch size 1 與 greedy decode。

第一個 DSpark 版本提供明確的啟用開關。

實作優先順序與預設啟用條件是兩件事。

即使 DSpark 尚未加速，專案仍會先完成正確的實驗版本。

只有本機量測顯示淨加速時，App 才會預設啟用 DSpark。

## 已知條件

checkpoint 的 `mtp.*` namespace 儲存 DSpark。

`mtp.*` 共有 4,705 個 tensor。

這些 tensor 約占 10.12 GiB。

官方 inference configuration 定義三個 DSpark layers。

官方 configuration 也定義 draft block size 5、target hidden layers 40 至 42，以及 Markov rank 256。[官方 inference configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json)

DSpark 會共用 main model 的 embedding 與 output head。

DSpark 也需要 main model 的 hidden states、Markov head 與 confidence head。[官方 reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L743-L936)

目前 repacker 會排除全部 `mtp.*` tensor。

目前 installed model 因此不包含 DSpark。

目前 MXFP8 cache 不能 rollback。

DSpark verification 需要 cache checkpoint、commit 與 rollback。

## 執行流程

```text
main model 產生 hidden states
        ↓
DSpark 產生 draft block
        ↓
confidence head 選擇 verification prefix
        ↓
main model 一次驗證多個候選 token
        ↓
接受正確 prefix，回復未接受的 cache state
```

DSpark paper 報告的 60–85% 改善來自正式服務環境。

這個結果不能直接套用到 Apple Silicon 與 SSD expert streaming。[DSpark paper](https://arxiv.org/html/2607.05147)

本專案只使用本機 end-to-end 結果判定加速效果。

## 實作階段

### S0：固定 DSpark contract，預估 1–2 個工作日

S0 讀取 pinned checkpoint 的 `inference/config.json`。

S0 建立完整的 `mtp.*` tensor inventory。

inventory 記錄 tensor name、dtype、shape、bytes 與 source shard。

S0 將 tensor 分成 common tensor 與 routed expert。

S0 不依照名稱猜測 layout。

S0 以官方 configuration 與 tensor index 驗證 layout。

S0 建立小型 fixture checkpoint。

S0 也保存 main model 的 autoregressive 基線。

基線至少記錄下列項目：

- Decode Tok/s。
- 每個 token 的時間。
- expert bytes。
- cache hit rate。
- peak memory。

S0 完成條件：

- DSpark tensor inventory 沒有缺少或重複 tensor。
- configuration 與 checkpoint revision 完全固定。
- fixture 可以測試 DSpark repack plan。

### S1：加入 DSpark 安裝格式，預估 2–4 個工作日

S1 建立向後相容的 manifest version。

舊 installed model 仍可載入 main model。

新 installed model 可以選擇包含 DSpark。

manifest 明確記錄 DSpark 是否完整安裝。

manifest 也記錄 DSpark tensor table 與檔案 checksum。

repacker 將 `mtp.*` 寫入獨立的 `dspark/` 目錄。

既有 installed model 可以只下載並加入 `mtp.*`。

升級流程不重新下載 main model。

升級流程先寫入 staging 目錄。

所有 DSpark files 驗證完成後，升級流程才原子更新 manifest。

Swift 與 Python manifest reader 必須同時支援新版本。

S0 完成 tensor 分類後，repacker 才決定 routed expert 的 blob layout。

App 顯示 DSpark 的額外下載大小。

App 允許使用者選擇安裝或移除 DSpark。

main model 驗證與 DSpark 驗證分開執行。

S1 完成條件：

- fixture repack test 通過。
- 遺失或損壞的 DSpark file 會被偵測。
- 不含 DSpark 的 installed model 仍可正常啟動。
- DSpark 未完整安裝時，runtime 不允許啟用 DSpark。

### S2：加入 cache transaction，預估 3–5 個工作日

S2 為每一種 main model cache 加入相同介面：

```text
checkpoint()
commit(accepted_tokens)
rollback()
```

cache transaction 必須處理下列 state：

- compressed KV chunks。
- incomplete compression tail。
- sliding-window state。
- attention offset。
- lightning index cache。
- cached packed representation。

S2 不複製完整 context cache。

S2 只保存最多一個 draft block 所需的暫存 state。

已接受 token 會 commit。

未接受 token 會 rollback。

S2 完成條件：

- 接受 0 至完整 draft block 的每一種情況都通過測試。
- compression boundary 前後都通過測試。
- sliding-window boundary 前後都通過測試。
- rollback 後的下一個 logits 與 autoregressive path 相同。
- 12K-output 測試沒有持續 resource growth。

### S3：載入並執行 DSpark model，預估 5–10 個工作日

S3 將官方 `DSparkBlock` 移植到 MLX runtime。

S3 先實作官方 checkpoint 已使用的運算。

S3 不先建立通用 speculative decoding framework。

main model 暴露 layers 40 至 42 的 hidden states。

DSpark 共用 main model 的 embedding 與 output head。

DSpark 載入 Markov head 與 confidence head。

DSpark 保留 checkpoint 的 FP4、FP8 與 scale layout。

S3 先測試 resident DSpark。

目前 8K main model peak memory 是 22.2 GiB。

DSpark weights 約占 10.12 GiB。

兩者相加約 32.3 GiB，但這個數字不包含執行暫存 memory。

若 peak memory 保持在 48 GiB 內，第一版讓 DSpark weights resident。

若 peak memory 超過限制，runtime 才對 DSpark routed expert 使用 streaming。

S3 完成條件：

- 每個 DSpark layer 的 tensor shape 完全符合 contract。
- fixture forward output 符合 reference path。
- route IDs、draft logits 與 confidence values 通過比較。
- main model 在 DSpark 關閉時沒有輸出差異。

### S4：加入 greedy DSpark scheduler，預估 3–5 個工作日

S4 產生一個 draft block。

S4 依照 confidence head 選擇 verification prefix。

S4 使用 main model 一次驗證候選 tokens。

S4 接受與 main model 相同的最長 prefix。

S4 使用 target token 修正第一個不相同的 token。

S4 commit 已接受的 cache state。

S4 rollback 未接受的 cache state。

API streaming 只送出已確定接受的 token。

API 不送出可能 rollback 的 token。

S4 先支援 greedy decode。

sampling 支援放在 S6。

S4 完成條件：

- DSpark 與 autoregressive path 產生完全相同的 greedy token sequence。
- 接受長度 0 至完整 block 的測試都通過。
- SSE 與 `/v1/responses` 不會重複或遺失 token。
- Tool call output 與非 DSpark path 相同。
- 使用者可以在不重啟 App 的情況下停用 DSpark。

### S5：量測並最佳化 DSpark，預估 3–7 個工作日

S5 先量測完整 DSpark round。

每個 round 記錄下列項目：

- draft latency。
- target verification latency。
- proposed tokens。
- accepted tokens。
- rejected tokens。
- accepted length。
- confidence threshold。
- DSpark expert bytes。
- main model expert bytes。
- cache hit rate。
- peak memory。

API metrics endpoint 回傳這些 DSpark 指標。

App 顯示目前 round 與歷史 DSpark 指標。

主要判定值是：

```text
(draft time + target verification time) / accepted output tokens
```

S5 比較 draft block size 與 confidence threshold。

S5 先使用官方 configuration 的預設值。

S5 只有在量測支持時，才調整這些值。

S5 也比較 resident DSpark 與 streaming DSpark。

S5 完成條件：

- 4K、8K 與 14K context 完成 256-output 測試。
- code、Tool-like 與一般對話 workload 分開報告。
- 每個短基準執行五次。
- 報告 median Decode Tok/s 與 p95 token time。
- 14K context 完成一次 2K-output 測試。
- 14K context 完成一次 12K-output 穩定性測試。

### S6：加入 sampling，延後處理

S6 驗證 speculative sampling 的分布正確性。

S6 不屬於第一個 DSpark 版本。

## 預設啟用門檻

DSpark 完成實作後，App 先將 DSpark 標示為實驗功能。

DSpark 只有同時符合下列條件時，才預設啟用：

- 三種 workload 的 median Decode Tok/s 都至少改善 10%。
- 沒有一種 workload 的 p95 token time 惡化超過 5%。
- greedy token sequence 完全相同。
- peak memory 不超過 48 GiB。
- 12K-output 測試沒有 resource growth。
- API streaming、Tool call 與 `/v1/responses` 測試全部通過。

若 DSpark 沒有達到門檻，App 仍保留手動啟用選項。

App 必須清楚顯示目前 DSpark 是 resident 或 streaming 模式。

## 主要風險

### Cache rollback

這是第一個必要技術項目。

如果 rollback 不正確，DSpark 會產生錯誤 token。

### Memory

DSpark 會增加約 10.12 GiB weights。

runtime 必須量測實際 peak memory。

runtime 不能只計算檔案大小。

### Expert I/O

DSpark draft layers 也可能選擇 routed expert。

target verification 也可能增加每個 round 的 main model expert bytes。

較高 acceptance 不一定代表較高 Decode Tok/s。

### Hidden states

main model 必須保留指定 layers 的 hidden states。

runtime 不能讓這些 hidden states 延長整個 request 的 memory lifetime。

### Streaming

API 只能送出已確認 token。

runtime 不能先送出 draft token 再撤回。

## 第一個實作工作

下一步先完成 S0。

S0 會修改下列範圍：

- checkpoint companion files。
- DSpark configuration decoder。
- `mtp.*` tensor inventory。
- DSpark model contract。
- fixture checkpoint 與 repack tests。
- autoregressive decode 基線指令。

S0 不會先修改 generation scheduler。

S0 完成後，S1 才會修改 installed model 格式。

## 來源

- [官方 inference configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json)
- [官方 tensor index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json)
- [官方 reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L743-L936)
- [官方 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md#how-to-run-with-vllm)
- [DSpark paper](https://arxiv.org/html/2607.05147)
- [本地 runtime research](RUNTIME_RESEARCH_2026-08-07.md)
- [本地 validation](../../docs/VALIDATION.md)
- [本地 repack planner](../../Sources/DeepSeekRepack/RepackPlanner.swift)
- [本地 MXFP8 cache](../../runtime/deepseek_v4_ssd/fp8_cache.py)
