# DSpark runtime 最佳化研究

> [!WARNING]
> 本文件是 2026-08-08 的歷史研究。部分問題已由目前程式碼修正。
> 請以[目前研究結論](../../docs/RESEARCH.md)和
> [驗證紀錄](../../docs/VALIDATION.md)為準。

日期：2026-08-08

## 結論

DSpark 應繼續預設停用。

本機的 16-token 對照測試顯示：normal decode 是 5.68 Tok/s。最佳化前的 DSpark 是 2.49 Tok/s。最佳化後的 DSpark 是 4.59 Tok/s。DSpark 比自身舊版快約 84.6%，但仍比 normal decode 慢約 19.2%。兩條路徑的 greedy 輸出完全相同。

runtime 已加入四項保守最佳化：

1. confidence scheduler 可把低 confidence draft 縮短成零個 token。
2. runtime 依本機實測時間判斷 DSpark 是否較慢。較慢時，後續 token 自動改走 normal decode。
3. target verification 不再建立永遠不會使用的最後一份 43 層 cache checkpoint。
4. DSpark routed expert 使用獨立的 256-slot SSD cache。DSpark common tensor 仍常駐記憶體。

目前的下一個主要目標是用 logical trim 或 crop 取代每一個 rollback 位置的 43 層 cache copy。

256-slot SSD cache 的 16-token 實測結果如下：

- 峰值記憶體從 33.75 GB 降到 24.10 GB。
- 冷 cache Decode 從 4.59 提升到 5.06 Tok/s。
- 第一個請求從 SSD 讀取約 599 MiB 的 DSpark expert blob。
- 同一個 runtime 的第二個相同請求達到 100% DSpark cache hit rate。該請求沒有再次讀取 DSpark expert blob。
- 第二個相同請求的 Decode 是 6.47 Tok/s。峰值記憶體是 24.54 GB。
- normal decode、resident DSpark 和 SSD-cache DSpark 的 greedy 輸出完全相同。

第二個目標是降低 DSpark MoE 成本。runtime 應至少比較兩種模式：

1. DSpark 權重常駐統一記憶體。
2. DSpark routed expert 使用獨立的 SSD slot cache。

第三個目標才是 batch size 1 的 confidence scheduling。confidence 只能縮短送入 target verification 的前綴。confidence 不能直接接受 token。

## 成功條件

本機的 16-token 短測試容易受到首次 MLX 編譯和 SSD cache 狀態影響。完整判斷仍需要多種 prompt、較長輸出和多次重複測試。

DSpark 論文把每 token 延遲寫成：

`(draft latency + verification latency) / committed tokens per round`

因此，runtime 必須同時降低 draft latency、降低 verification latency，或提高每回合提交的 token 數。只提高 acceptance rate 不保證 Tok/s 提高。[DSpark 論文，第 2.1 節](https://arxiv.org/html/2607.05147v1#S2.SS1)

每次基準測試必須記錄：

- draft latency
- target verification latency
- 每回合 draft 長度
- 每回合接受長度
- 每個位置的接受率
- confidence 分布
- main model 與 DSpark 的 expert cache hit rate
- main model 與 DSpark 的 SSD bytes read
- cache rollback latency
- end-to-end Tok/s

## 1. Target verification 與 cache rollback

DSpark 的正確工作單位是一個 draft block。target model 一次計算整個候選區塊。verifier 從左到右接受最長連續前綴。第一個拒絕位置之後的 draft token 全部失效。[DSpark 論文，第 2.1 節](https://arxiv.org/html/2607.05147v1#S2.SS1)

DeepSpec 的參考實作一次把 `draft_token_count + 1` 個 token 送入 target model。實作再計算接受前綴和 correction 或 bonus token。[DeepSpec verifier](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L186-L304)

DeepSpec 在每回合結束後，把 target KV cache 裁切到已提交位置。DeepSpec 也在 draft forward 後裁掉 speculative suffix。[target cache crop](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py#L385-L426) [draft cache crop](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py#L22-L45)

目前 runtime 在每一層逐 token 執行 attention。每個位置都呼叫 `mx.eval()`。runtime 也為 rollback 位置複製各層 cache array。[目前 verification 實作](../../runtime/deepseek_v4_ssd/model.py#L555-L647) 目前 scheduler 在拒絕後，以該位置的完整 checkpoint 取代 prompt cache。[目前 rollback 實作](../../runtime/deepseek_v4_ssd/dspark.py#L493-L533)

MLX-LM 也要求 speculative decoding 使用可裁切的 cache。驗證後，MLX-LM 依 `num_draft - num_accept` 裁切 target cache。MLX-LM 另行裁切 draft cache。[MLX-LM speculative decode](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/generate.py#L509-L635)

vLLM 會依拒絕的 token 數回退 request 的計算位置。vLLM 的 KV cache manager 只提交 finalized token。[vLLM scheduler rollback](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/core/sched/scheduler.py#L1781-L1793) [vLLM KV cache commit](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/core/kv_cache_manager.py#L554-L563)

工程推論：本專案應保留一份回合起點和每層的邏輯長度。target verification 可以先寫入預留尾端。驗證後，runtime 只更新邏輯長度。runtime 只有在 cache 實作無法裁切時，才需要複製 cache 資料。這個方法可避免每個 draft 位置各複製一次 43 層 cache。

工程推論：target verification 可能把被拒絕 token 所需的 routed expert 載入 main model expert cache。這些 expert 可能擠掉下一個正常 token 需要的 expert。基準測試必須分開記錄 verification 對 cache eviction 和後續 miss rate 的影響。可以測試一個短期 verification cache 或較低的 speculative admission priority。

## 2. Confidence scheduling

DSpark confidence head 預測「前面位置已接受時，目前位置會被接受」的條件機率。完整前綴的存活機率是各位置 confidence 的累積乘積。[DSpark 論文，第 3.2.1 節](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS1)

論文的 hardware-aware scheduler 使用兩項資料。第一項是前綴存活機率。第二項是硬體上實測的 `SPS(B)` 曲線。排程器選擇預期 throughput 最高的驗證長度。[DSpark 論文，第 3.2.2 節](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS2)

DeepSpec 沒有公開 production scheduler。DeepSpec 的公開 evaluator 只在第一個低於靜態 threshold 的位置截斷前綴。[DeepSpec confidence prefix](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/draft_ops.py#L82-L152)

vLLM 官方文章也把 confidence-based scheduling 標為仍在進行。現有結果不能證明 confidence threshold 在本專案會加速。[vLLM Kimi K3 DSpark 說明](https://github.com/vllm-project/vllm-project.github.io/blob/f7f4a8034cd4796c59e76a388c9a5e38e607c737/_posts/2026-07-27-k3.md)

工程推論：batch size 1 沒有跨 request 的 verification budget。論文的多 request 排程器不能直接套用。本專案可離線量測每個候選長度 `0...5` 的實際回合時間。runtime 再用以下值選擇長度：

`預期提交 token 數 / 實測回合時間`

confidence 門檻應維持可選。runtime 應先校準每個位置的 confidence。未校準的 raw confidence 不適合直接估算 throughput。論文也使用後處理校準來修正累積接受機率。[DSpark 論文，Post-hoc Calibration](https://arxiv.org/html/2607.05147v1#S3.SS2.SSS1)

## 3. Greedy output equality

greedy verifier 必須在每個位置計算 target argmax。draft token 只有在等於 target argmax 時才能接受。第一個不相等的位置必須輸出 target argmax。後續 draft token 必須丟棄。[vLLM greedy verifier](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/sample/rejection_sampler.py#L453-L469) [vLLM greedy kernel](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/sample/rejection_sampler.py#L743-L769)

MLX-LM 使用相同規則。MLX-LM 預設 sampler 是 `argmax`。MLX-LM 只接受 target token 和 draft token 相同的連續前綴。[MLX-LM greedy speculative decode](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/generate.py#L520-L625)

工程推論：greedy verifier 不需要計算完整 softmax 或完整 log probability normalization。runtime 只需要 target logits 的 argmax 和 draft token 比對。若 API 不要求 logprobs，runtime 應避免 `logsumexp`。這個變更必須保持相同的 dtype、logits processor、tie 規則和 token 順序。

confidence 只能刪除尚未驗證的 suffix。confidence 不能改寫 target argmax。只要 target forward、cache 狀態和 argmax tie 規則相同，confidence 截短不會改變 greedy token 序列。

驗證應比較 normal decode 和 DSpark 的完整 token ID 序列。測試至少應包含 code、Tool-like、open chat、4K context、容易在前段拒絕的輸入，以及全 draft 接受的輸入。vLLM 也有 greedy equality 測試。[vLLM greedy equality test](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/tests/v1/spec_decode/test_rejection_sampler_utils.py#L199-L231)

## 4. Quantized MoE kernel

MLX-LM 的 `QuantizedSwitchLinear` 不會先解量化整個 expert tensor。它把 expert index 傳給 MLX `gather_qmm`。`gather_qmm` 直接執行量化的選取式矩陣乘法。[MLX-LM `QuantizedSwitchLinear`](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/models/switch_layers.py#L27-L90) [MLX `gather_qmm` API](https://github.com/ml-explore/mlx/blob/ed116d24ba016c200917ea0d4779d68c8859c4fe/mlx/ops.h#L1603-L1617)

工程推論：DSpark 的 routed expert 應保持每層堆疊格式。runtime 應把 routed expert index 交給 `gather_qmm`。runtime 應避免逐 expert Python dispatch。W1、W2、W3 應各形成一次批次操作。這個方向必須用 Metal capture 證明 dispatch 或解量化是主要成本。

vLLM 可從 checkpoint 的 `hf_config` 讀取 `dspark_draft_topk`。未設定時，vLLM 使用完整 draft vocabulary。設定後，vLLM 先依 base logits 選擇每個位置的 top-k 候選。vLLM 只對這些候選套用 Markov bias。vLLM 把其他 logits 設為 `-inf`。[vLLM DSpark top-k 設定](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L72-L77) [vLLM DSpark top-k 實作](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L135-L203)

工程推論：`dspark_draft_topk` 可作為基準候選。它可能減少 sequential draft head 的候選工作。但是它可能改變 draft token 和 acceptance rate。它不能改變 target verifier。專案應把它保持為關閉的實驗選項，並比較端到端 Tok/s 和 greedy equality。

## 5. SSD expert streaming 與 MLX

MLX 利用 Apple Silicon 統一記憶體。CPU 和 GPU 可直接存取相同的 array。MLX scheduler 會處理不同 stream 的資料相依。[MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)

但是，MLX 的 safetensors loader 不會建立 SSD-resident GPU tensor。loader 先建立 lazy `Load` primitive。primitive 評估時配置完整輸出 buffer，再從檔案 offset 讀入完整 tensor。[MLX safetensors loader](https://github.com/ml-explore/mlx/blob/ed116d24ba016c200917ea0d4779d68c8859c4fe/mlx/io/safetensors.cpp#L110-L217) [MLX load evaluation](https://github.com/ml-explore/mlx/blob/ed116d24ba016c200917ea0d4779d68c8859c4fe/mlx/backend/common/load.cpp#L30-L54)

工程推論：官方 MLX lazy load 不能取代本專案的 expert slot、eviction 和 prefetch。專案應繼續使用固定 slot。MLX 可負責 slot 上的量化計算。

工程推論：DSpark 只有三層，但 DSpark 權重約 10.12 GiB。權重常駐可避免每回合額外 SSD 讀取，但會壓縮 main model expert cache 和 KV cache 的可用記憶體。完整 streaming 可節省記憶體，但每個 speculative round 都可能增加 SSD latency。

應測試三個模式：

1. DSpark 全部常駐。
2. DSpark common tensor 常駐，routed expert 使用獨立 slot cache。
3. DSpark 與 main model 共用 slot pool，但使用獨立容量下限和 eviction 統計。

測試必須在相同 prompt 和相同 greedy 輸出下執行。選擇標準是端到端 Tok/s、peak memory 和 main model cache hit rate。SSD 頻寬本身不是選擇標準。

## 建議順序

1. 保持 DSpark 預設停用。
2. 確認 target model 每回合只執行一次 block verification。
3. 用 logical trim 或 crop 取代每位置的 43 層 cache copy。
4. 移除 greedy path 不需要的完整 normalization。
5. 記錄 verification 造成的 expert cache pollution。
6. 比較 DSpark 常駐和獨立 SSD slot cache。
7. 確認 DSpark routed expert 使用批次 `gather_qmm`。
8. 離線建立 batch size 1 的候選長度成本表。
9. 校準 confidence，再測試動態前綴長度。
10. 只有 DSpark 在代表性工作負載穩定高於 7.9 Tok/s，且完整 greedy token ID 相等時，才考慮預設啟用。

## 證據限制

DeepSeek 沒有公開 DeepSeek-V4 production verifier 和 cache runtime。DeepSpec 可以證明公開演算法和參考行為。DeepSpec 不能證明 production runtime 的 bitwise logits 或 cache 實作。

DSpark 論文的 60%–85% 改善來自多 request 的正式服務環境。該結果比較 MTP-1，並使用 production scheduler。該結果不能直接推定 Apple Silicon、batch size 1 或 SSD expert streaming 的速度。[DSpark 論文，摘要與部署結果](https://arxiv.org/html/2607.05147v1#S1)
