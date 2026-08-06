# DeepSeek-V4-Flash-0731 與 TurboFieldfare 可行性

查證日期：2026-08-06

## 結論

TurboFieldfare 的「常駐共同權重，從 SSD 按需讀取 routed expert」設計可以用於 DeepSeek-V4-Flash-0731。

現有程式不能直接使用此模型。TurboFieldfare 是 Gemma 4 26B-A4B 專用執行器。DeepSeek-V4-Flash-0731 需要新的權重轉換器、Metal 核心、MoE、attention、KV cache 與提示編碼。

8 GB Mac 不可行。16 GB Mac 也不實際。24 GB Mac 可用於短 context 的實驗。32 GB 或更多記憶體較合理。這些容量判斷是估算，不是實機結果。

第一個版本應停用 DSpark。第一個版本應先完成主模型的短 context 單批次解碼。

## 模型名稱與供應狀態

**事實**

- 正確名稱是 `deepseek-ai/DeepSeek-V4-Flash-0731`。
- 官方倉庫在 2026-07-31 建立。查證時倉庫公開且未設 gate。
- DeepSeek 稱此模型為 DeepSeek-V4-Flash 的正式版本。此模型取代 preview 版本。
- 此 checkpoint 附加 DSpark speculative decoding 模組。
- 倉庫和模型權重採 MIT License。

來源：[官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)、[官方 Hugging Face API 記錄](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731)、[官方 LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE)

## 架構比較

| 項目 | TurboFieldfare 固定的 Gemma 4 | DeepSeek-V4-Flash-0731 | 影響 |
| --- | --- | --- | --- |
| 主模型層數 | 30 | 43 | 必須改寫固定的 `ArchConfig` 與執行流程。 |
| hidden size | 2,816 | 4,096 | 所有 Metal 矩陣核心要使用新 shape。 |
| routed expert | 每層 128 個 | 每層 256 個 | expert 索引、檔案與 cache 都要擴充。 |
| 每 token 選擇數 | 8 個 | 6 個 | 必須實作 6-of-256 router。 |
| shared expert | 每層 1 個 | 每層 1 個 | 可以保留「shared expert 與 SSD 讀取重疊」的排程概念。 |
| expert FFN | GELU，寬度 704 | SwiGLU，寬度 2,048 | 現有 expert 核心不能重用。 |
| routing | score top-8 | 前 3 層使用 token ID hash；其他層使用 `sqrtsoftplus` top-6 | 必須實作兩種 routing。 |
| attention | 25 個 sliding-window 層；5 個 full-attention 層 | CSA/HCA 混合；壓縮比例 4 或 128；window 128 | 現有 attention 與 KV cache 不能重用。 |
| residual | 一般 residual 加模型專用 scalar | 4 路 mHC | 必須實作 mHC 與 Sinkhorn 混合。 |
| embedding/head | tied | untied | 需要另一份 output head。 |
| context 上限 | 執行器目前針對較短 context | 1,048,576 tokens | 長 context 需要額外 KV 記憶體。 |
| speculative decoding | 無 | DSpark，block size 5 | 應在主模型完成後再加入。 |

DeepSeek 的比例 4 attention 層另有 indexer。Indexer 最多選 512 個壓縮位置。官方 reference code 目前用 BF16 儲存 KV。官方 vLLM 指令則建議 FP8 KV cache。

來源：[DeepSeek config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/config.json)、[DeepSeek reference config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/config.json)、[DeepSeek reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py)、[TurboFieldfare system design](https://github.com/drumih/turbo-fieldfare/blob/3249be40a33ea6560b35531c184609d7be67ac1a/docs/SYSTEM_DESIGN.md)、[TurboFieldfare `ArchConfig`](https://github.com/drumih/turbo-fieldfare/blob/3249be40a33ea6560b35531c184609d7be67ac1a/Sources/TurboFieldfare/Infrastructure/ModelIO/ModelTypes.swift)

## 參數與權重布局

**事實**

- DeepSeek-V4-Flash 主模型的官方數字是 284B 總參數與 13B activated parameters。
- 0731 checkpoint 的 Hugging Face 記錄是 304,180,418,494 個 tensor elements。此記錄包含 DSpark 權重。
- 官方 index 記錄 48 個 safetensors shard。總檔案資料是 166,878,536,440 bytes，約 155.4 GiB。
- 0731 checkpoint 內有 43 個主模型 MoE block。Checkpoint 也有三組 `mtp.*` MoE 權重。
- 官方資料沒有公布「啟用 DSpark 後，每一個輸出 token 的精確 activated parameter 數」。

來源：[DeepSeek-V4 模型卡中的主模型數字](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)、[0731 API 記錄](https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash-0731)、[0731 safetensors index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/model.safetensors.index.json)

**權重格式事實**

- DeepSeek routed expert 使用 packed FP4 E2M1。
- 每一個 byte 包含兩個 FP4 值。
- FP4 沿 K 維度每 32 個值使用一個 UE8M0 scale。
- 多數其他權重使用 FP8 E4M3。FP8 block 是 128 × 128。Scale 格式是 UE8M0。
- TurboFieldfare 使用 MLX affine INT4。Group size 是 64。每組有 BF16 scale 與 BF16 bias。Router 使用 INT8。
- 兩個布局不相容。現有 TurboFieldfare INT4 Metal 核心不能直接讀取 DeepSeek FP4。

來源：[DeepSeek 轉換程式](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/convert.py)、[DeepSeek reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py)、[TurboFieldfare README](https://github.com/drumih/turbo-fieldfare/blob/3249be40a33ea6560b35531c184609d7be67ac1a/README.md)

Apple 的 Metal 4 tensor 格式包含 `MetalFloat4E2M1`、`MetalFloat8E4M3` 與 `MetalFloat8UE8M0` block scale。這項支援有助於實作新核心。這項支援不會讓現有 TurboFieldfare 核心自動相容。[Apple Metal Feature Set Tables，第 16 頁](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

## 儲存空間、記憶體與 I/O

以下數字是依官方 shape 與檔案大小計算的**估算**。

### Routed expert 大小

一個 routed expert 有 `w1`、`w2` 與 `w3`。三個矩陣共有 25,165,824 個 FP4 值。Packed 值與 scale 合計約 13,369,344 bytes，也就是 12.75 MiB。

- 43 層主模型的全部 routed expert 約 137.06 GiB。
- 加上 checkpoint 中三組 `mtp.*` routed expert 後，總量約 146.62 GiB。
- 從官方 checkpoint 大小扣除這些 routed expert 後，剩餘資料約 8.79 GiB。此數字包含共同權重、DSpark 非 routed 權重與檔案資料。
- 若主模型每一層的六個 expert 都是 cache miss，SSD 每個 token 要讀約 3.21 GiB。此數字不含 DSpark。

因此，SSD 串流可以大幅降低記憶體需求。SSD 串流仍會是主要速度限制。

TurboFieldfare 在目標 M2 上量得單一冷 expert 的 `pread` 是 2.79 ms。其 untimed `F_RDADVISE` probe 是 3.61–3.78 GB/s。這些數字不保證 DeepSeek 的速度。若只用這些數字估計，全部 cache miss 的 I/O 上限約是 0.3–1.1 token/s。GPU 計算會降低速度。Expert cache hit 會提高速度。

來源與量測限制：[TurboFieldfare expert I/O experiments](https://github.com/drumih/turbo-fieldfare/blob/3249be40a33ea6560b35531c184609d7be67ac1a/docs/experiments/summaries/01-model-install-and-expert-io.md)

### Expert cache 記憶體

TurboFieldfare 預設為每層 16 個 slot。若 DeepSeek 沿用相同容量，43 層需要約 8.57 GiB slot 容量。即使每層只保留六個 slot，容量仍約 3.21 GiB。

DeepSeek port 不應直接複製「每層 16 slot」。Port 應使用較小的全域 page pool，或按層延遲配置 slot。Port 也應保留固定 stride、平行 `pread` 與 LFU 的概念。

TurboFieldfare 使用 `makeBuffer(bytesNoCopy:)` 包裝 page-aligned slot。Apple Silicon 使用 unified memory。CPU 與 GPU 可以共用此資源。實作仍須正確同步 CPU 寫入與 GPU 讀取。[TurboFieldfare streamer](https://github.com/drumih/turbo-fieldfare/blob/3249be40a33ea6560b35531c184609d7be67ac1a/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift)、[Apple `hasUnifiedMemory`](https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory)、[Apple `recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)

### KV cache

以下估算使用官方 reference code 的 BF16 cache shape。估算只含主模型 attention cache 與比例 4 indexer cache。Batch size 是 1。

| Context | 估算 KV cache |
| ---: | ---: |
| 4,096 | 32.25 MiB |
| 65,536 | 435.38 MiB |
| 1,048,576 | 6.72 GiB |

此估算不含 DSpark cache、activation、Metal scratch、allocator overhead 與 macOS file cache。FP8 KV 可以降低容量。FP8 KV 需要新的正確性測試。

### 實際容量判斷

- 官方完整 checkpoint 需要約 155.4 GiB。安裝時還需要 manifest、tokenizer 與工作空間。
- 直接保留完整 snapshot 再建立 repack 會接近兩倍空間。Port 應沿用 TurboFieldfare 的 HTTP range repack。
- 8 GB Mac 連估算的共同資料都放不下。
- 16 GB Mac 無法穩定容納共同資料、最小 expert cache、runtime 與作業系統。
- 24 GB Mac 可能執行短 context。Port 必須限制 cache 與 scratch。
- 32 GB 或更多記憶體較適合開發與量測。
- 1M context 再增加約 6.72 GiB BF16 KV。32 GB 仍需要嚴格控制 file cache 與 scratch。
- Apple 沒有為所有 Mac 保證一個固定的 SSD 持續讀取速度。Port 必須在目標機器量測。

## 可重用與必須重寫的部分

**可以重用的設計**

- 將共同權重常駐。
- 將 routed expert 依層寫入固定 stride 檔案。
- Router 完成後，CPU 讀取 expert ID。
- CPU 用有上限的平行 `pread` 填入 Metal 可見 slot。
- Shared expert 計算與 SSD 讀取重疊。
- 使用 LFU 與 recency tie-breaker。
- Prefill 使用 bounded chunk。
- 安裝器直接從固定 Hugging Face revision 做 range repack。

**必須重寫的部分**

- `ArchConfig` 與 checkpoint repacker。
- FP4 E2M1、FP8 E4M3 與 UE8M0 scale 核心。
- 6-of-256 router 與前三層 hash routing。
- SwiGLU routed expert 與 shared expert。
- CSA/HCA sparse attention、compressor 與 indexer。
- mHC、低秩 Q/O projection 與 YaRN。
- Untied embedding/output head。
- DeepSeek 專用提示編碼。
- DSpark speculative decoding。

TurboFieldfare 官方文件明確說明目前只支援固定的 Gemma 4 checkpoint。[TurboFieldfare README](https://github.com/drumih/turbo-fieldfare/blob/3249be40a33ea6560b35531c184609d7be67ac1a/README.md)

## 建議實作順序

1. 固定 0731 revision。建立主模型專用的 `.gturbo` v2 格式。
2. 先停用 DSpark。只轉換 43 層主模型。
3. 實作 FP4 routed expert 與 FP8 共同權重。
4. 實作兩種 router、MoE 與短 context attention。
5. 用官方 PyTorch reference code 做逐層 tensor parity 測試。
6. 收集真實 routing trace。量測 cache hit、讀取 bytes/token、RAM 與 token/s。
7. 再加入長 context cache。
8. 最後測試 DSpark 是否有淨加速。

## 未知風險

- 真實 expert cache hit rate 未知。
- 目標 Mac 的持續 SSD 讀取速度未知。
- Metal FP4/FP8 實作的實際速度未知。
- 共同權重是否需要再量化未知。
- 再量化後的品質損失未知。
- DSpark 在 SSD 串流條件下是否有淨加速未知。
- Apple Silicon 上的穩定 token/s 未經實機量測。

最終判斷是「技術上可行，但屬於新的 DeepSeek 專用執行器」。這不是小型適配工作。
