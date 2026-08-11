# 外部技術主張審核

日期：2026-08-10  
範圍：`docs/` 中涉及 DeepSeek-V4、safetensors、MLX、Metal、DSpark、效能與瓶頸的外部主張。  
目的：區分可由第一方來源證實的事實、本專案的量測結果、仍需重新量測的假設，以及已被目前程式碼取代的舊內容。

這份報告是文件整理的證據表。它不是新的效能結果，也不取代
[`docs/VALIDATION.md`](../docs/VALIDATION.md) 的量測記錄。外部來源只能證明 API、格式或論文中的結果。外部來源不能證明本專案在 M5 Pro、目前 SSD、目前 MLX fork 上的速度。

## 結論

1. Checkpoint 合約可信。程式固定使用 `DeepSeek-V4-Flash-0731` revision
   `7872f01b1d1fe23eabc4c98b48bffcef5a386062`，並在 repack 時檢查模型形狀。
2. safetensors 的 header、`data_offsets` 與 Range 讀取方式符合官方格式說明。HTTP 伺服器是否在每次請求正確回傳 `206` 和完整範圍，仍是本專案的執行時檢查，不是格式規格的保證。
3. 歷史文件 `research/archive/RUNTIME_RESEARCH_2026-08-07.md` 以舊的 upstream `mlx-lm` commit
   `254d153f...` 作為主要結論。目前 [`requirements.txt`](../requirements.txt) 使用
   `Blaizzy/mlx-lm@5c10538136b9038b9626c134612b08afc18d697a`。目前的 pinned fork 有
   `deepseek_v4.py`，所以「MLX-LM 沒有 DeepSeek-V4 模組」不能描述目前環境。
4. DSpark 不是「不支援」。目前 repacker 可選擇安裝 `mtp.*`，runtime 有 DSpark 路徑，但預設關閉。現有 M4 量測沒有證明淨加速，因此「可選、預設關閉、未證明加速」才是可信描述。
5. Apple MTLIO、MLX raw-pointer/no-copy、Metal capture 與其他模型的 prefetch 結果只能作為研究方向。它們不能直接寫成目前 runtime 的效能改善。
6. 本地速度、吞吐、hit rate、記憶體和 DSpark 結果必須附帶測試條件。沒有條件的估計百分比不能放在結果段落。

## 分類規則

| 標記 | 意義 |
| --- | --- |
| 外部事實 | 可由官方 API、官方模型檔案、官方格式說明或第一方論文直接支持。 |
| 本專案證據 | 只能由本 repo 的程式碼、測試或量測記錄支持。 |
| 需重新量測 | 外部來源沒有足夠資訊，或結果依硬體、revision、cache 狀態而變。 |
| 已過時 | 文件的描述與目前程式碼、依賴版本或安裝流程不一致。 |
| 研究假設 | 可以保留作為實驗方向，但不可寫成已完成或預期一定達成的結果。 |

## 證據表

| 主題 | 外部事實 | 目前專案狀態 | 文件處理 |
| --- | --- | --- | --- |
| Checkpoint 身分 | 官方 Hugging Face 固定 revision 的 `config.json`、`model.safetensors.index.json` 和 `inference/config.json` 定義模型與 DSpark 合約。見 [config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)、[index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json)、[inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json)。 | [`Model.swift`](../Sources/DeepSeekRepack/Model.swift#L8-L23) 固定 model ID、revision、43 層、256 experts、每 token 6 experts、3 個 hash 層和 1,048,576 context；repack 會拒絕不合約的 shape。 | 保留為外部事實加本地驗證。每個 benchmark 都要記錄 revision。 |
| DeepSeek 模型規模與量化 | 官方 model card 說明 284B total、13B active、1M context，以及 FP4 expert 和 FP8 其他權重。見 [官方 model page](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 和 [0731 model page](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)。 | 本地 contract 也檢查 `expert_dtype=fp4`、hidden size 4096、expert intermediate size 2048 和權重 shape。 | 把「模型規格」和「本地速度」分開。規格不代表本機能在 1M context 執行。 |
| safetensors 格式 | 官方文件定義前 8 bytes 為 little-endian header length，接著是 JSON header；每個 tensor 的 `data_offsets` 以 data buffer 起點為基準。見 [Hugging Face metadata parsing](https://huggingface.co/docs/safetensors/metadata_parsing) 和 [safetensors format source](https://github.com/huggingface/safetensors)。 | [`Checkpoint.swift`](../Sources/DeepSeekRepack/Checkpoint.swift#L27-L41) 先讀 header，再讀 tensor 範圍，並要求 HTTP 206 和精確 byte 數。 | 可以寫「實作符合格式」。不要寫「Hugging Face 一定支援任意 Range」；保留 206、長度、checksum 的 runtime 驗證。 |
| DSpark checkpoint | 官方 `inference/config.json` 和 reference code 定義 3 層、block size 5、target layers 40–42、Markov rank 256；DSpark weights 使用 `mtp.*` namespace。見 [reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py) 和 [official encoder/inference files](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference)。 | [`Checkpoint.swift`](../Sources/DeepSeekRepack/Checkpoint.swift#L67-L95) 預設可建立含 DSpark 的 plan；`installDSpark` 可把 DSpark 加入既有 installed model。`runtime/deepseek_v4_ssd/dspark.py` 有 loading、draft 和 verification。 | 把「mtp 排除、DSpark 不支援」標為已過時。正確狀態是「DSpark 可選安裝與執行，預設關閉；M4 尚無淨加速證據」。 |
| MLX-LM 依賴來源 | upstream 在 `254d153f...` 沒有 `deepseek_v4.py`，只支持該舊 commit 的觀察。這不能代表目前所有 MLX-LM 版本。 | [`requirements.txt`](../requirements.txt#L1-L6) 固定 `mlx==0.32.0` 和非 upstream 的 `Blaizzy/mlx-lm@5c105381...`；目前 pinned fork 有 [deepseek_v4.py](https://github.com/Blaizzy/mlx-lm/blob/5c10538136b9038b9626c134612b08afc18d697a/mlx_lm/models/deepseek_v4.py)。 | 歷史 `RUNTIME_RESEARCH` 的 `254d153f` 段落只代表當時的 upstream 檢查。目前 runtime 的來源應以 requirements 和 fork commit 為準。不要把 fork 稱為官方 upstream。 |
| MLX lazy evaluation | MLX graph 在 `eval` 前不會完成計算；`eval` 有固定 dispatch/synchronization 成本，列印或轉 NumPy 也可能觸發 evaluation。見 [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)。 | runtime 的計時若沒有明確 `mx.eval` 邊界，可能把路由、讀取和同步成本合在一起。 | 效能文件必須說明計時邊界，至少分開 host wall time、I/O time、compute synchronization time。不要把 Python 函式返回時間直接稱為 GPU kernel 時間。 |
| `gather_qmm` | 官方 MLX API 支援 `lhs_indices`、`rhs_indices` 和 `sorted_indices` 等參數；這是 API 能力，不是每個 shape 都走最快 kernel。見 [`gather_qmm` API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_qmm.html)。 | 本地 routed expert path 使用 `mx.gather_qmm`、packed/strided expert view 和排序 hint。 | 「可以呼叫 API」是外部事實；「排序或 strided view 變快」必須以本機 paired benchmark 和輸出相等測試支持。 |
| MLX unified memory 與 stream | Apple silicon 的 CPU/GPU 共用 unified memory。MLX 會在 stream 間處理相依性。見 [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) 和 [MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)。 | [`expert_cache.py`](../runtime/deepseek_v4_ssd/expert_cache.py) 直接把讀取寫入 MLX 配置的 buffer。這是本專案的 lifetime、同步和 `preadv` 實作。 | unified memory 不等於任意外部 `MTLBuffer` 都能被 Python MLX zero-copy 使用。這個結論要靠 native bridge、pointer lifetime、race test 和端到端量測。 |
| MLX compile 與 custom Metal | `mx.compile` 可合併或融合部分 graph；第一次呼叫會建立 compiled graph。MLX 也提供 custom Metal extension 與 Metal kernel。見 [compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)、[extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)、[custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)。 | 本地是否減少 kernel、是否觸發重新編譯、是否改善總時間，取決於實際 shape、dtype、prompt 和 cache。 | 把「MLX 可提供機制」和「本專案得到 X% 改善」分開。每個改善都要保存 warm-up、shape、compile 次數和 token hash。 |
| MLX memory cache | `mx.clear_cache()` 清理 MLX 的 cache memory；它不等同於釋放 active arrays 或降低整個 process RSS。見 [`clear_cache`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.clear_cache.html)。 | `model.py` 以 `mx.get_cache_memory()` 觸發清理。peak memory、active memory 和 cache memory 在本專案中是不同指標。 | 文件不可用單一「記憶體」數字混合三者。記錄指標名稱、取樣時間和是否包含 OS page cache。 |
| Apple MTLIO | Metal 提供 `MTLIOCommandQueue`、I/O command buffer、file handle、shared event 和 cancel，用於把檔案載入 GPU resource。見 [resource loading](https://developer.apple.com/documentation/metal/resource-loading)、[MTLIOCommandQueue](https://developer.apple.com/documentation/metal/mtliocommandqueue) 和 [work submission](https://developer.apple.com/documentation/metal/work-submission)。 | 目前 runtime 使用 Swift/Python SSD path，沒有完成 MTLIO 到 MLX routed kernel 的 bridge。 | MTLIO 只能列為 prototype 方向。Apple API 沒有保證外部 SSD、MLX array 包裝、zero-copy 或端到端速度會更好。 |
| Metal profiling | Xcode Metal debugger、Metal System Trace 和 GPU counters 可查看 CPU/GPU、記憶體和 command timing。見 [Metal developer workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows)、[GPU counters](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers) 和 [MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)。 | 本地 4K/8K 和 SSD 表是 runtime validation，不是 GPU counter profile。 | capture 只用於定位瓶頸。正式速度比較必須關閉 capture，並分開 cold page cache、warm page cache、warm compile。Xcode capture 有少量 CPU 影響，不能直接當 production time。 |
| DeepSeek heterogeneous/on-disk KV | DeepSeek-V4 paper 說明 sliding-window、未完成壓縮狀態、壓縮 KV，以及可把完整壓縮 block 存到 disk。見 [paper cache design](https://arxiv.org/html/2606.19348#S3.SS5)。 | 本地有 BF16/FP8 KV cache 和 prompt cache，但這些不等於已重現官方 on-disk KV 格式或 1M context。 | 把論文設計列為外部背景。只有完成格式、長 context、重啟和輸出 parity 測試後，才能宣稱本地支援。 |
| DSpark 60–85% | DSpark paper 的 60–85% 是 DeepSeek-V4 production serving、matched aggregate throughput 下的 per-user speed，不是 batch 1 Apple Silicon 或 SSD 的固定倍率。見 [DSpark paper](https://arxiv.org/html/2607.05147#S2)。 | [`VALIDATION.md`](../docs/VALIDATION.md) 記錄 DSpark 增加約 10.12 GiB weights，且目前沒有本 runtime 的淨加速證據。 | 保留 60–85% 作為外部背景，不能放進本專案預期結果。比較時要記錄 draft time、verify time、committed tokens（含 bonus/correction）、token parity 和 scheduler。 |
| DeepSpec reference | DeepSeek 的 [DeepSpec repository](https://github.com/deepseek-ai/DeepSpec) 是 reference evaluation/training stack，不是本專案 Apple runtime 的速度證明。 | 本地 DSpark path 是獨立整合。它不等於官方 vLLM/SGLang scheduler。 | 可引用演算法流程，不可把 reference evaluator 的數字當成本機結果。 |
| SpecMD、ProMoE、AcceptMoE | 這些論文研究不同模型、硬體或 router/eligibility policy。見 [Apple SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching)、[ProMoE](https://arxiv.org/html/2410.22134)、[AcceptMoE](https://arxiv.org/html/2608.02989)。 | 本專案是 DeepSeek-V4、Apple M5 Pro、SSD streaming。AcceptMoE 類方法可能改變 router 分布和輸出路徑。 | 數字只能作為研究假設。若改變 eligibility mask 或 router，必須標為 approximate mode，不能宣稱與目前 greedy path 等價。 |
| MLX no-copy raw pointer | [MLX PR #2875](https://github.com/ml-explore/mlx/pull/2875) 討論 C++ raw-pointer constructor；pull request 不是穩定的 Python API 合約。 | 本地 direct-`preadv` buffer 是目前 runtime 的實作，不代表任意 NumPy/MTLBuffer import 都 zero-copy。 | 不要用 PR 作為「支援 zero-copy」的證明。要以目前 0.32.0 headers、pointer lifetime test、memory copy counter 和端到端 benchmark 證明。 |
| 其他 MLX-LM issue | [Issue #1332](https://github.com/ml-explore/mlx-lm/issues/1332) 是 issue 報告，不是穩定 API 或所有版本的行為規格。 | runtime 有自己的 cache cleanup workaround。 | 引用時加上 issue、fork commit、測試日期和「未驗證 12K」等限制。不要把 issue 報告寫成官方保證。 |

## 本專案量測的可信邊界

目前唯一可直接當成本專案結果引用的數字，應來自有條件的本地記錄，例如：

- [`docs/VALIDATION.md`](../docs/VALIDATION.md)：目前 M5 Pro 的自動測試、完整 SHA-256、SSD microbenchmark 和端到端結果。
- [`docs/benchmarks/2026-08-10-m5-pro.json`](../docs/benchmarks/2026-08-10-m5-pro.json)：目前量測的機器可讀 artifact。
- [`docs/VALIDATION.md`](../docs/VALIDATION.md) 的歷史區段：4K/8K prefill、ready expert decode 和 DSpark 決策。歷史數字不是吞吐保證。

每筆新的效能結果至少要記錄：

1. 硬體、OS、Metal/MLX/MLX-LM 版本、fork commit 和 checkpoint revision。
2. prompt token 數、output token 數、batch、greedy/sampling、temperature 和停止條件。
3. worker、slot、prefill step、KV dtype、layer-major/batched path、DSpark 開關。
4. cold/warm compile、cold/warm OS page cache、prompt cache 狀態。
5. wall time、read time、bytes、hit/miss、active/cache/peak memory、CPU/GPU synchronization 邊界。
6. 完整輸出 token hash。比較兩個設定時，必須確認 committed token 數相同。

沒有這些條件時，`GiB/s`、`Tok/s`、`% improvement` 和「瓶頸已確認」都只能是初步觀察。

## 需要改寫的舊主張

### `RUNTIME_RESEARCH_2026-08-07.md`

- 把 `254d153f...` 標為歷史 upstream 檢查，不要用它代表目前依賴。
- 把「MTP excluded、DSpark not supported」改成目前程式碼的狀態：預設安裝可以包含 DSpark；`install-dspark` 可以補裝；runtime 有可選 DSpark；runtime 預設關閉 DSpark，且目前尚無淨加速證據。
- 保留官方 DeepSeek 規格和 MLX API 說明，但把本地測試限制放在同一段。

### `DSPARK_FIRST_PLAN_2026-08-08.md`

- 這份文件是歷史計畫。安裝 DSpark 的部分已由目前 `Checkpoint.swift`、`RepackPlanner.swift` 和 runtime 實作覆蓋。
- 可以保留驗證項目，但不要寫成「目前 installed model 必定沒有 DSpark」或「repacker 排除所有 `mtp.*`」。

### `EXPERT_STREAMING_RESEARCH_2026-08-09.md`

- `MTLIO` 保留為 native prototype，不要寫成已接入 MLX。
- 文件中的 `512 slots` 與目前 `RuntimeConfig.slots = 1_152` 不一致。每個 benchmark 應明確記錄實際 CLI/config，不能用研究建議值代替執行值。
- ProMoE、SpecMD 和 AcceptMoE 的結果要加上「其他模型/硬體」標記。

### `PREFILL_DECODE_RESEARCH_2026-08-08.md`、`RUNTIME_SPEED_RESEARCH_2026-08-07.md`、`PREFILL_DECODE_RUNTIME_OPTIMIZATION_2026-08-09.md`

- `mx.eval`、Metal capture、`gather_qmm` 和 custom Metal 是能力或量測方法，不是已確認的速度提升。
- 任何 `3–35%`、`80–97%`、`2–8%` 或從其他論文移入的百分比，都應移到「研究假設」表，不得與 `VALIDATION.md` 的本地結果混列。
- 歷史 baseline、optimized result、diagnostic smoke test 要分開。每張表要有日期、commit、條件和輸出 parity。

## 建議的文件來源順序

1. 外部規格：固定 revision 的 Hugging Face files、safetensors 官方格式、MLX 0.32.0 API、Apple Metal API、DeepSeek first-party papers。
2. 本地合約：`CONTEXT.md`、`Model.swift`、`requirements.txt`、manifest 和 CLI help。
3. 本地結果：`docs/VALIDATION.md` 加上可重跑的 command、日期和 artifact。
4. 研究筆記：保留推論和未驗證方向，但在標題或表格加上「研究假設」或「歷史」標記。

這樣可以避免用舊研究結論覆蓋目前程式碼，也可以避免把其他硬體的結果誤當成本專案的效能承諾。
