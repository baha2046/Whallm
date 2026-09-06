# Qwen3.8 Prefill／Decode：證據整理與新研究路線

日期：2026-09-06。狀態：A 第一輪 direct QSA 候選在完整模型 exact-token gate 被拒絕。

這份文件回答如何縮短使用者等待，包含現有 runtime 之外的新方向。
它是研究文件；候選已量測 component 與部分完整模型，但沒有通過採用門檻。
論文提供機制依據，不會使一個本機假設自動成立。
本文的實驗門檻是研究設計，不是論文宣稱的通用標準。

## 1. 本次查核範圍

以 [docs 入口](../docs/README.md)、[Qwen 合約](../docs/QWEN.md)、
[效能定義](../docs/PERFORMANCE.md)、[既有研究決策](../docs/RESEARCH.md)、
[Qwen Q0–Q4](QWEN_RUNTIME_OPTIMIZATION_2026-08-29.md) 和
[SSD 研究](SSD_STREAMING_BOTTLENECKS_2026-08-31.md) 為起點，核對 runtime、
本機已安裝 dependency、installed model manifest 和 benchmark JSON。
舊 DeepSeek／M2 Max 經驗只用來形成問題，不能代替 Qwen／M5 Pro 的結論。

初始文件查核沒有執行完整模型 inference；使用者選 A 後，第 7 節執行了有界實驗。
Production runtime 與 weights 均未改動，候選只存在獨立 research runner。
執行了既有 Qwen 相關 46 項測試與 block prompt-cache 6 項測試，全部通過。
測試包含小型 tensor／fixture，不是完整模型的數值或品質認證。
查核清單、來源 SHA-256、重新計算的 API medians 和測試結果保存在
[inventory JSON](artifacts/2026-09-06-qwen-research-inventory.json)。

| 項目 | 本次現場證據 |
| --- | --- |
| Source commit | `7dc9cf8f050c75def77c0563cd7b8ac03f2d8435`；開始查核時工作樹乾淨 |
| 執行主機 | Apple M2 Max、64 GiB、macOS 26.6.2／25G83 |
| 本機 dependency | MLX 0.32.0、mlx-lm 0.31.3；具體來源檔 hash 已保存 |
| Qwen checkpoint revision | `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` |
| 本機 manifest SHA-256 | `a71f38985d7b46919e4ba5abd5ca37f209c6864e6a51fe635e48cd45788326dc` |
| Installed model | 59 個檔案 size 全部符合 manifest；本次未重算約 127 GB weight files 的完整 SHA-256 |
| 歷史主要效能主機 | 既有 QSA／ANE artifact 標示 M5 Pro；不能與這次 M2 Max 合併比較 |

專案包含 Swift repack／安裝驗證工具、SwiftUI App、Python／MLX runtime、
OpenAI-compatible server、Metal-visible expert slot cache、prompt persistence，
以及 default-off MTP。API model ID 雖帶 `fp8`，installed routed experts 實際是
MXFP4；不能把「與目前 runtime 相同」寫成「與官方 FP8 模型相同」。

## 2. 現行資料路徑與可定位的成本

主模型有 48 層，其中 36 層 Gated DeltaNet、12 層 QSA。每層有 512 個 routed
experts，每 token 選 10 個；hidden size 2,560、expert intermediate size 640。
QSA 使用 24 query heads／2 KV heads、head dimension 256、2,048-token indexer
budget、4-token micro-block。這些數值已與本機 config／manifest 核對。

| 路徑 | 現行程式行為 | 研究意義；不是已量到的瓶頸占比 |
| --- | --- | --- |
| Prefill expert loading | [generation.py](../runtime/deepseek_v4_ssd/generation.py) 的 `_qwen_layer_major_prefill` 每層載入完整 expert layer，再跑 prompt chunks | 一個完整 pass 的 expert payload 約 64.17 GB；實際 SSD bytes 仍受 page cache 影響 |
| Decode experts | [qwen4_exp.py](../runtime/deepseek_v4_ssd/qwen4_exp.py) 的 `StreamingExperts` 將 indices materialize 到 NumPy，`get_many` 等待 missing experts，再分 expert 計算 | 應區分必要 read wait、CPU/GPU 同步、QMM 與 memory traffic |
| Gated DeltaNet | 載入 mlx-lm `qwen3_5.GatedDeltaNet`；`gated_delta_update` 選 Metal kernel，但 kernel 內有 `for (int t = 0; t < T; ++t)` | 已融合、但時間軸仍循序；不是 Python 逐 token 迴圈 |
| QSA | `_bounded_attention` 已按兩個 KV groups 計算；每 4 個 queries 做 gather、scores、softmax、value reduction | 還有 intermediates；不可重做已淘汰的 KV head expansion |
| QSA indexer | 每次呼叫重新 pool 完整歷史 raw index keys，做 norm／RoPE；短 context 分支仍在前面建立這些 graph | 新候選是已完成 block 的增量 cache；lazy graph 是否真正執行無用工作要由 trace 確認 |
| N-gram／PLE | 第 2 個 model layer 使用 token IDs 計算 16 個 row IDs，再同步 memmap gather／NumPy FP8 decode | Address 不依賴該層 hidden，可研究提前讀取；不需要推測 router |
| LM head／common tensors | `lm_head.weight` 是 BF16 `[248320,2560]`，1,271,398,400 bytes；common.bin 約 9.90 GB | Common tensor／大 vocabulary head 也是 Decode 候選，不能只看 expert SSD |
| ANE | [ane_prefill.py](../runtime/deepseek_v4_ssd/ane_prefill.py) 已先 materialize FP16 input，再 `mx.async_eval` GPU 子矩陣並同步呼叫 ANE | 預設 25% channels，僅固定 1,024-token QSA projection；已有 materialization，不能把論文的同類修復當成本專案缺漏 |
| Prompt cache | format 5，包含 Qwen recurrent state／convolution state 的 snapshot isolation | 多輪工作流可省重算；不能只快取標準 Transformer KV |

由 manifest 直接相乘，48 × 10 × 2,611,200 = **1,253,376,000 bytes** 是
「每個 target step、所有 selected experts 都 miss」時的 expert blob payload。
它不是實測每 token SSD 流量，也不含 common tensor、cache state 與 N-gram。
N-gram 每 token 的 requested row payload 則是 16 × 160 = **2,560 bytes**；
51.20 GB 是整個 table 大小，不是每 token 讀取量。OS page granularity 與解碼暫存
可能放大成本，必須量測，不能因檔案大就判定它主宰 Decode。

## 3. 成功、失敗與可遷移的經驗

下表均為既有 artifact 的結果，不是本次重新執行的效能數字。

| 嘗試 | 證據與狀態 | 下一輪要汲取的經驗 |
| --- | --- | --- |
| QSA Grouped-KV | [採用 quick gate](../docs/benchmarks/2026-09-02-qwen-qsa-grouped-kv-adoption-quick-gate-m5-pro.json)：TTFT 85.36 → 44.88 s；102.00 vs 53.63 tok/s；4 次首 token 相同 | 減少重複 memory traffic 有實質價值。但只覆蓋單一 4,577-token prompt／1 output token，非完整品質或 Decode gate |
| Prompt cache | format 5 修復 mutable list alias；[Qwen 驗證](../docs/QWEN.md) 的 cold／same-process／restart 輸出一致 | 快取必須擁有完整、不可被後續 generation 改寫的 state；每次節省要扣掉 load／snapshot／serialize 成本 |
| 短 prompt memory 修復 | [Qwen 驗證](../docs/QWEN.md)：peak 約 74.07 → 13.05 GB，token hash 相同 | Lazy graph lifetime 可以比模型算子本身更重要 |
| Selected Prefill／threshold | [拒絕紀錄](../docs/benchmarks/2026-08-29-qwen-prefill-threshold-rejection-m5-pro.json)、[完整 Q3](QWEN_RUNTIME_OPTIMIZATION_2026-08-29.md)：部分 workload 很快，但技術中文／短 math token diverge | 數學上等價的 shape／累加順序變更，仍可能改變 route 與完整輸出；單元測試不是 full-model parity |
| Qwen ready expert | [Q1](QWEN_RUNTIME_OPTIMIZATION_2026-08-29.md)：Decode +3.63%、request -2.34%，未達 5% | Removing a barrier 不必然是主要收益來源；需要新設計才重開 |
| LFU decay | [離線 simulation](../docs/benchmarks/2026-08-29-qwen-q2-lfu-simulation-m5-pro.json)：候選 miss 最少 +62.5% | 先在 trace 做低成本淘汰；不要先接完整模型再找收益 |
| Next-layer prefetch | [deadline gate](../docs/benchmarks/2026-09-01-qwen-next-layer-prefetch-gate-m5-pro.json)：暴露 read wait -27.41%，ready rate 0.52%；default-off | Read wait 降低不是端到端通過；trace observer 時間不能當正式 benchmark |
| LZ4／LZFSE | [壓縮 gate](../docs/benchmarks/2026-09-01-expert-blob-compression-m5-pro.json)：Qwen LZ4 儲存只省 2.47%，與 Metal 同跑反而慢 5.08% | 搬更少 bytes 仍可能更慢；一般無損 codec 不值得原樣重試 |
| MTP block | [formal stop](../docs/benchmarks/2026-08-30-qwen-mtp-block-formal-stop-code4k64-m5-pro.json)：Decode +65.07%，request +7.34%、P95 +153.13% | 看 acceptance／tok/s 會漏掉啟動成本與 burst stalls |
| MTP bounded Prefill | [formal stop](../docs/benchmarks/2026-08-30-qwen-mtp-bounded-prefill-formal-stop-code4k64-m5-pro.json)：接受率 96.30%，Decode +40.01%，request 只 -2.48%、P95 +212.95% | Drafter 準確仍不夠；需要改變 verifier critical path，而非只縮 draft block |
| Private ANE | [exploratory gate](../docs/benchmarks/2026-09-02-qwen-private-ane-prefill-exploratory-m5-pro.json)：median Prefill +8.34%，但反向順序近乎無改善 | 現在預設啟用是產品狀態；速度結論仍 inconclusive，不能當新優化的已知收益 |
| DeepSeek approximate | [Phase 6E](../docs/benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json)：Decode +8.37%，10 pairs 在已測 outputs 相同 | 只支持該 DeepSeek candidate；不能推論 Qwen top-10 可以直接 drop 一個 expert |

DeepSeek 的 MTLIO→MLX handoff、staged expert loading、frozen router transfer 等停止
條件仍有參考價值，但不是 Qwen 所有同類架構的永久禁令。重新打開必須列出
「新機制／新 prerequisite／新 evidence」，而不是再跑同一份候選。

### 最新 API snapshot 的正確讀法

從 [2026-09-04 API 原始 runs](../docs/benchmarks/2026-09-04-132210-api-qwen3-8-flash-next-fp8.json)
重新計算各 input size 的中位數，每組 3 個不同 prompt、64 output tokens：

| Input tokens | TTFT s | Prefill tok/s | Decode tok/s | Client wall s |
| --- | ---: | ---: | ---: | ---: |
| 1,024 | 15.173 | 67.488 | 10.359 | 21.412 |
| 2,048 | 24.291 | 84.312 | 9.782 | 30.930 |
| 8,192 | 74.725 | 109.628 | 9.917 | 81.210 |
| 16,384 | 147.187 | 111.315 | 9.444 | 154.640 |

這份 snapshot 的 commit 是 `b25ae07e...` 加 dirty diff，hardware model 是 `Mac17,8`。
它不是今天 commit／M2 Max 的 baseline，也不是配對 A/B。所有 runs 回報 reuse=0，
但 prompt cache／OS page cache 未清除；API 只記 output **text** hash，沒有 token IDs。
現行 summary 的 3-sample nearest-rank P95 等於 maximum，throughput 的高 P95 也不是
尾端慢速保障。本表使用原始 runs 的 median，避免將 maximum 當 typical speed。
8K／16K 有 API 執行證據，不等於長 context 品質或 token-ID parity 已驗證。

## 4. 論文依據與新假設

以下每個 H 編號都是待驗假設；「新」指本次查閱的 active research／code 未見
同一候選的完成 gate，不宣稱整個專案歷史或全世界從未研究過。

### H1：Gated DeltaNet 分塊平行 Prefill

**機制依據：** [Gated Delta Networks](https://arxiv.org/html/2412.06464v3) §3.3／Appendix A
用擴充 WY representation 得到 chunkwise parallel form。
[Qwen3.8 技術報告](https://arxiv.org/html/2608.30320) §2.1.1 也採 fused GDN kernels。
將同一 recurrence 的 chunkwise forward 移植至本機 Prefill，是由論文推導出的候選；
論文的 training／NVIDIA 成績不是 Apple Metal 的 Prefill 成績。

**本機缺口：** 現行 Metal kernel 沿 T 循序迭代，36 層都可能涉及此成本。
先量 GDN recurrence 與 projections 的獨立占比；不能用「36/48 層」當「75% 時間」。
只改 GDN 內部數學排程，先保持外部 1,024-token chunk、MoE 路線、gate、norm 不變。

**最小實驗：** 固定 BF16 inputs／FP32 state，以 T=1、128、256、1,024，
Hk=16、Hv=48、Dk=Dv=128，檢查 outputs 與最後 state、非零初始 state、mask、
跨 chunk continuation。內部 tile 先測 32／64，不做大規模 sweep。
**停止：** trace 顯示可改善 section <5% request，或 full-model greedy hash 不同，
或額外 buffer 抹去收益。不同累加順序的誤差要診斷，不能偷偷降低 exact gate。

### H2：直接讀 selected K/V 的 fused QSA

**機制依據：** [FlashAttention](https://arxiv.org/abs/2205.14135) 的 tiling／online softmax
減少 intermediate memory traffic；[FlashInfer](https://arxiv.org/abs/2501.01005)
提供可依 attention pattern／KV layout 設計 kernel 的系統依據。
候選是把現有 QSA 的 selected-index gather、mask、softmax 與 value reduction 融合，
不改 indexer budget、selection set 或 query heads。

**最小實驗：** 先固定已產生的 selected IDs，測 2 KV groups、24 query heads、
256 head dimension、2,048 selected tokens，再加入 tail／causal mask。
**停止：** correctness／tie／NaN 任一失敗，或完整 TTFT 不達 gate。
CUDA library 不是可直接匯入的 Metal backend；要比較目前 Grouped-KV，不能比較已移除
的 24-head KV expansion。這條路線也可能影響 Decode，必須另測 T=1。

### H3：QSA 增量 pooled-key cache，避免 Decode 重算歷史

**機制依據：** [Qwen3.8 報告](https://arxiv.org/html/2608.30320) §2.1.2 定義固定大小
micro-block compression。推論：已完成的 4-token block 內容與位置不再變動，因此
其 pooled key／norm／RoPE 可快取，只追加新完成的 block；每個 query 的 score／top-k
仍重新計算。這不是沿用前一 token 的 attention selection。

**最小實驗：** 2,047／2,048／2,049 context 邊界、4-token 邊界、追加、trim／rollback、
prompt-cache restart；cache key 包含 position、model 與算法版本。
**停止：** 新舊 pooled keys 不一致、舊 snapshot 被改寫、MTP branch state 污染，
或 pooled-key work 實測太小。預期 cache 容量及收益都要實測。

### H4：N-gram deterministic lookahead＋有限 row cache

**機制依據：** [Engram](https://arxiv.org/html/2601.07372) §2.5／§6.4 利用 token-based
deterministic addressing 提前搬移；[Qwen3.8 報告](https://arxiv.org/html/2608.30320)
§2.3 同樣討論 host-resident N-gram。候選在 prompt tokenization 後提前算 rows，
Decode 則在 token 已確定後、進入 PLE 前提交 row load，與前面 layer work 重疊。

**最小實驗：** 分解 hash CPU、page faults、FP8 decode、NumPy→MLX handoff；比較同步
基準與有上限的 raw-row cache／prefetch，包含 EOS 與跨 chunk context。
**停止：** 暴露成本 <5% 且無合理端到端收益，或 row cache 搶走 expert working set。
SSD 隨機小讀與論文的 host DRAM prefetch 不同；不能套用其 negligible-overhead 結論。

### H5：工作流 prefix reuse，改善多輪 Prefill

**機制依據：** [SGLang](https://arxiv.org/abs/2312.07104) 的 RadixAttention 支持跨呼叫
重用共有 prefix。此專案已有 content-addressed blocks，因此新問題是實際 agent
對話中 snapshot boundary／eviction／load cost 是否阻礙可用 reuse。

**最小實驗：** 固定 tool definitions、保持原 message 順序的多輪軌跡，分別測
同 process、restart、短 suffix、長 suffix；記 client TTFT 與 cache load/write。
**停止：** 沒有真實共同 prefix，或載入／snapshot 比重算更慢。
不重排或刪改使用者內容來人為增加 hit rate；這條路線不提高全新 prompt 的純計算速度。

### H6：Prefill 靜態算子的 CPU／GPU／ANE 分工

**機制依據：** [FusionML](https://arxiv.org/abs/2607.22785) 研究 MLX lazy dependency
造成的 co-execution serialization；[NPUMoE](https://arxiv.org/html/2604.18788)
用 static capacity／grouped execution 攤薄 NPU launch cost。

**與現行路線的差別：** 不只 sweep 現有 25% q_proj ratio；可評估 common dense
projections／shared expert 的分組執行、CPU+GPU，以及較粗粒度的 ANE 任務。
現有 ANE wrapper 已 materialize input，須先量 copy／FP16 conversion／launch／join。

**最小實驗：** 同 dtype 的完整 lazy dependency chain、單算子與連續 12 層、
持續 GPU／expert I/O contention、初始化成本與 active steady state 分開。
**停止：** overlap 無法成立、shared bandwidth 飽和，或額外 resident weights／padding
抵銷收益。NPUMoE 的容量配置可能 drop overflow tokens；本專案 exact 路線必須 fallback
計算所有 selected experts，不能照搬 drop policy。
兩篇都是本次查到的 preprint；其工作負載與 Qwen3.8 SSD runtime 不同。

### H7：改造 speculative verifier／排程，而非再縮 draft block

**機制依據：** [SpecExec](https://papers.neurips.cc/paper_files/paper/2024/file/1d91d5689e251d27993a3c2182dddcf7-Paper-Conference.pdf)
研究 offloading 下攤薄 target pass；[AMUSD](https://arxiv.org/abs/2410.17375)
研究不同 device 上 draft／verify 重疊。分布正確性的依據是
[Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) 的 acceptance／correction。

**本機候選：** expert union 的 verifier 排程、可獨立運作的小 drafter、或
[Qwen3.8 報告 §2.1.2](https://arxiv.org/html/2608.30320) 提到的 draft-only
QSA index reuse；先拆成本，不直接新增長 tree。現有 MTP 依賴 target hidden，不能
假設它能像獨立小模型般無限制領先 target。GDN／PLE 需要各 branch 獨立 state，
也不能只改成 tree attention mask 就聲稱 tree verifier 完成。

**最小實驗：** 紀錄每輪 draft／verify／replay／committed token、API token arrival、
expert union 與 physical-I/O proxy。先計算理想 draft cost=0 的收益上限。
既有 bounded gate 的一個 run，draft 約 0.361 s，而 request 約 78.85 s；
即使刪掉全部 draft 工作，該 run 也僅省約 0.46% request，這是算術上限，不是新量測。
**停止：** 無法改善 verifier critical path、P95 超標、或 target distribution 不正確。
不能提前把未驗證 token 發給使用者來掩蓋延遲。

### H8：Common tensor／LM head 的選擇性量化

**機制依據：** [AWQ](https://arxiv.org/abs/2306.00978) 以 activation calibration 控制
weight-only quantization 誤差，並配合實際 packed kernels 降低成本。
新候選先量 1.27 GB BF16 LM head 與其餘 dense projections，挑成本最大的部分量化，
而不是再把已經 MXFP4 的 experts 做一次一般壓縮。

**最小實驗：** 固定校準資料與未看過的品質集，先單層 logits／top-k agreement，再測
中文、code、math、tool arguments、long-context retrieval／termination。
**停止：** 沒有相應高效 Metal kernel、dequantization 成本抵銷收益，或品質不合格。
這會改變 target 數值，需要獨立 model/cache identity，屬使用者要選擇的品質取捨。

### H9：Expert 多精度副本／更小常駐 working set

**機制依據：** [HOBBIT](https://arxiv.org/html/2411.01433) 把較不重要的 miss experts
換成低精度版本以降低 loading；[LLM in a Flash](https://arxiv.org/abs/2312.11514)
說明 flash-aware 搬移粒度的重要性。這是有損表示／runtime redesign，不是重試 LZ4。

**最小實驗：** 先對 checkpoint-derived expert sample 建低精度獨立 sidecar，測
kernel support、read+decode+compute、cache residency 與品質；不要覆蓋 canonical blob。
精度／scale metadata／alignment 必須計入實際 bytes。64 GiB 機器不能只看 raw expert
size 就宣稱「全部放得下」，還有 common tensors、OS、KV／SSM、scratch 與 N-gram pages。
**停止：** 無合理品質與端到端收益，或產生額外版本／cache consistency 問題。

### H10：讓 routing 提前可知，或把小計算送到資料旁

**機制依據：** [Pre-gated MoE](https://arxiv.org/abs/2308.12066) 改變 gating 的層間
依賴來提前取得 experts；[MoE-Infinity](https://arxiv.org/html/2401.14361) 使用
activation-aware cache／prefetch；[Fiddler](https://arxiv.org/abs/2402.07033) 使用
CPU 計算以減少 CPU↔GPU weight migration。

**兩種不能混淆的候選：** predictor 只提示 prefetch、miss 時執行原 router，可保留
target 語意；真正改 router／pre-gate 是需要訓練及品質驗證的模型改造。
Fiddler 避免 PCIe 搬移的前提不能原樣套到 Apple unified memory；CPU 也不能省掉
尚未在 RAM 的 SSD reads。
**最小實驗：** 先做 Qwen trace oracle／deadline 可行性與 runtime 無關的 frozen
predictor gate；CPU 路線先做含 GPU contention 的工作量比較。
**停止：** oracle 上限不足、prefetch 用錯 bytes 或太晚、共享頻寬使 CPU+GPU 更慢。
若需要訓練，必須先界定資料、checkpoint identity、成本與品質門檻，再由使用者決定。

## 5. 研究排序與實驗合約

推薦先選 A：優先量 H1 的 GDN recurrence、H3 的 Decode indexer，以及 H2 的
QSA intermediates。H4／H5 是低成本的不同方向；H6／H7 屬更大的 runtime 架構
投資；H8／H9／H10 的模型改造部分有新的品質取捨。
這個排序依據是可定位的本機程式缺口與既有失敗，沒有承諾任何倍數。

下一輪開始時執行以下共同 gate：

1. **固定現行 control。** 保存 commit＋diff、dependency hashes、manifest hash、
   機器／功耗模式與所有生效設定。M2 Max exploratory 和 M5 Pro comparison 分開。
   Production control 保留現在的 ANE 預設；需要做 attribution 時另設 GPU-only control，
   不把兩種 control 混用。新 cache/state implementation 必須隔離 persistent cache。
2. **低成本成本分解。** Prefill-only 1 output token；Decode 先 128-input／64-output
   screening，再 4K／256。QSA 需包含 2,048 邊界；較長 context 先做有界 correctness
   診斷。CPU wall boundary、GPU interval、I/O worker timer不能直接加總，須找 critical path。
   Trace run 只做診斷，正式 timing 關閉 observer。
3. **單一候選 component gate。** 相同 inputs、dtype、layout、初始 cache／state；
   每候選先設 output／final-state tolerance 與 exact token gate，再量時間。
   algebraic equivalence 不保證浮點 bit-identical；偏離時保存失敗、不更改門檻追認。
4. **代表性完整模型 gate。** code、mixed_math、repeated、tool_like、zh_technical；
   經 Qwen tokenizer 確認實際 token 數，不以 prompt 檔名中的 4096 代替。
   配對 ABBA，再反向 BAAB wave；每個 workload 的 4K／256 用完整 generated token IDs，
   同時保留短 prompt 與 warm/restart prefix 測試。長 context 品質另列，不以速度測試替代。
5. **依預先門檻決定。** exact 候選 prompt/output token hash 相同；主要指標 median
   至少改善 5%，Prefill 候選 Decode regression ≤5%，Decode 候選 TTFT regression ≤5%，
   token P95 regression ≤10%，peak active memory regression ≤15%；expert bytes/生成 token
   regression ≤5%，除非 request 至少改善 10%。同報 client wall、process RSS、swap，
   不用整體平均掩蓋單一 workload／短 prompt regression。
   中位改善不到 5% 視為 inconclusive；明確失敗則停止。
6. **有損路線另訂品質合約。** 不能拿 exact token hash 當唯一品質指標，也不能把單一
   calibration set 的高 agreement 當能力保持。先停在未採用 sidecar／研究 artifact，
   待使用者決定允許的品質差異後，才有正式通過／不通過結論。

時間解釋採 `T_request = T_first + T_remaining_decode`。若一個可完全移除的 section
只有 request 的 f，最多也只省 f 的時間；有 overlap 時必須改用 critical-path 的 f。
這是成本分解的算術限制，不能從 isolated kernel 倍數直接推導產品倍數。
訓練、額外模型／硬體購買、部署預設值改動均不屬於這次文件研究的成果。

## 6. 需要使用者決定的三條路線

| 選項 | 下一個有界交付 | 優點 | 代價／trade-off |
| --- | --- | --- | --- |
| **A．保持模型，重做計算與資料路徑（推薦）** | H1／H3 成本分解＋第一個有證據的 component candidate；H2／H4／H5 按 gate 排序 | 不改 weights／top-10；可同時追 Prefill、Decode；最容易與現行結果比較 | exact greedy gate 嚴格，平行計算的浮點差異可能令候選停止；收益未知 |
| **B．重做硬體分工與推測解碼** | H6 contention／dispatch gate，H7 verifier critical-path 與獨立 drafter 可行性 | 探索 CPU／GPU／ANE 和較大排程變化，不局限既有 kernel | 開發與 cache ownership 複雜；共用頻寬、ANE 精度／啟動、P95 burst 可能抵銷收益；若需新 drafter 再決策 |
| **C．允許獨立量化／模型改造研究** | H8 LM-head／common tensor 成本與量化可行性，H9 expert sidecar；訓練型 H10 先做方案 | 可以從根本減少權重搬移與記憶體需求，拓展 Decode 的可能路線 | 不保證與現在輸出相同，需要品質資料與新 model identity；品質容忍度和訓練預算須再決定 |

使用者已選 A，授權進入保持 weights／top-10 的計算與資料路徑研究。
B／C 未被選取；若 A 出現需要改變品質門檻或硬體目標的分岔，再提供三個選項。

## 7. A 第一輪預先登記

執行主機固定為 M2 Max。本輪先保留 CLI defaults 的 1,152 slots、ANE requested=true、
ratio=0.25；並記錄實際 ANE active／fallback。App 4,096 slots 是另一組設定，不能混用。
Greedy temperature=0、top_p=1、top_k=0，persistent prompt cache 關閉，OS cache not purged。
先執行 normal code 4K／64 control，再用
[`qwen_a_probe.py`](qwen_a_probe.py) 在另一 fresh process 做相同 workload 的 section probe。
所有 attempts 保存在 `scratch/qwen-a-2026-09-06/`，不覆寫失敗檔案。

Probe 在輸入／輸出加同步，保存 GDN recurrence 與 QSA 實際 inputs；它改變 lazy graph
與 overlap，只是診斷，不能當正式效能結果。Probe 的 outputs 需與 normal control 比較。
若候選 section 在診斷中明顯不足 5%，不先寫完整 runtime prototype，改測另一候選。
Component gate 在任何執行前固定：BF16 output `atol=0.03125, rtol=0.01`、
FP32 recurrent state `atol=0.001, rtol=0.001`，全部有限；這只是早期數值 screening，
之後完整模型仍要求 exact greedy token IDs。QSA pooled keys 的 cache-only 變更要求
byte-exact。通過後才可做 32／64 tile 的 microbenchmark 與 full-model stop gate。

### 第一輪 attribution 與 H2 reopening

Normal code control（4,577 prompt／64 output）TTFT=61.189 s、request=71.123 s；
另一次 synchronized probe 的 64 個 tokens 與 control 相同，ANE active、48 evaluations、
0 fallbacks。Probe Prefill section sums：GDN recurrence 0.864 s、QSA bounded attention
10.284 s、N-gram lookup 0.139 s。這不是可直接加總的正式 critical-path 分析。
GDN recurrence 與 N-gram 此 workload 未達 5% 入場線，暫不實作 H1／H4。

固定真實 inputs 的 H3 ideal pooled-key cache oracle 未顯示可用改善；只替換成
MLX fused SDPA 的 Prefill component 改善約 0.32%，停止該候選。
這不否定融合 selected-index loading 的另一種 kernel。

H2 新候選先測 post-adoption Grouped-KV retile=16／64。舊 QSA chunk=8 quick gate
早於 Grouped-KV 採用，仍有 12 倍 KV expansion；新候選保留兩個 KV heads，
因此是不同 memory layout 下的 reopening，並非重跑同一個失敗候選。
這裡的 16／64 是 QSA query tile；前面的 32／64 是未進入實作的 GDN 內部 tile。
先以真實 1,024-query／4,096-key capture 比較 tensor、peak memory 與 ABBA component time；
只有數值通過且有足夠收益才進入完整模型。QSA indexer budget／top-k／mask 不變。

Retile=64 的 component peak 約 2.70 GB，control 約 0.56 GB，僅改善約 8%；
收益相對整體成本太小、額外記憶體過高，因此不進 full-model。
H2 隨後實作獨立的 direct selected-row QK／PV Metal kernels：每組 KV head 共用
讀入值，同時計算該組 query heads；保留 BF16 score rounding、FP32 softmax、
BF16 probability 及 output。它保留 indexer／mask／selected set，減少中間 gathered K/V。
這是本專案依 IO-aware attention 原則撰寫的候選，不是移植外部論文的速度數字。

真實 Prefill capture 的 component median 0.22286 → 0.14212 s（-36.23%），
最大 absolute error=0.0078125，通過預先 tolerance，但不 tensor-exact。
Decode capture tensor-exact，component 約 -16.13%；整體 Decode 收益仍未知。
第一個完整模型候選只替換 Prefill，保持單 token Decode 原路徑，先做 code／64
output pilot。2 個 boundary tests 通過（包含 dense／sparse budget、tail、future mask）；
這些測試不能代替 full-model token parity。Runtime defaults 尚未改動。

Code／64 pilot 的 prompt 與 64 個 output tokens 均與 normal control 相同，
ANE active、48 evaluations、0 fallbacks。單次 TTFT 61.189 → 56.680 s，
request 71.123 → 66.541 s；peak MLX allocation 約 13.47 GB、幾乎相同。
這不是配對重複的效能通過：兩次相隔其他實驗，OS cache 與系統順序未受控。

因此在完整 ABBA／BAAB 效能 gate 前，先用
[`qwen_a_quality_gate.py`](qwen_a_quality_gate.py) 順序執行五類 prompt 的
control／candidate 256-output exact-token screen；遇到 process、ANE contract 或
token parity 失敗立即停止，不再消耗後續效能測量。這只是增加前置 correctness
screen，沒有取代或放寬第 5 節的正式效能門檻。Raw prompt files 是歷史 DeepSeek
tokenizer 產生的合成延展 workload；只能使用這次 Qwen 實際 token 數與 hash，
也不能把 exact parity 稱為真實任務能力提升。

Decode 的 QSA probe 合計約 0.708 s，normal control 的剩餘 Decode 約 9.93 s；
兩者來自不同同步條件的 run，不能直接作可加總占比。不過 component 的約 16%
改善不足以單憑此診斷支持整體 Decode 改善 5%，本輪先不啟用 Decode candidate。
GDN、N-gram 與 QSA indexer 的負結果同樣只適用本次 workload／M2 Max。

## 8. A 第一輪結案：未採用，保留失敗

完整證據入口為
[M2 Max first-round summary](../docs/benchmarks/2026-09-06-qwen-a-first-round-m2-max/summary.json)。
內含 commit、source／dependency hashes、電源與記憶體狀態、實際 ANE status、
完整 output token IDs／hash、命令與 raw logs。約 160 MB 的真實 tensor captures
保留在本機 scratch，artifact index 記錄位置／大小／SHA-256，沒有混入 Git 文件。

下表均為 **單次 screening，不是正式效能採用結果**；機器為 M2 Max、1,152 slots、
ANE active、persistent cache 關閉、OS cache 未清空。TTFT 不含模型初始化。

| Workload | Qwen prompt／output tokens | Control → candidate TTFT | 完整 request | Greedy output |
| --- | --- | --- | --- | --- |
| code pilot | 4,577／64 | 61.189 → 56.680 s | 71.123 → 66.541 s | 全部一致；非交錯配對 |
| mixed_math | 4,291／256 | 57.488 → 53.331 s | 93.751 → 89.703 s | 全部一致 |
| tool_like | 4,539／256 | 59.753 → 54.061 s | 96.562 → 93.502 s | **第 34 token 起不同，拒絕** |

Tool control 再執行一次，256 tokens 與首次 control 完全一致；兩次 hash 都是
`e72720ab3b92e42ac0c51fd88927c734134ec3ab24abad1c3ac4d7e12aca3212`，
candidate hash 為 `910e18ab77ecfff8fd02291e84d1e50aeced229d437d9217a0c219db456ba872`。
有 status 記錄的 full-model runs 皆 ANE active、48 evaluations、0 fallbacks。
Tool candidate 的 Decode throughput 還下降約 6.68%，但 output 已不同，不能把不同
生成工作量的速度差當成獨立因果結論。Mandatory token failure 已足以停止。

停止後沒有再跑該完整模型候選。zh_technical／repeated／code-256、ABBA／BAAB、
短 prompt／warm-restart prefix、長 context 與 M5 Pro／App 4,096-slot gate 都未執行，
是前置 gate 失敗後的停止，並非已通過或待偷偷補跑的採用結果。

另以獨立 attempt-02 做一次有界定位，使用
[`qwen_a_hybrid_gate.py`](qwen_a_hybrid_gate.py)：保留一段原版 grouped matmul 及
query_chunk=4，只替換另一段。執行前要求 real capture tensor-exact 且 component
改善至少 30% 才能進 full-model；兩個 hybrid 都在數值第一關停止，沒有量速度。

| Hybrid | 實際 Prefill tensor 不同元素 | 最大 absolute error | 結論 |
| --- | --- | --- | --- |
| direct QK + original PV | 482 | 0.00390625 | 不 exact，停止 |
| original QK + direct PV | 417 | 0.0078125 | 不 exact，停止 |

這定位出兩個計算階段都會造成可觀測差異；**尚未定位第 34 token 分歧的特定層、
累加指令或 router margin**。不能直接斷言哪段浮點累加造成該 token，亦未證明
語意品質下降。這些都是下一輪才可驗證的問題。原版 control 重複穩定是本次觀察，
不是所有平台／prompt 的 determinism 保證。

新增研究用 boundary tests 2 項通過（多個 dense／sparse、tail 與 future-mask cases），
Python 語法、JSON／artifact hashes 與文件連結另行檢查；這些成功不抵銷 full-model
failure。Production runtime、weights、預設值與 cache format 均未改動。

### 下一輪的具體決策

這是 A 內部下一筆研究投入的取捨；第 2 項另需改變 exact-output 品質合約。
未決定前不啟動下一輪，也不把新 kernel 放進產品。

| 選項 | 有界工作與論文機制依據 | 優點 | 代價／trade-off |
| --- | --- | --- | --- |
| 1．繼續追求完全相同的 Prefill | 固定實際輸入，逐段對照 MLX QK／PV 的 reduction／rounding；只有 tensor-exact 新設計才重啟完整模型。[FlashAttention](https://arxiv.org/abs/2205.14135)、[FlashInfer](https://arxiv.org/abs/2501.01005) 支持減少 attention 資料搬移的方向，不保證 bit parity | 沿用輸出合約；已有可量到的 Prefill 成本缺口 | 兩段都要處理；開發成本較高，修正後可能失去速度收益 |
| 2．維持 weights，改做品質等價評估 | 先定義獨立驗證集與 code／tool／中文等品質門檻，再評估同一 kernel；不以 token 差異直接判品質下降。FlashAttention 的 exact attention 指算法，不能替代本專案的品質評估 | 可保留已有 kernel，探索浮點實作的自由度 | 輸出可能不同；需要新品質合約，這次失敗不能追認為成功；未授權前不執行 |
| 3．先轉攻 Decode 的資料與呼叫成本 | 先分解 routed expert 的必要讀取、CPU/GPU 同步、小算子 dispatch，再選新實作。[DeepSpeed Inference](https://arxiv.org/abs/2207.00032) 的小 batch 算子融合／記憶體調度提供機制依據；其 CUDA／多 GPU 結果不適用本機速度預測 | 保留 weights／top-10／exact gate，探索本輪還沒實作的另一個等待來源 | 暫緩 Prefill kernel；要先重新定位，沒有整體 Decode 加速證據 |

第 3 項不重跑已拒絕的 LFU、ready-expert、next-layer prefetch 或 staged loading。
只有新的成本證據與實質不同的設計才可進候選 gate。推薦第 3 項作下一輪有界研究；
這是依本輪成本與相容性結果作出的研究排序，並非論文證明它必定更快。

使用者隨後選擇「先轉攻 Decode」。新一輪執行與結果另記於
[Decode 研究](QWEN_DECODE_RESEARCH_2026-09-06.md)，本輪已拒絕的 artifacts 保持原樣。
