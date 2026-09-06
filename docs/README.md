# Whallm 文件

本目錄只放目前有效的文件。
Whallm 的舊名稱是 DeepSeekV4SSD。

最後核對日期是 2026-09-06。
Qwen 支援核對版本是目前工作樹。
DeepSeek runtime 研究的核對基準是 commit
`997e2ca756d3d6c8ae97aa4df3effcf566ed449f`
加上目前 working tree 的 storage-aware profiling 與預設關閉的 hash exact
prefetch／adaptive block prototypes、sequential verification oracle、hybrid verifier
candidate、逐層 block-parity／layer 0 component diagnosis，以及 pinned checkpoint
的 native MTP-1 contract audit、DSpark 96-slot reduced-cache pilot，以及
expert-file page-cache residency proxy 與 default-cached expert-file bypass
research control、verifier expert-union acquisition tradeoff gate，以及 default-off
atomic DSpark prompt-context snapshot gate，並包含 native MTLIO ownership／copy-path
gate 與 MLX 0.32.0 public handoff stop，以及 fixed-arena／full-model staged
`w13`／`w2` streaming stop decision，以及 adaptive full-layer／selective prefill
route-union gate 與 exact-but-slower runtime stop，另包含 DSpark coherent Markov
candidate-path 的 greedy-only first-round gate 與 0／5 continuation stop，以及
6,000-label learned-router frozen-transfer baseline 與 training-required stop，另包含
normal prompt-cache format 5 的完整 contract、content-addressed block chain、partial
restart reuse、immutable sharing、mutable-state snapshot isolation 與 frequency-aware eviction
gate，並完成剩餘 training／
ANE／native trace／physical-I/O directions 的本機 prerequisite closure audit。
DeepSeek checkpoint revision 是
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`。
Qwen FP8 checkpoint revision 是
`bcd9f01ddc9cff2316eb84281bebcd5b058bddce`。
Qwen installed model revision 是
`753d0aa57059fad70a5f7e6cc249f25df56bbd34`。

## 文件入口

| 文件 | 內容 |
| --- | --- |
| [架構](ARCHITECTURE.md) | checkpoint 合約、installed model、runtime 資料路徑、cache 與目前限制。 |
| [API](API.md) | OpenAI 相容 endpoint、欄位、串流、驗證與錯誤。 |
| [驗證](VALIDATION.md) | 目前測試、完整 SHA-256、SSD 量測、端到端量測與歷史基準。 |
| [效能與瓶頸](PERFORMANCE.md) | 指標定義、瓶頸判讀、A/B 方法與 profiling 流程。 |
| [研究結論](RESEARCH.md) | 已採用、未採用、延後研究與 DSpark 決策。 |
| [Qwen 支援](QWEN.md) | Qwen checkpoint、installed model、runtime、API 和目前驗證邊界。 |

專案使用方式仍以根目錄的 [README](../README.md) 為入口。

## 目前狀態

| 里程碑 | 狀態 | 目前證據 |
| --- | --- | --- |
| M1：checkpoint 與 repack | 完成 | repack plan、續傳、repair、manifest 與完整 SHA-256 驗證。 |
| M2：正確 reference decode | 完成 | batch size 1、greedy decode、4K 測試與 MXFP4 單元測試。 |
| M3：SSD expert streaming | 完成 | 固定 slot、LFU、`preadv`、SSD microbenchmark 與 runtime 指標。 |
| M4：throughput 與 context | 完成 | chunked prefill、MXFP8 KV cache、8K/14K 歷史測試與 DSpark 決策。 |

M4 完成不代表 runtime 已驗證 1M context。
本專案只驗證文件列出的測試長度。

Qwen text model 第一版已完成。
官方 FP8 checkpoint 轉換、installed model SHA-256、對話、thinking、tool call、
greedy 4K、prompt cache 和舊安裝路徑的 packaged App 驗證都已通過。
目前 direct published installed model 下載路徑已通過 file 續傳單元測試。
目前尚未重新執行完整 125 GB 的 App direct download。

App 的 Model 頁面固定顯示 DeepSeek 與 Qwen。
使用者可以管理尚未安裝的模型。
Server 頁面只顯示 server 狀態、系統檢查和 server 設定。
啟動按鈕不依賴 Model 頁面的選擇。
server 可以使用空 model catalog 啟動。
Model 頁面的標題列提供模型資料夾按鈕。
模型卡片會顯示模型資料夾路徑與可用空間。
每個未安裝模型的列提供下載按鈕。
已安裝的 Qwen 缺少 MTP 時，模型列會顯示 MTP 下載按鈕。
MTP 安裝會保留既有 Qwen 主模型，並可續傳未完成的 MTP 下載。
每個模型的進階設定頁提供 Alias 欄位。
有效的 Alias 變更會自動儲存。
使用者可以在安裝模型前設定 Alias。
server 執行期間，App 只會停用 Loaded 或 Loading 模型的 Alias 和模型進階設定。
未載入模型的有效變更會同步到 server，並在下次載入時生效。
DeepSeek 下載固定包含 DSpark。模型列不提供排除 DSpark 的選項。
App 使用 pinned revision 的固定 installed model 大小執行下載前空間檢查。
App 啟動時不會為了取得下載大小連線到 Hugging Face。
每個模型列的「進階設定」動作使用 Tertiary icon button。
每個模型列在「進階設定」左側提供 Load 或 Unload icon button。
按鈕提供 40 pt 點擊區域、Tooltip 和 VoiceOver 名稱。
模型資料夾可用空間不足時，App 會停用下載按鈕。
App 會在模型列顯示停用原因、所需空間與可用空間。
游標停留在停用按鈕上時，App 也會顯示相同原因。
下載期間，模型列會依序顯示目前階段、百分比、完成容量、下載速度與剩餘時間。
下載模型時，使用者可以選擇另一個已安裝模型並啟動 Server。
Server 執行時，App 可以下載另一個尚未安裝的模型。
App 一次只執行一個模型下載。
server 啟動時只讀取 installed model 清單。
第一個 generation request 會載入指定模型。
使用者也可以在 Model 頁面手動載入或卸載模型。
server 一次只保留一個載入的模型。
已載入模型會移到 Model 區塊上方的 Loaded 區塊。
Chat 頁面的模型選單只顯示 server 啟動時可用的 installed model。
使用者切換 Chat 模型時，App 會保留對話。
使用者在回覆期間切換頁面時，App 會繼續接收 Chat 回覆。
Metric 頁面顯示目前載入或載入中的 API model ID。
模型切換時，App 會清除舊模型的效能歷史。
Qwen 預設啟用 private ANE Prefill projection。
模型進階設定可以調整 GPU 和 ANE 的 output channels 分配比例。
預設 ANE 比例是 0.25。
原本 GPU Prefill 路線仍保留。
private interface 錯誤時，runtime 會自動回退到原本路線。
DeepSeek 的 layer-major Prefill 門檻可以在模型進階設定中調整。
預設門檻是 1,024 個未快取 token。
「系統檢查」固定顯示在 Server 頁面。
該列使用硬體 SF Symbol 和狀態圖示顯示 Apple Silicon、記憶體和高速 SSD 檢查。

`PLAN.md` 的 21 個研究方向在 2026-08-27 snapshot 已全部取得 scoped disposition：
採用、default-off、candidate stop、feasibility rejection，或有本機證據的 prerequisite
closure。這不表示 deferred training／ANE drafter 已實作；目前 Qwen 的 fixed-shape
projection 是另一條已實作路線。其餘方向的 reopening
條件以 requirement tracker 為準。

## 可信度規則

文件使用下列四種證據。

1. **目前程式碼**：描述目前介面、預設值和資料路徑。
2. **目前測試**：描述自動測試已覆蓋的行為。
3. **本機量測**：描述指定硬體、設定和 cache 狀態下的結果。
4. **研究假設**：描述尚未實作或尚未量測的方向。

外部文件只能證明外部 API、格式或論文結果。
外部文件不能證明本專案的速度。

若文件與程式碼衝突，請先以目前程式碼為準。
接著請更新文件和測試。

## 歷史研究

SSD Streaming 的 active 瓶頸整理、目前程式碼對照和分階段執行規則位於
[`research/SSD_STREAMING_BOTTLENECKS_2026-08-31.md`](../research/SSD_STREAMING_BOTTLENECKS_2026-08-31.md)。
Phase 1 cache replacement 上限分析已完成。
五種 workload 的 formal artifact 位於
[`benchmarks/2026-09-01-expert-cache-oracle-m5-pro.json`](benchmarks/2026-09-01-expert-cache-oracle-m5-pro.json)。
Phase 2 causal cache policy quick gate 已停止。
最佳候選只取得 Belady 改善的 41.63% 中位數。
完整結果位於
[`benchmarks/2026-09-01-expert-cache-causal-policy-m5-pro.json`](benchmarks/2026-09-01-expert-cache-causal-policy-m5-pro.json)。
Phase 3 deadline-ready trace 與 default-off Qwen next-layer prefetch gate 已完成。
候選把 aggregate 暴露 read wait 降低 27.41%。
Ready-before-deadline rate 只從 0% 提高到 0.52%。
Logical wasted bytes 沒有增加。
候選保持 default-off。
Baseline 位於
[`benchmarks/2026-09-01-prefetch-deadline-baseline-m5-pro.json`](benchmarks/2026-09-01-prefetch-deadline-baseline-m5-pro.json)。
候選 gate 位於
[`benchmarks/2026-09-01-qwen-next-layer-prefetch-gate-m5-pro.json`](benchmarks/2026-09-01-qwen-next-layer-prefetch-gate-m5-pro.json)。
Phase 4 expert blob 壓縮 microbenchmark 已完成。
60 個解壓縮結果全部 byte-exact，Peak RSS gate 也通過。
但是，LZ4 和 LZFSE 都沒有同時通過無 Metal 與有 Metal 的 5% 時間 gate。
正式結果位於
[`benchmarks/2026-09-01-expert-blob-compression-m5-pro.json`](benchmarks/2026-09-01-expert-blob-compression-m5-pro.json)。
Exact track 已停止，不進入 Phase 5 runtime prototype。
使用者已明確啟動獨立的 Phase 6 approximate 研究。
`learned-route-drop-lowest-1` 已完成 Phase 6A 至 Phase 6E。
Phase 6E 使用五種 4K workload、兩個 reversed-order paired waves 和 20 個
fresh-process 4K／256 runs。
10 個 paired outputs 都是 256/256 token 相同。
Aggregate Decode logical expert bytes 減少 15.14%。
Decode throughput change 中位數是 +8.37%。
五個 workload 的 Decode p95 change 中位數都沒有 regression。
2026-09-01 後續使用者決定把 candidate 設為一般 DeepSeek request 的預設模式。
API、CLI 和 APP request 未指定 mode 時會使用 `learned-route-drop-lowest-1`。
Client 可以明確指定 `exact`。
Qwen 和啟用 DSpark 的 DeepSeek 維持 exact。
研究合約位於
[`research/APPROXIMATE_MODE_2026-09-01.md`](../research/APPROXIMATE_MODE_2026-09-01.md)。
Entry artifact 位於
[`benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json)。
Component、pilot 和 formal artifacts 分別位於
[`benchmarks/2026-09-01-approximate-expert-drop-component-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-component-m5-pro.json)、
[`benchmarks/2026-09-01-approximate-expert-drop-4k32-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-4k32-m5-pro.json) 和
[`benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json`](benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json)。

日期型研究草稿已移到 [`research/archive`](../research/archive/README.md)。
歷史研究保留原始推論與量測。
歷史研究不描述目前 runtime。

目前的研究查核位於
[`research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md`](../research/EXTERNAL_TECHNICAL_CLAIM_AUDIT_2026-08-10.md)。
Qwen runtime 的 active 效能研究、採用門檻和 MTP 合約查核計畫位於
[`research/QWEN_RUNTIME_OPTIMIZATION_2026-08-29.md`](../research/QWEN_RUNTIME_OPTIMIZATION_2026-08-29.md)。
2026-09-06 的現況查核、成功／失敗經驗、論文支持的新 Prefill／Decode 假設和
研究路線與第一輪拒絕證據位於
[`Qwen Prefill／Decode 新研究`](../research/QWEN_PREFILL_DECODE_RESEARCH_2026-09-06.md)。
使用者已選擇轉攻 Decode，後續成本分解、resident grouped QMM 原型與 gate 位於
[`Qwen Decode 研究`](../research/QWEN_DECODE_RESEARCH_2026-09-06.md)。
後續 default-off runtime、4,096-slot 分區 arena 與多 request gate 位於
[`Qwen Decode 整合研究`](../research/QWEN_DECODE_INTEGRATION_2026-09-06.md)。
最新分區版本因長工具 Decode +4.58% 未達預定 5% gate 而停止，保持預設關閉，
使用者已選 B：保留原門檻，接續 [跨區塊 kernel 研究](../research/QWEN_DECODE_KERNEL_2026-09-06.md)。
新 kernel 已通過 M2 Max 的 14 個效能波次：Decode +9.59–16.42%、request 縮短
2.35–5.12%，輸出一致，cold memory 與 lifecycle 也通過。它只由 research runner
啟用，尚未整合到原 CLI flag；M5 Pro、能耗與產品採用未完成。
研究原型與診斷量測不代表已採用的 runtime 預設。
Qwen QSA Grouped-KV 的採用 quick gate 位於
[`docs/benchmarks/2026-09-02-qwen-qsa-grouped-kv-adoption-quick-gate-m5-pro.json`](benchmarks/2026-09-02-qwen-qsa-grouped-kv-adoption-quick-gate-m5-pro.json)。
Qwen private ANE Prefill 的探索性 gate 位於
[`docs/benchmarks/2026-09-02-qwen-private-ane-prefill-exploratory-m5-pro.json`](benchmarks/2026-09-02-qwen-private-ane-prefill-exploratory-m5-pro.json)。
Qwen MTP installed payload 驗證位於
[`docs/benchmarks/2026-08-30-qwen-q4-mtp-installed-payload-m5-pro.json`](benchmarks/2026-08-30-qwen-q4-mtp-installed-payload-m5-pro.json)。
Qwen MTP 與固定 Ollama commit 的 reference parity 位於
[`docs/benchmarks/2026-08-30-qwen-q4-mtp-ollama-reference-parity-m5-pro.json`](benchmarks/2026-08-30-qwen-q4-mtp-ollama-reference-parity-m5-pro.json)。
Qwen MTP block verifier 的五種 greedy output parity 位於
[`docs/benchmarks/2026-08-30-qwen-mtp-block-greedy-parity-128-m5-pro.json`](benchmarks/2026-08-30-qwen-mtp-block-greedy-parity-128-m5-pro.json)。
Qwen MTP 的 4K 正式 stop gate 位於
[`docs/benchmarks/2026-08-30-qwen-mtp-block-formal-stop-code4k64-m5-pro.json`](benchmarks/2026-08-30-qwen-mtp-block-formal-stop-code4k64-m5-pro.json)。
Qwen MTP bounded Prefill 的 4K 正式 stop gate 位於
[`docs/benchmarks/2026-08-30-qwen-mtp-bounded-prefill-formal-stop-code4k64-m5-pro.json`](benchmarks/2026-08-30-qwen-mtp-bounded-prefill-formal-stop-code4k64-m5-pro.json)。
32-slot bounded Prefill 的 4K output parity 位於
[`docs/benchmarks/2026-08-30-qwen-mtp-bounded-prefill-slots32-code4k64-parity-m5-pro.json`](benchmarks/2026-08-30-qwen-mtp-bounded-prefill-slots32-code4k64-parity-m5-pro.json)。
MTP verifier burst latency 的 Draft-2 與 Draft-1 探索結果位於
[`Draft-2`](benchmarks/2026-08-30-qwen-mtp-draft2-latency-code4k64-exploratory-m5-pro.json)
和
[`Draft-1`](benchmarks/2026-08-30-qwen-mtp-draft1-latency-code4k64-exploratory-m5-pro.json)。
較早的本機 distribution artifact 已標示為 superseded。
`PLAN.md` 全方向的 active requirement／evidence／remaining-gate tracker 位於
[`research/PLAN_EXECUTION_STATUS_2026-08-27.md`](../research/PLAN_EXECUTION_STATUS_2026-08-27.md)。
Hash exact prefetch 的 active experiment plan 位於
[`research/HASH_EXACT_PREFETCH_2026-08-26.md`](../research/HASH_EXACT_PREFETCH_2026-08-26.md)。
Native MTP-1 checkpoint contract 的可重現 feasibility audit 位於
[`research/MTP1_CONTRACT_AUDIT_2026-08-27.md`](../research/MTP1_CONTRACT_AUDIT_2026-08-27.md)。
DSpark state ownership 與 reduced-cache pilot 位於
[`research/DSPARK_STATE_OWNERSHIP_2026-08-27.md`](../research/DSPARK_STATE_OWNERSHIP_2026-08-27.md)。
Expert-file page-cache proxy 的合約與限制位於
[`research/PAGE_CACHE_RESIDENCY_PROXY_2026-08-27.md`](../research/PAGE_CACHE_RESIDENCY_PROXY_2026-08-27.md)。
Expert-file descriptor bypass 的預先登記、installed-range contract 與 full-model
決策位於
[`research/EXPERT_FILE_CACHE_BYPASS_2026-08-27.md`](../research/EXPERT_FILE_CACHE_BYPASS_2026-08-27.md)。
Sequential／grouped／hybrid verifier 的 union-call 與 execution-shape tradeoff 位於
[`research/VERIFIER_UNION_TRADEOFF_2026-08-27.md`](../research/VERIFIER_UNION_TRADEOFF_2026-08-27.md)。
Atomic target-KV + DSpark-context reuse 的 contract 與 functional result 位於
[`research/DSPARK_PROMPT_CONTEXT_SNAPSHOT_2026-08-27.md`](../research/DSPARK_PROMPT_CONTEXT_SNAPSHOT_2026-08-27.md)。
Native MTLIO bytes／shared／private buffer、shared-event、cancellation 與 installed MLX
handoff audit 位於
[`research/MTLIO_EXPERT_STREAMING_2026-08-27.md`](../research/MTLIO_EXPERT_STREAMING_2026-08-27.md)。
Fixed `w13`／`w2` arenas 與 default-off split-slot runtime 的 protocol／stop decision
位於
[`research/STAGED_EXPERT_STREAMING_2026-08-27.md`](../research/STAGED_EXPERT_STREAMING_2026-08-27.md)。
DSpark coherent Markov beam 的 proposal contract、greedy-only gate 與 stop decision
位於
[`research/DSPARK_CANDIDATE_PATHS_2026-08-27.md`](../research/DSPARK_CANDIDATE_PATHS_2026-08-27.md)。
DSpark hidden 到 target learned routers 的 fixed-label dataset、top-k recall／union-byte
gate 與 direct-transfer stop 位於
[`research/DSPARK_LEARNED_ROUTER_PREDICTOR_2026-08-27.md`](../research/DSPARK_LEARNED_ROUTER_PREDICTOR_2026-08-27.md)。
Normal block-granular immutable persistent prefix cache 的 format-4 contract、format-5
mutable-state snapshot isolation 修正與 full-model functional gate 位於
[`research/BLOCK_PROMPT_CACHE_2026-08-27.md`](../research/BLOCK_PROMPT_CACHE_2026-08-27.md)。
剩餘 dense drafter、ANE、native trace、physical-I/O 與 non-equivalent model directions 的
local evidence、scoped stop/defer 與 reopening rules 位於
[`research/PLAN_PREREQUISITE_CLOSURE_2026-08-27.md`](../research/PLAN_PREREQUISITE_CLOSURE_2026-08-27.md)。
完整 installed model 的探索性 artifacts 位於
[`hash exact prefetch smoke`](benchmarks/2026-08-26-hash-exact-prefetch-smoke-m2-max.json)
與
[`adaptive block smoke`](benchmarks/2026-08-26-adaptive-block-smoke-m2-max.json)。
五類 128-token 校準另保存
[`失敗的第一輪`](benchmarks/2026-08-26-adaptive-block-calibration-wave1-m2-max.json)、
[`修正後第二輪`](benchmarks/2026-08-26-adaptive-block-calibration-wave2-m2-max.json)
與
[`低信心 repair validation`](benchmarks/2026-08-26-adaptive-block-repair-validation-m2-max.json)。
五類 4K／256-output selector gate 保存於
[`adaptive decision survey`](benchmarks/2026-08-26-adaptive-block-4k256-survey-m2-max.json)，
其 prompts 與 token hashes 保存於
[`4K manifest`](benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json)。
低信心 discovery 的 prompts 保存於
[`discovery manifest`](benchmarks/prompts/2026-08-26-adaptive-discovery-4096/manifest.json)，
短 screening 保存於
[`4K／16 discovery`](benchmarks/2026-08-26-adaptive-block-discovery-4k16-m2-max.json)。
長 decode 雖觸發 storage-aware selection，但沒有通過 exact output parity；失敗證據保存於
[`4K／256 adaptive`](benchmarks/2026-08-26-adaptive-block-random-hex-4k256-no-fallback-m2-max.json)、
[`4K／256 fixed`](benchmarks/2026-08-26-fixed-block-random-hex-4k256-no-fallback-m2-max.json)、
[`4K／32 exact-token reproduction`](benchmarks/2026-08-26-random-hex-4k32-fixed-adaptive-failed-m2-max.json)
與
[`block parity diagnostic`](benchmarks/2026-08-26-dspark-block-parity-diagnostic-m2-max.json)。
逐 token target verification follow-up 保存於
[`fixed oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-fixed-oracle-m2-max.json)
與
[`adaptive oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-adaptive-oracle-m2-max.json)；
兩者恢復相同 sequential reference token hash，但移除了 block batching，不能作為速度結果。
逐層 follow-up 保存於
[`layer parity diagnostic`](benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json)：
exact embeddings 在第 0 層 `LocalAttention` 後首次產生數值差異，learned router 到後續層
才更換 selected expert，因此 router 是後續放大邊界，不是最早觀測到的起點。
Layer 0 component follow-up 保存於
[`attention component diagnostic`](benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json)：
attention input、norm 與 KV projection／RoPE exact；HyperConnection `post`／`combine`
先出現極小 shape-dependent 差異，`wq_b` 在 exact input 下也出現 block-shape 差異。
Sequential 的兩次 one-token in-place cache update 與 block 的 two-token concatenate／mask
路徑不同，但最後 temporal-order cache 的共同 128-token suffix exact。這仍未把差異歸因
到單一 kernel。
Layer 0 FFN follow-up 保存於
[`FFN component diagnostic`](benchmarks/2026-08-27-dspark-layer0-ffn-component-diagnostic-m2-max.json)：
router IDs／scores 與 routed expert outputs exact，FFN HyperConnection `post`／`combine`
先出現差異；shared expert 的極小差異在 MoE output dtype 邊界被消除。

Default-off hybrid verifier 的演進保存於
[`single-state hybrid diagnostic`](benchmarks/2026-08-27-dspark-hybrid-verifier-diagnostic-m2-max.json)、
[`v1 multi-round stop`](benchmarks/2026-08-27-dspark-hybrid-random-hex-4k32-m2-max.json)、
[`v2 five-workload stop`](benchmarks/2026-08-27-dspark-hybrid-v2-discovery-4k32-m2-max.json)
與
[`v3 five-workload correctness gate`](benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json)。
V3 將 attention、FFN HyperConnection、router、shared／routed expert math 維持 one-token
shape，但每層只 acquire 一次 expert union；五個 4K workloads 的 normal／fixed greedy
tokens 全部 exact。

Exact hash prefetch 與 adaptive selector 的 v3 composition 分別保存於
[`hybrid + hash`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json)
與
[`hybrid + hash + adaptive`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json)。
五組 normal／fixed／adaptive tokens 全部 exact；這些單次、未控制 OS page cache 的
artifact 只作 correctness、selector behavior 與 logical-byte accounting 證據。
這些 artifact 都不是正式速度結果。

Research-only `mincore` probe 的 installed-model contract 與 hash-prefetch
composition 分別保存於
[`probe ABBA`](benchmarks/2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json)、
[`short composition`](benchmarks/2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json)
與
[`4K useful/wasted coverage`](benchmarks/2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json)。
4K gate 的 logical useful rate 是 23.53%，nonresident-byte useful rate 是 29.65%；
1.618 GB wasted prefetch 在讀取前仍為 nonresident。這是 expert-file-specific 的
page-cache-miss proxy，不是 physical SSD byte counter，且預設關閉。

Explicit expert-file policy 的 installed-range contract 保存於
[`bypass contract`](benchmarks/2026-08-27-expert-file-cache-bypass-contract-m2-max.json)；
六個完全 nonresident 的 13,369,344-byte expert ranges 在 aligned bypass read 後仍
完全 nonresident，cached control 後則完全 resident，且兩條 byte hashes exact。
原先一次選定六個 ranges 的方法受 cached read-ahead 影響；5/6 通過的 pilot 保留為
[`selection-race evidence`](benchmarks/2026-08-27-expert-file-cache-bypass-selection-race-pilot-m2-max.json)。
修正為 just-in-time recheck 後才允許完整 4K／32 三波 Latin-square gate。
[`full-model bypass artifact`](benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json)
的 12 條 token sequence exact，observer accounting 全部閉合；fixed 與 adaptive 的
request／Decode 中位數分別相對 normal 變差 +18.19%／-59.67% 與
+5.02%／-25.58%，所以兩個候選都拒絕。Runtime 的 expert-file policy 預設維持
`cached`，DSpark、hybrid、hash prefetch 與 adaptive 仍預設關閉。

P1 verifier-union gate 另以 128-token `repeated` prompt、8 output tokens 執行四波
normal／sequential／grouped／hybrid Latin square 與三個 observer runs。19 條 output
exact；hybrid 將 union acquisition calls 從 258 降為 43、target bytes 比 sequential
少 12.31%，但 verification 中位數仍慢 17.34%，且比 grouped 慢 87.59%。因此
one-per-layer union contract 已確認，但目前 token-shaped hybrid 的 aggregate execution
cost 仍抵銷 dedupe。Grouped path 在這個高信心案例較快，卻已在既有低信心
multi-workload gate 失去 exact parity，不能採用。Corrected artifact 位於
[`union tradeoff`](benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json)；
最初過度歸因 QMM 的標籤版保留為
[`attribution-label pilot`](benchmarks/2026-08-27-verifier-union-tradeoff-attribution-label-pilot-m2-max.json)，
不使用其 timing。

P2 atomic prompt-context gate 以獨立 format-3 bundle 同時保存 target KV 與三個 DSpark
context states。128-token `repeated` prompt 的首次／memory／restart-persistent 三條
8-token outputs 全部 exact，reuse sources 是 `none`／`memory`／`persistent`，重用 token
數是 0／127／127。Memory 與 restart hits 的 combined logical expert bytes 分別下降
92.23%／65.59%，metadata contract exact。這只完成 functional integration boundary，
不是 balanced performance result；功能與 DSpark 仍預設關閉。Artifact 位於
[`atomic snapshot`](benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json)。

P2 MTLIO native gate 對 1／6／32／128 個 canonical expert ranges 執行 `preadv`、
MTLIO bytes、shared buffer 與 private buffer，共 16 列 8,930,721,792 bytes，所有
candidate／reference hashes exact。Shared/private 的 external shared-event → GPU wait
與取消 destination admission 也通過；但 installed MLX 0.32.0 的公開介面沒有外部
`MTLSharedEvent` dependency handoff。32／128 aggregate exploratory timing 也沒有達到
10% 門檻。Runtime integration 因此停止，現有四 worker `preadv` 路徑不變。Artifact
位於 [`MTLIO gate`](benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json)；OS page
cache 未 purge，這不是正式效能或 physical-SSD 結果。

P2 staged streaming 先在 fixed arenas 完成 36 組 paired samples；4／8-row
optimistic shapes 的 complete wall 分別改善 10.74%／13.67%，byte、六個 region hashes
與 float32 output hashes 全部 exact，因此允許 default-off runtime follow-up。完整模型
再以 128-token `repeated` prompt、32 greedy outputs 跑四波 control／staged fresh-process
pairs：八條 output hashes exact、logical expert bytes／evictions／peak memory 都不增加，
每個 staged run 的 1,024 次 split reads 也完整閉合。但 request paired median +2.17%、
Decode throughput -4.26%，未達 5% 改善線，所以 candidate 停止且不提供 CLI／server／
APP 開關。Artifacts 位於 [`fixed-arena gate`](benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json)
與 [`runtime gate`](benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json)；
兩者都不是正式效能或 physical-SSD 結果。

P2 adaptive expert prefill 的五類 4K route-union gate 讓 70%／80%／90% 三個門檻
全部取得 runtime follow-up 資格；跨 workload 的 prefill expert-byte estimate 中位數
分別是 -23.19%／-33.38%／-34.24%。Internal fixed-buffer prototype 保留原本 256-row
layout 與 `gather_qmm`，只把 union rows 讀回原 offset。`repeated` 4K 的兩組反向
fresh-process pairs 全部 token／bytes exact，request expert bytes -66.78%、peak MLX
memory -15.78%，但 TTFT paired median +16.87%、p95 +17.02%。三個 threshold 在此
workload 的 42 層決策完全相同，全部已無法通過 per-workload p95 guard，因此完整矩陣
提前停止。Full-layer 預設不變，也沒有 CLI／server／APP opt-in。Artifacts 位於
[`route gate`](benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json)
與 [`runtime stop`](benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json)。

P1 storage-aware candidate-path gate 從同一次 DSpark backbone 建立 coherent
branch-4／beam-8 Markov paths，並以 64 組 probability／requested hash union／LFU miss
權重選路。五組 128-token fresh workers 的 draft reconstruction、sequential target
truth 與 exact hash routes 全部通過，但 baseline 全部接受 5/5；`code`／`tool_like`
選到 lower-union alternatives 時 acceptance 降到 4／1，0/5 通過 continuation。
Candidate 已停止，沒有 runtime setting，sampling proposal 也未證明。Artifact 位於
[`candidate-path gate`](benchmarks/2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json)。

P1 learned-router predictor baseline 另建立 6,000 個 layers 3--42 target labels，並比較
四個既有 DSpark feature taps 的 top-6／12／24／48 direct frozen-router transfer。最佳
允許候選 `dspark_final_norm` top-24 只有 17.15% assignment recall、33.03% union recall、
11.15% useful rate，且 0/40 layers 達到 75% recall；top-48 也不合格。Direct transfer
因此停止，不建立 speculative prefetch／probation／deadline pinning。Artifact 位於
[`learned-router baseline`](benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json)。

Pinned checkpoint 的 MTP audit 保存於
[`MTP-1 contract artifact`](benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json)。
官方 inference graph 使用三個 `mtp.*` stages：`mtp.0` 擁有 target-hidden input
adapter，`mtp.2` 擁有 output heads，沒有任何單一 stage 是 self-contained。
因此 runtime 不會把 `num_nextn_predict_layers=1` 解讀為可直接裁切的 faithful MTP-1；
這是 checkpoint contract 的 feasibility rejection，不是效能結果。

DSpark 96-slot reduced independent-cache pilot 保存於
[`three-workload summary`](benchmarks/2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json)
與
[`4K/128 long-decode stop`](benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json)。
三組 4K／32 workload 都維持 exact tokens 且 peak memory 較低；但在預先宣告的
`random_hex` 4K／128 gate，96 slots 雖使 peak memory 降低 7.35%，draft expert
bytes/committed token 卻增加 83.17%，超過 +50% 停止線。這個 candidate 已停止，
預設維持 768 slots。

## 專案用語

本專案固定使用下列用語。

- **checkpoint**：固定 revision 的 Hugging Face 模型與 safetensors shards。
- **common tensor**：runtime 保留在記憶體的非 routed expert tensor。
- **routed expert**：一個模型專用 expert。DeepSeek 使用 `w1`、`w2`、`w3`。
  Qwen 使用 fused `gate_up` 和 `down`。
- **expert blob**：一個 routed expert 的標準封裝 bytes。
- **repack plan**：checkpoint byte range 到 installed model byte range 的完整對應。
- **installed model**：由 repack plan 產生並驗證的本機目錄。
- **API model ID**：一個 model kind 的固定且區分大小寫的 API 名稱。
- **Alias**：API model ID 的選用 request 名稱。Alias 區分大小寫。
- **manifest**：定義 installed model 與完整性資料的 JSON 檔案。
- **slot**：可放置一個 expert blob 的固定 Metal 可見記憶體區域。
- **main model**：43 個目標模型層。main model 不包含 DSpark。
- **DSpark**：儲存在 `mtp.*` 的可選 speculative decoding 模組。
- **N-gram store**：Qwen 使用的 read-only `ngram.bin` row store。
- **model kind**：manifest 用來選擇 DeepSeek 或 Qwen runtime 的欄位。
