# 效能與瓶頸

本文件定義 runtime 指標和瓶頸檢測方法。
量測結果請見[驗證紀錄](VALIDATION.md)。

## 先固定測試狀態

同一個 prompt 也可能得到不同速度。
下列狀態會改變結果。

- 第一次 MLX graph compile 或 warm graph。
- 作業系統 page cache。
- expert cache 的 resident slot。
- 記憶體 prompt cache。
- persistent prompt cache。
- prompt 長度和 routed expert 分布。
- output 長度。
- SSD、FileVault 和背景 I/O。

每個結果必須標示 fresh 或 warm 狀態。
「cold」只能用於有明確清除方法的 cache。
如果作業系統 page cache 沒有清除，請寫 `not purged`。

## 指標定義

### Request 時間

`RuntimeMetrics` 在 prompt tokenization 和 prompt cache acquisition 之後開始計時。
因此 runtime TTFT 不包含 tokenizer 時間。
runtime TTFT 也不包含 persistent prompt cache 載入時間。

`time_to_first_token_seconds` 從 runtime request 開始計到第一個 output token
完成 cache evaluation。

第一個 output token 出現前，App 使用即時 `request_seconds` 顯示
First Token wait time。
第一個 output token 出現後，App 改為顯示最終的
`time_to_first_token_seconds`。
效能歷史只記錄每個 request 的最終 First Token wait time。

`request_seconds` 在 generation 結束時停止。
`request_seconds` 不包含 response 結束後的 persistent cache serialize 和 write。

外部 client wall time 可能包含更多工作。
正式報告應同時記錄 client wall time 和 runtime request time。

### Prefill

runtime 使用下列公式：

```text
uncached_prompt_tokens = runtime_prompt_tokens - prompt_cache_reused_tokens
prefill_tokens_per_second = uncached_prompt_tokens / time_to_first_token_seconds
```

這個 prefill 指標包含第一個 output token 的工作。
這個 prefill 指標不是純 attention kernel throughput。

`request_prefill_step_size` 是本次 request 實際使用的 step。
`layer_major_prefill` 表示本次 request 是否進入 layer-major path。
`layer_major_prefill_tokens` 是 layer-major kernel 實際處理的 token 數。

`request_batched_expert_layers` 記錄 full-layer batched MoE 次數。
`request_gather_qmm_calls` 記錄 batched `gather_qmm` call 次數。

CLI 的 `prompt_token_sha256` 記錄完整 prompt token ID hash。
CLI 的 `token_sha256` 記錄完整 output token ID hash。
CLI 也會記錄 attention 與 MoE 的實際 Prefill 區塊大小。

### Decode

第一個 output token 計入 TTFT。
decode token 數使用：

```text
decode_tokens = max(0, runtime_generation_tokens - 1)
```

`accumulated_generation_tokens` 是目前 server process 產生的 output token 總數。
這個欄位包含進行中的 request。
server restart 會把這個欄位重設為 0。

主要 decode 指標如下。

| 欄位 | 意義 |
| --- | --- |
| `decode_model_step_seconds` | 取得後續 token 的 generator step wall time。這個時間可以包含 expert I/O 和 MLX evaluation。 |
| `decode_cache_eval_seconds` | step 後的 prompt cache array evaluation。 |
| `decode_end_to_end_seconds` | model step 加 cache evaluation。 |
| `decode_tokens_per_second` | decode token 除以 end-to-end seconds。 |
| `decode_model_step_tokens_per_second` | decode token 除以 model step seconds。 |
| `decode_latency_p50_seconds` | 每 token model step 加 cache evaluation 的 p50。 |
| `decode_latency_p95_seconds` | 每 token model step 加 cache evaluation 的 p95。 |

`performance.tokens_per_second` 是 server streaming 的即時速率。
正式結果應使用 `decode_tokens_per_second`。

### Expert I/O

| 欄位 | 意義 |
| --- | --- |
| `request_expert_bytes_read` | 本次 request 的 logical expert bytes。包含 full-layer batched read、LFU slot miss 和 speculative scratch read；OS page-cache hit 也會計入。 |
| `request_process_disk_bytes_read` | macOS 對本 process 計帳的 request 期間 disk read bytes。不是 expert file 專用計數器；無法取得時是 `null`。 |
| `request_process_disk_bytes_written` | macOS 對本 process 計帳的 request 期間 disk write bytes；無法取得時是 `null`。 |
| `request_expert_page_cache_probe_calls` / `failures` | Research-only pre-read `mincore` calls 與無法分類的 calls；probe 預設關閉。 |
| `request_expert_page_cache_classified_bytes` | 由 expert-file VM page residency 成功分類的 logical bytes。 |
| `request_expert_page_cache_resident_bytes_before_read` | `preadv` 前所在 VM page 已 resident 的 classified bytes。 |
| `request_expert_page_cache_nonresident_bytes_before_read` | `preadv` 前所在 VM page 未 resident 的 classified bytes；是 page-cache-miss proxy，不是 physical SSD bytes。 |
| `request_expert_page_cache_unclassified_bytes` | Probe unavailable／failure 時仍由 `preadv` 成功返回、但無 residency classification 的 bytes。 |
| `request_expert_page_cache_nonresident_bytes_per_generated_token` | Request nonresident proxy bytes 除以 generated tokens。 |
| `request_expert_read_seconds` | 每個 read batch 的 critical wall time 加總。 |
| `request_ssd_read_bytes_per_second` | bytes 除以 expert read timer。 |
| `request_expert_cache_hits` | 個別 slot lookup hit。 |
| `request_expert_cache_misses` | 個別 slot lookup miss。 |
| `request_expert_evictions` | 本次 request 的 slot eviction。 |
| `request_prefetched_layer_hits` | 進入 batched layer 時，所有 prefetch worker 已完成的 layer 數。 |
| `request_expert_union_calls` | 以 routed expert union 取得權重的 layer call 數。 |
| `request_routed_expert_assignments` | union 去重前的 routed expert assignment 數。 |
| `request_expert_union_experts` | 每個 layer call 的 unique expert 數加總。 |
| `request_expert_union_reused_assignments` | assignments 減去 unique experts。 |
| `request_expert_union_reuse_rate` | union 去除的 assignment 比例。 |
| `request_expert_union_misses` | union 中不在 runtime expert cache 的 unique expert 數。 |
| `request_speculative_prefetch_requested_experts` | exact prefetch union 的 `(layer, expert)` 數。 |
| `request_speculative_prefetch_cache_resident_experts` | prefetch 啟動時已在主 LFU cache、只需暫時 pin 的 expert 數。 |
| `request_speculative_prefetch_bytes_read` | 讀入 verification scratch 的 logical bytes。這些 bytes 已包含在 `request_expert_bytes_read`。 |
| `request_speculative_prefetch_wait_seconds` | target 因 scratch future 尚未完成而等待的 wall time。 |
| `request_speculative_scratch_hits` | main model 從 active verification scratch 取得 expert 的次數；replay 重用會再次計數。 |
| `request_staged_expert_reads` | Research-only split-slot path 完成的 direct expert miss 數；正常 runtime 是 0。 |
| `request_staged_w13_bytes_read` / `request_staged_w2_bytes_read` | 已完成 split reads 的 logical bytes；兩者之和必須等於 staged reads 乘 canonical expert blob bytes，且已包含在 `request_expert_bytes_read`。 |
| `request_staged_read_seconds` | Staged miss batch 從第一個 `w13` read 開始到最後一個 `w2` complete 的 critical wall 加總。 |
| `request_staged_w2_wait_seconds` | First-stage graph submit 後等待特定 routed expert `w2` future 的 wall 加總。 |
| `request_staged_first_stage_submit_seconds` | 建立並提交 first-stage MLX graph 的 CPU wall 加總；不是 GPU kernel duration。 |
| `request_adaptive_prefill_planned_layers` | Internal adaptive prefill 完成 exact route planning 的 expert layer 數；正常 runtime 是 0。 |
| `request_adaptive_prefill_full_layers` / `request_adaptive_prefill_selective_layers` | Threshold decision 的 full／selected-row layer 數；兩者之和應等於 planned layers。 |
| `request_adaptive_prefill_union_experts` | 每層 exact routed union 大小的 request 總和。 |
| `request_adaptive_prefill_read_experts` | Decision 後實際要求 full／selective batched read 的 expert rows 總和；必須覆蓋 union。 |
| `request_adaptive_prefill_bytes_read` | 成功完成的 adaptive batched logical bytes；等於 read experts 乘 canonical expert blob bytes，且已包含在 `request_expert_bytes_read`。 |
| `request_adaptive_prefill_avoided_bytes` | 相對每層 256-row full read 的 deterministic logical byte difference；不是 physical SSD bytes。 |
| `request_adaptive_prefill_plan_seconds` | 建立 tile tensors、同步 exact routes 並形成 union 的 CPU wall；shared-expert GPU work 可能仍為 asynchronous，不可當成 isolated kernel time。 |

full-layer batched prefill 不增加 expert cache hit 和 miss。
因此 expert cache hit rate 不能代表 prefill 的全部 I/O。

prefetch 可以和前一層 GPU work 重疊。
因此 expert read timer 不能直接從 request time 扣除。

`request_process_disk_bytes_read` 來自 Darwin `proc_pid_rusage` 的 process-level
counter。它不包含由 OS page cache 滿足、且未產生實體 disk I/O 的 logical read。
它也可能包含 tokenizer、prompt cache 或其他同 process 檔案讀取。
因此它是 request 級實體 I/O 的代理值，不是精確的 expert-file physical bytes。
正式分析應同時列出它與 `request_expert_bytes_read`，不可用其中一個取代另一個。

`request_expert_page_cache_*` 是第三種、expert-file-specific 但仍非 physical 的量測。
它要求 `resident + nonresident = classified` 與
`classified + unclassified = logical expert bytes`。`mincore` 只觀察 VM page residency；
APFS、storage-controller caches、kernel read-ahead 與 coalescing 不在其可見範圍。
Probe 的 system calls 是 observer overhead，所以開啟 probe 的 timing 只能診斷量測成本。

expert union reuse rate 只描述同一個 layer call 內有多少 assignments 指向重複
expert。它不是 I/O 節省率：sequential path 可能在第二次使用時命中 runtime cache，
而 union miss 也可能命中 OS page cache。速度與流量結論必須使用實際 bytes 和
wall-time 指標。

歷史 M1 artifact 的 `prefill_no_file_cache` 記錄當時的實驗組別。
`false` 代表 Control。
`true` 代表 full-layer Prefill 使用另一組 `F_NOCACHE` file descriptor。
目前 CLI 不再輸出這個欄位。

### Routing synchronization

`request_routing_sync_seconds` 量測 `mx.eval(indices)` 的 boundary。
MLX 使用 lazy evaluation。
這個 boundary 可以等待 attention、router 和其他 upstream graph work。

請勿把 `request_routing_sync_seconds` 稱為純 router kernel time。
純 GPU 分解需要 Metal capture 或 GPU counter。

### Memory

| 欄位 | 來源 |
| --- | --- |
| `active_memory_bytes` | `mx.get_active_memory()` |
| `cache_memory_bytes` | `mx.get_cache_memory()` |
| `peak_memory_bytes` | `mx.get_peak_memory()` |
| APP memory usage | runtime Python process RSS |

四個值具有不同語意。
報告必須寫出指標名稱。

`mx.clear_cache()` 只清理 MLX cache memory。
`mx.clear_cache()` 不會移除仍被引用的 active array。
`mx.clear_cache()` 也不代表 process RSS 立即下降。

### Prompt cache

| 欄位 | 意義 |
| --- | --- |
| `prompt_cache_reused_tokens` | 本次 request 重用的 prefix token。 |
| `prompt_cache_snapshot_seconds` | normal first／final prefill checkpoint、layer-major 或 atomic DSpark snapshot 時間。 |
| `prompt_cache_serialize_seconds` | quantized cache state serialize 時間。 |
| `prompt_cache_write_seconds` | safetensors 和 metadata write 時間。 |
| `prompt_cache_write_errors` | Write failure 數。 |
| `dspark_prompt_cache_source` | 最近一次 DSpark request 的 `disabled`、`none`、`memory` 或 `persistent`。 |

一般 format-4 prompt cache 在 prefill 期間只擷取 bounded first／final checkpoint state，
並在最後一個 token 之後 serialize／write；client generator 必須正常結束才保存。Snapshot
capture 進入 TTFT，完成後的 serialize／write 不進入 `request_seconds`。Atomic DSpark
snapshot 則在 target KV 與三個 context states 都到達同一個 `prompt[:-1]` boundary 時立即
寫入，所以其 snapshot／serialize／write 會進入該 request TTFT；persistent
acquisition／load 仍發生在 `RuntimeMetrics.start` 之前。

### DSpark

`dspark_accepted_tokens` 只計算接受的 draft token。
`dspark_committed_tokens` 也包含每個 round 的 bonus 或 correction token。
它只計算 request output budget 內可實際提交的 token，不包含驗證後不會輸出的 suffix。
`dspark_fallback_cost_ratios` 保存每輪
`(draft seconds + verification seconds) / (reference target-step seconds ×
(accepted draft tokens + 1))`。大於 1 的 round 觸發目前的 wall-time fallback。
`dspark_last_fallback_*` 保存最近一輪的 reference、speculative、break-even seconds
與 ratio；`dspark_fallback_would_trigger_rounds` 記錄依正常 policy 應停止的 round，
`dspark_fallback_triggered_rounds` 記錄實際停止次數。只有明確的 research control
`--no-dspark-fallback` 會讓兩者不同；正式 runtime 維持 fallback 啟用。
`dspark_round_trace` 逐輪保存 proposed／accepted／committed tokens、adaptive
selection、verification mode／position count 與 fallback decision，供長 decode
correctness audit。`dspark_block_verification_rounds` 與
`dspark_sequential_verification_rounds`、`dspark_hybrid_verification_rounds` 分開計數
verifier shape；
`dspark_last_verification_mode` 與 `dspark_last_sequential_position_seconds` 保存最近一輪
oracle 診斷，`dspark_last_hybrid_verification_positions` 保存最近一輪 hybrid block
位置數。CLI metrics JSON 另保存
`generated_token_ids`；正式 parity 必須直接比較這個序列或其 SHA-256。

```text
average_accepted_length = accepted_draft_tokens / rounds
average_committed_length = committed_tokens / rounds
```

DSpark throughput 比較必須使用 committed token。
DSpark 報告也必須列出 draft、verification、cache fork、cache replay 和 fallback。

storage-aware speculative profiling 使用下列欄位。

| 欄位 | 意義 |
| --- | --- |
| `dspark_output_budget_trimmed_tokens` | DSpark 已產生、但因本輪最多只能驗證 `remaining output tokens - 1` 而未送入 target 的 draft token。 |
| `dspark_hybrid_verification_rounds` | 使用 token-shaped hybrid target verifier 的 round 數。 |
| `dspark_hybrid_attention_layers` | Hybrid rounds 執行的 main-model attention layer 數。 |
| `dspark_hybrid_attention_token_calls` | Hybrid attention 的 one-token calls。 |
| `dspark_hybrid_ffn_token_calls` | Hybrid FFN HyperConnection／expand 的 one-token calls。 |
| `dspark_hybrid_moe_token_calls` | Hybrid router、shared／routed expert math 的 one-token calls。 |
| `dspark_verification_routed_expert_assignments` | 所有 target verification block 在 union 前的 assignments。 |
| `dspark_verification_expert_union_calls` | Target verification／replay 實際執行的 expert `get_many` acquisition 次數。 |
| `dspark_verification_expert_union_experts` | 所有 target verification block 的 per-call unique experts 加總。 |
| `dspark_verification_expert_union_reuse_rate` | Verification 每層一次 union 去除的 assignment 比例；block 與 hybrid 都適用。 |
| `dspark_verification_expert_union_misses` | initial verification union 的 runtime cache misses。 |
| `dspark_verification_expert_bytes_read` | initial target verification 的 logical expert bytes。 |
| `dspark_replay_expert_bytes_read` | rejected prefix replay 的 logical expert bytes。 |
| `dspark_target_expert_bytes_read` | verification 加 replay 的 logical target expert bytes。 |
| `dspark_draft_expert_bytes_read` | speculative rounds 的 DSpark cache logical expert bytes。 |
| `dspark_speculative_expert_bytes_read` | draft 加 target 的 logical expert bytes。 |
| `dspark_draft_expert_bytes_per_committed_token` | DSpark cache logical bytes 除以 speculative rounds committed tokens。 |
| `dspark_target_expert_bytes_per_committed_token` | target verification 與 replay logical bytes 除以 committed tokens。 |
| `dspark_speculative_expert_bytes_per_committed_token` | draft 加 target logical bytes 除以 committed tokens。 |
| `dspark_last_verification_expert_union_layer_ids` | 最近一輪 per-layer arrays 對應的 main model layer IDs。 |
| `dspark_last_verification_expert_union_by_layer` | 最近一輪每層的 unique expert 數。 |
| `dspark_last_verification_expert_misses_by_layer` | 最近一輪每層的 union miss 數。 |
| `dspark_hash_prefetch_requested_experts` | exact hash-layer unions 的 `(layer, expert)` 數，包含啟動時已 resident 的 expert。 |
| `dspark_hash_prefetch_cache_resident_experts` | prefetch 啟動時已 resident、沒有重讀的 expert 數。 |
| `dspark_hash_prefetch_bytes_read` | exact prefetch 實際讀入 scratch 的 logical bytes。 |
| `dspark_hash_prefetch_useful_bytes` | 被 committed input prefix 使用的 prefetched bytes。 |
| `dspark_hash_prefetch_wasted_bytes` | 只被 rejected suffix 使用的 prefetched bytes。 |
| `dspark_draft_page_cache_*` | DSpark cache logical reads 的 pre-read resident／nonresident／unclassified partition、calls 與 failures。 |
| `dspark_hash_prefetch_page_cache_*` | Exact hash-prefetch logical bytes 的 pre-read residency partition。 |
| `dspark_hash_prefetch_useful_page_cache_*` | Useful logical prefetch bytes 的 resident／nonresident／unclassified partition。 |
| `dspark_hash_prefetch_wasted_page_cache_*` | Wasted logical prefetch bytes 的 resident／nonresident／unclassified partition。 |
| `dspark_hash_prefetch_on_demand_expert_bytes_read` | target logical bytes 減去 hash prefetch bytes；包含未預取 layer 的 on-demand read。 |
| `dspark_hash_prefetch_plan_seconds` | 從 checkpoint `tid2eid` 建立 exact route plan 的 wall time，包含 MLX evaluation boundary。 |
| `dspark_hash_prefetch_wait_seconds` | verification/replay 等待 exact-prefetch future 的 wall time。 |
| `dspark_hash_prefetch_bytes_per_committed_token` | hash prefetch logical bytes 除以 speculative committed tokens。 |
| `dspark_last_hash_prefetch_union_by_layer` | 最近一輪三個 hash layer 的 planned union sizes。 |
| `dspark_adaptive_block_decisions` | request 中實際執行 adaptive selection 的 round 數。 |
| `dspark_adaptive_block_original_tokens` | output-budget cap 後、adaptive selection 前的 draft token 總數。 |
| `dspark_adaptive_block_selected_tokens` | selection 後送入 target verification 的 draft token 總數。 |
| `dspark_adaptive_block_trimmed_tokens` | original 減 selected；不是 rejected token。 |
| `dspark_adaptive_block_expected_committed` | 所有 selected candidate 的 confidence-based 預期 committed tokens 加總。 |
| `dspark_adaptive_block_requested_hash_experts` | selected prefixes 的 exact hash union `(layer, expert)` 總數。 |
| `dspark_adaptive_block_resident_hash_experts` | decision 時已在主 LFU cache 的 selected hash experts。 |
| `dspark_adaptive_block_missing_hash_experts` | decision 時不在主 LFU cache 的 selected hash experts。 |
| `dspark_adaptive_block_predicted_hash_bytes` | missing hash experts 乘 expert blob size。 |
| `dspark_adaptive_block_predicted_hash_bytes_per_committed_token` | predicted hash bytes 除以實際 speculative committed tokens。 |
| `dspark_adaptive_block_high_confidence_full_decisions` | 完整候選預期利用率至少 0.90，因護欄直接選完整 block 的 decision 數。 |
| `dspark_adaptive_block_storage_score_decisions` | 未觸發高信心護欄、由 storage score 選擇的 decision 數。 |
| `dspark_adaptive_block_full_block_decisions` | 不論 selection reason，最後選擇完整可用 block 的 decision 數。 |
| `dspark_adaptive_block_selected_length_counts` | `(selected draft tokens, decision count)` 配對，依長度排序；用來檢查長 decode 是否真的觸發不同 block size。 |
| `dspark_adaptive_block_plan_seconds` | exact route resolution 加 candidate scoring wall time。 |
| `dspark_last_adaptive_block_*` | 最近實際 decision 的 candidate lengths、預期 commits、requested／resident／missing experts、bytes、scores、selected score、selection reason 與完整候選預期利用率。 |

Hash exact prefetch 只有在 `--dspark --dspark-hash-prefetch` 同時啟用時執行。
停用時上述 prefetch 欄位為零，target path 維持原本的 on-demand 行為。
`useful` 的定義是該 `(layer, expert)` 至少出現在 anchor 加 accepted draft prefix；
`wasted` 是實際讀入 scratch、但只出現在 rejected suffix 的其餘集合。因此
`useful_bytes + wasted_bytes = hash_prefetch_bytes_read`。這是 committed-prefix I/O
分類，不是 predictor accuracy，也不能由 acceptance rate 推算。

scratch read 不做主 LFU admission。啟動 prefetch 時已 resident 的 expert 只在該
verification transaction 期間 pin，transaction 結束後解除。這避免 rejected-only
expert 改變長期 LFU frequency 或驅逐主 cache entry，但會增加最多 108 個 expert blob
slots 的實驗性 scratch 容量。

Adaptive block 只有在 `--dspark --dspark-adaptive-block` 同時啟用時執行；可與
hash exact prefetch 分開或一起使用。目前候選是 1、2、4 與 checkpoint 最大 block。
每個候選的預期 committed tokens 是 `1 +` 各位置 cumulative confidence survival
probability 的總和；score 是該值除以 `max(1, missing hash experts)`。這是前三個
hash layers 的精確 storage proxy，不是全 43 層 target bytes 或 wall-time predictor。
每輪先把可驗證 draft 限制為 `remaining output tokens - 1`。完整候選的預期 committed
tokens 除以 `draft tokens + 1` 若至少為 0.90，就由高信心護欄保留完整 block；否則才
以 storage score 選擇。DSpark 仍先計算完整 5-position draft，因此 output-budget 或
adaptive trimmed tokens 都不代表 draft compute 減少。

Research-only coherent candidate-path gate 不在 runtime metrics path。它以同一次 DSpark
backbone 的 conditional Markov distributions 建立 branch-4／beam-8 paths，並離線比較
joint normalized draft probability、exact hash union、target-LFU snapshot misses 與
sequential greedy accepted prefix。五組 first-round baseline 都接受 5/5；任何被 storage
weights 選中的 alternative 都降低 acceptance，因此 0/5 continuation、candidate 停止。
這個 artifact 沒有可解讀的 throughput／latency，runtime 也沒有 path-selector option。
Sampling 仍需要完整 selected-path proposal probability 與 rejection proof。

Learned-router predictor gate 同樣在 runtime 外。它用 6,000 個 layers 3--42 target
assignments 評估四個未訓練 DSpark taps；最佳 top-24 assignment／union recall 是
17.15%／33.03%，predicted union useful rate 11.15%。因此沒有執行 learned-layer
prefetch，這些 union bytes 只是 labels，不是 reads。任何 trained follow-up 必須另外列出
anchor、executed useful／wasted／missed bytes、on-demand correction、scratch capacity、
evictions、peak memory 與 request p95；不能從 offline recall 推算速度。

### 2026-08-26 full-model exploratory smoke

M2 Max 上的一個 7-token prompt、8-token greedy output 完成 off/on/on/off hash-prefetch
smoke，以及一個 adaptive follow-up。所有 run 的 prompt token hash 與 output token hash
都一致。Fixed hash-prefetch 的 deterministic logical counters 在兩次 on run 相同。

| 設定 | Rounds | Proposed | Committed | Target logical GB | Request logical GB | Prefetch GB | Useful rate | Wasted GB | Replay GB | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 5-token hash prefetch | 1 | 5 | 3 | 9.011 | 30.201 | 1.056 | 59.49% | 0.428 | 0.027 | 27.556 median |
| Adaptive hash prefetch | 2 | 2 | 3 | 6.030 | 26.872 | 0.896 | 73.13% | 0.241 | 0.080 | 28.195 |

相對 fixed 5-token prefetch，adaptive run 少讀 249 個 request-level logical expert
blobs，其中 target 少 223 個；wasted prefetch 少 14 個 blobs，但 replay 多 4 個。
這能支持 selector、correctness 與 byte-accounting gate，不能支持速度採用：adaptive
只有一次 warm run，OS page cache 未清除，且 speculative round 數已改變。Hash
prefetch 的兩次 timing 中位數也受第一個較冷的 off run 影響；回程 on/off pair 方向
相反。原始數值與限制位於
[`2026-08-26-hash-exact-prefetch-smoke-m2-max.json`](benchmarks/2026-08-26-hash-exact-prefetch-smoke-m2-max.json)
與
[`2026-08-26-adaptive-block-smoke-m2-max.json`](benchmarks/2026-08-26-adaptive-block-smoke-m2-max.json)。

### 2026-08-26 五類 adaptive calibration

五類 128-token、8-output-token 的第一輪 fixed／adaptive／adaptive／fixed ABBA
否決了原始 storage-only selection policy。雖然五類全部保持 greedy parity，跨 workload
中位數的 request time 增加 2.013%、Decode 降低 14.843%，request logical expert
bytes 增加 0.593%；沒有任何 workload 降低整個 request logical bytes。主要原因是
score 只獎勵前三個 hash layers 的 missing-byte 減少，卻把實際會 5/5 接受的高信心
block 縮成 1 或 2 tokens，增加後續 target 工作。Tool-like 還暴露舊 path 驗證超出
剩餘 output budget 並高估 committed tokens。

修正加入 0.90 完整候選預期利用率護欄、output-budget cap 與實際 committed 計數。
第二輪相同 ABBA 的 10 個 adaptive timed runs 全都選擇完整 5-token block、接受 5、
committed 6；五類 fixed／adaptive logical counters 完全一致。跨 workload request-time
差異中位數是 +0.0495%，Decode 是 +0.2164%，兩者都只表示 candidate 回到 control
等價範圍，不是速度改善。

7-token 低信心 A/B/B/A 仍由 storage score 每輪選 1 token，兩個 adaptive runs
各裁掉 2 個 output-budget suffix，並正確回報 3 committed tokens。相對 fixed，
deterministic request logical bytes 少 11.02%，target 少 33.09%；因 prompt 太短且 OS
page cache 未控制，不採用 timing。原始 artifacts 是
[`失敗第一輪`](benchmarks/2026-08-26-adaptive-block-calibration-wave1-m2-max.json)、
[`修正第二輪`](benchmarks/2026-08-26-adaptive-block-calibration-wave2-m2-max.json)
與
[`repair validation`](benchmarks/2026-08-26-adaptive-block-repair-validation-m2-max.json)。

五類校準可用下列命令重建；prompt generator 與 manifest 固定 model revision、seed、
文字 SHA 和 token SHA。正式保存前仍須使用新的 artifact 檔名，不可覆寫既有 wave。

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/benchmark_dspark_adaptive.py \
  --model ~/.dsmodel/deepseek-v4-flash-0731.dsv4 \
  --prompt-manifest docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json \
  --raw-directory scratch/adaptive-block-wave \
  --output scratch/adaptive-block-wave.json \
  --python .venv/bin/python \
  --max-tokens 8
```

### 2026-08-26 4K／256 adaptive decision survey

下一個 gate 重建了歷史 R0 的五類 4,096-token prompts；五個 prompt token hash
全部匹配 2026-08-10 R0 artifact。每類以 fresh process、無 persistent prompt cache、
無 warmup 執行一次 adaptive 256-token decode。五個 output token hash 與 generated
token count 也全部匹配歷史 fixed R0 baseline。

五類合計 54 個 adaptive decisions，54 個全部由 high-confidence guard 選擇完整可用
block，storage-score decision 是 0。Selected-length 分布是 5 tokens 共 53 次、1 token
共 1 次；後者是中文 workload 最後一輪經 output-budget cap 後的完整 block，不是
storage-aware 縮短。Repeated、Code、Mixed-math 在第一輪後 fallback，Tool-like 在第
8 輪後 fallback；只有 Chinese-technical 完成 43 個 speculative rounds 而未 fallback。

因此這個 R0 矩陣沒有 candidate/control 策略差異，依 stop gate 不執行後續
fixed／adaptive A/B/B/A。單次 survey 的 timing、process disk bytes 與 memory 不作
效能結論。完整 counters、歷史 parity audit 與 stop decision 位於
[`4K／256 survey artifact`](benchmarks/2026-08-26-adaptive-block-4k256-survey-m2-max.json)，
固定 prompts 位於
[`4K manifest`](benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json)。

目前結論是：原始 score policy 已拒絕；修正版仍保持預設關閉。R0 五類 4K／256
workloads 已通過 correctness audit，卻完全沒有觸發 storage score。下一次 paired gate
必須先固定一組可重現、能跨多 round 觸發 storage-score selection 的低／中 confidence
workload，再量測整個 request bytes、速度與 memory。

### 2026-08-26 低信心 discovery 與 parity stop gate

後續建立五個固定 instruction suffix 的 4,096-token prompts。16-output screening 中，
5/5 都由 storage score 選 1 token，也全都在第一個 round 觸發正常 fallback；其中
`balanced_choice` 只產生 5 tokens 就 EOS。重跑 `storage_sentence` 得到相同 output hash，
speculative cost 是 0.837 秒、break-even 是 0.643 秒，ratio 1.301，因此正式 runtime
應保持 fallback。明確加上 research-only `--no-dspark-fallback` 後，同 prompt 可完成
8 個 storage-score rounds且保持相同 16-token hash；5/8 rounds 依正常 policy 仍應停止。

`random_hex` 的 no-fallback 4K／256 run 完成 86 rounds，其中 66 次 storage-score、
20 次 high-confidence full。Selected length 1／2／4／5 分別出現 55／6／1／24 次，確實
建立了原 R0 矩陣缺少的多輪 confidence 分布。但 fixed 與 adaptive 的 256-token output
hash 不同，因此所有 request time、Decode、logical bytes 與 memory 差異全部無效。

32-token exact-ID 重現把問題進一步定位：layer-major normal 相對 sequential-prefill
normal 在 token index 6 分歧；fixed 與 adaptive block verification 都在 index 15 離開
sequential reference，兩者再於 index 17 互相分歧。從共同 prefix 重建的 block diagnostic
顯示，position 0 的 sequential/block top token 相同且 margins 為 6.375／6.25；position 1
的 sequential top-2 是 token 20400／15207、margin 0.125，而 block 的 token 15207／729
logits 同為 26.875、margin 0，最大 absolute logit delta 是 1.9375。證據符合 near-tie
對 block shape／低精度數值敏感，不支持把這組 output 差異寫成 adaptive 改善。

因此 adaptive performance calibration 再次停止。沒有採用 heuristic margin threshold，
因單一 diagnostic 無法建立全 workload 的安全數值界線。DSpark、hash prefetch 與
adaptive block 仍預設關閉；fallback 仍預設啟用。主要失敗 artifact 是
[`4K／256 adaptive`](benchmarks/2026-08-26-adaptive-block-random-hex-4k256-no-fallback-m2-max.json)、
[`4K／256 fixed`](benchmarks/2026-08-26-fixed-block-random-hex-4k256-no-fallback-m2-max.json)、
[`4K／32 exact tokens`](benchmarks/2026-08-26-random-hex-4k32-fixed-adaptive-failed-m2-max.json)
與
[`block diagnostic`](benchmarks/2026-08-26-dspark-block-parity-diagnostic-m2-max.json)。

### 2026-08-26 sequential verification oracle

為排除 adaptive policy 與 rejected-prefix replay 本身，runtime 新增預設關閉的
`--dspark-sequential-verification` correctness oracle。它仍在每輪建立一次 cache fork，
但 target verification 與被拒絕後的 committed-prefix replay 都一次只執行一個 token。
Oracle 明確不允許 hash exact prefetch，避免 speculative scratch 造成另一條 expert
execution path；因此這不是 prefetch 或 block-verification 效能候選。

同一個 `random_hex` 4K／32 prompt 上，fixed oracle 的 13/13 rounds 與 adaptive oracle
的 17/17 rounds 都使用 sequential verification。兩者的完整 generated token IDs 與
sequential-prefill normal reference 完全一致，output hash 都是
`e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`。Adaptive oracle
仍有 16 個 storage-score decisions，selected length 分布是 `[[1, 15], [2, 1]]`，因此
exact parity 不是因 selector 沒有作用。

這個結果排除了該 workload 的 adaptive prefix policy 與 sequential replay wiring，並與
共同-prefix diagnostic 一起把 correctness boundary 縮小到 block-shaped target
verification。Fixed oracle 有 11/13 rounds、adaptive oracle 有 5/17 rounds 依正常
fallback policy 應停止；no-fallback 只用來讓 32-token trace 完成。所有 timing、bytes
與 memory 仍是無效能結論的診斷數據。

Oracle artifacts 位於
[`fixed sequential oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-fixed-oracle-m2-max.json)
與
[`adaptive sequential oracle`](benchmarks/2026-08-26-random-hex-4k32-sequential-adaptive-oracle-m2-max.json)。
下一個 correctness gate 不是量速度，而是讓 block verifier 對 near-tie workload 保持
sequential-equivalent token IDs；在取得跨 workload 證據前，不採用 margin heuristic，
也不改變任何正式預設值。

### 2026-08-27 layer-wise verifier diagnosis

相同 exact common prefix 的逐層診斷先驗證 instrumentation 不會改變 sequential 或
block logits，再比較 43 層的 post-attention、FFN input、router top-k 與 post-layer
states。兩個位置的 embeddings 完全相同，但第 0 層 `LocalAttention` 後已非 exact：
position 0／1 的最大 absolute delta 分別是 0.0009765625／0.00048828125。第 0 層
router 的 selected expert IDs、順序與 weights 仍完全相同。

Position 1 到第 11 層才首次改變 router 順序、第 12 層才首次改變 expert set；position 0
到第 16 層才首次改變 set。到第 42 層，post-layer 最大 delta 已增至 3.0／3.75，並重現
既有 position 1 top-token 分歧。這把最早觀測到的 numerical boundary 從整個 block
verifier 縮到第 0 層 attention branch，learned router 是後續 amplifier，而不是起點。

這仍是單一 two-token、單一 cache state 的 correctness diagnostic。它沒有證明某個
attention 子算子單獨造成差異，也沒有建立可接受 tolerance；因此不重啟 adaptive
performance gate。Artifact 位於
[`layer parity diagnostic`](benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json)。

### 2026-08-27 layer 0 component diagnosis

Component follow-up 使用相同 exact prefix、anchor 與 two-token block，並要求攔截後的
sequential／block logits 逐值匹配各自未攔截 reference。兩個位置的 layer input、
HyperConnection collapsed value、attention norm、`wq_a`、`q_norm`，以及 KV projection、
norm 與 RoPE 都 exact。嚴格 stage order 最先在 HyperConnection `post`／`combine` 出現
極小 shape-dependent 差異；`wq_b` 在 exact input 下也分別有 3／1 個值不同。Attention
output 的最大 absolute delta 是 0.001953125／0.0001220703125，output projection 後
`post_attention` 精確重現逐層 artifact 的 0.0009765625／0.00048828125。

Cache audit 顯示 sequential 是兩次 one-token in-place update、沒有 mask；block 是一次
two-token concatenate update，使用 2x129 float32 mask。Raw fetched cache 因長度 128／129
不同而不可直接逐值比較；轉回 temporal order 後，共同 128-token suffix 的 65,536 個值
完全相同。HyperConnection、Q projection、mask、cache layout 與 SDPA execution shape
仍同時不同，因此 artifact 不把結果歸因到單一 kernel。

這是 correctness diagnostic，沒有 timing、expert-I/O 或 throughput 採用結論。後續
hybrid gate 先以 one-token attention／block-shaped FFN-MoE，再逐步縮小剩餘的 FFN
shape；結果記錄於下一節。
Artifact 位於
[`layer 0 attention component diagnostic`](benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json)。

### 2026-08-27 hybrid verifier correctness 與 composition gate

第一個 hybrid diagnostic 讓 attention 逐 token 執行、FFN／MoE 保持 block shape。在
同一 exact cache state，兩個位置的 top token 都回到 sequential target；position 1
恢復 near-tie token 20400，但完整 logits 仍非 exact。`random_hex` 4K／32 的多 round
follow-up 隨後在 index 15 再次分歧：normal token 是 20400，hybrid v1 是 15207，
32 tokens 中有 17 個 mismatch。這證明單狀態 gate 不足以排除 cache 內累積 drift。

Layer 0 FFN component diagnostic 顯示 router IDs／scores、routed selected outputs、
routed reduction 與 MoE output exact；FFN HyperConnection `post`／`combine` 最先不同。
Hybrid v2 因此把 FFN HyperConnection 與 final expand 也改為 token-shaped，但保留
block-shaped MoE。五組 4K／32 survey 中，`random_hex`、`creative_metaphor`、
`balanced_choice` exact；`storage_sentence` 在 index 16、`multilingual_choice` 在 index 6
分歧。Correctness gate 再次停止。

Hybrid v3 將 router、shared expert 與 routed expert math 也改為 token-shaped；每層仍先
收集全部 route IDs，只 acquire 一次 expert union。五組 normal／fixed 完整 token
序列全部 exact；四組產生 32 tokens，`balanced_choice` 在 5 tokens EOS。Fixed v3 的
per-layer union assignment reuse rate 範圍是 37.08% 至 46.90%，且所有 run
`fallback=false`（research control）。這解除 hash／adaptive correctness 前置阻擋，
但不等於 block grouped-QMM 或 throughput gate 通過。

與 exact hash prefetch 組合後，五組 normal／fixed 仍全部 exact，且每組
`useful + wasted = hash_prefetch_bytes_read`。Fixed useful rate 範圍是 23.53% 至
56.23%。再加入 adaptive selector後，五組 normal／fixed／adaptive 仍全部 exact；
63 個 decisions 的 selected length 1／2／3／4／5 分布是 56／4／1／1／1。Adaptive
useful rate 上升到 65.22% 至 91.31%，但 union reuse rate 降到 15.59% 至 23.98%。
這是縮短 speculative prefix 的明確 tradeoff：減少 rejected-only hash prefetch，同時
減少同一 layer union 內可重用的 expert assignments。

主要 artifacts：

- [`single-state hybrid diagnostic`](benchmarks/2026-08-27-dspark-hybrid-verifier-diagnostic-m2-max.json)
- [`hybrid v1 multi-round stop`](benchmarks/2026-08-27-dspark-hybrid-random-hex-4k32-m2-max.json)
- [`layer 0 FFN component diagnostic`](benchmarks/2026-08-27-dspark-layer0-ffn-component-diagnostic-m2-max.json)
- [`hybrid v2 five-workload stop`](benchmarks/2026-08-27-dspark-hybrid-v2-discovery-4k32-m2-max.json)
- [`hybrid v3 five-workload gate`](benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json)
- [`hybrid v3 + hash`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json)
- [`hybrid v3 + hash + adaptive`](benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json)

上述 survey 沒有 warmup、沒有清除或控制 OS page cache，每個模式只有一 run，且
明確停用正常 fallback 以完成 correctness trace。Artifact 全部標記
`formal_performance_result=false`；request time、process disk bytes、memory 與功耗都
不是採用證據。DSpark、hybrid、hash prefetch 與 adaptive 仍預設關閉。

### 2026-08-27 verifier union／execution-shape tradeoff

在 validated bypass policy 下，128-token `repeated` prompt 以 8 greedy outputs 執行
四波 normal／sequential／grouped／hybrid Latin square，再執行三個 probe-on observer
runs。19 條 token sequence 全部匹配既有 SHA-256；每個 DSpark run 都是一次完整接受的
5-token draft block、6 committed tokens、零 replay。Observer 的 main／draft partitions
全部閉合，且 timing 不放入中位數。

| Mode | Union calls | Union experts | Misses | Target bytes | Expert read | Verification |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential | 258 | 1,548 | 195 | 2.607 GB | 0.4196 s | 0.8524 s |
| Grouped | 43 | 660 | 146 | 1.952 GB | 0.2854 s | 0.5331 s |
| Hybrid v3 | 43 | 675 | 171 | 2.286 GB | 0.3460 s | 1.0001 s |

Hybrid 相對 sequential 把 acquisition calls 降低 83.33%、union-expert accounting
降低 56.40%、target bytes 降低 12.31%，但 verification time 反而增加 17.34%。相對
grouped，hybrid verification time 增加 87.59%，超過預先宣告的 20% material line。
因此 one-per-layer acquisition／dedupe 合約通過，但沒有抵銷目前所有 token-shaped
target execution 的總成本。

Grouped 與 hybrid 雖有相同 output tokens，內部 union experts 仍是 660／675；兩條 path
也同時改變 attention、HyperConnection、router、shared expert 與 routed expert shapes。
所以這個 87.59% 不能稱為 QMM-only timing。Grouped 已在低信心 multi-workload gate
失去 exact parity，不能因本短案例較快而採用；hybrid 也已在 4K／32 end-to-end gate
被拒絕。Corrected artifact 位於
[`union tradeoff`](benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json)。
第一輪 attribution label 過度窄化成 QMM，已保留為
[`label pilot`](benchmarks/2026-08-27-verifier-union-tradeoff-attribution-label-pilot-m2-max.json)，
其 timing 不作結論。

### 2026-08-27 atomic DSpark prompt-context reuse gate

Default-off `--dspark-prompt-cache` 以獨立 format-3 bundle 原子保存 target KV 與三個
DSpark context states。Installed-model gate 固定 128-token `repeated` prompt、8 greedy
outputs、hybrid v3、fallback/hash/adaptive off、persistent cache on 與 bypass policy；先在
runtime A 執行兩次，再關閉並由 runtime B 還原。

| Request | Source | Reused | Main logical bytes | Draft logical bytes | Combined logical bytes | Request | TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime A first | `none` | 0 | 28.958 GB | 0.455 GB | 29.413 GB | 6.979 s | 5.750 s |
| Runtime A second | `memory` | 127 | 2.286 GB | 0 | 2.286 GB | 1.214 s | 0.117 s |
| Runtime B restart | `persistent` | 127 | 9.666 GB | 0.455 GB | 10.121 GB | 2.814 s | 0.751 s |

三條 output token SHA-256 都是
`03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3`。
Memory 與 persistent reuse 的 combined logical bytes 分別下降 92.23% 與 65.59%，兩者
都通過至少 -50% gate；metadata 的 mode／format／revision／layers／127 tokens 與 data
file 也全部 exact。這是三次 request 的 functional integration gate，不是 balanced
performance result；memory hit 保留 runtime A expert slots，restart hit 使用新 expert
caches，因此 timing 不能互相比較或升格成 adoption claim。Artifact 位於
[`atomic prompt-context snapshot`](benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json)。

### 2026-08-27 block prompt-cache partial-restart gate

Normal persistent format 4 使用完整 model／RoPE／KV／attention contract 與 128-token
content-address chain。Gate 從 pinned `repeated-128.txt` 建立兩個 453-token prompts；兩者
有 442 個 exact common prefix tokens，但在第一個 request 的 final prefill checkpoint
之前分岔。Runtime A 建立 cold cache，Runtime B 重啟後處理分岔 prompt，Runtime C 以
persistent cache disabled 執行同一分岔 cold reference。

| Request | Reused | Output token/hash | Logical expert bytes | Request | TTFT |
| --- | ---: | --- | ---: | ---: | ---: |
| Runtime A cold seed | 0 | 36,363／`6c094c63…20d51` | 146,902,351,872 | 28.487 s | 28.466 s |
| Runtime B restart branch | 128 | 36,363／`6c094c63…20d51` | 125,070,213,120 | 23.948 s | 23.918 s |
| Runtime C isolated cold branch | 0 | 36,363／`6c094c63…20d51` | 146,848,874,496 | 29.537 s | 29.536 s |

Restart 與 isolated cold branch 的 token ID／SHA-256 exact。共享的 128-token payload 是
7,002,850 bytes；data／metadata SHA 在 persistent hit 前後不變、同 key 只有一份 payload，
access sidecar reuse count 由 0 增為 1。第一次 integration run 曾找到 mutable MXFP8 chunk
list 汙染 checkpoint 的 bug；修正 list snapshot 並新增 regression 後才重跑通過。

Artifact SHA-256 是
`678034d8a68278f87a9071fa65f662140a8479e9d4c591e0f189c5db3d367620`。
這是 functional gate；三條 timing 沒有 balanced OS-cache/repeated-wave control，且
persistent load 發生在 runtime metrics 起點之前，不能把表中的差異宣稱為 speedup。
Format 4 分享 cumulative immutable checkpoints，未實作 per-layer KV delta dedupe。

### 2026-08-27 native MTLIO ownership/copy-path gate

Standalone Swift probe 對 canonical 13,369,344-byte expert ranges 執行四-worker
`preadv`、MTLIO bytes、shared `MTLBuffer` 與 private `MTLBuffer`。每種方法各跑
1／6／32／128 ranges，交錯方法順序並使用不同 installed layer files。16/16 rows
的 candidate/reference SHA-256 exact；shared/private external event signal → GPU wait
與 cancellation destination admission 也通過。

| Expert count | `preadv` GiB/s | MTLIO bytes GiB/s | Shared GiB/s | Private GiB/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.824 | 4.963 | 5.021 | 2.995 |
| 6 | 5.968 | 5.888 | 5.861 | 3.553 |
| 32 | 5.960 | 5.949 | 6.007 | 5.636 |
| 128 | 6.054 | 6.078 | 6.041 | 6.026 |

單 expert bytes/shared 的 p50 direction 相對 `preadv` 改善 21.61%／22.22%；但六
expert p95 分別增加 51.44%／52.38%，32／128 aggregate improvement 最高只有 0.78%。
Private mode 的 CPU byte validation 明確需要 GPU blit；shared 的 direct CPU hash 不需
這個 copy，artifact 中的 shared blit只用來驗證 GPU event visibility。

所有 selected ranges 在 run 前至少 99.9885% nonresident，但 OS page cache 標記為
`not purged`，方法使用不同 layer files，而且沒有 repeated balanced waves。這是 native
correctness／ownership gate 加 exploratory timing，不是正式 performance 或 physical
SSD 結果。Installed MLX 0.32.0 又缺少 external `MTLSharedEvent` public dependency
handoff，所以不建立 runtime prototype。Artifact 位於
[`MTLIO ownership gate`](benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json)。

### 2026-08-27 staged `w13`／`w2` runtime gate

Fixed-arena probe 先用 1／4／8 rows 各跑 12 paired samples。所有 canonical region 與
float32 output hashes exact；4／8-row complete wall 中位數分別 -10.74%／-13.67%，
hidden `w2` read 80.98%／87.90%，因此允許 internal default-off runtime prototype。
這些 shape 都把 rows 集中到單一 routed expert，只是 optimistic component evidence。

Runtime gate 使用 128-token `repeated` prompt、32 greedy outputs、四個 fresh-process
control/staged pairs，交錯 ABBA order，停用 persistent cache、layer-major prefill 與
DSpark，並使用 explicit expert-file bypass。

| Metric | Control median | Staged median | Paired median change |
| --- | ---: | ---: | ---: |
| Request | 9.9083 s | 10.1274 s | +2.17% |
| Decode | 7.1176 tok/s | 6.8164 tok/s | -4.26% |
| Decode p95 | 208.29 ms | 212.72 ms | +2.12% |
| Peak MLX memory | 24.665 GB | 24.665 GB | 0% |
| Logical expert bytes | 40.121 GB | 40.121 GB | 0% |
| Evictions | 1,849 | 1,849 | 0% |

八條 32-token outputs exact。每個 staged run 完成 1,024 reads，`w13`／`w2` bytes
分別是 9,126,805,504／4,563,402,752，精確閉合 canonical expert blob budget；每一 pair
的 logical bytes 與 evictions 都未增加。Performance gate 要求 request 至少 -5% 或
Decode 至少 +5%，所以 +2.17%／-4.26% 明確失敗。Candidate 保持 default off 且不提供
CLI／server／APP 開關。Artifact 位於
[`fixed-arena gate`](benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json) 與
[`runtime gate`](benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json)。
單一短 workload 與四 pairs 不是正式效能證據；`F_NOCACHE` 也不是 physical SSD counter。

### 2026-08-27 adaptive full-layer／selective prefill gate

第一階段使用五種 exact 4,096-token prompts，各執行 full-layer reference 與 route-trace
fresh process。所有 reference／trace output IDs exact；每個 trace 都隔離出 42 個
24,570-assignment layer histograms。70%／80%／90% threshold 的跨 workload prefill
expert-byte estimate 中位數分別是 -23.19%／-33.38%／-34.24%，且至少 -10% 的
workloads 數是 4／5、5／5、5／5，因此三者都通過 byte eligibility。Trace timing 因
observer synchronization 不可解讀。

Conditional runtime prototype 保留完整 256-row destination 與既有 `gather_qmm` shape，
只讀 exact union rows 到原 offset。它在 attention 後規劃 routes，因此不執行目前的
next-layer prefetch overlap。Installed-model early gate 使用 `repeated` 4K／2 greedy
outputs、兩組反向 fresh-process pairs、persistent cache off、DSpark off 與 explicit
expert-file bypass。所有四條 outputs 都是 `[1950, 1950]`，token SHA-256 是
`dd22c238773ee5642280c221b7a7a51094e6dcec6882cce0f749378ee01e4891`。

| Metric | Full-layer control median | Adaptive median | Paired median change |
| --- | ---: | ---: | ---: |
| Request | 44.4157 s | 51.8707 s | +16.78% |
| TTFT | 44.1908 s | 51.6443 s | +16.87% |
| TTFT p95 | 44.1999 s | 51.7207 s | +17.02% |
| Logical request expert bytes | 149.001 GB | 49.493 GB | -66.78% |
| Peak MLX memory | 21.407 GB | 18.028 GB | -15.78% |
| Expert read wall sum | 22.8102 s | 7.3113 s | observational only |
| Routing sync wall | 0.1206 s | 2.0613 s | observational only |
| Adaptive plan wall | 0 | 2.5471 s | observational only |

每個 candidate 的 42 層都是 selective：union／read experts 同為 3,309，actual adaptive
batched bytes 精確為 44,239,159,296；另有 393 個一般 expert-cache misses，所以 request
total 是 49,493,311,488 bytes。Control 的 10,752 full-prefill rows 加相同 393 misses
是 149,001,338,880 bytes；avoided bytes 精確閉合為 99,508,027,392。Control 有 41 個
prefetched-layer hits，candidate 是 0。

Byte 與 memory gates 通過，但必要的 TTFT -5% 與 p95 不超過 +5% 都失敗。`repeated`
的 42 個 union 全低於 70%，所以 70%／80%／90% 三門檻執行完全相同的 decisions；這個
workload 已使所有 threshold 的 per-workload p95 gate 失敗。依預先停止規則，完整
五-workload threshold matrix 與 cached observer 不執行，decision 是
`stop_runtime_candidate_keep_default_off`。Artifact 位於
[`route-union gate`](benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json)
與 [`runtime early stop`](benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json)。
這是指定 M2 Max、單一 runtime workload 的 exploratory stop，不是 physical SSD、power
或一般化效能結果。

### 2026-08-27 DSpark reduced-cache pilot

為先降低 DSpark 獨立 cache 的 memory footprint，96-slot candidate 以三層 × 五位置 ×
六 experts = 90 assignments 作單輪 working-set 上界。單一 `storage_sentence` 4K／32
使用 normal reference 與 768/96/96/768 fresh-process 順序；DSpark 使用 hybrid v3、
confidence threshold 0，並停用 fallback 以完成 trace。

五次 generated token IDs 全部 exact，且 output hash 匹配既有 hybrid v3 artifact。
96 slots 的 peak MLX memory 中位數由 28,601,817,360 降到 27,653,809,943 bytes，
降低 3.31%（0.883 GiB）；代價是 draft expert reads 由 2,847,670,272 增至
3,623,092,224 bytes（+27.23%），hit rate 由 61.55% 降到 51.08%，並產生 175
evictions。整體 speculative expert reads 增加 0.515%，因 target verification bytes
遠大於 draft cache 差異。

Request-time paired change 是 +0.494%／-0.120%；OS page cache 未控制，因此沒有
timing 採用結論。這個 candidate 通過初始 workload 的 exactness 與 memory-direction
gate，但 768-slot 預設不變。原始資料位於
[`768 vs 96 slots`](benchmarks/2026-08-27-dspark-slots-768-vs-96-storage-4k32-m2-max.json)。

後續三 workload 4K／32 screen 全部 exact 且 peak memory 都較低，但降幅為 -0.40% 至
-3.31%，draft reads 增加 16.56% 至 27.23%。組合資料位於
[`three-workload summary`](benchmarks/2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json)。

預先宣告的 `random_hex` 4K／128 long-decode gate 使用 fresh normal／768／96 single
pair。三條 128-token sequence exact；96-slot peak memory 從 29,885,480,224 降到
27,688,162,682 bytes（-7.35%），all speculative bytes/committed token 增加 0.91%，
但 draft bytes/committed token 從 32,786,724.57 增至 60,055,942.10 bytes（+83.17%），
超過 +50% 停止線，並產生 470 evictions。因此停止此 candidate、保留 768-slot
預設。單一 pair 的 request-time -0.14% 不作效能結論。原始資料位於
[`4K/128 long-decode stop`](benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json)。

### 2026-08-27 expert-file page-cache residency proxy

Default-off `--expert-page-cache-probe` 先以 `code` 128/32 的
control/probe/probe/control 驗證。四條 token sequence exact；兩個 probe runs 各把
90,323,288,064 logical expert bytes 完整分類，6,756 calls、零 failure、零
unclassified。Median partition 是 32.10% resident、67.90% nonresident。Probe request
time 相對 control median +3.89%、reported decode throughput -10.77%；這是 observer
overhead，不是 runtime performance 結論。Artifact 位於
[`probe contract`](benchmarks/2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json)。

短 `code` hash composition 的 logical 與 nonresident useful rate 都是 100%，只證明
plumbing。4K `balanced_choice` normal/hash single pair 隨後同時覆蓋 useful 和 wasted：

| Prefetch partition | Logical bytes | Resident before read | Nonresident before read |
| --- | ---: | ---: | ---: |
| Useful | 802,160,640 | 120,455,168 | 681,705,472 |
| Wasted | 2,607,022,080 | 989,462,528 | 1,617,559,552 |
| Total | 3,409,182,720 | 1,109,917,696 | 2,299,265,024 |

兩條 output exact，所有 main、draft、prefetch、useful／wasted partitions 都零 failure、
零 unclassified 並完整閉合。Logical useful rate 是 23.53%，nonresident useful rate 是
29.65%；1.618 GB wasted bytes 在讀取前仍為 nonresident。Artifact 位於
[`4K useful/wasted residency coverage`](benchmarks/2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json)；
短 composition 位於
[`code 128/32`](benchmarks/2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json)。
OS page cache 未控制且 probe 會改變 timing，因此這些不是正式效能結果，也不支援啟用
hash prefetch。

### 2026-08-27 explicit expert-file bypass-policy gate

系統帳號無法執行 system-wide `purge`，所以本輪沒有把未控制的首個 run 標成 cold。
Runtime 新增 default-`cached`、research-only `bypass` descriptor policy；bypass 對 main
與 DSpark expert files 設定 `F_NOCACHE`、停用 read-ahead，並要求 APFS 4,096-byte
destination／offset／length alignment。它不驅逐啟用前已 resident 的 pages。

Installed-range contract 對 layer 42 六個完全 nonresident expert blobs 執行 bypass 後
cached control。六組 byte hashes exact；每個 13,369,344-byte range 在 bypass 後仍是
0 resident bytes，cached read 後則全部 resident。原本一次預選 ranges 的 pilot 被前一
cached control 的 read-ahead 影響，因此改成每個 range just-in-time recheck；兩個
artifacts 都保留。這個 contract 驗證 descriptor 與 installed layout，不是效能結果。

通過 contract 後，`random_hex` 4K／32 使用 fresh process、persistent cache off、
layer-major off、fallback off、probe off 的三波 Latin square；另外三個 observer runs
開 probe，只做 accounting，不放入中位數。12 條 output token IDs 全部 exact，SHA-256
都是 `e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`；observer 的
main／draft／prefetch／useful／wasted partitions 全部閉合，零 failure、零
unclassified。

| Mode | Request median | Decode median | Peak memory median | Process disk bytes/token median |
| --- | ---: | ---: | ---: | ---: |
| Normal | 85.864 s | 3.032 Tok/s | 26.799 GB | 9.615 GB |
| Fixed hybrid + hash | 101.487 s | 1.223 Tok/s | 29.227 GB | 10.878 GB |
| Adaptive hybrid + hash | 90.171 s | 2.256 Tok/s | 29.374 GB | 9.674 GB |

預先門檻要求 request time 至少 -5%、decode 至少 +5%、peak memory 不超過 +15%、
process disk bytes/token 不超過 +5%；adaptive 另要求 speculative logical
bytes/committed token 不高於 fixed +5%。Fixed 相對 normal 是 request +18.19%、decode
-59.67%、memory +9.06%、disk/token +13.14%。Adaptive 是 request +5.02%、decode
-25.58%、memory +9.61%、disk/token +0.61%；其 speculative bytes/committed token
雖比 fixed 低 42.96%，仍未補回 token-shaped target-math 成本。兩個候選都拒絕，所有
實驗開關與 cached policy 預設不變。

完整資料位於
[`bypass full-model gate`](benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json)、
[`installed-range contract`](benchmarks/2026-08-27-expert-file-cache-bypass-contract-m2-max.json)
與
[`selection-race pilot`](benchmarks/2026-08-27-expert-file-cache-bypass-selection-race-pilot-m2-max.json)。
這是一個固定 workload／32-output candidate gate，`formal_performance_result=false`；
它不能推廣成所有 workload 的速度，也不能證明 storage-controller 或 device cache 為冷。

## 目前瓶頸證據

2026-08-10 的 R0 baseline 使用五種固定 prompt。
每種 prompt 都測試 4,096、8,192 和 14,363 token。
每個 run 都產生 256 個 output token。

| Prompt token | TTFT 中位數 | Prefill 中位數 | Decode 中位數 | Expert bytes 中位數 | Misses 中位數 | Evictions 中位數 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 21.611 s | 189.53 Tok/s | 8.00 Tok/s | 473.970 GB | 24,700 | 23,548 |
| 8,192 | 39.124 s | 209.39 Tok/s | 7.76 Tok/s | 477.393 GB | 24,956 | 23,804 |
| 14,363 | 67.466 s | 212.89 Tok/s | 7.43 Tok/s | 482.593 GB | 25,345 | 24,193 |

每種 prompt 和長度只有一個 run。
這些中位數表示跨 prompt 的中心值。
這些結果不表示重複執行的變異。

4K repeated prompt 的 Decode 是 12.22 Tok/s。
4K repeated prompt 有 1,535 次 miss 和 383 次 eviction。
四種非 repeated prompt 的 Decode 是 6.44 至 8.28 Tok/s。
四種非 repeated prompt 有 24,307 至 28,765 次 miss。
四種非 repeated prompt 有 23,155 至 27,613 次 eviction。

因此，repeated prompt 低估一般 Decode 的 SSD 與 slot 壓力。
後續效能決策不能只使用 repeated prompt。

### Metal boundary 結果

R1 對 repeated 4K、Tool-like 4K 和 Tool-like 14K 執行
`Metal System Trace`。

| Workload | Prefill GPU busy | Decode GPU busy | Decode GPU idle |
| --- | ---: | ---: | ---: |
| Repeated 4K | 57.6% | 57.8% | 2.604 s |
| Tool-like 4K | 60.7% | 35.5% | 7.968 s |
| Tool-like 14K | 70.2% | 37.1% | 8.114 s |

Tool-like Decode 有較長的 GPU idle boundary。
Tool-like Decode 同時有較多 expert miss、expert read 和 `preadv` CPU sample。
因此，R2 先測試 slot 與 worker 設定。

14K Tool-like Prefill 的 GPU busy 是 70.2%。
這個結果高於 4K Tool-like 的 60.7%。
目前不應假設長 Prefill 只受 SSD 限制。

R1 Metal System Trace 沒有提供 shader timing sample。
R1 資料不能把 GPU time 分配到 attention、routed expert 和 shared expert。
request 結束時間是依最後一個 Python GPU interval 推定。
capture wall time 不是正式效能結果。

CPU profile 沒有顯示 MLX 圖走訪或 identity index 穩定超過 5%。
因此，D1 和 P5 的 profile gate 都沒有通過。

完整資料位於
[`benchmarks/2026-08-10-r0-m5-pro.json`](benchmarks/2026-08-10-r0-m5-pro.json)。

### P3 `gather_qmm` shader profile

P3 使用 `Metal GPU Counters` capture Tool-like 4K Prefill。
這次 capture 產生 shader timeline。
capture 開啟時的 wall time 不是 production 效能結果。

目前 `gather_qmm` kernel 是
`mxfp4_gather_qmm_rhs_nax`。
kernel 使用 BM64、BN64 和 BK64。

| 指標 | 結果 |
| --- | ---: |
| Request | 23.220 秒 |
| Python GPU busy | 13.306 秒 |
| Shader sample work | 12.146 秒 |
| `gather_qmm` shader sample | 2.966 秒 |
| `gather_qmm` / Request | 12.77% |
| 涵蓋率校正後 `gather_qmm` / Request | 13.99% |

shader sample work 涵蓋 Python GPU busy 的 91.28%。
涵蓋率校正假設未分配的 GPU busy 使用相同 shader 組成。

trace 包含 42 個 expert layer burst。
runtime metrics 記錄 84 次 `gather_qmm` call。
shader timeline 有 108 個 `gather_qmm` interval。
一個 logical call 可以被 profiler 分成多個 interval。

同一個固定 prompt 的 R0 route trace 顯示，平均是 95.98 rows/expert。
24.53% 的 expert run 是空的。
27.37% 的 expert run 只有 1 至 15 rows。
BM64 的理論 padding 是配置 rows 的 24.43%。

P3 的停止門檻是 Prefill 的 15%。
原始結果與涵蓋率校正結果都低於門檻。
P3 已停止。
專案沒有建立 isolated MLX build，也沒有修改 runtime。

完整資料位於
[`benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json`](benchmarks/2026-08-11-p3-gather-qmm-profile-m5-pro.json)。

### P4 attention step pilot

P4 使用 Tool-like 4K prompt 和一個 output token。
P4 依 attention 類型測試 512、2,048 和 4,096 step。
目前全域 Control step 是 1,024。

較大的 step 增加 temporary memory，且沒有穩定改善 phase time。
512 step 在單次 run 中降低 CSA 和 dense compressed attention phase。
但是，Control request 在研究期間漂移 7.91%。
這些單次 request time 不是正式速度結果。

組合的 512-step 候選改變 39/42 個 MoE layer 的 expert route。
至少 32,075 個 assignment 改變。
差異下限是 3.11%。
候選的 expert bytes 也減少 6 個 expert blob。

P4 沒有通過正確性 gate。
專案沒有執行正式 ABBA。
runtime 已移除 P4 prototype，並保留單一全域 step。

完整資料位於
[`benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json`](benchmarks/2026-08-11-p4-attention-step-pilot-m5-pro.json)。

### D2 routed expert top-6 profile

D2 使用 Tool-like 4K prompt 和 64 個 output token。
runtime metrics 記錄 63 個 Decode token。
D2 先量測 `_expert_select` Python dispatch。
D2 再使用 `Metal GPU Counters` 量測相同工作負載。

| 指標 | 結果 |
| --- | ---: |
| Decode wall | 12.916 s |
| `_expert_select` Python dispatch | 13.252 ms |
| Dispatch / Decode wall | 0.121% |
| Decode GPU busy | 4.559 s |
| Shader sample work | 2.137 s |
| Shader Timeline 涵蓋率 | 46.88% |
| `_expert_select` shader sample | 58.834 ms |
| Shader sample / Decode wall | 0.456% |
| 涵蓋率校正後 / Decode wall | 0.972% |

一個 bfloat16、256-to-6 microbenchmark 確認 6 個 `_expert_select` shader label。
microbenchmark 只用來對應 shader label。
microbenchmark 的同步 wall time 不用於 profile gate。

涵蓋率校正假設未分配的 GPU busy 使用相同 shader 組成。
D2 的停止門檻是 Decode wall time 的 3%。
dispatch、原始 shader sample 和涵蓋率校正結果都低於門檻。
D2 已停止。
專案沒有建立專用 top-6 prototype，也沒有修改 runtime。

完整資料位於
[`benchmarks/2026-08-11-d2-top6-profile-m5-pro.json`](benchmarks/2026-08-11-d2-top6-profile-m5-pro.json)。

### D3 CSA row 與 `gather` profile

D3 使用 Tool-like prompt 和 64 個 output token。
D3 分別記錄 512、2,048、4,096、8,192 和 14,363 token context。
每個 context 有 63 個相鄰 Decode transition。
每個 transition 比較 21 個 CSA layer。

| Context | 交集比例中位數 | Jaccard 中位數 |
| ---: | ---: | ---: |
| 512 | 100.00% | 100.00% |
| 2,048 | 99.02% | 98.07% |
| 4,096 | 78.71% | 64.90% |
| 8,192 | 70.51% | 54.45% |
| 14,363 | 62.30% | 45.25% |

五種 context 的合併交集比例中位數是 85.94%。
這個結果低於 90% row gate。
Row trace 會同步 selected row array。
因此，row trace 的 wall time 不是速度結果。

D3 使用 microbenchmark 對應 `MXFP8PoolingCache.gather` shader label。
D3 使用既有的 D2 Tool-like 4K Metal trace 計算 Decode 占比。

| 指標 | 結果 |
| --- | ---: |
| Decode wall | 12.916 s |
| Shader Timeline 涵蓋率 | 46.88% |
| 核心 `gather` shader sample | 15.118 ms |
| 核心 sample / Decode wall | 0.117% |
| 核心 sample，依涵蓋率校正 | 0.250% |
| 包含通用 clip/select shader sample | 20.467 ms |
| 包含通用 shader，依涵蓋率校正 | 0.338% |

通用 clip 和 select shader 也可能出現在其他 runtime 操作。
因此，0.338% 不是純 `gather` 結果。
核心結果與包含通用 shader 的結果都低於 5% profile gate。

D3 沒有通過兩個 gate。
D3 已停止。
專案沒有建立 row 重用 prototype，也沒有修改 runtime。

完整資料位於
[`benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json`](benchmarks/2026-08-11-d3-csa-row-profile-m5-pro.json)。

### R2a slot 結果

R2a 使用探索性 ABBA 比較 1,152 與 2,048 slots。

| Prompt | 1,152 slots | 2,048 slots | Decode 差異 | MLX peak 差異 |
| --- | ---: | ---: | ---: | ---: |
| Repeated 4K | 14.42 Tok/s | 13.95 Tok/s | -3.31% | +2.901 GiB |
| Code 4K | 8.25 Tok/s | 2.39 Tok/s | -71.09% | +11.156 GiB |

Code 的 2,048-slot 設定把 expert bytes 降低 21.73%。
Miss 降低 31.34%。
Eviction 降低 36.77%。
但是，expert read timer 沒有降低。
Routing synchronization boundary 從 9.829 秒增加到 111.488 秒。

這個結果顯示較高的 cache hit rate 不等於較高的端到端速度。
2,048-slot 候選設定已拒絕。
目前預設值維持 1,152 slots。

完整資料位於
[`benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-slot-pilot-m5-pro.json)。

### R2b 2-worker 結果

R2b 使用 code 4K prompt 執行一輪 ABBA。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8.29 Tok/s | 110.58 ms | 172.12 ms | 28.042 s |
| 2 | 7.94 Tok/s | 116.54 ms | 182.41 ms | 29.435 s |

2 workers 讓 Decode 降低 4.20%。
兩個配對結果都是負值。
Expert bytes、miss、eviction 和 routing boundary 沒有明顯差異。
Expert read timer 增加 4.97%。

2-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker2-pilot-m5-pro.json)。

### R2b 8-worker 結果

R2b 使用相同的 code 4K prompt 執行一輪 ABBA。

| Read workers | Decode 中位數 | Decode p50 | Decode p95 | Expert read |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8.27 Tok/s | 110.88 ms | 173.60 ms | 28.202 s |
| 8 | 8.29 Tok/s | 110.61 ms | 173.13 ms | 27.875 s |

8 workers 只讓 Decode 增加 0.25%。
兩個配對改善是 0.48% 和 0.01%。
這個結果低於 3% 採用門檻。

8-worker 候選設定已拒絕。
目前預設值維持 4 個 read worker。

完整資料位於
[`benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-worker8-pilot-m5-pro.json)。

### R2c 4-prefetch-worker 結果

R2c 使用 code 4K prompt 和一個 output token 執行一輪 ABBA。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read |
| ---: | ---: | ---: | ---: |
| 2 | 21.956 s | 186.56 Tok/s | 14.478 s |
| 4 | 22.061 s | 185.67 Tok/s | 12.009 s |

4 workers 讓 expert read timer 降低 17.05%。
但是，TTFT 增加 0.48%。
這個結果顯示 expert read timer 不是可以直接相加的 request boundary。

4-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch4-pilot-m5-pro.json)。

### R2c 1-prefetch-worker 結果

R2c 使用相同的 code 4K prompt 執行一輪 ABBA。

| Prefetch workers | TTFT 中位數 | Prefill | Expert read | Prefetch hits 中位數 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 22.037 s | 185.87 Tok/s | 14.531 s | 39.5 |
| 1 | 23.740 s | 172.54 Tok/s | 19.394 s | 0.5 |

1 worker 讓 TTFT 增加 7.73%。
Expert read timer 增加 33.46%。
Prefetch hit 中位數從 39.5 降到 0.5。

1-prefetch-worker 候選設定已拒絕。
目前預設值維持 2 個 prefetch worker。
R2 設定 sweep 已完成。

完整資料位於
[`benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json`](benchmarks/2026-08-10-r2-prefetch1-pilot-m5-pro.json)。

### R3 DSpark 結果

R3 使用 code 4K prompt 和 64 個 output token 執行一輪 ABBA。
R3 比較 normal generation 與 DSpark。

| Mode | Request 中位數 | TTFT 中位數 | 總 throughput | 回報的 Decode | Main model expert bytes | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 31.152 s | 22.072 s | 2.054 Tok/s | 6.94 Tok/s | 239.659 GB | 23.274 GiB |
| DSpark | 41.301 s | 33.722 s | 1.549 Tok/s | 8.32 Tok/s | 316.666 GB | 27.257 GiB |

DSpark 讓 request 增加 32.58%。
DSpark 讓總 throughput 降低 24.60%。
DSpark 讓 TTFT 增加 52.78%。
DSpark 讓 peak footprint 增加 3.983 GiB。

兩個 DSpark run 都接受全部 5 個 draft token。
兩個 DSpark run 都提交 6 個 token。
兩個 DSpark run 都在第一個 speculative round 後 fallback。
Runtime 的成本 gate 因此判定 speculative work 比 normal step 昂貴。

回報的 Decode boundary 增加 19.82%。
這個 boundary 從第一個 token 之後開始。
這個 boundary 不包含較慢的 DSpark prefill。
DSpark 也沒有在整個 Decode phase 持續運作。
因此，端到端決策必須使用 request 與總 throughput。

Normal run 記錄 42 個 batched expert layer 和 84 次 `gather_qmm`。
DSpark run 的兩個欄位都是 0。
DSpark 的 main model expert bytes 增加 32.13%。
DSpark 的 main model cache miss 增加 230.16%。

四個 output token hash 都相同。
R3 已觸發停止條件。
研究沒有執行正式五個 prompt 測試。
目前 DSpark 預設值維持停用。

完整資料位於
[`benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json`](benchmarks/2026-08-10-r3-dspark-pilot-m5-pro.json)。

### M1 Prefill file-cache 歷史結果

正式研究已結束。
runtime 已移除 M1 prototype。

M1 使用 code 4K prompt 和 256 個 output token 執行一輪 ABBA。
Control 使用一般 file descriptor。
M1 為 full-layer Prefill 使用獨立 `F_NOCACHE` file descriptor。
Decode 仍使用一般 file descriptor。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 52.508 s | 21.884 s | 187.17 Tok/s | 8.33 Tok/s | 27.945 s | 23.296 GiB |
| M1 | 50.239 s | 20.221 s | 202.56 Tok/s | 8.50 Tok/s | 24.542 s | 23.297 GiB |

M1 讓 request 改善 4.32%。
M1 讓 TTFT 改善 7.60%。
M1 讓 Prefill throughput 增加 8.22%。
M1 讓 Decode throughput 增加 2.02%。
M1 讓 expert read timer 降低 12.18%。

兩個 request 配對都改善超過 4%。
四個 output token hash 都相同。
Expert bytes、miss 和 eviction 都相同。

Process peak footprint 只增加 1.77 MiB。
M1 沒有達到至少降低 5% 的記憶體接受條件。
測試也沒有出現可重現的 memory pressure。
因此，專案不把 M1 採用為記憶體最佳化。

System pagein 計數降低 61.88%。
這個計數包含其他 process，也受到 ABBA 順序影響。
這個計數不能取代 process footprint。

本次測試只有一種 workload 和一輪 ABBA。
作業系統 page cache 沒有清除。
本次測試沒有 bootstrap 信賴區間。
Code pilot 本身不能支持正式效能採用。
這個 pilot 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-pilot-m5-pro.json)。

#### Tool-like 4K 獨立重跑

Tool-like 重跑使用相同設定和 ABBA 順序。

| Mode | Request 中位數 | TTFT 中位數 | Prefill | Decode | Expert read | Peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 62.574 s | 22.860 s | 179.33 Tok/s | 6.42 Tok/s | 36.282 s | 23.296 GiB |
| M1 | 60.380 s | 20.966 s | 195.38 Tok/s | 6.47 Tok/s | 33.173 s | 23.297 GiB |

M1 讓 request 改善 3.51%。
M1 讓 TTFT 改善 8.29%。
M1 讓 Prefill throughput 增加 8.95%。
M1 讓 Decode throughput 增加 0.75%。
M1 讓 expert read timer 降低 8.57%。

兩個 request 配對分別改善 2.34% 和 4.63%。
兩個配對都是同向改善。
四個 output token hash 都相同。
Process peak footprint 只增加 0.98 MiB。

Code 與 Tool-like 的 request 中位數分別改善 4.32% 和 3.51%。
兩種 workload 的 TTFT 分別改善 7.60% 和 8.29%。
這個結果確認 pilot 等級的效能訊號。

System swap 在完整序列中增加 2.13 MiB。
這個 system-wide 變化不能歸因於 M1。
後續正式測試因此繼續監控 swap。

兩種 workload 都只有一輪 ABBA。
這兩個 pilot 當時沒有 bootstrap 信賴區間。
專案因此進入正式多 workload 重複 ABBA。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-tool-pilot-m5-pro.json)。

#### 正式效能評估 Wave 1

Wave 1 使用五個 R0 4K prompt。
每個 prompt 有一個不納入統計的 Control warm-up。
每個 prompt 接著執行一輪 ABBA。
Wave 1 共有 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 3.03% | 8.23% | 8.98% | -2.93% |
| Code | 4.54% | 8.55% | 9.34% | 1.73% |
| 繁體中文技術文字 | 3.40% | 8.12% | 8.84% | 0.13% |
| Mixed math | 3.69% | 8.24% | 8.98% | 0.57% |
| Tool-like | 2.70% | 7.97% | 8.66% | -0.32% |

四個主要 workload 的 request 改善中位數是 3.54%。
TTFT 改善中位數是 8.18%。
Prefill throughput 增加中位數是 8.91%。
Decode throughput 增加中位數是 0.35%。
八個主要 workload request 配對都改善。

四個主要 workload 的 process peak footprint 增加中位數是 0.36 MiB。
正式序列的 system swap 降低 8.00 MiB。
測試沒有出現 thermal 或 performance warning。

Repeated Decode 在本輪降低 2.93%。
後續 wave 因此繼續確認這個 regression。
Wave 1 單獨不能支持正式採用。
Wave 1 當時沒有改變 runtime 預設值。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave1-m5-pro.json)。

#### 正式效能評估 Wave 2

Wave 2 輪替 prompt 順序，並完成 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.63% | 11.05% | 12.48% | 3.64% |
| Code | 3.78% | 7.83% | 8.49% | 0.93% |
| 繁體中文技術文字 | 3.66% | 8.26% | 9.01% | 0.47% |
| Mixed math | 3.57% | 8.51% | 9.30% | 0.17% |
| Tool-like | 2.61% | 7.77% | 8.43% | -0.36% |

四個主要 workload 的 request 改善中位數是 3.62%。
TTFT 改善中位數是 8.05%。
Prefill throughput 增加中位數是 8.75%。
Decode throughput 增加中位數是 0.32%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 降低 8.00 MiB。
System-wide compressor 增加 209.67 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

#### 正式效能評估 Wave 3

Wave 3 再次輪替 prompt 順序，並完成 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.93% | 11.87% | 13.45% | 3.25% |
| Code | 4.08% | 7.71% | 8.35% | 1.53% |
| 繁體中文技術文字 | 4.28% | 8.68% | 9.50% | 1.22% |
| Mixed math | 3.34% | 6.97% | 7.52% | 0.84% |
| Tool-like | 2.07% | 8.48% | 9.17% | -1.64% |

四個主要 workload 的 request 改善中位數是 3.71%。
TTFT 改善中位數是 8.09%。
Prefill throughput 增加中位數是 8.76%。
Decode throughput 增加中位數是 1.03%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 沒有變化。
System-wide compressor 增加 244.77 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

#### 正式效能評估 Wave 4

Wave 4 再次輪替 prompt 順序，並完成 20 個有效量測 run。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 7.76% | 10.82% | 12.18% | 4.34% |
| Code | 3.31% | 5.81% | 6.17% | 1.53% |
| 繁體中文技術文字 | 3.56% | 7.95% | 8.63% | 0.48% |
| Mixed math | 3.17% | 7.75% | 8.42% | 0.01% |
| Tool-like | 3.32% | 7.90% | 8.56% | 0.63% |

四個主要 workload 的 request 改善中位數是 3.32%。
TTFT 改善中位數是 7.82%。
Prefill throughput 增加中位數是 8.49%。
Decode throughput 增加中位數是 0.56%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 沒有變化。
System-wide compressor 增加 340.41 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

#### 正式效能評估 Wave 5

Wave 5 再次輪替 prompt 順序，並完成 20 個有效量測 run。
第一次序列在 Tool-like B1 啟動時失去執行 session。
專案排除整個不完整序列，並從新的 warm-up 完整重跑。

| Workload | Request 改善 | TTFT 改善 | Prefill throughput | Decode throughput |
| --- | ---: | ---: | ---: | ---: |
| Repeated | 8.36% | 12.44% | 14.24% | 3.89% |
| Code | 2.85% | 6.26% | 6.68% | 0.51% |
| 繁體中文技術文字 | 3.44% | 9.36% | 10.32% | -0.69% |
| Mixed math | 3.68% | 8.83% | 9.66% | 0.14% |
| Tool-like | 2.36% | 6.32% | 6.82% | 0.10% |

四個主要 workload 的 request 改善中位數是 3.15%。
TTFT 改善中位數是 7.58%。
Prefill throughput 增加中位數是 8.24%。
Decode throughput 增加中位數是 0.12%。
八個主要 workload request 配對都改善。

Process peak footprint 沒有實質變化。
System swap 降低 8.56 MiB。
System-wide compressor 增加 1,236.50 MiB。
這個 system-wide 變化不能歸因於 M1。
測試沒有出現 thermal 或 performance warning。

完整資料位於
[`benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-wave5-m5-pro.json)。

#### 五輪正式結果

五輪共有 100 個有效量測 run 和 50 個配對。
全部 50 個 request 配對都改善。

四個主要 workload 的 request 改善中位數是 3.53%。
95% bootstrap 信賴區間是 3.27% 至 3.76%。
TTFT 改善中位數是 7.99%。
Prefill throughput 增加中位數是 8.69%。
Decode throughput 增加中位數是 0.23%。
Expert read timer 改善中位數是 10.88%。

主要 workload 的 Decode p95 regression 最大值是 1.71%。
這個結果通過 2% 條件。
Repeated Decode p95 regression 是 3.01%。
這個結果超過 2% 上限。

Process peak footprint 的配對中位數增加 0.32 MiB。
配對 p95 增加 19.24 MiB。
每一輪的 system swap 都沒有增加。
部分 wave 的 system-wide compressor 總量增加。
這個總量包含測試兩組與其他 process，因此不能歸因於 M1。
這組證據仍沒有證明 compressor 不增加。

M1 沒有通過全部正式採用條件。
專案拒絕預設啟用 M1。
runtime 已移除 M1 prototype。

完整結果位於
[`benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json`](benchmarks/2026-08-11-m1-file-cache-formal-final-m5-pro.json)。

## 瓶頸判讀

| 觀察 | 先檢查 | 不應直接下的結論 |
| --- | --- | --- |
| TTFT 高，`batched_expert_layers=42` | expert bytes、read timer、prefetch hits、Metal capture | 所有 TTFT 都是 SSD。 |
| Expert bytes 高，read throughput 低 | direct SSD microbenchmark、背景 I/O、worker A/B | 增加 worker 一定更快。 |
| Cache miss 和 eviction 都高 | slot 數、prompt 類型、每層 reserve | 提高 hit rate 一定提高端到端速度。 |
| Routing sync 高 | 前一個 `mx.eval` boundary、attention 和 MoE graph | router kernel 本身很慢。 |
| Decode cache eval 高 | cache array 數、context、MXFP8 chunk、persistence path | KV cache 量化必定較快。 |
| `gather_qmm_calls=0` 但長 prefill | layer-major 和 batched flag、未快取 token 數 | MLX 不支援 `gather_qmm`。 |
| Prompt reuse 高但 TTFT 仍高 | persistent cache load、suffix 長度、第一個 decode step | prompt cache 沒有效果。 |
| DSpark acceptance 高但速度低 | committed length、draft time、verify time、fallback | acceptance 高等於 throughput 高。 |

## 檢測流程

### 1. 建立 baseline

先固定 checkpoint、commit、prompt 和 output token 數。
先停用 persistent prompt cache。
先使用 greedy generation。

R0 可以用下列命令建立固定 prompt 檔。

```sh
PYTHONPATH=runtime .venv/bin/python Scripts/prepare_r0_prompts.py \
  --model /path/to/model.dsv4 \
  --output scratch/r0-profile/prompts
```

prompt manifest 會保存 prompt token 數、文字 SHA-256 和
`prompt_token_sha256`。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file fixed-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --top-p 1 \
  --no-persistent-prompt-cache \
  --metrics-json baseline.json
```

### 2. 一次只改一個設定

runtime 提供下列比較開關。

| 問題 | 比較設定 |
| --- | --- |
| Layer-major 收益 | `--no-layer-major-prefill` |
| Batched MoE 收益 | `--no-batched-expert-prefill` |
| Attention step | `--prefill-step-size N` |
| MoE tile | `--moe-prefill-step-size N` |
| 一般 read concurrency | `--read-workers N` |
| Full-layer prefetch concurrency | `--prefetch-read-workers N` |
| Slot 壓力 | `--slots N` |
| KV cache | `--bf16-kv-cache` |
| Index cache | `--no-fp4-index-cache` |
| Ready expert decode | `--no-ready-expert-decode` |
| Prompt persistence | `--no-persistent-prompt-cache` |
| Expert-file descriptor policy | `--expert-file-cache-policy cached|bypass` |
| DSpark | `--dspark` |
| DSpark atomic prompt-context reuse | `--dspark --dspark-prompt-cache` |
| DSpark sequential verifier oracle | `--dspark --dspark-sequential-verification` |
| DSpark hybrid v3 verifier | `--dspark --dspark-hybrid-verification` |
| DSpark hash exact prefetch | `--dspark --dspark-hash-prefetch` |
| DSpark storage-aware prefix | `--dspark --dspark-adaptive-block` |

每組 A/B 必須產生相同 greedy token hash。
如果 output token 不同，速度比較無效。

### 3. 分開 SSD microbenchmark

```sh
swift run dsv4-repack benchmark \
  --model /path/to/model.dsv4 \
  --samples 32
```

microbenchmark 只回答 expert blob read 能力。
microbenchmark 不包含 router、GPU compute、slot admission 或 MLX graph。

### 4. 記錄 routed expert trace

CLI 可以記錄 prefill histogram、decode route 和真實 miss。
目前 trace 要求 DSpark 停用。

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file fixed-prompt.txt \
  --max-tokens 256 \
  --temperature 0 \
  --expert-route-trace routes.json

PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.route_trace \
  routes.json \
  --hot-per-layer 6 8 \
  --decode-tokens 256
```

route coverage 是 counterfactual 分析。
route coverage 不等於實際 speedup。

route trace 的 `prefill_chunk_histograms` 保存每個 Prefill MoE 區塊的
rows/expert。
route trace 會同步 router index。
因此 route trace 的 wall time 不是正式效能結果。

### 5. 使用 Metal capture

Metal capture 用來分解 GPU operation 和 CPU/GPU wait。
capture 開啟時的 wall time 不是正式效能結果。

capture 應回答下列問題。

- attention、routed MoE 和 shared expert 各使用多少 GPU time。
- `gather_qmm` 是否使用預期 shape 和排序 path。
- graph 是否有大量小 operation 或空檔。
- CPU `preadv`、router index materialization 和 GPU submission 是否重疊。
- 某一個 section 是否穩定超過 request time 的 5%。

下列命令會啟動一個獨立的 capture run。
capture 的 metrics 不可作為正式 wall-time 結果。

```sh
xcrun xctrace record \
  --template 'Metal System Trace' \
  --output scratch/r0-profile/repeated-4096.trace \
  --env PYTHONPATH=runtime \
  --target-stdout /dev/null \
  --launch -- .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/model.dsv4 \
  --prompt-file scratch/r0-profile/prompts/repeated-4096.txt \
  --max-tokens 64 \
  --temperature 0 \
  --top-p 1 \
  --no-persistent-prompt-cache \
  --metrics-json scratch/r0-profile/repeated-4096-capture.json
```

只有 capture 證明 section 足夠大時，才評估 custom Metal kernel。

## API input size benchmark

`Scripts/benchmark_api.py` 透過 Whallm API 執行 input size benchmark。
預設 input size 是 1K、2K、8K、16K 和 32K token。
腳本使用 SPEED-Bench throughput subset 的 `mixed` category。
腳本先縮短每筆資料的第一個 user message。
腳本再套用 installed model 的完整 chat template。
產生的 prompt 會保留 chat template 尾端，並符合指定的 input token 數。
腳本也會確認 server 回報相同的 input token 數。

[SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)
使用 NVIDIA Evaluation Dataset License Agreement。
使用者必須先確認該授權適合測試用途。
使用者必須使用 NVIDIA 的官方
[`prepare.py`](https://github.com/NVIDIA-NeMo/Skills/blob/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/dataset/speed-bench/prepare.py)
建立 JSONL。
下列命令固定目前核對的 NVIDIA-NeMo/Skills commit：

```sh
python3 -m venv scratch/speed-bench-venv
scratch/speed-bench-venv/bin/python -m pip install datasets pandas numpy tiktoken
curl -fsSLo scratch/prepare-speed-bench.py \
  https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/dataset/speed-bench/prepare.py
scratch/speed-bench-venv/bin/python scratch/prepare-speed-bench.py \
  --config all \
  --output_dir scratch/speed-bench
```

`SPEED_BENCH_DIR` 可以指定另一個已準備的資料夾。

```sh
make benchmark-dsv4
make benchmark-qwen
```

Python CLI 的預設值是每個 input size 執行 5 次。
目前兩個 Make entry 明確指定每個 input size 執行 3 次。

腳本會先重用目前的 Server。
本機 Server 未執行時，腳本會從 Whallm model folder 找出指定的 installed model，
並使用 App 已儲存的設定啟動 Server。
測試完成後，腳本只會停止自己啟動的 Server。
腳本不會停止 App 或其他 process 啟動的 Server。

`--model-path` 可以指定另一個 installed model。
自動啟動只支援本機 `http://HOST:PORT/v1` URL。
`--server-start-timeout` 的預設值是 900 秒。
API key 從 `OPENAI_API_KEY` environment variable 讀取。

```sh
OPENAI_API_KEY='<local-key>' \
  .venv/bin/python Scripts/benchmark_api.py \
  --model-path /Volumes/Models/deepseek-v4-flash-0731.dsv4 \
  --model deepseek-v4-flash-0731 \
  --speed-bench-dir scratch/speed-bench \
  --runs 5
```

```sh
.venv/bin/python Scripts/benchmark_api.py \
  --model qwen3.8-flash-next-fp8 \
  --speed-bench-dir scratch/speed-bench \
  --runs 5 \
  --max-output-tokens 64
```

每個 run 記錄總花費時間、最終 TTFT、Prefill tok/s、Decode tok/s 和 request
期間最高的 `performance.active_memory_bytes`。
每個 input size 的 Maximum 和 P95 使用個別 run 的結果計算。
P95 使用 nearest-rank 方法。
少於 20 個 run 時，nearest-rank P95 會等於 Maximum。

腳本會在終端顯示 ASCII 表格。
表格會並排顯示 P95 和 Maximum。
兩組欄位都包含 `Total time (s)`、TTFT、Prefill、Decode 和 Memory。
腳本也會把每個 run、source、environment、runtime configuration、cache state
和 output text hash 寫入 `docs/benchmarks/`。
artifact 的 `server.started_by_benchmark` 會記錄 Server 是否由腳本啟動。
每個 run 使用不同的 SPEED-Bench input。
artifact 會記錄 SPEED-Bench question ID、來源、subset file SHA-256 和 `mixed` category。
腳本不會清除 prompt cache 或作業系統 page cache。
Memory 是 MLX active memory，不是 process RSS。
API 不提供生成 token ID，所以 artifact 不宣稱具有正式 output token hash。

## A/B 驗收

正式 A/B 使用下列規則。

1. 使用至少五個代表性 prompt。
2. 每個 prompt 產生至少 256 個 token，除非測試目標是 TTFT。
3. A/B 執行順序要交錯。
4. 分開 fresh runtime、warm graph 和 OS page-cache 狀態。
5. 保存每次 run，不只保存平均值。
6. 報告 median 和 p95。
7. 比較完整 token hash。
8. 記錄 peak memory 和 process RSS。
9. 記錄 expert bytes、read timer、hit、miss 和 eviction。
10. 任何改善不得讓短 prompt 或低記憶體設定發生未說明的 regression。

## 目前不成立的結論

- 官方 DSpark 的 60–85% 不是本機 batch size 1 的預期倍率。
- Apple unified memory 不代表任意 `MTLBuffer` 可以由 Python MLX zero-copy 使用。
- MTLIO API 與 native byte/event gate 通過不代表 MLX handoff 存在或本 runtime 會更快。
- `mx.compile` 可用不代表所有 graph 都會融合或加速。
- 高 route coverage 不代表 handoff policy 會降低實際 miss。
- 更大的 slot cache 不代表更快。slot 也會增加記憶體和 eviction scan 成本。

目前研究狀態請見[研究結論](RESEARCH.md)。
