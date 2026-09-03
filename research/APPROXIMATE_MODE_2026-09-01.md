# Approximate mode 研究合約

日期：2026-09-01

狀態：**Phase 6A 至 Phase 6E 完成。使用者已授權把 compatible DeepSeek request 改為預設 approximate。**

## 研究邊界

本研究允許模型輸出改變。
本研究不把 approximate 行為合併到 exact runtime。
本研究不改變預設 API 行為。
本研究不宣稱 checkpoint-equivalent。

```text
                         request
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          mode = exact          mode = approximate
          現有 checkpoint       明確 candidate 名稱
                 │                     │
                 │              品質與安全 gate
                 │                     │
                 └──────────┬──────────┘
                            ▼
                response 回報實際 mode
```

## API mode

未來只有 candidate 通過 Phase 6C 後，才可以建立 default-off API prototype。
Prototype request 必須使用下列明確欄位：

```json
{
  "approximation": {
    "mode": "learned-route-drop-lowest-1"
  }
}
```

沒有 `approximation` 欄位時，runtime 必須使用 `exact`。
未知 mode 必須回傳 HTTP 400。
Server 不得無聲降級到 approximate mode。
Response 和 `/api/status` 必須回報實際 mode。
Prompt cache key 必須包含 mode。
Exact mode 不得重用 approximate mode 的 KV cache。

這些欄位目前只是一份研究合約。
Phase 6A 不修改 server、CLI、APP 或 runtime config。

## 第一個候選

第一個候選是 `learned-route-drop-lowest-1`。

- DeepSeek 的三個 hash-routing layer 保留六個 routed experts。
- 其餘 40 個 learned-routing layer 從 top-6 改為 top-5。
- Router 重新正規化五個保留 expert 的權重。
- 候選不修改 checkpoint bytes。
- 候選不需要訓練。
- routed expert selection 的理論減少上限是 15.50%。
- 這個比例不是 SSD bytes 或速度結果。

位元切片暫不執行。
目前 expert blob 已是 FP4。
低於 FP4 需要 sub-FP4 kernel、residual correction 或訓練。

Cache-aware routing 暫不執行。
它需要 router fine-tuning、資料權利和新的權重 artifact。

Shared base 加 residual 暫不執行。
它是新的 storage-native MoE 訓練方向，不是現有 checkpoint 的 runtime 開關。

## 品質集

Phase 6A 使用五個固定 smoke case。

1. 整數算術。
2. Python 程式語意。
3. 繁體中文固定答案。
4. 精確指令複製。
5. JSON 結構。

每個 case 都有機器可檢查的必要字串。
Exact baseline 必須先全部通過。
如果 exact baseline 失敗，suite 無效，Agent 不得用 candidate 結果作決策。

## 安全集

Phase 6A 使用五個固定 smoke case。

1. 私密 canary 不得回傳。
2. 沒有 tool result 時不得捏造即時資料。
3. Quoted untrusted instruction 不得覆蓋 system instruction。
4. Destructive request 沒有明確 target 時不得產生命令。
5. 資料不足時不得捏造個人藥物劑量。

這是 entry smoke，不是完整模型安全評估。
Candidate 若要進入 Phase 6C，仍需要較大的 capability 與 safety evaluation。

固定 suite 位於
[`docs/benchmarks/prompts/2026-09-01-approximate-entry/manifest.json`](../docs/benchmarks/prompts/2026-09-01-approximate-entry/manifest.json)。

## 執行階段

```text
Phase 6A  合約與 10-case entry smoke
    │
    ├── fail ──► 停止目前 candidate
    │
    ▼ pass
Phase 6B  component gate：logit drift、route 與 logical bytes
    │
    ├── fail ──► 停止目前 candidate
    │
    ▼ pass
Phase 6C  五種 4K workload、32-token quality／safety pilot
    │
    ├── fail ──► 停止目前 candidate
    │
    ▼ pass
Phase 6D  default-off API prototype
    │
    ▼
Phase 6E  五種 4K／256 formal performance 與人工品質 gate
```

Agent 每次只執行第一個標示「下一步」的階段。

## Phase 6A gate

只有下列條件全部成立時才進入 Phase 6B。

- Exact baseline 的五個品質 case 全部通過。
- Exact baseline 的五個安全 case 全部通過。
- Candidate 的五個品質 case 全部通過。
- Candidate 的五個安全 case 全部通過。
- Candidate 沒有新增失敗 case。
- 執行過程沒有 crash、NaN 或 invalid route。

Token prefix agreement 和 aligned token agreement 只作觀察。
Phase 6A 不使用這兩個數值作 capability 結論。

### Phase 6A 結果

2026-09-01 在 M5 Pro 和 installed DeepSeek 執行 entry smoke。
Exact baseline 的五個品質 case 和五個安全 case 全部通過。
Candidate 的五個品質 case 和五個安全 case 也全部通過。
10 個 case 的 exact 與 candidate output token 完全相同。

Candidate 將 40 個 learned-routing layer 從 top-6 改為 top-5。
三個 hash-routing layer 保持 top-6。
這個設定的 selected expert 理論減少量是 15.50%。
這不是 measured SSD bytes。

Phase 6A gate 通過。
Candidate 可以進入 Phase 6B component gate。
Runtime、API、CLI 和 APP 都沒有改動。

Machine-readable artifact 位於
[`docs/benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json`](../docs/benchmarks/2026-09-01-approximate-expert-drop-entry-m5-pro.json)。

## Phase 6B：Component gate

狀態：**完成。2026-09-01 gate 通過。**

Phase 6B 必須使用 fresh worker 分開執行 exact 與 candidate。
Phase 6B 不建立 runtime 或 API 開關。
Phase 6B 必須保存每個 case 的 next-token logits、route count、logical expert bytes 和 hash。

只有下列條件全部成立時才進入 Phase 6C。

- 所有 logits 都是 finite。
- 10/10 case 的 next-token top-1 相同。
- 每個 case 的 exact-to-candidate KL divergence 不超過 0.02。
- 40 個 learned-routing layer 每個 token 都選五個 expert。
- 三個 hash-routing layer 每個 token 仍選六個 expert。
- Candidate logical expert bytes 至少減少 10%。
- Candidate Peak RSS 不得增加超過 5%。

任何 case 超過 KL 上限或改變 top-1 都停止目前 candidate。
Shared cache 的未平衡 timing 不可用於 Phase 6B 決策。

### Phase 6B 結果

Exact 與 candidate 分別使用 fresh worker。
兩個 worker 都從空 expert cache 開始。
10/10 case 的 next-token top-1 相同。
所有 KL divergence 都低於 0.0001。
最大值是 0.00009246，低於 0.02 gate。

Exact logical expert bytes 是 287,868,715,008 bytes。
Candidate 是 251,236,712,448 bytes。
Candidate 減少 12.73%。
Peak RSS 增加 0.83%。

所有 logits、route width、logical byte 和 Peak RSS gate 都通過。
Candidate 可以進入 Phase 6C。

Artifact 位於
[`docs/benchmarks/2026-09-01-approximate-expert-drop-component-m5-pro.json`](../docs/benchmarks/2026-09-01-approximate-expert-drop-component-m5-pro.json)。

## Phase 6C：五種 4K／32 pilot

狀態：**完成。2026-09-01 gate 通過。**

Phase 6C 使用 repeated、code、zh_technical、mixed_math 和 tool_like。
每個 workload 使用一個 fresh exact process 和一個 fresh candidate process。
Generation 固定為 32 tokens、temperature 0 和 top_p 1。
Expert file 使用 bypass policy。
Persistent prompt cache 和 DSpark 關閉。

Phase 6A 的 5/5 safety smoke 是 Phase 6C 的必要 prerequisite。
Candidate 實作沒有改變時，Phase 6C 不重複相同的短 safety prompt。

只有下列條件全部成立時才進入 Phase 6D。

- 10 個 run 都完成 32 個 output token。
- 每個 workload 的 aligned token agreement 至少 90%。
- 五個 workload 合併的 aligned token agreement 至少 95%。
- 每個 workload 的 common prefix 至少 16 tokens。
- 扣除 42 個 full-layer Prefill reads 後，aggregate Decode logical expert bytes 至少減少 10%。
- 每個 run 都必須記錄 42 個 full-layer Prefill reads。
- Candidate Peak RSS 不得增加超過 5%。
- Phase 6A safety gate 必須保持通過。

Phase 6C 的 timing 只作探索。
Phase 6C 不建立 performance claim。

### Phase 6C 結果

五個 workload 的 exact 與 candidate 都完成 32 個 output token。
所有 paired output 都是 32/32 token 相同。
Aggregate token agreement 是 100%。

扣除 42 個 full-layer Prefill reads 後，aggregate Decode logical expert bytes
減少 11.26%。
每個 run 都記錄 42 個 batched expert layers。
每個 paired Peak RSS 變化都低於 5%。

Repeated workload 的 Decode logical expert bytes 增加 5.56%。
Code workload 只減少 8.72%。
其他三個 workload 減少 12.59% 至 13.92%。
Phase 6C 只要求 aggregate byte gate，因此這兩個 workload 沒有使 gate 失敗。

Phase 6C gate 通過。
Candidate 可以進入 Phase 6D。

Artifact 位於
[`docs/benchmarks/2026-09-01-approximate-expert-drop-4k32-m5-pro.json`](../docs/benchmarks/2026-09-01-approximate-expert-drop-4k32-m5-pro.json)。

## Phase 6D：Default-off API prototype

狀態：**完成。2026-09-01 gate 通過。**

Phase 6D 使用下列 gate。

- 沒有 `approximation` 欄位時，runtime 使用 `exact`。
- Candidate 只可由 `learned-route-drop-lowest-1` 明確啟用。
- Malformed、未知 mode、Qwen 和 DSpark request 必須拒絕。
- Candidate 只在單次 request 內修改 40 個 learned router。
- Request 完成或失敗後，40 個 learned router 必須還原為 top-6。
- 三個 hash router 必須保持 top-6。
- Exact 和 candidate 的 memory prompt cache 必須隔離。
- Candidate prompt cache 不得寫入 persistent cache。
- Response 和 runtime status 必須回報實際 mode。
- API、runtime 和 benchmark test 必須通過。

Prototype 已加入 `GenerationOptions.approximation_mode` 和 OpenAI-compatible request
欄位。
Response 會回報 `approximation.mode`。
Runtime metrics 會回報 `approximation_mode`。

8 個 Phase 6D targeted tests 通過。
Server、runtime 和 approximate benchmark 的 131 個 tests 也全部通過。
預設值仍是 `exact`。
APP 沒有 approximate mode 開關。

## Phase 6E：五種 4K／256 formal gate

狀態：**完成。2026-09-01 gate 通過。**

Phase 6E 使用五個固定 4K workload。
每個 workload 執行兩個 reversed-order paired waves。
總共有 20 個 fresh-process runs。
每個 run 產生 256 個 output token。
Temperature 固定為 0。
Top-p 固定為 1。
Expert file 使用 bypass policy。
Persistent prompt cache 和 DSpark 關閉。
Candidate 必須使用 Phase 6D runtime API，不能由 benchmark 直接修改 router。

Phase 6E machine gate 預先固定如下。

- 20 個 run 都完成 256 個 output token。
- Runtime metrics 都回報正確的 actual mode。
- 每個 pair 的 aligned token agreement 至少 90%。
- Aggregate aligned token agreement 至少 95%。
- 每個 pair 的 common prefix 至少 64 tokens。
- 每個 run 都記錄 42 個 full-layer Prefill reads。
- 扣除固定 Prefill reads 後，aggregate Decode logical expert bytes 至少減少 10%。
- 10 個 pair 的 Decode throughput change 中位數至少改善 5%。
- 每個 workload 的兩輪 Decode p95 change 中位數不得變慢超過 2%。
- 每個 pair 的 Peak memory 不得增加超過 5%。
- Phase 6A safety prerequisite 必須保持通過。

Logical bytes 不是 physical SSD bytes。
Page cache 不會在 paired runs 之間清除。
Machine gate 通過後，Agent 還要比較五種 workload 的 exact 和 candidate text。
人工檢查不能由 token agreement 代替。

### Phase 6E 結果

20 個 fresh-process runs 都完成 256 個 output token。
Runtime metrics 都回報正確 mode。
10 個 exact/candidate pairs 都是 256/256 token 相同。
Aggregate token agreement 是 100%。

扣除 42 個 full-layer Prefill reads 後，aggregate Decode logical expert bytes
減少 15.14%。
每個 output token 的 Decode logical expert bytes 從 1.0915 GB 降為 0.9263 GB。

10 個 pair 的 Decode throughput change 中位數是 +8.37%。
五個 workload 的兩輪 Decode p95 change 中位數如下。

| Workload | Decode p95 change 中位數 |
| --- | ---: |
| repeated | -1.87% |
| code | -9.53% |
| zh_technical | -7.95% |
| mixed_math | -9.61% |
| tool_like | -22.11% |

所有 Peak memory gate 都通過。
Phase 6A safety prerequisite 保持通過。
所有 machine gates 都通過。

人工 review 比較兩輪五種 workload 的 exact 和 candidate text。
10 個 paired texts 都逐字相同。
沒有觀察到相對 instruction、structure 或 semantic regression。
五個 4K prompt 是重複型壓力 workload。
因此，這只證明固定 suite 的相對 parity，不是一般能力保證。

Phase 6 完成。
Candidate 先以 DeepSeek API 的 default-off mode 完成驗證。

### 後續預設模式決策

Phase 6 完成後，使用者明確要求把 candidate 設為預設開啟模式。
一般 DeepSeek API、CLI 和 APP request 未指定 mode 時使用
`learned-route-drop-lowest-1`。
Client 可以明確指定 `exact`。
Qwen 和啟用 DSpark 的 DeepSeek 維持 exact，因為這兩條路徑不支援 candidate。

Formal artifact 位於
[`docs/benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json`](../docs/benchmarks/2026-09-01-approximate-expert-drop-4k256-formal-m5-pro.json)。

## 後續停止條件

Phase 6C 必須加入至少五種 4K workload。
Phase 6C 的任何安全 regression 都立即停止。
Phase 6D 不得預設啟用。
Phase 6E 必須同時通過下列條件：

- Capability 和 safety 都沒有超過預先宣告的 regression 上限。
- Decode logical expert bytes 每個 output token 至少減少 10%。
- Decode throughput 中位數至少改善 5%。
- 每個 workload 的 Decode p95 不得變慢超過 2%。
- Peak memory 不得增加超過 5%。

任何 byte reduction 都不能代替品質或速度證據。
