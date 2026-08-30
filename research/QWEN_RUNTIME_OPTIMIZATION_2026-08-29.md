# Qwen3.8-Flash-Next runtime 效能研究計畫

狀態：執行中  
開始日期：2026-08-29  
固定 checkpoint revision：`bcd9f01ddc9cff2316eb84281bebcd5b058bddce`  
固定 installed model revision：`753d0aa57059fad70a5f7e6cc249f25df56bbd34`

本文件記錄研究假設、實驗順序、停止條件和目前證據。
本文件不是目前 runtime 的速度保證。

## 目標

本研究要降低 Qwen request wall time。
本研究分別量測 Prefill、Decode、expert I/O 和 peak memory。
本研究不以降低輸出正確性換取速度。

第一階段只使用目前 installed model。
第一階段不加入新的 checkpoint tensor。
Qwen MTP 會先執行 checkpoint 合約查核。
合約查核通過後，MTP 才能進入 runtime prototype。

## 目前基準

2026-08-28 的 M5 Pro API benchmark 記錄下列範圍：

- Prefill：59.8 至 65.0 tok/s。
- Decode：7.8 至 9.3 tok/s。
- Peak active memory：15.19 至 18.92 GiB。
- Input：1,024 至 16,384 tokens。
- Output：64 tokens。

這些結果使用不同的 SPEED-Bench prompt。
這些結果沒有清除作業系統 page cache。
這些結果是起始證據，不是候選 A/B 的 control。

## 已知執行路徑

```text
prompt
  │
  ▼
Qwen layer-major Prefill
  │  每層讀取 512 個 routed expert
  │  每個完整 expert layer 約 1.25 GiB
  ▼
48 個 model layer
  ├─ 36 個 Gated DeltaNet layer
  ├─ 12 個 QSA layer
  └─ 每層 top-10 routed expert + shared expert
  │
  ▼
Decode
  │  selected expert cache
  │  目前 Qwen 路徑先等待全部 missing expert
  ▼
下一個 token
```

## 證據規則

每個正式 A/B artifact 必須記錄下列資料：

1. source commit 和 working tree diff SHA-256。
2. checkpoint revision 和 installed model manifest SHA-256。
3. macOS、SoC、GPU core、記憶體、MLX 和 Python 版本。
4. prompt text SHA-256、prompt token SHA-256 和 output token SHA-256。
5. prompt token、output token、request wall time、TTFT、Prefill 和 Decode。
6. expert bytes、expert read time、cache hit、cache miss 和 eviction。
7. peak MLX active memory 和 peak MLX cache memory。
8. persistent prompt cache 和作業系統 page cache 狀態。
9. 執行順序和 warmup 規則。

探索性測試可以減少重複次數。
探索性測試不可直接成為採用結論。

## 採用門檻

候選必須先通過正確性門檻：

- greedy output token hash 必須與 control 完全相同。
- prompt token hash 必須完全相同。
- 所有 cache counter 和 expert byte counter 必須可以對帳。
- 測試不可產生例外、swap 壓力或非有限數值。

候選必須再通過效能門檻：

- 主要目標的 paired median 至少改善 5%。
- Decode 候選不可讓 TTFT 變差超過 5%。
- Prefill 候選不可讓 Decode 變差超過 5%。
- Peak active memory 不可增加超過 15%。
- Expert bytes per generated token 不可增加超過 5%，除非 request wall time至少改善 10%。
- P95 token latency 不可增加超過 10%。

小於 5% 的差異視為未解決。
候選必須停止或使用更多重複次數重新測試。

## 執行順序

```text
Q0 現況基準與瓶頸
  │
  ├── GPU／CPU／SSD section 足夠大？ ──否──► 停止該方向
  │                                  
  ▼
Q1 Qwen ready expert
  │  移除「等待最慢 expert」的 barrier
  ▼
Q2 Qwen LFU decay
  │  只調整 cache policy
  ▼
Q3 Prefill 與 QSA
  │  threshold、step、QSA query chunk
  ▼
Q4 Qwen MTP 合約查核
  │  tensor、graph、cache、draft token contract
  ▼
MTP prototype gate
```

Q4 的合約查核可以和 Q1 至 Q3 同時進行。
MTP runtime prototype 必須等待合約查核完成。

## Q0：現況基準與瓶頸

### 問題

- Prefill 是 expert I/O、MXFP4 `gather_qmm`、Gated DeltaNet 或 QSA 限制嗎？
- Decode 是 expert I/O、MXFP4 QMM、QSA、LM head 或 Python／MLX graph 限制嗎？
- Qwen 的 selected expert cache 是否經常等待單一較慢讀取？

### 第一輪 workload

- `code-128.txt`，64 output tokens。
- `code-4096.txt`，64 output tokens。
- greedy sampling。
- persistent prompt cache 關閉。
- 每次使用 fresh process。
- 第一輪是探索性單次執行。

### 後續正式 workload

- code、mixed math、repeated、tool-like 和 Traditional Chinese technical。
- 4,096 prompt tokens 和 64 output tokens。
- control／candidate／candidate／control。
- 每個 process 只執行一個量測 request。

## Q1：Qwen ready expert

### 假設

目前 Qwen selected expert path 呼叫 `get_many`。
`get_many` 會等待全部 missing expert 完成讀取。
Qwen 可以使用現有 `iter_ready` protocol。
runtime 可以先計算 resident 或先讀完的 routed expert。

### 實作界線

- 不改 router IDs 或 router scores。
- 不改 expert blob。
- 不改 slot admission contract。
- 不改 Prefill batched path。
- 只改 Qwen selected expert execution order。

### 停止條件

- output token hash 不同。
- request wall time 或 Decode 沒有改善 5%。
- cache admission order 讓 expert bytes 增加超過門檻。
- MLX graph 不能在讀取期間開始計算。

## Q2：Qwen LFU decay

### 假設

目前 decay 門檻使用 expert access count。
Qwen 每個 decode token 最多產生 480 個 routed expert assignment。
目前門檻可能過早降低 Qwen hot expert frequency。

### 候選

- 目前 control。
- 以生成 token 計算的 128-token decay。
- 以生成 token 計算的 512-token decay。
- 以生成 token 計算的 1,024-token decay。
- 不 decay。

第一輪只執行 trace replay 或 deterministic cache simulation。
只有 miss、bytes 或 eviction 明確改善的候選才執行完整模型。

### 停止條件

- cache simulation 沒有改善 miss 或 bytes。
- 完整模型只改善 hit rate，但 request wall time沒有改善。
- 候選增加長尾 latency 或 SSD bytes。

## Q3：Prefill 與 QSA

### 候選順序

1. 量測 Qwen layer-major threshold。
2. 量測 Prefill step size。
3. 量測 QSA query chunk。
4. 只有 QSA section 超過 request wall time 5% 時，才評估 custom Metal kernel。

DeepSeek DK=512 Flash Attention patch 不適用於 Qwen。
Qwen 使用 head dimension 256 和 QSA micro-block selection。

### 停止條件

- 較大的 step 或 query chunk 增加 peak memory 超過門檻。
- output token hash 不同。
- 改善只出現在單一 prompt。
- custom kernel 的可改善 section 小於 request wall time 5%。

## Q4：Qwen MTP checkpoint 合約查核

### 外部事實

固定 checkpoint config 宣告一個 MTP layer。
官方 model card 把 MTP 列為 4B parameters，並說明它接受 multi-step training。

- [官方 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#model-overview)
- [官方 config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/blob/bcd9f01ddc9cff2316eb84281bebcd5b058bddce/config.json)

外部事實不能證明本機速度。

### 合約查核

1. 列出固定 checkpoint 的全部 `mtp.*` tensor。
2. 記錄 tensor shape、dtype、bytes 和 shard。
3. 找出 official inference graph 或 serving implementation。
4. 定義 MTP input hidden state、embedding、position、cache 和 output head。
5. 確認一個 MTP layer 是否可以獨立產生多個 draft token。
6. 定義 installed model format 和 MXFP4 conversion。
7. 計算額外 storage、slot 和 peak memory 上限。
8. 定義 target verification 和 committed-token metrics。

### Prototype 進入條件

- tensor set 和 graph contract 完整。
- 沒有裁切或猜測 checkpoint stage。
- 一個短 prompt 可以和 reference MTP token distribution 比較。
- 額外 installed model payload 有完整 SHA-256。

### Runtime 採用條件

- 至少五個 workload 的 output token 和 normal greedy reference 完全相同。
- acceptance、draft、verification、replay 和 fallback time 完整記錄。
- 使用 committed token 計算 Decode throughput。
- request wall time 和 Decode 都通過採用門檻。

## 目前進度

```text
Q0  現況盤點                 ██████████ 100%  第一輪完成
Q1  Qwen ready expert         ██████████ 100%  未達 5%，已拒絕
Q2  Qwen LFU decay            ██████████ 100%  current policy 保留
Q3  Prefill／QSA              ██████░░░░ 60%  threshold 已拒絕
Q4  Qwen MTP 合約查核         ██████████ 100%  payload 完成，runtime prototype 已進入
```

目前已確認下列事實：

- 本機有完整 Qwen installed model。
- 本機有 SPEED-Bench 1K、2K、8K、16K 和 32K 資料。
- Qwen selected expert path 目前等待 `get_many` 完成。
- `ExpertCache` 已提供 `iter_ready`。
- Qwen QSA 目前固定使用 4-token query chunk。
- Qwen installed model 已安裝獨立的 `mtp/` 側載資料。

## 2026-08-29 第一輪探索結果

本輪使用 Apple M5 Pro 和 64 GiB 記憶體。
本輪使用 macOS 26.6.2、Python 3.14.7、MLX 0.32.0 和 mlx-lm 0.31.3。
每個量測使用 fresh process。
persistent prompt cache 已關閉。
本輪沒有清除作業系統 page cache。

### Prefill threshold

```text
                  143 tokens       858 tokens       4,577 tokens
                  ──────────       ──────────       ────────────
layer-major       TTFT 9.78 s      TTFT 15.82 s     TTFT 72.44 s
                       │                │                 │
                       ▼                ▼                 ▼
selected          TTFT 2.76 s      TTFT 11.98 s     TTFT 55.37 s
                  改善 71.8%       改善 24.3%       改善 23.6%
```

| Prompt tokens | Control | Candidate | TTFT | Prefill | Request | Peak memory |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 143 | layer-major | selected | -71.8% | +254.0% | -44.8% | +3.5% |
| 858 | layer-major | selected | -24.3% | +32.1% | -20.4% | +12.0% |
| 4,577 | layer-major | selected | -23.6% | +30.8% | -22.3% | +13.2% |

三組 candidate 的 prompt token hash 和 output token hash 都與 control 相同。
143-token candidate 的邏輯 expert bytes 減少 27.5%。
858-token candidate 的邏輯 expert bytes 增加 65.0%。
4,577-token candidate 的邏輯 expert bytes 增加 63.3%。
4,577-token candidate 的 process disk bytes 減少 56.5%。
page cache 未受控制，所以 process disk bytes 不能證明冷啟動 I/O 改善。

相關原始 artifact：

- [`143-token control`](../docs/benchmarks/2026-08-29-qwen-q0-code128-baseline-m5-pro.json)
- [`143-token candidate`](../docs/benchmarks/2026-08-29-qwen-q3-code128-config-default-m5-pro.json)
- [`858-token control`](../docs/benchmarks/2026-08-29-qwen-q3-code858-threshold128-m5-pro.json)
- [`858-token candidate`](../docs/benchmarks/2026-08-29-qwen-q3-code858-threshold1024-m5-pro.json)
- [`4,577-token control`](../docs/benchmarks/2026-08-29-qwen-q0-code4096-baseline-m5-pro.json)
- [`4,577-token candidate`](../docs/benchmarks/2026-08-29-qwen-q3-code4096-threshold8192-m5-pro.json)

### 第一個候選修正（已撤回）

Qwen 原本把 layer-major prefill 門檻固定為 128。
Qwen 原本忽略 `layer_major_prefill_threshold`。

候選修正讓 Qwen 和 DeepSeek 都使用設定中的門檻。
預設門檻仍為 1,024。
8,192 只用於探索量測。
81 個 runtime 測試已通過。

單元測試通過不能證明輸出一致。
後續正式 A/B 發現輸出 token 不同。
候選修正已撤回。
目前 Qwen runtime 仍使用固定 128-token 門檻。

### 證據限制

本輪每組只有一次量測。
本輪 prompt 類型只有 code。
本輪沒有控制作業系統 page cache。
因此本輪不能採用 8,192 作為 Qwen 預設門檻。

## 2026-08-29 正式 threshold A/B

正式測試使用 `control／candidate／candidate／control`。
每個 request 使用新的 process。
作業系統 page cache 沒有清除。

### 4K workload

Control 使用 128-token 門檻。
Candidate 使用 8,192-token 門檻。

| Workload | TTFT | Request | Peak memory | Expert bytes | Output exact |
| --- | ---: | ---: | ---: | ---: | --- |
| code | -21.9% | -20.7% | +13.2% | +63.3% | 是 |
| mixed math | -22.0% | -20.8% | +15.01% | +64.0% | 是 |
| repeated | -29.6% | -27.3% | +10.3% | +14.2% | 是 |
| tool-like | -19.6% | -18.9% | +13.4% | +83.1% | 是 |
| Traditional Chinese technical | -23.7% | -23.8% | +10.3% | +78.9% | 否 |

Traditional Chinese technical 的兩次 control 都產生 9 個 output tokens。
兩次 candidate 都產生 8 個 output tokens。
兩組各自可重現，但兩組的 output token hash 不同。

Mixed math 的 peak memory 增加 15.01%。
此結果也略高於 15% 門檻。

### 短 workload

Control 使用 128-token 門檻。
Candidate 使用 1,024-token 門檻。

| Workload | Prompt tokens | TTFT | Request | Output exact |
| --- | ---: | ---: | ---: | --- |
| code | 143 | -74.4% | -45.8% | 是 |
| mixed math | 134 | -71.6% | -42.6% | 否 |

Mixed math 的兩次 candidate 產生相同 hash。
該 hash 與 control 不同。
測試在發現差異後停止。

### 結論

8,192-token 門檻不通過正確性門檻。
1,024-token 門檻也不通過正確性門檻。
兩個候選都不採用。
正式 machine-readable 結果位於
[`2026-08-29-qwen-prefill-threshold-rejection-m5-pro.json`](../docs/benchmarks/2026-08-29-qwen-prefill-threshold-rejection-m5-pro.json)。

下一個動作是執行 Q1 ready expert decode。
Q1 必須先通過相同的 output token hash 門檻。

## 2026-08-29 Q1 ready expert Decode

候選重用 `ExpertCache.iter_ready`。
候選只改單 token Decode。
Prefill 與 batched expert path 不變。

143-token code workload 使用 `control／candidate／candidate／control`。
四次 output token hash 完全相同。
Expert bytes 和 cache miss 完全相同。

| Metric | Control median | Candidate median | 差異 |
| --- | ---: | ---: | ---: |
| Decode | 9.607 tok/s | 9.955 tok/s | +3.63% |
| Request | 16.079 s | 15.702 s | -2.34% |

Decode 改善低於 5% 採用門檻。
候選已撤回。
原始 artifact 使用
`docs/benchmarks/2026-08-29-qwen-q1-ready-code128-*-m5-pro.json`。

## 2026-08-29 Q2 LFU decay simulation

Qwen route trace 原本沒有記錄 expert ID。
本輪讓 Qwen 使用既有 `record_routes` protocol。
一般 runtime 不啟用 route trace，因此一般 request 不受影響。

本輪使用 128-token repeated prompt。
模型產生 1,025 個 output tokens。
Trace 包含 1,024 個 Decode steps 和 491,520 個 expert assignments。

Simulator 先重播 Prefill handoff cache state。
Simulator 再重播全部 Decode route。
Current policy 的 simulator miss 是 114,821。
Runtime trace 的 miss 也是 114,821。

| Policy | Decay 次數 | Decode miss | 相對 current |
| --- | ---: | ---: | ---: |
| current access-clock | 53 | 114,821 | 0% |
| 每 128 output tokens | 8 | 186,593 | +62.5% |
| 每 512 output tokens | 2 | 215,028 | +87.3% |
| 每 1,024 output tokens | 1 | 215,013 | +87.3% |
| 不 decay | 0 | 215,013 | +87.3% |

所有候選都增加 miss 和 expert bytes。
Q2 在離線 gate 停止。
Current LFU decay 保留。

相關檔案：

- [`LFU simulation`](../docs/benchmarks/2026-08-29-qwen-q2-lfu-simulation-m5-pro.json)
- [`1,024-step route trace`](../docs/benchmarks/2026-08-29-qwen-q2-repeated128-output1025-route-m5-pro.json)
- [`runtime metrics`](../docs/benchmarks/2026-08-29-qwen-q2-repeated128-output1025-metrics-m5-pro.json)
- [`simulator`](../Scripts/simulate_qwen_lfu.py)

Q2 只有一個長 repeated workload。
但所有候選最少增加 62.5% miss。
此結果已符合停止條件，不執行完整模型候選。

## 2026-08-30 Q4 MTP checkpoint 合約查核

本輪只讀取 checkpoint config、index 和 28 個 safetensors header。
本輪沒有下載 tensor payload。
完整 machine-readable tensor 清單位於
[`Q4 checkpoint contract`](../docs/benchmarks/2026-08-30-qwen-q4-mtp-checkpoint-contract-m5-pro.json)。
可重現腳本位於
[`audit_qwen_mtp_checkpoint.py`](../Scripts/audit_qwen_mtp_checkpoint.py)。

### Checkpoint tensor

固定 checkpoint 有 3,101 個 `mtp.*` tensor。
這些 tensor 分布在 28 個 shard。

| 類別 | Tensor | 原始 bytes |
| --- | ---: | ---: |
| Routed expert | 3,072 | 2,516,889,600 |
| Common tensor | 29 | 181,136,896 |
| 合計 | 3,101 | 2,698,026,496 |

1,536 個 expert weight 使用 `F8_E4M3`。
另外 1,536 個 expert scale 使用 `BF16`。
29 個 common tensor 全部使用 `BF16`。

MTP 有 512 個 routed expert。
每個 routed expert 有 `gate_proj`、`up_proj` 和 `down_proj`。
每個 weight 都有一個 `weight_scale_inv`。

Common tensor 包含下列元件：

- 兩個 input norm。
- `fc_embedding` 和 `fc_hidden`。
- MTP input hyper-connection mixer。
- 一個 QSA decoder layer。
- QSA indexer。
- 一個 top-10 MoE layer。
- MTP output hyper-connection mixer。

Config 宣告 `mtp_num_hidden_layers=1`。
Config 也宣告 `mtp_use_dedicated_embeddings=false`。
因此 MTP 共用主模型的 token embedding 和 LM head。

### Serving graph contract

本輪使用 upstream Ollama commit
[`f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a`](https://github.com/ollama/ollama/tree/f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a/x/models/qwen4_exp)
作為可執行 serving graph 來源。
這是外部實作證據，不是本專案速度證據。

Ollama 的 MTP graph 使用下列配對：

```text
target hidden[S]                 token[S+1]
  │  4 × 2560                      │
  │                                ▼
  │                         shared embedding
  │                                │
  ▼                                ▼
hidden norm + fc_hidden      embedding norm + fc_embedding
  │                                │
  └────────────── add ─────────────┘
                   │
                   ▼
       one QSA + MoE decoder layer
                   │
                   ├──► multi-stream hidden for the next MTP step
                   ▼
       output hyper-connection mixer
                   │
                   ▼
              shared LM head
                   │
                   ▼
             one draft token
```

Target 必須輸出 final mixer 之前的 multi-stream hidden state。
這個 hidden state 的 shape 是四個 2,560 維分支。
MTP 先對 hidden state 和 token embedding 分別做 norm 與 projection。
MTP 再把 embedding 加到四個分支。

MTP decoder layer 使用自己的 QSA cache。
Ollama 建立一個 main K/V cache 和一個 indexer side cache。
Indexer side cache 保存 raw index key 和三軸 position。

Prompt Prefill 不能只保留最後一個 target hidden state。
MTP cache 使用 `token[S+1]` 和 `target hidden[S]` 建立一個配對。
因此 MTP cache 會比 target cache 少一個 token。
Runtime 收到下一個 token 後，才可以完成最後一個配對。

一個 MTP forward 只產生一個 draft token。
一個 MTP layer 可以重複執行，產生多個 draft token。
第二步以後使用前一步的 draft token 和 MTP multi-stream hidden state。
所以 `n-max=4` 代表最多連續執行四步。
它不代表一個 forward 同時輸出四個 token。

Target verification 必須一次驗證 current token 和 draft chain。
Greedy 路徑只提交與 target 相同的 draft prefix。
第一個不同位置必須提交 target token。
Runtime 必須回復所有未提交的 MTP cache 和 target cache state。

Formal metrics 必須記錄下列值：

- proposed draft tokens。
- accepted draft tokens。
- committed output tokens。
- acceptance rate 和 accepted length。
- MTP Prefill、draft、target verification、rollback 和 fallback time。
- MTP expert bytes、target expert bytes 和兩個 cache 的 peak memory。
- normal greedy 和 MTP greedy 的 committed output token SHA-256。

Decode throughput 必須使用 committed output tokens。
Runtime 不可把 rejected draft tokens 算成輸出。

### Installed model 與 memory 上限

2026-08-30 的 installed model manifest 已加入 MTP descriptor。
MTP descriptor 有 29 個 common tensor。

MTP routed expert 使用目前 Qwen MXFP4 expert blob。新增 expert 檔案是
1,336,934,400 bytes。
29 個 common tensor 保持 BF16。新增 common 檔案是 181,136,896 bytes。
實際 installed payload 合計是 1,518,071,296 bytes，或 1.414 GiB。

完整安裝證據位於
[`Q4 installed payload`](../docs/benchmarks/2026-08-30-qwen-q4-mtp-installed-payload-m5-pro.json)。

MTP 每一步至少需要 10 個 routed expert slot。
10 個 slot 是 26,112,000 bytes，或 24.90 MiB。
這只是最低需求。
較大的 cache 可能減少 MTP expert miss。

若 prototype 重用目前 `CacheList(KVCache(), KVCache())`，MTP QSA cache 每個 token
需要 2,560 bytes。
這包含 BF16 main K/V 和 duplicated BF16 index K/V。

| Context | MTP QSA cache 上限 |
| ---: | ---: |
| 4,096 | 10 MiB |
| 32,768 | 80 MiB |
| 131,072 | 320 MiB |
| 262,144 | 640 MiB |

這些值不包含 activation、allocator cache 和 MTP expert cache。
這些值不是本機 peak memory 量測。

### `mihailescu2m/llama.cpp` claim audit

查核 commit 是
[`45d140415f3d77bcc5261af674c2675b6373cb73`](https://github.com/mihailescu2m/llama.cpp/tree/45d140415f3d77bcc5261af674c2675b6373cb73)。
MTP 實作 commit 是
[`0640e7421dd2578d54b55014aa41a5bd9826ff90`](https://github.com/mihailescu2m/llama.cpp/commit/0640e7421dd2578d54b55014aa41a5bd9826ff90)。

該 repo 確實實作下列功能：

- Qwen MTP tensor loader。
- MTP graph。
- target multi-stream hidden state export。
- MTP KV cache。
- speculative verification 和 rollback。
- MTP prompt 長度 gate。

該 repo 的 Qwen 研究文件報告三個不同數字：

- 無 MTP 的 graph 優化：9.548 到 11.522 tok/s，改善 20.7%。
- MTP：11.25 到 15.60 tok/s，改善約 39%。
- 舊整體數字：9.55 到 16.3 tok/s，改善約 71%。

這些數字使用 M1 Max、64 GiB、`UD-Q3_K_XL`、32 GiB expert cache、
`Q4_0all` MTP sidecar 和 warm 4,751-token prompt。
這些條件與本專案的 M5 Pro、官方 FP8 checkpoint 和 MXFP4 installed model 不同。

該 repo 的目前文件明確標示 16.3 tok/s 尚待重新驗證。
原因是 MTP hidden-state export 曾經缺失，後來才修正。
該 repo 只保存研究敘述，沒有保存對應的 Qwen machine-readable raw result。
文件只說 output coherent。
文件沒有提供 normal greedy 與 MTP greedy 的 output token hash parity。

該 repo 的 MTP graph 還有一個重要差異。
該 repo 刻意讓 MTP 使用 dense attention。
該 repo 載入 checkpoint 的 QSA indexer tensor，但不執行 indexer。
Upstream Ollama 的 Qwen MTP graph 會執行 QSA，並使用 indexer side cache。

因此，本研究把該 repo 的速度數字視為有實作依據的外部結果。
本研究不把該數字視為已獨立重現的結果。
本研究也不把 dense MTP graph 視為 checkpoint-faithful reference。

### Q4 決策

Checkpoint tensor、input、cache、output head 和多步 draft contract 已完成。
Q4 合約查核結束。

Runtime prototype 已完成 manifest loader、最小 QSA graph 和 cache contract 測試。
可執行的 QSA reference MTP token distribution gate 已通過。

Repack plan、FP8 到 MXFP4 轉換、安裝和 SHA-256 驗證已完成。
Python manifest loader 已讀取 MTP descriptor 和兩個 MTP 檔案。
最小 MTP graph 已嚴格載入 29 個 common tensor。
Graph 使用 QSA main K/V cache 和 indexer side cache。
`hidden[S]`／`token[S+1]` Prefill 配對和雙快取 rollback 測試已通過。
第一次 reference 比較找出 `pre_fc_norm_hidden` 的錯誤。
本機 graph 原本分成四組做 RMSNorm。
固定 Ollama graph 對完整 10,240 維 hidden 做一次 RMSNorm。
本機 graph 已修正。

修正後的 `Hello world` 比較結果如下：

- argmax token 相同。
- top 10 token 和順序相同。
- distribution cosine similarity 是 `0.999992847442627`。
- distribution maximum absolute error 是 `0.125`。
- graph 在 MoE input 前相同。
- 第一個數值差異出現在兩個 runner 的 MXFP4 MoE kernel。

Reference parity 證據位於
[`Q4 Ollama reference parity`](../docs/benchmarks/2026-08-30-qwen-q4-mtp-ollama-reference-parity-m5-pro.json)。
舊的本機 distribution artifact 已標示為 superseded。

下列檔案可重現比較：

- [`compare_qwen_mtp_reference.py`](../Scripts/compare_qwen_mtp_reference.py) 產生輸入並比較結果。
- [`qwen_mtp_ollama_reference_test.go`](fixtures/qwen_mtp_ollama_reference_test.go) 在固定 Ollama commit 中執行 reference graph。

執行者必須把 Go fixture 複製到固定 Ollama checkout 的
`x/models/qwen4_exp/`。
執行者必須使用 Ollama commit 固定的 MLX 和 MLX-C 版本。

Prototype 進入條件已全部完成。
Speculative control flow 已接入 runtime。

## 2026-08-30 Q4 MTP runtime prototype

Prototype 預設關閉，且只支援 greedy decoding。
每個 round 最多提出 5 個 draft token。
Runtime 只提交 target verification 接受的 draft prefix。
第一個不相同的位置提交 target token。
一個 round 接受 0 個 draft token 時，該 request 停用 MTP。

第一次五種 workload 測試找到兩個 correctness 問題：

1. 32-slot MTP cache 無法處理一次包含 106 個 unique expert 的 Prefill。
2. MTP Target Prefill 使用 selected-expert 路徑，但 normal greedy 使用 layer-major 路徑。

Runtime 現在把 MTP Prefill 切成最多 `mtp_slots / 10` 個 token。
Runtime 也讓 MTP 使用和 normal greedy 相同的 Target Prefill 路徑。
先前的 block verification 差異來自 Target Prefill 路徑不一致。
Target Prefill 修正後，block logits 與逐 token logits 相同。
Runtime 現在一次驗證 anchor 和最多 5 個 draft token。
完整接受時，runtime 會提交 block cache。
部分接受時，runtime 只會逐 token 重播已接受的 prefix。

最終 correctness matrix 使用 code、mixed math、repeated、tool-like 和
Traditional Chinese technical workload。
每個 prompt 約 128 個 token。
每個 run 最多產生 64 個 token。
每次使用 fresh process，並停用 persistent prompt cache。

- 五種 workload 全部通過 output token parity。
- 320 個 committed output token 全部相同。
- MTP 提出 205 個 draft token。
- MTP 接受 156 個 draft token。
- 整體接受率是 76.10%。

結果位於
[`MTP block greedy output parity`](../docs/benchmarks/2026-08-30-qwen-mtp-block-greedy-parity-128-m5-pro.json)。
該 artifact 明確標示為非正式效能結果。

Block verifier 通過 correctness matrix 後，研究執行 4K 正式 stop gate。
Stop gate 使用 `code` workload 和 64 個 output token。
執行順序是 `control／candidate／candidate／control`。
每次使用 fresh process，並停用 persistent prompt cache。
作業系統 page cache 未受控制。
交錯順序用來降低執行順序偏差。

- 四次 output token hash 完全相同。
- paired median Decode throughput 改善 65.07%。
- paired median request wall time 變差 7.34%。
- paired median TTFT 變差 13.10%。
- paired median expert bytes／generated token 增加 37.46%。
- paired median P95 token latency 增加 153.13%。
- paired median peak memory 增加 1.22%。

Candidate 未通過 request、TTFT、expert bytes 和 P95 採用門檻。
研究依停止條件未執行其餘四種 4K workload。
這是單一 workload 的正式 stop gate，不是五種 workload 的完整效能矩陣。
結果位於
[`MTP block 4K formal stop gate`](../docs/benchmarks/2026-08-30-qwen-mtp-block-formal-stop-code4k64-m5-pro.json)。

Block verifier 保留。
MTP prototype 維持預設關閉。

## 2026-08-30 Q4 MTP bounded Prefill

第一個 follow-up 把 MTP cache 從 32 slots 增加到 512 slots。
完整 4K MTP Prefill 的 MTP expert bytes 從約 49.1 GB 降到約 399.5 MB。
這個配置可以讓目前 workload 使用的 MTP routed expert 保持 resident。

512-slot 完整 Prefill 的正式 stop gate 使用 4K `code` workload。
每個 run 產生 64 個 token。
執行順序是 `control／candidate／candidate／control`。

- 四次 output token hash 完全相同。
- paired median Decode throughput 改善 40.36%。
- paired median request wall time 變差 2.67%。
- paired median TTFT 變差 6.04%。
- paired median expert bytes／generated token 改善 2.99%。
- paired median P95 token latency 變差 216.64%。
- paired median peak memory 增加 1.23%。

完整 Prefill candidate 未通過 request、TTFT 和 P95 門檻。
結果位於
[`512-slot full Prefill stop gate`](../docs/benchmarks/2026-08-30-qwen-mtp-prefill-slots512-formal-stop-code4k64-m5-pro.json)。

第二個 follow-up 保留 512 slots，但只建立 MTP tail cache。
Tail token 上限是 `mtp_slots / num_experts_per_tok`。
512 slots 對應 51 個 tail token。
Target hidden state 仍包含完整 4K prompt context。
Target verifier 仍決定全部 committed token。

Bounded Prefill 的正式 stop gate 使用相同 workload 和執行順序。

- 四次 output token hash 完全相同。
- MTP 提出 108 個 draft token，接受 104 個。
- MTP acceptance rate 是 96.30%。
- paired median Decode throughput 改善 40.01%。
- paired median request wall time 改善 2.48%。
- paired median TTFT 變差 0.30%。
- paired median expert bytes／generated token 改善 3.06%。
- paired median P95 token latency 變差 212.95%。
- paired median peak memory 增加 1.22%。

Bounded Prefill 未通過 5% request 改善門檻和 P95 門檻。
研究依停止條件未執行其餘四種 4K workload。
結果位於
[`bounded Prefill formal stop gate`](../docs/benchmarks/2026-08-30-qwen-mtp-bounded-prefill-formal-stop-code4k64-m5-pro.json)。

Runtime 保留 default-off bounded Prefill 研究路徑。
Runtime 不把 MTP 或 512 slots 設為預設值。
目前採用門檻下，不應繼續調整 MTP Prefill window 或 cache slots。
新的 MTP 方向必須改變 verifier burst latency，或先定義不同的 throughput workload。

32-slot bounded Prefill 另執行一次 4K `code` output parity。
64 個 output token 全部相同。
第一個 round 接受 0／5 個 draft token，request 隨後 fallback。
單次 request wall time 變差 0.83%，Decode throughput 也變差。
這個 run 只證明目前 default-off 配置的 correctness。
它不是正式效能結果。
結果位於
[`32-slot bounded Prefill parity`](../docs/benchmarks/2026-08-30-qwen-mtp-bounded-prefill-slots32-code4k64-parity-m5-pro.json)。

## 2026-08-30 Q4 MTP verifier burst latency

正式 bounded Prefill trace 顯示 verifier 是主要 round latency：

- MTP draft 每 round 約 0.033 秒。
- Target block verification 每 round 約 0.426 秒。
- 部分接受 replay 只出現在最後的部分接受 round。
- Candidate P95 token latency 是約 0.484 秒。

研究依序把每 round draft 上限從 5 降到 2，再降到 1。
每次只改一個變數。

Draft-2 的單次 4K `code` 探索結果如下：

- 64 個 output token 與 control 完全相同。
- P95 token latency 是 0.352 秒，比 control 增加約 127%。
- Decode throughput 改善 18.13%。
- Request wall time 改善 1.35%。

結果位於
[`Draft-2 latency`](../docs/benchmarks/2026-08-30-qwen-mtp-draft2-latency-code4k64-exploratory-m5-pro.json)。

Draft-1 是同步 verifier 的最小 verification block。
單次結果如下：

- 64 個 output token 與 control 完全相同。
- P95 token latency 是 0.289 秒，比 control 增加 87.48%。
- Decode throughput 變差 4.03%。
- Request wall time 改善 0.14%。

結果位於
[`Draft-1 latency`](../docs/benchmarks/2026-08-30-qwen-mtp-draft1-latency-code4k64-exploratory-m5-pro.json)。

Draft-1 和 Draft-2 都未通過 P95 門檻。
Draft-1 也失去 Decode throughput 改善。
Runtime 已恢復每 round 最多 5 個 draft token。
同步 verifier 的 block 大小方向停止。
新的工作必須使用不同的 throughput workload contract，或改變同步驗證模型。
