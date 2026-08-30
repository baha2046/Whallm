# Qwen3.8-Flash-Next 支援

本文件描述目前的 Qwen text model 支援。
本文件不宣稱 vision、video 或 DSpark 支援。
Runtime 提供預設關閉的 MTP speculative decoding prototype。
Qwen3.8 的 Advanced Settings 可以啟用 MTP。
App 只會在 installed model 包含 MTP sidecar 時啟用 MTP。

## 固定 checkpoint 合約

| 欄位 | 值 |
| --- | ---: |
| checkpoint model ID | `Qwen/Qwen3.8-Flash-Next-FP8` |
| API model ID | `qwen3.8-flash-next-fp8` |
| Revision | `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` |
| manifest format | 2 |
| `modelKind` | `qwen3.8-flash-next` |
| text layers | 48 |
| routed experts per layer | 512 |
| selected experts per token | 10 |
| hidden size | 2,560 |
| MoE intermediate size | 640 |
| checkpoint maximum context | 262,144 |
| checkpoint routed expert dtype | F8_E4M3，128×128 BF16 inverse scale |
| installed routed expert dtype | MXFP4, 4 bits, group size 32 |
| installed N-gram dtype | F8_E4M3，BF16 `weight_scale` |

Swift 主模型 inspector 會檢查 QSA、Gated DeltaNet、hyper-connection、PLE、N-gram
和 MoE 主要 shape。主模型 repack plan 會排除 `model.visual.*` 和 `mtp.*`。
獨立的 MTP repack plan 會檢查與安裝 `mtp.*`。

外部規格來源是[固定 revision 的官方 FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/bcd9f01ddc9cff2316eb84281bebcd5b058bddce)。
本地 runtime 參考 Transformers 的
[`Qwen4Exp` text model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)。

## Installed model

```text
qwen3.8-flash-next.dsv4/
  manifest.json
  common.bin
  ngram.bin
  config.json
  generation_config.json
  tokenizer/
    tokenizer.json
    tokenizer_config.json
    chat_template.jinja
    vocab.json
    merges.txt
  experts/
    layer_00.bin
    ...
    layer_47.bin
  mtp/
    common.bin
    experts/
      layer_00.bin
```

2026-08-27 的完整安裝確認下列值。

| 資料 | Bytes |
| --- | ---: |
| `common.bin` | 9,895,409,152 |
| `ngram.bin` | 51,200,245,760 |
| 48 個 MXFP4 expert layer | 64,172,851,200 |
| installed weight files 合計 | 125,268,506,112 |
| manifest 內全部檔案合計 | 125,291,490,955 |
| 選取的 checkpoint tensor payload | 181,906,343,706 |
| 每個 expert blob | 2,611,200 |

2026-08-30 已在本機安裝可選的 Qwen MTP 側載資料。

| MTP 資料 | Bytes | SHA-256 |
| --- | ---: | --- |
| `mtp/common.bin` | 181,136,896 | `2604d969eabc990b83830581c85d1bb647f4994810939ded5637480b60fb6641` |
| `mtp/experts/layer_00.bin` | 1,336,934,400 | `8c4b24665ee3a74c86880d25dd44939232447fc4b4b2cd183e7bc0730aa06da2` |
| MTP 合計 | 1,518,071,296 | — |

安裝後的 manifest 有 59 個檔案。
這些檔案合計 126,809,562,251 bytes。
MTP descriptor 定義 1 個 layer、29 個 common tensor，並共用主模型 embedding。

MTP loader 會嚴格載入全部 29 個 common tensor。
Graph 使用 QSA main K/V cache 和 indexer side cache。
Runtime 會執行 Prefill、最多 5 個 draft token、target verification、rollback
和 fallback。
短 prompt 的 MTP Prefill 會依 `mtp_slots` 切分 expert request。
Layer-major 長 prompt 只會建立 MTP tail cache。
Tail token 上限是 `mtp_slots / 10`。
Target Prefill 會使用和 normal greedy 相同的 layer-major 規則。
Target verification 會一次處理 anchor 和最多 5 個 draft token。
完整接受時，runtime 會提交 block cache。
部分接受時，runtime 只會逐 token 重播已接受的 prefix。
MTP 不會重用一般 prompt cache。
一個 round 接受 0 個 draft token 時，該 request 會回退到 normal greedy。
`Hello world` 的 MTP distribution 已和固定 Ollama commit 比較。
兩個 runner 的 argmax token 和 top 10 token 順序相同。
Distribution cosine similarity 是 `0.999992847442627`。
Reference 比較也修正了 `pre_fc_norm_hidden` 的 RMSNorm 範圍。
完整結果位於
[`Q4 Ollama reference parity`](benchmarks/2026-08-30-qwen-q4-mtp-ollama-reference-parity-m5-pro.json)。

2026-08-30 的 block verifier correctness matrix 使用五種 128-token workload。
每種 workload 產生 64 個 token。
Normal greedy 和 MTP greedy 的 320 個 token 全部相同。
MTP 提出 205 個 draft token，接受 156 個。
整體接受率是 76.10%。
完整結果位於
[`MTP block greedy output parity`](benchmarks/2026-08-30-qwen-mtp-block-greedy-parity-128-m5-pro.json)。
該 artifact 不是正式效能結果。

4K 正式 stop gate 使用 `code` workload 和 64 個 output token。
執行順序是 control、candidate、candidate、control。
四次輸出的 token hash 完全相同。
paired median 的 Decode throughput 改善 65.07%。
request wall time 變差 7.34%，TTFT 變差 13.10%。
Expert bytes／generated token 增加 37.46%。
P95 token latency 增加 153.13%。
Peak memory 增加 1.22%。
Candidate 未通過 request、TTFT、expert bytes 和 P95 採用門檻。
研究依停止條件未執行其餘四種 4K workload。
結果位於
[`MTP block 4K formal stop gate`](benchmarks/2026-08-30-qwen-mtp-block-formal-stop-code4k64-m5-pro.json)。
MTP prototype 維持預設關閉。

後續 Prefill 研究先把 MTP cache 增加到 512 slots。
完整 4K MTP Prefill 的 MTP expert bytes 從約 49.1 GB 降到約 399.5 MB。
Request wall time 仍變差 2.67%，所以完整 Prefill candidate 停止。

Bounded Prefill 隨後只保留 51 個 tail token。
正式 stop gate 使用相同的 4K `code` workload 和四次交錯順序。
四次輸出的 token hash 完全相同，MTP acceptance rate 是 96.30%。
paired median 的 Decode throughput 改善 40.01%。
Request wall time 改善 2.48%，未達 5% 門檻。
TTFT 變差 0.30%，expert bytes／generated token 改善 3.06%。
Peak memory 增加 1.22%，P95 token latency 變差 212.95%。
Candidate 因 request 和 P95 門檻停止。
結果位於
[`MTP bounded Prefill formal stop gate`](benchmarks/2026-08-30-qwen-mtp-bounded-prefill-formal-stop-code4k64-m5-pro.json)。
Runtime 保留 default-off bounded Prefill 研究路徑，但不把 MTP 或 512 slots 設為預設值。

32-slot bounded Prefill 也通過 4K `code` 的 64-token output parity。
第一個 round 接受 0／5 個 draft token，request 隨後 fallback。
該單次 candidate 的 request wall time 變差 0.83%，Decode throughput 也變差。
這是正確性證據，不是正式效能結果。
結果位於
[`32-slot bounded Prefill parity`](benchmarks/2026-08-30-qwen-mtp-bounded-prefill-slots32-code4k64-parity-m5-pro.json)。

Verifier burst latency follow-up 量測 512-slot bounded Prefill。
原本每 round 最多提出 5 個 draft token。
該配置的 P95 token latency 是約 0.484 秒。
Draft-2 把 P95 降到 0.352 秒，但仍比 control 增加約 127%。
Draft-2 的 Decode throughput 只改善 18.13%。
Draft-1 把 P95 降到 0.289 秒，但仍比 control 增加 87.48%。
Draft-1 的 Decode throughput 變差 4.03%。
兩個 candidate 都通過 output token parity。
同步 verifier 已沒有更小的 verification block。
Runtime 因此保留每 round 最多 5 個 draft token。
兩個探索結果位於
[`Draft-2 latency`](benchmarks/2026-08-30-qwen-mtp-draft2-latency-code4k64-exploratory-m5-pro.json)
和
[`Draft-1 latency`](benchmarks/2026-08-30-qwen-mtp-draft1-latency-code4k64-exploratory-m5-pro.json)。

CLI 可以明確啟用 prototype：

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model /path/to/qwen3.8-flash-next.dsv4 \
  --prompt "Hello" \
  --mtp \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20
```

`--mtp-slots` 預設是 32，最小值是 10。
MTP 支援 Temperature、Top P、Top K、Min P 和 logit processor。
Sampled verification 使用 `min(1, p/q)` 接受 draft token。
拒絕時，runtime 從 `normalize(max(p-q, 0))` correction 分布抽樣。
研究假設、停止條件和目前證據記錄在
[`Qwen MTP sampled decoding`](../research/QWEN_MTP_SAMPLED_DECODING_2026-08-30.md)。

下列命令會下載、轉換、續傳及驗證 MTP 側載資料。

```sh
swift run -c release dsv4-repack install-mtp \
  --model /path/to/qwen3.8-flash-next.dsv4
```

每個 expert blob 依序包含：

1. `gate_up.weight`
2. `gate_up.scale`
3. `down.weight`
4. `down.scale`

## App 安裝來源

App 從
[`Yanun/Qwen3.8-Flash-Next-MXFP4`](https://huggingface.co/Yanun/Qwen3.8-Flash-Next-MXFP4)
下載 published installed model。
程式碼固定 Hugging Face commit
`753d0aa57059fad70a5f7e6cc249f25df56bbd34`。
遠端 `manifest.json` 的 SHA-256 是
`3f4cb52a88335591cfb8233778eb56396e9d2756c153485e4dcfb0a12daed0ce`。
App 不在使用者的 Mac 上執行 FP8 到 MXFP4 轉換。
App 安裝不使用 GPU。
App 使用 CPU 計算 SHA-256。

App 使用四個並行 file 工作。
每個工作使用 8 MiB range 續傳一個 file。
receipt 記錄已完成 file 的 SHA-256。
App 完成下載後，App 會驗證 manifest 合約、每個 file size 和每個 SHA-256。
完整驗證通過後，App 才完成安裝。
repair 只重新下載損壞的 installed file。

App 下載 125,291,490,955 bytes 的 manifest files。
這比從官方 checkpoint 選取的 181,906,343,706 tensor payload bytes
少 56,614,852,751 bytes。

## Artifact 產生

`dsv4-repack` 仍可從固定的官方 FP8 checkpoint 產生 installed model。
repacker 把連續 FP8 expert tensor 合併成最多 64 MiB 的 range request。
repacker 讓每個 checkpoint shard 的 BF16 inverse scale 只下載一次。
這會把完整 expert 轉換的 request 數從約 147,000 降到約 2,000。
repacker 不保存完整 FP8 expert 暫存檔。
receipt 會記錄 conversion version 和轉換後 digest。
這個轉換由 CPU 執行。

## Runtime

manifest loader 使用 `modelKind` 選擇 project-local `qwen4_exp` runtime。
runtime 包含下列 text model 元件：

- Gated DeltaNet。
- QSA attention 和 decode cache。
- gated residual hyper-connection。
- shared expert 和 top-10 routed experts。
- N-gram hashing、PLE 和 dilated depthwise convolution。

QSA 會把 query 分成最多 4 個 token 的 chunk。
runtime 不建立完整 262K attention matrix。

`ngram.bin` 使用 read-only memory map。
runtime 只複製目前 token chunk 所需的 FP8 row。
runtime 套用 checkpoint 的 BF16 `weight_scale`，然後轉成 BF16。

Qwen layer-major prefill 每層只載入一次完整 expert layer。
同一層的 prompt chunk 共用該次 batched MXFP4 buffer。
少於 128 個未快取 token 時，runtime 使用 selected expert cache。
短 prompt 不會載入完整 expert layer。

啟用 expert route trace 時，Qwen 會記錄 Prefill histogram、完整 Decode route
和 Decode miss。一般 request 不會建立 route trace。

prompt cache 路徑會包含 model ID、revision 和 manifest format。
DeepSeek 和 Qwen 不會共用 prompt cache entry。

Qwen runtime 不支援 BF16 KV cache 選項。
App 不會在 Qwen 的進階設定中顯示這個選項。

## 對話與 API 預設值

Qwen 使用 installed tokenizer 的官方 `chat_template.jinja`。
Qwen tool call 使用官方 XML 格式。
完整 parser 和 streaming parser 會驗證相同的 function name 和 arguments。

Codex Responses request 會先轉成 Qwen prompt 所需的格式。

| Codex Responses 格式 | Qwen prompt 格式 |
| --- | --- |
| 非空的 `instructions`、`system` 和 `developer` message | 保留 instruction 順序。合併成唯一的開頭 `system` message。 |
| 沒有 `user` message 的 history | 在 `system` message 後加入空的 `user` anchor。 |
| text content item array | 合併成 string。 |
| top-level function tool | 轉成 OpenAI function tool shape。 |
| namespace tool | 使用 `namespace__name` 作為 prompt 內的 function name。 |
| `web_search` tool | 不加入 prompt。 |
| `function_call.arguments` JSON string | server 驗證 JSON object。`QwenToolCodec` 在 message 深層複本中解碼成 `dict`。 |
| `function_call_output.output` string | 轉成 `tool` message content。server 使用 `call_id` 驗證順序。 |
| message、reasoning、tool definition 和 tool history 中的 Qwen 保留標記 | 把開頭的 `<` 轉成 `&lt;`。tool call parser 會還原 parameter value。 |
| replay 的 `reasoning` item | 接受該 item，但不加入 prompt。 |

`QwenToolCodec` 不會修改原始 request message。
同一個 assistant message 可以包含多個 function call。
空參數使用 JSON string `"{}"`，並轉成空 `dict`。
request 有 tools 時，codec 會要求 model 使用相同的保留標記跳脫規則。
Qwen parameter name 可以包含空白和 Unicode。
Qwen XML 無法表示含有 `<`、`>`、CR 或 LF 的 parameter name。

Qwen `chat_template.jinja` 會移除每個 message content 外側的空白。
codec 不會偽造被 Responses normalization 移除的 reasoning item。
目前 text-only API 會拒絕 image、file 和其他 structured content。
`client_metadata`、`include` 和 `prompt_cache_key` 不會加入 Qwen prompt。

Qwen reasoning effort 使用 `low`、`medium` 和 `xhigh`。
API 的 `high` 與 `max` 會對應到 `xhigh`。

取樣參數來自 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices)。

| 模式 | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 思考 | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| 非思考 | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |

Qwen 使用 OR 規則判定模式。
`thinking_mode == "thinking"` 會選取思考模式。
Chat Completions 的非 `none` `reasoning_effort` 也會選取思考模式。
Responses 的非 `none` `reasoning.effort` 也會選取思考模式。
`thinking_mode: "chat"` 不會抵銷非 `none` effort。
其他 request 使用非思考模式。
Text Completions 固定使用非思考模式。

明確的 `temperature`、`top_p` 和 `top_k` 會覆寫模式配置。
明確的 `temperature: 0` 會使用 greedy sampling。
正數 `temperature` 會使用 categorical sampling。
`max_tokens` 行為不變。

`min_p`、`presence_penalty` 和 `repetition_penalty` 不是公開 request 欄位。
非思考模式會建立 `presence_penalty=1.5` processor。
思考模式不會建立 presence processor。
`repetition_penalty=1.0` 不會建立 processor。

目前 [pinned `mlx-lm` 的 presence processor](https://github.com/Blaizzy/mlx-lm/blob/5c10538136b9038b9626c134612b08afc18d697a/mlx_lm/sample_utils.py#L315-L338)
只檢查 generation call 可見的最近 20 個 token。
processor 不會檢查 prompt cache 已重用的 prefix。
因此，這項配置不是完整 prompt 的重複偵測器。

App 的 Qwen 進階設定只顯示 Max tokens。
App 不顯示 Temperature、Top P 或 Top K。
Qwen 儲存設定會在 normalization 時更新為 `0.7 / 0.8 / 20`。
DeepSeek 仍顯示並使用三個欄位。

Qwen 不支援 `--dspark`。
Qwen 可以使用預設關閉的 `--mtp` speculative decoding prototype。
Qwen3.8 的 Advanced Settings 提供 `Use MTP` 和 `MTP slots`。
`Use MTP` 預設關閉，`MTP slots` 預設為 32。
啟用 MTP 時，App 產生的 model catalog 會設定 `mtp_enabled=true`。
App 不會變更 Qwen 的 sampling 設定。
Installed model 沒有 MTP sidecar 時，App 會停用 `Use MTP`。
Installed model 沒有 MTP sidecar 時，模型列會顯示 MTP 下載按鈕。
App 使用既有的 MTP installer 加入 sidecar，且不重新下載 Qwen 主模型。
中斷的 MTP 下載會保留 `.mtp-install.partial`，下次可以繼續下載。

`memory_limit_gib=0` 會選擇模型安全自動上限。
Qwen 的自動上限不超過 48 GiB。
DeepSeek 的既有規則不變。

## 目前驗證邊界

2026-08-27 的主模型已通過下列完整驗證：

- 57 個 manifest files 的完整 SHA-256。
- 短文字 completion。
- thinking 關閉與開啟。
- forced tool call 和 streaming tool call。
- greedy 4,096-token prompt。
- cold 與 warm prompt cache output token hash 比較。
- packaged App、解壓後 ZIP、三種 localization、隔離啟動和 runtime import。

上述 packaged App 驗證早於 direct installed artifact 下載路徑。
目前 direct 下載路徑已通過 file 續傳和 SHA-256 單元測試。
目前尚未重新執行完整 125 GB 的 App direct download。

2026-08-30 已通過 254 個 Python 測試和 64 個 Swift 測試。
MTP 測試包含 Prefill 分段、cache rollback、greedy 全部接受、零接受 fallback、
sampled 全部接受和 sampled correction 分布。

CLI 使用預設 `memory_limit_gib=0` 啟動完整模型時，runtime 套用 48 GiB
自動上限。5-token prompt 產生 ` Paris`。runtime 從 SSD 讀取
4,658,380,800 bytes 的 routed expert。MLX peak memory 是 13,048,438,112
bytes。request 結束時的 active memory 是 13,040,928,562 bytes。

修正前，短 prompt 會讓 48 層完整 expert buffer 留在同一個 lazy graph。
MLX peak memory 是 74,069,529,060 bytes。
修正後的輸出 token SHA-256 與修正前相同。

目前程式的 4K prompt-cache cold run 是 65.42 秒。
MLX peak memory 是 15,182,206,210 bytes。
warm prompt cache 重用 4,095 個 token，並在 0.214 秒產生相同 token。
兩次 output token SHA-256 都是
`6dfb97632210ac38a071667cf8be7df83a16178e12f1248e45b2a3d24b3b2bd1`。
這些數值是一個 M5 Pro 本機驗證結果。
這些數值不是效能保證。

完整結果位於
[`benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json`](benchmarks/2026-08-27-qwen3.8-flash-next-fp8-m5-pro.json)。

262,144 只代表 checkpoint 合約上限。
目前完整模型只驗證到 4,096 prompt tokens。
MXFP4 routed expert 輸出不保證等同官方 FP8。
