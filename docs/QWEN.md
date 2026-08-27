# Qwen3.8-Flash-Next 支援

本文件描述目前的 Qwen text model 支援。
本文件不宣稱 vision、video、MTP 或 DSpark 支援。

## 固定 checkpoint 合約

| 欄位 | 值 |
| --- | ---: |
| Model ID | `Qwen/Qwen3.8-Flash-Next-FP8` |
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

Swift inspector 會檢查 QSA、Gated DeltaNet、hyper-connection、PLE、N-gram
和 MoE 主要 shape。Inspector 會排除 `model.visual.*` 和 `mtp.*`。

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

每個 expert blob 依序包含：

1. `gate_up.weight`
2. `gate_up.scale`
3. `down.weight`
4. `down.scale`

repacker 把連續 FP8 expert tensor 合併成最多 64 MiB 的 range request。
repacker 讓每個 checkpoint shard 的 BF16 inverse scale 只下載一次。
這會把完整 expert 轉換的 request 數從約 147,000 降到約 2,000。
repacker 不保存完整 FP8 expert 暫存檔。
receipt 會記錄 conversion version 和轉換後 digest。
repair 只重建失敗的 installed file。

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

prompt cache 路徑會包含 model ID、revision 和 manifest format。
DeepSeek 和 Qwen 不會共用 prompt cache entry。

## 對話與 API 預設值

Qwen 使用 installed tokenizer 的官方 `chat_template.jinja`。
Qwen tool call 使用官方 XML 格式。
完整 parser 和 streaming parser 會驗證相同的 function name 和 arguments。

Qwen reasoning effort 使用 `low`、`medium` 和 `xhigh`。
API 的 `high` 與 `max` 會對應到 `xhigh`。

| 設定 | Qwen 預設值 |
| --- | ---: |
| temperature | 1.0 |
| top-p | 0.95 |
| top-k | 20 |

明確 request 欄位會覆寫預設值。
Qwen 不支援 `--dspark`。

`memory_limit_gib=0` 會選擇模型安全自動上限。
Qwen 的自動上限不超過 48 GiB。
DeepSeek 的既有規則不變。

## 目前驗證邊界

2026-08-27 已通過下列完整驗證：

- 57 個 manifest files 的完整 SHA-256。
- 短文字 completion。
- thinking 關閉與開啟。
- forced tool call 和 streaming tool call。
- greedy 4,096-token prompt。
- cold 與 warm prompt cache output token hash 比較。
- packaged App、解壓後 ZIP、三種 localization、隔離啟動和 runtime import。

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
