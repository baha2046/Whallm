# Whallm

<p align="center">
  <img src="Packaging/AppIcon.png" alt="Whallm App Icon" width="160">
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

Inspired by [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare),
Whallm lets an M-series Mac run all 284B parameters of the pinned
`DeepSeek-V4-Flash-0731` checkpoint by streaming routed experts from SSD. It
also supports the pinned `DeepSeek-V4.1-Flash` and
`Qwen3.8-Flash-Next-FP8` text checkpoints.

## Memory guidance

| Model | Recorded peak memory | expert cache slots | chipset
| --- | ---: | ---: | --:
| `DeepSeek-V4-Flash-0731` |  23 GiB | 1152 | M5 Pro
| `Qwen3.8-Flash-Next-FP8` | 18 GiB | 3072 | M5 Pro
| `DeepSeek-V4.1-Flash` | 33 GiB | 1152 | M2 Max

These v1.1.4 results cover chat prompts with 1,024 to 16,384 input tokens. They
are measurements, not minimum memory requirements or performance guarantees.
Prompt length, tools, cache state, and runtime settings can change peak memory.
See the [benchmark](BENCHMARK.md) and
[validation record](docs/VALIDATION.md) for the measured workloads.

## Benchmark 
### M5 Pro (v1.1.7)
| Model | Context | Slots | Output limit | Input / Output | TTFT (ms) | TPOT (ms) | PP tok/s | TG tok/s | Total (s) | Throughput | Peak MLX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.8-flash-next-fp8 | Code | 3072 | 128 | 1024 / 128 | 15229.3 | 102.5 | 67.2 | 9.8 | 28.3 | 40.8 | 16.88 GB |
|                        |      |      |     | 4096 / 128 | 34024.5 | 119.0 | 120.4 | 8.4 | 49.1 | 86.0 | 16.97 GB |
|                        |      |      |     | 8192 / 128 | 59609.8 | 112.2 | 137.4 | 8.9 | 73.9 | 112.6 | 17.10 GB |
|                        |      |      |     | 16384 / 128 | 113580.3 | 123.1 | 144.3 | 8.1 | 129.2 | 127.8 | 17.36 GB |
| deepseek-v4-flash-0731 | Code | 1152 | 128 | 1024 / 128 | 17402.8 | 130.1 | 58.8 | 7.7 | 33.9 | 33.9 | 22.77 GB |
|                        |      |      |     | 4096 / 128 | 23613.8 | 167.0 | 173.5 | 6.0 | 44.8 | 94.2 | 22.80 GB |
|                        |      |      |     | 8192 / 128 | 48800.2 | 162.5 | 167.9 | 6.2 | 69.4 | 119.8 | 22.83 GB |
|                        |      |      |     | 16384 / 128 | 89513.5 | 213.7 | 183.0 | 4.7 | 116.7 | 141.5 | 22.90 GB |

### M2 Max (v1.1.7)


## How to use it

**Download the app → Open the app → Select and download a model → Start the
server → Chat in the app or connect Codex**

1. Download the latest `Whallm-macOS-arm64.zip` from
   [GitHub Releases](https://github.com/yanun0323/Whallm/releases/latest).
2. Extract the ZIP and open `Whallm.app`.
3. Open the **Model** page. Select DeepSeek V4, DeepSeek V4.1, or Qwen, and then select
   **Download Model**. The app checks the required storage. Qwen downloads the
   published MXFP4 installed model. You can stop the download and resume it
   later.
4. Open the **Server** page and select **Start Server**. The server can start
   with no installed model, but generation needs an installed model.
5. Open the chat and select a model. You can also connect Codex with the
   configuration below.

The local server starts at `http://127.0.0.1:11434` by default.

![Whallm app](docs/assets/deepseekv4ssd-app.png)

## Requirements

| Item | Requirement |
| --- | --- |
| Mac | Apple Silicon M-series Mac |
| macOS | macOS 15 or later |
| Unified memory | 64 GiB or more |
| Free storage | The app checks the selected model and existing partial data |
| Model storage | A fast internal, Thunderbolt, or USB4 SSD |
| Internet | Required to download the model and app updates |

> [!IMPORTANT]
> Whallm is experimental. Model weights are not included with the app.
> Keep the default local server address unless another device must connect.

## Codex `config.toml`

Start the server in Whallm. Then add this configuration to
`~/.codex/config.toml`:

```toml
model = "deepseek-v4-flash-0731"
model_provider = "deepseek-v4-ssd"
model_reasoning_effort = "high"

[model_providers.deepseek-v4-ssd]
name = "Whallm"
base_url = "http://127.0.0.1:11434/v1"
wire_api = "responses"
requires_openai_auth = false
```

Restart Codex after you save the file. The local address does not need an API
key. The provider settings must be in the user-level config file. See the
[official Codex configuration reference](https://developers.openai.com/codex/config-reference/)
for more options.

## Other technical details

### How it works

- DeepSeek V4 Flash 0731 has 284B total parameters and about 13B active
  parameters per token.
- Common tensors stay in unified memory.
- Routed experts use checkpoint-native FP4 weights and stream from SSD when
  needed.
- DeepSeek V4.1 engram embedding rows are also fetched from SSD on demand.
- The runtime uses an FP8 KV cache and a bounded expert cache to control memory
  use.
- The installed model is verified against the pinned checkpoint revision.
- Server startup reads the installed model list but does not load model
  weights. The first generation request loads its selected model.
- The server keeps one model loaded. A request for another model closes the old
  runtime before it loads the new runtime.
- The Model page can load or unload a model. A loaded model moves to the
  **Loaded** section.
- The DeepSeek V4 Flash 0731 layer-major prefill threshold is configurable. Its default is
  1,024 uncached prompt tokens.

### Model storage and DSpark

- The main model uses about 145 GiB.
- DSpark adds about 10.12 GiB. Every new DeepSeek download includes it.
- Installing DSpark does not enable it. Enable **Use DSpark** in the runtime
  settings when you want to test speculative decoding.
- You can remove DSpark without reinstalling the main model.
- Qwen installed weight files use 125,268,506,112 bytes. Qwen does not support
  DSpark.
- Qwen downloads a verified MXFP4 installed model. Model installation does not
  quantize the Qwen checkpoint on the user's Mac.
- DeepSeek V4.1 installs the exact pinned Hugging Face text checkpoint. Its
  expert and engram files alone require at least 491,535,862,800 bytes
  (457.78 GiB), before common tensors and metadata. The app calculates the
  complete requirement from the repack plan.
- DeepSeek V4.1 does not install or enable vision, MTP/DSpark, or reusable
  prompt-cache state.

### OpenAI-compatible server

The server supports these endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /api/models/load`
- `POST /api/models/unload`

The fixed API model IDs are `deepseek-v4-flash-0731`,
`deepseek-v4.1-flash`, and `qwen3.8-flash-next-fp8`. The CLI also accepts
`deepseek-flash` as a DeepSeek V4.1 alias. Each model's **Advanced Settings** view lets you set an
optional Alias. Valid changes are saved automatically. Generation requests
accept the API model ID or its Alias. The chat model picker shows only the
installed models that were available when the server started. Restart the
server after a download finishes while it is running.

The Responses API supports Codex tools and OpenAI function tools. The client
must run each tool and send the result back to the server. Read the
[API guide](docs/API.md) for fields, examples, and current limits.

### Metrics and privacy

The app shows prefill speed, decode speed, token counts, memory use, SSD read
speed, cache hit rate, first-token wait time, and completion time. The app
clears metric history when the loaded model changes.

Inference runs on your Mac. Prompts and generated text stay in the local
runtime unless the connected client sends them elsewhere. The app uses the
network to download the model, check for updates, and accept configured API
requests.

### Current limits

- The runtime supports only the three pinned checkpoint revisions in the current
  documentation.
- DeepSeek V4.1 support is text-only. Vision, MTP/DSpark, layer-major prefill,
  persistent prompt cache, and prompt-cache reuse are disabled for this model.
- DeepSeek V4.1 contract, native FP8/MLX loading, SSD Engram lookup, cache,
  parser, full Python runtime suite, and core build validation pass in this
  worktree. A complete checkpoint install and full-model generation have not
  yet been run, so no memory or performance claim is made for V4.1.
- Qwen supports text only. Qwen vision, video, MTP, and DSpark are not supported.
- Qwen full-model SHA-256, text, thinking, tool call, greedy 4K, prompt cache,
  and packaged App validation passed on the recorded M5 Pro environment. See
  the [Qwen support status](docs/QWEN.md).
- The server keeps one model loaded and processes one generation request at a
  time. Other generation requests wait until the full request stream ends.
- Images, audio, logprobs, `response_format`, and `stop` are not supported.
- Request bodies are limited to 1 MiB.
- Very long input and output need more KV cache memory.
- Performance depends on SSD speed, input length, and cache state.

Read the [current documentation](docs/README.md) for the model contract,
runtime design, validation, performance, and research conclusions.
See [DeepSeek V4.1 support](docs/DEEPSEEK_V41.md) for its exact contract and
current validation boundary.

Whallm is not affiliated with DeepSeek. Review the model terms before
you download and use the model.

## License

The Whallm source code is available under the [MIT License](LICENSE).
Model weights are not included and remain subject to their own terms.
