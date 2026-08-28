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

> [!NOTE]
> Whallm was previously named DeepSeekV4SSD. Releases published before the
> rename, including version 1.0.4, use the old app and archive names.

Inspired by [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare),
Whallm lets an M-series Mac run all 284B parameters of the pinned
`DeepSeek-V4-Flash-0731` checkpoint by streaming routed experts from SSD. It
also supports the pinned `Qwen3.8-Flash-Next-FP8` text checkpoint.

## Memory guidance

| Model | Recorded peak memory |
| --- | ---: |
| `DeepSeek-V4-Flash-0731` | 23.03–35.64 GiB |
| `Qwen3.8-Flash-Next-FP8` | 15.19–18.92 GiB |

These v1.1.0 results cover chat prompts with 1,024 to 16,384 input tokens. They
are measurements, not minimum memory requirements or performance guarantees.
Prompt length, tools, cache state, and runtime settings can change peak memory.
See the [benchmark](BENCHMARK.md) and
[validation record](docs/VALIDATION.md) for the measured workloads.

## Benchmark

These v1.1.0 results were measured on a MacBook Pro with an Apple M5 Pro,
64 GB of unified memory, and 1 TB of storage. Both models used
`reasoning_effort: low` and `thinking_mode: chat`. TTFT means time to first
token.

### DeepSeek V4 Flash 0731

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 46.72 s | 37.26 s | 20.7 tok/s | 6.7 tok/s | 23.03 GiB |
| 2,048 | 27.08 s | 16.91 s | 108.1 tok/s | 6.6 tok/s | 33.19 GiB |
| 8,192 | 47.66 s | 37.96 s | 209.5 tok/s | 6.7 tok/s | 34.74 GiB |
| 16,384 | 86.01 s | 76.34 s | 212.3 tok/s | 6.7 tok/s | 35.64 GiB |

### Qwen3.8 Next Flash FP8

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 24.45 s | 17.20 s | 59.8 tok/s | 9.3 tok/s | 15.19 GiB |
| 2,048 | 40.45 s | 32.63 s | 65.0 tok/s | 8.3 tok/s | 16.41 GiB |
| 8,192 | 142.39 s | 134.44 s | 61.7 tok/s | 8.3 tok/s | 17.90 GiB |
| 16,384 | 276.41 s | 267.88 s | 61.8 tok/s | 7.8 tok/s | 18.92 GiB |

Performance changes with the prompt, SSD speed, and cache state. See the
[full benchmark](BENCHMARK.md) and [validation record](docs/VALIDATION.md) for
more details.

## How to use it

**Download the app → Open the app → Select and download a model → Start the
server → Chat in the app or connect Codex**

> [!IMPORTANT]
> Version 1.0.3 cannot install version 1.0.4 through automatic update because
> the previous Sparkle signing key is no longer available. Quit the app,
> download `DeepSeekV4SSD-macOS-arm64.zip` from the
> [1.0.4 release](https://github.com/yanun0323/deepseek_ssd/releases/tag/v1.0.4),
> and replace the existing app manually. Automatic updates work again after
> you install version 1.0.4.

1. Download the latest `Whallm-macOS-arm64.zip` from
   [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest).
2. Extract the ZIP and open `Whallm.app`.
3. Open the **Model** page. Select DeepSeek or Qwen, and then select
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

- The main model has 284B total parameters and about 13B active parameters per
  token.
- Common tensors stay in unified memory.
- Routed experts use checkpoint-native FP4 weights and stream from SSD when
  needed.
- The runtime uses an FP8 KV cache and a bounded expert cache to control memory
  use.
- The installed model is verified against the pinned checkpoint revision.
- Server startup reads the installed model list but does not load model
  weights. The first generation request loads its selected model.
- The server keeps one model loaded. A request for another model closes the old
  runtime before it loads the new runtime.
- The Model page can load or unload a model. A loaded model moves to the
  **Loaded** section.
- The DeepSeek layer-major prefill threshold is configurable. Its default is
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

### OpenAI-compatible server

The server supports these endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /api/models/load`
- `POST /api/models/unload`

The fixed API model IDs are `deepseek-v4-flash-0731` and
`qwen3.8-flash-next-fp8`. Each model's **Advanced Settings** view lets you set an
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

- The runtime supports only the two pinned checkpoint revisions in the current
  documentation.
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

Whallm is not affiliated with DeepSeek. Review the model terms before
you download and use the model.

## License

The Whallm source code is available under the [MIT License](LICENSE).
Model weights are not included and remain subject to their own terms.
