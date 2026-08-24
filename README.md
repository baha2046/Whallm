

# DeepSeekV4SSD

<p align="center">
  <img src="Packaging/AppIcon.png" alt="DeepSeekV4SSD App Icon" width="160">
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/English-Click-yellow" alt="English"></a>
  <a href="README-tw.md"><img src="https://img.shields.io/badge/繁體中文-點擊查看-orange" alt="繁體中文"></a>
  <a href="README-cn.md"><img src="https://img.shields.io/badge/简体中文-点击查看-orange" alt="简体中文"></a>
  <a href="README-ja.md"><img src="https://img.shields.io/badge/日本語-クリック-青" alt="日本語"></a>
  <a href="README-ko.md"><img src="https://img.shields.io/badge/한국어-클릭-yellow" alt="한국어"></a>
</p>

Inspired by [Turbo Fieldfare](https://github.com/drumih/turbo-fieldfare),
DeepSeekV4SSD streams routed experts from SSD to run all 284 billion parameters
of `DeepSeek-V4-Flash-0731` on an M-series Mac with about 30 GB of memory.

## Benchmark

These results were measured on a MacBook Pro with an Apple M5 Pro, 18 CPU
cores, 20 GPU cores, and 64 GiB of unified memory. DSpark was disabled.

| Test | Prefill | Decode | Peak memory |
| --- | ---: | ---: | ---: |
| Codex request with 14,000 input tokens | 180 Tok/s | 6.5 Tok/s | 30 GB |
| 4,096-token prompt with one output token | 144.53 Tok/s | — | 15.56 GiB |
| Short prompt, second run in one runtime | — | 6.41 Tok/s | 15.05 GiB |

The first two rows were measured with runtime versions `v1.0.3` and `v1.0.2`,
respectively. Performance changes with the prompt, SSD speed, and cache state.
See the [validation record](docs/VALIDATION.md) for the full test details.

## How to use it

**Download the app → Open the app → Download the 167 GB full model → Start the
server → Chat in the app or connect Codex**

> [!IMPORTANT]
> Version 1.0.3 cannot install version 1.0.4 through automatic update because
> the previous Sparkle signing key is no longer available. Quit the app,
> download `DeepSeekV4SSD-macOS-arm64.zip` from the
> [1.0.4 release](https://github.com/yanun0323/deepseek_ssd/releases/tag/v1.0.4),
> and replace the existing app manually. Automatic updates work again after
> you install version 1.0.4.

1. Download the latest `DeepSeekV4SSD-macOS-arm64.zip` from
   [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest).
2. Extract the ZIP and open `DeepSeekV4SSD.app`.
3. Select **Download Model**. The default installation includes DSpark and uses
   about 167 GB. You can stop the download and resume it later.
4. Select **Start Server** after the model is ready.
5. Use the chat in the app, or connect Codex with the configuration below.

The local server starts at `http://127.0.0.1:11434` by default.

![DeepSeekV4SSD app](docs/assets/deepseekv4ssd-app.png)

## Requirements

| Item | Requirement |
| --- | --- |
| Mac | Apple Silicon M-series Mac |
| macOS | macOS 15 or later |
| Unified memory | 64 GiB or more |
| Free storage | About 172 GB (160 GiB) |
| Model storage | A fast internal, Thunderbolt, or USB4 SSD |
| Internet | Required to download the model and app updates |

> [!IMPORTANT]
> DeepSeekV4SSD is experimental. Model weights are not included with the app.
> Keep the default local server address unless another device must connect.

## Codex `config.toml`

Start the server in DeepSeekV4SSD. Then add this configuration to
`~/.codex/config.toml`:

```toml
model = "deepseek-v4-flash-0731"
model_provider = "deepseek-v4-ssd"
model_reasoning_effort = "high"

[model_providers.deepseek-v4-ssd]
name = "DeepSeekV4SSD"
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

### Model storage and DSpark

- The main model uses about 145 GiB.
- DSpark adds about 10.12 GiB and is included in the default download.
- Installing DSpark does not enable it. Enable **Use DSpark** in the runtime
  settings when you want to test speculative decoding.
- You can remove DSpark without reinstalling the main model.

### OpenAI-compatible server

The server supports these endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

The Responses API supports Codex tools and OpenAI function tools. The client
must run each tool and send the result back to the server. Read the
[API guide](docs/API.md) for fields, examples, and current limits.

### Metrics and privacy

The app shows prefill speed, decode speed, token counts, memory use, SSD read
speed, cache hit rate, first-token wait time, and completion time.

Inference runs on your Mac. Prompts and generated text stay in the local
runtime unless the connected client sends them elsewhere. The app uses the
network to download the model, check for updates, and accept configured API
requests.

### Current limits

- The runtime supports only the pinned `DeepSeek-V4-Flash-0731` checkpoint.
- The runtime processes one generation request at a time.
- Images, audio, logprobs, `response_format`, and `stop` are not supported.
- Request bodies are limited to 1 MiB.
- Very long input and output need more KV cache memory.
- Performance depends on SSD speed, input length, and cache state.

Read the [current documentation](docs/README.md) for the model contract,
runtime design, validation, performance, and research conclusions.

DeepSeekV4SSD is not affiliated with DeepSeek. Review the model terms before
you download and use the model.

## License

The DeepSeekV4SSD source code is available under the [MIT License](LICENSE).
Model weights are not included and remain subject to their own terms.
