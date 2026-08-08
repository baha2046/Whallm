<div align="right">
  <strong>English</strong> · <a href="README_TW.md">繁體中文</a>
</div>

# DeepSeekV4SSD

![DeepSeekV4SSD app](docs/assets/deepseekv4ssd-app.png)

Run `DeepSeek-V4-Flash-0731` locally on Apple Silicon. DeepSeekV4SSD keeps the
common tensors in unified memory and reads routed experts from a high-speed SSD.

> [!IMPORTANT]
> DeepSeekV4SSD is experimental. It needs a Mac with at least 64 GiB of unified
> memory and about 160 GiB of free SSD storage. Model weights are not included
> with the app.

## Features

- Download, install, verify, and repair the supported model in the app.
- Resume an interrupted model download.
- Use the default model folder at `~/.dsmodel/` or select another folder.
- Run an OpenAI-compatible API server on your Mac.
- Use `/v1/responses`, `/v1/chat/completions`, and `/v1/completions`.
- Stream text, reasoning content, and function Tool calls.
- Monitor prefill speed, decode speed, token counts, memory, SSD reads, cache
  hit rate, first-token wait time, and completion time.
- View live, minimum, average, and maximum metrics.
- Use English, Simplified Chinese, or Traditional Chinese in the app.
- Receive signed app updates through GitHub Releases and Sparkle.

## Requirements

| Item | Requirement |
| --- | --- |
| Mac | Apple Silicon |
| macOS | macOS 15 or later |
| Unified memory | 64 GiB or more |
| Storage | About 160 GiB of free space |
| Model storage | A high-speed internal or external SSD |
| Internet | Required for model downloads and update checks |

SSD speed has a direct effect on token generation speed. Use a fast Thunderbolt
or USB4 SSD when you store the model on an external disk.

## Install the app

1. Download the latest signed and notarized ZIP from
   [GitHub Releases](https://github.com/yanun0323/deepseek_ssd/releases/latest).
2. Extract `DeepSeekV4SSD-macOS-arm64.zip`.
3. Move `DeepSeekV4SSD.app` to the Applications folder.
4. Open the app.

The app checks GitHub Releases for updates. You can also select
**Check for Updates…** from the app menu.

## Install the model

1. Open DeepSeekV4SSD.
2. Keep the default model folder at `~/.dsmodel/`, or select another folder.
3. Decide if you want to install DSpark with the model.
4. Select **Download Model**.
5. Wait for the download and installation to finish.

The main model uses about 145 GiB. DSpark adds about 10.12 GiB. You can stop the
download and resume it later. The app keeps verified partial data.

To use an existing installed model, select the folder that contains the
installed model or its parent folder. The app detects valid models
automatically.

## Start the local server

1. Select an installed model.
2. Review the server and runtime settings.
3. Select **Start Server**.
4. Use the test chat, or connect another client.

The default server address is:

```text
http://127.0.0.1:11434
```

The default OpenAI base URL is:

```text
http://127.0.0.1:11434/v1
```

A local server does not require an API key. A non-local host requires an API
key. Do not expose the server directly to the public Internet.

## Call the API

### Chat Completions

```sh
curl -N http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "deepseek-v4-flash-0731",
    "messages": [
      {"role": "user", "content": "Explain why the sky is blue."}
    ],
    "stream": true,
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.98
  }'
```

### Responses API

```sh
curl -N http://127.0.0.1:11434/v1/responses \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "deepseek-v4-flash-0731",
    "instructions": "Answer briefly.",
    "input": "Explain why the sky is blue.",
    "stream": true,
    "max_output_tokens": 256
  }'
```

Supported endpoints include:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/chat/completions`
- `POST /v1/completions`

The server supports OpenAI function tools. Your client must run each function
and send the result back to the server. The server does not run tools or
external commands.

Read the [API guide](docs/API.md) for request fields, Python examples, Codex
tool support, and current limits.

## DSpark

DSpark is an optional speculative decoding module. Installing DSpark adds about
10.12 GiB to the model folder. Installing DSpark does not enable it.

Enable **Use DSpark** in the runtime settings when you want to test it. The
default DSpark cache holds 256 routed experts in independent SSD-backed slots.
The runtime can stop using DSpark for the rest of a request when normal decode
is faster.

DSpark performance depends on the prompt, SSD, and cache state. DSpark is not
faster for every request. You can disable DSpark without removing its files.
Select **Remove DSpark** when you want to recover its storage space.

## Metrics

The app shows metrics for test-chat and API requests:

- Prefill Tok/s
- Decode Tok/s
- Input Tokens
- Output Tokens
- Memory usage
- SSD read speed
- Cache Hit rate
- First Token wait time
- Completion time

Each metric shows its live, minimum, average, and maximum value. The history
uses one-second samples across all requests. Select **Clear metric history** to
reset the minimum, average, and maximum values.

## Privacy and network access

Inference runs on your Mac. Prompts and generated text stay in the local
runtime unless the client that calls the API sends them elsewhere.

The app uses the network for these tasks:

- Download the model from Hugging Face.
- Check and download app updates from GitHub Releases.
- Accept API requests on the host and port that you configure.

Keep the default `127.0.0.1` host unless another device must connect. Set a
strong API key before you use a non-local host.

## Troubleshooting

### The model download stopped

Open the same model folder and select **Resume Download**. The app reuses the
verified partial data.

### The model is missing or damaged

Select **Verify Complete Model**. If the app reports damaged files, select
**Verify and Repair**. The app downloads only missing or damaged data again.

### The first response is slow

A cold expert cache and a long input increase first-token wait time. Later
requests can be faster after the cache is warm. Check Prefill Tok/s, SSD read
speed, and Cache Hit rate in the metrics panel.

### Memory use is too high

Stop the server before you change runtime settings. Reduce **Prompt cache GiB**,
reduce **Prompt cache entries**, disable DSpark, or reduce **DSpark slots**.
DSpark needs at least 30 slots.

### API requests wait for a long time

The current runtime processes one generation request at a time. A later request
waits until the active request finishes. Long input also increases prefill time.

### The server does not start

Verify the selected model. Check that the configured port is available. Change
the port when another app already uses `11434`.

## Current limits

- Only `DeepSeek-V4-Flash-0731` at the pinned revision is supported.
- The runtime processes one generation request at a time.
- Images, audio, logprobs, `response_format`, and `stop` are not supported.
- The maximum requested output is 272,000 tokens.
- Very long output needs more KV cache memory.
- Performance depends on SSD speed, input length, and cache state.

DeepSeekV4SSD is not affiliated with DeepSeek. Review the model terms before you
download and use the model.
