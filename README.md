# DeepSeekV4SSD

DeepSeekV4SSD is an experimental Apple Silicon runtime for
`DeepSeek-V4-Flash-0731`.

The runtime targets a 64 GiB M5 Pro. MLX runs the model on Metal. The runtime
keeps common tensors in memory and reads routed experts from SSD.

## What works

- The code validates the pinned official model contract.
- The code reads safetensors headers with HTTP Range requests.
- The code builds a deterministic repack plan for the 43-layer main model.
- The code excludes DSpark weights.
- The code copies selected byte ranges through an 8 MiB buffer.
- A receipt resumes an interrupted repack.
- The code writes SHA-256 values and verifies an installed model.
- MLX provides the DeepSeek-V4 model operations.
- A 1,024-slot LFU cache loads routed experts with four parallel `pread` calls.
- Chunked prefill limits the routed experts needed by one model call.
- MXFP8 stores the long compressed-attention cache. The 128-token local cache
  stays in BF16.
- The complete 145 GiB model is installed and its 49 files pass SHA-256
  verification.
- BF16 and MXFP8 pass greedy generation with the same output.
- BF16 passes a 4K context. BF16 and MXFP8 pass the same 8K context.

## Commands

Inspect official metadata without downloading tensor data:

```sh
swift run -c release dsv4-repack inspect
```

Write the repack plan:

```sh
swift run -c release dsv4-repack plan --output plan.json
```

Run the complete main-model repack:

```sh
swift run -c release dsv4-repack repack \
  --output deepseek-v4-flash-0731.dsv4 \
  --plan plan.json
```

The complete repack writes about 145 GiB. The command writes to a sibling
`.partial` directory first. Run the same command again after an interruption.
The command checks the receipt and continues completed data. The command never
overwrites an existing destination.

Verify an installed model:

```sh
swift run -c release dsv4-repack verify \
  --model deepseek-v4-flash-0731.dsv4
```

Measure expert SSD reads:

```sh
swift run -c release dsv4-repack benchmark \
  --model deepseek-v4-flash-0731.dsv4 \
  --samples 32
```

## Run the model

Create the local Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run greedy generation with the M5 Pro defaults:

```sh
PYTHONPATH=runtime .venv/bin/python -m deepseek_v4_ssd.cli \
  --model deepseek-v4-flash-0731.dsv4 \
  --prompt "Explain why the sky is blue." \
  --max-tokens 32
```

The M5 Pro defaults use 1,024 expert slots, four SSD read workers, 128-token
prefill chunks, and an MXFP8 compressed-attention cache. Use
`--bf16-kv-cache` for the correctness baseline.

Each expert slot stores one packed expert blob. A cache miss creates one Metal
array instead of six arrays. The cache uses layer-aware LFU heaps with frequency
aging. MXFP8 index scoring and sparse pooled attention operate on cache chunks
without rebuilding the complete index cache. The API status and CLI metrics
include time to first token, decode speed, cache-state evaluation, packing,
eviction, and routing synchronization times. The runtime evaluates cache state
after each output token. This limits Metal resource growth during long output.
The runtime also reuses one in-memory prompt prefix for a continued chat.
Single-token decode runs the six routed experts directly. Multi-token prefill
groups token routes by expert. Neither path builds stacked weight buffers.

Measured direct SSD throughput reached 14.30 GiB/s with four read workers.
After prefill grouping, a 4,096-token BF16 prefill and one output token took
30.74 seconds. An 8,192-token MXFP8 prefill and one output token took 65.47
seconds. See the validation record for the full measurements and their limits.

## Run the macOS app

Start the SwiftUI control app:

```sh
make run
```

The app detects installed models in the selected model folder. The app can
download and repack the pinned model directly from Hugging Face. The complete
installed model needs about 145 GiB of storage. An interrupted download resumes
from its validated partial data.

Before downloading, the app checks Apple Silicon, memory, folder permissions,
available storage, and the target SSD. The download view shows progress, speed,
and estimated time. The user can stop a download and continue it later. The app
can run a complete SHA-256 verification and redownload only missing or damaged
model data. The default model folder is `~/.dsmodel/`.

The app configures and controls the OpenAI-compatible API server and the
DeepSeekV4SSD runtime. It also provides a small test chat. The server provides
`/v1/models`, `/v1/responses`, `/v1/chat/completions`, and `/v1/completions`.
It supports normal JSON responses, SSE streaming, and OpenAI function Tool
calls. The test chat can expose `get_current_time` and display the model's Tool
call while it streams. The App does not execute the Tool.

Build a distributable macOS app:

```sh
make package
```

The package command embeds the Python framework, the current `.venv` packages,
and the runtime. The output is `dist/DeepSeekV4SSD-macOS-arm64.zip`. Set
`CODE_SIGN_IDENTITY` to a Developer ID Application identity for distribution.
Without this value, the command uses an ad hoc signature for local testing.
Set `NOTARY_PROFILE` to a `notarytool` keychain profile to submit the signed App
to Apple and staple the accepted ticket.

The app uses Sparkle to check GitHub Releases for updates. Publish a signed and
notarized update with a version that is higher than the previous release:

```sh
CODE_SIGN_IDENTITY="Developer ID Application: Yanun Yang (Y366CJ66L6)" \
NOTARY_PROFILE=DeepSeekV4SSD \
make release VERSION=1.0.0
```

This command uploads the app ZIP and signed `appcast.xml` to the matching GitHub
Release. The Sparkle private key stays in the local macOS Keychain under the
`deepseek_ssd` account.

Use `make server` to start the server without the app. Set
`DEEPSEEK_API_KEY=local-key` when a Bearer API key is needed.

The current runtime processes one generation request at a time because the
model supports batch size 1. Later requests wait in the operating-system
connection queue. See [the API guide](docs/API.md) for examples, supported
parameters, security rules, and current limits.

Run tests:

```sh
make test
PYTHONPATH=runtime .venv/bin/python -m unittest discover -s runtime/tests -v
```

See [the implementation plan](docs/IMPLEMENTATION_PLAN.md),
[the validation record](docs/VALIDATION.md),
[the API guide](docs/API.md), and
[the feasibility report](research/deepseek-v4-flash-0731-turbofieldfare-feasibility.md).
