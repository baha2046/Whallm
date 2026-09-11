# Project context

This project builds a memory-bounded Apple Silicon runtime for
`DeepSeek-V4-Flash-0731`, `DeepSeek-V4.1-Flash`, and
`Qwen3.8-Flash-Next`.

## Ubiquitous language

- **checkpoint**: The pinned Hugging Face model revision and its safetensors shards.
- **common tensor**: A tensor that the runtime keeps resident. It is not a routed expert tensor.
- **routed expert**: One model-specific expert. DeepSeek uses `w1`, `w2`, and `w3`.
  Qwen uses fused `gate_up` and `down`.
- **expert blob**: The canonical packed bytes for one routed expert.
- **repack plan**: The complete mapping from checkpoint byte ranges to installed model byte ranges.
- **installed model**: A verified local directory produced from a repack plan.
- **API model ID**: The fixed, case-sensitive API name for one model kind.
- **Alias**: An optional, case-sensitive request name for an API model ID. An Alias does not
  change the API model ID.
- **manifest**: The JSON file that defines an installed model and its integrity metadata.
- **slot**: A fixed Metal-visible memory region that can hold one expert blob.
- **main model**: The 43 target-model layers. It excludes DSpark.
- **DSpark**: The optional speculative decoding module stored under `mtp.*`.
- **N-gram store**: Qwen FP8 N-gram rows stored in `ngram.bin` for read-only row lookup.
- **Engram store**: DeepSeek V4.1 FP8 embedding rows and E8M0 scales stored in
  layer-specific files for read-only row lookup.
- **model kind**: The runtime family selected by manifest format and `modelKind`.

## Current scope

The project must complete M1 through M4 in order.
M2 first supports batch size 1, greedy decode, and a 4K context.
M3 adds measured SSD expert streaming.
M4 adds chunked prefill, FP8 KV cache, validated longer contexts, and a measured DSpark decision.

Each checkpoint revision is fixed. Code must reject incompatible model shapes and tensor layouts.
DeepSeek V4.1 support is text-only and uses the exact `deepseek_v41` MLX architecture.
Its vision tower, MTP/DSpark, reusable prompt cache, and layer-major prefill are outside
the current support boundary.
