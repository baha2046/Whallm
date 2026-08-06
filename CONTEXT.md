# Project context

This project builds a memory-bounded Apple Silicon runtime for `DeepSeek-V4-Flash-0731`.

## Ubiquitous language

- **checkpoint**: The pinned Hugging Face model revision and its safetensors shards.
- **common tensor**: A tensor that the runtime keeps resident. It is not a routed expert tensor.
- **routed expert**: One `w1`, `w2`, and `w3` expert with its scales.
- **expert blob**: The canonical packed bytes for one routed expert.
- **repack plan**: The complete mapping from checkpoint byte ranges to installed model byte ranges.
- **installed model**: A verified local directory produced from a repack plan.
- **manifest**: The JSON file that defines an installed model and its integrity metadata.
- **slot**: A fixed Metal-visible memory region that can hold one expert blob.
- **main model**: The 43 target-model layers. It excludes DSpark.
- **DSpark**: The optional speculative decoding module stored under `mtp.*`.

## Current scope

The project must complete M1 through M4 in order.
M2 first supports batch size 1, greedy decode, and a 4K context.
M3 adds measured SSD expert streaming.
M4 adds chunked prefill, FP8 KV cache, validated longer contexts, and a measured DSpark decision.

The checkpoint revision is fixed. Code must reject incompatible model shapes and tensor layouts.
