# DeepSeek V4.1 support

Whallm supports the text model from the exact pinned checkpoint below.

| Field | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Hugging Face model type | `deepseek_v41` |
| Whallm model kind | `deepseek-v4.1` |
| API model ID | `deepseek-v4.1-flash` |
| CLI selector aliases | `deepseek-v4.1`, `deepseek-v4.1-flash`, `deepseek-flash` |
| Manifest format | 3 |
| Text layers | 40 |
| Routed experts per layer | 384 |
| Selected experts per token | 6 |
| Shared experts per layer | 1 |
| Hidden size | 5,120 |
| Expert intermediate size | 2,304 |
| Checkpoint maximum context | 1,048,576 |

The runtime vendors an MLX architecture derived from the Apache-2.0
`PipeNetwork/deepseek-v41-mlx` port at commit
`8bdd543c7160800f3451d9595c996129b7abecdf`. Its license is preserved in
`runtime/deepseek_v4_ssd/deepseek_v41/THIRD_PARTY_LICENSE.txt`, and the local
adaptations are recorded beside the vendored package.

The pinned index contains 96,085 tensors: 92,160 routed-expert tensors, four
Engram table tensors, 2,401 MTP tensors, 266 vision/aligner tensors, and 1,254
text common tensors. The repacker rejects any unrecognized top-level tensor
family instead of silently copying it into `common.bin`.

## Installed model

The repacker rejects any model ID, revision, architecture shape, expert layout,
or Engram layout that differs from the pinned contract.

```text
deepseek-v4.1-flash.dsv4/
  manifest.json
  common.bin
  config.json
  encoding/encoding.py
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  experts/layer_00.bin
  ...
  experts/layer_39.bin
  engram/layer_01.weight.bin
  engram/layer_01.scale.bin
  engram/layer_14.weight.bin
  engram/layer_14.scale.bin
```

Each expert blob is 18,800,640 bytes. The 40 expert files contain
288,777,830,400 bytes (268.95 GiB). The two Engram tables contain
202,758,032,400 bytes (188.83 GiB). Those files require at least
491,535,862,800 bytes (457.78 GiB) before `common.bin`, tokenizer files, and
metadata. The app derives the complete required storage from the repack plan.

## Runtime path

- Common native FP8 tensors are loaded into MLX MXFP8 modules. The checkpoint's
  32-by-32 E8M0 scales are expanded to the row-scale layout MLX expects.
- Routed expert blobs remain checkpoint-native FP4 and use Whallm's bounded SSD
  expert cache.
- Engram FP8 rows and E8M0 scales remain memory-mapped on SSD. The runtime reads
  and dequantizes only the rows selected for the current tokens.
- The V4.1 cache grows on demand up to the pinned context ceiling instead of
  allocating the full context at startup.
- The V4.1 DSML markers include spaces in the tag names, so V4.1 uses a separate
  incremental tool-call parser.

## Current boundary

This integration is text-only. It excludes the checkpoint's vision tower and
aligner. It also disables MTP/DSpark, staged or adaptive expert prefill,
layer-major prefill, persistent prompt cache, and prompt-cache reuse. Each
generation starts with a fresh V4.1 cache.

The manifest contract, native FP8-to-MLX path, strict quantized loading, SSD
Engram lookup, cache ceiling, tool parser, a synthetic adapter load and real
forward pass, the full Python runtime suite, and Swift core targets have been
validated in this worktree. A complete checkpoint install and full-model
generation have not yet been run. No V4.1 memory, throughput, or quality claim
is made until that validation is complete.
