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
metadata. The app uses a fixed display size of 501,382,643,728 bytes (501.38 GB),
verified against the repack plan for revision `dba1be0a40aa45a94ad051997016db3960a90277`
on 2026-09-13. This is the planned installed weight size, matching the existing
UI size calculation; it excludes companion metadata and is not measured network traffic.
The model list no longer fetches a plan just to display this size.

For CLI installation, use `dsv4-repack repack --model deepseek-v4.1 --output MODEL.dsv4`.
When `--plan plan.json` is supplied, the plan's `modelKind` selects the repacker;
legacy plans without that field continue to select V4.

## Runtime path

Advanced Settings defaults to 1152 slots. The recommendation text in English,
Simplified Chinese, and Traditional Chinese uses the same model descriptor value.
Existing saved slot settings are preserved.

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
and layer-major prefill. Prompt-cache reuse and persistence are now implemented
and included in the 2026-09-14 local build, after the published `v1.1.7-dev.2` Alpha.

## Prompt cache

Advanced Settings offers Off / Memory / Disk. New settings default to Memory;
explicitly saved choices are preserved. Memory reuses a matching prefix until
unload; Disk also restores it after restarting the runtime. Off starts fresh.
The CLI and server use the same existing `--prompt-cache` controls.

The model support package captures the global position, buffer capacity,
per-layer window KV, compressed KV, index keys, unfinished compressor KV/scores,
and Engram compressed-token history. Restored branches own independent state.
Shared attention references are rebuilt by each forward call. A cache cannot
be trimmed backward; the runtime reuses only complete matching saved prefixes.
The existing disk contract binds the cache to the model revision and configuration;
the V4.1 state additionally validates its schema version, positions, shapes, and dtypes.
Cache memory accounting includes the CPU Engram history.

Four tests run a small four-layer V4.1 model with nonzero Engram weights, shared
compressed attention, and compression ratios 1 and 2. They cover a partial group,
window wrap, capacity growth, independent branches, invalid-state rejection,
Off / Memory / Disk runtime behavior, disk reload, and interrupted generation.
Continuation logits after cloning and serialization match exactly; cached runtime
output tokens match cold generation. Negative controls that erase Engram history
or partial compressor state change the outputs, confirming that the tests exercise
those states. These are functional checks, not full-checkpoint performance results.

```sh
PYTHONPATH=runtime .venv/bin/python -m unittest discover -s runtime/tests -p test_v41_prompt_cache.py
```

The manifest contract, native FP8-to-MLX path, strict quantized loading, SSD
Engram lookup, cache ceiling, tool parser, a synthetic adapter load and real
forward pass, the full Python runtime suite, and Swift core targets have been
validated in this worktree. A complete checkpoint install and full-model
generation have not yet been run. No V4.1 memory, throughput, or quality claim
is made until that validation is complete.
