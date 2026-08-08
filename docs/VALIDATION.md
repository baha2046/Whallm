# Validation record

## Target hardware

- MacBook Pro `Mac17,8`
- Apple M5 Pro
- 18 CPU cores
- 20 GPU cores
- 64 GiB unified memory
- Metal 4
- MLX recommended working set: 51.8 GiB

## Checkpoint contract

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Main-model layers: 43
- Routed experts per layer: 256
- Selected experts per token: 6
- Main-model installed weight bytes: 156,015,738,880
- Common tensor bytes: 8.24 GiB
- Common tensor count: 1,564

The sanitized common tensor names match all 1,564 MLX model parameters. The
shape check has no mismatch.

## Automated tests

- Fifteen Swift tests pass.
- Twenty-eight Python runtime tests pass.
- Tool codec tests cover official encoder loading and Tool call parsing.
- Server tests cover Tool results, incremental SSE output, single-character DSML
  fragments, and invalid Tool output.
- MXFP4 routed expert output matches a dequantized reference.
- Ratio-4 compression gives the same result across prefill chunk boundaries.
- Sparse top-k selection resolves equal scores by position.
- MXFP8 completed cache chunks use less memory than BF16.

## Complete installation

- Installed files: 49
- Installed model weights: 156,015,738,880 bytes
- Transfer receipt chunks: 18,615
- Independent size and SHA-256 verification: pass
- Resume validation RSS after the autorelease fix: about 140 MiB

## SSD measurements

| Mode | Workers | GiB/s | Mean read time |
| --- | ---: | ---: | ---: |
| Direct | 1 | 10.86 | 1.15 ms |
| Direct | 2 | 14.04 | 1.75 ms |
| Direct | 4 | 14.30 | 3.30 ms |
| Direct | 8 | 13.25 | 6.71 ms |
| Cached | 1 | 25.27 | 0.49 ms |
| Cached | 2 | 44.02 | 0.56 ms |
| Cached | 4 | 67.36 | 0.73 ms |
| Cached | 8 | 79.26 | 1.17 ms |

Each row uses 32 routed-expert reads. Four workers give the best direct SSD
throughput. The runtime uses four workers by default.

## End-to-end measurements

Short prompt: `The capital of France is`

- BF16 output: ` Paris. The capital of Spain is Madrid`
- MXFP8 output: ` Paris. The capital of Spain is Madrid`
- BF16 first-run speed: 1.48 token/s
- MXFP8 second-run speed: 2.83 token/s
- Peak memory: about 21.2 GiB

The second run used a warm operating-system page cache. The speed difference
does not measure the MXFP8 effect.

| Context | KV cache | Time | Greedy output | Cache hit rate | Expert bytes read | Peak memory |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 4,096 | BF16 | 124.0 s | ` test` | 60.5% | 572.9 GB | 21.9 GiB |
| 8,192 | MXFP8 | 264.1 s | ` test` | 57.2% | 1,340.2 GB | 22.0 GiB |
| 8,192 | BF16 | 264.6 s | ` test` | 54.5% | 1,377.7 GB | 21.9 GiB |

All tests use batch size 1, greedy decoding, and a 32-token prefill chunk.
The whole-model peak does not show an MXFP8 reduction at 8K. Model weights and
temporary expert stacks dominate this peak. The cache unit test confirms that
completed MXFP8 cache chunks use less storage than BF16 chunks.

## Optimized prefill measurements

These measurements use the 128-token prefill default and grouped routed expert
execution. The older table remains as the fixed historical baseline.

| Context | KV cache | Time | Greedy output | Cache hit rate | Expert bytes read | Peak memory |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 4,096 | BF16 | 30.74 s | ` test` | 59.0% | 219.1 GB | 22.2 GiB |
| 8,192 | MXFP8 | 65.47 s | ` test` | 58.7% | 488.8 GB | 22.2 GiB |

The 4K and 8K greedy outputs match the historical baseline. The improvement
includes the 128-token prefill step and grouped routed expert execution.
The 8K result also includes packed-row MXFP8 gather and one packed matrix
multiplication for completed index-cache chunks. The runtime reuses the packed
cache until new completed chunks invalidate it.
The expert cache uses one global LFU heap while it keeps the per-layer reserve.

## DSpark decision

DSpark stays disabled for M4. It adds about 10.12 GiB of weights. The measured
main-model baseline spends most time in model and expert work. SSD reads use
about 14.0 seconds of the 65.5-second 8K run. There is no measured evidence that
DSpark gives a net speed increase in this runtime.

## Layer-major prefill measurements

These measurements use the same installed model and a deterministic repeated
token prompt. Each row generates one greedy token. The 4K and 8K outputs were
` test` in both prefill modes.

| Context | Mode | Step | Time | Expert bytes read |
| ---: | --- | ---: | ---: | ---: |
| 321 | Chunk-major | 128 | 7.23 s | 52.4 GB |
| 321 | Layer-major | 128 | 7.44 s | 38.7 GB |
| 4,096 | Layer-major | 128 | 29.30 s | 47.3 GB |
| 4,096 | Layer-major | 256 | 22.49 s | 47.0 GB |
| 4,096 | Chunk-major | 512 | 21.18 s | 101.5 GB |
| 4,096 | Layer-major | 512 | 20.21 s | 46.7 GB |
| 8,192 | Chunk-major | 512 | 47.04 s | 251.9 GB |
| 8,192 | Layer-major | 512 | 38.20 s | 60.3 GB |
| 14,363 | Chunk-major | 512 | 386.65 s | 2,290.6 GB |
| 14,363 | Layer-major | 512 | 105.81 s | 131.5 GB |

Layer-major prefill is slower for the 321-token test. The runtime therefore
uses it only when at least 4,096 prompt tokens are not already cached. Automatic
step selection uses 128 below 1K, 256 below 4K, and 512 from 4K onward.
The 14K prompt contains varied Tool-like names, parameters, and descriptions.
Layer-major prefill was 3.65 times faster and reduced expert blob reads by about
94.3 percent. It is not the original Codex request payload.

A same-process 4K continuation reused 4,097 of 4,098 prompt tokens. Time to
first token decreased from 20.89 seconds to 0.56 seconds. The current runtime
also saves completed prompt cache entries for reuse after a server restart.

## Batched layer-local MoE measurements

These measurements use a varied Tool-like prompt. Each row generates one
greedy token. The optimized path uses a 1,024-token attention step, a
4,096-token MoE tile, two full-layer prefetch workers, strided routed expert
views, and batched `gather_qmm`.

| Context | Path | Time | Expert read time | Expert bytes | `gather_qmm` calls |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4,097 | Grouped fallback | 44.03 s | 10.12 s | 128.64 GB | 0 |
| 4,097 | Batched layer-local | 28.02 s | 11.74 s | 150.42 GB | 126 |
| 14,363 | Previous layer-major | 105.81 s | 11.48 s | 131.50 GB | — |
| 14,363 | Batched layer-local | 76.54 s | 11.85 s | 150.41 GB | 504 |

The 4,097-token output matched the grouped fallback. The new path improved
time to first token by 36.4%. The 14,363-token path improved time to first
token by 27.7%. It reads every routed expert in 42 layers. The last layer does
not run MoE during cache-only prefill.

Four full-layer prefetch workers reduced measured SSD time to 11.30 seconds,
but total time increased to 29.47 seconds. Two prefetch workers gave the lowest
4,097-token time. FP4 and MXFP8 index paths produced the same 14,363-token
greedy token. FP4 took 76.54 seconds. MXFP8 took 77.34 seconds.

Persistent prompt cache now stores up to eight entries under
`~/.dsmodel/prompt-cache/`. The restart test restores the complete cache state
and reuses the matching token prefix.

## Ready expert decode measurements

Ready expert decode submits all missing routed expert reads first. The runtime
starts compute for each resident or completed routed expert without waiting for
the slowest read. The runtime keeps the native router output order.

Five paired tests used 4,096 prompt tokens, 256 output tokens, 512 slots,
greedy decoding, and disabled DSpark. The prompt types were repeated text,
code, Traditional Chinese technical text, English prose, and mixed math. The
run order alternated between baseline-first and ready-first.

| Result | Measurement |
| --- | ---: |
| Median decode throughput improvement | 12.9% |
| Minimum improvement | 8.1% |
| Maximum improvement | 14.0% |
| Aggregate decode-time reduction | 10.5% |
| Matching 256-token hashes | 5 of 5 |

A 2,000-token repeated-text stability pair measured 10.67 Tok/s for the
baseline and 11.92 Tok/s for ready expert decode. This is an 11.7% throughput
improvement. Both runs generated all 2,000 tokens and produced the same token
hash. The ready run did not report a Metal-resource or slot-lifetime error.

Ready expert decode is now the default. CLI and server users can pass
`--no-ready-expert-decode` to run the old baseline. These tests are paired
measurements, not a throughput guarantee.

## Partial-data smoke tests

These smoke tests used the sparse partial install. They validate execution
shape and memory only. They do not validate model output or final speed.

One-token forward:

- Output shape: `1 × 1 × 129280`
- Finite output: yes
- Active common tensor memory: 9.0 GB
- Routed expert bytes read: 3,449,290,752
- Forward time: 1.73 seconds

Eight-token cached decode:

- Finite output: yes
- Cache hits: 1,764
- Cache misses: 300
- Resident slots: 300
- Routed expert bytes read: 4,010,803,200
- Peak memory: 13.12 GB
- Total forward time: 2.04 seconds

## Final status

All M1 through M4 checks pass on the target M5 Pro. DSpark is the only optional
item, and the measured baseline does not justify adding it.
