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
