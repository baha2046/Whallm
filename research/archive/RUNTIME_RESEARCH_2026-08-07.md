# DeepSeek-V4-Flash-0731 runtime research

> [!WARNING]
> Historical snapshot from 2026-08-07. This file does not describe the current
> runtime. Use [the current research conclusions](../../docs/RESEARCH.md) and
> [validation record](../../docs/VALIDATION.md).

Checked: 2026-08-07

## Implementation progress

The first low-risk changes from this report are now active:

- The runtime records time to first token and decode time separately.
- The runtime records decode tokens per second.
- The runtime records the time and count for cache-state evaluation.
- The runtime evaluates the generation cache after each output token. This
  limits the Metal resource growth reported by MLX-LM issue 1332.
- The CLI and `GET /api/status` return the new measurements.
- The runtime keeps up to two in-memory prompt cache timelines. It moves a
  matching timeline forward when the next request continues that token prefix.
- Long prompts use layer-major prefill. This keeps one layer's routed experts
  resident while the runtime processes its token chunks.
- Automatic prefill selects 128, 256, or 512 tokens from the uncached prompt
  length.
- Single-token decode runs each selected routed expert directly. Multi-token
  prefill groups token routes by expert. Neither path builds stacked weights.
- MXFP8 sparse attention selects packed top-k rows before dequantization. It
  does not dequantize and merge every 64-row chunk.
- MXFP8 index scoring combines completed chunks and uses one quantized matrix
  multiplication. The unquantized tail stays on the BF16 path.
- The MXFP8 cache reuses its packed representation until a completed chunk is
  added.
- Expert eviction uses one global LFU heap. It keeps the same per-layer reserve
  without scanning all 43 layers for each eviction.

A short same-process test produced the same greedy output in both runs. The
first run took 2.82 seconds. The second run took 1.83 seconds and reached 7.99
decode tokens per second. Cache-state evaluation took about 0.064 seconds for
eight output tokens in each run. The second run still read about 11.1 GB of
expert data. This result makes expert-cache traffic the next measured target.

The cache-state change is a protection. It is not a 12K-token stability test.
Long-output support remains unverified until that test passes.

The prefill comparison used the same 321-token prompt and one output token.
The 16, 32, 64, and 128-token steps took 28.86, 22.78, 19.03, and 17.14
seconds. The 128-token step was 24.8 percent faster than the old 32-token
default. Peak memory increased by about 569 MB. The runtime now uses 128 as
the default.

A two-turn chat reused 17 of 26 prompt tokens in the second turn. Time to first
token decreased from 2.32 to 1.63 seconds. Total request time decreased from
3.48 to 2.94 seconds. Both paths produced the same greedy answer.

Three other measured changes were not kept. A layer-order eviction policy
increased cache misses. The sorted decode gather reached 7.69 to 7.75 tokens
per second, while the original path reached 8.30 to 8.35. Increasing the cache
from 1,024 to 1,536 slots did not improve total time, and 2,048 slots was
slower because of memory pressure.

The routed expert microbenchmark used real expert blobs and complete matrix
sizes. The original stack and gather path took about 1.27 ms per routed layer.
The direct six-expert path took about 0.56 ms and produced an identical result.
In the complete model, warm decode increased from 8.35 to 10.70 tokens per
second. The 32-token request decreased from 4.86 to 3.97 seconds. The greedy
output stayed identical. The 321-token prefill did not regress.

The grouped prefill microbenchmark used a real 128-token layer with 71 routed
experts. The stack and gather path took about 41.3 ms. The grouped path took
about 13.0 ms. A 321-token prompt decreased from 16.25 to 10.76 seconds. The
4K BF16 test decreased from the historical 124.0 seconds to 30.74 seconds. The
8K MXFP8 test decreased from 264.1 to 124.32 seconds. Both long tests kept the
same greedy output, ` test`.

The original 2,048-row MXFP8 gather took about 64.8 ms for 128 queries. Packed
row selection took about 1.08 ms and had zero numerical difference. Metal
capture reduced buffer records from 132 to 41. The number of 64 MiB buffers
decreased from 13 to one. The 8K MXFP8 test decreased again from 124.32 to
67.32 seconds. Its greedy output stayed ` test`.

For 2,048 cached rows and 128 queries, combined MXFP8 index scoring decreased
from about 1.30 ms to 0.48 ms. The maximum numerical difference was zero. The
8K MXFP8 test then decreased from 67.32 to 67.14 seconds. Its greedy output
stayed ` test`.

Reusing the packed representation reduced the repeated index and gather
microbenchmarks by about 16 percent. The 321-token test increased decode speed
from 13.73 to 14.14 tokens per second. The 8K MXFP8 test decreased from 67.14
to 66.50 seconds. Its output and 22.2 GiB peak memory stayed unchanged.

The global eviction heap produced the same hit and miss counts as the previous
layer heaps. In the 8K test, eviction time decreased from 1.03 to 0.23 seconds.
Total time decreased from 66.50 to 65.47 seconds. Output and peak memory stayed
unchanged.

## Scope and terms

This note checks three kinds of support:

- **main model**: The 43 target-model layers. It excludes DSpark.
- **MTP**: The checkpoint namespace and configuration used for the attached prediction module.
- **DSpark**: The speculative decoding module stored under `mtp.*`.

“Fully supported” means that the runtime loads the relevant checkpoint weights, runs the official algorithm, and has validation for the model's stated operating range. A successful short greedy test is narrower than full support.

This note uses only first-party DeepSeek, Hugging Face model-repository, MLX, and MLX-LM sources. Local project files provide evidence about this runtime.

## Result

| Component | This runtime | Official MLX-LM main | Finding |
| --- | --- | --- | --- |
| Main model | Runs validated batch-size-1 greedy paths at 4K and 8K | No `deepseek_v4` model module at the checked commit | Supported for the project's tested scope. Not fully supported for the official 1M range or every official message feature. |
| MTP weights | Excluded from the installed model | No DeepSeek-V4 model module | Not supported. |
| DSpark | Disabled and not installed | Generic separate-draft speculation only | Not supported. |

The checked MLX-LM revision is `254d153fdeb6f150edd4fc5a54f9828638481fa8`, dated 2026-08-05. Its [model directory](https://github.com/ml-explore/mlx-lm/tree/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/models) has no `deepseek_v4.py`. Its loader imports `mlx_lm.models.<model_type>` and rejects a missing module ([source](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/utils.py#L177-L194)). The DeepSeek-V4 pull request was closed without merge ([MLX-LM PR 1192](https://github.com/ml-explore/mlx-lm/pull/1192)).

## Facts

### Official checkpoint

- Revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062` defines 43 main-model layers, a 1,048,576-token limit, 256 routed experts, six selected routed experts per token, three hash-routed layers, a 128-token sliding window, compression ratios 4 and 128, and an index top-k of 512 ([official config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)).
- The main model has 284B total parameters and 13B active parameters per token ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S4.SS2.SSS1)). The full Hugging Face checkpoint is larger because it also contains the attached module under `mtp.*` ([official tensor index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json)).
- The official inference configuration defines three DSpark layers, a draft block size of five, target hidden states from main-model layers 40 through 42, and a Markov rank of 256 ([official inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json)).
- The checkpoint index contains 4,705 tensor names under `mtp.*`. Their calculated size is about 10.1168 GiB ([official tensor index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json), [local size record](../../docs/VALIDATION.md)).
- The official reference code defines `DSparkBlock` under the `mtp.*` namespace. It shares the main embedding and output head. It consumes main-model hidden states and adds Markov and confidence heads ([official reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L743-L936)).
- The bundled simple generator calls only the main-model forward path. It does not call `forward_spec` ([official generator](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/generate.py)). The model card directs users to vLLM or SGLang for DSpark ([official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md#how-to-run-with-vllm)).
- The official vLLM example enables `method: "dspark"` with seven speculative tokens. The official SGLang example also selects `DSPARK` and loads target and draft weights from the same checkpoint ([official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md#how-to-run-with-vllm)).
- Routed expert weights use checkpoint-native FP4. Other quantized weights use FP8 E4M3 with UE8M0 scales and 128 by 128 blocks ([official config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json), [official conversion code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/convert.py)).

### This runtime

- The repack plan removes every `mtp.*` tensor from common tensors. The installed model therefore contains the main model only ([repack planner](../../Sources/DeepSeekRepack/RepackPlanner.swift#L25-L30)).
- The runtime imports a pinned, non-upstream `deepseek_v4` implementation. It replaces routed expert execution with SSD streaming and uses `mx.gather_qmm` in MXFP4 mode ([requirements](../../requirements.txt), [runtime model](../../runtime/deepseek_v4_ssd/model.py#L7-L74)).
- Local validation covers batch size 1 and greedy decoding. It covers 4,096 tokens with BF16 cache and 8,192 tokens with BF16 and MXFP8 cache. It does not cover the official 1M-token range, sampling parity, or DSpark ([validation record](../../docs/VALIDATION.md)).
- A local mHC Metal-kernel comparison supplied for this review passed against its fallback path with a maximum absolute difference of `1.49e-7`. The compared implementation comes from the pinned runtime dependency ([requirements](../../requirements.txt)). This result supports the tested mHC operation. It does not extend the context or generation support claims above.
- The official encoder supports `reasoning_effort` values `low`, `high`, and `max`, tool definitions, tool calls, tool results, and structured `response_format` instructions ([official encoding guide](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/README.md), [official encoding code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/encoding_dsv4.py)). The runtime loads this pinned official encoder from the installed model. The API exposes `reasoning_effort` through Chat Completions and `reasoning.effort` through Responses API. The API does not yet expose `response_format` ([local Tool codec](../../runtime/deepseek_v4_ssd/tool_codec.py)).
- The 8K MXFP8 run took 264.1 seconds. Routed expert reads used about 34 seconds and read 1,340.2 GB. The recorded hit rate was 57.2 percent ([validation record](../../docs/VALIDATION.md), [DSpark decision](../../docs/VALIDATION.md)).
- Multi-token prefill slices and stacks expert regions, then uses compact stack indices for three routed MXFP4 matrix multiplications. Single-token decode now runs the six experts directly without stacked weight buffers ([expert pool](../../runtime/deepseek_v4_ssd/expert_cache.py), [routed execution](../../runtime/deepseek_v4_ssd/model.py)).
- The runtime keeps up to two in-memory prompt cache timelines within an 8 GiB
  default limit. It reuses the longest complete token prefix. A mismatch creates
  a new cache timeline ([generation path](../../runtime/deepseek_v4_ssd/generation.py)).
- The custom MXFP8 pooling cache becomes non-trimmable after it stores compressed entries ([local cache](../../runtime/deepseek_v4_ssd/fp8_cache.py#L217-L218)). Generic MLX-LM speculative decoding needs cache rollback after rejected draft tokens, so it cannot be connected to this cache without a new trim or checkpoint-and-restore operation.
- An MLX-LM issue reports unbounded Metal-resource growth during long DeepSeek-V4 decode on the same experimental model branch. The reported workaround evaluates cache state every decode step ([MLX-LM issue 1332](https://github.com/ml-explore/mlx-lm/issues/1332)). This runtime now applies that workaround. It has not validated a 12K-token decode.

### MLX and MLX-LM capabilities

- MLX supports MXFP4 and MXFP8. It defines `gather_qmm` with right-hand matrix indices and a `sorted_indices` hint ([MLX operation source](https://github.com/ml-explore/mlx/blob/39d9a8ac2b5449a49a910429e0edfedc8ff81372/python/src/ops.cpp#L4623-L4806)).
- The Metal implementation has a separate right-index gather path. It can use a sorted-index fast path for suitable matrix shapes ([MLX Metal source](https://github.com/ml-explore/mlx/blob/39d9a8ac2b5449a49a910429e0edfedc8ff81372/mlx/backend/metal/quantized.cpp#L1416-L1894)).
- MLX-LM supports speculative decoding with a separate autoregressive `draft_model`. It drafts in a loop, verifies with the target model, and requires trimmable caches ([MLX-LM generation source](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/generate.py#L464-L644)). This path is not the integrated, parallel DSpark module stored under `mtp.*`.
- DeepSeek-V4 uses heterogeneous caches. The official design keeps sliding-window state and incomplete compression state separate from compressed cache blocks ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S3.SS5.SSS1)). The paper uses BF16 for RoPE dimensions and FP8 for the other KV dimensions ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S2.SS3.SSS3)).
- The official production design can store compressed KV entries on disk and reuse shared prefixes. Sliding-window state needs full caching, periodic checkpoints, or bounded recomputation ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S3.SS5.SSS2)).

## Inferences

### Support status

1. **The main model has targeted support, not full support.** The local 4K and 8K greedy results show that the implemented path works for those tests. They do not validate the official 1M limit, all generation modes, or the complete official message format. The use of a closed, non-upstream MLX-LM branch also makes updates harder ([local validation](../../docs/VALIDATION.md), [official encoding guide](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/README.md), [upstream model directory](https://github.com/ml-explore/mlx-lm/tree/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/models)).
2. **MTP is not a separate supported local feature.** For this checkpoint, `mtp.*` stores the attached DSpark layers. The repacker excludes that namespace, so the runtime cannot load those weights ([official reference model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py#L818-L936), [local repack planner](../../Sources/DeepSeekRepack/RepackPlanner.swift#L25-L30)).
3. **DSpark is not supported.** Generic MLX-LM speculation cannot replace DSpark. DSpark needs main-model hidden states, a parallel draft backbone, a Markov head, a confidence head, and a prefix scheduler ([DSpark paper](https://arxiv.org/html/2607.05147#S3), [MLX-LM generation source](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/generate.py#L464-L644)).

### Runtime bottleneck

The latest 8K measurement assigns about 14.0 of 65.5 seconds to SSD reads. Routing synchronization records about 37.5 seconds, but that value can include queued model work. The next speed work must separate attention matrix multiplication, routed matrix multiplication, and synchronization ([local validation](../../docs/VALIDATION.md)).

## Speed opportunities

The order below uses expected value and implementation risk. Expected gains are inferences until a target-machine benchmark confirms them.

| Priority | Work | Decision |
| --- | --- | --- |
| P0 | Add phase timers | Required before more optimization. |
| P0 | Run a 12K-token decode stability test | Confirm or fix cache-state growth before claiming long output support. |
| Done | Remove expert stack creation | Complete for decode and prefill. |
| Done | Group prefill work by expert and tune chunk size | Grouping and automatic 128, 256, or 512-token chunks are active. |
| Done | Add layer-major prefill | Prompts with at least 4K uncached tokens keep one layer's experts resident. |
| Done | Add prefix-cache reuse | Two continued-chat timelines are retained within a memory limit. |
| Done | Remove repeated layer scans during expert eviction | One global LFU heap keeps the per-layer reserve. |
| Partial | Fuse sparse attention and quantized cache reads | Packed top-k selection and packed index scoring are complete. Full kernel fusion remains. |
| P3 | Test an FP4 indexer cache | Keep only after top-k and output validation pass. |
| P4 | Implement DSpark | Keep disabled until measured net speed is positive. |

### 1. Remove expert stack creation from the main path

The runtime reads individual expert views from stable slots. Decode runs the six selected experts directly. Prefill groups token routes by expert and restores the original route order before weighted reduction ([current pool](../../runtime/deepseek_v4_ssd/expert_cache.py), [routed execution](../../runtime/deepseek_v4_ssd/model.py)).

This removes the stacked weight buffers without copying the complete slot pool.

### 2. Group routed work by expert during prefill

The runtime groups token-expert pairs by resident expert. It runs the three expert matrix multiplications for each group and restores the original route order before weighted reduction.

Single-token decode keeps its smaller direct path. It does not pay the grouping cost.

### 3. Tune prefill chunk size with phase metrics

The local comparison now covers 128, 256, and 512 tokens. The runtime uses 128
below 1K uncached tokens, 256 below 4K, and 512 from 4K onward. Layer-major
prefill reduced the 8K test from 47.04 to 38.20 seconds at a 512-token step
([validation record](../../docs/VALIDATION.md)).

Select the chunk size from total prompt time, peak memory, expert bytes read, stack time, routing synchronization, and cache hit rate.

### 4. Fuse sparse attention and quantized cache reads

The MXFP8 path now concatenates small packed chunks for index scoring. It also
selects top-k packed rows and dequantizes only the selected result
([local cache gather](../../runtime/deepseek_v4_ssd/fp8_cache.py), [local sparse attention](../../runtime/deepseek_v4_ssd/model.py)). A future DeepSeek-V4-specific Metal kernel could also apply masks, include attention sinks, and accumulate attention without materializing the selected BF16 values.

This matches the official model's cache-and-kernel co-design. DeepSeek states that cache layout, alignment, and sparse attention kernels must be designed together ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S3.SS5.SSS1)). This work has higher implementation risk than expert-pool changes.

### 5. Add prefix-cache reuse

The runtime now keeps two in-memory prompt cache timelines. A token-prefix check
selects the longest matching timeline. An optional warmup prompt prepares one
fixed prefix before the server accepts requests ([local generation path](../../runtime/deepseek_v4_ssd/generation.py)). MLX-LM supports an explicit prompt cache ([MLX-LM generation source](https://github.com/ml-explore/mlx-lm/blob/254d153fdeb6f150edd4fc5a54f9828638481fa8/mlx_lm/generate.py#L298-L436)). DeepSeek's official design stores compressed entries and recomputes incomplete tails when it reuses a disk prefix ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S3.SS5.SSS2)).

This change improves time to first token for repeated prefixes. It does not improve steady one-token decode.

### 6. Evaluate an FP4 indexer cache separately

DeepSeek applies quantization-aware training to routed expert weights and the indexer QK path. The official launch recipe enables an FP4 indexer cache ([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S5.SS2.SSS1), [official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md#how-to-run-with-vllm)). MLX supports MXFP4 quantized matrix multiplication ([MLX operation source](https://github.com/ml-explore/mlx/blob/39d9a8ac2b5449a49a910429e0edfedc8ff81372/python/src/ops.cpp#L4623-L4700)).

Keep the attention-value cache in its validated format. Add an FP4 copy only for index scoring. Compare top-k indices, final greedy tokens, memory, and speed at 8K before longer tests.

### 7. Keep DSpark last

DSpark can improve production generation speed. DeepSeek reports 60 to 85 percent higher per-user V4-Flash generation speed than its MTP-1 baseline at matched production throughput ([DSpark paper](https://arxiv.org/html/2607.05147#S1)). That result comes from a multi-user GPU serving system. It is not a speed forecast for batch-size-1 Apple Silicon with SSD expert streaming.

DSpark adds three routed-MoE layers to this checkpoint ([official inference config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json), [official tensor index](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json)). Target verification of several tokens can also select more unique main-model experts than one-token decode. Therefore, DSpark can increase both draft I/O and verification I/O in this runtime.

Implement DSpark only after the main path has phase metrics. Enable it only if this measured value is lower than the autoregressive baseline:

```text
(draft time + target verification time) / accepted output tokens
```

The measurement must also record accepted tokens per round, DSpark expert bytes, target expert bytes, cache hit rate, peak memory, and output parity. This follows the DSpark paper's latency model ([DSpark paper](https://arxiv.org/html/2607.05147#S2.SS1)).

## Recommended next work

1. Add GPU-phase timers and separate prefill from decode. **Started:** request-level prefill and decode timers are active. Kernel-level timers still need a Metal capture.
2. Run a 12K-token decode while tracking active Metal resources and cache state.
3. Remove expert stack creation. **Done:** decode and prefill use individual expert views.
4. Add expert-grouped prefill. **Done:** grouped output passes route-order tests.
5. Run the prefill chunk-size matrix on the target M5 Pro. **Done:** 128 tokens is the default.
6. Add in-memory prefix reuse. **Done:** one continued-chat prefix is reused.
7. Prototype fused sparse attention. **Partial:** packed top-k gather and packed index scoring are complete. Full attention fusion remains measurement-dependent.
8. Add cache rollback, then keep `mtp.*` excluded until a complete DSpark prototype beats the autoregressive baseline.

## Validation gates

Every optimization must keep these checks:

- Identical routed expert output against the dequantized reference.
- Identical 4K and 8K greedy tokens against the current baseline.
- Stable top-k tie behavior.
- Identical output across prefill chunk boundaries.
- Recorded peak memory, prompt time, decode tokens per second, expert bytes, cache hit rate, and each phase time.
- A separate 1M support claim only after a 1M end-to-end test passes. The official context limit alone is not validation ([official config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)).
