# DeepSeek-V4-Flash-0731 runtime speed research

> [!WARNING]
> Historical snapshot from 2026-08-07. Its estimates are not current results.
> Use [the current performance guide](../../docs/PERFORMANCE.md) and
> [validation record](../../docs/VALIDATION.md).

Checked: 2026-08-07

## Decision

The current layer-major prefill is the correct base. Keep it.

The next cold-prefill work should use this order:

First, take a Metal capture. This is a validation tool. It is not a speed
optimization.

The first implementation item is a larger layer-local MoE tile with official
MLX-LM `gather_qmm`:

1. Split each layer into chunked attention/router work and layer-local MoE work.
2. Use official MLX-LM `gather_qmm` if a no-copy expert layout is possible.
3. Read routed experts in large ranges and overlap those reads with GPU work.
4. Test 1,024, 2,048, and 4,096-token attention/router chunks.
5. Compile fixed-shape, pure MLX sections that run many times.
6. Fuse only the sections that the capture shows are expensive.
7. Test an FP4 lightning-index cache separately from the attention-value cache.

Persistent shared-prefix cache is the highest-value request-level change. It
does not make a new, unrelated prompt faster. It can remove nearly all prefill
for a repeated system prompt, Tool schema, or document prefix.

Do not treat the reported 30.07 seconds of routing synchronization as router
overhead. The runtime calls `mx.eval(indices)`. MLX is lazy, so this call also
waits for all earlier work that produces the layer input. The value is a useful
boundary timer. It is not 30.07 seconds that a different stream can remove.

## Scope and evidence

This report excludes DSpark and MTP.

The current 14,363-token Tool-like test is the main cold-prefill baseline:

| Mode | Step | Total | Routed expert reads | Expert read time | Routing boundary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chunk-major | 512 | 386.65 s | 2.29 TB | — | — |
| Layer-major | 512 | 105.81 s | 131.5 GB | 11.48 s | 30.07 s |

Layer-major prefill reduced routed expert traffic by 94.3%. The remaining
131.5 GB is about 89.4% of all routed expert bytes in the main model. A long,
diverse prompt therefore selects almost every routed expert once. A larger
expert cache cannot remove much more traffic from this case.

The read-time ceiling is also clear. Removing all 11.48 seconds would reduce
105.81 seconds by at most 10.85%. Any larger cold-prefill gain must also reduce
GPU work, graph overhead, or the waits between CPU and GPU work. These values
come from the [local validation record](../../docs/VALIDATION.md)
and the current [routed expert cache](../../runtime/deepseek_v4_ssd/expert_cache.py).

### New 2026-08-07 local measurement

A separate 4,097-token diverse Tool-like prompt used layer-major prefill and a
512-token tile. It measured:

| Value | 2026-08-07 result |
| --- | ---: |
| Time to first token | 43.19 s |
| `get_many` calls across 43 layers | 430 |
| Mean unique routed experts per call | 138.4 |
| Sum of unique routed experts across calls | 59,518 |
| Estimated `w1` + `w2` + `w3` matrix calls | 178,554 |
| Routed expert read time | 10.32 s |
| Routed expert bytes | 131.35 GB |

The current grouped path runs three quantized matrix operations for each unique
expert in each tile. If one layer gathers all of its tile routes before it runs
the MoE, each of the 256 experts needs at most three matrix operations for that
layer. The upper count becomes `43 × 256 × 3 = 33,024`. This is 5.41 times fewer
matrix nodes than 178,554. It is not a 5.41 times end-to-end speed claim.

The same measurement also confirms that I/O is not the only target. Expert
reads used 23.9% of time. The other 32.87 seconds include attention, routing,
MoE compute, graph work, and waits.

A small local MLX 0.32.0 feasibility test also passed. `gather_qmm` read MXFP4
weights and E8M0 scales through padded `mx.as_strided` expert views. Its
output had zero maximum absolute difference from the contiguous reference.
This result shows that the matrix operation accepts a strided expert axis. It
does not yet prove that the runtime can fill a complete layer view from SSD
without an extra full-layer copy.

DeepSeek-V4 uses 256 routed experts and selects six per token in each main-model
layer. Its official inference code groups tokens by routed expert before it
runs each expert. The model uses FP4 routed expert weights
([official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json),
[official inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py)).
The project already follows that grouping rule.

All improvement ranges below are engineering estimates unless a local result
is stated. They are ranges for this 14,363-token cold test. The ranges are not
additive.

## Implementation priority table

| Priority | Work | Expected cold-prefill improvement | Main risk |
| --- | --- | ---: | --- |
| P0 | Larger layer-local MoE tile, then `gather_qmm` | 10–35% | A changed evaluation order or weight copy |
| P0 | Persistent and shared-prefix cache | 80–97% for a large cache hit; 0% for an unrelated prompt | Incorrect hybrid-cache restoration |
| P1 | Large-range expert reads and I/O-to-GPU waves | 4–10% | Extra reads or slot pressure |
| P1 | Test 1,024–4,096-token layer-local chunks | 3–15% | Peak memory and large graphs |
| P1 | Compile fixed-shape pure MLX sections | 3–10% | Recompile cost and state capture errors |
| P2 | Fuse measured sparse-attention and MoE sections | 3–12% | Numerical or Metal-kernel errors |
| P2 | FP4 lightning-index cache | 2–8% | Changed top-k positions |
| P3 | KV-cache layout tuning | 0–5% for 14K cold prefill; larger at longer context | Wrong compression-tail or SWA state |
| P3 | Further common tensor quantization | 0–3% | Accuracy loss for little I/O benefit |
| Reject now | More GPU streams by themselves | 0–2% | More synchronization and unsafe state |

## 1. Measure the real GPU sections

**Applicability:** Immediate. The current routing timer includes earlier
attention, mHC, shared work, and queue delay. It does not isolate router cost.

MLX records a graph until `mx.eval()` runs it. MLX states that each evaluation
has fixed overhead and that both very small and very large graphs can cost more
([MLX lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html#when-to-evaluate)).
MLX also provides Metal capture. Xcode shows the operations and their
dependencies
([MLX Metal debugger](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)).

**Expected improvement:** None by itself. It prevents work on the wrong
bottleneck.

**Risk:** Extra `mx.eval()` calls change the graph and make the measured path
slower.

**Validation:** Capture one warm 512-token chunk at an early, middle, and late
layer. Record GPU duration and allocated bytes for attention, router, shared
expert, routed `quantized_matmul`, activation, route restore, and cache update.
Keep request-level timing outside the capture. Confirm that adding labels or
timers changes total time by less than 1%.

## 2. Pipeline routed expert I/O and compute

**Applicability:** High for long, diverse prompts. The test reads 89.4% of the
main model's routed expert bytes. The first 512-token chunk of a layer is likely
to select most or all 256 experts. The runtime now starts one `pread` task per
missing routed expert and waits for every task before it builds routed expert
work.

Use two related changes:

- Merge adjacent missing expert blobs into large `pread` ranges. Split each
  range into expert blobs after the read.
- Process ready experts in waves. Start GPU work for one wave while the CPU
  reads the next wave. For prompts with measured layer coverage above a fixed
  threshold, prefetch the complete next layer into unpinned slots.

DeepSeek uses the same basic overlap rule for expert parallelism. It splits
experts into waves and starts compute when one wave is ready
([DeepSeek-V4 paper, section 3.1](https://arxiv.org/html/2606.19348#S3.SS1)).
This project uses SSD I/O instead of network communication, so the expected
value is an inference from the same scheduling rule.

**Expected improvement:** 4–10%. The hard upper bound from the measured read
time is 10.85%. Large reads can also reduce Python future and system-call
overhead. A reasonable first gate is to hide 50–80% of expert read time without
increasing total bytes by more than 12%.

**Risk:** Full-layer prefetch reads unused routed experts on shorter or less
diverse prompts. A prefetched layer can evict useful decode experts. Partial
GPU outputs must preserve stable route order and score weighting.

**Validation:** Run 321, 4,096, 8,192, and 14,363-token prompts. Record expert
bytes, number and mean size of reads, read time, GPU idle gaps, peak memory,
cache hit rate, and total time. Compare cold and warm file-cache runs. Require
identical routed outputs and identical greedy tokens. Disable full-layer
prefetch when its last four layers use less than 75% of prefetched experts.

## 3. Test larger layer-local chunks

**Applicability:** High. The current order already keeps one layer's routed
experts resident across all token chunks. Moving back to token-major order
would repeat routed expert reads. A larger chunk keeps the same layer order but
reduces CPU router conversions, evaluations, and small matrix operations.

Test 1,024, 2,048, and 4,096 tokens against 512. DeepSeek's official SGLang
command uses a 4,096-token chunk
([official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md#how-to-run-with-sglang)).
That value comes from a different GPU system. It is a test point, not an Apple
Silicon default.

Also test a fixed-shape padded tail so MLX can reuse one compiled shape.
Replace the list of per-chunk outputs plus final concatenate with one bounded
output plan if Metal capture shows material copies or graph construction. Do
not make a tile span several layers unless the cache can keep all routed
experts for those layers. A multi-layer tile can undo the main I/O gain.

**Expected improvement:** 3–15% if 512 is still evaluation-bound. The result
can be 0% or negative if attention memory or graph size dominates. This work
does not reduce the near-one-pass expert traffic. The 30.07-second routing
boundary is not a valid estimate of the removable time.

**Risk:** Padding can change masks or cache offsets. A larger tile can increase
peak memory. A smaller tile adds more evaluations.

**Validation:** Use the existing chunk-boundary tests. Add prompt lengths 511,
512, 513, 1,023, 1,024, 2,047, 2,048, 4,095, 4,096, and 14,363. Require
identical cache offsets, top-k positions, and greedy output. Record peak
memory, graph build time, evaluation count, and Metal copy kernels. Keep the
fastest size that stays within the current memory limit.

## 4. Run MoE once per layer tile, then test `gather_qmm`

**Applicability:** Highest compute priority. The 2026-08-07 local measurement
counted about 178,554 routed expert matrix calls for 4,097 tokens. The current
prefill sorts routes inside each 512-token tile. It then runs `w1`, `w3`, and
`w2` once per selected expert in that tile.

First split one layer into two phases:

1. Process attention and router work in bounded token tiles. Retain the MoE
   input, route IDs, and route scores for the complete layer.
2. Group all layer routes by expert. Run each selected expert once over all of
   its layer positions. Restore token and route order. Then pass the complete
   layer output to the next layer.

This keeps attention cache updates bounded. It also keeps the current
individual expert slot layout. For the measured prompt, the routed matrix node
count has an upper bound of 33,024. This is 5.41 times fewer nodes. The GPU work
does not fall by 5.41 times because the matrix rows still contain the same
routes.

The phase split is valid only if the same-layer attention cache update does not
depend on that layer's MoE output. Confirm this against the official block
order before implementation. The official reference block runs attention and
then MoE in sequence
([official inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py)).

Official MLX-LM sorts large route sets and runs three `gather_qmm` operations
over an expert tensor
([official `SwitchGLU` source](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/switch_layers.py#L92-L158)).
MLX 0.32.0 has a sorted right-index path that batches work and reuses reads of
the input and weights
([official MLX Metal source](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/quantized.cpp#L1571-L1600)).

This project cannot use that path by stacking expert arrays on every call. The
old stack-and-gather path was slower than the current grouped path. Prototype a
single fixed, strided slot tensor that `gather_qmm` can index directly. SSD
fills must update slots without rebuilding or copying the complete tensor. If
MLX cannot provide that no-copy update safely, stop this work.

There are two independent prototypes:

- Layer-local grouping keeps the current individual expert arrays. It reduces
  the measured matrix-node upper count from 178,554 to 33,024.
- A fixed strided expert tensor lets each tile use three `gather_qmm` nodes.
  The measured 430 MoE calls would then use about 1,290 matrix nodes. Combining
  both changes can reduce that further, but the larger retained tile can cost
  more memory.

After the batched path works, test a fused `w1` and `w3` stored layout. Accept
it only when a local benchmark shows lower total time and no extra full-weight
copy.

**Expected improvement:** 10–35% for layer-local MoE tiling plus a no-copy
batched path. This estimate comes from the measured 5.41 times node reduction,
not from a measured end-to-end result. Layer-local grouping by itself can still
help if fixed-slot `gather_qmm` is not possible. Any full-layer or full-pool
weight copy can erase the batched-path gain.

**Risk:** MLX arrays do not expose a simple mutable SSD-fill interface. Slot
updates can race with GPU reads. One complete layer is about 3.42 GB. A second
layer tensor can cause memory pressure. Sorted route restoration must stay
exact.

**Validation:** Use the measured 4,097-token request and the 14K request. Record
routed matrix nodes and Metal dispatches separately, slot-fill copies, routed
compute time, total time, and peak memory. Compare the current per-tile grouped
path, layer-local grouping, stack-and-gather, and fixed-slot `gather_qmm`.
Require exact routed output and identical cache state after every layer. Reject
the `gather_qmm` layout if it copies more than the newly read expert bytes or
loses the layer-local grouped-path speed.

## 5. Compile only stable, pure MLX sections

**Applicability:** Medium to high. MLX compilation merges common work and fuses
some operations. MLX caches a compiled function, but a new input shape or type
can cause compilation again. MLX recommends compiling the outermost suitable
pure function
([MLX compilation guide](https://ml-explore.github.io/mlx/build/html/usage/compile.html)).

The whole layer is not a good first compile target. The routed expert path
converts router indices to NumPy and performs SSD reads. MLX does not allow an
array evaluation inside a compiled transformation. Start with repeated pure
sections with fixed shapes:

- router score, bias, normalization, and stable top-k before the CPU boundary;
- shared expert activation and reduction;
- routed score weighting, route restoration, and reduction;
- sparse-attention masks, score normalization, and output combine;
- mHC element-wise sections if the capture shows many small kernels.

Use explicit functions that live for the process lifetime. Do not create a
compiled lambda inside a loop. Use separate compiled entries for 128, 256, and
512 tokens. Treat the final tail separately. Test `shapeless=True` only where
no branch or reshape uses a static length.

**Expected improvement:** 3–10%. The low end is more likely if quantized matrix
multiplications dominate. The high end requires many small element-wise and
copy kernels in the capture.

**Risk:** Wrong state capture, long first-call compilation, or frequent
recompilation. Compilation cannot remove the CPU need to read router indices.

**Validation:** Count compilation events and distinct shapes. Warm all three
standard shapes before timing. Compare first request, second request, and
steady state. Require numerical parity at every compiled boundary and
identical greedy output.

## 6. Use `async_eval` for overlap, not extra streams by default

**Applicability:** Limited. The runtime already calls `mx.async_eval(shared)`
before it loads routed experts. MLX-LM also starts the next decode step with
`mx.async_eval` before it waits for the current token
([official MLX-LM generation source](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/generate.py#L453-L469)).

Continue this pattern for I/O waves. Do not add a second GPU stream until a
capture shows idle GPU time that an independent graph can fill. MLX operations
use the default stream unless a stream is specified
([MLX streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)).
The router indices still depend on the current layer input, so a new stream
cannot make them available early.

**Expected improvement:** 0–2% from stream changes alone. Wave-based
`async_eval` is included in the 4–10% I/O estimate.

**Risk:** Cache updates and slot reuse can race with GPU reads. More streams can
add waits and make graph ownership harder to verify.

**Validation:** Use a Metal dependency trace. Require no read of a slot until
its fill completes and no overwrite until all GPU users complete. Run the same
request 100 times and compare output and memory.

## 7. Fuse the measured sparse-attention and MoE gaps

**Applicability:** Medium. DeepSeek replaced many fine-grained operations with
fused kernels and states that its KV layout and sparse-attention kernel must be
designed together
([DeepSeek-V4 fused kernels](https://arxiv.org/html/2606.19348#S3.SS2),
[DeepSeek-V4 KV and sparse-kernel co-design](https://arxiv.org/html/2606.19348#S3.SS5.SSS1)).
MLX supports reusable custom Metal kernels through `mx.fast.metal_kernel`
([MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)).

Do not replace a tuned MLX `quantized_matmul` first. Fuse operations around it:

- index score post-processing, masks, and stable top-k when supported;
- packed top-k KV gather, score mask, sink handling, softmax, and value reduce;
- routed expert score multiply, restore, and reduce;
- LimitedSwiGLU element-wise work if it is a visible gap between matrix
  multiplications.

Build each custom kernel once. Reuse it for all layers with the same shapes and
types.

**Expected improvement:** 3–12%. Proceed only when the captured candidate is at
least 5% of total request time. Its request-level gain cannot exceed its
captured share.

**Risk:** Stable top-k tie behavior, masks, sinks, FP32 accumulation, and tail
lengths can change output. A custom kernel adds maintenance cost for each MLX
and macOS update.

**Validation:** Compare each kernel with the current MLX path on random and
real model tensors. Include equal-score top-k cases, masked rows, incomplete
compression blocks, and all three chunk sizes. Require identical selected
positions and the existing greedy outputs. Re-run the capture and require
fewer dispatches and lower GPU duration.

## 8. Add an FP4 lightning-index cache, not general FP4 KV

**Applicability:** Medium. The current project stores completed pooling-cache
chunks as MXFP8. DeepSeek uses BF16 for RoPE dimensions, FP8 for other KV
dimensions, and FP4 for lightning-indexer attention
([DeepSeek-V4 efficiency discussion](https://arxiv.org/html/2606.19348#S2.SS3.SSS4)).
The model received FP4 quantization-aware training for the indexer QK path
([DeepSeek-V4 paper](https://arxiv.org/html/2606.19348#S1)).

Keep the validated MXFP8 attention-value cache. Add a separate FP4 packed view
only for index scoring. Rebuild the view only when a complete compression
block is added. MLX 0.32.0 supports MXFP4 quantized matrix multiplication; the
project already uses it for routed experts.

**Expected improvement:** 2–8% at 14K. The gain should grow with context length
because index scoring reads more cached rows. Memory for that packed index view
should fall by close to half, before metadata and tails.

**Risk:** FP4 can change selected positions even when the final values stay in
MXFP8. A separate view also adds memory and update work.

**Validation:** Measure top-k overlap and exact equality for every query at 4K,
8K, and 14K. Report the maximum score error, changed positions, final hidden
error, greedy tokens, cache bytes, and index time. Reject the change if the
official checkpoint path does not supply the needed trained scale layout or
if greedy parity fails.

## 9. Persist complete shared-prefix states

**Applicability:** Very high for Tool-like work. The runtime now holds two
prompt timelines in memory. A restart removes them. It also stores only the
completed request timeline, so divergent requests can miss a common system or
Tool-schema boundary.

MLX-LM can save and load a prompt cache as safetensors
([official cache source](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/cache.py#L43-L85)).
Its LRU prompt cache uses a token trie and returns the nearest shorter prefix
without trimming it
([official LRU source](https://github.com/ml-explore/mlx-lm/blob/ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd/mlx_lm/models/cache.py#L1623-L1707)).
DeepSeek-V4 stores complete compressed CSA and HCA blocks on disk. It recomputes
an incomplete tail. For sliding-window state, it supports full state,
periodic checkpoints, or bounded recomputation
([DeepSeek-V4 on-disk KV cache](https://arxiv.org/html/2606.19348#S3.SS5.SSS2)).

Use content-addressed cache keys that include checkpoint revision, runtime
cache format version, exact token prefix, and cache options. Save explicit
checkpoints at stable message boundaries, especially after the system prompt
and Tool schema. Store only complete compression blocks. Store the fixed SWA
state at that boundary or use periodic SWA checkpoints.

**Expected improvement:** 80–97% time-to-first-token reduction when at least
80–99% of a long prompt is reused. The local 4K continuation already decreased
from 20.89 seconds to 0.56 seconds, a 97.3% reduction. An unrelated prompt gets
0%.

**Risk:** DeepSeek-V4 has heterogeneous caches and incomplete compression
tails. A generic save can restore an internally inconsistent position. Disk
cache corruption or a stale checkpoint can silently change output.

**Validation:** Restart the process before reuse. Test exact prefixes and
divergent suffixes at every message boundary. Test offsets just before and
after compression boundaries. Compare all cache states with a fresh prefill,
then compare the next-token logits and greedy output. Use an atomic write and a
checksum. Reject any checkpoint with a different checkpoint revision or cache
format version.

## 10. Keep the current heterogeneous KV design

**Applicability:** Medium for memory, lower for this 14K speed test. DeepSeek-V4
has separate classical compressed KV entries and fixed-size state for SWA and
incomplete compression tails. The paper states that layer cache sizes and
update rules differ
([DeepSeek-V4 cache structure](https://arxiv.org/html/2606.19348#S3.SS5.SSS1)).

Keep MXFP8 completed compressed entries, BF16 tails, and bounded SWA state.
Align completed blocks to the least common multiple of the layer compression
ratios if a future fused kernel uses block reads. Do not replace the model's
hybrid cache with the generic MLX-LM rotating cache. MLX-LM documents rotating
cache as a memory and quality trade-off for conventional models
([MLX-LM long-prompt guide](https://github.com/ml-explore/mlx-lm#long-prompts-and-generations)).

**Expected improvement:** 0–5% at 14K. The main value is lower memory and
better scaling beyond 14K. Persistent prefix reuse can provide a much larger
request-level gain.

**Risk:** A wrong block boundary changes attention. Evaluating too little cache
state can also retain long graph chains. Preserve the current per-token cache
state evaluation until a local memory test proves that a different boundary is
safe.

**Validation:** Preserve the current per-token cache-state evaluation. Test
compression boundaries, 4K, 8K, 14K, and a 12K-token decode. Track active Metal
resources, cache bytes, top-k positions, and output parity.

## 11. Do not quantize common tensors without a measured target

**Applicability:** Low. Common tensors stay resident. They do not cause the
131.5 GB of routed expert reads. The checkpoint already stores routed experts
in FP4 and other quantized weights in its trained formats
([official configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json),
[official conversion code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/convert.py)).
MLX unified memory lets CPU and GPU use the same arrays without device copies
([MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)).

Keep checkpoint-native common tensor precision and keep the common tensors
wired within the tested memory limit. Only test lower precision for a common
matrix when the Metal capture shows that matrix is bandwidth-bound and the
checkpoint has a trained quantized representation for it.

**Expected improvement:** 0–3% for the 14K test. The more likely gain is memory
headroom, not less routed expert I/O.

**Risk:** New quantization changes logits and can add dequantization cost. It
also changes the installed-model contract.

**Validation:** Benchmark one matrix at a time. Compare its output with the
checkpoint-native path, then run 4K, 8K, and 14K greedy tests. Record common
tensor bytes, peak memory, GPU duration, and page faults.

## Recommended test sequence

1. Capture the current 512-token path. Correct the routing-boundary label.
2. Add persistent message-boundary caches with restart validation.
3. Prototype layer-local MoE grouping behind a flag.
4. Prototype a fixed slot tensor and `gather_qmm`. Stop if it needs a pool copy.
5. Test 1,024, 2,048, and 4,096-token layer-local chunks.
6. Add large-range expert-read benchmarks without changing the runtime path.
7. Prototype expert waves behind a flag. Measure cold and warm file-cache runs.
8. Compile the two largest pure element-wise sections from the capture.
9. Prototype FP4 index scoring and compare every selected position.
10. Write a custom Metal kernel only for a section that remains at least 5% of
   total time.

For the cold 14,363-token test, the near-term target is 85–100 seconds. A
no-copy fixed expert tensor plus effective `gather_qmm` should target 65–90
seconds. The lower end also needs useful I/O overlap and fewer GPU gaps. A
repeated 4K-or-longer prefix should target less than 5 seconds plus the cost of
the new suffix. These are targets, not measured results.

Every accepted change must report total time, time to first token, peak memory,
routed expert bytes, expert read time, cache hit rate, Metal GPU time, routing
boundary time, and exact greedy output. Keep the current 321, 4K, 8K, and 14K
cases so a long-prompt win cannot hide a short-prompt regression.
