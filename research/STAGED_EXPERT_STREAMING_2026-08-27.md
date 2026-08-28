# Staged `w13`/`w2` expert-streaming microbenchmark

Status: research-complete for the tested fixed split and runtime schedule.
The fixed-arena feasibility gate passed, so a default-off runtime prototype was
measured. The runtime correctness gate passed, but the performance gate failed;
the candidate is stopped and all defaults remain unchanged. The protocol below
was fixed before either artifact was recorded.

## Question

The current direct path reads one canonical 13,369,344-byte expert blob into a
fixed fused slot before running `w13`, activation, and `w2`. The PLAN proposes
reading the first-stage `w1`/`w3` bytes separately, starting the `w2` read, and
overlapping that read with first-stage GPU work.

This gate asks whether that overlap window is material on this Mac before the
project changes cache ownership or the installed layout.

## Equivalence boundary

The candidate is target-equivalent only if it preserves all six canonical
expert regions and performs the same MXFP4 operations in the same order:

```text
quantized_matmul(w13) -> LimitedSwiGLU -> quantized_matmul(w2)
```

The control uses the runtime's existing fused full-slot layout. The candidate
uses two preallocated arrays with the same total byte budget:

- `w13`: `w3.weight`, `w1.weight`, `w3.scale`, `w1.scale` = 8,912,896 bytes;
- `w2`: `w2.weight`, `w2.scale` = 4,456,448 bytes.

Their sum must remain exactly 13,369,344 bytes. No quantization, tensor shape,
activation, output dtype, or arithmetic order may change.

## Read and compute protocol

Use the verified installed model and Darwin descriptor bypass policy. Select
expert ranges just in time only when `mincore` reports the whole canonical blob
nonresident. Disable read-ahead and apply `F_NOCACHE`; record residency before
and after each control and candidate read. This is still a page-cache contract,
not a storage-controller or physical-device byte counter.

Preallocate and evaluate all MLX arrays before timing. Warm every tested shape
outside the measured samples. For each selected expert and input-row count:

1. Control: one positional vectored read writes all six source regions into the
   existing fused full arena; evaluate `w13`, activation, and `w2` in two
   explicit synchronization sections.
2. Candidate: two positional vectored reads load the physically separated
   `w1` and `w3` groups into the fixed `w13` arena.
3. Start one background positional vectored read for the contiguous `w2`
   weight/scale group into the fixed `w2` arena.
4. Evaluate the real MLX `w13` QMM and `LimitedSwiGLU` while the `w2` read is in
   flight, wait only for any remainder, then evaluate the same down QMM.
5. Alternate control/candidate order across samples.

Test 1, 4, and 8 rows with at least 12 paired samples per shape. One row models
direct decode. Four and eight rows intentionally put every row on one expert;
they are an optimistic upper bound for block verification, not an observed V4
route distribution.

## Required measurements

For every pair record:

- layer, expert, method order, source offsets, destination offsets, and syscall
  count;
- full, `w13`, and `w2` bytes plus per-region SHA-256;
- float32 output SHA-256 for control and candidate;
- full-read, `w13`-read, `w2`-read, `w13` compute, residual `w2` wait, down-QMM,
  and complete wall time;
- actual interval intersection between `w2` read and `w13` compute;
- fraction of `w2` read hidden by first-stage compute;
- page residency before and after both paths;
- fixed-arena sizes, MLX version, source hashes, checkpoint revision, hardware,
  OS, and cache state.

The artifact must embed every raw pair. Hashing and residency probes remain
outside timing.

## Correctness gate

The native microbenchmark passes correctness only when:

1. every control and candidate reads exactly one canonical blob's total bytes;
2. all six corresponding region hashes match;
3. every control/candidate float32 output hash is exact;
4. all destination offsets and lengths satisfy the installed contract;
5. fixed candidate arena bytes equal the current full-slot bytes;
6. no selected range becomes resident under the tested bypass descriptor.

Any mismatch, short read, unsafe arena reuse, or residency-policy failure stops
the candidate.

## Continuation and stop criteria

Continue to a runtime token-hash prototype only if at least one of the 4- or
8-row shapes satisfies all of the following paired medians:

- at least 20% of the `w2` read interval is hidden by measured first-stage GPU
  compute;
- complete candidate wall time is at least 5% lower than the full-slot control;
- candidate complete-wall p95 does not regress by more than 5%;
- the complete correctness gate passes.

Stop when overlap exists only on the timeline but split-read syscall cost or
the residual wait removes the end-to-end gain. A passing optimistic row shape
does not authorize a runtime default. It only permits an isolated direct-path
prototype, which must then preserve full-model greedy token hashes, expert
bytes, cache lifetime, and slot fences under real route distributions.

## Evidence limits

This gate does not measure a complete MoE layer, weighted route reduction,
attention overlap, end-to-end decode tokens/s, power, ANE contention, private
Metal buffers, MTLIO, or physical SSD bytes. Failure rejects the tested split
layout and overlap schedule on this environment; it does not prove that a new
kernel or a future storage/resource API cannot create a larger compute window.

## Predeclared runtime token-hash follow-up

Only if the fixed-arena gate continues, add a default-off runtime prototype.
The prototype may split each lazily allocated direct expert slot into separate
fixed `w13` and `w2` MLX arrays, but their total byte budget must remain one
canonical blob. It must not change the manifest, full-layer batched prefill,
DSpark, speculative scratch semantics, quantization, route order, or defaults.

For a direct miss, read `w13`, launch `w2` on a separate bounded executor,
submit the first-stage MLX graph, wait for the specific expert's `w2`, and only
then submit the down QMM. Do not admit the slot as complete until `w2` succeeds.
Errors or cancellation must release the partial slot. Resident hits must not
perform a new staged read.

Run four fresh-process waves over the exact 128-token `repeated` prompt with 32
greedy output tokens. Alternate `control, staged` and `staged, control` per wave.
Disable persistent prompt cache, layer-major prefill, DSpark, and approximate
behavior; use the explicit expert-file bypass policy. Record complete token
IDs/hash, request/TTFT/decode p50/p95, logical expert bytes, cache hits/misses,
evictions, staged read/wait counters, peak MLX memory, source hashes, and raw
subprocess output.

The runtime correctness gate requires every staged token sequence to match its
paired control and one shared reference hash; logical expert bytes and
evictions may not increase. Stop immediately on a partial-slot admission,
short read, deadlock, resource error, or token mismatch.

The isolated performance candidate passes only when paired median request time
or decode throughput improves by at least 5%, decode p95 does not regress more
than 5%, and peak MLX memory does not increase more than 5%. Passing this short
single-workload gate still leaves the option default off and requires a later
multi-workload, long-decode evaluation. Failure stops the tested runtime
candidate while preserving the fixed-arena artifact as a component result.

## Fixed-arena result

The fixed-arena probe completed 36 paired samples on Mac14,13／Apple M2 Max
64 GiB with MLX 0.32.0: 12 pairs each at 1, 4, and 8 input rows. Every sample
used one 13,369,344-byte control arena or fixed 8,912,896-byte `w13` plus
4,456,448-byte `w2` arenas. All six region hashes, float32 output hashes,
destination budgets, direct-I/O alignment checks, and post-read nonresidency
checks were exact.

| Rows | Candidate complete-wall median change | p95 change | Median `w2` read hidden | Decision |
| ---: | ---: | ---: | ---: | --- |
| 1 | -8.06% | +31.02% | 49.24% | Stop for one-row decode because p95 failed. |
| 4 | -10.74% | -10.24% | 80.98% | Continue to runtime prototype. |
| 8 | -13.67% | -12.67% | 87.90% | Continue overall; one order stratum was only -2.60%. |

The four-row result passed in both order strata (-5.72% and -15.81%). The
eight-row paired median passed, but its control-first stratum did not reach the
5% improvement threshold. Both shapes place every row on one routed expert and
therefore remain optimistic component evidence, not a measured route
distribution or a formal performance result.

The machine-readable artifact is
[`2026-08-27-staged-w13-w2-overlap-m2-max.json`](../docs/benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json),
SHA-256
`fa764c02403502fc4a50adbdbb37edd78e612774547ce28c6f8d898da3855f9f`.

## Default-off runtime prototype

The follow-up keeps each slot's total Metal-visible bytes equal to one expert
blob but allocates fixed `w13` and `w2` arrays. On a decode miss it reads `w13`
on the normal bounded executor, starts `w2` on a separate bounded executor,
submits the real first-stage MLX graph, waits only for that expert's `w2`, then
submits the down QMM. A split slot is marked loaded only after `w2` succeeds;
cancellation or error releases the partial slot. Resident hits perform no new
staged read. Full-layer batched prefill still uses the existing complete expert
path. `ModelRuntime` serializes generation, so another request cannot observe a
reserved partial slot.

The prototype is reachable only through the internal
`RuntimeConfig(staged_expert_streaming=True)` research harness. It defaults to
`false`, requires `ready_expert_decode`, rejects DSpark composition, and has no
CLI, server, or APP opt-in.

The installed-model gate ran four fresh-process waves in
control/staged, staged/control, control/staged, staged/control order. Every run
used the exact 128-token `repeated` prompt, produced 32 greedy tokens, disabled
persistent cache, layer-major prefill, and DSpark, and used expert-file bypass.
All eight output sequences share SHA-256
`06de0413ef6f0d6f67a8328b6cfbea9cf97fe28689b6503ad9975367a59d3fa7`.

| Metric | Control median | Staged median | Paired median change |
| --- | ---: | ---: | ---: |
| Request | 9.9083 s | 10.1274 s | +2.17% |
| Decode | 7.1176 tok/s | 6.8164 tok/s | -4.26% |
| Decode p95 | 208.29 ms | 212.72 ms | +2.12% |
| Peak MLX memory | 24,665,121,392 B | 24,665,121,392 B | 0% |
| Logical expert bytes | 40,121,401,344 B | 40,121,401,344 B | 0% |
| Expert evictions | 1,849 | 1,849 | 0% |

Every staged run completed exactly 1,024 split reads: 9,126,805,504 `w13`
bytes plus 4,563,402,752 `w2` bytes, which closes to 1,024 canonical expert
blobs. Median staged read wall was 2.4450 s, residual `w2` wait was 0.8780 s,
and first-stage submission time was 0.2503 s. Control staged counters were all
zero. Logical bytes and evictions never increased in any pair, and no short
read, deadlock, partial admission, or resource failure occurred.

Correctness therefore passes, as do the p95 and memory guards. The required
request-time or decode-throughput improvement does not: request time regresses
2.17% and throughput regresses 4.26%. The decision is
`stop_runtime_candidate_keep_default_off`. This exact split-slot schedule must
not be rerun or enabled unchanged. Reopen only for a materially different
first-stage kernel or I/O schedule that enlarges the real one-row compute window,
then predeclare a new gate.

The runtime artifact is
[`2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json`](../docs/benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json),
SHA-256
`3dd265bf3fb9a78b23d596bbf17adc4f522ad211e2317b75a87a660161198542`.
It embeds all eight token sequences, metrics, commands, and raw subprocess
stdout/stderr. It remains an exploratory single-workload result, not a formal
performance result or physical-SSD measurement.
