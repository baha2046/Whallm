# Hash-layer exact prefetch experiment

## Status

The runtime now has a default-off prototype behind
`--dspark --dspark-hash-prefetch`. This file is an active experiment plan, not
evidence that the prototype improves current performance. A second default-off
prototype, `--dspark-adaptive-block`, selects a verification prefix using exact
hash-layer residency and confidence. The original storage-only policy failed the
first five-workload calibration. The current repair keeps a complete block when
its expected commit fraction is at least 0.90 and caps target verification to the
remaining output budget; it remains under the same evidence limits.

## Verified implementation premise

For the pinned `mlx-lm` fork, each configured hash router selects routed expert
IDs with the checkpoint tensor `tid2eid[input_ids]`. The target logits still
determine expert weights, but the expert IDs for the first three main model
layers are available as soon as the draft token IDs are known.

The prototype therefore:

1. resolves routes for `[anchor, *draft_tokens]` from the loaded checkpoint
   tables;
2. preserves first-use order while deduplicating each layer's expert union;
3. temporarily pins union members already resident in the main LFU cache;
4. reads the remaining members into a fixed Metal-visible verification scratch;
5. reuses that scratch for rejected-prefix replay, then releases the pins;
6. never admits scratch-only experts into the long-lived LFU cache.

The target router and target forward still execute normally. Prefetch does not
substitute its IDs or scores into model output computation.

## Assumptions to test

- The three early-layer reads have enough deadline slack to reduce target wait.
- Avoiding LFU admission offsets the extra scratch memory and rejected-suffix
  reads.
- Logical read reduction or overlap is visible in request wall time, not only in
  cache-side counters.
- Rejected drafts do not make wasted bytes dominate bytes per committed token.

## Required test conditions

- Fixed installed model revision and recorded code commit or working-tree hash.
- Batch size 1, greedy generation, equal prompt and output token counts.
- Identical output token hash for prefetch off and on.
- Interleaved off/on execution order across the existing fixed prompt set.
- Explicit cold/warm compile, OS page-cache, expert-cache, and prompt-cache
  states for every row.
- The same DSpark confidence threshold, slots, workers, and power limit.

Record at least:

- request seconds and committed tokens per second;
- logical target, draft, prefetch, on-demand, useful, and wasted expert bytes;
- process disk read bytes;
- exact-plan resolve, prefetch read, and target wait seconds;
- cache misses, evictions, and per-layer union sizes;
- active, cache, peak, and process RSS memory;
- acceptance length, replay time, fallback, and output token hash.

## Stop criteria

Reject or pause this candidate if any of the following occurs:

- greedy output token hashes differ;
- scratch experts enter the long-lived LFU cache or survive the transaction as
  pinned entries;
- useful plus wasted bytes does not equal actual scratch bytes read;
- scratch allocation violates the configured MLX memory limit;
- a reproducible wait-time reduction does not produce end-to-end improvement;
- process disk bytes, peak memory, or request time regress enough to erase the
  committed-token benefit.

## 2026-08-26 exploratory evidence

The complete installed model at the pinned revision passed one short greedy
correctness smoke. The fixed hash-prefetch comparison used off/on/on/off order.
All four runs produced the same prompt and output token hashes and the same
five-proposed, two-accepted, three-committed outcome.

The two enabled runs had identical logical counters: 79 scratch reads, of which
47 were useful and 32 were wasted. Relative to disabled runs, target logical
reads fell by 16 expert blobs, rejected-prefix replay fell by 13 blobs, and the
whole request fell by 19 blobs. The fixed 108-slot scratch added exactly
1,443,889,152 bytes to the two-run peak-memory median.

The adaptive follow-up evaluated candidate draft lengths 1, 2, 4, and 5. It
selected one token in both decisions and preserved the output token hash. Across
the changed two-round execution it read 67 scratch experts: 49 useful and 18
wasted. Relative to fixed five-token prefetch, request logical reads fell by 249
expert blobs and target reads by 223, while replay increased by four blobs.

These are correctness and logical-byte observations. Timing is not adopted: the
first disabled run was colder than the return pair, the operating-system page
cache was not purged, and the adaptive follow-up has only one warm observation.
See the two 2026-08-26 artifacts under `docs/benchmarks/` for exact conditions and
raw selected metrics.

### Five-workload adaptive calibration and repair

The next exploratory gate used five 128-token domains, eight greedy output
tokens, one same-prompt warmup, and fixed/adaptive/adaptive/fixed fresh-process
runs. Persistent prompt cache was disabled; OS page cache was not purged.

Wave 1 preserved prompt and output hashes for all five workloads but rejected
the original policy. The median change across workloads was +2.013% request
time, -14.843% decode tokens/s, and +0.593% request logical expert bytes. No
workload reduced total request logical bytes. Code, Chinese technical, and mixed
math had almost fully utilized five-token control blocks, yet the score trimmed
them to two, one, and one token. Tool-like entered two rounds and exposed a
separate accounting bug: target verification exceeded the remaining output
budget and reported 11 committed tokens for an eight-token output.

The repair makes two changes:

1. if full-candidate expected commits divided by `draft tokens + 1` is at least
   0.90, select the full candidate instead of the storage-score winner;
2. before adaptive selection, cap eligible draft tokens at
   `remaining output tokens - 1`, and count only output-budget commits.

Wave 2 again passed parity for all five workloads. Every adaptive run selected
five tokens through the high-confidence guard, accepted all five, and committed
six. Fixed and adaptive logical counters were identical for every workload. The
median request and decode changes across workloads were +0.0495% and +0.2164%,
respectively. This establishes control-equivalent repair behavior, not speedup.

A separate seven-token low-confidence A/B/B/A retained the storage-score path:
both adaptive runs made two storage-score decisions, made no high-confidence
decision, trimmed two DSpark tokens for output budget, and reported three
committed tokens. All hashes matched. Request and target logical bytes were
11.02% and 33.09% lower than fixed, but timing is not adopted because the prompt
is short and OS page cache was uncontrolled.

See `docs/benchmarks/2026-08-26-adaptive-block-calibration-wave1-m2-max.json`,
`docs/benchmarks/2026-08-26-adaptive-block-calibration-wave2-m2-max.json`, and
`docs/benchmarks/2026-08-26-adaptive-block-repair-validation-m2-max.json`.

### 4K/256 decision survey and stop gate

The next gate regenerated the five historical R0 4,096-token prompts. All five
prompt token hashes matched the 2026-08-10 R0 artifact. One fresh-process
adaptive run per workload generated 256 greedy output tokens without persistent
prompt cache or a same-prompt warmup. Every output token hash and generated-token
count matched the historical fixed R0 baseline.

Across the five workloads, all 54 adaptive decisions used the high-confidence
full-block guard and none used the storage score. The selected-length histogram
was 53 five-token blocks and one one-token block. The one-token block was the
complete output-budget-capped final block in the Chinese workload, not adaptive
trimming. Repeated, code, mixed math, and tool-like eventually fell back; Chinese
technical completed 43 speculative rounds without fallback.

The stop condition therefore fired before a paired fixed/adaptive ABBA wave:
the R0 matrix offers no selection-policy difference to measure. A new paired
gate requires a reproducible 4K/256 low- or medium-confidence workload that uses
the storage-score path across multiple rounds. The survey artifact is
`docs/benchmarks/2026-08-26-adaptive-block-4k256-survey-m2-max.json`; its prompt
manifest is
`docs/benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json`.

### Low-confidence discovery and parity failure

Five exact 4,096-token instruction-suffix prompts were then screened with 16
greedy output tokens. Every prompt selected one token through the storage score,
and every run triggered the normal fallback after its first round. A repeated
`storage_sentence` run preserved its output hash and measured a 1.301 fallback
cost ratio. With fallback explicitly disabled as a research control, the same
prompt completed eight storage-score rounds with the same 16-token hash, while
five rounds still reported that the normal policy would have stopped.

The `random_hex` candidate completed 256 tokens and 86 forced speculative rounds:
66 decisions used the storage score and selected lengths covered 1, 2, 4, and 5.
However, its fixed and adaptive output hashes differed. A 32-token reproduction
with exact token IDs found that fixed and adaptive block verification both left
the sequential-prefill reference at index 15 and then diverged from each other
at index 17. Layer-major normal generation separately left the sequential
reference at index 6.

A diagnostic reconstructed the common cache prefix and compared two sequential
target steps with one two-token block. At the divergent second position, the
sequential top-two logit margin was 0.125, while the block top logits tied; the
maximum absolute logit delta at that position was 1.9375. This supports a
near-tie numerical-sensitivity explanation, not a semantic improvement. The
parity gate failed, all apparent performance and byte changes were invalidated,
and no unbounded logit-margin heuristic was adopted.

See `docs/benchmarks/2026-08-26-adaptive-block-discovery-4k16-m2-max.json`,
`docs/benchmarks/2026-08-26-random-hex-4k32-fixed-adaptive-failed-m2-max.json`,
and `docs/benchmarks/2026-08-26-dspark-block-parity-diagnostic-m2-max.json`.

### Sequential target verification oracle

A default-off correctness oracle now keeps one round-level cache fork but runs
each target verification position separately. Rejected committed prefixes are
also replayed one token at a time. The oracle refuses hash exact prefetch so the
speculative scratch path cannot change ready-expert execution while verifier
shape is being isolated.

On the same `random_hex` 4K/32 workload, all 13 fixed-oracle rounds and all 17
adaptive-oracle rounds used sequential verification. Both runs exactly matched
the sequential-prefill reference token IDs and hash
`e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`.
The adaptive oracle still made 16 storage-score decisions, selecting one token
15 times and two tokens once. This rules out adaptive prefix selection and
sequential replay wiring as the source of this workload's divergence and
narrows the current correctness boundary to block-shaped target verification.

The fixed and adaptive artifacts are
`docs/benchmarks/2026-08-26-random-hex-4k32-sequential-fixed-oracle-m2-max.json`
and
`docs/benchmarks/2026-08-26-random-hex-4k32-sequential-adaptive-oracle-m2-max.json`.
The oracle removes block-verification batching and is not a performance
candidate. Its timing, bytes, and memory values are not adoption evidence.

### Layer-wise block verifier diagnosis

The 2026-08-27 follow-up used the same exact common cache prefix, anchor index
13, and two target tokens. It captured post-attention, FFN input, selected
router experts and weights, and post-layer states for all 43 main-model layers.
The diagnostic first ran both verifiers without instrumentation and required
the captured runs to reproduce both reference logit arrays exactly. It captured
86 sequential layer positions and 43 block layers without changing either
verifier's logits.

Both broadcast input embeddings were exact. The first non-exact boundary was
immediately after layer 0 `LocalAttention`: position 0 changed 112 of 16,384
elements with a maximum absolute delta of 0.0009765625; position 1 changed 46
elements with a maximum delta of 0.00048828125. Layer 0 selected expert IDs,
order, and weights were still exact. Position 1 first changed router order at
layer 11 and its selected set at layer 12; position 0 first changed both at
layer 16. By layer 42, maximum post-layer deltas were 3.0 and 3.75, and the
known second-position top-token mismatch was reproduced.

This narrows the earliest observed boundary to the layer 0 attention branch.
Learned routing later amplifies the hidden-state difference but is not its
origin in this reproduction. The evidence does not yet identify one projection,
attention kernel, or output operation as solely causal. The next correctness
experiment must isolate those layer 0 sub-boundaries; selector tuning and
performance measurement remain stopped. The artifact is
`docs/benchmarks/2026-08-27-dspark-layer-parity-diagnostic-m2-max.json`.

### Layer 0 attention component diagnosis

The component follow-up reused the exact prefix, anchor index, and two target
tokens. It required two sequential layer 0 captures and one block capture, and
the captured full-model logits reproduced both uninstrumented references
exactly.

For both positions, input hidden state, the collapsed HyperConnection value,
attention norm, `wq_a`, `q_norm`, `wkv`, KV norm, and new-key RoPE were exact.
The first non-exact stage under the strict recorded order was HyperConnection
`post`, with maximum deltas of 6.821210263296962e-12 and
2.9558577807620168e-12; `combine` also differed. Independently, `wq_b` differed
despite exact `q_norm` input. The KV projection path remained exact.

Sequential execution performed two one-token in-place cache updates with no
mask. Block execution performed one two-token concatenate update with a 2x129
mask. The raw fetched caches had lengths 128 and 129 and were therefore not
directly comparable, but all 65,536 values in their common 128-token
temporal-order suffix were exact. Attention and output projection then
reproduced the prior post-attention and final-logit differences.

This excludes the layer input, collapsed attention value, attention norm, KV
projection, and common temporal cache suffix as the earliest boundary in this
reproduction. It does not assign independent causality because HyperConnection,
Q projection, mask/cache layout, and SDPA execution shapes differ together.
The next experiment is a verifier with token-sequential attention and
block-shaped FFN/MoE union. Selector tuning and performance measurement remain
stopped. The artifact is
`docs/benchmarks/2026-08-27-dspark-layer0-attention-component-diagnostic-m2-max.json`.

### Hybrid verifier follow-up

The first hybrid kept one-token attention but retained block-shaped FFN/MoE.
At the exact two-token diagnostic state it restored both sequential top token
IDs, including near-tie token 20400, while full logits remained non-exact. A
4K/32 multi-round run then left the normal reference at output index 15 and
had 17 mismatches. This stopped v1 before multi-workload adoption.

The layer 0 FFN component diagnostic found exact normalized input, router IDs,
router scores, routed selected outputs, routed reduction, and MoE output. The
first strict non-exact stage was FFN HyperConnection `post`; `combine` also
differed. One shared-expert output value differed per position, but target-dtype
addition made the recorded MoE output exact. Hybrid v2 therefore executed FFN
HyperConnection and final expand tokenwise while keeping block MoE. It passed
three of five 4K workloads but `storage_sentence` and `multilingual_choice`
first differed at output indices 16 and 6.

Hybrid v3 kept attention, FFN HyperConnection, router, shared-expert, routed-
expert, and final-expand math in one-token shapes. It evaluated all token routes
first and acquired the complete layer expert union once, then reused those
weights for token-shaped routed expert math. This preserves union I/O dedupe but
does not preserve grouped multi-row QMM. Five 4K normal/fixed comparisons were
exact: four generated 32 tokens and `balanced_choice` reached EOS at 5. Fixed
union assignment reuse ranged from 37.08% to 46.90%.

Artifacts are:

- `docs/benchmarks/2026-08-27-dspark-hybrid-verifier-diagnostic-m2-max.json`
- `docs/benchmarks/2026-08-27-dspark-hybrid-random-hex-4k32-m2-max.json`
- `docs/benchmarks/2026-08-27-dspark-layer0-ffn-component-diagnostic-m2-max.json`
- `docs/benchmarks/2026-08-27-dspark-hybrid-v2-discovery-4k32-m2-max.json`
- `docs/benchmarks/2026-08-27-dspark-hybrid-v3-discovery-4k32-m2-max.json`

### Hybrid, exact hash prefetch, and adaptive composition

Hash exact prefetch was then composed with hybrid v3. All five normal/fixed
token sequences remained exact, and every candidate run satisfied
`useful_bytes + wasted_bytes = hash_prefetch_bytes_read`. Fixed useful rates
ranged from 23.53% to 56.23%.

The adaptive follow-up reused the same normal and fixed raw runs and added one
adaptive run per workload with current source hashes. Normal, fixed, and
adaptive token sequences were exact for all five workloads. Across 63 adaptive
decisions, selected lengths 1, 2, 3, 4, and 5 occurred 56, 4, 1, 1, and 1 times.
Hash-prefetch useful rates rose to 65.22%-91.31%, while expert-union assignment
reuse fell to 15.59%-23.98%. Shortening the verified prefix therefore reduced
rejected-only exact prefetch in this survey but also removed expert reuse across
positions.

The composition artifacts are:

- `docs/benchmarks/2026-08-27-dspark-hybrid-v3-hash-discovery-4k32-m2-max.json`
- `docs/benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json`

These surveys used fresh processes, disabled persistent prompt cache, disabled
layer-major prefill, ran without warmup, did not purge the OS page cache, and
disabled normal fallback only to complete correctness traces. Each mode has one
run. They are correctness, selector-behavior, and logical-byte evidence, not
performance, physical-I/O, memory, or power adoption evidence.

## Evidence limits

Fixture tests cover exact table lookup, first-use union order, committed-prefix
byte classification, scratch reuse, absence of LFU admission, adaptive
candidate scoring, high-confidence full-block selection, output-budget capping,
prefix truncation, actual committed-token accounting, fallback diagnostics,
sequential verification/replay, and fixture greedy parity. The full-model
evidence now includes one short smoke,
two five-workload short ABBA waves, one low-confidence repair A/B/B/A, one
five-workload 4K/256 decision survey, one five-prompt low-confidence screen, and
one failed long-decode parity reproduction, plus two 4K/32 sequential-oracle
runs, one two-token 43-layer parity diagnosis, two layer 0 component diagnoses,
three hybrid verifier stages, one five-workload hybrid v3 gate, and five-workload
hash/adaptive composition gates. The R0 survey establishes
historical greedy parity and selector behavior,
but the adversarial low-confidence workload shows that this evidence does not
generalize to block-shape-sensitive near ties. The oracle restores exact parity
for that single workload without retaining block batching. Hybrid v3 restores
five-workload greedy parity while retaining one union acquisition per layer, but
it uses token-shaped expert math rather than grouped multi-row QMM. The layer and
component diagnoses narrow the observed boundary but do not prove one causal
kernel or a general tolerance. There is still no valid
4K/256 adaptive throughput, physical expert-file SSD traffic, memory adoption,
or power result. The adaptive score only covers the three exact hash layers;
it does not model the 40 learned-router layers, hidden I/O, GPU idle time, or
memory pressure.
DSpark still computes all five draft positions before runtime selection and
output-budget capping.
