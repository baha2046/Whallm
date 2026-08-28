# DSpark-hidden to learned-router predictor baseline protocol

Status: completed; structural gate passed and every continuation candidate
failed. Direct frozen-router transfer is stopped. No prefetch runtime option or
default changed.

## Question

Main-model layers 0--2 have exact token-ID hash routes. Layers 3--42 use learned
routers, so the current DSpark hash prefetch cannot know their expert sets
before target verification. This experiment asks whether already-computed
DSpark hidden features have enough alignment with frozen target routers to
justify a trained predictor or a later scratch-prefetch prototype.

Accuracy alone is not the decision metric. The gate reports both route recall
and the exact expert-blob union that a predictor would request, use, waste, or
miss.

## Fixed labels and alignment

Use exactly the five 128-token prompts in
[`2026-08-26-adaptive-128`](../docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json):
`repeated`, `code`, `zh_technical`, `mixed_math`, and `tool_like`. Each workload
runs in a fresh process and captures only its first complete five-token DSpark
draft.

The sequential target verifier processes `[anchor, t_1, ..., t_5]`. The primary
alignment maps DSpark position `i` to the learned-router labels produced when
the target processes emitted draft token `t_i`; target anchor routes are
excluded. This produces exactly:

```text
5 workloads * 5 positions * 40 learned layers * 6 experts = 6,000 labels.
```

The exclusion is explicit: any future full-round byte estimate must add the
unpredicted anchor routes. Labels come from one-token-at-a-time target calls on
a round-level prompt-cache fork. Grouped verification is not a label oracle
because this checkout has known shape-sensitive route differences.

All label inputs are the actual current DSpark baseline tokens, regardless of
whether a later target would accept them. This matches the initial speculative
verification workload: the target evaluates the draft inputs before it knows
the accepted prefix.

## No-training feature candidates

Tap the hidden tensor after each of the three DSpark backbone layers and
collapse its four HyperConnection streams with the existing DSpark
`hc_head`. Test four fixed sources:

1. `dspark_layer_0_head`;
2. `dspark_layer_1_head`;
3. `dspark_layer_2_head`;
4. `dspark_final_norm`, the layer-2 head after the existing DSpark output norm.

For target learned layer `l`, apply that layer's frozen `ffn_norm`, router
weight, configured score function, and correction bias to the five aligned
features. Rank all 256 experts stably by corrected routing score. No target
route, target hidden state, target outcome, fitted projection, or cross-workload
statistic may enter the prediction.

This direct-transfer baseline is intentionally cheap to define, not assumed to
be well calibrated. A failure does not rule out a trained reducer or router
head.

## Reported candidate sizes

Evaluate stable predicted top-k sets for `k = 6, 12, 24, 48`. For every feature
source, k, workload, and learned layer, record:

- target assignments and assignments recovered by predicted top-k;
- fraction of token/layer positions whose complete target top-6 is recovered;
- target expert union, predicted union, useful intersection, wasted predicted
  experts, and missed target experts;
- corresponding logical expert-blob bytes using the installed manifest;
- per-layer and per-workload recall.

Definitions are:

```text
assignment recall = recovered target assignments / target assignments
union recall      = useful predicted union / target union
useful rate       = useful predicted union / predicted union
wasted bytes      = (predicted union - target union) * expert_blob_size
missed bytes      = (target union - predicted union) * expert_blob_size
```

These are logical label bytes, not executed reads, OS page residency, or
physical SSD traffic. Duplicated expert IDs within a five-position layer union
count once for byte accounting.

## Configuration and validity

Use greedy DSpark (`temperature=0`, `top_p=1`) with confidence truncation off,
normal cached descriptors, 1,152 main slots, 768 DSpark slots, and persistent
prompt cache, layer-major prefill, hash prefetch, adaptive block, fallback, and
page-cache probing off. Timing is not interpreted.

The artifact is valid only when:

1. all prompt bytes, token counts, and token hashes match the manifest;
2. five fresh workers complete with a full five-token draft;
3. reconstructed greedy tokens from the tapped final feature match the actual
   `DSparkModel.draft` tokens;
4. every target layer has exactly six valid labels for each of five aligned
   positions, and only layers 3--42 are scored;
5. every prediction is a unique, valid, stable top-k subset;
6. all assignment and union identities close for every feature/k row.

Any structural failure stops interpretation.

## Continuation and stop gate

A frozen-router candidate is eligible only for `k <= 24` and only when all of
the following hold after aggregating all five workloads:

1. assignment recall is at least 80%;
2. every workload assignment recall is at least 70%;
3. at least 30 of 40 learned layers have assignment recall at least 75%;
4. union recall is at least 80%; and
5. useful predicted-union rate is at least 50%.

If one or more candidates pass, choose the smallest k, then highest useful
rate, then highest assignment recall, then earliest feature source in the fixed
list. That winner remains default off and only qualifies for an isolated
scratch-prefetch runtime gate with exact target output, useful/wasted executed
bytes, target on-demand misses, eviction isolation, peak memory, total request
time, and p95 guards.

If none pass, stop direct frozen-router transfer. Keep the fixed dataset as the
baseline for a separately trained reducer/predictor; do not hide low recall by
prefetching more than 24 experts per token. Training requires a new protocol,
data-rights statement, train/validation split, held-out workloads, and model
artifact hash.

## Evidence limits

This first-round dataset does not cover low-confidence drafts, rejected suffix
replay, multiple speculative rounds, long context, learned predictor training,
actual I/O overlap, cache admission, queue depth, GPU idle time, physical SSD
traffic, energy, ANE execution, or serving concurrency. It predicts only the
five emitted draft-token inputs and omits the anchor. A passing offline gate
would not establish end-to-end speed; a failing direct-transfer baseline would
not establish that DSpark hidden states contain no trainable route signal.

## Result

All five fresh workers passed every structural criterion. Prompt contracts,
full drafts, reconstructed greedy tokens, six-position sequential route
captures, learned-layer range 3--42, four-feature prediction matrices, and all
assignment/union accounting identities are exact. The artifact contains 6,000
scored target assignments plus the separately captured, unscored anchor routes.

The best candidate at each k is the final DSpark output-normalized feature:

| Predicted k | Assignment recall | Union recall | Useful predicted-union rate | Full target sets recovered |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 6.27% | 12.18% | 13.50% | 0% |
| 12 | 10.13% | 20.56% | 12.40% | 0% |
| 24 | 17.15% | 33.03% | 11.15% | 0% |
| 48 | 28.93% | 51.26% | 10.03% | 0% |

For the best continuation-eligible size, `dspark_final_norm` top-24, per-
workload assignment recall is 19.33% (`repeated`), 14.50% (`code`), 22.42%
(`zh_technical`), 12.42% (`mixed_math`), and 17.08% (`tool_like`). No learned
layer reaches the required 75% aggregate recall; the best layer is layer 40 at
41.33%. The other three feature sources are worse at top-24, with aggregate
assignment recall between 11.00% and 15.70%.

No candidate approaches the 80% assignment/union, 70% per-workload, 30-layer,
or 50% useful-rate gates. Allowing top-48 does not rescue the design and is
ineligible by protocol: it requests a much larger union while roughly 90% of
that predicted union is still wasted relative to the target labels.

The machine-readable artifact is
[`2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json)
with SHA-256
`4eb2574b7a9e6b613784288833a3d6ff327e71af145e6d822c2b501d387f7739`.
The benchmark script SHA-256 at the run was
`0ffc0c54b2629b3694c51de53232a55dd4c246c171a892b5255585267dbeec5b`;
the predeclared protocol SHA-256 captured in the artifact was
`92c5381f71601654cfba5a201f2e01a78fb92c169ddc8028ea0be951d12c46e6`.

## Decision

The decision is `stop_direct_frozen_router_transfer_training_required`. Do not
build learned-layer scratch prefetch, cache probation, deadline pinning, or a
runtime setting from these predictions. The frozen target routers do not
directly interpret the tested DSpark features with sufficient recall or byte
precision.

The fixed labels remain a reproducible baseline, not a training dataset large
enough for adoption. Reopening this direction requires a new protocol that
defines data rights, substantially more route traces, train/validation and
held-out workload splits, a fitted reducer/router head, and immutable model and
dataset hashes. This negative direct-transfer result does not prove that a
trained predictor cannot recover useful signal.
