# Atomic DSpark prompt-context snapshot protocol

Status: implemented and passed the predeclared P2 functional integration gate.
The option remains default-off.

## Question

Before this candidate, DSpark mode bypassed the normal prompt cache and repeated
the complete target prefill on every request. Target KV alone is insufficient: the three
DSpark attention caches are derived from target hidden taps at layers 40–42.

This gate asks whether one namespaced snapshot can atomically preserve:

1. target prompt KV for `prompt_tokens[:-1]`;
2. all three DSpark attention-context cache states for the same token prefix;
3. token IDs, checkpoint revision, target-layer IDs, and cache format.

Restoring only one side is forbidden. This candidate does not share target and
DSpark expert weights or physical slots.

## Runtime contract

Add a default-off `dspark_prompt_cache` option which requires DSpark. When off,
runtime behavior is unchanged. When on:

- use a separate in-memory DSpark prompt-cache namespace;
- match only a strict shorter token prefix, checkpoint revision, and target
  layer tuple;
- deep-copy target cache and DSpark context state together;
- restore both before processing the uncached suffix;
- snapshot only after target and DSpark context have both consumed the same
  prefix;
- never admit normal target-only prompt entries into the DSpark namespace;
- report reuse source as `none`, `memory`, or `persistent`.

If general persistent prompt cache is enabled, write a separate atomic bundle
with `mode=dspark`. Metadata and safetensors must be written through temporary
paths and renamed only after both target and DSpark state serialize. Normal
entries (format 1/2 when this gate ran; format 4 currently) must ignore DSpark
bundles; DSpark scan must ignore normal
entries, wrong revisions, wrong target layers, incomplete files, and malformed
state.

The App and server default remain off. The status API and CLI JSON must expose
the option and last reuse source.

## Fixture and lifetime gate

Before full-model measurement, tests must prove:

1. exact-prefix in-memory acquisition returns an independent target/context
   clone;
2. suffix updates do not mutate the stored snapshot;
3. mismatched token prefix, revision, target layers, or missing context produces
   no hit;
4. normal and DSpark namespaces cannot cross-load;
5. persistent bundle round-trips across a new runtime instance;
6. malformed, partial, or wrong-contract bundles are ignored without exposing
   half-restored state;
7. eviction removes target and context state as one entry;
8. closing a runtime releases the in-memory DSpark entries.

Any failed lifetime test stops the full-model gate.

## Predeclared full-model gate

Use the installed model, 128-token `repeated` prompt, eight greedy outputs,
hybrid v3 verification, fallback off, hash/adaptive off, persistent prompt cache
on, and the validated expert-file bypass policy. Start with a new isolated cache
directory.

Run:

1. runtime A, first request: expected source `none`, zero reused tokens;
2. runtime A, second request: expected source `memory`, 127 reused tokens;
3. close A; runtime B, same request: expected source `persistent`, 127 reused
   tokens.

All three output token-ID sequences must equal pinned SHA-256
`03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3`.
The saved metadata must contain the exact revision, target layers, tokens,
`mode=dspark`, and matching data file.

For both reuse requests, main and DSpark draft logical expert bytes must not
exceed the first request, and their combined logical expert bytes must fall by
at least 50%. Request time and TTFT are recorded descriptively; this three-run
functional gate is not an adoption-quality performance result.

## Decision boundary

Passing proves only exact atomic prompt-context reuse for this checkpoint,
prefix, process lifetime, and restart boundary. It does not make DSpark faster
than normal generation, enable DSpark by default, share expert caches, or prove
concurrent-request safety. Failure keeps the existing full prefill path.

## Implementation result

The runtime now exposes `dspark_prompt_cache=false`. Enabling it requires
DSpark. The implementation keeps normal prompt entries and DSpark format-3
bundles in separate memory and persistent namespaces. Normal persistence has
since moved to format 4 without changing this boundary. A format-3 entry
contains the target cache, all three DSpark context states, token prefix,
checkpoint revision, and target layers. Acquisition accepts only a strict
shorter token prefix and returns an evaluated independent clone.

The snapshot callback runs after the uncached target suffix and all three
DSpark contexts have consumed the same `prompt[:-1]` prefix. It deep-copies and
evaluates both sides before admitting one entry. Persistent data and metadata
are written to temporary files; data is renamed first and metadata last, so a
scanner cannot discover a metadata-complete half bundle. Mode-specific pruning
cannot evict the other namespace.

Eight DSpark prompt-cache tests cover independent cloning, mutation isolation,
prefix/revision/layer/context rejection, namespace separation, restart
round-trip, malformed and partial bundles, atomic eviction, close lifetime,
reuse-source metrics, and resumed-prefill alignment. Two benchmark decision
tests separately cover the byte and contract gate.

## Installed-model result

The full gate used the pinned 128-token `repeated` prompt and produced the exact
expected output token SHA-256 in all three requests. Observed state was:

| Request | Source | Reused tokens | Main logical expert bytes | Draft logical expert bytes | Combined logical expert bytes | Request | TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Runtime A first | `none` | 0 | 28,957,999,104 | 454,557,696 | 29,412,556,800 | 6.979 s | 5.750 s |
| Runtime A second | `memory` | 127 | 2,286,157,824 | 0 | 2,286,157,824 | 1.214 s | 0.117 s |
| Runtime B after restart | `persistent` | 127 | 9,666,035,712 | 454,557,696 | 10,120,593,408 | 2.814 s | 0.751 s |

Combined logical expert bytes fell 92.23% for the memory hit and 65.59% for the
persistent hit. Each main and draft component was no larger than the first
request. The persistent data file was 12,985,677 bytes; metadata was 850 bytes.
Its mode, format, revision, layers, 127 tokens, and data-file existence all
matched exactly. The artifact SHA-256 is
`2694fb9bf1f77b15a2547babaa961854c3ec487528ff21013ebf0e964e350a25`.

Artifact:
[`2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json).

## Decision

The atomic prompt-context reuse boundary is functionally validated for this
checkpoint, exact prefix, process lifetime, and restart boundary. The candidate
remains opt-in because this is a three-request functional gate rather than a
balanced repeated performance experiment. DSpark, hybrid verification, cache
reuse, and expert-file bypass all remain default-off; the independent 768-slot
DSpark expert cache is unchanged.
