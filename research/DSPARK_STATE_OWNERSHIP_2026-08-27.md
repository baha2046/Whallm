# DSpark state ownership and reduced-cache pilot

Status: ownership audit complete; 96-slot reduced independent-cache candidate
passes a three-workload 4K/32 correctness and memory-direction screen but fails
the predeclared 4K/128 draft-read gate. The candidate is stopped and remains
default off.

## Scope and equivalence

This investigation covers PLAN P2 integration: remove duplicate DSpark prompt
work and reduce its independent large cache without changing target-model
output. It does not change checkpoint tensors, routing, or target verification.

The first candidate changes only DSpark expert-cache capacity from 768 to 96
slots. A cache miss still reads the canonical installed expert blob, so the
candidate is intended to be exact. The stop rule is any generated-token
difference, inability to hold one complete draft working set, higher peak
memory, or an expert-read/request-time penalty too large to justify broader
validation.

## Current ownership map

| State | Owner and lifetime | Reusable boundary | Current duplication or constraint |
| --- | --- | --- | --- |
| Main common tensors | main model, process lifetime | shared by all requests | DSpark has a separate 595,185,408-byte `common.bin`; only embedding and final target head are reused from main model. |
| Main routed experts | target `ExpertCache`, 1,152 slots, layers 0–42 | process lifetime LFU residency | keys are `(target layer, expert)` and files are under `experts/`. |
| DSpark routed experts | DSpark `ExpertCache`, 768 slots by default, layers 0–2 | process lifetime LFU residency | files are under `dspark/experts/`; numeric layer/expert keys overlap main keys but identify different weights, so the two caches cannot share the current un-namespaced key space. |
| Main prompt KV | one target prompt cache per request | normal mode supports memory and persistent prefix reuse | DSpark mode always allocates a fresh prompt cache and bypasses `_acquire_prompt_cache`. |
| DSpark attention context | three `RotatingKVCache` instances | reused across draft rounds in one request | reset at every DSpark request; each stage projects captured target hidden state through DSpark-specific `main_proj`/attention KV weights. It is not derivable from target KV alone. |
| Target hidden taps | target layers 40–42 during `forward_with_hidden` | transient within prefill/decode | normal prompt-cache entries do not retain these hiddens, so restoring only target KV cannot reconstruct DSpark context. |
| Verification fork | target prompt-cache copy per speculative round | committed directly on full acceptance | already shared by the exact block/sequential/hybrid verifier paths; not DSpark draft state. |

This map rules out a direct `ExpertCache` object share: `(0, expert)` means a
main-model layer-0 blob in one directory and a DSpark stage-0 blob in another.
A future shared physical slot pool needs a model namespace, separate residency
policy, and ownership/fence tests. It also rules out enabling ordinary prompt
cache reuse without storing a matching DSpark context snapshot or the target
hidden taps needed to rebuild it.

## Why 96 slots

The complete DSpark graph has three stages, block size 5, and six selected
experts per token. The maximum routed assignments in one draft transaction is:

```text
3 stages * 5 positions * 6 experts = 90 assignments
```

Ninety is an assignment upper bound; duplicate routes can make the unique
working set smaller. The 96-slot candidate adds six slots of slack and reduces
hard payload capacity from 10,267,656,192 to 1,283,457,024 bytes, an 87.5%
capacity reduction. It does not promise an 8.367 GiB peak-memory reduction,
because the lazy slot pool allocates only slots actually touched.

## Full-model pilot

[`Scripts/benchmark_dspark_slots.py`](../Scripts/benchmark_dspark_slots.py)
runs a fresh-process normal reference followed by 768/96/96/768 slots. The
workload is the fixed 4,096-token `storage_sentence` prompt with 32 greedy output
tokens. Persistent prompt cache and layer-major prefill are disabled. Both
DSpark modes use hybrid v3 target verification, confidence threshold zero, no
hash/adaptive prototype, and research-only no-fallback so the full trace
completes.

The result is
[`2026-08-27-dspark-slots-768-vs-96-storage-4k32-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-slots-768-vs-96-storage-4k32-m2-max.json).

| Metric, median of two DSpark runs | 768 slots | 96 slots | Candidate change |
| --- | ---: | ---: | ---: |
| Generated-token parity | exact | exact | all five runs exact |
| Peak MLX memory | 28,601,817,360 B | 27,653,809,943 B | -3.31% (-0.883 GiB) |
| Resident DSpark experts | 213 | 96 | -54.93% |
| Draft expert reads | 2,847,670,272 B | 3,623,092,224 B | +27.23% (+0.722 GiB) |
| Draft cache hit rate | 61.55% | 51.08% | -17.01% relative |
| Draft misses | 213 | 271 | +27.23% |
| Draft evictions | 0 | 175 | new thrashing boundary |
| All speculative expert reads | 150,659,137,536 B | 151,434,559,488 B | +0.515% |
| Request time | 100.748 s | 100.935 s | +0.186% |

Both request-time pairs have opposite direction (+0.494%, -0.120%), and the OS
page cache was not controlled. Timing and process disk counters are not adoption
evidence. The exact 32-token hash
`1483ad586d94ed4029518c36095d51780e8da2637e06f05d63e290bdc6ea9b7e`
also matches the earlier hybrid v3 `storage_sentence` normal/fixed artifact.

## Decision and next gate

The 96-slot candidate passes the initial one-workload correctness and
memory-direction gate, so it was expanded to `multilingual_choice` and
`random_hex`. The three-workload composition is
[`2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-slots-768-vs-96-three-workload-summary-m2-max.json).

| 4K/32 workload | Exact and matches hybrid v3 | Peak-memory change | Draft-read change | All speculative-read change | 96-slot evictions |
| --- | --- | ---: | ---: | ---: | ---: |
| `storage_sentence` | yes | -3.31% | +27.23% | +0.515% | 175 |
| `multilingual_choice` | yes | -2.44% | +21.11% | +0.473% | 145 |
| `random_hex` | yes | -0.40% | +16.56% | +0.216% | 80 |

Every run preserves exact token IDs and all three hashes match the earlier
hybrid v3 references. Peak memory is lower in all three workloads, but the
benefit is not stable in magnitude and every candidate run trades memory for
more draft reads. Only `storage_sentence` has an ABBA pair; the other two are
single control/candidate pairs. Request-time changes from -0.176% to +0.193%
are screening observations, not timing evidence.

Before running the next experiment, the 4K/128 `random_hex` long-decode gate was
fixed as follows. It used fresh normal, 768-slot control, and 96-slot candidate
processes; greedy decoding; hybrid v3; fallback disabled; and the same installed
checkpoint. The candidate stops if any condition fails:

1. all three generated token-ID sequences are exact;
2. candidate peak memory is at least 1% below control;
3. draft expert bytes per committed token increase by no more than 50%;
4. all speculative expert bytes per committed token increase by no more than 2%;
5. request time increases by no more than 5%.

The three runs each produce the same 128-token hash,
`aa163ef3caef62b9e177d651ccceb1c05d80df9aad0183a4df09e57a387b03b9`.

| 4K/128 metric | 768 slots | 96 slots | Candidate change | Gate |
| --- | ---: | ---: | ---: | --- |
| Peak MLX memory | 29,885,480,224 B | 27,688,162,682 B | -7.35% | pass, at least -1% |
| Draft expert bytes/committed token | 32,786,724.57 B | 60,055,942.10 B | +83.17% | **fail**, at most +50% |
| All speculative bytes/committed token | 2,987,093,430.86 B | 3,014,362,648.38 B | +0.91% | pass, at most +2% |
| Request time | 132.717 s | 132.530 s | -0.14% | pass as screening only |
| Draft misses / evictions | 309 / 0 | 566 / 470 | +83.17% / new churn | descriptive |

The machine-readable result is
[`2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json).
Because the draft-read condition fails, the decision is to stop the 96-slot
candidate and keep 768 slots. The timing threshold was only a gross regression
screen because one pair and an uncontrolled OS page cache cannot establish
speed. The next P2 integration experiment must address a different state
boundary: an atomic target+DSpark prompt-context snapshot or a namespaced shared
physical slot pool, with ownership/lifetime tests first.

## Evidence limits

These are exploratory runs, not a formal performance result. The OS page cache
was not purged, fallback was deliberately disabled, and process disk bytes are
not expert-file physical I/O. The results support exactness and a local
memory/read tradeoff only under the recorded conditions.
