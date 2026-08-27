# Verifier expert-union tradeoff protocol

Status: completed diagnostic. The one-per-layer hybrid union contract passed,
but its aggregate token-shaped execution cost was material. No verifier default
changed.

## Question

Hybrid v3 recovered exact target tokens by keeping attention, FFN
HyperConnection, router, shared expert, and routed expert math in one-token
shapes. It still collects all routes for a layer and acquires the expert union
once. This preserves acquisition dedupe but gives up the original grouped
multi-row expert QMM.

This gate quantifies three target-verification implementations on the same
accepted five-token draft block:

- `sequential`: token-shaped target math and one expert acquisition per token
  per layer; correctness oracle;
- `grouped`: one block-shaped target call and one expert-union acquisition per
  layer; known to fail exact parity on some low-margin workloads;
- `hybrid`: token-shaped target math and one expert-union acquisition per
  layer; current exact candidate.

Normal autoregressive generation is included only as an output and end-to-end
reference. No result from this short diagnostic can enable DSpark by default.

## Fixed workload and cache policy

Use the 128-token `repeated` prompt from
[`2026-08-26-adaptive-128`](../docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json)
and generate eight greedy tokens. The existing current-checkout reference token
SHA-256 is
`03f40e52aa3a6f874badbf2c339e67b1f8adba4177067e9ef36e7a6d99e289f3`.
The high-confidence case previously produced one DSpark round with five
proposed and accepted draft tokens, six committed tokens, and no replay.

Every run uses a fresh process, persistent prompt cache off, layer-major
prefill off, hash prefetch off, adaptive selection off, fallback off, and the
validated `expert_file_cache_policy=bypass`. The installed-range contract must
have `contract_passed=true` before this benchmark starts.

## Repetition and order

The uninstrumented performance sequence is the four-mode Latin square:

```text
wave 1: normal, sequential, grouped, hybrid
wave 2: sequential, grouped, hybrid, normal
wave 3: grouped, hybrid, normal, sequential
wave 4: hybrid, normal, sequential, grouped
```

One final sequential／grouped／hybrid observer wave enables the page-residency
probe. Observer timing is excluded from medians.

## Correctness and structural gate

The diagnostic is valid only if:

1. all 19 output token-ID sequences match the pinned SHA-256;
2. every performance mode has four successful runs;
3. every DSpark performance and observer run has exactly one round, five
   proposed tokens, five accepted tokens, six committed tokens, and zero replay
   bytes;
4. sequential reports one sequential round, 258 expert-union acquisition calls,
   and 1,548 routed assignments;
5. grouped reports one block round, 43 union calls, and 1,548 assignments;
6. hybrid reports one hybrid round, 43 union calls, 1,548 assignments, 43
   attention layers, 258 attention token calls, 258 FFN token calls, and 258 MoE
   token calls;
7. every observer run has zero probe failures/unclassified bytes and closes the
   main and DSpark draft logical resident/nonresident partitions;
8. cache policy and 4,096-byte alignment labels are exact.

The call counts come from 43 target layers, six verification positions
(`anchor + five drafts`), and six routed experts per position.

## Interpretation and stop criteria

Report medians for request time, decode throughput, verification time, union
calls, assignments, acquired union experts, misses, target expert bytes, target
expert read time, and peak memory.

The one-per-layer union contract passes when hybrid uses exactly 43 calls,
preserves all 1,548 assignments, records nonzero reused assignments, and does
not read more target expert bytes than sequential by over 5%. Grouped and
hybrid union-expert counts are compared descriptively because block-shaped
floating-point differences may change routes even when this short output stays
token-exact.

Treat the aggregate grouped-execution advantage as material for this case when
hybrid median target verification time is at least 20% above grouped. Treat
acquisition dedupe as insufficient to overcome token-shaped math when both the
union contract passes and the 20% grouped-time condition holds. The timing
cannot isolate grouped QMM alone: grouped and hybrid also differ in attention,
HyperConnection, router, shared-expert, and routed-expert execution shapes.

Any token, shape, mode, or accounting failure stops interpretation. Even a
fully passing diagnostic does not override the prior 4K/32 end-to-end
rejection, the grouped verifier's multi-workload correctness rejection, or the
default-off settings.

## Instrumentation result

Runtime metrics now expose
`dspark_verification_expert_union_calls` in addition to assignments, acquired
union experts, reused assignments, misses, bytes, and read time. The counter is
the exact number of captured `get_many` acquisitions across all target
verification and replay calls. Tests cover accumulation and the predeclared
258／43／43 mode contracts.

The benchmark helper also accepts an explicit hash-prefetch override so this
gate can keep hash prefetch off for grouped mode instead of inheriting the older
fixed-mode default.

## Attribution-label pilot

The first complete 19-run artifact used the result label
`grouped_qmm_loss_material`. The measurements were valid, but that label was
too narrow: grouped and hybrid differ in every target execution shape, not only
routed QMM. It is retained for traceability at
[`2026-08-27-verifier-union-tradeoff-attribution-label-pilot-m2-max.json`](../docs/benchmarks/2026-08-27-verifier-union-tradeoff-attribution-label-pilot-m2-max.json)
with SHA-256
`1f1ef076b960ff87eb43a313bb74e29cd1b918d3f20727e150a8c70c8e52a700`.
No pilot timing is used below.

The label and evidence limit were corrected to aggregate token-shaped
execution cost, then all 19 runs were repeated from a new raw directory. No old
metrics were resumed.

## Corrected result

All 19 output sequences match the pinned token SHA-256. All four performance
runs per mode and all three observer runs pass the cache-policy, verifier-shape,
and accounting gates. Every DSpark run has one round, five proposed and accepted
tokens, six committed tokens, and zero replay bytes.

| Mode | Union calls | Assignments | Acquired union experts | Target bytes | Verification median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential | 258 | 1,548 | 1,548 | 2,607,022,080 | 0.8524 s |
| Grouped | 43 | 1,548 | 660 | 1,951,924,224 | 0.5331 s |
| Hybrid v3 | 43 | 1,548 | 675 | 2,286,157,824 | 1.0001 s |

Hybrid reduces acquisition calls 83.33%, acquired-union accounting 56.40%, and
target bytes 12.31% versus sequential. It nevertheless increases target
verification time 17.34% versus sequential and 87.59% versus grouped. The
one-per-layer union contract therefore passes, while acquisition dedupe is
insufficient to overcome the aggregate token-shaped execution cost in this
case.

Grouped and hybrid do not have identical union counts despite identical output
tokens: they acquire 660 versus 675 experts and miss 146 versus 171. This is
consistent with shape-sensitive internal routes and is another reason not to
interpret the timing gap as a QMM-only measurement.

The corrected artifact is
[`2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json`](../docs/benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json)
with SHA-256
`9d6a12028b85bdf492e82292a651f5a203c30d4fedd29ab1255dd52a4c2073a8`.

## Decision

The P0 union profiler and current hybrid one-per-layer acquisition contract are
confirmed. The current hybrid verifier remains a correctness implementation,
not a performance candidate. The grouped verifier remains stopped because it
is not exact on the existing low-margin multi-workload gate, even though this
high-confidence short case is exact and faster.

Do not rerun this exact comparison. A next verifier candidate must preserve
token-shaped equivalence while recovering more grouped execution, then pass the
five-workload correctness gate before any broader speed matrix.
