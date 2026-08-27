# Adaptive full-layer versus selective expert prefill

Status: research-complete for the tested route-union gate and post-attention
selective-read schedule. The runtime candidate is exact but stopped on TTFT;
the full-layer default is unchanged.

## Question

Layer-major prefill currently reads all 256 routed expert blobs for each of 42
expert layers into a full fixed Metal-visible buffer, then uses batched
`gather_qmm`. This preserves sequential I/O and overlaps the next layer's read
with current-layer work, but it also reads experts that receive no prompt rows.

This gate asks whether actual 4K route unions leave enough unused expert blobs
to justify a target-equivalent selective-read prototype at the PLAN's proposed
70%, 80%, and 90% thresholds.

## Phase 1: route-union eligibility

Use the pinned installed model and all five exact 4,096-token prompts from
`docs/benchmarks/prompts/2026-08-26-adaptive-4096/manifest.json`. For each
workload run two fresh processes with two greedy output tokens:

1. reference: current layer-major, batched full-layer prefill;
2. trace: the same path plus the existing exact route recorder.

Disable persistent prompt cache and DSpark. Use the explicit expert-file bypass
policy so repeated full-layer reads do not intentionally populate previously
nonresident expert-file pages. The trace observer changes synchronization and
therefore its timing is not a performance result. Reference and trace must have
the same prompt hash and complete output token IDs/hash.

The runtime deliberately leaves the final prompt token for the first ordinary
model step. The route recorder's `prefill` phase can therefore also contain
one-token first-response calls. Use `prefill_chunk_histograms`, not the
cumulative prefill histogram, and require exactly one
`4,095 * 6 = 24,570`-assignment chunk for each of the 42 layer-major expert
layers. Other chunks must be token-aligned, and their elementwise sum must equal
the cumulative histogram. Record the isolated 256-bin layer-major histogram,
union count, union fraction, empty experts, and threshold decision. The final
layer must have no 24,570-assignment chunk and is excluded from byte estimates.

For threshold `t`, choose full-layer read when `union / 256 > t`; otherwise
choose selective read. The estimated bytes are:

```text
sum(256 if union_fraction > threshold else union_count) * expert_blob_bytes
```

This is only a byte eligibility estimate. It assigns zero cost to router
synchronization, scattered positional reads, reduced queue depth, delayed I/O,
and lost next-layer overlap.

### Phase-1 continuation gate

Continue only if:

1. all five trace/reference token hashes are exact;
2. all histogram shapes, assignment counts, IDs, and byte identities are exact;
3. at least one of 70%, 80%, or 90% estimates at least 10% fewer prefill expert
   bytes on at least three of five workloads;
4. no selected layer requires more than the fixed full-layer buffer capacity.

If no threshold passes, stop without changing runtime code. A passing threshold
only authorizes a default-off runtime prototype.

## Conditional runtime prototype

Preserve one fixed full-layer destination buffer and the existing batched
`gather_qmm` shapes. Selective mode may leave unselected expert rows
uninitialized, but every gathered expert ID must have all six canonical regions
loaded into its original row offset. The fixed allocation and installed model
format must not change.

Because routes depend on current-layer attention output, the first prototype
must explicitly pay for route planning before choosing full or selective I/O.
It must not claim the current next-layer prefetch overlap. Compute each tile's
router indices/scores and shared-expert output once, retain the exact tensors,
form one layer union, choose 70%／80%／90%, read the chosen rows, and finish the
same route reduction and HyperConnection math. Do not recompute the router or
change QMM grouping.

Add request metrics for planned layers, full/selective decisions, selected
experts, estimated avoided bytes, plan wall, and actual batched bytes. Reject
composition with DSpark for the isolated gate. Keep the option internal and
default off; do not add a CLI, server, or APP setting.

### Correctness gate

For every runtime candidate require:

- exact prompt and greedy output token IDs/hash against paired full-layer
  control;
- every routed ID is included in the loaded union;
- selected read bytes equal selected rows times 13,369,344;
- full read bytes equal 256 rows times 13,369,344;
- actual request expert bytes close against full/selective decisions plus
  decode misses;
- no short read, uninitialized gathered row, unsafe buffer reuse, deadlock, or
  resource failure.

Any mismatch stops the candidate before performance interpretation.

## Conditional performance gate

Screen all three thresholds on the five 4K workloads with fresh-process paired
control/candidate order reversal and expert-file bypass. A threshold continues
only when all five outputs are exact, median expert bytes fall by at least 10%,
median TTFT improves by at least 5%, per-workload TTFT p95 does not regress more
than 5%, and peak MLX memory does not increase more than 5%. Since a two-pair
screen is exploratory, a passing threshold remains default off and requires a
later repeated-wave gate.

Also run one cached-not-purged observer pair for the best byte candidate. Do not
call it warm unless pre-read residency proves the relevant ranges resident; a
42-layer full expert working set exceeds this Mac's unified memory. Separate
bypass and cached observer results and never relabel either as physical SSD
bytes.

## Stop and evidence limits

Stop an exact candidate when router planning, scattered reads, or lost overlap
eliminates the TTFT gain even if logical bytes fall. Reopen only for a design
that overlaps route planning or selective I/O without prediction error, or for
a predictor with separately measured useful/wasted bytes.

This research does not measure physical storage-controller traffic, energy,
ANE contention, multi-request serving, long-context attention scaling, or model
quality. It does not authorize changing the current full-layer default.

## Results and decision

### Route-union eligibility passed

The five-workload route gate is saved in
[`2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json`](../docs/benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json)
(SHA-256
`bbd003adf9ba24df977e88de79ce5e616902a7496289b6d5743efaf808ae79f4`).
All five fresh-process reference/trace pairs have exact prompt and output token
hashes. Each trace isolates exactly 42 histograms with 24,570 assignments, and
all chunk histograms close to their cumulative histograms.

Median estimated prefill expert-byte changes across the five workloads were
-23.19% at 70%, -33.38% at 80%, and -34.24% at 90%. The number of workloads
saving at least 10% was 4/5, 5/5, and 5/5 respectively, so all thresholds
authorized the conditional runtime prototype. These are route-derived logical
byte estimates, not timing or physical-device results.

### Default-off runtime prototype was exact

The prototype retains the full 256-row fixed destination and original expert
row offsets. After current-layer attention it evaluates each tile's router once,
retains indices, scores, shared output, and HyperConnection tensors, forms the
layer union, then performs either a full or selected-row batched read before the
unchanged `gather_qmm`, route reduction, and HyperConnection expansion.

`RuntimeConfig.adaptive_expert_prefill_threshold` is internal, defaults to
`None`, accepts only 0.7/0.8/0.9, requires layer-major batched prefill, and
rejects DSpark and staged expert streaming. It has no CLI, server, or APP
opt-in. Request metrics separately record route planning, full/selective layer
decisions, union/read experts, successful adaptive batched bytes, avoided bytes,
and total request expert bytes.

The installed-model runtime gate is saved in
[`2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json`](../docs/benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json)
(SHA-256
`9138b15655c45e7493d1b40aa01b61b2e8afb908225641a11034e1fb6d9ff99f`).
It uses two reversed fresh-process pairs, the exact 4,096-token `repeated`
prompt, two greedy outputs, persistent cache off, DSpark off, and the explicit
expert-file bypass policy.

All four outputs are `[1950, 1950]` with SHA-256
`dd22c238773ee5642280c221b7a7a51094e6dcec6882cce0f749378ee01e4891`.
Each candidate plans 42 layers, chooses 42 selective and zero full layers, and
reads the exact 3,309-expert union. Successful adaptive batched bytes are
44,239,159,296 = 3,309 × 13,369,344. Adding 393 ordinary cache misses gives
49,493,311,488 total request expert bytes. The control reads all 10,752 prefill
experts plus the same 393 misses for 149,001,338,880 bytes. Avoided bytes close
exactly at 99,508,027,392.

| Metric | Control median | Adaptive median | Paired median change |
| --- | ---: | ---: | ---: |
| Request | 44.4157 s | 51.8707 s | +16.78% |
| TTFT | 44.1908 s | 51.6443 s | +16.87% |
| Logical request expert bytes | 149.001 GB | 49.493 GB | -66.78% |
| Peak MLX memory | 21.407 GB | 18.028 GB | -15.78% |
| Expert read wall sum | 22.8102 s | 7.3113 s | observational only |
| Routing sync wall | 0.1206 s | 2.0613 s | observational only |
| Adaptive plan wall | 0 | 2.5471 s | observational only |

TTFT p95 changes by +17.02%, which fails both the required 5% median TTFT
improvement and the no-more-than-5% per-workload p95 regression guard. The
control reports 41 completed next-layer prefetch hits while the candidate has
zero, as required by this first design. The measurements attribute the stop to
the combined route-planning, scattered-read, and lost-overlap schedule; they do
not isolate one of those as the sole cause.

For `repeated`, every observed union is below 70%, so 70%, 80%, and 90% produce
the same 42 decisions and the same selected rows. This one workload therefore
makes the per-workload p95 gate impossible for all three thresholds. The full
five-workload × three-threshold matrix and cached observer were not run after
this predeclared early stop. The decision is
`stop_runtime_candidate_keep_default_off`.

Reopen only for a design that overlaps exact route planning or selected I/O
without losing the current layer pipeline, or for a separately gated predictor
that reports useful and wasted bytes. Do not rerun this post-attention schedule
or describe its logical-byte reduction as a speedup.
