# PLAN.md research execution status

This file maps every active direction in [`PLAN.md`](../PLAN.md) to current
code, reproducible evidence, a stop or adoption decision, and the reopening gate.
It is the completed research ledger for this PLAN snapshot, not the source of truth for runtime behavior;
current behavior remains documented under [`docs/`](../docs/README.md).

Snapshot date: 2026-08-27. Base commit:
`997e2ca756d3d6c8ae97aa4df3effcf566ed449f`, plus the current dirty research
working tree. Checkpoint revision:
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

## Completion rule

A PLAN direction is research-complete only when all applicable items exist:

1. A precise local hypothesis and target-equivalence classification.
2. A prototype, measurement probe, or documented feasibility rejection.
3. Reproducible workload, configuration, hashes, and raw artifact.
4. Correctness and stop criteria appropriate to the claimed scope.
5. An adoption, rejection, or deferred-prerequisite decision in `docs/`.

`Implemented` below does not mean enabled by default. `Partial` means current
evidence covers only part of the PLAN requirement. `Pending` means the required
local evidence does not exist yet. Negative results count as completed research
only for the exact candidate and conditions tested; they do not reject an
entire research direction.

All 21 rows below now satisfy this scoped completion rule. There are no current
`Partial` or `Pending` rows: executable exact candidates are implemented or
stopped by their predeclared gates, while training／approximate／platform work is
closed through explicit local prerequisite evidence and reopening conditions.
This does not mean every candidate was adopted or that deferred training was
performed. The final local closure artifact is
[`2026-08-27-plan-prerequisite-closure-m2-max.json`](../docs/benchmarks/2026-08-27-plan-prerequisite-closure-m2-max.json),
SHA-256
`213087121a7dc7b0c39d1e15066c92f470dfaf6c5326dfcf9889a6d8a6fafcfa`.

## Requirement matrix

The `P3-ANE` row records the historical DeepSeek drafter and predictor scope.
It does not describe the later Qwen fixed-shape `q_proj` ANE Prefill route.

| ID | PLAN direction | Equivalence | Current state and evidence | Reopening or adoption gate |
| --- | --- | --- | --- | --- |
| P0-I/O | Physical I/O, useful/wasted prefetch, and bytes per committed token | Exact | **Instrumentation boundary research-complete.** Runtime separates logical, process-wide disk, and expert-file pre-read residency bytes. Six fully nonresident installed ranges pass aligned bypass／cached byte and residency checks. A 4K／32 three-wave candidate gate plus observer wave records output hashes, peak memory, committed-token bytes, process disk bytes, and closed residency partitions. The prerequisite audit confirms no artifact field provides attributable physical-device bytes; local `fs_usage`／`iostat`／`powermetrics` executables do not by themselves establish that attribution. See the [bypass protocol](EXPERT_FILE_CACHE_BYPASS_2026-08-27.md), [full-model gate](../docs/benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json), and [closure audit](PLAN_PREREQUISITE_CLOSURE_2026-08-27.md). | Reopen a physical-device claim only with a lower-level per-request attributable source. Keep logical, process, and residency proxy labels separate; never relabel them physical SSD traffic. |
| P0-union | Verification expert-union profiler | Exact | **Implemented and full-model exercised.** Runtime reports acquisition calls, assignments, acquired union experts, reuse, misses, verification/replay bytes, read time, and committed-token cost. The corrected 128/8 four-wave gate proves sequential/grouped/hybrid calls 258/43/43 with closed observer accounting. | Keep as required instrumentation for every future verifier candidate; do not infer bytes or speed from calls alone. |
| P0-GPU | GPU busy/idle and synchronization boundaries | Exact | **Measurement scope research-complete; recapture deferred.** Historical M5 Pro Metal captures cover repeated 4K, tool-like 4K, and tool-like 14K and were sufficient to avoid an unsupported kernel-first conclusion. The closure audit finds full-Xcode `xctrace` available but no current candidate-bound `.trace`; tool presence is not timing evidence. | Reopen only when a materially changed candidate names one stable boundary. Bind the new capture to exact source, workload, configuration, and token hash before using per-operation timing. |
| P1-hash | Exact prefetch for the first three hash-routing layers | Exact | **Correctness complete; current performance candidate rejected, default off.** Exact unions, pinning, scratch, replay, and byte classification are tested; five 4K workloads pass parity. Under the 4K／32 bypass-policy repeated gate, fixed hybrid + hash has request +18.19%, Decode -59.67%, memory +9.06%, and disk/token +13.14% versus normal, so it fails the predeclared request, throughput, and disk gates. See the [full-model artifact](../docs/benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json). | Do not rerun this exact candidate. Reopen only after verifier/prefetch architecture changes reduce target-math cost; a new gate still needs multiple workloads and attributable I/O if physical-byte claims are made. |
| P1-union | Block verification with per-layer union, dedupe, and grouped QMM | Exact | **Current implementations research-complete; no adoptable candidate.** Grouped block is faster on the exact high-confidence case but fails low-margin multi-workload parity. Hybrid v3 passes five-workload greedy parity and exactly reduces acquisitions from 258 to 43 for one five-token block; versus sequential it reduces target bytes 12.31% but raises verification time 17.34%, and is 87.59% slower than grouped. The timing covers all execution-shape differences, not QMM alone. See the [protocol](VERIFIER_UNION_TRADEOFF_2026-08-27.md) and [corrected artifact](../docs/benchmarks/2026-08-27-verifier-union-tradeoff-repeated-128x8-m2-max.json). | Do not rerun these exact paths. A new candidate must preserve five-workload exactness while restoring more grouped execution, then repeat the union and end-to-end gates. |
| P1-cache | Speculative cache bypass, scratch, probation, and deadline pinning | Exact | **Executable scope complete; predictor-dependent policies deferred.** Main and DSpark expert descriptors support a default-off aligned bypass policy; six installed ranges prove bypass reads do not populate previously nonresident pages. Fixed verification scratch bypasses LFU admission, resident exact-prefetch entries are transaction-pinned, and useful/wasted reads are classified. Direct frozen-router transfer failed its gate, so there is no eligible predictor for probation/deadline admission. | Reopen probation/deadline only after a trained predictor passes held-out useful/wasted-byte gates; then prove it preserves the resident hot set and committed-token byte cost. |
| P1-adaptive | SSD-aware adaptive verification length | Exact | **Correctness complete; current performance candidate rejected, default off.** Five-workload composition is exact and 63 decisions exercise variable prefixes. In the 4K／32 bypass-policy repeated gate, adaptive lowers speculative bytes/committed 42.96% versus fixed but changes request/Decode by +5.02%/-25.58% versus normal; it fails both adoption gates. See the [composition artifact](../docs/benchmarks/2026-08-27-dspark-hybrid-v3-hash-adaptive-discovery-4k32-m2-max.json) and [repeated gate](../docs/benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json). | Do not rerun this exact cost model. Reopen after target-verifier or predictor changes; remaining inputs include learned-router union, attributable I/O, queue depth, GPU idle, memory pressure, and full-draft cost. |
| P1-path | Storage-aware multi-candidate path selector | Exact for greedy; rejection sampling required otherwise | **Research-complete for the tested Markov beam; candidate stopped.** The local proposal was defined as a coherent first-order Markov factorization over one fixed five-position DSpark backbone result. Five fresh 128-token first-round workers pass baseline reconstruction, confidence, top-4 transition, sequential target-oracle, committed-prefix, and exact hash-layer route gates. All baselines are accepted 5/5. Storage weights select lower-union alternatives for `code` and `tool_like`, but accepted prefixes fall to 4 and 1; 0/5 workloads preserve acceptance while lowering requested/missing hash blobs. See the [protocol and decision](DSPARK_CANDIDATE_PATHS_2026-08-27.md) and [artifact](../docs/benchmarks/2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json). | Keep the branch-4／beam-8 candidate out of runtime／CLI／server／APP and do not rerun the exact gate. Reopen only with a materially different candidate generator or trained selector/drafter. Sampling remains prohibited until the selected-path proposal probability and rejection rule are proved. |
| P1-predictor | Draft-hidden-to-target route predictor for learned routers | Exact if used only for prefetch | **Fixed-label baseline complete; direct frozen-router transfer stopped.** Five fresh first rounds provide 6,000 aligned target assignments for learned layers 3--42 plus separately captured anchor routes. Four DSpark feature taps × top-6／12／24／48 close assignment and useful／wasted／missed union bytes exactly. Best eligible-size candidate (`dspark_final_norm`, top-24) reaches only 17.15% assignment recall, 33.03% union recall, 11.15% useful rate, 12.42%--22.42% per-workload recall, and 0/40 layers at 75%. Even ineligible top-48 reaches only 28.93% assignment recall and 10.03% useful rate. See the [protocol and decision](DSPARK_LEARNED_ROUTER_PREDICTOR_2026-08-27.md) and [artifact](../docs/benchmarks/2026-08-27-dspark-learned-router-frozen-transfer-m2-max.json). | Do not build scratch prefetch, probation, deadline pinning, or runtime settings from direct transfer. Reopen only with a new data-rights／collection protocol, substantially larger route dataset, train／validation／held-out splits, and an immutable trained reducer/predictor artifact. The negative baseline does not rule out trainable signal. |
| P2-MTP | Native MTP-1 minimum-cost baseline | Exact | **Research-complete feasibility rejection for the pinned checkpoint.** The pinned inference graph consumes `n_mtp_layers=3`, not Transformers metadata `num_nextn_predict_layers=1`; `mtp.0` owns the input adapter, `mtp.2` owns the output heads, and no stage is self-contained. All 4,705 indexed `mtp.*` tensors and the 10,862,841,600-byte installed payload pass contract and SHA checks. See the [audit](MTP1_CONTRACT_AUDIT_2026-08-27.md) and [artifact](../docs/benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json). | Reopen only for a reference-defined one-stage checkpoint or a separately trained/distilled drafter. Do not label stage truncation checkpoint-equivalent. |
| P2-integration | Remove DSpark duplicate prompt prefill and independent large cache | Exact | **Current candidates research-complete.** The default-off format-3 atomic target+DSpark snapshot passes independent-clone, namespace, malformed/partial, eviction, close, memory-hit, and restart tests. Its installed 128/8 gate has exact `none/memory/persistent` sources, 0/127/127 reused tokens, exact output hashes, and combined logical expert bytes -92.23%/-65.59%. The separate 96-slot candidate is stopped because long-decode draft bytes/committed rise 83.17%. See the [snapshot protocol](DSPARK_PROMPT_CONTEXT_SNAPSHOT_2026-08-27.md), [snapshot artifact](../docs/benchmarks/2026-08-27-dspark-prompt-context-snapshot-repeated-128x8-m2-max.json), and [96-slot stop](../docs/benchmarks/2026-08-27-dspark-slots-768-vs-96-random-hex-4k128-m2-max.json). | Keep prompt-context reuse and 768 slots default-off/unchanged. Adoption needs balanced multi-workload waves including external acquisition/load wall time. A different shared physical pool requires a new namespaced slot-ownership/fence design, not another 96-slot rerun. |
| P2-drafter | HyperDFlash/ReDrafter-style resident dense drafter | Exact after target verification | **Prerequisite-closed for this checkout.** The installed manifest has zero separately named dense drafter／DFlash／ReDrafter／predictor tensors; the repo has no trained candidate, approved data-rights/split/training contract, or training stack. Apple ReDrafter and HyperDFlash are architecture/training references, not V4 weights. See the [local closure audit](PLAN_PREREQUISITE_CLOSURE_2026-08-27.md) and [artifact](../docs/benchmarks/2026-08-27-plan-prerequisite-closure-m2-max.json). | Reopen only with a V4-specific immutable trained artifact and approved data, train/validation/held-out split, and training contract. Start with block 4, then measure acceptance, complete drafter cost, memory, and target-equivalent output. |
| P2-staged | `w13`/`w2` staged expert streaming | Exact | **Research-complete for the tested split schedule; runtime candidate stopped.** A 36-pair fixed-arena gate is byte/output exact and passes the optimistic 4/8-row continuation criteria. The resulting default-off split-slot runtime passes four fresh-process 128/32 correctness waves: all eight token hashes match, per-pair logical bytes/evictions do not increase, peak memory is unchanged, and 1,024 staged reads/run close exactly. It fails performance: paired median request +2.17%, Decode -4.26%, p95 +2.12%. See the [protocol and decision](STAGED_EXPERT_STREAMING_2026-08-27.md), [fixed-arena artifact](../docs/benchmarks/2026-08-27-staged-w13-w2-overlap-m2-max.json), and [runtime artifact](../docs/benchmarks/2026-08-27-staged-expert-runtime-repeated-128x32-m2-max.json). | Keep `staged_expert_streaming=false` and do not expose this candidate in CLI/server/APP. Reopen only after a materially different first-stage kernel or I/O schedule enlarges the real one-row overlap window; predeclare a new multi-workload gate rather than rerunning this schedule. |
| P2-MTLIO | MTLIO, fixed Metal buffers, and explicit fences | Exact | **Research-complete for the installed MLX 0.32.0 surface; runtime integration stopped.** A 16-row 1／6／32／128 native matrix covers 8,930,721,792 candidate bytes; CPU bytes, shared/private buffers, shared-event GPU visibility, direct shared view, explicit private validation copy, and cancellation admission all pass. Public MLX exposes a raw-pointer/DLPack resource candidate but no supported external `MTLSharedEvent` dependency handoff. Exploratory 32／128 bytes/shared throughput remains within 0.78% of separate-file `preadv`; OS cache was not purged. See the [protocol](MTLIO_EXPERT_STREAMING_2026-08-27.md) and [artifact](../docs/benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json). | Reopen only when installed MLX publicly supports the resource + event/stream conjunction. Then repeat balanced cache-state waves and require the predeclared 10% latency/throughput gate without p95 regression before an isolated runtime prototype. Do not use allocator/backend private casts. |
| P2-prefill | Adaptive full-layer versus selective expert prefill | Exact | **Research-complete for the tested post-attention schedule; runtime candidate stopped.** Five exact 4K route traces make 70/80/90% eligible, with median estimated prefill-byte changes -23.19%/-33.38%/-34.24%. The internal fixed-buffer prototype then passes two reversed installed-model correctness pairs: all tokens exact, 3,309 selected rows and 44,239,159,296 adaptive batched bytes close exactly, request expert bytes fall 66.78%, and peak MLX memory falls 15.78%. TTFT instead regresses 16.87% paired median and 17.02% at p95. `repeated` has identical decisions at all three thresholds, so every threshold fails its per-workload p95 guard. See the [protocol and decision](ADAPTIVE_EXPERT_PREFILL_2026-08-27.md), [route gate](../docs/benchmarks/2026-08-27-adaptive-expert-prefill-route-union-4k-m2-max.json), and [runtime gate](../docs/benchmarks/2026-08-27-adaptive-expert-prefill-runtime-repeated-4k-m2-max.json). | Keep full-layer prefill as the default and expose no CLI/server/APP opt-in. Do not run the stopped full matrix or cached observer. Reopen only with overlapped exact planning/selective I/O or a separately measured predictor; require useful/wasted bytes and a new predeclared performance gate. |
| P2-prefix | Block-granular immutable persistent prefix cache | Exact | **Implemented; functional gate complete.** Normal format 4 keys cumulative cache checkpoints by checkpoint/model-config SHA, explicit RoPE, KV/index format, attention mode, state schema, and a 128-token SHA-256 parent chain. Non-layer-major prefill captures bounded first/final checkpoints; identical prefixes share one immutable payload, while a sidecar drives reuse-count/access-recency eviction. Fixture tests cover contract changes, sharing, suffix partial restart, eviction, namespace isolation, and MXFP8 snapshot isolation. The installed 453-token branch gate has 442 common tokens, reuses exactly 128 after restart, matches isolated-cold token `36363`/hash, and leaves the 7,002,850-byte payload plus metadata hashes unchanged. See the [protocol](BLOCK_PROMPT_CACHE_2026-08-27.md) and [artifact](../docs/benchmarks/2026-08-27-block-prompt-cache-partial-restart-m2-max.json). | Keep format 4 and the current two-checkpoint/eight-payload bounds. Balanced multi-workload waves are required before a performance claim or larger default. Per-layer KV delta-object dedupe is a separate storage-amplification/crash-consistency candidate, not required for the validated cumulative immutable-sharing contract. |
| P3-ANE | ANE drafter, confidence head, route predictor, and selector | Exact after target verification | **Upstream prerequisite-closed.** Full Xcode exposes `coremlcompiler`, but the checkout has no Core ML model, `coremltools`, or resident dense drafter/predictor candidate. Compiler presence does not prove ANE placement or concurrent ANE/GPU benefit. Main routed experts remain outside this candidate. | Reopen after a resident dense candidate is converted to Core ML. Then verify placement and compare CPU/GPU/ANE latency, energy, memory, and concurrent target-verification contention. |
| P3-native | Native Metal zero-allocation decode path | Exact | **Measurement-deferred with explicit entry gate.** Expert slots are fixed, but Python/MLX graph construction, cache mutation, ownership, and fences are not a zero-allocation loop. Full Xcode has `metal`/`xctrace`; the checkout has no current candidate-bound `.trace`, so no stable boundary is selected for a justified rewrite. | Reopen only with a current source/workload/token-hashed trace that selects one stable boundary. Preallocate that boundary and require exact tokens plus reduced CPU/GPU bubbles before widening scope. |
| P4-router | Router locality fine-tuning | Non-equivalent | **Non-equivalent prerequisite-closed.** The design and external evidence are documented, but this checkout has no approved data/training contract, immutable trained router artifact, or capability/safety evaluation. The audit intentionally does not authorize training. | Reopen only with approved data, train/validation/held-out splits, trust-KL/balance/cache-cost objectives, immutable weights, and broad capability/safety gates; never label the mode checkpoint-equivalent. |
| P4-mixed | Lower-precision miss experts or expert dropping | Non-equivalent | **Non-equivalent prerequisite-closed.** Checkpoint experts are already FP4; there is no explicit approximate-mode contract, sub-FP4/residual candidate, trained correction, or quality/safety artifact. | Reopen only after defining an explicit approximate mode and quality/safety gate; include residual correction or training where required. |
| P4-native-model | Storage-native MoE architecture | New model | **Separate-project prerequisite-closed.** The architecture ideas are documented, but no local project spec defines data, compute budget, architecture, cache-cost objective, or evaluation suite. This is not an optimization of the installed checkpoint. | Reopen as a separate model-training phase with all five program inputs specified; do not expand the current runtime goal implicitly. |

## Verifier correctness resolution

The original grouped block verifier remains stopped. The 2026-08-27 diagnostics
located shape-dependent differences in attention HyperConnection/Q execution,
then in FFN HyperConnection; hybrid v1 and v2 proved that removing only those
subsets did not generalize across rounds and workloads.

Hybrid v3 keeps attention, FFN HyperConnection, router, shared and routed expert
math, and final expansion in one-token shapes. It first evaluates every token's
routes and then acquires the complete layer expert union once. This preserves
I/O dedupe but gives up grouped multi-row expert QMM. The five-workload 4K/32
normal/fixed gate is exact, including the two workloads that failed v2. Hash
prefetch and adaptive composition also preserve exact normal/fixed/adaptive
tokens across all five workloads. This resolves the correctness prerequisite
for those exact candidates; it does not prove sampling correctness, long-decode
coverage, physical-I/O savings, or performance adoption.

## Next executable gates

1. Keep the stopped fixed/adaptive bypass-policy candidates out of runtime
   defaults. A new speed gate requires a materially different verifier,
   predictor, or cost model rather than another repetition of the same path.
2. Keep the stopped 96-slot candidate and newly validated atomic prompt-context
   option out of runtime defaults. Revisit prompt reuse only with balanced
   multi-workload waves; revisit expert-cache sharing only through a namespaced
   physical slot-pool ownership/fence contract.
3. Keep the native-correct MTLIO mechanism out of runtime ownership while MLX
   lacks a supported external event dependency handoff. Reopen only after that
   public prerequisite changes; do not rerun the same native matrix now.
4. Keep the stopped staged `w13`/`w2` split-slot schedule out of runtime
   interfaces and defaults. Reopen only for a materially different first-stage
   compute or I/O schedule; the existing fixed-arena result alone is not an
   end-to-end speed claim.
5. Keep the stopped post-attention adaptive prefill schedule internal and off.
   Its byte reduction is exact but its lost overlap fails TTFT for every tested
   threshold on `repeated`; reopen only with a different overlap/prediction
   architecture.
6. Keep the stopped branch-4／beam-8 DSpark Markov path selector out of runtime.
   Its coherent greedy-only contract is proven, but 0/5 workloads preserve the
   fully accepted baseline while lowering hash storage. Sampling remains
   prohibited without a selected-path proposal proof.
7. Keep direct frozen-router transfer stopped: its 6,000-label gate reaches only
   17.15% top-24 assignment recall and 11.15% useful rate. A learned predictor
   now requires an explicit data-rights／collection and training protocol.
8. Keep normal prompt-cache format 4. Its restart／partial-match／sharing contract
   is functionally validated; do not interpret the three-run timing as a speedup
   or relabel cumulative checkpoints as per-layer KV delta dedupe.
9. The remaining drafter／trained predictor／router-locality／storage-native-model
   directions require explicit data-rights, training, capability/safety, or
   platform prerequisites. ANE work additionally waits for a resident dense
   candidate. They are not safe implicit runtime implementation work.

No current evidence supports enabling DSpark, atomic DSpark prompt-context
reuse, hash exact prefetch, adaptive block selection, or approximate-model
behavior by default.
