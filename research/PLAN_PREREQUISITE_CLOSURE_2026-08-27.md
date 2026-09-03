# Remaining PLAN prerequisite-closure protocol

Status: local feasibility audit passed; all remaining prerequisites have explicit
scoped stop/defer and reopening rules.

2026-09-02 note: The ANE closure below applies to the missing DeepSeek dense
drafter or predictor candidate. Qwen now has a separate fixed-shape `q_proj`
implementation that directly uses private `AppleNeuralEngine.framework`.
That implementation does not supply the DeepSeek candidate described here.

## Purpose

All executable checkpoint-equivalent candidates in the current `PLAN.md` have
either passed a scoped gate, failed a stop criterion, or reached a documented
public-runtime boundary. The remaining tracker rows mix three different things:

1. measurement claims that need a lower-level tracer;
2. trained or approximate model candidates that do not exist locally;
3. platform experiments whose upstream dense candidate does not exist yet.

Leaving all three as generic `Pending` obscures whether more runtime code is
actually authorized or useful. This audit makes every prerequisite explicit and
defines the exact reopening condition. It does not invent training-data rights,
download a substitute model, install a training stack, or turn an approximate
model direction into checkpoint-equivalent runtime work.

## External design boundary

Core ML lets an application allow CPU + Neural Engine execution through
`MLComputeUnits.cpuAndNeuralEngine`, but that API surface does not supply a model
or prove that a graph runs on ANE. Apple ReDrafter's official repository exposes
separate training dependencies and describes training its published drafter
against a specific target/data setup; its MLX experiment is an Apple-Silicon
implementation, not a DeepSeek-V4 drafter checkpoint. HyperDFlash likewise
defines an MHC-aligned trained drafter, not weights embedded in the installed
model. ReMoE changes router weights through fine-tuning and therefore belongs to
the non-equivalent training scope.

Primary sources:

- [Apple `MLComputeUnits`](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [Apple Recurrent Drafter repository](https://github.com/apple/ml-recurrent-drafter)
- [Apple ReDrafter MLX experiment](https://github.com/apple/ml-recurrent-drafter/blob/main/recurrent_drafting/mlx/experiments/README.md)
- [HyperDFlash paper](https://arxiv.org/abs/2606.26744)
- [ReMoE paper](https://arxiv.org/abs/2605.27081)

## Predeclared local audit

The script must hash its own source and the working tree, inspect only the local
project and installed model, and record:

- installed revision, manifest hash, DSpark block/layer contract, and whether a
  separately named dense drafter/predictor tensor namespace exists;
- whether the project contains trained drafter/router artifacts, Core ML model
  packages, an approved data-rights/training contract, capability/safety
  evaluation artifacts, or a storage-native-model project specification;
- whether `coremltools`, PyTorch, datasets, and accelerate are present in the
  project venv;
- whether full Xcode exposes `xctrace`, `metal`, and `coremlcompiler` when the
  known `DEVELOPER_DIR` is selected;
- whether a current `.trace` capture or any benchmark field claiming attributable
  physical device bytes exists;
- whether the only learned-router labels are the bounded 6,000-label direct
  transfer feasibility artifact rather than a rights-approved train/validation/
  held-out dataset.

Tool presence is reported as capability only. It must never be treated as proof
of ANE placement, concurrent ANE/GPU benefit, native-loop speedup, or attributable
physical SSD measurement.

## Decision rules

The audit classifies directions as follows:

- **Dense drafter / trained predictor:** prerequisite-closed when no V4-specific
  trained artifact and no approved data/split/training contract exist. Reopen
  only with both.
- **ANE:** upstream-blocked when no resident dense candidate exists. Core ML
  compiler availability alone cannot open the gate. Reopen after a candidate,
  then require CPU/GPU/ANE placement, latency, energy, memory, and concurrent
  target-verification measurements.
- **Native zero-allocation loop:** measurement-deferred when there is no current
  per-operation trace selecting one stable boundary. Reopen only for a named
  boundary and token-hashed workload; do not rewrite the whole runtime first.
- **Physical I/O claim:** instrumentation-boundary-complete when logical bytes,
  process disk counters, and page-residency proxies are closed but no attributable
  device-byte field/tracer exists. Reopen only with a lower-level attributable
  source; never relabel proxies.
- **Router locality / mixed miss experts:** non-equivalent prerequisite-closed
  without training/approximate-mode contracts and capability/safety gates.
- **Storage-native MoE:** separate-project prerequisite-closed without explicit
  data, compute budget, architecture, objective, and evaluation suite.

These are scoped feasibility decisions, not proofs that the broad research ideas
can never work.

## Result

The audit ran against checkpoint revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062` and manifest SHA-256
`61b10294…fce64`. The installed model exposes the three-layer, block-5 DSpark
contract but zero separately named dense drafter／DFlash／ReDrafter／predictor
tensors.

Local state was:

| Evidence | Result |
| --- | --- |
| Trained candidate artifacts | 0 |
| Core ML model artifacts | 0 |
| Approved data/training/approximate-mode contracts | 0 |
| Current `.trace` captures | 0 |
| Benchmark fields claiming attributable physical-device bytes | 0 |
| Storage-native-model project specification | 0 |
| Frozen direct-transfer feasibility labels | 6,000; explicitly not trained data |
| `coremltools`／PyTorch／datasets／accelerate in `.venv` | all absent |
| Full-Xcode `xctrace`／`metal`／`coremlcompiler` | all present |
| System `fs_usage`／`iostat`／`powermetrics` executables | all present |

Tool availability therefore does not unblock ANE: there is no resident dense
Core ML candidate to place or compare. Likewise, observation executables do not
create an attributable per-request physical SSD counter or a candidate-bound
per-operation GPU trace. All nine audited directions received a distinct state,
`current_scope_closed=true`, and an explicit reopening requirement. Every audit
criterion passed.

Artifact:
[`2026-08-27-plan-prerequisite-closure-m2-max.json`](../docs/benchmarks/2026-08-27-plan-prerequisite-closure-m2-max.json),
SHA-256
`213087121a7dc7b0c39d1e15066c92f470dfaf6c5326dfcf9889a6d8a6fafcfa`.

## Decision

Close the remaining PLAN scope as prerequisite-deferred or scoped-stop, not as
implemented runtime features. No training, approximate checkpoint, DeepSeek
dense ANE model, native rewrite, or physical-device claim is authorized by
this audit.

The PLAN research can be considered covered when the main tracker points each
row to its implemented gate, stopped candidate, feasibility rejection, or this
prerequisite audit. Future work reopens only the affected row after its recorded
condition changes; it does not invalidate completed exact-runtime milestones.
