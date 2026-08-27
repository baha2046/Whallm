# DSpark storage-aware candidate-path feasibility protocol

Status: completed; structural gate passed and continuation gate failed. The
current branch-4 / beam-8 Markov-path candidate is stopped. No runtime option
or default changed.

## Question

The current DSpark runtime produces one five-token draft and the adaptive
scheduler can only shorten that draft. This experiment asks whether the same
single DSpark backbone evaluation contains a coherent alternative path that a
storage-only selector can choose while preserving or increasing the target's
greedy accepted prefix and reducing the exact first-three-layer hash-route
working set.

This is a candidate-existence and selector-feasibility gate. It is not an
end-to-end performance result and cannot enable DSpark by default.

## External architecture boundary

The [DFlash paper](https://arxiv.org/abs/2602.06036),
[official implementation](https://github.com/z-lab/dflash), and
[DFlash2 model card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
describe retaining multiple candidates at each position and tracing a coherent
path with a lightweight selector. The published DFlash2
[`config.json`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json)
uses its own selector and block configuration.

This project does not have that architecture or its Qwen checkpoint. The local
experiment uses only the pinned DeepSeek-V4 DSpark weights and its existing
low-rank Markov head. A positive result would establish feasibility for this
local factorization, not reproduce DFlash2 quality or speed.

## Local proposal semantics

For anchor token `a`, DSpark target features `h`, fixed per-position backbone
logits `b_i`, Markov embedding `E`, and projection `W`, define the five-token
proposal as

```text
q(t_1, ..., t_5 | a, h)
  = product_i softmax(b_i + W(E(t_{i-1})))[t_i],  t_0 = a.
```

The existing greedy DSpark path takes the local argmax at each transition. The
experiment evaluates a beam over the same conditional distributions. Joint
path scores use normalized log probabilities, not the raw temperature-zero
logits returned by the current runtime.

Greedy target verification remains target-equivalent because every selected
draft token is checked against a one-token-at-a-time target oracle and the
first mismatch is corrected by the target. Sampling is explicitly prohibited:
a deterministic storage selector changes the proposal distribution, while the
current rejection step only has per-transition probabilities for the original
single path. Sampled use requires a separate derivation and implementation of
the selector's complete proposal probability.

## Fixed experiment

Run exactly the five 128-token prompts in
[`2026-08-26-adaptive-128`](../docs/benchmarks/prompts/2026-08-26-adaptive-128/manifest.json):
`repeated`, `code`, `zh_technical`, `mixed_math`, and `tool_like`.

Each workload runs in a fresh process with:

- the installed model revision recorded in its manifest;
- greedy target decode (`temperature=0`, `top_p=1`);
- one first DSpark speculative round only;
- DSpark confidence truncation disabled;
- persistent prompt cache, layer-major prefill, hash prefetch, adaptive block,
  fallback, and page-cache probing disabled;
- the normal cached expert-file descriptor policy and the default 1,152 main / 
  768 DSpark slots.

The candidate lattice uses block size 5, stable branch width 4, and stable beam
width 8. Ties are resolved by lower token ID, then lexicographic token path.
The existing sequential-greedy path is included even if global beam pruning
would otherwise remove it.

The experiment must snapshot the target expert-cache residency immediately
after prompt prefill and the two target seed tokens, before generating target
truth. It records both cache-independent requested hash experts and
snapshot-dependent resident/missing hash experts.

## Candidate evidence

For every unique candidate, record:

- token IDs and joint normalized DSpark log probability;
- per-position confidence and confidence-derived expected commits;
- exact `tid2eid` routes and union sizes for target hash layers 0--2 over the
  anchor plus all five draft inputs;
- requested, resident, and missing expert blobs at the fixed snapshot;
- sequential target-oracle accepted-prefix length;
- realized useful and wasted hash-route keys at that accepted prefix.

Target truth is generated autoregressively from a cloned prompt cache. A
grouped target block is not an oracle because this checkout has already shown
shape-sensitive grouped/sequential differences. For each candidate, simulated
greedy verification must produce the same committed prefix as target truth.

The script also repeats the current `DSparkModel.draft` call from the same
DSpark context snapshot. Its untruncated greedy tokens must exactly match the
locally reconstructed sequential-Markov path.

## Predeclared selectors

Selectors may use only draft probability and pre-verification storage state;
they may not use realized target acceptance. For every pair
`lambda, mu` in

```text
{0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4} x
{0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4}
```

choose the stable maximum of

```text
joint_log_q(path)
  - lambda * requested_hash_expert_blobs(path)
  - mu * missing_hash_expert_blobs(path).
```

The probability-only choice is therefore the `lambda=mu=0` member. Report the
complete candidate Pareto frontier and an acceptance-aware oracle frontier
only as upper-bound diagnosis; neither may be used to make a selector choice.

## Validity, continuation, and stop criteria

The run is structurally valid only when all five prompts and prompt hashes
match the manifest, all five workers finish, the reconstructed baseline tokens
match `DSparkModel.draft`, every candidate has five valid Markov transitions,
all target truth comes from sequential one-token calls, every simulated
verified prefix matches target truth, and every hash key belongs to layers
0--2 of the installed main model.

A workload has a selector-feasible improvement when at least one predeclared
selector chooses a path different from the current sequential-greedy baseline
and that path:

1. accepts at least one draft token;
2. accepts at least as many draft tokens as the baseline;
3. requests no more hash expert blobs and has no more snapshot misses; and
4. lowers requested blobs or snapshot misses by at least one expert blob.

Continue to a multi-round runtime prototype only if the structural gate passes
and at least one of the five workloads has a selector-feasible improvement.
That follow-up must preselect one score without target outcomes, retain
sequential greedy verification, run exact multi-workload output gates, account
for learned-router as well as hash-layer target bytes, and measure total
request cost. This first experiment alone cannot justify a speed claim.

If no workload passes, stop the current branch-4 / beam-8 Markov-path candidate
and do not add runtime, CLI, server, or App settings. Reopen only with a
materially different candidate generator, a trained selector/drafter, or a
proposal-correct sampling design.

## Evidence limits

The five short first rounds do not measure steady-state cache churn, learned
router prediction, target verifier cost, physical SSD traffic, decode
throughput, latency distribution, energy, ANE execution, or long-context
behavior. Snapshot misses are LFU-cache state, not OS page residency or device
bytes. Realized target acceptance is used only to judge feasibility after a
selector has made its choice.

## Result

All five fresh workers passed every structural criterion. Each reconstructed
sequential-Markov path exactly matched the untruncated `DSparkModel.draft`
tokens and confidence (maximum absolute confidence error at most `1e-5`). All
candidate transitions belonged to the predeclared top-4 branches, hash routes
resolved to main-model layers 0--2, and every simulated committed prefix
matched the six-token sequential target truth.

The baseline draft was fully target-accepted, 5/5, in every workload:

| Workload | Baseline requested hash blobs | Baseline snapshot misses | Best alternative accepted prefix |
| --- | ---: | ---: | ---: |
| `repeated` | 18 | 0 | 4 |
| `code` | 104 | 75 | 4 |
| `zh_technical` | 102 | 67 | 4 |
| `mixed_math` | 103 | 38 | 4 |
| `tool_like` | 84 | 52 | 4 |

The 64 predeclared selectors chose an alternative in only two workloads. For
`code`, 40 selectors chose a path with 87 requested blobs and 59 misses, down
17 and 16 from baseline, but its accepted prefix fell from 5 to 4. For
`tool_like`, 37 selectors chose a path with 68 requested blobs and 38 misses,
down 16 and 14, but acceptance fell from 5 to 1. Every selector retained the
baseline for the other three workloads. The baseline joint normalized log
probabilities were between `-0.00000572` and `-0.000162`; the two selected
alternatives were much less likely at `-14.2063` and `-15.3769`.

No workload therefore met the predeclared requirement to reduce storage
without reducing accepted draft tokens. The machine-readable artifact is
[`2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-storage-aware-candidate-paths-first-round-m2-max.json)
with SHA-256
`de2ee382fff19f030a3c5150fe1251c352e51d6e5928fb2bfa7c5e40211d29ce`.
The benchmark script SHA-256 at the run was
`4d06391a5719f073d12eec6bfa60e2d7c495ee1612f4259ce751096413614d04`;
the predeclared protocol SHA-256 captured in the artifact was
`802f9407c59ffb5d6b8931ee8a65bbc9f6875146fbd7d008843d21dd7426192d`.

## Decision

The decision is `stop_current_markov_beam_candidate`. Do not implement a
multi-round runtime path selector, public setting, or sampled mode from this
candidate. The result does show that the local DSpark Markov factorization can
produce coherent beams and that storage weighting can materially lower exact
hash-route unions, but these high-confidence first rounds provide no accepted-
token headroom: any distinct path necessarily diverges before the fully
accepted baseline ends.

Do not rerun this exact five-workload branch-4 / beam-8 gate. Reopen only with
a materially different candidate generator or trained selector/drafter, and
predeclare a workload that tests that new mechanism. Sampling remains blocked
until the selected-path proposal probability and rejection rule are proved.
