# Native MTP-1 checkpoint contract audit

Status: complete feasibility rejection for the pinned checkpoint. This file is
an active research record, not a runtime performance result.

## Question and stop rule

[`PLAN.md`](../PLAN.md) proposes native MTP-1 as the minimum-cost drafter
baseline because the Transformers [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json)
sets `num_nextn_predict_layers=1`.

The local hypothesis was that the same pinned checkpoint might expose one
self-contained `mtp.*` stage that accepts target hidden state and produces
draft logits. The implementation gate was:

1. The pinned inference graph must define a one-stage execution path.
2. That path must have every checkpoint tensor needed at its input and output.
3. The installed model must preserve those tensors and their integrity.
4. No stage remapping, skipped trained block, substituted head, or changed
   block contract may be called checkpoint-faithful without a primary-source
   graph that defines it.

If no such path exists, stop before adding a misleading runtime flag and record
a reproducible feasibility rejection.

## Primary sources and method

[`Scripts/audit_mtp1_contract.py`](../Scripts/audit_mtp1_contract.py) downloads
and hashes four files at revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`:

- the Transformers [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json);
- the official [`inference/config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/config.json);
- the official [`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py);
- the checkpoint [`model.safetensors.index.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/model.safetensors.index.json).

It then compares those files with the installed model manifest and both local
configuration files. It reconstructs the complete installed `mtp.*` tensor set
from common tensors plus every expert region, compares that set with the
checkpoint index, classifies stage ownership, and hashes all four installed
DSpark payload files.

Reproduce with:

```sh
.venv/bin/python Scripts/audit_mtp1_contract.py \
  --model ~/.dsmodel/deepseek-v4-flash-0731.dsv4 \
  --output docs/benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json
```

## Result

The two MTP-related configuration fields are not interchangeable:

- Transformers metadata: `num_nextn_predict_layers=1`.
- Official inference contract: `n_mtp_layers=3`, block size 5, target layers
  40–42, and Markov rank 256.

The pinned reference graph does not read `num_nextn_predict_layers`. It creates
`n_mtp_layers` blocks, enters through `mtp.0.forward_embed`, executes every
block, and exits through `mtp[-1].forward_head`.

The checkpoint index contains 4,705 `mtp.*` tensors across exactly three
stages:

| Stage | Tensor count | Input adapter | Output head | Self-contained |
| ---: | ---: | --- | --- | --- |
| `mtp.0` | 1,568 | complete | absent | no |
| `mtp.1` | 1,565 | absent | absent | no |
| `mtp.2` | 1,572 | absent | complete | no |

`mtp.0` owns `main_proj` and `main_norm`. `mtp.2` owns `norm`, `hc_head`, the
Markov head, and the confidence head. No stage contains both sides of a
one-stage drafter contract. The reconstructed installed tensor set exactly
matches the pinned checkpoint index.

The installed DSpark payload is 10,862,841,600 bytes (10.117 GiB). The manifest
and actual SHA-256 values match for `dspark/common.bin` and all three expert
layer files.

The raw machine-readable evidence is
[`2026-08-27-mtp1-checkpoint-contract-audit.json`](../docs/benchmarks/2026-08-27-mtp1-checkpoint-contract-audit.json).
It is explicitly marked `formal_performance_result=false`.

## Decision

The pinned checkpoint does not expose a faithful, separable native MTP-1
baseline. Runtime stage truncation is rejected.

Running only `mtp.0` lacks the trained output-side tensors defined by the
reference graph. Jumping from `mtp.0` to the `mtp.2` heads, substituting the
main-model head, skipping stages, or changing block size would create a graph
outside the pinned reference contract. Such a candidate may be investigated
only as a separately trained or explicitly approximate model; it cannot be
presented as checkpoint-equivalent MTP-1.

Valid follow-ups are therefore:

1. obtain a checkpoint with a reference-defined self-contained one-stage MTP
   drafter;
2. train or distill a new resident one-stage drafter and treat it as a new
   checkpoint candidate; or
3. continue reducing integration cost of the complete three-stage DSpark graph
   without relabeling it MTP-1.

## Evidence limits

This audit proves only the pinned checkpoint and reference-graph contract. It
does not claim that separately trained MTP-1 is impossible, does not evaluate
an approximate truncation, and contains no timing, memory-pressure, power, or
quality result.
