# DeepSeek-V4-Flash-0731 implementation plan

Current status: M1 through M4 are complete on the target M5 Pro. The 145 GiB
main model is installed. All 49 installed files pass SHA-256 verification.
The 4K BF16 and 8K BF16/MXFP8 end-to-end checks pass.

## Outcome

Build a model-specific Apple Silicon runtime that keeps common tensors resident and reads routed experts from SSD when the router selects them.

## Runtime flow

1. The loader verifies `manifest.json`.
2. The loader maps common tensors once.
3. Metal computes attention, the router, and the shared expert.
4. The CPU reads six expert IDs.
5. The expert cache returns resident slots or fills missing slots with bounded `pread` calls.
6. Metal computes the six routed experts and reduces their outputs.
7. The runtime repeats the flow for 43 layers.

## Installed model layout

```text
deepseek-v4-flash-0731.dsv4/
  manifest.json
  common.bin
  experts/
    layer_00.bin
    ...
    layer_42.bin
```

Each layer file contains 256 fixed-stride expert blobs. Each expert blob uses this order:

1. `w1.weight`
2. `w1.scale`
3. `w2.weight`
4. `w2.scale`
5. `w3.weight`
6. `w3.scale`

The repacker copies checkpoint bytes without changing their values.

## Milestones

### M1.5: Recoverable installation and hardware evidence

- Resume a partial repack without repeating completed writes.
- Store a durable receipt for completed source spans.
- Install tokenizer and prompt-encoding inputs.
- Measure expert `pread` with controlled cache and concurrency settings.
- Complete the pinned main-model installation on the target M5 Pro.

### M1: Checkpoint and repack foundation

- Validate the pinned model contract.
- Read Hugging Face metadata with HTTP Range requests.
- Parse safetensors headers without downloading tensor data.
- Produce a deterministic repack plan.
- Repack through a bounded transfer buffer.
- Verify installed file sizes and SHA-256 values.

Implementation result:

- The pinned official checkpoint produces 67,612 copy operations.
- The installed main-model weight size is 156,015,738,880 bytes.
- The repacker uses an 8 MiB transfer buffer.
- The receipt contains 18,615 verified transfer chunks.
- Per-chunk autorelease pools keep resume validation near 140 MiB RSS.
- The fixture suite verifies planning, layout rejection, bounded copying, and safetensors parsing.

### M2: Correct reference decode

- Implement FP4 E2M1 and FP8 E4M3 Metal operations.
- Implement mHC, attention, both router modes, shared expert, and routed MoE.
- Support batch size 1, greedy decode, and a 4K context.
- Compare intermediate tensors with the official reference implementation.

Implementation result:

- The runtime uses the pinned DeepSeek-V4 MLX implementation.
- The installed model Adapter maps all 1,564 common tensors.
- Sanitized weight names match all 1,564 model parameters exactly.
- Routed expert MXFP4 output matches a dequantized reference in the test suite.
- A 4,096-token BF16 prefill and one greedy output token complete on Metal.

### M3: SSD expert streaming

- Allocate a small global slot pool.
- Add LFU expert caching.
- Overlap shared expert compute with bounded `pread` calls.
- Record cache hit rate, bytes read per token, memory, and decode speed.

Implementation result:

- The global slot pool uses LFU with recency as its tie breaker.
- Four `pread` workers fill missing slots.
- Shared expert Metal work starts before the CPU waits for missing slots.
- The command reports cache hit rate, SSD bytes, read time, evictions, and peak memory.
- Four direct-read workers reach 14.30 GiB/s on the target SSD.

### M4: Throughput and context

- Add chunked prefill.
- Add FP8 KV cache after correctness validation.
- Increase validated context lengths.
- Add DSpark only if measurements show a net speed increase.

Implementation result:

- Generation uses a 32-token prefill chunk by default on the target M5 Pro.
- Completed compressed-attention cache chunks use MXFP8.
- The 128-token local attention cache stays in BF16 because its size is bounded.
- BF16 and MXFP8 produce the same greedy token for the 8,192-token test.
- DSpark stays disabled. The measured baseline is dominated by expert work and
  would require another 10.12 GiB of installed weights. No measured result
  shows that DSpark gives a net speed increase for this SSD runtime.

## Acceptance criteria for M1

- A fixture checkpoint produces the expected expert layout and manifest.
- The planner rejects a missing expert tensor, an invalid shape, and a changed model contract.
- The remote inspector validates the pinned official checkpoint without downloading model weights.
- Tests run with `swift test` and use no third-party package.
