# Block-granular immutable persistent prefix-cache protocol

Status: implemented; fixture and installed-model functional gates passed.

## Question

The normal runtime currently retains two in-memory timelines and writes complete
target-only cache timelines to disk. A saved shorter timeline can be reused when
it is an exact prefix of a new prompt, but the old format does not prove that the
RoPE, KV representation, attention implementation, or model configuration is
compatible. It also cannot recover a safe checkpoint when a later suffix of the
saved request diverges.

This gate asks whether a normal, target-only cache can use immutable,
content-addressed token-block checkpoints to resume an exact shared prefix after
a process restart.

## Runtime contract

Normal persistent entries use format 4. A cache identity is a SHA-256 chain over
128-token blocks. The root contract contains:

- installed-model ID and checkpoint revision;
- canonical SHA-256 of the complete model `config.json`;
- explicit RoPE fields;
- compressed KV and index-cache formats;
- sparse pooled-attention implementation version, sliding window, compression
  ratios, and attention fields;
- cache-state schema and token-block size.

Metadata repeats the complete contract, its hash, every ordered token-block
descriptor, the terminal cache key, the exact token IDs, and the immutable data
filename. The scanner recomputes and compares all fields. Legacy normal format
1/2 entries are ignored because their metadata cannot prove this contract.
DSpark format 3 remains in its separate atomic namespace.

The normal prefill path retains at most two additional persistence checkpoints
per request: the first completed prefill chunk and the final `prompt[:-1]`
checkpoint. This bounds live snapshot state while giving a shared system/tool
prefix and the near-complete conversation prefix a reusable boundary. The final
request timeline is still saved. An identical contract and token prefix maps to
one data/metadata pair; later requests do not overwrite it.

Reuse accounting lives in a mutable sidecar, not in immutable cache metadata.
Eviction ranks reuse count first and access recency second so a repeatedly shared
system/tool checkpoint survives one-off suffixes.

## Fixture gate

Tests must prove:

1. KV format, model configuration/RoPE, or token changes produce a different
   contract or content key;
2. two writes of the same prefix leave one unchanged data/metadata payload;
3. a new runtime can reuse the first checkpoint when the old suffix diverges;
4. a reused checkpoint survives frequency-aware eviction;
5. a different KV contract is rejected;
6. an MXFP8 pooling snapshot owns an independent active remainder;
7. normal format 4 and DSpark format 3 cannot cross-load.

Any fixture failure stops the installed-model gate.

## Predeclared installed-model gate

Use the pinned installed model, batch size 1, greedy decoding, one output token,
layer-major prefill off, and an isolated new cache directory. Build two prompts
from the pinned `repeated-128.txt` fixture followed by the same 24-line repository
policy body. Change only the final branch task. The tokenizer must report at
least 128 common prefix tokens, and the divergence must occur before the final
prefill checkpoint of the first request.

Run in this order:

1. runtime A, branch A, cold persistent cache;
2. close A; runtime B, branch B, same persistent cache;
3. close B; runtime C, branch B, persistent cache disabled.

The gate passes only if:

- runtime A reuses zero tokens;
- runtime B reuses exactly the immutable 128-token checkpoint after restart;
- runtime C reuses zero tokens;
- runtime B and runtime C produce identical token IDs and output-token SHA-256;
- the 128-token metadata/data payload is byte-identical before and after runtime
  B and occurs only once;
- its access sidecar records the persistent hit;
- metadata, content-address chain, revision, complete cache contract, and data
  file all validate;
- no cache write reports an error.

Wall time, TTFT, logical expert bytes, and file sizes are descriptive. This is
not a balanced performance experiment and cannot establish a speedup.

## Stop and adoption boundary

A parity failure, contract mismatch, duplicate or overwritten shared payload,
failed restart hit, or write error stops format-4 adoption. Passing establishes
an exact restart/partial-prefix/sharing contract for this checkpoint and
workload. It does not prove physical per-layer KV-delta deduplication, concurrent
multi-request safety, cold-prefill improvement, or adoption-quality speedup.

## Result

The first full-model attempt exposed a real snapshot-isolation defect before an
artifact was written: `MXFP8PoolingCache.persistence_state()` copied scalar
lengths but retained the mutable `_chunks` and `_index_chunks` list objects.
Later prefill appended to those lists, so the restored 128-token checkpoint had
128-token scalar state with longer chunk collections. The restart path failed
an attention-mask shape check. The implementation now snapshots both
collections as new immutable lists, and regression tests cover completed chunks
and active remainders independently.

The corrected gate used two 453-token prompts with 442 exact common prefix
tokens. Results were:

| Request | Reused | Output token | Output hash | Logical expert bytes | Request | TTFT |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Runtime A cold branch A | 0 | 36,363 | `6c094c63…20d51` | 146,902,351,872 | 28.487 s | 28.466 s |
| Runtime B restart branch B | 128 | 36,363 | `6c094c63…20d51` | 125,070,213,120 | 23.948 s | 23.918 s |
| Runtime C isolated cold branch B | 0 | 36,363 | `6c094c63…20d51` | 146,848,874,496 | 29.537 s | 29.536 s |

The restart and isolated-cold branch produced identical token IDs and SHA-256.
The 128-token shared payload was 7,002,850 bytes and its metadata was 2,148
bytes. Data SHA-256 `e736a83e…f0e9e` and metadata SHA-256
`ecadf6d1…6fac54` were unchanged across the second request; exactly one payload
owned that token/contract key, while the access sidecar reuse count changed from
0 to 1. Every metadata-contract and write criterion passed.

The request times are descriptive only. The three runs are not balanced for OS
cache or repeated-wave statistics, and persistent-cache acquisition occurs
before runtime metrics begin. The result therefore validates exact partial
restart reuse and immutable sharing, not a 19% speedup.

Artifact:
[`2026-08-27-block-prompt-cache-partial-restart-m2-max.json`](../docs/benchmarks/2026-08-27-block-prompt-cache-partial-restart-m2-max.json),
SHA-256
`678034d8a68278f87a9071fa65f662140a8479e9d4c591e0f189c5db3d367620`.

## Decision

Adopt normal prompt-cache format 4 as the exact persistent format. Keep the two
memory timelines and eight-payload default. Reuse-count/access-recency eviction
is active for normal format-4 entries. DSpark format 3 and its default-off
contract are unchanged and remain isolated.

This milestone shares immutable cumulative cache checkpoints addressed by a
token-block chain. It does not split each layer's KV state into independently
deduplicated delta objects; that stronger storage optimization would require a
new schema, byte-amplification measurement, crash-consistency gate, and proof
that reconstruction does not add more TTFT than it saves.
