# Expert-file cache-bypass protocol

Status: implemented and measured. The installed-range contract passed; the
fixed and adaptive full-model candidates both failed the predeclared
performance gate. The runtime default remains `cached`.

## Question

The page-residency proxy closes logical resident／nonresident accounting, but
the current account cannot run the system-wide `purge` command: macOS returns
`Unable to purge disk buffers: Operation not permitted`. A repeated comparison
must therefore not label an uncontrolled first run as cold.

This gate asks whether a per-descriptor expert-file cache-bypass mode can give
the target, DSpark draft, and hash-prefetch paths a repeatable storage policy
without changing model output. It is an experiment policy, not a new default.

## Local platform contract

The local macOS 26.6.2 `fcntl(2)` manual defines `F_NOCACHE` as turning data
caching off for a descriptor and `F_RDAHEAD` with zero as disabling read-ahead.
The local SDK assigns them command values 48 and 45. Apple XNU at pinned source
revision
[`f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`](https://github.com/apple-oss-distributions/xnu/tree/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea)
maps the descriptor flags to `IO_NOCACHE`／`IO_RAOFF` and selects the direct-read
path for aligned user-space buffers:

- [`F_NOCACHE` descriptor handling](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_descrip.c#L3600-L3618)
- [`IO_NOCACHE` read dispatch](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vfs/vfs_vnops.c#L1133-L1147)
- [direct versus cached cluster-read selection](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vfs/vfs_cluster.c#L4605-L4640)

The installed model is on APFS with a 4,096-byte device block. Its 13,369,344
byte expert blob and all six expert regions are 16,384-byte aligned. The runtime
must additionally reject cache-bypass reads if an MLX destination address,
file offset, or iovec length is not aligned to the local filesystem block size.
An accepted `fcntl` call alone is not enough evidence that a particular read
used the direct path.

`F_NOCACHE` does not provide a supported per-file purge operation and this gate
does not claim that it evicts pages resident before descriptor configuration.
The existing `mincore` probe remains the source for pre-read residency labels.

## Runtime contract

Add one explicit expert-file policy:

- `cached`: current default; do not change descriptor cache flags.
- `bypass`: set `F_NOCACHE=1` and `F_RDAHEAD=0` on every main and DSpark expert
  descriptor, and validate direct-I/O alignment before reading.

Common tensors, tokenizer files, prompt-cache files, and benchmark raw output
remain on their normal cache policy. The App and server default stay `cached`.
The bypass policy must be visible in CLI JSON and `/api/status`.

## Predeclared installed-range contract gate

A reproducible script will locate at least six fully nonresident expert blobs
using the existing page-residency probe. For each selected range it will:

1. read the blob through an aligned `F_NOCACHE`／no-read-ahead descriptor;
2. sample residency immediately afterward;
3. read the same blob through a normal descriptor;
4. sample residency again and compare byte hashes.

The contract passes only if:

1. every selected range is fully nonresident before its first read;
2. cache-bypass and cached byte counts equal one expert blob;
3. their SHA-256 hashes are identical;
4. the cache-bypass read leaves every selected byte nonresident;
5. the subsequent cached read makes every selected byte resident;
6. all `fcntl`, read, and residency operations succeed.

Any failed condition stops the full-model wave. Process disk counters are
recorded but remain process-wide signals, not expert-file device counters.

## Predeclared full-model repeated gate

If the installed-range contract passes, run the fixed 4,096-token
`random_hex` discovery prompt with 32 greedy output tokens. Every run uses a
fresh process, persistent prompt cache off, layer-major prefill off,
`expert_file_cache_policy=bypass`, and research-only DSpark fallback disabled.

The uninstrumented performance sequence is a three-wave Latin square:

```text
wave 1: normal, fixed, adaptive
wave 2: fixed, adaptive, normal
wave 3: adaptive, normal, fixed
```

`fixed` and `adaptive` use hybrid v3 target verification and exact hash
prefetch; `adaptive` also enables the storage-aware block selector. A final
`normal / fixed / adaptive` observer wave enables the page-residency probe.
Observer-wave timing is never mixed into the performance medians.

Correctness and accounting require:

1. all 12 generated token-ID sequences are exact;
2. every performance mode has three successful runs;
3. every observer run has zero probe failures and zero unclassified bytes;
4. main, draft, hash-prefetch, and useful／wasted identities close;
5. source, environment, workload, cache policy, output hash, memory, logical
   bytes, process disk bytes, and per-committed-token bytes are recorded.

A candidate can be considered for a later adoption gate only if its
uninstrumented median also satisfies all of:

- request time at least 5% below normal;
- end-to-end decode throughput at least 5% above normal;
- peak memory no more than 15% above normal;
- process disk bytes per generated token no more than 5% above normal;
- total speculative logical bytes per committed token no more than 5% above
  the fixed candidate when comparing adaptive with fixed.

Failing a threshold rejects only this exact candidate and protocol. It does not
turn the bypass policy into a runtime default, and it does not prove that a
kernel, filesystem, or storage-controller cache was absent below the XNU
direct-I/O boundary.

## Implementation result

The runtime now exposes `expert_file_cache_policy=cached|bypass`. `cached`
remains the default. On Darwin, `bypass` configures every main and DSpark expert
descriptor with `F_NOCACHE=1` and `F_RDAHEAD=0`, then rejects any read whose
destination address, file offset, or iovec length is not aligned to the APFS
4,096-byte allocation block. CLI metrics and `/api/status` report both the
policy and the direct-I/O alignment.

Unit tests cover the descriptor calls, cached no-op, invalid policy, alignment
rejection, CLI/server defaults, status reporting, and both benchmark decision
helpers. This is a research control; the App does not opt into it.

## Installed-range result

The first pilot selected all six initially nonresident ranges before executing
any control reads. A preceding cached control read then read ahead into layer 42
expert 2 before its turn. Five of six ranges passed, but the sixth was already
resident at the required just-before-read checkpoint. This is a selection race,
not evidence that `F_NOCACHE` populated that range. The failed pilot is retained
at
[`2026-08-27-expert-file-cache-bypass-selection-race-pilot-m2-max.json`](../docs/benchmarks/2026-08-27-expert-file-cache-bypass-selection-race-pilot-m2-max.json).

The corrected protocol rechecks and selects a fully nonresident range
immediately before each bypass read. It measured layer 42 experts 12, 15, 16,
19, 20, and 24. All six 13,369,344-byte reads had exact cached/bypass SHA-256
parity. Every range was 0 bytes resident before and after the bypass read, then
13,369,344 bytes resident after the cached control read. Each bypass read also
recorded exactly 13,369,344 process disk-read bytes. The contract passed and is
stored at
[`2026-08-27-expert-file-cache-bypass-contract-m2-max.json`](../docs/benchmarks/2026-08-27-expert-file-cache-bypass-contract-m2-max.json)
with artifact SHA-256
`fd1a9eb680267321c264c00d8d397983a115e85bf0692bbc7b823304f2746f71`.

## Full-model result and decision

All nine uninstrumented performance runs and three observer runs completed.
Every mode produced the same 32-token sequence with SHA-256
`e76fb39a649d7997bdb1dcdace279117f6c7fc2ba02062f28c65389c85048570`.
All observer partitions closed with zero probe failures and zero unclassified
bytes.

| Mode | Request median | Decode median | Peak memory median | Process disk bytes/token median |
| --- | ---: | ---: | ---: | ---: |
| Normal | 85.864 s | 3.032 Tok/s | 26,799,005,168 B | 9,614,723,584 B |
| Fixed hybrid + hash | 101.487 s | 1.223 Tok/s | 29,227,496,140 B | 10,877,890,048 B |
| Adaptive hybrid + hash | 90.171 s | 2.256 Tok/s | 29,373,966,228 B | 9,673,751,808 B |

Relative to normal, fixed changed request time by +18.19%, decode throughput by
-59.67%, peak memory by +9.06%, and process disk bytes/token by +13.14%.
Adaptive changed the same metrics by +5.02%, -25.58%, +9.61%, and +0.61%.
Adaptive speculative logical bytes per committed token were 42.96% below fixed,
but that byte saving did not overcome the target-math cost. Neither candidate
met the required request-time or throughput improvement, and fixed also failed
the disk-bytes/token limit.

The exact fixed and adaptive candidates are therefore rejected under this
4K/32 bypass protocol. DSpark, hybrid verification, hash prefetch, and adaptive
selection remain default off; `expert_file_cache_policy` remains `cached` by
default. The complete artifact is
[`2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json`](../docs/benchmarks/2026-08-27-cache-bypass-dspark-random-hex-4k32-m2-max.json)
with SHA-256
`0810de125a00b5edfe3720b3170226318dd6ef00d0245c5d6f44b436813ca36e`.

This negative result applies only to this installed model, machine, workload,
output length, verifier implementations, and descriptor policy. It does not
claim a system-wide cold cache or reject future verifiers with less target
math.
