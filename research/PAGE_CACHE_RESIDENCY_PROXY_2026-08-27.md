# Expert-file page-cache residency proxy

Status: research-only probe contract, short composition, and 4K useful/wasted
coverage gates pass. The probe remains default off and is not a physical SSD
byte counter.

## Question

PLAN P0 needs physical storage bytes, operating-system page-cache behavior,
runtime cache hits, useful/wasted speculative prefetch, and bytes per committed
token to remain distinguishable. The runtime already records logical expert
bytes and Darwin process-wide disk counters, but neither identifies whether a
specific expert-file range was resident before its read.

This candidate adds an optional `mincore(2)` sample immediately before every
expert `preadv`. For the exact byte range actually returned by `preadv`, it
records:

- bytes whose containing VM page was resident before the read;
- bytes whose containing VM page was nonresident before the read;
- bytes that could not be classified and probe failures;
- the same partition for useful and wasted exact hash-prefetch reads.

The page-size boundary is handled by byte overlap, so partial first and last
pages do not inflate the logical byte total. The probe is disabled by default
because its `mmap`/`mincore`/`munmap` calls are observer overhead.

## Evidence boundary

`mincore` reports current virtual-memory page residency. Nonresident bytes are
a page-cache-miss proxy, not measured physical SSD bytes. APFS behavior,
storage-controller caches, request coalescing, speculative kernel read-ahead,
and concurrent process I/O remain outside this attribution. Darwin
`proc_pid_rusage` stays as a separate process-wide counter and is not relabeled
as expert-file traffic.

The exact invariants are:

```text
resident_before_read + nonresident_before_read = classified
classified + unclassified = logical expert bytes returned by preadv
```

For exact hash prefetch, the useful/wasted resident and nonresident partitions
must also sum to their corresponding read classifications.

## Predeclared installed-model gate

[`Scripts/benchmark_page_cache_probe.py`](../Scripts/benchmark_page_cache_probe.py)
will run the installed pinned checkpoint with:

- the fixed 128-token `code` prompt from the calibration manifest;
- 32 greedy output tokens;
- fresh-process `control / probe / probe / control` order;
- persistent prompt cache, layer-major prefill, and DSpark disabled;
- operating-system page cache explicitly labeled uncontrolled.

The probe contract passes only if:

1. all four generated token-ID sequences are exact;
2. every probe run has zero probe failures and zero unclassified bytes;
3. resident plus nonresident bytes exactly equals classified bytes;
4. classified plus unclassified bytes exactly equals logical expert bytes.

Any failure stops this proxy. Request-time changes are recorded only to expose
observer overhead and cannot be used as runtime performance evidence. Passing
permits a DSpark hybrid-v3 + hash-prefetch trace that partitions useful and
wasted logical prefetch bytes by pre-read page residency; it still does not
satisfy the PLAN requirement for a true physical SSD byte counter.

## Installed-model gate result

The result is
[`2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json`](../docs/benchmarks/2026-08-27-expert-page-cache-probe-code-128x32-m2-max.json).
All four 32-token sequences have hash
`104ee3cbefed6251682b4f1d8ff48ba6ea918af70cf811bc3ada33e1281ca002`.
Each probe run classifies all 90,323,288,064 logical expert bytes with 6,756
successful calls, zero failures, and zero unclassified bytes.

| Probe run | Resident before read | Nonresident before read | Process-wide disk read |
| --- | ---: | ---: | ---: |
| `code-02-probe` | 29,027,827,712 B | 61,295,460,352 B | 61,617,750,016 B |
| `code-03-probe` | 28,959,195,136 B | 61,364,092,928 B | 61,570,039,808 B |

The median pre-read partition is 32.10% resident and 67.90% nonresident. Its
close-but-not-equal relationship with the process counter is expected and is
evidence that the two measures must remain separate. Probe request time is
3.89% above the control median and reported decode throughput is 10.77% lower;
those values describe observer overhead only.

## Predeclared hash-prefetch composition gate

The next artifact uses the same `code` 128/32 workload and fresh-process
`normal / hash / hash / normal` order. All runs enable the probe. Hash runs use
the complete DSpark graph, hybrid v3 target verification, exact hash prefetch,
and research-only no-fallback. The composition passes only if:

1. all four token-ID sequences are exact;
2. main target and DSpark draft logical bytes each close against their page
   residency partitions with zero failures and zero unclassified bytes;
3. hash-prefetch classified plus unclassified bytes equals logical prefetch
   bytes;
4. useful plus wasted bytes close independently for logical, resident,
   nonresident, and unclassified partitions.

The gate reports both logical useful rate and nonresident-byte useful rate. It
does not set a speed or adoption threshold because cache state is uncontrolled
and the probe itself changes timing.

## Short composition result and 4K coverage gate

The short composition result is
[`2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-hash-page-cache-code-128x32-m2-max.json).
All four sequences match the same normal hash, and every main, draft, prefetch,
useful, and wasted accounting identity closes with zero unclassified bytes.
Each hash run reads 4,812,963,840 prefetch bytes: 2,406,875,136 were resident
and 2,406,088,704 were nonresident before read. This workload uses every
prefetched expert, so logical and nonresident useful rates are both 100% and the
wasted partition is zero.

To exercise both sides of the partition, the next predeclared gate uses the
4,096-token `balanced_choice` discovery prompt, where the earlier logical-only
artifact recorded the lowest useful rate. It runs one fresh normal/hash pair,
up to 32 greedy output tokens, with the same hybrid v3, exact hash prefetch,
probe, and no-fallback configuration. In addition to the four accounting rules
above, both useful and wasted logical prefetch bytes must be nonzero. A zero
partition or any token/accounting failure stops the coverage claim. Cache state
remains uncontrolled and timing remains out of scope.

## 4K useful/wasted coverage result

The result is
[`2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json`](../docs/benchmarks/2026-08-27-dspark-hash-page-cache-balanced-choice-4k32-m2-max.json).
Normal and hash produce the same five-token hash,
`80f5773c67f8e4b9cad8ff41f6fe7a064f61c1b2c821cfb7cd4fcc437039c8c7`.
All main, draft, prefetch, useful, and wasted partitions close with zero probe
failure and zero unclassified bytes.

| Hash-prefetch partition | Logical bytes | Resident before read | Nonresident before read |
| --- | ---: | ---: | ---: |
| Useful | 802,160,640 | 120,455,168 | 681,705,472 |
| Wasted | 2,607,022,080 | 989,462,528 | 1,617,559,552 |
| Total | 3,409,182,720 | 1,109,917,696 | 2,299,265,024 |

The logical useful rate is 23.53%, while the nonresident-byte useful rate is
29.65%. This difference is expected: a larger fraction of wasted logical bytes
was already resident before the read. More importantly, 1,617,559,552 wasted
bytes were still nonresident at the read boundary, which is directly visible
to this proxy but not to logical accounting alone.

## Decision

The page-residency proxy is contract-complete for the measured scope and may be
used in future exploratory gates. It remains research-only and default off.
P0 physical-I/O work is still partial because `mincore` cannot observe true
device bytes. Before judging a runtime candidate, run repeated waves with an
explicit cache-state protocol and retain process-wide disk counters, output
hashes, memory, and observer-overhead labels alongside the proxy.
