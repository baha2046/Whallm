# MTLIO expert-streaming ownership microbenchmark

Status: research-complete native feasibility gate. Native byte, cancellation,
and event-ordering checks passed; runtime integration stopped because the
installed public MLX surface cannot express the required external-event
dependency. Runtime defaults are unchanged.

## Question

The current runtime uses four read workers and `preadv` into fixed MLX-visible
expert slots. The PLAN proposes MTLIO, fixed Metal buffers, and explicit fences,
but platform capability alone does not establish a usable MLX ownership path or
a local speedup.

This gate asks four separate questions:

1. Can MTLIO load the canonical 13,369,344-byte installed expert range exactly
   into CPU bytes, a shared `MTLBuffer`, and a private `MTLBuffer`?
2. Can an `MTLSharedEvent` order an I/O load before a GPU command that consumes
   the destination, without a CPU wait between the queues?
3. What copy is required to validate or hand off shared and private resources?
4. Does the installed MLX 0.32.0 API expose a supported way to adopt the Metal
   resource and the external shared-event dependency without copying or
   reaching into private implementation state?

The official API contract used by the probe is limited to Apple's
[`MTLIOCommandQueue`](https://developer.apple.com/documentation/metal/mtliocommandqueue),
[`MTLIOCommandBuffer`](https://developer.apple.com/documentation/metal/mtliocommandbuffer),
and
[`resource loading`](https://developer.apple.com/documentation/metal/resource-loading)
documentation plus the installed macOS SDK headers. Those sources establish
load, barrier, cancellation, status, and shared-event mechanisms; they do not
establish this project's performance or MLX interoperability.

## Native probe contract

Build one standalone Swift executable against the installed Xcode macOS SDK.
Use complete expert ranges from the verified installed model. Preallocate all
destinations before timing and compare:

- four-worker positional `preadv` into one fixed aligned CPU arena;
- MTLIO `loadBytes` into one fixed aligned CPU arena;
- MTLIO `loadBuffer` into one shared `MTLBuffer`;
- MTLIO `loadBuffer` into one private `MTLBuffer`, followed by an explicitly
  labeled GPU blit only for CPU byte validation.

For shared and private buffers, encode an I/O signal and GPU wait using the same
device's shared event. The GPU copies the loaded bytes into a CPU-visible
validation buffer only after the event. Record I/O completion, GPU-visible
completion, validation-copy time, command status/error, buffer storage mode,
and exact SHA-256.

Exercise 1, 6, 32, and 128 queued expert ranges. Six-range runs record
per-command p50/p95 latency; 32/128 runs record aggregate throughput. A
cancellation probe may finish before cancellation wins the race; it passes only
if status is `complete` with exact bytes or `cancelled` with the destination
explicitly treated as invalid. Error or partial admission fails.

## Cache and timing controls

System-wide purge is unavailable. MTLIO owns its file handle, so the runtime's
descriptor `F_NOCACHE` policy cannot be assumed to apply. The probe therefore:

- labels OS page cache `not purged`;
- never calls MTLIO timings a physical-SSD comparison;
- separates byte correctness, ownership, and synchronization results from
  exploratory timing;
- records method order, file/layer/ranges, SDK, device, source hashes, and raw
  per-command samples;
- does not change runtime defaults from timing alone.

## MLX handoff audit

Audit the installed MLX Python and public C++ headers. A supported runtime path
must provide both:

1. ownership of or a no-copy view over the loaded Metal resource; and
2. a supported dependency that prevents MLX GPU work from reading before the
   MTLIO signal.

Raw-pointer adoption without an external-event bridge is not sufficient.
Private-buffer throughput without an MLX resource bridge is not sufficient.
Any proposed private API or allocator-internal cast is a stop, not an
implementation plan.

## Stop and continuation criteria

Stop runtime integration when any of the following holds:

- any complete command has a byte mismatch, error, or GPU-visibility failure;
- cancellation can expose a non-complete destination as valid;
- shared mode requires an unaccounted copy;
- private mode requires a CPU staging copy before the target can use it;
- the installed public MLX surface cannot express both resource ownership and
  the MTLIO-to-MLX dependency;
- apparent speed depends on an uncontrolled cache-state ordering.

Continue to an isolated runtime prototype only if all correctness/lifetime
checks pass, a supported MLX handoff exists, and repeated balanced runs show at
least 10% lower p50 for one/six experts or 10% higher aggregate throughput for
32/128 experts without a p95 regression above 5%. The timing condition is
necessary but not sufficient; it is invalid when cache state is not balanced.

## Evidence boundary

Passing the native probe would validate only raw resource loading and queue
synchronization on this Mac. It would not prove target-model speed, physical
SSD bytes, MLX graph integration, ANE/GPU overlap, power savings, or a safe
runtime default. Failure rejects only the tested handoff and buffer modes.

## Execution and artifact

The predeclared probe was executed on 2026-08-27 with:

- Mac14,13, Apple M2 Max, 64 GiB unified memory;
- macOS 26.6.2, Xcode 26.6, macOS SDK 26.5;
- installed checkpoint revision
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`;
- canonical 13,369,344-byte expert blobs;
- MLX 0.32.0 and mlx-lm 0.31.3;
- OS page cache explicitly labeled `not purged`.

The machine-readable result is
[`2026-08-27-mtlio-expert-streaming-m2-max.json`](../docs/benchmarks/2026-08-27-mtlio-expert-streaming-m2-max.json).
Its SHA-256 is
`891d2b9635157837c6107f77b928be09063e391d1efa1a0cd410f6783074aaa9`.
The artifact embeds all 16 native run payloads, the cancellation payload,
method order, source/header hashes, and per-command samples. The 16 measured
runs cover 8,930,721,792 bytes before each probe's independent reference read.

Each row used a different installed layer file. `mincore` observed all selected
ranges as at least 99.9885% nonresident before the command, but the system cache
was not purged and the run was not repeated in balanced waves. Timing remains
exploratory and is not a physical-SSD or formal performance result.

## Native result

All 16 method/count rows completed with exact candidate and reference SHA-256.
Shared and private `MTLBuffer` rows signaled the shared event to value 1 and the
GPU validation command observed exact bytes after its encoded wait. Shared
targets were also exact through their direct CPU-visible pointer, so that mode
did not require the validation blit for CPU access. Private targets required the
explicit GPU blit to a shared validation buffer for CPU hashing. The cancel
probe returned `cancelled`, exposed no digest, and marked its destination not
admitted.

| Expert count | Method | p50 ms | p95 ms | Aggregate GiB/s | GPU validation blit ms |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | four-worker `preadv` | 3.188 | 3.188 | 3.824 | — |
| 1 | MTLIO bytes | 2.499 | 2.499 | 4.963 | — |
| 1 | MTLIO shared | 2.480 | 2.480 | 5.021 | 0.068 |
| 1 | MTLIO private | 4.158 | 4.158 | 2.995 | 0.037 |
| 6 | four-worker `preadv` | 7.514 | 8.365 | 5.968 | — |
| 6 | MTLIO bytes | 12.345 | 12.668 | 5.888 | — |
| 6 | MTLIO shared | 12.242 | 12.747 | 5.861 | 0.476 |
| 6 | MTLIO private | 17.695 | 21.024 | 3.553 | 0.436 |
| 32 | four-worker `preadv` | — | — | 5.960 | — |
| 32 | MTLIO bytes | — | — | 5.949 | — |
| 32 | MTLIO shared | — | — | 6.007 | 2.316 |
| 32 | MTLIO private | — | — | 5.636 | 2.276 |
| 128 | four-worker `preadv` | — | — | 6.054 | — |
| 128 | MTLIO bytes | — | — | 6.078 | — |
| 128 | MTLIO shared | — | — | 6.041 | 9.054 |
| 128 | MTLIO private | — | — | 6.026 | 9.037 |

The single-expert bytes/shared rows were directionally faster than `preadv`,
but the six-expert p95 values regressed by 51.44% and 52.38%. At 32 and 128
experts, MTLIO bytes/shared aggregate throughput remained within 0.78% of the
separate-file `preadv` baseline. Private mode was slower for the one-, six-, and
32-expert rows. These directions do not meet the repeated balanced timing gate
and are not adoption evidence.

## Installed MLX handoff audit

The installed Python surface has `mx.from_dlpack`, and an MLX-owned Metal
DLPack round trip with `copy=False` was exact. The shipped public C++ headers
also expose a raw-pointer no-copy candidate through `array(void*, ...)` and
`allocator::make_buffer(void*, size)`. This establishes a resource-view
candidate, not the required synchronization conjunction.

The same audit found:

- no direct public Metal buffer/event/fence import on `mx.metal`;
- no public `Event` or Metal `EventImpl` constructor that adopts an external
  `MTLSharedEvent`;
- the observed `mx.from_dlpack` consumer call invoked the provider with no
  positional or keyword stream/event dependency.

A provider could CPU-block before returning a DLPack capsule, but that would not
preserve the asynchronous I/O-to-MLX GPU dependency required by this gate.
Accessing allocator/backend implementation state to recover MLX's internal
event is explicitly outside the supported surface and triggers the stop rule.

## Decision

The native MTLIO mechanism is byte-correct on this Mac, including shared-event
GPU visibility and safe cancellation admission. The tested MLX 0.32.0 public
surface does not provide the other half of the contract: a supported external
event dependency paired with the resource handoff. Runtime integration is
therefore stopped, no MTLIO ownership path is added to `ModelRuntime`, and the
existing four-worker `preadv` slot path remains unchanged.

Reopen this exact direction only when an installed MLX release exposes a
supported external Metal resource plus event/stream handoff. A future run must
repeat balanced cache-state waves and recheck the 10% latency/throughput and 5%
p95 criteria before any isolated runtime prototype.
