# Benchmark

Recorded measurements for v1.1.7, v1.1.4, and v1.1.0. Each section states its
workload and measurement conditions.

## v1.1.7

These results were recorded with the built-in **Throughput** benchmark using
**Code** context, input lengths of **1,024, 4,096, 8,192, and 16,384 tokens**,
and an output limit of **128 tokens**. The same rows appear in the
[README benchmark tables](README.md#benchmarks).

Each row is an individual measurement, not a P95 statistic. Build revisions
and cache state were not recorded alongside these rows, so they are reference
results rather than a controlled comparison of acceleration settings.
The workload and output limit differ from the older API benchmarks below;
no cross-version improvement percentages are calculated for these runs.

TTFT is the time to the first token, in milliseconds. Prefill measures input
processing and Decode measures output generation, both in tokens per second.
Peak MLX is MLX allocation in **GiB**, not process RSS or total Mac memory.
The app export labels this value GB but divides bytes by 1024³.

### M5 Pro

| Model | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 | 1152 | 1024 | 19112.3 | 53.6 | 7.7 | 22.77 |
| DeepSeek V4 | 1152 | 4096 | 24498.6 | 167.2 | 5.9 | 22.80 |
| DeepSeek V4 | 1152 | 8192 | 42137.5 | 194.4 | 6.8 | 22.83 |
| DeepSeek V4 | 1152 | 16384 | 81517.9 | 201.0 | 6.3 | 22.90 |
| Qwen3.8 | 3072 | 1024 | 10336.3 | 99.1 | 10.6 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 28831.0 | 142.1 | 9.2 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 53303.2 | 153.7 | 10.1 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 112353.1 | 145.8 | 8.5 | 17.18 |

### M2 Max

| Model | Slots | Input tokens | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak MLX (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 3072 | 1024 | 14016.3 | 73.1 | 9.0 | 16.86 |
| Qwen3.8 | 3072 | 4096 | 40902.4 | 100.1 | 7.6 | 16.92 |
| Qwen3.8 | 3072 | 8192 | 79125.0 | 103.5 | 8.3 | 17.01 |
| Qwen3.8 | 3072 | 16384 | 158838.7 | 103.1 | 7.1 | 17.18 |
| DeepSeek V4.1 | 1152 | 1024 | 73426.4 | 13.9 | 2.2 | 32.05 |
| DeepSeek V4.1 | 1152 | 4096 | 106742.4 | 38.4 | 1.8 | 32.11 |
| DeepSeek V4.1 | 1152 | 8192 | 144011.4 | 56.9 | 2.1 | 32.19 |
| DeepSeek V4.1 | 1152 | 16384 | 248481.9 | 65.9 | 1.9 | 32.75 |

## v1.1.4

Run on a MacBook Pro with Apple M5 Pro, 64 GB of unified memory, and 1 TB of
storage.

The benchmark used mixed SPEED-Bench prompts through `/v1/completions`, three
runs per input size, and a 64-token output limit. The prompt cache was not
cleared between runs, but every run used a different input and reused zero
prompt tokens. The filesystem cache was not purged. With three runs,
nearest-rank P95 equals the maximum.

Percentages in parentheses show the improvement over v1.1.0. For total time,
TTFT, and memory, improvement is `(v1.1.0 − v1.1.4) / v1.1.0`. For prefill
and decode throughput, improvement is `(v1.1.4 − v1.1.0) / v1.1.0`. Positive
values are improvements and negative values are regressions. The v1.1.0
baseline used two runs per size, so these percentages are a descriptive release
comparison, not a paired causal measurement.

### DeepSeek V4 Flash 0731

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 27.10 s(+42.0%) | 17.75 s(+52.4%) | 59.9 tok/s(+188.7%) | 7.8 tok/s(+15.6%) | 32.84 GiB(−42.6%) |
| 2,048 | 27.35 s(−1.0%) | 17.60 s(−4.0%) | 117.3 tok/s(+8.5%) | 7.3 tok/s(+11.7%) | 33.26 GiB(−0.2%) |
| 8,192 | 50.12 s(−5.2%) | 40.47 s(−6.6%) | 206.3 tok/s(−1.5%) | 7.4 tok/s(+10.3%) | 34.50 GiB(+0.7%) |
| 16,384 | 88.27 s(−2.6%) | 78.61 s(−3.0%) | 209.0 tok/s(−1.5%) | 7.2 tok/s(+8.0%) | 35.70 GiB(−0.2%) |

### Qwen3.8 Next Flash FP8

v1.1.4 increases the slot count from 1,152 to 4,096 to improve the expert cache
hit rate, at the cost of higher memory use.

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 22.60 s(+7.6%) | 15.65 s(+9.0%) | 69.2 tok/s(+15.7%) | 10.4 tok/s(+11.6%) | 20.92 GiB(−37.7%) |
| 2,048 | 31.42 s(+22.3%) | 24.90 s(+23.7%) | 87.5 tok/s(+34.7%) | 9.8 tok/s(+18.1%) | 21.26 GiB(−29.6%) |
| 8,192 | 84.88 s(+40.4%) | 77.74 s(+42.2%) | 111.9 tok/s(+81.4%) | 10.4 tok/s(+25.0%) | 21.90 GiB(−22.3%) |
| 16,384 | 157.89 s(+42.9%) | 150.33 s(+43.9%) | 113.3 tok/s(+83.4%) | 9.7 tok/s(+23.9%) | 22.75 GiB(−20.3%) |

## v1.1.0 baseline

The v1.1.0 benchmark used the same MacBook Pro, endpoint, dataset files, input sizes, and
64-token output limit, with two runs per input size. With two runs,
nearest-rank P95 also equals the maximum.

### DeepSeek V4 Flash 0731

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 46.72 s | 37.26 s | 20.7 tok/s | 6.7 tok/s | 23.03 GiB |
| 2,048 | 27.08 s | 16.91 s | 108.1 tok/s | 6.6 tok/s | 33.19 GiB |
| 8,192 | 47.66 s | 37.96 s | 209.5 tok/s | 6.7 tok/s | 34.74 GiB |
| 16,384 | 86.01 s | 76.34 s | 212.3 tok/s | 6.7 tok/s | 35.64 GiB |

### Qwen3.8 Next Flash FP8

| Input tokens | P95 total time | P95 TTFT | P95 prefill | P95 decode | Peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 24.45 s | 17.20 s | 59.8 tok/s | 9.3 tok/s | 15.19 GiB |
| 2,048 | 40.45 s | 32.63 s | 65.0 tok/s | 8.3 tok/s | 16.41 GiB |
| 8,192 | 142.39 s | 134.44 s | 61.7 tok/s | 8.3 tok/s | 17.90 GiB |
| 16,384 | 276.41 s | 267.88 s | 61.8 tok/s | 7.8 tok/s | 18.92 GiB |

Performance changes with the prompt, SSD speed, cache state, and runtime
configuration. Memory is peak MLX active memory observed during each request,
not process RSS. These API benchmarks are exploratory measurements and do not
claim model-quality results.
