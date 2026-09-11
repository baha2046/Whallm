# Upstream provenance

This package is derived from `PipeNetwork/deepseek-v41-mlx` at commit
`8bdd543c7160800f3451d9595c996129b7abecdf`, distributed under Apache-2.0.
The unmodified upstream license is stored in `THIRD_PARTY_LICENSE.txt`.

Whallm omits the upstream conversion, loading, generation, and streaming
entrypoints because those responsibilities are owned by Whallm. Local runtime
adaptations add checkpoint-native quantized output projection support, bounded
cache growth, and SSD-backed routed experts and Engram tables.
