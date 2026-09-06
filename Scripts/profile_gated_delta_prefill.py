"""Research-only synchronized timing of the installed Qwen recurrent kernel.

Runs the existing CLI with a process-local observer; never changes the runtime.
Synchronization disturbs overlap: these are component diagnostics, not speedups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    import mlx.core as mx
    from mlx_lm.models import gated_delta
    from deepseek_v4_ssd.cli import main as cli_main

    original = gated_delta.gated_delta_kernel
    calls = []
    samples = {}
    seen = set()

    def observe(q, k, v, g, beta, state, mask=None):
        inputs = dict(q=q, k=k, v=v, g=g, beta=beta, state=state)
        if mask is not None:
            inputs["mask"] = mask
        mx.eval(*inputs.values())
        mx.synchronize()
        shape = (tuple(q.shape), tuple(v.shape), str(q.dtype), str(state.dtype), mask is not None)
        started = time.perf_counter()
        result = original(q, k, v, g, beta, state, mask)
        mx.eval(*result)
        mx.synchronize()
        seconds = time.perf_counter() - started
        calls.append(dict(q_shape=list(q.shape), v_shape=list(v.shape), dtype=str(q.dtype),
                          state_dtype=str(state.dtype), g_shape=list(g.shape),
                          seconds=seconds, first_shape=shape not in seen))
        seen.add(shape)
        if q.shape[1] > 1 and shape not in samples:
            # Retain at most one real input per observed shape; serialize after generation.
            samples[shape] = inputs
        return result

    gated_delta.gated_delta_kernel = observe
    sys.argv = ["profile_gated_delta_prefill", *(args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args)]
    status = "failed"
    try:
        cli_main()
        status = "completed"
    finally:
        gated_delta.gated_delta_kernel = original
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        saved = []
        if status == "completed":
            for number, values in enumerate(samples.values()):
                path = args.profile_output.with_name(f"{args.profile_output.stem}-inputs-{number}.safetensors")
                mx.save_safetensors(str(path), values)
                saved.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        args.profile_output.write_text(json.dumps(dict(
            status=status, evidence_kind="synchronized_component_observer", formal_performance_result=False,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dependency_sha256=hashlib.sha256(Path(gated_delta.__file__).read_bytes()).hexdigest(),
            calls=calls, samples=saved,
            limits=["Synchronization removes overlap and adds overhead; observer wall time is not normal latency.",
                    "First call per shape may include compilation; report separately.",
                    "Retained sample arrays add memory; only one input per Prefill shape is saved.",
                    "Timer excludes upstream input computation, but includes dispatch and output synchronization.",
                    "T>1 is Prefill; T=1 includes the last prompt position and Decode."]
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
