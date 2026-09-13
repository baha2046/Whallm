"""Isolated Qwen CLI wrapper; production files/defaults are never patched on disk.

Initial integration supports one existing Prefill chunk (up to 1024 tokens).
Longer/multi-chunk inputs use the original path BEFORE allocating any banks.
Observer mode adds synchronization/copies and is correctness evidence only.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import resource
import runpy
import sys
import time
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import generation, qwen4_exp as qwen
from deepseek_v4_ssd.model_support import qwen as qwen_support
from deepseek_v4_ssd.ane_prefill import ANEPrefillController
from research.ssd_prefill_pipeline import ExpertBanks, qwen_pipeline, workspace_bound


def tensor_hash(value):
    mx.eval(value)
    # Casting BF16 to FP32 is lossless; this works without NumPy BF16 support.
    raw = np.asarray(value.astype(mx.float32) if value.dtype == mx.bfloat16 else value)
    return hashlib.sha256(raw.tobytes()).hexdigest()


@contextmanager
def install_pipeline(mode, *, observe_tensors=False):
    """Scoped research dispatch for one serialized runtime; always restore hooks."""
    if mode not in ("control", "pipeline"):
        raise ValueError("unknown pipeline mode")
    events, tensors, ane = [], [], []
    original_prefill = qwen_support._qwen_layer_major_prefill
    original_experts = qwen.StreamingExperts.__call__
    original_close = ANEPrefillController.close
    original_load = generation.load_model
    calls = dict(pipeline=0, original=0)

    def observe(self, value, indices, result):
        if observe_tensors and value.shape[1] > 1:
            tensors.append(dict(layer=self.layer, shape=list(value.shape),
                value=tensor_hash(value), indices=tensor_hash(indices), output=tensor_hash(result),
                finite=bool(mx.isfinite(result).all().item())))
        return result

    def normal(self, value, indices):
        calls['original'] += 1
        return observe(self, value, indices, original_experts(self, value, indices))

    def prefill(model, token_ids, prompt_cache, step_size, expert_cache, next_layer_prefetch=False):
        count = len(token_ids)
        bound = workspace_bound(expert_cache.model, count, expert_cache.model.selected_expert_count, 32)
        eligible = (mode == 'pipeline' and 0 < count <= min(1024, step_size)
            and not next_layer_prefetch and expert_cache.file_cache_policy == 'cached'
            and expert_cache._read_limiter is None
            and expert_cache.expert_directory == expert_cache.model.root / 'experts'
            and sum(bound[k] for k in
                ('banks_bytes', 'temporary_array_allowance', 'routing_allowance')) <= bound['baseline_layer_payload_bytes'])
        events.append(dict(prefill_tokens=count, step_size=step_size, eligible=eligible, workspace=bound))
        if not eligible:
            return original_prefill(model, token_ids, prompt_cache, step_size, expert_cache, next_layer_prefetch)

        @contextmanager
        def streamed_layer(layer):
            # Exactly the original layer traversal; no full-layer allocation.
            yield None

        with ExpertBanks(expert_cache.model, 32, bound['baseline_layer_payload_bytes']) as banks:
            total_waves = total_bytes = 0
            def streamed(self, value, indices):
                nonlocal total_waves, total_bytes
                if self.cache is not expert_cache:
                    return normal(self, value, indices)
                calls['pipeline'] += 1
                result = qwen_pipeline(value, indices, banks, self.layer)
                rows = banks.rows
                bytes_read = sum(r['bytes_read'] for r in rows)
                total_waves += len(rows)
                total_bytes += bytes_read
                expert_cache.metrics.bytes_read += bytes_read
                expert_cache.metrics.gather_qmm_calls += 2 * len(rows)
                # Summed worker service time is separate from overlapped wall time.
                expert_cache.metrics.read_seconds += sum(r['read_complete'] - r['read_start'] for r in rows)
                return observe(self, value, indices, result)

            with patch.object(expert_cache, 'batched_layer', streamed_layer), \
                 patch.object(qwen.StreamingExperts, '__call__', streamed):
                result = original_prefill(model, token_ids, prompt_cache, step_size, expert_cache, False)
            events.append(dict(waves=total_waves, bytes_read=total_bytes))
            return result

    def close(controller):
        ane.append(controller.snapshot())
        return original_close(controller)

    def load(*a, **kw):
        started = time.perf_counter()
        try:
            return original_load(*a, **kw)
        finally:
            events.append(dict(model_load_seconds=time.perf_counter() - started))

    with patch.object(qwen_support, '_qwen_layer_major_prefill', prefill), \
         patch.object(generation, 'load_model', load), \
         patch.object(qwen.StreamingExperts, '__call__', normal), \
         patch.object(ANEPrefillController, 'close', close):
        yield dict(calls=calls, events=events, tensors=tensors, ane=ane)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', choices=['control', 'pipeline'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('cli', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cli or args.cli[0] != '--' or '--no-persistent-prompt-cache' not in args.cli:
        parser.error('ordinary CLI after -- must disable persistent prompt cache')
    if '--mtp' in args.cli or '--dspark' in args.cli:
        parser.error('speculative generation is outside this first Prefill gate')
    args.output.mkdir(parents=True, exist_ok=False)
    status = 'failed'
    state = {}
    try:
        with install_pipeline(args.mode, observe_tensors=args.observe) as state:
            sys.argv = ['deepseek_v4_ssd.cli', *args.cli[1:]]
            runpy.run_module('deepseek_v4_ssd.cli', run_name='__main__')
        status = 'completed'
    finally:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        (args.output / 'status.json').write_text(json.dumps(dict(status=status, mode=args.mode,
            observer=args.observe, formal_performance_result=False,
            process_peak_rss_bytes=rss, **state), indent=2) + '\n')


if __name__ == '__main__':
    main()
