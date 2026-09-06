"""Opt-in Qwen section probe; synchronized times are diagnostic, not A/B results.

Run with PYTHONPATH=runtime and pass ordinary CLI arguments after --.
Captures bounded real inputs for reproducible component work, without changing
the production runtime. GDN algorithm: https://arxiv.org/abs/2412.06464.
QSA block contract: https://arxiv.org/abs/2608.30320.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm.models import qwen3_5
from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.ane_prefill import ANEPrefillController


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("cli", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cli or args.cli[0] != "--":
        parser.error("CLI arguments must follow --")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    counters = defaultdict(int)
    samples = []
    ane_status = []
    original_close = ANEPrefillController.close

    def close(controller):
        ane_status.append(controller.snapshot())
        return original_close(controller)

    ANEPrefillController.close = close

    def save(name, arrays, metadata):
        path = args.output / (name + ".safetensors")
        if path.exists():
            return
        mx.save_safetensors(str(path), arrays)
        samples.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **metadata})

    original_gdn = qwen3_5.gated_delta_update

    def gdn(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, **kw):
        phase = "prefill" if q.shape[1] > 1 else "decode"
        arrays = dict(q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias)
        if state is not None:
            arrays["state"] = state
        if mask is not None:
            arrays["mask"] = mask
        before = time.perf_counter()
        mx.eval(*arrays.values())
        ready = time.perf_counter()
        result = original_gdn(q, k, v, a, b, A_log, dt_bias, state, mask, **kw)
        mx.eval(*result)
        ended = time.perf_counter()
        records.append(dict(section="gdn_recurrence", phase=phase, tokens=q.shape[1], upstream_wait=ready-before, seconds=ended-ready))
        key = (phase, q.shape[1])
        ordinal = counters[key]
        if (phase == "prefill" and q.shape[1] == 1024 and ordinal in (0, 12, 35)) or (phase == "decode" and ordinal == 0):
            save(f"gdn-{phase}-{ordinal}", {**arrays, "output":result[0], "final_state":result[1]}, dict(section="gdn", phase=phase, ordinal=ordinal))
        counters[key] += 1
        return result

    original_qsa = qwen.QSAAttention._bounded_attention

    def qsa(self, query, key, value, index_query, raw_index_keys, offset):
        phase = "prefill" if query.shape[2] > 1 else "decode"
        arrays = dict(query=query, key=key, value=value, index_query=index_query, raw_index_keys=raw_index_keys)
        before = time.perf_counter()
        mx.eval(*arrays.values())
        ready = time.perf_counter()
        result = original_qsa(self, query, key, value, index_query, raw_index_keys, offset)
        mx.eval(result)
        ended = time.perf_counter()
        records.append(dict(section="qsa_bounded", phase=phase, tokens=query.shape[2], context=key.shape[2], upstream_wait=ready-before, seconds=ended-ready))
        if key.shape[2] >= 4096:
            save(f"qsa-{phase}", {**arrays, "output":result, "index_norm_weight":self.indexer.k_layernorm.weight}, dict(section="qsa", phase=phase, offset=offset))
        return result

    original_lookup = qwen.NGramStore.lookup

    def lookup(self, row_ids):
        before = time.perf_counter()
        result = original_lookup(self, row_ids)
        mx.eval(result)
        records.append(dict(section="ngram_lookup", phase="prefill" if row_ids.shape[1] > 1 else "decode", seconds=time.perf_counter()-before))
        return result

    qwen3_5.gated_delta_update = gdn
    qwen.QSAAttention._bounded_attention = qsa
    qwen.NGramStore.lookup = lookup
    cli = args.cli[1:]
    sys.argv = ["deepseek_v4_ssd.cli", *cli]
    status = "failed"
    try:
        runpy.run_module("deepseek_v4_ssd.cli", run_name="__main__")
        status = "completed"
    finally:
        totals = defaultdict(float)
        for record in records:
            totals[record["phase"] + "/" + record["section"]] += record["seconds"]
        payload = dict(status=status, evidence_kind="synchronized diagnostic section times", formal_performance_result=False,
            environment=dict(platform=platform.platform(), chip=subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"],text=True).strip()),
            command=cli, ane_status=ane_status, samples=samples, totals=dict(totals), records=records,
            limitations=["Input and output synchronization changes overlap and graph execution; not normal runtime timing.", "Upstream waits include other lazy work and must not be attributed to the measured kernel.", "Input capture writes occur outside section timers but inside the request; no speed claim."])
        (args.output / "profile.json").write_text(json.dumps(payload, indent=2)+"\n")


if __name__ == "__main__":
    main()
