"""Apply the registered QSA paired-screen gates without changing runtime defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(artifact: dict) -> dict:
    if artifact["status"] != "completed" or len(artifact["runs"]) != 20:
        raise ValueError("need all twenty paired runs")
    pairs = []
    for wave in (0, 1):
        cases = dict.fromkeys(r["case"] for r in artifact["runs"])
        for case in cases:
            rows = [r for r in artifact["runs"] if r["wave"] == wave and r["case"] == case]
            if len(rows) != 2 or not all(r["token_parity"] for r in rows):
                raise ValueError("missing pair or failed output parity")
            control = next(r for r in rows if r["chunk"] == 4)
            candidate = next(r for r in rows if r["chunk"] != 4)
            a, b = control["metrics"], candidate["metrics"]
            if a["generated_token_ids"] != b["generated_token_ids"] or len(a["generated_token_ids"]) != 256:
                raise ValueError("256-token paired outputs differ")
            pairs.append(dict(wave=wave, case=case,
                              ttft_reduction=1-b["time_to_first_token_seconds"]/a["time_to_first_token_seconds"],
                              decode_change=b["decode_tokens_per_second"]/a["decode_tokens_per_second"]-1,
                              p95_change=b["decode_latency_p95_seconds"]/a["decode_latency_p95_seconds"]-1,
                              mlx_memory_change=b["peak_memory_bytes"]/a["peak_memory_bytes"]-1,
                              rss_change=candidate["max_rss_bytes"]/control["max_rss_bytes"]-1,
                              token_sha256=a["token_sha256"]))
    metrics = ("ttft_reduction", "decode_change", "p95_change", "mlx_memory_change", "rss_change")
    per_case = [dict(case=case, **{key:statistics.median(p[key] for p in pairs if p["case"] == case) for key in metrics})
                for case in cases]
    main_ttft = statistics.median(c["ttft_reduction"] for c in per_case if not c["case"].startswith("repeated"))
    checks = dict(ttft=main_ttft >= .05,
                  decode=all(c["decode_change"] >= -.05 for c in per_case),
                  p95=all(c["p95_change"] <= .10 for c in per_case),
                  memory=all(c["mlx_memory_change"] <= .05 and c["rss_change"] <= .05 for c in per_case))
    return dict(pairs=pairs, per_case=per_case, primary_median_ttft_reduction=main_ttft,
                checks=checks, passed=all(checks.values()),
                decision="eligible_for_heldout_validation" if all(checks.values()) else "stop_chunk16_at_this_gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    result = dict(evidence_kind="paired_synthetic_workload_screen", formal_performance_result=False,
                  artifact_sha256=hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  summary=summarize(artifact),
                  limits=["Only two pairs per synthetic workload; no independent held-out adoption result.",
                          "Per-workload gates use the median of the two paired relative changes.",
                          "OS cache and system noise remain possible confounders."])
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["summary"]["checks"]))


if __name__ == "__main__":
    main()
