"""Validate evidence and apply the frozen N1 extension gates."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def token_hash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def summarize(data):
    if data["status"] != "completed" or len(data["runs"]) != 32 or len(data["prompts"]) != 8:
        raise ValueError("need the complete 32-run N1 evidence")
    if {(p["category"], p["length"]) for p in data["prompts"]} != {
        (category, length) for category in ("code", "zh", "math", "tool") for length in (1024, 16384)
    } or len({p["case"] for p in data["prompts"]}) != 8:
        raise ValueError("the fixed workload matrix is incomplete")
    pairs = []
    for prompt in data["prompts"]:
        for wave in (0, 1):
            rows = [r for r in data["runs"] if r["wave"] == wave and r["case"] == prompt["case"]]
            if len(rows) != 2 or {r["variant"] for r in rows} != {"control", "candidate"}:
                raise ValueError("missing or duplicated pair")
            control = next(r for r in rows if r["variant"] == "control")
            candidate = next(r for r in rows if r["variant"] == "candidate")
            a, b = control["metrics"], candidate["metrics"]
            for row in rows:
                m = row["metrics"]
                if (m["token_sha256"] != token_hash(m["generated_token_ids"])
                    or m["generated_tokens"] != len(m["generated_token_ids"])
                    or not 1 < m["generated_tokens"] <= data["max_tokens"]
                    or m["prompt_token_sha256"] != prompt["token_sha256"]
                    or m["prompt_tokens"] != prompt["length"]
                    or m["mtp_enabled"] or m["approximation_mode"] != "exact"
                    or m["prompt_cache_reused_tokens"]
                    or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]):
                    raise ValueError("invalid output, prompt or configuration evidence")
            record = candidate["record"]
            if (record["sorting_hint"] or record["status"] != "completed"
                or record["modified_calls"] != b["request_batched_expert_layers"] * len(b["prefill_attention_chunk_sizes"])
                or record["modified_calls"] <= 0
                or record["script_sha256"] != data["source"]["files_sha256"]["Scripts/research_qwen_sorted_experts.py"]
                or record["source_sha256"] != data["source"]["files_sha256"]["runtime/deepseek_v4_ssd/qwen4_exp.py"]):
                raise ValueError("candidate not exercised as registered")
            if any(a[k] != b[k] for k in ("generated_token_ids", "request_expert_bytes_read", "request_gather_qmm_calls")):
                raise ValueError("output or IO contract failed")
            pairs.append(dict(case=prompt["case"], length=prompt["length"], category=prompt["category"], wave=wave,
                ttft_reduction=1-b["time_to_first_token_seconds"]/a["time_to_first_token_seconds"],
                decode_change=b["decode_tokens_per_second"]/a["decode_tokens_per_second"]-1,
                p95_change=b["decode_latency_p95_seconds"]/a["decode_latency_p95_seconds"]-1,
                mlx_memory_change=b["peak_memory_bytes"]/a["peak_memory_bytes"]-1,
                rss_change=candidate["max_rss_bytes"]/control["max_rss_bytes"]-1,
                generated_tokens=a["generated_tokens"], token_sha256=a["token_sha256"],
                control_read_seconds=a["request_expert_read_seconds"],
                candidate_read_seconds=b["request_expert_read_seconds"],
                logical_expert_bytes=a["request_expert_bytes_read"]))
    keys = ("ttft_reduction", "decode_change", "p95_change", "mlx_memory_change", "rss_change")
    per_case = []
    for prompt in data["prompts"]:
        rows = [r for r in pairs if r["case"] == prompt["case"]]
        per_case.append(dict(case=prompt["case"], length=prompt["length"], category=prompt["category"],
            minimum_generated_tokens=min(r["generated_tokens"] for r in rows),
            repeat_output_stable=len({r["token_sha256"] for r in rows}) == 1,
            **{k: statistics.median(r[k] for r in rows) for k in keys}))
    ttft = {str(length): statistics.median(r["ttft_reduction"] for r in per_case if r["length"] == length)
            for length in (1024, 16384)}
    checks = dict(ttft=all(v >= .05 for v in ttft.values()),
        decode=all(r["decode_change"] >= -.05 for r in per_case),
        p95=all(r["p95_change"] <= .10 for r in per_case),
        memory=all(r["mlx_memory_change"] <= .05 and r["rss_change"] <= .05 for r in per_case),
        long_code=all(r["minimum_generated_tokens"] >= 512 for r in per_case if r["category"] == "code"))
    return dict(pairs=pairs, per_case=per_case, ttft_reduction_by_length=ttft, checks=checks,
        passed=all(checks.values()), decision="continue_to_cache_validation" if all(checks.values()) else "inspect_failed_gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    result = dict(formal_performance_result=False,
        results_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(),
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), summary=summarize(data))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]["checks"]))


if __name__ == "__main__":
    main()
